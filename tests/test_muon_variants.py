"""The two opt-in Muon arms: the NorMuon direction and the schedule-free wrapper.

Both are `off`/`muon` by default, so the FIRST gate of each is that the default
path is byte-identical to its history — and the reference values for that are
**captured literals** (sha256 of each parameter's bytes after 10 fixed-seed
steps, taken from the tree BEFORE the arms landed, `/tmp/opencode/capture_muon.py`
in the session that landed them), not a re-transcription. A re-transcription can
drift with the code; a hash cannot.

The rest of the file is the three things an arm owes (AGENTS §1.2/§1.4):

* a **hand-computed gold step** — the arithmetic written out in the test in
  float64, including the eps and the EMA, and compared to the port's own call
  with the DIRECTION injected (`_nor_apply` / `_nor_batched` / `_sf_one`). The
  direction (momentum → ColRow → NS) is shared with the control and gated in
  `tests/test_muon_fast.py`; what these arms change is the tail, so the tail is
  what is pinned here.
* a **transcribed reference** — an independent program (Appendix F's tail,
  `2605.23061v1`) over several steps, so a single-step coincidence cannot pass;
* **state, resume and refusal** gates — the new state keys round-trip, a resumed
  run continues identically, the arms are refused when they cannot run, and the
  trainer's SF seams (`X` at eval/checkpoint, `Y` for the step) are wired.

Sources, quoted at the call site: `zichongli5/NorMuon` `normuon.py:48-56` (the
Arm A tail, transcribed line by line) and `normuon.py:35,89,92-93` (its
defaults, its `[..., m, 1]` statistic and its apply); arXiv:2510.05491 Alg 1
lines 5-11 (the NorMuon paper the repo implements); `2605.23061v1` Alg 1 lines
3-18 + Appendix F (SF-NorMuon, Arm B, and its `√(mn)`/`√max(m,n)` disagreement —
the CODE says `mn`). **Arm A's composition is a MEASUREMENT, not a preference:**
composing ColRow makes the tail inert (`test_composing_colrow_makes_the_arm_inert_
which_is_why_it_replaces_it`) and against our ColRow control its magnitude is not
preserved (`test_the_normuon_arm_is_not_norm_matched_to_our_colrow_control`).
"""

from __future__ import annotations

import hashlib
import math

import pytest
import torch
from conftest import local_pin, tiny_run
from torch import nn

from dormouse.config import load_run
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train import optim, train
from dormouse.train.optim import (
    MUON_VARIANTS,
    MixOptimizer,
    Muon,
    colrow_normalize,
    nor_eta_hat,
    zeropower_via_ns,
)

# --- the pre-change reference, captured as BYTE HASHES ----------------------
#
# `/tmp/opencode/capture_muon.py`, run against the tree BEFORE either arm
# landed: 10 steps of `Muon(params, lr=1e-3)` (every default: variant `muon`,
# `colrow_3d="colrow"`, `state_dtype="fp32"`) on `_SHAPES` with per-step grads
# from `torch.Generator().manual_seed(7)`, `randn * 0.1`. Order = `_SHAPES`
# insertion order (p0 is the `(2, 3)` singleton, i.e. the `_step_one` branch).

_SHAPES = {(2, 3): 1, (64, 64): 2, (8, 128): 2}

_PRE_CHANGE_SHA = (
    "56fb301e85d3e8fed67c66f7fdaf43fb31e56c5ab324a9c6f70607a768944ccd",  # (2, 3) singleton
    "8d4ae51e90c5f777da8e09f2e6bcdf109ec59ebd68b59203b2ff112561935f7e",  # (64, 64)
    "aeff5e961608b40e769067e4dac52bf5af10114d3ce59d14f3187603cc48a968",  # (64, 64)
    "a043caef48fd8a793804502671e8a8baf412177cba44f25fa079acf98db168de",  # (8, 128)
    "2b4189cc2733ec5b1d0d33e1ee330d09d30555558d1cc5861af840511d1e8ec8",  # (8, 128)
)


def _shape_model(seed: int = 0) -> nn.Module:
    gen = torch.Generator().manual_seed(seed)
    mod = nn.Module()
    i = 0
    for shape, count in _SHAPES.items():
        for _ in range(count):
            mod.register_parameter(f"p{i}", nn.Parameter(torch.randn(*shape, generator=gen) * 0.05))
            i += 1
    return mod


def _sha(p: torch.Tensor) -> str:
    return hashlib.sha256(p.detach().numpy().tobytes()).hexdigest()


def _ten_steps(muon: Muon, model: nn.Module) -> None:
    """The captured run: 10 steps, per-step grads from the same generator."""
    gen = torch.Generator().manual_seed(7)
    for _ in range(10):
        for p in model.parameters():
            p.grad = torch.randn(*p.shape, generator=gen) * 0.1
        muon.batched_step_from_model(model)


# --- the hand references (float64, written out; they never call the port) ---


def _hand_colrow(o: torch.Tensor) -> torch.Tensor:
    """`colrow_normalize` for a 2-D matrix, in float64, eps spelled out.

    The port's default eps is `1e-7` and the two normalizations are columns
    (dim 0) then rows (dim 1) — `optim.colrow_normalize`'s docstring.
    """
    x = o.double() / (o.double().norm(dim=0, keepdim=True) + 1e-7)
    return x / (x.norm(dim=1, keepdim=True) + 1e-7)


def _hand_normuon(o: torch.Tensor, v_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """NorMuon's tail in float64: `v_1`, `P̂ = O ⊘ (√V + ε)` (β2 = 0.95, ε = 1e-8)."""
    row_ms = (o.double() * o.double()).mean(dim=-1)
    v = 0.95 * v_prev.double() + 0.05 * row_ms
    return o.double() / (v.sqrt() + 1e-8).unsqueeze(-1), v


def _hand_eta(lr: float, m: int, n: int, p_hat: torch.Tensor) -> float:
    """`η̂ = 0.2·η·√(mn)/‖P̂‖_F` (Alg 1 line 13 / Appendix F:77), float64."""
    return 0.2 * lr * math.sqrt(m * n) / float(p_hat.norm())


def _hand_normuon_repo(o: torch.Tensor, v_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The AUTHORS' repo tail in float64, written out from `normuon.py:48-56`.

    `:48` `vnorm = ‖O‖_F`, `:49` `v_mean = mean_cols(O⊙O)`, `:50`
    `v ← lerp(v, v_mean, 1−β2)`, `:51-52` `O ← O ⊘ (√v + 1e-10)`, `:53-54`
    `O ← O · ‖O‖_F/(‖Ô‖_F + 1e-10)`, `:56` `O ← O · max(1, m/n)^0.5`. Returns
    the final update and the new row statistic. **No ColRow**: the repo feeds its
    tail the bare polar and so does this port's arm (see the module docstring of
    `optim.normuon_repo_update_` for why composing is inert).
    """
    m, n = o.shape[-2], o.shape[-1]
    x = o.double()
    vnorm = x.norm(dim=(-1, -2), keepdim=True)
    v_mean = (x * x).mean(dim=-1)
    v = v_prev.double() + 0.05 * (v_mean - v_prev.double())  # `lerp_(., 1-beta2)`
    x = x * (1.0 / (v.sqrt() + 1e-10)).unsqueeze(-1)
    vnorm_new = x.norm(dim=(-1, -2), keepdim=True)
    x = x * (vnorm / (vnorm_new + 1e-10))
    return x * max(1, m / n) ** 0.5, v


# --- ARM A: the `normuon` direction (the authors' repo form) -----------------


def test_the_normuon_tail_is_the_reference_repo_form_hand_computed():
    """The gold step, from `normuon.py:48-56`, on a `[3, 2]` matrix, to 1e-6.

    A `[3, 2]` shape on purpose: `max(1, m/n)^0.5 = sqrt(1.5) != 1`, so the one
    factor that a square or wide shape would hide is exercised. The direction is
    INJECTED (`_nor_apply(p, O, lr)`) because the arm's whole change is the tail
    (momentum and the NS polar are shared with the control and gated in
    `tests/test_muon_fast.py`), and the hand side is float64 arithmetic written
    out in this file — including the two `1e-10`s the repo spells at `:51` and
    `:54` and the `lerp_` at `:50`.
    """
    lr = 1e-3
    o = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # [3, 2]
    w0 = torch.full((3, 2), 0.25)

    # --- by hand, one factor at a time -----------------------------------
    update_hand, v1_hand = _hand_normuon_repo(o, torch.zeros(3))
    row_ms = (o.double() * o.double()).mean(dim=1)
    assert torch.allclose(v1_hand, 0.05 * row_ms), "v_1 is not 0.05*mean_cols(O*O)"
    # `:54`'s whole point: the pre-normalization norm is restored, so `:56`'s
    # factor is the ONLY thing that changes the norm.
    assert float(update_hand.norm()) == pytest.approx(
        float(o.double().norm()) * math.sqrt(1.5), rel=1e-9
    )
    w_hand = w0.double() - lr * update_hand

    # --- the port --------------------------------------------------------
    p = nn.Parameter(w0.clone())
    muon = Muon([p], lr=lr, variant="normuon")
    muon._nor_apply(p, o.clone(), lr)
    assert torch.allclose(p.detach().double(), w_hand, atol=1e-6), (p.detach(), w_hand)
    assert torch.allclose(muon.state[p]["v"].double(), v1_hand, atol=1e-7), (
        "the row second moment is not the hand-computed v_1"
    )
    assert muon.state[p]["v"].dtype == torch.float32, "v is fp32"

    # --- the batched twin takes the same step (N = 1 stack) --------------
    p2 = nn.Parameter(w0.clone())
    muon2 = Muon([p2], lr=lr, variant="normuon")
    muon2._build_groups()
    muon2._nor_batched(muon2._groups[0], 0, 1, [p2], o.unsqueeze(0).clone(), lr)
    assert torch.allclose(p2.detach().double(), w_hand, atol=1e-6)
    assert torch.allclose(muon2.state[p2]["v"].double(), v1_hand, atol=1e-7), (
        "the batched tail and the per-param tail must share one definition"
    )


def test_the_normuon_ema_recursion_is_the_transcribed_reference():
    """Three steps against a hand-written float64 loop — the EMA, not one step.

    A single step cannot distinguish "v accumulates" from "v is the last row
    mean": the reference here carries `v` across steps the way `normuon.py:50`'s
    `second_momentum.lerp_(v_mean, 1 - beta2)` does.
    """
    lr = 2e-3
    dirs = [
        torch.tensor([[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]]),
        torch.tensor([[2.0, 0.0, -1.5], [-0.5, 1.0, 2.0]]),
        torch.tensor([[0.3, 0.7, -0.9], [1.5, -2.5, 0.1]]),
    ]
    w0 = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])

    # --- by hand: three steps of the tail, v carried ---------------------
    v = torch.zeros(2)
    w = w0.double()
    for o in dirs:
        update, v = _hand_normuon_repo(o, v)
        w = w - lr * update

    # --- the port --------------------------------------------------------
    p = nn.Parameter(w0.clone())
    muon = Muon([p], lr=lr, variant="normuon")
    for o in dirs:
        muon._nor_apply(p, o.clone(), lr)
    assert torch.allclose(p.detach().double(), w, atol=1e-6), (p.detach(), w)
    assert torch.allclose(muon.state[p]["v"].double(), v, atol=1e-7)
    assert not torch.allclose(p.detach(), w0), "the arm moved nothing — it is inert"


@pytest.mark.parametrize(
    "shape,cos_floor",
    [((64, 5461), 0.9999), ((64, 64), 0.999), ((2048, 64), 0.99), ((3, 2), 0.8)],
)
def test_the_normuon_magnitude_is_the_repo_muons_magnitude(shape, cos_floor):
    """**The repo's design intent, gated**: same η, same `max(1, m/n)^0.5`, same norm.

    `normuon.py:54`'s own comment says the rescale "keep[s] the update norm the
    same as pre-normalization", and `:56` applies the same `max(1, m/n)^0.5` the
    repo's Muon applies — so the arm's `‖ΔW‖_F` equals the REPO Muon's
    (`-η·max(1, m/n)^0.5·polar`, no ColRow) to 1e-6. That is the property that
    makes the repo's NorMuon-vs-Muon a direction-only comparison, and it holds on
    every shape measured (ratio 0.99999975 … 1.00000046).

    The `cos_floor` column is the honest second half: **the direction change is
    SMALL where the polar's rows are already near-unit-norm** — 0.99998 on the
    wide `[64, 5461]` factor, because a `m <= n` polar factor has orthonormal
    rows, which is exactly the statistic NorMuon re-scales — and only clearly
    visible on the tall/small shapes (0.9963 at `[2048, 64]`, 0.8735 at
    `[3, 2]`). So NorMuon's row half is close to a no-op over the bare polar for
    our dominant wide shape; what the arm really moves is our ColRow (next test).
    """
    torch.manual_seed(0)
    lr = 1e-3
    m, n = shape
    o = zeropower_via_ns(
        torch.randn(*shape, generator=torch.Generator().manual_seed(0)), 5, "classic"
    )
    scale = max(1.0, m / n) ** 0.5
    d_repo_muon = (-lr * scale * o).double()
    p = nn.Parameter(torch.zeros(*shape))
    arm = Muon([p], lr=lr, variant="normuon")
    arm._nor_apply(p, o.clone(), lr)
    d_arm = p.detach().double()

    assert float(d_arm.norm()) == pytest.approx(float(d_repo_muon.norm()), rel=1e-6), (
        f"{shape}: the repo's own magnitude invariant broke"
    )
    cos = float((d_arm.flatten() @ d_repo_muon.flatten()) / (d_arm.norm() * d_repo_muon.norm()))
    assert cos < 1.0, f"{shape}: the direction did not move at all ({cos})"
    assert cos > cos_floor, f"{shape}: cos={cos} — re-measure the characterisation"


def test_the_normuon_arm_is_not_norm_matched_to_our_colrow_control():
    """The mismatch the lane report carries, pinned instead of forced.

    The arm's magnitude invariant is against the REPO's Muon. Against OUR control
    (Muon + ColRow) it does not hold, because ColRow's row normalization gives
    `‖ColRow(O)‖_F = √m` while the Jordan quintic's polar gives
    `‖O‖_F ≈ 0.7·√min(m,n)`, and the tall factor carries `√(n/m)` on top. The
    ratios below are the measurement: **the arm is a REPLACE of our ColRow tail,
    not a direction-only knob**, and the A/B must be read that way.
    """
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    lr = 1e-3
    ratios, cosines = {}, {}
    for shape in [(64, 5461), (64, 64), (2048, 64)]:
        m, n = shape
        o = zeropower_via_ns(torch.randn(*shape, generator=gen), 5, "classic")
        scale = max(1.0, m / n) ** 0.5
        d_ctl = (-lr * scale * colrow_normalize(o)).double()
        p = nn.Parameter(torch.zeros(*shape))
        arm = Muon([p], lr=lr, variant="normuon")
        arm._nor_apply(p, o.clone(), lr)
        d_arm = p.detach().double()
        ratios[shape] = float(d_arm.norm() / d_ctl.norm())
        cosines[shape] = float((d_arm.flatten() @ d_ctl.flatten()) / (d_arm.norm() * d_ctl.norm()))
    assert ratios[(64, 5461)] == pytest.approx(0.706647, abs=2e-3)
    assert ratios[(64, 64)] == pytest.approx(0.872287, abs=2e-3)
    assert ratios[(2048, 64)] == pytest.approx(0.129549, abs=2e-3)
    assert cosines[(64, 5461)] == pytest.approx(0.996106, abs=2e-3)
    assert cosines[(64, 64)] == pytest.approx(0.999397, abs=2e-3)
    assert cosines[(2048, 64)] == pytest.approx(0.999894, abs=2e-3)


def test_composing_colrow_makes_the_arm_inert_which_is_why_it_replaces_it():
    """The measurement that decided the arm's shape, pinned so it cannot rot.

    NorMuon's row scaling is ColRow's row half (a per-row scaling by the row
    norm), and `colrow_normalize` makes every row unit-norm — so feeding the tail
    a ColRow'd matrix makes `mean_cols(O⊙O) = 1/n` for EVERY row (measured spread
    `1.8e-7`, i.e. the eps) and the arm reproduces the control's update. Composing
    was the first plan; this is the number that killed it. The arm therefore
    REPLACES the ColRow tail (`_nor_apply` feeds it the bare polar).
    """
    torch.manual_seed(0)
    o = zeropower_via_ns(
        torch.randn(64, 5461, generator=torch.Generator().manual_seed(0)), 5, "classic"
    )
    c = colrow_normalize(o)
    row_ms = (c * c).mean(dim=1)
    assert float((row_ms.max() - row_ms.min()) / row_ms.mean()) < 1e-6, (
        "a ColRow'd matrix must have a (near-)constant row statistic"
    )

    p = nn.Parameter(torch.zeros(64, 5461))
    composed = Muon([p], lr=1e-3, variant="normuon")
    composed._nor_apply(p, c.clone(), 1e-3)
    d_comp = p.detach().double()
    d_ctl = (-1e-3 * colrow_normalize(o)).double()
    cos = float((d_comp.flatten() @ d_ctl.flatten()) / (d_comp.norm() * d_ctl.norm()))
    assert cos == pytest.approx(1.0, abs=1e-6), f"composed is not inert: cos={cos}"
    assert float(d_comp.norm()) == pytest.approx(float(d_ctl.norm()), rel=1e-6)


def test_the_eta_hat_scale_is_the_sf_arms_and_the_paper_reads_sqrt_mn():
    """`η̂ = 0.2·η·√(mn)/‖P̂‖_F` belongs to the `sf` arm only.

    `2605.23061v1` §2.2's prose and Lemma 3.1 write `√max(m,n)`; Algorithm 1 line
    13 and Appendix F's code write `√(mn)` (41% apart at `[2, 3]`). The code is
    what this port runs — and the `normuon` arm does NOT use this function at all
    (its magnitude comes from the repo's rescale), which the last assertion pins.
    """
    lr, o = 3e-3, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # the SF arm's P̂ on the `muon` variant is ColRow(polar) (the SF paper's
    # `P_t = polar(M_t)` then our normalization); on the `normuon` variant it is
    # the Alg-1 row form. This test pins the `muon` one.
    p_hat = _hand_colrow(o)
    sqrt_mn = _hand_eta(lr, 2, 3, p_hat)
    sqrt_max = 0.2 * lr * math.sqrt(3) / float(p_hat.norm())
    assert sqrt_mn / sqrt_max == pytest.approx(math.sqrt(2), rel=1e-9)  # 41%

    got = float(nor_eta_hat(lr, (2, 3), p_hat.float()))
    assert got == pytest.approx(sqrt_mn, rel=1e-6)
    assert got != pytest.approx(sqrt_max, rel=1e-6), "the sqrt(max(m,n)) reading ran"

    # the SF arm's step is η̂·P̂ (the scale is inside `_sf_direction`)
    p = nn.Parameter(torch.zeros(2, 3))
    muon = Muon([p], lr=lr, sf=True)
    muon._sf_one(p, o.clone(), lr)
    assert torch.allclose(muon.state[p]["z"].double(), -sqrt_mn * p_hat, atol=1e-6), (
        "the sf arm's step is not η̂·ColRow(polar)"
    )

    # ... and the `normuon` arm is NOT that update: it is the repo's rescale
    q = nn.Parameter(torch.zeros(2, 3))
    nor = Muon([q], lr=lr, variant="normuon")
    nor._nor_apply(q, o.clone(), lr)
    update_hand, _ = _hand_normuon_repo(o, torch.zeros(2))
    assert torch.allclose(q.detach().double(), -lr * update_hand, atol=1e-6)
    assert not torch.allclose(q.detach().double(), -sqrt_mn * p_hat, atol=1e-5), (
        "the normuon arm is still the paper's η̂ form"
    )


@local_pin
def test_the_default_path_is_bit_identical_to_the_pre_change_tree():
    """`muon` is the default and takes the step the captured tree took.

    The literals are byte hashes of the parameters after the 10-step run in
    `/tmp/opencode/capture_muon.py`, captured BEFORE either arm landed. A hash
    cannot drift with a re-transcription, which is why it is the gate here and
    `tests/test_muon_fast.py`'s transcribed `old_batched_step` is the other one.
    """
    assert MUON_VARIANTS == ("muon", "normuon")
    assert load_run("small").train.muon_variant == "muon"
    assert load_run("small").train.muon_sf is False

    torch.manual_seed(0)
    model = _shape_model()
    muon = Muon(model.parameters(), lr=1e-3)
    assert muon.variant == "muon" and muon.sf is False and muon.wd == 0.0
    _ten_steps(muon, model)
    got = tuple(_sha(p) for p in model.parameters())
    assert got == _PRE_CHANGE_SHA, "the default path moved"

    # ... and the arms are not silently on: their state keys do not exist
    assert all("v" not in muon.state[p] for p in muon.param_groups[0]["params"])
    assert all("z" not in muon.state[p] for p in muon.param_groups[0]["params"])
    assert all(g.v is None and g.x is None and g.z is None for g in muon._groups)


def test_the_normuon_arm_is_a_different_step_on_the_same_fixture():
    """The arm ran, and it is not the control's step (the `inert` failure class)."""
    a, b = _shape_model(), _shape_model()
    b.load_state_dict(a.state_dict())
    gen = torch.Generator().manual_seed(11)
    for p in a.parameters():
        p.grad = torch.randn(*p.shape, generator=gen) * 0.1
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        q.grad = p.grad.clone()
    control = Muon(a.parameters(), lr=1e-3)
    arm = Muon(b.parameters(), lr=1e-3, variant="normuon")
    control.batched_step_from_model(a)
    arm.batched_step_from_model(b)
    assert arm._groups[0].v is not None, "the arm built no row statistic"
    assert not all(
        torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters(), strict=True)
    ), "the normuon arm took the control's step — it is inert"


def test_the_normuon_state_round_trips_and_the_next_step_is_identical():
    """`v` travels in the per-parameter state (torch's own format) and a resumed
    optimizer's NEXT step is the uninterrupted one's."""
    shapes = {(64, 5461): 3, (2, 3): 1}
    a, b = _twin(shapes)

    oa = Muon(a.parameters(), lr=1e-3, variant="normuon")
    ob = Muon(b.parameters(), lr=1e-3, variant="normuon")
    oa.batched_step_from_model(a)
    ob.batched_step_from_model(b)
    sd = oa.state_dict()
    assert "v" in sd["state"][0] and sd["state"][0]["v"].shape == (64,)

    fresh = _shape_model_from(shapes)
    resumed = Muon(fresh.parameters(), lr=1e-3, variant="normuon")
    resumed.load_state_dict(sd)
    for i, p in enumerate(resumed.param_groups[0]["params"]):
        assert torch.equal(resumed.state[p]["v"], sd["state"][i]["v"])
    # `a` takes step 2; a twin that starts from the checkpoint takes the same one
    c, d = _twin(shapes)
    d.load_state_dict(a.state_dict())  # d starts where `a` is after step 1
    for p, q in zip(a.parameters(), c.parameters(), strict=True):
        p.grad = q.grad.clone()
    resumed = Muon(d.parameters(), lr=1e-3, variant="normuon")
    resumed.load_state_dict(sd)
    oa.batched_step_from_model(a)
    resumed.batched_step_from_model(d)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), d.parameters(), strict=True)), (
        "the resumed normuon optimizer's next step diverged"
    )


def _shape_model_from(shapes: dict[tuple[int, int], int], seed: int = 0) -> nn.Module:
    gen = torch.Generator().manual_seed(seed)
    mod = nn.Module()
    i = 0
    for shape, count in shapes.items():
        for _ in range(count):
            mod.register_parameter(f"p{i}", nn.Parameter(torch.randn(*shape, generator=gen) * 0.05))
            i += 1
    return mod


def _twin(shapes: dict[tuple[int, int], int], seed: int = 1) -> tuple[nn.Module, nn.Module]:
    """Two models with identical weights AND identical gradients."""
    a, b = _shape_model_from(shapes), _shape_model_from(shapes)
    b.load_state_dict(a.state_dict())
    gen = torch.Generator().manual_seed(seed)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        pa.grad = torch.randn(*pa.shape, generator=gen) * 0.1
        pb.grad = pa.grad.clone()
    return a, b


@pytest.mark.parametrize("state_dtype", optim.OPT_DTYPES)
def test_the_normuon_arm_runs_on_both_state_rungs(state_dtype):
    """`bf16` momentum + fp32 `v`: finite, and the row statistic stays fp32."""
    shapes = {(64, 5461): 2, (8, 128): 2}
    m = _shape_model_from(shapes)
    gen = torch.Generator().manual_seed(5)
    for p in m.parameters():
        p.grad = torch.randn(*p.shape, generator=gen) * 0.1
    muon = Muon(m.parameters(), lr=1e-3, variant="normuon", state_dtype=state_dtype)
    muon.batched_step_from_model(m)
    want = torch.bfloat16 if state_dtype == "bf16" else torch.float32
    assert all(g.mom.dtype == want for g in muon._groups)
    assert all(g.v.dtype == torch.float32 for g in muon._groups)
    assert all(bool(torch.isfinite(g.v).all()) for g in muon._groups)
    assert all(bool(torch.isfinite(p).all()) for p in m.parameters())


def test_a_bad_variant_is_loud_everywhere_it_can_be_set():
    model = DormouseModel(load_run("small").model)
    for build in (
        lambda: Muon([nn.Parameter(torch.zeros(8, 8))], variant="NorMuon"),
        lambda: MixOptimizer(model, lr=1e-3, wd=0.0, muon_variant="nor_muon"),
        lambda: load_run("small", train_overrides={"muon_variant": "normuon2"}),
    ):
        with pytest.raises(LoudError) as e:
            build()
        assert "muon_variant=" in str(e.value) and "normuon" in str(e.value)


# --- ARM B: the schedule-free wrapper ---------------------------------------


def test_the_sf_gold_step_on_a_one_matrix_problem_is_hand_computable():
    """10 steps of the SF bookkeeping, by hand, against the port's `_sf_one`.

    The direction is INJECTED (`_sf_one(p, O, lr)`) for the same reason as the
    NorMuon gold: the tail is what the arm adds. The hand side runs Alg 1 lines
    13-18 in float64 on `X_1 = Z_1 = Y_1 = 0`:

        s_t = s_{t-1} + η_t² ; c = η_t²/s_t
        Z_{t+1} = Z_t − η̂_t·P̂_t        (λ = 0: the Muon group carries no wd)
        X_{t+1} = (1−c)·X_t + c·Z_{t+1}
        Y_{t+1} = (1−β1)·Z_{t+1} + β1·X_{t+1}      (β1 = 0.9, Appendix F:23)

    and the first step is pinned as a LITERAL relation: `c_1 = 1`, so `X_1 = Z_1`
    and `Y_1 = Z_1 = −η̂_1·P̂_1`.
    """
    lr = 1e-2
    o = torch.tensor([[3.0, 4.0]])  # what NS+ColRow would hand the tail
    p = nn.Parameter(torch.zeros(1, 2))
    muon = Muon([p], lr=lr, sf=True)
    assert muon.sf_beta == 0.9 and muon.wd == 0.0

    p_hat, eta = _hand_colrow(o), _hand_eta(lr, 1, 2, _hand_colrow(o))
    z = x = torch.zeros(1, 2, dtype=torch.float64)
    s = 0.0
    for t in range(1, 11):
        s = s + lr * lr
        c = lr * lr / s
        z = z - eta * p_hat
        x = (1.0 - c) * x + c * z
        muon._sf_one(p, o.clone(), lr)
        if t == 1:
            assert c == 1.0
            assert torch.allclose(z, x), "c_1 = 1 must make X_1 = Z_1"
        assert torch.allclose(p.detach().double(), (1.0 - 0.9) * z + 0.9 * x, atol=1e-6), (
            f"step {t}: Y is not (1-β1)Z + β1 X"
        )
        assert float(muon.state[p]["sf_s"]) == pytest.approx(s, rel=1e-12)
    # the accumulator is the sum of the squared rates, and it is a host float
    assert isinstance(muon.state[p]["sf_s"], float)
    assert float(muon.state[p]["sf_s"]) == pytest.approx(10 * lr * lr, rel=1e-12)
    assert not torch.allclose(p.detach(), torch.zeros(1, 2)), "the arm moved nothing"


def test_the_sf_batched_twin_takes_the_same_iterates_as_the_per_param_tail():
    """`_sf_one` and `_sf_batched` are one definition (N = 1 stack), 2 steps.

    The residual is the 2-D vs 3-D norm reduction order, not the arithmetic —
    the same tolerance `tests/test_muon_fast.py` uses for the batched ColRow.
    """
    o = torch.tensor([[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]])
    w0 = torch.tensor([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]])
    a, b = nn.Parameter(w0.clone()), nn.Parameter(w0.clone())
    per = Muon([a], lr=1e-2, sf=True)
    bat = Muon([b], lr=1e-2, sf=True)
    bat._build_groups()
    # c_1 = 1, c_2 = 1/2 at a constant lr (`_sf_ckp`; the batched path takes it
    # as an argument because `batched_step_from_model` computes it once per group)
    for c in (1.0, 0.5):
        per._sf_one(a, o.clone(), 1e-2)
        bat._sf_batched(bat._groups[0], 0, 1, [b], o.unsqueeze(0).clone(), 1e-2, c)
    assert torch.allclose(a.detach(), b.detach(), atol=1e-6), (a.detach(), b.detach())
    assert float(per.state[a]["sf_s"]) == pytest.approx(2e-4)
    assert "sf_s" not in bat.state[b], (
        "the batched tail does not own `s`: `batched_step_from_model` writes it once "
        "per group (gated by the round-trip test)"
    )


def test_sf_switch_returns_x_bit_exactly_and_train_restores_y():
    """The eval/train seam: `eval` is a copy of X, `train` is `(1−β1)Z + β1 X`.

    Both writes are ABSOLUTE functions of the optimizer's own state, so they are
    idempotent — the property the reference gets from its `train_mode` flag and
    this port gets from keeping X explicit.
    """
    p = nn.Parameter(torch.full((2, 3), 0.25))
    muon = Muon([p], lr=1e-2, sf=True)
    # TWO steps: after one, c_1 = 1 makes X_1 = Z_1 and the two iterates coincide
    muon._sf_one(p, torch.ones(2, 3), 1e-2)
    muon._sf_one(p, torch.ones(2, 3), 1e-2)
    x = muon.state[p]["x"].clone()
    z = muon.state[p]["z"].clone()
    y = (1.0 - 0.9) * z + 0.9 * x
    assert not torch.equal(x, y), "the two iterates must differ for this to bite"

    muon.sf_switch("eval")
    assert torch.equal(p.detach(), x), "eval must be X bit-exactly"
    muon.sf_switch("eval")  # idempotent
    assert torch.equal(p.detach(), x)
    muon.sf_switch("train")
    assert torch.equal(p.detach(), y), "train must restore Y"
    muon.sf_switch("train")  # idempotent
    assert torch.equal(p.detach(), y)


def test_sf_state_round_trips_and_a_resumed_optimizer_continues_identically():
    """`x`, `z` and `s` travel; the resumed next step equals the uninterrupted one."""
    shapes = {(64, 5461): 3, (2, 3): 1}
    a, b = _twin(shapes)

    oa = Muon(a.parameters(), lr=1e-2, sf=True)
    ob = Muon(b.parameters(), lr=1e-2, sf=True)
    oa.batched_step_from_model(a)
    ob.batched_step_from_model(b)
    sd = oa.state_dict()
    for key in ("x", "z", "sf_s"):
        assert key in sd["state"][0], f"{key} must travel with the checkpoint"

    fresh = _shape_model_from(shapes)
    resumed = Muon(fresh.parameters(), lr=1e-2, sf=True)
    resumed.load_state_dict(sd)
    for i, p in enumerate(resumed.param_groups[0]["params"]):
        assert torch.equal(resumed.state[p]["x"], sd["state"][i]["x"])
        assert torch.equal(resumed.state[p]["z"], sd["state"][i]["z"])
        assert float(resumed.state[p]["sf_s"]) == float(sd["state"][i]["sf_s"])

    c, d = _twin(shapes)
    d.load_state_dict(a.state_dict())
    for p, q in zip(a.parameters(), c.parameters(), strict=True):
        p.grad = q.grad.clone()
    resumed = Muon(d.parameters(), lr=1e-2, sf=True)
    resumed.load_state_dict(sd)
    oa.batched_step_from_model(a)
    resumed.batched_step_from_model(d)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), d.parameters(), strict=True)), (
        "the resumed SF optimizer's next step diverged"
    )


def test_sf_composes_with_the_normuon_direction():
    """`sf` + `normuon` is one arm, not two step sizes wearing one name.

    `_sf_direction` applies `η̂` to whichever normalization produced `P̂`, so the
    SF arm's step magnitude does not depend on `variant` — the composed arm must
    therefore build the row statistic AND move Z/X/Y.
    """
    p = nn.Parameter(torch.full((2, 3), 0.5))
    muon = Muon([p], lr=1e-2, sf=True, variant="normuon")
    muon._build_groups()
    muon._sf_batched(muon._groups[0], 0, 1, [p], torch.ones(1, 2, 3), 1e-2, 1.0)
    assert muon._groups[0].v is not None and float(muon._groups[0].v.abs().sum()) > 0.0
    assert not torch.equal(p.detach(), torch.full((2, 3), 0.5))
    # the step magnitude is the η̂ one on BOTH variants: ΔZ = −η̂·P̂ (Z_1 starts
    # at the param's own value, Alg 1's X_1 = Z_1 = Y_1)
    p_hat = _hand_normuon(torch.ones(2, 3), torch.zeros(2))[0]
    assert torch.allclose(
        muon.state[p]["z"].double() - 0.5,
        -_hand_eta(1e-2, 2, 3, p_hat) * p_hat,
        atol=1e-6,
    )


def test_a_bad_sf_combination_is_loud_and_names_the_escape():
    """`muon_sf` needs a Muon group: refused with `opt='adamw'`, at both doors."""
    model = DormouseModel(load_run("small").model)
    for build in (
        lambda: MixOptimizer(model, lr=1e-3, wd=0.0, opt="adamw", muon_sf=True),
        lambda: load_run("small", train_overrides={"opt": "adamw", "muon_sf": True}),
    ):
        with pytest.raises(LoudError) as e:
            build()
        msg = str(e.value)
        assert "muon_sf=true" in msg and "opt='adamw'" in msg and "Escape" in msg
    # a bare sf_switch on an optimizer without the arm is refused too
    plain = MixOptimizer(model, lr=1e-3, wd=0.0)
    with pytest.raises(LoudError, match="muon_sf is off"):
        plain.sf_switch("train")
    with pytest.raises(LoudError, match="sf=false"):
        Muon(model.parameters(), lr=1e-3).sf_switch("eval")


def test_the_trainer_forwards_both_arms_and_switches_x_for_every_eval(monkeypatch):
    """The wiring gate: the config reaches the optimizer AND the SF seams run.

    Without the kwargs a `--set muon_variant=normuon` would be INERT (the SILENT
    class), and without the switches an eval would score `Y` — the
    gradient-evaluation point — instead of the returned iterate `X`.
    """
    from dormouse.train import loop as train_loop

    seen: list[dict] = []
    switches: list[str] = []
    real = train_loop.MixOptimizer

    def spy(model, **kw):
        seen.append(kw)
        opt = real(model, **kw)
        real_switch = opt.sf_switch

        def recording(mode):
            switches.append(mode)
            return real_switch(mode)

        opt.sf_switch = recording
        return opt

    monkeypatch.setattr(train_loop, "MixOptimizer", spy)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        run = tiny_run(
            Path(tmp),
            steps=4,
            log_every=4,
            ckpt_every=4,
            eval=True,
            eval_every=2,
            eval_batches=2,
            muon_variant="normuon",
            muon_sf=True,
        )
        train(DormouseModel(run.model), run, log=lambda *_: None)

    assert seen and seen[0]["muon_variant"] == "normuon" and seen[0]["muon_sf"] is True
    # built -> train; then eval/train around each eval, and around the ckpt
    assert switches[0] == "train", switches
    assert switches.count("eval") == switches.count("train") - 1, switches
    assert "eval" in switches and switches[-1] == "train", switches


def test_the_sf_muon_lr_is_constant_after_warmup_while_the_control_decays():
    """The arm's whole point, measured on the LR the Muon group actually gets.

    The trainer hands the Muon group the WARMUP-ONLY schedule under `muon_sf`
    (no WSD decay) and the full WSD curve otherwise; the AdamW half keeps WSD in
    both. The LRs are recorded by wrapping the batched step, which is the one
    place that reads `param_groups["lr"]` per step.
    """
    import tempfile
    from pathlib import Path

    real = Muon.batched_step_from_model

    def run_one(**over) -> list[float]:
        recorded: list[float] = []

        def spy(self, model):
            recorded.append(self.param_groups[0]["lr"])
            return real(self, model)

        Muon.batched_step_from_model = spy
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run = tiny_run(Path(tmp), steps=6, log_every=6, warmup_tokens=2 * 64 * 2, **over)
                train(DormouseModel(run.model), run, log=lambda *_: None)
        finally:
            Muon.batched_step_from_model = real
        return recorded

    sf = run_one(muon_sf=True)
    control = run_one()
    assert len(sf) == 6 and len(control) == 6
    assert sf[:2] == control[:2], "the warmup ramp must be the same schedule"
    assert sf[2:] == [sf[-1]] * 4, f"the SF Muon LR must be constant after warmup: {sf}"
    assert control[-1] < control[2], f"the control must decay: {control}"


def test_a_nan_skipped_step_advances_none_of_z_x_or_s():
    """The firewall's contract with the arm: a skipped step is not a step.

    A masked loss must not advance `Z`, `X` or `s` — the SF average is
    step-indexed, so a skipped step that still moved them would average a
    trajectory the run did not take. Forced here by making the forward's loss
    non-finite on step 2 (`train/loop.py`'s `finite` branch).
    """
    import tempfile
    from pathlib import Path

    from dormouse.model.model import DormouseModel as DM

    real_forward = DM.forward
    state: dict[str, list] = {"steps": []}

    def forward(self, *a, **kw):
        out = real_forward(self, *a, **kw)
        state["steps"].append(len(state["steps"]) + 1)
        if len(state["steps"]) == 2:  # the SECOND forward returns a NaN loss
            return out[0], torch.tensor(float("nan")), out[2]
        return out

    real_batched = Muon.batched_step_from_model
    seen: list[float] = []

    def spy(self, model):
        seen.append(float(self.state[self.param_groups[0]["params"][0]].get("sf_s", 0.0)))
        return real_batched(self, model)

    DM.forward = forward
    Muon.batched_step_from_model = spy
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run = tiny_run(Path(tmp), steps=3, log_every=3, muon_sf=True)
            report = train(DormouseModel(run.model), run, log=lambda *_: None)
    finally:
        DM.forward = real_forward
        Muon.batched_step_from_model = real_batched
    assert report["nan_skipped"] == 1, report["nan_skipped"]
    assert seen == [0.0, 1e-4], f"the skipped step advanced s: {seen}"
