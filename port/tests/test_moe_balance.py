"""The load-balance controller arms and the routing-health counters.

Evidence (`docs/reviews/2026-11-04-routing-collapse-CONFIRMED.md`, `m400` at
step 149 999, real bytes, 8 192 assignments): **80 of 512 experts alive, top-10
holding 50.1% of all tokens**. The sign LFB controller could not repair it —
`b_e -= coef·sign(L_e − mean L)` gives a dead expert one `coef` per forward, its
own weights receive no gradient, and every resume restarted the integrator. The
arms here are `prop` (the RELATIVE deviation instead of its sign), `aux` (a
differentiable Switch/GShard term), the counter set that makes the collapse
visible in a log line, and the resume refusal that stops a missing bias from
silently restarting the controller.

Gates, each with a scratch-revert proof in the lane report:

a. every arm OFF is bit-identical to clean `main` — every digest below was
   captured on `main` at `34f1460` BEFORE the edit and is asserted here;
b. `prop` moves the bias by exactly `-gamma·(L−mean)/max(1,mean)`;
c. `aux` is the hand-computed `E·Σ_e f_e·P_e` (1e-6) and its gradient reaches
   the router and nothing else;
d. the health counters equal a hand-computed histogram;
e. a missing `lfb_bias` on resume raises with an `Escape:` — and the refusal
   follows the MODE, not `moe_lb_coef` (a `prop` run at `coef=0` still needs it,
   an `aux` run does not);
f. the routing path reads nothing back from the device.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import traceback

import pytest
import torch
from conftest import SEQ, tiny_cfg, tiny_run

from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model.experts import SparseMoE, switch_aux_loss
from dormouse.train import train
from dormouse.train.loop import ckpt_paths, load_model_half, load_model_state

#: Captured on clean `main` (`34f1460`) with `torch.manual_seed(0/1/2)` and the
#: `conftest` tiny fixture — the pin the OFF path must keep reproducing.
PIN_MOE_ZERO = {
    "bias": "374708fff7719dd5",
    "counts": "374708fff7719dd5",
    "down_grad": "3abc3d695a72bae8",
    "out": "5fdc21494b8dff49",
    "router_grad": "77d959fdde38916c",
    "counters": {
        "moe_dropped_assignments": 4,
        "moe_expert_calls": 4,
        "moe_naive_dispatch": 1,
        "moe_shared_forwards": 64,
        "moe_tokens": 64,
    },
}
PIN_MOE_SIGN = {
    "bias": "56d107b0c5a7cbed",
    "counts": "7f80e260dbc7bb4b",
    "down_grad": "3abc3d695a72bae8",
    "out": "5fdc21494b8dff49",
    "router_grad": "77d959fdde38916c",
    "counters": {
        "moe_dropped_assignments": 4,
        "moe_expert_calls": 4,
        "moe_lfb_updates": 1,
        "moe_naive_dispatch": 1,
        "moe_shared_forwards": 64,
        "moe_tokens": 64,
    },
}
PIN_MODEL = {
    "lfb_bias": "374708fff7719dd5",
    "logits": "50d547c8c517ae76",
    "router_grad": "07ba90f0c8f892be",
    "loss": 6.378957271575928,
    "counters": {
        "depth": 2,
        "engram_arms": 2,
        "engram_rows": 768,
        "forward": 1,
        "iter": 2,
        "kda_sdpa": 2,
        "moe_dropped_assignments": 11,
        "moe_expert_calls": 8,
        "moe_naive_dispatch": 2,
        "moe_shared_forwards": 256,
        "moe_tokens": 256,
        "rezero_write": 2,
    },
}


def dig(t: torch.Tensor) -> str:
    """The pin form: sha256 of the tensor's raw bytes, first 16 hex chars."""
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()[:16]


def moe_pin(coef: float) -> dict:
    """One forward + backward of a bare `SparseMoE`, digests of everything it moved."""
    torch.manual_seed(0)
    cfg = tiny_cfg(moe_lb_coef=coef)
    moe = SparseMoE(cfg)
    torch.manual_seed(1)
    x = torch.randn(SEQ, cfg.d_model)
    c = Counters()
    out = moe(x, c)
    out.square().mean().backward()
    c.drain()
    return {
        "out": dig(out),
        "bias": dig(moe.lfb_bias),
        "counts": dig(moe.lfb_counts),
        "router_grad": dig(moe.router.weight.grad),
        "down_grad": dig(moe.down.weight.grad),
        "counters": dict(sorted(c.items())),
    }


# --- a. every new arm OFF is bit-identical to clean main ----------------------


def test_off_path_is_bit_identical_to_clean_main():
    """`moe_lb_mode="sign"` + `moe_routing_health=False` IS the pre-edit program.

    Revert: flip `moe_lb_mode`'s default to `"prop"` (or `moe_routing_health`'s
    to `True`) and this test reads a different `bias`/`out`/`counters` digest.
    """
    assert moe_pin(0.0) == PIN_MOE_ZERO, "the zero-coef OFF path moved"
    assert moe_pin(0.05) == PIN_MOE_SIGN, "the sign controller's arithmetic moved"


def test_model_seam_off_is_bit_identical_to_clean_main():
    """The `begin_step()`/`aux` plumbing must not move a model that never asks.

    Revert: drop the `pure`/mode guard around the aux add (a `None` term is
    fine, but a stray term or a cleared accumulator under the wrong mode changes
    the loss digest) or run `begin_step` after the loop; this pin goes red.
    """
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = DormouseModel(cfg)
    torch.manual_seed(2)
    ids = torch.randint(0, cfg.vocab, (2, SEQ))
    tgts = torch.randint(0, cfg.vocab, (2, SEQ))
    c = Counters()
    logits, loss, _ = model(ids, targets=tgts, counters=c)
    loss.backward()
    c.drain()
    router = next(m for m in model.modules() if isinstance(m, SparseMoE))
    got = {
        "logits": dig(logits),
        "loss": float(loss.detach()),
        "router_grad": dig(router.router.weight.grad),
        "lfb_bias": dig(router.lfb_bias),
        "counters": dict(sorted(c.items())),
    }
    assert got == PIN_MODEL, "the model seam moved with every arm off"


# --- b. prop: the documented step, exactly ----------------------------------


def test_prop_moves_the_bias_by_the_relative_deviation():
    """`b -= gamma·(L−mean)/max(1,mean)` on THIS window, exactly, then clamps.

    Revert: put `prop` back on the CUMULATIVE counts and the dead expert's step
    shrinks as the run lengthens (the measured freeze); leave it on the sign
    branch and it moves `gamma` regardless of the gap. The exact compare reads
    either.
    """
    cfg = tiny_cfg(moe_lb_mode="prop", moe_lb_gamma=0.001, moe_lb_coef=0.0)
    moe = SparseMoE(cfg)
    assert moe.lb_active == 0.001, "prop's live step is gamma, not moe_lb_coef"
    counts = torch.tensor([6, 1, 1, 0])  # 8 assignments, expert 3 DEAD
    moe._lfb_step(torch.zeros((1, 1), dtype=torch.long), Counters(), counts=counts)

    load = counts.float()
    mean = load.mean()
    moved = 0.001 * (load - mean) / mean.clamp_min(1.0)
    torch.testing.assert_close(moe.lfb_bias, (0.0 - moved).clamp(-2.0, 2.0), rtol=0.0, atol=0.0)
    # The dead expert gains, the busiest loses, and the RELATIVE gaps set the
    # steps: deviations −1 (dead) and +2 (6 of 8 assignments), so the busiest
    # moves twice as far as the dead one.
    assert float(moe.lfb_bias[3]) > 0.0 > float(moe.lfb_bias[0])
    assert float(moe.lfb_bias[3]) == -0.5 * float(moe.lfb_bias[0])


def test_prop_is_clamped_inside_the_band():
    """`moe_lb_clamp` is a hard output bound: the ±200 drift cannot happen.

    Revert: drop the `clamp_` and five steps at `gamma=1.0` leave the bias at
    ±8 instead of exactly ±2 — which is the shape the saved `m400` file carries.
    """
    cfg = tiny_cfg(moe_lb_mode="prop", moe_lb_gamma=1.0, moe_lb_clamp=2.0)
    moe = SparseMoE(cfg)
    counts = torch.tensor([8, 0, 0, 0])  # every assignment to expert 0
    for _ in range(5):
        moe._lfb_step(torch.zeros((1, 1), dtype=torch.long), Counters(), counts=counts)
    torch.testing.assert_close(
        moe.lfb_bias, torch.tensor([-2.0, 2.0, 2.0, 2.0]), rtol=0.0, atol=0.0
    )


def test_the_sign_arm_still_moves_by_exactly_coef():
    """The default arm is untouched: one `coef`, sign only, no scale."""
    cfg = tiny_cfg(moe_lb_mode="sign", moe_lb_coef=0.05)
    moe = SparseMoE(cfg)
    moe.lfb_counts.copy_(torch.tensor([100.0, 100.0, 100.0, 0.0]))
    moe._lfb_step(
        torch.zeros((1, 1), dtype=torch.long),
        Counters(),
        counts=torch.zeros(cfg.n_experts, dtype=torch.long),
    )
    torch.testing.assert_close(
        moe.lfb_bias, torch.tensor([-0.05, -0.05, -0.05, 0.05]), rtol=0.0, atol=1e-7
    )


# --- c. aux: the Switch term and its gradient ---------------------------------


def test_aux_is_the_hand_computed_switch_term():
    """`E·Σ_e f_e·P_e` on a 4-token / 3-expert fixture, computed by hand.

    Revert: drop the `E·` normalization (or use the max share instead of `P_e`)
    and the hand number misses by a factor the 1e-6 tolerance cannot hide.
    """
    probs = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.4, 0.4, 0.2]])
    index = torch.tensor([[0, 1], [1, 2], [0, 0], [2, 2]])
    got = float(switch_aux_loss(probs, index, 3))
    # hand: 8 assignments, counts [3, 2, 3]; P is the column mean of `probs`.
    n, k = 4, 2
    counts = [3, 2, 3]
    rows = probs.tolist()
    p = [sum(r[e] for r in rows) / n for e in range(3)]
    expected = 3 * sum(c / (n * k) * p[e] for e, c in enumerate(counts))
    assert got == pytest.approx(expected, abs=1e-6), (got, expected)
    # And the paper's scale: a UNIFORM load and uniform gates read exactly 1
    # (the Switch value), whatever `k` is — `f` sums to 1 over the bank.
    uniform = torch.full((n, 3), 1 / 3)
    every = torch.tensor([[0, 1, 2]] * n)
    assert float(switch_aux_loss(uniform, every, 3)) == pytest.approx(1.0, abs=1e-6)
    assert float(
        switch_aux_loss(uniform, torch.zeros((n, 1), dtype=torch.long), 3)
    ) == pytest.approx(1.0, abs=1e-6)
    # One expert taking every assignment with a confident gate reads ~E.
    one_hot = torch.tensor([[1.0, 0.0, 0.0]] * n)
    assert float(switch_aux_loss(one_hot, torch.zeros((n, 2), dtype=torch.long), 3)) == (
        pytest.approx(3.0, abs=1e-6)
    )


def test_aux_gradient_reaches_the_router_and_nothing_else():
    """`f_e` is a hard count (a constant); `P_e` is the gate — router only.

    Revert: make `f` differentiable (any op on `index` that keeps a grad_fn) and
    the expert parameters receive gradient from the aux term; this test names
    every parameter that moved.
    """
    torch.manual_seed(0)
    cfg = tiny_cfg(moe_topk=2, moe_lb_mode="aux", moe_aux_coef=0.01)
    moe = SparseMoE(cfg).train()
    torch.manual_seed(1)
    x = torch.randn(SEQ, cfg.d_model)
    moe(x, Counters())
    term = moe.aux_loss()
    assert term is not None, "an aux forward must accumulate a term"
    term.backward()
    assert float(term.detach()) != 0.0
    assert moe.router.weight.grad is not None
    assert float(moe.router.weight.grad.abs().sum()) > 0.0, "the term's gradient missed the router"
    moved = [
        name
        for name, p in moe.named_parameters()
        if name != "router.weight" and p.grad is not None and float(p.grad.abs().sum()) > 0.0
    ]
    assert not moved, f"the aux term reached {moved[:4]}"


# --- d. the health counters vs a hand-computed histogram ----------------------


def test_health_counters_equal_a_hand_computed_histogram():
    """One forward, a scripted histogram, five counters, exact integers.

    Revert each leg: `moe_route_live` as `counts.numel()` (E, not the live
    count), `max` as `mean`, `top10` as `max`, or truncating instead of
    rounding the entropy, and the named number moves.
    """
    moe = SparseMoE(tiny_cfg(n_experts=16, moe_routing_health=True))
    hist = [5, 4, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  # 24 assignments
    c = Counters()
    moe._route_health(torch.tensor(hist), c)
    c.drain()
    total = sum(hist)
    assert c["moe_route_n"] == 1
    assert c["moe_route_live"] == 12
    assert c["moe_route_max_x1000"] == 5000 // 24  # 5/24, integer floor
    assert c["moe_route_top10_x1000"] == 22 * 1000 // 24  # the ten busiest
    hand = -sum(h / total * math.log(h / total) for h in hist if h)
    assert c["moe_route_ent_x1000"] == round(1000 * hand / math.log(16))

    # A UNIFORM bank reads the two fixed points: every expert alive, max = 1/E,
    # entropy 1000 (rounded — a truncation would read 999), top-10 = 10/E.
    c = Counters()
    moe._route_health(torch.tensor([2] * 16), c)
    c.drain()
    assert (
        c["moe_route_live"],
        c["moe_route_max_x1000"],
        c["moe_route_top10_x1000"],
        c["moe_route_ent_x1000"],
    ) == (16, 62, 625, 1000)


def test_a_real_forward_counts_its_own_histogram():
    """The wiring, not the helper: a forward's counters describe THAT forward."""
    torch.manual_seed(0)
    cfg = tiny_cfg(moe_lb_coef=0.0, moe_routing_health=True)
    moe = SparseMoE(cfg)
    torch.manual_seed(1)
    x = torch.randn(SEQ, cfg.d_model)
    c = Counters()
    moe(x, c)
    c.drain()
    hist = moe._last_counts
    assert hist is not None and int(hist.sum()) == SEQ * cfg.moe_topk
    assert c["moe_route_n"] == 1
    assert c["moe_route_live"] == int((hist > 0).sum())
    assert c["moe_route_max_x1000"] == int(hist.max()) * 1000 // int(hist.sum())


# --- e. the resume refusal follows the MODE ----------------------------------


def run_of(tmp_path, name: str, **model_over):
    """A tiny run whose MODEL config carries the controller arm under test."""
    run = tiny_run(tmp_path, steps=2, ckpt_every=2, log_every=1000, out_dir=str(tmp_path / name))
    return dataclasses.replace(run, model=dataclasses.replace(run.model, **model_over))


def train_then_drop_moe(run) -> None:
    torch.manual_seed(0)
    train(DormouseModel(run.model), run, log=lambda *_: None)
    weights, _snap = ckpt_paths(run.train)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    assert ck.get("moe"), "every routed run must write the bias section"
    ck.pop("moe")  # the pre-fix file: counts in `model`, no bias anywhere
    torch.save(ck, weights)


@pytest.mark.parametrize(
    "arm",
    [
        dict(moe_lb_mode="sign", moe_lb_coef=0.01),
        # The case the OLD `max(lfb_coef)` check missed: `prop` at coef=0 has a
        # LIVE controller (gamma=0.05) and still needs its bias back.
        dict(moe_lb_mode="prop", moe_lb_coef=0.0, moe_lb_gamma=0.05),
    ],
)
def test_a_missing_bias_refuses_a_live_controller(tmp_path, arm):
    """Revert: gate on `moe_lb_coef` again and the `prop` arm resumes silently."""
    run = run_of(tmp_path, "old", **arm)
    train_then_drop_moe(run)
    longer = run_of(tmp_path, "old", **arm)
    longer = dataclasses.replace(longer, train=dataclasses.replace(longer.train, steps=4))
    model = DormouseModel(longer.model)
    with pytest.raises(LoudError) as e:
        train(model, longer, resume=True, log=lambda *_: None)
    msg = str(e.value)
    assert "lfb_counts" in msg and "lfb_bias" in msg and "Escape:" in msg


def test_an_aux_run_needs_no_bias(tmp_path):
    """`aux` has NO bias controller, so a file written before the section resumes.

    Revert: make `lb_active` return `moe_lb_coef` under `aux` and this resume
    refuses a file it is about to use correctly (the controller is a loss term).
    """
    arm = dict(moe_lb_mode="aux", moe_lb_coef=0.01, moe_aux_coef=0.01)
    run = run_of(tmp_path, "aux", **arm)
    train_then_drop_moe(run)
    longer = run_of(tmp_path, "aux", **arm)
    longer = dataclasses.replace(longer, train=dataclasses.replace(longer.train, steps=4))
    report = train(DormouseModel(longer.model), longer, resume=True, log=lambda *_: None)
    assert report["opt_steps"] == 4


# --- the divergence the arm exists for ---------------------------------------


def _pinned_skew(mode: str, *, steps: int, gamma: float = 0.001, coef: float = 0.05):
    """`steps` of a SKEWED SCRIPTED routing: experts 0/1 take every assignment.

    The script is the point: a pinned routing cannot answer the controller, which
    is the `m400` situation in miniature (9 live experts; the bias is added to
    the logits but the order no longer changes). Returns the model and the
    largest |bias| seen at any step.
    """
    torch.manual_seed(0)
    cfg = tiny_cfg(
        n_experts=16,
        moe_topk=2,
        moe_lb_mode=mode,
        moe_lb_coef=coef,
        moe_lb_gamma=gamma,
        moe_lb_clamp=2.0,
    )
    moe = SparseMoE(cfg).train()
    pinned = torch.tensor([[0, 1]] * 64)
    worst = 0.0
    for _ in range(steps):
        moe._lfb_step(pinned, Counters())
        worst = max(worst, float(moe.lfb_bias.abs().max()))
    return moe, worst


def test_the_sign_arm_winds_past_the_clamp_and_prop_does_not():
    """THE BUG, reproduced: the same pinned skew, the two controllers.

    `sign` is a cumulative bang-bang integrator: with the routing pinned it steps
    by `coef` every forward and never gets an answer back, so 3 000 forwards at
    `coef=0.05` reach ±150 — the order of the saved `m400` bias (`[-218, +72]`),
    which is 40x the router's own logit scale and is why 9 experts run. `prop`
    reads THIS window's share and clamps, so the same 3 000 forwards end at
    exactly ±2.0 and never exceed it — and the released routing recovers the
    live bank.

    Revert: give `prop` the cumulative load or drop `clamp_` and every number
    below moves; that is what the file on disk carries.
    """
    sign, sign_worst = _pinned_skew("sign", steps=3000)
    prop, prop_worst = _pinned_skew("prop", steps=3000)

    assert sign_worst == pytest.approx(150.0, abs=0.01), "the legacy wind-up moved"
    assert sign_worst > 10 * 2.0, "the sign arm must leave the clamp's band"
    assert prop_worst <= 2.0, f"prop left the clamp: {prop_worst}"
    assert prop_worst == pytest.approx(2.0, abs=1e-6), "prop must USE its band, not crawl"
    # The directions are right in both arms: the pinned experts give up bias,
    # the fifteen untouched ones gain it.
    assert float(prop.lfb_bias[0]) == pytest.approx(-2.0, abs=1e-6)
    assert float(prop.lfb_bias[2]) == pytest.approx(2.0, abs=1e-6)
    assert float(sign.lfb_bias[0]) == pytest.approx(-150.0, abs=0.01)

    # RECOVERY: release the skew (scores favour experts 0/1 by 3 logits) and let
    # the routing answer. 40 steps of `prop` restore the live bank; the sign
    # arm's ±150 is still 150 two steps later (it has no answer to give).
    torch.manual_seed(1)
    base = torch.randn(64, 16)
    base[:, :2] += 3.0
    live = []
    for _ in range(40):
        index = (base + prop.lfb_bias).topk(2, dim=-1).indices
        live.append(len(set(index.flatten().tolist())))
        prop._lfb_step(index, Counters())
    assert live[0] >= 0.5 * 16, f"the released routing started at {live[0]}/16"
    assert live[-1] >= 0.8 * 16, f"prop did not recover the bank: {live[-1]}/16"
    assert max(live) >= 12, f"prop never spread the routing: {live}"
    sign._lfb_step(torch.tensor([[0, 1]] * 64), Counters())
    assert float(sign.lfb_bias.abs().max()) > 150.0, "the sign arm stopped winding"


def test_a_diverged_bias_refuses_a_training_resume(tmp_path):
    """`max|bias| > 10` — the `m400` shape — is not resumable, and says why.

    The bias is added to the logits unconditionally, so a file past the bound
    routes 9 of 512 experts whatever the optimizer does next. Revert: drop the
    `loud(msg)` for `refuse=True` and this resume succeeds silently on the
    collapsed router.
    """
    run = run_of(tmp_path, "div", moe_lb_mode="sign", moe_lb_coef=0.01)
    torch.manual_seed(0)
    train(DormouseModel(run.model), run, log=lambda *_: None)
    weights, _snap = ckpt_paths(run.train)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    ck["moe"] = {name: torch.full_like(b, -218.27) for name, b in ck["moe"].items()}
    torch.save(ck, weights)

    longer = run_of(tmp_path, "div", moe_lb_mode="sign", moe_lb_coef=0.01)
    longer = dataclasses.replace(longer, train=dataclasses.replace(longer.train, steps=4))
    with pytest.raises(LoudError) as e:
        train(DormouseModel(longer.model), longer, resume=True, log=lambda *_: None)
    msg = str(e.value)
    assert "lfb_bias" in msg and "DIVERGED" in msg and "Escape:" in msg


def test_a_reader_serves_a_diverged_bias_and_says_so(tmp_path, capsys):
    """`eval`/`decode`/`serve` must still be able to INSPECT the evidence.

    Same file, reader path: one LOUD line, the file served as written, no raise.
    Revert: make `report_lfb_range` raise for readers too and a collapsed
    checkpoint becomes uninspectable.
    """
    run = run_of(tmp_path, "read", moe_lb_mode="sign", moe_lb_coef=0.01)
    torch.manual_seed(0)
    train(DormouseModel(run.model), run, log=lambda *_: None)
    weights, _snap = ckpt_paths(run.train)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    ck["moe"] = {name: torch.full_like(b, +72.67) for name, b in ck["moe"].items()}
    torch.save(ck, weights)

    torch.manual_seed(0)
    fresh = DormouseModel(run.model)
    load_model_state(fresh, run.train, load_model_half(weights))
    out = capsys.readouterr().out
    assert "DIVERGED" in out and "72.670" in out
    assert float(fresh.loop.moe.lfb_bias.max()) == pytest.approx(72.67, abs=1e-3)


# --- f. the routing path reads nothing back -----------------------------------


def test_the_routing_path_never_reads_the_host(monkeypatch):
    """A D2H read is a `Tensor.item/.tolist/.cpu/__bool__/__float__/__int__`.

    The patched readers record their IMMEDIATE caller. `_naive_dispatch`'s
    `torch.unique(ids_in).tolist()` is the pre-existing CPU-fallback read this
    test names; the gate is that NO read comes from the routing code added here
    (`forward`, `_lfb_step`, `_route_health`, `switch_aux_loss`) on any arm.

    Revert: put a `float(term)` in `aux_loss()`, an `.item()` on `counts.max()`
    in `_route_health`, or a `bool(mask.any())` in the forward — the caller set
    gains that function's name and the test fails.
    """
    readers: list[str] = []

    def wrap(name: str, fn):
        def inner(self, *a, **k):
            frames = [fr.name for fr in traceback.extract_stack()[:-1]]
            readers.append(frames[-1] if frames else "?")
            return fn(self, *a, **k)

        return inner

    for name in ("item", "tolist", "cpu", "__bool__", "__float__", "__int__"):
        monkeypatch.setattr(torch.Tensor, name, wrap(name, getattr(torch.Tensor, name)))

    torch.manual_seed(0)
    x = torch.randn(SEQ, 64)
    for arm in (
        dict(moe_lb_mode="sign", moe_lb_coef=0.05),
        dict(moe_lb_mode="prop", moe_lb_coef=0.0, moe_lb_gamma=0.05),
        dict(moe_lb_mode="aux", moe_aux_coef=0.01),
    ):
        moe = SparseMoE(tiny_cfg(moe_routing_health=True, **arm)).train()
        moe.begin_step()
        moe(x, Counters())
        if moe.lb_mode == "aux":
            assert moe.aux_loss() is not None

    routed = {
        "forward",
        "_lfb_step",
        "_route_health",
        "switch_aux_loss",
        "aux_loss",
        "begin_step",
        "lb_active",
    }
    offenders = sorted(set(readers) & routed)
    assert not offenders, f"the routing path read the device back in {offenders}"
    assert set(readers) <= {"_naive_dispatch"}, f"unexpected read sites {sorted(set(readers))}"
