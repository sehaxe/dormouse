"""`muon_variant="orth_dion"`: ONE sketch instead of five Newton–Schulz passes.

The lane's ask: the Newton–Schulz chain is 113 ms of the live step's 377 ms at
b16×1024, and its FLOPs are already at the full-rank floor, so the only way down
is to orthogonalize a `rank`-r approximation of the momentum. This file is the
gate on the four claims that can fail:

1. **it points at the rank-r polar** — cosine against a reference computed IN THE
   TEST (`torch.linalg.svd` of the fixture, never a golden captured from our own
   run), on the live model's four momentum shapes;
2. **it points there better than Newton–Schulz does** — at every rank, and on a
   flat spectrum too, where the ABSOLUTE 0.99 gate cannot be met by anything;
3. **more power iterations never hurt** (the sketch's own convergence, which a
   non-re-orthonormalised implementation silently reverses);
4. **at `rank = min(n, f)` it IS the polar**, and it is a partial isometry of
   norm `√rank` — the scale Muon's ColRow tail and `-lr·scale` are built on.

Two negative controls are pinned as tests, because both are plausible
"simplifications" of this code and both were measured wrong:

* `qr(Bᵀ)` for the right factor (the brief's literal form) — cosine **-0.022**;
* an fp32 Gram/basis — cosine **0.80** where fp64 gives 0.9989.

The numbers on the LIVE momentum (60 matrices of `arch_gated_swiglu` after 40
steps) are in `orth_dion_polar`'s docstring; the ones here are on fixtures a CPU
gate can afford, and they state which regime each assertion is about.
"""

from __future__ import annotations

import pytest
import torch

from dormouse.instrument import LoudError
from dormouse.train.optim import (
    ORTH_DION_OVERSAMPLE,
    ORTH_DION_POWER,
    ORTH_DION_RANK,
    MixOptimizer,
    Muon,
    colrow_normalize,
    orth_dion_polar,
    zeropower_via_ns,
)

#: The live model's four Muon shapes (`configs/arch_gated_swiglu.toml`, b16):
#: `qkv [2496, 832]` ×12, `proj`/`[832, 832]` ×12, `gate`+`up` `[2048, 832]` ×24,
#: `down [832, 2048]` ×12 — the 60 matrices `NS-5` spends 113 ms on.
LIVE_SHAPES = [(2048, 832), (2496, 832), (832, 832), (832, 2048)]

#: The live momentum's decay: `σ_64/σ_1 ≈ 0.024` and `σ_128/σ_1 ≈ 6e-4`
#: (measured on those 60 matrices). The fixture below reproduces those two ratios
#: because they are what decides whether the sketch can resolve the cut.
K_DECAY = 48.0


def momentum_like(n: int, f: int, seed: int = 0) -> torch.Tensor:
    """A `[n, f]` fixture with the live momentum's measured spectrum.

    Not a golden and not a copy of a run: two seeded random factors with an
    explicit exponential spectrum, so the spectrum is a PARAMETER of the fixture
    (`K_DECAY`) a reader can move to leave the regime. The `1e-3` floor is the
    fixture's numerical rank: a matrix whose spectrum decays past fp32's epsilon
    has no `rank-r` polar to speak of at `r = min(n, f)` (the live momentum's own
    `σ_min/σ_1` stays above 1e-6), and a gate on a null space would be measuring
    round-off.
    """
    g = torch.Generator().manual_seed(seed)
    k = min(n, f)
    spec = torch.exp(-torch.arange(k, dtype=torch.float32) / (k / K_DECAY)).clamp_min(1e-3)
    u = torch.randn(n, k, generator=g)
    v = torch.randn(f, k, generator=g)
    return ((u * spec) @ v.T) / k**0.5


def rank_r_polar(m: torch.Tensor, r: int) -> torch.Tensor:
    """`U_r·V_rᵀ`, the object the arm approximates — computed here, not stored."""
    u, _s, vh = torch.linalg.svd(m, full_matrices=False)
    return u[:, :r] @ vh[:r]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.flatten() @ b.flatten() / (a.norm() * b.norm())).item()


def test_the_gate_holds_on_the_live_shapes() -> None:
    """Cosine ≥ 0.99 with the rank-64 polar on all four shapes, at the defaults.

    The fixture is the live momentum's regime (`σ_64/σ_1 = 0.024`). On the real
    60 matrices the same call measured 0.9989 mean / 0.9976 min; the gate is the
    task's, and it is anchored on an SVD this test computes.
    """
    assert (ORTH_DION_RANK, ORTH_DION_POWER, ORTH_DION_OVERSAMPLE) == (64, 1, 32)
    for n, f in LIVE_SHAPES:
        m = momentum_like(n, f)
        got = cosine(orth_dion_polar(m, ORTH_DION_RANK), rank_r_polar(m, ORTH_DION_RANK))
        assert got >= 0.99, f"[{n}, {f}] cosine {got:.4f} < 0.99"


def test_no_oversample_is_what_fails_the_gate() -> None:
    """`oversample=0` is 0.97, not 0.99: the guard block is the mechanism.

    Measured 0.9388 on the live matrices and 0.9711 here — this is the assertion
    that fails if someone "simplifies" the default to a bare rank-r sketch.
    """
    m = momentum_like(*LIVE_SHAPES[0])
    ref = rank_r_polar(m, 64)
    bare = cosine(orth_dion_polar(m, 64, 1, 0), ref)
    guarded = cosine(orth_dion_polar(m, 64, 1, 32), ref)
    assert bare < 0.99 <= guarded, f"bare {bare:.4f} guarded {guarded:.4f}"


@pytest.mark.parametrize("r", [32, 64, 128])
def test_the_sketch_is_closer_to_the_polar_than_newton_schulz(r: int) -> None:
    """The claim that survives a flat spectrum: better than NS-5 at every rank.

    The ABSOLUTE gate is a property of the spectrum — on an iid Gaussian the
    rank-r polar weights `r` directions of a bulk spectrum equally, so no sketch
    and not even the full polar gets close (measured: ours 0.61, NS-5 0.32, the
    full rank-832 polar 0.28 at r=64). What still holds, and is what this test
    pins, is that the sketch is the closest of the three at every rank.
    """
    g = torch.Generator().manual_seed(11)
    m = torch.randn(832, 832, generator=g)
    ref = rank_r_polar(m, r)
    ours = cosine(orth_dion_polar(m, r, 1, 32), ref)
    ns5 = cosine(zeropower_via_ns(m, 5), ref)
    full = cosine(rank_r_polar(m, 832), ref)
    assert ours > ns5, f"r{r}: sketch {ours:.4f} vs NS-5 {ns5:.4f}"
    assert ours > full, f"r{r}: sketch {ours:.4f} vs full polar {full:.4f}"


def test_more_power_iterations_never_hurt() -> None:
    """`p=0 < p=1 < p=2`. Fails if the power iteration is not re-orthonormalised.

    Without re-orthonormalisation the sketch's spectrum is cubed by `M·Mᵀ·M` and
    `p=2` measures WORSE than `p=0` (that is the measured defect this pins: the
    same shape, the same seed, 0.337 -> 0.585 -> 0.732 only with a basis step).
    """
    m = momentum_like(832, 832)
    ref = rank_r_polar(m, 32)
    got = [cosine(orth_dion_polar(m, 32, p, 16), ref) for p in (0, 1, 2)]
    assert got[0] < got[1] < got[2], f"power iterations not monotone: {got}"


@pytest.mark.parametrize("shape", [(832, 832), (1024, 832)])
def test_full_rank_is_the_exact_polar(shape: tuple[int, int]) -> None:
    """`rank = min(n, f)` reduces to the polar (1e-2 relative; measured 2e-05).

    This is the analytic anchor — it is why `rank == min(n, f)` is served with
    `power=0`: a power iteration at full rank cubes the spectrum in fp32 and
    measured 0.62 cosine where the bare sketch measures 1.000000. The fixture is
    a plain Gaussian (condition number ~1e2) on purpose: the identity is about
    the arithmetic, and a spectrum with a null tail would make it a statement
    about round-off.
    """
    torch.manual_seed(1234)
    m = torch.randn(*shape)
    r = min(shape)
    ref = rank_r_polar(m, r)
    got = orth_dion_polar(m, r, power=3, oversample=64)  # both are overridden
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel <= 1e-2, f"{shape}: relative {rel:.2e}"


def test_qr_of_the_right_factor_is_not_the_polar() -> None:
    """The brief's literal `qr(Bᵀ)` form is a ROTATION, not the polar — pinned.

    `Q` is merely some basis of the subspace, so `B = RᵀΣWᵀ` and `qr(Bᵀ)` returns
    `V·orth(ΣRᵀ)ᵀ`. Measured -0.022 on the live matrices; this test would pass
    for a "simplification" that kept only the cosine of the *subspace*.
    """
    m = momentum_like(*LIVE_SHAPES[0])
    ref = rank_r_polar(m, 64)
    r, over = 64, 32
    g = torch.Generator().manual_seed(7)
    omega = torch.randn(m.shape[1], r + over, generator=g)
    s = m @ omega
    q, _ = torch.linalg.qr(s)
    b = q.T @ m
    v, _ = torch.linalg.qr(b.T)
    literal = q @ v.T
    assert cosine(literal, ref) < 0.5, "the literal QR form 'works' only if it is wrong"
    assert cosine(orth_dion_polar(m, r, 1, over), ref) > 0.9


def test_the_update_is_a_partial_isometry_of_norm_sqrt_rank() -> None:
    """`tr(UᵀU) = rank` and `‖U‖_F = √rank` — the scale Muon's tail assumes.

    Without it the ColRow tail and `-lr·scale` would no longer give RMS(dW) = lr,
    which is the whole `shape_fair` argument (`_muon_scale`'s docstring).
    """
    m = momentum_like(512, 256)
    u = orth_dion_polar(m, 48, 1, 16)
    assert abs((u.T @ u).trace().item() - 48.0) < 1e-2
    assert abs(u.norm().item() - 48**0.5) < 1e-3


def test_the_per_head_split_is_refused() -> None:
    """An untested rung combination: `head_dim` sketches, per projection.

    The arm claims a cost at `[n, f]` with `n, f >= 832`; a per-head split makes
    the sketch `head_dim x cols` and `rank=64` is then the FULL-rank rung of that
    shape — a different program with a different price, so it is refused rather
    than run unmeasured.
    """
    p = torch.nn.Parameter(torch.zeros(64, 64))
    p._muon_per_head = (4, 16)
    with pytest.raises(LoudError, match="muon_per_head"):
        Muon([p], variant="orth_dion", orth_dion_rank=16)


def test_a_batch_equals_the_same_matrices_one_by_one() -> None:
    """The 3-D form is the 2-D form per slice — the batched step relies on it."""
    a = momentum_like(512, 256, seed=1)
    b = momentum_like(512, 256, seed=2)
    batched = orth_dion_polar(torch.stack([a, b]), 32, 1, 16)
    one = orth_dion_polar(a, 32, 1, 16)
    two = orth_dion_polar(b, 32, 1, 16)
    # `bmm` and the batched `eigh` round differently from `mm`/single-slice, and
    # an ill-conditioned sketch amplifies that in the basis: the bar is 1e-3 on
    # the update (the values are O(1/sqrt(f))), not bit-identity.
    assert (batched[0] - one).abs().max() < 1e-3
    assert (batched[1] - two).abs().max() < 1e-3
    assert cosine(batched[0], one) > 0.9999


def test_a_quadratic_decreases_for_twenty_steps() -> None:
    """`0.5‖W − W*‖²` decreases every step under the arm's own update.

    The sketch is an approximation, so this is the check that it is still a
    DESCENT direction and not merely a unit-norm matrix: the angle to `-∇` is
    acute, so a small enough step decreases the loss monotonically.
    """
    target = momentum_like(256, 128, seed=3)
    w = torch.zeros(256, 128)
    losses = []
    for _ in range(20):
        grad = w - target
        w = w - 0.05 * colrow_normalize(orth_dion_polar(grad, 32, 1, 16))
        losses.append(0.5 * (w - target).pow(2).sum().item())
    assert all(b < a for a, b in zip(losses, losses[1:], strict=False)), losses
    assert losses[-1] < 0.6 * losses[0], losses


# --- the door: every rung the arm cannot serve is LOUD ----------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(rank=0), "whole number >= 1"),
        (dict(rank=-3), "whole number >= 1"),
        (dict(rank=True), "whole number >= 1"),
        (dict(rank=65), r"cannot\s+exceed min\(n, f\)=64"),
        (dict(rank=8, power=-1), "whole number >= 0"),
        (dict(rank=8, oversample=-1), "whole number >= 0"),
    ],
)
def test_impossible_ranks_are_refused_loudly(kwargs: dict, match: str) -> None:
    m = torch.randn(128, 64)
    with pytest.raises(LoudError, match=match):
        orth_dion_polar(m, **kwargs)


def test_a_non_matrix_is_refused() -> None:
    with pytest.raises(LoudError, match="rank-1 tensor"):
        orth_dion_polar(torch.randn(64), 4)
    with pytest.raises(LoudError, match="rank-4 tensor"):
        orth_dion_polar(torch.randn(2, 2, 2, 2), 2)


# --- wiring: the arm is reachable, COUNTED, and off by default ---------------


def _tiny_step(model, ids: torch.Tensor | None = None) -> None:
    ids = torch.randint(0, 256, (2, 32)) if ids is None else ids
    _logits, loss, _c = model(ids, ids)
    loss.backward()


def test_the_arm_is_reachable_and_counted(model) -> None:
    """`muon_variant="orth_dion"` runs, moves the Muon half and COUNTS itself."""
    _tiny_step(model)
    opt = MixOptimizer(model, lr=1e-3, wd=0.0, muon_variant="orth_dion", orth_dion_rank=4)
    assert opt.muon.variant == "orth_dion"
    graded = [p for p in opt.muon_params if p.grad is not None]
    before = [p.detach().clone() for p in graded]
    opt.step()
    assert opt.counters.get("muon_orth_dion", 0) > 0
    assert getattr(opt.counters, "why", None) is None, opt.counters.why
    assert graded, "the fixture gave no Muon parameter a gradient"
    assert all(not torch.equal(b, p) for b, p in zip(before, graded, strict=True)), (
        "the arm reported itself on and moved nothing"
    )


def test_the_default_path_never_calls_the_sketch(model, monkeypatch) -> None:
    """Flag off is the Newton–Schulz path, structurally — not by a tolerance."""

    def boom(*_a, **_k):  # pragma: no cover - it must not be reached
        raise AssertionError("the default path called the orth_dion sketch")

    monkeypatch.setattr("dormouse.train.optim.orth_dion_polar", boom)
    _tiny_step(model)
    opt = MixOptimizer(model, lr=1e-3, wd=0.0)
    opt.step()
    assert "muon_orth_dion" not in opt.counters


def test_a_rank_without_the_arm_is_refused(model) -> None:
    """The rank configures `orth_dion` only; set alone it would be inert."""
    with pytest.raises(LoudError, match="would\\s+be inert|inert"):
        MixOptimizer(model, lr=1e-3, wd=0.0, orth_dion_rank=32)


def test_the_arm_without_a_rank_is_refused(model) -> None:
    with pytest.raises(LoudError, match="rank below 1"):
        MixOptimizer(model, lr=1e-3, wd=0.0, muon_variant="orth_dion", orth_dion_rank=0)


def test_a_rank_the_model_cannot_carry_is_refused_at_the_step(model) -> None:
    """The rank is checked against the SHAPE, which only the step knows.

    The tiny block's smallest Muon matrix is `[4, 64]`, so a rank of 4096 is
    refused the moment the shape is in hand — and the message names it.
    """
    _tiny_step(model)
    opt = MixOptimizer(model, lr=1e-3, wd=0.0, muon_variant="orth_dion", orth_dion_rank=4096)
    with pytest.raises(LoudError, match=r"the rank cannot\s+exceed min\(n, f\)"):
        opt.step()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(ns_kind="triton"), "inert"),
        (dict(ns_dtype="bf16"), "inert"),
        (dict(muon_sf=True, muon_sf_wd=0.01), "inert"),
    ],
)
def test_the_rungs_the_arm_does_not_run_are_refused(model, kwargs: dict, match: str) -> None:
    """No silent no-op: the arm runs no NS chain and no SF magnitude path."""
    with pytest.raises(LoudError, match=match):
        MixOptimizer(model, lr=1e-3, wd=0.0, muon_variant="orth_dion", orth_dion_rank=4, **kwargs)


def test_the_two_arms_apply_the_same_magnitude(model) -> None:
    """Both arms are ColRow-tailed, so `‖ΔW‖_F` is the same to a few percent.

    This is the convention check the lane asked for ("compare against the existing
    update, not against a fresh convention"): the sketch changes the DIRECTION of
    the step and its rank, not the size Muon applies.
    """
    import copy

    twin = copy.deepcopy(model)  # cloned BEFORE any gradient exists (a graph is not copyable)
    ids = (torch.arange(64).reshape(2, 32) * 7) % 256
    deltas = {}
    for name, m, kwargs in (
        ("ns", model, {}),
        ("sketch", twin, dict(muon_variant="orth_dion", orth_dion_rank=4)),
    ):
        _tiny_step(m, ids)
        opt = MixOptimizer(m, lr=1e-3, wd=0.0, **kwargs)
        graded = [p for p in opt.muon_params if p.grad is not None]
        before = [p.detach().clone() for p in graded]
        opt.step()
        delta = [(p.detach() - b).flatten() for b, p in zip(before, graded, strict=True)]
        deltas[name] = torch.cat(delta).norm().item()
    ratio = deltas["sketch"] / deltas["ns"]
    assert 0.9 < ratio < 1.1, (
        f"‖ΔW‖ ratio {ratio:.3f} (ns {deltas['ns']:.4f}, sketch {deltas['sketch']:.4f})"
    )
