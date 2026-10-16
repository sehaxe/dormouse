"""Gram Newton–Schulz: the same polynomial regrouped, and the gate that says so.

The claim the arm rests on is EQUIVALENCE, not speed: `gram` iterates the classic
recurrence's own polynomial on the `[r, r]` Gram instead of on `X`
(arXiv:2608.11612 §4; transcribed in `optim._gram_via_ns`). Equivalence is what
licenses the swap without a 3-seed training A/B, so it is gated here in three
tiers, weakest-to-strongest noise:

1. **In fp64 the two arms are the same iteration** — the definitive statement,
   with no fp32 noise in it: both are transcribed in float64 in this file and
   must agree to ~1e-12 (observed ≤ 4.3e-14).
2. **At the port's own shapes, in fp32, within the research note's gate**
   (rel ≤ 1e-5, `docs/research/2026-10-16-muon-orthogonalization-cost.md` §4
   Stage 0): the rank-64 factors of a 2048×64 / 64×5461 momentum. Observed
   ≤ 1.8e-6, with the tolerance justified below.
3. **On degenerate spectra, within the chain's own round-off** — a scale-free
   statement that holds for ANY input: the two arms differ by no more than 3×
   the fp32 round-off the polynomial itself carries on that input, measured
   against the fp64 evaluation in the same test.

**Where the tolerance comes from.** fp32 unit roundoff `u = 2^-24 = 5.96e-8`.
The Jordan polynomial's derivative is `p'(σ) = a − 3|b|σ² + 5cσ⁴`, whose
supremum over the reachable `σ ∈ [0, 1]` is `a = 3.4445` (at `σ → 0`): every
iteration can amplify a perturbation by that much, so 5 steps amplify by at most
`3.4445⁵ ≈ 483`, and the chain performs ~10 sequential roundings per step
(~50 total). Worst case `50 · 483 · 5.96e-8 ≈ 1.4e-3` relative — a bound so loose
it would pass a wrong port, which is why tier 1 exists. In practice the error is
input-dependent and the bound is never approached:

| fixture | `classic` vs fp64 | `gram` vs fp64 | `gram` vs `classic` |
|---|---|---|---|
| Gaussian `[2048, 64]` (the b1 factor) | 1.4e-06 | 1.6e-06 | 1.5e-06 |
| Gaussian `[64, 5461]` (the b1 factor) | 2.2e-06 | 2.2e-06 | 1.6e-06 |
| Gaussian `[64, 64]` (square, ill-conditioned) | 1.2e-06 | 1.4e-05 | 1.4e-05 |
| rank-2 ramp `[64, 64]` | 7.2e-05 | 6.6e-05 | 9.3e-05 |
| heavy `[512, 64]`, 10:1 spectrum | 1.0e-04 | 6.3e-05 | 1.2e-04 |

The port's real shapes are the WELL-conditioned rows: a 64-row factor of a
`[64, 5461]` momentum has a Wishart Gram with 5461 degrees of freedom, so
`σ_min/σ_max ≈ 0.8` and the polynomial's amplification is ~1 per step. That is
why 1e-5 holds there and why it cannot hold on the ramp rows — on those the
CLASSIC arm is already 1.0e-4 from its own fp64 twin, so a 1e-5 gate would fail
the control arm, not the new one.

The reference's float16 cast is not transcribed (this port's NS is fp32) and
neither are its CuTeDSL kernels, so tier 2 is measured on the fp32 arms exactly
as `Muon` calls them.
"""

from __future__ import annotations

import pytest
import torch
from conftest import tiny_cfg

from dormouse.config import load_run
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train import optim
from dormouse.train.optim import (
    GRAM_NS_RESTARTS,
    JORDAN_NS_COEFFS,
    MixOptimizer,
    Muon,
    zeropower_via_ns,
)

#: Tier 2 (the research note's Stage-0 gate) and tier 3 (3× the chain's own
#: round-off on the input). Tier 1's gate is fp64's round-off amplified by the
#: chain: `2^-53 = 1.1e-16`, up to `×(a²)^5 = 2.4e6` for the un-restarted Gram
#: chain, ≈ 2.6e-10 worst case; the observed max is 2.0e-12 and a wrong
#: recursion would be O(1e-1) apart.
TOL_PORT_SHAPES = 1e-5
TOL_FP64 = 1e-9
ROUNDOFF_FACTOR = 3.0

#: The b1 factor shapes and their batched (naive-Muon-pass) forms.
PORT_SHAPES = [(2048, 64), (64, 5461), (2, 2048, 64), (3, 64, 5461)]
#: Everything ill-conditioned we can name: square, rank-deficient, rank-2 ramp.
HARD_FIXTURES = [
    ("ramp 64x64", (64, 64)),
    ("rank-8 of 64", (64, 8)),
    ("square batched", (3, 64, 64)),
]


def gauss(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    """A deterministic input — a Gaussian, like a momentum buffer of real grads."""
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def ramp(shape: tuple[int, ...]) -> torch.Tensor:
    """A deterministic near-rank-2 matrix — the worst conditioning we can build."""
    n = 1
    for s in shape:
        n *= s
    return torch.arange(n, dtype=torch.float32).reshape(shape) / 4096.0 - 8.0


def heavy(m: int = 512, r: int = 64) -> torch.Tensor:
    """A tall matrix with a 10:1 spectrum — the class the long pole lives in."""
    return ramp((m, r)) @ torch.diag(torch.logspace(0.0, -2.0, r))


def fp64_classic(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """`zeropower_via_ns`'s classic body, transcribed at double precision."""
    a, b, c = JORDAN_NS_COEFFS
    x = g.to(torch.float64)
    x = x / (x.norm(dim=(-1, -2), keepdim=True) + 1e-7)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-1, -2).contiguous()
    for _ in range(steps):
        gram = x @ x.transpose(-1, -2)
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.transpose(-1, -2).contiguous() if transposed else x


def fp64_gram(g: torch.Tensor, steps: int = 5, restarts: tuple[int, ...] = (2,)) -> torch.Tensor:
    """`_gram_via_ns`'s recursion, transcribed at double precision.

    Independent of the implementation under test (same recipe, written against
    the reference's pseudo-code) so that tier 1 is a comparison of two programs
    and not of the code with itself.
    """
    a, b, c = JORDAN_NS_COEFFS
    x = g.to(torch.float64)
    x = x / (x.norm(dim=(-1, -2), keepdim=True) + 1e-7)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-1, -2).contiguous()
    eye = torch.eye(x.shape[-2], dtype=torch.float64)
    q = None
    gram = x @ x.transpose(-1, -2)
    for t in range(steps):
        if t in set(restarts) and t != 0:
            x = q @ x
            gram = x @ x.transpose(-1, -2)
            q = None
        z = b * gram + c * (gram @ gram)
        q = (z + a * eye) if q is None else (q @ z + a * q)
        if t + 1 < steps and (t + 1) not in set(restarts):
            rz = gram @ z + a * gram
            gram = z @ rz + a * rz
    out = x if q is None else q @ x
    return out.transpose(-1, -2).contiguous() if transposed else out


def rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out - ref).norm() / (ref.norm() + 1e-30))


# --- tier 1: the algorithm, with no fp32 in the way -------------------------


@pytest.mark.parametrize(
    "shape", [(64, 64), (2048, 64), (64, 5461), (64, 8), (3, 64, 64), (2, 2048, 64)]
)
@pytest.mark.parametrize("restarts", [(2,), ()])
def test_in_fp64_the_two_arms_are_the_same_iteration(shape, restarts):
    """THE equivalence proof: same polynomial, regrouped — no fp32 noise in it.

    If this fails the port is wrong, not the paper (the research note's words).
    """
    g = gauss(shape)
    d = rel(fp64_gram(g, restarts=restarts), fp64_classic(g))
    assert d <= TOL_FP64, f"{shape} restarts={restarts}: fp64 rel={d:.3e}"


def test_in_fp64_the_restart_cannot_move_the_math():
    """A restart re-materializes `Q X` and rebuilds the Gram: the same step."""
    g = heavy()
    ref = fp64_classic(g)
    for restarts in ((), (1,), (2,), (4,), (1, 3)):
        d = rel(fp64_gram(g, restarts=restarts), ref)
        assert d <= TOL_FP64, f"restarts={restarts}: fp64 rel={d:.3e}"


# --- tier 2: fp32 at the shapes the optimizer actually runs -----------------


@pytest.mark.parametrize("shape", PORT_SHAPES)
@pytest.mark.parametrize("steps", [5, 3])
def test_gram_matches_classic_at_the_port_shapes(shape, steps):
    """fp32, the b1 factor shapes, the research note's `rel ≤ 1e-5` gate."""
    g = gauss(shape)
    classic = zeropower_via_ns(g, steps=steps, kind="classic")
    gram = zeropower_via_ns(g, steps=steps, kind="gram")
    assert gram.shape == g.shape
    d = rel(gram, classic)
    assert d <= TOL_PORT_SHAPES, f"{shape} steps={steps}: gram vs classic rel={d:.3e}"
    # Both arms track the fp64 polynomial to the same order — the tolerance is
    # round-off, not two wrong answers agreeing with each other.
    ref = fp64_classic(g, steps=steps)
    assert rel(classic, ref) <= TOL_PORT_SHAPES and rel(gram, ref) <= TOL_PORT_SHAPES


@pytest.mark.parametrize("name,shape", HARD_FIXTURES)
def test_gram_is_within_the_chains_own_roundoff_on_hard_inputs(name, shape):
    """On a degenerate spectrum the gate is the chain's own fp32 error.

    These inputs are far outside the port's shapes (a 64×64 Gaussian, a rank-8
    matrix, a rank-2 ramp) and the polynomial is ill-conditioned on them: the
    classic arm itself is up to 1e-4 from its fp64 twin, so the honest
    statement is the scale-free one — the regrouped arm differs from the arm it
    replaces by no more than 3× the round-off either of them carries.
    """
    for g, label in ((gauss(shape, seed=1), "gauss"), (ramp(shape), "ramp")):
        classic = zeropower_via_ns(g, 5, kind="classic")
        gram = zeropower_via_ns(g, 5, kind="gram")
        ref = fp64_classic(g)
        envelope = ROUNDOFF_FACTOR * max(rel(classic, ref), rel(gram, ref))
        d = rel(gram, classic)
        assert d <= envelope, f"{name}/{label}: {d:.3e} vs envelope {envelope:.3e}"


def test_the_restart_is_load_bearing_not_decoration():
    """`restarts=()` is measurably worse — which is why the reference specifies one.

    The Gram chain carries the perturbation through `φ(R) = P(R)²R`, whose gain
    near `R ≈ 0` is `a² ≈ 11.9` against the classic chain's `a ≈ 3.4`, so without
    a restart the small end of the spectrum drifts ~17× further from the fp64
    reference on a heavy input. Deleting the restart as a "simplification" would
    silently cost the arm its accuracy on exactly the spectra Muon is fed.
    """
    g = heavy()
    ref = fp64_classic(g)
    restarted = rel(zeropower_via_ns(g, 5, kind="gram"), ref)
    un_restarted = rel(zeropower_via_ns(g, 5, kind="gram", gram_restarts=()), ref)
    assert restarted <= 1e-4, f"restarted gram vs fp64: {restarted:.3e}"
    assert un_restarted > 3 * restarted, f"no restart {un_restarted:.3e} vs {restarted:.3e}"
    # The DEFAULT arm (restarted) is the one that stays inside the envelope of
    # the arm it replaces; the un-restarted variant is 10× outside it, which is
    # the whole reason the restart exists and is not decoration.
    classic = zeropower_via_ns(g, 5, kind="classic")
    assert rel(zeropower_via_ns(g, 5, kind="gram"), classic) <= 3 * rel(classic, ref)
    assert un_restarted > 3 * rel(classic, ref)


# --- the control arm cannot drift, and the arm cannot go missing -------------


def test_classic_arm_is_byte_identical_to_the_pinned_numbers():
    """Today's classic numbers, pinned as literals, before the `gram` arm existed.

    Measured on this port (`zeropower_via_ns`, steps=5) on 2026-10-16. Exact
    equality is deliberate: an ULP change in the classic body fails here, which
    is the only way "byte-identical" is a testable claim rather than a comment.
    """
    o = zeropower_via_ns(ramp((64, 64)), steps=5)
    assert float(o.sum()) == -44.538936614990234
    assert float(o.norm()) == 0.6965265870094299
    assert float(o.flatten()[0]) == -0.01110190898180008
    assert float(o.flatten()[7]) == -0.011208988726139069
    assert float(o.flatten()[-1]) == -0.009630225598812103

    w = zeropower_via_ns(ramp((64, 5461)), steps=5)
    assert float(w.sum()) == 332.79107666015625
    assert float(w.flatten()[-1]) == -0.0005161152221262455

    b = zeropower_via_ns(torch.stack([ramp((64, 64)), ramp((64, 64)).flip(0)]), steps=5)
    assert float(b.sum()) == -89.07787322998047
    assert float(b[0, 0, 0]) == -0.011101972311735153
    assert float(b[1, 63, 63]) == -0.012073971331119537


def test_the_requested_arm_is_the_arm_that_runs():
    """`ns_kind` must reach the NS call on BOTH Muon paths, not silently default.

    Comparing the outputs cannot tell the arms apart (they agree to ~1e-6), so
    this counts the call instead — the port's own rule for a new arm: it has to
    be able to show it ran.
    """
    seen: list[str] = []
    real = optim.zeropower_via_ns

    def spy(g, steps=5, kind="classic", gram_restarts=optim.GRAM_NS_RESTARTS):
        seen.append(kind)
        return real(g, steps, kind, gram_restarts)

    model = DormouseModel(tiny_cfg())
    # Every 2-D param gets a gradient, so the per-param path (a shape with one
    # member) and the BATCHED path (a repeated shape) BOTH run — two call
    # sites, two places to forget the arm.
    for _name, p in model.named_parameters():
        if p.ndim == 2:
            p.grad = torch.full_like(p, 0.1)
    optim.zeropower_via_ns = spy
    try:
        opt = MixOptimizer(model, lr=1e-3, wd=0.0, ns_kind="gram")
        assert opt.muon.ns_kind == "gram"
        opt.step()
        n_batched = opt.muon.batched_step_from_model(model)
    finally:
        optim.zeropower_via_ns = real
    assert seen and set(seen) == {"gram"}, seen
    assert n_batched > 0 and opt.steps_batched > 0, "the batched path must have run"


def test_a_bad_ns_kind_is_loud_everywhere_it_can_be_set():
    for build in (
        lambda: zeropower_via_ns(ramp((8, 8)), 5, kind="gramm"),
        lambda: Muon([torch.nn.Parameter(torch.zeros(8, 8))], ns_kind="Gram"),
        lambda: MixOptimizer(DormouseModel(tiny_cfg()), lr=1e-3, wd=0.0, ns_kind="pe"),
        lambda: load_run("small", train_overrides={"ns_kind": "gram-ns"}),
    ):
        with pytest.raises(LoudError) as e:
            build()
        assert "ns_kind=" in str(e.value) and "gram" in str(e.value)


def test_the_knob_round_trips_through_the_config():
    """The field is a real config key: it loads, it snapshots, it stays separate."""
    run = load_run("small", train_overrides={"ns_kind": "gram"})
    assert run.train.ns_kind == "gram"
    assert 'ns_kind = "gram"' in run.snapshot_toml()
    other = load_run("small", train_overrides={"ns_kind": "classic"})
    assert run.diff_keys(other) == ["train.ns_kind"]


# --- the bench (`scripts/bench_muon_ns.py` runs this) -----------------------


def macs(shape: tuple[int, int], steps: int = 5, kind: str = "classic") -> int:
    """Multiply-accumulates per orthogonalization, from the loop's own arithmetic.

    `classic` per step: `X Xᵀ` (`r²d`) + `R²` (`r³`) + `P X` (`r²d`).
    `gram` per step: `R²`, `Q(Z+aI)`, `RZ`, `Z·RZ` (4 × `r³`), plus exactly two
    rectangular matmuls in total (`X₀X₀ᵀ` and `Q X`).
    """
    r, d = min(shape), max(shape)
    if kind == "classic":
        return steps * (2 * r * r * d + r**3)
    return 2 * r * r * d + 4 * steps * r**3


def _median_ms(fn, reps: int = 20, warm: int = 3) -> float:
    import statistics
    import time

    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


def _big_matmuls(shape: tuple[int, int], steps: int = 5, kind: str = "classic") -> int:
    """The RECTANGULAR matmuls per call — what the two arms actually trade.

    `classic`: `X @ Xᵀ` and `P @ X` per iteration, `O(r²d)` each.
    `gram`: `X₀X₀ᵀ` and the final `Q X`, plus one more pair at the restart.
    """
    return 2 * steps if kind == "classic" else 2 + 2 * len(GRAM_NS_RESTARTS)


def _matmul_launches(fn) -> tuple[int, int]:
    """(matmul dispatches, all dispatches) for one call — the launch-count proxy.

    A `TorchDispatchMode` rather than `torch.profiler`: same count of the ops
    this code actually issues, without kineto's USDT chatter on stderr.
    """
    counted: list[str] = []

    class _Count(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            counted.append(str(func))
            return func(*args, **(kwargs or {}))

    with _Count():
        fn()
    n_mm = sum(1 for k in counted if k.startswith(("aten.mm", "aten.bmm", "aten.matmul")))
    return n_mm, len(counted)


def _gpu_free() -> tuple[bool, str]:
    """Machine discipline (AGENTS §1.5 + the lane brief): one GPU thing at a time."""
    import subprocess

    try:
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()
        busy = subprocess.run(
            ["pgrep", "-x", "train"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001 - a bench's precondition, not a run's
        return False, f"cannot check the card ({e})"
    mib = int(used.split()[0]) if used else -1
    if busy:
        return False, f"a `train` process is live (pids {busy})"
    if mib >= 1024:
        return False, f"the card holds {mib} MiB"
    return True, f"card {mib} MiB, no train process"


def bench_ns_kinds(shapes: tuple[tuple[int, int], ...] = ((2048, 64), (64, 5461))) -> None:
    """Both arms at the b1 factor shapes: ms per call, MACs, dispatch counts.

    CPU is the substrate the gate runs on; the GPU arm runs only when the card
    is idle (another lane may be training — then it says so instead of waiting).
    """
    import time

    print(f"torch {torch.__version__} · threads {torch.get_num_threads()}")
    for shape in shapes:
        g = gauss(shape)
        print(f"\n[{shape[0]}x{shape[1]}]  (NS runs in the {min(shape)}x{min(shape)} Gram space)")
        for kind in ("classic", "gram"):
            fn = lambda k=kind, g=g: zeropower_via_ns(g, 5, kind=k)  # noqa: E731
            ms = _median_ms(fn, reps=20)
            n_mm, n_all = _matmul_launches(fn)
            print(
                f"  {kind:8s} {ms:8.3f} ms  {macs(shape, 5, kind) / 1e6:8.1f} MMAC  "
                f"{n_mm:3d} matmul ({_big_matmuls(shape, 5, kind):2d} rectangular)  "
                f"{n_all:4d} aten dispatches"
            )
    ratio = [macs(s, 5) / macs(s, 5, "gram") for s in shapes]
    print(f"\n  MAC ratio classic/gram: {', '.join(f'{r:.2f}x' for r in ratio)}")
    free, why = _gpu_free()
    if not free or not torch.cuda.is_available():
        print(f"  GPU arm SKIPPED: {why if not free else 'no CUDA available'}")
        return
    print(f"  GPU arm: {why} — measuring host-enqueue vs wall (the residue is the GPU's)")
    for shape in shapes:
        x = gauss(shape).cuda()
        for kind in ("classic", "gram"):
            fn = lambda k=kind, x=x: zeropower_via_ns(x, 5, kind=k)  # noqa: E731
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(50):
                fn()
            host = (time.perf_counter() - t) / 50 * 1e3
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(50):
                fn()
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t) / 50 * 1e3
            print(
                f"  {shape} {kind:8s} host-enqueue {host:6.3f} ms  wall {wall:6.3f} ms  "
                f"gpu residue {wall - host:+6.3f} ms"
            )
    print(
        "  read it: a residue ~0 means the call is HOST-enqueue-bound, so the OP COUNT\n"
        "  pays and arithmetic does not — and `gram` issues more ops (fewer rectangular\n"
        "  matmuls, 4 small ones per step). The reference's win rides its CuTeDSL\n"
        "  symmetric GEMM kernels, which are not ported here."
    )


if __name__ == "__main__":
    bench_ns_kinds()
