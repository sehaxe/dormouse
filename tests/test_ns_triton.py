"""The fused Triton Newton–Schulz arm (`train/ns_kernels.py`, `ns_kind="triton"`).

The claim this arm rests on is **speed at equal quality**, not equivalence, so the
gates are:

1. **the recursion is the classic polynomial** — a faithful fp64 transcription of
   the kernel's own three phases (prescale → `R = X Xᵀ` → `X ← aX + P(R)X`, in the
   logical `[r, d]` orientation) against `fp64_classic` (shared from
   `conftest.py`, one transcription of the classic chain, not two), at ≤1e-9.
   No fp32 in the way, so this is the tier the 1e-9 belongs to.
2. **fp32 parity at the b1 shapes** (GPU): the kernel vs the classic chain at
   rel ≤ 1e-5, the research note's Stage-0 gate
   (`docs/research/2026-10-16-muon-orthogonalization-cost.md` §4).
3. **ONE kernel launch per chain** (CPU, by construction): the op's only launch is
   counted with a proxy — the claim that this arm exists to make is a launch
   count, and a claim like that gets a gate, not a comment.
4. **orientation**: the strides the op hands the kernel are the LOGICAL `[r, d]`
   ones, so a tall `[2048, 64]` runs its chain on the 64-side without a transpose
   copy. Checked on the launch's own arguments, on CPU.
5. **the default path cannot move** — sha256 of a fixed-seed 10-step Muon run,
   captured BEFORE this file existed (`/tmp/opencode/capture_ns_triton.py`,
   `optim.py` sha1 `8c9d0f425b9d4eb02aa4d7da14af7a203d1cc2bc` = HEAD `6fda506` +
   the concurrent NorMuon lane's working-tree edits). A re-transcription can drift
   with the code; a hash cannot. **If this goes red after an `optim.py` edit, that
   edit moved the DEFAULT path** — which is the finding, not a broken test.
6. **the fallback is COUNTED, with its reason** (never silent, ADR-0019), and the
   fallback's VALUE is the classic chain bit-for-bit — a fallback that computed
   something else would be a second bug wearing a degradation's clothes. That
   includes the LAUNCH's own refusal (a tile set the device cannot fit): at the
   shipped `BLOCK_D=64` no b1 shape reaches it, so the gate patches the constant
   to 256 — the real tall-shape launch needs 212992 B — and asserts the COUNT.
   (128 was the audit's candidate, but the real tall `(2048, 64)` launch asks
   114688 B and refuses; the table gate records both.)

`≤1e-9` against an fp64 reference on a small matrix was the ask for tier 2; it
cannot hold for this kernel and the split above is why. The kernel is fp32 by
construction (`ns_kernels.zeropower_triton`: "fp32 in / fp32 out"), ~50 sequential
roundings at `u = 2^-24 = 5.96e-8`, and the Jordan quintic amplifies by up to
`a = 3.4445` per step — the honest fp32 envelope is ~1.4e-3 (observed 1.4e-6 at
the b1 shapes). So tier 1 carries the 1e-9 and tier 3 carries that envelope
against fp64.

The GPU tiers are `skipif`'d on `torch.cuda.is_available()`; the CPU tiers are
NOT (a gate that is skipped everywhere it could fire is the defect ADR-0019
exists to prevent). On 2026-10-17 the card was BUSY with the live b1 run, so the
GPU tiers were written and NOT run — the report says so.
"""

from __future__ import annotations

import hashlib
import subprocess

import pytest
import torch
from conftest import fp64_classic, gauss, ramp, rel
from torch import nn

from dormouse.instrument import Counters, LoudError
from dormouse.train import ns_kernels as nk
from dormouse.train import optim
from dormouse.train.optim import JORDAN_NS_COEFFS, MixOptimizer, Muon, zeropower_via_ns

#: Tier 1 (the recursion, in fp64) and tier 2 (fp32, at the port's shapes).
TOL_FP64 = 1e-9
TOL_PORT_SHAPES = 1e-5

#: The b1 factor shapes and the standalone ones the arm must serve. `[2048, 2048]`
#: is the COUNTED-fallback case: its `[r, r]` Gram is 16 MB and does not fit in
#: registers (`nk.MAX_RANK`), which is why the port keeps `[d, d]` projections off
#: Muon in the first place (AGENTS §2.3).
B1_SHAPES = [(1024, 64, 64), (2048, 64), (64, 5461), (2048, 2048)]
FALLBACK_SHAPES = [(2048, 2048)]


def _card_reason() -> str | None:
    """`None` when a GPU gate may run, else why not — the machine discipline.

    Stricter than `test_moe_kernel.py`'s mark on purpose (`torch.cuda.is_available()`
    alone): this port runs ONE heavy GPU thing at a time (AGENTS §1.5), so a gate
    that fires while the b1 run owns the card does not measure anything — it
    allocates into a full card and dies with an OOM that says nothing about the
    kernel. `nvidia-smi`/`pgrep`, not `torch.cuda.mem_get_info()`, so the check
    itself does not build a CUDA context on a card that is in use.
    """
    if not torch.cuda.is_available():
        return "no CUDA device"
    if not nk.TRITON_AVAILABLE:
        return "no triton (the fused chain is the CUDA arm)"
    try:
        busy = subprocess.run(
            ["pgrep", "-f", "dormouse trai[n]"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001 - a gate's precondition, not a run's
        return f"cannot check the card ({e})"
    if busy:
        return f"the live training run holds the card (pids {busy}) — one GPU thing at a time"
    mib = int(used.split()[0]) if used else -1
    if mib >= 1024:
        return f"the card already holds {mib} MiB"
    return None


_GPU_REASON = _card_reason()
CUDA = pytest.mark.skipif(_GPU_REASON is not None, reason=_GPU_REASON or "GPU gate")

#: The optim/config/loop hook is `ns_triton_hook.patch` (a concurrent lane
#: holds `optim.py` + `config.py`). These three gates run the moment it lands.
_HOOK = "triton" in optim.NS_KINDS
needs_hook = pytest.mark.skipif(
    not _HOOK,
    reason=(
        "ns_kind='triton' is not in optim.NS_KINDS: apply ns_triton_hook.patch "
        "(optim.py + config.py + loop.py). This gate is written and runs then."
    ),
)

# --- the pre-change reference, captured as BYTE HASHES ----------------------
#
# `/tmp/opencode/capture_ns_triton.py` against the tree BEFORE this arm landed:
# 10 steps of `Muon(params, lr=1e-3)` (every default: `ns_kind="classic"`,
# `colrow_3d="colrow"`, `state_dtype="fp32"`) on `_SHAPES`, per-step grads from
# `torch.Generator().manual_seed(7)`, `randn * 0.1`. Order = `_SHAPES` insertion
# order: the `(2, 3)` singleton runs `_step_one`, the repeats run the batched
# stack, `(128, 64)` is the tall (transposed) NS orientation and `(8, 128)` the
# wide one. Reproduced twice, byte-identical.

_SHAPES = {(2, 3): 1, (64, 64): 2, (128, 64): 1, (8, 128): 2}

_PRE_CHANGE_SHA = (
    "d0e53b6913cb649092a799b69605d969a21fa4bab9388330a0d5192b73377de3",  # (2, 3)
    "314e018192ad634405fb2de887a177415dc2094311ce97d42a391e60f06a50a1",  # (64, 64)
    "c666141414239f32462e6848b7d093c54e9c58613a5b0507dd400e26ae68bb3b",  # (64, 64)
    "3a899c504fda40a3eba711341466d88fa7c745c83167abc983cdfdab78030651",  # (128, 64)
    "811e97086fb87b406f45d8b57654a8d270e76cbcbb0a00ba3d70a8e730ae6da9",  # (8, 128)
    "cf6b2f860ff7d070f0af9d18db0cc8ed6f04a5851f5a28c13700a2f3756a9f9b",  # (8, 128)
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
    gen = torch.Generator().manual_seed(7)
    for _ in range(10):
        for p in model.parameters():
            p.grad = torch.randn(*p.shape, generator=gen) * 0.1
        muon.batched_step_from_model(model)


def fp64_fused(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """The kernel's own three phases, transcribed in float64.

    Independent of `fp64_classic` on purpose: it builds the LOGICAL `[r, d]`
    orientation (the kernel's stride trick), prescales per slice, and streams the
    Gram/poly/update loop exactly as `_ns_chain_kernel` does — so tier 1 compares
    two programs (the kernel's algorithm vs the classic chain), not the code with
    itself. Written against the kernel's source, not against the reference.
    """
    a, b, c = JORDAN_NS_COEFFS
    x = g.to(torch.float64)
    tall = x.shape[-2] > x.shape[-1]
    w = x.transpose(-1, -2) if tall else x
    w = w / (w.norm(dim=(-1, -2), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = w @ w.transpose(-1, -2)
        poly = b * gram + c * (gram @ gram)
        w = a * w + poly @ w
    return w.transpose(-1, -2) if tall else w


# --- tier 1: the recursion, with no fp32 in the way -------------------------


@pytest.mark.parametrize("shape", [(64, 64), (2048, 64), (64, 5461), (16, 16), (2, 64, 64)])
def test_in_fp64_the_fused_recursion_is_the_classic_polynomial(shape):
    """THE algorithm gate: the kernel's phase structure computes the same thing.

    If this fails the kernel is wrong, not the paper. `steps` is parametrised so a
    transcription that dropped the chain would be caught on `steps=1` too.
    """
    g = gauss(shape)
    d = rel(fp64_fused(g), fp64_classic(g))
    assert d <= TOL_FP64, f"{shape}: fp64 rel={d:.3e}"


def test_in_fp64_the_orientation_trick_cannot_move_the_math():
    """A tall input's chain runs on the `[r, d]` side: the same polar factor.

    The classic chain materializes the transpose; the kernel swaps two stride
    arguments. Both must land on `fp64_classic`'s own answer, at the fp64
    round-off, for the tall, square and wide orientations.
    """
    for shape in [(512, 64), (64, 512), (64, 64), (3, 128, 64), (3, 64, 128)]:
        g = gauss(shape)
        assert rel(fp64_fused(g), fp64_classic(g)) <= TOL_FP64, shape


def test_the_prescale_is_the_frobenius_one_per_slice():
    """`‖x‖_F + 1e-7`, taken PER SLICE when batched — not over the stack.

    The 3-D classic branch normalizes `dim=(1, 2)` with `keepdim`, so each slice
    is its own matrix. A kernel that collapsed the batch would tie every slice's
    scale to the stack's largest; this separates two slices whose norms differ by
    ~100x and checks the fp64 transcription keeps them independent (which the
    tier-1 comparison above then pins against the classic arm).
    """
    big = gauss((1, 32, 32), seed=3)
    small = gauss((1, 32, 32), seed=4) * 0.01
    g = torch.cat([big, small], dim=0)
    assert float(g[0].norm()) > 50 * float(g[1].norm())
    assert rel(fp64_fused(g), fp64_classic(g)) <= TOL_FP64


# --- tier 3: the fallback is COUNTED, and it IS the classic chain -----------


@pytest.mark.parametrize(
    "name,x,dtype,reason",
    [
        ("cpu", torch.randn(64, 64), "fp32", "not CUDA"),
        ("rank", torch.randn(2048, 2048), "fp32", "MAX_RANK"),
        ("fp64", torch.randn(8, 8, dtype=torch.float64), "fp64", "has no kernel"),
        ("rank1d", torch.randn(8), "fp32", "matrix-only"),
    ],
)
def test_a_fallback_is_counted_and_names_its_reason(name, x, dtype, reason):
    """Every degradation a reader of the log cannot see is a defect (ADR-0019)."""
    c = Counters()
    if x.ndim not in (2, 3):
        # The classic chain refuses a non-matrix too (its own `assert`), so the
        # fallback is the SAME refusal, not a different answer. Both are LOUD;
        # the counter is what makes the arm's refusal visible in the log.
        with pytest.raises(AssertionError):
            nk.zeropower_triton_or_classic(x, 5, counters=c)
        assert c.get("ns_triton_fallback") == 1 and reason in c.why, (name, c.why)
        return
    out = nk.zeropower_triton_or_classic(x, 5, counters=c)
    assert c.get("ns_triton_fallback") == 1, (name, dict(c))
    assert c.get("ns_triton", 0) == 0, (name, dict(c))
    assert reason in c.why, (name, c.why)
    assert torch.equal(out, zeropower_via_ns(x, 5, "classic")), name


def test_no_triton_is_a_counted_fallback(monkeypatch):
    """The kernels are an ABILITY, not a dependency: without them, COUNTED."""
    monkeypatch.setattr(nk, "TRITON_AVAILABLE", False)
    x = torch.randn(64, 64)
    c = Counters()
    out = nk.zeropower_triton_or_classic(x, 5, counters=c)
    assert c.get("ns_triton_fallback") == 1 and "no triton" in c.why
    assert torch.equal(out, zeropower_via_ns(x, 5, "classic"))


def test_the_bf16_rung_does_not_compose_with_the_triton_arm():
    """`ns_dtype="bf16"` is a COUNTED fallback, not a silent compose.

    The kernels are fp32-native, so this arm SUBSUMES the bf16 rung
    (`docs/research/2026-10-17-muon-variants.md` §5: "the two must not ride in one
    A/B"). Composing would run the chain in fp32 while the config reported bf16 —
    a config that lies.
    """
    g = torch.randn(64, 64)
    c = Counters()
    out = nk.zeropower_triton_or_classic(g, 5, counters=c, ns_dtype="bf16")
    assert c.get("ns_triton_fallback") == 1 and "SUBSUMES" in c.why, c.why
    assert torch.equal(out, zeropower_via_ns(g, 5, "classic", ns_dtype="bf16"))


def test_the_arm_is_loud_when_it_cannot_count_itself():
    """No `Counters` means the arm could fall back invisibly — that is LOUD."""
    with pytest.raises(LoudError) as e:
        nk.zeropower_triton_or_classic(torch.randn(64, 64), 5)
    assert "Counters" in str(e.value)


def test_a_non_cuda_tensor_never_reaches_the_kernel(monkeypatch):
    """The op refuses what it cannot serve; the DISPATCHER is what counts it."""
    calls: list[tuple] = []
    monkeypatch.setattr(nk, "_ns_chain_kernel", _fake_kernel(calls))
    with pytest.raises(AssertionError) as e:
        nk.zeropower_triton(torch.randn(64, 64))
    assert "contiguous CUDA" in str(e.value)
    assert calls == []


# --- tier 3b: ONE launch per chain, and the orientation of that launch ------


class _fake_kernel:
    """A stand-in for the JITFunction: records the grid and the arguments.

    `_ns_chain_kernel[(grid,)](...)` is Triton's launch syntax, so a proxy with a
    `__getitem__` sees exactly what the op would launch — which makes the arm's
    central claim ("one launch per chunk, on the `min(m, n)` side") a CPU gate.
    """

    def __init__(self, sink: list) -> None:
        self.sink = sink

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.sink.append((grid, args, kwargs))

        return run


def _launch_of(x: torch.Tensor, steps: int = 5, monkeypatch=None):
    seen: list = []
    monkeypatch.setattr(nk, "_check", lambda t: None)
    monkeypatch.setattr(nk, "_ns_chain_kernel", _fake_kernel(seen))
    out = nk.zeropower_triton(x, steps)
    assert len(seen) == 1, f"expected ONE launch per chain, got {len(seen)}"
    grid, args, kwargs = seen[0]
    return out, grid, args, kwargs


@pytest.mark.parametrize(
    "shape,batch,r,d,sr,sd",
    [
        ((1024, 64, 64), (1024,), 64, 64, 64, 1),
        ((2048, 64), (1,), 64, 2048, 1, 64),  # tall: the chain runs on the 64-side
        ((64, 5461), (1,), 64, 5461, 5461, 1),  # wide: no transpose at all
        ((8, 128), (1,), 8, 128, 128, 1),
    ],
)
def test_one_launch_per_chain_in_the_logical_orientation(shape, batch, r, d, sr, sd, monkeypatch):
    """The launch count AND the orientation, both read off the launch itself.

    `sr`/`sd` are the strides of the LOGICAL `[r, d]` matrix: for a tall input
    `sr == 1` (the chain runs down the original columns) and no transpose copy is
    materialized — that copy is one of the dispatches this arm exists to remove.
    """
    x = torch.randn(*shape)
    out, grid, args, kwargs = _launch_of(x, 5, monkeypatch)
    assert grid == batch, (grid, batch)
    assert args[4] == sr and args[5] == sd, (args[4], args[5], sr, sd)
    assert args[6] == r and args[7] == d, (args[6], args[7], r, d)
    assert kwargs["STEPS"] == 5 and kwargs["BLOCK_D"] == nk.BLOCK_D
    assert kwargs["RT"] == max(16, nk.triton.next_power_of_2(r))
    assert out.shape == x.shape and out.dtype == torch.float32
    assert out.device == x.device


def test_the_caller_s_tensor_is_never_written(monkeypatch):
    """The chain runs in place on the OUTPUT; the momentum buffer must survive.

    `Muon._orthogonalize` hands this arm the optimizer's own momentum row — an
    in-place prescale would corrupt the optimizer's state, and the loss curve
    would look merely odd for a thousand steps.
    """
    x = torch.randn(3, 64, 64)
    before = x.clone()
    _launch_of(x, 5, monkeypatch)
    assert torch.equal(x, before)


# --- the default path cannot move -------------------------------------------


@pytest.mark.local_pin
def test_the_default_path_is_byte_identical_to_the_pinned_hashes():
    """10 fixed-seed steps of `Muon` on every default: the hashes captured BEFORE.

    `ns_kind` defaults to `classic`, and this arm must not move it — the pinned
    values come from `/tmp/opencode/capture_ns_triton.py` run against the tree
    before `ns_kernels.py` existed, so they cannot have drifted with it.
    """
    model = _shape_model()
    muon = Muon(list(model.parameters()), lr=1e-3)
    _ten_steps(muon, model)
    got = tuple(_sha(p) for p in model.parameters())
    assert got == _PRE_CHANGE_SHA, [
        (a, b) for a, b in zip(got, _PRE_CHANGE_SHA, strict=False) if a != b
    ]


# --- the hook's own gates (skip until `ns_triton_hook.patch` lands) -----


@needs_hook
def test_the_requested_arm_is_the_arm_that_runs():
    """`ns_kind="triton"` must reach the NS call on BOTH Muon paths, and count.

    Comparing outputs cannot tell the arms apart here (on CPU both are the classic
    chain), so this counts: `ns_triton` on the per-param path and on the batched
    one. A shape group with one member goes through `_step_one`, a repeated shape
    through the stack — two call sites, two places to forget the arm.
    """
    from conftest import tiny_cfg

    from dormouse.model import DormouseModel

    model = DormouseModel(tiny_cfg())
    for _name, p in model.named_parameters():
        if p.ndim == 2:
            p.grad = torch.full_like(p, 0.1)
    opt = MixOptimizer(model, lr=1e-3, wd=0.0, ns_kind="triton")
    assert opt.muon.ns_kind == "triton"
    # `opt.muon.step()` (the per-param path) and `batched_step_from_model` (the
    # stacked path) — NOT `MixOptimizer.step()`: torch's AdamW does a CUDA
    # graph-capture health check on every step, and a CPU gate must not need the
    # card (it is in use by the live run). Both NS call sites are still covered.
    opt.muon.step()
    n_batched = opt.muon.batched_step_from_model(model)
    assert n_batched > 0
    # On CPU every call falls back COUNTED with the CUDA reason — the counter
    # proves the arm was reached on both paths, which is what this gate is for.
    assert opt.counters.get("ns_triton_fallback", 0) >= 2, dict(opt.counters)
    assert "not CUDA" in opt.counters.why


@needs_hook
def test_a_bad_ns_kind_is_loud_and_names_the_new_arm():
    from conftest import tiny_cfg

    from dormouse.config import load_run
    from dormouse.model import DormouseModel

    for build in (
        lambda: zeropower_via_ns(ramp((8, 8)), 5, kind="tritonn"),
        lambda: Muon([nn.Parameter(torch.zeros(8, 8))], ns_kind="Triton"),
        lambda: MixOptimizer(DormouseModel(tiny_cfg()), lr=1e-3, wd=0.0, ns_kind="triton-ns"),
        lambda: load_run("small", train_overrides={"ns_kind": "triton-ns"}),
    ):
        with pytest.raises(LoudError) as e:
            build()
        assert "ns_kind=" in str(e.value) and "triton" in str(e.value)


@needs_hook
def test_the_knob_round_trips_through_the_config():
    from dormouse.config import load_run

    run = load_run("small", train_overrides={"ns_kind": "triton"})
    assert run.train.ns_kind == "triton"
    assert 'ns_kind = "triton"' in run.snapshot_toml()
    other = load_run("small", train_overrides={"ns_kind": "classic"})
    assert run.diff_keys(other) == ["train.ns_kind"]


# --- tier 2: fp32 parity at the port's shapes (GPU) -------------------------


@pytest.fixture
def highest_precision():
    """TF32 off: `--set tf32=true` makes the CLASSIC arm the less accurate one.

    `loop.py` sets `torch.set_float32_matmul_precision("high" if cfg.tf32 else
    "highest")`, and the b1 recipe runs `tf32=true` — so without this pin a
    parity gate between an ieee Triton kernel and a TF32 cuBLAS chain would
    measure the precision SETTING, not the kernel (`moe_kernels._prec`'s own
    warning). Restored after, so a gate cannot leak it into another test.
    """
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


@CUDA
@pytest.mark.usefixtures("highest_precision")
@pytest.mark.parametrize("shape", B1_SHAPES)
def test_triton_matches_classic_at_the_b1_shapes(shape):
    """The research note's Stage-0 gate (rel ≤ 1e-5), at the shapes b1 runs.

    `[2048, 2048]` is in the list on purpose and it is the COUNTED-fallback case:
    the assertion below splits on which arm ran, so the test cannot pass by
    measuring the classic chain against itself while claiming a kernel.
    """
    g = gauss(shape).cuda()
    classic = zeropower_via_ns(g, 5, "classic")
    c = Counters()
    out = nk.zeropower_triton_or_classic(g, 5, counters=c)
    assert out.shape == g.shape
    d = rel(out, classic)
    ran = "fallback" if c.get("ns_triton_fallback") else "triton"
    print(f"[ns parity] {str(shape):16s} ran={ran:8s} rel={d:.3e}")
    if min(shape[-2:]) > nk.MAX_RANK:
        assert c.get("ns_triton_fallback") == 1 and c.get("ns_triton", 0) == 0, dict(c)
        assert "MAX_RANK" in c.why
        assert torch.equal(out, classic), "the fallback must BE the classic chain"
    else:
        assert c.get("ns_triton") == 1 and c.get("ns_triton_fallback", 0) == 0, dict(c)
        assert d <= TOL_PORT_SHAPES, f"{shape}: triton vs classic rel={d:.3e}"


@CUDA
@pytest.mark.usefixtures("highest_precision")
def test_the_kernel_is_within_the_classic_chains_own_roundoff():
    """A small well-conditioned matrix against the fp64 reference.

    The honest fp32 tier: the kernel is fp32, so it lands where the classic arm
    lands (~1e-6 at these shapes), not at 1e-9 — the tier-1 fp64 gate carries
    that bound. The claim here is that the kernel is no worse than the chain it
    replaces, on the same input.
    """
    g = gauss((64, 64), seed=11).cuda()
    classic = zeropower_via_ns(g, 5, "classic")
    c = Counters()
    out = nk.zeropower_triton_or_classic(g, 5, counters=c)
    ref = fp64_classic(g.cpu())
    assert c.get("ns_triton") == 1
    envelope = 3.0 * max(rel(classic.cpu(), ref), rel(out.cpu(), ref))
    assert rel(out.cpu(), classic.cpu()) <= envelope, (
        f"triton {rel(out.cpu(), ref):.3e} vs classic {rel(classic.cpu(), ref):.3e} against fp64"
    )
    assert rel(out.cpu(), ref) <= 1e-4, rel(out.cpu(), ref)


@CUDA
def test_a_non_contiguous_input_is_a_counted_fallback():
    """The one fallback branch that is unreachable on CPU (it needs a CUDA tensor)."""
    g = gauss((64, 128)).cuda().t()  # a transposed view: [128, 64], non-contiguous
    assert not g.is_contiguous()
    c = Counters()
    out = nk.zeropower_triton_or_classic(g, 5, counters=c)
    assert c.get("ns_triton_fallback") == 1 and "non-contiguous" in c.why
    assert torch.equal(out, zeropower_via_ns(g, 5, "classic"))


@CUDA
def test_a_shape_that_cannot_fit_falls_back_counted(monkeypatch):
    """A genuine over-ceiling tile, as a COUNTED refusal, not a crash.

    `BLOCK_D=256` is the caliper: the tall b1 `(2048, 64)` chain's real launch at
    256 needs 212992 B against this card's 101376 B (measured by
    `JITFunction.warmup` with the production args; the host `ASTSource` table
    under-reports it at 149504 B). Remove the catch and this test dies in a
    traceback; the shipped 64 tile is 65536 B and launches.
    """
    monkeypatch.setattr(nk, "BLOCK_D", 256)
    g = gauss((2048, 64)).cuda()
    c = Counters()
    out = nk.zeropower_triton_or_classic(g, 5, counters=c)
    print(f"[launch refusal] {c.line()} why={c.why}")
    assert c.get("ns_triton_fallback") == 1, dict(c)
    assert c.get("ns_triton", 0) == 0, dict(c)
    assert "OutOfResources" in c.why and "shared memory" in c.why, c.why
    assert "212992" in c.why and "101376" in c.why, c.why
    assert torch.equal(out, zeropower_via_ns(g, 5, "classic"))


def _compiled_shared(rt: int, block_d: int, warps: int) -> int:
    """`metadata.shared` for the real chain at `(rt, block_d)`, host-side."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler import compile as tcompile

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch = major * 10 + minor
    else:
        arch = 120
    sig = {
        "X_ptr": "*fp32",
        "O_ptr": "*fp32",
        "sxs": "i32",
        "sos": "i32",
        "sr": "i32",
        "sd": "i32",
        "R_": "i32",
        "D_": "i32",
        "A": "fp32",
        "B": "fp32",
        "C": "fp32",
        "EPS": "fp32",
        "STEPS": "constexpr",
        "RT": "constexpr",
        "BLOCK_D": "constexpr",
    }
    src = ASTSource(
        nk._ns_chain_kernel,
        signature=sig,
        constexprs=dict(STEPS=5, RT=rt, BLOCK_D=block_d),
    )
    return int(
        tcompile(
            src, target=GPUTarget("cuda", arch, 32), options={"num_warps": warps}
        ).metadata.shared
    )


def test_the_measured_shared_memory_table_is_the_docstring():
    """The docstring's ASTSource table, recompiled: 128 is 83968, 256 is 149504.

    This is table (a) — host compile, no stride specialization. It is the number
    the audit asked for and it is a LOWER BOUND; the real-launch gate below pins
    the orientation-sensitive truth that keeps `BLOCK_D` at 64.
    """
    if not nk.TRITON_AVAILABLE:
        pytest.skip("triton is the ability this table measures")
    print(f"{'RT':>3} {'BLOCK_D':>8} {'warps4':>8} {'warps8':>8} {'fits4':>6}")
    for rt in (16, 32, 64):
        for bd in (64, 128, 256):
            n4 = _compiled_shared(rt, bd, 4)
            n8 = _compiled_shared(rt, bd, 8)
            print(f"{rt:>3} {bd:>8} {n4:>8} {n8:>8} {str(n4 <= nk.SMEM_CEILING_BYTES):>6}")
            assert n4 == nk.shared_bytes(rt, bd), (rt, bd, n4)
            assert n8 >= n4, (rt, bd, n4, n8)
    assert nk.BLOCK_D == 64
    assert nk.shared_bytes(64, 128) == 83968 <= nk.SMEM_CEILING_BYTES
    assert nk.shared_bytes(64, 256) == 149504 > nk.SMEM_CEILING_BYTES


def _launch_shared(shape: tuple[int, ...], block_d: int) -> int:
    """The REAL launcher's `metadata.shared` for `shape` — `warmup`, no launch.

    The production args (strides included) are what Triton specializes on; this
    is table (b) in the module docstring and it is the number a pre-check must
    read.
    """
    from dormouse.train.optim import JORDAN_NS_COEFFS

    x = torch.zeros(shape, dtype=torch.float32, device="cuda")
    out = torch.empty(shape, dtype=torch.float32, device="cuda")
    m, n = shape[-2], shape[-1]
    r, d = min(m, n), max(m, n)
    sr, sd = (x.stride(-1), x.stride(-2)) if m > n else (x.stride(-2), x.stride(-1))
    a, b, c = JORDAN_NS_COEFFS
    compiled = nk._ns_chain_kernel.warmup(
        x,
        out,
        x.stride(0) if x.ndim == 3 else 0,
        out.stride(0) if x.ndim == 3 else 0,
        sr,
        sd,
        r,
        d,
        a,
        b,
        c,
        nk.NS_EPS,
        STEPS=5,
        RT=max(16, nk.triton.next_power_of_2(r)),
        BLOCK_D=block_d,
        num_warps=nk.NUM_WARPS,
        grid=(x.shape[0] if x.ndim == 3 else 1,),
    )
    return int(compiled.metadata.shared)


@CUDA
def test_the_real_launch_table_keeps_block_d_at_64():
    """The launch, not the host compile, decides: the tall 128 cell is 114688.

    `ASTSource` omits stride specialization, so it reports 83968 for 128 and
    would call it a fit. The real tall `(2048, 64)` compile — the b1 Muon group's
    own shape — needs 114688 B and refuses. The wide and batched chains do fit at
    128, which is why the audit's single-shape compile looked green; the shipped
    tile is 64 because 128 also measured slower on every chain.
    """
    print(f"{'shape':>16} {'BD64':>7} {'BD128':>7} {'BD256':>7}")
    for shape in ((2048, 64), (1024, 64, 64), (64, 5461)):
        n64 = _launch_shared(shape, 64)
        n128 = _launch_shared(shape, 128)
        n256 = _launch_shared(shape, 256)
        print(f"{str(shape):>16} {n64:>7} {n128:>7} {n256:>7}")
        assert (
            n64
            == {  # the shipped arm, per orientation
                (2048, 64): 65536,
                (1024, 64, 64): 57344,
                (64, 5461): 53248,
            }[shape]
        )
        assert (
            n128
            == {
                (2048, 64): 114688,  # REFUSED on the b1 tall chain
                (1024, 64, 64): 90112,
                (64, 5461): 83968,
            }[shape]
        )
        assert n256 > nk.SMEM_CEILING_BYTES
    assert nk.BLOCK_D == 64, "the measured tall chain cannot launch 128"


# --- the bench (`scripts/bench_ns_triton.py` runs this) ---------------------


def _dispatches(fn) -> tuple[int, int]:
    """(matmul dispatches, all dispatches) for one call — the classic side's cost.

    A `TorchDispatchMode` rather than `torch.profiler` (the idiom used
    elsewhere in this port): the same count of the ops this code issues, without
    kineto's USDT chatter, and it works on CPU.
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


def launch_delta(shapes: tuple[tuple[int, ...], ...] = tuple((s,) for s in B1_SHAPES)) -> None:
    """classic vs triton: aten dispatches vs kernel launches, per chain.

    The triton column is the op's launch COUNT by construction (one
    `_ns_chain_kernel[(B,)]` call per chain, asserted in
    `test_one_launch_per_chain_in_the_logical_orientation`); the classic column is
    MEASURED here on CPU, where the classic chain's dispatch count is
    device-independent. `scripts/bench_ns_triton.py` adds the GPU wall clock and
    `torch.profiler`'s kernel count when the card is free.
    """
    print("classic dispatches are device-independent; triton is ONE launch per chain")
    for (shape,) in shapes:
        g = gauss(shape)
        n_mm, n_all = _dispatches(lambda g=g: zeropower_via_ns(g, 5, "classic"))
        print(
            f"  {str(shape):18s} classic {n_mm:3d} matmul / {n_all:4d} aten  ->  "
            f"triton 1 launch  (-{n_all - 1})"
        )


if __name__ == "__main__":
    launch_delta()
