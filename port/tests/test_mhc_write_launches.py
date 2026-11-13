"""Why `mhc_write` cost 1 395 `add` launches / 10 steps, and the arm that drops them.

The 2026-11-05 kernel profile (`docs/reviews/2026-11-05-kernel-profile.md`) put
`mhc_write` at 23.2 ms/step, and its largest kernel family was
`at::native::CUDAFunctor_add` — **1 395 launches / 10 steps**. That family is
`mhc.sinkhorn_knopp`'s 20-iteration Python loop (`mhc.py`, `sinkhorn_knopp`):
each normalisation is `m = m - torch.logsumexp(m, dim)`, whose reduction ends
in `max + log(sum(exp))` (an `add`), and the caller's subtract is another add
(CUDA lowers `sub` to `CUDAFunctor_add` with `alpha=-1`). That is ~120
add-family launches per call, ×3 iterations/step on an eager mHC write.

**The fused kernel is not that loop.** `fused_residual._write_kernel` runs all
20 iterations in registers (`for it in tl.static_range(ITERS)`), one launch per
iteration — which is why an iteration-count arm would buy nothing there. The
profiled run had `fused_residual = false` (`port/configs/m400b.toml:58`), so the
23.2 ms is PyTorch's eager chain, not the Triton kernel.

`mhc_sinkhorn="log_softmax"` is the arm this file gates: `log_softmax(dim)` IS
`m - logsumexp(m)`, so the per-normalisation subtract leaves the graph and the
add family goes to zero. It is NOT bit-identical (fp32 roundoff only: measured
≤2.4e-7 on the probability, ≤1.8e-7 on the gradient), so it is an arm and
`"logsumexp"` stays the default; OFF is pinned first.

The counts here are CPU ATen dispatches from `torch.profiler`, i.e. the
CPU-side shape of the launch graph: a composite like `logsumexp` expands to the
same max/sub/exp/sum/log/add sequence the CUDA dispatcher launches, so the
RATIO is the launch ratio. An allocation (`aten::empty`) is not a launch, so
the absolute number is an upper bound; `scripts/bench_mhc_write.py` prints it
for one iteration at the m400 shape.
"""

from __future__ import annotations

import collections
import hashlib

import pytest
import torch
from conftest import tiny_cfg, tiny_model
from torch.profiler import ProfilerActivity, profile

from dormouse.config import ModelConfig, parse_preset, validate
from dormouse.instrument import LoudError
from dormouse.model import fused_residual as fr
from dormouse.model.mhc import SINKHORN_IMPLS, MHCMix, sinkhorn_knopp

#: Ops the dispatcher records that allocate or only recompute a view. They are
#: not kernel launches on CUDA (`empty` is the allocator), so the counter skips
#: them; `fill_`/`copy_`/`to` are real kernels and stay in.
_VIEWS = frozenset(
    {
        "aten::as_strided",
        "aten::view",
        "aten::_unsafe_view",
        "aten::resolve_conj",
        "aten::detach",
        "aten::alias",
        "aten::lift_fresh",
        "aten::expand",
        "aten::permute",
        "aten::transpose",
        "aten::t",
        "aten::select",
        "aten::slice",
        "aten::unsqueeze",
        "aten::squeeze",
        "aten::empty",
        "aten::empty_like",
        "aten::empty_strided",
        "aten::empty_permuted",
    }
)

#: The family the kernel profile named as the region's largest kernel:
#: `CUDAFunctor_add` serves `add`, `add_` AND `sub` (alpha=-1).
_ADD_FAMILY = ("aten::sub", "aten::add", "aten::add_")

#: The shipped arithmetic on a fixed fixture — computed on `main` BEFORE this
#: arm existed. The default path is unchanged, so a moved pin is a real change.
SINKHORN_PIN = "f5a4d808412734b16b0fbfb8e164ccea78db8067ee43faf3bbdd7592b3ac90c8"
MHC_PIN = "3a70e3c5635e433748c30eea7a8e9812998f08cdd8b12c8479bf6636cc40dd7c"


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


def _ops(fn) -> collections.Counter:
    """Top-level ATen dispatches of one call, views/allocations excluded."""
    fn()  # warm the lazy caches; the count is shape-stable
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
    return collections.Counter(
        e.name for e in prof.events() if e.name.startswith("aten::") and e.name not in _VIEWS
    )


def _launches(fn) -> int:
    return sum(_ops(fn).values())


def _add_family(c: collections.Counter) -> int:
    return sum(c[k] for k in _ADD_FAMILY)


def _fixture(arm: str = "logsumexp"):
    """`MHCMix(64, 2)` + one iteration's write inputs, deterministic."""
    torch.manual_seed(0)
    mhc = MHCMix(64, 2, sinkhorn=arm)
    torch.manual_seed(1)
    xs = torch.randn(2, 8, 2, 64) * 0.5
    y = torch.randn(2, 8, 64) * 0.5
    return mhc, xs, y


def _region(mhc, xs, y):
    """`mhc_write`'s two record_function spans: `pre_mix` + `forward`."""
    return lambda: (mhc.pre_mix(xs), mhc(xs, y))


# --------------------------------------------------------------------------- #
# 1. OFF bit-identical: the pins were taken on main before the arm existed
# --------------------------------------------------------------------------- #
@pytest.mark.pins
def test_off_is_the_shipped_bits_and_the_default() -> None:
    assert ModelConfig().mhc_sinkhorn == "logsumexp"
    assert set(SINKHORN_IMPLS) == {"logsumexp", "log_softmax"}

    torch.manual_seed(0)
    x = torch.randn(2, 4, 2, 2)
    assert _sha(sinkhorn_knopp(x)) == SINKHORN_PIN

    mhc, xs, y = _fixture()
    assert mhc.sinkhorn == "logsumexp"
    assert _sha(mhc(xs, y)) == MHC_PIN
    validate(ModelConfig(use_mhc=True, mhc_sinkhorn="logsumexp"))  # silent


def test_the_wiring_hands_the_arm_to_the_mixer() -> None:
    model = tiny_model(use_mhc=True, mhc_sinkhorn="log_softmax")
    assert model.loop.mhc.sinkhorn == "log_softmax"
    assert tiny_cfg(use_mhc=True).mhc_sinkhorn == "logsumexp"


# --------------------------------------------------------------------------- #
# 2. the relaunch family is the loop, and the arm removes it
# --------------------------------------------------------------------------- #
def test_the_profiled_add_family_is_the_sinkhorn_subtract(capsys) -> None:
    """Pin the mechanism: 120 add-family ops per Sinkhorn call, 0 under the arm."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 2, 2)
    base = _ops(lambda: sinkhorn_knopp(x))
    fast = _ops(lambda: sinkhorn_knopp(x, impl="log_softmax"))
    print(
        f"\n  sinkhorn ATen ops: logsumexp {sum(base.values())} "
        f"(add family {_add_family(base)}) -> log_softmax {sum(fast.values())} "
        f"(add family {_add_family(fast)})"
    )
    # 20 iterations x 2 normalisations x (logsumexp's `add` + the caller's `sub`)
    assert _add_family(base) == 120, dict(base)
    assert _add_family(fast) == 0, dict(fast)
    # 761 -> 81 non-view dispatches (9.4x): the reduction's other five ops per
    # normalisation still dispatch, they just stop being five separate kernels.
    assert sum(base.values()) >= 9 * sum(fast.values())


def test_the_region_drops_by_4x_and_its_relaunch_family_by_10x(capsys) -> None:
    mhc, xs, y = _fixture()
    fast_mhc, _, _ = _fixture("log_softmax")
    base = _ops(_region(mhc, xs, y))
    fast = _ops(_region(fast_mhc, xs, y))
    n_base, n_fast = sum(base.values()), sum(fast.values())
    a_base, a_fast = _add_family(base), _add_family(fast)
    print(
        f"\n  mhc_write (pre_mix+forward) ATen ops {n_base} -> {n_fast} "
        f"({n_base / n_fast:.1f}x), add family {a_base} -> {a_fast} "
        f"({a_base / max(a_fast, 1):.1f}x)"
    )
    # The task's 10x: the launches the profile named as the region's biggest
    # kernel family. The rest of the region is the two phi GEMMs, the lane norm
    # and the write adds — never in the loop, so it does not fall 10x.
    assert a_base >= 10 * max(a_fast, 1)
    assert n_base >= 4 * n_fast


def test_the_mutation_that_reinstates_the_loop_turns_the_gate_red(monkeypatch) -> None:
    import dormouse.model.mhc as mhc_mod

    real = mhc_mod.sinkhorn_knopp
    monkeypatch.setattr(
        mhc_mod,
        "sinkhorn_knopp",
        lambda logits, iters=20, impl="logsumexp": real(logits, iters, "logsumexp"),
    )
    mhc, xs, y = _fixture("log_softmax")
    base = _ops(_region(*_fixture()))
    mutated = _ops(_region(mhc, xs, y))
    a_base, a_mut = _add_family(base), _add_family(mutated)
    print(f"\n  mutated add family {a_mut} vs base {a_base}")
    with pytest.raises(AssertionError):
        assert a_base >= 10 * max(a_mut, 1)


# --------------------------------------------------------------------------- #
# 3. parity where the order changes: fp32 roundoff only
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", [0.005, 0.1, 1.5, 10.0])
def test_the_arm_is_within_roundoff_of_the_shipped_sinkhorn(scale: float) -> None:
    torch.manual_seed(0)
    x = (torch.randn(2, 16, 2, 2) * scale).requires_grad_(True)
    w = torch.randn(2, 16, 2, 2)
    ref = sinkhorn_knopp(x)
    (ref * w).sum().backward()
    g_ref = x.grad.clone()
    x.grad = None
    arm = sinkhorn_knopp(x, impl="log_softmax")
    (arm * w).sum().backward()
    g_arm = x.grad.clone()
    dv = (ref - arm).abs().max().item()
    dg = (g_ref - g_arm).abs().max().item()
    print(f"\n  scale {scale}: max|Δvalue| {dv:.2e}  max|Δgrad| {dg:.2e}")
    assert dv <= 1e-6 * max(1.0, ref.abs().max().item())
    assert dg <= 1e-6 * max(1.0, g_ref.abs().max().item())


def test_the_whole_write_parity_is_roundoff() -> None:
    mhc, xs, y = _fixture()
    fast = _fixture("log_softmax")[0]
    ref = mhc(xs, y)
    arm = fast(xs, y)
    d = (ref - arm).abs().max().item()
    print(f"\n  MHCMix.forward max|Δ| {d:.2e} on scale {ref.abs().max():.3f}")
    assert d <= 1e-6 * max(1.0, ref.abs().max().item())


# --------------------------------------------------------------------------- #
# 4. the Sinkhorn answer: in-register in the kernel, a loop only in eager
# --------------------------------------------------------------------------- #
def test_the_fused_kernel_does_not_run_the_python_loop() -> None:
    src = open(fr.__file__, encoding="utf-8").read()
    assert "for it in tl.static_range(ITERS)" in src, (
        "the fused write's Sinkhorn is no longer a static_range: it would be a "
        "launch per iteration again"
    )
    assert "sinkhorn_knopp" not in src, (
        "fused_residual must not call the eager Sinkhorn: the kernel IS the in-register normaliser"
    )


# --------------------------------------------------------------------------- #
# 5. refusals and no host syncs
# --------------------------------------------------------------------------- #
def test_refusals_carry_the_escape() -> None:
    with pytest.raises(LoudError) as e:
        validate(ModelConfig(use_mhc=True, mhc_sinkhorn="nope"))
    assert "Escape:" in str(e.value) and "mhc_sinkhorn" in str(e.value)

    with pytest.raises(LoudError) as e:
        validate(ModelConfig(use_mhc=False, mhc_sinkhorn="log_softmax"))
    assert "Escape:" in str(e.value) and "inert" in str(e.value)

    with pytest.raises(LoudError):
        parse_preset("use_mhc = true\nmhc_sinkhorn = 'nope'\n")


def test_the_arm_never_reads_a_device_scalar(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise AssertionError("host sync in the Sinkhorn arm")

    torch.manual_seed(0)
    x = (torch.randn(2, 4, 2, 2)).requires_grad_(True)
    sinkhorn_knopp(x, impl="log_softmax").sum().backward()
    x.grad = None
    mhc, xs, y = _fixture("log_softmax")
    for name in ("item", "tolist", "__int__", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, boom, raising=True)
    mhc(xs, y).sum().backward()
