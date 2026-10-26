"""Gates for the Triton grouped-GEMM expert pass (`model/moe_kernels.py`).

Skipped wherever `torch.cuda.is_available()` is false: the kernels are the CUDA
arm, and on CPU the module's own dispatch already falls back COUNTED to the
naive loop. That fallback chain and the config literals are the two tests that
run on any device (they are deliberately not under the `@CUDA` mark — a gate
that is skipped everywhere it could fire is a gate that cannot fail).

What is claimed here, in the order the research doc lists them
(`docs/research/2026-10-15-tsct-super-kernel.md` op 3):

1. **parity** — the unpadded grouped pass computes what the cap-padded bank
   computes, on the same rows, at fp32 (<=1e-5) and bf16 (<=2e-3), the bounds
   the padded-vs-naive gate already uses;
2. **autograd** — the hand-written backward (Triton forward, Triton adjoint
   GEMMs) matches the padded arm's autograd to <=2e-2 bf16 per parameter, so
   the arm is swappable without a loss-curve audit (house rule #8, §4 of the
   study);
3. **ragged bounds** — the `offs` contract holds for empty groups and for
   groups shorter than a tile, which is what "unpadded" has to mean;
4. **the kill-switch** — `ModelConfig.moe_kernel` selects the arm, an unknown
   name is LOUD at config load, and an arm that cannot run is COUNTED with its
   reason (never silent).

The two benches at the bottom are NOT tests: `bench_small` (E=8, correctness
scale) and `bench_b1` (the real shape, the perf gate). Run them with
`uv run --extra gpu python scripts/bench_moe_kernel.py`.
"""

from __future__ import annotations

import statistics
import subprocess

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg

from dormouse.config import MOE_KERNELS, ModelConfig, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model.experts import SparseMoE

try:  # the kernels are an ABILITY: without triton the module is not importable
    import dormouse.model.moe_kernels as mk
except ImportError:  # pragma: no cover - the CPU-only wheel
    mk = None  # type: ignore[assignment]

#: The CUDA gates. NOT a module-wide mark: the config literal and the COUNTED
#: fallback chain are testable on any device, and a gate that is skipped
#: everywhere it could fire is the defect ADR-0019 exists to prevent.
CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or mk is None,
    reason="needs a CUDA device and triton (the grouped GEMMs are the CUDA arm)",
)

DEV = "cuda"


def _moe(**over) -> SparseMoE:
    over.setdefault("rank", 8)
    cfg = tiny_cfg(n_experts=4, moe_topk=2, d_model=64, d_ffn=128, **over)
    return SparseMoE(cfg, device=DEV)


def _routing(moe: SparseMoE, x: torch.Tensor, seed: int = 0):
    """The router's top-k as a FIXED assignment: no graph back into the router.

    Both arms must be handed the same rows and the same gate weights, and the
    backward test calls each arm's autograd twice — a live router graph would
    make the second call fail for a reason that has nothing to do with the arms.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        probs = F.softmax(moe.router(x), dim=-1)
        weight, index = probs.topk(2, dim=-1)
    return index, weight / weight.sum(dim=-1, keepdim=True)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """`max|a-b| / max|b|` — the metric the padded-vs-naive gate already uses."""
    return float((a.float() - b.float()).abs().max()) / float(b.float().abs().max())


@pytest.mark.parametrize("bf16,bound", [(False, 1e-5), (True, 2e-3)])
@CUDA
def test_triton_and_padded_agree_on_the_same_rows(bf16, bound):
    moe = _moe(bf16=bf16)
    x = torch.randn(64, moe.d, device=DEV)
    index, weight = _routing(moe, x)
    order, keep, dropped = moe._permute(index)
    with torch.no_grad():
        padded = moe._padded_dispatch(x, weight, index, order, keep)
        triton = moe._triton_dispatch(x, weight, index, order, keep)
    rel = _rel(triton, padded)
    print(
        f"[kernel parity] bf16={bf16} E={moe.n_experts} d={moe.d} r={moe.experts[0].rank} "
        f"f={moe.f} rows={x.shape[0]} kept={int(keep.sum())} dropped={dropped} "
        f"max_rel={rel:.3e} bound={bound}"
    )
    assert rel <= bound, f"{rel:.3e} exceeds {bound}"


@pytest.mark.parametrize("bf16,bound", [(False, 1e-4), (True, 2e-2)])
@CUDA
def test_triton_backward_matches_the_padded_autograd(bf16, bound):
    """The hand-written adjoint vs the padded arm's autograd, per parameter.

    At `bf16=False` the bound is 1e-4 — that is the math, and it is the number
    that moves if the adjoint is wired wrong. At `bf16=True` the bound is 2e-2:
    the padded arm rounds to bf16 at every step of its own chain (bmm grad,
    GLU grad, gather grad) while this arm keeps fp32 activations between its
    Triton GEMMs, so at bf16 the gate compares two rounding chains, not two
    implementations.
    """
    moe = _moe(bf16=bf16)
    x = torch.randn(64, moe.d, device=DEV)
    index, weight = _routing(moe, x)
    order, keep, _ = moe._permute(index)

    def run(arm) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        moe.zero_grad(set_to_none=True)
        xi = x.detach().clone().requires_grad_(True)
        out = arm(xi)
        out.float().pow(2).mean().backward()
        grads = {n: p.grad.detach().clone() for n, p in moe.experts.named_parameters()}
        assert all(p.grad is not None for p in moe.experts.parameters())
        return grads, xi.grad.detach().clone()

    ref, dx_ref = run(lambda xi: moe._padded_dispatch(xi, weight, index, order, keep))
    got, dx = run(lambda xi: moe._triton_dispatch(xi, weight, index, order, keep))
    per = {n: _rel(got[n], ref[n]) for n in ref}
    worst = max(per.values())
    dx_rel = _rel(dx, dx_ref)
    print(
        f"[kernel grads] bf16={bf16} worst param rel={worst:.3e} dx rel={dx_rel:.3e} "
        f"bound={bound} worst={max(per, key=per.get)}"
    )
    assert worst <= bound, f"parameter grads disagree: {per}"
    assert dx_rel <= bound, f"dx disagrees: {dx_rel:.3e}"


def test_mm_mixed_returns_the_compute_dtype_not_fp32():
    """The premise of the combine crash, pinned on ANY device: `mm_mixed` OUT is `dt`.

    Its docstring used to claim "fp32 out" and the MoE call site repeated the
    claim, which is why a bf16 weight reached `combine` unnoticed. With
    `bf16 = true` the router's logits are bf16, so the softmax and the routed
    weight are bf16 too. This is the half of the bug a CPU can see; the CUDA
    gate below is the consequence.
    """
    from dormouse.model.tsct import mm_mixed

    x, w = torch.randn(4, 8), torch.randn(3, 8)
    assert mm_mixed(x, w, torch.float32).dtype == torch.float32
    assert mm_mixed(x, w, torch.bfloat16).dtype == torch.bfloat16


@CUDA
@pytest.mark.parametrize("capacity", [0.0, 1.0])
def test_the_bf16_module_forward_trains_through_the_fused_combine(capacity, monkeypatch):
    """The REAL path: a bf16 `SparseMoE.forward` through `combine`, fwd + bwd.

    Every other gate on this arm hand-builds its fp32 router weight by calling
    `moe.router(x)` directly (`_routing` above) — an fp32 `nn.Linear` — so the
    dtype the SHIPPING forward produces was never exercised. With `bf16 = true`
    the logits come out of `mm_mixed` in bf16, the softmax and the routed
    `weight` follow, and `combine` asserts `w.dtype == torch.float32`: `9876ab8`
    dropped the explicit widening the pre-fusion adjoint did (`y.to(fp32) * w`),
    so this crashed on b1's FIRST forward, with every arm off, because the bug is
    the dtype and not the arm.

    Both routes are covered (`capacity = 0` is b1's dropless one, `1.0` the
    capped one — both funnel into the same `combine`). The gate spies on
    `_triton_dispatch` to get the EXACT weight/index the module's forward built,
    asserts that weight is bf16 (so the gate cannot silently become the fp32
    gate next door), and compares the real path to the reference loop with the
    SAME bf16-valued weight — the bounds are the file's own bf16 ones.
    """
    torch.manual_seed(0)
    moe = _moe(bf16=True, moe_capacity=capacity, moe_kernel="triton")
    x = torch.randn(64, moe.d, device=DEV)
    dropless = capacity == 0.0

    seen: dict = {}
    real = SparseMoE._triton_dispatch

    def spy(self, xi, weight, index, order, keep, *, dropless=False, fused_bwd=True):
        seen.update(weight=weight, index=index, order=order, keep=keep, dropless=dropless)
        seen["out"] = real(
            self, xi, weight, index, order, keep, dropless=dropless, fused_bwd=fused_bwd
        )
        return seen["out"]

    monkeypatch.setattr(SparseMoE, "_triton_dispatch", spy)
    counters = Counters()
    out = moe(x, counters)  # the module's own forward: router → bf16 probs → combine
    monkeypatch.undo()

    weight, index, order, keep = (seen[k] for k in ("weight", "index", "order", "keep"))
    assert weight.dtype == torch.bfloat16, (
        f"the gate must feed combine the bf16 weight the forward builds, got {weight.dtype}"
    )
    assert seen["dropless"] == dropless
    assert counters["moe_triton_gemm"] == 1 and "moe_triton_fallback" not in counters, counters
    assert torch.isfinite(out).all()

    # forward: the widened-weight path (the cast is the ONLY difference — same
    # rows, same kernels) and the reference loop on the same bf16-valued weight.
    w = weight.detach()
    with torch.no_grad():
        same = moe._triton_dispatch(x, w.float(), index, order, keep, dropless=dropless)
        loop = moe._naive_dispatch(x, w, index, order, keep, Counters())
    routed = seen["out"].detach()
    rel_cast, rel_loop = _rel(routed, same), _rel(routed, loop)

    # fwd + bwd on the real graph: the router's own gradient rides `dw` through
    # the gather, so a `combine` that dropped it would leave the router dead.
    moe.zero_grad(set_to_none=True)
    xg = x.detach().clone().requires_grad_(True)
    moe(xg, Counters()).float().pow(2).mean().backward()
    dead = [n for n, p in moe.named_parameters() if p.grad is None]
    assert not dead, f"the bf16 forward left {dead} without a gradient"
    assert float(moe.router.weight.grad.abs().sum()) > 0, "the router gradient is zero"
    assert xg.grad is not None and torch.isfinite(xg.grad).all()

    def run(arm):
        moe.zero_grad(set_to_none=True)
        xi = x.detach().clone().requires_grad_(True)
        arm(xi).float().pow(2).mean().backward()
        grads = {n: p.grad.detach().clone() for n, p in moe.experts.named_parameters()}
        assert all(p.grad is not None for p in moe.experts.parameters())
        return grads, xi.grad.detach().clone()

    got, dx = run(lambda xi: moe._triton_dispatch(xi, w, index, order, keep, dropless=dropless))
    ref, dx_ref = run(lambda xi: moe._naive_dispatch(xi, w, index, order, keep, Counters()))
    per = {n: _rel(got[n], ref[n]) for n in ref}
    worst, dx_rel = max(per.values()), _rel(dx, dx_ref)
    print(
        f"[bf16 combine] capacity={capacity} w.dtype={weight.dtype} "
        f"fwd rel(cast)={rel_cast:.3e} rel(loop)={rel_loop:.3e} "
        f"grad worst={worst:.3e} dx={dx_rel:.3e} worst={max(per, key=per.get)}"
    )
    assert rel_cast <= 1e-5, f"the cast changed the forward: {rel_cast:.3e}"
    assert rel_loop <= 2e-3, f"the bf16 forward differs from the loop: {rel_loop:.3e}"
    assert worst <= 2e-2, f"the bf16 grads differ from the loop: {per}"
    assert dx_rel <= 2e-2, f"dx differs from the loop: {dx_rel:.3e}"


@CUDA
@pytest.mark.parametrize("arm", ["fused", "split"])
def test_both_glu_forward_arms_are_the_same_function_and_trainable(arm, monkeypatch):
    """`glu_arm`'s two forwards: the padded bank's numbers, and live gradients.

    The shipped forward is E-dependent (`mk.glu_arm` / `mk.FUSED_GLU_MIN_E`), so
    at the tiny E=4 fixture only ONE arm would ever run and the other would go
    unmeasured. Patching the boundary is what forces each — and the SPLIT arm is
    the one that needs saying: its `y` comes from `expert_gemm`, a plain grouped
    GEMM with no autograd of its own, so "the backward still reaches every expert
    parameter" is the thing to check rather than assume (the adjoint is
    `_ExpertGemmGlu`'s in either case and never walks a graph through `v`).
    """
    monkeypatch.setattr(mk, "FUSED_GLU_MIN_E", 1 if arm == "fused" else 1 << 30)
    assert mk.glu_arm(4) == arm
    moe = _moe()
    x = torch.randn(64, moe.d, device=DEV)
    index, weight = _routing(moe, x)
    order, keep, _ = moe._permute(index)
    with torch.no_grad():
        padded = moe._padded_dispatch(x, weight, index, order, keep)
        got = moe._triton_dispatch(x, weight, index, order, keep)
    rel = _rel(got, padded)

    moe.zero_grad(set_to_none=True)
    xi = x.detach().clone().requires_grad_(True)
    out = moe._triton_dispatch(xi, weight, index, order, keep)
    out.float().pow(2).mean().backward()
    dead = [n for n, p in moe.experts.named_parameters() if p.grad is None]
    print(f"[glu arm] {arm} E={moe.n_experts} parity-vs-padded={rel:.3e} dead={dead}")
    assert rel <= 1e-5, f"{arm}: {rel:.3e} vs the padded bank"
    assert not dead, f"{arm}: the forward left {dead} without a gradient"
    assert xi.grad is not None and torch.isfinite(xi.grad).all()


@CUDA
def test_ragged_offsets_hold_for_empty_and_short_groups():
    """`offs` is the whole contract: an empty group is legal, a short one too."""
    torch.manual_seed(0)
    e, d, r, f = 5, 96, 24, 130
    counts = torch.tensor([0, 1, 3, 7, 2], device=DEV, dtype=torch.int32)
    rows_total = int(counts.sum())
    offs = torch.zeros(e + 1, dtype=torch.int32, device=DEV)
    offs[1:] = torch.cumsum(counts, 0)
    seg = torch.repeat_interleave(torch.arange(e, device=DEV), counts.to(torch.int64))
    xg = torch.randn(rows_total, d, device=DEV).to(torch.bfloat16)
    gu = torch.randn(e, d, r, device=DEV).to(torch.bfloat16)
    uu = torch.randn(e, d, r, device=DEV).to(torch.bfloat16)
    gs = torch.randn(e, r, device=DEV).to(torch.bfloat16)
    us = torch.randn(e, r, device=DEV).to(torch.bfloat16)
    v = torch.randn(e, r, f, device=DEV).to(torch.bfloat16)

    got = mk.expert_gemm_glu(xg, gu, uu, gs, us, v, offs, cap=int(counts.max()))
    ref = torch.zeros(rows_total, f, dtype=torch.bfloat16, device=DEV)
    for i in range(e):
        sel = seg == i
        if not bool(sel.any()):
            continue
        g = (xg[sel] @ gu[i]) * gs[i]
        u = (xg[sel] @ uu[i]) * us[i]
        ref[sel] = (F.silu(g) * u) @ v[i]
    rel = _rel(got, ref)
    print(f"[ragged] counts={counts.tolist()} rel={rel:.3e} (group 0 is empty, group 1 is short)")
    assert rel <= 2e-3, f"{rel:.3e}"
    # The contract the kernel reads: offs is a partition of the rows, and group 0
    # really owns none — the parity above is what proves the empty group is not
    # silently filled with a padding row.
    assert int(offs[0]) == 0 and int(offs[-1]) == rows_total and int(counts[0]) == 0

    # op (b): the plain grouped GEMM runs on the same ragged bounds.
    h = torch.randn(rows_total, r, device=DEV).to(torch.bfloat16)
    y = mk.expert_gemm(h, v, offs, cap=int(counts.max()))
    y_ref = torch.zeros(rows_total, f, dtype=torch.bfloat16, device=DEV)
    for i in range(e):
        sel = seg == i
        if bool(sel.any()):
            y_ref[sel] = h[sel] @ v[i]
    assert _rel(y, y_ref) <= 2e-3


@CUDA
def test_dispatch_routes_by_config_and_counts_the_arm():
    x = torch.randn(64, 64, device=DEV)
    moe = _moe(moe_kernel="triton")
    index, weight = _routing(moe, x)
    counters = Counters()
    with torch.no_grad():
        out = moe._dispatch(x, weight, index, counters)
    print(f"[route] {counters.line()}")
    assert counters["moe_triton_gemm"] == 1
    assert "moe_padded_bmm" not in counters and "moe_triton_fallback" not in counters
    assert torch.isfinite(out).all()

    moe_k = _moe(moe_kernel="naive")
    counters = Counters()
    with torch.no_grad():
        moe_k._dispatch(x, weight, index, counters)
    assert counters["moe_naive_dispatch"] == 1 and counters.why.startswith("moe_kernel=naive")

    moe_p = _moe(moe_kernel="padded")
    counters = Counters()
    with torch.no_grad():
        moe_p._dispatch(x, weight, index, counters)
    assert counters["moe_padded_bmm"] == 1


def test_unknown_moe_kernel_is_loud_and_the_literals_are_declared():
    with pytest.raises(LoudError) as e:
        validate(ModelConfig(moe_kernel="triton-please"))
    print(f"[config] {e.value}")
    assert "moe_kernel" in str(e.value) and "padded" in str(e.value)
    assert MOE_KERNELS == ("padded", "triton", "naive")
    for name in MOE_KERNELS:
        validate(ModelConfig(moe_kernel=name))


def test_the_glu_arm_is_e_dependent():
    """`glu_arm` is a pure function of E, and its boundary is a pinned literal.

    Both arms compute the same `y` and share the backward, so the choice is pure
    speed — and its boundary is a SWEPT CONSTANT: the bench measured E=8 and
    E=1024 and nothing in between. Pinning it here is what makes the bench's
    assertion ("the shipped arm is not beaten by >10%") a statement about a
    choice rather than about whatever the constant happens to be.
    """
    if mk is None:
        pytest.skip("no triton in this build")
    assert mk.FUSED_GLU_MIN_E == 64, "the swept boundary is a module literal"
    assert mk.glu_arm(8) == "split" and mk.glu_arm(1024) == "fused"
    assert mk.glu_arm(mk.FUSED_GLU_MIN_E) == "fused"
    assert mk.glu_arm(mk.FUSED_GLU_MIN_E - 1) == "split"
    with pytest.raises(AssertionError):
        mk.glu_arm(0)


def test_dispatch_counts_the_triton_fallback_on_cpu():
    """A CPU tensor with `moe_kernel="triton"` falls back COUNTED, with reasons.

    The whole ability gate, on the only device where it can be exercised without
    a GPU: no triton on the tensor's device -> `moe_triton_fallback`, and the
    padded arm cannot run either -> `moe_naive_dispatch`, both reasons on the
    counter object so a log reader sees the chain.
    """
    moe = SparseMoE(tiny_cfg(n_experts=4, moe_topk=2, rank=8, moe_kernel="triton"), device="cpu")
    x = torch.randn(32, 64)
    index, weight = _routing(moe, x)
    counters = Counters()
    out = moe._dispatch(x, weight, index, counters)
    print(f"[fallback] {counters.line()} why={counters.why}")
    assert counters["moe_triton_fallback"] == 1
    assert counters["moe_naive_dispatch"] == 1
    assert "triton:" in counters.why and "cpu" in counters.why
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------
# the GLU scale loads: masked at every rank, and a NaN tail must not leak
# --------------------------------------------------------------------------

#: Every rank the port can route: r=1..64, power-of-two and not. `RP` is 16 for
#: r < 16 and `next_pow2(r)` above, so these are the two shapes of the bug.
RANKS = [1, 2, 3, 5, 7, 8, 15, 16, 17, 24, 64]


@CUDA
@pytest.mark.parametrize("rank", RANKS)
def test_the_glu_scales_are_masked_and_a_nan_tail_cannot_reach_y(rank, monkeypatch):
    """`gate_s`/`up_s` read exactly `r` floats — the OOB read is gone.

    `_glu_kernel` loads `[E, r]` rows with `tl.arange(0, RP)`; without the mask
    every `RP > r` load reads the next row and, on the last expert, past the
    tensor. The fixture puts `gate_s`/`up_s` in a flat buffer whose tail is NaN.
    The OUTPUT half asserts the padded arm's numbers and finite values; the
    SOURCE half is the falsifier: on Triton 3.7.1 the compiler drops the unused
    lanes (rank 1's PTX has no live load past `r`), so removing the mask happens
    to leave the output finite — measured on the scratch revert, which is why the
    assertion on the shipped source line is here too. A future Triton that keeps
    the lanes live fails the OUTPUT half; the mask is correct either way.
    """
    if mk is None:
        pytest.skip("no triton in this build")
    src = mk._glu_kernel.src
    assert "gs = tl.load(gs_ptr + e * R_ + rr, mask=r_mask, other=0.0)" in src
    assert "us = tl.load(us_ptr + e * R_ + rr, mask=r_mask, other=0.0)" in src
    torch.manual_seed(rank)
    moe = _moe(bf16=True, moe_capacity=1.0, moe_kernel="triton", rank=rank)
    n, k = 64, 2
    index = torch.stack(
        [
            torch.arange(n, device=DEV) % moe.n_experts,
            (torch.arange(n, device=DEV) + 1) % moe.n_experts,
        ],
        dim=1,
    )
    weight = torch.full((n, k), 1.0 / k, dtype=torch.bfloat16, device=DEV)
    x = torch.randn(n, moe.d, device=DEV)
    order, keep, _dropped = moe._permute(index)

    real_banks = moe._banks

    def nan_tail_banks(dt):
        gu, uu, v, gs, us = real_banks(dt)

        def tail(t):
            flat = torch.full((t.numel() + 64,), float("nan"), dtype=t.dtype, device=t.device)
            flat[: t.numel()] = t.reshape(-1)
            return flat[: t.numel()].view(t.shape)

        return gu, uu, v, tail(gs), tail(us)

    monkeypatch.setattr(moe, "_banks", nan_tail_banks)
    with torch.no_grad():
        padded = moe._padded_dispatch(x, weight, index, order, keep)
        got = moe._triton_dispatch(x, weight, index, order, keep)
    finite = bool(torch.isfinite(got).all())
    rel = _rel(got, padded) if finite else float("nan")
    print(f"[glu mask] rank={rank} RP={mk._rp(rank)} finite={finite} rel={rel:.3e}")
    assert finite, f"rank={rank}: a NaN from the unmasked scale tail reached the output"
    assert rel <= 2e-3, f"rank={rank}: {rel:.3e} vs the padded arm"


@CUDA
def test_a_refused_launch_is_counted_and_returns_the_torch_reference(monkeypatch):
    """`OutOfResources` is a COUNTED fallback (with the bytes), never a traceback.

    The proxy refuses on the GLU launch, which is the first kernel in the split
    arm. The public entry point must count `moe_triton_fallback` and return the
    torch reference — the same math, so the fallback is still a training step.
    """
    if mk is None:
        pytest.skip("no triton in this build")
    from triton.runtime.errors import OutOfResources

    from dormouse.instrument import Counters

    class _Refusing:
        def __getitem__(self, grid):
            def run(*a, **k):
                raise OutOfResources(999999, mk.SMEM_CEILING_BYTES, "shared memory")

            return run

    monkeypatch.setattr(mk, "_glu_kernel", _Refusing())
    torch.manual_seed(4)
    e, d, r, f = 3, 32, 8, 48
    counts = torch.tensor([2, 3, 1], dtype=torch.int32, device=DEV)
    offs = torch.zeros(e + 1, dtype=torch.int32, device=DEV)
    offs[1:] = torch.cumsum(counts, 0)
    rows = int(counts.sum())
    xg = torch.randn(rows, d, device=DEV)
    gu = torch.randn(e, d, r, device=DEV)
    uu = torch.randn(e, d, r, device=DEV)
    gs = torch.randn(e, r, device=DEV) * 0.1 + 1.0
    us = torch.randn(e, r, device=DEV) * 0.1 + 1.0
    v = torch.randn(e, r, f, device=DEV)
    counters = Counters()
    got = mk.expert_gemm_glu(xg, gu, uu, gs, us, v, offs, cap=int(counts.max()), counters=counters)
    ref = torch.zeros(rows, f, device=DEV)
    for i in range(e):
        s, t = int(offs[i]), int(offs[i + 1])
        if t > s:
            g = (xg[s:t] @ gu[i]) * gs[i]
            u = (xg[s:t] @ uu[i]) * us[i]
            ref[s:t] = (F.silu(g) * u) @ v[i]
    print(f"[launch refusal] {counters.line()} why={counters.why}")
    assert counters["moe_triton_fallback"] == 1, dict(counters)
    assert "999999" in counters.why and str(mk.SMEM_CEILING_BYTES) in counters.why, counters.why
    assert torch.isfinite(got).all()
    assert _rel(got, ref) <= 1e-5, _rel(got, ref)


@CUDA
def test_a_previously_refused_adjoint_tile_launches():
    """BLOCK_N=256 was refused by the old arithmetic; it launches and computes.

    The pair kernel with `t_wide` at 256 measures 40960 B and fits. This runs the
    REAL adjoint at that tile and compares it to the default-tile reference, so
    the guard's new "allowed" is a launch, not a compile-only claim.
    """
    if mk is None:
        pytest.skip("no triton in this build")
    assert mk.tile_reason(torch.bfloat16, 32, 64, 8, tiles=dict(mk.TILES, BLOCK_N=256)) is None, (
        "the guard must allow the measured 256 tile"
    )
    torch.manual_seed(3)
    e, d, r, f = 5, 24, 8, 32
    counts = torch.tensor([0, 3, 7, 2, 6], dtype=torch.int32, device=DEV)
    rows = int(counts.sum())
    offs = torch.zeros(e + 1, dtype=torch.int32, device=DEV)
    offs[1:] = torch.cumsum(counts, 0)
    seg = torch.repeat_interleave(torch.arange(e, device=DEV), counts.to(torch.int64)).to(
        torch.int32
    )
    xg = torch.randn(rows, d, device=DEV)
    gu = torch.randn(e, d, r, device=DEV) / d**0.5
    uu = torch.randn(e, d, r, device=DEV) / d**0.5
    gs = torch.randn(e, r, device=DEV) * 0.1 + 1.0
    us = torch.randn(e, r, device=DEV) * 0.1 + 1.0
    v = torch.randn(e, r, f, device=DEV) / r**0.5
    gpre, upre = torch.randn(rows, r, device=DEV), torch.randn(rows, r, device=DEV)
    h = torch.randn(rows, r, device=DEV)
    dy = torch.randn(rows, f, device=DEV)
    cap = int(counts.max())
    base = (xg, gu, uu, gs, us, v, offs, cap, gpre, upre, h, dy)
    ref = mk._glu_backward(*base, mk.TILES)
    got = mk._glu_backward_fused(*base, dict(mk.TILES, BLOCK_N=256), seg, mk.BwdWorkspace())
    names = ("dx", "dgate_u", "dup_u", "dgate_s", "dup_s", "dv")
    per = {n: _relz(a, b) for n, a, b in zip(names, got, ref, strict=True)}
    print(f"[256 adjoint] worst rel={max(per.values()):.3e} ({max(per, key=per.get)})")
    assert max(per.values()) <= 1e-5, per


@CUDA
def test_a_refused_backward_launch_falls_back_counted(monkeypatch):
    """The adjoint's own refusal runs the torch reference, counted and equal.

    The forward's tiles fit; `_grouped_nt_pair` is the launch the old arithmetic
    refused. Patch it to raise and assert the backward still produces the same
    gradients (the reference chain, same op order) with `moe_triton_fallback`.
    """
    if mk is None:
        pytest.skip("no triton in this build")
    from triton.runtime.errors import OutOfResources

    from dormouse.instrument import Counters

    class _Refusing:
        def __getitem__(self, grid):
            def run(*a, **k):
                raise OutOfResources(999999, mk.SMEM_CEILING_BYTES, "shared memory")

            return run

    torch.manual_seed(6)
    moe = _moe(bf16=False, moe_capacity=1.0, moe_kernel="triton")
    x = torch.randn(64, moe.d, device=DEV)
    index, weight = _routing(moe, x)
    order, keep, _dropped = moe._permute(index)
    dt = moe._bank_dtype()
    gu, uu, v, gs, us = (t.detach().clone().requires_grad_(True) for t in moe._banks(dt))
    rows = order[keep]
    owner = rows // index.shape[1]
    xg = moe._gather_rows(x, owner, dt).detach().clone().requires_grad_(True)
    counts = torch.bincount(index.reshape(-1)[rows], minlength=moe.n_experts).to(torch.int32)
    offs = torch.zeros(moe.n_experts + 1, dtype=torch.int32, device=DEV)
    offs[1:] = torch.cumsum(counts, 0)
    cap = moe._capacity(*index.shape)

    def run(counters=None):
        tensors = [xg, gu, uu, gs, us, v]
        for t in tensors:
            t.grad = None
        out = mk.expert_gemm_glu(*tensors, offs, cap=cap, counters=counters)
        out.float().pow(2).mean().backward()
        return out.detach(), [t.grad.detach().clone() for t in tensors]

    ref_out, ref_grads = run()
    monkeypatch.setattr(mk, "_grouped_gemm_pair_kernel", _Refusing())
    counters = Counters()
    got_out, got_grads = run(counters)
    print(f"[bwd refusal] {counters.line()} why={counters.why}")
    assert counters["moe_triton_fallback"] == 1, dict(counters)
    assert "999999" in counters.why and str(mk.SMEM_CEILING_BYTES) in counters.why, counters.why
    assert torch.equal(got_out, ref_out), "the forward moved under a backward refusal"
    worst = max(float((a - b).abs().max()) for a, b in zip(got_grads, ref_grads, strict=True))
    assert worst <= 1e-5, worst


# --------------------------------------------------------------------------
# the fp32 tile — the ByteFlow arm must not fall back to the naive loop
# --------------------------------------------------------------------------

#: The ByteFlow channel's loop shape (`/tmp/opencode/profile_bf.py`, 2026-10-21):
#: `medium`'s d=1024 / f=2730 and rank 64, with the b4 x T512 window compressed
#: to K=128 patch positions (the loop's 512 rows, topk 2 -> 1024 routed rows).
BYTEFLOW = dict(d_model=1024, d_ff=2730, rank=64)


def test_fp32_tiles_halve_and_fit_under_the_shared_memory_ceiling():
    """Gate (a): 4-byte elements halve the wide tiles and the measured estimate fits.

    Shared memory is bytes, so `BLOCK_N`/`BLOCK_F` halve at fp32. The measured
    pair at BM=64 BK=64 BN=64 stages=1 is **32768 B** (`metadata.shared`); the old
    arithmetic model said 65536 from a two-stage/two-pair multiplication that no
    compile produced.
    """
    if mk is None:
        pytest.skip("no triton in this build")
    t = mk.tiles_for(torch.float32, **BYTEFLOW)
    assert t["BLOCK_N"] == mk.TILES["BLOCK_N"] // 2 == 64
    assert t["BLOCK_F"] == mk.TILES["BLOCK_F"] // 2 == 64
    for name in ("BLOCK_M", "BLOCK_K", "BLOCK_T"):
        assert t[name] == mk.TILES[name], f"{name} must not move at fp32"
    n = mk.smem_estimate(t, torch.float32, BYTEFLOW["rank"])
    print(f"[fp32 tile] {t} est={n} B <= {mk.SMEM_CEILING_BYTES} B")
    assert n == 32768, "the measured halved pair tile at one stage is 32768 B"
    assert n <= mk.SMEM_CEILING_BYTES
    assert mk.tile_reason(torch.float32, **BYTEFLOW) is None, "fp32 must not fall back"


def test_bf16_tiles_and_their_estimate_do_not_move():
    """Gate (d): the two-byte path keeps every number it shipped.

    Same dict, same estimate, and no explicit `num_stages` at the launch (Triton's
    own default is what the bf16 numbers on record were taken with). The measured
    shared memory at the shipped tile is **24576 B**, not the old model's 98304.
    """
    if mk is None:
        pytest.skip("no triton in this build")
    for dtype in (torch.bfloat16, torch.float16):
        assert mk.tiles_for(dtype, **BYTEFLOW) == mk.TILES
        assert mk._launch_stages(dtype) == {}
    n = mk.smem_estimate(mk.TILES, torch.bfloat16, BYTEFLOW["rank"])
    print(f"[bf16 tile] {mk.TILES} est={n} B <= {mk.SMEM_CEILING_BYTES} B")
    assert n == 24576, "the measured bf16 pair tile is 24576 B"
    assert n <= mk.SMEM_CEILING_BYTES
    assert mk._launch_stages(torch.float32) == {"num_stages": 1}


def _compiled_shared(fn, signature, constexprs: dict, warps: int) -> int:
    """`metadata.shared` for one host-side compile (ptxas; no device, no launch)."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler import compile as tcompile

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch = major * 10 + minor
    else:
        arch = 120  # this box's card; a compile-only gate does not need the device
    src = ASTSource(fn, signature=signature, constexprs=constexprs)
    compiled = tcompile(src, target=GPUTarget("cuda", arch, 32), options={"num_warps": warps})
    return int(compiled.metadata.shared)


def test_a_tile_the_compiled_device_can_hold_is_not_refused():
    """The measured table, and the tiles the old arithmetic wrongly refused.

    Reproduces the calibration the docstring names: compile the REAL pair kernel
    with `ASTSource`, read `metadata.shared`, and assert (a) the estimate equals
    it, (b) the tile fits, and (c) `tile_reason` does not refuse it. On revert to
    `2 stages * 2 pairs` this is red for BN=256 and 512 — the two tiles the
    guard used to reject while the compiled device held them.
    """
    if mk is None:
        pytest.skip("no triton in this build")
    sig = _sig(
        "a1_ptr:*bf16 w1_ptr:*bf16 a2_ptr:*bf16 w2_ptr:*bf16 o_ptr:*bf16 offs_ptr:*i32 "
        "N:i32 K:i32 sam:i32 sak:i32 swe:i32 swk:i32 swn:i32 som:i32 son:i32 "
        "BLOCK_M:constexpr BLOCK_N:constexpr BLOCK_K:constexpr PREC:constexpr"
    )
    print(f"{'tile':>16} {'measured':>9} {'estimate':>9} {'fits':>5} {'allowed':>8}")
    for bn in (64, 128, 256, 512):
        tiles = dict(mk.TILES, BLOCK_N=bn)
        measured = _compiled_shared(
            mk._grouped_gemm_pair_kernel,
            sig,
            dict(BLOCK_M=64, BLOCK_N=bn, BLOCK_K=64, PREC="tf32"),
            8,
        )
        est = mk.smem_estimate(tiles, torch.bfloat16, BYTEFLOW["rank"])
        allowed = mk.tile_reason(torch.bfloat16, **BYTEFLOW, tiles={"BLOCK_N": bn}) is None
        print(
            f"{'BN=' + str(bn):>16} {measured:>9} {est:>9} "
            f"{str(measured <= mk.SMEM_CEILING_BYTES):>5} {str(allowed):>8}"
        )
        assert measured <= est, f"BN={bn}: estimate {est} is BELOW the measured {measured}"
        if bn >= 128:
            # At BN>=128 the pair is the family's widest consumer; at 64 the GLU
            # is (24576 B), so the pair is smaller than the max there.
            assert measured == est, f"BN={bn}: estimate {est} != measured {measured}"
        assert measured <= mk.SMEM_CEILING_BYTES, f"BN={bn} does not fit this compile"
        assert allowed, f"BN={bn} fits ({measured} B) but the guard refused it"
    # At the shipped tile the GLU is the family max: measure it and pin 24576.
    sig_glu = _sig(
        "x_ptr:*bf16 gu_ptr:*bf16 uu_ptr:*bf16 gs_ptr:*bf16 us_ptr:*bf16 v_ptr:*bf16 "
        "y_ptr:*bf16 offs_ptr:*i32 gpre_ptr:*fp32 upre_ptr:*fp32 h_ptr:*bf16 "
        "D:i32 R_:i32 F_:i32 BLOCK_M:constexpr BLOCK_K:constexpr BLOCK_F:constexpr "
        "RP:constexpr PREC:constexpr SAVE:constexpr SPLIT:constexpr"
    )
    glu = _compiled_shared(
        mk._glu_kernel,
        sig_glu,
        dict(BLOCK_M=64, BLOCK_K=64, BLOCK_F=128, RP=64, PREC="tf32", SAVE=True, SPLIT=False),
        4,
    )
    print(f"{'GLU shipped':>16} {glu:>9} {mk.smem_estimate(mk.TILES, torch.bfloat16, 64):>9}")
    assert glu == 24576 == mk.smem_estimate(mk.TILES, torch.bfloat16, 64), glu
    # The 512 tile FITS at bf16 (73728 B); the fp32 override of the same tile does
    # not (147456 B), which is the genuine refusal the next gate exercises.
    assert mk.tile_reason(torch.bfloat16, **BYTEFLOW, tiles={"BLOCK_N": 512}) is None


def test_a_tile_that_cannot_fit_is_refused_before_launch_with_its_numbers():
    """Gate (3): the refusal names the numbers, and it fires before any launch.

    The genuine over-ceiling tile is fp32 `BLOCK_N=512` (147456 B measured-model);
    the fp32 two-stage pair at the shipped BN=64 is 65536 B and fits. This is a
    CALIBRATION point, not the tiles the shipped path uses — the gate below proves
    the bf16 256/512 tiles it used to refuse really do fit.
    """
    if mk is None:
        pytest.skip("no triton in this build")
    t32 = mk.tiles_for(torch.float32, **BYTEFLOW)
    two_stage = mk.smem_estimate(t32, torch.float32, BYTEFLOW["rank"], stages=2)
    assert two_stage == 65536 and two_stage <= mk.SMEM_CEILING_BYTES
    reason = mk.tile_reason(torch.float32, **BYTEFLOW, tiles=dict(mk.TILES, BLOCK_N=512))
    print(f"[tile refusal] {reason}")
    assert reason == "fp32 tile BLOCK_N=512 needs 147456 bytes > 101376"
    with pytest.raises(AssertionError, match=r"fp32 tile BLOCK_N=512 needs 147456 bytes > 101376"):
        mk._tiles({"BLOCK_N": 512}, torch.float32, **BYTEFLOW)


def test_the_tile_fallback_is_counted_with_the_numbers(monkeypatch):
    """Gate (c): when the tile does not fit, the dispatch COUNTS it and says so.

    `tile_reason` returning a reason must reach `moe_triton_fallback` and the
    counter's `why` unchanged — a fallback a reader of the log cannot see is the
    ADR-0019 defect this gate exists for. The reason is the same string the
    refusal above prints.
    """
    if mk is None:
        pytest.skip("no triton in this build")
    reason = "fp32 tile BLOCK_N=512 needs 147456 bytes > 101376"
    monkeypatch.setattr(mk, "tile_reason", lambda *a, **k: reason)
    moe = SparseMoE(tiny_cfg(n_experts=4, moe_topk=2, rank=8, moe_kernel="triton"), device="cpu")
    x = torch.randn(32, 64)
    index, weight = _routing(moe, x)
    counters = Counters()
    with torch.no_grad():
        out = moe._dispatch(x, weight, index, counters)
    print(f"[tile fallback] {counters.line()} why={counters.why}")
    assert counters["moe_triton_fallback"] == 1
    assert reason in counters.why, counters.why
    assert counters["moe_naive_dispatch"] == 1  # CPU: the padded bank cannot run either
    assert torch.isfinite(out).all()


@CUDA
def test_fp32_triton_runs_the_halved_tile_and_matches_the_padded_bank():
    """Gate (b): fp32 Triton == fp32 padded at the small shape, rel <= 1e-5.

    This is the existing fp32 parity shape with the NEW tile underneath: if the
    dtype-aware tile or the one-stage launch changed the arithmetic, this is the
    gate that fails. It also proves the launch is LEGAL at fp32 (the old tile
    died in `OutOfResources`), which a CPU test cannot.
    """
    moe = _moe(bf16=False)
    t = mk.tiles_for(torch.float32, moe.d, moe.f, moe.experts[0].rank)
    assert t["BLOCK_N"] == 64 and mk._launch_stages(torch.float32) == {"num_stages": 1}
    x = torch.randn(64, moe.d, device=DEV)
    index, weight = _routing(moe, x)
    order, keep, dropped = moe._permute(index)
    with torch.no_grad():
        padded = moe._padded_dispatch(x, weight, index, order, keep)
        triton = moe._triton_dispatch(x, weight, index, order, keep)
    rel = _rel(triton, padded)
    print(
        f"[fp32 parity] E={moe.n_experts} d={moe.d} f={moe.f} r={moe.experts[0].rank} "
        f"BLOCK_N={t['BLOCK_N']} stages=1 rel={rel:.3e}"
    )
    assert rel <= 1e-5, f"{rel:.3e} exceeds 1e-5"


# --------------------------------------------------------------------------
# the FUSED backward — parity against the previous adjoint, per parameter group
# --------------------------------------------------------------------------


def _relz(a: torch.Tensor, b: torch.Tensor) -> float:
    """`_rel` with the degenerate case DEFINED: 0 against 0 is 0, not NaN.

    A capacity'd routing leaves experts with no rows, and their gradients are
    legitimately all-zero on both arms — `0/0` would read as a failure.
    """
    scale = float(b.float().abs().max())
    d = float((a.float() - b.float()).abs().max())
    return d / scale if scale > 0 else d


def _run_arm(moe, x, weight, index, order, keep, *, fused: bool, dropless: bool):
    """`(expert grads, dx, dprobs)` for one arm — the five groups the gate names.

    `dprobs` is the gradient of the ROUTER WEIGHT (`weight.grad`), which is what
    the fused combine backward produces in registers; `dv`/`dgate_u`/`dup_u` are
    the expert banks' (`v`, `gate_u`, `up_u`) and `dx` is the input's.
    """
    moe.zero_grad(set_to_none=True)
    xi = x.detach().clone().requires_grad_(True)
    wi = weight.detach().clone().requires_grad_(True)
    out = moe._triton_dispatch(xi, wi, index, order, keep, dropless=dropless, fused_bwd=fused)
    out.float().pow(2).mean().backward()
    grads = {n: p.grad.detach().clone() for n, p in moe.experts.named_parameters()}
    assert all(p.grad is not None for p in moe.experts.parameters()), "a dead expert param"
    assert xi.grad is not None and wi.grad is not None
    return grads, xi.grad.detach().clone(), wi.grad.detach().clone()


#: `(capacity, dropless)` — the three routings the fused backward has to serve:
#: the tight cap (the b1 default), a cap that really DROPS rows (so the combine
#: kernel's `-1` sentinel is on the tested path), and the dropless static bound.
ROUTINGS = [("cap1.0", 1.0, False), ("cap0.25", 0.25, False), ("dropless", 0.0, True)]


@CUDA
@pytest.mark.parametrize("bf16,bound", [(False, 1e-6), (True, 2e-2)])
@pytest.mark.parametrize("label,cap,dropless", ROUTINGS, ids=[r[0] for r in ROUTINGS])
def test_the_fused_backward_matches_the_current_one(bf16, bound, label, cap, dropless):
    """The fused adjoint == the previous one, forward and on all five groups.

    The bounds are the study's: **1e-6 at fp32** (the differences are summation
    ORDER only — `dv`/`dgate_u`/`dup_u` run the same `_wgrad` on bit-identical
    inputs, `dx` accumulates both dots into one fp32 accumulator instead of
    adding two rounded results, and `dprobs` is a tree sum in the kernel against
    torch's `sum(-1)`), and **2e-2 at bf16**, the bound every other gradient gate
    in this file uses. At bf16 `dx` also differs by ONE rounding instead of two
    (the fused accumulator rounds once), which is 2^-8 of the value and is why
    the bf16 bound is the loose one.
    """
    moe = _moe(bf16=bf16, moe_capacity=cap)
    x = torch.randn(64, moe.d, device=DEV)
    index, weight = _routing(moe, x)
    order, keep, dropped = moe._permute(index)
    with torch.no_grad():
        fwd_new = moe._triton_dispatch(
            x, weight, index, order, keep, dropless=dropless, fused_bwd=True
        )
        fwd_cur = moe._triton_dispatch(
            x, weight, index, order, keep, dropless=dropless, fused_bwd=False
        )
    fwd_rel = _relz(fwd_new, fwd_cur)
    new, dx_new, dp_new = _run_arm(
        moe, x, weight, index, order, keep, fused=True, dropless=dropless
    )
    cur, dx_cur, dp_cur = _run_arm(
        moe, x, weight, index, order, keep, fused=False, dropless=dropless
    )
    per = {n: _relz(new[n], cur[n]) for n in cur}
    worst = max(per.values())
    dx_rel, dp_rel = _relz(dx_new, dx_cur), _relz(dp_new, dp_cur)
    print(
        f"[fused bwd] {label} bf16={bf16} dropped={int(dropped)}/{index.numel()} "
        f"fwd={fwd_rel:.3e} dv/dgate_u/dup_u={worst:.3e} ({max(per, key=per.get)}) "
        f"dx={dx_rel:.3e} dprobs={dp_rel:.3e} bound={bound}"
    )
    assert fwd_rel <= bound, f"the forward moved: {fwd_rel:.3e} > {bound}"
    assert worst <= bound, f"expert parameter grads disagree: {per}"
    assert dx_rel <= bound, f"dx disagrees: {dx_rel:.3e}"
    assert dp_rel <= bound, f"dprobs disagrees: {dp_rel:.3e}"


@CUDA
def test_the_fused_backward_is_the_one_the_module_runs_by_default():
    """`_triton_dispatch`'s default IS the fused arm — the counter says so.

    The whole point of the lane is that the shipped path changes; a default that
    silently kept the old adjoint would make every number below a measurement of
    the wrong program (and the module's own docstring names `fused_bwd=False` as
    the bench's control, not the default).
    """
    moe = _moe(moe_kernel="triton", moe_capacity=0)
    x = torch.randn(64, moe.d, device=DEV)
    index, weight = _routing(moe, x)
    counters = Counters()
    out = moe._dispatch(x, weight, index, counters)
    print(f"[fused default] {counters.line()} why={getattr(counters, 'why', None)}")
    assert counters["moe_triton_gemm"] == 1 and "moe_triton_fallback" not in counters
    assert torch.isfinite(out).all()
    # ... and the workspace it uses is the module's own, created on first use.
    assert isinstance(moe._bwd_workspace(), mk.BwdWorkspace)
    assert moe._bwd_workspace() is moe._bwd_workspace(), "one workspace per module object"


def test_the_combine_forward_is_the_scatter_it_replaced():
    """The fused combine's FORWARD is the three lines it replaced, bit for bit.

    `_Combine.forward` is pure torch, so this gate needs no card — and it is the
    one that says the new graph did not change the forward's numbers (the
    backward is a kernel and is gated on CUDA; a forward that moved would make
    every CUDA parity number a measurement of a different function).
    """
    if mk is None:
        pytest.skip("no triton in this build")
    torch.manual_seed(5)
    n, k, f = 7, 2, 6
    owner = torch.tensor([0, 0, 1, 2, 3, 4, 5, 6, 6, 2, 1, 3, 4, 5])  # 14 kept rows
    rows = owner.numel()
    y = torch.randn(rows, f, dtype=torch.bfloat16)
    w = torch.rand(rows, dtype=torch.float32)
    rowmap = torch.full((n, k), -1, dtype=torch.int32)
    rowmap[0, 0], rowmap[0, 1] = 0, 1
    rowmap[1, 0], rowmap[1, 1] = 2, 10
    rowmap[2, 0], rowmap[2, 1] = 3, 9
    rowmap[3, 0], rowmap[3, 1] = 4, 11
    rowmap[4, 0], rowmap[4, 1] = 5, 12
    rowmap[5, 0], rowmap[5, 1] = 6, 13
    rowmap[6, 0], rowmap[6, 1] = 7, 8

    with torch.no_grad():
        got = mk._Combine.apply(y, w, owner, rowmap, n, mk.BwdWorkspace(), mk.TILES, None)
    ref = torch.zeros(n, f, dtype=torch.float32)
    ref.index_add_(0, owner, y.to(torch.float32) * w.unsqueeze(-1))
    print(f"[combine fwd] rows={rows} n={n} equal={torch.equal(got, ref)}")
    assert torch.equal(got, ref), "the combine's forward must not have moved"


def test_the_row_map_is_the_inverse_of_the_gather():
    """The `-1` int32 map the combine kernel reads is the permute's own inverse.

    Built exactly as `_triton_dispatch` builds it — index arithmetic, so no card
    is needed — and checked against the gather it must invert: every KEPT
    (token, slot) pair points at its gathered row, every DROPPED one is `-1`.
    The kernel's per-token loop reads `rowmap[token*K + k]`, so a transposed map
    (`k*N + token`, OLMo-core's own layout) would read the wrong row for every
    token but the first; this is the gate that fails if that convention drifts.
    """
    torch.manual_seed(7)
    moe = SparseMoE(tiny_cfg(n_experts=4, moe_topk=2, rank=8, moe_capacity=0.25), device="cpu")
    x = torch.randn(32, moe.d)
    index, _weight = _routing(moe, x)
    n, k = index.shape
    order, keep, dropped = moe._permute(index)
    rows = order[keep]
    rowmap = torch.full((n * k,), -1, dtype=torch.int32)
    rowmap[rows] = torch.arange(rows.numel(), dtype=torch.int32)
    assert int(dropped) > 0, "the fixture must DROP rows — that is the `-1` path"
    for j in range(rows.numel()):
        flat = int(rows[j])
        assert int(rowmap[flat]) == j and int(rowmap.view(n, k)[flat // k, flat % k]) == j
    for p in order[~keep].tolist():  # `keep` masks SORTED positions, rowmap is by flat id
        assert int(rowmap[p]) == -1, f"a dropped assignment ({p}) must be -1"
    print(f"[row map] n={n} k={k} kept={rows.numel()} dropped={int(dropped)} -1s checked")


def test_the_fused_backward_refuses_only_an_out_of_range_topk():
    """The fused path's ONE refusal, and it is a name, not a crash (CPU gate).

    Everything else the fused path needs is `_triton_reason`'s gate (CUDA, triton);
    what is left is the combine kernel's UNROLLED top-k loop. The literal is
    pinned because `experts._fused_bwd_reason` and the kernel's `tl.static_range`
    must agree on it — a drift is a config that silently loses its fast backward.
    """
    if mk is None:
        pytest.skip("no triton in this build: the ability gate is the reason itself")
    moe = SparseMoE(tiny_cfg(n_experts=4, moe_topk=2, rank=8), device="cpu")
    assert moe._fused_bwd_reason(1) is None and moe._fused_bwd_reason(2) is None
    why = moe._fused_bwd_reason(mk.FUSED_TOPK_MAX + 1)
    print(f"[fused gate] {why}")
    assert f"topk={mk.FUSED_TOPK_MAX + 1}" in why and str(mk.FUSED_TOPK_MAX) in why
    assert mk.FUSED_TOPK_MAX == 32, "the unrolled top-k tile is a module literal"
    assert mk._topk_tile(1) == 2 and mk._topk_tile(3) == 4 and mk._topk_tile(2) == 2


def test_the_bwd_workspace_is_grow_only_and_reuses_its_buffers():
    """The caller-owned scratch is pure torch, so this gate needs no card.

    Grow-only is the whole contract: a smaller call must hand back the SAME
    storage (that is what "no per-call allocation" means), a bigger one must
    grow, a dtype/device move must reallocate, and two names must never alias.
    """
    if mk is None:
        pytest.skip("no triton in this build")
    ws = mk.BwdWorkspace()
    dev = torch.device("cpu")
    a = ws.slot("s", (4, 8), torch.float32, dev)
    a.fill_(1.0)
    b = ws.slot("s", (2, 3), torch.float32, dev)
    assert b.data_ptr() == a.data_ptr(), "a smaller call must reuse the buffer"
    assert b.shape == (2, 3) and float(b.abs().sum()) == 6.0, "the same storage, a view"
    c = ws.slot("s", (5, 9), torch.float32, dev)
    assert c.data_ptr() != a.data_ptr() and c.numel() == 45, "a bigger call must grow"
    d = ws.slot("s", (5, 9), torch.float16, dev)
    assert d.dtype == torch.float16 and d.data_ptr() != c.data_ptr(), "a dtype move grows"
    e = ws.slot("t", (4, 8), torch.float32, dev)
    assert e.data_ptr() != c.data_ptr() and e.data_ptr() != a.data_ptr(), "one buffer per name"


def _fake_grouped_nt(a, w, offs, cap, n, k, tiles, out_dtype, out=None):
    """CPU stand-in for `_grouped_nt`: the same product per group, in fp64.

    fp64 so that the two adjoints are handed BIT-IDENTICAL tensors and the
    comparison below can be `torch.equal` instead of a bound — a fake that
    rounded would hide exactly the class of defect this gate is for.
    """
    res = torch.zeros(a.shape[0], n, dtype=out_dtype)
    for i in range(w.shape[0]):
        s, t = int(offs[i]), int(offs[i + 1])
        if t > s:
            res[s:t] = (a[s:t].double() @ w[i].double().T).to(out_dtype)
    if out is not None:
        out.copy_(res)
        return out
    return res


def _fake_wgrad(a, b, offs, e, cap, k, n, tiles, out_dtype):
    res = torch.zeros(e, k, n, dtype=out_dtype)
    for i in range(e):
        s, t = int(offs[i]), int(offs[i + 1])
        if t > s:
            res[i] = (a[s:t].double().T @ b[s:t].double()).to(out_dtype)
    return res


def _fake_pair(a1, w1, a2, w2, offs, cap, n, k, tiles, out_dtype):
    """`_grouped_nt_pair` on CPU = the reference's two GEMMs plus `add_`."""
    return _fake_grouped_nt(a1, w1, offs, cap, n, k, tiles, out_dtype) + _fake_grouped_nt(
        a2, w2, offs, cap, n, k, tiles, out_dtype
    )


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("give_seg", [True, False], ids=["seg-int32", "seg-rebuilt"])
def test_the_fused_adjoint_is_the_reference_op_for_op_on_cpu(monkeypatch, dt, give_seg):
    """The elementwise chain and the workspace discipline, WITHOUT a card.

    The three Triton GEMMs are replaced by fp64 stand-ins, which is what makes
    this gate sharp rather than merely available: both adjoints are then handed
    the same tensors and the comparison is BITWISE, so a reordered multiply, a
    lost `.to(dtype)`, a workspace slot read after it was overwritten, a swapped
    return, or an int32 row map that disagrees with `_segments` all fail HERE.

    What it does not verify is the two new KERNELS' arithmetic — the AOT gate
    says they build and the CUDA parity gate says they compute the same function
    — and this docstring says so on purpose.
    """
    if mk is None:
        pytest.skip("no triton in this build")
    monkeypatch.setattr(mk, "_grouped_nt", _fake_grouped_nt)
    monkeypatch.setattr(mk, "_wgrad", _fake_wgrad)
    monkeypatch.setattr(mk, "_grouped_nt_pair", _fake_pair)

    torch.manual_seed(3)
    e, d, r, f = 5, 24, 8, 32
    counts = torch.tensor([0, 3, 7, 2, 6], dtype=torch.int32)  # group 0 is EMPTY
    rows = int(counts.sum())
    offs = torch.zeros(e + 1, dtype=torch.int32)
    offs[1:] = torch.cumsum(counts, 0)
    seg = torch.repeat_interleave(torch.arange(e, dtype=torch.int32), counts.to(torch.int64))

    def cast(t):
        return t.to(dt)

    xg = cast(torch.randn(rows, d))
    gu = cast(torch.randn(e, d, r) / d**0.5)
    uu = cast(torch.randn(e, d, r) / d**0.5)
    gs = cast(torch.randn(e, r) * 0.1 + 1.0)
    us = cast(torch.randn(e, r) * 0.1 + 1.0)
    v = cast(torch.randn(e, r, f) / r**0.5)
    gpre, upre = torch.randn(rows, r), torch.randn(rows, r)
    h = cast(torch.randn(rows, r))
    dy = cast(torch.randn(rows, f))
    cap = int(counts.max())
    args = (xg, gu, uu, gs, us, v, offs, cap, gpre, upre, h, dy, mk.TILES)

    ref = mk._glu_backward(*args)
    got = mk._glu_backward_fused(*args, seg if give_seg else None, mk.BwdWorkspace())
    names = ("dx", "dgate_u", "dup_u", "dgate_s", "dup_s", "dv")
    # `strict=True`: the adjoint's six outputs, in the order `_ExpertGemmGlu`
    # returns them — a swapped pair must fail here, not pass as a tolerance.
    bad = [n for n, a, b in zip(names, got, ref, strict=True) if not torch.equal(a, b)]
    worst = max(float((a.float() - b.float()).abs().max()) for a, b in zip(got, ref, strict=True))
    print(
        f"[fused bwd op-for-op] dt={dt} seg={'caller int32' if give_seg else 'rebuilt'} "
        f"rows={rows} counts={counts.tolist()} mismatched={bad} max|d|={worst:.3e}"
    )
    assert not bad, f"the fused adjoint differs from the reference on {bad}"
    assert worst == 0.0, "the fused adjoint must be bit-identical on the same GEMMs"
    # The reference really did visit every expert: an empty group is a legal call,
    # and a fused path that silently dropped it would still pass a bitwise test.
    assert int(counts[0]) == 0 and int(offs[-1]) == rows


def _sig(pairs: str) -> dict[str, str]:
    """`"x:*fp32 n:i32"` -> an `ASTSource` signature — one line instead of twelve."""
    out: dict[str, str] = {}
    for item in pairs.split():
        name, _, ty = item.partition(":")
        out[name] = ty
    return out


def test_the_fused_kernels_compile_for_this_arch():
    """A COMPILE gate that needs no card — the only way to fail a kernel here.

    The card is held by a live run for most of this project's hours, so a gate
    that needs a device is a gate that never runs. `triton.compile` with an
    explicit `GPUTarget` lowers a kernel to PTX/cubin on the HOST (ptxas; no
    device memory, no launch), which catches every frontend failure a new kernel
    can have: an unknown op, a bad broadcast, a `tl.arange` that is not a power
    of two, a mask of the wrong rank. It is NOT a parity gate — the numeric gates
    above are — and it does not replace one; it is what makes a kernel that
    cannot even be built fail HERE instead of on the first training step.
    """
    if mk is None:
        pytest.skip("no triton in this build")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler import compile as tcompile

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch = major * 10 + minor
    else:
        arch = 120  # this box's card; a compile-only gate does not need the device
    target = GPUTarget("cuda", arch, 32)

    sig_combine = _sig(
        "dout_ptr:*fp32 y_ptr:*bf16 w_ptr:*fp32 rowmap_ptr:*i32 dy_ptr:*bf16 dw_ptr:*fp32 "
        "N:i32 F_:i32 K:constexpr TOP_K:constexpr BLOCK_T:constexpr BLOCK_F:constexpr"
    )
    sig_pair = _sig(
        "a1_ptr:*bf16 w1_ptr:*bf16 a2_ptr:*bf16 w2_ptr:*bf16 o_ptr:*bf16 offs_ptr:*i32 "
        "N:i32 K:i32 sam:i32 sak:i32 swe:i32 swk:i32 swn:i32 som:i32 son:i32 "
        "BLOCK_M:constexpr BLOCK_N:constexpr BLOCK_K:constexpr PREC:constexpr"
    )
    t = mk.TILES
    comb = {"BLOCK_T": t["BLOCK_T"], "BLOCK_F": t["BLOCK_F"]}
    pair = {"BLOCK_M": t["BLOCK_M"], "BLOCK_N": t["BLOCK_N"], "BLOCK_K": t["BLOCK_K"]}
    cases = [
        # our top-k, and the two padded tiles the `if k < K` static branch skips
        ("combine K=2", mk._combine_bwd_kernel, sig_combine, dict(comb, K=2, TOP_K=2), 4),
        ("combine K=1", mk._combine_bwd_kernel, sig_combine, dict(comb, K=1, TOP_K=2), 4),
        ("combine K=3", mk._combine_bwd_kernel, sig_combine, dict(comb, K=3, TOP_K=4), 4),
        ("fused dx bf16", mk._grouped_gemm_pair_kernel, sig_pair, dict(pair, PREC="tf32"), 8),
        ("fused dx fp32", mk._grouped_gemm_pair_kernel, sig_pair, dict(pair, PREC="ieee"), 8),
    ]
    built = []
    for label, fn, sig, consts, warps in cases:
        src = ASTSource(fn, signature=sig, constexprs=consts)
        k = tcompile(src, target=target, options={"num_warps": warps})
        assert "cubin" in k.asm, f"{label}: no cubin"
        built.append(f"{label}={len(k.asm['cubin'])}B")
    print(f"[kernel compile] arch=sm_{arch} " + " ".join(built))


# --------------------------------------------------------------------------
# benches — NOT tests: the perf gate, run by scripts/bench_moe_kernel.py
# --------------------------------------------------------------------------


def preflight() -> None:
    """AGENTS §1.5: no other train process, card < 1 GiB held."""
    busy = subprocess.run(["pgrep", "-ax", "train"], capture_output=True, text=True).stdout.strip()
    assert not busy, f"a train process is alive:\n{busy}\nONE GPU process at a time (AGENTS §1.5)"
    free, total = torch.cuda.mem_get_info()
    assert free > (1 << 30), f"only {free >> 20} MiB free of {total >> 20} — the card is held"
    print(f"[preflight] no train process; {free >> 20} MiB free", flush=True)


def _timed(fn, reps: int, warm: int = 2, x=None) -> tuple[float, float, float]:
    """`(median ms, min ms, peak GiB)` over `reps`, each rep its own alloc peak."""

    def once():
        if x is not None and x.grad is not None:
            x.grad = None
        fn()

    for _ in range(warm):
        once()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(reps):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        once()
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
    return statistics.median(times), min(times), torch.cuda.max_memory_allocated() / 2**30


def _bench_arms(e: int, d: int, r: int, f: int, tokens: int, cap: float, reps: int = 8) -> None:
    """Interleaved padded vs current-triton vs new-triton at one shape.

    All three arms see the same rows, the same weights and the same gate, so the
    table is a comparison of the passes and not of two routings. `current-triton`
    is `fused_bwd=False` (two `dx` GEMMs, a `repeat_interleave` segment rebuild,
    autograd's combine backward), `new-triton` is the default fused adjoint —
    the forward is the SAME code on both, which is why their fwd rows should tie
    and their fwd+bwd rows should not.

    Interleaved on purpose: this card's dispatch timings drift a few ms between
    runs of one binary, so measuring one arm to completion and then the other
    attributes the drift to the arm (the padded arm measured 39-45 ms for the
    same call in consecutive runs of the same script).
    """
    torch.manual_seed(11)
    cfg = ModelConfig(
        d_model=d,
        n_heads=4,
        head_dim=d // 4,
        d_ffn=f,
        rank=r,
        n_experts=e,
        moe_topk=2,
        moe_capacity=cap,
        bf16=True,
    )
    moe = SparseMoE(cfg, device=DEV)
    x = torch.randn(tokens, d, device=DEV)
    with torch.no_grad():
        probs = F.softmax(moe.router(x), dim=-1)
        weight, index = probs.topk(2, dim=-1)
        weight = weight / weight.sum(-1, keepdim=True)
    order, keep, dropped = moe._permute(index)
    print(
        f"\n[{e}x d{d} r{r} f{f}] cap={moe._capacity(*index.shape)} rows={tokens} "
        f"kept={int(keep.sum())} dropped={dropped} ({100 * dropped / index.numel():.1f}%) "
        f"tiles={mk.TILES}",
        flush=True,
    )
    with torch.no_grad():
        padded = moe._padded_dispatch(x, weight, index, order, keep)
        cur = moe._triton_dispatch(x, weight, index, order, keep, fused_bwd=False)
        new = moe._triton_dispatch(x, weight, index, order, keep, fused_bwd=True)
    print(
        f"[parity] padded-vs-new-triton max_rel {_rel(new, padded):.3e}; "
        f"current-vs-new triton max_rel {_rel(new, cur):.3e}",
        flush=True,
    )
    del padded, cur, new
    torch.cuda.empty_cache()

    xp, xct, xnt = (x.clone().requires_grad_(True) for _ in range(3))

    def fwd_p():
        with torch.no_grad():
            moe._padded_dispatch(x, weight, index, order, keep)

    def fwd_ct():
        with torch.no_grad():
            moe._triton_dispatch(x, weight, index, order, keep, fused_bwd=False)

    def fwd_nt():
        with torch.no_grad():
            moe._triton_dispatch(x, weight, index, order, keep, fused_bwd=True)

    def both_p():
        moe._padded_dispatch(xp, weight, index, order, keep).float().pow(2).sum().backward()

    def both_ct():
        out = moe._triton_dispatch(xct, weight, index, order, keep, fused_bwd=False)
        out.float().pow(2).sum().backward()

    def both_nt():
        out = moe._triton_dispatch(xnt, weight, index, order, keep, fused_bwd=True)
        out.float().pow(2).sum().backward()

    runs = {
        "padded fwd": (fwd_p, None),
        "current-triton fwd": (fwd_ct, None),
        "new-triton fwd": (fwd_nt, None),
        "padded fwd+bwd": (both_p, xp),
        "current-triton fwd+bwd": (both_ct, xct),
        "new-triton fwd+bwd": (both_nt, xnt),
    }
    acc: dict[str, list] = {k: [] for k in runs}
    for _ in range(reps):
        for name, (fn, xg) in runs.items():
            acc[name].append(_timed(fn, reps=1, warm=0, x=xg))
    print(f"{'arm':>16} {'median ms':>10} {'min ms':>9} {'peak GiB':>9}")
    for name, rows in acc.items():
        med = statistics.median(r[0] for r in rows)
        lo = min(r[1] for r in rows)
        peak = max(r[2] for r in rows)
        print(f"{name:>16} {med:>10.3f} {lo:>9.3f} {peak:>9.2f}")
    _bench_variants(moe, x, index, order, keep, moe._capacity(*index.shape), reps=max(3, reps // 2))
    print("reference (2026-10-15, pre-bank-cache): padded fwd 50.6 ms, fwd+bwd ~124 ms")


def _bench_variants(moe, x, index, order, keep, cap, reps=6) -> None:
    """The op's own kernels, chrome removed: fused (h in registers) vs split.

    The study's step-2 note asks for both and says ship the faster; this is the
    measurement that picks, and the pick is E-DEPENDENT (`mk.glu_arm` /
    `mk.FUSED_GLU_MIN_E`): measured 2026-10-17, the split pair is 2x faster at
    E=8 (0.184 vs 0.375 ms) and the fused kernel 1.35x faster at b1's E=1024
    (4.699 vs 6.335). So the gate asserts on the arm that SHIPS at this E — a
    reading that wins on the arm the trainer does not run is not a gate.
    """
    n, k = index.shape
    dt = moe._bank_dtype()
    rows = order[keep]
    owner = rows // k
    counts = torch.bincount(index.reshape(-1)[rows], minlength=moe.n_experts).to(torch.int32)
    offs = torch.zeros(moe.n_experts + 1, dtype=torch.int32, device=x.device)
    offs[1:] = torch.cumsum(counts, 0)
    gate_u, up_u, v, gate_s, up_s = moe._banks(dt)
    xg = x[owner].to(dt)

    def split():
        _, (_, _, h) = mk._glu_fused_or_head(
            xg, gate_u, up_u, gate_s, up_s, v, offs, cap, mk.TILES, split=True, save=True
        )
        return mk.expert_gemm(h, v, offs, cap=cap)

    chosen = mk.glu_arm(moe.n_experts)
    with torch.no_grad():
        fused = _timed(
            lambda: mk.expert_gemm_glu(xg, gate_u, up_u, gate_s, up_s, v, offs, cap=cap), reps, 1
        )[0]
        sp = _timed(split, reps, 1)[0]
    mine, theirs = (fused, sp) if chosen == "fused" else (sp, fused)
    print(
        f"[kernel] fused {fused:.3f} ms | split (head + @v) {sp:.3f} ms | shipped for "
        f"E={moe.n_experts}: {chosen} (FUSED_GLU_MIN_E={mk.FUSED_GLU_MIN_E})"
    )
    # The arm that ships may only be beaten by the other one with a real margin:
    # a 2 % reading at a small shape is launch noise, and this card's dispatch
    # timings drift a few ms between runs of one binary.
    assert mine <= 1.1 * theirs, (
        f"the shipped {chosen} arm ({mine:.3f} ms) loses to the other one ({theirs:.3f} ms) "
        f"by >10% at E={moe.n_experts}: move mk.FUSED_GLU_MIN_E"
    )


def bench_small() -> None:
    """E=8, one-tile-ish shapes: the ragged path at 1/128th of b1."""
    preflight()
    _bench_arms(e=8, d=256, r=32, f=512, tokens=2048, cap=2.0, reps=5)


def bench_b1() -> None:
    """The b1 expert pass: E=1024, d=2048, r=64, f=5461, cap 64, 16384 rows bf16."""
    preflight()
    _bench_arms(e=1024, d=2048, r=64, f=5461, tokens=16 * 1024, cap=2.0, reps=6)


if __name__ == "__main__":
    bench_small()
    bench_b1()
