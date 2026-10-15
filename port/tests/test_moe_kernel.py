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
    cfg = tiny_cfg(n_experts=4, moe_topk=2, rank=8, d_model=64, d_ffn=128, **over)
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
    """Interleaved padded-vs-triton at one shape — both arms see the same rows.

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
        triton = moe._triton_dispatch(x, weight, index, order, keep)
    print(f"[parity] padded-vs-triton max_rel {_rel(triton, padded):.3e}", flush=True)
    del padded, triton
    torch.cuda.empty_cache()

    xp, xt = x.clone().requires_grad_(True), x.clone().requires_grad_(True)

    def fwd_p():
        with torch.no_grad():
            moe._padded_dispatch(x, weight, index, order, keep)

    def fwd_t():
        with torch.no_grad():
            moe._triton_dispatch(x, weight, index, order, keep)

    def both_p():
        moe._padded_dispatch(xp, weight, index, order, keep).float().pow(2).sum().backward()

    def both_t():
        moe._triton_dispatch(xt, weight, index, order, keep).float().pow(2).sum().backward()

    runs = {
        "padded fwd": (fwd_p, None),
        "triton fwd": (fwd_t, None),
        "padded fwd+bwd": (both_p, xp),
        "triton fwd+bwd": (both_t, xt),
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
    measurement that picks. `mk.TILES` is what ships, so both variants carry the
    same tile choices and the difference is only where `h` lives.
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

    with torch.no_grad():
        fused = _timed(
            lambda: mk.expert_gemm_glu(xg, gate_u, up_u, gate_s, up_s, v, offs, cap=cap), reps, 1
        )[0]
        sp = _timed(split, reps, 1)[0]
    print(
        f"[kernel] fused {fused:.3f} ms | split (head + @v) {sp:.3f} ms | "
        f"shipped: fused (TILES) — at small shapes the two tie, `h` is 1/128th of "
        f"the work and both are launch-bound"
    )
    # The split arm may only win the SHIPPED choice back with a real margin: a
    # 2 % reading at a small shape is launch noise, and `h` in registers is the
    # whole point of the fused kernel.
    assert sp >= 0.9 * fused, f"split ({sp:.3f}) beats the shipped fused ({fused:.3f}) by >10%"


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
