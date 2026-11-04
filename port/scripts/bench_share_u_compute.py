"""Does the expert bank actually USE the shared input basis at compute time?

    CUDA_VISIBLE_DEVICES=0 python scripts/bench_share_u_compute.py --tokens 8192

Why: `share_u` stores ONE basis `U[d, rank]` per site and per-expert
`gate_s/up_s [rank]` + `v [rank, f]`, but the dispatch used to materialize the
basis E-fold and multiply by a PER-EXPERT `[d, rank]` matrix
(`bmm(xg[E, cap, d], gate_u[E, d, rank])`). Storage is shared, compute is not:
every expert re-did the `d -> rank` projection on its own rows.

`moe_shared_compute=true` makes the dispatch read the SAME parameters as ONE
`[d, 2·rank]` matrix and project the TOKENS once:

    lat = x[n, d] @ [gate_u | up_u][d, 2·rank]       ONE GEMM over the tokens
    g   = bank_lat[E, cap, 2·rank][..., :rank]       rank-space GLU as before
    y   = bmm(silu(g)·u, v)                          unchanged

Two assignments of a token share its latent (top-2 projects each token once,
not twice), so the projection MAC drops by roughly the top-k factor; the
remaining cost, the `h @ v` write, is unchanged.

This script builds the REAL `SparseMoE` for the shape (the parameters, `_banks`
and dispatch code the training step runs) and times `_dispatch` for the OFF and
ON shapes on the configured `moe_kernel`. It prints ms and MAC for both, plus
the output parity of the two real paths. Nothing here is synthetic tensor math.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from dormouse.config import ModelConfig  # noqa: E402
from dormouse.instrument import Counters  # noqa: E402
from dormouse.model.experts import SparseMoE  # noqa: E402


def bench(fn, iters: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) * 1000.0 / iters


def routing(tokens: int, topk: int, experts: int, device, dtype) -> tuple[torch.Tensor, ...]:
    """A uniform, deterministic routing: every expert gets the same row count.

    The routing is the fixture; the TENSORS and the dispatch are the model's
    own. `moe_capacity=1.0` then gives `cap = max(k, ceil(n·k/E))` exactly, so
    neither arm drops a row and the two run the same assignment set.
    """
    index = (torch.arange(tokens * topk, device=device) % experts).view(tokens, topk)
    weight = torch.full((tokens, topk), 1.0 / topk, dtype=dtype, device=device)
    return index, weight


def macs(
    tokens: int, topk: int, experts: int, cap: int, d: int, f: int, rank: int, shared: bool
) -> dict:
    """MACs of one routed dispatch, from the shapes it actually ran.

    OFF: the bank is `[E, cap, d]` and BOTH projections repeat over its rows
    (`2 · E·cap · d · rank`), then the write `E·cap · rank · f`.

    ON: the projection reads the `n` TOKENS once through `[d, 2·rank]`
    (`n · d · 2·rank` — the same expression, but with `n` rows and one GEMM),
    then the same `E·cap` write. `E·cap >= n` at any legal routing.
    """
    rows = experts * cap
    proj = tokens * d * 2 * rank if shared else 2 * rows * d * rank
    write = rows * rank * f
    return {"projection": proj, "write": write, "total": proj + write, "bank_rows": rows}


def build(args, kernel: str, shared: bool, device, dtype) -> SparseMoE:
    cfg = ModelConfig(
        d_model=args.d,
        n_heads=8,
        head_dim=args.d // 8,
        d_ffn=args.f,
        vocab=256,
        n_experts=args.experts,
        rank=args.rank,
        moe_topk=args.topk,
        moe_kernel=kernel,
        moe_capacity=1.0,
        share_u=True,
        bf16=dtype == torch.bfloat16,
        moe_shared_compute=shared,
    )
    torch.manual_seed(0)
    return SparseMoE(cfg, device=device)


def one_arm(args, kernel: str, shared: bool, device, dtype) -> tuple[float, dict, torch.Tensor]:
    moe = build(args, kernel, shared, device, dtype)
    index, weight = routing(args.tokens, args.topk, args.experts, device, dtype)
    x = torch.randn(args.tokens, args.d, device=device, dtype=dtype)
    counters = Counters()

    with torch.no_grad():
        out = moe._dispatch(x, weight, index, counters)
        ms = bench(lambda: moe._dispatch(x, weight, index, counters), iters=args.iters)
        ms_forward = bench(lambda: moe(x, counters=counters), iters=args.iters)
        # ONE fresh dispatch after the timing loop: the counter line then reads
        # per-call (a training step accumulates exactly this once per forward).
        counters = Counters()
        moe._dispatch(x, weight, index, counters)
    counters.drain()
    cap = max(args.topk, math.ceil(args.tokens * args.topk / args.experts))
    m = macs(
        args.tokens,
        args.topk,
        args.experts,
        cap,
        args.d,
        args.f,
        args.rank,
        shared,
    )
    print(
        f"  shared={'ON ' if shared else 'OFF'}   {ms:8.2f} ms dispatch / "
        f"{ms_forward:8.2f} ms forward  proj {m['projection'] / 1e9:6.3f} GMAC  "
        f"total {m['total'] / 1e9:6.3f} GMAC"
    )
    print(
        f"               bank {m['bank_rows']} rows  cap {cap}  "
        f"counters shared_calls={counters.get('moe_shared_compute_calls', 0)} "
        f"projection_rows={counters.get('moe_projection_rows', 0)}"
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ms, m, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--d", type=int, default=1536)
    ap.add_argument("--f", type=int, default=4096)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--kernels", default="batched,triton")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device: the triton arm cannot run and the bf16 path is the point")
        return 2
    device = torch.device("cuda")
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    if a.d % 8 != 0:
        print(f"d={a.d} must be divisible by 8 (the placeholder head split)")
        return 2
    print(
        f"shape: tokens {a.tokens} · d {a.d} · f {a.f} · E {a.experts} · r {a.rank} "
        f"· top{a.topk} · {a.dtype}"
    )
    for kernel in (k.strip() for k in a.kernels.split(",") if k.strip()):
        if kernel not in ("batched", "triton"):
            print(f"kernel {kernel!r}: only batched/triton carry the shared-compute arm")
            continue
        print(f"\n-- moe_kernel={kernel} --")
        ms_off, m_off, out_off = one_arm(a, kernel, False, device, dtype)
        ms_on, m_on, out_on = one_arm(a, kernel, True, device, dtype)
        spe = ms_off / ms_on
        mp = m_off["projection"] / 1e9
        mp_on = m_on["projection"] / 1e9
        print(
            f"  speed {spe:5.2f}x (off/on) · projection {mp:.3f} -> {mp_on:.3f} GMAC "
            f"({100 * (1 - mp_on / mp):.0f}% fewer) · total "
            f"{m_off['total'] / 1e9:.3f} -> {m_on['total'] / 1e9:.3f} GMAC"
        )
        rel = (out_on.float() - out_off.float()).norm() / (out_off.float().norm() + 1e-30)
        print(f"  parity (bf16 {a.dtype}, reduction order moved): rel {float(rel):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
