"""Time one loop iteration's PARTS at a fixed token count, on the loaded model.

    python scripts/part_bench.py --tokens 8192

The step is ~107 ms per iteration for 8192 tokens on a 45 TFLOP/s card, which is
~150x off the arithmetic floor. This says which module eats it, by timing the
same shapes the iteration builds, one module at a time, with the CUDA event API
(no host sync inside the loop, one sync at the end of each arm).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402


def bench(fn, iters: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--d", type=int, default=1536)
    ap.add_argument("--f", type=int, default=4096)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--topk", type=int, default=2)
    a = ap.parse_args()

    t, d, f, e, r = a.tokens, a.d, a.f, a.experts, a.rank
    dev, dt = "cuda", torch.bfloat16
    x = torch.randn(t, d, device=dev, dtype=dt)

    # the dense projections one iteration does (latent attention + out)
    wqkv = torch.randn(4 * 256, d, device=dev, dtype=dt)
    wout = torch.randn(d, 256, device=dev, dtype=dt)

    # one expert's gate/up through the shared rank-r basis + the shared down
    u = torch.randn(r, d, device=dev, dtype=dt)
    gu = torch.randn(2 * f, r, device=dev, dtype=dt)
    down = torch.randn(d, f, device=dev, dtype=dt)

    print(f"tokens {t} · d {d} · f {f} · E {e} · rank {r} · top{a.topk}")

    def dense():
        # 4 latent projections (d -> 256) then out (256 -> d), the shapes the
        # block really multiplies
        q = x @ wqkv[:256].T            # [t, 256]  (the 4 latent projections)
        k = x @ wqkv[256:512].T
        return (k + q) @ wout.T         # [t, d]    (out_proj back to d)

    def expert_one():
        # what ONE expert does with the rows it owns
        rows = max(1, t * a.topk // e)
        h = torch.randn(rows, d, device=dev, dtype=dt)
        g = h @ u.T                      # [rows, r]
        y = g @ gu.T                     # [rows, 2f]
        v = y[..., :f] * torch.nn.functional.silu(y[..., f:])
        return v @ down.T

    def expert_all_loop():
        for _ in range(e):
            expert_one()

    def expert_all_stacked():
        # the same arithmetic as ONE grouped GEMM over all experts' rows
        rows = t * a.topk
        h = torch.randn(rows, d, device=dev, dtype=dt)
        g = h @ u.T
        y = g @ gu.T
        v = y[..., :f] * torch.nn.functional.silu(y[..., f:])
        return v @ down.T

    # per-expert weights, ONE batched GEMM over padded row blocks (what a
    # grouped/batched expert kernel does when the shapes are homogeneous)
    rows = max(1, t * a.topk // e)
    UB = torch.randn(e, r, d, device=dev, dtype=dt)
    GUB = torch.randn(e, 2 * f, r, device=dev, dtype=dt)
    DOWNB = torch.randn(e, d, f, device=dev, dtype=dt)
    xb = torch.randn(e, rows, d, device=dev, dtype=dt)

    def expert_bmm():
        g = torch.bmm(xb, UB.transpose(1, 2))          # [E, rows, r]
        y = torch.bmm(g, GUB.transpose(1, 2))          # [E, rows, 2f]
        v = y[..., :f] * torch.nn.functional.silu(y[..., f:])
        return torch.bmm(v, DOWNB.transpose(1, 2))     # [E, rows, d]

    def norm_epilogue():
        for _ in range(6):  # the per-iteration norms/mHC/AttnRes chain, roughly
            x2 = x * x
            x3 = torch.rsqrt(x2.mean(-1, keepdim=True) + 1e-6) * x
            x4 = x3 + 0.5 * x
        return x4

    arms = {
        "dense attention projections (4×latent + out)": dense,
        f"one expert on its rows ({max(1, t * a.topk // e)} rows)": expert_one,
        f"ALL {e} experts, python loop": expert_all_loop,
        f"ALL experts' rows, ONE grouped pass": expert_all_stacked,
        f"per-expert weights, ONE bmm ({rows} rows/expert)": expert_bmm,
        "norm/mHC-style elementwise chain (6 ops)": norm_epilogue,
    }
    for name, fn in arms.items():
        ms = bench(fn, iters=3 if "loop" in name else 5)
        print(f"  {ms:9.2f} ms  {name}")

    # what one iteration of the real model costs, for the comparison
    budget = 8192 / 25600 * 1000 / 3  # ms/iteration implied by the measured B/s
    print(f"\n  {budget:9.2f} ms  ← what ONE real iteration currently costs (from 25.6k B/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
