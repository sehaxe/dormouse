"""Free-wins bench, research step 2/3 (docs/research/2026-10-15-tsct-super-kernel.md):

(1) the port's operating cap was never measured — `b1.toml` says capacity 1.0
    (cap 32 at b16·s1024·topk2) while the kernel work used cap 64: measure the
    dropped-tokens / time trade directly;
(2) torch.compile(dynamic=True) on the same arm: MEASURED-BLOCKED
    2026-10-15 — dynamo graph-breaks at the dispatch's drop counter
    (`int((~keep).sum())`, experts.py:184, a data-dependent scalar read out
    of the compiled region), then churns the allocator mid-trace; the
    10-min budget never produced a warm graph (log:
    ~/logs/bench_free_wins_compile_20261015.log). The spike's 1.22x stands
    as the WHOLE-MODEL claim; the port's counters must be moved out of the
    hot path (or read under TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1) before
    the arm could compile.

One process, preflight first, one GPU thing at a time. Output: a table.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
from pathlib import Path

import torch

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight, skip  # noqa: E402


def build(
    e: int = 1024, d: int = 2048, r: int = 64, f: int = 5461, tokens: int = 16384, cap: int = 32
):
    from dormouse.config import ModelConfig
    from dormouse.model.experts import SparseMoE

    cfg = ModelConfig(
        d_model=d,
        n_heads=32,
        head_dim=64,
        d_ffn=f,
        vocab=256,
        max_seq_len=1024,
        rank=r,
        shared_rank=192,
        n_experts=e,
        moe_topk=2,
        engram_orders=[2, 3, 4],
        engram_rows=1024,
        engram_dim=32,
        max_iter=2,
        moe_capacity=cap * e / (tokens * 2),
    )
    # The module is NOT cast wholesale: experts steer their own compute dtype
    # (compute_dtype=bf16 via cfg bf16=True) while the router/down stay fp32 —
    # exactly the model's mixed rule (cast to fp32 before every Linear).
    moe = SparseMoE(cfg, device="cuda")
    x = torch.randn(tokens, d, device="cuda", dtype=torch.float32, requires_grad=True)
    return moe, x


def time_step(moe, x, reps: int = 5, *, backprop: bool = False) -> float:
    """Median forward (or fwd+bwd) seconds after 3 warm runs."""

    def once() -> float:
        if x.grad is not None:
            x.grad = None
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        y = moe(x)
        if backprop:
            y.float().square().sum().backward()
            x.grad = None
        else:
            y.float().square().sum().item()  # reads the result: no free dessert
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / 1e3

    for _ in range(3):
        once()
    return statistics.median(once() for _ in range(reps))


def main() -> None:
    print(f"{'cap':>4} {'dropped%':>9} {'fwd ms':>8} {'fwd+bwd ms':>11}")
    for cap in (32, 64, 96, 128):
        moe, x = build(cap=cap)
        with torch.no_grad():
            index = torch.softmax(moe.router(x), -1).topk(2, -1).indices
            _order, _keep, dropped = moe._permute(index)
        pct = 100.0 * dropped / index.numel()
        fwd = time_step(moe, x) * 1e3
        fwbw = time_step(moe, x, backprop=True) * 1e3
        print(f"{cap:>4} {pct:>8.1f} {fwd:>8.1f} {fwbw:>11.1f}", flush=True)
        del moe, x
        torch.cuda.empty_cache()

    moe_c, x_c = build(cap=64)
    cs = torch.compile(moe_c, dynamic=True)
    fwd = time_step(cs, x_c) * 1e3
    fwbw = time_step(cs, x_c, backprop=True) * 1e3
    print(f"{'compiled cap64':>16} {'-':>9} {fwd:>8.1f} {fwbw:>11.1f}")


if __name__ == "__main__":
    busy = subprocess.run(["pgrep", "-ax", "train"], capture_output=True, text=True).stdout.strip()
    why = f"a train process is live:\n{busy}" if busy else preflight("cuda")
    if why is not None:
        raise SystemExit(skip(why))
    main()
