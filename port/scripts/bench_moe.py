"""b1-shape MoE dispatch: the padded bank vs the naive loop, fwd-only.

    uv run --extra gpu python scripts/bench_moe.py

The shape is the readiness plan's b1 row — E=1024, d=2048, r=64, f=5461, cap 64,
b16×s1024 = 16384 rows, bf16 — the run that decides the dispatch at scale. These
are the PORT's own tensors through the PORT's own `_dispatch`, banks stacked per
forward exactly as `_padded_dispatch` does it; the spike's numbers were taken on
pre-stacked banks and are the story, not this measurement.

fwd-only (`torch.no_grad`), median of 5 warm iterations, peak allocated VRAM per
arm. Machine discipline first (AGENTS §1.5): no `train` process, card < 1 GiB.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from dormouse.config import ModelConfig
from dormouse.instrument import Counters
from dormouse.model.experts import SparseMoE

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight, skip  # noqa: E402

E, D, R, F_, N_TOK, DEV = 1024, 2048, 64, 5461, 16 * 1024, "cuda"
REPS, WARM = 5, 2


def timed(fn) -> tuple[float, float, float]:
    """`(median ms, min ms, peak GiB)` over warm `REPS`, each its own alloc peak."""
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    times, peaks = [], []
    for _ in range(REPS):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(1e3 * (time.perf_counter() - t0))
        peaks.append(torch.cuda.max_memory_allocated() / 2**30)
    return statistics.median(times), min(times), max(peaks)


def main() -> int:
    busy = subprocess.run(["pgrep", "-ax", "train"], capture_output=True, text=True).stdout.strip()
    why = f"a train process is live:\n{busy}" if busy else preflight("cuda")
    if why is None:
        # This bench's own ceiling is 1 GiB, tighter than the lane guard's 2.0:
        # a card already holding a second model is contended however idle it looks.
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if smi and int(smi.split()[0]) >= 1024:
            why = f"the card already holds {smi.split()[0]} MiB (this bench's ceiling is 1024)"
    if why is not None:
        return skip(why)
    torch.manual_seed(11)
    cfg = ModelConfig(
        d_model=D,
        n_heads=16,
        head_dim=128,
        d_ffn=F_,
        rank=R,
        n_experts=E,
        moe_topk=2,
        moe_capacity=2.0,
        bf16=True,
    )
    moe = SparseMoE(cfg, device=DEV)
    x = torch.randn(N_TOK, D, device=DEV)
    with torch.no_grad():
        probs = F.softmax(moe.router(x), dim=-1)
        weight, index = probs.topk(2, dim=-1)
        weight = weight / weight.sum(-1, keepdim=True)
    order, keep, dropped = moe._permute(index)
    print(
        f"[shape] E={E} d={D} r={R} f={F_} cap={moe._capacity(*index.shape)} rows={N_TOK} "
        f"bf16 assignments={index.numel()} dropped={dropped} "
        f"({100 * dropped / index.numel():.1f}%)",
        flush=True,
    )
    arms = {
        "padded": lambda: moe._padded_dispatch(x, weight, index, order, keep),
        "naive": lambda: moe._naive_dispatch(x, weight, index, order, keep, Counters()),
    }
    out: dict[str, tuple[float, float, float, float]] = {}
    with torch.no_grad():
        ref = arms["padded"]()
        scale = float(ref.abs().max())
        for name, fn in arms.items():
            err = float((fn() - ref).abs().max()) / scale
            med, best, peak = timed(fn)
            out[name] = (med, best, peak, err)
            print(
                f"[{name}] median {med:.3f} ms  min {best:.3f} ms  "
                f"peak {peak:.2f} GiB  {N_TOK / (med / 1e3):.0f} tok/s  "
                f"max_rel_vs_padded {err:.2e}",
                flush=True,
            )
    print("\n| arm | median ms | min ms | peak GiB | tok/s |")
    print("|---|---|---|---|---|")
    for name, (med, best, peak, _) in out.items():
        print(f"| {name} | {med:.3f} | {best:.3f} | {peak:.2f} | {N_TOK / (med / 1e3):.0f} |")
    ratio = out["padded"][0] / out["naive"][0]
    print(f"\npadded/naive = {ratio:.3f}x ({'padded wins' if ratio < 1 else 'NAIVE WINS'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
