#!/usr/bin/env -S uv run --script
"""Standard vs fixed cross-entropy: forward+backward ms and CPU bytes.

The `ce_fixed` arm may not be claimed "free" without the number, and a
per-step loss that costs more than the step's logits GEMM would be a
different decision entirely. This measures the LOSS ONLY (the same logits
tensor, one forward + one backward each), min of `--reps`, on the shape the
brief names: `[8*512, 256]` logits and byte targets.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python port/scripts/bench_ce.py

`MB` is ALLOCATION CHURN: the sum of the positive `self_cpu_memory_usage` the
profiler attributes to the region's ops, in MiB. It is not peak RSS (a Python
process cannot attribute that to a region) and it counts every `[N, V]` fp32
temporary the forward and the backward build, so it is the number that says
whether the fix doubles the loss's memory footprint. Both arms are measured the
same way.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from dormouse.instrument import Counters
from dormouse.losses import fixed_cross_entropy


def _one(step, logits, targets) -> None:
    x = logits.clone().requires_grad_(True)
    loss = step(x, targets)
    loss.backward()


def _time(step, logits, targets, reps: int, warm: int) -> tuple[float, float]:
    for _ in range(warm):
        _one(step, logits, targets)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        _one(step, logits, targets)
        best = min(best, time.perf_counter() - t0)
    with torch.profiler.profile(profile_memory=True) as prof:
        _one(step, logits, targets)
    # Positive self allocations only: the profiler emits a matching negative
    # release event, and summing both nets to ~0 (measured).
    mb = sum(max(0, e.self_cpu_memory_usage) for e in prof.key_averages()) / 2**20
    return best * 1e3, mb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8 * 512)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    a = ap.parse_args()

    g = torch.Generator().manual_seed(0)
    logits = torch.randn(a.rows, a.vocab, generator=g)
    targets = torch.randint(0, a.vocab, (a.rows,), generator=g)
    counters = Counters()

    std_ms, std_mb = _time(F.cross_entropy, logits, targets, a.reps, a.warm)
    fix_ms, fix_mb = _time(
        lambda x, t: fixed_cross_entropy(x, t, counters=counters), logits, targets, a.reps, a.warm
    )
    # The same numbers as the training loss: `log_softmax` + gather + mean, the
    # line `_byte_ce`/`loopblock` actually run with the flag OFF.
    std2_ms, std2_mb = _time(
        lambda x, t: -F.log_softmax(x, dim=-1).gather(1, t.reshape(-1, 1)).mean(),
        logits,
        targets,
        a.reps,
        a.warm,
    )
    print(
        f"[{a.rows}, {a.vocab}] fwd+bwd, min of {a.reps} (warm {a.warm}) — CPU\n"
        f"  standard  F.cross_entropy          {std_ms:8.3f} ms  {std_mb:7.1f} MB\n"
        f"  standard  log_softmax+gather+mean  {std2_ms:8.3f} ms  {std2_mb:7.1f} MB\n"
        f"  fixed     fixed_cross_entropy      {fix_ms:8.3f} ms  {fix_mb:7.1f} MB\n"
        f"  ratio fixed / log_softmax          {fix_ms / std2_ms:8.2f}x ms, "
        f"{fix_mb / std2_mb if std2_mb else 0:5.2f}x MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
