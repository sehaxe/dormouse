#!/usr/bin/env python
"""One iteration of the mHC write seam, both Sinkhorn arms: dispatches and µs.

The 2026-11-05 kernel profile (`docs/reviews/2026-11-05-kernel-profile.md`)
priced `mhc_write` at 23.2 ms/step and its largest kernel family
(`CUDAFunctor_add`) at 1 395 launches / 10 steps. This bench measures where
that comes from on the arm the profile actually ran (`fused_residual=false`):
one iteration's `MHCMix.pre_mix` + `MHCMix.forward`, min-of-5, with the
Sinkhorn's `logsumexp` loop versus the `log_softmax` arm
(`mhc_sinkhorn="log_softmax"`).

    # CPU (the card is owned by a training run; CUDA_VISIBLE_DEVICES="")
    cd port
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src .venv/bin/python \
        scripts/bench_mhc_write.py

    # the card must be free (scripts/_lane.preflight refuses otherwise)
    PYTHONPATH=src .venv/bin/python scripts/bench_mhc_write.py --device cuda

The `dispatches` column is the CPU-side launch graph from `torch.profiler`
(non-view ATen ops: composite ops expand to the kernels the CUDA dispatcher
would launch). On a CPU host the µs are CPU µs and are NOT a GPU estimate; on
the card the same lines print CUDA-event µs. The fused kernel
(`fused_residual._write_kernel`) is NOT measured here: it is one launch per
iteration and its 20 Sinkhorn iterations are in registers, so this seam's
launch cost does not exist on that arm.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight as _lane_preflight  # noqa: E402

from dormouse.model.mhc import SINKHORN_IMPLS, MHCMix  # noqa: E402

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
_ADD_FAMILY = ("aten::sub", "aten::add", "aten::add_")


def _ops(fn) -> collections.Counter:
    fn()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
    return collections.Counter(
        e.name for e in prof.events() if e.name.startswith("aten::") and e.name not in _VIEWS
    )


def _us(fn, iters: int, device: str) -> float:
    for _ in range(2):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(end) * 1_000.0)
        return best
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1_000_000.0)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--d", type=int, default=1536, help="m400's d_model")
    ap.add_argument("--iters", type=int, default=5, help="min-of-N timing")
    ap.add_argument("--depth", type=int, default=3, help="loop iterations per step")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = ap.parse_args()
    if args.device == "cuda":
        _lane_preflight()

    torch.manual_seed(0)
    xs = torch.randn(args.batch, args.seq, 2, args.d, device=args.device) * 0.5
    y = torch.randn(args.batch, args.seq, args.d, device=args.device) * 0.5

    rows = []
    for arm in SINKHORN_IMPLS:
        torch.manual_seed(0)
        mhc = MHCMix(args.d, 2, device=args.device, sinkhorn=arm)

        def run(mhc=mhc):
            return mhc.pre_mix(xs), mhc(xs, y)

        c = _ops(run)
        n = sum(c.values())
        adds = sum(c[k] for k in _ADD_FAMILY)
        us = _us(run, args.iters, args.device)
        rows.append((arm, n, adds, us))

    base_n, base_adds, base_us = rows[0][1], rows[0][2], rows[0][3]
    unit = "CUDA-event µs" if args.device == "cuda" else "CPU µs (not a GPU estimate)"
    print(
        f"mhc_write: one iteration = MHCMix.pre_mix + MHCMix.forward, "
        f"b{args.batch}×s{args.seq} K=2 d{args.d}, min-of-{args.iters} {unit}"
    )
    print(f"{'arm':<12} {'dispatches':>10} {'add family':>10} {'µs/iter':>9} {'µs/step':>9}")
    for arm, n, adds, us in rows:
        print(f"{arm:<12} {n:>10} {adds:>10} {us:>9.1f} {us * args.depth:>9.1f}")
    fast_n, fast_adds, fast_us = rows[1][1], rows[1][2], rows[1][3]
    print(
        f"arm ratio: dispatches {base_n / fast_n:.1f}x, add family "
        f"{base_adds / max(fast_adds, 1):.1f}x, µs/iter {base_us / fast_us:.1f}x  "
        f"(×{args.depth} iterations = the step's write, forward only)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
