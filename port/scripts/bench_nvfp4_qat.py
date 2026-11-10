#!/usr/bin/env python
r"""The NVFP4 lane's two numbers: the GEMM and the max batch it buys.

TASK (QAT lane, 2026-11-10): at `[24576, 832] × [832, 2048]` measure the bf16
GEMM against the packed NVFP4 `_scaled_mm`, and the max batch at which the LIVE
arm (`configs/arch_gated_swiglu.toml`, 12 layers, d=832, gated SwiGLU, d_ffn
2048, seq 1024) trains for both activation arms — bf16 vs `act_qat="e2m1"` —
plus steady VRAM.

WHAT THIS CANNOT ANSWER. The owner's acceptance rule is quality per GPU-hour at
each arm's OWN maximum batch; this script measures the COST side only (batch,
ms/step, bytes/s, VRAM) and a 5-step loss, never BPB. A BPB needs a training run
longer than the benchmark, which the lane's rules forbid here.

The batch sweep is a `for` loop that doubles as the OOM probe: the first batch
that does not fit is the ceiling, reported with the previous step's time. Both
arms run the SAME recipe (`dense_qat="ternary"` + AdamW + backward), so the only
difference is the activation quantizer.

Run it through the card's gate:

    GPU_TIMEOUT=1800 bash scripts/gpu_lock.sh .venv/bin/python scripts/bench_nvfp4_qat.py
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from dormouse.config import load_preset
from dormouse.model import DormouseModel
from dormouse.model.nvfp4 import bytes_per_elements, nvfp4_mm

PORT = Path(__file__).resolve().parents[1]
LIVE_ARM = PORT / "configs" / "arch_gated_swiglu.toml"

#: The GEMM the task names: `d_ffn` of the live arm at b24 (M = 24·1024).
GEMM_M, GEMM_N, GEMM_K = 24576, 2048, 832


def _median_ms(fn, *, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def bench_gemm() -> dict:
    """bf16 `x @ Wᵀ` vs the packed NVFP4 lane on the live arm's FFN shape."""
    torch.manual_seed(0)
    x = torch.randn(GEMM_M, GEMM_K, device="cuda", dtype=torch.bfloat16) * 2.0
    w = torch.randn(GEMM_N, GEMM_K, device="cuda", dtype=torch.bfloat16) * 0.02
    bf16_bytes = x.numel() * 2 + w.numel() * 2
    fp4_bytes = (x.numel() + w.numel()) * bytes_per_elements(GEMM_K)

    bf16_ms = _median_ms(lambda: x @ w.t())
    fp4_ms = _median_ms(lambda: nvfp4_mm(x, w, torch.bfloat16))
    return {
        "shape": f"[{GEMM_M},{GEMM_K}]x[{GEMM_K},{GEMM_N}]",
        "bf16_ms": bf16_ms,
        "nvfp4_ms": fp4_ms,
        "speedup": bf16_ms / fp4_ms,
        "bf16_GB_per_s": bf16_bytes / bf16_ms / 1e6,
        "nvfp4_GB_per_s": fp4_bytes / fp4_ms / 1e6,
        "bf16_bytes": bf16_bytes,
        "nvfp4_bytes": int(fp4_bytes),
    }


def _step(model: DormouseModel, opt, x: torch.Tensor) -> float:
    opt.zero_grad(set_to_none=True)
    _, loss, _ = model(x, x)
    loss.backward()
    opt.step()
    return float(loss.detach())


def bench_batch(arm: str, batch: int, *, seq: int, steps: int) -> dict | None:
    """Train `steps` at `batch`; `None` on OOM (the caller treats it as the ceiling)."""
    cfg = dataclasses.replace(load_preset(str(LIVE_ARM)), act_qat=arm, dense_qat="ternary")
    torch.manual_seed(0)
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = DormouseModel(cfg, device="cuda")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
        x = torch.randint(0, 256, (batch, seq), device="cuda")
        for _ in range(2):  # warmup (compile/allocator settle)
            _step(model, opt, x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        losses = [_step(model, opt, x) for _ in range(steps)]
        torch.cuda.synchronize()
        per_step = (time.perf_counter() - t0) / steps
        peak = torch.cuda.max_memory_allocated() / 2**30
        out = {
            "arm": arm,
            "batch": batch,
            "seq": seq,
            "ms_per_step": per_step * 1e3,
            "bytes_per_s": batch * seq / per_step,
            "peak_GiB": peak,
            "loss": losses[-1],
        }
    except torch.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return None
    finally:
        for name in ("model", "opt", "x"):
            if name in locals():
                del locals()[name]  # noqa: B014 - the reference must go before empty_cache
    gc.collect()
    torch.cuda.empty_cache()
    return out


def bench_arms(batches: list[int], *, seq: int, steps: int) -> dict:
    """Ascending batch until OOM, per activation arm, at each arm's own ceiling."""
    rows = {}
    for arm in ("none", "e2m1"):
        ok: list[dict] = []
        ceiling = None
        for b in batches:
            r = bench_batch(arm, b, seq=seq, steps=steps)
            print(
                f"  arm={arm:5s} batch={b:3d} -> "
                + (
                    "OOM"
                    if r is None
                    else f"{r['ms_per_step']:.1f} ms/step "
                    f"{r['peak_GiB']:.2f} GiB loss={r['loss']:.4f}"
                ),
                flush=True,
            )
            if r is None:
                ceiling = b
                break
            ok.append(r)
        rows[arm] = {"fits": ok, "first_oom": ceiling, "max_batch": ok[-1]["batch"] if ok else None}
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=3, help="timed steps per point")
    ap.add_argument("--batches", type=str, default="4,8,12,16,20,24,28,32")
    ap.add_argument("--skip-gemm", action="store_true")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("loud: no CUDA device; nothing was measured", file=sys.stderr)
        return 75
    torch.backends.cuda.matmul.allow_tf32 = True

    out: dict = {"device": torch.cuda.get_device_name(), "seq": args.seq}
    if not args.skip_gemm:
        out["gemm"] = bench_gemm()
        g = out["gemm"]
        print(
            f"\nGEMM {g['shape']}: bf16 {g['bf16_ms']:.2f} ms "
            f"({g['bf16_GB_per_s']:.1f} GB/s) vs NVFP4 {g['nvfp4_ms']:.2f} ms "
            f"({g['nvfp4_GB_per_s']:.1f} GB/s) = {g['speedup']:.2f}x",
            flush=True,
        )

    batches = [int(b) for b in args.batches.split(",")]
    print("\nmax-batch sweep (live arm, dense_qat=ternary, forward+backward+step):", flush=True)
    out["arms"] = bench_arms(batches, seq=args.seq, steps=args.steps)

    print("\n" + json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
