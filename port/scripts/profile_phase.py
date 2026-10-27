#!/usr/bin/env -S uv run --python /home/sehaxe/dormouse/port/.venv/bin/python --script
"""Where does a training step's time actually go, at a chosen shape?

`bench_step.py` answers "how fast"; this answers "why". It runs a few steps under
`torch.profiler`, prints the phase split (forward / backward / clip / optimizer /
zero_grad) from the wall clock, the top kernels by self CUDA time, the launch
count per phase, and the MFU against the card's measured bf16 GEMM peak.

    uv run --python port/.venv/bin/python port/scripts/profile_phase.py \
        --batch 1 --seq 8192 --iters 3 --steps 3 --warmup 2
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict

import torch

from dormouse.cli import _device, _model, _snapshot_run
from dormouse.config import validate_train
from dormouse.instrument import Counters
from dormouse.train.loop import load_model_half, load_model_state
from dormouse.train.optim import MixOptimizer

PEAK_TFLOPS = 44.0  # measured bf16 GEMM on this card, 2026-10-17


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--latent", type=int, default=256)
    ap.add_argument("--mm-dtype", default="bf16")
    ap.add_argument("--ckpt-dir", default="/home/sehaxe/dormouse/checkpoints")
    ap.add_argument("--ckpt-name", default="longctx1")
    ap.add_argument("--top", type=int, default=14)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    run = _snapshot_run(
        pathlib.Path(a.ckpt_dir) / a.ckpt_name, a.ckpt_name, eval_file=None, eval_batches=None
    )
    for k, v in {
        "batch": a.batch,
        "seq_len": a.seq,
        "max_seq_len": a.seq,
        "max_iter": a.iters,
        "mm_dtype": a.mm_dtype,
        "attn_latent_rank": a.latent,
        "use_csa": True,
        "use_msa": False,
        "use_mhc": True,
        "use_attnres": True,
        "share_u": True,
        "use_engram": False,
        "moe_kernel": "triton",
        "moe_capacity": 0.0,
        "doc_masking": False,
        "eval": False,
    }.items():
        setattr(run.train, k, v)
    validate_train(run.train)

    dev = _device(a.device)
    model = _model(run).to(dev).train()
    load_model_state(
        model,
        run.train,
        load_model_half(pathlib.Path(a.ckpt_dir) / a.ckpt_name / f"{a.ckpt_name}.pt"),
    )
    c = run.train
    opt = MixOptimizer(
        model,
        lr=c.lr,
        wd=c.wd,
        opt=c.opt,
        ns_steps=c.ns_steps,
        ns_kind=c.ns_kind,
        colrow_3d=c.muon_colrow_3d,
        muon_state_dtype=c.muon_state_dtype,
        ns_dtype=c.ns_dtype,
        muon_variant=c.muon_variant,
        muon_sf=c.muon_sf,
        muon_sf_wd=c.muon_sf_wd,
    )
    ids = torch.randint(0, 256, (a.batch, a.seq), device=dev)

    def one_step() -> dict[str, float]:
        phases: dict[str, float] = defaultdict(float)
        t0 = time.perf_counter()
        logits, loss, _cs = model(ids, targets=ids, counters=Counters())
        torch.cuda.synchronize(dev)
        phases["forward"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize(dev)
        phases["backward"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.cuda.synchronize(dev)
        phases["clip"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        opt.step()
        torch.cuda.synchronize(dev)
        phases["opt"] = time.perf_counter() - t0
        return phases

    for _ in range(a.warmup):
        one_step()
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)

    acc: dict[str, list[float]] = defaultdict(list)
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(a.steps):
            for k, v in one_step().items():
                acc[k].append(v)

    total = sum(sum(v) for v in acc.values()) / a.steps
    print(f"\n{a.preset} b={a.batch} s={a.seq} it={a.iters} {a.mm_dtype}")
    for k, v in acc.items():
        ms = sum(v) / len(v) * 1e3
        print(f"  {k:9s} {ms:8.2f} ms  {100 * ms / (total * 1e3):5.1f}%")
    print(f"  {'TOTAL':9s} {total * 1e3:8.2f} ms  -> {a.batch * a.seq / total:,.0f} B/s")

    rows = sorted(prof.key_averages(), key=lambda r: r.self_device_time_total, reverse=True)
    gpu_total = sum(r.self_device_time_total for r in rows) or 1.0
    print(f"\n  top kernels by self CUDA time ({gpu_total / 1e3:.0f} ms total):")
    for r in rows[: a.top]:
        if r.self_device_time_total <= 0:
            continue
        print(
            f"    {r.self_device_time_total / gpu_total * 100:5.1f}%  "
            f"{r.self_device_time_total / 1e3:8.2f} ms  {r.count:7d}x  {r.key[:58]}"
        )
    launches = sum(r.count for r in rows)
    print(f"\n  launches/step: {launches / (a.steps * 5):,.0f}   peak {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB")

    # MFU from an independent FLOP count of the same shape.
    from dormouse.width import macs_per_token

    try:
        m = run.model
        macs = macs_per_token(m, iters=m.max_iter, n_heads=m.n_heads)
    except Exception:
        macs = None
    if macs:
        flops_per_byte = 3 * 2 * macs
        achieved = (a.batch * a.seq / total) * flops_per_byte
        print(
            f"  FLOP/byte {flops_per_byte / 1e6:.1f}M -> {achieved / 1e12:.2f} TFLOP/s "
            f"= {100 * achieved / 1e12 / PEAK_TFLOPS:.1f}% MFU"
        )
    print(json.dumps({"bytes_per_s": a.batch * a.seq / total, "ms": total * 1e3}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
