#!/usr/bin/env -S uv run --python /home/sehaxe/dormouse/port/.venv/bin/python --script
"""Time the training step at a given shape and dtype — no eval, no checkpoints.

The `dormouse train` CLI cannot answer "bytes/second" because every candidate run
pays ~12 s of startup and the per-step timing only appears at `log_every`, which
is longer than a 20-step probe. This script builds the same model and optimizer,
warms up, and times a fixed number of steps with `torch.cuda.Event`, reporting the
median and the implied bytes/second.

    uv run --python port/.venv/bin/python port/scripts/bench_step.py \
        --batch 8 --seq 1024 --steps 20 --warmup 5 --iters 3 --mm-dtype bf16
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import torch

from dormouse.cli import _device, _model, _snapshot_run
from dormouse.config import validate, validate_train
from dormouse.instrument import Counters
from dormouse.train.loop import load_model_half, load_model_state
from dormouse.train.optim import MixOptimizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--mm-dtype", default="bf16", choices=("bf16", "fp8"))
    ap.add_argument("--latent", type=int, default=256)
    ap.add_argument("--out-rank", type=int, default=0)
    ap.add_argument("--no-csa", action="store_true")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--fused", action="store_true", help="fused mHC+AttnRes chain")
    ap.add_argument("--ckpt-dir", default="/home/sehaxe/dormouse/checkpoints")
    ap.add_argument("--ckpt-name", default="longctx1")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    overrides = {
        "batch": a.batch,
        "seq_len": a.seq,
        "max_seq_len": a.seq,
        "max_iter": a.iters,
        "mm_dtype": a.mm_dtype,
        "attn_latent_rank": a.latent,
        "out_proj_rank": a.out_rank,
        "use_csa": not a.no_csa,
        "use_msa": False,
        "use_mhc": True,
        "use_attnres": True,
        "share_u": True,
        "use_engram": not a.no_engram,
        "moe_kernel": "triton",
        "moe_capacity": 0.0,
        "doc_masking": False,
        "eval": False,
        "steps": a.steps + a.warmup,
    }
    run = _snapshot_run(
        __import__("pathlib").Path(a.ckpt_dir) / a.ckpt_name,
        a.ckpt_name,
        eval_file=None,
        eval_batches=None,
    )
    for k, v in overrides.items():
        setattr(run.train, k, v)
    validate_train(run.train)

    run.model.fused_residual = a.fused
    run.model.rank = a.rank
    run.model.n_experts = a.experts
    validate(run.model)
    dev = _device(a.device)
    model = _model(run).to(dev).train()
    ckpt = pathlib.Path(a.ckpt_dir) / a.ckpt_name / f"{a.ckpt_name}.pt"
    load_model_state(model, run.train, load_model_half(ckpt))
    c = run.train
    opt = MixOptimizer(
        model, lr=c.lr, wd=c.wd, opt=c.opt, ns_steps=c.ns_steps, ns_kind=c.ns_kind,
        colrow_3d=c.muon_colrow_3d, muon_state_dtype=c.muon_state_dtype,
        ns_dtype=c.ns_dtype, muon_variant=c.muon_variant, muon_sf=c.muon_sf,
        muon_sf_wd=c.muon_sf_wd,
    )

    b, t = a.batch, a.seq
    ids = torch.randint(0, 256, (b, t), device=dev)
    times: list[float] = []
    peak = 0.0
    for step in range(a.steps + a.warmup):
        t0 = time.perf_counter()
        logits, loss, _c = model(ids, targets=ids, counters=Counters())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize(dev)
        dt = time.perf_counter() - t0
        peak = max(peak, torch.cuda.max_memory_allocated(dev) / 2**30)
        if step >= a.warmup:
            times.append(dt)
    med = statistics.median(times)
    bps = (b * t) / med
    print(
        f"{a.preset} b={b} s={t} it={a.iters} {a.mm_dtype} latent={a.latent} out={a.out_rank} "
        f"csa={not a.no_csa} -> {med * 1e3:.1f} ms/step  {bps:,.0f} B/s  "
        f"{bps * 86400 / 1e9:.2f} GB/day  peak {peak:.2f} GiB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
