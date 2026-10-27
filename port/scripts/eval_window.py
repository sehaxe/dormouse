#!/usr/bin/env -S uv run --script
"""Held-out BPB of a checkpoint over a WINDOW you choose.

`dormouse eval` pins `eval_batches` to the checkpoint's own snapshot (it is not a
progress key), so the 32 768-byte window a training run logs cannot be widened
after the fact. A curve logged on 32 KB is noisy enough that two neighbouring
points can invert; this script scores the same checkpoint on as many held-out
batches as you ask for.

    uv run port/scripts/eval_window.py --ckpt-dir checkpoints/longctx1 \
        --ckpt-name longctx1 --eval-batches 64

It reuses the ONE loader and the ONE `evaluate`, so the number is comparable with
the training log's line — same window rule (`eval_batches x batch x seq_len`
bytes from the held-out tail), same `fin` metric.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch

from dormouse.cli import _device, _model, _snapshot_run
from dormouse.instrument import loud
from dormouse.train.data import eval_window_bytes, open_streams
from dormouse.train.loop import evaluate, load_model_half, load_model_state

LN2 = 0.6931471805599453


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--ckpt-name", required=True)
    ap.add_argument("--eval-batches", type=int, default=64)
    ap.add_argument("--eval", default=None, help="score against THIS file")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    if a.eval_batches < 1:
        loud(f"eval_batches={a.eval_batches} must be >= 1")
    run = _snapshot_run(a.ckpt_dir, a.ckpt_name, eval_file=a.eval, eval_batches=None)
    run.train = dataclasses.replace(run.train, eval=True, eval_batches=a.eval_batches)

    have = eval_window_bytes(run.train)
    want = a.eval_batches * run.train.batch * run.train.seq_len
    if have < want:
        loud(
            f"the held-out tail is {have} B, the requested window is {want} B "
            f"({a.eval_batches} x {run.train.batch} x {run.train.seq_len}). "
            f"Escape: --eval-batches {max(1, have // (run.train.batch * run.train.seq_len))}"
        )

    dev = _device(a.device)
    model = _model(run).to(dev)
    load_model_state(model, run.train, load_model_half(a.ckpt_dir / f"{a.ckpt_name}.pt"))
    _train, stream = open_streams(run)
    if stream is None:
        loud("no eval stream (the snapshot carries eval_data)")
    loss, scored, fin, per_iter = evaluate(
        model, stream, a.eval_batches, device=None if dev.type == "cpu" else dev
    )
    print(
        f"{a.ckpt_name} · window {scored} B ({a.eval_batches} x {run.train.batch} x "
        f"{run.train.seq_len}) · headline {loss / LN2:.4f} BPB · fin {fin / LN2:.4f} BPB · "
        f"periter {','.join(f'{v / LN2:.4f}' for v in per_iter)}",
        flush=True,
    )
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
