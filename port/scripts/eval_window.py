#!/usr/bin/env -S uv run --script
"""Held-out BPB of a checkpoint over a WINDOW you choose.

A training run pins its eval window in the snapshot (`eval_bytes`, or a legacy
snapshot's `eval_batches`) and that is the comparable number. This script scores
the same checkpoint on another window and, unlike the old version, prints the
TWO metrics separately: `bpb_fin` (the final readout — the model's BPB) and
`bpb_mean` (the depth-averaged training objective). A reader who quotes one
never accidentally quotes the other.

    uv run port/scripts/eval_window.py --ckpt-dir checkpoints/longctx1 \
        --ckpt-name longctx1 --eval-bytes 196608

The window starts at a PINNED byte offset (`--eval-offset`, printed as `off=`)
so two invocations are comparable: same offset + same bytes = same text, while
a different offset is a different sample and is labelled as one.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch

from dormouse.cli import _device, _model, _snapshot_run
from dormouse.eval_window import training_overlap, training_paths  # noqa: F401
from dormouse.instrument import loud
from dormouse.train.data import eval_batches_for, open_streams
from dormouse.train.loop import evaluate, load_model_half, load_model_state

LN2 = 0.6931471805599453


def refuse_training_overlap(run, eval_path: Path, allow: bool) -> None:
    """LOUD when the eval window sits on the corpus: that number is not held out.

    Found by the data/eval audit (2026-11-08): the `off=100,000,000` window on
    `best/fwedu.bin` was scored while the run's own fwedu cursor had already
    passed 108.9-110.0 MB, i.e. the "held-out" BPB was reading training bytes.
    A path comparison is the cheapest honest detector; it cannot see a copy of
    the same text under another name, and it says so.
    """
    if allow:
        return
    hit = training_overlap(run.train.data, run.train.mix_plan, eval_path)
    if hit is not None:
        loud(
            f"--eval {eval_path} is ALSO a training source of this run ({hit}). "
            f"A window in it is not held out: the model may have read those bytes. "
            f"Escape: point --eval at a file no source reads, or pass "
            f"--allow-train-overlap and name the number accordingly."
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--ckpt-name", required=True)
    ap.add_argument("--eval-bytes", type=int, default=196_608)
    ap.add_argument("--eval-offset", type=int, default=0, help="pinned byte offset")
    ap.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="legacy window in batches (overrides --eval-bytes)",
    )
    ap.add_argument("--eval", default=None, help="score against THIS file")
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--allow-train-overlap",
        action="store_true",
        help="score a window that lives in the training corpus (the number is then not held out)",
    )
    a = ap.parse_args()

    run = _snapshot_run(a.ckpt_dir, a.ckpt_name, eval_file=a.eval, eval_batches=None)
    if run.train.eval_data:
        refuse_training_overlap(run, Path(run.train.eval_data), a.allow_train_overlap)
    over = {"eval": True, "eval_offset": a.eval_offset}
    if a.eval_batches is not None:
        over["eval_batches"] = a.eval_batches
    else:
        over["eval_batches"] = None
        over["eval_bytes"] = a.eval_bytes
    run.train = dataclasses.replace(run.train, **over)
    batches = eval_batches_for(run.train)

    dev = _device(a.device)
    model = _model(run).to(dev)
    load_model_state(model, run.train, load_model_half(a.ckpt_dir / f"{a.ckpt_name}.pt"))
    _train, stream = open_streams(run)
    if stream is None:
        loud("no eval stream (the snapshot carries eval_data)")
    loss, scored, fin, per_iter = evaluate(
        model, stream, batches, device=None if dev.type == "cpu" else dev
    )
    print(
        f"{a.ckpt_name} · window {scored} B ({batches} x {run.train.batch} x "
        f"{run.train.seq_len}) off={stream.lo} · "
        f"bpb_fin={fin / LN2:.4f} · bpb_mean={loss / LN2:.4f} · "
        f"periter {','.join(f'{v / LN2:.4f}' for v in per_iter)}",
        flush=True,
    )
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
