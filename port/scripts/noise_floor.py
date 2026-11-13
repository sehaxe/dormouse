#!/usr/bin/env python3
"""The noise floor: rerun N steps twice from the same seed, print the BPB spread.

`docs/records/reviews/2026-11-06-control-transformer.md` measured a same-seed rerun
0.043 BPB apart (CUDA nondeterminism on a shared GPU), which is larger than the
0.01-0.03 effects the A/B queue was deciding on. A difference is only a result
if it clears the floor of the SAME setup, so this script measures that floor
instead of assuming it:

    python port/scripts/noise_floor.py --preset tiny --steps 50 --device cpu

Two independent trials, identical seed, identical bytes, identical config;
the printed `spread` is `max(bpb_fin) - min(bpb_fin)` over the trials, at the
configured eval window (`eval_bytes`, pinned `eval_offset`). Compare an A/B's
delta against THIS number, never against 0.01.

Exit codes: 0 = measured, 1 = a trial failed. A spread of 0.0000 is a real
answer (a deterministic setup); it is not a missing measurement.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

LN2 = 0.6931471805599453


def _corpus(path: Path, n: int, seed: int) -> None:
    """A deterministic byte corpus: the probe must not depend on a mounted drive."""
    import random

    path.write_bytes(random.Random(seed).randbytes(n))


def _trial(run, dev, seed: int) -> tuple[float, int]:
    """One training run from a fresh same-seed init, then the configured eval."""
    from dormouse.cli import _model
    from dormouse.train.data import eval_batches_for, open_streams
    from dormouse.train.loop import _eval_loss, bpb, train

    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = _model(run).to(dev)
    stream, eval_stream = open_streams(run)
    assert stream is not None and eval_stream is not None
    train(
        model,
        run,
        stream=stream,
        eval_stream=None,
        log=lambda *_: None,
        device=None if dev.type == "cpu" else dev,
    )
    _mean, scored, fin, _per = _eval_loss(
        model,
        eval_stream,
        eval_batches_for(run.train),
        None if dev.type == "cpu" else dev,
        None,
    )
    return bpb(fin), scored


def report(trials: list[float], *, window: int, offset: int, steps: int) -> str:
    """The printed report. The spread is COMPUTED from the trials, never a constant."""
    assert len(trials) >= 2, "the noise floor needs two trials"
    spread = max(trials) - min(trials)
    lines = [
        f"noise floor: {len(trials)} trials x {steps} steps, same seed, "
        f"window={window} B off={offset}",
    ]
    lines += [f"  trial {i + 1}: bpb_fin={v:.4f}" for i, v in enumerate(trials)]
    lines.append(f"  spread={spread:.4f} BPB")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="tiny", help="model preset name or path")
    ap.add_argument("--steps", type=int, default=50, help="optimizer steps per trial")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--data", default=None, help="training .bin (default: synthetic)")
    ap.add_argument("--eval-data", default=None, help="held-out .bin (default: synthetic)")
    ap.add_argument("--eval-bytes", type=int, default=65_536)
    ap.add_argument("--eval-offset", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--json", default=None, help="write the trials as JSON here")
    a = ap.parse_args(argv)

    from dormouse.cli import _device
    from dormouse.config import load_run

    tmp = Path(tempfile.mkdtemp(prefix="noise_floor_"))
    data = Path(a.data) if a.data else tmp / "train.bin"
    ev = Path(a.eval_data) if a.eval_data else tmp / "eval.bin"
    if a.data is None:
        _corpus(data, 1 << 20, a.seed)
    if a.eval_data is None:
        _corpus(ev, 1 << 20, a.seed + 1)

    over = {
        "steps": a.steps,
        "batch": a.batch,
        "seq_len": a.seq_len,
        "seed": a.seed,
        "data": str(data),
        "eval_data": str(ev),
        "eval": False,
        "eval_bytes": a.eval_bytes,
        "eval_batches": None,
        "eval_offset": a.eval_offset,
        "out_dir": str(tmp / "ckpts"),
        "ckpt_name": "noise",
        "ckpt_every": 10**9,
        "log_every": max(1, a.steps),
        "json_log": "",
    }
    run = load_run(a.preset, train_overrides=over)
    dev = _device(a.device)

    trials: list[float] = []
    scored = 0
    for _ in range(2):
        value, scored = _trial(run, dev, a.seed)
        trials.append(value)
    print(report(trials, window=scored, offset=a.eval_offset, steps=a.steps), flush=True)
    if a.json:
        Path(a.json).write_text(
            json.dumps(
                {
                    "preset": a.preset,
                    "steps": a.steps,
                    "seed": a.seed,
                    "eval_bytes": scored,
                    "eval_offset": a.eval_offset,
                    "trials": trials,
                    "spread": max(trials) - min(trials),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
