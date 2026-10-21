"""Inference adaptivity, measured as a Pareto curve: quality at matched depth.

The question is not "does a halt head beat fixed depth" (ADR-0013 says it does
not, and an independent 2026 paper replicates that). The question for INFERENCE
is: **at the same average depth, does a policy score better than always running
that depth?** A policy that cannot beat `always-d` at depth d is worthless, and
this file decides that with numbers instead of an opinion.

What it does, per eval batch:
  * runs the model at depth 1..max_iter and records the held-out CE of each;
  * runs three policies and records (average depth, CE):
      - `fixed_d`      — the control: always d iterations;
      - `kl_stop`      — smallest d whose next-iteration KL is below eps
                         (no parameters, no training: an inference-only rule);
      - `oracle_stop`  — smallest d whose own CE improvement is below eps
                         (the upper bound any learned head could reach).
  * prints the KK distance and the CE gap between iterations, so the curve is
    visible and not just summarised.

BPB = CE / ln 2, the same conversion the trainer's eval line uses.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dormouse.data import open_streams  # noqa: E402

from dormouse.config import load_run  # noqa: E402
from dormouse.instrument import Counters  # noqa: E402
from dormouse.model.model import DormouseModel  # noqa: E402

LN2 = math.log(2.0)


def _ce(model, x, y, depth, counters):
    logits, loss, _ = model(x, targets=y, counters=counters, depth=depth)
    return float(loss), logits.detach()


def _kl(a: torch.Tensor, b: torch.Tensor) -> float:
    pa = a.float().softmax(-1)
    pb = b.float().softmax(-1)
    return float((pa * (pa.clamp_min(1e-12).log() - pb.clamp_min(1e-12).log())).sum(-1).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-name", required=True)
    ap.add_argument("--out-dir", default="checkpoints/port")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--max-iter", type=int, default=0, help="0 = the run's max_iter")
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-3, 1e-2, 5e-2])
    args = ap.parse_args()

    run = load_run(args.ckpt_name, out_dir=args.out_dir)
    dev = torch.device(args.device)
    model = DormouseModel(run.model, device=dev).eval()
    state = torch.load(
        os.path.join(run.train.out_dir, f"{args.ckpt_name}.pt"),
        map_location=dev,
        weights_only=False,
    )
    model.load_state_dict(state.get("model", state), strict=False)
    max_iter = args.max_iter or run.model.max_iter

    _, ev = open_streams(run)
    counters = Counters()
    rows = []
    with torch.no_grad():
        for _ in range(args.batches):
            x, y = ev.next_batch()
            x, y = x.to(dev), y.to(dev)
            ces, logits = [], []
            for d in range(1, max_iter + 1):
                ce, lg = _ce(model, x, y, d, counters)
                ces.append(ce)
                logits.append(lg)
            kls = [0.0] + [_kl(logits[i - 1], logits[i]) for i in range(1, max_iter)]
            rows.append((ces, kls))
            counters = Counters()

    n = len(rows)
    print(f"{n} batches, T={run.train.seq_len}, max_iter={max_iter}")
    print("\nfixed depth (the control):")
    print("  d   CE      BPB")
    fixed = {}
    for d in range(1, max_iter + 1):
        ce = sum(r[0][d - 1] for r in rows) / n
        fixed[d] = ce
        print(f"  {d}   {ce:.4f}  {ce / LN2:.4f}")

    def policy(name, pick):
        ce = 0.0
        depth = 0
        for ces, kls in rows:
            d = pick(ces, kls)
            ce += ces[d - 1]
            depth += d
        ce, depth = ce / n, depth / n
        base = fixed[int(round(depth))]
        print(
            f"{name:24s} avg_depth={depth:5.2f}  CE={ce:.4f}  BPB={ce / LN2:.4f}  "
            f"vs fixed@{int(round(depth))}: {ce - base:+.4f} CE"
        )

    print("\npolicies (a policy is only useful if it beats fixed depth at the SAME depth):")
    for eps in args.eps:
        policy(f"kl_stop eps={eps:g}", lambda ces, kls, e=eps: _first_below(kls, e, max_iter))
        policy(
            f"oracle_stop eps={eps:g}",
            lambda ces, kls, e=eps: _first_ce_below(ces, e, max_iter),
        )
    return 0


def _first_below(vals: list[float], eps: float, hi: int) -> int:
    for i in range(1, hi):
        if vals[i] < eps:
            return i
    return hi


def _first_ce_below(ces: list[float], eps: float, hi: int) -> int:
    for i in range(1, hi):
        if ces[i - 1] - ces[i] < eps:
            return i
    return hi


if __name__ == "__main__":
    raise SystemExit(main())
