#!/usr/bin/env python3
"""Collect every A/B arm measured on the mix4 screening protocol into one table.

Protocol every row shares: `medium`, `data_local/mix4/web.bin`, eval
`mix4/web_eval.bin` (doc-sha1 disjoint), 600 steps, window 40 960 B
(`10 eval_batches x batch 8 x seq_len 512`), 3 seeds.

Control = `logs/n1_ctl600_s{1,2,3}.log` (plain bf16, r64, d1024).
Arms   = `logs/{n2,f1,a1,tr}_<arm>_s{1,2,3}.log`.
"""

from __future__ import annotations

import glob
import os
import re
import statistics
import sys

BPB = re.compile(r"bpb=([0-9.]+)")
CONTROL_FILES = "~/logs/n1_ctl600_s*.log"
ARM_FILE = re.compile(r"/(?:n2|f1|a1|tr)_([a-z0-9_]+)_s(\d)\.log$")


def last_bpb(path: str) -> float | None:
    last = None
    with open(path, errors="ignore") as fh:
        for line in fh:
            hit = BPB.search(line)
            if hit:
                last = float(hit.group(1))
    return last


def main() -> int:
    ctl = [v for p in sorted(glob.glob(os.path.expanduser(CONTROL_FILES))) if (v := last_bpb(p))]
    if not ctl:
        print("no control found (logs/n1_ctl600_s*.log)", file=sys.stderr)
        return 1

    arms: dict[str, list[float]] = {}
    for path in glob.glob(os.path.expanduser("~/logs/*.log")):
        m = ARM_FILE.search(path)
        if not m:
            continue
        val = last_bpb(path)
        if val is not None:
            arms.setdefault(m.group(1), []).append(val)

    # A CONTROL WITH ONE SEED HAS NO SPREAD, AND `bar = c_mean - 2*0` MAKES AN
    # EXACT TIE A WIN. Measured class (audit 2026-11-04): one control log, or
    # identical seeds, and every arm that does nothing "wins". The rule below
    # refuses to promote on fewer than MIN_SEEDS control runs and requires the
    # effect to clear both the control's spread AND a floor that a BPB difference
    # has to beat to mean anything at our eval resolution.
    MIN_SEEDS = 3
    MIN_EFFECT = 0.01  # BPB; our eval spread is ~0.006-0.02 at 600-2000 steps
    if len(ctl) < MIN_SEEDS:
        print(
            f"control has {len(ctl)} seed(s), need {MIN_SEEDS}: a spread of 0 makes "
            f"an exact tie a WIN, so no promotion is possible. "
            f"Escape: run the control at {MIN_SEEDS} seeds.",
            file=sys.stderr,
        )
        return 2
    c_mean = statistics.mean(ctl)
    c_spread = max(ctl) - min(ctl)
    if c_spread <= 0.0:
        print(
            f"control seeds are identical ({ctl}): the spread is 0 and the bar is "
            f"the mean, so a tie would promote. Escape: distinct seeds.",
            file=sys.stderr,
        )
        return 2
    need = max(2 * c_spread, MIN_EFFECT)
    bar = c_mean - need
    print(f"control: n={len(ctl)} {[round(x, 4) for x in ctl]}")
    print(
        f"  mean={c_mean:.4f}  spread={c_spread:.4f}  need={need:.4f}  "
        f"promotion bar={bar:.4f}\n"
    )
    print(f"{'arm':16s} {'n':>2s} {'mean':>7s} {'spread':>7s} {'delta':>8s}  verdict")
    print("-" * 58)
    for arm in sorted(arms, key=lambda a: statistics.mean(arms[a])):
        vals = arms[arm]
        mean = statistics.mean(vals)
        spread = max(vals) - min(vals) if len(vals) > 1 else 0.0
        delta = mean - c_mean
        if len(vals) < 3:
            verdict = "incomplete"
        elif mean <= bar:
            verdict = "WIN"
        elif delta <= c_spread:
            verdict = "tie"
        else:
            verdict = "LOSS"
        print(f"{arm:16s} {len(vals):2d} {mean:7.4f} {spread:7.4f} {delta:+8.4f}  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
