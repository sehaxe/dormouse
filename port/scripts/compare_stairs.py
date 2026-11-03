"""The stair table: held-out BPB by TOKENS SEEN, one block per run.

    python scripts/compare_stairs.py                    # m400 vs the 105M flagship
    python scripts/compare_stairs.py --runs m400 flagship

Why not by step: a bigger model is behind at equal steps by construction, so a
step axis answers "which converges faster per update" when the question is
"which is further along for the same wall-clock". Tokens are what the clock buys.

Each run's bytes-per-optimizer-step comes from its own `<out_dir>/<name>.config.toml`
where one exists; the two 105M files predate that snapshot and are pinned here
with the reason (b8 x s1024 x grad_accum 3, their launcher).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LOGS = Path.home() / "logs"
CKPTS = Path.home() / "dormouse" / "checkpoints"

#: bytes per optimizer step for runs whose snapshot is not on disk.
PINNED = {"flagship": 8 * 1024 * 3, "flagship_resume": 8 * 1024 * 3}


def evals(name: str) -> list[tuple[int, float]]:
    p = LOGS / f"{name}.jsonl"
    if not p.is_file():
        return []
    rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    return sorted({(r["step"], r["bpb"]) for r in rows if r.get("event") == "eval"})


def bytes_per_step(name: str) -> int:
    if name in PINNED:
        return PINNED[name]
    try:
        import tomllib

        cand = list(CKPTS.glob(f"*/{name}.config.toml"))
        if not cand:
            return 0
        t = tomllib.loads(cand[0].read_text()).get("train", {})
        return int(t["batch"]) * int(t["seq_len"]) * int(t.get("grad_accum", 1))
    except Exception:
        return 0


def interp(rows: list[tuple[float, float]], x: float) -> float | None:
    """BPB at token count `x`, linear between the bracketing evals; None outside."""
    if not rows or x < rows[0][0] or x > rows[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(rows, rows[1:], strict=False):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return rows[-1][1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    a = ap.parse_args(argv)
    names = a.runs or ["m400", "flagship_resume", "flagship"]

    series: dict[str, list[tuple[float, float]]] = {}
    for n in names:
        bps, rows = bytes_per_step(n), evals(n)
        if not rows or not bps:
            print(f"({n}: skipped — {len(rows)} evals, {bps} B/step)")
            continue
        series[n] = [(st * bps / 1e9, b) for st, b in rows]

    if not series:
        return 1

    for n, pts in series.items():
        print(f"\n{n}  ({len(pts)} evals, {pts[-1][0]:.3f} GB seen)")
        for tok, bpb in pts:
            print(f"  {tok:8.3f} GB  {bpb:.4f}")

    # the comparison that matters: every run at the leader's newest token count
    lead = max(series, key=lambda k: series[k][-1][0])
    x = series[lead][-1][0]
    print(f"\nat {x:.3f} GB seen:")
    base = series[lead][-1][1]
    for n, pts in series.items():
        v = interp(pts, x)
        if v is None:
            print(f"  {n:16} not there yet ({pts[-1][0]:.3f} GB, {pts[-1][1]:.4f})")
        else:
            print(f"  {n:16} {v:.4f}   {v - base:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
