"""Rollback-safe run-history reader for the audit (2026-11-07).

`<run>.jsonl` is append-only ACROSS restarts: a resume writes a new `run` row
and continues from the checkpoint's step, so the file can go BACKWARDS
(`~/logs/flagship.jsonl` jumps 102800 -> 100100 at line 1081 and 128300 -> 100
at line 1379). Anything that plots or sums the file as written therefore draws
a non-monotone curve and counts the same training step more than once.

[`dedupe`] keeps the LAST row written at each step — the newest measurement of
that step, which is the one the run that continued reports — and returns the
steps in ascending order. [`curve`] is `read_rows` + `tag_segments` +
`dedupe`. Each row carries `segment` (0-based, one per `run` row) and
`bytes_per_step`, so an audit can restrict itself to one trajectory instead of
silently mixing runs; `bytes_seen` is derived for files written before the
trainer logged it (`step x batch x seq_len x grad_accum`, the trainer's own
arithmetic — the 2026-10-17 fix that `eval_row` documents).

`bytes_seen` is CUMULATIVE. [`unique_bytes`] therefore takes the largest
`bytes_seen` of a deduped curve, never the sum: summing rows double-counts by
construction even with no rollback.

A reader for the audit, not for the trainer: nothing here writes, and a
malformed line is LOUD (ADR-0019), never skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

from .instrument import loud


def read_rows(path: str | Path) -> list[dict]:
    """Every JSON object of a JSONL log, in FILE order. A bad line is LOUD."""
    p = Path(path)
    rows: list[dict] = []
    with p.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                loud(f"{p}:{lineno} is not JSON ({e}). Escape: fix or truncate the log.")
            if not isinstance(row, dict) or "event" not in row:
                loud(
                    f"{p}:{lineno} is not a run row (no `event`): a reader that skips "
                    f"it would silently drop part of the history. Escape: fix the line."
                )
            rows.append(row)
    return rows


def bytes_per_step(run: dict) -> int:
    """The optimizer-step byte count of a `run` row.

    The one arithmetic every trainer row uses: `grad_accum` micro-batches of
    `batch x seq_len` bytes per optimizer step (the 2026-10-17 `bytes_seen`
    fix). A missing field is LOUD because a guessed zero would make the
    curve's byte axis wrong in the direction nobody checks.
    """
    try:
        bps = int(run["batch"]) * int(run["seq_len"]) * int(run.get("grad_accum", 1))
    except (KeyError, TypeError, ValueError):
        loud(
            f"run row {run.get('resumed')!r} has no usable batch/seq_len/grad_accum: "
            f"{run}. Escape: a run row written by this trainer."
        )
    if bps <= 0:
        loud(f"run row names {bps} bytes per step: a step that consumes no bytes is not a run")
    return bps


def tag_segments(rows: list[dict]) -> list[dict]:
    """Assign `segment` and `bytes_per_step`; derive old files' `bytes_seen`.

    Segment 0 starts at the first `run` row. A `log`/`eval` row before it is
    LOUD: its config (and therefore its byte arithmetic) is unknown, and that is
    exactly the row a naive reader would silently attribute to the next run.
    """
    out: list[dict] = []
    seg = -1
    bps = 0
    for row in rows:
        row = dict(row)
        if row["event"] == "run":
            seg += 1
            bps = bytes_per_step(row)
        elif seg < 0:
            loud(
                f"{row['event']} row at step {row.get('step')!r} precedes the first "
                f"`run` row: its batch/seq_len are unknown. Escape: a log whose first "
                f"line is the `run` row (this trainer's format)."
            )
        row["segment"] = seg
        row["bytes_per_step"] = bps
        if "bytes_seen" not in row and "step" in row:
            row["bytes_seen"] = int(row["step"]) * bps
        out.append(row)
    return out


def dedupe(rows: list[dict], *, event: str = "log") -> list[dict]:
    """LAST row per `step` wins, output sorted by step — the rollback adapter.

    Last, not first: after a rollback the file contains an OLD measurement of
    the step followed by the run that continued from a checkpoint, and the
    newest one is what that run reports. The rule is stated here because a
    curve drawn with first-wins would silently splice the dead branch back in.
    Input rows must already carry `segment` (call [`tag_segments`]).
    """
    last: dict[int, dict] = {}
    for row in rows:
        if row["event"] != event:
            continue
        if "segment" not in row:
            loud(f"dedupe: a {event} row carries no `segment` — tag_segments first")
        last[int(row["step"])] = row
    return [last[step] for step in sorted(last)]


def curve(path: str | Path, *, event: str = "log") -> list[dict]:
    """A rollback-safe curve: last-wins per step, ascending, bytes derived."""
    return dedupe(tag_segments(read_rows(path)), event=event)


def unique_bytes(rows: list[dict]) -> int:
    """Cumulative bytes of a DEDUPED curve — the max, never the sum.

    Requires strictly ascending steps (dedupe's contract) and ONE
    `bytes_per_step`: a curve that mixes runs with different token budgets is
    not one byte axis, and adding its rows would invent a number. A row whose
    `bytes_seen` runs backwards is LOUD for the same reason.
    """
    bps = None
    last_step = -1
    seen = 0
    for row in rows:
        step = int(row["step"])
        if step <= last_step:
            loud(
                f"unique_bytes: step {step} after {last_step} — the curve is not "
                f"deduped/sorted. Escape: curve()/dedupe() first."
            )
        last_step = step
        row_bps = int(row["bytes_per_step"])
        if bps is None:
            bps = row_bps
        elif row_bps != bps:
            loud(
                f"unique_bytes: the curve mixes {bps} and {row_bps} bytes/step runs; "
                f"their byte axes are different and cannot be added. Escape: filter "
                f"to one `segment`."
            )
        seen = int(row["bytes_seen"])
        if seen != step * row_bps:
            loud(
                f"unique_bytes: step {step} carries bytes_seen={seen}, but the "
                f"trainer's arithmetic is step x batch x seq_len x grad_accum = "
                f"{step * row_bps}. A `bytes_seen` that ignores grad_accum (the "
                f"2026-10-17 defect) is not a byte count. Escape: the log's own "
                f"run row, or rebuild the field."
            )
    return seen
