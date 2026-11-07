"""Run-history adapter (2026-11-07): rollbacks are deduped, bytes counted once.

`~/logs/flagship.jsonl` is the real evidence: it has four `run` rows and two
backwards jumps (102800 -> 100100 after a resume, 128300 -> 100 after a fresh
restart). Plotting it as written draws a non-monotone curve; summing
`bytes_seen` over its rows counts the re-run steps again (and, because the
field is CUMULATIVE, the earlier rows too).

`dormouse.runlog` is the tiny reader the audit uses: tag each row with its run
segment, derive `bytes_seen` for files that predate the field, keep the LAST
row per step, and take the max — never the sum — as the unique byte count.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dormouse.instrument import LoudError
from dormouse.runlog import (
    bytes_per_step,
    curve,
    dedupe,
    read_rows,
    tag_segments,
    unique_bytes,
)

LOG = Path("/home/sehaxe/logs/flagship.jsonl")


def _row(**kw) -> dict:
    return kw


def test_tag_segments_and_bytes_from_a_run_row():
    rows = [
        _row(event="run", batch=8, seq_len=1024, grad_accum=3),
        _row(event="log", step=100, loss=5.0),
        _row(event="run", batch=8, seq_len=1024, grad_accum=3),
        _row(event="log", step=50, loss=4.0),
    ]
    tagged = tag_segments(rows)
    assert [r["segment"] for r in tagged] == [0, 0, 1, 1]
    assert bytes_per_step(rows[0]) == 8 * 1024 * 3 == 24_576
    assert tagged[1]["bytes_seen"] == 100 * 24_576
    assert tagged[3]["bytes_seen"] == 50 * 24_576


def test_dedupe_keeps_the_last_measurement_and_sorts():
    rows = tag_segments(
        [
            _row(event="run", batch=1, seq_len=8, grad_accum=1),
            _row(event="log", step=20, loss=2.0),
            _row(event="log", step=10, loss=1.0),
            _row(event="log", step=20, loss=9.0),  # the re-run's answer wins
        ]
    )
    got = dedupe(rows)
    assert [r["step"] for r in got] == [10, 20]
    assert got[1]["loss"] == 9.0
    assert unique_bytes(got) == 20 * 8


def test_a_row_before_the_first_run_is_loud():
    with pytest.raises(LoudError, match="precedes the first"):
        tag_segments([_row(event="log", step=1)])


def test_a_bad_jsonl_line_is_loud(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"event": "run", "batch": 1, "seq_len": 8}\nnot json\n')
    with pytest.raises(LoudError, match="is not JSON"):
        read_rows(p)
    p.write_text('{"noevent": 1}\n')
    with pytest.raises(LoudError, match="not a run row"):
        read_rows(p)


def test_unique_bytes_refuses_a_sum_and_a_mixed_curve():
    unsorted = tag_segments(
        [
            _row(event="run", batch=1, seq_len=8),
            _row(event="log", step=2),
            _row(event="log", step=1),
        ]
    )
    with pytest.raises(LoudError, match="not deduped"):
        unique_bytes([r for r in unsorted if r["event"] == "log"])
    # Dedupe then a mixed-budget curve is refused, not added.
    mixed = tag_segments(
        [
            _row(event="run", batch=1, seq_len=8),
            _row(event="log", step=1),
            _row(event="run", batch=2, seq_len=8),
            _row(event="log", step=2),
        ]
    )
    with pytest.raises(LoudError, match="different"):
        unique_bytes(dedupe(mixed))
    # A `bytes_seen` that disagrees with the step arithmetic is the 2026-10-17
    # defect (the field ignored grad_accum); the reader refuses it, not sums it.
    bad = tag_segments(
        [_row(event="run", batch=1, seq_len=8, grad_accum=3), _row(event="log", step=2)]
    )
    bad[1]["bytes_seen"] = 2 * 8  # the old grad_accum-less value
    with pytest.raises(LoudError, match="ignores grad_accum"):
        unique_bytes([bad[1]])


@pytest.mark.skipif(not LOG.is_file(), reason="the flagship run log is not on this box")
def test_the_real_flagship_log_dedupes_monotone_and_counts_bytes_once():
    rows = read_rows(LOG)
    logs = [r for r in rows if r["event"] == "log"]
    raw_steps = [int(r["step"]) for r in logs]
    assert any(b < a for a, b in zip(raw_steps, raw_steps[1:], strict=False)), (
        "the fixture must contain a rollback; without one this gate proves nothing"
    )
    runs = [r for r in rows if r["event"] == "run"]
    assert len({bytes_per_step(r) for r in runs}) == 1, (
        "the audit's byte arithmetic assumes one token budget; this log's segments all "
        "consume the same bytes per step (b24 x g1 == b8 x g3 == 24 576 bytes)"
    )

    tagged = tag_segments(rows)
    segments = sorted({r["segment"] for r in tagged if r["event"] == "log"})
    # Segment 0 is the first `run` row with no log line after it (the launch died
    # before its first cadence); segments 1..3 carry the curve.
    assert segments == [1, 2, 3]
    for seg in segments:
        seg_steps = [int(r["step"]) for r in tagged if r["event"] == "log" and r["segment"] == seg]
        assert seg_steps == sorted(set(seg_steps)), f"segment {seg} is not monotone on its own"

    deduped = curve(LOG)
    steps = [int(r["step"]) for r in deduped]
    last_segment = {}
    for row in tagged:
        if row["event"] == "log":
            last_segment[int(row["step"])] = row["segment"]
    for row in deduped:
        assert row["segment"] == last_segment[int(row["step"])], (
            "dedupe must keep the LAST-written measurement of a re-run step"
        )
    assert steps == sorted(set(steps)), "the deduped curve must be strictly ascending"
    assert len(deduped) < len(logs), "dedupe must remove the re-run rows"
    assert all(steps[i] < steps[i + 1] for i in range(len(steps) - 1))

    unique = unique_bytes(deduped)
    bps = bytes_per_step(runs[0])
    assert unique == steps[-1] * bps
    raw_sum = sum(int(r["step"]) * bps for r in logs)
    assert raw_sum > unique, "the naive sum must double-count; otherwise the fixture is weak"
    # Dedupe removed exactly the EXTRA occurrence of each re-measured step:
    # the summed-step difference between the raw file and the curve is the
    # duplicate rows' bytes, no more and no less.
    counts: dict[int, int] = {}
    for step in raw_steps:
        counts[step] = counts.get(step, 0) + 1
    duplicated = sum(step * bps * (n - 1) for step, n in counts.items())
    assert duplicated > 0
    assert sum(raw_steps) * bps - sum(steps) * bps == duplicated
    # And `bytes_seen` is CUMULATIVE: summing the raw rows is not a byte count
    # at all — it is the area under the step axis, ~664x the true total here.
    assert raw_sum > unique
    assert raw_sum > sum(steps) * bps > unique
    print(
        f"{LOG}: {len(logs)} log rows, {len(deduped)} unique steps, "
        f"max step {steps[-1]}, unique {unique} B, naive sum {raw_sum} B "
        f"(double-count {raw_sum - unique} B)"
    )
