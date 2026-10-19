"""The eval loop: items in, one report per task out, one JSONL row per report.

The loop is small on purpose and holds three rules that are not obvious from the
code that follows them:

1. **Both accuracies** — the raw log-prob sum and the per-byte mean — are always
   reported together. They are different protocols: a sum rewards the shorter
   choice, a mean rewards the denser one, and on a task whose choices differ in
   length they disagree. The headline is the mean ([`TaskReport.acc_mean`]) and
   the sum rides next to it, so a reader can see the disagreement instead of
   inheriting whichever one the harness happened to pick.
2. **An overlong item is DROPPED, never truncated.** `C + c_i` above
   `max_seq_len` is an item this model cannot be asked; cutting the choice would
   score a different question under the same `tid`. The drop is counted
   (`bench_dropped`) and printed, and an all-dropped task refuses: an accuracy
   over 0 items is not a number.
3. **Every number carries its `n` and its bytes.** `bytes_scored` counts the
   choice bytes the log-probs actually covered — the scored positions, not the
   context, which is the same for every choice and is never scored.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..instrument import Counters, loud
from .score import LogitsFn, score_choices
from .tasks import MCItem, load_task

__all__ = ["TaskReport", "item_bytes", "run_bench", "run_task"]


@dataclass(frozen=True)
class TaskReport:
    """One task's result: n, drops, both accuracies, bytes, and the arm counters."""

    task: str
    n: int
    dropped: int
    acc_raw: float
    acc_mean: float
    bytes_scored: int
    max_seq_len: int
    counters: dict[str, int] = field(default_factory=dict)

    def row(self, arm: Mapping[str, int] | None = None) -> dict[str, object]:
        """The JSONL row: flat core numbers + the harness and arm counters."""
        return {
            "event": "bench",
            "task": self.task,
            "n": self.n,
            "dropped": self.dropped,
            "acc_raw": round(self.acc_raw, 6),
            "acc_mean": round(self.acc_mean, 6),
            "bytes_scored": self.bytes_scored,
            "max_seq_len": self.max_seq_len,
            "counters": dict(self.counters),
            "arm": dict(arm or {}),
        }

    def line(self) -> str:
        """The printed form. Every number here is accompanied by its n."""
        return (
            f"bench {self.task}  n={self.n}  dropped={self.dropped}  "
            f"acc_mean={self.acc_mean:.4f}  acc_raw={self.acc_raw:.4f}  "
            f"bytes={self.bytes_scored:,}  (max_seq_len={self.max_seq_len})"
        )


def item_bytes(item: MCItem) -> tuple[bytes, tuple[bytes, ...]]:
    """The item's UTF-8 bytes — the ONE place text becomes tokens.

    The corpus is raw UTF-8 and `vocab=256`, so a task's text is its bytes and
    nothing else: no tokenizer, no normalization, no special tokens.
    """
    return item.context.encode("utf-8"), tuple(c.encode("utf-8") for c in item.choices)


def run_task(
    items: Sequence[MCItem],
    *,
    logits_fn: LogitsFn,
    max_seq_len: int,
    task: str = "?",
    limit: int = 0,
) -> TaskReport:
    """Score `items` (capped at `limit`) and return the report.

    The prediction is `argmax` of the raw sum — the protocol's score. `acc_mean`
    re-picks the argmax on the per-byte means, so the two columns are two
    protocols' accuracies over the same forwards, not one accuracy and one
    average of it.
    """
    assert max_seq_len >= 2, f"max_seq_len={max_seq_len}: a window must hold a context and a byte"
    counters = Counters()
    n = 0
    dropped = 0
    bytes_scored = 0
    correct_raw = 0
    correct_mean = 0
    with torch.no_grad():
        for item in list(items)[: limit or None]:
            ctx, choices = item_bytes(item)
            longest = max(len(c) for c in choices)
            if len(ctx) + longest > max_seq_len:
                # Dropped, never truncated: the counter is the record.
                counters.count("bench_dropped")
                dropped += 1
                continue
            scores = score_choices(logits_fn, ctx, choices)
            raw = [s.logp for s in scores]
            mean = [s.mean for s in scores]
            n += 1
            bytes_scored += sum(s.n_bytes for s in scores)
            correct_raw += int(max(range(len(raw)), key=raw.__getitem__) == item.label)
            correct_mean += int(max(range(len(mean)), key=mean.__getitem__) == item.label)
    if n == 0:
        loud(
            f"task {task}: 0 of {len(list(items))} items were scored ({dropped} dropped: "
            f"longer than max_seq_len={max_seq_len}). An accuracy over 0 items is not a "
            f"number. Escape: raise max_seq_len, or use a task whose items fit the window."
        )
    return TaskReport(
        task=task,
        n=n,
        dropped=dropped,
        acc_raw=correct_raw / n,
        acc_mean=correct_mean / n,
        bytes_scored=bytes_scored,
        max_seq_len=max_seq_len,
        counters=dict(counters),
    )


def run_bench(
    logits_fn: LogitsFn,
    names: Sequence[str],
    *,
    data_dir: Path | str,
    max_seq_len: int,
    limit: int = 0,
    json_log: Path | str | None = None,
    arm: Counters | None = None,
) -> list[TaskReport]:
    """Every named task, in order, printed as it finishes; rows appended to `json_log`.

    `arm` is the run-level counter set the caller's model writes into (the FLA
    KDA path vs the SDPA fallback): it is copied into each row, so a JSONL row
    says which attention arm produced its number and not only what the number
    was.
    """
    reports = []
    for name in names:
        items = load_task(name, data_dir, limit=limit)
        report = run_task(
            items, logits_fn=logits_fn, max_seq_len=max_seq_len, task=name, limit=limit
        )
        print(report.line())
        reports.append(report)
        if json_log is not None:
            with open(json_log, "a", encoding="utf-8") as sink:
                sink.write(json.dumps(report.row(arm)) + "\n")
    return reports
