"""The instrument layer: every degradation is LOUD or COUNTED, never silent.

`docs/decisions/2026-10-15-1b-readiness-plan.md` §7 carries the burn lessons
forward as day-one code (ADR-0011/ADR-0019, ADR-0018 rule 2): every degradation
of a value, a kernel or a config field takes exactly one of three marks, and
there is no fourth:

| mark | meaning | test |
|---|---|---|
| **LOUD** | hard error naming the cause *and* the escape | the run dies saying what to do |
| **COUNTED** | the fallback happens and a counter says so | a reader of the log sees it |
| **SILENT** | it just happened | **a defect** |

This module provides the two supported marks — [`LoudError`] (raised via
[`loud`]) and [`Counters`] — plus the step timer the log line needs. A silent
degradation has no helper here on purpose: it is the thing this module exists
to make unnecessary, not a third option to call.
"""

from __future__ import annotations

import time
from typing import NoReturn


class LoudError(RuntimeError):
    """A degradation that stops the run, naming cause and escape.

    A `RuntimeError` so that an uncaught one is a non-zero exit for any caller
    (`python -m ...`, a test, a script) — the burn side's "the run dies", not a
    warning anyone can scroll past.
    """


def loud(msg: str) -> NoReturn:
    """Raise [`LoudError`] with a message that names cause AND escape."""
    raise LoudError(msg)


class Counters(dict):
    """Named arm counters, printed on the log line.

    Burn's `probe.rs` shape, reduced to what the day-one port runs: a dict of
    `name -> int`, where a non-zero entry is the evidence that a COUNTED
    degradation happened (`kda=sdpa`), that an arm was entered
    (`engram_rows`), or that the loop ran (`iter`). A zero entry is a
    statement of its own — `kda_fla=0` means the fast path is not on this
    machine, not that it was slow.
    """

    def count(self, name: str, n: int = 1) -> None:
        if n == 0:
            return
        assert isinstance(n, int) and n > 0, f"counter {name}: n must be > 0, got {n!r}"
        self[name] = self.get(name, 0) + n

    def line(self) -> str:
        """The one-line form, deterministic (sorted) so two runs' lines diff."""
        return " ".join(f"{k}={self[k]}" for k in sorted(self))

    def merge(self, other: Counters) -> None:
        for k, v in other.items():
            self.count(k, v)


class StepTimer:
    """Wall-clock step timing, printed on the log cadence (`--timers`).

    One instrument, one reading: burn learned the hard way that a step-time
    number without a step index is not a measurement of a step (AGENTS §3.1),
    so [`report`] always carries the step count, the elapsed wall clock and the
    token rate the interval actually produced.
    """

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.mark = self.t0
        self.steps = 0

    def tick(self) -> None:
        self.steps += 1

    def report(self, tokens_in_interval: int) -> str:
        now = time.perf_counter()
        dt = max(now - self.mark, 1e-9)
        ms = 1e3 * dt / max(self.steps, 1)
        rate = tokens_in_interval / dt
        self.mark = now
        self.steps = 0
        return f"{ms:.1f} ms/step {rate:.0f} tok/s"
