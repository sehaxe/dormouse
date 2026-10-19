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

import torch


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

    Device-side counters (`count_device`) stay on their device until the
    trainer drains the whole window once per log cadence — no per-pass
    GPU→CPU read survives, which is what blocked torch.compile's graphs in
    the arm-level probe (the `int((~keep).sum())` sync, now a device counter).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._device: dict[str, torch.Tensor] = {}

    #: The reason a COUNTED fallback ran is part of the row, not an attribute
    #: beside it: the JSON sink serialises `dict(window)` and the log line is
    #: `line()`, so a plain attribute would reach neither (`why` was written by
    #: eleven call sites and printed by nothing — 2026-10-19 audit M2). It lives
    #: under the one string key `"why"`; every int-only path below skips it.
    @property
    def why(self) -> str | None:
        v = self.get("why")
        return str(v) if isinstance(v, str) else None

    @why.setter
    def why(self, value: str) -> None:
        if value:
            dict.__setitem__(self, "why", str(value))
        else:
            dict.pop(self, "why", None)

    def count(self, name: str, n: int = 1) -> None:
        # THE ORDER IS LOAD-BEARING (audit 2026-10-17, CRITICAL 1): a device
        # tensor must reach `count_device` BEFORE any comparison. `if n == 0`
        # first builds a device bool and `if <bool tensor>` is a D2H read —
        # every MoE forward syncs, and torch.compile's graphs break.
        if isinstance(n, torch.Tensor):
            self.count_device(name, n)
            return
        if n == 0:
            return
        assert isinstance(n, int) and n > 0, f"counter {name}: n must be > 0, got {n!r}"
        self[name] = self.get(name, 0) + n

    def count_device(self, name: str, n: torch.Tensor) -> None:
        """A DEVICE-side counter: `n` stays on its device; no host read.

        This is the 0-sync rule (the port's rule 1.3): a counter built from a
        device value must never call `.item()`. The drain happens ONCE per
        window — `drain()` is called by the trainer on the log cadence only.
        """
        self._device[name] = self._device.get(name, 0) + n

    def drain(self) -> dict[str, int]:
        """Read every pending device counter ONE time, as a batch.

        One GPU→CPU transfer for the whole window, at the log cadence's
        declared cost. The pending state empties; counters stay honest.
        """
        out: dict[str, int] = {}
        for name, t in list(self._device.items()):
            out[name] = int(t.item()) if t.numel() == 1 else [int(x) for x in t.tolist()]
            del self._device[name]
        for name, value in out.items():
            if isinstance(value, list):
                for i, v in enumerate(value):
                    if v:
                        self.count(f"{name}[{i}]", v)
            else:
                self.count(name, value)
        return out

    def line(self) -> str:
        """The one-line form, deterministic (sorted) so two runs' lines diff.

        Counters first, `why` last as `fallback: <reason>` — the reason is
        free text, so it goes at the END where a whitespace parser can take the
        rest of the line, and it is printed WHOLE (a counter name longer than
        24 characters is not a parser column, and truncating it would hide the
        name the reader needs).
        """
        body = " ".join(f"{k}={self[k]}" for k in sorted(self) if k != "why")
        why = self.why
        if not why:
            return body
        return f"{body} fallback: {why}" if body else f"fallback: {why}"

    def merge(self, other: Counters) -> None:
        for k, v in other.items():
            if k == "why":
                continue
            self.count(k, v)
        for k, v in other._device.items():
            self._device[k] = self._device.get(k, 0) + v
        # Distinct reasons only: a per-step fallback merges thousands of times
        # per window and the same reason must not grow the line.
        prev = self.why or ""
        if other.why and other.why not in prev:
            self.why = f"{prev}; {other.why}" if prev else other.why


class GuardCadence:
    """When a device→host GUARD may take its one read: first N calls, then every M.

    Rule 1.3 (zero host-device synchronization) forbids a per-forward
    `bool(any())` / `isfinite(...).all()`: the read serialises CPU against GPU
    on every pass. A guard whose job is to refuse a corrupt input still needs to
    run somewhere, so it runs on this cadence and the step's NaN firewall is the
    detector in the gap. `reset()` is the test seam — a guard test must not
    inherit the count of the tests before it, or "the guard refuses" would be
    asserted against a skipped call.
    """

    def __init__(self, first: int = 8, every: int = 1000) -> None:
        assert first >= 1 and every >= 1, (first, every)
        self.first = first
        self.every = every
        self.calls = 0

    def due(self) -> bool:
        """True on calls 1..`first` and every `every`-th call after that."""
        self.calls += 1
        return self.calls <= self.first or self.calls % self.every == 0

    def reset(self) -> None:
        self.calls = 0


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

    def window_s(self) -> float:
        """The seconds this window of steps took, WITHOUT resetting the mark."""
        return max(time.perf_counter() - self.mark, 1e-9)
