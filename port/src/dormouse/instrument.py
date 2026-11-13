"""The instrument layer: every degradation is LOUD or COUNTED, never silent.

`docs/records/decisions/2026-10-15-1b-readiness-plan.md` §7 carries the burn lessons
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

from . import stepaudit


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
        window — `drain_if_due()` is called by the trainer on the log cadence.

        The accumulator is updated IN PLACE after the first call: the old
        `get(name, 0) + n` allocated a fresh tensor per count, i.e. per step per
        counter per counter object. The first tensor is cloned so a caller's
        scalar (a firewall flag view) can never be mutated through the counter.
        """
        cur = self._device.get(name)
        if cur is None:
            self._device[name] = n.detach().clone()
        else:
            cur.add_(n)

    def drain(self) -> dict[str, int]:
        """Read every pending device counter with ONE host read for the window.

        One GPU→CPU transfer for the whole window, at the log cadence's
        declared cost. The pending state empties; counters stay honest.

        The read is batched on purpose: the first version called `.item()` /
        `.tolist()` once per counter, so a 12-counter window paid 12 device
        syncs (`scripts/sync_audit.py` measured exactly that) while the
        docstring claimed one. Every value is cast to float64 and concatenated
        on the device, then `.tolist()` crosses ONCE — counts stay exact below
        2**53; float32 readings widen exactly.

        Integer tensors accumulate as integer counters (a zero is not an
        entry). A 0-dim FLOAT tensor is kept as a float under the same name,
        INCLUDING an exact 0.0: a norm is a measurement, not a count, and
        truncating it to `int` would make `g_norm_muon=3` unable to tell 0.31
        from 0.39 (the per-group gradient-norm diagnostic,
        `train/optim.py::group_grad_norms`); an all-zero group is a statement
        ("every gradient in it was gated"), not an absence.
        """
        if not self._device:
            return {}
        items = list(self._device.items())
        self._device.clear()
        meta: list[tuple[str, bool, int]] = []
        flats: list[torch.Tensor] = []
        for name, t in stepaudit.iterations("counters.drain.pack", items):
            meta.append((name, t.dtype.is_floating_point, t.numel()))
            flats.append(t.detach().reshape(-1).to(torch.float64))
        # THE ONE READ of the window: a single `.tolist()` on a CPU tensor (one
        # stream drain on CUDA), not one per counter.
        values = torch.cat(flats).tolist()
        out: dict[str, float | int | list[int]] = {}
        pos = 0
        for name, is_float, n in stepaudit.iterations("counters.drain", meta):
            chunk = values[pos : pos + n]
            pos += n
            if n == 1:
                out[name] = chunk[0] if is_float else int(chunk[0])
            else:
                out[name] = [int(v) for v in chunk]
        self._fold(out)
        return out

    def _fold(self, out: dict[str, float | int | list[int]]) -> None:
        """Merge already-drained values into the host counters (the ONE policy)."""
        for name, value in stepaudit.iterations("counters.fold", out.items()):
            if isinstance(value, list):
                for i, v in stepaudit.iterations("counters.fold.list", enumerate(value)):
                    if v:
                        self.count(f"{name}[{i}]", v)
            elif isinstance(value, float):
                self[name] = self.get(name, 0) + value
            elif value:
                self.count(name, value)

    def absorb(self, drained: dict[str, float | int | list[int]]) -> None:
        """Fold a window's drained values without a second device read.

        The trainer keeps a WINDOW counter and a run-TOTAL counter, and both
        accumulate the same device tensors (`window.merge(step)` and
        `total.merge(step)` sit side by side). `window.drain()` reads them
        once; the total's copy is dropped here and the values are folded
        instead, so a report step costs ONE GPU→CPU transfer, not two. Only
        valid when the two counters hold the same window — which is the
        trainer's construction, not an assumption callers may make.
        """
        extra = set(self._device) - set(drained)
        assert not extra, f"absorb would drop device counters {sorted(extra)}"
        self._device.clear()
        self._fold(drained)

    def drain_if_due(self, step: int, every: int) -> dict[str, int] | None:
        """Drain only on a REPORT step — the cadence owns the sync cost.

        `step` is the 1-based count of completed steps (the trainer's
        `step + 1`). A non-report step returns `None` without touching
        `_device`, so the audit can assert ZERO host reads on it rather than
        trusting a ratio. `every <= 0` is LOUD: a zero cadence would silently
        never report.
        """
        if every <= 0:
            loud(
                f"log_every={every}: a zero/negative cadence never drains the device "
                f"counters, so the log line would silently lose every arm reading. "
                f"Escape: log_every >= 1."
            )
        if step % every != 0:
            return None
        return self.drain()

    def line(self) -> str:
        """The one-line form, deterministic (sorted) so two runs' lines diff."""
        return " ".join(f"{k}={self[k]}" for k in sorted(self))

    def merge(self, other: Counters) -> None:
        for k, v in stepaudit.iterations("counters.merge.host", other.items()):
            if isinstance(v, float):
                self[k] = self.get(k, 0) + v
            else:
                self.count(k, v)
        for k, v in stepaudit.iterations("counters.merge.device", other._device.items()):
            self.count_device(k, v)


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
