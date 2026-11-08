"""`dormouse diagnose`: the failures this repo has paid for, read from one log.

Each row below exists because a run died in a way that only a human reading
the log end to end could see, and each row names the line or counter it read:

* **loss NaN/inf** — the firewall (`losses.py`) skipped a step, or the printed
  loss itself is non-finite; the row names the step it started;
* **LFB divergence** — `lfb_bias range [..] is past ±10` on a load, or the
  in-training `moe_route_max_x1000`/`_top10_x1000` signature
  (`docs/reviews/2026-11-04-routing-collapse-CONFIRMED.md`);
* **dead experts** — mean `moe_route_live / moe_route_n` against `n_experts`;
* **LR frozen/zero** — the printed `lr` stops moving inside the WSD warmup or
  decay window that should move it, or is 0 while steps remain (the
  2026-11-07 `muon_lr_mult` class: a knob overwritten before it is read);
* **dataloader stall** — JSONL `bytes_seen` is short of `step x batch x
  seq_len x grad_accum`, i.e. a step consumed less than its window;
* **eval window overlap** — the held-out file is also a training source
  (`eval_window.training_overlap`, the script's path predicate);
* **checkpoint** — the run finished but the last `ckpt step=` is behind it.

Parsing only: this module imports no torch and builds no model. Its one input
is a file — a run's stdout log or its `--json-log` telemetry — plus, when a
check needs the run's config (expert count, schedule, eval paths), the
checkpoint's own `<ckpt>.config.toml` via `--snapshot`.

`--json` prints the same rows for a script. Exit status is 1 iff a row is
FAIL; WARN and SKIP (no evidence in this log) are not failures.
"""

from __future__ import annotations

import json
import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .eval_window import training_overlap, training_sources

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"

#: The bank's health bars. `0.8` of the bank live is the acceptance line in
#: `scripts/ab_balance.py`; below half of it the bank is not a bank. The top-10
#: share is only meaningful for E > 10 (at E <= 10 the ten busiest ARE the bank).
DEAD_FAIL, DEAD_WARN = 0.5, 0.8
TOP10_FAIL, TOP10_WARN = 0.9, 0.6
MAX_FAIL, MAX_WARN = 0.5, 0.25
#: Consecutive log rows with the same printed LR before it counts as frozen.
#: `lr` prints at `%.3e` (0.1 % resolution), so the schedule must be given at
#: least that much room: see `_frozen`.
FREEZE_ROWS = 3
LR_RESOLUTION = 1e-3

_STEP_LINE = re.compile(r"^step=(\d+)\s+loss=(\S+)\s+lr=(\S+)(?:\s+(.*))?$")
_EVAL_LINE = re.compile(r"^eval step=(\d+)\s+off=(-?\d+)\b.*?\bover (\d+) B")
_CKPT_LINE = re.compile(r"^ckpt step=(\d+) -> (\S+)")
_DONE_LINE = re.compile(r"^done steps=(\d+)")
_LFB_LINE = re.compile(r"lfb_bias range \[\s*([-+0-9.eE]+),\s*([-+0-9.eE]+)\]")
_MIX_LINE = re.compile(r"^mix_plan=(\S+?):")
_HEADER_LINE = re.compile(r"\bbatch=(\d+)\b")
_HDR_KV = re.compile(r"\b(steps|batch|seq|grad_accum|lr|opt)=([^\s│─]+)")
_KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


def _num(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# facts
# ---------------------------------------------------------------------------


@dataclass
class LogRow:
    step: int
    loss: float | None = None
    lr: float | None = None
    counters: dict[str, float] = field(default_factory=dict)
    bytes_seen: int | None = None
    #: The WB's own count for THIS log window (not the run total).
    nan_window: int = 0
    #: The run-total `nan_skipped` when the row carries it (JSONL).
    nan_total: int | None = None
    #: One index per `run` row: a resumed log can go backwards, and a byte
    #: delta across a resume is not a stall.
    segment: int = 0
    #: The segment's `bytes_per_step` (JSONL run row), for the stall check.
    bps: int | None = None


@dataclass
class EvalRow:
    step: int
    offset: int | None = None
    scored: int | None = None


@dataclass
class Facts:
    path: Path
    kind: str = "text"
    lines: int = 0
    parsed: int = 0
    bad_lines: list[int] = field(default_factory=list)
    steps: int | None = None
    batch: int | None = None
    seq_len: int | None = None
    grad_accum: int | None = None
    logs: list[LogRow] = field(default_factory=list)
    evals: list[EvalRow] = field(default_factory=list)
    ckpts: list[tuple[int, str]] = field(default_factory=list)
    done: int | None = None
    #: `(lo, hi, past_the_bound, line)` — the loader's one controller line.
    lfb: list[tuple[float, float, bool, str]] = field(default_factory=list)
    mix_plan: str | None = None


@dataclass
class Snapshot:
    """The few resolved fields a diagnosis needs, from `<ckpt>.config.toml`.

    Deliberately not `RunConfig`: a reader must open an old snapshot without
    the strict resume gate (a diagnosis is a post-mortem, not a continuation).
    Missing keys take the schema default, the same rule a preset does.
    """

    path: Path
    n_experts: int = 1
    data: str = ""
    mix_plan: str = ""
    eval_data: str = ""
    steps: int | None = None
    batch: int = 8
    seq_len: int = 512
    grad_accum: int = 1
    warmup_tokens: int = 0
    decay_frac: float = 0.1
    min_lr_frac: float = 0.1
    ckpt_every: int = 100
    ckpt_name: str = "port"
    out_dir: str = "checkpoints"

    @classmethod
    def read(cls, path: Path) -> Snapshot:
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise ValueError(f"{path}: not a readable checkpoint snapshot ({e})") from e
        if "model" not in raw or "train" not in raw:
            raise ValueError(f"{path}: no [model]/[train] tables — not a checkpoint snapshot")
        model = raw["model"] or {}
        train = raw["train"] or {}
        return cls(
            path=path,
            n_experts=int(model.get("n_experts", 1)),
            data=str(train.get("data", "")),
            mix_plan=str(train.get("mix_plan", "")),
            eval_data=str(train.get("eval_data", "")),
            steps=train.get("steps"),
            batch=int(train.get("batch", 8)),
            seq_len=int(train.get("seq_len", 512)),
            grad_accum=int(train.get("grad_accum", 1)),
            warmup_tokens=int(train.get("warmup_tokens", 0)),
            decay_frac=float(train.get("decay_frac", 0.1)),
            min_lr_frac=float(train.get("min_lr_frac", 0.1)),
            ckpt_every=int(train.get("ckpt_every", 100)),
            ckpt_name=str(train.get("ckpt_name", "port")),
            out_dir=str(train.get("out_dir", "checkpoints")),
        )


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    evidence: str


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def _counters(tail: str) -> dict[str, float]:
    """Numeric `key=value` pairs of a log line's trailing counters.

    A field like `kda=fla,n=1600` is one token whose value is not a number and
    is dropped whole — half-parsing it would invent an `n` counter.
    """
    out: dict[str, float] = {}
    for match in _KV.finditer(tail):
        value = _num(match.group(2))
        if value is not None:
            out[match.group(1)] = value
    return out


def _parse_text(path: Path, text: str) -> Facts:
    facts = Facts(path=path)
    segment = 0
    for line in text.splitlines():
        facts.lines += 1
        line = line.strip()
        if not line:
            continue
        step_match = _STEP_LINE.match(line)
        if step_match is not None:
            step = int(step_match.group(1))
            counters = _counters(step_match.group(4) or "")
            facts.logs.append(
                LogRow(
                    step=step,
                    loss=_num(step_match.group(2)),
                    lr=_num(step_match.group(3)),
                    counters=counters,
                    nan_window=int(counters.get("nan_skipped", 0)),
                    segment=segment,
                )
            )
            facts.parsed += 1
            continue
        eval_match = _EVAL_LINE.match(line)
        if eval_match is not None:
            facts.evals.append(
                EvalRow(
                    step=int(eval_match.group(1)),
                    offset=int(eval_match.group(2)),
                    scored=int(eval_match.group(3)),
                )
            )
            facts.parsed += 1
            continue
        ckpt_match = _CKPT_LINE.match(line)
        if ckpt_match is not None:
            facts.ckpts.append((int(ckpt_match.group(1)), ckpt_match.group(2)))
            facts.parsed += 1
            continue
        done_match = _DONE_LINE.match(line)
        if done_match is not None:
            facts.done = int(done_match.group(1))
            facts.parsed += 1
            continue
        lfb_match = _LFB_LINE.search(line)
        if lfb_match is not None:
            facts.lfb.append(
                (
                    float(lfb_match.group(1)),
                    float(lfb_match.group(2)),
                    "is past" in line,
                    line,
                )
            )
            facts.parsed += 1
            continue
        mix_match = _MIX_LINE.match(line)
        if mix_match is not None:
            facts.mix_plan = mix_match.group(1)
        if _HEADER_LINE.search(line):
            for key, value in _HDR_KV.findall(line):
                number = _num(value)
                if number is None:
                    continue
                if key == "steps":
                    facts.steps = int(number)
                elif key == "batch":
                    facts.batch = int(number)
                elif key == "seq":
                    facts.seq_len = int(number)
                elif key == "grad_accum":
                    facts.grad_accum = int(number)
    return facts


def _bps(batch: int | None, seq_len: int | None, grad_accum: int) -> int | None:
    if not batch or not seq_len:
        return None
    return int(batch) * int(seq_len) * int(grad_accum)


def _parse_jsonl(path: Path, text: str) -> Facts:
    facts = Facts(path=path, kind="jsonl")
    segment = -1
    bps: int | None = None
    for lineno, line in enumerate(text.splitlines(), 1):
        facts.lines += 1
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            facts.bad_lines.append(lineno)
            continue
        if not isinstance(row, dict) or "event" not in row:
            facts.bad_lines.append(lineno)
            continue
        facts.parsed += 1
        event = row["event"]
        if event == "run":
            segment += 1
            facts.steps = row.get("steps", facts.steps)
            facts.batch = row.get("batch", facts.batch)
            facts.seq_len = row.get("seq_len", facts.seq_len)
            facts.grad_accum = row.get("grad_accum", facts.grad_accum)
            bps = row.get("bytes_per_step")
            if bps is None:
                bps = _bps(facts.batch, facts.seq_len, int(facts.grad_accum or 1))
            bps = int(bps) if bps else None
        elif event == "log":
            counters = row.get("counters") or {}
            facts.logs.append(
                LogRow(
                    step=int(row["step"]),
                    loss=row.get("loss"),
                    lr=row.get("lr"),
                    counters={k: float(v) for k, v in counters.items() if _num(str(v)) is not None},
                    bytes_seen=row.get("bytes_seen"),
                    nan_window=int(counters.get("nan_skipped", 0) or 0),
                    nan_total=row.get("nan_skipped"),
                    segment=segment,
                    bps=bps if bps is not None else row.get("bytes_per_step"),
                )
            )
        elif event == "eval":
            facts.evals.append(
                EvalRow(
                    step=int(row["step"]),
                    offset=row.get("eval_offset"),
                    scored=row.get("scored"),
                )
            )
        elif event == "ckpt":
            facts.ckpts.append((int(row["step"]), str(row.get("path", "?"))))
    return facts


def parse(path: Path) -> Facts:
    """A stdout log or a `--json-log` file — chosen by the first non-blank line."""
    text = path.read_text(encoding="utf-8", errors="replace")
    first = next((line for line in text.splitlines() if line.strip()), "")
    if first.lstrip().startswith("{"):
        return _parse_jsonl(path, text)
    return _parse_text(path, text)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def _check_readable(facts: Facts) -> Check:
    if facts.parsed == 0:
        return Check(
            "log readable",
            FAIL,
            f"{facts.path}: no run/log/eval/ckpt line recognized in {facts.lines} lines "
            f"— is this a training log?",
        )
    if facts.bad_lines:
        where = ", ".join(str(n) for n in facts.bad_lines[:3])
        if facts.bad_lines == [facts.lines]:
            return Check(
                "log readable",
                WARN,
                f"{facts.kind}: the LAST line ({where}) is truncated — a killed writer, "
                f"or a log cut mid-flush",
            )
        return Check(
            "log readable",
            FAIL,
            f"{facts.kind}: line(s) {where} are not JSON rows — part of the history "
            f"would be silently dropped",
        )
    return Check(
        "log readable",
        PASS,
        f"{facts.kind} · {facts.parsed} record(s) · steps "
        f"{facts.logs[0].step if facts.logs else '?'}.."
        f"{facts.logs[-1].step if facts.logs else '?'}",
    )


def _check_loss(facts: Facts) -> Check:
    if not facts.logs:
        return Check("loss finite", SKIP, "no `step=` log rows")
    bad = [row for row in facts.logs if row.loss is not None and not math.isfinite(row.loss)]
    if bad:
        row = bad[0]
        return Check(
            "loss finite",
            FAIL,
            f"step {row.step}: loss={row.loss} — the objective went non-finite; "
            f"the firewall skips the update, it cannot restore a finite loss",
        )
    skips = [row for row in facts.logs if row.nan_window > 0]
    if skips:
        totals = [row.nan_total for row in facts.logs if row.nan_total is not None]
        total = totals[-1] if totals else None
        run = longest = 1
        for prev, cur in zip(facts.logs, facts.logs[1:], strict=False):
            run = run + 1 if cur.nan_window > 0 and prev.nan_window > 0 else 1
            longest = max(longest, run)
        first = skips[0]
        count = f"nan_skipped={first.nan_window} in window ending step {first.step}"
        if total is not None:
            count += f" ({total} skipped in total)"
        if longest >= 2 or sum(row.nan_window for row in skips) >= 10:
            return Check(
                "loss finite",
                FAIL,
                f"{count}; {longest} consecutive window(s) skipped — a drowning run "
                f"must stop for a human (readiness plan nan_windows>=2)",
            )
        return Check(
            "loss finite",
            WARN,
            f"{count}; the run recovered, but a skipped update is a COUNTED defect "
            f"that wants the raw loss and the step in the report",
        )
    counted = any("nan_skipped" in row.counters or row.nan_total is not None for row in facts.logs)
    return Check(
        "loss finite",
        PASS,
        f"{len(facts.logs)} log row(s), every printed loss finite"
        + (", nan_skipped=0" if counted else " (the log carries no nan_skipped counter)"),
    )


def _shares(row: LogRow) -> tuple[float, float, float] | None:
    """`(live_per_forward, max_share, top10_share)` of a window, or `None`."""
    n = row.counters.get("moe_route_n", 0.0)
    live = row.counters.get("moe_route_live")
    if n <= 0 or live is None:
        return None
    top10 = row.counters.get("moe_route_top10_x1000")
    return (
        live / n,
        row.counters.get("moe_route_max_x1000", 0.0) / n / 1000.0,
        (top10 / n / 1000.0) if top10 is not None else 0.0,
    )


def _check_lfb(facts: Facts, snap: Snapshot | None) -> Check:
    for lo, hi, past, line in facts.lfb:
        if past or max(abs(lo), abs(hi)) > 10.0:
            return Check(
                "lfb divergence",
                FAIL,
                f"{line[:160]} — the controller's bias is outside the router's own "
                f"logit scale (±5); the file's routing is collapsed",
            )
    routed = [(row, _shares(row)) for row in facts.logs]
    routed = [(row, share) for row, share in routed if share is not None]
    if routed:
        worst_row, worst = max(routed, key=lambda pair: pair[1][1])
        _live, max_share, top10 = worst
        experts = snap.n_experts if snap else None
        if max_share >= MAX_FAIL or (experts and experts > 10 and top10 >= TOP10_FAIL):
            return Check(
                "lfb divergence",
                FAIL,
                f"step {worst_row.step}: busiest expert holds {max_share:.0%}, top-10 "
                f"{top10:.0%} (moe_route_max_x1000/_top10_x1000 over moe_route_n) — the "
                f"collapsed-router signature of 2026-11-04",
            )
        if max_share >= MAX_WARN or (experts and experts > 10 and top10 >= TOP10_WARN):
            return Check(
                "lfb divergence",
                WARN,
                f"step {worst_row.step}: busiest expert holds {max_share:.0%}, top-10 "
                f"{top10:.0%} — past the 2026-11-04 acceptance bar; watch the bias",
            )
        if facts.lfb:
            lo, hi, _past, _line = facts.lfb[-1]
            return Check("lfb divergence", PASS, f"loaded bias range [{lo:+.3f}, {hi:+.3f}]")
        return Check(
            "lfb divergence",
            PASS,
            f"routing stays spread (worst window step {worst_row.step}: max "
            f"{max_share:.0%}, top-10 {top10:.0%})",
        )
    if facts.lfb:
        lo, hi, _past, _line = facts.lfb[-1]
        return Check("lfb divergence", PASS, f"loaded bias range [{lo:+.3f}, {hi:+.3f}]")
    return Check(
        "lfb divergence",
        SKIP,
        "no `lfb_bias range` line and no moe_route_* counters — the loader only "
        "prints the range on a checkpoint load, and `moe_routing_health` is off",
    )


def _check_dead(facts: Facts, snap: Snapshot | None) -> Check:
    routed = [(row, _shares(row)) for row in facts.logs]
    routed = [(row, share) for row, share in routed if share is not None]
    if not routed:
        if snap and snap.n_experts > 1:
            return Check(
                "dead experts",
                WARN,
                f"n_experts={snap.n_experts} but moe_routing_health=false: the bank's "
                f"live count is not counted, so a collapse would be invisible — re-run "
                f"with --set moe_routing_health=true",
            )
        return Check("dead experts", SKIP, "no moe_route_live/moe_route_n counters")
    if snap is None or snap.n_experts <= 1:
        return Check(
            "dead experts",
            SKIP,
            "moe_route_* counters present, but the bank size is not in the log — "
            "pass --snapshot <ckpt>.config.toml for n_experts",
        )
    worst_row, worst = min(routed, key=lambda pair: pair[1][0])
    live, _max_share, _top10 = worst
    frac = live / snap.n_experts
    evidence = (
        f"step {worst_row.step}: live {live:.1f}/{snap.n_experts} ({frac:.1%}) "
        f"(moe_route_live/moe_route_n)"
    )
    if frac < DEAD_FAIL:
        return Check(
            "dead experts",
            FAIL,
            f"{evidence} — under half the bank receives no token and no gradient; "
            f"no optimizer step revives it",
        )
    if frac < DEAD_WARN:
        return Check(
            "dead experts",
            WARN,
            f"{evidence} — under the 80 % acceptance line of ab_balance",
        )
    return Check("dead experts", PASS, evidence)


def _frozen(rows: list[LogRow], lo: int, hi: int) -> tuple[int, LogRow, LogRow]:
    """The longest run of identical `lr` inside `[lo, hi]` (at least one row)."""
    seq = [row for row in rows if lo <= row.step <= hi and row.lr is not None]
    best: tuple[int, LogRow, LogRow] = (0, seq[0], seq[0]) if seq else (0, rows[0], rows[0])
    i = 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1].lr == seq[i].lr:
            j += 1
        if j - i + 1 > best[0]:
            best = (j - i + 1, seq[i], seq[j])
        i = j + 1
    return best


def _check_lr(facts: Facts, snap: Snapshot | None) -> Check:
    rows = [row for row in facts.logs if row.lr is not None]
    if len(rows) < 2:
        return Check("lr schedule", SKIP, "fewer than two log rows carry `lr`")
    end = facts.steps or (snap.steps if snap else None) or rows[-1].step
    zeros = [row for row in rows if row.lr == 0.0]
    if zeros:
        expected = snap is not None and snap.min_lr_frac == 0.0 and zeros[-1].step >= end
        if expected and all(row.step >= end for row in zeros):
            return Check(
                "lr schedule",
                PASS,
                f"lr=0 at the final step ({end}) with min_lr_frac=0 — the full cooldown",
            )
        if any(row.step < end for row in zeros):
            row = next(row for row in zeros if row.step < end)
            return Check(
                "lr schedule",
                FAIL,
                f"step {row.step}: lr=0 while the run declares {end} steps — every "
                f"remaining step is a no-op update",
            )
        return Check(
            "lr schedule",
            WARN,
            f"lr=0 at the last log row (step {rows[-1].step}); legal only as a full "
            f"cooldown (min_lr_frac=0) — pass --snapshot to decide",
        )

    if snap is None:
        if len({row.lr for row in rows}) == 1:
            return Check(
                "lr schedule",
                WARN,
                f"lr={rows[-1].lr:.3e} on all {len(rows)} log rows; a constant "
                f"schedule is legal, and this is the shape of the 2026-11-07 frozen "
                f"schedule — pass --snapshot to separate the two",
            )
        return Check(
            "lr schedule",
            PASS,
            f"lr moved {rows[0].lr:.3e} → {rows[-1].lr:.3e} over {len(rows)} rows",
        )

    # The log's own header is what RAN; the snapshot is the fallback for a
    # truncated log (the resume gate pins batch/seq_len, so they should agree).
    batch = facts.batch or snap.batch
    seq_len = facts.seq_len or snap.seq_len
    grad = facts.grad_accum or snap.grad_accum
    if min(batch, seq_len, grad) <= 0:
        return Check(
            "lr schedule",
            SKIP,
            f"snapshot names batch/seq_len/grad_accum={batch}/{seq_len}/{grad}",
        )
    warm = math.ceil(snap.warmup_tokens / (batch * grad * seq_len)) if snap.warmup_tokens else 0
    decay = int(round(snap.decay_frac * end))
    decay_start = max(end - decay, warm)
    phases = [
        ("warmup", 0, warm, 1.0),
        ("decay", decay_start, end, 1.0 - snap.min_lr_frac),
    ]
    for name, lo, hi, scale in phases:
        length = hi - lo
        if length <= 0 or scale <= 0:
            continue
        found = _frozen(rows, lo, hi)
        if found[0] < FREEZE_ROWS:
            continue
        _count, first, last = found
        movement = (last.step - first.step) / length * scale
        if movement <= LR_RESOLUTION:
            continue
        return Check(
            "lr schedule",
            FAIL,
            f"lr={first.lr:.3e} is identical at steps {first.step}..{last.step}, inside "
            f"the {name} window [{lo}, {hi}] where it should have moved "
            f"{movement:.1%} — the schedule is written but not reaching the step",
        )
    return Check(
        "lr schedule",
        PASS,
        f"lr {rows[0].lr:.3e} → {rows[-1].lr:.3e}; no frozen stretch inside warmup "
        f"[0, {warm}] or decay [{decay_start}, {end}]",
    )


def _check_stall(facts: Facts, snap: Snapshot | None) -> Check:
    if facts.kind != "jsonl":
        return Check(
            "dataloader stall",
            SKIP,
            "the plain log prints no `bytes_seen`; the telemetry sink does "
            "(--json-log), and that row is the byte count",
        )
    rows = [row for row in facts.logs if row.bytes_seen is not None]
    if len(rows) < 1:
        return Check("dataloader stall", SKIP, "no `bytes_seen` in the log rows")
    short = None
    long = None
    checked = 0
    for row in rows:
        bps = row.bps or (snap and _bps(snap.batch, snap.seq_len, snap.grad_accum))
        if not bps:
            continue
        checked += 1
        expected = row.step * bps
        if row.bytes_seen < expected:
            short = (row, expected)
            break
        if row.bytes_seen > expected:
            long = (row, expected)
    if not checked:
        return Check(
            "dataloader stall",
            SKIP,
            "no batch x seq_len x grad_accum on the run row, so `bytes_seen` cannot be checked",
        )
    if short is not None:
        row, expected = short
        return Check(
            "dataloader stall",
            FAIL,
            f"step {row.step}: bytes_seen={row.bytes_seen:,} against the step's own "
            f"arithmetic {expected:,} (step x batch x seq_len x grad_accum) — the "
            f"loader fed the step less than its window",
        )
    if long is not None:
        row, expected = long
        return Check(
            "dataloader stall",
            WARN,
            f"step {row.step}: bytes_seen={row.bytes_seen:,} is ABOVE {expected:,} — "
            f"the step counted the same bytes twice (a resume here double-counts)",
        )
    return Check(
        "dataloader stall",
        PASS,
        f"{len(rows)} row(s): bytes_seen == step x batch x seq_len x grad_accum",
    )


def _check_overlap(facts: Facts, snap: Snapshot | None) -> Check:
    if not facts.evals:
        return Check("eval window overlap", SKIP, "no held-out `eval step=` line")
    if snap is None:
        return Check(
            "eval window overlap",
            SKIP,
            "the eval's file is in the run's config, not the log — pass "
            "--snapshot <ckpt>.config.toml",
        )
    if not snap.eval_data:
        return Check(
            "eval window overlap",
            PASS,
            "eval_data is unset: the eval is the tail split of the training file, "
            "disjoint by construction",
        )
    plan = snap.mix_plan or facts.mix_plan or ""
    try:
        hit = training_overlap(snap.data, plan, Path(snap.eval_data))
        sources = len(training_sources(snap.data, plan))
    except (OSError, tomllib.TOMLDecodeError) as e:
        return Check(
            "eval window overlap",
            WARN,
            f"mix plan {plan!r} could not be read ({e}) — the eval file's overlap "
            f"with a plan source is unverified",
        )
    if hit is not None:
        return Check(
            "eval window overlap",
            FAIL,
            f"--eval file {snap.eval_data} is ALSO a training source ({hit}) — the "
            f"window is not held out, so its BPB is not a measurement of unseen text",
        )
    return Check(
        "eval window overlap",
        PASS,
        f"{snap.eval_data} is not one of {sources} training source(s)"
        + (f" (plan {plan})" if plan else ""),
    )


def _check_ckpt(facts: Facts, snap: Snapshot | None) -> Check:
    """The LAST checkpoint the cadence owes, not simply the last logged step.

    The trainer writes only on `(step + 1) % ckpt_every == 0`, so a run of
    8000 steps at `ckpt_every=3000` owes its last file at 6000 — "behind the
    end" alone would fail a healthy run.
    """
    declared = facts.steps or (snap.steps if snap else None)
    end = facts.done or (facts.logs[-1].step if facts.logs else None) or declared
    cadence = snap.ckpt_every if snap is not None else None
    expected = (declared // cadence) * cadence if declared is not None and cadence else None
    finished = facts.done is not None or (
        declared is not None and end is not None and end >= declared
    )
    if facts.ckpts:
        step, where = max(facts.ckpts)
        due = expected if expected is not None else end
        if due is None:
            return Check("checkpoint", WARN, f"ckpt step={step} -> {where}; no run end known")
        if step >= due:
            return Check("checkpoint", PASS, f"ckpt step={step} -> {where}")
        if finished:
            return Check(
                "checkpoint",
                FAIL,
                f"the run finished at step {declared or end} but the last `ckpt step=` is "
                f"{step} ({due - step} step(s) never written) -> {where}",
            )
        return Check(
            "checkpoint",
            WARN,
            f"the log stops at step {end}; the last checkpoint is step {step} -> {where} "
            f"({end - step} step(s) rolled back on resume)",
        )
    if expected == 0:
        return Check(
            "checkpoint",
            PASS,
            f"no checkpoint was due (ckpt_every={cadence} > steps={declared})",
        )
    if facts.kind == "jsonl":
        return Check(
            "checkpoint",
            SKIP,
            "the telemetry sink records run/log/eval rows only — no checkpoint event; "
            "pass the stdout log for this check",
        )
    if expected is not None:
        return Check(
            "checkpoint",
            FAIL,
            f"no `ckpt step=` line in a run of {declared} steps at ckpt_every={cadence} "
            f"(the last file was due at step {expected}): the weights were never written",
        )
    if facts.done is not None:
        return Check(
            "checkpoint",
            WARN,
            f"`done steps={facts.done}` with no `ckpt step=` line — if ckpt_every <= "
            f"{facts.done}, the run's weights are only in memory (pass --snapshot to decide)",
        )
    return Check(
        "checkpoint",
        WARN,
        "no `ckpt step=` line and no `done` line — the log ends before the run does, "
        "so a checkpoint is unverified",
    )


def judge(facts: Facts, snap: Snapshot | None) -> list[Check]:
    """The seven pathologies, each read from its own evidence."""
    return [
        _check_readable(facts),
        _check_loss(facts),
        _check_lfb(facts, snap),
        _check_dead(facts, snap),
        _check_lr(facts, snap),
        _check_stall(facts, snap),
        _check_overlap(facts, snap),
        _check_ckpt(facts, snap),
    ]


def inspect(path: Path, snapshot: Path | None = None) -> list[Check]:
    """The report rows for `path`; the CLI's `--snapshot` is a config.toml path."""
    facts = parse(path)
    snap = Snapshot.read(snapshot) if snapshot is not None else None
    return judge(facts, snap)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

_STATUS_STYLE = {PASS: "green", WARN: "yellow", FAIL: "bold red", SKIP: "dim"}


def render(
    path: Path,
    snapshot: Path | None,
    checks: list[Check],
    *,
    as_json: bool = False,
    console: Console | None = None,
) -> None:
    counts = {
        status: sum(c.status == status for c in checks) for status in (PASS, WARN, FAIL, SKIP)
    }
    if as_json:
        print(
            json.dumps(
                {
                    "input": str(path),
                    "snapshot": str(snapshot) if snapshot is not None else None,
                    "counts": counts,
                    "checks": [
                        {"check": c.name, "status": c.status, "evidence": c.evidence}
                        for c in checks
                    ],
                    "fail": counts[FAIL],
                },
                indent=2,
            )
        )
        return
    console = console or Console()
    table = Table(
        title=f"dormouse diagnose · {path}",
        title_style="bold",
        show_edge=True,
        pad_edge=False,
    )
    table.add_column("check", style="cyan", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("evidence", overflow="fold")
    for check in checks:
        table.add_row(
            check.name,
            f"[{_STATUS_STYLE[check.status]}]{check.status}[/]",
            check.evidence,
        )
    console.print(table)
    summary = (
        f"[green]{counts[PASS]} pass[/] · [yellow]{counts[WARN]} warn[/] · "
        f"[bold red]{counts[FAIL]} fail[/] · [dim]{counts[SKIP]} skip[/]"
    )
    if snapshot is not None:
        summary += f" · snapshot {snapshot}"
    console.print(summary)


def run(path: Path, snapshot: Path | None = None, *, as_json: bool = False) -> int:
    """Print the report; return 1 iff any row is FAIL, 2 on an unreadable input."""
    try:
        checks = inspect(path, snapshot)
    except (OSError, ValueError) as e:
        print(f"diagnose: {e}")
        return 2
    render(path, snapshot, checks, as_json=as_json)
    return 1 if any(c.status == FAIL for c in checks) else 0
