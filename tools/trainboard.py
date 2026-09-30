#!/usr/bin/env python3
"""trainboard — turn a dormouse training log into a dark dashboard of PNGs.

Provenance: parses the **2026-09-30 trainer stdout format** —
`dormouse pretrain <preset> params=… `, `step N ce=… bpb=… best=… lr=… aux=…`,
`step N EVAL ce=… bpb=… BEST over <bytes> B (fixed window) fused kda=…/…`,
`timer step N: total=…ms data=…ms fwd=…ms bwd=…ms opt=…ms retr=…ms ema=…ms
gpu_step=…ms`, `firewall: N non-finite loss read(s)`, `guard: restarting in …`,
`thread 'main' panicked at …`, `train failed: …`. Unknown lines are ignored;
a line that LOOKS like one of those but does not parse is counted, not fatal.

Usage
  tools/trainboard.py ~/logs/first_run_500_0930_1843.log          # one-shot board
  tools/trainboard.py ~/logs/first_run_2000_*.log --follow 15     # keep it live

Only dependency is matplotlib. Everything else is stdlib.
"""
from __future__ import annotations

import argparse
import html
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

FORMAT_VERSION = "2026-09-30"  # log format this script was written against

# ── one dark palette, one accent family (cool ramp) + amber for "the sum" ──
BG, PANEL, GRID = "#0e1117", "#161b22", "#2a313c"
FG, DIM = "#c9d1d9", "#8b949e"
C_CE, C_BPB, C_AUX = "#5eead4", "#38bdf8", "#818cf8"
C_EVAL = "#f472b6"
C_TOTAL, C_BEST = "#fbbf24", "#fbbf24"
STACK = ["#5eead4", "#38bdf8", "#818cf8", "#c084fc", "#e879f9"]  # fwd bwd opt retr ema
C_OTHER = "#484f58"
EV_STYLE = {
    "nan": ("#fb7185", "NaN"),
    "panic": ("#f97316", "panic"),
    "restart": ("#fbbf24", "restart"),
    "fail": ("#ef4444", "fail"),
}

RC = {
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "savefig.facecolor": BG,
    "text.color": FG,
    "axes.labelcolor": FG,
    "axes.edgecolor": GRID,
    "xtick.color": DIM,
    "ytick.color": DIM,
    "grid.color": GRID,
    "grid.alpha": 0.45,
    "grid.linewidth": 0.7,
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "axes.axisbelow": True,
}

STEP_RE = re.compile(r"^step\s+(\d+)\s+(.*)$")
TIMER_RE = re.compile(r"^timer\s+step\s+(\d+):\s*(.*)$")
KV_RE = re.compile(r"([a-z_][a-z0-9_]*)=(\S+)")
MS_RE = re.compile(r"([a-z_][a-z0-9_]*)=([0-9]*\.?[0-9]+(?:e-?\d+)?)(ms|s)\b")
WINDOW_RE = re.compile(r"over\s+(\d+)\s*B")
FLOATS = ("ce", "bpb", "best", "lr", "aux")


# ─────────────────────────────── parse ───────────────────────────────
@dataclass
class StepPt:
    step: int
    sess: int = 0
    ce: float | None = None
    bpb: float | None = None
    best: float | None = None
    lr: float | None = None
    aux: float | None = None
    seam: str = ""


@dataclass
class EvalPt:
    step: int
    sess: int
    ce: float | None
    bpb: float | None
    window: int | None
    is_best: bool
    seam: str


@dataclass
class TimerPt:
    step: int
    sess: int
    ms: dict = field(default_factory=dict)

    def get(self, k):
        v = self.ms.get(k)
        return v if v is not None else 0.0


@dataclass
class Event:
    step: int
    kind: str
    text: str


@dataclass
class Run:
    path: Path
    header: dict = field(default_factory=dict)
    quant: str = ""
    optim: str = ""
    steps: list = field(default_factory=list)
    evals: list = field(default_factory=list)
    timers: list = field(default_factory=list)
    events: list = field(default_factory=list)
    done: str = ""
    headers: int = 0
    session: int = -1
    skipped: int = 0
    skip_examples: list = field(default_factory=list)
    lines: int = 0

    @property
    def name(self):
        return self.path.name

    @property
    def preset(self):
        return self.header.get("preset", "?")

    @property
    def params(self):
        return int(self.header["params"]) if str(self.header.get("params", "")).isdigit() else None

    def subtitle(self):
        bits = [self.preset]
        p = self.params
        bits.append(f"{p:,} params" if p else "params ?")
        s = self.header.get("steps")
        if s:
            bits.append(f"{s} steps planned")
        if self.done:
            # the done line names a checkpoint path; the full text is in index.html
            bits.append(self.done if len(self.done) <= 64 else self.done[:61] + "…")
        return " · ".join(bits)

    def x(self, pts, attr="step"):
        return [getattr(p, attr) for p in pts]

    def col(self, pts, attr):
        """(xs, ys) for one field, GAPS DROPPED — so len(xs) differs per field.

        Plot each field against its OWN xs. Pairing one field's ys with another
        field's xs is a crash the first time a log has a truncated line, and a
        silently shifted curve every time before that.
        """
        xs, ys = [], []
        for p in pts:
            v = getattr(p, attr)
            if v is not None and math.isfinite(v):
                xs.append(p.step)
                ys.append(v)
        return xs, ys


def _f(v):
    """float or None. Never 0-for-missing: a gap is None (ADR-0019: no silent fill)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def parse_log(path: Path) -> Run:
    run = Run(path=path)
    last_step = -1
    try:
        text = path.read_text(errors="replace")
    except OSError as e:
        print(f"trainboard: cannot read {path}: {e}", file=sys.stderr)
        return run
    for lineno, line in enumerate(text.splitlines(), 1):
        run.lines = lineno
        s = line.strip()
        if not s:
            continue

        m = TIMER_RE.match(s)
        if m:
            vals = {}
            for k, v, unit in MS_RE.findall(m.group(2)):
                vals[k] = float(v) * (1000.0 if unit == "s" else 1.0)
            if not vals:
                _skip(run, lineno, s)
                continue
            run.timers.append(TimerPt(int(m.group(1)), run.session, vals))
            last_step = max(last_step, int(m.group(1)))
            continue

        m = STEP_RE.match(s)
        if m:
            step, rest = int(m.group(1)), m.group(2)
            kv = dict(KV_RE.findall(rest))
            if not kv:
                _skip(run, lineno, s)
                continue
            is_eval = re.search(r"\bEVAL\b", rest) is not None
            # `ce=NaN` / `bpb=inf` is the firewall firing, not corruption: it is an
            # EVENT (marked on every time chart), and the point is dropped because
            # there is no number to plot. Counted as skipped too, so the stderr
            # count and the plot never disagree about what was left out.
            nonfinite = [k for k in FLOATS if k in kv and _f(kv[k]) is None
                         and kv[k].lower().lstrip("+-") in ("nan", "inf", "infinity")]
            # A field that is ABSENT is a gap (None). A field that is PRESENT and
            # does not parse at all is corruption: skip the line and count it,
            # never plot it as 0 and never drop it silently.
            corrupt = [k for k in FLOATS if k in kv and _f(kv[k]) is None and k not in nonfinite]
            if nonfinite:
                run.events.append(Event(step, "nan",
                                        f"step {step}: non-finite "
                                        + ", ".join(f"{k}={kv[k]}" for k in nonfinite)
                                        + ("  (EVAL)" if is_eval else "")))
            if corrupt or all(_f(kv.get(k)) is None for k in FLOATS if k in kv):
                _skip(run, lineno, s)
                continue
            vals = {k: _f(kv.get(k)) for k in FLOATS}
            if is_eval:
                w = WINDOW_RE.search(rest)
                seam = _seam(rest)
                run.evals.append(
                    EvalPt(step, run.session, vals["ce"], vals["bpb"],
                           int(w.group(1)) if w else None,
                           re.search(r"\bBEST\b", rest) is not None, seam)
                )
            else:
                run.steps.append(StepPt(step, run.session, vals["ce"], vals["bpb"], vals["best"],
                                        vals["lr"], vals["aux"], _seam(rest)))
            last_step = max(last_step, step)
            continue

        if s.startswith("dormouse pretrain "):
            parts = s.split()
            run.header = {"preset": parts[2] if len(parts) > 2 else "?",
                          **{k: v.strip('"') for k, v in KV_RE.findall(s)}}
            # backend is space-delimited (`backend=Device<Autodiff { … }>`): KV_RE
            # truncates it at the first space, so take it to end of line or it is a lie.
            b = re.search(r"\bbackend=(.*)$", s)
            if b:
                run.header["backend"] = b.group(1).strip()
            run.headers += 1
            run.session += 1  # a restart resets `step`; the log holds several sessions
        elif s.startswith("quant format:"):
            run.quant = s.split(":", 1)[1].strip()
        elif s.startswith("optimizer:"):
            run.optim = s.split(":", 1)[1].strip()
        elif s.startswith("firewall:"):
            n = re.search(r"(\d+)", s)
            run.events.append(Event(last_step, "nan",
                                    s.strip().replace("  ", "")))
            _ = n
        elif re.search(r"\d+ non-finite loss reads in one log window", s):
            run.events.append(Event(last_step, "nan", s.strip()))
        elif s.startswith("guard: "):
            kind = "restart" if "restarting" in s else "fail"
            run.events.append(Event(last_step, kind, s.strip()))
        elif s.startswith("train failed:"):
            run.events.append(Event(last_step, "fail", s.strip()[:200]))
        elif "panicked at" in s:
            run.events.append(Event(last_step, "panic", s.strip()[:200]))
        elif s.startswith("done "):
            run.done = s[5:].strip()
    return run


def _seam(rest):
    """The counter tail of a step/EVAL line: `fused kda=2012/0 asked=4096 engram=0/0`.

    `\\b` on the key is load-bearing: without it `bwd=` would eat `node_bwd=`.
    """
    t = re.sub(r"\b(?:EVAL|BEST)\b", " ", rest)
    t = re.sub(r"over\s+\d+\s*B\s*\(fixed window\)", " ", t)
    for k in FLOATS:
        t = re.sub(rf"\b{k}=(\S+)", " ", t)
    return " ".join(t.split())[:200]


def _skip(run, lineno, line):
    run.skipped += 1
    if len(run.skip_examples) < 3:
        run.skip_examples.append(f"{lineno}: {line[:90]}")


# ─────────────────────────────── charts ───────────────────────────────
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)
    return plt


def _fignote(fig, run):
    try:
        mt = datetime.fromtimestamp(run.path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        mt = "mtime ?"
    # constrained_layout knows nothing about fig.text, so the note lands on the
    # xlabel unless the layout is told to leave the strip for it.
    try:
        # top 7% for the two-line title, bottom 3.5% for the note
        fig.get_layout_engine().set(rect=(0, 0.035, 1, 0.93))
    except AttributeError:
        pass
    fig.text(0.008, 0.008,
             f"{run.name} · mtime {mt} · {len(run.steps)} step / {len(run.evals)} eval / "
             f"{len(run.timers)} timer lines · {run.skipped} skipped · "
             f"format {FORMAT_VERSION} · tools/trainboard.py",
             color=DIM, fontsize=7.5, family="monospace")


def _events(ax, run):
    """Vertical marks at the step each event happened.

    One mark per STEP, not per event: pretrain_v2.log has three `guard: restarting`
    at step 0 and drawing three identical vlines puts three rotated labels on the
    same pixels. Kinds are stacked in the label instead, and the legend carries the
    counts.
    """
    by_step = {}
    for e in run.events:
        by_step.setdefault(e.step, []).append(e)
    for step, evs in by_step.items():
        kinds = []
        for e in evs:
            if e.kind not in kinds:
                kinds.append(e.kind)
        col = EV_STYLE.get(kinds[0], ("#ef4444", kinds[0]))[0]
        n = len(evs)
        ax.axvline(step, color=col, ls=(0, (4, 3)), lw=1.1, alpha=0.55, zorder=1)
        txt = "+".join(EV_STYLE.get(k, ("", k))[1] for k in kinds)
        if n > 1:
            txt += f" ×{n}"
        ax.annotate(f" {txt}@{step}", (step, ax.get_ylim()[1]), color=col,
                    fontsize=8, va="top", ha="left", rotation=90, zorder=6)


def _legend_seen(ax, run):
    """One legend entry per event kind present, so marks are readable."""
    seen = {}
    for e in run.events:
        seen.setdefault(e.kind, EV_STYLE.get(e.kind, ("#ef4444", e.kind))[0])
    if seen:
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color=c, ls=(0, (4, 3)), lw=1.2, label=f"{k} ×{sum(1 for e in run.events if e.kind == k)}")
                           for k, c in seen.items()],
                  loc="upper right", ncol=min(4, len(seen)))


def _autosymlog(ax, ys):
    f = [y for y in ys if y is not None and math.isfinite(y) and y > 0]
    return bool(f) and (max(f) / min(f)) > 3.0


def _sessions(pts):
    """Consecutive runs of the same session id.

    A log with a guard restart holds MORE THAN ONE session and `step` resets, so
    joining them would draw a ramp from step 1500 back through step 1000 as if it
    were one series (official_v5e.log, 2 headers; official_v5g.log, 17). Each
    session gets its own polyline instead. Returns [(sess_id, [pts])].
    """
    out = []
    for p in pts:
        if out and out[-1][0] == p.sess:
            out[-1][1].append(p)
        else:
            out.append((p.sess, [p]))
    return out


def _sess_note(ax, sess, xstep, colour):
    for i, (sid, pts) in enumerate(sess):
        if i:
            ax.axvline(pts[0].step, color=colour, ls=(0, (1, 2)), lw=1.0, alpha=0.5)
            ax.annotate(f"restart · session {sid + 1} from step {pts[0].step}",
                        (pts[0].step, 0.02), xycoords=("data", "axes fraction"),
                        color=colour, fontsize=8.5, va="bottom", rotation=90, ha="right")
    _ = xstep


def chart_loss(run, out: Path, logy=True):
    plt = _mpl()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 8.4), height_ratios=[3, 1],
                                  sharex=True, constrained_layout=True)
    allx, allce = run.col(run.steps, "ce")
    # A one-point "curve" (pretrain_v2.log: two sessions, one step each) is not a
    # chart — the log never trained. Say so instead of drawing a line between two
    # unrelated initialisations.
    degenerate = len({s.step for s in run.steps}) < 2
    if degenerate:
        ax.text(0.5, 0.62, f"only {len(run.steps)} step line(s) at {len({s.step for s in run.steps})} "
                            f"distinct step(s) — this log never trained past step 0",
                transform=ax.transAxes, ha="center", color=DIM, fontsize=12)
    if allce:
        for i, (_sid, part) in enumerate(_sessions(run.steps)):
            first = i == 0
            # each field against its OWN xs: a gap in one field must not shift or
            # truncate another field's curve
            xs_c, ce = run.col(part, "ce")
            if xs_c:
                ax.plot(xs_c, ce, color=C_CE, lw=1.9, marker="o", ms=3, mec=BG, mew=0.5,
                        label="train CE  (nats/byte)" if first else None)
            xs_b, bpb = run.col(part, "bpb")
            if xs_b:
                ax.plot(xs_b, bpb, color=C_BPB, lw=1.4, alpha=0.9, marker="o", ms=3,
                        mec=BG, mew=0.5, label="train BPB (log10 bytes)" if first else None)
            xs_a, aux = run.col(part, "aux")
            if xs_a:
                ax2.plot(xs_a, aux, color=C_AUX, lw=1.2, marker="o", ms=3, mec=BG, mew=0.5,
                         label="aux objective" if first else None)
        b = min(range(len(allce)), key=allce.__getitem__)
        ax.plot([allx[b]], [allce[b]], marker="*", ms=17, color=C_BEST,
                mec=BG, mew=0.8, zorder=5, ls="none",
                label=f"best CE {allce[b]:.3f} @ step {allx[b]}")
        if logy and _autosymlog(ax, allce + run.col(run.steps, "bpb")[1]):
            ax.set_yscale("log")
        ax.annotate(f"{allce[b]:.3f}", (allx[b], allce[b]), textcoords="offset points",
                    xytext=(0, 14), color=C_BEST, fontsize=9.5, ha="center", weight="bold")
        _sess_note(ax, _sessions(run.steps), allx[0], DIM)
    else:
        ax.text(0.5, 0.5, "no `step N ce=` line in this log", transform=ax.transAxes,
                ha="center", color=DIM, fontsize=13)
        ax2.text(0.5, 0.5, "no aux", transform=ax2.transAxes, ha="center", color=DIM, fontsize=11)
        ax.set_yticks([])
        ax2.set_yticks([])
    ax.set_title(f"TRAIN LOSS — {run.name}\n{run.subtitle()}", loc="left", pad=12)
    ax.set_ylabel("CE / BPB" + ("  (log scale)" if logy else ""))
    ax2.set_ylabel("aux")
    ax2.set_xlabel("training step")
    for a in (ax, ax2):
        a.grid(True, axis="y")
        if allce:
            _events(a, run)
            _legend_seen(a, run)
    if allce:
        ax.legend(loc="upper right")
        ax2.legend(loc="upper right")
    _fignote(fig, run)
    p = out / "loss.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def chart_eval(run, out: Path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(13, 6.0), constrained_layout=True)
    exs, _ebpb = run.col(run.evals, "bpb")
    _txs, tbpb = run.col(run.steps, "bpb")
    wins = sorted({e.window for e in run.evals if e.window})
    if exs:
        for i, (_sid, part) in enumerate(_sessions(run.evals)):
            xs, ys = run.col(part, "bpb")
            ax.plot(xs, ys, color=C_EVAL, lw=2.0, marker="H", ms=10,
                    mec=BG, mew=1.2, label="held-out BPB (EVAL)" if i == 0 else None)
        for e in run.evals:
            if e.bpb is not None and e.is_best:
                ax.annotate("BEST", (e.step, e.bpb), textcoords="offset points",
                            xytext=(0, 13), color=C_BEST, fontsize=9,
                            ha="center", weight="bold")
        if len(wins) > 1:
            for e in run.evals:
                ax.annotate(f"{e.window} B", (e.step, e.bpb), textcoords="offset points",
                            xytext=(0, -20), color=C_EVAL, fontsize=8, ha="center")
        ax.axvspan(min(exs) - 1, max(exs) + 1, color=C_EVAL, alpha=0.05, lw=0)
        _sess_note(ax, _sessions(run.evals), exs[0], DIM)
    if tbpb:
        for i, (_sid, part) in enumerate(_sessions(run.steps)):
            bx, by = run.col(part, "bpb")
            ax.plot(bx, by, color=C_BPB, lw=1.2, alpha=0.5,
                    label="train BPB (for scale)" if i == 0 else None)
    wl = "unknown window" if not wins else (
        f"fixed window = {wins[0]:,} B" if len(wins) == 1 else
        "window varies: " + " / ".join(f"{w:,} B" for w in wins))
    ax.set_title(f"HELD-OUT EVAL BPB — {run.name}\n{run.subtitle()}", loc="left", pad=12)
    ax.set_ylabel(f"eval BPB   [{wl}]")
    ax.set_xlabel("training step")
    ax.grid(True, axis="y")
    if run.events:
        _events(ax, run)
        _legend_seen(ax, run)
    if exs or tbpb:
        ax.legend(loc="best")
    if not exs:
        ax.text(0.5, 0.74, "no EVAL line in this log — an eval number without its window is not a number,\n"
                          "and this log carries none (see index.html)", transform=ax.transAxes,
                ha="center", va="center", color=DIM, fontsize=12)
    _fignote(fig, run)
    p = out / "eval.png"
    fig.savefig(p)
    plt.close(fig)
    return p


PARTS = [("fwd", "forward"), ("bwd", "backward"), ("opt", "optimizer"),
         ("retr", "TSCT retract"), ("ema", "EMA teacher")]


def chart_steptime(run, out: Path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(13, 6.6), constrained_layout=True)
    ts = run.timers
    xs = [t.step for t in ts]
    logy = False
    if not xs:
        ax.text(0.5, 0.5, "no `timer step N:` lines in this log\n"
                          "(this run predates --timers; nothing to break down)",
                transform=ax.transAxes, ha="center", color=DIM, fontsize=13)
        ax.set_yticks([])
    else:
        first = True
        warm_all = []
        for i, (_sid, part) in enumerate(_sessions(ts)):
            xs = [t.step for t in part]
            series, labels = [], []
            for k, lab in PARTS:
                v = [t.get(k) for t in part]
                if any(v):
                    series.append(v)
                    labels.append(lab)
            stacked = [0.0] * len(part)
            for v in series:
                stacked = [a + b for a, b in zip(stacked, v)]
            total = [t.get("total") for t in part]
            other = [max(0.0, tot - st) for tot, st in zip(total, stacked)]
            if any(other):
                series.append(other)
                labels.append("other / host overhead")
            ax.stackplot(xs, *series, colors=STACK[:len(series) - 1] + [C_OTHER],
                         labels=labels if first else [], baseline="zero",
                         step="mid", alpha=0.92, zorder=2)
            gt = [t.get("gpu_step") or total[j] for j, t in enumerate(part)]
            ax.plot(xs, gt, color=C_TOTAL, lw=1.7, marker="o", ms=3.5, mec=BG, mew=0.6,
                    label="gpu_step (total)" if first else None, zorder=4)
            warm_all += [g for x, g in zip(xs, gt) if x > 0]
            first = False
        if ts[0].step == 0 and len(ts) > 1:
            ax.annotate("step 0 is cold cubecl autotune,\nnot a measurement of a step (AGENTS.md §3.1)",
                        (0.30, 0.86), xycoords="axes fraction", color=C_TOTAL, fontsize=9)
        if warm_all:
            med = sorted(warm_all)[len(warm_all) // 2]
            ax.axhline(med, color=C_TOTAL, ls=":", lw=1, alpha=0.45)
            ax.annotate(f"warm median {med:,.0f} ms", (xs[-1], med), textcoords="offset points",
                        xytext=(-4, 5), ha="right", color=C_TOTAL, fontsize=9)
        if _autosymlog(ax, [t.get("total") for t in ts] + [t.get(k) for t in ts for k, _ in PARTS]):
            ax.set_yscale("log")
            logy = True
        _sess_note(ax, _sessions(ts), xs[0], DIM)
        ax.legend(loc="upper right", ncol=3)
    ax.set_title(f"STEP TIME — {run.name}\n{run.subtitle()}", loc="left", pad=12)
    ax.set_ylabel("milliseconds per step" + ("   (log scale — a stack is not additive on a log axis;\n"
                                         "read the gpu_step line for the total)" if logy else ""))
    ax.set_xlabel("training step")
    ax.grid(True, axis="y")
    if run.events:
        _events(ax, run)
        _legend_seen(ax, run)
    _fignote(fig, run)
    p = out / "steptime.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def _rate(mbps):
    """(scale, unit) so the axis reads in the largest unit that keeps a number."""
    for s, u in ((1 << 30, "GiB/s"), (1 << 20, "MiB/s"), (1 << 10, "KiB/s"), (1, "B/s")):
        if abs(mbps) >= s:
            return s, u
    return 1, "B/s"


def chart_throughput(run, out: Path, batch=None, seq=None):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(13, 6.0), constrained_layout=True)
    ts = [t for t in run.timers if t.step > 0 and t.get("total") > 0]
    if not ts:
        ax.text(0.5, 0.5, "no warm `timer step N:` line (need a step > 0 with total>0)\n"
                          "(this run predates --timers; throughput is not derivable)",
                transform=ax.transAxes, ha="center", color=DIM, fontsize=13)
        ax.set_yticks([])
        ax.set_title(f"THROUGHPUT — {run.name}\n{run.subtitle()}", loc="left", pad=12)
        ax.set_ylabel("corpus bytes per second")
        ax.set_xlabel("training step")
        _fignote(fig, run)
        p = out / "throughput.png"
        fig.savefig(p)
        plt.close(fig)
        return p
    if batch and seq:
        nbytes = batch * seq
        flat = [(t.step, nbytes / (t.get("total") / 1000.0)) for t in ts]
        med = sorted(v for _, v in flat)[len(flat) // 2]
        sc, unit = _rate(med)
        for _sid, part in _sessions(ts):
            ax.plot([t.step for t in part],
                    [nbytes / (t.get("total") / 1000.0) / sc for t in part],
                    color=C_BPB, lw=2.0, marker="o", ms=4, mec=BG, mew=0.5)
        ax.axhline(med / sc, color=C_TOTAL, ls=":", lw=1.2, alpha=0.6)
        ax.annotate(f"median {med / sc:.2f} {unit}", (flat[-1][0], med / sc),
                    textcoords="offset points", xytext=(-4, 6), ha="right",
                    color=C_TOTAL, fontsize=10)
        ax.set_ylabel(f"corpus bytes per second   [{unit}]")
        note = f"{nbytes:,} B per step = {batch} batch × {seq} seq  (from --batch/--seq-len)"
    else:
        for _sid, part in _sessions(ts):
            ax.plot([t.step for t in part], [1000.0 / t.get("total") for t in part],
                    color=C_BPB, lw=2.0, marker="o", ms=4, mec=BG, mew=0.5)
        ax.set_ylabel("steps per second   [bytes/s NOT derivable: no --batch/--seq-len]")
        note = "⚠ steps/s shown, NOT bytes/s — the timer lines carry no batch/seq;\n     pass --batch N --seq-len M for bytes/s"
    ax.set_title(f"THROUGHPUT — {run.name}\n{run.subtitle()}", loc="left", pad=12)
    ax.set_xlabel("training step")
    ax.grid(True, axis="y")
    ax.text(0.985, 0.87, note, transform=ax.transAxes, ha="right", va="top",
            color=C_TOTAL if not (batch and seq) else DIM, fontsize=10)
    _sess_note(ax, _sessions(ts), ts[0].step, DIM)
    if run.events:
        _events(ax, run)
        _legend_seen(ax, run)
    _fignote(fig, run)
    p = out / "throughput.png"
    fig.savefig(p)
    plt.close(fig)
    return p


WATCH_COLS = [("vram_mib", "VRAM used", "#c084fc", 1.0, "MiB"),
              ("ram_used_gib", "host RAM used", "#38bdf8", 1.0, "GiB"),
              ("gpu_util_pct", "GPU utilisation", "#5eead4", 1.0, "%")]


def chart_resources(watch: Path, out: Path, label=""):
    """VRAM / host RAM / GPU util over WALL time, from the watcher TSV.

    The TSV is appended to, so it can hold several sessions (each starts with a
    header and restarts `wall_s` at 0). A decrease in `wall_s` means a new
    session: break the line there rather than draw a ramp backwards in time.
    """
    plt = _mpl()
    idx, rows = {}, []
    for line in watch.read_text(errors="replace").splitlines():
        f = line.rstrip("\n").split("\t")
        if len(f) < 2:
            continue
        if f[0] == "wall_s":
            idx = {c: i for i, c in enumerate(f)}
            continue
        if idx and len(f) == len(idx):
            rows.append(f)
    if not rows:
        return None
    sessions, cur = [], [rows[0]]
    for r in rows[1:]:
        if float(r[idx["wall_s"]]) < float(cur[-1][idx["wall_s"]]):
            sessions.append(cur)
            cur = []
        cur.append(r)
    sessions.append(cur)
    # Small multiples, not three curves on one axis: MiB, GiB and % share no scale,
    # and one axis renders RAM and util as two flat lines on the floor.
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.6), sharex=True, constrained_layout=True)
    for ax, (key, lab, col, _, unit) in zip(axes, WATCH_COLS):
        if key not in idx:
            ax.set_visible(False)
            continue
        drawn = 0
        for sess in sessions:
            xs = [float(r[idx["wall_s"]]) / 60.0 for r in sess]
            ys = [float(r[idx[key]]) for r in sess]
            if not any(math.isfinite(y) and y > 0 for y in ys):
                continue
            ax.plot(xs, ys, color=col, lw=1.8, marker="o", ms=3, mec=BG, mew=0.5,
                    label=lab)
            drawn += 1
        fin = [float(r[idx[key]]) for r in rows if math.isfinite(float(r[idx[key]]))]
        if fin:
            lo, hi = min(fin), max(fin)
            pad = (hi - lo) * 0.15 or max(abs(hi) * 0.1, 1.0)
            ax.set_ylim(lo - pad, hi + pad)
            ax.annotate(f"min {lo:,.0f} {unit} · max {hi:,.0f} {unit}",
                        (0.995, 0.90), xycoords="axes fraction", ha="right",
                        color=DIM, fontsize=9)
            if drawn:
                ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.5))
        else:
            ax.text(0.5, 0.5, "every sample failed (nvidia-smi hiccup on every tick?)",
                    transform=ax.transAxes, ha="center", color=DIM, fontsize=11)
        ax.grid(True, axis="y")
        ax.set_ylabel(f"{lab}\n[{unit}]")
    every = rows[0][idx["every_s"]] if "every_s" in idx else "?"
    axes[-1].set_xlabel("minutes since the watcher started")
    fig.suptitle(f"RESOURCES over wall time — {watch.name}{label}\n"
                 f"sampled every {every} s · {len(rows)} samples · {len(sessions)} session(s)"
                 f" · nvidia-smi + /proc/meminfo", x=0.008, ha="left", fontsize=12.5)
    fig.text(0.008, 0.008,
             f"{watch.name} · tools/trainboard.py",
             color=DIM, fontsize=7.5, family="monospace")
    p = out / "resources.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def chart_compare(runs, out: Path):
    """Train CE across every log given — one panel per run, plus a best-CE table."""
    plt = _mpl()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.4), constrained_layout=True,
                                  width_ratios=[2.1, 1])
    ramp = ["#5eead4", "#38bdf8", "#c084fc", "#fbbf24", "#fb7185", "#34d399"]
    colors, bests = {}, []
    for i, run in enumerate(runs):
        if not run.steps:
            continue
        colors[run.name] = ramp[i % len(ramp)]
        xs, ce = run.col(run.steps, "ce")   # col() drops gaps, so len can be 0
        if not ce:
            continue
        for j, (_sid, part) in enumerate(_sessions(run.steps)):
            cx, cy = run.col(part, "ce")
            ax.plot(cx, cy, color=colors[run.name], lw=1.8,
                    label=f"{run.name}  ({run.preset})" if j == 0 else None)
        bests.append((min(ce), run, xs[ce.index(min(ce))]))
    if bests:
        bests.reverse()
        vals = [v for v, _, _ in bests]
        # start the axis at the best value so the bars compare, not the empty run-up
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.08 or 0.1
        bars = ax2.barh(range(len(vals)), vals,
                        color=[colors[r.name] for _, r, _ in bests], height=0.6)
        ax2.set_yticks(range(len(vals)))
        ax2.set_yticklabels([f"{r.name[:34]}\n@ step {st}" for _, r, st in bests], fontsize=8)
        for b, v in zip(bars, vals):
            ax2.annotate(f"{v:.3f}", (b.get_width(), b.get_y() + b.get_height() / 2),
                         textcoords="offset points", xytext=(6, 0), va="center",
                         color=FG, fontsize=9.5)
        ax2.set_xlim(lo - pad, hi + pad * 3)
    ax.set_title("TRAIN CE — all logs given", loc="left", pad=12)
    ax.set_ylabel("CE (nats/byte)")
    ax.set_xlabel("training step")
    ax.legend(loc="best")
    ax.grid(True, axis="y")
    ax2.set_title("best train CE per run", loc="left", pad=12)
    ax2.set_xlabel("CE (nats/byte)")
    ax2.grid(True, axis="x")
    ax2.invert_yaxis()
    p = out / "compare.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ─────────────────────────────── watch ───────────────────────────────
def _meminfo():
    d = {}
    try:
        for ln in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = ln.partition(":")
            d[k] = int(v.split()[0])  # kB
    except (OSError, ValueError, IndexError):
        return None, None
    total, avail = d.get("MemTotal"), d.get("MemAvailable")
    if total is None:
        return None, None
    return (total - avail) / 1048576.0, avail / 1048576.0  # GiB


def _nvidia(gpu):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader,nounits", "-i", str(gpu)],
        capture_output=True, text=True, timeout=15)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(out.stderr.strip()[:80] or "no output")
    util, mem = (p.strip() for p in out.stdout.splitlines()[0].split(","))
    return float(util), float(mem)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def sample_watch(pid, log: Path, every=5.0, seconds=None, gpu=0, quiet=False):
    """Sample VRAM/RAM/util into `<log>.watch.tsv`. Fail-soft: a bad tick is a NaN row."""
    tsv = Path(str(log) + ".watch.tsv")
    cols = ["wall_s", "wall_iso", "pid", "every_s", "gpu_util_pct", "vram_mib",
            "ram_used_gib", "ram_avail_gib", "errors"]
    t0 = time.time()
    errs = 0
    with open(tsv, "a", buffering=1) as fh:
        fh.write("\t".join(cols) + "\n")
        while True:
            w = time.time() - t0
            try:
                util, mem = _nvidia(gpu)
            except Exception as e:  # nvidia-smi hiccup must not kill the watcher
                util, mem = float("nan"), float("nan")
                if errs == 0 and not quiet:
                    print(f"trainboard: nvidia-smi: {e} — sampling continues (fail-soft)",
                          file=sys.stderr)
                errs += 1
            ru, ra = _meminfo() or (float("nan"), float("nan"))
            fh.write("\t".join([
                f"{w:.1f}", datetime.now().isoformat(timespec="seconds"), str(pid),
                str(every), f"{util:.0f}", f"{mem:.0f}", f"{ru:.2f}", f"{ra:.2f}",
                str(errs)]) + "\n")
            if not quiet:
                print(f"trainboard: watch {w:7.1f}s  gpu {util:5.0f}%  vram {mem:7.0f} MiB  "
                      f"ram {ru:5.1f} GiB", file=sys.stderr, end="\r")
            if seconds is not None and w >= seconds:
                break
            if pid and not _pid_alive(pid):
                if not quiet:
                    print(f"\ntrainboard: pid {pid} gone, sampling stops", file=sys.stderr)
                break
            time.sleep(every)
    return tsv


# ────────────────────────────── index ──────────────────────────────
def _tb(rows, head):
    h = "".join(f"<th>{html.escape(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def _num(v, f="{:.3f}"):
    return f.format(v) if isinstance(v, float) and math.isfinite(v) else "—"


def write_index(runs, out: Path, pngs: dict, batch=None, seq=None):
    sections = []
    # `run` is the session column: a restarted log has TWO step-1000 lines with
    # different numbers (official_v5e.log), so a table keyed on step alone is
    # ambiguous exactly where it matters.
    h_step = ["run", "step", "ce", "bpb", "best", "lr", "aux"]
    h_eval = ["run", "step", "ce", "bpb", "window", "", "seam"]
    h_time = ["run", "step", "total", "fwd", "bwd", "opt", "retr", "ema", "gpu_step"]
    for run in runs:
        rdir = out / run.path.stem
        evrows = [[e.step, e.kind, e.text] for e in run.events]
        strows = [[s.sess + 1, s.step, _num(s.ce), _num(s.bpb), _num(s.best),
                   _num(s.lr, "{:.3e}"), _num(s.aux, "{:.4f}")] for s in run.steps]
        evp = [[e.sess + 1, e.step, _num(e.ce), _num(e.bpb),
                f"{e.window:,} B" if e.window else "—", "BEST" if e.is_best else "", e.seam]
               for e in run.evals]
        tmr = [[t.sess + 1, t.step] + [f"{t.get(k):.1f}"
                                       for k in ("total", "fwd", "bwd", "opt", "retr", "ema", "gpu_step")]
               for t in run.timers[-40:]]
        meta = [("preset", run.preset), ("params", f"{run.params:,}" if run.params else "—"),
                ("planned steps", run.header.get("steps", "—")), ("lr", run.header.get("lr", "—")),
                ("data", run.header.get("data", "—")), ("backend", run.header.get("backend", "—")),
                ("quant", run.quant or "—"), ("run headers (sessions)", run.headers),
                ("lines read", run.lines), ("skipped (unparsable)", run.skipped),
                ("throughput basis", f"batch {batch} × seq {seq}" if batch and seq
                 else "NOT derivable (timer lines carry no batch/seq)")]
        sections.append(
            f'<section><h2>{html.escape(run.name)}</h2>'
            f'<p class="sub">{html.escape(run.subtitle())}</p>'
            f'<p class="path">{html.escape(str(run.path))}</p>'
            + ("".join(f'<figure><a href="{rdir.name}/{p.name}"><img src="{rdir.name}/{p.name}" loading="lazy"></a>'
                       f"<figcaption>{html.escape(desc)}</figcaption></figure>"
                       for p, desc in pngs.get(run.path.stem, []) if p) or "<p>no charts</p>")
            + _tb([[k, v] for k, v in meta], ["field", "value"])
            + f"<h3>eval points ({len(run.evals)}) — a BPB is only comparable within one window</h3>"
            + (_tb(evp, h_eval) or "<p>none</p>")
            + f"<h3>events ({len(run.events)})</h3>"
            + (_tb(evrows, ["step", "kind", "text"]) or "<p>none — clean run</p>")
            + f"<h3>step lines ({len(run.steps)})</h3>"
            + (_tb(strows, h_step) or "<p>none</p>")
            + f"<h3>last {len(tmr)} timer lines of {len(run.timers)}</h3>"
            + (_tb(tmr, h_time) or "<p>none</p>")
            + "</section>")
    css = """
:root{color-scheme:dark}
body{background:#0e1117;color:#c9d1d9;font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;padding:28px 32px}
h1{color:#5eead4;font-size:20px;margin:0 0 4px} h2{color:#fbbf24;font-size:16px;margin:28px 0 2px}
h3{color:#38bdf8;font-size:13px;margin:18px 0 6px;text-transform:uppercase;letter-spacing:.06em}
p.sub{color:#8b949e;margin:2px 0} p.path{color:#484f58;margin:2px 0 12px;font-size:12px}
section{max-width:1700px}
figure{display:inline-block;margin:0 18px 18px 0;vertical-align:top}
figure img{width:820px;border:1px solid #2a313c;border-radius:8px;background:#161b22}
figcaption{color:#8b949e;font-size:11.5px;max-width:820px;margin-top:4px}
table{border-collapse:collapse;margin:6px 0 14px;font-size:12px}
th{color:#5eead4;text-align:left;border-bottom:1px solid #2a313c;padding:3px 12px 3px 0;font-weight:600}
td{padding:2px 12px 2px 0;border-bottom:1px solid #161b22;color:#c9d1d9}
tr:nth-child(even) td{background:#12161d}
footer{color:#484f58;margin-top:34px;font-size:11px}
"""
    page = (f"<!doctype html><meta charset=utf-8><title>trainboard</title><style>{css}</style>"
            f"<h1>dormouse trainboard</h1>"
            f"<p class=sub>{len(runs)} run(s) · generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"· log format {FORMAT_VERSION}</p>" + "".join(sections) +
            "<footer>trainboard.py — dark, matplotlib Agg, stdlib otherwise. "
            "A held-out BPB without its window byte count is not a number (AGENTS.md §2.6); "
            "a step-time reading with no step index is not a measurement of a step (§3.1).</footer>")
    p = out / "index.html"
    p.write_text(page)
    return p


# ─────────────────────────────── main ───────────────────────────────
PNG_DESC = {
    "loss.png": "train CE and BPB vs step (log-y when the span warrants it), best CE marked; aux below",
    "eval.png": "held-out EVAL BPB with the window byte count in the axis label; train BPB for scale",
    "steptime.png": "step-time stack fwd/bwd/opt/retr/ema (+ residual overhead) with gpu_step on top",
    "throughput.png": "corpus bytes/s per step window (needs --batch/--seq-len; says so without them)",
    "resources.png": "VRAM / host RAM / GPU utilisation over wall time, from the watcher TSV",
}


def build(runs, out: Path, batch, seq, want_resources=True, logy=True):
    pngs = {}
    for run in runs:
        d = out / run.path.stem
        d.mkdir(parents=True, exist_ok=True)
        got = [(chart_loss(run, d, logy), PNG_DESC["loss.png"]),
               (chart_eval(run, d), PNG_DESC["eval.png"]),
               (chart_steptime(run, d), PNG_DESC["steptime.png"]),
               (chart_throughput(run, d, batch, seq), PNG_DESC["throughput.png"])]
        if want_resources:
            tsv = Path(str(run.path) + ".watch.tsv")
            if tsv.exists():
                p = chart_resources(tsv, d, label=f"  (watching {run.name})")
                if p:
                    got.append((p, PNG_DESC["resources.png"]))
        pngs[run.path.stem] = got
    if len(runs) > 1:
        chart_compare(runs, out)
    return pngs


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="trainboard.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="training log file(s)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output dir (default: <log dir>/trainboard)")
    ap.add_argument("--batch", type=int, help="batch size — needed to derive bytes/s")
    ap.add_argument("--seq-len", type=int, help="sequence length — needed to derive bytes/s")
    ap.add_argument("--watch", type=int, metavar="PID",
                    help="sample VRAM/RAM/GPU-util every --watch-every s until PID exits, then plot")
    ap.add_argument("--watch-every", type=float, default=5.0, help="sampling period (default 5)")
    ap.add_argument("--watch-seconds", type=float, default=None, help="stop sampling after N s")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index to sample (default 0)")
    ap.add_argument("--follow", nargs="?", type=float, const=15.0, metavar="N",
                    help="regenerate the whole board every N s (default 15)")
    ap.add_argument("--linear-loss", action="store_true", help="do not use log-y on the loss chart")
    ap.add_argument("--no-resources", action="store_true", help="skip the resources chart")
    args = ap.parse_args(argv)

    paths = [Path(p).expanduser().resolve() for p in args.logs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("trainboard: no such log: " + ", ".join(str(p) for p in missing), file=sys.stderr)
        return 2
    out = args.out or (paths[0].parent / "trainboard")
    out.mkdir(parents=True, exist_ok=True)

    if args.watch:
        for i, p in enumerate(paths):
            print(f"trainboard: watching pid {args.watch} -> {p}.watch.tsv", file=sys.stderr)
            sample_watch(args.watch, p, args.watch_every, args.watch_seconds, args.gpu)
            _ = i

    if args.follow is None:
        runs = [parse_log(p) for p in paths]
        pngs = build(runs, out, args.batch, args.seq_len, not args.no_resources, not args.linear_loss)
        idx = write_index(runs, out, pngs, args.batch, args.seq_len)
        for run in runs:
            for p, _ in pngs.get(run.path.stem, []):
                print(f"trainboard: {p}")
            if run.skipped:
                print(f"trainboard: {run.name}: skipped {run.skipped} unparsable line(s), "
                      f"first: {run.skip_examples[0]}", file=sys.stderr)
        print(f"trainboard: {idx}")
        return 0

    tick = args.follow
    while True:
        runs = [parse_log(p) for p in paths]
        pngs = build(runs, out, args.batch, args.seq_len, not args.no_resources, not args.linear_loss)
        write_index(runs, out, pngs, args.batch, args.seq_len)
        done = [r for r in runs if r.done]
        last = runs[0].steps[-1].step if runs[0].steps else 0
        print(f"trainboard: {datetime.now():%H:%M:%S} step {last} "
              f"({len(paths)} log(s), {'done: ' + done[0].done if done else 'running'}) -> {out}",
              file=sys.stderr)
        if all(r.done for r in runs) and len(runs) == 1:
            return 0
        time.sleep(tick)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("trainboard: interrupted", file=sys.stderr)
        sys.exit(130)
