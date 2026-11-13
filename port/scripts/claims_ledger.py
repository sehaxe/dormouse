#!/usr/bin/env python3
"""The claims ledger: every citable number in `docs/records/reviews` and `docs/records/research`,
re-derived or marked absent.

The project has cited numbers that measured a different network (the in-VRAM
Engram eval defect), a different metric (headline BPB vs the final readout) or a
broken instrument (`muon_lr_mult` inert under `opt=mix`). This script reads the
two evidence-heavy doc trees, extracts the claim shapes the rulebook allows,
tries to re-derive each from the artifact it names, and prints a ledger with a
risk list. It is a gate, not a report: with `--check` it exits non-zero when a
claim dated `--today` or later is UNVERIFIABLE and names no artifact at all.

    python port/scripts/claims_ledger.py --repo . --docs docs \\
        --out docs/records/reviews/2026-11-07-claims-ledger.md

Claim shapes (inline and table cells):
  * `4.997 BPB`            unit bpb
  * `9 195 854 params`     unit params
  * `0.462 s/step`         unit speed
  * `2.4x faster`          unit ratio
  * `±0.05`                unit delta

Verification is mechanical only: a BPB is recomputed from the logged nats
(`bpb = nats / ln 2`; the logged byte count is checked, never used as a
divisor), a parameter count is rebuilt from the config by the repo's own
counter (`dormouse info`'s `param_count`, meta device), a speed is recomputed
from paired `tokens_per_s`, and a ratio is recomputed from two named runs.
Everything else is UNVERIFIABLE and must name the artifact that is missing.
Stdlib only; the repo's modules are reached through one subprocess.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

LN2 = math.log(2.0)

# Noise floors: a difference at or below these is not a discrepancy.
FLOOR_BPB = 0.043
FLOOR_PARAMS = 0.02
FLOOR_SPEED = 0.10
FLOOR_RATIO = 0.10

# The two authoritative fix instants. `0a7dfcd` (2026-11-07 04:27:19 +0300) is
# the commit that gave the Muon half one LR seam; `7adda92` is the Engram eval
# fix, landed 2026-09-28. A run is "before the fix" by its artifact's mtime
# because no run event carries a timestamp (D3 of the Muon audit).
MUON_FIX = datetime.fromisoformat("2026-11-07T04:27:19")
ENGRAM_FIX_DATE = "2026-09-28"

# The retraction authority is AGENTS.md §3.2 plus the two audits that scoped the
# defects further. The rules are data, each with its own citation, so that a
# claim status is always traceable to a line a human wrote.
RETRACTIONS = {
    # run name -> (reason, cite)
    "engram_ce": (
        "in-VRAM Engram eval defect: eval forward passed hashed_ids=None",
        "AGENTS.md §3.2; fix 7adda92",
    ),
    "small12": (
        "in-VRAM Engram eval defect (config in the affected class; no eval line on record)",
        "AGENTS.md §3.2; fix 7adda92",
    ),
    "kda_ce": ("attention arm had no gradient (chunk_wy forward leaf)", "AGENTS.md §3.2; 8fa5d4c"),
    "kda_full": ("attention arm had no gradient", "AGENTS.md §3.2; 8fa5d4c"),
    "kda_smoke": ("attention arm had no gradient", "AGENTS.md §3.2; 8fa5d4c"),
    "kda_smoke2": ("attention arm had no gradient", "AGENTS.md §3.2; 8fa5d4c"),
    "kda_smoke3": ("attention arm had no gradient", "AGENTS.md §3.2; 8fa5d4c"),
    "kda_smoke4": ("attention arm had no gradient", "AGENTS.md §3.2; 8fa5d4c"),
    "kda_smoke5": ("attention arm had no gradient", "AGENTS.md §3.2; 8fa5d4c"),
    "noengram_ce": ("attention arm had no gradient", "AGENTS.md §3.2; 8fa5d4c"),
    "probe2": ("attention arm had no gradient", "AGENTS.md §3.2; 8fa5d4c"),
}

# Numbers §3.2 names as retracted, keyed by unit. A claim is retracted when its
# value is within tolerance of the listed value AND the context matches `where`.
RETRACTED_NUMBERS = (
    ("bpb", 6.453, r"engram", "the memory-enabled run scored by a memory-disabled eval"),
    ("params", 7_500_000, r"small", "pre-2026-09-27 repricing (`small` = 7.5M)"),
    ("params", 12_200_000, r"base", "pre-2026-09-27 repricing (`base` = 12.2M)"),
    (
        "ratio",
        11.2,
        r"fused|kda",
        "only the forward claim survives; the fwd+bwd bench ran the tensor adjoint",
    ),
    (
        "speed",
        956.0,
        r"kda|no-kda|ablation",
        "measured alongside a live run: a ratio, not an absolute",
    ),
    (
        "speed",
        188.0,
        r"kda|no-kda|ablation",
        "measured alongside a live run: a ratio, not an absolute",
    ),
    ("speed", 8303.0, r"opt|optimizer", "step-0 warmup artifact (opt is 43 ms warm)"),
    ("speed", 10809.0, r"opt|optimizer", "step-0 warmup artifact (step is ~245 ms warm)"),
    ("ratio", 1.7, r"fused", "`fused` whole-loop was 1.3x SLOWER at flagship (ADR-0009; §3.2)"),
    ("ratio", 2.0, r"fused", "`fused` whole-loop was 1.3x SLOWER at flagship (ADR-0009; §3.2)"),
    (
        "delta",
        0.026,
        r"gr|gated residual",
        "the GR readout ran before the write; the arm has never been A/B'd",
    ),
)

# A qualifier on the line means the document knows it is quoting the
# depth-averaged training objective, not the model's BPB.
HEADLINE_QUALIFIERS = (
    "headline",
    "depth-averag",
    "depth averag",
    "training objective",
    "train objective",
    "objective",
    "bpb_mean",
    "not the model's bpb",
)

PYTHON_PARAM_SNIPPET = """
import json, sys
from dormouse.config import load_run
from dormouse.model import DormouseModel, param_count
out = {}
for p in sys.argv[1:]:
    try:
        out[p] = param_count(DormouseModel(load_run(p).model, device="meta"))
    except Exception as e:  # noqa: BLE001 - a missing preset is data, not a crash
        out[p] = {"error": type(e).__name__ + ": " + str(e)[:160]}
json.dump(out, sys.stdout)
"""

UNIT_LABEL = {
    "bpb": "BPB",
    "params": "params",
    "speed": "s/step or tok/s",
    "ratio": "ratio",
    "delta": "Δ",
}


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    doc: str
    line: int
    raw: str
    value: float
    unit: str
    kind: str
    evidence: list[str] = field(default_factory=list)
    context: str = ""
    col: str = ""
    metric: str = "bpb"
    quantity: str = "bpb"
    is_delta: bool = False
    step: int | None = None
    status: str = "UNVERIFIABLE"
    derived: float | None = None
    delta: float | None = None
    missing: str = ""
    reason: str = ""

    @property
    def floor(self) -> float:
        return {
            "bpb": FLOOR_BPB,
            "params": FLOOR_PARAMS,
            "speed": FLOOR_SPEED,
            "ratio": FLOOR_RATIO,
        }.get(self.unit, 0.0)

    @property
    def rel(self) -> float:
        """Discrepancy in units of the noise floor; comparable across units."""
        if self.delta is None or not self.floor:
            return 0.0
        return abs(self.delta) / self.floor

    @property
    def mismatch(self) -> bool:
        return self.status == "DERIVED" and self.delta is not None and abs(self.delta) > self.floor


def doc_date(path: Path | str) -> str:
    m = re.match(r"(\d{4}-\d{2}-\d{2})", Path(path).name)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


INLINE_BPB = re.compile(r"(?P<v>[-−+]?\d+(?:\.\d+)?)\s*(?:BPB|bpb)\b")
INLINE_PARAMS = re.compile(r"(?P<v>\d[\d _,]*(?:\.\d+)?)\s*(?P<suf>[KMB])?\s*params?\b")
INLINE_RATIO = re.compile(
    r"(?P<v>\d+(?:\.\d+)?)\s*[×x]\s*(?P<dir>faster|slower|speedup|the speed)\b"
    r"|(?P<dir2>faster|slower)\s+(?:by\s+)?(?P<v2>\d+(?:\.\d+)?)\s*[×x]"
)
INLINE_DELTA = re.compile(r"±\s*(?P<v>\d+(?:\.\d+)?)")
NUMBER = re.compile(r"[-−+]?\d+(?:\.\d+)?")

LOG_RE = re.compile(r"(?P<p>(?:~|/[\w.@+-]+)?/?[\w.@+/-]*\.(?:jsonl|log|tsv))")
CONFIG_RE = re.compile(r"(?P<p>[\w./-]*(?:configs/[\w./-]+\.toml|[\w./-]+\.config\.toml))")
CKPT_RE = re.compile(r"(?P<p>checkpoints/[\w./-]+|[\w./-]+\.pt)\b")
COMMIT_RE = re.compile(r"`(?P<c>[0-9a-f]{7,40})`")
PAPER_RE = re.compile(r"(?:arXiv[:\s]*|arxiv\.org/abs/)(?P<a>\d{4}\.\d{4,5})", re.I)
STEP_RE = re.compile(r"(?:@\s*|\bstep\s+)(?P<s>\d[\d,]{1,9})")

# A number followed by these is a count, a duration or a share, not the column's
# metric. Skipping them is what keeps "61.4 MB -> 2.863" from yielding two BPBs.
OTHER_UNIT = re.compile(
    r"\s*(?:%|MB\b|GB\b|KB\b|kB\b|TB\b|ms\b|µs\b|us\b|s\b|k\b|K\b|M\b|B\b|tokens?\b|steps?\b|bytes?\b|×|x\b)",
    re.I,
)


def _clean(line: str) -> str:
    return line.replace("**", "").replace("`", "").replace("~~", "")


def _evidence(text: str) -> list[str]:
    out: list[str] = []
    for m in LOG_RE.finditer(text):
        out.append("log:" + m.group("p"))
    for m in CONFIG_RE.finditer(text):
        out.append("config:" + m.group("p"))
    for m in CKPT_RE.finditer(text):
        out.append("checkpoint:" + m.group("p"))
    for m in COMMIT_RE.finditer(text):
        out.append("commit:" + m.group("c"))
    for m in PAPER_RE.finditer(text):
        out.append("paper:" + m.group("a"))
    return list(dict.fromkeys(out))


def _step(text: str) -> int | None:
    found = STEP_RE.findall(text)
    return int(found[-1].replace(",", "")) if found else None


def _num(raw: str) -> float:
    return float(raw.replace("−", "-").replace(" ", "").replace("_", "").replace(",", ""))


def _params_value(raw: str, suffix: str | None) -> float:
    # The claim regexes are deliberately loose (thousand separators, footnote
    # marks); a token that still does not parse is a claim about a number the
    # ledger cannot read. Returning nan marks it unverifiable instead of taking
    # the whole gate down with a ValueError.
    try:
        v = _num(raw)
    except ValueError:
        return float("nan")
    return v * {"K": 1e3, "M": 1e6, "B": 1e9}.get((suffix or "").upper(), 1.0)


def _is_delta(prefix: str, suffix: str) -> bool:
    if re.search(r"[±+−~-]\s*$", prefix):
        return True
    return bool(
        re.match(r"\s*(?:BPB\s+)?(?:away|above|below|worse|better|gap|premium|spread|band)", suffix)
    )


def _metric(context: str) -> str:
    low = context.lower()
    return "bpb_fin" if ("bpb_fin" in low or re.search(r"\bfin\b", low)) else "bpb"


def _kind(context: str, evidence: list[str], is_delta: bool) -> str:
    low = context.lower()
    if any(e.startswith("paper:") for e in evidence) or "et al" in low or "arxiv" in low:
        return "cited-paper"
    if any(
        w in low
        for w in (
            "speculat",
            "expect",
            "should be",
            "probably",
            "unverified",
            "not verified",
            "hypothesis",
            "predict",
            "projected",
            "extrapolat",
            "estimated",
            "estimate:",
        )
    ):
        return "speculation"
    if any(e.split(":")[0] in ("log", "checkpoint") for e in evidence) or "measured" in low:
        return "measured"
    if is_delta or any(e.split(":")[0] in ("config", "commit") for e in evidence):
        return "computed"
    return "speculation"


def _inline_claims(doc: str, lineno: int, line: str) -> list[Claim]:
    text = _clean(line)
    claims: list[Claim] = []
    step = _step(text)
    ev = _evidence(text)

    for m in INLINE_BPB.finditer(text):
        pre, post = text[: m.start()], text[m.end() : m.end() + 12]
        claims.append(
            Claim(
                doc=doc,
                line=lineno,
                raw=m.group(0).strip(),
                value=_num(m.group("v")),
                unit="bpb",
                kind="",
                evidence=list(ev),
                context=text.strip(),
                metric=_metric(text),
                is_delta=_is_delta(pre, post),
                step=step,
            )
        )
    for m in INLINE_PARAMS.finditer(text):
        raw = m.group("v")
        if not raw.strip():
            continue
        claims.append(
            Claim(
                doc=doc,
                line=lineno,
                raw=m.group(0).strip(),
                value=_params_value(raw, m.group("suf")),
                unit="params",
                kind="",
                evidence=list(ev),
                context=text.strip(),
                step=step,
            )
        )
    for m in INLINE_RATIO.finditer(text):
        raw = m.group("v") or m.group("v2")
        direction = m.group("dir") or m.group("dir2") or ""
        if not raw:
            continue
        claims.append(
            Claim(
                doc=doc,
                line=lineno,
                raw=m.group(0).strip(),
                value=_num(raw),
                unit="ratio",
                kind="",
                evidence=list(ev),
                context=text.strip(),
                col=direction,
                quantity="ratio",
                step=step,
            )
        )
    for m in INLINE_DELTA.finditer(text):
        unit = "bpb" if re.search(r"bpb|bpw", text, re.I) else "delta"
        claims.append(
            Claim(
                doc=doc,
                line=lineno,
                raw=m.group(0).strip(),
                value=_num(m.group("v")),
                unit=unit,
                kind="",
                evidence=list(ev),
                context=text.strip(),
                is_delta=True,
                step=step,
            )
        )
    for c in claims:
        c.kind = _kind(c.context, c.evidence, c.is_delta)
    return _dedupe(claims)


def _dedupe(claims: list[Claim]) -> list[Claim]:
    seen: set[tuple[str, int, str, float]] = set()
    out = []
    for c in claims:
        key = (c.doc, c.line, c.unit, round(c.value, 9), c.quantity)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _column_unit(header: str) -> tuple[str, bool, str, float]:
    """(unit, is_delta, quantity, scale) for a table column."""
    h = header.strip().strip("*").lower()
    if "param" in h:
        return "params", False, "params", 1.0
    if "bpb" in h or re.search(r"\bfin\b", h):
        return "bpb", bool(re.search(r"Δ|delta|gap|diff", h)), "bpb", 1.0
    if "±" in h or h.strip() in ("delta", "δ", "gap", "improvement", "difference"):
        return "delta", True, "delta", 1.0
    if "tok/s" in h or "tokens/s" in h or "tokens per s" in h:
        return "speed", False, "tok_per_s", 1.0
    if "ms/step" in h or "ms / step" in h:
        return "speed", False, "s_per_step", 1e-3
    if any(k in h for k in ("s/step", "step time", "time/step")):
        return "speed", False, "s_per_step", 1.0
    if any(k in h for k in ("speedup", "× faster", "x faster", "faster")):
        return "ratio", False, "ratio", 1.0
    return "", False, "", 1.0


def _number_ok(cell: str, m: re.Match, unit: str) -> bool:
    """Reject a number that names a count, a byte size or a step index."""
    pre = cell[: m.start()]
    post = cell[m.end() :]
    if re.search(r"(?:@|step|of|over)\s*$", pre, re.I):
        return False
    if OTHER_UNIT.match(post):
        return False
    raw = m.group(0)
    # An integer in a prose BPB cell is a step/token count. Keep an integer that
    # IS the cell (a headline can legitimately be `8`), or one next to `BPB`.
    if unit == "bpb" and "." not in raw and cell.strip() != raw and not re.search(r"\bBPB\b", cell):
        return False
    return True


def _cell_claims(
    doc: str, lineno: int, header: str, cell: str, row: str, ev: list[str]
) -> list[Claim]:
    unit, is_delta, quantity, scale = _column_unit(header)
    if not unit or not cell.strip():
        return []
    cell = _clean(cell)
    low = cell.lower()
    if any(bad in low for bad in ("n/a", "none", "holds", "fails", "tie", "deleted", "—")):
        if not NUMBER.search(cell):
            return []
    context = f"{_clean(header)} | {_clean(row)}"
    step = _step(cell) or _step(_clean(header)) or _step(_clean(row))
    claims: list[Claim] = []
    if unit == "params":
        for m in INLINE_PARAMS.finditer(cell):
            claims.append(
                Claim(
                    doc=doc,
                    line=lineno,
                    raw=m.group(0).strip(),
                    value=_params_value(m.group("v"), m.group("suf")),
                    unit="params",
                    kind="",
                    evidence=list(ev),
                    context=context,
                    col=_clean(header),
                    quantity="params",
                    step=step,
                )
            )
        if claims:
            return _finish(claims, is_delta)
        for m in re.finditer(r"(?P<v>\d[\d, _.]*)(?:\.(?P<f>\d+))?\s*(?P<suf>[KMB])?\b", cell):
            if not _number_ok(cell, m, "params"):
                continue
            raw = m.group(0)
            claims.append(
                Claim(
                    doc=doc,
                    line=lineno,
                    raw=raw.strip(),
                    value=_params_value(
                        m.group("v") + ("" if not m.group("f") else "." + m.group("f")),
                        m.group("suf"),
                    ),
                    unit="params",
                    kind="",
                    evidence=list(ev),
                    context=context,
                    col=_clean(header),
                    quantity="params",
                    step=step,
                )
            )
        return _finish(claims, is_delta)

    for m in NUMBER.finditer(cell):
        if not _number_ok(cell, m, unit):
            continue
        v = _num(m.group(0))
        if unit == "bpb" and (v > 100 or v < -100):
            continue
        if unit in ("speed", "ratio") and v <= 0:
            continue
        prefix, suffix = cell[: m.start()], cell[m.end() : m.end() + 12]
        claims.append(
            Claim(
                doc=doc,
                line=lineno,
                raw=m.group(0),
                value=v * scale,
                unit=unit,
                kind="",
                evidence=list(ev),
                context=context,
                col=_clean(header),
                metric="bpb" if unit != "bpb" else _metric(context),
                quantity=quantity,
                is_delta=is_delta or _is_delta(prefix, suffix),
                step=step,
            )
        )
    return _finish(claims, is_delta)


def _finish(claims: list[Claim], is_delta: bool) -> list[Claim]:
    for c in claims:
        if is_delta:
            c.is_delta = True
        c.kind = _kind(c.context, c.evidence, c.is_delta)
    return _dedupe(claims)


def _table_claims(doc: str, start: int, block: list[str], intro_ev: list[str]) -> list[Claim]:
    if len(block) < 3:
        return []
    header = block[0]
    if not re.match(r"^\s*\|?[\s:-]+\|", block[1]):
        return []
    headers = [h.strip() for h in header.strip().strip("|").split("|")]
    claims: list[Claim] = []
    for offset, row in enumerate(block[2:], start=start + 2):
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        ev = _evidence(_clean(row)) or _evidence(_clean(header)) or intro_ev
        for i, cell in enumerate(cells):
            if i >= len(headers):
                break
            claims.extend(_cell_claims(doc, offset, headers[i], cell, row, ev))
    return claims


def iter_claims(path: Path, doc: str) -> list[Claim]:
    """Stream a document; tables are handled block-wise, everything else by line.

    A table with no artifact in its header or rows inherits the evidence of the
    paragraph that introduces it -- but only when that paragraph names exactly
    one log/config/checkpoint, so an ambiguous attribution is never invented.
    """
    claims: list[Claim] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        block: list[str] = []
        block_start = 0
        intro: list[str] = []
        intro_lines = 0
        for lineno, line in enumerate(fh, start=1):
            if line.lstrip().startswith("|"):
                if not block:
                    block_start = lineno
                    intro = intro[-4:]
                block.append(line.rstrip("\n"))
                continue
            if block:
                claims.extend(_table_claims(doc, block_start, block, intro))
                block = []
                intro = []
            ev = _evidence(_clean(line))
            if ev:
                intro = [
                    r
                    for r in dict.fromkeys(ev)
                    if r.split(":")[0] in ("log", "checkpoint", "config")
                ]
                intro_lines = lineno
            elif lineno - intro_lines > 6:
                intro = []
            claims.extend(_inline_claims(doc, lineno, line.rstrip("\n")))
        if block:
            claims.extend(_table_claims(doc, block_start, block, intro))
    return claims


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------


class Artifacts:
    """Resolves a doc's evidence strings against the repo and the home logs."""

    def __init__(self, repo: Path, params: bool = True, python: str | None = None):
        self.repo = repo
        self.home = Path.home()
        self.want_params = params
        self.python = python or self._find_python()
        self._param_cache: dict[str, float | str] = {}
        self._eval_cache: dict[tuple[str, int], list[dict]] = {}
        self._run_cache: dict[tuple[str, int], dict] = {}

    def _find_python(self) -> str:
        for c in (
            self.repo / "port" / ".venv" / "bin" / "python",
            self.repo / ".venv" / "bin" / "python",
        ):
            if c.exists():
                return str(c)
        return sys.executable

    def resolve(self, ref: str) -> Path | None:
        kind, _, name = ref.partition(":")
        if kind == "paper":
            return None
        p = Path(name)
        candidates: list[Path] = []
        if name.startswith("~"):
            candidates.append(Path(name).expanduser())
        elif p.is_absolute():
            candidates.append(p)
        else:
            candidates += [self.repo / name, self.repo / "port" / name]
            if name.startswith("logs/"):
                candidates.append(self.home / name)
            candidates.append(self.home / "logs" / Path(name).name)
            candidates.append(self.home / "logs" / name)
        for c in candidates:
            if c.exists():
                return c
        return None

    def config_for(self, claim: Claim) -> Path | None:
        for ref in claim.evidence:
            if ref.startswith(("config:", "checkpoint:")):
                p = self.resolve(ref)
                if p and p.suffix == ".toml":
                    return p
                if p and ref.startswith("checkpoint:"):
                    snap = Path(str(p).removesuffix(".pt") + ".config.toml")
                    if snap.exists():
                        return snap
        return None

    def param_count(self, path: Path) -> float | str:
        return self._param_cache.get(str(path), "not requested")

    def fill_params(self, paths: list[Path]) -> None:
        todo = [str(p) for p in dict.fromkeys(paths) if self.param_count(p) == "not requested"]
        if not todo or not self.want_params:
            return
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.repo / "port" / "src")
        env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = [self.python, "-c", PYTHON_PARAM_SNIPPET, *todo]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300, check=False, env=env
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            for p in todo:
                self._param_cache[p] = f"param instrument unavailable: {e}"
            return
        if proc.returncode != 0:
            for p in todo:
                self._param_cache[p] = f"param instrument failed: {proc.stderr.strip()[-160:]}"
            return
        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as e:
            for p in todo:
                self._param_cache[p] = f"param instrument emitted no JSON: {e}"
            return
        for p in todo:
            v = data.get(p)
            self._param_cache[p] = v if isinstance(v, (int, float)) else str(v)

    def evals(self, path: Path) -> list[dict]:
        key = (str(path), path.stat().st_mtime_ns)
        if key not in self._eval_cache:
            self._eval_cache[key] = list(_eval_records(path))
        return self._eval_cache[key]

    def run_event(self, path: Path) -> dict:
        key = (str(path), path.stat().st_mtime_ns)
        if key not in self._run_cache:
            self._run_cache[key] = _run_event(path)
        return self._run_cache[key]


# ---------------------------------------------------------------------------
# derivation
# ---------------------------------------------------------------------------


TEXT_EVAL_OLD = re.compile(
    r"\bstep\s+(?P<step>\d+)\s+EVAL\s+ce=(?P<ce>[-+0-9.eE]+)\s+bpb=(?P<bpb>[-+0-9.eE]+)"
    r".*?over\s+(?P<bytes>\d+)\s*B"
)
TEXT_EVAL_NEW = re.compile(
    r"\bstep[= ](?P<step>\d+)\b.*?\bbpb_fin=(?P<fin_bpb>[-+0-9.eE]+)"
    r".*?\bbpb_mean=(?P<mean_bpb>[-+0-9.eE]+).*?\bfin=(?P<fin>[-+0-9.eE]+)"
    r".*?\bloss=(?P<loss>[-+0-9.eE]+).*?over\s+(?P<bytes>\d+)\s*B"
)


def _eval_records(path: Path):
    """Yield eval readings as dicts, streaming; no whole-file read."""
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(d, dict) or d.get("event") != "eval":
                    continue
                yield {
                    "step": d.get("step"),
                    "bpb": d.get("bpb"),
                    "bpb_fin": d.get("bpb_fin"),
                    "bpb_mean": d.get("bpb_mean"),
                    "loss": d.get("loss"),
                    "fin": d.get("fin"),
                    "scored": d.get("scored") or d.get("eval_bytes"),
                    "offset": d.get("eval_offset"),
                }
        return
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = TEXT_EVAL_NEW.search(line)
            if m:
                yield {
                    "step": int(m.group("step")),
                    "bpb": float(m.group("fin_bpb")),
                    "bpb_fin": float(m.group("fin_bpb")),
                    "bpb_mean": float(m.group("mean_bpb")),
                    "loss": float(m.group("loss")),
                    "fin": float(m.group("fin")),
                    "scored": int(m.group("bytes")),
                    "offset": None,
                }
                continue
            m = TEXT_EVAL_OLD.search(line)
            if m:
                yield {
                    "step": int(m.group("step")),
                    "bpb": float(m.group("bpb")),
                    "bpb_fin": None,
                    "bpb_mean": None,
                    "loss": float(m.group("ce")),
                    "fin": None,
                    "scored": int(m.group("bytes")),
                    "offset": None,
                }


def _pick(records: list[dict], step: int | None) -> dict | None:
    if not records:
        return None
    if step is not None:
        for r in records:
            if r["step"] == step:
                return r
        return None
    return records[-1]


def _bpb_of(rec: dict) -> tuple[float | None, str]:
    """The model's BPB from a reading: `fin / ln 2`, else `loss / ln 2`."""
    scored = rec.get("scored")
    if scored is None or scored <= 0:
        return None, "the eval line's byte count is missing or not positive"
    if rec.get("fin") is not None:
        return float(rec["fin"]) / LN2, ""
    if rec.get("loss") is not None:
        return float(rec["loss"]) / LN2, ""
    return None, "the eval line has neither `fin` nor `loss` (nats)"


def derive_bpb(rec: dict | None, path: Path) -> tuple[float | None, str]:
    if rec is None:
        return None, f"{path.name}: no eval at the claimed step"
    got, why = _bpb_of(rec)
    if got is None:
        return None, f"{path.name}: {why}"
    logged = rec.get("bpb_fin") if rec.get("bpb_fin") is not None else rec.get("bpb")
    if logged is not None and abs(float(logged) - got) > 1e-6:
        # The log's own metric disagrees with its nats. The recomputation wins;
        # the discrepancy is named rather than silently averaged away.
        return got, (
            f"{path.name}: logged bpb {float(logged):.6f} disagrees with "
            f"nats/ln2 {got:.6f}; recomputed from nats"
        )
    return got, ""


def derive_bpb_pair(
    path1: Path, path2: Path, recs1: list[dict], recs2: list[dict], step: int | None
) -> tuple[float | None, str, dict | None]:
    """The difference of two paired readings at the latest common step."""
    if step is None:
        common = sorted({r["step"] for r in recs1} & {r["step"] for r in recs2})
        if not common:
            return None, f"{path1.name} and {path2.name}: no common eval step", None
        step = common[-1]
    r1 = _pick(recs1, step)
    r2 = _pick(recs2, step)
    if r1 is None or r2 is None:
        return None, f"no eval at step {step} in {path1.name if r1 is None else path2.name}", None
    v1, w1 = _bpb_of(r1)
    v2, w2 = _bpb_of(r2)
    if v1 is None or v2 is None:
        return None, w1 or w2, None
    return v1 - v2, "", r1


def derive_speed(path: Path, quantity: str) -> tuple[float | None, str]:
    """Recompute s/step or tokens/s from the JSONL's own counters."""
    if path.suffix != ".jsonl":
        return None, f"{path.name}: only JSONL runs carry tokens_per_s"
    run = None
    per_s: list[float] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not isinstance(d, dict):
                continue
            if d.get("event") == "run" and run is None:
                run = d
            elif d.get("event") == "log" and d.get("tokens_per_s"):
                per_s.append(float(d["tokens_per_s"]))
    if not per_s:
        return None, f"{path.name}: no tokens_per_s on any log line"
    per_s.sort()
    median = per_s[len(per_s) // 2]
    if quantity == "tok_per_s":
        return median, ""
    if run is None:
        return None, f"{path.name}: no run event"
    tokens = float(run.get("batch") or 0) * float(run.get("seq_len") or 0)
    tokens *= float(run.get("grad_accum") or 1)
    if not tokens:
        return None, f"{path.name}: run event has no batch/seq_len"
    return tokens / median, ""


def derive_config_run(path: Path) -> dict:
    """The flags that decide which defect class a run is in (stdlib toml)."""
    import tomllib

    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    model = data.get("model", data)
    return {
        "use_engram": model.get("use_engram"),
        "engram_ram": model.get("engram_ram"),
        "use_kda": model.get("use_kda"),
    }


def _run_event(path: Path) -> dict:
    if path.suffix != ".jsonl":
        return {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i > 20:
                break
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("event") == "run":
                return d
    return {}


# ---------------------------------------------------------------------------
# retraction
# ---------------------------------------------------------------------------


def _run_names(claim: Claim) -> list[str]:
    names = []
    for ref in claim.evidence:
        _, _, name = ref.partition(":")
        names.append(Path(name).name)
        names.append(Path(name).stem)
        names.append(Path(name).name.replace(".config.toml", ""))
    return [n for n in names if n]


def retraction(claim: Claim, arts: Artifacts) -> str:
    """Return a reason when §3.2 or a scoping audit retracts this claim."""
    # A parameter count of a config is still that config's count; the defects
    # below invalidate what a run *measured*, not what it instantiated.
    if claim.unit != "params":
        for n in _run_names(claim):
            if n in RETRACTIONS:
                return "{} ({})".format(*RETRACTIONS[n])
            for key, (reason, cite) in RETRACTIONS.items():
                if n.startswith(key):
                    return f"{reason} ({cite})"

    # Pre-fix in-VRAM Engram: the eval forward passed hashed_ids=None. The
    # config says whether the arm was in VRAM; the doc date says whether the
    # fix had landed. A speed claim is not an eval reading, so it survives.
    if (
        claim.unit in ("bpb", "ratio", "delta")
        and doc_date(claim.doc)
        and doc_date(claim.doc) < ENGRAM_FIX_DATE
    ):
        cfg = arts.config_for(claim)
        if cfg is not None:
            run = derive_config_run(cfg)
            if run.get("use_engram") and run.get("engram_ram") is False:
                return (
                    "in-VRAM Engram eval defect: `use_engram=true`, `engram_ram=false` "
                    f"in {cfg.name}, before 7adda92 (AGENTS.md §3.2)"
                )

    low = (claim.context + " " + claim.col).lower()
    if claim.unit == "ratio" and "fused" in low:
        if re.search(r"faster than burn", low) or re.search(r"fused.*faster", low):
            if 1.6 <= claim.value <= 2.1:
                return "`fused` whole-loop was 1.3x SLOWER at flagship (ADR-0009; §3.2)"
    for unit, value, where, reason in RETRACTED_NUMBERS:
        if claim.unit != unit:
            continue
        if unit == "params":
            tol = 0.02
        elif unit == "bpb":
            tol = 0.001
        else:
            tol = 0.01
        if abs(claim.value - value) <= max(tol * abs(value), 1e-9) and re.search(where, low):
            return reason
    return ""


def headline_confusion(rec: dict, claim: Claim) -> bool:
    """True when the claim quotes the depth-averaged objective as the model's BPB."""
    if claim.unit != "bpb" or claim.metric != "bpb":
        return False
    low = claim.context.lower()
    if any(q in low for q in HEADLINE_QUALIFIERS):
        return False
    mean = rec.get("bpb_mean")
    if mean is None and rec.get("loss") is not None:
        mean = float(rec["loss"]) / LN2
    fin = rec.get("bpb_fin")
    if fin is None and rec.get("fin") is not None:
        fin = float(rec["fin"]) / LN2
    if mean is None or fin is None or abs(float(mean) - float(fin)) <= FLOOR_BPB:
        return False
    # Only retract when the cited value is the mean and is NOT the fin reading.
    return abs(claim.value - float(mean)) < 0.02 and abs(claim.value - float(fin)) > FLOOR_BPB


def muon_defect(claim: Claim, arts: Artifacts) -> str:
    """A bpb/ratio/delta claim from an `opt=mix` run before 0a7dfcd is retracted."""
    if claim.unit not in ("bpb", "ratio", "delta"):
        return ""
    for ref in claim.evidence:
        if not ref.startswith("log:"):
            continue
        path = arts.resolve(ref)
        if path is None or path.suffix != ".jsonl":
            continue
        run = arts.run_event(path)
        if run.get("opt") != "mix":
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if mtime < MUON_FIX:
            return (
                "broken-Muon `opt=mix` run before 0a7dfcd (2026-11-07 04:27): the "
                "schedule overwrote `muon_lr_mult` every step, so the arm ran at "
                "the AdamW LR (docs/records/reviews/2026-11-07-muon-audit.md D1-D2)"
            )
    return ""


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def verify(claim: Claim, arts: Artifacts) -> None:
    retract = retraction(claim, arts)
    if retract:
        claim.status = "RETRACTED"
        claim.reason = retract

    logs = [r for r in claim.evidence if r.startswith("log:")]
    configs = [r for r in claim.evidence if r.startswith(("config:", "checkpoint:"))]

    if claim.unit == "params" and not claim.is_delta:
        path = arts.config_for(claim)
        if path is not None:
            got = arts.param_count(path)
            if isinstance(got, (int, float)):
                claim.status = "RETRACTED" if retract else "DERIVED"
                claim.derived = float(got)
                claim.delta = claim.value - float(got)
                return
            if not retract:
                claim.status = "UNVERIFIABLE"
            claim.missing = f"param count unavailable for {path.name}: {got}"
            return
        if retract:
            return
        claim.missing = (
            f"no config named (claim names {', '.join(configs) or 'nothing'})"
            if configs
            else "no config or checkpoint named"
        )
        return

    if claim.unit == "bpb":
        if logs:
            resolved = [(r, arts.resolve(r)) for r in logs]
            hit = [(r, p) for r, p in resolved if p is not None]
            missing = [r for r, p in resolved if p is None]
            if hit and claim.is_delta and len(hit) >= 2:
                (_r1, p1), (_r2, p2) = hit[0], hit[1]
                got, why, rec = derive_bpb_pair(p1, p2, arts.evals(p1), arts.evals(p2), claim.step)
                if got is None:
                    if not retract:
                        claim.status = "UNVERIFIABLE"
                    claim.missing = why
                    return
                if claim.value < 0:
                    got = -got
                claim.derived = abs(got)
                claim.delta = claim.value - claim.derived
                claim.status = "RETRACTED" if retract else "DERIVED"
                return
            if hit:
                ref, path = hit[0]
                rec = _pick(arts.evals(path), claim.step)
                got, why = derive_bpb(rec, path)
                if got is None:
                    if not retract:
                        claim.status = "UNVERIFIABLE"
                    claim.missing = why
                    return
                claim.derived = got
                claim.delta = claim.value - got
                if retract:
                    claim.status = "RETRACTED"
                    return
                if rec is not None and headline_confusion(rec, claim):
                    claim.status = "RETRACTED"
                    claim.reason = (
                        "headline-vs-fin: the cited value is the depth-averaged training "
                        "objective (`bpb_mean`/`loss`), not the final readout decode uses "
                        "(docs/records/reviews/2026-11-06-bpb-accounting.md)"
                    )
                    return
                claim.status = "DERIVED"
                return
            claim.missing = f"log not found: {', '.join(missing)}"
            return
        if retract:
            return
        if claim.kind == "cited-paper":
            paper = [r for r in claim.evidence if r.startswith("paper:")]
            claim.missing = (
                f"external source {paper[0].split(':')[1]} is not in the repo"
                if paper
                else "external source not named"
            )
            return
        claim.missing = (
            f"no eval log named (config only: {configs[0].split(':', 1)[1]})"
            if configs
            else "no log or checkpoint named"
        )
        return

    if claim.unit in ("speed", "ratio"):
        resolved = [(r, arts.resolve(r)) for r in logs]
        hit = [(r, p) for r, p in resolved if p is not None]
        if claim.unit == "ratio":
            if claim.is_delta:
                if not retract:
                    claim.missing = "a ratio delta needs the two ratios it subtracts; none named"
                return
            if len(hit) >= 2:
                (_r1, p1), (_r2, p2) = hit[0], hit[1]
                v1, w1 = derive_speed(p1, "s_per_step")
                v2, w2 = derive_speed(p2, "s_per_step")
                if v1 and v2:
                    ratio = v2 / v1
                    if claim.col.startswith("slower"):
                        ratio = 1.0 / ratio
                    claim.derived = ratio
                    claim.delta = claim.value - ratio
                    claim.status = "RETRACTED" if retract else "DERIVED"
                    return
                if not retract:
                    claim.status = "UNVERIFIABLE"
                claim.missing = w1 or w2
                return
            if not retract:
                claim.status = "UNVERIFIABLE"
            claim.missing = (
                f"ratio needs two named runs; found {len(hit)}"
                if resolved
                else "no timing artifact named (a ratio needs two runs)"
            )
            return

        if claim.is_delta:
            if not retract:
                claim.missing = "a speed delta needs the two timings it subtracts; none named"
            return
        if len(hit) >= 1:
            ref, path = hit[0]
            got, why = derive_speed(path, claim.quantity)
            if got is not None and got > 0:
                claim.derived = got
                claim.delta = claim.value - got
                claim.status = "RETRACTED" if retract else "DERIVED"
                return
            if not retract:
                claim.status = "UNVERIFIABLE"
            claim.missing = why
            return
        if not retract:
            claim.status = "UNVERIFIABLE"
            claim.missing = (
                f"timing artifact not found: {', '.join(r for r, p in resolved if p is None)}"
                if resolved
                else "no timing artifact named (a speed claim needs a run log)"
            )
        return

    # delta: a stated noise band. Nothing mechanical to re-derive from one number.
    if not retract:
        claim.missing = "noise estimate: the repeated readings it was computed from are not named"


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _fmt(v: float | None, unit: str) -> str:
    if v is None:
        return "-"
    if unit == "params":
        return f"{v:,.0f}"
    if unit in ("speed", "ratio"):
        return f"{v:.5g}"
    return f"{v:g}"


def _totals(claims: list[Claim]) -> dict[str, int]:
    t = defaultdict(int)
    t["claims"] = len(claims)
    for c in claims:
        t[c.status.lower()] += 1
        t["kind_" + c.kind.replace("-", "_")] += 1
        t["unit_" + c.unit] += 1
        if c.mismatch:
            t["mismatch"] += 1
        if c.status == "UNVERIFIABLE":
            t["no_evidence" if not c.evidence else "absent_artifact"] += 1
    return dict(t)


def _repo_relative(refs: list[str], repo: Path | str) -> list[str]:
    """A path in the ledger must resolve from the repo root: `tools/check_doc_refs.py`
    resolves doc references there, and the evidence a report writes is often
    package-relative (`configs/b1.toml`). Prefixing `port/` makes the emitted
    table verifiable instead of adding dangling references (10 in the first run)."""
    root = Path(repo)
    out: list[str] = []
    for ref in refs:
        kind, _, name = ref.partition(":")
        p = Path(name)
        if (
            kind in ("config", "checkpoint", "log")
            and not p.is_absolute()
            and not name.startswith(("~", "logs/"))
        ):
            for prefix in ("port", "python"):
                if (root / prefix / name).exists():
                    ref = f"{kind}:{prefix}/{name}"
                    break
        out.append(ref)
    return out


def render(claims: list[Claim], args, gate: list[Claim]) -> str:
    t = _totals(claims)
    docs = sorted({c.doc for c in claims})
    out: list[str] = []
    w = out.append
    w("# Claims ledger — every number in `docs/records/reviews` and `docs/records/research`")
    w("")
    w(
        f"Generated by `port/scripts/claims_ledger.py` on **{args.today}**. "
        f"Commit `{args.commit}`; docs root `{args.docs}`; artifact root `{args.repo}`."
    )
    w("")
    w(
        f"**{t['claims']} claims** in {len(docs)} documents — "
        f"**{t.get('derived', 0)} derived**, {t.get('unverifiable', 0)} unverifiable, "
        f"{t.get('retracted', 0)} retracted; {t.get('mismatch', 0)} derived values differ "
        "from the stated one by more than the noise floor."
    )
    w("")
    w("| total | derived | unverifiable | retracted | mismatches |")
    w("|---:|---:|---:|---:|---:|")
    w(
        f"| {t['claims']} | {t.get('derived', 0)} | {t.get('unverifiable', 0)} | "
        f"{t.get('retracted', 0)} | {t.get('mismatch', 0)} |"
    )
    w("")
    if args.no_params:
        w(
            "**`--no-params` was passed: param-count claims were NOT checked.** A "
            "`DERIVED` claim that needs the module tree is left at its stated value, "
            "so this mismatch count is a LOWER bound (2026-11-08: 1 here, 9 with the "
            "param subprocess). Do not quote it as the ledger's verdict."
        )
        w("")
    w("## Method")
    w("")
    w(
        "Five shapes are extracted, inline and from table columns: `X BPB`, "
        "`N params`, `Y s/step` (or `ms/step`, `tok/s`), `Y× faster/slower`, "
        "`±Z`. Every claim records its `file:line`, value, unit, kind and the "
        "evidence the document itself names. Kinds: **measured** (a log, "
        "checkpoint or bench file is named), **computed** (arithmetic, a config "
        "or a commit is named), **cited-paper** (an arXiv id or a paper is "
        "named), **speculation** (nothing is named)."
    )
    w("")
    w(
        "Scope is `docs/records/reviews/**/*.md` and `docs/records/research/**/*.md`; this ledger "
        "excludes itself, since its own appendix is a list of claims."
    )
    w("")
    w("Verification is mechanical and deliberately narrow:")
    w("")
    w(
        f"* **BPB** — `bpb = nats / ln 2`, recomputed from the eval line at the "
        f"claimed step (JSONL `fin`/`loss`; text logs' `fin=`/`ce=`). The logged "
        f"byte count is checked to be positive and is **never** a divisor; a "
        f"logged `bpb` that disagrees with its own nats is reported, not trusted. "
        f"Floor **{FLOOR_BPB} BPB**."
    )
    w(
        "* **params** — rebuilt from the config or the checkpoint's own "
        "`.config.toml` by the repo's counter (`dormouse info`'s "
        "`param_count`, meta device, one subprocess). Floor **2%**."
    )
    w(
        "* **speed** — `batch × seq_len × grad_accum / median(tokens_per_s)` "
        f"from the JSONL run, or the median `tokens_per_s` itself. Floor **{FLOOR_SPEED:.0%}**."
    )
    w(
        "* **ratio** — a `×` comparison recomputed from two named runs; the "
        f"claimed direction decides which is the denominator. Floor **{FLOOR_RATIO:.0%}**."
    )
    w("* **±** — a noise estimate; only its absence can be checked.")
    w("")
    w(
        "Status is `DERIVED` (recomputed, with the delta shown), `UNVERIFIABLE` "
        "(no artifact that can re-derive it — the missing artifact is named), or "
        "`RETRACTED` (§3.2 or a scoping audit says the number measured something "
        "else). A derived mismatch is still `DERIVED`; it goes on the risk list."
    )
    w("")
    w("### Retraction rules applied")
    w("")
    w("| rule | what it catches | authority |")
    w("|---|---|---|")
    w(
        "| in-VRAM Engram eval defect | a BPB/ratio/Δ claim from `engram_ce`/`small12`, "
        "or from a pre-2026-09-28 doc whose config has `use_engram=true` and "
        "`engram_ram=false` | AGENTS.md §3.2, fix `7adda92` |"
    )
    w(
        "| frozen attention arm | any quality or timing claim from `kda_ce`, "
        "`kda_full`, `kda_smoke{,2..5}`, `noengram_ce`, `probe2` | AGENTS.md §3.2, `8fa5d4c` |"
    )
    w(
        "| broken Muon | a BPB/ratio/Δ claim from an `opt=mix` JSONL whose mtime is "
        "before `0a7dfcd` (2026-11-07 04:27), when the schedule overwrote "
        "`muon_lr_mult` | `docs/records/reviews/2026-11-07-muon-audit.md` D1-D2 |"
    )
    w(
        "| headline vs fin | a cited BPB that is `bpb_mean`/`loss` while `bpb_fin` "
        "differs by more than the floor, and the line does not say "
        "headline/depth-avg/objective | `docs/records/reviews/2026-11-06-bpb-accounting.md` |"
    )
    w(
        "| retracted literals | 6.453 BPB (`engram_ce`), 7.5M/12.2M params, "
        "1.7-2.0× and 11.2× fused, 956/188 ms `--no-kda`, 8303/10809 ms optimizer, "
        "GR −0.026 | AGENTS.md §3.2 |"
    )
    w("")
    w(
        "The Muon date is the artifact's mtime because no run event carries a "
        "timestamp (D3 of the Muon audit) — an absence the ledger names rather "
        "than hides. Only JSONL runs can be checked: a text log records no `opt`."
    )
    w("")
    w("## Gate")
    w("")
    w(
        "A claim is **new** when its document's date (the `YYYY-MM-DD` filename "
        "prefix) is `--today` or later. `--check` exits non-zero when a new claim "
        "is `UNVERIFIABLE` and names **no artifact at all** (no log, checkpoint, "
        "config, commit or paper): a new number must name its evidence, per "
        "ADR-0018 rule 1. An `UNVERIFIABLE` claim that names an artifact the "
        "ledger cannot read passes — that is an absence on record, not a silent one."
    )
    w("")
    if gate:
        w(f"**FAIL — {len(gate)} new claim(s) name no evidence.** `--check` exits 1.")
        w("")
        w(
            "Read this count with its scope: `new` means the DOCUMENT's date is "
            f"`{args.today}` or later, and a report written today that quotes an "
            "earlier document's number is new by that rule. The count is a "
            "precision problem before it is a finding — the CI job is report-only "
            "until the evidence association is line-accurate (`.bulba/plan.md`)."
        )
        w("")
        for c in gate:
            w(f"* `{c.doc}:{c.line}` — {c.raw} ({c.unit})")
    else:
        w(
            f"**PASS** — no claim dated {args.today} or later is unverifiable "
            "without a named artifact."
        )
    w("")
    w("## Risk list — derived values that disagree beyond the noise floor")
    w("")
    risks = sorted([c for c in claims if c.mismatch], key=lambda c: (-c.rel, c.doc, c.line))
    if not risks:
        w(
            "None. Every claim the ledger could re-derive agrees with its artifact "
            "within the stated floor."
        )
    else:
        by_doc: dict[str, list[Claim]] = defaultdict(list)
        for c in risks:
            by_doc[c.doc].append(c)
        w(
            "Grouped by document; within a document sorted by discrepancy in units "
            "of the noise floor (`rel`), so BPB, params and speed are comparable."
        )
        w("")
        for doc in sorted(by_doc):
            w(f"### `{doc}`")
            w("")
            w("| file:line | claim | derived | delta | rel | evidence |")
            w("|---|---:|---:|---:|---:|---|")
            for c in sorted(by_doc[doc], key=lambda c: -c.rel):
                ev = ", ".join(_repo_relative(c.evidence, args.repo)) or "—"
                w(
                    f"| `{c.doc}:{c.line}` | {c.raw} | {_fmt(c.derived, c.unit)} | "
                    f"{c.delta:+.4g} | {c.rel:.1f}× | `{ev}` |"
                )
            w("")
    w("## Top 10 by magnitude of discrepancy")
    w("")
    if not risks:
        w("None.")
    else:
        w("| # | file:line | unit | stated | derived | delta | rel |")
        w("|---:|---|---:|---:|---:|---:|---:|")
        for i, c in enumerate(risks[:10], start=1):
            w(
                f"| {i} | `{c.doc}:{c.line}` | {c.unit} | {c.raw} | "
                f"{_fmt(c.derived, c.unit)} | {c.delta:+.4g} | {c.rel:.1f}× |"
            )
    w("")
    w("## Retractions")
    w("")
    retracted = [c for c in claims if c.status == "RETRACTED"]
    if not retracted:
        w("None.")
    else:
        by_reason: dict[str, list[Claim]] = defaultdict(list)
        for c in retracted:
            by_reason[c.reason].append(c)
        w(
            f"{len(retracted)} claims. A retracted number may still be *derived* "
            "(the arithmetic is usually right); what is retracted is the "
            "identification of what it measured."
        )
        w("")
        for reason in sorted(by_reason, key=lambda r: -len(by_reason[r])):
            group = by_reason[reason]
            w(f"### {reason} — {len(group)}")
            w("")
            for c in sorted(group, key=lambda c: (c.doc, c.line))[:400]:
                derived = f" → derived {_fmt(c.derived, c.unit)}" if c.derived is not None else ""
                w(f"* `{c.doc}:{c.line}` — {c.raw} ({c.unit}){derived}")
            if len(group) > 400:
                w(f"* … {len(group) - 400} more (see `--csv`)")
            w("")
    w("## Unverifiable, by missing artifact")
    w("")
    un = [c for c in claims if c.status == "UNVERIFIABLE"]
    buckets: dict[str, int] = defaultdict(int)
    for c in un:
        key = c.missing.split(":")[0]
        buckets[key] += 1
    w("| missing | claims |")
    w("|---|---:|")
    for k in sorted(buckets, key=lambda k: -buckets[k]):
        w(f"| {k} | {buckets[k]} |")
    w("")
    w("## Per-document totals")
    w("")
    w("| document | claims | derived | unverifiable | retracted | mismatches |")
    w("|---|---:|---:|---:|---:|---:|")
    per: dict[str, list[Claim]] = defaultdict(list)
    for c in claims:
        per[c.doc].append(c)
    for doc in sorted(per):
        cs = per[doc]
        w(
            f"| `{doc}` | {len(cs)} | {sum(c.status == 'DERIVED' for c in cs)} | "
            f"{sum(c.status == 'UNVERIFIABLE' for c in cs)} | "
            f"{sum(c.status == 'RETRACTED' for c in cs)} | {sum(c.mismatch for c in cs)} |"
        )
    w("")
    w("## Appendix — the ledger")
    w("")
    w(
        "Every extracted claim: `file:line`, value, unit, kind, status, the "
        "evidence the document states, and the recomputed value when there is one."
    )
    w("")
    w("| file:line | value | unit | kind | status | derived | evidence |")
    w("|---|---:|---|---|---|---:|---|")
    for c in sorted(claims, key=lambda c: (c.doc, c.line, c.unit, c.value)):
        ev = ", ".join(_repo_relative(c.evidence, args.repo)) or "—"
        w(
            f"| `{c.doc}:{c.line}` | {c.raw} | {c.unit} | {c.kind} | {c.status} | "
            f"{_fmt(c.derived, c.unit)} | `{ev}` |"
        )
    w("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def collect(docs_root: Path) -> list[Claim]:
    claims: list[Claim] = []
    for sub in ("reviews", "research"):
        root = docs_root / sub
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.md")):
            if "claims-ledger" in path.name:
                continue
            doc = str(path.relative_to(docs_root.parent))
            claims.extend(iter_claims(path, doc))
    return claims


def run(args) -> tuple[list[Claim], list[Claim]]:
    docs_root = Path(args.docs).resolve()
    repo = Path(args.repo).resolve()
    claims = collect(docs_root)
    arts = Artifacts(repo, params=not args.no_params, python=args.python)

    # Parameter counts in one subprocess, so the instrument is paid for once.
    if not args.no_params:
        paths = []
        for c in claims:
            if c.unit == "params" and not c.is_delta:
                p = arts.config_for(c)
                if p is not None:
                    paths.append(p)
        arts.fill_params(paths)

    for c in claims:
        verify(c, arts)
        if c.status != "RETRACTED":
            reason = muon_defect(c, arts)
            if reason:
                c.status = "RETRACTED"
                c.reason = reason

    gate = [
        c
        for c in claims
        if doc_date(c.doc)
        and doc_date(c.doc) >= args.today
        and c.status == "UNVERIFIABLE"
        and not c.evidence
    ]
    return claims, gate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Ledger of every numeric claim in docs/records/reviews and docs/records/research."
        ),
    )
    ap.add_argument("--docs", default="docs", help="docs root (scans reviews/ and research/)")
    ap.add_argument("--repo", default=".", help="root the evidence paths resolve against")
    ap.add_argument(
        "--today", default="2026-11-07", help="the gate's date; docs dated today or later are new"
    )
    ap.add_argument("--out", default="", help="write the markdown here (default stdout)")
    ap.add_argument("--csv", default="", help="write every claim as CSV")
    ap.add_argument("--no-params", action="store_true", help="skip the param-count subprocess")
    ap.add_argument(
        "--python", default=None, help="interpreter that can import dormouse (default: repo venv)"
    )
    ap.add_argument("--commit", default="", help="commit named in the header")
    ap.add_argument("--check", action="store_true", help="exit 1 when the gate fails")
    args = ap.parse_args(argv)

    docs = Path(args.docs)
    if not docs.is_dir():
        print(f"loud: no docs root at {docs}", file=sys.stderr)
        return 2
    if args.repo == ".":
        args.repo = str(docs.resolve().parent)

    claims, gate = run(args)
    args.commit = args.commit or _git_commit(Path(args.repo))
    text = render(claims, args, gate)

    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    if args.csv:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(
                [
                    "doc",
                    "line",
                    "raw",
                    "value",
                    "unit",
                    "quantity",
                    "kind",
                    "status",
                    "derived",
                    "delta",
                    "rel",
                    "evidence",
                    "missing",
                    "reason",
                ]
            )
            for c in claims:
                wr.writerow(
                    [
                        c.doc,
                        c.line,
                        c.raw,
                        c.value,
                        c.quantity,
                        c.kind,
                        c.status,
                        c.derived,
                        c.delta,
                        f"{c.rel:.3f}",
                        " ".join(c.evidence),
                        c.missing,
                        c.reason,
                    ]
                )

    t = _totals(claims)
    print(
        f"claims {t['claims']} | derived {t.get('derived', 0)} | "
        f"unverifiable {t.get('unverifiable', 0)} | retracted {t.get('retracted', 0)} | "
        f"mismatch {t.get('mismatch', 0)} | gate {'FAIL' if gate else 'PASS'}",
        file=sys.stderr,
    )
    return 1 if (args.check and gate) else 0


def _git_commit(repo: Path) -> str:
    try:
        return (
            subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
            or "unknown"
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
