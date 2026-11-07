#!/usr/bin/env -S uv run --script
"""Does every `mix_v7` source have document boundaries a `doc_masking` run can use?

One row per source: bytes, whether `<source>.docs.json` exists, its document
count, the implied mean document length, and a verdict.

* **REAL** — the sidecar parses, matches the file's size, and the count/length
  are plausible: mean document length in `[40 B, 20 KB]` and at least
  `bytes / 1e6 * 2` documents (a 1 MB-document corpus cannot be said to have
  document boundaries; the 6.3 MB shard sidecars this tool exists to reject had
  means of 6.3 MB and counts of 19-74 for 100-466 MB).
* **BOGUS** — a sidecar exists but fails that rule, is stale (its `size` is not
  the file's), or its starts are malformed. A wrong boundary cuts mid-sentence.
* **MISSING** — no sidecar, no refusal: nothing was ever built.
* **REFUSED-BY-BUILDER** — `build_doc_sidecar.py` wrote
  `<source>.docs.refused.json`: the source is a flat concatenation whose record
  boundaries are not recoverable from the bytes, and it must be excluded from a
  `doc_masking=true` plan (or run flat, which the trainer refuses).

    uv run python port/scripts/docmask_audit.py
    uv run python port/scripts/docmask_audit.py --mix port/configs/mix_v7.toml \
        --json /tmp/docmask_audit.json

Gate: `--bin A --bin B` audits named files instead of a mix plan, so a synthetic
`.bin` + sidecar can assert REAL and the exact document count.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

REAL_MAX_MEAN = 20 * 1024
REAL_MIN_MEAN = 40
DOCS_PER_MB = 2.0
SUFFIX = ".docs.json"
REFUSED_SUFFIX = ".docs.refused.json"


def _human(n: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n} B"


def _refusal(path: Path) -> dict | None:
    p = Path(str(path) + REFUSED_SUFFIX)
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {"reason": f"{p} is not JSON"}
    if not isinstance(raw, dict):
        return {"reason": f"{p} is not a JSON object"}
    return raw


def audit_one(path: Path) -> dict:
    """One source's row. Every non-REAL verdict carries the reason it failed."""
    size = path.stat().st_size
    side = Path(str(path) + SUFFIX)
    row: dict = {
        "source": path.stem.replace(".bin", "") or path.name,
        "path": str(path),
        "bytes": size,
        "sidecar": str(side) if side.is_file() else None,
        "documents": None,
        "mean_doc_bytes": None,
        "verdict": "MISSING",
        "reason": "no .docs.json sidecar",
    }
    if side.is_file():
        row["sidecar"] = str(side)
        try:
            raw = json.loads(side.read_text())
        except json.JSONDecodeError as e:
            row.update(verdict="BOGUS", reason=f"sidecar is not JSON: {e}")
            return row
        if not isinstance(raw, dict):
            row.update(verdict="BOGUS", reason="sidecar is not a JSON object")
            return row
        if raw.get("version") != 1:
            row.update(verdict="BOGUS", reason=f"sidecar version {raw.get('version')!r} != 1")
            return row
        try:
            declared = int(raw.get("size", -1))
        except (TypeError, ValueError):
            row.update(verdict="BOGUS", reason="sidecar `size` is not an integer")
            return row
        if declared != size:
            row.update(
                verdict="BOGUS",
                reason=f"stale sidecar: declares {declared} B, file is {size} B",
            )
            return row
        try:
            starts = [int(s) for s in raw.get("starts", [])]
        except (TypeError, ValueError):
            row.update(verdict="BOGUS", reason="sidecar `starts` is not a list of integers")
            return row
        count = len(starts)
        row["documents"] = count
        if not starts or starts[0] != 0:
            row.update(verdict="BOGUS", reason="starts must begin at 0")
            return row
        if any(b <= a for a, b in zip(starts, starts[1:], strict=False)):
            row.update(verdict="BOGUS", reason="starts are not strictly ascending")
            return row
        if starts[-1] >= size:
            row.update(verdict="BOGUS", reason=f"a document starts at {starts[-1]} >= {size}")
            return row
        mean = size / count
        row["mean_doc_bytes"] = mean
        if mean > REAL_MAX_MEAN:
            row.update(
                verdict="BOGUS",
                reason=(
                    f"mean document {mean / 1024:.1f} KB > {REAL_MAX_MEAN / 1024:.0f} KB: "
                    f"the sidecar cuts at file/shard seams, not document seams"
                ),
            )
            return row
        if mean < REAL_MIN_MEAN:
            row.update(
                verdict="BOGUS",
                reason=f"mean document {mean:.1f} B < {REAL_MIN_MEAN} B: starts are too dense",
            )
            return row
        floor = size / 1e6 * DOCS_PER_MB
        if count < floor:
            row.update(
                verdict="BOGUS",
                reason=(
                    f"{count} documents for {size / 1e6:.1f} MB: below the "
                    f"{floor:.0f}-document floor ({DOCS_PER_MB:g}/MB)"
                ),
            )
            return row
        row.update(verdict="REAL", reason=f"{count} documents, {_human(int(mean))} mean")
        return row
    refusal = _refusal(path)
    if refusal is not None:
        row.update(
            verdict="REFUSED-BY-BUILDER",
            reason=refusal.get("reason") or "builder refused (no reason recorded)",
        )
    return row


def audit(paths: list[Path], mix: Path | None = None) -> dict:
    rows = [audit_one(p) for p in paths]
    total = sum(r["bytes"] for r in rows)
    by = {}
    for r in rows:
        by[r["verdict"]] = by.get(r["verdict"], 0) + r["bytes"]
    out = {
        "mix": str(mix) if mix else None,
        "sources": rows,
        "totals": {
            "bytes": total,
            "by_verdict": by,
            "real_bytes": by.get("REAL", 0),
            "real_share": (by.get("REAL", 0) / total) if total else 0.0,
            "sources": len(rows),
            "real_sources": sum(1 for r in rows if r["verdict"] == "REAL"),
        },
    }
    return out


def _print_table(report: dict) -> None:
    rows = report["sources"]
    name_w = max([len("source")] + [len(r["source"]) for r in rows])
    print(
        f"{'source':<{name_w}}  {'bytes':>10}  {'sidecar':>7}  {'docs':>9}  "
        f"{'mean':>9}  {'verdict':<17}  detail"
    )
    for r in rows:
        docs = "-" if r["documents"] is None else f"{r['documents']:,}"
        mean = "-" if r["mean_doc_bytes"] is None else _human(int(r["mean_doc_bytes"]))
        side = "yes" if r["sidecar"] else "no"
        print(
            f"{r['source']:<{name_w}}  {_human(r['bytes']):>10}  {side:>7}  "
            f"{docs:>9}  {mean:>9}  {r['verdict']:<17}  {r['reason']}"
        )
    t = report["totals"]
    print(
        f"\n{t['real_sources']}/{t['sources']} sources REAL · "
        f"{_human(t['real_bytes'])} / {_human(t['bytes'])} = "
        f"{t['real_share'] * 100:.1f}% of the mix"
    )
    for verdict, n in sorted(t["by_verdict"].items()):
        print(f"  {verdict:<17} {_human(n):>10}  ({n / t['bytes'] * 100:.1f}%)")


def _sources_from_mix(path: Path) -> list[Path]:
    raw = tomllib.loads(path.read_text())
    sources = raw.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise SystemExit(f"{path} has no [sources] table")
    out = []
    for _name, value in sources.items():
        p = Path(value)
        if not p.is_absolute():
            p = (path.parent / p).resolve()
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--mix",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "mix_v7.toml",
        help="a mix plan; every [sources] value is audited",
    )
    ap.add_argument(
        "--bin",
        type=Path,
        action="append",
        default=None,
        help="audit this file (repeatable) instead of --mix",
    )
    ap.add_argument("--json", type=Path, default=None, help="also write the report here")
    args = ap.parse_args()

    if args.bin:
        paths = args.bin
        mix = None
    else:
        mix = args.mix
        if not mix.is_file():
            raise SystemExit(f"mix plan {mix} does not exist")
        paths = _sources_from_mix(mix)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"source file(s) do not exist: {missing}")
    report = audit(paths, mix)
    _print_table(report)
    text = json.dumps(report, indent=1, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
        print(f"wrote {args.json}", file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
