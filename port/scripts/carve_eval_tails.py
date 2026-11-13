#!/usr/bin/env -S uv run --script
"""Carve a disjoint held-out tail out of every mix-plan source that lacks one.

`eval_by_domain.py` refuses to score a window that IS a training source (the
path rule). A flat `.bin` source has no held-out bytes inside it, so this
splits one: the training file keeps its path and is TRUNCATED to the head, and
the tail becomes `<stem>_eval.bin` — a file no source names. That is the shape
`best/fwedu_train.bin` / `best/fwedu_eval.bin` already has.

    cd port
    uv run scripts/carve_eval_tails.py --plan configs/mix_refetch.toml

The cut is the last `--tail-mb` MiB. Where the file carries the refetch's
document separator (`\\n\\n---\\n\\n`, the `best/` sources), the cut starts at
the LAST such separator at or before that byte; the `mix/` sources were
written raw with no separator, so the cut is a fixed line-aligned offset. Every
window's offset, size and SHA-256 are printed; they are the record in
`docs/records/reviews/2026-11-08-domain-eval.md`.

ONE WAY: an existing `<stem>_eval.bin` is left alone and its hash printed.
Truncation cannot be undone, so rebuilding a lost window means re-fetching the
source first; the script refuses to guess.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from dormouse.instrument import loud
from dormouse.train.mixture import load_mix_plan

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_by_domain import eval_path_for, resolve_plan  # noqa: E402

#: The separator `refetch_pretrain.py` writes between documents. The `mix/`
#: files come from `fetch_mix_sources.py`, which writes raw text, so they have
#: none (a `---` inside code/math is content, not a marker).
SEPARATOR = b"\n\n---\n\n"


def separator_before(fh, target: int, search: int, sep: bytes) -> int | None:
    """The last `sep` starting at or before `target`, within `search` bytes."""
    lo = max(0, target - search)
    fh.seek(lo)
    buf = fh.read(target - lo + len(sep))
    j = buf.rfind(sep, 0, target - lo + len(sep))
    return None if j < 0 else lo + j


def next_line(fh, target: int, limit: int = 4096) -> int:
    """The first byte after a newline at/after `target` (a fixed offset, text-safe)."""
    fh.seek(target)
    buf = fh.read(limit)
    j = buf.find(b"\n")
    return target + j + 1 if j >= 0 else target


def carve(src: Path, tail: int, search: int, sep: bytes | None) -> tuple[Path, int, int, str, str]:
    """Truncate `src` to the head; write the tail beside it. Returns the record."""
    size = src.stat().st_size
    if size <= tail + 1:
        loud(
            f"{src} is {size} B, not more than the {tail} B tail: it is already "
            f"carved or too small. Escape: re-fetch the source, or a smaller "
            f"--tail-mb."
        )
    target = size - tail
    with src.open("rb") as fh:
        start = separator_before(fh, target, search, sep) if sep else None
        rule = "separator"
        if start is None:
            start = next_line(fh, target)
            rule = "line-offset"
    if not 0 < start < size:
        loud(f"{src}: carve offset {start} is outside (0, {size})")
    out = eval_path_for(src)
    with src.open("rb") as fh:
        fh.seek(start)
        data = fh.read()
    out.write_bytes(data)
    with src.open("r+b") as fh:
        fh.truncate(start)
    return out, start, len(data), hashlib.sha256(data).hexdigest(), rule


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--plan",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "mix_refetch.toml"),
        help="mix plan whose [sources] are carved",
    )
    ap.add_argument("--tail-mb", type=int, default=8, help="tail size in MiB (default 8)")
    ap.add_argument(
        "--search",
        type=int,
        default=65536,
        help="how far back to look for a document separator (default 64 KiB)",
    )
    ap.add_argument(
        "--no-separator",
        action="store_true",
        help="never use the `\\n\\n---\\n\\n` marker; always a fixed line offset",
    )
    a = ap.parse_args()

    plan_path = resolve_plan(a.plan)
    plan = load_mix_plan(plan_path)
    tail = a.tail_mb << 20
    sep = None if a.no_separator else SEPARATOR
    print(f"plan={plan_path} tail={tail:,} B search={a.search:,} B separator={sep!r}")
    print("| domain | train B (after) | eval file | offset | bytes | sha256 | rule |")
    print("|---|---:|---|---:|---:|---|---|")
    for name, src in plan.sources.items():
        src = Path(src)
        out = eval_path_for(src)
        if out.is_file():
            data = out.read_bytes()
            print(
                f"| {name} | {src.stat().st_size:,} | {out} | — | {len(data):,} "
                f"| `{hashlib.sha256(data).hexdigest()}` | existing |"
            )
            continue
        out, off, n, digest, rule = carve(src, tail, a.search, sep)
        print(
            f"| {name} | {src.stat().st_size:,} | {out} | {off:,} | {n:,} | `{digest}` | {rule} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
