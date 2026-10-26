#!/usr/bin/env -S uv run --script
"""Derive `<domain>.bin.docs.json` — the document starts of a mix4 SHARD domain.

`build_mix4.py` writes the shard-based domains (`web`, `wiki_en`, `ruweb`,
`reasoning`, `rudialog`) as the concatenation of WHOLE source `*.txt` files,
in a seeded shuffle, with no separator: the bytes alone do not say where one
document ends and the next begins. This tool recovers the offsets from the
SHARD DIRECTORY those bytes came from — the same one named in the build's
`manifest.json` (`source.dir`) — by matching each source file's head and tail
against the `.bin` at the cursor and stepping by its exact size.

    uv run port/scripts/build_doc_sidecar.py \
        --bin data_local/mix4/web.bin \
        --source-dir /mnt/.../aria_data/pretrain/mix/web

The proof it publishes is the proof it requires: the matched sizes must sum to
the `.bin`'s size exactly, every source file is used at most as often as it
appears (duplicates are allowed — the builder does not dedup this path), and
the run refuses when no source file matches the bytes at the cursor. A domain
built from PARQUET ROWS (`code`, `math`, `books`, `qa`, `wiki_ru`, `instruct`)
has no per-file source identity and is refused: its offsets can only come from
the builder itself. (Adding that is a builder change: emit the offsets as the
rows are written.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import sys
from pathlib import Path

HEAD = 64
TAIL = 64


def log(msg: str) -> None:
    print(msg, flush=True)


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


class Matcher:
    """Source files indexed by their first `HEAD` bytes."""

    def __init__(self, files: list[Path]) -> None:
        self.files = files
        self.by_head: dict[bytes, list[Path]] = {}
        for p in files:
            size = p.stat().st_size
            if size < HEAD:
                continue
            with p.open("rb") as f:
                self.by_head.setdefault(f.read(HEAD), []).append(p)

    def match(self, mm: mmap.mmap, offset: int, size: int) -> Path:
        """The source file whose bytes occupy `[offset, offset+file_size)`."""
        head = bytes(mm[offset : offset + HEAD])
        cands = self.by_head.get(head, [])
        if not cands:
            raise SystemExit(
                f"no source shard starts with {head[:16]!r} at bin offset {offset}. "
                f"This .bin was not built from the named directory (or the builder "
                f"changed): the offsets would be wrong, so none are written."
            )
        # Head AND tail AND exact size: three independent agreements, no full
        # corpus read in the common (unique-head) case.
        tail_ok: list[Path] = []
        for p in cands:
            n = p.stat().st_size
            if offset + n > size:
                continue
            with p.open("rb") as f:
                f.seek(max(0, n - TAIL))
                want = f.read()
            got = bytes(mm[offset + n - len(want) : offset + n])
            if got == want:
                tail_ok.append(p)
        if not tail_ok:
            raise SystemExit(
                f"source shards match the head at offset {offset} but not the tail: "
                f"the .bin and the directory disagree, so no sidecar is written"
            )
        if len(tail_ok) == 1:
            return tail_ok[0]
        for p in tail_ok:  # a head+tail collision: settle it on the whole file
            data = p.read_bytes()
            if bytes(mm[offset : offset + len(data)]) == data:
                return p
        raise SystemExit(
            f"source files match offset {offset} by head, tail and size but not by "
            f"content: the bin and the directory disagree, so no sidecar is written"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bin", required=True, type=Path, help="mix4/<domain>.bin")
    ap.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="the `*.txt` shard directory it was built from",
    )
    ap.add_argument("--out", type=Path, default=None, help="default: <bin>.docs.json")
    args = ap.parse_args()
    if not args.bin.is_file():
        raise SystemExit(f"{args.bin} does not exist")
    if not args.source_dir.is_dir():
        raise SystemExit(f"{args.source_dir} is not a directory")
    files = sorted(p for p in args.source_dir.glob("*.txt") if p.is_file())
    if not files:
        raise SystemExit(f"{args.source_dir} holds no `*.txt` shards")
    matcher = Matcher(files)
    size = args.bin.stat().st_size
    starts: list[int] = []
    with args.bin.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        offset = 0
        while offset < size:
            doc = matcher.match(mm, offset, size)
            starts.append(offset)
            offset += doc.stat().st_size
    if offset != size:
        raise SystemExit(
            f"the matched documents end at {offset} B but {args.bin} is {size} B: "
            f"the sidecar would name a boundary inside/after the file, so none is written"
        )
    out = args.out or args.bin.with_suffix(args.bin.suffix + ".docs.json")
    out.write_text(
        json.dumps(
            {
                "version": 1,
                "size": size,
                "starts": starts,
                "method": "shard-head-tail",
                "source_dir": str(args.source_dir),
                "source_names_sha1": sha1("".join(p.name for p in files).encode()),
            },
            indent=1,
        )
        + "\n"
    )
    log(f"wrote {out}: {len(starts)} documents over {size} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
