#!/usr/bin/env -S uv run --script
"""Write `<domain>.bin.docs.json` — the document starts a `doc_masking` run may cut at.

Two ways in, one proof out:

* `--source-dir DIR` — the `.bin` is a concatenation of WHOLE `*.txt` shards
  (mix4's `web`, `wiki_en`, `ruweb`, `reasoning`, `rudialog`). Match every
  source file at the cursor by head, tail, size and full content, step by its
  exact size, and refuse if the matched records do not sum to the `.bin`.
* `--domain NAME` — the `.bin` was written by `build_mix4.py` from parquet/jsonl
  records (`code`, `math`, `wiki_ru`, `instruct`). The same builder code is
  replayed in its recorded order, and every record must match the bytes at the
  cursor. This is not a boundary *guess*: the record's own bytes are the check.

A flat `.bin` with no recoverable record structure cannot be annotated. The
builder REFUSES (`--refuse --reason ...`) and writes `<bin>.docs.refused.json`
so `docmask_audit.py` can say REFUSED-BY-BUILDER instead of MISSING. It never
writes a sidecar whose records do not sum to the file: the last check before
the write is `sum(record lengths) == bin size`, and an off-by-one names the
offset it failed at.

    uv run --with pyarrow port/scripts/build_doc_sidecar.py \
        --bin data_local/mix4/code.bin --domain code
    uv run port/scripts/build_doc_sidecar.py \
        --bin data_local/mix4/web.bin --source-dir /mnt/.../mix/web
    uv run port/scripts/build_doc_sidecar.py --bin data_local/mix4/wiki_en.bin \
        --refuse --reason "the 6.3 MB shards concatenate articles with no record token"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import random
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import build_mix4 as mix  # noqa: E402

HEAD = 64
TAIL = 64
BUILDER = "build_doc_sidecar.py"

#: The mix4 domains this builder can replay. A domain absent here has no
#: recorded record boundary and must be refused, not guessed.
REPLAYABLE = ("code", "math", "wiki_ru", "instruct")


def log(msg: str) -> None:
    print(msg, flush=True)


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _atomic_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _refusal_path(bin_path: Path) -> Path:
    return Path(str(bin_path) + ".docs.refused.json")


def refuse(bin_path: Path, reason: str, *, out: Path | None = None) -> Path:
    """Write the machine-readable refusal; never a sidecar."""
    side = _refusal_path(bin_path) if out is None else Path(out)
    _atomic_json(
        side,
        {
            "version": 1,
            "size": bin_path.stat().st_size,
            "verdict": "REFUSED-BY-BUILDER",
            "reason": reason,
            "builder": BUILDER,
        },
    )
    log(f"refused {bin_path}: {reason}")
    return side


def write_sidecar(
    bin_path: Path,
    starts: list[int],
    *,
    method: str,
    source: str | None = None,
    out: Path | None = None,
) -> Path:
    """Validate the boundaries against the file, then write the sidecar."""
    size = bin_path.stat().st_size
    if not starts or starts[0] != 0:
        raise SystemExit(
            f"{bin_path}: starts must begin at 0, got {starts[:1]}; refusing to write a sidecar"
        )
    for a, b in zip(starts, starts[1:], strict=False):
        if b <= a:
            raise SystemExit(
                f"{bin_path}: starts are not strictly ascending at {a} -> {b}; "
                f"refusing to write a sidecar"
            )
    if starts[-1] >= size:
        raise SystemExit(
            f"{bin_path}: a document starts at {starts[-1]} >= {size}; refusing to write a sidecar"
        )
    side = Path(str(bin_path) + ".docs.json") if out is None else Path(out)
    doc = {
        "version": 1,
        "size": size,
        "starts": starts,
        "method": method,
        "builder": BUILDER,
    }
    if source is not None:
        doc["source"] = source
    _atomic_json(side, doc)
    log(f"wrote {side}: {len(starts)} documents over {size} B")
    return side


def starts_from_records(
    bin_path: Path,
    records,
    *,
    expected_units: int | None = None,
    expected_bytes: int | None = None,
) -> list[int]:
    """The starts of `records`, each verified byte-for-byte at the cursor.

    `records` is an iterable of `bytes` in the order the `.bin` holds them.
    The final check is the one that matters: a record list that ends one byte
    early (or late) does not sum to the file and is refused, naming the offset.
    """
    size = bin_path.stat().st_size
    starts: list[int] = []
    cursor = 0
    with bin_path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for i, rec in enumerate(records):
            if not rec:
                raise SystemExit(
                    f"{bin_path}: record {i} is empty at offset {cursor}; "
                    f"refusing to write a sidecar"
                )
            end = cursor + len(rec)
            if end > size:
                raise SystemExit(
                    f"{bin_path}: record {i} ends at {end} B past the file's {size} B "
                    f"(offset {cursor}, off by {end - size} B); refusing to write "
                    f"a sidecar"
                )
            got = mm[cursor:end]
            if got != rec:
                first = next(
                    (k for k in range(len(rec)) if got[k : k + 1] != rec[k : k + 1]),
                    0,
                )
                raise SystemExit(
                    f"{bin_path}: record {i} does not match at offset {cursor + first} "
                    f"(record starts at {cursor}, is {len(rec)} B); refusing to write "
                    f"a sidecar — the source and the .bin disagree"
                )
            starts.append(cursor)
            cursor = end
        if expected_units is not None and len(starts) != expected_units:
            raise SystemExit(
                f"{bin_path}: replayed {len(starts)} records but the build recorded "
                f"{expected_units}; refusing to write a sidecar"
            )
        if expected_bytes is not None and cursor != expected_bytes:
            raise SystemExit(
                f"{bin_path}: records end at {cursor} B but the build recorded "
                f"{expected_bytes} B; refusing to write a sidecar"
            )
        if cursor != size:
            raise SystemExit(
                f"{bin_path}: records end at {cursor} B but the file is {size} B "
                f"(off by {size - cursor} B at offset {cursor}); refusing to write "
                f"a sidecar — a boundary after the end or before a missing byte "
                f"would cut the wrong document"
            )
    return starts


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
                f"no source shard starts at bin offset {offset}: the head "
                f"{head[:16]!r} matches no source file. The .bin and the named "
                f"directory disagree, so no sidecar is written."
            )
        # Head AND tail AND exact size, then the whole file: four independent
        # agreements. A file edited in the middle is exactly the silent wrong
        # boundary this check exists to refuse.
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
        for p in tail_ok:
            data = p.read_bytes()
            if bytes(mm[offset : offset + len(data)]) == data:
                return p
        raise SystemExit(
            f"source files match the head at offset {offset} but not the tail or "
            f"content: the .bin and the directory disagree, so no sidecar is written"
        )


def starts_from_shard_dir(bin_path: Path, source_dir: Path) -> list[int]:
    if not source_dir.is_dir():
        raise SystemExit(f"{source_dir} is not a directory")
    files = sorted(p for p in source_dir.glob("*.txt") if p.is_file())
    if not files:
        raise SystemExit(f"{source_dir} holds no `*.txt` shards")
    matcher = Matcher(files)
    size = bin_path.stat().st_size
    starts: list[int] = []
    with bin_path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        offset = 0
        while offset < size:
            try:
                doc = matcher.match(mm, offset, size)
            except SystemExit as e:
                raise SystemExit(
                    f"{bin_path}: records end at {offset} B but the file is {size} B "
                    f"(off by {size - offset} B): {e}"
                ) from e
            starts.append(offset)
            offset += doc.stat().st_size
        if offset != size:
            raise SystemExit(
                f"{bin_path}: records end at {offset} B but the file is {size} B "
                f"(off by {size - offset} B at offset {offset}); refusing to write "
                f"a sidecar"
            )
    return starts


# --- the replayable mix4 domains --------------------------------------------


def _manifest_entry(domain: str) -> dict:
    path = mix.MIX / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"{path} does not exist; cannot replay {domain!r}")
    return json.loads(path.read_text())["domains"][domain]


def _entry(domain: str) -> tuple[dict, int, int, set[str]]:
    e = _manifest_entry(domain)
    if "entry" in e:  # older manifest shape
        e = e["entry"]
    eval_set = set(e["eval"]["exclusion"].get("eval_unit_sha1", []))
    return e, int(e["target_bytes"]), int(e["train_units"]), eval_set


def _code_records():
    """`build_code`'s two passes, in the order they wrote the `.bin`."""
    e, target, _units, eval_set = _entry("code")
    files = [Path(p["path"]) for p in e["source"]["files"]]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise SystemExit(f"code: source parquet missing: {missing[:3]}")
    cols = ["repo_name", "path", "language", "license", "code"]
    seen: set[str] = set()
    written: set[str] = set()
    quotas = dict(mix.LANG_QUOTA)
    cursor = 0
    total = 0

    for idx, row in mix.iter_parquet_rows(files, cols):
        if idx < cursor:
            continue
        out, _reason, lang = mix.filter_code(row)
        if out is None:
            continue
        h = sha1(out)
        if h in seen:
            continue
        seen.add(h)
        if h in eval_set:
            continue
        if quotas.get(lang, 0) < len(out):
            continue
        rec = out + b"\n\n"
        quotas[lang] = quotas.get(lang, 0) - len(out)
        written.add(h)
        cursor = idx + 1
        total += len(rec)
        yield rec
        if total >= target:
            return

    # `build_code` fills a short quota from the TOP of the source, skipping only
    # what the first pass already wrote. Same iteration, same test.
    for _idx, row in mix.iter_parquet_rows(files, cols):
        out, _reason, _lang = mix.filter_code(row)
        if out is None:
            continue
        h = sha1(out)
        if h in written or h in eval_set:
            continue
        rec = out + b"\n\n"
        written.add(h)
        total += len(rec)
        yield rec
        if total >= target:
            return


def _math_records():
    """`build_math`'s shuffled file order, split on `\\n---\\n`, `doc + b"\\n\\n"`."""
    e, target, _units, eval_set = _entry("math")
    files = sorted((mix.MOUNT / "math").glob("*.txt"))
    if not files:
        raise SystemExit(f"{mix.MOUNT / 'math'} holds no `*.txt` shards")
    order = list(files)
    random.Random(mix.SEED ^ 0x4D415448).shuffle(order)
    seen: set[str] = set()
    total = 0
    for path in order:
        for doc in mix.split_math_docs(path.read_bytes()):
            ok, _reason = mix.filter_math_doc(doc)
            if not ok:
                continue
            h = sha1(doc)
            if h in seen:
                continue
            seen.add(h)
            if h in eval_set:
                continue
            rec = doc + b"\n\n"
            total += len(rec)
            yield rec
            if total >= target:
                return


def _wiki_ru_records():
    """`build_parquet_docs` over the MediaWiki parquet: one row is one document."""
    e, target, _units, eval_set = _entry("wiki_ru")
    path = mix.RUWIKI_FILE
    if not path.is_file():
        raise SystemExit(f"wiki_ru: source parquet missing: {path}")
    seen: set[str] = set()
    total = 0
    for _idx, row in mix.iter_parquet_rows([path], ["title", "text"]):
        out, _reason = mix.filter_wiki_ru(row)
        if out is None:
            continue
        h = sha1(out)
        if h in seen:
            continue
        seen.add(h)
        if h in eval_set:
            continue
        total += len(out)
        yield out
        if total >= target:
            return


def _instruct_records():
    """`build_instruct`: dedupe the rendered pairs, shuffle with `SEED`, write."""
    e, target, _units, eval_set = _entry("instruct")
    records: list[tuple[bytes, str]] = []
    seen: set[str] = set()
    for path in mix.INSTRUCT_FILES:
        if not path.is_file():
            raise SystemExit(f"instruct: source jsonl missing: {path}")
        with path.open("rb") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = row.get("instruction") or ""
                a = row.get("response") or ""
                if not q or not a:
                    continue
                try:
                    rec = mix.render_pair(q, a)
                except AssertionError:
                    continue
                h = sha1(rec)
                if h in seen:
                    continue
                seen.add(h)
                records.append((rec, h))
    random.Random(mix.SEED).shuffle(records)
    total = 0
    for rec, h in records:
        if h in eval_set:
            continue
        total += len(rec)
        yield rec
        if total >= target:
            return


REPLAYERS = {
    "code": _code_records,
    "math": _math_records,
    "wiki_ru": _wiki_ru_records,
    "instruct": _instruct_records,
}


def build_domain(bin_path: Path, domain: str, out: Path | None) -> Path:
    if domain not in REPLAYABLE:
        raise SystemExit(
            f"domain {domain!r} has no replayable record boundary. It is a flat "
            f"concatenation: use --refuse and name why, or --source-dir if the "
            f"bin is whole files."
        )
    e, _target, units, _eval = _entry(domain)
    if bin_path.stat().st_size != int(e["train_bytes"]):
        raise SystemExit(
            f"{bin_path} is {bin_path.stat().st_size} B but the recorded {domain} "
            f"train build is {e['train_bytes']} B; refusing to replay a different "
            f"file"
        )
    starts = starts_from_records(
        bin_path,
        REPLAYERS[domain](),
        expected_units=units,
        expected_bytes=int(e["train_bytes"]),
    )
    return write_sidecar(
        bin_path,
        starts,
        method=f"mix4-{domain}-replay",
        source=e.get("source", {}).get("dir") or domain,
        out=out,
    )


# --- CLI --------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bin", required=True, type=Path, help="the `.bin` to annotate")
    ap.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="the `*.txt` shard directory the bin is a concatenation of",
    )
    ap.add_argument(
        "--domain",
        choices=REPLAYABLE,
        default=None,
        help="a mix4 parquet/jsonl domain whose records can be replayed",
    )
    ap.add_argument(
        "--refuse",
        action="store_true",
        help="write `<bin>.docs.refused.json` instead of a sidecar",
    )
    ap.add_argument("--reason", default="", help="why the boundaries are not recoverable")
    ap.add_argument("--out", type=Path, default=None, help="default: <bin>.docs.json")
    args = ap.parse_args()

    if not args.bin.is_file():
        raise SystemExit(f"{args.bin} does not exist")
    if args.refuse:
        if not args.reason:
            raise SystemExit("--refuse needs --reason (the audit reports it verbatim)")
        refuse(args.bin, args.reason, out=args.out)
        return 0
    if (args.source_dir is None) == (args.domain is None):
        raise SystemExit(
            "name exactly one source: --source-dir for whole-file shards, or "
            "--domain for a replayable mix4 record build. A flat .bin with no "
            "recoverable records is `--refuse --reason ...`."
        )
    if args.source_dir is not None:
        starts = starts_from_shard_dir(args.bin, args.source_dir)
        write_sidecar(
            args.bin,
            starts,
            method="shard-full-content",
            source=str(args.source_dir),
            out=args.out,
        )
        return 0
    build_domain(args.bin, args.domain, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
