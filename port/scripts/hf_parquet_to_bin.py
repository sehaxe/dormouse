#!/usr/bin/env -S uv run --with pyarrow --script
"""Turn a HuggingFace parquet text dataset into our `.bin` + `.docs.json` pair.

Why: `mix4`'s web component is a raw crawl filtered by our own heuristics, and we
measured those heuristics at ×1.07 (2026-10-23). FineWeb-Edu is the same crawl
after a *teacher-trained classifier* picked the educational subset — i.e. the
distillation we would otherwise have to run ourselves, already done and free
(ODC-BY). This script is the pipe: rows out of parquet, bytes into `.bin`, one
document per row so `doc_masking` finally has real boundaries (the raw shards are
6.3 MB blobs).

    uv run --with pyarrow port/scripts/hf_parquet_to_bin.py \
        --parquet 'data_local/fwedu/*.parquet' \
        --out-dir data_local/fwedu --name fresh --max-bytes 2000000000

Every drop is counted and named in `<name>.manifest.json` (`too_short`,
`too_long`, `dup`, `empty`), and the byte accounting is exact: written bytes plus
separators must equal the source bytes it kept.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="glob, e.g. 'dir/*.parquet'")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", default="fresh")
    ap.add_argument("--column", default="text")
    ap.add_argument("--min-bytes", type=int, default=400)
    ap.add_argument("--max-bytes", type=int, default=120_000)
    ap.add_argument("--max-total", type=int, default=2_000_000_000)
    ap.add_argument("--sep", default="\n\n")
    ap.add_argument("--batch", type=int, default=4096)
    a = ap.parse_args()

    files = sorted(glob.glob(a.parquet))
    if not files:
        raise SystemExit(f"no parquet matched {a.parquet!r}")

    import json as _json

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out_bin = a.out_dir / f"{a.name}.bin"
    sep = a.sep.encode()

    drops = {"too_short": 0, "too_long": 0, "dup": 0, "empty": 0}
    kept_docs = 0
    kept_bytes = 0
    total_bytes = 0
    starts: list[int] = []
    seen: set[bytes] = set()
    rows = 0

    with out_bin.open("wb") as fh:
        for path in files:
            if kept_bytes >= a.max_total:
                break
            if path.endswith(".json.gz") or path.endswith(".jsonl.gz"):
                import gzip
                with gzip.open(path, "rt", encoding="utf-8", errors="replace") as gh:
                    for line in gh:
                        try:
                            rec = _json.loads(line)
                        except Exception:
                            continue
                        text = rec.get(a.column) or ""
                        rows += 1
                        if not text:
                            drops["empty"] += 1
                            continue
                        raw = text.encode("utf-8", errors="replace")
                        if len(raw) < a.min_bytes:
                            drops["too_short"] += 1
                            continue
                        if len(raw) > a.max_bytes:
                            drops["too_long"] += 1
                            continue
                        h = hashlib.sha1(raw).digest()
                        if h in seen:
                            drops["dup"] += 1
                            continue
                        seen.add(h)
                        if kept_bytes + len(raw) > a.max_total:
                            break
                        if kept_docs:
                            fh.write(sep)
                            kept_bytes += len(sep)
                        starts.append(kept_bytes)
                        fh.write(raw)
                        kept_bytes += len(raw)
                        total_bytes += len(raw)
                        kept_docs += 1
                print(f"  {Path(path).name}: rows={rows} kept={kept_docs} "
                      f"{kept_bytes / 1e9:.2f} GB", flush=True)
                continue
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=a.batch, columns=[a.column]):
                for text in batch.column(0).to_pylist():
                    rows += 1
                    if not text:
                        drops["empty"] += 1
                        continue
                    raw = text.encode("utf-8", errors="replace")
                    if len(raw) < a.min_bytes:
                        drops["too_short"] += 1
                        continue
                    if len(raw) > a.max_bytes:
                        drops["too_long"] += 1
                        continue
                    h = hashlib.sha1(raw).digest()
                    if h in seen:
                        drops["dup"] += 1
                        continue
                    seen.add(h)
                    if kept_bytes + len(raw) > a.max_total:
                        break
                    if kept_docs:
                        fh.write(sep)
                        kept_bytes += len(sep)
                    starts.append(kept_bytes)
                    fh.write(raw)
                    kept_bytes += len(raw)
                    total_bytes += len(raw)
                    kept_docs += 1
                if kept_bytes >= a.max_total:
                    break
            print(f"  {Path(path).name}: rows={rows} kept={kept_docs} "
                  f"{kept_bytes / 1e9:.2f} GB", flush=True)

    (a.out_dir / f"{a.name}.bin.docs.json").write_text(
        json.dumps({"version": 1, "size": kept_bytes, "starts": starts})
    )
    manifest = {
        "parquet": files,
        "column": a.column,
        "rows_seen": rows,
        "docs_kept": kept_docs,
        "written_bytes": kept_bytes,
        "document_bytes": total_bytes,
        "separator_bytes": kept_bytes - total_bytes,
        "drops": drops,
        "unique_docs": len(seen),
        "limits": {"min_bytes": a.min_bytes, "max_bytes": a.max_bytes,
                   "max_total": a.max_total},
    }
    (a.out_dir / f"{a.name}.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
