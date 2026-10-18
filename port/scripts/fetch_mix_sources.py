# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb>=1.3"]
# ///
"""Fetch SMALL slices of the code/math/RU mixture sources as raw-text `.bin` files.

The pretrain corpus is one 19.7 GB filtered-web file; the research recipe
(`docs/research/2026-10-16-data-mixing.md` §6.1-6.2) needs ~1.5 GB of code,
math and Russian on top of it. This fetches ~120 MB of each — enough to wire the
mixture and run the 9.2M smoke, not the full acquisition.

    uv run scripts/fetch_mix_sources.py                 # all three, 120 MB each
    uv run scripts/fetch_mix_sources.py --source code   # one
    uv run scripts/fetch_mix_sources.py --cap-mb 60 --out-dir /somewhere

**Why duckdb**: the three sources are parquet on the Hugging Face hub, and two
of the shards are far larger than the slice we want (fineweb-2's `rus_Cyrl`
shard 0 is 4.8 GB). DuckDB reads parquet over HTTPS with range requests, so
`LIMIT n` costs roughly the bytes it returns instead of the whole shard. The
dependency is declared inline (PEP 723) — it is NOT a dependency of `dormouse`.

Licences are the datasets' own (codeparrot-clean: MIT per its card; open-web-math
and fineweb-2: ODC-By). The script refuses to fake a file: a source that cannot
be reached is reported with its exact URL and a non-zero exit, and nothing is
written for it.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

#: `/home/sehaxe/dormouse/data_local/mix/` — beside the pretrain corpus.
DEFAULT_OUT = Path("/home/sehaxe/dormouse/data_local/mix")

#: Total text written across all sources stays under this (the task's ~500 MB cap).
DEFAULT_TOTAL_CAP_MB = 500


@dataclass(frozen=True)
class Source:
    """One parquet slice: where it lives, which column is text, what it is."""

    name: str
    dataset: str
    config: str
    split: str
    shard: int
    column: str
    licence: str

    @property
    def url(self) -> str:
        """The hub's own parquet endpoint — public, no token, redirects to the CDN."""
        return (
            f"https://huggingface.co/api/datasets/{self.dataset}/parquet/"
            f"{self.config}/{self.split}/{self.shard}.parquet"
        )

    @property
    def page(self) -> str:
        return f"https://huggingface.co/datasets/{self.dataset}"


#: The three acquired domains. Every one was probed non-gated (`gated: false` on
#: the hub API, 2026-10-18); the research doc's first choices were The Stack v2
#: (gated) and FineMath (not on the hub under that name) — see the run report.
SOURCES: dict[str, Source] = {
    "code": Source(
        "code",
        "codeparrot/codeparrot-clean",
        "default",
        "train",
        0,
        "content",
        "MIT (per the dataset card)",
    ),
    "math": Source("math", "open-web-math/open-web-math", "default", "train", 0, "text", "ODC-By"),
    "ru": Source("ru", "HuggingFaceFW/fineweb-2", "rus_Cyrl", "train", 0, "text", "ODC-By"),
}


def fetch(con: duckdb.DuckDBPyConnection, src: Source, out: Path, cap: int) -> int:
    """Write at most `cap` bytes of `src`'s text column to `out`; return the size.

    Two queries: a 500-row probe for the mean document size, then one `LIMIT`
    sized for the cap. The write stops at the cap even if more rows arrive, so
    the cap is a cap and not a target.
    """
    probe = con.execute(
        f"SELECT avg(strlen({src.column})), count(*) FROM "
        f"(SELECT {src.column} FROM read_parquet('{src.url}') LIMIT 500)"
    ).fetchone()
    mean = float(probe[0] or 0.0)
    if mean <= 0:
        raise RuntimeError(f"the probe read no text from {src.column} (mean {mean})")
    rows = int(cap / mean * 1.25) + 1000
    print(f"  probe: {mean:.0f} B/doc, asking for {rows} rows ({src.url})")
    cur = con.execute(f"SELECT {src.column} FROM read_parquet('{src.url}') LIMIT {rows}")
    written = 0
    docs = 0
    with out.open("wb") as fh:
        while written < cap:
            batch = cur.fetchmany(2000)
            if not batch:
                break
            for (text,) in batch:
                if not isinstance(text, str):
                    continue
                raw = text.encode("utf-8", "replace")
                if written + len(raw) + 1 > cap:
                    return written
                fh.write(raw + b"\n")
                written += len(raw) + 1
                docs += 1
    print(f"  {docs} docs, {written} B")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", action="append", choices=sorted(SOURCES), default=None)
    ap.add_argument("--cap-mb", type=int, default=120, help="per source (default 120)")
    ap.add_argument("--total-cap-mb", type=int, default=DEFAULT_TOTAL_CAP_MB)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="re-download a slice that exists")
    args = ap.parse_args(argv)

    names = args.source or sorted(SOURCES)
    cap = args.cap_mb << 20
    total_cap = args.total_cap_mb << 20
    args.out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    done: dict[str, int] = {}
    failed: list[str] = []
    for name in names:
        src = SOURCES[name]
        out = args.out_dir / f"{name}.bin"
        if out.exists() and not args.force:
            print(f"{name}: {out} exists ({out.stat().st_size} B) — skipping")
            done[name] = out.stat().st_size
            continue
        if sum(done.values()) >= total_cap:
            print(f"{name}: SKIPPED — the {args.total_cap_mb} MB total cap is already spent")
            failed.append(name)
            continue
        print(f"{name}: {src.page} ({src.licence})")
        t = time.time()
        try:
            size = fetch(con, src, out, cap)
        except Exception as e:  # noqa: BLE001 — a fetch failure is reported, never faked
            if out.exists():
                out.unlink()
            print(f"{name}: FAILED ({type(e).__name__}: {e})\n  url: {src.url}")
            failed.append(name)
            continue
        print(f"{name}: {out} {size} B in {time.time() - t:.0f} s")
        done[name] = size

    print("\n--- fetched ---")
    for name in names:
        print(f"  {name:5s} {done.get(name, 0):>12d} B  {args.out_dir / f'{name}.bin'}")
    if failed:
        print(f"\nNEEDS MANUAL FETCHING: {failed}")
        for name in failed:
            print(f"  {name}: {SOURCES[name].url} ({SOURCES[name].page})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
