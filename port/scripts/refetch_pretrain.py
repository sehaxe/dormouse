# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb>=1.3"]
# ///
"""Refetch the pretraining sources after the 2026-11-08 artifact loss.

The merge deleted `data_local/` (80 GB). The sources are open Hub datasets, and
DuckDB reads parquet over HTTPS with range requests, so a capped slice costs
roughly the bytes it returns. This writes the three largest domains back to the
paths `configs/mix_v7.toml` names.

    uv run scripts/refetch_pretrain.py --cap-mb 2000
    uv run scripts/refetch_pretrain.py --only fwedu

It is NOT a corpus build: `data_local/best/<name>.bin` is raw UTF-8 text, the
same shape the trainer's flat reader takes. The mixture's small domains
(code/math/RU) come from `fetch_mix_sources.py`; the eval window must be carved
fresh and every number measured on it is a NEW number (the old window is gone).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class Slice:
    name: str
    dataset: str
    config: str
    shard: int
    licence: str

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/api/datasets/{self.dataset}/parquet/"
            f"{self.config}/train/{self.shard}.parquet"
        )


SLICES = {
    # the three largest shares of configs/mix_v7.toml
    "fwedu": Slice("fwedu", "HuggingFaceFW/fineweb-edu", "default", 0, "ODC-By"),
    "cosmo": Slice("cosmo", "HuggingFaceTB/cosmopedia", "auto_math_text", 0, "ODC-By"),
    "finemath": Slice("finemath", "HuggingFaceTB/finemath", "finemath-3plus", 0, "ODC-By"),
}

OUT_DIR = "/home/sehaxe/dormouse/data_local/best"


def text_column(con: duckdb.DuckDBPyConnection, url: str) -> str:
    """The widest VARCHAR column of the shard — `text` in all three, checked."""
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{url}')").fetchall()
    names = [r[0] for r in rows if r[1] in ("VARCHAR", "STRING")]
    if not names:
        raise SystemExit(f"no VARCHAR column in {url}: {[r[:2] for r in rows]}")
    return "text" if "text" in names else names[0]


def fetch(con: duckdb.DuckDBPyConnection, s: Slice, cap: int) -> int:
    t0 = time.time()
    col = text_column(con, s.url)
    # one row ≈ 6-8 kB of UTF-8; ask for a little more than the cap and stop
    rows = max(1, int(cap / 6500))
    q = (
        f"SELECT {col} FROM read_parquet('{s.url}') LIMIT {rows}"
    )
    data = con.execute(q).fetchall()
    out = f"{OUT_DIR}/{s.name}.bin"
    n = 0
    with open(out, "wb") as fh:
        for (text,) in data:
            if not isinstance(text, str):
                continue
            b = text.encode("utf-8", "replace")
            fh.write(b)
            fh.write(b"\n\n---\n\n")
            n += len(b) + 8
            if n >= cap:
                break
    print(f"{s.name}: {n / 2**20:.0f} MiB in {time.time() - t0:.0f}s -> {out}", flush=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-mb", type=int, default=2000, help="bytes per source")
    ap.add_argument("--only", default="", help="one source name")
    a = ap.parse_args()
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    cap = a.cap_mb * 2**20
    total = 0
    for name, s in SLICES.items():
        if a.only and name != a.only:
            continue
        try:
            total += fetch(con, s, cap)
        except Exception as e:  # noqa: BLE001 - a source that cannot be reached must be seen
            print(f"loud: {name} failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
    print(f"total {total / 2**30:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
