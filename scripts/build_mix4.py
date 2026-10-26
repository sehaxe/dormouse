#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow>=15", "numpy>=1.26"]
# ///
"""Build `data_local/mix4/`: the 5.75 GB pretraining mixture, one byte shard per domain.

This is the builder the `mix_v4.toml` plan points at. It is deterministic
(one seed, sorted inputs, no global RNG stream), resumable (a per-domain state
file records the source cursor and the partial file is truncated to that
cursor), and idempotent (a finished domain is skipped when its build key still
matches the sources).

Every domain carves its EVAL bytes FIRST and excludes them from the training
shard by source range or document SHA-1. `finish()` does not trust the
construction: it proves the exclusion and refuses to publish a shard on a
failed proof.

`--near-dup` and `--quality-band` are OFF by default (the mix4 bytes reproduce
exactly). With them on, the two crawl domains (`web`, `ruweb`) are cut into
line-aligned 8 KiB chunks, near-duplicates are removed by MinHash LSH, and a
cheap heuristic band drops junk. The eval shard is carved first AND indexed
first, so a near-duplicate of an eval chunk is refused a train write.

    uv run port/scripts/build_mix4.py                 # all domains
    uv run port/scripts/build_mix4.py --only web wiki_en
    uv run port/scripts/build_mix4.py --force --only math
    uv run port/scripts/build_mix4.py --only web ruweb --out-dir data_local/mix5 \\
        --near-dup --quality-band --max-source-bytes 1200000000

The code domain is `codeparrot/github-code-clean` (Apache-2.0 dataset; the card
says the files are permissively licensed). `bigcode/starcoderdata` is gated and
an anonymous download returns HTTP 401, so it is NOT the source.

Licences: codeparrot/github-code-clean Apache-2.0; wikimedia/wikipedia
CC BY-SA 4.0 + GFDL; the local `aria_data` corpora are the operator's own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIX = ROOT / "data_local" / "mix4"
RAW = ROOT / "data_local" / "mix4_src"
MOUNT = Path("/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/mix")

SEED = 20261022
FILTER_VERSION = 4
UNIT = 256 << 10
EVAL_TARGET = 4 << 20

#: Near-duplicate removal and the quality band run on the two crawl domains.
CRAWL_DOMAINS = ("web", "ruweb")
#: The chunk is the document unit of both stages (line-aligned, lossless: the
#: kept chunks concatenated are a byte subsequence of the shard).
CHUNK_BYTES = 8 << 10
NEAR_DUP_PERMS = 128
NEAR_DUP_BANDS = 16
NEAR_DUP_ROWS = NEAR_DUP_PERMS // NEAR_DUP_BANDS  # 8
#: Banding target. P(collide at J=0.8) = 1 - (1 - 0.8**8)**16 = 0.946.
NEAR_DUP_THRESHOLD = 0.8
#: The 128-perm estimate of a true 0.80 pair has sigma = 0.035, so the
#: verification accepts a 0.05 margin below the banding target.
NEAR_DUP_VERIFY = 0.75
#: Loud ceiling on the index: beyond this the run refuses instead of growing.
NEAR_DUP_MAX_DOCS = 1_500_000

# Byte targets at a 5.75 GB total. Percentages are in docs/guides/dataset.md.
TARGETS = {
    "web": 2_346_000_000,
    "wiki_en": 460_000_000,
    "wiki_ru": 230_000_000,
    "ruweb": 862_500_000,
    "code": 690_000_000,
    "math": 632_500_000,
    "books": 57_500_000,
    "qa": 57_500_000,
    "reasoning": 172_500_000,
    "instruct": 120_500_000,
    "rudialog": 115_000_000,
}
DOMAIN_ORDER = tuple(TARGETS)

#: The code quotas, bytes. The sum is the code target; a short language
#: (Rust/SQL in codeparrot-clean) is filled from Python/JS at the end.
LANG_QUOTA = {
    "Python": 190_000_000,
    "JavaScript": 150_000_000,
    "TypeScript": 130_000_000,
    "C": 100_000_000,
    "Go": 60_000_000,
    "SQL": 40_000_000,
    "Rust": 20_000_000,
}

CODE_URL = "https://huggingface.co/api/datasets/codeparrot/github-code-clean/parquet"
CODE_FILES = [
    (f"{CODE_URL}/all-mit/train/{i}.parquet", RAW / "codeparrot_mit" / f"train-{i}.parquet")
    for i in range(10)
] + [
    (
        f"{CODE_URL}/all-apache-2.0/train/{i}.parquet",
        RAW / "codeparrot_apache" / f"train-{i}.parquet",
    )
    for i in range(2)
]
RUWIKI_URL = (
    "https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.ru/train/2.parquet"
)
RUWIKI_FILE = RAW / "wikipedia_ru" / "train-2.parquet"

INSTRUCT_FILES = [
    ROOT / "port" / "sft_pairs" / "aria_ru.jsonl",
    ROOT / "port" / "sft_pairs" / "aria_en.jsonl",
    ROOT / "port" / "sft_pairs" / "aria_reason.jsonl",
    ROOT / "data_local" / "distill_clean2.jsonl",
    ROOT / "data_local" / "more_short.jsonl",
]

SHARD_DIRS = {
    "web": MOUNT / "web",
    "wiki_en": MOUNT / "wiki",
    "ruweb": MOUNT / "ruweb",
    "reasoning": MOUNT / "reasoning",
    "rudialog": MOUNT / "rudialog",
}
PARQUET_SOURCES = {
    "wiki_ru": [RUWIKI_FILE],
    "books": [MOUNT / "books" / "005_00000.parquet"],
    "qa": sorted((MOUNT / "qa").glob("*.parquet")),
}

#: The 0x01/0x02/0x03 template, duplicated from `dormouse.train.sft.render`
#: so this builder has no torch import. The bytes are the contract; the test
#: pins them against the literals.
USER, ASSIST, END = 0x01, 0x02, 0x03

EQ_RE = re.compile(rb"\\begin\{|\\frac|\\boxed|\\\[|\\\(|\$\$|\$[^$\n]{1,200}\$")
WORD_RE = re.compile(r"[A-Za-zА-Яа-я]{2,}")
LINK_RE = re.compile(r"https?://|www\.")
VENDOR_RE = re.compile(
    r"(^|/)(vendor|vendored|third_party|third-party|node_modules|dist|build|"
    r"bower_components|\.git|generated|autogen)(/|$)",
    re.IGNORECASE,
)
GENERATED_MARKERS = (
    b"generated by",
    b"do not edit",
    b"auto-generated",
    b"automatically generated",
    b"this file is generated",
    b"code generated",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def pread(path: Path, offset: int, length: int) -> bytes:
    with path.open("rb") as f:
        f.seek(offset)
        return f.read(length)


def iter_chunks(raw: bytes, size: int = CHUNK_BYTES):
    """Line-aligned, lossless chunks: `b"".join(kept)` is a subsequence of `raw`."""
    start = 0
    n = len(raw)
    while start < n:
        end = min(n, start + size)
        if end < n:
            nl = raw.rfind(b"\n", start, end)
            if nl > start:
                end = nl + 1
        yield start, raw[start:end]
        start = end


def file_contains(path: Path, needle: bytes) -> bool:
    """True when `needle` occurs anywhere in `path` — the eval-leak proof."""
    if not needle:
        return True
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        return m.find(needle) >= 0


def stable_offset(salt: str, maximum: int) -> int:
    if maximum <= 0:
        return 0
    raw = hashlib.sha256(f"{SEED}:{salt}".encode()).digest()[:8]
    return int.from_bytes(raw, "big") % (maximum + 1)


def dir_fingerprint(path: Path, pattern: str = "*.txt") -> dict:
    files = sorted(path.glob(pattern))
    return {
        "dir": str(path),
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "names_sha1": sha1("".join(p.name for p in files).encode()),
    }


def file_fingerprint(files: list[Path]) -> dict:
    return {
        "files": [{"path": str(p), "size": p.stat().st_size} for p in files if p.exists()],
    }


def source_fingerprint(name: str) -> dict:
    if name in SHARD_DIRS:
        return dir_fingerprint(SHARD_DIRS[name])
    if name == "math":
        return dir_fingerprint(MOUNT / "math")
    if name == "code":
        return file_fingerprint([p for _, p in CODE_FILES])
    if name == "wiki_ru":
        return file_fingerprint([RUWIKI_FILE])
    if name == "instruct":
        return file_fingerprint(INSTRUCT_FILES)
    if name in PARQUET_SOURCES:
        return file_fingerprint(PARQUET_SOURCES[name])
    raise ValueError(f"unknown domain {name!r}")


# --- downloads --------------------------------------------------------------


def download(url: str, dest: Path) -> None:
    """Fetch `url` to `dest`, resuming a `.part` with an HTTP Range request."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "dormouse-mix4-builder/1"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"download {url} failed: HTTP {e.code}") from e
    if have and resp.status == 200:
        have = 0  # the server ignored Range: restart rather than append twice
    mode = "ab" if have else "wb"
    log(f"  fetch {url} -> {dest} ({have} B present)")
    with resp, part.open(mode) as f:
        shutil.copyfileobj(resp, f, 1 << 20)
    os.replace(part, dest)


def _valid_parquet(path: Path) -> bool:
    try:
        import pyarrow.parquet as pq

        _ = pq.ParquetFile(path).metadata.num_rows
        return True
    except Exception:  # noqa: BLE001 — a truncated download is re-fetched, not trusted
        return False


def ensure_code_files() -> list[Path]:
    out = []
    for url, path in CODE_FILES:
        if not path.exists() or path.stat().st_size == 0 or not _valid_parquet(path):
            path.unlink(missing_ok=True)
            download(url, path)
        out.append(path)
    return out


def ensure_ruwiki() -> Path:
    if (
        not RUWIKI_FILE.exists()
        or RUWIKI_FILE.stat().st_size == 0
        or not _valid_parquet(RUWIKI_FILE)
    ):
        RUWIKI_FILE.unlink(missing_ok=True)
        download(RUWIKI_URL, RUWIKI_FILE)
    return RUWIKI_FILE


# --- the domain writer ------------------------------------------------------


class Domain:
    """One `.bin` domain: a partial train file, a partial eval file, a state file."""

    def __init__(
        self,
        name: str,
        target: int,
        out_dir: Path,
        *,
        eval_target: int = EVAL_TARGET,
        near_dup: bool = False,
        quality_band: bool = False,
        max_source_bytes: int = 0,
    ) -> None:
        self.name = name
        self.target = target
        self.eval_target = eval_target
        self.out_dir = out_dir
        self.near_dup = near_dup
        self.quality_band = quality_band
        self.max_source_bytes = max_source_bytes
        self.train_path = out_dir / f"{name}.bin"
        self.eval_path = out_dir / f"{name}_eval.bin"
        self.state_path = out_dir / f".{name}.state.json"
        self.ptrain = out_dir / f".{name}.train.part"
        self.peval = out_dir / f".{name}.eval.part"
        self.fh_train = None
        self.fh_eval = None
        self._last_save = 0.0
        self._last_bytes = 0
        self.state: dict = {}
        self.key = ""

    def _key(self) -> str:
        spec = {
            "v": FILTER_VERSION,
            "name": self.name,
            "target": self.target,
            "eval_target": self.eval_target,
            "seed": SEED,
            "source": source_fingerprint(self.name),
        }
        # Only when a stage is on: the mix4 build key of a default run must not
        # change, or every finished mix4 domain would be rebuilt on a rerun.
        if self.near_dup or self.quality_band or self.max_source_bytes:
            spec.update(
                near_dup=self.near_dup,
                quality_band=self.quality_band,
                max_source_bytes=self.max_source_bytes,
                chunk_bytes=CHUNK_BYTES,
            )
        return sha1(json.dumps(spec, sort_keys=True).encode())

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {}

    def save_state(self) -> None:
        self.state["train_sha1_count"] = len(self.state.get("train_sha1", []))
        atomic_json(self.state_path, self.state)
        self._last_save = time.monotonic()
        self._last_bytes = self.state.get("train_bytes", 0)

    def start(self) -> bool:
        """Open the partials for writing. False = this domain is already done."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.key = self._key()
        self.state = self._load_state()
        if self.state.get("key") != self.key:
            for p in (self.ptrain, self.peval):
                p.unlink(missing_ok=True)
            self.state = {
                "key": self.key,
                "phase": "eval",
                "cursor": 0,
                "doc_i": 0,
                "train_bytes": 0,
                "eval_bytes": 0,
                "train_units": 0,
                "eval_units": 0,
                "drops": {},
                "eval_sha1": [],
                "train_sha1": [],
                "eval_windows": [],
                "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        if self.state.get("phase") == "done" and not (
            self.train_path.exists() and self.eval_path.exists()
        ):
            # The data files were removed after the state was written: rebuild.
            for p in (self.ptrain, self.peval, self.state_path):
                p.unlink(missing_ok=True)
            self.state = {}
            return self.start()
        if (
            self.train_path.exists()
            and self.eval_path.exists()
            and self.state.get("phase") == "done"
        ):
            return False
        for path, recorded in (
            (self.ptrain, self.state.get("train_bytes", 0)),
            (self.peval, self.state.get("eval_bytes", 0)),
        ):
            if not path.exists():
                path.write_bytes(b"")
            size = path.stat().st_size
            if size < recorded:
                raise RuntimeError(
                    f"{self.name}: {path} is {size} B but state recorded {recorded} B"
                )
            if size > recorded:
                with path.open("r+b") as f:
                    f.truncate(recorded)
        self.fh_train = self.ptrain.open("ab")
        self.fh_eval = self.peval.open("ab")
        return True

    def _maybe_save(self) -> None:
        now = time.monotonic()
        if now - self._last_save > 20 or self.state["train_bytes"] - self._last_bytes > (1 << 28):
            self.save_state()

    def write_train(self, data: bytes, unit_sha: str | None = None) -> None:
        assert self.fh_train is not None
        self.fh_train.write(data)
        self.state["train_bytes"] += len(data)
        self.state["train_units"] += 1
        if unit_sha is not None:
            self.state["train_sha1"].append(unit_sha)
        self._maybe_save()

    def write_eval(self, data: bytes, unit_sha: str | None = None) -> int:
        assert self.fh_eval is not None
        offset = self.state["eval_bytes"]
        self.fh_eval.write(data)
        self.state["eval_bytes"] += len(data)
        self.state["eval_units"] += 1
        if unit_sha is not None:
            self.state["eval_sha1"].append(unit_sha)
        return offset

    def drop(self, reason: str, nbytes: int = 0) -> None:
        entry = self.state["drops"].setdefault(reason, {"units": 0, "bytes": 0})
        entry["units"] += 1
        entry["bytes"] += nbytes

    def _prove(self) -> dict:
        windows = self.state.get("eval_windows") or []
        if windows:
            ev = self.peval.read_bytes()
            checks = []
            for w in windows:
                window = ev[w["offset"] : w["offset"] + w["len"]]
                if len(window) != w["len"]:
                    raise RuntimeError(f"{self.name}: eval window is short")
                found = file_contains(self.ptrain, window)
                checks.append(
                    {
                        "source": w.get("source", ""),
                        "source_offset": w.get("source_offset", 0),
                        "len": len(window),
                        "sha256": sha256(window),
                        "found_in_train": found,
                    }
                )
                if found:
                    raise RuntimeError(
                        f"{self.name}: eval window {w.get('source')} found in the train shard"
                    )
            return {
                "method": "window_absent",
                "windows": len(checks),
                "checks": checks,
                "pass": True,
            }
        ev = set(self.state.get("eval_sha1", []))
        tr = set(self.state.get("train_sha1", []))
        overlap = sorted(ev & tr)
        if overlap:
            raise RuntimeError(f"{self.name}: {len(overlap)} eval docs leaked into train")
        return {
            "method": "doc_sha1_disjoint",
            "eval_docs": len(ev),
            "train_docs": len(tr),
            "overlap": 0,
            "eval_unit_sha1": sorted(ev),
            "pass": True,
        }

    def finish(self) -> dict:
        assert self.fh_train is not None and self.fh_eval is not None
        self.fh_train.flush()
        self.fh_train.close()
        self.fh_eval.flush()
        self.fh_eval.close()
        if self.state["train_bytes"] == 0:
            raise RuntimeError(f"{self.name}: no training bytes were written")
        proof = self._prove()
        if self.near_dup:
            nd = self.state.get("near_dup") or {}
            if not nd.get("eval_chunks"):
                raise RuntimeError(
                    f"{self.name}: --near-dup indexed no eval chunks — the "
                    "near-duplicate no-leak proof would be vacuous"
                )
            collisions = self.state["drops"].get("near_dup_eval", {}).get("units", 0)
            if nd.get("eval_collisions", 0) != collisions:
                raise RuntimeError(
                    f"{self.name}: near-dup eval collisions {nd.get('eval_collisions')} "
                    f"!= dropped {collisions}"
                )
            proof["near_dup"] = {
                "eval_chunks_indexed": nd["eval_chunks"],
                "eval_collisions_dropped": collisions,
                "dropped_before_write": True,
                "pass": True,
            }
        train_sha = sha256_file(self.ptrain)
        eval_sha = sha256_file(self.peval)
        entry = {
            "target_bytes": self.target,
            "train_bytes": self.state["train_bytes"],
            "eval_bytes": self.state["eval_bytes"],
            "train_units": self.state["train_units"],
            "eval_units": self.state["eval_units"],
            "drops": self.state["drops"],
            "train_sha256": train_sha,
            "eval": {
                "file": self.eval_path.name,
                "sha256": eval_sha,
                "carve": self.state.get("eval_ranges") or self.state.get("eval_source"),
                "exclusion": proof,
            },
            "build_key": self.key,
        }
        if self.near_dup:
            entry["near_dup"] = self.state["near_dup"]
        if self.quality_band:
            entry["quality_band"] = self.state["quality_band"]
        if self.max_source_bytes:
            entry["max_source_bytes"] = self.max_source_bytes
            entry["source_bytes_read"] = self.state.get("source_bytes", 0)
        os.replace(self.ptrain, self.train_path)
        os.replace(self.peval, self.eval_path)
        self.state["phase"] = "done"
        self.state["entry"] = entry
        self.state["train_sha1"] = []
        self.save_state()
        return entry

    def saved_entry(self) -> dict:
        return self.state["entry"]


# --- filters ----------------------------------------------------------------


def cyr_ratio(text: str) -> float:
    cyr = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    alpha = sum(1 for c in text if c.isalpha())
    return cyr / max(1, alpha)


def alnum_ratio(raw: bytes) -> float:
    good = sum(1 for b in raw if 48 <= b <= 57 or 65 <= b <= 90 or 97 <= b <= 122)
    return good / max(1, len(raw))


def filter_wiki_en(raw: bytes) -> tuple[bool, str]:
    if len(raw) < 100_000:
        return False, "too_short"
    return True, ""


def filter_web(raw: bytes) -> tuple[bool, str]:
    """General English web: language, link density, uppercase, a size floor."""
    if len(raw) < 100_000:
        return False, "too_short"
    text = raw.decode("utf-8", "replace")
    alpha = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return False, "no_letters"
    cyr = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    if cyr / alpha > 0.02:
        return False, "non_english"
    ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
    if ascii_alpha / alpha < 0.80:
        return False, "non_english"
    if text.count("http") > len(raw) * 0.002:
        return False, "link_density"
    upper = sum(1 for c in text if c.isupper())
    if upper / alpha > 0.40:
        return False, "uppercase"
    return True, ""


def filter_ruweb(raw: bytes) -> tuple[bool, str]:
    if len(raw) < 100_000:
        return False, "too_short"
    text = raw.decode("utf-8", "replace")
    if cyr_ratio(text) < 0.50:
        return False, "non_russian"
    if text.count("http") > len(raw) * 0.001:
        return False, "link_density"
    alpha = sum(1 for c in text if c.isalpha())
    upper = sum(1 for c in text if c.isupper())
    if alpha and upper / alpha > 0.35:
        return False, "uppercase"
    return True, ""


def filter_rudialog(raw: bytes) -> tuple[bool, str]:
    if len(raw) < 100_000:
        return False, "too_short"
    text = raw.decode("utf-8", "replace")
    if cyr_ratio(text) < 0.40:
        return False, "non_russian"
    return True, ""


def filter_reasoning(raw: bytes) -> tuple[bool, str]:
    if len(raw) < 100_000:
        return False, "too_short"
    if raw.count(b"<|begin_of_thought|>") < 20:
        return False, "no_thought_traces"
    return True, ""


def filter_math_doc(doc: bytes) -> tuple[bool, str]:
    """Equation + >=120 prose words, no link list / forum chatter. Whole doc or drop."""
    n = len(doc)
    if n < 400:
        return False, "too_short"
    if n > 200_000:
        return False, "too_long"
    text = doc.decode("utf-8", "replace")
    if not EQ_RE.search(doc):
        return False, "no_equation"
    words = len(WORD_RE.findall(text))
    if words < 120:
        return False, "too_few_words"
    links = len(LINK_RE.findall(text))
    if links > max(2, words // 150):
        return False, "link_density"
    alpha = sum(1 for c in text if c.isalpha())
    upper = sum(1 for c in text if c.isupper())
    if alpha and upper / alpha > 0.25:
        return False, "uppercase"
    if text.count(">>") >= 3:
        return False, "forum_quote"
    lines = [ln for ln in text.split("\n") if ln.strip()]
    quote_lines = sum(1 for ln in lines if ln.lstrip().startswith(">"))
    if quote_lines > max(3, len(lines) // 20):
        return False, "forum_quote"
    link_lines = sum(1 for ln in lines if "http" in ln or ln.lstrip().startswith(("* ", "- ")))
    if link_lines > max(5, len(lines) // 5):
        return False, "link_list"
    return True, ""


def split_math_docs(raw: bytes) -> list[bytes]:
    """The math shards use `\\n---\\n` between generated sections; a section is a doc."""
    return [d for d in raw.split(b"\n---\n") if d.strip()]


def filter_code(row: dict) -> tuple[bytes | None, str, str]:
    """One GitHub file: size, lines, alnum, vendored/generated/minified."""
    code = row.get("code") or ""
    path = row.get("path") or ""
    lang = row.get("language") or ""
    lang = "Go" if lang == "GO" else lang
    if lang not in LANG_QUOTA:
        return None, "language", lang
    raw = code.encode("utf-8", "replace")
    if len(raw) < 200:
        return None, "too_short", lang
    if len(raw) > 100_000:
        return None, "too_long", lang
    if raw.count(b"\n") + 1 < 3:
        return None, "too_few_lines", lang
    if alnum_ratio(raw) < 0.20:
        return None, "alnum", lang
    if VENDOR_RE.search(path) or path.endswith((".min.js", ".min.css")):
        return None, "vendored_minified_path", lang
    if raw.count(b"\n"):
        max_line = max((len(ln) for ln in raw.split(b"\n")), default=0)
        avg_line = len(raw) / raw.count(b"\n")
        if max_line > 2000 or avg_line > 200:
            return None, "minified", lang
    head = raw[:4096].lower()
    if any(marker in head for marker in GENERATED_MARKERS):
        return None, "generated", lang
    return raw, "", lang


def render_pair(instruction: str, answer: str) -> bytes:
    q, a = instruction.encode("utf-8"), answer.encode("utf-8")
    for raw in (q, a):
        for sentinel in (0, USER, ASSIST, END):
            if bytes([sentinel]) in raw:
                raise AssertionError(f"sentinel 0x{sentinel:02x} in corpus text")
    return bytes([USER]) + q + b"\n" + bytes([ASSIST]) + a + bytes([END])


def filter_books(row: dict) -> tuple[bytes | None, str]:
    if row.get("language") != "en":
        return None, "wrong_language"
    if (row.get("int_score") or 0) < 3:
        return None, "low_int_score"
    if (row.get("token_count") or 0) < 200:
        return None, "short_tokens"
    raw = (row.get("text") or "").encode("utf-8", "replace")
    if not 500 <= len(raw) <= 100_000:
        return None, "size"
    if alnum_ratio(raw) < 0.40:
        return None, "alnum"
    return raw + b"\n\n", ""


def filter_wiki_ru(row: dict) -> tuple[bytes | None, str]:
    text = row.get("text") or ""
    raw = text.encode("utf-8", "replace")
    if not 300 <= len(raw) <= 200_000:
        return None, "size"
    if text.lower().startswith("#redirect"):
        return None, "redirect"
    if cyr_ratio(text) < 0.15:
        return None, "non_russian"
    title = (row.get("title") or "").strip()
    head = f"# {title}\n\n".encode() if title else b""
    return head + raw + b"\n\n", ""


def strip_html(html: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    return re.sub(r"\s+", " ", text).strip()


def render_qa(row: dict) -> tuple[bytes | None, str]:
    q = (row.get("question") or {}).get("text") or ""
    ann = row.get("annotations") or {}
    short = ""
    answers = ann.get("short_answers") or []
    if answers and answers[0].get("text"):
        short = answers[0]["text"][0]
    long = ""
    if not short:
        long_answers = ann.get("long_answer") or []
        if long_answers:
            la = long_answers[0]
            start, end = la.get("start_token"), la.get("end_token")
            if start is not None and start >= 0:
                toks = (row.get("document") or {}).get("tokens", {}).get("token", [])
                long = " ".join(toks[start:end])
    answer = short or long
    if not q or not answer:
        return None, "no_answer"
    html = (row.get("document") or {}).get("html") or ""
    article = strip_html(html)
    if len(article) < 500:
        return None, "article_too_short"
    if len(article) > 200_000:
        return None, "article_too_long"
    out = f"Question: {q}\nArticle: {article}\nAnswer: {answer}\n\n".encode()
    return out, ""


# --- near-duplicate removal (MinHash LSH) -----------------------------------
#
# The corpus does not fit in RAM. A shard is cut into line-aligned chunks (the
# unit the builder writes), each chunk is shingled at word 5-grams (byte
# 5-grams when the chunk has no spaces, e.g. CJK), MinHashed with 128
# permutations, and split into 16 bands x 8 rows. A pair at Jaccard >= 0.8
# collides in at least one band with P = 1 - (1 - 0.8**8)**16 = 0.946, so a
# candidate is found without comparing every pair. The candidate is then
# verified on the 128-value signature (with a 0.05 estimator margin, because
# the standard error of a 128-perm estimate at J=0.8 is 0.035), and the exact
# Jaccard of the first five drops is recomputed from the bytes so the number
# is auditable.
#
# Memory is bounded by the index, not the corpus:
#     n_docs * 128 * 4 B signatures
#   + 16 band keys and one (path, offset, len) per kept document.
# The shingles of the chunk being processed are the only text held.
# `NearDupIndex.signature_nbytes()` is the gate.

_NUMPY = None


def _np():
    """numpy, imported lazily so the mix4 byte-path tests run without it."""
    global _NUMPY
    if _NUMPY is None:
        import numpy

        _NUMPY = numpy
    return _NUMPY


_QPOW = None
_RPOW = None
_Q = 0x9E3779B97F4A7C15  # odd: a bijection modulo 2**64
_R = 0xC2B2AE3D27D4EB4F
_MIX = 0x9E3779B97F4A7C15


def _qpow(n: int):
    global _QPOW
    np = _np()
    if _QPOW is None or len(_QPOW) <= n:
        size = max(n + 1, 4096)
        p = np.empty(size, np.uint64)
        p[0] = 1
        v = 1
        for i in range(1, size):
            v = (v * _Q) & 0xFFFFFFFFFFFFFFFF
            p[i] = v
        _QPOW = p
    return _QPOW


def _rpow():
    global _RPOW
    np = _np()
    if _RPOW is None:
        p = np.empty(5, np.uint64)
        p[0] = 1
        v = 1
        for i in range(1, 5):
            v = (v * _R) & 0xFFFFFFFFFFFFFFFF
            p[i] = v
        _RPOW = p
    return _RPOW


def _perm_ab():
    """128 (a, b) multiply-shift permutations, a pure function of SEED."""
    np = _np()
    rng = np.random.default_rng(SEED)
    a = rng.integers(1, 1 << 63, size=NEAR_DUP_PERMS, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 1 << 63, size=NEAR_DUP_PERMS, dtype=np.uint64)
    return a, b


_PERM_A = None
_PERM_B = None


def _perms():
    global _PERM_A, _PERM_B
    if _PERM_A is None:
        _PERM_A, _PERM_B = _perm_ab()
    return _PERM_A, _PERM_B


class ChunkFeatures:
    """What one chunk contributes: word hashes, 5-gram hashes, char counts.

    `words` are 64-bit hashes of the whitespace tokens (empty in byte mode);
    `shingles` are 64-bit hashes of consecutive 5-grams. The quality band and
    the MinHash signature share them, so a chunk is tokenised once.
    """

    __slots__ = ("nbytes", "n_words", "word_bytes", "words", "shingles", "space_frac", "byte_mode")

    def __init__(self, nbytes, n_words, word_bytes, words, shingles, space_frac, byte_mode):
        self.nbytes = nbytes
        self.n_words = n_words
        self.word_bytes = word_bytes
        self.words = words
        self.shingles = shingles
        self.space_frac = space_frac
        self.byte_mode = byte_mode


def _hash_word(word: bytes):
    np = _np()
    b = np.frombuffer(word, np.uint8).astype(np.uint64)
    if b.size == 0:
        return np.uint64(0)
    h = (b * _qpow(b.size)[: b.size]).sum(dtype=np.uint64)
    return h ^ np.uint64((b.size * _MIX) & 0xFFFFFFFFFFFFFFFF)


def chunk_features(raw: bytes) -> ChunkFeatures:
    """Tokenise one chunk and hash its word 5-grams (byte 5-grams if spaceless)."""
    np = _np()
    n = len(raw)
    b = np.frombuffer(raw, np.uint8)
    if n == 0:
        empty = np.empty(0, np.uint64)
        return ChunkFeatures(0, 0, 0, empty, empty, 0.0, True)
    space = b <= 32
    nspace = int(space.sum())
    prev = np.empty(n, dtype=bool)
    prev[0] = True
    prev[1:] = space[:-1]
    nxt = np.empty(n, dtype=bool)
    nxt[-1] = True
    nxt[:-1] = space[1:]
    starts = np.flatnonzero(~space & prev)
    ends = np.flatnonzero(~space & nxt) + 1
    lens = ends - starts
    total = int(lens.sum())
    byte_mode = nspace / n < 0.03 or len(starts) < 5
    if byte_mode:
        words = np.empty(0, np.uint64)
        shingles = _byte_shingles(b)
    else:
        base = np.cumsum(lens) - lens
        rel = np.arange(total, dtype=np.int64) - np.repeat(base, lens)
        idx = np.repeat(starts, lens) + rel
        q = _qpow(int(lens.max()))
        vals = b[idx].astype(np.uint64) * q[rel]
        words = np.add.reduceat(vals, base)
        words = words ^ (lens.astype(np.uint64) * np.uint64(_MIX))
        shingles = _word_shingles(words)
    return ChunkFeatures(n, len(starts), total, words, shingles, nspace / n, byte_mode)


def _word_shingles(words):
    np = _np()
    if words.size < 5:
        return np.empty(0, np.uint64)
    r = _rpow()
    out = words[:-4] * r[4]
    for k in range(1, 5):
        out = out + words[k : words.size - 4 + k] * r[4 - k]
    return out


def _byte_shingles(b):
    np = _np()
    if b.size < 5:
        return np.empty(0, np.uint64)
    q = _qpow(4)
    v = b.astype(np.uint64)
    out = v[:-4] * q[4]
    for k in range(1, 5):
        out = out + v[k : v.size - 4 + k] * q[4 - k]
    return out


def minhash_signature(feat: ChunkFeatures):
    """The 128-permutation MinHash signature, stored as uint32."""
    np = _np()
    a, b = _perms()
    best = np.full(NEAR_DUP_PERMS, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
    sh = feat.shingles
    if sh.size:
        for i in range(0, sh.size, 4096):
            block = sh[i : i + 4096]
            cand = block[:, None] * a[None, :] + b[None, :]  # uint64 wraps
            np.minimum(best, cand.min(axis=0), out=best)
    return (best & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def shingle_set(raw: bytes) -> set[bytes]:
    """Exact 5-gram set (word-joined, or raw bytes) for the audited Jaccard."""
    words = raw.split()
    if len(words) < 5 or sum(len(w) for w in words) / max(1, len(raw)) < 0.5:
        return {raw[i : i + 5] for i in range(max(0, len(raw) - 4))}
    return {b" ".join(words[i : i + 5]) for i in range(len(words) - 4)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


class NearDupIndex:
    """Streaming MinHash-LSH index. Keeps signatures and band keys, never text."""

    def __init__(self, *, threshold: float = NEAR_DUP_VERIFY, max_docs: int = NEAR_DUP_MAX_DOCS):
        np = _np()
        self.threshold = threshold
        self.max_docs = max_docs
        self.sig = np.empty((1024, NEAR_DUP_PERMS), dtype=np.uint32)
        self.n = 0
        self.bands: dict[int, object] = {}
        self.parent: list[int] = []
        self.is_eval: list[bool] = []
        self.ranges: list[tuple[str, int, int]] = []
        self.eval_docs = 0

    def signature_nbytes(self) -> int:
        """The stated bound: one 128 x 4 B row per document, nothing per byte."""
        return self.n * NEAR_DUP_PERMS * 4

    def _grow(self) -> None:
        np = _np()
        if self.n < len(self.sig):
            return
        self.sig = np.concatenate([self.sig, np.empty_like(self.sig)])
        assert len(self.sig) <= 2 * max(1024, self.n + 1)

    def _find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def _union(self, a: int, b: int) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # the first document stays the root

    def _band_keys(self, sig) -> list[int]:
        np = _np()
        rows = sig.reshape(NEAR_DUP_BANDS, NEAR_DUP_ROWS)
        keys = np.bitwise_xor.reduce(rows, axis=1)
        return [int(k) for k in keys]

    def _candidates(self, keys: list[int]):
        for key in keys:
            entry = self.bands.get(key)
            if entry is None:
                continue
            if isinstance(entry, int):
                yield entry
            else:
                yield from entry

    def _remember(self, keys: list[int], doc_id: int) -> None:
        for key in keys:
            entry = self.bands.get(key)
            if entry is None:
                self.bands[key] = doc_id
            elif isinstance(entry, int):
                self.bands[key] = [entry, doc_id]
            else:
                entry.append(doc_id)

    def add(self, feat: ChunkFeatures, source: str, offset: int, *, is_eval: bool = False):
        """Index one chunk; return (reason, representative, est_jaccard) if dropped.

        `reason` is None for a kept document, `near_dup_eval` when the match is
        an eval chunk (never written), `near_dup` for a train duplicate.
        """
        if self.n >= self.max_docs:
            raise RuntimeError(
                f"near-dup index holds {self.n} docs (cap {self.max_docs}); "
                "raise NEAR_DUP_MAX_DOCS or lower --max-source-bytes"
            )
        sig = minhash_signature(feat)
        keys = self._band_keys(sig)
        self._grow()
        doc_id = self.n
        self.sig[doc_id] = sig
        self.n += 1
        self.parent.append(doc_id)
        self.is_eval.append(is_eval)
        self.ranges.append((source, offset, feat.nbytes))
        self._remember(keys, doc_id)
        if is_eval:
            self.eval_docs += 1
            return None, None, 0.0

        best_train: tuple[int | None, float] = (None, 0.0)
        best_eval: tuple[int | None, float] = (None, 0.0)
        for cid in self._candidates(keys):
            if cid == doc_id:
                continue  # its own band entry
            j = float((self.sig[cid] == sig).mean())
            if self.is_eval[cid]:
                if j > best_eval[1]:
                    best_eval = (cid, j)
            else:
                if j >= self.threshold:
                    self._union(doc_id, cid)
                if j > best_train[1]:
                    best_train = (cid, j)
        if best_eval[1] >= self.threshold:
            return "near_dup_eval", best_eval[0], best_eval[1]
        if best_train[1] >= self.threshold:
            return "near_dup", best_train[0], best_train[1]
        return None, None, 0.0

    def cluster_count(self) -> int:
        """Connected components over the train documents (the first doc is the root)."""
        return len({self._find(i) for i in range(self.n) if not self.is_eval[i]})


# --- the cheap quality band (no model) --------------------------------------

EN_STOP = frozenset(
    "the a an and or of to in is are was were be been being it its this that these those "
    "for on with as by at from not but you we they he she i his her their our your "
    "have has had do does did will would can could should may might must about into over".split()
)
RU_STOP = frozenset(
    "и в не на что я он с как это по но они к у из за от так же бы для мы вы "
    "то все она его ее они мне мой тебя ты был была было быть есть при о об "
    "если чем или ни уже только еще".split()
)
BOILERPLATE = (b"cookie", b"subscribe", b"sign in", b"privacy policy", b"accept all", b"log in")
_END_PUNCT = frozenset(b".!?\xe2\x80\xa6:;)]}\"'\xbb")
_STOP_HASHES: dict[str, object] = {}


def stopword_hashes(lang: str):
    np = _np()
    cached = _STOP_HASHES.get(lang)
    if cached is None:
        words = RU_STOP if lang == "ru" else EN_STOP
        cached = np.array([_hash_word(w.encode()) for w in sorted(words)], dtype=np.uint64)
        _STOP_HASHES[lang] = cached
    return cached


def _char_stats(raw: bytes) -> tuple[int, int, int, int, int]:
    """(alpha, upper, space, digit, chars); Cyrillic is 2 bytes/char, the rest is close."""
    np = _np()
    b = np.frombuffer(raw, np.uint8)
    if b.size == 0:
        return 0, 0, 0, 0, 0
    counts = np.bincount(b, minlength=256)
    ascii_alpha = int(counts[65:91].sum() + counts[97:123].sum())
    ascii_upper = int(counts[65:91].sum())
    cyr = int(
        np.count_nonzero((b[:-1] == 0xD0) & (b[1:] >= 0x80) & (b[1:] <= 0xBF))
        + np.count_nonzero((b[:-1] == 0xD1) & (b[1:] >= 0x80) & (b[1:] <= 0xBF))
    )
    cyr_upper = int(np.count_nonzero((b[:-1] == 0xD0) & (b[1:] >= 0x80) & (b[1:] <= 0xAF)))
    other = max(0, int(counts[128:].sum()) - 2 * cyr)
    alpha = ascii_alpha + cyr + other // 2
    upper = ascii_upper + cyr_upper
    space = int(counts[32] + counts[9] + counts[10] + counts[13])
    digit = int(counts[48:58].sum())
    nchars = int(b.size) - cyr
    return alpha, upper, space, digit, nchars


#: filter name -> (min, max) inclusive. A value outside the band drops the doc.
QUALITY_BANDS = {
    "words": (50, 100_000),
    "mean_word_len": (2.0, 20.0),
    "stopword_frac": (0.05, 1.0),
    "line_punct_frac": (0.30, 1.0),
    "list_line_frac": (0.0, 0.90),
    "repetition": (0.50, 1.0),
    "symbol_alpha": (0.0, 0.50),
    "uppercase_frac": (0.0, 0.40),
    "boilerplate_hits": (0, 2),
}


def quality_metrics(raw: bytes, feat: ChunkFeatures, lang: str) -> dict:
    np = _np()
    words = feat.n_words
    alpha, upper, space, digit, nchars = _char_stats(raw)
    lines = [ln for ln in raw.split(b"\n") if ln.strip()]
    punct = 0
    listy = 0
    for ln in lines:
        stripped = ln.rstrip()
        if stripped and stripped[-1] in _END_PUNCT:
            punct += 1
        head = stripped.lstrip()
        if head[:1] in (b"-", b"*", b"\xe2", b"\xc2", b"#") or (
            head[:1].isdigit() and (head[1:2] in (b".", b")"))
        ):
            listy += 1
    nlines = max(1, len(lines))
    distinct = int(np.unique(feat.shingles).size) if feat.shingles.size else 0
    sw = 0.0
    if not feat.byte_mode and words:
        sw = float(np.isin(feat.words, stopword_hashes(lang)).sum()) / words
    low = raw.lower()
    boiler = sum(low.count(m) for m in BOILERPLATE)
    metrics = {
        "words": words,
        "mean_word_len": feat.word_bytes / words if words else 0.0,
        "stopword_frac": sw,
        "line_punct_frac": punct / nlines,
        "list_line_frac": listy / nlines,
        "repetition": distinct / int(feat.shingles.size) if feat.shingles.size else 0.0,
        # `_char_stats` returns character counts: bytes minus the two-byte
        # Cyrillic letters is the character count for these two domains.
        "symbol_alpha": (nchars - alpha - space - digit) / alpha if alpha else 1.0,
        "uppercase_frac": upper / alpha if alpha else 1.0,
        "boilerplate_hits": boiler,
    }
    # The per-document score: the fraction of the nine bands it satisfies.
    metrics["score"] = sum(
        1 for name, (lo, hi) in QUALITY_BANDS.items() if lo <= metrics[name] <= hi
    ) / len(QUALITY_BANDS)
    return metrics


def quality_band(raw: bytes, feat: ChunkFeatures, lang: str) -> tuple[bool, str, dict, list[str]]:
    """First failing band wins. Returns (ok, reason, metrics, filters_checked)."""
    m = quality_metrics(raw, feat, lang)
    checked: list[str] = []
    for name, (lo, hi) in QUALITY_BANDS.items():
        checked.append(name)
        if not lo <= m[name] <= hi:
            return False, name, m, checked
    return True, "", m, checked


# --- generic parquet iteration ----------------------------------------------


def iter_parquet_rows(files: list[Path], columns: list[str], batch_size: int = 128):
    import pyarrow.parquet as pq

    idx = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                yield idx, row
                idx += 1


def parquet_rows(files: list[Path]) -> int:
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(p).metadata.num_rows for p in files)


# --- the domain builders ----------------------------------------------------


# `data_local/corpus_bin_filtered.bin` (19.7 GB) was measured and REJECTED as
# the web source: 200 random 4 KiB windows held LaTeX in 180, math keywords in
# 110 and code in 60; it is a math/code corpus, not general English web. The
# web share comes from the mount's `web/` shards instead.


def _shard_order(d: Domain) -> list[Path]:
    files = sorted(SHARD_DIRS[d.name].glob("*.txt"))
    if not files:
        raise RuntimeError(f"{d.name}: no shards in {SHARD_DIRS[d.name]}")
    rng = random.Random(SEED ^ int(sha1(d.name.encode())[:8], 16))
    order = list(files)
    rng.shuffle(order)
    return order


def _shard_filter(name: str):
    return {
        "web": filter_web,
        "wiki_en": filter_wiki_en,
        "ruweb": filter_ruweb,
        "reasoning": filter_reasoning,
        "rudialog": filter_rudialog,
    }[name]


def _carve_eval_shard(d: Domain, order: list[Path], filt) -> None:
    """The eval carve: FIRST, before any train byte, and one whole shard."""
    if d.state["phase"] != "eval":
        return
    chosen = None
    for idx in range(len(order) // 2, len(order)):
        raw = order[idx].read_bytes()
        ok, _ = filt(raw)
        if ok:
            chosen = order[idx]
            d.write_eval(raw, sha1(raw))
            d.state["eval_source"] = chosen.name
            log(f"  {d.name} eval shard {chosen.name} ({len(raw)} B)")
            break
    if chosen is None:
        raise RuntimeError(f"{d.name}: no shard passed the filter for the eval carve")
    d.state["phase"] = "train"
    d.state["cursor"] = 0
    d.state["doc_i"] = 0
    d.save_state()


def build_shards(d: Domain) -> None:
    order = _shard_order(d)
    filt = _shard_filter(d.name)
    _carve_eval_shard(d, order, filt)

    eval_set = set(d.state.get("eval_sha1", []))
    i = d.state["cursor"]
    src_read = d.state.get("source_bytes", 0)
    while i < len(order) and d.state["train_bytes"] < d.target:
        path = order[i]
        i += 1
        if path.name == d.state.get("eval_source"):
            continue
        if d.max_source_bytes and src_read + path.stat().st_size > d.max_source_bytes and src_read:
            break
        raw = path.read_bytes()
        src_read += len(raw)
        d.state["source_bytes"] = src_read
        ok, reason = filt(raw)
        if not ok:
            d.drop(reason, len(raw))
            continue
        h = sha1(raw)
        if h in eval_set:
            d.drop("eval_duplicate", len(raw))
            continue
        d.write_train(raw, h)
        d.state["cursor"] = i


def _near_dup_state(name: str) -> dict:
    return {
        "method": "minhash_lsh",
        "shingle": "word5",
        "perms": NEAR_DUP_PERMS,
        "bands": NEAR_DUP_BANDS,
        "rows": NEAR_DUP_ROWS,
        "threshold": NEAR_DUP_THRESHOLD,
        "verify_threshold": NEAR_DUP_VERIFY,
        "chunk_bytes": CHUNK_BYTES,
        "chunks_in": 0,
        "bytes_in": 0,
        "docs_in": 0,
        "docs_dropped": 0,
        "bytes_dropped": 0,
        "exact_duplicates": 0,
        "eval_chunks": 0,
        "eval_collisions": 0,
        "clusters": 0,
        "peak_signature_bytes": 0,
        "memory_bound": "docs * 128 * 4 B signatures + 16 band keys + 1 range per doc",
        "dropped_ranges": f".{name}.near_dup.jsonl",
        "spot_check": [],
    }


def _state_with_defaults(state: dict, key: str, defaults: dict) -> dict:
    """A resumed state may predate a counter; fill it in instead of KeyError."""
    cur = state.setdefault(key, defaults)
    for k, v in defaults.items():
        cur.setdefault(k, v)
    return cur


def _quality_state() -> dict:
    return {
        "bands": {k: list(v) for k, v in QUALITY_BANDS.items()},
        "filters": {k: {"checked": 0, "dropped": 0, "bytes": 0} for k in QUALITY_BANDS},
        "docs_in": 0,
        "bytes_in": 0,
        "docs_dropped": 0,
        "bytes_dropped": 0,
        "score_sum": 0.0,
        "score_n": 0,
        "mean_score": 0.0,
    }


def build_shards_quality(d: Domain) -> None:
    """`--near-dup` / `--quality-band` on a crawl domain.

    The eval shard is carved first and its chunks are indexed BEFORE any train
    chunk is read, so a near-duplicate of an eval chunk is refused a write
    (`near_dup_eval`). Chunks are line-aligned 8 KiB slices of the shard; the
    kept slices concatenated are a byte subsequence of the source, so the
    shard's bytes are never rewritten.
    """
    order = _shard_order(d)
    filt = _shard_filter(d.name)
    lang = "ru" if d.name == "ruweb" else "en"
    _carve_eval_shard(d, order, filt)

    index = None
    nd = None
    eval_source = d.state.get("eval_source")
    if d.near_dup:
        if not eval_source:
            raise RuntimeError(f"{d.name}: --near-dup needs the eval carve; none recorded")
        index = NearDupIndex()
        eval_path = SHARD_DIRS[d.name] / eval_source
        for off, chunk in iter_chunks(eval_path.read_bytes()):
            index.add(chunk_features(chunk), str(eval_path), off, is_eval=True)
        nd = _state_with_defaults(d.state, "near_dup", _near_dup_state(d.name))
        nd["eval_chunks"] = index.eval_docs
        log(f"  {d.name}: {index.eval_docs} eval chunks indexed for near-dup no-leak")
    qs = _state_with_defaults(d.state, "quality_band", _quality_state()) if d.quality_band else None

    exact: dict[str, int] = {}
    resumed = bool(d.state["cursor"] or d.state.get("doc_i"))
    log_fh = None
    if index is not None:
        log_fh = (d.out_dir / nd["dropped_ranges"]).open("a" if resumed else "w", encoding="utf-8")

    def apply(path: Path, chunk: bytes, off: int, *, write: bool) -> None:
        if write and nd is not None:
            nd["chunks_in"] += 1
            nd["bytes_in"] += len(chunk)
        feat = chunk_features(chunk)
        if d.quality_band:
            ok, reason, metrics, checked = quality_band(chunk, feat, lang)
            if write:
                qs["docs_in"] += 1
                qs["bytes_in"] += len(chunk)
                qs["score_sum"] += metrics["score"]
                qs["score_n"] += 1
                for name in checked:
                    qs["filters"][name]["checked"] += 1
            if not ok:
                if write:
                    qs["filters"][reason]["dropped"] += 1
                    qs["filters"][reason]["bytes"] += len(chunk)
                    qs["docs_dropped"] += 1
                    qs["bytes_dropped"] += len(chunk)
                    d.drop(f"quality:{reason}", len(chunk))
                return
        if index is None:
            if write:
                d.write_train(chunk, sha1(chunk))
            return
        h = sha1(chunk)
        if h in exact:
            if write:
                d.drop("duplicate", len(chunk))
                nd["exact_duplicates"] += 1
            return
        if write:
            nd["docs_in"] += 1
        reason, rep, est = index.add(feat, str(path), off)
        if reason is None:
            exact[h] = index.n - 1
            if write:
                d.write_train(chunk, h)
            return
        if not write:
            return
        d.drop(reason, len(chunk))
        nd["docs_dropped"] += 1
        nd["bytes_dropped"] += len(chunk)
        if reason == "near_dup_eval":
            nd["eval_collisions"] += 1
        rep_src, rep_off, rep_len = index.ranges[rep]
        line = {
            "reason": reason,
            "source": str(path),
            "offset": off,
            "len": len(chunk),
            "rep_source": rep_src,
            "rep_offset": rep_off,
            "rep_len": rep_len,
            "est_jaccard": round(est, 4),
        }
        if len(nd["spot_check"]) < 5:
            rep_raw = pread(Path(rep_src), rep_off, rep_len)
            exact_j = jaccard(shingle_set(chunk), shingle_set(rep_raw))
            line["exact_jaccard"] = round(exact_j, 4)
            nd["spot_check"].append(dict(line))
        log_fh.write(json.dumps(line, sort_keys=True) + "\n")

    i = d.state["cursor"]
    doc_i = d.state.get("doc_i", 0)
    src_read = d.state.get("source_bytes", 0)
    if resumed and index is not None:
        log(f"  {d.name}: replaying the written prefix to rebuild the near-dup index")
        for k in range(i):
            p = order[k]
            if p.name == eval_source:
                continue
            raw = p.read_bytes()
            for off, chunk in iter_chunks(raw):
                apply(p, chunk, off, write=False)
        if doc_i:
            raw = order[i].read_bytes()
            for off, chunk in list(iter_chunks(raw))[:doc_i]:
                apply(order[i], chunk, off, write=False)

    while i < len(order) and d.state["train_bytes"] < d.target:
        path = order[i]
        if path.name == eval_source:
            i += 1
            doc_i = 0
            d.state["cursor"] = i
            d.state["doc_i"] = 0
            continue
        size = path.stat().st_size
        if d.max_source_bytes and src_read and src_read + size > d.max_source_bytes:
            break
        raw = path.read_bytes()
        src_read += len(raw)
        d.state["source_bytes"] = src_read
        ok, reason = filt(raw)
        if not ok:
            d.drop(reason, len(raw))
            i += 1
            doc_i = 0
            d.state["cursor"] = i
            d.state["doc_i"] = 0
            continue
        chunks = list(iter_chunks(raw))
        j = doc_i
        while j < len(chunks):
            off, chunk = chunks[j]
            j += 1
            apply(path, chunk, off, write=True)
            d.state["cursor"] = i
            d.state["doc_i"] = j
            if d.state["train_bytes"] >= d.target:
                break
        if d.state["train_bytes"] >= d.target:
            break
        i += 1
        doc_i = 0
        d.state["cursor"] = i
        d.state["doc_i"] = 0
    if log_fh is not None:
        log_fh.close()
    if nd is not None:
        nd["clusters"] = index.cluster_count()
        nd["peak_signature_bytes"] = index.signature_nbytes()
        nd["docs_kept"] = nd["docs_in"] - nd["docs_dropped"]
        nd["duplicate_fraction"] = nd["docs_dropped"] / nd["docs_in"] if nd["docs_in"] else 0.0
        nd["bytes_fraction"] = nd["bytes_dropped"] / nd["bytes_in"] if nd["bytes_in"] else 0.0
    if qs is not None:
        qs["mean_score"] = qs["score_sum"] / qs["score_n"] if qs["score_n"] else 0.0
        qs["score_docs"] = qs["score_n"]
        qs["pass_rate"] = 1.0 - qs["docs_dropped"] / qs["docs_in"] if qs["docs_in"] else 0.0
        # nd.bytes_in counts every chunk; use it when near-dup is on so the
        # denominator cannot lag a resumed state that predates a counter.
        denom = nd["bytes_in"] if nd is not None else qs["bytes_in"]
        qs["bytes_kept"] = denom - qs["bytes_dropped"]
        qs["byte_pass_rate"] = 1.0 - qs["bytes_dropped"] / denom if denom else 0.0


def build_math(d: Domain) -> None:
    files = sorted((MOUNT / "math").glob("*.txt"))
    rng = random.Random(SEED ^ 0x4D415448)  # "MATH"
    order = list(files)
    rng.shuffle(order)

    if d.state["phase"] == "eval":
        seen: set[str] = set()
        per_window = d.eval_target // 2
        for frac in (0.25, 0.75):
            start = max(0, min(len(order) - 1, int(len(order) * frac)))
            got = 0
            for idx in range(start, len(order)):
                if got >= per_window:
                    break
                raw = order[idx].read_bytes()
                for doc in split_math_docs(raw):
                    ok, _ = filter_math_doc(doc)
                    if not ok:
                        continue
                    h = sha1(doc)
                    if h in seen:
                        continue
                    seen.add(h)
                    out = doc + b"\n\n"
                    d.write_eval(out, h)
                    got += len(out)
                    if got >= per_window:
                        break
            log(f"  math eval window at {frac:.0%}: {got} B")
            if got == 0:
                raise RuntimeError("math: no document passed the filter for the eval carve")
        d.state["phase"] = "train"
        d.state["cursor"] = 0
        d.state["doc_i"] = 0
        d.save_state()

    eval_set = set(d.state.get("eval_sha1", []))
    seen = set()
    if d.state["cursor"] or d.state.get("doc_i"):
        # Rebuild the dedup set over the already-consumed prefix after a resume.
        for idx in range(d.state["cursor"]):
            for doc in split_math_docs(order[idx].read_bytes()):
                ok, _ = filter_math_doc(doc)
                if ok:
                    seen.add(sha1(doc))
        if d.state.get("doc_i"):
            docs = split_math_docs(order[d.state["cursor"]].read_bytes())
            for doc in docs[: d.state["doc_i"]]:
                ok, _ = filter_math_doc(doc)
                if ok:
                    seen.add(sha1(doc))

    i = d.state["cursor"]
    doc_i = d.state.get("doc_i", 0)
    while i < len(order) and d.state["train_bytes"] < d.target:
        docs = split_math_docs(order[i].read_bytes())
        j = doc_i
        while j < len(docs):
            doc = docs[j]
            j += 1
            ok, reason = filter_math_doc(doc)
            if not ok:
                d.drop(reason, len(doc))
                continue
            h = sha1(doc)
            if h in seen:
                d.drop("duplicate", len(doc))
                continue
            seen.add(h)
            if h in eval_set:
                continue
            d.write_train(doc + b"\n\n", h)
            d.state["cursor"] = i
            d.state["doc_i"] = j
            if d.state["train_bytes"] >= d.target:
                break
        if j < len(docs) and d.state["train_bytes"] >= d.target:
            d.state["cursor"] = i
            d.state["doc_i"] = j
            break
        d.state["cursor"] = i + 1
        d.state["doc_i"] = 0
        i += 1


def build_code(d: Domain) -> None:
    files = ensure_code_files()
    total = parquet_rows(files)
    stride = max(1, total // 1500)
    cols = ["repo_name", "path", "language", "license", "code"]

    if d.state["phase"] == "eval":
        seen: set[str] = set()
        for idx, row in iter_parquet_rows(files, cols):
            if idx % stride:
                continue
            out, reason, _ = filter_code(row)
            if out is None:
                continue
            h = sha1(out)
            if h in seen:
                continue
            seen.add(h)
            d.write_eval(out + b"\n\n", h)
            if d.state["eval_bytes"] >= d.eval_target:
                break
        log(f"  code eval: {d.state['eval_bytes']} B from {d.state['eval_units']} files")
        if d.state["eval_bytes"] == 0:
            raise RuntimeError("code: no file passed the filter for the eval carve")
        d.state["phase"] = "train"
        d.state["cursor"] = 0
        d.save_state()

    eval_set = set(d.state.get("eval_sha1", []))
    quotas = dict(LANG_QUOTA)
    seen: set[str] = set()
    cursor = d.state["cursor"]
    if cursor:
        for idx, row in iter_parquet_rows(files, cols):
            if idx >= cursor:
                break
            out, _, _ = filter_code(row)
            if out is not None:
                seen.add(sha1(out))

    def consume(fill: bool) -> None:
        nonlocal cursor
        for idx, row in iter_parquet_rows(files, cols):
            if idx < cursor:
                continue
            out, reason, lang = filter_code(row)
            if out is None:
                d.drop(reason, len(row.get("code") or ""))
                continue
            h = sha1(out)
            if h in seen:
                d.drop("duplicate", len(out))
                continue
            seen.add(h)
            if h in eval_set:
                continue
            if not fill and quotas.get(lang, 0) < len(out):
                d.drop("quota", len(out))
                continue
            d.write_train(out + b"\n\n", h)
            quotas[lang] = quotas.get(lang, 0) - len(out)
            cursor = idx + 1
            d.state["cursor"] = cursor
            if d.state["train_bytes"] >= d.target:
                return

    consume(fill=False)
    if d.state["train_bytes"] < d.target:
        log(f"  code: filling {d.target - d.state['train_bytes']} B over quota")
        written = set(d.state["train_sha1"])
        for _idx, row in iter_parquet_rows(files, cols):
            out, reason, _lang = filter_code(row)
            if out is None:
                continue
            h = sha1(out)
            if h in written or h in eval_set:
                continue
            d.write_train(out + b"\n\n", h)
            written.add(h)
            if d.state["train_bytes"] >= d.target:
                return


def build_parquet_docs(
    d: Domain, files: list[Path], columns: list[str], render, *, stride_div: int = 1200
) -> None:
    total = parquet_rows(files)
    stride = max(1, total // stride_div)

    if d.state["phase"] == "eval":
        seen: set[str] = set()
        for idx, row in iter_parquet_rows(files, columns):
            if idx % stride:
                continue
            out, reason = render(row)
            if out is None:
                d.drop(reason)
                continue
            h = sha1(out)
            if h in seen:
                d.drop("duplicate", len(out))
                continue
            seen.add(h)
            d.write_eval(out, h)
            if d.state["eval_bytes"] >= d.eval_target:
                break
        log(f"  {d.name} eval: {d.state['eval_bytes']} B from {d.state['eval_units']} docs")
        if d.state["eval_bytes"] == 0:
            raise RuntimeError(f"{d.name}: no document passed the filter for the eval carve")
        d.state["phase"] = "train"
        d.state["cursor"] = 0
        d.save_state()

    eval_set = set(d.state.get("eval_sha1", []))
    seen = set()
    for idx, row in iter_parquet_rows(files, columns):
        if idx < d.state["cursor"]:
            continue
        out, reason = render(row)
        if out is None:
            d.drop(reason)
            continue
        h = sha1(out)
        if h in seen:
            d.drop("duplicate", len(out))
            continue
        seen.add(h)
        if h in eval_set:
            continue
        d.write_train(out, h)
        d.state["cursor"] = idx + 1
        if d.state["train_bytes"] >= d.target:
            break


def build_wiki_ru(d: Domain) -> None:
    build_parquet_docs(d, [ensure_ruwiki()], ["title", "text"], filter_wiki_ru)


def build_books(d: Domain) -> None:
    build_parquet_docs(
        d,
        PARQUET_SOURCES["books"],
        ["text", "language", "int_score", "token_count"],
        filter_books,
    )


def build_qa(d: Domain) -> None:
    import pyarrow.parquet as pq

    good, bad = [], []
    for path in PARQUET_SOURCES["qa"]:
        try:
            _ = pq.ParquetFile(path).metadata.num_rows
            good.append(path)
        except Exception:  # noqa: BLE001 — a corrupt local shard is named, not read
            bad.append(path.name)
    if bad:
        d.state["corrupted_files"] = bad
        log(f"  qa: skipping {len(bad)} corrupt file(s): {bad}")
    build_parquet_docs(
        d,
        good,
        ["question", "annotations", "document"],
        render_qa,
        stride_div=300,
    )


def build_instruct(d: Domain) -> None:
    records: list[tuple[bytes, str]] = []
    seen: set[str] = set()
    drops = {"empty": 0, "sentinel": 0, "duplicate": 0}
    for path in INSTRUCT_FILES:
        with path.open("rb") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    drops["empty"] += 1
                    continue
                q = row.get("instruction") or ""
                a = row.get("response") or ""
                if not q or not a:
                    drops["empty"] += 1
                    continue
                try:
                    rec = render_pair(q, a)
                except AssertionError:
                    drops["sentinel"] += 1
                    continue
                h = sha1(rec)
                if h in seen:
                    drops["duplicate"] += 1
                    continue
                seen.add(h)
                records.append((rec, h))
    for reason, count in drops.items():
        if count:
            d.state["drops"][reason] = {"units": count, "bytes": 0}
    rng = random.Random(SEED)
    rng.shuffle(records)
    log(f"  instruct: {len(records)} clean pairs")

    if d.state["phase"] == "eval":
        for rec, h in records:
            d.write_eval(rec, h)
            if d.state["eval_bytes"] >= d.eval_target:
                break
        d.state["phase"] = "train"
        d.state["cursor"] = 0
        d.save_state()

    eval_set = set(d.state.get("eval_sha1", []))
    i = d.state["cursor"]
    while i < len(records) and d.state["train_bytes"] < d.target:
        rec, h = records[i]
        i += 1
        if h in eval_set:
            continue
        d.write_train(rec, h)
        d.state["cursor"] = i


BUILDERS = {
    "web": build_shards,
    "wiki_en": build_shards,
    "wiki_ru": build_wiki_ru,
    "ruweb": build_shards,
    "code": build_code,
    "math": build_math,
    "books": build_books,
    "qa": build_qa,
    "reasoning": build_shards,
    "instruct": build_instruct,
    "rudialog": build_shards,
}


# --- the run ----------------------------------------------------------------


def ensure_downloads(names: tuple[str, ...]) -> None:
    if "code" in names:
        ensure_code_files()
    if "wiki_ru" in names:
        ensure_ruwiki()


def run(
    names: tuple[str, ...],
    *,
    out_dir: Path = MIX,
    force: bool = False,
    near_dup: bool = False,
    quality_band: bool = False,
    max_source_bytes: int = 0,
) -> dict:
    unsupported = sorted(set(names) - set(CRAWL_DOMAINS))
    if (near_dup or quality_band) and unsupported:
        raise SystemExit(
            f"--near-dup/--quality-band are implemented for "
            f"{', '.join(CRAWL_DOMAINS)} only; refusing to silently ignore them "
            f"for {unsupported}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("schema_version", 1)
    manifest["seed"] = SEED
    manifest["filter_version"] = FILTER_VERSION
    manifest.setdefault("domains", {})
    if near_dup or quality_band or max_source_bytes:
        manifest["schema_version"] = 2
        manifest["pipeline"] = {
            "near_dup": near_dup,
            "quality_band": quality_band,
            "max_source_bytes": max_source_bytes,
            "chunk_bytes": CHUNK_BYTES,
        }
    ensure_downloads(names)

    for name in names:
        target = TARGETS[name]
        eval_target = (2 << 20) if name == "instruct" else EVAL_TARGET
        if force:
            for p in (
                out_dir / f"{name}.bin",
                out_dir / f"{name}_eval.bin",
                out_dir / f".{name}.state.json",
                out_dir / f".{name}.train.part",
                out_dir / f".{name}.eval.part",
                out_dir / f".{name}.near_dup.jsonl",
            ):
                p.unlink(missing_ok=True)
        d = Domain(
            name,
            target,
            out_dir,
            eval_target=eval_target,
            near_dup=near_dup,
            quality_band=quality_band,
            max_source_bytes=max_source_bytes,
        )
        if not d.start():
            log(f"{name}: done ({d.saved_entry()['train_bytes']} B) — skipped")
            manifest["domains"][name] = d.saved_entry()
            continue
        log(f"{name}: building {target} B")
        t0 = time.time()
        builder = (
            build_shards_quality
            if (near_dup or quality_band) and name in CRAWL_DOMAINS
            else BUILDERS[name]
        )
        builder(d)
        entry = d.finish()
        entry["filters"] = FILTER_VERSION
        entry["source"] = source_fingerprint(name)
        if d.state.get("corrupted_files"):
            entry["corrupted_files"] = d.state["corrupted_files"]
        manifest["domains"][name] = entry
        manifest["total_train_bytes"] = sum(e["train_bytes"] for e in manifest["domains"].values())
        manifest["total_eval_bytes"] = sum(e["eval_bytes"] for e in manifest["domains"].values())
        manifest["total_bytes"] = manifest["total_train_bytes"] + manifest["total_eval_bytes"]
        atomic_json(manifest_path, manifest)
        log(
            f"{name}: {entry['train_bytes']} B train + {entry['eval_bytes']} B eval "
            f"in {time.time() - t0:.0f}s"
        )

    total = manifest.get("total_bytes", 0)
    if not 5_500_000_000 <= total <= 6_000_000_000:
        log(f"WARNING: total {total} B is outside the 5.5-6.0 GB budget")
    log(f"manifest: {manifest_path} ({total} B)")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=DOMAIN_ORDER, default=None)
    ap.add_argument("--out-dir", type=Path, default=MIX)
    ap.add_argument("--force", action="store_true", help="rebuild the named domains")
    ap.add_argument(
        "--near-dup",
        action="store_true",
        help=f"MinHash-LSH near-duplicate removal ({', '.join(CRAWL_DOMAINS)} only; default off)",
    )
    ap.add_argument(
        "--quality-band",
        action="store_true",
        help=f"cheap heuristic quality band ({', '.join(CRAWL_DOMAINS)} only; default off)",
    )
    ap.add_argument(
        "--max-source-bytes",
        type=int,
        default=0,
        help="cap source bytes read per domain (0 = until the byte target); "
        "the cap is recorded in the manifest",
    )
    args = ap.parse_args(argv)
    names = tuple(args.only) if args.only else DOMAIN_ORDER
    run(
        names,
        out_dir=args.out_dir,
        force=args.force,
        near_dup=args.near_dup,
        quality_band=args.quality_band,
        max_source_bytes=args.max_source_bytes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
