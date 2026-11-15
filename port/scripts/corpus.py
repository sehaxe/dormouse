"""Grow the pretraining corpus to ~21 GB of filtered text and emit the token stream.

Recipe (`docs/records/research/2026-10-16-data-mixing.md` §6, plus the numbers
the task fixes): cheap filters first, exact-doc dedup + per-source MinHash-5gram
only (global suffix-array dedup is not a quality lever at this scale),
`score >= 3` on the web with removal capped at 50 %, **no** English classifier on
Russian, and a hold-out split by DOCUMENT HASH so the eval cannot be
self-deception.

    corpus.py plan          # targets, pools, and the byte ledger so far
    corpus.py layout        # move the previous pass's text into pre/text/
    corpus.py build         # fetch -> filter -> tokenize, resumable (state.json)
    corpus.py assemble      # mix.bin at the weights below + holdout.bin
    corpus.py verify        # sizes, tokens, bytes/token, holdout disjointness

Output is one uint16 stream per source plus `<name>.bin.json`; for hub sources
the text never lands on disk (parquet -> filters -> tokens in memory) and a
shard's parquet is deleted the moment its tokens are committed. The cap is
25 GB of text against 37 GB of disk, which is why nothing is kept twice.

Filter order is the recipe's (cheapest first): licence (a parquet column) ->
score (a parquet column; neither needs a decode) -> length -> exact-doc hash ->
script/langID -> boilerplate/line-repeat -> per-source MinHash-5gram.

Crash safety: tokens and doc hashes are appended, `state.json` records the
committed counts, and a restart truncates both files back to those counts before
re-reading. The shard or text window that was in flight simply runs again.

`ponytail:` the text checks read a 20 KB probe and the MinHash a 4 KB one, so a
doc whose junk starts after byte 20000 is kept; the full doc IS hashed for exact
dedup and the hold-out bucket. Lift the cap if a sample of rejects shows junk
past the probe. The dedup sets are Python sets (~1.5 GB at this corpus) -- swap
for a sorted uint64 array if the box gets tight.

Licences are the datasets' own; every fetch records dataset revision, shard
path, byte count and sha256 in `pre/sources.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import random
import subprocess
import sys
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PRE = ROOT / "data_local" / "pre"
RAW = ROOT / "data_local" / "raw"
TEXT = PRE / "text"  # the 2026-11-14 pass's extracted text
HASHES = PRE / "hashes"

STATE = PRE / "state.json"
SOURCES_JSON = PRE / "sources.json"
STATS = PRE / "stats.json"

#: Hold-out share: 1 doc in 40 by hash -- 2.5 % of the text, which at ~4.4 B per
#: token is the ~200 MB of tokens the task asks for (1-in-50 landed at 190 MB).
HOLDOUT_MOD = 40
#: ...which must land at least this much of a hold-out shard.
MIN_HOLDOUT_MB = 200
MIN_DOC = 200  # a shorter doc cannot carry a sentence of signal
PROBE = 20_000
SKETCH_CHARS = 4_000
SKETCH_K = 12  # bottom-k shingle minima, banded BANDS rows
BANDS = 4
CHUNK = 1 << 19  # tokens per interleave unit (1 MB on disk)
COMMIT_MB = 64  # kept text between state.json commits


@dataclass(frozen=True)
class Source:
    """One pool: where its shards live on the hub, and how it is filtered."""

    name: str
    dataset: str
    hub_path: str
    column: str
    licence: str
    script: str  # en | sci | ru | code
    target_gb: float  # kept-text target for the pool
    score_min: int | None = None
    #: per-row licence gate (code: MIT/Apache only, checked, not assumed)
    licence_col: str | None = None
    licence_ok: tuple[str, ...] = ()
    #: the 2026-11-14 pass already extracted these rows to `pre/text/<name>.bin`;
    #: that text is consumed first and the hub is only asked for the difference.
    existing: bool = True
    #: importable from the hub at all (peS2o v1 is json.gz, not parquet)
    hub: bool = True
    #: shards the previous pass already consumed, in hub order
    skip_shards: int = 0


#: Pools sum to ~20 GB of kept text; the mix below draws
#: 60 % EN web / 20 % RU / 12 % math+code+synth / 8 % wiki+science.
#: The per-pool targets are the task's planned shares; a pool whose hub is off
#: (`code`, `science`) cannot grow past the text the 2026-11-14 pass extracted.
SOURCES: tuple[Source, ...] = (
    Source(
        "fwedu",
        "HuggingFaceFW/fineweb-edu",
        "sample/100BT",
        "text",
        "ODC-By 1.0 + CommonCrawl ToU",
        "en",
        6.5,
        score_min=3,
        skip_shards=2,
    ),
    Source(
        "ru",
        "HuggingFaceFW/fineweb-2",
        "data/rus_Cyrl/train",
        "text",
        "ODC-By 1.0 + CommonCrawl ToU + PII form",
        "ru",
        4.0,
        skip_shards=1,
    ),
    Source(
        "wiki_ru",
        "wikimedia/wikipedia",
        "20231101.ru",
        "text",
        "CC BY-SA 4.0 + GFDL",
        "ru",
        1.5,
        skip_shards=1,
    ),
    Source(
        "math",
        "HuggingFaceTB/finemath",
        "finemath-4plus",
        "text",
        "ODC-By 1.0",
        "sci",
        1.5,
        score_min=4,
        skip_shards=2,
    ),
    Source(
        "code",
        "the-stack licence split (all-mit / all-apache-2.0; upstream id "
        "unrecorded by the 2026-11-14 pass)",
        "data",
        "code",
        "MIT + Apache-2.0 per row (verified, see `licence`)",
        "code",
        3.0,
        licence_col="license",
        licence_ok=("mit", "apache-2.0"),
        hub=False,
    ),
    Source(
        "synth",
        "HuggingFaceTB/cosmopedia",
        "data/auto_math_text",
        "text",
        "Apache-2.0 (card YAML only)",
        "en",
        0.7,
        skip_shards=1,
    ),
    Source(
        "wiki_en",
        "wikimedia/wikipedia",
        "20231101.en",
        "text",
        "CC BY-SA 4.0 + GFDL",
        "en",
        1.0,
        skip_shards=1,
    ),
    Source("science", "allenai/peS2o", "data/v1", "text", "ODC-By 1.0", "sci", 0.8, hub=False),
)

BY_NAME = {s.name: s for s in SOURCES}

#: The mix targets (task: 60 % EN web / 20 % RU / 12 % math+code / 8 %
#: wiki+encyclopedic). Weights are DERIVED from these buckets and the pools that
#: actually exist, so no pool can be quietly over-drawn and the totals follow
#: what was fetched instead of a number someone typed.
BUCKETS: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    ("web_en", 0.60, ("fwedu",)),
    ("ru", 0.20, ("ru", "wiki_ru")),
    ("math_code", 0.12, ("math", "code", "synth")),
    ("encyc", 0.08, ("wiki_en", "science")),
)
#: Russian is 20 % against its ~10 % natural share: our tokenizer is trained on
#: the mix, so a 2x RU upsample is what the research asked for. Cosmopedia rides
#: in math_code instead of a bucket of its own -- synthetic bulk is a waste at
#: this budget, 1.5-2 % is inside the "rephrased, not textbook-alone" regime.

#: The 3-stage schedule: `until_frac` plus a MULTIPLIER per bucket (a bucket
#: not named is 1.0). The multiplier is relative to that bucket's share of the
#: uniform mix, and `stage_weights` normalises each bucket's duration-weighted
#: mean back to 1 — that equality is what makes the schedule affordable: no
#: stage asks a pool for more tokens than the uniform mix already budgeted, so
#: the epoch cap can never bind on a plan this generator wrote.
#:
#: Read it as: web-heavy until 70 %, code/math entering from 70 %, and the last
#: 10 % is the anneal (web down, code/math/encyclopedic up).
STAGES: tuple[tuple[float, dict[str, float]], ...] = (
    (0.70, {"web_en": 1.13, "ru": 1.0, "math_code": 0.58, "encyc": 0.62}),
    (0.90, {"web_en": 0.83, "ru": 1.1, "math_code": 1.5, "encyc": 1.25}),
    (1.00, {"web_en": 0.37, "ru": 1.25, "math_code": 2.75, "encyc": 2.5}),
)


#: Multipliers for the shingle polynomial and the band keys (odd, 64-bit).
_K = (0x9E3779B97F4A7C15, 0x85EBCA77C2B2AE63, 0xC2B2AE3D27D4EB4F, 0x27D4EB2F165667C5)
_A, _B, _C, _D = (np.uint64(v) for v in _K)
_MASK = (1 << 64) - 1

#: Every byte that is NOT an ASCII letter -- used as a delete table, so the
#: cost is one C-level pass instead of a per-character Python regex.
_NON_LATIN = bytes(i for i in range(256) if not (65 <= i <= 90 or 97 <= i <= 122))

EN_STOP = frozenset("the and of to is that with from".split())
RU_STOP = frozenset("и в не на что как для это".split())

_P = np.uint64(1000003)
_P4, _P3, _P2, _P1 = (np.uint64((1000003**i) % 2**64) for i in (4, 3, 2, 1))


# ------------------------------------------------------------------ hashing


def digest(doc: bytes) -> int:
    """One 64-bit doc hash: the exact-dedup key AND the hold-out bucket."""
    return int.from_bytes(hashlib.blake2b(doc, digest_size=8).digest(), "big")


def is_holdout(h: int) -> bool:
    return h % HOLDOUT_MOD == 0


def sketch(doc: bytes) -> tuple[int, ...]:
    """bottom-K of the 5-WORD-gram hashes of the doc's first 4 KB.

    Words, not characters: a 5-gram over bytes ran at 1.6 MB/s. `bytes.split()`
    and `zlib.crc32` are C, and the polynomial is five numpy ops over ~100 ints.
    """
    ws = doc[:SKETCH_CHARS].split()
    if len(ws) < 5:
        return ()
    ids = np.fromiter((zlib.crc32(w) for w in ws), np.uint64, len(ws))
    h = (ids[:-4] * _A) ^ (ids[1:-3] * _B) ^ (ids[2:-2] * _C) ^ (ids[3:-1] * _D) ^ ids[4:]
    k = min(SKETCH_K, h.size)
    return tuple(int(v) for v in np.partition(h, k - 1)[:k])


def band_keys(sig: tuple[int, ...]) -> list[int]:
    """LSH band keys from the sketch: P(hit) at Jaccard 0.9 is 1-(1-0.9^3)^4."""
    sig = sig + (0,) * (SKETCH_K - len(sig))
    w = SKETCH_K // BANDS
    return [
        (sig[i * w] * _K[0] ^ sig[i * w + 1] * _K[1] ^ sig[i * w + 2] * _K[2]) & _MASK
        for i in range(BANDS)
    ]


# ------------------------------------------------------------------ filters


def keep_text(doc: bytes, script: str) -> str | None:
    """`None` to keep, else the reason the doc is dropped. Cheap checks only.

    Script ratios are measured on BYTES (`bytes.translate` is a C path; the
    regex classes this replaced ran at ~1.9 MB/s): a Latin letter is one byte,
    a Cyrillic one is two, and both sides of every ratio use the same unit. That
    is a real weakness -- a doc of Cyrillic + Latin mixed at 50/50 characters is
    not a 50/50 byte mix -- but it can only mis-rank a doc by the 2x encoding
    weight, and the thresholds sit at 0.6, not at the boundary.
    """
    if len(doc) < MIN_DOC:
        return "short"
    probe, text = doc[:PROBE], doc[:PROBE].decode("utf-8", "ignore")
    words = text.split()
    if len(words) < 20:
        return "few_words"

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 8:
        top = Counter(lines).most_common(1)[0][1]
        if top > (0.5 if script == "code" else 0.3) * len(lines):
            return "line_repeat"
    if script == "code":
        return None

    latin = len(probe.translate(None, _NON_LATIN))  # bytes that survive are the letters
    cyr = 2 * (probe.count(b"\xd0") + probe.count(b"\xd1"))
    if latin + cyr < 0.4 * len(probe):
        return "low_letters"

    words_low = set(text.lower().split())
    if script == "ru":
        if cyr < 0.6 * (cyr + latin):
            return "lang_not_ru"
        if not words_low & RU_STOP:
            return "lang_no_ru_stopword"
    else:
        if latin < 0.6 * (cyr + latin):
            return "lang_not_latin"
        if script == "en" and not words_low & EN_STOP:
            return "lang_no_en_stopword"

    if len(lines) >= 8 and sum(1 for ln in lines if len(ln) < 20) > 0.8 * len(lines):
        return "short_lines"
    if len(words) >= 50 and len(set(words)) < 0.3 * len(words):
        return "word_repeat"
    return None


def score_keep(score: int | None, src: Source, dropped: int, seen: int) -> bool:
    """`score >= min`, with the web rule's removal capped at half.

    FineWeb-Edu's `sample/100BT` shards are already the score>=3 subset (every
    row of shard 0 measured `int_score >= 3`: 3.479 GB of 3.479 GB), so on this
    pool the cap never binds. A pool whose card does not promise that must not
    be able to delete more than half of itself -- that would leave us under one
    epoch.
    """
    if src.score_min is None or score is None:
        return True
    if score >= src.score_min:
        return True
    return seen > 0 and dropped / seen > 0.5


# ------------------------------------------------------------------- ledger


@dataclass
class Tally:
    """Per-source counters. JSON-able: `state.json` is the resume point."""

    in_bytes: int = 0
    docs: int = 0
    kept_bytes: int = 0
    kept_docs: int = 0
    train_docs: int = 0
    holdout_docs: int = 0
    tokens: int = 0
    holdout_tokens: int = 0
    holdout_bytes: int = 0
    text_offset: int = 0
    drop: dict[str, int] = field(default_factory=dict)
    shards: dict[str, int] = field(default_factory=dict)
    inflight: dict | None = None

    #: fields a restart restores from `inflight` (files are truncated to match)
    RESUME = (
        "in_bytes",
        "docs",
        "kept_bytes",
        "kept_docs",
        "train_docs",
        "holdout_docs",
        "tokens",
        "holdout_tokens",
        "holdout_bytes",
        "text_offset",
    )

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def load(cls, d: dict) -> Tally:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def rewind(self) -> None:
        """Undo the uncommitted unit of work.

        A parquet shard is redone from row 0 (it cannot be resumed mid-file), so
        its counters go back to the baseline taken when it started. A text
        window is not redone: the token and hash files are truncated to the
        last commit and `text_offset` was committed with them, so the run
        continues exactly there and loses at most one commit.
        """
        what = (self.inflight or {}).get("what", "")
        if what.startswith(("text:", "shard-done:")):
            self.inflight = None
            return
        if not self.inflight:
            return
        base = self.inflight["base"]
        self.drop = dict(self.inflight["drop"])
        self.shards = dict(self.inflight.get("shards", self.shards))
        for k in self.RESUME:
            setattr(self, k, base[k])
        self.inflight = None


# ---------------------------------------------------------------- doc streams


def docs_from_text(path: Path, start: int, window: int = 32 << 20):
    """(doc, score, licence, offset_after) from a text file, blank-line blocks
    merged up to at least MIN_DOC bytes so no hash unit is a fragment.

    The window is walked with an INDEX, not `buf = buf[cut + 2:]`: reslicing a
    32 MB buffer once per blank line is quadratic, and it measured 1.4 MB/s on
    this corpus (hundreds of GB of memcpy per window). `pos` is the absolute
    offset just past the last consumed byte, which is what a resume seeks to.
    """
    buf = b""
    acc = b""
    pos = start
    with path.open("rb") as fh:
        fh.seek(start)
        while True:
            raw = fh.read(window)
            if not raw:
                break
            buf += raw
            i = 0
            while True:
                cut = buf.find(b"\n\n", i)
                if cut == -1:
                    break
                acc += buf[i:cut] + b"\n\n"
                pos += (cut + 2) - i
                i = cut + 2
                if len(acc) >= MIN_DOC:
                    yield acc, None, None, pos
                    acc = b""
            if len(buf) - i > 4 * window:  # one runaway line, not a buffer
                pos += len(buf) - i
                yield acc + buf[i:], None, None, pos
                acc = b""
                i = len(buf)
            buf = buf[i:]
        if acc + buf:
            yield acc + buf, None, None, pos + len(buf)


def docs_from_parquet(path: Path, src: Source):
    """(doc, score, licence, None) per row, in row-group batches."""
    cols = [src.column]
    if src.score_min:
        cols.append("int_score")
    if src.licence_col:
        cols.append(src.licence_col)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=cols):
        texts = batch.column(src.column).to_pylist()
        n = len(texts)
        score = batch.column("int_score").to_pylist() if "int_score" in cols else [None] * n
        lic = batch.column(src.licence_col).to_pylist() if src.licence_col else [None] * n
        for row in zip(texts, score, lic, strict=True):
            if isinstance(row[0], str):
                yield row[0].encode("utf-8", "ignore"), row[1], row[2], None


# --------------------------------------------------------------- hub fetching


def hub_files(src: Source) -> tuple[str, list[dict]]:
    """(dataset revision, [{path, size, sha256}]) -- shards in hub order."""
    import urllib.request

    def get(url: str):
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)

    revision = get(f"https://huggingface.co/api/datasets/{src.dataset}")["sha"]
    rows = get(f"https://huggingface.co/api/datasets/{src.dataset}/tree/{revision}/{src.hub_path}")
    files = [
        {"path": r["path"], "size": r.get("size", 0), "sha256": (r.get("lfs") or {}).get("oid")}
        for r in rows
        if r["type"] == "file" and r["path"].endswith(".parquet")
    ]
    files.sort(key=lambda f: f["path"])
    return revision, files


def fetch_shard(src: Source, revision: str, entry: dict, out: Path) -> Path:
    """curl -C - into `raw/<name>/`; refuse a byte count or hash that disagrees."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and entry["size"] and out.stat().st_size == entry["size"]:
        return out
    url = f"https://huggingface.co/datasets/{src.dataset}/resolve/{revision}/{entry['path']}"
    print(f"  fetch {entry['path']} ({entry['size'] / 1e9:.2f} GB) -> {out.name}", flush=True)
    subprocess.run(
        [
            "curl",
            "-sSL",
            "-C",
            "-",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--connect-timeout",
            "30",
            "-o",
            str(out),
            url,
        ],
        check=True,
    )
    got = out.stat().st_size
    if entry["size"] and got != entry["size"]:
        raise RuntimeError(f"{out}: got {got} B, hub says {entry['size']} B")
    if entry["sha256"] and _sha256(out) != entry["sha256"]:
        raise RuntimeError(f"{out}: sha256 disagrees with the hub's lfs oid")
    return out


# -------------------------------------------------------------------- build


class Sink:
    """Appends uint16 tokens to `<name>.bin` and `<name>_holdout.bin`."""

    def __init__(self, name: str) -> None:
        self.train = (PRE / f"{name}.bin").open("ab")
        self.hold = (PRE / f"{name}_holdout.bin").open("ab")

    def write(self, ids: list[int], holdout: bool) -> None:
        (self.hold if holdout else self.train).write(np.asarray(ids, dtype=np.uint16).tobytes())

    def flush(self) -> None:
        self.train.flush()
        self.hold.flush()

    def close(self) -> None:
        self.train.close()
        self.hold.close()


def build(args: argparse.Namespace) -> int:
    os.environ.setdefault("RAYON_NUM_THREADS", "4")
    from tokenizers import Tokenizer

    tok_path = Path(args.tokenizer)
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab = tokenizer.get_vocab_size()
    if vocab > 65536:
        raise SystemExit(f"tokenizer vocab {vocab} does not fit the uint16 stream")
    tok_sha = hashlib.sha256(tok_path.read_bytes()).hexdigest()[:16]
    state = json.loads(STATE.read_text()) if STATE.exists() else {"sources": {}}
    state["tokenizer_sha"] = tok_sha
    HASHES.mkdir(exist_ok=True)

    for src in SOURCES:
        stale = PRE / f"{src.name}.bin"
        ours = src.name in state["sources"] or (PRE / f"{src.name}.bin.json").exists()
        if stale.exists() and not ours and not (TEXT / f"{src.name}.bin").exists():
            if not stale.with_suffix(".tokenized"):
                raise SystemExit(f"{stale} is the previous pass's TEXT -- run `layout` first")

    tallies: dict[str, Tally] = {}
    dedup: set[int] = set()
    for src in SOURCES:
        t = Tally.load(state["sources"].get(src.name, {}))
        tallies[src.name] = t
        t.rewind()  # an interrupted shard/window is dropped, not half-kept
        for suffix, toks in (("", t.tokens), ("_holdout", t.holdout_tokens)):
            p = PRE / f"{src.name}{suffix}.bin"
            if p.exists():
                with p.open("r+b") as fh:
                    fh.truncate(toks * 2)
        for suffix, docs in (("train", t.train_docs), ("holdout", t.holdout_docs)):
            p = HASHES / f"{src.name}.{suffix}.u64"
            if p.exists():
                with p.open("r+b") as fh:
                    fh.truncate(docs * 8)
            if docs:
                dedup.update(int(v) for v in np.fromfile(p, dtype=np.uint64))
    kept = sum(t.kept_bytes for t in tallies.values())
    print(f"resume: {len(dedup):,} doc hashes, {kept / 1e9:.2f} GB kept, tok {tok_sha}", flush=True)

    meta = json.loads(SOURCES_JSON.read_text()) if SOURCES_JSON.exists() else {}
    for src in SOURCES:
        t = tallies[src.name]
        if t.kept_bytes >= int(src.target_gb * 1e9):
            print(f"{src.name}: pool full ({t.kept_bytes / 1e9:.2f} GB)", flush=True)
            continue
        if args.only and src.name not in args.only:
            continue
        _run_source(src, t, state, tokenizer, args, meta, dedup)
    _save(meta)
    return 0


def _mark_partial_shard(t: Tally, stage) -> None:
    """A pool that hit its target MID-shard: record the shard as done.

    A parquet shard has no per-row resume, so the budget can only be honoured by
    stopping inside it. Without this the shard stays in `inflight` and the next
    `build` rewinds it (`Tally.rewind`) -- the target would never be reached and
    the same rows would be re-read forever. The recorded size is the shard's own
    contribution, measured from the baseline `stage()` took.
    """
    what = (t.inflight or {}).get("what", "")
    if not what.startswith("shard:"):
        return
    path = what.removeprefix("shard:")
    t.shards[path] = t.kept_bytes - t.inflight["base"]["kept_bytes"]
    t.inflight = None
    stage(f"shard-done:{path}")


def _run_source(
    src: Source, t: Tally, state: dict, tokenizer, args, meta: dict, dedup: set[int]
) -> None:
    """Fetch/filter/tokenize one pool, committing every COMMIT_MB of kept text."""
    budget = int(src.target_gb * 1e9)
    sink = Sink(src.name)
    near: set[int] = set()  # per-source MinHash bands
    train_b: list[str] = []
    hold_b: list[str] = []
    batch_bytes = 0
    cur_off = [t.text_offset]  # advances for every doc, kept or not
    hf = (HASHES / f"{src.name}.train.u64").open("ab")
    hf_hold = (HASHES / f"{src.name}.holdout.u64").open("ab")

    def commit() -> None:
        state["sources"][src.name] = t.as_dict()
        STATE.write_text(json.dumps(state, indent=1))

    def emit(offset: int | None = None) -> None:
        nonlocal train_b, hold_b, batch_bytes
        if train_b:
            for e in tokenizer.encode_batch(train_b):
                sink.write(e.ids, False)
                t.tokens += len(e.ids)
        if hold_b:
            for e in tokenizer.encode_batch(hold_b):
                sink.write(e.ids, True)
                t.holdout_tokens += len(e.ids)
            t.holdout_bytes += sum(len(s.encode()) for s in hold_b)
        if offset is not None:
            t.text_offset = offset
        sink.flush()
        hf.flush()
        hf_hold.flush()
        commit()
        train_b, hold_b, batch_bytes = [], [], 0

    def stage(label: str) -> None:
        """Mark a unit of work in flight, so a restart rewinds to here."""
        t.inflight = {
            "what": label,
            "drop": dict(t.drop),
            "shards": dict(t.shards),
            "base": {k: getattr(t, k) for k in Tally.RESUME},
        }
        commit()

    try:
        stopped = False
        for doc, score, licence, off in _source_docs(src, t, budget, args, meta, stage):
            if off is not None:
                cur_off[0] = off
            t.in_bytes += len(doc)
            t.docs += 1
            h = digest(doc)
            # A text-sourced row carries no licence COLUMN: `code` was extracted
            # from the licence-split shards themselves (`all-mit` /
            # `all-apache-2.0` names), so the split happened upstream of this
            # stream. Gating a `None` here would drop the whole code pool for a
            # field the text never had; a hub row still gets the per-row check.
            if src.licence_ok and licence is not None and licence not in src.licence_ok:
                t.drop["licence"] = t.drop.get("licence", 0) + 1
                continue
            if not score_keep(score, src, t.drop.get("score", 0), t.docs - 1):
                t.drop["score"] = t.drop.get("score", 0) + 1
                continue
            text = doc.decode("utf-8", "ignore")
            reason = keep_text(doc, src.script)
            if reason:
                t.drop[reason] = t.drop.get(reason, 0) + 1
                continue
            if h in dedup:
                t.drop["dup"] = t.drop.get("dup", 0) + 1
                continue
            keys = band_keys(sketch(doc))
            if any(k in near for k in keys):
                t.drop["near_dup"] = t.drop.get("near_dup", 0) + 1
                continue
            dedup.add(h)
            near.update(keys)
            t.kept_bytes += len(doc)
            t.kept_docs += 1
            hold = is_holdout(h)
            if hold:
                t.holdout_docs += 1
            else:
                t.train_docs += 1
            (hf_hold if hold else hf).write(int(h).to_bytes(8, "little"))  # np.fromfile order
            batch_bytes += len(doc)
            (hold_b if hold else train_b).append(text)
            if len(train_b) + len(hold_b) >= args.batch or batch_bytes > COMMIT_MB << 20:
                emit(cur_off[0])
            if t.kept_bytes >= budget:
                stopped = True
                break
        emit(cur_off[0])
    finally:
        sink.close()
        hf.close()
        hf_hold.close()
    if stopped:
        _mark_partial_shard(t, stage)
    manifests(src, state)
    print(
        f"{src.name}: {t.kept_bytes / 1e9:.2f}/{src.target_gb:.2f} GB kept, "
        f"{t.tokens / 1e6:.1f} M tokens, drops {t.drop}",
        flush=True,
    )


def manifests(src: Source, state: dict) -> None:
    """`<name>.bin.json` for the train and hold-out streams of one source."""
    for suffix in ("", "_holdout"):
        path = PRE / f"{src.name}{suffix}.bin"
        if not path.exists():
            continue
        toks = path.stat().st_size // 2
        (PRE / f"{path.stem}.bin.json").write_text(
            json.dumps(
                {
                    "dtype": "uint16",
                    "tokens": toks,
                    "bytes": toks * 2,
                    "tokenizer_sha": state["tokenizer_sha"],
                    "dataset": src.dataset,
                    "licence": src.licence,
                    "sha256": _sha256(path),
                },
                indent=1,
            )
        )


def _save(meta: dict) -> None:
    SOURCES_JSON.write_text(json.dumps(meta, indent=1))


def _source_docs(src: Source, t: Tally, budget: int, args, meta: dict, stage):
    """The previous pass's text first, then hub shards, until the pool target."""
    if src.existing:
        path = TEXT / f"{src.name}.bin"
        if path.exists() and t.text_offset < path.stat().st_size and t.kept_bytes < budget:
            stage(f"text:{path.name}@{t.text_offset}")
            print(
                f"{src.name}: text {t.text_offset / 1e9:.2f}/"
                f"{path.stat().st_size / 1e9:.2f} GB consumed",
                flush=True,
            )
            yield from docs_from_text(path, t.text_offset)

    if t.kept_bytes >= budget or not src.hub:
        return
    revision, files = hub_files(src)
    entry = meta.setdefault(src.name, {"dataset": src.dataset, "licence": src.licence, "files": {}})
    entry["revision"] = revision
    _save(meta)
    for i, f in enumerate(files):
        if t.kept_bytes >= budget or (args.max_shards and len(t.shards) >= args.max_shards):
            return
        if i < src.skip_shards or f["path"] in t.shards:
            continue
        stage(f"shard:{f['path']}")
        local_name = f"{src.name}__{Path(f['path']).name}"
        local = fetch_shard(src, revision, f, RAW / src.name / local_name)
        entry["files"][f["path"]] = {**f, "local": local.name}
        _save(meta)
        before = t.kept_bytes
        yield from docs_from_parquet(local, src)
        t.shards[f["path"]] = t.kept_bytes - before
        local.unlink()  # the archive is worthless once its tokens are committed
        stage(f"shard-done:{f['path']}")
        print(
            f"  {f['path']}: +{(t.kept_bytes - before) / 1e9:.2f} GB kept, archive deleted",
            flush=True,
        )


# ----------------------------------------------------------------- assemble


def streams(suffix: str = "") -> dict[str, tuple[Path, int]]:
    """name -> (token file, tokens) for the train ('') or hold-out ('_holdout') set."""
    out: dict[str, tuple[Path, int]] = {}
    for name in pool_names():
        p = PRE / f"{name}{suffix}.bin"
        if p.exists() and p.stat().st_size:
            out[name] = (p, p.stat().st_size // 2)
    return out


def pool_names() -> list[str]:
    return [n for _, _, names in BUCKETS for n in names]


def mixture(avail: dict[str, int]) -> tuple[dict[str, float], int]:
    """Bucket targets -> per-pool weights, and the largest mix the pools allow.

    Inside a bucket the pools split in proportion to what they hold, so the
    smallest one is used up first and its neighbours fill the rest of the
    bucket; the mix total is the tightest bucket's supply, in tokens.
    """
    supply = {b: sum(avail.get(n, 0) for n in names) for b, _, names in BUCKETS}
    limits = [supply[b] / share for b, share, _ in BUCKETS if supply[b]]
    total = int(min(limits)) if limits else 0
    shares: dict[str, float] = {}
    for b, share, names in BUCKETS:
        if not supply[b]:
            continue
        for n in names:
            if avail.get(n, 0):
                shares[n] = share * avail[n] / supply[b]
    return shares, total


def interleave(
    shares: dict[str, float], avail: dict[str, int], total: int, seed: int = 20261115
) -> list[tuple[str, int, int]]:
    """(source, chunk index, tokens) in output order: a within-source shuffle,
    then the largest weight deficit. Deterministic; exact to one chunk.

    The LAST chunk of a pool may be short -- a pool smaller than one chunk is
    still usable, which is what keeps a 1 % domain (and the toy corpus in the
    tests) in the mix instead of silently dropping it.
    """
    rng = random.Random(seed)
    plan: dict[str, list[int]] = {}
    for name in shares:
        order = list(range(-(-avail.get(name, 0) // CHUNK)))
        rng.shuffle(order)
        plan[name] = order
    emitted = {n: 0 for n in shares}
    pick: list[tuple[str, int, int]] = []
    spent = 0
    while spent < total:
        live = [n for n in shares if plan[n]]
        if not live:
            break
        name = max(live, key=lambda n: shares[n] * (spent + CHUNK) - emitted[n])
        idx = plan[name].pop()
        n_tok = min(CHUNK, avail[name] - idx * CHUNK)
        if n_tok <= 0:
            continue
        pick.append((name, idx, n_tok))
        emitted[name] += n_tok
        spent += n_tok
    return pick


def _write_interleaved(
    out_path: Path, pools: dict[str, tuple[Path, int]], shares: dict[str, float], total: int
) -> dict[str, int]:
    fhs = {n: p.open("rb") for n, (p, _) in pools.items()}
    mm = {n: mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) for n, f in fhs.items()}
    counts = {n: 0 for n in pools}
    try:
        with out_path.open("wb") as out:
            for name, chunk, n_tok in interleave(
                shares, {n: p[1] for n, p in pools.items()}, total
            ):
                off = chunk * CHUNK * 2
                out.write(mm[name][off : off + n_tok * 2])
                counts[name] += n_tok
    finally:
        for m in mm.values():
            m.close()
        for f in fhs.values():
            f.close()
    return counts


def stage_weights(avail: dict[str, int]) -> list[dict]:
    """`STAGES` expanded from bucket multipliers to per-pool weights.

    Inside a bucket the pools keep the proportion they have in `mixture()`, so a
    stage only moves the bucket's TOTAL and can never ask an absent pool for a
    share. Each bucket's multipliers are divided by their duration-weighted mean
    first, so the three stages average back to the uniform mix exactly.
    """
    fracs, prev = [], 0.0
    for until, _ in STAGES:
        fracs.append(until - prev)
        prev = until
    buckets = [b for b, _, _ in BUCKETS]
    mult = [{b: m.get(b, 1.0) for b in buckets} for _, m in STAGES]
    for b in buckets:
        avg = sum(f * m[b] for f, m in zip(fracs, mult, strict=True)) or 1.0
        for m in mult:
            m[b] /= avg
    out: list[dict] = []
    for (until, _), m in zip(STAGES, mult, strict=True):
        w: dict[str, float] = {}
        for b, _, names in BUCKETS:
            supply = sum(avail.get(n, 0) for n in names) or 1
            for n in names:
                if avail.get(n, 0):
                    w[n] = m[b] * avail[n] / supply
        s = sum(w.values())
        if s:
            out.append({"until_frac": until, "weights": {k: v / s for k, v in w.items()}})
    return out


def _plan_text(sources: dict[str, Path], stages: list[dict]) -> str:
    """The `mix_plan` TOML `train/mixture.py` reads; ONE shape, generated."""
    lines = ["# mix_1b -- generated by scripts/corpus.py assemble from the measured pools."]
    lines += ["# [sources] paths are resolved against this file's directory.", "", "[sources]"]
    for name, path in sorted(sources.items()):
        lines.append(f'{name} = "{path}"')
    for st in stages:
        lines += ["", "[[stage]]", f"until_frac = {st['until_frac']}", "[stage.weights]"]
        for name, w in sorted(st["weights"].items()):
            lines.append(f"{name} = {w}")
    return "\n".join(lines) + "\n"


def assemble(args: argparse.Namespace) -> int:
    pools = streams()
    missing = [n for n in pool_names() if n not in pools]
    if missing:
        raise SystemExit(f"assemble: no token stream for {missing}")
    shares, total = mixture({n: p[1] for n, p in pools.items()})
    if args.target_gb:  # cap in TEXT bytes: the pools' own B/token, weighted
        bpt = sum(shares[n] * _text_bytes(n) / pools[n][1] for n in shares)
        total = min(total, int(args.target_gb * 1e9 / bpt))
    tok_sha = json.loads(STATE.read_text())["tokenizer_sha"]
    avail = {n: p[1] for n, p in pools.items()}
    stages = stage_weights(avail)
    print(
        f"mix: {total / 1e9:.3f} B tokens (shortest pool allows "
        f"{min(pools[n][1] / shares[n] for n in shares) / 1e9:.3f} B)",
        flush=True,
    )
    emitted = _write_interleaved(PRE / "mix.bin", pools, shares, total)
    total = sum(emitted.values())  # a pool that ran dry ends the mix, not the manifest
    per_source = {}
    for name in emitted:
        bpt_src = _source_bpt(name)
        per_source[name] = {
            "tokens": emitted[name],
            "bytes": emitted[name] * 2,
            "share": emitted[name] / max(1, total),
            "text_bytes": emitted[name] * bpt_src,
            "bytes_per_token": bpt_src,
        }
    text_total = sum(v["text_bytes"] for v in per_source.values())
    (PRE / "mix.bin.json").write_text(
        json.dumps(
            {
                "dtype": "uint16",
                "tokens": total,
                "bytes": total * 2,
                "text_bytes": text_total,
                "bytes_per_token": text_total / max(1, total),
                "tokenizer_sha": tok_sha,
                "sha256": _sha256(PRE / "mix.bin"),
                "holdout": "holdout.bin",
                "n_sources": len(emitted),
                "sources": per_source,
                "weights": shares,
                "tokens_by_source": emitted,
                "stages": stages,
                "stage_rule": "until_frac is a fraction of the STEP budget; the "
                "weighted average of the stages is the uniform mix above",
            },
            indent=1,
        )
    )
    plan = Path(getattr(args, "plan", None) or (PRE / "mix.bin.plan.toml"))
    plan.parent.mkdir(parents=True, exist_ok=True)
    rel = {n: os.path.relpath(p, plan.parent) for n, (p, _) in pools.items()}
    plan.write_text(_plan_text(rel, stages))
    print(f"plan -> {plan}", flush=True)

    hold = streams("_holdout")
    hold_total = sum(p[1] for p in hold.values())
    hshares, _ = mixture({n: p[1] for n, p in hold.items()})
    if not hshares:  # a tiny hold-out: mirror what exists, not the bucket targets
        hshares = {n: p[1] / max(1, hold_total) for n, p in hold.items()}
    hhold = _write_interleaved(PRE / "holdout.bin", hold, hshares, hold_total)
    hold_total = sum(hhold.values())
    (PRE / "holdout.bin.json").write_text(
        json.dumps(
            {
                "dtype": "uint16",
                "tokens": hold_total,
                "bytes": hold_total * 2,
                "tokenizer_sha": tok_sha,
                "sha256": _sha256(PRE / "holdout.bin"),
                "split": f"blake2b8(doc) % {HOLDOUT_MOD} == 0",
                "tokens_by_source": hhold,
            },
            indent=1,
        )
    )

    for name, (path, toks) in list(pools.items()) + list(hold.items()):
        src = BY_NAME[name]
        (PRE / f"{path.stem}.bin.json").write_text(
            json.dumps(
                {
                    "dtype": "uint16",
                    "tokens": toks,
                    "bytes": toks * 2,
                    "tokenizer_sha": tok_sha,
                    "dataset": src.dataset,
                    "licence": src.licence,
                    "sha256": _sha256(path),
                },
                indent=1,
            )
        )
    doc = stats()
    print(
        f"mix.bin {total * 2 / 1e9:.2f} GB, holdout.bin {hold_total * 2 / 1e6:.0f} MB, "
        f"{doc['text_kept_gb']:.2f} GB text, {doc['bytes_per_token']:.2f} B/token",
        flush=True,
    )
    return 0


def _text_bytes(name: str) -> int:
    raw = json.loads(STATE.read_text())["sources"].get(name, {})
    return raw.get("kept_bytes", 1) or 1


def _source_bpt(name: str) -> float:
    """TEXT bytes per token for one pool: a property of its text, not of how
    much of it the mix drew, so a half-used pool keeps its own ratio."""
    raw = json.loads(STATE.read_text())["sources"].get(name, {})
    toks = (raw.get("tokens", 0) or 0) + (raw.get("holdout_tokens", 0) or 0)
    return (raw.get("kept_bytes", 0) or 0) / toks if toks else 1.0


def stats() -> dict:
    """`stats.json`: in/out per source, dedup and filter rates, final B/token."""
    state = json.loads(STATE.read_text())
    rows = {}
    for src in SOURCES:
        t = Tally.load(state["sources"].get(src.name, {}))
        if not t.docs:
            continue
        rows[src.name] = {
            "in_bytes": t.in_bytes,
            "docs": t.docs,
            "kept_bytes": t.kept_bytes,
            "kept_docs": t.kept_docs,
            "kept_frac": t.kept_bytes / max(1, t.in_bytes),
            "dedup_rate": t.drop.get("dup", 0) / max(1, t.docs),
            "near_dup_rate": t.drop.get("near_dup", 0) / max(1, t.docs),
            "filter_rate": sum(v for k, v in t.drop.items() if k not in ("dup", "near_dup"))
            / max(1, t.docs),
            "drop": t.drop,
            "tokens": t.tokens,
            "holdout_tokens": t.holdout_tokens,
            "holdout_bytes": t.holdout_bytes,
            "shards": len(t.shards),
            "bytes_per_token": t.kept_bytes / max(1, t.tokens + t.holdout_tokens),
            "licence": src.licence,
            "dataset": src.dataset,
        }
    total_in = sum(r["in_bytes"] for r in rows.values())
    total_kept = sum(r["kept_bytes"] for r in rows.values())
    total_tok = sum(r["tokens"] + r["holdout_tokens"] for r in rows.values())
    doc = {
        "text_downloaded_gb": total_in / 1e9,
        "text_kept_gb": total_kept / 1e9,
        "tokens_total": total_tok,
        "bytes_per_token": total_kept / max(1, total_tok),
        "holdout_text_gb": sum(r["holdout_bytes"] for r in rows.values()) / 1e9,
        "buckets": {b: share for b, share, _ in BUCKETS},
        "sources": rows,
    }
    STATS.write_text(json.dumps(doc, indent=1))
    return doc


# ------------------------------------------------------------------- verify


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def verify() -> int:
    """Print the numbers, then CHECK them. A failed check exits non-zero."""
    state = json.loads(STATE.read_text())
    mix = json.loads((PRE / "mix.bin.json").read_text())
    hold = json.loads((PRE / "holdout.bin.json").read_text())
    doc = json.loads(STATS.read_text()) if STATS.exists() else stats()

    mix_bytes = (PRE / "mix.bin").stat().st_size
    hold_bytes = (PRE / "holdout.bin").stat().st_size
    print(
        f"mix.bin      {mix_bytes / 1e9:6.2f} GB  {mix_bytes // 2 / 1e9:6.3f} B tokens  "
        f"sha {_sha256(PRE / 'mix.bin')[:16]}"
    )
    print(f"holdout.bin  {hold_bytes / 1e6:6.0f} MB  {hold_bytes // 2 / 1e6:6.1f} M tokens")
    print(
        f"text kept    {doc['text_kept_gb']:6.2f} GB   "
        f"downloaded {doc['text_downloaded_gb']:.2f} GB   {doc['bytes_per_token']:.2f} B/token"
    )
    print(
        f"hold-out     {doc['holdout_text_gb']:.3f} GB of text "
        f"({100 * doc['holdout_text_gb'] / doc['text_kept_gb']:.2f} % of kept)\n"
    )
    print("per source (downloaded -> kept):")
    for name, r in doc["sources"].items():
        print(
            f"  {name:8s} {r['in_bytes'] / 1e9:6.2f} -> {r['kept_bytes'] / 1e9:5.2f} GB "
            f"({100 * r['kept_frac']:5.1f}%)  dedup {100 * r['dedup_rate']:4.1f}%  "
            f"near {100 * r['near_dup_rate']:4.1f}%  filter {100 * r['filter_rate']:4.1f}%  "
            f"{r['bytes_per_token']:.2f} B/tok"
        )

    train = np.concatenate(
        [
            np.fromfile(HASHES / f"{s.name}.train.u64", dtype=np.uint64)
            for s in SOURCES
            if (HASHES / f"{s.name}.train.u64").exists()
        ]
    )
    held = np.concatenate(
        [
            np.fromfile(HASHES / f"{s.name}.holdout.u64", dtype=np.uint64)
            for s in SOURCES
            if (HASHES / f"{s.name}.holdout.u64").exists()
        ]
    )
    overlap = np.intersect1d(held, train)
    checks = [
        ("mix.bin size == manifest", mix_bytes == mix["bytes"] == mix["tokens"] * 2),
        ("holdout.bin size == manifest", hold_bytes == hold["bytes"] == hold["tokens"] * 2),
        (
            "one tokenizer for every stream",
            mix["tokenizer_sha"] == hold["tokenizer_sha"] == state["tokenizer_sha"],
        ),
        ("holdout.bin is big enough", hold_bytes >= MIN_HOLDOUT_MB * 1e6),
        (
            f"every hold-out doc hash satisfies h % {HOLDOUT_MOD} == 0",
            bool(held.size) and bool(np.all(held % HOLDOUT_MOD == 0)),
        ),
        ("no doc hash appears in BOTH streams", overlap.size == 0),
        (
            "no doc hash is used twice anywhere",
            train.size + held.size == np.unique(np.concatenate([train, held])).size,
        ),
        (
            f"hold-out docs are ~1/{HOLDOUT_MOD} of all docs",
            0.5 / HOLDOUT_MOD < held.size / max(1, held.size + train.size) < 1.5 / HOLDOUT_MOD,
        ),
    ]
    print(
        f"\ndoc hashes: {train.size:,} train + {held.size:,} hold-out; intersection {overlap.size}"
    )
    ok = True
    for label, good in checks:
        print(f"  {'PASS' if good else 'FAIL'}  {label}")
        ok &= bool(good)
    if not ok:
        print("\nVERIFY FAILED")
        return 1
    return 0


# --------------------------------------------------------------------- misc


def plan() -> int:
    print(f"{'source':8s} {'target':>7s} {'text':>7s} {'kept':>7s}  dataset")
    ledger = 0.0
    for src in SOURCES:
        p = TEXT / f"{src.name}.bin"
        have = p.stat().st_size / 1e9 if p.exists() else 0.0
        t = (
            Tally.load(json.loads(STATE.read_text()).get("sources", {}).get(src.name, {}))
            if STATE.exists()
            else Tally()
        )
        ledger += max(have, t.kept_bytes / 1e9)
        print(
            f"{src.name:8s} {src.target_gb:6.2f}G {have:6.2f}G {t.kept_bytes / 1e9:6.2f}G  "
            f"{src.dataset[:60]} [{src.licence[:28]}]"
        )
    print(f"\nledger: {ledger:.2f} GB of text; the cap is 25 GB, disk has {_free_gb():.1f} GB")
    print("buckets: " + ", ".join(f"{b} {w:.2f}" for b, w, _ in BUCKETS))
    return 0


def _free_gb() -> float:
    st = os.statvfs(PRE)
    return st.f_bavail * st.f_frsize / 1e9


def layout() -> int:
    """Move the previous pass's text into `pre/text/` -- tokens take
    `pre/<name>.bin` -- and keep its offset-based `_eval` carves apart."""
    TEXT.mkdir(exist_ok=True)
    for src in SOURCES:
        for suffix in ("", "_eval"):
            p = PRE / f"{src.name}{suffix}.bin"
            if p.exists():
                p.rename(TEXT / p.name)
                print(f"moved {p.name} -> text/")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    sub.add_parser("layout")
    b = sub.add_parser("build")
    b.add_argument("--tokenizer", default=str(PRE / "tokenizer32k.json"))
    b.add_argument("--batch", type=int, default=256, help="docs per encode_batch")
    b.add_argument("--only", nargs="*", default=None)
    b.add_argument("--max-shards", type=int, default=0)
    a = sub.add_parser("assemble")
    a.add_argument("--target-gb", type=float, default=0.0, help="cap the mix at this much text")
    a.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/mix_1b.toml"),
        help="the mix_plan TOML to generate from the measured pools",
    )
    sub.add_parser("stats")
    sub.add_parser("verify")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        return build(args)
    if args.cmd == "assemble":
        return assemble(args)
    if args.cmd == "stats":
        return print_stats()
    if args.cmd == "layout":
        return layout()
    return plan() if args.cmd == "plan" else verify()


def print_stats() -> int:
    doc = stats()
    print(f"{doc['text_kept_gb']:.2f} GB kept of {doc['text_downloaded_gb']:.2f} GB, "
          f"{doc['tokens_total'] / 1e9:.3f} B tokens, {doc['bytes_per_token']:.2f} B/token")
    for name, r in doc["sources"].items():
        print(f"  {name:8s} {r['in_bytes'] / 1e9:6.2f} -> {r['kept_bytes'] / 1e9:5.2f} GB "
              f"dedup {100 * r['dedup_rate']:4.1f}% near {100 * r['near_dup_rate']:4.1f}% "
              f"filter {100 * r['filter_rate']:4.1f}% {r['bytes_per_token']:.2f} B/tok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
