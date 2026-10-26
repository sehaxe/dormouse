"""The byte stream: a memory-mapped `.bin`, a shard DIRECTORY, a fixed eval window.

`crates/dormouse-data`'s contract, at the port's day-one scope: bytes in, one
`[b, t]` window out, a deterministic train/eval split by BYTE RANGE, `rewind()`
for a repeatable measurement and `skip_bytes()` for a resume fast-forward. No
shuffling, no parquet, no images — a corpus is a byte sequence and a byte-level
model predicts the next byte.

**Two modes, chosen by the path's SHAPE** (`open_source`):

* a FILE is [`ByteStream`][dormouse.train.data.ByteStream] — the mmap'd `.bin`,
  one contiguous byte range, the ring that wraps at the end of it;
* a DIRECTORY is [`ShardStream`][dormouse.train.data.ShardStream] — the mount's
  corpora are 15 000 × 6.3 MB `*.txt` shards (103 GB for `web`), so the stream
  reads them in a SEEDED SHUFFLED order, one open file at a time, wrapping into
  the next epoch at the end of the last shard. `plan_txt_streamer.md` is why the
  shuffle is the mode's whole point: the shards were written in download order,
  so consecutive files are one crawl/period and reading them in order is the
  category-boundary loss spike the burn side paid for (1.46 → 8.69 at ~step 25k
  of a joint run). The mixture interleaves SOURCES per step; this is the missing
  shuffle WITHIN a source.

Both modes carry the same position semantics, so the trainer's resume path
(`skip_bytes(start × (seq_len+1) × batch)`) is exact for either: the flat stream
is periodic in its byte range and the shard stream is periodic in
`sum(shard sizes)`.

Two properties are load-bearing because the burn history paid for them:

* **the eval window is `eval_batches × batch × seq_len` BYTES** (AGENTS §2.6),
  printed on the eval line by the trainer. A BPB is only comparable within one
  window, and quoting "20 batches" without `batch × seq_len` is not a number.
* **the eval stream never trains.** The split is by byte range, not by a flag,
  so a train batch cannot be drawn from the eval tail (`c4214ad`'s no-leak
  enforcement).

`doc_masking=true` (a THIRD reader, [`DocPackedStream`][dormouse.train.data.
DocPackedStream]) replaces the train stream with one DOCUMENT per row and a
`[b,t]` loss mask; the eval stream stays flat. It needs `<data>.docs.json`
(the offsets `scripts/build_doc_sidecar.py` derives), refuses without it, and
is by STEP on resume, not by flat byte — see the class docstring for why.
"""

from __future__ import annotations

import bisect
import json
import math
import mmap
import os
import random
from pathlib import Path
from typing import IO

import torch

from ..config import TrainConfig
from ..instrument import loud

DEFAULT_RING = 64 << 20  # burn's 64 MB ring
DEFAULT_REFILL = 8 << 20  # refilled in 8 MB chunks

#: The document-offset sidecar `doc_masking` reads: `<data>.docs.json`, written
#: by `scripts/build_doc_sidecar.py`. A convention, not a config key: a run
#: naming a file either has its boundaries beside it or is refused, so a
#: sidecar cannot point at a different file than the bytes being streamed.
DOC_SIDECAR_SUFFIX = ".docs.json"


def _windows(raw: bytes, batch: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`batch × (seq_len+1)` bytes → one `[b, t]` window of inputs and the byte after each.

    ONE definition of the next-byte shift, shared by both stream modes: the
    target at position `q` is the byte that follows the input's, which is what
    makes a window `batch × seq_len` PREDICTIONS and not `batch × seq_len` bytes
    of context.
    """
    ids = torch.tensor(list(raw), dtype=torch.long).view(batch, seq_len + 1)
    return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


def _refuse_dir(path: Path, what: str) -> None:
    """A directory where the single-file mode is required — LOUD, naming the escape."""
    if path.is_dir():
        loud(
            f"{what} {path} is a directory. The flat `.bin` mode takes ONE file; a "
            f"directory of `*.txt` shards streams through the mixture (name it in a mix "
            f"plan's [sources] -> `ShardStream`). Escape: set `mix_plan=<plan.toml>`, or "
            f"point it at a `.bin`."
        )


# --- the shard directory ----------------------------------------------------


def shard_files(directory: str | Path) -> list[Path]:
    """A shard directory's `*.txt` files in DOWNLOAD order (sorted by name).

    This is the list the shuffle permutes. Non-`.txt` files are never read (the
    mount's `math/` also holds two `.jsonl` dumps), and the two shapes that
    would stream nothing are LOUD:

    * the path is not a directory at all (a `.bin` file belongs to `ByteStream`);
    * the directory holds no `*.txt` — a wrong path must not read 0 bytes
      (pretrain v21 trained on constant filler; the size floor is why).
    """
    d = Path(directory)
    if not d.is_dir():
        loud(
            f"shard directory {d} is not a directory. Escape: pass the directory that "
            f"holds the `*.txt` shards, or a `.bin` file for the flat mode."
        )
    files = sorted(p for p in d.glob("*.txt") if p.is_file())
    if not files:
        loud(
            f"shard directory {d} holds no `*.txt` shards: a stream with no bytes is "
            f"not a run. Escape: point at the directory that holds the shards, or "
            f"fetch them (`scripts/fetch_mix_sources.py`)."
        )
    return files


def shard_sizes(files: list[Path]) -> list[int]:
    """Each shard's size in bytes — LOUD on a zero-byte shard.

    A shard that yields no text is NAMED, not silently skipped: the parquet
    lesson (AGENTS §3.2) was a source that read 21 811 B where 8 000 000 were
    asked for, with the run's loss curve reporting as if it had not.
    """
    sizes = [p.stat().st_size for p in files]
    empty = [p.name for p, size in zip(files, sizes, strict=True) if size == 0]
    if empty:
        loud(
            f"{len(empty)} zero-byte shard(s) in {files[0].parent}, first {empty[0]!r}: a "
            f"shard that yields no text is named, never skipped. Escape: re-fetch or "
            f"delete it, or drop the directory from the plan."
        )
    return sizes


def shard_order(files: list[Path], seed: int) -> list[Path]:
    """`files` in the order ONE epoch reads them: a seeded Fisher–Yates.

    A pure function of `(files, seed)`: a hand-rolled Fisher–Yates over
    `random.Random(seed)`'s stream (pinned by a golden test), never a global
    RNG — the same seed is the same file order in any process and after any
    resume (ADR-0021's doctrine, the defect that fixed DSpark's stride).
    """
    assert seed >= 0, f"seed must be >= 0, got {seed}"
    rng = random.Random(seed)
    out = list(files)
    for i in range(len(out) - 1, 0, -1):
        j = rng.randrange(i + 1)
        out[i], out[j] = out[j], out[i]
    return out


class ShardStream:
    """`ByteStream`'s interface over a DIRECTORY of `*.txt` shards.

    The mount's corpora are directories of ~6.3 MB shards (103 GB for `web`,
    86 GB for `math`), and concatenating them into a `.bin` would duplicate
    227 GB for nothing while the drive sustains 146 MB/s against a training read
    of ~40 KB/s. So the stream reads the directory itself: `shard_files` (sorted)
    → `shard_order` (seeded permutation) → ONE open file at a time, read
    sequentially, and at EOF the next file in the shuffled order.

    **The position is `(file index in the shuffled order, byte offset in that
    file)`** and the epoch is the concatenation of the shuffled files; past the
    last file the order WRAPS (epoch 2), exactly as `ByteStream`'s ring wraps at
    the end of its range. That periodicity is what makes `skip_bytes` exact with
    no sidecar file: `n % sum(shard sizes)` names the same byte as `n`, so a
    resume lands on the byte an unbroken run would have read — across shard
    boundaries and across epochs (`tests/test_shard.py`).

    Bytes are bytes: nothing is decoded, so a shard that is not UTF-8 streams
    exactly as the byte vocabulary expects.
    """

    def __init__(
        self,
        directory: str | Path,
        batch: int,
        seq_len: int,
        *,
        seed: int = 1,
        ring_bytes: int = DEFAULT_RING,
        refill_bytes: int = DEFAULT_REFILL,
    ) -> None:
        self.directory = Path(directory)
        self.seed = seed
        self.files = shard_order(shard_files(self.directory), seed)
        self.sizes = shard_sizes(self.files)
        self.total = sum(self.sizes)
        #: Epoch offsets: `_cum[i]` is where shard `i` starts in the epoch.
        self._cum = [0]
        for size in self.sizes:
            self._cum.append(self._cum[-1] + size)
        self.batch, self.seq_len = batch, seq_len
        self.ring_bytes = ring_bytes
        self.refill_bytes = refill_bytes
        self.tokens_per_batch = batch * seq_len
        if self.total < self.tokens_per_batch + 1:
            loud(
                f"shard directory {self.directory} holds {self.total} B over "
                f"{len(self.files)} shard(s), less than one batch of "
                f"{self.tokens_per_batch + 1}: enlarge the directory or shrink the batch"
            )
        self._buf = bytearray()
        self._idx = 0
        self._off = 0
        self._fh: IO[bytes] | None = None

    # -- position -------------------------------------------------------
    @property
    def position(self) -> tuple[int, int]:
        """`(file index in the shuffled order, byte offset)` of the NEXT byte out.

        Derived from the bytes CONSUMED, not from the open file's offset: the
        read cursor leads the consumption cursor by the ring's contents, and a
        window may straddle a shard boundary (`_read` refills across it), so a
        handle-relative answer would name a byte the stream has not reached.
        """
        at = self._epoch_offset()
        idx = bisect.bisect_right(self._cum, at) - 1
        return (idx, at - self._cum[idx])

    def _epoch_offset(self) -> int:
        """The next byte's offset in the shuffled concatenation, modulo one epoch.

        The ring holds the bytes from `R - len(_buf)` to `R` in epoch
        coordinates (`R = _cum[_idx] + _off`), and the modulo carries the wrap:
        after the last shard the buffer holds the epoch's tail followed by its
        head, so the difference goes negative and `% total` names the tail.
        """
        return (self._cum[self._idx] + self._off - len(self._buf)) % self.total

    # -- ring -----------------------------------------------------------
    def _close_fh(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _next_file(self) -> None:
        """EOF of a shard: the next file in the shuffled order, wrapping into epoch 2."""
        assert self._off >= self.sizes[self._idx], "a shard is left before its end"
        self._close_fh()
        self._idx = (self._idx + 1) % len(self.files)
        self._off = 0

    def _refill(self) -> None:
        want = min(self.refill_bytes, self.ring_bytes - len(self._buf))
        assert want > 0, "the ring is full and still cannot satisfy the request"
        if self._off >= self.sizes[self._idx]:
            self._next_file()
        if self._fh is None:
            self._fh = open(self.files[self._idx], "rb")
            self._fh.seek(self._off)
        take = min(want, self.sizes[self._idx] - self._off)
        chunk = self._fh.read(take)
        if len(chunk) != take:
            loud(
                f"shard {self.files[self._idx]} ended at byte {self._off + len(chunk)} but "
                f"its size at open was {self.sizes[self._idx]}: the drive changed under "
                f"the stream (a partial download?). Escape: re-fetch the shard, or drop "
                f"the directory from the plan."
            )
        self._buf += chunk
        self._off += len(chunk)

    def _read(self, n: int) -> bytes:
        assert n > 0
        while len(self._buf) < n:
            self._refill()
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """One `[batch, seq_len]` window of inputs and the byte after each."""
        want = self.batch * (self.seq_len + 1)
        return _windows(self._read(want), self.batch, self.seq_len)

    # -- positioning ----------------------------------------------------
    def rewind(self) -> None:
        """Back to the first file of the shuffled order — every eval starts here."""
        self._buf.clear()
        self._close_fh()
        self._idx = 0
        self._off = 0

    def skip_bytes(self, n: int) -> None:
        """Fast-forward `n` bytes of the shuffled concatenation — EXACTLY.

        The epoch is `sum(shard sizes)` bytes and the stream is periodic in it
        (every epoch re-reads the same shards in the same order), so `n % total`
        names the same byte as `n`: the shard is found by cumulative sizes and the
        offset inside it. No whole-file consumption, no sidecar — and a resume
        lands on the byte an unbroken run would have read.
        """
        if n < 0:
            loud(f"skip_bytes({n}): a negative fast-forward is a rewind")
        target = n % self.total
        self._buf.clear()
        self._close_fh()
        self._idx = bisect.bisect_right(self._cum, target) - 1
        self._off = target - self._cum[self._idx]

    def close(self) -> None:
        self._buf.clear()
        self._close_fh()


class ByteStream:
    """A read-only ring over `[lo, hi)` of a memory-mapped file.

    The wrap is BYTE-EXACT: byte `hi - 1` is followed by byte `lo`, so one cycle
    is the range's `hi - lo` bytes in file order and a window that straddles the
    end is the range's tail followed by its head — nothing is skipped or
    repeated at the seam (`tests/test_data.py` scans a full cycle). `_pos` is
    the next byte to read and stays in `[lo, hi)`; `rewind()` and
    `skip_bytes()` only ever move it inside that range.
    `check_min=False` is the document-packing reader's construction
    ([`DocPackedStream`][dormouse.train.data.DocPackedStream]): its rows come
    from documents, so a range shorter than one flat `batch × (seq_len+1)`
    window is not a defect there. Every flat caller keeps the check.
    """

    def __init__(
        self,
        path: str | Path,
        batch: int,
        seq_len: int,
        *,
        lo: int = 0,
        hi: int | None = None,
        ring_bytes: int = DEFAULT_RING,
        refill_bytes: int = DEFAULT_REFILL,
        check_min: bool = True,
    ) -> None:
        self.path = Path(path)
        _refuse_dir(self.path, "data file")
        if not self.path.is_file():
            loud(f"data file {self.path} does not exist (mount the drive, or fix the path)")
        self.batch, self.seq_len = batch, seq_len
        self.ring_bytes = ring_bytes
        self.refill_bytes = refill_bytes
        self._fd = os.open(self.path, os.O_RDONLY)
        self._mm = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)
        size = len(self._mm)
        if size == 0:
            loud(f"data file {self.path} is empty: a stream with no bytes is not a run")
        self.size = size
        self.lo = lo
        self.hi = size if hi is None else hi
        if not 0 <= self.lo < self.hi <= size:
            loud(f"byte range [{self.lo}, {self.hi}) is not inside {self.path} ({size} B)")
        self.tokens_per_batch = batch * seq_len
        if check_min and self.hi - self.lo < self.tokens_per_batch + 1:
            loud(
                f"byte range [{self.lo}, {self.hi}) holds {self.hi - self.lo} B, less than one "
                f"batch of {self.tokens_per_batch + 1}: enlarge the range or shrink the batch"
            )
        self._buf = bytearray()
        self._pos = self.lo
        self._rng_lo = self.lo
        #: `(requested, granted)` when `open_streams` capped the eval tail at
        #: 1 MiB; `None` is the uncapped path (no line, no changed number).
        self.eval_tail_capped: tuple[int, int] | None = None

    # -- ring -----------------------------------------------------------
    def _refill(self) -> None:
        want = min(self.refill_bytes, self.ring_bytes - len(self._buf))
        assert want > 0, "the ring is full and still cannot satisfy the request"
        take = min(want, self.hi - self._pos)
        if take <= 0:
            # End of the range: wrap to the FIRST byte, byte-exact. This is the
            # ring's whole point — the stream is a fixed byte range read
            # repeatedly, not a file cursor — and `lo`, not `lo + k`, is what
            # keeps the cycle free of skips and repeats.
            self._pos = self.lo
            take = min(want, self.hi - self._pos)
        self._buf += self._mm[self._pos : self._pos + take]
        self._pos += take

    def _read(self, n: int) -> bytes:
        assert n > 0
        while len(self._buf) < n:
            self._refill()
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """One `[batch, seq_len]` window of inputs and the byte after each."""
        want = self.batch * (self.seq_len + 1)
        return _windows(self._read(want), self.batch, self.seq_len)

    # -- positioning ----------------------------------------------------
    def rewind(self) -> None:
        """Back to the start of the range — every eval starts here.

        Before this existed on the burn side, the same checkpoint scored 6.443
        and 6.551 on consecutive evals from stream position alone (AGENTS
        §2.6), i.e. cross-run A/Bs were noise.
        """
        self._buf.clear()
        self._pos = self.lo

    def skip_bytes(self, n: int) -> None:
        """Fast-forward `n` bytes (resume): a resumed run must not re-read the
        corpus head it already trained on."""
        if n < 0:
            loud(f"skip_bytes({n}): a negative fast-forward is a rewind")
        self._buf.clear()
        span = self.hi - self.lo
        self._pos = self.lo + (n % span)

    def close(self) -> None:
        self._buf.clear()
        self._mm.close()
        os.close(self._fd)

    # -- random access (the document-packing reader) --------------------
    def read_at(self, pos: int, n: int) -> bytes:
        """`n` bytes from absolute `pos`, wrapping at `hi` into `lo`.

        The ring's `_read` is sequential; the packed reader needs the byte at a
        document's start, which can be anywhere in the range. The wrap is the
        same periodicity `_refill` implements, so `read_at` and `next_batch`
        name the same byte at the same position.
        """
        span = self.hi - self.lo
        assert 0 < n <= span, f"read_at({pos}, {n}) outside the {span}-byte range"
        assert self.lo <= pos < self.hi, f"read_at({pos}) outside [{self.lo}, {self.hi})"
        if pos + n <= self.hi:
            return bytes(self._mm[pos : pos + n])
        first = bytes(self._mm[pos : self.hi])
        return first + bytes(self._mm[self.lo : self.lo + (n - len(first))])


# --- document boundaries (doc_masking) ---------------------------------------


def doc_sidecar_path(data: str | Path) -> Path:
    """Where the document offsets of `data` live: `<data>.docs.json`."""
    return Path(str(data) + DOC_SIDECAR_SUFFIX)


def load_doc_starts(data: str | Path, sidecar: str | Path | None = None) -> list[int]:
    """The document START offsets of a `.bin`, LOUD when it has no sidecar.

    The sidecar is JSON `{"version": 1, "size": <bytes>, "starts": [...]}`.
    Every way it can lie about the file is a refusal: a missing file, a wrong
    version, a size that is not the file's (a stale sidecar describes a
    previous build, so every offset after the change would slice the wrong
    document), a first start that is not 0, a non-ascending or duplicated
    start, or a start at/after the end. There is no "fall back to the flat
    stream" — that would train with the boundaries this flag exists to add.
    """
    p = Path(data)
    side = doc_sidecar_path(p) if sidecar is None else Path(sidecar)
    if not p.is_file():
        loud(f"data file {p} does not exist (mount the drive, or fix the path)")
    if not side.is_file():
        loud(
            f"doc_masking=true but {p} has no document-boundary sidecar {side}. The "
            f"bytes are a flat concatenation: no reader can recover where one "
            f"document ends and the next begins without the offsets. Escape: build "
            f"it with `uv run scripts/build_doc_sidecar.py --bin {p} --source-dir "
            f"<the *.txt shard directory the domain was built from>`, or set "
            f"doc_masking=false (the flat stream, every number on record)."
        )
    try:
        raw = json.loads(side.read_text())
    except json.JSONDecodeError as e:
        loud(f"document sidecar {side} is not JSON: {e}. Escape: rebuild it.")
    if not isinstance(raw, dict):
        loud(f"document sidecar {side} is not a JSON object. Escape: rebuild it.")
    if raw.get("version") != 1:
        loud(
            f"document sidecar {side}: version {raw.get('version')!r} is not 1. "
            f"Escape: rebuild it with scripts/build_doc_sidecar.py."
        )
    try:
        size = int(raw.get("size", -1))
    except (TypeError, ValueError):
        loud(f"document sidecar {side}: `size` is not an integer. Escape: rebuild it.")
    file_size = p.stat().st_size
    if size != file_size:
        loud(
            f"document sidecar {side} describes {size} B but {p} is {file_size} B: "
            f"the sidecar is stale (the shard was rebuilt or moved). Escape: rebuild "
            f"it with scripts/build_doc_sidecar.py."
        )
    try:
        starts = [int(s) for s in raw.get("starts", [])]
    except (TypeError, ValueError):
        loud(f"document sidecar {side}: `starts` is not a list of integers. Escape: rebuild it.")
    if not starts or starts[0] != 0:
        loud(f"document sidecar {side}: starts must begin at 0, got {starts[:1]}")
    for a, b in zip(starts, starts[1:], strict=False):
        if b <= a:
            loud(f"document sidecar {side}: starts are not strictly ascending at {a} -> {b}")
    if starts[-1] >= file_size:
        loud(f"document sidecar {side}: a document starts at {starts[-1]} >= {file_size}")
    return starts


class DocPackedStream:
    """`ByteStream`'s interface with ONE DOCUMENT per row — the `doc_masking` arm.

    Why not glue whole documents into one row (sequence PACKING as usually
    meant) and mask the boundary targets: the KDA arm is a RECURRENCE whose
    state is carried through the fused `chunk_kda` over the whole window, so a
    row holding two documents would let document B's positions attend to
    document A's — the cross-document dependency this arm exists to remove.
    Resetting the recurrent state mid-window is not available inside that
    kernel, so the honest implementation is a window that never holds two
    documents: every row is a whole document, or a contiguous chunk of ONE
    document when the document is longer than `seq_len`, padded at the tail.

    The consequence is named rather than hidden: a corpus of short documents
    pays padding (a 200-byte doc leaves ~80 % of a 1024-byte row empty), while
    the 6 MB `web` shards see no padding at all and no behavioural change —
    which is exactly the A/B `scripts/ab_docmask.py` pre-registers.

    The loss mask is `[b, t]` bool: 1 where the target byte belongs to the
    same document as the input byte. That is every position of a chunk that
    continues inside its document, and every position of a whole document
    except its last byte (the successor of a document's last byte is the next
    document, or padding — never a byte this row may be scored against).

    Resume is by STEP, not by flat byte: `skip_bytes(n)` accepts the trainer's
    own `steps × (seq_len+1) × batch` formula and REPLAYS that many packing
    steps from the start of the range (lengths only, no reads), because a
    packed step consumes a document-dependent number of bytes. Replaying 200k
    steps is a few hundred thousand bisects, not a corpus read.
    """

    def __init__(
        self,
        source: ByteStream,
        starts: list[int],
    ) -> None:
        self.source = source
        self.batch, self.seq_len = source.batch, source.seq_len
        self.lo, self.hi = source.lo, source.hi
        self._doc_starts: list[int] = []
        self._doc_ends: list[int] = []
        for i, s in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else self.hi
            a, b = max(s, self.lo), min(end, self.hi)
            if a < b:
                self._doc_starts.append(a)
                self._doc_ends.append(b)
        if not self._doc_starts:
            loud(
                f"doc_masking: {source.path} carries no document inside its byte "
                f"range [{self.lo}, {self.hi}). Escape: a range that contains a "
                f"document, or doc_masking=false."
            )
        self._pos = self.lo
        #: Boundary counters (the long-context gate reads them): how many
        #: windows were packed, how many document pieces they held, how many
        #: supervised positions were inside a document, how many target
        #: positions the mask dropped, and how many rows ended at a document's
        #: last byte (a real boundary, as opposed to a mid-document chunk).
        self.counts = {
            "docmask_windows": 0,
            "docmask_docs": 0,
            "docmask_scored": 0,
            "docmask_masked": 0,
            "docmask_doc_ends": 0,
            "docmask_padded": 0,
        }

    # -- position -------------------------------------------------------
    @property
    def position(self) -> int:
        """The absolute offset of the next document byte (the packing cursor)."""
        return self._pos

    def _doc_at(self, pos: int) -> tuple[int, int]:
        i = bisect.bisect_right(self._doc_starts, pos) - 1
        assert i >= 0, f"position {pos} before the first document"
        return self._doc_starts[i], self._doc_ends[i]

    def _advance(self) -> int:
        """Move the cursor over ONE row's bytes without reading, return the bytes used."""
        a, b = self._doc_at(self._pos)
        remaining = b - self._pos
        whole = remaining <= self.seq_len
        used = remaining if whole else self.seq_len + 1
        nxt = b if whole else self._pos + used
        self._pos = self.lo if nxt >= self.hi else nxt
        return used

    def _row(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """One packed row → (inputs, targets, loss mask, scored positions)."""
        a, b = self._doc_at(self._pos)
        remaining = b - self._pos
        if remaining <= self.seq_len:
            chunk = self.source.read_at(self._pos, remaining)
            x = torch.zeros(self.seq_len, dtype=torch.long)
            y = torch.zeros(self.seq_len, dtype=torch.long)
            mask = torch.zeros(self.seq_len, dtype=torch.bool)
            x[:remaining] = torch.tensor(list(chunk), dtype=torch.long)
            if remaining > 1:
                y[: remaining - 1] = torch.tensor(list(chunk[1:]), dtype=torch.long)
                mask[: remaining - 1] = True
            # The last byte of a document has no successor in it: its target is
            # the next document (a real cross-boundary target) or padding.
            scored = max(remaining - 1, 0)
            self.counts["docmask_doc_ends"] += 1
            nxt = b
        else:
            # A chunk of ONE document: the extra byte is the last position's
            # in-document target, so every position is supervised.
            chunk = self.source.read_at(self._pos, self.seq_len + 1)
            x = torch.tensor(list(chunk[: self.seq_len]), dtype=torch.long)
            y = torch.tensor(list(chunk[1:]), dtype=torch.long)
            mask = torch.ones(self.seq_len, dtype=torch.bool)
            scored = self.seq_len
            nxt = self._pos + self.seq_len + 1
        self._pos = self.lo if nxt >= self.hi else nxt
        self.counts["docmask_docs"] += 1
        self.counts["docmask_scored"] += scored
        self.counts["docmask_masked"] += self.seq_len - scored
        self.counts["docmask_padded"] += self.seq_len - min(remaining, self.seq_len)
        return x, y, mask, scored

    def next_batch_masked(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One `[batch, seq_len]` window of inputs, targets and the loss mask."""
        xs, ys, ms = [], [], []
        for _ in range(self.batch):
            x, y, mask, _ = self._row()
            xs.append(x)
            ys.append(y)
            ms.append(mask)
        self.counts["docmask_windows"] += self.batch
        out = torch.stack(xs), torch.stack(ys), torch.stack(ms)
        if not bool(out[2].any()):
            # Every row was a 1-byte document (or shorter): no target exists, so
            # the masked mean divides by zero. LOUD, not NaN.
            loud(
                "doc_masking: this window contains no scorable position (every row's "
                "document is 1 byte). Escape: a corpus whose documents have a "
                "successor byte, or doc_masking=false."
            )
        return out

    def doc_counts(self) -> dict[str, int]:
        """The boundary counters (a copy: a reader cannot mutate the arm's state)."""
        return dict(self.counts)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        loud(
            "next_batch() on a doc_masking stream: the window carries a loss mask "
            "and dropping it would score padding and cross-document targets. "
            "Escape: read `next_batch_masked()` (the trainer does this when "
            "doc_masking=true), or set doc_masking=false for the flat window."
        )

    # -- positioning ----------------------------------------------------
    def rewind(self) -> None:
        """Back to the start of the range — deterministic and repeatable."""
        self._pos = self.lo

    def skip_steps(self, steps: int) -> None:
        """Replay `steps` packing STEPS from the range start (lengths only).

        One step is one `[batch, seq_len]` window, i.e. `batch` packed rows —
        the same unit `skip_bytes` receives from the trainer's formula.
        """
        if steps < 0:
            loud(f"skip_steps({steps}): a negative fast-forward is a rewind")
        self._pos = self.lo
        for _ in range(steps * self.batch):
            self._advance()

    def skip_bytes(self, n: int) -> None:
        """Fast-forward `n` flat bytes = `n // (batch × (seq_len+1))` packing steps."""
        window = self.batch * (self.seq_len + 1)
        if n < 0:
            loud(f"skip_bytes({n}): a negative fast-forward is a rewind")
        if n % window:
            loud(
                f"skip_bytes({n}) on a doc_masking stream: not a whole number of "
                f"{window}-byte steps (batch={self.batch}, seq_len={self.seq_len}). "
                f"Escape: pass steps × (seq_len+1) × batch."
            )
        self.skip_steps(n // window)

    def close(self) -> None:
        self.source.close()


def open_source(
    path: str | Path,
    batch: int,
    seq_len: int,
    *,
    seed: int = 1,
    ring_bytes: int = DEFAULT_RING,
) -> ByteStream | ShardStream:
    """A source path → the stream its SHAPE names: a directory is a `ShardStream`.

    ONE definition of the rule, so the mixture and any other caller cannot
    disagree about which mode a path opens. `seed` is the shard order's; a file
    ignores it.
    """
    p = Path(path)
    if p.is_dir():
        return ShardStream(p, batch, seq_len, seed=seed, ring_bytes=ring_bytes)
    return ByteStream(p, batch, seq_len, ring_bytes=ring_bytes)


def eval_window_bytes(cfg: TrainConfig) -> int:
    """`eval_batches × batch × seq_len` — the ONLY comparable window size."""
    return cfg.eval_batches * cfg.batch * cfg.seq_len


def distinct_scored_bytes(stream: ByteStream, batches: int) -> int:
    """How many DISTINCT byte positions `batches` windows of `stream` score.

    `over N B` on the eval line is a count of scored tokens: for a window that
    fits in the stream's range it is `batches × batch × seq_len`. Above the
    eval tail's 1 MiB cap the window is LONGER than the range, `next_batch`
    wraps to `lo`, and the same bytes are scored again — summing `x.numel()`
    then counts duplicates (measured: 100 × 16 × 1024 = 1 638 400 requested
    over a 1 048 576 B range, batch 15 byte-identical to batch 0). This walks
    the same positions the loop reads and counts each once. The `total <= span`
    fast path is exactly the old number, so an uncapped window does not move.
    """
    span = stream.hi - stream.lo
    total = batches * stream.batch * stream.seq_len
    if span <= 0 or total <= span:
        return total
    window = stream.batch * (stream.seq_len + 1)
    seen = bytearray(span)
    for k in range(batches):
        base = (k * window) % span
        for row in range(stream.batch):
            start = (base + row * (stream.seq_len + 1)) % span
            end = start + stream.seq_len
            if end <= span:
                seen[start:end] = b"\x01" * stream.seq_len
            else:
                seen[start:] = b"\x01" * (span - start)
                seen[: end - span] = b"\x01" * (end - span)
    return sum(seen)


def split_points(size: int, eval_bytes: int) -> tuple[int, int]:
    """`(hi_train, hi_total)`: the eval tail is the LAST `eval_bytes` bytes."""
    if eval_bytes < 0:
        loud(f"eval_bytes={eval_bytes} must be >= 0")
    if eval_bytes >= size:
        loud(f"eval_bytes={eval_bytes} >= file size {size}: the train side would be empty")
    return size - eval_bytes, size


def open_streams(
    run, *, ring_bytes: int = DEFAULT_RING, eval_only: bool = False
) -> tuple[ByteStream | DocPackedStream | None, ByteStream | None]:
    """The train stream and (when an eval file is set) the eval stream.

    `eval_data` names a DIFFERENT file (the burn recipe's held-out tail carve).
    When it is empty the eval tail of the train file is used, and the train
    stream's high bound stops one window short of it — the no-leak split.

    `doc_masking=true` replaces the TRAIN stream with a
    [`DocPackedStream`][dormouse.train.data.DocPackedStream] (one document per
    row, mask included) and leaves the eval stream flat ON PURPOSE: the
    held-out BPB is the recorded, comparable number, and re-cutting its window
    would make every doc-masked run's eval a different measurement. The flag
    changes what the training objective sees, not the held-out instrument.

    `eval_only=True` is the mixture's call site: the TRAIN side comes from
    `MixtureStream`, so building (and validating) a second, unused train stream
    here would make a `mix_plan` run demand a sidecar for a file it never reads
    (`train/loop.py`'s `open_mixture` branch).
    """
    tr = run.train
    if tr.eval_data:
        ev = ByteStream(
            tr.eval_data,
            tr.batch,
            tr.seq_len,
            ring_bytes=min(ring_bytes, max(1 << 20, eval_window_bytes(tr))),
        )
        if eval_only:
            return None, ev
        if not tr.data:
            loud(
                "no data file: `data` is empty in the train config. Escape: set "
                '`data = "/path/to/corpus.bin"` (a flat .bin of bytes).'
            )
        # A directory here would `stat()` as a few KB of inode and stream a
        # "corpus" of 4096 bytes with no error — the silent-fallback class. The
        # shard mode lives in the mixture, so the escape is the plan.
        _refuse_dir(Path(tr.data), "data")
        train = ByteStream(
            tr.data, tr.batch, tr.seq_len, ring_bytes=ring_bytes, check_min=not tr.doc_masking
        )
        if tr.doc_masking:
            return DocPackedStream(train, load_doc_starts(tr.data)), ev
        return train, ev
    if not tr.data:
        loud(
            "no data file: `data` is empty in the train config. Escape: set "
            '`data = "/path/to/corpus.bin"` (a flat .bin of bytes).'
        )
    _refuse_dir(Path(tr.data), "data")
    ev_bytes = min(max(eval_window_bytes(tr), 1), 1 << 20) if tr.eval else 0
    hi_train, _ = split_points(Path(tr.data).stat().st_size, ev_bytes)
    ev = (
        ByteStream(tr.data, tr.batch, tr.seq_len, lo=hi_train, ring_bytes=1 << 20)
        if tr.eval
        else None
    )
    requested = eval_window_bytes(tr) if tr.eval else 0
    if ev is not None and ev_bytes < requested:
        # COUNTED, never silent: the request is larger than the tail this
        # stream may read, so the stream wraps and the eval re-scores bytes it
        # already scored. The eval line prints `eval_tail_capped=req/granted`
        # and its `over N B` counts DISTINCT bytes (`distinct_scored_bytes`).
        ev.eval_tail_capped = (requested, ev_bytes)
        print(
            f"eval_tail_capped={requested}/{ev_bytes}: eval_batches x batch x seq_len "
            f"asks for {requested} B but the eval tail is capped at {ev_bytes} B; the "
            f"stream wraps and `over N B` counts distinct bytes only. Escape: shrink "
            f"eval_batches/batch/seq_len below {1 << 20} B, or point eval_data at a "
            f"larger held-out file."
        )
    if eval_only:
        return None, ev
    train = ByteStream(
        tr.data,
        tr.batch,
        tr.seq_len,
        lo=0,
        hi=hi_train,
        ring_bytes=ring_bytes,
        check_min=not tr.doc_masking,
    )
    if tr.doc_masking:
        train = DocPackedStream(train, load_doc_starts(tr.data))
    return train, ev


def warmup_steps(warmup_tokens: int, batch: int, seq_len: int) -> int:
    """The WSD warmup is defined in TOKENS; the schedule runs in steps."""
    if warmup_tokens <= 0:
        return 0
    return int(math.ceil(warmup_tokens / (batch * seq_len)))
