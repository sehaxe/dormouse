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
"""

from __future__ import annotations

import bisect
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
    """A read-only ring over `[lo, hi)` of a memory-mapped file."""

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
        if self.hi - self.lo < self.tokens_per_batch + 1:
            loud(
                f"byte range [{self.lo}, {self.hi}) holds {self.hi - self.lo} B, less than one "
                f"batch of {self.tokens_per_batch + 1}: enlarge the range or shrink the batch"
            )
        self._buf = bytearray()
        self._pos = self.lo
        self._rng_lo = self.lo

    # -- ring -----------------------------------------------------------
    def _refill(self) -> None:
        want = min(self.refill_bytes, self.ring_bytes - len(self._buf))
        assert want > 0, "the ring is full and still cannot satisfy the request"
        take = min(want, self.hi - self._pos)
        if take <= 0:
            # End of the range: wrap. This is the ring's whole point — the
            # stream is a fixed byte range read repeatedly, not a file cursor.
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


def eval_read_bytes(cfg: TrainConfig) -> int:
    """`eval_batches × batch × (seq_len + 1)` — the bytes the eval READS.

    One window READS one byte more than it SCORES (the next-byte shift), so a
    tail sized as scored bytes holds one window fewer than the config asks for
    (review round 2 R2-1: 20 x 8 x 512 on a 81 920 B tail scored 19 windows).
    `open_streams` carves the no-`--eval` tail with THIS number; the 1 MiB cap
    is then the only reason the eval can still fall short, and `_eval_line`
    names it.
    """
    return cfg.eval_batches * cfg.batch * (cfg.seq_len + 1)


def split_points(size: int, eval_bytes: int) -> tuple[int, int]:
    """`(hi_train, hi_total)`: the eval tail is the LAST `eval_bytes` bytes."""
    if eval_bytes < 0:
        loud(f"eval_bytes={eval_bytes} must be >= 0")
    if eval_bytes >= size:
        loud(f"eval_bytes={eval_bytes} >= file size {size}: the train side would be empty")
    return size - eval_bytes, size


def open_streams(run, *, ring_bytes: int = DEFAULT_RING) -> tuple[ByteStream, ByteStream | None]:
    """The train stream and (when an eval file is set) the eval stream.

    `eval_data` names a DIFFERENT file (the burn recipe's held-out tail carve).
    When it is empty the eval tail of the train file is used, and the train
    stream's high bound stops one window short of it — the no-leak split. The
    tail is sized in READS (`eval_read_bytes`), not scored bytes, so
    `eval_batches` whole windows fit (capped at 1 MiB; a shortfall above the
    cap is named on the eval line).
    """
    tr = run.train
    if not tr.data:
        loud(
            "no data file: `data` is empty in the train config. Escape: set "
            '`data = "/path/to/corpus.bin"` (a flat .bin of bytes).'
        )
    # A directory here would `stat()` as a few KB of inode and stream a
    # "corpus" of 4096 bytes with no error — the silent-fallback class. The
    # shard mode lives in the mixture, so the escape is the plan.
    _refuse_dir(Path(tr.data), "data")
    if tr.eval_data:
        ev = ByteStream(
            tr.eval_data,
            tr.batch,
            tr.seq_len,
            ring_bytes=min(ring_bytes, max(1 << 20, eval_window_bytes(tr))),
        )
        train = ByteStream(tr.data, tr.batch, tr.seq_len, ring_bytes=ring_bytes)
        return train, ev
    ev_bytes = min(max(eval_read_bytes(tr), 1), 1 << 20) if tr.eval else 0
    hi_train, _ = split_points(Path(tr.data).stat().st_size, ev_bytes)
    train = ByteStream(tr.data, tr.batch, tr.seq_len, lo=0, hi=hi_train, ring_bytes=ring_bytes)
    ev = (
        ByteStream(tr.data, tr.batch, tr.seq_len, lo=hi_train, ring_bytes=1 << 20)
        if tr.eval
        else None
    )
    return train, ev


def warmup_steps(warmup_tokens: int, batch: int, seq_len: int) -> int:
    """The WSD warmup is defined in TOKENS; the schedule runs in steps."""
    if warmup_tokens <= 0:
        return 0
    return int(math.ceil(warmup_tokens / (batch * seq_len)))
