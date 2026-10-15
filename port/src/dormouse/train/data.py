"""The byte stream: a memory-mapped `.bin`, a ring buffer, a fixed eval window.

`crates/dormouse-data`'s contract, at the port's day-one scope: bytes in, one
`[b, t]` window out, a deterministic train/eval split by BYTE RANGE, `rewind()`
for a repeatable measurement and `skip_bytes()` for a resume fast-forward. No
shuffling, no parquet, no images — a corpus is a byte sequence and a byte-level
model predicts the next byte.

Two properties are load-bearing because the burn history paid for them:

* **the eval window is `eval_batches × batch × seq_len` BYTES** (AGENTS §2.6),
  printed on the eval line by the trainer. A BPB is only comparable within one
  window, and quoting "20 batches" without `batch × seq_len` is not a number.
* **the eval stream never trains.** The split is by byte range, not by a flag,
  so a train batch cannot be drawn from the eval tail (`c4214ad`'s no-leak
  enforcement).
"""

from __future__ import annotations

import math
import mmap
import os
from pathlib import Path

import torch

from ..config import TrainConfig
from ..instrument import loud

DEFAULT_RING = 64 << 20  # burn's 64 MB ring
DEFAULT_REFILL = 8 << 20  # refilled in 8 MB chunks


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
        raw = self._read(want)
        ids = torch.tensor(list(raw), dtype=torch.long).view(self.batch, self.seq_len + 1)
        return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()

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


def eval_window_bytes(cfg: TrainConfig) -> int:
    """`eval_batches × batch × seq_len` — the ONLY comparable window size."""
    return cfg.eval_batches * cfg.batch * cfg.seq_len


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
    stream's high bound stops one window short of it — the no-leak split.
    """
    tr = run.train
    if not tr.data:
        loud(
            "no data file: `data` is empty in the train config. Escape: set "
            '`data = "/path/to/corpus.bin"` (a flat .bin of bytes).'
        )
    if tr.eval_data:
        ev = ByteStream(
            tr.eval_data,
            tr.batch,
            tr.seq_len,
            ring_bytes=min(ring_bytes, max(1 << 20, eval_window_bytes(tr))),
        )
        train = ByteStream(tr.data, tr.batch, tr.seq_len, ring_bytes=ring_bytes)
        return train, ev
    ev_bytes = min(max(eval_window_bytes(tr), 1), 1 << 20) if tr.eval else 0
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
