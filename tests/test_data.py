"""The byte stream: the eval window, `rewind()`, the split, and the ring's wrap."""

from __future__ import annotations

import re
from pathlib import Path

from conftest import BATCH, SEQ, tiny_run, write_corpus

from dormouse.model import DormouseModel
from dormouse.train import ByteStream, eval_window_bytes, split_points, train
from dormouse.train.data import open_streams


def _aperiodic(size: int) -> bytes:
    """`size` bytes with no period shorter than the file: a shifted read shows.

    The shared `write_corpus` repeats every 16 bytes, so a 16-byte skip at the
    seam is invisible in its bytes — useless for a POSITION gate.
    """
    return bytes(((i * 37) ^ (i >> 8)) & 0xFF for i in range(size))


def test_next_batch_shape_and_shift(tmp_path):
    path = write_corpus(tmp_path / "c.bin", 4096)
    stream = ByteStream(path, BATCH, SEQ)
    ids, targets = stream.next_batch()
    assert ids.shape == (BATCH, SEQ) == targets.shape
    # Next-byte LM: the target at position q is the byte after the input's.
    assert (targets[:, :-1] == ids[:, 1:]).all()


def test_rewind_returns_exactly_the_same_bytes(tmp_path):
    """(c) `rewind()` is what makes two evals comparable."""
    path = write_corpus(tmp_path / "c.bin", 4096)
    stream = ByteStream(path, BATCH, SEQ, ring_bytes=1 << 12, refill_bytes=1 << 9)
    first = [stream.next_batch() for _ in range(3)]
    stream.rewind()
    again = [stream.next_batch() for _ in range(3)]
    for (a, b), (c, d) in zip(first, again, strict=True):
        assert (a == c).all() and (b == d).all()
    # And advancing past the ring's capacity must not change what rewind means.
    stream.rewind()
    assert (stream.next_batch()[0] == first[0][0]).all()


def test_eval_window_is_batches_times_batch_times_seq(tmp_path):
    """(c) The window is `eval_batches x batch x seq_len` BYTES, not batches."""
    run = tiny_run(tmp_path, eval=True, eval_batches=3)
    assert eval_window_bytes(run.train) == 3 * BATCH * SEQ
    _train, ev = open_streams(run)
    assert ev is not None
    scored = 0
    for _ in range(run.train.eval_batches):
        ids, _t = ev.next_batch()
        scored += ids.numel()
    assert scored == eval_window_bytes(run.train) == 384
    # The number is the TRAINER's, not a local copy: two logged evals of a real
    # run, `over N B` parsed off the line, must print this window.
    run = tiny_run(tmp_path, eval=True, eval_batches=3, steps=2, eval_every=1)
    lines: list[str] = []
    train(DormouseModel(run.model), run, resume=False, log=lines.append)
    evals = [line for line in lines if line.startswith("eval step=")]
    assert len(evals) == 2, lines
    printed = [int(n) for line in evals for n in re.findall(r"over (\d+) B\b", line)]
    assert printed == [eval_window_bytes(run.train)] * 2 == [384, 384]


def test_train_and_eval_ranges_do_not_overlap(tmp_path):
    """The split is by BYTE RANGE — a train batch cannot come from the tail.

    The corpus is aperiodic over the file, so a window's OWN bytes name its
    position: every window the train stream returns is a slice of
    `[lo, hi_train)` at the ring's arithmetic, and no byte read has a position
    inside the eval tail. The carve's arithmetic is pinned on the side
    (`hi_train == size - eval_bytes`), which is the half a leak cannot fake.
    """
    run = tiny_run(tmp_path, eval=True, eval_batches=8)
    eval_bytes = eval_window_bytes(run.train)
    assert eval_bytes == 8 * BATCH * SEQ == 1024
    path = Path(run.train.data)
    size = 4096
    raw = _aperiodic(size)
    path.write_bytes(raw)
    training, ev = open_streams(run)
    assert (training.lo, training.hi) == (0, split_points(size, eval_bytes)[0])
    # The carve: train + eval are the file, and the train range lost exactly the
    # eval bytes — not one window, not one byte more.
    span = training.hi - training.lo
    assert span == size - eval_bytes
    assert (ev.lo, ev.hi) == (training.hi, size)
    assert ev.hi - ev.lo == eval_bytes
    assert span + eval_bytes == size
    # Windows that do not straddle the end: each is ONE slice of the file, so
    # `find` names its position exactly (the corpus has no repeated 130-byte
    # window).
    window = BATCH * (SEQ + 1)
    got = bytearray()
    for _ in range(span // window):
        ids, targets = training.next_batch()
        for row in range(BATCH):
            got += bytes(ids[row]) + bytes([int(targets[row, -1])])
        at = raw.find(bytes(got[-window:]))
        assert at != -1, "the window is not in the file at all"
        assert not (ev.lo <= at < ev.hi), f"a train window at {at} reads the eval tail"
        assert at == len(got) - window, "sequential: no byte skipped or repeated"
    # ... and the whole read is the file's train slice, byte for byte.
    assert bytes(got) == raw[: len(got)]


def test_ring_wraps_at_the_end_of_the_range(tmp_path):
    """The wrap is BYTE-EXACT: `hi - 1` is followed by `lo`, nothing skipped or repeated.

    Three full cycles of the range: the stream must hand back the range's bytes
    three times over, and each window's head must be the byte AT that position.
    (This test used to assert `len(seen) == 12` — true for any byte sequence.)
    The corpus is aperiodic, so a skipped or repeated byte at the seam cannot
    hide behind the 16-byte cycle `write_corpus` uses.
    """
    path = tmp_path / "c.bin"
    raw = _aperiodic(512)
    path.write_bytes(raw)
    stream = ByteStream(path, 1, SEQ, ring_bytes=1 << 10, refill_bytes=64)
    span = stream.hi - stream.lo
    assert span == stream.size == len(raw)
    window = SEQ + 1  # one batch is `SEQ + 1` consecutive bytes, target included
    cycles = 3
    got = bytearray()
    heads: list[int] = []
    for _ in range(-(-cycles * span // window)):  # enough windows for 3 cycles
        ids, targets = stream.next_batch()
        heads.append(int(ids[0, 0]))
        got += bytes(ids[0]) + bytes([int(targets[0, -1])])
    # The heads are the byte AT each position, wrapped at `hi`: a position gate,
    # not a loop counter.
    assert heads == [raw[(k * window) % span] for k in range(len(heads))]
    # The seam, stated twice: byte `hi - 1` then byte `lo` (the middle and the
    # tail of `got` come from windows that straddle it).
    assert got[span - 1] == raw[span - 1] and got[span] == raw[0]
    # ... and the whole read is the range repeated, with nothing dropped or
    # duplicated: a skip at the seam would misalign every cycle after it.
    assert bytes(got[: cycles * span]) == raw * cycles


def test_skip_bytes_fast_forwards_for_resume(tmp_path):
    path = write_corpus(tmp_path / "c.bin", 4096)
    a = ByteStream(path, BATCH, SEQ)
    b = ByteStream(path, BATCH, SEQ)
    per_batch = BATCH * (SEQ + 1)
    b.skip_bytes(3 * per_batch)
    for _ in range(3):
        a.next_batch()
    assert (a.next_batch()[0] == b.next_batch()[0]).all()
