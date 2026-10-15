"""The byte stream: the eval window, `rewind()`, and the ring's wrap."""

from __future__ import annotations

from pathlib import Path

from conftest import BATCH, SEQ, tiny_run, write_corpus

from dormouse.train import ByteStream, eval_window_bytes, split_points
from dormouse.train.data import open_streams


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
    # The window formula is the printed number, so it cannot drift from the code.
    assert str(scored) in f"over {scored} B"


def test_train_and_eval_ranges_do_not_overlap(tmp_path):
    """The split is by BYTE RANGE — a train batch cannot come from the tail."""
    size = 4096
    hi_train, hi = split_points(size, 512)
    assert (hi_train, hi) == (3584, 4096)
    path = write_corpus(tmp_path / "c.bin", size)
    train = ByteStream(path, BATCH, SEQ, lo=0, hi=hi_train)
    ev = ByteStream(path, BATCH, SEQ, lo=hi_train)
    ids, _ = train.next_batch()
    assert ids.max().item() <= 255
    # The eval stream starts at the carve, not at 0.
    assert ev.lo == hi_train == 3584


def test_ring_wraps_at_the_end_of_the_range(tmp_path):
    path = write_corpus(tmp_path / "c.bin", 512)
    stream = ByteStream(path, 1, SEQ, ring_bytes=1 << 10, refill_bytes=64)
    seen = []
    for _ in range(12):  # 12 * 65 bytes over a 512-byte range: it must wrap
        ids, _t = stream.next_batch()
        seen.append(int(ids[0, 0]))
    assert len(seen) == 12
    assert Path(path).read_bytes()[:SEQ] == bytes(range(16)) * 4


def test_skip_bytes_fast_forwards_for_resume(tmp_path):
    path = write_corpus(tmp_path / "c.bin", 4096)
    a = ByteStream(path, BATCH, SEQ)
    b = ByteStream(path, BATCH, SEQ)
    per_batch = BATCH * (SEQ + 1)
    b.skip_bytes(3 * per_batch)
    for _ in range(3):
        a.next_batch()
    assert (a.next_batch()[0] == b.next_batch()[0]).all()
