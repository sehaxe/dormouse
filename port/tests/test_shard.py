"""The shard directory mode: the seeded order, the epoch, exact resume, refusals.

The mount's corpora are directories of ~6.3 MB `*.txt` shards (103 GB for
`web`, 17 383 files), so `ShardStream` reads the directory itself instead of
duplicating 227 GB into a concatenated `.bin`. Three claims carry the mode, and
all three are CPU/tiny here:

1. the file order is a PURE function of the seed (`shard_order`, pinned by a
   golden — a resume that read a different order would read different bytes);
2. an epoch is the CONCATENATION of the shuffled files: every shard exactly
   once, in that order, wrapping into the next epoch (the order is what
   `plan_txt_streamer.md` was missing — download order means consecutive files
   are one crawl/period, and the boundary showed up as a 1.46 → 8.69 loss spike);
3. `skip_bytes` lands on the byte an unbroken run would have read — mid-shard,
   across a shard boundary, and across epochs — because the sequence is periodic
   in `sum(shard sizes)`. No sidecar file is needed for that, and the last gate
   here is the trainer's own resume path over a shard source.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import BATCH, SEQ, tiny_run

from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train import ShardStream, train
from dormouse.train.data import ByteStream, open_source, open_streams, shard_files, shard_order
from dormouse.train.mixture import (
    MixPlan,
    MixStage,
    MixtureStream,
    check_epoch_cap,
    source_size,
)

SEED = 1
SHARDS, SHARD_BYTES = 6, 64
TOTAL = SHARDS * SHARD_BYTES


def shard_dir(tmp_path: Path, *, name: str = "shards", n: int = SHARDS, size: int = SHARD_BYTES):
    """`n` shards, each `size` bytes of ONE identifying byte (`b'B'`, `b'C'`, …)."""
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"s_{i:05d}.txt").write_bytes(bytes([ord("B") + i]) * size)
    return d


def epoch_bytes(d: Path, seed: int = SEED) -> bytes:
    """One epoch as bytes: the shuffled files concatenated."""
    return b"".join(p.read_bytes() for p in shard_order(shard_files(d), seed))


def stream_bytes(stream, n: int) -> bytes:
    """The raw bytes of `n` batch-1 windows — `x[0]` plus the targets, i.e. what was read.

    Only valid on a `batch=1` stream (a 2-row window interleaves two reads), which
    is what every absolute-comparison test here builds.
    """
    out = bytearray()
    for _ in range(n):
        x, y = stream.next_batch()
        assert x.shape[0] == 1, "stream_bytes reconstructs a batch-1 stream's bytes"
        out += bytes([int(x[0, 0])]) + bytes(y.reshape(-1).tolist())
    return bytes(out)


def read(stream, n: int) -> bytes:
    """The input bytes of `n` windows, for stream-to-stream comparisons."""
    out = bytearray()
    for _ in range(n):
        x, _y = stream.next_batch()
        out += bytes(x.reshape(-1).tolist())
    return bytes(out)


def one_source_plan(src: Path, **kw) -> MixPlan:
    return MixPlan((MixStage(1.0, {"web": 1.0}),), {"web": src}, **kw)


# --- the seeded file order ---------------------------------------------------


def test_a_directory_source_streams_in_the_seeded_order(tmp_path):
    """One epoch out of a batch-1 stream is the shuffled files, byte for byte."""
    d = shard_dir(tmp_path)
    order = shard_order(shard_files(d), SEED)
    stream = ShardStream(d, 1, SHARD_BYTES - 1, seed=SEED, ring_bytes=1 << 12, refill_bytes=8)
    assert stream_bytes(stream, SHARDS) == epoch_bytes(d)
    assert order[0].name != sorted(p.name for p in shard_files(d))[0], "it must shuffle"


def test_the_shuffle_is_pinned_and_a_pure_function_of_the_seed(tmp_path):
    """Same seed → same order in any process; different seed → a different order.

    The literal is the golden: the permutation comes from `random.Random`'s
    stream, and a resume must not move it (a streamed RNG would).
    """
    d = shard_dir(tmp_path)
    files = shard_files(d)
    assert [p.name for p in shard_order(files, 1)] == [
        "s_00002.txt",
        "s_00003.txt",
        "s_00005.txt",
        "s_00000.txt",
        "s_00004.txt",
        "s_00001.txt",
    ]
    assert shard_order(files, 7) == shard_order(files, 7)
    assert shard_order(files, 7) != shard_order(files, 8)
    assert read(ShardStream(d, 1, 1, seed=7), 12) == read(ShardStream(d, 1, 1, seed=7), 12)
    assert read(ShardStream(d, 1, 1, seed=7), 12) != read(ShardStream(d, 1, 1, seed=8), 12)


def test_an_epoch_covers_every_shard_exactly_once_and_wraps(tmp_path):
    """One epoch = every shard once, in order; the next epoch repeats it."""
    d = shard_dir(tmp_path)
    stream = ShardStream(d, 1, SHARD_BYTES - 1, seed=SEED)
    assert stream_bytes(stream, SHARDS) == epoch_bytes(d)
    assert stream_bytes(stream, SHARDS) == epoch_bytes(d)  # epoch 2, same order


def test_a_window_straddling_two_shards_is_contiguous(tmp_path):
    """The epoch is a concatenation, not a list of files: 6-byte shards, 4-byte windows."""
    d = shard_dir(tmp_path, n=4, size=6)
    stream = ShardStream(d, 1, 3, seed=SEED, refill_bytes=4)
    assert stream_bytes(stream, 6) == epoch_bytes(d)


def test_rewind_returns_exactly_the_same_bytes(tmp_path):
    stream = ShardStream(shard_dir(tmp_path), BATCH, SEQ, seed=SEED)
    first = read(stream, 4)
    stream.rewind()
    assert stream.position == (0, 0)
    assert read(stream, 4) == first


def test_position_names_the_next_byte_out(tmp_path):
    """`(file index in the shuffled order, offset)` — consumed bytes, not the handle's."""
    stream = ShardStream(shard_dir(tmp_path), 1, 1, seed=SEED, refill_bytes=4)
    assert stream.position == (0, 0)
    stream_bytes(stream, SHARD_BYTES // 2)  # 64 B = exactly the first shard of the order
    assert stream.position == (1, 0)
    stream.next_batch()  # 2 bytes into shard 1 of the order
    assert stream.position == (1, 2)
    stream.skip_bytes(SHARD_BYTES + 6)
    assert stream.position == (1, 6)
    stream.skip_bytes(2 * TOTAL + 3)  # two epochs later: the modulo coordinate
    assert stream.position == (0, 3)


def test_non_txt_files_are_not_read(tmp_path):
    """The mount's `math/` also holds `.jsonl` dumps: they are not shards."""
    d = shard_dir(tmp_path)
    (d / "dump.jsonl").write_bytes(b"{" + b"x" * 4096 + b"}")
    (d / "notes.md").write_bytes(b"not a shard")
    assert [p.suffix for p in shard_files(d)] == [".txt"] * SHARDS
    assert stream_bytes(ShardStream(d, 1, SHARD_BYTES - 1, seed=SEED), SHARDS) == epoch_bytes(d)


# --- resume ------------------------------------------------------------------


def test_skip_bytes_lands_on_the_byte_an_unbroken_run_would_read(tmp_path):
    """The resume property: mid-shard, across a boundary, and across epochs."""
    d = shard_dir(tmp_path)
    window = BATCH * (SEQ + 1)
    for skip in (window, 2 * window, TOTAL + window, 2 * TOTAL + 2 * window):
        whole = ShardStream(d, BATCH, SEQ, seed=SEED)
        read(whole, (skip % TOTAL) // window)  # walk to the byte the skip names
        resumed = ShardStream(d, BATCH, SEQ, seed=SEED)
        resumed.skip_bytes(skip)
        assert resumed.position == whole.position, f"skip {skip} landed elsewhere"
        assert read(resumed, 5) == read(whole, 5), f"skip {skip} read different bytes"
    # An offset inside a window is exact too — the resume math never rounds.
    a, b = ShardStream(d, BATCH, SEQ, seed=SEED), ShardStream(d, BATCH, SEQ, seed=SEED)
    a.skip_bytes(window + 7)
    b.skip_bytes(137)
    assert a.position == b.position and read(a, 5) == read(b, 5)


def test_skip_bytes_is_the_epoch_modulo(tmp_path):
    """`n` and `n % total` name the same byte: the sequence is periodic in the epoch."""
    d = shard_dir(tmp_path)
    a, b = ShardStream(d, BATCH, SEQ, seed=SEED), ShardStream(d, BATCH, SEQ, seed=SEED)
    a.skip_bytes(TOTAL * 3 + 17)
    b.skip_bytes(17)
    assert a.position == b.position
    assert read(a, 4) == read(b, 4)
    c = ShardStream(d, BATCH, SEQ, seed=SEED)
    c.skip_bytes(TOTAL)
    assert c.position == (0, 0)
    assert read(c, 2) == read(ShardStream(d, BATCH, SEQ, seed=SEED), 2)


def test_a_resumed_mixture_reads_the_bytes_an_unbroken_one_would(tmp_path):
    """The mixture's own resume: one source, one seed, bit-identical windows."""
    plan = one_source_plan(shard_dir(tmp_path))
    whole = MixtureStream(plan, BATCH, SEQ, steps=40, seed=SEED)
    head, tail = read(whole, 10), read(whole, 30)
    resumed = MixtureStream(plan, BATCH, SEQ, steps=40, seed=SEED)
    resumed.skip_bytes(10 * BATCH * (SEQ + 1))
    assert resumed.step == 10
    assert read(resumed, 30) == tail
    assert len(head) == 10 * BATCH * SEQ


class _Recorder:
    """A stream proxy that records what the trainer read and fast-forwarded by."""

    def __init__(self, inner) -> None:
        self.inner, self.seen, self.skipped = inner, [], []

    def next_batch(self):
        x, y = self.inner.next_batch()
        self.seen.append((bytes(x.reshape(-1).tolist()), bytes(y.reshape(-1).tolist())))
        return x, y

    def skip_bytes(self, n: int) -> None:
        self.skipped.append(n)
        self.inner.skip_bytes(n)

    def rewind(self) -> None:
        self.inner.rewind()


def test_a_resumed_run_reads_the_same_batches(tmp_path):
    """The trainer's OWN resume path (`start x (seq_len+1) x batch`) over a shard source.

    The gate a sidecar would have had to buy: an unbroken 4-step run and a
    2-step run resumed to 4 hand out the same batches, batch for batch.
    """
    src = shard_dir(tmp_path)
    plan_path = tmp_path / "mix_plan.toml"
    plan_path.write_text(
        "max_epochs = 0.0\nseed = 1\n\n[[stage]]\nuntil_frac = 1.0\n[stage.weights]\n"
        f'web = 1.0\n\n[sources]\nweb = "{src}"\n'
    )
    kwargs = dict(batch=BATCH, seq_len=SEQ, log_every=10, ckpt_every=2)
    first = tiny_run(tmp_path, steps=2, mix_plan=str(plan_path), **kwargs)
    train(DormouseModel(first.model), first, log=lambda *_: None)

    reference = _batches(MixtureStream(one_source_plan(src), BATCH, SEQ, steps=4))
    second = tiny_run(tmp_path, steps=4, mix_plan=str(plan_path), **kwargs)
    stream = _Recorder(MixtureStream(one_source_plan(src), BATCH, SEQ, steps=4))
    train(DormouseModel(second.model), second, stream=stream, resume=True, log=lambda *_: None)
    assert stream.skipped == [2 * (SEQ + 1) * BATCH], "the fast-forward is step x window"
    assert stream.seen == reference[2:], "a resumed run must not re-read the head"


def _batches(stream, n: int = 4) -> list[tuple[bytes, bytes]]:
    out = []
    for _ in range(n):
        x, y = stream.next_batch()
        out.append((bytes(x.reshape(-1).tolist()), bytes(y.reshape(-1).tolist())))
    return out


# --- coexistence with the flat mode -----------------------------------------


def test_a_plan_with_a_bin_and_a_directory_source_steps(tmp_path):
    """One `.bin` source + one shard directory, staged: both are read, in their stage."""
    flat = tmp_path / "flat.bin"
    flat.write_bytes(b"A" * 4096)
    plan = MixPlan(
        (MixStage(0.5, {"web": 1.0}), MixStage(1.0, {"math": 1.0})),
        {"web": flat, "math": shard_dir(tmp_path)},
    )
    stream = MixtureStream(plan, BATCH, SEQ, steps=20, seed=SEED)
    assert isinstance(stream._streams["web"], ByteStream)
    assert isinstance(stream._streams["math"], ShardStream)
    first, second = read(stream, 10), read(stream, 10)
    assert set(first) == {ord("A")}, "stage 1 reads the flat source"
    assert ord("A") not in set(second), "stage 2 reads the shards"
    assert all(ord("B") <= b <= ord("B") + SHARDS - 1 for b in second)


def test_open_source_picks_the_mode_by_shape(tmp_path):
    flat = tmp_path / "flat.bin"
    flat.write_bytes(b"A" * 4096)
    assert isinstance(open_source(flat, BATCH, SEQ), ByteStream)
    assert isinstance(open_source(shard_dir(tmp_path), BATCH, SEQ), ShardStream)


def test_two_sources_do_not_share_a_file_order(tmp_path):
    """The shard seed mixes the run seed with the source NAME (`_stream_seed`)."""
    a, b = shard_dir(tmp_path, name="a"), shard_dir(tmp_path, name="b")
    plan = MixPlan((MixStage(1.0, {"web": 0.5, "math": 0.5}),), {"web": a, "math": b})
    stream = MixtureStream(plan, BATCH, SEQ, steps=10, seed=SEED)
    assert stream._streams["web"].seed != stream._streams["math"].seed


def test_a_directory_source_counts_toward_the_epoch_cap(tmp_path):
    """`source_size` sums the shards, so the cap sees what the stream will read."""
    src = shard_dir(tmp_path)
    assert source_size("web", src) == TOTAL
    plan = one_source_plan(src)
    window = BATCH * (SEQ + 1)
    epochs = check_epoch_cap(plan, steps=3, batch=BATCH, seq_len=SEQ)["web"]
    assert epochs == pytest.approx(3 * window / TOTAL)
    with pytest.raises(LoudError, match="epoch cap"):
        check_epoch_cap(plan, steps=10 * TOTAL // window, batch=BATCH, seq_len=SEQ)


# --- refusals ----------------------------------------------------------------


def test_an_empty_directory_is_loud(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(LoudError, match=r"holds no `\*\.txt` shards"):
        ShardStream(d, BATCH, SEQ)
    with pytest.raises(LoudError, match="fetch them"):
        shard_files(d)


def test_a_directory_with_only_other_files_is_loud(tmp_path):
    """A wrong path must not stream 0 bytes (v21 trained on constant filler)."""
    d = tmp_path / "junk"
    d.mkdir()
    (d / "corpus.bin").write_bytes(b"A" * 1024)
    with pytest.raises(LoudError, match=r"holds no `\*\.txt` shards"):
        shard_files(d)


def test_a_zero_byte_shard_is_loud(tmp_path):
    d = shard_dir(tmp_path)
    (d / "s_99999.txt").write_bytes(b"")
    with pytest.raises(LoudError, match="zero-byte shard"):
        ShardStream(d, BATCH, SEQ)


def test_a_file_where_a_directory_is_required_is_loud(tmp_path):
    flat = tmp_path / "flat.bin"
    flat.write_bytes(b"A" * 1024)
    with pytest.raises(LoudError, match="is not a directory"):
        shard_files(flat)


def test_a_directory_where_a_file_is_required_is_loud(tmp_path):
    """The old code path must refuse, not `stat()` an inode and call it 4096 bytes."""
    d = shard_dir(tmp_path)
    with pytest.raises(LoudError, match="is a directory"):
        ByteStream(d, BATCH, SEQ)
    with pytest.raises(LoudError, match="mix_plan"):
        open_streams(tiny_run(tmp_path, data=str(d)))


def test_a_directory_too_small_for_one_batch_is_loud(tmp_path):
    with pytest.raises(LoudError, match="less than one batch"):
        ShardStream(shard_dir(tmp_path, n=2, size=8), BATCH, SEQ)
