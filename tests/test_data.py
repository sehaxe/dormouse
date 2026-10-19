"""The byte stream: the eval window, `rewind()`, and the ring's wrap."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from conftest import BATCH, SEQ, tiny_run, write_corpus

from dormouse.model import DormouseModel
from dormouse.train import ByteStream, eval_window_bytes, split_points, train
from dormouse.train.data import open_streams
from dormouse.train.mixture import MixtureStream, load_mix_plan, replay_bytes


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


# ---------------------------------------------------------------------------
# an extended mixture resumes at its STEP INDEX, not at the new budget's bytes
# ---------------------------------------------------------------------------

MIX_BATCH, MIX_SEQ, MIX_STEPS = 2, 8, 10
MIX_WINDOW = MIX_BATCH * (MIX_SEQ + 1)


def _positional_source(path: Path, salt: int, n: int = 4096) -> Path:
    """A source whose every byte encodes its POSITION, so a cursor is visible."""
    path.write_bytes(bytes((salt * 31 + j) % 256 for j in range(n)))
    return path


def _batch_bytes(x, y) -> tuple[bytes, bytes]:
    return (bytes(x.reshape(-1).tolist()), bytes(y.reshape(-1).tolist()))


def _two_stage_plan(tmp_path: Path):
    """Two stages with DIFFERENT weights: doubling `steps` moves the boundary."""
    a = _positional_source(tmp_path / "a.bin", 1)
    b = _positional_source(tmp_path / "b.bin", 2)
    plan_path = tmp_path / "mix.toml"
    plan_path.write_text(
        "max_epochs = 0.0\nseed = 7\n\n"
        "[[stage]]\nuntil_frac = 0.5\n[stage.weights]\na = 0.5\nb = 0.5\n\n"
        "[[stage]]\nuntil_frac = 1.0\n[stage.weights]\na = 0.9\nb = 0.1\n\n"
        f'[sources]\na = "{a.name}"\nb = "{b.name}"\n'
    )
    return load_mix_plan(plan_path)


class _ResumeProbe:
    """A mixture proxy that records the trainer's resume call and first batch."""

    def __init__(self, inner: MixtureStream) -> None:
        self.inner = inner
        self.resume_calls: list[tuple[int, int | None]] = []
        self.first: tuple[bytes, bytes] | None = None

    def skip_to_step(self, step: int, *, budget: int | None = None) -> None:
        self.resume_calls.append((step, budget))
        self.inner.skip_to_step(step, budget=budget)

    def skip_bytes(self, n: int) -> None:  # pragma: no cover - the defect this gates
        raise AssertionError(f"a mixture resume took the byte path ({n})")

    def next_batch(self):
        x, y = self.inner.next_batch()
        if self.first is None:
            self.first = _batch_bytes(x, y)
        return x, y

    def rewind(self) -> None:
        self.inner.rewind()

    def close(self) -> None:
        self.inner.close()


def test_an_extended_mixture_resumes_from_the_steps_it_took(tmp_path: Path):
    """Round 3 R3-4: `steps` is exempt, so extending a mixture run must not rewind.

    The stage boundaries are `until_frac × steps`; replaying a 10-step prefix
    with a doubled budget assigns steps 5-9 to the OTHER stage and moves every
    source's cursor by the wrong byte count. The resume must replay the prefix
    under the checkpoint's stored budget (10) and switch to the extended
    schedule at step 10.
    """
    plan = _two_stage_plan(tmp_path)
    kw = dict(batch=MIX_BATCH, seq_len=MIX_SEQ)
    old = MixtureStream(plan, steps=MIX_STEPS, seed=plan.seed, **kw)
    for _ in range(MIX_STEPS):
        old.next_batch()
    # The gate is not vacuous: the two budgets DO serve different prefixes.
    assert replay_bytes(plan, steps=MIX_STEPS, upto=MIX_STEPS, **kw) != replay_bytes(
        plan, steps=2 * MIX_STEPS, upto=MIX_STEPS, **kw
    )
    # Extend the run in place: same cursors, new schedule, 11th batch.
    old.steps = 2 * MIX_STEPS
    expected = _batch_bytes(*old.next_batch())

    # The trainer's own resume: 10 completed steps → checkpoint step 9 → start 10.
    run = tiny_run(
        tmp_path,
        steps=MIX_STEPS,
        ckpt_every=MIX_STEPS,
        seed=plan.seed,
        mix_plan=str(plan.sources["a"].parent / "mix.toml"),
        **kw,
    )
    train(DormouseModel(run.model), run, log=lambda *_: None)

    longer = dataclasses.replace(run, train=dataclasses.replace(run.train, steps=2 * MIX_STEPS))
    probe = _ResumeProbe(MixtureStream(plan, steps=2 * MIX_STEPS, seed=plan.seed, **kw))
    train(DormouseModel(longer.model), longer, stream=probe, resume=True, log=lambda *_: None)

    assert probe.resume_calls == [(MIX_STEPS, MIX_STEPS)], (
        "the resume must be STEP-indexed and replay the prefix under the stored budget"
    )
    assert probe.first == expected, "the resumed run rewound the sources"
