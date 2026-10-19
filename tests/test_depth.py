"""The ADAPTIVE DEPTH arm: a pure `(seed, step)` depth schedule and its curve.

The arm is rand-depth training: per TRAIN step the loop runs `T` iterations,
`T` sampled in `1..=max_iter` as a PURE function of `(seed, step)` — never a
global RNG draw, because burn's RNG is unseeded and a resume must continue the
same objective (ADR-0021). The loss divides by the depth ACTUALLY run, so a
`T=1` step and a `T=max_iter` step are one objective at two depths rather than
two differently-scaled numbers; and the eval line carries the
depth-efficiency curve the arm is judged by.
"""

from __future__ import annotations

import math

import torch
from conftest import BATCH, SEQ, tiny_cfg, tiny_model, tiny_run

from dormouse.instrument import Counters
from dormouse.model import DormouseModel
from dormouse.train import open_streams, train
from dormouse.train.loop import _sample_depth, evaluate


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    return ids, (ids + 1) % 251


def _h0() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, SEQ, tiny_cfg().d_model)


def test_sample_depth_is_pure_and_covers_every_depth():
    """(a) Same `(seed, step)` → same `T`; 64 steps touch all of 1..=4."""
    assert _sample_depth(1, 7, 4) == _sample_depth(1, 7, 4)
    assert _sample_depth(9, 3, 4) == _sample_depth(9, 3, 4)
    depths = [_sample_depth(1, step, 4) for step in range(64)]
    assert set(depths) == {1, 2, 3, 4}, sorted(set(depths))
    assert all(1 <= d <= 4 for d in depths)
    # A second seed is a DIFFERENT schedule, not the same one offset.
    assert any(_sample_depth(2, s, 4) != _sample_depth(1, s, 4) for s in range(64))


def test_depth_two_equals_the_first_two_iterations_of_depth_four():
    """(b) The T-iteration run is the first T iterations of the full loop.

    The same weights on both models (the extra `iter_embed` rows dropped, they
    are the only shape a shorter `max_iter` changes), so the claim is exact.
    """
    m4 = tiny_model(max_iter=4)
    m2 = tiny_model(max_iter=2)
    sd = dict(m4.state_dict())
    sd["loop.iter_embed"] = sd["loop.iter_embed"][:2]
    m2.load_state_dict(sd)

    h0 = _h0()
    out2, _, _ = m2.loop(h0, None, None, None, counters=Counters())
    out4, _, _ = m4.loop(h0, None, None, None, counters=Counters(), depth=2)
    assert torch.allclose(out2, out4, atol=1e-6)


def test_depth_counter_reads_the_depth_actually_run():
    """(c) `depth` is the run depth, not the configured `max_iter`."""
    model = tiny_model(max_iter=4)
    h0 = _h0()
    short, full = Counters(), Counters()
    model.loop(h0, None, None, None, counters=short, depth=2)
    model.loop(h0, None, None, None, counters=full)
    assert short["depth"] == short["iter"] == 2
    assert full["depth"] == full["iter"] == 4


def test_depth_none_is_exactly_the_max_iter_path():
    """(f) The default is byte-identical, and depth really changes the loss."""
    model = tiny_model(max_iter=4)
    x, y = _batch()
    _, loss_none, _ = model(x, y)
    _, loss_max, _ = model(x, y, depth=4)
    _, loss_one, _ = model(x, y, depth=1)
    assert torch.equal(loss_none, loss_max)
    assert not torch.equal(loss_none, loss_one)


def test_rand_depth_training_follows_the_pure_schedule(tmp_path):
    """(g) A run's total depth is exactly the pure schedule — resume-safe.

    Two trainers built from the same `(seed, step)` would run the same depths:
    that is what makes a resume legal (ADR-0021).
    """
    run = tiny_run(tmp_path, steps=8, log_every=8, rand_depth=True)
    report = train(DormouseModel(run.model), run, log=lambda *_: None)
    want = sum(_sample_depth(run.train.seed, s, run.model.max_iter) for s in range(8))
    assert report["counters"]["iter"] == want
    assert report["counters"]["depth"] == want
    assert report["nan_skipped"] == 0


def test_thirty_rand_depth_steps_decrease_the_loss(tmp_path):
    """(d) The arm trains: 30 sampled-depth steps, finite, loss down, no NaN."""
    run = tiny_run(tmp_path, steps=30, log_every=10, rand_depth=True)
    lines: list[str] = []
    report = train(DormouseModel(run.model), run, log=lines.append)
    losses = [float(ln.split("loss=")[1].split()[0]) for ln in lines if "loss=" in ln]
    assert len(losses) >= 3, lines
    assert all(math.isfinite(x) for x in losses)
    assert report["nan_skipped"] == 0
    assert losses[-1] < losses[0]


def test_eval_depths_returns_the_curve(tmp_path):
    """(e) One pass per depth, all finite; max depth is best within a loose bound."""
    run = tiny_run(tmp_path, steps=1, eval=True, eval_batches=3)
    torch.manual_seed(0)
    model = DormouseModel(run.model)
    _train_stream, eval_stream = open_streams(run)
    assert eval_stream is not None
    curve = evaluate(model, eval_stream, 3, depths=True)
    assert sorted(curve) == list(range(1, run.model.max_iter + 1))
    ces = [curve[d][0] for d in sorted(curve)]
    assert all(math.isfinite(c) for c in ces)
    assert all(curve[d][1] == 3 * BATCH * SEQ for d in curve)
    # Monotone-ish, NOT strictly: the deepest run is the smallest CE within a
    # loose bound. That is an empirical claim about an untrained loop, and the
    # eval line is the authority (§1.4), so the test only fences it.
    assert ces[-1] <= min(ces) + 0.25, ces


def test_eval_line_carries_the_curve_when_asked(tmp_path):
    """The plan's `--eval-depths` deliverable prints on the train eval line."""
    run = tiny_run(
        tmp_path, steps=2, log_every=1, eval=True, eval_every=2, eval_batches=2, eval_depths=True
    )
    lines: list[str] = []
    train(DormouseModel(run.model), run, log=lines.append)
    evals = [ln for ln in lines if ln.startswith("eval ")]
    assert evals, lines
    assert "depths=1:" in evals[0] and ",2:" in evals[0], evals[0]
