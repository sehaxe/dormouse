"""M1: `mix_plan` x `grad_accum` — the schedule and the epoch cap count MICRO-batches.

The stream draws one `next_batch()` per micro-batch, but the loop built it with
the OPTIMIZER step budget: the staged mix ran `grad_accum`x fast (stages at
optimizer steps 18/23 instead of 70/90 at steps=100, accum=4) and the epoch
cap under-counted bytes by the same factor (guard 3.52x while the loop read
14.55x). The fix is one budget in one unit; these gates pin the boundaries, the
guard and a mid-stage resume.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import BATCH, SEQ, tiny_run

from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train import loop as loopmod
from dormouse.train import train
from dormouse.train.mixture import (
    MixPlan,
    MixStage,
    MixtureStream,
    dumps_mix_plan,
    stage_bounds,
    stage_index,
)


def sources(tmp_path, n: int = 4096) -> dict[str, Path]:
    out = {}
    for i, name in enumerate(("web", "code")):
        p = tmp_path / f"{name}.bin"
        p.write_bytes(bytes([ord("A") + i]) * n)
        out[name] = p
    return out


def plan_file(tmp_path, *, max_epochs: float = 4.0, sizes: int = 4096):
    """The 70/80/62-ish three-stage shape from the research doc, on tiny sources."""
    src = sources(tmp_path, sizes)
    plan = MixPlan(
        (
            MixStage(0.70, {"web": 0.90, "code": 0.10}),
            MixStage(0.90, {"web": 0.80, "code": 0.20}),
            MixStage(1.00, {"web": 0.62, "code": 0.38}),
        ),
        src,
        max_epochs=max_epochs,
    )
    path = tmp_path / "mix_plan.toml"
    path.write_text(dumps_mix_plan(plan))
    return plan, path


def test_stage_bounds_land_at_optimizer_steps_70_and_90(tmp_path):
    plan, _ = plan_file(tmp_path)
    steps, accum = 100, 4
    stream = MixtureStream(plan, BATCH, SEQ, steps=steps * accum)
    assert stage_bounds(plan, stream.steps) == (0, 280, 360, 400)
    transitions, stage = [], stream.stage
    for micro in range(steps * accum):
        stream.next_batch()
        if stream.stage != stage:
            transitions.append(((micro + 1) // accum, stream.stage))
            stage = stream.stage
    assert transitions == [(70, 1), (90, 2)], transitions
    # The same indices read as one stage lookup: 69 -> S1, 70..89 -> S2, 90+ -> S3.
    assert [stage_index(plan, opt * accum, stream.steps) for opt in (69, 70, 89, 90)] == [
        0,
        1,
        1,
        2,
    ]


def test_the_trainer_passes_the_micro_step_budget(tmp_path, monkeypatch):
    _plan, path = plan_file(tmp_path)
    seen: dict = {}
    real = loopmod.open_mixture

    def spy(plan_path, **kw):
        seen.update(kw)
        stream = real(plan_path, **kw)
        seen["stream_steps"] = stream.steps
        return stream

    monkeypatch.setattr(loopmod, "open_mixture", spy)
    run = tiny_run(tmp_path, steps=3, grad_accum=4, mix_plan=str(path), ckpt_every=1000)
    train(DormouseModel(run.model), run, log=lambda *_: None)
    assert seen["steps"] == 12, "the mixer must get steps x grad_accum, not steps"
    assert seen["stream_steps"] == 12


def test_the_epoch_cap_sees_the_micro_batches(tmp_path):
    """steps=4, accum=4, one 900 B source: the loop reads 2.31x, the old count said 0.58x."""
    src = tmp_path / "web.bin"
    src.write_bytes(b"A" * 900)
    plan = MixPlan((MixStage(1.0, {"web": 1.0}),), {"web": src}, max_epochs=1.0)
    path = tmp_path / "cap.toml"
    path.write_text(dumps_mix_plan(plan))
    run = tiny_run(
        tmp_path, steps=4, grad_accum=4, mix_plan=str(path), max_epochs=1.0, ckpt_every=1000
    )
    with pytest.raises(LoudError, match="epoch cap"):
        train(DormouseModel(run.model), run, log=lambda *_: None)


def test_a_mid_stage_resume_reproduces_the_stage(tmp_path):
    src = sources(tmp_path)
    plan = MixPlan(
        (MixStage(0.5, {"web": 1.0}), MixStage(1.0, {"code": 1.0})),
        src,
    )
    steps = 40
    assert stage_bounds(plan, steps) == (0, 20, 40)
    whole = MixtureStream(plan, BATCH, SEQ, steps=steps)
    for _ in range(26):
        whole.next_batch()
    assert whole.stage == stage_index(plan, 26, steps) == 1, "the fixture must resume mid-stage"

    resumed = MixtureStream(plan, BATCH, SEQ, steps=steps)
    resumed.skip_bytes(26 * BATCH * (SEQ + 1))
    assert resumed.step == 26 and resumed.stage == whole.stage == 1
    for _ in range(steps - 26):
        a, b = whole.next_batch(), resumed.next_batch()
        assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
