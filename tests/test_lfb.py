"""LFB's routing controller is MODEL state: a checkpoint must carry it.

Audit 2026-10-19 M8: `lfb_counts` persisted while `lfb_bias` did not, so a
resume restored the integrator and reset the bias it had already moved — a
different routing controller under the same step count.
"""

from __future__ import annotations

import dataclasses

import torch
from conftest import tiny_cfg, tiny_run

from dormouse.model import DormouseModel
from dormouse.train import train
from dormouse.train.loop import ckpt_paths, lfb_path, load_model_half, load_model_state


def _moe(model: DormouseModel):
    return next(m for m in model.modules() if hasattr(m, "lfb_bias"))


def test_a_fresh_model_starts_with_a_zero_bias():
    """Persistence must not be an accident of a shared/global buffer."""
    model = DormouseModel(tiny_cfg(moe_lb_coef=0.05))
    moe = _moe(model)
    assert torch.equal(moe.lfb_bias, torch.zeros_like(moe.lfb_bias))
    assert torch.equal(moe.lfb_counts, torch.zeros_like(moe.lfb_counts))


def test_a_resume_restores_the_lfb_controller_state(tmp_path):
    """5 routed steps → save → reload: the bias comes back identical."""
    torch.manual_seed(0)
    run = dataclasses.replace(
        tiny_run(tmp_path, steps=5, log_every=1, ckpt_every=5, ckpt_name="lfb"),
        model=tiny_cfg(moe_lb_coef=0.05),
    )
    model = DormouseModel(run.model)
    train(model, run, log=lambda *_: None)
    moe = _moe(model)
    moved = moe.lfb_bias.detach().clone()
    counts = moe.lfb_counts.detach().clone()
    assert moved.abs().sum() > 0, "five routed steps must move the controller off zero"

    weights, _snapshot = ckpt_paths(run.train)
    assert weights.is_file() and lfb_path(run.train).is_file()
    fresh = DormouseModel(run.model)
    assert _moe(fresh).lfb_bias.abs().sum() == 0, "the reload must not share the live buffer"
    load_model_state(fresh, run.train, load_model_half(weights))
    assert torch.equal(_moe(fresh).lfb_bias, moved)
    assert torch.equal(_moe(fresh).lfb_counts, counts)
