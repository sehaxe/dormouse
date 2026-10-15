"""Checkpoints: a resume continues, and a changed config refuses (ADR-0021)."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from conftest import tiny_run

from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train import load_ckpt, train
from dormouse.train.loop import ckpt_paths
from dormouse.train.optim import MixOptimizer


def test_save_resume_same_config_continues(tmp_path):
    """(d) save → load with an identical resolved config passes, steps continue."""
    first = tiny_run(tmp_path, steps=2, ckpt_every=2, log_every=10)
    train(DormouseModel(first.model), first, log=lambda *_: None)
    weights, snapshot = ckpt_paths(first.train)
    assert weights.is_file() and snapshot.is_file()

    # Only a PROGRESS key changed (more steps): a legal resume.
    second = tiny_run(tmp_path, steps=4, ckpt_every=2, log_every=10)
    assert second.diff_keys(first) == []
    report = train(DormouseModel(second.model), second, resume=True, log=lambda *_: None)
    assert report["opt_steps"] == 4, "the optimizer step count must continue, not restart"

    # The checkpoint's weights are the ones a load produces — not a fresh init.
    ck, _stored = load_ckpt(second)
    again = DormouseModel(second.model)
    again.load_state_dict(ck["model"])
    assert torch.equal(again.embed.weight, ck["model"]["embed.weight"])
    assert not torch.equal(again.embed.weight, DormouseModel(second.model).embed.weight)


def test_resume_refuses_a_changed_non_progress_key(tmp_path):
    """(d) A non-progress key at resume raises, naming the key and the escape."""
    run = tiny_run(tmp_path, steps=2, ckpt_every=2)
    train(DormouseModel(run.model), run, log=lambda *_: None)
    hotter = tiny_run(tmp_path, steps=4, ckpt_every=2)
    hotter = dataclasses.replace(hotter, train=dataclasses.replace(hotter.train, lr=0.5))
    with pytest.raises(LoudError) as e:
        load_ckpt(hotter)
    msg = str(e.value)
    assert "train.lr" in msg and "resume refused" in msg
    assert "progress keys" in msg and "new ckpt_name" in msg
    with pytest.raises(LoudError):
        train(DormouseModel(hotter.model), hotter, resume=True, log=lambda *_: None)


def test_resume_without_a_checkpoint_is_loud(tmp_path):
    run = tiny_run(tmp_path, steps=1)
    with pytest.raises(LoudError, match="must both exist"):
        load_ckpt(run)


def test_optimizer_state_round_trips(tiny, byte_tensor):
    model = DormouseModel(tiny)
    opt = MixOptimizer(model, lr=1e-3, wd=0.01)
    ids, targets = byte_tensor()
    _, loss, _ = model(ids, targets)
    loss.backward()
    opt.step()
    assert opt.steps == 1
    other = MixOptimizer(DormouseModel(tiny), lr=1e-3, wd=0.01)
    other.load_state_dict(opt.state_dict())
    assert other.steps == 1
    assert other.adamw.state_dict()["state"], "AdamW moments must travel with the checkpoint"
