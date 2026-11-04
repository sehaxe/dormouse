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


def test_json_log_may_move_on_a_resume(tmp_path):
    """Audit MINOR 23: the telemetry sink is a progress key, so a resume that
    only moves it continues (the field's own docstring, now true)."""
    first = tiny_run(tmp_path, steps=2, ckpt_every=2, log_every=10)
    train(DormouseModel(first.model), first, log=lambda *_: None)
    moved = tiny_run(
        tmp_path, steps=4, ckpt_every=2, log_every=10, json_log=str(tmp_path / "moved.jsonl")
    )
    assert moved.diff_keys(first) == []
    report = train(DormouseModel(moved.model), moved, resume=True, log=lambda *_: None)
    assert report["opt_steps"] == 4, "the moved sink must not restart the run"


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

def test_the_window_is_a_progress_key_but_a_shrink_is_not(tmp_path):
    """`seq_len` widens mid-run (curriculum); `max_seq_len` follows it up but
    never down, and only on arms with no positional table.

    Measured reason (2026-11-04): every pretrain run to date trained on a
    1024-BYTE window while GPT-2's 1024 TOKENS is ~4096 bytes, and the 8 KB
    `longctx1` run had come in 0.19 BPB lower. Being unable to widen the window
    without throwing away the optimizer state is a policy bug, not safety.
    """
    from dormouse.config import PROGRESS_KEYS, load_run

    base = load_run("configs/m400.toml", train_overrides={"seq_len": 1024})
    wide = load_run(
        "configs/m400.toml",
        model_overrides={"max_seq_len": 8192},
        train_overrides={"seq_len": 8192},
    )
    assert "seq_len" in PROGRESS_KEYS
    assert "train.seq_len" not in base.diff_keys(wide)
    # the window may not exceed the model's buffer bound
    with pytest.raises(LoudError, match="exceeds max_seq_len"):
        load_run("configs/m400.toml", train_overrides={"seq_len": 8192})

    # a wider BUFFER bound is allowed on a KDA/CSA arm...
    grown = load_run(
        "configs/m400.toml",
        model_overrides={"max_seq_len": 8192},
        train_overrides={"seq_len": 8192},
    )
    assert "model.max_seq_len" not in base.diff_keys(grown)

    # ...but a SHRINK is a different run, not a resume. `diff_keys(self=stored,
    # other=incoming)`, so the incoming run is the narrow one.
    narrowed = load_run("configs/m400.toml", model_overrides={"max_seq_len": 512},
                        train_overrides={"seq_len": 512})
    # direction-agnostic: BOTH call sites must agree (`load_ckpt` passes
    # (new, stored), the test below passes (stored, new))
    assert base.diff_keys(narrowed) == narrowed.diff_keys(base) == []
    # a SHRUNKEN training window is still legal on such an arm — the field only
    # bounds decode; `validate_train` keeps seq_len <= max_seq_len
    assert "train.seq_len" not in base.diff_keys(narrowed)

    # the TOKEN BUDGET per step is the real identity: re-shard it freely...
    reshard = load_run("configs/m400.toml", model_overrides={"max_seq_len": 8192},
                       train_overrides={"batch": 1, "seq_len": 8192})
    assert base.diff_keys(reshard) == []
    # ...but a different budget is a different run
    smaller = load_run("configs/m400.toml", train_overrides={"batch": 1, "seq_len": 1024})
    assert "train.batch" in base.diff_keys(smaller)

    # an arm that BUILDS a positional table keeps the strict refusal
    bf = load_run(
        "configs/m400.toml",
        model_overrides={
            "use_byteflow": True,
            "byteflow_k_tokens": 64,
            "use_engram": False,
        },
        train_overrides={"seq_len": 1024},
    )
    assert "model.max_seq_len" in bf.diff_keys(wide)
