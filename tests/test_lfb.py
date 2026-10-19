"""LFB's routing controller is MODEL state: a checkpoint must carry it.

Audit 2026-10-19 M8: `lfb_counts` persisted while `lfb_bias` did not, so a
resume restored the integrator and reset the bias it had already moved — a
different routing controller under the same step count.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch
from conftest import tiny_cfg, tiny_run
from typer.testing import CliRunner

from dormouse.cli import app
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train import train
from dormouse.train.loop import ckpt_paths, lfb_path, load_model_half, load_model_state

runner = CliRunner()


def _moe(model: DormouseModel):
    return next(m for m in model.modules() if hasattr(m, "lfb_bias"))


def _checkpoint_without_sidecar(tmp_path: Path, *, coef: float = 0.05):
    """A trained checkpoint (controller on by default) with its `.lfb` removed."""
    run = dataclasses.replace(
        tiny_run(tmp_path, steps=2, log_every=10, ckpt_every=2, ckpt_name="lfb"),
        model=tiny_cfg(moe_lb_coef=coef),
    )
    train(DormouseModel(run.model), run, log=lambda *_: None)
    lfb_path(run.train).unlink()
    return run


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
    lines: list[str] = []
    load_model_state(fresh, run.train, load_model_half(weights), log=lines.append)
    assert torch.equal(_moe(fresh).lfb_bias, moved)
    assert torch.equal(_moe(fresh).lfb_counts, counts)
    assert not any(line.startswith("lfb:") for line in lines), "a present sidecar is silent"


def test_a_training_resume_without_the_sidecar_is_loud(tmp_path: Path):
    """Controller on + no sidecar + a RESUME: the one case that must refuse.

    The resume would continue the routing controller from zero under the
    checkpoint's own step count, which is a different network (review round 2
    R2-2 scoped the old LOUD here, not to every reader).
    """
    run = _checkpoint_without_sidecar(tmp_path)
    fresh = DormouseModel(run.model)
    with pytest.raises(LoudError, match="LFB sidecar"):
        load_model_state(
            fresh,
            run.train,
            load_model_half(ckpt_paths(run.train)[0]),
            resume=True,
        )
    with pytest.raises(LoudError, match="LFB sidecar"):
        train(DormouseModel(run.model), run, resume=True, log=lambda *_: None)


def test_a_resume_with_the_controller_off_is_accepted_and_counted(tmp_path: Path):
    """The documented escape: `moe_lb_coef=0` turns the controller off, so the
    missing bias steers nothing; the resume loads and the line says zeroed.

    The snapshot gate used to refuse this as `model.moe_lb_coef` drift, which
    made the escape unreachable (review round 2 R2-2).
    """
    run = _checkpoint_without_sidecar(tmp_path)
    off = dataclasses.replace(run, model=dataclasses.replace(run.model, moe_lb_coef=0.0))
    lines: list[str] = []
    train(DormouseModel(off.model), off, resume=True, log=lines.append)
    assert any(line.startswith("lfb: no sidecar, controller zeroed") for line in lines), lines


def test_readers_load_a_pre_sidecar_checkpoint_with_a_counted_line(tmp_path: Path, capsys):
    """`eval` and `serve` are readers: a pre-sidecar checkpoint must load.

    The refusal belongs to a training resume; a reader has no objective to
    continue. Each reader gets the COUNTED `lfb: no sidecar, controller
    zeroed` line instead (review round 2 R2-2: the old LOUD refused serve,
    eval, bench, decode and export for every checkpoint on disk).
    """
    run = _checkpoint_without_sidecar(tmp_path)
    r = runner.invoke(
        app,
        [
            "eval",
            "--out-dir",
            run.train.out_dir,
            "--ckpt-name",
            run.train.ckpt_name,
            "--eval",
            run.train.data,
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "lfb: no sidecar, controller zeroed" in r.output

    from dormouse.serve import load_checkpoint

    capsys.readouterr()  # drop the eval command's own output
    model, loaded, _dev = load_checkpoint(
        Path(run.train.out_dir), run.train.ckpt_name, device="cpu"
    )
    assert isinstance(model, DormouseModel) and loaded.model == run.model
    assert "lfb: no sidecar, controller zeroed" in capsys.readouterr().out
