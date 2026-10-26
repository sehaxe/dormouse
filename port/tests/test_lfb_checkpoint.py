"""C1: `lfb_bias` is checkpoint state, not a zeroed buffer on every load.

The ledger's repro: with `moe_lb_coef=0.01` (every shipped preset) an 8-step
unbroken run and a 4+resume run differ in 50/51 tensors, because `lfb_counts`
is persistent and `lfb_bias` was not. With `moe_lb_coef=0.0` the same split is
bit-exact. These gates pin all three legs of the fix: the `moe` section is
written and restored, a file that lost it REFUSES a live controller, and the
readers (`load_model_half`) take it too.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from conftest import tiny_run

from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train import train
from dormouse.train.loop import (
    _lfb_modules,
    ckpt_paths,
    load_model_half,
    load_model_state,
)


def run_of(tmp_path, name: str, *, steps: int, ckpt_every: int, coef: float, **over):
    """A tiny run with a live (or dead) LFB controller and its own checkpoint dir."""
    run = tiny_run(
        tmp_path,
        steps=steps,
        ckpt_every=ckpt_every,
        log_every=1000,
        out_dir=str(tmp_path / name),
        **over,
    )
    return dataclasses.replace(run, model=dataclasses.replace(run.model, moe_lb_coef=coef))


def model_of(run) -> DormouseModel:
    """Same seed, same weights — the comparison is a resume, not an init draw."""
    torch.manual_seed(0)
    return DormouseModel(run.model)


def state_of(model: torch.nn.Module) -> dict:
    """Every tensor that decides the next forward — `state_dict` + the LFB biases."""
    out = dict(model.state_dict())
    out.update(
        {f"{name}.lfb_bias": m.lfb_bias.detach().clone() for name, m in _lfb_modules(model).items()}
    )
    return out


@pytest.mark.parametrize("coef", [0.01, 0.0])
def test_4_plus_resume_is_bit_identical_to_8_unbroken(tmp_path, coef):
    full = run_of(tmp_path, "full", steps=8, ckpt_every=8, coef=coef)
    whole = model_of(full)
    train(whole, full, log=lambda *_: None)

    head = run_of(tmp_path, "split", steps=4, ckpt_every=4, coef=coef)
    first = model_of(head)
    train(first, head, log=lambda *_: None)
    weights, _snap = ckpt_paths(head.train)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    assert ck.get("moe"), "the checkpoint must carry a per-module lfb_bias section"
    saved = next(iter(ck["moe"].values()))
    if coef:
        assert float(saved.abs().max()) > 0.0, "a live controller moves the bias before the ckpt"

    tail = run_of(tmp_path, "split", steps=8, ckpt_every=8, coef=coef)
    resumed = model_of(tail)
    train(resumed, tail, resume=True, log=lambda *_: None)

    a, b = state_of(whole), state_of(resumed)
    assert set(a) == set(b)
    worst = max(float((a[k] - b[k]).abs().max()) for k in a)
    assert worst == 0.0, f"4+resume diverged from the unbroken run by {worst:g} at coef={coef:g}"


def test_load_model_half_takes_the_sidecar(tmp_path):
    """eval/decode/serve read the model half — they must get the trained router."""
    run = run_of(tmp_path, "serve", steps=2, ckpt_every=2, coef=0.01)
    train(model_of(run), run, log=lambda *_: None)
    weights, _snap = ckpt_paths(run.train)
    ck = torch.load(weights, map_location="cpu", weights_only=False)

    fresh = model_of(run)
    assert all(float(m.lfb_bias.abs().max()) == 0.0 for m in _lfb_modules(fresh).values())
    load_model_state(fresh, run.train, load_model_half(weights))
    for name, module in _lfb_modules(fresh).items():
        assert torch.equal(module.lfb_bias, ck["moe"][name]), name


def test_a_file_without_the_section_refuses_a_live_controller(tmp_path):
    run = run_of(tmp_path, "old", steps=2, ckpt_every=2, coef=0.01)
    train(model_of(run), run, log=lambda *_: None)
    weights, _snap = ckpt_paths(run.train)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    ck.pop("moe")  # the pre-fix file: counts in `model`, no bias anywhere
    torch.save(ck, weights)

    longer = run_of(tmp_path, "old", steps=4, ckpt_every=2, coef=0.01)
    with pytest.raises(LoudError) as e:
        train(model_of(longer), longer, resume=True, log=lambda *_: None)
    msg = str(e.value)
    assert "lfb_counts" in msg and "lfb_bias" in msg
    assert "Escape:" in msg and "moe_lb_coef=0" in msg


def test_the_zero_coef_escape_resumes_the_old_file(tmp_path):
    """`moe_lb_coef=0` is the documented escape: the controller never moves the bias."""
    run = run_of(tmp_path, "zero", steps=2, ckpt_every=2, coef=0.0)
    train(model_of(run), run, log=lambda *_: None)
    weights, _snap = ckpt_paths(run.train)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    ck.pop("moe")
    torch.save(ck, weights)

    longer = run_of(tmp_path, "zero", steps=4, ckpt_every=2, coef=0.0)
    report = train(model_of(longer), longer, resume=True, log=lambda *_: None)
    assert report["opt_steps"] == 4
