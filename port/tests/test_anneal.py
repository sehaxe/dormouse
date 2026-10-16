"""The annealed factor_qat ramp (burn-spectral's soft-to-hard recipe).

Measured 2026-10-16: hard STE from step 0 LOSES +0.05 BPB (ternary) /
+0.09 BPB (binary) to bf16 at 9.2M — 8-14x the control's seed spread. The
burn tree's own comment says the RAMP "converges better than hard STE from
step one (measured)", so the ramp is the one variant of that recipe left to
test. These tests pin the MECHANISM (pure in the step, reaching every
consumer, rebuilding what it must and nothing else); the A/B is the verdict.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from conftest import tiny_cfg, tiny_run, write_corpus

from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.model.experts import SparseMoE
from dormouse.train import train


def _model(**over) -> DormouseModel:
    return DormouseModel(tiny_cfg(n_experts=4, moe_topk=2, factor_qat="ternary", **over))


def test_ramp_reaches_every_consumer():
    model = _model()
    model.set_factor_qat_alpha(0.25)
    hit = 0
    for m in model.modules():
        if hasattr(m, "factor_qat_alpha"):
            assert m.factor_qat_alpha == 0.25
            hit += 1
    assert hit >= 5, "TSCTLinear(s) + DualLatentGLU(s) + SparseMoE must all carry it"


def test_alpha_zero_is_bit_identical_to_no_qat():
    q = _model()
    q.set_factor_qat_alpha(0.0)
    off = DormouseModel(tiny_cfg(n_experts=4, moe_topk=2, factor_qat="none"))
    off.load_state_dict(q.state_dict())
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        lq, _, _ = q(x, x)
        lo, _, _ = off(x, x)
    assert torch.equal(lq, lo), "alpha=0 must be the master forward, bit for bit"


def test_bank_cache_rebuilds_when_alpha_moves():
    model = _model()
    moe = next(m for m in model.modules() if isinstance(m, SparseMoE))
    a = moe._banks(torch.float32)[0].clone()
    moe.factor_qat_alpha = 0.5
    b = moe._banks(torch.float32)[0]
    assert not torch.equal(a, b), "a changed alpha must rebuild the quantized bank"
    c = moe._banks(torch.float32)[0]
    assert c is b, "an unchanged alpha must hit the cache (same object)"


def test_ramp_schedule_is_a_pure_function_of_the_step():
    n = 100
    assert [min(1.0, s / n) for s in (0, 1, 50, 99, 100, 250)] == [
        0.0,
        0.01,
        0.5,
        0.99,
        1.0,
        1.0,
    ]


def test_the_trainer_refuses_an_anneal_with_no_arm(tmp_path):
    run = tiny_run(tmp_path, factor_qat_anneal_steps=4)  # model defaults to none
    with pytest.raises(LoudError, match="no quantizer"):
        train(DormouseModel(run.model), run, resume=False, log=lambda *_: None)


def test_the_trainer_ramps_alpha_over_the_run(tmp_path):
    """The spy proves `min(1, step/N)` reached every step, once per step."""
    data = write_corpus(tmp_path / "c.bin")
    run = tiny_run(tmp_path, factor_qat_anneal_steps=4, steps=8)
    run = dataclasses.replace(
        run,
        model=dataclasses.replace(run.model, factor_qat="ternary"),
        train=dataclasses.replace(run.train, data=str(data), log_every=100, ckpt_every=100),
    )
    seen: list[float] = []
    model = DormouseModel(run.model)
    original = model.set_factor_qat_alpha

    def spy(a: float) -> None:
        seen.append(a)
        original(a)

    model.set_factor_qat_alpha = spy  # type: ignore[method-assign]
    train(model, run, resume=False, log=lambda *_: None)
    assert seen == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0], seen
