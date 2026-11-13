"""`stiefel_qr` is WIRED: the retraction runs inside `train()`, not just as a helper.

The module's own gates (`tests/test_stiefel.py`) prove the maths. This file
proves the SEAM — that a trainer run with the flag actually retracts the TSCT
factors, and that the flag being off changes nothing. A flag that is accepted,
plumbed through `--set`, and never called is the defect class this repo calls
SILENT (ADR-0011): the run would report a mechanism it never executed.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
from conftest import tiny_run

from dormouse.config import ModelConfig
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train.loop import train
from dormouse.train.stiefel import orthonormality_error

TSCT_MODEL = dict(
    use_tsct=True,
    block_form="transformer",
    max_iter=1,
    use_kda=False,
    use_engram=False,
    n_experts=1,
    moe_topk=0,
)


def _tsct_run(tmp_path, **train_over):
    run = tiny_run(tmp_path, **train_over)
    return dataclasses.replace(run, model=dataclasses.replace(run.model, **TSCT_MODEL))


def _factor_errors(model: DormouseModel) -> dict[str, float]:
    out = {}
    for name, p in model.named_parameters():
        if name.split(".")[-1] in ("u", "v") and p.ndim == 2:
            out[name] = orthonormality_error(p.detach())
    return out


@pytest.mark.parametrize("flag", [True, False])
def test_the_flag_decides_whether_the_factors_are_retracted(tmp_path, flag):
    run = _tsct_run(tmp_path, steps=3, log_every=10, stiefel_qr=flag)
    model = DormouseModel(run.model)
    report = train(model, run, log=lambda *_: None)
    assert math.isfinite(report["loss"])
    errors = _factor_errors(model)
    assert errors, "the tiny TSCT model must have factor parameters to test the seam"
    worst = max(errors.values())
    if flag:
        assert worst <= 1e-5, f"stiefel_qr=true left a factor off the manifold: {errors}"
    else:
        # The control must NOT already be orthonormal, or the gate above proves
        # nothing. A random-init factor is far from the manifold (measured
        # 5.5e-1 on the tiny shape).
        assert worst > 1e-3, f"the control is orthonormal by accident: {errors}"


def test_the_flag_is_refused_on_a_model_without_factors(tmp_path, monkeypatch):
    """Every model here carries at least the `lm_head` factor, so the guard is
    exercised by making the factor predicate answer False — the branch is
    otherwise untestable, and an untestable refusal is not evidence."""
    import dormouse.model.tsct as tsct_mod

    monkeypatch.setattr(tsct_mod, "is_factor_key", lambda _key: False)
    run = tiny_run(tmp_path, steps=1, stiefel_qr=True)
    with pytest.raises(LoudError, match="no TSCT factors"):
        train(DormouseModel(run.model), run, log=lambda *_: None)


def test_the_flag_is_refused_with_the_captured_optimizer(tmp_path):
    run = _tsct_run(tmp_path, steps=1, stiefel_qr=True, opt_graph=True)
    with pytest.raises(LoudError, match="opt_graph"):
        train(DormouseModel(run.model), run, log=lambda *_: None)


def test_the_default_is_off_and_reaches_no_code():
    """`TrainConfig.stiefel_qr` defaults to False and the model side is untouched."""
    from dormouse.config import TrainConfig

    assert TrainConfig().stiefel_qr is False
    cfg = ModelConfig()
    assert not hasattr(cfg, "stiefel_qr"), "the flag belongs to the TRAINER, not the model"
