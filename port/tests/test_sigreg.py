"""SIGReg: the formula, the strided input, the hook, the refusals.

`docs/research/2026-10-16-jepa-into-dormouse.md` §2/§3 is the spec — SIGReg
(arXiv:2511.08544) replaces KoLeo, and the assembly is
`CE + sigreg_weight · sigreg_loss(z)` with `z` the strided sample of the state
`lm_head` reads: teacher-free, predictor-free, mask-free, 0 params, no second
forward. The transcription source is
`vendor/dormouse-fused/crates/burn-jepa/src/losses.rs:33-50`.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from conftest import tiny_cfg, tiny_model, tiny_run

from dormouse.config import ModelConfig, load_preset, load_run, parse_preset, validate
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.model.sigreg import SIGREG_ROWS, sigreg_loss, stride_sample
from dormouse.train import train


def _first_loss(model, run) -> float:
    from dormouse.train.data import open_streams

    stream, _ = open_streams(run)
    ids, targets = stream.next_batch()
    _, loss, _ = model(ids, targets)
    stream.rewind()
    assert loss is not None
    return float(loss.detach())


def test_gold_direction_normal_is_small_collapsed_is_large():
    """(a) the DIRECTION, printed: this loss is a collapse detector.

    Two readings. The hook's own shape (`SIGREG_ROWS x d = 256 x 768`) gives the
    strict direction. The magnitude comes from the row count, and that is
    structural rather than seed luck: for iid N(0,1) rows `E[L] ≈ 2d/n`, while a
    constant row scores `‖mean‖² + 1 = d + 1` (the covariance is 0, so
    `‖0 − I‖²_F / d = 1`), i.e. `L_collapsed / L_normal ≈ n/2`. At 256 rows the
    gap is ~1.3e2; at 4 096 rows it is ~2e3, which is where the >= 1e3 bound is
    assertable without depending on a draw.
    """
    torch.manual_seed(0)
    normal = float(sigreg_loss(torch.randn(SIGREG_ROWS, 768)))
    collapsed = float(sigreg_loss(torch.ones(SIGREG_ROWS, 768)))
    wide = float(sigreg_loss(torch.randn(4096, 768)))
    wide_collapsed = float(sigreg_loss(torch.ones(4096, 768)))
    print(
        f"SIGReg GOLD: {SIGREG_ROWS}x768 normal={normal:.6g} collapsed={collapsed:.6g} "
        f"({collapsed / normal:.1f}x); 4096x768 normal={wide:.6g} "
        f"collapsed={wide_collapsed:.6g} ({wide_collapsed / wide:.1f}x)"
    )
    assert normal < collapsed
    assert collapsed / normal > 10.0
    assert wide_collapsed >= 1e3 * wide


def test_hand_reference_two_by_two_is_eighteen():
    """(b) the docstring's 2x2 case, recomputed line by line -> 18.0 (1e-6).

    ``z = [[1, 2], [3, 4]]``:  mean = [2, 3] -> 13;  cov = [[2, 2], [2, 2]];
    ``cov − I = [[1, 2], [2, 1]]`` -> 10, /D = 5;  13 + 5 = 18.
    """
    z = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mean = z.mean(dim=0)
    z_c = z - mean
    cov = z_c.transpose(0, 1) @ z_c / float(max(z.shape[0], 2) - 1)
    manual = mean.pow(2).sum() + (cov - torch.eye(2)).pow(2).sum() / 2
    assert (mean.tolist(), cov.tolist()) == ([2.0, 3.0], [[2.0, 2.0], [2.0, 2.0]])
    assert float(manual) == pytest.approx(18.0, abs=1e-6)
    assert float(sigreg_loss(z)) == pytest.approx(18.0, abs=1e-6)


def test_stride_sample_is_every_ceil_step_and_leaves_a_short_tensor_alone():
    """The sampler: `ceil(10/4) = 3` -> rows 0, 3, 6, 9; `n <= rows` is a no-op."""
    z = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    sampled = stride_sample(z, 4)
    assert sampled.shape == (4, 2)
    assert sampled[:, 0].tolist() == [0.0, 6.0, 12.0, 18.0]
    short = torch.zeros(3, 2)
    assert stride_sample(short, 4) is short


def test_zero_weight_is_byte_parity_and_the_counter_is_absent(byte_tensor):
    """(c) weight 0: the unarmed model's loss, bit for bit, and no counter.

    Bit-parity is the claim that makes pure CE a valid control without a
    re-baseline: the field exists, so a preset that does not set it cannot
    change a number.
    """
    ids, targets = byte_tensor()
    torch.manual_seed(0)
    plain = tiny_model()
    torch.manual_seed(0)
    zero = tiny_model(sigreg_weight=0.0)
    assert zero.cfg.sigreg_weight == 0.0
    _, loss_a, ca = plain(ids, targets)
    _, loss_b, cb = zero(ids, targets)
    assert torch.equal(loss_a, loss_b)
    assert "sigreg" not in ca and "sigreg" not in cb


def test_nonzero_weight_counts_the_term_and_the_model_still_trains(tmp_path):
    """(d) weight 0.1: the counter confirms the term was ADDED, and training works.

    The counter is the research doc's metric 2 (§3): a term bumped nowhere is a
    term that never ran while the arm looks alive. One bump per training forward
    (`eval=False`, so nothing else calls the model with labels).
    """
    torch.manual_seed(0)
    run = tiny_run(tmp_path, steps=30, lr=0.01, log_every=100)
    run = dataclasses.replace(run, model=tiny_cfg(sigreg_weight=0.1))
    model = DormouseModel(run.model)
    first = _first_loss(model, run)
    report = train(model, run, log=lambda *_: None)
    assert report["counters"]["sigreg"] == run.train.steps
    assert report["nan_skipped"] == 0
    assert report["loss"] < first, f"loss did not drop: {first} -> {report['loss']}"


def test_the_weight_is_a_real_config_key():
    """(e) `load_run("small")` and a preset both accept it; small ships 0.0."""
    assert load_preset("small").sigreg_weight == 0.0
    run = load_run("small", model_overrides={"sigreg_weight": 0.1})
    assert run.model.sigreg_weight == 0.1
    cfg = parse_preset("d_model = 64\nn_heads = 4\nhead_dim = 16\nsigreg_weight = 0.005\n")
    assert cfg.sigreg_weight == 0.005
    # A regulariser beside anything else is allowed: it is not a head, so the
    # one-signal-per-run rule (which is about two HEADS in one run) does not
    # apply to it.
    both = load_run("small", model_overrides={"sigreg_weight": 0.1, "dspark_weight": 0.1})
    assert (both.model.sigreg_weight, both.model.dspark_weight) == (0.1, 0.1)


def test_a_negative_weight_is_loud():
    """(f) a negative weight is a sign flip, not a weight: LOUD, naming the escape."""
    with pytest.raises(LoudError) as e:
        validate(ModelConfig(sigreg_weight=-1e-3))
    assert "sigreg_weight" in str(e.value) and "Escape" in str(e.value)
    with pytest.raises(LoudError):
        parse_preset("d_model = 64\nn_heads = 4\nhead_dim = 16\nsigreg_weight = -0.1\n")


def test_the_byteflow_channel_refuses_it_rather_than_dropping_it():
    """A term with nothing to attach to is refused at load, never silently inert.

    The channel's readout is `bf.out_proj(out_acc)`; SIGReg reads `norm(out)`.
    """
    preset = (
        "d_model = 64\nn_heads = 4\nhead_dim = 16\nuse_byteflow = true\n"
        "byteflow_k_tokens = 8\nbf16 = false\nuse_engram = false\nsigreg_weight = 0.1\n"
    )
    with pytest.raises(LoudError) as e:
        parse_preset(preset)
    assert "sigreg_weight" in str(e.value) and "use_byteflow" in str(e.value)
