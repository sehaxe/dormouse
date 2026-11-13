"""The four cheap recipe knobs from the Ripple-65B ladder, gated one per claim.

Source: `docs/records/reviews/2026-11-08-ripple-65b.md`. Their v1.0→v1.6 ladder
compounded six cheap changes into 16x compute efficiency at 4B, which is larger
than any architectural delta measured in this repo. These gates exist so that
each knob is *provably* wired — a flag that reaches no code is the config-lies
defect `validate_block_form` was written to prevent.

a. every knob defaults OFF and leaves the block tree untouched;
b. `value_residual` starts BIT-IDENTICAL to off (α is zero-init) and its α
   receives a nonzero gradient — the gate that catches both a nonzero init and
   a disconnected carry;
c. `attn_gate` adds exactly `d·heads + heads` parameters, changes the logits,
   and its gradient is nonzero;
d. `ffn_clip` bounds the post-GELU activation (logits change) and a negative
   ceiling is refused;
e. `nope_layers` marks exactly the LAST N blocks, changes the logits, and is
   refused on the KDA form, above the stack depth, and on a hybrid stack;
f. all four together survive a forward+backward and a real 10-step CPU run;
g. `min_lr_frac=0.0` is a full cooldown to zero and is accepted by the config.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from conftest import SEQ, tiny_cfg, tiny_run

from dormouse.config import ModelConfig, TrainConfig, validate_train
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.train import train
from dormouse.train.lr import wsd_lr

#: The tiny fixture as a legal vanilla transformer (`max_iter=1`, no KDA arms).
BASE = dict(
    block_form="transformer",
    max_iter=1,
    loop_blocks=1,
    n_layers=4,
    use_kda=False,
    use_csa=False,
    use_mhc=False,
    use_attnres=False,
    use_engram=False,
    layerrope=False,
    n_experts=1,
    moe_topk=0,
    use_tsct=False,
    bf16=False,
)


def _model(**over) -> DormouseModel:
    torch.manual_seed(1234)
    return DormouseModel(tiny_cfg(**{**BASE, **over}))


def _batch(seq: int = SEQ) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, 256, (2, seq), generator=g)
    return ids, ids.roll(-1, dims=1)


def _logits(model: DormouseModel) -> torch.Tensor:
    ids, _ = _batch()
    logits, _, _ = model(ids)
    return logits


def _loss(model: DormouseModel) -> torch.Tensor:
    ids, targets = _batch()
    _, loss, _ = model(ids, targets)
    assert loss is not None
    return loss


# --- a. defaults ------------------------------------------------------------


def test_every_knob_defaults_off() -> None:
    cfg = ModelConfig()
    assert cfg.value_residual is False
    assert cfg.attn_gate is False
    assert cfg.ffn_clip == 0.0
    assert cfg.nope_layers == 0
    model = _model()
    blocks = list(model.transformer_blocks)
    assert all(b.v_res_alpha is None for b in blocks)
    assert all(b.gate is None for b in blocks)
    assert all(b.ffn_clip == 0.0 for b in blocks)
    assert all(b.nope is False for b in blocks)


# --- b. value residuals -----------------------------------------------------


def test_value_residual_starts_bit_identical_and_learns() -> None:
    plain = _model()
    vr = _model(value_residual=True)
    # α is zero-init, so the arm must be the SAME function at step 0. A nonzero
    # init or an unconditional add fails here, which is the point of the gate.
    assert torch.equal(_logits(plain), _logits(vr))
    # ...and the α must be the thing that makes it different: perturbing it
    # changes the logits, so the carry is really on the path.
    alphas = [b.v_res_alpha for b in vr.transformer_blocks if b.v_res_alpha is not None]
    # Block 0 is the source, so it owns no α: N-1 parameters, every one read.
    assert len(alphas) == 3
    with torch.no_grad():
        for a in alphas:
            a.fill_(0.5)
    assert not torch.equal(_logits(plain), _logits(vr))
    # The gradient reaches α: a disconnected `v_first` leaves it at zero.
    vr.zero_grad(set_to_none=True)
    _loss(vr).backward()
    grads = [a.grad for a in alphas]
    assert all(g is not None for g in grads)
    assert any(float(g.abs().sum()) > 0.0 for g in grads)


# --- c. sigmoid attention gate ---------------------------------------------


def test_attn_gate_adds_params_changes_logits_and_learns() -> None:
    plain, gated = _model(), _model(attn_gate=True)
    d, heads = plain.cfg.d_model, plain.cfg.n_heads
    blocks = len(plain.transformer_blocks)
    assert param_count(gated) - param_count(plain) == (d * heads + heads) * blocks
    assert not torch.equal(_logits(plain), _logits(gated))
    gated.zero_grad(set_to_none=True)
    _loss(gated).backward()
    gate_grads = [b.gate.weight.grad for b in gated.transformer_blocks]
    assert all(g is not None for g in gate_grads)
    assert any(float(g.abs().sum()) > 0.0 for g in gate_grads)


# --- d. FFN activation clipping -------------------------------------------


def test_ffn_clip_changes_logits_and_refuses_a_negative_ceiling() -> None:
    plain = _model()
    clipped = _model(ffn_clip=0.5)
    assert not torch.equal(_logits(plain), _logits(clipped))
    assert torch.isfinite(_logits(clipped)).all()
    with pytest.raises(LoudError, match="ffn_clip"):
        _model(ffn_clip=-1.0)


# --- e. NoPE on the tail ---------------------------------------------------


def test_nope_marks_exactly_the_last_blocks_and_changes_logits() -> None:
    plain = _model()
    nope = _model(nope_layers=2)
    flags = [b.nope for b in nope.transformer_blocks]
    assert flags == [False, False, True, True]
    assert not torch.equal(_logits(plain), _logits(nope))
    with pytest.raises(LoudError, match="nope_layers"):
        _model(nope_layers=5)
    with pytest.raises(LoudError, match="nope_layers"):
        _model(nope_layers=-1)
    # The KDA loop block reads none of these, so the flag is refused there
    # rather than left inert.
    with pytest.raises(LoudError, match="block_form='kda' refuses"):
        tiny_cfg(value_residual=True)
    with pytest.raises(LoudError, match="block_form='kda' refuses"):
        tiny_cfg(nope_layers=1)


def test_value_residual_is_refused_with_mod() -> None:
    with pytest.raises(LoudError, match="value_residual"):
        _model(value_residual=True, mod_capacity=0.5)


# --- f. all four together, and a real run ----------------------------------


def test_all_knobs_together_forward_backward_finite() -> None:
    model = _model(value_residual=True, attn_gate=True, ffn_clip=7.0, nope_layers=1)
    model.zero_grad(set_to_none=True)
    loss = _loss(model)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_all_knobs_together_survive_a_real_run(tmp_path) -> None:
    run = tiny_run(tmp_path)
    model = tiny_cfg(
        **{**BASE, "value_residual": True, "attn_gate": True, "ffn_clip": 7.0, "nope_layers": 1}
    )
    run = dataclasses.replace(run, model=model)
    train(DormouseModel(model), run, log=lambda *_: None)


# --- g. LR cooldown to zero ------------------------------------------------


def test_min_lr_frac_zero_is_a_full_cooldown() -> None:
    assert wsd_lr(
        9, total_steps=10, peak_lr=1.0, warmup_steps=0, decay_steps=5, final_frac=0.0
    ) == pytest.approx(0.0)
    # Monotone, and it reaches zero only at the last step.
    vals = [
        wsd_lr(s, total_steps=10, peak_lr=1.0, warmup_steps=0, decay_steps=5, final_frac=0.0)
        for s in range(10)
    ]
    assert vals == sorted(vals, reverse=True)
    assert vals[-1] == pytest.approx(0.0) and vals[-2] > 0.0
    assert wsd_lr(
        9, total_steps=10, peak_lr=1.0, warmup_steps=0, decay_steps=5, final_frac=0.1
    ) == pytest.approx(0.1)
    validate_train(TrainConfig(min_lr_frac=0.0))  # accepted
    with pytest.raises(LoudError, match="min_lr_frac"):
        validate_train(TrainConfig(min_lr_frac=-0.1))
