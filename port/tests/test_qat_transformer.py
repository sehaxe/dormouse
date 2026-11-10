"""Gates for the QAT seam on the LIVE dense transformer block.

The owner's arm: ternary forward weights (`dense_qat`) + 4-bit activations
(`act_qat`) inside `TransformerBlock` — `block_form="transformer"`, the gated
FFN of `configs/arch_gated_swiglu.toml`. Both flags default OFF and the off
forward must be the recorded one, bit for bit.

What these pins are, and what they are NOT:

- the QUANTIZERS are other files' (`tests/test_ternary.py`, `test_act_quant.py`,
  the new `test_nvfp4.py`). What is tested here is the WIRING: that the block's
  six projections read the projector, that `alpha=0` is the master and `alpha=1`
  is a forward whose weights were pre-quantized in place, that the activation
  seam lands on e2m1 with one scale per 16 columns, and that the flags are
  refused where no GEMM was wired;
- the STE is asserted where it is a property, not an approximation: the backward
  of `ste_ternary` is EXACTLY the identity, so the master's gradient equals the
  gradient of a forward that used the pre-quantized weight directly;
- the off arm is pinned against a fixture captured from HEAD
  (`tests/fixtures/transformer_off_prechange.pt`, commit b7c2f72), logits AND
  parameter names, because "no RNG order moved" is only provable that way.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch
from conftest import tiny_run, write_corpus
from torch import nn

from dormouse.config import (
    ModelConfig,
    TrainConfig,
    load_preset,
    load_run,
    validate,
    validate_model_train,
)
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.model.ternary import ste_ternary, ternarize
from dormouse.model.tsct import DENSE_QAT_ATTN_SITES, DenseLinear, dense_qat_project

PORT = Path(__file__).resolve().parents[1]
PRECHANGE = Path(__file__).resolve().parent / "fixtures" / "transformer_off_prechange.pt"
LIVE_ARM = PORT / "configs" / "arch_gated_swiglu.toml"


def build(**over) -> ModelConfig:
    """The tiny stand-in for the live block: pre-LN transformer + gated SwiGLU."""
    kw = dict(
        d_model=64,
        n_heads=4,
        head_dim=16,
        d_ffn=128,
        d_ffn_gelu=192,
        vocab=256,
        max_seq_len=64,
        n_layers=2,
        max_iter=1,
        block_form="transformer",
        ffn_gated=True,
        expert_act="silu",
        bf16=False,
        use_tsct=False,
        tie_embeddings=False,
        use_kda=False,
        use_engram=False,
        use_csa=False,
        layerrope=False,
    )
    kw.update(over)
    return ModelConfig(**kw)


def _model(**over) -> DormouseModel:
    torch.manual_seed(1234)
    return DormouseModel(build(**over))


def _input() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randint(0, 256, (2, 16))


def _forward(model: DormouseModel, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits, _, _ = model(x, x)
    return logits


# ---------------------------------------------------------------------------
# flag OFF: the recorded forward, parameter names and all
# ---------------------------------------------------------------------------


def test_off_forward_is_head_bit_identical():
    """The golden was captured from HEAD (b7c2f72) with this exact fixture.

    One thread on BOTH sides: fp32 reduction order on CPU follows the thread
    count, so the bits are comparable only at the recorded width.
    """
    golden = torch.load(PRECHANGE, weights_only=False)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = _model()
        assert sorted(model.state_dict()) == golden["keys"], (
            "the off tree's parameter names moved — a flag-off build must be the "
            "control's, RNG order included"
        )
        assert torch.equal(_forward(model, golden["input"]), golden["logits"])
    finally:
        torch.set_num_threads(old)


def test_off_forward_never_calls_the_projector(monkeypatch):
    """`dense_qat='none'` must not even CALL the projector (the `mm_fp8` rule)."""
    import dormouse.model.tsct as tsct

    def spy(*a, **k):
        raise AssertionError("dense_qat='none' reached the projector")

    monkeypatch.setattr(tsct, "dense_qat_project", spy)
    _forward(_model(), _input())


# ---------------------------------------------------------------------------
# the weight side: alpha=1 is the pre-quantized forward, the STE is identity
# ---------------------------------------------------------------------------


def _pre_quantize(model: nn.Module, *, skip_attn: bool = True) -> int:
    """Copy the BLENDED projection into every covered weight, in place.

    `dense_qat_project(w, "ternary", 1.0)` is the tensor the flag-on forward
    actually multiplies by (`w + (q(w) − w).detach()`), so a twin built this way
    must reproduce the flag-on logits EXACTLY — which is the wiring gate. Both
    `nn.Linear` and `DenseLinear` are walked: the head and the block projections
    are different classes on purpose, and a fixture that saw only one would
    leave a site unverified.
    """
    n = 0
    with torch.no_grad():
        for name, m in model.named_modules():
            if not isinstance(m, (nn.Linear, DenseLinear)):
                continue
            if skip_attn and name.endswith((".qkv", ".proj")):
                continue
            m.weight.copy_(dense_qat_project(m.weight, "ternary", 1.0))
            n += 1
    return n


def test_alpha_one_forward_equals_pre_quantized_weights():
    a = _model(dense_qat="ternary")
    b = _model(dense_qat="none")
    b.load_state_dict(a.state_dict())
    n = _pre_quantize(b)
    assert n == 2 * 3 + 1, f"the fixture projected {n} weights, expected 3 per block + the head"
    x = _input()
    assert torch.equal(_forward(a, x), _forward(b, x)), (
        "alpha=1 must be the pre-quantized-weight forward — a site that ignored "
        "dense_qat would multiply the master and split the two here"
    )


def test_alpha_zero_forward_is_bit_identical():
    q = _model(dense_qat="ternary")
    q.set_factor_qat_alpha(0.0)
    off = _model()
    off.load_state_dict(q.state_dict())
    x = _input()
    assert torch.equal(_forward(q, x), _forward(off, x)), "alpha=0 is the master, bit for bit"


def test_ste_forward_has_three_levels_and_identity_backward():
    torch.manual_seed(0)
    w = torch.randn(16, 32) * 0.2
    q = ternarize(w)
    alpha = float(w.abs().mean())
    levels = torch.unique(q).tolist()
    assert len(levels) == 3, f"ternary is {{-α, 0, +α}}, got {levels}"
    assert sorted(abs(x) for x in levels) == [0.0, pytest.approx(alpha), pytest.approx(alpha)]
    # `ste_ternary`'s forward IS that projection (up to the add's rounding).
    assert torch.allclose(ste_ternary(w), q, rtol=0, atol=1e-7)

    # The estimator is EXACTLY the identity in the backward.
    w = w.clone().requires_grad_(True)
    c = 3.0
    (ste_ternary(w) * c).sum().backward()
    assert torch.equal(w.grad, torch.full_like(w, c)), "ste_ternary's dL/dw must be 1"


def test_master_gradient_equals_the_pre_quantized_gradient():
    """The STE's claim, at model scale: dL/dw(master) == dL/d(weight used)."""
    a = _model(dense_qat="ternary")
    b = _model(dense_qat="none")
    b.load_state_dict(a.state_dict())
    _pre_quantize(b)
    x = _input()
    for m in (a, b):
        _, loss, _ = m(x, x)
        loss.backward()
    fa = a.transformer_blocks[0].fc2.weight.grad
    fb = b.transformer_blocks[0].fc2.weight.grad
    assert fa is not None and fb is not None
    assert fa.abs().sum() > 0 and torch.isfinite(fa).all()
    torch.testing.assert_close(fa, fb, rtol=0, atol=1e-6)


def test_skip_attn_keeps_qkv_and_proj_on_their_master():
    """The BitNet rule, at the block: attention weights read the raw master."""
    skip = DormouseModel(build(dense_qat="ternary"))
    nope = DormouseModel(build(dense_qat="ternary", dense_qat_skip_attn=False))
    blk_skip, blk_nope = skip.transformer_blocks[0], nope.transformer_blocks[0]
    assert (blk_skip._qat_attn, blk_skip._qat_ffn) == ("none", "ternary")
    assert (blk_nope._qat_attn, blk_nope._qat_ffn) == ("ternary", "ternary")
    assert "transformer.qkv" in DENSE_QAT_ATTN_SITES and "transformer.proj" in DENSE_QAT_ATTN_SITES
    nope.load_state_dict(skip.state_dict())
    x = _input()
    assert not torch.equal(_forward(skip, x), _forward(nope, x)), (
        "skip_attn=false must project qkv/proj and move the logits"
    )


# ---------------------------------------------------------------------------
# the activation side: e2m1, one e4m3 scale per 16 columns
# ---------------------------------------------------------------------------


def test_act_qat_quantizes_the_ffn_gemm_inputs_with_block_16(monkeypatch):
    import dormouse.model.transformer as tr

    seen: list[tuple[str, int, torch.Tensor]] = []
    real = tr.fake_quant_act

    def spy(x, fmt, group=0):
        out = real(x, fmt, group)
        seen.append((fmt, group, out.detach()))
        return out

    monkeypatch.setattr(tr, "fake_quant_act", spy)
    model = _model(act_qat="e2m1")
    x = _input()
    _, _, cnt = model(x, x)
    assert {f for f, _, _ in seen} == {"e2m1"}
    assert {g for _, g, _ in seen} == {16}
    # 3 GEMM inputs per block (h2 once, the post-activation twice? no: gate/up
    # share h2's quantized copy) — the count is per SITE, and every one of them
    # lands on the 16-wide scale grid.
    assert len(seen) == 2 * 2, f"{len(seen)} activation quantizations for 2 blocks"
    assert cnt.get("act_qat_arms") == 2, "one counter per block forward"


def test_act_qat_output_is_on_the_e2m1_grid():
    from dormouse.model.act_quant import fake_quant_act
    from dormouse.model.tsct import _NVFP4_MAGNITUDES

    torch.manual_seed(0)
    x = torch.randn(4, 64) * 3.0
    q = fake_quant_act(x, "e2m1", 16)
    step = q.reshape(4, 4, 16).abs().amax(-1, keepdim=True) / 6.0
    norm = (q.reshape(4, 4, 16) / step).abs()
    grid = torch.tensor(_NVFP4_MAGNITUDES)
    on = torch.zeros_like(norm, dtype=torch.bool)
    for v in grid:
        on |= (norm - v).abs() < 1e-5
    assert bool(on.all()), "an e2m1-quantized activation left the grid"
    assert not torch.equal(q, x)


def test_act_qat_trains_and_backprops_through_the_ste():
    model = _model(act_qat="e2m1")
    x = _input()
    _, loss, _ = model(x, x)
    loss.backward()
    for name, p in model.transformer_blocks[0].named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
    assert model.transformer_blocks[0].fc2.weight.grad.abs().sum() > 0


def test_no_host_sync_in_the_training_forward(monkeypatch):
    """ADR-0018 rule 2: the quantized forward reads no device scalar back."""
    model = _model(dense_qat="ternary", act_qat="e2m1")
    x = _input()

    def boom(*_a, **_k):
        raise AssertionError("host sync in the transformer QAT forward")

    for name in ("item", "tolist", "cpu", "numpy", "__float__", "__int__"):
        monkeypatch.setattr(torch.Tensor, name, boom)
    _, loss, _ = model(x, x)
    loss.backward()


# ---------------------------------------------------------------------------
# config: the refusals, and the LIVE arm accepts the flags
# ---------------------------------------------------------------------------


def test_live_arch_gated_swiglu_accepts_both_flags():
    base = load_preset(str(LIVE_ARM))
    for over in (
        {},
        {"dense_qat": "ternary"},
        {"act_qat": "e2m1"},
        {"dense_qat": "ternary", "act_qat": "e2m1"},
    ):
        cfg = dataclasses.replace(base, **over)
        validate(cfg)  # LOUD on anything inconsistent


def test_config_refuses_a_bad_literal_and_an_unwired_block_form():
    with pytest.raises(LoudError, match="not an activation arm"):
        validate(build(act_qat="int4"))
    with pytest.raises(LoudError, match="block_form"):
        load_run("small", model_overrides={"act_qat": "e2m1"})  # the loop form


def test_config_refuses_two_activation_quantizers():
    with pytest.raises(LoudError, match="TWO activation quantizers"):
        validate(build(act_qat="e2m1", act_quant="fp4"))


def test_config_refuses_a_width_that_does_not_tile_16():
    with pytest.raises(LoudError, match="tile 16"):
        validate(build(act_qat="e2m1", d_ffn=120, ffn_gated=False, d_ffn_gelu=0))


def test_config_refuses_e2m1_under_torch_compile():
    cfg = build(act_qat="e2m1")
    with pytest.raises(LoudError, match="torch_compile"):
        validate_model_train(cfg, TrainConfig(torch_compile=True))
    validate_model_train(cfg, TrainConfig(torch_compile=False))  # the legal side


# ---------------------------------------------------------------------------
# the trainer path: 5 steps with ternary weights + e2m1 activations
# ---------------------------------------------------------------------------


def test_5_step_train_smoke(tmp_path):
    data = write_corpus(tmp_path / "c.bin")
    run = tiny_run(tmp_path, steps=5, log_every=2, ckpt_every=100)
    run = dataclasses.replace(
        run,
        model=build(dense_qat="ternary", act_qat="e2m1"),
        train=dataclasses.replace(run.train, data=str(data)),
    )
    lines: list[str] = []
    report = train_model(run, lines)
    assert report["loss"] == report["loss"]  # finite
    assert report["counters"]["dense_qat_arms"] == 5
    assert report["counters"]["act_qat_arms"] == 5 * 2  # two blocks per forward
    # The per-step cache path ran: one refresh per covered weight per step (the
    # 6 block projections + the head), never the per-forward re-quantization.
    assert report["counters"]["qat_refresh_calls"] >= 5 * 6


def train_model(run, lines):
    from dormouse.train import train

    return train(DormouseModel(run.model), run, resume=False, log=lines.append)
