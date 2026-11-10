"""`ffn_gated` in the LIVE `block_form="transformer"` block: gate/up/down, and the proofs.

The flag swaps `gelu(fc(x))@fc2^T` for `fc2(expert_glu(act)(gate(x), up(x)))`.
Each claim below fails on its own if the code regresses:

a. param parity: the gated FFN at `d_ffn = 2·d_ffn_gelu/3` holds EXACTLY the
   `2·d·d_ffn_gelu` weights of the control it replaces (3·832·2048 = 5,111,808 =
   2·832·3072, +0.00%), and a 9.4%-wider gated width is refused;
b. SiTU is the PAPER's Eq. 12 (`β1·tanh(g/β1)·σ(g) ⊙ β2·tanh(u/β2)`), computed
   HERE from the formula at arXiv 2607.24653's anchor points — never a golden
   captured from our own run;
c. `‖SiTU-GLU‖∞ ≤ β1·β2 = 100` (the paper's Eq. 19) up to |x| = 1e3;
d. near the origin SiTU is SiLU to 1e-2 relative (`β·tanh(z/β) = z + O(z³/β²)`);
e. a gate that FAILS if the activation silently reverts to SiLU: situ(10) ≈ 3.946
   against SiLU's 9.9995, a >5 absolute gap;
f. the full gated block forward equals a hand-written GLU (both activations,
   `ffn_clip` on and off) at 1e-5 fp32;
g. flag OFF: no `gate_proj`/`up_proj`, the GELU block's name set and parameter
   count, and a forward equal to `x + fc2(gelu(fc(ln2(x))))`;
h. `validate_ffn` refuses the four illegal combinations LOUD with an `Escape:`.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from dormouse.config import ModelConfig, load_preset, validate
from dormouse.instrument import LoudError
from dormouse.model.experts import expert_glu
from dormouse.model.transformer import TransformerBlock

#: Kimi K3's own SiTU scales (arXiv 2607.24653 §2.3.2), spelled here as the
#: paper's constants rather than read off the config under test.
B1, B2 = 4.0, 25.0


def _block_cfg(**over) -> ModelConfig:
    """A legal tiny `block_form="transformer"` config; `d_ffn=170` ≈ 8·64/3."""
    base = dict(
        block_form="transformer",
        max_iter=1,
        loop_blocks=1,
        n_layers=2,
        d_model=64,
        n_heads=4,
        head_dim=16,
        d_ffn=170,
        n_experts=1,
        moe_topk=0,
        bf16=False,
        use_kda=False,
        use_engram=False,
        use_tsct=False,
    )
    base.update(over)
    return ModelConfig(**base)


def _ffn_params(blk: TransformerBlock) -> int:
    return sum(
        p.numel()
        for n, p in blk.named_parameters()
        if n.startswith(("fc.", "fc2.", "gate_proj.", "up_proj."))
    )


def _mute_attention(blk: TransformerBlock) -> TransformerBlock:
    """Zero `qkv`/`proj` so the block forward IS the FFN on the raw residual.

    The FFN is the unit under test; a zeroed attention write makes
    `block(x) = x + ffn(x)` exact, so the reference below pins the GLU wiring
    and not a second transcription of RoPE/SDPA.
    """
    with torch.no_grad():
        blk.qkv.weight.zero_()
        blk.proj.weight.zero_()
    return blk


# --- (a) param parity -------------------------------------------------------


def test_gated_ffn_is_param_neutral_against_the_control_exactly():
    """3·832·2048 = 5,111,808 = 2·832·3072, the control's own FFN bill."""
    control = load_preset("flagship3_dense")
    assert control.d_ffn == 3072 and not control.ffn_gated
    base = _ffn_params(TransformerBlock(control, n_layers=12))
    assert base == 2 * 832 * 3072 == 5_111_808
    for name in ("arch_gated_swiglu", "arch_gated_situ"):
        cfg = load_preset(name)
        assert cfg.d_ffn_gelu == control.d_ffn, name
        assert cfg.d_ffn == 2 * cfg.d_ffn_gelu // 3 == 2048, name
        got = _ffn_params(TransformerBlock(cfg, n_layers=12))
        assert got == base, f"{name}: {got:,} vs the control's {base:,}"


def test_a_9_4_percent_wider_gated_width_is_refused():
    """`d_ffn=2240` is 5,591,040 against the control's 5,111,808 (+9.38%)."""
    cfg = dataclasses.replace(load_preset("arch_gated_swiglu"), d_ffn=2240)
    with pytest.raises(LoudError) as e:
        validate(cfg)
    msg = str(e.value)
    assert "5,591,040" in msg and "5,111,808" in msg and "Escape:" in msg, msg
    # The runtime seam a hand-built config reaches holds the same rule.
    with pytest.raises(LoudError, match="ffn_gated"):
        TransformerBlock(cfg, n_layers=12)


def test_both_gated_presets_load_and_differ_only_in_the_activation():
    silu, situ = load_preset("arch_gated_swiglu"), load_preset("arch_gated_situ")
    assert silu.expert_act == "silu" and situ.expert_act == "situ"
    diff = {
        f.name
        for f in dataclasses.fields(silu)
        if getattr(silu, f.name) != getattr(situ, f.name)
    }
    assert diff == {"expert_act"}, diff
    assert silu.ffn_gated and situ.ffn_gated


# --- (b) SiTU is the paper's formula ---------------------------------------


def test_situ_matches_the_paper_formula_at_the_anchor_points():
    """Eq. 12 transcribed in this TEST: `β1·tanh(g/β1)·σ(g) ⊙ β2·tanh(u/β2)`."""
    x = torch.tensor([0.0, 1.0, 4.0, 10.0, -10.0])
    got = expert_glu("situ", x, x, beta1=B1, beta2=B2)
    gate = B1 * torch.tanh(x / B1) * torch.sigmoid(x)
    up = B2 * torch.tanh(x / B2)
    assert torch.allclose(got, gate * up, atol=1e-6, rtol=1e-6), (got, gate * up)


def test_situ_glu_is_bounded_by_beta1_times_beta2():
    """Eq. 19: `|f| ≤ β1·β2 = 100` for inputs up to 1e3."""
    x = torch.linspace(-1e3, 1e3, 20_001)
    out = expert_glu("situ", x, x, beta1=B1, beta2=B2)
    assert float(out.abs().max()) <= B1 * B2


def test_situ_is_silu_near_the_origin():
    x = torch.linspace(-1e-2, 1e-2, 201)
    x = x[x != 0]
    situ = expert_glu("situ", x, torch.ones_like(x), beta1=B1, beta2=B2)
    silu = x * torch.sigmoid(x)
    rel = (situ - silu).abs() / silu.abs()
    assert float(rel.max()) <= 1e-2, float(rel.max())


def test_a_silu_revert_fails_loudly_at_x_10():
    """SiLU would read 9.9995 here; SiTU's gate is capped at 3.946 (Eq. 12)."""
    ten, one = torch.tensor([10.0]), torch.ones(1)
    situ = float(expert_glu("situ", ten, one, beta1=B1, beta2=B2))
    silu = float(ten * torch.sigmoid(ten))
    assert abs(situ - silu) > 5.0, (situ, silu)


# --- (c) the block forward -------------------------------------------------


def _hand_glu(act: str, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Eq. 12 / SwiGLU written out here — deliberately NOT `expert_glu`."""
    if act == "situ":
        gate = B1 * torch.tanh(g / B1) * torch.sigmoid(g)
        return gate * (B2 * torch.tanh(u / B2))
    assert act == "silu"
    return (g * torch.sigmoid(g)) * u


@pytest.mark.parametrize("act", ["silu", "situ"])
@pytest.mark.parametrize("clip", [0.0, 0.5])
def test_gated_block_forward_matches_a_hand_written_glu(act, clip):
    cfg = _block_cfg(ffn_gated=True, expert_act=act, ffn_clip=clip)
    torch.manual_seed(0)
    blk = _mute_attention(TransformerBlock(cfg, n_layers=2))
    x = torch.randn(2, 5, cfg.d_model, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        out = blk(x)
        h = blk.ln2(x)
        g = F.linear(h, blk.gate_proj.weight)
        u = F.linear(h, blk.up_proj.weight)
        f = _hand_glu(act, g, u)
        if clip > 0.0:
            f = f.clamp(max=clip)
        ref = x + F.linear(f, blk.fc2.weight)
    err = float((out - ref).abs().max())
    assert err <= 1e-5, f"{act}/{clip}: max |block - reference| = {err:g}"


# --- (d) flag OFF ----------------------------------------------------------


def test_flag_off_keeps_the_gelu_block_names_count_and_forward():
    cfg = _block_cfg(ffn_gated=False)
    torch.manual_seed(0)
    blk = _mute_attention(TransformerBlock(cfg, n_layers=2))
    names = {n for n, _ in blk.named_parameters()}
    assert names == {
        "qkv.weight",
        "proj.weight",
        "ln1.weight",
        "ln1.bias",
        "ln2.weight",
        "ln2.bias",
        "fc.weight",
        "fc2.weight",
    }, sorted(names)
    assert not hasattr(blk, "gate_proj") and not hasattr(blk, "up_proj")
    d, f = cfg.d_model, cfg.d_ffn
    assert sum(p.numel() for p in blk.parameters()) == 4 * d * d + 2 * d * f + 4 * d
    x = torch.randn(2, 5, d, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        out = blk(x)
        h = blk.ln2(x)
        ref = x + F.linear(F.gelu(F.linear(h, blk.fc.weight)), blk.fc2.weight)
    assert float((out - ref).abs().max()) <= 1e-5


# --- (e) the config refuses the four lies ----------------------------------


@pytest.mark.parametrize(
    ("over", "named"),
    [
        ({"block_form": "kda", "n_layers": 0}, "block_form"),
        ({"d_ffn": 128}, "d_ffn"),  # 3·64·128 vs 2·64·(4·64): -25%
        ({"expert_act": "memglu"}, "memglu"),
        ({"ffn_form": "dense", "fused_glu": True}, "ffn_form"),
        ({"ffn_gated": False, "d_ffn_gelu": 3072}, "d_ffn_gelu"),
    ],
)
def test_validate_ffn_refuses_the_illegal_gated_combinations(over, named):
    # Construction is inside the block: the `block_form` half of these refusals
    # runs in `ModelConfig.__post_init__` (the loader seam), the `validate_ffn`
    # half on the explicit `validate()` call.
    with pytest.raises(LoudError) as e:
        validate(_block_cfg(**{"ffn_gated": True, **over}))
    msg = str(e.value)
    assert named in msg, msg
    assert "ffn_gated" in msg and "Escape:" in msg, msg
