"""Golden gates for the a4.8 ACTIVATION quantization arm (`model/act_quant.py`).

The property under test is BOTH layers, because each has its own failure mode:

- the FORMULA: the scale maps a block's max onto the FORMAT's max, so all 8
  e2m1 magnitudes are reachable (the old `block_max` scale made a 3-level
  quantizer with a 4-bit label — AGENTS §3.7 act-quant);
- the WIRING: `act_quant` really reaches the forward (`DualLatentGLU`, the
  expert bank, `TSCTLinear`) and the a4.8 attention upgrade
  (`attn_format`) cannot be bypassed by a 4-bit format on an attention path.

Sources are named per test; the error bounds asserted are the format's own
analytic ones (`|x_q − x| ≤ block_max / 2·L`), not tuned constants.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from dormouse.config import ModelConfig, load_preset, load_run, parse_preset
from dormouse.instrument import LoudError
from dormouse.model import act_quant as aq
from dormouse.model.act_quant import (
    ACT_FORMATS,
    attn_format,
    fake_quant_act,
    max_value,
    quantize_act,
)
from dormouse.model.experts import DualLatentGLU, SparseMoE
from dormouse.model.hadamard import v2_quantize
from dormouse.model.mxfp import BLOCK as MX_BLOCK
from dormouse.model.mxfp import fake_quant_mxfp8
from dormouse.model.ternary import activation_quant_8bit
from dormouse.model.tsct import TSCTLinear


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*shape)


# --------------------------------------------------------------------------
# the formula
# --------------------------------------------------------------------------


def test_max_value_is_the_format_max_not_one():
    """`act_quant.rs:54-59`: 6 for e2m1, `2^(b-1)-1` for the integers.

    The value 1.0 is called out explicitly: normalizing a block onto 1 is the
    defect this module exists to not repeat.
    """
    assert max_value("fp4") == 6.0
    assert max_value("int4") == 7.0
    assert max_value("int8") == 127.0
    assert max_value("fp4") != 1.0
    for fmt in ACT_FORMATS:
        if fmt == "none":
            with pytest.raises(LoudError, match="no format max"):
                max_value(fmt)
        else:
            assert max_value(fmt) > 1.0


def test_int8_per_token_error_bound_and_grid():
    """Analytic bound: `|x_q − x| ≤ block_max / (2·127)`; and the grid is 8-bit.

    Measured (this box, torch 2.13, fp32, seed 0, `randn [16, 128]`): rms err
    **6.98e-3** (rel 6.84e-3), max abs err **1.60e-2** against the bound
    `max|x|/(2·127) = 1.61e-2`.
    """
    x = _randn(16, 128)
    q = quantize_act(x, "int8")
    assert q.shape == x.shape and q.dtype == x.dtype
    block_max = x.abs().amax(dim=-1, keepdim=True)
    err = (q - x).abs()
    assert bool((err <= block_max / (2 * 127) + 1e-6).all())
    assert float(err.max()) > 0.0, "quantization must actually quantize"
    # The dequantized values sit ON the code grid: `q / scale` is an integer.
    scale = block_max / 127
    codes = q / scale
    assert torch.allclose(codes, codes.round(), atol=1e-4)
    assert bool((codes.abs() <= 127 + 1e-4).all())
    rms = float(torch.sqrt((err**2).mean()))
    assert rms < 0.02, f"int8 rms error {rms} (measured 6.98e-3)"


def test_int4_per_token_error_bound_and_levels():
    """int4: 7 levels, so the bound is `block_max / 14` — 18x int8's step.

    Measured (same protocol): rms err **1.267e-1** (rel 1.243e-1), max abs err
    **2.917e-1** against the bound `max|x|/14 = 2.930e-1`.
    """
    x = _randn(16, 128)
    q = quantize_act(x, "int4")
    block_max = x.abs().amax(dim=-1, keepdim=True)
    err = (q - x).abs()
    assert bool((err <= block_max / (2 * 7) + 1e-6).all())
    codes = q / (block_max / 7)
    assert bool((codes.abs() <= 7 + 1e-4).all())
    # The whole symmetric range is reachable, on a ramp with the block max at 1.
    ramp = torch.linspace(0, 1, 4096).reshape(1, -1)
    out = quantize_act(ramp, "int4")
    want = [k / 7 for k in range(8)]
    missing = [v for v in want if not bool((out - v).abs().min() < 1e-6)]
    assert not missing, f"int4 never emitted {missing}"
    rms = float(torch.sqrt((err**2).mean()))
    assert rms < 0.15, f"int4 rms error {rms} (measured 1.267e-1)"


def test_fp4_reaches_all_eight_magnitudes_and_scales_onto_6():
    """THE anti-3-level test: every e2m1 magnitude must come back.

    A block whose max is `M` gets `scale = M/6`, so a value placed at `L·M/6`
    must dequantize to exactly that. The old `scale = block_max` normalized to
    [-1, 1] and the magnitudes above 1 were dead code — a ~3-level quantizer
    (AGENTS §3.7 act-quant; `act_quant.rs:9-19`).
    """
    mags = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    m = 3.0
    row = torch.tensor([[m * level / 6.0 for level in mags]])
    row = torch.cat([row, -row], dim=-1)  # the sign mirror too
    out = quantize_act(row, "fp4")
    want = [m * level / 6.0 for level in mags]
    missing = [v for v in want if not bool((out - v).abs().min() < 1e-6)]
    assert not missing, f"fp4 never emitted {missing} of {want}: the scale is not reaching 6.0"
    missing_neg = [v for v in want if not bool((out + v).abs().min() < 1e-6)]
    assert not missing_neg, f"fp4 lost the negative mirror: {missing_neg}"
    # And the block max lands ON 6.0 (scale = block_max/6), not on 1.
    assert abs(float(out.abs().max()) - m) < 1e-6
    assert abs(float(out.abs().max()) - 1.0) > 0.5, "the scale normalized onto 1, not 6"


def test_fp4_ramp_emits_the_whole_grid_and_never_0_75():
    """The burn `fp4_reaches_every_level_through_the_public_quantizer` port.

    A 4096-step ramp over [0, 1] in ONE block: the block max is 1, so level `L`
    comes back as `L/6` — including 1.5/6, 2/6, 3/6, 4/6 and 6/6, which the
    broken scale could not produce. 0.75 is not a level of e2m1 (`act_quant.rs:
    137-146`) and must never appear.
    """
    ramp = torch.linspace(0, 1, 4096).reshape(1, -1)
    out = quantize_act(ramp, "fp4").flatten()
    levels = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    missing = [lv / 6 for lv in levels if not bool((out - lv / 6).abs().min() < 1e-6)]
    assert not missing, f"fp4 through quantize_act never emitted {missing}"
    assert not bool((out - 0.75 / 6).abs().min() < 1e-7), "0.75 is not an e2m1 level"
    assert abs(float(out.max()) - 1.0) < 1e-6, "the ramp max must map onto the format max"


def test_fp4_ties_go_to_the_even_code():
    """`act_quant.rs:150-170`'s tie table, pinned as literals.

    Ties-up (a plain `>=` ladder) disagrees with the reference at 4 of these 7
    and is 50 % wrong at 5.0, where the block scale puts the most-used values.
    """
    vals = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    want = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
    row = torch.tensor([[6.0] + vals + [-v for v in vals]])
    out = quantize_act(row, "fp4").flatten()
    for i, (v, w) in enumerate(zip(vals, want, strict=True)):
        assert abs(out[1 + i].item() - w) < 1e-6, f"tie {v} → {out[1 + i].item()}, want {w}"
    # The negative mirror rounds the same way (sign is never lost).
    for i, w in enumerate(want):
        assert abs(out[1 + len(vals) + i].item() + w) < 1e-6


def test_non_e2m1_output_is_impossible():
    """Every fp4 output magnitude is a member of the grid (a golden of the FORMAT)."""
    x = _randn(8, 256, seed=3) * 5.0
    out = quantize_act(x, "fp4")
    scale = x.abs().amax(dim=-1, keepdim=True) / 6.0
    mag = (out / scale).abs().flatten()
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    dist = (mag.unsqueeze(-1) - grid).abs().min(dim=-1).values
    assert bool((dist < 1e-4).all()), "an fp4 output is not on the e2m1 grid"


def test_group_scales_match_a_hand_computed_per_block_reference():
    """`group=32` on `[4, 128]` == per-32-block absmax scales, computed by hand."""
    x = _randn(4, 128)
    got = quantize_act(x, "int4", 32)
    ref = []
    for row in x:
        blocks = []
        for start in range(0, 128, 32):
            blk = row[start : start + 32]
            s = blk.abs().max() / 7
            blocks.append((blk / s).round().clamp(-7, 7) * s)
        ref.append(torch.cat(blocks))
    ref = torch.stack(ref)
    assert torch.equal(got, ref), "group scales are not per-32-block absmax"
    # The test would pass trivially with the group ignored; it must NOT agree
    # with the per-token scale on this data.
    assert not torch.allclose(got, quantize_act(x, "int4", 0))


def test_group_equal_to_the_last_dim_is_per_token():
    """`group == d` is one block per row: exactly `group=0`, bit for bit."""
    x = _randn(6, 128, seed=7)
    for fmt in ("int8", "int4", "fp4"):
        assert torch.equal(quantize_act(x, fmt, 128), quantize_act(x, fmt, 0))


def test_quantization_is_idempotent():
    """`q(q(x)) == q(x)` — the property that lets two seams both be armed.

    `SparseMoE` quantizes the activation before the bank and `DualLatentGLU`
    quantizes its own input, so a shared expert sees an already-quantized `x`:
    the block max of a quantized block is the block max, so the second pass is
    the identity. If this ever fails, the double-armed path drifts.
    """
    x = _randn(4, 128, seed=11)
    for fmt in ("int8", "int4", "fp4"):
        for group in (0, 32):
            once = fake_quant_act(x, fmt, group)
            assert torch.equal(fake_quant_act(once, fmt, group), once), (fmt, group)


# --------------------------------------------------------------------------
# the refusals (LOUD: cause + escape)
# --------------------------------------------------------------------------


def test_unknown_format_is_loud():
    x = _randn(2, 8)
    for fmt in ("int3", "fp8", "INT8", ""):
        with pytest.raises(LoudError, match="not an activation format"):
            quantize_act(x, fmt)
        with pytest.raises(LoudError, match="not an activation format"):
            fake_quant_act(x, fmt)


def test_group_must_divide_the_last_dim_and_be_non_negative():
    x = _randn(2, 128)
    with pytest.raises(LoudError, match="does not divide the last dim 128"):
        quantize_act(x, "int4", 48)
    with pytest.raises(LoudError, match="does not divide the last dim 128"):
        quantize_act(x, "int4", 130)  # > d is the same failure, named the same way
    with pytest.raises(LoudError, match="negative"):
        quantize_act(x, "int4", -8)
    with pytest.raises(LoudError, match="Escape"):
        quantize_act(torch.randn(2), "int4", 4)  # 1-dim, group > d


def test_non_finite_and_non_float_inputs_are_loud():
    bad = _randn(2, 8)
    bad[0, 3] = float("nan")
    aq._GUARD.reset()
    with pytest.raises(LoudError, match="non-finite"):
        quantize_act(bad, "int8")
    inf = _randn(2, 8)
    inf[1, 0] = float("inf")
    aq._GUARD.reset()
    with pytest.raises(LoudError, match="non-finite"):
        fake_quant_act(inf, "fp4")
    with pytest.raises(LoudError, match="float tensors"):
        quantize_act(torch.tensor([[1, 2], [3, 4]]), "int8")
    with pytest.raises(LoudError, match="LAST dim"):
        quantize_act(torch.tensor(3.0), "int8")


def test_finite_guard_is_cadenced_and_the_forward_takes_no_host_read(monkeypatch):
    """The 2026-10-19 audit M7: one D2H per MoE forward became a cadence.

    The guard still refuses (it runs on the first calls), and a forward after
    that window performs no `.item()` / `bool(any())` — every such read goes
    through `Tensor.item`, so raising from it is the gate.
    """
    x = _randn(4, 64)
    aq._GUARD.reset()
    with pytest.raises(LoudError, match="non-finite"):
        fake_quant_act(torch.full((1, 8), float("nan")), "fp4")
    aq._GUARD.calls = aq._GUARD.first  # the next call is NOT the cadence
    assert not aq._GUARD.due()

    def _no_item(self, *args, **kwargs):
        raise AssertionError("the per-forward path took a host read (.item())")

    def _no_bool(self):
        raise AssertionError("the per-forward path branched on a device value (bool)")

    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "item", _no_item)
        # `if bool(tensor)` goes through `__bool__` in C, NOT through the Python
        # `item` attribute — both must raise or the gate cannot see the guard.
        m.setattr(torch.Tensor, "__bool__", _no_bool)
        y = fake_quant_act(x, "fp4", 32)
        y8 = fake_quant_act(x, "int8")
    assert bool(torch.isfinite(y).all()) and bool(torch.isfinite(y8).all())


def test_none_is_the_identity_and_all_zero_blocks_are_safe():
    x = _randn(2, 8)
    assert fake_quant_act(x, "none") is x
    assert quantize_act(x, "none") is x
    zeros = torch.zeros(3, 16)
    for fmt in ("int8", "int4", "fp4"):
        out = quantize_act(zeros, fmt, 8)
        assert bool(torch.isfinite(out).all()) and float(out.abs().max()) == 0.0


# --------------------------------------------------------------------------
# a4.8: the attention path is never below 8 bits
# --------------------------------------------------------------------------


def test_attn_format_upgrades_every_4_bit_format():
    """`act_quant.rs:38-46`, unconditional: fp4/int4 → int8, int8 → int8.

    The two imported families are read the way their own source reads them:
    `mxfp8` is already 8 bits (e4m3) and stays itself, and `v2_int4` upgrades
    INSIDE BitNet v2 (arXiv 2504.18415's activation format is the rotated
    integer arm at 8 bits) rather than dropping to the unrelated plain `int8`.
    """
    assert attn_format("fp4") == "int8"
    assert attn_format("int4") == "int8"
    assert attn_format("int8") == "int8"
    assert attn_format("mxfp8") == "mxfp8"
    assert attn_format("v2_int4") == "v2_int8"
    assert attn_format("v2_int8") == "v2_int8"
    assert attn_format("none") == "none"
    with pytest.raises(LoudError, match="not a format"):
        attn_format("int3")


def test_tsct_attn_path_uses_the_upgraded_format():
    """The flag that carries the a4.8 rule into a projection on the attention path.

    `attn_path=True` with `act_quant="fp4"` must produce EXACTLY the `int8`
    forward on the same weights — that is the upgrade, not a rounding of it.
    """
    torch.manual_seed(0)
    a = TSCTLinear(32, 32, 8, dtype=torch.float32, act_quant="fp4")
    b = TSCTLinear(32, 32, 8, dtype=torch.float32, act_quant="fp4", attn_path=True)
    c = TSCTLinear(32, 32, 8, dtype=torch.float32, act_quant="int8")
    b.load_state_dict(a.state_dict())
    c.load_state_dict(a.state_dict())
    x = _randn(4, 32, seed=5)
    with torch.no_grad():
        ya, yb, yc = a(x), b(x), c(x)
    assert torch.equal(yb, yc), "attn_path must run the int8 format"
    assert not torch.equal(ya, yb), "fp4 and the upgraded int8 must differ"
    assert bool(torch.isfinite(yb).all())


# --------------------------------------------------------------------------
# the STE and the wiring
# --------------------------------------------------------------------------


def test_ste_gradient_is_one():
    """Forward quantized, backward identity: `d/dx = 1` (`act_quant.rs:103-104`)."""
    for fmt in ("int8", "int4", "fp4"):
        for group in (0, 32):
            x = _randn(4, 64).requires_grad_(True)
            y = fake_quant_act(x, fmt, group)
            y.sum().backward()
            assert x.grad is not None
            assert torch.equal(x.grad, torch.ones_like(x)), (fmt, group)


def test_int8_per_token_matches_activation_quant_8bit_to_fp_rounding():
    """The two 8-bit implementations are ONE formula (`ternary.activation_quant_8bit`).

    Its scale is `block_max` with an explicit `·127`; this module's is
    `block_max/127` (the burn spelling, `act_quant.rs:88-89`) — the same map in
    real arithmetic, differing only by the ORDER of the two fp32 roundings
    (`x/b·127` vs `x/(b/127)`) and by the ε guard (1e-5 vs 1e-8). Measured on
    seed 13, `[8, 128]`: max abs deviation **2.38e-7** (rel 1.6e-7 ≈ 2 fp32 ULP
    at |a| ~ 1); on a degenerate spike row the two are bit-identical. The test is
    the reason one implementation can serve both call sites.
    """
    x = _randn(8, 128, seed=13)
    mine, theirs = fake_quant_act(x, "int8", 0), activation_quant_8bit(x)
    assert torch.allclose(mine, theirs, rtol=1e-6, atol=0)
    assert float((mine - theirs).abs().max()) < 1e-6
    # A degenerate row (a single huge spike, everything else zero) is exact.
    spike = torch.zeros(2, 128)
    spike[0, 0] = 1e4
    assert torch.equal(fake_quant_act(spike, "int8", 0), activation_quant_8bit(spike))


def test_duallatentglu_quantizes_the_xu_seam():
    """The expert seam: `DualLatentGLU.forward` must feed quantized activations.

    The expected value is computed from `fake_quant_act` directly, so this pins
    the format AND the group reaching the matmuls, not just "the output moved".
    """
    torch.manual_seed(0)
    d, f, r = 64, 32, 8
    m = DualLatentGLU(d, f, r, dtype=torch.float32, act_quant="int4", act_group=32)
    x = _randn(4, d, seed=17)
    with torch.no_grad():
        got = m(x)
        xq = fake_quant_act(x, "int4", 32)
        g = (xq @ m.gate_u) * m.gate_s
        u = (xq @ m.up_u) * m.up_s
        want = (torch.nn.functional.silu(g) * u) @ m.v
    assert torch.equal(got, want)
    plain = DualLatentGLU(d, f, r, dtype=torch.float32)
    plain.load_state_dict(m.state_dict())
    with torch.no_grad():
        assert not torch.equal(plain(x), got), "the arm must change the expert's output"


def test_quantization_moves_a_model_forward_and_stays_finite():
    """A vs B, same weights: only `act_quant` differs, the logits must move."""
    from conftest import tiny_model

    a = tiny_model()
    b = tiny_model(act_quant="int8")
    c = tiny_model(act_quant="int4", act_group=32)
    b.load_state_dict(a.state_dict())
    c.load_state_dict(a.state_dict())
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        la, _, _ = a(x, x)
        lb, _, _ = b(x, x)
        lc, _, _ = c(x, x)
    assert torch.isfinite(lb).all() and torch.isfinite(lc).all()
    assert not torch.equal(la, lb), "an armed forward must move the logits"
    assert not torch.equal(lb, lc), "int4 and int8 are different formats"
    # The routed/banked expert path is the one that must carry the arm: the
    # counter says it ran, so the arm cannot be silently inert here.
    a2, b2, c2 = tiny_model(), tiny_model(act_quant="int8"), tiny_model(act_quant="int4")
    b2.load_state_dict(a2.state_dict())
    c2.load_state_dict(a2.state_dict())
    with torch.no_grad():
        _, _, cnt = b2(x, x)
    assert cnt.get("act_quant_arms", 0) > 0, "the activation arm did not run in the MoE"


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def test_config_default_is_none_and_presets_keep_it():
    assert ModelConfig().act_quant == "none"
    assert ModelConfig().act_group == 0
    assert load_preset("small").act_quant == "none"
    assert load_preset("b1").act_quant == "none"
    assert load_run("small", model_overrides={"act_quant": "int4"}).model.act_quant == "int4"
    grouped = load_run("small", model_overrides={"act_quant": "fp4", "act_group": 32})
    assert (grouped.model.act_quant, grouped.model.act_group) == ("fp4", 32)


def test_config_unknown_act_quant_is_loud():
    with pytest.raises(LoudError, match="not an activation format"):
        parse_preset('d_model = 64\nn_heads = 4\nhead_dim = 16\nact_quant = "int3"\n')
    with pytest.raises(LoudError, match="ACT_FORMATS|Known"):
        load_run("small", model_overrides={"act_quant": "ternary"})


def test_config_act_group_validation():
    with pytest.raises(LoudError, match="negative"):
        load_run("small", model_overrides={"act_quant": "int4", "act_group": -4})
    with pytest.raises(LoudError, match="must divide d_model"):
        load_run("small", model_overrides={"act_quant": "int4", "act_group": 100})
    with pytest.raises(LoudError, match="inert"):
        load_run("small", model_overrides={"act_group": 32})
    # The legal pairs load: 0, a divisor, and `group == d_model`.
    load_run("small", model_overrides={"act_quant": "int4", "act_group": 0})
    load_run("small", model_overrides={"act_quant": "int4", "act_group": 256})
    load_run("small", model_overrides={"act_quant": "int8", "act_group": 768})


def test_moe_arm_reaches_every_expert_path():
    """The banked dense path AND the standalone module both carry the arm."""
    from conftest import tiny_cfg

    cfg = tiny_cfg(act_quant="int8")
    moe = SparseMoE(cfg)
    assert all(e.act_quant == "int8" for e in moe.experts)
    assert moe.shared.act_quant == "int8"
    assert moe.act_quant == "int8"
    x = _randn(4, cfg.d_model)
    with torch.no_grad():
        before = moe.experts[0](x)
        moe.experts[0].act_quant = "none"
        after = moe.experts[0](x)
    assert not torch.equal(before, after), "the module's own seam must quantize"


def test_banked_dense_bmm_quantizes_the_same_activation_as_the_module():
    """The banked branch (`topk=0`, budget) reaches the factor matmuls directly.

    `SparseMoE`'s dense-banked branch multiplies `x` by the stacked `[E, d, r]`
    factors with `bmm` — `DualLatentGLU.forward` never runs there, so the arm has
    to be applied in `SparseMoE` too. The parity claim: the banked blend equals
    the per-expert loop under the SAME arm (both quantize the same per-row/group
    scale), which is what makes the arm's A/B comparable across dispatch arms.
    """
    from conftest import tiny_cfg

    cfg = tiny_cfg(moe_topk=0, act_quant="int4", act_group=32)
    moe = SparseMoE(cfg)
    x = _randn(4, cfg.d_model, seed=23)
    with torch.no_grad():
        got = moe(x)
        probs = torch.softmax(moe.router(x), -1)
        weighted = torch.zeros(4, cfg.d_ffn, dtype=torch.float32)
        for ex, expert in enumerate(moe.experts):
            weighted += expert(x) * probs[:, ex : ex + 1]
        weighted = weighted + moe.shared(x)
        dt = moe.experts[0].compute_dtype
        want = (weighted.to(dt) @ moe.down.weight.t().to(dt)).to(torch.float32)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5), "banked != module under the arm"
    # And the arm is not inert in this branch: same weights, arm off ⇒ different.
    off = SparseMoE(tiny_cfg(moe_topk=0))
    off.load_state_dict(moe.state_dict())
    with torch.no_grad():
        assert not torch.equal(off(x), got), "the banked branch ignored the arm"


# --------------------------------------------------------------------------
# the imported families: mxfp8 (OCP MX) and v2_int4/v2_int8 (BitNet v2 FWHT)
# --------------------------------------------------------------------------


def _mxfp8_ref(x: torch.Tensor) -> torch.Tensor:
    """The pad rule spelled out: zero-pad the last dim per row to a 32-multiple."""
    n = x.shape[-1]
    pad = (-n) % MX_BLOCK
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return fake_quant_mxfp8(x)[..., :n]


def test_mxfp8_on_a_768_activation_is_exactly_the_mx_module():
    """The wiring claim: the seam hands `x` to `mxfp`, it does not reimplement it.

    `d_model = 768 = 24·32` is the real preset width, so this is the no-pad
    path; the assertion is bit-equality with the module's own fake-quant, i.e.
    the seam adds NOTHING (no second STE, no re-scale).
    """
    x = _randn(6, 768, seed=31)
    got = fake_quant_act(x, "mxfp8")
    assert torch.equal(got, fake_quant_mxfp8(x))
    assert torch.equal(quantize_act(x, "mxfp8"), fake_quant_mxfp8(x))
    assert not torch.equal(got, x), "the arm must quantize"


def test_mxfp8_pads_a_non_multiple_of_32_last_dim_and_the_module_still_refuses():
    """1000 = 31·32 + 8: padded per row here, LOUD in `mxfp` itself."""
    x = _randn(4, 1000, seed=5)
    got = fake_quant_act(x, "mxfp8")
    assert got.shape == x.shape and bool(torch.isfinite(got).all())
    assert torch.equal(got, _mxfp8_ref(x))
    per_row = torch.cat([_mxfp8_ref(x[i : i + 1]) for i in range(4)])
    assert torch.equal(got, per_row), "the pad must be per row, not one flat tail"
    with pytest.raises(LoudError, match="divisible by 32"):
        fake_quant_mxfp8(x)


def test_mxfp8_model_forward_runs_and_moves_the_logits():
    from conftest import tiny_model

    a, b = tiny_model(), tiny_model(act_quant="mxfp8")
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        la, _, _ = a(x, x)
        lb, _, cnt = b(x, x)
    assert bool(torch.isfinite(lb).all())
    assert not torch.equal(la, lb)
    assert cnt.get("act_quant_arms", 0) > 0, "the arm must reach the MoE, not idle"


def test_v2_int4_forward_equals_v2_quantize_through_the_same_weights():
    """Wiring, not reimplementation: the expert seam feeds `v2_quantize(x, 4)`.

    The expected value is computed from the module directly, so this pins the
    FORMAT and the rotation reaching the factor matmuls — a seam that quantized
    a different width or skipped the rotation would fail here.
    """
    torch.manual_seed(0)
    d, f, r = 64, 32, 8
    m = DualLatentGLU(d, f, r, dtype=torch.float32, act_quant="v2_int4")
    x = _randn(4, d, seed=17)
    with torch.no_grad():
        got = m(x)
        xq = v2_quantize(x, 4)
        g = (xq @ m.gate_u) * m.gate_s
        u = (xq @ m.up_u) * m.up_s
        want = (F.silu(g) * u) @ m.v
    assert torch.equal(got, want)
    # `v2_int8` is a different arm on the same weights (a 4x finer grid).
    m8 = DualLatentGLU(d, f, r, dtype=torch.float32, act_quant="v2_int8")
    m8.load_state_dict(m.state_dict())
    x8 = v2_quantize(x, 8)
    want8 = (F.silu((x8 @ m.gate_u) * m.gate_s) * ((x8 @ m.up_u) * m.up_s)) @ m.v
    with torch.no_grad():
        got8 = m8(x)
    assert torch.equal(got8, want8)
    assert not torch.equal(got, got8)


def test_v2_formats_refuse_a_non_power_of_two_last_dim():
    """96 = 3·32: the FWHT has no basis there, and padding would change it.

    LOUD with the escape named (hadamard._check), unlike MX's tail pad — the
    MX pad only decides which tail columns share a scale, the FWHT pad decides
    the rotation itself.
    """
    x = _randn(2, 96, seed=3)
    for fmt in ("v2_int4", "v2_int8"):
        with pytest.raises(LoudError, match=r"2\^k"):
            fake_quant_act(x, fmt)
        # The escape is named (pad to 128 or pick a 2^k width) — the assertion
        # is on the PAD path being the escape, since this rung refuses it.
        with pytest.raises(LoudError, match="power-of-two width"):
            quantize_act(x, fmt)


def test_own_grid_formats_ste_gradient_is_one_and_there_is_only_one_ste():
    """`mxfp8`/`v2_*` return their module's own STE — the seam must not re-wrap.

    `fake_quant_act` hands those three through unchanged (`act_quant.py`), so the
    gradient is the identity exactly ONCE: `d/dx = 1`, and the forward value
    equals the module's fake-quant (no second `x + (dq − x).detach()` hiding
    which layer owns the estimator).
    """
    for fmt in ("mxfp8", "v2_int4", "v2_int8"):
        x = _randn(4, 64, seed=41).requires_grad_(True)
        y = fake_quant_act(x, fmt)
        y.sum().backward()
        assert x.grad is not None and torch.equal(x.grad, torch.ones_like(x)), fmt
        assert torch.equal(y.detach(), quantize_act(x.detach(), fmt)), fmt


def test_own_grid_formats_refuse_a_group_instead_of_ignoring_it():
    """Their grid is not the caller's to choose: `act_group` is LOUD-inert."""
    x = _randn(4, 64, seed=7)
    for fmt in ("mxfp8", "v2_int4", "v2_int8"):
        with pytest.raises(LoudError, match="INERT"):
            fake_quant_act(x, fmt, 32)
    # ... and the a4.8 formats still take a group (the control for that rule).
    assert not torch.equal(fake_quant_act(x, "int4", 32), fake_quant_act(x, "int4", 0))


@pytest.mark.parametrize("fmt", ["mxfp8", "v2_int4", "v2_int8"])
def test_each_new_act_rung_runs_is_finite_and_moves_the_logits(fmt):
    from conftest import tiny_model

    a, b = tiny_model(), tiny_model(act_quant=fmt)
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        la, _, _ = a(x, x)
        lb, _, cnt = b(x, x)
    assert bool(torch.isfinite(lb).all())
    assert not torch.equal(la, lb), f"{fmt} left the logits untouched"
    assert cnt.get("act_quant_arms", 0) > 0, f"{fmt} did not reach the MoE bank"


def test_config_new_act_rungs_their_inert_combinations_and_the_2k_rule():
    load_run("small", model_overrides={"act_quant": "mxfp8"})
    # `small` is d_model=768 = 3·2^8: the rotated arms refuse at load, not on
    # the first forward, and the escape is a power-of-two width.
    with pytest.raises(LoudError, match=r"2\^k"):
        load_run("small", model_overrides={"act_quant": "v2_int4"})
    with pytest.raises(LoudError, match=r"2\^k"):
        load_run("small", model_overrides={"act_quant": "v2_int8"})
    ok = load_run(
        "small",
        model_overrides={"act_quant": "v2_int8", "d_model": 512, "n_heads": 8},
    )
    assert ok.model.act_quant == "v2_int8"
    for fmt in ("mxfp8", "v2_int4", "v2_int8"):
        with pytest.raises(LoudError, match="inert"):
            load_run("small", model_overrides={"act_quant": fmt, "act_group": 32})
    with pytest.raises(LoudError, match="not an activation format"):
        load_run("small", model_overrides={"act_quant": "mxfp4"})
    assert set(ACT_FORMATS) == {
        "none",
        "int8",
        "int4",
        "fp4",
        "mxfp8",
        "v2_int4",
        "v2_int8",
    }
    assert max_value("mxfp8") == 448.0 and max_value("v2_int4") == 7.0
