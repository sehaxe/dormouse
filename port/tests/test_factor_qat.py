"""Golden gates for `factor_qat` — the compressed-forward sprint's switch.

The property under test is NOT "does it quantize" (that is
`tests/test_ternary.py`'s job) — it is that the MODEL is wired correctly:

- `none` (the default) leaves every existing number bit-identical;
- `alpha = 0` is bit-identical to `none` on every path (the annealing floor);
- the factor projection really sees {−α, 0, α} / {−α, +α} values;
- gradients still reach the fp32 masters through the STE;
- the expert bank quantizes per-EXPERT (α_e = mean(|W_e|) of that expert),
  so a banked expert and the same expert run alone agree;
- the config refusals are LOUD.

The rungs added 2026-10-16 (`mxfp4`, `ternary_s3t`) have two extra properties
each, and both are the reason their tests exist rather than re-using
`test_mxfp.py`/`test_ternary.py` (which own the FORMATS):
  * `mxfp4` pads a factor whose last dim is not a multiple of 32 PER ROW, and
    that per-row pad is exactly what makes the `[E, d, r]` bank agree with E
    standalone `[d, r]` factors — a 5461-wide `v` is the case on record;
  * `ternary_s3t` is the one rung whose mask is a drawn stream, so its
    determinism (same seed ⇒ same bits, within one run) and its limit (a new
    seed differs; the bank and the standalone path agree on the SCALE, never on
    the mask) are tested rather than described.
"""

from __future__ import annotations

import pytest
import torch

from dormouse.config import load_run
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.model.experts import DualLatentGLU, SparseMoE
from dormouse.model.mxfp import BLOCK as MX_BLOCK
from dormouse.model.mxfp import fake_quant_mxfp4
from dormouse.model.tsct import TSCTLinear, factor_qat_project, set_factor_qat_seed


def _tsct(mode: str, alpha: float = 1.0) -> TSCTLinear:
    torch.manual_seed(0)
    return TSCTLinear(32, 32, 8, dtype=torch.float32, factor_qat=mode, factor_qat_alpha=alpha)


def test_none_is_the_master_untouched():
    lin = _tsct("none")
    u, v = lin._factors()
    assert u is lin.u and v is lin.v  # not a copy — the same tensors


def test_alpha_zero_is_bit_identical_to_none():
    lin = _tsct("ternary", alpha=0.0)
    u, v = lin._factors()
    assert torch.equal(u, lin.u) and torch.equal(v, lin.v)
    x = torch.randn(4, 32)
    none = _tsct("none")
    with torch.no_grad():
        none.u.copy_(lin.u)
        none.s.copy_(lin.s)
        none.v.copy_(lin.v)
    assert torch.equal(lin(x), none(x))


def test_ternary_factors_are_on_the_ternary_grid():
    lin = _tsct("ternary")
    u, _ = lin._factors()
    alpha = lin.u.abs().mean()
    hit = (u.abs() - alpha).abs() < 1e-6
    zero = u.abs() < 1e-7
    assert bool((hit | zero).all()), "every factor value must be {−α, 0, +α}"
    assert bool(zero.any()), "a random master must produce real zeros"


def test_binary_factors_are_pm_alpha():
    lin = _tsct("binary")
    u, _ = lin._factors()
    alpha = float(lin.u.abs().mean())
    expect = torch.full_like(u, alpha)
    assert bool((((u - expect).abs() < 1e-6) | ((u + expect).abs() < 1e-6)).all())
    assert not bool((u.abs() < 1e-7).any()), "binary has no zero level on randn"


def test_annealing_is_the_exact_lerp_between_master_and_projection():
    lin = _tsct("ternary", alpha=0.5)
    u, _ = lin._factors()
    q = factor_qat_project(lin.u, "ternary", 1.0)
    assert torch.allclose(u, lin.u + 0.5 * (q - lin.u), atol=0, rtol=0)


def test_gradients_flow_to_the_masters_through_the_ste():
    lin = _tsct("ternary")
    out = lin(torch.randn(4, 32))
    out.pow(2).mean().backward()
    assert lin.u.grad is not None and lin.v.grad is not None
    assert torch.isfinite(lin.u.grad).all() and lin.u.grad.abs().sum() > 0
    assert torch.isfinite(lin.v.grad).all() and lin.v.grad.abs().sum() > 0


def _moe_cfg(**over):
    from conftest import tiny_cfg

    return tiny_cfg(n_experts=4, moe_topk=2, **over)


def test_expert_bank_quantizes_per_expert_alpha():
    cfg = _moe_cfg(factor_qat="ternary")
    model = DormouseModel(cfg)
    moe = next(m for m in model.modules() if isinstance(m, SparseMoE))
    mats = moe._banks(torch.float32)
    gate_u = mats[0]  # [E, d, r], quantized
    for e, expert in enumerate(moe.experts):
        alpha_e = expert.gate_u.abs().mean()
        block = gate_u[e]
        on_grid = ((block.abs() - alpha_e).abs() < 1e-5) | (block.abs() < 1e-7)
        assert bool(on_grid.all()), f"expert {e} not on its own ternary grid"
    # The scales are NOT quantized: gate_s/up_s stay exact.
    scales = mats[3]
    assert torch.equal(scales, torch.stack([e.gate_s for e in moe.experts]))


def test_banked_and_standalone_experts_agree_under_qat():
    """The A/B's parity: one expert through `_banks` == the same expert alone.

    Both routes must quantize with the SAME α_e (`dim=(1,2)` over the bank
    slice vs per-tensor over the 2-D factor — algebraically the same mean).
    """
    cfg = _moe_cfg(factor_qat="ternary")
    model = DormouseModel(cfg)
    moe = next(m for m in model.modules() if isinstance(m, SparseMoE))
    expert = moe.experts[2]
    x = torch.randn(6, cfg.d_model)
    _ = expert(x)  # the standalone path must run end-to-end under qat
    # The banked path for one expert: quantized stack sliced back to [d, r]/[r, f].
    gate_u_b, up_u_b, v_b, gate_s_b, up_s_b = moe._banks(torch.float32)
    q_alone = DualLatentGLU(
        expert.d,
        expert.f,
        expert.rank,
        dtype=torch.float32,
        factor_qat="ternary",
    )
    with torch.no_grad():
        q_alone.gate_u.copy_(gate_u_b[2])
        q_alone.up_u.copy_(up_u_b[2])
        q_alone.v.copy_(v_b[2])
        q_alone.gate_s.copy_(gate_s_b[2])
        q_alone.up_s.copy_(up_s_b[2])
    # q_alone re-quantizes its (already quantized) inputs — idempotent on the
    # grid up to the α of an already-quantized tensor, so compare the
    # effective forward of the BANK tensors without re-quantizing: the parity
    # claim is that expert-alone == bank-slice, checked on the effective weights.
    u_direct = factor_qat_project(expert.gate_u, "ternary", 1.0)
    assert torch.allclose(u_direct, gate_u_b[2], atol=1e-6)
    assert torch.allclose(factor_qat_project(expert.v, "ternary", 1.0), v_b[2], atol=1e-6)


def test_moe_forward_runs_and_differs_from_unquantized():
    a = DormouseModel(_moe_cfg())
    b = DormouseModel(_moe_cfg(factor_qat="ternary"))
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        la, _, _ = a(x, x)
        lb, _, _ = b(x, x)
    assert torch.isfinite(lb).all()
    assert not torch.equal(la, lb), "a quantized forward must move the logits"


def test_config_refusals_are_loud():
    with pytest.raises(LoudError, match="factor_qat"):
        load_run("small", model_overrides={"factor_qat": "int4"})
    with pytest.raises(LoudError, match="factor_qat_alpha"):
        load_run("small", model_overrides={"factor_qat": "ternary", "factor_qat_alpha": 1.5})
    with pytest.raises(LoudError, match="inert"):
        load_run("small", model_overrides={"factor_qat": "none", "factor_qat_alpha": 0.5})
    # The legal pairs load.
    load_run("small", model_overrides={"factor_qat": "ternary", "factor_qat_alpha": 0.3})
    load_run("small", model_overrides={"factor_qat": "binary"})


# ---------------------------------------------------------------------------
# mxfp4 — the MX codec as a factor rung (pad rule + bank parity)
# ---------------------------------------------------------------------------


def _mxfp4_ref(w: torch.Tensor) -> torch.Tensor:
    """The pad rule spelled out, built from the module's own pieces.

    That is the point: this test pins the RULE (zero-pad the last dim per row up
    to a multiple of 32, quantize the padded view, slice back), not a second
    copy of the e2m1 grid — which `tests/test_mxfp.py` already owns against its
    own scalar reference.
    """
    n = w.shape[-1]
    pad = (-n) % MX_BLOCK
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    return fake_quant_mxfp4(w)[..., :n]


def test_mxfp4_factors_are_the_padded_mx_projection():
    """`u [32, 8]` is padded (8 % 32 != 0), `v [8, 32]` is aligned: one rule."""
    lin = _tsct("mxfp4", alpha=1.0)
    u, v = lin._factors()
    assert torch.equal(u, _mxfp4_ref(lin.u))
    assert torch.equal(v, _mxfp4_ref(lin.v))
    assert not torch.equal(u, lin.u), "the arm must quantize, not pass through"


def test_mxfp4_annealing_is_the_exact_lerp():
    """The annealing is the ONE mechanism here; mxfp4 rides it like ternary."""
    lin = _tsct("mxfp4", alpha=0.5)
    u, _ = lin._factors()
    q = factor_qat_project(lin.u, "mxfp4", 1.0)
    assert torch.allclose(u, lin.u + 0.5 * (q - lin.u), atol=0, rtol=0)


def test_mxfp4_pads_the_5461_wide_factor_and_the_raw_module_still_refuses():
    """The real `v [r, f]` shape: f = 5461 = 170·32 + 21.

    Two claims at once: the rung runs (padded per row, finite, rule-exact), and
    `fake_quant_mxfp4` itself still LOUD-refuses a partial block — the pad is
    the CALLER's decision (`mxfp._blocks`), taken in one place for both MX
    consumers, not a change to the codec's contract.
    """
    torch.manual_seed(0)
    lin = TSCTLinear(32, 5461, 8, dtype=torch.float32, factor_qat="mxfp4")
    u, v = lin._factors()
    assert u.shape == (32, 8) and v.shape == (8, 5461)
    assert bool(torch.isfinite(v).all())
    assert torch.equal(v, _mxfp4_ref(lin.v))
    # The pad is PER ROW, so row-by-row == all-rows-at-once (bit for bit).
    per_row = torch.cat([_mxfp4_ref(lin.v[i : i + 1]) for i in range(8)])
    assert torch.equal(v, per_row)
    with pytest.raises(LoudError, match="divisible by 32"):
        fake_quant_mxfp4(lin.v)


def test_mxfp4_bank_equals_standalone_expert_aligned_and_padded():
    """The aligned-grid claim, in both regimes.

    MX's scale grid is per-32 along the LAST dim and `fake_quant_mx` pads per
    ROW, so a block never mixes two rows and the `[E, d, r]` stack quantizes to
    exactly what E separate `[d, r]` slices get. Tiny's `rank = 8` is the
    PADDED regime (8 % 32 != 0); the `[3, 16, 64]` stack below is the ALIGNED
    one. If the pad ever moves to a flat tail, this test fails on the padded
    case and says why.
    """
    cfg = _moe_cfg(factor_qat="mxfp4")
    model = DormouseModel(cfg)
    moe = next(m for m in model.modules() if isinstance(m, SparseMoE))
    assert moe.experts[0].rank % MX_BLOCK != 0, "tiny's rank is the padded regime"
    mats = moe._banks(torch.float32)
    for e, expert in enumerate(moe.experts):
        assert torch.equal(mats[0][e], factor_qat_project(expert.gate_u, "mxfp4", 1.0))
        assert torch.equal(mats[1][e], factor_qat_project(expert.up_u, "mxfp4", 1.0))
        assert torch.equal(mats[2][e], factor_qat_project(expert.v, "mxfp4", 1.0))
    torch.manual_seed(3)
    stack = torch.randn(3, 16, 64) * 0.2
    whole = factor_qat_project(stack, "mxfp4", 1.0, dim=(1, 2))
    for i in range(3):
        assert torch.equal(whole[i], factor_qat_project(stack[i], "mxfp4", 1.0))


# ---------------------------------------------------------------------------
# ternary_s3t — the drawn mask (reproducibility, grid, and its limit)
# ---------------------------------------------------------------------------


def test_s3t_same_seed_is_the_same_mask_and_a_new_seed_is_not():
    """Seed ⇒ bits, and the stream ADVANCES inside a run (that is the caveat)."""
    torch.manual_seed(0)
    w = torch.randn(64, 64)
    set_factor_qat_seed(7)
    a = factor_qat_project(w, "ternary_s3t", 1.0)
    set_factor_qat_seed(7)
    b = factor_qat_project(w, "ternary_s3t", 1.0)
    set_factor_qat_seed(8)
    c = factor_qat_project(w, "ternary_s3t", 1.0)
    assert torch.equal(a, b), "same seed must reproduce the mask exactly"
    assert not torch.equal(a, c), "a different seed must draw a different mask"
    d = factor_qat_project(w, "ternary_s3t", 1.0)
    assert not torch.equal(a, d), "one generator, advancing — a new draw per call"


def test_s3t_factors_are_zero_or_pm_the_tensor_scale():
    """S3T's grid, through the projector: `{0, ±mean(|W|)}`, both reachable."""
    set_factor_qat_seed(0)
    lin = _tsct("ternary_s3t")
    u, _ = lin._factors()
    a = float(lin.u.float().abs().mean())
    zeros = u.abs() < 1e-9
    on = (u.abs() - a).abs() < 1e-6
    assert bool((zeros | on).all()), "S3T emits 0 or ±scale, never anything else"
    assert bool(zeros.any()), "a stochastic mask must zero some elements"
    assert bool((~zeros).any())


def test_s3t_bank_and_standalone_share_the_scale_not_the_mask():
    """The ONE divergence in the projector, pinned so it cannot silently widen.

    The bank's per-slice scale is the expert's own absmean (as for `ternary`),
    but the Bernoulli draw is one `[E, ·, ·]` stream against the standalone
    path's per-factor draws — no RNG layout makes those identical, so parity is
    on the SCALE only, never bit-for-bit. A future change that quietly made the
    bank use a whole-tensor scale would fail the `a_e` assert below.
    """
    set_factor_qat_seed(0)
    cfg = _moe_cfg(factor_qat="ternary_s3t")
    model = DormouseModel(cfg)
    moe = next(m for m in model.modules() if isinstance(m, SparseMoE))
    gates = moe._banks(torch.float32)[0]
    for e, expert in enumerate(moe.experts):
        a_e = float(expert.gate_u.float().abs().mean())
        block = gates[e]
        on = ((block.abs() - a_e).abs() < 1e-5) | (block.abs() < 1e-7)
        assert bool(on.all()), f"expert {e} is not on its own S3T scale"
    mine = factor_qat_project(moe.experts[0].gate_u, "ternary_s3t", 1.0)
    assert not torch.equal(gates[0], mine), "the mask differs; the scale is the claim"


@pytest.mark.parametrize("mode", ["mxfp4", "ternary_s3t"])
def test_each_new_factor_rung_runs_is_finite_and_moves_the_logits(mode):
    a = DormouseModel(_moe_cfg())
    b = DormouseModel(_moe_cfg(factor_qat=mode))
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        la, _, _ = a(x, x)
        lb, _, _ = b(x, x)
    assert bool(torch.isfinite(lb).all())
    assert not torch.equal(la, lb), f"{mode} left the logits untouched"


@pytest.mark.parametrize("mode", ["mxfp4", "ternary_s3t"])
def test_new_factor_rungs_keep_the_gradient_on_the_master(mode):
    set_factor_qat_seed(0)
    lin = _tsct(mode)
    lin(torch.randn(4, 32)).pow(2).mean().backward()
    for name in ("u", "v"):
        g = getattr(lin, name).grad
        assert g is not None and bool(torch.isfinite(g).all()) and g.abs().sum() > 0, name


def test_config_new_factor_rungs_and_the_seed_inertness():
    load_run("small", model_overrides={"factor_qat": "mxfp4"})
    load_run("small", model_overrides={"factor_qat": "ternary_s3t"})
    load_run("small", model_overrides={"factor_qat": "ternary_s3t", "factor_qat_seed": 7})
    with pytest.raises(LoudError, match="inert"):
        load_run("small", model_overrides={"factor_qat": "ternary", "factor_qat_seed": 7})
    with pytest.raises(LoudError, match="inert"):
        load_run("small", model_overrides={"factor_qat": "mxfp4", "factor_qat_seed": 3})
    with pytest.raises(LoudError, match="factor_qat"):
        load_run("small", model_overrides={"factor_qat": "mxfp3"})
