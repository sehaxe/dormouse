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

The rung added 2026-10-16 (`ternary_s3t`) has an extra property, and it is
the reason its tests exist rather than re-using `test_ternary.py` (which owns
the FORMAT): `ternary_s3t` is the one rung whose mask is a drawn stream, so
its determinism (same seed ⇒ same bits, within one run) and its limit (a new
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
    """The A/B's parity, run END-TO-END: one expert through `_banks` == alone.

    Both routes must quantize with the SAME α_e (`dim=(1,2)` over the bank
    slice vs per-tensor over the 2-D factor — algebraically the same mean), so
    the bank slice IS the expert's own quantized tensors and an expert built
    from them must reproduce the qat expert's forward. The banked arm feeds
    those same tensors to `bmm`, so this is the arithmetic the banked path
    runs, per expert — not a projection compared with itself.
    """
    cfg = _moe_cfg(factor_qat="ternary")
    model = DormouseModel(cfg)
    moe = next(m for m in model.modules() if isinstance(m, SparseMoE))
    expert = moe.experts[2]
    x = torch.randn(6, cfg.d_model)
    with torch.no_grad():
        alone = expert(x)  # the standalone path, quantizing with its own α_e
    # The banked path for one expert: quantized stack sliced back to [d, r]/[r, f].
    gate_u_b, up_u_b, v_b, gate_s_b, up_s_b = moe._banks(torch.float32)
    # `factor_qat="none"`: the slice is ALREADY quantized, and re-quantizing it
    # would take the absmean of a quantized tensor — a second α the bank never
    # applies.
    q_alone = DualLatentGLU(expert.d, expert.f, expert.rank, dtype=torch.float32, factor_qat="none")
    with torch.no_grad():
        q_alone.gate_u.copy_(gate_u_b[2])
        q_alone.up_u.copy_(up_u_b[2])
        q_alone.v.copy_(v_b[2])
        q_alone.gate_s.copy_(gate_s_b[2])
        q_alone.up_s.copy_(up_s_b[2])
        banked = q_alone(x)
    assert banked.shape == alone.shape
    assert torch.equal(banked, alone), (
        "the bank slice must BE the expert's own quantized forward: "
        f"max |Δ| = {(banked - alone).abs().max().item():.3e}"
    )


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
    load_run("small", model_overrides={"factor_qat": "ternary_s3t"})


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


@pytest.mark.parametrize("mode", ["ternary_s3t"])
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


@pytest.mark.parametrize("mode", ["ternary_s3t"])
def test_new_factor_rungs_keep_the_gradient_on_the_master(mode):
    set_factor_qat_seed(0)
    lin = _tsct(mode)
    lin(torch.randn(4, 32)).pow(2).mean().backward()
    for name in ("u", "v"):
        g = getattr(lin, name).grad
        assert g is not None and bool(torch.isfinite(g).all()) and g.abs().sum() > 0, name


def test_config_ternary_s3t_and_the_seed_inertness():
    load_run("small", model_overrides={"factor_qat": "ternary_s3t"})
    load_run("small", model_overrides={"factor_qat": "ternary_s3t", "factor_qat_seed": 7})
    with pytest.raises(LoudError, match="inert"):
        load_run("small", model_overrides={"factor_qat": "ternary", "factor_qat_seed": 7})
    with pytest.raises(LoudError, match="factor_qat"):
        load_run("small", model_overrides={"factor_qat": "mxfp3"})
