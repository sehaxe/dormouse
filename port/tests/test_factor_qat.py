"""Golden gates for `factor_qat` — the compressed-forward sprint's switch.

The property under test is NOT "does it quantize" (that is
`tests/test_ternary.py`'s job) — it is that the MODEL is wired correctly:

- `none` (the default) leaves every existing number bit-identical;
- `alpha = 0` is bit-identical to `none` on every path (the annealing floor);
- the factor projection really sees {−α, 0, α} / {−α, +α} values;
- gradients still reach the fp32 masters through the STE;
- the expert bank quantizes per-EXPERT (α_e = mean(|W_e|) of that expert),
  so a banked expert and the same expert run alone agree;
- the config refusals are LOUD, and a rung the 2026-11-13 cut removed
  (`ternary_s3t`) refuses with its removal, not as a typo.
"""

from __future__ import annotations

import pytest
import torch

from dormouse.config import load_run
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.model.experts import DualLatentGLU, SparseMoE
from dormouse.model.tsct import TSCTLinear, factor_qat_project


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
    # The S3T rung is gone (2026-11-13) and says so, with the survivors.
    with pytest.raises(LoudError) as e:
        load_run("small", model_overrides={"factor_qat": "ternary_s3t"})
    assert "REMOVED from the quant ladder on 2026-11-13" in str(e.value)
    assert "Surviving factor_qat" in str(e.value)
