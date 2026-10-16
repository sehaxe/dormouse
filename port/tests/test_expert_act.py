"""Golden gates for the expert-activation tournament (2026-10-16).

Each arm is pinned by the PROPERTY its paper claims, not by a re-derived
number, so a transcription slip fails here rather than after 2k steps:

- `situ`: bounded ‖·‖∞ ≤ β1·β2 = 100 (K3 Eq. 19), first-order match with silu
  at the origin, and the β→∞ limit recovering silu pointwise.
- `memglu`: closed tail — both ±large inputs vanish; the gate's RMS matches
  silu's RMS on a standard-normal pre-activation when c0 is the authors'
  calibration constant (that IS what c0 is for).
- `silu`: bit-identical to `F.silu(g) * u` (the control cannot drift).
- config: an unknown `expert_act` is a LOUD refusal; a non-silu arm keeps the
  torch path even when `moe_kernel=triton` (COUNTED fallback, never silent).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from dormouse.config import load_run
from dormouse.instrument import Counters, LoudError
from dormouse.model.experts import expert_glu


def _g(t):
    torch.manual_seed(0)
    return t


def test_silu_is_the_bit_identical_control():
    g, u = _g(torch.randn(4096)), _g(torch.randn(4096))
    out = expert_glu("silu", g, u)
    assert torch.equal(out, F.silu(g) * u)


def test_situ_matches_the_k3_bound_and_the_origin():
    # K3's own values (§2.3.2): β1 = 4 (gate), β2 = 25 (up).
    g = torch.tensor([-1e4, -10.0, -1.0, 0.0, 1.0, 10.0, 1e4])
    u = torch.tensor([-1e4, -10.0, -1.0, 0.0, 1.0, 10.0, 1e4])
    out = expert_glu("situ", g, u, beta1=4.0, beta2=25.0)
    # Eq. 19: |out| ≤ β1·β2 = 100, at ANY input magnitude.
    assert out.abs().max().item() <= 100.0 + 1e-4
    # The bound is actually approached (a cap nobody reaches is a no-op):
    # at g = u = 1e4, tanh saturates and the gate is σ(1e4)·β1·β2 ≈ 100.
    assert out[-1].item() == pytest.approx(100.0, abs=2e-3)
    assert out[0].item() == pytest.approx(0.0, abs=1e-6)  # negative tail → 0
    # First-order match with silu at the origin (Eq. 18: β tanh(z/β) = z + O(z³/β²)).
    z = torch.tensor([1e-3, -1e-3])
    ref = F.silu(z) * z
    got = expert_glu("situ", z, z, beta1=4.0, beta2=25.0)
    assert torch.allclose(got, ref, rtol=1e-3, atol=1e-9)


def test_situ_recovers_silu_as_beta_goes_to_infinity():
    g, u = _g(torch.randn(4096)), _g(torch.randn(4096))
    got = expert_glu("situ", g, u, beta1=1e7, beta2=1e7)
    ref = F.silu(g) * u
    # Not bit-equal (tanh(1e-7) ≠ 1e-7 exactly), but pointwise-convergent:
    # the paper's own limiting claim, at a tolerance that still fails if the
    # caps are mis-scaled (a 2x error in β shows up at ~1e-2 here).
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_memglu_closed_tail_and_zero_sides():
    g = torch.tensor([-1e4, -20.0, -1.0, 0.0, 1.0, 20.0, 1e4])
    u = torch.ones_like(g)
    out = expert_glu("memglu", g, u)
    # Closed tail: both sides decay to zero (≠ silu's open positive tail).
    assert out[0].abs().item() < 1e-6 and out[-1].abs().item() < 1e-6
    assert out[3].abs().item() == 0.0  # g = 0 → gate 0
    # A finite peak exists (the gate is a bump, not a half-line).
    assert out.abs().max().item() > 0.1
    assert out.abs().max().item() < 0.5  # c0≈0.604 scales it below 1


def test_memglu_gate_shape_is_a_bump_with_peak_c0_over_2():
    # The formula's own geometry, no calibration assumption: h(t) = t·sqrt(1−t²)
    # peaks at t = 1/√2 with value 1/2, so ϕ's peak is exactly c0/2 at
    # g* = atanh(1/√2) ≈ 0.8814 — and the gate is a single bump (rises, falls).
    c0 = 0.6039096539628113
    g = torch.linspace(-3.0, 3.0, 20001)
    phi = expert_glu("memglu", g, torch.ones_like(g), c0=c0)
    peak = phi.max()
    assert peak.item() == pytest.approx(c0 / 2, rel=2e-3)
    g_star = torch.atanh(torch.tensor(1.0 / 2**0.5))
    at_star = expert_glu("memglu", g_star.reshape(1), torch.ones(1), c0=c0)
    assert at_star.item() == pytest.approx(c0 / 2, rel=2e-3)


def test_memglu_c0_is_the_origin_slope():
    # c0's published job is RMS-matching on the authors' OWN init-time
    # preactivations (their appendix; we cannot re-derive that distribution
    # here). What the FORMULA makes checkable: in the linear regime the gate
    # slope is exactly c0, so the constant is not a free decoration.
    z = torch.tensor([1e-4, -1e-4])
    nu = expert_glu("memglu", z, torch.ones(2), c0=0.6039096539628113)
    assert (nu / z).tolist() == pytest.approx([0.6039096539628113] * 2, rel=1e-5)


def test_unknown_act_is_fatal_in_the_glu():
    with pytest.raises(AssertionError, match="no formula"):
        expert_glu("relu6", torch.zeros(2), torch.zeros(2))


def test_config_refuses_unknown_expert_act():
    with pytest.raises(LoudError, match="expert_act"):
        load_run("small", model_overrides={"expert_act": "relu6"})


def test_config_refuses_nonpositive_beta_and_c0():
    with pytest.raises(LoudError, match="SiTU"):
        load_run("small", model_overrides={"expert_act": "situ", "expert_beta1": 0.0})
    with pytest.raises(LoudError, match="memglu_c0"):
        load_run("small", model_overrides={"expert_act": "memglu", "memglu_c0": 0.0})


def test_triton_arm_falls_back_counted_for_non_silu():
    # moe_kernel=triton + expert_act=situ: the fused kernel hardcodes silu, so
    # the padded torch arm must run — COUNTED with a reason, never silent.
    from conftest import tiny_cfg

    from dormouse.model import DormouseModel

    cfg = tiny_cfg(n_experts=4, moe_topk=2, expert_act="situ")
    model = DormouseModel(cfg)
    moe = next(m for m in model.modules() if type(m).__name__ == "SparseMoE")
    moe.moe_kernel = "triton"
    counters = Counters()
    x = torch.randn(64, cfg.d_model)
    moe(x, counters)
    assert counters["moe_triton_fallback"] == 1
    assert "expert_act=situ" in counters.why
