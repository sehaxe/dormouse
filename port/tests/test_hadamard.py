"""Gold tests for the BitNet v2 Hadamard rotation (`dormouse/model/hadamard.py`).

The reference is INDEPENDENT of the code under test in both directions:

* the butterfly path (`fwht`) is checked against the explicit matrix
  `H_p @ xᵀ / sqrt(p)` built by Kronecker products (`hadamard_matrix`), the same
  two-construction split `fwt.rs:75-99` uses so a broken butterfly cannot agree
  with a broken kernel;
* the quantization is checked against the *format's* definition, not against the
  implementation: the level count `2^(bits-1)-1` (7 / 127,
  `crates/dormouse-core/src/act_quant.rs:54-59`) and the scale rule that maps a
  token max ONTO that max — the defect that left the fp4 grid with 3 of its 8
  magnitudes reachable (AGENTS §3.7 act-quant). The pure-spike round trip below
  fails a factor-of-7 (or factor-of-127) scale error, which is exactly that bug.

MEASURED ON THIS FIXTURE (torch 2.13 CPU, seeds as written):

| fixture | number |
|---|---|
| `fwht(fwht(x))` vs `x`, fp32, `[4, p]`, p = 32…4096 | max abs error 5.07e-7 (assert < 1e-5) |
| per-row L2 norm, one `fwht`, fp32 | max error 7.6e-6 (assert < 1e-4) |
| steep probe `randn [1,64]`, `x[0,3]=1000` | max abs 1000.0 → 126.6, kurtosis 62.0 → 1.0 |
| shear probe `randn [1,256]`, `x[0,3]=1000` | max abs 1000.0 → 65.3, kurtosis 253.9 → 1.0 |
| `v2_quantize` int4 on `randn [4,64]`: rms err / rms(x) | 0.107 |
| `v2_quantize` int8 on the same | 0.0054 |
| spike `[64, 0…]`, p=64, through `v2_quantize(_, 4 or 8)` | round-trips to < 1e-4 |
| bf16 `fwht(fwht(x))` vs `x`, p = 32…4096 | max rel error 8.9e-3 ≈ 2.3 bf16 eps |
"""

from __future__ import annotations

import math

import pytest
import torch

from dormouse.instrument import LoudError
from dormouse.model import hadamard


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    """A seeded torch.randn (the suite has no global seed fixture)."""
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


# --------------------------------------------------------------------------
# (a) the butterfly is the normalized, orthogonal transform
# --------------------------------------------------------------------------


def test_double_application_is_identity_and_l2_norm_is_preserved() -> None:
    """`fwht(fwht(x)) == x` and one `fwht` preserves every row's L2 norm."""
    worst = 0.0
    worst_norm = 0.0
    for p in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        x = _randn(4, p)
        worst = max(worst, float((hadamard.fwht(hadamard.fwht(x)) - x).abs().max()))
        n0 = x.norm(dim=-1)
        n1 = hadamard.fwht(x).norm(dim=-1)
        worst_norm = max(worst_norm, float((n1 - n0).abs().max()))
    # measured 5.07e-7 / 7.6e-6 for p <= 4096 (fp32); the bounds are headroom.
    assert worst < 1e-5, f"double FWHT is not the identity: max |err| {worst:.3e}"
    assert worst_norm < 1e-4, f"FWHT is not orthogonal: max row-norm err {worst_norm:.3e}"


def test_butterfly_matches_explicit_hadamard_matrix() -> None:
    """`fwht(x) == (H_p @ xᵀ)ᵀ / sqrt(p)` on two random shapes (the reference)."""
    for shape in ((4, 8), (3, 16)):
        x = _randn(*shape, seed=1)
        p = shape[-1]
        want = (hadamard.hadamard_matrix(p) @ x.T).T / math.sqrt(p)
        got = hadamard.fwht(x)
        assert torch.allclose(got, want, atol=1e-5), (
            f"{shape}: butterfly != the H_p@xᵀ/sqrt(p) reference (maxdiff "
            f"{(got - want).abs().max():.3e})"
        )


def test_hadamard_matrix_is_symmetric_and_squares_to_p_i() -> None:
    """The ±1 matrix itself: symmetric, `H_pᵀ H_p = p·I` (so `H/sqrt(p)` is orthogonal)."""
    for p in (1, 2, 4, 8, 16):
        h = hadamard.hadamard_matrix(p)
        assert h.shape == (p, p)
        assert set(h.flatten().tolist()) <= {1.0, -1.0}, "H is ±1"
        assert torch.equal(h, h.T), "H must be symmetric (why one transform is its own inverse)"
        assert torch.equal(h @ h, p * torch.eye(p, dtype=h.dtype)), "HᵀH = p·I"


def test_hadamard_matrix_builds_by_kronecker() -> None:
    """`H_{2k} = kron(H_2, H_{2k-1})` — the recursion, spelled out."""
    h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    for p in (1, 2, 4, 8, 16, 32):
        h = hadamard.hadamard_matrix(p)
        want = torch.ones((1, 1))
        while want.shape[0] < p:
            want = torch.kron(h2, want)
        assert torch.equal(h, want), f"p={p}: not the kron recursion"


# --------------------------------------------------------------------------
# (b) the outlier-suppression mechanism (what the paper buys)
# --------------------------------------------------------------------------


def test_rotation_suppresses_the_steep_outlier() -> None:
    """One 1000x spike: strictly smaller max|x| AND smaller kurtosis after `fwht`."""
    x = _randn(1, 64, seed=0)
    x[0, 3] = 1000.0
    m0, k0 = hadamard.outlier_metric(x)
    m1, k1 = hadamard.outlier_metric(hadamard.fwht(x))
    assert m1 < m0 and k1 < k0, f"no suppression: max {m0}->{m1}, kurt {k0}->{k1}"
    # measured 1000.0 / 62.0 -> 126.6 / 1.0 at p=64 (a 7.9x max cut, kurtosis to ~gaussian)
    assert m0 == pytest.approx(1000.0)
    assert m1 == pytest.approx(126.6, rel=0.01)
    assert k0 == pytest.approx(62.0, rel=0.02)
    assert k1 == pytest.approx(1.0, abs=0.02)
    assert m1 < m0 / 4.0, "the spike must actually be spread, not nudged"

    # p=256 spreads it further: ~1000/sqrt(p) = 62.5, measured 65.3
    y = _randn(1, 256, seed=0)
    y[0, 3] = 1000.0
    b0, j0 = hadamard.outlier_metric(y)
    b1, j1 = hadamard.outlier_metric(hadamard.fwht(y))
    assert b1 < b0 and j1 < j0
    assert b0 == pytest.approx(1000.0) and b1 == pytest.approx(65.3, rel=0.02)
    assert j0 == pytest.approx(253.9, rel=0.02) and j1 == pytest.approx(1.0, abs=0.02)


def test_rotation_is_exactly_invertible_so_the_spike_returns() -> None:
    """Orthogonality: the spike comes back at its own position at its own height."""
    x = _randn(1, 256, seed=0)
    x[0, 3] = 1000.0
    back = hadamard.fwht(hadamard.fwht(x))
    assert int(back.argmax()) == 3, "the inverse rotation moved the spike"
    assert float(back[0, 3]) == pytest.approx(1000.0, rel=1e-5)
    assert float((back - x).abs().max()) < 1e-4


# --------------------------------------------------------------------------
# (c) v2_quantize: the level convention, the round trips, the STE
# --------------------------------------------------------------------------


def test_level_convention_is_the_a4_8_symmetric_levels() -> None:
    """int4 = 7 levels, int8 = 127 (`act_quant.rs:54-59`), not 16/256 and not 1."""
    assert hadamard._levels(4) == 7
    assert hadamard._levels(8) == 127
    assert hadamard._levels(2) == 1 and hadamard._levels(16) == 32767


def test_scale_maps_the_token_max_onto_the_format_max() -> None:
    """A pure spike at p=64: `fwht` gives exactly sqrt(p)=8 per slot, scale 8/7 or 8/127.

    With the scale normalized to 1 instead (the act-quant bug), every `q` would be
    one level and the spike would come back at `p·x/sqrt(p)`-worth of nonsense.
    """
    x = torch.zeros(1, 64)
    x[0, 0] = 64.0
    rot = hadamard.fwht(x)
    assert float(rot.abs().max()) == pytest.approx(8.0, rel=1e-6), "a spike smears to A/sqrt(p)"
    for bits in (4, 8):
        out = hadamard.v2_quantize(x, bits)
        assert float((out - x).abs().max()) < 1e-4, (
            f"bits={bits}: the spike did not round-trip ({out[0, :4].tolist()}) - the "
            "scale is not reaching the format max (act_quant.rs:54-59)"
        )
        assert float(out[0, 1:].abs().max()) < 1e-4, "the zero tail must stay zero"


def test_high_bits_is_the_identity_bit_exact() -> None:
    """`bits >= 16` returns `x` unchanged (`activations.rs:31-33`) — the same object."""
    x = _randn(3, 32, seed=6)
    for bits in (16, 32):
        out = hadamard.v2_quantize(x, bits)
        assert out is x and torch.equal(out, x), f"bits={bits} must be the identity"
    bf = _randn(2, 16, seed=7).to(torch.bfloat16)
    assert hadamard.v2_quantize(bf, 16) is bf


def test_v2_quantize_rms_error_bounds() -> None:
    """int4 rms rel 0.107, int8 0.0054 (measured [4,64], seed 0)."""
    x = _randn(4, 64, seed=0)
    rms = {}
    for bits in (4, 8):
        out = hadamard.v2_quantize(x, bits)
        assert out.dtype == x.dtype and out.shape == x.shape
        rms[bits] = float((out - x).pow(2).mean().sqrt() / x.pow(2).mean().sqrt())
    assert rms[4] == pytest.approx(0.107, abs=0.02), f"int4 rms rel {rms[4]:.4f}"
    assert rms[8] == pytest.approx(0.0054, abs=0.002), f"int8 rms rel {rms[8]:.4f}"
    assert rms[8] < rms[4], "int8 must be the tighter format"


def test_v2_quantize_keeps_the_spike_at_its_position() -> None:
    """The lossy part is only the quant: the spike stays at index 10, within half a level."""
    x = _randn(1, 128, seed=2)
    x[0, 10] = 1000.0
    for bits, half_level in ((4, 1.0 / 7.0), (8, 1.0 / 127.0)):
        out = hadamard.v2_quantize(x, bits)
        assert int(out.argmax()) == int(x.argmax()) == 10, f"bits={bits}: spike moved"
        rel = abs(float(out[0, 10]) - 1000.0) / 1000.0
        assert rel < half_level, f"bits={bits}: spike rel err {rel:.5f} > {half_level:.5f}"
    # measured: int4 2.6 %, int8 0.037 %


def test_v2_quantize_ste_gradient_is_identity() -> None:
    """`v2_quantize(x, b).sum().backward()` ⇒ `x.grad == 1` through the rotation and quant."""
    for bits in (4, 8):
        x = _randn(4, 64, seed=3).requires_grad_(True)
        out = hadamard.v2_quantize(x, bits)
        assert out.requires_grad and out.shape == x.shape
        out.sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x)), f"bits={bits}: STE grad != identity"


def test_v2_quantize_is_deterministic_bit_for_bit() -> None:
    x = _randn(4, 256, seed=9)
    assert torch.equal(hadamard.fwht(x), hadamard.fwht(x))
    assert torch.equal(hadamard.v2_quantize(x, 4), hadamard.v2_quantize(x, 4))
    assert torch.equal(hadamard.v2_quantize(x, 4), hadamard.v2_quantize(x.clone(), 4))


# --------------------------------------------------------------------------
# (d) bf16 policy: dtype-preserving, orthogonal to the measured bound
# --------------------------------------------------------------------------


def test_bf16_is_dtype_preserving_and_orthogonal() -> None:
    """bf16 in → bf16 out, round trip ≤ 8.9e-3 rel ≈ 2.3 bf16 eps (the stated bound)."""
    worst = 0.0
    for p in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        x = _randn(4, p, seed=4).to(torch.bfloat16)
        y = hadamard.fwht(x)
        assert y.dtype == torch.bfloat16, "fwht must not up-cast bf16"
        back = hadamard.fwht(y)
        assert back.dtype == torch.bfloat16
        xf, bf = x.float(), back.float()
        worst = max(worst, float((bf - xf).abs().max()) / float(xf.abs().max()))
        n0, n1 = xf.norm(dim=-1), y.float().norm(dim=-1)
        assert float(((n1 - n0).abs() / n0).max()) < 5e-3, f"p={p}: bf16 norm not preserved"
    assert worst < 2e-2, f"bf16 round trip {worst:.3e} exceeds the documented 8.9e-3 bound"


def test_v2_quantize_preserves_the_activation_dtype() -> None:
    """Under --bf16 the activations stay bf16; only the quant math is fp32."""
    x = _randn(2, 32, seed=8).to(torch.bfloat16)
    for bits in (4, 8):
        out = hadamard.v2_quantize(x, bits)
        assert out.dtype == torch.bfloat16 and out.shape == x.shape
        assert torch.isfinite(out.float()).all()


# --------------------------------------------------------------------------
# (e) LOUD
# --------------------------------------------------------------------------


def test_loud_on_non_power_of_two_last_dim() -> None:
    for bad in (3, 7, 12, 100):
        with pytest.raises(LoudError, match=r"2\^k"):
            hadamard.fwht(torch.randn(2, bad))
    with pytest.raises(LoudError, match=r"2\^k"):
        hadamard.v2_quantize(torch.randn(2, 3), 4)
    with pytest.raises(LoudError, match="power-of-two"):
        hadamard.hadamard_matrix(12)
    with pytest.raises(LoudError, match="last dim"):
        hadamard.fwht(torch.randn(2, 0))
    with pytest.raises(LoudError, match="0-dim"):
        hadamard.fwht(torch.tensor(1.0))
    with pytest.raises(LoudError, match="MiB"):
        hadamard.hadamard_matrix(8192)  # the test-reference cap


def test_loud_on_non_finite_input() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        x = torch.ones(2, 8)
        x[0, 3] = bad
        with pytest.raises(LoudError, match="non-finite"):
            hadamard.fwht(x)
        with pytest.raises(LoudError, match="non-finite"):
            hadamard.v2_quantize(x, 4)


def test_loud_on_non_float_and_bad_bits() -> None:
    with pytest.raises(LoudError, match="float math"):
        hadamard.fwht(torch.ones(2, 8, dtype=torch.int32))
    with pytest.raises(LoudError, match="bits >= 2"):
        hadamard.v2_quantize(torch.randn(2, 8), 1)
