r"""Gold tests for the BitNet ternary family (`dormouse/model/ternary.py`).

The reference is the burn transcription (the formulas in the module docstring,
each with its `file:line`), and the assertions are written to fail if a formula
drifts: the codebook is checked as a SET, the STE is checked on the BACKWARD
(no forward oracle can see it), the dead zone is pinned where `ternarize` and
`sign_binary` must disagree, and the two Monte-Carlo properties of S3T are
checked against their analytic expectations rather than against a re-run.

MEASURED, `torch.manual_seed(0)`, fp32, CPU (the numbers the bounds below are
set from; they are measurements of THIS code, dated 2026-10-16):

| what | measurement |
|---|---|
| `ternarize(randn(64,64))` non-zero magnitude | one value, α = 0.7872172 |
| ... zeros | 1279 of 4096, the RoundClip dead zone |
| 2nd pass `ternarize(t)`: sign/zero mask | identical; `max = α·f`, f = 0.687744 |
| `activation_quant_8bit(randn(4,64,128))` `\|err\|`/absmax | 0.003937 = half a code step |
| ... rms err / rms x, and max rel err at `\|x\| ≥ amax/3` | 0.00649 and 0.0118 |
| S3T `E[out]` vs `sign·min(\|w\|, scale)`, 4000 seeded draws | max err 0.0210, tol 0.05 |

The one place the task's own test list is NOT transcribed verbatim is the
"idempotence" bullet — `ternarize(ste_ternary(w)) == ste_ternary(w)` is FALSE
for any quantizer whose scale is read from its input (the second pass sees
`α' = f·α`), and the test named for it asserts the true statement instead. See
its docstring; that deviation is the point of the test, not a gap in it.
"""

from __future__ import annotations

import math

import pytest
import torch

from dormouse.instrument import LoudError
from dormouse.model import ternary

# --------------------------------------------------------------------------
# (a) ternarize — the b1.58 RoundClip codebook
# --------------------------------------------------------------------------


def test_hand_computed_absmean_on_a_256x64_tensor() -> None:
    """`[256, 64]`, magnitudes {1.0, 0.25} half and half ⇒ α = 0.625 EXACTLY.

    Every value is dyadic and the counts are 8192/8192, so
    `α = (8192·1.0 + 8192·0.25)/16384 = 0.625` with no fp32 rounding to hide
    behind: the levels, the zero, and the clamp are all pinned literally.
    0.25/0.625 = 0.4 → rounds to 0; 1.0/0.625 = 1.6 → rounds to 2 → clamped to 1.
    """
    pat = torch.tensor([-1.0, -0.25, 0.25, 1.0]).repeat(256 * 16)
    w = pat.reshape(256, 64)
    alpha = w.abs().mean()
    assert float(alpha) == 0.625, "hand arithmetic: 8192·1.0 + 8192·0.25 over 16384"
    t = ternary.ternarize(w)
    assert t.shape == (256, 64) and t.dtype == w.dtype
    levels = set(t.unique().tolist())
    assert levels == {-0.625, 0.0, 0.625}, f"codebook is {{−α, 0, +α}}, got {levels}"
    assert torch.equal(t[w.abs() == 0.25], torch.zeros(8192)), "0.4 rounds to 0"
    assert torch.equal(t[w == 1.0], torch.full((4096,), 0.625))
    # PER-TENSOR, not per-row: every row's own absmean is also 0.625 (the
    # pattern is periodic), so pin the axis with a tensor whose rows differ.
    rows = torch.stack([torch.full((8,), 1.0), torch.full((8,), 0.25)])
    tr = ternary.ternarize(rows)
    assert torch.equal(tr[1], torch.full((8,), 0.0)), "α is ONE mean over the tensor"
    assert torch.equal(tr[0], torch.full((8,), 0.625))


def test_ternarize_levels_on_randn_and_a_measured_alpha() -> None:
    """Three levels, bounded by the absmean of the INPUT, zeros present."""
    torch.manual_seed(0)
    w = torch.randn(64, 64)
    alpha = float(w.abs().mean())
    t = ternary.ternarize(w)
    mags = set(t[t != 0].abs().tolist())
    assert len(mags) == 1, f"one non-zero magnitude, got {mags}"
    assert mags.pop() == pytest.approx(alpha, rel=1e-6)
    assert int((t == 0).sum()) > 0, "|w| < α/2 must land on 0 (1279 of 4096 measured)"
    assert bool((t.abs() <= alpha + 1e-6).all()), "no value exceeds α"


# --------------------------------------------------------------------------
# (b) ternarize vs sign_binary — the pinned difference
# --------------------------------------------------------------------------


def test_ternarize_and_sign_binary_disagree_at_point_three_alpha() -> None:
    """`w = [1, 1, 1, 0.3·α]` ⇒ α = 0.8108108; ternarize gives 0, sign_binary +α.

    0.3 rounds to 0 (RoundClip's dead zone) while `sign_binary` has no zero
    level at all — the one line that says why they are not the same function.
    """
    w = torch.tensor([1.0, 1.0, 1.0, 0.9 / 3.7])  # (3 + x)/4 = x/0.3 ⇒ x = 0.9/3.7
    alpha = float(w.abs().mean())
    assert alpha == pytest.approx(0.8108108043670654, rel=1e-9)
    assert w[3] == pytest.approx(0.3 * alpha)
    t = ternary.ternarize(w)
    b = ternary.sign_binary(w)
    assert t[3] == 0.0, "0.3·α is inside ternarize's dead zone"
    assert b[3] == pytest.approx(alpha, rel=1e-6), "sign_binary has no zero"
    assert torch.equal(t[:3], torch.full((3,), alpha))
    assert torch.equal(b, torch.full((4,), alpha))
    assert not torch.equal(t, b), "the two quantizers must not alias"
    assert int((b == 0).sum()) == 0
    # w == 0 keeps 0 in both (torch's sign(0) = 0), so a zero master stays zero.
    assert ternary.sign_binary(torch.zeros(4)).tolist() == [0.0] * 4


# --------------------------------------------------------------------------
# (c) STE — checked on the BACKWARD only
# --------------------------------------------------------------------------


def test_ste_gradient_is_exactly_identity() -> None:
    """`ste_ternary` and the annealed form: d/dw of a ones-loss is 1 everywhere.

    Forward values cannot see this (the forward IS the ternary projection), so
    the assertion is on `.grad` — every path from `w` to the output runs through
    the detached residual, which is what makes the master trainable.
    """
    w = torch.randn(8, 64, requires_grad=True)
    ternary.ste_ternary(w).sum().backward()
    assert torch.equal(w.grad, torch.ones_like(w)), "STE is the identity backward"

    w2 = torch.randn(8, 64, requires_grad=True)
    ternary.ste_ternary_annealed(w2, 0.5).sum().backward()
    assert torch.equal(w2.grad, torch.ones_like(w2)), "annealing must not scale the grad"


def test_alpha_zero_is_bit_identical_and_alpha_one_is_pure_ternary() -> None:
    """The two endpoints of the anneal: 0 = the master, 1 = the pure projection."""
    torch.manual_seed(1)
    w = torch.randn(64, 64)
    assert torch.equal(ternary.ste_ternary_annealed(w, 0.0), w), "alpha=0 is a no-op"
    one = ternary.ste_ternary_annealed(w, 1.0)
    assert torch.equal(one, ternary.ternarize(w))
    alpha = float(w.abs().mean())
    assert bool(((one.abs() - alpha).abs() < 1e-5).logical_or(one == 0).all())
    assert int((one == 0).sum()) > 0, "randn must produce zeros"
    with pytest.raises(AssertionError, match="refused, not clamped"):
        ternary.ste_ternary_annealed(w, 1.5)


def test_ternarize_second_pass_is_a_scale_change_not_a_code_change() -> None:
    """`ternarize(ste_ternary(w)) == ste_ternary(w)` — the literal claim is FALSE.

    The brief asked for that equality. It cannot hold for any quantizer that
    reads its scale from its input: the second pass computes
    `α' = mean|t| = f·α` (f = the non-zero fraction of the first pass), so the
    codebook shrinks by f while the CODE — signs and the zero mask — is an
    exact fixed point. What is asserted here is the true pair of statements,
    and the false one is pinned as false, so a later "fix" that made the
    equality hold would have to break the codebook bound to do it.
    """
    torch.manual_seed(0)
    w = torch.randn(64, 64)
    t = ternary.ste_ternary(w)
    t2 = ternary.ternarize(t)
    assert torch.equal(t2.sign(), t.sign()), "signs are a fixed point"
    assert torch.equal(t2 == 0, t == 0), "the zero mask is a fixed point"
    f = float((t != 0).float().mean())
    alpha = float(w.abs().mean())
    assert float(t2.abs().max()) == pytest.approx(alpha * f, rel=1e-6)
    assert not torch.equal(t2, t), "the literal equality the brief asked for is false"
    # ... and it DOES hold on the projection's own zero-free image (f = 1):
    zf = torch.where(torch.rand(8, 64) > 0.5, 1.0, -1.0)
    assert torch.equal(ternary.ternarize(ternary.ternarize(zf)), zf)


# --------------------------------------------------------------------------
# (d) the projection is stable where stability is the requirement
# --------------------------------------------------------------------------


def test_projection_of_a_projected_tensor_keeps_its_code() -> None:
    """Applying the STE twice changes the scale, never the ternary CODE.

    This is the property the training path needs (`ternarize(ste_ternary(w))`
    has the same sign pattern as `ste_ternary(w)`), on three shapes.
    """
    torch.manual_seed(3)
    for shape in ((256, 64), (4, 16, 32), (4096,)):
        w = torch.randn(*shape)
        t = ternary.ternarize(w)
        t2 = ternary.ternarize(t)
        assert torch.equal(t2.sign(), t.sign())
        assert torch.equal(t2 == 0, t == 0)


# --------------------------------------------------------------------------
# (e) S3T
# --------------------------------------------------------------------------


def test_s3t_is_deterministic_at_and_above_the_scale() -> None:
    """`|w| ≥ scale ⇒ out = sign(w)·scale` — analytic, not statistical."""
    torch.manual_seed(2)
    w = torch.randn(4096)
    scale = float(w.abs().mean())
    g = torch.Generator().manual_seed(5)
    out = ternary.stochastic_ternary(w, generator=g)
    above = w.abs() >= scale
    assert int(above.sum()) > 0, "a gaussian has elements above its absmean"
    assert torch.equal(out[above], w[above].sign() * scale)
    assert set(out.unique().tolist()) <= {-scale, 0.0, scale}


def test_s3t_monte_carlo_matches_its_analytic_expectation() -> None:
    """`E[out] = sign(w)·min(|w|, scale)` over 4000 seeded draws (max err 0.021).

    Pointwise unbiasedness (`E[out] = w`) needs `|w| ≤ scale`; on elements above
    it the expectation is the clamp, which is the second assertion.
    """
    torch.manual_seed(4)
    w = torch.randn(4096)
    scale = float(w.abs().mean())
    g = torch.Generator().manual_seed(11)
    n = 4000
    acc = torch.zeros_like(w)
    for _ in range(n):
        acc += ternary.stochastic_ternary(w, generator=g)
    mean = acc / n
    want = w.sign() * torch.minimum(w.abs(), torch.tensor(scale))
    assert float((mean - want).abs().max()) < 0.05, "analytic expectation, 0.0210 measured"
    inside = w.abs() <= scale
    assert int(inside.sum()) > 1000
    assert float((mean[inside] - w[inside]).abs().max()) < 0.05, "unbiased where |w| ≤ scale"


def test_s3t_same_seed_is_bit_identical() -> None:
    """A seeded `torch.Generator` is the whole reproducibility contract (ADR-0021)."""
    torch.manual_seed(6)
    w = torch.randn(1024)
    a = ternary.stochastic_ternary(w, generator=torch.Generator().manual_seed(7))
    b = ternary.stochastic_ternary(w, generator=torch.Generator().manual_seed(7))
    c = ternary.stochastic_ternary(w, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b), "same seed, same bits"
    assert not torch.equal(a, c), "a different seed is a different draw"
    assert torch.equal(ternary.stochastic_ternary(torch.zeros(64)), torch.zeros(64)), (
        "an all-zero tensor quantizes to zeros, not to NaN"
    )


# --------------------------------------------------------------------------
# (f) activation_quant_8bit
# --------------------------------------------------------------------------


def test_activation_quant_8bit_roundtrip_bound() -> None:
    """Half a code step is the structural bound: 1/(2·127) = 0.003937 of absmax.

    MEASURED `torch.manual_seed(0)`, `randn(4, 64, 128)`: max `|err|/row_absmax`
    = 0.003937 (the bound), rms err / rms x = 0.00649, max relative error over
    `|x| ≥ amax/3` = 0.0118. Over ALL elements the max relative error is 1.0 —
    an element far below its token's absmax rounds to 0, which is the format
    working, not a defect.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 64, 128)
    q = ternary.activation_quant_8bit(x)
    assert q.shape == x.shape and q.dtype == x.dtype
    amax = x.abs().amax(dim=-1, keepdim=True)
    err = (q - x).abs()
    assert bool((err <= amax * (1.0 / (2.0 * 127.0)) + 1e-7).all()), "half a code step"
    assert float((err / amax).max()) == pytest.approx(0.003937, abs=1e-6)
    assert float(err.pow(2).mean().sqrt() / x.pow(2).mean().sqrt()) < 0.01
    sig = x.abs() >= amax / 3.0
    assert float((err[sig] / x[sig].abs()).max()) == pytest.approx(0.011757, rel=1e-3)
    # the scale maps onto ±127, so no dequantized value exceeds the row absmax
    assert bool((q.abs() <= amax + 1e-6).all())
    # per-TOKEN: the last dim is scaled on its own, not the whole tensor
    rows = torch.cat([torch.full((1, 8), 1.0), torch.full((1, 8), 1e-3)], dim=0)
    qr = ternary.activation_quant_8bit(rows)
    assert float(qr[1].abs().max()) == pytest.approx(1e-3, rel=1e-2), "weak token survives"
    # what per-token buys, in one line: under ONE global absmax (1.0) the weak
    # token's code is round(1e-3/1.0·127) = 0, i.e. the token would be erased.
    assert round(1e-3 / 1.0 * 127.0) == 0


def test_activation_quant_8bit_ste_is_identity_backward() -> None:
    """burn's residual trick (`activations.rs:22-23`): forward quantized, grad 1."""
    x = torch.randn(2, 8, 16, requires_grad=True)
    ternary.activation_quant_8bit(x).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


# --------------------------------------------------------------------------
# (g) LOUD, dtype, and bundle info
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        ternary.ternarize,
        ternary.sign_binary,
        ternary.stochastic_ternary,
        ternary.activation_quant_8bit,
    ],
)
def test_non_finite_input_refuses_loudly(fn) -> None:
    """A NaN would poison the scale silently: cause AND escape in the message."""
    for bad in (
        torch.tensor([1.0, float("nan")]),
        torch.tensor([1.0, float("inf"), -float("inf")]),
    ):
        with pytest.raises(LoudError) as e:
            fn(bad)
        msg = str(e.value)
        assert "non-finite" in msg and "Escape" in msg, msg
    with pytest.raises(AssertionError, match="float tensors"):
        fn(torch.arange(4))


def test_bf16_input_dtype_is_preserved() -> None:
    """fp32 masters, bf16 boundary: every entry point returns its input's dtype.

    The bf16 levels are `{−bf16(α), 0, +bf16(α)}` — α itself is computed in
    fp32 from the upcast values, so the pinned check is against α in bf16.
    """
    torch.manual_seed(9)
    w = torch.randn(64, 64).to(torch.bfloat16)
    alpha = float(w.float().abs().mean())
    t = ternary.ternarize(w)
    assert t.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
    assert ternary.sign_binary(w).dtype == torch.bfloat16
    assert ternary.ste_ternary(w).dtype == torch.bfloat16
    assert ternary.stochastic_ternary(w, torch.Generator().manual_seed(1)).dtype == (torch.bfloat16)
    assert ternary.activation_quant_8bit(w).dtype == torch.bfloat16
    mags = t[t != 0].abs().unique()
    assert mags.numel() == 1
    assert float(mags[0]) == pytest.approx(alpha, rel=1e-2), "bf16 has 8 mantissa bits"


def test_packed_bits_per_weight_and_the_b1_bank_arithmetic() -> None:
    """log2(3) ternary / 1.0 binary, and the byte math the module doc carries.

    The bank is b1's routed-expert shape `[1024, 64, 5461]` (the same 357 892 096
    elements `tests/test_mxfp.py` prices MXFP4 against), so the two byte budgets
    are directly comparable: 2-bit ternary + per-row scales = 7.98x under bf16
    and 15.95x under fp32; base-3 packing (5 weights/byte, the log2(3) number)
    reaches 9.96x.
    """
    assert ternary.packed_bits_per_weight() == pytest.approx(math.log2(3))
    assert ternary.packed_bits_per_weight() == pytest.approx(1.584962500721156)
    assert ternary.packed_bits_per_weight(2) == 1.0
    assert 1.0 < ternary.packed_bits_per_weight() < 2.0
    with pytest.raises(AssertionError, match=">= 2 levels"):
        ternary.packed_bits_per_weight(1)

    bank = 1024 * 64 * 5461
    assert bank == 357_892_096
    bf16_bytes, fp32_bytes = 2 * bank, 4 * bank
    assert bf16_bytes == 715_784_192 and fp32_bytes == 1_431_568_384
    two_bit = bank // 4
    assert two_bit == 89_473_024
    scales = 1024 * 64 * 4  # one fp32 absmean per row (weight_quant_2bit's axis)
    assert scales == 262_144 and scales / (two_bit + scales) < 0.003
    assert two_bit + scales == 89_735_168
    assert bf16_bytes / (two_bit + scales) == pytest.approx(7.9765, abs=1e-3)
    assert fp32_bytes / (two_bit + scales) == pytest.approx(15.9530, abs=1e-3)
    base3 = -(-bank // 5)  # 3^5 = 243 <= 256: five ternary weights per byte
    assert base3 == 71_578_420
    assert bf16_bytes / (base3 + scales) == pytest.approx(9.9635, abs=1e-3)
