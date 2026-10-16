r"""Gold tests for the OCP MX codec (`dormouse/model/mxfp.py`).

The reference here is a SLOW scalar loop, deliberately independent of the
vectorized path: `_ref_block_mxfp4` walks one 32-element block in python and
packs nibbles by hand, so "3-level quantizer wearing an fp4 label" (AGENTS §3.7
act-quant: 0.75 is not an e2m1 magnitude, and scaling a block's max onto 1 made
five of the eight magnitudes dead code) cannot pass. `test_e2m1_grid_...` and
the reachability asserts make that a literal check. The e4m3 side is
cross-checked twice: the module uses torch's `float8_e4m3fn`, and this file's
own 256-code table + scalar RNE reference are verified against it.

MEASURED ERROR BOUNDS (test_error_bounds, `torch.manual_seed(0)`,
`torch.randn(4096, 32)` fp32, per-element relative error vs the fp32 master;
"significant" = `|x| >= block_amax/6`, i.e. inside the range the block scale
was picked for — 69.9% of a gaussian's elements):

| metric                              | MXFP4 (e2m1) | MXFP8 (e4m3) |
|-------------------------------------|--------------|--------------|
| rms error / rms(x)                  | 0.1155       | 0.02653      |
| mean \|err\|/\|x\|, all elements     | 0.2355       | 0.02251      |
| max \|err\|/\|x\|, all elements      | 1.0          | 0.2285       |
| mean \|err\|/\|x\|, significant      | 0.1011       | 0.02247      |
| max \|err\|/\|x\|, significant       | 0.3333       | 0.05882      |
| max \|err\|/block_amax               | 0.1999       | 0.05881      |
| mean \|err\|/block_amax              | 0.0390       | 0.00783      |

The task brief guessed "~5-10% max rel" for MXFP4; the measurement says the max
is **33.3% for significant elements and exactly 1.0 overall**, and both are
structural, not a bug: e2m1's coarsest tie is the 0.5↔1.0 midpoint (0.75, a
25/75 split ⇒ 1/3 relative), and an element orders of magnitude below its block
amax rounds to zero (100%). MXFP8's bound IS the classical one: its significant
max (5.88%) sits under `2**-4 = 6.25%`, the RNE relative bound for a 4-bit
significand. The assertions below are the principled bound where one exists
(e4m3 ≤ 2⁻⁴; e2m1 ≤ 1.0) and the measured value plus headroom otherwise.
"""

from __future__ import annotations

import math

import pytest
import torch

from dormouse.instrument import LoudError
from dormouse.model import mxfp
from dormouse.model.tsct import _NVFP4_MAGNITUDES

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e4m3_value(code: int) -> float:
    """Code → value, from §5.3.1 Table 2 (bias 7, min subnorm 2⁻⁹, max 448)."""
    sign = -1.0 if code & 0x80 else 1.0
    exp, mant = (code >> 3) & 0xF, code & 0x7
    if exp == 0xF and mant == 0x7:
        return float("nan")  # the one NaN encoding
    if exp == 0:
        return sign * mant * 2.0**-9
    return sign * (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)


E4M3_REFERENCE = {code: _e4m3_value(code) for code in range(256)}
#: Positive, non-NaN codes in ascending value order (0x00 … 0x7E).
_E4M3_POSITIVE = [(E4M3_REFERENCE[c], c) for c in range(0x7F)]


def _scale_exp(ratio: float) -> int:
    """Scalar form of the codec's scale rule: the smallest `e` with `2^e >= ratio`."""
    e = 0
    while 2.0**e < ratio:
        e += 1
    while 2.0 ** (e - 1) >= ratio:
        e -= 1
    return e


def _ref_scale_code(amax: float, elem_max: float) -> int:
    if amax == 0:
        return 0  # the codec's declared zero (§5.4.1 Table 7 has no zero encoding)
    return _scale_exp(amax / elem_max) + 127


def _ref_nearest_mag(q: float) -> int:
    """Nearest e2m1 magnitude index, ties to the even mantissa (RNE)."""
    lo = max(i for i, m in enumerate(E2M1) if m <= q)
    if lo == len(E2M1) - 1:
        return lo
    mid = (E2M1[lo] + E2M1[lo + 1]) / 2
    if q > mid:
        return lo + 1
    if q < mid:
        return lo
    return lo if lo % 2 == 0 else lo + 1


def _ref_e4m3_code(q: float) -> int:
    """Nearest e4m3 code, ties to the even code — a scalar independent path."""
    best, best_d = 0, math.inf
    for value, code in _E4M3_POSITIVE:
        d = abs(value - q)
        if d < best_d or (d == best_d and code % 2 == 0):
            best, best_d = code, d
    return best


def _ref_block_mxfp4(block: list[float]) -> tuple[list[int], int]:
    """One 32-block → (16 packed bytes, 1 scale byte), byte by byte."""
    amax = max(abs(v) for v in block)
    code = _ref_scale_code(amax, 6.0)
    scale = 2.0 ** (code - 127) if code else 0.0
    codes = []
    for v in block:
        q = 0.0 if scale == 0 else v / scale
        mag = _ref_nearest_mag(abs(q))
        codes.append(mag + 8 if (q < 0 and mag > 0) else mag)
    packed = [codes[2 * i] | (codes[2 * i + 1] << 4) for i in range(len(codes) // 2)]
    return packed, code


def _ref_block_values(block: list[float]) -> list[float]:
    """The values `_ref_block_mxfp4`'s bytes decode to."""
    packed, code = _ref_block_mxfp4(block)
    scale = 2.0 ** (code - 127) if code else 0.0
    out = []
    for byte in packed:
        for shift in (0, 4):
            c = (byte >> shift) & 0xF
            out.append(E2M1[c & 7] * (-1.0 if c >= 8 else 1.0) * scale)
    return out


# --------------------------------------------------------------------------
# (a) round trip against the slow scalar reference
# --------------------------------------------------------------------------


def test_mxfp4_hand_block_matches_scalar_reference_bit_for_bit() -> None:
    """Every one of the 8 e2m1 magnitudes is reached, and the bytes match a loop."""
    # amax = 6.0 ⇒ scale 2^0, so q = v and each value lands on its own magnitude.
    block = [
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.0,
        -1.0,
        1.5,
        -1.5,
        2.0,
        -2.0,
        3.0,
        -3.0,
        4.0,
        -4.0,
        6.0,
        -6.0,
        0.4,
        -0.6,
        1.4,
        -2.6,
        3.4,
        -4.2,
        5.4,
        -5.6,
        0.1,
        -0.2,
        2.2,
        -3.2,
        4.4,
        -5.6,
        6.0,
        -0.0,
    ]
    assert max(abs(v) for v in block) == 6.0, "the block scale must be exactly 1.0"
    ref_packed, ref_code = _ref_block_mxfp4(block)

    packed, scales = mxfp.quantize_mxfp4(torch.tensor(block).reshape(1, 32))
    dequant = mxfp.dequantize_mxfp4(packed, scales, (1, 32))

    assert packed.flatten().tolist() == ref_packed, "packed bytes differ from the scalar loop"
    assert scales.flatten().tolist() == [ref_code] == [127], "amax 6 ⇒ e8m0 0x7F = 2^0"
    assert dequant.flatten().float().tolist() == _ref_block_values(block), "values differ"
    # ...and not a 3-level quantizer wearing an fp4 label (AGENTS §3.7 act-quant):
    codes = {c for byte in ref_packed for c in (byte & 0xF, byte >> 4)}
    assert {c & 7 for c in codes} == set(range(8)), "all 8 magnitudes must be reachable"
    assert codes == set(range(16)) - {8}, "each magnitude with each sign; no negative zero"
    assert 8 not in codes, "a zero magnitude carries no sign (code 8 is the unused one)"


def test_mxfp4_random_blocks_match_scalar_reference_bit_for_bit() -> None:
    """64 random blocks: codes and decoded values equal the scalar loop exactly."""
    torch.manual_seed(7)
    w = torch.randn(64, 32) * 3.0
    packed, scales = mxfp.quantize_mxfp4(w)
    dq = mxfp.dequantize_mxfp4(packed, scales, tuple(w.shape))
    for i, row in enumerate(w.tolist()):
        ref_packed, ref_code = _ref_block_mxfp4(row)
        assert packed[i].tolist() == ref_packed, f"block {i}: packed bytes differ"
        assert scales[i].flatten().tolist() == [ref_code], f"block {i}: scale byte differs"
        assert dq[i].float().tolist() == _ref_block_values(row), f"block {i}: values differ"


def test_mxfp4_ties_round_to_even_mantissa() -> None:
    """RNE on the magnitude (`S E1M2`: index parity == mantissa parity).

    amax = 12 ⇒ scale 2^1, so the block's values are 2× the grid's midpoints.
    §5.3.3 Table 5 requires roundTiesToEven, and ties-low (tsct.py's NVFP4
    convention) answers differently at three of these values.
    """
    block = [12.0, 1.5, 2.5, 5.0, 7.0, 10.0, 0.5, -1.5] + [0.0] * 24
    packed, scales = mxfp.quantize_mxfp4(torch.tensor(block).reshape(1, 32))
    assert scales.flatten().tolist() == [128], "12/6 = 2.0 ⇒ e8m0 0x80"
    got = mxfp.dequantize_mxfp4(packed, scales, (1, 32)).float().flatten()[:8].tolist()
    # q = value/2: 6.0→6.0, 0.75→1.0, 1.25→1.0, 2.5→2.0, 3.5→4.0, 5.0→4.0,
    # 0.25→0.0 (index 0 is even), and the sign is a bit, not a second table.
    assert got == [12.0, 2.0, 2.0, 4.0, 8.0, 8.0, 0.0, -2.0]
    assert got == _ref_block_values(block)[:8], "the reference must agree on ties"


def test_e2m1_grid_is_the_tsct_grid_single_sourced() -> None:
    """One grid, imported — not a second table that can drift (AGENTS §3.7)."""
    assert mxfp.E2M1_MAGNITUDES is _NVFP4_MAGNITUDES
    assert mxfp.E2M1_MAGNITUDES == E2M1
    assert mxfp.E2M1_MAX == 6.0 and mxfp.E4M3_MAX == 448.0 == E4M3_REFERENCE[0x7E]
    assert mxfp.BLOCK == 32


# --------------------------------------------------------------------------
# MXFP8, and the 256-code e4m3 cross-check
# --------------------------------------------------------------------------


def test_e4m3_reference_table_matches_torch_on_all_256_codes() -> None:
    """Pure-integer check of the e4m3 grid against `torch.float8_e4m3fn`."""
    codes = torch.arange(256, dtype=torch.uint8)
    for code, torch_value in enumerate(codes.view(torch.float8_e4m3fn).float().tolist()):
        ref = E4M3_REFERENCE[code]
        if math.isnan(ref):
            assert math.isnan(torch_value), f"code {code:#04x}: torch {torch_value}, ref NaN"
        else:
            assert ref == torch_value, f"code {code:#04x}: torch {torch_value}, ref {ref}"
        got = codes[code : code + 1].view(torch.uint8).item()
        assert got == code  # the bit pattern really is the code (integer path, no cast)
    # ...and the table's own anchors against §5.3.1 Table 2, not only against torch.
    assert E4M3_REFERENCE[0x00] == 0.0
    assert E4M3_REFERENCE[0x01] == 2.0**-9  # min subnorm
    assert E4M3_REFERENCE[0x08] == 2.0**-6  # min normal
    assert E4M3_REFERENCE[0x7E] == 448.0  # max normal
    assert math.isnan(E4M3_REFERENCE[0x7F]) and math.isnan(E4M3_REFERENCE[0xFF])


def test_e4m3_scalar_reference_agrees_with_torch_rounding() -> None:
    """The test's own RNE reference vs torch's cast on 20k values (both checked)."""
    torch.manual_seed(11)
    q = (torch.rand(20000, dtype=torch.float64) * 2.0 - 1.0).to(torch.float32) * 500.0
    torch_codes = q.to(torch.float8_e4m3fn).view(torch.uint8).tolist()
    for value, torch_code in zip(q.tolist(), torch_codes, strict=True):
        want = (0x80 if value < 0 else 0x00) | _ref_e4m3_code(abs(value))
        assert want == torch_code, f"{value}: ref {want:#04x} != torch {torch_code:#04x}"


def test_mxfp8_block_matches_scalar_reference() -> None:
    """Hand block: the codec's bytes are the reference's codes of `q = v/scale`."""
    block = [4096.0, -4096.0, 453.0, 33.0, 1.0, 0.5, -0.3, 0.0] + [0.0] * 24
    packed, scales = mxfp.quantize_mxfp8(torch.tensor(block).reshape(1, 32))
    assert scales.flatten().tolist() == [131], "4096/448 = 9.14 ⇒ 2^4 ⇒ e8m0 0x83"
    scale = 2.0**4
    want = [(0x80 if v < 0 else 0x00) | _ref_e4m3_code(abs(v) / scale) for v in block]
    assert packed.flatten().view(torch.uint8).tolist() == want
    dq = mxfp.dequantize_mxfp8(packed, scales, (1, 32)).float().flatten().tolist()
    assert dq == [(E4M3_REFERENCE[c & 0x7F] * (-1.0 if c & 0x80 else 1.0)) * scale for c in want]
    assert dq[0] == 4096.0, "the block max round-trips exactly at its own scale"
    assert dq[2] == pytest.approx(453.0, rel=0.1)


def test_mxfp8_roundtrip_shapes_and_blocks() -> None:
    """Multi-block 3D: element count and the per-32 scale grid both hold."""
    torch.manual_seed(3)
    w = torch.randn(2, 3, 64)
    elems, scales = mxfp.quantize_mxfp8(w)
    assert elems.shape == w.shape and elems.dtype == torch.float8_e4m3fn
    assert scales.shape == (2, 3, 2) and scales.dtype == torch.uint8
    dq = mxfp.dequantize_mxfp8(elems, scales, tuple(w.shape))
    assert dq.shape == w.shape and dq.dtype == torch.bfloat16
    # One scale per 32 elements, shared by the whole block (§5.1).
    assert int(scales[0, 0, 1]) == _ref_scale_code(float(w[0, 0, 32:].abs().amax()), 448.0)
    # MXFP4 on the same shape: 2 codes/byte, one scale per block.
    packed, scales4 = mxfp.quantize_mxfp4(w)
    assert packed.shape == (2, 3, 32) and scales4.shape == (2, 3, 2)
    assert mxfp.dequantize_mxfp4(packed, scales4, tuple(w.shape)).shape == w.shape


# --------------------------------------------------------------------------
# (b) error bounds
# --------------------------------------------------------------------------


def test_error_bounds() -> None:
    """Measured relative error bounds — the table in this file's docstring."""
    torch.manual_seed(0)
    w = torch.randn(4096, 32)
    amax = w.abs().reshape(-1, 32).amax(-1, keepdim=True).expand(-1, 32)
    significant = w.abs() >= amax / 6
    assert 0.69 < float(significant.float().mean()) < 0.70, "gaussian sig fraction"
    # bounds: (rms rel, sig mean rel, sig max rel, mean |err|/amax, max |err|/amax, max rel)
    bounds = {
        "mxfp4": (0.13, 0.11, 0.34, 0.04, 0.21, 1.0),
        "mxfp8": (0.03, 0.023, 2.0**-4, 0.008, 0.06, 0.24),
    }
    for name, quant, dequant in (
        ("mxfp4", mxfp.quantize_mxfp4, mxfp.dequantize_mxfp4),
        ("mxfp8", mxfp.quantize_mxfp8, mxfp.dequantize_mxfp8),
    ):
        packed, scales = quant(w)
        dq = dequant(packed, scales, tuple(w.shape))
        assert dq.dtype == torch.bfloat16 and dq.shape == w.shape
        dq = dq.float()
        rel = (dq - w).abs() / w.abs().clamp_min(1e-12)
        rms = (dq - w).pow(2).mean().sqrt() / w.pow(2).mean().sqrt()
        over_amax = (dq - w).abs() / amax
        assert float(rms) < bounds[name][0], f"{name} rms rel {float(rms)}"
        assert float(rel[significant].mean()) < bounds[name][1], f"{name} sig mean rel"
        assert float(rel[significant].max()) < bounds[name][2], f"{name} sig max rel"
        assert float(over_amax.mean()) < bounds[name][3], f"{name} mean |err|/amax"
        assert float(over_amax.max()) < bounds[name][4], f"{name} max |err|/amax"
        assert float(rel.max()) <= bounds[name][5], f"{name} max rel {float(rel.max())}"


# --------------------------------------------------------------------------
# (c) STE
# --------------------------------------------------------------------------


def test_fake_quant_ste_gradient_is_identity() -> None:
    """`fake_quant(w).sum().backward()` ⇒ `w.grad == 1` everywhere."""
    for name in ("mxfp4", "mxfp8"):
        quant = getattr(mxfp, f"quantize_{name}")
        dequant = getattr(mxfp, f"dequantize_{name}")
        fake_quant = getattr(mxfp, f"fake_quant_{name}")
        torch.manual_seed(5)
        x = torch.randn(4, 64, requires_grad=True)
        out = fake_quant(x)
        assert out.shape == x.shape and out.requires_grad
        out.sum().backward()
        assert x.grad is not None
        assert torch.equal(x.grad, torch.ones_like(x)), f"{name}: grad is not identity"
        # The forward really is the quantized value, not a passthrough.
        packed, scales = quant(x.detach())
        assert torch.equal(out.detach().to(torch.bfloat16), dequant(packed, scales, tuple(x.shape)))
        assert not torch.equal(out.detach(), x.detach()), f"{name}: fake-quant moved nothing"


# --------------------------------------------------------------------------
# (d) zero block + power-of-two scales
# --------------------------------------------------------------------------


def test_zero_block_roundtrips_to_exact_zeros() -> None:
    """The declared zero encoding (e8m0 0x00) decodes to exact zeros."""
    zeros = torch.zeros(3, 64)
    for name, quant, dequant in (
        ("mxfp4", mxfp.quantize_mxfp4, mxfp.dequantize_mxfp4),
        ("mxfp8", mxfp.quantize_mxfp8, mxfp.dequantize_mxfp8),
    ):
        packed, scales = quant(zeros)
        assert scales.shape == (3, 2)
        assert scales.flatten().tolist() == [0, 0] * 3, "a zero block declares e8m0 0x00"
        if name == "mxfp4":
            assert set(packed.flatten().tolist()) == {0}, "zero codes carry no sign"
        dq = dequant(packed, scales, tuple(zeros.shape))
        assert torch.equal(dq, torch.zeros_like(dq)), "exact zeros, not 2^-127-scale noise"
    # A block that is only PARTLY zero still gets a real power-of-two scale.
    mixed = torch.zeros(1, 64)
    mixed[0, 33] = 3.0
    packed, scales = mxfp.quantize_mxfp4(mixed)
    assert scales.flatten().tolist() == [0, 126], "3/6 = 0.5 ⇒ 2^-1 ⇒ e8m0 0x7E"
    assert mxfp.dequantize_mxfp4(packed, scales, (1, 64)).float().flatten()[33].item() == 3.0


def test_scale_bytes_decode_to_powers_of_two() -> None:
    """Every non-zero scale byte is exactly `2**(code-127)`; 0x00 is the zero."""
    torch.manual_seed(13)
    # 512 blocks whose amax spans 2^-60 … 2^60, so the scale grid is walked
    # widely (121 distinct codes measured; the assert below is the smoke check).
    w = torch.randn(512, 32) * (2.0 ** torch.linspace(-60, 60, 512)).reshape(-1, 1)
    for quant in (mxfp.quantize_mxfp4, mxfp.quantize_mxfp8):
        _, scales = quant(w)
        codes = scales.flatten().tolist()
        values = mxfp.e8m0_decode(scales).flatten().tolist()
        assert len(set(codes)) > 100, "the scale grid must actually be exercised"
        assert all(1 <= c <= 0xFE for c in codes), "0x00 is the zero, 0xFF the spec's NaN"
        for code, value in zip(codes, values, strict=True):
            assert value == 2.0 ** (code - 127), f"code {code:#04x} → {value}"
            assert value == 2.0 ** int(math.log2(value)), "must be a power of two"
    assert mxfp.e8m0_decode(torch.zeros(2, dtype=torch.uint8)).tolist() == [0.0, 0.0]
    assert mxfp.e8m0_decode(torch.tensor([1, 127, 254], dtype=torch.uint8)).tolist() == [
        2.0**-126,
        1.0,
        2.0**127,
    ]


# --------------------------------------------------------------------------
# (e) LOUD
# --------------------------------------------------------------------------


def test_loud_on_non_divisible_last_dim() -> None:
    for shape in ((3, 33), (31,), (2, 3, 5461), (0,)):
        for quant in (mxfp.quantize_mxfp4, mxfp.quantize_mxfp8):
            with pytest.raises(LoudError, match="divisible by 32"):
                quant(torch.zeros(*shape))
    with pytest.raises(LoudError, match="5461"):
        mxfp.quantize_mxfp4(torch.zeros(2, 3, 5461))
    with pytest.raises(LoudError, match="pad"):
        mxfp.quantize_mxfp4(torch.zeros(3, 33))
    with pytest.raises(LoudError):
        mxfp.quantize_mxfp4(torch.tensor(1.0))  # 0-dim: no last dim at all


def test_loud_on_nan_and_inf_input() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        w = torch.ones(1, 32)
        w[0, 5] = bad
        for quant in (mxfp.quantize_mxfp4, mxfp.quantize_mxfp8):
            with pytest.raises(LoudError, match="non-finite"):
                quant(w)
        with pytest.raises(LoudError, match="non-finite"):
            mxfp.fake_quant_mxfp4(w)


def test_loud_on_denormal_amax_block() -> None:
    """A block whose scale needs e8m0 < 2^-126 refuses (its code 0x00 is zero)."""
    w = torch.full((1, 32), 1e-40)
    for quant in (mxfp.quantize_mxfp4, mxfp.quantize_mxfp8):
        with pytest.raises(LoudError, match="exponent range"):
            quant(w)
    with pytest.raises(LoudError, match="prescale"):
        mxfp.quantize_mxfp4(w)  # the message names the escape


def test_loud_on_e8m0_nan_byte_and_bad_shapes() -> None:
    with pytest.raises(LoudError, match="NaN"):
        mxfp.e8m0_decode(torch.tensor([0xFF], dtype=torch.uint8))
    packed = torch.zeros(16, dtype=torch.uint8)
    scales = torch.zeros(1, dtype=torch.uint8)
    with pytest.raises(AssertionError):
        mxfp.dequantize_mxfp4(packed, scales, (1, 64))  # 64 elements ≠ 32 covered
    with pytest.raises(AssertionError):
        mxfp.dequantize_mxfp4(torch.zeros(8, dtype=torch.uint8), scales, (1, 32))
    with pytest.raises(AssertionError):
        mxfp.dequantize_mxfp4(packed.float(), scales, (1, 32))  # storage must be uint8
    with pytest.raises(AssertionError):
        mxfp.dequantize_mxfp8(torch.zeros(32, dtype=torch.float8_e4m3fn), scales.float(), (1, 32))


# --------------------------------------------------------------------------
# (f) byte budget — the VRAM argument
# --------------------------------------------------------------------------


def test_byte_budget_mxfp4() -> None:
    """32 elements ⇒ 16 B codes + 1 B scale = 17 B, INCLUDING the scale bytes."""
    assert mxfp.bytes_per_elements_mxfp4(32) == 17
    assert mxfp.bytes_per_elements_mxfp4(0) == 0
    assert mxfp.bytes_per_elements_mxfp4(33) == 34, "a partial block is counted whole"
    n = 32 * 1024 * 1024
    assert mxfp.bytes_per_elements_mxfp4(n) == 17 * 1024 * 1024 == 17_825_792
    assert mxfp.bytes_per_elements_mxfp4(n) == n // 2 + n // 32  # hand arithmetic
    # THE VRAM ARGUMENT: b1's routed-expert bank is 1024 × [r=64, f=5461]
    # (configs/b1.toml:100) = 357 892 096 elements = 1 431 568 384 B fp32
    # (1.333 GiB). MXFP4 flat: 190 130 176 B. But 5461 % 32 != 0 and MX has no
    # partial block, so every row pads to 171 blocks: 65 536 rows × 171 × 17 =
    # 190 513 152 B (181.7 MiB) — a 7.51x cut. The PADDED number is the one a
    # VRAM plan must use (the caller pads; mxfp refuses instead).
    bank = 1024 * 64 * 5461
    assert bank == 357_892_096
    assert mxfp.bytes_per_elements_mxfp4(bank) == 190_130_176
    padded = 1024 * 64 * math.ceil(5461 / 32) * 17
    assert padded == 190_513_152
    assert 4 * bank == 1_431_568_384
    assert 4 * bank / padded == pytest.approx(7.5143, abs=1e-4)


def test_byte_budget_mxfp8() -> None:
    """32 elements ⇒ 32 B elements + 1 B scale = 33 B, INCLUDING the scale."""
    assert mxfp.bytes_per_elements_mxfp8(32) == 33
    assert mxfp.bytes_per_elements_mxfp8(0) == 0
    n = 32 * 1024 * 1024
    assert mxfp.bytes_per_elements_mxfp8(n) == 33 * 1024 * 1024 == 34_603_008
    assert mxfp.bytes_per_elements_mxfp8(n) == n + n // 32  # hand arithmetic
    assert mxfp.bytes_per_elements_mxfp8(n) > mxfp.bytes_per_elements_mxfp4(n)
    # 1024 x 64 x 5461 padded per row: 65 536 × 171 × 33 = 369 819 648 B = 3.87x
    # under the 1 431 568 384 B fp32 bank.
    padded = 1024 * 64 * math.ceil(5461 / 32) * 33
    assert padded == 369_819_648
    assert 1_431_568_384 / padded == pytest.approx(3.8710, abs=1e-4)
    with pytest.raises(AssertionError):
        mxfp.bytes_per_elements_mxfp8(-1)
