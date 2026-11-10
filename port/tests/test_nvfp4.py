"""Gates for `model/nvfp4.py` — the e2m1/e4m3 packing and the fp4 `_scaled_mm` lane.

The format is fixed by the kernel, not by us: e2m1 elements, ONE e4m3 scale per
16 logical elements, two nibbles per byte (`float4_e2m1fn_x2`). These gates pin
the parts a packing bug would move silently:

- the 16 e2m1 levels (`tsct._NVFP4_MAGNITUDES`, the codebook
  `act_quant._e2m1_round` rounds onto) are the ONLY reachable magnitudes, and a
  hand-computed 2-block example reproduces the arithmetic expected in the test;
- the nibble order (element 0 in the LOW nibble) and the per-16 scale layout are
  pinned on a hand-built byte, because a swap is invisible to any round-trip
  that packs and unpacks with the same convention;
- `dequant(quant(x))` stays inside the format's own error bound;
- the kernel lane and the emulated lane agree on the SAME packed inputs (the
  GPU half; skipped on a CPU-only box), and the counters NAME the path;
- no host read: the second-level scale is a device scalar.
"""

from __future__ import annotations

import pytest
import torch

from dormouse.instrument import Counters
from dormouse.model.nvfp4 import (
    E2M1_MAX,
    GROUP,
    bytes_per_elements,
    dequantize_nvfp4,
    emulated_mm,
    nvfp4_fallbacks,
    nvfp4_mm,
    quantize_nvfp4,
)
from dormouse.model.tsct import _NVFP4_MAGNITUDES

GRID = _NVFP4_MAGNITUDES


def test_the_format_is_16_wide_and_six_high():
    """The two numbers the kernel reads: block width and the grid's top."""
    assert GROUP == 16
    assert E2M1_MAX == 6.0 and GRID[-1] == 6.0
    # 4-bit data + e4m3 scale per 16 = 0.5 + 0.0625 B/element (2.84x under bf16).
    assert bytes_per_elements(1024) == pytest.approx(0.5625)
    assert bytes_per_elements(16) == pytest.approx(0.5625)


def test_hand_computed_two_blocks():
    """K=32, two blocks, amax chosen so both scales are exact in e4m3.

    Block 0: amax 6.0 → scale 6/6 = 1.0; block 1: amax 3.0 → scale 3/6 = 0.5.
    The EXPECTED values are arithmetic written out here (`grid_value × scale`),
    never a golden captured from this module. The three ties in the input pin
    the format's rule: 0.25 → 0 (even code), 0.75 → 1.0, and 1.5/2.5 are exact
    grid points in the second block's units.
    """
    x = torch.zeros(1, 32)
    x[0, 0:16] = torch.tensor(
        [6.0, 3.0, 1.5, 0.75, 0.25, 0.0, -6.0, -1.5, 2.0, 0.5, 4.0, -0.75, -0.25, -3.0, 1.0, -2.0]
    )
    x[0, 16:32] = torch.tensor(
        [
            3.0,
            1.5,
            0.75,
            0.5,
            0.25,
            0.0,
            -3.0,
            -1.5,
            -0.75,
            1.0,
            2.0,
            -0.5,
            -0.25,
            0.125,
            -2.0,
            0.25,
        ]
    )
    packed, scales, g = quantize_nvfp4(x)

    assert packed.shape == (1, 16) and packed.dtype == torch.uint8
    assert scales.shape == (1, 2) and scales.dtype == torch.float8_e4m3fn
    assert float(g) == 1.0, "amax 6 is e4m3-safe: no second-level scale"
    assert scales.float()[0].tolist() == [1.0, 0.5]

    want_block0 = [
        6.0,
        3.0,
        1.5,
        1.0,
        0.0,
        0.0,
        -6.0,
        -1.5,
        2.0,
        0.5,
        4.0,
        -1.0,
        0.0,
        -3.0,
        1.0,
        -2.0,
    ]
    # block 1: x/0.5 lands on codes, dequant = code × 0.5.
    want_block1 = [
        3.0,
        1.5,
        0.75,
        0.5,
        0.25,
        0.0,
        -3.0,
        -1.5,
        -0.75,
        1.0,
        2.0,
        -0.5,
        -0.25,
        0.0,
        -2.0,
        0.25,
    ]
    got = dequantize_nvfp4(packed, scales, torch.float32, global_scale=g)
    assert torch.equal(got, torch.tensor([want_block0 + want_block1]))


def test_nibble_order_and_scale_layout_are_the_kernel_layout():
    """Element 0 in the LOW nibble — pinned on one hand-built byte.

    `[-6.0, 0.5]` in a 16-wide block (amax 6 → scale 1.0): `6.0` is grid index 7
    (the code IS the bit pattern: exp 3, mantissa 1) plus the sign bit 8 → 15
    in the low nibble, `0.5` is index 1 → high nibble 1, so the byte is
    `1 << 4 | 15 = 0x1F`. A swap of the two nibbles reads 0xF1 instead.
    """
    x = torch.zeros(1, 32)
    x[0, 0] = -6.0
    x[0, 1] = 0.5
    packed, scales, _ = quantize_nvfp4(x)
    assert int(packed[0, 0]) == 0x1F, f"byte {int(packed[0, 0]):#04x}"
    assert scales.float()[0, 0] == 1.0


def test_every_quantized_value_is_a_grid_point_times_its_scale():
    torch.manual_seed(0)
    x = torch.randn(4, 96) * 3.0
    packed, scales, g = quantize_nvfp4(x)
    dq = dequantize_nvfp4(packed, scales, torch.float32, global_scale=g)
    norm = dq / (scales.float().repeat_interleave(GROUP, dim=-1) * g)
    grid = torch.tensor(GRID)
    on_grid = torch.zeros_like(norm, dtype=torch.bool)
    for v in grid:
        on_grid |= (norm.abs() - v).abs() < 1e-6
    assert bool(on_grid.all()), "a dequantized value left the e2m1 grid"
    # All eight magnitudes are REACHABLE on a random tensor (the 3-level defect's
    # gate): a quantizer that only ever emits {0, 0.5·s, s} fails here.
    codes = torch.searchsorted(grid, norm.abs().reshape(-1).contiguous())
    assert set(codes.unique().tolist()) == set(range(8)), codes.unique()


def test_dequant_quant_relative_error_is_bounded():
    """The format's own bound: |x_q − x| ≤ one scale step, plus the rms error.

    In normalized units (block max = 6) the worst gap between adjacent grid
    points is 2 (between 4 and 6), so half a step is 1 — i.e. one BLOCK SCALE.
    The scale itself is e4m3, so its own rounding is a second, smaller term
    (`amax/6·2^-4` at most); measuring against the scale the module actually
    rounded to keeps this test about the GRID, not about e4m3.
    """
    torch.manual_seed(0)
    x = torch.randn(16, 128) * 2.0
    packed, scales, g = quantize_nvfp4(x)
    dq = dequantize_nvfp4(packed, scales, torch.float32, global_scale=g)
    step = scales.float().repeat_interleave(GROUP, dim=-1) * g
    assert bool(((dq - x).abs() <= step + 1e-6).all())
    # ... and the scale is `amax/6` to within e4m3's own 2^-4 half-ulp.
    block_max = (
        x.reshape(16, 8, 16).abs().amax(-1, keepdim=True).repeat_interleave(16, -1).reshape(x.shape)
    )
    assert bool(((step - block_max / 6).abs() <= block_max / 6 * 2**-4 + 1e-6).all())
    rel = ((dq - x).norm() / x.norm()).item()
    assert 0.02 < rel < 0.2, f"rms relative error {rel:.3f} is outside the format's band"


def test_second_level_scale_absorbs_an_e4m3_overflow():
    """amax 6000 → block scale 1000 > 448: `g` carries the overflow, not a clamp."""
    x = torch.full((1, 16), 6000.0)
    x[0, 0] = -6000.0
    packed, scales, g = quantize_nvfp4(x)
    assert float(g) > 1.0
    assert float(scales.float().max()) <= 448.0
    dq = dequantize_nvfp4(packed, scales, torch.float32, global_scale=g)
    assert torch.allclose(dq, x, rtol=1e-3, atol=0), dq


def test_k_must_tile_the_block():
    with pytest.raises(AssertionError, match="K % 16"):
        quantize_nvfp4(torch.randn(2, 40))


def test_no_host_read_in_the_helper(monkeypatch):
    """The second-level scale is a DEVICE scalar: no `.item()` anywhere in a call."""

    def boom(*_a, **_k):
        raise AssertionError("host read in nvfp4_mm")

    for name in ("item", "tolist", "cpu", "numpy", "__float__", "__int__"):
        monkeypatch.setattr(torch.Tensor, name, boom)
    nvfp4_mm(torch.randn(8, 32), torch.randn(4, 32), torch.float32, Counters())


def test_the_fallback_names_itself_and_its_reason():
    c = Counters()
    out = nvfp4_mm(torch.randn(8, 32), torch.randn(4, 32), torch.float32, c)
    assert out.shape == (8, 4)
    assert c.get("nvfp4_emulated", 0) == 1 and c.get("nvfp4_gemm", 0) == 0
    assert "not CUDA" in getattr(c, "why", "") or "fp32" in getattr(c, "why", "")
    assert nvfp4_fallbacks(), "the process-local tally must see the call too"


def test_a_non_contracting_weight_is_refused_by_name_not_by_the_fallback():
    """The MEASURED failure: `nvfp4_mm(x[24576,832], w[832,2048])` raised a raw
    torch shape error from inside `_dequant_mm`.

    A `[K, N]` weight cannot be packed (`quantize_nvfp4` packs the contraction,
    which is the LAST axis), so `_reason` called the pair "the operands do not
    contract" — a fallback REASON — and the emulated lane, whose whole job is to
    be what the kernel would have computed, then transposed a weight that is
    already `[K, N]` into `[N, K]`. Both halves are bugs: a lane that runs on a
    shape it cannot mean silently changes what "emulated" means. This gate is
    the exact pair, and it asserts nothing was emulated.
    """
    c = Counters()
    before = nvfp4_fallbacks()
    with pytest.raises(AssertionError, match=r"\[N, K\]"):
        nvfp4_mm(torch.randn(24576, 832), torch.randn(832, 2048), torch.float32, c)
    assert c.get("nvfp4_emulated", 0) == 0, "the fallback ran on an impossible pair"
    assert nvfp4_fallbacks() == before, "the refusal must not be counted as a fallback"
    # ... and the same pair, contractible, is accepted: the gate is the SHAPE.
    out = nvfp4_mm(torch.randn(8, 832), torch.randn(2048, 832), torch.float32, Counters())
    assert out.shape == (8, 2048)


def test_the_16_e2m1_levels_are_hit_exactly():
    """Hand-picked values: every magnitude of the grid, both signs, one block.

    The block's amax is 6.0, so the scale is exactly 1.0 and each value must
    come back bit-identical — a codec that rounds onto a shifted ladder fails
    here even though a random round-trip looks fine (the measured 2.0-for-6.0
    bug in a kernel-side magnitude table did exactly that).
    """
    vals = torch.tensor([0.0, *GRID, *[-v for v in GRID[1:]]], dtype=torch.float32)
    x = torch.zeros(1, 16)
    x[0, : vals.numel()] = vals
    packed, scales, g = quantize_nvfp4(x)
    assert float(g) == 1.0 and scales.float()[0, 0] == 1.0
    got = dequantize_nvfp4(packed, scales, torch.float32, global_scale=g)
    assert torch.equal(got[0, : vals.numel()], vals), (got[0].tolist(), vals.tolist())


def test_the_scale_swizzle_is_the_512_byte_tile():
    """One 128-row x 4-column tile: element (m, k) sits at `(m%32)*16 + (m//32)*4 + k%4`.

    That closed form is `_blocked_scale`'s body for a single tile (rows < 128,
    scale columns < 4), written out as arithmetic here rather than compared
    against the function itself.
    """
    from dormouse.model.nvfp4 import _blocked_scale

    s = torch.arange(128 * 4, dtype=torch.float32).reshape(128, 4)
    s8 = (s + 1.0).to(torch.float8_e4m3fn)
    flat = _blocked_scale(s8).view(torch.uint8).to(torch.int64)
    assert flat.numel() == 512
    for m in (0, 1, 31, 32, 33, 127):
        for j in range(4):
            want = int((s[m, j] + 1.0).to(torch.float8_e4m3fn).view(torch.uint8))
            assert int(flat[(m % 32) * 16 + (m // 32) * 4 + j]) == want, (m, j)


# ---------------------------------------------------------------------------
# the kernel lane: GPU only, and it must agree with the emulated lane
# ---------------------------------------------------------------------------

HAS_FP4 = hasattr(torch, "float4_e2m1fn_x2") and torch.cuda.is_available()


def triton_ok() -> bool:
    from dormouse.model import nvfp4

    return nvfp4._HAS_TRITON


#: The fused lane needs the card AND triton; the CPU lane is the reference it is
#: compared against, so every gate below is a CROSS-LANE equality, not a golden.
HAS_FUSED = HAS_FP4 and triton_ok()


@pytest.mark.skipif(not HAS_FUSED, reason="needs the CUDA fused pack kernel")
def test_the_fused_pack_and_unpack_are_bit_identical_to_the_cpu_lane():
    """The CUDA lane IS the CPU lane: same bytes, same scales, same values.

    37x96 on purpose — rows do not tile `BLOCK_M` (4) and K does not tile
    `BLOCK_K` (128), so the masked tail is exercised, and the CPU body is the
    reference the hand-computed gates above pin (a second codebook in the kernel
    would show up here and nowhere else).
    """
    torch.manual_seed(0)
    x = (torch.randn(37, 96) * 3.0).bfloat16()
    pc, sc, gc = quantize_nvfp4(x)
    ph, sh, gh = quantize_nvfp4(x.cuda())
    assert torch.equal(ph.cpu(), pc), "packed bytes differ from the CPU lane"
    assert torch.equal(sh.view(torch.uint8).cpu(), sc.view(torch.uint8))
    assert float(gh) == float(gc)
    a = dequantize_nvfp4(ph, sh, torch.float32, global_scale=gh)
    b = dequantize_nvfp4(pc, sc, torch.float32, global_scale=gc)
    assert torch.equal(a.cpu(), b), "dequantized values differ from the CPU lane"


@pytest.mark.skipif(not HAS_FUSED, reason="needs the CUDA fused pack kernel")
def test_the_fused_dequant_agrees_with_the_pack_it_came_from():
    """`deq_dtype` is the SAME pass, so it must equal the unpack of the bytes."""
    torch.manual_seed(1)
    x = (torch.randn(129, 128, device="cuda") * 2.0).bfloat16()
    packed, scales, g, deq = quantize_nvfp4(x, deq_dtype=torch.bfloat16)
    assert torch.equal(deq, dequantize_nvfp4(packed, scales, torch.bfloat16, global_scale=g))
    # Relative to the UNQUANTIZED input: a 4-bit block format's own error band.
    rel = ((deq.float() - x.float()).norm() / x.float().norm()).item()
    assert rel < 0.2, f"rms relative error {rel:.3f} is outside the format's band"


@pytest.mark.skipif(not HAS_FUSED, reason="needs the CUDA fused swizzle kernel")
def test_the_fused_swizzle_is_the_same_512_byte_tile():
    """833x55: both paddings (128 rows, 4 scale columns) are exercised.

    A wrong swizzle "succeeds" and returns a 46%-error GEMM, so this is pinned
    as bytes against the torch body, not through a GEMM.
    """
    from dormouse.model.nvfp4 import _blocked_scale

    s = ((torch.arange(833 * 55, dtype=torch.float32).reshape(833, 55) % 200) + 1.0).to(
        torch.float8_e4m3fn
    )
    want = _blocked_scale(s).view(torch.uint8)
    got = _blocked_scale(s.cuda()).view(torch.uint8)
    assert torch.equal(got.cpu(), want)
    assert got.numel() == -(-833 // 128) * -(-55 // 4) * 512


@pytest.mark.skipif(not HAS_FUSED, reason="needs the CUDA fused lane")
def test_the_fused_lane_reads_nothing_back(monkeypatch):
    """ADR-0018 rule 2: a training-path call may not sync to the host.

    Covers the fused pack (with its dequantized output) AND the emulated latn's
    unpack — the two functions `packed_acts` calls four times per layer.
    """

    def boom(*_a, **_k):
        raise AssertionError("host read in the fused lane")

    x = torch.randn(8, 32, device="cuda", dtype=torch.bfloat16)
    for name in ("item", "tolist", "cpu", "numpy", "__float__", "__int__"):
        monkeypatch.setattr(torch.Tensor, name, boom)
    packed, scales, g, deq = quantize_nvfp4(x, deq_dtype=torch.bfloat16)
    dequantize_nvfp4(packed, scales, torch.bfloat16, global_scale=g)


@pytest.mark.skipif(not HAS_FP4, reason="needs a CUDA fp4 `_scaled_mm` (sm_120, torch >= 2.13)")
def test_kernel_agrees_with_the_emulated_lane_on_the_same_packed_inputs():
    """M = 129 on purpose: M is free, `N % 16` and `K % 32` are the rules."""
    torch.manual_seed(0)
    x = torch.randn(129, 128, device="cuda", dtype=torch.bfloat16) * 2.0
    w = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    c = Counters()
    fast = nvfp4_mm(x, w, torch.bfloat16, c)
    emu = emulated_mm(x, w, torch.bfloat16)
    assert c.get("nvfp4_gemm", 0) == 1, f"the kernel did not run: {getattr(c, 'why', '')}"
    assert c.get("nvfp4_emulated", 0) == 0
    rel = ((fast.float() - emu.float()).norm() / emu.float().norm()).item()
    assert rel < 1e-2, f"kernel vs emulated relative error {rel:.4f}"


@pytest.mark.skipif(not HAS_FP4, reason="needs a CUDA fp4 `_scaled_mm` (sm_120, torch >= 2.13)")
def test_a_tile_that_the_kernel_refuses_emulates_counted():
    """N=3 is refused by the fp4 kernel (`N%16`); the reason is named, not guessed."""
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    c = Counters()
    out = nvfp4_mm(x, w, torch.bfloat16, c)
    assert out.shape == (64, 3)
    assert c.get("nvfp4_emulated", 0) == 1 and c.get("nvfp4_gemm", 0) == 0
    assert "N % 16" in getattr(c, "why", ""), getattr(c, "why", "")


@pytest.mark.skipif(not HAS_FP4, reason="needs a CUDA fp4 `_scaled_mm` (sm_120, torch >= 2.13)")
def test_a_nonsquare_weight_cannot_hide_an_orientation_bug():
    """N != K, so `x @ W.T` and `x @ W` are different shapes, not just values.

    Checked against the FP32 EMULATED result, never a bf16 GEMM: a bf16 reference
    agrees with a wrong scale swizzle to 46% (measured) and would pass here.
    """
    torch.manual_seed(0)
    x = torch.randn(129, 128, device="cuda", dtype=torch.bfloat16) * 2.0
    w = torch.randn(96, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    c = Counters()
    fast = nvfp4_mm(x, w, torch.bfloat16, c)
    assert fast.shape == (129, 96)
    assert c.get("nvfp4_gemm", 0) == 1, getattr(c, "why", "")
    emu = emulated_mm(x, w, torch.bfloat16)
    rel = ((fast.float() - emu.float()).norm() / emu.float().norm()).item()
    assert rel < 1e-2, f"kernel vs emulated relative error {rel:.4f}"


@pytest.mark.skipif(not HAS_FP4, reason="needs a CUDA fp4 `_scaled_mm` (sm_120, torch >= 2.13)")
def test_autograd_falls_back_counted_and_the_values_still_match():
    """A training-shaped call (requires_grad) must NOT reach `_scaled_mm`."""
    x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    c = Counters()
    out = nvfp4_mm(x, w, torch.bfloat16, c)
    assert out.shape == (256, 128)
    assert c.get("nvfp4_emulated", 0) == 1 and c.get("nvfp4_gemm", 0) == 0
    assert "autograd" in getattr(c, "why", "")
