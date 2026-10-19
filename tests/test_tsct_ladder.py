"""The NVFP4 storage rung's small-block gate (2026-10-19 audit M1).

The pre-fix prescale was `2 ** finfo.min` = `2^-448` (finfo.min is the most
negative VALUE, −448, not an exponent), so it fired only for block amaxes
below ~1e-134: effectively never. A block with amax 1e-3 then had its e4m3
scale round to zero, `q = blocks / 0`, and the decode multiplied by 0 — the
whole block, silently, 64/64 elements.

These tests pin the two honest outcomes of the two-level format:
* a tensor whose blocks are all small IS carried by the global prescale
  (amax 1e-3 and, alone, 1e-9 round-trip);
* a block more than e4m3's ~18 octaves below the tensor's largest block
  cannot be carried by one global multiplier, so it is COUNTED
  (`nvfp4_zero_scale_blocks`) instead of silently zeroed.
"""

from __future__ import annotations

import torch

from dormouse.instrument import Counters
from dormouse.model.tsct import nvfp4_zeroed_blocks, pack_tsct, unpack_tsct

MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _grid_block(amax: float) -> torch.Tensor:
    """One 16-wide block holding every e2m1 magnitude and its negative mirror."""
    pos = [amax * m / 6.0 for m in MAGS]
    return torch.tensor([pos + [-v for v in pos]])


def test_nvfp4_1e_3_block_round_trips() -> None:
    """The audit's measured case: amax 1e-3 survives the save→load round trip."""
    block = _grid_block(1e-3)
    rec = pack_tsct(block, "nvfp4")
    got = unpack_tsct(rec)
    assert got.shape == block.shape and got.dtype == torch.float32
    assert rec["global_scale"] > 1.0, "a 1e-3 block must trigger the prescale"
    assert got.abs().max() > 0.0, "the block was zeroed (the pre-fix defect)"
    # The global is a power of two and the e4m3 scale is RNE: every value moves
    # by the SCALE's own rounding (e4m3 has a 3-bit mantissa), not to zero.
    rel = (got - block).norm() / block.norm()
    assert rel.item() < 5e-2, f"1e-3 block relative error {rel.item():.3e}"
    assert torch.allclose(got, block, rtol=5e-2, atol=0.0)


def test_nvfp4_1e_9_block_alone_round_trips() -> None:
    """Prescale is relative to the tensor's own max, so a tiny tensor works too."""
    block = _grid_block(1e-9)
    rec = pack_tsct(block, "nvfp4")
    got = unpack_tsct(rec)
    assert rec["global_scale"] > 1.0
    assert got.abs().max() > 0.0
    assert (got - block).norm() / block.norm() < 5e-2


def test_nvfp4_block_below_a_wide_tensor_is_counted_not_silent() -> None:
    """1e-3 + 1e-9 in one tensor: the 1e-3 block survives, the 1e-9 one is counted.

    One global multiplier has to put the LARGEST block scale at e4m3's max, so
    a block 6 decades lower is below the scale grid and decodes to zeros. That
    is the format's limit, not an implementation choice — but the caller must
    be able to see it (ADR-0019: COUNTED, never SILENT).
    """
    t = torch.zeros(2, 16)
    t[0] = _grid_block(1e-3)
    t[1] = torch.linspace(-1e-9, 1e-9, 16)
    counters = Counters()
    before = nvfp4_zeroed_blocks()
    rec = pack_tsct(t, "nvfp4", counters)
    got = unpack_tsct(rec)
    assert counters.get("nvfp4_zero_scale_blocks") == 1, counters
    assert nvfp4_zeroed_blocks() == before + 1, "the counter must be unconditional"
    assert got[0].abs().max() > 0.0 and torch.allclose(got[0], t[0], rtol=5e-2)
    assert got[1].abs().max().item() == 0.0, "the tiny block cannot be encoded"


def test_nvfp4_common_path_is_unchanged() -> None:
    """No block would round to zero ⇒ global 1.0 and the pre-fix record shape.

    This is the byte-identity half of the fix: a normal factor bank must not
    start carrying a global scale (and must not be re-encoded) just because the
    prescale bug was fixed.
    """
    torch.manual_seed(0)
    t = torch.randn(64, 4096) * 0.05
    rec = pack_tsct(t, "nvfp4")
    assert rec["global_scale"] == 1.0
    assert set(rec) == {"rung", "codes", "scale", "global_scale", "shape", "n"}
    got = unpack_tsct(rec)
    assert (got - t).norm() / t.norm() < 1e-1
    # Each decoded value is still a grid entry × its stored block scale.
    blocks = got.reshape(-1, 16)
    scales = rec["scale"].float().unsqueeze(-1)
    grid = torch.tensor(MAGS)
    dist = (blocks / scales).abs().unsqueeze(-1).sub(grid).abs().amin(-1).max()
    assert dist.item() < 1e-6, dist.item()
