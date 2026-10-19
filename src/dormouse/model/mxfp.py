"""OCP Microscaling (MX) formats as pure tensor math: MXFP4 (E2M1) + MXFP8 (E4M3).

NEW module, not wired into any model yet (a later lane does the integration):
this is the codec the b1 recipe's QAT stage needs — Kimi K3 §4.1.4 quantizes
expert weights to MXFP4 and the non-expert weights/activations to MXFP8, with
QAT starting at the SFT stage (`docs/architecture/post-training.md`).

FORMAT SOURCE — **OCP *Microscaling Formats (MX) Specification*, v1.0, Sep
2023**, fetched 2026-10-16 from
`https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf`
(16 pp.; read with `pdftotext -layout`, so the page/section numbers below are
that file's and every quoted line is verbatim):

* **§5.2 Table 1, p. 10** — the concrete formats, one row each:
  `MXFP8 | FP8 (E5M2)/FP8 (E4M3) | d=8 | k=32 | E8M0 | w=8` and
  `MXFP4 | FP4 (E2M1) | d=4 | k=32 | E8M0 | w=8` — i.e. **block size 32 and
  an E8M0 scale, 4 bits per MXFP4 element, 8 bits per MXFP8 element**.
  §5.2.1: "different encodings (e.g. E5M2 and E4M3 for MXFP8) are considered
  different formats" → this module implements MXFP8's **E4M3** encoding only.
* **§5.3.3 Table 5, p. 12** — E2M1: exponent bias 1, Infinities N/A, NaN N/A,
  zeros `S 00 0₂`, min normal `±1.0`, max subnorm `±0.5`, max normal `±6.0` ⇒
  the magnitude grid `(0, .5, 1, 1.5, 2, 3, 4, 6)`, and "No encodings are
  reserved for Inf or NaN in FP4" (why inf/nan input is a LOUD refusal here).
* **§5.3.1 Table 2, p. 11** (reproduced from the OCP FP8 spec) — E4M3: bias 7,
  Infinities N/A, NaN `S 1111 111₂`, max normal `±448`, min normal `±2⁻⁶`, min
  subnorm `±2⁻⁹`; no Inf encoding, so an inf input also refuses.
* **§5.4.1 Table 7, p. 14** — E8M0: exponent bias 127, supported exponent
  range **-127 to 127**, Infinities N/A, NaN `1111 1111₂` (0xFF), and
  **Zeros: N/A** — the scale format has no zero encoding at all, which is what
  forces the declared zero convention below.
* **§5.1, p. 9** — an MX block is `(X, [P_i])` with `v_i = X·P_i` for all k
  elements ("The scale factor X is shared across all k elements"), `X = NaN`
  makes the whole block NaN, and "The layout of the block in physical memory is
  not prescribed in this specification" (so the packing below is a convention,
  not a spec claim).

Also cross-checked against arXiv:2310.10537 (the MX paper, "Microscaling Data
Formats for Deep Learning", Rouhani et al.): §3.1 states the MX family is
"a per-block scaling factor" with **block size 32** and a power-of-two
(shared-exponent) scale. The spec above is the primary source; the paper agrees
on k=32 and the e8m0 rule, and is not cited for anything the spec does not say.

THE NUMERIC DECISIONS, each with its source (LOUD/COUNTED doctrine: every one
of these is either the spec's or marked as ours):

* **The element grid is not a second table.** `E2M1_MAGNITUDES` is imported
  from `dormouse/model/tsct.py:145` (`_NVFP4_MAGNITUDES`), which is the same
  8-magnitude ITU e2m1 grid. One grid, one place to fix — the repo already
  shipped a ~3-level quantizer wearing an fp4 label once (AGENTS §3.7 act-quant:
  0.75 was not a level of e2m1 and only `{0, .5, .75, 1}` were reachable), and
  a second copy of the grid is how that class of defect comes back.
* **Element rounding is roundTiesToEven on the magnitude**, as §5.3.3 requires
  ("An implementation must support the roundTiesToEven rounding mode"): the
  nearest grid magnitude, and at an exact midpoint the neighbour whose mantissa
  bit is 0. On this grid the magnitude index's parity IS the mantissa's parity
  (idx 1 = 0.5 = `S 00 1₂`, idx 2 = 1.0 = `S 01 0₂`, …), so the tie rule is
  `idx + (idx odd)`. Note this differs from tsct.py's NVFP4 path, which uses
  `searchsorted` ties-to-lower (`tsct.py:188-189`, NVIDIA's convention): MX is
  RNE, NVFP4 is left alone.
* **The scale is `ceil(log2(amax/elem_max))`** — the SMALLEST power of two that
  keeps the block inside the element grid, so `|P_i| ≤ elem_max` always and the
  spec's OVF column (Table 3, overflow→NaN) is unreachable; SAT is the only
  reachable overflow behaviour and it needs no branch, because the magnitude
  search already saturates at idx 7 = 6.0. The spec does not prescribe how the
  block scale is chosen (§5.1 leaves the layout to the implementation), so this
  line is **ours**, chosen for "no element ever overflows".
* **Declared ZERO: e8m0 code 0x00 means "this block is zero".** The spec's
  Table 7 says E8M0 has *no* zero encoding (0x00 is 2⁻¹²⁷) and no Inf, so a
  zero convention must be declared by the codec rather than inherited. We
  declare 0x00, and the consequence is pinned: an all-zero block decodes to
  exact zeros (`0 · P = 0`). Escape for a spec-exact consumer: read 0x00 as
  2⁻¹²⁷; the codec never emits it for a non-zero block (below).
* **e8m0 exponent range [-126, 127] for non-zero blocks** (codes `0x01..0xFE`,
  `2⁻¹²⁶ … 2¹²⁷`): -127 is excluded because its code 0x00 is our declared zero,
  and 0xFF is the spec's NaN which this codec never emits (on decode it is a
  LOUD refusal, §5.1 would propagate NaN to the whole block — a silent NaN for
  a reader who cannot tell why). A block whose amax needs exponent < -126 is a
  **denormal/tiny-amax block and refuses loudly**, escape: prescale the tensor
  by a power of two, which moves nothing (NVFP4's global prescale,
  `tsct.py:163-176`).
* **`bytes_per_elements_*` counts the scale bytes**: 32 elements ⇒ 16 B of fp4
  (2 codes/byte) + 1 B e8m0 = **17 B/32** (0.53125 B/elem) for MXFP4, and
  32 B (1 value/byte) + 1 B = **33 B/32** (1.03125 B/elem) for MXFP8. An
  unquantized fp32 master is 4 B/elem, so MXFP4 is a 7.53x storage cut and
  MXFP8 a 3.88x one.
* **Nibble order: element 2i in the LOW nibble** (`codes[0::2] | codes[1::2]<<4`),
  matching `tsct.py:192`. §5.1 again: the layout is ours to pick.
* **Sign is bit 3 of the nibble** (`S E1M2`), and a zero magnitude carries no
  sign — the code for a negative zero is 0x0, the canonical zero.
"""

from __future__ import annotations

import math

import torch

from ..instrument import GuardCadence, loud
from .tsct import _NVFP4_MAGNITUDES as E2M1_MAGNITUDES

#: The guard cadence (§1.3): the finite / e8m0-range / NaN-byte checks each
#: cost a device→host read (`bool(any())`), which is forbidden on the hot path.
#: They run on the first `first` calls and every `every`-th after that; in the
#: gap a non-finite value reaches the step's NaN firewall, and an out-of-range
#: e8m0 scale is handled WITHOUT a host read: the block is zeroed (the codec's
#: declared 0x00) and its count accumulates on device in `_RANGE_REJECTED`, so
#: the degradation is still COUNTED, never silent. `_GUARD.reset()` is the test
#: seam.
_GUARD = GuardCadence(first=8, every=1000)

#: COUNTED tally of blocks whose e8m0 exponent is out of range on an
#: off-cadence call. A DEVICE tensor, so the hot path adds `bad.sum()` with no
#: host read; [`range_rejected_blocks`] is the declared drain (a log/test
#: cadence), like `Counters.drain`.
_RANGE_REJECTED: dict[str, torch.Tensor] = {}


def range_rejected_blocks() -> int:
    """Blocks the e8m0 range guard would have refused, zeroed COUNTED.

    An off-cadence call cannot afford the `bool((off & ~zero).any())` host read,
    so it zeroes those blocks and tallies them here instead. Read this on a log
    or test cadence (it takes the one `.item()`), not per packed tensor.
    """
    return sum(int(t.item()) for t in _RANGE_REJECTED.values())


#: §5.2 Table 1: the scaling block size k for every concrete MX format.
BLOCK = 32

#: §5.3.3 Table 5: max normal of e2m1; §5.3.1 Table 2: max normal of e4m3.
E2M1_MAX = 6.0
E4M3_MAX = 448.0

#: §5.4.1 Table 7: exponent bias 127, NaN = 0xFF, and (our declaration, the
#: spec has none) 0x00 = zero block.
_E8M0_BIAS = 127
_E8M0_NAN = 0xFF
_E8M0_ZERO = 0x00
_E8M0_EXP_MIN = -126  # -127 is excluded: its code 0x00 is the declared zero
_E8M0_EXP_MAX = 127


def _blocks(t: torch.Tensor) -> torch.Tensor:
    """The last-dim 32-blocks of `t` as `[..., BLOCK]` fp32 — or LOUD.

    The caller pads, not this module: MX has no partial block (§5.2 Table 1
    fixes k=32 for both formats), and silently zero-padding would change the
    value of every element in the padded block's scale.
    """
    if t.dim() == 0 or t.shape[-1] == 0 or t.shape[-1] % BLOCK:
        last = t.shape[-1] if t.dim() else 0
        loud(
            f"MX formats need a last dim divisible by {BLOCK}, got shape "
            f"{tuple(t.shape)} (last dim {last}): pad the last dim up to a "
            f"multiple of {BLOCK} before calling (zero-padding the tail is the "
            "caller's decision, it moves the block amax)"
        )
    return t.float().reshape(-1, BLOCK)


def _e8m0_encode(amax: torch.Tensor, elem_max: float) -> torch.Tensor:
    """Per-block e8m0 scale code from the block amax (see the module doc)."""
    zero = amax == 0
    ratio = torch.where(zero, torch.ones_like(amax), amax / elem_max)
    exp = torch.ceil(torch.log2(ratio))
    off = (exp < _E8M0_EXP_MIN) | (exp > _E8M0_EXP_MAX)
    bad = off & ~zero
    # LOUD on the guard cadence; COUNTED (zero + device tally) in the gap. The
    # cadence may skip the HOST READ, never the consequence: a block a checked
    # call would refuse is zeroed here and counted in `_RANGE_REJECTED`, so it
    # cannot decode as plausible text and cannot vanish from the record (review
    # round 1 M4).
    if _GUARD.due():
        if bool(bad.any()):
            bad_exp = exp[bad][0]
            loud(
                f"MX block scale {float(bad_exp):.0f} is outside e8m0's exponent range "
                f"[{_E8M0_EXP_MIN}, {_E8M0_EXP_MAX}] (spec §5.4.1 Table 7; -127 is "
                "reserved as this codec's zero): the block amax is denormal/tiny. "
                "Escape: prescale the tensor by a power of two (a power of two moves "
                "no element — see tsct.py:163-176's NVFP4 global prescale)"
            )
    else:
        key = str(amax.device)
        prev = _RANGE_REJECTED.get(key)
        n = bad.sum()
        _RANGE_REJECTED[key] = n if prev is None else prev + n
    # `zero | bad` carries code 0x00 (the declared zero): the value this codec
    # writes for a block it cannot scale. No wrap through `exp + bias`.
    drop = zero | bad
    exp = torch.where(drop, torch.zeros_like(exp), exp)
    codes = torch.where(drop, torch.zeros_like(exp), exp + _E8M0_BIAS)
    return codes.to(torch.uint8)


def e8m0_decode(codes: torch.Tensor) -> torch.Tensor:
    """e8m0 codes → fp32 scale values: `2^(code-127)`, with 0x00 = exact zero.

    The inverse of the encode above, exposed because it is half of the codec
    (and the thing a kernel would do): 0x00 is our declared zero block, 0xFF is
    the spec's NaN and refuses loudly rather than propagating NaN into a whole
    block of values (§5.1: `X = NaN` ⇒ every `v_i` is NaN, which for a reader
    of our numbers is a silent corruption with an obvious cause we can name).
    """
    codes = codes.to(torch.int32)
    # Guard cadence (module _GUARD): one host read per 1000 decodes, not per
    # packed tensor; a 0xFF scale makes the block NaN either way.
    if _GUARD.due() and bool((codes == _E8M0_NAN).any()):
        loud(
            "e8m0 0xFF is the spec's NaN scale (§5.4.1 Table 7) and this codec "
            "never emits it: refusing to decode a NaN block. Cause: the scale "
            "bytes are not from this codec (or are uninitialized memory)"
        )
    assert codes.dtype == torch.int32
    # 2^(c-127) in fp32 is exact for every c in [0, 254] (integer exponent).
    values = torch.pow(2.0, (codes - _E8M0_BIAS).to(torch.float32))
    return torch.where(codes == _E8M0_ZERO, torch.zeros_like(values), values)


def _check_finite(w: torch.Tensor, what: str) -> None:
    """inf/NaN refuse loudly: neither element format has an Inf encoding.

    §5.3.3 Table 5 (e2m1) and §5.3.1 Table 2 (e4m3) both list Infinities N/A,
    so an inf input would silently become the grid's max (SAT) or NaN, and a
    NaN input would poison the block amax and therefore the scale.

    The finite test is a device→host read, so it runs on [`_GUARD`]'s cadence
    (§1.3): a NaN that slips between checks still reaches the step's NaN
    firewall. `what` is kept for the message.
    """
    assert w.is_floating_point(), f"MX quantize: {what} must be a float tensor, got {w.dtype}"
    if _GUARD.due() and not bool(torch.isfinite(w).all()):
        n_bad = int((~torch.isfinite(w)).sum())
        loud(
            f"MX quantize: {what} has {n_bad} non-finite element(s). MX e2m1/e4m3 "
            "have no Inf encoding (spec §5.3.3 Table 5, §5.3.1 Table 2) and a NaN "
            "would corrupt the block amax. Escape: sanitize the tensor (a NaN "
            "firewall belongs upstream — AGENTS §1.1), or mask the bad elements"
        )


def quantize_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4: `[..., n]` float → (uint8 packed `[..., n/2]`, uint8 scales `[..., n/32]`).

    RNE onto the e2m1 grid after the per-32 power-of-two e8m0 scale; element 2i
    in the low nibble, sign in bit 3 (see the module doc for every rule's
    source). The original dtype is irrelevant — the master stays the caller's.
    """
    _check_finite(w, "input")
    blocks = _blocks(w)
    amax = blocks.abs().amax(dim=-1)
    scale_codes = _e8m0_encode(amax, E2M1_MAX)
    scale = e8m0_decode(scale_codes).unsqueeze(-1)
    # A zero block (scale 0) would divide 0/0; its codes are 0 by convention, so
    # pin `q = 0` for it instead of letting a NaN through the search below.
    q = torch.where(scale == 0, torch.zeros_like(blocks), blocks / scale)
    mags = torch.as_tensor(E2M1_MAGNITUDES, device=q.device, dtype=q.dtype)
    mid = 0.5 * (mags[:-1] + mags[1:])
    lo = torch.searchsorted(mid, q.abs().reshape(-1)).reshape(q.shape)
    # RNE: at an exact midpoint the even-mantissa neighbour wins, and on this
    # grid "index even" == "mantissa bit 0" (mags[1]=0.5=S001, mags[2]=1.0=S010).
    at_mid = torch.gather(mid, 0, lo.flatten().clamp(max=mid.numel() - 1)).reshape(q.shape)
    tie = (lo < mid.numel()) & (q.abs() == at_mid)
    mag_idx = lo + (tie & (lo % 2 == 1)).to(lo.dtype)
    sign = (q < 0) & (mag_idx > 0)
    codes = (sign.to(torch.int32) * 8 + mag_idx).reshape(-1)
    assert codes.numel() == blocks.numel(), "one 4-bit code per element"
    packed = (codes[0::2] | (codes[1::2] << 4)).to(torch.uint8)
    tail = w.shape[:-1]
    return packed.reshape(*tail, -1), scale_codes.reshape(*tail, -1)


def dequantize_mxfp4(
    packed: torch.Tensor, scales: torch.Tensor, shape: tuple[int, ...]
) -> torch.Tensor:
    """MXFP4 `→` bf16 `[..., *shape]` — the inverse of [`quantize_mxfp4`]."""
    assert packed.dtype == torch.uint8 and scales.dtype == torch.uint8, (
        f"MXFP4 storage is uint8 packed codes + uint8 e8m0 scales, got "
        f"{packed.dtype}/{scales.dtype}"
    )
    n = math.prod(shape)
    assert scales.numel() * BLOCK == n, (
        f"scales cover {scales.numel() * BLOCK} elements but shape {tuple(shape)} has {n}"
    )
    assert packed.numel() * 2 == n, (
        f"packed holds {packed.numel() * 2} elements but shape {tuple(shape)} has {n}"
    )
    low = packed.to(torch.int32) & 0x0F
    high = (packed.to(torch.int32) >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(-1)
    mags = torch.as_tensor(E2M1_MAGNITUDES, device=packed.device, dtype=torch.float32)
    vals = mags[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    vals = vals.reshape(-1, BLOCK) * e8m0_decode(scales).reshape(-1).unsqueeze(-1)
    return vals.reshape(shape).to(torch.bfloat16)


def quantize_mxfp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 (E4M3): `[..., n]` float → (float8_e4m3fn `[..., n]`, uint8 `[..., n/32]`).

    torch 2.13 ships `float8_e4m3fn` and its cast is IEEE RNE with SAT on
    overflow (§5.3.1 Table 2's e4m3 grid, §5.3.3's "must support
    roundTiesToEven"), so the element encoding is torch's — a hand-written
    table would be a second grid with the same failure mode as above. The
    ceil-power-of-two scale keeps `|q| ≤ 448`, so SAT never binds.
    """
    _check_finite(w, "input")
    if not hasattr(torch, "float8_e4m3fn"):  # pragma: no cover - torch ≥ 2.1 has it
        loud(
            "MXFP8 needs torch.float8_e4m3fn (torch 2.13 ships it): this torch "
            "has no e4m3 dtype. Escape: the 256-code table in "
            "tests/test_mxfp.py::E4M3_REFERENCE is the ready reference if a "
            "manual e4m3 codec ever has to be carried here"
        )
    blocks = _blocks(w)
    amax = blocks.abs().amax(dim=-1)
    scale_codes = _e8m0_encode(amax, E4M3_MAX)
    scale = e8m0_decode(scale_codes).unsqueeze(-1)
    q = torch.where(scale == 0, torch.zeros_like(blocks), blocks / scale).to(torch.float8_e4m3fn)
    assert q.numel() == blocks.numel(), "one 8-bit element per value"
    tail = w.shape[:-1]
    return q.reshape(*tail, -1), scale_codes.reshape(*tail, -1)


def dequantize_mxfp8(
    elems: torch.Tensor, scales: torch.Tensor, shape: tuple[int, ...]
) -> torch.Tensor:
    """MXFP8 `→` bf16 `[..., *shape]` — the inverse of [`quantize_mxfp8`]."""
    assert elems.dtype == torch.float8_e4m3fn and scales.dtype == torch.uint8, (
        f"MXFP8 storage is float8_e4m3fn elements + uint8 e8m0 scales, got "
        f"{elems.dtype}/{scales.dtype}"
    )
    assert scales.numel() * BLOCK == math.prod(shape), (
        f"scales cover {scales.numel() * BLOCK} elements, shape {tuple(shape)} "
        f"has {math.prod(shape)}"
    )
    assert elems.numel() == math.prod(shape), (
        f"elements {elems.numel()} != numel of shape {tuple(shape)}"
    )
    vals = elems.to(torch.float32).reshape(-1, BLOCK) * e8m0_decode(scales).reshape(-1, 1)
    return vals.reshape(shape).to(torch.bfloat16)


def fake_quant_mxfp4(w: torch.Tensor) -> torch.Tensor:
    """STE fake-quant for MXFP4 — `w + (dequantize(quantize(w)) - w).detach()`.

    WHY this form: every op inside the quantizer (the e8m0 `ceil`, the RNE grid
    search, the nibble pack) is piecewise CONSTANT, so its true gradient is zero
    almost everywhere and undefined at the grid points — `dequantize(quantize(w))`
    alone would hand the fp32 master an exactly-zero gradient and the weights
    would never move under QAT. The `.detach()` on the residual makes the
    forward the quantized value and the backward the identity (`d/dw = 1`): the
    rounded value is a constant offset, and the master sees a straight-through
    estimator. This is the same object `act_quant` uses on the burn side.
    """
    packed, scales = quantize_mxfp4(w)
    dq = dequantize_mxfp4(packed, scales, tuple(w.shape)).to(w.dtype)
    return w + (dq - w).detach()


def fake_quant_mxfp8(w: torch.Tensor) -> torch.Tensor:
    """STE fake-quant for MXFP8 (E4M3) — see [`fake_quant_mxfp4`] for the why."""
    elems, scales = quantize_mxfp8(w)
    dq = dequantize_mxfp8(elems, scales, tuple(w.shape)).to(w.dtype)
    return w + (dq - w).detach()


def fake_quant_mx(w: torch.Tensor, fmt: str) -> torch.Tensor:
    """`fake_quant_mxfp4`/`fake_quant_mxfp8` for ANY last dim — pad, quantize, slice back.

    THE PADDING RULE, in one place for both consumers (the factor projector and
    the activation seam, so the two can never disagree): the last dim is
    zero-padded UP to a multiple of `BLOCK` (≤ 31 columns), the PADDED view is
    fake-quantized, and the result is sliced back to the original width. MX has
    no partial block (§5.2 Table 1 fixes k=32) and [`_blocks`] refuses one, so
    the caller decides — this is that decision, taken once.

    WHY PADDING IS EXACT HERE, not an approximation (`fake_quant_mxfp4`'s own
    contract is unchanged and still refuses a bad width):
    * the block scale is `ceil(log2(amax / elem_max))` and a zero never raises an
      `amax`, so every block holding a real element keeps the scale it would
      have had alone;
    * the pad is PER ROW — torch pads the last dim of every `[..., n]` row — so
      no block ever mixes two rows. Therefore the `[E, d, r]` expert bank
      quantizes to exactly the values E separate `[d, r]` slices get, for an
      ALIGNED `r` (`r % 32 == 0`, blocks already inside one row's last dim) and
      for a padded one alike. That is the equality that makes a banked expert
      identical to the same expert run alone, and it is pinned by test;
    * the STE residual is subtracted on the PADDED tensor and sliced AFTER, so
      every real element's gradient is exactly 1 (`fake_quant_mxfp4`'s
      straight-through) and the pad's is discarded.

    COST: ≤ 31 extra columns per row, i.e. one extra block per row of scale
    bytes — `bytes_per_elements_mxfp4` / `bytes_per_elements_mxfp8` already
    count a partial tail block whole, so the accounting above the codec is
    unchanged. A caller that would rather refuse calls
    `fake_quant_mxfp4`/`fake_quant_mxfp8` directly: `_blocks` names the shape
    and the escape.
    """
    fns = {"mxfp4": fake_quant_mxfp4, "mxfp8": fake_quant_mxfp8}
    if fmt not in fns:
        loud(
            f"fake_quant_mx: {fmt!r} is not an MX format. Known: {list(fns)} — "
            f"`mxfp4` e2m1 and `mxfp8` e4m3, the two concrete rows of §5.2 Table 1. "
            f"Escape: name one of those."
        )
    if w.dim() == 0 or w.shape[-1] == 0:
        loud(
            f"fake_quant_mx({fmt}): MX blocks run along the LAST dim and this "
            f"tensor is {tuple(w.shape)}. Escape: pass [..., features] with "
            f"features >= 1."
        )
    n = w.shape[-1]
    pad = (-n) % BLOCK
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    return fns[fmt](w)[..., :n]


def bytes_per_elements_mxfp4(n: int) -> int:
    """Packed MXFP4 bytes for `n` elements INCLUDING the e8m0 scale bytes.

    32 elements ⇒ 16 B of codes (2 per byte) + 1 B of scale = **17 B/32**,
    i.e. 0.53125 B/element against fp32's 4 B (7.53x). A partial trailing block
    is counted whole — MX has no partial block, so the caller pads first.
    """
    assert n >= 0, f"element count must be ≥ 0, got {n}"
    return -(-n // BLOCK) * (BLOCK // 2 + 1)


def bytes_per_elements_mxfp8(n: int) -> int:
    """Packed MXFP8 (E4M3) bytes for `n` elements INCLUDING the scale bytes.

    32 elements ⇒ 32 B (1 value/byte) + 1 B of scale = **33 B/32**, i.e.
    1.03125 B/element against fp32's 4 B (3.88x); same padding rule as MXFP4.
    """
    assert n >= 0, f"element count must be ≥ 0, got {n}"
    return -(-n // BLOCK) * (BLOCK + 1)
