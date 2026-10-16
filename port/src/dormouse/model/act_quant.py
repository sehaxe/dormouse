r"""BitNet a4.8 ACTIVATION quantization — the FFN side of the quant track.

RECOVERY, not new design: the burn tree had `crates/dormouse-core/src/act_quant.rs`
and the torch port suppressed it as an `_unsupported` config stub. Every
formula below is transcribed from that file, named line by line.

| what | burn source | paper |
|---|---|---|
| e2m1 grid, tie rule | `act_quant.rs:150-170` (`fp4_round`) | OCP MX FP4 spec; `2504.12285` §2.2 |
| `max_value` (block max → FORMAT max) | `act_quant.rs:54-59` | — (the fix for the 3-level bug) |
| `attn()` — attention never below 8 bits | `act_quant.rs:38-46` | a4.8 `2504.12285` |
| per-token / per-group scale, STE | `act_quant.rs:70-104` | a4.8 group scales |

THE FORMULA (one line, both formats):

    g      = d if group == 0 else group                  # block width
    s_b    = max|x| over the block / max_value(fmt)      # NOT /1 — see below
    q      = round_nearest(x / s_b)                      # int: clamp to ±(2^(b-1)-1)
    x_q    = q · s_b                                     # dequantized, in the format
    STE    : x + (x_q − x).detach()                       # forward quantized, backward identity

`max_value` is the format's OWN max — fp4 → 6.0, int8 → 127, int4 → 7 — so a
block's max lands ON the top level. **Normalizing a block onto 1 is the bug this
module exists to not repeat** (`act_quant.rs:9-19`): with `scale = block_max` an
e2m1 label only ever emitted `{0, 0.5, 1}` plus the exact grid max — a 3-level
quantizer wearing a 4-bit label, and the reason `tests/test_act_quant.py` pins
that all eight magnitudes are reachable (AGENTS §3.7 act-quant).

THE e2m1 GRID IS NOT DECLARED HERE. `_NVFP4_MAGNITUDES` (`tsct.py:216`) is the
same 8-magnitude ITU grid `mxfp.py:99` imports for MXFP4; one grid, one place to
fix, because a second copy is how the 3-level defect comes back. Rounding is
round-half-to-EVEN CODE, the format's own rule (IEEE 754; the hardware MX-FP4
conversion follows it) — NOT ties-up, which disagrees with the reference at 4 of
the 7 interior midpoints (the table is in `_e2m1_round`, transcribed from
`act_quant.rs:150-170`).

THE ATTENTION RULE IS UNCONDITIONAL (`act_quant.rs:38-46`): `attn_format` maps
fp4 → int8, int4 → int8, int8 → int8. `fp4`/`int4` therefore never run 4-bit
attention, which is the a4.8 recipe and was the CORRECTION that invalidated every
pre-2026-09-27 `--act-quant fp4` reading.

MEASURED, this box, torch 2.13 CPU, fp32, seed 0 — the module's OWN reading (no
external oracle exists for these bounds, so they are measurements, not a
certificate; the analytic bound asserted by the gold test is
`|x_q − x| ≤ block_max / 2·L`):

| what on `randn [16, 128]` | per-token | group 32 |
|---|---|---|
| int8 rms err (rms err / rms x) | 6.98e-3 (6.84e-3) | 5.50e-3 |
| int4 rms err | 1.267e-1 (1.243e-1) | 9.81e-2 |
| fp4 rms err | 1.123e-1 (1.102e-1) | 1.027e-1 |
| int8 max abs err vs `max|x|/254` | 1.60e-2 vs 1.61e-2 | — |
| int4 max abs err vs `max|x|/14` | 2.917e-1 vs 2.930e-1 | — |
| fp4 max abs err vs `2·max|x|/6` | 6.68e-1 vs 6.84e-1 | — |

WHAT IS **NOT** HERE (each has a home, none is wired — YAGNI until a lane needs
it):

* `ternary.activation_quant_8bit` — the same object for `int8` + `group=0`: its
  scale is `block_max` and it multiplies by 127 explicitly, which is this
  module's `block_max / 127` (the values agree to within the order of the two
  fp32 roundings: measured max 2.38e-7 abs / 1.6e-7 rel — 2 ULP; bit-identical on
  a degenerate spike row). This module keeps ONE implementation because the
  group/format generalisation has to exist anyway, and two copies of one
  quantizer drifting apart is the defect that shape produced on the burn side
  (`tsct.factor_qat_project`'s docstring); the equivalence to
  `activation_quant_8bit` is pinned by test so a drift fails loudly.

THE THREE IMPORTED FORMATS (they are rungs HERE since 2026-10-16, and each
keeps its own estimator — this module does not re-implement any of them):

* `mxfp8` → `mxfp.fake_quant_mx(x, "mxfp8")`: the OCP MX codec's per-32 e8m0
  power-of-two block scales and the e4m3 element grid (spec §5.2 Table 1). It
  carries its own block grid, so `act_group` is refused (LOUD) rather than
  silently ignored, and a last dim that is not a multiple of 32 is PADDED
  per row, quantized and sliced back — one padding rule for both MX consumers
  (`mxfp.fake_quant_mx`'s docstring carries the exactness argument).
* `v2_int4` / `v2_int8` → `hadamard.v2_quantize(x, 4|8)`: BitNet v2's
  Hadamard rotation → per-token int quant → rotate back (arXiv:2504.18415,
  `burn-bitnet/src/quant/activations.rs`). The FWHT needs a **2^k** last dim and
  that is a LOUD refusal, NOT a pad — zero-padding to 2^k changes the rotation
  basis (`fwt.rs:105-112`), where padding an MX block only changes which tail
  columns share a scale.

Both of those modules return an **STE'd** tensor (`x + (dq − x).detach()` is
their own last line), so [`fake_quant_act`] hands their result straight through
instead of wrapping it a second time: the value is identical, and one estimator
in one place is what makes a divergence visible.

Verified by `tests/test_act_quant.py` (CPU, fp32); no external oracle exists for
the exact error bounds, so the bound asserted is the format's own analytic one
(`|x_q − x| ≤ block_max / 2·L`) plus a measured RMS.
"""

from __future__ import annotations

import torch

from ..instrument import loud
from .hadamard import v2_quantize
from .mxfp import E4M3_MAX, fake_quant_mx
from .tsct import _NVFP4_MAGNITUDES as _E2M1

#: The activation formats: `none` (off), the a4.8 symmetric integers and e2m1
#: fp4, then the two families that bring their own scale rule — the OCP MX
#: codec (`mxfp8`, per-32 e8m0 blocks) and BitNet v2's rotated integer arms
#: (`v2_int4`/`v2_int8`, per-token absmax after the FWHT).
ACT_FORMATS = ("none", "int8", "int4", "fp4", "mxfp8", "v2_int4", "v2_int8")

#: The formats whose scale is NOT `block_max / max_value(...)`: they carry
#: their own grid (`mxfp8`: 32-wide e8m0 blocks, `mxfp.fake_quant_mx`) or their
#: own rotation + per-token absmax (`v2_*`, `hadamard.v2_quantize`). `act_group`
#: has nothing to group on them, so it is refused rather than silently ignored,
#: and `fake_quant_act` does not re-wrap their (already STE'd) result.
_OWN_GRID_FORMATS = ("mxfp8", "v2_int4", "v2_int8")

#: The scale guard: an all-zero block must quantize to zeros, not to NaN.
_EPS = 1e-8


def max_value(fmt: str) -> float:
    """The format's largest magnitude, and therefore what a block max maps ONTO.

    fp4 → 6.0 (`_E2M1`'s top), int8 → 127, int4 → 7 — `2^(b-1) − 1`
    (`act_quant.rs:54-59`); the imported families keep their own maxima (e4m3 →
    448, the MX spec's element max; `v2_*` → 7/127, the a4.8 levels after the
    rotation). `none` is the identity format: it has no max and is refused here
    rather than answered with a number that would silently scale.

    The three `_OWN_GRID_FORMATS` never CALL this — their scale is the codec's
    (`ceil(log2(amax/448))`) or the rotated token absmax — but their maxima are
    still what their block/token max is scaled against (`v2_*`: exactly, the
    token max lands on 7/127; `mxfp8`: the ceiling keeps it inside [224, 448]),
    so the table stays complete and a reader asking "how big can this format
    get" gets one answer.
    """
    if fmt == "fp4":
        return _E2M1[-1]
    if fmt == "int8":
        return 127.0
    if fmt == "int4":
        return 7.0
    if fmt == "mxfp8":
        return E4M3_MAX
    if fmt == "v2_int8":
        return 127.0
    if fmt == "v2_int4":
        return 7.0
    loud(
        f"act_quant max_value({fmt!r}): no format max — known: "
        f"{[f for f in ACT_FORMATS if f != 'none']} (fp4 = e2m1's 6.0, int8 = 127, "
        f"int4 = 7, mxfp8 = e4m3's {E4M3_MAX}, v2_int8 = 127, v2_int4 = 7). "
        f"Escape: pass a real format, or ask `none` for nothing."
    )


def attn_format(fmt: str) -> str:
    """The a4.8 attention upgrade: the attention path is never below 8 bits.

    `act_quant.rs:38-46`: `Fp4 → Int(8)`, `Int(b) → Int(b.max(8))`. Unconditional
    — `fp4` and `int4` run 8-bit attention, never 4-bit, and a reading that
    claims otherwise measured the FFN path alone. `none` stays `none` (the whole
    arm off).

    The two imported families are OURS, not a line in `act_quant.rs`, and each
    is read the way its own source reads it: `mxfp8` is already 8 bits (e4m3),
    so it stays `mxfp8`; `v2_int4` becomes `v2_int8`, because the rotated 8-bit
    arm IS BitNet v2's published activation format (arXiv:2504.18415) — dropping
    to the unrelated plain `int8` would measure a different arm on the attention
    path than the one the rung names.
    """
    if fmt == "none":
        return "none"
    if fmt in ("int8", "mxfp8", "v2_int8"):
        return fmt
    if fmt in ("fp4", "int4"):
        return "int8"
    if fmt == "v2_int4":
        return "v2_int8"
    loud(
        f"act_quant attn_format({fmt!r}) is not a format: known {list(ACT_FORMATS)}. "
        f"The attention path is upgraded to >= 8 bits for every 4-bit format "
        f"(act_quant.rs:38-46; `v2_int4` upgrades inside its own family). Escape: "
        f"name one of those."
    )


def _check(x: torch.Tensor, fmt: str, group: int) -> None:
    """Float, finite, and a group that tiles the last dim — or LOUD.

    Every message names the cause AND the escape (ADR-0011/ADR-0019). The
    finite test is the load-bearing one: `max|x|` over a block holding one NaN
    is NaN, so the scale would be NaN and EVERY element of the tensor would come
    back NaN with nothing said — a silent corruption with an obvious cause.
    """
    if fmt not in ACT_FORMATS:
        loud(
            f"act_quant: {fmt!r} is not an activation format. Known: "
            f"{list(ACT_FORMATS)} — `none` off, `int8`/`int4` symmetric integer "
            f"levels (127/7, act_quant.rs:54-59), `fp4` e2m1 (8 magnitudes, "
            f"tsct.py:216), `mxfp8` the OCP MX e4m3 codec (per-32 e8m0 blocks, "
            f"mxfp.py), `v2_int4`/`v2_int8` BitNet v2's rotated integer arms "
            f"(hadamard.py). Escape: name one of those, or land the format and its "
            f"gold test (tests/test_act_quant.py) before the config can lie about it."
        )
    if fmt in _OWN_GRID_FORMATS and group != 0:
        loud(
            f"act_quant {fmt} carries its own scale grid (mxfp8: 32-wide e8m0 "
            f"blocks; v2_*: per-token absmax after the FWHT), so act_group={group} "
            f"is INERT here — the quantizer would silently ignore it. Escape: "
            f"act_group=0, or a format whose scale is `block_max / max_value` "
            f"(int8/int4/fp4)."
        )
    if not x.is_floating_point():
        loud(
            f"act_quant {fmt}: activations are float tensors, got {x.dtype} — an "
            f"integer tensor has no block max to scale against. Escape: cast the "
            f"activation to fp32/bf16 before the quantizer."
        )
    if x.dim() == 0 or x.shape[-1] == 0:
        loud(
            f"act_quant {fmt}: the scale reduces the LAST dim and this tensor is "
            f"{tuple(x.shape)}. Escape: pass [..., features] with features >= 1."
        )
    d = x.shape[-1]
    if group < 0:
        loud(
            f"act_quant group={group} is negative: it is a block WIDTH in "
            f"activation columns (0 = one scale per token/row). Escape: 0, or a "
            f"positive divisor of the last dim ({d})."
        )
    if group != 0 and (group > d or d % group != 0):
        loud(
            f"act_quant group={group} does not divide the last dim {d}: the "
            f"quantizer reshapes the activations to [..., {d}//group, group], so a "
            f"non-divisor fails the reshape on the first forward, a long way from "
            f"the flag that set it (act_quant.rs's caller rule). Escape: group=0 "
            f"for one scale per token, or a divisor of {d}."
        )
    if not bool(torch.isfinite(x).all()):
        n_bad = int((~torch.isfinite(x)).sum())
        loud(
            f"act_quant {fmt}: {n_bad} non-finite element(s) (inf/NaN). Cause: a "
            f"block max over a NaN is NaN, so the whole block would quantize to "
            f"NaN with nothing said. Escape: sanitize the activation upstream "
            f"(the NaN firewall is the training step's, ADR-0019)."
        )


def _e2m1_round(x: torch.Tensor) -> torch.Tensor:
    """Nearest e2m1 value, ties to the EVEN CODE — `x` is already in ±6 units.

    Transcribed from `act_quant.rs:150-170`: an ascending ladder of midpoints,
    each overwriting the last, where the code's parity picks the tie rule, so no
    single `>` or `>=` expresses it. The table (`act_quant.rs:158-166`), which the
    gold test pins:

        0.25 → 0.0   0.75 → 1.0   1.25 → 1.0   1.75 → 2.0
        2.5  → 2.0   3.5  → 4.0   5.0  → 4.0

    Below 0.25 nothing fires and the magnitude stays 0 (the only subnormal is
    0.5); at 5.0 and above the top level 6.0 saturates. `0.75` is NOT a level of
    e2m1 and is never produced.
    """
    a = x.abs()
    out = torch.zeros_like(a)
    for i in range(1, len(_E2M1)):
        lo = 0.5 * (_E2M1[i - 1] + _E2M1[i])
        # `i` is the CODE we would claim: even → claim it on a tie (`>=`),
        # odd → leave it to the level below (`>`). Same ladder as the burn source.
        claimed = a >= lo if i % 2 == 0 else a > lo
        out = torch.where(claimed, torch.full_like(a, _E2M1[i]), out)
    return out * x.sign()


def quantize_act(x: torch.Tensor, fmt: str, group: int = 0) -> torch.Tensor:
    """The DEQUANTIZED activation `x_q`, in `x`'s dtype. `none` is the identity.

    `group=0` → one scale per token (a max over the last dim); `group>0` → one
    scale per block of `group` columns (the a4.8 FFN recipe), and the group must
    tile the last dim (LOUD otherwise). The scale is `block_max / max_value(fmt)`,
    so the block max lands on the format's top level — see the module docstring
    for why `block_max` alone is the 3-level bug.

    The three `_OWN_GRID_FORMATS` do not take that path: they hand the tensor to
    the module that owns their estimator — `mxfp.fake_quant_mx(x, "mxfp8")`
    (per-row pad, per-32 e8m0 scales, e4m3 elements) or
    `hadamard.v2_quantize(x, 4|8)` (FWHT rotate → per-token int → rotate back) —
    and `group != 0` is refused for them (their grid is not the caller's to
    choose). Their returned VALUE is the dequantized activation, and because
    those modules end in `x + (dq − x).detach()` their gradient through this
    function is the identity too: same contract, one estimator per family.
    """
    if fmt == "none":
        return x
    _check(x, fmt, group)
    if fmt == "mxfp8":
        return fake_quant_mx(x, "mxfp8")
    if fmt in ("v2_int4", "v2_int8"):
        return v2_quantize(x, 4 if fmt == "v2_int4" else 8)
    xf = x.float()
    d = xf.shape[-1]
    fmax = max_value(fmt)
    if group == 0:
        block_max = xf.abs().amax(dim=-1, keepdim=True)  # [..., 1]
    else:
        # [..., d/g, g] → max over the block → broadcast back inside the block.
        blocks = xf.reshape(*xf.shape[:-1], d // group, group)
        block_max = blocks.abs().amax(dim=-1, keepdim=True)  # [..., d/g, 1]
        block_max = block_max.expand(*block_max.shape[:-1], group).reshape(xf.shape)
    scale = (block_max / fmax).clamp_min(_EPS)
    norm = xf / scale  # in [−fmax, fmax]
    if fmt == "fp4":
        q = _e2m1_round(norm)
    else:
        limit = max_value(fmt)
        q = norm.round().clamp(-limit, limit)
    xq = q * scale
    assert xq.shape == x.shape, "the quantizer must not change the activation shape"
    return xq.to(x.dtype)


def fake_quant_act(x: torch.Tensor, fmt: str, group: int = 0) -> torch.Tensor:
    """`x + (quantize_act(x) − x).detach()` — the STE the forward runs.

    Forward sees the quantized value; backward is the identity, so the fp32
    masters keep a gradient (`round` has none, `act_quant.rs:103-104`).

    The `_OWN_GRID_FORMATS` are handed through WITHOUT this wrap: their module
    already returns its own STE'd tensor (its last line is exactly this
    expression), and wrapping it twice would compute the same value while
    hiding which layer owns the estimator. One STE, one place.
    """
    if fmt == "none":
        return x
    if fmt in _OWN_GRID_FORMATS:
        return quantize_act(x, fmt, group)
    xq = quantize_act(x, fmt, group)
    return x + (xq - x).detach()
