r"""NVFP4 — e2m1 elements, e4m3 block scales (one per 16), packed, `_scaled_mm`.

THE FORMAT, and what is NOT in it. Four bits per element on the e2m1 grid
(`tsct._NVFP4_MAGNITUDES`, the codebook `act_quant._e2m1_round` rounds onto),
one scale per **16** logical elements along the contraction, the scale stored as
e4m3 (`mxfp.E4M3_MAX` = 448). Two nibbles per byte, element `2i` in the LOW
nibble (torch's `float4_e2m1fn_x2` packing; pinned by the kernel-vs-emulated
gate in `tests/test_nvfp4.py`). This is NVFP4, not MXFP4: `mxfp.py` owns the
MX formats (e8m0 scale per 32) and the fp4 `_scaled_mm` kernel REFUSES them, so
nothing here is shared with `mxfp.fake_quant_mx` beyond the element grid.

THE SECOND-LEVEL SCALE IS NOT OPTIONAL. A block scale is `amax/6`; that
overflows e4m3 once `amax > 2688`, and the cast SATURATES silently (ADR-0011).
NVFP4's answer is one fp32 per-TENSOR scale `g = max(1, max_block_scale/448)`:
the e4m3 scales are `s/g`, the codes are computed against the ROUNDED scale
(so codes and dequantization cannot disagree), and the GEMM output is
multiplied by `g_a·g_b` on the way out. `g` is a device scalar — the helper
never reads it back to the host, so a call is sync-free (ADR-0018 rule 2).

THE GEMM: `torch._scaled_mm` on `float4_e2m1fn_x2` operands with e4m3 scale
 tensors `[M, K/16]` / `[N, K/16]`. Torch has no `.to(fp4)` cast, which is why
the packing lives here.

THE SHAPE RULES ARE MEASURED, not assumed (torch 2.13.0+cu129, sm_120, 20
shapes bisected): `K % 32 == 0` (the packed operand's trailing BYTE dim must
tile 16) and `N % 16 == 0`; `M` is FREE. Outside them the op raises, so the
emulated lane runs COUNTED with the reason named. The scale is NOT a
`[M, K/16]` matrix even though the op accepts one by ELEMENT COUNT — it reads
the 1x16 recipe's blocked/swizzled buffer (`_blocked_scale`), and passing the
plain matrix measured a 46% relative error on a silently "successful" call.

THE EMULATED PATH IS COUNTED, and the counters NAME which path ran —
`nvfp4_gemm` (kernel) vs `nvfp4_emulated` (dequantize → fp32 GEMM → `dt`), with
the reason on `counters.why` and the process-local tally in
`nvfp4_fallbacks()`: the `mm_fp8` pattern (`tsct.mm_fp8`). The emulated path
dequantizes the SAME packed bytes the kernel would read, so the two are
comparable by construction — `tests/test_nvfp4.py` compares them on one packing.

Autograd is the reason a TRAINING forward always emulates: torch 2.13 registers
no derivative for `_scaled_mm` (`tsct._fp8_reason` measured it). The training
arm's 4-bit activations are `act_quant.fake_quant_act(x, "e2m1", 16)` — the
same grid and block width, fake-quantized with the STE — and this module is the
kernel the bench measures that arm against, not a second training path.
"""

from __future__ import annotations

import torch

from ..instrument import Counters
from .act_quant import _e2m1_round
from .mxfp import E4M3_MAX
from .tsct import _NVFP4_MAGNITUDES

#: Logical elements per scale — NVFP4's own layout. `mxfp.py`'s MXFP4 uses 32
#: with an e8m0 scale and the fp4 kernel refuses it; this is not a knob.
GROUP = 16

#: The element grid (8 magnitudes) and its top level, from the ONE codebook.
_GRID = torch.tensor(_NVFP4_MAGNITUDES, dtype=torch.float32)
E2M1_MAX = float(_NVFP4_MAGNITUDES[-1])

_SCALE_DTYPE = torch.float8_e4m3fn

#: The scale guard: an all-zero block quantizes to zeros, not NaN.
_EPS = 1e-12

#: Process-local emulated-call tally, by reason (`mm_fp8_fallbacks`' twin):
#: a call with no `Counters` is still not silent (ADR-0019).
_NVFP4_FALLBACKS: dict[str, int] = {}


def nvfp4_fallbacks() -> dict[str, int]:
    """A copy of the emulated-call tally, keyed by reason."""
    return dict(_NVFP4_FALLBACKS)


def quantize_nvfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack `x [..., K]` (fp32/bf16) → `(packed uint8 [..., K/2], e4m3 scales [..., K/16], g)`.

    `g` is the fp32 per-tensor second-level scale (1.0 in every ordinary case),
    a 0-dim DEVICE tensor: the caller folds it into the output and nothing
    reads it back. The codes are computed against the e4m3-ROUNDED block scale,
    so `dequantize_nvfp4` reproduces exactly the values the kernel multiplies.

    `K` must be a multiple of 16 — that is the format's scale width, not a
    convenience: a partial block would need a different (padded) scale rule and
    every number would then describe a different format.
    """
    assert x.is_floating_point(), f"NVFP4 packs float tensors, got {x.dtype}"
    k = x.shape[-1]
    assert k % GROUP == 0, (
        f"NVFP4 needs K % {GROUP} == 0 (one e4m3 scale per {GROUP} elements), "
        f"got K={k}. Escape: a width that tiles {GROUP}, or the bf16 lane."
    )
    xf = x.float()
    blocks = xf.reshape(*xf.shape[:-1], k // GROUP, GROUP)
    scale = (blocks.abs().amax(dim=-1) / E2M1_MAX).clamp_min(_EPS)
    # The second-level scale: one fp32 per tensor, DEVICE-side (no `.item()`).
    g = (scale.amax() / E4M3_MAX).clamp_min(1.0)
    scales = (scale / g).to(_SCALE_DTYPE)
    norm = blocks / (scales.to(torch.float32) * g).unsqueeze(-1)
    q = _e2m1_round(norm)  # signed, exactly on the grid
    codes = torch.searchsorted(_GRID.to(x.device), q.abs().reshape(-1)).reshape(q.shape)
    codes = codes.to(torch.uint8) | ((q < 0).to(torch.uint8) * 8)  # sign bit = nibble 8
    codes = codes.reshape(*x.shape)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
    return packed, scales, g


def dequantize_nvfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    *,
    global_scale: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Unpack `[..., K/2]` nibbles + `[..., K/16]` scales → the dequantized values.

    `global_scale` is [`quantize_nvfp4`]'s `g`; omit it for a 1.0 scale. The
    arithmetic is fp32 (the grid values are exact there) and the result is cast
    to `dtype` — the caller's, never a format the caller did not ask for.
    """
    n = packed.shape[-1]
    low = (packed & 0x0F).to(torch.long)
    high = (packed >> 4).to(torch.long)
    codes = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], 2 * n)
    # Bit 3 of a nibble is the SIGN (`e2m1fn` is sign + 2 exp + 1 mantissa); the
    # magnitude is the codebook index, so 8..15 must be masked before the gather.
    mag = (codes & 0x7).reshape(*packed.shape[:-1], (2 * n) // GROUP, GROUP)
    vals = _GRID.to(packed.device)[mag]
    vals = torch.where(codes.reshape(mag.shape) >= 8, -vals, vals)
    vals = (vals * scales.float().unsqueeze(-1)).reshape(*packed.shape[:-1], 2 * n)
    if global_scale is not None:
        vals = vals * global_scale
    return vals.to(dtype)


def _reason(x: torch.Tensor, weight: torch.Tensor, dt: torch.dtype) -> tuple[str, str] | None:
    """`None` if the fp4 kernel can run THIS call, else `(reason, detail)`.

    Order is the operands' own properties first (dtype, rank, the 16-tiling the
    scale layout needs, the contraction), the environment after (device,
    autograd): a CPU call is told the shape reason rather than "not CUDA".
    """
    if not hasattr(torch, "float4_e2m1fn_x2"):
        return ("this torch has no `float4_e2m1fn_x2` dtype", torch.__version__)
    if not hasattr(torch, "_scaled_mm"):
        return ("this torch has no `torch._scaled_mm`", torch.__version__)
    if dt not in (torch.bfloat16, torch.float16):
        return ("the compute dtype is fp32 (the lane's out_dtype is bf16/fp16)", str(dt))
    if x.dim() != 2 or weight.dim() != 2:
        return (
            "non-2-D operand (`_scaled_mm` is the 2-D GEMM)",
            f"x{tuple(x.shape)} W{tuple(weight.shape)}",
        )
    if weight.shape[-1] != x.shape[-1]:
        return (
            "the operands do not contract",
            f"x{tuple(x.shape)} @ W{tuple(weight.shape)}.T",
        )
    if x.shape[-1] % GROUP:
        return (
            f"K is not a multiple of {GROUP} (NVFP4's scale is one per {GROUP})",
            f"K={x.shape[-1]}",
        )
    # MEASURED tile rules of this torch's fp4 `_scaled_mm` (see the module
    # docstring): outside them the op raises, so name the reason here and let
    # the counter carry it rather than a generic "refused the call".
    k, n = x.shape[-1], weight.shape[0]
    if k % 32:
        return (
            "K % 32 != 0 (the packed operand's trailing byte dim must tile 16)",
            f"K={k}",
        )
    if n % 16:
        return ("N % 16 != 0 (the fp4 `mat2` row rule)", f"N={n}")
    if not x.is_cuda or not weight.is_cuda:
        return ("not CUDA (the lane is the fp4 `_scaled_mm` kernel path)", str(x.device))
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        return (
            "autograd is on (torch's `_scaled_mm` has no backward)",
            "the lane is forward-only",
        )
    return None


def _emulate(reason: str, detail: str, counters: Counters | None) -> None:
    """The COUNTED mark of the emulated path: the tally always, the log if given."""
    _NVFP4_FALLBACKS[reason] = _NVFP4_FALLBACKS.get(reason, 0) + 1
    if counters is not None:
        counters.count("nvfp4_emulated")
        note = f"nvfp4: {reason} [{detail}]"
        prev = getattr(counters, "why", None)
        counters.why = f"{prev}; {note}" if prev else note  # type: ignore[attr-defined]


def _blocked_scale(scale: torch.Tensor) -> torch.Tensor:
    """`[mn, K/16]` e4m3 → the flat BLOCKED buffer the fp4 `_scaled_mm` reads.

    The 1x16 blockwise recipe does not take a `[mn, K/16]` matrix: it takes one
    512-BYTE tile per 128 rows × 4 scale columns, with the 128 row index split
    (4 outer, 32 inner) and swapped — the swizzle in torch's own
    `_vendor/quack/blockscaled_gemm_utils.py:239`
    (`pack_scale_2d_to_blocked_contig`), which `scale_blocked_for_cublas` (:284)
    says is "the 1D swizzled layout `torch._scaled_mm` expects". Rows pad to a
    128 multiple and columns to 4 with ZEROS. Verified end to end by the
    kernel-vs-emulated gate: the unswizzled matrix is accepted (the op checks
    the ELEMENT COUNT) and silently reads the wrong scale, 46% relative error.
    """
    mn, sf_k = scale.shape
    rm, rk = -(-mn // 128), -(-sf_k // 4)
    u8 = scale.detach().contiguous().view(torch.uint8)
    padded = torch.zeros(rm * 128, rk * 4, dtype=torch.uint8, device=scale.device)
    padded[:mn, :sf_k] = u8
    blocks = padded.view(rm, 128, rk, 4).permute(0, 2, 1, 3)  # (rm, rk, 128, 4)
    # 128 rows = (4 outer, 32 inner); the tile swaps them, so one 512-byte block
    # reads `(m % 32, m // 32, k % 4)` — that IS the swizzle.
    blocks = blocks.reshape(rm, rk, 4, 32, 4).transpose(2, 3).contiguous()  # (rm,rk,32,4,4)
    return blocks.view(rm, rk, 512).reshape(-1).view(_SCALE_DTYPE)


def _dequant_mm(
    a: torch.Tensor,
    sa: torch.Tensor,
    ga: torch.Tensor,
    b: torch.Tensor,
    sb: torch.Tensor,
    gb: torch.Tensor,
    dt: torch.dtype,
) -> torch.Tensor:
    """`dequant(a) @ dequant(b)ᵀ` in fp32, cast to `dt` — the emulated lane."""
    xd = dequantize_nvfp4(a, sa, torch.float32, global_scale=ga)
    wd = dequantize_nvfp4(b, sb, torch.float32, global_scale=gb)
    return (xd @ wd.t()).to(dt)


def emulated_mm(x: torch.Tensor, weight: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """The emulated lane alone: pack both operands, dequantize, bf16-class GEMM."""
    a, sa, ga = quantize_nvfp4(x)
    b, sb, gb = quantize_nvfp4(weight)
    return _dequant_mm(a, sa, ga, b, sb, gb, dt)


def nvfp4_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    dt: torch.dtype,
    counters: Counters | None = None,
) -> torch.Tensor:
    """`x @ Wᵀ` on NVFP4: the fp4 kernel where it runs, the emulated lane otherwise.

    Both operands are packed from their live values (the masters are not read
    back, so this is a COMPUTE rung). The return dtype is `dt` in both lanes,
    and the path that ran is on the counters: `nvfp4_gemm` or `nvfp4_emulated`
    (+ the reason on `counters.why`). An UNLISTED kernel refusal for a call the
    preconditions accepted is counted and emulated, not fatal — a failed
    accelerated arm degrades, it does not kill the run (`moe_triton_fallback`'s
    rule). The PACKING above it is plain torch and outside that handler: a
    failure there is a bug in this module and stays LOUD.
    """
    a, sa, ga = quantize_nvfp4(x)
    b, sb, gb = quantize_nvfp4(weight)
    why = _reason(x, weight, dt)
    if why is not None:
        _emulate(why[0], why[1], counters)
        return _dequant_mm(a, sa, ga, b, sb, gb, dt)
    try:
        out = torch._scaled_mm(
            a.view(torch.float4_e2m1fn_x2),
            b.view(torch.float4_e2m1fn_x2).t(),
            scale_a=_blocked_scale(sa),
            scale_b=_blocked_scale(sb),
            out_dtype=dt,
        )
    except RuntimeError as e:
        _emulate("`_scaled_mm` refused the call", str(e)[:160], counters)
        return _dequant_mm(a, sa, ga, b, sb, gb, dt)
    if counters is not None:
        counters.count("nvfp4_gemm")
    return out * (ga * gb).to(dt)


def bytes_per_elements(n: int) -> float:
    """Bytes per logical element for the WHOLE operand set: 4-bit data + e4m3 scale.

    `n/2` packed bytes + `n/16` scale bytes over `n` elements = 0.5625 B/element
    (2.84x under bf16). The per-tensor `g` is 4 B and vanishes.
    """
    assert n >= GROUP, f"an NVFP4 operand is at least one {GROUP}-wide block, got {n}"
    return (n / 2 + n / GROUP) / n
