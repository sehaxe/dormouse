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
multiplied by `g_a·g_b` on the way out. That epilogue is the kernel's OWN
second level when the build has the v2 recipe (`_two_level_mm`), and one
`[M, N]` pass otherwise — an `[M, N]` pass is 0.29 ms at the live shape
against the GEMM's 0.20, which is the whole reason the recipe is worth
reaching for. `g` is a device scalar — the helper never reads it back to the
host, so a call is sync-free (ADR-0018 rule 2).

THE GEMM: `torch._scaled_mm` on `float4_e2m1fn_x2` operands with e4m3 scale
 tensors `[M, K/16]` / `[N, K/16]`. Torch has no `.to(fp4)` cast, which is why
the packing lives here.

THE PACKING IS A KERNEL ON CUDA, and it is the reason this lane is usable.
The eager body below measured 13.57 ms at `[24576, 832]` — 42x the GEMM it
feeds — because it is `x.float()` + 7 `where`s + a `searchsorted` + two
strided nibble gathers over 20M elements. `_pack_kernel` does it in one pass
(0.10 ms), `_amax_kernel` one more (0.09), `_swizzle_kernel` replaces an
eager permutation that was launch-bound at 0.10 ms for 1.3 MB, and
`_unpack_kernel` is the dequantize `packed_acts`' BACKWARD calls three times a
layer (0.10 vs 3.72 ms). The eager body is kept as the CPU lane and as the
REFERENCE: `tests/test_nvfp4.py` asserts the CUDA lane is bit-identical to it,
so the format is still defined in exactly one place.

THE WEIGHT IS `[N, K]` (`nn.Linear`), and that is forced, not a preference:
the scale is one per 16 along the LAST axis, so only `[N, K]` can be packed
with the CONTRACTION axis tiled, which is what the GEMM reads after the
`.t()`. A `[K, N]` weight is refused LOUD by `_contraction` — passing one to
the fallback used to raise `mat1 and mat2 shapes cannot be multiplied
(24576x832 and 2048x832)` from inside the emulated lane, i.e. a shape no lane
could mean was being counted as "the kernel could not run it".

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
from .qat_cache import SHARED_PACK_CACHE
from .tsct import _NVFP4_MAGNITUDES

try:  # the fused pack/unpack kernels; the eager code below is the reference
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - a torch-only box keeps the eager lane
    _HAS_TRITON = False

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


if _HAS_TRITON:

    @triton.jit
    def _mag(idx):
        """`_NVFP4_MAGNITUDES[idx]` for idx in 0..7, computed instead of gathered.

        The grid is 0, 0.5, 1, 1.5, 2, 3, 4, 6 — `(1 + ½·mantissa)·2^exponent`
        from idx 2 up (`exponent = (idx-2) // 2`, `mantissa = (idx-2) & 1`), but
        NOT a clean bit ladder: e2m1's own encoding would put 0.75 at idx 1 and
        the codebook skips it, so 0.5 is its own case. Writing it as one formula
        over the whole range measured 2.0 for `6.0` (idx 6) — a quarter of the
        grid wrong, and invisible unless the check is against the EAGER dequant.
        """
        return tl.where(
            idx == 0,
            0.0,
            tl.where(
                idx == 1,
                0.5,
                (1.0 + 0.5 * ((idx - 2) & 1).to(tl.float32))
                * tl.exp2(((idx - 2) >> 1).to(tl.float32)),
            ),
        )

    @triton.jit
    def _swizzle_kernel(S, OUT, MN, SFK, RK, BLOCK: tl.constexpr):
        """One 512-byte tile per (128 rows, 4 scale columns), bytes copied as-is.

        The tile's own order is `(m%32)*16 + (m//32)*4 + j` — `_blocked_scale`'s
        closed form, and the one `tests/test_nvfp4.py` checks arithmetically.
        """
        t = tl.program_id(0)
        k = tl.program_id(1)
        p = tl.arange(0, BLOCK)
        row = t * 128 + ((p // 4) % 4) * 32 + (p // 16) % 32
        col = k * 4 + p % 4
        v = tl.load(S + row * SFK + col, mask=(row < MN) & (col < SFK), other=0)
        tl.store(OUT + (t * RK + k) * BLOCK + p, v)

    @triton.jit
    def _amax_kernel(
        X, OUT, K, NROWS, GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        """`[rows, K]` -> the per-16 block amax `[rows, K/16]`, one pass, exact.

        torch's `view(..., 16).abs().amax(-1)` measures 0.45 ms here against 0.09:
        a 16-wide innermost reduce is the shape TensorIterator is worst at, and
        the `.abs()` temp is a second 20M-element pass.
        """
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        kb = tl.program_id(1)
        BG: tl.constexpr = BLOCK_K // GROUP
        cols = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        rm = rows < NROWS
        x = tl.load(
            X + rows[:, None].to(tl.int64) * K + cols[None, :],
            mask=rm[:, None] & (cols < K)[None, :],
            other=0.0,
        ).to(tl.float32)
        m = tl.max(tl.abs(tl.reshape(x, (BLOCK_M, BG, GROUP))), axis=2)
        ocol = kb * BG + tl.arange(0, BG)
        tl.store(
            OUT + rows[:, None].to(tl.int64) * (K // GROUP) + ocol[None, :],
            m,
            mask=rm[:, None] & (ocol < K // GROUP)[None, :],
        )

    @triton.jit
    def _pack_kernel(
        X,
        S,
        G,
        OUT,
        DEQ,
        K,
        NROWS,
        HAS_DEQ: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """One pass: `x / (s·g)` -> e2m1 code, two nibbles per byte, optional dequant.

        `S` holds the e4m3 block scales as FLOATS already (torch rounded them, so
        the scale is the format's own value, not a second rounding here), and the
        per-tensor `g` is loaded from device memory — never read on the host.
        The code rule is `_e2m1_round`'s midpoint ladder as arithmetic: ties go to
        the EVEN code, so even midpoints use `>=` and odd ones `>`. A byte is
        `code[2i] | code[2i+1] << 4` (element 0 in the LOW nibble).

        The DIVISION is `div_rn`, not `/`: Triton's default fp32 divide is a
        2-ulp approximation, and on an exact midpoint (1.25, 2.5, 5.0 — the ties
        the format's rule is about) one ulp decides a different code. Measured:
        6 wrong codes in 8192 without it, 0 with it.
        """
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        kb = tl.program_id(1)
        BG: tl.constexpr = BLOCK_K // GROUP
        sk = K // GROUP
        cols = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        scol = kb * BG + tl.arange(0, BG)
        rm = rows < NROWS
        x = tl.load(
            X + rows[:, None].to(tl.int64) * K + cols[None, :],
            mask=rm[:, None] & (cols < K)[None, :],
            other=0.0,
        ).to(tl.float32)
        s = tl.load(
            S + rows[:, None].to(tl.int64) * sk + scol[None, :],
            mask=rm[:, None] & (scol < sk)[None, :],
            other=1.0,
        ).to(tl.float32)
        s = tl.reshape(
            tl.broadcast_to((s * tl.load(G))[:, :, None], (BLOCK_M, BG, GROUP)),
            (BLOCK_M, BLOCK_K),
        )
        norm = tl.math.div_rn(x, s)
        a = tl.abs(norm)
        idx = (
            (a > 0.25).to(tl.int32)
            + (a >= 0.75).to(tl.int32)
            + (a > 1.25).to(tl.int32)
            + (a >= 1.75).to(tl.int32)
            + (a > 2.5).to(tl.int32)
            + (a >= 3.5).to(tl.int32)
            + (a > 5.0).to(tl.int32)
        )
        # The sign bit rides bit 3, and a zero magnitude keeps the code 0 — the
        # same rule as `_e2m1_round`'s `q < 0` (-0.0 is not negative there).
        code = idx.to(tl.uint8) + ((norm < 0) & (idx > 0)).to(tl.uint8) * 8
        pair = tl.reshape(code, (BLOCK_M, BLOCK_K // 2, 2)).to(tl.int32)
        hi = tl.where(tl.arange(0, 2)[None, None, :] == 1, 16, 1)
        byte = tl.sum(pair * hi, axis=2)
        pcols = kb * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
        tl.store(
            OUT + rows[:, None].to(tl.int64) * (K // 2) + pcols[None, :],
            byte.to(tl.uint8),
            mask=rm[:, None] & (pcols < K // 2)[None, :],
        )
        if HAS_DEQ:
            val = _mag(idx) * s * tl.where(norm < 0, -1.0, 1.0)
            tl.store(
                DEQ + rows[:, None].to(tl.int64) * K + cols[None, :],
                val.to(DEQ.dtype.element_ty),
                mask=rm[:, None] & (cols < K)[None, :],
            )

    @triton.jit
    def _unpack_kernel(
        P,
        S,
        G,
        OUT,
        K,
        NROWS,
        GROUP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """`dequantize_nvfp4` as ONE pass: nibble -> grid value × block scale × g."""
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        kb = tl.program_id(1)
        BG: tl.constexpr = BLOCK_K // GROUP
        sk = K // GROUP
        cols = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        scol = kb * BG + tl.arange(0, BG)
        rm = rows < NROWS
        byte = tl.load(
            P + rows[:, None].to(tl.int64) * (K // 2) + (cols // 2)[None, :],
            mask=rm[:, None] & (cols < K)[None, :],
            other=0,
        ).to(tl.int32)
        code = tl.where(cols[None, :] % 2 == 0, byte & 0xF, byte >> 4)
        s = tl.load(
            S + rows[:, None].to(tl.int64) * sk + scol[None, :],
            mask=rm[:, None] & (scol < sk)[None, :],
            other=1.0,
        ).to(tl.float32)
        s = tl.reshape(
            tl.broadcast_to((s * tl.load(G))[:, :, None], (BLOCK_M, BG, GROUP)),
            (BLOCK_M, BLOCK_K),
        )
        val = _mag(code & 0x7) * s * tl.where((code & 0x8) != 0, -1.0, 1.0)
        tl.store(
            OUT + rows[:, None].to(tl.int64) * K + cols[None, :],
            val.to(OUT.dtype.element_ty),
            mask=rm[:, None] & (cols < K)[None, :],
        )


#: The tile the three kernels above run at. Measured at `[24576, 832]`: a
#: `BLOCK_M` of 4 is 0.14 ms for pack+dequant against 0.22 at one row per
#: program (the launch count, not the arithmetic, is what was costing).
_BLOCK_M, _BLOCK_K = 4, 128


def _launch(fn, rows: int, k: int, **kwargs: object) -> None:
    """The one place the launch shape lives (`BLOCK_M` rows × `BLOCK_K` columns)."""
    fn[(-(-rows // _BLOCK_M), -(-k // _BLOCK_K))](
        K=k, NROWS=rows, GROUP=GROUP, BLOCK_M=_BLOCK_M, BLOCK_K=_BLOCK_K, **kwargs
    )


def _fused_quantize(
    x: torch.Tensor, deq_dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`quantize_nvfp4`'s CUDA lane: torch rounds the scale, one Triton pass packs.

    The scale stays in torch on purpose — it is 1/32 of the elements, and the
    e4m3 rounding is the one place a kernel-side re-implementation would drift
    from the format for no speed. The pack itself is one pass over `x`, and with
    `deq_dtype` the same pass also writes the DEQUANTIZED values, so a caller
    that needs both does not pay a second kernel (`packed_acts`' forward).
    """
    k = x.shape[-1]
    rows = x.numel() // k
    x2 = x.reshape(rows, k).contiguous()
    amax = torch.empty((rows, k // GROUP), dtype=torch.float32, device=x.device)
    _launch(_amax_kernel, rows, k, X=x2, OUT=amax)
    scale = (amax / E2M1_MAX).clamp_min(_EPS)
    g = (scale.amax() / E4M3_MAX).clamp_min(1.0)
    s8 = (scale / g).to(_SCALE_DTYPE)
    packed = torch.empty((rows, k // 2), dtype=torch.uint8, device=x.device)
    deq = None if deq_dtype is None else torch.empty((rows, k), dtype=deq_dtype, device=x.device)
    _launch(
        _pack_kernel,
        rows,
        k,
        X=x2,
        S=s8,
        G=g,
        OUT=packed,
        DEQ=x2 if deq is None else deq,
        HAS_DEQ=deq is not None,
    )
    packed, scales = packed.view(*x.shape[:-1], k // 2), s8.view(*x.shape[:-1], k // GROUP)
    if deq is None:
        return packed, scales, g
    return packed, scales, g, deq.view(*x.shape[:-1], k)  # type: ignore[return-value]


def quantize_nvfp4(
    x: torch.Tensor, *, deq_dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, ...]:
    """Pack `x [..., K]` (fp32/bf16) → `(packed uint8 [..., K/2], e4m3 scales [..., K/16], g)`.

    `g` is the fp32 per-tensor second-level scale (1.0 in every ordinary case),
    a 0-dim DEVICE tensor: the caller folds it into the output and nothing
    reads it back. The codes are computed against the e4m3-ROUNDED block scale,
    so `dequantize_nvfp4` reproduces exactly the values the kernel multiplies.

    `K` must be a multiple of 16 — that is the format's scale width, not a
    convenience: a partial block would need a different (padded) scale rule and
    every number would then describe a different format.

    `deq_dtype` (CUDA + Triton only) appends the DEQUANTIZED values to the tuple
    — `dequantize_nvfp4`'s result, from the same one pass. The CUDA lane is the
    fused kernel; the CPU/`--no-triton` lane is the eager reference right here,
    which the gates compare against.
    """
    assert x.is_floating_point(), f"NVFP4 packs float tensors, got {x.dtype}"
    k = x.shape[-1]
    assert k % GROUP == 0, (
        f"NVFP4 needs K % {GROUP} == 0 (one e4m3 scale per {GROUP} elements), "
        f"got K={k}. Escape: a width that tiles {GROUP}, or the bf16 lane."
    )
    if _HAS_TRITON and x.is_cuda:
        return _fused_quantize(x, deq_dtype)
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
    if deq_dtype is None:
        return packed, scales, g
    return packed, scales, g, dequantize_nvfp4(packed, scales, deq_dtype, global_scale=g)


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
    if _HAS_TRITON and packed.is_cuda:
        return _fused_dequantize(packed, scales, dtype, global_scale)
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


def _fused_dequantize(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype,
    global_scale: torch.Tensor | float | None,
) -> torch.Tensor:
    """`dequantize_nvfp4`'s CUDA lane: one pass, no int64 code tensor.

    The eager body builds `(low, high)` as int64 (8 B per element), stacks, and
    gathers the codebook — 3.7 ms at `[24576, 832]` against 0.2 ms here, and it
    sits in `packed_acts`' BACKWARD three times per layer.
    """
    n = packed.shape[-1]
    k = 2 * n
    rows = packed.numel() // n
    g = global_scale
    if g is None:
        g = torch.ones((), dtype=torch.float32, device=packed.device)
    elif not torch.is_tensor(g):
        g = torch.tensor(float(g), dtype=torch.float32, device=packed.device)
    out = torch.empty((rows, k), dtype=dtype, device=packed.device)
    _launch(
        _unpack_kernel,
        rows,
        k,
        P=packed.reshape(rows, n).contiguous(),
        S=scales.reshape(rows, k // GROUP).float(),
        G=g.float(),
        OUT=out,
    )
    return out.view(*packed.shape[:-1], k)


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

    One 512-byte byte-shuffle either way; the CUDA lane is a kernel because the
    eager chain measured 0.10 ms for `[24576, 52]` and 0.11 ms for `[832, 128]`
    — both launch-bound, so a plain permutation cost the same as 20x its size.
    """
    mn, sf_k = scale.shape
    rm, rk = -(-mn // 128), -(-sf_k // 4)
    out = torch.empty(rm * rk * 512, dtype=torch.uint8, device=scale.device)
    if _HAS_TRITON and scale.is_cuda:
        _swizzle_kernel[(rm, rk)](
            scale.detach().contiguous().view(torch.uint8), out, mn, sf_k, rk, BLOCK=512
        )
        return out.view(_SCALE_DTYPE)
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
    """`dequant(a) @ dequant(b)ᵀ` in fp32, cast to `dt` — the emulated lane.

    The `.t()` here is NOT redundant with the kernel lane's `.t()`, and it is
    the reason `_contraction` exists: `quantize_nvfp4` packs the LAST axis, and
    the fp4 GEMM needs the CONTRACTION axis packed, so a weight is only
    packable as `[N, K]` and both lanes transpose it to `[K, N]` afterwards.
    A `[K, N]` weight here would be packed along N and then transposed into a
    `[N/2, K]` operand — the fallback would ``succeed'' on a shape the kernel
    refuses and stop being an emulation of anything.
    """
    xd = dequantize_nvfp4(a, sa, torch.float32, global_scale=ga)
    wd = dequantize_nvfp4(b, sb, torch.float32, global_scale=gb)
    return (xd @ wd.t()).to(dt)


def _contraction(x: torch.Tensor, weight: torch.Tensor, who: str) -> None:
    """LOUD on a pair NEITHER lane can mean: `x [M, K]`, weight `[N, K]`.

    `[N, K]` is `nn.Linear`'s layout and the only one this module can pack
    (the scale is one per 16 along the last axis, and the GEMM contracts over
    K). The measured failure this guards: `nvfp4_mm(x[24576, 832],
    w[832, 2048])` — a `[K, N]` weight — passed `_reason` as ``the operands do
    not contract'' (a REASON, which the fallback then treated as runnable) and
    surfaced as `mat1 and mat2 shapes cannot be multiplied (24576x832 and
    2048x832)` from inside the emulated lane. A shape no lane can mean is not
    a fallback reason; it is a caller error, and it is named as one.
    """
    assert weight.dim() == 2 and weight.shape[-1] == x.shape[-1], (
        f"{who}: x{tuple(x.shape)} does not contract with W{tuple(weight.shape)}; "
        "the weight is [N, K] (`nn.Linear`, contraction LAST) — the layout the "
        "packer can pack along K. A [K, N] weight is `W.T` here, and "
        f"`x @ W` is not the GEMM this module computes ({who}(x, W) = x @ W.T)."
    )


def emulated_mm(x: torch.Tensor, weight: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """The emulated lane alone: pack both operands, dequantize, bf16-class GEMM."""
    _contraction(x, weight, "emulated_mm")
    a, sa, ga = quantize_nvfp4(x)
    b, sb, gb = quantize_nvfp4(weight)
    return _dequant_mm(a, sa, ga, b, sb, gb, dt)


def can_use_kernel(x: torch.Tensor, weight: torch.Tensor, dt: torch.dtype) -> bool:
    """True if the fp4 `_scaled_mm` can run THIS pair — a pure shape/dtype/device fact.

    `_reason`'s `None`, named for callers that decide a LANE (not a fallback):
    `packed_acts.packed_act_mm` asks this before it chooses between one pack +
    the fp4 GEMM and one pack + a bf16 GEMM. It reads no values, so the branch
    it drives is static — a CPU call is answered `False` by the device check.
    """
    return _reason(x, weight, dt) is None


def pack_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Everything the kernel needs from a `[N, K]` weight: `(packed, scales, g, blocked)`.

    `blocked` is the 512-byte-tile swizzle of the e4m3 scales — static like the
    weight, so the per-step cache (`qat_cache.QuantCache.pack_nvfp4`) stores it
    with the pack instead of re-permuting 1.3 MB per forward.
    """
    packed, scales, g = quantize_nvfp4(weight)
    return packed, scales, g, _blocked_scale(scales)


#: NVFP4's OWN scale recipe, which `_scaled_mm_v2` takes as two lists: a 1x16
#: e4m3 block scale plus the fp32 per-tensor second level, per operand. The
#: ints are torch 2.13's own enums (`ScalingType.BlockWise1x16` = 2,
#: `TensorWise` = 0, `SwizzleType.SWIZZLE_32_4_4` = 1, `NO_SWIZZLE` = 0); the
#: op's schema takes `int[]`, so the enums are read once and written as ints.
try:  # pragma: no cover - the enums exist on every torch with `_scaled_mm_v2`
    from torch.nn.functional import ScalingType as _Scaling
    from torch.nn.functional import SwizzleType as _Swizzle

    _NV_RECIPE = [int(_Scaling.BlockWise1x16), int(_Scaling.TensorWise)]
    _NV_SWIZZLE = [int(_Swizzle.SWIZZLE_32_4_4), int(_Swizzle.NO_SWIZZLE)]
    _HAS_V2 = hasattr(torch.ops.aten, "_scaled_mm_v2")
except ImportError:  # pragma: no cover
    _NV_RECIPE, _NV_SWIZZLE, _HAS_V2 = [], [], False

#: Does this build's fp4 kernel apply the second level INSIDE the epilogue?
#: `None` until the first call answers it; a process constant, not a per-call
#: decision (the answer is a property of the build). `_EPILOGUE_WHY` is the
#: refusal, for the log line and for a reader of `bench_nvfp4_qat`'s report.
_EPILOGUE: bool | None = None
_EPILOGUE_WHY = ""


def epilogue() -> str:
    """Which epilogue the fp4 lane uses: `"v2"` (in-kernel), `"mul"`, or `"?"`."""
    if _EPILOGUE is None:
        return "?"
    return "v2" if _EPILOGUE else "mul"


def _note(counters: Counters | None, note: str) -> None:
    """Append a reason to `counters.why` — the port's seam channel (`_emulate`'s)."""
    if counters is None:
        return
    prev = getattr(counters, "why", None)
    counters.why = f"{prev}; {note}" if prev else note  # type: ignore[attr-defined]


def _two_level_mm(
    a: torch.Tensor,
    sa: torch.Tensor,
    ga: torch.Tensor,
    b: torch.Tensor,
    blocked_b: torch.Tensor,
    gb: torch.Tensor,
    dt: torch.dtype,
    counters: Counters | None,
) -> torch.Tensor:
    """`(codes·sa·ga) @ (codes·sb·gb)`ᵀ with the second level in the kernel.

    NVFP4's recipe IS two levels, and `_scaled_mm_v2` takes both:
    `scale_a=[blocked 1x16 e4m3, fp32 per-tensor]` with
    `recipe=[BlockWise1x16, TensorWise]`, and the same for B. That is the
    `ga·gb` epilogue FOR FREE — one fused multiply on the accumulator, reading
    the scales the codes were computed against, so nothing is re-rounded.

    THE COST IT REMOVES, measured at `[24576, 832]×[832, 2048]`: a
    `out * (ga·gb)` pass is `[M, N]` at the memory roofline, **0.29 ms against
    the GEMM's 0.20**. Folding the factor into the weight's e4m3 block scales
    instead is neither exact (e4m3 has 3 mantissa bits, `ga·gb` is an arbitrary
    fp32) nor always representable (a block scale is already at the top of the
    format whenever its own second level is > 1) — so the alternative is the
    `[M, N]` pass, not a fold.

    THE FALLBACK IS MEASURED, not defensive. The v1 op also takes a
    `scale_result` argument and **ignores it silently** on this build: a 2.4×
    factor came back 0.58 relative short with no exception, which is why nothing
    here passes one. If v2 refuses this build, one `[M, N]` multiply runs and
    the reason is on `counters.why` and in `_EPILOGUE_WHY`.
    """
    global _EPILOGUE, _EPILOGUE_WHY
    if _EPILOGUE is not False and _HAS_V2:
        try:
            out = torch._scaled_mm_v2(
                a.view(torch.float4_e2m1fn_x2),
                b.view(torch.float4_e2m1fn_x2).t(),
                [_blocked_scale(sa), ga.float()],
                _NV_RECIPE,
                _NV_SWIZZLE,
                [blocked_b, gb.float()],
                _NV_RECIPE,
                _NV_SWIZZLE,
                None,
                dt,
            )
        except RuntimeError as e:
            if _EPILOGUE:  # it worked before: this is a new shape, not the build
                raise
            _EPILOGUE, _EPILOGUE_WHY = False, str(e)[:160]
            _note(counters, f"nvfp4: no two-level epilogue [{_EPILOGUE_WHY}]")
        else:
            _EPILOGUE = True
            if counters is not None:
                counters.count("nvfp4_gemm")
            return out
    out = torch._scaled_mm(
        a.view(torch.float4_e2m1fn_x2),
        b.view(torch.float4_e2m1fn_x2).t(),
        scale_a=_blocked_scale(sa),
        scale_b=blocked_b,
        out_dtype=dt,
    )
    if counters is not None:
        counters.count("nvfp4_gemm")
    return out * (ga * gb).to(dt)


def nvfp4_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    dt: torch.dtype,
    counters: Counters | None = None,
    *,
    cache: object | None = SHARED_PACK_CACHE,
    packed: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """`x @ Wᵀ` on NVFP4: the fp4 kernel where it runs, the emulated lane otherwise.

    `x` is `[M, K]` and `W` is `[N, K]` (`nn.Linear`'s weight, contraction
    last). A `[K, N]` weight is refused LOUD by `_contraction` rather than
    packed along the wrong axis — see that function for the measured failure.

    Both operands are packed from their live values (the masters are not read
    back, so this is a COMPUTE rung). The return dtype is `dt` in both lanes,
    and the path that ran is on the counters: `nvfp4_gemm` or `nvfp4_emulated`
    (+ the reason on `counters.why`). An UNLISTED kernel refusal for a call the
    preconditions accepted is counted and emulated, not fatal — a failed
    accelerated arm degrades, it does not kill the run (`moe_triton_fallback`'s
    rule). The PACKING above it is plain torch and outside that handler: a
    failure there is a bug in this module and stays LOUD.

    `cache` is the per-optimizer-step pack cache (`QuantCache.pack_nvfp4`). A
    weight does not change inside a training step, and its pack (0.30 ms at
    `[2048, 832]`) plus the swizzle of its scales (0.05) are 0.35 of the call —
    the cache is what makes the fp4 GEMM beat the bf16 one at our shape. The
    DEFAULT is the process-wide cache (`qat_cache.SHARED_PACK_CACHE`), because
    the free-function shape has no module to hang one on; `cache=None` opts out, and
    `packed_acts.packed_act_mm` does, because the weight it passes is the
    per-forward `dense_qat` STE tensor: a fresh OBJECT every call, so a cache
    would miss every time and retain a `[N, K]` master per site until the step
    boundary. The invalidation is the trainer's one per-step call
    (`invalidate_qat_cache_all`), not a timer — a stale pack trains on old
    weights silently, which is the worst failure this design has.

    `packed` is a prior `quantize_nvfp4(x)` result: the gated FFN packs ONE
    activation for `gate` and `up`, and this is how the second product reads it
    instead of repacking 20.4 M elements.
    """
    _contraction(x, weight, "nvfp4_mm")
    why = _reason(x, weight, dt)
    if why is not None:
        a, sa, ga = quantize_nvfp4(x) if packed is None else packed
        b, sb, gb = quantize_nvfp4(weight)
        _emulate(why[0], why[1], counters)
        return _dequant_mm(a, sa, ga, b, sb, gb, dt)
    b, sb, gb, blocked_b = (
        cache.pack_nvfp4(weight) if cache is not None else pack_weight(weight)
    )
    a, sa, ga = quantize_nvfp4(x) if packed is None else packed
    try:
        return _two_level_mm(a, sa, ga, b, blocked_b, gb, dt, counters)
    except RuntimeError as e:
        _emulate("`_scaled_mm` refused the call", str(e)[:160], counters)
        return _dequant_mm(a, sa, ga, b, sb, gb, dt)


def bytes_per_elements(n: int) -> float:
    """Bytes per logical element for the WHOLE operand set: 4-bit data + e4m3 scale.

    `n/2` packed bytes + `n/16` scale bytes over `n` elements = 0.5625 B/element
    (2.84x under bf16). The per-tensor `g` is 4 B and vanishes.
    """
    assert n >= GROUP, f"an NVFP4 operand is at least one {GROUP}-wide block, got {n}"
    return (n / 2 + n / GROUP) / n
