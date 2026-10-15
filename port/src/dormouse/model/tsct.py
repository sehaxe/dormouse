"""TSCT: `LinearLike` reinterpreted in torch — `W = U diag(s) Vᵀ`.

`burn-spectral`'s `SpectralLinear` keeps the factors `U [in, r]`, `s [r]`,
`V [r, out]` and computes the effective weight on the fly, so a `d → f`
projection costs `r·(d + f + 1)` parameters instead of `d·f`. That is what
makes a 5461-wide FFN affordable, and it is what the retraction and the Muon
group operate on: the polar retraction is done in the FACTORED form, so fp32
Newton-Schulz over `[64, f]` replaces "~40 s/step" over `[d, d]` (AGENTS §2.3).

Precision (`docs/architecture/bf16-plan.md`): the factors are fp32 MASTERS;
the forward casts to bf16, matmuls, and returns fp32 — softmax/loss/KDA state
stay fp32 at the arm boundary. `quant = "none"` is the day-one format and it
resolves to bf16 factors; the STORAGE rungs (`fp8`, `nvfp4`) exist since
2026-10-15 and dequantize inside the forward (see below); every other rung
(LittleBit semantics, act quant) is still refused at CONFIG load (see
`config._refuse_ladder_rung`), so a rung can never silently compute bf16
while the config reads a name it did not implement.

STORAGE rungs vs bf16, the exact mechanics (validated by test):
* master fp32 factors are quantized elementwise to the storage format with
  a per-tensor scale computed from the format's own max (E4M3 = 448,
  e2m1/grid-max = 6) **at state dict save/load time** — the rung is a
  checkpoint-compression device and a memory format for the bank; the fp32
  master floats stay the optimizer's object inside a live run.
* the forward path never changes numerics within a run: the rung pins a
  round trip `fp32 → storage → fp32 ≥` the test tolerance (1e-2 relative),
  which is where the 8-bit/4-bit losses are allowed to live and no further.
* LittleBit is NOT the storage rung semantics above (it is a QAT ladder
  step: sign(U)s·sign(V)ᵀ + learned scales, G-A..G-D gates in
  `docs/reviews/littlebit-design-2026-10-15.md` — a training-time change,
  not a checkpoint format).

Deviations from burn, both deliberate and listed in the port report:

* **no `N % 4` output padding.** burn pads `out_features` to a multiple of 4
  and slices back because cubek's matmul vectorizes on `N % 4 == 0`
  (AGENTS §2.2). torch has no such constraint, so the port drops the pad — the
  parameter count is the honest `r·(in + out + 1)` either way.
* **no STE quantizer in the forward.** burn's factor quant is a
  straight-through estimator (`act_quant.rs`); the day-one port has no rung to
  run, so the only format is bf16 and no fake-quant path exists to be wrong.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..instrument import loud


def mm_mixed(x: torch.Tensor, weight: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """`x @ Wᵀ` in `dt` with fp32-out — the port's mixed-dtype rule (F.linear).

    torch's `Linear` dispatched on the INPUT's dtype: fp32 masters with fp32
    activation inputs run fp32 GEMM, which on sm_120 means the MAGMA lane —
    measured 46.6% of small's forward CUDA time in the 2026-10-15 profiler
    probe (280 × ~240 µs fp32 gemms against TSCT's 13-38 µs bf16 cutlass).
    Casting per forward (masters stay fp32, exactly TSCTLinear's contract)
    routes every projection through the bf16 cuBLAS tensor-core path.
    """
    w = weight if weight.dtype == dt else weight.to(dt)
    return (x if x.dtype == dt else x.to(dt)) @ w.t()


class TSCTLinear(nn.Module):
    """A factorized linear: `x → ((x·U) ⊙ s)·V`, masters fp32, forward in `dtype`."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        bias: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if rank > min(in_features, out_features):
            loud(
                f"TSCT rank {rank} exceeds min(in={in_features}, out={out_features}): "
                f"a rank above either dimension is not a factorization (burn clamps "
                f"only the lm_head, explicitly, in `model.rs`). Escape: lower `rank` "
                f"or make the projection dense (`use_tsct = false`)."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.compute_dtype = dtype
        # U ~ N(0, 1/sqrt(in)), V ~ N(0, 1/sqrt(rank)): the composed map has
        # unit-ish output scale at any width, which matters because ReZero's
        # residual_scale is initialised to 1.0 (not 0 — at 0 the block body
        # gets no gradient at all, the 2026-09-27 burn bug).
        self.u = nn.Parameter(
            torch.randn(in_features, rank, device=device) / math.sqrt(in_features)
        )
        self.s = nn.Parameter(torch.ones(rank, device=device))
        self.v = nn.Parameter(torch.randn(rank, out_features, device=device) / math.sqrt(rank))
        self.bias = nn.Parameter(torch.zeros(out_features, device=device)) if bias else None

    def effective_weight(self) -> torch.Tensor:
        """`U diag(s) Vᵀ` in fp32 — the weight the optimizer and the report
        talk about. Never used in a forward (it would materialize `d·f`)."""
        return (self.u * self.s) @ self.v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = self.compute_dtype
        if dt == torch.float32:
            y = ((x @ self.u) * self.s) @ self.v
        else:
            # Matmuls in the factor dtype with fp32 accumulate (torch's own
            # reduction), the master params untouched.
            xu = x.to(dt)
            y = ((xu @ self.u.to(dt)) * self.s.to(dt)) @ self.v.to(dt)
            y = y.to(torch.float32)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"dtype={self.compute_dtype}"
        )


# ---------------------------------------------------------------------------
# STORAGE rungs: checkpoint/bank compression of the fp32 factor masters
# ---------------------------------------------------------------------------

#: The format's own signed max, by rung: E4M3 fn grid top = 448; NVFP4's e2m1
#: grid top = 6 (0, .5, 1, 1.5, 2, 3, 4, 6 — LittleBit's grid, which is why the
#: sign rung maps onto it exactly).
RUNG_MAX = {"fp8": 448.0, "nvfp4": 6.0}

#: The matrix-parameter names that ARE factor matrices (everything else —
#: scales `.s`, router, engram, norms — stays fp32/goes bf16 untouched).
_FACTOR_SUFFIXES = ("u", "v", "gate_u", "up_u")

#: NVFP4 packing constants: ITU e2m1, magnitude grid ascending then two's-compl
#: sign per code; two codes per byte, low nibble first.
_NVFP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e2m1_quant(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Per-16-block NVFP4 quantize: codes u8 + e4m3 scales [n/16] + a global
    fp32 correction.

    NVIDIA's NVFP4 is TWO-LEVEL for exactly the underflow this would hit:
    a block amax smaller than e4m3's min normal rounds the block scale to
    zero and the block is lost entirely. So the tensor is first prescaled by
    a global power of two that pushes block scales into e4m3's normal range,
    and the global comes back on decode.

    Encode = quotient by the block scale then nearest grid magnitude (the
    grid is LittleBit's, so a later sign rung re-uses the same codes).
    """
    n = t.numel()
    assert n % 16 == 0, f"nvfp4 quantizes blocks of 16, got numel={n}"
    # Global prescale: aim the SMALLEST nonzero block scale at e4m3's min
    # normal (the largest block scale then lands wherever it lands — e4m3
    # covers [tiny, 448] = ~16 decades, ample). A power of two moves nothing.
    float(torch.finfo(torch.float8_e4m3fn).tiny)  # smallest positive
    min_normal = float(2.0 ** torch.finfo(torch.float8_e4m3fn).min)  # 2^-6
    amaxs = t.float().abs().reshape(-1, 16).amax(dim=1)
    amaxs = amaxs[amaxs > 0]
    if amaxs.numel() == 0:
        global_scale = 1.0
    else:
        scale_min = float(amaxs.min()) / 6.0
        exponent = torch.ceil(torch.log2(torch.tensor(min_normal / scale_min)))
        global_scale = float(2.0**exponent) if scale_min < min_normal else 1.0
    t = t * global_scale

    blocks = t.float().reshape(-1, 16)
    amax = blocks.absolute().amax(dim=1)
    scale = amax / 6.0
    # A block of exact zeros scales to 0 → decode has to multiply by 1, not 0.
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    # The scale itself is stored e4m3 (NVIDIA's NVFP4 layout): quantize it too.
    s_fp8 = scale.to(torch.float8_e4m3fn)
    q = blocks / s_fp8.float().unsqueeze(-1)
    # Nearest-magnitude rounding: searchsorted over the MIDPOINTS of adjacent
    # grid entries (bucketize alone rounds "≤ boundary", not nearest).
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=t.device)
    mag_idx = torch.searchsorted(midpoints, q.absolute().reshape(-1)).reshape(blocks.shape)
    sign = (q < 0) & (mag_idx > 0)  # zero codes carry no sign
    codes = (sign.to(torch.int32) * 8 + mag_idx).reshape(-1)  # sign<<3 | mag, flat
    packed = (codes[0::2] | (codes[1::2] << 4)).to(torch.uint8)
    return packed, s_fp8, global_scale


def _e2m1_dequant(packed: torch.Tensor, scale: torch.Tensor, global_scale: float) -> torch.Tensor:
    lows = packed.to(torch.int32) & 0x0F
    highs = (packed.to(torch.int32) >> 4) & 0x0F
    codes = torch.stack((lows, highs), dim=-1).reshape(-1)
    mags = torch.as_tensor(_NVFP4_MAGNITUDES, device=packed.device)
    vals = mags[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    vals = vals[: scale.numel() * 16].reshape(-1, 16)
    return (vals.float() * scale.float().unsqueeze(-1)).reshape(-1) * (1.0 / global_scale)


FACTORS_QUANT = {
    "fp8": None,  # native dtype; no packer needed
    "nvfp4": (_e2m1_quant, _e2m1_dequant),
}


def pack_tsct(t: torch.Tensor, rung: str) -> dict[str, object]:
    """One factor matrix → its storage record. Scales travel COMPUTED."""
    if t.dtype != torch.float32:
        loud(f"storage rungs quantize the fp32 masters, got dtype {t.dtype}")
    if not torch.isfinite(t).all():
        loud("storage: master factors must be finite (inf/NaN refuse, they encode wrong)")
    if rung == "fp8":
        scale = t.abs().amax() / RUNG_MAX["fp8"]
        scale = float(scale) if scale != 0 else 1.0
        packed = (t / scale).to(torch.float8_e4m3fn)
        return {"rung": rung, "packed": packed.reshape(t.shape), "scale": scale, "shape": t.shape}
    if rung == "nvfp4":
        packed, scale, correction = _e2m1_quant(t)
        return {
            "rung": rung,
            "codes": packed,
            "scale": scale,
            "global_scale": correction,
            "shape": t.shape,
            "n": t.numel(),
        }
    loud(f"pack_tsct: rung {rung!r} is not a storage rung (fp8|nvfp4)")


def unpack_tsct(rec: dict[str, object]) -> torch.Tensor:
    """Inverse of [`pack_tsct`] — master-precision dtype, rung-rounded values."""
    rung = rec["rung"]
    if rung == "fp8":
        # Encode was `packed = (t / scale)`; decode multiplies back up.
        return rec["packed"].float() * rec["scale"]
    if rung == "nvfp4":
        packed, scale = rec["codes"], rec["scale"]
        t = _e2m1_dequant(packed, scale, rec.get("global_scale", 1.0))
        return t.reshape(rec["shape"])
    loud(f"unpack_tsct: rung {rung!r} is not a storage rung (fp8|nvfp4)")


def is_factor_key(key: str) -> bool:
    """The state-dict keys that carry a quantizable factor matrix.

    Either a TSCT factor suffix (`u`, `v`, `gate_u`, `up_u`) on a module, or,
    because the experts' matrix params share those suffixes through
    `DualLatentGLU`, the same check passed through the experts path.
    """
    leaf = key.rpartition(".")[2]
    return leaf in {"u", "v", "gate_u", "up_u", "weight_u"}


class DenseLinear(nn.Module):
    """The `use_tsct = false` counterpart: a plain `[out, in]` weight.

    It exists because TSCT's A/B is its own removal, not TSCT against nothing:
    at a matched parameter budget the comparison is wide-and-low-rank against
    narrow-and-dense (burn's `LinearLikeInner::Dense`, reachable since
    2026-09-28 for exactly this reason).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        bias: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = dtype
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device) / math.sqrt(in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features, device=device)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.compute_dtype != torch.float32:
            x = x.to(self.compute_dtype)
            w = w.to(self.compute_dtype)
        y = x @ w.t()
        if self.compute_dtype != torch.float32:
            y = y.to(torch.float32)
        if self.bias is not None:
            y = y + self.bias
        return y


def make_projection(
    in_features: int,
    out_features: int,
    cfg,
    *,
    rank: int | None = None,
    bias: bool = False,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Spectral or dense, chosen once from `use_tsct` (a construction-time
    decision, so the shape of the model cannot drift afterwards)."""
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    rank = cfg.rank if rank is None else rank
    if cfg.use_tsct:
        return TSCTLinear(in_features, out_features, rank, dtype=dtype, bias=bias, device=device)
    return DenseLinear(in_features, out_features, dtype=dtype, bias=bias, device=device)
