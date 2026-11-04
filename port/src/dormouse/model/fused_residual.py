"""The loop's residual chain as fused Triton kernels: 2 forward, 2 backward.

**What this replaces, op by op.** One iteration of `loopblock._window` on the
mHC + AttnRes composition arm, in eager torch. Every tensor here is fp32: the
mHC hyper-network, the lanes and the body's write are fp32 even under
`bf16 = true`, because the attention and MoE branches cast their outputs back to
fp32 before the residual write (`_moe_branch`), and `mhc.py` builds its
parameters in the default dtype.

| # | op | shape | dtype |
|---|----|-------|-------|
| 1 | `xn = xl·rsqrt(mean(xl²)+1e-20)` (`_x_norm`, over the whole lane vector) | [b,t,K·d] | fp32 |
| 2 | `(xn @ phi_pre)·α + b_pre`, `sigmoid` → `h_pre` | [b,t,K] | fp32 |
| 3 | `Σ_k h_pre_k·xl_k` → `ctx` | [b,t,d] | fp32 |
| — | *(the body runs here: attention + Engram + MoE, not fusable)* | | |
| 4 | `rms(y)` → `s = Σ rms(y)·q_eff` → `2·softmax([0,s])[1]` → `w1·y` | [b,t,d] | fp32 |
| 5 | `xn @ phi_post`/`phi_res`, `α·`+`b`, `2σ` / Sinkhorn (20 iters) | [b,t,K], [b,t,K,K] | fp32 |
| 6 | `H_res xl + H_postᵀ·a` | [b,t,K,d] | fp32 |
| 7 | lane readout `Σ_k x_{l+1,k}/K` | [b,t,d] | fp32 |

Eager recomputes `_x_norm` **three times** over the same [b,t,K·d] state (once
in `pre_mix`, twice in `_post_res_tilde`/`forward`), and step 5 is 20 Sinkhorn
iterations × 2 `logsumexp` on a [b,t,2,2] tensor.

Measured 2026-10-27 on the 5060 Ti (16 GB), branch `fusedres` cut from `3d36844`,
`min of 5` fwd+bwd with the same cotangents, by
`scripts/bench_fused_residual.py --mode region`:

| shape | arm | ms min | launches | GPU ms |
|---|---|---|---|---|
| b1 s8192 K=2 d1024 | eager | 24.06 | 729 | 13.26 |
| b1 s8192 K=2 d1024 | **fused** | **7.09** | **29** | 5.24 |
| b8 s1024 K=2 d1024 | eager | 17.90 | 729 | 15.69 |
| b8 s1024 K=2 d1024 | **fused** | **5.24** | **29** | 5.59 |

3.4× on this chain, 25× fewer launches, 60-64 % less GPU time. On the WHOLE
trainer step (`--mode step`, same model, `min of 5` with 3 warm):

| shape | eager ms | fused ms | Δ | peak GiB |
|---|---|---|---|---|
| b1 s8192, depth 4 | 462.4 | 370.5 | **-19.9 %** | 8.30 → 7.63 |
| b8 s1024, depth 4 | 415.4 | 334.3 | **-19.5 %** | 6.91 → 6.18 |

The peak also drops: the eager chain's saved `pow`/`mean`/`mul` intermediates
are gone, and the fused path stores one `xn` instead of three.

**What this does instead**, per iteration:

* `_pre_kernel` — one program per row; `xn` once (stored, for the backward's
  parameter-gradient GEMM), all **eight** hyper-network dots (`phi_pre` 2,
  `phi_post` 2, `phi_res` 4) off that one `xn`, `h_pre`, `ctx`, the eight
  raw pre-activations (`RAW`) and the eight unscaled dots (`DOT`, for `dα`).
* `_write_kernel` — Sinkhorn in registers (40 normalisations, no launch between
  them), `rms(y)` + the AttnRes weight, `H_res xl + H_postᵀ a`, the lane
  readout. Stores the Sinkhorn *probability* trajectory: in a 2×2 each
  normalisation has one degree of freedom, so 4 sigmoid arguments per iteration
  (80 floats/row) are the whole backward recipe for 40 transposed maps.
* `_bwd_write_kernel` / `_bwd_pre_kernel` — the two VJPs, one pass each, in
  reverse order (readout → `H_post`/`H_res` → Sinkhorn → AttnRes → `H_pre` →
  RMSNorm), emitting the eight per-row parameter-gradient partials.

The parameter gradients themselves are torch calls: `xnᵀ @ dg` is a
`[K·d, 8]` GEMM and the `α`/`b` gradients are two reductions over [b,t,8].
Rolling those into Triton would buy no launch and no byte.

**Fidelity.** Same arithmetic in the same dtype: fp32 everywhere eager is
fp32, the same `rsqrt(mean+eps)` with the same two epsilons (`mhc.EPS = 1e-20`
for the lane norm, `cfg.norm_eps` for the AttnRes key norm), the same
20-iteration log-domain Sinkhorn, the same `2·softmax` AttnRes write. Two
deliberate departures, both printed by [`parity`]:

1. `_x_norm` is computed **once** instead of three times from the same input —
   bit-identical, since the eager recomputation is deterministic on the same
   tensor.
2. The AttnRes weight is `2σ(s)` instead of `2·softmax([0, s])[1]`: the same
   number in exact arithmetic, possibly the last ulp apart in fp32.

**The EPILOGUE read** ([`epilogue_read`], `fused_epilogue=true`). The same
`_pre_kernel` also emits the body's NORMED inputs: `ctx = H_pre x_l` is still
in registers, so `RMSNorm(ctx)` and its LayerRoPE gain multiply happen in that
one pass instead of a second kernel over `[b,t,d]`. Two gains when the paper
form owns separate `attn_read`/`mlp_read` sites, one when v1/plain RMSNorm does.
The gain VECTOR is built by the caller (the existing `LayerRoPENorm` /
`LayerRoPEPaperGains` call, O(d) = 6 kB): moving the trig into the kernel would
re-read the same vector once per row and buy nothing. Its gradient is a
per-channel reduction the backward kernel already has `normed` for
(`atomic_add`, 1.5 k adds/row).

**Refusals** ([`unsupported_reason`]) name the combinations that cannot be
fused legally, each with its escape; the caller routes those to eager. A host
without Triton/CUDA is a **COUNTED** fallback (`fused_residual_fallback` for
the write chain, `fused_epilogue_fallback` for the read chain), not a silent
one.

**Tiling.** Every program holds one row, so the block width is the row's
`next_power_of_2` and elements past `d` are masked: `d = 1536` (the m400
operating width) is not a power of two and the kernel has no reason to refuse
it. `MAX_TILE` is the register ceiling, and `d` must be even because LayerRoPE
rotates adjacent channel pairs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "EPS_XN",
    "SINKHORN_ITERS",
    "epilogue_read",
    "epilogue_reference",
    "fused_available",
    "parity",
    "parity_read",
    "pre",
    "unsupported_reason",
    "write",
]

#: `mhc.EPS` — the lane RMSNorm's ε (App. A.1 Table 5) — `mhc.py:69`.
EPS_XN = 1e-20
#: `mhc.SINKHORN_ITERS` — `sinkhorn.rs:5-6`, App. A.1's `t_max = 20`.
SINKHORN_ITERS = 20
#: Lane count. `config.validate` refuses anything but 2; the 2×2 is unrolled.
K = 2
#: Register ceiling: a program holds one whole lane vector (2·D floats).
MAX_KD = 4096
#: Register ceiling for ONE row tile (`next_power_of_2(d)`): 2 048 floats hold
#: `d = 1536` with a 33 % mask, at 8 warps = 8 floats/thread/lane.
MAX_TILE = 2048
#: The eight hyper-network outputs, in the order the kernels use them:
#: `[H̃_pre(2), H̃_post(2), H̃_res(4)]`, `H̃_res` row-major.
COLS = 8

_TRITON = None
try:  # pragma: no cover - the import IS the gate
    import triton
    import triton.language as tl

    _TRITON = True
except Exception:  # pragma: no cover - CPU-only host
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON = False


def fused_available() -> bool:
    """True when a Triton kernel can actually run here."""
    return bool(_TRITON) and torch.cuda.is_available()


def host_reason(device_type: str = "cuda") -> str | None:
    """Why this HOST cannot run the kernels, or None. COUNTED, not LOUD.

    A CPU run is a legal place to execute the model (the whole suite is), so a
    host without Triton/CUDA must not stop it — but the arm has to say so, which
    is what `fused_residual_fallback` counts.
    """
    if not bool(_TRITON):
        return "Triton is not importable on this host"
    if not torch.cuda.is_available():
        return "this host has no CUDA device"
    if device_type != "cuda":
        return f"the tensor is on {device_type}"
    return None


def unsupported_reason(
    dtype: torch.dtype, streams: int, kd: int, device_type: str = "cuda"
) -> str | None:
    """Why this shape cannot be fused, or None. The caller falls back on it."""
    host = host_reason(device_type)
    if host is not None:
        return (
            f"fused_residual needs a CUDA device and Triton: {host}. "
            "Escape: fused_residual=false, or run on the card."
        )
    if streams != K:
        return (
            f"fused_residual is a 2×2 Sinkhorn (K=2 lanes), got mhc_streams={streams}. "
            "Escape: mhc_streams=2, or fused_residual=false."
        )
    if dtype != torch.float32:
        return (
            f"fused_residual reproduces the fp32 chain only, got {dtype}. The mHC "
            "hyper-network and the lanes are fp32 in the reference; a lower-precision "
            "store changes the arithmetic, not just its speed. Escape: the fp32 arm, "
            "or fused_residual=false."
        )
    if kd > MAX_KD:
        return (
            f"fused_residual holds one lane vector (K·d = {kd} floats) in registers; "
            f"the ceiling is {MAX_KD}. Escape: fused_residual=false (the eager chain "
            "has no such limit)."
        )
    d = kd // max(streams, 1)
    if d < 32 or (d & 1):
        return (
            f"fused_residual's tile is the whole lane row and LayerRoPE rotates "
            f"adjacent pairs, so d must be even and >= 32; d_model={d}. Escape: "
            "fused_residual=false (the eager chain has no such limit), or an even "
            "d_model of at least 32."
        )
    tile = tile_for(d)
    if tile > MAX_TILE:
        return (
            f"fused_residual holds one lane row in registers: d_model={d} pads to a "
            f"{tile}-wide tile, and the ceiling is {MAX_TILE}. Escape: "
            "fused_residual=false (the eager chain has no such limit)."
        )
    return None


def tile_for(d: int) -> int:
    """`next_power_of_2(d)` — the block width a row-tile kernel must use."""
    return 1 << (int(d) - 1).bit_length()


def _warps(tile: int) -> int:
    """Threads per program: one lane vector is 2·tile floats, 4 floats per lane.

    `tile > 1024` is the masked `d = 1536` case: at 8 warps each thread would
    hold 8 elements of every one of the eight φ columns, which spills; 16 warps
    keeps the per-thread footprint at the 1024/8-warp level. Unmeasured here
    (the card is the owner's) — `scripts/bench_epilogue.py` is the check.
    """
    assert tile >= 32 and (tile & (tile - 1)) == 0, f"tile={tile} must be a power of two"
    return 4 if tile <= 512 else (8 if tile <= 1024 else 16)


if _TRITON:  # pragma: no cover - compiled at import
    #: A kernel may only read globals that are `constexpr` objects.
    _ITERS_C = tl.constexpr(SINKHORN_ITERS)

    @triton.jit
    def _lse2(a, b):
        """`logsumexp` over two scalars, the form `torch.logsumexp` uses."""
        mx = tl.maximum(a, b)
        return mx + tl.log(tl.exp(a - mx) + tl.exp(b - mx))

    @triton.jit
    def _pre_kernel(
        XL,
        PHI_PRE,
        PHI_POST,
        PHI_RES,
        A_PRE,
        A_POST,
        A_RES,
        B_PRE,
        B_POST,
        B_RES,
        CTX,
        XN,
        RAW,
        DOT,
        NORMED,
        GAIN,
        NORMED_MLP,
        GAIN_MLP,
        CNT,
        D: tl.constexpr,
        B: tl.constexpr,
        EPS: tl.constexpr,
        EPS_NORM: tl.constexpr,
        HAS_NORM: tl.constexpr,
        HAS_MLP: tl.constexpr,
    ):
        """One row: lane norm, the eight hyper-network dots, `h_pre`, `ctx`, and
        (the epilogue arm) the body's `RMSNorm(ctx)·gain` from the same registers.
        """
        row = tl.program_id(0).to(tl.int64)
        i = tl.arange(0, B)
        m = i < D
        base = row * (2 * D)
        x0 = tl.load(XL + base + i, mask=m, other=0.0)
        x1 = tl.load(XL + base + D + i, mask=m, other=0.0)
        inv = tl.rsqrt((tl.sum(x0 * x0) + tl.sum(x1 * x1)) / (2 * D) + EPS)
        xn0 = x0 * inv
        xn1 = x1 * inv
        tl.store(XN + base + i, xn0, mask=m)
        tl.store(XN + base + D + i, xn1, mask=m)

        # phi_* are [2·D, C] row-major: column c of row j is at j*C + c.
        p0 = tl.sum(xn0 * tl.load(PHI_PRE + i * 2, mask=m, other=0.0)) + tl.sum(
            xn1 * tl.load(PHI_PRE + (D + i) * 2, mask=m, other=0.0)
        )
        p1 = tl.sum(xn0 * tl.load(PHI_PRE + i * 2 + 1, mask=m, other=0.0)) + tl.sum(
            xn1 * tl.load(PHI_PRE + (D + i) * 2 + 1, mask=m, other=0.0)
        )
        q0 = tl.sum(xn0 * tl.load(PHI_POST + i * 2, mask=m, other=0.0)) + tl.sum(
            xn1 * tl.load(PHI_POST + (D + i) * 2, mask=m, other=0.0)
        )
        q1 = tl.sum(xn0 * tl.load(PHI_POST + i * 2 + 1, mask=m, other=0.0)) + tl.sum(
            xn1 * tl.load(PHI_POST + (D + i) * 2 + 1, mask=m, other=0.0)
        )
        r0 = tl.sum(xn0 * tl.load(PHI_RES + i * 4, mask=m, other=0.0)) + tl.sum(
            xn1 * tl.load(PHI_RES + (D + i) * 4, mask=m, other=0.0)
        )
        r1 = tl.sum(xn0 * tl.load(PHI_RES + i * 4 + 1, mask=m, other=0.0)) + tl.sum(
            xn1 * tl.load(PHI_RES + (D + i) * 4 + 1, mask=m, other=0.0)
        )
        r2 = tl.sum(xn0 * tl.load(PHI_RES + i * 4 + 2, mask=m, other=0.0)) + tl.sum(
            xn1 * tl.load(PHI_RES + (D + i) * 4 + 2, mask=m, other=0.0)
        )
        r3 = tl.sum(xn0 * tl.load(PHI_RES + i * 4 + 3, mask=m, other=0.0)) + tl.sum(
            xn1 * tl.load(PHI_RES + (D + i) * 4 + 3, mask=m, other=0.0)
        )
        tl.store(DOT + row * 8 + 0, p0)
        tl.store(DOT + row * 8 + 1, p1)
        tl.store(DOT + row * 8 + 2, q0)
        tl.store(DOT + row * 8 + 3, q1)
        tl.store(DOT + row * 8 + 4, r0)
        tl.store(DOT + row * 8 + 5, r1)
        tl.store(DOT + row * 8 + 6, r2)
        tl.store(DOT + row * 8 + 7, r3)

        # Eq. 7: H̃ = α·(x̃'φ) + b — α is applied here, so RAW is the
        # pre-activation the write kernel sigmoids / exponentiates.
        ap = tl.load(A_PRE)
        aq = tl.load(A_POST)
        ar = tl.load(A_RES)
        pre0 = ap * p0 + tl.load(B_PRE)
        pre1 = ap * p1 + tl.load(B_PRE + 1)
        post0 = aq * q0 + tl.load(B_POST)
        post1 = aq * q1 + tl.load(B_POST + 1)
        b0 = tl.load(B_RES + 0)
        b1 = tl.load(B_RES + 1)
        b2 = tl.load(B_RES + 2)
        b3 = tl.load(B_RES + 3)
        tl.store(RAW + row * 8 + 0, pre0)
        tl.store(RAW + row * 8 + 1, pre1)
        tl.store(RAW + row * 8 + 2, post0)
        tl.store(RAW + row * 8 + 3, post1)
        tl.store(RAW + row * 8 + 4, ar * r0 + b0)
        tl.store(RAW + row * 8 + 5, ar * r1 + b1)
        tl.store(RAW + row * 8 + 6, ar * r2 + b2)
        tl.store(RAW + row * 8 + 7, ar * r3 + b3)
        # Eq. 3's H_pre x_l: the sigmoid-weighted mix down to the body's width.
        ctx_v = tl.sigmoid(pre0) * x0 + tl.sigmoid(pre1) * x1
        tl.store(CTX + row * D + i, ctx_v, mask=m)
        if HAS_NORM:
            # The epilogue: the body's read is RMSNorm(ctx) with the caller's
            # LayerRoPE gain, computed while `ctx_v` is still in registers.
            invn = tl.rsqrt(tl.sum(ctx_v * ctx_v) / D + EPS_NORM)
            nrm = ctx_v * invn
            g = tl.load(GAIN + i, mask=m, other=0.0)
            tl.store(NORMED + row * D + i, nrm * g, mask=m)
            if HAS_MLP:
                g2 = tl.load(GAIN_MLP + i, mask=m, other=0.0)
                tl.store(NORMED_MLP + row * D + i, nrm * g2, mask=m)
        if tl.program_id(0) == 0:
            tl.store(CNT, 1)

    @triton.jit
    def _write_kernel(
        XL,
        Y,
        RAW,
        QEFF,
        SN,
        H,
        TRAJ,
        HRES,
        HPOST,
        W1,
        CNT,
        D: tl.constexpr,
        B: tl.constexpr,
        EPS: tl.constexpr,
        COMPOSE: tl.constexpr,
        ITERS: tl.constexpr,
    ):
        """One row: Sinkhorn, the AttnRes write, `H_res xl + H_postᵀ a`, readout."""
        row = tl.program_id(0).to(tl.int64)
        i = tl.arange(0, B)
        m = i < D
        base = row * (2 * D)
        x0 = tl.load(XL + base + i, mask=m, other=0.0)
        x1 = tl.load(XL + base + D + i, mask=m, other=0.0)
        y = tl.load(Y + row * D + i, mask=m, other=0.0)

        hp0 = 2.0 * tl.sigmoid(tl.load(RAW + row * 8 + 2))
        hp1 = 2.0 * tl.sigmoid(tl.load(RAW + row * 8 + 3))
        m00 = tl.load(RAW + row * 8 + 4)
        m01 = tl.load(RAW + row * 8 + 5)
        m10 = tl.load(RAW + row * 8 + 6)
        m11 = tl.load(RAW + row * 8 + 7)
        tob = TRAJ + row * (4 * ITERS)
        # Eq. 9, log domain: 20 × (row normalize, column normalize). Each
        # 2-element normalisation is one sigmoid — and that sigmoid IS the
        # backward's whole recipe for it, which is why it is stored.
        for it in tl.static_range(ITERS):
            p0 = tl.sigmoid(m00 - m01)
            p1 = tl.sigmoid(m10 - m11)
            l0 = _lse2(m00, m01)
            m00 -= l0
            m01 -= l0
            l1 = _lse2(m10, m11)
            m10 -= l1
            m11 -= l1
            q0 = tl.sigmoid(m00 - m10)
            q1 = tl.sigmoid(m01 - m11)
            l2 = _lse2(m00, m10)
            m00 -= l2
            m10 -= l2
            l3 = _lse2(m01, m11)
            m01 -= l3
            m11 -= l3
            tl.store(tob + it * 4 + 0, p0)
            tl.store(tob + it * 4 + 1, p1)
            tl.store(tob + it * 4 + 2, q0)
            tl.store(tob + it * 4 + 3, q1)
        hr00 = tl.exp(m00)
        hr01 = tl.exp(m01)
        hr10 = tl.exp(m10)
        hr11 = tl.exp(m11)

        if COMPOSE:
            # 2603.15031 Eq. 1/3 with S = 2 sources and u_0 = 0: `S·softmax` on
            # the one non-zero source is `2σ(s)`.
            ry = y * tl.rsqrt(tl.sum(y * y) / D + EPS)
            w1 = 2.0 * tl.sigmoid(tl.sum(ry * tl.load(QEFF + i, mask=m, other=0.0)))
        else:
            w1 = 1.0
        a = w1 * y

        o0 = hr00 * x0 + hr01 * x1 + hp0 * a
        o1 = hr10 * x0 + hr11 * x1 + hp1 * a
        if COMPOSE:
            h = (o0 + o1) * 0.5
        else:
            h = o0 + o1
        tl.store(SN + base + i, o0, mask=m)
        tl.store(SN + base + D + i, o1, mask=m)
        tl.store(H + row * D + i, h, mask=m)
        tl.store(HRES + row * 4 + 0, hr00)
        tl.store(HRES + row * 4 + 1, hr01)
        tl.store(HRES + row * 4 + 2, hr10)
        tl.store(HRES + row * 4 + 3, hr11)
        tl.store(HPOST + row * 2 + 0, hp0)
        tl.store(HPOST + row * 2 + 1, hp1)
        tl.store(W1 + row, w1)
        if tl.program_id(0) == 0:
            tl.store(CNT, 1)

    @triton.jit
    def _bwd_write_kernel(
        XL,
        Y,
        TRAJ,
        HRES,
        HPOST,
        W1,
        QEFF,
        DSN,
        DH,
        DX,
        DY,
        DA,
        DQROW,
        D: tl.constexpr,
        B: tl.constexpr,
        EPS: tl.constexpr,
        COMPOSE: tl.constexpr,
        ITERS: tl.constexpr,
    ):
        """VJP of the write phase: readout → `H_post`/`H_res` → Sinkhorn → AttnRes."""
        row = tl.program_id(0).to(tl.int64)
        i = tl.arange(0, B)
        m = i < D
        base = row * (2 * D)
        x0 = tl.load(XL + base + i, mask=m, other=0.0)
        x1 = tl.load(XL + base + D + i, mask=m, other=0.0)
        y = tl.load(Y + row * D + i, mask=m, other=0.0)
        dh = tl.load(DH + row * D + i, mask=m, other=0.0)
        hp0 = tl.load(HPOST + row * 2 + 0)
        hp1 = tl.load(HPOST + row * 2 + 1)
        hr00 = tl.load(HRES + row * 4 + 0)
        hr01 = tl.load(HRES + row * 4 + 1)
        hr10 = tl.load(HRES + row * 4 + 2)
        hr11 = tl.load(HRES + row * 4 + 3)
        w1 = tl.load(W1 + row)

        # h = (o0 + o1)/K (compose) or o0 + o1 (mhc only)
        if COMPOSE:
            g0 = tl.load(DSN + base + i, mask=m, other=0.0) + dh * 0.5
            g1 = tl.load(DSN + base + D + i, mask=m, other=0.0) + dh * 0.5
            a = w1 * y
        else:
            g0 = tl.load(DSN + base + i, mask=m, other=0.0) + dh
            g1 = tl.load(DSN + base + D + i, mask=m, other=0.0) + dh
            a = y
        dal = hp0 * g0 + hp1 * g1
        dhp0 = tl.sum(g0 * a)
        dhp1 = tl.sum(g1 * a)
        dhr00 = tl.sum(g0 * x0)
        dhr01 = tl.sum(g0 * x1)
        dhr10 = tl.sum(g1 * x0)
        dhr11 = tl.sum(g1 * x1)
        dx0 = hr00 * g0 + hr10 * g1
        dx1 = hr01 * g0 + hr11 * g1

        # h_res = exp(m), then the 40 transposed normalisations, in reverse.
        g00 = dhr00 * hr00
        g01 = dhr01 * hr01
        g10 = dhr10 * hr10
        g11 = dhr11 * hr11
        tob = TRAJ + row * (4 * ITERS)
        for j in tl.static_range(ITERS):
            it = ITERS - 1 - j
            q1 = tl.load(tob + it * 4 + 3)
            q0 = tl.load(tob + it * 4 + 2)
            p1 = tl.load(tob + it * 4 + 1)
            p0 = tl.load(tob + it * 4 + 0)
            s = g01 + g11
            g01 -= q1 * s
            g11 -= (1.0 - q1) * s
            s = g00 + g10
            g00 -= q0 * s
            g10 -= (1.0 - q0) * s
            s = g10 + g11
            g10 -= p1 * s
            g11 -= (1.0 - p1) * s
            s = g00 + g01
            g00 -= p0 * s
            g01 -= (1.0 - p0) * s
        # DA is the grad wrt the pre-activation; `_Pre.backward` scales it by α
        # into the φ GEMM. Columns 0/1 (H̃_pre) are zero: the write cannot see them.
        tl.store(DA + row * 8 + 0, 0.0)
        tl.store(DA + row * 8 + 1, 0.0)
        tl.store(DA + row * 8 + 4, g00)
        tl.store(DA + row * 8 + 5, g01)
        tl.store(DA + row * 8 + 6, g10)
        tl.store(DA + row * 8 + 7, g11)

        # H_post: hp = 2σ(post) → dpost = dhp·hp·(1 - hp/2)
        dpost0 = dhp0 * (hp0 * (1.0 - hp0 * 0.5))
        dpost1 = dhp1 * (hp1 * (1.0 - hp1 * 0.5))
        tl.store(DA + row * 8 + 2, dpost0)
        tl.store(DA + row * 8 + 3, dpost1)

        if COMPOSE:
            ry = y * tl.rsqrt(tl.sum(y * y) / D + EPS)
            ds1 = tl.sum(dal * y) * (w1 - 0.5 * w1 * w1)
            g_rms = ds1 * tl.load(QEFF + i, mask=m, other=0.0)
            dyq = dal * w1
            tl.store(DQROW + row * D + i, ds1 * ry, mask=m)
        else:
            g_rms = tl.zeros([B], dtype=tl.float32)
            dyq = dal
        inv_y = tl.rsqrt(tl.sum(y * y) / D + EPS)
        cy = inv_y * inv_y * inv_y * tl.sum(y * g_rms) / D
        tl.store(DY + row * D + i, dyq + inv_y * g_rms - cy * y, mask=m)
        tl.store(DX + base + i, dx0, mask=m)
        tl.store(DX + base + D + i, dx1, mask=m)

    @triton.jit
    def _bwd_pre_kernel(
        XL,
        RAW,
        D_RAW,
        DCTX,
        PHI_PRE,
        PHI_POST,
        PHI_RES,
        A_PRE,
        A_POST,
        A_RES,
        GAIN,
        GAIN_MLP,
        DNORMED,
        DNORMED_MLP,
        DX,
        DPRE,
        D: tl.constexpr,
        B: tl.constexpr,
        EPS: tl.constexpr,
        EPS_NORM: tl.constexpr,
        HAS_NORM: tl.constexpr,
        HAS_MLP: tl.constexpr,
    ):
        """VJP of the pre phase: the norm read → `H_pre` → the eight φ dots → the lane RMSNorm."""
        row = tl.program_id(0).to(tl.int64)
        i = tl.arange(0, B)
        m = i < D
        base = row * (2 * D)
        x0 = tl.load(XL + base + i, mask=m, other=0.0)
        x1 = tl.load(XL + base + D + i, mask=m, other=0.0)
        dctx = tl.load(DCTX + row * D + i, mask=m, other=0.0)
        h0 = tl.sigmoid(tl.load(RAW + row * 8 + 0))
        h1 = tl.sigmoid(tl.load(RAW + row * 8 + 1))
        ap = tl.load(A_PRE)
        aq = tl.load(A_POST)
        ar = tl.load(A_RES)
        if HAS_NORM:
            # VJP of `normed = RMSNorm(ctx)·gain`: ONE reduction over the two
            # read sites (attn_read + mlp_read when the paper form splits them).
            ctxv = h0 * x0 + h1 * x1
            invn = tl.rsqrt(tl.sum(ctxv * ctxv) / D + EPS_NORM)
            dn = tl.load(DNORMED + row * D + i, mask=m, other=0.0)
            gv = dn * tl.load(GAIN + i, mask=m, other=0.0)
            if HAS_MLP:
                dn2 = tl.load(DNORMED_MLP + row * D + i, mask=m, other=0.0)
                gv += dn2 * tl.load(GAIN_MLP + i, mask=m, other=0.0)
            cd = tl.sum(ctxv * gv)
            dctx += invn * gv - (invn * invn * invn / D) * cd * ctxv
        # `ctx = Σ h_pre_k·xl_k` contributes to the pre-activation gradient the
        # write kernel cannot see (it never reads RAW columns 0:2).
        dpre0 = tl.load(D_RAW + row * 8 + 0) + tl.sum(dctx * x0) * (h0 * (1.0 - h0))
        dpre1 = tl.load(D_RAW + row * 8 + 1) + tl.sum(dctx * x1) * (h1 * (1.0 - h1))
        tl.store(DPRE + row * 2 + 0, dpre0)
        tl.store(DPRE + row * 2 + 1, dpre1)
        # grad wrt the unscaled dot: dot_c is multiplied by α_c on its way to RAW.
        gp0 = dpre0 * ap
        gp1 = dpre1 * ap
        gq0 = tl.load(D_RAW + row * 8 + 2) * aq
        gq1 = tl.load(D_RAW + row * 8 + 3) * aq
        gr0 = tl.load(D_RAW + row * 8 + 4) * ar
        gr1 = tl.load(D_RAW + row * 8 + 5) * ar
        gr2 = tl.load(D_RAW + row * 8 + 6) * ar
        gr3 = tl.load(D_RAW + row * 8 + 7) * ar
        dxn0 = (
            gp0 * tl.load(PHI_PRE + i * 2, mask=m, other=0.0)
            + gp1 * tl.load(PHI_PRE + i * 2 + 1, mask=m, other=0.0)
            + gq0 * tl.load(PHI_POST + i * 2, mask=m, other=0.0)
            + gq1 * tl.load(PHI_POST + i * 2 + 1, mask=m, other=0.0)
            + gr0 * tl.load(PHI_RES + i * 4, mask=m, other=0.0)
            + gr1 * tl.load(PHI_RES + i * 4 + 1, mask=m, other=0.0)
            + gr2 * tl.load(PHI_RES + i * 4 + 2, mask=m, other=0.0)
            + gr3 * tl.load(PHI_RES + i * 4 + 3, mask=m, other=0.0)
        )
        dxn1 = (
            gp0 * tl.load(PHI_PRE + (D + i) * 2, mask=m, other=0.0)
            + gp1 * tl.load(PHI_PRE + (D + i) * 2 + 1, mask=m, other=0.0)
            + gq0 * tl.load(PHI_POST + (D + i) * 2, mask=m, other=0.0)
            + gq1 * tl.load(PHI_POST + (D + i) * 2 + 1, mask=m, other=0.0)
            + gr0 * tl.load(PHI_RES + (D + i) * 4, mask=m, other=0.0)
            + gr1 * tl.load(PHI_RES + (D + i) * 4 + 1, mask=m, other=0.0)
            + gr2 * tl.load(PHI_RES + (D + i) * 4 + 2, mask=m, other=0.0)
            + gr3 * tl.load(PHI_RES + (D + i) * 4 + 3, mask=m, other=0.0)
        )
        inv = tl.rsqrt((tl.sum(x0 * x0) + tl.sum(x1 * x1)) / (2 * D) + EPS)
        c = inv * inv * inv * (tl.sum(x0 * dxn0) + tl.sum(x1 * dxn1)) / (2 * D)
        tl.store(DX + base + i, h0 * dctx + inv * dxn0 - c * x0, mask=m)
        tl.store(DX + base + D + i, h1 * dctx + inv * dxn1 - c * x1, mask=m)


class _Pre(torch.autograd.Function):
    """`ctx = H_pre·xl`, plus the six other hyper-network dots for the write.

    The EPILOGUE arm (`HAS_NORM`) also returns `normed = RMSNorm(ctx)·gain` and,
    when the paper form splits the read site, `normed_mlp = RMSNorm(ctx)·gain_mlp`
    — both computed while `ctx` is in registers, so the body's read costs no
    second pass over `[b,t,d]`.
    """

    @staticmethod
    def forward(
        ctx,
        xl,
        phi_pre,
        phi_post,
        phi_res,
        a_pre,
        a_post,
        a_res,
        b_pre,
        b_post,
        b_res,
        cnt,
        gain,
        gain_mlp,
        eps_norm,
        has_norm,
        has_mlp,
    ):  # type: ignore[override]
        bt, kd = xl.shape
        d = kd // K
        tile = tile_for(d)
        assert xl.is_contiguous() and kd == K * d, f"xl must be [bt, K·d] contiguous, {xl.shape}"
        assert cnt.dtype == torch.int32, "the launch counter is int32"
        ctx_t = torch.empty((bt, d), device=xl.device, dtype=torch.float32)
        xn = torch.empty((bt, kd), device=xl.device, dtype=torch.float32)
        raw = torch.empty((bt, COLS), device=xl.device, dtype=torch.float32)
        dot = torch.empty((bt, COLS), device=xl.device, dtype=torch.float32)
        normed = (
            torch.empty((bt, d), device=xl.device, dtype=torch.float32)
            if has_norm
            else None
        )
        normed_mlp = (
            torch.empty((bt, d), device=xl.device, dtype=torch.float32)
            if has_norm and has_mlp
            else None
        )
        _pre_kernel[(bt,)](
            xl,
            phi_pre,
            phi_post,
            phi_res,
            a_pre,
            a_post,
            a_res,
            b_pre,
            b_post,
            b_res,
            ctx_t,
            xn,
            raw,
            dot,
            ctx_t if normed is None else normed,
            xl if normed is None else gain,
            ctx_t if normed_mlp is None else normed_mlp,
            xl if normed_mlp is None else gain_mlp,
            cnt,
            D=d,
            B=tile,
            EPS=EPS_XN,
            EPS_NORM=float(eps_norm),
            HAS_NORM=bool(has_norm),
            HAS_MLP=bool(has_mlp),
            num_warps=_warps(tile),
        )
        ctx.save_for_backward(
            xl, xn, raw, dot, phi_pre, phi_post, phi_res, a_pre, a_post, a_res, gain, gain_mlp,
            ctx_t,
        )
        ctx.cnt = cnt
        ctx.has_norm = bool(has_norm)
        ctx.has_mlp = bool(has_mlp)
        ctx.eps_norm = float(eps_norm)
        return ctx_t, raw, normed, normed_mlp

    @staticmethod
    def backward(ctx, dctx, d_raw, dnormed, dnormed_mlp):  # type: ignore[override]
        (
            xl,
            xn,
            raw,
            dot,
            phi_pre,
            phi_post,
            phi_res,
            a_pre,
            a_post,
            a_res,
            gain,
            gain_mlp,
            ctx_t,
        ) = ctx.saved_tensors
        bt, kd = xl.shape
        d = kd // K
        tile = tile_for(d)
        if dctx is None:
            dctx = torch.zeros((bt, d), device=xl.device, dtype=torch.float32)
        else:
            dctx = dctx.contiguous()
        if d_raw is None:
            d_raw = torch.zeros((bt, COLS), device=xl.device, dtype=torch.float32)
        has_norm = ctx.has_norm
        has_mlp = ctx.has_mlp
        dgain = dgain_mlp = None
        if not has_norm:
            dnormed = dctx
            dnormed_mlp = dctx
        else:
            if dnormed is None:
                dnormed = torch.zeros((bt, d), device=xl.device, dtype=torch.float32)
            else:
                dnormed = dnormed.contiguous()
            # The gain gradient is a reduction over rows — done in torch, not in
            # the kernel: float atomics make the sum order nondeterministic, and
            # this is the same expression `F.rms_norm`'s autograd uses.
            invn = torch.rsqrt(ctx_t.pow(2).mean(-1, keepdim=True) + ctx.eps_norm)
            nrm = ctx_t * invn
            dgain = (dnormed * nrm).sum(0)
            if has_mlp:
                if dnormed_mlp is None:
                    dnormed_mlp = torch.zeros((bt, d), device=xl.device, dtype=torch.float32)
                else:
                    dnormed_mlp = dnormed_mlp.contiguous()
                dgain_mlp = (dnormed_mlp * nrm).sum(0)
            else:
                dnormed_mlp = dctx
        dx = torch.empty_like(xl)
        dpre = torch.empty((bt, K), device=xl.device, dtype=torch.float32)
        _bwd_pre_kernel[(bt,)](
            xl,
            raw,
            d_raw,
            dctx,
            phi_pre,
            phi_post,
            phi_res,
            a_pre,
            a_post,
            a_res,
            gain,
            gain_mlp,
            dnormed,
            dnormed_mlp,
            dx,
            dpre,
            D=d,
            B=tile,
            EPS=EPS_XN,
            EPS_NORM=ctx.eps_norm,
            HAS_NORM=has_norm,
            HAS_MLP=has_mlp,
            num_warps=_warps(tile),
        )
        # The parameter gradients: ONE GEMM (the eight columns are α·d_raw, the
        # pre-columns using the total pre-activation gradient from the kernel)
        # and two reductions over [bt, 8].
        cols = torch.cat([dpre * a_pre, d_raw[:, 2:4] * a_post, d_raw[:, 4:8] * a_res], dim=1)
        grads = torch.cat([dpre, d_raw[:, 2:8]], dim=1)  # the raw pre-activation grads
        dphi_all = xn.t() @ cols  # [K·d, 8]
        dalpha = (grads * dot).sum(0)  # [8]
        db = grads.sum(0)  # [8]
        return (
            dx,
            dphi_all[:, 0:2].contiguous(),
            dphi_all[:, 2:4].contiguous(),
            dphi_all[:, 4:8].contiguous(),
            dalpha[0:2].sum().reshape(1),
            dalpha[2:4].sum().reshape(1),
            dalpha[4:8].sum().reshape(1),
            db[0:2].contiguous(),
            db[2:4].contiguous(),
            db[4:8].reshape(K, K).contiguous(),
            None,
            dgain,
            dgain_mlp,
            None,
            None,
            None,
        )


class _Write(torch.autograd.Function):
    """`x_{l+1} = H_res·xl + H_postᵀ·a` and the lane readout, in one kernel."""

    @staticmethod
    def forward(ctx, xl, y, raw, qeff, compose, eps, iters, cnt):  # type: ignore[override]
        bt, kd = xl.shape
        d = kd // K
        tile = tile_for(d)
        assert y.shape == (bt, d) and raw.shape == (bt, COLS), (
            f"the write's three shapes must agree: y {tuple(y.shape)} vs {(bt, d)}, "
            f"raw {tuple(raw.shape)} vs {(bt, COLS)}"
        )
        assert isinstance(compose, bool), "compose is a Python bool (a constexpr)"
        assert 0 <= iters <= SINKHORN_ITERS, f"iters={iters} outside 0..={SINKHORN_ITERS}"
        sn = torch.empty_like(xl)
        h = torch.empty_like(y)
        traj = torch.empty((bt, 4 * iters), device=xl.device, dtype=torch.float32)
        hres = torch.empty((bt, 4), device=xl.device, dtype=torch.float32)
        hpost = torch.empty((bt, K), device=xl.device, dtype=torch.float32)
        w1 = torch.empty((bt,), device=xl.device, dtype=torch.float32)
        _write_kernel[(bt,)](
            xl,
            y,
            raw,
            qeff,
            sn,
            h,
            traj,
            hres,
            hpost,
            w1,
            cnt,
            D=d,
            B=tile,
            EPS=eps,
            COMPOSE=compose,
            ITERS=iters,
            num_warps=_warps(tile),
        )
        ctx.save_for_backward(xl, y, traj, hres, hpost, w1, qeff, raw)
        ctx.cnt = cnt
        ctx.eps = eps
        ctx.compose = compose
        ctx.iters = iters
        return sn, h

    @staticmethod
    def backward(ctx, dsn, dh):  # type: ignore[override]
        xl, y, traj, hres, hpost, w1, qeff, raw = ctx.saved_tensors
        bt, kd = xl.shape
        d = kd // K
        tile = tile_for(d)
        if dsn is None:
            dsn = torch.zeros_like(xl)
        if dh is None:
            dh = torch.zeros_like(y)
        dx = torch.empty_like(xl)
        dy = torch.empty_like(y)
        da = torch.empty((bt, COLS), device=xl.device, dtype=torch.float32)
        dqrow = torch.empty_like(y) if ctx.compose else torch.empty(1, device=xl.device)
        _bwd_write_kernel[(bt,)](
            xl,
            y,
            traj,
            hres,
            hpost,
            w1,
            qeff,
            dsn,
            dh,
            dx,
            dy,
            da,
            dqrow,
            D=d,
            B=tile,
            EPS=ctx.eps,
            COMPOSE=ctx.compose,
            ITERS=ctx.iters,
            num_warps=_warps(tile),
        )
        dq = dqrow.sum(0) if ctx.compose else None
        return dx, dy, da, dq, None, None, None, None


def pre(mhc, xl: torch.Tensor, counters) -> tuple[torch.Tensor, torch.Tensor]:
    """`(ctx, raw)` for this iteration. `xl` is `[b,t,K,d]` contiguous, fp32.

    `raw` is `[b·t, 8]`, opaque to the caller: it is this module's hand-off to
    [`write`], not a value the loop reads.
    """
    b, t, k, d = xl.shape
    cnt = torch.empty(1, dtype=torch.int32, device=xl.device)
    out = _Pre.apply(
        xl.reshape(b * t, k * d),
        mhc.phi_pre,
        mhc.phi_post,
        mhc.phi_res,
        mhc.alpha_pre,
        mhc.alpha_post,
        mhc.alpha_res,
        mhc.b_pre,
        mhc.b_post,
        mhc.b_res,
        cnt,
        xl,
        xl,
        0.0,
        False,
        False,
    )
    # PREFIX-FORWARD COUNTER: `Counters.count_device` snapshots the value at
    # registration, so only kernels that have already run can be counted here —
    # the two backward launches are the same count and are measured by the
    # profiler instead (`scripts/bench_fused_residual.py`, and the test that
    # pins both numbers).
    counters.count_device("fused_residual_launches", cnt)
    ctx, raw = out[0], out[1]
    return ctx.reshape(b, t, d), raw


def epilogue_read(
    mhc,
    xl: torch.Tensor,
    gain: torch.Tensor,
    gain_mlp: torch.Tensor | None,
    eps_norm: float,
    counters,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """`(ctx, raw, normed, normed_mlp)` — the read half of one iteration, in ONE kernel.

    `xl [b,t,K,d]` fp32 lanes, `gain [d]` the LayerRoPE gain at THIS depth (built
    by the caller's existing `LayerRoPENorm`/paper-site call), `gain_mlp` the
    second read site when the paper form splits attn/mlp (`None` for v1 and the
    plain norm, where one gain serves both). The returned `ctx` is the
    controller's input, `normed*` the body's; `raw` is the write's hand-off.
    """
    b, t, k, d = xl.shape
    assert k == K, f"epilogue_read is the K=2 lane chain, got {k}"
    cnt = torch.empty(1, dtype=torch.int32, device=xl.device)
    ctx_t, raw, normed, normed_mlp = _Pre.apply(
        xl.reshape(b * t, k * d),
        mhc.phi_pre,
        mhc.phi_post,
        mhc.phi_res,
        mhc.alpha_pre,
        mhc.alpha_post,
        mhc.alpha_res,
        mhc.b_pre,
        mhc.b_post,
        mhc.b_res,
        cnt,
        gain,
        gain if gain_mlp is None else gain_mlp,
        float(eps_norm),
        True,
        gain_mlp is not None,
    )
    counters.count_device("fused_epilogue_launches", cnt)
    return (
        ctx_t.reshape(b, t, d),
        raw,
        normed.reshape(b, t, d),
        None if normed_mlp is None else normed_mlp.reshape(b, t, d),
    )


def write(
    mhc, slot, xl, y, raw, compose: bool, eps: float, counters, iters: int = SINKHORN_ITERS
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(x_{l+1}, h)` for this iteration — the mHC write plus the readout."""
    b, t, k, d = xl.shape
    cnt = torch.empty(1, dtype=torch.int32, device=xl.device)
    qeff = slot.gain * slot.query if compose else xl.reshape(-1)[:d]
    sn, h = _Write.apply(
        xl.reshape(b * t, k * d),
        y.reshape(b * t, d),
        raw,
        qeff,
        bool(compose),
        float(eps),
        int(iters),
        cnt,
    )
    counters.count_device("fused_residual_launches", cnt)
    return sn.reshape(b, t, k, d), h.reshape(b, t, d)


def epilogue_reference(
    mhc, xl: torch.Tensor, gain: torch.Tensor, gain_mlp: torch.Tensor | None, eps_norm: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The `_pre_kernel` read formula in torch — the CPU PARITY gate's fused arm.

    Deliberately NOT the eager modules: this is the arithmetic the kernel
    performs (one lane norm, one `ctx`, one `RMSNorm(ctx)·gain`), so a gate over
    it measures the fusion PLAN on a CPU host. The card's [`parity`] gate runs
    the actual kernel against the same numbers. Autograd through both gives the
    parameter gradients the rule also pins.
    """
    b, t, k, d = xl.shape
    flat = xl.reshape(b, t, k * d)
    inv = torch.rsqrt(flat.pow(2).sum(-1, keepdim=True) / (k * d) + EPS_XN)
    xn = flat * inv
    pre = mhc.alpha_pre * (xn @ mhc.phi_pre) + mhc.b_pre
    ctx = (torch.sigmoid(pre).unsqueeze(-1) * xl).sum(-2)
    n = ctx * torch.rsqrt(ctx.pow(2).mean(-1, keepdim=True) + eps_norm)
    return ctx, n * gain, None if gain_mlp is None else n * gain_mlp


def parity_read(
    b: int = 2,
    t: int = 8,
    d: int = 64,
    *,
    separate: bool = False,
    seed: int = 0,
    use_kernel: bool = False,
) -> dict:
    """`epilogue_read` (or its CPU mirror) vs the eager pre-norm chain, outputs + grads.

    `use_kernel=False` compares [`epilogue_reference`] against `MHCMix.pre_mix` +
    `F.rms_norm` — the CPU gate. `use_kernel=True` (the card) compares the real
    Triton kernel against the SAME eager calls; that is the kernel gate.
    """
    from .mhc import MHCMix

    dev = "cuda" if use_kernel else "cpu"
    if use_kernel and not fused_available():
        raise AssertionError("parity_read(use_kernel=True) needs Triton + a CUDA device")
    torch.manual_seed(seed)
    mhc = MHCMix(d, K, device=dev)
    with torch.no_grad():
        mhc.phi_pre.normal_(0, 0.02)
        mhc.phi_post.normal_(0, 0.02)
        mhc.phi_res.normal_(0, 0.02)
        mhc.alpha_pre.fill_(0.3)
        mhc.alpha_post.fill_(0.2)
        mhc.alpha_res.fill_(1.0)
        mhc.b_pre.normal_(0, 0.5)
        mhc.b_res.normal_(0, 1.5)
    eps = 1e-3
    gain = (torch.randn(d, device=dev) * 0.1 + 1.0).requires_grad_(True)
    gain2 = (torch.randn(d, device=dev) * 0.1 + 1.0).requires_grad_(True)
    base = (torch.randn(b, t, K, d, device=dev) * 0.5).requires_grad_(True)
    g1 = torch.randn(b, t, d, device=dev)
    g2 = torch.randn(b, t, d, device=dev)
    gc = torch.randn(b, t, d, device=dev)

    def run(fused: bool):
        for p in mhc.parameters():
            p.grad = None
        gain.grad = gain2.grad = None
        gain_mlp = gain2 if separate else None
        if fused:
            if use_kernel:
                from ..instrument import Counters

                ctx, _raw, n1, n2 = epilogue_read(mhc, base, gain, gain_mlp, eps, Counters())
            else:
                ctx, n1, n2 = epilogue_reference(mhc, base, gain, gain_mlp, eps)
        else:
            ctx = mhc.pre_mix(base)
            n1 = F.rms_norm(ctx, (d,), gain, eps)
            n2 = F.rms_norm(ctx, (d,), gain2, eps) if separate else None
        loss = (ctx * gc).sum() + (n1 * g1).sum()
        if n2 is not None:
            loss = loss + (n2 * g2).sum()
        loss.backward()
        grads = {n: p.grad.clone() for n, p in mhc.named_parameters() if p.grad is not None}
        grads["gain"] = gain.grad.clone()
        if separate:
            grads["gain_mlp"] = gain2.grad.clone()
        grads["xl"] = base.grad.clone()
        base.grad = None
        out = (ctx.detach(), n1.detach(), None if n2 is None else n2.detach())
        return out, grads

    (c0, n0, m0), g0 = run(False)
    (c1, n1, m1), g1 = run(True)
    out: dict[str, tuple[float, float]] = {
        "ctx": ((c0 - c1).abs().max().item(), c0.abs().max().item()),
        "normed": ((n0 - n1).abs().max().item(), n0.abs().max().item()),
    }
    if separate:
        assert m0 is not None and m1 is not None
        out["normed_mlp"] = ((m0 - m1).abs().max().item(), m0.abs().max().item())
    for k, v in g0.items():
        out["d_" + k] = ((v - g1[k]).abs().max().item(), v.abs().max().item())
    out["scale"] = (0.0, max(v[1] for k, v in out.items() if k.startswith("d_")))
    return out


def parity(
    b: int = 2,
    t: int = 8,
    d: int = 64,
    *,
    compose: bool = True,
    seed: int = 0,
    epilogue: bool = False,
    separate: bool = False,
) -> dict:
    """Max |Δ| of the fused chain against eager, on outputs AND on gradients.

    Both arms run the same inputs through autograd; the eager arm is the shipped
    modules themselves (`MHCMix.pre_mix`, `depth_attend`, `MHCMix.forward`), so
    this compares against the reference and not against a transcription of it.
    The init is deliberately NOT the model's (`b_pre = 10` saturates σ, which
    would make an `H_pre` defect invisible): `b_pre ~ N(0, 0.5)` and
    `b_res ~ N(0, 1.5)` keep every gate live.

    `epilogue=True` swaps the read half for [`epilogue_read`] and adds the
    body's `normed` (and, with `separate`, `normed_mlp`) to the compared
    tensors and the gains to the compared gradients.

    Returns `{name: (max|Δ|, max|ref|)}` plus `scale` (the largest reference
    gradient). [`assert_parity`] is the rule that turns it into a gate.
    """
    from .loopblock import AttnResSlot, depth_attend
    from .mhc import MHCMix

    dev = "cuda"
    torch.manual_seed(seed)
    mhc = MHCMix(d, K, device=dev)
    with torch.no_grad():
        mhc.phi_pre.normal_(0, 0.02)
        mhc.phi_post.normal_(0, 0.02)
        mhc.phi_res.normal_(0, 0.02)
        mhc.alpha_pre.fill_(0.3)
        mhc.alpha_post.fill_(0.2)
        mhc.alpha_res.fill_(1.0)
        mhc.b_pre.normal_(0, 0.5)
        mhc.b_res.normal_(0, 1.5)
    slot = AttnResSlot(d, dev)
    with torch.no_grad():
        slot.query.normal_(0, 0.5)
        slot.gain.normal_(1.0, 0.1)
    eps = 1e-3
    base = (torch.randn(b, t, K, d, device=dev) * 0.5).requires_grad_(True)
    y0 = (torch.randn(b, t, d, device=dev) * 0.5).requires_grad_(True)
    gain = (torch.randn(d, device=dev) * 0.1 + 1.0).requires_grad_(True)
    gain2 = (torch.randn(d, device=dev) * 0.1 + 1.0).requires_grad_(True)
    gc = torch.randn(b, t, d, device=dev)
    gn = torch.randn(b, t, d, device=dev)
    gn2 = torch.randn(b, t, d, device=dev)
    gh = torch.randn(b, t, d, device=dev)

    def run(fused: bool):
        for p in mhc.parameters():
            p.grad = None
        slot.query.grad = slot.gain.grad = None
        gain.grad = gain2.grad = None
        xl, y = base * 1.0, y0 * 1.0
        gain_mlp = gain2 if separate else None
        if fused:
            from ..instrument import Counters

            c = Counters()
            ctx, raw, normed, normed_mlp = epilogue_read(mhc, xl, gain, gain_mlp, eps, c)
            sn, h = write(mhc, slot, xl, y, raw, compose, eps, c)
        else:
            ctx = mhc.pre_mix(xl)
            normed = F.rms_norm(ctx, (d,), gain, eps)
            normed_mlp = F.rms_norm(ctx, (d,), gain2, eps) if separate else None
            upd = depth_attend([torch.zeros_like(y), y], slot, eps, float(K)) if compose else y
            sn = mhc(xl, upd)
            h = sn.sum(dim=-2) / K if compose else sn.sum(dim=-2)
        loss = (ctx * gc).sum() + (h * gh).sum() + (normed * gn).sum()
        if normed_mlp is not None:
            loss = loss + (normed_mlp * gn2).sum()
        loss.backward()
        grads = {n: p.grad.clone() for n, p in mhc.named_parameters() if p.grad is not None}
        if compose:
            grads["query"] = slot.query.grad.clone()
            grads["gain"] = slot.gain.grad.clone()
        if gain.grad is not None:
            grads["gain_read"] = gain.grad.clone()
        if separate:
            grads["gain_read_mlp"] = gain2.grad.clone()
        grads["xl"] = base.grad.clone()
        grads["y"] = y0.grad.clone()
        base.grad = y0.grad = None
        return (
            ctx.detach(),
            sn.detach(),
            h.detach(),
            normed.detach(),
            None if normed_mlp is None else normed_mlp.detach(),
        ), grads

    (c0, s0, h0, n0, m0), g0 = run(False)
    (c1, s1, h1, n1, m1), g1 = run(True)
    out: dict[str, tuple[float, float]] = {
        "ctx": ((c0 - c1).abs().max().item(), c0.abs().max().item()),
        "streams": ((s0 - s1).abs().max().item(), s0.abs().max().item()),
        "h": ((h0 - h1).abs().max().item(), h0.abs().max().item()),
        "normed": ((n0 - n1).abs().max().item(), n0.abs().max().item()),
    }
    if separate:
        assert m0 is not None and m1 is not None
        out["normed_mlp"] = ((m0 - m1).abs().max().item(), m0.abs().max().item())
    for k, v in g0.items():
        out["d_" + k] = ((v - g1[k]).abs().max().item(), v.abs().max().item())
    out["scale"] = (0.0, max(v[1] for k, v in out.items() if k.startswith("d_")))
    return out


#: Forward tolerance: relative to the output's own scale, ~100× fp32 epsilon.
PARITY_RTOL = 1e-5
#: Gradient tolerance, and the threshold below which a tensor is called
#: DEGENERATE — `1e-3` of the largest gradient in the graph:
#:
#: The 20-iteration Sinkhorn on a 2×2 is a near-constant map (it converges
#: geometrically to the uniform matrix), so its Jacobian — and therefore
#: `dphi_res`, `dalpha_res`, `db_res` — is of order 1e-6 while the gradients
#: beside it are of order 1e2. In fp32 both arms are at the noise floor there;
#: the comparison that means something is "both are tiny", not "they agree
#: relatively". Verified exact (4.9e-17) against autograd in float64.
PARITY_GRTOL = 5e-5
DEGENERATE = 1e-3


def assert_parity(rep: dict) -> None:
    """Raise LOUD on the first name that misses the rule [`parity`] states."""
    scale = rep["scale"][1]
    assert scale > 0, "parity: the reference gradient is identically zero — nothing measured"
    for name, (d, m) in rep.items():
        if name == "scale":
            continue
        forward = name in ("ctx", "streams", "h")
        tol = (PARITY_RTOL if forward else PARITY_GRTOL) * max(m, 1e-30)
        if not forward and m < DEGENERATE * scale:
            tol = DEGENERATE * scale  # a degenerate tensor: both arms are noise
        if d > tol:
            raise AssertionError(
                f"fused_residual parity: {name} moved by {d:.3e} (reference {m:.3e}, "
                f"tolerance {tol:.3e}). The fused kernel no longer reproduces the eager chain."
            )
