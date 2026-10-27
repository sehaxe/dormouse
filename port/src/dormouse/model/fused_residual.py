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

**Refusals** ([`unsupported_reason`]) name the combinations that cannot be
fused legally, each with its escape; the caller routes those to eager. A host
without Triton/CUDA is a **COUNTED** fallback (`fused_residual_fallback`), not a
silent one.
"""

from __future__ import annotations

import torch

__all__ = [
    "EPS_XN",
    "SINKHORN_ITERS",
    "fused_available",
    "parity",
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
    if d < 16 or (d & (d - 1)) != 0:
        return (
            f"fused_residual's tile is the whole lane row, so d must be a power of two "
            f">= 16; d_model={d}. Escape: fused_residual=false (the eager chain has no "
            "such limit), or a power-of-two d_model."
        )
    return None


def _warps(d: int) -> int:
    """Threads per program: one lane vector is 2·D floats, 4-8 floats per lane."""
    assert d >= 32 and (d & (d - 1)) == 0, f"D={d} must be a power of two (the tile is)"
    return 4 if d <= 512 else 8


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
        CNT,
        D: tl.constexpr,
        EPS: tl.constexpr,
    ):
        """One row: lane norm, the eight hyper-network dots, `h_pre`, `ctx`."""
        row = tl.program_id(0).to(tl.int64)
        i = tl.arange(0, D)
        base = row * (2 * D)
        x0 = tl.load(XL + base + i)
        x1 = tl.load(XL + base + D + i)
        inv = tl.rsqrt((tl.sum(x0 * x0) + tl.sum(x1 * x1)) / (2 * D) + EPS)
        xn0 = x0 * inv
        xn1 = x1 * inv
        tl.store(XN + base + i, xn0)
        tl.store(XN + base + D + i, xn1)

        # phi_* are [2·D, C] row-major: column c of row j is at j*C + c.
        p0 = tl.sum(xn0 * tl.load(PHI_PRE + i * 2)) + tl.sum(xn1 * tl.load(PHI_PRE + (D + i) * 2))
        p1 = tl.sum(xn0 * tl.load(PHI_PRE + i * 2 + 1)) + tl.sum(
            xn1 * tl.load(PHI_PRE + (D + i) * 2 + 1)
        )
        q0 = tl.sum(xn0 * tl.load(PHI_POST + i * 2)) + tl.sum(xn1 * tl.load(PHI_POST + (D + i) * 2))
        q1 = tl.sum(xn0 * tl.load(PHI_POST + i * 2 + 1)) + tl.sum(
            xn1 * tl.load(PHI_POST + (D + i) * 2 + 1)
        )
        r0 = tl.sum(xn0 * tl.load(PHI_RES + i * 4)) + tl.sum(xn1 * tl.load(PHI_RES + (D + i) * 4))
        r1 = tl.sum(xn0 * tl.load(PHI_RES + i * 4 + 1)) + tl.sum(
            xn1 * tl.load(PHI_RES + (D + i) * 4 + 1)
        )
        r2 = tl.sum(xn0 * tl.load(PHI_RES + i * 4 + 2)) + tl.sum(
            xn1 * tl.load(PHI_RES + (D + i) * 4 + 2)
        )
        r3 = tl.sum(xn0 * tl.load(PHI_RES + i * 4 + 3)) + tl.sum(
            xn1 * tl.load(PHI_RES + (D + i) * 4 + 3)
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
        tl.store(CTX + row * D + i, tl.sigmoid(pre0) * x0 + tl.sigmoid(pre1) * x1)
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
        EPS: tl.constexpr,
        COMPOSE: tl.constexpr,
        ITERS: tl.constexpr,
    ):
        """One row: Sinkhorn, the AttnRes write, `H_res xl + H_postᵀ a`, readout."""
        row = tl.program_id(0).to(tl.int64)
        i = tl.arange(0, D)
        base = row * (2 * D)
        x0 = tl.load(XL + base + i)
        x1 = tl.load(XL + base + D + i)
        y = tl.load(Y + row * D + i)

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
            w1 = 2.0 * tl.sigmoid(tl.sum(ry * tl.load(QEFF + i)))
        else:
            w1 = 1.0
        a = w1 * y

        o0 = hr00 * x0 + hr01 * x1 + hp0 * a
        o1 = hr10 * x0 + hr11 * x1 + hp1 * a
        if COMPOSE:
            h = (o0 + o1) * 0.5
        else:
            h = o0 + o1
        tl.store(SN + base + i, o0)
        tl.store(SN + base + D + i, o1)
        tl.store(H + row * D + i, h)
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
        EPS: tl.constexpr,
        COMPOSE: tl.constexpr,
        ITERS: tl.constexpr,
    ):
        """VJP of the write phase: readout → `H_post`/`H_res` → Sinkhorn → AttnRes."""
        row = tl.program_id(0).to(tl.int64)
        i = tl.arange(0, D)
        base = row * (2 * D)
        x0 = tl.load(XL + base + i)
        x1 = tl.load(XL + base + D + i)
        y = tl.load(Y + row * D + i)
        dh = tl.load(DH + row * D + i)
        hp0 = tl.load(HPOST + row * 2 + 0)
        hp1 = tl.load(HPOST + row * 2 + 1)
        hr00 = tl.load(HRES + row * 4 + 0)
        hr01 = tl.load(HRES + row * 4 + 1)
        hr10 = tl.load(HRES + row * 4 + 2)
        hr11 = tl.load(HRES + row * 4 + 3)
        w1 = tl.load(W1 + row)

        # h = (o0 + o1)/K (compose) or o0 + o1 (mhc only)
        if COMPOSE:
            g0 = tl.load(DSN + base + i) + dh * 0.5
            g1 = tl.load(DSN + base + D + i) + dh * 0.5
            a = w1 * y
        else:
            g0 = tl.load(DSN + base + i) + dh
            g1 = tl.load(DSN + base + D + i) + dh
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
            g_rms = ds1 * tl.load(QEFF + i)
            dyq = dal * w1
            tl.store(DQROW + row * D + i, ds1 * ry)
        else:
            g_rms = tl.zeros([D], dtype=tl.float32)
            dyq = dal
        inv_y = tl.rsqrt(tl.sum(y * y) / D + EPS)
        cy = inv_y * inv_y * inv_y * tl.sum(y * g_rms) / D
        tl.store(DY + row * D + i, dyq + inv_y * g_rms - cy * y)
        tl.store(DX + base + i, dx0)
        tl.store(DX + base + D + i, dx1)

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
        DX,
        DPRE,
        D: tl.constexpr,
        EPS: tl.constexpr,
    ):
        """VJP of the pre phase: `H_pre` → the eight φ dots → the lane RMSNorm."""
        row = tl.program_id(0).to(tl.int64)
        i = tl.arange(0, D)
        base = row * (2 * D)
        x0 = tl.load(XL + base + i)
        x1 = tl.load(XL + base + D + i)
        dctx = tl.load(DCTX + row * D + i)
        h0 = tl.sigmoid(tl.load(RAW + row * 8 + 0))
        h1 = tl.sigmoid(tl.load(RAW + row * 8 + 1))
        ap = tl.load(A_PRE)
        aq = tl.load(A_POST)
        ar = tl.load(A_RES)
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
            gp0 * tl.load(PHI_PRE + i * 2)
            + gp1 * tl.load(PHI_PRE + i * 2 + 1)
            + gq0 * tl.load(PHI_POST + i * 2)
            + gq1 * tl.load(PHI_POST + i * 2 + 1)
            + gr0 * tl.load(PHI_RES + i * 4)
            + gr1 * tl.load(PHI_RES + i * 4 + 1)
            + gr2 * tl.load(PHI_RES + i * 4 + 2)
            + gr3 * tl.load(PHI_RES + i * 4 + 3)
        )
        dxn1 = (
            gp0 * tl.load(PHI_PRE + (D + i) * 2)
            + gp1 * tl.load(PHI_PRE + (D + i) * 2 + 1)
            + gq0 * tl.load(PHI_POST + (D + i) * 2)
            + gq1 * tl.load(PHI_POST + (D + i) * 2 + 1)
            + gr0 * tl.load(PHI_RES + (D + i) * 4)
            + gr1 * tl.load(PHI_RES + (D + i) * 4 + 1)
            + gr2 * tl.load(PHI_RES + (D + i) * 4 + 2)
            + gr3 * tl.load(PHI_RES + (D + i) * 4 + 3)
        )
        inv = tl.rsqrt((tl.sum(x0 * x0) + tl.sum(x1 * x1)) / (2 * D) + EPS)
        c = inv * inv * inv * (tl.sum(x0 * dxn0) + tl.sum(x1 * dxn1)) / (2 * D)
        tl.store(DX + base + i, h0 * dctx + inv * dxn0 - c * x0)
        tl.store(DX + base + D + i, h1 * dctx + inv * dxn1 - c * x1)


class _Pre(torch.autograd.Function):
    """`ctx = H_pre·xl`, plus the six other hyper-network dots for the write."""

    @staticmethod
    def forward(
        ctx, xl, phi_pre, phi_post, phi_res, a_pre, a_post, a_res, b_pre, b_post, b_res, cnt
    ):  # type: ignore[override]
        bt, kd = xl.shape
        d = kd // K
        assert xl.is_contiguous() and kd == K * d, f"xl must be [bt, K·d] contiguous, {xl.shape}"
        assert cnt.dtype == torch.int32, "the launch counter is int32"
        ctx_t = torch.empty((bt, d), device=xl.device, dtype=torch.float32)
        xn = torch.empty((bt, kd), device=xl.device, dtype=torch.float32)
        raw = torch.empty((bt, COLS), device=xl.device, dtype=torch.float32)
        dot = torch.empty((bt, COLS), device=xl.device, dtype=torch.float32)
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
            cnt,
            D=d,
            EPS=EPS_XN,
            num_warps=_warps(d),
        )
        ctx.save_for_backward(xl, xn, raw, dot, phi_pre, phi_post, phi_res, a_pre, a_post, a_res)
        ctx.cnt = cnt
        return ctx_t, raw

    @staticmethod
    def backward(ctx, dctx, d_raw):  # type: ignore[override]
        xl, xn, raw, dot, phi_pre, phi_post, phi_res, a_pre, a_post, a_res = ctx.saved_tensors
        bt, kd = xl.shape
        d = kd // K
        if dctx is None:
            dctx = torch.zeros_like(xn[:, :d])
        if d_raw is None:
            d_raw = torch.zeros((bt, COLS), device=xl.device, dtype=torch.float32)
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
            dx,
            dpre,
            D=d,
            EPS=EPS_XN,
            num_warps=_warps(d),
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
        )


class _Write(torch.autograd.Function):
    """`x_{l+1} = H_res·xl + H_postᵀ·a` and the lane readout, in one kernel."""

    @staticmethod
    def forward(ctx, xl, y, raw, qeff, compose, eps, iters, cnt):  # type: ignore[override]
        bt, kd = xl.shape
        d = kd // K
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
            EPS=eps,
            COMPOSE=compose,
            ITERS=iters,
            num_warps=_warps(d),
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
            EPS=ctx.eps,
            COMPOSE=ctx.compose,
            ITERS=ctx.iters,
            num_warps=_warps(d),
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
    )
    # PREFIX-FORWARD COUNTER: `Counters.count_device` snapshots the value at
    # registration, so only kernels that have already run can be counted here —
    # the two backward launches are the same count and are measured by the
    # profiler instead (`scripts/bench_fused_residual.py`, and the test that
    # pins both numbers).
    counters.count_device("fused_residual_launches", cnt)
    ctx, raw = out
    return ctx.reshape(b, t, d), raw


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


def parity(b: int = 2, t: int = 8, d: int = 64, *, compose: bool = True, seed: int = 0) -> dict:
    """Max |Δ| of the fused chain against eager, on outputs AND on gradients.

    Both arms run the same inputs through autograd; the eager arm is the shipped
    modules themselves (`MHCMix.pre_mix`, `depth_attend`, `MHCMix.forward`), so
    this compares against the reference and not against a transcription of it.
    The init is deliberately NOT the model's (`b_pre = 10` saturates σ, which
    would make an `H_pre` defect invisible): `b_pre ~ N(0, 0.5)` and
    `b_res ~ N(0, 1.5)` keep every gate live.

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
    gc = torch.randn(b, t, d, device=dev)
    gh = torch.randn(b, t, d, device=dev)

    def run(fused: bool):
        for p in mhc.parameters():
            p.grad = None
        slot.query.grad = slot.gain.grad = None
        xl, y = base * 1.0, y0 * 1.0
        if fused:
            from ..instrument import Counters

            c = Counters()
            ctx, raw = pre(mhc, xl, c)
            sn, h = write(mhc, slot, xl, y, raw, compose, eps, c)
        else:
            ctx = mhc.pre_mix(xl)
            upd = depth_attend([torch.zeros_like(y), y], slot, eps, float(K)) if compose else y
            sn = mhc(xl, upd)
            h = sn.sum(dim=-2) / K if compose else sn.sum(dim=-2)
        ((ctx * gc).sum() + (h * gh).sum()).backward()
        grads = {n: p.grad.clone() for n, p in mhc.named_parameters() if p.grad is not None}
        if compose:
            grads["query"] = slot.query.grad.clone()
            grads["gain"] = slot.gain.grad.clone()
        grads["xl"] = base.grad.clone()
        grads["y"] = y0.grad.clone()
        base.grad = y0.grad = None
        return (ctx.detach(), sn.detach(), h.detach()), grads

    (c0, s0, h0), g0 = run(False)
    (c1, s1, h1), g1 = run(True)
    out: dict[str, tuple[float, float]] = {
        "ctx": ((c0 - c1).abs().max().item(), c0.abs().max().item()),
        "streams": ((s0 - s1).abs().max().item(), s0.abs().max().item()),
        "h": ((h0 - h1).abs().max().item(), h0.abs().max().item()),
    }
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
