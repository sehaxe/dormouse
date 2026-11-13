"""One Triton pass for the transformer block's residual→LayerNorm→bf16 seam.

WHAT THE EAGER BLOCK PAYS. `TransformerBlock._block` writes the residual twice
per layer and each write is its own kernel:

    x = x + proj(y).to(x.dtype)          # 1 cast (bf16→fp32) + 1 add   = 18 B/el
    h = mm_mixed(self.ln2(x), fc.w, ..)  # 1 LayerNorm + 1 cast (→bf16) = 14 B/el

so the seam costs 32 B/element forward, and the backward repeats it (cast of
`grad_h` back to fp32, `native_layer_norm_backward`, cast of `dx` down to bf16
for the projection GEMM) for ~26 B/element. At d=832, 12 layers, b8×s1024 that
is the `_to_copy` + `add` + half the `native_layer_norm` column of
`docs/records/reviews/2026-11-08-kernel-audit.md` §2.

WHAT THIS DOES. One program per row: read `x` (fp32) and `delta` (bf16), write
`x_new = x + delta` (fp32, the residual stream the VJP and the next block need)
and `h = layer_norm(x_new).to(bf16)` (the GEMM input), plus the row's `rstd`.
That is 12 B/element instead of 32, one kernel instead of four. The backward is
the same idea: read `grad_h` (bf16) and the saved `h` (bf16) and `rstd`, write
`grad_x` (fp32) and `grad_delta` (bf16, what the projection GEMM wants) — 10
B/element instead of ~26, one kernel instead of three plus the cast.

NUMERICS. The add is bit-identical to eager (`bf16→fp32` is exact, the add is
fp32). The norm is computed in fp32 with a two-stage in-register reduction —
sum for the mean, then the CENTERED sum of squares for the variance, not
`E[x²]−mean²` — so a large mean does not cost precision. `h` matches
`F.layer_norm(...).to(bf16)` to fp32 round-off (the test pins `atol=1e-6` on the
fp32 path and 1 ULP on the bf16 path); `rstd` is what the backward needs.
Padding lanes (`d=832` in a 1024-lane block) are masked out of both reductions.

The weight/bias VJP is a `sum` over rows. The kernel accumulates `dw`/`db`
across its rows in registers and writes one partial per program; the caller
reduces `[NR, d] → [d]` with `torch.sum` (0.4 MB, deterministic — no atomics,
so two runs of the same step get the same bits).

    # parity + mutation gate (CPU: the torch fallback; GPU: the kernel)
    .venv/bin/python -m pytest tests/test_fused_pointwise.py -q

`patch()` is a BENCH-ONLY entry point: it swaps `TransformerBlock._block` for a
verbatim copy of the body with the fused seam so `scripts/bench_pointwise_fusion.py`
and the CLI can A/B the change before it lands. The integration proposal is the
two-line diff in `docs/records/reviews/2026-11-08-pointwise-fusion.md` §5, not this.
"""

from __future__ import annotations

import torch

try:  # a CPU-only install (the test suite's default) has no triton wheel here
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover - import-time capability probe
    HAVE_TRITON = False

__all__ = ["eager", "fused_add_ln", "patch", "unpatch", "describe", "HAVE_TRITON"]

if HAVE_TRITON:

    @triton.jit
    def _add_ln_fwd(
        x_ptr,
        d_ptr,
        w_ptr,
        b_ptr,
        xn_ptr,
        mean_ptr,
        h_ptr,
        rstd_ptr,
        D: tl.constexpr,
        BD: tl.constexpr,
        EPS: tl.constexpr,
        OUT_FP32: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BD)
        mask = offs < D
        x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)
        dl = tl.load(d_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        xn = x + dl
        tl.store(xn_ptr + row * D + offs, xn, mask=mask)
        mean = tl.sum(xn, axis=0) / D
        c = tl.where(mask, xn - mean, 0.0)
        var = tl.sum(c * c, axis=0) / D
        r = 1.0 / tl.sqrt(var + EPS)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        y = c * r * w + b
        if OUT_FP32:
            tl.store(h_ptr + row * D + offs, y, mask=mask)
        else:
            tl.store(h_ptr + row * D + offs, y.to(tl.bfloat16), mask=mask)
        tl.store(mean_ptr + row, mean)
        tl.store(rstd_ptr + row, r)

    @triton.jit
    def _add_ln_bwd(
        g_ptr,
        xn_ptr,
        mean_ptr,
        rstd_ptr,
        w_ptr,
        dxn_ptr,
        dx_ptr,
        dd_ptr,
        dw_ptr,
        db_ptr,
        ROWS: tl.constexpr,
        RPP: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        HAS_DXN: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, BD)
        mask = offs < D
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        dw = tl.zeros([BD], tl.float32)
        db = tl.zeros([BD], tl.float32)
        # `tl.range`, not `static_range`: a 32x unroll of a 1024-lane
        # body spills registers and buys nothing (the loop is memory bound).
        for i in tl.range(0, RPP):
            row = pid * RPP + i
            rowm = row < ROWS
            m = mask & rowm
            g = tl.load(g_ptr + row * D + offs, mask=m, other=0.0).to(tl.float32)
            xn = tl.load(xn_ptr + row * D + offs, mask=m, other=0.0)
            mean = tl.load(mean_ptr + row, mask=rowm, other=0.0)
            r = tl.load(rstd_ptr + row, mask=rowm, other=0.0)
            h = (xn - mean) * r
            gw = g * w
            s1 = tl.sum(gw, axis=0) / D
            s2 = tl.sum(gw * h, axis=0) / D
            dx = r * (gw - s1 - h * s2)
            if HAS_DXN:
                dx += tl.load(dxn_ptr + row * D + offs, mask=m, other=0.0)
            tl.store(dx_ptr + row * D + offs, dx, mask=m)
            tl.store(dd_ptr + row * D + offs, dx.to(tl.bfloat16), mask=m)
            dw += tl.where(m, g * h, 0.0)
            db += tl.where(m, g, 0.0)
        tl.store(dw_ptr + pid * D + offs, dw, mask=mask)
        tl.store(db_ptr + pid * D + offs, db, mask=mask)


def eager(
    x: torch.Tensor,
    delta: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    *,
    out_fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The eager chain this kernel replaces, verbatim from `_block`.

    `x + delta.to(fp32)` then `F.layer_norm(...)` then the cast to bf16. Used as
    the CPU fallback and as the parity reference in the test.
    """
    xn = x + delta.to(torch.float32)
    y = torch.nn.functional.layer_norm(xn, (x.shape[-1],), weight, bias, eps)
    return xn, (y if out_fp32 else y.to(delta.dtype))


def _two(x: torch.Tensor) -> tuple[torch.Tensor, int, tuple[int, ...]]:
    shape = x.shape
    return x.reshape(-1, shape[-1]), shape[-1], shape


class _FusedAddLN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, weight, bias, eps, out_fp32):  # noqa: ANN001
        xf, D, shape = _two(x)
        df = delta.reshape(-1, D)
        xn = torch.empty_like(xf)
        h = torch.empty(
            xf.shape, dtype=torch.float32 if out_fp32 else delta.dtype, device=xf.device
        )
        mean = torch.empty(xf.shape[0], dtype=torch.float32, device=xf.device)
        rstd = torch.empty(xf.shape[0], dtype=torch.float32, device=xf.device)
        _add_ln_fwd[(xf.shape[0],)](
            xf,
            df,
            weight,
            bias,
            xn,
            mean,
            h,
            rstd,
            D=D,
            BD=triton.next_power_of_2(D),
            EPS=eps,
            OUT_FP32=out_fp32,
            num_warps=8,
        )
        # `h` is NOT saved: the backward recomputes it from `x_new` (fp32) and
        # the row's mean/rstd, which is exactly what `native_layer_norm_backward`
        # does. Saving the bf16 `h` instead would put a 0.4% quantization into
        # `grad_w`/`grad_b` — a real precision change for 2 B/element of traffic.
        ctx.save_for_backward(xn.reshape(shape), mean, rstd, weight)
        ctx.eps = eps
        ctx.shape = shape
        ctx.out_fp32 = out_fp32
        return xn.reshape(shape), h.reshape(shape)

    @staticmethod
    def backward(ctx, dxn, dh2):  # noqa: ANN001
        xn, mean, rstd, weight = ctx.saved_tensors
        shape = ctx.shape
        D = shape[-1]
        rows = xn.numel() // D
        xnf = xn.reshape(rows, D)
        gf = dh2.reshape(rows, D).contiguous()
        has_dxn = dxn is not None
        dxnf = (
            dxn.reshape(rows, D).contiguous()
            if has_dxn
            else torch.empty(1, dtype=torch.float32, device=xn.device)
        )
        dx = torch.empty(rows, D, dtype=torch.float32, device=xn.device)
        dd = torch.empty(rows, D, dtype=torch.bfloat16, device=xn.device)
        nr = min(rows, 256)
        rpp = triton.cdiv(rows, nr)
        nr = triton.cdiv(rows, rpp)
        dwp = torch.empty(nr, D, dtype=torch.float32, device=xn.device)
        dbp = torch.empty(nr, D, dtype=torch.float32, device=xn.device)
        _add_ln_bwd[(nr,)](
            gf,
            xnf,
            mean,
            rstd,
            weight,
            dxnf,
            dx,
            dd,
            dwp,
            dbp,
            ROWS=rows,
            RPP=rpp,
            D=D,
            BD=triton.next_power_of_2(D),
            HAS_DXN=has_dxn,
            num_warps=8,
        )
        dw = dwp.sum(0) if nr > 1 else dwp[0]
        db = dbp.sum(0) if nr > 1 else dbp[0]
        return (
            dx.reshape(shape),
            dd.reshape(shape),
            dw,
            db,
            None,
            None,
        )


def fused_add_ln(
    x: torch.Tensor,
    delta: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    *,
    out_fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(x + delta.float(), layer_norm(x + delta.float()).to(delta.dtype))`, one pass.

    `x` is the fp32 residual stream `[..., d]`, `delta` the bf16 projection
    output. Returns `(x_new, h)`: `x_new` is the residual the next block (and the
    add's VJP) needs, `h` is the bf16 GEMM input. `out_fp32=True` keeps `h` in
    fp32 — the parity test's path, not the model's.
    """
    # The kernel's contract is `x` fp32 + `delta` bf16 (what `mm_mixed` returns
    # with `bf16=true`). An fp32 `delta` (the `bf16=false` CUDA config) must NOT
    # enter it: the kernel would store `h` as bf16 and change the model's dtype.
    if not (
        HAVE_TRITON
        and x.is_cuda
        and delta.is_cuda
        and x.dtype == torch.float32
        and delta.dtype == torch.bfloat16
    ):
        return eager(x, delta, weight, bias, eps, out_fp32=out_fp32)
    return _FusedAddLN.apply(x, delta, weight, bias, float(eps), bool(out_fp32))


#: How many times the fused seam ran in this process (the bench reads it: a
#: number that claims a fusion must show the counter moved).
CALLS = 0


def _block_fused(self, x, counters=None, rope=None, v_first=None):
    """`TransformerBlock._block` with the residual→ln2 seam fused. BENCH COPY.

    Verbatim from `model/transformer.py::_block` with the two statements
    `x = x + ...to(x.dtype)` and `mm_mixed(self.ln2(x), ...)` replaced by one
    `fused_add_ln` call. The proposal lands this inside `_block` itself; the
    copy exists so the A/B runs today without touching the shared file.
    """
    global CALLS
    import torch.nn.functional as F

    from .transformer import apply_rope, rope_tables
    from .tsct import mm_mixed

    pdt = self.qkv.weight.dtype
    if x.dtype is not pdt:
        x = x.to(pdt)
    b, t, _ = x.shape
    d = self.d
    h1 = self.ln1(x)
    qkv = mm_mixed(h1, self.qkv.weight, self.compute_dtype, counters)
    q, k, v = qkv.split(d, dim=-1)
    q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
    k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
    if self.value_residual:
        if v_first is None:
            v_first = v
        else:
            v = v + self.v_res_alpha.to(v.dtype) * v_first
        self.last_v = v_first
    if not self.nope:
        if rope is None:
            cos, sin = rope_tables(t, self.head_dim, x.device, q.dtype)
        else:
            cos, sin = rope
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
    if self.gate is not None:
        g = torch.sigmoid(F.linear(h1, self.gate.weight, self.gate.bias))
        y = y * g.transpose(1, 2).unsqueeze(-1).to(y.dtype)
    y = y.transpose(1, 2).reshape(b, t, d)
    # >>> the fused seam: one pass for cast + residual add + LayerNorm + cast
    x, h = fused_add_ln(
        x,
        mm_mixed(y, self.proj.weight, self.compute_dtype, counters),
        self.ln2.weight,
        self.ln2.bias,
        self.ln2.eps,
    )
    CALLS += 1
    # <<<
    h = mm_mixed(h, self.fc.weight, self.compute_dtype, counters)
    h = F.gelu(h)
    if self.ffn_clip > 0.0:
        h = h.clamp(max=self.ffn_clip)
    h = mm_mixed(h, self.fc2.weight, self.compute_dtype, counters)
    return x + h.to(x.dtype)


_ORIG = None


def patch() -> None:
    """Swap `TransformerBlock._block` for the fused copy (bench A/B only)."""
    global _ORIG
    from .transformer import TransformerBlock

    if _ORIG is None:
        _ORIG = TransformerBlock._block
        TransformerBlock._block = _block_fused


def unpatch() -> None:
    global _ORIG
    from .transformer import TransformerBlock

    if _ORIG is not None:
        TransformerBlock._block = _ORIG
        _ORIG = None


def describe() -> str:
    return f"fused residual+LayerNorm seam (triton={HAVE_TRITON}); calls={CALLS}"
