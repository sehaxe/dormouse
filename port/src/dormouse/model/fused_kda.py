"""The KDA-side elementwise chain as two Triton kernels — `fused_kda_tail`.

**Why these two, and only these two.** `scripts/pass_census.py` counts every
elementwise pass of one m400b iteration (b8 x s1024, d 1536, fused residual +
epilogue ON): 4.17 GB and 40 launches. The groups, by traffic:

| group | GB/iter | us at 1.9 TB/s | launches |
|---|---|---|---|
| engram (value gate + `memory_floor_mix`) | 0.981 | 517 | 7 |
| **write (`y = attn·g1 + branch·g1 + ffn·g3`)** | **0.604** | **318** | **5** |
| csa (output scale + add) | 0.558 | 294 | 6 |
| loop (lane add, ctrl cat, out_acc add) | 0.554 | 291 | 3 |
| **kda (logsigmoid, sigmoid, 4 casts)** | **0.405** | **213** | **7** |
| mhc-write / mhc-read (already fused) | 0.554 | 292 | 2 |
| moe seam (2 casts + `w_ffn`) | 0.252 | 133 | 3 |
| readout + ce | 0.264 | 139 | 7 |

The owner's guess was the KDA tail; the census agrees it is a real group but
ranks the WRITE assembly higher. Both are fused here, because they are the two
groups the KDA arm's own output runs into and because the write assembly's
604 MB -> 201 MB is the largest genuine byte win outside the Engram read
(which is a separate arm and a follow-up, not this file: its `value` tensor is
[8192, 3, 1536] fp32 and the gate reduction is inside `engram._read`).

**Kernel 1 — `tail`.** `_fla_path` receives four fp32 projection outputs and
performs six eager passes: `q.to(bf16)`, `k.to(bf16)`, `v.to(bf16)`,
`beta.to(bf16)`, `F.logsigmoid(g)` and `torch.sigmoid(beta)` (the `.float()`
on `g`/`beta` is a no-op — the projection is already fp32). One program per row
does all six. The traffic does not drop (a cast reads its source and writes its
destination either way); FIVE LAUNCHES leave per iteration. That is the whole
argument for this half, stated so nobody sells it as a bandwidth win.

**Kernel 2 — `combine`.** `_step_post` computes
`y = (attn·attn_write + branch·attn_write + ffn·mlp_write)` with three scalar-
vector multiplies and two adds over [8192, 1536] fp32: 604 MB and five
launches. One row-program reads the three branch outputs and the two [d] gains,
writes `y`: 201 MB and one launch. Saves 403 MB (212 us at 1.9 TB/s) and four
launches per iteration. Without LayerRoPE-paper gains (`gains is None`) the
same kernel reads three tensors and writes their sum.

**Refusals.** Both kernels are fp32-input only: the residual stream and the
KDA projections are fp32 even under `bf16 = true` (the compute dtype is the
cast's DESTINATION, not its source), so the training arm qualifies. A bf16
input (a model cast whole to bf16) or a host without Triton/CUDA is a COUNTED
fallback to the eager lines (`fused_kda_tail_fallback`), each with an
`Escape:` in its reason. The OFF path is the eager code, bit for bit — the
flag is a route, not a formula.

**No host sync.** The launch counter is a device int32 written by program 0 and
handed to `Counters.count_device`; nothing reads it on the host here. The
backward's gain reductions are torch sums on the device.

**Numerics.** `logsigmoid(x) = min(x, 0) - log(1 + exp(-|x|))` is the stable
form; `sigmoid` is `1/(1+exp(-x))`. Neither is bit-identical to torch's
vectorised kernels (the last ulp differs), so the gate is 1e-5 relative on
outputs and every gradient, not "equal". The CPU mirror ([`tail_reference`],
[`combine_reference`]) is the same formula in torch: it lets a CPU host check
the fusion PLAN (`parity`) without a card.
"""

from __future__ import annotations

import torch

__all__ = [
    "RTOL",
    "GRTOL",
    "assert_parity",
    "combine",
    "combine_reference",
    "combine_supported",
    "host_reason",
    "parity",
    "tail",
    "tail_reference",
    "tail_supported",
    "unsupported_reason",
]

#: Output tolerance: 1e-5 relative, the fused-residual convention.
RTOL = 1e-5
#: Gradient tolerance; same rule, same reason (a VJP is a sum of several terms).
GRTOL = 5e-5

_TRITON = None
try:  # pragma: no cover - the import IS the gate
    import triton
    import triton.language as tl

    _TRITON = True
except Exception:  # pragma: no cover - CPU-only host
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON = False


def host_reason(device_type: str = "cuda") -> str | None:
    """Why this host cannot run the kernels, or None. COUNTED, not LOUD."""
    if not bool(_TRITON):
        return "Triton is not importable on this host"
    if not torch.cuda.is_available():
        return "this host has no CUDA device"
    if device_type != "cuda":
        return f"the tensors are on {device_type}"
    return None


def unsupported_reason(dtype: torch.dtype, device_type: str = "cuda") -> str | None:
    """Why this call cannot be fused, or None. The caller falls back on it."""
    host = host_reason(device_type)
    if host is not None:
        return (
            f"fused_kda_tail needs a CUDA device and Triton: {host}. "
            "Escape: fused_kda_tail=false, or run on the card."
        )
    if dtype != torch.float32:
        return (
            f"fused_kda_tail reproduces the fp32 chain only, got {dtype}. The "
            "projections and the residual stream are fp32 under bf16=true; a "
            "lower-precision input changes the arithmetic, not just its speed. "
            "Escape: the fp32 arm, or fused_kda_tail=false."
        )
    return None


def tail_supported(q: torch.Tensor) -> bool:
    return unsupported_reason(q.dtype, q.device.type) is None


def combine_supported(*tensors: torch.Tensor) -> bool:
    return all(t.dtype == torch.float32 for t in tensors) and (
        unsupported_reason(torch.float32, tensors[0].device.type) is None
    )


def tile_for(n: int) -> int:
    """`next_power_of_2(n)` — the block width a row-program must use."""
    return 1 << (int(n) - 1).bit_length()


def _warps(tile: int) -> int:
    """4/8/16 warps: one row of `d=1536` pads to 2048 and 8 warps spills."""
    assert tile >= 32 and (tile & (tile - 1)) == 0, f"tile={tile} must be a power of two"
    return 4 if tile <= 512 else (8 if tile <= 1024 else 16)


def _logsigmoid_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.minimum(x, torch.zeros_like(x)) - torch.log(1.0 + torch.exp(-x.abs()))


def tail_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dt: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The kernel's arithmetic in torch — the CPU parity arm."""
    return q.to(dt), k.to(dt), v.to(dt), _logsigmoid_ref(g), torch.sigmoid(beta).to(dt)


def combine_reference(
    attn: torch.Tensor,
    branch: torch.Tensor,
    ffn: torch.Tensor,
    gain_attn: torch.Tensor | None,
    gain_mlp: torch.Tensor | None,
) -> torch.Tensor:
    """The kernel's arithmetic in torch — the CPU parity arm."""
    if gain_attn is None or gain_mlp is None:
        return attn + branch + ffn
    return (attn + branch) * gain_attn + ffn * gain_mlp


if _TRITON:  # pragma: no cover - compiled at import

    @triton.jit
    def _tail_fwd_kernel(
        Q,
        K,
        V,
        G,
        BETA,
        QO,
        KO,
        VO,
        GO,
        BO,
        CNT,
        D: tl.constexpr,
        KV: tl.constexpr,
        NH: tl.constexpr,
        BD: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        OUT_BF16: tl.constexpr,
    ):
        """One row: cast q/k/v/beta, `logsigmoid(g)` and `sigmoid(beta)`."""
        row = tl.program_id(0)
        iq = tl.arange(0, BD)
        mq = iq < D
        q = tl.load(Q + row * D + iq, mask=mq, other=0.0)
        g = tl.load(G + row * D + iq, mask=mq, other=0.0)
        gl = tl.minimum(g, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(g)))
        tl.store(GO + row * D + iq, gl, mask=mq)
        if OUT_BF16:
            tl.store(QO + row * D + iq, q.to(tl.bfloat16), mask=mq)
        else:
            tl.store(QO + row * D + iq, q, mask=mq)
        ik = tl.arange(0, BK)
        mk = ik < KV
        k = tl.load(K + row * KV + ik, mask=mk, other=0.0)
        v = tl.load(V + row * KV + ik, mask=mk, other=0.0)
        if OUT_BF16:
            tl.store(KO + row * KV + ik, k.to(tl.bfloat16), mask=mk)
            tl.store(VO + row * KV + ik, v.to(tl.bfloat16), mask=mk)
        else:
            tl.store(KO + row * KV + ik, k, mask=mk)
            tl.store(VO + row * KV + ik, v, mask=mk)
        ib = tl.arange(0, BH)
        mb = ib < NH
        b = tl.load(BETA + row * NH + ib, mask=mb, other=0.0)
        bs = 1.0 / (1.0 + tl.exp(-b))
        if OUT_BF16:
            tl.store(BO + row * NH + ib, bs.to(tl.bfloat16), mask=mb)
        else:
            tl.store(BO + row * NH + ib, bs, mask=mb)
        if row == 0:
            tl.store(CNT, 1)

    @triton.jit
    def _tail_bwd_kernel(
        DQO,
        DKO,
        DVO,
        DGO,
        DBO,
        GRAW,
        BRAW,
        DQ,
        DK,
        DV,
        DG,
        DB,
        D: tl.constexpr,
        KV: tl.constexpr,
        NH: tl.constexpr,
        BD: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        OUT_BF16: tl.constexpr,
    ):
        """VJP: cast back for q/k/v, `σ(-x)` for logsigmoid, `σ(x)(1-σ(x))`."""
        row = tl.program_id(0)
        iq = tl.arange(0, BD)
        mq = iq < D
        if OUT_BF16:
            dq = tl.load(DQO + row * D + iq, mask=mq, other=0.0).to(tl.float32)
        else:
            dq = tl.load(DQO + row * D + iq, mask=mq, other=0.0)
        g = tl.load(GRAW + row * D + iq, mask=mq, other=0.0)
        dg = tl.load(DGO + row * D + iq, mask=mq, other=0.0) * (1.0 / (1.0 + tl.exp(g)))
        tl.store(DQ + row * D + iq, dq, mask=mq)
        tl.store(DG + row * D + iq, dg, mask=mq)
        ik = tl.arange(0, BK)
        mk = ik < KV
        if OUT_BF16:
            dk = tl.load(DKO + row * KV + ik, mask=mk, other=0.0).to(tl.float32)
            dv = tl.load(DVO + row * KV + ik, mask=mk, other=0.0).to(tl.float32)
        else:
            dk = tl.load(DKO + row * KV + ik, mask=mk, other=0.0)
            dv = tl.load(DVO + row * KV + ik, mask=mk, other=0.0)
        tl.store(DK + row * KV + ik, dk, mask=mk)
        tl.store(DV + row * KV + ik, dv, mask=mk)
        ib = tl.arange(0, BH)
        mb = ib < NH
        b = tl.load(BRAW + row * NH + ib, mask=mb, other=0.0)
        s = 1.0 / (1.0 + tl.exp(-b))
        if OUT_BF16:
            dbo = tl.load(DBO + row * NH + ib, mask=mb, other=0.0).to(tl.float32)
        else:
            dbo = tl.load(DBO + row * NH + ib, mask=mb, other=0.0)
        tl.store(DB + row * NH + ib, dbo * s * (1.0 - s), mask=mb)

    @triton.jit
    def _combine_fwd_kernel(
        A,
        B,
        C,
        G1,
        G3,
        Y,
        CNT,
        D: tl.constexpr,
        BD: tl.constexpr,
        GAIN: tl.constexpr,
    ):
        """One row: `(a+b)·g1 + c·g3`, or `a+b+c` when the gains are off."""
        row = tl.program_id(0)
        i = tl.arange(0, BD)
        m = i < D
        a = tl.load(A + row * D + i, mask=m, other=0.0)
        b = tl.load(B + row * D + i, mask=m, other=0.0)
        c = tl.load(C + row * D + i, mask=m, other=0.0)
        if GAIN:
            g1 = tl.load(G1 + i, mask=m, other=0.0)
            g3 = tl.load(G3 + i, mask=m, other=0.0)
            y = (a + b) * g1 + c * g3
        else:
            y = a + b + c
        tl.store(Y + row * D + i, y, mask=m)
        if row == 0:
            tl.store(CNT, 1)

    @triton.jit
    def _combine_bwd_kernel(
        DY,
        G1,
        G3,
        DA,
        DB,
        DC,
        D: tl.constexpr,
        BD: tl.constexpr,
        GAIN: tl.constexpr,
    ):
        """VJP: `dy·g1` for the two attention-side branches, `dy·g3` for the FFN."""
        row = tl.program_id(0)
        i = tl.arange(0, BD)
        m = i < D
        dy = tl.load(DY + row * D + i, mask=m, other=0.0)
        if GAIN:
            g1 = tl.load(G1 + i, mask=m, other=0.0)
            g3 = tl.load(G3 + i, mask=m, other=0.0)
            da = dy * g1
            dc = dy * g3
        else:
            da = dy
            dc = dy
        tl.store(DA + row * D + i, da, mask=m)
        tl.store(DB + row * D + i, da, mask=m)
        tl.store(DC + row * D + i, dc, mask=m)


class _Tail(torch.autograd.Function):
    """`q/k/v` cast, `logsigmoid(g)`, `sigmoid(beta)` — one forward launch."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, dt, cnt):  # type: ignore[override]
        b, t, d = q.shape
        kv, nh = k.shape[-1], beta.shape[-1]
        qo = torch.empty((b, t, d), dtype=dt, device=q.device)
        ko = torch.empty((b, t, kv), dtype=dt, device=k.device)
        vo = torch.empty((b, t, kv), dtype=dt, device=v.device)
        go = torch.empty((b, t, d), dtype=torch.float32, device=g.device)
        bo = torch.empty((b, t, nh), dtype=dt, device=beta.device)
        _tail_fwd_kernel[(b * t,)](
            q,
            k,
            v,
            g,
            beta,
            qo,
            ko,
            vo,
            go,
            bo,
            cnt,
            D=d,
            KV=kv,
            NH=nh,
            BD=tile_for(d),
            BK=tile_for(kv),
            BH=tile_for(nh),
            OUT_BF16=dt == torch.bfloat16,
            num_warps=_warps(tile_for(d)),
        )
        ctx.save_for_backward(q, k, v, g, beta)
        ctx.dt = dt
        return qo, ko, vo, go, bo

    @staticmethod
    def backward(ctx, dqo, dko, dvo, dgo, dbo):  # type: ignore[override]
        q, k, v, g, beta = ctx.saved_tensors
        b, t, d = q.shape
        kv, nh = k.shape[-1], beta.shape[-1]
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dg = torch.empty_like(g)
        db = torch.empty_like(beta)
        # The row-program reads contiguous gradients; `.contiguous()` is the
        # input itself on the normal path (an upstream kernel's fresh grad).
        dqo, dko, dvo, dgo, dbo = (t.contiguous() for t in (dqo, dko, dvo, dgo, dbo))
        _tail_bwd_kernel[(b * t,)](
            dqo,
            dko,
            dvo,
            dgo,
            dbo,
            g,
            beta,
            dq,
            dk,
            dv,
            dg,
            db,
            D=d,
            KV=kv,
            NH=nh,
            BD=tile_for(d),
            BK=tile_for(kv),
            BH=tile_for(nh),
            OUT_BF16=ctx.dt == torch.bfloat16,
            num_warps=_warps(tile_for(d)),
        )
        return dq, dk, dv, dg, db, None, None


class _Combine(torch.autograd.Function):
    """`y = (attn + branch)·attn_write + ffn·mlp_write` — one launch."""

    @staticmethod
    def forward(ctx, attn, branch, ffn, gain_attn, gain_mlp, cnt):  # type: ignore[override]
        b, t, d = attn.shape
        y = torch.empty_like(attn)
        gain = gain_attn is not None and gain_mlp is not None
        _combine_fwd_kernel[(b * t,)](
            attn,
            branch,
            ffn,
            gain_attn if gain else attn,
            gain_mlp if gain else attn,
            y,
            cnt,
            D=d,
            BD=tile_for(d),
            GAIN=gain,
            num_warps=_warps(tile_for(d)),
        )
        # `save_for_backward` rejects None; the dummy is never read when
        # `ctx.gain` is False (the kernel's GAIN constexpr is off there).
        ctx.save_for_backward(
            attn, branch, ffn, gain_attn if gain else attn, gain_mlp if gain else attn
        )
        ctx.gain = gain
        return y

    @staticmethod
    def backward(ctx, dy):  # type: ignore[override]
        attn, branch, ffn, gain_attn, gain_mlp = ctx.saved_tensors
        b, t, d = attn.shape
        da = torch.empty_like(attn)
        db = torch.empty_like(branch)
        dc = torch.empty_like(ffn)
        dy = dy.contiguous()
        _combine_bwd_kernel[(b * t,)](
            dy,
            gain_attn if ctx.gain else attn,
            gain_mlp if ctx.gain else attn,
            da,
            db,
            dc,
            D=d,
            BD=tile_for(d),
            GAIN=ctx.gain,
            num_warps=_warps(tile_for(d)),
        )
        if not ctx.gain:
            return da, db, dc, None, None, None
        # The gains are [d] vectors shared by every row: ONE reduction each.
        dg1 = (attn + branch).mul(dy).sum(dim=(0, 1))
        dg3 = ffn.mul(dy).sum(dim=(0, 1))
        return da, db, dc, dg1, dg3, None


def tail(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dt: torch.dtype,
    counters=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`[b,t,d]` fp32 projections -> the kernel's five tensors, one launch.

    The caller has already checked [`tail_supported`] (a CPU host or a bf16
    input must take the eager fallback and count it); asserting here keeps a
    mis-route from silently running a CPU tensor through a CUDA-only path.
    """
    why = unsupported_reason(q.dtype, q.device.type)
    if why is not None:
        raise AssertionError(why)
    assert dt in (torch.bfloat16, torch.float32), (
        f"fused_kda.tail writes bf16 or fp32; got dt={dt}. The kernel has no "
        "other output width (a float16 store would write the wrong bytes)."
    )
    assert q.shape == g.shape, f"q {tuple(q.shape)} and g {tuple(g.shape)} differ"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), (
        "the KDA tail reads row-contiguous projections; the reshape in "
        "`kda.projections_raw` is a view and keeps this true"
    )
    assert g.is_contiguous() and beta.is_contiguous(), "g/beta must be contiguous"
    cnt = torch.empty(1, dtype=torch.int32, device=q.device)
    out = _Tail.apply(q, k, v, g, beta, dt, cnt)
    if counters is not None:
        counters.count_device("fused_kda_tail_launches", cnt)
    return out


def combine(
    attn: torch.Tensor,
    branch: torch.Tensor,
    ffn: torch.Tensor,
    gain_attn: torch.Tensor | None,
    gain_mlp: torch.Tensor | None,
    counters=None,
) -> torch.Tensor:
    """`[b,t,d]` fp32 branches -> `y`, one launch; `gains=None` is the plain sum."""
    assert attn.shape == branch.shape == ffn.shape, (
        f"the write assembly needs one shape, got {tuple(attn.shape)} "
        f"{tuple(branch.shape)} {tuple(ffn.shape)}"
    )
    assert attn.is_contiguous() and branch.is_contiguous() and ffn.is_contiguous(), (
        "the write assembly reads row-contiguous branch outputs"
    )
    assert (gain_attn is None) == (gain_mlp is None), (
        "the two write gains are both present (LayerRoPE paper) or both absent"
    )
    if gain_attn is not None:
        assert gain_attn.dtype == torch.float32 and gain_mlp.dtype == torch.float32, (
            "the write gains are fp32 [d] vectors"
        )
        assert gain_attn.shape == (attn.shape[-1],), f"gain shape {tuple(gain_attn.shape)}"
    cnt = torch.empty(1, dtype=torch.int32, device=attn.device)
    y = _Combine.apply(attn, branch, ffn, gain_attn, gain_mlp, cnt)
    if counters is not None:
        counters.count_device("fused_kda_tail_launches", cnt)
    return y


def parity(
    b: int = 2,
    t: int = 8,
    d: int = 64,
    h: int = 4,
    *,
    seed: int = 0,
    use_kernel: bool = False,
    gains: bool = True,
) -> dict:
    """Max |Δ| of each fused plan against the eager calls, outputs + gradients.

    `use_kernel=False` is the CPU gate: [`tail_reference`] / [`combine_reference`]
    against `kda._fla_path`'s eager lines and `loopblock._step_post`'s eager
    lines. `use_kernel=True` is the card's gate: the same comparison against
    the Triton kernels. Every leaf the chain touches is compared, including
    the two [d] write gains — a fusion that loses a gradient path is the bug
    this function exists to catch.
    """
    dev = "cuda" if use_kernel else "cpu"
    if use_kernel:
        why = unsupported_reason(torch.float32, dev)
        assert why is None, why
    torch.manual_seed(seed)
    # ONE input set: both arms must see the same tensors, or the comparison is
    # between two random draws (the fused-residual parity makes the same point).
    q0 = (torch.randn(b, t, d, device=dev) * 0.5).requires_grad_(True)
    k0 = (torch.randn(b, t, d, device=dev) * 0.5).requires_grad_(True)
    v0 = (torch.randn(b, t, d, device=dev) * 0.5).requires_grad_(True)
    g0 = (torch.randn(b, t, d, device=dev) * 2.0).requires_grad_(True)
    beta0 = (torch.randn(b, t, h, device=dev) * 2.0).requires_grad_(True)
    a0 = (torch.randn(b, t, d, device=dev) * 0.5).requires_grad_(True)
    b0 = (torch.randn(b, t, d, device=dev) * 0.5).requires_grad_(True)
    c0 = (torch.randn(b, t, d, device=dev) * 0.5).requires_grad_(True)
    g10 = torch.randn(d, device=dev) * 0.1 + 1.0
    g30 = torch.randn(d, device=dev) * 0.1 + 1.0
    cot_tail = [torch.randn(b, t, d, device=dev) for _ in range(4)] + [
        torch.randn(b, t, h, device=dev)
    ]
    cot_y = torch.randn(b, t, d, device=dev)

    def _run_tail(fused: bool):
        q, k, v, g, beta = q0 * 1.0, k0 * 1.0, v0 * 1.0, g0 * 1.0, beta0 * 1.0
        dt = torch.bfloat16
        if fused:
            if use_kernel:
                outs = tail(q, k, v, g, beta, dt)
            else:
                outs = tail_reference(q, k, v, g, beta, dt)
        else:
            # The shipped eager chain: exactly kda.projections + _fla_path.
            outs = (
                q.to(dt),
                k.to(dt),
                v.to(dt),
                torch.nn.functional.logsigmoid(g),
                torch.sigmoid(beta).to(dt),
            )
        loss = sum((o.float() * c).sum() for o, c in zip(outs, cot_tail, strict=True))
        loss.backward()
        grads = {"q": q0.grad, "k": k0.grad, "v": v0.grad, "g": g0.grad, "beta": beta0.grad}
        for p in (q0, k0, v0, g0, beta0):
            p.grad = None
        return outs, grads

    (e0, e1, e2, e3, e4), eg = _run_tail(False)
    (f0, f1, f2, f3, f4), fg = _run_tail(True)
    out: dict[str, tuple[float, float]] = {}
    for name, a, bb in zip(
        ("q", "k", "v", "g", "beta"), (e0, e1, e2, e3, e4), (f0, f1, f2, f3, f4), strict=True
    ):
        out[name] = ((a.float() - bb.float()).abs().max().item(), a.float().abs().max().item())
    for name in ("q", "k", "v", "g", "beta"):
        out["d_" + name] = ((eg[name] - fg[name]).abs().max().item(), eg[name].abs().max().item())

    def _run_combine(fused: bool):
        a, bb, c = a0 * 1.0, b0 * 1.0, c0 * 1.0
        a.retain_grad()
        bb.retain_grad()
        c.retain_grad()
        if gains:
            g1 = (g10 * 1.0).requires_grad_(True)
            g3 = (g30 * 1.0).requires_grad_(True)
        else:
            g1 = g3 = None
        if fused:
            if use_kernel:
                y = combine(a, bb, c, g1, g3)
            else:
                y = combine_reference(a, bb, c, g1, g3)
        else:
            # Exactly loopblock._step_post's eager lines.
            if g1 is not None:
                a2, b2, c2 = a * g1, bb * g1, c * g3
            else:
                a2, b2, c2 = a, bb, c
            y = a2 + b2 + c2
        (y * cot_y).sum().backward()
        # A mutation may REMOVE a gradient path (a dropped gate has no grad);
        # that is a comparison the gate must see, not a crash here.
        grads = {
            "attn": None if a.grad is None else a.grad.clone(),
            "branch": None if bb.grad is None else bb.grad.clone(),
            "ffn": None if c.grad is None else c.grad.clone(),
        }
        if gains:
            grads["gain_attn"] = None if g1.grad is None else g1.grad.clone()
            grads["gain_mlp"] = None if g3.grad is None else g3.grad.clone()
        for p in (a0, b0, c0, g10, g30):
            p.grad = None
        return y, grads

    ey, eg2 = _run_combine(False)
    fy, fg2 = _run_combine(True)
    out["y"] = ((ey - fy).abs().max().item(), ey.abs().max().item())
    for name in eg2:
        ref, got = eg2[name], fg2[name]
        if ref is None or got is None:
            # One arm lost the path: an infinite delta on a zero scale, so
            # `assert_parity` names it instead of dividing by nothing.
            out["d_" + name] = (0.0 if ref is None and got is None else float("inf"), 0.0)
        else:
            out["d_" + name] = ((ref - got).abs().max().item(), ref.abs().max().item())
    out["scale"] = (0.0, max(v[1] for k, v in out.items() if k.startswith("d_") and v[1] > 0))
    return out


def assert_parity(rep: dict) -> None:
    """Raise LOUD naming the first tensor that misses the rule `parity` states."""
    scale = rep["scale"][1]
    assert scale > 0, "parity: the reference gradient is identically zero — nothing measured"
    for name, (delta, mag) in rep.items():
        if name == "scale":
            continue
        forward = not name.startswith("d_")
        tol = (RTOL if forward else GRTOL) * max(mag, 1e-30)
        if delta > tol:
            raise AssertionError(
                f"fused_kda_tail parity: {name} moved by {delta:.3e} (reference "
                f"{mag:.3e}, tolerance {tol:.3e}). The fused plan no longer "
                "reproduces the eager chain."
            )
