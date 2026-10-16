"""Triton fused gather attention for the MSA arm — the "fused gather drop-in"
`sparse/attention.py` names as the intended path, and the one this arm needs.

WHY THIS FILE EXISTS (measured, 2026-10-16): the chunked torch route
(`attend_selected_blocks`) MATERIALIZES the gathered keys and values as
`[b, t, chunk*r, h, hd]` and then runs two `einsum`s over them. That tensor is
the whole cost of the arm: at the `small` preset b8 s512 (h 12, hd 64, r 4,
K 32) the arm's own overhead was **615 ms/step** against 91 ms/step for the same
run with the arm off (7.7x; `~/logs/ab_msa_msaon_s1.jsonl` vs
`ab_msa_msoff_s1.jsonl`, 2026-10-16), and the chunk sweep says no chunk size
reaches below ~632 ms/step — the materialization IS the arm's cost, not the
chunking (`MSA_SLOT_CHUNK`'s comment carries the sweep).

This module never builds that tensor. One program owns `BLOCK_T` queries of one
(batch, head) pair and walks the selected columns in `BLOCK_C` steps, loading
each gathered key/value row straight out of `k`/`v` — which is 25 MB at the
shape above, so the whole gather is an L2 hit instead of a 1.6 GB round trip per
call. The online softmax is the same recurrence as the torch route's chunk loop
(running max, running denominator, rescaled numerator), so the two routes agree
by construction and the parity gate (`tests/test_msa_speed.py`) checks it.

House rules, the same ones `model/moe_kernels.py` runs under:

* **pure torch in / torch out.** Plain tensors in, a plain tensor out; the
  layout contract is the docstring of each entry point and is `assert`ed.
* **no module-level mutable state**, no allocator, no workspace, no cache.
* **no host sync** — nothing in this file calls `.item()`.
* **autograd correct or LOUD.** `gather_attention` is an `autograd.Function`
  whose backward is a second Triton kernel (the flash-style adjoint with the
  gather's scatter), not a second implementation of the math.
* **COUNTED fallback**: `attention.attend_selected_blocks` counts `msa_triton`
  when this arm runs and `msa_triton_fallback` + the reason when it does not.

Index contract (`attention.attend_selected_blocks` builds it):

    pos [b, t, C] int32   absolute key position of column c for query i, or
                          -1 for a column that is NOT valid for that query
                          (padded top-K slot, out of range, or not yet causal)

Tiles are module constants on purpose (the `TILES` rule in `moe_kernels.py`): a
Triton `autotune` inside a training loop re-tunes whenever a shape key moves.
`tests/test_msa_speed.py -- tiles` measured them; re-measure and edit the constants,
do not add a decorator.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: The shipped tile set, picked by the b8 s512 `small` sweep in
#: `tests/test_msa_speed.py -- tiles` (that module's `__main__` prints the table).
#: MEASURED 2026-10-16, b8 s512 h12 hd64 K32 r4, one call fwd+bwd, median of 5
#: (`small`'s own attention shape; the torch route is ~600 ms for the same call):
#: T8 C2 w8 **7.7-8.2 ms** · T16 C1 w8 7.9 · T16 C2 w16 8.3 · T8 C4 w4 12.8-13.0
#: · T16 C8 w4 21.2 · T32 C8 w8 24.7 · T64 C8 w8 78.5. The winner is the SMALL
#: tile: `BLOCK_C` 2 keeps the `[BLOCK_T, BLOCK_C, HD]` gather at 1 K floats per
#: program, so the kernel is occupancy-bound on the L2-resident gather rather
#: than register-bound, and the plateau from C2 to C1 is inside the noise band
#: (C1 also makes `tl.arange` size-1, which C2 avoids). Peak was 0.16 GiB at
#: every config — the fused route's memory is the score plan, not the gather.
TILES = {"BLOCK_T": 8, "BLOCK_C": 2, "NUM_WARPS": 8}

assert set(TILES) == {"BLOCK_T", "BLOCK_C", "NUM_WARPS"}

#: The largest head width this kernel tiles (`tl.arange` needs a power of two).
#: head_dim is 16/64/128 across the presets; a wider one is the torch arm's job.
MAX_HEAD_DIM = 128


def _tiles(tiles: dict | None) -> dict:
    t = dict(TILES)
    if tiles:
        unknown = set(tiles) - set(TILES)
        assert not unknown, f"unknown tile(s) {sorted(unknown)}; known: {sorted(TILES)}"
        t.update(tiles)
    return t


def _check(q, k, v, pos) -> None:
    assert q.ndim == 4 and q.shape == k.shape == v.shape, "q/k/v must be [b,t,h,hd]"
    b, t, h, hd = q.shape
    assert pos.shape[:2] == (b, t) and pos.dtype == torch.int32, "pos must be [b,t,C] int32"
    assert hd >= 16 and hd <= MAX_HEAD_DIM and (hd & (hd - 1)) == 0, (
        f"head_dim={hd} is not tileable"
    )
    for x in (q, k, v, pos):
        assert x.is_cuda and x.is_contiguous(), "the Triton arm wants contiguous CUDA tensors"
    assert q.dtype == k.dtype == v.dtype and q.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), f"dtype {q.dtype} has no gather-attention tile"
    # Every offset this kernel computes is int32 (the pointers are 64-bit, the
    # index arithmetic is not): assert the bound rather than silently wrapping.
    assert b * t * h * hd < 2**31, "the int32 index arithmetic cannot address this shape"


def _launch_fwd(q, k, v, pos, scale, s_out):
    """The forward kernel; `s_out` None means "do not write the score plan"."""
    b, t, h, hd = q.shape
    c = pos.shape[-1]
    out = torch.empty_like(q)
    stats = torch.empty(b, t, h, 2, dtype=torch.float32, device=q.device)
    tiles = _tiles(None)
    grid = (b * h, triton.cdiv(t, tiles["BLOCK_T"]))
    _fwd_kernel[grid](
        q,
        k,
        v,
        pos,
        out,
        out if s_out is None else s_out,
        stats,
        t,
        h,
        c,
        scale,
        BLOCK_T=tiles["BLOCK_T"],
        BLOCK_C=tiles["BLOCK_C"],
        HD=hd,
        WRITE_S=s_out is not None,
        num_warps=tiles["NUM_WARPS"],
    )
    return out, stats


@triton.jit
def _fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    pos_ptr,
    out_ptr,
    s_ptr,
    st_ptr,
    T,
    H,
    C,
    scale,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HD: tl.constexpr,
    WRITE_S: tl.constexpr,
):
    b = tl.program_id(0) // H
    h = tl.program_id(0) % H
    ot = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    mt = ot < T
    hd = tl.arange(0, HD)
    base = b * T * H * HD + h * HD
    ld = ot[:, None] * (H * HD) + hd[None, :]
    # fp32 accumulation, one rounding at the store: the seam is fp32 in the port
    # (`--bf16` casts at the Linear, not here), and a bf16 softmax would be a
    # 2^-8 step the parity gate could not tell from a wiring error.
    qt = tl.load(q_ptr + base + ld, mask=mt[:, None], other=0.0).to(tl.float32)
    kbase = k_ptr + base
    vbase = v_ptr + base
    # The torch route's running (max, denominator, numerator); `-inf` max is
    # sanitized to 0 exactly as `attend_selected_blocks` does, so an empty chunk
    # rescales by exp(-inf) = 0 instead of producing a NaN.
    m_i = tl.full([BLOCK_T], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_T], tl.float32)
    acc = tl.zeros([BLOCK_T, HD], tl.float32)
    for c0 in range(0, C, BLOCK_C):
        oc = c0 + tl.arange(0, BLOCK_C)
        mc = oc < C
        idx = tl.load(
            pos_ptr + b * T * C + ot[:, None] * C + oc[None, :],
            mask=mt[:, None] & mc[None, :],
            other=-1,
        )
        m = (idx >= 0) & mt[:, None] & mc[None, :]
        kk = tl.load(
            kbase + idx[:, :, None] * (H * HD) + hd[None, None, :], mask=m[:, :, None], other=0.0
        ).to(tl.float32)
        s = tl.where(m, tl.sum(qt[:, None, :] * kk, axis=2) * scale, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_used = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_used)
        p = tl.exp(s - m_used[:, None])
        vv = tl.load(
            vbase + idx[:, :, None] * (H * HD) + hd[None, None, :], mask=m[:, :, None], other=0.0
        ).to(tl.float32)
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * vv, axis=1)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_used
        if WRITE_S:
            # The RAW score, not the running `p`: `p` is normalized by the
            # running max at the time of its chunk, and a backward that read it
            # as the softmax would be wrong by a factor exp(m_chunk - m_final)
            # per chunk (this was a real 11x error on dq before the fix). The
            # final (max, denominator) go to `st_ptr`, so the backward can
            # rebuild the exact softmax with two loads.
            tl.store(
                s_ptr + b * T * H * C + ot[:, None] * (H * C) + h * C + oc[None, :],
                s,
                mask=mt[:, None] & mc[None, :],
            )
    if WRITE_S:
        st = st_ptr + (b * T + ot) * (H * 2) + h * 2
        tl.store(st, m_i, mask=mt)
        tl.store(st + 1, l_i, mask=mt)
    # Same clamp as the torch route (`den.clamp_min(finfo.tiny)`): a query with
    # no valid key reads 0.0 in both routes rather than a NaN in one of them.
    tl.store(
        out_ptr + base + ld,
        (acc / tl.maximum(l_i, 1.1754944e-38)[:, None]).to(out_ptr.dtype.element_ty),
        mask=mt[:, None],
    )


@triton.jit
def _bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    pos_ptr,
    s_ptr,
    st_ptr,
    do_ptr,
    out_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    T,
    H,
    C,
    scale,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HD: tl.constexpr,
):
    b = tl.program_id(0) // H
    h = tl.program_id(0) % H
    ot = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    mt = ot < T
    hd = tl.arange(0, HD)
    base = b * T * H * HD + h * HD
    ld = ot[:, None] * (H * HD) + hd[None, :]
    qt = tl.load(q_ptr + base + ld, mask=mt[:, None], other=0.0).to(tl.float32)
    dot = tl.load(do_ptr + base + ld, mask=mt[:, None], other=0.0).to(tl.float32)
    outt = tl.load(out_ptr + base + ld, mask=mt[:, None], other=0.0).to(tl.float32)
    # The flash-attention row sum: `delta_i = do_i . o_i` = `sum_c p_ic dp_ic`.
    delta = tl.sum(dot * outt, axis=1)
    # The forward saved the RAW scores plus the final (max, denominator), so the
    # softmax is rebuilt here exactly: a masked column's stored score is -inf.
    st = st_ptr + (b * T + ot) * (H * 2) + h * 2
    m_fin = tl.load(st, mask=mt, other=0.0)
    l_fin = tl.maximum(tl.load(st + 1, mask=mt, other=0.0), 1.1754944e-38)
    dq = tl.zeros([BLOCK_T, HD], tl.float32)
    kbase = k_ptr + base
    vbase = v_ptr + base
    for c0 in range(0, C, BLOCK_C):
        oc = c0 + tl.arange(0, BLOCK_C)
        mc = oc < C
        idx = tl.load(
            pos_ptr + b * T * C + ot[:, None] * C + oc[None, :],
            mask=mt[:, None] & mc[None, :],
            other=-1,
        )
        m = (idx >= 0) & mt[:, None] & mc[None, :]
        kk = tl.load(
            kbase + idx[:, :, None] * (H * HD) + hd[None, None, :], mask=m[:, :, None], other=0.0
        ).to(tl.float32)
        vv = tl.load(
            vbase + idx[:, :, None] * (H * HD) + hd[None, None, :], mask=m[:, :, None], other=0.0
        ).to(tl.float32)
        s = tl.load(
            s_ptr + b * T * H * C + ot[:, None] * (H * C) + h * C + oc[None, :],
            mask=mt[:, None] & mc[None, :],
            other=float("-inf"),
        )
        p = tl.exp(s - m_fin[:, None]) / l_fin[:, None]
        dp = tl.sum(dot[:, None, :] * vv, axis=2)
        ds = p * (dp - delta[:, None]) * scale
        dq += tl.sum(ds[:, :, None] * kk, axis=1)
        # The gather's adjoint is a scatter: every (query, column) pair adds its
        # own row into the key position it read. Atomics are the honest form of
        # "one key position, many readers" and the address pattern is the same
        # L2-resident 25 MB the forward reads.
        tl.atomic_add(
            dv_ptr + base + idx[:, :, None] * (H * HD) + hd[None, None, :],
            p[:, :, None] * dot[:, None, :],
            mask=m[:, :, None],
        )
        tl.atomic_add(
            dk_ptr + base + idx[:, :, None] * (H * HD) + hd[None, None, :],
            ds[:, :, None] * qt[:, None, :],
            mask=m[:, :, None],
        )
    tl.store(dq_ptr + base + ld, dq, mask=mt[:, None])


def _launch_bwd(q, k, v, pos, s, stats, do, out, scale):
    b, t, h, hd = q.shape
    c = pos.shape[-1]
    # fp32 gradient accumulators whatever the activation dtype: the scatter's
    # atomics add thousands of contributions per key position and a bf16
    # accumulator would swallow them. One cast at the end, back to the inputs'.
    dq = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
    dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device)
    tiles = _tiles(None)
    grid = (b * h, triton.cdiv(t, tiles["BLOCK_T"]))
    _bwd_kernel[grid](
        q,
        k,
        v,
        pos,
        s,
        stats,
        do,
        out,
        dq,
        dk,
        dv,
        t,
        h,
        c,
        scale,
        BLOCK_T=tiles["BLOCK_T"],
        BLOCK_C=tiles["BLOCK_C"],
        HD=hd,
        num_warps=tiles["NUM_WARPS"],
    )
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class _GatherAttention(torch.autograd.Function):
    """`(q, k, v, pos, scale) -> out`, forward and backward both Triton."""

    @staticmethod
    def forward(ctx, q, k, v, pos, scale):
        b, t, h, _ = q.shape
        s = torch.empty(b, t, h, pos.shape[-1], dtype=torch.float32, device=q.device)
        out, stats = _launch_fwd(q, k, v, pos, scale, s)
        ctx.save_for_backward(q, k, v, pos, s, stats, out)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, pos, s, stats, out = ctx.saved_tensors
        dq, dk, dv = _launch_bwd(q, k, v, pos, s, stats, dout.contiguous(), out, ctx.scale)
        return dq, dk, dv, None, None


def gather_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """The selected-block attention of `attend_selected_blocks`, fused.

    `q,k,v [b,t,h,hd]` contiguous CUDA; `pos [b,t,C]` int32 (-1 = invalid
    column); `scale` the softmax temperature. Returns `[b,t,h,hd]`. Differentiable
    in `q,k,v`; `pos` is an index tensor and carries no gradient.
    """
    _check(q, k, v, pos)
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        return _GatherAttention.apply(q, k, v, pos, scale)
    return _launch_fwd(q, k, v, pos, scale, None)[0]
