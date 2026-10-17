"""Triton grouped GEMMs for the dual-latent GLU expert pass — the ONE kernel
this port owns (`docs/research/2026-10-15-tsct-super-kernel.md`, op 3).

The padded arm (`experts._padded_dispatch`) gathers every kept row into ONE
zero-filled `[E, cap, d]` bank and runs the bank as three batched matmuls. Half
that bank is zeros at cap 64, the `[E, cap, f]` output is 716 MB at b1, and each
GEMM pays a bmm launch whose tiles are 64 rows tall. These kernels drop the
padding: the caller gathers the SAME rows expert-major (the port's own permute)
and hands over `offs [E+1]`, the kept-row boundaries; one program computes one
(expert, row-block) of that expert's RAGGED rows, and the GLU is fused into the
second GEMM's input through registers.

House rules (`docs/research/2026-10-15-tsct-super-kernel.md` §4):

* **pure torch in/out.** Plain tensors in, a plain tensor out; the layout
  contract is the docstring of each op and is `assert`ed.
* **no module-level mutable state.** Nothing is cached here: the caller owns the
  expert banks and the routing permute. No allocator, no graph, no workspace.
* **no host sync, ever.** `cap` is the caller's when it has one
  (`SparseMoE._dispatch` passes the capacity rule's value); with no `cap` the
  bound is the STATIC DROPLESS one — `R` rows in one group, `_dropless_cap` —
  and every kernel below EARLY-EXITS on an empty row-block, so nothing reads
  `offs` back to the host (ADR-0018 rule 2). The one read that used to live here
  (`_cap_from_offs`) is deleted: the early-exit replaced it, and a `cap` bound
  the caller cannot compute on the host is exactly what dropless is.
* **autograd correct or LOUD.** `expert_gemm_glu` is an `autograd.Function`; its
  backward runs the SAME kernels on the adjoint GEMMs (the three-GEMM grouped
  ld/scale pattern), not a second implementation of the math.
* **every fast arm has a COUNTED fallback**: `experts._dispatch` counts
  `moe_triton_gemm` when this arm runs, `moe_triton_fallback` + the reason when
  it does not.

DROPLESS (`moe_capacity = 0`, `experts.py` module docstring) rides the same
kernels through ONE addition: each program's first statement is

    if start + tl.program_id(row_block) * BLOCK_M >= end: return

with `start`/`end` read from `offs` on the device. The grid is then
`(E, cdiv(R, BLOCK_M))` — a bound the host can compute from `xg.shape[0]`, no
`offs` read — and an empty expert (or a block past its group's end) costs a
program launch, two `offs` loads and a branch, never GEMM work. With a tight
`cap` the branch fires only where a group smaller than the cap left a block
fully masked — the same work the row mask already skipped — so the capacity
mode's grids and numbers are unchanged; the static bound is simply the widest
one that is safe, which is what makes the routing independent of the window's
composition (a segmented training window is then the same function as the
one-shot one).

Layout contract (all CUDA, contiguous, `E` experts, latent `r`):

    xg      [R, d]      gathered rows, expert-major: group e owns offs[e]:offs[e+1]
    gate_u  [E, d, r]   up_u [E, d, r]
    gate_s  [E, r]      up_s [E, r]
    v       [E, r, f]
    offs    [E+1] int32 inclusive boundaries of the kept rows (offs[0] == 0,
                        offs[E] == R); every group is <= `cap` rows
    -> y    [R, f]

Tiles: measured on this card at the b1 shape by `scripts/bench_moe_kernel.py`
(see that file for the table that picked them); they are module constants on
purpose — a Triton `autotune` inside a training loop re-tunes whenever a shape
key moves, which is the treadmill `docs/research/2026-10-15-tsct-super-kernel.md`
§5.1 bans. Re-measure and edit the constants, do not add a decorator. `num_warps`
is 4 on the fused kernel and 8 on the three adjoint kernels — both measured, both
in the bench table; `BLOCK_F`/`BLOCK_N` are the ones that mattered.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: The shipped tile set, picked by the b1 bench (`scripts/bench_moe_kernel.py`).
#:   BLOCK_M  64 — one row-block per expert at cap 64, so a program never splits
#:                  a group and never touches a padding row
#:   BLOCK_K  64 — 2048/64 = 32 steps on the `x @ [d, r]` GEMMs, 64 floats = the
#:                  whole rank in one step on the `[r, f]` side
#:   BLOCK_F 128 — the f = 5461 write side, 43 steps; 128 measured fastest of
#:                  {64, 128, 256} in the sweep (4.58 vs 4.83 ms), and 256
#:                  blows the shared-memory budget at BLOCK_K 128+
#:   BLOCK_N 256 — the wide adjoint N tile (`dv`, `dx`, `expert_gemm`); 512 is
#:                  out of shared memory, 256 measured 22.0 ms against 27.1 for
#:                  64 on the b1 backward (15.9 with 8 warps)
TILES = {"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_F": 128, "BLOCK_N": 256}

assert set(TILES) == {"BLOCK_M", "BLOCK_K", "BLOCK_F", "BLOCK_N"}


def _tiles(tiles: dict | None) -> dict:
    t = dict(TILES)
    if tiles:
        unknown = set(tiles) - set(TILES)
        assert not unknown, f"unknown tile(s) {sorted(unknown)}; known: {sorted(TILES)}"
        t.update(tiles)
    return t


def _rp(r: int) -> int:
    """The power-of-two rank tile: `tl.dot` needs >= 16 on every axis."""
    return max(16, triton.next_power_of_2(r))


def _dropless_cap(rows: int) -> int:
    """The DROPLESS row-block bound: every row in ONE group.

    `R = xg.shape[0]` rows is the widest any single expert's group can be (one
    expert taking every assignment), and it is known on the host from a SHAPE —
    which is the whole point: the tight bound (`max group size`) exists only in
    `offs`, on the device, and reading it back is the sync ADR-0018 rule 2 bans
    in a step. The kernels early-exit on an empty row-block, so the blocks this
    over-estimate adds cost a program launch and two loads each, never GEMM work.
    """
    assert rows >= 1, f"a grouped GEMM needs at least one row, got {rows}"
    return int(rows)


def _prec(dt: torch.dtype) -> str:
    """fp32 must be IEEE, or the parity gate measures TF32, not this kernel."""
    return "ieee" if dt == torch.float32 else "tf32"


def _check(xg, gate_u, up_u, gate_s, up_s, v, offs) -> None:
    e, r = gate_u.shape[0], gate_u.shape[2]
    assert xg.ndim == 2 and gate_u.ndim == 3 and v.ndim == 3, "grouped GEMM wants 2-D/3-D"
    assert xg.shape[1] == gate_u.shape[1] == up_u.shape[1], "d mismatch"
    assert gate_u.shape == up_u.shape, "gate/up latents must share a shape"
    assert gate_s.shape == up_s.shape == (e, r), "scales must be [E, r]"
    assert v.shape[0] == e and v.shape[1] == r, "v must be [E, r, f]"
    assert offs.shape == (e + 1,) and offs.dtype == torch.int32, "offs must be [E+1] int32"
    for t in (xg, gate_u, up_u, gate_s, up_s, v, offs):
        assert t.is_cuda and t.is_contiguous(), "the Triton arm wants contiguous CUDA tensors"


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


@triton.jit
def _glu_kernel(
    x_ptr,
    gu_ptr,
    uu_ptr,
    gs_ptr,
    us_ptr,
    v_ptr,
    y_ptr,
    offs_ptr,
    gpre_ptr,
    upre_ptr,
    h_ptr,
    D,
    R_,
    F_,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_F: tl.constexpr,
    RP: tl.constexpr,
    PREC: tl.constexpr,
    SAVE: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """`silu(x@gate)·(x@up)` fused into the `@v` write, one program per row-block.

    SPLIT skips the `@v` phase and leaves the GLU result in `h_ptr` — the
    two-kernel variant the bench compares against the fused one.
    """
    e = tl.program_id(0)
    start = tl.load(offs_ptr + e)
    end = tl.load(offs_ptr + e + 1)
    # EARLY EXIT (the dropless bound, module docstring): with `cap = R` most
    # blocks are empty, and an empty block has nothing to compute and nothing to
    # store — two `offs` loads and a branch. With a tight `cap` it fires only
    # where a smaller group left a block fully masked.
    if start + tl.program_id(1) * BLOCK_M >= end:
        return
    rows = start + tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = rows < end
    rr = tl.arange(0, RP)
    r_mask = rr < R_
    kk = tl.arange(0, BLOCK_K)

    xp = x_ptr + rows[:, None] * D + kk[None, :]
    gp = gu_ptr + e * D * R_ + kk[:, None] * R_ + rr[None, :]
    up = uu_ptr + e * D * R_ + kk[:, None] * R_ + rr[None, :]
    g_acc = tl.zeros((BLOCK_M, RP), tl.float32)
    u_acc = tl.zeros((BLOCK_M, RP), tl.float32)
    for k in range(0, tl.cdiv(D, BLOCK_K)):
        rem = D - k * BLOCK_K
        a = tl.load(xp, mask=m_mask[:, None] & (kk[None, :] < rem), other=0.0)
        w_mask = (kk[:, None] < rem) & r_mask[None, :]
        g_acc = tl.dot(a, tl.load(gp, mask=w_mask, other=0.0), g_acc, input_precision=PREC)
        u_acc = tl.dot(a, tl.load(up, mask=w_mask, other=0.0), u_acc, input_precision=PREC)
        xp += BLOCK_K
        gp += BLOCK_K * R_
        up += BLOCK_K * R_

    gs = tl.load(gs_ptr + e * R_ + rr)
    us = tl.load(us_ptr + e * R_ + rr)
    # The padded arm's elementwise chain runs at the compute dtype on a ROUNDED
    # GEMM output: mirror that exactly (round the accumulator, multiply, round),
    # or the parity gate measures the rounding of a different program. fp32 is a
    # no-op throughout.
    g = (g_acc.to(y_ptr.dtype.element_ty) * gs[None, :]).to(y_ptr.dtype.element_ty)
    u = (u_acc.to(y_ptr.dtype.element_ty) * us[None, :]).to(y_ptr.dtype.element_ty)
    gf = g.to(tl.float32)
    silu = (tl.sigmoid(gf) * gf).to(y_ptr.dtype.element_ty)
    h = (silu * u).to(y_ptr.dtype.element_ty)
    if SAVE:
        st = m_mask[:, None] & r_mask[None, :]
        off = rows[:, None] * R_ + rr[None, :]
        tl.store(gpre_ptr + off, g_acc, mask=st)
        tl.store(upre_ptr + off, u_acc, mask=st)
        # The value phase 2 actually multiplies: the backward reuses it so the
        # adjoint is of the shipped forward, not of a rounded-away cousin.
        tl.store(h_ptr + off, h, mask=st)

    if not SPLIT:
        for n0 in range(0, tl.cdiv(F_, BLOCK_F)):
            ff = n0 * BLOCK_F + tl.arange(0, BLOCK_F)
            f_mask = ff < F_
            vv = tl.load(
                v_ptr + e * R_ * F_ + rr[:, None] * F_ + ff[None, :],
                mask=r_mask[:, None] & f_mask[None, :],
                other=0.0,
            )
            o = tl.dot(h, vv, input_precision=PREC)
            tl.store(
                y_ptr + rows[:, None] * F_ + ff[None, :],
                o.to(y_ptr.dtype.element_ty),
                mask=m_mask[:, None] & f_mask[None, :],
            )


@triton.jit
def _grouped_gemm_kernel(
    a_ptr,
    w_ptr,
    o_ptr,
    offs_ptr,
    N,
    K,
    sam,
    sak,
    swe,
    swk,
    swn,
    som,
    son,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TRANS_W: tl.constexpr,
    PREC: tl.constexpr,
):
    """`a @ w[e]` per group, ragged rows from `offs`.

    TRANS_W reads `w` as `[E, N, K]` instead of `[E, K, N]` — that is the
    adjoint GEMM (`dy @ vᵀ`, `dg @ gate_uᵀ`) without materializing a transpose.
    """
    e = tl.program_id(0)
    start = tl.load(offs_ptr + e)
    end = tl.load(offs_ptr + e + 1)
    # EARLY EXIT: same contract as `_glu_kernel` (the dropless static bound).
    if start + tl.program_id(1) * BLOCK_M >= end:
        return
    rows = start + tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = rows < end
    nn = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = nn < N
    kk = tl.arange(0, BLOCK_K)
    ap = a_ptr + rows[:, None] * sam + kk[None, :] * sak
    wp = w_ptr + e * swe + kk[:, None] * swk + nn[None, :] * swn
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        rem = K - k * BLOCK_K
        k_mask = kk < rem
        a = tl.load(ap, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(wp, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, w, acc, input_precision=PREC)
        ap += BLOCK_K * sak
        wp += BLOCK_K * swk
    tl.store(
        o_ptr + rows[:, None] * som + nn[None, :] * son,
        acc.to(o_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _wgrad_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    offs_ptr,
    N,
    K,
    sap,
    sak,
    sbp,
    sbk,
    soe,
    sok,
    son,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PREC: tl.constexpr,
):
    """`a[group]ᵀ @ b[group] -> out[e]` — the per-expert weight gradient.

    One program per (expert, K-tile, N-tile); the group's rows are the reduction
    axis. An expert with no kept rows stores zeros, which is its correct
    gradient (it received none).
    """
    e = tl.program_id(0)
    start = tl.load(offs_ptr + e)
    end = tl.load(offs_ptr + e + 1)
    kk = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    nn = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    k_mask = kk < K
    n_mask = nn < N
    acc = tl.zeros((BLOCK_K, BLOCK_N), tl.float32)
    for i in range(0, tl.cdiv(end - start, BLOCK_M)):
        rows = start + i * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = rows < end
        a = tl.load(
            a_ptr + rows[:, None] * sap + kk[None, :] * sak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptr + rows[:, None] * sbp + nn[None, :] * sbk,
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(tl.trans(a), b, acc, input_precision=PREC)
    tl.store(
        o_ptr + e * soe + kk[:, None] * sok + nn[None, :] * son,
        acc.to(o_ptr.dtype.element_ty),
        mask=k_mask[:, None] & n_mask[None, :],
    )


# --------------------------------------------------------------------------
# forward
# --------------------------------------------------------------------------


def _glu_fused_or_head(xg, gate_u, up_u, gate_s, up_s, v, offs, cap, tiles, split, save):
    """Shared launcher: returns `(y, gpre, upre, h)`; `gpre/upre/h` are None if not saved."""
    e, d, r = gate_u.shape
    f = v.shape[2]
    R = xg.shape[0]
    dt = xg.dtype
    rows = max(1, triton.cdiv(cap, tiles["BLOCK_M"]))
    save = save or split  # the SPLIT arm's whole output IS the saved activation
    y = torch.empty(R, 0 if split else f, dtype=dt, device=xg.device)
    bufs = (None, None, None)
    gpre = upre = h = None
    y_ptr = y
    if save:
        gpre = torch.empty(R, r, dtype=torch.float32, device=xg.device)
        upre = torch.empty(R, r, dtype=torch.float32, device=xg.device)
        h = torch.empty(R, r, dtype=dt, device=xg.device)
        bufs = (gpre, upre, h)
    if split:
        y_ptr = h  # the SPLIT arm leaves the GLU result here; y is never written
    _glu_kernel[(e, rows)](
        xg,
        gate_u,
        up_u,
        gate_s,
        up_s,
        v,
        y_ptr,
        offs,
        bufs[0] if bufs[0] is not None else xg,
        bufs[1] if bufs[1] is not None else xg,
        bufs[2] if bufs[2] is not None else xg,
        d,
        r,
        f,
        BLOCK_M=tiles["BLOCK_M"],
        BLOCK_K=tiles["BLOCK_K"],
        BLOCK_F=tiles["BLOCK_F"],
        RP=_rp(r),
        PREC=_prec(dt),
        SAVE=save,
        SPLIT=split,
        num_warps=4,  # 4 beat 8 on the fused kernel (4.58 vs 5.1 ms, b1 sweep)
    )
    return y, bufs


def expert_gemm_glu(
    xg: torch.Tensor,
    gate_u: torch.Tensor,
    up_u: torch.Tensor,
    gate_s: torch.Tensor,
    up_s: torch.Tensor,
    v: torch.Tensor,
    offs: torch.Tensor,
    *,
    cap: int | None = None,
    tiles: dict | None = None,
) -> torch.Tensor:
    """`y [R, f]` = `(silu((xg@gate_u)·gate_s) ⊙ ((xg@up_u)·up_s)) @ v`, grouped.

    `offs [E+1]` int32 are the kept-row boundaries; the gathers are the caller's
    (`experts._permute`). Rows are never padded: an expert with `n_e` rows costs
    `ceil(n_e / BLOCK_M)` row-blocks, and an EMPTY expert costs one program that
    returns on its first branch. Autograd is supported (see `_ExpertGemmGlu`).
    `cap` bounds the row-block grid: pass the capacity rule's value when the
    caller has one (the tight grid), or `None` for the DROPLESS static bound
    (`_dropless_cap`, no host read of `offs`).
    """
    _check(xg, gate_u, up_u, gate_s, up_s, v, offs)
    if cap is None:
        cap = _dropless_cap(xg.shape[0])
    assert cap >= 1, f"cap={cap} must be >= 1"
    return _ExpertGemmGlu.apply(xg, gate_u, up_u, gate_s, up_s, v, offs, int(cap), _tiles(tiles))


def expert_gemm(
    h: torch.Tensor,
    v: torch.Tensor,
    offs: torch.Tensor,
    *,
    cap: int | None = None,
    tiles: dict | None = None,
) -> torch.Tensor:
    """`y [R, f]` = `h @ v[e]` per group — the second half of the split arm.

    No silu and no autograd: it is the seam the bench compares against the fused
    kernel, and the shape the backward's `dy @ vᵀ` runs on.
    """
    e, r, f = v.shape
    assert h.ndim == 2 and h.shape[1] == r and h.is_cuda and h.is_contiguous()
    assert offs.shape == (e + 1,) and offs.dtype == torch.int32
    if cap is None:
        cap = _dropless_cap(h.shape[0])
    t = _tiles(tiles)
    y = torch.empty(h.shape[0], f, dtype=h.dtype, device=h.device)
    _grouped_gemm_kernel[_grid(e, cap, f, t["BLOCK_M"], t["BLOCK_N"])](
        h,
        v,
        y,
        offs,
        f,
        r,
        h.stride(0),
        h.stride(1),
        r * f,
        f,
        1,
        y.stride(0),
        1,
        BLOCK_M=t["BLOCK_M"],
        BLOCK_N=t["BLOCK_N"],
        BLOCK_K=_rp(r),
        TRANS_W=False,
        PREC=_prec(h.dtype),
        num_warps=8,
    )
    return y


# --------------------------------------------------------------------------
# backward
# --------------------------------------------------------------------------


def _segments(offs: torch.Tensor) -> torch.Tensor:
    """`[R]` int64 expert id of every kept row (the inverse of `offs`)."""
    e = offs.shape[0] - 1
    counts = (offs[1:] - offs[:-1]).to(torch.int64)
    return torch.repeat_interleave(torch.arange(e, device=offs.device), counts)


def _seg_sum(a: torch.Tensor, seg: torch.Tensor, e: int) -> torch.Tensor:
    """`out[e] = a[rows of e].sum(0)` — the `[E, r]` scale gradients."""
    out = torch.zeros(e, a.shape[1], dtype=torch.float32, device=a.device)
    return out.index_add_(0, seg, a)


def _grid(e: int, cap: int, n: int, block_m: int, block_n: int) -> tuple[int, int, int]:
    """`(experts, ragged row-blocks, N-blocks)` — the shared launch shape."""
    return (e, max(1, triton.cdiv(cap, block_m)), triton.cdiv(n, block_n))


def _grouped_nt(a, w, offs, cap, n, k, tiles, out_dtype):
    """`a [R, K] @ w[e]ᵀ` with `w` stored `[E, N, K]` — the adjoint GEMM."""
    e = w.shape[0]
    out = torch.empty(a.shape[0], n, dtype=out_dtype, device=a.device)
    _grouped_gemm_kernel[_grid(e, cap, n, tiles["BLOCK_M"], tiles["BLOCK_N"])](
        a,
        w,
        out,
        offs,
        n,
        k,
        a.stride(0),
        a.stride(1),
        w.stride(0),
        w.stride(2),
        w.stride(1),
        out.stride(0),
        1,
        BLOCK_M=tiles["BLOCK_M"],
        BLOCK_N=tiles["BLOCK_N"],
        BLOCK_K=tiles["BLOCK_K"],
        TRANS_W=True,
        PREC=_prec(a.dtype),
        num_warps=8,
    )
    return out


def _wgrad(a, b, offs, e, cap, k, n, tiles, out_dtype):
    """`out [E, K, N]` = `a[group]ᵀ @ b[group]`.

    `num_stages=2` is load-bearing, not taste: the row-loop body stages a
    `[BLOCK_M, BLOCK_K]` and a `[BLOCK_M, BLOCK_N]` tile, and at `BLOCK_N=256`
    three stages want 120 KB of the 99 KB of shared memory this card has
    (`sm_120`). Two stages fit at every rank from 16 to 128.
    """
    out = torch.empty(e, k, n, dtype=out_dtype, device=a.device)
    _wgrad_kernel[(e, triton.cdiv(k, tiles["BLOCK_K"]), triton.cdiv(n, tiles["BLOCK_N"]))](
        a,
        b,
        out,
        offs,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        k * n,
        n,
        1,
        BLOCK_M=tiles["BLOCK_M"],
        BLOCK_K=tiles["BLOCK_K"],
        BLOCK_N=tiles["BLOCK_N"],
        PREC=_prec(a.dtype),
        num_warps=8,
        num_stages=2,
    )
    return out


def _glu_backward(xg, gate_u, up_u, gate_s, up_s, v, offs, cap, gpre, upre, h, dy, tiles):
    """The three-GEMM grouped adjoint, all of it on this module's own kernels."""
    e, d, r = gate_u.shape
    f = v.shape[2]
    rp = _rp(r)
    assert gpre is not None and upre is not None and h is not None, "no saved activations"
    dyc = dy.to(xg.dtype).contiguous()
    seg = _segments(offs)
    # Tile rule: the rank axis always takes the rank tile, the wide axes (`f` and
    # `d`) the configured BLOCK_N. `dh` is `[R, f] @ [f, r]` and `dgate_u` is
    # `[d, R] @ [R, r]` — both rank-bounded in N; `dv`/`dx` are rank-bounded in K
    # and want the wide N so each program does real work instead of one mma tile.
    t_rank_n = dict(tiles, BLOCK_N=rp)
    t_wide = dict(tiles, BLOCK_K=rp)

    dh = _grouped_nt(
        dyc, v, offs, cap, r, f, dict(t_rank_n, BLOCK_K=tiles["BLOCK_K"]), torch.float32
    )
    # Re-round the gate exactly as the forward did (accumulator -> dtype, multiply
    # the scale, round): the padded arm's autograd differentiates ITS rounded gate,
    # so differentiating the unrounded one here would be a 1e-2 disagreement that
    # is not a bug but reads like one. `.to(dt)` is a no-op at fp32.
    dt = xg.dtype
    g = (gpre.to(dt) * gate_s[seg]).to(torch.float32)
    u = (upre.to(dt) * up_s[seg]).to(torch.float32)
    sig = torch.sigmoid(g)
    dsilu = sig * (1.0 + g * (1.0 - sig))
    dg = dh * u * dsilu  # dL/dg, the post-scale gate
    du = dh * (sig * g)  # dL/du, the post-scale up
    dg_pre = dg * gate_s[seg]
    du_pre = du * up_s[seg]
    dg_l, du_l = dg_pre.to(xg.dtype), du_pre.to(xg.dtype)

    dgate_u = _wgrad(xg, dg_l, offs, e, cap, d, r, t_rank_n, xg.dtype)
    dup_u = _wgrad(xg, du_l, offs, e, cap, d, r, t_rank_n, xg.dtype)
    dv = _wgrad(h, dyc, offs, e, cap, r, f, t_wide, xg.dtype)
    dx = _grouped_nt(dg_l, gate_u, offs, cap, d, r, t_wide, xg.dtype)
    dx = dx.add_(_grouped_nt(du_l, up_u, offs, cap, d, r, t_wide, xg.dtype))

    dgate_s = _seg_sum(dg * gpre, seg, e)
    dup_s = _seg_sum(du * upre, seg, e)
    return dx, dgate_u, dup_u, dgate_s, dup_s, dv


class _ExpertGemmGlu(torch.autograd.Function):
    """The fused pass with its own backward: `fwd` Triton, `bwd` Triton (adjoint GEMMs)."""

    @staticmethod
    def forward(ctx, xg, gate_u, up_u, gate_s, up_s, v, offs, cap, tiles):
        y, (gpre, upre, h) = _glu_fused_or_head(
            xg, gate_u, up_u, gate_s, up_s, v, offs, cap, tiles, split=False, save=True
        )
        ctx.save_for_backward(xg, gate_u, up_u, gate_s, up_s, v, offs, gpre, upre, h)
        ctx.cap, ctx.tiles = cap, tiles
        return y

    @staticmethod
    def backward(ctx, dy):
        xg, gate_u, up_u, gate_s, up_s, v, offs, gpre, upre, h = ctx.saved_tensors
        return (
            *_glu_backward(
                xg, gate_u, up_u, gate_s, up_s, v, offs, ctx.cap, gpre, upre, h, dy, ctx.tiles
            ),
            None,
            None,
            None,
        )
