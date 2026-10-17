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

BACKWARD (`_glu_backward_fused`, the default; `_glu_backward` stays as the
reference beside it). Three changes, each reimplementing an idea of OLMo-core's
`kernels/cuda/moe_unpermute_bwd_kernel.cu` in this module's own idiom:

* the COMBINE backward is ONE kernel (`_combine_bwd_kernel`, `_Combine`): per
  token it reads `grad_out` once, loops the k selected rows, writes the scaled
  contribution and accumulates `⟨grad_out, y_row⟩` in registers — one store of
  `[R]` where autograd ran an `index_add` backward, a `mul` backward and a
  `[R, f]` re-read plus reduction (their `:127-226`, the register `accum[k]` and
  the `__shfl_xor_sync` reduction; ours is a `tl.sum` over the tile, the Triton
  equivalent of the warp butterfly);
* `dx` is ONE grouped GEMM with two accumulators (`_grouped_gemm_pair_kernel`),
  the forward's own two-`tl.dot` pattern (`:193-194`) applied to the adjoint —
  not two GEMMs plus `add_`;
* the row→expert map is `int32` and arrives from the caller (`seg`, built once in
  `experts._triton_dispatch` from the routing the permute already produced)
  instead of a `torch.repeat_interleave` rebuild in every backward; the combine
  kernel's token→row map is the same idea's `-1` sentinel (their `:177-179`).

The scratch is a `BwdWorkspace` the CALLER owns (`experts.SparseMoE`), not this
module: the "no module-level mutable state" rule stands — there is no global
instance and this file keeps no state of its own. Every call the fused backward
cannot serve falls back to `_glu_backward` COUNTED (`moe_triton_fallback` + the
reason on the counter object).

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
#:   BLOCK_N 128 — the wide adjoint N tile (`dv`, `dx`, `expert_gemm`,
#:                  `_grouped_nt_pair`); 512 is out of shared memory, and 256 is
#:                  TOO (battery 2026-10-17: `_grouped_nt_pair` at the b1 shape
#:                  needs 163840 B against this card's 101376 B, so the fused
#:                  backward crashed at launch on `(2048, 64)`), so 128 is the
#:                  shipped N. **The battery's b1 fwd+bwd row for 128 (58.15
#:                  fused / 67.22 previous adjoint / 67.41 padded) was measured
#:                  with a POISONED bank cache** — `experts._banks` handed the
#:                  backward constant banks, so that backward never computed an
#:                  expert bank gradient (the defect `tests/test_expert_banks.py`
#:                  now gates). Re-measured on the same bench after that fix:
#:                  fused 202-240 ms, previous adjoint 211-256, padded 215-253 —
#:                  the fused adjoint wins by ~4-6%, not 13%, and reverting
#:                  `_banks` alone reproduces the battery's row exactly
#:                  (55.96/65.89/65.19). Quote the fwd rows (25.6/26.5/28.8 ms),
#:                  which the cache defect does not touch.
#:   BLOCK_T  16 — the combine backward's TOKEN tile (one program owns
#:                  BLOCK_T tokens and their whole row of `f`); 16 keeps the
#:                  per-token row map load amortised without spilling the
#:                  `[BLOCK_T, TOP_K]` register accumulator; an 8/16/32 sweep was
#:                  flat (1.4% spread), so 16 stands
#: `BLOCK_T` is the one tile here NOT picked by a sweep — the combine kernel was
#: written while the card was held by the live b1 run, so 16 is the starting
#: value `scripts/bench_moe_kernel.py` is set up to replace (same rule as the
#: rest: re-measure and edit the constant, never add an autotune).
TILES = {"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_F": 128, "BLOCK_N": 128, "BLOCK_T": 16}

assert set(TILES) == {"BLOCK_M", "BLOCK_K", "BLOCK_F", "BLOCK_N", "BLOCK_T"}

#: The widest top-k the fused combine backward carries: the kernel unrolls the
#: top-k loop (`tl.static_range`), so a top-k past this is code it does not have.
#: `experts.SparseMoE._fused_bwd_reason` is what counts the fallback to the
#: current backward when a config asks for more (our own top-k is 2).
FUSED_TOPK_MAX = 32

#: The expert count at/above which the GLU forward runs the FUSED kernel (`h` kept
#: in registers, `[R, f]` written in one launch) rather than the SPLIT pair (the
#: GLU head writing `[R, r]`, then `expert_gemm`). Both are the same function —
#: the backward is identical and takes `h` from the save buffer either way — so
#: the choice is pure speed, and the b1 bench measured the two ENDPOINTS
#: (2026-10-17): at **E=8** the split arm is **0.184 ms against the fused 0.375**
#: (2x — `h` is 1/128th of the work there and both are launch-bound, so the second
#: kernel costs less than the fused kernel's wider tile), at **E=1024/b1** the
#: fused arm is **4.699 against 6.335** (1.35x — the `[R, r]` round trip through
#: global memory is the bill). Nothing between them was swept, so this is a SWEPT
#: CONSTANT and not an interpolated crossing: re-measure with
#: `scripts/bench_moe_kernel.py` before moving it, and see `glu_arm` for the
#: caller (`_ExpertGemmGlu.forward`, i.e. the shipped forward).
FUSED_GLU_MIN_E = 64


def glu_arm(e: int) -> str:
    """`"fused"` or `"split"` — the GLU forward arm for an `e`-expert bank."""
    assert e >= 1, f"an expert bank needs at least one expert, got {e}"
    return "fused" if e >= FUSED_GLU_MIN_E else "split"


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


def _topk_tile(k: int) -> int:
    """The top-k tile for the combine backward: `tl.arange` wants a power of two.

    Floored at 2 (never `tl.arange(0, 1)`), and the kernel's `if k < K` static
    branch is what skips a padded slot, so a top-k of 3 is served by a tile of 4
    rather than refused.
    """
    return max(2, triton.next_power_of_2(k))


class BwdWorkspace:
    """Grow-only scratch for the fused backward — OWNED BY THE CALLER.

    OLMo-core's `_PermuteDropWorkspace` (`kernels/moe_permute_drop.py:11-70`,
    grow-only: `if ws.max_expanded_rows >= expanded_rows: return ws`) in this
    module's own idiom. The rule it respects is the module docstring's: nothing
    is cached HERE and there is no global instance — the owner is the module
    object that calls in (`experts.SparseMoE`, one workspace per `SparseMoE`), so
    two models, two devices or two callers never share a buffer.

    A slot is reallocated only when it must GROW (a bigger `R`), or when the
    dtype/device moves; a smaller call reuses the bigger buffer through a view,
    so the steady-state backward allocates no `[R, ·]` scratch at all.

    `ponytail:` one slot set per workspace, not per stream — a second CUDA stream
    running a backward concurrently with this one would race on the buffers.
    Nothing in this port runs multi-stream; key on the stream if it ever does.
    """

    __slots__ = ("_flat",)

    def __init__(self) -> None:
        self._flat: dict[str, torch.Tensor] = {}

    def slot(
        self, name: str, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """The `[shape]` `dtype` buffer named `name`, grown in place when needed."""
        n = 1
        for s in shape:
            n *= int(s)
        t = self._flat.get(name)
        if t is None or t.numel() < n or t.dtype != dtype or t.device != device:
            t = torch.empty(n, dtype=dtype, device=device)
            self._flat[name] = t
        return t[:n].view(shape)


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


@triton.jit
def _combine_bwd_kernel(
    dout_ptr,
    y_ptr,
    w_ptr,
    rowmap_ptr,
    dy_ptr,
    dw_ptr,
    N,
    F_,
    K: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """The combine's backward in ONE pass: `dy` scatter + `dw` reduction.

    `out[t] = Σ_k y[row(t, k)]·w[row(t, k)]`, so its backward is

        dy[row(t,k)] = grad_out[t] · w[row(t,k)]          (the expert's adjoint)
        dw[row(t,k)] = ⟨grad_out[t], y[row(t,k)]⟩         (the router-weight grad)

    and both fall out of the SAME read of `grad_out`: the k rows of a token are
    one gather each, so reading `grad_out` once per token and looping k costs
    `n·f` reads instead of `n·k·f`. `dw` is accumulated in registers (`acc`, one
    column per top-k slot) and stored once at the end — the `[R, f]` re-read and
    the f-reduction autograd ran between the `index_add` backward and the `mul`
    backward do not exist here.

    Inspiration, not a transcription: OLMo-core's
    `kernels/cuda/moe_unpermute_bwd_kernel.cu:127-226` (one block per token, the
    `float accum[topk_tile]` register array `:195-201`, the `__shfl_xor_sync`
    butterfly `:206-215`, `prob_grad` written by `tid == 0` `:217-224`). Triton
    has no warp shuffle in this position — `tl.sum(go · y_row, axis=1)` IS the
    reduction, one store per row, no separate writer thread.

    `rowmap [N, K]` int32 is the token→gathered-row map; `-1` marks a row the
    capacity dropped (their `dest_row < 0`, `:177-179`), and such a lane stores
    nothing and contributes nothing.
    """
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < N
    kk = tl.arange(0, TOP_K)
    acc = tl.zeros((BLOCK_T, TOP_K), tl.float32)
    for f0 in range(0, tl.cdiv(F_, BLOCK_F)):
        ff = f0 * BLOCK_F + tl.arange(0, BLOCK_F)
        fmask = ff < F_
        go = tl.load(
            dout_ptr + t[:, None] * F_ + ff[None, :],
            mask=tmask[:, None] & fmask[None, :],
            other=0.0,
        )
        for k in tl.static_range(TOP_K):
            if k < K:  # static: a padded top-k slot costs no instruction
                row = tl.load(rowmap_ptr + t * K + k, mask=tmask, other=-1)
                valid = row >= 0
                wv = tl.load(w_ptr + row, mask=valid, other=0.0)
                ymask = valid[:, None] & fmask[None, :]
                tl.store(
                    dy_ptr + row[:, None] * F_ + ff[None, :],
                    (go * wv[:, None]).to(dy_ptr.dtype.element_ty),
                    mask=ymask,
                )
                yv = tl.load(y_ptr + row[:, None] * F_ + ff[None, :], mask=ymask, other=0.0)
                contrib = tl.sum(go * yv.to(tl.float32), axis=1)
                acc += tl.where(kk[None, :] == k, contrib[:, None], 0.0)
    for k in tl.static_range(TOP_K):
        if k < K:
            row = tl.load(rowmap_ptr + t * K + k, mask=tmask, other=-1)
            dwv = tl.sum(tl.where(kk[None, :] == k, acc, 0.0), axis=1)
            tl.store(dw_ptr + row, dwv, mask=(row >= 0) & tmask)


@triton.jit
def _grouped_gemm_pair_kernel(
    a1_ptr,
    w1_ptr,
    a2_ptr,
    w2_ptr,
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
    PREC: tl.constexpr,
):
    """`a1 @ w1[e]ᵀ + a2 @ w2[e]ᵀ` into ONE accumulator, ONE store — the fused `dx`.

    `_grouped_gemm_kernel`'s adjoint with a second (a, w) pair folded in, which is
    the pattern the FORWARD already runs: `_glu_kernel` accumulates both `tl.dot`s
    of the dual latent in registers (`:193-194`) and combines them at the latent.
    The backward of that pass had been two grouped GEMMs plus `add_`, i.e. the
    same tiles computed twice and one extra `[R, d]` round trip.

    Both `a`s share shape and strides, both `w`s too (contiguous `[R, K]` and
    `[E, N, K]`), so the launcher passes one stride set for the two pairs.
    Accumulating the second dot INTO the first's fp32 accumulator is what makes
    this equal to `dot1 + dot2` rather than an approximation of it: the products
    are summed in the same order, only the accumulator's starting value differs.
    """
    e = tl.program_id(0)
    start = tl.load(offs_ptr + e)
    end = tl.load(offs_ptr + e + 1)
    # EARLY EXIT: the dropless static bound, same contract as every kernel here.
    if start + tl.program_id(1) * BLOCK_M >= end:
        return
    rows = start + tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = rows < end
    nn = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = nn < N
    kk = tl.arange(0, BLOCK_K)
    a1p = a1_ptr + rows[:, None] * sam + kk[None, :] * sak
    a2p = a2_ptr + rows[:, None] * sam + kk[None, :] * sak
    w1p = w1_ptr + e * swe + kk[:, None] * swk + nn[None, :] * swn
    w2p = w2_ptr + e * swe + kk[:, None] * swk + nn[None, :] * swn
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        rem = K - k * BLOCK_K
        k_mask = kk < rem
        am = m_mask[:, None] & k_mask[None, :]
        wm = k_mask[:, None] & n_mask[None, :]
        acc = tl.dot(
            tl.load(a1p, mask=am, other=0.0),
            tl.load(w1p, mask=wm, other=0.0),
            acc,
            input_precision=PREC,
        )
        acc = tl.dot(
            tl.load(a2p, mask=am, other=0.0),
            tl.load(w2p, mask=wm, other=0.0),
            acc,
            input_precision=PREC,
        )
        a1p += BLOCK_K * sak
        a2p += BLOCK_K * sak
        w1p += BLOCK_K * swk
        w2p += BLOCK_K * swk
    tl.store(
        o_ptr + rows[:, None] * som + nn[None, :] * son,
        acc.to(o_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
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
    fused_bwd: bool = True,
    seg: torch.Tensor | None = None,
    ws: BwdWorkspace | None = None,
) -> torch.Tensor:
    """`y [R, f]` = `(silu((xg@gate_u)·gate_s) ⊙ ((xg@up_u)·up_s)) @ v`, grouped.

    `offs [E+1]` int32 are the kept-row boundaries; the gathers are the caller's
    (`experts._permute`). Rows are never padded: an expert with `n_e` rows costs
    `ceil(n_e / BLOCK_M)` row-blocks, and an EMPTY expert costs one program that
    returns on its first branch. Autograd is supported (see `_ExpertGemmGlu`).
    `cap` bounds the row-block grid: pass the capacity rule's value when the
    caller has one (the tight grid), or `None` for the DROPLESS static bound
    (`_dropless_cap`, no host read of `offs`).

    `fused_bwd=True` (the default) runs `_glu_backward_fused`: pass the caller's
    `seg` (`[R]` int32 row→expert map — the routing the permute already had) and
    `ws` (the caller's `BwdWorkspace`) and the backward neither rebuilds the
    segments nor allocates its `[R, ·]` scratch. `seg=None` falls back to
    `_segments(offs)` and `ws=None` to a throwaway workspace: correct, and the
    reason a standalone call is still one line. `fused_bwd=False` is the
    reference adjoint, kept as the parity gate and the bench's control.
    """
    _check(xg, gate_u, up_u, gate_s, up_s, v, offs)
    if cap is None:
        cap = _dropless_cap(xg.shape[0])
    assert cap >= 1, f"cap={cap} must be >= 1"
    if seg is not None:
        assert seg.shape == (xg.shape[0],) and seg.dtype == torch.int32, (
            "seg must be the [R] int32 row→expert map"
        )
    if fused_bwd and ws is None:
        ws = BwdWorkspace()
    return _ExpertGemmGlu.apply(
        xg,
        gate_u,
        up_u,
        gate_s,
        up_s,
        v,
        offs,
        int(cap),
        _tiles(tiles),
        bool(fused_bwd),
        seg,
        ws,
    )


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


def _grouped_nt(a, w, offs, cap, n, k, tiles, out_dtype, out=None):
    """`a [R, K] @ w[e]ᵀ` with `w` stored `[E, N, K]` — the adjoint GEMM.

    `out` is the caller-owned grow-only destination (`BwdWorkspace.slot`), the
    `out=` idiom of `moe_unpermute_bwd_kernel.cu:16-46`; omitted, the tensor is
    allocated here as it always was.
    """
    e = w.shape[0]
    if out is None:
        out = torch.empty(a.shape[0], n, dtype=out_dtype, device=a.device)
    else:
        assert out.shape == (a.shape[0], n), f"out {tuple(out.shape)} != [{a.shape[0]}, {n}]"
        assert out.dtype == out_dtype and out.is_contiguous(), "out must be the caller's buffer"
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


def _grouped_nt_pair(a1, w1, a2, w2, offs, cap, n, k, tiles, out_dtype):
    """`a1 @ w1[e]ᵀ + a2 @ w2[e]ᵀ` — `_grouped_nt` twice, in ONE kernel.

    The fused `dx` (module docstring). The two pairs must agree on shape AND
    strides — the kernel carries one stride set — which is the caller's
    invariant here: both are the `[R, r]` post-scale adjoints and the `[E, d, r]`
    banks the forward used.
    """
    e = w1.shape[0]
    assert a1.shape == a2.shape, "the fused dx pairs must share a shape"
    assert w1.shape == w2.shape, "the fused dx weight pairs must share a shape"
    assert a1.stride() == a2.stride() and w1.stride() == w2.stride(), "shared strides"
    assert a1.is_contiguous() and a2.is_contiguous() and w1.is_contiguous() and w2.is_contiguous()
    out = torch.empty(a1.shape[0], n, dtype=out_dtype, device=a1.device)
    _grouped_gemm_pair_kernel[_grid(e, cap, n, tiles["BLOCK_M"], tiles["BLOCK_N"])](
        a1,
        w1,
        a2,
        w2,
        out,
        offs,
        n,
        k,
        a1.stride(0),
        a1.stride(1),
        w1.stride(0),
        w1.stride(2),
        w1.stride(1),
        out.stride(0),
        1,
        BLOCK_M=tiles["BLOCK_M"],
        BLOCK_N=tiles["BLOCK_N"],
        BLOCK_K=tiles["BLOCK_K"],
        PREC=_prec(a1.dtype),
        num_warps=8,
    )
    return out


def _wgrad(a, b, offs, e, cap, k, n, tiles, out_dtype):
    """`out [E, K, N]` = `a[group]ᵀ @ b[group]`.

    `num_stages=2` is load-bearing, not taste: the row-loop body stages a
    `[BLOCK_M, BLOCK_K]` and a `[BLOCK_M, BLOCK_N]` tile, and at `BLOCK_N=256`
    three stages want 120 KB of the 99 KB of shared memory this card has
    (`sm_120`) — which is also the tile size that made `_grouped_nt_pair` fail to
    launch at the b1 shape, so 128 is the shipped N and two stages fit at every
    rank from 16 to 128.
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
    """The three-GEMM grouped adjoint, all of it on this module's own kernels.

    The REFERENCE backward (`fused_bwd=False`): two `dx` GEMMs plus `add_`, a
    `repeat_interleave` segment rebuild, and one allocation per temporary. Kept
    because it is what the fused adjoint is compared against — and because it is
    the arm an unsupported call falls back to.
    """
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


def _glu_backward_fused(
    xg, gate_u, up_u, gate_s, up_s, v, offs, cap, gpre, upre, h, dy, tiles, seg, ws
):
    """The SAME adjoint with the three OLMo-core ideas (module docstring) in it.

    The arithmetic and the OP ORDER are `_glu_backward`'s, op for op — only the
    buffers and two kernels differ. That is deliberate: the parity gate compares
    this against the padded arm's autograd at 1e-4 (fp32), and a reordered
    elementwise chain would show up there as a wiring error.

    What is different:

    * `dx` is `_grouped_nt_pair` — ONE kernel, one accumulator, one `[R, d]`
      store, instead of two `_grouped_nt` calls plus `add_`;
    * `seg` is the caller's `int32` row→expert map (built from the routing the
      permute already had) instead of a `repeat_interleave` rebuild;
    * every `[R, r]` temporary is a slot of the caller-owned grow-only `ws`.

    The OUTPUTS (`dx`, `dgate_u`, `dup_u`, `dv`, `dgate_s`, `dup_s`) are fresh
    allocations: autograd owns them, so they cannot be reused. `dy` and `dw` —
    the combine backward's own outputs — are the one pair of GRADS that live in
    the workspace; the engine hands `dy` straight to this function and drops it
    after, so nothing outlives the single backward that owns the workspace.
    """
    e, d, r = gate_u.shape
    f = v.shape[2]
    R = xg.shape[0]
    rp = _rp(r)
    assert gpre is not None and upre is not None and h is not None, "no saved activations"
    assert ws is not None, "the fused backward needs a workspace (BwdWorkspace)"
    dt, dev = xg.dtype, xg.device
    if seg is None:
        seg = _segments(offs).to(torch.int32)
    assert seg.shape == (R,) and seg.dtype == torch.int32, "seg must be the [R] int32 row map"
    dyc = dy.to(dt).contiguous()
    t_rank_n = dict(tiles, BLOCK_N=rp)
    t_wide = dict(tiles, BLOCK_K=rp)

    dh = _grouped_nt(
        dyc,
        v,
        offs,
        cap,
        r,
        f,
        dict(t_rank_n, BLOCK_K=tiles["BLOCK_K"]),
        torch.float32,
        out=ws.slot("dh", (R, r), torch.float32, dev),
    )
    # The re-rounding chain, `_glu_backward`'s order verbatim (round to the
    # compute dtype, multiply the scale, widen). Each step writes into a slot
    # rather than a fresh tensor, so a `torch.mul(..., out=)` with an operand
    # that is already the right dtype costs no allocation at all.
    gs_row = torch.index_select(gate_s, 0, seg)
    us_row = torch.index_select(up_s, 0, seg)
    gb = ws.slot("gb", (R, r), dt, dev)
    torch.mul(gpre.to(dt), gs_row, out=gb)
    ub = ws.slot("ub", (R, r), dt, dev)
    torch.mul(upre.to(dt), us_row, out=ub)
    g = ws.slot("g", (R, r), torch.float32, dev)
    g.copy_(gb)
    u = ws.slot("u", (R, r), torch.float32, dev)
    u.copy_(ub)
    sig = ws.slot("sig", (R, r), torch.float32, dev)
    torch.sigmoid(g, out=sig)
    tmp = ws.slot("tmp", (R, r), torch.float32, dev)
    dsilu = ws.slot("dsilu", (R, r), torch.float32, dev)
    torch.sub(1.0, sig, out=dsilu)  # 1 - sig
    torch.mul(g, dsilu, out=tmp)  # g·(1 - sig)
    torch.add(tmp, 1.0, out=tmp)  # 1 + g·(1 - sig)
    torch.mul(sig, tmp, out=dsilu)  # dsilu = sig·(1 + g·(1 - sig))
    dg = ws.slot("dg", (R, r), torch.float32, dev)
    torch.mul(dh, u, out=dg)
    torch.mul(dg, dsilu, out=dg)  # dg = dh·u·dsilu, left-to-right like the reference
    du = ws.slot("du", (R, r), torch.float32, dev)
    torch.mul(sig, g, out=du)
    torch.mul(dh, du, out=du)  # du = dh·(sig·g)
    dg_l = ws.slot("dg_l", (R, r), dt, dev)
    torch.mul(dg, gs_row, out=tmp)
    dg_l.copy_(tmp)
    du_l = ws.slot("du_l", (R, r), dt, dev)
    torch.mul(du, us_row, out=tmp)
    du_l.copy_(tmp)

    dgate_u = _wgrad(xg, dg_l, offs, e, cap, d, r, t_rank_n, xg.dtype)
    dup_u = _wgrad(xg, du_l, offs, e, cap, d, r, t_rank_n, xg.dtype)
    dv = _wgrad(h, dyc, offs, e, cap, r, f, t_wide, xg.dtype)
    dx = _grouped_nt_pair(dg_l, gate_u, du_l, up_u, offs, cap, d, r, t_wide, xg.dtype)

    torch.mul(dg, gpre, out=tmp)
    dgate_s = _seg_sum(tmp, seg, e)
    torch.mul(du, upre, out=tmp)
    dup_s = _seg_sum(tmp, seg, e)
    return dx, dgate_u, dup_u, dgate_s, dup_s, dv


class _Combine(torch.autograd.Function):
    """`out = index_add(owner, y·w)` with the FUSED backward — one kernel, one read.

    The forward is the same three lines the caller used to run inline (`y` widened
    to fp32, scaled by the gathered router weight, scattered by token), so the
    forward's numbers are unchanged. What changes is the graph: the `[R, f]`
    scaled tensor is no longer an autograd node, so it is not saved and no `mul`
    backward or `index_add` backward exists — `_combine_bwd_kernel` produces both
    gradients in one pass.
    """

    @staticmethod
    def forward(ctx, y, w, owner, rowmap, n, ws, tiles):
        out = torch.zeros(n, y.shape[1], dtype=torch.float32, device=y.device)
        out.index_add_(0, owner, y.to(torch.float32) * w.unsqueeze(-1))
        ctx.save_for_backward(y, w, rowmap)
        ctx.ws, ctx.tiles = ws, tiles
        return out

    @staticmethod
    def backward(ctx, dout):
        y, w, rowmap = ctx.saved_tensors
        ws, tiles = ctx.ws, ctx.tiles
        R, f = y.shape
        n, k = rowmap.shape
        assert dout.dtype == torch.float32, "the combine's grad_out is fp32 by contract"
        dout = dout.contiguous()  # a no-op when autograd hands over a dense tensor
        dy = ws.slot("dy", (R, f), y.dtype, y.device)
        # `dw` is NOT a workspace slot: it becomes `weight.grad` when the router
        # weight is a leaf, and autograd would then alias the buffer into a leaf's
        # `.grad` — a second backward would silently rewrite the first's gradient.
        # It is `[R]` floats, so there is nothing to save by caching it.
        dw = torch.empty(R, dtype=torch.float32, device=y.device)
        _combine_bwd_kernel[(triton.cdiv(n, tiles["BLOCK_T"]),)](
            dout,
            y,
            w,
            rowmap,
            dy,
            dw,
            n,
            f,
            K=k,
            TOP_K=_topk_tile(k),
            BLOCK_T=tiles["BLOCK_T"],
            BLOCK_F=tiles["BLOCK_F"],
            num_warps=4,
        )
        return dy, dw, None, None, None, None, None


def combine(y, w, owner, rowmap, n, *, ws=None, tiles=None):
    """`out [n, f] = Σ_rows y[row]·w[row]`, the rows gathered by `owner` — fused bwd.

    The scatter half of the MoE pass (`experts.SparseMoE._triton_dispatch`), with
    `_Combine`'s fused backward. `rowmap [n, k]` int32 is the token→gathered-row
    map with `-1` where the capacity dropped the assignment; `w [R]` fp32 is the
    gathered router weight, `owner [R]` the token each gathered row belongs to.
    `ws` is the caller's grow-only `BwdWorkspace` (a throwaway one is made if
    omitted, which is correct and simply re-allocates every call).

    `w` is fp32 BY CONTRACT, asserted below rather than coerced: the kernel's
    `dy` is `grad_out · w` and its `dw` is the fp32 reduction, so a bf16 weight
    would put the whole combine path — forward and backward — a 2^-8 step off
    the adjoint it replaced. Under `bf16 = true` the routed weight really is
    bf16 (the softmax of `mm_mixed`'s bf16 logits), so the model widens it at
    this boundary (`experts.SparseMoE._triton_dispatch`); the assert is what
    keeps that a deliberate, single, visible cast instead of a silent one.
    """
    assert y.ndim == 2 and y.is_cuda and y.is_contiguous(), "y must be a dense CUDA [R, f]"
    assert w.shape == (y.shape[0],) and w.dtype == torch.float32 and w.is_contiguous()
    assert owner.shape == (y.shape[0],) and owner.is_contiguous()
    assert rowmap.ndim == 2 and rowmap.dtype == torch.int32 and rowmap.shape[0] == n
    assert rowmap.is_contiguous(), "rowmap must be dense"
    k = rowmap.shape[1]
    assert 1 <= k <= FUSED_TOPK_MAX, (
        f"topk={k} is outside the fused combine's unrolled tile [1, {FUSED_TOPK_MAX}]: "
        f"escape is a smaller moe_topk, or the current backward (`fused_bwd=False`)"
    )
    return _Combine.apply(
        y, w, owner, rowmap, n, ws if ws is not None else BwdWorkspace(), _tiles(tiles)
    )


class _ExpertGemmGlu(torch.autograd.Function):
    """The fused pass with its own backward: `fwd` Triton, `bwd` Triton (adjoint GEMMs).

    `fused_bwd` picks WHICH adjoint: `_glu_backward_fused` (default, the fused
    `dx` + the caller's row map + the caller's workspace) or `_glu_backward` (the
    reference). `seg`/`ws` are the caller's; the function neither builds nor owns
    them.

    The FORWARD is picked by `glu_arm(E)`: the fused kernel for a bank at/above
    `FUSED_GLU_MIN_E`, the GLU head plus `expert_gemm` below it (measured faster
    there — the constant's comment has both endpoints). The two compute the same
    `y`, and the backward is the same code in either case: it takes `h`, the
    saved gates and `dy` explicitly and recomputes `dh` from them, so it never
    needs a graph through `v` and the split forward is trainable, not bench-only.
    """

    @staticmethod
    def forward(ctx, xg, gate_u, up_u, gate_s, up_s, v, offs, cap, tiles, fused_bwd, seg, ws):
        if glu_arm(gate_u.shape[0]) == "split":
            _, (gpre, upre, h) = _glu_fused_or_head(
                xg, gate_u, up_u, gate_s, up_s, v, offs, cap, tiles, split=True, save=True
            )
            y = expert_gemm(h, v, offs, cap=cap, tiles=tiles)
        else:
            y, (gpre, upre, h) = _glu_fused_or_head(
                xg, gate_u, up_u, gate_s, up_s, v, offs, cap, tiles, split=False, save=True
            )
        ctx.save_for_backward(xg, gate_u, up_u, gate_s, up_s, v, offs, gpre, upre, h)
        ctx.cap, ctx.tiles = cap, tiles
        ctx.fused_bwd, ctx.seg, ctx.ws = fused_bwd, seg, ws
        return y

    @staticmethod
    def backward(ctx, dy):
        xg, gate_u, up_u, gate_s, up_s, v, offs, gpre, upre, h = ctx.saved_tensors
        args = (xg, gate_u, up_u, gate_s, up_s, v, offs, ctx.cap, gpre, upre, h, dy, ctx.tiles)
        if ctx.fused_bwd:
            grads = _glu_backward_fused(*args, ctx.seg, ctx.ws)
        else:
            grads = _glu_backward(*args)
        return (*grads, None, None, None, None, None, None)
