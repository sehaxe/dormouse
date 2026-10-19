"""SonicMoE's two mechanisms, priced against THIS pass — analysis + prototype.

**Source.** W. Guo, M. Mishra, X. Cheng, I. Stoica, T. Dao, *SonicMoE:
Accelerating MoE with IO and Tile-aware Optimizations*, arXiv:2512.14080v2
(2025-12-16). Read 2026-10-18 through the arXiv MCP at section level: the
abstract, §4.2 (activation memory), §5.1.1 (gather fusion with the GMEM-to-SMEM
load), §6.2 (token rounding), §7.2.1 (the B300/DeepGEMM++ table). Two
mechanisms, quoted rather than paraphrased:

* **gather fusion** (§5.1.1): "we fuse the input gather with the input loads
  from global memory (GMEM, often the HBM) to shared memory (SMEM) ... fetching
  the routed token indices for each expert and then using these indices to
  gather activations via the `cp.async` instruction". The ABI consequence: the
  grouped GEMM takes the ROUTED TOKEN INDICES, never a permuted copy of `X`.
  Backward half: "fusing this gather reduces the IO cost by 2TKd bytes".
* **token rounding** (§6.2): per-expert token counts are rounded to the nearest
  multiple of the MMA tile `M_tile` (128 in their tables) by padding or
  discarding the LAST tile, so "TR guarantees no tile quantization effects";
  measured 1.16x kernel speedup at high sparsity.

Their throughput claims (B300, OLMoE-sized 7B: +25% fwd / +15% bwd over
DeepGEMM++; 96xH100: 213B tokens/day) are *kernel* numbers on a bf16 activation
stream with `cp.async` and 2-CTA-cluster pipelines — none of which a Triton
program on sm_120 has, and none of which is a measurement of our pass.

**What THIS pass pays, per routed pass, from the code.** The shipped arm at
`b1` is `_triton_dispatch` (`experts.py:941-1038`; `configs/b1.toml:46-47` sets
`moe_kernel = "triton"`, `moe_capacity = 0`), and its per-pass transfers are:

1. `order = argsort(flat_ids)` -> `rows = order[keep]` -> `owner = rows // k`
   (`experts.py:982-983`). This IS the row->token map the fused kernel needs,
   and the pass already builds it — one `int32` cast is the whole ABI delta.
2. `xg = x[owner].to(dt)` (`experts.py:990`) — THE GATHER. `x` arrives fp32
   (`nn.RMSNorm`'s output; MEASURED 2026-10-18, see
   `tests/test_moe_sonic.py::test_the_moe_input_is_fp32_so_the_gather_is_two_copies`)
   and `dt` is bf16, so this line is a fp32 `index_select` (one [R,d] copy
   written) followed by a bf16 cast (a second [R,d] copy written), and
   `_glu_kernel` then loads the bf16 copy (`moe_kernels.py:306`). Four
   [R,d]-sized transfers where the fused load has one.
3. `y [R,f]` written by the kernel (`moe_kernels.py:351`) and read back by the
   combine (`moe_kernels.py:531`) — the second [R,f]-sized activation, needed
   by the `dw` reduction.
4. `dy [R,f]` (bf16) written by `_combine_bwd_kernel` (`moe_kernels.py:526-530`)
   and read twice by the adjoint (`_grouped_nt` for `dh`, `_wgrad` for `dv`).

`pass_bytes()` is that list as arithmetic. `SHAPES`' fields are the CONFIGS'
own — `tests/test_moe_sonic.py` reads `configs/medium.toml` and `configs/b1.toml`
with `tomllib` and asserts them, so a geometry change fails the table:

| per routed pass | medium b24·s1024 | b1 b8·s1024 |
|---|---|---|
| geometry (E, d, f, r) | 256, 1024, 2730, 64 | 1024, 2048, 5461, 64 |
| rows R = T·k | 49 152 | 16 384 |
| the gather, four transfers | 805 306 368 B (768 MiB) | 536 870 912 B (512 MiB) |
| the same rows, fused (one read) | 201 326 592 B (192 MiB) | 134 217 728 B (128 MiB) |
| **forward saving** | **603 979 776 B (576 MiB)** | **402 653 184 B (384 MiB)** |
| share of the forward's HBM bytes | 34% | 17% |
| forward traffic, all of it | 1 798 438 912 B | 2 336 849 920 B |
| fwd->bwd cache | 382 MiB (xg 96, y 256, gates 30) | 245 MiB (xg 64, y 171, gates 10) |
| fwd->bwd cache, fused + no `y` cache | 78 MiB | 42 MiB |
| backward, dy copy + 2 reads | 1024 MiB | 683 MiB |
| backward, 2 gathered fp32 reads | 1280 MiB (**+256 worse**) | 853 MiB (**+170 worse**) |

Two results that do NOT follow the paper, both from the same two facts — our
`grad_out` is **fp32** (`_Combine.backward` asserts it, `moe_kernels.py:1080`)
while our `dy` is bf16, and our `x` is **fp32** while the compute dtype is bf16:

* the **forward** fusion is a real byte win: two of the four transfers are
  copies that exist only because the ABI materializes the permutation;
* the **backward** fusion is a byte LOSS: gathering `dout` inside the `dH`
  kernel means two [R,f] reads at 4 B/element (8Rf) where the shipped pass
  writes 2Rf and reads 2·2Rf (6Rf). The paper's "-2TKd bytes" assumes a bf16
  activation stream end to end; ours has an fp32 boundary at the MoE output
  (`SparseMoE.forward` returns fp32), so the bf16 copy is CHEAPER than the
  fp32 re-reads. The same arithmetic is why their −45% activation-memory claim
  is the part that transfers better than the throughput claim: dropping the
  `xg` and `y` caches is a 4.9x reduction of the fwd->bwd set at `medium` and
  5.8x at `b1` (the paper's own −45% is at their geometry, not ours).

**Is the pass landed?** Yes. The dropless grouped GEMM with the fused backward
IS the shipped arm (`configs/b1.toml:47`), the padding a gather-fused kernel
would help remove was already removed (`moe_capacity = 0` plus the kernels'
on-device empty-block exit), and the measured cost of the b1 pass is CHROME,
not the gather: `_banks()` ~15 ms host + ~4 ms of `.to()` copies, the fp32
weight multiply 6.7 ms, the scatter 5.5 ms against a 6.7 ms bmm chain
(`.bulba/memory.md:312`, 2026-10-15). The research doc's "up to 15-25% step IF
the pass is unlanded" (`docs/research/2026-10-18-throughput-leftovers.md` §4.2)
is therefore NOT met. Honest expectation here: **<=1% of the step, most likely
<0.5%** — the bytes removed are 17% of a forward whose 2 337 MB of traffic runs
inside a measured 16.5 ms (`configs/b1.toml:47`), i.e. at ~32% of this card's
~450 GB/s ceiling (the figure `docs/research/2026-10-15-tsct-super-kernel.md:54`
uses), and part of the bank re-read is served by L2 — so bytes removed do not
convert one-for-one into time.

**What the numbers say IS worth a GPU bench** (nothing in this file is a
timing: the pass's isolated milliseconds are `scripts/bench_moe_dropless.py`'s,
and the parity gates at the bottom are the only thing here that touches the
card):

* **tile quantization at b1's dropless routing.** 16 384 rows over 1 024
  experts is 16 rows per expert against a 64-row tile, so `Σ_e
  cdiv(n_e, BLOCK_M)·BLOCK_M` = 65 536 row-slots are computed for 16 384 useful
  ones — **25% tile utilisation**, 80 GFLOP of MMA for a 20 GFLOP answer
  (`tile_work()` is that formula). This is what token rounding (§6.2) exists
  for, and our equivalent is not a kernel change but a ROUTING change
  (`_permute`/`_capacity`, another lane's file) or a smaller `BLOCK_M` — which
  the dropless static bound makes expensive: the grid is
  `(E, cdiv(R, BLOCK_M))` = 262 144 programs at the shipped tile, ~1 024 with
  work, and `scripts/bench_moe_dropless.py` prints exactly that pair.
* **the bank re-read.** At `b1` the [E,·] weight banks are 1 195 MiB of the
  forward's 2 337 MB — 54% of its HBM traffic for 20 GFLOP of useful work. That
  is the fine-grained-MoE problem §3.2 of the paper names, and it is not
  something the gather fusion touches.

**The prototype.** `_gather_glu_kernel` is a REAL Triton kernel —
`_glu_kernel`'s forward phase with ONE substitution, the A pointer's row index
becomes `toks[row]` — with `fused_reference` (its math, in torch) and
`gather_abi` (the `(toks, offs)` pair, built from the permute's own outputs)
beside it so the mechanism is testable without a card. It is NOT wired into
`experts.py` (this lane does not hold that file); the seam it would land on is
`_triton_dispatch`'s `xg = x[owner].to(dt)` line. PROVEN on CPU: the ABI
(`x[toks]` is bit-identical to the copy the pass gathers), the reference math
against the module's own `_naive_dispatch`, the address audit (no read outside
`x`, and every masked lane reads row 0), and that the kernel COMPILES for
sm_120 (`triton.compile`, host ptxas; the precedent is
`tests/test_moe_kernel.py::test_the_fused_kernels_compile_for_this_arch`).
MEASURED on the card, 2026-10-18: the preflight found it IDLE (849 MiB, no
`dormouse train` process), so the two parity gates RAN instead of skipping —
the fused kernel is `expert_gemm_glu`'s numbers at fp32, at bf16, and on the
dropless static bound. STILL unmeasured: the kernel's TIME, which needs a bench
and not a gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:  # the Triton arm is an ABILITY, not a dependency (`experts.py:141-147`)
    import triton
    import triton.language as tl
except Exception as exc:  # noqa: BLE001 — the reason is raised by the launcher
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_WHY: str | None = f"the triton kernels are not importable here: {exc}"
else:
    _TRITON_WHY = None

#: The dtype the MoE input ARRIVES in. Not a guess: `normed = self.norm(ctx)` is
#: an `nn.RMSNorm` over the fp32 loop state (`loopblock.py:278,130`), and
#: `tests/test_moe_sonic.py` measures it through a real `DormouseModel` forward
#: in both precision modes. It is 4 bytes because of that, and the whole
#: two-copies claim of the accounting hangs on it.
X_BYTES = 4
#: The compute dtype the banks and the kernels use (`bf16 = true`, b1's setting).
DT_BYTES = 2
#: The MoE output / `grad_out` dtype: `SparseMoE.forward` returns fp32 and
#: `_Combine.backward` asserts fp32 (`moe_kernels.py:1080`).
GRAD_BYTES = 4

#: The shipped tile set, re-declared so the prototype cannot silently drift from
#: `moe_kernels.TILES` — the test asserts the two are equal, and the kernel below
#: is only a fair comparison to `_glu_kernel` at the same tiles.
TILES = {"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_F": 128, "BLOCK_N": 128, "BLOCK_T": 16}


@dataclass(frozen=True)
class Shape:
    """One routed-pass geometry. Every field is a config field (see `SHAPES`)."""

    name: str
    tokens: int  # T = batch x seq for the pass (a CLI shape, not a config field)
    topk: int
    experts: int
    d: int
    f: int
    rank: int

    @property
    def rows(self) -> int:
        """`R` — the gathered rows `_permute` keeps (dropless: exactly T·k)."""
        return self.tokens * self.topk


#: The two shapes this file prices. `medium` is `configs/medium.toml` at the
#: batch/seq this analysis was asked for (b24·s1024, 24 576 tokens); `b1` is
#: `configs/b1.toml` at the b1 profile's own shape (b8·s1024, 8 192 tokens ->
#: 16 384 rows, `docs/reviews/2026-10-16-b1-profile.md:9` and the 16 384-row
#: fixtures of `scripts/bench_moe_dropless.py:51`).
SHAPES = (
    Shape("medium-b24-s1024", tokens=24 * 1024, topk=2, experts=256, d=1024, f=2730, rank=64),
    Shape("b1-b8-s1024", tokens=8 * 1024, topk=2, experts=1024, d=2048, f=5461, rank=64),
)


def pass_bytes(
    shape: Shape,
    *,
    x_bytes: int = X_BYTES,
    dt_bytes: int = DT_BYTES,
    grad_bytes: int = GRAD_BYTES,
) -> dict[str, int]:
    """Every [R,·]- and [E,·]-sized transfer of ONE routed pass, in bytes.

    Read the docstring's table with this function open: each key names the
    tensor and the code that moves it, and the two `*_saving` keys are the
    differences a fused pass would take. `forward_total` is the sum of the
    forward's transfers (so the saving has a denominator), and
    `backward_adjoint_*` covers only the combine + expert-adjoint operand path
    (the rest of the adjoint — `dx`, the two `wgrad`s — is unchanged by the
    fusion and would only add the same number to both sides).
    """
    t, e = shape.tokens, shape.experts
    d, f, r = shape.d, shape.f, shape.rank
    rows = shape.rows
    xg = rows * d  # one [R, d] copy, in elements
    y = rows * f
    out = {
        # --- the forward -------------------------------------------------
        "x_read": xg * x_bytes,  # `x[owner]`'s source rows (kept by the fusion)
        "xg_write": xg * x_bytes,  # the fp32 `index_select` result
        "xg_cast_read": xg * x_bytes,  # `.to(dt)`'s read
        "xg_dt_write": xg * dt_bytes,  # the bf16 copy the kernel gets today
        "xg_read": xg * dt_bytes,  # `_glu_kernel`'s A load (moe_kernels.py:306)
        "y_write": y * dt_bytes,  # the pass's [R, f] output
        "y_read": y * dt_bytes,  # `combine`'s read of it (`dw`)
        "out_write": t * f * grad_bytes,  # the [T, f] fp32 MoE output
        "banks_read": e * (2 * d * r + r * f) * dt_bytes,  # gate_u + up_u + v
        "gates_write": rows * r * (2 * 4 + dt_bytes),  # gpre, upre (fp32); h (dt)
        # --- the backward ------------------------------------------------
        "dy_write": y * dt_bytes,  # `_combine_bwd_kernel`'s [R, f] adjoint
        "dy_read": 2 * y * dt_bytes,  # read by `dh` and by the `dv` wgrad
        "dout_read": t * f * grad_bytes,  # `grad_out` once per token (combine)
        "dout_gathered": 2 * y * grad_bytes,  # what a fused adjoint reads instead
    }
    out["forward_total"] = sum(
        out[key]
        for key in (
            "x_read",
            "xg_write",
            "xg_cast_read",
            "xg_dt_write",
            "xg_read",
            "y_write",
            "y_read",
            "out_write",
            "banks_read",
            "gates_write",
        )
    )
    # The fusion removes the fp32 copy, its cast-read, the bf16 copy and the
    # kernel's read of that copy; the source read stays.
    out["forward_gather_saving"] = (
        out["xg_write"] + out["xg_cast_read"] + out["xg_dt_write"] + out["xg_read"]
    )
    out["backward_adjoint_current"] = out["dout_read"] + out["dy_write"] + out["dy_read"]
    out["backward_adjoint_fused"] = out["dout_read"] + out["dout_gathered"]
    # NEGATIVE at these shapes: the fp32 `grad_out` is twice the bf16 `dy` it
    # would replace, and it would be read twice (docstring).
    out["backward_gather_saving"] = out["backward_adjoint_current"] - out["backward_adjoint_fused"]
    # What the forward leaves on the device for the backward.
    out["cached_activation"] = out["xg_dt_write"] + out["y_write"] + out["gates_write"]
    out["routing_metadata"] = rows * 4  # `toks [R] int32` — the fused ABI's own cost
    return out


def tile_work(shape: Shape, counts: list[int] | tuple[int, ...], block_m: int = 64) -> int:
    """`Σ_e cdiv(n_e, BLOCK_M)·BLOCK_M` — the row-slots the kernel actually runs.

    The row-block grid is `(E, cdiv(cap, BLOCK_M))` and every program computes a
    FULL `[BLOCK_M, ·]` tile, so an expert with 16 rows costs the same MMA work
    as one with 64. `counts` is the per-expert row count the routing produced
    (`torch.bincount(kept, minlength=E)`); with no routing to hand, the uniform
    case is `[R // E] * E`, which is the docstring's 25%-at-b1 arithmetic.
    """
    assert block_m >= 1 and len(counts) >= 1, "a tile needs a block and an expert"
    assert sum(counts) >= 1, "a routing with no rows has no tile work"
    return sum(-(-int(n) // block_m) * block_m for n in counts)


def gather_abi(
    index: torch.Tensor, order: torch.Tensor, keep: torch.Tensor, experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(toks, offs)` — the fused kernel's ABI, built from the permute's OWN outputs.

    `SparseMoE._triton_dispatch` (`experts.py:980-990`) computes `rows`,
    `owner`, `kept` and then materializes `xg = x[owner]`; this is those same
    three lines with the gather replaced by a cast. `toks [R] int32` is the
    token of every gathered row IN GATHERED ORDER (so `toks[row]` is the source
    row of the block a program owns), and `offs [E+1] int32` is the same ragged
    partition the shipped kernels already take — the same `cumsum(bincount)`
    the shipped path builds at `experts.py:985-987`.

    Nothing here is new information: `owner` is `rows // k`, and `experts` is
    the bank's own `n_experts` (an empty expert must still get its `offs` slot,
    which is why the count cannot be inferred from the routing).
    """
    assert index.ndim == 2 and experts >= 1, "a routing needs [T, k] and a bank size"
    k = index.shape[1]
    rows = order[keep]
    toks = (rows // k).to(torch.int32).contiguous()
    kept = index.reshape(-1)[rows]
    counts = torch.bincount(kept, minlength=experts).to(torch.int32)
    offs = torch.zeros(experts + 1, dtype=torch.int32, device=index.device)
    offs[1:] = torch.cumsum(counts, 0)
    return toks, offs


def fused_reference(
    x: torch.Tensor,
    toks: torch.Tensor,
    gate_u: torch.Tensor,
    up_u: torch.Tensor,
    gate_s: torch.Tensor,
    up_s: torch.Tensor,
    v: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    """`_gather_glu_kernel`'s math, in torch: `y[row] = GLU(x[toks[row]]) @ v[e]`.

    The op order is the KERNEL's, not the naive path's: fp32 accumulate, then
    the rounding chain of `_glu_kernel:327-331` (`round -> scale -> round`, the
    silu on the widened gate, `round`), then `h @ v` rounded once. That is what
    makes a bf16 comparison against `_naive_dispatch` a comparison of two
    rounding chains (the file's own 2e-3 bound) rather than of one chain
    against itself, and it is why the fp32 arm can carry 1e-6.
    """
    dt = v.dtype
    e = v.shape[0]
    rows_total = toks.shape[0]
    assert x.ndim == 2 and toks.ndim == 1 and offs.ndim == 1
    assert offs.shape == (e + 1,) and int(offs[-1]) == rows_total, (
        "offs must partition the gathered rows: offs[-1] == toks.numel()"
    )
    out = torch.zeros(rows_total, v.shape[2], dtype=dt, device=x.device)
    for i in range(e):
        start, end = int(offs[i]), int(offs[i + 1])
        if end <= start:
            continue
        xg = x[toks[start:end].to(torch.int64)].to(dt)
        g = (xg @ gate_u[i].to(dt)).float()
        u = (xg @ up_u[i].to(dt)).float()
        g = (g.to(dt) * gate_s[i].to(dt)).to(dt)
        u = (u.to(dt) * up_s[i].to(dt)).to(dt)
        gf = g.float()
        h = ((torch.sigmoid(gf) * gf).to(dt) * u).to(dt)
        out[start:end] = (h @ v[i].to(dt)).to(dt)
    return out


if triton is not None:

    @triton.jit
    def _gather_glu_kernel(
        x_ptr,
        toks_ptr,
        gu_ptr,
        uu_ptr,
        gs_ptr,
        us_ptr,
        v_ptr,
        y_ptr,
        offs_ptr,
        D,
        R_,
        F_,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_F: tl.constexpr,
        RP: tl.constexpr,
        PREC: tl.constexpr,
    ):
        """`silu((x[toks[row]]@gate_u)·gate_s) ⊙ (...) @ v`, gather fused into the A load.

        `moe_kernels._glu_kernel`'s forward phase with exactly ONE substitution:
        the A pointer's row index is `toks[row]` — the TOKEN of the gathered row
        this program owns — instead of `row` itself. Same tiles, same masks,
        same early exit, same accumulator order, same GLU chain, so the two
        kernels are the same function over the same numbers and differ only in
        which memory the A operand comes from.

        `toks` is loaded with `m_mask` and `other=0`: a masked lane therefore
        reads ROW 0 of `x`, which is inside the tensor by construction — and the
        A load's own `m_mask` is what keeps that value out of the tile. That is
        the one safety property the ABI adds over `_glu_kernel`, and
        `tests/test_moe_sonic.py` audits it on CPU.
        """
        e = tl.program_id(0)
        start = tl.load(offs_ptr + e)
        end = tl.load(offs_ptr + e + 1)
        # EARLY EXIT: `_glu_kernel`'s contract (the dropless static bound).
        if start + tl.program_id(1) * BLOCK_M >= end:
            return
        rows = start + tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = rows < end
        # THE GATHER, FUSED: the source token of every row this program owns.
        toks = tl.load(toks_ptr + rows, mask=m_mask, other=0)
        rr = tl.arange(0, RP)
        r_mask = rr < R_
        kk = tl.arange(0, BLOCK_K)

        # The only line that differs from `_glu_kernel`: `toks` where `rows` was.
        xp = x_ptr + toks[:, None] * D + kk[None, :]
        gp = gu_ptr + e * D * R_ + kk[:, None] * R_ + rr[None, :]
        up = uu_ptr + e * D * R_ + kk[:, None] * R_ + rr[None, :]
        g_acc = tl.zeros((BLOCK_M, RP), tl.float32)
        u_acc = tl.zeros((BLOCK_M, RP), tl.float32)
        for k in range(0, tl.cdiv(D, BLOCK_K)):
            rem = D - k * BLOCK_K
            # The cast on load is the fp32 input meeting a bf16 dot: `x` is fp32
            # (`X_BYTES`), the banks are `dt`, and the fused load is where the
            # conversion now happens instead of in a separate `to(dt)` copy.
            a = tl.load(xp, mask=m_mask[:, None] & (kk[None, :] < rem), other=0.0).to(
                y_ptr.dtype.element_ty
            )
            w_mask = (kk[:, None] < rem) & r_mask[None, :]
            g_acc = tl.dot(a, tl.load(gp, mask=w_mask, other=0.0), g_acc, input_precision=PREC)
            u_acc = tl.dot(a, tl.load(up, mask=w_mask, other=0.0), u_acc, input_precision=PREC)
            xp += BLOCK_K
            gp += BLOCK_K * R_
            up += BLOCK_K * R_

        gs = tl.load(gs_ptr + e * R_ + rr)
        us = tl.load(us_ptr + e * R_ + rr)
        # The rounding chain of `_glu_kernel:327-331`, verbatim: the parity gate
        # compares a rounding order, not just a formula.
        g = (g_acc.to(y_ptr.dtype.element_ty) * gs[None, :]).to(y_ptr.dtype.element_ty)
        u = (u_acc.to(y_ptr.dtype.element_ty) * us[None, :]).to(y_ptr.dtype.element_ty)
        gf = g.to(tl.float32)
        h = ((tl.sigmoid(gf) * gf).to(y_ptr.dtype.element_ty) * u).to(y_ptr.dtype.element_ty)

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


def fused_expert_gemm_glu(
    x: torch.Tensor,
    toks: torch.Tensor,
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
    """`y [R, f]` from the UNSORTED `x [T, d]` and the row map — no gathered copy.

    The prototype's one entry point, with `moe_kernels`' own layout contract
    except that `xg [R, d]` is replaced by `x [T, d]` + `toks [R] int32`
    (`gather_abi` builds the pair). `cap` bounds the row-block grid exactly as
    it does there: the capacity rule's value when the caller has one, else the
    DROPLESS static bound from a shape (`toks.numel()`, no host read of `offs`).

    NOT wired into `experts.py` — this lane does not hold that file. The seam is
    `_triton_dispatch`'s `xg = x[owner].to(dt)` line: replace it with
    `toks, offs = gather_abi(index, order, keep)` and pass `x` here.
    """
    if _TRITON_WHY is not None:
        raise AssertionError(
            f"the fused gather kernel cannot run here: {_TRITON_WHY}. "
            f"Escape: the padded or naive arm (`SparseMoE._dispatch`), or a build "
            f"with triton on a CUDA device."
        )
    e, d, r = gate_u.shape
    f = v.shape[2]
    rows = toks.shape[0]
    assert x.ndim == 2 and x.shape[1] == d, f"x must be [T, {d}]"
    assert toks.ndim == 1 and toks.dtype == torch.int32 and rows >= 1, "toks must be [R] int32"
    assert gate_u.shape == up_u.shape and gate_s.shape == up_s.shape == (e, r)
    assert v.shape[0] == e and v.shape[1] == r and v.dtype == gate_u.dtype, "one compute dtype"
    assert offs.shape == (e + 1,) and offs.dtype == torch.int32, "offs must be [E+1] int32"
    for t in (x, toks, gate_u, up_u, gate_s, up_s, v, offs):
        assert t.is_cuda and t.is_contiguous(), "the Triton arm wants contiguous CUDA tensors"
    if cap is None:
        cap = rows
    assert cap >= 1, f"cap={cap} must be >= 1"
    t = dict(TILES)
    if tiles:
        unknown = set(tiles) - set(TILES)
        assert not unknown, f"unknown tile(s) {sorted(unknown)}; known: {sorted(TILES)}"
        t.update(tiles)
    dt = gate_u.dtype  # the banks' dtype IS the compute dtype (moe_kernels._prec's rule)
    y = torch.empty(rows, f, dtype=dt, device=x.device)
    _gather_glu_kernel[(e, max(1, triton.cdiv(cap, t["BLOCK_M"])))](
        x,
        toks,
        gate_u,
        up_u,
        gate_s,
        up_s,
        v,
        y,
        offs,
        d,
        r,
        f,
        BLOCK_M=t["BLOCK_M"],
        BLOCK_K=t["BLOCK_K"],
        BLOCK_F=t["BLOCK_F"],
        RP=max(16, triton.next_power_of_2(r)),
        PREC="ieee" if dt == torch.float32 else "tf32",
        num_warps=4,
    )
    return y
