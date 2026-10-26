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
2. `xg = x[owner].to(dt)` (`experts.py:1621`) — THE GATHER. Since the loop's
   SEAM cast (`loopblock.LoopBlock._moe_branch`, 2026-10-21) `x` arrives in
   `dt` (bf16), so `SparseMoE._gather_rows` skips the cast and this line is
   ONE `index_select` (an [R,d] copy written) that `_glu_kernel` reads back
   (`moe_kernels.py:306`): 4·R·d, the copy plus the kernel's read. The fp32
   seam paid 12·R·d (fp32 copy, cast read, bf16 write, bf16 read); four
   [R,d]-sized transfers became two, and the fused load still has one.
   `pass_bytes(..., x_bytes=4)` prices the fp32 control, which is what
   `tests/test_moe_sonic.py` uses to prove the number moved because the seam
   moved.
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
| the gather, three transfers (bf16 seam) | 301 989 888 B (288 MiB) | 201 326 592 B (192 MiB) |
| the same rows, fused (one read) | 100 663 296 B (96 MiB) | 67 108 864 B (64 MiB) |
| **forward saving** | **201 326 592 B (192 MiB)** | **134 217 728 B (128 MiB)** |
| share of the forward's HBM bytes | 16% | 7% |
| forward traffic, all of it | 1 295 122 432 B | 2 001 305 600 B |
| fwd->bwd cache | 382 MiB (xg 96, y 256, gates 30) | 245 MiB (xg 64, y 171, gates 10) |
| fwd->bwd cache, fused + no `y` cache | 78 MiB | 42 MiB |
| backward, dy copy + 2 reads | 1024 MiB | 683 MiB |
| backward, 2 gathered fp32 reads | 1280 MiB (**+256 worse**) | 853 MiB (**+170 worse**) |

Two results that do NOT follow the paper, both from the same two facts — our
`grad_out` is **fp32** (`_Combine.backward` asserts it, `moe_kernels.py:1080`)
while our `dy` is bf16, and the MoE's OUTPUT is **fp32** while the routed
interior is bf16:

* the **forward** fusion is a real byte win, but a smaller one since the loop
  casts `normed` to the bank dtype at its seam (2026-10-21): the transfer that
  exists only because the ABI materializes the permutation is 4·R·d, where the
  fp32 seam paid 12·R·d;
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
<0.5%** — the bytes removed are 7% of a forward whose 2 001 MB of traffic runs
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

**The prototype — and the arm it is wired as.** `_gather_glu_kernel` is a REAL
Triton kernel — `_glu_kernel`'s forward phase with ONE substitution, the A
pointer's row index becomes `toks[row]` — with `fused_reference` (its math, in
torch) and `gather_abi` (the `(toks, offs)` pair, built from the permute's own
outputs) beside it so the mechanism is testable without a card. PROVEN on CPU:
the ABI (`x[toks]` is bit-identical to the copy the pass gathers), the reference
math against the module's own `_naive_dispatch`, the address audit (no read
outside `x`, and every masked lane reads row 0), and that the kernel COMPILES
for sm_120 (`triton.compile`, host ptxas; the precedent is
`tests/test_moe_kernel.py::test_the_fused_kernels_compile_for_this_arch`).
MEASURED on the card, 2026-10-18: the preflight found it IDLE (849 MiB, no
`dormouse train` process), so the two parity gates RAN instead of skipping —
the fused kernel is `expert_gemm_glu`'s numbers at fp32, at bf16, and on the
dropless static bound.

**WIRED 2026-10-21 as `moe_gather = "kernel"`** (`experts.SparseMoE.
_gather_dispatch`), because the owner's instruction was to put it in the tree and
MEASURE it rather than delete it on an estimate. What that means for a reader
deciding whether to believe a number:

* the FORWARD is the kernel: `gather_abi` builds `(toks, offs)` from the
  permute's own `order`/`keep`, and `gather_glu` is the one entry point;
* the BACKWARD is **the shipped grouped adjoint, recomputed on the same rows**
  (`gather_glu`'s `_GatherGlu`), because the prototype has no adjoint of its
  own. So the forward saves `x`, `toks` and `y` and NOT `xg` or the GLU gates,
  and the arm pays ONE extra forward per step. The parity gate's gradient half
  therefore tests the row map and the wiring, not a new adjoint;
* the arm is refused (LOUD) off the bf16 loop-stack route, with the adaptive
  width and with `factor_qat` — `config.validate` names each reason;
* **the two findings stand**: the forward is a byte WIN (the ABI's gathered copy
  and the kernel's read of it are 4·R·d since the loop's seam cast, 12·R·d on
  the fp32 control) and the backward is a byte LOSS
  (`grad_out` is fp32 while `dy` is bf16, so gathering it costs two fp32 `[R, f]`
  reads against a bf16 copy that is written once and read twice). **The
  MEASUREMENT decides what ships** — `scripts/bench_gather.py` prices both
  passes from the code's own layout at the `b1` and `medium` shapes and times
  them; until that runs there is **no speed claim in this file at all**, only
  the byte arithmetic above and the estimate `<=1% of the step, most likely
  <0.5%`.

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

try:  # the SHIPPED grouped forward: the arm's backward re-runs it (`gather_glu`)
    from . import moe_kernels
except Exception as exc:  # noqa: BLE001 — the reason is raised by `gather_glu`
    moe_kernels = None  # type: ignore[assignment]
    _SHIPPED_WHY: str | None = f"the shipped grouped GEMM is not importable here: {exc}"
else:
    _SHIPPED_WHY = None

#: The compute dtype the banks and the kernels use (`bf16 = true`, b1's setting).
DT_BYTES = 2
#: The dtype the routed pass's `x` ARRIVES in. It moved 4 -> 2 on 2026-10-21
#: with the loop's SEAM cast (`loopblock.LoopBlock._moe_branch`): the residual
#: stream is still fp32, but the MoE branch is handed `dt` once, so the gather
#: is one `index_select` instead of an fp32 copy plus a cast copy. The model
#: tests measure it through a real `DormouseModel` forward in both precision
#: modes; `pass_bytes(..., x_bytes=4)` still prices the fp32-seam control.
X_BYTES = DT_BYTES
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
    cast = x_bytes != dt_bytes  # the fp32 seam's `.to(dt)`; skipped at the bf16 seam
    out = {
        # --- the forward -------------------------------------------------
        "x_read": xg * x_bytes,  # `x[owner]`'s source rows (kept by the fusion)
        "xg_write": xg * x_bytes,  # the `index_select` result, in x's dtype
        "xg_cast_read": xg * x_bytes if cast else 0,  # `.to(dt)`'s read
        "xg_dt_write": xg * dt_bytes if cast else 0,  # the bf16 copy a cast writes
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
    # What the fusion removes: the copy the ABI materializes and the kernel's
    # read of it, plus the cast pair WHEN there is one. At the bf16 seam that
    # is 4·R·d (`xg_write` + `xg_read`); at the fp32 control it is 12·R·d
    # (fp32 copy + cast read + bf16 write + kernel read). The source read
    # (`x_read`) stays either way.
    out["forward_gather_saving"] = (
        out["xg_write"] + out["xg_cast_read"] + out["xg_dt_write"] + out["xg_read"]
    )
    out["backward_adjoint_current"] = out["dout_read"] + out["dy_write"] + out["dy_read"]
    out["backward_adjoint_fused"] = out["dout_read"] + out["dout_gathered"]
    # NEGATIVE at these shapes: the fp32 `grad_out` is twice the bf16 `dy` it
    # would replace, and it would be read twice (docstring).
    out["backward_gather_saving"] = out["backward_adjoint_current"] - out["backward_adjoint_fused"]
    # The gathered bf16 copy the forward keeps for the backward: the cast's
    # write at the fp32 seam, the `index_select`'s own result at the bf16 one.
    out["xg_copy"] = xg * dt_bytes
    # What the forward leaves on the device for the backward.
    out["cached_activation"] = out["xg_copy"] + out["y_write"] + out["gates_write"]
    out["routing_metadata"] = rows * 4  # `toks [R] int32` — the fused ABI's own cost
    return out


def wired_pass_bytes(shape: Shape) -> dict[str, int]:
    """The WIRED arm's transfers per pass — the wiring's layout, NOT the paper's.

    `pass_bytes` prices the shipped `_triton_dispatch`. This prices what
    `moe_gather = "kernel"` actually moves, and the two are not the same set: the
    FORWARD is the kernel (no `xg`, no GLU gates), but `_GatherGlu`'s BACKWARD
    recomputes the grouped forward (`_shipped_forward`) and `combine` still caches
    `y`. So the keys are:

    * `gather_forward` = `pass_bytes`' forward, minus the gather's `[R, d]`
      transfers (`xg_write` + `xg_read` = 4·R·d at the bf16 seam, 12·R·d in
      the fp32 control), minus the gates (`gpre`, `upre`, `h` — the kernel
      saves none), plus the `[R]` int32 row map;
    * `gather_cache` = what the forward leaves for the backward: `x` (the bf16
      seam input), `toks`, and `y` (`combine`'s save). At `k = 2` the cached
      `x` is `T·d·2 = R·d` bytes against the shipped `xg`'s `R·d·2`, so the
      cache saving is the gates plus `R·d` — the 4.9x/5.8x figure in the module
      docstring is the FUSED design's, which needs an adjoint that reads `dout`
      through the map and does not have one here;
    * `gather_backward` = `combine`'s adjoint (unchanged) + the recompute, which
      is the shipped forward without its `out` write and without `combine`'s read
      of `y` (the recompute feeds the adjoint, not the scatter).

    EVERY line is asserted against the shipped table, so a change to either
    dispatch fails here rather than leaving a stale price in the bench.
    `scripts/bench_gather.py` prints these numbers; `tests/test_moe_sonic.py`
    pins them on CPU, because a bench that prices a wrong layout measures
    nothing.
    """
    assert shape.topk == 2, f"the cache identity below is k=2; got k={shape.topk}"
    b = pass_bytes(shape)
    rows = shape.rows
    recompute = b["forward_total"] - b["y_read"] - b["out_write"]
    gather_forward = (
        b["x_read"]
        + b["routing_metadata"]
        + b["y_write"]
        + b["y_read"]
        + b["out_write"]
        + b["banks_read"]
    )
    gather_cache = b["x_read"] // shape.topk + b["routing_metadata"] + b["y_write"]
    out = {
        "shipped_forward": b["forward_total"],
        "shipped_backward": b["backward_adjoint_current"],
        "shipped_cache": b["cached_activation"],
        "gather_forward": gather_forward,
        "gather_backward": b["backward_adjoint_current"] + recompute,
        "gather_cache": gather_cache,
        "recompute": recompute,
    }
    out["shipped_pass"] = out["shipped_forward"] + out["shipped_backward"]
    out["gather_pass"] = out["gather_forward"] + out["gather_backward"]
    # The forward saving is the shipped pass's own closed form under the bf16
    # seam: the `index_select` copy (2·R·d) + the kernel's read of it (2·R·d),
    # where the fp32 control still prices 12·R·d. This assertion is what fails
    # if the seam's cast comes back: `pass_bytes(x_bytes=4)` would move it.
    assert b["forward_gather_saving"] == 4 * rows * shape.d, (
        "bf16 seam: the index_select copy + the kernel's read"
    )
    assert gather_forward == (
        out["shipped_forward"]
        - b["forward_gather_saving"]
        - b["gates_write"]
        + b["routing_metadata"]
    )
    assert b["gates_write"] == rows * shape.rank * (2 * 4 + DT_BYTES)
    # k=2: the cached bf16 `x` (`T·d·2 = R·d`) is half the shipped arm's
    # gathered `xg` (`R·d·2`), so the cache saving is the gates plus `R·d`.
    assert gather_cache == b["x_read"] // shape.topk + b["routing_metadata"] + b["y_write"]
    assert out["gather_cache"] == (
        out["shipped_cache"]
        - b["xg_copy"]
        - b["gates_write"]
        + b["x_read"] // shape.topk
        + b["routing_metadata"]
    )
    assert out["gather_backward"] - out["shipped_backward"] == recompute
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
    from .experts import _bincount  # lazy: `experts` imports this module

    counts = _bincount(kept, experts).to(torch.int32)
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
            # The cast on load meets the banks' dtype (`dt`). At the loop's
            # bf16 seam `x` is already `dt` and this is a no-op; a direct API
            # caller passing fp32 gets the conversion here instead of in a
            # separate `to(dt)` copy.
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

    `gather_glu` is the WIRED entry point (it adds the backward — see
    `_GatherGlu`); this function is its forward alone, and it is the seam the
    CPU gate swaps for `fused_reference` (`_kernel_forward`). `SparseMoE.
    _gather_dispatch` is the only production caller's caller.
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


def _kernel_forward(
    x: torch.Tensor,
    toks: torch.Tensor,
    gate_u: torch.Tensor,
    up_u: torch.Tensor,
    gate_s: torch.Tensor,
    up_s: torch.Tensor,
    v: torch.Tensor,
    offs: torch.Tensor,
    cap: int | None,
) -> torch.Tensor:
    """The arm's FORWARD call, named so a card-less gate can swap it.

    A SEAM, not a fallback: production never substitutes it, and the CPU gate
    (`tests/test_moe_sonic.py`) replaces it with `fused_reference` to prove the
    ABI and the gradient routing without a GPU — the kernel itself is only
    measurable on the card, and saying which half a gate covers is the point of
    this seam existing at all.
    """
    return fused_expert_gemm_glu(x, toks, gate_u, up_u, gate_s, up_s, v, offs, cap=cap)


def _shipped_forward(
    xg: torch.Tensor,
    gate_u: torch.Tensor,
    up_u: torch.Tensor,
    gate_s: torch.Tensor,
    up_s: torch.Tensor,
    v: torch.Tensor,
    offs: torch.Tensor,
    *,
    cap: int | None,
    seg: torch.Tensor,
    ws: object,
) -> torch.Tensor:
    """The recompute the arm's BACKWARD is: the shipped grouped forward, on `xg`.

    The second seam (see `_kernel_forward`): production never substitutes it, and
    the card-less gate does, because `_glu_backward_fused` is a Triton launch and
    a CPU gradient gate would otherwise compare nothing. What the swap leaves
    unmeasured is the adjoint itself — it is the shipped one, so there is nothing
    new to measure — and what it still measures is exactly the arm's own risk:
    that `toks` gathers the rows the routing meant, in the order the banks and
    `offs` describe, and that `dx` lands on those token rows.
    """
    if moe_kernels is None:
        raise AssertionError(f"the gather arm cannot backpropagate: {_SHIPPED_WHY}")
    return moe_kernels.expert_gemm_glu(
        xg, gate_u, up_u, gate_s, up_s, v, offs, cap=cap, fused_bwd=True, seg=seg, ws=ws
    )


class _GatherGlu(torch.autograd.Function):
    """`forward` = the gather kernel; `backward` = the SHIPPED adjoint, recomputed.

    **The prototype has no backward**, and inventing one would be a second
    unmeasured adjoint next to a kernel the owner asked to MEASURE. So the
    backward re-runs `moe_kernels.expert_gemm_glu` on the SAME rows — gathered
    from `x` through `toks` inside `torch.enable_grad()` — and lets its own
    Triton adjoint produce `dxg`, the four bank grads and `dv`. `dx` arrives
    from autograd's `index_select` scatter, which is the same scatter the
    shipped arm's `x[owner].to(dt)` backward performs.

    What that costs is ONE extra forward per backward — the
    `torch.utils.checkpoint` trade, taken because `_gather_glu_kernel` saves no
    gate (`gpre`, `upre`, `h`) for the adjoint to read. What it buys is a
    fwd→bwd cache of `x`, `toks` and `y` only: no `xg [R, d]`, no GLU gates.
    `moe_sonic.pass_bytes` prices the cached set the fused design *would* have;
    `scripts/bench_gather.py` prices the set THIS wiring has, because a bench
    that prices the paper's design instead of the shipped one measures nothing.

    Consequence for the gate: the gradient half of the parity test is a test of
    the ROW MAP and of the wiring, not of an adjoint — there is no new adjoint.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        toks: torch.Tensor,
        gate_u: torch.Tensor,
        up_u: torch.Tensor,
        gate_s: torch.Tensor,
        up_s: torch.Tensor,
        v: torch.Tensor,
        offs: torch.Tensor,
        cap: int | None,
        seg: torch.Tensor,
        ws: object,
    ) -> torch.Tensor:
        ctx.cap, ctx.seg, ctx.ws = cap, seg, ws
        ctx.save_for_backward(x, toks, gate_u, up_u, gate_s, up_s, v, offs)
        return _kernel_forward(x, toks, gate_u, up_u, gate_s, up_s, v, offs, cap)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, toks, gate_u, up_u, gate_s, up_s, v, offs = ctx.saved_tensors
        with torch.enable_grad():
            # The recompute must be a FUNCTION OF THE SAME LEAVES, in a fresh
            # graph: `x` is the row source (so `index_select`'s backward scatters
            # `dxg` into the token rows), and the five bank tensors are detached
            # copies with their own `.grad` — the shipped adjoint differentiates
            # exactly these, and nothing in `_banks`' cache is walked twice.
            xd = x.detach().requires_grad_(True)
            xg = xd.index_select(0, toks.to(torch.int64))
            if xg.dtype != gate_u.dtype:
                xg = xg.to(gate_u.dtype)
            leaves = [t.detach().requires_grad_(True) for t in (gate_u, up_u, gate_s, up_s, v)]
            y = _shipped_forward(xg, *leaves, offs, cap=ctx.cap, seg=ctx.seg, ws=ctx.ws)
            grads = torch.autograd.grad(y, [xd, *leaves], dy, allow_unused=True)
        # x, toks, gate_u, up_u, gate_s, up_s, v, offs, cap, seg, ws
        return (grads[0], None, *grads[1:], None, None, None, None)


def gather_glu(
    x: torch.Tensor,
    toks: torch.Tensor,
    gate_u: torch.Tensor,
    up_u: torch.Tensor,
    gate_s: torch.Tensor,
    up_s: torch.Tensor,
    v: torch.Tensor,
    offs: torch.Tensor,
    *,
    cap: int | None,
    seg: torch.Tensor,
    ws: object,
) -> torch.Tensor:
    """`y [R, f]` from the UNSORTED `x` and the row map, forward AND backward.

    The one call `experts.SparseMoE._gather_dispatch` makes (`moe_gather =
    "kernel"`). `seg [R] int32` is the permute's own row→expert map and `ws` the
    caller's `moe_kernels.BwdWorkspace`, both passed straight to the recompute —
    the shipped fused adjoint takes them for the same reason there: no
    `repeat_interleave` rebuild and no per-call `[R, ·]` scratch.

    Refuses LOUDLY, with the escape, when either half's ability is missing: the
    forward needs the compiled kernel, the backward needs the shipped grouped
    GEMM (it IS that adjoint). Neither is a silent fallback to the other arm —
    `moe_gather = "kernel"` is a config's ask, and `_dispatch` refuses it on a
    CPU before this is reached.
    """
    if _TRITON_WHY is not None:
        raise AssertionError(
            f"the gather arm cannot run: {_TRITON_WHY}. Escape: moe_gather='off' "
            f"(the grouped GEMM / padded / naive arms), or a build with triton on "
            f"a CUDA device."
        )
    if _SHIPPED_WHY is not None:
        raise AssertionError(
            f"the gather arm's backward re-runs the shipped grouped GEMM: "
            f"{_SHIPPED_WHY}. Escape: moe_gather='off', or a build with triton."
        )
    return _GatherGlu.apply(x, toks, gate_u, up_u, gate_s, up_s, v, offs, cap, seg, ws)
