"""MoE: top-k dispatch over dual-latent GLU experts, plus an always-on shared expert.

The expert is TSCT2 §14's shape (`docs/reviews/tsct2-design-2026-10-04.md`):

* **two latents, one for gate and one for up.** §13.3 names the defect this
  fixes: a *shared* gate/up latent makes the GLU a rescaling of one subspace.
* **the GLU sits AT the latent** (`silu(gate_latent) ⊙ up_latent`, rank `r`),
  so the write into `f` is a single `r × f` factor.
* **the write goes through one shared `f → d` projection** shared by every
  expert and by the shared expert. This is the correction the burn-side
  pricing needed: plan §2.2's expert term `E·(d·r + r·f + r)` has **no
  down-projection at all** (the spike folded one shared `[5461→2048]`,
  +11.2M, and said so). The preset comment carries the recomputed count.
* **the shared expert is always on**, rank `shared_rank` (TSCT2 §13.4: writing
  only into a 64-dim subspace gives too small an `R_act`; §14 answers with a
  shared expert of rank ~192-256 next to routed `r=32-64`).

ONE switch on the routed experts, OFF by default: `share_u_mode` (owner
directive 2026-10-21). The two INPUT bases (`gate_u`, `up_u`, each `[d, rank]`)
stop being per-expert parameters and the bank keeps only each expert's `s` and
its write `V`. `share_u` is the old boolean spelling of `share_u_mode="plane"`;
one switch, two spellings, and a conflicting pair is refused (`config`).

* **`plane`** — `SparseMoE` owns ONE `[d, rank]` per site and binds it into
  every routed expert. What the bank loses is E interchangeable copies of the
  same shape; what it keeps is a per-expert output direction and scale. That is
  the shape a rank-2 bank needs — and its weakness is structural: every expert
  projects the input onto the SAME `rank`-dimensional plane, so at rank 2 the
  whole bank can only "see" two directions of a `d`-dim state.
* **`dict`** — the refinement that can plausibly give rank 2 life:
  `SparseMoE` owns ONE `[d, share_u_rank]` dictionary per site and each expert
  reads `rank` of its columns, `idx [2, E, rank]` (row 0 the gate latent, row 1
  the up latent). Two experts then see DIFFERENT `rank`-dimensional subspaces of
  the same dictionary, and the bank's capacity lives in `d·share_u_rank` instead
  of in E private copies. At medium (d=1024, f=2730, E=256, R=64, rank 2) the
  bank is 1 464 320 parameters against the per-expert-`U` control's 78 315 520
  (53x), the same order as `plane` rank 2 — which is the whole point: the
  comparison against `plane` isolates *per-expert subspaces*, not the size.
  The indices are **FIXED and seeded, never learned**: choosing a column is a
  discrete decision with no gradient signal, so a learned selector is a second
  mechanism (a straight-through estimator on top of a shared basis), and the
  cheap falsification asks whether per-expert subspaces help AT ALL before
  anyone pays for the selector. The dictionary gets its gradient from the
  gathered columns (the sum over every expert that read them), which is what
  `tests/test_share_u_dict.py` gate D pins.

`share_u_forward` counts the `plane` forward and `share_u_dict_forward` the
`dict` one, each once per `SparseMoE.forward` (the site) — the two are
mutually exclusive, so a log carrying both is two MoEs or a wiring error, and a
`share_u_mode="dict"` run with no `share_u_dict_forward` measured a forward
that did not run the arm. `share_u_qat_forward` counts the same forward when
`factor_qat != "none"` (a log without it says the rung did not ride the basis).
The arm is refused LOUDLY with `use_tsct = false`, and `dict` additionally with
`share_u_rank < 2` or `rank >= share_u_rank` (a width at or past the dictionary
has no subset to choose, so every expert would read the whole plane). Default
off is bit-identical: no parameter, no draw, no counter.

`factor_qat` rides the shared basis with ONE quantized copy per site: the
`[d, rank]` plane / `[d, share_u_rank]` dictionary gets its own single α
(`dim=None`, the per-TENSOR absmean) and `qat_values` projection, cached once
per optimizer step in the owner's `QuantCache` (`SparseMoE._qat_cache`, the
same per-step invalidation the per-expert factors use). The experts read that
projection — the plane whole, the dictionary through the seeded column gather
— so there is no second quantizer per expert and no double STE. The anneal is
`SparseMoE.factor_qat_alpha`, which `DormouseModel.set_factor_qat_alpha`
already walks to, so the basis and the per-expert factors move on ONE
schedule.

Dispatch has TWO arms over ONE permute, and the log line says which ran:

* **padded (CUDA, the fast arm)**: gather the kept rows into ONE zero-filled
  `[E, cap, d]` bank and run the whole expert bank as three batched matmuls
  (`bmm`) — one launch per GEMM instead of `3·E` small ones. This is the
  spike's phase-2 winner and the arm that survived the probe: at E=1024/b1 the
  padded bmm measured **21.4 ms fwd against `torch._grouped_mm`'s 44.2 ms**,
  and the ragged grouped op was REJECTED for the combined pass (its cost is
  ~19 µs PER GROUP, flat — 2× the whole padded bmm at that E; see
  `/tmp/opencode/dormouse-torch-spike/SPIKE_REPORT.md`, phase-2 probe table).
* **triton (research step 3/3, `model/moe_kernels.py`)**: the padding is gone —
  the caller gathers the same ragged rows expert-major and the unpadded grouped
  GEMM takes them with `offs [E+1]`, one program per (expert, row-block), the
  GLU fused into the second GEMM's input. `ModelConfig.moe_kernel` picks it and
  is the A/B kill-switch (`docs/research/2026-10-15-tsct-super-kernel.md` §5.5:
  expert-kernel verdicts come from isolated-pass benches, then the full step);
  when the kernel cannot run (no triton, not CUDA, a tile set the device cannot
  hold) the padded arm runs instead and `moe_triton_fallback` + the reason say
  so. The tile is DTYPE-AWARE (`moe_kernels.tiles_for`): fp32 halves the wide
  tiles and drops the pipeline to one stage, so the fp32 ByteFlow arm — which
  used to need 196608 B against a 101376 B ceiling and therefore ran the naive
  loop — takes the grouped GEMM. A set that still cannot fit is COUNTED with its
  byte numbers (`fp32 tile BLOCK_N=… needs … bytes > …`), never a silent switch.
  Its BACKWARD is FUSED by default (`fused_bwd=True`, `_triton_dispatch`): the
  scatter is `moe_kernels.combine` (one Triton kernel for both the `[R]`
  router-weight gradient and the expert adjoint — the OLMo-core combine-backward
  idea, reimplemented in `moe_kernels`), `dx` is one grouped GEMM with two
  accumulators instead of two plus `add_`, the row→expert map is an `int32`
  tensor the permute already had instead of a `repeat_interleave` rebuild, and
  the `[R, ·]` scratch is a grow-only workspace THIS module owns (one per
  `SparseMoE`) instead of ~12 allocations per backward. `fused_bwd=False` is the
  previous adjoint, kept as the parity gate and the bench's control; a top-k past
  `moe_kernels.FUSED_TOPK_MAX` is the one call the fused path refuses, and it
  says so on the counter line (`moe_triton_fallback`, reason `fused backward:
  …`) while `moe_triton_gemm` still counts — the grouped GEMM did run.
* **naive (the CPU path and the correctness reference)**: gather → per-expert
  matmul → scatter over a capacity-bound permute. The day-one implementation,
  kept because it is what the padded arm is compared against — the same rows,
  the same capacity rule, the same drop counter.

ONE more switch on the routed pass, OFF by default: **the gather-fused forward**
(`moe_gather = "kernel"`, `model/moe_sonic.py`, SonicMoE arXiv:2512.14080
§5.1.1). It is not a fourth `moe_kernel` — it is a second switch on the SAME
pass, and `moe_kernel` names the arm that runs when it cannot. With it on, the
grouped GEMM takes the routing's row map (`gather_abi`'s `toks`/`offs`) and reads
`x`'s rows itself, so the `xg = x[owner].to(dt)` line — an fp32 `index_select`
plus a bf16 cast, four `[R, d]` transfers where the fused load is one — does not
exist. The prototype's own analysis priced that as a FORWARD byte win and a
BACKWARD byte loss (`grad_out` is fp32 while `dy` is bf16); this wiring fuses the
forward ONLY and recomputes the shipped grouped adjoint on the same rows
(`moe_sonic._GatherGlu`), so its fwd→bwd cache is `x`, `toks` and `y`, and it
pays one extra forward per step. `scripts/bench_gather.py` measures both shapes
and decides; no speed claim exists until it runs. Counters: `moe_gather_forward`
(calls) and `moe_gather_rows` (rows read through the map) — a run labelled
`moe_gather="kernel"` without them did not run the arm. A call the kernel cannot
serve is COUNTED with its reason (`moe_gather_fallback` + `counters.why`) and
falls through to `moe_kernel`, exactly like the triton arm's own fallback.

ONE more switch on the routed pass, OFF by default: the **adaptive width**
(`moe_topk_mode`, `model/adaptive_topk.py`, 2026-10-21). `off` is the constant
`moe_topk` width every number on record runs and the module never calls the
policy at all; `entropy`/`margin` choose a per-token `k` in
`[moe_topk_min, moe_topk_max]` from the router logits alone (`off` = control).
The routed branch then masks the raised logits past each row's own `k`, keeps
`k_max` slots and gives the unused ones weight 0 — the experts, the permute and
the capacity rule are unchanged. A zero-weight slot is NOT an assignment: the
permute puts it in a sentinel group past the bank so it consumes no expert's
capacity and is dropped before any expert runs (`_permute`'s `valid`), which is
what makes the average width the compute that actually happens on the ragged
arms. Counters: `moe_k1`..`moe_k{max}` (per-token histogram, device
accumulators) and `moe_k_mean_x100` (the mean width × 100, an INT).

All of them count: `moe_padded_bmm`, `moe_triton_gemm`, `moe_gather_forward` or
`moe_naive_dispatch`
(with the reason attached to the counter object) on the step's counter line, so
a reader never has to guess which one produced the loss.

**`moe_capacity = 0` is DROPLESS** (2026-10-17, the segmented-training lane).
The capacity rule makes the DROP SET a function of the window's composition —
`keep = pos < max(k, ceil(capacity·tokens·k/E))` is a per-forward rank, so which
tokens get an expert depends on how many other tokens share their forward
(measured: 0.39 on the tiny readout at `capacity = 1.0` between a one-shot window
and the same window cut in four). A training forward that segments a long window
therefore changed the model, not just the memory, and `segment_len` refused the
pair. With `moe_capacity = 0` every assignment goes to its expert, the routing is
per position, and the segmented forward is the one-shot forward
(`tests/test_longctx.py`). Three consequences, each of them LOUD or COUNTED:

* the padded arm **refuses** (`_padded_dispatch`): a capacity-padded `[E, cap, ·]`
  bank has no shape without a cap, and sizing it to the largest expert would be
  an input-dependent bank — a different arm, not a flag;
* the arm that runs is the RAGGED one, and it is the GROUPED GEMM: its
  row-block grid takes a STATIC bound the host reads from a SHAPE
  (`moe_kernels._dropless_cap` — `R` rows in one group, the widest any expert
  can be) and every program EXITS on-device when its block is empty (two `offs`
  loads and a branch), so nothing reads `offs` back to the host (ADR-0018 rule
  2) and an empty expert costs a program launch instead of GEMM work. Before
  that early-exit the dropless route had to be the per-expert Python loop —
  correct, and ~1024 tiny matmuls per expert pass at `b1`'s E=1024;
* the reference loop stays as the COUNTED fallback: `moe_kernel=naive` (the
  config's own choice), an activation the fused kernel does not carry
  (`expert_act != silu`), a device without the grouped GEMM (CPU), or a build
  without triton — each with its reason on the counter object, plus
  `moe_triton_fallback` when the grouped GEMM was the config's ask;
* `_capacity` asserts, so no other arm can quietly read a capacity of 0 as "the
  floor k" and drop everything (the defect that would look like a working run).

One more thing the routed pass can do, OFF by default: `moe_offload` (SL1,
`docs/research/2026-10-16-budget-savers.md` §5). The MoE block owns the
activation peak, and the tensors that own it are the `[E, cap, d]` /
`[E, cap, f]` bank materializations plus the saved GLU gates; with the switch on,
the arm `_dispatch` picked runs inside `checkpoint(use_reentrant=False)` with
`torch.autograd.graph.saved_tensors_hooks` INSIDE it, so every activation the
pass saves is parked in PLAIN HOST RAM (pageable cpu tensors, pinned=False) and
handed back on the backward edge. Values are COPIED, never approximated, so loss
and gradients are bit-identical to the switch-off run — that parity is the gate
(`tests/test_offload_experts.py`), and `moe_offloaded` on the counter line is
what proves the arm ran (a zero means it did not).

THE TORCH CAVEATS this combination carries, all measured here on 2.13.0 (CPU
and CUDA) and pinned by `tests/test_offload_experts.py`:

1. **The hooks go INSIDE the checkpointed function.** Non-reentrant `checkpoint`
   is ITSELF a `saved_tensors_hooks` context, and only the INNERMOST pair is
   applied. A `with saved_tensors_hooks(...)` wrapped AROUND the `checkpoint()`
   call is SILENTLY INERT — checkpoint's hook is innermost, the pack hook fires
   zero times, nothing reaches host, and only the counter would claim otherwise.
   With the hooks inside and covering the whole pass, every save is captured by
   us, so the checkpoint's frame registers no holders and it does NOT recompute:
   the pass runs once and the saving is the host offload alone. With a partial
   scope the checkpoint recomputes and stays exact. Both are correct; the
   non-inert one ships, and the pack-hook spy in the test file keeps it that way.
2. **Unpack must return the tensor ON ITS ORIGINAL DEVICE** (the hook contract:
   equal in value, size, dtype *and device*). The copy back happens at the
   backward edge — a device temporary for exactly the node that needs it, which
   is what makes the peak fall.
3. **The parameter banks are packed too, and that buys nothing.** The pass's
   `bmm` operands are `_banks`' cached `[E, ·, ·]` stacks; they are saved by
   autograd like any other operand, so they ride the copy out and back — but
   their storage is the cache, which stays in VRAM either way. On a b1-shaped
   dispatch that is ~1.25 GB of traffic per pass for zero reclaimed bytes.
   `ponytail:` the hook copies whatever autograd saves; if the SL1 measurement
   says that traffic is the bill, the upgrade is to skip the tensors the cache
   already holds (the pack hook can identity-check them against `_bank_cache`).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.graph import saved_tensors_hooks
from torch.utils.checkpoint import checkpoint

from ..config import FACTOR_QATS, SHARE_U_MODES
from ..instrument import Counters, loud
from .act_quant import fake_quant_act
from .adaptive_topk import check_policy, choose_k, select_topk
from .qat_cache import QuantCache, qat_project_stacked
from .tsct import dense_qat_weight, factor_qat_project, init_u, mm_mixed

try:  # the Triton arm is an ABILITY, not a dependency
    from . import moe_kernels
except Exception as exc:  # noqa: BLE001 — the reason is COUNTED by `_dispatch`
    moe_kernels = None
    _TRITON_WHY: str | None = f"the triton kernels are not importable here: {exc}"
else:
    _TRITON_WHY = None

from . import moe_sonic  # the gather-fused forward (`moe_gather = "kernel"`)


def _bincount(x: torch.Tensor, minlength: int) -> torch.Tensor:
    """`torch.bincount` that survives a CUDA-graph capture.

    `torch.bincount` on CUDA is NOT capturable on torch 2.13: its kernel pulls a
    host-side value and the copy aborts the capture — measured on this box
    (2026-10-23), a bare `with torch.cuda.graph(g): torch.bincount(a,
    minlength=9)` raises "Cannot copy between CPU and CUDA tensors during CUDA
    graph capture unless the CPU tensor is pinned", and the same call inside
    `_padded_dispatch` is what made a full-step capture (`full_graph`) fail at
    every MoE shape. This is the same histogram without the host round trip:
    ONE scatter-add of ones, entirely on device.

    `minlength` must cover the input's whole range — every call site here counts
    expert ids whose sentinel is already `< minlength` — and an out-of-range id
    now fails the scatter LOUDLY instead of silently returning a longer array.
    """
    out = torch.zeros(minlength, dtype=torch.long, device=x.device)
    out.scatter_add_(0, x.to(torch.long), torch.ones_like(x, dtype=torch.long))
    return out


def expert_glu(
    act: str,
    g: torch.Tensor,
    u: torch.Tensor,
    *,
    beta1: float = 4.0,
    beta2: float = 25.0,
    c0: float = 0.6039096539628113,
) -> torch.Tensor:
    """The expert GLU's activation — ONE place, every torch path routes here.

    Formulas, each transcribed from its paper (never from another impl):

    - `silu` (the control, SwiGLU's gate): `silu(g) ⊙ u` — both factors
      unbounded, the overflow risk the K3 report names at scale.
    - `situ` — Kimi K3, arXiv:2607.24653 §2.3.2 Eq. 12:
      `[β1·tanh(g/β1) ⊙ σ(g)] ⊙ [β2·tanh(u/β2)]`, their β1 = 4 / β2 = 25.
      Matches silu to first order at the origin (β·tanh(z/β) = z + O(z³/β²))
      and is bounded: ‖·‖∞ ≤ β1·β2 = 100 (their Eq. 19). The smooth caps keep
      gradients alive away from saturation (their ablation vs hard clamping).
    - `memglu` — arXiv:2608.07323 Eq. 1: `ϕ(g) ⊙ u`, `ϕ(g) = c0·tanh(g)·sech(g)`,
      computed as `c0·t·sqrt(1−t²)` (t = tanh g) — identical in fp32 and it
      cannot overflow where `cosh` would at large |g|. `c0` is the authors'
      RMS-matched 9M-scale calibration (their appendix table): without it the
      closed-tail gate under-scales the gate by ~1.66x against silu's RMS.
    """
    if act == "silu":
        return F.silu(g) * u
    if act == "situ":
        gate = beta1 * torch.tanh(g / beta1) * torch.sigmoid(g)
        return gate * (beta2 * torch.tanh(u / beta2))
    if act == "memglu":
        t = torch.tanh(g)
        return (c0 * t * torch.sqrt((1.0 - t * t).clamp_min(0.0))) * u
    raise AssertionError(f"expert_act={act!r} reached the GLU with no formula")


class _OnHost:
    """A saved activation parked in host RAM by the offload pack hook.

    A CLASS, not the bare cpu tensor: the pack hook also sees values it must
    pass through untouched (a non-tensor), the unpack hook has to tell its own
    cargo from those, and the contract `saved_tensors_hooks` documents requires
    the unpacked tensor to come back on the ORIGINAL device — so the packed
    object carries it.
    """

    __slots__ = ("t", "device")

    def __init__(self, t: torch.Tensor, device: torch.device) -> None:
        self.t = t
        self.device = device


def _pack_activation(t: object) -> object:
    """`saved_tensors_hooks` pack: copy a saved activation to plain host RAM.

    `detach()` so the copy carries no graph of its own (the documented
    reference-cycle rule), `.to("cpu")` so it is a pageable host tensor — the
    copy is exact (same dtype, same bits), which is what makes the parity gate a
    bit-equality and not a bound. The original device travels with it.
    """
    if not isinstance(t, torch.Tensor):
        return t
    return _OnHost(t.detach().to("cpu"), t.device)


def _unpack_activation(h: object) -> object:
    """`saved_tensors_hooks` unpack: give the copy back ON ITS ORIGINAL DEVICE.

    The device leg is not optional and it is not decoration: the hook contract
    is "equal in value, size, dtype AND device", and a backward kernel handed a
    cpu tensor where the forward saved a cuda one dies on the device check
    (`index_select` refuses a cpu index, measured on 2.13.0). The copy back
    happens at the backward edge, so the tensor is a device temporary for
    exactly the node that needs it — which is the whole offload.
    """
    if not isinstance(h, _OnHost):
        return h
    return h.t if h.device.type == "cpu" else h.t.to(h.device)


def _refuse_shared_qat(factor_qat: str, where: str) -> None:
    """LOUD: a `factor_qat` rung the shared-basis quantizer cannot express.

    Kept as the seam for a FUTURE rung: the shared basis is projected with its
    own single α (`dim=None`), which every current mode supports, so no
    combination is refused today. A rung that needs a per-expert slice of the
    shared tensor (the one thing a shared basis cannot provide) belongs here,
    named, before the config can lie about it.
    """
    if factor_qat not in FACTOR_QATS:
        loud(
            f"share_u with factor_qat={factor_qat!r} ({where}): the shared basis "
            f"is projected with ONE α over the whole tensor, and this rung is not "
            f"in {list(FACTOR_QATS)}. Escape: name a known rung, or "
            f"factor_qat='none'."
        )


#: The seed the `dict` mode's column assignment is drawn from. FIXED on purpose:
#: the indices are a constant of the model, so two builds of one config see the
#: same subspaces, a resume reads the same ones, and `share_u_mode="off"` still
#: draws nothing from the global stream. It is not drawn from the run's `--seed`
#: because a seed that moves the architecture is a different model, not a
#: different run of the same one.
SHARE_U_DICT_SEED = 20_261_021


def _check_dict_index(idx: torch.Tensor, dict_rank: int, cols: int, where: str) -> None:
    """LOUD: a dictionary index that cannot gather `[d, dict_rank]` columns.

    The constructor generates the index, so this fires for a hand-built one (a
    test, a future resume that stores it) — never for silence. A wrong dtype is
    refused with the same weight as an out-of-range value: `index_select` would
    accept a `float` tensor by truncating it, which is a silent wrong subspace.
    """
    if idx.ndim != 3 or idx.shape[0] != 2 or idx.shape[2] != cols:
        loud(
            f"{where}: the share_u dictionary index must be [2, E, {cols}] "
            f"(gate row, up row, `rank` columns each), got {tuple(idx.shape)}. "
            f"Escape: build it with `share_u_dict_index`."
        )
    if idx.dtype != torch.long:
        loud(
            f"{where}: the share_u dictionary index must be int64, got {idx.dtype} "
            f"— a float index is silently truncated by `index_select`, i.e. a "
            f"different subspace than the one written down."
        )
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= dict_rank):
        loud(
            f"{where}: the share_u dictionary index is out of range for a "
            f"[d, {dict_rank}] dictionary: values [{int(idx.min())}, "
            f"{int(idx.max())}], legal [0, {dict_rank}). Escape: share_u_rank >= "
            f"rank and indices drawn from [0, share_u_rank)."
        )


def share_u_dict_index(
    n_experts: int, dict_rank: int, cols: int, *, seed: int = SHARE_U_DICT_SEED
) -> torch.Tensor:
    """`[2, E, cols]` column indices into the `dict` mode's `[d, dict_rank]` dictionary.

    Fixed and seeded (module docstring: a learned selector is a second
    mechanism). Each row names `cols` DISTINCT columns, and a row never repeats
    a column SET another row already read — the dictionary holds `C(R, cols)`
    distinct subspaces, so two experts see the same plane only once that pool is
    smaller than the bank (`2·E`, at E=1024/R=64/cols=2 it is by 32 rows; the
    repeat is then the arithmetic's, not a choice). An expert's gate row and up
    row are distinct sets too: with two identical sets the GLU is
    `silu(xB·s_g) ⊙ (xB·s_u)`, a rescaling of one subspace, exactly the
    dual-latent defect TSCT2 §13.3 exists to avoid. `cols < dict_rank` is
    checked by the caller (`config.validate` refuses `rank >= share_u_rank`,
    where there is no subset left to choose), so both redraws terminate.

    The draw is from a local `torch.Generator`, so it cannot move the global
    stream the model's init reads: `share_u_mode` changes the architecture, not
    the other parameters' values.
    """
    if dict_rank < 2 or not 1 <= cols < dict_rank:
        loud(
            f"share_u_dict_index with dict_rank={dict_rank} cols={cols}: the "
            f"dictionary needs dict_rank >= 2 and 1 <= cols < dict_rank (cols == "
            f"dict_rank leaves no subset to choose — that is `share_u_mode="
            f"'plane'`). Escape: share_u_rank > rank >= 1."
        )
    g = torch.Generator().manual_seed(seed)
    used: set[frozenset[int]] = set()

    def one_row(exclude: frozenset[int] | None) -> torch.Tensor:
        # 64 tries to land on an UNUSED subspace; at the pool's edge (E·2 close
        # to C(R, cols) or past it) the second loop accepts the repeat.
        for _ in range(64):
            row = torch.randint(0, dict_rank, (cols,), generator=g)
            cols_ = set(row.tolist())
            key = frozenset(cols_)
            if len(cols_) == cols and key not in used and key != exclude:
                used.add(key)
                return row
        while True:
            row = torch.randint(0, dict_rank, (cols,), generator=g)
            cols_ = set(row.tolist())
            if len(cols_) == cols and frozenset(cols_) != exclude:
                return row

    idx = torch.empty(2, n_experts, cols, dtype=torch.long)
    for e in range(n_experts):
        idx[0, e] = one_row(None)
        idx[1, e] = one_row(frozenset(idx[0, e].tolist()))
    _check_dict_index(idx, dict_rank, cols, "share_u_dict_index")
    return idx


class DualLatentGLU(nn.Module):
    """`x [n, d] → [n, f]`: two rank-`r` latents, the GLU at the latent."""

    def __init__(
        self,
        d: int,
        f: int,
        rank: int,
        *,
        dtype: torch.dtype,
        factor_qat: str = "none",
        factor_qat_alpha: float = 1.0,
        act_quant: str = "none",
        act_group: int = 0,
        share_u: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.d, self.f, self.rank = d, f, rank
        self.compute_dtype = dtype
        self.factor_qat = factor_qat
        self.factor_qat_alpha = factor_qat_alpha
        #: The once-per-step projection cache (`qat_cache.QuantCache`) for this
        #: expert's three factors on the naive/shared path. The BANKED path
        #: keeps its own per-step cache (`SparseMoE._banks`), so the two agree.
        self._qat_cache = QuantCache() if factor_qat != "none" else None
        #: `share_u` (owner directive 2026-10-21): the two INPUT bases are not
        #: this expert's. The parent `SparseMoE` owns ONE `[d, rank]` per site
        #: (`gate_u`, `up_u`) and binds it here (`bind_shared_u`); the expert
        #: keeps `s` (its own latent scale) and `V` (its own write). The bank's
        #: capacity then lives in a shared dictionary of input directions
        #: instead of in E interchangeable per-expert copies — the point of the
        #: arm, since at rank 2 a per-expert basis is just 2 directions of noise
        #: per expert. Default off = the per-expert basis, and the parameter set
        #: every snapshot on record carries, unchanged.
        self.share_u = share_u
        self._shared_u: tuple[torch.Tensor, torch.Tensor] | None = None
        #: `share_u` + `factor_qat`: the OWNER's per-step `QuantCache`. The
        #: shared basis is projected ONCE per site per step (`SparseMoE.
        #: _qat_cache`), and every expert — the banked arms and this standalone
        #: path — reads that one projection. `None` (a hand-bound expert) falls
        #: back to this module's own cache, which is the same arithmetic with
        #: one entry instead of the bank's one.
        self._shared_cache: QuantCache | None = None
        #: `dict` mode only: `[2, rank]` column indices into the parent's
        #: dictionary — row 0 the gate latent, row 1 the up latent. `None`
        #: under `plane`, where the bound pair IS the basis.
        self._shared_index: torch.Tensor | None = None
        if share_u:
            _refuse_shared_qat(factor_qat, "DualLatentGLU")
        #: The a4.8 ACTIVATION switch (`act_quant.fake_quant_act`): the `x` fed
        #: to this expert's factor matmuls is fake-quantized with an STE. Same
        #: switch the banked path in `SparseMoE` applies to its `x`, so one
        #: expert through the bank and the same expert alone quantize their
        #: activation identically (per-row/group scale, expert-independent).
        self.act_quant = act_quant
        self.act_group = act_group
        for name in ("gate", "up"):
            if not share_u:
                # `U [d, r]` and `s [r]`: the latent half of TSCT's factorization.
                setattr(self, f"{name}_u", nn.Parameter(init_u(d, rank, device)))
            setattr(self, f"{name}_s", nn.Parameter(torch.ones(rank, device=device)))
        # The write `V [r, f]` — one, at the latent, after the GLU.
        self.v = nn.Parameter(torch.randn(rank, f, device=device) / math.sqrt(rank))

    def bind_shared_u(
        self,
        gate_u: torch.Tensor,
        up_u: torch.Tensor,
        index: torch.Tensor | None = None,
        *,
        cache: QuantCache | None = None,
    ) -> None:
        """Take the parent's ONE shared basis — the only way `share_u` gets its `U`s.

        Not a constructor argument on purpose: the tensor is owned by `SparseMoE`
        (one `nn.Parameter` for the whole bank), so this module must hold a
        REFERENCE to it, never a copy or a second registration. A `share_u`
        expert with no binding refuses at `forward` (`_bases`) rather than
        inventing a basis — a silent private `U` would be a per-expert one
        wearing the shared arm's name.

        `index [2, rank]` is the `dict` mode's column subset: `gate_u`/`up_u` are
        then the same `[d, share_u_rank]` dictionary and each forward gathers its
        two rows. Gathered HERE, per forward, not at bind time — a bind-time
        gather would freeze step 0's columns and every later step would train a
        constant.

        `cache` is the OWNER's per-step `QuantCache` under `factor_qat`: the
        shared basis is projected by `SparseMoE` once per site per step (the
        same object the banked path reads), so binding the owner's cache is
        what makes "one quantized basis" true across the standalone and banked
        arms instead of one entry per expert.
        """
        assert self.share_u, "bind_shared_u on an expert that owns its own U"
        self._shared_u = (gate_u, up_u)
        self._shared_index = index
        self._shared_cache = cache

    def _bases(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The forward's `(gate_u, up_u)` — this expert's, or the bank's shared pair."""
        if not self.share_u:
            return self.gate_u, self.up_u
        if self._shared_u is None:
            loud(
                "share_u=True and no parent bound the shared basis: an expert that "
                "owns no `U` and was handed none has nothing to project through. "
                "Escape: build it under a SparseMoE with share_u on (it binds one "
                "`[d, rank]` per site), or share_u=false."
            )
        gate_u, up_u = self._shared_u
        if self.factor_qat != "none":
            # ONE quantized basis per site: project the WHOLE shared tensor (the
            # dictionary under `dict`) with its own single α, BEFORE the
            # per-expert column gather, through the owner's per-step cache. The
            # banked path (`SparseMoE._banks`) evaluates the same expression on
            # the same tensor, so a banked expert and this standalone path read
            # identical values. Quantizing the gathered slice instead would take
            # the absmean of `rank` columns of a `share_u_rank`-wide dictionary
            # — a second α `_banks` never applies (gate C of
            # `tests/test_share_u_qat.py`).
            q = factor_qat_project
            cache = self._shared_cache or self._qat_cache
            gate_u = q(gate_u, self.factor_qat, self.factor_qat_alpha, cache=cache)
            up_u = q(up_u, self.factor_qat, self.factor_qat_alpha, cache=cache)
        if self._shared_index is not None:
            gate_u = gate_u.index_select(1, self._shared_index[0])
            up_u = up_u.index_select(1, self._shared_index[1])
        return gate_u, up_u

    def forward(self, x: torch.Tensor, glu=None) -> torch.Tensor:
        dt = self.compute_dtype
        if self.act_quant != "none":
            # a4.8: the activation feeding the factor matmuls, quantized in the
            # dtype it arrives in (fp32 masters, cast to `dt` after) with an STE.
            x = fake_quant_act(x, self.act_quant, self.act_group)
        xu = x if x.dtype == dt else x.to(dt)
        gate_u, up_u = self._bases()
        v = self.v
        if self.factor_qat != "none":
            # Per-tensor α on a 2-D factor == per-expert α on that expert's
            # slice of the bank (`_banks`'s `dim=(1,2)`), so the naive/shared
            # path and the banked path quantize an expert IDENTICALLY. Under
            # `share_u` the basis was ALREADY projected by `_bases` (once per
            # site, on the owner's cache) — quantizing it here would be a
            # second α and a second STE, so only `v` is left.
            q = factor_qat_project
            cache = self._qat_cache
            if not self.share_u:
                gate_u = q(gate_u, self.factor_qat, self.factor_qat_alpha, cache=cache)
                up_u = q(up_u, self.factor_qat, self.factor_qat_alpha, cache=cache)
            v = q(v, self.factor_qat, self.factor_qat_alpha, cache=cache)
        g = (xu @ gate_u.to(dt)) * self.gate_s.to(dt)
        u = (xu @ up_u.to(dt)) * self.up_s.to(dt)
        h = F.silu(g) * u if glu is None else glu(g, u)
        y = h @ v.to(dt)
        return y if dt == torch.float32 else y.to(torch.float32)


class SparseMoE(nn.Module):
    """The expert bank + its router + the shared write.

    `moe_topk = 0` is the DENSE blend (every expert, softmax weights) — burn's
    default and the arm's own control; `moe_topk = k` is the sparse dispatch.
    Both share one router: the difference is whether all experts are computed.
    """

    def __init__(self, cfg, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        d, f = cfg.d_model, cfg.d_ffn
        dtype = torch.bfloat16 if cfg.bf16 else torch.float32
        self.d, self.f = d, f
        self.n_experts = cfg.n_experts
        self.topk = cfg.moe_topk
        #: The adaptive width (`model/adaptive_topk.py`): `off` is the constant
        #: `topk` above; a non-off mode runs the routed pass with `topk_max`
        #: slots of which each token activates its own `k`. The checks run HERE
        #: too because a hand-built config reaches this module without passing
        #: the loader — a bad policy is LOUD either way (ADR-0011).
        self.topk_mode = cfg.moe_topk_mode
        self.topk_min = cfg.moe_topk_min
        self.topk_max = cfg.moe_topk_max
        self.entropy_lo = cfg.moe_entropy_lo
        self.entropy_hi = cfg.moe_entropy_hi
        check_policy(self.topk_mode, self.topk_min, self.topk_max)
        if self.topk_mode != "off":
            if self.topk <= 0:
                loud(
                    f"moe_topk_mode={self.topk_mode!r} with moe_topk=0: the adaptive "
                    f"width lives on the ROUTED pass, and 0 is the dense blend. "
                    f"Escape: moe_topk > 0 (the width comes from moe_topk_max)."
                )
            if self.topk_max > self.n_experts:
                loud(
                    f"moe_topk_max={self.topk_max} > n_experts={self.n_experts}: a "
                    f"width above the bank cannot be filled. Escape: k_max <= n_experts."
                )
        self.capacity = cfg.moe_capacity
        self.moe_kernel = cfg.moe_kernel
        #: The gather-fused FORWARD (`model/moe_sonic.py`, SonicMoE §5.1.1): a
        #: second switch on the same pass. `moe_kernel` is what runs when this
        #: arm cannot (`_gather_reason` → COUNTED `moe_gather_fallback`), and
        #: `config.validate` refuses the combinations the arm has no evidence
        #: for. The arm's own counters are `moe_gather_forward` (calls) and
        #: `moe_gather_rows` (rows read through the map) — a log line without
        #: them did not run the arm it is labelled with.
        self.moe_gather = cfg.moe_gather
        self.moe_offload = cfg.moe_offload
        #: The compressed-forward sprint's factor quantization (`factor_qat`):
        #: `_banks` applies it to the fp32 stacks before the dt-conversion, and
        #: `DualLatentGLU` applies the SAME projection (dim=None on a 2-D
        #: factor == dim=(1,2) slice) on the naive/shared path.
        self.factor_qat = cfg.factor_qat
        self.factor_qat_alpha = cfg.factor_qat_alpha
        #: The DENSE path's switch (`tsct.dense_qat_project`, the SAME projector
        #: as the factor rung): the router and the shared `down` write are plain
        #: `nn.Linear`s, so they are read through the annealed STE. The
        #: experts' FACTORS stay on `factor_qat` (they are not weight matrices).
        self.dense_qat = cfg.dense_qat
        self.dense_qat_alpha = cfg.factor_qat_alpha
        #: The once-per-step projection cache for the two DENSE weights here
        #: (router, shared `down`) — the same object `qat_linear` reads on the
        #: plain linears — and, under `share_u` + `factor_qat`, for the ONE
        #: shared basis per site. `_banks` keeps its own cache for the expert
        #: stacks. Created whenever either arm needs it, so the trainer's
        #: per-step `invalidate_qat_cache_all` reaches the shared basis through
        #: this module's own `_qat_cache` (no second invalidation seam).
        self._qat_cache = (
            QuantCache()
            if (cfg.dense_qat != "none" or (cfg.share_u_mode != "off" and cfg.factor_qat != "none"))
            else None
        )
        #: The a4.8 ACTIVATION quantization (`act_quant.fake_quant_act`): the
        #: expert activations are fake-quantized with an STE — in the banked
        #: branches here (the factors are reached through `bmm`, not through
        #: `DualLatentGLU.forward`) and inside `DualLatentGLU` on the naive /
        #: dense-loop paths. Both quantize the same `x` per row/group, so a
        #: banked expert and the same expert alone agree.
        self.act_quant = cfg.act_quant
        self.act_group = cfg.act_group
        self.expert_act = cfg.expert_act
        self._glu = (
            None
            if cfg.expert_act == "silu"
            else (
                lambda g, u: expert_glu(
                    cfg.expert_act,
                    g,
                    u,
                    beta1=cfg.expert_beta1,
                    beta2=cfg.expert_beta2,
                    c0=cfg.memglu_c0,
                )
            )
        )
        #: The mode is resolved BEFORE the experts are built: `DualLatentGLU`'s
        #: own `share_u` flag says "I own no `U`, the parent binds one", which is
        #: a property of every non-`off` mode — not `cfg.share_u` (the legacy
        #: alias, which is `false` for a `dict` run that was spelled with
        #: `share_u_mode` alone).
        self.share_u_mode = cfg.share_u_mode
        if self.share_u_mode not in SHARE_U_MODES:
            loud(
                f"share_u_mode={self.share_u_mode!r}: known modes are "
                f"{list(SHARE_U_MODES)} (a hand-built config reaches this module "
                f"without the loader). Escape: name one of those."
            )
        self.share_u = self.share_u_mode != "off"
        self.experts = nn.ModuleList(
            DualLatentGLU(
                d,
                f,
                cfg.rank,
                dtype=dtype,
                factor_qat=cfg.factor_qat,
                factor_qat_alpha=cfg.factor_qat_alpha,
                act_quant=cfg.act_quant,
                act_group=cfg.act_group,
                share_u=self.share_u,
                device=device,
            )
            for _ in range(cfg.n_experts)
        )
        self.shared = DualLatentGLU(
            d,
            f,
            cfg.shared_rank,
            dtype=dtype,
            factor_qat=cfg.factor_qat,
            factor_qat_alpha=cfg.factor_qat_alpha,
            act_quant=cfg.act_quant,
            act_group=cfg.act_group,
            device=device,
        )
        #: `share_u` (owner directive 2026-10-21): the routed experts' INPUT
        #: basis is not their own. `plane` = ONE input basis per ROUTED site,
        #: owned HERE and bound into every routed expert; `dict` = ONE
        #: dictionary `[d, share_u_rank]` and a per-expert column subset. The
        #: vocabulary is TSCT's (`u [d, r]`, `s [r]`, `v [r, f]`), so this is the
        #: same factorization with E per-expert copies of `U` replaced by one
        #: shared dictionary — the direction the bank's capacity can live in when
        #: the rank is too small for a private basis to carry anything. Module
        #: docstring has the two modes, their arithmetic and why the `dict`
        #: indices are fixed.
        #:
        #: Init is `tsct.init_u` — the same draw a per-expert `u` gets
        #: (`N(0, 1/sqrt(d))`), so the arm changes HOW MANY copies exist, not
        #: the scale they start at (a different init would be a second change
        #: charged to the same A/B).
        #:
        #: The ALWAYS-ON shared expert is deliberately NOT bound: it runs at
        #: `shared_rank`, so a `[d, rank]` basis is the wrong shape for it, and
        #: it is one expert (E = 1) — there is nothing to share with. Its own
        #: basis is one copy already.
        #: The tensors the bank cache must key on beyond `self.experts`' own
        #: parameters (`_banks`): one basis per site under `plane`, the one
        #: dictionary under `dict`. Empty when off.
        self._shared_params: tuple[nn.Parameter, ...] = ()
        if self.share_u:
            _refuse_shared_qat(self.factor_qat, "SparseMoE")
            if not cfg.use_tsct:
                loud(
                    f"share_u_mode={self.share_u_mode!r} with use_tsct=false: the "
                    f"shared basis is a TSCT input factor, and a dense projection "
                    f"has no factor basis to share. Escape: use_tsct=true, or "
                    f"share_u_mode='off'."
                )
            if self.share_u_mode == "dict":
                if cfg.share_u_rank < 2:
                    loud(
                        f"share_u_mode='dict' with share_u_rank={cfg.share_u_rank}: "
                        f"two columns are the minimum for a plane. Escape: "
                        f"share_u_rank >= 2."
                    )
                if cfg.rank >= cfg.share_u_rank:
                    loud(
                        f"share_u_mode='dict' with rank={cfg.rank} >= "
                        f"share_u_rank={cfg.share_u_rank}: each expert reads `rank` "
                        f"columns of the dictionary; at or past its width there is "
                        f"no subset to choose and the index could read out of "
                        f"range. Escape: share_u_rank > rank."
                    )
                self.u_dict = nn.Parameter(init_u(d, cfg.share_u_rank, device))
                #: FIXED, SEEDED, NOT LEARNED (`share_u_dict_index`): a discrete
                #: column choice has no gradient, so a selector would be a second
                #: mechanism; this arm asks whether per-expert subspaces help at
                #: all. A buffer, not a parameter, and derived from (E, R, rank)
                #: — non-persistent because a resume rebuilds it identically.
                #: (`lfb_bias` is non-persistent for the opposite reason: it is
                #: NOT derivable, so it travels in the checkpoint's `moe` section.)
                self.register_buffer(
                    "share_u_idx",
                    share_u_dict_index(cfg.n_experts, cfg.share_u_rank, cfg.rank),
                    persistent=False,
                )
                for e, row in zip(self.experts, self.share_u_idx.permute(1, 0, 2), strict=True):
                    e.bind_shared_u(self.u_dict, self.u_dict, row, cache=self._qat_cache)
                self._shared_params = (self.u_dict,)
            else:
                self.gate_u = nn.Parameter(init_u(d, cfg.rank, device))
                self.up_u = nn.Parameter(init_u(d, cfg.rank, device))
                for e in self.experts:
                    e.bind_shared_u(self.gate_u, self.up_u, cache=self._qat_cache)
                self._shared_params = (self.gate_u, self.up_u)
        # The router: ONE 2-D linear on the raw hidden state (not on the
        # latent), softmax over the bank, then top-k + a capacity-bound permute.
        self.router = nn.Linear(d, cfg.n_experts, bias=False, device=device)
        # The LFB routing bias (arXiv:2408.15664): a per-expert `b_e` added to
        # the softmax probs before the top-k, moved after each routed forward
        # by `-coef·sign(load_e − mean)` WITHOUT gradients. A buffer, never a
        # parameter: the optimizer must not move it, LFB does.
        # Non-persistent because `state_dict()` is the optimizer/parameter view;
        # `save_ckpt` writes it per module in the checkpoint's `moe` section and
        # `load_model_state` restores it. Unlike `lfb_counts` below it is NOT
        # derivable from anything else, so a resume that lost it would restart
        # the controller at zero — a different router under the same step count.
        self.lfb_coef = cfg.moe_lb_coef
        self.register_buffer(
            "lfb_bias", torch.zeros(cfg.n_experts, device=device), persistent=False
        )
        # The cumulative routed-count integrator the bias reads (persistent:
        # a resume must keep its controller state or the rebalancer restarts
        # its memory of where the load went). This is the input; the bias above
        # is the output and cannot be reconstructed from it.
        self.register_buffer(
            "lfb_counts", torch.zeros(cfg.n_experts, device=device), persistent=True
        )
        # The shared write back to the residual width.
        self.down = nn.Linear(f, d, bias=False, device=device)

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        counters = Counters() if counters is None else counters
        if self.share_u_mode == "plane":
            # The shared-basis arm's evidence line (ADR-0019): `_banks` owns no
            # counters and the routed / dense arms reach the basis through
            # different code paths, so the ONE place every forward passes is
            # here. A `share_u=true` run whose log has no `share_u_forward` did
            # not run the arm it is labelled with. `dict` has its own counter and
            # the two are mutually exclusive — a log carrying both is two MoEs or
            # a wiring error.
            counters.count("share_u_forward")
        elif self.share_u_mode == "dict":
            counters.count("share_u_dict_forward")
        if self.share_u and self.factor_qat != "none":
            # Evidence that the quantized shared basis actually ran (ADR-0019):
            # absent under `factor_qat="none"`, so a log line cannot claim the
            # pair from a model whose rung was off. Once per forward, not per
            # site — the per-SITE cost is `qat_refresh_calls` (one per basis),
            # and that is the counter the once-per-step gate reads.
            counters.count("share_u_qat_forward")
        n = x.shape[0]
        assert x.ndim == 2 and x.shape[1] == self.d, f"MoE input {tuple(x.shape)} != [n, {self.d}]"
        dt = self.experts[0].compute_dtype  # the shared compute dtype
        # `mm_mixed` returns `dt`, NOT fp32: with bf16 on, `logits` — and so the
        # softmax `probs` and the routed `weight` — are bf16. The combine
        # boundary widens the weight back to fp32 (`_triton_dispatch`).
        # `dense_qat_project` is the identity when the arm is off.
        logits = mm_mixed(
            x,
            dense_qat_weight(
                self.router.weight,
                self.dense_qat,
                self.dense_qat_alpha,
                cache=self._qat_cache,
            ),
            dt,
            counters,
        )
        probs = F.softmax(logits, dim=-1)  # [n, E]
        # a4.8: the activation quantization arm (COUNTED — a reader of the log
        # line sees it) applies to the EXPERTS' activations only. The router
        # above reads the raw state on purpose (`router` is a dense `nn.Linear`,
        # no factor matmul, and the activation quantizer must not steer
        # routing); its WEIGHT is a `dense_qat` target like every other dense
        # weight, which is the owner's ternary-everywhere rule, not a second
        # activation quantizer.
        if self.act_quant != "none":
            x = fake_quant_act(x, self.act_quant, self.act_group)
            counters.count("act_quant_arms", n)
        weighted = torch.zeros(n, self.f, dtype=torch.float32, device=x.device)
        if self.moe_offload and self.topk == 0:
            # The offload switch is the ROUTED pass's (the capacity-padded bank
            # it materializes is what it exists for); a dense blend has no such
            # bank. Counted, never silent: an arm asked for and not run is the
            # thing the counters exist to make visible (ADR-0019).
            counters.count("moe_offload_inert_dense")
        if self.topk == 0:
            # Dense arm via ONE banked bmm per projection: the per-expert loop
            # was 24 sequential launches per forward at small (launch-bound,
            # measured 787 → 57 ms/step when banked 2026-10-15). The bank
            # carries E·n·d activations — budget-guarded, COUNTED fallback.
            e = self.n_experts
            budget = e * n * self.d * 2 <= (1 << 28)
            if budget:
                gate_u, up_u, v, gate_s, up_s = self._banks(dt)
                xg = x.to(dt).unsqueeze(0).expand(e, -1, -1)  # view, not copy
                g = torch.bmm(xg, gate_u) * gate_s[:, None, :]
                u = torch.bmm(xg, up_u) * up_s[:, None, :]
                h = F.silu(g) * u if self._glu is None else self._glu(g, u)
                y = torch.bmm(h, v)  # [E, n, f]
                weighted = torch.einsum(
                    "ne,enf->nf", probs, y.to(probs.dtype)
                )  # einsum keeps both operands in declared dtype (its own dispatch)
                counters.count("moe_dense_banked", n)
            else:
                for ex, expert in enumerate(self.experts):
                    weighted += expert(x, glu=self._glu) * probs[:, ex : ex + 1]
                counters.count("moe_dense_loop", n)  # the fallback, read in logs
        else:
            # LFB (arXiv:2408.15664): the top-k reads the RAISED logits
            # `logits + b_e` (the paper's bias is on the pre-softmax score).
            # The A/B caught BIASES ON PROBS flooding the bank: a ±0.4
            # cumulative bias swamps a 1/E = 0.015-range prob and every token
            # runs to the same two experts (67,459 drops vs 42,065). In logit
            # space the bias competes with logits' own scale; the gate weight
            # stays the honest prob from the UNbiased softmax (the bias
            # steers selection, never magnitude).
            raised = logits + self.lfb_bias
            if self.topk_mode == "off":
                # The control, unchanged: ONE global width, no policy call at all
                # (the off-arm bit-parity gate pins this branch against a digest
                # measured before the policy existed).
                index = raised.topk(self.topk, dim=-1).indices
                weight = torch.gather(probs, 1, index)
                self._lfb_step(index, counters)
            else:
                # The adaptive width: the policy sees the UNbiased logits (the
                # bias steers selection, never the confidence that picks k), the
                # mask is applied to the RAISED scores so LFB still selects, and
                # the weights come from the unbiased softmax, zeroed on the
                # padding slots.
                k = choose_k(
                    logits,
                    mode=self.topk_mode,
                    k_min=self.topk_min,
                    k_max=self.topk_max,
                    entropy_lo=self.entropy_lo,
                    entropy_hi=self.entropy_hi,
                )
                index, active = select_topk(raised, k, self.topk_max)
                self._k_counters(k, counters)
                # LFB reads REAL assignments: a weight-0 padding slot is not a load.
                self._lfb_step(index[active], counters)
                weight = torch.gather(probs, 1, index) * active.to(probs.dtype)
            # Renormalized (burn's `topk_blend`): a top-1 token's gate is
            # exactly 1.0, so the sparse arm and the dense control run at the
            # same output magnitude — the A/B compares specialization, not scale.
            weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            weighted = self._dispatch(x, weight, index, counters)
        # The shared expert is ALWAYS on: it is the write path, not a routed arm.
        weighted = weighted + self.shared(x, glu=self._glu)
        counters.count("moe_shared_forwards", n)
        # down: an fp32-master Linear; bf16 matmul + fp32 out (MAGMA fix).
        return mm_mixed(
            weighted.to(dt),
            dense_qat_weight(
                self.down.weight,
                self.dense_qat,
                self.dense_qat_alpha,
                cache=self._qat_cache,
            ),
            dt,
            counters,
        ).to(torch.float32)

    def _lfb_step(self, index: torch.Tensor, counters: Counters) -> None:
        """The LFB bias move: on-device, gradient-free, trains-mode only.

        `b_e -= coef · sign(L_e − mean L)` over the CUMULATIVE routed counts
        `L_e` — the integrator the paper specifies. The A/B of 2026-10-15
        caught the per-forward variant making drops WORSE (42,258→45,325 at
        cap 32): bang-bang control at one-step lag oscillates the whole
        bank; the cumulative mean is what converges. A MEASURED correction
        to the first implementation, not an opinion.
        """
        if self.lfb_coef == 0.0 or not self.training:
            return
        with torch.no_grad():
            counts = _bincount(index.reshape(-1), self.n_experts).float()
            self.lfb_counts.add_(counts)
            self.lfb_bias.sub_(self.lfb_coef * torch.sign(self.lfb_counts - self.lfb_counts.mean()))
        counters.count("moe_lfb_updates")

    def _k_counters(self, k: torch.Tensor, counters: Counters) -> None:
        """Per-token width histogram + mean, DEVICE accumulators (no host read).

        `moe_k{i}` is how many tokens ran exactly `i` experts; the histogram
        sums to the routed token count (the gate). `moe_k_mean_x100` is the
        mean width × 100 as an INT — the instrument counts integers, and an int
        counter cannot drift or print `2.0000000001`. Both live on the device
        lane and are drained once on the log cadence (`Counters.count_device`).
        """
        for width in range(self.topk_min, self.topk_max + 1):
            counters.count_device(f"moe_k{width}", (k == width).sum())
        counters.count_device("moe_k_mean_x100", (k.sum() * 100) // k.numel())

    def _capacity(self, n: int, k: int) -> int:
        """The routing capacity rule, in ONE place: `max(k, ceil(capacity·n·k/E))`.

        Burn's `[E, cap, f]` rule, kept exact, and floored at `k` so a token is
        never dropped by the floor itself.

        `capacity = 0` is DROPLESS and has NO cap: the floor would return `k`,
        i.e. "keep the first k rows of every expert and drop the rest" — the
        silent-collapse shape this assert exists for. The dropless path does not
        call this at all (`_permute` keeps every row).
        """
        assert self.capacity > 0, (
            f"capacity={self.capacity} is the DROPLESS mode and has no cap: the rule "
            f"would return the floor k and drop every row past it. Escape: "
            f"moe_capacity > 0, or the dropless path (which keeps every assignment)."
        )
        return max(k, math.ceil(self.capacity * n * k / self.n_experts))

    def _permute(
        self, index: torch.Tensor, valid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The routing permute: `(order, keep, dropped)` — one place, both arms.

        Both arms are permuted by the SAME rows and drop the SAME ones, which is
        what makes a parity run comparable and the drop count one number rather
        than two opinions. `dropped` STAYS A DEVICE TENSOR (a 0-dim int64
        counter the caller's `Counters.count_device` accumulates): the early
        form returned `int((~keep).sum())`, a per-forward GPU→CPU sync — one
        of exactly two places that blocked torch.compile(dynamic) in the
        arm-level probe (the other is the drain cadence itself, instrument.py).

        `valid` is the adaptive width's slot mask (`[n, k]`, True = a real
        assignment). A False slot is PADDING, not a droppable assignment: it is
        moved to a sentinel expert id PAST the bank (`self.n_experts`), where it
        sorts last, occupies no expert's capacity and is never kept — so the
        real assignments of a token see the same `pos` they would if the width
        had been its own `k`, and `dropped` counts only real rows. `valid=None`
        is the constant-width path, unchanged (`masked_fill` is not even built).

        `moe_capacity = 0` (dropless) keeps EVERY real row: `keep_cap` is all-True
        and `dropped` is 0, so the ragged arms see the full assignment set. That
        is what makes the drop set independent of the window's composition, and
        it is the whole reason this mode exists (module docstring).
        """
        n, k = index.shape
        flat_ids = index.reshape(-1)
        if valid is None:
            active = None
            safe = flat_ids
            groups = self.n_experts
        else:
            active = valid.reshape(-1).to(torch.bool)
            safe = flat_ids.masked_fill(~active, self.n_experts)
            groups = self.n_experts + 1
        order = torch.argsort(safe, stable=True)
        sorted_ids = safe[order]
        counts = _bincount(sorted_ids, groups)
        offsets = torch.cumsum(counts, 0) - counts
        pos = torch.arange(n * k, device=index.device) - offsets[sorted_ids]
        if self.capacity == 0:
            keep_cap = torch.ones_like(pos, dtype=torch.bool)
        else:
            keep_cap = pos < self._capacity(n, k)
        if active is None:
            keep = keep_cap
            dropped = (~keep).sum().view(())
        else:
            active_sorted = active[order]
            keep = keep_cap & active_sorted
            dropped = (active_sorted & ~keep_cap).sum().view(())
        return order, keep, dropped

    def _padded_reason(self, x: torch.Tensor) -> str | None:
        """`None` if the padded bank can run THIS call, else the reason why not.

        The padded arm needs no ABI: it is plain `bmm` over a bank the forward
        builds, so the only reason left is the device (`bmm` on fp32-CPU is the
        slow reference path, which is what the naive arm is for). The fallback
        is COUNTED, not silent (ADR-0019) — a reader of the log line sees it.
        """
        if not x.is_cuda:
            return f"input is on {x.device.type}: the padded bank is the CUDA path"
        return None

    def _bank_dtype(self) -> torch.dtype:
        """The dtype the experts compute in (their own `bf16` switch, not the device's)."""
        return self.experts[0].compute_dtype

    def _gather_rows(self, x: torch.Tensor, owner: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
        """`x[owner]` in `dt` — one gather copy, and a cast ONLY when one is needed.

        The loop block casts the MoE input to `dt` at its seam
        (`loopblock.LoopBlock._moe_branch`), so in the model's forward this is
        the `index_select` alone: 2·R·d written and 2·R·d read by the kernel,
        instead of the 12·R·d four-transfer chain an fp32 `x` paid (fp32 copy,
        cast read, bf16 write, kernel read). The dtype is ASSERTED, not
        coerced: a third dtype means the seam and the bank disagree, and
        casting it away would hide exactly that in the same numbers.

        `fp32` is the one other accepted input — a direct API caller
        (`SparseMoE(x_fp32)` without the loop block: every fixture, bench and
        test) keeps the old fp32 → `dt` cast, deliberate and counted by the
        caller's own byte model (`moe_sonic.pass_bytes(x_bytes=4)`).
        """
        assert x.dtype in (dt, torch.float32), (
            f"the routed gather expects the seam's {dt} (or fp32 for a direct "
            f"caller), got {x.dtype}: a third dtype is a wiring error, not a cast"
        )
        xg = x[owner]
        return xg if xg.dtype == dt else xg.to(dt)

    def invalidate_banks(self) -> None:
        """Drop the bank cache so the NEXT forward rebuilds it — the opt_graph fix.

        The cache key carries the parameters' `_version` tuple, and that assumed
        "an optimizer step bumps `_version`". **A CUDA-graph replay does not**
        (measured 2026-10-19: an eager step moves 0->1; the same step replayed
        from a captured graph leaves `_version` FROZEN while the value moves).
        With `opt_graph = true` the key therefore stopped moving at capture, the
        bank served the step-5 build for the whole run — the experts' FORWARD was
        frozen while the optimizer kept stepping them with stale gradients (they
        drifted ~44x; the saved checkpoint then exploded in any fresh process,
        and the live evals could not show it because they reused the same stale
        bank). The trainer now calls this once per TRAINING step, right after the
        parameters move: the cache keeps its real job (one build per step, reused
        by the loop's iterations) and can no longer outlive a step.
        """
        self._bank_cache = None

    @torch.compiler.disable
    def _banks(self, dt: torch.dtype) -> tuple[torch.Tensor, ...]:
        """The expert bank as `[E, ·]` tensors for the batched matmuls — CACHED IN `dt`.

        Two caches' lessons paid here. (1) The naive form stacked per forward
        (1.25 GB at b1). (2) The first cache stored the fp32 stack and called
        `.to(dt)` EVERY forward — measured ~15 ms host + ~4 ms copies per
        call at b1 (1.79 GB re-converted twice per step at max_iter 2; the
        TSCT-super-kernel bench flagged it as larger than the Triton kernel
        win). So this cache stores the CONVERTED tensors, keyed by the
        parameters' in-place `_version`: an optimizer step bumps it, nothing
        stale survives; one conversion per step is the honest floor, zero
        conversions inside the step.

        **`_version` alone was a one-step zero-gradient bug** (battery
        2026-10-17, measured 20/20 dead on both adjoints). The eval forward runs
        on the SAME model under `torch.no_grad()` (`train/loop.py`), and an
        optimizer step has just bumped `_version`, so the eval is exactly the
        call that REBUILDS the entry — with `requires_grad=False` stacks. The
        next training step's key then matches, the bank is read as a constant,
        and every expert parameter comes back with `.grad is None` until some
        other `_version` bump happens to rebuild it. So the grad state is part
        of the key, a no-grad build is never STORED, and a stored entry that
        cannot serve a tracked call is stale whatever the key says.

        **`torch.compiler.disable` keeps this method — and only the cache
        DECISION — out of dynamo's compiled regions.** The key reads per-step
        Python values (`factor_qat_alpha` moves on every annealed step,
        `_version` on every optimizer step) and dynamo guards a compiled region
        on every Python value it reads, so without the decorator the frame spent
        one recompile per step until it hit `config.recompile_limit` and fell
        back to eager (measured 2026-10-18, `docs/research/2026-10-18-compile-moe.md`;
        the trainer wraps the model with `torch.compile(dynamic=True)`,
        `train/loop.py:626`). The three call sites consume the bank tensors as
        ordinary inputs, so the compiled region is unchanged; the DECISION is
        unchanged too — the bank still rebuilds whenever the key moves
        (`tests/test_compile_moe.py` pins both halves, CPU and GPU).

        **Under `share_u`** the two input stacks are the SHARED basis
        materialized E-fold (`_one`), so every arm — padded, grouped GEMMs,
        reference loop — takes the same `[E, d, r]` operands as before and only
        the stored parameter count changes. `plane` expands one pair of
        `[d, rank]` per site; `dict` gathers each expert's `rank` columns out of
        the one `[d, share_u_rank]` dictionary (`u.t()[idx[row]]`), which is a
        differentiable gather — the backward sums every expert's contribution
        into the columns it read, and leaves the rest of the dictionary with no
        gradient from that expert. The shared tensors are not inside
        `self.experts`, so their `_version`s are part of the key explicitly.
        """
        grad = torch.is_grad_enabled()
        # `want` is what the cached bank must carry a graph FOR: a frozen expert
        # bank cannot, and keying on `is_grad_enabled()` alone would rebuild the
        # same untracked stack on every forward of a frozen model. Under
        # `share_u` the experts have no `u` of their own, so the shared basis is
        # the tensor whose grad state the cache must be able to serve.
        want = (
            grad
            and (
                self._shared_params[0] if self._shared_params else self.experts[0].gate_u
            ).requires_grad
        )
        key = (
            dt,
            self.factor_qat,
            self.factor_qat_alpha,  # the anneal ramp moves this every step
            tuple(p._version for e in self.experts for p in e.parameters()),
            # The shared basis is not inside `self.experts`, so its `_version`
            # is not in the tuple above: without this an optimizer step that
            # moved ONLY `u_dict` (or `gate_u`/`up_u`) would serve a stale bank.
            # (The trainer's `invalidate_banks` also covers that, but the cache
            # must not depend on a caller for its own correctness.)
            tuple(p._version for p in self._shared_params),
            want,
        )
        cached = getattr(self, "_bank_cache", None)
        if cached is not None:
            # A TRACKED entry serves a no-grad forward as it stands — same
            # tensors, no gradient wanted — so the eval keeps the cache hit. The
            # reverse (an untracked entry serving a tracked forward) is the
            # defect above, and it is what the key and the second clause forbid.
            reusable = cached[0] == key or (
                not grad and cached[0][:-1] == key[:-1] and cached[0][-1]
            )
            if not reusable or (want and not cached[1][0].requires_grad):
                cached = None
        if cached is None:
            # The stack is built PER EXPERT, converted to `dt` BEFORE the stack
            # (audit 2026-10-17, MAJOR 7): stacking in fp32 first costs a
            # 1.43 GB transient per name at b1 (`[1024, 5461, 64]`) and that
            # allocation is what OOM'd both b1 warm attempts and the compile
            # warm-up. Per-expert, the fp32 footprint is one expert (5.6 MB).
            # Scales (`gate_s`/`up_s`) are never quantized: they ARE the
            # diag(s) of the factorization.
            #
            # THE QAT ARM IS THE ONE EXCEPTION, on purpose: `factor_qat` needs
            # the whole `[E, ·, ·]` stack in fp32 to project it in ONE call
            # (`qat_cache.qat_project_stacked`) — the old loop ran the quantizer
            # E times per name (768 chains per step at medium, E=256), which is
            # launch-bound work for the same 24.5 M elements. The transient is
            # paid once per name per step; the non-QAT path every number on
            # record runs is untouched, and the QAT arm's b1 peak is re-measured
            # by `scripts/bench_qat_cost.py` before that shape is trusted.
            def _one(name: str) -> torch.Tensor:
                if self.share_u_mode == "dict" and name in ("gate_u", "up_u"):
                    # The bank's ONE dictionary, gathered into the [E, d, rank]
                    # stack the arms (and the triton ABI) take. `t()[idx]` picks
                    # each expert's columns; `.contiguous()` is a differentiable
                    # copy, so the backward sums every expert's contribution into
                    # the columns it READ and the rest stay at zero — which is
                    # what "the dictionary sees every expert that used it" means
                    # in code. Only the VALUES are E-fold; the parameter is one.
                    #
                    # Under `factor_qat` the DICTIONARY is projected whole, with
                    # its own single α, BEFORE the gather; the second `_one` call
                    # is a cache hit, so the site costs ONE quantizer call per
                    # step (not one per expert, and not one per expert-row).
                    row = 0 if name == "gate_u" else 1
                    d = self.u_dict
                    if self.factor_qat != "none":
                        d = factor_qat_project(
                            d,
                            self.factor_qat,
                            self.factor_qat_alpha,
                            cache=self._qat_cache,
                        )
                    cols = d.to(dt).t()[self.share_u_idx[row]]  # [E, r, d]
                    return cols.permute(0, 2, 1).contiguous()  # [E, d, r]
                if self.share_u_mode == "plane" and name in ("gate_u", "up_u"):
                    # The bank's ONE basis, materialized as the [E, d, r] stack
                    # the arms (and the triton ABI) take. `.contiguous()` is a
                    # differentiable copy, so the backward sums every expert's
                    # contribution into the single parameter — which is what
                    # "the shared U sees every expert that used it" means in
                    # code. Only the VALUES are E-fold; the parameter is one.
                    #
                    # Under `factor_qat` this is the SAME per-tensor projection
                    # `DualLatentGLU._bases` applies on the standalone path
                    # (one α over the whole `[d, rank]`), cached once per step
                    # through the owner's `QuantCache`, so the E experts read a
                    # basis that was quantized ONCE, not E times.
                    u = getattr(self, name)
                    if self.factor_qat != "none":
                        u = factor_qat_project(
                            u,
                            self.factor_qat,
                            self.factor_qat_alpha,
                            cache=self._qat_cache,
                        )
                    u = u.to(dt)
                    return u.unsqueeze(0).expand(len(self.experts), -1, -1).contiguous()
                params = [getattr(e, name) for e in self.experts]
                if self.factor_qat == "none":
                    return torch.stack([t.to(dt) for t in params])
                return qat_project_stacked(
                    torch.stack(params), self.factor_qat, self.factor_qat_alpha, dim=(1, 2)
                ).to(dt)

            mats = [_one(name) for name in ("gate_u", "up_u", "v")]
            cached = (
                key,
                tuple(mats),
                tuple(
                    torch.stack([getattr(e, name) for e in self.experts]).to(dt)
                    for name in ("gate_s", "up_s")
                ),
            )
            if want:
                self._bank_cache = cached
        mats, scales = cached[1], cached[2]
        return (mats[0], mats[1], mats[2], scales[0], scales[1])

    def _padded_dispatch(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        order: torch.Tensor,
        keep: torch.Tensor | None,
    ) -> torch.Tensor:
        """Gather → ONE capacity-padded `[E, cap, ·]` bank pass → weighted scatter.

        The spike's `moe_grouped`, adapted to the dual-latent GLU expert: the
        same math the naive path runs per expert, batched over the bank. Rows
        past capacity never enter the bank, and the ones that do are scattered
        back by the same `keep` mask, so this arm changes how the arithmetic is
        laid out, never which tokens get an expert.

        `moe_capacity = 0` (dropless) is REFUSED here: this arm IS a
        `[E, cap, ·]` bank and dropless has no cap to size it. A bank sized to
        the largest expert would be an input-dependent arm (a new mechanism, not
        a flag), and the rule's floor `k` would silently drop every row past it.
        `_dispatch` routes the dropless mode to the ragged arm instead; this
        guard is the seam for a direct call.
        """
        if self.capacity == 0:
            loud(
                "the padded MoE arm with moe_capacity=0: this arm materializes a "
                "[E, cap, ·] bank and dropless has no cap (the rule's floor would be "
                "k, which drops every row past it — a silently wrong forward). "
                "Escape: moe_capacity > 0 for the padded kernel, or the dropless "
                "route, which is moe_kernel=naive (the ragged arms keep every "
                "assignment)."
            )
        n, k = index.shape
        e, cap = self.n_experts, self._capacity(n, k)
        dt = self._bank_dtype()
        flat_w = weight.reshape(-1)
        flat_e = index.reshape(-1)
        if self.topk_mode == "off":
            counts = _bincount(flat_e, e)
        else:
            # The adaptive width's padding slots go to the SAME sentinel group
            # `_permute` used, or `pos` here would count them and disagree with
            # the permute's `keep` about which padded slot a real row owns
            # (a mismatch is an out-of-bounds `index_put`). The sentinel rows are
            # never kept, so their (large) slot numbers are computed, not used.
            flat_e = flat_e.masked_fill(~(weight > 0).reshape(-1), e)
            counts = _bincount(flat_e, e + 1)
        # EVERY assignment keeps its row here; the cap moves into a MASK. A
        # `rows = order[keep]` compression is `nonzero` on the device, i.e. a
        # D2H read and a row count that changes with the routing — neither is
        # legal in a CUDA graph (measured: `t[mask]` aborts a capture, and a
        # graph replays the FIRST call's shapes). `n · k` assignments is the
        # static bound, and a dropped row is routed to one dump slot that no
        # kept row can own (`e·cap`) and multiplied by the constant 0 mask, so
        # it contributes exactly 0.0 to the scatter and exactly 0 to both
        # gradients — the eager semantics, not an approximation of them.
        owner = order // k  # the token each assignment belongs to
        # The padded-bank slot of each sorted row: expert-major, capacity-padded.
        starts = torch.cumsum(counts, 0) - counts
        eid = flat_e[order]
        pos = torch.arange(n * k, device=index.device) - starts[eid]
        dump = e * cap
        if keep is None:
            # `_permute` always returns a mask; a direct caller may omit it, and
            # `None` then means "every assignment is kept" (the pre-cap
            # behaviour), which is the only reading that keeps the two arms'
            # row sets equal.
            keep = torch.ones_like(owner, dtype=torch.bool)
        slots = torch.where(keep, eid * cap + pos, dump)
        # Zero-filled, not `empty`: a row past capacity is never read, but a
        # zero row keeps the bank finite, and inf-garbage × 0 is NaN in the
        # backward (the spike measured exactly that — `probe_moe_variants.py`).
        xg = torch.zeros(dump + 1, self.d, dtype=dt, device=x.device)
        xg = xg.index_put((slots,), self._gather_rows(x, owner, dt))
        # The bank the GEMM sees is EXACTLY the eager `[e, cap, d]` — the dump
        # row is sliced off, so the expert pass keeps its shape, its kernel and
        # its bits. Only the lookup needs the extra row, and it gets a constant
        # zero row (the dumped assignments' output is 0 anyway).
        bank = xg.narrow(0, 0, dump).view(e, cap, self.d)

        gate_u, up_u, v, gate_s, up_s = self._banks(dt)
        g = torch.bmm(bank, gate_u) * gate_s[:, None, :]
        u = torch.bmm(bank, up_u) * up_s[:, None, :]
        h = F.silu(g) * u if self._glu is None else self._glu(g, u)
        y = torch.bmm(h, v).reshape(dump, self.f)
        y = torch.cat((y, torch.zeros(1, self.f, dtype=dt, device=x.device)))
        y = y[slots]  # [n·k, f], the dumped rows read the constant zero
        # The router weight is applied in fp32, exactly where the naive path
        # applies it: a bf16 multiply would put a 2^-8 relative step between the
        # two arms that the parity gate could not tell from a wiring error.
        # `keep` is a bool CONSTANT of the graph, so a dropped row's weight is
        # exactly 0 and `dL/dw` there is exactly 0 — the eager gradient.
        # ponytail: it may move to rank-space (w ⊙ h before `@ v`, the spike's
        # −8 % on the expert pass) once a b1-shape profile says the [R, f]
        # multiply matters.
        w = flat_w[order].unsqueeze(-1) * keep.to(flat_w.dtype).unsqueeze(-1)
        y = y.to(torch.float32) * w
        out = torch.zeros(n, self.f, dtype=torch.float32, device=x.device)
        out.index_add_(0, owner, y)
        return out

    def _naive_dispatch(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        order: torch.Tensor,
        keep: torch.Tensor,
        counters: Counters,
    ) -> torch.Tensor:
        """Gather → per-expert matmul → scatter. The reference and the CPU path."""
        n, k = index.shape
        flat_w = weight.reshape(-1)
        sorted_ids = index.reshape(-1)[order]
        owner = order // k  # which token each (token, slot) pair belongs to
        out = torch.zeros(n, self.f, dtype=torch.float32, device=x.device)
        # `keep=None` is the dropless route: every assignment reaches its expert,
        # so the selector is the id match alone (the capacity arms pass a mask).
        ids_in = sorted_ids if keep is None else sorted_ids[keep]
        for e in torch.unique(ids_in).tolist():
            sel = (sorted_ids == e) if keep is None else (keep & (sorted_ids == e))
            rows = owner[sel]
            y = self.experts[e](x[rows], glu=self._glu) * flat_w[order[sel]].unsqueeze(-1)
            out.index_add_(0, rows, y)
            counters.count("moe_expert_calls")
        counters.count("moe_tokens", n)
        return out

    def _expert_pass(self, run: Callable[[], torch.Tensor], counters: Counters) -> torch.Tensor:
        """Run ONE routed expert pass, host-offloading its saved activations (SL1).

        `moe_offload` off: `run()` and nothing else — byte-for-byte today's path,
        and no counter, so a reader can tell the two apart on the log line.

        `moe_offload` on: the pass runs inside `checkpoint(use_reentrant=False)`
        with the host pack/unpack hooks INSIDE it, so every activation the pass
        saves (the `[E, cap, ·]` banks, the GLU gates, the gathered rows) lives in
        host RAM between the forward and the backward instead of on the device.
        The hooks go inside the checkpointed FUNCTION, not around the
        `checkpoint()` call — the module docstring carries that measured caveat,
        and `tests/test_offload_experts.py` pins it with a spy on the pack hook
        (a scope wrapped around the call leaves it firing zero times, which is
        the silent no-op this ordering exists to avoid).
        The correctness claim is bit-parity, gated in the same test file.

        Non-reentrant is required, not preferred: the routed pass reads router
        outputs (`index`, `weight`) that are detached from the expert graph.
        The pack copy is SYNCHRONOUS `device→host` (pageable, no pin) — that is
        what the mechanism costs; pinned buffers plus a side-stream copy are the
        upgrade if the b1 step-time measurement says the sync is the bill
        (`docs/research/2026-10-16-budget-savers.md` §5: predicted +5-15% step).
        """
        if not self.moe_offload:
            return run()
        counters.count("moe_offloaded")

        def offloaded() -> torch.Tensor:
            with saved_tensors_hooks(_pack_activation, _unpack_activation):
                return run()

        return checkpoint(offloaded, use_reentrant=False)

    def _dispatch(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        counters: Counters,
    ) -> torch.Tensor:
        """Capacity-bound permute, then `moe_kernel`'s arm (fallbacks COUNTED).

        ONE permute feeds every arm, so all three see the same rows and drop the
        same ones; which one ran is on the counter, and a fallback carries the
        reason on the counter object (ADR-0019 — a degradation a reader of the
        log cannot see is a defect).

        `moe_capacity = 0` (dropless) short-circuits to the RAGGED arm: the
        permute keeps every assignment and the padded arm has no cap to size its
        bank with (`_padded_dispatch` refuses), so the grouped GEMM runs with the
        static row-block bound and the kernels' on-device empty-block exit — no
        host read-back of `offs` (ADR-0018 rule 2). `_dropless_dispatch` owns
        that route and its counters (`moe_dropless`, `moe_triton_gemm`, or
        `moe_naive_dispatch` + the reason when the kernel cannot serve the call).

        A zero router weight is an adaptive-width PADDING SLOT, not a dropped
        assignment: it must not consume capacity (a padding row that keeps a real
        row out would make the adaptive arm drop more than its own k) and it must
        not be gathered into an expert bank. The constant-width arm passes `None`
        — the pre-policy path, bit for bit.
        """
        valid = None if self.topk_mode == "off" else weight > 0
        order, keep, dropped = self._permute(index, valid)
        n, k = index.shape
        # `keep=None` = every permuted row survives, and `order[keep]` is then
        # `order`. Two cases are provably all-True: dropless (`capacity=0` keeps
        # every row by construction), and a one-token decode (`topk` returns
        # DISTINCT experts, so no expert gets more than one of the token's k
        # rows, and the capacity floor is >= 1). Skipping the mask matters
        # beyond the op: a boolean mask's output LENGTH is a device value, i.e. a
        # host sync, which a CUDA-graph capture refuses. `topk_mode != "off"` has
        # zero-weight PADDING slots and always needs the mask; the drop count is
        # still computed above either way.
        static = self.topk_mode == "off" and (self.capacity == 0 or n == 1)
        keep_rows = None if static else keep
        # Count EVERY call: a python `if dropped:` on the device tensor is
        # exactly the sync this counter exists to prevent (caught by the
        # drop-parity test when the tensor stopped being an int). A zero add
        # costs nothing once the drain batches it.
        counters.count("moe_dropped_assignments", dropped)
        # The gather-fused forward (`moe_gather = "kernel"`) runs BEFORE the
        # capacity split: it is one kernel for both routes (the cap bounds its
        # row-block grid, `None` is the dropless static bound), and its ABI is
        # the permute above. Everything it cannot serve is COUNTED with the
        # reason and falls through to the `moe_kernel` arms unchanged.
        gwhy = None
        if self.moe_gather == "kernel":
            gwhy = self._gather_reason(x)
            if gwhy is None:
                counters.count("moe_gather_forward")
                counters.count("moe_gather_rows", keep.sum())
                if self.moe_offload:
                    # The arm's graph already recomputes its forward in the
                    # backward (`moe_sonic._GatherGlu`), so `checkpoint`'s host
                    # parking would buy a third forward and offload `x` for
                    # nothing. Asked for and not run must be visible (ADR-0019).
                    counters.count("moe_offload_inert_gather")
                return self._gather_dispatch(x, weight, index, order, keep, counters)
            counters.count("moe_gather_fallback")
            counters.why = f"gather: {gwhy}"  # type: ignore[attr-defined]
        if self.capacity == 0:
            return self._dropless_dispatch(x, weight, index, order, keep, keep_rows, counters)
        notes = [] if gwhy is None else [f"gather: {gwhy}"]
        if self.moe_kernel == "triton":
            why = (
                None
                if self.expert_act == "silu"
                else (
                    f"expert_act={self.expert_act} has no fused kernel "
                    f"(the triton GLU hardcodes silu) — padded torch arm runs"
                )
            )
            if why is None:
                why = self._triton_reason(x)
            if why is None:
                # The fused backward's own gate (top-k): COUNTED, and it leaves
                # `moe_triton_gemm` counting because the grouped GEMM does run.
                fwhy = self._fused_bwd_reason(index.shape[1])
                if fwhy is not None:
                    counters.count("moe_triton_fallback")
                    counters.why = f"fused backward: {fwhy}"  # type: ignore[attr-defined]
                counters.count("moe_triton_gemm")
                return self._expert_pass(
                    lambda: self._triton_dispatch(
                        x, weight, index, order, keep_rows, fused_bwd=fwhy is None
                    ),
                    counters,
                )
            counters.count("moe_triton_fallback")
            notes.append(f"triton: {why}")
        why = self._padded_reason(x)
        if self.moe_kernel == "naive":
            why = "moe_kernel=naive: the config picked the reference loop"
        if why is None:
            if notes:
                counters.why = "; ".join(notes)  # type: ignore[attr-defined]
            counters.count("moe_padded_bmm")
            return self._expert_pass(
                lambda: self._padded_dispatch(x, weight, index, order, keep_rows), counters
            )
        counters.count("moe_naive_dispatch")
        counters.why = "; ".join([*notes, why])  # type: ignore[attr-defined]
        return self._expert_pass(
            lambda: self._naive_dispatch(x, weight, index, order, keep, counters), counters
        )

    def _dropless_dispatch(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        order: torch.Tensor,
        keep: torch.Tensor,
        keep_rows: torch.Tensor | None,
        counters: Counters,
    ) -> torch.Tensor:
        """The `moe_capacity = 0` route: the RAGGED arms, the grouped GEMM first.

        The padded arm is out by construction (a `[E, cap, ·]` bank has no cap),
        so the choice is between the grouped GEMM and the reference loop — and the
        grouped GEMM now SERVES dropless: its row-block grid takes a static bound
        the host reads from a shape (`moe_kernels._dropless_cap`) and every
        program exits on-device when its block is empty, so no `offs` read-back is
        needed (ADR-0018 rule 2). Before that early-exit existed this route had to
        be the per-expert Python loop, which at `b1`'s E=1024 is ~1024 tiny
        matmuls per pass.

        `moe_kernel` still decides: `naive` is the reference loop (the config's
        own choice), `triton` is the grouped GEMM, and `padded` — the default,
        which cannot run here — prefers the grouped GEMM with the note on the
        counter object. Anything the kernel cannot serve (no triton, not CUDA, a
        non-silu expert activation) falls back to the reference loop COUNTED,
        with `moe_triton_fallback` when the grouped GEMM was the config's ask.
        The grouped GEMM itself runs the FUSED backward (`_fused_bwd_reason` is
        the only thing that can send it back to the previous adjoint, COUNTED).
        """
        counters.count("moe_dropless")
        why = None
        if self.moe_kernel == "naive":
            why = "moe_kernel=naive: the config picked the reference loop"
        elif self.expert_act != "silu":
            why = (
                f"expert_act={self.expert_act} has no fused kernel (the triton GLU hardcodes silu)"
            )
        else:
            why = self._triton_reason(x)
        if why is None:
            if self.moe_kernel == "padded":
                prev = getattr(counters, "why", None)
                note = (
                    "capacity=0: dropless — the padded bank has no cap to size itself "
                    "with, so the grouped GEMM runs (the static row-block bound + the "
                    "kernels' on-device empty-block exit)"
                )
                counters.why = f"{prev}; {note}" if prev else note  # type: ignore[attr-defined]
            # The fused backward's own gate (top-k): COUNTED, and `moe_triton_gemm`
            # still counts because the grouped GEMM does run (its previous adjoint
            # is what the fallback selects).
            fwhy = self._fused_bwd_reason(index.shape[1])
            if fwhy is not None:
                counters.count("moe_triton_fallback")
                note = f"fused backward: {fwhy}"
                prev = getattr(counters, "why", None)
                counters.why = f"{prev}; {note}" if prev else note  # type: ignore[attr-defined]
            counters.count("moe_triton_gemm")
            return self._expert_pass(
                lambda: self._triton_dispatch(
                    x, weight, index, order, keep_rows, dropless=True, fused_bwd=fwhy is None
                ),
                counters,
            )
        if self.moe_kernel == "triton":
            counters.count("moe_triton_fallback")
            why = f"triton: {why}"
        prev = getattr(counters, "why", None)
        counters.why = f"{prev}; {why}" if prev else why  # type: ignore[attr-defined]
        counters.count("moe_naive_dispatch")
        return self._expert_pass(
            lambda: self._naive_dispatch(x, weight, index, order, keep, counters), counters
        )

    def _triton_reason(self, x: torch.Tensor) -> str | None:
        """`None` if the unpadded grouped GEMM can run THIS call, else why not.

        The arm is unpadded by construction (`offs` carries the ragged bounds),
        so the gates are the ability itself (no triton, no CUDA) and the TILE:
        `moe_kernels.tile_reason` estimates the CTA shared memory of the widest
        kernel from the tile constants and the compute dtype. A set that does
        not fit is COUNTED with its numbers instead of dying at launch — that is
        what `196608 B > 101376` on the fp32 ByteFlow arm used to do, and why
        that arm ran `moe_kernel=naive`. `moe_kernels.tiles_for` now halves the
        wide tiles at four bytes, so the fp32 shape fits and no fallback fires.
        """
        if _TRITON_WHY is not None:
            return _TRITON_WHY
        why = moe_kernels.tile_reason(self._bank_dtype(), self.d, self.f, self.experts[0].rank)
        if why is not None:
            return why
        if not x.is_cuda:
            return f"input is on {x.device.type}: the grouped GEMM is the CUDA path"
        return None

    def _gather_reason(self, x: torch.Tensor) -> str | None:
        """`None` if the gather-fused forward can run THIS call, else why not.

        COUNTED (`moe_gather_fallback` + the reason on `counters.why`), like the
        triton arm's own gate: a config asks for an arm, and an arm that cannot
        serve a call must not vanish. Three things can stop it, and the dtype
        route is NOT one of them here — `config.validate` already refused
        `bf16=false`, `use_byteflow=true`, the adaptive width and `factor_qat`,
        so a `moe_gather="kernel"` run that reached a forward is on the route
        the parity gates cover.

        What is left is ability: the kernel must be importable (`moe_sonic`'s
        own guard), the input must be on a CUDA device (`_gather_glu_kernel` is
        a Triton launch; `_triton_reason`'s device gate, same words), and the
        expert activation must be `silu` — the kernel has `tl.sigmoid` in
        registers and the shipped arm refuses the same three activations for the
        same reason (`_dispatch`'s `expert_act != "silu"` note).
        """
        if moe_sonic._TRITON_WHY is not None:
            return moe_sonic._TRITON_WHY
        if self.expert_act != "silu":
            return (
                f"expert_act={self.expert_act} has no fused kernel "
                f"(the gather GLU hardcodes silu) — the grouped GEMM / padded "
                f"torch arm runs instead"
            )
        if not x.is_cuda:
            return f"input is on {x.device.type}: the gather kernel is the CUDA path"
        return None

    def _fused_bwd_reason(self, k: int) -> str | None:
        """`None` if the FUSED backward can serve a top-k of `k`, else why not.

        The only thing it refuses is the top-k: the combine kernel UNROLLS the
        top-k loop (`tl.static_range`) up to `moe_kernels.FUSED_TOPK_MAX`, and
        everything else the fused path needs — CUDA, triton, a dense `[R, ·]`
        `dy` — is `_triton_reason`'s gate already. Our own top-k is 2, so this
        fires only for a config that asks for more.

        COUNTED, never silent, and the counter is `moe_triton_fallback` with
        `fused backward: …` on `counters.why` while `moe_triton_gemm` still
        counts: the grouped GEMM DID run, it just ran its previous adjoint
        (`fused_bwd=False`) — the reference is the fallback here, not the loop.
        """
        if moe_kernels is None:
            return _TRITON_WHY
        if k < 1 or k > moe_kernels.FUSED_TOPK_MAX:
            return (
                f"topk={k} is outside the fused backward's unrolled tile "
                f"[1, {moe_kernels.FUSED_TOPK_MAX}]"
            )
        return None

    def _bwd_workspace(self):
        """The grow-only backward scratch — ONE per `SparseMoE`, owned HERE.

        `moe_kernels.BwdWorkspace` holds no module-level state (its docstring:
        the caller owns the buffers); this is the caller. Created on first use
        so a model that never runs the triton arm never allocates it, and kept
        across steps so the steady-state backward allocates no `[R, ·]` scratch.
        `_banks`'s cache is the same shape of object for the same reason.
        """
        ws = getattr(self, "_moe_bwd_ws", None)
        if ws is None:
            assert moe_kernels is not None, "no workspace without the triton arm"
            ws = self._moe_bwd_ws = moe_kernels.BwdWorkspace()
        return ws

    def _triton_dispatch(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        order: torch.Tensor,
        keep: torch.Tensor | None,
        *,
        dropless: bool = False,
        fused_bwd: bool = True,
    ) -> torch.Tensor:
        """Gather → ONE unpadded grouped-GEMM pass (GLU fused) → weighted scatter.

        The arithmetic of `_padded_dispatch` with the padding removed: the kept
        rows are gathered expert-major (they already are — the permute sorts by
        expert id), `offs` carries the ragged group bounds, and no `[E, cap, f]`
        intermediate is ever materialized. Rows past capacity never enter, so with
        a capacity the drop set is byte-for-byte the padded arm's.

        `dropless=True` (the `moe_capacity = 0` route) hands the kernel NO cap:
        the row-block grid takes the static bound (`moe_kernels._dropless_cap`,
        `R` rows in one group) and every program exits on-device when its block is
        empty, so the grid is safe for ANY routing without reading `offs` back to
        the host (ADR-0018 rule 2). Empty experts cost a program launch and two
        loads — that is what makes this arm the dropless one instead of the
        per-expert reference loop.

        `fused_bwd=True` (the default) is the fused adjoint: the scatter is
        `moe_kernels.combine` (a custom Function whose backward is ONE kernel —
        `grad_out` read once per token, the `[R]` router-weight gradient summed
        in registers) and the expert pass gets the permute's own `int32`
        row→expert map plus this module's grow-only workspace, so the backward
        neither rebuilds the segments with `repeat_interleave` nor allocates its
        `[R, ·]` scratch per call. The forward's numbers are unchanged either
        way — only the graph the backward walks. `fused_bwd=False` is the
        previous path (a plain torch scatter and `_glu_backward`), kept as the
        parity gate and the bench's control.
        """
        assert moe_kernels is not None, "the triton arm cannot be dispatched without it"
        n, k = index.shape
        dt = self._bank_dtype()
        # `keep=None` = every permuted row survives (a one-token decode).
        rows = order if keep is None else order[keep]
        owner = rows // k
        kept = index.reshape(-1)[rows]
        counts = _bincount(kept, self.n_experts).to(torch.int32)
        offs = torch.zeros(self.n_experts + 1, dtype=torch.int32, device=x.device)
        offs[1:] = torch.cumsum(counts, 0)
        gate_u, up_u, v, gate_s, up_s = self._banks(dt)
        cap = None if dropless else self._capacity(n, k)
        xg = self._gather_rows(x, owner, dt)
        if not fused_bwd:
            y = moe_kernels.expert_gemm_glu(
                xg, gate_u, up_u, gate_s, up_s, v, offs, cap=cap, fused_bwd=False
            )
            # fp32 weight applied where the padded arm applies it (see the note
            # there): a bf16 multiply would be a 2^-8 step the parity gate could
            # not tell from a wiring error.
            y = y.to(torch.float32) * weight.reshape(-1)[rows].unsqueeze(-1)
            out = torch.zeros(n, self.f, dtype=torch.float32, device=x.device)
            out.index_add_(0, owner, y)
            return out
        # The row→expert map the fused backward needs IS the routing the permute
        # already produced — `kept` is the expert of every gathered row — so the
        # backward's `repeat_interleave` rebuild costs one cast instead. Built
        # once per forward here and carried through autograd (`_ExpertGemmGlu`
        # keeps it on `ctx`), which is the "built once, cached" the study asks for.
        seg = kept.to(torch.int32)
        # `rowmap[flat (token, slot)] = gathered row`, `-1` where the capacity
        # dropped it: the int32 row map OLMo-core's combine backward is built
        # around (`moe_unpermute_bwd_kernel.cu:177-179`), one per forward.
        rowmap = torch.full((n * k,), -1, dtype=torch.int32, device=x.device)
        rowmap[rows] = torch.arange(rows.numel(), dtype=torch.int32, device=x.device)
        ws = self._bwd_workspace()
        y = moe_kernels.expert_gemm_glu(
            xg,
            gate_u,
            up_u,
            gate_s,
            up_s,
            v,
            offs,
            cap=cap,
            fused_bwd=True,
            seg=seg,
            ws=ws,
        )
        # `combine`'s contract is fp32 `w` (and its fused backward's `dy` is
        # built from it), so the weight is widened HERE, at the boundary. The
        # weight arrives in the COMPUTE dtype — under `bf16 = true` it is the
        # softmax of `mm_mixed`'s bf16 logits — and the pre-fusion adjoint
        # (`y.to(fp32) * w`) ran the scatter and its `mul`/`index_add` backward
        # in fp32 for exactly this reason. Casting at the call site rather than
        # inside `combine` keeps the kernel's own `w.dtype == torch.float32`
        # assert LOUD: a future caller that hands it a non-fp32 weight still
        # fails instead of being silently coerced.
        return moe_kernels.combine(
            y, weight.reshape(-1)[rows].float(), owner, rowmap.view(n, k), n, ws=ws
        )

    def _gather_dispatch(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        order: torch.Tensor,
        keep: torch.Tensor,
        counters: Counters,
    ) -> torch.Tensor:
        """The gather-fused forward: same rows, same scatter, NO gathered copy.

        `_triton_dispatch` with its `xg = x[owner].to(dt)` line deleted. The
        routing is the permute's own (`gather_abi` turns `order`/`keep` into the
        `(toks, offs)` pair the kernel takes), the banks are the same
        `_banks(dt)` stacks, the output goes through the same
        `moe_kernels.combine`, so the ONLY thing that changed is where the A
        operand's rows come from: `x[toks[row]]` read inside the kernel, instead
        of four `[R, d]` transfers (fp32 gather, its cast-read, the bf16 copy,
        the kernel's read of that copy).

        Everything after the kernel is `_triton_dispatch`'s, deliberately: the
        weights are applied as fp32 where the padded arm applies them, the
        `rowmap` is the combine backward's own int32 map, and the workspace is
        the same grow-only `BwdWorkspace`. A different scatter here would make
        the parity gate compare two changes at once.

        The FORWARD is the only fused half (`moe_sonic._GatherGlu` recomputes the
        grouped pass for its backward), so the fwd→bwd cache is `x`, `toks` and
        `y` — no `xg`, no GLU gates. That is priced by
        `scripts/bench_gather.py`, not here.
        """
        assert moe_kernels is not None, "the gather arm needs the shipped grouped GEMM"
        n, k = index.shape
        dt = self._bank_dtype()
        rows = order[keep]
        owner = rows // k
        toks, offs = moe_sonic.gather_abi(index, order, keep, self.n_experts)
        gate_u, up_u, v, gate_s, up_s = self._banks(dt)
        cap = None if self.capacity == 0 else self._capacity(n, k)
        seg = index.reshape(-1)[rows].to(torch.int32)
        rowmap = torch.full((n * k,), -1, dtype=torch.int32, device=x.device)
        rowmap[rows] = torch.arange(rows.numel(), dtype=torch.int32, device=x.device)
        ws = self._bwd_workspace()
        y = moe_sonic.gather_glu(
            x,
            toks,
            gate_u,
            up_u,
            gate_s,
            up_s,
            v,
            offs,
            cap=cap,
            seg=seg,
            ws=ws,
        )
        return moe_kernels.combine(
            y, weight.reshape(-1)[rows].float(), owner, rowmap.view(n, k), n, ws=ws
        )


def invalidate_banks_all(model: nn.Module) -> int:
    """Drop every SparseMoE's bank cache; returns how many were cleared.

    The trainer's ONE per-step call (`train/loop.py`, `train/sft.py`): see
    `SparseMoE.invalidate_banks` for the measured reason (a CUDA-graph replay
    moves a parameter's VALUE without bumping `_version`, so a `_version`-keyed
    cache cannot outlive a step in which the optimizer ran).
    """
    n = 0
    for m in model.modules():
        if isinstance(m, SparseMoE):
            m.invalidate_banks()
            n += 1
    return n
