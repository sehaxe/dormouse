"""MoE: top-k dispatch over dual-latent GLU experts, plus an always-on shared expert.

The expert is TSCT2 §14's shape (`docs/records/reviews/tsct2-design-2026-10-04.md`):

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

Dispatch has FOUR arms over ONE permute, and the log line says which ran:

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
  is the A/B kill-switch (`docs/records/research/2026-10-15-tsct-super-kernel.md` §5.5:
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
* **batched (the dropless bank)**: the SAME `[E, cap, ·]` body as `padded`, but
  `cap` is chosen for THIS batch — the largest group when `moe_capacity = 0`,
  the capacity rule's number otherwise. It replaces the dropless per-expert
  loop's `E` tiny GEMMs with three `bmm`s: measured on the 5070 Ti at the m400
  shape (8192 tokens, E=512, r=64) the loop is 99.56 ms and the bank 11.45 ms
  (`scripts/part_bench.py`; this task's CPU run is the parity gate, not the
  timing). The dropless cap is one host scalar (`counts.max()`), because a bank
  shape needs a Python int; it is the only host read in the arm (see
  `_batched_cap`).
* **bucketed (the batched bank without the memory cliff)**: the same arithmetic
  as `batched`, over a PLAN of smaller banks instead of ONE `[E, cap, ·]`. The
  bank's row bill is `E · max_group · d`, and `max_group` is the routing's, not
  the model's: one expert holding 4 096 of 16 384 assignments asks for a 12 GiB
  fp32 bank (measured 2026-11-04 on a collapsed router). This arm partitions the
  experts into row-count classes (all counts within 2x share a bucket), splits a
  class when its rows pass `MOE_BUCKET_ROWS`, and runs three `bmm`s per bucket
  against `[n_b, cap_b, ·]`; the count is O(log(max/mean)) for a power-of-2
  routing and one for a uniform one. A bucket's bank holds at most
  `2 · MOE_BUCKET_ROWS` rows (an expert past the budget gets a bucket of its
  own, so a single-expert collapse is `1 · count · d`, 25 MB at 8 192 rows /
  d=1536 / bf16). `MOE_BUCKET_BYTES` is the LOUD refusal when even that is too
  large; `moe_buckets` and `moe_bank_bytes` are the counters that prove the
  shape. The plan costs ONE host read of the `counts` vector (`E` ints — bank
  shapes are Python ints) and is a deterministic function of the counts.
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

All of them count: `moe_padded_bmm`, `moe_batched_calls`, `moe_buckets` +
`moe_bank_bytes`, `moe_triton_gemm`, `moe_gather_forward` or
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

* the padded arm **refuses** (`_padded_dispatch`): its `[E, cap, ·]` bank has
  no shape without a FIXED cap. The dropless bank is `moe_kernel="batched"`,
  which sizes the cap to THIS batch's largest group (one host scalar), so it is
  a separate arm rather than the padded one silently reading a capacity of 0;
* with the default `padded` arm the dropless route runs the GROUPED GEMM: its
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
`docs/records/research/2026-10-16-budget-savers.md` §5). The MoE block owns the
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

**TSCT-2** (`docs/architecture/2026-11-05-tsct-2.md`) adds three OFF-by-default
mechanisms, all on this module because the routed counts are here:

* `tsct_expert_basis` — one `w [rank, shared_rank]` per routed expert, applied
  inside a `[d, shared_rank]` shared plane. The bank FOLDS it into the effective
  per-expert basis at the rebuild (`gate_u @ w.T -> [d, rank]`), so the operand
  shapes and the per-token MAC do not move; only the build pays
  `E·d·rank·shared_rank`, once per optimizer step. Init `[I | 0]` makes step 0
  the planeless forward. Counter: `tsct_basis_fold`.
* `tsct_expert_lr` — `expert_lr_scales()` returns `[E]` from the device-side
  `lfb_counts`; `train.optim.MixOptimizer` gathers it into the Muon stack's row
  scale (one grouped multiply, no per-parameter loop, no host read).
  Counter: `tsct_expert_lr`.
* `tsct_expert_revive` — `revive_experts()` copies the hottest expert onto the
  coldest plus seeded noise and resets both LFB entries. The two indices are
  the ONE declared host read, once per N optimizer steps; the LR path reads
  none. Counter: `tsct_revive`.

The three arms compose. All refusals are LOUD with an `Escape:` naming the
flag to change; `config.validate` and this constructor both enforce them
(a hand-built config skips the former).
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

from ..config import FACTOR_QATS, SHARE_U_MODES, validate_ffn
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

from . import (
    moe_sonic,  # the gather-fused forward (`moe_gather = "kernel"`)
    route_fused,  # the fused route head + the counting permute
)


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


#: Quantile Balancing's histogram resolution (`moe_qb`): each expert's router
#: scores are bucketed into this many bins over the batch's own score range, so
#: the target quantile is located to `1/_QB_BINS` of that range without a sort
#: and without a host read. 256 bins put the error at 0.4% of the range — well
#: inside the ±10% load band the mechanism is held to (`tests/test_router_qb.py`).
_QB_BINS = 256


def switch_aux_loss(probs: torch.Tensor, index: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Switch/GShard load-balance term, weight-free: `E · Σ_e f_e · P_e`.

    `f_e` is the fraction of the window's ASSIGNMENTS expert `e` took (`index`
    is the top-k it was sent, so a hard count — a constant with respect to every
    parameter); `P_e` is the mean gate probability over the tokens. The product
    therefore carries a gradient into the ROUTER and into nothing else: an
    expert that takes a large share raises the term in proportion to how
    confidently the gate picks it, and the optimizer's way to lower it is to
    move gate mass OFF that expert — including onto one the gate currently
    never picks (`P_e ≈ 0` still has a gradient through the softmax).

    That is the property the sign controller lacks on a collapsed bank (measured
    2026-11-04: `m400`'s saved bias left **9 of 512** experts alive — the
    bias-less router read 80/512, which was the instrument's view, not the
    file's). A dead expert's weights get no expert gradient, so only a
    routing-signal term can bring it back, and this one is differentiable.
    `n_experts ×` is the paper's normalization: `f` sums to 1 over the bank, so a
    uniform load and a uniform gate read exactly `1` (Switch's value) at ANY
    `k`, and one expert taking every assignment reads up to `E` — one weight
    means the same thing whatever the bank size.

    Computed in fp32 whatever the bank's compute dtype: it is a scalar whose
    gradient multiplies the router's logits, and 512 terms summed in bf16 would
    lose exactly the small `P_e` of the dead experts this term exists to lift.
    """
    n, k = index.shape
    assert n > 0 and k > 0, f"switch_aux_loss on an empty routing {tuple(index.shape)}"
    f = _bincount(index.reshape(-1), n_experts).to(torch.float32) / float(n * k)
    p = probs.to(torch.float32).mean(dim=0)
    return n_experts * (f * p).sum()


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


#: The local seed for the EXTRA shared-plane columns of `tsct_expert_basis`.
#: A local generator on purpose: the extra columns are new parameters, and
#: drawing them from the global stream would shift every later parameter and
#: confound the ON-vs-OFF A/B with a second initialisation.
TSCT_BASIS_EXTRA_SEED = 0x7A5C2

#: The pre-registered revive noise is `sigma = noise · std(v_hot)`; this is the
#: clip on the LR multiplier (`experts.expert_lr_multipliers`), as specified in
#: `docs/architecture/2026-11-05-tsct-2.md`.
EXPERT_LR_CLIP = (0.2, 5.0)


def tsct_basis_extra(d: int, cols: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """`N(0, 1/sqrt(d))` for the widened plane's new columns, from a LOCAL stream.

    Drawn on CPU and moved, for two reasons: a `meta` build (`capacity_math`'s
    price table) has no generator, and a CPU stream makes the new columns
    identical across devices. The draw is once per site at construction.
    """
    g = torch.Generator().manual_seed(TSCT_BASIS_EXTRA_SEED)
    return (torch.randn(d, cols, generator=g) / math.sqrt(d)).to(device)


def tsct_basis_param_delta(n_experts: int, d: int, rank: int, shared_rank: int) -> int:
    """The arm's exact parameter delta: `E·rank·r_shared` selectors + the wider plane.

    The doc's `+6.3M` counts only the selectors and assumes a `[d, 192]` plane
    already existed. It does not: the routed plane is `[d, rank]` (64), so the
    arm also pays `2·d·(r_shared − rank)` once per site. At m400b
    (E=512, d=1536, rank=64, r_shared=192): 6 291 456 + 393 216 = 6 684 672.
    """
    return n_experts * rank * shared_rank + 2 * d * (shared_rank - rank)


def expert_lr_multipliers(
    counts: torch.Tensor,
    coeff: float,
    clip: tuple[float, float] = EXPERT_LR_CLIP,
) -> torch.Tensor:
    """`1 + coeff·(mean(L)/max(L_j,1) − 1)`, clipped — a PURE function of `counts`.

    Device-in, device-out: no `.item()`, no branch on a device value, and no
    history (the caller recomputes it from the LFB integrator's `lfb_counts`
    every step, so a resume needs no extra state). All counts equal → exactly
    `1.0`; a dead expert reads `mean(L)` before the clip, so a small `coeff`
    gives it a bounded boost rather than an unbounded one.
    """
    assert counts.ndim == 1, f"expert_lr_multipliers wants [E] counts, got {tuple(counts.shape)}"
    assert coeff >= 0.0, f"expert_lr_multipliers: coeff={coeff} must be >= 0"
    ratio = counts.mean() / counts.clamp_min(1.0)
    return (1.0 + coeff * (ratio - 1.0)).clamp(clip[0], clip[1])


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
        tsct_basis: int = 0,
        basis_shared: int = 0,
        packed: bool = False,
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
        #: TSCT-2's per-expert INPUT selector (`SparseMoE.tsct_expert_basis`):
        #: `w [rank, basis_shared]`, applied INSIDE the widened shared plane.
        #: It has NO effect alone — `share_u` is what binds the plane — and the
        #: parent refuses the pairless case; the check here is for a hand-built
        #: expert. Init is `[I | 0]`, so at step 0 `x @ (plane @ w.T)` is exactly
        #: `x @ plane[:, :rank]`, i.e. today's path (gate b,
        #: `tests/test_tsct2.py`).
        self.tsct_basis = int(tsct_basis)
        self.basis_shared = int(basis_shared)
        self.w: nn.Parameter | None = None
        if self.tsct_basis:
            if not share_u:
                loud(
                    f"tsct_basis={self.tsct_basis} on an expert that owns its U: "
                    f"the selector lives INSIDE the shared plane (`share_u`). "
                    f"Escape: build it under SparseMoE with share_u_mode='plane', "
                    f"or tsct_basis=0."
                )
            if self.basis_shared <= rank:
                loud(
                    f"tsct_basis={self.tsct_basis} with basis_shared="
                    f"{self.basis_shared} <= rank={rank}: the wider plane must be "
                    f"wider than the latent or there is no subspace to select. "
                    f"Escape: shared_rank > rank, or tsct_basis=0."
                )
            self.w = nn.Parameter(
                torch.cat(
                    (
                        torch.eye(rank, device=device),
                        torch.zeros(rank, self.basis_shared - rank, device=device),
                    ),
                    dim=1,
                )
            )
        #: The a4.8 ACTIVATION switch (`act_quant.fake_quant_act`): the `x` fed
        #: to this expert's factor matmuls is fake-quantized with an STE. Same
        #: switch the banked path in `SparseMoE` applies to its `x`, so one
        #: expert through the bank and the same expert alone quantize their
        #: activation identically (per-row/group scale, expert-independent).
        self.act_quant = act_quant
        self.act_group = act_group
        #: PACKED BANK (`SparseMoE.tsct_packed`, OFF by default): this expert
        #: owns no `v`/`s` (and no `u` when the parent shares one). The parent
        #: owns ONE buffer per site and binds per-expert VIEWS through
        #: [`bind_packed`], so the once-per-step bank rebuild casts a whole
        #: `[E, ...]` tensor instead of E parameter `.to`s. The views are plain
        #: tensors on purpose — they must NOT appear in `named_parameters()`,
        #: because the optimizer's objects are the packed buffers (a second
        #: registration would double-step the same storage).
        self.packed = packed
        for name in ("gate", "up"):
            if not share_u and not packed:
                # `U [d, r]` and `s [r]`: the latent half of TSCT's factorization.
                setattr(self, f"{name}_u", nn.Parameter(init_u(d, rank, device)))
            if not packed:
                setattr(self, f"{name}_s", nn.Parameter(torch.ones(rank, device=device)))
        if not packed:
            # The write `V [r, f]` — one, at the latent, after the GLU.
            self.v = nn.Parameter(torch.randn(rank, f, device=device) / math.sqrt(rank))
        else:
            self.v = None
            self.gate_s = None
            self.up_s = None
            if not share_u:
                self.gate_u = None
                self.up_u = None

    def bind_packed(
        self,
        v: torch.Tensor,
        gate_s: torch.Tensor,
        up_s: torch.Tensor,
        gate_u: torch.Tensor | None = None,
        up_u: torch.Tensor | None = None,
    ) -> None:
        """Take the parent's packed buffers as per-expert VIEWS — never a copy.

        Called once at build and again after a `load_state_dict` (a loader may
        replace the storage a view pointed at, which would silently freeze the
        expert on the old tensor). The buffers are `nn.Parameter`s of the
        PARENT; these attributes are plain tensors, so `named_parameters()` has
        exactly one entry per site and the optimizer steps it once.
        """
        assert self.packed, "bind_packed on an expert that owns its own factors"
        self.v = v
        self.gate_s = gate_s
        self.up_s = up_s
        if gate_u is not None:
            self.gate_u = gate_u
            self.up_u = up_u

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
        if self.w is not None:
            # TSCT-2: the expert's own `rank`-dim subspace of the wider shared
            # plane. The LITERAL form of `docs/architecture/2026-11-05-tsct-2.md`
            # (`x @ plane @ w.T`); the banked path folds the same product into
            # the effective basis at the rebuild so the per-token MAC does not
            # move. `factor_qat` is refused with this arm by config.validate.
            w = self.w.to(gate_u.dtype)
            gate_u = gate_u @ w.T
            up_u = up_u @ w.T
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


class TSCTGLU(nn.Module):
    """The `ffn_form="tsct"` expert: all THREE matrices independently factored.

    `gate = U_g[d, r_g]·diag(s_g)·V_g[r_g, f]`, the same form for `up`, and
    `down = U_d[f, r_d]·diag(s_d)·V_d[r_d, d]`. The GLU is applied at the full
    hidden width `f` (that is the point: no rank bottleneck before the
    nonlinearity). At `r_g = r_d = min(d, f)` the map is the DENSE GLU
    reparameterised — same function class, more parameters (`r·(d + f + 1)`
    versus `d·f` per matrix) — which is why "compression" here is a knob, not
    an approximation.
    """

    ffn_form = "tsct"

    def __init__(
        self,
        d: int,
        f: int,
        rank_gate: int,
        rank_down: int,
        *,
        dtype: torch.dtype,
        factor_qat: str = "none",
        factor_qat_alpha: float = 1.0,
        act_quant: str = "none",
        act_group: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        lo = min(d, f)
        if rank_gate > lo or rank_down > lo:
            loud(
                f"TSCTGLU ranks gate={rank_gate} down={rank_down} above "
                f"min(d={d}, f={f})={lo}: a rank above either dimension is not a "
                f"factorisation. Escape: both <= {lo}."
            )
        if rank_gate < 1 or rank_down < 1:
            loud(
                f"TSCTGLU ranks gate={rank_gate} down={rank_down}: a factor rank "
                f"must be >= 1. Escape: positive ranks."
            )
        self.d, self.f = d, f
        self.rank_gate, self.rank_down = rank_gate, rank_down
        self.compute_dtype = dtype
        self.factor_qat = factor_qat
        self.factor_qat_alpha = factor_qat_alpha
        self.act_quant = act_quant
        self.act_group = act_group
        self._qat_cache = QuantCache() if factor_qat != "none" else None
        # The gate/up input bases and their shared-rank writes.
        for name, rg in (("gate", rank_gate), ("up", rank_gate)):
            setattr(self, f"{name}_u", nn.Parameter(init_u(d, rg, device)))
            setattr(self, f"{name}_s", nn.Parameter(torch.ones(rg, device=device)))
            setattr(
                self,
                f"{name}_v",
                nn.Parameter(torch.randn(rg, f, device=device) / math.sqrt(rg)),
            )
        # The down map is f -> d; its `U` lives on the f side and its `V` writes d.
        self.down_u = nn.Parameter(init_u(f, rank_down, device))
        self.down_s = nn.Parameter(torch.ones(rank_down, device=device))
        self.down_v = nn.Parameter(torch.randn(rank_down, d, device=device) / math.sqrt(rank_down))
        #: The dense form's two biases, kept here so the full-rank
        #: reparameterisation is EXACT rather than "exact up to biases": the
        #: hidden `f` bias enters the gate before the GLU, the `d` bias leaves
        #: the down write. They do not factor (a bias has no rank).
        self.gate_b = nn.Parameter(torch.zeros(f, device=device))
        self.down_b = nn.Parameter(torch.zeros(d, device=device))

    def _fq(self, t: torch.Tensor) -> torch.Tensor:
        if self.factor_qat == "none":
            return t
        return factor_qat_project(t, self.factor_qat, self.factor_qat_alpha, cache=self._qat_cache)

    def _proj(
        self, x: torch.Tensor, u: torch.Tensor, s: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        dt = self.compute_dtype
        u, v = self._fq(u), self._fq(v)
        xu = x if x.dtype == dt else x.to(dt)
        return ((xu @ u.to(dt)) * s.to(dt)) @ v.to(dt)

    def forward(self, x: torch.Tensor, glu=None) -> torch.Tensor:
        dt = self.compute_dtype
        if self.act_quant != "none":
            x = fake_quant_act(x, self.act_quant, self.act_group)
        g = self._proj(x, self.gate_u, self.gate_s, self.gate_v) + self.gate_b.to(dt)
        u = self._proj(x, self.up_u, self.up_s, self.up_v)
        h = F.silu(g) * u if glu is None else glu(g, u)
        y = self._proj(h, self.down_u, self.down_s, self.down_v) + self.down_b.to(dt)
        return y if dt == torch.float32 else y.to(torch.float32)

    def extra_repr(self) -> str:
        return (
            f"d={self.d}, f={self.f}, r_gate={self.rank_gate}, "
            f"r_down={self.rank_down}, dtype={self.compute_dtype}"
        )


#: How far the batched bank's row cap may exceed the uniform per-expert share
#: before the arm refuses. 8x is generous for a healthy router (measured 3.1x at
#: 150k: 126 rows against a uniform 40) and refuses a collapsed one.
MOE_BANK_SAFETY = 8


#: The bucketed arm's ROW budget for one bank. Experts whose counts are within
#: 2x of each other share a bucket, so a bucket's `[n_b, cap, d]` bank holds at
#: most `2 · MOE_BUCKET_ROWS` rows: `n_b · cap <= 2 · n_b · min_count <= 2 ·
#: total_rows`. 2 048 rows is 6 MiB at m400's `d=1536` in bf16, i.e. the bank
#: stops following `E · max_group` and follows the number of ROWS instead.
MOE_BUCKET_ROWS = 2048

#: The bucketed arm's total bank budget (bytes, the bank's own dtype). The
#: smallest card this project runs is 16 GB; a bank above 256 MiB is not the
#: activation peak's problem any more. Exceeding it is LOUD (`_bucketed_dispatch`
#: names the busiest bucket and the measured bytes), never a silent clamp: a
#: clamped bank would drop rows and look exactly like a capacity arm.
MOE_BUCKET_BYTES = 256 << 20


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
        self.ffn_form = getattr(cfg, "ffn_form", "bank")
        #: The config layer's own refusals, repeated at the runtime seam: a
        #: hand-built `ModelConfig` reaches this constructor without `validate`.
        validate_ffn(cfg)
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
        #: QUANTILE BALANCING (Kimi K3 §2.3.3, arXiv 2607.24653): the bias
        #: controller that sets each expert's bias from the score quantile its
        #: target load names, instead of taking a fixed-rate step. Read with
        #: `getattr` because `config.py` is another lane's file; the refusals
        #: live HERE beside the other runtime seams a hand-built config reaches
        #: without `validate` (ADR-0019: an arm asked for and silently not run
        #: is a defect).
        self.qb = bool(getattr(cfg, "moe_qb", False))
        #: The routed assignment share each expert is held to: `topk/E`, the
        #: mean of every expert's load under a uniform router. The quantile this
        #: names is `1 - qb_target` (one expert's share of the TOKENS is
        #: `p_i`; each token takes `topk` experts, so the assignments sum to
        #: `topk·n` and each expert's is `p_i·n`).
        self.qb_target = cfg.moe_topk / self.n_experts if self.n_experts else 0.0
        if self.qb:
            if self.ffn_form != "bank":
                loud(
                    f"moe_qb=true with ffn_form={self.ffn_form!r}: Quantile "
                    f"Balancing moves the ROUTED bank's `lfb_bias`, and this form "
                    f"has no such router. Escape: ffn_form='bank', or moe_qb=false."
                )
            if self.n_experts == 1:
                loud(
                    "moe_qb=true with n_experts=1: there is no router to "
                    "balance (one expert IS the FFN). Escape: n_experts >= 2, "
                    "or moe_qb=false."
                )
            if cfg.moe_topk == 0:
                loud(
                    "moe_qb=true with moe_topk=0: that is the dense blend, "
                    "which runs every expert for every token — there is no "
                    "routing decision for a quantile to make. Escape: "
                    "moe_topk >= 1, or moe_qb=false."
                )
        #: The FFN form's own modules. `bank` continues into the historical
        #: constructor below; the other two build ONLY their expert slot and
        #: return, so no router/shared/down parameter exists for them to lie
        #: about. The `ffn_form` branch lives here and at the top of `forward`
        #: — never inside the bank's dispatch.
        if self.ffn_form != "bank":
            self._init_ffn_form(cfg, dtype, device)
            # The bank-only attributes are set AFTER this early return, but
            # `state_dict(packed_form=...)`, `_load_from_state_dict` and the
            # checkpoint translation read `tsct_packed` for EVERY form — the
            # dense and `tsct` forms went `AttributeError` on save/load the day
            # this seam landed (caught by `test_checkpoints_round_trip_per_form`).
            # A non-bank form has no bank to pack, so `False` is the truth, not
            # a default chosen to silence the error.
            self.tsct_packed = False
            self._packed_params = ()
            # Every attribute the shared seams read must exist for the non-bank
            # forms too: `_banks` is never called there, but `invalidate_banks`
            # (the trainer's per-step call), the checkpoint translation and the
            # cache-key branch read the epoch flag unconditionally. Two separate
            # 200k-step launches died on the first step for this reason.
            self.moe_bank_epoch = bool(cfg.moe_bank_epoch)
            self.moe_combine = cfg.moe_combine
            self.moe_combine_dtype = cfg.moe_combine_dtype
            self.moe_permute = cfg.moe_permute
            self.moe_kernel = cfg.moe_kernel
            return
        self.router_jitter = getattr(cfg, "moe_router_jitter", 0.0)
        self.moe_kernel = cfg.moe_kernel
        #: The shared-basis COMPUTE shape (`config.ModelConfig.
        #: moe_shared_compute`): `gate_u`/`up_u` are read once as one
        #: `[d, 2·rank]` matrix and every token is projected once, instead of
        #: `_banks` materializing the E-fold stack and each expert repeating the
        #: same `d -> rank` product on its own rows. OFF is every pin on record.
        self.moe_shared_compute = cfg.moe_shared_compute
        #: The gather-fused FORWARD (`model/moe_sonic.py`, SonicMoE §5.1.1): a
        #: second switch on the same pass. `moe_kernel` is what runs when this
        #: arm cannot (`_gather_reason` → COUNTED `moe_gather_fallback`), and
        #: `config.validate` refuses the combinations the arm has no evidence
        #: for. The arm's own counters are `moe_gather_forward` (calls) and
        #: `moe_gather_rows` (rows read through the map) — a log line without
        #: them did not run the arm it is labelled with.
        self.moe_gather = cfg.moe_gather
        self.moe_offload = cfg.moe_offload
        #: The fused route head (`model/route_fused.py`): ONE kernel for
        #: softmax + LFB bias + top-k + the gate gather, instead of 4 tensor ops
        #: and ~9 CUDA kernels over the same `[n, E]`. The old path is what runs
        #: when the kernel cannot (`_route_fused_reason` → COUNTED
        #: `moe_route_fused_fallback`); `config.validate` refuses the two
        #: combinations the arm has no code for (adaptive width, `aux`).
        self.moe_route_fused = cfg.moe_route_fused
        #: The permute's algorithm (`argsort` today's stable sort, `count` the
        #: counting sort in `model/route_fused.py`). The permutation is the SAME
        #: tensor by construction; the knob is cost, not semantics.
        self.moe_permute = cfg.moe_permute
        #: HOW the routed scatter closes (`moe_combine`) and the dtype it stores
        #: (`moe_combine_dtype`): `torch` is today's three-pass fp32 form, `fused`
        #: the one-pass kernel. OFF (torch/fp32) is every pin on record.
        self.moe_combine = cfg.moe_combine
        self.moe_combine_dtype = cfg.moe_combine_dtype
        #: The bank cache's KEY (`moe_bank_epoch`). `False` keeps the `_version`
        #: tuple — a 1 536-parameter walk PER FORWARD (measured 2.09 ms/call at
        #: m400, x3 iterations = 6.3 ms/step of host time). `True` replaces it
        #: with ONE integer bumped by `invalidate_banks` (the trainer's
        #: per-optimizer-step call), by any `load_state_dict`, and by a resume.
        #: A stale entry is never served; one without an epoch or from the
        #: future is LOUD (`_banks`). `_bank_builds` is the host count the
        #: `moe_bank_builds` counter drains.
        self.moe_bank_epoch = cfg.moe_bank_epoch
        self._bank_epoch = 0
        self._bank_builds = 0
        #: The PACKED bank (`tsct_packed`, OFF): ONE `[E, rank, d_ffn]` master
        #: for `v` and one `[E, d, rank]` per input site instead of E separate
        #: parameters, so `_banks` casts a whole buffer once. The parent owns
        #: the parameters and the experts hold views (`bind_packed`); the tuple
        #: is the `_version` key's contribution from these buffers (they are not
        #: inside `self.experts`, so their versions must be part of the key).
        self.tsct_packed = cfg.tsct_packed
        self._packed_params: tuple[nn.Parameter, ...] = ()
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
        #: ---- TSCT-2 (`docs/architecture/2026-11-05-tsct-2.md`) -------------
        #: The three arms and their per-forward hooks. The checks run HERE too:
        #: a hand-built `ModelConfig` reaches this constructor without passing
        #: `config.validate`, and an arm that is silently INERT is the defect
        #: ADR-0019 bans.
        self.tsct_expert_basis = int(cfg.tsct_expert_basis)
        self.tsct_expert_lr = float(cfg.tsct_expert_lr)
        self.tsct_expert_revive = int(cfg.tsct_expert_revive)
        self.tsct_revive_noise = float(cfg.tsct_revive_noise)
        if self.tsct_expert_basis:
            if self.share_u_mode != "plane":
                loud(
                    f"tsct_expert_basis={self.tsct_expert_basis} with "
                    f"share_u_mode={self.share_u_mode!r}: the selector lives inside "
                    f"ONE shared plane. Escape: share_u_mode='plane', or "
                    f"tsct_expert_basis=0."
                )
            if self.tsct_expert_basis != cfg.rank:
                loud(
                    f"tsct_expert_basis={self.tsct_expert_basis} != rank={cfg.rank}: "
                    f"the selector's output is the GLU latent, and `v` is "
                    f"[rank, f]. Escape: tsct_expert_basis={cfg.rank}."
                )
            if cfg.shared_rank <= cfg.rank:
                loud(
                    f"tsct_expert_basis with shared_rank={cfg.shared_rank} <= "
                    f"rank={cfg.rank}: no wider plane to select from. Escape: "
                    f"shared_rank > rank."
                )
            if self.factor_qat != "none":
                loud(
                    f"tsct_expert_basis with factor_qat={self.factor_qat!r}: the "
                    f"shared plane is quantized before the selector. Escape: "
                    f"factor_qat='none'."
                )
            if cfg.moe_shared_compute:
                loud(
                    "tsct_expert_basis with moe_shared_compute=true: that arm uses a "
                    "constant selector and cannot carry a per-expert `w`. Escape: "
                    "moe_shared_compute=false."
                )
        if self.tsct_expert_lr or self.tsct_expert_revive:
            if cfg.n_experts < 2:
                loud(
                    f"tsct_expert_lr={self.tsct_expert_lr} / "
                    f"tsct_expert_revive={self.tsct_expert_revive} with "
                    f"n_experts={cfg.n_experts}: no second expert to reweight or "
                    f"revive. Escape: n_experts >= 2."
                )
            if cfg.moe_lb_mode == "aux":
                loud(
                    "tsct_expert_lr / tsct_expert_revive with moe_lb_mode='aux': "
                    "the aux arm never accumulates `lfb_counts`. Escape: "
                    "moe_lb_mode='sign' or 'prop'."
                )
            lb_step = cfg.moe_lb_gamma if cfg.moe_lb_mode == "prop" else cfg.moe_lb_coef
            if lb_step == 0.0:
                loud(
                    f"tsct_expert_lr / tsct_expert_revive with a zero "
                    f"{'moe_lb_gamma' if cfg.moe_lb_mode == 'prop' else 'moe_lb_coef'}: "
                    f"`_lfb_step` returns before accumulating `lfb_counts`. Escape: "
                    f"set the controller step positive."
                )
            if self.tsct_revive_noise < 0.0:
                loud(f"tsct_revive_noise={self.tsct_revive_noise} is negative. Escape: >= 0.")
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
                tsct_basis=self.tsct_expert_basis if self.share_u_mode == "plane" else 0,
                basis_shared=cfg.shared_rank,
                packed=cfg.tsct_packed,
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
                if self.tsct_expert_basis:
                    # TSCT-2: a WIDER shared plane, `[d, shared_rank]`, of which
                    # each expert selects its own `rank` columns. The first
                    # `rank` columns are today's exact draw (the global stream
                    # is consumed in the same order, so every non-new parameter
                    # stays bit-identical to OFF); the extra columns come from a
                    # LOCAL generator, so adding the arm does not shift the
                    # initialisation of anything else. At init `w = [I | 0]`
                    # drops the extra columns, so the forward is the OFF forward.
                    extra = cfg.shared_rank - cfg.rank
                    self.gate_u = nn.Parameter(
                        torch.cat(
                            (init_u(d, cfg.rank, device), tsct_basis_extra(d, extra, device)),
                            dim=1,
                        )
                    )
                    self.up_u = nn.Parameter(
                        torch.cat(
                            (init_u(d, cfg.rank, device), tsct_basis_extra(d, extra, device)),
                            dim=1,
                        )
                    )
                else:
                    self.gate_u = nn.Parameter(init_u(d, cfg.rank, device))
                    self.up_u = nn.Parameter(init_u(d, cfg.rank, device))
                for e in self.experts:
                    e.bind_shared_u(self.gate_u, self.up_u, cache=self._qat_cache)
                self._shared_params = (self.gate_u, self.up_u)
        #: PACKED BANK (`tsct_packed`): the parameters above were skipped by the
        #: experts; the parent owns one buffer per SITE and binds views. Under
        #: `share_u` the input side is already one tensor, so only `v` (the
        #: `[E, rank, d_ffn]` write — the 512-copy rebuild) and the `[E, 2,
        #: rank]` scales are packed here. Without `share_u` the two input bases
        #: are packed too: `packed_gate_u`/`packed_up_u`, one `[E, d, rank]`
        #: each rather than one concatenated `[E, 2d, rank]`. The split is
        #: deliberate: Muon orthogonalizes each expert's `[d, rank]` matrix, and
        #: a concatenated buffer would make the optimizer see a `[2d, rank]`
        #: matrix whose Gram mixes the gate and up rows and whose
        #: `max(1, m/n)^0.5` scale is not the per-site one. Two buffers = two
        #: casts, still 3 launches for the whole build at E=512.
        if self.tsct_packed:
            e, r = cfg.n_experts, cfg.rank
            self.packed_v = nn.Parameter(torch.randn(e, r, f, device=device) / math.sqrt(r))
            self.packed_s = nn.Parameter(torch.ones(e, 2, r, device=device))
            params: list[nn.Parameter] = [self.packed_v, self.packed_s]
            if not self.share_u:
                self.packed_gate_u = nn.Parameter(
                    torch.randn(e, d, r, device=device) / math.sqrt(d)
                )
                self.packed_up_u = nn.Parameter(torch.randn(e, d, r, device=device) / math.sqrt(d))
                params += [self.packed_gate_u, self.packed_up_u]
            # The optimizer marker: `[E, rows, cols]` is a BATCH of 2-D matrices,
            # and `train.optim.is_muon_param` reads this attribute (not the rank
            # alone) so the packed write keeps the per-expert Muon update. The
            # scales are NOT marked: `[E, 2, rank]` is a scale tensor and goes to
            # AdamW like every other `s`.
            for p in (
                self.packed_v,
                getattr(self, "packed_gate_u", None),
                getattr(self, "packed_up_u", None),
            ):
                if p is not None:
                    p._muon_batched = True
            self._packed_params = tuple(params)
            self._bind_packed()
        if self.tsct_expert_lr:
            # The optimizer's row-scale seam: an unpacked expert's `v` is one
            # parameter = one row of its shape group; the packed bank is ONE
            # parameter whose leading dim is E rows, so the same seam marks the
            # buffer and `_build_groups` expands the id per row. Not state, not
            # a buffer, never saved — the mapping is rebuilt from the module tree.
            if self.tsct_packed:
                self.packed_v._tsct_moe = self
            else:
                for j, e in enumerate(self.experts):
                    e.v._tsct_moe = self
                    e.v._tsct_index = j
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
        #: WHICH controller moves the bias (`config.MOE_LB_MODES`): `sign` is
        #: LFB's `-coef·sign(L−mean L)` (the default: every pin, preset and
        #: checkpoint on record), `prop` the proportional
        #: `-gamma·(L−mean L)/max(1,mean L)` (a far-off expert moves at the full
        #: step instead of one `coef` per forward), `aux` no bias controller at
        #: all — the Switch/GShard term is added to the loss instead.
        self.lb_mode = cfg.moe_lb_mode
        self.lb_gamma = cfg.moe_lb_gamma
        #: The `prop` controller's output bound, in LOGIT units. The measured
        #: failure is unbounded: `m400`'s saved bias spans [-218, +73] and in
        #: logit space that permanently overpowers a router whose own scores are
        #: ±5, so 9 of 512 experts run. `sign` is the legacy arm and is NOT
        #: clamped (its pins predate this field).
        self.lb_clamp = cfg.moe_lb_clamp
        self.aux_coef = cfg.moe_aux_coef
        #: The routing-health counter set (`_route_health`). OFF by default: it
        #: costs one `bincount` and four device scalars per routed forward.
        self.health = cfg.moe_routing_health
        #: The `aux` arm's per-forward terms, summed by `DormouseModel` after the
        #: loop (`begin_step` clears them). One per routed forward, so a
        #: T-iteration window sums T of them — the term is per forward, not per
        #: model, and dividing it by T would make the weight depend on depth.
        self._aux_terms: list[torch.Tensor] = []
        #: The LAST forward's routing histogram, on device (one `bincount` the
        #: routed path already pays for LFB). `scripts/route_hist.py` reads it —
        #: a probe that rebuilt the top-k from the UNBIASED logits (the old
        #: fallback) measures a router this model does not run.
        self._last_counts: torch.Tensor | None = None
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
        if self.moe_shared_compute:
            # A hand-built config skips `config.validate`; the shape refusals
            # have to live at the seam too (the same rule every other arm here
            # follows), and a heterogeneous bank is only visible once the
            # experts exist.
            self._assert_shared_compute()

    def _init_ffn_form(self, cfg, dtype: torch.dtype, device) -> None:
        """Build the two non-bank expert slots: `dense` and `tsct`."""
        self.capacity = 0.0
        self._aux_terms: list[torch.Tensor] = []
        self._bank_cache = None
        self._bank_epoch = 0
        self._bank_builds = 0
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
        if self.ffn_form == "dense":
            from .ffn import DenseFFN

            self.topk = 0
            self.experts = nn.ModuleList([DenseFFN(cfg, device=device)])
            return
        # ---- tsct -------------------------------------------------------
        if self.n_experts == 1 and cfg.moe_topk != 0:
            loud(
                f"ffn_form='tsct' with n_experts=1 and moe_topk={cfg.moe_topk}: "
                f"there is no second expert to route to. Escape: moe_topk=0 (the "
                f"one expert IS the dense blend)."
            )
        if self.n_experts > 1 and cfg.moe_topk > self.n_experts:
            loud(
                f"ffn_form='tsct' with moe_topk={cfg.moe_topk} > "
                f"n_experts={self.n_experts}: a width above the bank cannot be "
                f"filled. Escape: moe_topk <= n_experts."
            )
        rg = cfg.ffn_rank_gate or cfg.rank
        rd = cfg.ffn_rank_down or cfg.rank
        self.experts = nn.ModuleList(
            [
                TSCTGLU(
                    self.d,
                    self.f,
                    rg,
                    rd,
                    dtype=dtype,
                    factor_qat=cfg.factor_qat,
                    factor_qat_alpha=cfg.factor_qat_alpha,
                    act_quant=cfg.act_quant,
                    act_group=cfg.act_group,
                    device=device,
                )
                for _ in range(self.n_experts)
            ]
        )
        self.router = (
            nn.Linear(self.d, self.n_experts, bias=False, device=device)
            if self.n_experts > 1
            else None
        )
        self.dense_qat = cfg.dense_qat
        self.dense_qat_alpha = cfg.factor_qat_alpha
        self._qat_cache = QuantCache() if cfg.dense_qat != "none" else None
        if self.n_experts == 1:
            self.topk = 0

    def _tsct_forward(self, x: torch.Tensor, counters: Counters) -> torch.Tensor:
        """The `ffn_form="tsct"` dispatch: one expert, dense blend, or top-k.

        The routed arm is a bootstrap, not a kernel: one forward per expert
        that owns at least one row. `ponytail:` the `rows.any()` below is a
        host sync per expert; promote to a grouped/banked dispatch only if this
        form's bank is ever the recipe.
        """
        n = x.shape[0]
        if self.n_experts == 1:
            counters.count("ffn_tsct_forward", n)
            return self.experts[0](x, glu=self._glu)
        dt = self.experts[0].compute_dtype
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
        probs = F.softmax(logits, dim=-1)
        if self.topk == 0:
            y = torch.zeros(n, self.d, dtype=torch.float32, device=x.device)
            for e, expert in enumerate(self.experts):
                y = y + expert(x, glu=self._glu) * probs[:, e : e + 1]
            counters.count("ffn_tsct_dense_blend", n)
            return y
        index = logits.topk(self.topk, dim=-1).indices
        weight = torch.gather(probs, 1, index)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        y = torch.zeros(n, self.d, dtype=torch.float32, device=x.device)
        for e, expert in enumerate(self.experts):
            mask = index == e
            rows = mask.any(dim=-1)
            if not bool(rows.any()):
                continue
            sel = rows.nonzero(as_tuple=True)[0]
            ye = expert(x.index_select(0, sel), glu=self._glu)
            w = (weight * mask.to(weight.dtype)).sum(dim=-1).index_select(0, sel)
            y = y.index_add(0, sel, ye * w.unsqueeze(-1))
        counters.count("ffn_tsct_routed", n)
        return y

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        counters = Counters() if counters is None else counters
        if self.ffn_form != "bank":
            n = x.shape[0]
            assert x.ndim == 2 and x.shape[1] == self.d, (
                f"FFN input {tuple(x.shape)} != [n, {self.d}]"
            )
            if self.ffn_form == "dense":
                return self.experts[0](x, counters=counters)
            return self._tsct_forward(x, counters)
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
        if self.router_jitter > 0.0 and self.training:
            # TIE-BREAKING NOISE, Switch-Transformer style. Without it a bank of
            # near-identical experts at init collapses: measured 2026-11-04, one
            # expert took 3 800 of 8 192 tokens and the dropless bank needed
            # 5.56 GiB, i.e. the "512 experts" were doing the work of two. The
            # RNG is consumed only when the arm is on, so the OFF path's bits
            # and its RNG stream are untouched.
            logits = logits + torch.randn_like(logits) * self.router_jitter
            counters.count("moe_router_jitter")
        # The routed pass's cheap half: the fused route head replaces `softmax`,
        # the bias add, `topk` and the gate gather with ONE kernel
        # (`model/route_fused.py`). `probs` is then NOT materialized — which is
        # why the arm is refused under `moe_lb_mode="aux"` (its term reads the
        # softmax) and why the dense blend and the adaptive width keep the old
        # path. The fallback is COUNTED with its reason (ADR-0019).
        why = self._route_fused_reason(x) if self.moe_route_fused else None
        if why is not None:
            counters.count("moe_route_fused_fallback")
            prev = getattr(counters, "why", None)
            counters.why = f"{prev}; route: {why}" if prev else f"route: {why}"  # type: ignore[attr-defined]
        probs = None if self.moe_route_fused and why is None else F.softmax(logits, dim=-1)
        if probs is None:
            counters.count("moe_route_fused_calls")
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
                gate_u, up_u, v, gate_s, up_s = self._banks(dt, counters)
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
            if probs is None:
                # The fused route: same top-k (down to the tie rule) and the same
                # gathered gate, one launch.
                index, weight = route_fused.softmax_topk(logits, self.lfb_bias, self.topk)
                real = index
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
                    # The control, unchanged: ONE global width, no policy call at
                    # all (the off-arm bit-parity gate pins this branch against a
                    # digest measured before the policy existed).
                    index = raised.topk(self.topk, dim=-1).indices
                    weight = torch.gather(probs, 1, index)
                    real = index
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
                    # A weight-0 padding slot is neither a load (LFB) nor a dispatch
                    # (the aux term): both read the REAL assignments only.
                    real = index[active]
                    weight = torch.gather(probs, 1, index) * active.to(probs.dtype)
            # THIS forward's routing histogram, on device, computed ONCE: the
            # bias controller, the aux term and the health counters all read it,
            # and nothing here crosses to the host. A run that armed none of them
            # pays nothing — an OFF run's allocation count is pinned by
            # `tests/test_step_audit.py`, and an unconditional bincount here
            # pushed it over (measured: 30 vs the bound 27).
            aux_armed = self.lb_mode == "aux" and self.training and torch.is_grad_enabled()
            # `moe_qb` reads the histogram too: its controller still keeps
            # `lfb_counts` (the LR/revive seam) and the health counters honest.
            need = self.lb_active != 0.0 or self.health or aux_armed or (self.qb and self.training)
            counts = _bincount(real.reshape(-1), self.n_experts) if need else None
            self._last_counts = counts
            if self.qb and self.training:
                self._qb_step(counts, counters, logits)
            else:
                self._lfb_step(real, counters, counts)
            if counts is not None and self.health:
                self._route_health(counts, counters)
            if counts is not None and aux_armed:
                # COUNTED in `model.py`, exactly where the term is added to the
                # loss (ADR-0019): a `pure` eval computes no term and must not
                # report one. The coefficient is applied HERE so the arm's
                # mechanism is one expression, not two.
                self._aux_terms.append(self.aux_coef * switch_aux_loss(probs, real, self.n_experts))
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

    @property
    def lb_active(self) -> float:
        """The bias controller's per-forward step, `0.0` when it does not run.

        `aux` has NO bias controller (`moe_lb_coef`/`moe_lb_gamma` are inert
        there), and `sign`/`prop` read their own knob. ONE place answers "does
        this run move the bias": the step below and the resume refusal
        (`train/loop.py::load_model_state`) must never disagree — a refusal that
        read `moe_lb_coef` while the arm reads `moe_lb_gamma` would refuse a file
        it is about to use, or accept one it cannot reconstruct.
        """
        if self.lb_mode == "aux":
            return 0.0
        return self.lb_gamma if self.lb_mode == "prop" else self.lfb_coef

    def begin_step(self) -> None:
        """Open a model forward: the `aux` accumulator is PER FORWARD.

        `DormouseModel` calls this before the loop, so a T-iteration window
        accumulates T terms (the term is per routed forward; dividing it by T
        would tie its weight to the depth) and the next window starts empty. A
        recompute during backward appends its own copy AFTER the loss was read;
        clearing here is what keeps that copy out of the next step.
        """
        self._aux_terms = []

    def aux_loss(self) -> torch.Tensor | None:
        """The `aux` term the forward just accumulated, or `None` when it ran none."""
        if not self._aux_terms:
            return None
        return sum(self._aux_terms, torch.zeros((), device=self._aux_terms[0].device))

    def _route_health(self, counts: torch.Tensor, counters: Counters) -> None:
        """The routing-health counters: live experts, max share, normalized entropy.

        `counts` is this forward's `[E]` histogram (already on the device). The
        readings answer the question a LOSS cannot: a collapsed bank still
        lowers its loss (9 of 512 experts did the work of the bank at `m400` —
        `docs/records/reviews/2026-11-04-routing-collapse-CONFIRMED.md`; the bias-less
        router read 80/512, the file's own router 9/512), so the bank's health
        has to be counted where the routing is decided.

        All four counters are SUMS over the window (`count_device` accumulates:
        one D2H read per window, at the log cadence, never on the step path), so
        the reader divides by `moe_route_n`:

        * `moe_route_live` — experts with at least one assignment;
        * `moe_route_max_x1000` — the busiest expert's share of the forward's
          assignments × 1000 (integer; 1000/`E` is uniform, 1000 is one expert);
        * `moe_route_top10_x1000` — the ten busiest experts' share × 1000. This
          is the number the 2026-11-04 finding is stated in (top-10 held 50.1%
          of the bias-less router's tokens and 100% of the saved-controller
          one), so it is counted, not left to a script;
        * `moe_route_ent_x1000` — `H(p)/log(E)` × 1000, the histogram's entropy
          normalized so 1000 IS a uniform bank and 0 is a single live expert.
          Rounded, not truncated: uniform reads 1000, not 999;
        * `moe_route_n` — the number of routed forwards accumulated.

        Computed in float64: `H` is a sum of `E` logs, and this is the one place
        a 1-unit reading (one per mille of the bank) is the difference between
        "alive" and "collapsing". `p.clamp_min(...)` instead of a mask: a
        boolean index's output LENGTH is a device value, i.e. a host sync.
        """
        total = counts.sum().clamp_min(1)
        p = counts.to(torch.float64) / total.to(torch.float64)
        ent = -(p * p.clamp_min(1e-300).log()).sum()
        units = 1000.0 / math.log(self.n_experts) if self.n_experts > 1 else 0.0
        one = torch.ones((), dtype=torch.int64, device=counts.device)
        counters.count_device("moe_route_n", one)
        counters.count_device("moe_route_live", (counts > 0).sum())
        counters.count_device("moe_route_max_x1000", counts.max() * 1000 // total)
        counters.count_device(
            "moe_route_top10_x1000",
            torch.topk(counts, min(10, self.n_experts)).values.sum() * 1000 // total,
        )
        counters.count_device("moe_route_ent_x1000", (ent * units).round().to(torch.int64))

    def _lfb_step(
        self, index: torch.Tensor, counters: Counters, counts: torch.Tensor | None = None
    ) -> None:
        """The bias move: on-device, gradient-free, trains-mode only.

        `sign` (LFB, arXiv:2408.15664; the default) is
        `b_e -= coef · sign(L_e − mean L)` over the CUMULATIVE routed counts
        `L_e` — the integrator the paper specifies. The A/B of 2026-10-15
        caught the per-forward variant making drops WORSE (42,258→45,325 at
        cap 32): bang-bang control at one-step lag oscillates the whole
        bank; the cumulative mean is what converges. A MEASURED correction
        to the first implementation, not an opinion.

        `prop` is `b_e -= gamma · (L_e − mean L) / max(1, mean L)` over THIS
        forward's histogram, clamped to ±`moe_lb_clamp`. PER-WINDOW, not
        cumulative: a cumulative ratio has no equilibrium under a bang-bang step
        — the measured vector drifted for 150k steps and then froze the order.
        The normalized deviation is scale-free (a dead expert is `-1`, an
        overloaded one is `+share`), so `gamma` means the same thing at any batch
        size, and the clamp keeps the controller inside the router's own logit
        scale instead of the ±200 the sign arm reached. `aux` returns here: that
        arm's controller is the loss term.

        `moe_qb` does not come through here (`_qb_step` replaces the step with
        the score quantile); `lfb_counts` is still accumulated there —
        `expert_lr_scales` and `revive_experts` read it and are not `moe_qb`'s
        to switch off.
        """
        if not self.training or self.lb_mode == "aux":
            return
        step = self.lb_active
        if step == 0.0:
            return
        with torch.no_grad():
            if counts is None:
                counts = _bincount(index.reshape(-1), self.n_experts)
            self.lfb_counts.add_(counts.float())
            if self.lb_mode == "prop":
                # PER-WINDOW, not cumulative: a cumulative ratio has no
                # equilibrium under a bang-bang step (measured: `m400`'s saved
                # bias spans [-218, +73] and the order froze at 9 live experts).
                load = counts.float()
                mean = load.mean()
                moved = self.lb_gamma * (load - mean) / mean.clamp_min(1.0)
                self.lfb_bias.sub_(moved).clamp_(-self.lb_clamp, self.lb_clamp)
            else:
                self.lfb_bias.sub_(
                    self.lfb_coef * torch.sign(self.lfb_counts - self.lfb_counts.mean())
                )
        counters.count("moe_lfb_updates")

    def _qb_step(
        self, counts: torch.Tensor | None, counters: Counters, logits: torch.Tensor
    ) -> None:
        """Quantile Balancing's per-forward seam: state, counters, then the bias.

        `_lfb_step` is not called: QB has no rate, so nothing there describes
        it. `lfb_counts` still accumulates (the LR/revive seam reads it).
        """
        if counts is None:
            loud(
                "moe_qb=true ran a routed forward without its histogram: the "
                "`need` gate in `forward` must include `moe_qb`. Escape: fix "
                "the gate, or moe_qb=false."
            )
        with torch.no_grad():
            self.lfb_counts.add_(counts.float())
            self._qb_set_bias(logits)
        counters.count("moe_qb")
        note = f"qb: bias = score quantile (p={self.qb_target:.4f}, bins={_QB_BINS})"
        prev = getattr(counters, "why", None)
        if note not in (prev or ""):
            counters.why = f"{prev}; {note}" if prev else note  # type: ignore[attr-defined]

    def _qb_set_bias(self, logits: torch.Tensor) -> None:
        """Quantile Balancing (Kimi K3 §2.3.3): bias from the score quantile.

        For expert `i` with target load `p_i = topk/E`, `q_i` is the score above
        which a `p_i` fraction of THIS forward's tokens lie — the target
        quantile `1 − p_i`. The router ADDS the bias to the score, so the bias
        that puts expert `i`'s target quantile on a common level is
        `b_i = τ − q_i`; `τ` is the mean over the bank, and any common `τ` is
        invisible to a top-k (it shifts every expert's score of a token by the
        same amount), so the mean is used to keep a saved `lfb_bias` centred
        instead of parked at `−q_i`'s offset. The rule is STATELESS: `q_i`
        depends on the raw scores only, so the next forward with the same
        scores writes the same bias — the drift `sign`/`prop` show (a
        cumulative integrator that has no equilibrium under a bang-bang step)
        cannot happen, and an already-balanced batch moves the bias by 0.

        The histogram is built on DEVICE, per expert, over the batch's own
        score range: `round((s − min)/(max − min)·BINS)`, one `scatter_add_`,
        one `cumsum`, and the target bin is `(cdf < p_i·n).sum()`. No sort (the
        quantile of `[n,E]` would be `E` sorts) and no host round trip — not a
        scalar extraction, not a list conversion, not a CPU copy — so the
        update is capturable by a CUDA graph.
        """
        scores = logits.detach().float()  # bf16 scores would lose the bins
        n, e = scores.shape
        lo = scores.min(dim=0).values
        hi = scores.max(dim=0).values
        span = (hi - lo).clamp_min(1e-12)  # a constant column: all in bin 0
        idx = ((scores - lo) * (float(_QB_BINS) / span)).to(torch.int64)
        idx.clamp_(0, _QB_BINS - 1)
        hist = torch.zeros(_QB_BINS, e, dtype=torch.float32, device=scores.device)
        hist.scatter_add_(0, idx, torch.ones((), device=scores.device).expand_as(idx))
        # First bin whose cumulative count reaches `p_i·n`: the value at the
        # target quantile, as the bin's UPPER edge (so the count at or above it
        # is not above the target). `cdf` is non-decreasing, so `(cdf < target)`
        # counts the bins strictly below the crossing and the sum IS the bin.
        cdf = hist.cumsum(dim=0)
        target = torch.full((1, e), self.qb_target * n, device=scores.device)
        j = (cdf < target).sum(dim=0).clamp_(0, _QB_BINS - 1)
        q = lo + (j.to(lo.dtype) + 1.0) * span / float(_QB_BINS)
        self.lfb_bias.copy_(q.mean() - q)

    def expert_lr_scales(self) -> torch.Tensor:
        """TSCT-2's per-expert LR multiplier, `[E]`, PURE from the device counts.

        No host read and no state: the optimizer recomputes it every step from
        `lfb_counts` (the LFB integrator's device-side counter), so a resume
        needs nothing extra and a stale scale cannot outlive a step.
        """
        return expert_lr_multipliers(self.lfb_counts, self.tsct_expert_lr)

    def revive_experts(self, step: int, seed: int) -> int:
        """TSCT-2's revive, once per `tsct_expert_revive` optimizer steps.

        The coldest expert takes the hottest expert's `v`, `gate_s`, `up_s` (and
        its selector `w` when the basis arm is on) plus
        `σ = tsct_revive_noise · std(v_hot)` of Gaussian noise, and BOTH experts'
        `lfb_counts`/`lfb_bias` entries are reset so the controller starts them
        from zero traffic. Returns 1 when it ran, 0 when the cadence has not
        fired or the bank has no traffic yet.

        The two indices are the ONE declared host read at this cadence: a
        device `argmin`/`argmax` still needs a Python int to index the expert
        list, and there is no `[E, ·]` stack of `v` to gather from (it would be
        512 copies of the whole bank). It runs once per N steps, never on the
        LR path, and the values themselves are device-side. The noise generator
        is seeded `(seed + step)`, so a replayed step revives identically.

        Under `share_u_mode='off'` the expert also owns a private `gate_u`/
        `up_u`; this method copies only the three tensors the mechanism names
        (`v`, `gate_s`, `up_s`) and does NOT claim to revive the private basis
        — the arm is specified for the shared-basis bank the A/B runs.
        """
        if not self.training or self.tsct_expert_revive <= 0:
            return 0
        if step % self.tsct_expert_revive != 0:
            return 0
        counts = self.lfb_counts
        hot = int(counts.argmax())
        cold = int(counts.argmin())
        if cold == hot or float(counts[hot]) <= 0.0:
            # No traffic yet (step 0) or every expert tied: there is no
            # starvation to answer and copying an expert onto itself would only
            # inject noise.
            return 0
        donor = self.experts[hot]
        receiver = self.experts[cold]
        g = torch.Generator(device=counts.device).manual_seed(seed + step)
        with torch.no_grad():
            sigma = self.tsct_revive_noise * donor.v.detach().std()
            pairs = [
                (receiver.v, donor.v),
                (receiver.gate_s, donor.gate_s),
                (receiver.up_s, donor.up_s),
            ]
            if donor.w is not None and receiver.w is not None:
                pairs.append((receiver.w, donor.w))
            for dst, src in pairs:
                noise = torch.randn(src.shape, generator=g, device=src.device, dtype=src.dtype)
                dst.copy_(src.detach() + noise * sigma)
            counts[cold] = 0.0
            counts[hot] = 0.0
            self.lfb_bias[cold] = 0.0
            self.lfb_bias[hot] = 0.0
        return 1

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
        self,
        index: torch.Tensor,
        valid: torch.Tensor | None = None,
        counters: Counters | None = None,
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

        `moe_permute="count"` replaces the stable argsort with the counting sort
        in `model/route_fused.py`. The permutation is IDENTICAL (stable by
        construction, which the gate pins element-wise), so nothing downstream
        can tell which arm ran except the counters — and a kernel that cannot
        serve the call falls back to the argsort COUNTED (`moe_permute_fallback`
        + the reason), never silently.
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
        if self.moe_permute == "count":
            why = route_fused.reason(index)
            if why is None:
                order = route_fused.count_permute(safe, groups)
                if counters is not None:
                    counters.count("moe_permute_count")
            else:
                if counters is not None:
                    counters.count("moe_permute_fallback")
                    prev = getattr(counters, "why", None)
                    counters.why = (  # type: ignore[attr-defined]
                        f"{prev}; permute: {why}" if prev else f"permute: {why}"
                    )
                order = torch.argsort(safe, stable=True)
        else:
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

    def _assert_shared_compute(self) -> None:
        """Refuse every shape the shared projection cannot read — LOUD, with the escape.

        `config.validate` owns the same refusals on the CLI path; this is the
        seam a hand-built config and a hand-mutated bank reach (ADR-0019: an arm
        asked for and silently not run is a defect). The heterogeneous check is
        the one a config cannot see: a per-expert adaptive rank makes the E-fold
        `gate_s`/`up_s`/`v` stacks impossible, and the projection's one `W`
        would have to be E different matrices.
        """
        if self.share_u_mode == "off":
            loud(
                "moe_shared_compute=true with share_u_mode='off': the projection "
                "reads the ONE shared `gate_u`/`up_u [d, rank]`; with no shared "
                "basis every expert owns its own `U`. Escape: share_u_mode="
                "'plane' (or share_u=true), or moe_shared_compute=false."
            )
        if self.share_u_mode == "dict":
            loud(
                "moe_shared_compute=true with share_u_mode='dict': each expert "
                "reads a different column subset of the dictionary, so no single "
                "[d, 2·rank] matrix exists to project the tokens with once. "
                "Escape: share_u_mode='plane', or moe_shared_compute=false."
            )
        if self.experts[0].rank < 1:
            loud(
                f"moe_shared_compute=true with rank={self.experts[0].rank}: the "
                f"projection's latent has no columns. Escape: rank >= 1, or "
                f"moe_shared_compute=false."
            )
        shapes = [
            (tuple(ex.gate_s.shape), tuple(ex.up_s.shape), tuple(ex.v.shape)) for ex in self.experts
        ]
        if any(s != shapes[0] for s in shapes[1:]):
            loud(
                "moe_shared_compute=true with a per-expert ADAPTIVE RANK: the "
                f"bank's factor shapes differ (expert 0 {shapes[0]}, first "
                f"mismatch {next(s for s in shapes[1:] if s != shapes[0])}), so "
                "one shared [d, 2·rank] projection cannot feed every expert's "
                "bank. Escape: one rank for every routed expert, or "
                "moe_shared_compute=false."
            )
        if self.moe_gather == "kernel":
            loud(
                "moe_shared_compute=true with moe_gather='kernel': the gather "
                "arm owns the routed forward, so the shared projection would "
                "never run while the flag claimed it did. Escape: "
                "moe_gather='off', or moe_shared_compute=false."
            )

    @torch.compiler.disable
    def _shared_pair(self, dt: torch.dtype) -> torch.Tensor:
        """`[gate_u | up_u]` as ONE `[d, 2·rank]` matrix — the bank's own parameters.

        Not a copy kept on the side: `plane` mode owns these two tensors and the
        bank materializes their E-fold stack elsewhere. Reading them here is
        what makes the two shapes provably the same arithmetic; the parity gate
        (`tests/test_moe_shared_compute.py`) is what proves the values agree.
        `factor_qat` rides the SAME per-step `QuantCache` `_banks` uses, so the
        one projection is quantized once, not once per call site.
        """
        assert self.share_u_mode == "plane", "the shared projection needs `plane`"
        gate_u, up_u = self.gate_u, self.up_u
        if self.factor_qat != "none":
            gate_u = factor_qat_project(
                gate_u, self.factor_qat, self.factor_qat_alpha, cache=self._qat_cache
            )
            up_u = factor_qat_project(
                up_u, self.factor_qat, self.factor_qat_alpha, cache=self._qat_cache
            )
        return torch.cat((gate_u.to(dt), up_u.to(dt)), dim=1)

    def _shared_projection(
        self, x: torch.Tensor, dt: torch.dtype, counters: Counters | None
    ) -> torch.Tensor:
        """`lat = x @ [gate_u | up_u]` — ONE GEMM over the TOKEN rows `[n, 2·rank]`.

        The old shape projected the `n·k` assignment rows through an E-fold
        `[d, rank]` stack (twice — gate and up). The shared basis is the same
        for every assignment of a token, so one projection per token is the
        same math with the duplicated rows removed. Counters are DEVICE-side:
        the row count never crosses to the host on the step path.
        """
        lat = x if x.dtype == dt else x.to(dt)
        lat = lat @ self._shared_pair(dt)
        if counters is not None:
            counters.count("moe_shared_compute_calls")
            counters.count_device(
                "moe_projection_rows",
                torch.full((), x.shape[0], dtype=torch.int64, device=x.device),
            )
        return lat

    @torch.compiler.disable
    def _shared_select(self, dt: torch.dtype, device: torch.device) -> tuple[torch.Tensor, ...]:
        """The two constant `[E, 2·rank, rank]` selectors the grouped GEMM takes.

        `[I; 0]` reads the gate half of the combined latent, `[0; I]` the up
        half, so the shipped kernel's `xg @ gate_u` / `xg @ up_u` become column
        selections — the projection they used to do already happened once, over
        the tokens. Cached per `(dtype, device, rank, E)`: constants, not state,
        and never in `state_dict`.
        """
        rank = self.experts[0].rank
        key = (dt, str(device), rank, self.n_experts)
        cached = getattr(self, "_shared_select_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        eye = torch.eye(rank, dtype=dt, device=device)
        zero = torch.zeros_like(eye)
        gate = torch.cat((eye, zero), dim=0).unsqueeze(0)
        up = torch.cat((zero, eye), dim=0).unsqueeze(0)
        gate = gate.expand(self.n_experts, -1, -1).contiguous()
        up = up.expand(self.n_experts, -1, -1).contiguous()
        self._shared_select_cache = (key, gate, up)
        return gate, up

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

        Two keys implement the "one build per step" contract, and the flag picks:

        * `moe_bank_epoch = false` (the default) keys the cache on the parameters'
          `_version` tuple and CLEARS it here. That invented the step boundary
          because a `_version` key could not describe a weight move that leaves
          `_version` frozen — **a CUDA-graph replay does exactly that** (measured
          2026-10-19: an eager step moves 0->1; the same step replayed from a
          captured graph leaves `_version` FROZEN while the value moves). With
          `opt_graph = true` the key stopped moving at capture and the bank
          served the step-5 build for the whole run: the experts' FORWARD was
          frozen while the optimizer kept stepping them with stale gradients
          (they drifted ~44x; the saved checkpoint then exploded in any fresh
          process, and the live evals could not show it). The cost of that key is
          a 1 536-parameter walk on EVERY `_banks` call — measured 2 092 us/call
          at m400, 6.3 ms/step of HOST time at max_iter 3.
        * `moe_bank_epoch = true` BUMPS one integer and leaves the entry in
          place; `_banks` compares the entry's epoch and NEVER serves an older
          one (it drops it before rebuilding, so the peak is not two banks).
          The trainer's per-step call is unchanged; `_load_from_state_dict`
          bumps the same counter on a resume, so a pre-load bank cannot survive
          a load. No parameter is read on a hit.

        Either way this runs once per TRAINING step, right after the parameters
        move: the cache keeps its real job (one build per step, reused by the
        loop's iterations) and can no longer outlive a step.
        """
        self._bank_epoch += 1
        # Under the epoch key the entry is left in place ON PURPOSE: the next
        # `_banks` sees `entry_epoch < _bank_epoch`, drops it before allocating
        # the new one (so the peak is not two banks) and rebuilds. Clearing here
        # as well would hide the stale branch from every step — the branch the
        # frozen-epoch gate exists to exercise.
        if not self.moe_bank_epoch:
            self._bank_cache = None

    # -- packed bank: one buffer per site, and a checkpoint readable both ways --
    def _bind_packed(self) -> None:
        """Point every expert at its row of the parent's buffers (views)."""
        gate_u = getattr(self, "packed_gate_u", None)
        for e, ex in enumerate(self.experts):
            ex.bind_packed(
                self.packed_v[e],
                self.packed_s[e, 0],
                self.packed_s[e, 1],
                None if gate_u is None else gate_u[e],
                None if gate_u is None else self.packed_up_u[e],
            )

    def _packed_family(self, prefix: str) -> dict[str, tuple[str, ...]]:
        """`packed key -> the per-expert keys it replaces` — the FULL mapping.

        Both directions read it: a pack filters to the buffers the target model
        owns, an unpack reads whichever packed keys the checkpoint carries.
        """
        return {
            "packed_v": tuple(f"{prefix}experts.{e}.v" for e in range(self.n_experts)),
            # ALL gate_s then ALL up_s: the pack is `[E, 2, rank]`, so the
            # gather's two halves are `vals[:E]` and `vals[E:]`.
            "packed_s": tuple(f"{prefix}experts.{e}.gate_s" for e in range(self.n_experts))
            + tuple(f"{prefix}experts.{e}.up_s" for e in range(self.n_experts)),
            "packed_gate_u": tuple(f"{prefix}experts.{e}.gate_u" for e in range(self.n_experts)),
            "packed_up_u": tuple(f"{prefix}experts.{e}.up_u" for e in range(self.n_experts)),
        }

    def _translate(
        self, state_dict, prefix: str, *, to_packed: bool, own_only: bool = False
    ) -> None:
        """Rewrite the model half between the packed and per-expert forms.

        `to_packed`: gather `[E, ...]` from the per-expert keys (an old
        checkpoint into a packed model). Otherwise expand the packed buffer into
        per-expert keys (a packed checkpoint into an unpacked model). A family
        is translated only when the TARGET carries it: a `share_u` model has no
        per-expert `u`, so those keys are left for the loader's own
        missing/unexpected report (a config mismatch, not a format).
        `own_only` is the LOAD path's rule (the target model must own the packed
        buffer); `state_dict(packed_form=True)` writes the form even when the
        model itself is unpacked, so the export path passes `False`.
        """
        family = self._packed_family(prefix)
        # The per-expert form of the scales is TWO leaves per expert; the pack
        # is `[E, 2, rank]`, so the gather has to interleave (`dim=1`).
        for packed_key, old_keys in family.items():
            packed_full = f"{prefix}{packed_key}"
            present = [k in state_dict for k in old_keys]
            if to_packed:
                if own_only and not hasattr(self, packed_key):
                    # The target does not own this site (a share_u model has no
                    # per-expert `u`): leave the keys for the loader's own
                    # missing/unexpected report — a config mismatch, not a format.
                    continue
                if packed_full in state_dict:
                    if any(present):
                        loud(
                            f"{packed_full} and its per-expert keys are BOTH in the "
                            f"checkpoint: one bank, two formats. Escape: keep one "
                            f"(drop the packed key, or the experts.* keys)."
                        )
                    continue
                if not any(present):
                    continue
                if not all(present):
                    missing = [k for k, ok in zip(old_keys, present, strict=True) if not ok]
                    loud(
                        f"{packed_full}: a packed load needs EVERY per-expert key, "
                        f"missing {missing[:4]}. Escape: a complete checkpoint, or "
                        f"tsct_packed=false."
                    )
                vals = [state_dict.pop(k) for k in old_keys]
                if packed_key == "packed_s":
                    n = self.n_experts
                    gate = torch.stack(vals[:n])
                    up = torch.stack(vals[n:])
                    state_dict[packed_full] = torch.stack([gate, up], dim=1)
                else:
                    state_dict[packed_full] = torch.stack(vals)
                continue
            # packed -> per-expert
            if packed_full not in state_dict:
                continue
            if any(present):
                loud(
                    f"{packed_full} and its per-expert keys are BOTH in the "
                    f"checkpoint: one bank, two formats. Escape: keep one."
                )
            val = state_dict.pop(packed_full)
            if packed_key == "packed_s":
                n = self.n_experts
                for e in range(n):
                    state_dict[old_keys[e]] = val[e, 0]
                    state_dict[old_keys[n + e]] = val[e, 1]
            else:
                for e, key in enumerate(old_keys):
                    state_dict[key] = val[e]

    def state_dict(self, destination=None, prefix="", keep_vars=False, packed_form=None):
        """The bank's half in either form (`packed_form`, default: the model's).

        `None` writes what this model IS (`packed_v`/`packed_s` when packed,
        `experts.*` otherwise) — so a checkpoint is honest about the layout it
        came from. `packed_form=True|False` rewrites that half into the other
        form, which is what makes a packed run's checkpoint readable by an
        unpacked model (and the reverse) without a converter script. The values
        are the same tensors: the translation only regroups the keys.
        """
        dest = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        form = self.tsct_packed if packed_form is None else bool(packed_form)
        if form != self.tsct_packed:
            self._translate(dest, prefix, to_packed=form)
        return dest

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        """A load is a weight move AND a format conversion.

        The bank cache is dropped and the epoch bumped (no pre-load bank may be
        served). The model half is translated across the packed and per-expert
        layouts before the base loader runs: an old per-expert checkpoint
        gathers into `packed_v`/`packed_s` (and the u pair when the model owns
        it), and a packed checkpoint expands into `experts.{e}.{v,gate_s,
        up_s,...}` for an unpacked model. After the copy the per-expert views
        are rebound — a loader that replaced the storage would otherwise leave
        every expert reading the old tensor.
        """
        family = self._packed_family(prefix)
        has_packed = any(f"{prefix}{k}" in state_dict for k in family)
        has_old = any(k in state_dict for keys in family.values() for k in keys)
        if has_packed and has_old:
            loud(
                f"{prefix}: the checkpoint carries BOTH the packed bank and the "
                f"per-expert keys. One bank, two formats. Escape: keep one."
            )
        if self.tsct_packed and has_old:
            self._translate(state_dict, prefix, to_packed=True, own_only=True)
        elif not self.tsct_packed and has_packed:
            self._translate(state_dict, prefix, to_packed=False)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if self.tsct_packed:
            # `assign=True` can REPLACE the parameter objects; the tuple and the
            # expert views must follow the live ones.
            self._packed_params = (
                self.packed_v,
                self.packed_s,
                *((self.packed_gate_u, self.packed_up_u) if hasattr(self, "packed_gate_u") else ()),
            )
            self._bind_packed()
        self._bank_cache = None
        self._bank_epoch += 1

    @torch.compiler.disable
    def _banks(self, dt: torch.dtype, counters: Counters | None = None) -> tuple[torch.Tensor, ...]:
        """The expert bank as `[E, ·]` tensors for the batched matmuls — CACHED IN `dt`.

        Two caches' lessons paid here. (1) The naive form stacked per forward
        (1.25 GB at b1). (2) The first cache stored the fp32 stack and called
        `.to(dt)` EVERY forward — measured ~15 ms host + ~4 ms copies per
        call at b1 (1.79 GB re-converted twice per step at max_iter 2; the
        TSCT-super-kernel bench flagged it as larger than the Triton kernel
        win). So this cache stores the CONVERTED tensors: one conversion per
        step is the honest floor, zero conversions inside the step.

        **The key is the epoch** (`moe_bank_epoch`, the entry's `cached[3]`)
        **or, with the flag off, the `_version` tuple** (`cached[0]`, the old
        path every pin on record runs). The epoch path reads no parameter on a
        hit — the `_version` walk is 2 092 us/call at m400 (E=512, 1 536
        params, measured 2026-11-05) and 6.3 ms/step at max_iter 3. An entry
        whose epoch is OLDER is dropped before the rebuild; one with no epoch
        (built before the scheme) or one from the FUTURE is LOUD, because
        neither can be vouched for. The build count is unchanged either way:
        the trainer's `invalidate_banks_all` already made it one per step, and
        a `[E, ·]` stack of 512 separate parameters is 512 casts + a stack.

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
        back to eager (measured 2026-10-18, `docs/records/research/2026-10-18-compile-moe.md`;
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

        **Under `tsct_packed`** the per-expert branch of `_one` is gone: `v`
        (and the u pair when `share_u` is off) is ONE `[E, ·, ·]` buffer, so the
        build is `buffer.to(dt)` — one cast instead of E casts + a stack — and
        the scales are two views of `packed_s`. Measured at m400b/E=512:
        521 -> 5 launches and 1 681 -> 1 010 MB of operand traffic per build.
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
                self._packed_params[0]
                if self._packed_params
                else (self._shared_params[0] if self._shared_params else self.experts[0].gate_u)
            ).requires_grad
        )
        if self.moe_bank_epoch:
            # NO PARAMETER WALK. The freshness contract is the entry's own
            # `epoch` (checked below) — the trainer's `invalidate_banks_all`,
            # called after the only thing that moves the weights, plus this
            # module's load hook. Never `_version`, which a CUDA-graph replay
            # freezes while the values move (`test_bank_invalidation.py`).
            key = (dt, self.factor_qat, self.factor_qat_alpha, want)
        elif self.tsct_packed:
            # Same shape as the unpacked key plus the packed buffers' versions:
            # the experts own none, and a step that moved only `packed_v` must
            # rebuild the bank.
            key = (
                dt,
                self.factor_qat,
                self.factor_qat_alpha,
                tuple(p._version for e in self.experts for p in e.parameters()),
                tuple(p._version for p in self._packed_params),
                tuple(p._version for p in self._shared_params),
                want,
            )
        else:
            key = (
                dt,
                self.factor_qat,
                self.factor_qat_alpha,  # the anneal ramp moves this every step
                tuple(p._version for e in self.experts for p in e.parameters()),
                # The shared basis is not inside `self.experts`, so its
                # `_version` is not in the tuple above: an optimizer step that
                # moved ONLY `u_dict` (or `gate_u`/`up_u`) must still rebuild.
                tuple(p._version for p in self._shared_params),
                want,
            )
        cached = getattr(self, "_bank_cache", None)
        if cached is not None:
            entry_epoch = cached[3] if len(cached) > 3 else None
            if self.moe_bank_epoch:
                # Refuse to SERVE a bank the epoch does not vouch for. An entry
                # from an older epoch is simply rebuilt (the normal path); one
                # with no epoch, or from the future, is a wiring error and says
                # so instead of returning weights nobody can account for.
                if entry_epoch is None:
                    raise loud(
                        "the bank cache carries no epoch while moe_bank_epoch=true: an "
                        "entry built before the epoch scheme cannot be trusted. Escape: "
                        "invalidate_banks_all(model) after the load, or moe_bank_epoch=false."
                    )
                if entry_epoch > self._bank_epoch:
                    raise loud(
                        f"the bank cache is from epoch {entry_epoch}, the model is at "
                        f"{self._bank_epoch}: a cache cannot be newer than its model. "
                        f"Escape: invalidate_banks_all(model) and rebuild."
                    )
                if entry_epoch < self._bank_epoch:
                    # Stale: never served, and dropped BEFORE the rebuild so the
                    # old [E,·] stacks are freed instead of held beside the new.
                    self._bank_cache = None
                    cached = None
            # A TRACKED entry serves a no-grad forward as it stands — same
            # tensors, no gradient wanted — so the eval keeps the cache hit. The
            # reverse (an untracked entry serving a tracked forward) is the
            # defect above, and it is what the key and the second clause forbid.
            reusable = cached is not None and (
                cached[0] == key or (not grad and cached[0][:-1] == key[:-1] and cached[0][-1])
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
                    if self.tsct_expert_basis:
                        # TSCT-2: fold each expert's selector into its EFFECTIVE
                        # basis, `gate_u @ w_j.T` -> `[d, rank]`, so every bank
                        # arm below (padded/batched/bucketed/triton/gather) keeps
                        # the exact operand shape it had and the per-token MAC is
                        # UNCHANGED. The fold itself is one batched matmul per
                        # rebuild (once per optimizer step), not per token.
                        if counters is not None:
                            counters.count("tsct_basis_fold")
                        w = torch.stack([e.w.to(dt) for e in self.experts])
                        return torch.einsum("dr,ekr->edk", u, w).contiguous()
                    if self.tsct_packed:
                        # A VIEW, not a copy: the E slices share one `[d, r]`
                        # storage and `bmm`'s backward through `expand` sums the
                        # E contributions into the same parameter. This is what
                        # keeps the packed build at 1 cast per site (the whole
                        # `_banks` rebuild: v + 2 scales = 3 launches).
                        return u.unsqueeze(0).expand(len(self.experts), -1, -1)
                    return u.unsqueeze(0).expand(len(self.experts), -1, -1).contiguous()
                # PACKED SITE (`tsct_packed`): one `[E, ·, ·]` buffer, ONE cast.
                # This is the whole arm: no per-expert `.to`, no `stack`.
                packed = getattr(self, f"packed_{name}", None)
                if packed is not None:
                    if self.factor_qat == "none":
                        return packed.to(dt)
                    return qat_project_stacked(
                        packed, self.factor_qat, self.factor_qat_alpha, dim=(1, 2)
                    ).to(dt)
                params = [getattr(e, name) for e in self.experts]
                if self.factor_qat == "none":
                    return torch.stack([t.to(dt) for t in params])
                return qat_project_stacked(
                    torch.stack(params), self.factor_qat, self.factor_qat_alpha, dim=(1, 2)
                ).to(dt)

            mats = [_one(name) for name in ("gate_u", "up_u", "v")]
            if self.tsct_packed:
                # Views, then ONE cast each: the scales are fp32 masters and the
                # bank is `dt`, exactly as the unpacked `stack(...).to(dt)` path.
                scales = tuple(self.packed_s[:, i].to(dt) for i in (0, 1))
            else:
                scales = tuple(
                    torch.stack([getattr(e, name) for e in self.experts]).to(dt)
                    for name in ("gate_s", "up_s")
                )
            cached = (
                key,
                tuple(mats),
                scales,
                self._bank_epoch if self.moe_bank_epoch else None,
            )
            self._bank_builds += 1
            if counters is not None:
                # Counted HERE, inside the `torch.compiler.disable` frame: a
                # `counters` mutation in `forward` is a dynamo guard on the
                # dict's length, and the first build of a fresh Counters moves it
                # (measured: `recompile_limit (8), last reason: len(counters) != 9`
                # in `tests/test_compile_moe.py`).
                counters.count("moe_bank_builds")
            if want:
                self._bank_cache = cached
        mats, scales = cached[1], cached[2]
        return (mats[0], mats[1], mats[2], scales[0], scales[1])

    def _assert_homogeneous(self) -> None:
        """Refuse a bank whose experts have different factor shapes — LOUD.

        `moe_kernel="batched"` runs ONE `bmm` over `[E, cap, ·]`, which exists
        only when every expert's factors stack. Per-expert adaptive rank (or a
        hand-mutated expert) breaks that, and `torch.stack` would die with a
        shape error that names neither the arm nor an escape; this guard does.
        `share_u` experts own no `U` (the parent binds one), so the check reads
        what each expert actually carries.
        """

        def shapes(ex) -> tuple:
            gu = getattr(ex, "gate_u", None)
            uu = getattr(ex, "up_u", None)
            return (
                tuple(ex.gate_s.shape),
                tuple(ex.up_s.shape),
                tuple(ex.v.shape),
                None if gu is None else tuple(gu.shape),
                None if uu is None else tuple(uu.shape),
            )

        want = shapes(self.experts[0])
        for i, ex in enumerate(self.experts[1:], 1):
            got = shapes(ex)
            if got != want:
                loud(
                    f"moe_kernel='batched' needs every routed expert to have the "
                    f"same factor shapes for ONE [E, cap, ·] bank, got expert 0 "
                    f"{want} and expert {i} {got}. A per-expert rank/width makes "
                    f"the bank heterogeneous. Escape: one rank for every routed "
                    f"expert, or moe_kernel='naive' (the per-expert loop)."
                )

    def _batched_cap(self, index: torch.Tensor, weight: torch.Tensor) -> int:
        """The bank's common row cap: the capacity rule, or this batch's max group.

        `moe_capacity > 0` is the rule's own static cap — the same number the
        padded arm uses, so the capacity DROPS are the same ones (`keep`). With
        `moe_capacity = 0` every assignment is kept, so the tight bound is the
        LARGEST group in this batch; it is a device reduction plus ONE host
        scalar read, because a `[E, cap, ·]` allocation needs a Python int. It
        is not a per-expert read (the old loop's `unique().tolist()`), and the
        padded arm's route never reaches it.
        """
        # ROWS, not the shape's first dim: the dispatch is fed per-batch-element
        # in some paths, and `_capacity` is a function of the TOKEN count. Using
        # `index.shape[0]` there made capacity=2.0 keep 12 288 of 49 152
        # assignments — 74.5% dropped, measured 2026-11-04. The rule is exact
        # only on rows.
        k = index.shape[-1]
        n = index.numel() // k
        if self.capacity > 0:
            return self._capacity(n, k)
        flat_e = index.reshape(-1)
        groups = self.n_experts
        if self.topk_mode != "off":
            flat_e = flat_e.masked_fill(~(weight > 0).reshape(-1), self.n_experts)
            groups += 1
        counts = _bincount(flat_e, groups)
        top = int(counts[: self.n_experts].max().item())
        # A BOUND on the bank, because `max_group` is unbounded: a fresh router
        # whose experts are near-identical can put thousands of rows on one
        # expert, and `[E, max_group, d]` then asks for terabytes. The bound is
        # the uniform share times a generous factor; exceeding it is refused
        # LOUDLY rather than silently dropping rows (a silent drop here would
        # look exactly like a capacity arm with a bad rule).
        uniform = max(1.0, n * k / self.n_experts)
        bound = int(math.ceil(uniform * MOE_BANK_SAFETY))
        if top > bound:
            loud(
                f"moe_kernel='batched': the busiest expert holds {top} of {n * k} "
                f"assignments ({top / max(1, n * k):.1%}), and the bank's bound is "
                f"{bound} (= {MOE_BANK_SAFETY}x the uniform share {uniform:.1f}). "
                f"A [E, {top}, d] bank would not fit, and clamping would drop rows "
                f"silently. Escape: `moe_kernel=padded` (the correct but slower "
                f"arm), or `moe_capacity` above {top / uniform:.1f} (a capacity "
                f"arm that drops by a rule you can see)."
            )
        return max(1, top)

    def _bucket_plan(self, counts: torch.Tensor) -> list[tuple[list[int], int]]:
        """Partition experts into row-count classes: `[(members, cap), …]`.

        `counts` is the per-expert assignment count (`[E]`, zeros included). The
        plan is a pure function of those numbers: experts sort by count
        descending with the expert id as the tie-break, and a new bucket starts
        when the next count is more than 2x below the bucket's largest (the
        class rule: `n_b · cap <= 2 · rows`, so the padding is bounded by 2x) or
        when the bucket's rows would pass `MOE_BUCKET_ROWS`. A bucket's `cap` is
        its largest count, so the bank is `[len(members), cap, d]`; an expert
        past `MOE_BUCKET_ROWS` necessarily gets a bucket of its own.

        ONE host read: `counts.tolist()` (`E` ints), because an allocation shape
        is a Python int. It is the arm's documented sync — the plan cannot be
        built from device values alone, and pretending otherwise is how a
        `[E, top, d]` allocation becomes 12 GiB.
        """
        values = [int(c) for c in counts.tolist()]
        ranked = sorted((i for i, c in enumerate(values) if c > 0), key=lambda i: (-values[i], i))
        plan: list[tuple[list[int], int]] = []
        members: list[int] = []
        rows = 0
        cap = 0
        for e in ranked:
            c = values[e]
            # `2 · c < cap` is the class bound: the smallest count in the bucket
            # stays within 2x of the largest, so the padded rows are < the real
            # ones. The row bound keeps the bank's absolute size flat.
            if members and (2 * c < cap or rows + c > MOE_BUCKET_ROWS):
                plan.append((members, cap))
                members, rows, cap = [], 0, 0
            members.append(e)
            rows += c
            cap = max(cap, c)
        if members:
            plan.append((members, cap))
        return plan

    def _bucket_bytes(self, plan: list[tuple[list[int], int]], dt: torch.dtype) -> int:
        """The plan's bank bytes: `Σ n_b · cap_b · d · itemsize(dt)`."""
        rows = sum(len(members) * cap for members, cap in plan)
        return rows * self.d * dt.itemsize

    def _count_bank_rows(self, counters: Counters | None, dump: int, assignments: int) -> None:
        """COUNTED: the bank rows a padded pass RUNS, their waste, and the ratio.

        `dump = E·cap` (or `Σ n_b·cap_b` for the bucketed arm) is a Python int —
        a SHAPE, not a device read — so the padded arm's bill is arithmetic the
        log line can carry:

            rows_run    = dump
            ratio_x100  = max(1, round(100 · dump / assignments))
            waste_rows  = dump − assignments      # counted only when > 0

        With `moe_capacity=2.0` and a balanced router `dump = 2·assignments`, so
        `ratio_x100 = 200` and `waste_rows = assignments`; the dropless arm
        (`moe_capacity=0`, `cap` = the largest group) pays only `E·max − n·k`.
        A capacity BELOW the mean (`ratio_x100 < 100`) drops rows instead of
        padding: there is no waste to count (`Counters.count` refuses 0 and
        negatives by contract), the ratio says so, and `moe_dropped_assignments`
        owns the dropped count.
        This is the number `moe_capacity` is a trade on: see the recommended
        1.25-with-QB line in `config.ModelConfig.moe_capacity`.
        """
        if counters is None:
            return
        counters.count("moe_bank_rows", dump)
        if assignments <= 0:
            return
        # Clamped to >= 1: `Counters.count` refuses 0 (`instrument.py`), and a
        # ratio that would round to 0 is exactly the under-capacity case a reader
        # must still see — an absent counter reads as "no bank at all".
        counters.count("moe_bank_ratio_x100", max(1, round(100 * dump / assignments)))
        if dump > assignments:
            counters.count("moe_pad_waste_rows", dump - assignments)

    def _count_wide_weight(self, counters: Counters | None) -> None:
        """COUNTED: this route scales a `[R, f]` fp32 product with the weight itself.

        ONE caller: `_triton_dispatch`'s non-fused adjoint (`fused_bwd=False`),
        which runs `y.to(fp32) * weight[rows]` inline and never reaches
        `moe_kernels.combine`.

        Every OTHER wide product is counted where the branch is TAKEN —
        `_Combine.forward`'s `not fused` arm (`moe_kernels.py`) — because a
        `LaunchRefused` can flip `fused` to False after the caller has decided
        what to count (2026-11-10 review: the call site stayed silent on exactly
        that fallback).

        Why the Triton arms cannot ride the rank axis at all: they compute
        `h @ v` inside a kernel whose GLU output never leaves registers, and the
        weight's own gradient is `dw = <dout, h @ v>` — the UNSCALED expert
        output, i.e. the wide tensor this fix removed from the bank arm. The
        split GLU path (`E < moe_kernels.FUSED_GLU_MIN_E`) does materialise
        `[R, rank] h`, but inside `_ExpertGemmGlu`'s autograd Function, not on a
        graph the caller can scale without moving `dw` into the kernel too.
        """
        if counters is not None:
            counters.count("moe_wide_weight")

    def _bucketed_dispatch(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        order: torch.Tensor,
        keep: torch.Tensor | None,
        counters: Counters,
    ) -> torch.Tensor:
        """`_bank_dispatch`'s arithmetic over a plan of `[n_b, cap_b, ·]` banks.

        Every assignment is written ONCE into a flat scratch whose per-expert
        base is `bucket_offset + local_id · cap_b`; a row past capacity (or an
        adaptive padding slot) goes to the one dump row, exactly as the single
        bank does. Each bucket's slice is then its own `bmm` triple, and the
        per-bucket outputs are concatenated in the same order before the ONE
        `slots` gather — so the output is the single-bank arithmetic split by
        expert class, not a second definition of the expert.

        `moe_capacity > 0`: a bucket's cap is the smaller of its largest count
        and the capacity rule, so a dropped row's `pos` is past the bank and the
        existing `keep` mask sends it to the dump slot (its router weight is 0,
        so it contributes exactly 0 to the output and to every gradient).
        """
        self._assert_homogeneous()
        n, k = index.shape
        e = self.n_experts
        dt = self._bank_dtype()
        flat_w = weight.reshape(-1)
        flat_e = index.reshape(-1)
        if self.topk_mode == "off":
            counts = _bincount(flat_e, e)
            groups = e
        else:
            flat_e = flat_e.masked_fill(~(weight > 0).reshape(-1), e)
            counts = _bincount(flat_e, e + 1)
            groups = e + 1
        plan = self._bucket_plan(counts[:e])
        if self.capacity > 0:
            # An expert past the rule's cap is dropped by `keep`; sizing its
            # bank to the cap (not to the raw count) is what makes the capacity
            # arm's memory the rule's, not the router's.
            rule = self._capacity(n, k)
            plan = [(members, min(cap, rule)) for members, cap in plan]
        bank_bytes = self._bucket_bytes(plan, dt)
        if bank_bytes > MOE_BUCKET_BYTES:
            itemsize = dt.itemsize
            members, cap = max(plan, key=lambda p: len(p[0]) * p[1])
            largest = len(members) * cap * self.d * itemsize / 2**20
            loud(
                f"moe_kernel='bucketed': the plan's banks need {bank_bytes / 2**20:.1f} "
                f"MiB, over the {MOE_BUCKET_BYTES / 2**20:.0f} MiB budget. Largest "
                f"bucket: {len(members)} experts x {cap} rows x d={self.d} in "
                f"{dt} = {largest:.1f} MiB; {len(plan)} buckets, busiest expert "
                f"{int(counts[:e].max())} of {n * k} assignments. Escape: "
                f"moe_kernel='naive' (the per-expert loop builds no bank), "
                f"moe_capacity > 0 (drops by a rule you can see), or a router that "
                f"spreads the rows."
            )
        counters.count("moe_buckets", len(plan))
        counters.count("moe_bank_bytes", bank_bytes)
        if not plan:
            return torch.zeros(n, self.f, dtype=torch.float32, device=x.device)
        owner = order // k
        starts = torch.cumsum(counts, 0) - counts
        eid = flat_e[order]
        pos = torch.arange(n * k, device=index.device) - starts[eid]
        dump = sum(len(members) * cap for members, cap in plan)
        self._count_bank_rows(counters, dump, n * k)
        # The per-expert base in the flat scratch: bucket offset + local index *
        # cap. `groups` covers the adaptive-width sentinel (never kept, its
        # `base` entry is 0 and unused — the same computed-but-masked slot the
        # single bank gives it).
        base = torch.zeros(groups, dtype=torch.long, device=index.device)
        offset = 0
        local = []
        for members, cap in plan:
            idx = torch.tensor(members, dtype=torch.long, device=index.device)
            base.index_copy_(0, idx, offset + torch.arange(len(members), device=index.device) * cap)
            local.append((idx, cap))
            offset += len(members) * cap
        static_keep = keep is None
        if static_keep:
            keep = torch.ones_like(owner, dtype=torch.bool)
        slots = torch.where(keep, base[eid] + pos, dump)
        # THE ROUTER WEIGHT IN THE BANK LAYOUT, before each bucket's `@ v` — the
        # same fix `_bank_dispatch` carries, so `bucketed` no longer materializes
        # a `[R, f]` fp32 product just to scale rows. `w_flat` is `[dump + 1]`
        # with the constant 0 at the dump slot; a bucket reads its own slice.
        w = flat_w[order]
        if not static_keep:
            w = w * keep.to(w.dtype)
        w_flat = torch.zeros(dump + 1, dtype=w.dtype, device=x.device)
        w_flat.index_put_((slots,), w)
        xg = torch.zeros(dump + 1, self.d, dtype=dt, device=x.device)
        xg = xg.index_put((slots,), self._gather_rows(x, owner, dt))
        flat = xg.narrow(0, 0, dump)
        gate_u, up_u, v, gate_s, up_s = self._banks(dt, counters)
        ys = []
        offset = 0
        for idx, cap in local:
            bank = flat.narrow(0, offset, idx.numel() * cap).view(idx.numel(), cap, self.d)
            g = (
                torch.bmm(bank, gate_u.index_select(0, idx))
                * gate_s.index_select(0, idx)[:, None, :]
            )
            u = torch.bmm(bank, up_u.index_select(0, idx)) * up_s.index_select(0, idx)[:, None, :]
            h = F.silu(g) * u if self._glu is None else self._glu(g, u)
            wb = w_flat.narrow(0, offset, idx.numel() * cap).view(idx.numel(), cap, 1).to(dt)
            ys.append(torch.bmm(h * wb, v.index_select(0, idx)).reshape(idx.numel() * cap, self.f))
            offset += idx.numel() * cap
        y = ys[0] if len(ys) == 1 else torch.cat(ys)
        # Same sentinel as `_bank_dispatch`: a dropped row reads a clamped bank
        # row and lands on token row `n`, which the return slices off.
        rows = slots if static_keep else slots.clamp(max=dump - 1)
        y = y[rows]
        out = torch.zeros(n + 1, self.f, dtype=torch.float32, device=x.device)
        dest = owner if static_keep else torch.where(keep, owner, n)
        out.index_add_(0, dest, y.to(torch.float32))
        counters.count("moe_w_in_bank")
        return out[:n]

    def _batched_dispatch(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        order: torch.Tensor,
        keep: torch.Tensor | None,
        counters: Counters | None = None,
    ) -> torch.Tensor:
        """Gather → ONE `[E, cap, ·]` bmm bank → weighted scatter, dropless too.

        The same arithmetic as `_padded_dispatch`, with the cap chosen for THIS
        batch when `moe_capacity = 0` (`_batched_cap`): every assignment gets a
        row, so this arm is the per-expert loop's function with three batched
        GEMMs instead of E tiny ones. `moe_capacity > 0` uses the capacity
        rule's cap and drops exactly the rows the other capacity arms drop.

        `moe_shared_compute` swaps the bank's `[E, cap, d]` operand for the
        gathered `[E, cap, 2·rank]` latent and the two `d -> rank` bmms for the
        one projection over the tokens — same weights, same output (the parity
        gate's 1e-5, not bit-identity, because the reduction order moved).
        """
        self._assert_homogeneous()
        return self._bank_dispatch(
            x, weight, index, order, keep, self._batched_cap(index, weight), counters
        )

    def _padded_dispatch(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        order: torch.Tensor,
        keep: torch.Tensor | None,
        counters: Counters | None = None,
    ) -> torch.Tensor:
        """Gather → ONE capacity-padded `[E, cap, ·]` bank pass → weighted scatter.

        The spike's `moe_grouped`, adapted to the dual-latent GLU expert: the
        same math the naive path runs per expert, batched over the bank. Rows
        past capacity never enter the bank, and the ones that do are scattered
        back by the same `keep` mask, so this arm changes how the arithmetic is
        laid out, never which tokens get an expert.

        `moe_capacity = 0` (dropless) is REFUSED here: this arm IS a
        `[E, cap, ·]` bank and dropless has no cap to size it. The dropless
        bank is `moe_kernel="batched"` (it sizes the cap to this batch's largest
        group); this guard is the seam for a direct call.
        """
        if self.capacity == 0:
            loud(
                "the padded MoE arm with moe_capacity=0: this arm materializes a "
                "[E, cap, ·] bank and dropless has no cap (the rule's floor would be "
                "k, which drops every row past it — a silently wrong forward). "
                "Escape: moe_capacity > 0 for the padded kernel, moe_kernel='batched' "
                "for the dropless bank, or moe_kernel=naive (the ragged arms keep "
                "every assignment)."
            )
        n, k = index.shape
        return self._bank_dispatch(x, weight, index, order, keep, self._capacity(n, k), counters)

    def _bank_dispatch(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        order: torch.Tensor,
        keep: torch.Tensor | None,
        cap: int,
        counters: Counters | None = None,
    ) -> torch.Tensor:
        """The `[E, cap, ·]` bank body shared by `padded` and `batched`.

        `cap` is the ONE difference between the two arms: the capacity rule's
        number (`padded`) or the batch's largest group (`batched`, dropless).
        Everything else — the dump slot, the zero fill, the fp32 router weight —
        is one implementation so the arms cannot drift.
        """
        n, k = index.shape
        e = self.n_experts
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
        # static bound, and a dropped row is routed to a slot no kept row can
        # own (`e·cap` in the bank lookup, token row `n` in the scatter) with
        # the constant-0 router weight, so it contributes exactly 0.0 to the
        # output and exactly 0 to both gradients — the eager semantics, not an
        # approximation of them.
        owner = order // k  # the token each assignment belongs to
        # The padded-bank slot of each sorted row: expert-major, capacity-padded.
        starts = torch.cumsum(counts, 0) - counts
        eid = flat_e[order]
        pos = torch.arange(n * k, device=index.device) - starts[eid]
        dump = e * cap
        self._count_bank_rows(counters, dump, n * k)
        # `_permute` always returns a mask; a direct caller may omit it, and
        # `None` then means "every assignment is kept" (the pre-cap behaviour),
        # which is the only reading that keeps the two arms' row sets equal. It
        # also selects the STATIC scatter (`owner` directly, no `where`).
        static_keep = keep is None
        if static_keep:
            keep = torch.ones_like(owner, dtype=torch.bool)
        slots = torch.where(keep, eid * cap + pos, dump)
        # Zero-filled, not `empty`: a row past capacity is never read, but a
        # zero row keeps the bank finite, and inf-garbage x 0 is NaN in the
        # backward (the spike measured exactly that — `probe_moe_variants.py`).
        # The bank the GEMM sees is EXACTLY the eager `[e, cap, ·]` — the dump
        # row is sliced off, so the expert pass keeps its shape, its kernel and
        # its bits. Only the lookup needs the extra row, and it gets a constant
        # zero row (the dumped assignments' output is 0 anyway).
        gate_u, up_u, v, gate_s, up_s = self._banks(dt, counters)
        if self.moe_shared_compute:
            self._assert_shared_compute()
            rank = self.experts[0].rank
            # `[n, 2·rank]`: ONE `d -> rank` projection per TOKEN (the shared
            # basis is the same for both of a token's assignments), then the
            # same expert-major gather into the bank, on `2·rank` columns
            # instead of `d`. The rank-space GLU and the write are unchanged.
            lat = self._shared_projection(x, dt, counters)
            xg = torch.zeros(dump + 1, 2 * rank, dtype=dt, device=x.device)
            xg = xg.index_put((slots,), self._gather_rows(lat, owner, dt))
            bank = xg.narrow(0, 0, dump).view(e, cap, 2 * rank)
            g = bank[..., :rank] * gate_s[:, None, :]
            u = bank[..., rank:] * up_s[:, None, :]
        else:
            xg = torch.zeros(dump + 1, self.d, dtype=dt, device=x.device)
            xg = xg.index_put((slots,), self._gather_rows(x, owner, dt))
            bank = xg.narrow(0, 0, dump).view(e, cap, self.d)
            g = torch.bmm(bank, gate_u) * gate_s[:, None, :]
            u = torch.bmm(bank, up_u) * up_s[:, None, :]
        h = F.silu(g) * u if self._glu is None else self._glu(g, u)
        # THE ROUTER WEIGHT GOES INTO RANK SPACE, before the wide `@ v`. The old
        # tail scaled the `[n·k, f]` OUTPUT in fp32, which materializes a
        # `4·n·k·f` byte buffer per layer just to scale ROWS (at b24, rank 832,
        # f 1664: 327 MB), on top of its own cast. Here `w_bank` is `[dump + 1]`
        # with the constant 0 at the dump slot, so a dropped assignment's bank
        # row is exactly 0 and its `@ v` is exactly 0, and the wide write is a
        # `[E, cap, rank]` multiply in the GEMM's own dtype — half the bytes the
        # fp32 `[n·k, f]` product took at this sizing (f = 2·rank), with no
        # second pass over it and no fp32 cast of its input.
        #
        # NUMERICS: the multiply now happens in the GEMM's dtype instead of in
        # fp32 after the GEMM, so the bank is no longer bit-identical to the
        # naive loop in a bf16 run. That is the price of the layout, and it is
        # MEASURED, not assumed: `tests/test_moe_batched.py` holds the two
        # bounds (1e-5 in fp32, 1e-2 relative in bf16) with the delta printed.
        # A fp32-exact match after the GEMM is exactly the buffer this removes.
        w = flat_w[order]
        if not static_keep:
            # The exact-0 weight a dropped row needs (`keep` is a bool CONSTANT
            # of the graph, so `dL/dw` there is exactly 0).
            w = w * keep.to(w.dtype)
        w_bank = torch.zeros(dump + 1, dtype=w.dtype, device=x.device)
        w_bank.index_put_((slots,), w)
        h = h * w_bank.narrow(0, 0, dump).view(e, cap, 1).to(dt)
        y = torch.bmm(h, v).reshape(dump, self.f)
        # The dumped rows read a REAL bank row (clamped) and are scattered to
        # token row `n`, which the return slices off: their value is discarded
        # and their gradient is exactly 0 (the slice's backward zeroes that
        # row), so the `[dump + 1, f]` zero-row `cat` the old tail needed to make
        # the lookup index-legal is gone too.
        rows = slots if static_keep else slots.clamp(max=dump - 1)
        y = y[rows]  # [n·k, f], in the GEMM's dtype
        # The ONE fp32 buffer the exact-sum contract needs: the accumulation
        # target (row `n` is the sentinel and is sliced off below).
        out = torch.zeros(n + 1, self.f, dtype=torch.float32, device=x.device)
        dest = owner if static_keep else torch.where(keep, owner, n)
        out.index_add_(0, dest, y.to(torch.float32))
        if counters is not None:
            counters.count("moe_w_in_bank")
        return out[:n]

    @torch.compiler.disable
    def _naive_dispatch(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        order: torch.Tensor,
        keep: torch.Tensor,
        counters: Counters,
    ) -> torch.Tensor:
        """Gather → per-expert matmul → scatter. The reference and the CPU path.

        COUNTED COMPILE BREAK (`scripts/bench_compile.py`'s explain pass). The
        body is a per-expert Python loop over `torch.unique(ids_in).tolist()` —
        a device read whose output shape is data, so no graph can hold it. Left
        to dynamo it splits the graph FOUR times per dispatch (`Dynamic shape
        operator` at the `unique` below); the decorator makes that ONE break
        named `skip inlining torch.compiler.disable()d function`, which is the
        same program with a smaller, stable guard set (the loop's Python values
        are no longer re-specialised as the routed expert set changes). Eager is
        a no-op: `torch.compiler.disable` only affects tracing.
        """
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
        (`docs/records/research/2026-10-16-budget-savers.md` §5: predicted +5-15% step).
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

        `moe_capacity = 0` (dropless) short-circuits to `_dropless_dispatch`: the
        permute keeps every assignment and the padded arm has no fixed cap to
        size its bank with (`_padded_dispatch` refuses), so either the batched
        arm sizes one to the batch's largest group (`moe_kernel="batched"`) or
        the grouped GEMM runs with the static row-block bound and the kernels'
        on-device empty-block exit — no host read-back of `offs` (ADR-0018 rule
        2). `_dropless_dispatch` owns that route and its counters (`moe_dropless`,
        `moe_batched_calls`, `moe_triton_gemm`, or `moe_naive_dispatch` + the
        reason when the kernel cannot serve the call).

        A zero router weight is an adaptive-width PADDING SLOT, not a dropped
        assignment: it must not consume capacity (a padding row that keeps a real
        row out would make the adaptive arm drop more than its own k) and it must
        not be gathered into an expert bank. The constant-width arm passes `None`
        — the pre-policy path, bit for bit.
        """
        valid = None if self.topk_mode == "off" else weight > 0
        order, keep, dropped = self._permute(index, valid, counters)
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
                return self._gather_dispatch(x, weight, index, order, keep_rows, counters)
            counters.count("moe_gather_fallback")
            counters.why = f"gather: {gwhy}"  # type: ignore[attr-defined]
        if self.capacity == 0:
            return self._dropless_dispatch(x, weight, index, order, keep, keep_rows, counters)
        notes = [] if gwhy is None else [f"gather: {gwhy}"]
        if self.moe_kernel == "bucketed":
            if notes:
                counters.why = "; ".join(notes)  # type: ignore[attr-defined]
            return self._expert_pass(
                lambda: self._bucketed_dispatch(x, weight, index, order, keep_rows, counters),
                counters,
            )
        if self.moe_kernel == "batched":
            if notes:
                counters.why = "; ".join(notes)  # type: ignore[attr-defined]
            counters.count("moe_batched_calls")
            return self._expert_pass(
                lambda: self._batched_dispatch(x, weight, index, order, keep_rows, counters),
                counters,
            )
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
                self._count_tile(
                    counters,
                    2 * self.experts[0].rank if self.moe_shared_compute else None,
                )
                return self._expert_pass(
                    lambda: self._triton_dispatch(
                        x,
                        weight,
                        index,
                        order,
                        keep_rows,
                        fused_bwd=fwhy is None,
                        counters=counters,
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
                lambda: self._padded_dispatch(x, weight, index, order, keep_rows, counters),
                counters,
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

        The padded arm is out by construction (a `[E, cap, ·]` bank has no fixed
        cap), so the choice is the batched bank (its cap is this batch's largest
        group), the grouped GEMM, or the reference loop — and the grouped GEMM
        now SERVES dropless: its row-block grid takes a static bound
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
        if self.moe_kernel == "bucketed":
            return self._expert_pass(
                lambda: self._bucketed_dispatch(x, weight, index, order, keep_rows, counters),
                counters,
            )
        if self.moe_kernel == "batched":
            counters.count("moe_batched_calls")
            return self._expert_pass(
                lambda: self._batched_dispatch(x, weight, index, order, keep_rows, counters),
                counters,
            )
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
            self._count_tile(
                counters, 2 * self.experts[0].rank if self.moe_shared_compute else None
            )
            return self._expert_pass(
                lambda: self._triton_dispatch(
                    x,
                    weight,
                    index,
                    order,
                    keep_rows,
                    dropless=True,
                    fused_bwd=fwhy is None,
                    counters=counters,
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

    def _count_tile(self, counters: Counters, d: int | None = None) -> None:
        """COUNTED: the tile the search picked, as a counter NAME (`moe_tile_N…_M…_st…`).

        ADR-0019: the tile is a degradation of the ideal (the shape may have cost
        a pipeline stage or half of `BLOCK_N`), so a reader of the log must see
        which one ran without a profiler — `moe_tile_N128_M64_st1=<forwards>` on
        the log line says the arm gave up a stage, and `st2` says it kept it.
        Same seam as `moe_triton_gemm` (cardinality is one per tile set, so the
        counter line stays short). `d` is the kernel's A width: `2·rank` under
        `moe_shared_compute`, because the projection already happened.
        """
        counters.count(
            moe_kernels.tile_name(
                self._bank_dtype(), self.d if d is None else d, self.f, self.experts[0].rank
            )
        )

    def _triton_reason(self, x: torch.Tensor) -> str | None:
        """`None` if the unpadded grouped GEMM can run THIS call, else why not.

        The arm is unpadded by construction (`offs` carries the ragged bounds),
        so the gates are the ability itself (no triton, no CUDA) and the TILE:
        `moe_kernels.tile_reason` runs the ordered tile SEARCH (`TILE_SEARCH`)
        and estimates the CTA shared memory of the widest kernel for each
        candidate from the compute dtype. A shape no candidate fits is COUNTED
        with its numbers instead of dying at launch — the loud message names the
        dtype, the smallest candidate's bytes, the ceiling and the escape — and
        the search means a merely WIDE shape (rank 192 bf16) now runs at the
        largest tile that fits instead of killing the run at step 0.
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

    def _route_fused_reason(self, x: torch.Tensor) -> str | None:
        """`None` if the fused route head can run THIS call, else why not.

        COUNTED (`moe_route_fused_fallback` + the reason), the same contract as
        `_gather_reason`: a config asks for the arm and an arm that cannot serve
        a call must not vanish. Two of the reasons are config-level and are also
        refused LOUDLY by `config.validate` (the adaptive width moves unused
        slots to a sentinel id and zeroes them, which the fused kernel's
        constant-width selection does not express; `moe_lb_mode="aux"` reads the
        softmax, which the fused arm does not materialize). What is left is
        ability: triton importable, input on CUDA.
        """
        if self.topk <= 0:
            return "moe_topk=0 is the dense blend: there is no top-k to fuse"
        if self.topk_mode != "off":
            return (
                f"moe_topk_mode={self.topk_mode!r}: the adaptive width masks the "
                f"raised logits and keeps `probs` for its zeroed slots"
            )
        if self.lb_mode == "aux":
            return (
                "moe_lb_mode='aux' reads the softmax `probs`, which the fused route "
                "does not materialize"
            )
        return route_fused.reason(x)

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
        counters: Counters | None = None,
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

        `moe_shared_compute` feeds the SAME kernel the latent rows: the tokens
        are projected once through `[gate_u | up_u] [d, 2·rank]`, the gathered
        `[R, 2·rank]` rows become `xg`, and the kernel's two projections become
        the constant selector matrices `[I; 0]` / `[0; I]` (`_shared_select`).
        The kernel, its fused backward and the combine are unchanged — the
        `d -> rank` product they used to do was already applied once, over the
        tokens, by `_shared_projection`.
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
        gate_u, up_u, v, gate_s, up_s = self._banks(dt, counters)
        cap = None if dropless else self._capacity(n, k)
        if self.moe_shared_compute:
            self._assert_shared_compute()
            lat = self._shared_projection(x, dt, counters)
            xg = self._gather_rows(lat, owner, dt)
            gate_u, up_u = self._shared_select(dt, x.device)
        else:
            xg = self._gather_rows(x, owner, dt)
        if not fused_bwd:
            y = moe_kernels.expert_gemm_glu(
                xg, gate_u, up_u, gate_s, up_s, v, offs, cap=cap, fused_bwd=False
            )
            # fp32 weight applied where the padded arm applies it (see the note
            # there): a bf16 multiply would be a 2^-8 step the parity gate could
            # not tell from a wiring error.
            # NOTE (2026-11-10): the padded/batched/bucketed arms apply the
            # router weight in RANK space before `@ v`; this kernel cannot, so
            # the fp32 product stands here and is COUNTED (`moe_wide_weight` —
            # `_count_wide_weight` carries the why).
            self._count_wide_weight(counters)
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
        #
        # The wide-product counter is NOT decided here: `combine` counts it in
        # the branch that actually materialises it (`_Combine.forward`), which a
        # refused fused launch can select after this call is made.
        return moe_kernels.combine(
            y,
            weight.reshape(-1)[rows].float(),
            owner,
            rowmap.view(n, k),
            n,
            ws=ws,
            counters=counters,
            fused=self.moe_combine == "fused",
            dtype=self.moe_combine_dtype,
        )

    def _gather_dispatch(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        order: torch.Tensor,
        keep: torch.Tensor | None,
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
        weights are applied as fp32 (COUNTED `moe_wide_weight` unless
        `moe_combine="fused"`), the `rowmap` is the combine backward's own int32
        map, and the workspace is the same grow-only `BwdWorkspace`. A different
        scatter here would make the parity gate compare two changes at once.

        The FORWARD is the only fused half (`moe_sonic._GatherGlu` recomputes the
        grouped pass for its backward), so the fwd→bwd cache is `x`, `toks` and
        `y` — no `xg`, no GLU gates. That is priced by
        `scripts/bench_gather.py`, not here.
        """
        assert moe_kernels is not None, "the gather arm needs the shipped grouped GEMM"
        n, k = index.shape
        dt = self._bank_dtype()
        # `keep=None` is every permuted row (dropless / one token): `_dispatch`
        # hands `keep_rows`, so the ABI's `order[keep]` — a BOOLEAN mask, i.e. a
        # `nonzero` and a host-side length — is not built on the route that does
        # not need it. Same rule `_triton_dispatch` states one function up.
        rows = order if keep is None else order[keep]
        owner = rows // k
        toks, offs = moe_sonic.gather_abi(index, order, keep, self.n_experts)
        gate_u, up_u, v, gate_s, up_s = self._banks(dt, counters)
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
        # The wide-product counter is counted inside `_Combine.forward` (the
        # branch that materialises `y.to(fp32) * w`), not here: a refused fused
        # launch selects that branch after this call is made.
        return moe_kernels.combine(
            y,
            weight.reshape(-1)[rows].float(),
            owner,
            rowmap.view(n, k),
            n,
            ws=ws,
            counters=counters,
            fused=self.moe_combine == "fused",
            dtype=self.moe_combine_dtype,
        )


def revive_experts_all(model: nn.Module, step: int, seed: int) -> int:
    """Call every live TSCT-2 revive; returns how many experts were revived.

    The trainer's seam (`train/loop.py`), once per OPTIMIZER step, after the
    optimizer and the bank invalidation. A model with the arm off makes no call
    and draws nothing, so the OFF path is untouched.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, SparseMoE):
            n += m.revive_experts(step, seed)
    return n


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
