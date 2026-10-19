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
  when the kernel cannot run (no triton, not CUDA) the padded arm runs instead
  and `moe_triton_fallback` + the reason say so.
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

All three count: `moe_padded_bmm`, `moe_triton_gemm` or `moe_naive_dispatch`
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

from ..instrument import Counters, loud
from .act_quant import fake_quant_act
from .tsct import dense_qat_project, factor_qat_project, mm_mixed

try:  # the Triton arm is an ABILITY, not a dependency
    from . import moe_kernels
except Exception as exc:  # noqa: BLE001 — the reason is COUNTED by `_dispatch`
    moe_kernels = None
    _TRITON_WHY: str | None = f"the triton kernels are not importable here: {exc}"
else:
    _TRITON_WHY = None


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
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.d, self.f, self.rank = d, f, rank
        self.compute_dtype = dtype
        self.factor_qat = factor_qat
        self.factor_qat_alpha = factor_qat_alpha
        #: The a4.8 ACTIVATION switch (`act_quant.fake_quant_act`): the `x` fed
        #: to this expert's factor matmuls is fake-quantized with an STE. Same
        #: switch the banked path in `SparseMoE` applies to its `x`, so one
        #: expert through the bank and the same expert alone quantize their
        #: activation identically (per-row/group scale, expert-independent).
        self.act_quant = act_quant
        self.act_group = act_group
        for name in ("gate", "up"):
            # `U [d, r]` and `s [r]`: the latent half of TSCT's factorization.
            setattr(
                self, f"{name}_u", nn.Parameter(torch.randn(d, rank, device=device) / math.sqrt(d))
            )
            setattr(self, f"{name}_s", nn.Parameter(torch.ones(rank, device=device)))
        # The write `V [r, f]` — one, at the latent, after the GLU.
        self.v = nn.Parameter(torch.randn(rank, f, device=device) / math.sqrt(rank))

    def forward(self, x: torch.Tensor, glu=None) -> torch.Tensor:
        dt = self.compute_dtype
        if self.act_quant != "none":
            # a4.8: the activation feeding the factor matmuls, quantized in the
            # dtype it arrives in (fp32 masters, cast to `dt` after) with an STE.
            x = fake_quant_act(x, self.act_quant, self.act_group)
        xu = x if dt == torch.float32 else x.to(dt)
        gate_u, up_u, v = self.gate_u, self.up_u, self.v
        if self.factor_qat != "none":
            # Per-tensor α on a 2-D factor == per-expert α on that expert's
            # slice of the bank (`_banks`'s `dim=(1,2)`), so the naive/shared
            # path and the banked path quantize an expert IDENTICALLY.
            q = factor_qat_project
            gate_u = q(gate_u, self.factor_qat, self.factor_qat_alpha)
            up_u = q(up_u, self.factor_qat, self.factor_qat_alpha)
            v = q(v, self.factor_qat, self.factor_qat_alpha)
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
        self.capacity = cfg.moe_capacity
        self.moe_kernel = cfg.moe_kernel
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
        # The router: ONE 2-D linear on the raw hidden state (not on the
        # latent), softmax over the bank, then top-k + a capacity-bound permute.
        self.router = nn.Linear(d, cfg.n_experts, bias=False, device=device)
        # The LFB routing bias (arXiv:2408.15664): a per-expert `b_e` added to
        # the softmax probs before the top-k, moved after each routed forward
        # by `-coef·sign(load_e − mean)` WITHOUT gradients. A buffer, never a
        # parameter: the optimizer must not move it, LFB does. It is MODEL
        # state — the next resume continues from it — but NOT a persistent
        # state-dict buffer: the export artifact's key set is pinned by the
        # demo engine and `tests/test_export.py`, which have no LFB. The
        # checkpoint carries it in the `<ckpt_name>.lfb` sibling and
        # `train.loop.load_model_state` restores it, so a resume does not
        # restart a controller the counts say it already integrated (audit
        # 2026-10-19 M8).
        self.lfb_coef = cfg.moe_lb_coef
        self.register_buffer(
            "lfb_bias", torch.zeros(cfg.n_experts, device=device), persistent=False
        )
        # The cumulative routed-count integrator the bias reads (persistent:
        # a resume must keep its controller state or the rebalancer restarts
        # its memory of where the load went).
        self.register_buffer(
            "lfb_counts", torch.zeros(cfg.n_experts, device=device), persistent=True
        )
        # The shared write back to the residual width.
        self.down = nn.Linear(f, d, bias=False, device=device)

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        counters = Counters() if counters is None else counters
        n = x.shape[0]
        assert x.ndim == 2 and x.shape[1] == self.d, f"MoE input {tuple(x.shape)} != [n, {self.d}]"
        dt = self.experts[0].compute_dtype  # the shared compute dtype
        # `mm_mixed` returns `dt`, NOT fp32: with bf16 on, `logits` — and so the
        # softmax `probs` and the routed `weight` — are bf16. The combine
        # boundary widens the weight back to fp32 (`_triton_dispatch`).
        # `dense_qat_project` is the identity when the arm is off.
        logits = mm_mixed(
            x,
            dense_qat_project(self.router.weight, self.dense_qat, self.dense_qat_alpha),
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
            index = raised.topk(self.topk, dim=-1).indices
            weight = torch.gather(probs, 1, index)
            self._lfb_step(index, counters)
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
            dense_qat_project(self.down.weight, self.dense_qat, self.dense_qat_alpha),
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
            counts = torch.bincount(index.reshape(-1), minlength=self.n_experts).float()
            self.lfb_counts.add_(counts)
            self.lfb_bias.sub_(self.lfb_coef * torch.sign(self.lfb_counts - self.lfb_counts.mean()))
        counters.count("moe_lfb_updates")

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

    def _permute(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The routing permute: `(order, keep, dropped)` — one place, both arms.

        Both arms are permuted by the SAME rows and drop the SAME ones, which is
        what makes a parity run comparable and the drop count one number rather
        than two opinions. `dropped` STAYS A DEVICE TENSOR (a 0-dim int64
        counter the caller's `Counters.count_device` accumulates): the early
        form returned `int((~keep).sum())`, a per-forward GPU→CPU sync — one
        of exactly two places that blocked torch.compile(dynamic) in the
        arm-level probe (the other is the drain cadence itself, instrument.py).

        `moe_capacity = 0` (dropless) keeps EVERY row: `keep` is all-True and
        `dropped` is 0, so the ragged arms see the full assignment set. That is
        what makes the drop set independent of the window's composition, and it
        is the whole reason this mode exists (module docstring).
        """
        n, k = index.shape
        flat_ids = index.reshape(-1)
        order = torch.argsort(flat_ids, stable=True)
        sorted_ids = flat_ids[order]
        counts = torch.bincount(sorted_ids, minlength=self.n_experts)
        offsets = torch.cumsum(counts, 0) - counts
        pos = torch.arange(n * k, device=index.device) - offsets[sorted_ids]
        if self.capacity == 0:
            keep = torch.ones_like(pos, dtype=torch.bool)
        else:
            keep = pos < self._capacity(n, k)
        dropped = (~keep).sum().view(())
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
        """
        grad = torch.is_grad_enabled()
        # `want` is what the cached bank must carry a graph FOR: a frozen expert
        # bank cannot, and keying on `is_grad_enabled()` alone would rebuild the
        # same untracked stack on every forward of a frozen model.
        want = grad and self.experts[0].gate_u.requires_grad
        key = (
            dt,
            self.factor_qat,
            self.factor_qat_alpha,  # the anneal ramp moves this every step
            tuple(p._version for e in self.experts for p in e.parameters()),
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
            # warm-up. Per-expert, the fp32 footprint is one expert (5.6 MB),
            # and the qat arm — if on — quantizes the same way the stacked
            # form did: `dim=(1, 2)` over a `[1, r, d]` view is the identical
            # per-expert α, so a banked expert stays identical to the same
            # expert run through `DualLatentGLU.forward` (the parity the A/B
            # needs). Scales (`gate_s`/`up_s`) are never quantized: they ARE
            # the diag(s) of the factorization.
            def _one(name: str) -> torch.Tensor:
                parts = []
                for e in self.experts:
                    t = getattr(e, name)
                    if self.factor_qat != "none":
                        t = factor_qat_project(
                            t.unsqueeze(0), self.factor_qat, self.factor_qat_alpha, dim=(1, 2)
                        )[0]
                    parts.append(t.to(dt))
                return torch.stack(parts)

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
        keep: torch.Tensor,
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
        rows = order[keep]  # flat (token, slot) pairs that survive the cap
        owner = rows // k  # the token each kept row belongs to
        # The padded-bank slot of each sorted row: expert-major, capacity-padded.
        counts = torch.bincount(flat_e, minlength=e)
        starts = torch.cumsum(counts, 0) - counts
        pos = torch.arange(n * k, device=index.device) - starts[flat_e[order]]
        kept_slots = (flat_e[order] * cap + pos)[keep]
        # Zero-filled, not `empty`: a row past capacity is never read, but a
        # zero row keeps the bank finite, and inf-garbage × 0 is NaN in the
        # backward (the spike measured exactly that — `probe_moe_variants.py`).
        xg = torch.zeros(e * cap, self.d, dtype=dt, device=x.device)
        xg = xg.index_put((kept_slots,), x[owner].to(dt)).view(e, cap, self.d)

        gate_u, up_u, v, gate_s, up_s = self._banks(dt)
        g = torch.bmm(xg, gate_u) * gate_s[:, None, :]
        u = torch.bmm(xg, up_u) * up_s[:, None, :]
        h = F.silu(g) * u if self._glu is None else self._glu(g, u)
        y = torch.bmm(h, v).reshape(e * cap, self.f)[kept_slots]  # [R, f]
        # The router weight is applied in fp32, exactly where the naive path
        # applies it: a bf16 multiply would put a 2^-8 relative step between the
        # two arms that the parity gate could not tell from a wiring error.
        # ponytail: it may move to rank-space (w ⊙ h before `@ v`, the spike's
        # −8 % on the expert pass) once a b1-shape profile says the [R, f]
        # multiply matters.
        y = y.to(torch.float32) * flat_w[rows].unsqueeze(-1)
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
        for e in torch.unique(sorted_ids[keep]).tolist():
            sel = keep & (sorted_ids == e)
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
        """
        order, keep, dropped = self._permute(index)
        # Count EVERY call: a python `if dropped:` on the device tensor is
        # exactly the sync this counter exists to prevent (caught by the
        # drop-parity test when the tensor stopped being an int). A zero add
        # costs nothing once the drain batches it.
        counters.count("moe_dropped_assignments", dropped)
        if self.capacity == 0:
            return self._dropless_dispatch(x, weight, index, order, keep, counters)
        notes = []
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
                        x, weight, index, order, keep, fused_bwd=fwhy is None
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
                lambda: self._padded_dispatch(x, weight, index, order, keep), counters
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
                counters.why = (  # type: ignore[attr-defined]
                    "capacity=0: dropless — the padded bank has no cap to size itself "
                    "with, so the grouped GEMM runs (the static row-block bound + the "
                    "kernels' on-device empty-block exit)"
                )
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
                    x, weight, index, order, keep, dropless=True, fused_bwd=fwhy is None
                ),
                counters,
            )
        if self.moe_kernel == "triton":
            counters.count("moe_triton_fallback")
            why = f"triton: {why}"
        counters.why = why  # type: ignore[attr-defined]
        counters.count("moe_naive_dispatch")
        return self._expert_pass(
            lambda: self._naive_dispatch(x, weight, index, order, keep, counters), counters
        )

    def _triton_reason(self, x: torch.Tensor) -> str | None:
        """`None` if the unpadded grouped GEMM can run THIS call, else why not.

        The arm is unpadded by construction (`offs` carries the ragged bounds),
        so the only things that can stop it are the ability itself (no triton,
        no CUDA) — both COUNTED, never silent. The shapes need no gate: every
        dimension is masked in-kernel, and the op asserts what it cannot do.
        """
        if _TRITON_WHY is not None:
            return _TRITON_WHY
        if not x.is_cuda:
            return f"input is on {x.device.type}: the grouped GEMM is the CUDA path"
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
        keep: torch.Tensor,
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
        rows = order[keep]
        owner = rows // k
        kept = index.reshape(-1)[rows]
        counts = torch.bincount(kept, minlength=self.n_experts).to(torch.int32)
        offs = torch.zeros(self.n_experts + 1, dtype=torch.int32, device=x.device)
        offs[1:] = torch.cumsum(counts, 0)
        gate_u, up_u, v, gate_s, up_s = self._banks(dt)
        cap = None if dropless else self._capacity(n, k)
        xg = x[owner].to(dt)
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
