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
* **naive (the CPU path and the correctness reference)**: gather → per-expert
  matmul → scatter over a capacity-bound permute. The day-one implementation,
  kept because it is what the padded arm is compared against — the same rows,
  the same capacity rule, the same drop counter.

All three count: `moe_padded_bmm`, `moe_triton_gemm` or `moe_naive_dispatch`
(with the reason attached to the counter object) on the step's counter line, so
a reader never has to guess which one produced the loss.

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

from ..instrument import Counters
from .act_quant import fake_quant_act
from .tsct import factor_qat_project, mm_mixed

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
        # parameter: the optimizer must not move it, LFB does.
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
        logits = mm_mixed(x, self.router.weight, dt)  # bf16 tensor-core, fp32 out
        probs = F.softmax(logits, dim=-1)  # [n, E]
        # a4.8: the activation quantization arm (COUNTED — a reader of the log
        # line sees it) applies to the EXPERTS' activations only. The router
        # above reads the raw state on purpose (`router` is a dense `nn.Linear`,
        # no factor matmul, and routing must not be steered by a quantizer).
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
        return mm_mixed(weighted.to(dt), self.down.weight, dt).to(torch.float32)

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
        """
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
        """
        n, k = index.shape
        flat_ids = index.reshape(-1)
        order = torch.argsort(flat_ids, stable=True)
        sorted_ids = flat_ids[order]
        counts = torch.bincount(sorted_ids, minlength=self.n_experts)
        offsets = torch.cumsum(counts, 0) - counts
        pos = torch.arange(n * k, device=index.device) - offsets[sorted_ids]
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
        """
        key = (
            dt,
            self.factor_qat,
            self.factor_qat_alpha,  # the anneal ramp moves this every step
            tuple(p._version for e in self.experts for p in e.parameters()),
        )
        cached = getattr(self, "_bank_cache", None)
        if cached is None or cached[0] != key:
            # The stack builds in fp32 (the masters' dtype), the qat arm — if
            # on — quantizes THERE (per-expert α via `dim=(1,2)`, so a banked
            # expert is identical to the same expert run through
            # `DualLatentGLU.forward` — the parity the A/B needs), and only
            # then does the cache convert to `dt`. Scales (`gate_s`/`up_s`)
            # are never quantized: they ARE the diag(s) of the factorization.
            mats = []
            for name in ("gate_u", "up_u", "v"):
                stack = torch.stack([getattr(e, name) for e in self.experts])
                if self.factor_qat != "none":
                    stack = factor_qat_project(
                        stack, self.factor_qat, self.factor_qat_alpha, dim=(1, 2)
                    )
                mats.append(stack.to(dt))
            cached = (
                key,
                tuple(mats),
                tuple(
                    torch.stack([getattr(e, name) for e in self.experts]).to(dt)
                    for name in ("gate_s", "up_s")
                ),
            )
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
        """
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
        """
        order, keep, dropped = self._permute(index)
        # Count EVERY call: a python `if dropped:` on the device tensor is
        # exactly the sync this counter exists to prevent (caught by the
        # drop-parity test when the tensor stopped being an int). A zero add
        # costs nothing once the drain batches it.
        counters.count("moe_dropped_assignments", dropped)
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
                counters.count("moe_triton_gemm")
                return self._expert_pass(
                    lambda: self._triton_dispatch(x, weight, index, order, keep), counters
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

    def _triton_dispatch(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        order: torch.Tensor,
        keep: torch.Tensor,
    ) -> torch.Tensor:
        """Gather → ONE unpadded grouped-GEMM pass (GLU fused) → weighted scatter.

        The arithmetic of `_padded_dispatch` with the padding removed: the kept
        rows are gathered expert-major (they already are — the permute sorts by
        expert id), `offs` carries the ragged group bounds, and no `[E, cap, f]`
        intermediate is ever materialized. Rows past capacity never enter, so the
        drop set is byte-for-byte the padded arm's.
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
        y = moe_kernels.expert_gemm_glu(
            x[owner].to(dt), gate_u, up_u, gate_s, up_s, v, offs, cap=self._capacity(n, k)
        )
        # fp32 weight applied where the padded arm applies it (see the note
        # there): a bf16 multiply would be a 2^-8 step the parity gate could not
        # tell from a wiring error.
        y = y.to(torch.float32) * weight.reshape(-1)[rows].unsqueeze(-1)
        out = torch.zeros(n, self.f, dtype=torch.float32, device=x.device)
        out.index_add_(0, owner, y)
        return out
