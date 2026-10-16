"""The weight-shared loop block — one parameter set, `max_iter` iterations.

Mirrors `crates/dormouse-core/src/loop_block.rs`. One iteration, in order:

1. the **iteration embedding** `e_i` is added to the state (this is the only
   thing that makes the `T` passes distinguishable, since the body is shared);
2. a single **RMSNorm** (shared, like everything else here) pre-norms it;
3. the **controller** reads `[h_ctx ; h0]` and emits `sigmoid(w_attn)`,
   `sigmoid(w_mem)`, `sigmoid(w_ffn)` plus the expert blend logits;
4. the **attention** (KDA, or its counted SDPA fallback) and the **experts**
   run, each scaled by its gate;
5. the **Engram** memory joins as a convex mixture with a hard floor:
   `lam = min(w_mem, lam_max)`;
6. the body is deposited by **ReZero** (`h = h_ctx + α·y`, α init 1.0 — NOT 0:
   at 0 the body gets no gradient at all) or by **AttnRes**' pseudo-query
   mixture (mutually exclusive by construction);
7. the readout is `out_proj(h)`, and the final state is the AVERAGE over
   iterations. `L_Rec` — the per-iteration CE through the same `lm_head` — is
   accumulated inside the loop (deep supervision, burn's own shape), which is
   also why it is never materialized as `[T, b, t, d]`.

There is no halt head and no PonderNet: depth is fixed by default and the
per-iteration CE is an unweighted mean (ADR-0013). The ADAPTIVE DEPTH arm
passes `depth=T` to run `T` iterations instead, and the readout divides by the
`T` actually run so different depths are scale-comparable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import Counters
from .engram import Engram, memory_floor_mix
from .experts import SparseMoE
from .kda import KDAAttention
from .sparse import BlockSparseAttention
from .tsct import make_projection, mm_mixed

__all__ = ["AttnResSlot", "LoopBlock", "depth_attend", "memory_floor_mix"]


def _rms_no_gain(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class AttnResSlot(nn.Module):
    """One pseudo-query per loop-iteration slot (arXiv:2603.15031 §5).

    Zero-initialised query × gain of ones: the first forward is the equal-weight
    average, so the arm starts where ReZero starts and the A/B is not a
    head start.
    """

    def __init__(self, d: int, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d, device=device))
        self.gain = nn.Parameter(torch.ones(d, device=device))


def depth_attend(sources: list[torch.Tensor], slot: AttnResSlot, eps: float) -> torch.Tensor:
    """The AttnRes mixture: `softmax_i(q_eff · RMSNorm(k_i))` over the sources.

    The score convention is the PAPER's — `q·RMSNorm(k)` with **no `1/√d`**
    (`burn_attnres::ScoreForm::Paper`, the default). The burn side shipped
    `d^-0.5` plus an L2 norm, together a `d`-fold logit compression, both
    silent; the fold `q_eff = γ ⊙ w_l` is fidelity fix F-F7 (the paper's
    affine norm, with its gain).
    """
    q_eff = slot.gain * slot.query
    keys = torch.stack([_rms_no_gain(s, eps) for s in sources], dim=-2)  # [b,t,S,d]
    scores = (keys * q_eff).sum(-1)  # [b,t,S]
    w = F.softmax(scores, dim=-1)
    return (w.unsqueeze(-1) * torch.stack(sources, dim=-2)).sum(-2)


class LoopBlock(nn.Module):
    def __init__(self, cfg, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        d = cfg.d_model
        self.d = d
        self.max_iter = cfg.max_iter
        self.eps = cfg.norm_eps
        self.lam_max = cfg.engram_lam_max
        self.use_engram = cfg.use_engram
        self.use_attnres = cfg.use_attnres
        #: The compute dtype the bf16 plan names for the block's small mixed
        #: projections (controller, mem_dense) — the experts' own choice.
        self.compute_dtype = torch.bfloat16 if cfg.bf16 else torch.float32
        self.norm = nn.RMSNorm(d, eps=cfg.norm_eps, device=device)
        # The three gates only. burn's controller ALSO emits the expert-blend
        # logits and its MoE "adds NO parameters"; the port's MoE owns a
        # dedicated router linear (the baked decision for this port), so the
        # controller carries three columns and no dead ones — a parameter with
        # no gradient is the defect class this project's counters exist for.
        self.controller = nn.Linear(2 * d, 3, bias=False, device=device)
        self.attn = KDAAttention(cfg, device=device) if cfg.use_kda else None
        # The second attention arm (owner's 1M-context directive, 2026-10-16):
        # MSA rides an ADDITIVE branch beside KDA, off by default; the ADAPTIVE
        # schedule (`adaptive_msa`) gives it a per-ITERATION share whose mean
        # is `msa_capacity_ratio` (the owner's 1/4), decided BY the model: one
        # learnable logit per iteration slot, softmax-normalized across the
        # slots and scaled to the mean — so the model picks WHERE the sparse
        # attention's compute is spent, the total budget is fixed.
        self.adaptive_msa = bool(getattr(cfg, "adaptive_msa", False))
        self.msa = BlockSparseAttention(cfg, device=device) if cfg.use_msa else None
        self.msa_capacity_ratio = float(getattr(cfg, "msa_capacity_ratio", 0.25))
        if self.msa is not None:
            assert 0.0 < self.msa_capacity_ratio <= 1.0, "msa_capacity_ratio in (0, 1]"
            self.msa_gate = nn.Parameter(torch.zeros(cfg.max_iter, device=device))
        self.moe = SparseMoE(cfg, device=device)
        self.engram = Engram(cfg, device=device) if cfg.use_engram else None
        # The dense half of the memory branch's mixture — the `(1-g)·v_t` term.
        self.mem_dense = nn.Linear(d, d, device=device)
        self.out_proj = make_projection(d, d, cfg, rank=cfg.rank, device=device)
        self.iter_embed = nn.Parameter(torch.zeros(cfg.max_iter, d, device=device))
        self.residual_scale = nn.Parameter(torch.ones(1, device=device))
        self.attnres = (
            nn.ModuleList([AttnResSlot(d, device) for _ in range(cfg.max_iter)])
            if cfg.use_attnres
            else None
        )

    def forward(
        self,
        h0: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        lm_head: nn.Module | None = None,
        counters: Counters | None = None,
        depth: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Counters]:
        """`h0 [b,t,d]` → (`mean readout [b,t,d]`, SUM of per-iteration mean CEs, counters).

        The CE is a SUM over iterations, not a mean: the model divides by the
        number of terms once it has added the final readout's own CE (see
        `model.DormouseModel.forward` for why that term exists).

        `depth=None` runs `max_iter` iterations (ADR-0013's fixed depth,
        unchanged); `depth=T` runs exactly `T` and the readout mean divides by
        that `T`, never by `max_iter` — the ADAPTIVE DEPTH arm's contract.
        `counters["depth"]` accumulates the depth each forward actually ran.
        """
        counters = Counters() if counters is None else counters
        b, t, d = h0.shape
        assert d == self.d, f"loop block width {self.d} != input {d}"
        T = self.max_iter if depth is None else depth
        assert 1 <= T <= self.max_iter, f"depth {T} outside 1..={self.max_iter}"
        counters.count("depth", T)
        h = h0
        h0_flat = h0.reshape(b * t, d)
        out_acc = torch.zeros_like(h0)
        rec = torch.zeros((), dtype=torch.float32, device=h0.device)
        sources: list[torch.Tensor] = [h0] if self.use_attnres else []
        for i in range(T):
            counters.count("iter")
            ctx = h + self.iter_embed[i].view(1, 1, d)
            normed = self.norm(ctx)
            ctrl = torch.cat([ctx.reshape(b * t, d), h0_flat], dim=-1)
            raw = mm_mixed(ctrl, self.controller.weight, self.compute_dtype)  # MAGMA fix
            w_attn = torch.sigmoid(raw[:, 0:1])
            w_mem = torch.sigmoid(raw[:, 1:2])
            w_ffn = torch.sigmoid(raw[:, 2:3])

            attn = (
                self.attn(normed, counters=counters).reshape(b * t, d) * w_attn
                if self.attn is not None
                else torch.zeros(b * t, d, dtype=torch.float32, device=h0.device)
            )
            if self.msa is not None:
                # The ADAPTIVE share: softmax over the iteration gates × max_iter
                # = a per-slot share whose MEAN over slots is 1; × the arm's
                # capacity ratio (the owner's 1/4) it becomes the slot's budget.
                # The ROUNDED share picks the block count (compute follows), and
                # the SOFT share multiplies the branch's output below, so the
                # gates receive gradient — the model learns WHERE to spend the
                # 25%, and a zero share can switch a slot off entirely.
                if self.adaptive_msa and self.max_iter > 1:
                    share = torch.softmax(self.msa_gate, dim=0) * self.max_iter
                    counters.count("msa_iter_gate")
                    msa_out = self.msa.forward_k(
                        normed, counters, float(share[i].detach()) * self.msa_capacity_ratio
                    )
                    if msa_out is not None:
                        attn = attn + msa_out.reshape(b * t, d) * w_attn * share[i]
                else:
                    msa_out = self.msa(normed, counters=counters)
                    if msa_out is not None:
                        attn = attn + msa_out.reshape(b * t, d) * w_attn

            if self.engram is not None and input_ids is not None:
                mem = self.engram(normed, input_ids, counters).reshape(b * t, d)
                dense = (
                    (
                        mm_mixed(
                            normed.reshape(b * t, d), self.mem_dense.weight, self.compute_dtype
                        )
                        + self.mem_dense.bias.to(self.compute_dtype)
                    )
                    .to(torch.float32)
                    .reshape(b * t, d)
                )
                branch = memory_floor_mix(mem, dense, w_mem, self.lam_max)
            else:
                branch = torch.zeros(b * t, d, dtype=torch.float32, device=h0.device)

            ffn = self.moe(normed.reshape(b * t, d), counters=counters) * w_ffn
            y = (attn + branch + ffn).reshape(b, t, d)

            if self.use_attnres:
                counters.count("attnres_write")
                sources.append(y)
                h = depth_attend(sources, self.attnres[i], self.eps)
            else:
                counters.count("rezero_write")
                h = ctx + y * self.residual_scale

            step_out = self.out_proj(h.reshape(b * t, d)).reshape(b, t, d)
            out_acc = out_acc + step_out
            if targets is not None:
                assert lm_head is not None, "per-iteration CE needs the lm_head"
                logits = lm_head(step_out.reshape(b * t, d).to(torch.float32))
                ce = -F.log_softmax(logits, dim=-1).gather(1, targets.reshape(-1, 1))
                rec = rec + ce.mean()
        rec = None if targets is None else rec
        return out_acc / T, rec, counters
