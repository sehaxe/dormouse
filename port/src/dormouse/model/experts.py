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

Dispatch is the naive gather → expert → scatter over a capacity-bound permute.
That is the day-one CORRECT path, and it is also the measured wall the plan
names (spike phase 2: grouped topk-2 MoE was 339× the dense loop at iso-total
params). A fused grouped-MoE kernel is a TODO hook, not a hidden branch — see
`_dispatch`'s comment for the seam it replaces.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import Counters


class DualLatentGLU(nn.Module):
    """`x [n, d] → [n, f]`: two rank-`r` latents, the GLU at the latent."""

    def __init__(
        self,
        d: int,
        f: int,
        rank: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.d, self.f, self.rank = d, f, rank
        self.compute_dtype = dtype
        for name in ("gate", "up"):
            # `U [d, r]` and `s [r]`: the latent half of TSCT's factorization.
            setattr(
                self, f"{name}_u", nn.Parameter(torch.randn(d, rank, device=device) / math.sqrt(d))
            )
            setattr(self, f"{name}_s", nn.Parameter(torch.ones(rank, device=device)))
        # The write `V [r, f]` — one, at the latent, after the GLU.
        self.v = nn.Parameter(torch.randn(rank, f, device=device) / math.sqrt(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = self.compute_dtype
        xu = x if dt == torch.float32 else x.to(dt)
        g = (xu @ self.gate_u.to(dt)) * self.gate_s.to(dt)
        u = (xu @ self.up_u.to(dt)) * self.up_s.to(dt)
        h = F.silu(g) * u
        y = h @ self.v.to(dt)
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
        self.experts = nn.ModuleList(
            DualLatentGLU(d, f, cfg.rank, dtype=dtype, device=device) for _ in range(cfg.n_experts)
        )
        self.shared = DualLatentGLU(d, f, cfg.shared_rank, dtype=dtype, device=device)
        # The router: ONE 2-D linear on the raw hidden state (not on the
        # latent), softmax over the bank, then top-k + a capacity-bound permute.
        self.router = nn.Linear(d, cfg.n_experts, bias=False, device=device)
        # The shared write back to the residual width.
        self.down = nn.Linear(f, d, bias=False, device=device)

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        counters = Counters() if counters is None else counters
        n = x.shape[0]
        assert x.ndim == 2 and x.shape[1] == self.d, f"MoE input {tuple(x.shape)} != [n, {self.d}]"
        probs = F.softmax(self.router(x), dim=-1)  # [n, E]
        weighted = torch.zeros(n, self.f, dtype=torch.float32, device=x.device)
        if self.topk == 0:
            counters.count("moe_dense_forwards", n * self.n_experts)
            for e, expert in enumerate(self.experts):
                weighted = weighted + expert(x) * probs[:, e : e + 1]
        else:
            weight, index = probs.topk(self.topk, dim=-1)
            # Renormalized (burn's `topk_blend`): a top-1 token's gate is
            # exactly 1.0, so the sparse arm and the dense control run at the
            # same output magnitude — the A/B compares specialization, not scale.
            weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            weighted = self._dispatch(x, weight, index, counters)
        # The shared expert is ALWAYS on: it is the write path, not a routed arm.
        weighted = weighted + self.shared(x)
        counters.count("moe_shared_forwards", n)
        return self.down(weighted)

    def _dispatch(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        counters: Counters,
    ) -> torch.Tensor:
        """Capacity-bound permute + naive gather → expert → scatter.

        TODO(fused-moe): replace this pair (`index_add_` over a python loop)
        with a vLLM-class grouped GEMM (torch's `_grouped_mm`/Triton), the seam
        the readiness plan §4.1 measures at 339× the dense loop. The naive path
        below stays the correctness reference for it.
        """
        n, k = index.shape
        flat_ids = index.reshape(-1)
        flat_w = weight.reshape(-1)
        order = torch.argsort(flat_ids, stable=True)
        sorted_ids = flat_ids[order]
        counts = torch.bincount(sorted_ids, minlength=self.n_experts)
        offsets = torch.cumsum(counts, 0) - counts
        pos = torch.arange(n * k, device=x.device) - offsets[sorted_ids]
        # Tokens per expert, with a floor of `k` so a token is never dropped by
        # the floor itself (burn's `[E, cap, f]` materialization cap, kept exact).
        cap = max(k, math.ceil(self.capacity * n * k / self.n_experts))
        keep = pos < cap
        dropped = int((~keep).sum())
        if dropped:
            counters.count("moe_dropped_assignments", dropped)
        owner = order // k  # which token each (token, slot) pair belongs to
        out = torch.zeros(n, self.f, dtype=torch.float32, device=x.device)
        for e in torch.unique(sorted_ids[keep]).tolist():
            sel = keep & (sorted_ids == e)
            rows = owner[sel]
            y = self.experts[e](x[rows]) * flat_w[order[sel]].unsqueeze(-1)
            out.index_add_(0, rows, y)
            counters.count("moe_expert_calls")
        counters.count("moe_tokens", n)
        return out
