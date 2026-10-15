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

* **fused (CUDA, `torch._grouped_mm`)**: a ragged grouped GEMM, no `[E, cap, f]`
  materialization and no zero-filled padded rows. Taken when the op exists in
  this torch and the layer's own shapes satisfy its 16-byte stride rule; the
  arithmetic is the same, the layout is not.
* **naive (the CPU path and the correctness reference)**: gather → per-expert
  matmul → scatter over a capacity-bound permute. The day-one implementation,
  kept because it is what the fused arm is compared against — the same rows, the
  same capacity rule, the same drop counter.

Both count: `moe_grouped_mm` or `moe_naive_dispatch` (with the reason attached
to the counter object) on the step's counter line, so a reader never has to
guess which one produced the loss.
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

    def _permute(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """The routing permute: `(order, keep, dropped)` — one place, both arms.

        Both arms are permuted by the SAME rows and drop the SAME ones, which is
        what makes a parity run comparable and the drop count one number rather
        than two opinions. The capacity rule is burn's `[E, cap, f]` rule, kept
        exact: `cap = max(k, ceil(capacity·n·k/E))`, floored at `k` so a token is
        never dropped by the floor itself.
        """
        n, k = index.shape
        flat_ids = index.reshape(-1)
        order = torch.argsort(flat_ids, stable=True)
        sorted_ids = flat_ids[order]
        counts = torch.bincount(sorted_ids, minlength=self.n_experts)
        offsets = torch.cumsum(counts, 0) - counts
        pos = torch.arange(n * k, device=index.device) - offsets[sorted_ids]
        cap = max(k, math.ceil(self.capacity * n * k / self.n_experts))
        keep = pos < cap
        return order, keep, int((~keep).sum())

    def _grouped_reason(self, x: torch.Tensor) -> str | None:
        """`None` if `torch._grouped_mm` can run THIS call, else the reason why not.

        Checked from the ABI rather than by catching an exception at runtime:
        the op must exist in this torch (2.13+), the tensor must be on CUDA, and
        every output width must be a whole number of 16-byte elements — bf16
        wants `rank % 8 == 0`, fp32 `rank % 4 == 0`, on `rank` and on `f`. The
        three cases that fail are the three the spike's probe recorded
        ("strides should be multiple of 16 bytes"); catching them per step would
        hide a real shape bug behind a fallback.
        """
        if not hasattr(torch, "_grouped_mm"):
            return "this torch has no torch._grouped_mm"
        if not x.is_cuda:
            return f"input is on {x.device.type}: torch._grouped_mm is CUDA-only"
        es = self._bank_dtype().itemsize
        rank = self.experts[0].rank
        if (rank * es) % 16 or (self.f * es) % 16:
            return (
                f"torch._grouped_mm needs every output width a multiple of 16 bytes; "
                f"rank={rank}, f={self.f} at {es} bytes/element are not"
            )
        return None

    def _bank_dtype(self) -> torch.dtype:
        """The dtype the experts compute in (their own `bf16` switch, not the device's)."""
        return self.experts[0].compute_dtype

    def _stacked(self, dt: torch.dtype) -> tuple[torch.Tensor, ...]:
        """The expert bank as `[E, ·]` tensors for the grouped op.

        One small copy per forward — `E` is 3-to-8 here, so it is KB, not MB —
        and `torch.stack` keeps the graph, so the gradients land back on each
        expert's own parameters. Not a cached buffer: a cache would silently
        outlive an optimizer step.
        """

        def bank(name: str) -> torch.Tensor:
            return torch.stack([getattr(e, name) for e in self.experts]).to(dt)

        return bank("gate_u"), bank("up_u"), bank("v"), bank("gate_s"), bank("up_s")

    def _grouped_dispatch(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        order: torch.Tensor,
        keep: torch.Tensor,
    ) -> torch.Tensor:
        """Ragged grouped GEMM: `torch._grouped_mm(a [R, K], w [G, K, N], offs)`.

        `a[g] @ w[g]` for the rows `offs` slices into groups — the layout the
        spike's probe established (`mat_b` is `(G, K, N)`, and the docstring in
        `torch/nn/functional.py` is wrong about the 2-D-`mat_a` case). Rows past
        capacity are gone by the same `keep` mask the naive path drops them with,
        so the fused arm changes how the arithmetic is laid out, never which
        tokens get an expert.
        """
        n, k = index.shape
        dt = self._bank_dtype()
        flat_w = weight.reshape(-1)
        gid_all = index.reshape(-1)[order]
        rows = order[keep]  # flat (token, slot) pairs that survive the cap
        owner = rows // k  # the token each ragged row belongs to
        gid = gid_all[keep]  # the expert each ragged row belongs to
        offs = torch.bincount(gid, minlength=self.n_experts).cumsum(0).to(torch.int32)

        xg = x[owner].to(dt)
        gate_u, up_u, v, gate_s, up_s = self._stacked(dt)
        g = torch._grouped_mm(xg, gate_u, offs) * gate_s[gid]
        u = torch._grouped_mm(xg, up_u, offs) * up_s[gid]
        h = F.silu(g) * u
        y = torch._grouped_mm(h, v, offs)  # [R, f]
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
            y = self.experts[e](x[rows]) * flat_w[order[sel]].unsqueeze(-1)
            out.index_add_(0, rows, y)
            counters.count("moe_expert_calls")
        counters.count("moe_tokens", n)
        return out

    def _dispatch(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        index: torch.Tensor,
        counters: Counters,
    ) -> torch.Tensor:
        """Capacity-bound permute, then the fused arm if it can run here, else naive.

        The fallback is COUNTED (`moe_naive_dispatch` + the reason on the counter
        object) because the naive path is 3× the launches and a different GEMM
        layout — a degradation, not another name for the same function.
        """
        order, keep, dropped = self._permute(index)
        if dropped:
            counters.count("moe_dropped_assignments", dropped)
        why = self._grouped_reason(x)
        if why is None:
            counters.count("moe_grouped_mm")
            return self._grouped_dispatch(x, weight, index, order, keep)
        counters.count("moe_naive_dispatch")
        counters.why = why  # type: ignore[attr-defined]
        return self._naive_dispatch(x, weight, index, order, keep, counters)
