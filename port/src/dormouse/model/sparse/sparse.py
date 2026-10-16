"""The MSA block indexer (QSA → MSA) — the plan's §3.5 item 4 recipe.

**SOURCE**: `docs/decisions/2026-10-15-1b-readiness-plan.md` §3.5 item 4 (the
MQA + block-indexer recipe, carried from the burn re-entry review
`docs/reviews/msa-reentry-2026-10-02.md`), and the verdict row in
`docs/research/2026-10-15-1b-tech-verification.md`: *"sparse attention indexer:
open, no published rank/k sweep"*. That verdict is why `msa_block` and
`msa_topk` are SWEPT, never cited, and the defaults here are a starting point,
not a claim.

The recipe, one clause at a time:

* **MQA**: `MSA_Q_HEADS = 4` query heads, ONE shared key head. The indexer is a
  scorer, not the attention — it never carries values.
* **avg-pool keys into blocks of `r`**: `msa_block` (default 4) tokens per
  block; the pooled key is the block's mean.
* **block-causal ReLU scoring**: block `j` is a candidate for query `i` iff its
  first token is at or before `i` (`j*r <= i`), and the score is
  `Σ_h ReLU(q_h · k_block_j)` — non-negative by construction, so a logit is
  never a negative "similarity", which is the paper's ReLU choice.
* **top-K blocks**: `msa_topk` per query (default 32; K ~ 32-64 at 1M).

A query whose candidate blocks number fewer than K truncates the top-K (every
early position, and every sequence shorter than K blocks) — that is COUNTED as
`msa_truncated`, never a silent `-inf` row (the burn side's ADR-0015 lesson:
the pre.4 argtopk emitted the out-of-range seed coordinate for exactly this
case). The count is pure shape arithmetic, so it costs no device sync.

CEILING (`ponytail:`): the score tensor is `[b, t, n_blocks]` = O(t·ceil(t/r))
memory — fine at 512-2048, impossible at 1M. The 1M form scores blocks against
blocks (or streams the indexer in chunks); upgrade when a run at that length
exists, which the plan's verdict says has not been swept.
"""

from __future__ import annotations

import torch
from torch import nn

from ...instrument import Counters

#: Indexer query heads — the recipe's MQA shape ("4 q-heads, 1 shared k-head").
MSA_Q_HEADS = 4
#: Default tokens per pool block (the recipe's `r`).
MSA_BLOCK = 4
#: Default blocks selected per query (the recipe's `KB`, K ~ 32-64 at 1M).
MSA_TOPK = 32


class FlexSparseSelector(nn.Module):
    """`x [b,t,d]` → `(blocks [b,t,K] long, keep [b,t,K] bool)`.

    `keep[i,j]` is True iff slot `j` is a REAL selected block for query `i`; a
    padded slot (query truncated by the top-K) carries the sentinel index
    `n_blocks` and `keep=False`, so a consumer never attends a phantom block.

    Deterministic: the score is a pure function of `x` and the projections, and
    `topk` breaks ties by index order — the same input gives the same blocks
    (the port's rule that a discrete product is reproducible).
    """

    def __init__(self, cfg, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        d = cfg.d_model
        self.block = int(getattr(cfg, "msa_block", MSA_BLOCK))
        self.topk = int(getattr(cfg, "msa_topk", MSA_TOPK))
        self.q_heads = MSA_Q_HEADS
        self.head_dim = cfg.head_dim
        assert self.block >= 1, f"msa_block={self.block} must be >= 1"
        assert self.topk >= 1, f"msa_topk={self.topk} must be >= 1"
        self.q_idx = nn.Linear(d, self.q_heads * self.head_dim, bias=False, device=device)
        self.k_idx = nn.Linear(d, self.head_dim, bias=False, device=device)

    def forward(
        self, x: torch.Tensor, counters: Counters | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        r, k = self.block, self.topk
        n = (t + r - 1) // r  # ceil: the final block may be partial
        counters = Counters() if counters is None else counters
        assert x.ndim == 3, f"selector takes [b,t,d], got {tuple(x.shape)}"

        # avg-pool keys into blocks: zero-pad the tail, then the mean over the
        # block's r slots. A partial final block's padding drags its mean toward
        # 0 — acceptable for a SCORER (the aggregator masks the pad tokens out).
        kk = self.k_idx(x)
        pad = n * r - t
        if pad:
            kk = torch.cat([kk, kk.new_zeros(b, pad, self.head_dim)], dim=1)
        k_blk = kk.reshape(b, n, r, self.head_dim).mean(dim=2)  # [b,n,hd]

        q = self.q_idx(x).reshape(b, t, self.q_heads, self.head_dim)
        proj = torch.einsum("bihd,bjd->bijh", q, k_blk)  # [b,t,n,H]
        score = torch.relu(proj).sum(-1)  # [b,t,n], non-negative

        pos = torch.arange(t, device=x.device).view(1, t, 1)
        causal = torch.arange(n, device=x.device).view(1, 1, n) * r <= pos
        score = score.masked_fill(~causal, float("-inf"))

        k_eff = min(k, n)
        sel = score.topk(k_eff, dim=-1).indices  # [b,t,k_eff]
        keep = torch.isfinite(score.gather(-1, sel))
        # Padding sentinel = n (out of range) so `keep=False` slots can never be
        # gathered as block 0's real keys.
        sentinel = torch.full_like(sel, n)
        blocks = torch.where(keep, sel, sentinel)
        if k_eff < k:
            extra = torch.full((b, t, k - k_eff), n, dtype=sel.dtype, device=x.device)
            blocks = torch.cat([blocks, extra], dim=-1)
            keep = torch.cat([keep, torch.zeros_like(extra, dtype=torch.bool)], dim=-1)
        assert blocks.shape == (b, t, k), f"selector shape {tuple(blocks.shape)}"

        # Truncation is a property of the SHAPE: query i is short iff its
        # candidate count floor(i/r)+1 < K, i.e. i < (K-1)*r. No tensor read.
        counters.count("msa_truncated", min(t, (k - 1) * r))
        return blocks, keep
