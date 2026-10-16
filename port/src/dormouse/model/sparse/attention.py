"""`BlockSparseAttention` — the MSA arm: a sparse second attention arm.

The KDA linear arm REMAINS THE DEFAULT (`use_msa = false`): this is a new arm
behind a config flag, not a replacement. The flag is read off the config object
(`getattr(cfg, "use_msa", False)`) so the arm lands as new files only; adding
the `ModelConfig` field is a follow-up for the commit that wires it into
`LoopBlock`. When the flag is off the module builds
NO parameters and its forward returns `None` with a COUNTED `msa_off` — the
model's bytes are the same as before the arm existed.

**KERNEL CHOICE** (the plan's own open verdict, "no published rank/k sweep"):
the block MASK is a pure-torch product (`FlexSparseSelector`), and the dense
attention over the gathered keys is torch's `scaled_dot_product_attention` —
the owner's rule that kernel interfaces stay pure torch in / torch out. The
FLA alternatives were checked in the installed tree and rejected on THEIR
constraints, not on speed:
  * `fla.ops.nsa.parallel_nsa` (native sparse attention, has a backward) requires
    `HQ/H` to be a power of two and **>= 16** (`nsa/parallel.py:890`) and pulls
    in `flash_attn`; the port's head counts (12, 4) cannot satisfy it.
  * `fla.ops.attn.parallel_attn` (dense, has a backward) takes `window_size`
    and `cu_seqlens` but **no arbitrary mask head**, so it cannot express the
    indexer's selected-block set.
Both are recorded here as the drop-in for a fused CUDA gather later; neither is
used by this arm.

**AGGREGATION**: gather the `r·K` key positions named by the selected blocks,
apply token-level causality, and take one softmax over the gathered keys — the
"softmax-normalized block weights" of the recipe. That is EXACTLY the dense
attention restricted to the selected blocks (the union of the blocks IS the
mask), which is the equivalence `tests/test_msa.py` replays.

CEILING (`ponytail:`): the gather materializes `[b,t,r*K,h,hd]` — O(t·r·K)
memory, and at 1M with K=32, r=4 that is already 16 GB in bf16. The 1M form
computes a per-block logsumexp and combines blocks in a streaming loop; upgrade
when a 1M run exists (the plan's verdict says the k/r sweep has not been run).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ...instrument import Counters
from .sparse import MSA_BLOCK, MSA_TOPK, FlexSparseSelector

__all__ = ["MSA_BLOCK", "MSA_TOPK", "BlockSparseAttention"]


class BlockSparseAttention(nn.Module):
    """`x [b,t,d]` → `[b,t,d]`, or `None` when `use_msa` is off (COUNTED).

    The parameter set is KDA's own `q,k,v` projections (`3·d²`, the `g`/`beta`
    of the gated-delta arm are KDA-specific and are not carried here) plus the
    indexer — and ALL of it exists only when the arm is on, so the arm's absence
    is byte-for-byte the KDA-only model.
    """

    def __init__(self, cfg, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        assert h * cfg.head_dim == d, "validate() pins n_heads*head_dim == d_model"
        self.use_msa = bool(getattr(cfg, "use_msa", False))
        self.n_heads = h
        self.head_dim = cfg.head_dim
        self.scale = 1.0 / math.sqrt(cfg.head_dim)
        self.topk = int(getattr(cfg, "msa_topk", MSA_TOPK))
        self.block = int(getattr(cfg, "msa_block", MSA_BLOCK))
        assert self.block >= 1 and self.topk >= 1, "msa_block/topk must be >= 1"
        self.selector: FlexSparseSelector | None = None
        self.q_proj: nn.Linear | None = None
        self.k_proj: nn.Linear | None = None
        self.v_proj: nn.Linear | None = None
        if self.use_msa:
            # Built only when the arm runs: `use_msa=False` must leave the whole
            # model's parameter count untouched (test_msa.py, case d).
            self.selector = FlexSparseSelector(cfg, device=device)
            self.q_proj = nn.Linear(d, d, bias=False, device=device)
            self.k_proj = nn.Linear(d, d, bias=False, device=device)
            self.v_proj = nn.Linear(d, d, bias=False, device=device)

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor | None:
        """`x [b,t,d]` (block-body input) → `[b,t,d]`, or `None` when off."""
        counters = Counters() if counters is None else counters
        if not self.use_msa:
            counters.count("msa_off")
            return None
        assert self.selector is not None and self.q_proj is not None
        b, t, d = x.shape
        return self._forward(x, counters, k_override=None)

    def forward_k(self, x: torch.Tensor, counters: Counters, k_share: float) -> torch.Tensor | None:
        """The ADAPTIVE entry: K per query from `k_share` × candidate blocks.

        `k_share` is the per-ITERATION share (the adaptive schedule's ratio for
        this slot); 0 → MSA runs with no blocks this iteration (COUNTED
        `msa_iter_skip`, returns None so the caller adds nothing), the model
        spends its budget elsewhere. The share's mean over iterations is the
        arm's `msa_capacity_ratio` (owner's 1/4), the placement is learned.
        """
        assert self.use_msa and self.selector is not None
        b, t, _ = x.shape
        n_blocks = (t + self.block - 1) // self.block
        k = int(round(k_share * n_blocks))
        if k <= 0:
            counters.count("msa_iter_skip")
            return None
        return self._forward(x, counters, k_override=k)

    def _forward(self, x, counters, k_override) -> torch.Tensor:
        b, t, d = x.shape
        blocks, keep = self.selector(x, counters, k=k_override)
        counters.count("msa_blocks", b * t * blocks.shape[-1])
        q = self.q_proj(x).reshape(b, t, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, t, self.n_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, t, self.n_heads, self.head_dim)
        out = attend_selected_blocks(q, k, v, blocks, keep, self.block, self.scale)
        assert out.shape == (b, t, self.n_heads, self.head_dim)
        return out.reshape(b, t, d)


def attend_selected_blocks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    blocks: torch.Tensor,
    keep: torch.Tensor,
    block: int,
    scale: float,
) -> torch.Tensor:
    """Gather the selected blocks' keys and attend, exactly.

    `q,k,v [b,t,h,hd]`; `blocks [b,t,K]` (sentinel `n_blocks` for padded slots),
    `keep [b,t,K]`. Returns `[b,t,h,hd]`.

    A query keeps key `p` iff `p` lies in a kept selected block AND `p <= i`
    (token-level causality). With no valid key the query would divide by a
    clamped denominator and read 0.0 — impossible here: every kept block's first
    token sits at or before the query (the selector's block-causal rule), so a
    query with any kept block always has at least one valid key.
    """
    b, t, h, hd = q.shape
    r = block
    m = blocks.shape[-1]
    offsets = torch.arange(r, device=q.device)
    pos = (blocks.unsqueeze(-1) * r + offsets).reshape(b, t, m * r)  # [b,t,M]
    safe = pos.clamp(max=max(t - 1, 0))  # padded/sentinel slots point in range
    qpos = torch.arange(t, device=q.device).view(1, t, 1)
    valid = (pos < t) & (pos <= qpos) & keep.repeat_interleave(r, dim=-1)  # [b,t,M]
    assert valid.shape == (b, t, m * r)

    b_idx = torch.arange(b, device=q.device).view(b, 1, 1).expand(b, t, m * r)
    kf = k.reshape(b, t, h * hd)
    vf = v.reshape(b, t, h * hd)
    kk = kf[b_idx, safe].reshape(b, t, m * r, h, hd)
    vv = vf[b_idx, safe].reshape(b, t, m * r, h, hd)

    scores = torch.einsum("bthd,btmhd->bthm", q, kk) * scale
    scores = scores.masked_fill(~valid[:, :, None, :], float("-inf"))
    mx = scores.amax(-1, keepdim=True)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    p = torch.exp(scores - mx)  # exp(-inf) = 0 for the masked slots
    den = p.sum(-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
    assert torch.isfinite(den).all(), "every query must have a valid key"
    out = torch.einsum("bthm,btmhd->bthd", p, vv) / den
    return out
