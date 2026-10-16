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

MEMORY (2026-10-16, measured): the gather runs in SLOT CHUNKS of
`MSA_SLOT_CHUNK` selected-block columns and each chunk is a checkpoint region
(`attend_selected_blocks`), so the largest tensor this arm builds is
`[b,t,chunk*r,h,hd]` and not `[b,t,K*r,h,hd]` — and the backward recomputes it
instead of retaining all `K` of them. Measured on the `small` preset (fp32 at
this seam, d 768) at b8 s512: 13.59 GiB peak and an OOM before, 3.15 GiB after;
the whole-m gather alone was 1.61 GB ×2 (k and v) ×4 loop iterations = 12.9 GB
of autograd-retained copies.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...instrument import Counters
from .sparse import MSA_BLOCK, MSA_TOPK, FlexSparseSelector

__all__ = ["MSA_BLOCK", "MSA_TOPK", "MSA_SLOT_CHUNK", "BlockSparseAttention"]

#: Selected-block columns gathered per chunk in `attend_selected_blocks`, i.e.
#: `MSA_SLOT_CHUNK * block` key positions per gather. A module constant and not
#: a config field: nothing sweeps it, and the value that matters is the ONE
#: measured bound it sets (the chunk gather is the largest tensor the arm
#: builds). MEASURED 2026-10-16, `small` preset b8 s512, 6 steps each, peak is
#: `torch.cuda.max_memory_allocated` (`/tmp/opencode/chunk_sweep.py`):
#: 1 slot 882 ms / 5.06 GiB · 2 slots 850 ms / 3.56 GiB · **4 slots 661 ms /
#: 3.15 GiB** · 8 slots 633 ms / 3.51 GiB · all 32 slots in one region
#: 632 ms / 7.73 GiB. 4 is the lowest peak of the five and within 4.6% of the
#: fastest — smaller chunks pay more launches, bigger ones pay more memory.
MSA_SLOT_CHUNK = 4


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
    query with any kept block always has at least one valid key. That invariant
    is what guards the online-softmax denominator below, and the assert is the
    gate on it.

    ROUTE (2026-10-16): **slot chunks**, not a per-slot loop. The `m` selected
    columns are walked in groups of `MSA_SLOT_CHUNK`, each group gathered,
    scored and folded into a running `(max, den, num)` softmax accumulator with
    the standard rescale — mathematically the one softmax over all kept keys,
    bit-for-bit close to it (`tests/test_msa.py` case e pins the parity).

    Each chunk is a `torch.utils.checkpoint` region (only outside `no_grad`),
    and that half is load-bearing rather than hygiene: the chunk's gathered
    `[b,t,chunk*r,h,hd]` is what the *backward* needs, so a plain loop would
    leave one retained copy per chunk — all `m` of them, the full tensor again.
    Checkpointing recomputes the gather in the backward instead, which is what
    makes the peak O(b·chunk·r·h·hd) rather than O(b·m·r·h·hd). Measured on the
    `small` preset at b8 s512 (2026-10-16, `/tmp/opencode/chunk_sweep.py`, 6
    steps, `torch.cuda.max_memory_allocated`): 13.59 GiB and an OOM with no
    chunk and no region, 7.73 GiB with the region but one whole-`m` chunk,
    3.15 GiB at `MSA_SLOT_CHUNK` 4. The same gather at b1's shape
    (b16 s1024, d 2048, fp32) is 2.1 GB per chunk instead of 17 GB whole.
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

    mx = torch.full((b, t, h), float("-inf"), dtype=q.dtype, device=q.device)
    den = torch.zeros((b, t, h), dtype=q.dtype, device=q.device)
    num = torch.zeros((b, t, h, hd), dtype=q.dtype, device=q.device)
    for s0 in range(0, m, MSA_SLOT_CHUNK):
        sl = slice(s0 * r, min(s0 + MSA_SLOT_CHUNK, m) * r)
        mxc, denc, numc = _maybe_checkpoint(_chunk, q, k, v, safe[:, :, sl], valid[:, :, sl], scale)
        mx_new = torch.maximum(mx, mxc)
        # Nothing valid anywhere yet (impossible for a query with a kept block,
        # but reachable while the first chunks are all empty): 0 keeps the
        # rescale finite and leaves den at 0, so the assert below fires.
        mx_new = torch.where(torch.isfinite(mx_new), mx_new, torch.zeros_like(mx_new))
        alpha = torch.exp(mx - mx_new)
        beta = torch.exp(mxc - mx_new)
        mx = mx_new
        den = den * alpha + denc * beta
        num = num * alpha.unsqueeze(-1) + numc * beta.unsqueeze(-1)

    den = den.clamp_min(torch.finfo(den.dtype).tiny)
    assert torch.isfinite(den).all(), "every query must have a valid key"
    return num / den.unsqueeze(-1)


def _chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    safe: torch.Tensor,
    valid: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One slot chunk → `(chunk max, chunk denominator, chunk numerator)`.

    `safe/valid [b,t,c]` is a column slice of the gather plan; the returned
    tensors are chunk-normalized (`p = exp(scores - chunk max)`) and the caller
    rescales them into its running accumulator. The chunk max may be `-inf`
    (every slot in this chunk masked) — the caller's rescale zeroes that
    contribution, and the `finite or zero` rule is applied HERE so `p` is a
    clean 0 row rather than a NaN.
    """
    b, t, h, hd = q.shape
    c = safe.shape[-1]
    b_idx = torch.arange(b, device=q.device).view(b, 1, 1).expand(b, t, c)
    kk = k.reshape(b, t, h * hd)[b_idx, safe].reshape(b, t, c, h, hd)
    vv = v.reshape(b, t, h * hd)[b_idx, safe].reshape(b, t, c, h, hd)
    scores = torch.einsum("bthd,btchd->bthc", q, kk) * scale
    scores = scores.masked_fill(~valid[:, :, None, :], float("-inf"))
    mxc = scores.amax(-1)  # [b,t,h]
    mx_used = torch.where(torch.isfinite(mxc), mxc, torch.zeros_like(mxc))
    p = torch.exp(scores - mx_used[..., None])  # exp(-inf) = 0 for masked slots
    return mxc, p.sum(-1), torch.einsum("bthc,btchd->bthd", p, vv)


def _maybe_checkpoint(fn, *args):
    """`fn(*args)` — inside a checkpoint region when a gradient is being built.

    The region is what keeps the gather from being retained for the backward
    (see the route note above); outside `no_grad` it would only buy the
    recompute cost, so inference runs the plain call.
    """
    if torch.is_grad_enabled() and any(
        isinstance(a, torch.Tensor) and a.requires_grad for a in args
    ):
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)
