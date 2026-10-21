"""Per-token MoE width: the router's own confidence picks each token's `k`.

The router already computes the full logit vector for every token and
`SparseMoE` already gathers top-k indices, so a per-token width costs ONE masked
top-k — no new parameters, no second forward, no training. This module is the
pure policy (no module state, no host reads); `experts.SparseMoE` wires it into
the routed branch.

Two policies, both pure functions of the router logits:

* **entropy** — normalized Shannon entropy `h = H(softmax(z)) / log E`, in
  `[0, 1]` (1 is the uniform router). The width grows linearly from `k_min` at
  `entropy_lo` to `k_max` at `entropy_hi`::

      t = clamp((h - lo) / (hi - lo), 0, 1)
      k = k_min + floor(t * (k_max - k_min))

* **margin** — the top-1 minus top-2 LOGIT gap `z_(1) - z_(2)`, a confidence
  margin in logit units. A large gap means a confident token, which should run
  FEWER experts, so the same `[lo, hi]` window is read in the opposite
  direction::

      t = clamp((gap - lo) / (hi - lo), 0, 1)
      k = k_max - floor(t * (k_max - k_min))

`off` is the control: a constant `k_max`. `SparseMoE` never calls this module on
that arm at all — that is what keeps every existing run and snapshot
bit-identical (the gate is a pinned digest of the pre-change forward).

Why `floor` and not `round`: the width changes where the normalized confidence
crosses a level boundary, a deterministic integer map with no ties, and `t == 1`
lands exactly on `k_max`. Entropy is computed with `torch.special.entr`, so an
underflowed probability (`p = 0`, `log p = -inf`) contributes 0 instead of NaN.

**No host sync**: every function returns a device tensor. `select_topk` masks a
copy of the scores (`-inf` past each row's own `k`) and then asks the SAME
top-k for `k_max` slots; the slots past a row's `k` are returned as an `active`
mask so the caller can zero their weights. The `-inf` never reaches arithmetic
other than `topk`'s comparison — the expert bank sees only finite `x` rows with
zero weights — so the padded `bmm` path cannot see a NaN.
"""

from __future__ import annotations

import math

import torch

from ..config import MOE_TOPK_MODES, MOE_TOPK_SLOTS_MAX
from ..instrument import loud


def check_policy(mode: str, k_min: int, k_max: int) -> None:
    """LOUD refusals for the static policy bounds — config load and runtime seam.

    Called by `ModelConfig.validate` (through the field checks) and by
    `SparseMoE.__init__`, because a hand-built config reaches the module without
    passing the loader. Every refusal names cause AND escape (ADR-0011).
    """
    if mode not in MOE_TOPK_MODES:
        loud(
            f"moe_topk_mode={mode!r} is not a width policy. Known: {list(MOE_TOPK_MODES)} — "
            f"`off` the constant `moe_topk` (every number on record), `entropy` the "
            f"normalized router entropy, `margin` the top-1/top-2 logit gap. Escape: name "
            f"one of those, or land the policy and its gate before the config can lie "
            f"about it."
        )
    if k_min < 1:
        loud(
            f"moe_topk_min={k_min}: every token runs at least one expert (the router's "
            f"own top-1), so a width below 1 has no meaning. Escape: a value >= 1."
        )
    if k_min > k_max:
        loud(
            f"moe_topk_min={k_min} > moe_topk_max={k_max}: the window is inverted, so the "
            f"policy would clamp every token to one end. Escape: k_min <= k_max "
            f"(defaults 1 <= 4)."
        )
    if k_max > MOE_TOPK_SLOTS_MAX:
        loud(
            f"moe_topk_max={k_max} > {MOE_TOPK_SLOTS_MAX}: the per-token histogram "
            f"(moe_k1..moe_k{MOE_TOPK_SLOTS_MAX}) and the fused backward's unrolled "
            f"tile both stop there. Escape: k_max <= {MOE_TOPK_SLOTS_MAX}."
        )


def choose_k(
    router_logits: torch.Tensor,
    *,
    mode: str,
    k_min: int,
    k_max: int,
    entropy_lo: float,
    entropy_hi: float,
) -> torch.Tensor:
    """The per-token width — `LongTensor[…, ] matching `router_logits.shape[:-1]`.

    `router_logits` is `[n, E]` (a `[b, t, E]` tensor is accepted and flattened
    to `[b*t]`, the shape the task names). Pure: same logits + same knobs, same
    `k`, on any device, with no readback.
    """
    assert router_logits.ndim >= 2, (
        f"router_logits must carry the expert axis, got {tuple(router_logits.shape)}"
    )
    n_experts = router_logits.shape[-1]
    check_policy(mode, k_min, k_max)
    if k_max > n_experts:
        loud(
            f"moe_topk_max={k_max} > the router's {n_experts} experts: a width above "
            f"the bank cannot be filled. Escape: k_max <= n_experts."
        )
    if not entropy_lo < entropy_hi:
        loud(
            f"moe_entropy_lo={entropy_lo} >= moe_entropy_hi={entropy_hi}: the window "
            f"would be a step (or inverted), not a ramp. Escape: lo < hi "
            f"(defaults 0.2 < 0.6)."
        )
    flat = router_logits.reshape(-1, n_experts)
    levels = k_max - k_min
    if mode == "off":
        k = torch.full(flat.shape[:-1], k_max, dtype=torch.long, device=flat.device)
    elif mode == "entropy":
        h = _normalized_entropy(flat)
        t = ((h - entropy_lo) / (entropy_hi - entropy_lo)).clamp(0.0, 1.0)
        k = k_min + (t * levels).floor().to(torch.long)
    else:  # margin — checked above, so this is the only branch left
        if n_experts < 2:
            loud(
                f"mode='margin' needs a second expert to measure a gap, got "
                f"n_experts={n_experts}. Escape: mode='entropy' or 'off'."
            )
        top2 = flat.topk(2, dim=-1).values
        gap = top2[..., 0] - top2[..., 1]
        t = ((gap - entropy_lo) / (entropy_hi - entropy_lo)).clamp(0.0, 1.0)
        k = k_max - (t * levels).floor().to(torch.long)
    return k.reshape(router_logits.shape[:-1])


def select_topk(
    router_logits: torch.Tensor, k: torch.Tensor, k_max: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked top-k: `(index [n, k_max], active [n, k_max] bool)`.

    `index` is the top-`k_max` over `router_logits` with every entry past the
    row's own `k` forced to `-inf`; the first `k` columns are the real
    assignments in descending-score order. `active` marks those columns, so the
    caller zeroes the unused slots' WEIGHTS instead of letting a `-inf` logit
    reach a softmax or a gather (`softmax(-inf) = 0` is fine; `-inf * 0` is NaN,
    so the mask never travels into arithmetic).

    The precondition on `k` (1 <= k <= k_max) is `choose_k`'s contract; checking
    it here would be a device-to-host read in the hot path.
    """
    assert router_logits.ndim >= 2, (
        f"router_logits must carry the expert axis, got {tuple(router_logits.shape)}"
    )
    n_experts = router_logits.shape[-1]
    flat = router_logits.reshape(-1, n_experts)
    assert k.reshape(-1).numel() == flat.shape[0], (
        f"k has {k.reshape(-1).numel()} rows for {flat.shape[0]} tokens"
    )
    assert 1 <= k_max <= n_experts, f"k_max={k_max} outside [1, {n_experts}]"
    device = flat.device
    # `k` arrives as `[n]`; the slot mask is its row-wise prefix.
    active = torch.arange(k_max, device=device)[None, :] < k.reshape(-1, 1)
    # The candidate set: each row's own top-k_max. The mask keeps the prefix of
    # it (the real width) and `-inf`s the tail, so `topk(k_max)` cannot promote
    # an unused expert even when the tail values are large.
    rank = flat.topk(k_max, dim=-1).indices
    keep = torch.zeros(flat.shape, dtype=torch.bool, device=device)
    keep.scatter_(1, rank, active)
    masked = flat.masked_fill(~keep, float("-inf"))
    index = masked.topk(k_max, dim=-1).indices
    return index.reshape(*router_logits.shape[:-1], k_max), active.reshape(
        *router_logits.shape[:-1], k_max
    )


def _normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    """`H(softmax(z)) / log E`, clamped to `[0, 1]` (fp rounding at the ends).

    `torch.special.entr(p) = -p·log p` with `entr(0) = 0`, so a probability that
    underflowed to zero contributes nothing instead of a `0 * -inf` NaN. One
    expert has zero entropy by definition (`log 1 = 0`).
    """
    n = logits.shape[-1]
    p = torch.softmax(logits.float(), dim=-1)
    h = torch.special.entr(p).sum(-1)
    if n < 2:
        return torch.zeros_like(h)
    return (h / math.log(n)).clamp(0.0, 1.0)
