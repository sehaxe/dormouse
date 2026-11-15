"""Chunked cross-entropy: the `[T, V]` logits tensor never exists whole.

**Why this module exists.** The final CE reads the head's logits for every
position of every row: `b·t` positions by `V` vocabulary entries. At b24 ×
s1024 = 24,576 positions and a 32,768-token vocabulary that is **805M values** —
1.61 GB in bf16, 3.22 GB in fp32 — and the `log_softmax`/NLL backward keeps a
second copy of it, so the vocabulary alone costs 3.2-6.4 GB of peak. On a 16 GB
card where a step already uses 12-14 GB that is the difference between running
the tokenizer migration at b24 and falling back to b12. Nothing about the
ARITHMETIC needs those values to coexist: the loss is a mean over positions, so
it is a sum of independent per-position terms.

**The mechanism** (Liger-Kernel's `fused_linear_cross_entropy`, Apple's
Cut Cross-Entropy, arXiv:2411.09009 §3): for each chunk of positions, project
the hidden state to logits, compute the per-position NLL, accumulate the loss,
and free the chunk before the next one. The backward recomputes a chunk when
its gradient is needed (`torch.utils.checkpoint`, `use_reentrant=False`) instead
of storing it, so the peak is ONE chunk's logits rather than all of them. It is
exact, not an approximation: no logit is dropped, quantised, or shared, and the
gradient w.r.t. the hidden state and the head weight is the same number the
un-chunked graph produces (to fp32 summation order — see the gates in
`tests/test_chunk_ce.py`).

**Why not Liger.** An earlier attempt to use Liger's kernel in this repo failed
twice in a row: the loss read 40× off its own reference, and the run then died
with an illegal memory access. A dependency whose failure mode is "a plausible
number that is wrong" has to be re-verified faster than the arm it frees, so
this is pure PyTorch: nothing here is a kernel, everything is `F.linear` +
`F.cross_entropy` on a slice, and the gates are bit-level comparisons against
`F.cross_entropy` at three chunk sizes. Do not swap in a fused kernel for speed
without those comparisons passing.

**What it does NOT cover.** Only the final readout CE of the STACK forms
(`model.model._readout`). The loop's per-iteration CE (`loopblock._iter_ce`) and
the TST bag CE still take whole-logits tensors, and `use_byteflow` is refused
outright (its objective scores patch latents, not positions). `torch.utils.
checkpoint` is not capturable, so the arm is a plain eager step: a `full_graph`
capture of it has never been run.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .instrument import loud
from .losses import logit_softcap


def chunked_cross_entropy(
    hidden: torch.Tensor,
    project: Callable[[torch.Tensor], torch.Tensor],
    targets: torch.Tensor,
    chunk: int,
    *,
    mask: torch.Tensor | None = None,
    ignore_index: int = -100,
    softcap: float = 0.0,
) -> torch.Tensor:
    """Mean CE of `project(hidden)` against `targets`, one chunk at a time.

    `hidden [T, D]` (the readout's hidden states, already flattened over batch
    and position), `project` the head that maps `[n, D] → [n, V]` (any
    row-independent module: `nn.Linear`, `TiedHead`, a factorized head — it is
    called with SLICES, so it must not mix rows), `targets [T]` (or `[T, 1]`)
    the next-token ids, `chunk` the number of positions per slice (> 0).

    `mask [T]` mirrors `loopblock.masked_ce_mean`: the sum of
    `ce · mask / mask.sum()`, so a padded position contributes neither loss nor
    gradient. `ignore_index` is `F.cross_entropy`'s own (its rows are excluded
    from both the sum and the denominator) — `-100`, the torch convention, which
    no byte or token id can be. `softcap` composes the Gemma-2 final-logit cap
    (`losses.logit_softcap`) the way `_readout` applies it: INSIDE the chunk, on
    the logits the CE reads.

    The denominator is computed once (device-side, no host read): the count of
    un-ignored targets, or `mask.sum()`. A caller that can produce an empty
    selection gets a 0/0 NaN, exactly as `masked_ce_mean` and the `.mean()` path
    do — the trainer's NaN firewall owns that decision, not this module.
    """
    if chunk < 1:
        loud(
            f"ce_chunk={chunk}: the value is a COUNT of positions per chunk, so 0 "
            f"is not 'off' here (off is the caller's `F.cross_entropy` branch) and a "
            f"negative value would make `range` empty and the loss 0/0. Escape: "
            f"ce_chunk >= 1, or ce_chunk=0 to keep the un-chunked loss."
        )
    if hidden.ndim != 2:
        loud(
            f"chunked_cross_entropy hidden {tuple(hidden.shape)}: the function reads "
            f"`[T, D]` (positions already flattened over batch), because the chunk "
            f"boundary is a POSITION boundary. Escape: `hn.reshape(-1, d_model)`."
        )
    if not hidden.is_floating_point():
        loud(f"chunked_cross_entropy hidden dtype {hidden.dtype}: hidden must be floating.")
    n = hidden.shape[0]
    flat_t = targets.reshape(-1)
    if flat_t.numel() != n:
        loud(
            f"chunked_cross_entropy: {flat_t.numel()} targets for {n} positions. "
            f"Escape: one target per position."
        )
    flat_m = None if mask is None else mask.reshape(-1)
    if flat_m is not None and flat_m.numel() != n:
        loud(
            f"chunked_cross_entropy: mask {flat_m.numel()} entries for {n} positions. "
            f"Escape: one mask entry per position (`loss_mask.reshape(-1)`)."
        )

    if flat_m is None:
        denom = (flat_t != ignore_index).sum().to(torch.float32)
    else:
        denom = flat_m.to(torch.float32).sum()

    def terms(hh: torch.Tensor, tt: torch.Tensor, mm: torch.Tensor | None) -> torch.Tensor:
        z = project(hh)
        if softcap > 0.0:
            z = logit_softcap(z, softcap)
        ce = F.cross_entropy(z, tt, reduction="none", ignore_index=ignore_index)
        if mm is None:
            return ce.sum()
        return (ce * mm.to(ce.dtype)).sum()

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        part = checkpoint(
            terms,
            hidden[start:stop],
            flat_t[start:stop],
            None if flat_m is None else flat_m[start:stop],
            use_reentrant=False,
        )
        total = total + part
    return total / denom
