"""Full depth AttnRes (N = L) over the `block_form="transformer"` residual write.

WHAT THIS IS. Attention Residuals (arXiv:2603.15031) replace a layer's residual
write with a softmax mixture over the sources below it. `N` is the number of
sources: `N = 8` blocks recovers most of the ablation's benefit at the author's
93-layer depth, `N = L` is the FULL variant (one layer per source, no blocks),
and `N = 1` degenerates to a plain residual. At our L = 12 the full variant has
12 sources — above the saturation knee — and costs
`L·(L+1)/2 · O(d)` against the block's `~2·d²` per layer: 0.019 % of a step's
MACs and 0.011 % of the parameters. So this builds N = L, not blocks.

THE MIXTURE, AND WHY THE ZERO SOURCE CARRIES `log n`. One pseudo-query
`q_i ∈ R^d` per layer. At layer `i` (1-based) the sources are the "no update"
zero and the updates of every layer so far,

    u_0 = 0,   u_1, …, u_i          (u_i = THIS layer's body output)

and the write is `x_{i+1} = x_i + a` with

    s_j = q_i · RMSNorm(u_j)                        (the paper's Eq. 2 score)
    a   = Σ_j w_j u_j,   w = S · softmax(logits),   S = i + 1
    logits_j = prior[dist(j)] + s_j

The `S·softmax` convention is the house rule (`loopblock.depth_attend`): it sums
to `S` by construction, and at the uniform query it makes every weight `1`, so
the mixture is the plain SUM of the sources. For the loop arm that sum IS the
plain residual (the two sources are `0` and the body), which is why its zero-query
init is the identity bit for bit.

The full arm has i update sources, not one, and the residual stream already
carries layers `1..i-1`; only the CURRENT update must be added for the plain
residual. Under `S·softmax` that means `w_i = 1` while the other `S-1` units of
weight sit on a zero value. The zero source does exactly that job: its value is
`u_0 = 0`, so whatever weight it carries contributes nothing, and giving it the
logit `log n` (with `n = i` update sources) makes the softmax denominator exactly
`n + 1 = S` at step 0:

    softmax(u_0) = n/S,  softmax(u_i) = 1/S,  the older updates ≈ exp(-30)/S
    w_0 = S·n/S = n (× 0),  w_i = S·1/S = 1,  a = u_i

So the arm reproduces the plain residual at init (the A/B measures the variant,
not a perturbation), `Σ_j w_j == S` still holds (the house convention), and the
learnable prior can move the mass afterwards. The prior is one vector per
relative distance (`prior[0]` = the current update, `prior[1]` = the previous
one, …), initialised `(0, −30, −30, …)` so only the last update is active at
step 0. The zero source's `log n` is an anchor, not a learned parameter: it is
what makes the identity exact, and it grows with the source count by construction.

The stores are fp32 (the residual dtype) so the score reduction and the softmax
never run in bf16 — the same rule `loopblock.depth_attend` follows.

WHAT THE BACKWARD HOLDS, AND WHY NOTHING IS HELD. The loop made the FORWARD
flat and left the training step untouched: every source requires grad, so each
`.float()` upcast and each RMSNorm output was kept for the backward. Measured
with `saved_tensors_hooks`, unique storages, at the live shape
(T = 24576, d = 832, S = 13 — FWD and FWD+BWD are the same number, all of it is
taken in the forward):

    sources   path   FWD+BWD   the mixer's own   largest own tensor
    fp32      old    2,031.8       1,017.8 MiB   20,447,232 elems  = [T, d]
    fp32      new    1,016.6           2.6 MiB      344,064 elems  = [T, S+1]
    bf16      old    3,045.8       3,045.8 MiB   20,447,232 elems
    bf16      new      509.6           2.6 MiB      344,064 elems

What is left of those totals is the source history itself: 13 tensors resident
whatever the mixer does with them, because they ARE the residual stream. The
mixer's own copy is what OOMed the arm at b24/b20/b16 (bf16 line), and it is
the copy a `@torch.no_grad()` memory test structurally cannot see: in training
the upcasts and the norm outputs are 78 MiB each and there are three per
source.

Two changes, decided by that measurement (option (a) "save the sources packed"
collapses into (b) here: the sources are ALREADY bf16 and resident, so there is
nothing to pack — the fp32 copies exist only for the backward):

1. the score column `q·RMSNorm(u_j) + prior_j` is a `torch.utils.checkpoint`
   SEGMENT. Its inputs are the bf16 source, the query and one scalar, all
   already alive; the fp32 upcast and the RMSNorm output are RECOMPUTED in the
   backward, one source at a time, and dropped before the next. Doubling this
   mixer's arithmetic costs 0.019 % of a step's MACs.
2. the value sum multiplies the source in ITS dtype and lets promotion produce
   fp32 (`w` is fp32, so `w[..., j+1] * u` is fp32 — bit for bit the same as
   `w * u.float()`, whose fp32 copy autograd would otherwise save).

The result is bit-identical — forward, and every gradient through a 4-layer
`use_attnres` model in fp32 AND bf16 (`max|Δ| = 0.0`). The gate is
`test_attnres_full.py`'s `test_the_training_step_retains_no_t_d_residency`,
which builds the old expression in the test and runs it with gradients on.

WHY THE LOOP. The first cut stacked every RMSNorm'd source into one
`[b, t, n, d]` fp32 key tensor, contracted it against the query in one
broadcast, and did the same to the values. At the live shape (b24×1024 → 24576
tokens, d = 832, n = 12) that buffer is 936 MiB and its product with the query
another 936 MiB — 1.87 GiB live at the score step, and again at the value step
— against a 468 MiB bf16 history. Measured: the call allocates 7.58 GiB of
churn and its largest single block is 936 MiB. The arm OOMed at b24, b20 AND
b16 because of it. The loop below touches one `[T, d]` at a time: the scores
are `T·(n+1)·4` bytes (1,277,952 B = 1.22 MiB at S = 13) and the weighted sum
is one `[T, d]` fp32 accumulator. Same softmax over the same `n+1` scalars,
same order of accumulation: the result is not merely comparable to the stack,
the test finds it EQUAL (`max|Δ| = 0.0`, gated at ≤1e-6 against the old
expression written out in `test_attnres_full.py`). The new binding tensor is
the fp32 `[T, d]` RMSNorm transient — 78.0 MiB, ≤3 live at one instant, 234 MiB
— plus the 78 MiB accumulator; measured largest single block 78.0 MiB. That is
the FORWARD; the backward holds one of those transients at a time and nothing
permanent (…WHAT THE BACKWARD HOLDS).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = ["FullDepthAttnRes", "full_depth_attend", "full_depth_weights"]


def _rms_no_gain(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _score_column(
    s: torch.Tensor, q: torch.Tensor, prior_j: torch.Tensor, eps: float
) -> torch.Tensor:
    """One source's `q·RMSNorm(u_j) + prior_j` — a checkpoint SEGMENT.

    Called through `checkpoint`, so the fp32 `[T, d]` upcast and the RMSNorm
    output are recomputed in the backward, one source at a time, instead of
    being held for it (module docstring, WHAT THE BACKWARD HOLDS).
    """
    return (_rms_no_gain(s.float(), eps) * q).sum(-1) + prior_j


def _logits(
    sources: list[torch.Tensor], query: torch.Tensor, prior: torch.Tensor, eps: float
) -> torch.Tensor:
    """`[b, t, n+1]` fp32 logits: the zero source first, then the updates.

    `sources` is newest-first (`sources[0]` is the current layer's update), so
    `prior[0]` scores it and `prior[k]` scores the update `k` layers back.

    One `[b, t]` column per source, written into the `[b, t, n+1]` matrix: no
    `[b, t, n, d]` key tensor ever exists (module docstring, WHY THE LOOP), and
    no `[b, t, d]` one survives the forward either (…WHAT THE BACKWARD HOLDS).
    """
    n = len(sources)
    q = query.float()
    logits = sources[0].new_empty(sources[0].shape[:-1] + (n + 1,), dtype=torch.float32)
    logits[..., 0] = math.log(n)
    for j, s in enumerate(sources):
        logits[..., j + 1] = checkpoint(
            _score_column, s, q, prior[j].float(), eps, use_reentrant=False
        )
    return logits


def full_depth_weights(
    sources: list[torch.Tensor], query: torch.Tensor, prior: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """The `S·softmax(·)` weights `[b, t, n+1]` (`Σ_j w_j == S`, zero first)."""
    n = len(sources)
    return (n + 1) * F.softmax(_logits(sources, query, prior, eps), dim=-1)


def full_depth_attend(
    sources: list[torch.Tensor], query: torch.Tensor, prior: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """`a = Σ_j w_j u_j` for sources ordered newest-first (`u_0 = 0` first).

    The zero source is implicit: its value is zero, so only its weight (which
    the softmax needs) is computed, never its product. The sum accumulates in
    place, one `[b, t]` weight column against one `[T, d]` source, so neither
    the `[b, t, n, d]` value stack nor its `[b, t, n, d]` product is built.

    The multiply is `w * u`, NOT `w * u.float()`: promotion gives the same fp32
    product bit for bit while leaving the fp32 copy out of the saved set.
    """
    w = full_depth_weights(sources, query, prior, eps)
    a = sources[0].new_zeros(sources[0].shape, dtype=torch.float32)
    for j, u in enumerate(sources):
        a += w[..., j + 1].unsqueeze(-1) * u
    return a


class FullDepthAttnRes(nn.Module):
    """One pseudo-query per layer + the shared per-relative-distance log-prior."""

    def __init__(
        self, d: int, n_layers: int, *, eps: float = 1e-6, device: torch.device | str = "cpu"
    ) -> None:
        super().__init__()
        self.d = d
        self.eps = float(eps)
        #: Zero-init: the query contributes nothing at step 0 and the prior alone
        #: decides the weights (the paper's §5 rule, uniform in the loop arm).
        self.query = nn.Parameter(torch.zeros(n_layers, d, device=device))
        #: `prior[0]` = the current update, `prior[1]` = one layer back, …
        #: `(0, −30, −30, …)` puts the whole update on the most recent source.
        prior = torch.full((n_layers,), -30.0, device=device)
        prior[0] = 0.0
        self.prior = nn.Parameter(prior)

    def forward(self, sources: list[torch.Tensor], layer_idx: int) -> torch.Tensor:
        return full_depth_attend(sources, self.query[layer_idx], self.prior, self.eps)
