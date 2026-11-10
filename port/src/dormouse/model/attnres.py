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
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["FullDepthAttnRes", "full_depth_attend", "full_depth_weights"]


def _rms_no_gain(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _logits(
    sources: list[torch.Tensor], query: torch.Tensor, prior: torch.Tensor, eps: float
) -> torch.Tensor:
    """`[b, t, n+1]` logits: the zero source first, then the updates.

    `sources` is newest-first (`sources[0]` is the current layer's update), so
    `prior[0]` scores it and `prior[k]` scores the update `k` layers back.
    """
    n = len(sources)
    keys = torch.stack([_rms_no_gain(s.float(), eps) for s in sources], dim=-2)
    scores = (keys * query.float()).sum(-1)
    upd = scores + prior[:n].float()
    zero = upd.new_full(upd.shape[:-1] + (1,), math.log(n))
    return torch.cat([zero, upd], dim=-1)


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
    the softmax needs) is computed, never its product.
    """
    w = full_depth_weights(sources, query, prior, eps)
    vals = torch.stack([s.float() for s in sources], dim=-2)
    return (w[..., 1:].unsqueeze(-1) * vals).sum(-2)


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
