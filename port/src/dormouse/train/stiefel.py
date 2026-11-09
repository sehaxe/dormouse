"""Stiefel retraction for TSCT's spectral factors (arXiv:2604.00733, Eq. 5).

The paper's one mechanism this repo did not have: after every optimizer step the
factors `U` and `V` are retracted onto the Stiefel manifold,

    Q, R = QR(W);   W ← Q · sign(diag(R))            (SCT Eq. 5)

so the columns stay orthonormal. Two consequences that matter here:

* conditioning — a factor whose columns are orthonormal cannot drift into a
  near-degenerate basis, which is what keeps `diag(s)` interpretable as per-rank
  importance;
* the TERNARY projection stops being a random projection. A sign is a meaningful
  code for the direction of an orthonormal vector; on an arbitrary vector "the
  sign" carries no more information than the vector's own sparsity.

`sign(diag(R))` is the continuity correction: QR is unique only up to the sign of
each column, and without it the factors would flip sign on every step.

Cost: `O(n·r²)` per factor. At this repo's shapes (d=832, r=64) that is a
rounding error next to a step; the paper's `m × k` at 8192 × 32 pays 41-88 % of
its step for it [T, SCT Table 2], which is why this is a FLAG and not a default.

What is deliberately NOT here: no manifold optimizer (no Riemannian gradient, no
momentum transport). The paper uses plain AdamW followed by a projection, and a
projection is what this file is.
"""

from __future__ import annotations

import torch


def stiefel_retract_(w: torch.Tensor) -> None:
    """Retract `w [n, r]` in place to `Q · sign(diag(R))` (SCT Eq. 5).

    `r > n` (a rank larger than the ambient dimension) is a config lie: the
    columns cannot all be independent, so the call is refused rather than
    silently returning a rank-deficient factor.
    """
    n, r = w.shape
    if r > n:
        raise ValueError(f"stiefel_retract: rank {r} > rows {n}: the columns cannot be orthonormal")
    q, rr = torch.linalg.qr(w.float(), mode="reduced")
    # `sign(0) = 0` would zero a whole column; a zero pivot is a sign of its own,
    # and `1` keeps the factor's norm instead of annihilating it.
    sgn = torch.sign(torch.diagonal(rr)).clamp(min=-1.0, max=1.0)
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    w.copy_((q * sgn.unsqueeze(0)).to(w.dtype))


def orthonormality_error(w: torch.Tensor) -> float:
    """`‖WᵀW − I‖_max` — the quantity the retraction is supposed to hold."""
    g = w.float().t() @ w.float()
    eye = torch.eye(g.shape[0], dtype=g.dtype, device=g.device)
    return float((g - eye).abs().max())
