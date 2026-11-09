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
    """Retract `w` in place onto the Stiefel manifold along its RANK axis.

    Two layouts reach this function and both must work:

    * `u [d, r]` — the rank is the COLUMN axis; orthonormal columns are the
      retraction (`Q·sign(diag(R))`);
    * `v [r, f]` — the rank is the ROW axis (this repo stores `V` transposed
      against SCT's `V ∈ ℝ^{n×k}`); the retraction is the same one applied to
      `w.T`, i.e. orthonormal ROWS. Refusing here would make the flag
      unusable on half the factors; picking the wrong axis would silently
      orthonormalise across the feature dimension.

    The rule is therefore geometric, not nominal: retract along whichever axis
    is the SMALLER one (the rank), which is also the only axis that can be
    orthonormal when `r < min(d, f)`.
    """
    n, c = w.shape
    if c <= n:
        _retract_cols_(w)
    else:
        _retract_cols_(w.t().contiguous().view(c, n), write_back=w)


def _retract_cols_(view: torch.Tensor, write_back: torch.Tensor | None = None) -> None:
    """QR-retract the COLUMNS of `view`; copy back into `write_back` if given."""
    q, rr = torch.linalg.qr(view.float(), mode="reduced")
    # `sign(0) = 0` would zero a whole column; a zero pivot is a sign of its own,
    # and `1` keeps the factor's norm instead of annihilating it.
    sgn = torch.sign(torch.diagonal(rr)).clamp(min=-1.0, max=1.0)
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    out = (q * sgn.unsqueeze(0)).to(view.dtype)
    # `no_grad` because the caller passes a PARAMETER: an in-place write to a
    # leaf that requires grad is refused outside it, and the retraction is a
    # projection of the weights, not a differentiable op on them.
    with torch.no_grad():
        if write_back is None:
            view.copy_(out)
        else:
            write_back.copy_(out.t())


def orthonormality_error(w: torch.Tensor) -> float:
    """`‖WᵀW − I‖_max` on the RANK axis (rows when the tensor is wide).

    Both TSCT layouts are handled: a `[d, r]` factor is judged on its columns, a
    `[r, f]` factor on its rows — the same axis `stiefel_retract_` projects.
    """
    x = w.float() if w.shape[1] <= w.shape[0] else w.float().t()
    g = x.t() @ x
    eye = torch.eye(g.shape[0], dtype=g.dtype, device=g.device)
    return float((g - eye).abs().max())
