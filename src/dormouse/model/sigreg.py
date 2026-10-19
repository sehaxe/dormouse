"""SIGReg — the LeJEPA isotropic-Gaussian regulariser, and its strided input.

`docs/research/2026-10-16-jepa-into-dormouse.md` §2 names the family and §3 the
assembly this ports: `loss = CE + sigreg_weight · sigreg_loss(z)`, `z` the
strided sample of the state `lm_head` reads — **no teacher, no predictor, no
mask, no second forward, 0 parameters**. §3 is the ONE arm the document puts
first (the missing control): the burn recipe's `l1 + 0.1·koleo` bundled two
signals under one weight, so `d8062d1` cannot say which half produced 0.082 BPB,
and LeJEPA's own paper defines SIGReg without a teacher-student pair.

Transcription source: `vendor/dormouse-fused/crates/burn-jepa/src/losses.rs:33-50`
(`lejepa_loss`, itself the direct — non-sketched — form of arXiv:2511.08544).
The formula, whose every operation is that function's:

    L(z) = ‖mean(z)‖² + ‖cov(z) − I‖² / D

with `z_c = z − mean(z)` and `cov(z) = z_cᵀ z_c / (max(N, 2) − 1)` — burn's
`div_scalar((n.max(2) - 1) as f32)`. One deviation, shape only: the port returns
a 0-dim scalar rather than burn's `Tensor<1>` `unsqueeze()`.

Hand-checkable reference, `z = [[1, 2], [3, 4]]` (N=2, D=2), line by line:

    mean(z)    = [2, 3]                        -> ‖mean‖² = 4 + 9 = 13
    z_c        = [[-1, -1], [1, 1]]
    cov(z)     = z_cᵀ z_c / (max(2, 2) − 1)    = [[2, 2], [2, 2]]
    cov(z) − I = [[1, 2], [2, 1]]              -> Σ² = 1 + 4 + 4 + 1 = 10
    cov term   = 10 / D = 5
    L          = 13 + 5 = 18.0

`tests/test_sigreg.py` recomputes exactly that and asserts 1e-6.

Why the sample is small and strided: the covariance is O(S·D²), so the statistic
is read over `S = 256` rows strided across the batch's positions (burn's KoLeo
subsample count, `losses.rs:66`, and the same strided pick, `losses.rs:67-77`) —
the document's "3 tensor ops (mean, cov, eye), cheaper than KoLeo's O(S²)
pairwise distances" (§2). The stride is deterministic: burn's global RNG is never
seeded, so a random subset would make two runs of one config a different
objective (`losses.rs:53-57`, the KoLeo twin of that argument).

There is NO KoLeo here, no EMA teacher and no detach: the gradient of this term
reaches the backbone (LeJEPA's "no stop-gradient", §2), which is the whole point
of it as a regulariser.
"""

from __future__ import annotations

import math

import torch

#: Rows the covariance is read over — burn's KoLeo subsample count
#: (`losses.rs:66`), kept as the SIGReg sample because it is the same O(S·D²)
#: argument. At the tiny test shapes `b·t` is below it, and the whole batch is
#: used (`stride_sample`).
SIGREG_ROWS = 256


def stride_sample(z: torch.Tensor, rows: int = SIGREG_ROWS) -> torch.Tensor:
    """`[n, d]` -> `[min(n, rows), d]`, every `ceil(n / rows)`-th row.

    Deterministic, no RNG (`losses.rs:67-77`'s strided pick, re-expressed with
    `index_select` instead of burn's materialized `gather` indices). A short
    tensor is returned whole: there is nothing to subsample and upsampling a row
    would weight it twice in the covariance.
    """
    assert rows >= 2, f"a covariance needs >= 2 rows, got {rows}"
    n = z.shape[0]
    if n <= rows:
        return z
    idx = torch.arange(0, n, math.ceil(n / rows), device=z.device)
    return z.index_select(0, idx)


def sigreg_loss(z: torch.Tensor) -> torch.Tensor:
    """`L = ‖mean(z)‖² + ‖cov(z) − I‖² / d` — 0-dim, differentiable, no detach.

    `z [n, d]` is the sample the regulariser reads (`stride_sample`'s output of
    the fp32 state `lm_head` reads). Both terms are the transcription source's
    (`losses.rs:33-50`); the caller owns the weight.
    """
    assert z.ndim == 2, f"sigreg reads [n, d], got {tuple(z.shape)}"
    n, d = z.shape
    assert n >= 2, f"a covariance needs >= 2 rows, got {n}"
    mean = z.mean(dim=0)
    z_c = z - mean
    # max(n, 2) - 1 is burn's own denominator: it keeps the n <= 1 case from
    # dividing by zero rather than producing a NaN covariance (`losses.rs:46`).
    cov = z_c.transpose(0, 1) @ z_c / float(max(n, 2) - 1)
    eye = torch.eye(d, dtype=z.dtype, device=z.device)
    return mean.pow(2).sum() + (cov - eye).pow(2).sum() / d
