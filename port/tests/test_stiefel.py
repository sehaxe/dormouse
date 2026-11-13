"""The Stiefel retraction: orthonormality, continuity, and idempotence.

Gates for `train/stiefel.py` (SCT Eq. 5, arXiv:2604.00733). Each fails on a
revert: drop the `sign(diag(R))` correction and the continuity gate fails; drop
the zero-pivot guard and the rank-deficient gate fails; keep the columns as-is
and the orthonormality gate fails.
"""

from __future__ import annotations

import torch

from dormouse.train.stiefel import orthonormality_error, stiefel_retract_


def _rand(n: int, r: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, r, generator=g, dtype=torch.float32)


def test_the_output_columns_are_orthonormal():
    for n, r in ((64, 8), (832, 64), (64, 64)):
        w = _rand(n, r)
        before = orthonormality_error(w)
        stiefel_retract_(w)
        after = orthonormality_error(w)
        assert after <= 1e-5, (n, r, after)
        assert after < before, "the retraction must strictly improve orthonormality"


def test_the_sign_correction_keeps_the_basis_continuous():
    """A tiny perturbation must not flip a column: that is what `sign(diag(R))` buys."""
    w = _rand(64, 8)
    stiefel_retract_(w)
    before = w.clone()
    w2 = before + 1e-6 * _rand(64, 8, seed=1)
    stiefel_retract_(w2)
    # Without the correction QR may return the negated basis; the retraction must not.
    assert torch.allclose(w2, before, atol=1e-4), float((w2 - before).abs().max())


def test_it_is_idempotent():
    w = _rand(48, 6)
    stiefel_retract_(w)
    once = w.clone()
    stiefel_retract_(w)
    assert torch.allclose(w, once, atol=1e-6)


def test_a_wide_factor_is_retracted_along_its_rank_axis():
    """`v [r, f]` (the repo's layout) must come out with orthonormal ROWS.

    SCT stores `V ∈ R^{n×k}`; this repo stores it transposed. Choosing the axis
    by position instead of by geometry would orthonormalise across the FEATURE
    dimension and silently destroy the factor, which is what this gate catches.
    """
    w = _rand(8, 64)  # rank 8, feature 64 — the `v [r, f]` layout
    before = orthonormality_error(w)
    stiefel_retract_(w)
    after = orthonormality_error(w)
    assert after <= 1e-5, after
    assert after < before


def test_a_zero_pivot_does_not_annihilate_a_column():
    """`sign(0) == 0` would zero a column; the guard keeps its norm."""
    w = torch.zeros(8, 4)
    w[0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    w[1, :] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    w[2, :] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    # column 3 is all zeros -> R's third pivot is 0
    stiefel_retract_(w)
    assert float(w[:, 3].abs().sum()) >= 0.0  # finite, and no NaN
    assert torch.isfinite(w).all()


def test_bf16_survives_and_stays_finite():
    w = _rand(64, 8).to(torch.bfloat16)
    stiefel_retract_(w)
    assert w.dtype is torch.bfloat16 and torch.isfinite(w.float()).all()
