"""The Muon step must have the SAME per-element RMS on every matrix shape.

Why this file exists (2026-11-07): `colrow_normalize` leaves every row of the
polar update at norm 1, so the applied step's RMS is `lr · scale / sqrt(cols)`.
With the historical `scale = sqrt(max(1, rows/cols))` that put `[832,3072]` at
5.4e-6 against AdamW's 3.0e-4 — a 55x deficit — while `[3072,832]` was off by
only 15x. The factorised bank's `v [64,2730]` was therefore punished 3.5x harder
than a dense matrix of the same model, which contaminated every bank-vs-dense
A/B on record (`docs/records/reviews/2026-11-07-muon-audit.md` §D2, §2.2).

The gate is a measurement, not a formula: one real optimizer step on each shape,
then `RMS(dW) / lr` read off the parameter delta. `shape_fair` must land inside
[0.9, 1.1] on all of them; `uniform` must reproduce the audit's spread (which is
how we know the gate can fail).

CPU only: the optimizer is device-agnostic, and these are 2-D matrices.
"""

from __future__ import annotations

import torch

from dormouse.train.optim import MixOptimizer

SHAPES = [(832, 832), (3072, 832), (2496, 832), (832, 3072), (64, 2730)]


class _Model(torch.nn.Module):
    """The optimizer takes a module and splits its parameters by shape: 2-D goes
    to Muon, everything else to AdamW. One 2-D parameter is the whole model here."""

    def __init__(self, params):
        super().__init__()
        for i, p in enumerate(params):
            self.register_parameter(f"w{i}", p)
        # One 1-D parameter so the AdamW half has something: MixOptimizer splits
        # by rank, and torch refuses an empty parameter list.
        self.register_parameter("bias", torch.nn.Parameter(torch.zeros(4)))


LR = 3e-4


def _one_step(rows: int, cols: int, mode: str) -> float:
    """RMS of the parameter delta after ONE step, divided by the LR."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(rows, cols) * 0.02)
    opt = MixOptimizer(
        _Model([p]),
        lr=LR,
        wd=0.0,
        opt="mix",
        muon_lr_mode=mode,
        muon_lr_mult=1.0,
    )
    before = p.detach().clone()
    p.grad = torch.randn(rows, cols)
    opt.step()
    delta = (p.detach() - before).float()
    # Per-element RMS, not the Frobenius norm: the claim is about the step a
    # single weight receives, and every row carries the same norm by design.
    return float(delta.pow(2).mean().sqrt()) / LR


def test_shape_fair_has_one_step_scale_on_every_shape():
    for rows, cols in SHAPES:
        ratio = _one_step(rows, cols, "shape_fair")
        assert 0.9 <= ratio <= 1.1, (
            f"[{rows},{cols}] shape_fair RMS(dW)/lr = {ratio:.3e}, outside [0.9, 1.1]. "
            f"The Muon step is not shape-independent, so an arm with a different "
            f"matrix mix trains at a different effective LR."
        )


def test_uniform_reproduces_the_measured_deficit():
    """The pre-fix mode is kept only to reproduce old runs — it must still show
    the spread the audit measured, otherwise the audit and this test disagree
    about what the old code did."""
    ratios = {shape: _one_step(*shape, "uniform") for shape in SHAPES}
    # The audit's measured deficits at lr=3e-4 (optim.py's old formula), as
    # RMS(dW)/lr = 1/deficit: 28.8x on [832,832], 15.0x on [3072,832],
    # 55.4x on [832,3072] and 52.3x on the bank's v [64,2730].
    dense, bank = ratios[(3072, 832)], ratios[(64, 2730)]
    assert 0.05 < dense < 0.09, f"uniform on [3072,832] = {dense:.3e}, the audit measured ~0.067"
    assert 0.012 < bank < 0.028, f"uniform on [64,2730] = {bank:.3e}, the audit measured ~0.019"
    assert dense / bank > 3.0, (
        "the audit's key finding is that the deficit was NOT uniform across shapes: "
        f"the bank's v got 52.3x, a dense matrix 15.0x; got {dense / bank:.2f}x"
    )


def test_shape_fair_ratio_is_flat_where_uniform_is_not():
    """One assertion that states the fix in a single line: the spread across
    shapes collapses from >3x to ~1x."""
    fair = [_one_step(r, c, "shape_fair") for r, c in SHAPES]
    uni = [_one_step(r, c, "uniform") for r, c in SHAPES]
    assert max(fair) / min(fair) < 1.25, f"shape_fair spread is {max(fair) / min(fair):.2f}x"
    assert max(uni) / min(uni) > 3.0, f"uniform spread is {max(uni) / min(uni):.2f}x"


def test_unknown_mode_refuses_loudly():
    p = torch.nn.Parameter(torch.randn(8, 8))
    try:
        MixOptimizer(_Model([p]), lr=1e-3, wd=0.0, muon_lr_mode="fai")
    except Exception as exc:  # the repo's loud() raises
        assert "shape_fair" in str(exc) and "uniform" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an unknown muon_lr_mode must refuse, not silently default")
