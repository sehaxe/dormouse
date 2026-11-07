"""The Muon half against Keller Jordan's `muon.py`, and the LR the trainer hands it.

Two claims are pinned here, and they are different claims:

1. **The transcription is right.** `MixOptimizer(opt="mix")` on a 2-D parameter
   produces the direction the reference produces (`colrow_normalize(zeropower)`
   vs `zeropower`, cosine > 0.99 on a full-rank gradient) and the update
   magnitude is within 2x at the same `lr`. This test PASSES on the code as it
   stood on 2026-11-07 — the math was never the defect, and a future fix must
   not break it while moving the LR.

2. **The trainer delivers it.** `train/loop.py:1760-1764` (and
   `train/sft.py:1215-1218`) wrote `group["lr"] = lr` into EVERY Muon group on
   EVERY step, before the first optimizer step, so `MixOptimizer.__init__`'s
   `lr * muon_lr_mult` (`optim.py:1798`) is overwritten before it is ever read.
   Measured on CPU: with that write, `muon_lr_mult=8` and `1` produce a
   **bit-identical** parameter delta — a 3000-step A/B of the knob measured the
   knob off. Red here, green once the schedule writes through
   `MixOptimizer.set_lr`.

The reference is a transcription of `KellerJordan/muon@master:muon.py`
(fetched 2026-11-07, 286 lines): Nesterov momentum, the quintic Jordan triple,
`X /= X.norm() + 1e-7`, `update *= max(1, A/B)**0.5`, `p.add_(alpha=-lr)`. That
file carries no `0.2` factor; the older `0.2*sqrt(max(A,B))` convention is
checked as a second magnitude column in the audit doc.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from dormouse.train.optim import JORDAN_NS_COEFFS, MixOptimizer

LR = 3e-4
SHAPES = ((256, 256), (768, 3072), (3072, 768))
TRAINER = Path(__file__).resolve().parents[1] / "src" / "dormouse" / "train"


# ------------------------------------------------------------------ reference --
def zeropower(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """`muon.py::zeropower_via_newtonschulz5`, fp32 (the reference runs bf16)."""
    a, b, c = JORDAN_NS_COEFFS
    x = g.float()
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.mT
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.mT if tall else x


def reference_step(g: torch.Tensor, lr: float = LR) -> torch.Tensor:
    """`muon_update` + `p.add_(update, alpha=-lr)`, momentum started at zero."""
    mom = torch.zeros_like(g).lerp(g, 1 - 0.95)  # one `muon_update` momentum step
    update = g.lerp(mom, 0.95)  # Nesterov
    update = zeropower(update)
    update = update * max(1, update.size(-2) / update.size(-1)) ** 0.5
    return -lr * update


class Holder(nn.Module):
    """A model with one Muon-routed 2-D parameter and one AdamW-routed vector."""

    def __init__(self, shape: tuple[int, int]):
        super().__init__()
        self.w = nn.Parameter(torch.empty(shape))
        self.bias = nn.Parameter(torch.zeros(5))


def sealed(shape: tuple[int, int], mult: float = 1.0, mode: str = "uniform"):
    """(model, w0, g, optimizer) — routing asserted, so a rename cannot move it."""
    torch.manual_seed(0)
    model = Holder(shape)
    model.w.data.normal_(0, 0.02)
    w0 = model.w.detach().clone()
    g = torch.randn(shape)
    model.w.grad = g.clone()
    opt = MixOptimizer(model, lr=LR, wd=0.1, opt="mix", muon_lr_mult=mult, muon_lr_mode=mode)
    assert opt.routing["w"] == "muon", opt.routing
    return model, w0, g, opt


def colrow_unit_rms(shape: tuple[int, int]) -> float:
    """Per-element RMS of `colrow_normalize(X)` for ANY full-rank X — the step's unit.

    ColRow leaves every row at norm 1, so `||X||_F = sqrt(rows)` for every input
    that has no all-zero row: the Muon step's size is a constant of the SHAPE,
    not of the gradient. That is why `lr` is the only lever there is.
    """
    rows, cols = shape
    return math.sqrt(max(1.0, rows / cols)) / math.sqrt(cols)


def step_delta(shape, mult, lr_write=None):
    model, w0, g, opt = sealed(shape, mult)
    if lr_write is not None:
        opt.set_lr(lr_write)
    opt.step()
    return model.w.detach() - w0, g


# ---------------------------------------------------------------------- tests --
def test_muon_matches_the_reference_transcription() -> None:
    """Same direction, same magnitude at the same LR — the math is the reference.

    Pinned in `uniform` mode: the historical `sqrt(max(1, rows/cols))` scale is
    what this reference transcription was calibrated against. `shape_fair` is a
    deliberate magnitude change (see `TrainConfig.muon_lr_mode`), so it gets its
    own test below rather than loosening this bound.
    """
    for shape in SHAPES:
        ours, g = step_delta(shape, 1.0)
        ref = reference_step(g)
        cos = torch.nn.functional.cosine_similarity(
            ours.flatten(), ref.flatten(), dim=0
        ).item()
        rms_ratio = ours.pow(2).mean().sqrt().item() / ref.pow(2).mean().sqrt().item()
        assert cos > 0.99, f"{shape}: cosine {cos:.4f} — the transcription moved"
        # ColRow equalises ROW norms, so a tall matrix ends up sqrt(rows/cols)
        # hotter than the bare polar (2.1x at [3072,768]) — that is the
        # documented Muon+ arm (`colrow_normalize`, A/B'd -0.0428 BPB), not a
        # transcription error. A 5x error still fails this bound.
        assert 0.5 < rms_ratio < 2.5, f"{shape}: RMS ratio {rms_ratio:.3f} vs the reference"


def test_the_trainer_delivers_the_muon_lr() -> None:
    """A schedule write must not erase `muon_lr_mult`, and the half must not be frozen.

    `MixOptimizer.set_lr` is the one place a schedule writes a group's LR; it is
    what `train/loop.py` and `train/sft.py` call. Before it existed the write was
    inline, and the multiplier was dead.
    """
    for shape in SHAPES:
        d1, g = step_delta(shape, 1.0, lr_write=LR)
        d8, _ = step_delta(shape, 8.0, lr_write=LR)
        assert not torch.equal(d1, d8), f"{shape}: muon_lr_mult=8 is inert"
        ratio = d8.pow(2).mean().sqrt().item() / d1.pow(2).mean().sqrt().item()
        assert 7.0 < ratio < 9.0, f"{shape}: delivered step scaled {ratio:.3f}x, not 8x"

        # At the shape's own deficit the Muon half stops being a rounding error
        # next to the AdamW half at the SAME nominal LR.
        mult = 1.0 / colrow_unit_rms(shape)
        d_muon, _ = step_delta(shape, mult, lr_write=LR)
        d_adam = -LR * g.sign()  # AdamW's first step is `lr * sign(g)` exactly
        got = -(g * d_muon).sum().item()
        want = -(g * d_adam).sum().item()
        assert got / want > 0.25, (
            f"{shape}: Muon buys {got / want:.2%} of AdamW's first-order decrease "
            f"at mult={mult:.1f} — the LR is still not delivered"
        )


def test_no_trainer_writes_group_lr_behind_the_optimizer() -> None:
    """The 2026-11-07 defect as an invariant: the schedule has ONE seam."""
    import re

    # The defect is the assignment of the SCHEDULE's `lr` itself; the SF arm's
    # write-back (`= muon_lr * mult`, capture-refused and eager) is a different
    # quantity and is allowed to stay a host float.
    write = re.compile(r'^\s*group\["lr"\]\s*=\s*lr\s*$', re.MULTILINE)
    for name in ("loop.py", "sft.py"):
        src = (TRAINER / name).read_text()
        assert not write.search(src), (
            f"{name} writes a param group's LR behind MixOptimizer.set_lr; the "
            f"Muon half's lr*muon_lr_mult is silently replaced by the schedule "
            f"(the first offender is line "
            f"{src[: write.search(src).start()].count(chr(10)) + 1})"
        )
        assert "set_lr(" in src, f"{name} never calls the LR seam"
