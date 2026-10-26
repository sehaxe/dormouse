"""The HALT HEAD: a learned inference-time stop decision on the loop's readout.

`DormouseModel` runs a weight-shared loop. `max_iter` is the depth the loop
*can* run and `rand_depth` (train/loop.py) samples the depth a training step
*runs*; until this file, inference always ran the full depth. The halt head is
the missing half — at inference it answers "will one more iteration improve
*this* input?" and stops when it will not.

## The target: the model's OWN per-iteration CE, never the labels

For iteration `i` of a training step that ran `T` iterations:

    stop_i = 1  iff  CE_i − CE_{i+1} < halt_eps      (0 <= i < T-1)
    stop_i = 1  iff  i is the model's LAST iteration (i+1 == max_iter)
    stop_i = −   (excluded) when  i == T-1 < max_iter-1   (rand_depth)

`CE_i` is the loop's own per-iteration cross-entropy — the same tensor the eval
line prints as `periter=` (`LoopBlock._per_iter_ce`, produced at the point where
`lm_head` scores that iteration's readout). The head is trained with BCE over
the iterations the step actually ran, at weight `halt_weight`, on the DETACHED
readout state (so the term trains the head, not the backbone's representations).

Why the model's own CE and not a separate predictor:

* it is exactly the quantity a halt predictor should know — "is more thinking
  helping *this* input" — measured by the model itself;
* it reuses the seam that already exists (`_per_iter_ce`), so the target cannot
  drift from the loss the trainer computes, and no second forward is paid;
* **it never reads `targets` as the head's INPUT.** The DSpark D1 lesson
  (`docs/reviews/verify-tails-2026-09-30.md`) is that feeding the label
  sequence to an aux head hands it the answer one step early. Here the input is
  `step_out` (the loop's readout at iteration `i`, a function of `input_ids`
  only) — verified by `tests/test_halt.py::test_halt_input_does_not_see_targets`
  — and `targets` enters only by scoring that same readout, i.e. the same way
  the main CE does.

The exclusion in the third line is the `rand_depth` case: a step truncated at
`T < max_iter` has no `CE_{T}` to compare its last iteration against, so that
iteration is left out of the BCE rather than labelled "stop" from a depth the
RNG chose. The cap's last iteration is the only certain stop.

## The module

`Linear(d, d//4) → SiLU → Linear(d//4, 1)`: ~`d²/4` parameters, built only when
`halt_weight > 0`, so a weight of 0 adds nothing to the tree or the step. The
forward returns ONE scalar stop logit per iteration (the head's per-position
logits are averaged), which is exactly what the BCE target and the 0.5 decision
consume.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class HaltHead(nn.Module):
    """`[..., d]` readout state → ONE scalar stop logit for the iteration."""

    def __init__(
        self, d_model: int, *, hidden: int | None = None, device: torch.device | str = "cpu"
    ) -> None:
        super().__init__()
        assert d_model >= 1, f"d_model must be >= 1, got {d_model}"
        h = max(1, d_model // 4) if hidden is None else hidden
        self.down = nn.Linear(d_model, h, device=device)
        self.up = nn.Linear(h, 1, device=device)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """`[b, t, d]` (or one `[d]`) → ONE scalar stop logit for this iteration.

        The mean over positions matches the per-iteration CE target, which is a
        mean too, so the head scores the same aggregate the target names.
        """
        return self.up(F.silu(self.down(h))).mean()


def stop_prob(head: nn.Module, step_out: torch.Tensor) -> torch.Tensor:
    """One scalar stop probability per iteration, on the DEVICE.

    The loop's only decision input: `sigmoid(mean(head(step_out)))`, where
    `step_out` is the readout tensor the same iteration's CE was computed from.
    The mean over `[b, t]` matches the per-iteration CE target, which is also a
    mean, so the head scores the same aggregate the target named.
    """
    return torch.sigmoid(head(step_out))


def halt_targets(
    per_iter: list[torch.Tensor], eps: float, max_iter: int
) -> list[torch.Tensor | None]:
    """The BCE targets from the model's own per-iteration CE (module docstring).

    `per_iter[i]` is iteration `i`'s mean CE (a 0-dim tensor, gradient alive on
    the model's side; the target itself is detached). `max_iter` is the loop's
    CAP, not the depth this step ran — "last iteration" means the model cannot
    think further, not that the depth sampler truncated the window.
    """
    assert per_iter, "halt targets need at least one per-iteration CE"
    out: list[torch.Tensor | None] = []
    for i, ce in enumerate(per_iter):
        if i + 1 < len(per_iter):
            improve = (ce - per_iter[i + 1]).detach()
            out.append((improve < eps).to(ce.dtype))
        elif len(per_iter) == max_iter:
            out.append(ce.new_ones(()))
        else:
            out.append(None)
    return out


def halt_loss(
    head: nn.Module,
    hidden: list[torch.Tensor],
    per_iter: list[torch.Tensor],
    eps: float,
    max_iter: int,
) -> torch.Tensor | None:
    """Mean BCE over the iterations with a target, or `None` when none has one.

    `hidden[i]` is the readout `step_out` of iteration `i` (the tensor the same
    iteration's CE scores). It is DETACHED before the head: the halt term is the
    head's objective, so it must not reshape the backbone's states to satisfy a
    stopping rule. `None` happens only on a `rand_depth` step that ran one
    iteration with `max_iter > 1` — a step with no evidence either way.
    """
    assert len(hidden) == len(per_iter), f"{len(hidden)} states for {len(per_iter)} CEs"
    terms = [
        F.binary_cross_entropy_with_logits(head(h.detach()), target)
        for h, target in zip(hidden, halt_targets(per_iter, eps, max_iter), strict=True)
        if target is not None
    ]
    if not terms:
        return None
    loss = torch.stack(terms).mean()
    # No finiteness assert here: `bool(tensor)` is a host read, and the
    # trainer's DEVICE firewall (`train/loop.py`) already masks a non-finite
    # loss and gates the update without one (ADR-0018 rule 2).
    return loss
