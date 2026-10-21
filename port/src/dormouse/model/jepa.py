"""JEPA as ONE mechanism: a causal latent predictor + its internal anti-collapse.

The objective, added by `DormouseModel` as ONE term under one weight:

    J = (1 − cos(pred(h[p]), sg(h[p+h])))            # the prediction
        + jepa_reg_weight · sigreg_loss(pred(h[p]))  # the anti-collapse

    loss = CE + jepa_weight · J

`h` is the loop's own readout `[b, t, d]` from the SAME forward — no teacher, no
second forward, no mask, no EMA. The predictor is `Linear(d, d/4) → SiLU →
Linear(d/4, d)` applied to `h[p]`; the target `h[p+h]` is DETACHED, and in a
causal model it is a function of strictly more input than `h[p]`, so the
predictor has a real information bottleneck rather than the burn-era
lagged-self-distillation shape (`docs/research/2026-10-16-jepa-into-dormouse.md`
§1 (b), §4 D-e).

The anti-collapse term is INTERNAL, not a separate arm: a cosine predictor alone
can satisfy its objective by emitting one constant vector for every position,
which drives the loss to zero while carrying no information. SIGReg
(`sigreg.py`, arXiv:2511.08544) is read on the PREDICTED latents, so the
mechanism is self-contained — `jepa_reg_weight` only exists inside JEPA, and
`jepa_weight` scales the sum. `tests/test_jepa.py`'s collapse gate is the
evidence that this term is load-bearing: after 200 steps the predicted latents
stay in a bounded non-degenerate band with it and diverge without it (the gate
prints both numbers; at this scale the measured failure is scale divergence,
not a collapse to a constant).

What JEPA buys and what it does NOT. It SHAPES THE REPRESENTATION: it asks the
loop's readout to be predictive of its own near future and to stay isotropic, so
the state `lm_head` reads carries next-step information instead of a
rank-starved collection of shortcuts. It does NOT teach logic, reasoning or
anything a token-level objective cannot already extract — that is the RLVR
lane's job, and no amount of latent prediction substitutes for it. A tie in the
A/B therefore deletes this arm; it is not a prerequisite for reasoning.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .sigreg import sigreg_loss, stride_sample


class JepaHead(nn.Module):
    """`[b, t, d]` loop readout → the JEPA term (prediction + anti-collapse).

    Built only when `jepa_weight > 0`, so the off arm has no parameter and no
    compute (`tests/test_jepa.py` pins the bit-identity of the off path). The
    forward touches no host: every number it returns is a device scalar.
    """

    def __init__(
        self,
        d_model: int,
        *,
        horizon: int = 1,
        reg_weight: float = 0.1,
        hidden: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        assert d_model >= 1, f"d_model must be >= 1, got {d_model}"
        assert horizon >= 1, f"jepa horizon must be >= 1, got {horizon}"
        assert reg_weight >= 0.0, f"jepa_reg_weight must be >= 0, got {reg_weight}"
        self.horizon = horizon
        self.reg_weight = reg_weight
        h = max(1, d_model // 4) if hidden is None else hidden
        self.down = nn.Linear(d_model, h, device=device)
        self.up = nn.Linear(h, d_model, device=device)
        #: The last forward's PREDICTED latents, detached: the collapse gate
        #: (`tests/test_jepa.py`) reads them after training. Never read in a
        #: step, so it costs no host sync (ADR-0018 rule 2).
        self.last_pred: torch.Tensor | None = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """`[b, t, d]` → the 0-dim JEPA term (cosine + `reg_weight` · SIGReg)."""
        assert h.ndim == 3, f"JEPA reads the loop readout [b, t, d], got {tuple(h.shape)}"
        b, t, d = h.shape
        horizon = self.horizon
        n_pred = b * (t - horizon)
        # The covariance needs >= 2 rows and the prediction needs >= 1 position:
        # a window this short is a shape error, never a silent zero term.
        assert t > horizon and n_pred >= 2, (
            f"JEPA needs t > horizon and >= 2 predicted positions, got t={t}, "
            f"b={b}, horizon={horizon}"
        )
        pred = self.up(F.silu(self.down(h[:, : t - horizon])))
        target = h[:, horizon:].detach()
        pred_loss = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
        reg = sigreg_loss(stride_sample(pred.reshape(n_pred, d)))
        self.last_pred = pred.detach()
        return pred_loss + self.reg_weight * reg
