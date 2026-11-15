"""The swappable auxiliary objective: ONE seam, two heads, both OFF by default.

`aux_objective` in a preset names the arm and `aux_weight` is λ on a term ADDED
to the token loss:

    loss = CE + aux_weight · J

so the token loss never changes — the head reads the loop's readout and never
writes it (`tests/test_aux.py` pins the logits `torch.equal` against the off
arm). Both heads follow the same stop-gradient discipline: the target is the
model's OWN state (or a handcrafted summary of its own inputs), detached, so the
head shapes the representation through its INPUT and never through its target.

`lightmtp` — LightMTP (arXiv:2610.06031). A two-layer MLP (GELU) on the loop's
final readout `h[p]` predicts the trunk's own hidden state at `p+h`, scored with
the paper's cosine loss (Eq. 1) on normalised vectors. **Cosine, not MSE:** the
target is a stop-grad state whose scale the trunk is free to drift, and a
squared error spends the head's whole gradient chasing that scale instead of the
direction. It is also the failure this repo has already measured — JEPA's
scale-free cosine arm without its reg drove the latent's scale to 1e4 while the
loss went to 0 (`tests/test_jepa.py`), i.e. an unnormalised target is a
magnitude race, not a prediction.

`fsp` — future summary prediction (arXiv:2510.14751 §3.2, HANDCRAFTED variant).
ONE head predicts the binary bag-of-words of the future window,
`a(t,τ)_i = 1[x_i ∈ {x_{t+2}, …, x_{t+τ}}]`, as `V` logits under BCE — the
variant that needs no second model. The learned variant needs a reverse LM,
which is a second trunk and a second training run, so it is not implemented
here. The window starts at `t+2`, not `t+1`: `x_{t+1}` is the NTP target
already (the paper's Eq. 9), so including it would make this a second NTP.

Two documented deviations, named rather than hidden:

* FSP's `w(i)` (Eq. 10, tf-idf) is replaced by a BALANCED reduction: mean BCE
  over the positive entries plus mean BCE over the negatives, averaged. On a
  256-byte vocabulary a uniform per-entry mean puts ~τ/256 of the gradient into
  the positives, and the head can lower its loss by learning the marginal byte
  frequency without one bit about the position. Balanced halves keep both
  gradients at O(1). The A/B decides whether that is enough; a tie deletes it.
* `aux_horizon` is an int, so the SEAM builds ONE horizon. `LightMTPHead` takes
  a tuple (one output projection per horizon, one shared input projection) and
  `tests/test_aux.py` runs `(1, 2, 4)`; several horizons from a preset need a
  list field in `config.py` when the A/B asks for it.

PARAMETERS. LightMTP is `d·w + w·d` (plus biases) with `w = aux_dim` or, at the
`aux_dim = 0` default, `d // 24` — the width that measures ~0.5% of the model at
BOTH the tiny (64 881) and the small (9.2M) shapes, because the head and the
trunk both scale as `d²`. The model constructor refuses a LightMTP head over 1%
of the model, which is the paper's own budget. FSP adds `d·w + w·V`.

THE PRIOR, named rather than argued away: `docs/records/research/
2026-11-15-component-catalogue.md` §2 keeps this whole class OFF — a
preregistered audit of 22 auxiliary objectives on Llama-3.2-1B found none
surviving multiplicity correction, and this repo's own JEPA arm tied (+0.0017
BPB). Both papers here are positive at 3B/8B. This module exists so the arm is
one key away, not because a win at 9M–142M is assumed: a tie deletes it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import AUX_OBJECTIVES
from ..instrument import loud


def auto_dim(d_model: int) -> int:
    """`aux_dim = 0`: the width that keeps the head near 0.5% of the model.

    Both the head (`2·d·w`) and the trunk (~`16·d²` at the tiny and small
    shapes) scale as `d²`, so a width proportional to `d` keeps the RATIO fixed
    instead of shrinking with scale.
    """
    return max(1, d_model // 24)


class LightMTPHead(nn.Module):
    """`h[p]` → the trunk's own (detached) `h[p+h]`, cosine, several horizons.

    One shared input projection (`down`, the cheap direction) and one output
    projection per horizon, so `K` horizons cost `K` (`w→d`) matrices rather
    than `K` whole MLPs. `ids` is accepted and ignored: the target is a hidden
    state, never a token, which is exactly what separates this from MTP.
    """

    def __init__(
        self,
        d_model: int,
        *,
        horizons: tuple[int, ...] = (1,),
        dim: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        assert d_model >= 1, f"d_model must be >= 1, got {d_model}"
        assert horizons and min(horizons) >= 1, (
            f"lightmtp horizons {horizons}: each is a POSITIVE step into the "
            f"future (`h[p+h]`), and 0 would predict the position itself."
        )
        self.d_model = d_model
        self.horizons = tuple(horizons)
        w = auto_dim(d_model) if dim <= 0 else dim
        self.down = nn.Linear(d_model, w, device=device)
        self.up = nn.ModuleList([nn.Linear(w, d_model, device=device) for _ in self.horizons])

    def forward(self, h: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        """`[b, t, d]` → the 0-dim term: mean over horizons/positions of `1−cos`."""
        assert h.ndim == 3, f"lightmtp reads the loop readout [b, t, d], got {tuple(h.shape)}"
        z = F.gelu(self.down(h))
        terms: list[torch.Tensor] = []
        for j, step in enumerate(self.horizons):
            n = h.shape[1] - step
            if n <= 0:
                continue
            # The target is DETACHED and read from the same tensor the head's
            # input is sliced from: position `p` sees `h[p]` only, so the shift
            # can never carry information from `p+step` back into `p`.
            target = h[:, step:].detach()
            pred = self.up[j](z[:, :n])
            terms.append((1.0 - F.cosine_similarity(pred, target, dim=-1)).mean())
        if not terms:
            return torch.zeros((), dtype=torch.float32, device=h.device)
        return torch.stack(terms).mean()


class FSPHead(nn.Module):
    """`h[p]` → the bag-of-words of `x[p+2 .. p+horizon]`, `V` logits.

    The summary is HANDCRAFTED from `input_ids` (the paper's Eq. 9), so it carries
    no graph and no second model: `requires_grad` is false by construction, and
    the only trainable path into the trunk is the head's input `h[p]`.
    """

    def __init__(
        self,
        d_model: int,
        vocab: int,
        *,
        horizon: int = 8,
        dim: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        assert d_model >= 1 and vocab >= 1, f"d_model={d_model}, vocab={vocab}"
        assert horizon >= 2, (
            f"fsp horizon={horizon}: it is the END of the summary window "
            f"`x[t+2 .. t+τ]`, and τ < 2 has no window (t+1 is the NTP target)."
        )
        self.d_model = d_model
        self.vocab = vocab
        self.horizon = horizon
        w = auto_dim(d_model) if dim <= 0 else dim
        self.down = nn.Linear(d_model, w, device=device)
        self.out = nn.Linear(w, vocab, device=device)

    def summary(self, ids: torch.Tensor) -> torch.Tensor:
        """`[b, t]` byte ids → the `[b, t, V]` 0/1 membership of the future window.

        Position `p` counts the bytes at `p+2 .. p+horizon`; a position whose
        window runs past the end simply has fewer entries (sliced, never wrapped).
        """
        b, t = ids.shape
        assert t > self.horizon, (
            f"fsp needs t > horizon={self.horizon} to have any window, got t={t}"
        )
        bow = torch.zeros((b, t, self.vocab), dtype=torch.float32, device=ids.device)
        for step in range(2, self.horizon + 1):
            bow[:, : t - step].scatter_(2, ids[:, step:].unsqueeze(-1), 1.0)
        return bow

    def forward(self, h: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        """`[b, t, d]` + `[b, t]` → the balanced BCE of the predicted summary."""
        assert h.ndim == 3, f"fsp reads the loop readout [b, t, d], got {tuple(h.shape)}"
        assert ids is not None, "fsp needs the consumed ids: its target is a summary of them"
        bow = self.summary(ids)
        assert not bow.requires_grad, "the fsp summary is handcrafted; a graph here is a bug"
        logits = self.out(F.gelu(self.down(h))).float()
        # The last two positions have NO window left (`p+2 > t-1`), so their
        # target is the empty set — scoring them would teach the head to predict
        # "nothing" at the window's edge. They are sliced, never wrapped.
        n = h.shape[1] - 2
        logits, bow = logits[:, :n], bow[:, :n]
        per = F.binary_cross_entropy_with_logits(logits, bow, reduction="none")
        n_pos = bow.sum().clamp_min(1.0)
        n_neg = (bow.numel() - bow.sum()).clamp_min(1.0)
        return 0.5 * (per * bow).sum() / n_pos + 0.5 * (per * (1.0 - bow)).sum() / n_neg


def build_aux(
    objective: str,
    *,
    d_model: int,
    vocab: int,
    horizon: int,
    dim: int = 0,
    device: torch.device | str = "cpu",
) -> nn.Module | None:
    """The seam: `aux_objective` → a head (or `None` for `"none"`).

    The unknown-literal refusal lives in `config.py` AND here, because a
    hand-built `ModelConfig` never passes through `validate` (the same rule the
    halt/energy heads follow). `"none"` builds nothing at all, which is what
    makes the off arm bit-identical.
    """
    if objective == "none":
        return None
    if objective == "lightmtp":
        return LightMTPHead(d_model, horizons=(horizon,), dim=dim, device=device)
    if objective == "fsp":
        return FSPHead(d_model, vocab, horizon=horizon, dim=dim, device=device)
    loud(
        f"aux_objective={objective!r} is not an arm. Known: {list(AUX_OBJECTIVES)} "
        f"— `none` builds nothing, `lightmtp` predicts the trunk's own future "
        f"hidden state, `fsp` predicts a bag-of-words of the future window "
        f"(model/aux.py). Escape: name one of those, or leave aux_weight=0.0."
    )
