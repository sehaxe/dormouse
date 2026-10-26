"""Energy-based scoring (EBT, arXiv:2507.02092) over `(state, candidate byte)`.

EBT's inference is energy MINIMISATION over a candidate sequence, and its
reported win is +29% gain from inference compute against Transformer++ — the
kind of scaling that matters when tokens, not parameters, are the binding
constraint. The paper needs MCMC because its candidate space is open. This port
runs a **256-byte vocabulary**, so the candidate set is ENUMERABLE: every
candidate is scored by one matmul, and no sampler, no Langevin step and no
gradient-of-energy inference loop is needed here.

## The exact form

    u = W_h · h              # [.., d]   -> [.., k]
    v = W_y · y              # [.., d_e] -> [.., k]
    e(h, y) = w_o · SiLU(u + v)          # [..] one scalar per pair

`W_h` is `Linear(d, k)` and `w_o` is `Linear(k, 1, bias=False)`; `W_y` is the
shared `Linear(d_e, k, bias=False)`, so scoring all V candidates for one state
is `W_y · Yᵀ` once, an add, a SiLU and one dot — not V heads.

`h` is the state `lm_head` reads (`model.py`: `hn = self.norm(out)`), and `y`
is the model's OWN byte embedding row (`self.embed.weight`), so the head adds
no embedding table and the two uses read the same candidate vectors.

## The training term (aux, OFF by default)

`energy_contrastive_loss` is margin-InfoNCE over one positive and `n_negs`
negatives sampled from the `lm_head` softmax (with the target removed, so a
sampled negative is never the positive):

    L = CE([-e_pos/τ, -(e_neg + margin)/τ], class 0)

i.e. `e_pos` must be lower than every sampled negative by `margin`. The softmax
denominator is over the sampled set, which is InfoNCE's estimate of the full
partition (the 256-candidate exact denominator is available in the reranker,
where no gradient is needed).

## The inference use

`scripts/rerank.py` scores K rollouts of one prompt, picks the lowest-energy
rollout and prints pass@1 (rollout 0) / pass@1 (energy-selected) / pass@K from
the repo's deterministic verifiers. **No quality claim exists until that table
and the A/B (`scripts/ab_energy.py`) run** — the mechanism is built, not
measured.

## Purity (the DSpark D1 lesson)

`EnergyHead.energy` and `EnergyHead.scores` take `(h, candidate)` only. The
target byte reaches `energy_contrastive_loss` to GATHER the positive and to
draw negatives; it never reaches the scorer. `tests/test_energy.py::gate_c`
pins the signatures and hooks the head's inputs, so a future "just pass targets
in" edit fails a test rather than silently leaking the label into the score.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

#: The separation the training loss asks for, in energy units. It is part of
#: the LOSS, not the scorer: a fixed value keeps the arm's one swept knob the
#: weight (`energy_weight`), and τ (`energy_temp`) scales both sides.
MARGIN = 1.0

__all__ = ["MARGIN", "EnergyHead", "energy_contrastive_loss", "sample_negatives"]


class EnergyHead(nn.Module):
    """`e(h, y) = w_o · SiLU(W_h h + W_y y)` — one scalar per (state, candidate)."""

    def __init__(
        self,
        d_model: int,
        rank: int = 64,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        assert d_model > 0 and rank > 0, "energy head needs positive widths"
        self.d_model = d_model
        self.rank = rank
        self.h_proj = nn.Linear(d_model, rank, device=device)
        # Bias-free candidate projection: a bias would be a candidate-independent
        # offset, which is exactly a term the comparison cannot use.
        self.y_proj = nn.Linear(d_model, rank, bias=False, device=device)
        self.out = nn.Linear(rank, 1, bias=False, device=device)

    def energy(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """`h [.., d]`, `y [.., d]` -> `[..]`, broadcast as `(h + y)` does."""
        return self.out(F.silu(self.h_proj(h) + self.y_proj(y))).squeeze(-1)

    def scores(self, h: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """Every candidate's energy: `h [.., d]`, `candidates [V, d]` -> `[.., V]`.

        ONE candidate projection and ONE dot over all V rows; the reranker and
        the calibration control both read this, so "score all 256" is the same
        arithmetic in both places.
        """
        u = self.h_proj(h)  # [.., k]
        v = self.y_proj(candidates)  # [V, k]
        return self.out(F.silu(u.unsqueeze(-2) + v)).squeeze(-1)

    def forward(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.energy(h, y)


def sample_negatives(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_negs: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """`n_negs` byte ids per position, from the `lm_head` softmax, target removed.

    `logits [P, V]` are detached: the sampler is an instrument, not a path for
    gradient. The target's probability is zeroed and the row renormalised, so a
    sampled negative cannot be the positive (a false negative would cap the
    loss for no reason). Replacement is allowed — `n_negs=64` over V=256 draws
    an occasional duplicate, and de-duplicating on the device would cost more
    than the duplicate costs.
    """
    assert logits.ndim == 2, f"logits must be [P, V], got {tuple(logits.shape)}"
    assert targets.ndim == 1 and targets.shape[0] == logits.shape[0]
    assert n_negs >= 1, f"n_negs must be >= 1, got {n_negs}"
    probs = torch.softmax(logits.detach().to(torch.float32), dim=-1)
    probs.scatter_(1, targets.reshape(-1, 1), 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probs, n_negs, replacement=True, generator=generator)


def energy_contrastive_loss(
    head: EnergyHead,
    h: torch.Tensor,
    candidates: torch.Tensor,
    targets: torch.Tensor,
    logits: torch.Tensor,
    *,
    n_negs: int = 64,
    temp: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Margin-InfoNCE: the target byte's energy beats `n_negs` sampled negatives.

    `h [.., d]` (the state `lm_head` reads), `candidates [V, d_e]` (the byte
    embedding table), `targets [..]` (the positive id per position), `logits
    [P, V]` (the `lm_head` output the negatives are drawn from).

    No host read: `multinomial`, `gather`, `cat`, `cross_entropy` only. The
    returned scalar is a device tensor; the caller decides when to read it.
    """
    hf = h.reshape(-1, h.shape[-1])
    flat = targets.reshape(-1)
    assert flat.shape[0] == hf.shape[0], (
        f"energy loss: {hf.shape[0]} states against {flat.shape[0]} targets"
    )
    pos = candidates[flat]  # [P, d_e]
    neg_ids = sample_negatives(logits, flat, n_negs, generator=generator)  # [P, N]
    neg = candidates[neg_ids]  # [P, N, d_e]
    e_pos = head(hf, pos)  # [P]
    e_neg = head(hf.unsqueeze(1), neg)  # [P, N]
    # Class 0 is the positive; the margin is subtracted from every negative's
    # logit, so the positive is preferred only when e_pos < e_neg + MARGIN.
    logits_c = torch.cat(
        [(-e_pos / temp).unsqueeze(1), -(e_neg + MARGIN) / temp],
        dim=1,
    )
    zero = torch.zeros(hf.shape[0], dtype=torch.long, device=logits_c.device)
    return F.cross_entropy(logits_c, zero)
