"""Multi-Token Prediction at the BYTE level (DeepSeek-V3 §3.2; arXiv:2404.19737).

DeepSeek-V3 (`arXiv:2412.19437` §3.2) trains K extra prediction depths on the
trunk's own hidden state: module `j` normalises the previous module's output and
the embedding of the next *known* byte, concatenates them through one linear,
and reads the SHARED output head for the byte `j+1` steps ahead. The trunk keeps
its ordinary next-byte CE; the MTP terms are a second objective on the same
embedding and head, so the only new parameters are the K `Linear(2d, d)`
matrices.

## The prior evidence this arm must be read against (do not argue it away)

`arXiv:2404.19737` (Gloeckle et al.) is the paper this port already tried once:
`model/fb.py`'s `aux_fb_weight` was a dense future-byte head on the same hidden
state, and it **tied at 3 seeds x 2 000 steps on `small` (9.2M, 3.0490 arm vs
3.0517 control) and was deleted in `39d0485`** ("delete the aux_fb future-byte
head — a TIE"). The paper's own claim is scoped where it was measured — "more
useful for larger model sizes", 7B/13B LM gains — and `arXiv:2505.22757` is the
direct negative at small scale ("SLMs struggle with the MTP objective"). So:
**MTP's published wins are large-scale; at our 9M-scale the A/B decides**, and
a tie deletes the mechanism (AGENTS §1.2). This module exists to make that A/B
possible, not because the win is assumed.

DSpark (`dspark.py`) is a DRAFT HEAD for speculative decoding; MTP is a TRAINING
objective. They coexist: this module never touches `dspark`, and `model.py` adds
both terms independently when both are armed.

## The exact mapping (off-by-one named)

The brief's one-line sketch (`embed(ids shifted j)`, `targets shifted by j+1`)
cannot be read under one indexing convention. This implementation is the
paper's, with `j = 0..K-1`:

* module `j` reads the hidden state at position `q` and `embed(ids[q + j + 1])`
  — the next byte the trunk has NOT yet consumed at `q`;
* it is scored against `targets[q + j + 1]` (`targets[q] == x[q+1]`, so this is
  the byte at `q + j + 2`), which at `j = 0` is exactly `targets[t+1]` — the
  gate's contract.

The literal `embed(ids shifted j)` (shift 0 at `j = 0`) would feed the byte
`h_last` has already consumed and would disagree with DeepSeek-V3 §3.2, where
the module that predicts `t_{k+1}` consumes `t_k`; it is NOT implemented.
Positions whose shift runs past the window are dropped by SLICING (no wrap,
no modulo, no mask tensor): with a window of `t` bytes, module `j` scores
`t - (j + 1)` positions.

## Named design choices

* **The two RMSNorms are gain-free** (`elementwise_affine=False`). The stated
  parameter budget is the K projection matrices alone — at `d = 1024`, K = 4:
  4 x 2 097 152 = 8.4M — and a learnable norm gain would add 2*d*K more. This
  is a deviation from DeepSeek-V3's learnable norms, and it is the budget's.
* **No host reads.** The forward never calls `.item()`, `int(tensor)` or
  `.tolist()` (gate 5). The returned loss is a device scalar; `per_k` is a list
  of device scalars. The caller decides when to read them.
* **The embedding and the head are the MODEL's**, passed in and shared — the
  MTP gradient reaches them because that is the paper's design, not because a
  copy was made.
* **Uniform mean over the modules that produced positions.** At `t > K` that is
  all K; a short window simply scores fewer terms, and `t <= 1` returns an exact
  zero with an empty `per_k` rather than a fabricated term.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import loud

#: The config layer's ceiling (`config.validate`); the constructor re-checks it
#: because a hand-built `ModelConfig` never passes through `validate`.
MAX_K = 8


class MTPHead(nn.Module):
    """K sequential future-byte modules, sharing the model's embed and head.

    Module `j` owns `norms_h[j]`, `norms_e[j]` and `projs[j]`; `projs[j]` is the
    only parameterised part (the norms are gain-free), so `2 * d * d` per module
    at `[2d, d]`, bias-free. `forward` consumes the PREVIOUS module's output,
    which is what makes the K depths sequential rather than K independent shallow
    heads.
    """

    def __init__(
        self,
        d_model: int,
        k: int,
        *,
        eps: float = 1e-3,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if k < 1 or k > MAX_K:
            loud(
                f"mtp k={k}: the depth is a count of future-byte modules in 1..={MAX_K}. "
                f"Escape: mtp_k in 1..={MAX_K}, or 0 (off, nothing is built)."
            )
        assert d_model > 0, "d_model must be positive"
        self.k = k
        self.d_model = d_model
        self.norms_h = nn.ModuleList(
            [
                nn.RMSNorm(d_model, eps=eps, elementwise_affine=False, device=device)
                for _ in range(k)
            ]
        )
        self.norms_e = nn.ModuleList(
            [
                nn.RMSNorm(d_model, eps=eps, elementwise_affine=False, device=device)
                for _ in range(k)
            ]
        )
        # fp32, bias-free like `lm_head`'s callers: the burn rule is "cast to
        # fp32 before every Linear", and a bias here would be one more parameter
        # the budget did not name.
        self.projs = nn.ModuleList(
            [nn.Linear(2 * d_model, d_model, bias=False, device=device) for _ in range(k)]
        )

    def forward(
        self,
        h_last: torch.Tensor,
        ids: torch.Tensor,
        targets: torch.Tensor,
        *,
        embed: nn.Module,
        head: nn.Module,
        weight: float,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """(`weight` x mean CE over the modules, the per-module CEs).

        `h_last [b, t, d]` is the loop's final readout (pre-`norm`), `ids` and
        `targets [b, t]` are the consumed bytes and their next-byte labels. Each
        module's term is a plain CE; the returned scalar is `weight` times their
        uniform mean. No position is reused across the shift: module `j` scores
        the first `t - (j + 1)` positions against `targets[:, j+1:]`.
        """
        assert weight >= 0.0, f"mtp weight {weight} is a multiplier, not a sign"
        assert h_last.ndim == 3, f"h_last must be [b, t, d], got {tuple(h_last.shape)}"
        b, t, d = h_last.shape
        assert d == self.d_model, f"h_last width {d} != mtp d_model {self.d_model}"
        assert ids.shape == targets.shape == (b, t), (
            f"ids {tuple(ids.shape)} and targets {tuple(targets.shape)} must both be [{b}, {t}]"
        )
        assert targets.dtype in (torch.long, torch.int64), "targets must be byte ids"
        per_k: list[torch.Tensor] = []
        h = h_last
        for j in range(self.k):
            shift = j + 1
            n = t - shift
            if n <= 0:
                # Every later module shifts further; none has a position left.
                break
            hp = self.norms_h[j](h[:, :n].to(torch.float32))
            e = self.norms_e[j](embed(ids[:, shift : shift + n]).to(torch.float32))
            x = self.projs[j](torch.cat([hp, e], dim=-1))  # [b, n, d]
            logits = head(x.reshape(b * n, d).to(torch.float32))
            per_k.append(F.cross_entropy(logits, targets[:, shift : shift + n].reshape(b * n)))
            # The next module is deeper by one shift: its input positions are
            # this module's output WITHOUT the last row, so no label is ever
            # reused and nothing wraps.
            h = x
        if not per_k:
            return torch.zeros((), dtype=torch.float32, device=h_last.device), per_k
        # No finiteness check HERE: `bool(tensor)` is a host read, and the
        # caller's own `assert torch.isfinite(loss)` already fires on the sum.
        return torch.stack(per_k).mean() * weight, per_k
