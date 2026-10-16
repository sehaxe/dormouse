"""The FUTURE-BYTE auxiliary head (arXiv 2404.19737).

Ported from burn's `crates/dormouse-core/src/future_byte.rs`, which is **our
adaptation of the paper's per-horizon multi-token heads to a byte-level AR
model** (AB-PROTOCOL row 1b). The claim reused is "training a model to predict
ahead buys induction-head structure at 1-30M", not the paper's shape, which it
does not prescribe for a 256-symbol byte vocabulary on a 9M recurrent block.

Why this arm exists: it is the pre-registered competitor to JEPA+KoLeo at
**2.1% of `small`'s parameters and no extra forward** — one linear over a hidden
state the model already computed — against a JEPA term whose measured price is a
halved legal batch (`docs/research/2026-10-16-jepa-dspark-alternatives.md`,
rows 1/3/4/5 and §4.1 item 1; `benches/history.tsv:51-56`).

The head: one dense, **untied** `d_model -> vocab` readout over the same
`norm(out)` the `lm_head` reads. Untied for the paper's reason (a head per
horizon so each can specialise) and for ours: a tied head would put the
future-byte gradient into `lm_head`, and the arm's whole claim is that it shapes
the backbone through the hidden state without spending the main head's capacity.
No EMA teacher, no acceptance head, no auditor — ONE plain CE.

## Which labels are read, and why the read stays position-ordered

`targets[q]` is the byte at `q + 1` (the trainer's labelling: `next_batch`
returns `raw[:, :-1]` and `raw[:, 1:]`). The head at position `q` is supervised
on `targets[q + 1]` — the label sequence **rolled by -1**, i.e. the byte at
`q + 2`, which is strictly in position `q`'s future. Reading `targets[q]` at
`q` instead would hand the head the byte the main logits at `q` just predicted
one step early: that is DSpark's D1 defect (`docs/papers/dspark.md` §7) in a
different arm, and `tests/test_dspark_gold.py` carries the gold vector for it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import Counters

#: The head's price on the `small` preset (d_model 768, vocab 256), mirrored
#: from burn's `FB_PARAMS_AT_SMALL` (`future_byte.rs:85`): a dense linear with a
#: bias, no low-rank factorisation — `256*768 + 256 = 196 864`.
#:
#: The research doc and AB-PROTOCOL quote "197 120 = 2.1%": that is this
#: arithmetic plus one extra bias (burn's own doc comment carries the slipped
#: identity "197 120 = 256*768 + 256", which evaluates to 196 864). Both numbers
#: round to 2.14% of the burn `small` (9 197 390), so the row's "2.1%" holds
#: either way; `tests/test_fb.py` pins the count the tree actually builds.
FB_PARAMS_AT_SMALL = 256 * 768 + 256


def valid_positions(t: int) -> int:
    """Positions a `t`-byte window can supervise: `t - 1`.

    The head at `q` is supervised on `targets[q + 1]`, so `q = t - 1` has no
    shifted label inside the window. `t = 1` has none at all: the arm runs,
    returns an exact zero and bumps `fb_asked` and never `fb_ran` — the `0/<n>`
    the A/B's pre-registration reads as "the head did not run => void".
    """
    return max(t - 1, 0)


class FutureByteHead(nn.Module):
    """A dense, untied `d_model -> vocab` readout, one per model.

    A readout like `lm_head`, so it is routed to the fallback optimizer, not to
    Muon (`train/optim.py`, burn's `Group::Rest`).
    """

    def __init__(self, d_model: int, vocab: int, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab, device=device)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


def future_byte_loss(
    head: FutureByteHead,
    h: torch.Tensor,
    targets: torch.Tensor,
    counters: Counters,
) -> torch.Tensor:
    """CE of `head` at position `q` against the label rolled by -1 (`q + 1`).

    `h [b, t, d]` is the same fp32 `norm(out)` the main head reads, `targets
    [b, t]` the next-byte labels. Returns a scalar: the mean over the
    `b * (t - 1)` valid pairs, or an exact zero when `t = 1`.

    The gradient flows into `h` (so into the backbone) and into the head, and
    into nothing else — no teacher, no `lm_head`, no embedding.
    """
    counters.count("fb_asked")
    b, t, _ = h.shape
    n = valid_positions(t)
    if n == 0:
        return torch.zeros((), dtype=torch.float32, device=h.device)
    counters.count("fb_ran")
    labels = targets[:, 1 : 1 + n].reshape(b * n)
    logits = head(h[:, :n].reshape(b * n, -1))
    assert logits.shape[0] == labels.shape[0], "head rows and shifted labels must pair up"
    return F.cross_entropy(logits, labels)
