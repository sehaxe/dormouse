"""The byte-level MC scorer: one forward per choice, teacher-forced.

`score_choices` is the whole instrument, and it is deliberately small: the
protocol is the interesting part, not the code.

For a choice `c` of `m` bytes given a context `C` of `k` bytes, the model is run
ONCE on `C + c` and the logits at rows `k-1 .. k+m-2` are the next-byte
distributions for `c[0] .. c[m-1]`. The score is the sum of their log-probs at
the chosen bytes. That is `Σ_j log p(c[j] | C + c[:j])` — causal, teacher-forced,
no sampling, no temperature: a temperature would scale every score by the same
positive factor and change nothing but the printed number's units.

The choice is scored as a CONTINUATION, which is why the task loaders own the
separator (`tasks.py` pins the template per task): a leading space is a byte the
model must predict, exactly as GPT-2-era MC scoring did it.

Batching is not done. Choices of different lengths would need padding, and the
cheap argument ("right-padding is causal so the choice positions are unaffected")
is only true for a position-wise model: a capacity-dropping MoE
(`moe_topk > 0` at `moe_capacity > 0`) reads the whole window, and this repo has
already paid once for a measurement of a different network than the one named
(`7adda92`). One forward per choice costs more and cannot lie.
`ponytail:` per-choice forwards, batch equal-length choices when a task's
step time, not its arithmetic, becomes the bottleneck.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

#: A model call: byte ids `[1, t]` in → logits `[t, vocab]` out.
LogitsFn = Callable[[torch.Tensor], torch.Tensor]

__all__ = ["ChoiceScore", "LogitsFn", "model_logits", "score_choices"]


@dataclass(frozen=True)
class ChoiceScore:
    """One choice's score, in BOTH units the harness reports.

    `logp` is the raw sum (the protocol's score, and what `argmax` picks);
    `mean` is the per-byte mean — the length-fair number, and the headline.
    """

    logp: float
    n_bytes: int

    @property
    def mean(self) -> float:
        assert self.n_bytes > 0, "a scored choice has at least one byte"
        return self.logp / self.n_bytes


def model_logits(model, ids: torch.Tensor, counters=None) -> torch.Tensor:
    """`DormouseModel` → the scorer's logits function (the ONE adapter).

    `forward` returns `(logits [b*t, vocab], loss, counters)`; a decode-style
    call with no targets gives `loss=None`, and the scorer only wants the first
    element. `counters` is passed through so the arm that ran (FLA KDA vs the
    SDPA fallback) is on the harness's own line.
    """
    logits, _loss, _counters = model(ids, counters=counters)
    return logits


def score_choices(
    logits_fn: LogitsFn,
    ctx: bytes,
    choices: Sequence[bytes],
) -> list[ChoiceScore]:
    """Score every choice of one item: ONE forward per choice, teacher-forced.

    `logits_fn` takes `[1, t]` integer ids and returns `[t, vocab]` logits —
    `model_logits` wraps the real model, and the tests pass a hand-built fake.
    """
    assert len(ctx) >= 1, "an item needs a context byte: row -1 is the first prediction"
    assert choices, "an item needs at least one choice"
    out: list[ChoiceScore] = []
    for choice in choices:
        assert len(choice) >= 1, "a scored choice needs at least one byte"
        ids = torch.tensor([list(ctx) + list(choice)], dtype=torch.long)
        logits = logits_fn(ids)
        assert logits.ndim == 2 and logits.shape[0] == ids.shape[1], (
            f"logits {tuple(logits.shape)} do not match the {ids.shape[1]} positions scored"
        )
        start = len(ctx) - 1
        rows = logits[start : start + len(choice)].float()
        logp = torch.log_softmax(rows, dim=-1)
        target = torch.tensor(list(choice), dtype=torch.long, device=logp.device)
        out.append(
            ChoiceScore(
                logp=float(logp.gather(1, target[:, None]).sum()),
                n_bytes=len(choice),
            )
        )
    return out
