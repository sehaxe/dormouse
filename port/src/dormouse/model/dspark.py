"""The DSpark draft head (arXiv 2607.05147, DeepSeek-AI) — STANDALONE, unlanded.

Ported from our own `vendor/dormouse-fused/crates/burn-dspark` (`lib.rs`,
`markov.rs`) and from the call site in `crates/dormouse-core/src/aux.rs:314-419`.
The paper's formulas: Eq 5 `B = W2(W1[x])` (low-rank first-order bias,
`markov.rs:12-54`), Eq 6 the recurrent head `[s; W1[x]; h] -> gate|cand|out`
(`markov.rs:152-200`), Eq 7 the acceptance head `sigmoid(wᵀ[h_k; W1[x_{k-1}]])`
(`lib.rs:54-133`), Eq 9-12 the loss `0.1·CE + 0.9·TV + 1.0·BCE` (`lib.rs:301-303`).

## The three defects this port fixes, and where they lived

**D1 — the window was built from the LABEL sequence.** `model.rs` passed
`targets` where Eq 4/5/6 need `input_ids`, and since `targets[q] == x[q+1]` the
window's step `s` was fed `x[p+s+1]` — the byte `logits[p+s]` had just predicted
— and supervised toward `x[p+s+2]`, while the hidden state it was handed had
only seen `x[0..=p+s]`. Here the window is built from the CONSUMED sequence: at
anchor `p`, step `s` reads `x[p+s]` and predicts `x[p+s+1]`, gathered exactly as
`aux.rs:355` (`gather(1, pos)`) and `aux.rs:358` (`gather(1, pos+1)`) do. The
authority is `tests/test_dspark_gold.py::make_window`, whose `x[i] = i` vector
makes the off-by-one unmissable, and `tests/test_dspark_v2.py` runs that vector
through THIS builder.

**D9 — the acceptance head had no Markov feature.** `aux.rs:311` passed `None`
and `with_markov` had zero callers, so the head was a `wᵀ[h_k]` stand-in. The
input here is `[h_k; W1[x_{k-1}]]` — the draft head's OWN `W1`, the only copy of
that table in the module (`markov.rs:202-209`) — so the projection's input is
`d_model + markov_rank` and not `d_model`.

**D2 — γ was decoupled from `k`.** `aux.rs:27` hardcodes `DSPARK_GAMMA = 4.0`
while the paper ties one γ to the K-step block. This module takes NO gamma: the
returned scalar is the unweighted mean of Eq 12's mix, and the CALLER applies γ
(`position_weights` is exported for a caller that wants Eq 9's per-step
`w_k = exp(-k/γ)`). A module-level constant that no config can move is the
defect; a named export the caller must use is the fix.

## What is frozen, and what is detached

`logits` enter detached (`aux.rs:333`: the target is frozen) and the acceptance
target `c* = 1 - 0.5·‖p_draft - p_target‖₁` (`lib.rs:159-164`) is detached too
(`lib.rs:290`, DeepSpec `loss.py:145`): it is a LABEL derived from the drafter's
own output, not a second objective. Without the detach the confidence term
pushes `-c·∂c*/∂draft` into the drafter — 16.6× the reference's gradient on the
burn fixture's worst case (`lib.rs:243-253`).

The BCE is spelled `(1-c)·z + softplus(-z)` (`lib.rs:262-284`) and not the usual
`-log σ` pair: the naive form evaluates `log(0)` at `|z| = 30` in fp32. Same
algebra as `binary_cross_entropy_with_logits`, smooth, and its derivative is
`sigmoid(z) - c` identically.

## The integration hook this module expects (a LATER commit, not this one)

`model.py`'s byte-LM forward, beside the `fb` term (`model.py:210-216`), and
nowhere else:

    if cfg.dspark_weight > 0.0:
        loss = loss + cfg.dspark_weight * dspark_aux_loss(
            self.dspark, hn.to(torch.float32), input_ids, logits, counters=counters
        )

with the head built IFF `cfg.dspark_weight > 0.0` (the `fb` rule, `model.py:83-85`,
so a preset that never arms it keeps today's parameter set and checkpoint), and
NOT in any decode forward: a decode pass has no labels and no window, and the
burn side gates the aux graph on `targets.is_some()` for exactly that reason.

## The one REJECTED alternative

**No MTP-EMA teacher.** A teacher copy is what makes the JEPA term expensive
(a full second forward; the measured price is a halved legal batch,
`benches/history.tsv:51-56`), and the draft head needs none: the frozen target
logits are the same network's own (`dspark.md` D15), so the correction is
self-supervised at the cost of one low-rank projection
(`docs/research/2026-10-16-jepa-dspark-alternatives.md` §1.3, row 3, and
`2610.06031`'s teacher-free LightMTP). No EMA, no momentum schedule, no second
forward.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import Counters, loud

#: Paper Eq 5's default Markov rank `r`, the same constant
#: `tests/test_dspark_gold.py::MARKOV_RANK` pins (768 + 256 for `small`).
MARKOV_RANK = 256
#: Draft steps per anchor (the burn `dspark_k` default) and the anchor SPACING
#: in byte positions (`dspark_stride`, burn's default 16 — see `anchor_count`).
DEFAULT_K = 4
DEFAULT_STRIDE = 16
#: Eq 12's mix (`lib.rs:301-303`).
CE_WEIGHT = 0.1
TV_WEIGHT = 0.9


def anchor_count(t: int, k: int, stride: int) -> int:
    """Anchors that fit in `t` bytes: `floor((t-k-1)/stride)`, else zero.

    Burn's `aux.rs:331-336`: the whole `k+1`-byte window must be inside the
    sequence, so the first anchor needs `t >= k + 1 + stride` and anything
    shorter gets NO window (the term is then exactly zero, COUNTED by the
    caller). `stride` is a distance in BYTE POSITIONS — the stream is bytes, so
    one position is one byte, not a token and not a fraction.
    """
    if k < 1:
        loud(f"dspark k={k}: the draft window is at least one byte. Escape: k >= 1.")
    if stride < 1:
        loud(
            f"dspark stride={stride}: the anchor spacing is at least one byte "
            f"position (0 collapses the k-step window into k copies of position 0). "
            f"Escape: stride >= 1; burn's default is {DEFAULT_STRIDE}."
        )
    return (t - k - 1) // stride if t > k + 1 else 0


def anchor_positions(t: int, k: int, stride: int) -> torch.Tensor:
    """The anchors `p = 0, stride, 2·stride, …` as an int64 tensor `[n]`."""
    return torch.arange(anchor_count(t, k, stride), dtype=torch.long) * stride


def stable_bce(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """`(1-c)·z + softplus(-z)` — `binary_cross_entropy_with_logits`, no kink.

    Burn's spelling (`lib.rs:262-284`), which is the algebra BCE is written in:
    the standard `c·sp(-z) + (1-c)·sp(z)` rearranged so `softplus` is the only
    transcendental. Finite at `z = ±30` where `-log σ(±30)` overflows to `inf` in
    fp32, and its derivative is `sigmoid(z) - c` identically.
    """
    return (1.0 - c) * z + F.softplus(-z)


def accept_rate_target(draft_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """Eq 8: `c* = clamp(1 - 0.5·‖p_draft - p_target‖₁, 0, 1)` (`lib.rs:159-164`).

    ATTACHED on purpose — the `.detach()` belongs at the loss, where the value
    is used as a label (`dspark_aux_loss`); this function is also the exported
    reference a test compares the detached fixture against.
    """
    pd = F.softmax(draft_logits, dim=-1)
    pt = F.softmax(target_logits, dim=-1)
    tv = (pd - pt).abs().sum(dim=-1)
    return (1.0 - 0.5 * tv).clamp(0.0, 1.0)


def position_weights(k: int, gamma: float) -> torch.Tensor:
    """Eq 9's per-step decay `w_k = exp(-k/gamma)` (`lib.rs:145-150`).

    Exported for the CALLER (D2): this module's scalar is unweighted, because
    γ is a run-level quantity and a module constant cannot be A/B'd against
    `k`. A caller that wants the paper's per-step decay builds the terms itself
    off `DSparkHead.terms` and weights them with this row.
    """
    return torch.exp(-torch.arange(k, dtype=torch.float32) / gamma)


class Window(NamedTuple):
    """The per-anchor window over the CONSUMED sequence (D1)."""

    positions: torch.Tensor  # [n]    anchors p = 0, stride, 2·stride, …
    token_ids: torch.Tensor  # [b,n,k] x[p+s]   — the conditioning byte (Eq 5/6)
    target_ids: torch.Tensor  # [b,n,k] x[p+s+1] — the byte to predict (Eq 9)


class Terms(NamedTuple):
    """The draft, its frozen target, and the acceptance head's two operands."""

    draft: torch.Tensor  # [b,n,k,v] frozen base logits + the head's low-rank bias
    target: torch.Tensor  # [b,n,k,v] the frozen distribution at the SAME position
    target_ids: torch.Tensor  # [b,n,k]   the ground-truth next byte
    conf: torch.Tensor  # [b,n,k]   Eq 7 acceptance logits
    c_star: torch.Tensor  # [b,n,k]   Eq 8 label, DETACHED
    n_anchors: int


class DSparkHead(nn.Module):
    """Eq 5/6 draft readout + Eq 7 acceptance head, one module per model.

    One `W1` serves both heads, which is what Eq 7's `W1[x_{k-1}]` means: the
    acceptance head is conditioned on the draft head's own Markov embedding, not
    on a second table (`markov.rs:202-209`). Every layer is bias-free where burn
    says so (`lib.rs:74-78`, `markov.rs:32-35`, `markov.rs:170`), so a zeroed
    weight is an exactly-zero logit — the property a wiring test needs.
    """

    def __init__(
        self,
        cfg,
        device: torch.device | str = "cpu",
        *,
        k: int | None = None,
        stride: int | None = None,
        markov_rank: int | None = None,
    ) -> None:
        super().__init__()
        d = int(cfg.d_model)
        vocab = int(cfg.vocab)
        # The config fields exist on the burn side (`dspark_k`/`dspark_stride`)
        # and land here in a later commit; `getattr` keeps this file standalone
        # without touching config.py.
        self.k = int(k if k is not None else getattr(cfg, "dspark_k", DEFAULT_K))
        self.stride = int(
            stride if stride is not None else getattr(cfg, "dspark_stride", DEFAULT_STRIDE)
        )
        self.markov_rank = int(
            markov_rank
            if markov_rank is not None
            else getattr(cfg, "dspark_markov_rank", MARKOV_RANK)
        )
        self.d_model = d
        self.vocab = vocab
        if self.markov_rank < 1:
            loud(
                f"dspark markov_rank={self.markov_rank}: Eq 7's acceptance head is "
                f"conditioned on W1[x] of rank >= 1, else it is the hidden-only "
                f"stand-in D9 fixed. Escape: markov_rank >= 1 ({MARKOV_RANK} = Eq 5)."
            )
        # Silence the range checks that would otherwise run on the first
        # forward, where a shape error is a batch-sized crash.
        anchor_count(self.k * 2 + 2, self.k, self.stride)
        r = self.markov_rank
        self.w1 = nn.Embedding(vocab, r, device=device)  # W1[x], Eq 5
        self.w2 = nn.Linear(r, vocab, bias=False, device=device)  # W2, Eq 5
        self.joint = nn.Linear(2 * r + d, 3 * r, bias=False, device=device)  # Eq 6
        self.conf = nn.Linear(d + r, 1, bias=False, device=device)  # Eq 7, D9

    def step(
        self,
        state: torch.Tensor,
        prev: torch.Tensor,
        hidden: torch.Tensor,
        base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One Eq 6 step: `(s, W1[x], h, base) -> (s', base + W2·tanh(out))`.

        The recurrence `terms()` runs over a window, as ONE call — so the
        training path and the inference path (`specdecode.draft_block`, one
        step per proposed byte) share one implementation of Eq 6 and cannot
        drift. Flat shapes: `state [N,r]`, `prev [N,r]`, `hidden [N,d]`,
        `base [N,v]`; the returned logits are `[N,v]`.
        """
        gate, cand, out = self.joint(torch.cat([state, prev, hidden], dim=-1)).split(
            self.markov_rank, dim=-1
        )
        gate = torch.sigmoid(gate)
        new_state = gate * state + (1.0 - gate) * torch.tanh(cand)
        return new_state, base + self.w2(torch.tanh(out))

    def window(
        self, input_ids: torch.Tensor, k: int | None = None, stride: int | None = None
    ) -> Window:
        """The `k`-step windows at every anchor, from `input_ids` (CONSUMED).

        `input_ids [b,t]` is the sequence the backbone READ, never the label
        sequence — see D1 in the module docstring. `n = floor((t-k-1)/stride)`
        anchors, and step `s` of anchor `p` is `(x[p+s], x[p+s+1])`.
        """
        k = self.k if k is None else k
        stride = self.stride if stride is None else stride
        assert input_ids.ndim == 2, f"input_ids must be [b,t], got {tuple(input_ids.shape)}"
        b, t = input_ids.shape
        anchors = anchor_positions(t, k, stride)
        if anchors.numel() == 0:
            return Window(anchors, input_ids.new_zeros(b, 0, k), input_ids.new_zeros(b, 0, k))
        steps = torch.arange(k, dtype=torch.long)  # [k]
        pos = anchors.reshape(1, -1, 1) + steps.reshape(1, 1, -1)  # [1,n,k]
        tokens = input_ids[:, pos[0]]
        targets = input_ids[:, pos[0] + 1]
        assert tokens.shape == (b, anchors.numel(), k), "window shape drifted"
        return Window(anchors, tokens, targets)

    def terms(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        k: int | None = None,
        stride: int | None = None,
    ) -> Terms:
        """The draft, the frozen target, and both acceptance operands.

        `hidden [b,t,d]` (grad), `logits [b,t,v]` (frozen here), `input_ids
        [b,t]`. The recurrence is Eq 6's, one step per draft position; the
        base logits it corrects are the frozen backbone's OWN at that position
        (draft and target are one network), which is the same tensor as the
        target distribution of the TV / `c*` terms.
        """
        k = self.k if k is None else k
        stride = self.stride if stride is None else stride
        assert hidden.ndim == 3 and logits.ndim == 3, "hidden and logits are [b,t,*]"
        b, t, d = hidden.shape
        v = logits.shape[-1]
        assert d == self.d_model and v == self.vocab, (
            f"hidden is d={d}/{self.d_model}, logits v={v}/{self.vocab}"
        )
        win = self.window(input_ids, k=k, stride=stride)
        n = win.positions.numel()
        assert n > 0, "terms() needs at least one anchor; aux_loss returns the zero term"
        frozen = logits.detach()  # aux.rs:333 — the target is frozen
        steps = torch.arange(k, dtype=torch.long)
        pos = win.positions.reshape(-1, 1) + steps.reshape(1, -1)  # [n,k]

        r = self.markov_rank
        state = hidden.new_zeros(b * n, r)  # Eq 6's s_0
        drafts: list[torch.Tensor] = []
        confs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for s in range(k):
            h = hidden[:, pos[:, s]]  # [b,n,d]
            prev = self.w1(win.token_ids[:, :, s])  # [b,n,r] — Eq 7's W1[x_{k-1}]
            base = frozen[:, pos[:, s]]  # [b,n,v]
            state, step_logits = self.step(
                state, prev.reshape(b * n, r), h.reshape(b * n, d), base.reshape(b * n, v)
            )
            drafts.append(step_logits.reshape(b, n, v))
            confs.append(self.conf(torch.cat([h, prev], dim=-1)))  # [b,n,1]
            targets.append(base)
        draft = torch.stack(drafts, dim=2)  # [b,n,k,v]
        target = torch.stack(targets, dim=2)  # [b,n,k,v]
        return Terms(
            draft=draft,
            target=target,
            target_ids=win.target_ids,
            conf=torch.stack(confs, dim=2).squeeze(-1),  # [b,n,k]
            # Eq 8 as a LABEL: detached (lib.rs:290). A live `c*` would push a
            # gradient into the drafter that exists only to make its own output
            # look more like the frozen target.
            c_star=accept_rate_target(draft, target).detach(),
            n_anchors=n,
        )

    def aux_loss(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        k: int | None = None,
        stride: int | None = None,
        counters: Counters | None = None,
    ) -> torch.Tensor:
        """One scalar: Eq 12's `0.1·CE + 0.9·TV + 1.0·BCE(conf, c*)`, unweighted."""
        return dspark_aux_loss(
            self, hidden, input_ids, logits, k=k, stride=stride, counters=counters
        )


def dspark_aux_loss(
    head: DSparkHead,
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    k: int | None = None,
    stride: int | None = None,
    counters: Counters | None = None,
) -> torch.Tensor:
    """The DSpark auxiliary term as ONE scalar, γ-free (D2 — the caller scales).

    Bumps `dspark_asked` once for every call that asked for the term and
    `dspark` by the number of windows BUILT (`b · n_anchors`), so the log line
    reads `dspark=<windows>/<calls>`: `0/<n>` says the length budget allowed no
    window and the term was exactly zero, which is the one thing a bare `0.0`
    return could not say (ADR-0019, burn's row "`dspark_aux_loss` contributing
    nothing when the sequence is shorter than the draft window").
    """
    if counters is not None:
        counters.count("dspark_asked")
    k = head.k if k is None else k
    stride = head.stride if stride is None else stride
    b, t, _ = hidden.shape
    n = anchor_count(t, k, stride)
    if n == 0:
        return torch.zeros((), dtype=torch.float32, device=hidden.device)
    terms = head.terms(hidden, input_ids, logits, k=k, stride=stride)
    if counters is not None:
        counters.count("dspark", b * n)
    v = terms.draft.shape[-1]
    # Eq 9: CE against the CONSUMED sequence's next byte (x[p+s+1], D1).
    ce = F.cross_entropy(terms.draft.reshape(-1, v), terms.target_ids.reshape(-1))
    # Eq 10: L1 distance between the corrected and the frozen distribution.
    pd = F.softmax(terms.draft, dim=-1)
    pt = F.softmax(terms.target, dim=-1)
    tv = (pd - pt).abs().sum(dim=-1).mean()
    # Eq 11: the stable, kink-free BCE on the DETACHED acceptance label.
    conf = stable_bce(terms.conf, terms.c_star).mean()
    return CE_WEIGHT * ce + TV_WEIGHT * tv + conf
