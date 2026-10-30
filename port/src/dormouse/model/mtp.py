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

## Soft-MTP (arXiv:2610.38331, "Multi-Token Prediction Done Right", Teoh 2026)

`forward_soft` implements the paper's KL gradient-recovery trick. The standard
MTP CE at head `k` (module `j = k-1`, shift `k`) and the NTP CE at the ALIGNED
position — NTP position `q + k` predicts the same byte as head `k` at position
`q` — are both `(p - onehot)·W` in hidden space. Their difference cancels the
one-hot:

    d/dx_k [CE_mtp_k - CE_ntp_at_aligned] = (p_mtp - p_ntp)·W   (the KL gradient)

The trick gets that difference without a second softmax by adding a
straight-through term to the NTP hidden the CE reads:

    ntp_hidden[:, k:] += s_k · (x_k.detach() - x_k)

The forward value is exactly unchanged (`x - x = 0`); the backward at `x_k`
gets `-s_k · dCE_ntp/dntp_hidden[t+k]`, which, at the right `s_k`, is exactly
the NTP half of the KL gradient. `s_k` undoes the NTP CE's mean and total-loss
scale and re-applies head `k`'s own mean and the 1/K head average:

    s_k = weight · N_ntp / (n_heads · N_k · ntp_loss_scale)

with `ntp_loss_scale` the factor the NTP CE carries in the total loss (`1/(T+1)`
in `model.py`; `1.0` in the paper's own loss, which has no depth divisor).
`forward` is unchanged and bit-identical when the flag is off: the soft branch
is the only code that touches the coupling.

**The premise this repo satisfies, named rather than assumed:** `model.py`
passes the SAME `self.lm_head` to both the NTP CE and this module. That head may
be dense, TSCT-factorized or `TiedHead` (the tiny default is TSCT); all three
are linear in the hidden, which is the whole premise — the identity is in the
head's hidden-space gradient and never reads `W` as a matrix. It is NOT
recovered at `W` itself: the head's own parameter gradient stays the hard-CE
push (the reference's `test_the_two_objectives_differ_at_the_unembedding`).

**The seam, and why `mtp_soft` REQUIRES it.** The reference's `final_norm`
gives each MTP head its OWN post-norm hidden (`z_k`), which no later head reads.
Without that seam, `x_j` is both head `j`'s input and module `j+1`'s input, so
the coupling applied to `x_{j+1}` back-propagates through the sequential
projection and the identity is exact only at the LAST head. `head_norm=True`
adds the reference's seam: a per-head `RMSNorm` (with a learnable gain, see
below) applied to the projection output, read ONLY by that head's CE and its
coupling; the next module still consumes the RAW `x_j`. With it the KL
identity holds at EVERY head (`tests/test_mtp_soft.py`'s float64 gate).
`config.py` refuses `mtp_soft=true` with `mtp_head_norm=false`; a hand-built
`MTPHead` can still be called without the seam, and the tests pin that the old
chain term is then real, so the refusal is not ceremony.

**Alignment, off-by-one named:** module `j` (shift `j+1`) scores
`targets[:, j+1:]` from `x_j[:, :t-j-1]`; its label at index `q` is the same
byte the NTP CE at position `q + j + 1` scores. So the coupled slice is
`ntp_hidden[:, shift:]`, not `[:, :n]` and not a roll.

## Named design choices

* **The two INPUT RMSNorms are gain-free** (`elementwise_affine=False`) — the
  `norms_h`/`norms_e` before the concat. The per-head FINAL norm (the
  `mtp_head_norm` seam) is a different module and it has a learnable gain,
  matching the reference's `blocks.RMSNorm`; `K·d` parameters, which at
  `d = 1024`, K = 4 is 4 096 — 0.05% of the 8.4M the projections cost. The
  input norms stay gain-free because that is this port's stated projection
  budget (the reference's own input norms do carry gains; that is a
  pre-existing budget deviation, not the seam's).
* **No host reads.** The forward never calls `.item()`, `int(tensor)` or
  `.tolist()` (gate 5). The returned loss is a device scalar; `per_k` is a list
  of device scalars. The caller decides when to read them.
* **The embedding and the head are the MODEL's**, passed in and shared — the
  MTP gradient reaches them because that is the paper's design, not because a
  copy was made.
* **Uniform mean over the modules that produced positions.** At `t > K` that is
  all K; a short window simply scores fewer terms, and `t <= 1` returns an exact
  zero with an empty `per_k` rather than a fabricated term.
* **The seam (`head_norm=True`) is what makes the identity per-head.** The
  reference's `final_norm` gives each head its OWN post-norm hidden `z_k` that
  no later head reads. Without it, `x_j` is both head `j`'s input and module
  `j+1`'s input, so the coupling applied to `x_{j+1}` back-propagates through
  the sequential projection: the identity then holds only at the last head
  (test `test_without_the_seam_the_nonfinal_head_carries_the_chain_term`).
  With the seam, `z_j = head_norms[j](x_j)` is used ONLY for head `j`'s CE and
  its coupling; the next module still consumes the RAW `x_j`. `config.py`
  refuses `mtp_soft=true` without the seam, and `MTPHead` can still be built
  without it (the tests use that to pin the chain term).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import loud

#: The config layer's ceiling (`config.validate`); the constructor re-checks it
#: because a hand-built `ModelConfig` never passes through `validate`.
MAX_K = 8


def _widen(t: torch.Tensor) -> torch.Tensor:
    """Upcast a low-precision activation to fp32; leave fp32/fp64 alone.

    The burn rule this module follows is "cast to fp32 before every Linear".
    `fp64` is the KL-recovery gate's dtype: downcasting it would throw away the
    identity the gate measures, and no production path produces fp64 here. The
    reference's `hi()` lambda is the same rule (`never DOWNcast fp64`).
    """
    return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t


class MTPHead(nn.Module):
    """K sequential future-byte modules, sharing the model's embed and head.

    Module `j` owns `norms_h[j]`, `norms_e[j]`, `projs[j]` and (when the seam is
    on) `head_norms[j]`. `projs[j]` is the only parameterised part of the
    projection; `norms_h`/`norms_e` are gain-free, while `head_norms` carries a
    learnable gain (the reference's `RMSNorm`) when `head_norm=True` — `K·d`
    parameters. `forward` consumes the PREVIOUS module's output, which is what
    makes the K depths sequential rather than K independent shallow heads.
    """

    def __init__(
        self,
        d_model: int,
        k: int,
        *,
        eps: float = 1e-3,
        head_norm: bool = False,
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
        # The reference's per-head final norm (`final_norm(hiddens[head])`):
        # WITH a gain, matching `blocks.RMSNorm` in the reference. It isolates
        # head j's input from module j+1's, which is what makes the soft-MTP KL
        # identity hold at every head, not just the last. `None` when off, so
        # the off path allocates nothing and the state dict is unchanged.
        self.head_norms = (
            nn.ModuleList(
                [
                    nn.RMSNorm(d_model, eps=eps, elementwise_affine=True, device=device)
                    for _ in range(k)
                ]
            )
            if head_norm
            else None
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
        loss, per_k, _coupled = self._run(
            h_last,
            ids,
            targets,
            embed=embed,
            head=head,
            weight=weight,
            soft=False,
            ntp_hidden=None,
            ntp_loss_scale=1.0,
        )
        return loss, per_k

    def forward_soft(
        self,
        h_last: torch.Tensor,
        ids: torch.Tensor,
        targets: torch.Tensor,
        *,
        embed: nn.Module,
        head: nn.Module,
        weight: float,
        ntp_hidden: torch.Tensor,
        ntp_loss_scale: float,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Soft-MTP: the CE terms, plus the NTP hidden the caller must score.

        The returned third tensor is `ntp_hidden` with each module's
        straight-through coupling added at its ALIGNED slice:
        `out[:, shift:] = ntp_hidden[:, shift:] + s_j·(x_j.detach() - x_j)`.
        It is numerically equal to `ntp_hidden` in the forward; only its
        backward differs. The caller computes the NTP CE on it (with the SAME
        head and the same total-loss factor it passed as `ntp_loss_scale`) and
        uses that CE in place of the plain NTP CE.

        `ntp_loss_scale` is the factor the NTP CE's mean carries in the total
        loss — `1/(T+1)` in `model.py`, `1.0` in the paper's loss. It is an
        argument, not a constant here: the MTP module must not guess the
        caller's depth convention.
        """
        loss, per_k, coupled = self._run(
            h_last,
            ids,
            targets,
            embed=embed,
            head=head,
            weight=weight,
            soft=True,
            ntp_hidden=ntp_hidden,
            ntp_loss_scale=ntp_loss_scale,
        )
        assert coupled is not None, "soft=True must return the coupled NTP hidden"
        return loss, per_k, coupled

    def _run(
        self,
        h_last: torch.Tensor,
        ids: torch.Tensor,
        targets: torch.Tensor,
        *,
        embed: nn.Module,
        head: nn.Module,
        weight: float,
        soft: bool,
        ntp_hidden: torch.Tensor | None,
        ntp_loss_scale: float,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor | None]:
        assert weight >= 0.0, f"mtp weight {weight} is a multiplier, not a sign"
        assert h_last.ndim == 3, f"h_last must be [b, t, d], got {tuple(h_last.shape)}"
        b, t, d = h_last.shape
        assert d == self.d_model, f"h_last width {d} != mtp d_model {self.d_model}"
        assert ids.shape == targets.shape == (b, t), (
            f"ids {tuple(ids.shape)} and targets {tuple(targets.shape)} must both be [{b}, {t}]"
        )
        assert targets.dtype in (torch.long, torch.int64), "targets must be byte ids"
        # The modules that actually produce positions: `shift = j+1` needs
        # `t > j+1`, so the loop stops after `min(k, t-1)` of them. The mean
        # below divides by THAT count, and the soft scale must too.
        n_heads = min(self.k, t - 1)
        assert n_heads >= 0, "a negative module count is impossible"
        if soft:
            assert ntp_hidden is not None, "soft MTP needs the NTP hidden to couple"
            assert ntp_hidden.shape == h_last.shape, (
                f"ntp_hidden {tuple(ntp_hidden.shape)} != h_last {tuple(h_last.shape)}"
            )
            assert ntp_loss_scale > 0.0, (
                f"ntp_loss_scale={ntp_loss_scale}: it is the NTP CE's factor in the "
                f"total loss, so 0 would put the routed term at an infinite scale"
            )
        # A CLONE, never an in-place edit of the caller's hidden: `model.py`
        # still returns the logits/aux terms computed from the original.
        coupled = ntp_hidden.clone() if (soft and n_heads > 0) else None
        per_k: list[torch.Tensor] = []
        h = h_last
        for j in range(n_heads):
            shift = j + 1
            n = t - shift
            hp = self.norms_h[j](_widen(h[:, :n]))
            e = self.norms_e[j](_widen(embed(ids[:, shift : shift + n])))
            x = self.projs[j](torch.cat([hp, e], dim=-1))  # [b, n, d]
            # The reference's per-head seam: `z` is the ONLY input head `j`
            # reads, and the coupling writes its gradient into `z`; the next
            # module still consumes the RAW `x`, so a later head's coupling
            # cannot flow back into `z`.
            z = x if self.head_norms is None else self.head_norms[j](_widen(x))
            logits = head(_widen(z.reshape(b * n, d)))
            per_k.append(F.cross_entropy(logits, targets[:, shift : shift + n].reshape(b * n)))
            if coupled is not None:
                # `s_k = weight · N_ntp / (n_heads · N_k · ntp_loss_scale)`:
                # undo the NTP CE's mean (N_ntp) and total-loss factor, then
                # apply head k's own mean (N_k) and the 1/n_heads average.
                # This is the reference's `kl_recovery_scale` with the depth
                # divisor (`ntp_loss_scale`) made an explicit argument. N_ntp
                # is the full `b*t` because the NTP CE here has no ignore mask
                # (`model.py` asserts `loss_mask is None` on this path); N_k is
                # `b*n` because no MTP position is masked either.
                scale = weight * (b * t) / (n_heads * (b * n) * ntp_loss_scale)
                coupled[:, shift:] = coupled[:, shift:] + scale * (z.detach() - z)
            # The next module is deeper by one shift: its input positions are
            # this module's output WITHOUT the last row, so no label is ever
            # reused and nothing wraps.
            h = x
        if not per_k:
            return torch.zeros((), dtype=torch.float32, device=h_last.device), per_k, coupled
        # No finiteness check HERE: `bool(tensor)` is a host read, and the
        # caller's own `assert torch.isfinite(loss)` already fires on the sum.
        return torch.stack(per_k).mean() * weight, per_k, coupled
