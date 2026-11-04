"""Activation-recompute accounting: the modes, the kept bytes, the saved tensors.

Three arms, one enum (`TrainConfig.activation_recompute`):

* ``none`` — no checkpoint; autograd keeps every save (`_window`'s historical
  graph, bit for bit).
* ``full`` — `checkpoint(use_reentrant=False)` around the WHOLE iteration.
  Maximum memory drop, maximum recompute (`iter_checkpoint`'s arm).
* ``selective`` — keep the READ half (norm, controller, the q/k/v/g projections,
  the KDA/CSA boundary state and the Engram read) and checkpoint only the WRITE
  half (the MoE pass, the residual write and `out_proj`). The q/k/v/g OUTPUTS
  are kept because they are the expensive matmuls' outputs, and the boundary
  state is kept because it crosses iterations: recomputing either repeats work
  the forward already did, while recomputing the write half is what buys the
  memory.

THE ACCOUNTING is arithmetic, not a device measurement. [`saved_table`] installs
the same `saved_tensors_hooks` pair autograd itself uses and records every
tensor a forward hands to the backward with its exact byte count. With a
checkpoint installed the outer pair sees the checkpoint node's inputs (the
checkpoint's own inner pair hides the body's saves — measured in `act_ckpt.py`),
so the table is exactly the bytes that stay RESIDENT for the backward. The
[`projection_bytes`] helper is the selective arm's kept set from the module
shapes, so the test can prove the difference without a device.

THE GATE (`tests/test_vram_arms.py`) pins the per-arm tensor count and byte
total for the tiny fixture: adding a saved tensor without updating the pin
turns it red. The same test proves `selective` keeps the q/k/v/g output shapes
and `full` does not.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd.graph import saved_tensors_hooks

from ..instrument import loud

#: The three arms. `iter_checkpoint` remains the historical boolean alias for
#: `full`; the enum is the name a bench and a snapshot carry.
MODES = ("none", "full", "selective")

#: A projection whose OUTPUT is kept by `selective` (the expensive matmul in the
#: attention chains). The names are per attention arm — KDA's four dense
#: projections, block-sparse attention's three, CSA's per-level W_q/W_k/W_v.
PROJECTION_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "g_proj",
    "W_q",
    "W_k",
    "W_v",
)


def check_mode(mode: str) -> None:
    """LOUD on anything outside [`MODES`], naming the escape."""
    if mode not in MODES:
        loud(
            f"activation_recompute={mode!r} is not a recompute mode. Known: "
            f"{list(MODES)} — 'none' keeps every save, 'full' checkpoints the "
            f"whole iteration, 'selective' keeps the q/k/v/g projection outputs "
            f"and the boundary state and recomputes only the write half. Escape: "
            f"name one of those."
        )


class SavedTable:
    """Every tensor autograd saved in a region, with its exact bytes."""

    __slots__ = ("rows",)

    def __init__(self) -> None:
        self.rows: list[tuple[tuple[int, ...], torch.dtype, int]] = []

    def pack(self, t: object) -> object:
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            self.rows.append((tuple(t.shape), t.dtype, t.numel() * t.element_size()))
        return t

    def unpack(self, t: object) -> object:
        return t

    @property
    def tensors(self) -> int:
        return len(self.rows)

    @property
    def bytes(self) -> int:
        return sum(nbytes for _, _, nbytes in self.rows)

    def shapes(self) -> set[tuple[int, ...]]:
        return {shape for shape, _, _ in self.rows}


def saved_table(fn) -> tuple[SavedTable, object]:
    """Run `fn` under a counting pair; return `(table, result)`.

    The pair passes every value through untouched, so the forward's values and
    graph are the ones the caller asked for. It is the outer pair around a
    checkpoint: torch keeps one pair per save (innermost wins), so the table is
    exactly what the checkpoint holds for the backward.
    """
    table = SavedTable()
    with saved_tensors_hooks(table.pack, table.unpack):
        out = fn()
    return table, out


def projection_bytes(blocks, tokens: int, element_size: int = 2) -> int:
    """The selective arm's KEPT projection output bytes, from module shapes.

    `tokens` is the number of positions the attention sees (`b*t`). The output
    width is the projection's `out_features`; the dtype is the body's compute
    dtype (bf16 on the measured arm, 2 bytes), and the caller passes the real
    element size when it is not.
    """
    total = 0
    for blk in blocks:
        for owner in (blk.attn, blk.msa, blk.window_attn, blk.csa):
            if owner is None:
                continue
            for name in PROJECTION_NAMES:
                lin = getattr(owner, name, None)
                if isinstance(lin, nn.Linear):
                    total += tokens * lin.out_features * element_size
    return total
