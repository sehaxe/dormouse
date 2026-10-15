"""Hashed n-gram memory (Engram) — FNV byte n-grams, one table per order.

Semantics mirrored from the burn side, and they are the burn side's on purpose:
the keys the port looks up must be the keys the burn trainer looked up, or an
arm verdict does not transfer across substrates (readiness plan §7).

* `ORDERS = [2, 3, 4]` — `dormouse-data::ORDERS`, replacing the dead 3/5/8
  (only n=3 had per-key support at 8M rows; n=5/n=8 were ~5775-way averages).
* **FNV-1a, burn's constants**, low **31 bits** of the digest
  (`raw_keys`): `Int` is i32 on burn-flex and the cast panics above i32::MAX, to
  which CUDA wraps instead. The n-gram at position `t` is the last `min(n, t+1)`
  bytes ENDING AT AND INCLUDING `t` (burn's `saturating_sub`).
* the slot index is `key & (slots - 1)` with `slots = next_power_of_two(rows)`,
  masked by the MODEL (burn: "the model owns the capacity"); the power-of-two
  rounding is what makes the masked index in-range by construction.
* the gate is the DeepSeek one, `sigmoid(sign(s)·sqrt(clamp_min(|s|, 1e-6)))`
  with `s = key·query/√dim` — CLAMP, not `+1e-6`, which the burn head had to
  fix in `engram_oracle.rs::gate_is_on_the_clamp_min_side`.
* the memory joins its branch as a convex mixture with a HARD floor,
  `lam = min(w_mem, lam_max)`, the other half a dense projection of the same
  hidden state (FwPKM eq. 12 / kNN-LM eq. 3 — `loop_block::memory_floor_mix`).

Day-one table: a plain fp32 `nn.Parameter` on the module, routed to AdamW by
the optimizer policy (name-based: it is the one 2-D parameter that must NOT be
orthogonalized). Host-RAM offload is a STUB that refuses loudly — the burn
`--engram-ram` playbook (Qwen3.8-Flash-Next §2.3: pinned-memory prefetch of the
next batch's rows) is a plan §Phase-3 item, not a day-one behaviour.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from ..instrument import Counters, loud

FNV_OFFSET_BASIS = 1469598103934665603  # burn's constant, verbatim (NOT 2^40·sqrt(5))
FNV_PRIME = 1099511628211
FNV_MASK64 = (1 << 64) - 1
KEY_MASK = 0x7FFF_FFFF  # 31 bits: burn's i32-safe truncation


def fnv(data: bytes | Sequence[int]) -> int:
    """FNV-1a over bytes, 64-bit wrapping — `dormouse_data::fnv`."""
    h = FNV_OFFSET_BASIS
    for x in data:
        h ^= x & 0xFF
        h = (h * FNV_PRIME) & FNV_MASK64
    return h


def raw_keys(byte_ids: Sequence[int], orders: Sequence[int]) -> list[list[int]]:
    """`[t][order]` — the low 31 bits of each order's causal n-gram digest."""
    n = len(byte_ids)
    out: list[list[int]] = []
    for e in range(1, n + 1):
        row = []
        for order in orders:
            s = max(0, e - order)
            row.append(fnv(byte_ids[s:e]) & KEY_MASK)
        out.append(row)
    return out


def table_slots(rows: int) -> int:
    """`rows` rounded UP to a power of two (burn's `engram_tables`)."""
    if rows < 1:
        loud(f"engram_rows={rows} must be >= 1")
    return 1 << (rows - 1).bit_length()


class Engram(nn.Module):
    """Per-order hash tables + a gated read of the batch's own rows."""

    def __init__(self, cfg, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.orders = list(cfg.engram_orders)
        self.dim = cfg.engram_dim
        self.slots = table_slots(cfg.engram_rows)
        self.lam_max = cfg.engram_lam_max
        self.d = cfg.d_model
        # The table: fp32, and the arm is fp32 end to end (see `forward`).
        self.tables = nn.Parameter(
            torch.randn(len(self.orders), self.slots, self.dim, device=device) * 0.02
        )
        self.query_proj = nn.Linear(self.d, self.dim, bias=False, device=device)
        self.key_proj = nn.Linear(self.dim, self.dim, bias=False, device=device)
        self.value_proj = nn.Linear(self.dim, self.d, bias=False, device=device)

    def slots_for(self, input_ids: torch.Tensor) -> torch.Tensor:
        """`[b, t, n_orders]` slot indices — the model masks, per burn."""
        ids = input_ids.detach().to("cpu").tolist()
        keys = [raw_keys(row, self.orders) for row in ids]
        t = torch.tensor(keys, dtype=torch.long, device=input_ids.device)
        assert t.shape == (*input_ids.shape, len(self.orders)), f"key shape {tuple(t.shape)}"
        return t & (self.slots - 1)

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        counters: Counters | None = None,
    ) -> torch.Tensor:
        """`hidden [b,t,d]` + the byte ids of the SAME positions → `[b,t,d]`.

        fp32 end to end, table and projections alike: on the burn stack the
        Engram kernels are f32-only (ILLEGAL_ADDRESS on bf16, measured
        2026-08-29) and the arm is cast at its BOUNDARY (AGENTS §2.2). Keeping
        that here costs nothing measurable and makes the arm's dtype one
        decision instead of one per tensor.
        """
        counters = Counters() if counters is None else counters
        b, t, d = hidden.shape
        slots = self.slots_for(input_ids)  # [b, t, n_orders]
        order_idx = torch.arange(len(self.orders), device=hidden.device).view(1, 1, -1)
        emb = self.tables[order_idx, slots]  # [b, t, n_orders, dim]
        counters.count("engram_rows", int(b * t * len(self.orders)))
        query = self.query_proj(hidden)  # [b, t, dim]
        key = self.key_proj(emb)  # [b, t, n_orders, dim]
        s = (key * query.unsqueeze(2)).sum(-1) / math.sqrt(self.dim)
        # sign(s)·sqrt(clamp_min(|s|, 1e-6)) — the reference's CLAMP, not `+1e-6`
        # (the burn oracle test that separated the two).
        gate = torch.sigmoid(s.sign() * s.abs().clamp_min(1e-6).sqrt())
        value = self.value_proj(emb)  # [b, t, n_orders, d]
        return (value * gate.unsqueeze(-1)).sum(2)  # [b, t, d]


def memory_floor_mix(
    memory: torch.Tensor, dense: torch.Tensor, w_mem: torch.Tensor, lam_max: float
) -> torch.Tensor:
    """`out = lam·memory + (1-lam)·dense`, `lam = min(w_mem, lam_max)`.

    FwPKM eq. 12 / kNN-LM eq. 3 with lambda as a hard CONSTANT (kNN-LM: "a
    tuned constant, not a learned value" — which is the reason it generalizes):
    the guarantee is STRUCTURAL, so whatever the controller learns the
    memory's coefficient cannot exceed `lam_max` and the backbone's share of
    its own branch is never below `1 - lam_max`. The burn failure this
    prevents: a table that explained the targets (rec → 0.005, held-out at the
    uniform 8.000 BPB).
    """
    lam = w_mem.clamp(0.0, lam_max)
    return memory * lam + dense * (1.0 - lam)


def host_tables(*_args, **_kwargs):
    """Host-RAM Engram tables: NOT ported — loud, with the escape named."""
    loud(
        "engram_ram (host-RAM n-gram tables) is not ported: the day-one table is "
        "a plain fp32 parameter on the module. The burn path's playbook is "
        "Qwen3.8-Flash-Next §2.3 (pinned-memory prefetch of the next batch's "
        "rows over a separate stream, plan Phase 3.7) and it wants the CUDA "
        "prefetch path to exist first. Escape: remove `engram_ram`, or port the "
        "offload (engram.py:host_tables)."
    )
