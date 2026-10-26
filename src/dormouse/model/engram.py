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
orthogonalized).

**The host-RAM path** (`config.TrainConfig.engram_ram`, burn's `--engram-ram`)
keeps the SAME table values in plain CPU fp32 memory (`HostTables`, never an
`nn.Parameter` — the optimizer must not see them) plus ONE momentum buffer, and
gathers only the batch's unique rows to the device per forward as a grad-carrying
leaf. The optimizer state shrinks from AdamW's `m+v` (3 copies of the table) to
one momentum buffer (2), and the VRAM cost of the memory becomes
`batch × unique rows` instead of `orders × slots × dim × 4` — which is what makes
millions of slots affordable (Qwen3.8-Flash-Next §2.3, the design playbook's
item 1). The tables' own update is burn's, read off
`crates/dormouse-train/src/offload.rs`: Nesterov momentum with `MOMENTUM = 0.95`
(one look-ahead step per training step, not per forward) and — this is in the
burn code, not an addition — a Sinkhorn L1 balancing of the update block every
`SINKHORN_EVERY = 4` updates (`offload.rs::momentum_update` calls
`sinkhorn_l1`), because the batch's rows are a skewed sample of the table. Weight
decay is disabled on this path, as on the burn side.

**The bias correction on top of it** (`TrainConfig.engram_bias_correction`,
default on). Burn's look-ahead is an exponential sum whose total weight depends
on how many times the ROW has been touched: `block_t = (1-μ^{t+1})·g` for a
constant gradient, so a row touched once moves by `1-μ² = 0.0975` of its
gradient and needs ~`1/(1-μ) = 20` touches to reach full scale. That is right
for burn's 48M-row / 300k-step production table and wrong for a 600-step
screen, where most rows never get there. With the correction the block is
divided by that weight, `1-μ^{t+1}` (`t` = the row's OWN touch count, not the
global step: a global step over-corrects a row first touched at step 500), so
the first touch is exactly the gradient and a row with `t ≥ 337` (measured) divides by
`1.0` in fp32 — bit-unchanged. `false` is burn's formula verbatim, the A/B's
`ram_old_formula` arm.

What the host path does NOT do, and why that is honest: the rows that cross the
bus are `[unique, dim]` fp32 per forward and the grads `[unique, dim]` back per
step — the burn reference's own copy. There is no prefetch stream and no pinned
memory (the plan's Phase-3.7 item): this is the correct-by-construction version,
not the overlapped one, and the two transfers are the cost of it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import Counters, loud
from .tsct import qat_linear

FNV_OFFSET_BASIS = 1469598103934665603  # burn's constant, verbatim (NOT 2^40·sqrt(5))
FNV_PRIME = 1099511628211
FNV_MASK64 = (1 << 64) - 1
KEY_MASK = 0x7FFF_FFFF  # 31 bits: burn's i32-safe truncation

#: The pad value `slots_for` uses for the byte slot a window does NOT reach —
#: impossible for real bytes (0..255), so a fold on it is always a no-op.
_PAD_BYTE = 256

#: The host path's update, burn's constants verbatim
#: (`crates/dormouse-train/src/offload.rs`: `MOMENTUM = 0.95`,
#: `SINKHORN_EVERY = 4`, `SINKHORN_ITERS = 2`).
HOST_MOMENTUM = 0.95
HOST_SINKHORN_EVERY = 4
HOST_SINKHORN_ITERS = 2


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


def slot_keys_gpu(input_ids: torch.Tensor, orders: Sequence[int]) -> torch.Tensor:
    """The burn-identical keys, computed ON THE DEVICE — all torch ops.

    The host form (`raw_keys`) walked a python list per position — a
    GPU→CPU sync every forward, 15.6 ms/fwd in the profiler probe, and the
    single thing that graph-broke torch.compiled(dynamo) (an int64-overflow
    python literal inside the traced region). This version walks the LAGS
    (o-1..0): a window fold is `h = (h ^ byte) · prime & mask64`, bytes
    folded oldest→newest; a byte a window does not reach (position < lag)
    folds as a no-op via the impossible pad byte — the EXACT window
    semantics of `raw_keys` (`s = max(0, e - order)`), vectorized. int64
    multiply wrapping is what this hash REQUIRES (`& FNV_MASK64` restates
    the wrap); GPU/CPU agreement is pinned by test.
    """
    b, t = input_ids.shape
    dev = input_ids.device
    out = torch.empty(b, t, len(orders), dtype=torch.int64, device=dev)
    for oi, order in enumerate(orders):
        h = torch.full((b, t), FNV_OFFSET_BASIS, dtype=torch.int64, device=dev)
        for lag in range(order - 1, -1, -1):
            byte = F.pad(input_ids, (lag, 0), value=_PAD_BYTE)[:, :t]
            folded = ((h ^ byte) * FNV_PRIME) & FNV_MASK64
            h = torch.where(byte < _PAD_BYTE, folded, h)
        out[:, :, oi] = h & KEY_MASK
    return out


def table_slots(rows: int) -> int:
    """`rows` rounded UP to a power of two (burn's `engram_tables`)."""
    if rows < 1:
        loud(f"engram_rows={rows} must be >= 1")
    return 1 << (rows - 1).bit_length()


def sinkhorn_l1(block: torch.Tensor, iters: int) -> torch.Tensor:
    """SinkGD-style balancing of an update block — burn's `sinkhorn_l1`.

    Alternating L1 row/column normalization (signs preserved: the divisor is
    the sum of ABSOLUTE values), then a rescale to the block's ORIGINAL total
    L1 so the learning rate keeps its meaning. Rows/columns whose L1 is already
    ~0 are left alone (burn's `if s > 1e-12`) rather than divided by a clamped
    epsilon, which would turn a zero row into a spike.
    """
    if block.numel() == 0:
        return block
    orig = block.abs().sum()
    if not bool(torch.isfinite(orig)) or float(orig) <= 1e-12:
        return block
    keep = lambda s: torch.where(s > 1e-12, s, torch.ones_like(s))  # noqa: E731
    for _ in range(iters):
        block = block / keep(block.abs().sum(1, keepdim=True))
        block = block / keep(block.abs().sum(0, keepdim=True))
    now = block.abs().sum()
    if float(now) > 1e-12:
        block = block * (orig / now)
    return block


class HostTables:
    """The n-gram tables in plain CPU fp32 memory — never an `nn.Parameter`.

    Layout `[orders, slots, dim]`, contiguous, so a row's ABSOLUTE index is
    `order * slots + slot`; the slot is the masked key, in range by
    construction, so no bound check is needed here (burn's `unique_rows`). The
    optimizer state is `momentum`, same layout, plus `counts`, one int32 TOUCH
    COUNT per row (flat `[orders*slots]`, never per dim — the update is one row
    vector). The count is what the bias correction needs; `step` counts host
    updates and decides the Sinkhorn cadence.
    """

    def __init__(
        self,
        rows: torch.Tensor,
        momentum: torch.Tensor | None = None,
        step: int = 0,
        counts: torch.Tensor | None = None,
        *,
        bias_correction: bool = True,
    ) -> None:
        assert rows.dtype is torch.float32, f"host tables are fp32, got {rows.dtype}"
        assert rows.device.type == "cpu", f"host tables live on cpu, got {rows.device}"
        self.rows = rows
        self.momentum = torch.zeros_like(rows) if momentum is None else momentum
        self.step = int(step)
        assert self.momentum.shape == rows.shape, "momentum must mirror the tables"
        n_rows = rows.numel() // rows.shape[-1]
        self.counts = torch.zeros(n_rows, dtype=torch.int32) if counts is None else counts
        assert self.counts.shape == (n_rows,), (
            f"counts {tuple(self.counts.shape)} != ({n_rows},) — one per row"
        )
        assert self.counts.dtype is torch.int32, f"counts are int32, got {self.counts.dtype}"
        assert self.counts.device.type == "cpu", "counts live on cpu"
        self.bias_correction = bool(bias_correction)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.rows.shape)

    @property
    def dim(self) -> int:
        return self.rows.shape[-1]

    def flat(self) -> torch.Tensor:
        """`[orders*slots, dim]` — the absolute-indexed view the gather uses."""
        return self.rows.reshape(-1, self.dim)

    def momentum_update(self, uniq: torch.Tensor, grads: torch.Tensor, lr: float) -> int:
        """Nesterov + bias correction + (every 4th) Sinkhorn on the TOUCHED rows.

        burn's `momentum_update` line for line: `m = μ·m + (1-μ)·g`, the applied
        block is the look-ahead `μ·m_new + (1-μ)·g`, the table moves
        `t -= lr·block`, and the whole block is Sinkhorn-balanced before it is
        applied on every `HOST_SINKHORN_EVERY`-th update.

        The correction (`bias_correction`, default on) divides the block by the
        look-ahead's total weight `1-μ^{t+1}` on each row, `t` = that row's
        touch count after this update. It is applied AFTER Sinkhorn on purpose:
        Sinkhorn normalises the block's rows, so a scale applied before it is
        partly normalised away, and the correction must survive to the update.
        A row at `t ≥ 337` (measured) gets a divisor of exactly `1.0` in fp32, so its
        update is bit-identical to burn's.
        """
        n = int(uniq.numel())
        if n == 0:
            return 0
        assert grads.shape == (n, self.dim), f"grads {tuple(grads.shape)} != {(n, self.dim)}"
        assert uniq.dtype is torch.int64 and uniq.device.type == "cpu", "uniq is cpu int64"
        self.step += 1
        t, m = self.flat(), self.momentum.reshape(-1, self.dim)
        moved = HOST_MOMENTUM * m[uniq] + (1.0 - HOST_MOMENTUM) * grads
        m[uniq] = moved
        block = HOST_MOMENTUM * moved + (1.0 - HOST_MOMENTUM) * grads
        # Every touch is counted even on the old-formula arm: the count is the
        # row's history, and a checkpoint must not lose it because an A/B arm
        # happened to run without the correction.
        self.counts[uniq] += 1
        if self.step % HOST_SINKHORN_EVERY == 0:
            block = sinkhorn_l1(block, HOST_SINKHORN_ITERS)
        if self.bias_correction:
            # The look-ahead contains the current gradient twice — once through
            # `m`, once as its own `(1-μ)` term — so after `t` touches its total
            # weight is Σ_{k=0}^{t} μ^k = (1-μ^{t+1})/(1-μ), not `1-μ^t`.
            # Dividing the BLOCK by it (not `m` alone) is the correction of the
            # quantity actually applied.
            touched = self.counts[uniq].to(torch.float64) + 1.0
            denom = 1.0 - torch.pow(torch.tensor(HOST_MOMENTUM, dtype=torch.float64), touched)
            block = block / denom.to(block.dtype).unsqueeze(1)
        t[uniq] = t[uniq] - lr * block
        return n


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
        #: The DENSE path's switch (`tsct.dense_qat_project`, the SAME projector
        #: as the factor rung): the three projections are plain `nn.Linear`s,
        #: so their weights are read through the annealed STE. The TABLES
        #: (`self.tables` / the host rows) are memory, not a weight matrix —
        #: they stay fp32 (BitNet convention; also the burn Engram is fp32).
        self.dense_qat = cfg.dense_qat
        self.dense_qat_alpha = cfg.factor_qat_alpha
        #: Non-None = this Engram is on host tables (`to_host`): the tables are
        #: plain CPU fp32 (`HostTables`) and `self.tables` no longer exists, so
        #: the optimizer can never see them.
        self.host: HostTables | None = None
        #: Every forward of the CURRENT step's pending `(leaf, uniq)` pair. A
        #: step leaves one pair per ITERATION (the loop block calls the memory
        #: once per iteration over the SAME `input_ids`), and the gradients of
        #: those calls belong to the same rows: `host_step` sums them.
        self._pending: list[tuple[torch.Tensor, torch.Tensor]] = []

    def slots_for(self, input_ids: torch.Tensor) -> torch.Tensor:
        """`[b, t, n_orders]` slot indices — the model masks, per burn.

        The keys are now computed on the device (`slot_keys_gpu`): the same
        FNV-64 digest, no `.tolist()` sync, and nothing for dynamo to choke
        on. The burn-identical contract is now pinned by the pinned-value
        test rather than by the sharing of a python function.
        """
        t = slot_keys_gpu(input_ids, self.orders)
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

        The two table substrates (in-VRAM parameter, host tables) differ ONLY
        in how `emb` is obtained; the gate math below is `_read`, one
        implementation, so a parity claim is structural rather than two
        transcriptions that happen to agree.
        """
        counters = Counters() if counters is None else counters
        if self.host is not None:
            return self._forward_host(hidden, input_ids, counters)
        slots = self.slots_for(input_ids)  # [b, t, n_orders]
        order_idx = torch.arange(len(self.orders), device=hidden.device).view(1, 1, -1)
        emb = self.tables[order_idx, slots]  # [b, t, n_orders, dim]
        return self._read(hidden, emb, counters)

    def _read(self, hidden: torch.Tensor, emb: torch.Tensor, counters: Counters) -> torch.Tensor:
        """`hidden` + the gathered `[b,t,orders,dim]` memory → `[b,t,d]`.

        The DeepSeek gate, and the two counters the port's eval line reads:
        `engram_rows` (keys READ this forward — the burn `ENGRAM_KEYS` probe)
        and `engram_arms` (memory READS — the burn `ENGRAM` probe). A reader of
        `engram=0/<n>` learns the eval ran a memory-disabled forward (AGENTS
        §3.2), which is why they are counted on both substrates, in one place.
        """
        b, t, _ = hidden.shape
        counters.count("engram_rows", int(b * t * len(self.orders)))
        counters.count("engram_arms")
        dq = (self.dense_qat, self.dense_qat_alpha)
        query = qat_linear(self.query_proj, hidden, *dq)  # [b, t, dim]
        key = qat_linear(self.key_proj, emb, *dq)  # [b, t, n_orders, dim]
        s = (key * query.unsqueeze(2)).sum(-1) / math.sqrt(self.dim)
        # sign(s)·sqrt(clamp_min(|s|, 1e-6)) — the reference's CLAMP, not `+1e-6`
        # (the burn oracle test that separated the two).
        gate = torch.sigmoid(s.sign() * s.abs().clamp_min(1e-6).sqrt())
        value = qat_linear(self.value_proj, emb, *dq)  # [b, t, n_orders, d]
        return (value * gate.unsqueeze(-1)).sum(2)  # [b, t, d]

    def _forward_host(
        self, hidden: torch.Tensor, input_ids: torch.Tensor, counters: Counters
    ) -> torch.Tensor:
        """The host-table read: dedup the batch's rows, gather them to a leaf.

        Two INDEX-sized transfers per forward, both inherent to host tables: the
        slot indices go device→host (the addresses only exist as host rows — the
        burn path crosses the bus the same way, `offload.rs::rows_for_batch`)
        and the inverse map comes back. No `.item()`, no host branch on a device
        VALUE, so ADR-0018 rule 2 still holds for everything the step computes;
        what moves is the memory's own addressing.

        Dedup is `torch.unique` over the WHOLE `b·t·orders` batch (burn's
        `unique_rows`): a repeated n-gram appears once on the device and is read
        `k` times from the same row, so the row's gradient accumulates over its
        occurrences in one backward pass.
        """
        b, t, _ = hidden.shape
        orders = len(self.orders)
        slots = self.slots_for(input_ids)  # [b, t, orders], device
        order_idx = torch.arange(orders, device=slots.device).view(1, 1, -1)
        flat = (order_idx * self.slots + slots).reshape(-1).detach().to("cpu")
        uniq, inverse = torch.unique(flat, return_inverse=True)
        assert self.host is not None
        rows = self.host.flat()[uniq]  # [n, dim] cpu fp32
        leaf = rows.to(hidden.device).requires_grad_(True)
        emb = leaf[inverse.to(hidden.device)].reshape(b, t, orders, self.dim)
        self._pending.append((leaf, uniq))
        return self._read(hidden, emb, counters)

    def to_host(self, *, bias_correction: bool = True) -> None:
        """Move the tables OFF the module into plain CPU fp32 memory.

        `del self.tables` is the point of the path, not bookkeeping: the
        parameter leaves `parameters()`/`state_dict()`, so the optimizer group,
        the grad clip and the checkpoint's model half all stop carrying it — the
        VRAM the tables occupied is freed here. The values are COPIED (a device
        parameter would otherwise keep its allocation alive), and the resume
        path's state is the sibling `<ckpt>.ngram` file, not this dict.

        `bias_correction` is the update's formula (`HostTables`); the fresh
        counter starts at zero, which is the truth: no row has been updated.
        """
        assert self.host is None, "engram tables are already on host"
        rows = self.tables.detach().to("cpu").contiguous().clone()
        self.host = HostTables(rows, bias_correction=bias_correction)
        del self.tables

    def drop_pending(self) -> None:
        """Discard this step's pending leaves without updating anything.

        The NaN firewall's skip branch calls this (audit MINOR 13, 2026-10-17):
        a skipped step never reaches `host_step`, so the leaves it built would
        stay in `_pending` until the next REAL step — memory proportional to the
        NaN count, and their `uniq` rows re-deduped there. `host_step` clears the
        same list; this is the same clear with the update left out.
        """
        self._pending = []

    def host_step(self, lr: float, scale: float = 1.0) -> tuple[int, int]:
        """The ONE host update a TRAINING step gets: `(updates, rows)`.

        `scale` is the trainer's device firewall, already read on this arm
        (the row gradients come D2H here every step, so the flag rides an
        existing declared read): `0.0` means the step was non-finite, and the
        update is the OLD skip branch exactly — the pending leaves are dropped
        and neither the tables nor their momentum move. `1.0` is the healthy
        step, unchanged.

        Called by the trainer after `loss.backward()` + `opt.step()` — once per
        step, never per forward (the burn bug this rule exists for is the
        opposite one: `--host-adam-every` gated the update on log steps, so the
        tables got 1/100 of their updates, AGENTS §3.7).

        The step's forwards are summed into ONE update per row, which is what
        makes "once per step" true rather than "once per call": `T` iterations
        read the SAME `input_ids`, so they touch the same rows through `T`
        different leaves and `T` different hidden states. A row's gradient is
        the sum of its occurrences over the whole step — the same accumulation
        one backward over a single `[b·t·orders, dim]` gather would produce.
        Rows are re-deduped across the leaves for that reason.

        An eval forward runs under `no_grad` and leaves no grad: `(0, 0)` there,
        which the trainer counts as nothing rather than firing a zero update.
        """
        assert self.host is not None, "host_step on an in-VRAM Engram"
        pending, self._pending = self._pending, []
        if scale <= 0.0:
            # The device firewall's skip, on the substrate that has to see the
            # flag on the host: drop the leaves, move nothing (the old skip).
            for leaf, _ in pending:
                leaf.grad = None
            return (0, 0)
        grads = [
            leaf.grad.detach().to("cpu").reshape(-1, self.dim)
            for leaf, _ in pending
            if leaf.grad is not None
        ]
        touched = [uniq for leaf, uniq in pending if leaf.grad is not None]
        # The leaf dies with the step; its grad must not survive it (the trainer
        # zeroes the model's grads, and these leaves are not the model's).
        for leaf, _ in pending:
            leaf.grad = None
        if not grads:
            return (0, 0)
        uniq, inverse = torch.unique(torch.cat(touched), return_inverse=True)
        acc = torch.zeros(uniq.numel(), self.dim)
        acc.index_add_(0, inverse, torch.cat(grads))
        return (1, self.host.momentum_update(uniq, acc, lr))


def host_engrams(model: nn.Module) -> list[Engram]:
    """Every Engram in the tree that is on host tables (usually one)."""
    return [m for m in model.modules() if isinstance(m, Engram) and m.host is not None]


def enable_host_tables(model: nn.Module, *, bias_correction: bool = True) -> int:
    """Switch every Engram in `model` to host tables; returns how many did.

    Called by the trainer at setup, BEFORE the optimizer is built — the tables
    must never enter a parameter group. Zero means the run asked for host tables
    and there are none (`use_engram = false`), which the caller reports LOUD.
    `bias_correction` is the host update's formula; it is set on tables that are
    ALREADY host too, so a checkpoint loader can apply the resolved config to a
    model another call switched (the counter and tables are unchanged).
    """
    n = 0
    for m in model.modules():
        if isinstance(m, Engram):
            if m.host is None:
                m.to_host(bias_correction=bias_correction)
                n += 1
            else:
                m.host.bias_correction = bool(bias_correction)
    return n


def host_step_all(model: nn.Module, lr: float, scale: float = 1.0) -> tuple[int, int]:
    """The trainer's one host update per step, over the whole tree.

    `scale` is the device firewall's decision for this step (see
    `Engram.host_step`); the default keeps the pre-firewall caller unchanged.
    """
    updates = rows = 0
    for m in host_engrams(model):
        u, r = m.host_step(lr, scale)
        updates += u
        rows += r
    return (updates, rows)


def host_state(model: nn.Module) -> dict:
    """The `<ckpt>.ngram` payload: tables + momentum + step + counts, per Engram."""
    return {
        "engrams": [
            {
                "rows": m.host.rows,
                "momentum": m.host.momentum,
                "step": m.host.step,
                "counts": m.host.counts,
            }
            for m in host_engrams(model)
        ]
    }


def load_host_state(model: nn.Module, state: dict) -> None:
    """Restore [`host_state`] into the model's host tables, LOUD on a mismatch.

    A host run's checkpoint carries no `engram.tables` parameter (the model half
    is a model without the memory), so this file IS the memory's state — a
    silently-mismatched pair would train a different network than the one the
    step count claims, which is the `7adda92` class (a `.ngram` sidecar that did
    not travel with the weights).

    A file WITHOUT `counts` predates the bias correction and is REFUSED, not
    defaulted: the counter is the per-row history the corrected formula needs,
    and every guess (0, 1, the global step) changes what the next update does
    under the same step count. Refusing keeps the old run resumable on the
    commit that wrote it and makes a new run say so with its own `ckpt_name`;
    a silent default would be ADR-0021's defect with an extra number.
    """
    eng = host_engrams(model)
    sets = state.get("engrams") if isinstance(state, dict) else None
    if sets is None or len(sets) != len(eng):
        loud(
            f"the host-table file holds {0 if sets is None else len(sets)} table set(s) "
            f"but this model has {len(eng)}: the pair is mismatched (was it written by "
            f"another preset?). Escape: use the checkpoint's own ckpt_name, or delete "
            f"the file and start the memory from scratch knowingly."
        )
    for m, s in zip(eng, sets, strict=True):
        if tuple(s["rows"].shape) != m.host.shape:
            loud(
                f"host table shape {tuple(s['rows'].shape)} != {m.host.shape}: "
                f"engram_rows/engram_orders/engram_dim changed since the save. Escape: "
                f"the saved config, or a new ckpt_name."
            )
        if "counts" not in s:
            loud(
                f"the host-table file carries no per-row touch counter (`counts`): it was "
                f"written before the Engram update's bias correction, which needs each "
                f"row's own history (`1-{HOST_MOMENTUM}^(t+1)`). Loading it with a "
                f"guessed counter (0, 1 or the global step) would apply a different "
                f"optimizer under the same step count (ADR-0021). Escape: resume on the "
                f"commit that wrote it, or start a new ckpt_name and let the memory "
                f"retrain with the corrected formula."
            )
        counts = s["counts"]
        if tuple(counts.shape) != (m.host.counts.numel(),) or (counts.dtype is not torch.int32):
            loud(
                f"host counts {tuple(counts.shape)}:{counts.dtype} != "
                f"({m.host.counts.numel()},):torch.int32 — one int32 per table row. "
                f"Escape: the saved config's engram_rows/orders, or a new ckpt_name."
            )
        m.host.rows.copy_(s["rows"])
        m.host.momentum.copy_(s["momentum"])
        m.host.counts.copy_(counts)
        m.host.step = int(s["step"])


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
