"""Compressed, selectable KV memory (CSA) — an ADDITIVE branch beside KDA.

WHY. KDA is a linear attention: its state is `H × head_dim × head_dim`, O(1) in
the context, and therefore holds **no per-token memory at all**. Long-context
recall degrades because a 393 KB state cannot store exact details. DeepSeek-V4.1
solved this with two branches: **CSA** (compress `m` tokens into one KV entry,
then select the top-k entries sparsely) and **HCA** (coarse, dense). This file
is the byte-scale CSA half: it gives the model exact access to the earlier part
of the window at a memory cost linear in context and a compute cost that stays
bounded by `csa_topk` per query.

WHAT IT IS, EXACTLY.

* The sequence is cut into `csa_chunk` byte positions (4/8/16/32).
* ONE `k` and ONE `v` per chunk, shared across heads (MQA):
  `k = mean_pool(x_chunk) @ W_k`, `v = mean_pool(x_chunk) @ W_v`, both
  `csa_kv_dim` wide (64/128). `x` is the block's RMS-normalised input — the
  same tensor KDA sees.
* A query position scores `q = x @ W_q` against the keys of **strictly earlier**
  chunks (its own chunk is never read, so no position can see itself or a
  future byte), keeps the top `csa_topk` of them, and runs exact softmax
  attention over those `k`/`v` entries. `W_o` projects back to `d_model`.
* The branch is added to KDA's output behind a learned per-iteration gate in
  `LoopBlock` (`csa_gate`, the `msa_gate` pattern), so `use_csa=false` never
  builds it and every preset's parameter tree is unchanged.

TWO LEVELS. `csa_coarse_chunk > 0` turns the same module into the pair
DeepSeek-V4.1 describes: the FINE level above plus a COARSE, DENSE level (HCA).

* The coarse level mean-pools `csa_coarse_chunk` bytes (64/128/256/512) into
  ONE `k`/`v` entry (`csa_coarse_kv_dim`, 64/128) under its own `W_k_c`/`W_v_c`
  and its own `W_q_c`, and each query attends **densely over every strictly
  earlier coarse entry** — no top-k. At 1M bytes a 256-byte chunk has 4096
  entries, so the decode read is a 4096-key attention instead of a 131 072-key
  scan: the coarse level is cheap because it attends over a FEW entries.
* `csa_coarse_chunk` must be a multiple of `csa_chunk`, so a coarse entry is a
  whole number of fine entries and the two pools see the same byte boundaries.
* `csa_fine=false` is the documented off-switch for the fine level (refused
  without a coarse level): the pair then reduces to the coarse dense arm alone,
  which is what the exact-attention parity gate checks.
* **The two levels' outputs are SUMMED, and the sum passes through the ONE
  existing `csa_gate`** per iteration slot in `LoopBlock` — one learned gate
  for the pair, not one per level. The coarse branch is built only when
  `csa_coarse_chunk > 0`, so `csa_coarse_chunk=0` keeps the parameter tree (and
  every pinned `state_dict` hash) byte-identical.

PARITY, HONESTLY. With `csa_chunk=1` and `csa_coarse_chunk=1` BOTH levels are
exact causal attention under their own projections. The branch then returns
`W_o(fine_exact) + W_o_c(coarse_exact)` — a SUM of two independently
parameterised exact attentions, which double-counts the past and is NOT one
exact attention. The parity gate therefore switches the fine level off
(`csa_fine=false`) and asserts that the coarse level ALONE equals strict causal
softmax attention to 1e-5.

MEMORY. Each level is a chunk ring: `2 · kv · 2 bytes` per chunk (`k` and `v`,
bf16), i.e.

    fine   bytes/byte = 2 · csa_kv_dim · 2 / csa_chunk
    coarse bytes/byte = 2 · csa_coarse_kv_dim · 2 / csa_coarse_chunk

and `state_bytes` / `coarse_state_bytes` / `total_state_bytes` give the totals.
At `csa_kv_dim=64`, `csa_chunk=8` the fine ring is 32 B/byte: 32 KiB at 1k,
**1.0 MiB at 32k**, 32 MiB at 1M — per layer, per iteration slot. The coarse
level at 64/256 adds 1 B/byte: **1.0 MiB at 1M** (1.05 MB). Compare KDA: a fixed
393 KB regardless of context. Each ring is append-only and grows by doubling;
`DECODE_MAX_BYTES` caps the context at 1M bytes for every chunk size and a
decode step past the cap raises LOUDLY rather than dropping the oldest entries
— silently forgetting is the one thing this branch exists to prevent.

COST, HONESTLY. Prefill is `O(t · t / chunk)` scores; with `topk << t/chunk`
the *attention* is `O(t · topk)`, but the SELECTION is still a full scan. Decode
is `O(t / chunk)` per byte for scoring and top-k: at 1M context that is 131 072
chunk keys × `csa_kv_dim` per position per iteration slot per layer (8.4M MAC +
131k comparisons at `kv=64`), against KDA's O(1). The coarse level replaces
that scan with `2 · (t / csa_coarse_chunk) · csa_coarse_kv_dim` MAC/byte —
**0.52M MAC/byte at 256/64**, 4096 dense entries per byte — so it is the level
that can decode at 1M while the fine level is the one that can afford to be
selective. `decode_cost_table()` prints all three bills; the `--longctx` driver
in `scripts/ab_csa.py` is the instrument that measures the long-window gain
before anyone pays them.

REFUSALS (all in `config.validate` / `LoopBlock`): `use_msa` (two sparse branches
is not measured), `segment_len>0` (what a segment cut does to a half-full chunk
is undecided), `use_byteflow` (its loop runs over K patch latents, so a
`csa_chunk` of bytes would silently be a constant chunk of latents — refused
rather than repurposed), `use_kda=false`, `attn_kind='full'`, a chunk outside
{4,8,16,32}, a width outside {64,128}, a coarse chunk outside {64,128,256,512},
a coarse chunk that is not a multiple of `csa_chunk`, a coarse width outside
{64,128}, coarse without `use_csa`, coarse with `use_msa`, coarse with
`segment_len>0`, and `csa_fine=false` without a coarse level. `csa_topk > t` is
legal at runtime: the selection is `min(topk, available chunks)`; `csa_topk=0`
is legal only when the fine level is off.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import Counters, loud

__all__ = [
    "CSAState",
    "CompressedSelectiveAttention",
    "bytes_per_byte",
    "coarse_state_bytes",
    "decode_cost_table",
    "mac_per_byte",
    "memory_table",
    "state_bytes",
    "total_state_bytes",
]

#: The chunk sizes the config accepts. A chunk of 1 would be exact per-byte
#: attention (the parity gate uses it), but it is not a compression and is not
#: a legal run value.
CHUNKS = (4, 8, 16, 32)
KV_DIMS = (64, 128)
#: The coarse (HCA) level's chunk sizes. 256 is the measured 1M point: 4096
#: dense entries per 1M bytes, 1.0 MiB of state at kv=64.
COARSE_CHUNKS = (64, 128, 256, 512)
COARSE_KV_DIMS = (64, 128)
#: Two bytes per element — the rings' dtype and the accounting's own unit.
STATE_BYTES = 2
#: The decode rings are capped at 1 MiB of CONTEXT for every chunk size (so the
#: same 1M-byte claim holds at 4/8/16/32 and at 64/128/256/512); past it a ring
#: refuses to grow.
DECODE_MAX_BYTES = 1 << 20
#: The coarse level's decode bill must stay under this many MAC/byte at 1M
#: (`decode_cost_table` asserts it) — the reason the level exists.
COARSE_MAC_BUDGET = 3_000_000


def bytes_per_byte(csa_kv_dim: int, csa_chunk: int) -> float:
    """`2 · csa_kv_dim · 2 / csa_chunk` — one level's bytes per context byte."""
    assert csa_chunk >= 1 and csa_kv_dim >= 1, (csa_chunk, csa_kv_dim)
    return 2 * csa_kv_dim * STATE_BYTES / csa_chunk


def state_bytes(context: int, csa_kv_dim: int, csa_chunk: int, *, batch: int = 1) -> int:
    """One ring's exact bytes for `context` bytes: `ceil(ctx/chunk)` entries."""
    assert context >= 0 and batch >= 1, (context, batch)
    chunks = -(-context // csa_chunk)
    return chunks * 2 * csa_kv_dim * STATE_BYTES * batch


def coarse_state_bytes(
    context: int, csa_coarse_kv_dim: int, csa_coarse_chunk: int, *, batch: int = 1
) -> int:
    """The coarse ring's exact bytes: `ceil(ctx/coarse_chunk)` entries."""
    return state_bytes(context, csa_coarse_kv_dim, csa_coarse_chunk, batch=batch)


def total_state_bytes(
    context: int,
    csa_kv_dim: int,
    csa_chunk: int,
    *,
    csa_coarse_kv_dim: int = 64,
    csa_coarse_chunk: int = 0,
    batch: int = 1,
) -> int:
    """Fine + coarse ring bytes (`coarse_chunk=0` is the fine-only arm)."""
    total = state_bytes(context, csa_kv_dim, csa_chunk, batch=batch)
    if csa_coarse_chunk:
        total += coarse_state_bytes(context, csa_coarse_kv_dim, csa_coarse_chunk, batch=batch)
    return total


def memory_table(
    *,
    csa_kv_dim: int = 64,
    csa_chunk: int = 8,
    csa_coarse_kv_dim: int = 64,
    csa_coarse_chunk: int = 0,
    batch: int = 1,
    contexts: tuple[int, ...] = (1024, 32768, 1_048_576),
) -> dict[int, int]:
    """Print (and return) the per-context state size — the arm's memory claim.

    With a coarse level the table prints the FINE row, the COARSE row and their
    SUM per context, and returns the sums; with `csa_coarse_chunk=0` it prints
    and returns the fine row exactly as before.
    """
    fine = {c: state_bytes(c, csa_kv_dim, csa_chunk, batch=batch) for c in contexts}
    print(
        f"CSA fine memory per layer (kv={csa_kv_dim}, chunk={csa_chunk}, "
        f"batch={batch}): {bytes_per_byte(csa_kv_dim, csa_chunk):.1f} B/byte "
        f"(= 2·{csa_kv_dim}·2/{csa_chunk})"
    )
    for c, n in fine.items():
        print(f"  {c:>9,d} bytes context -> {n / 2**20:8.3f} MiB  (fine)")
    if not csa_coarse_chunk:
        return fine
    coarse = {
        c: coarse_state_bytes(c, csa_coarse_kv_dim, csa_coarse_chunk, batch=batch) for c in contexts
    }
    total = {c: fine[c] + coarse[c] for c in contexts}
    print(
        f"CSA coarse memory per layer (kv={csa_coarse_kv_dim}, "
        f"chunk={csa_coarse_chunk}, batch={batch}): "
        f"{bytes_per_byte(csa_coarse_kv_dim, csa_coarse_chunk):.1f} B/byte "
        f"(= 2·{csa_coarse_kv_dim}·2/{csa_coarse_chunk})"
    )
    for c, n in coarse.items():
        print(f"  {c:>9,d} bytes context -> {n / 2**20:8.3f} MiB  (coarse)")
    for c, n in total.items():
        print(
            f"  {c:>9,d} bytes context -> {n / 2**20:8.3f} MiB  "
            f"(fine + coarse = {n / 2**20:.3f} MiB)"
        )
    return total


def mac_per_byte(
    context: int,
    *,
    csa_kv_dim: int = 64,
    csa_chunk: int = 8,
    csa_topk: int = 32,
    csa_coarse_kv_dim: int = 64,
    csa_coarse_chunk: int = 0,
) -> dict[str, float]:
    """Decode MAC per context byte for each level, honestly counted.

    Per query position and per level: one `q·k` MAC per entry per KV dimension
    (the score) plus one multiply-accumulate per selected entry per dimension
    (the weighted value), i.e. `fine = n·kv + 2·min(topk, n)·kv` and
    `coarse = 2·m·kv_c`, where `n = ceil(t/chunk)` and
    `m = ceil(t/coarse_chunk)`. Projections and the softmax normalization are
    not counted (they are `O(d·kv)` and `O(entries)` respectively, below the
    attention bill at these widths). This is per layer, per iteration slot.
    """
    assert context >= 0 and csa_chunk >= 1, (context, csa_chunk)
    n = -(-context // csa_chunk)
    fine = float(n * csa_kv_dim + 2 * min(csa_topk, n) * csa_kv_dim)
    coarse = 0.0
    if csa_coarse_chunk:
        m = -(-context // csa_coarse_chunk)
        coarse = float(2 * m * csa_coarse_kv_dim)
    return {"fine": fine, "coarse": coarse, "both": fine + coarse}


def decode_cost_table(
    context: int = 1_048_576,
    *,
    csa_kv_dim: int = 64,
    csa_chunk: int = 8,
    csa_topk: int = 32,
    csa_coarse_kv_dim: int = 64,
    csa_coarse_chunk: int = 256,
    budget: int = COARSE_MAC_BUDGET,
) -> dict[str, float]:
    """Print the 1M decode bill for {fine only, coarse only, both} and ASSERT it.

    The coarse level is the whole reason the arm can reach 1M: it must stay
    under `budget` MAC/byte there. A regression that makes it scan the context
    (or drop the chunk multiple, effectively selecting every entry) reddens
    this assert.
    """
    bill = mac_per_byte(
        context,
        csa_kv_dim=csa_kv_dim,
        csa_chunk=csa_chunk,
        csa_topk=csa_topk,
        csa_coarse_kv_dim=csa_coarse_kv_dim,
        csa_coarse_chunk=csa_coarse_chunk,
    )
    assert bill["coarse"] < budget, (
        f"coarse decode at {context:,} B is {bill['coarse']:,.0f} MAC/byte "
        f"(budget {budget:,}) at chunk={csa_coarse_chunk}, kv={csa_coarse_kv_dim}: "
        f"the dense level has stopped being cheap — check the chunk/width."
    )
    print(
        f"CSA decode MAC/byte at {context:,} B (kv={csa_kv_dim}, chunk={csa_chunk}, "
        f"topk={csa_topk}; coarse kv={csa_coarse_kv_dim}, "
        f"chunk={csa_coarse_chunk}):"
    )
    print(f"  fine only   {bill['fine']:>12,.0f}")
    print(f"  coarse only {bill['coarse']:>12,.0f}")
    print(f"  both        {bill['both']:>12,.0f}  (budget {budget:,})")
    return bill


class CSAState:
    """The chunk ring the decode path appends to — one per iteration slot.

    `k`/`v` are `[capacity, batch, csa_kv_dim]` bf16 (2 bytes each, so the
    memory table is the true tensor size). `sum`/`count` are the current,
    not-yet-full chunk's pool. `capacity` doubles from the smallest power of two
    that holds the request (capped at 64), which makes the power-of-two contexts
    in the memory table exact rather than upper bounds — including the coarse
    level's 4 entries at 1k. A `reserve` past the cap (`DECODE_MAX_BYTES` of
    context) raises LOUDLY: the one loss this branch cannot accept is silently
    forgetting old chunks.
    """

    def __init__(
        self,
        d_model: int,
        csa_kv_dim: int,
        csa_chunk: int,
        *,
        batch: int = 1,
        device: torch.device | str = "cpu",
        max_chunks: int | None = None,
    ) -> None:
        self.kv_dim = csa_kv_dim
        self.chunk = csa_chunk
        self.batch = batch
        self.max_chunks = -(-DECODE_MAX_BYTES // csa_chunk) if max_chunks is None else max_chunks
        self.device = device
        self.k = torch.empty(0, batch, csa_kv_dim, dtype=torch.bfloat16, device=device)
        self.v = torch.empty_like(self.k)
        #: The number of FINALISED chunks (the ring's visible length).
        self.n = 0
        self.sum = torch.zeros(batch, d_model, dtype=torch.float32, device=device)
        self.count = 0

    def reserve(self, chunks: int) -> None:
        """Grow to at least `chunks` entries, doubling; LOUD past the cap."""
        if chunks <= self.k.shape[0]:
            return
        if self.k.shape[0] == 0:
            new = 1
            while new < chunks and new < 64:
                new *= 2
            new = min(new, self.max_chunks)
        else:
            new = self.k.shape[0] * 2
        while new < chunks:
            new *= 2
        if new > self.max_chunks:
            loud(
                f"CSA decode state cannot grow to {new} chunks: the cap is "
                f"{self.max_chunks} ({self.max_chunks * self.chunk:,} bytes of context "
                f"at csa_chunk={self.chunk}). The ring is append-only on purpose — "
                f"overwriting the oldest chunk would silently forget the start of the "
                f"window. Escape: a shorter context, a larger csa_chunk, or the "
                f"one-shot prefill forward."
            )
        k = torch.zeros(new, self.batch, self.kv_dim, dtype=self.k.dtype, device=self.k.device)
        v = torch.zeros_like(k)
        k[: self.n] = self.k[: self.n]
        v[: self.n] = self.v[: self.n]
        self.k, self.v = k, v

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append one finalised chunk: `k`, `v` are `[batch, csa_kv_dim]`."""
        assert k.shape == (self.batch, self.kv_dim), (k.shape, self.batch, self.kv_dim)
        self.reserve(self.n + 1)
        self.k[self.n] = k.to(self.k.dtype)
        self.v[self.n] = v.to(self.v.dtype)
        self.n += 1

    def bytes(self) -> int:
        """The ACTUAL allocated ring bytes (k + v), what the memory claim is about."""
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()


class CompressedSelectiveAttention(nn.Module):
    """Chunk-pooled MQA keys/values + top-k selection + exact attention.

    One call site per path, the same shape as the other arms: `forward` is the
    one-shot prefill, `forward_state` returns the ring after a prompt,
    `step_state` is the one-position decode step.

    With `cfg.csa_coarse_chunk > 0` the module runs TWO levels and returns
    `W_o(fine) + W_o_c(coarse)`, which `LoopBlock` passes through the ONE
    shared `csa_gate`. A coarse run carries `(fine_state, coarse_state)` (the
    fine half is `None` when `csa_fine=false`); a fine-only run carries the
    plain `CSAState`, so `csa_coarse_chunk=0` keeps every existing caller and
    every pinned hash unchanged. `coarse_enabled` is a runtime switch for the
    longctx driver's with/without-coarse scoring; it never changes the graph
    when left at its default `True`.
    """

    def __init__(self, cfg, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        d, kv = cfg.d_model, cfg.csa_kv_dim
        self.chunk = cfg.csa_chunk
        self.topk = cfg.csa_topk
        self.kv_dim = kv
        self.scale = 1.0 / math.sqrt(kv)
        self.fine = bool(getattr(cfg, "csa_fine", True))
        self.coarse_chunk = int(getattr(cfg, "csa_coarse_chunk", 0))
        self.coarse_kv_dim = int(getattr(cfg, "csa_coarse_kv_dim", 64))
        assert self.fine or self.coarse_chunk, (
            "CSA built with the fine level off and no coarse level: the branch "
            "would compute nothing."
        )
        assert self.coarse_chunk == 0 or self.coarse_chunk % self.chunk == 0, (
            f"csa_coarse_chunk={self.coarse_chunk} is not a multiple of "
            f"csa_chunk={self.chunk}: a coarse entry must be a whole number of "
            f"fine entries or the two levels would pool different byte boundaries."
        )
        if self.fine:
            self.W_q = nn.Linear(d, kv, bias=False, device=device)
            self.W_k = nn.Linear(d, kv, bias=False, device=device)
            self.W_v = nn.Linear(d, kv, bias=False, device=device)
            self.W_o = nn.Linear(kv, d, bias=False, device=device)
        if self.coarse_chunk:
            ckv = self.coarse_kv_dim
            self.coarse_scale = 1.0 / math.sqrt(ckv)
            self.W_q_c = nn.Linear(d, ckv, bias=False, device=device)
            self.W_k_c = nn.Linear(d, ckv, bias=False, device=device)
            self.W_v_c = nn.Linear(d, ckv, bias=False, device=device)
            self.W_o_c = nn.Linear(ckv, d, bias=False, device=device)
            #: Runtime only (the longctx with/without table); not a parameter.
            self.coarse_enabled = True

    # -- the pooled keys/values of a whole window -------------------------
    def _mean_pool(self, x: torch.Tensor, chunk: int) -> torch.Tensor:
        """`x [b,t,d]` → `mean [b, ceil(t/chunk), d]` (mean over each chunk)."""
        pooled = F.avg_pool1d(x.transpose(1, 2), chunk, chunk, ceil_mode=True)  # [b, d, n]
        return pooled.transpose(1, 2)  # [b, n, d]

    def _pool(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`x [b,t,d]` → fine `k, v [b, ceil(t/chunk), kv]`."""
        pooled = self._mean_pool(x, self.chunk)
        return self.W_k(pooled), self.W_v(pooled)

    def _pool_coarse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`x [b,t,d]` → coarse `k, v [b, ceil(t/coarse_chunk), kv]`."""
        pooled = self._mean_pool(x, self.coarse_chunk)
        return self.W_k_c(pooled), self.W_v_c(pooled)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        """Top-k exact attention. `q [b,Q,kv]`, `k/v [b,N,kv]`, `allowed [1,Q,N]`."""
        b, qn, _ = q.shape
        n = k.shape[1]
        k_eff = min(self.topk, n)
        if k_eff == 0 or qn == 0:
            return q.new_zeros(b, qn, self.kv_dim)
        scores = (q @ k.transpose(1, 2)) * self.scale  # [b,Q,N]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        idx = scores.topk(k_eff, dim=-1).indices  # [b,Q,k_eff]
        vv = (
            v.unsqueeze(1)
            .expand(b, qn, n, self.kv_dim)
            .gather(2, idx.unsqueeze(-1).expand(b, qn, k_eff, self.kv_dim))
        )
        sel_scores = scores.gather(2, idx)  # [b,Q,k_eff]
        sel_ok = allowed.expand(b, qn, n).gather(2, idx)
        w = torch.softmax(sel_scores, dim=-1) * sel_ok
        # An all-masked row (query 0, no earlier chunk) has zero weight: divide by
        # the clamped sum, not by softmax's own (which is uniform on garbage).
        w = w / w.sum(-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)
        return (w.unsqueeze(-1) * vv).sum(-2)

    def _attend_dense(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        """Dense exact attention over EVERY entry. `allowed [1,Q,N]` masks chunks."""
        b, qn, _ = q.shape
        n = k.shape[1]
        if n == 0 or qn == 0:
            return q.new_zeros(b, qn, self.coarse_kv_dim)
        scores = (q @ k.transpose(1, 2)) * self.coarse_scale  # [b,Q,N]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        w = torch.softmax(scores, dim=-1) * allowed
        w = w / w.sum(-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)
        return (w.unsqueeze(-1) * v.unsqueeze(1)).sum(-2)

    def _coarse_on(self) -> bool:
        return self.coarse_chunk > 0 and getattr(self, "coarse_enabled", True)

    def _split_state(self, state):
        """Normalise `None | CSAState | (fine_state, coarse_state)` by arm."""
        if not self.coarse_chunk:
            return state, None
        if state is None:
            return None, None
        assert isinstance(state, tuple) and len(state) == 2, (
            f"a two-level CSA state is (fine_state, coarse_state), got {type(state)}"
        )
        return state

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """One-shot prefill: `x [b,t,d]` (the block's normed input) → `[b,t,d]`.

        Fine top-k and coarse dense run on the same input; their projected
        outputs are summed here and the SINGLE shared `csa_gate` (in
        `LoopBlock`) scales the pair. When both levels are exact this is a sum
        of two exact attentions, i.e. the past is deliberately double-counted;
        the parity gate therefore exercises the coarse level alone.
        """
        counters = Counters() if counters is None else counters
        counters.count("csa_forward")
        b, t, _d = x.shape
        out = None
        if self.fine:
            k, v = self._pool(x)
            q = self.W_q(x)
            pos_chunk = torch.arange(t, device=x.device) // self.chunk  # [t]
            key_chunk = torch.arange(k.shape[1], device=x.device)  # [n]
            allowed = key_chunk.view(1, 1, -1) < pos_chunk.view(1, t, 1)  # [1,t,n]
            out = self.W_o(self._attend(q, k, v, allowed))
        if self._coarse_on():
            counters.count("csa_coarse_forward")
            ck, cv = self._pool_coarse(x)
            cq = self.W_q_c(x)
            pos_coarse = torch.arange(t, device=x.device) // self.coarse_chunk
            key_coarse = torch.arange(ck.shape[1], device=x.device)
            allowed_c = key_coarse.view(1, 1, -1) < pos_coarse.view(1, t, 1)
            cout = self.W_o_c(self._attend_dense(cq, ck, cv, allowed_c))
            out = cout if out is None else out + cout
        assert out is not None, "CSA has no active level: fine off and coarse disabled"
        return out

    def forward_state(
        self,
        x: torch.Tensor,
        state: CSAState | tuple[CSAState | None, CSAState | None] | None,
        counters: Counters | None = None,
    ) -> tuple[torch.Tensor, CSAState | tuple[CSAState | None, CSAState | None]]:
        """Prefill that also builds the decode ring (chunked prefill / prompt).

        A window may only start on a chunk boundary of the state's own timeline;
        a half-full chunk is carried by `step_state`, never by a second prefill
        window (its boundaries would move). The last partial chunk is carried in
        `sum`/`count` so the decode continues it exactly. With a coarse level the
        same rule holds for the coarse ring and the state is a
        `(fine_state, coarse_state)` pair.
        """
        counters = Counters() if counters is None else counters
        counters.count("csa_forward")
        b, t, d = x.shape
        fs, cs = self._split_state(state)
        out = None
        if self.fine:
            if fs is None:
                fs = CSAState(d, self.kv_dim, self.chunk, batch=b, device=x.device)
            assert (fs.batch, fs.kv_dim) == (b, self.kv_dim), (fs.batch, b)
            if fs.count != 0:
                loud(
                    "CSA forward_state cannot start inside a half-full chunk: the "
                    "prefill window's chunk boundaries would move, so the pooled key of "
                    "the first chunk would differ from the byte-by-byte decode's. "
                    "Escape: continue with step_state (decode), or start the prefill on "
                    "a chunk boundary."
                )
            n_old = fs.n
            n_full = t // self.chunk
            k, v = self._pool(x)
            # The last chunk is partial: it is a key for nothing in this window (no
            # position after it exists), so it is carried, not appended.
            if n_full:
                fs.reserve(fs.n + n_full)
                for i in range(n_full):
                    fs.append(k[:, i], v[:, i])
            if t % self.chunk:
                fs.sum = fs.sum + x[:, n_full * self.chunk :].sum(1)
                fs.count = t - n_full * self.chunk
            # Score against every FINALISED chunk (the final chunk index is never
            # allowed for any position in this window, so including it is harmless).
            q = self.W_q(x)
            pos_chunk = torch.arange(t, device=x.device) // self.chunk + n_old
            key_chunk = torch.arange(fs.n, device=x.device)
            allowed = key_chunk.view(1, 1, -1) < pos_chunk.view(1, t, 1)
            kk = fs.k[: fs.n].to(q.dtype).transpose(0, 1)  # [b,n,kv]
            vv = fs.v[: fs.n].to(q.dtype).transpose(0, 1)
            out = self.W_o(self._attend(q, kk, vv, allowed))
        if self.coarse_chunk:
            if cs is None:
                cs = CSAState(d, self.coarse_kv_dim, self.coarse_chunk, batch=b, device=x.device)
            assert (cs.batch, cs.kv_dim) == (b, self.coarse_kv_dim), (cs.batch, b)
            if cs.count != 0:
                loud(
                    "CSA forward_state cannot start inside a half-full COARSE chunk: "
                    "the pooled entry's byte boundaries would move. Escape: continue "
                    "with step_state (decode), or start the prefill on a coarse-chunk "
                    "boundary."
                )
            n_old_c = cs.n
            n_full_c = t // self.coarse_chunk
            ck, cv = self._pool_coarse(x)
            if n_full_c:
                cs.reserve(cs.n + n_full_c)
                for i in range(n_full_c):
                    cs.append(ck[:, i], cv[:, i])
            if t % self.coarse_chunk:
                cs.sum = cs.sum + x[:, n_full_c * self.coarse_chunk :].sum(1)
                cs.count = t - n_full_c * self.coarse_chunk
            if self._coarse_on():
                counters.count("csa_coarse_forward")
                cq = self.W_q_c(x)
                pos_coarse = torch.arange(t, device=x.device) // self.coarse_chunk + n_old_c
                key_coarse = torch.arange(cs.n, device=x.device)
                allowed_c = key_coarse.view(1, 1, -1) < pos_coarse.view(1, t, 1)
                ckk = cs.k[: cs.n].to(cq.dtype).transpose(0, 1)
                cvv = cs.v[: cs.n].to(cq.dtype).transpose(0, 1)
                cout = self.W_o_c(self._attend_dense(cq, ckk, cvv, allowed_c))
                out = cout if out is None else out + cout
        new_state = (fs, cs) if self.coarse_chunk else fs
        assert out is not None, "CSA has no active level: fine off and coarse disabled"
        return out, new_state

    def step_state(
        self,
        x: torch.Tensor,
        state: CSAState | tuple[CSAState | None, CSAState | None] | None,
        counters: Counters | None = None,
    ) -> tuple[torch.Tensor, CSAState | tuple[CSAState | None, CSAState | None]]:
        """One byte: score the finished chunks, then add this byte to the pool.

        The order is the causality: a chunk containing the current position must
        not be selectable by it, so the append happens AFTER the read. Both
        levels follow it; the coarse level reads its finished entries DENSELY.
        """
        counters = Counters() if counters is None else counters
        counters.count("csa_step")
        b, t, d = x.shape
        assert t == 1, f"the decode step is one position, got {t}"
        fs, cs = self._split_state(state)
        out = None
        if self.fine:
            if fs is None:
                fs = CSAState(d, self.kv_dim, self.chunk, batch=b, device=x.device)
            assert (fs.batch, fs.kv_dim) == (b, self.kv_dim), (fs.batch, b)
            fs.sum = fs.sum + x[:, 0]
            fs.count += 1
            q = self.W_q(x[:, 0])  # [b,kv]
            if fs.n:
                kk = fs.k[: fs.n].to(q.dtype).transpose(0, 1)  # [b,n,kv]
                vv = fs.v[: fs.n].to(q.dtype).transpose(0, 1)
                allowed = torch.ones(1, 1, fs.n, dtype=torch.bool, device=x.device)
                fine_out = self._attend(q.unsqueeze(1), kk, vv, allowed)
            else:
                fine_out = q.new_zeros(b, 1, self.kv_dim)
            if fs.count == self.chunk:
                mean = fs.sum / fs.count
                fs.append(self.W_k(mean), self.W_v(mean))
                fs.sum = torch.zeros_like(fs.sum)
                fs.count = 0
            out = self.W_o(fine_out)
        if self.coarse_chunk:
            if cs is None:
                cs = CSAState(d, self.coarse_kv_dim, self.coarse_chunk, batch=b, device=x.device)
            assert (cs.batch, cs.kv_dim) == (b, self.coarse_kv_dim), (cs.batch, b)
            cs.sum = cs.sum + x[:, 0]
            cs.count += 1
            cq = self.W_q_c(x[:, 0])  # [b,ckv]
            if cs.n:
                ckk = cs.k[: cs.n].to(cq.dtype).transpose(0, 1)
                cvv = cs.v[: cs.n].to(cq.dtype).transpose(0, 1)
                allowed_c = torch.ones(1, 1, cs.n, dtype=torch.bool, device=x.device)
                coarse_out = self._attend_dense(cq.unsqueeze(1), ckk, cvv, allowed_c)
            else:
                coarse_out = cq.new_zeros(b, 1, self.coarse_kv_dim)
            if cs.count == self.coarse_chunk:
                mean = cs.sum / cs.count
                cs.append(self.W_k_c(mean), self.W_v_c(mean))
                cs.sum = torch.zeros_like(cs.sum)
                cs.count = 0
            if self._coarse_on():
                counters.count("csa_coarse_forward")
                cout = self.W_o_c(coarse_out)
                out = cout if out is None else out + cout
        new_state = (fs, cs) if self.coarse_chunk else fs
        assert out is not None, "CSA has no active level: fine off and coarse disabled"
        return out, new_state
