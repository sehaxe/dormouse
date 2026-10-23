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

MEMORY. The state is the chunk ring: `2 · csa_kv_dim · 2 bytes` per chunk
(`k` and `v`, bf16), i.e.

    bytes/byte = 2 · csa_kv_dim · 2 / csa_chunk

and `state_bytes(context, ...)` gives the total. At `csa_kv_dim=64`,
`csa_chunk=8` that is 32 B/byte: 32 KiB at 1k, **1.0 MiB at 32k**, 32 MiB at
1M — per layer, per iteration slot. Compare KDA: a fixed 393 KB regardless of
context. The ring is append-only and grows by doubling; `DECODE_MAX_BYTES`
caps the context at 1M bytes for every chunk size and a decode step past the cap
raises LOUDLY rather than dropping the oldest entries — silently forgetting is
the one thing this branch exists to prevent.

COST, HONESTLY. Prefill is `O(t · t / chunk)` scores; with `topk << t/chunk`
the *attention* is `O(t · topk)`, but the SELECTION is still a full scan. Decode
is `O(t / chunk)` per byte for scoring and top-k: at 1M context that is 131 072
chunk keys × `csa_kv_dim` per position per iteration slot per layer (8.4M MAC +
131k comparisons at `kv=64`), against KDA's O(1). So the arm is affordable at
the 4k-32k windows this model targets and is NOT an O(1) decoder at 1M; the
`--longctx` driver in `scripts/ab_csa.py` is the instrument that measures the
long-window gain before anyone pays that decode bill.

REFUSALS (all in `config.validate` / `LoopBlock`): `use_msa` (two sparse branches
is not measured), `segment_len>0` (what a segment cut does to a half-full chunk
is undecided), `use_byteflow` (its loop runs over K patch latents, so a
`csa_chunk` of bytes would silently be a constant chunk of latents — refused
rather than repurposed), `use_kda=false`, `attn_kind='full'`, a chunk outside
{4,8,16,32}, a width outside {64,128}. `csa_topk > t` is legal at runtime: the
selection is `min(topk, available chunks)`.
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
    "memory_table",
    "state_bytes",
]

#: The chunk sizes the config accepts. A chunk of 1 would be exact per-byte
#: attention (the parity gate uses it), but it is not a compression and is not
#: a legal run value.
CHUNKS = (4, 8, 16, 32)
KV_DIMS = (64, 128)
#: Two bytes per element — the ring's dtype and the accounting's own unit.
STATE_BYTES = 2
#: The decode ring is capped at 1 MiB of CONTEXT for every chunk size (so the
#: same 1M-byte claim holds at 4/8/16/32); past it the state refuses to grow.
DECODE_MAX_BYTES = 1 << 20


def bytes_per_byte(csa_kv_dim: int, csa_chunk: int) -> float:
    """`2 · csa_kv_dim · 2 / csa_chunk` — the state's bytes per context byte."""
    assert csa_chunk >= 1 and csa_kv_dim >= 1, (csa_chunk, csa_kv_dim)
    return 2 * csa_kv_dim * STATE_BYTES / csa_chunk


def state_bytes(
    context: int, csa_kv_dim: int, csa_chunk: int, *, batch: int = 1
) -> int:
    """The chunk ring's exact bytes for `context` bytes: `ceil(ctx/chunk)` entries."""
    assert context >= 0 and batch >= 1, (context, batch)
    chunks = -(-context // csa_chunk)
    return chunks * 2 * csa_kv_dim * STATE_BYTES * batch


def memory_table(
    *,
    csa_kv_dim: int = 64,
    csa_chunk: int = 8,
    batch: int = 1,
    contexts: tuple[int, ...] = (1024, 32768, 1_048_576),
) -> dict[int, int]:
    """Print (and return) the per-context state size — the arm's memory claim."""
    out = {c: state_bytes(c, csa_kv_dim, csa_chunk, batch=batch) for c in contexts}
    print(
        f"CSA memory per layer (kv={csa_kv_dim}, chunk={csa_chunk}, "
        f"batch={batch}): {bytes_per_byte(csa_kv_dim, csa_chunk):.1f} B/byte "
        f"(= 2·{csa_kv_dim}·2/{csa_chunk})"
    )
    for c, n in out.items():
        print(f"  {c:>9,d} bytes context -> {n / 2**20:8.3f} MiB")
    return out


class CSAState:
    """The chunk ring the decode path appends to — one per iteration slot.

    `k`/`v` are `[capacity, batch, csa_kv_dim]` bf16 (2 bytes each, so the
    memory table is the true tensor size). `sum`/`count` are the current,
    not-yet-full chunk's pool. `capacity` doubles from 64, which makes the
    power-of-two contexts in the memory table exact rather than upper bounds.
    A `reserve` past the cap (`DECODE_MAX_BYTES` of context) raises LOUDLY: the one loss this
    branch cannot accept is silently forgetting old chunks.
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
        self.max_chunks = (
            -(-DECODE_MAX_BYTES // csa_chunk) if max_chunks is None else max_chunks
        )
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
        new = min(64, self.max_chunks) if self.k.shape[0] == 0 else self.k.shape[0] * 2
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
    """

    def __init__(self, cfg, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        d, kv = cfg.d_model, cfg.csa_kv_dim
        self.chunk = cfg.csa_chunk
        self.topk = cfg.csa_topk
        self.kv_dim = kv
        self.scale = 1.0 / math.sqrt(kv)
        self.W_q = nn.Linear(d, kv, bias=False, device=device)
        self.W_k = nn.Linear(d, kv, bias=False, device=device)
        self.W_v = nn.Linear(d, kv, bias=False, device=device)
        self.W_o = nn.Linear(kv, d, bias=False, device=device)

    # -- the pooled keys/values of a whole window -------------------------
    def _pool(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`x [b,t,d]` → `k, v [b, ceil(t/chunk), kv]` (mean over each chunk)."""
        b, t, _ = x.shape
        pooled = F.avg_pool1d(
            x.transpose(1, 2), self.chunk, self.chunk, ceil_mode=True
        )  # [b, d, n]
        pooled = pooled.transpose(1, 2)  # [b, n, d]
        return self.W_k(pooled), self.W_v(pooled)

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
        vv = v.unsqueeze(1).expand(b, qn, n, self.kv_dim).gather(
            2, idx.unsqueeze(-1).expand(b, qn, k_eff, self.kv_dim)
        )
        sel_scores = scores.gather(2, idx)  # [b,Q,k_eff]
        sel_ok = allowed.expand(b, qn, n).gather(2, idx)
        w = torch.softmax(sel_scores, dim=-1) * sel_ok
        # An all-masked row (query 0, no earlier chunk) has zero weight: divide by
        # the clamped sum, not by softmax's own (which is uniform on garbage).
        w = w / w.sum(-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)
        return (w.unsqueeze(-1) * vv).sum(-2)

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """One-shot prefill: `x [b,t,d]` (the block's normed input) → `[b,t,d]`."""
        counters = Counters() if counters is None else counters
        counters.count("csa_forward")
        b, t, d = x.shape
        k, v = self._pool(x)
        q = self.W_q(x)
        pos_chunk = torch.arange(t, device=x.device) // self.chunk  # [t]
        key_chunk = torch.arange(k.shape[1], device=x.device)  # [n]
        allowed = key_chunk.view(1, 1, -1) < pos_chunk.view(1, t, 1)  # [1,t,n]
        return self.W_o(self._attend(q, k, v, allowed))

    def forward_state(
        self,
        x: torch.Tensor,
        state: CSAState | None,
        counters: Counters | None = None,
    ) -> tuple[torch.Tensor, CSAState]:
        """Prefill that also builds the decode ring (chunked prefill / prompt).

        A window may only start on a chunk boundary of the state's own timeline;
        a half-full chunk is carried by `step_state`, never by a second prefill
        window (its boundaries would move). The last partial chunk is carried in
        `sum`/`count` so the decode continues it exactly.
        """
        counters = Counters() if counters is None else counters
        counters.count("csa_forward")
        b, t, d = x.shape
        if state is None:
            state = CSAState(d, self.kv_dim, self.chunk, batch=b, device=x.device)
        assert (state.batch, state.kv_dim) == (b, self.kv_dim), (state.batch, b)
        if state.count != 0:
            loud(
                "CSA forward_state cannot start inside a half-full chunk: the "
                "prefill window's chunk boundaries would move, so the pooled key of "
                "the first chunk would differ from the byte-by-byte decode's. "
                "Escape: continue with step_state (decode), or start the prefill on "
                "a chunk boundary."
            )
        n_old = state.n
        n_full = t // self.chunk
        k, v = self._pool(x)
        # The last chunk is partial: it is a key for nothing in this window (no
        # position after it exists), so it is carried, not appended.
        if n_full:
            state.reserve(state.n + n_full)
            for i in range(n_full):
                state.append(k[:, i], v[:, i])
        if t % self.chunk:
            state.sum = state.sum + x[:, n_full * self.chunk :].sum(1)
            state.count = t - n_full * self.chunk
        # Score against every FINALISED chunk (the final chunk index is never
        # allowed for any position in this window, so including it is harmless).
        q = self.W_q(x)
        pos_chunk = torch.arange(t, device=x.device) // self.chunk + n_old
        key_chunk = torch.arange(state.n, device=x.device)
        allowed = key_chunk.view(1, 1, -1) < pos_chunk.view(1, t, 1)
        kk = state.k[: state.n].to(q.dtype).transpose(0, 1)  # [b,n,kv]
        vv = state.v[: state.n].to(q.dtype).transpose(0, 1)
        return self.W_o(self._attend(q, kk, vv, allowed)), state

    def step_state(
        self,
        x: torch.Tensor,
        state: CSAState | None,
        counters: Counters | None = None,
    ) -> tuple[torch.Tensor, CSAState]:
        """One byte: score the finished chunks, then add this byte to the pool.

        The order is the causality: a chunk containing the current position must
        not be selectable by it, so the append happens AFTER the read.
        """
        counters = Counters() if counters is None else counters
        counters.count("csa_step")
        b, t, d = x.shape
        assert t == 1, f"the decode step is one position, got {t}"
        if state is None:
            state = CSAState(d, self.kv_dim, self.chunk, batch=b, device=x.device)
        assert (state.batch, state.kv_dim) == (b, self.kv_dim), (state.batch, b)
        state.sum = state.sum + x[:, 0]
        state.count += 1
        q = self.W_q(x[:, 0])  # [b,kv]
        if state.n:
            kk = state.k[: state.n].to(q.dtype).transpose(0, 1)  # [b,n,kv]
            vv = state.v[: state.n].to(q.dtype).transpose(0, 1)
            allowed = torch.ones(1, 1, state.n, dtype=torch.bool, device=x.device)
            out = self._attend(q.unsqueeze(1), kk, vv, allowed)
        else:
            out = q.new_zeros(b, 1, self.kv_dim)
        if state.count == self.chunk:
            mean = state.sum / state.count
            state.append(self.W_k(mean), self.W_v(mean))
            state.sum = torch.zeros_like(state.sum)
            state.count = 0
        return self.W_o(out), state
