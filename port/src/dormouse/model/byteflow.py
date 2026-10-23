"""ByteFlow Net (arXiv 2603.03583) — a port of `crates/burn-byteflow`.

Line-by-line source of every formula here, the five stages of `src/net.rs:1-5`:

1. **local encoder** — E pre-norm causal blocks: sliding-window attention
   (window `w_local`) + **Canon layers** (causal depthwise conv k=4 with
   per-channel gates, Allen-Zhu 2025, eq. 10) + SwiGLU FFN (`net.rs:209-248`);
2. **downsampling** — coding-rate chunking: marginal gains
   `ΔR_t = R(h_1:t) − R(h_1:t−1)`, Top-K selection with forced BOS position,
   chronological order, projection to `d_global` (`net.rs:438-461`, `chunk.rs`);
3. **global transformer** — G causal blocks on the K compressed tokens
   (`net.rs:428`);
4. **upsampling** — multi-linear reconstruction with B=16 shared bins and a
   large residual: `s_t = h_t + g_chunk(t)·W_bin(t)` (`net.rs:475-526`);
5. **decoder** — symmetric to the local encoder, projects to next-byte logits
   (`net.rs:467-473`).

Coding rate: the exact lossy form `R_ε(h) = ½·logdet(I + (d/ε²)·H·Hᵀ)` (host
Cholesky, analysis/validation — `chunk.rs:79-112`) and the DEFAULT L2 streaming
approximation `R ∝ ‖H‖₂` from Appendix B (`chunk.rs:119-131`). The streaming
patcher is ported from `src/stream.rs` as [`RatePatcher`], one window of state.

# Integration: the design decision (Option A / B / C)

The net is only half the question; the other half is where it sits in a
byte-level LM. Three options were on the table:

* **A — the net REPLACES the stack**: train [`ByteFlowNet`] alone,
  `net.forward(bytes) -> [b, t, 256]`. One model, no glue, and it is the
  configuration the paper's own ablations measure — at the cost of every loop
  arm (attention, memory, experts) and every knob they carry.
* **B — front/back stages AROUND our loop (chosen)**: the front stage
  (byte embedding → local encoder → coding-rate chunker → `proj`) compresses
  the T-byte window into K patch latents; **our** weight-shared loop — with its
  attention, memory and expert arms, unchanged — runs on those K positions; the
  back stage (multi-linear upsample with large residual → decoder blocks →
  byte head) lifts the K latents back to T byte logits. Two plain `Linear`s
  bridge `d_global ↔ d_model`; nothing else is added. The loop then pays for K
  patches instead of T bytes, which is the whole point of the architecture
  (~3-5x compression; the burn A/B measured the compression itself as the win —
  canon at equal params 3.466 vs plain bytes 3.947 BPB, and the full-arm
  dormouse `small` LOSES at 6.498, `docs/reviews/byteflow-ab-2026-10-02.md`).
* **C — the rate as an auxiliary score**: keep the byte model and add `ΔR` as
  an extra loss/regularizer. Rejected: what the paper buys is the compressed
  SEQUENCE, not a score; an aux term leaves the loop paying for every byte.

We ship **B when any loop arm is on**, and **A when none is** — "byteflow + our
arms" and "byteflow alone" are the two configurations that mean something, and
A builds the net without a dormant parameter tree beside it. Which one ran is
COUNTED (`bf_channel` / `bf_net` on the step's counter line), and a
`use_byteflow` config whose arms the channel does not feed is refused loudly
(`config.validate`) rather than silently ignored — an arm that runs and trains
nothing is exactly the defect the counters exist to make impossible.

# Precision: bf16 matmuls, fp32 masters, three fp32 islands

**All matmuls run in bf16 with fp32 masters** — the loop's policy, not a second
one: every projection is a [`_MixedLinear`] over an fp32 `nn.Parameter`, and
[`mm_mixed`] casts masters and activation per forward so the optimizer and the
checkpoint never see a bf16 tensor. The residual stream (`h`, `z`, `s`) stays
fp32, so every add is the same expression at both dtypes and the matmul output
is the only place a dtype is chosen. No `torch.autocast`: an autocast region
would be a silent cast copy around every projection, which is the failure mode
burn's `--bf16` refusal named (`byteflow.rs:85-89`) and the one the explicit
casts exist to rule out. burn's refusal is a property of that crate, not of this
math: the port has no burn-side reason to keep the net fp32.

Three islands stay fp32, and each is an explicit cast:

1. **The chunker's selection** (`marginal_gains_l2` → `select_positions`).
   A bf16 top-k can flip a boundary on a tie, and the selection has NO
   gradient, so a flip is a discrete change with no learning signal to correct
   it. The gains and the top-k compute in fp32. Measured on the real corpus
   (2026-10-21, the first 128 KiB of `real/corpus.bin`): top-1 vs top-2 gains
   are equal within bf16 epsilon on **0.00 %** of rows, but the k-th vs
   (k+1)-th — the pair that actually decides the boundary — tie on **100 %**
   of them. With the island the bf16 net selects the same K positions as the
   fp32 net on **96.88 %** of 256 rows (smoke geometry) and **100 %** of 64
   rows (medium); `h` merely stored in bf16 with the widening intact gives
   **90.62 %**, and scoring the gains in bf16 — what the widening prevents —
   agrees on **0.00 %** of the 256 rows, i.e. every row a different selection.
   It is a no-op cast on the already-fp32 `h` today, and it costs one `to`.
2. **`coding_rate_exact` / the logdet Cholesky.** bf16 Cholesky is unstable, and
   `marginal_gains_exact` is the only caller. The default path is the L2 fast
   path, so this only matters with `byteflow_logdet = true`; `config.validate`
   refuses that pair loudly instead (`bf16 + byteflow_logdet`) rather than
   pricing a host factorization in bf16.
3. **Norm and Canon reductions.** The pre-norm LayerNorms and the Canon gate
   sums read the fp32 stream and stay there — the model's existing pattern
   (the burned rule "final norm in fp32", `model.rs`), with `CanonLayer` and
   `FlowBlock` asserting the fp32 stream rather than promoting silently.

Everything else — encoder blocks, decoder blocks, `proj`, `in_proj`/`out_proj`,
`upsample_w`, the byte head — runs its GEMM in bf16. Parity at `bf16 = false`
is `torch.equal` against the pre-bf16 expressions
(`tests/test_byteflow_bf16.py`), not a tolerance.

# What is NOT ported

The paper's unimplemented items stay unimplemented and unclaimed: there is no
"Multi-level" head split, no θ from Table 5 (the crate's own `TODO(бумага)`
notes at `net.rs:66-68`, `:265-271`, `:280` carry what the paper does not
state — RoPE base 500 000 is the hierarchical-family's, not BFlowNet's).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig
from ..instrument import Counters, loud
from .tsct import mm_mixed

__all__ = [
    "VOCAB",
    "BfChannel",
    "ByteFlowConfig",
    "ByteFlowNet",
    "CanonLayer",
    "FlowAttention",
    "FlowBlock",
    "RateMode",
    "RatePatcher",
    "SwiGLU",
    "bf_for_model",
    "byteflow_config",
    "coding_rate_exact",
    "compat_armed",
    "effective_bins",
    "marginal_gains_exact",
    "marginal_gains_l2",
    "select_positions",
]

#: Pure UTF-8 byte vocabulary. The BOS boundary is position 0 OF THE SEQUENCE,
#: not a vocabulary entry (`net.rs:16-18`).
VOCAB = 256


class RateMode(Enum):
    """Selection scoring mode (`chunk.rs:11-18`)."""

    #: Appendix B's streaming approximation `R ∝ ‖H‖₂` — the paper's default.
    L2 = "l2"
    #: Exact lossy rate (eq. 11) through the host Cholesky, analysis-only.
    LOGDET = "logdet"


@dataclass
class ByteFlowConfig:
    """The crate's `ByteFlowConfig` — `net.rs:253-310`, defaults and all.

    Defaults are BFlowNet-600M from paper Appendix C.2/Table 5 (layers `[6, 20]`,
    dims `[512, 1536]`, `8192 → 3200 → 8192`); the crate's `TODO(бумага)` notes
    name every value the paper does not state. The port carries them verbatim:
    `configs/*.toml` and every dormouse preset keep `use_byteflow = false`, so
    the port's ModelConfig defaults (`byteflow_*`) are the crate's own numbers
    and no preset changes net by omission. (burn's `schema.rs:751-794` instead
    ships the reduced `byteflow_9m` build — that is a burn-side preset choice,
    not the crate's default.)
    """

    d_local: int = 512
    d_global: int = 1536
    k_tokens: int = 3200
    e_layers: int = 6
    g_layers: int = 20
    n_heads_local: int = 8
    n_heads_global: int = 12
    w_local: int = 512
    d_ff_local: int = 1408
    d_ff_global: int = 4096
    bins: int = 16
    eps2: float = 0.5
    rate_mode: RateMode = RateMode.L2
    max_bytes: int = 8192
    rope_base: float = 500_000.0
    #: PORT-ONLY: the COMPUTE dtype of every matmul (`byteflow.py`'s precision
    #: rule). The crate has no bf16 path, so this field has no counterpart in
    #: `ByteFlowConfig::default()`; `byteflow_config` sets it from
    #: `ModelConfig.bf16`, and the fp32 default keeps every direct construction
    #: (the oracle tests) on the pre-bf16 path.
    dtype: torch.dtype = torch.float32


def byteflow_config(cfg: ModelConfig) -> ByteFlowConfig:
    """The dormouse config's byteflow half → the crate's config (`bf.rs:65-82`).

    `rope_base` is not a config field on either side: the crate default stands
    (`bf.rs:80`'s `..ByteFlowConfig::default()`). `dtype` is the port's own
    precision knob (`ModelConfig.bf16`); it moves the compute, never a master.
    """
    return ByteFlowConfig(
        d_local=cfg.byteflow_d_local,
        d_global=cfg.byteflow_d_global,
        k_tokens=cfg.byteflow_k_tokens,
        e_layers=cfg.byteflow_e_layers,
        g_layers=cfg.byteflow_g_layers,
        n_heads_local=cfg.byteflow_heads_local,
        n_heads_global=cfg.byteflow_heads_global,
        w_local=cfg.byteflow_w_local,
        d_ff_local=cfg.byteflow_d_ff_local,
        d_ff_global=cfg.byteflow_d_ff_global,
        bins=cfg.byteflow_bins,
        eps2=cfg.byteflow_eps2,
        rate_mode=RateMode.LOGDET if cfg.byteflow_logdet else RateMode.L2,
        max_bytes=cfg.byteflow_max_bytes,
        dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
    )


# ---------------------------------------------------------------------------
# stage 1 / 5 — the Canon layer and the local block
# ---------------------------------------------------------------------------


class _MixedLinear(nn.Module):
    """Bias-free `nn.Linear` whose GEMM runs in the net's COMPUTE dtype.

    The `weight` is an fp32 MASTER (the same `torch.empty` + Kaiming-uniform draw
    `nn.Linear` makes, so the dtype knob does not move a single initial value or
    an RNG draw), and `mm_mixed` casts it per forward. The output is the compute
    dtype: a caller that feeds an fp32 residual widens it at the boundary
    (`fp32_out`), a caller that feeds another bf16 matmul does not.

    `bias` is deliberately absent: the one biased projection is the byte head,
    which keeps `nn.Linear` so its fp32 op is bit-identical (`addmm` and
    `matmul + bias` are not — 1.5e-5 at `[512, 1024] → 256`).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str = "cpu",
        *,
        dtype: torch.dtype = torch.float32,
        fp32_out: bool = False,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.compute_dtype = dtype
        self.fp32_out = fp32_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = mm_mixed(x, self.weight, self.compute_dtype)
        return y.to(torch.float32) if self.fp32_out else y


class CanonLayer(nn.Module):
    """Causal depthwise conv k=4 with learned per-channel gates — `net.rs:20-62`.

    `Canon(h_t) = Σ_{k<4} w_k ⊙ h_{t−k}` (paper eq. 10). Identity init
    (w0 = 1, rest 0): the layer starts as a no-op so the pre-norm block trains
    stably from step one (`net.rs:31-34`).

    FP32 ISLAND: the gate sum reads the fp32 residual stream and stays there
    (`byteflow.py`'s precision rule, reason 3) — asserted, not promoted.
    """

    def __init__(self, d: int, device: torch.device | str = "cpu") -> None:
        super().__init__()
        gates = torch.zeros(4, d, device=device)
        gates[0] = 1.0
        self.gates = nn.Parameter(gates)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.float32, "Canon is an fp32 island (bf16 streams are a bug)"
        b, t, d = x.shape
        # Left-pad by 3 so the taps read h_{t−3}..h_t without state shifting
        # (`net.rs:44-46`); tap k is the window starting at 3−k (`net.rs:50-53`).
        padded = F.pad(x, (0, 0, 3, 0))
        acc = None
        for k in range(4):
            tap = padded[:, 3 - k : 3 - k + t] * self.gates[k].view(1, 1, d)
            acc = tap if acc is None else acc + tap
        return acc


class RopeTable(nn.Module):
    """Half-split rotary table, applied per head to q/k (`net.rs:69-124`).

    The table is DERIVED data (not state, `persistent=False`) built in fp32 and
    stored in the compute dtype: a bf16 `q` times an fp32 cosine would promote
    the pair to fp32 — a cast copy of `[b, t, 3·d]` on every forward, the exact
    thing the precision rule bans. At `dtype = fp32` this is today's table.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        base: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        hd = d_model // n_heads
        half = hd // 2
        angles = torch.empty(max_seq_len, half, dtype=torch.float32, device=device)
        for p in range(max_seq_len):
            for i in range(half):
                inv = base ** (-(2 * i) / hd)  # net.rs:89
                angles[p, i] = torch.tensor(p * inv, dtype=torch.float32).item()
        self.n_heads = n_heads
        # Non-parameters, derived from the config: they are not state.
        self.register_buffer("cos", angles.cos().to(dtype), persistent=False)
        self.register_buffer("sin", angles.sin().to(dtype), persistent=False)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        hd = d // self.n_heads
        half = hd // 2
        x4 = x.reshape(b, t, self.n_heads, hd)
        x1, x2 = x4[..., :half], x4[..., half:]
        c = self.cos[:t].reshape(1, t, 1, half)
        s = self.sin[:t].reshape(1, t, 1, half)
        # net.rs:120-122: the half-split rotation, o1 = x1·c − x2·s, o2 = x1·s + x2·c.
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).reshape(b, t, d)


def causal_mask(t: int, window: int | None, device: torch.device | str) -> torch.Tensor:
    """`true` = masked out: strictly future, plus the far past under a window.

    `net.rs:126-139`: the window keeps the last `window` positions INCLUDING the
    current one, i.e. position `j` is masked when `j > i` or `j ≤ i − w`.
    """
    i = torch.arange(t, device=device).view(t, 1)
    j = torch.arange(t, device=device).view(1, t)
    mask = j > i
    if window is not None:
        mask = mask | (j <= i - window)
    return mask


class FlowAttention(nn.Module):
    """Causal attention, optional sliding window, RoPE over a fused QKV (`net.rs:143-207`).

    The two projections run at the compute dtype (`qkv`, `out`), so RoPE and
    SDPA are bf16 too; the fp32 return is the arm boundary the residual add
    needs.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window: int | None,
        max_seq_len: int,
        rope_base: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.qkv = _MixedLinear(d_model, 3 * d_model, device, dtype=dtype)
        self.out = _MixedLinear(d_model, d_model, device, dtype=dtype, fp32_out=True)
        self.rope = RopeTable(d_model, n_heads, max_seq_len, rope_base, device, dtype)
        self.n_heads = n_heads
        self.window = window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.float32, "the norm/residual stream feeding attention is fp32"
        b, t, d = x.shape
        hd = d // self.n_heads
        qkv = self.qkv(x)
        q = self.rope.apply(qkv[..., :d]).reshape(b, t, self.n_heads, hd).transpose(1, 2)
        k = self.rope.apply(qkv[..., d : 2 * d]).reshape(b, t, self.n_heads, hd).transpose(1, 2)
        v = qkv[..., 2 * d :].reshape(b, t, self.n_heads, hd).transpose(1, 2)
        # SDPA, same math as the naive path this replaces (scale is SDPA's own
        # `1/sqrt(hd)`, causality is the same strictly-lower triangle) but fused:
        # no `[b, heads, t, t]` scores tensor and no `masked_fill` copy. At the
        # channel's shape (t=512, 8 heads, b=4) that materialization was 33 MB
        # per layer per direction; the window is 512 = t in every shipped
        # preset, so the fast path is the one that runs.
        if self.window is None or self.window >= t:
            mixed = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # `causal_mask` returns True = masked OUT; SDPA's bool mask is the
            # opposite convention (True = may attend).
            mixed = F.scaled_dot_product_attention(
                q, k, v, attn_mask=~causal_mask(t, self.window, x.device)
            )
        mixed = mixed.transpose(1, 2).reshape(b, t, d)
        return self.out(mixed)


class SwiGLU(nn.Module):
    """`x → Linear(d, 2h) → SiLU(left)·right → Linear(h, d)` (`burn-swiglu/src/lib.rs:17-31`)."""

    def __init__(
        self,
        d_model: int,
        hidden: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.gate_up = _MixedLinear(d_model, 2 * hidden, device, dtype=dtype)
        self.down = _MixedLinear(hidden, d_model, device, dtype=dtype, fp32_out=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.float32, "the norm feeding the FFN is fp32"
        gu = self.gate_up(x)
        gate, up = gu[..., : self.hidden], gu[..., self.hidden :]
        return self.down(F.silu(gate) * up)


class FlowBlock(nn.Module):
    """One pre-norm block (`net.rs:209-248`): `ĥ = Canon(h + Attn(LN h))`,
    `h' = Canon(ĥ + SwiGLU(LN ĥ))`.

    The block's state is fp32 from end to end; only the two projections inside
    each sublayer run at the compute dtype, and each returns fp32 so the adds
    are dtype-free.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window: int | None,
        d_ff: int,
        max_seq_len: int,
        rope_base: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.attn = FlowAttention(d_model, n_heads, window, max_seq_len, rope_base, device, dtype)
        self.canon1 = CanonLayer(d_model, device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.canon2 = CanonLayer(d_model, device)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        assert h.dtype == torch.float32, "the block's residual stream is fp32"
        h_hat = self.canon1(h + self.attn(self.norm1(h)))
        return self.canon2(h_hat + self.ffn(self.norm2(h_hat)))


# ---------------------------------------------------------------------------
# stage 2 — coding-rate chunking (`chunk.rs`)
# ---------------------------------------------------------------------------


def _logdet_spd(m: torch.Tensor) -> float:
    """`½` of the log-det of an SPD matrix, through its Cholesky (`chunk.rs:21-40`).

    burn hand-rolls the Cholesky (no LAPACK); torch's is the same decomposition.
    The oracle fixture's values come from a different route again (eigenvalues),
    so the two stay honest — see `tests/test_byteflow_parity.py`.
    """
    try:
        chol = torch.linalg.cholesky(m)
    except torch.linalg.LinAlgError as e:  # the crate's `expect("I + c·HᵀH must be SPD")`
        loud(f"logdet of an SPD matrix failed its Cholesky: {e}")
    return 2.0 * float(torch.log(torch.diagonal(chol)).sum())


def _prefix_logdets(rows: torch.Tensor, d: int, c: float) -> list[float]:
    """`R(h_1:p)` for every prefix (Sylvester folding, `chunk.rs:46-71`).

    `det(I_T + c·HHᵀ) = det(I_d + c·HᵀH)`, accumulated one row at a time in f64
    from the f32 rows (`chunk.rs:50-58`).
    """
    t = rows.shape[0]
    gram = torch.zeros(d, d, dtype=torch.float64)
    out: list[float] = []
    for p in range(t):
        hi = rows[p]  # f64 already: the cast is the caller's, as in the crate
        gram += hi[:, None] * hi[None, :]
        m = c * gram
        m.diagonal().add_(1.0)
        out.append(0.5 * _logdet_spd(m))
    return out


def _f64_rows(h: torch.Tensor) -> torch.Tensor:
    """The crate's `h.into_data().convert::<f64>()` (`chunk.rs:83`)."""
    return h.detach().to(torch.float64).cpu()


def marginal_gains_exact(h: torch.Tensor, eps2: float) -> torch.Tensor:
    """`ΔR_t = R(h_1:t) − R(h_1:t−1)` by eq. (11)-(12), `[b, t]` f32 (`chunk.rs:79-95`).

    Host-synced `O(T·d³)` per row — analysis and small-scale validation only;
    the forward uses [`RateMode.L2`], within 0.01 BPB of log-det in the paper's
    Table 4 (the crate's own `ponytail:` note, `chunk.rs:76-78`).
    """
    b, t, d = h.shape
    c = d / eps2
    rows = _f64_rows(h)
    gains = torch.zeros(b, t, dtype=torch.float32)
    for row in range(b):
        prev = 0.0
        for p, r in enumerate(_prefix_logdets(rows[row], d, c)):
            gains[row, p] = r - prev  # the f64→f32 cast of `chunk.rs:93`
            prev = r
    return gains


def coding_rate_exact(h: torch.Tensor, eps2: float) -> torch.Tensor:
    """`R_ε(h_1:T)` by eq. (11), `[b]` f32 (`chunk.rs:100-112`)."""
    b, t, d = h.shape
    c = d / eps2
    rows = _f64_rows(h)
    return torch.tensor(
        [float(_prefix_logdets(rows[row], d, c)[t - 1]) for row in range(b)], dtype=torch.float32
    )


def marginal_gains_l2(h: torch.Tensor) -> torch.Tensor:
    """Appendix B's `ΔR_t = √S_t − √S_{t−1}`, `S_t = Σ_{i≤t} ‖h_i‖²` (`chunk.rs:119-131`).

    FP32 ISLAND (start): the gains feed the top-k and nothing else, and a bf16
    `cumsum` can move a boundary on a tie with no gradient to correct it — so a
    bf16 `h` is widened here rather than scored in place.
    """
    h = h.to(torch.float32)
    b, t, _ = h.shape
    cum_rate = h.pow(2).sum(-1).cumsum(1).sqrt()
    zero = cum_rate[:, :1] * 0.0
    return cum_rate - torch.cat([zero, cum_rate[:, : t - 1]], dim=1)


def select_positions(gains: torch.Tensor, k: int) -> torch.Tensor:
    """Top-K boundaries: `{position 0 (BOS)} ∪ top-(K−1)` of `1..T`, ascending.

    `chunk.rs:137-156`; the index dtype matches burn's `[b, k]` int tensor.
    FP32 ISLAND (end): `topk` on two gains that are equal within bf16 epsilon
    picks one of them by a rule the model cannot learn from.
    """
    gains = gains.to(torch.float32)
    b, t = gains.shape
    if k < 1:
        loud(f"select_positions k={k}: k must be >= 1 (position 0 is the BOS boundary)")
    if k >= t:
        # Everything is selected; identity order (`chunk.rs:141-146`).
        return torch.arange(t, device=gains.device).reshape(1, t).expand(b, t).contiguous()
    # Candidates exclude position 0: it is always kept as the BOS boundary, so
    # the candidate indices shift by one (`chunk.rs:147-153`).
    idx = gains[:, 1:].topk(k - 1, dim=1).indices + 1
    picked = idx.sort(dim=1).values  # chronological
    bos = torch.zeros(b, 1, dtype=torch.long, device=gains.device)
    return torch.cat([bos, picked], dim=1)


# ---------------------------------------------------------------------------
# stage 4 — multi-linear upsampling (`net.rs:312-326`, `:475-526`)
# ---------------------------------------------------------------------------


def effective_bins(bins: int, t: int) -> int:
    """Shared-bin count for a length that need not divide: largest divisor of
    `t` not exceeding `bins` (1 always divides — a single-bin linear upsample)."""
    bins = max(bins, 1)
    if t % bins == 0:
        return min(bins, max(t, 1))
    return next(b for b in range(min(bins, t), 0, -1) if t % b == 0)


# ---------------------------------------------------------------------------
# the net (`net.rs:328-527`)
# ---------------------------------------------------------------------------


class ByteFlowNet(nn.Module):
    """The five-stage net. `forward(bytes) -> [b, t, 256]` byte logits."""

    def __init__(self, cfg: ByteFlowConfig, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        if cfg.dtype != torch.float32 and cfg.rate_mode is RateMode.LOGDET:
            # The config layer refuses this pair, but the model constructors do
            # not call `validate`: the same refusal is held here, at the seam
            # (ADR-0019 — a hand-built net must not enter a bf16 Cholesky).
            loud(
                "bf16 + byteflow_logdet: `marginal_gains_exact` factorizes "
                "`I + c·HᵀH` with a Cholesky, and a bf16 Cholesky is unstable. "
                "Escape: `byteflow_logdet = false` (the L2 fast path), or an "
                "fp32 compute dtype."
            )
        if cfg.d_global % cfg.n_heads_global or cfg.d_local % cfg.n_heads_local:
            loud(
                f"byteflow: d_local={cfg.d_local}/heads_local={cfg.n_heads_local} and "
                f"d_global={cfg.d_global}/heads_global={cfg.n_heads_global} must divide "
                f"evenly (net.rs:349-353). Escape: fix the byteflow_* widths in the preset."
            )
        self.cfg = cfg
        #: The COMPUTE dtype of every matmul (`cfg.dtype`). Masters are fp32.
        self.dtype = cfg.dtype

        def block(d_model: int, n_heads: int, window: int | None, d_ff: int, max_seq: int):
            return FlowBlock(
                d_model,
                n_heads,
                window,
                d_ff,
                max_seq,
                cfg.rope_base,
                device=device,
                dtype=cfg.dtype,
            )

        self.embedding = nn.Embedding(VOCAB, cfg.d_local, device=device)
        self.encoder = nn.ModuleList(
            [
                block(cfg.d_local, cfg.n_heads_local, cfg.w_local, cfg.d_ff_local, cfg.max_bytes)
                for _ in range(cfg.e_layers)
            ]
        )
        self.proj = _MixedLinear(cfg.d_local, cfg.d_global, device, dtype=cfg.dtype, fp32_out=True)
        # `net.global` in the crate; `global` is a keyword in Python.
        self.global_blocks = nn.ModuleList(
            [
                block(cfg.d_global, cfg.n_heads_global, None, cfg.d_ff_global, cfg.k_tokens)
                for _ in range(cfg.g_layers)
            ]
        )
        # `[bins, d_global, d_local]`, Kaiming-uniform (net.rs:393-402). torch's
        # `a=sqrt(5)` default is the same bound `sqrt(3)·gain/sqrt(fan_in)` that
        # burn's `Initializer::KaimingUniform{gain: 1.0, fan_out_only: false}`
        # computes with `fan_in = d_local`.
        self.upsample_w = nn.Parameter(
            torch.empty(cfg.bins, cfg.d_global, cfg.d_local, device=device)
        )
        nn.init.kaiming_uniform_(self.upsample_w, a=math.sqrt(5))
        self.decoder = nn.ModuleList(
            [
                block(cfg.d_local, cfg.n_heads_local, cfg.w_local, cfg.d_ff_local, cfg.max_bytes)
                for _ in range(cfg.e_layers)
            ]
        )
        self.out = nn.Linear(cfg.d_local, VOCAB, device=device)
        self.k_tokens = cfg.k_tokens
        self.bins = cfg.bins
        self.eps2 = cfg.eps2
        self.use_logdet_rate = cfg.rate_mode is RateMode.LOGDET
        self.max_bytes = cfg.max_bytes

    def forward(self, bytes_: torch.Tensor) -> torch.Tensor:
        """Next-byte logits `[b, t, 256]` for raw byte ids `[b, t]` (`net.rs:426-430`).

        Causal end to end: the chunker sees the full-sequence importance profile
        exactly as in teacher-forced training (paper §3.2).
        """
        h, z, sel = self.encode_chunks(bytes_, self.k_tokens)
        for blk in self.global_blocks:
            z = blk(z)
        return self.decode_chunks(z, h, sel)

    def encode_chunks(self, bytes_: torch.Tensor, k: int) -> tuple[torch.Tensor, ...]:
        """`(h [b,t,d_local], z [b,k,d_global], sel [b,k])` — `net.rs:438-461`."""
        b, t = bytes_.shape
        assert t <= self.max_bytes, f"T={t} exceeds the RoPE table ({self.max_bytes})"
        assert 1 <= k <= t, f"K={k} must be in 1..={t}"
        h = self.embedding(bytes_)
        for blk in self.encoder:
            h = blk(h)
        # Discrete selection: the gains feed the top-k only, no gradient path
        # (`net.rs:449-454`).
        gains = marginal_gains_exact(h, self.eps2) if self.use_logdet_rate else marginal_gains_l2(h)
        sel = select_positions(gains, k)
        d_local = h.shape[2]
        z = self.proj(h.gather(1, sel.reshape(b, k, 1).expand(-1, -1, d_local)))
        return h, z, sel

    def decode_chunks(self, g: torch.Tensor, h: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
        """`g [b,k,d_global]`, `h`/`sel` from the encode call → `[b,t,256]` (`net.rs:467-473`)."""
        b, t, _ = h.shape
        s = h + self.upsample(g, sel, t, b, effective_bins(self.bins, t))
        for blk in self.decoder:
            s = blk(s)
        if self.dtype == torch.float32:
            # The pre-bf16 head, verbatim: `F.linear`'s bias rides the same
            # `addmm` as today, which `matmul + bias` is not (`_MixedLinear`).
            return self.out(s)
        logits = mm_mixed(s, self.out.weight, self.dtype) + self.out.bias.to(self.dtype)
        # fp32 logits for the CE, like the loop's head (`model.py`): the GEMM is
        # bf16, the arm boundary is not.
        return logits.to(torch.float32)

    def upsample(
        self, g: torch.Tensor, sel: torch.Tensor, t: int, b: int, bins: int
    ) -> torch.Tensor:
        """`s̃_t = g_chunk(t)·W_bin(t)`, eq. (14)-(17) (`net.rs:475-526`).

        `chunk(t)` counts selected boundaries ≤ t (position 0 always is one);
        `bin(t) = ⌊t·B/T⌋` groups contiguous positions, so the per-position
        weight gather is B batched GEMMs. The GEMM runs at the compute dtype
        (`upsample_w` is an fp32 master like every other weight); the result is
        widened for the fp32 residual.
        """
        k, dg = g.shape[1], g.shape[2]
        dl = self.upsample_w.shape[2]
        # Boundaries → chunk index per position, via an inclusive prefix count.
        # add is exactly a write here: the selection is ascending and unique
        # (`net.rs:493-504`).
        bound = torch.zeros(b, t, dtype=torch.long, device=g.device)
        bound.scatter_add_(1, sel, torch.ones(b, k, dtype=torch.long, device=g.device))
        chunk_idx = bound.cumsum(1) - 1
        per = t // bins
        g_expanded = g.gather(1, chunk_idx.reshape(b, t, 1).expand(-1, -1, dg))
        # [b, bins, per, dg] @ [bins, dg, dl] → [b, t, dl]. Fallback lengths use
        # a PREFIX of the shared bin weights (`net.rs:514-525`).
        per_pos = g_expanded.reshape(b, bins, per, dg)
        w = self.upsample_w[:bins].reshape(1, bins, dg, dl)
        if self.dtype != torch.float32:
            per_pos = per_pos.to(self.dtype)
            w = w.to(self.dtype)
        out = (per_pos @ w).reshape(b, t, dl)
        return out.to(torch.float32)


# ---------------------------------------------------------------------------
# the streaming patcher (`stream.rs`) — the same quantities as running state
# ---------------------------------------------------------------------------


class RatePatcher:
    """One window of hidden states, patched position by position (`stream.rs:38-176`).

    State is the d×d Gram (LogDet) or one running sum of squares (L2) — never a
    T-length buffer — so a stream of any length patches in O(d²) memory. A
    window is ONE sequence `h_1:W`; `reset()` begins the next one, which is
    exactly the semantics the batch path gives a teacher-forced window.
    """

    def __init__(self, d: int, eps2: float, mode: RateMode = RateMode.L2) -> None:
        assert d >= 1, "hidden width must be >= 1"
        assert eps2 > 0.0, "eps2 must be > 0 (it divides d inside the logdet)"
        self.mode = mode
        self.d = d
        self.c = d / eps2
        self.gram = [0.0] * (d * d)
        self.sum_sq = 0.0
        self.rate = 0.0
        self.prev_rate = 0.0
        self._gains: list[float] = []

    def push(self, h: list[float]) -> float:
        """Feed ONE hidden row; returns `ΔR_t` for its position (`stream.rs:78-117`)."""
        if len(h) != self.d:
            raise AssertionError(f"hidden row width {len(h)} must equal the patcher's d={self.d}")
        if self.mode is RateMode.LOGDET:
            d = self.d
            if any(v != 0.0 for v in h):
                for i, hi32 in enumerate(h):
                    hi = float(hi32)
                    if hi == 0.0:
                        continue
                    base = i * d
                    for j, hv in enumerate(h):
                        self.gram[base + j] += hi * float(hv)
            m = torch.tensor(self.gram, dtype=torch.float64).reshape(d, d) * self.c
            m.diagonal().add_(1.0)
            self.rate = 0.5 * _logdet_spd(m)
        else:
            self.sum_sq += sum(float(v) * float(v) for v in h)
            self.rate = math.sqrt(self.sum_sq)
        dr = self.rate - self.prev_rate
        self.prev_rate = self.rate
        self._gains.append(dr)
        return dr

    def rate_(self) -> float:
        """`R(h_1:t)` of the current window."""
        return self.rate

    def gains(self) -> list[float]:
        return list(self._gains)

    def __len__(self) -> int:
        return len(self._gains)

    def is_empty(self) -> bool:
        return not self._gains

    def reset(self) -> None:
        """Begin the next window: the gain history and the rate state clear."""
        self.gram = [0.0] * (self.d * self.d)
        self.sum_sq = 0.0
        self.rate = 0.0
        self.prev_rate = 0.0
        self._gains.clear()

    def borders(self, k: int) -> list[int]:
        """`{0} ∪ top-(K−1)` by `ΔR`, chronological (`stream.rs:150-175`).

        Ties break on position ascending, so the result is a pure function of
        the gains. `k = 1` is the BOS boundary alone; `k ≥ T` selects all.
        """
        assert k >= 1, "k must be >= 1"
        t = len(self._gains)
        assert t >= 1, "borders on an empty window: push at least one position"
        if k >= t:
            return list(range(t))
        idx = sorted(range(1, t), key=lambda a: (-self._gains[a], a))[: k - 1]
        return sorted([0, *idx])


# ---------------------------------------------------------------------------
# the channel — Option B: the front/back stages around our loop
# ---------------------------------------------------------------------------


def compat_armed(cfg: ModelConfig) -> bool:
    """True when `use_byteflow` runs the channel (Option B) instead of the net alone.

    The ONE list: `config.validate` refuses the arms it names and
    `DormouseModel` reads it, so a config that reaches a forward is a config
    the validator already ruled on. `use_kda` is the one arm the channel
    admits today — the others are not wired into it yet, and an unwired arm
    must not start running because a byteflow flag moved.
    """
    return cfg.use_byteflow and (
        cfg.use_kda or cfg.use_engram or cfg.use_attnres or cfg.moe_topk > 0
    )


class BfChannel(nn.Module):
    """The byteflow half of Option B: the net's own params + the two pairing Linears.

    `net.global_blocks` is built EMPTY (`g_layers = 0`): the loop IS the global
    stage here, and the net's own `forward` (never called on this path) would
    refuse `K = 0` loudly before it could be mistaken as wired. The empty
    `ModuleList` also writes no state, so the channel's byteflow side is the
    same parameter set as the standalone net's.
    """

    def __init__(self, cfg: ModelConfig, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        bfc = byteflow_config(cfg)
        self.net = ByteFlowNet(dataclasses.replace(bfc, g_layers=0), device=device)
        # The two pairing Linears are on the SAME compute-dtype rule as the net:
        # bf16 GEMM, fp32 out (the loop's state and the residual are fp32).
        self.in_proj = _MixedLinear(
            bfc.d_global, cfg.d_model, device, dtype=bfc.dtype, fp32_out=True
        )
        self.out_proj = _MixedLinear(
            cfg.d_model, bfc.d_global, device, dtype=bfc.dtype, fp32_out=True
        )
        self.k = bfc.k_tokens


def bf_for_model(cfg: ModelConfig, *, device: torch.device | str = "cpu") -> BfChannel | None:
    """The model's byteflow knob: `Some` iff the channel (Option B) is armed."""
    return BfChannel(cfg, device=device) if compat_armed(cfg) else None


def count_channel(counters: Counters) -> None:
    """One `bf_channel` per channel forward — the step's counter line is the evidence."""
    counters.count("bf_channel")
