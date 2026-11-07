"""A plain pre-LN transformer block — `scripts/control_transformer.py`'s Block.

WHY IT IS IN THE MODEL. The 2026-11-06 control
(`docs/reviews/2026-11-06-control-transformer.md`) trained a boring 13.5M
decoder on the trainer's own byte stream and beat the KDA/CSA/mHC/Engram block
by 0.126 BPB on all eight eval windows (0.13-0.37 by the report's range). An A/B
that answers "is the block form or the pipeline" therefore has to run BOTH forms
through this trainer: the embedding, data stream, eval window, optimizer,
checkpoint and resume are shared, and only the block differs.

THE BLOCK IS THE CONTROL'S, TRANSCRIBED. Pre-LayerNorm, no biases, packed
`qkv` (`3·d²`), RoPE on `head_dim`, `F.scaled_dot_product_attention(is_causal=True)`,
GELU FFN (`d_ffn = 4·d_model` in the control; an arm sets `d_ffn` and
`flagship3_dense` runs 3.69×, so the width is the config's, not this file's),
GPT-2 residual scaling on the two output projections
(`0.02/sqrt(2·n_layers)`). Parameters per block:
`4·d²` attention + `2·d·d_ffn` FFN + `4·d` of LayerNorm, which at
`d=304, ffn=1216` is 1 110 208 — the control's own count.

THE PORT'S DTYPE RULE, NOT A SECOND ONE. The residual stream stays fp32 (the
burn "mixed-dtype ops NaN" rule): every projection goes through
[`tsct.mm_mixed`] (fp32 masters, bf16 compute when `bf16=true`), the output is
widened back to the residual dtype, and SDPA runs in the compute dtype like
`FullAttention`. `bf16=false` (the CPU gates) is plain fp32 bit for bit.

MIXTURE-OF-DEPTHS (`mod_capacity < 1.0`, arXiv:2404.02258v1 §3.4). A per-block
router `r_i = w^T x_i` (one bias-free linear projection, `w ∈ R^d`) picks the
top-`C = round(capacity·T)` tokens of each sequence; those `C` tokens are
gathered, run through the block, scaled by their own `r_i` (Eq. 1's
`r_i·f(X̃) + x_i`, which puts the router on the gradient path) and scattered
back; the rest bypass through the residual. The gather is in POSITION order (a
scatter through the slots the selection occupies), so `is_causal=True` over the
compacted sequence is the same causal mask the full sequence had, and RoPE still
uses each token's ORIGINAL position. One `torch.topk` + gather/scatter per
block, static shapes, no `.item()`/`nonzero`/boolean-mask indexing — CUDA-graph
capturable by construction. `mod_capacity=1.0` builds no router at all: the
forward is the plain block, bit for bit.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import Counters, loud
from .tsct import mm_mixed

__all__ = [
    "TransformerBlock",
    "apply_rope",
    "mod_capacity_tokens",
    "mod_route",
    "rope_tables",
]


def rope_tables(
    t: int, head_dim: int, device: torch.device | str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """`cos, sin [1, 1, t, head_dim/2]` — the control's RoPE spectrum.

    `inv = 1 / 10000^(j/half)` for `j = 0..half-1`, `ang[p, j] = p·inv[j]`.
    Half the head rotates against the other half, so `head_dim` must be even
    (the `proj`/`qkv` split below is by `head_dim`, and an odd one would leave
    a channel unrotated).
    """
    assert head_dim % 2 == 0, f"RoPE needs an even head_dim, got {head_dim}"
    half = head_dim // 2
    inv = 1.0 / (10_000 ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    ang = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv)
    return ang.cos().to(dtype)[None, None], ang.sin().to(dtype)[None, None]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The control's pair rotation, `[..., hd] -> [..., hd]`, broadcast over heads."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def mod_capacity_tokens(capacity: float, top_k: int, t: int) -> int:
    """`C` — the tokens one block processes: `top_k`, else `round(capacity·T)`.

    Pure Python integers (`t` is a shape), so it costs no host read. `top_k`
    wins when nonzero; otherwise the capacity is rounded to the nearest integer
    and clamped into `1..t` (a static shape either way).
    """
    assert t >= 1, t
    if top_k:
        if top_k > t:
            loud(
                f"mod_top_k={top_k} > the sequence's {t} positions: the router "
                f"cannot select more tokens than the forward holds. Escape: "
                f"mod_top_k <= {t}, or 0 to derive it from mod_capacity."
            )
        return int(top_k)
    return min(max(int(round(capacity * t)), 1), t)


def mod_route(
    scores: torch.Tensor, capacity: float, top_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-`C` selection as ONE `topk` + scatter/cumsum — no host read.

    `scores [b, t]` are the router logits. Returns

    * `idx [b, C]` — the selected token positions (score order, as `topk`
      returns them);
    * `sel [b, t]` int64 — the selection mask (1 where selected), built by a
      scatter (never a boolean mask or `nonzero`);
    * `slots [b, C]` — each selected token's rank in POSITION order, built from
      `cumsum` of the mask. A `scatter` through `slots` compacts the selection
      into original-position order without a second sort, so `is_causal=True`
      on the compacted sequence is the same causal mask the full sequence has.

    Every op is device-side and static-shape; the only host values are `C`
    (from Python ints) and the buffer sizes. This is the capture-safe form the
    CUDA-graph probe in `scripts/probe_capture.py` looks for.
    """
    assert scores.dim() == 2, scores.shape
    b, t = scores.shape
    c = mod_capacity_tokens(capacity, top_k, t)
    idx = torch.topk(scores, c, dim=1).indices
    sel = torch.zeros((b, t), dtype=torch.long, device=scores.device)
    sel.scatter_(1, idx, 1)
    slots = (sel.cumsum(1) - 1).gather(1, idx)
    return idx, sel, slots


class TransformerBlock(nn.Module):
    """One vanilla decoder block: `x + proj(SDPA(RoPE(qkv(norm(x)))))`, then FFN.

    With `mod_capacity < 1.0` (or `mod_top_k > 0`) the same block also owns a
    Mixture-of-Depths router; its `forward` then routes `C` tokens through
    `_block` and passes the rest through unchanged (see the module docstring).
    """

    def __init__(self, cfg, *, n_layers: int, device: torch.device | str = "cpu") -> None:
        super().__init__()
        d, ffn, heads, hd = cfg.d_model, cfg.d_ffn, cfg.n_heads, cfg.head_dim
        assert heads * hd == d, "validate() pins n_heads*head_dim == d_model"
        assert d % heads == 0, f"d_model={d} must tile {heads} heads"
        assert hd % 2 == 0, f"RoPE needs an even head_dim, got {hd}"
        assert n_layers >= 1, n_layers
        self.d = d
        self.n_heads = heads
        self.head_dim = hd
        self.scale = 1.0 / math.sqrt(hd)
        self.compute_dtype = torch.bfloat16 if cfg.bf16 else torch.float32
        self.ln1 = nn.LayerNorm(d, device=device)
        self.qkv = nn.Linear(d, 3 * d, bias=False, device=device)
        self.proj = nn.Linear(d, d, bias=False, device=device)
        self.ln2 = nn.LayerNorm(d, device=device)
        self.fc = nn.Linear(d, ffn, bias=False, device=device)
        self.fc2 = nn.Linear(ffn, d, bias=False, device=device)
        # The control's init (`Control._init` + its residual scaling): every
        # projection N(0, 0.02), then the two writes into the residual stream
        # shrunk by sqrt(2·L) so the stream does not grow with depth. LayerNorm
        # keeps torch's (weight=1, bias=0), and the embedding is the port's
        # shared one, not re-drawn.
        std = 0.02
        nn.init.normal_(self.qkv.weight, std=std)
        nn.init.normal_(self.fc.weight, std=std)
        nn.init.normal_(self.proj.weight, std=std / math.sqrt(2 * n_layers))
        nn.init.normal_(self.fc2.weight, std=std / math.sqrt(2 * n_layers))
        # The MoD router (`r_i = w^T x_i`), built IFF routing is on. A
        # CONSTANT init, not `nn.Linear`'s: `nn.Linear` would draw from the
        # global RNG, shifting every later block's init and turning the arm
        # into a different seed. `0.02` gives `r ≈ 0.02·Σx ≈ O(1)` on the
        # embedding stream, so the scaled block output starts at full strength
        # and every token is selectable. At `mod_capacity=1.0` there is no
        # router and `forward` is the plain block, bit for bit.
        self.mod_capacity = float(cfg.mod_capacity)
        self.mod_top_k = int(cfg.mod_top_k)
        self.router: nn.Parameter | None = None
        if self.mod_capacity < 1.0 or self.mod_top_k:
            self.router = nn.Parameter(torch.full((1, d), 0.02, device=device))
        #: The last forward's BCE aux value, DETACHED (log-cadence read only).
        self.last_aux: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [b, t, d]` in the residual dtype → `[b, t, d]` in the same dtype."""
        if self.router is None:
            return self._block(x, counters)
        return self._forward_mod(x, counters)

    def _block(
        self,
        x: torch.Tensor,
        counters: Counters | None = None,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The plain block. `rope` is the pre-built `(cos, sin)` table of a
        compacted MoD call (per-token ORIGINAL positions); `None` is today's
        table over positions `0..t-1`."""
        pdt = self.qkv.weight.dtype
        if x.dtype is not pdt:
            x = x.to(pdt)
        b, t, _ = x.shape
        d = self.d
        qkv = mm_mixed(self.ln1(x), self.qkv.weight, self.compute_dtype, counters)
        q, k, v = qkv.split(d, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        if rope is None:
            cos, sin = rope_tables(t, self.head_dim, x.device, q.dtype)
        else:
            cos, sin = rope
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        y = y.transpose(1, 2).reshape(b, t, d)
        # Every projection runs in the compute dtype; the residual add is the
        # fp32 stream (mm_mixed's contract, widened back here).
        x = x + mm_mixed(y, self.proj.weight, self.compute_dtype, counters).to(x.dtype)
        h = mm_mixed(self.ln2(x), self.fc.weight, self.compute_dtype, counters)
        h = mm_mixed(F.gelu(h), self.fc2.weight, self.compute_dtype, counters)
        return x + h.to(x.dtype)

    def _forward_mod(self, x: torch.Tensor, counters: Counters | None) -> torch.Tensor:
        """Route `C = mod_capacity_tokens(...)` tokens through the block.

        One `topk` on the router score, gather into fixed buffers in POSITION
        order, the plain block on those `C` tokens, then `r_i·f_i + x_i` (Eq. 1)
        scattered back. No `.item()`, `nonzero`, boolean-mask indexing or host
        sync anywhere in the path — the buffers are sized from the static
        shape, so a CUDA graph can hold it.
        """
        b, t, d = x.shape
        scores = F.linear(x, self.router).squeeze(-1)  # [b, t], fp32 residual
        idx, sel, slots = mod_route(scores, self.mod_capacity, self.mod_top_k)
        c = idx.shape[1]
        g = idx.unsqueeze(-1).expand(-1, -1, d)
        sg = slots.unsqueeze(-1).expand(-1, -1, d)
        # Compact into position order through `slots` (a scatter, not a sort).
        buf = x.new_zeros((b, c, d)).scatter_(1, sg, x.gather(1, g))
        rbuf = scores.new_zeros((b, c)).scatter_(1, slots, scores.gather(1, idx))
        pos = torch.arange(t, device=x.device).expand(b, t).gather(1, idx)
        pbuf = torch.zeros((b, c), dtype=torch.long, device=x.device).scatter_(1, slots, pos)
        cos_tab, sin_tab = rope_tables(t, self.head_dim, x.device, self.compute_dtype)
        cos = cos_tab[0, 0][pbuf].unsqueeze(1)
        sin = sin_tab[0, 0][pbuf].unsqueeze(1)
        y = self._block(buf, counters, rope=(cos, sin))
        # Eq. 1: `r_i · f_i(X̃) + x_i`. The router weight is ON the gradient
        # path, which is the only gradient the router gets from the LM term
        # (the top-k identity itself is not differentiable).
        delta = (y - buf) * rbuf.unsqueeze(-1)
        out = x + x.new_zeros((b, t, d)).scatter(1, g, delta.gather(1, sg))
        # The paper's §3.5 aux: BCE(logits=r_i, target=top-k indicator). One
        # value per block, detached for the log line; the caller weights it.
        self.last_aux = F.binary_cross_entropy_with_logits(scores, sel.to(scores.dtype))
        if counters is not None:
            counters.count("mod_block")
            counters.count("mod_tokens", b * c)
        return out
