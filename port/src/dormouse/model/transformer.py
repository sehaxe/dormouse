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
GELU 4x FFN (`d_ffn = 4·d_model`), GPT-2 residual scaling on the two output
projections (`0.02/sqrt(2·n_layers)`). Parameters per block:
`4·d²` attention + `2·d·d_ffn` FFN + `4·d` of LayerNorm, which at
`d=304, ffn=1216` is 1 110 208 — the control's own count.

THE PORT'S DTYPE RULE, NOT A SECOND ONE. The residual stream stays fp32 (the
burn "mixed-dtype ops NaN" rule): every projection goes through
[`tsct.mm_mixed`] (fp32 masters, bf16 compute when `bf16=true`), the output is
widened back to the residual dtype, and SDPA runs in the compute dtype like
`FullAttention`. `bf16=false` (the CPU gates) is plain fp32 bit for bit.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import Counters
from .tsct import mm_mixed

__all__ = ["TransformerBlock", "apply_rope", "rope_tables"]


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


class TransformerBlock(nn.Module):
    """One vanilla decoder block: `x + proj(SDPA(RoPE(qkv(norm(x)))))`, then FFN."""

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

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [b, t, d]` in the residual dtype → `[b, t, d]` in the same dtype."""
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
        cos, sin = rope_tables(t, self.head_dim, x.device, q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        y = y.transpose(1, 2).reshape(b, t, d)
        # Every projection runs in the compute dtype; the residual add is the
        # fp32 stream (mm_mixed's contract, widened back here).
        x = x + mm_mixed(y, self.proj.weight, self.compute_dtype, counters).to(x.dtype)
        h = mm_mixed(self.ln2(x), self.fc.weight, self.compute_dtype, counters)
        h = mm_mixed(F.gelu(h), self.fc2.weight, self.compute_dtype, counters)
        return x + h.to(x.dtype)
