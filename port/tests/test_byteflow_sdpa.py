"""The SDPA rewrite of `FlowAttention` must be the same function, not a similar one.

`FlowAttention.forward` used to materialize `[b, heads, t, t]` scores, mask them
and softmax by hand. It now calls `F.scaled_dot_product_attention` with the same
scale and the same strictly-lower triangle. This file pins the equivalence
against a literal transcription of the old path, in all three mask modes the
shipped presets can reach: no window, a window that covers the whole sequence
(the fast path), and a window shorter than the sequence (the banded path).

The bound is fp32's own noise on this shape (measured 1.19e-7, i.e. 1 ULP of a
softmax output); a wrong mask, a wrong scale or a transposed q/k lands orders
above it, which is what the number is for.
"""

from __future__ import annotations

import torch

from dormouse.model.byteflow import FlowAttention, causal_mask


def _naive(attn: FlowAttention, x: torch.Tensor) -> torch.Tensor:
    """The pre-SDPA body, verbatim (scores -> masked_fill -> softmax -> matmul)."""
    b, t, d = x.shape
    hd = d // attn.n_heads
    qkv = attn.qkv(x)
    q = attn.rope.apply(qkv[..., :d]).reshape(b, t, attn.n_heads, hd).transpose(1, 2)
    k = attn.rope.apply(qkv[..., d : 2 * d]).reshape(b, t, attn.n_heads, hd).transpose(1, 2)
    v = qkv[..., 2 * d :].reshape(b, t, attn.n_heads, hd).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) * hd**-0.5
    scores = scores.masked_fill(causal_mask(t, attn.window, x.device), -1e9)
    mixed = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, t, d)
    return attn.out(mixed)


def test_the_sdpa_attention_is_the_old_function() -> None:
    torch.manual_seed(0)
    for window, t in ((512, 512), (128, 512), (None, 256)):
        attn = FlowAttention(64, 4, window, 512, 10_000.0).eval()
        x = torch.randn(2, t, 64)
        with torch.no_grad():
            got, want = attn(x), _naive(attn, x)
        diff = (got - want).abs().max().item()
        assert diff < 1e-6, f"window={window} t={t}: max|d|={diff:.3e}"


def test_causality_survives_the_rewrite() -> None:
    """Position 0 must not see position 1 under either mask mode."""
    torch.manual_seed(1)
    attn = FlowAttention(64, 4, None, 512, 10_000.0).eval()
    x = torch.randn(1, 64, 64)
    with torch.no_grad():
        base = attn(x)
        x2 = x.clone()
        x2[:, 1:] = torch.randn_like(x2[:, 1:])
        moved = attn(x2)
    assert torch.equal(base[:, :1], moved[:, :1]), "position 0 leaked a future token"
    assert not torch.allclose(base[:, 1:], moved[:, 1:]), "later positions ignored the edit"
