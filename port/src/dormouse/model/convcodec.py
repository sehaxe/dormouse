"""ConvCodec — the byte↔patch codec that replaces the attention codec.

`ByteFlowNet`'s encoder and decoder are two stacks of sliding-window attention
blocks over BYTES (`byteflow.py`: E=6 blocks each way). That is the whole cost
of the channel: the attention codec is 4·d² (qkv+out) + 2·w·d (SDPA) +
3·d·f (SwiGLU) MACs per byte per block, while the loop it feeds runs on K
latents. Measured 2026-10-30, b=4 T=512 K=128 bf16 (`scripts/bench_bf_dtype.py`):
encode 9.85 ms + decode 8.54 ms against a 40.48 ms loop = 31 % of a step, and
at 8 bytes/patch it gets WORSE (~49 %) because the encoder scales with BYTES
while the loop scales with latents.

`ConvCodec` keeps the same three-method contract — `encode_chunks(bytes, k) ->
(h, z, sel)`, `decode_chunks(g, h, sel) -> [b, t, 256]`, `forward(bytes) ->
[b, t, 256]` — with NO attention:

* **patchify** — one strided `nn.Conv1d(width → width, kernel = stride =
  `patch_bytes`) over the byte embedding, then `layers` residual `Conv1d(3)` +
  GELU blocks over the patch axis;
* **chunking** — the SAME `marginal_gains_l2` + `select_positions` bookkeeping
  as the attention codec, applied to the patch features. `sel` is still
  ASCENDING BYTE POSITIONS (patch starts), position 0 always selected, and
  `h` is still the per-byte stream `[b, t, width]` the patchifier reads;
* **unpatchify** — a low-rank linear `d_global → r → patch_bytes·256`
  (`_MixedLinear` masters, the port's bf16 rule) applied to the K selected
  latents, gathered back to all P patches by the attention codec's chunk rule
  (`chunk(p) = #selected boundaries ≤ p − 1`), reshaped to `[b, t, 256]`, plus
  a per-byte skip from `h`, the byte-logit bias, and one causal depthwise
  `Conv1d(3)` that smooths logits across byte boundaries.

`h`'s channel width is the codec's own `width` (the attention codec's is its
`byteflow_d_local`): no caller outside the codec reads it (`model.py` derives
`b, t` from it and passes it back to `decode_chunks`), so the rank-3/T contract
holds. It is NOT `byteflow_d_local`, because a dense `d_local → width` patchify
at the profile's 1024 would cost 512 KiB-worth of MACs per patch on its own.

# MAC arithmetic (the numbers the constants were tuned to)

Per patch, measured loop = **24.1 M MAC per latent** (d=1024, 3 iterations,
`scripts/bench_bf_dtype.py`, 2026-10-30); the fraction is of that figure.

Attention codec, d=1024, f=2730, E=3, w=512, per BYTE and per direction:
`4d² + 2wd + 3df + 8d = 13.64 M`; both directions = `27.28 M/byte`.

| codec | bytes/patch | MAC per patch | fraction of the 24.1 M loop |
|---|---|---|---|
| attention | 4 | 109.1 M | 453 % |
| attention | 8 | 218.2 M | 905 % |
| conv (this one, C=64, r=64, d_global=1024) | 4 | 306 176 | 1.27 % |
| conv (this one, C=64, r=64, d_global=1024) | 8 | 456 704 | 1.90 % |

The conv line is `C·C·s + L·C·C·3 + C·d_global + d_global·r + r·s·256 +
s·C·256 + s·256·3` with C=64, L=2, r=64: patchify 32 768 + residual convs
24 576 + proj 65 536 + low-rank 196 608 + skip 131 072 + smooth 6 144. The
attention fraction is arithmetic, not wall time: the loop is launch-bound on
this box (AGENTS §3.1), which is why 4.5x the loop's MACs measured as 31 % of
its wall time. Cutting the arithmetic to 1.9 % is what makes the codec cheap
in both currencies; `scripts/bench_bf_dtype.py`'s codec arms are the wall-time
check on a GPU.

Constants were tuned DOWN from the first draft (C=128, r=256, 4.8 %/patch):
`test_convcodec.py` pins `num_params()` and `mac_per_patch()` under ceilings so
an edit cannot quietly grow the codec back to attention cost.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig
from ..instrument import loud
from .byteflow import _MixedLinear, marginal_gains_l2, select_positions

__all__ = [
    "CODEC_MAC_CEILING",
    "CODEC_PARAM_CEILING",
    "LOOP_MAC_PER_LATENT",
    "VOCAB",
    "ConvCodec",
    "codec_kind",
]

#: Raw-byte vocabulary (next-byte LM), identical to `byteflow.VOCAB`.
VOCAB = 256

#: MAC per latent of the weight-shared loop this codec feeds (d=1024, 3
#: iterations; the measured figure named in the module docstring).
LOOP_MAC_PER_LATENT = 24_100_000

#: The codec may spend at most 2 % of the loop's MACs per patch. The test
#: asserts the estimate stays under it; a future edit that adds a per-byte
#: `d_local → 256` head (the attention cost in disguise) fails here.
CODEC_MAC_CEILING = LOOP_MAC_PER_LATENT // 50

#: Parameter ceiling at the profile shape. The attention codec is ~94 M params
#: at d=1024/E=3 (the upsampler's 16 bins alone are 16.8 M); 1 M is the
#: "stays a codec" line.
CODEC_PARAM_CEILING = 1_000_000


def codec_kind(cfg: ModelConfig) -> str:
    """`"attention"` (default, bit-identical) or `"conv"`; anything else is LOUD.

    The constructors deliberately do not call `validate` (a hand-built model
    must not run a field it never checked), so this seam holds the same two
    refusals the config layer holds: an unknown name must not silently fall
    back to attention, and `conv` without `use_byteflow` names a codec nothing
    builds.
    """
    kind = cfg.byteflow_codec
    if kind not in ("attention", "conv"):
        loud(
            f"byteflow_codec={kind!r} is not a codec. Known: 'attention' (the ported "
            f"ByteFlow net, the default) and 'conv' (`model/convcodec.py`, no "
            f"attention). Escape: name one of those."
        )
    if kind == "conv" and not cfg.use_byteflow:
        loud(
            "byteflow_codec='conv' with use_byteflow=false: the codec is only built "
            "by the byteflow shapes, so the field would name a codec that never runs. "
            "Escape: use_byteflow=true, or byteflow_codec='attention'."
        )
    return kind


class ConvCodec(nn.Module):
    """Bytes↔patch codec, convolution only (see the module docstring).

    Every weight is an fp32 MASTER; convs and linears run in `dtype` (bf16 under
    `bf16=true`, the port's precision rule) and each returns fp32, so the
    residual streams stay fp32 exactly as in `byteflow.py`.
    """

    def __init__(
        self,
        d_global: int,
        *,
        patch_bytes: int = 8,
        width: int = 64,
        rank: int = 64,
        layers: int = 2,
        dtype: torch.dtype = torch.float32,
        k_tokens: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if patch_bytes < 1:
            loud(f"conv codec: patch_bytes={patch_bytes} must be >= 1 byte per patch")
        if width < 1 or rank < 1:
            loud(f"conv codec: width={width}/rank={rank} must be >= 1")
        if not 1 <= layers <= 2:
            loud(
                f"conv codec: layers={layers}; the design is 1-2 residual conv layers. "
                f"Escape: 1 or 2 (each extra kernel-3 layer is C²·3 MAC/patch)."
            )
        self.d_global = d_global
        self.patch_bytes = patch_bytes
        self.width = width
        self.rank = rank
        self.dtype = dtype
        self.k_tokens = k_tokens
        self.embedding = nn.Embedding(VOCAB, width, device=device)
        self.patchify = nn.Conv1d(
            width, width, kernel_size=patch_bytes, stride=patch_bytes, device=device
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(width, width, kernel_size=3, padding=1, device=device)
                for _ in range(layers)
            ]
        )
        self.proj = _MixedLinear(width, d_global, device, dtype=dtype, fp32_out=True)
        self.up = _MixedLinear(d_global, rank, device, dtype=dtype, fp32_out=True)
        self.down = _MixedLinear(rank, patch_bytes * VOCAB, device, dtype=dtype, fp32_out=True)
        self.skip = _MixedLinear(width, VOCAB, device, dtype=dtype, fp32_out=True)
        self.bias = nn.Parameter(torch.zeros(VOCAB, device=device))
        #: Causal depthwise smoothing over the flat byte axis (groups=VOCAB:
        #: 768 MAC/byte, not the 196 608 of a dense 256→256 kernel 3).
        self.smooth = nn.Conv1d(VOCAB, VOCAB, kernel_size=3, groups=VOCAB, device=device)

    # -- compute-dtype plumbing (master stays fp32, the op runs in `dtype`) ---

    def _conv(
        self,
        conv: nn.Conv1d,
        x: torch.Tensor,
        *,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> torch.Tensor:
        dt = self.dtype
        w = conv.weight if conv.weight.dtype == dt else conv.weight.to(dt)
        b = None
        if conv.bias is not None:
            b = conv.bias if conv.bias.dtype == dt else conv.bias.to(dt)
        y = F.conv1d(
            x if x.dtype == dt else x.to(dt), w, b, stride=stride, padding=padding, groups=groups
        )
        return y.to(torch.float32)

    def _patches(self, h: torch.Tensor) -> torch.Tensor:
        """`[b, t, width]` bytes → `[b, p, width]` patch features (no attention)."""
        x = h.transpose(1, 2)
        p = F.gelu(self._conv(self.patchify, x, stride=self.patch_bytes))
        for conv in self.convs:
            p = p + F.gelu(self._conv(conv, p, padding=1))
        return p.transpose(1, 2)

    # -- the contract -------------------------------------------------------

    def encode_chunks(self, bytes_: torch.Tensor, k: int) -> tuple[torch.Tensor, ...]:
        """`(h [b,t,width], z [b,k,d_global], sel [b,k])`.

        `sel` holds BYTE positions of selected patch starts (ascending, position
        0 always present), the same convention as the attention codec's
        chunker; `k == p` selects every patch, `k < p` runs the shared L2
        gain/top-k rule on the patch features, `k > p` is LOUD.
        """
        b, t = bytes_.shape
        s = self.patch_bytes
        if t % s:
            loud(
                f"conv codec: T={t} is not a multiple of patch_bytes={s}: the strided "
                f"patchify covers floor(T/s) patches and would silently drop the last "
                f"{t % s} bytes, so their logits would come from nowhere. Escape: a "
                f"window divisible by {s}, or patch_bytes that divides T."
            )
        p = t // s
        if not 1 <= k <= p:
            loud(
                f"conv codec: K={k} must be in 1..={p} (one latent per selected patch "
                f"of {t} bytes / {s}). Escape: byteflow_k_tokens <= T/patch_bytes."
            )
        h = self.embedding(bytes_)
        feats = self._patches(h)
        if k == p:
            idx = torch.arange(p, device=bytes_.device).reshape(1, p).expand(b, p)
        else:
            idx = select_positions(marginal_gains_l2(feats), k)
        z = self.proj(feats.gather(1, idx.reshape(b, k, 1).expand(-1, -1, self.width)))
        return h, z, idx * s

    def decode_chunks(self, g: torch.Tensor, h: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
        """`g [b,k,d_global]`, `h`/`sel` from encode → byte logits `[b,t,256]`."""
        b, t, _ = h.shape
        s = self.patch_bytes
        assert t % s == 0, "decode without the encode's divisibility check"
        p = t // s
        k = g.shape[1]
        assert tuple(sel.shape) == (b, k), (tuple(sel.shape), (b, k))
        # The attention codec's boundary rule, at patch resolution: each patch
        # belongs to the last selected patch start at or before it.
        bound = torch.zeros(b, p, dtype=torch.long, device=g.device)
        bound.scatter_add_(1, sel // s, torch.ones(b, k, dtype=torch.long, device=g.device))
        chunk = bound.cumsum(1) - 1
        # Low-rank on the K LATENTS only, gathered to the P patches: with
        # K<P the rate chunking buys the same saving as in the attention codec.
        u = self.down(F.gelu(self.up(g)))  # [b,k,s*256], fp32
        u = u.gather(1, chunk.reshape(b, p, 1).expand(-1, -1, s * VOCAB))
        logits = u.reshape(b, t, VOCAB) + self.skip(h) + self.bias
        y = F.pad(logits.transpose(1, 2), (2, 0))  # causal, kernel 3
        y = self._conv(self.smooth, y, groups=VOCAB)
        out = logits + y.transpose(1, 2)
        assert tuple(out.shape) == (b, t, VOCAB), out.shape
        assert out.dtype == torch.float32, "the byte logits are the arm boundary"
        return out

    def forward(self, bytes_: torch.Tensor) -> torch.Tensor:
        """Option A: the codec alone (no loop), `[b,t,256]` byte logits."""
        if self.k_tokens is None:
            loud("conv codec: forward() needs k_tokens (Option A builds it with K)")
        h, z, sel = self.encode_chunks(bytes_, self.k_tokens)
        return self.decode_chunks(z, h, sel)

    # -- the cost gate ------------------------------------------------------

    def num_params(self) -> int:
        """The codec's parameter count — the gate's first ceiling."""
        return sum(p.numel() for p in self.parameters())

    def mac_per_patch(self) -> int:
        """Worst-case (K=P) MACs per patch; the gate's second ceiling.

        Biases, GELU and the gather/reshape are not counted: they are O(C) or
        copies, three orders below the terms below.
        """
        s, c, dg, r = self.patch_bytes, self.width, self.d_global, self.rank
        return (
            c * c * s  # patchify
            + len(self.convs) * c * c * 3  # residual convs
            + c * dg  # patch → global
            + dg * r  # low-rank up
            + r * s * VOCAB  # low-rank down
            + s * c * VOCAB  # per-byte skip
            + s * 3 * VOCAB  # causal depthwise smooth
        )
