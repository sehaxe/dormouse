"""Exact causal attention — the FULL arm beside KDA's compressed-state arm.

WHY THIS FILE EXISTS. Every number on record ran `KDAAttention`: a gated-delta
recurrence whose state is `H × head_dim × head_dim`, i.e. a LINEAR attention
that compresses the window before the output is read. The model it is compared
against (GPT-2 small, `port/scripts/bench_hf.py`'s protocol) runs 12 layers of
EXACT causal attention at 1 024 tokens. Without this arm the port has no way to
ask whether the gap on record is the attention, and `use_kda=false` is not that
arm: it is the degenerate zero branch (`loopblock._window`, bit-identical to
the model before this file).

THE OP, AND WHY IT IS `torch.nn.functional.scaled_dot_product_attention` AND
NOT FLA's dense kernel. FLA 0.5.2 ships `fla.ops.attn.parallel_attn`; its
signature was read in the installed wheel, not guessed:
`parallel_attn(q [B,T,HQ,K], k [B,T,H,K], v [B,T,H,V], g=None, scale=None,
window_size=None, cu_seqlens=None, chunk_indices=None, *, sink_bias=None)`,
returning `o [B,T,HQ,V]` only — there is no `causal` argument, causality is the
kernel's own. The layout FITS ours (`[b,t,h,head_dim]`), but the kernel is
Triton/CUDA-only: the gate suite runs CPU-only (`CUDA_VISIBLE_DEVICES=""`), so
honouring it would force a second, always-CPU implementation and the two paths
the task forbids. SDPA is ONE path on both devices (flash/mem-efficient kernels
on CUDA, the math backend on CPU), exact, causal, and differentiable — the
same op KDA's COUNTED fallback already uses (`kda._sdpa_path`), so the two
attention arms share their linear algebra and differ in what they compress.

PARAMETERS. `q,k,v,o` are four `d×d` projections = `4·d²`; KDA is
`q,k,v,g` (`4·d²`) plus `beta` (`d·n_heads`). The delta is therefore
`-d·n_heads`: at `medium` (`d=1024, n_heads=16`) 4 210 688 → 4 194 304,
i.e. **-16 384** parameters, total 141 910 405 → 141 894 021 (`dormouse info`).

NO STATE. Exact attention has no `O(1)` boundary state (the previous segments'
K/V are the context, `O(window)`), so `forward_state`/`step_state` do not
exist here. `segment_len>0` and the recurrent decode/prefill paths are refused
LOUDLY at `config._validate_segment`, `decode.recurrent_refusal` and
`LoopBlock._attn` — they name `attn_kind`, the reason and the escape.

THE LOCAL BAND (`window>0`, `ModelConfig.attn_window`). The same class, called
with a band instead of a full causal mask: query `p` reads keys
`max(0, p-W+1)..p` and NOTHING older. It is added BESIDE KDA/CSA
(`LoopBlock._window` sums the attention paths), so `kda + window` and
`kda + csa + window` are the arms — the band restores the exact local detail a
compressed state cannot hold. `torch.nn.functional.scaled_dot_product_attention`
on this stack has NO `window_size` argument (measured on torch 2.13.0+cu129:
`TypeError: unexpected keyword argument 'window_size'`), so the band is an
explicit boolean mask. `query_chunk>0` slices the queries, which caps the mask
at `C × (C+W-1)` per call — a MEMORY-only lever whose output equals the
unchunked call's to round-off (`tests/test_long4m.py`, gate 5).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..instrument import Counters
from .tsct import dense_qat_kind, qat_linear

__all__ = ["FullAttention", "window_mac_per_byte"]


def window_mac_per_byte(window: int, d_model: int) -> float:
    """The local window's cost per context byte: `2·W·d` MAC.

    `W·d` for the score against every surviving key, `W·d` for the weighted
    value — FLAT in the context length, which is the whole point of a banded
    arm: it is the only attention bill in the block that does not grow with `t`.
    """
    assert window >= 0 and d_model >= 1, (window, d_model)
    return 2.0 * window * d_model


class FullAttention(nn.Module):
    """`d → d` exact causal attention: `q,k,v,o` projections, SDPA over heads.

    Same constructor/call shape as [`KDAAttention`], so `LoopBlock._attn` is the
    ONE seam both arms run through: `forward(x [b,t,d], counters=...) → [b,t,d]`
    in the input's dtype, fp32 masters, bf16 inside the arm when `cfg.bf16`
    (the port's bf16 plan — the same cast KDA's fused arm makes).
    """

    def __init__(
        self,
        cfg,
        *,
        device: torch.device | str = "cpu",
        window: int = 0,
        query_chunk: int = 0,
    ) -> None:
        super().__init__()
        d, h, hd = cfg.d_model, cfg.n_heads, cfg.head_dim
        assert h * hd == d, "validate() pins n_heads*head_dim == d_model"
        self.n_heads = h
        self.head_dim = hd
        self.scale = 1.0 / math.sqrt(hd)
        self.compute_dtype = torch.bfloat16 if cfg.bf16 else torch.float32
        #: see ModelConfig.sdpa_mem_efficient (measured 2.1x on sm_120)
        self.sdpa_mem_efficient = bool(getattr(cfg, "sdpa_mem_efficient", False))
        #: `window=0` is the EXACT arm every number on record ran: `is_causal`
        #: over the whole window, no mask tensor. `window>0` is the LOCAL band
        #: (`ModelConfig.attn_window`): key `g` is readable by query `i` iff
        #: `0 <= i-g <= window-1`. `query_chunk>0` slices the queries so the
        #: banded mask is `C × (C+W-1)` per block, never `t × t`.
        assert window >= 0 and query_chunk >= 0, (window, query_chunk)
        assert window or not query_chunk, (
            "query_chunk without a window: the blocks exist to cap the banded "
            "mask, and an exact causal call has no mask to cap."
        )
        self.window = int(window)
        self.query_chunk = int(query_chunk)
        #: `dense_qat`'s per-site rule, the same call shape KDA makes: all four
        #: projections are ATTENTION sites (`tsct.DENSE_QAT_ATTN_SITES`), so the
        #: default `dense_qat_skip_attn` keeps them bf16 and an explicitly
        #: quantized run treats the arm exactly as it treats KDA.
        self.dense_qat = dense_qat_kind(
            cfg,
            "loop.attn.q_proj",
            "loop.attn.k_proj",
            "loop.attn.v_proj",
            "loop.attn.o_proj",
        )
        self.dense_qat_alpha = cfg.factor_qat_alpha
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(d, d, bias=False, device=device))

    def projections(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`x [b, t, d]` → `q, k, v [b, t, h, head_dim]` (the FLA/port head layout)."""
        b, t, _ = x.shape
        dq = (self.dense_qat, self.dense_qat_alpha)
        q = qat_linear(self.q_proj, x, *dq).reshape(b, t, self.n_heads, self.head_dim)
        k = qat_linear(self.k_proj, x, *dq).reshape(b, t, self.n_heads, self.head_dim)
        v = qat_linear(self.v_proj, x, *dq).reshape(b, t, self.n_heads, self.head_dim)
        return q, k, v

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        counters: Counters,
        *,
        causal: bool,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One SDPA call, with the backend choice recorded (never silent).

        THE BACKEND IS NOT NEUTRAL ON THIS CARD. Measured 2026-10-27 on the
        RTX 5060 Ti (sm_120, torch 2.13.0+cu129), b=8 h=16 t=1024 hd=64 bf16
        causal: flash 1.91 ms, EFFICIENT 0.91 ms, math 19.05 ms. PyTorch picks
        flash by default, i.e. 2.1x SLOWER than the memory-efficient kernel on
        this device. `sdpa_mem_efficient` (default False, so every number on
        record keeps its backend) forces EFFICIENT and records which kernel
        actually ran in `sdpa_backend`, so a run cannot silently drift.

        `mask` is the banded boolean of the local window. `window_size=` is NOT
        an option here: this torch has no such argument on
        `F.scaled_dot_product_attention` (`TypeError: unexpected keyword
        argument 'window_size'`, measured on 2.13.0+cu129), so the band is an
        explicit mask on every backend — exact, and the one the chunked path
        slices.
        """
        kw = {"is_causal": causal, "scale": self.scale}
        if mask is not None:
            kw["attn_mask"] = mask
        if getattr(self, "sdpa_mem_efficient", False):
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                y = F.scaled_dot_product_attention(q, k, v, **kw)
            counters.count("sdpa_mem_efficient")
        else:
            y = F.scaled_dot_product_attention(q, k, v, **kw)
            counters.count("sdpa_default_backend")
        return y

    def _window_forward(self, x: torch.Tensor, counters: Counters) -> torch.Tensor:
        """The LOCAL band: query `p` reads keys `max(0, p-W+1)..p` and nothing older.

        One SDPA call per query block (`attn_window_query_chunk`, one block when
        it is 0). Block `[s, e)` needs keys `[max(0, s-W+1), e)` only, so the
        mask is `(e-s) × (e-k0)` — the memory a 4 MiB window would otherwise
        spend on a `t × t` band, and the reason the chunked arm exists. The
        blocks' outputs are concatenated, so the chunking is a memory decision:
        the gate prints max|Δ| against the unchunked call.
        """
        counters.count("attn_window_forward")
        b, t, d = x.shape
        dq = (self.dense_qat, self.dense_qat_alpha)
        q, k, v = self.projections(x)
        dt = self.compute_dtype
        if dt != x.dtype:
            q, k, v = q.to(dt), k.to(dt), v.to(dt)
        if t == 0:
            return qat_linear(self.o_proj, x.new_zeros(b, 0, d), *dq)
        qh = q.transpose(1, 2)  # [b, h, t, hd]
        kh = k.transpose(1, 2)
        vh = v.transpose(1, 2)
        pos = torch.arange(t, device=x.device)
        step = self.query_chunk or t
        outs = []
        for s in range(0, t, step):
            e = min(s + step, t)
            k0 = max(0, s - self.window + 1)
            band = pos[s:e].view(-1, 1) - pos[k0:e].view(1, -1)  # query - key
            mask = (band >= 0) & (band <= self.window - 1)
            outs.append(
                self._sdpa(
                    qh[:, :, s:e],
                    kh[:, :, k0:e],
                    vh[:, :, k0:e],
                    counters,
                    causal=False,
                    mask=mask,
                )
            )
            counters.count("attn_window_query_blocks")
        y = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
        y = y.transpose(1, 2).reshape(b, t, d)
        if dt != x.dtype:
            y = y.to(x.dtype)
        return qat_linear(self.o_proj, y, *dq)

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [b, t, d]` (block-body input) → `[b, t, d]`.

        `is_causal=True` is the exact causal mask: query `p` reads keys `0..p`.
        With `window>0` the call is the BANDED path instead (`_window_forward`).
        One counter per call (`attn_full_forward` / `attn_window_forward`) and
        no host read — the counter is the log line's evidence that THIS arm
        produced the number.
        """
        counters = Counters() if counters is None else counters
        if self.window:
            return self._window_forward(x, counters)
        counters.count("attn_full_forward")
        b, t, d = x.shape
        dq = (self.dense_qat, self.dense_qat_alpha)
        q, k, v = self.projections(x)
        dt = self.compute_dtype
        if dt != x.dtype:
            q, k, v = q.to(dt), k.to(dt), v.to(dt)
        # [b, t, h, hd] → [b, h, t, hd]: SDPA's query/key/value layout.
        y = self._sdpa(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            counters,
            causal=True,
        )
        y = y.transpose(1, 2).reshape(b, t, d)
        if dt != x.dtype:
            y = y.to(x.dtype)
        return qat_linear(self.o_proj, y, *dq)
