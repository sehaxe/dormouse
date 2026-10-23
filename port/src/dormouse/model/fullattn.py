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
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import Counters
from .tsct import dense_qat_kind, qat_linear

__all__ = ["FullAttention"]


class FullAttention(nn.Module):
    """`d → d` exact causal attention: `q,k,v,o` projections, SDPA over heads.

    Same constructor/call shape as [`KDAAttention`], so `LoopBlock._attn` is the
    ONE seam both arms run through: `forward(x [b,t,d], counters=...) → [b,t,d]`
    in the input's dtype, fp32 masters, bf16 inside the arm when `cfg.bf16`
    (the port's bf16 plan — the same cast KDA's fused arm makes).
    """

    def __init__(self, cfg, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        d, h, hd = cfg.d_model, cfg.n_heads, cfg.head_dim
        assert h * hd == d, "validate() pins n_heads*head_dim == d_model"
        self.n_heads = h
        self.head_dim = hd
        self.scale = 1.0 / math.sqrt(hd)
        self.compute_dtype = torch.bfloat16 if cfg.bf16 else torch.float32
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

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [b, t, d]` (block-body input) → `[b, t, d]`.

        `is_causal=True` is the exact causal mask: query `p` reads keys `0..p`.
        One counter per call (`attn_full_forward`) and no host read — the
        counter is the log line's evidence that THIS arm produced the number.
        """
        counters = Counters() if counters is None else counters
        counters.count("attn_full_forward")
        b, t, d = x.shape
        dq = (self.dense_qat, self.dense_qat_alpha)
        q, k, v = self.projections(x)
        dt = self.compute_dtype
        if dt != x.dtype:
            q, k, v = q.to(dt), k.to(dt), v.to(dt)
        # [b, t, h, hd] → [b, h, t, hd]: SDPA's query/key/value layout.
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            scale=self.scale,
        )
        y = y.transpose(1, 2).reshape(b, t, d)
        if dt != x.dtype:
            y = y.to(x.dtype)
        return qat_linear(self.o_proj, y, *dq)
