"""The `ffn_form="dense"` FFN: three plain linears, no rank anywhere.

`gate [f, d]` (bias `f`), `up [f, d]` (no bias), `down [d, f]` (bias `d`),
SwiGLU/SiTU/MemGLU between them. That is `3·d·f + f + d` parameters, the
baseline this project had been calling "dense" while `DualLatentGLU` was still
routing every expert through a rank-`rank` bottleneck (`experts.py`).

The bank is deliberately absent: no router, no shared expert, no shared
`down`, no `share_u_mode`. `config.validate_ffn` refuses those on this form,
and `SparseMoE` routes here for the expert slot so the rest of the block's
seam (`_bank_dtype`, `begin_step`, `aux_loss`) is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import Counters
from .act_quant import fake_quant_act
from .experts import expert_glu
from .qat_cache import QuantCache
from .tsct import dense_qat_weight


class DenseFFN(nn.Module):
    """`x [n, d] → [n, d]` through `down(glu(gate(x), up(x)))`."""

    ffn_form = "dense"

    def __init__(self, cfg, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        d, f = cfg.d_model, cfg.d_ffn
        self.d, self.f = d, f
        self.compute_dtype = torch.bfloat16 if cfg.bf16 else torch.float32
        self.expert_act = cfg.expert_act
        self.beta1 = cfg.expert_beta1
        self.beta2 = cfg.expert_beta2
        self.c0 = cfg.memglu_c0
        self.act_quant = cfg.act_quant
        self.act_group = cfg.act_group
        self.dense_qat = cfg.dense_qat
        self.dense_qat_alpha = cfg.factor_qat_alpha
        self._qat_cache = QuantCache() if self.dense_qat != "none" else None
        #: The GLU's two d→f halves and the f→d write. The gate carries the
        #: hidden bias and `down` the output bias, which is exactly the
        #: `+ f + d` in the parameter bill (the up projection has none: a second
        #: hidden bias would be a different count and a different model).
        self.gate = nn.Linear(d, f, bias=True, device=device)
        self.up = nn.Linear(d, f, bias=False, device=device)
        self.down = nn.Linear(f, d, bias=True, device=device)
        #: The routed-arm spellings the block seam reads. A dense FFN is not a
        #: bank: `topk=0`/`capacity=0` are the values that make the decoder's
        #: capacity checks correctly inert.
        self.topk = 0
        self.capacity = 0.0

    def _glu(self, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return expert_glu(self.expert_act, g, u, beta1=self.beta1, beta2=self.beta2, c0=self.c0)

    def _linear(self, linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        w = dense_qat_weight(
            linear.weight, self.dense_qat, self.dense_qat_alpha, cache=self._qat_cache
        )
        if self.compute_dtype != torch.float32:
            x = x.to(self.compute_dtype)
            w = w.to(self.compute_dtype)
            b = None if linear.bias is None else linear.bias.to(self.compute_dtype)
        else:
            b = linear.bias
        return F.linear(x, w, b)

    def forward(
        self,
        x: torch.Tensor,
        glu=None,
        counters: Counters | None = None,
    ) -> torch.Tensor:
        if self.act_quant != "none":
            x = fake_quant_act(x, self.act_quant, self.act_group)
        g = self._linear(self.gate, x)
        u = self._linear(self.up, x)
        h = self._glu(g, u) if glu is None else glu(g, u)
        y = self._linear(self.down, h)
        if counters is not None:
            counters.count("ffn_dense_forward", x.shape[0])
        return y if self.compute_dtype == torch.float32 else y.to(torch.float32)

    def _bank_dtype(self) -> torch.dtype:
        """The seam's name for the compute dtype (`LoopBlock._moe_branch`)."""
        return self.compute_dtype

    def begin_step(self) -> None:
        """No routed `aux` accumulator: kept so the model's per-step seam is one call."""

    def aux_loss(self) -> torch.Tensor | None:
        return None

    def invalidate_banks(self) -> None:
        """No bank cache to drop; the trainer's one per-step seam still calls this."""

    def extra_repr(self) -> str:
        return f"d={self.d}, f={self.f}, act={self.expert_act}, dtype={self.compute_dtype}"
