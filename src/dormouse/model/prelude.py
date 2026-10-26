"""The UNIQUE layers outside the shared loop — the prelude/coda topology.

`arXiv 2610.23036` ("Towards Looped Models Done Right, Part I") puts a few
UNIQUE layers before the shared loop and a few after it, instead of running the
whole stack `T` times: `embed → prelude(0..N) → LoopBlock ×T → coda(0..M) →
RMSNorm → lm_head`. The loop stays weight-shared; the prelude and the coda are
the layers that are NOT, and they are what lets the shared body specialize on
iterative refinement while the cheap per-layer work runs exactly once.

**The layer is a pre-norm residual SwiGLU FFN, not a second KDA copy** — the
choice is the parameter budget, measured: a full `KDAAttention` at `medium`
(d=1024, h=16) is `4·d² + d·h = 4 210 688` per block, so one prelude + one coda
would add 8.4M to a 141.9M model and blow the +2-6M envelope; this block at
`d_ff = d/2` is `3·d·d_ff + d = 1 573 888`, so 1+1 is +3.1M. It also keeps the
loop's attention the ONLY attention arm in the model: no second FLA/device
dependency, no second `_sdpa_path` fallback, and the loop's per-iteration state
contract is untouched.

The block is fp32 on the residual stream (the port's rule: fp32 state between
arms, `docs/architecture/bf16-plan.md`). It owns one RMSNorm, three dense
projections and NO bias — `3·d·d_ff` is the whole new parameter delta, which is
what `tests/test_prelude.py` pins against the instantiated tree.

`dense_qat` reaches these three weights like every other dense site
(`tsct.qat_linear`); they are not attention, so the default BitNet rule
quantizes them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .tsct import dense_qat_kind, qat_linear

__all__ = ["UniqueLayer"]


class UniqueLayer(nn.Module):
    """`h = h + down(silu(gate(norm(h))) * up(norm(h)))` — one, not shared."""

    def __init__(self, cfg, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        d = cfg.d_model
        #: Half the loop's FFN width: the budget at d=1024 is 1.57M per layer
        #: (module docstring), and the layer exists to add a unique transform,
        #: not a second full-width body beside the loop.
        d_ff = max(d // 2, 1)
        self.d = d
        self.d_ff = d_ff
        self.norm = nn.RMSNorm(d, eps=cfg.norm_eps, device=device)
        self.gate = nn.Linear(d, d_ff, bias=False, device=device)
        self.up = nn.Linear(d, d_ff, bias=False, device=device)
        self.down = nn.Linear(d_ff, d, bias=False, device=device)
        #: The DENSE path's switch (`tsct.dense_qat_project`): one kind for the
        #: three weights. The site names are not attention, so the default
        #: `dense_qat_skip_attn` keeps `cfg.dense_qat` here.
        self.dense_qat = dense_qat_kind(cfg, "prelude.gate", "prelude.up", "prelude.down")
        self.dense_qat_alpha = cfg.factor_qat_alpha

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """`h [b, t, d]` in the residual dtype → `[b, t, d]`, same shape/dtype."""
        assert h.ndim == 3 and h.shape[-1] == self.d, f"unique layer input {tuple(h.shape)}"
        x = self.norm(h)
        dq = (self.dense_qat, self.dense_qat_alpha)
        gi = F.silu(qat_linear(self.gate, x, *dq))
        y = qat_linear(self.down, gi * qat_linear(self.up, x, *dq), *dq)
        out = h + y
        assert out.shape == h.shape, f"unique layer wrote {tuple(out.shape)} into {tuple(h.shape)}"
        assert out.dtype == h.dtype, (
            f"unique layer changed the residual dtype {h.dtype} -> {out.dtype}"
        )
        return out
