"""TSCT: `LinearLike` reinterpreted in torch — `W = U diag(s) Vᵀ`.

`burn-spectral`'s `SpectralLinear` keeps the factors `U [in, r]`, `s [r]`,
`V [r, out]` and computes the effective weight on the fly, so a `d → f`
projection costs `r·(d + f + 1)` parameters instead of `d·f`. That is what
makes a 5461-wide FFN affordable, and it is what the retraction and the Muon
group operate on: the polar retraction is done in the FACTORED form, so fp32
Newton-Schulz over `[64, f]` replaces "~40 s/step" over `[d, d]` (AGENTS §2.3).

Precision (`docs/architecture/bf16-plan.md`): the factors are fp32 MASTERS;
the forward casts to bf16, matmuls, and returns fp32 — softmax/loss/KDA state
stay fp32 at the arm boundary. `quant = "none"` is the day-one format and it
resolves to bf16 factors; every ladder rung (fp8 → NVFP4 → LittleBit) is
refused at CONFIG load (see `config._refuse_ladder_rung`), so a rung can never
silently compute bf16 while the config reads `fp8`.

Deviations from burn, both deliberate and listed in the port report:

* **no `N % 4` output padding.** burn pads `out_features` to a multiple of 4
  and slices back because cubek's matmul vectorizes on `N % 4 == 0`
  (AGENTS §2.2). torch has no such constraint, so the port drops the pad — the
  parameter count is the honest `r·(in + out + 1)` either way.
* **no STE quantizer in the forward.** burn's factor quant is a
  straight-through estimator (`act_quant.rs`); the day-one port has no rung to
  run, so the only format is bf16 and no fake-quant path exists to be wrong.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..instrument import loud


class TSCTLinear(nn.Module):
    """A factorized linear: `x → ((x·U) ⊙ s)·V`, masters fp32, forward in `dtype`."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        bias: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if rank > min(in_features, out_features):
            loud(
                f"TSCT rank {rank} exceeds min(in={in_features}, out={out_features}): "
                f"a rank above either dimension is not a factorization (burn clamps "
                f"only the lm_head, explicitly, in `model.rs`). Escape: lower `rank` "
                f"or make the projection dense (`use_tsct = false`)."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.compute_dtype = dtype
        # U ~ N(0, 1/sqrt(in)), V ~ N(0, 1/sqrt(rank)): the composed map has
        # unit-ish output scale at any width, which matters because ReZero's
        # residual_scale is initialised to 1.0 (not 0 — at 0 the block body
        # gets no gradient at all, the 2026-09-27 burn bug).
        self.u = nn.Parameter(
            torch.randn(in_features, rank, device=device) / math.sqrt(in_features)
        )
        self.s = nn.Parameter(torch.ones(rank, device=device))
        self.v = nn.Parameter(torch.randn(rank, out_features, device=device) / math.sqrt(rank))
        self.bias = nn.Parameter(torch.zeros(out_features, device=device)) if bias else None

    def effective_weight(self) -> torch.Tensor:
        """`U diag(s) Vᵀ` in fp32 — the weight the optimizer and the report
        talk about. Never used in a forward (it would materialize `d·f`)."""
        return (self.u * self.s) @ self.v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = self.compute_dtype
        if dt == torch.float32:
            y = ((x @ self.u) * self.s) @ self.v
        else:
            # Matmuls in the factor dtype with fp32 accumulate (torch's own
            # reduction), the master params untouched.
            xu = x.to(dt)
            y = ((xu @ self.u.to(dt)) * self.s.to(dt)) @ self.v.to(dt)
            y = y.to(torch.float32)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"dtype={self.compute_dtype}"
        )


class DenseLinear(nn.Module):
    """The `use_tsct = false` counterpart: a plain `[out, in]` weight.

    It exists because TSCT's A/B is its own removal, not TSCT against nothing:
    at a matched parameter budget the comparison is wide-and-low-rank against
    narrow-and-dense (burn's `LinearLikeInner::Dense`, reachable since
    2026-09-28 for exactly this reason).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        bias: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = dtype
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device) / math.sqrt(in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features, device=device)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.compute_dtype != torch.float32:
            x = x.to(self.compute_dtype)
            w = w.to(self.compute_dtype)
        y = x @ w.t()
        if self.compute_dtype != torch.float32:
            y = y.to(torch.float32)
        if self.bias is not None:
            y = y + self.bias
        return y


def make_projection(
    in_features: int,
    out_features: int,
    cfg,
    *,
    rank: int | None = None,
    bias: bool = False,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Spectral or dense, chosen once from `use_tsct` (a construction-time
    decision, so the shape of the model cannot drift afterwards)."""
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    rank = cfg.rank if rank is None else rank
    if cfg.use_tsct:
        return TSCTLinear(in_features, out_features, rank, dtype=dtype, bias=bias, device=device)
    return DenseLinear(in_features, out_features, dtype=dtype, bias=bias, device=device)
