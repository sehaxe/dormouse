"""KDA (gated-delta) attention, with FLA as an optional dependency.

The port's rule for attention is the readiness plan §7's: **FLA is a
dependency, and its absence is COUNTED, not silent.**

* The real arm is `fla.ops.kda.chunk_kda` (FLA 0.5.2) — the chunked
  **per-channel** gated-delta kernel, the op that takes the port's own
  `g [b, t, h, head_dim]` log-decay. `fla.ops.gated_delta_rule.chunk_gated_delta_rule`
  is the SCALAR-decay variant of the same recurrence, and a 4-D `g` handed to it
  runs and returns wrong numbers with no error (measured: 4.6e1 × the output RMS
  off the recurrence, while FLA's own naive reference matched at 1.5e-8). That is
  why this file names the op, not just "fla".
* Presence is proven by a **tiny warmup call at first use**, not by `import fla`
  succeeding: an installed-but-broken kernel (wrong arch, missing extension,
  sm_120 refusal) must fall back the same way a missing one does. The proof is
  per device TYPE — a CPU tensor on a CUDA box gets the fallback too, because
  FLA's kernels are CUDA-only and the CPU test path is the SDPA arm.
* The fallback is causal `scaled_dot_product_attention`, fp32-legal on CPU, and
  it increments a counter, so the log line reads `kda=sdpa,n=<calls>`.

WHAT RUNS OUTSIDE THE FUSED ARM, and why (COUNTED, not hidden):

* Our decay and write strength: `g = logsigmoid(g_proj(x))` (per channel) and
  `beta = sigmoid(beta_proj(x))` (per head) are computed in torch and passed in.
  FLA's in-kernel gate (`use_gate_in_kernel=True`) wants `A_log [H]` +
  `dt_bias [H·K]` parameters the port's set does not carry — the preset's count
  is pinned at `q,k,v,g = 4·d²` plus `beta = d·h` by `tests/test_param_count.py`,
  and what happened to be installed must not change the model's parameter count.
* The q/k L2 norm IS inside the kernel (`use_qk_l2norm_in_kernel=True`), which is
  where burn-kda has it too (K3: `q,k = L2Norm(ShortConv(Linear(x)))`). The port's
  projections are raw, and without that norm the gated-delta recurrence is
  numerically explosive (measured: 1e17 by t=64 at unit-variance q/k). **The
  SDPA fallback applies the same `F.normalize` explicitly**, so both arms
  implement K3's normalization and neither is anchored on inputs the other
  never saw.
* NOT in this model at all, so in neither arm: burn-kda's short convolutions, its
  output `Sigmoid(W_g x) ⊙ RMSNorm(o)` gate and its `o_proj`. Those are
  parameters the port does not have (the loop block norms BEFORE the layer and
  owns `out_proj` as its readout) — absent from the set, not inexpressible in FLA.

NUMERIC CONTRACT (the plan's rule): **the chunked kernel is NOT
tensor-comparable to burn-kda's KDA arm, and no cross-substrate bitwise claim exists
or can exist.** At the port's test shape (b1, h8, d_head 64, seq 256) it sits
1.3e-2 (fp32) / 4.9e-2 (bf16) of the output RMS away from the exact recurrence;
the spike measured 0.3 % / 1.8 % mean-relative at its own shape. The anchor is a
naive recurrent reference (`tests/test_kda_gpu.py`), which FLA's own
`naive_recurrent_kda` agrees with to 1.5e-8 in fp32. The bound asserted on it is
a WIRING gate (a transposed head axis is O(1) off), not a numerics certificate.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import Counters

#: Cached warmup verdict per device TYPE: `{"cuda": (ok, why), "cpu": (ok, why)}`.
_FLA: dict[str, tuple[bool, str]] = {}


def fla_status(device: torch.device | str = "cuda", recheck: bool = False) -> tuple[bool, str]:
    """Is FLA usable on THIS device type? Validated by a tiny call, once per process."""
    key = torch.device(device).type
    if key in _FLA and not recheck:
        return _FLA[key]
    if key != "cuda" or not torch.cuda.is_available():
        # Legal, not a defect: the CPU test path is the SDPA fallback.
        verdict = (False, f"device {key}: FLA kernels are CUDA-only")
    else:
        try:
            from fla.ops.kda import chunk_kda  # noqa: F401
        except Exception as e:  # ImportError, but also a broken extension
            verdict = (False, f"import fla failed: {type(e).__name__}: {e}")
        else:
            try:
                _warmup()
            except Exception as e:
                verdict = (False, f"warmup call failed: {type(e).__name__}: {e}")
            else:
                verdict = (True, "warmup ok")
    _FLA[key] = verdict
    return verdict


def _warmup() -> None:
    """One tiny chunked-KDA call at (b=1, t=8, h=1, d=16) — the presence proof.

    Same op, same flags and same input layout as the production path, so what it
    proves is the arm, not the import.
    """
    from fla.ops.kda import chunk_kda

    dev = torch.device("cuda")
    q = torch.randn(1, 8, 1, 16, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, 8, 1, 16, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, 8, 1, 16, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(1, 8, 1, 16, device=dev, dtype=torch.float32)
    beta = torch.rand(1, 8, 1, device=dev, dtype=torch.bfloat16)
    out, _ = chunk_kda(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    assert out.shape == q.shape, f"warmup shape {tuple(out.shape)} != {tuple(q.shape)}"
    if not torch.isfinite(out).all():
        raise RuntimeError("warmup produced non-finite output")


class KDAAttention(nn.Module):
    """`d → d` gated-delta attention: `q,k,v,g` projections plus per-head `beta`.

    The parameter set is the burn side's (KDA's own shape, `schema.rs`'s
    "q,k,v,g per-channel + beta" = `4·d² + d·n_heads`), and it is built the
    same way whatever device or library is present — the preset's parameter
    count must not depend on what happened to be installed.

    Both arms return `[b, t, d]` in the input's dtype; the fused arm runs its
    matmuls at `compute_dtype` (the port's bf16 plan: bf16 inside the arm, fp32
    masters and fp32 state between arms), the SDPA fallback runs at whatever the
    projections produced.
    """

    def __init__(self, cfg, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        assert h * cfg.head_dim == d, "validate() pins n_heads*head_dim == d_model"
        self.n_heads = h
        self.head_dim = cfg.head_dim
        self.scale = 1.0 / math.sqrt(cfg.head_dim)
        self.compute_dtype = torch.bfloat16 if cfg.bf16 else torch.float32
        for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
            setattr(self, name, nn.Linear(d, d, bias=False, device=device))
        self.beta_proj = nn.Linear(d, h, bias=False, device=device)

    def projections(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`x [b, t, d]` → `q, k, v [b, t, h, hd]`, `g [b, t, h, hd]` log decay, `beta [b, t, h]`.

        The `[B, T, H, ·]` layout is FLA 0.5.2's (`head_first` was removed from
        these ops) and it is also the port's head split, so nothing is
        transposed on the way into the kernel. `g` is log-space and fp32, which
        is what both the kernel and the recurrence reference read.
        """
        b, t, _ = x.shape
        h, hd = self.n_heads, self.head_dim
        q = self.q_proj(x).reshape(b, t, h, hd)
        k = self.k_proj(x).reshape(b, t, h, hd)
        v = self.v_proj(x).reshape(b, t, h, hd)
        g = F.logsigmoid(self.g_proj(x).float()).reshape(b, t, h, hd)
        beta = torch.sigmoid(self.beta_proj(x).float())  # [b, t, h]
        return q, k, v, g, beta

    def _heads(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        return x.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [b, t, d]` (block-body input) → `[b, t, d]`."""
        b, t, d = x.shape
        counters = Counters() if counters is None else counters
        ok, why = fla_status(x.device)
        if ok:
            counters.count("kda_fla")
            return self._fla_path(x)
        counters.count("kda_sdpa")
        # The reason is attached to the counter object, not printed per forward:
        # `log line` prints one `kda=` field, and the reason belongs there.
        counters.why = why  # type: ignore[attr-defined]
        return self._sdpa_path(x, b, t, d)

    def _fla_path(self, x: torch.Tensor) -> torch.Tensor:
        """The real arm: FLA 0.5.2's chunked per-channel gated-delta kernel."""
        from fla.ops.kda import chunk_kda

        b, t, d = x.shape
        dt = self.compute_dtype
        q, k, v, g, beta = self.projections(x)
        if dt != torch.float32:
            q, k, v, beta = q.to(dt), k.to(dt), v.to(dt), beta.to(dt)
        out, _ = chunk_kda(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=self.scale,
            use_qk_l2norm_in_kernel=True,  # burn-kda K3: L2Norm on q/k, in kernel
        )
        assert out.shape == (b, t, self.n_heads, self.head_dim), f"chunk_kda {tuple(out.shape)}"
        return out.reshape(b, t, d)

    def _sdpa_path(self, x: torch.Tensor, b: int, t: int, d: int) -> torch.Tensor:
        """COUNTED fallback: causal SDPA over L2-normalized q/k, g as an output gate.

        Counted because it is a DEGRADATION, not a different name for the same
        function: the delta-rule recurrence and `beta` do not run here. The
        counter on the log line is how a reader tells which arm produced a loss.

        q/k are L2-normalized BEFORE the attention, mirroring the fused arm's
        `use_qk_l2norm_in_kernel=True`: burn-kda K3 normalizes q/k, and an
        un-normalized fallback would make the two arms two different models
        (raw q/k are 1e17-explosive, see the module docs).
        """
        q = F.normalize(self._heads(self.q_proj(x), b, t), dim=-1)
        k = F.normalize(self._heads(self.k_proj(x), b, t), dim=-1)
        v = self._heads(self.v_proj(x), b, t)
        y = F.scaled_dot_product_attention(q * self.scale, k, v, is_causal=True)
        gate = torch.sigmoid(self.g_proj(x))  # [b, t, d]
        y = y.transpose(1, 2).reshape(b, t, d) * gate
        return y
