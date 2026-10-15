"""KDA (gated-delta) attention, with FLA as an optional dependency.

The port's rule for attention is the readiness plan §7's: **FLA is a
dependency, and its absence is COUNTED, not silent.**

* `fla` (flash-linear-attention) is the lineage burn-kda was transcribed from;
  on the torch path the fused gated-delta backward is a dependency rather than
  a repair item (plan §1.2's fused-adjoint debt dissolves there).
* Presence is proven by a **tiny warmup call at first use**, not by
  `import fla` succeeding: an installed-but-broken kernel (wrong arch, missing
  extension, sm_120 refusal) must fall back the same way a missing one does.
  The spike's own gate #1 is exactly this question, asked on the card.
* The fallback is causal `scaled_dot_product_attention`, fp32-legal on CPU, and
  it increments a counter, so the log line reads `kda=sdpa,n=<calls>`.

What the fallback does NOT run, and this is the honest part: the gated-delta
recurrence and the per-head write strength `beta`. `beta` is therefore a
parameter with no gradient on the fallback path — the burn-side defect class
(`8fa5d4c`: an arm that ran thousands of forwards and trained nothing) whose
only defence is the counter. The decay gate `g` is NOT dropped: it becomes the
sigmoid output gate the GDN playbook names (AGENTS §3.5 item 5), so the arm's
decay parameter still receives a gradient on the fallback.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import Counters

#: Cached warmup verdict: `{"checked": bool, "ok": bool, "why": str}`.
_FLA: dict[str, object] = {"checked": False, "ok": False, "why": ""}


def fla_status(recheck: bool = False) -> tuple[bool, str]:
    """Is FLA usable HERE? Validated by a tiny call, once per process."""
    if _FLA["checked"] and not recheck:
        return bool(_FLA["ok"]), str(_FLA["why"])
    _FLA["checked"] = True
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
    except Exception as e:  # ImportError, but also a broken extension
        _FLA["ok"] = False
        _FLA["why"] = f"import fla failed: {type(e).__name__}: {e}"
        return False, str(_FLA["why"])
    if not torch.cuda.is_available():
        # Legal, not a defect: the CPU test path is the SDPA fallback.
        _FLA["ok"] = False
        _FLA["why"] = "no CUDA device: FLA kernels are CUDA-only"
        return False, str(_FLA["why"])
    try:
        _warmup()
    except Exception as e:
        _FLA["ok"] = False
        _FLA["why"] = f"warmup call failed: {type(e).__name__}: {e}"
        return False, str(_FLA["why"])
    _FLA["ok"] = True
    _FLA["why"] = "warmup ok"
    return True, "warmup ok"


def _warmup() -> None:
    """One tiny gated-delta call at (b=1, h=1, t=8, d=16) — the presence proof."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    dev = torch.device("cuda")
    q = torch.randn(1, 1, 8, 16, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 8, 16, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, 1, 8, 16, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(1, 1, 8, 16, device=dev, dtype=torch.float32)
    beta = torch.rand(1, 1, 8, 1, device=dev, dtype=torch.bfloat16)
    out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta)
    assert out.shape == q.shape, f"warmup shape {tuple(out.shape)} != {tuple(q.shape)}"
    if not torch.isfinite(out).all():
        raise RuntimeError("warmup produced non-finite output")


class KDAAttention(nn.Module):
    """`d → d` gated-delta attention: `q,k,v,g` projections plus per-head `beta`.

    The parameter set is the burn side's (KDA's own shape, `schema.rs`'s
    "q,k,v,g per-channel + beta" = `4·d² + d·n_heads`), and it is built the
    same way whatever device or library is present — the preset's parameter
    count must not depend on what happened to be installed.
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

    def _heads(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        return x.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, counters: Counters | None = None) -> torch.Tensor:
        """`x [b, t, d]` (block-body input) → `[b, t, d]`."""
        b, t, d = x.shape
        counters = Counters() if counters is None else counters
        ok, why = fla_status()
        if ok:
            counters.count("kda_fla")
            return self._fla_path(x, counters)
        counters.count("kda_sdpa")
        # The reason is attached to the counter object, not printed per forward:
        # `log line` prints one `kda=` field, and the reason belongs there.
        counters.why = why  # type: ignore[attr-defined]
        return self._sdpa_path(x, b, t, d)

    def _fla_path(self, x: torch.Tensor, counters: Counters) -> torch.Tensor:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        b, t, d = x.shape
        q = self._heads(self.q_proj(x), b, t)
        k = self._heads(self.k_proj(x), b, t)
        v = self._heads(self.v_proj(x), b, t)
        g = F.logsigmoid(self.g_proj(x)).reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta)
        return out.transpose(1, 2).reshape(b, t, d)

    def _sdpa_path(self, x: torch.Tensor, b: int, t: int, d: int) -> torch.Tensor:
        """COUNTED fallback: causal SDPA over q/k/v, g as a sigmoid output gate.

        Counted because it is a DEGRADATION, not a different name for the same
        function: the delta-rule recurrence and `beta` do not run here. The
        counter on the log line is how a reader tells which arm produced a loss.
        """
        q = self._heads(self.q_proj(x), b, t)
        k = self._heads(self.k_proj(x), b, t)
        v = self._heads(self.v_proj(x), b, t)
        y = F.scaled_dot_product_attention(q * self.scale, k, v, is_causal=True)
        gate = torch.sigmoid(self.g_proj(x))  # [b, t, d]
        y = y.transpose(1, 2).reshape(b, t, d) * gate
        return y
