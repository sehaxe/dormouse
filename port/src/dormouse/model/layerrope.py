"""LayerRoPE: dynamic depth-wise magnitude & angular superposition.

Paper: *LayerRoPE: Dynamic Depth-wise Magnitude & Angular Superposition*
(arXiv:2610.09179). In a deep residual stack the per-layer normalisation gain
`gamma` empirically encodes **depth**: its magnitude grows with depth and its
direction rotates. LayerRoPE makes that explicit instead of learning one free
`[d]` vector per layer/site:

    gamma_i = amp_i * rotate_pairs(gamma_shared, theta_i)

    rotate_pairs(v, theta)[2k]   = v[2k]   * cos(theta) - v[2k+1] * sin(theta)
    rotate_pairs(v, theta)[2k+1] = v[2k]   * sin(theta) + v[2k+1] * cos(theta)

ONE shared vector `gamma_shared` of shape `[d]`, and per-depth scalars
`(amp_i, theta_i)`: `d + 2·n_depths` parameters instead of `n_sites·d`. The
rotation is a RoPE-style pair rotation broadcast over every adjacent channel
pair, so the FLOP change is a handful of elementwise ops per norm site
(`< 0.02%` of a step), all device-side.

At `amp_init=1.0, theta_init=0.0` the mechanism is an EXACT no-op:
`rotate_pairs(gamma_shared, 0) == gamma_shared` bit for bit (`y·0` is a signed
zero and `x - 0 == x`), and multiplying by `1.0` is exact — so the identity
initialisation reproduces the `layerrope=false` forward to the last bit (the
gate pins it).

Depth indexing in THIS repo
---------------------------
The model is a looped block, not a 512-layer stack:

    embed → prelude(0..P-1) → LoopBlock ×T → coda(0..C-1) → RMSNorm → lm_head

The depth index of a norm site is therefore:

* prelude layer `j`: `j`, for `j in 0..P-1`;
* loop iteration `i`: `P + i`, where **`i` is the ACTUAL iteration number**
  the loop ran, not the position of the run in a compacted list. With
  `rand_depth`/adaptive depth running only `T' < max_iter` iterations, the
  run uses depths `P..P+T'-1`; the coda and the final norm stay anchored at
  `P + max_iter + ...`, so a shorter run selects a PREFIX of the depth table
  and never shifts a later depth onto an earlier one. (A compacted "position
  in the run" index would put the coda at `P+T'` and silently reuse the
  depth of a loop iteration that a longer run computes differently.)
* coda layer `j`: `P + max_iter + j`;
* the model's final norm: the terminal depth `P + max_iter + C`.

`n_depths = P + max_iter + C + 1` in every configuration (`C` may be 0).
All norm SITES at one depth share the same `gamma_i` — the paper's single
shared vector — unless `layerrope_per_site=true`, which gives one
`(amp, theta)` pair per `(depth, site)` for a follow-up A/B. The site index is
`prelude j`, `loop body b`, `coda j`, final norm; `n_sites = P + loop_blocks +
C + 1`.

State-dict layout (only when `layerrope=true`; the OFF tree is unchanged)
------------------------------------------------------------------------
    layerrope.gamma_shared   [d]
    layerrope.amp            [n_depths]            (or [n_depths*n_sites])
    layerrope.theta          [n_depths]            (or [n_depths*n_sites])
                             a buffer when `layerrope_learn_theta=false`

The former per-site keys (`loop.norm.weight`, `norm.weight`,
`prelude.<j>.norm.weight`, `coda.<j>.norm.weight`) are ABSENT: the gains are
replaced, not augmented, so a resume across the flag is a different parameter
tree and is refused by the snapshot-drift gate.

Interactions (why they are allowed)
-----------------------------------
`use_mhc`, `use_attnres`, `use_csa`, `attn_window`, `segment_len`, `rand_depth`,
`loop_blocks`, `prelude_layers`/`coda_layers`, `tie_embeddings`, `ema_decay`,
`grad_accum` are independent of the norm gain: they change the residual or the
schedule, not which vector scales the norm. `muon_sf` is independent too: the
gain table is 1-D, so `optim.is_muon_param` routes it to AdamW exactly as the
RMSNorm gains it replaces. `factor_qat`/`dense_qat` do NOT walk norm gains
(their seam is `nn.Linear`/explicit weight tensors), so a LayerRoPE gain is
never half-quantised.

REFUSED: `use_byteflow` (its channel path returns before this tree is built
and its loop runs over patch latents), an odd `d_model` (the rotation needs
pairs), and — in `config.validate_model_train` — `full_graph`/`opt_graph`: the
per-depth gather is a device op, but the capture would freeze the depth index
and the identity is not worth proving inside a graph before it is measured.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import loud

__all__ = ["LayerRoPEGains", "LayerRoPENorm", "rotate_pairs"]


def rotate_pairs(v: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """RoPE-style pair rotation of `v [..., d]` by the 0-dim `theta`.

    Every adjacent pair `(v[2k], v[2k+1])` rotates by the SAME angle — that is
    the paper's "angular superposition": one angle per depth, not a per-channel
    frequency ladder (this is not positional RoPE). `d` must be even.
    """
    assert v.shape[-1] % 2 == 0, f"rotate_pairs needs an even width, got {v.shape[-1]}"
    pairs = v.reshape(*v.shape[:-1], -1, 2)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    x = pairs[..., 0]
    y = pairs[..., 1]
    return torch.stack((x * cos - y * sin, x * sin + y * cos), dim=-1).reshape(v.shape)


class LayerRoPEGains(nn.Module):
    """`gamma_shared [d]` + per-depth `(amp, theta)`; `gain(depth, site)` returns `gamma_i`.

    One instance is owned by `DormouseModel` and shared by every norm site, so
    the parameter tree carries ONE `gamma_shared` for the whole model. The
    learnable table is flat (`n_depths` entries, or `n_depths·n_sites` when
    `per_site`) so the optimizer never routes a small gain matrix to Muon.

    `_applications` is a DEVICE counter (one per `gain` call) drained by the
    trainer on the log cadence (`Counters.count_device`), so the proof the arm
    ran costs no host read.
    """

    def __init__(
        self,
        d: int,
        n_depths: int,
        *,
        n_sites: int = 1,
        per_site: bool = False,
        amp_init: float = 1.0,
        theta_init: float = 0.0,
        learn_theta: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if d % 2:
            loud(
                f"layerrope with d_model={d}: the rotation acts on adjacent channel "
                f"PAIRS, so the width must be even. Escape: an even d_model, or "
                f"layerrope=false."
            )
        if n_depths < 1:
            loud(f"layerrope n_depths={n_depths}: at least one depth is required")
        self.d = int(d)
        self.n_depths = int(n_depths)
        self.per_site = bool(per_site)
        self.n_sites = int(n_sites) if per_site else 1
        n = self.n_depths * self.n_sites
        self.gamma_shared = nn.Parameter(torch.ones(d, device=device))
        self.amp = nn.Parameter(torch.full((n,), float(amp_init), device=device))
        theta = torch.full((n,), float(theta_init), device=device)
        if learn_theta:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer("theta", theta)
        #: Per-forward application count, drained at the log cadence. NOT in the
        #: state dict (a checkpoint must not carry a run's telemetry).
        self.register_buffer(
            "_applications", torch.zeros((), dtype=torch.int64, device=device), persistent=False
        )

    def _index(self, depth: int, site: int) -> int:
        assert 0 <= depth < self.n_depths, f"layerrope depth {depth} outside 0..{self.n_depths}"
        if not self.per_site:
            return depth
        assert 0 <= site < self.n_sites, f"layerrope site {site} outside 0..{self.n_sites}"
        return depth * self.n_sites + site

    def gain(self, depth: int, site: int = 0) -> torch.Tensor:
        """`amp_i · rotate_pairs(gamma_shared, theta_i)` for one depth/site."""
        idx = self._index(depth, site)
        rotated = rotate_pairs(self.gamma_shared, self.theta[idx])
        self._applications.add_(1)
        return rotated * self.amp[idx]

    def take_applications(self) -> torch.Tensor:
        """The applications since the last take, as a DEVICE scalar (no host read)."""
        n = self._applications.clone()
        self._applications.zero_()
        return n


class LayerRoPENorm(nn.Module):
    """RMSNorm whose gain is the shared LayerRoPE `gamma_i` — no own parameter.

    `F.rms_norm` is the call `nn.RMSNorm.forward` makes, so with
    `gamma_i == ones` this module computes the OFF path's bits. `depth=None`
    uses the fixed depth bound at construction (prelude/coda/final norm); the
    loop block passes the ACTUAL iteration number per call.
    """

    def __init__(
        self,
        d: int,
        eps: float,
        gains: LayerRoPEGains,
        *,
        depth: int = 0,
        site: int = 0,
    ) -> None:
        super().__init__()
        self.d = int(d)
        self.eps = float(eps)
        self.depth = int(depth)
        self.site = int(site)
        # The gains module is owned by `DormouseModel`; a normal attribute would
        # register it here too and duplicate its keys in `state_dict` under
        # every site. `object.__setattr__` keeps the reference without the
        # registration — one shared vector, one set of keys.
        object.__setattr__(self, "_gains", gains)

    def forward(self, x: torch.Tensor, depth: int | None = None) -> torch.Tensor:
        d = self.depth if depth is None else depth
        gain = self._gains.gain(d, self.site)
        return F.rms_norm(x, (self.d,), gain, self.eps)
