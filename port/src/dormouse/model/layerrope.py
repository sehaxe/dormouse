"""LayerRoPE: dynamic depth-wise magnitude & angular superposition.

Paper: *LayerRoPE: Dynamic Depth-wise Magnitude & Angular Superposition*
(arXiv:2610.09179). In a deep residual stack the per-layer normalisation gain
`gamma` empirically encodes **depth**: its magnitude grows with depth and its
direction rotates. LayerRoPE makes that explicit instead of learning one free
`[d]` vector per layer/site.

TWO FORMS, one flag (`layerrope_form`)
--------------------------------------
`"v1"` (default) is this file's first implementation: ONE shared `gamma_shared`
`[d]` + per-depth scalars `(amp_i, theta_i)` (`d + 2·n_depths` parameters),
applied at the NORM READ sites only, every channel pair rotated by the SAME
angle. It is what the `layerrope` branch shipped and every v1 gate pins; it is
unchanged here.

`"paper"` is the project page's reference snippet, transcribed below as
[`LayerRoPESite`] — `gamma[d]` + FOUR scalars per site (`a_mag`, `b_mag`,
`a_rot`, `b_rot`), a log-linear magnitude `r_l = a_mag + b_mag·log_l`, a
RoPE-spectrum rotation `ang_j = θ_l · base^(-2j/d)` (a per-PAIR angle), and the
`log_l` / `freq` buffers:

    r_l     = a_mag + b_mag · log_l
    θ_l     = exp(a_rot + b_rot · log_l)
    ang_j   = θ_l · base^(-2j/d)              (j = 0 … d/2-1)
    gain_l  = Re/Im-interleave( gamma · exp(r_l + i·ang) )

One site is `d + 4` parameters + 2 buffers; a whole block is FOUR sites —
`attn_read`, `attn_write`, `mlp_read`, `mlp_write` — i.e. `4·(d+4)`, O(d),
against the per-layer baseline `2·L·d`.

INIT: `layerrope_init="paper"` uses the paper's `beta_init=-0.5`, and it is NOT
an identity: `r_l = -0.5·log(l+1)` and `θ_l = (l+1)^-0.5` from the first
forward. `layerrope_init="identity"` is the bit-identity gate.

**A correction to the obvious reading of "identity".** Setting
`b_mag = b_rot = 0` gives `r = 0, θ = 1`, and `θ = 1` is NOT an unrotated gain
in the reference: `ang = θ·freq = freq ≠ 0`, so the gain would still be rotated
by the spectrum. `identity` therefore ALSO turns the spectrum off (it
short-circuits `table()` to `gamma` broadcast over depth) — otherwise the
`torch.equal(gain, gamma)` gate could not exist. That is the only place this
file departs from the snippet's arithmetic.

WHERE THE FOUR GAINS GO IN THIS MODEL (`loopblock.py`)
------------------------------------------------------
The paper's stack is `h = h + attn_write·attn(normalize(h)·attn_read)` then
`h = h + mlp_write·mlp(normalize(h)·mlp_read)`. Our block computes ONE
`normalize(ctx)` and TWO branch sums joined into one residual update `y`
(`loopblock._window`), so the read gains are the norm's affine —
`F.rms_norm(ctx, (d,), attn_read, eps)` and `… , mlp_read, …`, one fused call
each — and the write gains multiply each branch's OUTPUT where it enters `y`:
`attn·attn_write`, `ffn·mlp_write`, immediately before `y = attn + branch +
ffn`. `y` is exactly the `F(H_pre·x_l)` the residual operator (mHC's
`H_post`, AttnRes' write, ReZero's `α`) deposits, so the write gain sits at the
branch→residual seam and NOT inside any mixer's internals. Because mHC/AttnRes
are LINEAR in `y`, scaling the branch at that seam is the paper's
`h + write·F(...)` for every residual arm this repo has.

Our block does NOT have the paper's four sites for free — it has a shared norm
and a third branch (the Engram memory read). The mapping is stated, not
invented: `attn_read` feeds the attention arm (KDA/CSA/the local window) AND
the Engram dense read, and `attn_write` scales the attention-side write
(`attn + branch`), because the memory read is a read of the same normalised
state and its output joins the same residual update; the paper has no memory
sub-layer and leaving it unscaled would make `attn_write` scale only part of
the attention-side write. `mlp_read`/`mlp_write` are the FFN/MoE branch. The
prelude/coda unique layers keep their own RMSNorm gains (they are outside the
looped block; paper form wires the block).

DEPTH INDEX. `layerrope_index="iteration"` (default) uses the ACTUAL loop
iteration `i`, v1's choice. `"position"` uses the block's position in the
network and is INDEPENDENT of the loop iteration — the choice Parcae's looped
model makes ("each layer's depth index is its position in the network,
independent of the loop iteration"), and the reason the paper's depth scaling
can transfer to a weight-shared loop: a single loop body is ONE depth, so all
its iterations read the same gain. With `loop_blocks=N` body `b` owns the
position `P+b`, still independent of the iteration.

v1's own depth docstring follows.

--- v1 ---

In a deep residual stack the per-layer normalisation gain `gamma` empirically
encodes **depth**: its magnitude grows with depth and its
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

__all__ = [
    "PAPER_SITES",
    "LayerRoPEGains",
    "LayerRoPENorm",
    "LayerRoPEPaperGains",
    "LayerRoPESite",
    "rotate_pairs",
]

#: The paper's four sites per block, in the order `for_iteration` returns them.
PAPER_SITES = ("attn_read", "attn_write", "mlp_read", "mlp_write")


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


class LayerRoPESite(nn.Module):
    """One LayerRoPE site, the paper's reference form (module docstring).

    `gamma [d]` + four scalars + the two buffers the snippet registers
    (`freq`, `log_l`). `table()` is the snippet's `forward()` verbatim: it
    returns the `[n_layers, d]` gain table, and `[index]` is one layer's gain.
    The table is INPUT-INDEPENDENT — it is a function of the parameters only,
    which is why the whole table is built once per forward (`begin_forward`)
    and sliced per iteration.

    `identity=True` sets `b_mag = b_rot = 0` AND short-circuits `table()` to
    `gamma`; the short-circuit is not cosmetic: `θ = exp(0) = 1` still rotates
    by `freq` in the snippet (`ang = θ·freq`), so a plain zero-beta init is NOT
    the identity (module docstring, "A correction to the obvious reading").
    """

    def __init__(
        self,
        d: int,
        n_layers: int,
        *,
        base: float = 100.0,
        beta_init: float = -0.5,
        identity: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if d % 2:
            loud(
                f"layerrope paper form with d_model={d}: the spectrum rotates "
                f"adjacent channel PAIRS, so the width must be even. Escape: an "
                f"even d_model, or layerrope=false."
            )
        assert n_layers >= 1, f"layerrope n_layers={n_layers}: at least one depth"
        assert base > 0.0, f"layerrope base={base}: the spectrum needs base > 0"
        self.d = int(d)
        self.n_layers = int(n_layers)
        self.base = float(base)
        self.identity = bool(identity)
        #: The shared per-site vector: the gain is `gamma` rotated/scaled by the
        #: depth index. `[d]`, one per site — NOT one per layer.
        self.gamma = nn.Parameter(torch.ones(d, device=device))
        self.a_mag = nn.Parameter(torch.zeros((), device=device))
        beta = 0.0 if identity else float(beta_init)
        self.b_mag = nn.Parameter(torch.tensor(beta, device=device))
        self.a_rot = nn.Parameter(torch.zeros((), device=device))
        self.b_rot = nn.Parameter(torch.tensor(beta, device=device))
        # buffer, not parameter: the spectrum is a constant of the width/base.
        j = torch.arange(d // 2, device=device, dtype=torch.float32)
        self.register_buffer("freq", base ** (-2.0 * j / d))
        self.register_buffer("log_l", torch.log(torch.arange(1.0, n_layers + 1, device=device)))

    def table(self) -> torch.Tensor:
        """The paper's `forward()`: `[n_layers, d]`, the gain at every depth."""
        if self.identity:
            return self.gamma.unsqueeze(0).expand(self.n_layers, self.d)
        r = self.a_mag + self.b_mag * self.log_l
        theta = torch.exp(self.a_rot + self.b_rot * self.log_l)
        ang = theta[:, None] * self.freq
        c = torch.polar(torch.exp(r)[:, None].expand_as(ang), ang)
        g = torch.view_as_complex(self.gamma.view(-1, 2))
        return torch.view_as_real(g * c).flatten(-2)


class LayerRoPEPaperGains(nn.Module):
    """The block's FOUR sites of [`LayerRoPESite`], one owner for the whole model.

    Keyed `layerrope.sites.<block*4 + site>.*`; `n_blocks = loop_blocks` (each
    distinct loop body is its own set, because it is its own layer). The gain
    table is built ONCE per forward (`begin_forward`, input-independent) and
    sliced per iteration (`for_iteration`), so the per-step cost is the table
    build, not one build per read.

    `_applications` is a DEVICE counter (four per `for_iteration`, i.e. per
    iteration per block), drained by the trainer on the log cadence — the proof
    the arm ran costs no host read.
    """

    def __init__(
        self,
        d: int,
        n_layers: int,
        *,
        n_blocks: int = 1,
        base: float = 100.0,
        beta_init: float = -0.5,
        identity: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        assert n_blocks >= 1, f"layerrope n_blocks={n_blocks}"
        self.d = int(d)
        self.n_layers = int(n_layers)
        self.n_blocks = int(n_blocks)
        self.identity = bool(identity)
        self.sites = nn.ModuleList(
            [
                LayerRoPESite(
                    d,
                    n_layers,
                    base=base,
                    beta_init=beta_init,
                    identity=identity,
                    device=device,
                )
                for _ in range(self.n_blocks * len(PAPER_SITES))
            ]
        )
        #: The built table, `[n_blocks·4, n_layers, d]`; a plain attribute (not
        #: a buffer) so it never reaches the state dict.
        self._table: torch.Tensor | None = None
        self.register_buffer(
            "_applications",
            torch.zeros((), dtype=torch.int64, device=device),
            persistent=False,
        )

    def begin_forward(self) -> None:
        """Build every site's table once — a function of the parameters only."""
        self._table = torch.stack([site.table() for site in self.sites])

    def table(self, block: int = 0, site: str = "attn_read") -> torch.Tensor:
        """One site's `[n_layers, d]` table (the reference's `forward()`)."""
        return self.sites[block * len(PAPER_SITES) + PAPER_SITES.index(site)].table()

    def for_iteration(
        self, block: int, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(attn_read, attn_write, mlp_read, mlp_write)` at one depth index."""
        assert self._table is not None, "begin_forward() must run once per forward"
        assert 0 <= index < self.n_layers, f"layerrope depth {index} outside 0..{self.n_layers}"
        assert 0 <= block < self.n_blocks, f"layerrope block {block} outside 0..{self.n_blocks}"
        base = block * len(PAPER_SITES)
        out = tuple(self._table[base + s][index] for s in range(len(PAPER_SITES)))
        self._applications.add_(len(PAPER_SITES))
        return out  # type: ignore[return-value]

    def take_applications(self) -> torch.Tensor:
        """The applications since the last take, as a DEVICE scalar (no host read)."""
        n = self._applications.clone()
        self._applications.zero_()
        return n
