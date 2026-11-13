"""Static census of the MoE DISPATCH — which op, how many bytes, how many launches.

One routed forward + its backward at the resolved config and `b x s`. For every
op the routing path runs (`experts.SparseMoE.forward` through `combine`), this
prints the tensor shape, the dtype it reads and writes, the bytes moved (read +
written: the traffic a pass actually costs) and the number of DEVICE kernels.
The rows are hand-mapped to the code (each names its call site), because a
launch count is a property of the *code path*, not of the config: the config
supplies the shapes and chooses which arm runs.

Why this exists: `scripts/kernel_profile.py` puts 31.6 ms of the m400 step in
`moe_dispatch` (`docs/records/reviews/2026-11-05-kernel-profile.md`), but a REGION is
not a target. This file is the target list: it says which of those kernels are
routing arithmetic (cheap, fusible), which are expert math the region's seam
attribution swallowed (`_grouped_gemm_pair_kernel`, `_wgrad_kernel`,
`_grouped_gemm_kernel`), which are the banks' per-step rebuild, and which are
the output projection (`down`) that happens to sit inside `SparseMoE.forward`.
`--ab` prices the two new arms against the shipped ones at the same geometry.

The launch column is the point: at this shape the profile measures 5 095 kernels
per step and ~29 us of wall per kernel launch, so a fusion's win is mostly the
launches it removes, and the bank rebuild (~1 000 of them) is the largest single
count in the region.

    uv run python scripts/dispatch_census.py                     # the m400b snapshot
    uv run python scripts/dispatch_census.py --config configs/m400.toml
    uv run python scripts/dispatch_census.py --ab                # shipped vs fused
    uv run python scripts/dispatch_census.py --set moe_gather=kernel
    uv run python scripts/dispatch_census.py --json /tmp/c.json

WHAT IS NOT COUNTED, stated so the total is not read as a whole step: the
attention/Engram/mHC regions (another census), the optimizer, and the router's
`mm` interior beyond its traffic. `expert` and `down` rows are CONTEXT — they
are in the profile's region but not in the routing path, and the group table
says so (`route+permute+gather+combine` is the fusible part).

The arithmetic is the code's, not a guess: each row names the call it prices.
Where a torch op's kernel count is not 1 (`topk`, `argsort`) the count is the
one the 2026-11-05 profile measured, and the row says so.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dormouse.config import ModelConfig, validate

#: `<repo>/port/scripts/dispatch_census.py` -> `<repo>`; run snapshots live under
#: `checkpoints/`, outside the `port/` package but inside the repo.
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = str(ROOT / "checkpoints/m400b/m400b.config.toml")
DEFAULT_BATCH, DEFAULT_SEQ = 8, 1024
#: RTX 5070 Ti HBM (GDDR7): 896 GB/s. The census quotes us at this ceiling and
#: says so; a kernel at 30% of it is a different finding, not a different table.
DEFAULT_BW = 0.896e12

ITEMSIZE = {"f32": 4, "bf16": 2, "f16": 2, "int64": 8, "int32": 4, "bool": 1}

#: The clusters. `ROUTE` is what a fused route/permute can remove; `EXPERT`,
#: `BANK` and `DOWN` are priced because the profile's `moe_dispatch` region
#: contains them, and a plan that "optimizes the dispatch" must not charge their
#: milliseconds to routing.
ROUTE_GROUPS = ("router", "route", "permute", "gather", "combine")
CONTEXT_GROUPS = ("bank", "down", "expert", "lfb")

#: The bank cache's key walk: `tuple(p._version for e in self.experts for p in
#: e.parameters())` runs on EVERY `_banks` call. MEASURED on this box
#: (2026-11-05, CPU, m400 geometry: E=512, 1 536 parameters, Python 3.12) at
#: 2.09 ms/call = 4.08 us per expert. `moe_bank_epoch` replaces it with one int
#: compare; the number is an input so a reader can re-measure, not a constant
#: baked into a total.
KEY_WALK_US_PER_EXPERT = 4.08
EPOCH_KEY_US = 0.5


def nbytes(shape: tuple[int, ...], dtype: str) -> int:
    if isinstance(shape, int):
        shape = (shape,)
    n = 1
    for s in shape:
        n *= int(s)
    return n * ITEMSIZE[dtype]


def _pairs(x) -> tuple[tuple[tuple[int, ...], str], ...]:
    """`(shape, dt)` or `((shape, dt), ...)` or a flat `shape, dt, shape, dt`."""
    x = tuple(x)
    if not x:
        return ()
    if all(isinstance(e, tuple) and len(e) == 2 and isinstance(e[1], str) for e in x):
        return x
    assert len(x) % 2 == 0, f"traffic must be (shape, dtype) pairs, got {x!r}"
    out = []
    for i in range(0, len(x), 2):
        shape, dt = x[i], x[i + 1]
        assert isinstance(dt, str) and isinstance(shape, (tuple, int)), f"bad pair {(shape, dt)!r}"
        if not isinstance(shape, tuple):
            shape = (shape,)
        if len(shape) == 1 and isinstance(shape[0], tuple):  # ((R,),) -> (R,)
            shape = shape[0]
        out.append((shape, dt))
    return tuple(out)


@dataclass(frozen=True)
class Row:
    """One op: the reason the census can be audited line by line."""

    group: str
    op: str
    site: str
    shape: tuple[int, ...]
    reads: tuple[tuple[tuple[int, ...], str], ...]
    writes: tuple[tuple[tuple[int, ...], str], ...]
    launches: int = 1
    #: `once per training step` (the bank cache) vs `once per routed forward`.
    per_step: bool = False
    #: HOST time this op spends, where the op is Python (the bank key walk).
    host_us: float = 0.0
    note: str = ""

    def __post_init__(self):
        # Accept both `((shape, dt),)` and `(shape, dt)` — a 60-row table written
        # by hand gets that wrong once and the error is a shape unpack six rows
        # later. Normalize to a tuple of (shape, dtype) pairs.
        object.__setattr__(self, "reads", _pairs(self.reads))
        object.__setattr__(self, "writes", _pairs(self.writes))

    @property
    def nbytes(self) -> int:
        return sum(nbytes(s, d) for s, d in self.reads) + sum(nbytes(s, d) for s, d in self.writes)

    @property
    def dtypes(self) -> str:
        seen = {d for _, d in self.reads + self.writes}
        return "/".join(sorted(seen))


@dataclass(frozen=True)
class Ctx:
    n: int  # tokens this forward routes
    E: int
    d: int
    f: int
    r: int
    k: int
    max_iter: int
    bf16: bool
    capacity: float
    gather: str
    permute: str
    route_fused: bool
    combine: str
    combine_dtype: str
    bank_epoch: bool
    share_u: str
    lb_mode: str

    @property
    def R(self) -> int:
        return self.n * self.k

    @property
    def dt(self) -> str:
        return "bf16" if self.bf16 else "f32"

    @property
    def dtb(self) -> int:
        return ITEMSIZE[self.dt]

    @property
    def cap(self) -> int:
        """The capacity rule (`experts._capacity`), 0 = dropless."""
        import math

        if self.capacity == 0:
            return 0
        return max(self.k, math.ceil(self.capacity * self.n * self.k / self.E))


def ctx_for(cfg: ModelConfig, b: int, s: int) -> Ctx:
    return Ctx(
        n=b * s,
        E=cfg.n_experts,
        d=cfg.d_model,
        f=cfg.d_ffn,
        r=cfg.rank,
        k=cfg.moe_topk,
        max_iter=cfg.max_iter,
        bf16=cfg.bf16,
        capacity=cfg.moe_capacity,
        gather=cfg.moe_gather,
        permute=cfg.moe_permute,
        route_fused=cfg.moe_route_fused,
        combine=cfg.moe_combine,
        combine_dtype=cfg.moe_combine_dtype,
        bank_epoch=cfg.moe_bank_epoch,
        share_u=cfg.share_u_mode if cfg.share_u else "off",
        lb_mode=cfg.moe_lb_mode,
    )


def _router_rows(c: Ctx) -> list[Row]:
    """The router itself: one GEMM and its per-forward weight cast (`mm_mixed`)."""
    n, E, d = c.n, c.E, c.d
    return [
        Row(
            "router",
            "router weight cast",
            "tsct.mm_mixed: weight.to(dt) (dense_qat='none')",
            (E, d),
            ((E, d), "f32"),
            ((E, d), c.dt),
            1,
        ),
        Row(
            "router",
            "logits = x @ router.weight.T",
            "tsct.mm_mixed: x @ w.t()",
            (n, E),
            ((n, d), c.dt) + ((d, E), c.dt),
            ((n, E), c.dt),
            1,
            note="bf16 tensor-core GEMM; not fusible with the route (it is its input)",
        ),
    ]


def _route_rows(c: Ctx) -> list[Row]:
    """`softmax` + `logits + lfb_bias` + `topk` + the gate gather (+ renorm)."""
    nE, nk, E1 = (c.n, c.E), (c.n, c.k), (c.E,)
    rows: list[Row] = []
    if c.route_fused:
        rows.append(
            Row(
                "route",
                "fused softmax+bias+topk+gather",
                "route_fused.softmax_topk (one Triton launch)",
                nE,
                ((nE, c.dt), (E1, "f32")),
                ((nk, "int64"), (nk, c.dt)),
                1,
            )
        )
    else:
        rows += [
            Row(
                "route",
                "softmax(logits)",
                "experts.forward: F.softmax(logits, -1)",
                nE,
                ((nE, c.dt),),
                ((nE, c.dt),),
                1,
            ),
            Row(
                "route",
                "raised = logits + lfb_bias",
                "experts.forward: raised = logits + self.lfb_bias",
                nE,
                ((nE, c.dt), (E1, "f32")),
                ((nE, "f32"),),
                1,
                note="torch promotes bf16+fp32 to fp32",
            ),
            Row(
                "route",
                "raised.topk(k)",
                "experts.forward: raised.topk(self.topk, -1)",
                nE,
                ((nE, "f32"), (nE, "f32")),
                ((nk, "int64"), (nk, "f32")),
                6,
                note="measured: mbtopk 5 kernels + the masked fill",
            ),
            Row(
                "route",
                "gather(probs, index)",
                "experts.forward: torch.gather(probs, 1, index)",
                nk,
                ((nE, c.dt), (nk, "int64")),
                ((nk, c.dt),),
                1,
            ),
        ]
    rows.append(
        Row(
            "route",
            "renorm weight",
            "experts.forward: weight / weight.sum(-1).clamp_min(1e-9)",
            nk,
            ((nk, c.dt), (nk, c.dt)),
            ((nk, c.dt),),
            3,
            note="sum + clamp + div",
        )
    )
    if c.route_fused:
        rows.append(
            Row(
                "route",
                "fused route backward",
                "route_fused._RouteFused.backward (one Triton launch)",
                nE,
                ((nE, c.dt), (nk, "int64"), (nk, c.dt)),
                ((nE, c.dt),),
                1,
            )
        )
    else:
        rows += [
            Row(
                "route",
                "gather backward (zero scatter)",
                "autograd: gather -> zeros(n,E).scatter_",
                nE,
                ((nk, "int64"), (nk, c.dt)),
                ((nE, c.dt),),
                1,
            ),
            Row(
                "route",
                "softmax backward",
                "autograd: _softmax_backward_data",
                nE,
                ((nE, c.dt), (nE, c.dt)),
                ((nE, c.dt),),
                1,
            ),
            Row(
                "route",
                "bias-add backward",
                "autograd: add",
                nE,
                ((nE, "f32"),),
                ((nE, c.dt),),
                1,
            ),
        ]
    return rows


def _permute_rows(c: Ctx) -> list[Row]:
    """`_permute`: the order + the rows every arm is permuted by."""
    R = (c.R,)
    rows: list[Row] = []
    if c.permute == "count":
        rows.append(
            Row(
                "permute",
                "seed order (count, one program/expert)",
                "route_fused.count_permute (bincount+cumsum+1 Triton launch)",
                R,
                ((R, "int32"), (R, "int32"), (R, "int32")),
                ((R, "int64"),),
                3,
                note="bincount + cumsum + the kernel",
            )
        )
    else:
        rows.append(
            Row(
                "permute",
                "order = argsort(safe, stable)",
                "experts._permute: torch.argsort(safe, stable=True)",
                R,
                ((R, "int64"),) * 8,
                ((R, "int64"),) * 8,
                12,
                note="measured: CUB radix, 8-bit digits -> 8 passes, 12 launches",
            )
        )
    rows += [
        Row(
            "permute",
            "sorted_ids = safe[order]",
            "experts._permute",
            R,
            ((R, "int64"), (R, "int64")),
            ((R, "int64"),),
            1,
        ),
        Row(
            "permute",
            "counts = bincount(sorted_ids)",
            "experts._permute: _bincount",
            (c.E,),
            ((R, "int64"),),
            ((c.E,), "int64"),
            1,
        ),
        Row(
            "permute",
            "offsets = cumsum(counts) - counts",
            "experts._permute",
            (c.E,),
            ((c.E,), "int64"),
            ((c.E,), "int64"),
            1,
        ),
        Row(
            "permute",
            "pos = arange - offsets[sorted_ids]",
            "experts._permute",
            R,
            ((R, "int64"), ((c.E,), "int64")),
            ((R, "int64"),),
            2,
        ),
        Row(
            "permute",
            "keep_cap = pos < cap",
            "experts._permute",
            R,
            ((R, "int64"),),
            ((R,), "bool"),
            1,
            note="dropless: ones_like",
        ),
        Row(
            "permute",
            "dropped = (~keep).sum()",
            "experts._permute",
            (1,),
            ((R,), "bool"),
            ((1,), "int64"),
            2,
        ),
    ]
    return rows


def _gather_rows(c: Ctx) -> list[Row]:
    """`_triton_dispatch`'s row map and the A operand (`xg` or `x[toks]`)."""
    R, n, k = c.R, c.n, c.k
    rows: list[Row] = [
        Row(
            "gather",
            "owner = rows // k",
            "_triton_dispatch",
            (R,),
            ((R, "int64"),),
            ((R, "int64"),),
            1,
        ),
        Row(
            "gather",
            "kept = index.reshape(-1)[rows]",
            "_triton_dispatch",
            (R,),
            ((R, "int64"), (R, "int64")),
            ((R, "int64"),),
            1,
        ),
        Row(
            "gather",
            "counts = bincount(kept)",
            "_triton_dispatch: _bincount",
            (c.E,),
            ((R, "int64"),),
            ((c.E,), "int64"),
            1,
        ),
        Row(
            "gather",
            "offs = zeros(E+1); offs[1:] = cumsum(counts)",
            "_triton_dispatch",
            (c.E + 1,),
            ((c.E,), "int64"),
            ((c.E + 1,), "int32"),
            3,
        ),
        Row(
            "gather",
            "rowmap = full(nk,-1); rowmap[rows] = arange(R)",
            "_triton_dispatch",
            (n, k),
            ((R, "int64"), (R, "int64")),
            ((n * k,), "int32"),
            2,
        ),
        Row(
            "gather",
            "seg = kept.to(int32)",
            "_triton_dispatch",
            (R,),
            ((R, "int64"),),
            ((R,), "int32"),
            1,
        ),
        Row(
            "gather",
            "w = weight.reshape(-1)[rows].float()",
            "_triton_dispatch: combine arg",
            (R,),
            ((R, "int64"), (R, c.dt)),
            ((R,), "f32"),
            2,
        ),
    ]
    if c.gather == "kernel":
        # `gather_abi` + the kernel's own `x[toks]` read: no materialized copy.
        rows += [
            Row(
                "gather",
                "toks = (rows//k).to(int32)",
                "moe_sonic.gather_abi",
                (R,),
                ((R, "int64"),),
                ((R,), "int32"),
                1,
            ),
            Row(
                "gather",
                "x[toks] read inside the expert kernel",
                "moe_sonic._gather_glu_kernel",
                (R, c.d),
                (((R, c.d), c.dt),),
                (),
                0,
            ),
            # `_GatherGlu` has no adjoint: its backward RE-RUNS the shipped
            # expert forward on the same rows. That is a whole forward's traffic
            # added to the PASS (the prototype's own finding).
            Row(
                "gather",
                "gather arm: recompute one expert forward",
                "moe_sonic._GatherGlu.backward -> _shipped_forward",
                (R, c.f),
                (((c.E, c.d, c.r), c.dt),) * 2 + (((c.E, c.r, c.f), c.dt),) + (((R, c.d), c.dt),),
                (((R, c.f), c.dt),),
                1,
                note="the arm's backward has no adjoint: one extra forward per pass",
            ),
        ]
    else:
        rows.append(
            Row(
                "gather",
                "xg = x[owner] (index_select)",
                "_triton_dispatch: xg = self._gather_rows(x, owner, dt)",
                (R, c.d),
                ((R, c.d), c.dt) * 2,
                ((R, c.d), c.dt),
                1,
                note="the copy AND the kernel's read of it are this row's traffic",
            )
        )
    if c.cap:
        # `moe_capacity > 0`: `rows = order[keep]` — a BOOLEAN-mask index, i.e.
        # `nonzero` underneath (the op ADR-0018's capture note bans).
        rows.insert(
            0,
            Row(
                "gather",
                "rows = order[keep]",
                "_dispatch: keep_rows = keep",
                (R,),
                ((R, "int64"), (R, "bool")),
                ((R, "int64"),),
                1,
                note="bool-mask index: a host-side length, refused by full_graph",
            ),
        )
    return rows


def _combine_rows(c: Ctx) -> list[Row]:
    """`moe_kernels.combine`: the scatter that closes the pass, and its adjoint.

    `torch` is the three-pass fp32 form (`zeros`, `y.to(f32)·w`, `index_add_`),
    1 476 MB/pass. `fused` is ONE kernel reading each token's k rows in fp32
    registers; `moe_combine_dtype="bf16"` stores the compute dtype (so the
    caller's `weighted.to(dt)` also disappears — `_down_rows` drops that row)
    and the backward reads a bf16 `grad_out`.
    """
    R, n, f = c.R, c.n, c.f
    out_dt = c.dt if c.combine_dtype == "bf16" else "f32"
    grad_dt = c.dt if c.combine_dtype == "bf16" else "f32"
    if c.combine == "fused":
        return [
            Row(
                "combine",
                "fused combine forward",
                "moe_kernels._combine_fwd_kernel (one launch)",
                (n, f),
                ((R, f), c.dt) + ((R,), "f32") + ((n, c.k), "int32"),
                ((n, f), out_dt),
                1,
                note="fp32 registers, one rounding on the store"
                + (" (bf16)" if out_dt == c.dt else ""),
            ),
            Row(
                "combine",
                "combine backward",
                "moe_kernels._combine_bwd_kernel (one launch)",
                (n, f),
                ((n, f), grad_dt) + ((R, f), c.dt) + ((R,), "f32"),
                ((R, f), c.dt) + ((R,), "f32"),
                1,
            ),
        ]
    return [
        Row(
            "combine",
            "out = zeros(n, f) fp32",
            "experts._triton_dispatch: combine",
            (n, f),
            (),
            ((n, f), "f32"),
            1,
        ),
        Row(
            "combine",
            "y.to(f32) * w",
            "moe_kernels._Combine.forward",
            (R, f),
            ((R, f), c.dt),
            ((R, f), "f32"),
            1,
            note="the fp32 widening at the MoE output boundary",
        ),
        Row(
            "combine",
            "out.index_add_(owner, ·)",
            "moe_kernels._Combine.forward",
            (n, f),
            ((R, f), "f32") + ((n, f), "f32"),
            ((n, f), "f32"),
            1,
        ),
        Row(
            "combine",
            "combine backward",
            "moe_kernels._Combine.backward (one launch)",
            (n, f),
            ((n, f), "f32") + ((R, f), c.dt) + ((R,), "f32"),
            ((R, f), c.dt) + ((R,), "f32"),
            1,
        ),
    ]


def _lfb_rows(c: Ctx) -> list[Row]:
    """The bias controller: one histogram + five `[E]` passes (per forwarded pass)."""
    if c.lb_mode == "aux":
        return []
    return [
        Row(
            "lfb",
            "counts = bincount(real)",
            "experts.forward: _bincount",
            (c.E,),
            ((c.R,), "int64"),
            ((c.E,), "int64"),
            1,
        ),
        Row(
            "lfb",
            "b -= coef·sign(L-mean) / prop",
            "experts._lfb_step",
            (c.E,),
            ((c.E, "int64"), (c.E, "f32"), (c.E, "f32"), (c.E, "f32")),
            ((c.E,), "f32"),
            5,
        ),
    ]


def _bank_rows(c: Ctx) -> list[Row]:
    """`_banks`: the per-STEP rebuild the trainer's `invalidate_banks` forces.

    One row per stack, `per_step=True`. The plane basis is one contiguous
    `expand`, the per-expert `v` is 512 `.to()` casts + a 512-tensor
    `torch.stack` — that is the launch count, and it is why the region has
    thousands of kernels.
    """
    E, d, f, r = c.E, c.d, c.f, c.r
    rows: list[Row] = [
        Row(
            "bank",
            "key: _version walk" if not c.bank_epoch else "key: bank_epoch int",
            "experts._banks: cache key",
            (E,),
            (),
            (),
            0,
            host_us=(E * KEY_WALK_US_PER_EXPERT if not c.bank_epoch else EPOCH_KEY_US),
            note="HOST per `_banks` call, i.e. per forward (measured 2026-11-05)",
        )
    ]
    if c.share_u == "plane":
        for name in ("gate_u", "up_u"):
            rows.append(
                Row(
                    "bank",
                    f"{name} plane -> [E,d,r]",
                    "experts._banks: u.expand(E).contiguous()",
                    (E, d, r),
                    ((d, r), c.dt),
                    ((E, d, r), c.dt),
                    1,
                    per_step=True,
                )
            )
    else:
        for name in ("gate_u", "up_u"):
            rows.append(
                Row(
                    "bank",
                    f"{name} stack [E,d,r]",
                    "experts._banks: torch.stack([e.gate_u.to(dt)...])",
                    (E, d, r),
                    ((E, d, r), "f32"),
                    ((E, d, r), c.dt),
                    E + 1,
                    per_step=True,
                )
            )
    rows.append(
        Row(
            "bank",
            "v stack [E,r,f]",
            "experts._banks: torch.stack([e.v.to(dt)...])",
            (E, r, f),
            ((E, r, f), "f32"),
            ((E, r, f), c.dt),
            E + 1,
            per_step=True,
        )
    )
    for name in ("gate_s", "up_s"):
        rows.append(
            Row(
                "bank",
                f"{name} stack [E,r]",
                f"experts._banks: {name}",
                (E, r),
                ((E, r), "f32"),
                ((E, r), c.dt),
                2,
                per_step=True,
            )
        )
    return rows


def _down_rows(c: Ctx) -> list[Row]:
    """`SparseMoE.forward`'s OUTPUT projection — in the region, not in routing."""
    n, d, f = c.n, c.d, c.f
    rows = []
    if not (c.combine == "fused" and c.combine_dtype == "bf16"):
        # With a bf16 combine store `weighted` is ALREADY dt: `.to(dt)` is a
        # no-op and its whole pass (134 MB read + 67 MB write) disappears.
        rows.append(
            Row(
                "down",
                "weighted.to(dt)",
                "experts.forward: mm_mixed(weighted.to(dt), ...)",
                (n, f),
                ((n, f), "f32"),
                ((n, f), c.dt),
                1,
            )
        )
    rows += [
        Row(
            "down",
            "down weight cast",
            "tsct.mm_mixed: weight.to(dt)",
            (d, f),
            ((d, f), "f32"),
            ((d, f), c.dt),
            1,
        ),
        Row(
            "down",
            "down gemm",
            "tsct.mm_mixed: x @ w.t()",
            (n, d),
            ((n, f), c.dt) + ((d, f), c.dt),
            ((n, d), c.dt),
            1,
        ),
        Row(
            "down",
            "out.to(f32)",
            "experts.forward: ...to(torch.float32)",
            (n, d),
            ((n, d), c.dt),
            ((n, d), "f32"),
            1,
        ),
    ]
    return rows


def _expert_rows(c: Ctx) -> list[Row]:
    """The expert pass + adjoint KERNELS the region's attribution swallowed.

    Priced from `moe_kernels`' own layout (banks read once per tile, `y` written,
    the three `_wgrad`s and the fused `dx` written) and from the 2026-11-05
    profile's kernel table — they are here so a plan cannot call them routing.
    """
    R, E, d, f, r = c.R, c.E, c.d, c.f, c.r
    bank = nbytes((E, d, r), c.dt) * 2 + nbytes((E, r, f), c.dt)
    return [
        Row(
            "expert",
            "expert fwd (bank read + y)",
            "moe_kernels._ExpertGemmGlu.forward",
            (R, f),
            (((E, d, r), c.dt), ((E, d, r), c.dt), ((E, r, f), c.dt)),
            ((R, f), c.dt),
            1,
            note=f"bank read {bank / 1e6:.0f} MB",
        ),
        Row(
            "expert",
            "expert bwd: dx (pair) + 3 wgrad",
            "moe_kernels._glu_backward_fused",
            (R, f),
            (((E, d, r), c.dt),) * 2,
            (((E, d, r), c.dt),) * 2 + (((E, r, f), c.dt),),
            4,
            note="1 grouped pair + 3 _wgrad",
        ),
        Row(
            "expert",
            "expert bwd: ~20 pointwise [R,r]",
            "moe_kernels._glu_backward_fused",
            (R, r),
            ((R, r), "f32") * 20,
            ((R, r), "f32") * 20,
            20,
        ),
    ]


def census(cfg: ModelConfig, batch: int, seq: int) -> tuple[Ctx, list[Row]]:
    c = ctx_for(cfg, batch, seq)
    rows: list[Row] = []
    rows += _router_rows(c)
    rows += _route_rows(c)
    rows += _permute_rows(c)
    rows += _gather_rows(c)
    rows += _combine_rows(c)
    rows += _lfb_rows(c)
    rows += _bank_rows(c)
    rows += _down_rows(c)
    rows += _expert_rows(c)
    return c, rows


def totals(rows: list[Row], *, per_step: bool | None = None) -> tuple[int, int]:
    """`(bytes, launches)` over `rows`, optionally only one cadence."""
    if per_step is not None:
        rows = [r for r in rows if r.per_step == per_step]
    return sum(r.nbytes for r in rows), sum(r.launches for r in rows)


def _shape(row: Row) -> str:
    return "x".join(str(x) for x in row.shape)


def report(cfg: ModelConfig, rows: list[Row], c: Ctx, bw: float, *, top: int = 0) -> dict:
    total = sum(r.nbytes for r in rows)
    launches = sum(r.launches for r in rows)
    print(
        f"\n== dispatch census — n={c.n} tokens, E={c.E}, d={c.d}, f={c.f}, r={c.r}, "
        f"k={c.k}, cap={c.cap or 'dropless'}, {c.dt}"
    )
    print(
        f"   arms: moe_gather={c.gather}  moe_permute={c.permute}  "
        f"moe_route_fused={c.route_fused}  share_u={c.share_u}  lb={c.lb_mode}  "
        f"max_iter={c.max_iter}"
    )
    print(
        f"\n   {'group':<9}{'op':<46}{'shape':<16}{'dt':<9}{'MB':>9}{'launch':>7}"
        f"{'host us':>9}  site"
    )
    for r in rows:
        print(
            f"   {r.group:<9}{r.op[:45]:<46}{_shape(r):<16}{r.dtypes:<9}"
            f"{r.nbytes / 1e6:9.2f}{r.launches:7d}{r.host_us:9.1f}  {r.site}"
        )
    print("\n   -- groups --")
    order = ROUTE_GROUPS + CONTEXT_GROUPS
    by_group = []
    for g in order:
        grp = [r for r in rows if r.group == g]
        if not grp:
            continue
        nb, nl = totals(grp)
        by_group.append((g, nb, nl))
        print(
            f"   {g:<9}{nb / 1e6:9.2f} MB  {nb / total * 100:5.1f} %  "
            f"{nb / bw * 1e6:8.1f} us  {nl:4d} launches"
        )
    host_us = sum(r.host_us for r in rows)
    route_b, route_l = totals([r for r in rows if r.group in ROUTE_GROUPS])
    print(
        f"   {'TOTAL':<9}{total / 1e6:9.2f} MB  100.0 %  "
        f"{total / bw * 1e6:8.1f} us  {launches:4d} launches"
    )
    print(
        f"   route+permute+gather+combine (the fusible part): "
        f"{route_b / 1e6:.2f} MB, {route_l} launches"
    )
    step_b, step_l = totals(rows, per_step=True)
    print(f"   of which the bank rebuild is per STEP: {step_b / 1e6:.2f} MB, {step_l} launches")
    if top:
        for r in sorted(rows, key=lambda r: -r.nbytes)[:top]:
            print(f"   top: {r.nbytes / 1e6:8.1f} MB  {r.group:<9} {r.op}")
    return {
        "config": {
            "n": c.n,
            "E": c.E,
            "d": c.d,
            "f": c.f,
            "r": c.r,
            "k": c.k,
            "cap": c.cap,
            "bf16": c.bf16,
            "max_iter": c.max_iter,
            "moe_gather": c.gather,
            "moe_permute": c.permute,
            "moe_route_fused": c.route_fused,
            "share_u": c.share_u,
        },
        "total_bytes": total,
        "total_launches": launches,
        "host_us": host_us,
        "route_bytes": route_b,
        "route_launches": route_l,
        "step_bytes": step_b,
        "step_launches": step_l,
        "groups": [
            {"group": g, "bytes": nb, "launches": nl, "us": nb / bw * 1e6} for g, nb, nl in by_group
        ],
        "rows": [
            {
                "group": r.group,
                "op": r.op,
                "site": r.site,
                "shape": list(r.shape),
                "dtypes": r.dtypes,
                "bytes": r.nbytes,
                "launches": r.launches,
                "per_step": r.per_step,
            }
            for r in rows
        ],
    }


def ab(cfg: ModelConfig, batch: int, seq: int) -> dict[str, dict]:
    """The shipped arms vs each new one, SAME geometry — the A/B the flags need."""
    arms = (
        (
            "shipped",
            dict(
                moe_route_fused=False,
                moe_permute="argsort",
                moe_combine="torch",
                moe_combine_dtype="fp32",
                moe_bank_epoch=False,
            ),
        ),
        ("fused-route", dict(moe_route_fused=True, moe_permute="count")),
        ("fused-combine", dict(moe_combine="fused")),
        ("combine-bf16", dict(moe_combine="fused", moe_combine_dtype="bf16")),
        ("bank-epoch", dict(moe_bank_epoch=True)),
        # The combination the report PICKS: the fused route + counting permute +
        # fp32 fused combine (bit-identical) + the epoch key. bf16 is priced
        # separately because it does not meet the 1e-3 logits target.
        (
            "picked",
            dict(
                moe_route_fused=True, moe_permute="count", moe_combine="fused", moe_bank_epoch=True
            ),
        ),
        (
            "all",
            dict(
                moe_route_fused=True,
                moe_permute="count",
                moe_combine="fused",
                moe_combine_dtype="bf16",
                moe_bank_epoch=True,
            ),
        ),
    )
    out: dict[str, dict] = {}
    for name, over in arms:
        c, rows = census(dataclasses.replace(cfg, **over), batch, seq)
        rb, rl = totals([r for r in rows if r.group in ROUTE_GROUPS])
        out[name] = {
            "route_bytes": rb,
            "route_launches": rl,
            "host_us": sum(r.host_us for r in rows),
            "ctx": c,
        }
    base = out["shipped"]
    print(
        f"\n== A/B at the same geometry — n={base['ctx'].n}, E={base['ctx'].E}, k={base['ctx'].k}"
    )
    print(
        f"   {'arm':<15}{'route MB':>10}{'Δ MB':>10}{'launches':>10}{'Δ launch':>10}"
        f"{'host us':>10}{'Δ host':>10}"
    )
    for name, v in out.items():
        print(
            f"   {name:<15}{v['route_bytes'] / 1e6:10.2f}"
            f"{(v['route_bytes'] - base['route_bytes']) / 1e6:+10.2f}"
            f"{v['route_launches']:10d}{v['route_launches'] - base['route_launches']:+10d}"
            f"{v['host_us']:10.1f}{v['host_us'] - base['host_us']:+10.1f}"
        )
    return {k: {kk: vv for kk, vv in v.items() if kk != "ctx"} for k, v in out.items()}


def _load(path: str) -> ModelConfig:
    """A flat preset OR a run snapshot (`[model]`) -> ModelConfig (no torch)."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(
            f"dispatch_census: {p} does not exist. Run snapshots live under "
            "<repo>/checkpoints/ (gitignored), so a fresh worktree has none: "
            "pass --config /path/to/<ckpt>.config.toml, or the tracked preset "
            "`--config configs/m400.toml` (same geometry, fewer arms)."
        )
    data = tomllib.loads(p.read_text())
    model = data.get("model", data)
    return ModelConfig(**{k: v for k, v in model.items()})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    ap.add_argument("--bw", type=float, default=DEFAULT_BW / 1e12, help="TB/s")
    ap.add_argument("--set", action="append", default=[], metavar="K=V")
    ap.add_argument("--ab", action="store_true", help="shipped vs fused-route vs count-permute")
    ap.add_argument("--json", default=None)
    ap.add_argument("--top", type=int, default=0)
    args = ap.parse_args(argv)
    over: dict = {}
    for kv in args.set:
        key, _, val = kv.partition("=")
        low = val.lower()
        if low in ("1", "0", "true", "false", "yes", "no"):
            over[key] = low in ("1", "true", "yes")
        else:
            try:
                over[key] = int(val)
            except ValueError:
                try:
                    over[key] = float(val)
                except ValueError:
                    over[key] = val
    cfg = _load(args.config)
    cfg = dataclasses.replace(cfg, **over) if over else cfg
    validate(cfg)
    c, rows = census(cfg, args.batch, args.seq)
    out = report(cfg, rows, c, args.bw * 1e12, top=args.top)
    if args.ab:
        out["ab"] = ab(cfg, args.batch, args.seq)
    if args.json:
        out["path"] = args.json
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\n   wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
