r"""The per-optimizer-step QAT cache: quantize each weight ONCE per step.

Measured on the 9.2 M polygon (2026-10-21): the full-quantized recipe
(`factor_qat=ternary` + `dense_qat=ternary` + `act_quant=int4`) ran at
**17.1 k tok/s against 40.9 k bf16 — 2.4x slower**. The waste was structural:
every forward re-quantized every weight (absmean, divide, round, clamp,
multiply), while the weights change only once per optimizer step.

The cache stores the DETACHED raw projection `q(w)` per weight and the forward
re-composes the straight-through tensor from it:

    q = qat_values(w, kind, alpha, dim)      once per step, detached
    y = w + alpha * (q - w).detach()         every forward, one sub + one add

The gradient is the same STE contract as the uncached path (dL/dw = dL/dw_q via
the detached residual), and the forward is BIT-IDENTICAL: the cached path
evaluates the same expression on the same `q` values, not a re-associated one
(`w + (wq - w)` loses the last bits; `w + alpha*(q - w)` does not).

THE TRAP THIS CACHE INHERITS (memory ID 77, the 2026-10-19 medbig2/b1night
defect): a CUDA-graph replay moves a parameter's VALUE without bumping
`_version`, so a `_version`-keyed cache freezes at capture. This cache is keyed
on the TENSOR OBJECT alone and the TRAINER invalidates it once per trained step
(`invalidate_qat_cache_all`, called beside `invalidate_banks_all` in
`train/loop.py`, `train/sft.py`, `post/rl_train.py`). Nothing here reads
`_version`; a stale cache is a caller bug, and `tests/test_qat_cache.py` pins
both halves (stale without the call, fresh with it).

`alpha` is part of the cache key as well: the anneal ramp moves it every step,
so even a missed invalidation cannot serve a projection blended at last step's
alpha. `alpha == 0` stores `w` itself and `ste` returns `w` untouched — the
bf16 path, bit for bit.

THE SECOND SEAM: a bank is ONE projection, not E. `SparseMoE._banks` builds
`[E, ·, ·]` stacks and used to call the projector once per expert — E launch
chains per projection per step (768 at medium, E = 256), which is launch-bound
work for a tensor the same size as one call's. `qat_project_stacked` takes the
whole stack through `qat_values` once (the STE residual folded in place), and
`refresh_stacked` is the cached form of the same idea. `qat_refresh_calls` /
`qat_refresh_tensors` (below) count the two shapes apart, so a regression to
per-tensor shows up as calls ≈ tensors instead of calls ≪ tensors.
"""

from __future__ import annotations

import torch
from torch import nn

#: Refresh telemetry, per step. `qat_refresh_calls` is how many times the
#: quantizer ran (`tsct.qat_values`); `qat_refresh_tensors` is how many weight
#: tensors those calls covered. ONE stacked call over `[E, ·, ·]` is one call
#: and E tensors; the pre-stacked bank was E calls of one tensor each. The pair
#: is what separates "one pass per site" from "one pass per tensor" on the log
#: line — a call count alone cannot tell a stack from a loop.
_REFRESH_CALLS = 0
_REFRESH_TENSORS = 0


def note_qat_values(w: torch.Tensor) -> None:
    """Count ONE quantizer invocation; its leading axis is the stack it covers."""
    global _REFRESH_CALLS, _REFRESH_TENSORS
    _REFRESH_CALLS += 1
    _REFRESH_TENSORS += int(w.shape[0]) if w.ndim >= 3 else 1


def take_qat_refresh_counts() -> tuple[int, int]:
    """`(calls, tensors)` since the last take, then reset — the per-step count."""
    global _REFRESH_CALLS, _REFRESH_TENSORS
    out = (_REFRESH_CALLS, _REFRESH_TENSORS)
    _REFRESH_CALLS = _REFRESH_TENSORS = 0
    return out


def qat_project_stacked(
    w: torch.Tensor, kind: str, alpha: float, *, dim: tuple | None = None
) -> torch.Tensor:
    """`w + alpha·(q(w) − w).detach()` for a whole stack in ONE quantizer call.

    The banked expert seam (`SparseMoE._banks`): the `[E, d, r]` stack is
    projected as ONE tensor, so the whole bank costs one quantizer chain (one
    launch per elementwise op) instead of E. `dim` is the per-slice absmean axis
    (`(1, 2)` for a bank); the expression is `factor_qat_project`'s, evaluated
    on the stack, so every element is the same arithmetic on the same values —
    a slice is bit-identical to the same tensor projected alone, `mxfp4`
    included (`mxfp.fake_quant_mx` pads per ROW, so a block never mixes two
    slices; pinned by `tests/test_qat_cache.py`).

    NOT a cache: `_banks` already caches the built bank once per step, and
    storing a second copy of every `q` would double the bank's footprint. The
    residual is folded IN PLACE on `q` (the projection is a constant for the
    step), so the peak is `w`, `q` and the result — not four transients.
    """
    if kind == "none" or alpha == 0.0:
        # `alpha = 0` is the master: no projection, the same short-circuit the
        # cache takes (and the same bits `w + 0·(q − w)` gives for a finite q).
        return w
    # Lazy import: `tsct` imports this module at module level.
    from .tsct import qat_values

    with torch.no_grad():
        q = qat_values(w, kind, alpha, dim=dim)
        q.sub_(w).mul_(alpha)  # the detached residual, in place
    return (w + q).to(w.dtype)


class QuantCache:
    """One detached quantized copy per weight, reused until `invalidate()`.

    `refresh_stacked(w, kind, alpha, dim)` returns the DETACHED raw projection
    for a whole `[N, ·, ·]` stack — ONE quantizer call covering N tensors, the
    banked expert path's seam. `refresh` is the same call with `N = 1` (a lone
    tensor: `out_proj`, `lm_head`, a dense weight). A miss runs
    `tsct.qat_values` once, a hit is a dict lookup that bumps `hits`; `ste(w)`
    re-composes `w + alpha·(q − w).detach()` from the stored entry, so the
    forward value equals the uncached path's and the gradient is identity.
    `invalidate()` drops every entry; the trainer calls it once per optimizer
    step, exactly like `SparseMoE.invalidate_banks`.

    `@torch.compiler.disable` keeps the cache DECISION out of dynamo's compiled
    region (the same reason `SparseMoE._banks` carries it: per-step Python state
    read inside a compiled forward otherwise forces one recompile per step).
    """

    def __init__(self) -> None:
        # id(w) -> (w, kind, alpha, dim, q): `w` is held so id() cannot be
        # reused while the entry lives, and `is` pins identity (two same-shaped
        # weights must not share a projection).
        self._q: dict[int, tuple[torch.Tensor, str, float, tuple | None, torch.Tensor]] = {}
        #: Cache hits since the last `take_hits()` — the `qat_reuse` counter.
        self.hits = 0

    @torch.compiler.disable
    def refresh_stacked(
        self, w: torch.Tensor, kind: str, alpha: float, dim: tuple | None = None
    ) -> torch.Tensor:
        """The detached projection for a whole stack; ONE call, N tensors.

        `w` is `[N, ·, ·]` (a bank) or any tensor `qat_values` accepts; `dim` is
        the per-slice absmean axis. The entry is keyed on `w`'s identity, so
        `ste(w)` reads it back exactly as for a lone tensor.
        """
        ent = self._q.get(id(w))
        if ent is not None and ent[0] is w and ent[1] == kind and ent[2] == alpha and ent[3] == dim:
            self.hits += 1
            return ent[4]
        if alpha == 0.0:
            # `alpha = 0` is the master: no projection to compute, and the
            # s3t mask must not be drawn for a blend that multiplies it by 0.
            q = w.detach()
        else:
            # Lazy import: `tsct` imports this module at module level, so the
            # math it owns is imported here at call time (the same cycle-break
            # `TSCTLinear._act` uses for `act_quant`).
            from .tsct import qat_values

            # No graph: the projection is a constant for the whole step (the
            # STE residual below is what carries the gradient).
            with torch.no_grad():
                q = qat_values(w, kind, alpha, dim=dim).detach()
        self._q[id(w)] = (w, kind, alpha, dim, q)
        return q

    def refresh(
        self, w: torch.Tensor, kind: str, alpha: float, dim: tuple | None = None
    ) -> torch.Tensor:
        """A lone tensor is a stack of one — the SAME code path, `N = 1`."""
        return self.refresh_stacked(w, kind, alpha, dim)

    def ste(self, w: torch.Tensor) -> torch.Tensor:
        """`w + alpha·(q − w).detach()` — the STE the forward reads.

        The exact expression `tsct.factor_qat_project` evaluates uncached, so a
        warm cache is bit-identical to a cold one; the residual is detached, so
        dL/dw = 1 (the STE contract).
        """
        ent = self._q.get(id(w))
        assert ent is not None and ent[0] is w, (
            "QuantCache.ste on a weight that was never refreshed: call "
            "`refresh(w, kind, alpha)` first (factor_qat_project does this for "
            "every caller that passes a cache)"
        )
        alpha, q = ent[2], ent[4]
        if alpha == 0.0:
            return w
        return (w + alpha * (q - w).detach()).to(w.dtype)

    def invalidate(self) -> None:
        """Drop every entry — the trainer's once-per-trained-step call."""
        self._q.clear()

    def take_hits(self) -> int:
        """Return the hits since the last take and reset — the per-forward count."""
        n, self.hits = self.hits, 0
        return n


def attach_qat_caches(model: nn.Module, *, dense: bool) -> list[QuantCache]:
    """Collect every module's cache, and give each plain `nn.Linear` one.

    Called ONCE from `DormouseModel.__init__`. Modules whose switch is on
    (`TSCTLinear`, `DenseLinear`, `LoopBlock`, `SparseMoE`, `DualLatentGLU`)
    built their own cache in their constructor; the plain `nn.Linear` sites
    (`qat_linear`) have no constructor seam, so the model's own build is where
    they get one. `dense=false` leaves them cacheless (the uncached path).
    """
    caches: list[QuantCache] = []
    seen: set[int] = set()
    for m in model.modules():
        c = getattr(m, "_qat_cache", None)
        if c is not None and id(c) not in seen:
            seen.add(id(c))
            caches.append(c)
        if dense and type(m) is nn.Linear and getattr(m, "_qat_cache", None) is None:
            m._qat_cache = QuantCache()
            caches.append(m._qat_cache)
    return caches


def invalidate_qat_cache_all(model: nn.Module) -> int:
    """Drop every QAT cache in `model`; returns how many were cleared.

    The trainer's ONE per-trained-step call, beside `invalidate_banks_all`
    (`train/loop.py`, `train/sft.py`, `post/rl_train.py`): a CUDA-graph replay
    moves a parameter's VALUE without bumping `_version` (measured 2026-10-19),
    so no version-keyed cache can outlive a step in which the optimizer ran.
    This one does not even look at `_version`; the call is the invalidation.
    """
    seen: set[int] = set()
    n = 0
    for m in model.modules():
        c = getattr(m, "_qat_cache", None)
        if c is not None and id(c) not in seen:
            seen.add(id(c))
            c.invalidate()
            n += 1
    return n
