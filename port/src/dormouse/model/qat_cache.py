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
"""

from __future__ import annotations

import torch
from torch import nn


class QuantCache:
    """One detached quantized copy per weight, reused until `invalidate()`.

    `refresh(w, kind, alpha)` returns the DETACHED raw projection — a miss runs
    `tsct.qat_values` once, a hit is a dict lookup that bumps `hits`.
    `ste(w)` re-composes `w + alpha·(q − w).detach()` from the stored entry, so
    the forward value equals the uncached path's and the gradient is identity.
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
    def refresh(
        self, w: torch.Tensor, kind: str, alpha: float, dim: tuple | None = None
    ) -> torch.Tensor:
        """The detached projection for `w`; computed on a miss, reused on a hit."""
        ent = self._q.get(id(w))
        if (
            ent is not None
            and ent[0] is w
            and ent[1] == kind
            and ent[2] == alpha
            and ent[3] == dim
        ):
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
