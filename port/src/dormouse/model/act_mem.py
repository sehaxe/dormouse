"""Activation memory: pinned host offload of the largest saved activations.

THE SEAM. Training keeps every tensor autograd saves for the backward; at
`b8 s1024`, `max_iter 3` the measured peak is 7.2 GiB of which ~6.2 GiB is
activations + the CUDA context (the fixed cost is ~1.0 GiB). This module spends
the PCIe link instead of the card: around the training forward it installs
`torch.autograd.graph.saved_tensors_hooks`, and the pack hook copies the
selected saved tensors to PINNED host memory, the unpack hook copies them back
on the backward edge. Values are COPIED, never approximated, so the loss and
every gradient are identical to the switch-off run (the gate is `1e-6` relative
on logits and gradients; on CPU the copies are exact).

THE POLICY is the classic vDNN one — the LARGEST saved tensors go first — with a
LINK-TIME budget as in SlackSwap (arXiv:2610.13003): the point is to use the
PCIe bandwidth that is idle while the card computes, not to discover capacity.
`budget_gib` is the cap on bytes moved to host PER FORWARD. A one-forward
PROFILE (the first training forward) records the sizes autograd saved; at its
end the sizes are sorted descending and the largest prefix that fits the budget
sets `threshold`. Every later forward offloads the saves at or above that
threshold, stops at the budget, and the threshold is never re-derived (the set
of op shapes is fixed by the model, so a deeper `rand_depth` step saves the same
sizes, just more iterations of them). That is one extra forward of bookkeeping,
not an extra model forward: the profiling forward IS the step's own forward.

BUFFERS. One pinned buffer per exact byte size, checked out by the pack hook
and returned to the pool by the unpack hook — so the pool is bounded by the
number of distinct save sizes, not by the step count, and `live` is the number
of buffers a backward has not yet returned. Reuse is safe on the single stream
the trainer uses: the copies are enqueued in order, so a buffer handed back by
one backward cannot be reused by a copy that overtakes it.

WHAT IS NOT OFFLOADED: parameters (a `Linear` saves its weight's `.t()` like
any operand; parking the weights would move the model, not the activations),
non-float tensors, and empty tensors. The parameter set is snapshotted from the
module tree once (`bind`), keyed by storage `data_ptr`.

COMPOSITION. With `iter_checkpoint` on, most of the body's saves live inside
the checkpoint's own hooks and never reach this pair (torch keeps ONE pair per
save; the innermost wins — `act_ckpt.py` measured the same rule). What reaches
it is the checkpoint node's INPUT state, which is exactly what survives the
forward, so the two compose without double-parking. With `grad_accum > 1` each
micro-batch's forward installs its own region and its backward returns every
buffer before the next forward.

REFUSED (LOUD, `config`): `full_graph` / `opt_graph` — a capture cannot see a
Python hook or an H2D/D2H copy, and replaying a graph that skipped them would
read a stale device tensor; `use_byteflow` — the chunked head builds a second
patch-latent forward whose save set is not measured here, so the arm would park
an unmeasured graph. A run with the flag off installs no hooks and adds no host
read to the step: `_act_offload is None` is the only branch `forward` takes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch.autograd.graph import saved_tensors_hooks

from ..instrument import Counters, loud

#: One GiB, the unit `activation_offload` is specified in.
GIB = 2**30


class _Offloaded:
    """One saved activation parked in host memory, plus what the unpack needs.

    A class, not the bare tensor, for the same reason `experts._OnHost` is one:
    the pack hook also sees values it passes through untouched (non-tensors,
    parameters, small saves), and the unpack hook must tell its cargo from those
    and hand back a tensor on the ORIGINAL device with the original shape/dtype.
    """

    __slots__ = ("host", "shape", "dtype", "device", "owner")

    def __init__(
        self,
        host: torch.Tensor,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        owner: ActivationOffload,
    ) -> None:
        self.host = host
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.owner = owner


class ActivationOffload:
    """The pack/unpack pair, the pinned pool and the budget (one per model)."""

    def __init__(self, budget_gib: float) -> None:
        if not torch.isfinite(torch.tensor(float(budget_gib))):
            loud(
                f"activation_offload={budget_gib}: not a finite GiB budget. Escape: "
                f"a finite number of GiB (e.g. 4.0), or 0.0."
            )
        if not (budget_gib > 0.0):
            loud(
                f"activation_offload={budget_gib}: the budget is a GiB count of "
                f"activations moved to host PER FORWARD, so 0 is the off value and a "
                f"negative one is a typo. Escape: activation_offload > 0 (e.g. 4.0), "
                f"or 0.0 to leave the hooks uninstalled."
            )
        self.budget = int(budget_gib * GIB)
        #: Set at the end of the PROFILE forward: the smallest save size that
        #: still fits the budget, taken largest-first. `None` before the first
        #: training forward.
        self.threshold: int | None = None
        self._profiled = False
        self._sizes: list[int] = []
        #: `[counters, bytes_used]` per active forward region (a stack because a
        #: model could nest regions; the pack hook reads the top).
        self._regions: list[list] = []
        self._pool: dict[int, list[torch.Tensor]] = {}
        self._live = 0
        self._param_storages: set[int] = set()
        self._pinned_ok = bool(torch.cuda.is_available())
        self._pageable_fallback = False

    # --- setup ----------------------------------------------------------

    def bind(self, model: torch.nn.Module) -> None:
        """Snapshot the parameter storages, the set the pack hook must NOT park."""
        self._param_storages = {
            p.untyped_storage().data_ptr() for p in model.parameters() if p.numel()
        }

    # --- the hooks ------------------------------------------------------

    def _acquire(self, nbytes: int) -> torch.Tensor:
        """A host buffer of exactly `nbytes`: pooled first, pinned when possible."""
        free = self._pool.get(nbytes)
        if free:
            return free.pop()
        if self._pinned_ok:
            try:
                return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            except RuntimeError:
                # A CUDA build with no visible device: the pinned allocator is
                # unavailable. Pageable memory still holds the values; the
                # degradation is COUNTED so a reader can tell the arms apart.
                self._pinned_ok = False
                self._pageable_fallback = True
        return torch.empty(nbytes, dtype=torch.uint8)

    def _release(self, host: torch.Tensor) -> None:
        self._pool.setdefault(host.numel(), []).append(host)
        self._live -= 1

    def _pack(self, t: object) -> object:
        """The pack hook: park a large float activation, pass everything else."""
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            return t
        n = t.numel()
        if n == 0 or t.device.type not in ("cpu", "cuda"):
            return t
        if self._param_storages and t.untyped_storage().data_ptr() in self._param_storages:
            return t
        nbytes = n * t.element_size()
        assert self._regions, "a save outside an `ActivationOffload.region`"
        counters, used = self._regions[-1]
        if not self._profiled:
            self._sizes.append(nbytes)
            return t
        if self.threshold is None or nbytes < self.threshold or used + nbytes > self.budget:
            return t
        host = self._acquire(nbytes)
        # Exact copy: same dtype, same bits, no codec. `non_blocking` is what
        # makes the pinned buffer worth having (D2H overlaps the forward).
        host.view(t.dtype).view(t.shape).copy_(t.detach(), non_blocking=True)
        self._regions[-1][1] = used + nbytes
        self._live += 1
        counters.count("offload_tensors")
        counters.count("offload_gib_x100", nbytes * 100 // GIB)
        return _Offloaded(host, t.shape, t.dtype, t.device, self)

    def _unpack(self, packed: object) -> object:
        """The unpack hook: copy back onto the saved device, return the buffer."""
        if not isinstance(packed, _Offloaded):
            return packed
        out = packed.host.view(packed.dtype).view(packed.shape).to(packed.device, non_blocking=True)
        packed.owner._release(packed.host)
        return out

    # --- the region -----------------------------------------------------

    @contextmanager
    def region(self, counters: Counters) -> Iterator[None]:
        """Install the pair for ONE training forward; profile on the first.

        The hooks only have to be installed while autograd RECORDS the saves;
        the unpack hook travels inside the graph and fires later, so the context
        ends before `.backward()`. Each region carries its own `Counters`, which
        is what makes the counters land on the micro-batch that produced the
        saves under `grad_accum`.
        """
        self._regions.append([counters, 0])
        try:
            with saved_tensors_hooks(self._pack, self._unpack):
                yield
            if not self._profiled:
                self._finish_profile(counters)
        finally:
            self._regions.pop()

    def _finish_profile(self, counters: Counters) -> None:
        """Pick the largest-first threshold that fits `budget`, once.

        A budget that cannot hold even the largest saved tensor would leave the
        flag on while the arm never fired, which is the silent-inert shape this
        repo refuses: LOUD with the measured size in the message.
        """
        self._profiled = True
        sizes = sorted(self._sizes, reverse=True)
        self._sizes = []
        if not sizes:
            loud(
                "activation_offload: the first training forward saved no float "
                "activation above zero elements — there is nothing to park, so the "
                "flag would be inert. Escape: activation_offload=0.0, or a model "
                "whose forward saves activations."
            )
        if sizes[0] > self.budget:
            loud(
                f"activation_offload={self.budget / GIB:g} GiB cannot hold the largest "
                f"saved tensor ({sizes[0] / GIB:.3f} GiB), so the largest-first arm "
                f"would never fire. Escape: a budget >= {sizes[0] / GIB:.3f} GiB "
                f"(the measured largest save), or 0.0 to leave the hooks off."
            )
        total = 0
        threshold = None
        for size in sizes:
            if total + size > self.budget:
                break
            total += size
            threshold = size
        self.threshold = threshold
        counters.count("offload_profile_forward")
        if self._pageable_fallback:
            counters.count("offload_pageable")

    # --- the report -----------------------------------------------------

    def stats(self) -> dict[str, int | bool | None]:
        """The bookkeeping a gate (and a curious reader) can check."""
        return {
            "threshold_bytes": self.threshold,
            "pool_tensors": sum(len(v) for v in self._pool.values()),
            "pool_bytes": sum(k * len(v) for k, v in self._pool.items()),
            "live": self._live,
            "budget_bytes": self.budget,
            "pinned": self._pinned_ok,
        }
