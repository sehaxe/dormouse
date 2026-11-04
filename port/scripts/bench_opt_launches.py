"""The optimizer stage by DISPATCH COUNT, not by milliseconds.

    uv run python scripts/bench_opt_launches.py --config configs/medium.toml

Builds the config's model on CPU, fills deterministic gradients and counts the
ATen dispatches of each stage under `TorchDispatchMode`:

* `grad_norm` (the per-micro-batch firewall),
* `clip_grads` (the global norm + one fused scale),
* `zero_grad`,
* `MixOptimizer.step` — first step (state allocation) and steady state,
* the same step with the fused AdamW path disabled, for the measured drop.

The meter counts DISPATCHES; a CUDA launch inside `multi_tensor_apply` is not
one dispatch, so this is the host-side bill the step pays. It is the instrument
the report's inventory quotes (`831 -> 559` was the same kind of count); the
device-side ms belongs to a card run, which this script deliberately does not do.
"""

from __future__ import annotations

import argparse
import collections

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class Meter(TorchDispatchMode):
    def __init__(self) -> None:
        self.counts: collections.Counter[str] = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _measure(label: str, fn) -> Meter:
    meter = Meter()
    with meter:
        fn()
    print(f"{label:>24}: {meter.total:>5} dispatches", flush=True)
    for name, n in meter.counts.most_common(5):
        print(f"    {name:<44} {n:>5}")
    return meter


def _grads(model) -> None:
    gen = torch.Generator().manual_seed(0)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn(p.shape, generator=gen)


def main() -> int:
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train.optim import MixOptimizer, clip_grads, grad_norm

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/medium.toml")
    ap.add_argument("--steps", type=int, default=2, help="steady-state steps to count")
    args = ap.parse_args()

    run = load_run(args.config)
    model = DormouseModel(run.model)
    print(
        f"{args.config}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
        f"{sum(1 for p in model.parameters() if p.requires_grad)} tensors",
        flush=True,
    )
    opt = MixOptimizer(model, lr=1e-3, wd=0.01, opt=run.train.opt)
    _grads(model)
    _measure("grad_norm (micro)", lambda: grad_norm(model.parameters()))
    _measure("clip_grads", lambda: clip_grads(model.parameters(), 1.0))
    _measure("zero_grad", lambda: opt.zero_grad(set_to_none=True))

    _grads(model)
    _measure("opt.step (first)", lambda: opt.step())
    _grads(model)
    for i in range(args.steps):
        _grads(model)
        _measure(f"opt.step (steady {i + 1})", lambda: opt.step())

    # The control: torch's AdamW walk instead of the fused writer.
    model2 = DormouseModel(run.model)
    opt2 = MixOptimizer(model2, lr=1e-3, wd=0.01, opt=run.train.opt)
    opt2._fast_adamw_ok = lambda: False  # type: ignore[method-assign]
    _grads(model2)
    opt2.step()
    _grads(model2)
    _measure("opt.step (torch adamw)", lambda: opt2.step())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
