"""Is `muon_lr_mult` live in a training run? Reproduce the trainer's write."""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "src")
torch.set_grad_enabled(False)

from dormouse.train.optim import MixOptimizer  # noqa: E402


class Holder(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(shape) * 0.02)
        self.bias = torch.nn.Parameter(torch.zeros(5))


def delta(mult: float, trainer_writes_lr: bool, lr: float = 3e-4):
    torch.manual_seed(7)
    m = Holder((256, 256))
    w0 = m.w.detach().clone()
    g = torch.randn(256, 256)
    opt = MixOptimizer(m, lr=lr, wd=0.1, opt="mix", muon_lr_mult=mult)
    if trainer_writes_lr:
        # train/loop.py:1760-1764, verbatim
        for group in opt.adamw.param_groups:
            group["lr"] = lr
        for group in opt.muon.param_groups:
            group["lr"] = lr
    m.w.grad = g.clone()
    opt.step()
    return m.w.detach() - w0


for writes in (False, True):
    d1 = delta(1.0, writes)
    d8 = delta(8.0, writes)
    print(
        f"trainer writes the schedule LR: {writes!s:5} | "
        f"delta(mult=1) rms {d1.pow(2).mean().sqrt():.6e} | "
        f"delta(mult=8) rms {d8.pow(2).mean().sqrt():.6e} | "
        f"identical={torch.equal(d1, d8)}"
    )
