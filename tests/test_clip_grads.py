"""`clip_grads` — the fused clip that replaced the stock per-tensor walk.

The stock `torch.nn.utils.clip_grad_norm_` profiled at 252 ms/step on the
small form (2920 `linalg_vector_norm` calls); this is ONE `_foreach_norm` +
ONE `_foreach_mul_`. The gates here are EQUIVALENCE (same coefficient the
stock routine would pick) and the edge cases the fused form can get wrong.
"""

from __future__ import annotations

import pytest
import torch

from dormouse.train.optim import clip_grads


def _params_with_grads(shapes, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    params = [torch.nn.Parameter(torch.randn(*s, generator=g) * scale) for s in shapes]
    for p in params:
        p.grad = torch.randn(p.shape, generator=g)
    return params


def test_matches_the_stock_routine_when_clipping():
    shapes = [(7, 3), (4, 4), (5,)]
    a = _params_with_grads(shapes, seed=1)
    b = _params_with_grads(shapes, seed=2)
    for p, q in zip(a, b, strict=True):
        q.grad.copy_(p.grad)
    stock = torch.nn.utils.clip_grad_norm_(a, 0.5)
    fused = clip_grads(b, 0.5)
    assert fused.item() == pytest.approx(stock.item(), rel=1e-5)
    for p, q in zip(a, b, strict=True):
        assert torch.allclose(p.grad, q.grad, atol=1e-6), "same clipped gradients"


def test_no_clip_when_under_the_norm():
    shapes = [(8, 2)]
    a = _params_with_grads(shapes, seed=3, scale=0.01)
    b = _params_with_grads(shapes, seed=4, scale=0.01)
    for p, q in zip(a, b, strict=True):
        q.grad.copy_(p.grad)
    torch.nn.utils.clip_grad_norm_(a, 10.0)
    clip_grads(b, 10.0)
    for p, q in zip(a, b, strict=True):
        assert torch.equal(p.grad, q.grad), "under the norm nothing may change"


def test_empty_and_no_grads():
    out = clip_grads([torch.nn.Parameter(torch.zeros(2))], 1.0)  # no grads at all
    assert out.item() == 0.0
    assert clip_grads([], 1.0).item() == 0.0


def test_all_finite_output():
    params = _params_with_grads([(16, 8)], seed=5, scale=1e3)
    total = clip_grads(params, 1.0)
    assert torch.isfinite(total)
