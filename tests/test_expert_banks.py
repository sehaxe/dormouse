"""The expert bank cache (`SparseMoE._banks`) and the ONE way it can be poisoned.

`_banks` stacks the per-expert parameters into `[E, ·]` tensors once per step and
hands the same tensors to every arm that batched-matmuls the bank. Its cache key
was the parameters' in-place `_version`, which is enough for VALUES and not
enough for the GRAPH: an eval forward runs on the same model under
`torch.no_grad()` and — because the optimizer step just bumped `_version` — is
exactly the call that rebuilds the entry, with `requires_grad=False` stacks. The
next training step then read the bank as a constant and **every expert parameter
came back with `.grad is None`** (the battery measured 20/20 dead, both adjoints,
until an unrelated `_version` bump rebuilt the cache).

The defect is device-independent — a CPU forward shows it — so these gates are
NOT `@CUDA`'d, and the arm they drive (`_padded_dispatch`) is called directly
rather than through `_dispatch`, whose `_padded_reason` sends a CPU tensor to the
reference loop, which never touches `_banks` at all.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from conftest import tiny_cfg

from dormouse.model.experts import SparseMoE


def _moe() -> SparseMoE:
    return SparseMoE(tiny_cfg(n_experts=4, moe_topk=2, rank=8, d_model=64, d_ffn=128), device="cpu")


def _routing(moe: SparseMoE, x: torch.Tensor, seed: int = 0):
    """The router's top-k as a FIXED assignment (no graph back into the router)."""
    torch.manual_seed(seed)
    with torch.no_grad():
        probs = F.softmax(moe.router(x), dim=-1)
        weight, index = probs.topk(2, dim=-1)
    return index, weight / weight.sum(dim=-1, keepdim=True)


def _grad_check(moe: SparseMoE, x: torch.Tensor, index, weight) -> None:
    """One tracked forward+backward; every expert parameter must come back live."""
    order, keep, _ = moe._permute(index)
    moe.zero_grad(set_to_none=True)
    xi = x.detach().clone().requires_grad_(True)
    out = moe._padded_dispatch(xi, weight, index, order, keep)
    out.float().pow(2).mean().backward()
    dead = [n for n, p in moe.experts.named_parameters() if p.grad is None]
    n = sum(1 for _ in moe.experts.parameters())
    print(f"[bank cache] expert params with a grad: {n - len(dead)}/{n} dead={dead}")
    assert not dead, f"the bank cache served constants to a tracked forward: {dead}"
    for name, p in moe.experts.named_parameters():
        assert torch.isfinite(p.grad).all(), f"{name}: a non-finite gradient"
    assert xi.grad is not None and torch.isfinite(xi.grad).all()


def test_a_no_grad_forward_cannot_poison_the_bank_cache():
    """The exact sequence: eval forward (no_grad) → training step → live grads."""
    moe = _moe()
    x = torch.randn(32, moe.d)
    index, weight = _routing(moe, x)

    order, keep, _ = moe._permute(index)
    with torch.no_grad():
        moe._padded_dispatch(x, weight, index, order, keep)
    assert getattr(moe, "_bank_cache", None) is None, (
        "a no-grad forward STORED a bank: the next step's backward reads it as a constant"
    )

    _grad_check(moe, x, index, weight)


def test_a_cache_entry_that_cannot_serve_a_tracked_forward_is_rebuilt():
    """The stale-cache check includes the GRAD state, not only `_version`.

    The entry is planted with the key a tracked call computes — which is what the
    pre-fix code did for real, since its key could not tell the two builds apart.
    """
    moe = _moe()
    x = torch.randn(32, moe.d)
    index, weight = _routing(moe, x)
    dt = moe._bank_dtype()
    versions = tuple(p._version for e in moe.experts for p in e.parameters())
    with torch.no_grad():
        g_u, u_u, v, g_s, u_s = moe._banks(dt)
        assert not g_u.requires_grad
    moe._bank_cache = (
        # The key shape is `_banks`': the experts' `_version`s, then the SHARED
        # basis's (`()` when `share_u` is off — this MoE), then the grad state.
        (dt, moe.factor_qat, moe.factor_qat_alpha, versions, (), True),
        (g_u, u_u, v),
        (g_s, u_s),
    )

    _grad_check(moe, x, index, weight)


def test_a_tracked_bank_still_serves_the_eval_forward():
    """The reuse the fix must keep: a graph-carrying bank is fine under no_grad.

    Without this the eval — 20 forwards per eval line — would re-stack and
    re-convert the whole bank on every one of them, which is the ~15 ms host +
    ~4 ms copies per call the cache was built to remove.
    """
    moe = _moe()
    dt = moe._bank_dtype()
    tracked = moe._banks(dt)[0]
    assert tracked.requires_grad
    with torch.no_grad():
        again = moe._banks(dt)[0]
    assert again is tracked, "a no-grad forward rebuilt a bank it could have reused"
