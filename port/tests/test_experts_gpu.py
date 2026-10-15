"""CUDA gates for the padded-bank dispatch vs the naive one.

Skipped wherever `torch.cuda.is_available()` is false — on CPU the naive path is
what runs, and the CPU suite already covers it.

The claims: the two arms compute the SAME function (a bound the naive path can
fail: a mirrored bank, a wrong padded slot, a dropped weight row), both arms
drop the same rows under the same capacity rule, and the padded arm runs at a
capacity that really overflows without corrupting the loss. The arm that ships
fast is the cap-padded `bmm` (`moe_padded_bmm`) — `torch._grouped_mm` measured
2× slower at E=1024 and was deleted with that reason (SPIKE_REPORT.md phase 2).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg

from dormouse.instrument import Counters
from dormouse.model.experts import SparseMoE

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a CUDA device (the padded bank is the CUDA arm)",
)

DEV = "cuda"


def _moe(**over) -> SparseMoE:
    cfg = tiny_cfg(n_experts=4, moe_topk=2, rank=8, d_model=64, d_ffn=128, **over)
    return SparseMoE(cfg, device=DEV)


def _routing(moe: SparseMoE, x: torch.Tensor, seed: int = 0):
    torch.manual_seed(seed)
    probs = F.softmax(moe.router(x), dim=-1)
    weight, index = probs.topk(2, dim=-1)
    return index, weight / weight.sum(dim=-1, keepdim=True)


@pytest.mark.parametrize("bf16,bound", [(False, 1e-5), (True, 2e-3)])
def test_padded_and_naive_agree_on_the_same_routing(bf16, bound):
    moe = _moe(bf16=bf16)
    x = torch.randn(64, 64, device=DEV)
    index, weight = _routing(moe, x)
    assert moe._padded_reason(x) is None, moe._padded_reason(x)
    order, keep, dropped = moe._permute(index)
    padded = moe._padded_dispatch(x, weight, index, order, keep)
    counters = Counters()
    naive = moe._naive_dispatch(x, weight, index, order, keep, counters)
    d = (padded - naive).detach().abs()
    scale = float(naive.detach().abs().max())
    m = {"max_abs": float(d.max()), "max_rel": float(d.max()) / scale, "scale": scale}
    print(f"[moe parity] bf16={bf16} E={moe.n_experts} rows={x.shape[0]} dropped={dropped} {m}")
    assert m["max_rel"] <= bound, f"{m} exceeds {bound}"


def test_padded_path_backpropagates_into_the_expert_bank():
    moe = _moe(bf16=True)
    x = torch.randn(64, 64, device=DEV).requires_grad_(True)
    index, weight = _routing(moe, x.detach())
    order, keep, _ = moe._permute(index)
    out = moe._padded_dispatch(x, weight, index, order, keep)
    out.float().pow(2).mean().backward()
    grads = {n: float(p.grad.norm()) for n, p in moe.named_parameters() if p.grad is not None}
    dead = [n for n, p in moe.experts.named_parameters() if p.grad is None]
    print(f"[moe grads] {len(grads)} parameters with a gradient, dead={dead}")
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())
    assert dead == [], f"the padded arm must train every expert parameter: {dead}"
    assert all(v > 0 for v in grads.values()), grads


def test_both_arms_drop_the_same_rows_and_count_them_twice_the_same(monkeypatch):
    """The capacity rule is ONE rule: the padded arm is not allowed to be kinder.

    Forced through the module's own `_dispatch` twice — once padded, once with the
    reason check pinned to non-None — so the counter the log prints is the same
    number on both arms at a capacity that really overflows.
    """
    moe = _moe(moe_capacity=0.25)
    x = torch.randn(64, 64, device=DEV)
    index, weight = _routing(moe, x)
    padded_c, naive_c = Counters(), Counters()
    padded = moe._dispatch(x, weight, index, padded_c)
    monkeypatch.setattr(moe, "_padded_reason", lambda _x: "forced: naive for the counter parity")
    naive = moe._dispatch(x, weight, index, naive_c)
    print(
        f"[moe drops] padded={padded_c['moe_dropped_assignments']} "
        f"naive={naive_c['moe_dropped_assignments']} rows={index.numel()}"
    )
    assert padded_c["moe_padded_bmm"] == 1 and naive_c["moe_naive_dispatch"] == 1
    assert naive_c.why == "forced: naive for the counter parity"
    assert padded_c["moe_dropped_assignments"] == naive_c["moe_dropped_assignments"] > 0
    # A dropped row contributes nothing on either arm: both are zero-rows in the
    # same places, which is what "same rows" means beyond the counter.
    dropped_rows = (padded == 0).all(-1) & (naive == 0).all(-1)
    assert bool(dropped_rows.any())
    assert bool(torch.allclose(padded, naive, atol=1e-4))


def test_capacity_one_runs_on_the_padded_arm_on_an_overflowing_bank():
    """Cap 1.0 = `ceil(n·k/E)` rows per expert, and the over-cap rows must vanish."""
    moe = _moe(moe_capacity=1.0)
    x = torch.randn(64, 64, device=DEV)
    counters = Counters()
    out = moe(x, counters)  # the module's own forward: router → dispatch → shared → down
    print(f"[moe cap1.0] out={tuple(out.shape)} padded={counters.get('moe_padded_bmm')}")
    assert torch.isfinite(out).all()
    assert counters["moe_padded_bmm"] == 1, counters
    assert "moe_naive_dispatch" not in counters


def test_the_naive_arm_is_counted_and_the_reason_is_kept_on_cpu():
    """A CPU tensor on a CUDA box is the COUNTED fallback, not a crash."""
    moe = SparseMoE(tiny_cfg(n_experts=4, moe_topk=2, rank=8), device="cpu")
    x = torch.randn(32, 64)
    index, weight = _routing(moe, x)
    counters = Counters()
    out = moe._dispatch(x, weight, index, counters)
    print(f"[moe fallback] {counters.line()} why={counters.why}")
    assert counters["moe_naive_dispatch"] == 1 and "moe_padded_bmm" not in counters
    assert "cpu" in counters.why and torch.isfinite(out).all()
