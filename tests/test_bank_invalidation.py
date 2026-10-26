"""The opt_graph bank-cache fix: a parameter's VALUE can move without `_version`.

Measured 2026-10-19 (the medbig2/b1night defect): an eager optimizer step moves
`_version` 0->1; the SAME step replayed from a captured CUDA graph leaves
`_version` FROZEN while the value moves. `SparseMoE._banks` keys its cache on
`_version`, so with `opt_graph = true` the key stopped moving at capture and the
bank served the step-5 build for the whole run: the experts were frozen in the
FORWARD while the optimizer kept stepping them with stale gradients (they
drifted ~44x, and the checkpoint exploded in any fresh process while the live
evals could not show it — they reused the same stale bank).

`p.data.add_(x)` is the CPU stand-in for a graph replay: the value moves, the
version does not.
"""

from __future__ import annotations

import torch
from conftest import tiny_cfg

from dormouse.model import DormouseModel
from dormouse.model.experts import SparseMoE, invalidate_banks_all


def _moe(model: DormouseModel) -> SparseMoE:
    return next(m for m in model.modules() if isinstance(m, SparseMoE))


def _bank_of(moe: SparseMoE, x: torch.Tensor) -> torch.Tensor:
    """The banked gate stack the forward would use, as one tensor."""
    return moe._banks(torch.float32)[0]


def test_data_write_moves_the_value_and_freezes_the_version():
    """The premise of the whole fix, pinned: `p.data.add_` is a version-free move."""
    p = torch.nn.Parameter(torch.zeros(4))
    v0, x0 = p._version, float(p[0])
    p.data.add_(1.0)
    assert float(p[0]) != x0, "the value must move"
    assert p._version == v0, "the version must stay — this is the graph-replay shape"


def test_a_version_frozen_write_leaves_the_cache_stale_until_invalidated():
    """The defect's mechanism, on CPU: without the invalidation the bank is stale."""
    cfg = tiny_cfg(n_experts=4, moe_topk=2, moe_capacity=0.0)
    model = DormouseModel(cfg)
    moe = _moe(model)
    x = torch.randn(2, 8, cfg.d_model)

    before = _bank_of(moe, x).clone()
    # A version-free move on every expert's gate factor (the graph-replay shape).
    for e in moe.experts:
        e.gate_u.data.add_(3.0)
    stale = _bank_of(moe, x)
    assert torch.equal(stale, before), "a frozen version must serve the STALE bank (the bug)"

    # The trainer's call: after it, the same forward sees the new values.
    assert invalidate_banks_all(model) == 1
    fresh = _bank_of(moe, x)
    assert not torch.equal(fresh, before), "invalidation must rebuild from the moved values"


def test_invalidation_is_one_build_per_step_not_per_forward():
    """The cache keeps its real job: within a step the loop's iterations reuse it."""
    cfg = tiny_cfg(n_experts=4, moe_topk=2, moe_capacity=0.0)
    model = DormouseModel(cfg)
    moe = _moe(model)
    x = torch.randn(2, 8, cfg.d_model)
    a = _bank_of(moe, x)
    b = _bank_of(moe, x)
    assert a is b, "the same key must hit the cache (the reason the cache exists)"
    invalidate_banks_all(model)
    c = _bank_of(moe, x)
    assert c is not a, "after invalidation the next call rebuilds exactly once"


def test_the_trainer_calls_the_invalidation_every_step(monkeypatch):
    """The wiring gate: `train()` must clear the cache on every trained step."""
    import dormouse.model.experts as experts
    import dormouse.train.loop as loop_mod

    calls: list[int] = []
    real = experts.SparseMoE.invalidate_banks

    def spy(self):
        calls.append(1)
        return real(self)

    monkeypatch.setattr(experts.SparseMoE, "invalidate_banks", spy)
    monkeypatch.setattr(loop_mod, "invalidate_banks_all", experts.invalidate_banks_all)

    import dataclasses

    from dormouse.config import load_run
    from dormouse.train.loop import train

    run = load_run("small")
    run.train = dataclasses.replace(
        run.train,
        steps=3,
        batch=2,
        seq_len=64,
        log_every=10**9,
        ckpt_every=10**9,
        eval=False,
        warmup_tokens=0,
    )
    run.train.data = "/tmp/opencode/bank_inval_probe.bin"
    torch.manual_seed(0)
    blob = torch.randint(0, 256, (4096,), dtype=torch.uint8).numpy().tobytes()
    with open(run.train.data, "wb") as f:
        f.write(blob)
    model = DormouseModel(run.model)
    train(model, run, device=None)
    assert len(calls) == 3, f"3 trained steps must clear the bank 3 times, saw {len(calls)}"
