"""The MoE bank cache under `torch.compile(dynamic=True)`: the cache DECISION
stays out of dynamo's guards, and the decision itself is unchanged.

`SparseMoE._banks`' cache key holds per-step Python values — the annealed
`factor_qat_alpha`, the experts' parameter `_version` tuple, the grad state —
and dynamo guards a compiled region on every Python value it reads. Measured
2026-10-18 (`docs/research/2026-10-18-compile-moe.md` §2-3): without the
decorator the frame spends one compiled variant per step until it hits
`config.recompile_limit` and falls back to eager (the trainer's step measured
2939 ms against 573 eager); with `@torch.compiler.disable` on `_banks` it
gains ZERO variants while the same key moves. The decorator is the one-line
fix; these gates pin both halves of it.

The step shape is the trainer's own, tiny: `torch.compile(model, dynamic=True)`,
a per-step `set_factor_qat_alpha` (`train/loop.py:750` — the annealed run's
mover), backward, optimizer step (the `_version` mover). `backend="eager"` on
purpose: the guard/recompile machinery is dynamo's and backend-independent (the
research probe used the same backend), while an inductor compile of the whole
tiny model costs ~15 s of the suite's time for nothing this gate asserts. The
GPU gate (`test_compile_moe_gpu.py`) runs the real backend.

Counting is DELTA-based — dynamo's caches and counters are process-global. Two
signals, both real in torch 2.13.0+cu129 (the lock's pin): the compiled-variant
count of the RAW `_banks` code object (`_debug_get_cache_entry_list`, what the
research probe used) and the recompile-limit entry in
`torch._dynamo.utils.counters["unimplemented"]`, the counter behind the
`hit config.recompile_limit` warning. `counters["guard_failures"]` does not
exist in this torch (the keys are aot_autograd / aten_mm_info / frames /
graph_break / inductor / resumes / stats / unimplemented), and `frames` counts
every dynamo frame in the run — it cannot isolate `_banks`. A missing counter
fails the gate, it does not skip: a gate that cannot fail is a defect.
"""

from __future__ import annotations

import math

import torch
from conftest import tiny_cfg
from torch._dynamo.eval_frame import _debug_get_cache_entry_list

from dormouse.model import DormouseModel
from dormouse.model.experts import SparseMoE

#: The dense blend `small` ships (`configs/small.toml`, `moe_topk = 0`) — the
#: only arm that calls `_banks` on CPU (`_dispatch` sends a CPU tensor to the
#: naive loop, which never touches the bank; `test_expert_banks.py`).
DENSE = dict(moe_topk=0)
BATCH, SEQ = 2, 64


def _raw_banks():
    """The undecorated body — the control's function, and the code object the
    compiled-variant count is taken from (`__wrapped__` is the reference
    `torch.compiler.disable` keeps). The `getattr` fallback keeps the failure a
    clean assertion if the decorator is ever removed, not an AttributeError."""
    return getattr(SparseMoE._banks, "__wrapped__", SparseMoE._banks)


def _entries(fn) -> int:
    """Compiled variants of `fn`'s frame — dynamo's own per-code cache."""
    return len(_debug_get_cache_entry_list(fn))


def _limit_hits() -> int:
    """The counter behind `hit config.recompile_limit (8)`: dynamo records the
    fallback under `unimplemented`, keyed by its full explanation."""
    return sum(
        v
        for k, v in dict(torch._dynamo.utils.counters["unimplemented"]).items()
        if "recompile_limit" in k
    )


def _steps() -> int:
    """One more than the budget, so a run that spends it hits the fallback."""
    return int(torch._dynamo.config.recompile_limit) + 1


def _drive(model: DormouseModel, steps: int) -> list[float]:
    """The trainer's step shape, tiny: anneal → forward → backward → opt step."""
    cm = torch.compile(model, dynamic=True, backend="eager")
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    tgt = (ids + 1) % 251
    losses = []
    for step in range(steps):
        model.set_factor_qat_alpha(step / (2 * steps))
        _, loss, _ = cm(ids, tgt)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    return losses


def _expert0_gate_u(model: DormouseModel) -> torch.Tensor:
    moe = next(m for m in model.modules() if isinstance(m, SparseMoE))
    return moe.experts[0].gate_u


def test_the_moving_bank_key_stays_out_of_dynamos_guards():
    torch._dynamo.reset()
    raw = _raw_banks()
    e0, l0 = _entries(raw), _limit_hits()
    model = DormouseModel(tiny_cfg(**DENSE))
    before = _expert0_gate_u(model).detach().clone()
    losses = _drive(model, _steps())
    e1, l1 = _entries(raw), _limit_hits()
    print(f"[compile moe] decorated: banks_entries {e0}->{e1}, limit_hits {l0}->{l1}")
    assert all(math.isfinite(v) for v in losses), losses
    assert not torch.equal(before, _expert0_gate_u(model).detach()), "the run did not train"
    assert e1 == e0, f"`_banks` gained {e1 - e0} compiled variants while its key moved"
    assert l1 == l0, f"a recompile_limit fallback was recorded: {l1 - l0}"
    # Not vacuous: under the SAME run the model's own frame did compile.
    assert _entries(model.forward) >= 1, "nothing compiled — the gate would pass vacuously"


def test_the_control_without_the_decorator_burns_the_recompile_budget(monkeypatch):
    """The same run against the RAW `_banks`: one compiled variant per step,
    then the limit fallback. This is the defect the decorator removes, and the
    proof the gate above can fail."""
    torch._dynamo.reset()
    raw = _raw_banks()
    monkeypatch.setattr(SparseMoE, "_banks", raw)
    e0, l0 = _entries(raw), _limit_hits()
    steps = _steps()
    losses = _drive(DormouseModel(tiny_cfg(**DENSE)), steps)
    e1, l1 = _entries(raw), _limit_hits()
    print(f"[compile moe] raw control: banks_entries {e0}->{e1}, limit_hits {l0}->{l1}")
    assert all(math.isfinite(v) for v in losses), losses
    limit = int(torch._dynamo.config.recompile_limit)
    assert e1 - e0 >= limit, f"the raw `_banks` gained only {e1 - e0} variants in {steps} steps"
    assert l1 - l0 >= 1, "the raw run never hit the recompile limit — the gate cannot fail"


def test_the_decorator_does_not_change_the_bank_decision():
    """The eager path the decorator must not touch: an in-place parameter change
    rebuilds the bank (the `_version` clause), an unchanged key still HITS (the
    conversion the cache exists to remove), and the forward reads the rebuilt
    bank."""
    moe = SparseMoE(tiny_cfg(**DENSE), device="cpu")
    x = torch.randn(16, moe.d)
    out0 = moe(x).detach()
    dt = moe._bank_dtype()
    bank0 = moe._banks(dt)[0]
    assert moe._banks(dt)[0] is bank0, "an unchanged key rebuilt the bank"
    with torch.no_grad():
        moe.experts[0].gate_u.mul_(2.0)  # in place: bumps `_version`
    bank1 = moe._banks(dt)[0]
    assert bank1 is not bank0, "a changed parameter served the stale bank"
    assert torch.equal(bank1[0], moe.experts[0].gate_u.to(dt))
    assert not torch.allclose(out0, moe(x).detach()), "the forward read the stale bank"
