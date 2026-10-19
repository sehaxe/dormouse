"""GPU gate: the compiled trainer step on the REAL backend.

`torch.compile(dynamic=True)` is what the trainer wraps the model in
(`train/loop.py:626`). On the port it hit `config.recompile_limit (8)` inside
`SparseMoE._banks`, whose cache key holds per-step Python values — the annealed
`factor_qat_alpha`, the parameters' `_version` tuple — because dynamo guards a
compiled region on every Python value it reads. `@torch.compiler.disable` on
`_banks` keeps the cache decision eager; this file is the GPU half of
`test_compile_moe.py` (CPU), on the backend that ships: inductor + CUDA,
`dynamic=True`, b2 s128, the dense blend `small` runs.

Written while a training run owns the card, so it SKIPS on the machine
discipline (`pgrep -f "dormouse trai[n]"` + `nvidia-smi`, the
`test_opt_graph.py` check). Run it when the card frees:

    CUDA_VISIBLE_DEVICES=0 uv run --extra gpu pytest tests/test_compile_moe_gpu.py -s

It asserts: the RAW `_banks` frame gains ZERO compiled variants over
`recompile_limit + 1` steps with the anneal alpha moving every step (one more
than the budget, so a run that still spent it would fall back — the 5-step
form would not discriminate), no `recompile_limit` fallback is recorded, the
loss stays finite and the parameters move. It also prints compiled vs eager
ms/step at the same shape — the number the decision needs, because the spike's
1.20x was measured on a simpler model
(`docs/research/2026-10-15-torch-spike.md`) and ours has FLA/triton graph
breaks.
"""

from __future__ import annotations

import math
import subprocess
import time

import pytest
import torch
from conftest import tiny_cfg
from torch._dynamo.eval_frame import _debug_get_cache_entry_list

from dormouse.model import DormouseModel
from dormouse.model.experts import SparseMoE

DEV = "cuda"
BATCH, SEQ = 2, 128


def _card_reason() -> str | None:
    """`None` when a GPU gate may run, else why not — `test_opt_graph.py`'s
    check, kept identical: `nvidia-smi`/`pgrep` rather than
    `torch.cuda.mem_get_info()`, so the check itself builds no CUDA context on
    a card that is in use."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    try:
        busy = subprocess.run(
            ["pgrep", "-f", "dormouse trai[n]"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001 - a gate's precondition, not a run's
        return f"cannot check the card ({e})"
    if busy:
        return f"the live training run holds the card (pids {busy}) — one GPU thing at a time"
    mib = int(used.split()[0]) if used else -1
    if mib >= 1024:
        return f"the card already holds {mib} MiB"
    return None


_GPU_REASON = _card_reason()
pytestmark = pytest.mark.skipif(_GPU_REASON is not None, reason=_GPU_REASON or "GPU gate")


@pytest.fixture
def quiet_card():
    """Re-check at RUN time (the card can become busy after collection)."""
    why = _card_reason()
    if why is not None:
        pytest.skip(f"GPU busy — nothing measured: {why}")


def _raw_banks():
    """The undecorated body (`__wrapped__` is the reference
    `torch.compiler.disable` keeps); the code object the variant count is taken
    from."""
    return getattr(SparseMoE._banks, "__wrapped__", SparseMoE._banks)


def _entries(fn) -> int:
    return len(_debug_get_cache_entry_list(fn))


def _limit_hits() -> int:
    return sum(
        v
        for k, v in dict(torch._dynamo.utils.counters["unimplemented"]).items()
        if "recompile_limit" in k
    )


def _model() -> DormouseModel:
    torch.manual_seed(0)
    return DormouseModel(tiny_cfg(moe_topk=0, max_seq_len=SEQ), device=DEV)


def _run(model: DormouseModel, steps: int, *, compiled: bool) -> list[float]:
    """The trainer's step shape on the device; returns per-step wall ms."""
    cm = torch.compile(model, dynamic=True) if compiled else model
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    ids = (torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251).to(DEV)
    tgt = ((ids + 1) % 251).to(DEV)
    times = []
    for step in range(steps):
        model.set_factor_qat_alpha(step / (2 * steps))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, loss, _ = cm(ids, tgt)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        assert math.isfinite(float(loss.detach())), f"step {step}: non-finite loss"
    return times


def test_the_compiled_step_spends_no_recompile_on_the_moving_bank_key(quiet_card):
    torch._dynamo.reset()
    raw = _raw_banks()
    steps = int(torch._dynamo.config.recompile_limit) + 1
    e0, l0 = _entries(raw), _limit_hits()
    model = _model()
    moe = next(m for m in model.modules() if isinstance(m, SparseMoE))
    before = moe.experts[0].gate_u.detach().clone()
    compiled_ms = _run(model, steps, compiled=True)
    e1, l1 = _entries(raw), _limit_hits()
    assert not torch.equal(before, moe.experts[0].gate_u.detach()), "the run did not train"
    assert e1 == e0, f"`_banks` gained {e1 - e0} compiled variants while its key moved"
    assert l1 == l0, f"a recompile_limit fallback was recorded: {l1 - l0}"
    assert _entries(model.forward) >= 1, "nothing compiled — the gate would pass vacuously"
    eager_ms = _run(_model(), steps, compiled=False)
    warm = slice(1, None)  # step 0 pays the compile warmup
    print(
        f"[compile moe gpu] b{BATCH} s{SEQ} steps={steps}\n"
        f"  compiled ms/step: {[round(t) for t in compiled_ms]} "
        f"(warm mean {sum(compiled_ms[warm]) / len(compiled_ms[warm]):.0f})\n"
        f"  eager    ms/step: {[round(t) for t in eager_ms]} "
        f"(warm mean {sum(eager_ms[warm]) / len(eager_ms[warm]):.0f})"
    )
