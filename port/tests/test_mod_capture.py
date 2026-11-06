"""The MoD routed block is CUDA-graph capturable — the gate `probe_capture.py` asks.

`tests/test_mod.py` gate (c) spies on the Python host-read methods; this file
adds the op-level check (no `aten::nonzero`, `aten::unique` or
`aten::_local_scalar_dense` in the forward+backward) and the real capture,
skipped unless a card is present. The capture test is the one command to run on
the GPU:

    CUDA_VISIBLE_DEVICES=0 uv run pytest tests/test_mod_capture.py -q

The routed path is deliberately static: `C` comes from Python ints, the gather
buffers are sized from `x.shape`, and the compaction is a scatter through
`cumsum` slots (never a boolean mask or `nonzero`), so a capture window sees no
dynamic shape and no stream read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from conftest import BATCH, SEQ, tiny_cfg
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dormouse.config import ModelConfig  # noqa: E402
from dormouse.model import DormouseModel  # noqa: E402

#: `probe_capture.NON_CAPTURABLE_MARKERS`, as aten names. The probe matches
#: strings; these are the dispatch ops behind them.
FORBIDDEN = ("aten::nonzero", "aten::unique", "aten::_local_scalar_dense")


def _cfg(**over) -> ModelConfig:
    base = dict(
        block_form="transformer",
        max_iter=1,
        loop_blocks=1,
        n_layers=2,
        use_kda=False,
        use_engram=False,
        use_tsct=False,
        moe_topk=0,
        n_experts=1,
        bf16=False,
        mod_capacity=0.5,
        mod_aux_weight=0.01,
    )
    base.update(over)
    return tiny_cfg(**base)


class _Record(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def test_the_routed_ops_carry_no_capture_blocker():
    """Op-level gate: the forward AND backward dispatch no host-read aten op."""
    model = DormouseModel(_cfg())
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    rec = _Record()
    with rec:
        _logits, loss, _ = model(ids, (ids + 1) % 251)
        assert loss is not None
        loss.backward()
    bad = [op for op in rec.ops if any(op.startswith(m) for m in FORBIDDEN)]
    assert bad == [], f"capture blockers in the routed path: {sorted(set(bad))}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_routed_forward_and_backward_capture_and_replay():
    """The real gate: one graph over forward+backward, then a replay."""
    dev = torch.device("cuda")
    model = DormouseModel(_cfg(bf16=False), device=dev)
    x = torch.randint(0, 251, (BATCH, SEQ), device=dev)
    y = (x + 1) % 256
    # Warmup must finish before the capture: the allocator and `topk`'s kernel
    # selection both happen here, and a capture window refuses host work.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _logits, loss, _ = model(x, y)
            assert loss is not None
            loss.backward()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _logits, loss, _ = model(x, y)
        assert loss is not None
        loss.backward()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(loss), float(loss)
    routers = [p for n, p in model.named_parameters() if n.endswith(".router")]
    assert routers and all(p.grad is not None for p in routers)
