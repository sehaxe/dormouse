"""SL1 gates: the routed expert pass with its saved activations parked in host RAM.

`docs/research/2026-10-16-budget-savers.md` §5 (SL1 = N2 recompute + N3 SCO): the
MoE block owns the activation peak, and the tensors that own it are the
`[E, cap, d]` / `[E, cap, f]` bank materializations plus the saved GLU gates.
`ModelConfig.moe_offload` puts the pass `_dispatch` picked inside
`checkpoint(use_reentrant=False)` with `torch.autograd.graph.saved_tensors_hooks`
INSIDE it, so those saves live in host RAM until the backward edge asks for them.

What is claimed here, and what is deliberately NOT:

1. **switch off = today's path**, and the counter says so by its absence (`a`);
2. **switch on = the arm ran and the model still learns** (`b`) — `moe_offloaded`
   on the log line is the ADR-0019 proof, since a zero means the arm did not run;
3. **EXACTNESS, the gate**: loss bit-identical and gradients ≤ 1e-6 relative on
   CPU (`c`, `d`), the same two claims at bf16 with the padded bank's 2e-3 bound
   on a CUDA box (`f`, skipped without one — the file stays skippable);
4. **the switch is a real config field**, OFF by default, reachable through
   `small.toml` and through `--set`/`model_overrides` (`e`).

**VRAM is NOT claimed here.** CPU cannot show a device peak, and the card belongs
to another job this round, so the peak claim (SL1's pre-registered prediction:
≥25% off at b2·s512 b1 shapes) is OPEN. The measurement, when the GPU is free:

    uv run --extra gpu python - <<'PY'
    import torch
    from torch.autograd.graph import saved_tensors_hooks  # noqa: F401
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    run = load_run("b1", train_overrides={"batch": 2, "seq_len": 512})
    ids = torch.randint(0, 256, (2, 512), device="cuda")
    for offload in (False, True):
        model = DormouseModel(dataclasses.replace(run.model, moe_offload=offload), device="cuda")
        model.train()
        torch.cuda.reset_peak_memory_stats()
        _, loss, c = model(ids, (ids + 1) % 256)
        loss.backward()
        print(offload, "peak_MB", torch.cuda.max_memory_allocated() / 2**20, c.line())
    PY

The two arms must print the same loss and differ in `max_memory_allocated()`;
`c.line()` must carry `moe_offloaded=2` (max_iter) on the ON arm and not on the
OFF one.
"""

from __future__ import annotations

import copy
import dataclasses
import re

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg, tiny_run

from dormouse.config import PRESET_DIR, ModelConfig, load_preset, load_run, parse_preset
from dormouse.instrument import Counters
from dormouse.model import DormouseModel
from dormouse.model.experts import SparseMoE
from dormouse.train import train

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a CUDA device (the padded bank is the CUDA arm)",
)


def _moe(model: DormouseModel) -> SparseMoE:
    return next(m for m in model.modules() if isinstance(m, SparseMoE))


def _switch(model: DormouseModel, on: bool) -> SparseMoE:
    """Flip the ONE switch on ONE module, so the A/B shares every parameter.

    Building two models would compare two initialisations; flipping the flag on
    one model makes any difference in the loss or the gradients an offload bug
    and nothing else.
    """
    moe = _moe(model)
    moe.moe_offload = on
    return moe


def _loss_of(line: str) -> float:
    m = re.search(r"loss=([\d.]+)", line)
    assert m is not None, line
    return float(m.group(1))


def test_a_switch_off_is_todays_path_and_counts_nothing(byte_tensor):
    """(a) The OFF arm is unchanged: the routed pass runs, and no offload arm ran."""
    torch.manual_seed(0)
    cfg = tiny_cfg(moe_topk=2)
    model = DormouseModel(cfg)
    ids, targets = byte_tensor()
    _, loss, counters = model(ids, targets)
    loss.backward()
    assert cfg.moe_offload is False
    assert "moe_offloaded" not in counters, counters.line()
    assert counters["moe_naive_dispatch"] == cfg.max_iter  # CPU: the reference arm
    assert torch.isfinite(loss)


def test_b_switch_on_counts_and_the_model_still_learns(tmp_path):
    """(b) ON: `moe_offloaded` on the log line, loss descending over the window."""
    run = tiny_run(tmp_path, steps=30, lr=0.01, log_every=10)
    model = DormouseModel(dataclasses.replace(run.model, moe_offload=True))
    assert _moe(model).moe_offload is True
    lines: list[str] = []
    report = train(model, run, log=lines.append)
    assert report["nan_skipped"] == 0
    # 2 dispatches per forward (max_iter) x log_every=10 steps, on the line a
    # reader uses to tell which arm produced the loss.
    assert "moe_offloaded=20" in lines[-1], lines[-1]
    assert _loss_of(lines[-1]) < _loss_of(lines[0]), lines


def test_c_loss_is_bit_identical_with_the_switch_on(byte_tensor):
    """(c) The gate: the SAME forward, hook-offloaded saves, bit-equal loss."""
    torch.manual_seed(0)
    model = DormouseModel(tiny_cfg(moe_topk=2, moe_capacity=0.25))  # really drops rows
    ids, targets = byte_tensor()
    losses = {}
    for on in (False, True):
        _switch(model, on)
        model.zero_grad(set_to_none=True)
        _, loss, counters = model(ids, targets)
        losses[on] = loss.detach()
        if on:
            assert counters["moe_offloaded"] == 2, counters.line()
    assert torch.equal(losses[False], losses[True]), (losses[False], losses[True])


@pytest.mark.parametrize("capacity", [0.25, 2.0])
def test_d_gradients_are_the_same_to_1e_6_relative(byte_tensor, capacity):
    """(d) Grad parity on the tinyset, at a dropping and a non-dropping capacity."""
    torch.manual_seed(0)
    model = DormouseModel(tiny_cfg(moe_topk=2, moe_capacity=capacity))
    ids, targets = byte_tensor()
    grads: dict[bool, dict[str, torch.Tensor]] = {}
    for on in (False, True):
        _switch(model, on)
        model.zero_grad(set_to_none=True)
        _, loss, _ = model(ids, targets)
        loss.backward()
        grads[on] = {
            n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None
        }
    assert set(grads[False]) == set(grads[True])
    worst, where = 0.0, ""
    for name, g in grads[False].items():
        rel = float((g - grads[True][name]).norm()) / max(float(g.norm()), 1e-12)
        assert torch.isfinite(grads[True][name]).all(), name
        if rel > worst:
            worst, where = rel, name
    print(f"[offload grads] capacity={capacity} n={len(grads[False])} worst={worst:.2e} @{where}")
    assert worst <= 1e-6, f"{where}: {worst} exceeds 1e-6"


@CUDA
def test_f_the_offloaded_arm_is_exact_on_the_padded_bank_at_bf16():
    """(f) CUDA both arms: the padded bank on `cuda`, bf16, the 2e-3 bound.

    Two modules, not one: `_banks` caches graph-connected stacks of the
    parameters keyed by their `_version`, so a second forward on ONE module
    without an optimizer step in between reuses a graph the first backward
    already freed. A `deepcopy` copies every parameter bit-for-bit, so the two
    arms still differ in the switch and in nothing else.
    """
    torch.manual_seed(0)
    cfg = tiny_cfg(n_experts=4, moe_topk=2, rank=8, bf16=True)
    off = SparseMoE(cfg, device="cuda")
    on = copy.deepcopy(off)
    on.moe_offload = True
    x = torch.randn(64, 64, device="cuda")
    # The routing is DETACHED: it would otherwise carry a graph into `off`'s
    # router, and arm 2's backward through `index` would re-enter a graph arm
    # 1 already freed ("backward a second time"). This test measures the
    # EXPERT PASS's parity; the router's parity is test e's job.
    with torch.no_grad():
        weight, index = F.softmax(off.router(x), dim=-1).topk(2, dim=-1)
        weight = weight / weight.sum(dim=-1, keepdim=True)
    out, grads = {}, {}
    for flag, moe in ((False, off), (True, on)):
        t = x.detach().clone().requires_grad_(True)
        counters = Counters()
        y = moe._dispatch(t, weight, index, counters)
        y.float().pow(2).mean().backward()
        assert counters["moe_padded_bmm"] == 1, counters.line()
        assert ("moe_offloaded" in counters) is flag, counters.line()
        out[flag] = y.detach()
        grads[flag] = {
            n: p.grad.detach().clone() for n, p in moe.named_parameters() if p.grad is not None
        }
    assert set(grads[False]) == set(grads[True]) != set()
    assert (out[True] - out[False]).abs().max() <= 2e-3 * out[False].abs().max()
    assert (out[True] - out[False]).abs().max() <= 2e-3 * out[False].abs().max()
    worst, where = 0.0, ""
    for name, g in grads[False].items():
        rel = float((g - grads[True][name]).norm()) / max(float(g.norm()), 1e-12)
        if rel > worst:
            worst, where = rel, name
    print(f"[offload bf16] n={len(grads[False])} worst_rel={worst:.2e} @{where}")
    assert worst <= 2e-3, f"{where}: {worst} exceeds 2e-3"


def test_e_the_switch_is_a_config_field_reachable_through_the_presets():
    """(e) OFF by default, carried by the preset, settable, and in the snapshot."""
    assert ModelConfig().moe_offload is False
    assert load_preset("small").moe_offload is False
    assert load_run("small").model.moe_offload is False
    assert load_run("b1").model.moe_offload is False
    assert "moe_offload" in (PRESET_DIR / "small.toml").read_text()
    assert parse_preset("d_model = 64\nn_heads = 4\nhead_dim = 16\nmoe_offload = true").moe_offload
    run = load_run("small", model_overrides={"moe_offload": True})
    assert run.model.moe_offload is True
    assert "moe_offload = true" in run.snapshot_toml()


def test_a_dense_blend_counts_the_switch_as_inert(byte_tensor):
    """`moe_topk = 0` has no capacity-padded bank: COUNTED, never silent."""
    model = DormouseModel(tiny_cfg(moe_topk=0, moe_offload=True))
    ids, targets = byte_tensor()
    _, _, counters = model(ids, targets)
    assert counters["moe_offload_inert_dense"] == 2
    assert "moe_offloaded" not in counters
