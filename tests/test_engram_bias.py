"""The host Engram update's bias correction (`TrainConfig.engram_bias_correction`).

WHY THIS FILE EXISTS. Burn's Nesterov look-ahead (`offload.rs:154-160`, ported
verbatim in `HostTables.momentum_update`) is an exponential sum whose total
weight depends on how often the ROW has been touched: `block_t = (1-μ^{t+1})·g`
for a constant gradient. A row touched once moves by `1-μ² = 0.0975` of its
gradient and needs ~20 touches to reach full scale. That is right for burn's
48M-row / 300k-step production run and wrong for a 600-step screen: measured
2026-10-23, `medium` + `engram_ram` scored **3.476 BPB** against **2.649** with
the same tables in VRAM, everything else identical (+0.83, iter3 reproduced).
The correction divides the applied block by its total weight on each row, so a
cold row's first touch IS the gradient and a hot row is bit-unchanged.

The gates, each named for the property it pins:

1. a cold row's first touch is the raw gradient, and the arithmetic is the
   analytic `m=(1-μ)g`, `block=(1-μ²)g`, `block/(1-μ²)=g` (not "it moved");
2. the counter is PER ROW: a hot row in the same update is bit-identical to
   burn's formula while the cold row is corrected (a global step would correct
   the hot row too, and this gate is the one that fails);
3. at `t ≥ 337` (measured) the divisor is exactly `1.0` in fp32, so four updates including
   a Sinkhorn step are bit-identical to the old formula;
4. save → load carries the counter and the tables bit for bit;
5. a `.ngram` without `counts` is REFUSED loudly (the chosen branch: a guessed
   counter changes the optimizer under the same step count, ADR-0021);
6. the update adds no host scalar read (`item`/`tolist` banned during a step);
7. the in-VRAM Engram is untouched — state_dict and logits sha256 against a
   golden captured from a clean `f6692e4` worktree. (It equals `test_prelude`'s
   `c8a2ff9` golden pair, which is an independent confirmation that the in-VRAM
   path did not move between those commits.)
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from conftest import BATCH, SEQ, tiny_cfg, tiny_model, tiny_run

from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.model.engram import (
    HOST_MOMENTUM,
    HostTables,
    enable_host_tables,
    host_state,
    load_host_state,
)
from dormouse.train import loop as train_loop
from dormouse.train import train

MU = HOST_MOMENTUM
#: Cold-row block scale of burn's formula: `μ·(1-μ)g + (1-μ)g = (1-μ²)g`.
COLD_SCALE_OLD = 1.0 - MU * MU
#: fp32's 1-2 ULP on the corrected quotient (measured 1.19e-7 relative).
FP32_SLACK = 4e-7


def _host(rows: torch.Tensor, *, corrected: bool, momentum=None, counts=None) -> HostTables:
    return HostTables(
        rows.clone(),
        None if momentum is None else momentum.clone(),
        counts=None if counts is None else counts.clone(),
        bias_correction=corrected,
    )


def test_a_cold_row_first_touch_moves_at_the_raw_gradient():
    """Touch 1: `m=(1-μ)g`, applied block `(1-μ²)g`, corrected block `g` — exactly.

    `lr=1.0` so the applied delta IS the block: the row starts at zero, so
    `row = -1.0·block` is exact and `-row` can be compared to the analytics
    without undoing a multiply. The old/new relation is asserted bitwise; the
    two analytic values are fp32, so their tolerance is the measured 1-2 ULP,
    printed on failure by `assert_close`.
    """
    g = torch.tensor([[0.5, -1.5, 2.0, -0.25]])
    rows = torch.zeros(1, 1, 4)  # [orders, slots, dim]
    uniq = torch.tensor([0])
    old = _host(rows, corrected=False)
    new = _host(rows, corrected=True)
    assert old.momentum_update(uniq, g.clone(), 1.0) == 1
    assert new.momentum_update(uniq, g.clone(), 1.0) == 1

    old_block, new_block = -old.flat()[0], -new.flat()[0]
    torch.testing.assert_close(
        old_block, COLD_SCALE_OLD * g[0], rtol=FP32_SLACK, atol=0, msg="burn's cold block"
    )
    torch.testing.assert_close(
        new_block, g[0], rtol=FP32_SLACK, atol=0, msg="the corrected block is the gradient"
    )
    # The code's own arithmetic: block_new = block_old / (1-μ^{t+1}), bitwise.
    denom = torch.tensor(1.0 - MU**2, dtype=torch.float32)
    assert torch.equal(new_block, old_block / denom), "the divisor is the whole fix"
    # The momentum is the UNcorrected (1-μ)g — the correction scales the applied
    # block, it does not rewrite the optimizer's state.
    torch.testing.assert_close(
        new.momentum.reshape(1, 4)[0], (1.0 - MU) * g[0], rtol=FP32_SLACK, atol=0
    )
    assert new.counts.dtype is torch.int32 and torch.equal(
        new.counts, torch.ones(1, dtype=torch.int32)
    )


def test_the_counter_is_per_row_and_a_hot_row_is_bit_identical():
    """One update, one cold row and one hot row: only the cold row is corrected.

    The hot row starts with momentum and `t=5000`, so `1-μ^{5001} = 1.0` in fp32
    and its applied delta must equal the old formula's BITWISE. The cold row's
    block must be the raw gradient. A correction driven by the GLOBAL step
    instead of the row's count would change the hot row and fail here.
    """
    rows = torch.zeros(2, 1, 4)
    momentum = torch.zeros(2, 1, 4)
    momentum[1] = torch.tensor([[0.3, -0.2, 0.1, 0.4]])
    counts = torch.tensor([0, 5000], dtype=torch.int32)
    g = torch.tensor([[1.0, 2.0, -3.0, 0.5], [1.0, 2.0, -3.0, 0.5]])
    uniq = torch.tensor([0, 1])
    old = _host(rows, corrected=False, momentum=momentum, counts=counts)
    new = _host(rows, corrected=True, momentum=momentum, counts=counts)
    old.momentum_update(uniq, g.clone(), 0.1)
    new.momentum_update(uniq, g.clone(), 0.1)

    assert torch.equal(new.flat()[1], old.flat()[1]), "a hot row must not move differently"
    cold_block = (rows[0, 0] - new.flat()[0]) / 0.1
    torch.testing.assert_close(
        cold_block, g[0], rtol=FP32_SLACK, atol=0, msg="the cold row is at full scale"
    )
    assert torch.equal(new.counts, torch.tensor([1, 5001], dtype=torch.int32))


def test_hot_rows_are_bit_identical_for_four_updates_including_sinkhorn():
    """At `t ≥ 337` the divisor is 1.0, so even a Sinkhorn step is unchanged.

    Four updates means the 4th is Sinkhorn-balanced; the correction is applied
    after it (a scale before Sinkhorn would be normalised away), and with every
    divisor exactly 1.0 the two tables and momentum buffers must be bit-equal at
    every step.
    """
    gs = torch.randn(3, 4, generator=torch.Generator().manual_seed(3))
    uniq = torch.tensor([0, 1, 2])
    old = _host(
        torch.zeros(3, 1, 4),
        corrected=False,
        counts=torch.full((3,), 5000, dtype=torch.int32),
    )
    new = _host(
        torch.zeros(3, 1, 4),
        corrected=True,
        counts=torch.full((3,), 5000, dtype=torch.int32),
    )
    for step in range(4):
        assert old.momentum_update(uniq, gs * (step + 1), 0.05) == 3
        assert new.momentum_update(uniq, gs * (step + 1), 0.05) == 3
        assert torch.equal(new.rows, old.rows), f"rows moved at step {step + 1}"
        assert torch.equal(new.momentum, old.momentum), f"momentum moved at step {step + 1}"
    assert torch.equal(new.counts, torch.tensor([5004] * 3, dtype=torch.int32))


def test_save_load_carries_the_counter_and_tables_bit_for_bit():
    """`host_state` → `load_host_state`: counters, rows, momentum, step — bitwise."""
    model = tiny_model()
    engram = model.loop.engram
    assert enable_host_tables(model) == 1
    ids = torch.tensor([[7, 9, 7, 9, 11, 13, 11, 13]])
    for step in range(3):
        hidden = torch.randn(
            1, ids.shape[1], model.cfg.d_model, generator=torch.Generator().manual_seed(step)
        ).requires_grad_(True)
        engram(hidden, ids).sum().backward()
        assert engram.host_step(0.1)[0] == 1
    assert int(engram.host.counts.sum()) > 0

    fresh = tiny_model()
    assert enable_host_tables(fresh) == 1
    load_host_state(fresh, host_state(model))
    src, dst = engram.host, fresh.loop.engram.host
    assert torch.equal(dst.rows, src.rows)
    assert torch.equal(dst.momentum, src.momentum)
    assert torch.equal(dst.counts, src.counts)
    assert dst.step == src.step == 3


def test_a_ngram_without_counts_is_refused_loudly():
    """The old format loads NOTHING: a guessed counter is a different optimizer.

    The refusal is the chosen branch (the alternative was a counted default).
    `counts` is the only record of each row's history; 0, 1 and the global step
    are all guesses that change `block/(1-μ^{t+1})` under the same step count,
    so the file is refused with an escape rather than silently re-meaning an
    existing checkpoint.
    """
    model = tiny_model()
    enable_host_tables(model)
    fresh = tiny_model()
    enable_host_tables(fresh)
    state = host_state(model)
    del state["engrams"][0]["counts"]
    with pytest.raises(LoudError) as e:
        load_host_state(fresh, state)
    msg = str(e.value)
    assert "counts" in msg and "bias correction" in msg and "ckpt_name" in msg
    # The counter must also be the right SHAPE and dtype: one int32 per row.
    state = host_state(model)
    state["engrams"][0]["counts"] = torch.zeros(3, dtype=torch.int32)
    with pytest.raises(LoudError) as e:
        load_host_state(fresh, state)
    assert "counts" in str(e.value)


def test_the_update_takes_no_host_scalar_read(monkeypatch):
    """`item()`/`tolist()` are banned inside an update (AGENTS §1.3).

    The host path already crosses the bus for the row grads once per step; the
    correction must add no SYNCHRONISATION on top. The update runs on CPU, so
    the only synchronisation it could add is a scalar extraction; this gate
    makes one fatal. (Sinkhorn's own `float(orig)` is pre-existing and only on
    the 4th update, so this fixture stays on step 1.)
    """
    model = tiny_model()
    engram = model.loop.engram
    enable_host_tables(model)

    def boom(*_a, **_k):
        raise AssertionError("a host scalar read entered the host-table update")

    monkeypatch.setattr(torch.Tensor, "item", boom)
    monkeypatch.setattr(torch.Tensor, "tolist", boom)
    ids = torch.tensor([[3, 5, 3, 11]])
    hidden = torch.randn(1, 4, model.cfg.d_model).requires_grad_(True)
    engram(hidden, ids).sum().backward()
    updates, rows = engram.host_step(0.1)
    assert updates == 1 and rows > 0
    assert engram.host.step == 1 and int(engram.host.counts.sum()) == rows


def test_the_trainer_wires_the_flag_and_logs_which_formula_ran(tmp_path, monkeypatch):
    """`TrainConfig.engram_bias_correction` reaches the tables and the log.

    The flag is a non-progress key, so a resume flips it LOUDLY via the snapshot
    drift check; this gate is the other half — a live run must HONOUR it. The
    host path's refusal is CUDA-only and its numerics are device-agnostic, so
    the seam that lets the CPU suite drive it is patched here.
    """
    monkeypatch.setattr(train_loop, "host_device_type", lambda _m: "cuda")
    run = tiny_run(tmp_path, steps=1, log_every=1, engram_ram=True, engram_bias_correction=False)
    model = DormouseModel(run.model)
    lines: list[str] = []
    train(model, run, log=lines.append)
    assert model.loop.engram.host.bias_correction is False
    assert any("bias_correction=off" in ln for ln in lines)
    assert model.loop.engram.host.step == 1, "the update still ran, with burn's formula"
    assert int(model.loop.engram.host.counts.sum()) > 0, "the count is kept in both arms"


def test_a_formula_flip_is_refused_by_the_snapshot_drift_check():
    """The flag is a non-progress key: a resume cannot change the optimizer.

    Two halves, both needed. (1) An OLD snapshot that lacks the field fills its
    default — the compat path stays open for every non-host run (a host
    checkpoint still hits the `.ngram` `counts` refusal, which is the gate that
    matters). (2) Flipping the field between runs on one ckpt is a diff key, so
    the resume refuses it instead of silently changing what "step 600" means.
    """
    import dataclasses

    from dormouse.config import RunConfig, TrainConfig

    run = RunConfig(source="tiny", model=tiny_cfg(), train=TrainConfig())
    old = RunConfig.from_snapshot(
        run.snapshot_toml().replace("engram_bias_correction = true\n", "")
    )
    assert old.train.engram_bias_correction is True
    flipped = RunConfig(
        source="tiny",
        model=tiny_cfg(),
        train=dataclasses.replace(TrainConfig(), engram_bias_correction=False),
    )
    assert run.diff_keys(flipped) == ["train.engram_bias_correction"]


#: Captured from a clean `f6692e4` worktree (CPU, torch 2.13.0+cu129, one
#: thread, seed 1234, `conftest.tiny_model()`): the in-VRAM path's goldens.
GOLDEN_STATE_SHA = "06eecd1d36bcf49ea3eb27247c32409487c3cf667fe48be1108e03cae6405b8a"
GOLDEN_LOGITS_SHA = "5de7160313c4db58b846061a299d4035ae84940f287276b380b012fdf846d9ba"
GOLDEN_LOSS = 6.200047969818115


def _state_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _logits_sha(logits: torch.Tensor) -> str:
    return hashlib.sha256(logits.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@pytest.mark.local_pin
def test_the_in_vram_engram_is_bit_identical_to_main():
    """The correction lives on `HostTables`; the VRAM path must not move at all.

    The golden is pre-change (`f6692e4`, a clean main worktree). Bit-identity is
    the right bar here: `_forward_host` shares `_read` with this path, so any
    accidental edit to the shared read, the table parameter or the init would
    show up as a different state_dict/logits hash, and 1 ULP of drift is a
    different network.
    """
    old = torch.get_num_threads()
    torch.set_num_threads(1)  # fp32 CPU reduction order is thread-count dependent
    try:
        torch.manual_seed(1234)
        model = tiny_model()
        model.eval()
        assert model.loop.engram.host is None, "this gate is the in-VRAM path"
        ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
        with torch.no_grad():
            logits, loss, _ = model(ids, (ids + 1) % 251)
        assert _state_sha(model) == GOLDEN_STATE_SHA
        assert _logits_sha(logits) == GOLDEN_LOGITS_SHA
        assert float(loss) == GOLDEN_LOSS
    finally:
        torch.set_num_threads(old)
