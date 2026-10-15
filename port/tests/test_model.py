"""Model semantics: shapes, counters, both residual modes, and that it LEARNS."""

from __future__ import annotations

import torch
from conftest import tiny_cfg, tiny_run

from dormouse.model import DormouseModel
from dormouse.train import kda_field, train


def test_forward_is_finite_and_shaped(model, tiny, byte_tensor):
    ids, targets = byte_tensor()
    logits, loss, counters = model(ids, targets)
    assert logits.shape == (ids.numel(), tiny.vocab)
    assert logits.dtype == torch.float32, "the head must run fp32 (bf16 logits NaN)"
    assert loss is not None and torch.isfinite(loss) and loss.item() > 0
    assert counters["forward"] == 1
    loss.backward()
    assert model.embed.weight.grad is not None
    # Loss with no targets is None, not a zero: "no labels" is not "loss 0".
    _, none_loss, _ = model(ids)
    assert none_loss is None


def test_every_parameter_trains_when_the_bank_is_dense(tiny, byte_tensor):
    """The gradients, on the DENSE blend (every expert gets a gradient).

    Exactly one parameter is allowed to miss a gradient, and it is named: the
    SDPA fallback does not run the delta-rule recurrence, so `beta` (the
    per-head write strength) is unused. That is the burn `8fa5d4c` defect class
    — an arm that runs and trains nothing — and the log line's
    `kda=sdpa,n=<calls>` is what tells a reader it is happening. The final
    `norm` gain is NOT in this set: burn's byte path trains on `L_Rec` alone
    and leaves it dead, the port's loss includes the final logits' CE.
    """
    model = DormouseModel(tiny_cfg(moe_topk=0))
    ids, targets = byte_tensor()
    _, loss, _ = model(ids, targets)
    loss.backward()
    grads = {n for n, p in model.named_parameters() if p.grad is not None}
    allp = {n for n, _ in model.named_parameters()}
    assert allp - grads == {"loop.attn.beta_proj.weight"}, sorted(allp - grads)


def test_every_arm_counts(model, tiny, byte_tensor):
    ids, targets = byte_tensor()
    _, _, c = model(ids, targets)
    # The loop ran its configured depth, and the COUNTED fallback says so.
    assert c["iter"] == tiny.max_iter
    assert c["kda_sdpa"] == tiny.max_iter
    # The arm counters ACCUMULATE over the loop's iterations (one read per
    # iteration is one read), which is what makes them a per-step quantity.
    assert c["engram_rows"] == ids.numel() * len(tiny.engram_orders) * tiny.max_iter
    assert c["moe_shared_forwards"] == ids.numel() * tiny.max_iter
    assert c["rezero_write"] == tiny.max_iter
    assert "attnres_write" not in c
    # The log field a reader uses to tell which attention arm produced a loss.
    assert kda_field(c) == f"kda=sdpa,n={tiny.max_iter}"


def test_attnres_mode_runs_and_replaces_rezero(byte_tensor):
    cfg = tiny_cfg(use_attnres=True)
    m = DormouseModel(cfg)
    ids, targets = byte_tensor()
    _, loss, c = m(ids, targets)
    assert torch.isfinite(loss)
    assert c["attnres_write"] == cfg.max_iter and "rezero_write" not in c
    # One pseudo-query per iteration SLOT, zero-initialised with a unit gain.
    assert len(m.loop.attnres) == cfg.max_iter
    assert torch.all(m.loop.attnres[0].query == 0) and torch.all(m.loop.attnres[0].gain == 1)


def test_bf16_forward_runs_and_stays_finite(byte_tensor):
    """bf16 is the PRESET default, so it is a covered path, not a config value.

    (Found by hand on the first end-to-end run: the Engram arm cast its input
    to bf16 and then called an fp32 `nn.Linear`, which torch refuses. The tiny
    test config runs fp32, so nothing else here would have caught it.)
    """
    cfg = tiny_cfg(bf16=True)
    model = DormouseModel(cfg)
    assert model.loop.out_proj.compute_dtype is torch.bfloat16
    # The Engram arm is an fp32 ISLAND: its "kernels" are f32-only on the burn
    # stack, so the arm is cast at its boundary, not inside (AGENTS §2.2).
    assert model.loop.engram.tables.dtype is torch.float32
    ids, targets = byte_tensor()
    _, loss, _ = model(ids, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.embed.weight.grad is not None


def test_no_engram_and_no_kda_arms_are_off_not_faked(byte_tensor):
    # `engram_ram=True` refuses in the config; the OFF switch is `use_engram`.
    cfg = tiny_cfg(use_engram=False, use_kda=False)
    m = DormouseModel(cfg)
    ids, targets = byte_tensor()
    _, loss, c = m(ids, targets)
    assert torch.isfinite(loss)
    assert "engram_rows" not in c and "kda_sdpa" not in c
    assert kda_field(c) == "kda=off,n=0"


def test_topk_dispatch_touches_only_selected_experts(byte_tensor):
    cfg = tiny_cfg(n_experts=8, moe_topk=2)
    m = DormouseModel(cfg)
    ids, _ = byte_tensor()
    _, _, c = m(ids)
    # At most n*k experts per iteration can be called; never the whole bank.
    assert c["moe_expert_calls"] <= ids.numel() * 2 * cfg.max_iter
    assert c["moe_tokens"] == ids.numel() * cfg.max_iter


def test_loss_decreases_on_fixed_bytes(tmp_path):
    """(b) 30 steps on fixed synthetic bytes at a fast lr: final < initial.

    Seeded, because an unseeded draw makes the gate itself a coin flip: at
    lr=0.02 the same 30-step run finished BELOW its start on 9 of 10 seeds and
    above it on the tenth. That is not the test being flaky, it is Muon's
    update scale — the orthogonalized update is O(1)·lr whatever the gradient
    magnitude, so a hot single lr (shared with the AdamW group, as burn's
    trainer does) destabilizes the small 1-D groups. lr=0.01 was clean on
    10/10 seeds; `torch.manual_seed` then pins the run this gate measures.
    """
    torch.manual_seed(0)
    run = tiny_run(tmp_path, steps=30, lr=0.01, log_every=100)
    model = DormouseModel(run.model)
    lines: list[str] = []
    first = _first_loss(model, run)
    report = train(model, run, log=lines.append)
    assert report["loss"] < first, f"loss did not drop: {first} -> {report['loss']}"
    assert report["nan_skipped"] == 0
    assert report["opt_steps"] == run.train.steps


def _first_loss(model, run) -> float:
    from dormouse.train.data import open_streams

    stream, _ = open_streams(run)
    ids, targets = stream.next_batch()
    _, loss, _ = model(ids, targets)
    stream.rewind()
    return float(loss.detach())
