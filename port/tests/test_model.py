"""Model semantics: shapes, counters, both residual modes, and that it LEARNS."""

from __future__ import annotations

import pytest
import torch
from conftest import tiny_cfg, tiny_run

from dormouse.config import ModelConfig
from dormouse.model import DormouseModel
from dormouse.model.byteflow import compat_armed
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


def test_a_cuda_box_does_not_make_a_cpu_tensor_run_on_the_fused_arm():
    """The FLA verdict is per device TYPE, not per machine.

    Installing FLA on a CUDA box must not switch the CPU path onto a CUDA kernel:
    the tests below run on CPU tensors, and a fused arm selected by "is there a
    GPU here" would either crash or (worse) get a fallback nobody counted.
    """
    from dormouse.model.kda import fla_status

    ok, why = fla_status("cpu")
    assert not ok and "cpu" in why, why


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
    # The OFF switch for an arm is `use_*` (`engram_ram` moves the tables to
    # host RAM, it does not remove the arm).
    cfg = tiny_cfg(use_engram=False, use_kda=False)
    m = DormouseModel(cfg)
    ids, targets = byte_tensor()
    _, loss, c = m(ids, targets)
    assert torch.isfinite(loss)
    assert "engram_rows" not in c and "kda_sdpa" not in c
    assert kda_field(c) == "kda=off,n=0"


def test_lfb_moves_the_bias_and_never_carries_grad(byte_tensor):
    """LFB (arXiv:2408.15664), pinned to the integrator rule itself.

    The bias after each routed forward must equal `-coef·sign(L_cum − mean L)`
    over ALL routed rows so far — cumulative, per the paper. The A/B of
    2026-10-15 caught the per-forward variant making drops WORSE (bang-bang
    control at one-step lag oscillates the whole bank). Consecutive forwards
    can also carry exactly mirrored loads (measured: [59,60,60,77] then
    [66,69,70,51] at seed 0) — per-call checking is the only honest pin.
    """
    torch.manual_seed(0)
    cfg = tiny_cfg(moe_lb_coef=0.05)
    model = DormouseModel(cfg)
    moe = next(m for m in model.modules() if hasattr(m, "lfb_bias"))
    coef = moe.lfb_coef
    counts_seen: list[torch.Tensor] = []
    real_step = moe._lfb_step

    def spy(index, counters):
        counts_seen.append(torch.bincount(index.reshape(-1), minlength=moe.n_experts).float())
        return real_step(index, counters)

    moe._lfb_step = spy  # observe the counts the real path sees

    expected = torch.zeros(moe.n_experts)
    cum = torch.zeros(moe.n_experts)
    for step in range(3):
        generator = torch.Generator().manual_seed(7 + step)
        ids = torch.randint(0, 256, (2, 64), generator=generator)
        targets = (ids + 1) % 256
        _, _, counters = model(ids, targets)
        assert counters["moe_lfb_updates"] == 2  # this call's own dict
        for c in (counts_seen[-2], counts_seen[-1]):
            cum = cum + c
            expected -= coef * torch.sign(cum - cum.mean())
        assert torch.allclose(moe.lfb_bias.cpu(), expected, atol=1e-6), (
            f"bias {moe.lfb_bias.tolist()} != the integrator rule after step {step}"
        )
    model.eval()
    with torch.no_grad():
        ids = torch.randint(0, 256, (2, 64))
        targets = (ids + 1) % 256
        _, _, eval_counters = model(ids, targets)  # eval: no update, no counter
    assert "moe_lfb_updates" not in eval_counters
    assert torch.allclose(moe.lfb_bias.cpu(), expected), "eval moved the bias"
    # The optimizer must never see it: buffers by name, not a Parameter.
    assert "lfb_bias" not in dict(model.named_parameters())


def test_topk_dispatch_touches_only_selected_experts(byte_tensor):
    cfg = tiny_cfg(n_experts=8, moe_topk=2)
    m = DormouseModel(cfg)
    ids, _ = byte_tensor()
    _, _, c = m(ids)
    # At most n*k experts per iteration can be called; never the whole bank.
    assert c["moe_expert_calls"] <= ids.numel() * 2 * cfg.max_iter
    assert c["moe_tokens"] == ids.numel() * cfg.max_iter


def test_loss_decreases_on_fixed_bytes(tmp_path):
    """Fixed synthetic bytes, 30 steps, determinant: loss descends."""
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


def test_expert_bank_cache_serves_one_step_and_rejects_after_an_update():
    """The research op-1: the stack is identity-stable within a step, rebuilt on update.

    The naive per-forward stack copies 1.25 GB at b1 per iteration (the
    tsct-super-kernel doc named it "delete, not write"); `_version` is what
    makes a one-step cache honest.
    """
    cfg = tiny_cfg(n_experts=4, moe_topk=2)
    model = DormouseModel(cfg)
    moe = next(m for m in model.modules() if hasattr(m, "experts"))
    b1 = moe._banks(torch.float32)  # the cache populates on first use
    b2 = moe._banks(torch.float32)
    assert b1[0] is b2[0] and b1[2] is b2[2]  # identity, not a second copy
    with torch.no_grad():
        moe.experts[0].gate_u.add_(1.0)  # what an optimizer step does in place
    b3 = moe._banks(torch.float32)
    assert b3[0] is not b1[0] and float(b3[0][0].max()) > float(b1[0][0].max())


def _first_loss(model, run) -> float:
    from dormouse.train.data import open_streams

    stream, _ = open_streams(run)
    ids, targets = stream.next_batch()
    _, loss, _ = model(ids, targets)
    stream.rewind()
    return float(loss.detach())


# ---------------------------------------------------------------------------
# the byteflow arm's two shapes (`byteflow.py`'s Option A/B), counted
# ---------------------------------------------------------------------------

#: A byteflow-sized tiny model: the crate's `tiny_config` shape
#: (`net.rs:540-558`) at the port's tiny width.
BYTEFLOW = dict(
    d_model=16,
    n_heads=2,
    head_dim=8,
    d_ffn=32,
    max_seq_len=32,
    max_iter=2,
    rank=4,
    n_experts=2,
    moe_topk=0,
    engram_rows=8,
    engram_dim=4,
    bf16=False,
    shared_rank=4,
    use_byteflow=True,
    use_engram=False,
    byteflow_d_local=16,
    byteflow_d_global=16,
    byteflow_k_tokens=8,
    byteflow_e_layers=2,
    byteflow_g_layers=2,
    byteflow_heads_local=2,
    byteflow_heads_global=2,
    byteflow_w_local=8,
    byteflow_d_ff_local=32,
    byteflow_d_ff_global=32,
    byteflow_bins=8,
    byteflow_max_bytes=64,
)


def _bf_cfg(**over) -> ModelConfig:
    return tiny_cfg(**{**BYTEFLOW, **over})


def test_byteflow_channel_runs_the_loop_on_patch_latents(byte_tensor):
    """Option B: front stage → OUR loop on K latents → back stage, COUNTED.

    The evidence a reader needs is in the counters: `bf_channel=1` says the
    channel ran, `iter`/`kda_sdpa` say the loop and its own attention arm ran
    over K patches, and the byte head's gradient says the back stage trains
    (the burn defect class is an arm that runs and trains nothing).
    """
    torch.manual_seed(0)
    cfg = _bf_cfg()
    assert compat_armed(cfg)
    model = DormouseModel(cfg)
    assert model.bf is not None and model.bf_net is None
    ids, targets = byte_tensor(2, 64)
    logits, loss, c = model(ids, targets)
    assert logits.shape == (ids.numel(), cfg.vocab), "the decoder's byte logits are flat"
    assert loss is not None and torch.isfinite(loss) and loss.item() > 0
    assert c["bf_channel"] == 1 and c["forward"] == 1
    assert c["iter"] == cfg.max_iter, "the loop ran"
    assert c["kda_sdpa"] == cfg.max_iter, "with its own attention arm"
    assert "engram_rows" not in c, "the memory arm has no byte position here"
    loss.backward()
    grads = {n for n, p in model.named_parameters() if p.grad is not None}
    assert {
        "bf.net.encoder.0.attn.qkv.weight",
        "bf.net.proj.weight",
        "bf.net.upsample_w",
        "bf.net.decoder.0.attn.qkv.weight",
        "bf.net.out.weight",
        "bf.in_proj.weight",
        "bf.out_proj.weight",
        "loop.out_proj.u",
    } <= grads, sorted(grads)


def test_byteflow_standalone_is_the_net_and_nothing_else():
    """Option A: no loop arm on ⇒ the net alone, no dormant parameter tree."""
    torch.manual_seed(0)
    cfg = _bf_cfg(use_kda=False)
    assert not compat_armed(cfg)
    model = DormouseModel(cfg)
    assert model.bf is None and model.bf_net is not None
    assert not hasattr(model, "loop"), "the loop must not be built beside the net"
    ids = torch.randint(0, 256, (2, 32))
    logits, loss, c = model(ids, (ids + 1) % 256)
    assert logits.shape == (64, 256) and torch.isfinite(loss)
    assert c["bf_net"] == 1 and c["forward"] == 1 and "bf_channel" not in c
    loss.backward()
    assert model.bf_net.out.weight.grad is not None


def test_byteflow_channel_refuses_aux_weights_at_the_runtime_seam(byte_tensor):
    """`validate` refuses the aux weights with `use_byteflow`; a hand-built
    config bypasses it, so the forward holds the same line (ADR-0019)."""
    model = DormouseModel(_bf_cfg(jepa_weight=0.1))
    ids, targets = byte_tensor()
    with pytest.raises(AssertionError, match="aux"):
        model(ids, targets)
