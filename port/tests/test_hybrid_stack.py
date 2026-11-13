"""`block_stack`: the frontier's 3:1 KDA/attention mix, and uniform's parity.

Every frontier lab's small/mid model is a HYBRID — Qwen3-Next and Kimi Linear
are 3:1 Gated-DeltaNet : full attention, Jamba interleaves, MiniMax-01 mixes.
The trainer could build an all-KDA stack (`block_form="kda"`, `loop_blocks=N`,
`max_iter=N`) or an all-attention one (`block_form="transformer"`,
`n_layers=N`), never the mix, so the most-used non-vanilla mechanism in the
field had no arm. This file gates the six claims the task makes, each able to
fail on its own:

a. `block_stack="uniform"` (the default) is BIT-IDENTICAL to clean `main`
   `79331e4`: state/logits/loss/gradient/parameter sha256 captured there before
   the field existed (`/tmp/opencode/pin_hybrid.py`);
b. `"3:1"` at depth 12 builds the ordered `[K,K,K,A]×3` stack from the existing
   two block modules — asserted on the type list, not a count;
c. a 4-block hybrid trains 20 CPU steps with finite loss, and every block owns a
   parameter whose gradient is present and non-zero (the KDA-without-backward
   defect, AGENTS §3.2, is the reason this is a gate and not a smoke test);
d. a ratio that does not divide the depth refuses with `Escape:` naming the
   cycle and the depth;
e. `block_stack="hybrid"` with `max_iter>1` refuses (a hybrid is a stack of
   unique blocks, not a loop), and so does the `block_form` mismatch;
f. the two new presets' parameter counts are asserted exactly, and
   `dormouse info` prints the same number.

The pair `arch_hybrid31` (109,892,835 params) / `arch_dense890` (109,947,930)
holds the parameter budget fixed at the FLOP-matched width d=890: B1 is
ISO-FLOPs with the d=832 dense control C0 (692.2 vs 692.9 MFLOP/byte), and B2
gives all of B1's parameters to a plain transformer. See
`docs/records/reviews/2026-11-07-hybrid-stack.md`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math

import pytest
import torch
from conftest import BATCH, SEQ, tiny_cfg
from typer.testing import CliRunner

from dormouse.cli import app
from dormouse.config import (
    ModelConfig,
    TrainConfig,
    block_kinds,
    load_run,
    validate_model_train,
)
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel, param_count

#: Captured on clean `main` `79331e4` (CPU, torch 2.13.0+cu129, one thread,
#: seed 1234, `conftest.tiny_cfg()`), BEFORE `block_stack` existed:
#: `/tmp/opencode/pin_hybrid.py`.
PIN_STATE_SHA = "06eecd1d36bcf49ea3eb27247c32409487c3cf667fe48be1108e03cae6405b8a"
PIN_LOGITS_SHA = "5de7160313c4db58b846061a299d4035ae84940f287276b380b012fdf846d9ba"
PIN_LOSS = 6.200047969818115
PIN_GRAD_SHA = "b6d64da3a7b2602fa1bf8f638f2ac0ed1e60ebb9c4355eb6788d26e5b53b36f6"
PIN_PARAM_COUNT = 64_881
PIN_PARAM_SHA = "d886cd85df860ab9b00e060627b616e31393c500c723c99c122bf0dacf0ab714"

#: The exact parameter counts of the two new presets (asserted in gate f and
#: printed by `dormouse info`).
HYBRID31_PARAMS = 109_892_835
DENSE890_PARAMS = 109_947_930
#: C0 (`flagship3_dense`, picked up by `dormouse info`) — the equal-parameter
#: baseline the hybrid's FLOP saving buys width against.
C0_PARAMS = 95_035_200


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().to(torch.float32).contiguous().numpy().tobytes()).hexdigest()


def _state_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().to(torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def _grad_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, p in sorted(model.named_parameters()):
        h.update(k.encode())
        if p.grad is None:
            h.update(b"NONE")
        else:
            h.update(p.grad.detach().to(torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def _param_sha(model: torch.nn.Module) -> str:
    """Hash of `(name, shape, numel)` — the parameter SET, not the weights."""
    h = hashlib.sha256()
    for k, p in sorted(model.named_parameters()):
        h.update(k.encode())
        h.update(str(tuple(p.shape)).encode())
        h.update(str(p.numel()).encode())
    return h.hexdigest()


def _hybrid_cfg(**over) -> ModelConfig:
    """The tiny fixture as a legal hybrid stack (`3:1`, depth 4 by default)."""
    base = dict(
        block_form="kda",
        block_stack="3:1",
        max_iter=1,
        loop_blocks=1,
        n_layers=4,
        ffn_form="dense",
        use_kda=True,
        use_csa=False,
        use_mhc=False,
        use_attnres=False,
        use_engram=False,
        layerrope=False,
        use_tsct=False,
        n_experts=1,
        moe_topk=0,
        bf16=False,
    )
    base.update(over)
    return tiny_cfg(**base)


def _batch(seq: int = SEQ) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(BATCH * seq, dtype=torch.long).view(BATCH, seq) % 251
    return ids, (ids + 1) % 251


def _blocks(model: DormouseModel) -> list[str]:
    return [type(block).__name__ for block in model.hybrid_blocks]


# --- (a) uniform is bit-identical ------------------------------------------


@pytest.mark.pins
def test_uniform_is_bit_identical_to_clean_main():
    """Gate a: the default `block_stack` changes no number, gradient or key."""
    threads, precision = torch.get_num_threads(), torch.get_float32_matmul_precision()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    try:
        torch.manual_seed(1234)
        cfg = tiny_cfg()
        assert cfg.block_stack == "uniform"
        model = DormouseModel(cfg)
        assert not hasattr(model, "hybrid_blocks"), "uniform built the hybrid tree"
        x, y = _batch()
        logits, loss, _ = model(x, y)
        assert loss is not None
        loss.backward()
        assert _state_sha(model) == PIN_STATE_SHA, "uniform's state dict moved"
        assert _sha(logits) == PIN_LOGITS_SHA, "uniform's logits moved"
        assert float(loss.detach()) == PIN_LOSS
        assert _grad_sha(model) == PIN_GRAD_SHA, "uniform's gradients moved"
        assert param_count(model) == PIN_PARAM_COUNT
        assert _param_sha(model) == PIN_PARAM_SHA, "uniform's parameter set moved"

        torch.manual_seed(1234)
        explicit = DormouseModel(dataclasses.replace(cfg, block_stack="uniform"))
        same_logits, same_loss, _ = explicit(x, y)
        assert torch.equal(same_logits, logits)
        assert float(same_loss.detach()) == float(loss.detach())
        assert _state_sha(explicit) == PIN_STATE_SHA
    finally:
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision(precision)


# --- (b) the ordered stack --------------------------------------------------


def test_hybrid_builds_the_expected_block_sequence():
    """Gate b: `[K,K,K,A]×3` for `"3:1"` at depth 12 — the ORDERED type list."""
    model = DormouseModel(_hybrid_cfg(n_layers=12))
    assert _blocks(model) == ["LoopBlock", "LoopBlock", "LoopBlock", "TransformerBlock"] * 3
    assert _blocks(model).count("LoopBlock") == 9
    assert _blocks(model).count("TransformerBlock") == 3

    # Any ratio that tiles: 4:1 at depth 10 is [K,K,K,K,A]×2.
    four = DormouseModel(_hybrid_cfg(block_stack="4:1", n_layers=10))
    assert _blocks(four) == ["LoopBlock"] * 4 + ["TransformerBlock"] + (
        ["LoopBlock"] * 4 + ["TransformerBlock"]
    )


def test_block_kinds_is_the_one_definition():
    """The pure planner the model builds from: 12 at 3:1 is 9 K, 3 A."""
    kinds = block_kinds(12, (3, 1))
    assert kinds == ("kda", "kda", "kda", "attn") * 3
    assert len(kinds) == 12


# --- (c) 20 CPU steps, finite, every block learns ---------------------------


@pytest.mark.parametrize("ratio,n_layers", [("3:1", 4), ("1:1", 4)])
def test_hybrid_trains_twenty_cpu_steps_and_every_block_gets_gradient(ratio, n_layers):
    """Gate c: forward+backward+step ×20, finite, no block starved of gradient."""
    torch.manual_seed(0)
    model = DormouseModel(_hybrid_cfg(block_stack=ratio, n_layers=n_layers))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = _batch()
    losses: list[float] = []
    checked: list[int] = []
    for step in range(20):
        opt.zero_grad()
        logits, loss, _ = model(x, y)
        assert logits.shape == (BATCH * SEQ, 256)
        assert loss is not None
        assert math.isfinite(float(loss.detach())), f"step {step}: loss={loss}"
        loss.backward()
        if step == 0:
            for i, block in enumerate(model.hybrid_blocks):
                live = [
                    p.grad
                    for p in block.parameters()
                    if p.grad is not None and float(p.grad.detach().abs().sum()) > 0.0
                ]
                assert live, f"block {i} ({type(block).__name__}) received no gradient"
                checked.append(i)
        opt.step()
        losses.append(float(loss.detach()))
    assert checked == list(range(len(model.hybrid_blocks)))
    assert all(math.isfinite(v) for v in losses), losses
    assert losses[-1] < losses[0], f"20 steps did not reduce the loss: {losses[0]} -> {losses[-1]}"


# --- (d) the ratio must tile the depth -------------------------------------


def test_a_ratio_that_does_not_divide_the_depth_refuses_naming_both_numbers():
    """Gate d: 3:2 is a 5-block cycle and 12 is not a multiple of 5."""
    with pytest.raises(LoudError) as e:
        _hybrid_cfg(block_stack="3:2", n_layers=12)
    msg = str(e.value)
    assert "Escape:" in msg, msg
    assert "5" in msg, msg  # the cycle, named
    assert "12" in msg, msg  # the depth, named
    assert "3:2" in msg, msg


@pytest.mark.parametrize("ratio", ["0:1", "3:0", "3:1:1", "three", "3.0:1", ""])
def test_a_malformed_ratio_refuses(ratio):
    with pytest.raises(LoudError) as e:
        _hybrid_cfg(block_stack=ratio, n_layers=12)
    assert "Escape:" in str(e.value), str(e.value)


# --- (e) the combinations that cannot run ----------------------------------


def test_hybrid_with_max_iter_above_one_refuses():
    """Gate e: `"hybrid"` is a stack of unique blocks, not a loop."""
    with pytest.raises(LoudError) as e:
        _hybrid_cfg(block_stack="hybrid", max_iter=2)
    msg = str(e.value)
    assert "max_iter=2" in msg, msg
    assert "Escape:" in msg, msg


def test_hybrid_refuses_a_block_form_mismatch():
    with pytest.raises(LoudError) as e:
        _hybrid_cfg(block_stack="3:1", block_form="transformer")
    msg = str(e.value)
    assert "block_form" in msg and "'transformer'" in msg, msg
    assert "Escape:" in msg, msg


def test_hybrid_refuses_n_layers_zero_and_loop_blocks():
    with pytest.raises(LoudError) as e:
        _hybrid_cfg(block_stack="3:1", n_layers=0)
    assert "n_layers=0" in str(e.value) and "Escape:" in str(e.value)
    with pytest.raises(LoudError) as e:
        _hybrid_cfg(block_stack="3:1", loop_blocks=2)
    # `loop_blocks > max_iter` fires first in `__post_init__`; either refusal
    # names the inert field and the escape.
    assert "loop_blocks" in str(e.value) and "Escape:" in str(e.value)


def test_the_refusals_hold_at_the_model_seam():
    """A config mutated after construction must not build a lying model."""
    cfg = _hybrid_cfg()
    cfg.block_stack = "3:2"
    with pytest.raises(LoudError, match="Escape:"):
        DormouseModel(cfg)
    cfg = _hybrid_cfg()
    cfg.max_iter = 2
    with pytest.raises(LoudError, match="max_iter"):
        DormouseModel(cfg)


def test_hybrid_refuses_the_offload_and_recompute_arms():
    """The KDA block's host offload is not wired across the attention half."""
    cfg = _hybrid_cfg(use_engram=True)
    with pytest.raises(LoudError) as e:
        validate_model_train(cfg, TrainConfig(engram_ram=True))
    assert "engram_ram" in str(e.value) and "Escape:" in str(e.value)

    with pytest.raises(LoudError) as e:
        validate_model_train(cfg, TrainConfig(activation_offload=4.0))
    assert "activation_offload" in str(e.value) and "Escape:" in str(e.value)

    with pytest.raises(LoudError) as e:
        validate_model_train(cfg, TrainConfig(activation_recompute="full"))
    assert "activation_recompute" in str(e.value) and "Escape:" in str(e.value)

    with pytest.raises(LoudError) as e:
        validate_model_train(cfg, TrainConfig(rand_depth=True))
    assert "rand_depth" in str(e.value) and "Escape:" in str(e.value)

    # A loss term the stack's readout does not carry is refused, not inert.
    with pytest.raises(LoudError) as e:
        _hybrid_cfg(z_loss=0.01)
    assert "z_loss" in str(e.value) and "Escape:" in str(e.value)

    # The model seam a hand-built run reaches (ADR-0019).
    model = DormouseModel(_hybrid_cfg())
    with pytest.raises(LoudError, match="activation_offload"):
        model.configure_actmem(iter_checkpoint=False, activation_offload=4.0)
    with pytest.raises(LoudError, match="iter_checkpoint|activation_recompute"):
        model.configure_actmem(iter_checkpoint=True, activation_offload=0.0)
    with pytest.raises(LoudError, match="activation_recompute"):
        model.configure_actmem(
            iter_checkpoint=False, activation_offload=0.0, activation_recompute="full"
        )


def test_hybrid_refuses_segment_len_and_adaptive_depth_at_the_forward():
    model = DormouseModel(_hybrid_cfg())
    x, y = _batch()
    with pytest.raises(LoudError, match="segment_len"):
        model(x, y, segment_len=8)
    with pytest.raises(LoudError, match="depth"):
        model(x, y, depth=1)


# --- (f) the two presets price exactly -------------------------------------


def test_the_two_new_presets_have_exact_parameter_counts():
    """Gate f: the pair's counts, instantiated AND printed by `dormouse info`."""
    hybrid = load_run("arch_hybrid31")
    assert hybrid.model.block_stack == "3:1"
    assert hybrid.model.block_form == "kda"
    hmodel = DormouseModel(hybrid.model, device="meta")
    assert param_count(hmodel) == HYBRID31_PARAMS
    assert _blocks(hmodel).count("LoopBlock") == 9
    assert _blocks(hmodel).count("TransformerBlock") == 3

    dense = load_run("arch_dense890")
    assert dense.model.block_stack == "uniform"
    assert dense.model.block_form == "transformer"
    dmodel = DormouseModel(dense.model, device="meta")
    assert param_count(dmodel) == DENSE890_PARAMS
    # The pair is equal-parameter to within 0.1%: that is what makes the
    # mechanism the only variable.
    assert abs(HYBRID31_PARAMS - DENSE890_PARAMS) / DENSE890_PARAMS < 0.001

    c0 = param_count(DormouseModel(load_run("flagship3_dense").model, device="meta"))
    assert c0 == C0_PARAMS

    for name, want in (
        ("arch_hybrid31", HYBRID31_PARAMS),
        ("arch_dense890", DENSE890_PARAMS),
    ):
        result = CliRunner().invoke(app, ["info", name])
        assert result.exit_code == 0, result.output
        print(result.output)
        total = int(result.output.split(" params")[0].split()[-1].replace(",", ""))
        assert total == want, f"`info {name}` printed {total}, expected {want}"


def test_the_hybrid_preset_is_the_iso_flop_arm():
    """B1 is ISO-FLOPs with C0 (692.9) and B2 pays +14.0% for the same params.

    The ledger is the one `docs/records/research/2026-11-07-frontier-vs-vanilla.md` §2.2
    uses: 6N per byte plus the full-attention term `12·L_attn·d·s` at s=1024.
    It is a comparison bill, not a measured FLOP count.
    """
    s = 1024
    c0 = C0_PARAMS
    b1 = HYBRID31_PARAMS
    b2 = DENSE890_PARAMS
    c0_flops = 6 * c0 + 12 * 12 * 832 * s
    b1_flops = 6 * b1 + 3 * 12 * 890 * s
    b2_flops = 6 * b2 + 12 * 12 * 890 * s
    assert abs(b1_flops - c0_flops) / c0_flops < 0.005, (b1_flops, c0_flops)
    assert b2_flops / c0_flops > 1.13, (b2_flops, c0_flops)
