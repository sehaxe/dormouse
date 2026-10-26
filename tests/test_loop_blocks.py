"""Distinct blocks INSIDE the loop (`loop_blocks`) — depth from new bodies.

One shared block iterated `T` times buys compute, not a new transformation.
This file gates the knob that runs `T` iterations over `N` DISTINCT bodies:

    h = prelude(...)
    for i in range(max_iter):
        h = blocks[i % loop_blocks](h, iteration=i)
    h = coda(...)

The gates, in the order the task states them:

1. `loop_blocks=1` is main (`962d831`) BIT FOR BIT — state-dict sha256, logits
   sha256 and the loss, captured from a clean detached worktree of that commit
   before this branch existed;
2. the parameter delta is EXACTLY one block body per extra block, on the tiny
   fixture and on `medium` through `dormouse info` (`loop_extra` row);
3. block 0 runs iterations 0/2 and block 1 runs 1/3 at `loop_blocks=2,
   max_iter=4` (spied forward hooks), and a gradient reaches BOTH bodies;
4. perturbing block 1 cannot change block 0's ITERATION-0 output (bit-equal);
5. the `medium` totals for `loop_blocks` 1..4 are the scaling number, asserted;
6. the declared refusals are LOUD with an `Escape:` — `< 1`, `> max_iter`,
   `use_byteflow`;
7. the new forward path adds no host sync;
8. `segment_len` keeps its parity contract with distinct bodies (the state
   carry is per iteration, and the body is selected per iteration).
"""

from __future__ import annotations

import hashlib
import re

import pytest
import torch
from conftest import BATCH, SEQ, tiny_model
from typer.testing import CliRunner

from dormouse.cli import app
from dormouse.config import ModelConfig, parse_preset
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel, param_count

#: Captured from a clean detached worktree of main `962d831` (CPU, torch
#: 2.13.0+cu129, one thread, seed 1234, `conftest.tiny_model()`): the
#: `loop_blocks=1` goldens. They coincide with `test_prelude.py`'s `c8a2ff9`
#: goldens, as they must: this knob at 1 changes nothing.
GOLDEN_STATE_SHA = "06eecd1d36bcf49ea3eb27247c32409487c3cf667fe48be1108e03cae6405b8a"
GOLDEN_LOGITS_SHA = "5de7160313c4db58b846061a299d4035ae84940f287276b380b012fdf846d9ba"
GOLDEN_LOSS = 6.200047969818115

#: `tiny_cfg()` prices: the base tree and ONE extra `body_only` block
#: (norm + controller + KDA + Engram + MoE + mem_dense + out_proj + residual
#: scale; NO `iter_embed`/`attnres`/`*_gate` — those are per iteration).
TINY_BASE = 64_881
TINY_BLOCK = 45_737

#: The `medium` preset: base and one extra block, from `dormouse info`.
MEDIUM_BASE = 141_910_405
MEDIUM_BLOCK = 141_561_153
MEDIUM_TOTALS = {
    1: 141_910_405,
    2: 283_471_558,
    3: 425_032_711,
    4: 566_593_864,
}
#: The driver block's own count — constant across `loop_blocks` (the
#: per-iteration slots live once, not once per body).
MEDIUM_LOOP = 141_565_253


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    return ids, (ids + 1) % 251


def _state_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _logits_sha(logits: torch.Tensor) -> str:
    return hashlib.sha256(logits.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _info(loop_blocks: int) -> tuple[int, int, int]:
    """`dormouse info medium --set loop_blocks=N`: (total, loop, loop_extra)."""
    r = CliRunner().invoke(app, ["info", "medium", "--set", f"loop_blocks={loop_blocks}"])
    assert r.exit_code == 0, r.output
    total = int(r.output.split(" params")[0].split()[-1].replace(",", ""))
    loop = re.search(r"│ loop\s+│\s*([\d,]+)", r.output)
    extra = re.search(r"loop_extra\s*│\s*([\d,]+)", r.output)
    assert loop is not None and extra is not None, r.output
    return total, int(loop.group(1).replace(",", "")), int(extra.group(1).replace(",", ""))


def test_one_block_is_bit_identical_to_main():
    """Gate 1: `loop_blocks=1` adds no module, no RNG draw, no key."""
    threads, precision = torch.get_num_threads(), torch.get_float32_matmul_precision()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    try:
        torch.manual_seed(1234)
        model = tiny_model(loop_blocks=1)
        assert len(model.loop_extra) == 0
        assert not [k for k in model.state_dict() if k.startswith("loop_extra.")]
        assert param_count(model) == TINY_BASE
        x, y = _batch()
        with torch.no_grad():
            logits, loss, _ = model(x, y)
        assert loss is not None
        assert _state_sha(model) == GOLDEN_STATE_SHA, "the one-block state dict moved"
        assert _logits_sha(logits) == GOLDEN_LOGITS_SHA, "the one-block logits moved"
        assert float(loss.detach()) == GOLDEN_LOSS, float(loss.detach())
        print(
            f"[multiblock] 1 block state={GOLDEN_STATE_SHA[:16]} "
            f"logits={GOLDEN_LOGITS_SHA[:16]} loss={GOLDEN_LOSS!r}"
        )
    finally:
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision(precision)


def test_each_extra_block_adds_exactly_one_block_body():
    """Gate 2 (tiny): the delta is the named `loop_extra` stack, no hidden copy."""
    base = param_count(tiny_model())
    assert base == TINY_BASE
    for n in (1, 2, 3):
        model = tiny_model(loop_blocks=n, max_iter=n)
        same_depth_base = param_count(tiny_model(max_iter=n))
        named = sum(p.numel() for k, p in model.named_parameters() if k.startswith("loop_extra."))
        total = param_count(model)
        print(
            f"[multiblock] tiny loop_blocks={n}: {same_depth_base:,} -> {total:,} "
            f"(+{total - same_depth_base:,})"
        )
        assert len(model.loop_extra) == n - 1
        assert total - same_depth_base == (n - 1) * TINY_BLOCK
        assert named == (n - 1) * TINY_BLOCK, "a new parameter is not in the named stack"
        # The per-iteration slots are LOOP-level: a body must not carry a
        # second `iter_embed`/`attnres`/`*_gate` set (dead rows by construction).
        for body in model.loop_extra:
            keys = body.state_dict()
            assert "iter_embed" not in keys
            assert not [
                k for k in keys if k.startswith(("attnres.", "msa_gate", "csa_gate", "loom_gate"))
            ]


def test_medium_parameter_table_is_the_scaling_number():
    """Gates 2 + 5: `dormouse info medium` prints one body's delta per block."""
    got = {}
    for n in (1, 2, 3, 4):
        total, loop, extra = _info(n)
        got[n] = total
        print(
            f"[multiblock] medium loop_blocks={n}: total={total:,} loop={loop:,} "
            f"loop_extra={extra:,} (+{total - MEDIUM_BASE:,})"
        )
        assert total == MEDIUM_TOTALS[n], (n, total)
        assert loop == MEDIUM_LOOP, "the per-iteration slots were duplicated per body"
        assert extra == (n - 1) * MEDIUM_BLOCK, (n, extra)
        assert total == MEDIUM_BASE + (n - 1) * MEDIUM_BLOCK
    assert got[2] - got[1] == MEDIUM_BLOCK
    assert got[4] - got[1] == 3 * MEDIUM_BLOCK


def test_each_block_runs_on_its_iterations_and_both_get_a_gradient():
    """Gate 3: 0/2 run body 0, 1/3 run body 1; the backward reaches both."""
    torch.manual_seed(0)
    model = tiny_model(loop_blocks=2, max_iter=4)
    seen: list[int] = []
    handles = []
    # `norm` (not `controller`): the controller's weight is read directly by
    # `mm_mixed`, so its module `forward` is never called.
    for idx, body in enumerate((model.loop, *model.loop_extra)):
        handles.append(body.norm.register_forward_hook(lambda *_, i=idx: seen.append(i)))
    x, y = _batch()
    try:
        logits, loss, counters = model(x, y)
        assert loss is not None
        loss.backward()
    finally:
        for h in handles:
            h.remove()
    assert counters["iter"] == 4, counters
    assert seen == [0, 1, 0, 1], seen
    print(f"[multiblock] iteration -> body: {seen}")

    driver = dict(model.loop.named_parameters())
    extra = dict(model.loop_extra[0].named_parameters())
    # The CPU SDPA fallback never reads KDA's `beta_proj` (it is dead on main
    # too — asserted below); every OTHER body parameter must own a finite,
    # non-zero gradient in BOTH blocks.
    dead = {k for k, p in driver.items() if p.grad is None}
    assert dead == {"attn.beta_proj.weight"}, dead
    for name, p in list(driver.items()) + list(extra.items()):
        if name == "attn.beta_proj.weight":
            assert p.grad is None and extra[name].grad is None
            continue
        assert p.grad is not None, f"{name}: no gradient"
        assert bool(torch.isfinite(p.grad).all()), f"{name}: non-finite gradient"
        peak = float(p.grad.abs().max())
        assert peak > 0.0, f"{name}: gradient is exactly zero"
    total = sum(float(p.grad.norm()) for p in extra.values() if p.grad is not None)
    print(f"[multiblock] block 1 gradient norm over its own params: {total:.4e}")
    assert total > 0.0


def test_block_one_cannot_change_block_zero_iteration_zero():
    """Gate 4: body 1 is downstream only; iteration 0 reads body 0 alone."""
    torch.manual_seed(0)
    model = tiny_model(loop_blocks=2, max_iter=4)
    captured: list[torch.Tensor] = []
    handle = model.loop.out_proj.register_forward_hook(
        lambda _m, _a, out: captured.append(out.detach().clone())
    )
    x, _ = _batch()
    try:
        with torch.no_grad():
            model(x)
        assert len(captured) == 2, "body 0 must run on iterations 0 and 2"
        first = captured[0].clone()
        with torch.no_grad():
            for p in model.loop_extra.parameters():
                p.normal_()
        captured.clear()
        with torch.no_grad():
            model(x)
        after = captured[0]
    finally:
        handle.remove()
    assert torch.equal(first, after), "body 1 leaked into body 0's iteration 0"
    print(
        "[multiblock] body-0 iteration-0 output unchanged "
        f"(max|d|={float((first - after).abs().max()):.1e})"
    )


def test_segment_carry_selects_the_same_body_per_iteration():
    """Gate 8: `segment_len` is a memory decision with distinct bodies too."""
    torch.manual_seed(0)
    model = tiny_model(loop_blocks=2, max_iter=4, moe_topk=0)
    x, y = _batch()
    with torch.no_grad():
        one, one_loss, _ = model(x, y)
        whole, whole_loss, whole_c = model(x, y, segment_len=SEQ)
        cut, cut_loss, cut_c = model(x, y, segment_len=SEQ // 2)
    assert torch.equal(one, whole), "one segment must be the one-shot forward"
    assert torch.equal(one_loss, whole_loss)
    assert whole_c["loop_segments"] == 1
    assert torch.allclose(one, cut, atol=1e-4), float((one - cut).abs().max())
    assert torch.allclose(one_loss, cut_loss, atol=1e-5)
    assert cut_c["loop_segments"] == 2
    print(
        f"[multiblock] segmented parity: 1seg exact, 2seg max|d|="
        f"{float((one - cut).abs().max()):.2e}"
    )


def test_mhc_and_attnres_are_per_block_and_still_run():
    """The two residual arms build on a body; the lane state stays the loop's."""
    torch.manual_seed(0)
    for over in ({"use_attnres": True}, {"use_mhc": True}, {"use_mhc": True, "use_attnres": True}):
        model = tiny_model(loop_blocks=2, max_iter=4, **over)
        assert (model.loop.mhc is not None) == bool(over.get("use_mhc"))
        assert all(body.mhc is not None for body in model.loop_extra) == bool(over.get("use_mhc"))
        x, y = _batch()
        _, loss, _ = model(x, y)
        assert loss is not None and bool(torch.isfinite(loss))
        loss.backward()
        for body in model.loop_extra:
            grads = [p.grad for p in body.parameters() if p.grad is not None]
            assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
            assert max(float(g.abs().max()) for g in grads) > 0.0
        print(f"[multiblock] {over}: loss={float(loss.detach()):.4f}")


def test_the_declared_refusals_are_loud():
    """Gate 6: each conflict names the key, the reason and an `Escape:`."""
    cases = [
        ("loop_blocks = 0\n", "loop_blocks=0"),
        ("loop_blocks = 5\n", "loop_blocks=5 > max_iter=4"),
        ("loop_blocks = 2\nmax_iter = 1\n", "loop_blocks=2 > max_iter=1"),
        ("loop_blocks = 2\nuse_byteflow = true\n", "loop_blocks=2 with use_byteflow=true"),
    ]
    for text, needle in cases:
        with pytest.raises(LoudError) as e:
            parse_preset(text)
        msg = str(e.value)
        assert needle in msg and "Escape:" in msg, msg
        print(f"[multiblock] refused {text.splitlines()[0]}: {msg.splitlines()[0]}")
    # The runtime seam: a hand-built model never calls `validate`.
    with pytest.raises(LoudError) as e:
        DormouseModel(ModelConfig(loop_blocks=2, max_iter=2, use_byteflow=True))
    assert "loop_blocks=2" in str(e.value) and "Escape:" in str(e.value)
    # `ModelConfig`'s own bounds fire without `validate` too.
    with pytest.raises(LoudError):
        ModelConfig(loop_blocks=0)
    with pytest.raises(LoudError):
        ModelConfig(loop_blocks=3, max_iter=2)


def test_the_new_forward_adds_no_host_sync(monkeypatch):
    """Gate 7: the multi-block path reads no device scalar (tensor ops only)."""

    def boom(*_a, **_k):
        raise AssertionError("host sync added by the multi-block forward")

    for name in ("item", "tolist", "__int__", "__float__", "__bool__"):
        monkeypatch.setattr(torch.Tensor, name, boom, raising=True)
    # No targets: the training path's existing `assert torch.isfinite(loss)` is
    # a pre-existing read outside this knob. `moe_topk=0` avoids the dense
    # reference dispatch's own `tolist` (the same scoping `test_prelude.py` uses).
    model = tiny_model(loop_blocks=2, max_iter=4, moe_topk=0)
    counters = Counters()
    x, _ = _batch()
    with torch.no_grad():
        model(x, counters=counters)
    assert counters["iter"] == 4, counters
