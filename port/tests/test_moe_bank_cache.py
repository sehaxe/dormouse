"""The bank cache's explicit epoch — and why the `_version` key was expensive.

Two facts, measured, that this file gates:

1. **The `_version` key walks every parameter on EVERY `_banks` call.** At m400
   (E=512, 1 536 parameters) that is 2 092 us/call on this box (2026-11-05,
   CPU, Python 3.12) — 6.3 ms/step of HOST time at max_iter 3, on a step whose
   GPU is idle 87 % of the time. `moe_bank_epoch=True` replaces the walk with one
   integer compare; the bank BUILD is unchanged (one per optimizer step, which
   the trainer already forced).
2. **The build count does not move.** `invalidate_banks_all` runs once per
   trained step (`train/loop.py:1768`, `train/sft.py:1279`) and `_banks` already
   reuses its entry across the loop's iterations. One build is ~519 launches
   (the census) because a `[E, ·]` stack of 512 separate parameter tensors IS
   512 casts + a stack; the epoch does not change that, and this file says so
   instead of claiming a launch win.

What the epoch buys beyond the host time: an invalidation contract the CUDA
graph cannot freeze. `_version` froze under `opt_graph` replay and the bank
served the step-5 build for a whole run (the medbig2/b1night defect,
`tests/test_bank_invalidation.py`). The epoch is bumped by the trainer's
per-step call, by any `load_state_dict` and by a resume, and `_banks` NEVER
serves an entry whose epoch is older — a frozen epoch is stale by construction,
which the mutation gate at the bottom turns red.

OFF is every pin on record: the digests below were measured on `main` @
`28977f13` and this branch must still produce them.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from conftest import tiny_cfg
from torch.overrides import TorchFunctionMode

from dormouse.config import ModelConfig
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model.experts import SparseMoE, invalidate_banks_all

#: `main` @ 28977f13: a tiny model's OFF logits and its 27 parameter grads.
OFF_LOGITS = "f1af8f9d39852417c63e2a8776d48c48b0b054f76ff31185aeb03ce7cb54abac"
OFF_GRADS = "33a11a48d0a0e539dbd6bfb7264ca69cffe2ed90e12038859586b1d6a7280d87"


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


def _cfg(**over) -> ModelConfig:
    return tiny_cfg(n_experts=4, moe_topk=2, rank=4, d_ffn=64, **over)


def _moe(**over) -> SparseMoE:
    """`moe_kernel="batched"` so the CPU forward actually CALLS `_banks`.

    The padded/triton arms are CUDA-gated and fall back to the per-expert loop on
    CPU, which reads the parameters directly and never builds a bank; the batched
    arm's `[E, cap, ·]` bmm is plain torch and takes the same cache on any device.
    """
    torch.manual_seed(0)
    return SparseMoE(_cfg(moe_capacity=1.0, moe_kernel="batched", **over), device="cpu")


def _bank(moe: SparseMoE) -> tuple:
    return moe._banks(torch.float32)


def _move_weights(moe: SparseMoE, delta: float = 0.5) -> None:
    """A version-free weight move — the CUDA-graph-replay shape."""
    for e in moe.experts:
        e.gate_u.data.add_(delta)


# --------------------------------------------------------------------------- #
# A. OFF is clean main, and the flags are off
# --------------------------------------------------------------------------- #
def test_the_new_flags_are_off_by_default() -> None:
    cfg = ModelConfig()
    assert cfg.moe_bank_epoch is False
    assert cfg.moe_combine == "torch" and cfg.moe_combine_dtype == "fp32"
    print("[defaults] moe_bank_epoch=False moe_combine='torch' moe_combine_dtype='fp32'")


@pytest.mark.pins
def test_off_is_bit_identical_to_clean_main() -> None:
    torch.manual_seed(0)
    moe = SparseMoE(_cfg(moe_capacity=1.0), device="cpu")
    torch.manual_seed(1)
    out = moe(torch.randn(8, moe.d), Counters())
    lo = _sha(out)
    out.pow(2).mean().backward()
    blob = torch.cat(
        [
            p.grad.contiguous().view(-1)
            for _, p in sorted(moe.named_parameters())
            if p.grad is not None
        ]
    )
    assert lo == OFF_LOGITS, f"the OFF logits moved: {lo}"
    assert _sha(blob) == OFF_GRADS, "the OFF gradients moved"
    print("[pin] OFF logits/grads == main")


# --------------------------------------------------------------------------- #
# B. the key: one int, not a 1 536-parameter walk
# --------------------------------------------------------------------------- #
def test_the_epoch_key_is_one_integer_and_the_old_key_is_the_walk() -> None:
    off, on = _moe(moe_bank_epoch=False), _moe(moe_bank_epoch=True)
    _bank(off)
    _bank(on)
    old_key, new_key = off._bank_cache[0], on._bank_cache[0]
    assert len(new_key) == 4 and (on._bank_cache[3],) == (on._bank_epoch,), on._bank_cache[3]
    n_versions = sum(1 for e in off.experts for _ in e.parameters())
    assert isinstance(old_key[3], tuple) and len(old_key[3]) == n_versions, (
        "the OFF key's version tuple IS the parameter walk"
    )
    print(
        f"[key] ON is a 4-tuple + entry epoch {on._bank_cache[3]}; "
        f"OFF carries {len(old_key[3])} versions"
    )


def test_a_cache_hit_does_not_touch_the_parameters() -> None:
    """The epoch arm's hit path must not iterate `self.experts` at all."""
    moe = _moe(moe_bank_epoch=True)
    _bank(moe)

    class _NoWalk(list):
        def __iter__(self):
            raise AssertionError("the epoch key walked the expert parameters")

    modules = moe._modules.pop("experts")
    object.__setattr__(moe, "experts", _NoWalk(modules))
    try:
        again = _bank(moe)
        assert again[0] is not None
        assert moe._bank_builds == 1, "the hit must not rebuild"
    finally:
        object.__delattr__(moe, "experts")
        moe._modules["experts"] = modules
    print("[key] a cache hit reads no parameter (`_version`) at all")


# --------------------------------------------------------------------------- #
# C. epoch semantics: one build per step, zero on a second forward
# --------------------------------------------------------------------------- #
def test_two_forwards_reuse_and_a_step_rebuilds_exactly_once() -> None:
    moe = _moe(moe_bank_epoch=True)
    x = torch.randn(8, moe.d)

    c1 = Counters()
    moe(x, c1)
    assert moe._bank_builds == 1 and c1["moe_bank_builds"] == 1, c1
    c2 = Counters()
    moe(x, c2)
    assert moe._bank_builds == 1, "the second forward rebuilt"
    assert "moe_bank_builds" not in c2, f"a reuse counted a build: {c2}"

    invalidate_banks_all(moe)  # the trainer's post-`opt.step` call
    c3 = Counters()
    moe(x, c3)
    assert moe._bank_builds == 2 and c3["moe_bank_builds"] == 1, (moe._bank_builds, c3)
    invalidate_banks_all(moe)
    moe(x, Counters())
    moe(x, Counters())
    assert moe._bank_builds == 3, "a step with two forwards must build exactly once"
    print("[epoch] builds: 1 after one forward, 2 after a step, 3 after two steps")


def test_the_epoch_key_survives_a_trained_step_per_step() -> None:
    """The trainer's 3-step loop builds exactly 3 times — with the flag ON too."""
    import dataclasses as _dc

    from dormouse.config import load_run
    from dormouse.train.loop import train

    run = load_run("small")
    run.model = _dc.replace(run.model, moe_bank_epoch=True)
    run.train = _dc.replace(
        run.train,
        steps=3,
        batch=2,
        seq_len=64,
        log_every=10**9,
        ckpt_every=10**9,
        eval=False,
        warmup_tokens=0,
    )
    run.train.data = "/tmp/opencode/bank_epoch_probe.bin"
    with open(run.train.data, "wb") as f:
        f.write(bytes(range(256)) * 64)
    torch.manual_seed(0)
    model = DormouseModel(run.model)
    moe = next(m for m in model.modules() if isinstance(m, SparseMoE))
    train(model, run, device=None)
    assert moe._bank_builds == 3, f"3 trained steps, {moe._bank_builds} builds"
    print(f"[train] 3 steps -> {moe._bank_builds} builds (one per optimizer step)")


def test_a_load_bumps_the_epoch() -> None:
    moe = _moe(moe_bank_epoch=True)
    _bank(moe)
    e0 = moe._bank_epoch
    moe.load_state_dict(moe.state_dict())
    assert moe._bank_epoch == e0 + 1 and moe._bank_cache is None
    print("[load] load_state_dict bumps the epoch and drops the entry")


# --------------------------------------------------------------------------- #
# D. stale epochs: rebuilt, and the unexplained ones are LOUD
# --------------------------------------------------------------------------- #
def test_a_stale_epoch_is_never_served() -> None:
    moe = _moe(moe_bank_epoch=True)
    before = _bank(moe)[0].clone()
    stale_entry = moe._bank_cache
    _move_weights(moe)
    invalidate_banks_all(moe)
    after = _bank(moe)[0]
    assert not torch.equal(after, before), "a bumped epoch must rebuild from the moved weights"
    assert moe._bank_cache is not stale_entry, "the stale entry must be replaced"
    print("[stale] a bumped epoch rebuilds; the old entry is dropped before the rebuild")


def test_a_cache_without_an_epoch_or_from_the_future_is_loud() -> None:
    moe = _moe(moe_bank_epoch=True)
    _bank(moe)
    key, mats, scales, _ = moe._bank_cache
    moe._bank_cache = (key, mats, scales, None)
    with pytest.raises(LoudError) as e:
        _bank(moe)
    assert "Escape:" in str(e.value) and "no epoch" in str(e.value)

    moe._bank_cache = (key, mats, scales, moe._bank_epoch + 1)
    with pytest.raises(LoudError) as e:
        _bank(moe)
    assert "Escape:" in str(e.value) and "newer than its model" in str(e.value)
    print("[stale] no epoch / future epoch both refused with Escape:")


def test_a_frozen_epoch_serves_a_stale_bank_which_the_parity_gate_forbids() -> None:
    """THE bug class, pinned: the epoch bump is the ONLY thing that makes it fresh.

    `p.data.add_` moves the value without `_version` — the graph-replay shape. A
    frozen epoch serves the old bank, and the forward is then a measurement of a
    different model: the same call on a bumped epoch (or on a fresh cache) sees
    the moved weights. Reverting the bump in `invalidate_banks` turns the second
    assertion red.
    """
    moe = _moe(moe_bank_epoch=True)
    x = torch.randn(4, moe.d)
    pre = moe(x, Counters()).detach().clone()
    _move_weights(moe)

    frozen = moe(x, Counters()).detach()  # no bump: the bug
    assert torch.equal(frozen, pre), "a frozen epoch must serve the STALE bank (the bug)"

    invalidate_banks_all(moe)
    fresh = moe(x, Counters()).detach()
    assert not torch.equal(fresh, pre), "the bumped epoch must see the moved weights"
    moe._bank_cache = None
    assert torch.equal(moe(x, Counters()).detach(), fresh), "bumped == a forced rebuild"
    print("[frozen] no bump -> stale (== pre-move); bump -> the moved weights")


# --------------------------------------------------------------------------- #
# E. no host syncs added
# --------------------------------------------------------------------------- #
class _SyncCalls(TorchFunctionMode):
    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", "")
        if name in ("item", "cpu", "tolist", "numpy", "__bool__", "__float__", "__int__"):
            self.names.append(name)
        return func(*args, **(kwargs or {}))


def test_the_epoch_path_adds_no_host_sync() -> None:
    off = _moe(moe_bank_epoch=False)
    on = _moe(moe_bank_epoch=True)
    on.load_state_dict(off.state_dict())
    x = torch.randn(8, on.d)
    with _SyncCalls() as old:
        off(x, Counters())
    with _SyncCalls() as new:
        on(x, Counters())
    assert new.names == old.names, f"the epoch path read the host: {new.names} vs {old.names}"
    print(f"[no sync] both arms: {new.names} (the reference dispatch's own)")
