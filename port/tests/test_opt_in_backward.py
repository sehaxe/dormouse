"""`opt_in_backward`: the bank's update at its gradient, bit-identical to the tail.

Sonar-MoE (arXiv:2611.03577) reports -19…-22 % memory and 17…-23 % step time on
OLMoE-1B-7B with BITWISE IDENTICAL results, by applying an expert tensor's
optimizer update while its backward is still running, at the moment the weight
stops being needed. This port's rule for that trick is the flat one: a
RELOCATION may move where work happens, never a number. So the gates are

A. **bit equality, not a tolerance** — the same step with the flag off and on
   hands the optimizer the SAME gradients and leaves the SAME weights, the same
   momentum state and the same loss (`torch.equal`, on a packed AND an unpacked
   bank: the two ways this model can store E experts);
B. **the bytes are computed, not quoted** — `optbwd_bytes` is exactly the
   gradient bytes the hooks dropped, summed from the groups' own shapes, and
   those gradients are gone while the tail pass's are still there;
C. **the refusals are loud** — gradient accumulation (a gradient is not final
   until the last micro-batch), clipping and AGC (both need the global norm over
   every gradient), a CUDA-graph capture (a replay runs no Python hook), the SF
   arm (its skip is host-decided), `tsct_expert_lr` (a per-step device vector
   gathered after backward), and a model with no bank at all;
D. **OFF is untouched** — no parameter carries the marker, no hook runs, no
   counter is written, so a run on record cannot move.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dormouse.config import ModelConfig, TrainConfig, validate_train
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train.optim import MixOptimizer, expert_bank_params

#: The `tests/test_tsct_packed.py` fixture, shrunk: d=64, E=4, rank=8.
BASE = dict(
    d_model=64,
    n_heads=4,
    head_dim=16,
    d_ffn=128,
    vocab=256,
    max_seq_len=32,
    max_iter=2,
    rank=8,
    n_experts=4,
    moe_topk=2,
    engram_rows=64,
    engram_dim=8,
    bf16=False,
    shared_rank=16,
    use_kda=True,
    use_engram=False,
    use_msa=False,
    use_csa=False,
    share_u=True,
    moe_kernel="batched",
    moe_bank_epoch=True,
)
IDS = torch.arange(2 * 32).reshape(2, 32) * 7 % 256
STEPS = 2
BATCHES = [
    torch.randint(0, 256, (2, 32), generator=torch.Generator().manual_seed(seed))
    for seed in (3, 11, 29)
]


def _model(packed: bool) -> DormouseModel:
    torch.manual_seed(0)
    return DormouseModel(ModelConfig(**{**BASE, "tsct_packed": packed}), device="cpu").train()


def _run(packed: bool, on: bool):
    """One arm: `STEPS` optimizer steps, every gradient the optimizer saw kept."""
    model = _model(packed)
    opt = MixOptimizer(model, lr=1e-3, wd=0.0, ns_steps=5, in_backward=on)
    if on:
        opt.muon.keep_grads = {}
    losses: list[float] = []
    grads: list[dict[str, torch.Tensor]] = []
    for i in range(STEPS):
        x = BATCHES[i % len(BATCHES)]
        opt.zero_grad()
        _logits, loss, _counters = model(x, targets=torch.roll(x, -1, dims=1))
        opt.arm_in_backward(torch.isfinite(loss.detach()).all())
        loss.backward()
        seen = dict(opt.muon.keep_grads or {})
        grads.append(
            {
                name: seen[id(p)] if id(p) in seen else p.grad
                for name, p in model.named_parameters()
                if id(p) in seen or p.grad is not None
            }
        )
        opt.step()
        losses.append(float(loss.detach()))
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    mom = {
        n: opt.muon.state[p]["momentum"].detach().clone()
        for n, p in model.named_parameters()
        if p in opt.muon.state
    }
    return losses, grads, params, mom, dict(opt.counters)


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# --------------------------------------------------------------------------- #
# A. the parity gate: a relocation, not a second optimizer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("packed", [True, False], ids=["packed", "unpacked"])
def test_on_is_bit_identical_to_off(packed: bool) -> None:
    """Flag on == flag off, bit for bit: loss, gradients, weights and momentum.

    The gradients are the point: `keep_grads` is what the HOOK consumed, so the
    comparison is "the same update ran", not "the same update would have run".
    """
    off = _run(packed, on=False)
    on = _run(packed, on=True)

    assert off[0] == on[0], f"losses moved: {off[0]} vs {on[0]}"
    for i, (a, b) in enumerate(zip(off[1], on[1], strict=True)):
        assert _same(a, b), f"step {i}: the optimizer consumed different gradients"
    assert _same(off[2], on[2]), "the weights moved"
    assert _same(off[3], on[3]), "the momentum state moved"

    # ... and the split really happened: both halves of the step served parameters.
    assert on[4].get("optbwd_params", 0) > 0, f"nothing was updated in backward: {on[4]}"
    assert on[4].get("optbwd_tail", 0) > 0, f"the tail pass served nothing: {on[4]}"
    assert "optbwd_params" not in off[4], "the OFF arm must not count the arm"


# --------------------------------------------------------------------------- #
# B. the memory: the bytes the hooks dropped, computed from the shapes
# --------------------------------------------------------------------------- #
def test_the_bytes_are_the_freed_gradients() -> None:
    """`optbwd_bytes` == the summed gradient bytes of the opted-in groups.

    Summed from the groups' shapes and the gradients' dtype, read BEFORE the
    step — the counter is asserted against the model, not against a constant, so
    a shape change cannot quietly make the number a lie. The bank's gradients are
    gone after the step and the tail pass's are not.
    """
    model = _model(packed=True)
    opt = MixOptimizer(model, lr=1e-3, wd=0.0, ns_steps=5, in_backward=True)
    opt.zero_grad()
    _logits, loss, _c = model(IDS, targets=torch.roll(IDS, -1, dims=1))
    opt.arm_in_backward(torch.isfinite(loss.detach()).all())
    loss.backward()

    ib = [g for g in opt.muon._groups if g.in_backward]
    bank = [p for g in ib for p in g.params]
    tail = [p for g in opt.muon._groups if not g.in_backward for p in g.params]
    assert bank and tail
    # The hook reports the GRADIENT's bytes; on this fixture a gradient is the
    # parameter's own dtype (`bf16=False`), so the expectation is a shape sum —
    # asserted, not assumed.
    assert all(p.dtype == torch.float32 for p in bank)
    expect = sum(p.numel() * p.element_size() for p in bank)
    opt.step()

    assert opt.counters["optbwd_bytes"] == expect
    assert opt.counters["optbwd_params"] == len(bank)
    assert all(p.grad is None for p in bank), "a bank gradient survived its own update"
    # The tail pass's gradients still exist (a parameter NO group received a
    # gradient for is still `None` in both arms — that is not this arm's doing).
    assert sum(p.grad is not None for p in tail) > 0, "the tail pass's gradients vanished too"


# --------------------------------------------------------------------------- #
# C. the refusals: every consumer that needs the gradient after backward
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs, frag",
    [
        ({"grad_accum": 2}, "post-accumulate-grad HOOK"),
        ({"grad_clip": 1.0}, "global coefficient"),
        ({"agc_clip": 0.01}, "AGC rescales"),
        ({"opt_graph": True}, "replays DEVICE ops"),
        ({"full_graph": True}, "replays DEVICE ops"),
        ({"opt": "adamw"}, "routes every parameter to AdamW"),
    ],
)
def test_the_config_refuses_what_needs_the_gradient_after_backward(kwargs, frag) -> None:
    with pytest.raises(LoudError, match=frag):
        validate_train(TrainConfig(opt_in_backward=True, **{"grad_clip": 0.0, **kwargs}))


@pytest.mark.parametrize(
    "kwargs, frag",
    [
        ({"muon_sf": True, "muon_sf_wd": 0.01}, "host-gated path"),
        ({"tsct_expert_lr": 1.0}, "per-expert scale"),
        ({"opt": "adamw"}, "routes every parameter to AdamW"),
    ],
)
def test_the_optimizer_refuses_its_own_arms(kwargs, frag) -> None:
    model = _model(packed=True)
    with pytest.raises(LoudError, match=frag):
        MixOptimizer(model, lr=1e-3, wd=0.0, in_backward=True, **kwargs)


def test_a_model_with_no_bank_is_refused() -> None:
    """The flag would be inert on a dense model — say so instead of running."""
    dense = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    with pytest.raises(LoudError, match="no expert bank"):
        MixOptimizer(dense, lr=1e-3, wd=0.0, in_backward=True)


# --------------------------------------------------------------------------- #
# D. OFF is the historical path
# --------------------------------------------------------------------------- #
def test_off_registers_nothing() -> None:
    model = _model(packed=True)
    opt = MixOptimizer(model, lr=1e-3, wd=0.0, in_backward=False)
    assert opt.in_backward is False
    assert opt.muon.in_backward is False
    assert opt.muon._ib_handles == []
    assert not any(getattr(p, "_opt_in_backward", False) for p in model.parameters())
    assert not any(g.in_backward for g in opt.muon._groups)
    # one healthy step writes no arm counter at all
    opt.zero_grad()
    _logits, loss, _c = model(IDS, targets=torch.roll(IDS, -1, dims=1))
    loss.backward()
    opt.step()
    assert dict(opt.counters) == {}


def test_the_trainer_wires_the_arm_and_moves_no_number(tmp_path, monkeypatch) -> None:
    """The whole loop: `train()` with the flag off and on, same seed, same weights.

    The unit gate above proves the optimizer's arithmetic; this one proves the
    TRAINER's half of the wiring — the kwarg reaches `MixOptimizer`, `micro_step`
    arms the hook before `backward()`, the hook really fired (the `_step_group`
    spy, or an unarmed run would pass this test vacuously in the tail pass), and
    a 3-step run lands on the same parameters bit for bit.
    """
    from conftest import tiny_run

    from dormouse.train import train
    from dormouse.train.optim import Muon

    real = Muon._step_group
    calls: list[int] = []

    def spy(self, grp):
        calls.append(id(grp))
        return real(self, grp)

    monkeypatch.setattr(Muon, "_step_group", spy)

    def one(tag: str):
        (tmp_path / tag).mkdir()
        run = tiny_run(
            tmp_path / tag,
            steps=3,
            log_every=1000,
            ckpt_every=1000,
            grad_clip=0.0,
            opt_in_backward=tag == "on",
        )
        torch.manual_seed(0)
        model = DormouseModel(run.model)
        lines: list[str] = []
        train(model, run, log=lines.append, runs_dir=str(tmp_path / "runs"))
        return {n: p.detach().clone() for n, p in model.named_parameters()}, lines

    off, _off_log = one("off")
    assert calls == [], "the OFF arm ran a hook"
    on, on_log = one("on")
    assert calls, "the trainer never armed the hook — the tail pass did all of it"
    assert _same(off, on), "the flag moved a weight through the trainer"
    assert any("opt_in_backward" in line for line in on_log), "the arm ran unannounced"


def test_the_bank_list_is_the_bank() -> None:
    """`expert_bank_params` is the expert bank, not every 2-D tensor in the model."""
    model = _model(packed=True)
    marked = set(map(id, expert_bank_params(model)))
    bank = {n for n, p in model.named_parameters() if id(p) in marked}
    assert any(n.endswith("packed_v") for n in bank), f"the packed bank was not found: {bank}"
    for n in bank:
        assert not any(k in n for k in ("attn", "router", "embed", "norm")), f"{n} is not a bank"
