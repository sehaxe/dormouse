"""The LR-band stabilizers: AGC, AdamW's eps, Gemma-2's soft-cap, group norms.

Each stabilizer is OFF by default and must not move a default run's bits; the
three filters and the diagnostic get one gate each, plus the pinned-bit gate and
the snapshot gate. Every gate has been proved RED by scratch-reverting the code
it protects (the revert evidence is in the task report, not in this file).

The pinned numbers were captured from a clean detached `main` worktree
(`06109ae`) BEFORE the edit: 5 steps of `tiny_run(steps=5, log_every=5,
warmup_tokens=0)`, seed 0, sha256 over `state_dict()` (keys sorted) and over the
final logits of the fixed `[2, 64]` batch. `HEAD_STATE_SHA` is also the golden
`tests/test_nan_firewall.py` pins, so the two harnesses agree on what
"bit-identical" means.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import math

import pytest
import torch
import torch.nn.functional as F
from conftest import TINY, tiny_cfg, tiny_model, tiny_run

import dormouse.train.loop as loop_mod
from dormouse.config import ModelConfig, RunConfig, TrainConfig
from dormouse.instrument import Counters, LoudError
from dormouse.losses import fixed_cross_entropy, logit_softcap
from dormouse.model import DormouseModel
from dormouse.train.loop import train
from dormouse.train.optim import AGC_EPS, MixOptimizer, agc_clip, is_muon_param

#: Clean-main golden: `state_dict` after 5 tiny steps (the same value
#: `tests/test_nan_firewall.py::HEAD_SHA` pins). Re-pinned 2026-11-11 with
#: Muon's small-side Newton–Schulz; the forward-only logits below moved with it
#: because the parameters the forward sees are the stepped ones.
HEAD_STATE_SHA = "62cf3843ac570a67c089ef8a7a0f19529ac5e45cbd9e7ef0f71c1bf98a6bf988"
#: Clean-main golden: final logits of `(arange(128)*7+3) % 256`, `[2, 64]`.
HEAD_LOGITS_SHA = "3df478c82b1605ff0772cb26791aa899625d9eb3f1872c435c1c7e247467b3b1"

GOLDEN_STEPS = 5
SEQ, BATCH = 64, 2


def sha_state(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sha_logits(model: torch.nn.Module) -> str:
    x = (torch.arange(BATCH * SEQ).reshape(BATCH, SEQ) * 7 + 3) % 256
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(x.long(), None)
    return hashlib.sha256(logits.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _tiny_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -F.log_softmax(logits, dim=-1).gather(1, targets.reshape(-1, 1)).mean()


# --- gate 1: AGC -----------------------------------------------------------------


def test_agc_scales_each_gradient_by_the_published_ratio(monkeypatch):
    """One `_foreach_norm` per norm list, the exact `λ·‖p‖/max(‖g‖,ε)` ratio,
    and the specified variant's zero-init consequence (scale 0)."""
    p1 = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3) / 7 - 0.5)
    p1.grad = torch.linspace(-1.0, 2.0, 12).reshape(4, 3)
    p2 = torch.nn.Parameter(torch.zeros(5))  # zero-init: the stated variant's hazard
    p2.grad = torch.ones(5)
    p3 = torch.nn.Parameter(torch.ones(3))
    p3.grad = None
    orig1, orig2 = p1.grad.clone(), p2.grad.clone()

    calls: list[int] = []
    real = torch._foreach_norm

    def spy(tensors, p=2, *a, **k):
        calls.append(len(tensors))
        return real(tensors, p, *a, **k)

    monkeypatch.setattr(torch, "_foreach_norm", spy)
    agc_clip([p1, p2, p3], 0.01)

    # One launch for the norms of the live params, one for their gradients.
    assert calls == [2, 2], calls
    scale = min(1.0, 0.01 * float(p1.detach().norm()) / max(float(orig1.norm()), AGC_EPS))
    assert torch.allclose(p1.grad, orig1 * scale, rtol=0, atol=1e-7), (p1.grad, orig1 * scale)
    # `max(‖g‖, 1e-3)` is on the gradient; ‖p‖=0 is NOT floored in this
    # variant, so a zero-initialised tensor is scaled to exactly zero.
    assert torch.equal(p2.grad, torch.zeros_like(orig2)), "the specified variant has no ‖p‖ floor"
    assert p3.grad is None, "a parameter with no gradient must not grow one"


def test_agc_runs_before_the_global_clip_and_once_per_accumulated_step(monkeypatch, tmp_path):
    """`opt_stage` order is AGC then `clip_grads`, and `grad_accum` gets ONE pass."""
    run = tiny_run(tmp_path, steps=1, log_every=1, agc_clip=0.01, grad_accum=2)
    model = DormouseModel(run.model)
    calls: list[str] = []
    real_agc, real_clip = loop_mod.agc_clip, loop_mod.clip_grads

    def spy_agc(params, clipping):
        calls.append("agc")
        return real_agc(params, clipping)

    def spy_clip(params, max_norm):
        calls.append("clip")
        return real_clip(params, max_norm)

    monkeypatch.setattr(loop_mod, "agc_clip", spy_agc)
    monkeypatch.setattr(loop_mod, "clip_grads", spy_clip)
    train(model, run, log=lambda *_: None)
    assert calls == ["agc", "clip"], (
        f"AGC must be one pass over the ACCUMULATED gradient, before the global clip: {calls}"
    )

    # `agc_clip=0.0` is OFF: the stage is the pre-flag clip alone.
    calls.clear()
    run0 = tiny_run(tmp_path, steps=1, log_every=1, agc_clip=0.0, grad_accum=2)
    train(DormouseModel(run0.model), run0, log=lambda *_: None)
    assert calls == ["clip"], calls


def test_agc_does_not_sync_or_leave_the_device(monkeypatch, tmp_path):
    """No `.item()` / `.cpu()` in AGC: the scale is a device tensor."""
    p = torch.nn.Parameter(torch.ones(2, 2))
    p.grad = torch.ones(2, 2)
    for name in ("item", "tolist", "cpu"):
        real = getattr(torch.Tensor, name)

        def boom(self, *a, _real=real, _name=name, **k):
            raise AssertionError(f"AGC called Tensor.{_name}")

        monkeypatch.setattr(torch.Tensor, name, boom)
    agc_clip([p], 0.01)
    assert p.grad is not None and torch.isfinite(p.grad).all()


# --- gate 2: AdamW eps -----------------------------------------------------------


def test_adam_eps_reaches_torch_adamw(monkeypatch, tmp_path):
    """The config field is the eps of the torch.optim.AdamW the trainer builds."""
    model = tiny_model()
    opt = MixOptimizer(model, lr=0.01, wd=0.0, eps=1e-6)
    assert opt.adamw.defaults["eps"] == 1e-6
    assert all(g["eps"] == 1e-6 for g in opt.adamw.param_groups)

    seen: dict = {}
    real = loop_mod.MixOptimizer

    class Spy(real):
        def __init__(self, *a, **kw):
            seen.update(kw)
            super().__init__(*a, **kw)

    monkeypatch.setattr(loop_mod, "MixOptimizer", Spy)
    run = tiny_run(tmp_path, steps=1, log_every=1, adam_eps=1e-6)
    train(DormouseModel(run.model), run, log=lambda *_: None)
    assert seen["eps"] == 1e-6, f"adam_eps did not reach the optimizer: {seen}"

    run_default = tiny_run(tmp_path, steps=1, log_every=1)
    seen.clear()
    train(DormouseModel(run_default.model), run_default, log=lambda *_: None)
    assert seen["eps"] == 1e-8, "the default must stay torch's default"


# --- gate 3: Gemma-2 logit soft-cap ---------------------------------------------


def test_softcap_caps_both_ce_sites_and_the_eval_objective(tmp_path):
    """Per-iteration CE, final CE and the pure eval all score capped logits."""
    run = tiny_run(tmp_path, steps=1, log_every=1, warmup_tokens=0)
    cap = 15.0
    torch.manual_seed(0)
    model = DormouseModel(dataclasses.replace(run.model, logit_softcap=cap))
    torch.manual_seed(0)
    raw_model = DormouseModel(run.model)  # same init, cap off

    x = torch.tensor([[i % 256 for i in range(SEQ)] for _ in range(BATCH)], dtype=torch.long)
    y = torch.roll(x, shifts=-1, dims=1)
    heads: list[torch.Tensor] = []
    handle = model.lm_head.register_forward_hook(lambda _m, _i, out: heads.append(out.detach()))
    try:
        logits, loss, _ = model(x, y)
    finally:
        handle.remove()
    t = model.cfg.max_iter
    assert len(heads) == t + 1, f"{len(heads)} head calls, expected {t} per-iter + 1 final"

    capped = [cap * torch.tanh(h / cap) for h in heads]
    assert torch.equal(logits, capped[-1]), "the returned final logits are not the capped ones"
    assert torch.all(logits.abs() < cap), "the cap did not bound the final logits"
    with torch.no_grad():
        raw_logits, _, _ = raw_model(x, None)
    assert torch.equal(heads[-1], raw_logits), "the cap must not move the head's raw output"

    per = sum(_tiny_ce(capped[i], y) for i in range(t))
    expected = (per + _tiny_ce(capped[-1], y)) / (t + 1)
    assert torch.allclose(loss, expected, rtol=1e-6, atol=1e-7), (loss, expected)
    uncapped = (sum(_tiny_ce(heads[i], y) for i in range(t)) + _tiny_ce(heads[-1], y)) / (t + 1)
    assert not torch.allclose(loss, uncapped, rtol=1e-4), (
        "the loss equals the uncapped objective: a CE site is not capped"
    )

    # The eval objective is the same two capped sites (`pure=True`).
    model.eval()
    with torch.no_grad():
        _, ev_loss, _ = model(x, y, pure=True)
    assert torch.allclose(ev_loss, expected, rtol=1e-6, atol=1e-7), (ev_loss, expected)

    # And the cap composes with `ce_fixed`: the fixed loss reads capped logits.
    model.train()
    with torch.no_grad():
        model.zero_grad(set_to_none=True)
        _, fixed_loss, _ = model(x, y, ce_fixed=True)
    fixed_expected = (
        sum(fixed_cross_entropy(capped[i], y.reshape(-1), reduction="mean") for i in range(t))
        + fixed_cross_entropy(capped[-1], y.reshape(-1), reduction="mean")
    ) / (t + 1)
    assert torch.allclose(fixed_loss, fixed_expected, rtol=1e-6, atol=1e-7), (
        fixed_loss,
        fixed_expected,
    )


def test_softcap_refuses_byteflow_and_non_positive_caps():
    with pytest.raises(LoudError, match="Escape:"):
        ModelConfig(**TINY, use_byteflow=True, logit_softcap=15.0)
    with pytest.raises(LoudError, match="Escape:"):
        ModelConfig(**TINY, mtp_weight=1.0, mtp_k=1, logit_softcap=15.0)
    with pytest.raises(LoudError, match="Escape:"):
        ModelConfig(**TINY, logit_softcap=-1.0)
    with pytest.raises(LoudError, match="Escape:"):
        logit_softcap(torch.zeros(2, 2), 0.0)
    off = torch.randn(3, 5)
    assert torch.equal(logit_softcap(off, 15.0), 15.0 * torch.tanh(off / 15.0))


# --- gate 4: per-group gradient norms -------------------------------------------


def test_group_grad_norms_are_counted_once_per_log_window(monkeypatch, tmp_path):
    """Calls == number of log windows (never per step), and the values are the
    group norms of the gradients the optimizer consumed."""
    calls: list[int] = []
    real = loop_mod.MixOptimizer.group_grad_norms

    def spy(self):
        calls.append(1)
        return real(self)

    monkeypatch.setattr(loop_mod.MixOptimizer, "group_grad_norms", spy)
    run = tiny_run(tmp_path, steps=2, log_every=2, warmup_tokens=0)
    model = DormouseModel(run.model)
    report = train(model, run, log=lambda *_: None)
    assert len(calls) == 1, f"{len(calls)} norm windows for 2 steps at log_every=2"
    counters = report["counters"]
    groups: dict[str, list[torch.Tensor]] = {"muon": [], "adamw": [], "norms_biases": []}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if is_muon_param(name, p):
            groups["muon"].append(p.grad)
        elif p.ndim <= 1:
            groups["norms_biases"].append(p.grad)
        else:
            groups["adamw"].append(p.grad)
    for name, grads in groups.items():
        want = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
        got = counters[f"g_norm_{name}"]
        assert isinstance(got, float), f"g_norm_{name} must stay a float, got {type(got)}"
        assert math.isclose(got, float(want), rel_tol=1e-6), (name, got, float(want))

    # 5 steps / log_every=2 -> 2 windows, never 5.
    calls.clear()
    run2 = tiny_run(tmp_path, steps=5, log_every=2, warmup_tokens=0)
    train(DormouseModel(run2.model), run2, log=lambda *_: None)
    assert len(calls) == 2, f"{len(calls)} norm windows for 5 steps at log_every=2"


# --- the pinned-bit gate ---------------------------------------------------------


@pytest.mark.pins
def test_off_defaults_are_bit_identical_to_the_pinned_main_run(tmp_path):
    """`agc_clip=0`, `adam_eps=1e-8`, `logit_softcap=0` move no bit."""
    run = tiny_run(tmp_path, steps=GOLDEN_STEPS, log_every=5, warmup_tokens=0)
    torch.manual_seed(0)
    model = DormouseModel(run.model)
    train(model, run, log=lambda *_: None)
    assert sha_state(model) == HEAD_STATE_SHA, "the default step moved state_dict's bits"
    assert sha_logits(model) == HEAD_LOGITS_SHA, "the default logits moved"

    # Explicitly spelling the three defaults is the same program as omitting them.
    run2 = tiny_run(
        tmp_path,
        steps=GOLDEN_STEPS,
        log_every=5,
        warmup_tokens=0,
        agc_clip=0.0,
        adam_eps=1e-8,
    )
    run2 = dataclasses.replace(run2, model=dataclasses.replace(run2.model, logit_softcap=0.0))
    torch.manual_seed(0)
    model2 = DormouseModel(run2.model)
    train(model2, run2, log=lambda *_: None)
    assert sha_state(model2) == HEAD_STATE_SHA
    assert sha_logits(model2) == HEAD_LOGITS_SHA


# --- the snapshot gate -----------------------------------------------------------


def test_snapshot_pins_every_behavioural_field(tmp_path):
    """The three fields travel in the snapshot and drift-check on a resume."""
    base = tiny_run(tmp_path)
    changed = dataclasses.replace(
        base,
        model=dataclasses.replace(base.model, logit_softcap=15.0),
        train=dataclasses.replace(base.train, agc_clip=0.01, adam_eps=1e-6),
    )
    drift = changed.diff_keys(base)
    for key in ("model.logit_softcap", "train.agc_clip", "train.adam_eps"):
        assert key in drift, f"{key} is not in the checked set: {drift}"
    back = RunConfig.from_snapshot(changed.snapshot_toml())
    assert back.model.logit_softcap == 15.0
    assert back.train.agc_clip == 0.01
    assert back.train.adam_eps == 1e-6
    assert back.diff_keys(changed) == []


# --- the 0-sync gate (the `test_nan_firewall.py` pattern) -----------------------


def test_the_stabilizers_take_no_host_read_in_the_step(monkeypatch, tmp_path):
    """AGC + softcap + the counter cadence: no host read inside a step.

    Same patch set as `tests/test_nan_firewall.py::test_the_training_step_takes
    _no_host_read`: the ONLY exemptions are torch.optim's CPU step counter, the
    declared `Counters.drain` cadence and `loop._host_scalar` (the log line).
    `log_every=10**9` keeps the cadence out of the three steps.
    """
    names = ("item", "tolist", "__bool__", "__float__", "__int__", "cpu")
    real = {n: getattr(torch.Tensor, n) for n in names}

    def boom(name):
        def check(self, *a, **k):
            caller = inspect.currentframe().f_back
            module = caller.f_globals.get("__name__", "")
            if name == "item" and module.startswith("torch.optim"):
                return real[name](self, *a, **k)
            raise AssertionError(
                f"host read `{name}` during the step, called from {module} "
                f"(shape {tuple(self.shape)})"
            )

        return check

    def install() -> None:
        for n in names:
            setattr(torch.Tensor, n, boom(n))

    def restore() -> None:
        for n in names:
            setattr(torch.Tensor, n, real[n])

    real_drain = Counters.drain

    def drain(self):
        restore()
        try:
            return real_drain(self)
        finally:
            install()

    real_read = loop_mod._host_scalar

    def read(value):
        restore()
        try:
            return real_read(value)
        finally:
            install()

    monkeypatch.setattr(Counters, "drain", drain)
    monkeypatch.setattr(loop_mod, "_host_scalar", read)
    run = tiny_run(
        tmp_path,
        steps=3,
        log_every=10**9,
        agc_clip=0.01,
        adam_eps=1e-6,
        warmup_tokens=0,
    )
    run = dataclasses.replace(
        run, model=dataclasses.replace(run.model, logit_softcap=15.0, moe_topk=0)
    )
    model = DormouseModel(run.model)
    install()
    try:
        train(model, run, log=lambda *_: None)
    finally:
        restore()


# --- the config refusals --------------------------------------------------------


def test_agc_and_eps_refuse_impossible_values():
    from dormouse.config import validate_train

    with pytest.raises(LoudError, match="Escape:"):
        validate_train(TrainConfig(agc_clip=-0.1))
    with pytest.raises(LoudError, match="Escape:"):
        validate_train(TrainConfig(adam_eps=0.0))
    with pytest.raises(LoudError, match="Escape:"):
        validate_train(TrainConfig(adam_eps=float("nan")))


def test_a_hand_built_model_holds_the_softcap_refusal():
    """A hand-built `ModelConfig` skips `__post_init__` only in no known path,
    but the model seam must still refuse the impossible pair."""
    cfg = tiny_cfg(use_byteflow=True)
    # Bypass `ModelConfig.__post_init__` (a hand-built object does): the model
    # seam must hold the same refusal on its own.
    cfg.logit_softcap = 15.0
    with pytest.raises(LoudError, match="Escape:"):
        DormouseModel(cfg)
