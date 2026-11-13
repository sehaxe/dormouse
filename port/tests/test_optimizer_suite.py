"""The optimizer suite: every training knob against a numeric fingerprint.

The 2026-11-07 Muon audit found one knob (`muon_lr_mult`) that the trainer
overwrote every step, so a 3000-step A/B measured it OFF. This file is the
instrument that asks the same question of the remaining knobs — does the value
the config promises reach the arithmetic, and is the arithmetic the documented
formula? Each gate is numerical and names its own revert, so a scratch mutation
reddens it (the four the brief names: drop bias correction, apply the clip after
the step, ignore weight decay, restart warmup on resume).

CPU only: every number is a semantic fingerprint, not a card measurement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from pathlib import Path

import pytest
import torch
from conftest import SEQ, tiny_cfg, tiny_run
from torch import nn

from dormouse.config import ModelConfig, RunConfig, TrainConfig, validate_train
from dormouse.instrument import LoudError
from dormouse.losses import logit_softcap
from dormouse.model import DormouseModel
from dormouse.post.rl_train import rl_train
from dormouse.train import train
from dormouse.train.loop import _ema_update, _shadow_state, load_model_half, save_ckpt
from dormouse.train.lr import wsd_lr
from dormouse.train.optim import MixOptimizer, Muon, agc_clip, clip_grads
from dormouse.train.sft import build_sft_corpus, refuse_unwired, sft_train

SRC = Path(__file__).resolve().parents[1] / "src" / "dormouse"
B1, B2 = 0.9, 0.95


# --------------------------------------------------------------------------
# references (implemented here, not imported from the code they judge)
# --------------------------------------------------------------------------


def _adamw_ref(p, m, v, g, lr, wd, eps, t):
    """torch.optim.AdamW's step, written out: decay, moments, bias correction, eps."""
    p = p * torch.as_tensor(1.0 - lr * wd, dtype=p.dtype)
    m = B1 * m + (1.0 - B1) * g
    v = B2 * v + (1.0 - B2) * g * g
    bc1 = 1.0 - B1**t
    bc2_sqrt = math.sqrt(1.0 - B2**t)
    denom = v.sqrt() / bc2_sqrt + eps
    return p - (lr / bc1) * m / denom, m, v


def _colrow_ref(x: torch.Tensor) -> torch.Tensor:
    """`Norm_d(ColRow)`: column norms, then row norms (`optim.colrow_normalize`)."""
    x = x / (x.norm(dim=0, keepdim=True) + 1e-7)
    return x / (x.norm(dim=1, keepdim=True) + 1e-7)


def _closed_form_lr(step, *, total, peak, warm, decay, final):
    """The WSD contract: token warmup, hold, then a decay landing on `final·peak`."""
    if warm and step < warm:
        return peak * (step + 1) / warm
    decay_start = max(total - decay, warm)
    if decay == 0 or step < decay_start:
        return peak
    frac = min(1.0, (step - decay_start + 1) / decay)
    return peak * (1.0 - frac * (1.0 - final))


class Holder(nn.Module):
    """A model with one Muon-routed 2-D weight and one AdamW-routed vector."""

    def __init__(self, shape: tuple[int, int] = (8, 8)):
        super().__init__()
        self.w = nn.Parameter(torch.randn(shape))
        self.bias = nn.Parameter(torch.zeros(shape[1]))


def _state_sha(model: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# 1. AdamW
# --------------------------------------------------------------------------


def test_adamw_first_step_is_lr_and_steps_1_2_carry_bias_correction():
    """Step 1 magnitude is `lr`, and with a CONSTANT gradient so is every step.

    Revert: drop the `1/bc1` or the `sqrt(bc2)` division -> step 1 is
    `(1-beta1)·lr = 0.1·lr` and step 2 is `0.19·lr`; this fails.
    """
    model = Holder()
    model.w.data.zero_()
    model.bias.data.zero_()
    opt = MixOptimizer(model, lr=1e-3, wd=0.0, opt="adamw")
    g = torch.tensor([1.0, -1.0] + [0.0] * (model.bias.numel() - 2))
    for t in (1, 2):
        model.bias.grad = g.clone()
        before = model.bias.detach().clone()
        opt.step()
        delta = model.bias.detach() - before
        assert torch.allclose(delta, -1e-3 * g.sign(), rtol=0, atol=2e-9), (
            f"step {t}: delta {delta.tolist()} is not lr*sign(g)"
        )


def test_adamw_matches_the_written_reference_over_two_steps():
    """A hand-written AdamW on arbitrary gradients, steps 1 and 2.

    Revert: drop bias correction, move eps outside the bias-corrected sqrt, or
    ignore wd -> the trajectory leaves the reference.
    """
    torch.manual_seed(0)
    model = Holder()
    opt = MixOptimizer(model, lr=2e-3, wd=0.1, opt="adamw", eps=1e-8)
    p = model.w.detach().clone()
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    for t in (1, 2):
        g = torch.randn_like(p)
        model.w.grad = g.clone()
        p, m, v = _adamw_ref(p, m, v, g, 2e-3, 0.1, 1e-8, t)
        opt.step()
        assert torch.allclose(model.w.detach(), p, rtol=1e-5, atol=1e-7), (
            f"step {t}: measured {model.w.detach().flatten()[:3].tolist()} vs "
            f"reference {p.flatten()[:3].tolist()}"
        )


def test_adamw_fused_writer_is_bit_identical_to_torch_adamw():
    """The fused writer reproduces `torch.optim.AdamW` bit for bit for 5 steps.

    Revert: change any of decay/moment/bias-correction/eps order -> the two
    trajectories stop being `torch.equal`.
    """

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.randn(4, 3))
            self.b = nn.Parameter(torch.randn(7))

    torch.manual_seed(1)
    ref_model, got_model = M(), M()
    got_model.load_state_dict(ref_model.state_dict())
    ref = torch.optim.AdamW(
        ref_model.parameters(), lr=1e-3, weight_decay=0.1, betas=(B1, B2), eps=1e-8
    )
    got = MixOptimizer(got_model, lr=1e-3, wd=0.1, betas=(B1, B2), eps=1e-8, opt="adamw")
    for t in range(1, 6):
        for p in ref_model.parameters():
            p.grad = torch.full_like(p, 1e-3) * (t + 1)
        for p in got_model.parameters():
            p.grad = torch.full_like(p, 1e-3) * (t + 1)
        ref.step()
        got.step()
        for (name, a), b in zip(ref_model.named_parameters(), got_model.parameters(), strict=True):
            assert torch.equal(a, b), f"step {t} {name}: max diff {(a - b).abs().max().item()}"


def test_adamw_weight_decay_is_decoupled():
    """Zero gradient still decays by `(1-lr*wd)` per step; wd is not in m/v.

    Revert: feed `wd*p` into the gradient (coupled decay) or ignore wd -> the
    zero-gradient trajectory is not the geometric factor.
    """
    model = Holder()
    model.w.data.fill_(1.0)
    model.bias.data.fill_(2.0)
    opt = MixOptimizer(model, lr=1e-2, wd=0.1, opt="adamw")
    for _ in range(10):
        model.bias.grad = torch.zeros_like(model.bias)
        opt.step()
    ref = torch.tensor([2.0])
    factor = torch.as_tensor(1.0 - 1e-2 * 0.1, dtype=torch.float32)
    for _ in range(10):
        ref = ref * factor
    assert torch.allclose(model.bias.detach(), ref, rtol=1e-6, atol=1e-7), (
        f"wd factor {model.bias[0].item():.9f} vs {ref[0].item():.9f}"
    )
    assert model.bias[0].item() < 1.999, "the decay did not move the parameter"


def test_adamw_eps_is_inside_the_bias_corrected_denominator():
    """A tiny gradient makes eps dominate; its placement is the fingerprint.

    Revert: add eps before dividing by `sqrt(bc2)` (or outside the denominator)
    -> the step differs by the bias-correction factor at t=1.
    """
    model = Holder()
    model.bias.data.zero_()
    opt = MixOptimizer(model, lr=1.0, wd=0.0, opt="adamw", eps=1e-6)
    g = torch.tensor([1e-6, 2e-6] + [0.0] * (model.bias.numel() - 2))
    model.bias.grad = g.clone()
    opt.step()
    m = (1.0 - B1) * g
    v = (1.0 - B2) * g * g
    correct = -(1.0 / (1.0 - B1)) * m / (v.sqrt() / math.sqrt(1.0 - B2) + 1e-6)
    wrong = -(1.0 / (1.0 - B1)) * m / (v.sqrt() + 1e-6)
    assert torch.allclose(model.bias.detach(), correct, rtol=1e-6, atol=1e-12)
    assert not torch.allclose(model.bias.detach(), wrong, rtol=1e-3), (
        "eps must sit AFTER the bias-corrected sqrt, not on the raw second moment"
    )


def test_adamw_lr_is_the_value_set_mid_run_not_the_constructor_value():
    """The Muon failure mode, on the AdamW half: a mid-run LR must reach the step.

    Revert: cache the constructor LR (or write the group behind the optimizer)
    -> the second step is still 1e-3-sized.
    """
    model = Holder()
    model.w.data.zero_()
    model.bias.data.zero_()
    opt = MixOptimizer(model, lr=1e-3, wd=0.0, muon_lr_mult=8.0)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    opt.set_lr(2e-3)
    assert opt.adamw.param_groups[0]["lr"] == pytest.approx(2e-3)
    assert opt.muon.param_groups[0]["lr"] == pytest.approx(2e-3 * 8.0), (
        "the Muon group must receive lr*muon_lr_mult through the ONE seam"
    )
    before = model.bias.detach().clone()
    opt.step()
    delta = (model.bias.detach() - before).abs()
    assert torch.allclose(delta, torch.full_like(delta, 2e-3), rtol=1e-6, atol=2e-9), (
        f"AdamW step after set_lr(2e-3): {delta[0].item():.9f}"
    )


def test_set_lr_fills_device_tensors_in_place_for_the_graph_arms():
    """The graph arms read a 0-dim tensor; `set_lr` must fill, never rebind.

    Revert: assign `group["lr"] = value` -> the tensor a capture recorded keeps
    its old value and the multiplier is lost.
    """
    model = Holder()
    opt = MixOptimizer(model, lr=1e-3, wd=0.0, muon_lr_mult=8.0)
    bound = {}
    for group in opt.adamw.param_groups + opt.muon.param_groups:
        group["lr"] = torch.tensor(float(group["lr"]))
        bound[id(group)] = group["lr"]
    opt.set_lr(2e-3, 2e-3)
    for group in opt.adamw.param_groups + opt.muon.param_groups:
        assert group["lr"] is bound[id(group)], "the LR tensor object was replaced"
    assert opt.adamw.param_groups[0]["lr"].item() == pytest.approx(2e-3)
    assert opt.muon.param_groups[0]["lr"].item() == pytest.approx(2e-3 * 8.0)


def test_the_trainer_delivers_muon_lr_mult_end_to_end(tmp_path):
    """1 tiny training step at mult=1 vs 8: the Muon delta must scale ~8x.

    This is the audit's exact failure (`loop.py` wrote the schedule LR into the
    Muon group every step). Revert the `set_lr` seam -> the ratio is 1.0.
    """
    deltas = {}
    for mult in (1.0, 8.0):
        run = tiny_run(
            tmp_path,
            steps=1,
            log_every=1,
            ckpt_every=1000,
            ckpt_name=f"m{mult:g}",
            lr=1e-3,
            warmup_tokens=0,
            grad_clip=0.0,
            muon_lr_mult=mult,
        )
        torch.manual_seed(0)
        model = DormouseModel(run.model)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        train(model, run, log=lambda *a, **k: None)
        deltas[mult] = (
            model.get_parameter("loop.attn.q_proj.weight").detach()
            - before["loop.attn.q_proj.weight"]
        )
    ratio = deltas[8.0].norm().item() / deltas[1.0].norm().item()
    assert ratio == pytest.approx(8.0, rel=0.02), f"muon_lr_mult delivered {ratio:.3f}x"


# --------------------------------------------------------------------------
# 2. global-norm gradient clip
# --------------------------------------------------------------------------


def test_clip_norm_spans_every_optimizer_group():
    """The norm is over ALL parameters, not one group at a time.

    Revert: compute the coefficient per group -> each group lands at 1.0 and the
    global post-clip norm is sqrt(2), not 1.
    """

    class G(nn.Module):
        def __init__(self):
            super().__init__()
            self.muon_side = nn.Parameter(torch.zeros(100))
            self.adamw_side = nn.Parameter(torch.zeros(100))

    g = G()
    g.muon_side.grad = torch.full((100,), 3.0)  # group norm 30
    g.adamw_side.grad = torch.full((100,), 4.0)  # group norm 40
    total = clip_grads(g.parameters(), 1.0)
    assert total.item() == pytest.approx(50.0, rel=1e-6)
    post = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in g.parameters()]))
    assert post.item() == pytest.approx(1.0, rel=1e-5), (
        f"global norm after clip = {post.item():.6f} (per-group clipping gives sqrt(2))"
    )
    ratio = g.muon_side.grad.norm().item() / 30.0
    assert g.adamw_side.grad.norm().item() / 40.0 == pytest.approx(ratio, rel=1e-6), (
        "both groups must get the SAME coefficient"
    )


def test_clip_fires_only_above_the_norm_and_is_bit_identical_below():
    """Below the threshold nothing moves, bit for bit; above it the ratio is exact.

    Revert: always multiply by `max_norm/(norm+eps)` without the `min(1, ·)`
    clamp -> the below-threshold gradients shrink and `torch.equal` fails.
    """
    p = nn.Parameter(torch.zeros(5))
    p.grad = torch.linspace(-0.1, 0.1, 5)
    before = p.grad.clone()
    clip_grads([p], 10.0)
    assert torch.equal(p.grad, before), "under the norm the clip must be the identity"

    p.grad = torch.full((5,), 6.0)  # norm = 6*sqrt(5)
    total = p.grad.norm().item()
    clip_grads([p], 1.0)
    assert p.grad.norm().item() == pytest.approx(1.0, rel=1e-6)
    assert p.grad[0].item() == pytest.approx(1.0 / math.sqrt(5), rel=1e-6), (
        f"coefficient {p.grad[0].item():.9f} vs {1 / math.sqrt(5):.9f} (total {total:.4f})"
    )


def test_clip_is_applied_to_the_gradients_BEFORE_the_step():
    """Clip-then-step == feed the already-scaled gradient to a fresh step.

    Revert: apply the clip to the UPDATE (after `opt.step`) -> the parameter
    delta is the unclipped one, 100x off at this ratio.
    """

    class W(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(6))

    def step(clip: bool):
        torch.manual_seed(3)
        m = W()
        m.w.grad = torch.tensor([5.0, -5.0, 5.0, -5.0, 5.0, -5.0])
        total = m.w.grad.norm().item()
        opt = MixOptimizer(m, lr=1e-3, wd=0.0, opt="adamw")
        if clip:
            clip_grads(m.parameters(), 0.5)
        else:
            m.w.grad.mul_(0.5 / (total + 1e-6))
        before = m.w.detach().clone()
        opt.step()
        return m.w.detach() - before

    clipped, pre_scaled = step(True), step(False)
    assert torch.allclose(clipped, pre_scaled, rtol=1e-5, atol=1e-9), (
        f"clipped {clipped.flatten()[:2].tolist()} vs pre-scaled "
        f"{pre_scaled.flatten()[:2].tolist()}"
    )


def test_the_training_stage_clips_before_opt_step(monkeypatch, tmp_path):
    """The loop's stage order is AGC -> global clip -> `opt.step`.

    Revert: move `clip_grads` after `opt.step()` -> the recorded order flips and
    the clip stops bounding the update the step consumes.
    """
    import dormouse.train.loop as loop_mod

    run = tiny_run(tmp_path, steps=1, log_every=1, grad_clip=0.5)
    model = DormouseModel(run.model)
    calls: list[str] = []
    real_clip, real_step = loop_mod.clip_grads, MixOptimizer.step

    def spy_clip(params, max_norm):
        calls.append("clip")
        return real_clip(params, max_norm)

    def spy_step(self, flag=None):
        calls.append("step")
        return real_step(self, flag)

    monkeypatch.setattr(loop_mod, "clip_grads", spy_clip)
    monkeypatch.setattr(MixOptimizer, "step", spy_step)
    train(model, run, log=lambda *_: None)
    assert calls == ["clip", "step"], f"clip must precede the step: {calls}"


def test_the_loop_clips_every_parameter_not_one_group(monkeypatch, tmp_path):
    """`opt_stage` hands `clip_grads` the WHOLE `model.parameters()` tree.

    Revert: pass only `muon_params` (or one group) -> the spy's id set misses
    the AdamW half and the norm no longer bounds the step it feeds.
    """
    import dormouse.train.loop as loop_mod

    run = tiny_run(tmp_path, steps=1, log_every=1, grad_clip=0.5)
    model = DormouseModel(run.model)
    seen: set[int] = set()
    real = loop_mod.clip_grads

    def spy(params, max_norm):
        seen.update(id(p) for p in params)
        return real(params, max_norm)

    monkeypatch.setattr(loop_mod, "clip_grads", spy)
    train(model, run, log=lambda *_: None)
    assert seen == {id(p) for p in model.parameters()}, (
        f"the clip saw {len(seen)}/{len(list(model.parameters()))} parameters"
    )


def test_clip_update_ratio_is_one_at_step_one_and_ten_percent_later():
    """The ratio of clipped to unclipped updates — the measured size of the knob.

    AdamW normalises the gradient, so the FIRST clipped step is the same size as
    the unclipped one (ratio 1.0); once the coefficient varies run to run the
    trajectory changes (measured ~1.10 on this alternating pattern). A clip
    applied to the update would give ratio 0.01 (0.5/50), not 1.0.
    """

    class W(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(4))

    def deltas(clip: bool):
        m = W()
        opt = MixOptimizer(m, lr=1e-3, wd=0.0, opt="adamw")
        out = []
        for g in (
            [30.0, 40.0, 0.0, 0.0],
            [0.03, 0.04, 0.0, 0.0],
            [30.0, 40.0, 0.0, 0.0],
            [0.03, 0.04, 0.0, 0.0],
        ):
            m.w.grad = torch.tensor(g)
            if clip:
                clip_grads(m.parameters(), 0.5)
            before = m.w.detach().clone()
            opt.step()
            out.append(m.w.detach() - before)
        return out

    clipped, unclipped = deltas(True), deltas(False)
    first = clipped[0].norm().item() / unclipped[0].norm().item()
    multi = clipped[3].norm().item() / unclipped[3].norm().item()
    assert first == pytest.approx(1.0, rel=1e-4), f"first-step ratio {first:.6f}"
    assert 1.05 < multi < 1.15, f"multi-step ratio {multi:.4f} (the clip moved the trajectory)"


# --------------------------------------------------------------------------
# 3. AGC
# --------------------------------------------------------------------------


def test_agc_is_the_per_parameter_ratio_and_the_identity_when_it_does_not_fire():
    """`scale = clamp(lambda*||p|| / max(||g||, eps), max=1)` per TENSOR.

    Revert: clip per element or omit the clamp -> the second tensor (ratio > 1)
    is scaled up, and the first tensor's ratio is no longer lambda*||p||/||g||.
    """
    p1 = nn.Parameter(torch.ones(10))  # ||p|| = sqrt(10)
    p1.grad = torch.full((10,), 100.0)  # ||g|| = 100 sqrt(10) -> scale 0.005
    p2 = nn.Parameter(torch.full((10,), 100.0))
    p2.grad = torch.ones(10)  # ratio 50 -> clamped to 1
    p3 = nn.Parameter(torch.zeros(4))  # the documented zero-init hazard
    p3.grad = torch.ones(4)
    p4 = nn.Parameter(torch.ones(3))
    p4.grad = None
    orig1, orig2 = p1.grad.clone(), p2.grad.clone()
    agc_clip([p1, p2, p3, p4], 0.5)
    assert torch.allclose(p1.grad, orig1 * 0.005, rtol=1e-6, atol=1e-9)
    assert torch.equal(p2.grad, orig2), "a scale above 1 must be clamped to the identity"
    assert torch.equal(p3.grad, torch.zeros_like(p3.grad)), "||p||=0 gives scale 0 (documented)"
    assert p4.grad is None

    # An inactive AGC (lambda so large every ratio exceeds 1) is bit-identical.
    p = nn.Parameter(torch.ones(7))
    p.grad = torch.linspace(-1.0, 1.0, 7)
    before = p.grad.clone()
    agc_clip([p], 1e9)
    assert torch.equal(p.grad, before), "AGC must be the multiply identity when it never fires"


# --------------------------------------------------------------------------
# 4. EMA
# --------------------------------------------------------------------------


def test_ema_is_the_analytic_exponential_average():
    """`shadow_n = d*shadow_{n-1} + (1-d)*p_n`, starting at the initial weights.

    Revert: remove the shadow update (or update with `(1-d)` backwards) -> the
    measured sequence leaves the recurrence and its closed form.
    """

    class Mod(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(4))

    model = Mod()
    shadow = {n: p.detach().clone() for n, p in model.named_parameters()}
    flag = torch.ones((), dtype=torch.int32)
    decay = 0.9
    ref = torch.zeros(4)
    seen = []
    for t in range(1, 6):
        with torch.no_grad():
            model.w.fill_(float(t))
        _ema_update(model, shadow, decay, flag)
        ref = decay * ref + (1.0 - decay) * float(t)
        seen.append(shadow["w"][0].item())
        assert shadow["w"][0].item() == pytest.approx(ref[0].item(), rel=1e-6)
    closed = sum((1.0 - decay) * decay ** (5 - k) * k for k in range(1, 6))
    assert seen[-1] == pytest.approx(closed, rel=1e-6), f"EMA {seen[-1]} vs closed form {closed}"


def test_ema_skipped_step_leaves_the_shadow_bit_identical():
    """The device firewall: a non-finite step's flag is 0, `lerp_(0)` is a no-op.

    Revert: drop the flag from `_ema_update`'s weight -> the shadow moves on a
    skipped step.
    """

    class Mod(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(3))

    model = Mod()
    shadow = {"w": torch.zeros(3)}
    before = shadow["w"].clone()
    _ema_update(model, shadow, 0.9, torch.zeros((), dtype=torch.int32))
    assert torch.equal(shadow["w"], before)


def test_ema_checkpoint_carries_BOTH_copies_and_serves_the_shadow(tmp_path):
    """`model` is raw, `model_ema` is the shadow, and the server prefers the shadow.

    Revert: pass `ema=None` to `save_ckpt` -> `model_ema` disappears; swap the
    halves -> `model` equals the shadow instead of the raw weights.
    """
    run = tiny_run(tmp_path, steps=1, ckpt_every=1000)
    model = DormouseModel(run.model)
    opt = MixOptimizer(model, lr=run.train.lr, wd=run.train.wd)
    shadow = _shadow_state(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.25)
        for s in shadow.values():
            s.add_(-0.5)  # make the two copies visibly different
    weights, _snapshot = save_ckpt(run, model, opt, 0, ema=shadow)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    raw = model.state_dict()
    assert set(ck["model"]) == set(raw)
    for k, v in raw.items():
        assert torch.equal(ck["model"][k], v), f"raw half moved at {k}"
    for k, v in shadow.items():
        assert torch.equal(ck["model_ema"][k], v), f"EMA half moved at {k}"
        assert not torch.equal(ck["model"][k], v), f"the two halves are identical at {k}"
    served = load_model_half(weights)
    for k, v in shadow.items():
        assert torch.equal(served[k], v), f"the served model is not the EMA at {k}"


def test_ema_resume_continues_the_average(tmp_path):
    """3 steps + resume to 5 == an unbroken 5-step run, shadow included.

    Revert: ignore `ck["model_ema"]` on the resume (or restart the shadow from
    the loaded weights) -> the resumed shadow diverges from the unbroken one.
    """
    seed = 7
    common = dict(steps=3, ckpt_every=3, log_every=3, ema_decay=0.9, lr=0.01, seed=seed)
    run_a = tiny_run(tmp_path, **common)
    torch.manual_seed(seed)
    report_a = train(DormouseModel(run_a.model), run_a, log=lambda *a, **k: None)

    run_b = tiny_run(tmp_path, **{**common, "steps": 5})
    torch.manual_seed(seed)
    model_b = DormouseModel(run_b.model)
    report_b = train(model_b, run_b, resume=True, log=lambda *a, **k: None)

    run_c = dataclasses.replace(
        run_b, train=dataclasses.replace(run_b.train, ckpt_name="u", out_dir=str(tmp_path / "u"))
    )
    torch.manual_seed(seed)
    model_c = DormouseModel(run_c.model)
    report_c = train(model_c, run_c, log=lambda *a, **k: None)

    assert report_a["opt_steps"] == 3
    for k, v in report_c["ema"].items():
        assert torch.equal(report_b["ema"][k], v), f"resumed EMA differs at {k}"
    for a, b in zip(model_b.parameters(), model_c.parameters(), strict=True):
        assert torch.equal(a, b), "resumed raw weights differ from the unbroken run"


# --------------------------------------------------------------------------
# 5. schedule-free Muon
# --------------------------------------------------------------------------


def test_schedule_free_averages_z_and_x_as_algorithm_1():
    """The three-line algebra: Z decays/steps, X averages Z, params hold Y.

    Revert: drop the `(1-c)` on X, the `wd` at Z, or the Y lerp -> the measured
    z/x/Y leave the reference.
    """
    torch.manual_seed(4)
    o = torch.randn(4, 4)
    lr, wd, beta1 = 0.1, 0.01, 0.9
    p = nn.Parameter(torch.zeros(4, 4))
    muon = Muon([p], lr=lr, sf=True, wd=wd, momentum=0.0, ns_steps=1)

    z = torch.zeros(4, 4)
    x = torch.zeros(4, 4)
    s = 0.0
    for _ in range(3):
        phat = _colrow_ref(o)
        eta = 0.2 * lr * math.sqrt(16) / phat.norm().clamp_min(1e-12)
        s += lr * lr
        c = lr * lr / s
        z = (1.0 - lr * wd) * z - eta * phat
        x = (1.0 - c) * x + c * z
        y = (1.0 - beta1) * z + beta1 * x
        muon._sf_one(p, o.clone(), lr)
        st = muon.state[p]
        assert torch.allclose(st["z"], z, rtol=1e-6, atol=1e-9), "Z moved"
        assert torch.allclose(st["x"], x, rtol=1e-6, atol=1e-9), "X moved"
        assert st["sf_s"] == pytest.approx(s, rel=1e-12)
        assert torch.allclose(p.detach(), y, rtol=1e-6, atol=1e-9), "params must hold Y"

    muon.sf_switch("eval")
    assert torch.equal(p.detach(), muon.state[p]["x"]), "eval must expose X"
    muon.sf_switch("train")
    assert torch.allclose(
        p.detach(),
        (1.0 - beta1) * muon.state[p]["z"] + beta1 * muon.state[p]["x"],
        rtol=1e-6,
        atol=1e-9,
    ), "train must expose Y"


def test_schedule_free_refuses_zero_weight_decay_at_both_seams():
    """`muon_sf` needs wd > 0 at Z (Lemma 3.1): config and constructor refuse 0.

    Revert: drop either refusal -> a run reports the arm outside its theorem's
    precondition.
    """
    with pytest.raises(LoudError, match="muon_sf_wd"):
        validate_train(dataclasses.replace(TrainConfig(), muon_sf=True, muon_sf_wd=0.0))
    with pytest.raises(LoudError, match="muon_sf_wd"):
        MixOptimizer(Holder(), lr=1e-3, wd=0.0, muon_sf=True, muon_sf_wd=0.0)
    MixOptimizer(Holder(), lr=1e-3, wd=0.0, muon_sf=True, muon_sf_wd=0.01)


def test_muon_sf_retract_is_a_refusal_and_has_no_call_site():
    """`muon_sf_retract=true` refuses; there is no retraction to switch on.

    Revert: delete the config refusal -> a flag with no implementation is
    accepted and reported as an arm that never runs.
    """
    with pytest.raises(LoudError, match="muon_sf_retract"):
        validate_train(dataclasses.replace(TrainConfig(), muon_sf_retract=True))
    assert not hasattr(Muon, "retract_tsct")
    import re

    call = re.compile(r"retract_tsct\s*\(")
    for path in SRC.rglob("*.py"):
        assert not call.search(path.read_text()), f"an unported retraction call appeared in {path}"


def test_the_sf_arm_keeps_its_muon_lr_constant_after_warmup(monkeypatch, tmp_path):
    """The Muon half gets `decay_steps=0`; the AdamW half keeps the WSD tail.

    Revert: pass the WSD `decay_steps` to the Muon call -> both arms decay and
    the last-step values converge.
    """
    import dormouse.train.loop as loop_mod

    seen: list[tuple[int, int, float]] = []
    real = loop_mod.wsd_lr

    def spy(step, **kw):
        value = real(step, **kw)
        seen.append((step, kw["decay_steps"], value))
        return value

    monkeypatch.setattr(loop_mod, "wsd_lr", spy)
    run = tiny_run(
        tmp_path,
        steps=6,
        log_every=6,
        ckpt_every=1000,
        warmup_tokens=100,
        decay_frac=0.5,
        muon_sf=True,
        muon_sf_wd=0.01,
    )
    train(DormouseModel(run.model), run, log=lambda *a, **k: None)
    per_step = {}
    for step, decay, value in seen:
        per_step.setdefault(step, {})[decay] = value
    last = per_step[5]
    muon_value = last[0]
    adamw_value = next(v for d, v in last.items() if d > 0)
    assert muon_value == pytest.approx(run.train.lr, rel=1e-12), "Muon LR must stay at peak"
    assert adamw_value < muon_value, "AdamW LR must still be on the decay tail"


# --------------------------------------------------------------------------
# 6. the WSD schedule
# --------------------------------------------------------------------------


def test_wsd_last_100_steps_match_the_closed_form():
    """The last 100 values of a full WSD curve against the written formula.

    Revert: drop `final_frac`, or measure the decay from step 0 -> the tail
    leaves the closed form (and the last value is not `peak*min_lr_frac`).
    """
    peak, total, warm, decay, final = 3e-4, 1000, 100, 100, 0.01
    curve = [
        wsd_lr(
            s,
            total_steps=total,
            peak_lr=peak,
            warmup_steps=warm,
            decay_steps=decay,
            final_frac=final,
        )
        for s in range(total)
    ]
    for step in range(total - 100, total):
        want = _closed_form_lr(step, total=total, peak=peak, warm=warm, decay=decay, final=final)
        assert curve[step] == pytest.approx(want, rel=1e-12, abs=1e-15), (
            f"step {step}: {curve[step]} vs {want}"
        )
    assert curve[-1] == pytest.approx(peak * final, rel=1e-12)
    assert curve[total - 100] == pytest.approx(peak * (1 - 0.01 * (1 - final)), rel=1e-12)
    print(
        f"wsd tail: {total - 100}:{curve[total - 100]:.3e} "
        f"{total - 50}:{curve[total - 50]:.3e} {total - 1}:{curve[-1]:.3e}"
    )


def test_a_resume_recomputes_the_schedule_from_the_global_step(monkeypatch, tmp_path):
    """A 6-step run + resume to 10 must land on steps 6..9, not 0..3.

    Revert: call `wsd_lr(step - start)` -> the resumed curve restarts the warmup
    and the spy's values differ from the global closed form.
    """
    import dormouse.train.loop as loop_mod

    seen: list[tuple[int, float]] = []
    real = loop_mod.wsd_lr

    def spy(step, **kw):
        value = real(step, **kw)
        seen.append((step, value))
        return value

    monkeypatch.setattr(loop_mod, "wsd_lr", spy)
    common = dict(steps=6, ckpt_every=6, log_every=6, warmup_tokens=100, decay_frac=0.5)
    run = tiny_run(tmp_path, **common)
    train(DormouseModel(run.model), run, log=lambda *a, **k: None)
    assert [s for s, _ in seen] == list(range(6))

    seen.clear()
    resumed = tiny_run(tmp_path, **{**common, "steps": 10})
    train(DormouseModel(resumed.model), resumed, resume=True, log=lambda *a, **k: None)
    assert [s for s, _ in seen] == list(range(6, 10)), f"resumed steps {seen}"
    warm = math.ceil(100 / (resumed.train.batch * resumed.train.seq_len))
    decay = round(0.5 * 10)
    for step, value in seen:
        want = _closed_form_lr(
            step,
            total=10,
            peak=resumed.train.lr,
            warm=warm,
            decay=decay,
            final=resumed.train.min_lr_frac,
        )
        assert value == pytest.approx(want, rel=1e-12, abs=1e-15), (
            f"step {step}: {value} vs {want} (warmup restarted?)"
        )


# --------------------------------------------------------------------------
# 7. z-loss / logit softcap (and the QK-clip that does not exist)
# --------------------------------------------------------------------------


def _logit_model(**over) -> DormouseModel:
    cfg = ModelConfig(**{**tiny_cfg().__dict__, "moe_capacity": 0.0, "use_tsct": False})
    return DormouseModel(dataclasses.replace(cfg, **over))


def test_z_loss_off_is_bit_identical_and_on_is_coef_times_logsumexp_squared():
    """`loss = CE + coef * mean(LSE(logits)^2)`, logits untouched; 0 adds no op.

    Revert: add the term with a different reduction, or apply it to the logits
    -> the delta or the logits move.
    """
    ids = torch.arange(2 * SEQ, dtype=torch.long).view(2, SEQ) % 251
    targets = (ids + 1) % 251
    torch.manual_seed(0)
    off = _logit_model()
    on = _logit_model(z_loss=1e-3)
    on.load_state_dict(off.state_dict())
    logits_off, loss_off, _ = off(ids, targets)
    logits_on, loss_on, counters = on(ids, targets)
    assert torch.equal(logits_off, logits_on), "z-loss must not change the logits"
    term = 1e-3 * torch.logsumexp(logits_off.detach(), dim=-1).pow(2).mean()
    assert (loss_on - loss_off).item() == pytest.approx(term.item(), rel=1e-5)
    assert counters.get("z_loss") == 1
    _, pure_off, _ = off(ids, targets, pure=True)
    _, pure_on, _ = on(ids, targets, pure=True)
    assert torch.equal(pure_off, pure_on), "the held-out (pure) loss skips the regulariser"


def test_logit_softcap_off_is_bit_identical_and_on_is_cap_tanh():
    """`logits <- cap * tanh(logits/cap)`: exact formula, and 0 is a no-op.

    Revert: use a hard clamp (`clamp(-cap, cap)`) or apply the cap to the loss
    instead of the logits -> the formula equality fails.
    """
    x = torch.tensor([[-20.0, -1.0, 0.0, 1.0, 20.0]])
    cap = 15.0
    assert torch.equal(logit_softcap(x, cap), cap * torch.tanh(x / cap))
    ids = torch.arange(2 * SEQ, dtype=torch.long).view(2, SEQ) % 251
    targets = (ids + 1) % 251
    torch.manual_seed(0)
    off = _logit_model()
    capped = _logit_model(logit_softcap=cap)
    capped.load_state_dict(off.state_dict())
    raw, _, _ = off(ids, targets)
    got, _, _ = capped(ids, targets)
    assert torch.equal(got, cap * torch.tanh(raw / cap)), "the model path must use the formula"
    assert got.abs().max().item() < cap


def test_no_qk_clip_knob_is_advertised():
    """QK-clip is not implemented here; a config field without code is the defect class."""
    from dormouse.config import ModelConfig as MC

    assert "qk_clip" not in {f.name for f in dataclasses.fields(MC)}
    for path in SRC.rglob("*.py"):
        assert "qk_clip" not in path.read_text(), f"an unwired qk_clip appeared in {path}"


# --------------------------------------------------------------------------
# 8. bf16 mixed precision
# --------------------------------------------------------------------------


def test_bf16_model_keeps_fp32_masters_and_steps_them_in_fp32():
    """`bf16=true` changes the GEMM compute dtype, not the master weights.

    Revert: cast the parameters to bf16 (a "pure bf16" model) -> the dtype
    assertions fail and the AdamW reference no longer matches.
    """
    cfg = tiny_cfg(bf16=True, moe_capacity=0.0, use_tsct=False)
    torch.manual_seed(0)
    model = DormouseModel(cfg)
    assert all(p.dtype is torch.float32 for p in model.parameters())
    ids = torch.arange(2 * SEQ, dtype=torch.long).view(2, SEQ) % 251
    _, loss, _ = model(ids, (ids + 1) % 251)
    loss.backward()
    assert all(p.grad.dtype is torch.float32 for p in model.parameters() if p.grad is not None)

    head = model.lm_head.weight
    p0, g = head.detach().clone(), head.grad.clone()
    opt = MixOptimizer(model, lr=1e-3, wd=0.01)
    assert opt.routing["lm_head.weight"] == "adamw"
    opt.step()
    ref, _, _ = _adamw_ref(p0, torch.zeros_like(p0), torch.zeros_like(p0), g, 1e-3, 0.01, 1e-8, 1)
    assert all(p.dtype is torch.float32 for p in model.parameters())
    assert torch.allclose(head.detach(), ref, rtol=1e-5, atol=1e-7), (
        "the fp32 master did not take the AdamW step computed from its fp32 gradient"
    )


def test_muon_bf16_state_stores_one_rounded_fp32_accumulation_per_step():
    """The bf16 momentum rung: fp32 accumulation, one bf16 store per step.

    Revert: accumulate in bf16 (`mom.mul_(mu)` in bf16) -> the second-step
    state drifts from `round(fp32 recurrence)`.
    """
    model = Holder()
    opt = MixOptimizer(model, lr=1e-2, wd=0.0, muon_state_dtype="bf16")
    g1, g2 = torch.randn(8, 8), torch.randn(8, 8)
    model.w.grad = g1.clone()
    opt.step()
    st = opt.muon.state[model.w]["momentum"]
    assert st.dtype is torch.bfloat16
    m1 = ((1.0 - 0.95) * g1).to(torch.bfloat16)
    assert torch.equal(st, m1), "step 1: bf16 state is not the rounded fp32 momentum"
    model.w.grad = g2.clone()
    opt.step()
    m2 = (0.95 * m1.float() + (1.0 - 0.95) * g2).to(torch.bfloat16)
    assert torch.equal(opt.muon.state[model.w]["momentum"], m2), (
        "step 2 accumulated in bf16 instead of rounding one fp32 accumulation"
    )


def test_muon_chunking_and_the_fp32_ns_rung_are_bit_identical_to_the_default():
    """`muon_chunk_elems` and an explicit `ns_dtype="fp32"` are pure selections.

    Revert: make the NS chain depend on the chunk boundary, or change the fp32
    arm when the argument is spelled out -> the updates stop being `torch.equal`.
    """
    g = torch.randn(8, 8)

    def update(**kw):
        torch.manual_seed(0)
        model = Holder()
        opt = MixOptimizer(model, lr=1e-2, wd=0.0, **kw)
        model.w.grad = g.clone()
        before = model.w.detach().clone()
        opt.step()
        return model.w.detach() - before

    base = update()
    assert torch.equal(update(muon_chunk_elems=1), base), "chunking moved the update"
    assert torch.equal(update(ns_dtype="fp32"), base), "the explicit fp32 rung moved the update"


# --------------------------------------------------------------------------
# 9. the post-training loops: fields the shared config accepts but they cannot run
# --------------------------------------------------------------------------


def test_post_training_loops_refuse_the_arms_they_cannot_honour():
    """`agc_clip` and `ema_decay` are `train()`-only; SFT/RL must refuse, not drop.

    Revert: delete either refusal from `sft.refuse_unwired` -> the loop accepts
    a knob that changes nothing (the m5/`muon_lr_mult` class).
    """
    for loop in ("sft", "rl"):
        for kwargs in ({"agc_clip": 1.0}, {"ema_decay": 0.9}):
            cfg = dataclasses.replace(TrainConfig(), **kwargs)
            with pytest.raises(LoudError, match=next(iter(kwargs))):
                refuse_unwired(cfg, loop)
    refuse_unwired(TrainConfig(), "sft")
    refuse_unwired(TrainConfig(), "rl")


def _sft_run(tmp_path: Path, **over) -> RunConfig:
    data = tmp_path / "sft.bin"
    build_sft_corpus(data, seed=1, seq_len=SEQ, n_tasks=32)
    cfg = TrainConfig(
        steps=1,
        batch=2,
        seq_len=SEQ,
        lr=0.01,
        log_every=1,
        ckpt_every=1000,
        ckpt_name="s",
        out_dir=str(tmp_path / "sftck"),
        data=str(data),
    )
    return RunConfig(
        source="opt-suite-sft", model=tiny_cfg(), train=dataclasses.replace(cfg, **over)
    )


def test_sft_forwards_adam_eps_to_the_optimizer(monkeypatch, tmp_path):
    """A non-default `adam_eps` used to be dropped; the SFT optimizer must see it.

    Revert: remove `eps=cfg.adam_eps` from `sft.py`'s constructor -> the spy
    sees the MixOptimizer default `1e-8`.
    """
    import dormouse.train.sft as sft_mod

    seen: dict = {}
    real = sft_mod.MixOptimizer

    class Spy(real):
        def __init__(self, *a, **kw):
            seen.update(kw)
            super().__init__(*a, **kw)

    monkeypatch.setattr(sft_mod, "MixOptimizer", Spy)
    run = _sft_run(tmp_path, adam_eps=1e-3)
    sft_train(DormouseModel(run.model), run, log=lambda *_: None)
    assert seen.get("eps") == 1e-3, f"SFT dropped adam_eps: {seen.get('eps')}"


def test_rl_forwards_adam_eps_to_the_optimizer(monkeypatch, tmp_path):
    """The same wiring on the RL loop (it builds the optimizer through its own module)."""
    import sys

    rl_mod = sys.modules["dormouse.post.rl_train"]
    seen: dict = {}
    real = rl_mod.MixOptimizer

    class Spy(real):
        def __init__(self, *a, **kw):
            seen.update(kw)
            super().__init__(*a, **kw)

    monkeypatch.setattr(rl_mod, "MixOptimizer", Spy)
    tmp = tmp_path / "rl"
    cfg = TrainConfig(
        steps=1,
        batch=2,
        seq_len=64,
        lr=0.01,
        log_every=1,
        ckpt_every=1000,
        ckpt_name="r",
        out_dir=str(tmp),
        rl_task_cls="math",
        rl_tasks=2,
        rl_group=2,
        rl_budget=96,
        rl_holdout=1,
        adam_eps=1e-3,
    )
    run = RunConfig(source="opt-suite-rl", model=tiny_cfg(max_seq_len=128), train=cfg)
    torch.manual_seed(0)
    rl_train(run, log=lambda *a, **k: None)
    assert seen.get("eps") == 1e-3, f"RL dropped adam_eps: {seen.get('eps')}"


def _adamw_arm(lr: float = 0.05):
    """A tiny model whose 2-D weight goes to Muon and the rest to AdamW.

    Seeded: two arms must start from the SAME weights, or the comparison
    measures the initialisation, not the optimizer (a 0.7 divergence that looks
    like a bug and is not).
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    opt = MixOptimizer(model, lr=lr, wd=0.01)
    return model, opt


def _fixed_grads(model, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=g)


def test_the_torch_adamw_path_advances_the_bias_correction_counter(monkeypatch):
    """A mid-run switch of AdamW implementation must not reuse a stale step.

    The fused writer forms the bias correction from `_adamw_t`; torch's own
    `step()` advances each parameter's `state["step"]` but not that counter. A
    run that falls back from the captured graph (or from the fused writer) to
    the torch path therefore resumed the fused correction at t=1 — measured as a
    3.1e-05 divergence at `embed.weight` with losses equal to nine digits
    (docs/reviews/2026-11-08-fused-cluster.md §1).
    """
    slow_m, slow_o = _adamw_arm()
    fast_m, fast_o = _adamw_arm()
    _fixed_grads(slow_m)
    _fixed_grads(fast_m)

    monkeypatch.setattr(slow_o, "_fast_adamw_ok", lambda: False)
    slow_o.step()
    monkeypatch.setattr(slow_o, "_fast_adamw_ok", lambda: True)
    slow_o.step()

    fast_o.step()
    fast_o.step()

    assert slow_o._adamw_t == 2, slow_o._adamw_t
    assert fast_o._adamw_t == 2, fast_o._adamw_t
    for (n, a), (_, b) in zip(slow_m.named_parameters(), fast_m.named_parameters(), strict=True):
        assert torch.allclose(a, b, rtol=0, atol=0), (
            f"{n}: the torch path and the fused path diverged — "
            f"max|d|={float((a - b).abs().max()):.3e}"
        )
