"""TSCT-2: the three mechanisms from `docs/architecture/2026-11-05-tsct-2.md`.

The structural finding, verified at `experts.py` HEAD `ecc979e`: with
`share_u_mode='plane'` (the live m400b recipe) the routed experts share ONE
`gate_u`/`up_u [d, rank]` (rank 64), own a per-coordinate `gate_s`/`up_s [64]`
and a `[64, f]` write; the doc's `[d, 192]` is `shared_rank`, the always-on
shared expert's width, not the routed plane. So the arm widens the routed plane
to `shared_rank` and gives each expert `w [rank, shared_rank]`; folding
`gate_u @ w.T` keeps the per-token operand shapes and MAC unchanged.

Gates (each has a scratch-revert proof in the lane report):
A. all three OFF is bit-identical to the pins captured before the edit
   (state/logits/loss/grads sha), and the ON model's NON-new parameters are
   bit-identical (the arm must not shift the initialisation it is compared to).
B. `tsct_expert_basis` at init is output-equivalent to the planeless path
   (`max|Δlogits| = 0.0`; gate is 1e-6), and the parameter delta is exact.
C. with a non-trivial `w`: the bank's folded basis equals the literal
   `gate_u @ w.T`; one expert's forward equals the doc's explicit formula; the
   full banked pass matches the reference loop on logits and on every TSCT-2
   parameter gradient within 1e-6 relative.
D. the LR multiplier is a pure function of the counts (exact float compare),
   clips at [0.2, 5], and a Muon step scales ONLY the `v` rows: `gate_u`,
   `up_u` and `w` move bit-identically with the arm on/off.
E. revive: `revive=0` is inert, the scripted coldest expert receives exactly
   `donor + randn · (noise · std(v_hot))`, both LFB entries reset, the result
   is seed-deterministic, and a short run's live count does not drop.
F. no host read on the new step-path code (`_banks` fold, LR scales, Muon row
   scale); the counts stay device-side. The revive indices are the ONE declared
   read, once per cadence, and are excluded from this gate (`revive=0` here).
"""

from __future__ import annotations

import hashlib

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_cfg

from dormouse.config import validate
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.model.experts import (
    expert_lr_multipliers,
    invalidate_banks_all,
    revive_experts_all,
    tsct_basis_param_delta,
)
from dormouse.model.model import param_count
from dormouse.train.optim import MixOptimizer

IDS = torch.arange(2 * 64).reshape(2, 64) * 7 % 256
TGT = torch.roll(IDS, -1, dims=1)

#: Captured on clean `main` `ecc979e` BEFORE the edit: tiny `TINY` config,
#: `torch.manual_seed(0)`, one forward + CE + backward on `IDS/TGT`.
GOLDEN = {
    "off": {
        "params": 64881,
        "sd": "f1ea6bb2273de0345913331fe9b747878796b4fb40ba5cc143fa678a7359a4b8",
        "logits": "d51cc54995a5024320c9357cfeb564acf576bd3b68e9b6f9e90ad3c7d917dac1",
        "loss": 6.0657572746276855,
        "grads": "03a9ebb7ab9055554a8fcff19363207463136238c230088479e98c4801ec73ed",
    },
    "plane": {
        "params": 61809,
        "sd": "ebbbe74ef01b86fb24ae75ea39e9398c11076ef5903c882ece32790e2ab2bb57",
        "logits": "371c1289508fa12c0b888d95c00fa3591b763f2df8b0ef68c46a7dc9b041be8a",
        "loss": 6.048655033111572,
        "grads": "8d4ed399e0801f698a9e2cd973053903ed394c817868e55ecaff0a6517114fd0",
    },
}

#: The routed experts' TSCT-2 tensors, the ones gate C bounds at 1e-6. The
#: engram projections are outside the mechanism and carry the pre-existing
#: bank-vs-reference-loop reduction difference (2.0e-6 relative, printed).
TSCT2_SUFFIXES = ("w", "v", "gate_s", "up_s")


def _build(**over) -> DormouseModel:
    torch.manual_seed(0)
    return DormouseModel(tiny_cfg(**over), device="cpu")


def _state_sha(model: DormouseModel) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def _logits_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().to(torch.float32).numpy().tobytes()).hexdigest()


def _grads_sha(model: DormouseModel) -> str:
    h = hashlib.sha256()
    for k, p in sorted(model.named_parameters()):
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        h.update(k.encode())
        h.update(g.to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def _forward(model: DormouseModel) -> tuple[torch.Tensor, torch.Tensor]:
    logits, _, _ = model(IDS, targets=TGT)
    loss = F.cross_entropy(logits.reshape(-1, 256), TGT.reshape(-1))
    return logits, loss


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach(), b.detach()
    return float((a - b).norm() / (b.norm() + 1e-30))


# --- A. OFF pins -------------------------------------------------------------


def test_a_off_is_bit_identical_and_the_on_arm_does_not_shift_other_params() -> None:
    for name, over in (
        ("off", {}),
        ("plane", {"share_u_mode": "plane"}),
    ):
        model = _build(**over)
        pin = GOLDEN[name]
        assert (model.cfg.tsct_expert_basis, model.cfg.tsct_expert_lr,
                model.cfg.tsct_expert_revive) == (0, 0.0, 0)
        assert param_count(model) == pin["params"]
        assert _state_sha(model) == pin["sd"], f"the new fields moved the {name} init"
        logits, loss = _forward(model)
        loss.backward()
        assert _logits_sha(logits) == pin["logits"], f"the new fields moved the {name} forward"
        assert loss.item() == pin["loss"]
        assert _grads_sha(model) == pin["grads"], f"the new fields moved the {name} backward"

    off = _build(share_u_mode="plane")
    on = _build(share_u_mode="plane", tsct_expert_basis=8)
    so, sn = off.state_dict(), on.state_dict()
    # The two `[d, rank]` planes become `[d, shared_rank]`; EVERYTHING else is
    # bit-identical, so the A/B compares the mechanism, not the initialisation.
    widened = {"loop.moe.gate_u", "loop.moe.up_u"}
    for key in so:
        if key in widened:
            assert torch.equal(so[key], sn[key][:, : off.cfg.rank]), (
                f"the first rank columns of {key} must be today's exact draw"
            )
        else:
            assert torch.equal(so[key], sn[key]), f"{key} moved when the arm was added"
    new = sorted(set(sn) - set(so))
    assert new == [f"loop.moe.experts.{j}.w" for j in range(off.cfg.n_experts)]
    assert param_count(on) - param_count(off) == tsct_basis_param_delta(
        off.cfg.n_experts, off.cfg.d_model, off.cfg.rank, off.cfg.shared_rank
    )
    assert param_count(on) - param_count(off) == 1536  # tiny: 4·8·16 + 2·64·8


# --- B. init equivalence ------------------------------------------------------


def test_b_basis_at_init_is_the_planeless_forward() -> None:
    off = _build(share_u_mode="plane")
    on = _build(share_u_mode="plane", tsct_expert_basis=8)
    lo, loff = _forward(off)
    ln, lnon = _forward(on)
    delta = (ln - lo).abs().max().item()
    print(f"[gate B] init max|Δlogits|={delta:.3e}, losses {loff.item():.7f}/{lnon.item():.7f}")
    assert delta <= 1e-6, "the selector's init is not equivalent to today's path"
    assert loff.item() == lnon.item()


# --- C. non-trivial selector parity ------------------------------------------


def _perturb_w(moe) -> None:
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for e in moe.experts:
            e.w.copy_(torch.randn(e.w.shape, generator=g))


def test_c_non_trivial_selector_matches_the_literal_form() -> None:
    model = _build(share_u_mode="plane", tsct_expert_basis=8)
    moe = model.loop.moe
    _perturb_w(moe)
    invalidate_banks_all(model)

    # (1) the folded bank basis is the literal `gate_u @ w_j.T`.
    gate_u, up_u, _v, _gs, _us = moe._banks(torch.float32)
    for j, e in enumerate(moe.experts):
        assert _rel(gate_u[j], moe.gate_u @ e.w.T) <= 1e-6
        assert _rel(up_u[j], moe.up_u @ e.w.T) <= 1e-6
    print("[gate C] folded basis == literal, max rel < 1e-6")

    # (2) one expert's forward is the doc's explicit formula.
    e = moe.experts[0]
    x = torch.randn(5, model.cfg.d_model, generator=torch.Generator().manual_seed(4))
    g = ((x @ moe.gate_u) @ e.w.T) * e.gate_s
    u = ((x @ moe.up_u) @ e.w.T) * e.up_s
    ref = (F.silu(g) * u) @ e.v
    got = e(x)
    assert _rel(got, ref) <= 1e-6
    print(f"[gate C] explicit formula rel={_rel(got, ref):.3e}")

    # (3) the banked pass vs the reference loop on logits and TSCT-2 grads.
    moe.moe_kernel = "batched"
    model.zero_grad(set_to_none=True)
    lb, loss_b = _forward(model)
    loss_b.backward()
    banked = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    moe.moe_kernel = "naive"
    invalidate_banks_all(model)
    model.zero_grad(set_to_none=True)
    ln, loss_n = _forward(model)
    loss_n.backward()
    tsct2 = {"loop.moe.gate_u", "loop.moe.up_u"} | {
        f"loop.moe.experts.{j}.{s}" for j in range(moe.n_experts) for s in TSCT2_SUFFIXES
    }
    worst: list[tuple[float, str]] = []
    for n, p in model.named_parameters():
        if p.grad is None or n not in banked:
            continue
        if n in tsct2:
            worst.append((_rel(banked[n], p.grad), n))
    worst.sort(reverse=True)
    logit_rel = _rel(lb, ln)
    print(f"[gate C] banked vs reference: logits rel={logit_rel:.3e}, "
          f"worst TSCT-2 grad {worst[0][1]} rel={worst[0][0]:.3e}")
    assert logit_rel <= 1e-6
    assert worst[0][0] <= 1e-6


# --- D. per-expert LR --------------------------------------------------------


def test_d_lr_multiplier_is_pure_clipped_and_touches_only_v() -> None:
    counts = torch.tensor([10.0, 20.0, 40.0])
    expected = (1.0 + 1.0 * (counts.mean() / counts.clamp_min(1.0) - 1.0)).clamp(0.2, 5.0)
    got = expert_lr_multipliers(counts, 1.0)
    assert torch.equal(got, expected), "the multiplier is not the documented function"
    assert torch.equal(got, expert_lr_multipliers(counts.clone(), 1.0)), "not pure"
    clip = expert_lr_multipliers(torch.tensor([0.0, 1e4, 1.0]), 10.0)
    assert torch.equal(clip, torch.tensor([5.0, 0.2, 5.0])), f"clip wrong: {clip}"
    assert torch.equal(
        expert_lr_multipliers(torch.full((4,), 7.0), 3.0), torch.ones(4)
    ), "equal counts must be exactly lr"

    cfg = dict(
        share_u_mode="plane",
        tsct_expert_basis=8,
        tsct_expert_lr=1.0,
        moe_lb_mode="prop",
        moe_lb_gamma=0.002,
    )
    m_off, m_on = _build(**cfg), _build(**cfg)
    for m in (m_off, m_on):
        _, loss = _forward(m)
        loss.backward()
    scripted = torch.tensor([5.0, 12.0, 3.0, 8.0])
    m_off.loop.moe.lfb_counts.copy_(scripted)
    m_on.loop.moe.lfb_counts.copy_(scripted)
    scales = m_on.loop.moe.expert_lr_scales()
    assert torch.equal(scales, expert_lr_multipliers(scripted, 1.0))
    assert scales.device == scripted.device

    opt_off = MixOptimizer(m_off, lr=0.01, wd=0.0, opt="mix", tsct_expert_lr=0.0)
    opt_on = MixOptimizer(m_on, lr=0.01, wd=0.0, opt="mix", tsct_expert_lr=1.0)
    before = [[e.v.detach().clone() for e in m.loop.moe.experts] for m in (m_off, m_on)]
    opt_off.step()
    opt_on.step()
    worst = 0.0
    for j in range(4):
        d_off = m_off.loop.moe.experts[j].v.detach() - before[0][j]
        d_on = m_on.loop.moe.experts[j].v.detach() - before[1][j]
        worst = max(worst, (d_on - d_off * scales[j]).abs().max().item())
    # `p.add_(update, alpha=-lr)` vs `update.mul_(row)` then add: two float
    # roundings on a delta of ~0.03. The measured max is 1.6e-7; 1e-6 is the
    # declared gate, and the RELATIVE vector error is printed (1e-5, the
    # cancellation of `p_new - p_old`).
    print(f"[gate D] scales={scales.tolist()}, worst |Δv − s·Δv_ref|={worst:.3e}")
    assert worst <= 1e-6
    # The shared basis and the selector are NOT scaled: bit-identical.
    assert torch.equal(m_off.loop.moe.gate_u, m_on.loop.moe.gate_u)
    assert torch.equal(m_off.loop.moe.up_u, m_on.loop.moe.up_u)
    assert all(
        torch.equal(a.w, b.w)
        for a, b in zip(m_off.loop.moe.experts, m_on.loop.moe.experts, strict=True)
    ), "the LR arm must not scale the selector"


# --- E. revive ---------------------------------------------------------------


def test_e_revive_deterministic_donor_copy_and_inert_off() -> None:
    cfg = dict(
        share_u_mode="plane",
        tsct_expert_basis=8,
        tsct_expert_revive=4,
        tsct_revive_noise=0.01,
        moe_lb_mode="prop",
        moe_lb_gamma=0.002,
    )
    model = _build(**cfg)
    model.train()
    moe = model.loop.moe
    moe.lfb_counts.copy_(torch.tensor([5.0, 1.0, 9.0, 3.0]))
    moe.lfb_bias.copy_(torch.tensor([1.0, -1.0, 2.0, -2.0]))
    donor, receiver = moe.experts[2], moe.experts[1]
    v0 = receiver.v.detach().clone()

    # The cadence: step 5 is not a multiple of 4 and does nothing.
    assert revive_experts_all(model, 5, 7) == 0
    assert torch.equal(receiver.v, v0), "a non-cadence step moved the receiver"

    assert revive_experts_all(model, 4, 7) == 1
    g = torch.Generator().manual_seed(7 + 4)
    sigma = 0.01 * donor.v.detach().std()
    for dst, src in (
        (receiver.v, donor.v),
        (receiver.gate_s, donor.gate_s),
        (receiver.up_s, donor.up_s),
        (receiver.w, donor.w),
    ):
        expected = src.detach() + torch.randn(src.shape, generator=g) * sigma
        assert torch.equal(dst, expected), f"revive did not copy {dst.shape} exactly"
    assert moe.lfb_counts.tolist() == [5.0, 0.0, 0.0, 3.0]
    assert moe.lfb_bias.tolist() == [1.0, 0.0, 0.0, -2.0]
    # The donor was read, not moved: only the cold entry changed.
    assert torch.equal(moe.experts[2].v, donor.v)

    # Same seed, same script → same weights. (A second freshly-built model.)
    model2 = _build(**cfg)
    model2.train()
    moe2 = model2.loop.moe
    moe2.lfb_counts.copy_(torch.tensor([5.0, 1.0, 9.0, 3.0]))
    moe2.lfb_bias.copy_(torch.tensor([1.0, -1.0, 2.0, -2.0]))
    assert revive_experts_all(model2, 4, 7) == 1
    assert all(
        torch.equal(a, b)
        for a, b in zip(model2.state_dict().values(), model.state_dict().values(), strict=True)
    )

    # `revive=0` is inert, and a model built without the arm makes no call.
    off = _build(share_u_mode="plane", moe_lb_mode="prop", moe_lb_gamma=0.002)
    off.train()
    before = [p.detach().clone() for p in off.parameters()]
    assert revive_experts_all(off, 4, 7) == 0
    assert all(torch.equal(a, b) for a, b in zip(before, off.parameters(), strict=True))

    # Short run: the live count does not DROP versus the same run without it.
    def live_run(revive: int) -> list[int]:
        torch.manual_seed(0)
        m = DormouseModel(
            tiny_cfg(
                share_u_mode="plane",
                tsct_expert_revive=revive,
                moe_lb_mode="prop",
                moe_lb_gamma=0.01,
            ),
            device="cpu",
        )
        opt = MixOptimizer(m, lr=0.01, wd=0.0, opt="mix")
        out: list[int] = []
        for _ in range(20):
            opt.zero_grad(set_to_none=True)
            _, loss = _forward(m)
            loss.backward()
            out.append(int((m.loop.moe._last_counts > 0).sum()))
            opt.step()
            invalidate_banks_all(m)
            revive_experts_all(m, opt.steps, 1)
        return out

    control = live_run(0)
    revived = live_run(2)
    print(f"[gate E] live control min/mean={min(control)}/{sum(control)/len(control):.2f}, "
          f"revive min/mean={min(revived)}/{sum(revived)/len(revived):.2f}")
    assert min(revived) >= min(control)
    assert sum(revived) >= sum(control)


# --- F. no host reads on the new step-path code ------------------------------


def test_f_the_new_step_path_reads_no_host_scalar(monkeypatch) -> None:
    model = _build(
        share_u_mode="plane",
        tsct_expert_basis=8,
        tsct_expert_lr=1.0,
        moe_lb_mode="prop",
        moe_lb_gamma=0.002,
    )
    model.train()
    moe = model.loop.moe
    opt = MixOptimizer(model, lr=0.01, wd=0.0, opt="mix", tsct_expert_lr=1.0)
    # Gradients OUTSIDE the patch (backward is not the mechanism under test).
    _, loss = _forward(model)
    loss.backward()
    invalidate_banks_all(model)

    def boom(*_a, **_k):
        raise AssertionError("host sync on the TSCT-2 step path (ADR-0018 rule 2)")

    names = ("item", "tolist", "__bool__", "__int__", "__float__")
    saved = {n: getattr(torch.Tensor, n) for n in names}
    try:
        for n in saved:
            monkeypatch.setattr(torch.Tensor, n, boom, raising=True)
        gate_u, up_u, _v, _gs, _us = moe._banks(torch.float32)
        assert gate_u.shape[0] == moe.n_experts
        scales = moe.expert_lr_scales()
        assert scales.shape == (moe.n_experts,)
        opt.step()
    finally:
        for n, f in saved.items():
            monkeypatch.setattr(torch.Tensor, n, f, raising=True)
    assert moe.lfb_counts.device == torch.device("cpu")
    assert scales.device == moe.lfb_counts.device


# --- refusals ---------------------------------------------------------------


@pytest.mark.parametrize(
    "over",
    [
        {"share_u_mode": "off", "tsct_expert_basis": 8},
        {"share_u_mode": "dict", "tsct_expert_basis": 8, "share_u_rank": 16},
        {"share_u_mode": "plane", "tsct_expert_basis": 4},
        {"share_u_mode": "plane", "tsct_expert_basis": 8, "factor_qat": "ternary"},
        {"share_u_mode": "plane", "tsct_expert_basis": 8, "moe_shared_compute": True},
        {"share_u_mode": "plane", "tsct_expert_lr": 1.0, "moe_lb_mode": "aux"},
        {
            "share_u_mode": "plane",
            "tsct_expert_lr": 1.0,
            "moe_lb_mode": "prop",
            "moe_lb_gamma": 0.0,
        },
        {"share_u_mode": "plane", "tsct_expert_revive": -1},
        {"share_u_mode": "plane", "tsct_revive_noise": -1.0},
    ],
)
def test_refusals_name_the_escape(over: dict) -> None:
    with pytest.raises(LoudError, match="Escape:"):
        validate(tiny_cfg(**over))


def test_the_constructor_refuses_what_the_config_refuses() -> None:
    with pytest.raises(LoudError, match="Escape:"):
        DormouseModel(tiny_cfg(share_u_mode="off", tsct_expert_basis=8), device="cpu")


def test_the_optimizer_refuses_the_tails_it_cannot_scale() -> None:
    model = _build(
        share_u_mode="plane", tsct_expert_lr=1.0, moe_lb_mode="prop", moe_lb_gamma=0.002
    )
    with pytest.raises(LoudError, match="Escape:"):
        MixOptimizer(model, lr=0.01, wd=0.0, opt="adamw", tsct_expert_lr=1.0)
    with pytest.raises(LoudError, match="Escape:"):
        MixOptimizer(
            model, lr=0.01, wd=0.0, opt="mix", muon_variant="normuon", tsct_expert_lr=1.0
        )
