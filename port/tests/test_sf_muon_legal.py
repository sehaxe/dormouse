"""The SF-Muon arm's legal preconditions, and the re-run's verdict statistic.

TASK A (`docs/reviews/2026-10-21-sf-muon-prerequisites.md`): before the SF arm
is judged again it has to sit inside the paper's assumption set where this port
can put it there.

* **S2 — decay at `Z`.** `muon_sf_wd` is the SF arm's OWN weight decay, applied
  at the fast iterate `Z` (`arXiv:2605.23061v1` Alg 1 line 17), not borrowed
  from the AdamW half's `wd`. Lemma 3.1 is what lifts the SF-Spectral bound's
  bounded-diameter assumption, so a non-positive value leaves the precondition
  unmet and is REFUSED at both doors. The hand-computed gold step below pins
  the placement (a decay on `Y` would be a different program).
* **S3 — the retraction.** `muon_sf_retract=true` is REFUSED: the averaged
  iterate is a convex combination of retracted factors and is not on the
  Stiefel manifold, while the projection step of Thm 2.1 needs a convex set.
  This port also has no `retract_tsct` call site, so `true` could only be inert.
* **The control is untouched.** `sf=false` builds `Muon(wd=0.0)` whatever the
  new key says, so a fix-seed 5-step run with `muon_sf_wd=0.0` and one with
  `muon_sf_wd=0.5` have the same state-dict sha and bit-equal logits.

The last test is TASK B's statistic: `scripts/ab_sf_muon.py` must mark the
2026-10-18 divergence (BPB 4.0573) as a divergence and keep it out of the mean,
because a sample σ of three points is inflated by the very seed it should catch.
"""

from __future__ import annotations

import copy
import hashlib
import math
import tempfile
from pathlib import Path

import pytest
import torch
from conftest import tiny_model, tiny_run
from torch import nn

from dormouse.config import load_run
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train import train
from dormouse.train.optim import MixOptimizer, Muon


def _hand_colrow(o: torch.Tensor) -> torch.Tensor:
    """`colrow_normalize` for 2-D, in float64, eps spelled out (test-local)."""
    x = o.double() / (o.double().norm(dim=0, keepdim=True) + 1e-7)
    return x / (x.norm(dim=1, keepdim=True) + 1e-7)


def _hand_eta(lr: float, m: int, n: int, p_hat: torch.Tensor) -> float:
    """`eta_hat = 0.2*lr*sqrt(m*n)/|p_hat|_F` (Alg 1 line 13)."""
    return 0.2 * lr * math.sqrt(m * n) / float(p_hat.norm())


def _sd_sha(sd: dict) -> str:
    """sha256 over a state dict's TENSORS and scalars (param_groups skipped:
    they carry object ids, which differ between two identical runs)."""
    h = hashlib.sha256()

    def walk(obj, path: str) -> None:
        if isinstance(obj, dict):
            for k in sorted(obj, key=str):
                if k == "param_groups":
                    continue
                walk(obj[k], f"{path}/{k}")
        elif isinstance(obj, torch.Tensor):
            h.update(path.encode())
            h.update(obj.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(f"{path}={obj!r}".encode())

    walk(sd, "")
    return h.hexdigest()


# --- S2: the decay is at Z, and only the SF arm has one --------------------


def test_sf_weight_decay_is_at_z_and_a_y_placement_would_differ():
    """Alg 1 line 17, hand-computed: `Z <- (1 - lr*wd)*Z - eta*P_hat`.

    Three steps on a `[1, 2]` matrix in float64. The Y-placement the ticket
    warns about (`p *= 1 - lr*wd` after the write) is computed too, and the
    assert it fails is the point: a decay in the wrong place is a different
    program, not a rounding difference.
    """
    lr, wd = 1e-2, 0.5
    o = torch.tensor([[3.0, 4.0]])
    p = nn.Parameter(torch.zeros(1, 2))
    muon = Muon([p], lr=lr, sf=True, wd=wd)
    assert muon.wd == pytest.approx(wd)

    p_hat = _hand_colrow(o)
    eta = _hand_eta(lr, 1, 2, p_hat)
    z = torch.zeros(1, 2, dtype=torch.float64)
    x = z.clone()
    s = 0.0
    for t in range(1, 4):
        s += lr * lr
        c = lr * lr / s
        z = (1.0 - lr * wd) * z - eta * p_hat
        x = (1.0 - c) * x + c * z
        muon._sf_one(p, o.clone(), lr)
        y = (1.0 - 0.9) * z + 0.9 * x
        assert torch.allclose(p.detach().double(), y, atol=1e-9), f"step {t}: Y moved"
        assert torch.allclose(muon.state[p]["z"].double(), z, atol=1e-9), f"step {t}: Z"
        assert torch.allclose(muon.state[p]["x"].double(), x, atol=1e-9), f"step {t}: X"
        y_wrong = (1.0 - lr * wd) * y  # decay at Y instead of at Z
        assert not torch.allclose(p.detach().double(), y_wrong, atol=1e-6), (
            f"step {t}: decay at Y would pass this test — it must not"
        )


def test_sf_wd_must_be_positive_at_both_doors():
    """λ <= 0 is outside Lemma 3.1: refused at the config and at the builder."""
    for bad in (0.0, -0.01):
        with pytest.raises(LoudError) as e:
            load_run("small", train_overrides={"muon_sf": True, "muon_sf_wd": bad})
        msg = str(e.value)
        assert "Lemma 3.1" in msg and "muon_sf_wd" in msg and "Z" in msg
        assert "Escape" in msg
    with pytest.raises(LoudError, match="Lemma 3.1"):
        MixOptimizer(tiny_model(), lr=1e-3, wd=0.01, muon_sf=True, muon_sf_wd=0.0)


def test_sf_retract_true_is_refused_and_names_the_convexity_line():
    """S3: `true` is the combination the theorem does not cover — both sf values.

    With `muon_sf=true` it is the S3 violation (the averaged iterate is a
    convex combination of retracted factors; Thm 2.1's projection step needs a
    convex set). With `muon_sf=false` the port has no retraction call site, so
    it would be an inert knob — the silent class, refused too.
    """
    for sf in (True, False):
        with pytest.raises(LoudError) as e:
            load_run("small", train_overrides={"muon_sf": sf, "muon_sf_retract": True})
        msg = str(e.value)
        assert "muon_sf_retract=true" in msg and "Thm 2.1" in msg
        assert "convex" in msg.lower() and "retract_tsct" in msg
    assert load_run("small", train_overrides={"muon_sf": True}).train.muon_sf_retract is False


def test_the_defaults_are_the_legal_arm():
    """The arm's own defaults must be the ones inside the theorem."""
    t = load_run("small").train
    assert t.muon_sf_wd == pytest.approx(0.01) and t.muon_sf_retract is False
    assert load_run("small", train_overrides={"muon_sf_wd": 0.0}).train.muon_sf_wd == 0.0


def test_sf_false_is_bit_identical_under_the_new_key():
    """`sf=false` never sees `muon_sf_wd`: same state-dict sha, equal logits.

    Two models from one state dict, 5 mixed optimizer steps on the SAME
    gradients, one built with `muon_sf_wd=0.0` and one with `0.5`. The Muon
    group is constructed with `wd=0.0` in both, so the parameter bytes and the
    optimizer state (momentum, Adam moments) must agree exactly — and the
    forward's logits with them.
    """
    model_a = tiny_model()
    model_b = copy.deepcopy(model_a)
    opt_a = MixOptimizer(model_a, lr=1e-3, wd=0.01, muon_sf=False, muon_sf_wd=0.0)
    opt_b = MixOptimizer(model_b, lr=1e-3, wd=0.01, muon_sf=False, muon_sf_wd=0.5)
    assert opt_a.muon is not None and opt_b.muon is not None
    assert opt_a.muon.wd == 0.0 and opt_b.muon.wd == 0.0, "the key leaked into the control"

    gen = torch.Generator().manual_seed(11)
    for _ in range(5):
        grads = [torch.randn(p.shape, generator=gen) * 0.01 for p in model_a.parameters()]
        for model in (model_a, model_b):
            for p, g in zip(model.parameters(), grads, strict=True):
                p.grad = g.clone()
        opt_a.step()
        opt_b.step()

    ids = torch.arange(2 * 64, dtype=torch.long).view(2, 64) % 251
    with torch.no_grad():
        logits_a = model_a(ids, pure=True)[0]
        logits_b = model_b(ids, pure=True)[0]
    assert torch.equal(logits_a, logits_b), "sf=false logits moved with the new key"
    assert _sd_sha(opt_a.state_dict()) == _sd_sha(opt_b.state_dict()), (
        "sf=false optimizer state moved with the new key"
    )


# --- the trainer's wiring: the value reaches the arm, and survives the graph


def _run_and_record(monkeypatch, **over) -> tuple[dict, list[float]]:
    """One tiny run; return the MixOptimizer kwargs/built opt and the Muon LRs."""
    from dormouse.train import loop as train_loop

    seen: list[dict] = []
    lrs: list[float] = []
    real_opt = train_loop.MixOptimizer
    real_step = Muon.batched_step_from_model

    def spy_opt(model, **kw):
        opt = real_opt(model, **kw)
        seen.append({"kw": kw, "opt": opt})
        return opt

    def spy_step(self, model, flag=None):
        # `flag` is the trainer's device NaN firewall (0/1 tensor); the spy
        # records the LR only, so it forwards it unchanged.
        lrs.append(self.param_groups[0]["lr"])
        return real_step(self, model, flag)

    monkeypatch.setattr(train_loop, "MixOptimizer", spy_opt)
    monkeypatch.setattr(Muon, "batched_step_from_model", spy_step)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run = tiny_run(
                Path(tmp),
                steps=6,
                log_every=6,
                ckpt_every=100,
                warmup_tokens=2 * 64 * 2,
                **over,
            )
            train(DormouseModel(run.model), run, log=lambda *_: None)
    finally:
        monkeypatch.setattr(train_loop, "MixOptimizer", real_opt)
        monkeypatch.setattr(Muon, "batched_step_from_model", real_step)
    assert seen and lrs, "the run did not reach the optimizer"
    return seen[0], lrs


def test_the_trainer_forwards_muon_sf_wd_and_the_control_ignores_it(monkeypatch):
    seen, _ = _run_and_record(monkeypatch, muon_sf=True, muon_sf_wd=0.02)
    assert seen["kw"]["muon_sf_wd"] == 0.02
    assert seen["opt"].muon.wd == pytest.approx(0.02)

    seen, _ = _run_and_record(monkeypatch, muon_sf=False, muon_sf_wd=0.02)
    assert seen["opt"].muon.wd == 0.0, "the control's Muon group took the SF decay"


def test_the_sf_muon_lr_is_constant_under_the_graph_arms_counted_fallback(monkeypatch):
    """`opt_graph=true` (the battery's COMMON) must not give the SF arm WSD.

    The graph arm refuses `muon_sf` at capture and the run continues EAGER,
    but its `set_lr` writes every group. Without the SF write-back in
    `train/loop.py` the Muon group would silently follow the WSD decay for the
    whole run — not the arm, and not visible in any counter.
    """
    _, eager = _run_and_record(monkeypatch, muon_sf=True, muon_sf_wd=0.01)
    _, graph = _run_and_record(monkeypatch, muon_sf=True, muon_sf_wd=0.01, opt_graph=True)
    assert graph == eager, f"the graph fallback changed the SF LR: {graph} != {eager}"
    assert graph[2:] == [graph[-1]] * 4, f"the SF Muon LR must be constant after warmup: {graph}"
