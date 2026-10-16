"""The LOOM arm (`2610.01153`): per-loop router conditioning + variance control.

Source: `docs/research/2026-10-16-new-tech-scan.md` §2.1 — the row that measures
a weight-shared loop as stuck at two iterations (hidden-state variance growth +
expert-selection collapse) and reports 9-12 stable loops at 100M-1.7B, our band.
The arm is a TRANSCRIPTION of that scan's summary, not of the paper's full
recipe: one learned per-iteration gate on the RMS-normalized write, zero-init,
so `loom = true` at init IS `loom = false` and the gate can only act after
gradient has moved it. Wiring only — no quality claim here; the A/B against a
fixed-depth control is the arbiter.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from conftest import BATCH, SEQ, tiny_cfg, tiny_model, tiny_run

from dormouse.config import ModelConfig, load_run, parse_preset
from dormouse.instrument import Counters
from dormouse.model import DormouseModel
from dormouse.train import open_streams, train
from dormouse.train.loop import evaluate


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    return ids, (ids + 1) % 251


def _loss(model: DormouseModel, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        _, loss, _ = model(x, y)
    assert loss is not None
    return loss


def _loom_run(tmp_path, **train_over) -> tuple[object, DormouseModel]:
    """A tiny run whose MODEL carries `loom = true` (conftest's run does not)."""
    run = dataclasses.replace(tiny_run(tmp_path, **train_over), model=tiny_cfg(loom=True))
    return run, DormouseModel(run.model)


def test_loom_default_off_and_the_parameter_tree_is_unchanged():
    """(a) Off by default: no gate parameter, no counter, the same loss."""
    assert ModelConfig().loom is False
    x, y = _batch()
    off = tiny_model()
    assert "loop.loom_gate" not in dict(off.named_parameters())
    counters = Counters()
    off(x, y, counters=counters)
    assert "loom" not in counters
    # Byte-parity with today: the default path carries today's parameter set and
    # today's forward, so two models built from ONE state_dict read the same loss
    # to the bit.
    twin = tiny_model()
    twin.load_state_dict(off.state_dict())
    assert torch.equal(_loss(off, x, y), _loss(twin, x, y))


def test_loom_at_init_is_identity():
    """(b) A zero-init gate: loom ON reads the loom-OFF loss, bit for bit."""
    x, y = _batch()
    off = tiny_model()
    on = tiny_model(loom=True)
    assert on.loop.loom_gate is not None
    assert torch.equal(on.loop.loom_gate, torch.zeros_like(on.loop.loom_gate))
    # Both from ONE state_dict: the gate is the only key the flag adds.
    loaded = on.load_state_dict(off.state_dict(), strict=False)
    assert list(loaded.missing_keys) == ["loop.loom_gate"], loaded
    assert not loaded.unexpected_keys, loaded
    a, b = _loss(off, x, y), _loss(on, x, y)
    assert torch.equal(a, b), (float(a), float(b))
    assert abs(float(a) - float(b)) < 1e-6


def test_the_gate_receives_gradient_and_moves_off_zero(tmp_path):
    """(c) After 20 steps the gate has a gradient and is no longer zero.

    Zero-init plus AdamW's decay cannot move a parameter whose gradient is
    exactly zero (`p *= (1 - lr·wd)` on 0 is 0), so a NON-zero gate is itself
    the proof that gradient arrived.
    """
    run, model = _loom_run(tmp_path, steps=20, log_every=20)
    train(model, run, log=lambda *_: None)
    grad = model.loop.loom_gate.grad
    assert grad is not None, "no gradient reached the LOOM gate"
    assert int(torch.count_nonzero(grad)) > 0, grad
    gate = model.loop.loom_gate.detach()
    assert int(torch.count_nonzero(gate)) > 0, gate


def test_thirty_steps_with_loom_decrease_the_loss(tmp_path):
    """(d) The arm trains: loss down, finite, no NaN, one gate touch per iter."""
    run, model = _loom_run(tmp_path, steps=30, log_every=10)
    lines: list[str] = []
    report = train(model, run, log=lines.append)
    losses = [float(ln.split("loss=")[1].split()[0]) for ln in lines if "loss=" in ln]
    assert len(losses) >= 3, lines
    assert all(math.isfinite(v) for v in losses)
    assert report["nan_skipped"] == 0
    assert losses[-1] < losses[0]
    # The counter is the arm's own evidence: one write per iteration, no more.
    assert report["counters"]["loom"] == report["counters"]["iter"] == 30 * 2


def test_loom_counter_is_present_only_when_the_arm_is_on():
    """(e) `loom=<iterations>` when on; the key does not exist when off."""
    x, y = _batch()
    on, off = tiny_model(loom=True), tiny_model()
    c_on, c_off = Counters(), Counters()
    on(x, y, counters=c_on)
    off(x, y, counters=c_off)
    assert c_on["loom"] == on.cfg.max_iter == c_on["iter"]
    assert "loom" not in c_off
    assert c_off["iter"] == off.cfg.max_iter


def test_load_run_accepts_the_key():
    """(f) The flag is a real ModelConfig field: preset, override and CLI seam."""
    assert load_run("small").model.loom is False
    assert load_run("small", model_overrides={"loom": True}).model.loom is True
    assert parse_preset("loom = true\n").loom is True


def test_depth_curve_is_wired_with_loom_on(tmp_path):
    """(g) Wiring only: depths 2 and 4 are finite and the curve's keys complete."""
    cfg = tiny_cfg(max_iter=4, loom=True)
    model = DormouseModel(cfg)
    torch.manual_seed(0)
    h0 = torch.randn(BATCH, SEQ, cfg.d_model)
    for depth in (2, 4):
        out, _, counters = model.loop(h0, None, None, None, counters=Counters(), depth=depth)
        assert torch.isfinite(out).all()
        assert counters["loom"] == counters["depth"] == depth
    run = dataclasses.replace(tiny_run(tmp_path, steps=1, eval=True, eval_batches=3), model=cfg)
    _train_stream, eval_stream = open_streams(run)
    assert eval_stream is not None
    curve = evaluate(model, eval_stream, 3, depths=True)
    assert sorted(curve) == [1, 2, 3, 4], sorted(curve)
    for ce, scored in curve.values():
        assert math.isfinite(ce)
        assert scored == 3 * BATCH * SEQ
