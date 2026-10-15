"""The trainer: the WSD schedule, the optimizer router, the eval line, the firewall."""

from __future__ import annotations

import math

import pytest
import torch
from conftest import BATCH, SEQ, tiny_run
from torch import nn

from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train import eval_window_bytes, is_muon_param, train, wsd_lr
from dormouse.train.optim import MixOptimizer


def test_wsd_warmup_peak_and_decay():
    lr = lambda s: wsd_lr(s, total_steps=100, peak_lr=1e-3, warmup_steps=10, decay_steps=20)  # noqa: E731
    assert lr(0) == pytest.approx(1e-4)
    assert lr(9) == pytest.approx(1e-3)
    assert lr(50) == pytest.approx(1e-3)
    assert lr(79) == pytest.approx(1e-3)
    assert lr(99) == pytest.approx(1e-4)  # decayed to 0.1x peak
    assert all(lr(s) <= 1e-3 for s in range(100))
    # The warmup is a TOKEN count: a bigger batch reaches the peak sooner.
    from dormouse.train import warmup_steps

    assert warmup_steps(2 * BATCH * SEQ, BATCH, SEQ) == 2
    assert warmup_steps(0, BATCH, SEQ) == 0


def test_optimizer_routing(tiny):
    """2-D (both dims > 1) goes to Muon; 1-D, the s-scales and the tables do not."""
    model = DormouseModel(tiny)
    assert is_muon_param("loop.attn.q_proj.weight", torch.zeros(64, 64))
    assert is_muon_param("loop.out_proj.u", torch.zeros(64, 8))
    assert not is_muon_param("loop.engram.tables", torch.zeros(3, 64, 8))
    assert not is_muon_param("loop.out_proj.s", torch.zeros(8))
    assert not is_muon_param("loop.residual_scale", torch.zeros(1))
    mix = MixOptimizer(model, lr=1e-3, wd=0.0)
    assert mix.routing["loop.engram.tables"] == "adamw"
    assert mix.routing["loop.out_proj.s"] == "adamw"
    assert mix.routing["loop.attn.q_proj.weight"] == "muon"
    assert mix.n_muon > 0 and mix.n_adamw > 0
    # `opt="adamw"` is the control: the same code with the routing off.
    allw = MixOptimizer(DormouseModel(tiny), lr=1e-3, wd=0.0, opt="adamw")
    assert allw.muon is None and all(g == "adamw" for g in allw.routing.values())


def test_muon_update_compresses_the_spectrum_and_is_not_a_copy():
    """The polar polynomial, with burn's Frobenius prescale (`orthogonalize`).

    What it DOES, measured here and consistent with the burn record: a heavy
    spectrum comes back compressed and bounded. What it does NOT do is reach
    `polar(X)` exactly — AGENTS §3.7: NS-3 sits ~1e-15 from the Stiefel
    MANIFOLD while being 14.5% from `polar(X)` in SPECTRUM at a 10:1 spread,
    because the polynomial suppresses tiny singular values instead of
    normalizing them. The gate is the property Muon's lr is calibrated
    against (a ~1-scale update with the spread squeezed out), not exactness.
    """
    from dormouse.train.optim import colrow_normalize, zeropower_via_ns

    torch.manual_seed(0)
    heavy = torch.randn(64, 8) @ torch.diag(torch.logspace(0, -2, 8))
    o = zeropower_via_ns(heavy, steps=5)
    assert o.shape == heavy.shape and not torch.allclose(o, heavy)
    s_in = torch.linalg.svdvals(heavy)
    s_out = torch.linalg.svdvals(o)
    assert float(s_in[0] / s_in[-1]) > 50.0, "the fixture must have a heavy spectrum"
    assert float(s_out[0] / s_out[-1]) < 2.0, "the spectrum must be compressed"
    assert 0.5 < float(s_out[-1]) and float(s_out[0]) < 1.2, "the update scale must be ~1"
    # Wide matrices take the same path (orthogonalized in the Gram orientation).
    wide = torch.randn(8, 64)
    ow = zeropower_via_ns(wide, steps=5)
    assert ow.shape == wide.shape and torch.linalg.svdvals(ow).max() <= 1.2
    n = colrow_normalize(heavy)
    assert torch.allclose(n.norm(dim=1), torch.ones(n.shape[0]), atol=1e-5)


def test_eval_line_prints_the_byte_window(tmp_path):
    """The window is `eval_batches x batch x seq_len` bytes, and it is PRINTED."""
    run = tiny_run(tmp_path, steps=2, log_every=1, eval=True, eval_every=2, eval_batches=3)
    lines: list[str] = []
    report = train(DormouseModel(run.model), run, log=lines.append)
    window = eval_window_bytes(run.train)
    assert window == 3 * BATCH * SEQ
    assert report["eval_window_bytes"] == window
    evals = [ln for ln in lines if ln.startswith("eval ")]
    assert evals and f"over {window} B (3 x {BATCH} x {SEQ})" in evals[0]


def test_log_line_carries_the_kda_arm_and_counters(tmp_path):
    run = tiny_run(tmp_path, steps=1, log_every=1, timers=True)
    lines: list[str] = []
    train(DormouseModel(run.model), run, log=lines.append)
    line = lines[0]
    assert "step=1" in line and "loss=" in line and "lr=" in line
    assert "kda=sdpa,n=2" in line, line
    assert "engram_rows=" in line and "rezero_write=" in line
    assert "ms/step" in line and "tok/s" in line


class _NaNModel(nn.Module):
    """A model whose loss is never finite — the firewall's own fixture."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x, targets=None, counters=None):
        logits, loss, counters = self.inner(x, targets, counters)
        return logits, loss * float("nan"), counters


def test_nan_step_is_skipped_and_counted_not_learned(tmp_path):
    run = tiny_run(tmp_path, steps=1, log_every=1)
    model = _NaNModel(DormouseModel(run.model))
    before = [p.detach().clone() for p in model.parameters()]
    report = train(model, run, log=lambda *_: None)
    assert report["nan_skipped"] == 1
    assert report["counters"]["nan_skipped"] == 1
    for p, b in zip(model.parameters(), before, strict=True):
        assert torch.equal(p, b), "a non-finite step must not touch a weight"


def test_two_all_nan_windows_stop_the_run(tmp_path):
    """The plan's `nan_windows >= 2` hard error: a drowning run must not exit 0."""
    run = tiny_run(tmp_path, steps=5, log_every=1)
    with pytest.raises(LoudError) as e:
        train(_NaNModel(DormouseModel(run.model)), run, log=lambda *_: None)
    msg = str(e.value)
    assert "entirely non-finite" in msg and "nan_windows>=2" in msg
    assert "readiness-plan.md" in msg
    assert "not exit 0" in msg


def test_loss_is_finite_and_the_loop_runs_the_configured_depth(tmp_path):
    run = tiny_run(tmp_path, steps=3, log_every=1)
    lines: list[str] = []
    report = train(DormouseModel(run.model), run, log=lines.append)
    assert math.isfinite(report["loss"])
    assert report["counters"]["iter"] == 3 * run.model.max_iter
