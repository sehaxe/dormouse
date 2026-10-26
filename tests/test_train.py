"""The trainer: the WSD schedule, the optimizer router, the eval line, the firewall."""

from __future__ import annotations

import math

import pytest
import torch
from conftest import BATCH, SEQ, tiny_cfg, tiny_run
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
    """burn's routing policy, mirrored: linear maps to Muon, the excluded names to AdamW."""
    model = DormouseModel(tiny)
    assert is_muon_param("loop.attn.q_proj.weight", torch.zeros(64, 64))
    assert is_muon_param("loop.out_proj.u", torch.zeros(64, 8))
    assert not is_muon_param("loop.engram.tables", torch.zeros(3, 64, 8))
    assert not is_muon_param("loop.out_proj.s", torch.zeros(8))
    assert not is_muon_param("loop.residual_scale", torch.zeros(1))
    # orthogonalization hurts them (burn's measured routing policy, ADR-listed)
    assert not is_muon_param("embed.weight", torch.zeros(256, 8))
    assert not is_muon_param("lm_head.weight", torch.zeros(256, 8))
    assert not is_muon_param("loop.controller.gates", torch.zeros(3, 8))
    assert not is_muon_param("loop.experts.router.weight", torch.zeros(8, 8))
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


def test_a_nan_skipped_step_drops_the_pending_host_rows(tmp_path):
    """Audit MINOR 13: the skip branch never calls `host_step`, so the leaves a
    NaN step built would sit in `Engram._pending` until the next REAL step (and
    their rows be re-deduped there) — growth proportional to the NaN count."""
    from dormouse.model.engram import Engram

    run = tiny_run(tmp_path, steps=1, log_every=1)
    model = _NaNModel(DormouseModel(run.model))
    engrams = [m for m in model.modules() if isinstance(m, Engram)]
    assert engrams, "the fixture needs an Engram to leak through"
    for m in engrams:
        # The host substrate is where `_pending` is written at all: the in-VRAM
        # read appends nothing. `to_host` directly, because the trainer's
        # `host_tables` is a CUDA-budget refusal and this gate is CPU-only.
        m.to_host()
    report = train(model, run, log=lambda *_: None)
    assert report["nan_skipped"] == 1
    assert all(m._pending == [] for m in engrams), "a skipped step must drop its pending rows"


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


def test_torch_compile_flag_wraps_the_model_after_resume(tmp_path, monkeypatch):
    """The wiring gate: with `torch_compile=true` the trainer hands the model
    to `torch.compile(dynamic=True)` — once, after any resume load (the
    `_orig_mod.` prefix rule). `torch.compile` itself is patched out: this is
    a wiring test, not a compiler test."""
    calls: list[dict] = []
    real = torch.compile

    def spy(mod, **kw):
        calls.append(kw)
        return real(mod, **kw)

    monkeypatch.setattr(torch, "compile", spy)
    run = tiny_run(tmp_path, steps=2, log_every=2, torch_compile=True)
    lines: list[str] = []
    report = train(DormouseModel(run.model), run, log=lines.append)
    assert len(calls) == 1 and calls[0].get("dynamic") is True
    assert any("torch_compile" in s for s in lines)
    assert math.isfinite(report["loss"])


def test_torch_compile_off_by_default(tmp_path, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(torch, "compile", lambda *a, **k: calls.append(1))
    run = tiny_run(tmp_path, steps=1, log_every=1)
    train(DormouseModel(run.model), run, log=lambda *_: None)
    assert calls == []


def test_init_from_carries_the_checkpoints_lfb_controller(tmp_path):
    """`--init-from` must not silently switch the routing controller on.

    The LFB bias is controller state that is NOT a parameter and NOT in the
    model half (a non-persistent buffer), so no lr freezes it: a cold integrator
    started over a converged, skewed router winds up. That is the whole
    `train --init-from mix1` explosion: `train medium`'s preset has
    `moe_lb_coef=0.01`, the checkpoint trained with 0, and the bias reached
    ±4.8 in ~480 forwards — with the weights frozen (lr=1e-12) the loss read
    1.27 -> 1.50 -> 1.1e7 by step 300 (`logs/diag_lr0.log`). The init file's
    own snapshot is the authority on what the model is; the seam carries its
    coefficient, LOUD, instead of the preset's.
    """
    import dataclasses

    src_dir = tmp_path / "src"
    src = tiny_run(tmp_path, steps=1, log_every=1, ckpt_every=1, out_dir=str(src_dir))
    src = dataclasses.replace(src, model=dataclasses.replace(src.model, moe_lb_coef=0.0))
    train(DormouseModel(src.model), src, log=lambda *_: None)
    ckpt = src_dir / "t.pt"
    assert ckpt.is_file() and ckpt.with_suffix(".config.toml").is_file()

    dst_dir = tmp_path / "dst"
    dst = tiny_run(tmp_path, steps=1, log_every=1, ckpt_every=1, out_dir=str(dst_dir))
    dst = dataclasses.replace(dst, model=dataclasses.replace(dst.model, moe_lb_coef=0.01))
    model = DormouseModel(dst.model)
    lines: list[str] = []
    train(model, dst, init_from=str(ckpt), log=lines.append)

    moe = next(m for m in model.modules() if hasattr(m, "lfb_bias"))
    assert moe.lfb_coef == 0.0, (
        "init-from kept the preset's moe_lb_coef=0.01 over a checkpoint trained "
        "with the controller off: the bias integrates from zero on a converged "
        "router and routing collapses"
    )
    assert any("moe_lb_coef" in ln for ln in lines), lines
    assert "moe_lb_coef = 0.0" in (dst_dir / "t.config.toml").read_text()


def test_lfb_integrates_without_bound_when_the_load_cannot_change():
    """The mechanism: the LFB bias is an open-loop integrator once routing is fixed.

    `_lfb_step` moves every expert by `±moe_lb_coef` per routed forward from the
    CUMULATIVE counts (`model/experts.py`). When the load cannot answer — a
    converged router under `--init-from`, whose top-k is pinned by weights the
    bias does not reach until it is several logits tall — the sign never flips
    and the bias ramps linearly. It is not a parameter and no lr freezes it:
    the diag4 run (lr=1e-12, weights frozen) measured ±4.8 after ~520 forwards
    and the loss went 1.27 -> 5.1e6 by step 173; `moe_lb_coef=0` on the same
    checkpoint and data stayed 1.2-1.5 for all 400 steps (`logs/diag_lr0.log`,
    forward-only repro). That is why the init seam must carry the checkpoint's
    coefficient rather than a preset's.
    """
    from dormouse.instrument import Counters

    cfg = tiny_cfg(n_experts=8, moe_topk=2, moe_lb_coef=0.05)
    moe = next(m for m in DormouseModel(cfg).modules() if hasattr(m, "lfb_bias"))
    # the frozen load: the same two experts every forward, 64 tokens each
    index = torch.tensor([[0, 1]] * 64)
    counters = Counters()
    for _ in range(100):
        moe._lfb_step(index, counters)
    b = moe.lfb_bias
    assert float(b[0]) < -4.0 and float(b[1]) < -4.0, b
    assert float(b[2:].min()) > 4.0, b
    assert counters["moe_lfb_updates"] == 100


def test_init_controller_is_not_on_the_fresh_path(tmp_path, monkeypatch):
    """The fix is the init seam only: a FRESH run never reaches it (bit-identity
    by construction, and the test fails if the call ever moves out of the
    `init_from` branch)."""
    from dormouse.train import loop

    def boom(*a, **k):  # pragma: no cover - the assertion is that it never runs
        raise AssertionError("init_controller ran on a fresh run")

    monkeypatch.setattr(loop, "init_controller", boom)
    run = tiny_run(tmp_path, steps=3, log_every=1)
    report = train(DormouseModel(run.model), run, log=lambda *_: None)
    assert math.isfinite(report["loss"])
