"""The trainer: the WSD schedule, the optimizer router, the eval line, the firewall."""

from __future__ import annotations

import math

import pytest
import torch
from conftest import BATCH, SEQ, tiny_run, write_corpus
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
    # The tail is sized in READS (`eval_batches × batch × (seq_len + 1)` = 390 B
    # here), so ALL THREE windows the config asks for fit; the old fixture
    # encoded the 384 B scored-byte tail that held only two (review round 2
    # R2-1).
    assert evals and f"over {3 * BATCH * SEQ} B (3 x {BATCH} x {SEQ})" in evals[0]
    assert "shortfall" not in evals[0]


def test_the_no_eval_tail_is_sized_in_reads(tmp_path):
    """R2-1: the default 20-window request scores 20 windows, not 19.

    One window READS `batch × (seq_len + 1)` bytes and scores `batch × seq_len`
    (the next-byte shift), so a tail carved as scored bytes held one window
    fewer: the default `20 × 8 × 512` asked for 81 920 B, one read is 4 104 B,
    only 19 fits, and 77 824 B were scored. The tail is now carved in reads.
    The 1 MiB cap is still a real shortfall, and the line must name it.
    """
    import dataclasses

    from conftest import tiny_cfg

    from dormouse.train.data import eval_read_bytes, open_streams
    from dormouse.train.loop import _eval_line

    run = dataclasses.replace(
        tiny_run(
            tmp_path,
            steps=1,
            eval=True,
            eval_every=1,
            eval_batches=20,
            batch=8,
            seq_len=512,
            data=str(tmp_path / "corpus.bin"),
        ),
        model=tiny_cfg(max_seq_len=512),
    )
    write_corpus(tmp_path / "corpus.bin", 2 << 20)
    _train, ev = open_streams(run)
    assert ev is not None
    assert eval_read_bytes(run.train) == 20 * 8 * (512 + 1) == 82_080
    assert ev.hi - ev.lo >= 20 * 8 * (512 + 1), "the tail must hold every read"

    class _Counting(nn.Module):
        """The bytes scored are the tokens this model is handed."""

        def __init__(self) -> None:
            super().__init__()
            self.seen = 0

        def forward(self, x, targets=None, counters=None, **kw):
            self.seen += x.numel()
            return None, torch.tensor(1.0), counters

    model = _Counting()
    line, _loss, scored = _eval_line(model, ev, run.train, run.model.max_iter, None, 1)
    assert eval_window_bytes(run.train) == 20 * 8 * 512 == 81_920
    assert scored == model.seen == 20 * 8 * 512
    assert "over 81920 B (20 x 8 x 512)" in line, line
    assert "shortfall" not in line

    # A deliberately tiny tail: the 1 MiB cap binds, so the honest shortfall
    # is still reported (COUNTED) rather than claimed away.
    capped = dataclasses.replace(run.train, eval_batches=4096)
    _train, ev = open_streams(dataclasses.replace(run, train=capped))
    assert ev is not None and ev.hi - ev.lo == 1 << 20
    model = _Counting()
    line, _loss, scored = _eval_line(model, ev, capped, run.model.max_iter, None, 1)
    assert scored == model.seen < eval_window_bytes(capped)
    assert "shortfall" in line and f"requested {eval_window_bytes(capped)} B" in line


def test_the_tail_eval_prints_only_the_bytes_it_scored(tmp_path):
    """M5: the no-`--eval` tail is capped at 1 MiB, so a window larger than the
    tail used to wrap. The eval line used to print the REQUESTED window; it must
    print the bytes the pass actually scored and name the shortfall (COUNTED)."""
    import dataclasses

    from conftest import tiny_cfg

    from dormouse.train.data import open_streams
    from dormouse.train.loop import _eval_line

    # The audit's own shape: 2048 x 10 x 512 = 10 MiB requested against the
    # 1 MiB tail. 204 windows fit exactly (204 x 5130 B read <= 2^20), so the
    # pass scores 204 x 5120 B with no wrapped re-read.
    run = dataclasses.replace(
        tiny_run(
            tmp_path,
            steps=1,
            eval=True,
            eval_every=1,
            eval_batches=2048,
            batch=10,
            seq_len=512,
        ),
        model=tiny_cfg(max_seq_len=512),
    )
    write_corpus(tmp_path / "corpus.bin", 3 << 20)  # > the 1 MiB carve
    _train, ev = open_streams(run)
    assert ev is not None
    requested = eval_window_bytes(run.train)
    assert requested == 10 * (1 << 20), "the fixture must exceed the tail by 10x"

    class _Counting(nn.Module):
        """The bytes scored are the tokens this model is handed."""

        def __init__(self):
            super().__init__()
            self.seen = 0

        def forward(self, x, targets=None, counters=None, **kw):
            self.seen += x.numel()
            return None, torch.tensor(1.0), counters

    model = _Counting()
    line, _loss, scored = _eval_line(model, ev, run.train, run.model.max_iter, None, 1)
    assert scored == model.seen == 204 * 10 * 512
    assert scored <= 1 << 20, "the count must not claim more bytes than the tail holds"
    assert f"over {scored} B" in line
    assert "shortfall" in line and f"requested {requested} B" in line
    assert f"the eval stream holds {1 << 20} B" in line


def test_only_whole_eval_windows_are_scored_and_the_leftover_is_named(tmp_path):
    """M5: one window reads `batch × (seq_len + 1)` bytes; score only wholes.

    160 B with `batch × seq_len == 512` (batch 1) is below one read (513) and
    must score 0 B and say why; 672 B with batch 2 / seq_len 256 holds one read
    (514), scores 512 B, and says the other 160 B were left over.
    """
    from dormouse.config import TrainConfig
    from dormouse.train.data import ByteStream
    from dormouse.train.loop import _eval_batches, _eval_line

    class _Counting(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen = 0

        def forward(self, x, targets=None, counters=None, **kw):
            self.seen += x.numel()
            return None, torch.tensor(1.0), counters

    class _Range:
        """160 B with batch × seq_len == 512 — below even one 513 B read."""

        batch, seq_len, lo, hi = 1, 512, 0, 160

        def rewind(self) -> None:
            pass

        def next_batch(self):  # pragma: no cover - must never be called
            raise AssertionError("a range with no whole window must not be read")

    stub = _Range()
    assert _eval_batches(stub, 4) == 0
    zero_line, _loss, zero = _eval_line(
        _Counting(), stub, TrainConfig(batch=1, seq_len=512), 1, None, 1
    )
    assert zero == 0, "a range with no whole read must score nothing"
    assert "over 0 B" in zero_line and "no whole window fits" in zero_line
    assert "holds 160 B" in zero_line and "left over 160 B" in zero_line

    data = tmp_path / "long.bin"
    data.write_bytes(bytes(range(256)) * 3 + b"x" * 160)
    assert data.stat().st_size == 928
    stream = ByteStream(data, 2, 256, hi=672)
    cfg = TrainConfig(batch=2, seq_len=256, eval_batches=4)
    model = _Counting()
    line, _loss, scored = _eval_line(model, stream, cfg, 1, None, 1)
    assert scored == model.seen == 512
    assert "over 512 B (1 x 2 x 256, requested 4)" in line
    assert "holds 672 B" in line and "left over 160 B" in line


def test_a_tiny_nvfp4_block_is_counted_on_the_checkpoint_line(tmp_path):
    """M3: a real save surfaces the zero-scale block count, not just a test tally.

    A factor matrix spanning more than e4m3's ~18 octaves loses its smallest
    16-block even after the global prescale. `save_ckpt` used to call
    `pack_tsct(v, rung)` with no counters, so `nvfp4_zero_scale_blocks` reached
    no log. The checkpoint line must carry it — and the common path must not
    grow a field it never used.
    """
    import dataclasses

    from conftest import tiny_cfg

    from dormouse.model.tsct import is_factor_key

    run = dataclasses.replace(
        tiny_run(tmp_path, steps=1, log_every=1, ckpt_every=1),
        model=tiny_cfg(quant="nvfp4"),
    )
    model = DormouseModel(run.model)
    params = dict(model.named_parameters())
    for key in sorted(params):
        if is_factor_key(key) and params[key].numel() >= 32:
            with torch.no_grad():
                params[key].zero_()
                flat = params[key].reshape(-1)
                flat[0] = 1.0  # a block at e4m3's top after the prescale
                flat[16] = 1e-40  # ...and one ~130 octaves below it, lost
            # Freeze it: one optimizer step with lr=0.01 would overwrite 1e-40
            # with an O(lr) update and hide the fixture, not the defect.
            params[key].requires_grad_(False)
            break
    else:  # pragma: no cover - the tiny model always has a factor
        raise AssertionError("the fixture has no factor matrix")
    lines: list[str] = []
    report = train(model, run, log=lines.append)
    ckpts = [ln for ln in lines if ln.startswith("ckpt ")]
    assert ckpts and "nvfp4_zero_scale_blocks=1" in ckpts[0], ckpts
    assert report["counters"]["nvfp4_zero_scale_blocks"] == 1

    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    clean_run = dataclasses.replace(
        tiny_run(clean_dir, steps=1, log_every=1, ckpt_every=1),
        model=tiny_cfg(quant="nvfp4"),
    )
    clean_lines: list[str] = []
    train(DormouseModel(clean_run.model), clean_run, log=clean_lines.append)
    clean_ckpts = [ln for ln in clean_lines if ln.startswith("ckpt ")]
    assert clean_ckpts and "nvfp4_zero_scale_blocks" not in clean_ckpts[0]


def test_a_missing_lfb_sidecar_is_loud_on_a_training_resume(tmp_path):
    """M7/R2-2: a RESUME must not continue with bias=0 silently.

    With `moe_lb_coef > 0` the bias steers routing; the sidecar is the only
    copy of it, and a resume integrates from whatever it loads. Refusing is the
    reading that cannot mislead — for a training resume only; the readers load
    with a COUNTED line (`test_lfb.py`).
    """
    import dataclasses

    from conftest import tiny_cfg

    from dormouse.train.loop import ckpt_paths, lfb_path, load_model_half, load_model_state

    run = dataclasses.replace(
        tiny_run(tmp_path, steps=2, log_every=10, ckpt_every=2, ckpt_name="lfb"),
        model=tiny_cfg(moe_lb_coef=0.05),
    )
    train(DormouseModel(run.model), run, log=lambda *_: None)
    weights, _snap = ckpt_paths(run.train)
    lfb_path(run.train).unlink()
    fresh = DormouseModel(run.model)
    with pytest.raises(LoudError, match="LFB sidecar"):
        load_model_state(fresh, run.train, load_model_half(weights), resume=True)
    # The same checkpoint loads as a READER: no objective to continue, and the
    # zeroed controller is on the log (R2-2: a LOUD here bricks every reader).
    lines: list[str] = []
    load_model_state(fresh, run.train, load_model_half(weights), log=lines.append)
    assert any("lfb: no sidecar, controller zeroed" in line for line in lines), lines


def test_a_missing_lfb_sidecar_is_counted_when_the_controller_is_off(tmp_path):
    """The inert case: no sidecar is reported (COUNTED), not a crash."""
    import dataclasses

    from conftest import tiny_cfg

    from dormouse.train.loop import ckpt_paths, lfb_path, load_model_half, load_model_state

    run = dataclasses.replace(
        tiny_run(tmp_path, steps=2, log_every=10, ckpt_every=2, ckpt_name="lfb"),
        model=tiny_cfg(moe_lb_coef=0.0),
    )
    train(DormouseModel(run.model), run, log=lambda *_: None)
    weights, _snap = ckpt_paths(run.train)
    lfb_path(run.train).unlink()
    lines: list[str] = []
    load_model_state(
        DormouseModel(run.model), run.train, load_model_half(weights), log=lines.append
    )
    assert any("lfb: no sidecar, controller zeroed" in line for line in lines)


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
