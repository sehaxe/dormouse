"""The four train4 gates: grad accumulation, EMA, tied embeddings, z-loss.

Each gate is written so its REVERT reddens it:

* `test_grad_accum_matches_a_wide_batch` — drop the `/ grad_accum` and the
  accumulated gradient is the SUM (params move ~2x); step the optimizer per
  micro-batch and both the schedule and the update count move. The gate compares
  `accum=2, batch=2` against `batch=4` after 5 steps at 1e-5, on the DROPLESS
  MoE (`moe_capacity=0`) because a capacity-padded dispatch lets OTHER rows in a
  forward decide which tokens get an expert, so the wide and micro forwards are
  not separable by construction — that is the honest data/model caveat, not a
  data-order one (`ByteStream.next_batch` is exactly the concatenation).
* `test_ema_off_is_bit_identical` / `test_ema_shadow_is_the_hand_computed_average`
  / `test_ema_is_what_eval_reads` / `test_ema_checkpoint_round_trip` — remove the
  shadow update and the hand-computed average fails; swap the eval to the raw
  params and the spy fails; stop writing `model_ema` and resume restarts cold.
* `test_tie_off_is_bit_identical` / `test_tie_saves_exactly_the_head` /
  `test_tie_gradient_is_the_sum_of_both_paths` — an accidental extra registered
  parameter shows in the count and the state dict; a head that does not share
  the matrix fails the sum.
* `test_z_loss_zero_is_bit_identical` / `test_z_loss_is_the_formula` /
  `test_z_loss_gradient_reaches_the_head` — a term added to the wrong place
  moves the logits or misses the head.

Plus the LOUD refusals each field owns (`test_refusals`).

CPU only: the suite runs with `CUDA_VISIBLE_DEVICES=""`.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from conftest import SEQ, TINY, write_corpus
from torch import nn

from dormouse.config import ModelConfig, RunConfig, TrainConfig, load_run, validate
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.train import load_ckpt, train
from dormouse.train.loop import load_model_half
from dormouse.train.optim import MixOptimizer

#: `TINY` is the port's small-model fixture; dropping the MoE capacity makes the
#: model row-independent (see the module docstring).
BASE = {**TINY, "moe_capacity": 0.0}


def _cfg(**over) -> ModelConfig:
    return ModelConfig(**{**BASE, **over})


def _with(cfg: ModelConfig, **over) -> ModelConfig:
    return dataclasses.replace(cfg, **over)


def _model(cfg: ModelConfig, seed: int = 0) -> DormouseModel:
    torch.manual_seed(seed)
    return DormouseModel(cfg)


def _run_cfg(tmp_path, name: str, cfg: ModelConfig, **over) -> RunConfig:
    data = write_corpus(tmp_path / "corpus.bin", 16384)
    tc = TrainConfig(
        steps=5,
        batch=2,
        seq_len=32,
        lr=1e-3,
        log_every=1000,
        ckpt_every=10**9,
        ckpt_name=name,
        out_dir=str(tmp_path / name),
        data=str(data),
        opt="adamw",
    )
    return RunConfig(source="tiny", model=cfg, train=dataclasses.replace(tc, **over))


def _state_equal(a: nn.Module, b: nn.Module, tol: float = 1e-5) -> float:
    worst = 0.0
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb), "the parameter trees differ in keys"
    for key in sa:
        worst = max(worst, float((sa[key] - sb[key]).abs().max()))
    assert worst <= tol, f"state dict differs by {worst:g}"
    return worst


# --- (1) gradient accumulation ---------------------------------------------


def test_grad_accum_matches_a_wide_batch(tmp_path):
    """`accum=2, batch=2` == `batch=4` at 1e-5 for 5 optimizer steps.

    The stream hands out consecutive `batch·(seq_len+1)` windows, so the two
    micro-batches ARE the wide window's rows in order. A dropless MoE keeps
    each row's forward independent, which is what makes the comparison a
    statement about the accumulation arithmetic and not about routing.
    """
    cfg = _cfg()
    wide = _run_cfg(tmp_path, "wide", cfg, batch=4, grad_accum=1)
    micro = _run_cfg(tmp_path, "micro", cfg, batch=2, grad_accum=2)

    m_wide = _model(cfg)
    r_wide = train(m_wide, wide, log=lambda *a, **k: None)
    m_micro = _model(cfg)
    r_micro = train(m_micro, micro, log=lambda *a, **k: None)

    assert r_wide["opt_steps"] == r_micro["opt_steps"] == 5
    assert r_wide["steps"] == r_micro["steps"] == 5
    # Positive evidence that accumulation really ran: 5 steps x 2 forwards.
    assert r_wide["counters"].get("forward") == 5
    assert r_micro["counters"].get("forward") == 10
    assert r_wide["loss"] == pytest.approx(r_micro["loss"], abs=1e-5)
    worst = _state_equal(m_wide, m_micro)
    print(
        f"grad_accum: 5 steps, worst |Δparam| = {worst:.3e}, losses "
        f"{r_wide['loss']:.6f}/{r_micro['loss']:.6f}"
    )


def test_grad_accum_skips_the_whole_group_on_one_bad_micro(tmp_path):
    """A non-finite micro-batch zeroes the group: no half-step, no param move."""
    cfg = _cfg()
    run = _run_cfg(tmp_path, "nangroup", cfg, steps=1, grad_accum=2)
    inner = _model(cfg)
    before = {k: v.detach().clone() for k, v in inner.named_parameters()}

    class _NaNSecondMicro(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner
            self.calls = 0

        def forward(self, x, targets=None, counters=None, **kw):
            logits, loss, counters = self.inner(x, targets, counters, **kw)
            self.calls += 1
            return logits, loss * (float("nan") if self.calls == 2 else 1.0), counters

    report = train(_NaNSecondMicro(inner), run, log=lambda *a, **k: None)
    assert report["nan_skipped"] == 1
    for name, p in inner.named_parameters():
        assert torch.equal(p, before[name]), name


def test_grad_accum_resume_skips_the_groups_bytes(tmp_path):
    """A resume fast-forwards `steps × batch × (seq+1) × grad_accum` bytes.

    With the wrong skip the resumed stream re-reads rows it trained on; the
    gate is the bit-identical continuation of an unbroken run, which is what
    the `1` in that product exists for.
    """
    cfg = _cfg()
    full = _run_cfg(tmp_path, "full", cfg, steps=4, batch=2, grad_accum=2)
    m_full = _model(cfg)
    r_full = train(m_full, full, log=lambda *a, **k: None)

    first = _run_cfg(tmp_path, "split", cfg, steps=2, batch=2, grad_accum=2, ckpt_every=2)
    m_first = _model(cfg)
    train(m_first, first, log=lambda *a, **k: None)
    # The checkpoint is written at step index 1, so resume starts at 2.
    resumed = dataclasses.replace(first, train=dataclasses.replace(first.train, steps=4))
    m_res = _model(cfg)
    # The model is rebuilt and loaded from the file, exactly as a CLI resume.
    r_res = train(m_res, resumed, resume=True, log=lambda *a, **k: None)
    assert r_res["opt_steps"] == 4
    worst = _state_equal(m_full, m_res)
    print(f"grad_accum resume: worst |Δparam| = {worst:.3e}")
    assert r_full["opt_steps"] == r_res["opt_steps"] == 4


# --- (2) EMA ----------------------------------------------------------------


def test_ema_off_is_bit_identical(tmp_path):
    """`ema_decay=0` adds no shadow and moves no number."""
    cfg = _cfg()
    a = _run_cfg(tmp_path, "off_a", cfg, steps=2)
    b = _run_cfg(tmp_path, "off_b", cfg, steps=2, ema_decay=0.0)
    ma = _model(cfg)
    ra = train(ma, a, log=lambda *x, **k: None)
    mb = _model(cfg)
    rb = train(mb, b, log=lambda *x, **k: None)
    assert ra["ema"] is None and rb["ema"] is None
    assert ra["loss"] == rb["loss"]
    for key in ma.state_dict():
        assert torch.equal(ma.state_dict()[key], mb.state_dict()[key]), key

    ids = torch.arange(2 * SEQ, dtype=torch.long).view(2, SEQ) % 251
    with torch.no_grad():
        la, _, _ = ma(ids, (ids + 1) % 251)
        lb, _, _ = mb(ids, (ids + 1) % 251)
    assert torch.equal(la, lb)


@pytest.mark.parametrize("accum", [1, 2])
def test_ema_shadow_is_the_hand_computed_average(tmp_path, accum):
    """After 3 OPTIMIZER steps the shadow is exactly the recursive `0.9·s + 0.1·p`.

    Parametrized over `grad_accum`: with two micro-batches the shadow must still
    update once per GROUP (the recorder sees one optimizer step per group), not
    once per forward.
    """
    cfg = _cfg()
    run = _run_cfg(tmp_path, "ema", cfg, steps=3, lr=0.01, grad_accum=accum, ema_decay=0.9)
    model = _model(cfg)
    trajectory = [{k: v.detach().clone() for k, v in model.named_parameters()}]
    original = MixOptimizer.step

    def recording_step(self, flag=None):
        original(self, flag)
        trajectory.append({k: v.detach().clone() for k, v in self.model.named_parameters()})

    MixOptimizer.step = recording_step
    try:
        report = train(model, run, log=lambda *a, **k: None)
    finally:
        MixOptimizer.step = original
    shadow = report["ema"]
    assert shadow is not None and len(trajectory) == 4
    assert report["counters"].get("forward") == 3 * accum
    worst = 0.0
    for name, value in shadow.items():
        expected = trajectory[0][name].clone()
        for after in trajectory[1:]:
            expected = 0.9 * expected + 0.1 * after[name]
        worst = max(worst, float((value - expected).abs().max()))
    print(f"ema: 3 steps, worst |shadow - hand| = {worst:.3e}")
    assert worst <= 1e-6


def test_ema_is_what_eval_reads(tmp_path, monkeypatch):
    """The eval forward runs on the shadow, not on the raw parameters."""
    import dormouse.train.loop as loopmod

    cfg = _cfg()
    run = _run_cfg(
        tmp_path,
        "emaeval",
        cfg,
        steps=3,
        lr=0.01,
        ema_decay=0.9,
        eval=True,
        eval_every=3,
        eval_batches=2,
    )
    model = _model(cfg)
    seen: list[torch.Tensor] = []

    def spy(model, stream, batches, device=None, **kw):
        seen.append(model.embed.weight.detach().clone())
        return (1.0, 1, 1.0, [])

    monkeypatch.setattr(loopmod, "evaluate", spy)
    report = train(model, run, log=lambda *a, **k: None)
    assert len(seen) == 1
    shadow = report["ema"]["embed.weight"]
    assert torch.equal(seen[0], shadow)
    assert not torch.equal(seen[0], model.embed.weight.detach())


def test_ema_checkpoint_round_trip_and_legal_decay_change(tmp_path):
    """The shadow is saved under `model_ema`, resumed, and a decay change is legal."""
    cfg = _cfg()
    run = _run_cfg(tmp_path, "emackpt", cfg, steps=3, lr=0.01, ema_decay=0.9, ckpt_every=3)
    model = _model(cfg)
    report = train(model, run, log=lambda *a, **k: None)
    weights = tmp_path / "emackpt" / "emackpt.pt"
    assert weights.is_file()

    ck = torch.load(weights, map_location="cpu", weights_only=False)
    assert "model_ema" in ck and "model" in ck
    served = load_model_half(weights)
    raw = ck["model"]
    assert any(not torch.equal(served[name], raw[name]) for name in report["ema"])
    for name, value in report["ema"].items():
        assert torch.equal(served[name], value)
    loaded, _stored = load_ckpt(run)
    assert "model_ema" in loaded

    # `ema_decay` is a PROGRESS key: a changed value is a legal resume, because
    # the file carries both copies of the same trajectory.
    changed = dataclasses.replace(run, train=dataclasses.replace(run.train, ema_decay=0.5, steps=3))
    assert run.diff_keys(changed) == []
    resumed = _model(cfg)
    out = train(resumed, changed, resume=True, log=lambda *a, **k: None)
    assert out["opt_steps"] == 3
    for name, value in report["ema"].items():
        assert torch.equal(out["ema"][name], value), name


# --- (3) tied embeddings ----------------------------------------------------


def test_tie_off_is_bit_identical():
    a = _model(_cfg())
    b = _model(_with(_cfg(), tie_embeddings=False))
    assert list(a.state_dict()) == list(b.state_dict())
    for key in a.state_dict():
        assert torch.equal(a.state_dict()[key], b.state_dict()[key]), key
    ids = torch.arange(2 * SEQ, dtype=torch.long).view(2, SEQ) % 251
    with torch.no_grad():
        la, _, _ = a(ids, (ids + 1) % 251)
        lb, _, _ = b(ids, (ids + 1) % 251)
    assert torch.equal(la, lb)


def test_tie_saves_exactly_the_head():
    """A dense head is `d·vocab` parameters; tying removes exactly that."""
    untied = _model(_cfg(use_tsct=False))
    tied = _model(_with(_cfg(use_tsct=False), tie_embeddings=True))
    delta = param_count(untied) - param_count(tied)
    assert delta == untied.cfg.d_model * untied.cfg.vocab
    print(f"tie: param delta = {delta} (d_model*vocab = {untied.cfg.d_model * untied.cfg.vocab})")
    assert sum(p.numel() for p in tied.lm_head.parameters()) == 0
    keys = list(tied.state_dict())
    assert keys.count("embed.weight") == 1
    assert not any(k.startswith("lm_head.") for k in keys)


def test_tie_gradient_is_the_sum_of_both_paths():
    """The shared matrix's grad is the embedding path PLUS the head path."""
    cfg = _cfg(use_tsct=False)
    # The reference keeps two separate parameters holding the SAME matrix, so
    # its two gradients are the two paths' gradients of one function.
    ref = _model(cfg)
    ref.lm_head.weight.data = ref.embed.weight.data.clone()
    tied = _model(_with(cfg, tie_embeddings=True))
    tied.load_state_dict(
        {k: v for k, v in ref.state_dict().items() if not k.startswith("lm_head.")}
    )
    assert torch.equal(tied.embed.weight, ref.embed.weight)

    ids = torch.arange(2 * SEQ, dtype=torch.long).view(2, SEQ) % 251
    targets = (ids + 1) % 251
    _, loss_ref, _ = ref(ids, targets)
    loss_ref.backward()
    g_embed = ref.embed.weight.grad.detach().clone()
    g_head = ref.lm_head.weight.grad.detach().clone()
    assert g_embed.abs().sum() > 0 and g_head.abs().sum() > 0

    ref.zero_grad()
    _, loss_tied, _ = tied(ids, targets)
    loss_tied.backward()
    assert loss_tied.item() == pytest.approx(loss_ref.item(), rel=1e-6, abs=1e-6)
    worst = float((tied.embed.weight.grad - (g_embed + g_head)).abs().max())
    print(f"tie: worst |grad - (embed + head)| = {worst:.3e}")
    assert worst <= 1e-6


def test_tie_replaces_the_tsct_head():
    """On a TSCT body the tied head is the dense matrix; the low-rank head is gone."""
    untied = _model(_cfg(use_tsct=True))
    tied = _model(_with(_cfg(use_tsct=True), tie_embeddings=True))
    tied_head = sum(p.numel() for p in tied.lm_head.parameters())
    untied_head = sum(p.numel() for p in untied.lm_head.parameters())
    assert tied_head == 0 and untied_head > 0
    assert param_count(tied) == param_count(untied) - untied_head


# --- (4) z-loss -------------------------------------------------------------


def _logits(model: DormouseModel, ids: torch.Tensor, targets: torch.Tensor):
    return model(ids, targets)


def test_z_loss_zero_is_bit_identical():
    a = _model(_cfg(use_tsct=False))
    b = _model(_with(_cfg(use_tsct=False), z_loss=0.0))
    ids = torch.arange(2 * SEQ, dtype=torch.long).view(2, SEQ) % 251
    targets = (ids + 1) % 251
    la, lossa, _ = a(ids, targets)
    lb, lossb, _ = b(ids, targets)
    assert torch.equal(la, lb)
    assert torch.equal(lossa, lossb)


def test_z_loss_is_the_formula():
    """`loss(z) - loss(0) == coef · mean(logsumexp(logits)²)`; logits unchanged."""
    ids = torch.arange(2 * SEQ, dtype=torch.long).view(2, SEQ) % 251
    targets = (ids + 1) % 251
    zero = _model(_cfg(use_tsct=False))
    with_z = _model(_with(_cfg(use_tsct=False), z_loss=1e-3))
    logits0, loss0, _ = zero(ids, targets)
    logits1, loss1, counters = with_z(ids, targets)
    assert torch.equal(logits0, logits1)
    expected = loss0.detach() + 1e-3 * torch.logsumexp(logits0.detach(), dim=-1).pow(2).mean()
    assert loss1.detach() == pytest.approx(expected, rel=1e-6, abs=1e-7)
    z_term = 1e-3 * torch.logsumexp(logits0.detach(), dim=-1).pow(2).mean()
    print(
        f"z_loss: loss0={loss0.item():.9f} loss={loss1.item():.9f} "
        f"delta={(loss1 - loss0).item():.6e} expected_z={z_term.item():.6e}"
    )
    assert counters.get("z_loss") == 1
    # `pure=True` is the held-out number: the regulariser is skipped there.
    _, pure0, _ = zero(ids, targets, pure=True)
    _, pure1, _ = with_z(ids, targets, pure=True)
    assert torch.equal(pure0, pure1)


def test_z_loss_gradient_reaches_the_head():
    ids = torch.arange(2 * SEQ, dtype=torch.long).view(2, SEQ) % 251
    targets = (ids + 1) % 251
    zero = _model(_cfg(use_tsct=False))
    with_z = _model(_with(_cfg(use_tsct=False), z_loss=1e-3))
    with_z.load_state_dict(zero.state_dict())

    _, loss0, _ = zero(ids, targets)
    loss0.backward()
    g0 = zero.lm_head.weight.grad.detach().clone()

    _, loss1, _ = with_z(ids, targets)
    loss1.backward()
    g1 = with_z.lm_head.weight.grad.detach().clone()
    assert g1.abs().sum() > 0
    assert not torch.allclose(g0, g1)


# --- refusals ---------------------------------------------------------------


def test_refusals(tmp_path):
    with pytest.raises(LoudError, match="grad_accum"):
        load_run("small", train_overrides={"grad_accum": 0})
    with pytest.raises(LoudError, match="opt_graph"):
        load_run("small", train_overrides={"grad_accum": 2, "opt_graph": True})
    with pytest.raises(LoudError, match="ema_decay"):
        load_run("small", train_overrides={"ema_decay": 1.0})
    with pytest.raises(LoudError, match="ema_decay"):
        load_run("small", train_overrides={"ema_decay": -0.01})
    with pytest.raises(LoudError, match="muon_sf"):
        load_run("small", train_overrides={"ema_decay": 0.5, "muon_sf": True})
    with pytest.raises(LoudError, match="z_loss"):
        _cfg(z_loss=-1e-3)
    # The hand-built path (no `load_run`) must refuse too.
    cfg = _cfg()
    run = _run_cfg(tmp_path, "bad", cfg, grad_accum=0)
    with pytest.raises(LoudError, match="grad_accum"):
        train(_model(cfg), run, log=lambda *a, **k: None)

    # The SAFE combinations are decided, not left to discovery: accumulation
    # composes with halt, segmentation and host tables.
    # `halt` and `segment_len` each compose with accumulation; the pair is
    # already refused by `_validate_segment` on its own (two unvalidated axes),
    # so the gate asks for them separately.
    legal_halt = load_run(
        "small",
        model_overrides={"halt_weight": 0.01},
        train_overrides={"grad_accum": 2},
    )
    legal_seg = load_run("small", train_overrides={"grad_accum": 2, "segment_len": 64})
    assert legal_halt.train.grad_accum == 2 and legal_halt.model.halt_weight > 0
    assert legal_seg.train.segment_len == 64
    # z-loss and tying are loud on the byteflow shapes, where no lm_head runs.
    with pytest.raises(LoudError, match="use_byteflow"):
        _cfg(use_byteflow=True, z_loss=1e-3)
    with pytest.raises(LoudError, match="use_byteflow"):
        _cfg(use_byteflow=True, tie_embeddings=True)
    validate(ModelConfig())
