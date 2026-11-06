"""`block_form`: the KDA loop and the vanilla transformer are first-class forms.

The 2026-11-06 control (`docs/reviews/2026-11-06-control-transformer.md`) beat
the loop by 0.126 BPB on every eval window at equal bytes, so the block form is
now a config choice rather than an assumption. This file gates the six claims
the task makes, each able to fail on its own:

a. `block_form="kda"` is BIT-IDENTICAL to the pre-change model (state/logits/
   loss/gradient sha256 captured on clean `main` `ff34bf7` before the edit);
b. `"transformer"` refuses `max_iter != 1` and every KDA-loop knob it cannot
   honour, with `Escape:`;
c. the `mirror_control` preset's parameter count matches the control script's
   13 478 752 to 1% (the 304-parameter delta is the shared final RMSNorm, where
   the control has an affine LayerNorm);
d. a checkpoint round-trips `strict=True` and a resume reproduces the next
   step's loss bit-exactly;
e. 20-step CPU runs of both forms are finite and take no host read in a step
   (`stepaudit`, which wraps `item`/`tolist`/`__bool__`/`__float__`/`__int__`,
   plus a direct `monkeypatch` spy on the forward);
f. `block_form` through `config.load_run` reaches the model — logits differ from
   the KDA form on the same seed and inputs, so the flag is not inert.

Resume identity (task 5) is gated here too: a checkpoint written by one form
refuses a resume on the other, naming `model.block_form`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math

import pytest
import torch
from conftest import BATCH, SEQ, tiny_cfg, tiny_model, tiny_run
from typer.testing import CliRunner

from dormouse import stepaudit
from dormouse.cli import app
from dormouse.config import ModelConfig, load_run
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.train import load_ckpt, train
from dormouse.train.loop import load_model_state

#: Captured on clean `main` `ff34bf7` (CPU, torch 2.13.0+cu129, one thread,
#: seed 1234, `conftest.tiny_model()`), BEFORE this file existed:
#: `/tmp/opencode/pin_blockform.py` — state, logits, loss and the named-gradient
#: hash with `grad is None` encoded as the literal `NONE`.
PIN_STATE_SHA = "06eecd1d36bcf49ea3eb27247c32409487c3cf667fe48be1108e03cae6405b8a"
PIN_LOGITS_SHA = "5de7160313c4db58b846061a299d4035ae84940f287276b380b012fdf846d9ba"
PIN_LOSS = 6.200047969818115
PIN_GRAD_SHA = "b6d64da3a7b2602fa1bf8f638f2ac0ed1e60ebb9c4355eb6788d26e5b53b36f6"

#: `scripts/control_transformer.py --d-model 304 --layers 12 --heads 8 --ffn 1216`
#: prints `params=13,478,752` (report §2). The instantiated port model is 304
#: lower: the control's final affine LayerNorm (608) minus the shared RMSNorm
#: (304), which the task says to reuse. 0.0023%, far inside the 1% gate.
CONTROL_PARAMS = 13_478_752
MIRROR_PARAMS = 13_478_448


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().to(torch.float32).contiguous().numpy().tobytes()).hexdigest()


def _state_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().to(torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def _grad_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, p in sorted(model.named_parameters()):
        h.update(k.encode())
        if p.grad is None:
            h.update(b"NONE")
        else:
            h.update(p.grad.detach().to(torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def _transformer_cfg(**over) -> ModelConfig:
    """The tiny fixture as a legal vanilla transformer (`max_iter=1`, no arms)."""
    base = dict(
        block_form="transformer",
        max_iter=1,
        loop_blocks=1,
        n_layers=2,
        use_kda=False,
        use_engram=False,
        use_tsct=False,
        moe_topk=0,
        n_experts=1,
        bf16=False,
    )
    base.update(over)
    return tiny_cfg(**base)


def _transformer_run(tmp_path, **train_over):
    tmp_path.mkdir(parents=True, exist_ok=True)
    run = tiny_run(tmp_path, **train_over)
    return dataclasses.replace(run, model=_transformer_cfg())


def _batch(seq: int = SEQ) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(BATCH * seq, dtype=torch.long).view(BATCH, seq) % 251
    return ids, (ids + 1) % 251


# --- (a) the KDA form is bit-identical -------------------------------------


def test_kda_form_is_bit_identical_to_clean_main():
    """Gate a: selecting `"kda"` (the default) changes no number, no gradient."""
    threads, precision = torch.get_num_threads(), torch.get_float32_matmul_precision()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    try:
        torch.manual_seed(1234)
        model = tiny_model()
        assert model.cfg.block_form == "kda" and model.cfg.n_layers == 0
        x, y = _batch()
        logits, loss, _ = model(x, y)
        assert loss is not None
        loss.backward()
        assert _state_sha(model) == PIN_STATE_SHA, "the KDA state dict moved"
        assert _sha(logits) == PIN_LOGITS_SHA, "the KDA logits moved"
        assert float(loss.detach()) == PIN_LOSS
        assert _grad_sha(model) == PIN_GRAD_SHA, "the KDA gradients moved"
    finally:
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision(precision)


# --- (b) the transformer refuses what it cannot honour ---------------------


@pytest.mark.parametrize(
    ("over", "named"),
    [
        ({"max_iter": 2}, "max_iter"),
        ({"use_kda": True}, "use_kda"),
        ({"use_csa": True}, "use_csa"),
        ({"use_mhc": True}, "use_mhc"),
        ({"use_attnres": True}, "use_attnres"),
        ({"use_engram": True}, "use_engram"),
        ({"layerrope": True}, "layerrope"),
        ({"jepa": True, "jepa_weight": 0.1}, "jepa_weight"),
        ({"prelude_layers": 1}, "prelude_layers"),
    ],
)
def test_transformer_refuses_the_kda_only_arms(over, named):
    """Gate b: every flag the vanilla decoder cannot run is a LOUD `Escape:`."""
    with pytest.raises(LoudError) as e:
        _transformer_cfg(**over)
    msg = str(e.value)
    assert named in msg, msg
    assert "Escape:" in msg, msg
    assert "block_form='transformer'" in msg, msg


def test_the_refusals_hold_at_the_model_seam_too():
    """A config mutated after construction must not build a lying model (ADR-0019)."""
    cfg = _transformer_cfg()
    cfg.use_csa = True
    with pytest.raises(LoudError, match="use_csa"):
        DormouseModel(cfg)
    cfg.use_csa = False
    cfg.layerrope = True
    with pytest.raises(LoudError, match="layerrope"):
        DormouseModel(cfg)


def test_kda_refuses_an_explicit_n_layers():
    """`n_layers` is a transformer field; on the loop it is an inert key."""
    with pytest.raises(LoudError) as e:
        tiny_cfg(n_layers=4)
    assert "n_layers" in str(e.value) and "Escape:" in str(e.value)


# --- (c) the mirror preset prices like the control ------------------------


def test_mirror_control_matches_the_control_scripts_parameter_count():
    """Gate c: 13 478 448 vs the control's 13 478 752 (0.0023% < 1%)."""
    run = load_run("mirror_control")
    assert run.model.block_form == "transformer"
    assert run.model.n_layers == 12
    n = param_count(DormouseModel(run.model))
    assert n == MIRROR_PARAMS
    assert abs(n - CONTROL_PARAMS) / CONTROL_PARAMS <= 0.01
    r = CliRunner().invoke(app, ["info", "mirror_control"])
    assert r.exit_code == 0, r.output
    total = int(r.output.split(" params")[0].split()[-1].replace(",", ""))
    assert total == MIRROR_PARAMS, f"`info` printed {total}"


# --- (d) checkpoint + resume ----------------------------------------------


def test_transformer_checkpoint_round_trips_strict(tmp_path):
    """Gate d: every transformer key travels and the loader has nothing to say."""
    run = _transformer_run(tmp_path, steps=2, ckpt_every=2, log_every=10)
    torch.manual_seed(3)
    model = DormouseModel(run.model)
    train(model, run, log=lambda *_: None)
    ck, stored = load_ckpt(run)
    assert stored.model.block_form == "transformer"
    assert stored.model.n_layers == 2
    again = DormouseModel(run.model)
    result = load_model_state(again, run.train, ck["model"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys, result
    for k, v in again.state_dict().items():
        assert torch.equal(v, ck["model"][k]), k


def test_resume_reproduces_the_next_step_loss_bit_exactly(tmp_path):
    """Gate d: step 3 after a 2-step checkpoint == step 3 of an unbroken run."""
    torch.manual_seed(3)
    straight = _transformer_run(tmp_path / "a", steps=3, ckpt_every=100, log_every=100)
    report_a = train(DormouseModel(straight.model), straight, log=lambda *_: None)

    torch.manual_seed(3)
    first = _transformer_run(tmp_path / "b", steps=2, ckpt_every=2, log_every=100)
    train(DormouseModel(first.model), first, log=lambda *_: None)
    longer = dataclasses.replace(first, train=dataclasses.replace(first.train, steps=3))
    report_b = train(DormouseModel(longer.model), longer, resume=True, log=lambda *_: None)
    assert math.isfinite(report_a["loss"]) and math.isfinite(report_b["loss"])
    assert report_b["loss"] == report_a["loss"], (report_b["loss"], report_a["loss"])


def test_resume_refuses_a_block_form_switch(tmp_path):
    """Task 5: the form is resume identity, like `n_experts` — LOUD, not silent."""
    kda = tiny_run(tmp_path, steps=2, ckpt_every=2, log_every=10)
    torch.manual_seed(1)
    train(DormouseModel(kda.model), kda, log=lambda *_: None)
    switched = dataclasses.replace(kda, model=_transformer_cfg())
    assert "model.block_form" in switched.diff_keys(kda)
    with pytest.raises(LoudError) as e:
        load_ckpt(switched)
    msg = str(e.value)
    assert "resume refused" in msg and "model.block_form" in msg, msg


# --- (e) 20 CPU steps, both forms, finite and no host read ----------------


def _audited_20_steps(tmp_path, model_cfg, tag):
    torch.manual_seed(0)
    (tmp_path / tag).mkdir(parents=True, exist_ok=True)
    run = tiny_run(tmp_path / tag, steps=20, log_every=5, ckpt_every=1000)
    run = dataclasses.replace(run, model=model_cfg)
    audit = stepaudit.install()
    try:
        report = train(DormouseModel(model_cfg), run, log=lambda *_: None)
    finally:
        stepaudit.uninstall()
    return report, audit.records


@pytest.mark.parametrize("form", ["kda", "transformer"])
def test_twenty_cpu_steps_are_finite_and_take_no_host_read(tmp_path, form):
    """Gate e: a real 20-step run of each form; non-report steps read nothing."""
    cfg = tiny_cfg(moe_topk=0) if form == "kda" else _transformer_cfg()
    report, records = _audited_20_steps(tmp_path, cfg, form)
    assert math.isfinite(report["loss"]), report["loss"]
    assert report["opt_steps"] == 20
    assert len(records) >= 20
    assert any(r.report for r in records), "no log cadence fired, gate is vacuous"
    for rec in records:
        assert sum(rec.device_reads.values()) == 0, "a CPU run has no device tensors"
        if rec.report:
            continue
        unexpected = [
            f"{n}x {frame}"
            for frame, n in rec.frames.items()
            if not any(m in frame for m in stepaudit.CPU_ONLY_FRAMES)
        ]
        assert not unexpected, f"{form} step {rec.index} read the host: {unexpected}"


def test_each_forward_survives_an_item_tolist_bool_spy(monkeypatch):
    """Gate e, direct: `item`/`tolist`/`__bool__`/`__int__`/`__float__` are refused."""

    def boom(*_a, **_k):
        raise AssertionError("host sync in a block_form forward")

    for name in ("item", "tolist", "__bool__", "__int__", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, boom, raising=True)
    x, _ = _batch()
    for cfg in (tiny_cfg(moe_topk=0), _transformer_cfg()):
        with torch.no_grad():
            DormouseModel(cfg)(x)


# --- (f) the flag reaches the model ---------------------------------------


def test_load_run_block_form_is_not_inert(tmp_path):
    """Gate f: same preset/seed/inputs, only `block_form` differs; logits move."""
    preset = tmp_path / "kda_base.toml"
    preset.write_text(
        "\n".join(
            [
                "d_model = 64",
                "n_heads = 4",
                "head_dim = 16",
                "d_ffn = 128",
                "vocab = 256",
                "max_seq_len = 512",
                "max_iter = 2",
                "loop_blocks = 1",
                "n_experts = 1",
                "moe_topk = 0",
                "rank = 8",
                "use_engram = false",
                "use_tsct = false",
                "bf16 = false",
            ]
        ),
        encoding="utf-8",
    )
    kda_run = load_run(str(preset))
    tr_run = load_run(
        str(preset),
        model_overrides={
            "block_form": "transformer",
            "max_iter": 1,
            "n_layers": 2,
            "use_kda": False,
        },
    )
    assert kda_run.model.block_form == "kda"
    assert tr_run.model.block_form == "transformer"
    x, _ = _batch(16)
    torch.manual_seed(11)
    kda = DormouseModel(kda_run.model)
    torch.manual_seed(11)
    tr = DormouseModel(tr_run.model)
    assert hasattr(tr, "transformer_blocks"), "load_run did not reach the model"
    assert not hasattr(kda, "transformer_blocks")
    with torch.no_grad():
        logits_k, _, _ = kda(x)
        logits_t, _, _ = tr(x)
    assert logits_k.shape == logits_t.shape
    assert not torch.equal(logits_k, logits_t), (
        "block_form is inert: both forms produced the same logits"
    )
