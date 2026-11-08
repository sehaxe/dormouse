"""`train.torch_compile` is an EXECUTOR switch, not a new experiment.

The wrapper `torch.compile(model, dynamic=True)` is applied AFTER the resume
load (`train/loop.py`), so every gate here is about the seam between the
wrapper and the code that expects an unwrapped `DormouseModel`:

* the checkpoint file never carries the wrapper's `_orig_mod.` prefix — in
  `model`, `model_ema` AND the `moe` section — and loads into an unwrapped
  model, which is what makes a resume across the compile boundary legal in
  both directions;
* the EMA shadow is CHECKPOINT-keyed and the same swap reaches a compiled
  forward;
* the optimizer built before the wrap still steps the same `Parameter`
  objects the compiled forward reads, so Muon/AdamW see no second spelling;
* the forward's arm counters and the eval's own counters survive the wrapper
  (graph breaks re-run the Python between recorded regions; a captured
  counter would go SILENT);
* eval windows are the TRAINING `batch x seq_len` shape, so `dynamic=True`
  does not recompile per eval shape (PyTorch's `dynamic=True` covers tensor
  shapes; the int `depth` and the eval signature cost one variant each, and
  the torch-compile guide discusses the recompile budget);
* numerics: the compiled forward/backward equals the eager one at fp32
  round-off, which is the contract the resume exemption stands on.

`backend="eager"` is used throughout: the guard/prefix/counter machinery is
dynamo's and backend-independent (`tests/test_compile_moe.py` documents the
same choice), while a full inductor compile of the tiny model costs ~90 s of
this suite. The lane's manual CPU spot-check with the default inductor backend
read the same loss bit-for-bit; the guide names it.
"""

from __future__ import annotations

import math
import re

import pytest
import torch
from conftest import tiny_cfg, tiny_run

from dormouse.model import DormouseModel
from dormouse.train.data import eval_window_bytes

# real inductor compilation; see the `slow` marker in pyproject.
pytestmark = pytest.mark.slow
from dormouse.train.loop import (
    _ema_update,
    _lfb_modules,
    _param_key,
    _shadow_params,
    _shadow_state,
    ckpt_paths,
    evaluate,
    load_ckpt,
    load_model_state,
    save_ckpt,
    train,
    unpack_state_dict,
)
from dormouse.train.optim import MixOptimizer


def _compile(model: torch.nn.Module) -> torch.nn.Module:
    """A real `OptimizedModule` (the wrapper the trainer builds), fast backend."""
    torch._dynamo.reset()
    return torch.compile(model, dynamic=True, backend="eager")


def _batch(batch: int = 2, seq: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(batch * seq, dtype=torch.long).view(batch, seq) % 251
    return ids, (ids + 1) % 251


def _eager_compile_spy(monkeypatch) -> list[dict]:
    """Patch `torch.compile` to the trainer's call shape but the eager backend."""
    calls: list[dict] = []
    real = torch.compile

    def spy(mod, **kw):
        calls.append(dict(kw))
        return real(mod, **kw, backend="eager")

    monkeypatch.setattr(torch, "compile", spy)
    return calls


# --- the progress-key gate --------------------------------------------------


def test_torch_compile_is_a_progress_key(tmp_path):
    """A resume may flip the executor; the checkpoint is the same file either way.

    Without this exemption the wrapper's own docstring flow ("the wrapper goes
    on AFTER resume") is impossible: every run that turned it on was refused by
    `diff_keys` as a different experiment, so an existing run could only buy the
    measured compile win by restarting its step count.
    """
    eager = tiny_run(tmp_path, torch_compile=False)
    compiled = tiny_run(tmp_path, torch_compile=True)
    assert compiled.diff_keys(eager) == []
    assert eager.diff_keys(compiled) == []


# --- numerics ---------------------------------------------------------------


def test_compiled_forward_and_backward_match_eager_at_fp32_round_off():
    """The compiled graph must BE the eager program, gradients included.

    This is the evidence the progress-key exemption stands on: a resume that
    flips `torch_compile` continues the same trajectory, not a different model.
    """
    eager = DormouseModel(tiny_cfg())
    compiled = DormouseModel(tiny_cfg())
    compiled.load_state_dict(eager.state_dict())
    cm = _compile(compiled)
    ids, tgt = _batch()

    with torch.no_grad():
        logits_e, loss_e, _ = eager(ids, tgt)
        logits_c, loss_c, _ = cm(ids, tgt)
    torch.testing.assert_close(logits_c, logits_e, rtol=1e-6, atol=1e-8)
    assert math.isclose(float(loss_c), float(loss_e), rel_tol=1e-6, abs_tol=1e-8)

    for p in eager.parameters():
        p.grad = None
    for p in cm.parameters():
        p.grad = None
    eager(ids, tgt)[1].backward()
    cm(ids, tgt)[1].backward()
    shared = 0
    for (name_e, p_e), (name_c, p_c) in zip(
        eager.named_parameters(), cm.named_parameters(), strict=True
    ):
        assert _param_key(name_c) == name_e, "the compiled wrapper renamed a parameter"
        if p_e.grad is None and p_c.grad is None:
            continue
        shared += 1
        torch.testing.assert_close(p_c.grad, p_e.grad, rtol=1e-6, atol=1e-8, msg=name_e)
    assert shared > 0, "no gradient reached a parameter — the gate would be vacuous"


# --- the checkpoint seam ----------------------------------------------------


def test_a_compiled_checkpoint_is_unwrapped_in_all_three_sections(tmp_path):
    """`model`, `model_ema` and `moe` name the UNWRAPPED tree; an eager run loads them.

    The EMA half is the one the old gate missed: it is built from
    `model.named_parameters()` (prefixed on a wrapper) and written through
    `_checkpoint_half` beside the raw iterate, so a prefix leak here makes the
    served/`resume`d EMA a differently-keyed copy of the same weights.
    """
    run = tiny_run(tmp_path, steps=1, ckpt_every=1, ema_decay=0.9, torch_compile=True)
    model = DormouseModel(run.model)
    opt = MixOptimizer(model, lr=0.01, wd=0.0, opt="mix")
    cm = _compile(model)
    shadow = _shadow_state(cm)
    weights, _snapshot = save_ckpt(run, cm, opt, step=0, ema=shadow)

    ck = torch.load(weights, map_location="cpu", weights_only=False)
    for section in ("model", "model_ema"):
        assert section in ck
        assert not any(k.startswith("_orig_mod.") for k in ck[section]), section
    assert set(ck["model"]) == set(model.state_dict())
    assert set(ck["model_ema"]) == {name for name, _ in model.named_parameters()}
    assert set(ck["moe"]) == set(_lfb_modules(model))

    # The reverse boundary: this file was written under torch_compile=true and
    # is resumed by a run that says false. The snapshot gate now permits it and
    # the canonical keys are what `load_model_state` accepts.
    eager_run = tiny_run(tmp_path, steps=1, ckpt_every=1, ema_decay=0.9, torch_compile=False)
    back, _stored = load_ckpt(eager_run)
    fresh = DormouseModel(eager_run.model)
    load_model_state(
        fresh, eager_run.train, unpack_state_dict(back["model"], back["model_quant_rung"])
    )
    for name, p in model.named_parameters():
        assert torch.equal(dict(fresh.named_parameters())[name], p.detach()), name


@pytest.mark.parametrize("first_compiled,second_compiled", [(True, False), (False, True)])
def test_resume_across_the_compile_boundary(tmp_path, monkeypatch, first_compiled, second_compiled):
    """Both directions continue the step count and leave canonical keys behind."""
    calls = _eager_compile_spy(monkeypatch)
    first = tiny_run(
        tmp_path, steps=1, log_every=10, ckpt_every=1, torch_compile=first_compiled, ema_decay=0.9
    )
    train(DormouseModel(first.model), first, log=lambda *_: None)
    assert len(calls) == int(first_compiled)

    second = tiny_run(
        tmp_path, steps=2, log_every=10, ckpt_every=2, torch_compile=second_compiled, ema_decay=0.9
    )
    lines: list[str] = []
    report = train(DormouseModel(second.model), second, resume=True, log=lines.append)
    assert len(calls) == int(first_compiled) + int(second_compiled)
    # Static shapes: every trainer step and every eval scores the same
    # batch x seq_len, so `dynamic=True` only paid its guards (0.952x vs 1.149x
    # static in docs/reviews/2026-11-08-compile-matrix.md).
    assert all(call.get("dynamic") is False for call in calls)
    assert all(call.get("mode") == "max-autotune-no-cudagraphs" for call in calls)
    assert report["opt_steps"] == 2, "the optimizer state must continue across the flip"
    assert math.isfinite(report["loss"])
    assert (any("torch_compile" in line for line in lines)) == second_compiled

    ck = torch.load(ckpt_paths(second.train)[0], map_location="cpu", weights_only=False)
    assert not any(k.startswith("_orig_mod.") for k in ck["model"])
    assert not any(k.startswith("_orig_mod.") for k in ck["model_ema"])


# --- the EMA shadow ---------------------------------------------------------


def test_the_ema_shadow_is_checkpoint_keyed_and_reaches_the_compiled_forward():
    """The shadow's keys are `state_dict()`'s, and `.data` swaps reach a compiled call.

    `train()` returns the shadow and `save_ckpt` writes it as `model_ema`; one
    spelling for both is the contract. The swap is also the eval path: a
    compiled graph reads each `Parameter` by object at call time, so a
    `.data` assignment must move the compiled loss exactly as it moves eager.
    """
    eager = DormouseModel(tiny_cfg())
    compiled = DormouseModel(tiny_cfg())
    compiled.load_state_dict(eager.state_dict())
    cm = _compile(compiled)

    shadow = _shadow_state(cm)
    assert set(shadow) == {name for name, _ in eager.named_parameters()}
    assert not any(k.startswith("_orig_mod.") for k in shadow)

    # `_ema_update` reads the LIVE parameters and writes the shadow: one key.
    before = {k: v.clone() for k, v in shadow.items()}
    with torch.no_grad():
        for p in cm.parameters():
            p.add_(0.5)
    _ema_update(cm, shadow, 0.5, torch.ones((), dtype=torch.int32))
    for name, p in cm.named_parameters():
        key = _param_key(name)
        torch.testing.assert_close(
            shadow[key], torch.lerp(before[key], p.detach(), 0.5), rtol=1e-6, atol=1e-7
        )

    # `_shadow_params` swaps by the same key; compare against eager with the
    # same weights rather than against the live ones (a no-op would pass both).
    shifted = {k: torch.full_like(v, 0.01) for k, v in _shadow_state(cm).items()}
    ids, tgt = _batch()
    with torch.no_grad():
        with _shadow_params(cm, shifted):
            _, loss_compiled, _ = cm(ids, tgt)
    for p in eager.parameters():
        p.data.fill_(0.01)
    with torch.no_grad():
        _, loss_eager, _ = eager(ids, tgt)
    assert math.isclose(float(loss_compiled), float(loss_eager), rel_tol=1e-6, abs_tol=1e-8)


# --- the optimizer and the compile boundary ---------------------------------


def test_the_optimizer_steps_the_same_parameters_the_compiled_model_reads():
    """Built BEFORE the wrap — and still one object tree, so no prefix can split it.

    `MixOptimizer` keys its routing on the unwrapped names and Muon's lazy
    groups on the unwrapped module; the compiled forward must read the same
    `Parameter` objects or the step would update weights nothing reads.
    """
    model = DormouseModel(tiny_cfg())
    opt = MixOptimizer(model, lr=0.01, wd=0.0, opt="mix")
    cm = _compile(model)
    assert opt.muon is not None
    wrapped = {id(p) for p in cm.parameters()}
    groups = list(opt.adamw.param_groups) + list(opt.muon.param_groups)
    assert groups
    for group in groups:
        for p in group["params"]:
            assert id(p) in wrapped, "the optimizer holds a copy the compiled forward cannot see"
    assert set(opt.routing) == {name for name, _ in model.named_parameters()}

    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.enable_grad():
        _, loss, _ = cm(*_batch())
        loss.backward()
    opt.step(torch.ones((), dtype=torch.int32))
    assert opt.steps == 1
    moved = [
        name for name, p in model.named_parameters() if not torch.equal(p.detach(), before[name])
    ]
    assert moved, "the compiled step updated nothing — the object identity is not the step"


# --- counters and eval ------------------------------------------------------


def test_a_compiled_run_keeps_its_counters_and_eval_line(tmp_path, monkeypatch):
    """A captured Python counter goes SILENT; these are the ones a reader checks.

    The forward writes `iter`/`forward`/`engram_*` in Python between recorded
    regions, and the eval creates its OWN `Counters` per line. Both must still
    move under the wrapper (a stale one is the ADR-0019 defect the graph arms
    had to fix with an explicit re-merge).
    """
    _eager_compile_spy(monkeypatch)
    run = tiny_run(
        tmp_path,
        steps=2,
        log_every=1,
        ckpt_every=100,
        torch_compile=True,
        eval=True,
        eval_every=1,
        eval_batches=2,
    )
    lines: list[str] = []
    report = train(DormouseModel(run.model), run, log=lines.append)
    assert report["counters"]["iter"] == 2 * run.model.max_iter
    assert report["counters"]["forward"] == 2
    assert report["counters"]["engram_rows"] > 0

    evals = [line for line in lines if line.startswith("eval step=")]
    assert len(evals) == 2, evals
    for line in evals:
        m = re.search(r"engram=(\d+)/(\d+)", line)
        assert m is not None and int(m.group(1)) > 0 and int(m.group(2)) > 0, line
        assert "over 256 B (2 x 2 x 64)" in line, line
    assert math.isfinite(report["loss"])


def test_eval_windows_are_the_training_shape(tmp_path):
    """`dynamic=True` cannot recompile on an eval shape that never changes.

    `eval_window_bytes` is `eval_batches x batch x seq_len` and
    `eval_batches_for` refuses a window that is not a whole multiple of
    `batch x seq_len`, so every eval batch has the training shape. This is what
    keeps the eval from paying a shape recompile at every eval.
    """
    run = tiny_run(tmp_path, eval=True, eval_batches=3)
    cm = _compile(DormouseModel(run.model))
    seen: list[tuple[int, int]] = []
    batch, seq = run.train.batch, run.train.seq_len

    class _Stream:
        lo = 0

        def rewind(self) -> None:
            return None

        def next_batch(self):
            ids = torch.arange(batch * seq, dtype=torch.long).view(batch, seq) % 251
            seen.append(tuple(ids.shape))
            return ids, (ids + 1) % 251

    ev, scored, _fin, _per_iter = evaluate(cm, _Stream(), batches=3)
    assert seen == [(batch, seq)] * 3
    assert scored == 3 * batch * seq
    assert eval_window_bytes(run.train) == scored
    assert math.isfinite(ev)
