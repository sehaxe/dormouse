"""The checkpoint/sidecar step gate and the off-cadence final checkpoint.

`save_ckpt` writes weights, then the `.ngram` sibling, then the snapshot. The
snapshot gate sees only config, so a crash between the first two writes used to
resume new weights with old host rows/momentum/counts/step. The sidecar now
carries the weights' `step`, and `load_ckpt` refuses a mismatch. These gates are
CPU-only and tiny; the pilot's `engram_ram=true` path is what they exercise.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from conftest import tiny_run

from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train import load_ckpt, train
from dormouse.train import loop as train_loop
from dormouse.train.loop import (
    LOAD_WARNINGS,
    ckpt_paths,
    load_model_state,
    load_ngram,
    load_weights_only,
    ngram_path,
)


def _patch_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_loop, "host_device_type", lambda _m: "cuda")


def test_stale_ngram_sidecar_is_refused(tmp_path, monkeypatch):
    """A sidecar stamped with a DIFFERENT step is a refused resume, LOUD."""
    _patch_device(monkeypatch)
    run = tiny_run(tmp_path, steps=2, ckpt_every=2, log_every=10, engram_ram=True)
    train(DormouseModel(run.model), run, log=lambda *_: None)
    weights, _ = ckpt_paths(run.train)
    assert weights.is_file() and ngram_path(run.train).is_file()
    assert int(load_weights_only(weights)["step"]) == 1

    # Simulate the crash between the weights write and the sidecar write: the
    # weights stay at step 1, the sibling is restamped as if from step 999.
    stale = load_ngram(run.train)
    stale["step"] = 999
    torch.save(stale, ngram_path(run.train))
    with pytest.raises(LoudError) as e:
        load_ckpt(run)
    msg = str(e.value)
    assert "resume refused" in msg
    assert "stamped step 999" in msg
    assert "Escape:" in msg


def test_matching_step_resume_is_bit_identical(tmp_path, monkeypatch):
    """A resume whose sidecar step matches the weights loads bit-for-bit."""
    _patch_device(monkeypatch)
    run = tiny_run(tmp_path, steps=2, ckpt_every=2, log_every=10, engram_ram=True)
    train(DormouseModel(run.model), run, log=lambda *_: None)

    ck, _ = load_ckpt(run)  # matching stamp: must NOT refuse
    saved = load_ngram(run.train)
    assert int(saved["step"]) == int(ck["step"]) == 1

    fresh = DormouseModel(run.model)
    load_model_state(fresh, run.train, ck["model"])
    host = fresh.loop.engram.host
    per_engram = saved["engrams"][0]
    assert host.step == int(per_engram["step"]), "the host UPDATE counter is per-engram"
    assert torch.equal(host.rows, per_engram["rows"])
    assert torch.equal(host.momentum, per_engram["momentum"])
    assert torch.equal(host.counts, per_engram["counts"])


def test_legacy_sidecar_without_step_is_counted_and_warned(tmp_path, monkeypatch, capsys):
    """A pre-gate sidecar (no `step`) still reads, but is COUNTED and warned."""
    _patch_device(monkeypatch)
    run = tiny_run(tmp_path, steps=2, ckpt_every=2, log_every=10, engram_ram=True)
    train(DormouseModel(run.model), run, log=lambda *_: None)
    legacy = load_ngram(run.train)
    del legacy["step"]
    torch.save(legacy, ngram_path(run.train))

    before = int(LOAD_WARNINGS.get("ngram_no_step", 0))
    ck, _ = load_ckpt(run)  # the live pilot's old format must stay readable
    out = capsys.readouterr().out
    assert int(ck["step"]) == 1
    assert "ngram_no_step" in out and "COUNTED" in out
    assert int(LOAD_WARNINGS["ngram_no_step"]) == before + 1


def test_final_checkpoint_when_last_step_off_cadence(tmp_path, monkeypatch):
    """steps=7, ckpt_every=3: a checkpoint at 7 reproduces the final eval."""
    _patch_device(monkeypatch)
    common = dict(
        steps=7,
        ckpt_every=3,
        log_every=100,
        eval=True,
        eval_every=2,
        eval_batches=2,
        engram_ram=True,
    )
    (tmp_path / "a").mkdir()
    run_a = tiny_run(tmp_path / "a", **common)
    lines_a: list[str] = []
    train(DormouseModel(run_a.model), run_a, log=lines_a.append)

    weights, _ = ckpt_paths(run_a.train)
    assert weights.is_file() and ngram_path(run_a.train).is_file()
    ck = load_weights_only(weights)
    # Cadence (7 // 3 = 2 saves) last stamped step 5; the off-cadence final
    # checkpoint re-stamps both halves to the run's last step index, 6 (= step 7).
    assert int(ck["step"]) == 6
    assert int(load_ngram(run_a.train)["step"]) == 6

    # Reproduce the final eval by resuming the finished checkpoint with ZERO new
    # steps: the only thing that runs is the same off-cadence final eval, and it
    # must print the SAME line because it scores the checkpoint's weights.
    run_b = dataclasses.replace(run_a)
    lines_b: list[str] = []
    report_b = train(DormouseModel(run_b.model), run_b, resume=True, log=lines_b.append)
    assert report_b["opt_steps"] == 7, "no steps ran on the zero-step resume"
    eval_a = [ln for ln in lines_a if ln.startswith("eval step=7")]
    eval_b = [ln for ln in lines_b if ln.startswith("eval step=7")]
    assert eval_a and eval_b, (lines_a, lines_b)
    assert eval_a[-1] == eval_b[-1], (eval_a[-1], eval_b[-1])
