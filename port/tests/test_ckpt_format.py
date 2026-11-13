"""The checkpoint FILE: self-describing (v2), atomic, and safe to load from abroad.

This is the gate for the commercial-load contract, and each test names the
failure it prevents:

1. **v2 metadata** — `format_version`, `git_sha`, `corpus` are written, and the
   corpus hash covers the plan text plus source sizes. Without it a checkpoint
   cannot say which code or mixture produced its step count (AGENTS §1.4).
2. **atomicity** — a write that dies mid-file leaves the OLD checkpoint intact,
   and the write fsyncs the file AND the directory entry. `os.replace` alone
   orders the name, not the bytes.
3. **v1 compatibility** — the files already on disk have no version; they load
   with v1 semantics and lose only the provenance fields, never a weight, the
   optimizer or the step.
4. **`weights_only=True`** — a foreign `.pt` that names a class is refused; a
   plain tensor record (ours is one) loads.
"""

from __future__ import annotations

import dataclasses
import os
import stat
from pathlib import Path

import pytest
import torch
from conftest import tiny_run

from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train import load_ckpt
from dormouse.train.loop import (
    CHECKPOINT_FORMAT_VERSION,
    load_model_half,
    load_weights_only,
    save_ckpt,
)
from dormouse.train.optim import MixOptimizer


class NotAPlainRecord:
    """A class a foreign pickle may name; `weights_only=True` must refuse it."""

    pass


def _saved(tmp_path: Path, **train_over):
    """One checkpoint through the real writer; the run/model/opt stay live."""
    run = tiny_run(tmp_path, **train_over)
    model = DormouseModel(run.model)
    opt = MixOptimizer(model, lr=0.01, wd=0.0, opt="mix")
    weights, snapshot = save_ckpt(run, model, opt, step=3)
    return run, model, opt, weights, snapshot


def test_v2_file_is_self_describing(tmp_path):
    run, model, _opt, weights, snapshot = _saved(tmp_path)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    assert ck["format_version"] == CHECKPOINT_FORMAT_VERSION == 2
    assert ck["git_sha"] is None or isinstance(ck["git_sha"], str)
    data = Path(run.train.data)
    assert ck["corpus"]["mix_plan"] is None
    assert ck["corpus"]["sources"] == {data.name: data.stat().st_size}
    assert ck["corpus"]["sha256"] and len(ck["corpus"]["sha256"]) == 64
    assert ck["corpus"]["missing"] == []
    assert int(ck["step"]) == 3
    # The config snapshot travels beside it and is the pair's other half.
    assert snapshot == weights.with_suffix(".config.toml") and snapshot.is_file()
    assert set(ck["model"]) == set(model.state_dict())


def test_a_crash_mid_write_keeps_the_old_checkpoint(tmp_path, monkeypatch):
    run, model, opt, weights, snapshot = _saved(tmp_path)
    before = weights.read_bytes()
    snapshot_before = snapshot.read_text()

    def half_a_file(payload, f, *a, **k):
        f.write(b"half a checkpoint, then the power went")
        raise RuntimeError("disk yanked")

    # The failure is INSIDE `torch.save`, i.e. after the tmp file exists and
    # before the rename: the state a power loss leaves behind.
    monkeypatch.setattr("dormouse.train.loop.torch.save", half_a_file)
    with pytest.raises(RuntimeError, match="disk yanked"):
        save_ckpt(run, model, opt, step=4)

    assert weights.read_bytes() == before, "the old checkpoint must survive"
    assert snapshot.read_text() == snapshot_before
    assert not weights.with_name(weights.name + ".tmp").exists(), "no stale tmp"


def test_the_write_fsyncs_the_file_and_the_directory(tmp_path, monkeypatch):
    run = tiny_run(tmp_path)
    model = DormouseModel(run.model)
    opt = MixOptimizer(model, lr=0.01, wd=0.0, opt="mix")
    synced_dirs: list[bool] = []
    real_fsync = os.fsync

    def spy(fd):
        synced_dirs.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        return real_fsync(fd)

    monkeypatch.setattr("dormouse.train.loop.os.fsync", spy)
    save_ckpt(run, model, opt, step=0)
    # weights + its directory entry, snapshot + its directory entry
    assert len(synced_dirs) == 4, synced_dirs
    assert sum(synced_dirs) == 2, "the rename needs the DIRECTORY fsynced too"


def test_a_v1_file_loads_and_loses_only_provenance(tmp_path):
    run, _model, _opt, weights, _snapshot = _saved(tmp_path)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    for key in ("format_version", "git_sha", "corpus"):
        del ck[key]
    torch.save(ck, weights)  # the v1 shape every pre-2026-11-08 file has

    back, stored = load_ckpt(run)
    assert back["format_version"] == 1
    assert "git_sha" not in back and "corpus" not in back
    assert torch.equal(back["model"]["embed.weight"], ck["model"]["embed.weight"])
    assert int(back["step"]) == 3 and "opt" in back
    assert stored.train.steps == run.train.steps


def test_a_newer_format_version_is_refused(tmp_path):
    run, _model, _opt, weights, _snapshot = _saved(tmp_path)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    ck["format_version"] = CHECKPOINT_FORMAT_VERSION + 1
    torch.save(ck, weights)
    with pytest.raises(LoudError, match="format_version"):
        load_ckpt(run)
    with pytest.raises(LoudError, match="format_version"):
        load_model_half(weights)
    with pytest.raises(LoudError, match="format_version"):
        load_model_half(weights, weights_only=True)


def test_the_corpus_manifest_hashes_the_plan_and_source_sizes(tmp_path):
    for name in ("a.bin", "b.bin"):
        (tmp_path / name).write_bytes(b"x" * 64)
    plan = tmp_path / "mix.toml"
    plan.write_text(
        "max_epochs = 4.0\nseed = 1\n\n[[stage]]\nuntil_frac = 1.0\n"
        "[stage.weights]\na = 0.5\nb = 0.5\n\n[sources]\n"
        f'a = "{tmp_path / "a.bin"}"\nb = "{tmp_path / "b.bin"}"\n'
    )
    run = tiny_run(tmp_path)
    run = dataclasses.replace(run, train=dataclasses.replace(run.train, mix_plan=str(plan)))
    model = DormouseModel(run.model)
    opt = MixOptimizer(model, lr=0.01, wd=0.0, opt="mix")
    weights, _snapshot = save_ckpt(run, model, opt, step=1)

    corpus = torch.load(weights, map_location="cpu", weights_only=False)["corpus"]
    assert corpus["sources"] == {"a": 64, "b": 64}
    first = corpus["sha256"]
    assert first and len(first) == 64

    # An edited PLAN is a different mixture even when every size matches.
    plan.write_text(plan.read_text().replace("seed = 1", "seed = 2"))
    save_ckpt(run, model, opt, step=2)
    edited = torch.load(weights, map_location="cpu", weights_only=False)["corpus"]
    assert edited["sha256"] != first
    assert edited["sources"] == {"a": 64, "b": 64}

    # A missing source records null AND its name; the save still happens.
    (tmp_path / "b.bin").unlink()
    save_ckpt(run, model, opt, step=3)
    gone = torch.load(weights, map_location="cpu", weights_only=False)["corpus"]
    assert gone["sha256"] is None
    assert gone["missing"] == ["b"]
    assert gone["sources"] == {"a": 64, "b": None}


def test_weights_only_loads_plain_tensors_and_refuses_a_pickle(tmp_path):
    _run, model, _opt, weights, _snapshot = _saved(tmp_path)
    plain = load_weights_only(weights)
    assert set(plain["model"]) == set(model.state_dict())
    half = load_model_half(weights, weights_only=True)
    assert set(half) == set(model.state_dict())

    bad = tmp_path / "foreign.pt"
    torch.save({"model": {"w": torch.zeros(1)}, "extra": NotAPlainRecord()}, bad)
    with pytest.raises(LoudError, match="weights_only=True"):
        load_weights_only(bad)
    with pytest.raises(LoudError):
        load_model_half(bad, weights_only=True)


def test_the_weights_only_reader_still_prefers_the_served_ema_half(tmp_path):
    run = tiny_run(tmp_path)
    model = DormouseModel(run.model)
    opt = MixOptimizer(model, lr=0.01, wd=0.0, opt="mix")
    ema = {k: v.detach().clone() + 1.0 for k, v in model.state_dict().items()}
    weights, _snapshot = save_ckpt(run, model, opt, step=0, ema=ema)
    served = load_model_half(weights, weights_only=True)
    assert torch.equal(served["embed.weight"], ema["embed.weight"])
    assert not torch.equal(served["embed.weight"], model.state_dict()["embed.weight"])
    raw, _stored = load_ckpt(run)
    assert torch.equal(raw["model"]["embed.weight"], model.state_dict()["embed.weight"])
    assert torch.equal(raw["model_ema"]["embed.weight"], ema["embed.weight"])
