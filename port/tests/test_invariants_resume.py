"""Resume determinism (2026-11-07): a checkpoint is the WHOLE state.

The audit's requirement, in CPU tests at a fixed seed: 30 steps on a
`block_form` model, interrupted at the step-15 checkpoint and resumed, must end
bit-identical to the uninterrupted 30-step run — final loss and every
parameter/buffer. Then the two schedule seams:

* mid-decay, with `steps` unchanged: still bit-identical (the WSD schedule is a
  pure function of `(step, total_steps)`, so a resume lands on it);
* `--set steps=` changed mid-run (a PROGRESS key): the past LR prefix can no
  longer be the new plan's, so equality with a fresh run of the new length is
  impossible BY CONSTRUCTION. What is checkable — and what this file checks —
  is that the resumed run equals the STITCHED plan (old schedule below the
  checkpoint, new schedule from it), that the new plan is applied rather than
  frozen, and that a `steps` that would run zero steps REFUSES LOUDLY instead
  of reporting a run it did not run.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from pathlib import Path

import pytest
import torch
from conftest import tiny_cfg

from dormouse.config import RunConfig, TrainConfig
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train import loop as loopmod
from dormouse.train import train
from dormouse.train.lr import wsd_lr

TFORM = dict(
    block_form="transformer",
    n_layers=2,
    max_iter=1,
    loop_blocks=1,
    use_kda=False,
    use_csa=False,
    use_mhc=False,
    use_attnres=False,
    use_engram=False,
    layerrope=False,
    n_experts=1,
    moe_topk=0,
    use_tsct=False,
)
KDAFORM = dict(block_form="kda", max_iter=2, moe_topk=2, moe_lb_coef=0.5)
FORMS = [pytest.param(TFORM, "transformer"), pytest.param(KDAFORM, "kda")]


class _Stop(Exception):
    """The interruption: raise after the step-15 checkpoint, on the step-16 log."""


def _base(form: dict) -> DormouseModel:
    torch.manual_seed(1234)
    return DormouseModel(tiny_cfg(**form))


def _run(tmp_path: Path, form: dict, *, steps: int, **train_over) -> RunConfig:
    (tmp_path / "corpus.bin").write_bytes(bytes(range(16)) * 256)
    cfg = TrainConfig(
        steps=steps,
        batch=2,
        seq_len=64,
        lr=0.01,
        log_every=1,
        ckpt_every=15,
        ckpt_name="t",
        out_dir=str(tmp_path / "ck"),
        data=str(tmp_path / "corpus.bin"),
    )
    if train_over:
        cfg = dataclasses.replace(cfg, **train_over)
    return RunConfig(source="det", model=tiny_cfg(**form), train=cfg)


def _go(model, run, *, stop_at: int | None = None, resume: bool = False, log=lambda *_: None):
    """Run (optionally `resume`), optionally stopping at `stop_at`'s log line."""

    def _log(line: str) -> None:
        log(line)
        if stop_at is not None and line.startswith("step="):
            step = int(line.split("step=")[1].split()[0])
            if step == stop_at:
                raise _Stop

    try:
        return train(model, run, resume=resume, log=_log)
    except _Stop:
        return None


def _hash(model: DormouseModel) -> str:
    """Every parameter AND buffer, name-tagged: the state a resume must carry."""
    h = hashlib.sha256()
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _opt_tensors(opt) -> dict[str, torch.Tensor]:
    """Flatten an optimizer state_dict to `path -> tensor` (ids kept as paths)."""

    def walk(prefix: str, obj, out: dict) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(f"{prefix}.{key}", value, out)
        elif isinstance(obj, (list, tuple)):
            for i, value in enumerate(obj):
                walk(f"{prefix}[{i}]", value, out)
        elif torch.is_tensor(obj):
            out[prefix] = obj.detach().cpu().clone()

    out: dict[str, torch.Tensor] = {}
    walk("", opt.state_dict(), out)
    return out


def _assert_same_opt(a, b) -> None:
    ta, tb = _opt_tensors(a), _opt_tensors(b)
    assert set(ta) == set(tb), f"optimizer state keys moved: {set(ta) ^ set(tb)}"
    for key in ta:
        assert torch.equal(ta[key], tb[key]), f"optimizer state {key} differs"


@pytest.mark.parametrize("form,name", FORMS)
def test_resume_at_15_is_bit_identical_to_30(tmp_path, form, name):
    """Uninterrupted 30 vs 15 + resume: same loss, same weights, same moments."""
    ref_dir = tmp_path / "ref"
    int_dir = tmp_path / "cut"
    ref_dir.mkdir()
    int_dir.mkdir()

    base = _base(form)
    ref = _base(form)
    report = _go(ref, _run(ref_dir, form, steps=30))

    cut = copy.deepcopy(base)
    _go(cut, _run(int_dir, form, steps=30), stop_at=16)  # 15's checkpoint is on disk
    resumed = _go(cut, _run(int_dir, form, steps=30), resume=True)

    assert resumed is not None
    assert resumed["loss"] == report["loss"], f"{name}: final loss moved"
    assert _hash(cut) == _hash(ref), f"{name}: the parameter/buffer hash moved"
    assert resumed["opt_steps"] == report["opt_steps"] == 30
    # ...and not just coincidentally equal: the files both runs wrote at step
    # 30 agree at the tensor level, model half, `moe` section and optimizer
    # moments. That is ONE ckpt path writing the same bytes from two different
    # processes' worth of history.
    a = torch.load(ref_dir / "ck" / "t.pt", map_location="cpu", weights_only=False)
    b = torch.load(int_dir / "ck" / "t.pt", map_location="cpu", weights_only=False)
    assert a["step"] == b["step"] == 29
    assert set(a["model"]) == set(b["model"])
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key
    assert a["moe"].keys() == b["moe"].keys()
    for key in a["moe"]:
        assert torch.equal(a["moe"][key], b["moe"][key]), key
    _assert_same_opt_state(a["opt"], b["opt"])


def _assert_same_opt_state(a: dict, b: dict) -> None:
    def walk(x, y, path=""):
        if isinstance(x, dict):
            assert set(x) == set(y), path
            for key in x:
                walk(x[key], y[key], f"{path}.{key}")
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y), path
            for i, (u, v) in enumerate(zip(x, y, strict=True)):
                walk(u, v, f"{path}[{i}]")
        elif torch.is_tensor(x):
            assert torch.equal(x, y), path
        else:
            assert x == y, path

    walk(a, b)


@pytest.mark.parametrize("form,name", FORMS)
def test_resume_mid_decay_is_bit_identical(tmp_path, form, name):
    """The checkpoint at step 15 sits INSIDE the decay, not before it."""
    over = dict(warmup_tokens=4 * 128, decay_frac=0.6)  # 18 decay steps: start = 12
    assert wsd_lr(15, total_steps=30, peak_lr=0.01, warmup_steps=4, decay_steps=18) < 0.01
    assert wsd_lr(14, total_steps=30, peak_lr=0.01, warmup_steps=4, decay_steps=18) < 0.01

    ref_dir, int_dir = tmp_path / "ref", tmp_path / "cut"
    ref_dir.mkdir()
    int_dir.mkdir()
    ref = _base(form)
    report = _go(ref, _run(ref_dir, form, steps=30, **over))
    cut = _base(form)
    _go(cut, _run(int_dir, form, steps=30, **over), stop_at=16)
    resumed = _go(cut, _run(int_dir, form, steps=30, **over), resume=True)
    assert resumed is not None
    assert resumed["loss"] == report["loss"], f"{name}: mid-decay loss moved"
    assert _hash(cut) == _hash(ref), f"{name}: mid-decay hash moved"


def test_extending_steps_mid_run_matches_the_new_plan(tmp_path):
    """`--set steps=45` from a step-15 checkpoint: bit-identical to a clean 45.

    The progress key is only coherent when the two plans agree on the steps
    ALREADY trained. `warmup=20 > 15` makes the first 15 LR values identical
    under both totals (the warmup ramp does not read `total_steps`), so the
    extended resume is comparable with an uninterrupted 45-step run and must
    land on it exactly. The LR at step 15 onwards is the 45-step plan's, not a
    frozen 30-step leftover (asserted below).
    """
    over = dict(warmup_tokens=20 * 128, decay_frac=0.5)
    for step in range(15):
        assert wsd_lr(
            step, total_steps=30, peak_lr=0.01, warmup_steps=20, decay_steps=15
        ) == wsd_lr(step, total_steps=45, peak_lr=0.01, warmup_steps=20, decay_steps=22), step
    # Both totals share the 20-step warmup ramp; the plans separate AFTER it.
    assert wsd_lr(20, total_steps=30, peak_lr=0.01, warmup_steps=20, decay_steps=15) != (
        wsd_lr(20, total_steps=45, peak_lr=0.01, warmup_steps=20, decay_steps=22)
    ), "the fixture must actually re-plan the LR past the warmup"

    ref_dir, int_dir = tmp_path / "ref", tmp_path / "cut"
    ref_dir.mkdir()
    int_dir.mkdir()
    ref = _base(TFORM)
    report = _go(ref, _run(ref_dir, TFORM, steps=45, **over))
    cut = _base(TFORM)
    _go(cut, _run(int_dir, TFORM, steps=30, **over), stop_at=16)
    resumed = _go(cut, _run(int_dir, TFORM, steps=45, **over), resume=True)
    assert resumed is not None
    assert resumed["steps"] == 45 and resumed["opt_steps"] == 45
    assert resumed["loss"] == report["loss"], "the extended run did not match the 45 plan"
    assert _hash(cut) == _hash(ref)


def test_a_steps_change_matches_the_stitched_schedule(tmp_path, monkeypatch):
    """When the past prefix differs, the contract is the STITCHED plan.

    `warmup=4, decay_frac=0.6`: for `total=30` the decay starts at step 12 and
    for `total=45` at step 18, so steps 12..14 were trained under a curve the
    45-run never used. A fresh 45-step run therefore CANNOT match by
    construction; what must hold is that the resumed run is exactly "old
    schedule below step 15, new schedule from step 15", with no other state
    lost. The stitched reference below is that sentence as code.
    """
    over = dict(warmup_tokens=4 * 128, decay_frac=0.6)

    def stitched(step, *, total_steps, peak_lr, warmup_steps=0, decay_steps=0, final_frac=0.1):
        total = 30 if step < 15 else 45
        decay = int(round(0.6 * total))
        return wsd_lr(
            step,
            total_steps=total,
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay,
            final_frac=final_frac,
        )

    ref_dir, cut_dir, fresh_dir = tmp_path / "ref", tmp_path / "cut", tmp_path / "fresh"
    for d in (ref_dir, cut_dir, fresh_dir):
        d.mkdir()

    monkeypatch.setattr(loopmod, "wsd_lr", stitched)
    ref = _base(TFORM)
    report = _go(ref, _run(ref_dir, TFORM, steps=45, **over))
    monkeypatch.undo()

    cut = _base(TFORM)
    _go(cut, _run(cut_dir, TFORM, steps=30, **over), stop_at=16)
    resumed = _go(cut, _run(cut_dir, TFORM, steps=45, **over), resume=True)
    assert resumed is not None
    assert resumed["loss"] == report["loss"], "the stitched-plan resume diverged"
    assert _hash(cut) == _hash(ref)

    # Named difference: a FRESH 45-step run is a different trajectory, and the
    # gap is the re-planned prefix, not a lost checkpoint tensor.
    fresh = _base(TFORM)
    other = _go(fresh, _run(fresh_dir, TFORM, steps=45, **over))
    assert other["loss"] != report["loss"]
    gap = abs(other["loss"] - report["loss"])
    print(f"steps-change: fresh-45 loss {other['loss']:.9f} vs stitched {report['loss']:.9f} "
          f"(gap {gap:.3e} nats, cause: the 12..14 LR prefix)")
    assert gap > 1e-9


def test_resuming_past_the_plan_refuses_loudly(tmp_path):
    """A `steps` that ends at/before the checkpoint is a no-op, not a run."""
    d = tmp_path / "cut"
    d.mkdir()
    cut = _base(TFORM)
    _go(cut, _run(d, TFORM, steps=30), stop_at=16)  # checkpoint at step 15
    weights = d / "ck" / "t.pt"
    assert weights.is_file()
    before = _hash(cut)
    with pytest.raises(LoudError) as e:
        _go(cut, _run(d, TFORM, steps=10), resume=True)
    msg = str(e.value)
    assert "is behind it" in msg and "checkpoint is at step 15" in msg
    assert "--set steps=N" in msg
    # The refusal fires BEFORE the model is touched: a refused command leaves
    # the caller's weights (the interrupted process's 16-update state) alone.
    assert _hash(cut) == before, "the refused resume must not move a weight"

    # Resuming a FINISHED file at the same `steps` is a legal zero-step LOAD:
    # `steps == opt_steps`, so the report is true (the EMA-read test in
    # `test_train4` depends on it). One more step is the extension.
    done = tmp_path / "done"
    done.mkdir()
    finished = _base(TFORM)
    _go(finished, _run(done, TFORM, steps=30, ckpt_every=30))
    loaded = _go(_base(TFORM), _run(done, TFORM, steps=30, ckpt_every=30), resume=True)
    assert loaded is not None and loaded["opt_steps"] == loaded["steps"] == 30
    longer = _go(_base(TFORM), _run(done, TFORM, steps=31, ckpt_every=30), resume=True)
    assert longer is not None and longer["opt_steps"] == 31
