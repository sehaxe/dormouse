"""Gates for the BPB accounting: fin is the headline, the window and offset are pinned.

Background (`docs/reviews/2026-11-06-bpb-accounting.md`): the logged `bpb` was
the depth-AVERAGED training objective while generation reads the FINAL readout
(1.729 headline vs 1.617 fin on one checkpoint), the eval scored a 65 536-byte
window that was never named as a choice, and `bytes_seen` ignored `grad_accum`
(0.53 GB logged vs 1.573 GB real at step 64 000 x3). Each test below fails if
its fix is reverted: (a) headline mapping, (b) window floor + exact width,
(c) pinned offset / bit-exact repeat, (d) accumulation arithmetic,
(e) the noise probe computes a spread from its own trials.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from conftest import BATCH, SEQ, tiny_run

from dormouse.config import validate_train
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train import loop as loopmod
from dormouse.train.data import (
    ByteStream,
    eval_batches_for,
    eval_window_bytes,
    open_streams,
)
from dormouse.train.loop import _eval_line, _eval_loss, eval_row, train

ROOT = Path(__file__).resolve().parents[1]
LN2 = 0.6931471805599453


# ---------------------------------------------------------------- (a) headline


class _ThreeDepth:
    """A 3-iteration readout whose per-iteration CEs DIFFER (2.0, 3.0, 4.0).

    `fin` is 1.0, so the depth mean is 2.5 — a fixture where reading the wrong
    field is visible by 1.5 nats (2.16 BPB), not by rounding.
    """

    last_breakdown = None
    cfg = SimpleNamespace(max_iter=3)

    def __init__(self) -> None:
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def __call__(self, x, y, **kw):
        self.last_breakdown = {"fin": 1.0, "per_iter": [2.0, 3.0, 4.0]}
        return None, torch.tensor((2.0 + 3.0 + 4.0 + 1.0) / 4.0), None


def test_the_headline_is_the_final_iteration_not_the_depth_mean(tmp_path):
    run = tiny_run(tmp_path, eval=True, eval_batches=2)
    stream = ByteStream(run.train.data, BATCH, SEQ, lo=0)
    line, ev, scored, fin = _eval_line(_ThreeDepth(), stream, run.train, 3, None, 7)

    assert fin == pytest.approx(1.0) and ev == pytest.approx(2.5)
    assert f"bpb_fin={1.0 / LN2:.3f}" in line
    assert f"bpb_mean={2.5 / LN2:.3f}" in line
    # No bare `bpb=`: a reader cannot pick the old field by accident.
    assert " bpb=" not in line

    row = eval_row(
        step=7, ev=ev, fin=fin, scored=scored, cfg=run.train, offset=stream.lo
    )
    assert row["bpb"] == pytest.approx(1.0 / LN2)
    assert row["bpb_fin"] == row["bpb"]
    assert row["bpb_mean"] == pytest.approx(2.5 / LN2)
    assert row["bpb"] != row["bpb_mean"], "the two fields must not be the same number"

    # And on a REAL 3-iteration model, where the per-iteration CEs come from
    # logits: `fin` is the model's own breakdown value, `ev` is its mean.
    real = tiny_run(tmp_path, eval_batches=1)
    model = DormouseModel(dataclasses.replace(real.model, max_iter=3))
    stream = ByteStream(real.train.data, BATCH, SEQ, lo=0)
    ev, _scored, fin, per = _eval_loss(model, stream, 1, None, None)
    bd = model.last_breakdown
    assert fin == pytest.approx(bd["fin"])
    assert per == pytest.approx(bd["per_iter"])
    assert len(per) == 3 and len(set(per)) > 1, "the fixture must have distinct iterations"
    assert ev == pytest.approx((sum(bd["per_iter"]) + bd["fin"]) / 4.0)
    assert fin < ev, "the final readout must not be reported as the depth mean"


# ---------------------------------------------------------- (b) window floor


def _big_corpus(tmp_path: Path, n: int = 1 << 18) -> Path:
    p = tmp_path / "big.bin"
    p.write_bytes(bytes(range(256)) * (n // 256))
    return p


def test_eval_refuses_a_window_below_the_floor(tmp_path):
    run = tiny_run(
        tmp_path,
        data=str(_big_corpus(tmp_path)),
        eval=True,
        eval_batches=None,
        eval_bytes=65_535,
    )
    with pytest.raises(LoudError, match="65535.*floor"):
        validate_train(run.train)
    with pytest.raises(LoudError, match="65535.*floor"):
        eval_window_bytes(run.train)
    with pytest.raises(LoudError, match="65535.*floor"):
        open_streams(run)


def test_the_eval_runs_exactly_the_configured_window(tmp_path, monkeypatch):
    run = tiny_run(
        tmp_path,
        data=str(_big_corpus(tmp_path)),
        eval=True,
        eval_batches=None,
        eval_bytes=65_536,
    )
    _train, stream = open_streams(run)
    assert stream is not None
    assert eval_window_bytes(run.train) == 65_536
    assert eval_batches_for(run.train) == 512  # 65536 / (2 x 64)

    seen: dict = {}

    def spy(model, stream, batches, *a, **k):
        seen.update(batches=batches, lo=stream.lo, span=stream.hi - stream.lo)
        return 1.0, 65_536, 1.0, []

    monkeypatch.setattr(loopmod, "evaluate", spy)
    line, _ev, scored, _fin = _eval_line(None, stream, run.train, 2, None, 5)
    assert seen["batches"] == 512
    assert seen["batches"] * BATCH * SEQ == 65_536 == scored
    assert seen["lo"] == (1 << 18) - 65_536  # the eval tail carve starts here
    assert f"over 65536 B (512 x {BATCH} x {SEQ})" in line


# -------------------------------------------------------- (c) pinned offset


def test_same_offset_is_bit_exact_and_a_different_offset_differs(tmp_path):
    data = tmp_path / "eval.bin"
    # Two very different regions: zeros, then repeated English-ish bytes.
    data.write_bytes(b"\x00" * (1 << 16) + b"the quick brown fox " * 9000)

    def cfg(offset: int):
        return tiny_run(
            tmp_path,
            batch=8,
            seq_len=64,
            data=str(data),
            eval_data=str(data),
            eval=True,
            eval_batches=None,
            eval_bytes=65_536,
            eval_offset=offset,
        )

    run_a = cfg(0)
    model = DormouseModel(run_a.model)
    _t, stream_a = open_streams(run_a)
    assert stream_a is not None and stream_a.lo == 0
    line1, _e1, _s1, fin1 = _eval_line(model, stream_a, run_a.train, 2, None, 10)
    line2, _e2, _s2, fin2 = _eval_line(model, stream_a, run_a.train, 2, None, 20)
    assert fin1 == fin2, "a rewound stream at the same offset must be bit-exact"
    assert "off=0" in line1 and "off=0" in line2

    run_b = cfg(65_536)
    _t2, stream_b = open_streams(run_b)
    assert stream_b is not None and stream_b.lo == 65_536
    line3, _e3, _s3, fin3 = _eval_line(model, stream_b, run_b.train, 2, None, 30)
    assert fin3 != fin1, "a different offset is a different sample, not the same number"
    assert "off=65536" in line3 and "off=65536" not in line1


# ---------------------------------------------------- (d) bytes / grad_accum


def test_bytes_seen_counts_grad_accum_micro_batches(tmp_path):
    def one(accum: int) -> tuple[dict, dict]:
        log = tmp_path / f"run{accum}.jsonl"
        run = tiny_run(
            tmp_path,
            steps=1,
            log_every=1,
            grad_accum=accum,
            eval=True,
            eval_every=1,
            eval_batches=2,
            json_log=str(log),
        )
        train(DormouseModel(run.model), run, log=lambda *_: None)
        rows = [json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()]
        return (
            next(r for r in rows if r["event"] == "log"),
            next(r for r in rows if r["event"] == "eval"),
        )

    micro_log, micro_eval = one(1)
    times3_log, times3_eval = one(3)
    micro = BATCH * SEQ
    assert micro_log["bytes_seen"] == micro
    assert times3_log["bytes_seen"] == 3 * micro == 3 * micro_log["bytes_seen"]
    assert micro_eval["bytes_seen"] == micro
    assert times3_eval["bytes_seen"] == 3 * micro


# -------------------------------------------------------- (e) noise probe


def _noise_module():
    spec = importlib.util.spec_from_file_location(
        "noise_floor_under_test", ROOT / "scripts" / "noise_floor.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_the_noise_report_is_computed_from_the_trials():
    mod = _noise_module()
    moved = mod.report([1.5000, 1.5030], window=65_536, offset=0, steps=1)
    assert "spread=0.0030 BPB" in moved
    same = mod.report([1.5000, 1.5000], window=65_536, offset=0, steps=1)
    assert "spread=0.0000 BPB" in same
    assert moved != same, "the spread must be a function of the trials, not a constant"


def test_the_noise_probe_measures_a_cpu_tiny_run():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run(
        [
            sys.executable,
            "scripts/noise_floor.py",
            "--steps",
            "1",
            "--batch",
            "8",
            "--seq-len",
            "64",
            "--eval-bytes",
            "65536",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    vals = [float(v) for v in re.findall(r"trial \d+: bpb_fin=([0-9.]+)", r.stdout)]
    hit = re.search(r"spread=([0-9.]+) BPB", r.stdout)
    assert len(vals) == 2 and hit, r.stdout
    # The reported number is the spread of the two trials THIS run printed.
    assert float(hit.group(1)) == pytest.approx(abs(vals[0] - vals[1]), abs=1e-4)
