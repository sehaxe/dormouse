"""The shape table's arithmetic — the two recorded points, and the eval window.

`scripts/bench_shape.py` is a tool written for a free card, so the only thing a
CPU suite can honestly gate is the arithmetic it will print. That arithmetic is
load-bearing twice: the B/s and per-byte columns must reproduce the two measured
points the task was opened with, and the eval-window column must make the
81 920 B record impossible to misread.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    """Load `scripts/<name>.py` by path — `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load("bench_shape")


def test_rates_reproduce_the_two_measured_points() -> None:
    """b8 x s512 / 355 ms and b24 x s1024 / 600 ms — the task's own numbers."""
    bps, us_b, us_kb = bench.rates(8, 512, 355.0)
    assert bps == pytest.approx(4096 / 0.355)
    assert bps == pytest.approx(11538, rel=1e-3)
    # The brief calls this "86.7 us/KB"; 355 000 / 4 096 is per BYTE.
    assert us_b == pytest.approx(86.67, rel=1e-3)
    assert us_kb == pytest.approx(us_b * 1024)

    bps2, us_b2, us_kb2 = bench.rates(24, 1024, 600.0)
    assert bps2 == pytest.approx(40960)
    assert us_b2 == pytest.approx(24.41, rel=1e-3)
    assert us_kb2 == pytest.approx(us_b2 * 1024)
    # The per-byte gap that motivates the whole probe: 3.55x, the brief's "3.6x".
    assert us_b / us_b2 == pytest.approx(3.55, rel=1e-2)


def test_eval_window_is_eval_batches_times_bytes() -> None:
    assert bench.RECORDED_WINDOW == 20 * 8 * 512 == 81920
    assert bench.eval_window(8, 512) == 81920
    assert bench.eval_window(16, 512) == 163840
    assert bench.eval_window(8, 1024) == 163840
    assert bench.eval_window(16, 1024) == 327680
    assert bench.eval_window(24, 1024) == 491520
    assert bench.batches_for(81920, 8, 512) == 20.0
    assert bench.batches_for(81920, 16, 512) == 10.0
    assert bench.batches_for(81920, 16, 1024) == 5.0
    # b24 x s1024 cannot hit the record exactly: 3 = 73 728 B, 4 = 98 304 B.
    assert bench.batches_for(81920, 24, 1024) == pytest.approx(10 / 3)
    assert bench.eval_window(24, 1024, 3) == 73728
    assert bench.eval_window(24, 1024, 4) == 98304


def test_skip_path_measures_nothing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(bench, "preflight", lambda: "the card is busy")
    assert bench.main() == 0
    out = capsys.readouterr().out
    assert "SKIP" in out
    assert "nothing was measured" in out
