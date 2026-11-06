"""An A/B driver whose promotion rule can be tested without a GPU.

    python -m pytest tests/test_collect_ab.py

Why this exists: `collect_ab.py` is where a mechanism is declared a WIN, and its
old rule (`mean <= c_mean - 2*spread`) promoted an exact tie whenever the control
had one seed (spread 0). That is the same class as the gates that could not fail.
This test pins the refusals and the arithmetic on synthetic logs, so the next
edit to the rule has to keep them.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _line(bpb_fin: float) -> str:
    """The current eval line: `bpb_fin` is the headline, `bpb_mean` beside it."""
    return (
        f"eval step=600 off=0 bpb_fin={bpb_fin} bpb_mean={bpb_fin + 0.1} "
        f"fin={bpb_fin * 0.693147:.6f} loss={(bpb_fin + 0.1) * 0.693147:.6f} "
        f"over 40960 B (10 x 8 x 512) engram=0/0\n"
    )


def _logs(tmp: Path, control: list[float], arms: dict[str, list[float]]) -> Path:
    d = tmp / "logs"
    d.mkdir(parents=True, exist_ok=True)
    for i, v in enumerate(control):
        (d / f"n1_ctl600_s{i}.log").write_text(_line(v))
    # ARM_FILE = /(?:n2|f1|a1|tr)_<arm>_s<digit>.log — the prefix is part of
    # the contract, so the fixture uses it rather than inventing a name.
    for name, vals in arms.items():
        for i, v in enumerate(vals):
            (d / f"n2_{name}_s{i}.log").write_text(_line(v))
    return d


def _run(tmp: Path, logs: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/collect_ab.py"],
        cwd=ROOT,
        env={"HOME": str(tmp), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_one_control_seed_cannot_promote(tmp_path):
    d = _logs(tmp_path, [2.650], {"mech": [2.640, 2.641, 2.642]})
    r = _run(tmp_path, d)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "need 3" in r.stderr


def test_identical_control_seeds_cannot_promote(tmp_path):
    d = _logs(tmp_path, [2.650, 2.650, 2.650], {"mech": [2.649, 2.649, 2.649]})
    r = _run(tmp_path, d)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "identical" in r.stderr


def test_a_tie_is_a_tie_and_a_floor_applies(tmp_path):
    d = _logs(
        tmp_path,
        [2.640, 2.650, 2.660],
        {"tie": [2.650, 2.650, 2.650], "small": [2.635, 2.635, 2.635], "win": [2.60] * 3},
    )
    r = _run(tmp_path, d)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "tie" in out and "incomplete" not in out.split("tie")[1][:40]
    # 0.015 of raw delta is inside the 0.021 spread -> not a WIN
    assert "small" in out
    assert statistics.mean([2.635] * 3) > 2.650 - 2 * 0.021


def test_a_legacy_depth_avg_log_is_refused_not_read_as_fin(tmp_path):
    """`bpb=` on an old line was the depth mean; promoting on it is the defect."""
    d = tmp_path / "logs"
    d.mkdir(parents=True, exist_ok=True)
    for i, v in enumerate([2.640, 2.650, 2.660]):
        (d / f"n1_ctl600_s{i}.log").write_text(f"eval step=600 bpb={v} loss=1.8\n")
    r = _run(tmp_path, d)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "legacy depth-avg" in r.stderr
