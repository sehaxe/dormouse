"""The eval-window overlap gate: a window inside the training corpus is not held out.

Found by the data/eval audit (2026-11-08): the `off=100,000,000` window on
`best/fwedu.bin` was scored while the run's own fwedu cursor had passed
108.9-110.0 MB, so the "held-out" BPB was reading training bytes. The gate is a
path comparison — the cheapest honest detector — and it refuses loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from conftest import tiny_run  # noqa: E402
from eval_window import refuse_training_overlap, training_paths  # noqa: E402

from dormouse.instrument import LoudError  # noqa: E402


def _run_with(tmp_path: Path, **over):
    return tiny_run(tmp_path, **over)


def test_a_window_on_the_data_file_is_refused(tmp_path):
    run = _run_with(tmp_path)
    data = Path(run.train.data)
    with pytest.raises(LoudError, match="not held out"):
        refuse_training_overlap(run, data, allow=False)
    # the escape works and is the caller's explicit choice
    refuse_training_overlap(run, data, allow=True)


def test_a_separate_eval_file_passes(tmp_path):
    run = _run_with(tmp_path)
    other = tmp_path / "eval.bin"
    other.write_bytes(b"\x00" * 16)
    refuse_training_overlap(run, other, allow=False)


def test_every_mix_source_counts_as_training(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"\x01" * 32)
    plan = tmp_path / "mix.toml"
    plan.write_text(f'[sources]\na = "{src}"\n', encoding="utf-8")
    run = _run_with(tmp_path, mix_plan=str(plan))
    assert src.resolve() in training_paths(run)
    with pytest.raises(LoudError, match="not held out"):
        refuse_training_overlap(run, src, allow=False)
