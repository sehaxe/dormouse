"""The ByteFlow A/B driver's readers and the patch-rate probe's statistics.

Three things are pinned here, none of which needs a GPU or a model build:

* `ab_byteflow.read_run` on a synthetic JSONL — the exact rows the trainer's
  sink writes (`run`, `log`, `eval`) plus the driver's own two (`peak`,
  `patch`): the last eval's BPB, the second-half median rate, ms/step from the
  run row's own batch x seq_len, `nan_skipped` read as the RUNNING total it is
  (the last row, not the sum), and the counters summed across windows;
* `ab_byteflow.verdict` on hand-built runs — the pre-registered AGENTS §1.2 rule
  (a win must beat the CONTROL's own seed spread; a tie deletes) and every void
  path (the channel's `bf_channel` zero, the control's non-zero, a NaN-skipped
  step, an eval window that does not match);
* `patch_rate.patch_lengths` / `patch_stats` on a hand fixture, with the mean
  and median pinned as literals and the two structural invariants the probe
  leans on: a row's lengths sum to the window and the patch count is K.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    """Load `scripts/<name>.py` by path — `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ab = _load("ab_byteflow")
probe = _load("patch_rate")

WINDOW = 20 * 4 * 512  # eval_batches x batch x seq_len, the protocol's window


def _rows() -> list[dict]:
    """The synthetic JSONL: the trainer's rows plus the driver's two."""
    logs = [
        # (step, rate, counters, nan_skipped) — `nan_skipped` is CUMULATIVE.
        (50, 1000.0, {"forward": 50, "bf_channel": 50}, 0),
        (100, 1000.0, {"forward": 50, "bf_channel": 50}, 1),
        (150, 2000.0, {"forward": 50, "bf_channel": 50}, 1),
        (200, 4000.0, {"forward": 50, "bf_channel": 50}, 1),
    ]
    return [
        {"event": "run", "source": "x", "steps": 200, "batch": 4, "seq_len": 512, "lr": 1e-3},
        *(
            {
                "event": "log",
                "step": s,
                "loss": 4.0,
                "lr": 1e-3,
                "tokens_per_s": r,
                "counters": c,
                "nan_skipped": n,
            }
            for s, r, c, n in logs
        ),
        {"event": "eval", "step": 100, "loss": 5.1, "bpb": 5.0, "scored": WINDOW},
        {"event": "eval", "step": 200, "loss": 4.6, "bpb": 4.5, "scored": WINDOW},
        {"event": "peak", "step": 200, "peak_gib": 3.5},
        {"event": "patch", "step": 200, "forwards": 200, "n": 800, "mean": 4.0, "median": 1.0},
    ]


def _write(tmp_path: Path) -> Path:
    path = tmp_path / "channel_s1.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _rows()) + "\n")
    return path


def test_read_run_reads_the_trainers_own_rows(tmp_path: Path) -> None:
    r = ab.read_run(_write(tmp_path), steps=200)
    assert r["bpb"] == 4.5  # the LAST eval, not the first
    assert r["scored"] == WINDOW
    assert r["n_evals"] == 2
    # The second half only: steps 150 and 200 -> median(2000, 4000) = 3000 tok/s.
    assert r["tok_s"] == 3000.0
    # ms/step = batch x seq_len / rate: 4 * 512 / 3000 s.
    assert r["ms_per_step"] == pytest.approx(4 * 512 / 3000.0 * 1e3)
    # `nan_skipped` is a RUNNING total in the trainer: the last row (1), not 3.
    assert r["nan"] == 1
    assert r["counters"] == {"forward": 200, "bf_channel": 200}
    assert r["peak_gib"] == 3.5
    assert r["patch"]["median"] == 1.0


def test_read_run_refuses_a_jsonl_without_an_eval(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text(json.dumps({"event": "run", "batch": 4, "seq_len": 512}) + "\n")
    with pytest.raises(AssertionError, match="no eval row"):
        ab.read_run(path, steps=200)


def _run(seed: int, bpb: float, *, channel: int = 0, nan: int = 0, scored: int = WINDOW) -> dict:
    """One run's reader output. `channel` defaults to 0 — the CONTROL's value."""
    return {
        "seed": seed,
        "bpb": bpb,
        "scored": scored,
        "nan": nan,
        "counters": {"forward": 200, "bf_channel": channel},
    }


CONTROL = [_run(1, 5.0), _run(2, 5.2), _run(3, 5.1)]  # mean 5.1, spread 0.2


def test_verdict_win_beats_the_controls_own_spread() -> None:
    arm = [_run(1, 4.5, channel=200), _run(2, 4.6, channel=200), _run(3, 4.55, channel=200)]
    v = ab.verdict(CONTROL, arm, WINDOW)
    assert v["verdict"] == "win"
    assert v["control_spread"] == pytest.approx(0.2)
    assert v["delta"] == pytest.approx(4.55 - 5.1)


def test_verdict_inside_the_spread_is_a_tie_which_deletes() -> None:
    arm = [_run(1, 5.0, channel=200), _run(2, 5.1, channel=200), _run(3, 5.05, channel=200)]
    v = ab.verdict(CONTROL, arm, WINDOW)
    assert v["verdict"] == "tie"  # 5.05 > 5.1 - 0.2: the delta does not clear the spread


def test_verdict_void_when_the_channels_counter_is_zero() -> None:
    arm = [_run(1, 3.0, channel=0), _run(2, 3.0, channel=0), _run(3, 3.0, channel=0)]
    v = ab.verdict(CONTROL, arm, WINDOW)
    assert v["verdict"] == "void"
    assert "bf_channel=0" in v["reason"]


def test_verdict_void_when_the_control_ran_the_channel() -> None:
    control = [_run(1, 5.0, channel=200), _run(2, 5.2)]
    v = ab.verdict(control, [_run(1, 4.0, channel=200)], WINDOW)
    assert v["verdict"] == "void"
    assert "the wrong shape ran" in v["reason"]


def test_verdict_void_on_nan_or_a_mismatched_window() -> None:
    arm = [_run(1, 3.0, channel=200, nan=1)]
    assert ab.verdict(CONTROL, arm, WINDOW)["verdict"] == "void"
    arm = [_run(1, 3.0, channel=200, scored=WINDOW // 2)]
    v = ab.verdict(CONTROL, arm, WINDOW)
    assert v["verdict"] == "void"
    assert "eval window" in v["reason"]


def test_preflight_on_cpu_skips_the_card_checks() -> None:
    assert ab.preflight("cpu") is None


def test_patch_lengths_sum_to_the_window_and_count_k() -> None:
    sel = torch.tensor([[0, 2, 5, 9]])  # t=10, k=4
    lengths = probe.patch_lengths(sel, t=10)
    assert lengths.tolist() == [[2, 3, 4, 1]]  # diffs + the tail 10 - 9
    assert int(lengths.sum()) == 10
    assert lengths.shape[1] == 4


def test_patch_lengths_refuses_unsorted_boundaries() -> None:
    with pytest.raises(AssertionError, match="strictly ascending"):
        probe.patch_lengths(torch.tensor([[0, 5, 2]]), t=10)


def test_patch_stats_pins_the_hand_fixture() -> None:
    stats = probe.patch_stats(probe.patch_lengths(torch.tensor([[0, 2, 5, 9]]), t=10))
    assert stats["n"] == 4
    assert stats["mean"] == pytest.approx(2.5)  # (2 + 3 + 4 + 1) / 4
    assert stats["median"] == pytest.approx(2.5)  # (2 + 3) / 2
    assert stats["p90"] == pytest.approx(3.7)  # torch.quantile linear: 3 + 0.7*(4-3)
    assert stats["max"] == 4.0
    assert stats["histogram"] == {
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 1,
        "5-8": 0,
        "9-16": 0,
        "17-32": 0,
        "33-64": 0,
        "65+": 0,
    }


def test_patch_stats_mean_is_t_over_k_by_construction() -> None:
    sel = torch.tensor([[0, 1, 4, 7], [0, 3, 6, 7]])  # two windows, t=8, k=4
    stats = probe.patch_stats(probe.patch_lengths(sel, t=8))
    assert stats["n"] == 8
    assert stats["mean"] == pytest.approx(8 / 4)  # the geometry, not the chunker
    assert stats["median"] == pytest.approx(2.0)  # sorted 1,1,1,1,3,3,3,3
    assert stats["histogram"]["1"] == 4
    assert stats["histogram"]["3"] == 4
