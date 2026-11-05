"""Gates for `scripts/ab_capacity.py` and `scripts/capacity_math.py` (CPU only).

Three things must be true before the driver may run on the card, and each one
is a refusal, not a convention:

1. an arm outside 5% of the control's active MAC is rejected (a BPB comparison
   at unequal compute prices FLOPs, not capacity);
2. a tie never wins, and a tie goes to the smaller parameter count (the
   pre-registered rule);
3. `--dry-run` touches neither the card nor a checkpoint (it must be safe to
   run beside the live training job).

The two known configs pin the arithmetic: `small` (9 476 993 params, the preset
the parameter tests already pin) and `m400b` (199 275 551 params, the live
199M/512-expert recipe the question is about).
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dormouse.config import load_run  # noqa: E402
from dormouse.instrument import LoudError  # noqa: E402


def _load(name: str):
    """Load `scripts/<name>.py` by path — `scripts/` is not a package.

    The module is registered in `sys.modules` BEFORE exec: `@dataclass` resolves
    `cls.__module__` through `sys.modules` at class-creation time, and a
    spec-loaded module that is not registered there fails with
    `'NoneType' object has no attribute '__dict__'`.
    """
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


capacity_math = _load("capacity_math")
ab = _load("ab_capacity")


# ---------------------------------------------------------------------------
# 1. equal active compute
# ---------------------------------------------------------------------------


def test_equal_compute_rejects_a_pair_outside_five_percent():
    ab.assert_equal_compute(
        [
            {"arm": "A", "active_macs": 1_000},
            {"arm": "ok", "active_macs": 1_049},
            {"arm": "low", "active_macs": 951},
        ]
    )
    with pytest.raises(LoudError, match="UNEQUAL|do not share active compute"):
        ab.assert_equal_compute(
            [
                {"arm": "A", "active_macs": 1_000},
                {"arm": "bad", "active_macs": 1_060},
            ]
        )


def test_equal_compute_requires_the_control():
    with pytest.raises(LoudError, match="no control arm"):
        ab.assert_equal_compute([{"arm": "B", "active_macs": 1_000}])


# ---------------------------------------------------------------------------
# 2. the pre-registered decision rule
# ---------------------------------------------------------------------------


def _row(arm: str, params: int, mean: float, verdict: str, need: float = 0.02) -> dict:
    return {
        "arm": arm,
        "params": params,
        "mean": mean,
        "spread": 0.0,
        "delta": 0.0,
        "need": need,
        "bar": mean - need,
        "verdict": verdict,
    }


def test_a_synthetic_tie_does_not_win():
    control = [2.640, 2.650, 2.660]
    tie = [2.650, 2.650, 2.650]
    d = ab.decide_row(control, tie, control_params=199_000_000, arm_params=100_000_000)
    assert d["verdict"] == "TIE", d
    # The bar is the control mean minus max(2 x 0.020 spread, 0.01) = 2.610.
    assert d["bar"] == pytest.approx(2.610)


def test_the_floor_rejects_a_difference_that_is_not_a_number():
    # A tiny control spread forces the 0.01 floor; 0.005 better is still a tie.
    ctl = [2.640, 2.641, 2.642]  # spread 0.002 -> need 0.01
    d = ab.decide_row(
        ctl, [2.635, 2.635, 2.635], control_params=199_000_000, arm_params=100_000_000
    )
    assert d["verdict"] == "TIE", d
    assert d["need"] == pytest.approx(0.01)


def test_a_real_win_needs_both_conditions():
    ctl = [2.640, 2.650, 2.660]  # spread 0.020 -> need 0.04
    d = ab.decide_row(
        ctl, [2.600, 2.600, 2.600], control_params=199_000_000, arm_params=100_000_000
    )
    assert d["verdict"] == "WIN", d


def test_a_worse_arm_is_a_loss():
    ctl = [2.640, 2.650, 2.660]
    d = ab.decide_row(
        ctl, [2.700, 2.700, 2.700], control_params=199_000_000, arm_params=100_000_000
    )
    assert d["verdict"] == "LOSS", d


def test_one_control_seed_and_a_zero_spread_refuse():
    with pytest.raises(LoudError, match="need 3"):
        ab.decide_row([2.650], [2.60, 2.61, 2.62], control_params=1, arm_params=1)
    with pytest.raises(LoudError, match="identical"):
        ab.decide_row([2.650, 2.650, 2.650], [2.60, 2.61, 2.62], control_params=1, arm_params=1)


def test_ties_go_to_the_smaller_parameter_count():
    rows = [
        _row("A", 199_000_000, 2.650, "CONTROL"),
        _row("B", 100_000_000, 2.650, "TIE"),
        _row("C", 80_000_000, 2.655, "TIE"),
    ]
    assert ab.recommend(rows)["arm"] == "C"
    # A larger tied arm does not beat the control.
    rows[1]["params"] = 300_000_000
    rows[2]["params"] = 400_000_000
    assert ab.recommend(rows)["arm"] == "A"


def test_a_win_is_not_displaced_by_a_cheaper_tie():
    rows = [
        _row("A", 199_000_000, 2.650, "CONTROL"),
        _row("B", 100_000_000, 2.650, "TIE"),
        _row("C", 150_000_000, 2.600, "WIN"),
    ]
    assert ab.recommend(rows)["arm"] == "C"


# ---------------------------------------------------------------------------
# 3. the arm set and the resolved table
# ---------------------------------------------------------------------------


def test_seeds_are_distinct_across_the_sweep():
    every = [seed for spec in ab.ARMS for seed in spec.seeds]
    assert len(every) == len(set(every)), f"duplicate seed(s): {every}"
    assert [len(spec.seeds) for spec in ab.ARMS] == [3] * len(ab.ARMS)


def test_every_resolved_arm_shares_the_control_active_compute():
    arms = ab.resolve_arms()
    assert [a.spec.name for a in arms] == ["A", "B", "C", "D", "E"]
    ab.assert_equal_compute(
        [{"arm": a.spec.name, "active_macs": a.row["active_macs"]} for a in arms]
    )
    by_name = {a.spec.name: a.row for a in arms}
    assert by_name["A"]["params"] == 199_275_551
    # The freed bank parameters are moved into `d_ffn` only as far as the
    # compute target allows; the width must actually move, and the bank must
    # actually shrink. If this stops holding, the sweep is not the question.
    for name in ("B", "C", "E"):
        assert by_name[name]["d_ffn"] > by_name["A"]["d_ffn"], name
        assert by_name[name]["bank"] < by_name["A"]["bank"], name
    assert by_name["D"]["d_ffn"] < by_name["A"]["d_ffn"]


# ---------------------------------------------------------------------------
# 4. the static arithmetic is the pinned arithmetic
# ---------------------------------------------------------------------------


def test_capacity_math_pins_two_known_configs():
    small = capacity_math.measure(load_run("small"))
    assert small["params"] == 9_476_993
    assert small["bank"] == 688_512
    assert small["bank_share"] == pytest.approx(0.0726509, rel=1e-6)
    assert small["active_macs"] == 17_500_416
    # small's dense blend: every one of the 3 experts sees every token.
    assert small["tokens_per_expert_step"] == 8_192.0
    assert small["bytes_for_1m_per_expert"] == 1_000_000

    m400b = capacity_math.measure(load_run(str(ROOT / "configs" / "m400b.toml")))
    assert m400b["params"] == 199_275_551
    assert m400b["bank"] == 134_479_872
    assert m400b["bank_share"] == pytest.approx(0.6748438, rel=1e-6)
    assert m400b["active_macs"] == 28_442_624
    # top-2 of 512 at b8 x s1024: 16 384 assignments / 512 = 32 per expert.
    assert m400b["tokens_per_expert_step"] == 32.0
    assert m400b["bytes_for_1m_per_expert"] == 256_000_000
    assert m400b["tokens_per_expert_at_budget"] == 86_718_750


def test_active_macs_calls_the_project_width_ledger(monkeypatch):
    import dormouse.width as width

    calls: list[tuple] = []
    real = width.macs_per_token

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(width, "macs_per_token", spy)
    capacity_math.active_macs_per_token(load_run("small").model)
    assert calls, "active_macs_per_token did not call dormouse.width.macs_per_token"


# ---------------------------------------------------------------------------
# 5. dry-run safety
# ---------------------------------------------------------------------------


def test_dry_run_touches_no_card_and_no_checkpoint(tmp_path, monkeypatch, capsys):
    import torch

    def _boom(*args, **kwargs):
        raise AssertionError("--dry-run touched CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", _boom)
    monkeypatch.setattr(torch.cuda, "mem_get_info", _boom)
    monkeypatch.setattr(ab.subprocess, "run", _boom)

    out = tmp_path / "out"
    assert ab.main(["--dry-run", "--arms", "A,B", "--out-root", str(out)]) == 0
    text = capsys.readouterr().out
    assert "active MAC/byte" in text and "dry-run: no card" in text
    assert not out.exists(), f"--dry-run created {out}"
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*.config.toml"))


def test_the_compute_table_covers_all_five_arms(capsys):
    arms = ab.resolve_arms()
    ab._print_compute(arms)
    text = capsys.readouterr().out
    for name in ("A", "B", "C", "D", "E"):
        assert f" {name} " in text, text
    # The question's number is visible in the row: A's bank share is 67.5%.
    assert "67.5%" in text


def test_summarize_is_mean_and_max_minus_min():
    mean, spread = ab.summarize([2.64, 2.65, 2.66])
    assert mean == pytest.approx(statistics.mean([2.64, 2.65, 2.66]))
    assert spread == pytest.approx(0.02)
