"""The attention A/B driver's arms and pre-registered rule.

`scripts/ab_attn.py` must not be run by the suite (it is a GPU driver), but its
arm set, its counters and its decision rule are pure functions and are pinned
here — a driver whose arms silently drift from the docstring would produce a
table for something other than the comparison it claims to test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    """Load `scripts/<name>.py` by path — `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ab = _load("ab_attn")


def test_the_two_arms_are_one_flag():
    assert ab.ARMS == ("kda", "full"), "the control and the arm differ only in attn_kind"
    assert ab.COMMON_MODEL["use_kda"] is True, (
        "use_kda is the hard on/off switch; both arms must run attention"
    )
    assert (ab.PRESET, ab.STEPS, ab.SEEDS) == ("medium", 600, (1, 2, 3))
    assert ab.EVAL_BATCHES * ab.BATCH * ab.SEQ == 81_920, "the protocol's one eval window"
    assert ab.DATA.endswith("mix4/web.bin") and ab.EVAL.endswith("mix4/web_eval.bin")
    # The battery's COMMON, minus mix_plan (the docstring's declared deviation:
    # it would make the named --data file inert).
    assert "mix_plan" not in ab.COMMON_TRAIN and ab.COMMON_TRAIN["engram_ram"] is True
    assert (ab.COMMON_TRAIN["batch"], ab.COMMON_TRAIN["seq_len"]) == (8, 512)


def test_the_counters_split_the_arms():
    assert ab.EXPECT["kda"] == {"forward": ">0", "attn_full_forward": "==0"}
    assert ab.EXPECT["full"]["attn_full_forward"] == ">0"
    assert ab.EXPECT["full"]["kda_fla"] == "==0" and ab.EXPECT["full"]["kda_sdpa"] == "==0"


def _runs(bpbs: list[float], *, full: bool, scored: int = 81_920, nan: int = 0) -> list[dict]:
    counters = (
        {"forward": 600, "attn_full_forward": 600}
        if full
        else {"forward": 600, "attn_full_forward": 0}
    )
    return [
        {
            "seed": i + 1,
            "bpb": b,
            "scored": scored,
            "nan": nan,
            "ms_per_step": 120.0 if full else 100.0,
            "tok_s": 1.0,
            "counters": counters,
        }
        for i, b in enumerate(bpbs)
    ]


def test_win_is_beat_the_control_spread():
    # control spread 0.004, arm is 0.020 better: win.
    v = ab.verdict(
        _runs([2.000, 2.004, 2.002], full=False),
        _runs([1.980, 1.982, 1.981], full=True),
        81_920,
    )
    assert v["verdict"] == "win", v
    assert v["delta"] < 0 and v["ms_delta"] > 0, "cost is reported separately"


def test_tie_keeps_the_cheaper_control():
    v = ab.verdict(
        _runs([2.000, 2.004, 2.002], full=False),
        _runs([1.998, 2.003, 2.001], full=True),
        81_920,
    )
    assert v["verdict"] == "tie", v


def test_void_when_a_void_check_fails():
    v = ab.verdict(
        _runs([2.0, 2.0, 2.0], full=False, scored=20_480),
        _runs([1.9, 1.9, 1.9], full=True),
        81_920,
    )
    assert v["verdict"] == "void" and "eval window" in v["reason"]
    # The control must NOT run full attention, and the arm must not run KDA.
    bad = _runs([2.0, 2.0, 2.0], full=False)
    bad[0]["counters"]["attn_full_forward"] = 3
    v = ab.verdict(bad, _runs([1.9, 1.9, 1.9], full=True), 81_920)
    assert v["verdict"] == "void" and "wrong arm" in v["reason"]
    bad = _runs([1.9, 1.9, 1.9], full=True)
    bad[0]["counters"]["kda_sdpa"] = 7
    v = ab.verdict(_runs([2.0, 2.0, 2.0], full=False), bad, 81_920)
    assert v["verdict"] == "void" and "wrong arm" in v["reason"]


def test_ms_per_step_uses_the_same_rate_instrument():
    assert ab.ms_per_step(8 * 512 / 0.1) == 100.0
    assert ab.PROFILE_STEPS > 0 and ab.LOG_EVERY > 0
