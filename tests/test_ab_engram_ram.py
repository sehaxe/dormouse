"""The host-Engram bias-correction driver's arms and pre-registered rule.

`scripts/ab_engram_ram.py` must not be run by the suite (it is a GPU driver),
but its arm set, its counters and its decision rule are pure functions and are
pinned here — a driver whose arms silently drift from the docstring would
produce a table for something other than the formula it claims to test.
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


ab = _load("ab_engram_ram")


def test_the_three_arms_are_one_pair_of_flags():
    assert tuple(ab.ARMS) == (
        ("vram", False, True),
        ("ram_old_formula", True, False),
        ("ram_corrected", True, True),
    ), "the control and the two RAM arms differ only in the substrate/formula"
    assert ab.EVAL_BATCHES * ab.BATCH * ab.SEQ == 81_920, "the protocol's one eval window"
    assert (ab.PRESET, ab.STEPS, ab.SEEDS) == ("medium", 600, (1, 2, 3))
    assert ab.DATA.endswith("mix4/web.bin") and ab.EVAL.endswith("mix4/web_eval.bin")
    # The substrate counter is what makes an arm NOT_ENGAGED if it did not run.
    assert ab.EXPECT["vram"] == {"engram_rows": ">0", "engram_host_updates": "==0"}
    assert ab.EXPECT["ram_old_formula"]["engram_host_updates"] == ">0"
    assert ab.EXPECT["ram_corrected"]["engram_host_updates"] == ">0"


def _runs(bpbs: list[float], *, host: int, scored: int = 81_920, nan: int = 0) -> list[dict]:
    return [
        {
            "seed": i + 1,
            "bpb": b,
            "scored": scored,
            "nan": nan,
            "ms_per_step": 100.0,
            "tok_s": 1.0,
            "counters": {
                "engram_rows": 1000,
                "engram_host_updates": host,
                "engram_host_update_rows": 10,
            },
        }
        for i, b in enumerate(bpbs)
    ]


def _arms(vram: list[float], old: list[float], new: list[float], **kw) -> dict:
    return ab.verdict(
        _runs(vram, host=0, **kw), _runs(old, host=1, **kw), _runs(new, host=1, **kw), 81_920
    )


def test_held_when_the_gap_reproduces_and_the_correction_closes_it():
    v = _arms([2.649, 2.651, 2.650], [3.476, 3.480, 3.478], [2.652, 2.655, 2.651])
    assert v["verdict"] == "held", v
    assert v["old_delta"] > 0.8 and abs(v["new_delta"]) < 0.01


def test_insufficient_when_the_gap_survives_the_correction():
    assert _arms([2.65, 2.651, 2.649], [3.47, 3.48, 3.46], [2.70, 2.71, 2.69])["verdict"] == (
        "insufficient"
    )


def test_no_effect_when_it_closes_but_not_by_more_than_the_spread():
    v = _arms([2.650, 2.652, 2.648], [2.660, 2.664, 2.662], [2.635, 2.655, 2.675])
    assert v["verdict"] == "no-effect", v


def test_not_needed_when_the_screen_does_not_reproduce_the_gap():
    assert _arms([2.650, 2.652, 2.648], [2.651, 2.653, 2.649], [2.650, 2.651, 2.649])[
        "verdict"
    ] == "not-needed"


def test_harmful_when_the_corrected_arm_is_worse_than_control():
    assert _arms([2.650, 2.652, 2.648], [2.651, 2.653, 2.649], [2.70, 2.71, 2.69])["verdict"] == (
        "harmful"
    )


def test_void_when_a_void_check_fails():
    v = _arms([2.65, 2.65, 2.65], [3.47, 3.47, 3.47], [2.65, 2.65, 2.65], scored=20_480)
    assert v["verdict"] == "void" and "eval window" in v["reason"]
    v = _arms([2.65, 2.65, 2.65], [3.47, 3.47, 3.47], [2.65, 2.65, 2.65], nan=1)
    assert v["verdict"] == "void" and "NaN" in v["reason"]
    # The control must not show a host update; a RAM arm must show one.
    v = ab.verdict(
        _runs([2.65] * 3, host=1), _runs([3.47] * 3, host=1), _runs([2.65] * 3, host=1), 81_920
    )
    assert v["verdict"] == "void" and "wrong substrate" in v["reason"]
    v = ab.verdict(
        _runs([2.65] * 3, host=0), _runs([3.47] * 3, host=0), _runs([2.65] * 3, host=0), 81_920
    )
    assert v["verdict"] == "void" and "did not run" in v["reason"]
