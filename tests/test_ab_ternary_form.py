"""The ternary FORM driver: the three release arms, the window, and the rule.

`scripts/ab_ternary_form.py` must not be run by the suite (it is a GPU driver),
but its arm set, its config resolution and its pre-registered decision rule are
pure functions and are pinned here — a driver whose arms silently drift from
the docstring would produce a table for something other than the release form.
"""

from __future__ import annotations

import argparse
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


ab = _load("ab_ternary_form")


def _args(tmp_path: Path, **over) -> argparse.Namespace:
    base = dict(
        steps=2000,
        batch=8,
        seq=512,
        eval_batches=20,
        eval_every=500,
        log_every=250,
        data=ab.DATA,
        eval=ab.EVAL,
        out_dir=str(tmp_path),
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_the_three_arms_and_the_dropped_fourth():
    assert tuple(ab.ARMS) == ("f", "b", "n")
    assert ab.WINDOW == 81_920, "the protocol's one eval window"
    assert "int8 WEIGHT codec" in ab.DROPPED["a"], "the drop must state why"
    # The arm set is the release-form grid, not an arbitrary list.
    assert ab.EXPECT["f"] == {"dense_qat_arms": "==0", "act_quant_arms": "==0"}
    assert ab.EXPECT["b"] == {"dense_qat_arms": ">0", "act_quant_arms": "==0"}
    assert ab.EXPECT["n"] == {"dense_qat_arms": ">0", "act_quant_arms": ">0"}


@pytest.mark.parametrize(
    ("arm", "dense_qat", "act_quant"),
    [("f", "none", "none"), ("b", "ternary", "none"), ("n", "ternary", "int4")],
)
def test_each_arm_resolves_to_its_form(tmp_path, arm, dense_qat, act_quant):
    run = ab.resolve(arm, 1, str(tmp_path / f"{arm}.jsonl"), _args(tmp_path))
    assert run.model.factor_qat == "ternary"
    assert run.model.dense_qat == dense_qat
    assert run.model.dense_qat_skip_attn is True, "attention is skipped in every arm"
    assert run.model.act_quant == act_quant
    assert run.train.factor_qat_anneal_steps == 500, "one ramp, COMMON to every arm"
    assert run.train.batch * run.train.seq_len * run.train.eval_batches == ab.WINDOW
    assert run.train.seed == 1


def _runs(arm: str, bpbs: list[float], **counters) -> list[dict]:
    return [
        {
            "arm": arm,
            "seed": i + 1,
            "bpb": b,
            "scored": ab.WINDOW,
            "nan": 0,
            "counters": counters,
        }
        for i, b in enumerate(bpbs)
    ]


def test_step_rule_keeps_a_tie_and_reverts_a_larger_loss():
    f = _runs("f", [3.3000, 3.3100, 3.3000], dense_qat_arms=0, act_quant_arms=0)
    b_tie = _runs("b", [3.3040, 3.3060, 3.3050], dense_qat_arms=10, act_quant_arms=0)
    v = ab.step(f, b_tie, ab.WINDOW, "b")
    assert v["verdict"] == "keep", "inside f's spread: keep the compressed form"
    assert v["delta"] == pytest.approx(3.3050 - 3.3033, abs=1e-4)

    b_loss = _runs("b", [3.3600, 3.3600, 3.3600], dense_qat_arms=10, act_quant_arms=0)
    assert ab.step(f, b_loss, ab.WINDOW, "b")["verdict"] == "revert"

    n = _runs("n", [3.3050, 3.3070, 3.3060], dense_qat_arms=10, act_quant_arms=10)
    assert ab.step(b_tie, n, ab.WINDOW, "n")["verdict"] == "keep"
    assert ab.step(b_tie, b_tie, ab.WINDOW, "b")["delta"] == 0.0


def test_step_rule_voids_a_wrong_window_or_a_dead_arm():
    f = _runs("f", [3.3, 3.3, 3.3], dense_qat_arms=0, act_quant_arms=0)
    b = _runs("b", [3.3, 3.3, 3.3], dense_qat_arms=10, act_quant_arms=0)
    short = [dict(r, scored=ab.WINDOW // 2) for r in b]
    assert ab.step(f, short, ab.WINDOW, "b")["verdict"] == "void"
    dead = _runs("b", [3.3, 3.3, 3.3], dense_qat_arms=0, act_quant_arms=0)
    v = ab.step(f, dead, ab.WINDOW, "b")
    assert v["verdict"] == "void" and "dense_qat_arms" in v["reason"]
