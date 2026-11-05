"""Gates for `scripts/ab_ffn_form.py` (CPU/meta only).

The driver must resolve every arm through `config.load_run` (the refusals are
part of the experiment), price the FFN slot with `width.ffn_params`, and print
the pre-registered rule on `--dry-run` without touching a card or a checkpoint.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dormouse import width  # noqa: E402
from dormouse.instrument import LoudError  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ab = _load("ab_ffn_form")


def test_arms_are_the_three_compressed_ranks_plus_equal_and_full():
    arms = {a.name: a for a in ab.resolve_arms()}
    assert list(arms) == [
        "dense",
        "tsct_r256",
        "tsct_r512",
        "tsct_r1024",
        "tsct_eq",
        "tsct_full",
    ]
    for arm in arms.values():
        cfg = arm.run.model
        assert cfg.use_tsct is False, "the non-FFN stack must be held dense"
        assert cfg.n_experts == 1 and cfg.moe_topk == 0
    assert arms["dense"].run.model.ffn_form == "dense"
    assert arms["tsct_full"].run.model.ffn_rank_gate == min(1536, 4096)
    d = width.ffn_params("dense", 1536, 4096)
    eq = arms["tsct_eq"].row["ffn_slot"]
    assert abs(eq - d) <= d // 1000, f"tsct_eq slot {eq:,} != dense {d:,}"
    for name in ("tsct_r256", "tsct_r512", "tsct_r1024", "tsct_full"):
        assert arms[name].row["ffn_slot"] != d, name


def test_rule_is_the_preregistered_one():
    # Spread 0.02 -> need max(0.04, 0.01) = 0.04.
    win = ab.decide_row([2.64, 2.66], [2.60, 2.60], cparams=100, aparams=90)
    assert win["verdict"] == "WIN"
    loss = ab.decide_row([2.64, 2.66], [2.72, 2.72], cparams=100, aparams=90)
    assert loss["verdict"] == "LOSS"
    tie = ab.decide_row([2.64, 2.66], [2.64, 2.66], cparams=100, aparams=90)
    assert tie["verdict"] == "TIE" and "fewer params" in tie["reason"]
    with pytest.raises(LoudError, match="zero control spread"):
        ab.decide_row([2.65, 2.65], [2.60, 2.61], cparams=100, aparams=90)


def test_dry_run_prints_the_table_and_rule_without_a_card(monkeypatch, capsys):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    rc = ab.main(["--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "rule (preregistered)" in out
    assert "params" in out and "active MAC/B" in out
    assert "dense" in out and "tsct_full" in out
    assert "dry-run: no card" in out
