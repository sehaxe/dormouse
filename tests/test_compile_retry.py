"""The compile retry's gate: graph breaks are legal, a recompile storm is not.

`scripts/compile_retry.py` needs the card for its timing columns, but the
verdict it exists to print is pure arithmetic over dynamo's counters. These
gates pin that verdict, including the torch 2.13 detail that `recompile` is not
a counter in this tree and `stats["unique_graphs"]` is the proxy.
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


retry = _load("compile_retry")


def test_the_threshold_is_the_briefs_twenty() -> None:
    assert retry.RECOMPILE_FAIL == 20


def test_the_recompile_counter_wins_where_it_exists() -> None:
    assert retry.recompiles({"recompile": 7, "graphs": 3}) == 7
    # torch 2.13: no `recompile` counter, so captured graphs are the proxy.
    assert retry.recompiles({"recompile": 0, "graphs": 9}) == 9


def test_legal_means_no_fallback_and_no_storm() -> None:
    clean = {"graph_break": 0, "recompile": 0, "graphs": 4, "limit_hits": 0}
    assert retry.legal(clean)
    # A graph break is fine: 40 breaks with 4 graphs is still a legal compile.
    assert retry.legal({**clean, "graph_break": 40})
    assert retry.legal({**clean, "graphs": 20})
    assert not retry.legal({**clean, "graphs": 21})
    # The 0.20x failure was the eager fallback; it is never legal.
    assert not retry.legal({**clean, "limit_hits": 1})
    assert not retry.legal({**clean, "recompile": 21})


def test_delta_is_per_arm_because_dynamos_counters_are_global() -> None:
    before = {"graph_break": 1, "recompile": 0, "graphs": 2, "limit_hits": 0}
    after = {"graph_break": 4, "recompile": 0, "graphs": 6, "limit_hits": 0}
    assert retry.counts_delta(after, before) == {
        "graph_break": 3,
        "recompile": 0,
        "graphs": 4,
        "limit_hits": 0,
    }


def test_dynamo_counts_always_returns_the_four_fields() -> None:
    counts = retry.dynamo_counts()
    assert set(counts) == {"graph_break", "recompile", "graphs", "limit_hits"}
    assert all(isinstance(v, int) for v in counts.values())


def test_skip_path_measures_nothing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(retry, "preflight", lambda: "the card is busy")
    assert retry.main([]) == 0
    out = capsys.readouterr().out
    assert "SKIP" in out
    assert "nothing was measured" in out
