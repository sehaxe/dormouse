"""The routing-health panel: three levels, and the red line only at collapse.

`route_hist.py --json` is the producer, `dashboard.frag_routing` the consumer;
one schema test pins them together without a model, a GPU or a live run.
"""

# ruff: noqa: E501

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _counters(live: int, top10_x100: int, max_x100: int, e: int = 512) -> dict:
    return {
        "moe_live_experts": live,
        "moe_n_experts": e,
        "moe_top10_share_x100": top10_x100,
        "moe_max_share_x100": max_x100,
    }


def _render(dash, counters: dict, e: int = 512) -> str:
    return dash.frag_routing({"counters": counters, "run": {"experts": e}})


def test_routing_panel_three_health_levels():
    dash = _load("dashboard")
    healthy = _render(dash, _counters(480, 1800, 250))
    degraded = _render(dash, _counters(320, 3500, 700))
    collapsed = _render(dash, _counters(80, 50.1, 8.3))

    assert "route-ok" in healthy and "healthy" in healthy
    assert "route-warn" in degraded and "degraded" in degraded
    assert "route-bad" in collapsed and "collapsed" in collapsed
    assert "route-bad" not in healthy and "route-bad" not in degraded

    # The bar is live/E, and the top-10 share is the number the collapse showed.
    assert "width:93.8%" in healthy
    assert "width:62.5%" in degraded
    assert "width:15.6%" in collapsed
    assert "50.1%" in collapsed and "8.3%" in collapsed

    # The red line is STRICTLY below 50%: exactly half is degraded, not red.
    edge = _render(dash, _counters(256, 900, 200))
    assert "route-warn" in edge and "route-bad" not in edge


def test_routing_panel_reads_the_last_histogram_file(tmp_path: Path, monkeypatch):
    dash = _load("dashboard")
    path = tmp_path / "m400.route.json"
    path.write_text(
        json.dumps(
            {
                "n_experts": 512,
                "live_experts": 80,
                "top10_share_x100": 50.1,
                "max_share_x100": 8.3,
            }
        )
    )
    monkeypatch.setattr(dash, "ROUTING_JSON", path)
    r = dash.routing_data({"counters": {}, "run": {"experts": 512}})
    assert (r["live"], r["experts"], r["state"]) == (80, 512, "bad")
    assert r["source"] == str(path)
    html = dash.frag_routing({"routing": r})
    assert "route-bad" in html and "50.1%" in html


def test_routing_panel_degrades_without_data(tmp_path: Path, monkeypatch):
    dash = _load("dashboard")
    monkeypatch.setattr(dash, "ROUTING_JSON", tmp_path / "missing.json")
    html = dash.frag_routing({"counters": {}, "run": {"experts": 512}})
    assert "no routing health" in html and "route_hist.py --json" in html
    assert "route-bad" not in html


def test_route_hist_json_is_the_panel_schema(tmp_path: Path, monkeypatch):
    dash = _load("dashboard")
    hist = _load("route_hist")
    rec = hist.summary([10] * 80 + [0] * 432, 512, ckpt="m400")
    assert rec["live_experts"] == 80 and rec["n_experts"] == 512
    assert rec["assignments"] == 800
    assert rec["top10_share_x100"] == 12.5
    assert rec["max_share_x100"] == 1.25

    path = tmp_path / "route.json"
    hist.write_json(path, rec)
    monkeypatch.setattr(dash, "ROUTING_JSON", path)
    r = dash.routing_data({"counters": {}, "run": {"experts": 512}})
    assert (r["live"], r["experts"], r["state"]) == (80, 512, "bad")
    assert r["top10"] == 0.125 and r["max_share"] == 0.0125
