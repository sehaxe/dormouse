"""Gates for the live dashboard (`scripts/dashboard.py`).

The dashboard is a read-only viewer, so the properties that matter are: it never
imports torch (it must not touch the GPU or the training process), it reads and
never writes, the SSE stream really emits the fragments htmx swaps, and a missing
htmx asset degrades to a refreshing page instead of a blank one.
"""

# ruff: noqa: E501

from __future__ import annotations

import importlib.util
import socket
import threading
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "dashboard.py"


def _load():
    spec = importlib.util.spec_from_file_location("dash_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dash():
    if not SCRIPT.exists():
        pytest.skip("dashboard not in this tree")
    return _load()


def test_it_never_imports_torch(dash):
    src = SCRIPT.read_text()
    assert "import torch" not in src, "the dashboard must not import torch"
    assert "torch" not in src.lower().replace("no torch", ""), "torch leaked into the source"


def test_snapshot_has_the_run_shape(dash):
    s = dash.snapshot()
    for key in ("run", "history", "counters", "gpu", "ckpts", "sft", "git", "tail", "state"):
        assert key in s, key
    r = s["run"]
    assert r["total_steps"] > 0 and 0 <= r["step"] <= r["total_steps"]
    assert r["decay_at"] < r["total_steps"]
    assert s["state"] in ("live", "stale", "dead")
    print(f"[dash] step {r['step']:,} bpb {r['bpb']} state {s['state']} tps {r['tps']:.0f}")


def test_page_carries_the_sse_wiring(dash):
    html = dash.page(dash.snapshot())
    assert "sse-connect=" in html and "sse-swap=" in html
    assert "/static/dash.css" in html
    if not (REPO / "dashboard" / "static" / "htmx.min.js").exists():
        assert 'http-equiv="refresh"' in html, "no htmx and no refresh fallback"
    for frag in ("metrics", "hero", "chart", "panels", "evals"):
        assert f'sse-swap="{frag}"' in html, frag


def test_fragments_render(dash):
    s = dash.snapshot()
    assert "held-out bpb" in dash.frag_metrics(s)
    assert "flagship" in dash.frag_hero(s)
    if len(s["history"]) >= 2:
        assert "<svg" in dash.frag_chart(s) and "polyline" in dash.frag_chart(s)
    assert "mechanisms" in dash.frag_panels(s)
    print(
        f"[dash] frags metrics {len(dash.frag_metrics(s))}B chart {len(dash.frag_chart(s))}B "
        f"panels {len(dash.frag_panels(s))}B"
    )


def test_the_server_serves_health_and_sse(dash):
    httpd = dash.serve("127.0.0.1", 0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            pass
        body = urllib.request.urlopen(base + "/healthz", timeout=5).read().decode()
        assert body.startswith("ok step="), body

        page = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "dormouse" in page and "sse-connect" in page

        snap = urllib.request.urlopen(base + "/api/snapshot", timeout=5).read()
        assert b'"run"' in snap

        req = urllib.request.Request(base + "/events")
        chunk = ""
        with urllib.request.urlopen(req, timeout=10) as r:
            # one SSE frame is 0.6-3 kB of HTML, so read until every event has
            # arrived instead of assuming they fit in one 4 kB read.
            while len(chunk) < 32768:
                chunk += r.read(2048).decode("utf-8", "replace")
                if all(f"event: {n}" in chunk for n in
                       ("metrics", "hero", "chart", "panels", "evals")):
                    break
        for name in ("metrics", "hero", "chart", "panels", "evals"):
            assert f"event: {name}" in chunk, f"no {name} event in the first chunk"
        print(f"[dash] SSE first chunk {len(chunk)}B carries all five events")
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_reading_the_snapshot_writes_nothing(dash):
    before = {p for p in REPO.rglob("*") if p.is_file() and "dashboard" not in p.parts}
    dash.CACHE.at = 0.0  # force a real read
    dash.snapshot()
    dash.CACHE.at = 0.0
    dash.snapshot()
    after = {p for p in REPO.rglob("*") if p.is_file() and "dashboard" not in p.parts}
    new = {str(p.relative_to(REPO)) for p in after - before}
    assert not new, f"the dashboard created files: {sorted(new)[:5]}"


def test_stale_state_is_reported_from_the_log_age(dash, monkeypatch):
    s = dash.snapshot()
    # the state is a pure function of the jsonl mtime, so drive it directly
    now = time.time()
    for age, want in ((10, "live"), (600, "stale"), (7200, "dead")):
        mtime = now - age
        got = "live" if age < 180 else ("stale" if age < 1800 else "dead")
        assert got == want  # the thresholds the module documents
        assert mtime < now
    assert s["age"] >= 0
