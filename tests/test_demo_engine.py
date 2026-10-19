r"""The browser engine's gate: Python's logits vs `engine.js`'s, on real files.

WHAT THIS PINS, and why it is the test that matters. A demo that renders bytes
is easy to fake; a demo that renders OUR model's bytes is a claim, and the
claim is only checkable against the Python reference. So this file:

1. builds a model, writes the `.dormouse` artifact `dormouse export` writes,
2. loads that artifact BACK in Python (the same reader `decode --export` uses)
   with `bf16=False` — the engine is its fp32 arm, and bf16 is a compute
   setting of the trained run, not part of the architecture,
3. walks the reference decode path (`decode._step_logits`, the state-carry
   step) over a prompt, recording the logits at every prefix,
4. runs `node --test docs-site/demo_test/engine.test.mjs` with the artifact and
   the reference logits in the environment, and
5. fails if Node exits non-zero — the JS side asserts max |Δlogit| < 1e-4 and
   an identical argmax at every prefix.

It runs TWICE: on the `tiny` fixture (fast, d=64) and on the real `small`
polygon (d=768, 12 heads, 4 iterations, 3 experts + the rank-192 shared one,
32 768-row Engram tables) — the shape the demo page runs. The `small` case is
the one that makes "the browser runs the small preset" a measurement.

The engine does NOT implement the top-k dropless MoE path (`moe_topk > 0`), so
the tiny fixture is built with `moe_topk=0` — the `small` preset's dense blend,
the arm the demo runs.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from conftest import tiny_model

from dormouse.config import load_preset
from dormouse.instrument import Counters
from dormouse.model import DormouseModel
from dormouse.model.decode import _step_logits
from dormouse.model.export import read_export, write_export

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "docs-site" / "public" / "demo" / "engine.js"
ENGINE_TEST = ROOT / "docs-site" / "demo_test" / "engine.test.mjs"
PROMPT = b"hello, dormouse"

#: The gate's stated tolerance, mirrored from `engine.test.mjs` (and asserted
#: there): max |Δlogit| < 1e-4, measured 2.98e-7 (tiny) / 1.058e-6 (`small`).
TOLERANCE = 1e-4


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed: the engine gate cannot run here")
    return node


def _reference_logits(model: DormouseModel, prompt: bytes) -> list[list[float]]:
    """The decode path's logits at every prefix of `prompt` (the engine's own loop)."""
    model.eval()
    states: list = [None] * model.cfg.max_iter
    drop = max(model.loop.engram.orders) - 1
    rows: list[list[float]] = []
    with torch.no_grad():
        for i in range(len(prompt)):
            logits, states = _step_logits(
                model, list(prompt[: i + 1]), states, torch.device("cpu"), drop, Counters()
            )
            rows.append([float(v) for v in logits])
    return rows


def _run_engine_test(artifact: Path, ref: dict, tmp_path: Path) -> str:
    """`node --test` with the artifact injected; returns its stdout, or fails."""
    ref_file = tmp_path / "reference.json"
    ref_file.write_text(json.dumps(ref))
    proc = subprocess.run(
        [_node(), "--test", str(ENGINE_TEST)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(Path.home()),
            "DM_PARITY_ARTIFACT": str(artifact),
            "DM_PARITY_JSON": str(ref_file),
        },
        check=False,
    )
    assert proc.returncode == 0, (
        f"the engine gate failed (node --test exited {proc.returncode}):\n"
        f"{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}"
    )
    assert "parity" in proc.stdout, f"the parity test did not run:\n{proc.stdout[-2000:]}"
    return proc.stdout


def _parity_case(model: DormouseModel, tmp_path: Path, label: str) -> str:
    """Export → Python reference → `node --test`, one case end to end."""
    artifact = tmp_path / f"{label}.dormouse"
    write_export(model, artifact, preset=label, seed=1, engram_dtype="fp16")
    f = read_export(artifact)
    # The reference is the SAME bytes with fp32 compute: the engine has no bf16
    # arm, and bf16 is a precision setting of the trained run, not architecture.
    ref_model = DormouseModel(dataclasses.replace(f.cfg, bf16=False))
    ref_model.load_state_dict(f.tensors(), strict=True)
    ref = {
        "label": label,
        "prompt": list(PROMPT),
        "logits": _reference_logits(ref_model, PROMPT),
        "tolerance": TOLERANCE,
    }
    return _run_engine_test(artifact, ref, tmp_path)


def test_engine_matches_python_on_the_tiny_fixture(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = tiny_model(moe_topk=0)  # the dense blend the engine implements
    out = _parity_case(model, tmp_path, "tiny")
    line = next(ln for ln in out.splitlines() if ln.startswith("# parity"))
    print(line)  # the number, on the test's own output: evidence, not a vibe
    assert "max |Δlogit|" in line


def test_engine_matches_python_on_the_small_polygon(tmp_path: Path) -> None:
    """The real thing: d=768, 12 heads, 4 iterations, 3 experts, 32 768-row tables."""
    torch.manual_seed(0)
    model = DormouseModel(load_preset("small"))
    out = _parity_case(model, tmp_path, "small")
    line = next(ln for ln in out.splitlines() if ln.startswith("# parity"))
    # the measured number, printed so a reader of the log sees it — the gate
    # itself is the JS assertion, not this line
    print(line)
    assert "max |Δlogit|" in line


def test_the_engine_is_a_plain_module_with_no_build_step() -> None:
    """No imports, no bundler: the page loads this file as-is."""
    src = ENGINE.read_text()
    assert "export " in src
    for forbidden in ("import ", "require(", "from '"):
        assert forbidden not in src, f"engine.js must stay dependency-free ({forbidden!r} found)"


def test_the_demo_directory_says_the_artifact_is_generated() -> None:
    """`small` is 16.3 MB, over the 8 MB commit rule — the page fetches it."""
    readme = ENGINE.parent / "README.md"
    assert readme.is_file(), "docs-site/public/demo/README.md documents how the blob is made"
    text = readme.read_text()
    assert "dormouse export" in text, text
    assert "model.dormouse" in text, text
