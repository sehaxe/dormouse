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

It runs FIVE cases: the `tiny` fixture (fast, d=64, dense blend), the real
`small` polygon (d=768, 12 heads, 4 iterations, 3 experts + the rank-192 shared
one, 32 768-row Engram tables) — the shape the demo page runs, the full
`medium`-shaped export (d=1024, 256 experts, top-2 dropless, 50.3M-row Engram
table at fp16) — the 138 MB geometry the size table is about, the routed `tiny`
fixture (d=64, 4 experts, top-2 dropless, no Engram), and that same routed
fixture exported through the page's memory-off arm
(`write_export(no_engram=True)`). The `small` case is the one that makes "the
browser runs the small preset" a measurement; the medium case is the one that
makes "the browser runs a medium-shaped top-k export" a measurement at every
packed factor; the no-Engram case is what `?engram=0` loads.

The engine implements both MoE arms: the dense blend (`moe_topk = 0`, the
`small` polygon) and the DROPLESS top-k dispatch (`moe_topk = k`,
`moe_capacity = 0`). The tiny dense fixture runs `moe_topk = 0`; the routed
fixture is `tests/conftest.py`'s TINY geometry (`n_experts = 4`) with
`moe_topk = 2`, `moe_capacity = 0` and the Engram OFF — the smallest model that
walks `experts.py`'s routed arm, so the gate measures the top-k normalisation,
the expert weighting and the shared expert, not the dense control's.
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
    # `recurrent_decode`'s own rule: no Engram means no left context to carry.
    drop = 0 if model.loop.engram is None else max(model.loop.engram.orders) - 1
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


def _parity_line(out: str) -> str:
    """The gate's own number, in whichever reporter the local node defaults to.

    node's spec reporter (the default from v22) prints a test's `console.log` as
    `# parity ...`; the TAP reporter (the default on v20, and on a CI runner)
    ESCAPES the leading hash as `# \\# parity ...`. Same line, two spellings — a
    gate that reads only one of them is a gate that fails on the other machine,
    which is exactly what happened: green here, red on the runner.
    """
    for raw in out.splitlines():
        line = raw.replace("# \\# ", "# ")
        if line.startswith("# parity"):
            return line
    raise AssertionError(f"no parity line in the engine test's output:\n{out[-2000:]}")


def _parity_case(
    model: DormouseModel, tmp_path: Path, label: str, *, no_engram: bool = False
) -> str:
    """Export → Python reference → `node --test`, one case end to end."""
    artifact = tmp_path / f"{label}.dormouse"
    write_export(model, artifact, preset=label, seed=1, engram_dtype="fp16", no_engram=no_engram)
    f = read_export(artifact)
    # the artifact must be one the PAGE can load, not just one this test can
    # read: the refusal is the writer's, and the loader's is the same table.
    assert f.engine_refusal() is None, f"{label}: {f.engine_refusal()}"
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
    line = _parity_line(out)
    print(line)  # the number, on the test's own output: evidence, not a vibe
    assert "max |Δlogit|" in line


def test_engine_matches_python_on_the_small_polygon(tmp_path: Path) -> None:
    """The real thing: d=768, 12 heads, 4 iterations, 3 experts, 32 768-row tables."""
    torch.manual_seed(0)
    model = DormouseModel(load_preset("small"))
    out = _parity_case(model, tmp_path, "small")
    line = _parity_line(out)
    # the measured number, printed so a reader of the log sees it — the gate
    # itself is the JS assertion, not this line
    print(line)
    assert "max |Δlogit|" in line


def test_engine_matches_python_on_the_medium_shaped_export(tmp_path: Path) -> None:
    """The 138 MB geometry the page's size table is about: d1024, E256, top-2.

    Random init (the gate is arithmetic, not quality) and the engine's own arms
    (msa off, dropless); the export is 138,058,200 B and the two sides read the
    same bytes. This is the case that exercises every packed ternary factor and
    the fp16 Engram table at the scale the demo quotes.
    """
    torch.manual_seed(0)
    cfg = dataclasses.replace(load_preset("medium"), bf16=False, use_msa=False, moe_capacity=0.0)
    model = DormouseModel(cfg)
    out = _parity_case(model, tmp_path, "medium-shaped")
    line = _parity_line(out)
    print(line)
    assert "max |Δlogit|" in line


def test_engine_matches_python_on_the_routed_dropless_fixture(tmp_path: Path) -> None:
    """The top-k arm: 4 experts, top-2, `moe_capacity=0`, no Engram.

    The routing is per position, so the one-byte decode step and Python's
    `_step_logits` are the same function: softmax over all 4, the top 2
    renormalised, expert + shared + `down`.
    """
    torch.manual_seed(0)
    model = tiny_model(use_engram=False, moe_capacity=0.0)
    out = _parity_case(model, tmp_path, "routed-tiny")
    line = _parity_line(out)
    print(line)
    assert "max |Δlogit|" in line


def test_engine_matches_python_with_the_engram_off_export(tmp_path: Path) -> None:
    """The page's `?engram=0` arm: the exporter drops the table and the cfg says so.

    The model HAS an Engram; `write_export(no_engram=True)` removes its records
    and writes `use_engram=false`. Both sides then run the reference's
    "engram is None" branch (the memory arm is an exact zero), so the parity
    gate covers the same engine path the smaller page artifact loads.
    """
    torch.manual_seed(0)
    model = tiny_model(moe_topk=2, moe_capacity=0.0)
    assert model.loop.engram is not None, "this case is only meaningful with a table to drop"
    out = _parity_case(model, tmp_path, "routed-tiny-engram-off", no_engram=True)
    line = _parity_line(out)
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
