"""Gates for the RLVR task corpus (`scripts/build_rl_tasks.py`) and its loader.

The corpus is the thing the RL run trains on, so three properties are gates, not
comments:

- **≥2 000 deterministic tasks**, five kinds, a difficulty spread — the same
  seed is the same bytes in any process.
- **No leakage by RENDERED PROMPT.** The builder hashes the prompt, not the id,
  and refuses a corpus that collides with the SFT corpus (`train.sft._examples`,
  reused rather than re-derived). This is the check the project has paid for
  before: two ids for one question is not two tasks.
- **The verifier accepts the reference answer** for every kind. A corpus of
  unsolvable tasks would make `pass@1` a constant zero and the RL run a no-op
  that looks alive.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dormouse.instrument import LoudError
from dormouse.post.tasks import SFT_TASK_OFFSET, Task, build_tasks, load_rl_tasks
from dormouse.post.verifiers import verify

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_rl_tasks.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_corpus_is_deterministic_and_meets_the_size_target(tmp_path: Path):
    """Same seed → same file bytes; the default build is ≥2 000 tasks over 5 kinds."""
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    ra = _run("--out", str(a), "--per-kind", "400")
    rb = _run("--out", str(b), "--per-kind", "400")
    assert ra.returncode == 0, ra.stderr
    assert rb.returncode == 0, rb.stderr
    assert a.read_bytes() == b.read_bytes(), "one seed must be one corpus, in any process"
    records = _records(a)
    assert len(records) == 2000
    kinds = {r["kind"] for r in records}
    assert kinds == {
        "arith_chain",
        "algebra_rational",
        "unit_conversion",
        "string_list",
        "algorithmic",
    }
    for kind in kinds:
        assert sum(1 for r in records if r["kind"] == kind) == 400
    # The schema is the six fields; `verifier` is a string for math and an object
    # carrying the tests for code.
    for r in records:
        assert set(r) == {"id", "kind", "prompt", "answer", "verifier", "difficulty_hint"}
        assert isinstance(r["difficulty_hint"], float) and 0.0 <= r["difficulty_hint"] <= 1.0
        if r["kind"] in ("arith_chain", "algebra_rational", "unit_conversion"):
            assert r["verifier"] == "math"
        else:
            assert r["verifier"]["cls"] == "code" and r["verifier"]["tests"]
    assert "overlap=0" in ra.stdout, ra.stdout
    assert "corpus sha256:" in ra.stdout


def test_every_rendered_prompt_is_unique_and_disjoint_from_sft(tmp_path: Path):
    """The leak proof is the RENDERED prompt's sha256, in both directions."""
    out = tmp_path / "c.jsonl"
    # The FULL default corpus, not a 20-task slice: a collision can sit at any
    # index, and the builder's own proof is only as strong as what it is run on.
    r = _run("--out", str(out), "--per-kind", "500")
    assert r.returncode == 0, r.stderr
    records = _records(out)
    rl = {hashlib.sha256(rec["prompt"].encode()).hexdigest() for rec in records}
    assert len(rl) == len(records), "two records render the same prompt"
    from dormouse.train.sft import _examples

    sft = {hashlib.sha256(instruction.encode()).hexdigest() for instruction, _ in _examples(1, 32)}
    assert rl.isdisjoint(sft)
    assert "sft=405 overlap=0" in r.stdout, r.stdout


def test_the_verifier_accepts_every_reference_answer(tmp_path: Path):
    """A reference answer that does not verify would make the corpus untrainable."""
    out = tmp_path / "small.jsonl"
    r = _run("--out", str(out), "--per-kind", "2", "--verify-all")
    assert r.returncode == 0, r.stderr
    assert "verified reference answers: 10/10" in r.stdout
    tasks = load_rl_tasks(out)
    assert len(tasks) == 10
    for task in tasks:
        response = f"answer: {task.expected}" if task.cls == "math" else task.solution
        verdict = verify(task.as_dict(), response)
        assert verdict.passed, (task.tid, verdict.reason)


def test_the_loader_refuses_a_duplicate_prompt_and_a_known_bad_verifier(tmp_path: Path):
    """The loader's identity is the prompt hash, not the id; a bad record is LOUD."""
    good = {
        "id": "a",
        "kind": "arith_chain",
        "prompt": "p",
        "answer": "1",
        "verifier": "math",
        "difficulty_hint": 0.1,
    }
    p = tmp_path / "dup.jsonl"
    p.write_text(json.dumps(good) + "\n" + json.dumps(dict(good, id="b")) + "\n")
    with pytest.raises(LoudError, match="duplicate"):
        load_rl_tasks(p)
    p.write_text(json.dumps(dict(good, verifier="oracle")) + "\n")
    with pytest.raises(LoudError, match="not known"):
        load_rl_tasks(p)
    p.write_text(json.dumps({"id": "a"}) + "\n")
    with pytest.raises(LoudError, match="missing"):
        load_rl_tasks(p)


def test_the_sft_offset_keeps_the_generated_pools_apart():
    """`SFT_TASK_OFFSET` is the second line of defence behind the hash proof."""
    sft = build_tasks("math", 7, 8)
    rl = build_tasks("math", 7 + SFT_TASK_OFFSET, 8)
    assert [t.prompt for t in sft] != [t.prompt for t in rl]
    assert {t.tid for t in sft}.isdisjoint({t.tid for t in rl})
    assert {hashlib.sha256(t.prompt.encode()).hexdigest() for t in sft}.isdisjoint(
        {hashlib.sha256(t.prompt.encode()).hexdigest() for t in rl}
    )


def test_the_loaded_corpus_keeps_the_response_format_as_the_thrift_reference(tmp_path: Path):
    """A math task's correct RESPONSE is `answer: <value>`, so that is `solution`.

    `reward`'s `S_sol` prices `len(solution)`; carrying the bare value would
    score every correct reply at ~0.1 and make the thrift term dominate the
    verdict. The loader pins the reference the response is actually compared to.
    """
    out = tmp_path / "one.jsonl"
    assert _run("--out", str(out), "--per-kind", "1").returncode == 0
    tasks = load_rl_tasks(out)
    math = next(t for t in tasks if t.cls == "math")
    assert math.solution == f"answer: {math.expected}"
    code = next(t for t in tasks if t.cls == "code")
    assert code.solution.startswith("def ")
    assert isinstance(code, Task)
