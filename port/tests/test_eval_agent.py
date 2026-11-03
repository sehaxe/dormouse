"""The capability harness: four metrics, six gates, no GPU and no checkpoint.

Every arm here is a `ScriptedModel` — a fixed byte script, not a language model
(`post/agent.py` says why that is legitimate for exercising the plumbing). What
the gates prove is that the HARNESS can tell the arms apart: a valid call scores
format/execution 1.0, prose scores 0.0, an out-of-set name lands in
`invalid_name`, an escape attempt is counted with its reason, two runs agree,
a missing checkpoint is a state, and `--limit`/`--seed` sample deterministically.

The gates that need a checkpoint-shaped path assert the `no-model` state; no
weights are read or written, and `CUDA_VISIBLE_DEVICES` is irrelevant because
the harness is CPU-only here.
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_agent", ROOT / "scripts" / "eval_agent.py")
assert _spec is not None and _spec.loader is not None
eval_agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_agent)

SAMPLE = ROOT / "sft_pairs" / "agent_eval_sample.jsonl"


def _report(out: Path, *argv: str) -> dict:
    """Run `main` into `out` and return the parsed report; a crash is a failure."""
    code = eval_agent.main([*argv, "--json-out", str(out)])
    assert code == 0, f"eval_agent exited {code}"
    return json.loads(out.read_text(encoding="utf-8"))


def _ids(report: dict) -> list[str]:
    return [task["id"] for task in report["per_task"]]


# --------------------------------------------------------------------------- (a) the metric sees


def test_gate_a_valid_call_and_prose_are_told_apart(tmp_path: Path):
    """One arm emits a schema-valid call, the other prose; the metrics must differ."""
    valid = _report(
        tmp_path / "valid.json", "--tasks", str(SAMPLE), "--dry-run", "--model", "scripted:valid"
    )
    prose = _report(
        tmp_path / "prose.json", "--tasks", str(SAMPLE), "--dry-run", "--model", "scripted:prose"
    )
    # The sample's 12 math tasks are the prompts that offer `calc`; the 8 code
    # tasks offer no tool, so they are not in the format denominator.
    assert valid["format"]["prompts"] == 12
    assert valid["format"]["rate"] == 1.0
    assert valid["execution"]["accepted"] == 12
    assert valid["execution"]["ran"] == 12
    assert valid["execution"]["rate"] == 1.0
    assert prose["format"]["prompts"] == 12
    assert prose["format"]["accepted"] == 0
    assert prose["format"]["rate"] == 0.0
    assert prose["format"]["no_call"] == 12
    assert prose["execution"]["accepted"] == 0
    assert prose["execution"]["rate"] is None
    assert valid["format"]["rate"] != prose["format"]["rate"]


# --------------------------------------------------------------------------- (b) an out-of-set name


def test_gate_b_out_of_set_name_is_counted_not_accepted(tmp_path: Path):
    report = _report(
        tmp_path / "name.json",
        "--tasks",
        str(SAMPLE),
        "--dry-run",
        "--model",
        "scripted:invalid-name",
    )
    assert report["format"]["prompts"] == 12
    assert report["format"]["accepted"] == 0
    assert report["format"]["invalid_name"] == 12
    assert report["format_invalid_name"] == 12
    assert report["format"]["rate"] == 0.0


# ----------------------------------------------------------------- (c) a real escape attempt


def test_gate_c_escape_attempt_is_a_refusal_with_its_reason(tmp_path: Path):
    report = _report(
        tmp_path / "escape.json",
        "--tasks",
        str(SAMPLE),
        "--dry-run",
        "--model",
        "scripted:escape",
    )
    # The call parsed (format accepts it) and the sandbox screen refused it.
    assert report["format"]["rate"] == 1.0
    assert report["execution"]["accepted"] == 12
    assert report["execution"]["ran"] == 0
    assert report["execution"]["refused"] == 12
    assert report["escape_audit"]["findings"] is True
    assert report["escape_audit"]["attempts"] == 12
    assert report["escape_audit"]["by_reason"] == {"process spawn": 12}
    assert report["state"] == "scripted"


# --------------------------------------------------------------------------- (d) determinism


def test_gate_d_two_runs_are_identical(tmp_path: Path):
    args = ("--tasks", str(SAMPLE), "--dry-run", "--model", "scripted:valid", "--limit", "9")
    first = _report(tmp_path / "first.json", *args)
    second = _report(tmp_path / "second.json", *args)
    assert first == second
    assert first["per_task"] == second["per_task"]


# --------------------------------------------------------------------------- (e) a missing model


def test_gate_e_missing_checkpoint_is_a_state_with_every_metric(tmp_path: Path):
    report = _report(
        tmp_path / "missing.json", "--tasks", str(SAMPLE), "--model", str(tmp_path / "no-such-ckpt")
    )
    assert report["state"] == "no-model"
    assert "missing checkpoint" in report["error"]
    assert report["format"]["rate"] is None
    assert report["execution"]["rate"] is None
    assert set(report["task_success"]) == {"math", "code", "all"}
    for key in ("math", "code", "all"):
        assert report["task_success"][key]["pass@1"] is None
        assert report["task_success"][key]["pass@budget"] is None
    assert report["escape_audit"] == {"attempts": 0, "by_reason": {}, "findings": False}
    assert report["per_task"] == []


# --------------------------------------------------------------------------- (f) sampling


def test_gate_f_limit_and_seed_change_the_sample_deterministically(tmp_path: Path):
    base = ("--tasks", str(SAMPLE), "--dry-run", "--model", "scripted:valid")
    five_a = _report(tmp_path / "5a.json", *base, "--seed", "1", "--limit", "5")
    five_b = _report(tmp_path / "5b.json", *base, "--seed", "1", "--limit", "5")
    five_other = _report(tmp_path / "5c.json", *base, "--seed", "2", "--limit", "5")
    ten = _report(tmp_path / "10.json", *base, "--seed", "1", "--limit", "10")
    assert len(_ids(five_a)) == len(_ids(five_b)) == len(_ids(five_other)) == 5
    assert len(_ids(ten)) == 10
    assert _ids(five_a) == _ids(five_b), "same seed and limit is the same sample"
    assert _ids(five_a) != _ids(five_other), "a different seed draws a different sample"
    assert _ids(five_a) == _ids(ten)[:5], "a smaller limit is a prefix of a larger one"
    assert _ids(five_a) != _ids(ten)


# ------------------------------------------------------- the sample + the CLI edges


def test_the_sample_is_the_shipped_builder_output():
    """20 tasks, 4 of each kind: the RL corpus builder's `--per-kind 4` output."""
    records = [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 20
    assert Counter(record["kind"] for record in records) == {
        "arith_chain": 4,
        "algebra_rational": 4,
        "unit_conversion": 4,
        "string_list": 4,
        "algorithmic": 4,
    }
    assert all(set(record) == {
        "id", "kind", "prompt", "answer", "verifier", "difficulty_hint"
    } for record in records)


def test_dry_run_refuses_a_checkpoint_model(tmp_path: Path):
    out = tmp_path / "report.json"
    code = eval_agent.main(
        [
            "--tasks",
            str(SAMPLE),
            "--dry-run",
            "--model",
            str(tmp_path / "ckpt"),
            "--json-out",
            str(out),
        ]
    )
    assert code == 2
    assert not out.exists()


def test_a_missing_task_file_is_a_clean_error(tmp_path: Path, capsys):
    code = eval_agent.main(["--tasks", str(tmp_path / "absent.jsonl")])
    assert code == 2
    assert "eval_agent:" in capsys.readouterr().err
