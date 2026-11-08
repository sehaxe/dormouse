"""The doctor: one command that says whether a 9-hour run can start.

Every failure driven here happened in this repo: a data source that was not on
disk, an eval window that sat inside the corpus it was supposed to be held out
from, and a checkpoint snapshot the loader rightly refuses. The test runs the
REAL command on a tiny CPU config + tmp dirs, so a doctor that crashes, exits 1
on a clean run, or calls a warn a fail is caught.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from conftest import write_corpus
from typer.testing import CliRunner

from dormouse.cli import app

runner = CliRunner()

#: A tiny preset + one batch is 2 x 65 B; 4 KiB per file leaves room.
CORPUS_BYTES = 4096


def _git(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.email=doctor@test",
            "-c",
            "user.name=doctor",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "base",
        ],
        check=True,
    )


def _files(tmp_path: Path) -> dict[str, Path]:
    """The run's named files, a legal one-stage plan, and a clean git checkout."""
    corpus = write_corpus(tmp_path / "corpus.bin", n_bytes=CORPUS_BYTES)
    eval_file = write_corpus(tmp_path / "eval.bin", n_bytes=CORPUS_BYTES, cycle=32)
    source = write_corpus(tmp_path / "src.bin", n_bytes=CORPUS_BYTES, cycle=8)
    plan = tmp_path / "mix.toml"
    plan.write_text(
        "[[stage]]\nuntil_frac = 1.0\n[stage.weights]\na = 1.0\n\n"
        f'[sources]\na = "{source.name}"\n',
        encoding="utf-8",
    )
    root = tmp_path / "gitroot"
    root.mkdir()
    _git(root)
    return {"corpus": corpus, "eval": eval_file, "source": source, "plan": plan, "root": root}


def _invoke(
    tmp_path: Path,
    *,
    files: dict[str, Path] | None = None,
    sets: dict[str, object] | None = None,
    root: Path | None = None,
    plan_text: str | None = None,
):
    files = files or _files(tmp_path)
    if plan_text is not None:
        files["plan"].write_text(plan_text, encoding="utf-8")
    over: dict[str, object] = {
        "data": str(files["corpus"]),
        "eval_data": str(files["eval"]),
        "mix_plan": str(files["plan"]),
        "out_dir": str(tmp_path / "ckpts"),
        "batch": 2,
        "seq_len": 64,
        "eval": True,
        # the legacy window: 196 608 B would not fit the 4 KiB fixture
        "eval_batches": 1,
        "torch_compile": False,
    }
    over.update(sets or {})
    args = [
        "doctor",
        "tiny",
        "--device",
        "cpu",
        "--root",
        str(root or files["root"]),
        "--json",
    ]
    for key, value in over.items():
        if value is not None:
            args += ["--set", f"{key}={value}"]
    return runner.invoke(app, args), files


def _rows(result) -> dict[str, dict]:
    return {c["name"]: c for c in json.loads(result.output)["checks"]}


def test_a_clean_tiny_run_passes(tmp_path):
    result, _ = _invoke(tmp_path)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["counts"]["fail"] == 0
    rows = {c["name"]: c for c in payload["checks"]}
    assert {
        "config",
        "torch",
        "gpu",
        "vram",
        "mix",
        "data",
        "mix:a",
        "eval",
        "overlap",
        "snapshot",
        "disk",
        "cache",
        "git",
    } <= set(rows)
    assert all(row["status"] == "pass" for row in rows.values()), rows


def test_the_json_shape_is_the_ci_contract(tmp_path):
    result, _ = _invoke(tmp_path)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload) == {"ok", "preset", "counts", "git", "checks"}
    assert payload["preset"] == "tiny"
    assert set(payload["counts"]) == {"pass", "warn", "fail"}
    assert payload["counts"]["pass"] == len(payload["checks"]) - (
        payload["counts"]["warn"] + payload["counts"]["fail"]
    )
    assert isinstance(payload["git"]["sha"], str) and payload["git"]["sha"]
    assert isinstance(payload["git"]["dirty"], bool)
    for check in payload["checks"]:
        assert set(check) == {"name", "status", "detail"}
        assert check["status"] in {"pass", "warn", "fail"}
        assert check["name"] and check["detail"]


def test_a_missing_data_source_fails(tmp_path):
    result, _ = _invoke(tmp_path, sets={"data": str(tmp_path / "gone.bin")})
    assert result.exit_code == 1, result.output
    row = _rows(result)["data"]
    assert row["status"] == "fail" and "does not exist" in row["detail"]


def test_an_eval_window_inside_the_corpus_fails(tmp_path):
    files = _files(tmp_path)
    result, _ = _invoke(tmp_path, files=files, sets={"eval_data": str(files["corpus"])})
    assert result.exit_code == 1, result.output
    row = _rows(result)["overlap"]
    assert row["status"] == "fail" and "not held out" in row["detail"]


def test_a_relative_plan_source_is_compared_beside_the_plan(tmp_path):
    files = _files(tmp_path)
    # the plan names `src.bin` relative to itself; eval IS that file
    result, _ = _invoke(tmp_path, files=files, sets={"eval_data": str(files["source"])})
    assert result.exit_code == 1, result.output
    row = _rows(result)["overlap"]
    assert row["status"] == "fail" and "not held out" in row["detail"]


def test_a_mix_source_that_does_not_exist_fails(tmp_path):
    plan = (
        "[[stage]]\nuntil_frac = 1.0\n[stage.weights]\ngone = 1.0\n\n"
        '[sources]\ngone = "never-written.bin"\n'
    )
    result, _ = _invoke(tmp_path, plan_text=plan)
    assert result.exit_code == 1, result.output
    rows = _rows(result)
    assert rows["mix"]["status"] == "pass"  # the plan parses…
    assert rows["mix:gone"]["status"] == "fail"  # …and the file it names is missing
    assert "does not exist" in rows["mix:gone"]["detail"]


def test_a_malformed_mix_plan_is_a_row_not_a_traceback(tmp_path):
    result, _ = _invoke(tmp_path, plan_text="this is not TOML")
    assert result.exit_code == 1, result.output
    row = _rows(result)["mix"]
    assert row["status"] == "fail" and "not TOML" in row["detail"]


def test_a_preset_refusal_is_reported_loudly(tmp_path):
    result, _ = _invoke(tmp_path, sets={"opt": "bogus"})
    assert result.exit_code == 1, result.output
    row = _rows(result)["config"]
    assert row["status"] == "fail" and "load_run refused" in row["detail"]
    assert "opt=" in row["detail"]  # the config's own escape, not a traceback


def test_a_snapshot_with_an_unknown_key_fails(tmp_path):
    files = _files(tmp_path)
    snap_dir = tmp_path / "ckpts"
    snap_dir.mkdir()
    # `port` is TrainConfig's default ckpt_name.
    (snap_dir / "port.config.toml").write_text(
        "[model]\nno_such_key = 1\n\n[train]\n", encoding="utf-8"
    )
    result, _ = _invoke(tmp_path, files=files)
    assert result.exit_code == 1, result.output
    row = _rows(result)["snapshot"]
    assert row["status"] == "fail" and "unknown key" in row["detail"]


def test_a_dirty_tree_warns_but_does_not_fail(tmp_path):
    files = _files(tmp_path)
    (files["root"] / "scratch.txt").write_text("x", encoding="utf-8")
    result, _ = _invoke(tmp_path, files=files)
    assert result.exit_code == 0, result.output
    row = _rows(result)["git"]
    assert row["status"] == "warn" and "dirty" in row["detail"]


def test_a_non_git_root_fails(tmp_path):
    not_git = tmp_path / "not-a-checkout"
    not_git.mkdir()
    result, _ = _invoke(tmp_path, root=not_git)
    assert result.exit_code == 1, result.output
    row = _rows(result)["git"]
    assert row["status"] == "fail" and "git checkout" in row["detail"]
