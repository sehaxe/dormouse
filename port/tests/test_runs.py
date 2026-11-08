"""The run registry: the table order, the running→done move, the crash path.

CPU only and tiny: the registry is JSON plus two calls from `train()`, so the
gate is its semantics, not a training run. The end-to-end test reads the record
INSIDE the first step to prove `running` was on disk before the loop returned.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
from conftest import tiny_model, tiny_run, write_corpus
from typer.testing import CliRunner

from dormouse import runs
from dormouse.cli import app
from dormouse.train import loop as loopmod
from dormouse.train import open_streams, train

runner = CliRunner()


def _fake_run(
    root: Path,
    run_id: str,
    timestamp: str,
    *,
    status: str = "done",
    bpb: float | None = 1.5,
    params: int = 1000,
    train_bytes: int = 2048,
    duration: float = 12.5,
) -> None:
    path = root / run_id
    path.mkdir(parents=True)
    (path / "run.json").write_text(
        json.dumps(
            {
                "schema": runs.SCHEMA,
                "run_id": run_id,
                "status": status,
                "timestamp": timestamp,
                "duration_s": duration,
                "params": params,
                "train_bytes": train_bytes,
                "final_bpb": bpb,
            }
        )
    )


def test_list_is_newest_first_and_show_prints_the_record(tmp_path: Path):
    root = tmp_path / "registry"
    _fake_run(root, "a-old", "2026-11-01T09:00:00Z", bpb=1.75)
    _fake_run(root, "b-new", "2026-11-02T10:30:00Z", status="failed", bpb=None)

    listed = runner.invoke(app, ["runs", "list", "--runs-dir", str(root)])
    assert listed.exit_code == 0, listed.output
    assert listed.output.index("b-new") < listed.output.index("a-old")
    assert "failed" in listed.output and "done" in listed.output
    assert "1.750" in listed.output  # the old run's held-out BPB

    shown = runner.invoke(app, ["runs", "show", "b-new", "--runs-dir", str(root)])
    assert shown.exit_code == 0, shown.output
    record = json.loads(shown.output)
    assert record["run_id"] == "b-new" and record["status"] == "failed"


def test_train_writes_running_then_done_with_the_recipe_and_the_eval(tmp_path: Path):
    corpus = write_corpus(tmp_path / "corpus.bin", n_bytes=4096)
    eval_file = write_corpus(tmp_path / "eval.bin", n_bytes=2048, cycle=8)
    preset = tmp_path / "tiny.toml"
    preset.write_text("d_model = 64\n")

    run = tiny_run(
        tmp_path,
        steps=5,
        log_every=5,
        eval=True,
        eval_every=5,
        eval_batches=1,
        ckpt_every=100,
    )
    run = dataclasses.replace(
        run,
        source=str(preset),
        train=dataclasses.replace(run.train, eval_data=str(eval_file)),
    )
    model = tiny_model()
    root = tmp_path / "registry"
    added = sum(int(p.numel()) for p in model.parameters())

    class WatchingStream:
        """Forwards to the real stream, but reads the record on the first draw."""

        def __init__(self, inner, registry: Path):
            self._inner = inner
            self._registry = registry
            self.status_at_first_step: str | None = None

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def next_batch(self):
            if self.status_at_first_step is None:
                self.status_at_first_step = runs.read_runs(self._registry)[0]["status"]
            return self._inner.next_batch()

    inner, eval_stream = open_streams(run)
    stream = WatchingStream(inner, root)
    report = train(
        model,
        run,
        stream=stream,
        eval_stream=eval_stream,
        log=lambda *_: None,
        runs_dir=root,
    )

    assert stream.status_at_first_step == "running"
    records = runs.read_runs(root)
    assert len(records) == 1
    record = records[0]
    assert record["status"] == "done"
    assert record["duration_s"] is not None and record["duration_s"] >= 0
    assert record["params"] == added
    assert record["train_bytes"] == 5 * 2 * 64
    assert record["config"]["train"]["steps"] == 5
    assert record["config"]["train"]["data"] == str(corpus)
    assert record["preset"]["path"] == str(preset)
    assert record["preset"]["sha256"] == hashlib.sha256(preset.read_bytes()).hexdigest()
    assert record["host"]["gpu"] == "cpu" and record["host"]["torch"]
    roles = {row["role"]: row for row in record["data"]}
    assert roles["train"]["bytes"] == corpus.stat().st_size
    assert roles["eval"]["bytes"] == eval_file.stat().st_size
    assert "eval step=5" in record["eval_line"]
    assert record["final_bpb"] == pytest.approx(report["final_bpb"])
    assert (root / record["run_id"] / "run.json").is_file()


def test_a_crash_writes_failed_with_the_error_string(tmp_path: Path, monkeypatch):
    run = tiny_run(tmp_path, steps=2)
    root = tmp_path / "registry"

    def explode(*_args, **_kwargs):
        raise RuntimeError("stream open exploded")

    monkeypatch.setattr(loopmod, "open_streams", explode)
    with pytest.raises(RuntimeError, match="stream open exploded"):
        train(tiny_model(), run, log=lambda *_: None, runs_dir=root)

    records = runs.read_runs(root)
    assert len(records) == 1
    record = records[0]
    assert record["status"] == "failed"
    assert "stream open exploded" in record["error"]
    assert record["finished"] is not None and record["duration_s"] is not None
    assert record["eval_line"] is None and record["final_bpb"] is None


def test_preset_identity_hashes_the_file_and_never_invents_one():
    identity = runs.preset_identity("small")
    assert identity["path"].endswith("small.toml")
    assert len(identity["sha256"]) == 64

    missing = runs.preset_identity("no-such-preset-anywhere")
    assert missing == {"source": "no-such-preset-anywhere", "path": None, "sha256": None}
