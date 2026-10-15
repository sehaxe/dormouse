"""CLI telemetry round-trip: train with --json-log → the rows parse → plot makes a PNG."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import write_corpus
from typer.testing import CliRunner

from dormouse.cli import app

runner = CliRunner()


def _smoke_dir(tmp_path: Path) -> Path:
    """4 steps of the tiny preset, telemetry on, in tmp_path/ck."""
    corpus = write_corpus(tmp_path / "corpus.bin", n_bytes=4096)
    preset = tmp_path / "tiny.toml"
    preset.write_text(
        "\n".join(
            [
                "d_model = 64",
                "n_heads = 4",
                "head_dim = 16",
                "d_ffn = 128",
                "max_iter = 2",
                "rank = 8",
                "shared_rank = 16",
                "moe_topk = 2",
                "engram_orders = [2, 3]",
                "engram_rows = 64",
                "engram_dim = 8",
                "max_seq_len = 32",
            ]
        )
    )
    r = runner.invoke(
        app,
        [
            "train",
            str(preset),
            "--data",
            str(corpus),
            "--json-log",
            str(tmp_path / "run.jsonl"),
            "--set",
            f"out_dir={tmp_path / 'ck'}",
            "--set",
            "ckpt_name=smoke",
            "--set",
            "batch=2",
            "--set",
            "seq_len=32",
            "--set",
            "steps=4",
            "--set",
            "log_every=2",
            "--set",
            "ckpt_every=4",
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
    return tmp_path


def test_json_log_has_run_log_and_eval_parse(tmp_path: Path):
    import json

    _smoke_dir(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "run.jsonl").read_text().splitlines()]
    events = [r["event"] for r in rows]
    assert "run" in events and "log" in events
    head = next(r for r in rows if r["event"] == "run")
    assert head["source"].endswith("tiny.toml") and head["steps"] == 4
    log_rows = [r for r in rows if r["event"] == "log"]
    assert len(log_rows) == 2 and all(r["step"] in (2, 4) for r in log_rows)
    assert all("loss" in r and "counters" in r for r in log_rows)


def test_plot_renders_a_png(tmp_path: Path):
    _smoke_dir(tmp_path)
    r = runner.invoke(app, ["plot", str(tmp_path / "run.jsonl")])
    assert r.exit_code == 0, r.output
    png = tmp_path / "run.png"
    assert png.is_file() and png.stat().st_size > 5_000
    assert "saved" in r.output


def test_plot_on_empty_log_refuses(tmp_path: Path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    r = runner.invoke(app, ["plot", str(empty)])
    assert r.exit_code != 0 and "empty telemetry" in str(r.exception)
