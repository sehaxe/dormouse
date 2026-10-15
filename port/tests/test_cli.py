"""The CLI: info honesty, strictness surfacing, and one smoke round-trip.

`info small` re-asserts the pinned preset total through the user's own
command; `--set` refusals assert that the CLI never owns a config rule; the
train/eval/decode round-trip proves the four commands form one loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import write_corpus
from typer.testing import CliRunner

from dormouse.cli import app

runner = CliRunner()


def test_info_prints_the_pinned_count_and_the_budget():
    r = runner.invoke(app, ["info", "small"])
    assert r.exit_code == 0, r.output
    assert "9,476,993" in r.output  # the preset's pinned total, via the CLI
    assert "lm_head" in r.output


def test_info_set_changes_the_number():
    base = runner.invoke(app, ["info", "small"]).output
    over = runner.invoke(app, ["info", "small", "--set", "rank=16"])
    assert over.exit_code == 0
    assert over.output != base and "9,476,993" not in over.output


def test_unknown_set_key_refuses():
    r = runner.invoke(app, ["info", "small", "--set", "no_such_key=1"])
    assert r.exit_code != 0 and "neither a model nor a train key" in str(r.exception)


def test_unported_mechanism_refuses_through_the_cli():
    r = runner.invoke(app, ["info", "small", "--set", 'quant="fp8"'])
    assert r.exit_code != 0 and "Escape" in str(
        r.exception
    )  # the exact message belongs to `quant`'s own rule


@pytest.fixture()
def ckpt_dir(tmp_path: Path) -> Path:
    """A 4-step smoke train: corpus file → checkpoint directory."""
    corpus = write_corpus(tmp_path / "corpus.bin", n_bytes=4096)
    model = "\n".join(
        [
            "d_model = 64",
            "n_heads = 4",
            "head_dim = 16",
            "d_ffn = 128",
            "max_iter = 2",
            "rank = 8",
            "shared_rank = 16",
            "moe_topk = 2",
            "moe_capacity = 1.0",
            "engram_orders = [2, 3]",
            "engram_rows = 64",
            "engram_dim = 8",
            "max_seq_len = 32",
        ]
    )
    preset = tmp_path / "tiny.toml"
    preset.write_text(model)
    r = runner.invoke(
        app,
        [
            "train",
            str(preset),
            "--data",
            str(corpus),
            "--set",
            f"out_dir={tmp_path / 'ck'}",  # a string value through the same path
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
            "--set",
            "eval=false",
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
    return tmp_path / "ck"


def test_train_writes_ckpt_and_snapshot_json_ok(ckpt_dir: Path):
    we, snap = ckpt_dir / "smoke.pt", ckpt_dir / "smoke.config.toml"
    assert we.is_file() and snap.is_file()


def test_eval_runs_the_trained_checkpoint(ckpt_dir: Path, tmp_path: Path):
    r = runner.invoke(
        app,
        [
            "eval",
            "--out-dir",
            str(tmp_path / "ck"),
            "--ckpt-name",
            "smoke",
            "--eval",
            str(tmp_path / "corpus.bin"),
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "BPB" in r.output


def test_decode_samples_from_the_checkpoint(ckpt_dir: Path, tmp_path: Path):
    r = runner.invoke(
        app,
        [
            "decode",
            "--out-dir",
            str(tmp_path / "ck"),
            "--ckpt-name",
            "smoke",
            "--prompt",
            "abcdefgh",
            "--n",
            "16",
            "--temp",
            "0.7",
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
