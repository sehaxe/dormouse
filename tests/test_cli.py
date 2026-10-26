"""The CLI: info honesty, strictness surfacing, and one smoke round-trip.

`info small` re-asserts the pinned preset total through the user's own
command; `--set` refusals assert that the CLI never owns a config rule; the
train/eval/decode round-trip proves the four commands form one loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
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


def test_a_bool_set_is_read_as_a_bool_not_as_a_truthy_string():
    """`--set flag=False` must turn the flag OFF.

    TOML spells booleans lowercase, so the capitalised literal used to survive
    as the string "False" — truthy, i.e. the opposite of what was typed. This
    hit every bool flag in the CLI; the field's own type is the fix.
    """
    from dormouse.cli import _split_sets

    model, _ = _split_sets(["share_u=False", "use_msa=false", "use_attnres=TRUE"])
    assert model == {"share_u": False, "use_msa": False, "use_attnres": True}

    # a literal that is not a boolean is refused, not silently truthy
    import pytest

    from dormouse.instrument import LoudError

    with pytest.raises(LoudError, match="boolean field"):
        _split_sets(["share_u=maybe"])


def test_an_int_set_is_read_as_an_int():
    from dormouse.cli import _split_sets

    _, train = _split_sets(["steps=1000", "log_every=50"])
    assert train == {"steps": 1000, "log_every": 50}


def test_unported_mechanism_refuses_through_the_cli():
    r = runner.invoke(app, ["info", "small", "--set", 'quant="lbit"'])  # a training-time rung
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


# --- the engram_ram loader gates: one rule, every CLI loader ----------------
#
# A host-table run stores its Engram OUTSIDE the model (`Engram.to_host` deletes
# the `engram.tables` parameter) in the sibling `<ckpt>.ngram`. `eval`, `decode`
# and `serve` called `model.load_state_dict(...)` directly, so ANY such
# checkpoint died with `Missing key(s): "loop.engram.tables"` — the loaders
# could not read a single b1/medium/sft checkpoint. `train.loop.load_model_state`
# is now the ONE definition; these gates pin it from the CLI.


def _host_table_ckpt(tmp_path: Path):
    """A tiny `engram_ram` checkpoint pair + its `.ngram` sibling (CPU-only)."""
    from conftest import tiny_run

    from dormouse.model import DormouseModel
    from dormouse.model.engram import enable_host_tables
    from dormouse.train.loop import save_ckpt
    from dormouse.train.optim import MixOptimizer

    run = tiny_run(tmp_path, steps=1, engram_ram=True)
    model = DormouseModel(run.model)
    assert enable_host_tables(model) == 1, "what train()'s host_tables does"
    ids = torch.arange(2 * 64, dtype=torch.long).view(2, 64) % 251
    model(ids, (ids + 1) % 251)[1].backward()
    opt = MixOptimizer(model, lr=run.train.lr, wd=run.train.wd, opt=run.train.opt)
    save_ckpt(run, model, opt, 0)
    return run


def test_eval_loads_an_engram_ram_checkpoint(tmp_path: Path):
    """(a) The CLI eval loader round-trips a host-table checkpoint."""
    run = _host_table_ckpt(tmp_path)
    r = runner.invoke(
        app,
        [
            "eval",
            "--out-dir",
            run.train.out_dir,
            "--ckpt-name",
            run.train.ckpt_name,
            "--eval",
            run.train.data,
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "BPB" in r.output


def test_eval_refuses_an_engram_ram_checkpoint_without_its_sidecar(tmp_path: Path):
    """(b) A host-table checkpoint with no `.ngram` is a LOUD refusal."""
    run = _host_table_ckpt(tmp_path)
    ngram = Path(run.train.out_dir) / f"{run.train.ckpt_name}.ngram"
    assert ngram.is_file()
    ngram.unlink()
    r = runner.invoke(
        app,
        [
            "eval",
            "--out-dir",
            run.train.out_dir,
            "--ckpt-name",
            run.train.ckpt_name,
            "--eval",
            run.train.data,
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code != 0, r.output
    msg = r.output + str(r.exception)
    assert ".ngram" in msg and "sibling" in msg, msg


def test_in_vram_checkpoint_still_loads_unchanged(ckpt_dir: Path, tmp_path: Path):
    """(c) The old path is untouched: `engram_ram=false` stays a parameter."""
    from dormouse.cli import _model, _snapshot_run
    from dormouse.train.loop import load_model_half, load_model_state

    run = _snapshot_run(tmp_path / "ck", "smoke")
    assert not run.train.engram_ram
    model = _model(run)
    sd = load_model_half(tmp_path / "ck" / "smoke.pt")
    load_model_state(model, run.train, sd)
    assert model.loop.engram.host is None, "an in-VRAM run must not switch to host"
    assert "loop.engram.tables" in model.state_dict()
    assert torch.equal(model.loop.engram.tables, sd["loop.engram.tables"])
