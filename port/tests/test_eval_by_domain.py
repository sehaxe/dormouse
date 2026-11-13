"""The domain table and its held-out gate — CPU only, no checkpoint, no forward.

`eval_by_domain.py` prints one markdown row per source of a mix plan and refuses
a window that is a training source. Both are pure enough to pin here: the table
is the artifact a reader quotes, and the refusal is the difference between a
held-out number and a number the model already read.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from carve_eval_tails import SEPARATOR, carve  # noqa: E402
from conftest import tiny_run  # noqa: E402
from eval_by_domain import (  # noqa: E402
    DomainScore,
    checkpoint_step,
    eval_path_for,
    render_table,
    trained_bytes,
    window_cfg,
    windows_for,
)

from dormouse.instrument import LoudError  # noqa: E402
from dormouse.train.data import eval_batches_for  # noqa: E402
from dormouse.train.mixture import load_mix_plan  # noqa: E402


def _plan(tmp_path: Path, sources: dict[str, Path]) -> Path:
    """A legal one-stage plan: every source weighted evenly, last takes the rest."""
    share = 1.0 / len(sources)
    weights = dict.fromkeys(sources, share)
    weights[list(sources)[-1]] = 1.0 - share * (len(sources) - 1)
    lines = ["[[stage]]", "until_frac = 1.0", "[stage.weights]"]
    lines += [f"{name} = {w!r}" for name, w in weights.items()]
    lines += ["[sources]"] + [f'{name} = "{path}"' for name, path in sources.items()]
    plan = tmp_path / "mix.toml"
    plan.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan


def test_eval_path_drops_the_train_suffix(tmp_path: Path) -> None:
    assert eval_path_for(tmp_path / "fwedu_train.bin").name == "fwedu_eval.bin"
    assert eval_path_for(tmp_path / "code.bin").name == "code_eval.bin"


def test_table_is_the_five_columns() -> None:
    rows = [
        DomainScore("code", 32_768, 1.125, 1.25, 28_836_000),
        DomainScore("math", 32_768, 2.5, 2.25, None),
    ]
    assert render_table(rows) == (
        "| domain | bytes scored | bpb_fin | bpb_mean | bytes trained |\n"
        "|---|---:|---:|---:|---:|\n"
        "| code | 32,768 | 1.1250 | 1.2500 | 28,836,000 |\n"
        "| math | 32,768 | 2.5000 | 2.2500 | n/a |"
    )


def test_a_window_that_is_also_a_training_source_is_refused(tmp_path: Path) -> None:
    train = tmp_path / "code_train.bin"
    train.write_bytes(b"\x01" * 4096)
    window = tmp_path / "code_eval.bin"
    window.write_bytes(b"\x02" * 4096)
    # the window is declared as a SOURCE too: the path gate must catch it
    plan = load_mix_plan(_plan(tmp_path, {"code": train, "code_eval": window}))
    run = tiny_run(tmp_path, mix_plan=str(_plan(tmp_path, {"code": train, "code_eval": window})))
    with pytest.raises(LoudError, match="not held out"):
        windows_for(plan, run, "code", allow_overlap=False)


def test_a_distinct_file_passes_and_a_missing_one_names_the_carve(tmp_path: Path) -> None:
    train = tmp_path / "code_train.bin"
    train.write_bytes(b"\x01" * 4096)
    plan = load_mix_plan(_plan(tmp_path, {"code": train}))
    run = tiny_run(tmp_path, mix_plan=str(_plan(tmp_path, {"code": train})))
    with pytest.raises(LoudError, match="no held-out window"):
        windows_for(plan, run, "", allow_overlap=False)
    window = tmp_path / "code_eval.bin"
    window.write_bytes(b"\x02" * 4096)
    assert windows_for(plan, run, "", allow_overlap=False) == [("code", window)]


def test_the_default_window_keeps_the_snapshot_geometry(tmp_path: Path) -> None:
    train = tmp_path / "code_train.bin"
    train.write_bytes(b"\x00" * 4096)
    cfg = replace(tiny_run(tmp_path).train, batch=2, seq_len=64, eval_batches=1)
    kept = window_cfg(cfg, tmp_path / "code_eval.bin", None, 0)
    assert kept.eval_batches == 1  # the trainer's recorded window, 2 x 64 = 128 B
    assert eval_batches_for(kept) == 1
    given = window_cfg(cfg, tmp_path / "code_eval.bin", 196_608, 0)
    assert given.eval_batches is None
    assert given.eval_bytes == 196_608


def test_trained_bytes_replays_optimizer_steps(tmp_path: Path) -> None:
    src = tmp_path / "code.bin"
    src.write_bytes(b"\x00" * 4096)
    plan = load_mix_plan(_plan(tmp_path, {"code": src}))
    cfg = replace(tiny_run(tmp_path).train, steps=10, grad_accum=2, seed=7)
    assert trained_bytes(plan, cfg, done_steps=3) == {
        "code": 3 * 2 * cfg.batch * (cfg.seq_len + 1)
    }


def test_checkpoint_step_is_the_saved_record(tmp_path: Path) -> None:
    path = tmp_path / "tiny.pt"
    torch.save({"step": 41, "model": {}}, path)
    assert checkpoint_step(path) == 41


def test_carve_splits_at_a_document_separator(tmp_path: Path) -> None:
    content = b"aaa\n\n---\n\nbbb\n\n---\n\nccc\n"
    src = tmp_path / "code.bin"
    src.write_bytes(content)
    out, off, n, digest, rule = carve(src, tail=12, search=64, sep=SEPARATOR)
    assert rule == "separator"
    # the LAST separator at/before size-tail (12) is the one at 3, not the one at 13
    assert off == 3
    assert src.read_bytes() == content[:off]
    assert out == tmp_path / "code_eval.bin"
    assert out.read_bytes() == content[off:]
    assert n == len(content) - off
    assert digest == hashlib.sha256(content[off:]).hexdigest()


def test_carve_without_a_separator_is_a_fixed_line_offset(tmp_path: Path) -> None:
    content = b"aa\nbb\ncc\ndd\n"
    src = tmp_path / "ru.bin"
    src.write_bytes(content)
    out, off, _n, _digest, rule = carve(src, tail=5, search=64, sep=None)
    assert rule == "line-offset"
    assert off == 9
    assert src.read_bytes() == b"aa\nbb\ncc\n"
    assert out.read_bytes() == b"dd\n"
