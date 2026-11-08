"""A checkpoint reader must not unpickle a foreign file (audit 2026-11-08).

Every file this engine writes is plain tensors/ints/strings, so
`weights_only=True` loads all of them. `weights_only=False` on a
user-supplied `--init-from` / `--ckpt-name` path is therefore an avoidable
code-execution sink: the pickle reaches `os.system` before any shape check.

The tests are the compatibility proof (`save_ckpt`'s output passes the safe
loader) and the refusal proof (a pickled class is rejected, not executed).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch
from conftest import tiny_run
from test_sft import sft_run

from dormouse.model import DormouseModel
from dormouse.train.loop import save_ckpt
from dormouse.train.optim import MixOptimizer
from dormouse.train.sft import sft_train


class _NotAPlainTensor:
    """A class the weights-only unpickler does not allow-list."""


def _pickled_class(path: Path) -> Path:
    torch.save({"model": _NotAPlainTensor()}, path)
    return path


def test_a_saved_checkpoint_loads_under_weights_only(tmp_path: Path):
    """The reason the safe loader is not a compatibility break: the engine's
    own file passes it, optimizer section included."""
    run = tiny_run(tmp_path, steps=1)
    model = DormouseModel(run.model)
    opt = MixOptimizer(model, lr=1e-3, wd=0.0)
    weights, _ = save_ckpt(run, model, opt, step=0)
    ck = torch.load(weights, map_location="cpu", weights_only=True)
    assert "model" in ck and "opt" in ck


def test_sft_init_from_refuses_a_pickled_class(tmp_path: Path):
    """`--init-from` is the foreign-file path: a class-name pickle is refused
    by the unpickler, never executed."""
    run = sft_run(tmp_path, steps=1)
    evil = _pickled_class(tmp_path / "evil.pt")
    with pytest.raises(pickle.UnpicklingError):
        sft_train(DormouseModel(run.model), run, init_from=str(evil), log=lambda *_: None)
