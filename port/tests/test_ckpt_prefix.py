"""The checkpoint format under torch.compile: `_orig_mod.` never reaches the file.

The 2026-10-17 audit's CRITICAL 2: with `torch_compile=true` the trainer wraps
the model in an OptimizedModule whose `state_dict()` keys carry a `_orig_mod.`
prefix; `save_ckpt` stored them verbatim while every reader (load_ckpt, eval,
decode, serve) loads into the UNWRAPPED model — the b1 run's checkpoints were
unresumable and un-evaluable. These gates pin both directions: the save strips,
and the load tolerates an old prefixed file.
"""

from __future__ import annotations

import torch
from conftest import tiny_run
from torch import nn

from dormouse.model import DormouseModel
from dormouse.train.loop import load_ckpt, save_ckpt
from dormouse.train.optim import MixOptimizer


class _Prefixed(nn.Module):
    """A stand-in for torch.compile's OptimizedModule: prefixed state_dict keys.

    Cheaper and more deterministic than a real `torch.compile` in a unit test
    (which would need a full inductor warmup), and it reproduces the exact
    property under test.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self._orig_mod = inner

    def state_dict(self, *a, **k):  # type: ignore[override]
        return {f"_orig_mod.{key}": v for key, v in self._orig_mod.state_dict(*a, **k).items()}


def test_save_strips_the_compile_prefix(tmp_path):
    run = tiny_run(tmp_path)
    model = DormouseModel(run.model)
    opt = MixOptimizer(model, lr=0.01, wd=0.0, opt="mix")
    weights, _snap = save_ckpt(run, _Prefixed(model), opt, step=0)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    keys = list(ck["model"])
    assert keys and not any(k.startswith("_orig_mod.") for k in keys), keys[:4]
    assert set(keys) == set(model.state_dict()), "the file must be the unwrapped model"


def test_load_tolerates_an_old_prefixed_file(tmp_path):
    run = tiny_run(tmp_path)
    model = DormouseModel(run.model)
    opt = MixOptimizer(model, lr=0.01, wd=0.0, opt="mix")
    weights, _snap = save_ckpt(run, model, opt, step=0)
    # Rewrite the file the way a PRE-FIX compile run wrote it.
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    ck["model"] = {f"_orig_mod.{k}": v for k, v in ck["model"].items()}
    torch.save(ck, weights)
    back, _stored = load_ckpt(run)
    keys = list(back["model"])
    assert keys and not any(k.startswith("_orig_mod.") for k in keys)
    model.load_state_dict(back["model"])  # the round trip must actually load
