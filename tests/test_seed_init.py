"""C1 gate: `train.seed` owns the weights — in one process and across processes.

The 2026-10-19 audit measured two processes with `cfg_seed = 1` building
different initializations (`39cb6df0…` vs `c70cb36b…`): `DormouseModel.__init__`
draws from the global torch RNG and no trainer seeded it, so the seed in every
snapshot and `.dormouse` header did not describe the weights. `cli._model` now
calls `torch.manual_seed(run.train.seed)` immediately before construction; the
gates below pin that, including the resume half (a load must replace the init,
not inherit it).
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import torch

from dormouse.cli import _model
from dormouse.config import ModelConfig, RunConfig, TrainConfig
from dormouse.train.loop import load_ckpt, load_model_state, save_ckpt
from dormouse.train.optim import MixOptimizer

#: The suite's tiny shape (conftest.TINY), repeated here because the
#: subprocess script below is standalone.
TINY = dict(
    d_model=64,
    n_heads=4,
    head_dim=16,
    d_ffn=128,
    vocab=256,
    max_seq_len=64,
    max_iter=2,
    rank=8,
    n_experts=4,
    moe_topk=2,
    engram_rows=64,
    engram_dim=8,
    bf16=False,
    shared_rank=16,
)


def _run(seed: int, out_dir: Path | None = None) -> RunConfig:
    train = TrainConfig(
        steps=1,
        batch=2,
        seq_len=64,
        lr=3e-4,
        ckpt_name="s",
        out_dir=str(out_dir if out_dir is not None else Path(".")),
        seed=seed,
    )
    return RunConfig(source="seed-gate", model=ModelConfig(**TINY), train=train)


def _param_hash(model: torch.nn.Module) -> str:
    """SHA-256 over every name/shape/value in the state dict, first 16 hex."""
    h = hashlib.sha256()
    for name, p in model.state_dict().items():
        h.update(name.encode())
        h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().cpu().contiguous().to(torch.float64).numpy().tobytes())
    return h.hexdigest()[:16]


_SUBPROCESS = """
import hashlib, sys
import torch
from dormouse.cli import _model
from dormouse.config import ModelConfig, RunConfig, TrainConfig

def _hash(model):
    h = hashlib.sha256()
    for name, p in model.state_dict().items():
        h.update(name.encode())
        h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().cpu().contiguous().to(torch.float64).numpy().tobytes())
    return h.hexdigest()[:16]

cfg = ModelConfig(
    d_model=64, n_heads=4, head_dim=16, d_ffn=128, vocab=256, max_seq_len=64,
    max_iter=2, rank=8, n_experts=4, moe_topk=2, engram_rows=64, engram_dim=8,
    bf16=False, shared_rank=16,
)
run = RunConfig(source="seed-gate", model=cfg, train=TrainConfig(seed=int(sys.argv[1])))
print(_hash(_model(run)))
"""


def _subprocess_hash(seed: int) -> str:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS, str(seed)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_the_same_seed_builds_the_same_weights_in_one_process():
    """The builder seeds itself: a polluted global stream cannot move the init."""
    torch.manual_seed(0xBADC0DE)  # whatever the caller's stream held
    a = _param_hash(_model(_run(1)))
    torch.manual_seed(0x5EED)
    b = _param_hash(_model(_run(1)))
    c = _param_hash(_model(_run(2)))
    assert a == b, "two seed=1 builds drew different weights (C1)"
    assert a != c, "seed=2 did not move the weights (the seed is still inert)"


def test_the_same_seed_builds_the_same_weights_across_processes():
    """The audit measured this cross-process; two fresh interpreters must agree."""
    first = _subprocess_hash(1)
    second = _subprocess_hash(1)
    other = _subprocess_hash(2)
    assert first == second, f"two seed=1 processes built {first} vs {second} (C1)"
    assert first != other, "seed=2 built the same initialization as seed=1"


def test_a_resume_load_replaces_the_init_regardless_of_the_seed(tmp_path: Path):
    """The saved weights are the run; the seed only decides what a resume overwrites."""
    run = _run(1, tmp_path)
    saved = _model(run)
    saved_hash = _param_hash(saved)
    opt = MixOptimizer(
        saved,
        lr=run.train.lr,
        wd=run.train.wd,
        opt=run.train.opt,
        ns_steps=run.train.ns_steps,
        ns_kind=run.train.ns_kind,
        colrow_3d=run.train.muon_colrow_3d,
    )
    save_ckpt(run, saved, opt, step=0)

    # A fresh model built under a DIFFERENT seed really is a different init —
    # otherwise the equality below would prove nothing.
    fresh = _model(dataclasses.replace(run, train=dataclasses.replace(run.train, seed=2)))
    assert _param_hash(fresh) != saved_hash

    ck, _stored = load_ckpt(run)
    load_model_state(fresh, run.train, ck["model"])
    assert _param_hash(fresh) == saved_hash, "a resume inherited its init, not the file"
