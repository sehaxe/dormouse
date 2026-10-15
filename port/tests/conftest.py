"""Shared fixtures: a tiny model, a tiny corpus, a resolved run.

Everything here is CPU and tiny on purpose — the gate that matters on the port
is the SEMANTICS (which arm ran, what the config refuses, what the window
scored), and none of those need a GPU or a 1B model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dormouse.config import ModelConfig, RunConfig, TrainConfig
from dormouse.model import DormouseModel

#: d=64, E=4, rank=8 — the shape the port's task names for the model tests.
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

SEQ, BATCH = 64, 2


def tiny_cfg(**over) -> ModelConfig:
    cfg = dict(TINY)
    cfg.update(over)
    return ModelConfig(**cfg)


def tiny_model(**over) -> DormouseModel:
    return DormouseModel(tiny_cfg(**over))


def write_corpus(path: Path, n_bytes: int = 4096, cycle: int = 16) -> Path:
    """A fixed synthetic byte sequence with a learnable short cycle."""
    path.write_bytes(bytes(range(cycle)) * (n_bytes // cycle + 1))
    return path


def tiny_run(tmp_path: Path, **train_over) -> RunConfig:
    cfg = TrainConfig(
        steps=30,
        batch=BATCH,
        seq_len=SEQ,
        lr=0.01,
        log_every=10,
        ckpt_every=100,
        ckpt_name="t",
        out_dir=str(tmp_path / "ckpts"),
        data=str(write_corpus(tmp_path / "corpus.bin")),
    )
    return RunConfig(source="tiny", model=tiny_cfg(), train=_with(cfg, train_over))


def _with(cfg: TrainConfig, over: dict) -> TrainConfig:
    import dataclasses

    return dataclasses.replace(cfg, **over) if over else cfg


@pytest.fixture
def tiny() -> ModelConfig:
    return tiny_cfg()


@pytest.fixture
def model() -> DormouseModel:
    return tiny_model()


@pytest.fixture
def run(tmp_path: Path) -> RunConfig:
    return tiny_run(tmp_path)


@pytest.fixture
def byte_tensor():
    import torch

    def make(batch: int = BATCH, seq: int = SEQ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.arange(batch * seq, dtype=torch.long).view(batch, seq) % 251
        return ids, (ids + 1) % 251

    return make
