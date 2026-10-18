"""Shared fixtures: a tiny model, a tiny corpus, a resolved run.

Everything here is CPU and tiny on purpose — the gate that matters on the port
is the SEMANTICS (which arm ran, what the config refuses, what the window
scored), and none of those need a GPU or a 1B model.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dormouse.config import ModelConfig, RunConfig, TrainConfig
from dormouse.model import DormouseModel
from dormouse.train.optim import JORDAN_NS_COEFFS

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


# --- Newton–Schulz references shared by the NS gates and benches -------------
#
# These lived in `test_muon_gram.py` until the Gram-NS arm was deleted (it
# measured 1.2-1.4x SLOWER than `classic` on this GPU). The classic arm's fp64
# transcription and the fixtures are still what `test_ns_triton.py` and
# `scripts/bench_ns_triton.py` measure the fused chain against, so they moved
# here rather than dying with the file.


def gauss(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    """A deterministic input — a Gaussian, like a momentum buffer of real grads."""
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def ramp(shape: tuple[int, ...]) -> torch.Tensor:
    """A deterministic near-rank-2 matrix — the worst conditioning we can build."""
    n = 1
    for s in shape:
        n *= s
    return torch.arange(n, dtype=torch.float32).reshape(shape) / 4096.0 - 8.0


def rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out - ref).norm() / (ref.norm() + 1e-30))


def fp64_classic(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """`zeropower_via_ns`'s classic body, transcribed at double precision."""
    a, b, c = JORDAN_NS_COEFFS
    x = g.to(torch.float64)
    x = x / (x.norm(dim=(-1, -2), keepdim=True) + 1e-7)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-1, -2).contiguous()
    for _ in range(steps):
        gram = x @ x.transpose(-1, -2)
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.transpose(-1, -2).contiguous() if transposed else x


def gpu_free() -> tuple[bool, str]:
    """Machine discipline (AGENTS §1.5): one GPU thing at a time — a bench precondition."""
    import subprocess

    try:
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()
        busy = subprocess.run(
            ["pgrep", "-af", "dormouse trai[n]|[/]target/release/train "],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001 - a bench's precondition, not a run's
        return False, f"cannot check the card ({e})"
    mib = int(used.split()[0]) if used else -1
    if busy:
        return False, f"a train process is live (pids {busy.splitlines()[0].split()[0]})"
    if mib >= 1024:
        return False, f"the card holds {mib} MiB"
    return True, f"card {mib} MiB, no train process"


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
