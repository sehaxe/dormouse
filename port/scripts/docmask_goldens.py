#!/usr/bin/env -S uv run --script
"""Capture the `doc_masking=false` parity goldens — run on a CLEAN `main` checkout.

The document-masking arm is OFF by default, and "off" must be `main`, byte for
byte: the same state_dict, the same logits, the same batch tensors, the same
losses. This script writes the JSON the gate (`tests/test_docmask.py`) pins.
It touches no `doc_masking` key on purpose, so it runs identically on `main`
before the arm exists and on the branch after it.

    cd port && uv run python scripts/docmask_goldens.py  # -> tests/fixtures/docmask_golden.json
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import torch

from dormouse.config import ModelConfig, RunConfig, TrainConfig
from dormouse.model import DormouseModel
from dormouse.train import train
from dormouse.train.data import ByteStream

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
SEQ, BATCH, STEPS = 16, 2, 3
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "docmask_golden.json"


def _hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _state_hash(model: DormouseModel) -> str:
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        corpus = root / "corpus.bin"
        corpus.write_bytes(bytes(range(16)) * 256)
        run = RunConfig(
            source="docmask-golden",
            model=ModelConfig(**TINY),
            train=TrainConfig(
                steps=STEPS,
                batch=BATCH,
                seq_len=SEQ,
                lr=0.01,
                log_every=1,
                ckpt_every=10**9,
                ckpt_name="g",
                out_dir=str(root),
                data=str(corpus),
            ),
        )
        torch.manual_seed(0)
        model = DormouseModel(run.model)
        state = _state_hash(model)

        stream = ByteStream(corpus, BATCH, SEQ)
        batches = [stream.next_batch() for _ in range(3)]
        torch.manual_seed(7)
        ids = (torch.arange(2 * 16, dtype=torch.long) % 251).reshape(2, 16)
        model.eval()
        with torch.no_grad():
            logits, loss, _ = model(ids, ids + 1)
        assert loss is not None
        model.train()

        losses: list[str] = []
        train(model, run, stream=stream, log=lambda line: _capture(line, losses))
        out = {
            "fixture": {"corpus": "bytes(range(16))*256", "batch": BATCH, "seq": SEQ},
            "init_state": state,
            "logits": _hash(logits),
            "loss": repr(float(loss)),
            "batches": [[_hash(x), _hash(y)] for x, y in batches],
            "train_losses": losses,
            "train_state": _state_hash(model),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    return 0


def _capture(line: str, losses: list[str]) -> None:
    m = re.search(r"loss=([0-9.eE+-]+)", line)
    if m and "step=" in line:
        losses.append(m.group(1))


if __name__ == "__main__":
    sys.exit(main())
