"""Does the ByteFlow front stage leak future bytes into earlier logits?

The question NCM forces: one forward produces all T byte logits. If a logit at
position i moves when a LATER byte changes, then (a) the reported BPB is
optimistic and (b) parallel emission would be "free" only because the model was
already reading the answer.

Method: fixed batch, fixed seed, run forward; then mutate ONE byte at position
i and diff every logit row j < i. A strictly causal model gives exactly 0.0
there. Any non-zero number is the leak, in nats of logit.

Run: CUDA_VISIBLE_DEVICES="" python scripts/probe_byteflow_leak.py [--dense]
"""

from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, "src")

import dataclasses  # noqa: E402

from dormouse.config import load_preset  # noqa: E402
from dormouse.model.model import DormouseModel  # noqa: E402


def build(dense: bool, seq: int) -> DormouseModel:
    """The shipped `tiny` preset, resized to `seq`, optionally with the channel."""
    cfg = load_preset("tiny")
    over = dict(max_seq_len=seq, use_engram=False)
    if not dense:
        over.update(use_byteflow=True, byteflow_k_tokens=max(4, seq // 4))
    cfg = dataclasses.replace(cfg, **over)
    torch.manual_seed(0)
    return DormouseModel(cfg).eval()


@torch.no_grad()
def logits_of(m: DormouseModel, ids: torch.Tensor) -> torch.Tensor:
    out, _, _ = m(ids)
    return out.reshape(ids.shape[0], ids.shape[1], -1)


def probe(m: DormouseModel, name: str, seq: int, b: int = 1) -> None:
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (b, seq))
    base = logits_of(m, ids)
    print(f"\n== {name}: {sum(p.numel() for p in m.parameters()):,} params, t={seq} ==")
    worst = 0.0
    for i in (1, 2, 3, seq // 4, seq // 2, seq - 1):
        if i >= seq:
            continue
        mut = ids.clone()
        mut[0, i] = (int(ids[0, i]) + 97) % 256
        after = logits_of(m, mut)
        # rows strictly BEFORE i must not move if the model is causal
        d = (after[0, :i] - base[0, :i]).abs().max().item() if i else 0.0
        # the mutated row itself and later rows are expected to move
        d_self = (after[0, i] - base[0, i]).abs().max().item()
        worst = max(worst, d)
        print(
            f"  mutating byte {i:>3}: max|d logit| rows<{i}: {d:.3e}   "
            f"row {i}: {d_self:.3e}"
        )
    print(f"  WORST pre-mutation drift: {worst:.3e}  "
          f"{'LEAK' if worst > 1e-5 else 'causal'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", action="store_true", help="KDA-only control (no patcher)")
    ap.add_argument("--seq", type=int, default=128)
    a = ap.parse_args()
    m = build(a.dense, a.seq)
    probe(m, "dense (KDA, no patcher)" if a.dense else "ByteFlow channel", a.seq)


if __name__ == "__main__":
    main()
