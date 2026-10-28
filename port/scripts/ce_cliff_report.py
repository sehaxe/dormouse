#!/usr/bin/env -S uv run --script
"""How often does the Softmax Collapse cliff actually fire? (arXiv:2610.31642 §2)

Loads a checkpoint + held-out bytes, runs the model's own forward, and reports
the per-token logit margin `z_t - LSE_{j != t} z_j` (the paper's `m`) plus the
gradient-sum violation the standard `log_softmax → NLL` path leaves behind.
This is the number that says whether `train.ce_fixed=true` is a decision that
ever bites: the cliff is 16.6 nats in fp32 (5.5 in bf16), and a token above it
gets a target gradient of exactly 0 from the standard path.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python port/scripts/ce_cliff_report.py
    ... --ckpt port/checkpoints/port.pt --data data_local/mix4/web_eval.bin

The gradient is the mean CE's own backward, so ONE pass gives the per-token
`dL/dlogits` rows — no per-token loop. `--tokens` bounds the run (CPU).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dormouse.cli import _model, _snapshot_run
from dormouse.instrument import Counters, loud
from dormouse.losses import ce_margin, cliff, fixed_cross_entropy
from dormouse.train.loop import load_model_half, load_model_state

CLIFFS = (5.5, 16.6)


def _stats(x: torch.Tensor) -> str:
    x = x.detach().float()
    q = torch.quantile(x, torch.tensor([0.5, 0.99, 0.999], dtype=x.dtype))
    return (
        f"mean={float(x.mean()):+8.3f} median={float(q[0]):+8.3f} "
        f"p99={float(q[1]):+8.3f} p99.9={float(q[2]):+8.3f} max={float(x.max()):+9.3f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/port.pt"))
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("/home/sehaxe/dormouse/data_local/mix4/web_eval.bin"),
    )
    ap.add_argument("--tokens", type=int, default=8 * 512)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    if not a.ckpt.is_file():
        loud(f"no checkpoint at {a.ckpt}. Escape: --ckpt <path>.pt")
    if not a.data.is_file():
        loud(f"no eval bytes at {a.data}. Escape: --data <path>_eval.bin")
    run = _snapshot_run(a.ckpt.parent, a.ckpt.stem)
    seq = run.train.seq_len
    batch = a.batch
    batches = max(1, a.tokens // (batch * seq))

    raw = np.memmap(a.data, dtype=np.uint8, mode="r")
    if len(raw) < batches * batch * seq + 1:
        loud(f"{a.data} holds {len(raw)} B, need {batches * batch * seq + 1}. Escape: --tokens")
    data = torch.from_numpy(np.array(raw[: batches * batch * seq + 1], dtype=np.uint8)).long()

    model = _model(run)
    load_model_state(model, run.train, load_model_half(a.ckpt))
    model.eval()
    counters = Counters()

    margins, std_viol, fix_viol, ces = [], [], [], []
    std_collapsed = fix_collapsed = 0
    n_tok = 0
    for i in range(batches):
        lo = i * batch * seq
        x = data[lo : lo + batch * seq].view(batch, seq)
        y = data[lo + 1 : lo + batch * seq + 1].view(batch, seq)
        logits, _loss, _c = model(x, y, counters=counters)
        flat = y.reshape(-1)
        margins.append(ce_margin(logits.detach(), flat))
        ces.append(-F.log_softmax(logits.detach(), dim=-1).gather(1, flat.reshape(-1, 1)))
        # The STANDARD path's own backward, one row of dL/dlogits per token.
        zs = logits.detach().clone().requires_grad_(True)
        (-F.log_softmax(zs, dim=-1).gather(1, flat.reshape(-1, 1)).mean()).backward()
        g = zs.grad
        row_max = g.abs().max(dim=-1).values
        std_viol.append((g.sum(dim=-1).abs() / row_max.clamp_min(1e-30)).detach())
        std_collapsed += int(((g.gather(1, flat.reshape(-1, 1)) == 0) & (row_max > 0)).sum())
        # The fix, for contrast: the same rows, the same quantity.
        zf = logits.detach().clone().requires_grad_(True)
        fixed_cross_entropy(zf, flat).backward()
        gf = zf.grad
        fix_viol.append(gf.sum(dim=-1).abs() / gf.abs().max(dim=-1).values.clamp_min(1e-30))
        fix_collapsed += int(
            ((gf.gather(1, flat.reshape(-1, 1)) == 0) & (gf.abs().max(-1).values > 0)).sum()
        )
        n_tok += flat.numel()

    m = torch.cat(margins).float()
    sv = torch.cat(std_viol).float()
    fv = torch.cat(fix_viol).float()
    print(f"checkpoint {a.ckpt}  ({a.ckpt.stat().st_size / 2**20:.0f} MiB, bf16={run.model.bf16})")
    print(f"data {a.data}  {n_tok} tokens ({batches} x {batch} x {seq})")
    print("logits dtype seen by the loss: float32 (the head widens on every path)")
    ce = torch.cat(ces).float()
    print(
        f"held-out CE from the same logits, nats: {float(ce.mean()):.4f} "
        f"(bpb {float(ce.mean()) / 0.6931471805599453:.3f}); "
        f"it must equal mean softplus(-margin) = {float(F.softplus(-m).mean()):.4f}"
    )
    print(f"margin (z_t - LSE_{{j!=t}} z_j), nats:\n  {_stats(m)}")
    for c in CLIFFS:
        frac = float((m > c).sum()) / n_tok
        print(f"  P(margin > {c:4.1f}) = {frac:.6f}  ({int((m > c).sum())} tokens)")
    print(f"  the loss's own cliff (fp32 logits) = {cliff(torch.float32):.1f} nats")
    print(f"STANDARD path, |sum g| / max|g| per token:\n  {_stats(sv)}")
    print(f"  fraction > 1e-2 = {float((sv > 1e-2).sum()) / n_tok:.6f}")
    print(
        f"  target entry exactly 0 with non-zero others = {std_collapsed} "
        f"({std_collapsed / n_tok:.6f})"
    )
    print(f"FIXED path, |sum g| / max|g| per token:\n  {_stats(fv)}")
    print(f"  target entry exactly 0 with non-zero others = {fix_collapsed}")
    counters.drain()
    print(f"model counters: {counters.line()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
