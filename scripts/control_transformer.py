#!/usr/bin/env python3
"""THE control: a plain decoder-only transformer on the flagship's own bytes.

Why this exists: the flagship's curve (BPB 1.7446 after 1.62 GB on mix_v7) may be
the architecture or the shared pipeline (data/mixture/trainer/BPB accounting).
This script is deliberately boring -- byte vocab, pre-norm blocks, RoPE, GELU
FFN, AdamW + cosine, no expert bank, no recurrence, no memory, no state -- and
it reads the SAME byte stream through the SAME `MixtureStream` the trainer uses,
scores the SAME eval file through the SAME `ByteStream` window, and reports
`CE / ln 2` exactly as `dormouse.train.loop.bpb` does. If this is also ~1.7 BPB
at 4.2 GB, the architecture is not the suspect. If it lands near the reference,
the gap is the architecture's.

Compute match is measured, not guessed: `--ours-flops-per-byte` defaults to
137.42 MFLOP/byte, the value torch's FlopCounterMode reads off the flagship2
checkpoint (fwd 45.8 + bwd 91.6, batch 1, seq 1024). This model's own bill is
printed at startup (same instrument) and warns if it drifts past +-10%.

Usage (from the repo root, with port/.venv/bin/python):
    scripts/control_transformer.py --bytes 50000000  --out /tmp/c50.json
    scripts/control_transformer.py --bytes 200000000 --out /tmp/c200.json
    scripts/control_transformer.py --bytes 500000000 --out /tmp/c500.json

The stage schedule is the LIVE RUN's: the mixture is built with the flagship's
513000 micro-batch budget (171000 optimizer steps x grad_accum 3), so step k of
this stream is the byte-for-byte window step k of the flagship's stream -- not
merely the same distribution. A run of N bytes consumes that prefix.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "port" / "src"))

from dormouse.train.data import ByteStream  # noqa: E402
from dormouse.train.mixture import open_mixture, source_for_step  # noqa: E402

LN2 = math.log(2.0)

#: The flagship's own bill, measured with FlopCounterMode on
#: checkpoints/flagship2/flagship2.pt (batch 1, seq 1024, fwd+bwd).
OURS_FLOPS_PER_BYTE = 137.42e6

#: The flagship's mixture budget in MICRO-BATCHES: 171000 opt steps x grad_accum 3.
#: Passing it (not this run's own length) is what keeps the stage/source draw
#: identical to the run we are explaining.
FLAGSHIP_MICRO_STEPS = 171_000 * 3


# --- the model: nothing exotic ---------------------------------------------


class Block(nn.Module):
    """Pre-norm, RoPE self-attention, 4x GELU FFN. No biases, no tricks."""

    def __init__(self, d: int, heads: int, ffn: int, dropout: float = 0.0):
        super().__init__()
        assert d % heads == 0
        self.heads, self.hd = heads, d // heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, ffn, bias=False)
        self.fc2 = nn.Linear(ffn, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(d, dim=-1)
        q = q.view(b, t, self.heads, self.hd).transpose(1, 2)
        k = k.view(b, t, self.heads, self.hd).transpose(1, 2)
        v = v.view(b, t, self.heads, self.hd).transpose(1, 2)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(b, t, d)
        x = x + self.drop(self.proj(y))
        h = self.ln2(x)
        return x + self.drop(self.fc2(F.gelu(self.fc(h))))


def rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def rope_tables(t: int, hd: int, device, dtype):
    half = hd // 2
    inv = 1.0 / (10_000 ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    ang = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv)
    return ang.cos().to(dtype)[None, None], ang.sin().to(dtype)[None, None]


class Control(nn.Module):
    def __init__(self, vocab=256, d=352, layers=12, heads=8, ffn=1408, seq=1024):
        super().__init__()
        self.seq = seq
        self.hd = d // heads
        self.tok = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads, ffn) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)
        # GPT-2 residual scaling: keep the residual stream from growing with depth.
        for blk in self.blocks:
            nn.init.normal_(blk.proj.weight, std=0.02 / math.sqrt(2 * layers))
            nn.init.normal_(blk.fc2.weight, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, y, cos, sin):
        h = self.tok(x)
        for blk in self.blocks:
            h = blk(h, cos, sin)
        logits = self.head(self.ln_f(h))
        return logits, F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


# --- the instrument: the trainer's own eval, byte for byte ------------------


@torch.no_grad()
def eval_bpb(model, path, batch, seq, batches, device):
    """`dormouse.train.loop._eval_loss`, reduced to what this model needs.

    Same reader, same window (batch x seq_len x batches BYTES), same rewind-first
    rule, mean over batches of the mean-over-positions CE, BPB = CE / ln 2.
    """
    stream = ByteStream(path, batch, seq)
    stream.rewind()
    total = 0.0
    was = model.training
    model.eval()
    cos, sin = rope_tables(seq, model.hd, device, torch.float32)
    for _ in range(batches):
        x, y = stream.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss = model(x, y, cos, sin)
        total += float(loss)
    model.train(was)
    stream.close()
    ce = total / max(batches, 1)
    return ce, ce / LN2


def flops_per_byte(model, seq, device):
    """Forward+backward FLOPs per predicted byte, FlopCounterMode (the flagship's instrument)."""
    from torch.utils.flop_counter import FlopCounterMode

    was = model.training
    model.eval()
    x = torch.randint(0, 256, (1, seq), device=device)
    y = torch.randint(0, 256, (1, seq), device=device)
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as fc:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss = model(x, y, *rope_tables(seq, model.hd, device, torch.float32))
        loss.backward()
    total = fc.get_total_flops() / seq
    model.zero_grad(set_to_none=True)
    model.train(was)
    return total


# --- train ------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bytes", type=int, default=50_000_000, help="predicted bytes to train on")
    ap.add_argument("--steps", type=int, default=0, help="override: optimizer steps")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches per optimizer step (the flagship uses 3)")
    ap.add_argument("--micro-budget", type=int, default=FLAGSHIP_MICRO_STEPS,
                    help="the LIVE run's micro-batch budget (decides the stage schedule)")
    ap.add_argument("--mix-plan", type=Path, default=REPO / "port/configs/mix_v7.toml")
    ap.add_argument("--eval-data", type=Path, default=REPO / "data_local/mix4/web_eval.bin")
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--eval-at-bytes", default="", help="comma list of extra eval points")
    ap.add_argument("--d-model", type=int, default=304)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=1216)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=Path("control_result.json"))
    ap.add_argument("--save", type=Path, default=None,
                    help="write the final weights (for a multi-window eval)")
    ap.add_argument("--tag", default="control")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device != "cpu":
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        # Hard ceiling: the live flagship job owns this card. 2.5 GB is ~3x this
        # control's real footprint, and a runaway allocation dies HERE, not in
        # the 100M run it would otherwise evict.
        torch.cuda.set_per_process_memory_fraction(2.5 / 16.0)
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")

    bytes_per_step = args.batch * args.seq_len * args.grad_accum
    steps = args.steps or max(1, args.bytes // bytes_per_step)
    bytes_total = steps * bytes_per_step

    model = Control(d=args.d_model, layers=args.layers, heads=args.heads,
                    ffn=args.ffn, seq=args.seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    flops = flops_per_byte(model, args.seq_len, device)
    drift = flops / OURS_FLOPS_PER_BYTE - 1.0
    line = (
        f"[{args.tag}] d={args.d_model} L={args.layers} H={args.heads} f={args.ffn} "
        f"params={n_params:,} flops/byte={flops/1e6:.1f} M ({drift:+.1%} vs flagship "
        f"{OURS_FLOPS_PER_BYTE/1e6:.1f} M) steps={steps} bytes={bytes_total:,} dev={device}"
    )
    print(line, flush=True)
    if abs(drift) > 0.10:
        print(f"[{args.tag}] WARNING: compute match is outside +-10% ({drift:+.1%})", flush=True)

    # The SAME stream: same plan, same seed, same flagship micro-budget.
    stream = open_mixture(
        args.mix_plan,
        steps=args.micro_budget,
        batch=args.batch,
        seq_len=args.seq_len,
        seed=args.seed,
    )

    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )
    warmup = max(20, int(args.warmup_frac * steps))
    eval_at = sorted({int(v) for v in args.eval_at_bytes.split(",") if v.strip()})
    pending = [b for b in eval_at if b < bytes_total]

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        t = (step - warmup) / max(1, steps - warmup)
        return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))

    cos, sin = rope_tables(args.seq_len, args.d_model // args.heads, device, torch.float32)
    rows, src_counts = [], {}
    t0 = time.time()
    tokens_seen = 0
    for step in range(steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        micro_loss = 0.0
        for _micro in range(args.grad_accum):
            src = source_for_step(stream.plan, stream._step, stream.steps, stream.seed)
            src_counts[src] = src_counts.get(src, 0) + 1
            x, y = stream.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, loss = model(x, y, cos, sin)
            (loss / args.grad_accum).backward()
            micro_loss += loss.detach().item() / args.grad_accum
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        tokens_seen += bytes_per_step

        if step % args.log_every == 0 or step == steps - 1:
            rate = tokens_seen / max(1e-9, time.time() - t0)
            print(f"[{args.tag}] step={step+1}/{steps} bytes={tokens_seen:,} "
                  f"loss={micro_loss:.4f} lr={lr:.2e} {rate/1000:.0f} kB/s", flush=True)
        while pending and tokens_seen >= pending[0]:
            ce, bp = eval_bpb(model, args.eval_data, args.batch, args.seq_len,
                              args.eval_batches, device)
            rows.append({"tag": args.tag, "bytes": tokens_seen, "ce": ce, "bpb": bp,
                         "step": step + 1, "budget": bytes_total})
            print(f"[{args.tag}] EVAL bytes={tokens_seen:,} ce={ce:.4f} bpb={bp:.4f}", flush=True)
            pending.pop(0)

    wall = time.time() - t0
    ce, bp = eval_bpb(model, args.eval_data, args.batch, args.seq_len, args.eval_batches, device)
    rows.append({"tag": args.tag, "bytes": tokens_seen, "ce": ce, "bpb": bp,
                 "step": steps, "budget": bytes_total})
    print(f"[{args.tag}] FINAL bytes={tokens_seen:,} ce={ce:.4f} bpb={bp:.4f} "
          f"wall={wall/60:.1f} min", flush=True)

    out = {
        "tag": args.tag,
        "arch": {"d_model": args.d_model, "layers": args.layers, "heads": args.heads,
                 "head_dim": args.d_model // args.heads, "ffn": args.ffn,
                 "params": n_params, "rope": True, "pre_norm": True, "activation": "gelu",
                 "tied_head": False, "optimizer": "adamw", "betas": [0.9, 0.95],
                 "lr": args.lr, "wd": args.wd, "warmup_steps": warmup,
                 "batch": args.batch, "seq_len": args.seq_len, "grad_accum": args.grad_accum,
                 "schedule": f"cosine to {args.min_lr_frac:g}x", "seed": args.seed},
        "flops_per_byte_fwd_bwd": flops,
        "ours_flops_per_byte": OURS_FLOPS_PER_BYTE,
        "compute_drift": drift,
        "bytes": bytes_total,
        "bytes_per_step": bytes_per_step,
        "stream": {"mix_plan": str(args.mix_plan), "micro_budget": args.micro_budget,
                   "batch": args.batch, "seq_len": args.seq_len,
                   "sources": src_counts},
        "eval": {"file": str(args.eval_data), "batches": args.eval_batches,
                 "window_bytes": args.eval_batches * bytes_per_step},
        "wall_s": wall,
        "evals": rows,
        "final": rows[-1],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "arch": out["arch"],
                    "bytes": bytes_total, "flops_per_byte": flops}, args.save)
        print(f"[{args.tag}] saved {args.save}", flush=True)
    print(f"[{args.tag}] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
