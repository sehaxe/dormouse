"""Where the VRAM goes, and how big a batch fits. Read-only probe, no checkpoints.

    python scripts/probe_vram.py --accounting         # CPU-side: what is fixed cost
    python scripts/probe_vram.py --seq 1024 --batches 8,16,24
    python scripts/probe_vram.py --levers             # + bf16 optimizer state

The batch curve is the useful half: it answers "we sit at 7 GiB of 16, how far can
`batch` go before the allocator refuses", with the torch allocator's own numbers
(`max_memory_allocated`, `max_memory_reserved`, `mem_get_info`) rather than an
estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from dormouse.cli import _coerce, _model  # noqa: E402
from dormouse.config import TrainConfig, load_run  # noqa: E402
from dormouse.train.data import open_streams  # noqa: E402
from dormouse.train.loop import train  # noqa: E402

OUT = "/tmp/opencode/vramprobe"
CKPT = "vramprobe"


def overrides(**kv: object) -> dict:
    d = TrainConfig()
    return {k: _coerce(k, v, getattr(d, k)) for k, v in kv.items()}


def one(batch: int, seq: int, steps: int, levers: bool) -> dict:
    base = overrides(
        batch=batch,
        seq_len=seq,
        max_seq_len=seq,
        grad_accum=1,
        steps=steps,
        eval=False,
        ckpt_every=10**9,
        log_every=1,
        warmup_tokens=0,
    )
    if levers:
        base.update(overrides(muon_state_dtype="bf16", ns_dtype="bf16"))
    run = load_run("medium", train_overrides=base, model_overrides=overrides(max_seq_len=seq))
    model = _model(run)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        stream, _ = open_streams(run)
        train(model=model, run=run, stream=stream, device="cuda", log=lambda *a, **k: None)
        peak = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        free, total = torch.cuda.mem_get_info()
        return {
            "ok": True,
            "batch": batch,
            "seq": seq,
            "levers": levers,
            "peak_gib": round(peak, 2),
            "reserved_gib": round(reserved, 2),
            "free_gib": round(free / 2**30, 2),
            "total_gib": round(total / 2**30, 2),
        }
    except torch.cuda.OutOfMemoryError as exc:
        free, total = torch.cuda.mem_get_info()
        return {
            "ok": False,
            "batch": batch,
            "seq": seq,
            "levers": levers,
            "error": str(exc).splitlines()[0][:110],
            "free_gib": round(free / 2**30, 2),
            "total_gib": round(total / 2**30, 2),
        }
    finally:
        del model
        torch.cuda.empty_cache()


def accounting() -> None:
    run = load_run("medium")
    model = _model(run)
    total = sum(p.numel() for p in model.parameters())
    muon = sum(p.numel() for n, p in model.named_parameters() if p.ndim == 2 and "engram" not in n)
    adam = total - muon
    g = 2**30
    print(f"params            {total / 1e6:8.1f} M")
    print(f"  bf16 weights    {total * 2 / g:8.3f} GiB")
    print(f"  bf16 grads      {total * 2 / g:8.3f} GiB")
    print(f"  muon fp32 mom   {muon * 4 / g:8.3f} GiB   ({muon / 1e6:.1f} M in the muon group)")
    print(f"  adamw fp32 m+v  {adam * 8 / g:8.3f} GiB   ({adam / 1e6:.1f} M in adamw)")
    print(f"  -> bf16 muon state frees {(muon * 2) / g:.3f} GiB")
    print()
    for b in (8, 16, 24, 32):
        t = b * 1024
        moe = t * 2730 * 2 / 2**20
        print(
            f"  batch {b:2d}: one [b*t, f] MoE buffer {moe:6.0f} MiB; "
            f"saved activations scale linearly with batch"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batches", default="8,16,24")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--levers", action="store_true")
    ap.add_argument("--accounting", action="store_true")
    args = ap.parse_args(argv)

    if args.accounting:
        accounting()
        return 0
    if not torch.cuda.is_available():
        print("no CUDA device; the CPU-side half follows")
        accounting()
        return 1

    rows = []
    for levers in [False, True] if args.levers else [False]:
        for b in (int(x) for x in args.batches.split(",")):
            row = one(b, args.seq, args.steps, levers)
            rows.append(row)
            print(json.dumps(row), flush=True)
    good = [r for r in rows if r["ok"]]
    if good:
        best = max(good, key=lambda r: r["batch"])
        print(
            f"\nlargest fitting here: batch {best['batch']} seq {best['seq']} "
            f"levers={best['levers']} peak {best['peak_gib']} GiB of {best['total_gib']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
