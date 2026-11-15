"""Where the VRAM goes, and how big a batch fits. Read-only probe, no checkpoints.

    python scripts/probe_vram.py --accounting         # CPU-side: what is fixed cost
    python scripts/probe_vram.py --seq 1024 --batches 8,16,24
    python scripts/probe_vram.py --levers             # + bf16 optimizer state
    python scripts/probe_vram.py --preset configs/dormouse_1b.toml \
        --set data_item_bytes=2 --bytes-per-token 3.9 --batches 2,4,8

The batch curve is the useful half: it answers "we sit at 7 GiB of 16, how far can
`batch` go before the allocator refuses", with the torch allocator's own numbers
(`max_memory_allocated`, `max_memory_reserved`, `mem_get_info`) rather than an
estimate. `ms_per_step` comes from the log lines' own arrival times (the first
step is excluded: it carries the CUDA/kernel warm-up), and `kb_s` is the TEXT
throughput the run would sustain -- `tokens/s x bytes_per_token / 1000`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from dormouse.cli import _coerce, _model  # noqa: E402
from dormouse.config import ModelConfig, TrainConfig, load_preset, load_run  # noqa: E402
from dormouse.train.data import open_streams  # noqa: E402
from dormouse.train.loop import train  # noqa: E402

OUT = "/tmp/opencode/vramprobe"
CKPT = "vramprobe"


def overrides(**kv: object) -> dict:
    d = TrainConfig()
    return {k: _coerce(k, v, getattr(d, k)) for k, v in kv.items()}


def one(batch: int, seq: int, steps: int, levers: bool, args) -> dict:
    base = overrides(
        batch=batch,
        seq_len=seq,
        grad_accum=1,
        steps=steps,
        eval=False,
        ckpt_every=10**9,
        log_every=1,
        warmup_tokens=0,
    )
    base.update(args.over)
    if levers:
        base.update(overrides(muon_state_dtype="bf16", ns_dtype="bf16"))
    # `max_seq_len` is a MODEL field: a window longer than the preset's is refused
    # by the model, and coercing it as a train key is what broke this probe.
    model_over = {"max_seq_len": seq} if seq > load_preset(args.preset).max_seq_len else {}
    model_over |= args.model_over
    run = load_run(args.preset, train_overrides=base, model_overrides=model_over)
    model = _model(run).to("cuda")  # a cpu model refuses `engram_ram` by design
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    stamps: list[float] = []

    def log(_line: str, *_a: object, **_k: object) -> None:
        stamps.append(time.monotonic())

    t0 = time.monotonic()
    try:
        stream, _ = open_streams(run)
        train(model=model, run=run, stream=stream, device="cuda", log=log)
        wall = time.monotonic() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        free, total = torch.cuda.mem_get_info()
        # Between consecutive STEP lines: the build/eval chatter is not a step.
        gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
        ms = 1000 * sum(gaps) / len(gaps) if gaps else 1000 * wall / steps
        tps = batch * seq / (ms / 1000) if ms else 0.0
        return {
            "ok": True,
            "batch": batch,
            "seq": seq,
            "levers": levers,
            "peak_gib": round(peak, 2),
            "reserved_gib": round(reserved, 2),
            "free_gib": round(free / 2**30, 2),
            "total_gib": round(total / 2**30, 2),
            "ms_per_step": round(ms, 1),
            "tok_s": round(tps),
            "kb_s": round(tps * args.bytes_per_token / 1000),
        }
    except torch.cuda.OutOfMemoryError as exc:
        free, total = torch.cuda.mem_get_info()
        return {
            "ok": False,
            "batch": batch,
            "seq": seq,
            "levers": levers,
            "error": str(exc).splitlines()[0][:110],
            "want_mib": _want_mib(str(exc)),
            "free_gib": round(free / 2**30, 2),
            "total_gib": round(total / 2**30, 2),
        }
    finally:
        del model
        torch.cuda.empty_cache()


def _want_mib(msg: str) -> float | None:
    """`Tried to allocate X MiB` -- how far past the card the step went."""
    m = re.search(r"Tried to allocate ([0-9.]+) (MiB|GiB)", msg)
    if not m:
        return None
    n = float(m.group(1))
    return n * 1024 if m.group(2) == "GiB" else n


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
    ap.add_argument("--preset", default="medium", help="preset NAME or a .toml PATH")
    ap.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a train key, coerced exactly as the CLI does (repeatable)",
    )
    ap.add_argument(
        "--bytes-per-token",
        type=float,
        default=1.0,
        help="TEXT bytes per stream item for the kB/s column (1.0 = a byte corpus)",
    )
    ap.add_argument(
        "--model-set",
        dest="model_sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a MODEL key (e.g. moe_offload=true); coerced against ModelConfig",
    )
    args = ap.parse_args(argv)
    args.over = {}
    for s in args.sets:
        k, _, v = s.partition("=")
        args.over[k] = _coerce(k, v, getattr(TrainConfig(), k))
    args.model_over = {}
    for s in args.model_sets:
        k, _, v = s.partition("=")
        args.model_over[k] = _coerce(k, v, getattr(ModelConfig(), k))

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
            row = one(b, args.seq, args.steps, levers, args)
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
