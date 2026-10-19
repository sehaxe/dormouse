"""The patch-rate probe: the chunker's own boundaries over real corpus bytes, CPU.

    uv run python scripts/patch_rate.py                       # 1 MiB of the local corpus
    uv run python scripts/patch_rate.py --bytes 262144 --k 128

What this measures, and from where. `ByteFlowNet.encode_chunks` (`byteflow.py`)
returns `sel [b, k]` — the coding-rate chunker's chosen boundary positions, the
SAME tensor the channel feeds to `decode_chunks`. The probe runs that front
stage only (embedding + E local blocks + the marginal-gain Top-K), never the
global blocks or the decoder, and turns `sel` into patch lengths: boundary `j`
is the START of patch `j`, so the lengths are the consecutive differences plus
the tail `t − sel[-1]` (the last patch runs to the end of the window). One row's
lengths therefore sum to exactly `t`, and with `k` patches over a `t`-byte
window the MEAN is `t/k` by construction — the geometry, not the chunker's
taste. The distribution is the part that is a measurement: how unevenly the K
boundaries fall. That is the number the compute economics turn on (bytes per
patch vs BPE's ~4).

**RANDOM INIT — the loud caveat.** This probe builds the net from the config and
never loads a checkpoint, so every boundary here is an artifact of random
features. A random-init patch rate is a LOWER BOUND at best and never a claim
about a trained model; the only readings that survive the caveat are the
geometry (mean = t/k, exactly k distinct ascending boundaries) and the
distribution's shape as a sanity check on the chunker's wiring. Do not quote the
median or the p90 as "the model's patch rate".

The geometry is the A/B polygon's (`scripts/smoke_byteflow.py`): burn's
`byteflow_9m` widths (schema.rs:751-794) at d_local 96 / d_global 256, E=2,
window 512, K=128 — the same config the channel arm of `scripts/ab_byteflow.py`
runs. The rate mode is the net's own (`byteflow_logdet = false` → Appendix B's
L2 streaming approximation, the forward's default); the exact logdet is
`O(T·d³)` per row and is analysis-only, so the probe refuses it.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from dormouse.config import load_run
from dormouse.model.byteflow import bf_for_model

#: The port's LOCAL corpus (AGENTS §2.6) — no /mnt dependency.
DATA = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"

#: The A/B polygon: `small` at d_model 256 / n_heads 4 (4 x 64) with the smoke's
#: byteflow widths. `bf16=false` and `use_engram=false` are the two loud refusals
#: the arm carries (`config._validate_byteflow`); `use_kda` stays true so the
#: channel (Option B) is what gets armed.
GEOM = dict(
    use_byteflow=True,
    use_engram=False,
    bf16=False,
    d_model=256,
    n_heads=4,
    byteflow_k_tokens=128,
    byteflow_d_local=96,
    byteflow_d_global=256,
    byteflow_heads_local=4,
    byteflow_heads_global=4,
    byteflow_e_layers=2,
    byteflow_w_local=256,
    byteflow_d_ff_local=256,
    byteflow_max_bytes=512,
)


def patch_lengths(sel: torch.Tensor, t: int) -> torch.Tensor:
    """Boundary indices `[b, k]` over a `t`-byte window → patch lengths `[b, k]`.

    Boundary `j` is the START of patch `j`, so the lengths are the consecutive
    differences plus the tail `t − sel[-1]`. The row sums to exactly `t` and the
    patch count is exactly `k` (position 0 is always a boundary, the rest are
    distinct positions `1..t-1` in ascending order — `select_positions`).
    """
    assert sel.ndim == 2, f"sel must be [b, k], got {tuple(sel.shape)}"
    assert sel.shape[1] >= 1, "a window has at least the BOS boundary"
    assert int(sel.min()) >= 0 and int(sel.max()) < t, "a boundary is inside the window"
    assert bool((sel[:, 1:] > sel[:, :-1]).all()), "boundaries must be strictly ascending"
    diffs = sel[:, 1:] - sel[:, :-1]
    tail = (t - sel[:, -1:]).clamp(min=0)
    return torch.cat([diffs, tail], dim=1)


def length_histogram(x: torch.Tensor) -> dict[str, int]:
    """Patch lengths → counts, bucketed `1 · 2 · 3 · 4 · 5-8 · 9-16 · 17-32 · 33-64 · 65+`."""
    buckets = (
        ("1", 1, 1),
        ("2", 2, 2),
        ("3", 3, 3),
        ("4", 4, 4),
        ("5-8", 5, 8),
        ("9-16", 9, 16),
        ("17-32", 17, 32),
        ("33-64", 33, 64),
        ("65+", 65, None),
    )
    out: dict[str, int] = {}
    for label, lo, hi in buckets:
        hit = x >= lo if hi is None else (x >= lo) & (x <= hi)
        out[label] = int(hit.sum())
    assert sum(out.values()) == x.numel(), "the histogram must cover every length"
    return out


def patch_stats(lengths: torch.Tensor) -> dict:
    """Mean / median / p90 / max / histogram of a flat length vector (all f64)."""
    x = torch.as_tensor(lengths, dtype=torch.float64).reshape(-1)
    assert x.numel() > 0, "no patches to describe"
    assert float(x.min()) >= 1.0, "a patch is at least one byte"
    return {
        "n": int(x.numel()),
        "mean": float(x.mean()),
        # `torch.median` returns the LOWER of the two middle values on an even
        # count; quantile 0.5 is numpy's/`statistics.median`'s convention, which
        # is what a reader of a distribution expects.
        "median": float(torch.quantile(x, 0.5)),
        "p90": float(torch.quantile(x, 0.9)),
        "max": float(x.max()),
        "histogram": length_histogram(x),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=DATA, help="the corpus file to sample")
    ap.add_argument("--bytes", type=int, default=1 << 20, help="how many bytes to feed")
    ap.add_argument("--window", type=int, default=512, help="bytes per window (<= max_bytes)")
    ap.add_argument("--k", type=int, default=128, help="patch budget K per window")
    ap.add_argument("--batch", type=int, default=16, help="windows per forward")
    ap.add_argument("--seed", type=int, default=1, help="seeds the net init and the read offset")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    path = Path(args.data)
    if not path.is_file():
        print(f"no corpus at {path}: pass --data")
        return 1
    size = path.stat().st_size
    if args.bytes > size:
        print(f"--bytes {args.bytes} > file size {size}: pass a smaller count")
        return 1

    run = load_run("small", model_overrides=GEOM)
    assert run.model.use_byteflow and run.model.max_seq_len >= args.window
    torch.manual_seed(args.seed)
    channel = bf_for_model(run.model, device=args.device)
    assert channel is not None, "the geometry did not arm the channel (use_kda is off?)"
    net = channel.net
    if net.use_logdet_rate:
        print("byteflow_logdet is on: the exact rate is O(T*d^3) per row (analysis-only).")
        print("Escape: byteflow_logdet = false (Appendix B's L2 streaming approximation).")
        return 1
    assert args.window <= net.max_bytes, f"window {args.window} > max_bytes {net.max_bytes}"
    assert 1 <= args.k <= args.window, f"K={args.k} must be in 1..={args.window}"

    rng = random.Random(args.seed)
    offset = rng.randrange(0, size - args.bytes + 1)
    with open(path, "rb") as fh:
        fh.seek(offset)
        raw = fh.read(args.bytes)
    windows = len(raw) // args.window
    if windows < 1:
        print(f"--bytes {args.bytes} < one window of {args.window} B: feed more bytes")
        return 1
    used = windows * args.window
    x = torch.frombuffer(bytearray(raw[:used]), dtype=torch.uint8).to(torch.long)
    x = x.reshape(windows, args.window)
    print(
        "\n=== RANDOM INIT — the net was never trained. The boundaries below are an\n"
        "=== artifact of random features: a LOWER BOUND at best, never a claim.\n"
        "=== What is not random: with K boundaries over T bytes the MEAN is T/K by\n"
        "=== construction; the distribution is the sanity check on the chunker's wiring.\n",
        flush=True,
    )
    print(
        f"[probe] {path} @ {offset} ({args.bytes} B), window={args.window} K={args.k} "
        f"windows={windows} used={used} B mode=l2 device={args.device} "
        f"params={sum(p.numel() for p in net.parameters()):,}",
        flush=True,
    )

    t0 = time.perf_counter()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, windows, args.batch):
            block = x[i : i + args.batch]
            _h, _z, sel = net.encode_chunks(block, args.k)
            chunks.append(patch_lengths(sel.cpu(), args.window).reshape(-1))
    lengths = torch.cat(chunks)
    dt = time.perf_counter() - t0

    stats = patch_stats(lengths)
    geom_mean = args.window / args.k
    assert abs(stats["mean"] - geom_mean) < 1e-9, (
        f"mean {stats['mean']} != T/K {geom_mean}: the lengths must sum to the window"
    )
    assert stats["n"] == windows * args.k, "one patch per selected boundary"
    print(
        f"\npatch length (bytes per patch) over {stats['n']:,} patches, {dt:.1f} s CPU:\n"
        f"  mean {stats['mean']:.3f} (= T/K = {geom_mean:.3f}, by construction)\n"
        f"  median {stats['median']:.1f}   p90 {stats['p90']:.1f}   max {stats['max']:.0f}\n"
        f"  histogram: "
        + "  ".join(f"{k}={v}" for k, v in stats["histogram"].items())
        + f"\n  BPE reference ~4 bytes/token; the channel's loop pays for K={args.k} "
        f"patches instead of T={args.window} bytes.",
        flush=True,
    )
    print(
        "\n=== random-init caveat again: quote the mean as the geometry and the\n"
        "=== distribution as a wiring check, never as a trained model's patch rate.\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
