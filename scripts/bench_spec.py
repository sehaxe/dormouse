"""Speculative decode on the real checkpoint: bytes/s, forwards/byte, accept rate.

    uv run --extra gpu python scripts/bench_spec.py --ckpt-name port

The `k=0` row is plain `decode_bytes` — the same prompt, the same `n` bytes, the
same temperature — so every other row's bytes/s, forwards/byte and speedup are
read against it, and the acceptance rate is the share of drafted bytes the
target agreed with (`SpecStats.acceptance_rate`). That rate is the whole verdict
on the draft head: a head that proposes nothing the target wants makes spec
decode a slower plain decode, and this table says so instead of guessing.

What the columns mean (all of them come from `SpecStats`, none is inferred):
`forwards/byte` counts EVERY target forward, the per-round state forward
included, over the bytes actually emitted; `accept` is `accepted/drafted`; the
counters column is `Counters.line()` and `spec_non_causal` there means the
target's MoE drops assignments, so its forward is not prefix-causal and the
bytes are not guaranteed to be plain decode's (the module docstring of
`model/specdecode.py` has the measurement). `spec_over_window` means the verify
window outgrew `max_seq_len`, which happens once the reply passes the window.

Machine discipline first (AGENTS 1.5): no `dormouse` process and < 2 GiB on the
card. A live training run makes this script SKIP loudly with exit 0 — while
`dormouse train b1` is on the card that is the correct answer, not a failure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

from dormouse.cli import _snapshot_run
from dormouse.instrument import loud
from dormouse.model import DormouseModel
from dormouse.model.decode import decode_bytes
from dormouse.model.specdecode import spec_decode_bytes
from dormouse.train.loop import load_model_half, load_model_state

SPEC_KS = (1, 2, 4, 8)


def preflight() -> str | None:
    """`None` when the card is ours; else the reason to SKIP (never an error)."""
    busy = subprocess.run(
        ["pgrep", "-ax", "dormouse"], capture_output=True, text=True
    ).stdout.strip()
    if busy:
        return f"a dormouse process is alive:\n{busy}"
    if not torch.cuda.is_available():
        return "no CUDA device: this bench measures the card, and a CPU run would measure the host"
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 2**30
    if used >= 2.0:
        return f"the card already holds {used:.2f} GiB"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--ckpt-name", default="port")
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. ")
    ap.add_argument("--n", type=int, default=128, help="bytes sampled per arm")
    ap.add_argument("--temp", type=float, default=0.8)
    args = ap.parse_args()

    reason = preflight()
    if reason is not None:
        print(f"[SKIP] {reason}")
        print(
            "[SKIP] ONE heavy thing at a time (AGENTS 1.5): the bench needs the card "
            "to itself — a contended measurement is a measurement of the contention."
        )
        return 0

    out_dir, ckpt = Path(args.out_dir), args.ckpt_name
    run = _snapshot_run(out_dir, ckpt)
    model = DormouseModel(run.model).to("cuda")
    # The ONE loader: an engram_ram checkpoint is host tables + `.ngram`.
    load_model_state(model, run.train, load_model_half(out_dir / f"{ckpt}.pt"))
    model.eval()
    if model.dspark is None:
        loud(
            f"{ckpt} carries no DSpark draft head (dspark_weight=0.0): spec decode has "
            f"nothing to draft with. Escape: a checkpoint trained with "
            f"--set dspark_weight>0 (the head is built only when the weight is non-zero)."
        )
    prompt = args.prompt.encode("utf-8")
    params = sum(p.numel() for p in model.parameters())
    print(
        f"[shape] {ckpt} · {params:,} params · prompt {len(prompt)} B · n {args.n} B · "
        f"temp {args.temp} · max_seq_len {run.model.max_seq_len} · k in {SPEC_KS}",
        flush=True,
    )

    def timed(k: int) -> tuple[bytes, object | None, float]:
        gen = torch.Generator(device="cuda").manual_seed(run.train.seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            if k == 0:
                out, stats = (
                    decode_bytes(
                        model, prompt, args.n, temp=args.temp, device="cuda", generator=gen
                    ),
                    None,
                )
            else:
                out, stats = spec_decode_bytes(
                    model,
                    prompt,
                    args.n,
                    spec_k=k,
                    temp=args.temp,
                    device="cuda",
                    generator=gen,
                )
        torch.cuda.synchronize()
        return out, stats, time.perf_counter() - t0

    # One warm pass pays the autotuner/compile before anything is timed (the
    # plain arm runs first, so an unwarmed card would be charged to IT).
    with torch.no_grad():
        decode_bytes(model, prompt, 8, temp=args.temp, device="cuda")
    print("[warm] one 8-byte plain decode done (autotune/compile paid)", flush=True)

    rows: list[tuple[int, int, float, float, float, float, str]] = []
    for k in (0, *SPEC_KS):
        out, stats, dt = timed(k)
        emitted = len(out) - len(prompt)
        rows.append(
            (
                k,
                emitted,
                dt,
                emitted / max(dt, 1e-9),
                float("nan") if stats is None else stats.per_byte_forwards,
                float("nan") if stats is None else stats.acceptance_rate,
                "plain" if stats is None else stats.counters.line(),
            )
        )
        k_, emitted_, dt_, rate_, fwd_, acc_, counters_ = rows[-1]
        print(
            f"[k={k_}] {emitted_} B in {dt_:.1f}s · {rate_:.0f} B/s · "
            f"{'1.00' if k_ == 0 else f'{fwd_:.2f}'} forwards/byte · "
            f"{'n/a' if k_ == 0 else f'{acc_:.1%}'} accept · {counters_}",
            flush=True,
        )

    base = rows[0][3]
    print("\n| spec_k | bytes | seconds | bytes/s | forwards/byte | accept | speedup |")
    print("|---|---|---|---|---|---|---|")
    for k, emitted, dt, rate, fwd, acc, _ in rows:
        speed = "—" if k == 0 else f"{rate / base:.2f}x"
        print(
            f"| {k} | {emitted} | {dt:.1f} | {rate:.0f} | "
            f"{'1.00' if k == 0 else f'{fwd:.2f}'} | "
            f"{'—' if k == 0 else f'{acc:.1%}'} | {speed} |"
        )
    print(
        "\nRead it as: speedup = bytes/s(k) / bytes/s(plain), all at temp "
        f"{args.temp}. A k with accept near 0 is a slower plain decode — the draft "
        "head is not earning its keep on this checkpoint."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
