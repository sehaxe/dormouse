"""The measured table for the two fast decode arms, on a REAL checkpoint.

    cd port && PYTHONPATH=src python scripts/bench_decode_fast.py \
        --out-dir checkpoints/flagship --ckpt-name flagship1

Four modes, same prompt, `--runs 3` (median), `--bytes 32` each:

  naive                    `decode_bytes` (window re-forward)
  recurrent                `recurrent_decode` (KDA state carried)
  recurrent+graph          + `decode_graph`
  recurrent+graph+multibyte(4)  + `decode_multibyte=4`

The multibyte arm is GREEDY (temp <= 1e-6) by contract, so every row runs at
`--temp 1e-6` — comparing a sampled row to an argmax row would measure the
sampler, not the arm. The table is honest about what did not run: each row
prints its counter line and `why`, so a `decode_multibyte_fallback=1` row says
"this checkpoint has no MTP head" instead of hiding behind an identical time.

The optional `--random-mtp-k K` row attaches an UNTRAINED `MTPHead` to the real
model and measures the multibyte machinery on it. The bytes must still be the
one-byte loop's (the draft is only a proposal), but an untrained draft is
rejected every round, so that row is the COST of the mechanism, not its win —
label it as such. A trained head is what the machinery needs to pay off, and no
checkpoint on this box has one (`mtp_weight = 0` everywhere).
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from dormouse.cli import _model, _snapshot_run
from dormouse.instrument import Counters
from dormouse.model.decode import decode_bytes, recurrent_decode
from dormouse.train.loop import load_model_half, load_model_state


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _line(c: Counters | None) -> str:
    if c is None:
        return "(decode_bytes takes no counters)"
    line = c.line()
    why = getattr(c, "why", None)
    return f"{line} · why: {why}" if why else line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="checkpoints/flagship")
    ap.add_argument("--ckpt-name", default="flagship1")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--bytes", type=int, default=32)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temp", type=float, default=1e-6)
    ap.add_argument("--random-mtp-k", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.temp > 1e-6:
        raise SystemExit("bench at temp <= 1e-6: the multibyte row is greedy by contract")

    dev = torch.device(args.device)
    run = _snapshot_run(Path(args.out_dir), args.ckpt_name)
    model = _model(run).to(dev)
    load_model_state(model, run.train, load_model_half(Path(args.out_dir) / f"{args.ckpt_name}.pt"))
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    prompt = args.prompt.encode()
    card = torch.cuda.get_device_name(0) if dev.type == "cuda" else str(dev)
    print(
        f"{card} · torch {torch.__version__} · {args.ckpt_name} · {params:,} params · "
        f"max_seq {model.cfg.max_seq_len} · iters {model.cfg.max_iter} · "
        f"experts {model.cfg.n_experts}/topk {model.cfg.moe_topk} · "
        f"mtp_k {model.cfg.mtp_k} · {args.bytes} bytes · median of {args.runs}"
    )

    def modes():
        yield "naive", lambda c: decode_bytes(model, prompt, args.bytes, temp=args.temp, device=dev)
        yield (
            "recurrent",
            lambda c: recurrent_decode(
                model, prompt, args.bytes, temp=args.temp, device=dev, counters=c
            ),
        )
        yield (
            "recurrent+graph",
            lambda c: recurrent_decode(
                model,
                prompt,
                args.bytes,
                temp=args.temp,
                device=dev,
                counters=c,
                decode_graph=True,
            ),
        )
        yield (
            "recurrent+graph+multibyte(4)",
            lambda c: recurrent_decode(
                model,
                prompt,
                args.bytes,
                temp=args.temp,
                device=dev,
                counters=c,
                decode_graph=True,
                decode_multibyte=4,
            ),
        )
        if args.random_mtp_k:
            from dormouse.model.mtp import MTPHead

            # The shipped checkpoint is a CAPACITY-DROPPING MoE (topk 2, cap 2.0),
            # and the multibyte verify is refused there because a block's drop set
            # is not prefix-causal. The SAME WEIGHTS run dropless ARE
            # prefix-causal, so that is the honest way to measure the machinery on
            # a real model shape: a modified control, labelled as one.
            model.loop.moe.capacity = 0.0
            model.mtp = MTPHead(model.cfg.d_model, args.random_mtp_k, device=dev)
            yield (
                "recurrent [moe_capacity=0]",
                lambda c: recurrent_decode(
                    model, prompt, args.bytes, temp=args.temp, device=dev, counters=c
                ),
            )
            yield (
                "recurrent+graph [moe_capacity=0]",
                lambda c: recurrent_decode(
                    model,
                    prompt,
                    args.bytes,
                    temp=args.temp,
                    device=dev,
                    counters=c,
                    decode_graph=True,
                ),
            )
            yield (
                "recurrent+graph+multibyte(4) [moe_capacity=0, RANDOM draft]",
                lambda c: recurrent_decode(
                    model,
                    prompt,
                    args.bytes,
                    temp=args.temp,
                    device=dev,
                    counters=c,
                    decode_graph=True,
                    decode_multibyte=4,
                ),
            )

    rows = []
    ref = drop_ref = None
    for label, call in modes():
        # One throwaway call warms the card, the kernels and (for the graph) the
        # capture; the timed runs then measure the steady state.
        call(Counters())
        _sync(dev)
        times, counters, out = [], Counters(), None
        for _ in range(args.runs):
            _sync(dev)
            t0 = time.perf_counter()
            out = call(counters)
            _sync(dev)
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        n_new = len(out) - len(prompt)
        if ref is None:
            ref = out
        # The dropless random-draft pair is its own control (the capacity change
        # makes it a different forward); every other row is compared to naive.
        if "moe_capacity=0" in label:
            if drop_ref is None:
                drop_ref = out
            control = drop_ref
        else:
            control = ref
        same = out == control
        rows.append((label, med, n_new, same, counters))
        print(
            f"{label:<58} {med:7.2f} s  {med / n_new * 1000:7.2f} ms/byte  "
            f"{n_new / med:7.2f} B/s  same={same}"
        )

    print("\ncounter lines (the timed runs summed):")
    for label, _med, _n, _same, c in rows:
        print(f"  {label}: {_line(c)}")


if __name__ == "__main__":
    main()
