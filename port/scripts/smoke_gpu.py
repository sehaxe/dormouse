"""One real GPU smoke on the `small` preset, with the fused arms engaged.

    uv run --extra gpu python scripts/smoke_gpu.py

20 steps, bf16, batch 8 x seq 512 (4096 tokens/step) on the mounted real corpus,
TWO arms because they prove different things:

* `small` as shipped — it sets `moe_topk = 0` (burn's DENSE blend), so it proves
  the FLA KDA arm and nothing about the dispatch;
* `small` + `moe_topk = 2` — the sparse dispatch, the only path that enters
  `SparseMoE._padded_dispatch`.

Each prints the window counters (`kda=fla,n=…`, `moe_padded_bmm=…`), the held-out
BPB over its own window, ms/step, tok/s and peak allocated VRAM. The numbers go
in the commit message.

Machine discipline first (AGENTS §1.5): no other `train` process alive and the
card holding < 1 GiB before the first kernel.
"""

from __future__ import annotations

import dataclasses
import re
import statistics
import subprocess
import sys
import time

import torch

from dormouse.config import load_run
from dormouse.instrument import loud
from dormouse.model import DormouseModel
from dormouse.train import kda_field, open_streams, train

#: The port's data on this box (AGENTS §2.6). The train file and its held-out tail.
DATA = "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real/corpus.bin"
EVAL = "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_eval/eval_tail.bin"

STEPS, BATCH, SEQ = 20, 8, 512
DEV = "cuda"


def preflight() -> None:
    busy = subprocess.run(
        ["pgrep", "-ax", "train"], capture_output=True, text=True
    ).stdout.strip()
    if busy:
        loud(f"a train process is alive:\n{busy}\nONE GPU process at a time (AGENTS §1.5)")
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 2**30
    if used >= 1.0:
        loud(f"the card already holds {used:.2f} GiB: waiting is the answer, not sharing")
    print(f"[preflight] no train process; card holds {used:.2f} GiB", flush=True)


def moe_mode(counters) -> str:
    if counters.get("moe_padded_bmm"):
        return "padded"
    if counters.get("moe_naive_dispatch"):
        return "naive"
    return "dense"


def run_arm(name: str, model_over: dict, train_over: dict) -> dict:
    run = load_run(
        "small",
        train_overrides=dict(
            steps=STEPS,
            batch=BATCH,
            seq_len=SEQ,
            log_every=5,
            eval=True,
            eval_every=10,
            eval_batches=4,
            timers=True,
            data=DATA,
            eval_data=EVAL,
            **train_over,
        ),
    )
    if model_over:
        run = dataclasses.replace(run, model=dataclasses.replace(run.model, **model_over))
    model = DormouseModel(run.model, device=DEV)
    n_params = sum(p.numel() for p in model.parameters())
    stream, eval_stream = open_streams(run)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x, y = stream.next_batch()
    _, first_loss, _ = model(x.to(DEV), y.to(DEV))
    first_loss = float(first_loss.detach())
    stream.rewind()

    lines: list[str] = []
    t0 = time.perf_counter()
    report = train(
        model,
        run,
        stream=stream,
        eval_stream=eval_stream,
        device=DEV,
        log=lambda s: (lines.append(s), print(f"[{name}] {s}", flush=True)),
    )
    wall = time.perf_counter() - t0
    counters = report["counters"]
    evals = [line for line in lines if line.startswith("eval")]
    # The FIRST window is the compile/autotune window (FLA's Triton kernels and
    # torch's first allocations), so it is reported and then excluded: a step
    # time with no step index is not a measurement of a step (AGENTS §2.2).
    windows = [
        (float(m.group(1)), float(m.group(2)))
        for m in (re.search(r"([\d.]+) ms/step ([\d.]+) tok/s", ln) for ln in lines)
        if m
    ]
    warm = windows[1:] or windows
    warm_ms = statistics.median(w for w, _ in warm)
    warm_tok = statistics.median(t for _, t in warm)
    out = {
        "arm": name,
        "moe_topk": run.model.moe_topk,
        "params": n_params,
        "first_loss": first_loss,
        "final_loss": report["loss"],
        "wall_ms_per_step": 1e3 * wall / STEPS,
        "ms_per_step": warm_ms,
        "tok_s": warm_tok,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "nan_skipped": report["nan_skipped"],
        "opt_steps": report["opt_steps"],
        "kda": kda_field(counters),
        "moe": moe_mode(counters),
        "dropped": counters.get("moe_dropped_assignments", 0),
        "eval": evals[-1] if evals else "",
        "counters": dict(counters),
    }
    print(
        f"[{name}] loss {out['first_loss']:.4f} -> {out['final_loss']:.4f}  "
        f"{out['ms_per_step']:.1f} ms/step warm (window 1: {windows[0][0]:.1f})  "
        f"{out['tok_s']:.0f} tok/s  "
        f"peak {out['peak_gib']:.2f} GiB  {out['kda']}  moe={out['moe']}  "
        f"dropped={out['dropped']}  nan_skipped={out['nan_skipped']}",
        flush=True,
    )
    if out["eval"]:
        print(f"[{name}] {out['eval']}", flush=True)
    return out


def main() -> int:
    preflight()
    arms = [
        run_arm("small", {}, {}),
        run_arm("small+topk2", {"moe_topk": 2}, {}),
    ]
    print("\n| arm | moe_topk | loss start -> end | ms/step | tok/s | peak GiB | kda | moe |")
    print("|---|---|---|---|---|---|---|---|")
    for a in arms:
        print(
            f"| {a['arm']} | {a['moe_topk']} | {a['first_loss']:.3f} -> {a['final_loss']:.3f} | "
            f"{a['ms_per_step']:.1f} | {a['tok_s']:.0f} | {a['peak_gib']:.2f} | "
            f"{a['kda']} | {a['moe']} |"
        )
    dropped = [a["arm"] for a in arms if a["final_loss"] >= a["first_loss"]]
    print(
        f"\nloss decreasing on {len(arms) - len(dropped)}/{len(arms)} arms; not on {dropped}",
        flush=True,
    )
    return 1 if dropped else 0


if __name__ == "__main__":
    sys.exit(main())
