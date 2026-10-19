"""One real GPU smoke of the ByteFlow arm, 10 steps — the WIRING gate.

    uv run --extra gpu python scripts/smoke_byteflow.py

What this verifies, and what it does not. It verifies that the channel runs
(`bf_channel=1` on the counter line), that OUR loop runs on the K patch latents
(`iter`, `kda=`), that the byte decoder's loss moves and that its gradient
reaches the back stage — a 10-step run cannot judge quality, and this file never
claims it. The step cost, tokens/s and peak VRAM are printed with the step index
they were measured at (a step-time reading with no step index is not a
measurement, AGENTS §3.1; the first window is the autotune window and is
reported separately).

The resolved config is `small` with the byteflow geometry named explicitly: the
crate's defaults are BFlowNet-600M (d_local 512, K 3200), so a smoke names its
own widths and prints every override. Nothing else about the run is hidden —
`bf16=false` because the byteflow arm is fp32 by construction, `use_engram=false`
because the memory arm has no byte position to hash on the patch path (both are
loud refusals in `config.validate`).

Machine discipline first (AGENTS §1.5): no other `train` process alive, and the
card holding < 1 GiB before the first kernel.
"""

from __future__ import annotations

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

#: The port's data on this box (AGENTS §2.6): the train file and its held-out tail.
DATA = "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real/corpus.bin"
EVAL = "/mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_eval/eval_tail.bin"

STEPS, BATCH, SEQ, DEV = 10, 4, 512, "cuda"

#: d=256 (n_heads*head_dim = 4*64) and a byteflow geometry the smoke can afford:
#: burn's `byteflow_9m` widths (schema.rs:751-794) at d_local 96 / d_global 256,
#: K = 128 patches over a 512-byte window.
MODEL_OVER = dict(
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

TRAIN_OVER = dict(
    steps=STEPS,
    batch=BATCH,
    seq_len=SEQ,
    log_every=2,
    ckpt_every=10_000,  # nothing is written: this is a wiring gate
    eval=True,
    eval_every=5,
    eval_batches=2,
    timers=True,
    data=DATA,
    eval_data=EVAL,
)


def preflight() -> None:
    busy = subprocess.run(["pgrep", "-ax", "train"], capture_output=True, text=True).stdout.strip()
    if busy:
        loud(f"a train process is alive:\n{busy}\nONE GPU process at a time (AGENTS §1.5)")
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 2**30
    if used >= 1.0:
        loud(f"the card already holds {used:.2f} GiB: waiting is the answer, not sharing")
    print(f"[preflight] no train process; card holds {used:.2f} GiB", flush=True)


def main() -> int:
    preflight()
    run = load_run("small", model_overrides=MODEL_OVER, train_overrides=TRAIN_OVER)
    assert run.model.use_byteflow and run.model.byteflow_k_tokens == 128
    model = DormouseModel(run.model, device=DEV)
    print(
        f"[config] d_model={run.model.d_model} iter={run.model.max_iter} "
        f"k={run.model.byteflow_k_tokens} d_local={run.model.byteflow_d_local} "
        f"d_global={run.model.byteflow_d_global} E={run.model.byteflow_e_layers} "
        f"bins={run.model.byteflow_bins} bf16={run.model.bf16} "
        f"params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    stream, eval_stream = open_streams(run)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x, y = stream.next_batch()
    _, first_loss, first_counters = model(x.to(DEV), y.to(DEV))
    first_loss = float(first_loss.detach())
    print(f"[step 0] loss={first_loss:.4f} {first_counters.line()}", flush=True)
    stream.rewind()

    lines: list[str] = []
    t0 = time.perf_counter()
    report = train(
        model,
        run,
        stream=stream,
        eval_stream=eval_stream,
        device=DEV,
        log=lambda s: (lines.append(s), print(s, flush=True)),
    )
    wall = time.perf_counter() - t0
    counters = report["counters"]
    evals = [line for line in lines if line.startswith("eval")]
    windows = [
        (float(m.group(1)), float(m.group(2)))
        for m in (re.search(r"([\d.]+) ms/step ([\d.]+) tok/s", ln) for ln in lines)
        if m
    ]
    # Window 1 is the autotune/compile window; the rest is the steady reading.
    warm = windows[1:] or windows
    ms = statistics.median(w for w, _ in warm)
    tok = statistics.median(t for _, t in warm)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"\n[summary] params={sum(p.numel() for p in model.parameters()):,} "
        f"loss {first_loss:.4f} -> {report['loss']:.4f} "
        f"nan_skipped={report['nan_skipped']} opt_steps={report['opt_steps']}\n"
        f"[summary] {ms:.1f} ms/step warm ({windows[0][0]:.1f} on window 1) "
        f"{tok:.0f} tok/s peak {peak:.2f} GiB wall {1e3 * wall / STEPS:.1f} ms/step\n"
        f"[summary] {kda_field(counters)} {counters.line()}",
        flush=True,
    )
    for line in evals:
        print(f"[summary] {line}", flush=True)
    n_channel = counters.get("bf_channel", 0)
    # `evaluate` runs its own counter set per batch (it returns only the loss), so
    # the total below is the TRAIN forwards — a number that says exactly what it
    # covers.
    loss_down = report["loss"] < first_loss
    print(
        f"\nwiring: bf_channel={n_channel} of {STEPS} train forwards; "
        f"iter={counters.get('iter', 0)} = {STEPS} x max_iter({run.model.max_iter}); "
        f"loss {'down' if loss_down else 'NOT down'}"
    )
    return 0 if n_channel == STEPS and loss_down else 1


if __name__ == "__main__":
    sys.exit(main())
