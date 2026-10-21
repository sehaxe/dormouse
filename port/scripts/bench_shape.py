"""The throughput x shape table for the ternary model, with its eval windows.

    uv run --extra gpu python scripts/bench_shape.py

**What is measured.** `medium` in its FULL ternary form — `factor_qat=ternary`,
`dense_qat=ternary`, `act_quant=int4`, alpha 1.0, i.e. the arm the 2026-10-21
QAT bench timed at 392 ms/step for b8 x s512 — across the five shapes the next
decision compares:

    b8 s512, b16 s512, b8 s1024, b16 s1024, b24 s1024

For each shape: ms/step (median of `REPS`, one `synchronize()` per step),
bytes/step (`batch x seq_len`), B/s, us/B and us/KiB, and peak GiB. The step is
the trainer's: invalidate, fwd + bwd, fused clip, `MixOptimizer.step()`. (The
brief quotes "us/KB" at 86.7 for b8 x s512; that is `355 000 / 4 096`, a
per-BYTE figure. The table prints both so neither number is lost.)

**Why the eval-window column is in the same table.** A BPB is only comparable
inside one window (`eval_batches x batch x seq_len`, AGENTS §2.6). The record is
**81 920 B = 20 x 8 x 512**. The table prints each shape's window at the
protocol's 20 eval batches, its ratio to 81 920, and the (possibly fractional)
batch count that WOULD hit 81 920 — so a shape is never compared to the record
on a different number of bytes without the reader seeing it. b24 x s1024 cannot
hit 81 920 exactly: 3 batches = 73 728 B, 4 = 98 304 B.

**Protocol.** One model and one optimizer; 2 warm steps per shape, then 5 timed
steps. Peak is `max_memory_allocated` since a per-shape reset (so it includes
the model and optimizer state, not just the activations). Synthetic byte ids —
the corpus is irrelevant to a shape table.

**Preflight, and NOT run.** `pgrep -f "dormouse trai[n]"` + `nvidia-smi` under
1 GiB, or a SKIP banner and exit 0. Written while the card is busy: **no row has
been measured**, and the table is a tool waiting for a free card.
"""

from __future__ import annotations

import functools
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from dormouse.config import load_run
from dormouse.model import DormouseModel
from dormouse.model.experts import invalidate_banks_all
from dormouse.model.qat_cache import invalidate_qat_cache_all
from dormouse.train.optim import MixOptimizer, clip_grads

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight as _lane_preflight  # noqa: E402
from _lane import skip  # noqa: E402

# a test pins a 0-arg `preflight`: this table only ever measures the card.
preflight = functools.partial(_lane_preflight, "cuda")

PRESET = "medium"
SHAPES = ((8, 512), (16, 512), (8, 1024), (16, 1024), (24, 1024))
WARM, REPS = 2, 5
DEV = "cuda"
#: The protocol's eval window: `eval_batches x batch x seq_len`, with
#: `eval_batches = 20` and the recorded control shape b8 x s512.
RECORDED_WINDOW = 20 * 8 * 512  # 81 920 B
EVAL_BATCHES = 20

#: The full ternary arm, exactly as `scripts/bench_qat_cost.py` builds its
#: `stacked` arm (and the trainer's own quantized recipe).
TERNARY = {
    "factor_qat": "ternary",
    "dense_qat": "ternary",
    "act_quant": "int4",
    "factor_qat_alpha": 1.0,
}


def rates(batch: int, seq: int, ms: float) -> tuple[float, float, float]:
    """`(B/s, us/B, us/KB)` for one step of `batch x seq` bytes at `ms` ms.

    Both per-byte and per-KiB are returned on purpose: the brief quotes
    "us/KB" at 86.7 for b8 x s512, which is `355 000 / 4 096` — a per-BYTE
    figure. Printing both keeps the brief's number visible and its unit honest.
    """
    per_step = batch * seq
    us = ms * 1e3
    return per_step / (ms / 1e3), us / per_step, us / (per_step / 1024)


def eval_window(batch: int, seq: int, batches: float = EVAL_BATCHES) -> float:
    """`eval_batches x batch x seq_len` bytes — the only comparable BPB unit."""
    return batches * batch * seq


def batches_for(target: float, batch: int, seq: int) -> float:
    """How many eval batches of this shape cover exactly `target` bytes."""
    return target / (batch * seq)


def _optimizer(model: DormouseModel, train) -> MixOptimizer:
    return MixOptimizer(
        model,
        lr=train.lr,
        wd=train.wd,
        opt=train.opt,
        ns_steps=train.ns_steps,
        ns_kind=train.ns_kind,
        colrow_3d=train.muon_colrow_3d,
        muon_state_dtype=train.muon_state_dtype,
        ns_dtype=train.ns_dtype,
        muon_variant=train.muon_variant,
        muon_sf=train.muon_sf,
        muon_sf_wd=train.muon_sf_wd,
    )


def _step(model, opt, train, x: torch.Tensor) -> None:
    model.zero_grad(set_to_none=True)
    invalidate_qat_cache_all(model)
    invalidate_banks_all(model)
    _, loss, _ = model(x, x)
    loss.backward()
    if train.grad_clip > 0:
        clip_grads(model.parameters(), train.grad_clip)
    opt.step()
    invalidate_banks_all(model)
    invalidate_qat_cache_all(model)


def timed(model, opt, train, x: torch.Tensor) -> tuple[float, float]:
    """`(median ms/step, peak GiB)` for one shape, after `WARM` warm steps."""
    for _ in range(WARM):
        _step(model, opt, train, x)
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(REPS):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _step(model, opt, train, x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times), torch.cuda.max_memory_allocated() / 2**30


def main() -> int:
    why = preflight()
    if why is None:
        # This table's own ceiling is 1 GiB, tighter than the lane guard's 2.0:
        # a card already holding a second model is contended however idle it looks.
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if smi and int(smi.split()[0]) >= 1024:
            why = f"the card already holds {smi.split()[0]} MiB (this table's ceiling is 1024)"
    if why is not None:
        return skip(why)
    run = load_run(PRESET, model_overrides=TERNARY)
    torch.manual_seed(0)
    model = DormouseModel(run.model, device=DEV)
    opt = _optimizer(model, run.train)

    print(
        f"\n{PRESET} FULL TERNARY (factor_qat={run.model.factor_qat}, "
        f"dense_qat={run.model.dense_qat}, act_quant={run.model.act_quant}, "
        f"alpha={run.model.factor_qat_alpha}); {WARM} warm + {REPS} timed steps/shape"
    )
    print(
        f"recorded eval window: {EVAL_BATCHES} x 8 x 512 = {RECORDED_WINDOW} B "
        f"(every comparison below is against this many bytes)\n"
    )
    print(
        f"{'shape':12s} {'ms/step':>9s} {'B/step':>8s} {'B/s':>10s} {'us/B':>7s}"
        f" {'us/KB':>9s} {'peak GiB':>9s} {'window@20':>11s} {'x81920':>8s}"
        f" {'n for 81920':>12s}"
    )
    for batch, seq in SHAPES:
        x = torch.randint(0, 256, (batch, seq), device=DEV)
        ms, peak = timed(model, opt, run.train, x)
        bps, us_b, us_kb = rates(batch, seq, ms)
        window = eval_window(batch, seq)
        n = batches_for(RECORDED_WINDOW, batch, seq)
        flag = "" if abs(n - round(n)) < 1e-9 else "*"
        print(
            f"b{batch:<2d} s{seq:<5d} {ms:9.1f} {batch * seq:8d} {bps:10.0f} {us_b:7.1f}"
            f" {us_kb:9.0f} {peak:9.2f} {window:11.0f} {window / RECORDED_WINDOW:7.2f}x"
            f" {n:11.2f}{flag}"
        )
        del x
        torch.cuda.empty_cache()

    print(
        "\n*window@20 is the protocol's 20 eval batches; `n for 81920` is the "
        "fractional batch\ncount that would cover the recorded 81 920 B. Only "
        "b8 x s512 hits it exactly at 20\nbatches; b24 x s1024 cannot (3 = 73 728 B, "
        "4 = 98 304 B). Quote a BPB with its\nwindow, never across two of these rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
