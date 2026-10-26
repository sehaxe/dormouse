"""The QAT cost: bf16 vs per-tensor refresh vs the one-call stacked refresh.

    uv run --extra gpu python scripts/bench_qat_cost.py

**What was measured (2026-10-21, medium b8 x s512, identical shape).** bf16 ran
at **11.5 k tok/s (355 ms/step)**; `factor_qat=ternary` + `dense_qat=ternary`
ran at **6.6 k tok/s (620 ms/step)** — 1.75x slower. The per-forward
re-quantization was already removed by the once-per-step cache
(`model/qat_cache.py`, commit `f34bb92`). What remained is the per-STEP refresh
issued once per projection TENSOR: the expert bank alone was E=256 quantizer
chains per projection (768 per step) for 24.5 M weight elements — launch-bound
work that one fused pass covers. `qat_project_stacked` projects the whole
`[E, ·, ·]` bank in ONE call; the `stacked` arm below is that path, the
`per_tensor` arm is the old seam emulated slice-by-slice at the same site.

**Arms** (medium exactly as the preset resolves; same weights across arms):

* `bf16`       — the control: both QAT switches off, every number on record.
* `per_tensor` — ternary factors + ternary dense weights + int4 activations,
                 with the bank's stacked projection replaced by E per-slice
                 calls (the pre-fix launch shape). Everything else ships as-is,
                 so the delta is the launch count and nothing else.
* `stacked`    — the same flags, the shipped one-call bank refresh.

**Protocol.** medium (d=1024, E=256, rank 64, max_iter 4), b8 x s512, bf16,
`act_group=0`, synthetic byte ids (the quantizer does not read the corpus), 2
warm steps then 5 timed steps. A STEP here is forward + backward; the optimizer
stage is identical across arms and is excluded so the columns differ only in the
QAT work. Each step starts with the trainer's own `invalidate_qat_cache_all` +
`invalidate_banks_all` — without that the cache stays warm between timed steps
and the refresh count reads 0, which measures nothing. `torch.cuda.synchronize()`
brackets each step and the reported number is the median. Peak allocated VRAM is
printed per arm. `qat_refresh_calls` / `qat_refresh_tensors` are the per-step
launch counts from `model/qat_cache.py` (one stacked call = one call, E tensors);
`qat_reuse` is the cache-hit count from the forward's own counter.

**Preflight, and it is the whole reason this script is safe to run** (the
`bench_opt_graph.py` check): one GPU process at a time (AGENTS §1.5) — no
`dormouse train` alive and the card holding < 1 GiB. On either it prints a SKIP
banner and exits 0: nothing was measured, so there is no number to quote.

**Not run by this task's author**: the card is busy, so every number here is a
tool waiting for a free card. There is NO TIMING for the stacked path until this
script runs. The CPU gates for the same property are `tests/test_qat_cache.py`
(stacked == per-tensor bit for bit, gradient contract, no sync, and the medium
bank site's 3 calls / 768 tensors).
"""

from __future__ import annotations

import dataclasses
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

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight, skip  # noqa: E402

PRESET = "medium"
BATCH, SEQ = 8, 512
WARM, REPS = 2, 5
DEV = "cuda"
TOKENS = BATCH * SEQ


#: The pre-fix seam, kept here as the `per_tensor` arm's own path: the bank's
#: `[E, ·, ·]` stack projected one slice at a time (E calls, same bits).
_SLICE = {}


def _set_sliced(on: bool) -> None:
    import dormouse.model.experts as experts

    if on:
        _SLICE["real"] = experts.qat_project_stacked
        real = _SLICE["real"]

        def per_tensor(w: torch.Tensor, kind: str, alpha: float, *, dim=None):
            return torch.stack(
                [real(w[i : i + 1], kind, alpha, dim=dim)[0] for i in range(w.shape[0])]
            )

        experts.qat_project_stacked = per_tensor
    else:
        if "real" in _SLICE:
            experts.qat_project_stacked = _SLICE.pop("real")


def _model(arm: str, state: dict) -> DormouseModel:
    run = load_run(PRESET)
    over = {"factor_qat": "none", "dense_qat": "none", "act_quant": "none"}
    if arm != "bf16":
        over = {"factor_qat": "ternary", "dense_qat": "ternary", "act_quant": "int4"}
        over["factor_qat_alpha"] = 1.0
    cfg = dataclasses.replace(run.model, **over)
    model = DormouseModel(cfg, device=DEV)
    if state:
        model.load_state_dict(state)
    return model


def _one_step(model: DormouseModel, x: torch.Tensor) -> tuple[int, int, int]:
    """One training step (fwd + bwd) after the trainer's per-step invalidation.

    Returns `(qat_refresh_calls, qat_refresh_tensors, qat_reuse)` from the
    forward's own counters. The invalidation is the step boundary: with a warm
    cache every refresh is a hit and the launch counts read 0.
    """
    model.zero_grad(set_to_none=True)
    invalidate_qat_cache_all(model)
    invalidate_banks_all(model)
    _, loss, counters = model(x, x)
    loss.backward()
    return (
        int(counters.get("qat_refresh_calls", 0)),
        int(counters.get("qat_refresh_tensors", 0)),
        int(counters.get("qat_reuse", 0)),
    )


def timed(model: DormouseModel, x: torch.Tensor) -> tuple[float, float, tuple[int, int, int]]:
    """`(median ms, peak GiB, last step's launch counts)` over warm `REPS`."""
    for _ in range(WARM):
        _one_step(model, x)
    torch.cuda.synchronize()
    times, peaks = [], []
    counts = (0, 0, 0)
    for _ in range(REPS):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        counts = _one_step(model, x)
        torch.cuda.synchronize()
        times.append(1e3 * (time.perf_counter() - t0))
        peaks.append(torch.cuda.max_memory_allocated() / 2**30)
    return statistics.median(times), max(peaks), counts


def main() -> int:
    why = preflight("cuda")
    if why is None:
        # This bench's own ceiling is 1 GiB, tighter than the lane guard's 2.0:
        # a card already holding a second model is contended however idle it looks.
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if smi and int(smi.split()[0]) >= 1024:
            why = f"the card already holds {smi.split()[0]} MiB (this bench's ceiling is 1024)"
    if why is not None:
        return skip(why)
    torch.manual_seed(0)
    x = torch.randint(0, 256, (BATCH, SEQ), device=DEV)
    # ONE parameter set under every arm: build the control, then load its state
    # into the quantized models (QAT adds no parameters, so this is exact).
    base = _model("bf16", {})
    state = base.state_dict()
    rows = []
    for arm in ("bf16", "per_tensor", "stacked"):
        try:
            _set_sliced(arm == "per_tensor")
            model = _model(arm, state)
            ms, peak, counts = timed(model, x)
        finally:
            _set_sliced(False)
        tps = TOKENS / (ms / 1e3) / 1e3
        rows.append((arm, ms, tps, peak, counts))
        del model
        torch.cuda.empty_cache()

    print(
        f"\nshape: {PRESET} d=1024 E=256 r=64 max_iter=4, "
        f"b{BATCH} x s{SEQ} = {TOKENS} tok"
    )
    print("step: forward + backward, per-step invalidation (the optimizer stage is common)\n")
    print(
        f"{'arm':10s} {'ms/step':>9s} {'tok/s':>10s} {'peak GiB':>9s}"
        f" {'refresh_calls':>14s} {'refresh_tensors':>16s} {'qat_reuse':>10s}"
    )
    control = rows[0][1]
    for arm, ms, tps, peak, (calls, tensors, reuse) in rows:
        print(
            f"{arm:10s} {ms:9.2f} {tps:10.0f} {peak:9.2f}"
            f" {calls:14d} {tensors:16d} {reuse:10d}   x{control / ms:.2f}"
        )
    print(
        "\nVOID check: `bf16` must report 0 refresh calls and `qat_reuse=0`; the\n"
        "quantized arms must report `qat_reuse > 0`. `per_tensor` and `stacked`\n"
        "must cover the SAME tensor count: only the call count may differ.\n"
        "NO NUMBER EXISTS for the stacked arm until this script runs on a free card."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
