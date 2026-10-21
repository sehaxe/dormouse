"""The QAT cost: bf16 vs per-forward quantization vs the once-per-step cache.

    uv run --extra gpu python scripts/bench_qat_cost.py

**What was measured (2026-10-21, the 9.2 M polygon).** The full-quantized
recipe — `factor_qat=ternary`, `dense_qat=ternary`, `act_quant=int4` — ran at
**17.1 k tok/s against 40.9 k bf16: 2.4x slower**. The waste was structural:
every forward re-quantized every weight (absmean, divide, round, clamp,
multiply), although the weights change only once per optimizer step.
`model/qat_cache.py` computes the detached projection once per trained step and
every forward re-composes `w + alpha·(q − w).detach()`.

**Arms** (`small` exactly as the preset resolves; same weights across arms):

* `bf16`   — the control: both QAT switches off, every number on record.
* `old`    — ternary factors + ternary dense weights + int4 activations, with
             the caches stripped (`_qat_cache = None`): the pre-fix forward that
             re-quantizes everything per call. The activation STE is also put
             back to its two-pass `x + (xq − x).detach()` form, so this arm is
             the old code path and not a hybrid.
* `cached` — the same flags, the cache attached and invalidated once per step
             (the trainer's `invalidate_qat_cache_all`): the shipped path.

**Protocol.** `small` (d_768, E=3, rank 64, max_iter 4), b8 × s512, bf16,
`act_group=0`, synthetic byte ids (the quantizer does not read the corpus), 2
warm steps then 5 timed steps. A STEP here is forward + backward; the optimizer
stage is identical across arms and is excluded so the columns differ only in
the QAT work. `torch.cuda.synchronize()` brackets each step and the reported
number is the median. Peak allocated VRAM is printed per arm. `qat_reuse` is
the cache-hit count from the forward's own counter, so a cached arm reporting
0 measured the uncached path wearing the cache's name (VOID).

**Preflight, and it is the whole reason this script is safe to run** (the
`bench_opt_graph.py` check): one GPU process at a time (AGENTS §1.5) — no
`dormouse train` alive and the card holding < 1 GiB. On either it prints a SKIP
banner and exits 0: nothing was measured, so there is no number to quote.

**Not run by this task's author**: the card is busy, so every number here is a
tool waiting for a free card. The CPU gates for the same property are
`tests/test_qat_cache.py` (bit-identical forward, identical gradient, the
stale-cache bug, `qat_reuse`).
"""

from __future__ import annotations

import dataclasses
import statistics
import subprocess
import sys
import time

import torch

from dormouse.config import load_run
from dormouse.model import DormouseModel
from dormouse.model.act_quant import quantize_act

PRESET = "small"
BATCH, SEQ = 8, 512
WARM, REPS = 2, 5
DEV = "cuda"
TOKENS = BATCH * SEQ


def skip(why: str) -> int:
    """A loud SKIP: the reason, and no table. Exit 0 — the code is not at fault."""
    print(
        f"\n=== SKIP: {why}\n"
        f"=== ONE GPU PROCESS AT A TIME (AGENTS §1.5): nothing was measured, so there\n"
        f"=== is no number to quote. Re-run when the card is free.\n",
        flush=True,
    )
    return 0


def preflight() -> str | None:
    """`None` when the card is ours, else the reason to skip."""
    busy = subprocess.run(
        ["pgrep", "-af", "dormouse trai[n]"], capture_output=True, text=True
    ).stdout.strip()
    if busy:
        return f"a dormouse train process is alive:\n{busy}"
    if not torch.cuda.is_available():
        return "no CUDA device is visible"
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 2**30
    if used >= 1.0:
        return f"the card already holds {used:.2f} GiB: waiting is the answer, not sharing"
    print(f"[preflight] no train process; card holds {used:.2f} GiB", flush=True)
    return None


#: The pre-fix activation STE, kept here as the `old` arm's own path. The
#: modules bind `fake_quant_act` at import (`experts.py`) or lazily
#: (`TSCTLinear._act`), so the arm patches BOTH bindings.
_LEGACY = {}


def _legacy_fake_quant_act(x: torch.Tensor, fmt: str, group: int = 0) -> torch.Tensor:
    if fmt == "none":
        return x
    if fmt in ("mxfp8", "v2_int4", "v2_int8"):
        return quantize_act(x, fmt, group)
    xq = quantize_act(x, fmt, group)
    return x + (xq - x).detach()


def _set_legacy(on: bool) -> None:
    import dormouse.model.act_quant as aq
    import dormouse.model.experts as experts

    if on:
        _LEGACY["aq"] = aq.fake_quant_act
        _LEGACY["experts"] = experts.fake_quant_act
        aq.fake_quant_act = _legacy_fake_quant_act
        experts.fake_quant_act = _legacy_fake_quant_act
    else:
        if "aq" in _LEGACY:
            aq.fake_quant_act = _LEGACY.pop("aq")
        if "experts" in _LEGACY:
            experts.fake_quant_act = _LEGACY.pop("experts")


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
    if arm == "old":
        for m in model.modules():
            if getattr(m, "_qat_cache", None) is not None:
                m._qat_cache = None
    return model


def _one_step(model: DormouseModel, x: torch.Tensor) -> int:
    """Forward + backward; returns the forward's `qat_reuse` count."""
    model.zero_grad(set_to_none=True)
    _, loss, counters = model(x, x)
    loss.backward()
    return int(counters.get("qat_reuse", 0))


def timed(model: DormouseModel, x: torch.Tensor) -> tuple[float, float, int]:
    """`(median ms, peak GiB, qat_reuse per step)` over warm `REPS`."""
    for _ in range(WARM):
        _one_step(model, x)
    torch.cuda.synchronize()
    times, peaks, reuse = [], [], 0
    for _ in range(REPS):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reuse = _one_step(model, x)
        torch.cuda.synchronize()
        times.append(1e3 * (time.perf_counter() - t0))
        peaks.append(torch.cuda.max_memory_allocated() / 2**30)
    return statistics.median(times), max(peaks), reuse


def main() -> int:
    why = preflight()
    if why is not None:
        return skip(why)
    torch.manual_seed(0)
    x = torch.randint(0, 256, (BATCH, SEQ), device=DEV)
    # ONE parameter set under every arm: build the control, then load its state
    # into the quantized models (QAT adds no parameters, so this is exact).
    base = _model("bf16", {})
    state = base.state_dict()
    rows = []
    for arm in ("bf16", "old", "cached"):
        try:
            _set_legacy(arm == "old")
            model = _model(arm, state)
            ms, peak, reuse = timed(model, x)
        finally:
            _set_legacy(False)
        tps = TOKENS / (ms / 1e3) / 1e3
        rows.append((arm, ms, tps, peak, reuse))
        del model
        torch.cuda.empty_cache()

    print(f"\nshape: {PRESET} d=768 E=3 r=64 max_iter=4, b{BATCH} x s{SEQ} = {TOKENS} tok")
    print("step: forward + backward (the optimizer stage is identical across arms)\n")
    print(f"{'arm':8s} {'ms/step':>9s} {'tok/s':>10s} {'peak GiB':>9s} {'qat_reuse':>10s}")
    control = rows[0][1]
    for arm, ms, tps, peak, reuse in rows:
        print(f"{arm:8s} {ms:9.2f} {tps:10.0f} {peak:9.2f} {reuse:10d}   x{control / ms:.2f}")
    print(
        "\nVOID check: `old` must report qat_reuse=0 and `cached` > 0. The recorded\n"
        "reference is 40.9 k tok/s bf16 / 17.1 k tok/s per-forward at this polygon\n"
        "(2026-10-21); this table is a fresh measurement, not that number re-quoted."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
