"""Bench: the eager SwiGLU vs the fused Triton epilogue (`model/glu_kernel.py`).

    GPU_TIMEOUT=900 bash scripts/gpu_lock.sh .venv/bin/python scripts/bench_glu.py

The shape is the DENSE FFN at the live recipe's batch (`glu_kernel.BENCH_SHAPE`:
24·1024 tokens, d=832, d_ffn=3072), bf16 — the shape a `ffn_form="dense"` run of
`configs/flagship3_dense.toml` would use. It is NOT a shape the live checkpoint
runs: that preset is `block_form="transformer"`, whose GELU FFN has no
`silu(g)·u` at all (module docstring, `glu_kernel.py`). The table below measures
the kernel against the function it claims to replace, nothing else.

Arms, `torch.no_grad()`, median of `REPS` warm windows (min and peak allocated
VRAM per arm):

* `eager` — `F.silu(F.linear(x, Wg, bg)) * F.linear(x, Wu)`;
* `fused` — one Triton kernel: both dot products + epilogue in registers, a
  single `[tokens, f]` store.

Also measured, because the honesty rule is "parity is measured, not asserted":

* max |Δ| of the FFN output (bf16 and fp32) against the eager arm at the bench
  shape, with the chosen tolerance;
* max |Δ| of the LOGITS and LOSS of a `ffn_form="dense"` model run with
  `fused_glu=false` vs `true` (bf16 and fp32) — the kernel's view as the model
  consumes it. On CPU the flag is the eager path bit for bit (the training gate
  in `tests/test_glu_kernel.py`); this is the CUDA/eval measurement.

PREFLIGHT (AGENTS §1.5): no CUDA job starts here. `_lane.preflight("cuda")` reads
`nvidia-smi`, refuses when a compute process holds the card, and the script exits
2 with the pid in the reason — nothing was measured, so there is no number to
quote. Run it through `scripts/gpu_lock.sh`, which also queues.

Exit codes: 0 = measured, 2 = card busy / no device (nothing measured).
"""

from __future__ import annotations

import dataclasses
import statistics
import sys
import time
from pathlib import Path

import torch

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path APPENDED, not inserted.
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight  # noqa: E402

from dormouse.config import load_preset  # noqa: E402
from dormouse.instrument import Counters  # noqa: E402
from dormouse.model import DormouseModel  # noqa: E402
from dormouse.model import glu_kernel as gk  # noqa: E402

M, D, F = gk.BENCH_SHAPE
REPS, WARM = 20, 5


def timed(fn) -> tuple[float, float, float]:
    """`(median ms, min ms, peak GiB)` over warm `REPS`, each its own alloc peak."""
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    times, peaks = [], []
    for _ in range(REPS):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(1e3 * (time.perf_counter() - t0))
        peaks.append(torch.cuda.max_memory_allocated() / 2**30)
    return statistics.median(times), min(times), max(peaks)


def tensors(dt: torch.dtype):
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(M, D, generator=gen, device="cuda")
    wg = torch.randn(F, D, generator=gen, device="cuda") / D**0.5
    wu = torch.randn(F, D, generator=gen, device="cuda") / D**0.5
    b = torch.randn(F, generator=gen, device="cuda") * 0.01
    return tuple(t.to(dt) for t in (x, wg, wu, b))


def ffn_table(dt: torch.dtype) -> dict[str, tuple[float, float, float, float]]:
    x, wg, wu, b = tensors(dt)
    arms = {
        "eager": lambda: gk.swiglu_eager(x, wg, wu, b, dt=dt),
        "fused": lambda: gk.fused_swiglu(x, wg, wu, b, dt=dt),
    }
    out: dict[str, tuple[float, float, float, float]] = {}
    with torch.no_grad():
        ref = arms["eager"]()
        scale = float(ref.float().abs().max())
        for name, fn in arms.items():
            err = float((fn().float() - ref.float()).abs().max()) / max(scale, 1e-30)
            med, best, peak = timed(fn)
            out[name] = (med, best, peak, err)
            print(
                f"[{dt} {name}] median {med:.3f} ms  min {best:.3f} ms  "
                f"peak {peak:.2f} GiB  max_rel_vs_eager {err:.3e}",
                flush=True,
            )
        # The path the seam actually takes, recorded (not inferred).
        counters = Counters()
        gk.fused_swiglu_or_eager(x, wg, wu, b, dt=dt, counters=counters)
        print(f"[{dt} seam] {dict(counters)} why={getattr(counters, 'why', '')!r}", flush=True)
    return out


def loss_parity(dt: torch.dtype) -> tuple[float, float, float]:
    """`(max |Δ logits|, |Δ loss|, max |Δ ffn out|)` for a dense model, eval."""
    base = dataclasses.replace(
        load_preset("tiny"), ffn_form="dense", n_experts=1, moe_topk=0, bf16=dt is torch.bfloat16
    )
    torch.manual_seed(0)
    off = DormouseModel(base).to("cuda").eval()
    torch.manual_seed(0)
    on = DormouseModel(dataclasses.replace(base, fused_glu=True)).to("cuda").eval()
    on.load_state_dict(off.state_dict())
    ids = (torch.arange(2 * 16, dtype=torch.long) % base.vocab).reshape(2, 16).to("cuda")
    with torch.no_grad():
        l_off, loss_off, _ = off(ids, ids)
        l_on, loss_on, counters = on(ids, ids)
        if dt is torch.float32:
            # The flag REFUSES fp32 (measured 1.9x slower than cuBLAS), so the
            # model-level delta is the fallback's exact 0. The fp32 KERNEL's own
            # parity is the direct `fused_swiglu` row in `ffn_table` above.
            assert counters.get("glu_kernel_fallback", 0) > 0, dict(counters)
            assert "glu_kernel" not in counters, dict(counters)
        else:
            assert counters["glu_kernel"] > 0, dict(counters)
            assert "glu_kernel_fallback" not in counters, dict(counters)
        # The FFN module itself, at its own width: the kernel's output delta.
        x = torch.randn(256, base.d_model, device="cuda").to(dt)
        ffn_off = off.loop.moe.experts[0]
        ffn_on = on.loop.moe.experts[0]
        y_off = ffn_off(x, counters=Counters())
        y_on = ffn_on(x, counters=Counters())
    d_logits = float((l_on - l_off).abs().max())
    d_loss = abs(float(loss_on) - float(loss_off))
    d_ffn = float((y_on - y_off).abs().max())
    print(
        f"[{dt} model] max_abs dlogits {d_logits:.3e}  |dloss| {d_loss:.3e}  "
        f"max_abs dffn {d_ffn:.3e}",
        flush=True,
    )
    return d_logits, d_loss, d_ffn


def main() -> int:
    why = preflight("cuda")
    if why is not None:
        print(
            f"\n=== BUSY: {why}\n"
            f"=== ONE GPU PROCESS AT A TIME (AGENTS 1.5): nothing was measured.\n"
            f"=== Escape: wait for the pid above, or run through scripts/gpu_lock.sh.\n",
            flush=True,
        )
        return 2
    if not gk.TRITON_AVAILABLE:
        print("=== no triton in this environment: the fused arm cannot run", flush=True)
        return 2
    torch.manual_seed(11)
    x, wg, wu, b = tensors(torch.bfloat16)
    with torch.no_grad():
        if gk.glu_reason(x, wg, wu, torch.bfloat16) is not None:
            print(f"=== the kernel refuses this call: {gk.glu_reason(x, wg, wu, torch.bfloat16)}")
            return 2
    print(f"[shape] tokens={M} d={D} f={F} tile={gk.tile_for(M, F, D)}", flush=True)

    tables = {}
    for dt in (torch.bfloat16, torch.float32):
        print(f"\n=== FFN output, {dt} ===", flush=True)
        tables[dt] = ffn_table(dt)
        loss_parity(dt)

    print("\n| dtype | arm | median ms | min ms | peak GiB | max_rel_vs_eager |")
    print("|---|---|---|---|---|---|")
    for dt, table in tables.items():
        for name, (med, best, peak, err) in table.items():
            print(f"| {dt} | {name} | {med:.3f} | {best:.3f} | {peak:.2f} | {err:.3e} |")
    print("\n| dtype | fused/eager | verdict |")
    print("|---|---|---|")
    for dt, table in tables.items():
        ratio = table["fused"][0] / table["eager"][0]
        verdict = "fused wins" if ratio < 1 else "EAGER WINS"
        if 1 - 0.03 <= ratio < 1:
            verdict = "fused wins <3% (delete per the honesty rule)"
        print(f"| {dt} | {ratio:.3f}x | {verdict} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
