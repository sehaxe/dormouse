"""The fp8 2-D GEMM lane's bench (`mm_dtype = "fp8"`, `tsct.mm_fp8`).

    uv run python scripts/bench_fp8_mm.py            # the shape inventory (CPU-safe)
    uv run python scripts/bench_fp8_mm.py --gpu      # the CUDA bench, after the preflight

What it answers, in the order the lane's claim is made:

1. **Which of the medium model's real 2-D GEMMs the lane can take.** The shape
   inventory is READ FROM THE MODEL, not typed in: a real forward at the
   requested batch/seq runs under a `TorchDispatchMode` that records every
   `aten.mm` / `aten.linear` / `aten.addmm` shape, so a new projection in the
   model shows up here instead of silently missing from the table. Each (K, N)
   is then marked with `tsct.mm_fp8_shape_ok` — the lane's own rule, so the
   coverage column is the lane's answer and not a second opinion.
2. **What the fp8 GEMM costs and buys at those shapes** (GPU only): bf16 `mm`
   vs `torch._scaled_mm` e4m3 per-tensor amax, median ms and TFLOP/s, plus the
   CONVERSION measured as its own three pieces — the amax reduction, the
   scale multiply + fp8 cast (both operands), and the GEMM on already-quantized
   operands — because the conversion is what a real forward pays on top of the
   GEMM. Accuracy is reported the way the spike reported it: max abs error as a
   fraction of the reference's own absmax (the spike's `bench_f_fp8.py` measured
   3.3-4.5 % there, on random-but-realistic weights).
3. **What the spike predicts vs what this card does.** The spike's phase-2
   numbers (1.6-1.8×: [32768,2048]×[2048→64] 23.8 → 39.3 TFLOP/s,
   [32768,64]×[64→5461] 14.2 → 22.3, M=32768, L2 flushed) are printed beside the
   measured rows as the PREDICTION they are; the shapes differ (M=24576 here),
   so the comparison is a range check, not a reproduction.

PREFLIGHT (AGENTS §1.5): with `--gpu` the script refuses to touch the card while
a training run is live (`pgrep -f "dormouse trai[n]"`) or the card holds memory
(≥ 1024 MiB) — one GPU thing at a time. Without `--gpu` nothing is allocated on
the card at all: the inventory is a CPU forward.

Timing contract, the repo's (`bench_free_wins.py`): CUDA events around one call,
median of `--reps` (default 20), `--warmup` (default 5) calls before. No L2
flush — the spike flushed, so its absolute rates are the colder reading and the
columns are not directly comparable across the two documents.

**Not in this file**: the lane's backward (there is none on torch 2.13 —
`RuntimeError: derivative for aten::_scaled_mm is not implemented`, measured
2026-10-18) and an end-to-end forward comparison. Both are named follow-ups in
the lane's report; the GEMM ladder is what this script measures.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from dormouse.config import load_preset
from dormouse.instrument import Counters
from dormouse.model import tsct
from dormouse.model.model import DormouseModel

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight  # noqa: E402

#: The spike's phase-2 fp8 rows (`bench_f_fp8.py`, M=32768, bf16 in/out, per-tensor
#: scales, L2 flushed) — the PREDICTION this bench checks against.
SPIKE_ROWS = (
    # (K, N, bf16 ms, bf16 TFLOP/s, fp8 ms, fp8 TFLOP/s)
    (2048, 64, 0.361, 23.82, 0.219, 39.31),
    (64, 5472, 1.611, 14.21, 1.026, 22.32),
)


# ---------------------------------------------------------------------------
# 1. the shape inventory, read from the model
# ---------------------------------------------------------------------------


class _ShapeSpy(torch.utils._python_dispatch.TorchDispatchMode):
    """Records every 2-D matmul shape a real forward runs.

    `aten.linear` arrives with a `[b, t, d]` input (the model calls `F.linear`
    on the unflattened state), so the row count is `numel() // K`; `aten.mm` and
    `aten.addmm` are already 2-D. Grouped by `(M, K, N)` with a per-forward hit
    count: the count is what makes the FLOP weight visible.
    """

    def __init__(self) -> None:
        self.calls: dict[tuple[int, int, int], int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        shape = None
        if name == "aten.mm.default":
            a, b = args[0], args[1]
            shape = (int(a.shape[0]), int(a.shape[-1]), int(b.shape[-1]))
        elif name == "aten.addmm.default":
            a, b = args[1], args[2]
            shape = (int(a.shape[0]), int(a.shape[-1]), int(b.shape[-1]))
        elif name == "aten.linear.default":
            inp, w = args[0], args[1]
            k, n = int(w.shape[-1]), int(w.shape[0])
            shape = (int(inp.numel()) // k, k, n)
        if shape is not None:
            self.calls[shape] = self.calls.get(shape, 0) + 1
        return func(*args, **(kwargs or {}))


def inventory(preset: str, probe_batch: int, seq: int) -> dict[tuple[int, int], dict]:
    """`{(K, N): {"hits": n, "M": m}}` from ONE real forward at the probe batch."""
    cfg = load_preset(preset)
    model = DormouseModel(cfg)
    model.eval()
    ids = (torch.arange(probe_batch * seq, dtype=torch.long) % 251).view(probe_batch, seq)
    spy = _ShapeSpy()
    with torch.no_grad(), spy:
        model(ids, counters=Counters())
    out: dict[tuple[int, int], dict] = {}
    for (m, k, n), hits in sorted(spy.calls.items()):
        rec = out.setdefault((k, n), {"hits": 0, "M": set()})
        rec["hits"] += hits
        rec["M"].add(m)
    for rec in out.values():
        rec["M"] = sorted(rec["M"])
    return out


# ---------------------------------------------------------------------------
# 2. the bench
# ---------------------------------------------------------------------------


def _time(fn, warmup: int, reps: int) -> float:
    """Median ms over `reps` CUDA-event-timed calls, `warmup` calls before."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        fn()
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
    return statistics.median(times)


def _tflops(m: int, k: int, n: int, ms: float) -> float:
    return 2 * m * k * n / (ms / 1e3) / 1e12


def bench_shape(m: int, k: int, n: int, dt: torch.dtype, warmup: int, reps: int) -> dict:
    """bf16 `mm` vs the fp8 lane, with the conversion split into its pieces."""
    g = torch.Generator(device="cuda").manual_seed(5)
    a = torch.randn(m, k, device="cuda", dtype=dt, generator=g)
    w = torch.randn(n, k, device="cuda", dtype=dt, generator=g) / k**0.5

    ref = a @ w.t()
    bf16_ms = _time(lambda: a @ w.t(), warmup, reps)

    # The conversion, measured as its own three pieces: (1) the amax reduction,
    # (2) amax + scale + cast (both operands) — so (2) - (1) is the scale
    # multiply and the cast, and (3) the GEMM on already-quantized operands.
    def amax_only() -> tuple[torch.Tensor, torch.Tensor]:
        return a.detach().abs().amax(), w.detach().abs().amax()

    def quantize() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        aq, sa = tsct.fp8_quantize_2d(a, dt)
        wq, sw = tsct.fp8_quantize_2d(w, dt)
        return aq, wq, sa, sw

    amax_ms = _time(amax_only, warmup, reps)
    quant_ms = _time(quantize, warmup, reps)
    aq, wq, sa, sw = quantize()
    wq_c = wq.t()
    gemm_ms = _time(lambda: torch._scaled_mm(aq, wq_c, sa, sw, out_dtype=dt), warmup, reps)
    lane_ms = _time(lambda: tsct.mm_fp8(a, w, dt), warmup, reps)

    out = tsct.mm_fp8(a, w, dt)
    diff = (out.float() - ref.float()).abs()
    rec = {
        "M": m,
        "K": k,
        "N": n,
        "dtype": str(dt),
        "bf16_ms": round(bf16_ms, 4),
        "bf16_tflops": round(_tflops(m, k, n, bf16_ms), 2),
        "conv_amax_ms": round(amax_ms, 4),
        "conv_quant_ms": round(quant_ms, 4),
        "conv_cast_ms": round(quant_ms - amax_ms, 4),
        "fp8_gemm_ms": round(gemm_ms, 4),
        "fp8_gemm_tflops": round(_tflops(m, k, n, gemm_ms), 2),
        "lane_ms": round(lane_ms, 4),
        "lane_tflops": round(_tflops(m, k, n, lane_ms), 2),
        "speedup_gemm": round(bf16_ms / gemm_ms, 2),
        "speedup_lane": round(bf16_ms / lane_ms, 2),
        "max_abs_err": float(diff.max()),
        "max_abs_err_over_absmax_ref": float(diff.max() / ref.float().abs().max()),
    }
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu", action="store_true", help="run the CUDA bench (preflight first)")
    ap.add_argument("--preset", default="medium", help="the model whose shapes to read")
    ap.add_argument("--batch", type=int, default=24, help="the M = batch x seq row count")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--probe-batch", type=int, default=2, help="rows the inventory forward runs at")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--json", default="", help="write the measured rows here (e.g. ~/logs/...)")
    args = ap.parse_args()

    print(
        f"[env] torch {torch.__version__} cuda {torch.version.cuda} "
        f"fp8={hasattr(torch, '_scaled_mm')}"
    )
    print(
        f"[inventory] reading {args.preset}'s 2-D GEMM shapes from a real forward "
        f"(batch {args.probe_batch} x seq {args.seq}) ...",
        flush=True,
    )
    inv = inventory(args.preset, args.probe_batch, args.seq)
    rows = args.batch * args.seq
    payload: dict = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "preset": args.preset,
        "batch": args.batch,
        "seq": args.seq,
        "probe_batch": args.probe_batch,
        "rows": rows,
        "warmup": args.warmup,
        "reps": args.reps,
        "l2_flushed": False,
        "spike_rows": SPIKE_ROWS,
        "inventory": [
            {
                "K": k,
                "N": n,
                "hits_per_forward": rec["hits"],
                "M_seen": rec["M"],
                "lane": "fp8" if tsct.mm_fp8_shape_ok(k, n) else "bf16",
            }
            for (k, n), rec in sorted(inv.items())
        ],
        "measured": [],
    }
    print(f"{'M':>7} {'K':>6} {'N':>6} {'hits/fwd':>8}  lane")
    for (k, n), rec in sorted(inv.items()):
        ok = tsct.mm_fp8_shape_ok(k, n)
        print(
            f"{rows:>7} {k:>6} {n:>6} {rec['hits']:>8}  "
            f"{'fp8' if ok else 'bf16 (K or N not a multiple of 16)'}"
        )
    covered = sum(r["hits"] for (k, n), r in inv.items() if tsct.mm_fp8_shape_ok(k, n))
    total = sum(r["hits"] for r in inv.values())
    print(
        f"[inventory] {len(inv)} distinct 2-D GEMM shapes, {total} calls/forward; "
        f"the lane covers {covered}/{total} by shape"
    )

    if not args.gpu:
        print(
            "\nCPU run: no CUDA work done. `--gpu` runs the bench (preflight: "
            '`pgrep -f "dormouse trai[n]"` + nvidia-smi, refuses a busy card).'
        )
        _write_json(args.json, payload)
        return 0

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
        print(f"[preflight] REFUSED: {why}. One GPU thing at a time (AGENTS §1.5).")
        _write_json(args.json, payload)
        return 2
    dt = torch.bfloat16
    print("\n[spike prediction — PREDICTED, not measured here: M=32768, L2 flushed]")
    for k, n, _bf16_ms, bf16_tf, _fp8_ms, fp8_tf in SPIKE_ROWS:
        print(
            f"  K={k} N={n}: bf16 {bf16_tf:.2f} TFLOP/s -> fp8 {fp8_tf:.2f} TFLOP/s "
            f"({bf16_tf / fp8_tf:.2f}x)"
        )
    print(
        f"\n{'M':>7} {'K':>6} {'N':>6} {'bf16 ms':>9} {'bf16 TF/s':>10} "
        f"{'fp8 gemm ms':>12} {'fp8 TF/s':>9} {'conv ms':>8} {'lane ms':>8} "
        f"{'lane TF/s':>10} {'gemm x':>7} {'lane x':>7} {'err/absmax':>11}"
    )
    measured = []
    for k, n in sorted(inv):
        if not tsct.mm_fp8_shape_ok(k, n):
            print(
                f"{rows:>7} {k:>6} {n:>6}   bf16: K or N is not a multiple of 16 "
                f"(the lane's own rule) — not benched"
            )
            continue
        r = bench_shape(rows, k, n, dt, args.warmup, args.reps)
        measured.append(r)
        print(
            f"{r['M']:>7} {r['K']:>6} {r['N']:>6} {r['bf16_ms']:>9.3f} "
            f"{r['bf16_tflops']:>10.2f} {r['fp8_gemm_ms']:>12.3f} "
            f"{r['fp8_gemm_tflops']:>9.2f} {r['conv_quant_ms']:>8.3f} "
            f"{r['lane_ms']:>8.3f} {r['lane_tflops']:>10.2f} "
            f"{r['speedup_gemm']:>7.2f} {r['speedup_lane']:>7.2f} "
            f"{r['max_abs_err_over_absmax_ref']:>11.4f}"
        )
    if measured:
        print(
            f"[conversion] amax {statistics.median(r['conv_amax_ms'] for r in measured):.3f} ms, "
            f"scale+cast {statistics.median(r['conv_cast_ms'] for r in measured):.3f} ms "
            f"(median over the benched shapes, both operands)"
        )
        med = statistics.median
        print(
            f"[lane] median gemm speedup {med(r['speedup_gemm'] for r in measured):.2f}x, "
            f"median end-to-end {med(r['speedup_lane'] for r in measured):.2f}x "
            f"(conversion included — what a forward pays)"
        )
    payload["measured"] = measured
    _write_json(args.json, payload)
    return 0


def _write_json(path: str, payload: dict) -> None:
    """The artifact, written on EVERY path (a CPU run records the inventory)."""
    if not path:
        return
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[json] {p}")


if __name__ == "__main__":
    raise SystemExit(main())
