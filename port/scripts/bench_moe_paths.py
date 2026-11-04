"""Time the routed expert pass: `naive` / `triton` / `batched` / `bucketed`, one shape at a time.

    uv run --extra gpu python scripts/bench_moe_paths.py --geometry all --routing uniform
    uv run --extra gpu python scripts/bench_moe_paths.py --geometry m400 --routing pathological
    uv run --extra gpu python scripts/bench_moe_paths.py --geometry m400 --arm bucketed

The 2026-11-04 finding: at 8192 tokens, E=512, d=1536, f=4096, rank 64, top-2 the
per-expert Python loop is ~99.6 ms of a ~107 ms iteration and the per-expert
bank (`torch.bmm` over `[E, cap, ·]`) is ~11.5 ms — the same arithmetic, 8.7x.
This script measures the dispatch arms on the REAL module at that shape and at
the b8-s1024 / b1-s8192 presets, forward and forward+backward, with the kernel
launch count of ONE forward+backward pass and the bank bytes of the arm.

`--routing uniform` is the router's own top-2 (the shape every healthy run
sees). `--routing pathological` puts 90% of the assignments on expert 0 — the
2026-11-04 collapse — where `batched` REFUSES (`MOE_BANK_SAFETY`) and `bucketed`
runs one bucket for the hot expert plus one for the tail. `--arm` filters the
table; the default measures every arm that can run. The owner runs it on the
card; without a free card it prints the preflight reason and exits 0 (nothing
measured).

Shapes: the MoE sees `[batch*seq, d]`, so b8 x s1024 and b1 x s8192 both route
8192 tokens; the difference is the model geometry each preset names, not the MoE
input length. `m400` is the measured 93%-shape.

The arms share ONE routing (the router's top-k on the same `x`, or the
pathological fixture), the same weights and the same rows, so the table compares
implementations, not samples. `moe_capacity=0` (dropless) is the config that
needs the fix; `padded` refuses there, `naive` and `triton(dom/le)` are the
current paths, `batched` is the one-bank arm, `bucketed` is the same bank split
by row-count class. `triton` is reported as the reason it cannot run when the
build/card cannot serve it; it is never silently omitted.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from _lane import preflight, skip  # noqa: E402

from dormouse.config import ModelConfig  # noqa: E402
from dormouse.instrument import Counters, LoudError  # noqa: E402
from dormouse.model.experts import MOE_BUCKET_BYTES, SparseMoE  # noqa: E402

try:
    from dormouse.model import moe_kernels as mk
except ImportError:  # pragma: no cover - a CPU-only wheel
    mk = None

#: `tokens` is the routed MoE input (batch*seq for the preset named).
SHAPES = {
    "m400": dict(tokens=8 * 1024, d=1536, f=4096, experts=512, rank=64),  # the 93% shape
    "b8s1024": dict(tokens=8 * 1024, d=1024, f=2730, experts=256, rank=64),  # medium
    "b1s8192": dict(tokens=8 * 1024, d=2048, f=5461, experts=1024, rank=64),  # b1
}

DEV = "cuda"


def _timed(fn, reps: int, warm: int) -> tuple[float, float]:
    """`(median ms, min ms)` from CUDA events; `reps` calls, one warm-up excluded.

    Events, not `perf_counter`: the forward is a queue of launches, and a host
    clock around it measures the host's enqueue rate (the bug `part_bench.py`
    exists to avoid). One event pair per call keeps the min honest.
    """
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times), min(times)


def _launches(fn) -> int | None:
    """`cudaLaunchKernel` events for ONE call, or `None` if kineto is absent."""
    try:
        from torch.profiler import ProfilerActivity, profile
    except ImportError:  # pragma: no cover
        return None
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.name == "cudaLaunchKernel")


def _arms(moe: SparseMoE, x, weight, index, order, keep, want: str):
    """The dispatch arms as closures, with the reason an arm cannot run."""

    def naive():
        return moe._naive_dispatch(x, weight, index, order, keep, Counters())

    def batched():
        return moe._batched_dispatch(x, weight, index, order, keep)

    def bucketed():
        return moe._bucketed_dispatch(x, weight, index, order, keep, Counters())

    def triton():
        return moe._triton_dispatch(x, weight, index, order, keep, dropless=True)

    arms = {"naive": naive, "batched": batched, "bucketed": bucketed}
    why = None
    if mk is None:
        why = "no triton import"
    elif not torch.cuda.is_available():
        why = "no CUDA device"
    if why is None:
        arms["triton"] = triton
    if want != "all":
        arms = {name: fn for name, fn in arms.items() if name == want}
        if not arms:
            print(f"  arm {want!r} cannot run here: {why}", flush=True)
    return arms, why


def _routing(moe: SparseMoE, x, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """`(index, weight)` — the router's top-2, or the 90%-on-one-expert collapse."""
    if mode == "uniform":
        with torch.no_grad():
            probs = F.softmax(moe.router(x), dim=-1)
            weight, index = probs.topk(2, dim=-1)
            weight = weight / weight.sum(dim=-1, keepdim=True)
        return index, weight
    n, k = x.shape[0], 2
    flat = torch.zeros(n * k, dtype=torch.long, device=x.device)
    cut = int(0.9 * n * k)
    flat[cut:] = torch.arange(n * k - cut, device=x.device) % (moe.n_experts - 1) + 1
    return flat.reshape(n, k), torch.ones(n, k, device=x.device) / k


def _bank_bytes(moe: SparseMoE, arm: str, x, weight, index, order, keep):
    """`(bank bytes, buckets)` for the arm's ONE real bank, or `(None, None)`.

    A `[n_b, cap_b, ·]` allocation needs its numbers before the timing loop, so
    this runs the arm once (`no_grad`, a warm-up the timing excludes anyway).
    `batched` may REFUSE here — that is the row the pathological case exists to
    show, and the arm's own call then prints the message.
    """
    itemsize = moe._bank_dtype().itemsize
    if arm == "bucketed":
        counters = Counters()
        with torch.no_grad():
            moe._bucketed_dispatch(x, weight, index, order, keep, counters)
        return counters.get("moe_bank_bytes"), counters.get("moe_buckets")
    if arm == "batched":
        try:
            cap = moe._batched_cap(index, weight)
        except LoudError:
            return None, None
        return moe.n_experts * cap * moe.d * itemsize, None
    return None, None


def _backward(moe: SparseMoE, fn) -> None:
    """One forward+backward, grads zeroed in place first."""
    moe.zero_grad(set_to_none=True)
    out = fn()
    out.square().sum().backward()


def _shape(name: str, args) -> int:
    spec = SHAPES[name]
    torch.manual_seed(args.seed)
    cfg = ModelConfig(
        d_model=spec["d"],
        n_heads=4,
        head_dim=spec["d"] // 4,
        d_ffn=spec["f"],
        rank=spec["rank"],
        n_experts=spec["experts"],
        moe_topk=2,
        moe_capacity=0.0,  # DROPLESS: the route the loop wins today
        moe_kernel="padded",  # the arm names are called DIRECTLY, not through _dispatch
        bf16=True,
    )
    moe = SparseMoE(cfg, device=DEV)
    x = torch.randn(spec["tokens"], spec["d"], device=DEV)
    index, weight = _routing(moe, x, args.routing)
    order, keep, _ = moe._permute(index)
    counts = torch.bincount(index.reshape(-1), minlength=moe.n_experts)
    try:
        cap_s = str(moe._batched_cap(index, weight))
    except LoudError as e:
        cap_s = f"REFUSED ({str(e).splitlines()[0][:72]})"
    buckets = len(moe._bucket_plan(counts))
    print(
        f"\n[{name}] routing={args.routing} tokens={spec['tokens']} d={spec['d']} "
        f"f={spec['f']} E={spec['experts']} rank={spec['rank']} top2 | "
        f"rows={int(keep.sum())} max_group={int(counts.max())} cap={cap_s} "
        f"buckets={buckets} budget={MOE_BUCKET_BYTES / 2**20:.0f} MiB",
        flush=True,
    )

    arms, why = _arms(moe, x, weight, index, order, keep, args.arm)
    if why is not None:
        print(f"  triton: SKIP ({why}) — the arm is not in the table", flush=True)
    x.requires_grad_(True)
    rows = []
    for arm, fwd in arms.items():
        bank, nb = _bank_bytes(moe, arm, x, weight, index, order, keep)

        def fwd_bwd(fwd=fwd):
            _backward(moe, fwd)

        def fwd_no_grad(fwd=fwd):
            with torch.no_grad():
                return fwd().detach()

        try:
            f_ms, f_min = _timed(fwd_no_grad, args.iters, args.warm)
            b_ms, b_min = _timed(fwd_bwd, args.iters, args.warm)
            nlaunch = _launches(fwd_bwd)
        except Exception as e:  # noqa: BLE001 - an arm that cannot run is a row, not a crash
            print(f"  {arm:>8s}  FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        rows.append((arm, f_ms, b_ms, nlaunch))
        bank_s = f"{bank / 2**20:8.1f} MiB" if bank is not None else "       n/a"
        bucket_s = f" buckets={nb}" if nb is not None else ""
        print(
            f"  {arm:>8s}  fwd {f_ms:8.2f} ms (min {f_min:7.2f})   "
            f"fwd+bwd {b_ms:8.2f} ms (min {b_min:7.2f})   "
            f"launches {nlaunch if nlaunch is not None else 'n/a'}   "
            f"bank {bank_s}{bucket_s}",
            flush=True,
        )
    if len(rows) >= 2:
        best = min(rows, key=lambda r: r[2])
        worst = max(rows, key=lambda r: r[2])
        if best[2] > 0:
            print(
                f"  -> {best[0]} is {worst[2] / best[2]:.2f}x faster than {worst[0]} "
                f"on fwd+bwd at this shape",
                flush=True,
            )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=[*SHAPES, "all"], default="all")
    ap.add_argument(
        "--arm",
        choices=["all", "naive", "batched", "bucketed", "triton"],
        default="all",
        help="measure ONE arm; the default measures every arm that can run",
    )
    ap.add_argument(
        "--routing",
        choices=["uniform", "pathological"],
        default="uniform",
        help="uniform = the router's top-2; pathological = 90%% of rows on expert 0",
    )
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    why = preflight(DEV)
    if why and args.allow_busy:
        print(f"[COUNTED] measuring with a foreign process on the card: {why}")
        print("[COUNTED] numbers below are an UPPER bound on ms/step, not a clean baseline")
        why = None
    if why is not None:
        return skip(why)
    if not torch.cuda.is_available():
        return skip("torch sees no CUDA device (CUDA_VISIBLE_DEVICES?)")
    names = list(SHAPES) if args.geometry == "all" else [args.geometry]
    for name in names:
        _shape(name, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
