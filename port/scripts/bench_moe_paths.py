"""Time the routed expert pass: `naive` / `triton` / `batched`, one shape at a time.

    uv run --extra gpu python scripts/bench_moe_paths.py --geometry m400

The 2026-11-04 finding: at 8192 tokens, E=512, d=1536, f=4096, rank 64, top-2 the
per-expert Python loop is ~99.6 ms of a ~107 ms iteration and the per-expert
bank (`torch.bmm` over `[E, cap, ·]`) is ~11.5 ms — the same arithmetic, 8.7x.
This script measures the three dispatch arms on the REAL module at that shape
and at the b8-s1024 / b1-s8192 presets, forward and forward+backward, with the
kernel launch count of ONE forward+backward pass. The owner runs it on the card;
without a free card it prints the preflight reason and exits 0 (nothing measured).

Shapes: the MoE sees `[batch*seq, d]`, so b8 x s1024 and b1 x s8192 both route
8192 tokens; the difference is the model geometry each preset names, not the MoE
input length. `m400` is the measured 93%-shape.

The arms share ONE routing (the router's top-k on the same `x`), the same
weights and the same rows, so the table compares implementations, not samples.
`moe_capacity=0` (dropless) is the config that needs the fix; `padded` refuses
there, `naive` and `triton(dom/le)` are the current paths, `batched` is the new
one. `triton` is reported as the reason it cannot run when the build/card cannot
serve it; it is never silently omitted.
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
from dormouse.instrument import Counters  # noqa: E402
from dormouse.model.experts import SparseMoE  # noqa: E402

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


def _arms(moe: SparseMoE, x, weight, index, order, keep):
    """The three dispatch arms as closures, with the reason an arm cannot run."""

    def naive():
        return moe._naive_dispatch(x, weight, index, order, keep, Counters())

    def batched():
        return moe._batched_dispatch(x, weight, index, order, keep)

    def triton():
        return moe._triton_dispatch(x, weight, index, order, keep, dropless=True)

    arms = {"naive": naive, "batched": batched}
    why = None
    if mk is None:
        why = "no triton import"
    elif not torch.cuda.is_available():
        why = "no CUDA device"
    if why is None:
        arms["triton"] = triton
    return arms, why


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
    with torch.no_grad():
        probs = F.softmax(moe.router(x), dim=-1)
        weight, index = probs.topk(2, dim=-1)
        weight = weight / weight.sum(-1, keepdim=True)
    order, keep, _ = moe._permute(index)
    counts = torch.bincount(index.reshape(-1), minlength=moe.n_experts)
    cap = moe._batched_cap(index, weight)
    print(
        f"\n[{name}] tokens={spec['tokens']} d={spec['d']} f={spec['f']} "
        f"E={spec['experts']} rank={spec['rank']} top2 | rows={int(keep.sum())} "
        f"max_group={int(counts.max())} cap={cap}",
        flush=True,
    )

    arms, why = _arms(moe, x, weight, index, order, keep)
    if why is not None:
        print(f"  triton: SKIP ({why}) — the arm is not in the table", flush=True)
    x.requires_grad_(True)
    rows = []
    for arm, fwd in arms.items():

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
        print(
            f"  {arm:>8s}  fwd {f_ms:8.2f} ms (min {f_min:7.2f})   "
            f"fwd+bwd {b_ms:8.2f} ms (min {b_min:7.2f})   "
            f"launches {nlaunch if nlaunch is not None else 'n/a'}",
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
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    why = preflight(DEV)
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
