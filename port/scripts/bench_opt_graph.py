"""Eager vs CUDA-graph replay of the optimizer STAGE, measured on this card.

    uv run python scripts/bench_opt_graph.py                    # medium, then b1 if it fits
    uv run python scripts/bench_opt_graph.py --preset medium --steps 30
    uv run python scripts/bench_opt_graph.py --warmup 5 --reps 2

**What is measured, and why these instruments.**

* **The optimizer STAGE only** — `clip_grads` + `MixOptimizer.step()`, exactly
  the callable `train/loop.py` hands the wrapper (`opt_stage`). Two arms on the
  SAME preset and the SAME grads:
  - `eager` — `OptimizerGraph` with a warmup that never ends (`10**9`), i.e. the
    arm's own eager path, per step;
  - `replay` — `OptimizerGraph` with `--warmup` eager steps, then `graph.step()`.
  Both carry the LR as the same device tensor and are timed with the SAME loop,
  so the difference between the columns is the capture and nothing else.
* **Wall ms per step**, over `--steps` steps after `--warmup + --reps` warm
  steps, with one `torch.cuda.synchronize()` at the end of the window. The port's
  step is HOST-DISPATCH-BOUND (the card is idle 87% of a warm step, AGENTS §3.1),
  so wall time is the number that can move; a per-kernel GPU time cannot.
* **The launch count** — `torch.profiler`'s `cudaLaunchKernel` events for ONE
  step of each arm. The arm's whole claim is "one launch replaces hundreds", so
  the claim is COUNTED here, not asserted. If the field reads 0 for both arms the
  profiler did not surface launch events on this build: the wall column is then
  the only measurement and the file says so instead of printing a fake 0.

**What this file does NOT claim.** Every ms is a measurement of THIS box at THIS
commit. `docs/research/2026-10-17-olmo-core-ideas.md` §2.1 predicts the direction,
not the size, and its band is not quoted as a result.

**Preflight, and it is the whole reason the script is safe to run** (`bench_opt.py`'s
check, kept identical): one GPU process at a time (AGENTS §1.5) — no `dormouse`
process alive and the card holding < 2 GiB. On either, it prints a SKIP banner
naming what it found and exits 0: nothing was measured, so there is no number to
quote. **If it skips, do not quote a table from a busy card.**
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import time

import torch

from dormouse.config import load_run
from dormouse.model import DormouseModel
from dormouse.train.graph import OptimizerGraph
from dormouse.train.optim import MixOptimizer, clip_grads

#: The presets to try, in order. `b1` is the shape the arm exists for; it is
#: attempted and skipped LOUDLY if the card cannot hold the model + one grad set
#: + the optimizer state (the VRAM ladder, `scripts/run_b1.sh`).
PRESETS = ("medium", "b1")
LR = 3e-4


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
    # The bracket keeps `pgrep` from matching this script's own command line.
    busy = subprocess.run(
        ["pgrep", "-af", "dormouse trai[n]"], capture_output=True, text=True
    ).stdout.strip()
    if busy:
        return f"a dormouse process is alive:\n{busy}"
    if not torch.cuda.is_available():
        return "no CUDA device is visible"
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 2**30
    if used >= 2.0:
        return f"the card already holds {used:.2f} GiB (the preflight's ceiling is 2.0)"
    print(f"[preflight] no dormouse process; card holds {used:.2f} GiB", flush=True)
    return None


def launches(fn) -> int:
    """`cudaLaunchKernel` events for one call — the arm's own claim, counted."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.name == "cudaLaunchKernel")


def build(preset: str):
    """The preset's own model + optimizer, exactly as `train()` builds them."""
    run = load_run(preset)
    model = DormouseModel(run.model, device="cuda")
    opt = MixOptimizer(model, lr=LR, wd=run.train.wd, opt=run.train.opt)
    return run, model, opt


def arm(preset: str, *, warmup_steps: int, steps: int, warm: int, reps: int):
    """`(wall ms/step, launches, parameter-tensor count)` for one arm of one preset."""
    run, model, opt = build(preset)

    def stage() -> None:
        if run.train.grad_clip > 0:
            clip_grads(model.parameters(), run.train.grad_clip)
        opt.step()

    graph = OptimizerGraph(opt, stage, warmup_steps=warmup_steps)
    # Stable `.grad` buffers (the graph pins them) and ONE fixed grad set: the
    # grads do not need to vary for a timing bench, and regenerating them per
    # step would put the RNG inside the window being timed.
    gen = torch.Generator(device="cuda").manual_seed(7)
    grads = [torch.randn(p.shape, generator=gen, device="cuda") * 0.1 for p in model.parameters()]
    for p, g in zip(model.parameters(), grads, strict=True):
        p.grad = torch.zeros_like(p)
        p.grad.copy_(g)

    def one() -> None:
        graph.set_lr(LR)
        for p, g in zip(model.parameters(), grads, strict=True):
            p.grad.copy_(g)
        graph.step()

    for _ in range(warm):
        one()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(steps):
            one()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3 / steps)
    if warmup_steps < 10**9:
        assert graph._graph is not None, "the replay arm never captured — nothing was measured"
    n = launches(one)
    ntensors = len(grads)
    torch.cuda.empty_cache()  # `arm`'s locals die here; the next preset needs the room
    return statistics.median(times), n, ntensors


def bench(preset: str, args: argparse.Namespace) -> bool:
    """One preset's table. `False` when the preset did not fit the card."""
    print(f"\n--- {preset} (steps={args.steps} warm={args.warm} reps={args.reps}) ---", flush=True)
    try:
        eager = arm(preset, warmup_steps=10**9, steps=args.steps, warm=args.warm, reps=args.reps)
        replay = arm(
            preset, warmup_steps=args.warmup, steps=args.steps, warm=args.warm, reps=args.reps
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"  SKIP {preset}: does not fit this card ({e}). Nothing was measured.")
        torch.cuda.empty_cache()
        return False
    ms_e, n_e, ntensors = eager
    ms_r, n_r, _ = replay
    print(f"{'arm':>7s} {'wall ms/step':>13s} {'launches/step':>14s}")
    print(f"{'eager':>7s} {ms_e:13.1f} {n_e:14d}")
    print(f"{'replay':>7s} {ms_r:13.1f} {n_r:14d}")
    print(
        f"  speedup {ms_e / ms_r:.2f}x on {ntensors} parameter tensors; "
        f"launches {n_e} -> {n_r} ({n_e / max(n_r, 1):.0f}x fewer)"
    )
    if n_e == 0 and n_r == 0:
        print("  NOTE: the profiler surfaced no cudaLaunchKernel events on this build —")
        print("  the launch column is unusable here and the wall column is the measurement.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", action="append", default=None, help="repeatable; default medium+b1")
    ap.add_argument("--steps", type=int, default=20, help="timed steps per rep")
    ap.add_argument("--warm", type=int, default=3, help="untimed warm steps per rep")
    ap.add_argument("--reps", type=int, default=3, help="rep count (the median is reported)")
    ap.add_argument("--warmup", type=int, default=5, help="eager steps before the capture")
    args = ap.parse_args()

    why = preflight()
    if why is not None:
        return skip(why)
    for preset in args.preset or PRESETS:
        bench(preset, args)
    print("\nThe A/B verdict is NOT here: this table prices the stage, it does not judge a run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
