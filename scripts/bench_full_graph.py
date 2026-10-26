"""Eager vs `opt_graph` vs `full_graph`: ms/step, bytes/s and kernel launches.

    uv run --extra gpu python scripts/bench_full_graph.py --shape b8s512
    uv run --extra gpu python scripts/bench_full_graph.py --shape b24s1024 --reps 3

**What is measured.** The SAME model and the SAME data for all three arms; the
only difference is what is captured:

* `eager` — `full_graph=false, opt_graph=false`: the pre-arm step.
* `opt_graph` — the optimizer stage (grad clip + mixed step) in one graph, the
  LR a device tensor (`train/graph.py::OptimizerGraph`).
* `full_graph` — forward + backward + firewall + `zero_grad` + the optimizer
  stage in one graph.

**Instruments, and what each can and cannot see.** Wall ms/step (`min of
--reps`, each rep `--steps` after `--warm` warm steps, one `synchronize()` per
rep) is the number that can move: the step is HOST-DISPATCH-bound (the card is
idle 87% of a warm step, AGENTS §3.1), so a per-kernel GPU time would not move.
`cudaLaunchKernel` events from `torch.profiler` are the arm's own claim, counted
over ONE step. If the profiler surfaces no launch events on this build the
column is unusable and the file says so rather than printing a fake 0.

**Preflight** (`_lane.py`, the same check every bench here makes): one GPU
process at a time, and it prints SKIP + the reason and exits 0 when the card is
busy — nothing measured, so there is no number to quote.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

from dormouse.config import load_run
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train import train
from dormouse.train.data import ByteStream

sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight, skip  # noqa: E402

SHAPES = {
    "b8s512": (8, 512),
    "b24s1024": (24, 1024),
}
LR = 3e-4

#: `--engram-ram` before any arm is built. OFF by default: it is the one arm
#: `full_graph` REFUSES (the host table resolves rows with a D2H inside the
#: forward), so the default substrate is the one both arms can run on. A table
#: taken with `--engram-ram true` is a different substrate and says so.
ENGRAM_RAM = False


class OneStepLaunches:
    """`cudaLaunchKernel` count for EXACTLY ONE step, not averaged over warmups.

    The trainer logs at the END of a step, so the log callback is a step
    boundary: entering the profiler on step `k`'s line and leaving on `k+1`'s
    wraps one whole step and nothing else. `warm`/`steps` are then irrelevant to
    the count — which the previous `events / (steps + warm)` could not say, and
    that average is exactly what made a replay invisible (`full_graph` replays
    ONE launch per step; averaging it over 25 steps reported thousands).
    """

    def __init__(self, step: int) -> None:
        self.step = step
        self.seen = 0
        self.count: int | None = None
        self._prof = None

    def __call__(self, line: str) -> None:
        if not line.startswith("step="):
            return
        self.seen += 1
        from torch.profiler import ProfilerActivity, profile

        if self.seen == self.step:
            self._prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            self._prof.__enter__()
        elif self.seen == self.step + 1 and self._prof is not None:
            self._prof.__exit__(None, None, None)
            torch.cuda.synchronize()
            self.count = sum(1 for e in self._prof.events() if e.name == "cudaLaunchKernel")
            self._prof = None


def arm(shape: str, mode: str, *, steps: int, warm: int, reps: int, data: Path):
    """One arm: `(min ms/step, launches/step, mode actually run, reason)`."""
    batch, seq = SHAPES[shape]
    over = dict(
        batch=batch,
        seq_len=seq,
        steps=steps + warm,
        log_every=10**9,
        eval=False,
        ckpt_every=10**9,
        lr=LR,
        muon_state_dtype="bf16",
        ns_dtype="bf16",
        grad_clip=1.0,
        # The PRODUCTION substrate (`final_byte` at b24 s1024 does not fit the
        # card without it), and the config `full_graph` refuses: its host rows
        # are a D2H inside the forward. So the full_graph arm is a LOUD refusal
        # at this shape, which is the arm's honest answer, not a table cell.
        engram_ram=ENGRAM_RAM,
    )
    over["full_graph"] = mode == "full_graph"
    over["opt_graph"] = mode == "opt_graph"
    run = load_run("final_byte", train_overrides=over, model_overrides=dict(bf16=True))

    # ONE `train()` CALL PER MODEL. `engram_ram` is destructive: the first call
    # switches the Engram to host tables and DELETES `engram.tables`, so a
    # second call on the same object refuses ("no Engram to host") — measured
    # here, and the reason every rep builds its own model.
    def once(*, timed: bool, launches_at: int = 0):
        torch.manual_seed(0)
        model = DormouseModel(run.model, device="cuda")
        stream = ByteStream(data, batch, seq)
        hook = OneStepLaunches(launches_at) if launches_at else None
        # The hook rides the trainer's OWN log cadence, so the profiled run logs
        # every step and the timed runs log none (a log step reads the loss back
        # to the host — that D2H must not land in a ms/step number).
        run.train.log_every = 1 if launches_at else 10**9
        t0 = time.perf_counter()
        rep = train(
            model,
            run,
            stream=stream,
            device="cuda",
            log=(hook if hook is not None else (lambda *a, **k: None)),
        )
        if timed:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        del model, stream
        # The graph's pool is mapped for the GRAPH's life and the graph lives in
        # `train()`'s frame: a rep that starts before the garbage collector has
        # dropped it sees the previous pool still reserved and REFUSES the
        # capture on headroom (measured: rep 1 captures with ~2 GiB free, rep 2
        # refuses at 140 MiB). Collect FIRST, then release the cached blocks.
        gc.collect()
        torch.cuda.empty_cache()
        return seconds, rep, (hook.count if hook is not None else None)

    # ONE step's launch count, profiled in a separate (untimed) run at a step
    # that is a REPLAY for the graph arms (`warm` = the arm's own warmup, so
    # step 10 is past the capture at step 6).
    n = once(timed=False, launches_at=steps // 2 + warm)[2]
    times: list[float] = []
    rep = None
    for _ in range(reps):
        torch.cuda.synchronize()
        seconds, rep, _ = once(timed=True)
        times.append(seconds * 1e3 / (steps + warm))
    ms = min(times)
    return ms, n, rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shape", choices=sorted(SHAPES), default="b8s512")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--modes", default="eager,opt_graph,full_graph")
    ap.add_argument("--engram-ram", default="false", choices=("true", "false"))
    ap.add_argument("--reps-alias", help=argparse.SUPPRESS)
    args = ap.parse_args()
    global ENGRAM_RAM
    ENGRAM_RAM = args.engram_ram == "true"

    why = preflight("cuda")
    if why is not None:
        return skip(why)
    data = Path("/tmp/opencode/bench_full_graph_bytes.bin")
    if not data.is_file():
        data.write_bytes(bytes(range(256)) * 240_000)  # 61 MB: 24*1025*20 steps fits
    print(
        f"\n--- {args.shape} (steps={args.steps} reps={args.reps} engram_ram={ENGRAM_RAM}) ---",
        flush=True,
    )
    print(f"{'arm':>11s} {'ms/step':>9s} {'bytes/s':>12s} {'launches/step':>14s}  note")
    batch, seq = SHAPES[args.shape]
    bytes_per_step = batch * seq
    for mode in args.modes.split(","):
        ms, n, rep = None, None, None
        try:
            ms, n, rep = arm(
                args.shape, mode, steps=args.steps, warm=args.warm, reps=args.reps, data=data
            )
        except LoudError as e:
            print(f"{mode:>11s} {'refused':>9s} {'-':>12s} {'-':>14s}  LOUD: {str(e)[:80]}")
            torch.cuda.empty_cache()
            continue
        except torch.cuda.OutOfMemoryError:
            print(f"{mode:>11s} {'OOM':>9s} {'-':>12s} {'-':>14s}  does not fit this card")
            torch.cuda.empty_cache()
            continue
        note = ""
        counters = rep["counters"] if rep else {}
        if counters.get("opt_graph_fallback"):
            note = "COUNTED fallback -> eager (" + str(getattr(counters, "why", "?"))[:70] + ")"
        if counters.get("opt_graph_frozen"):
            note += f" frozen={counters['opt_graph_frozen']}"
        rate = bytes_per_step / (ms / 1e3)
        print(f"{mode:>11s} {ms:9.1f} {rate:12,.0f} {n:14d}  {note}", flush=True)
    print("\nThe A/B verdict is NOT here: this table prices the modes, it does not judge a run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
