#!/usr/bin/env -S uv run --python /home/sehaxe/dormouse/port/.venv/bin/python --script
"""Eager vs `full_graph` on the LIVE m400b shape: ms/step, launches, B/s.

PROTOCOL
--------
Each arm is `--reps` independent processes-in-one (a fresh model per rep),
`--steps` timed steps after `--warm` warmup steps, one `synchronize()` at the
end; the table prints the MINIMUM ms/step. The minimum is the number that can
move: the step is host-dispatch-bound (the card is idle most of a warm step,
`AGENTS` §3.1), so per-kernel GPU time would not.

Launches are counted over EXACTLY ONE step (the existing
`bench_full_graph.OneStepLaunches`), at `warm + steps//2` — past the capture
(step 6) and therefore a replay for the graph arm.

The recipe is the CHECKPOINT's own snapshot when it exists
(`checkpoints/m400b/m400b.config.toml`), so the bench measures the run the card
is training, not a copy that can drift. Without the snapshot the tracked
`configs/m400b.toml` preset is used (model half) plus the train keys in
`M400B_TRAIN`.

    # during a checkpoint gap, ONE GPU process (AGENTS 1.5)
    cd /home/sehaxe/dormouse/port
    uv run --extra gpu python scripts/bench_graph.py --reps 5 --steps 20 --parity 20

`--parity N` adds the control the arm's own tests use: the SAME `full_graph`
arm with `FullGraph.capture` disabled, N replays, compared parameter-by-
parameter. The first replay is gated at 1e-6 (fp32 ulp); the 20-step drift is
REPORTED, not gated, because mHC's near-zero gains make Muon normalise float
chaos (measured ~1e-1, `.bulba/memory.md` 2026-10-24). `within_1e-6` is printed
so the answer is explicit.

A busy card is a SKIP (exit 0, no table): nothing was measured, so there is no
number to quote.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import sys
import time
from pathlib import Path

import torch

from dormouse.config import RunConfig, load_run
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.train import train
from dormouse.train.data import ByteStream
from dormouse.train.graph import FullGraph

sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight, skip  # noqa: E402
from bench_full_graph import OneStepLaunches  # noqa: E402

DEFAULT_SNAPSHOT = Path("/home/sehaxe/dormouse/checkpoints/m400b/m400b.config.toml")
DEFAULT_DATA = Path("/tmp/opencode/bench_graph_bytes.bin")

#: The snapshot's train half, reduced to the keys a timing table moves. The
#: model half comes from `configs/m400b.toml`; the live snapshot overrides both
#: when it exists.
M400B_TRAIN = dict(
    batch=8,
    seq_len=1024,
    lr=5e-4,
    wd=0.01,
    grad_clip=1.0,
    steps=22,
    eval=False,
    eval_every=10**9,
    ckpt_every=10**9,
    log_every=10**9,
    json_log="",
    muon_state_dtype="fp32",
    ns_dtype="fp32",
    muon_chunk_elems=1 << 26,
    mix_plan="",
    data="",
    eval_data="",
)


def _run_config(args) -> RunConfig:
    snap = Path(args.snapshot)
    if snap.is_file():
        run = RunConfig.from_snapshot(snap.read_text(encoding="utf-8"))
    else:
        run = load_run(args.config, train_overrides=M400B_TRAIN)
    over = dict(M400B_TRAIN)
    over.pop("steps", None)
    return dataclasses.replace(run, train=dataclasses.replace(run.train, **over))


def _free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)  # noqa: SLF001
    if clear is not None:
        clear()
        torch.cuda.empty_cache()


def _arm(
    run: RunConfig,
    mode: str,
    *,
    steps: int,
    warm: int,
    reps: int,
    data: Path,
):
    """`(min ms/step, launches/step, report)` for one arm.

    `mode` is `eager` (`full_graph=false`) or `graph` (`full_graph=true`).
    `_parity` uses `capture=False` to keep the arm's device-LR / AdamW-
    capturable arithmetic while disabling only the recording — the arm's OWN
    eager program, which is the only reference a 1e-6 bound can compare
    against.
    """
    if mode not in ("eager", "graph"):
        raise ValueError(f"unknown mode {mode!r}")
    train_cfg = dataclasses.replace(
        run.train,
        full_graph=mode == "graph",
        opt_graph=False,
        torch_compile=False,
    )

    def once(*, timed: bool, launches_at: int = 0):
        torch.manual_seed(0)
        model = DormouseModel(run.model, device="cuda")
        stream = ByteStream(data, train_cfg.batch, train_cfg.seq_len)
        hook = OneStepLaunches(launches_at) if launches_at else None
        cfg = dataclasses.replace(
            train_cfg,
            steps=steps + warm,
            log_every=1 if launches_at else 10**9,
        )
        t0 = time.perf_counter()
        report = train(
            model,
            dataclasses.replace(run, train=cfg),
            stream=stream,
            device="cuda",
            log=(hook if hook is not None else (lambda *a, **k: None)),
        )
        if timed:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        del model, stream
        _free_gpu()
        return seconds, report, (hook.count if hook is not None else None)

    n = once(timed=False, launches_at=warm + steps // 2)[2]
    times: list[float] = []
    report = None
    for _ in range(reps):
        torch.cuda.synchronize()
        seconds, report, _n = once(timed=True)
        times.append(seconds * 1e3 / (steps + warm))
    return min(times), n, report


def _parity(run: RunConfig, *, steps: int, data: Path) -> tuple[float, float]:
    """`(first-replay-equivalent worst delta, 20-step worst delta)`, parameter-wise.

    The first-replay bound needs a 6-step run (5 eager + one capture + one
    replay); the long run is the requested `steps`. Both compare the graph arm
    with its own capture-disabled twin, same seed and data.
    """
    results = []
    # `steps` is a REPLAY count: the arm needs 5 eager warmup steps before the
    # capture, so the long run is `5 + steps` to give exactly that many replays.
    for label, n_steps in ((6, 6), (steps, 5 + steps)):
        pair = []
        for capture in (True, False):
            torch.manual_seed(0)
            model = DormouseModel(run.model, device="cuda")
            cfg = dataclasses.replace(
                run.train,
                full_graph=True,
                opt_graph=False,
                torch_compile=False,
                steps=n_steps,
                log_every=10**9,
            )
            original = None
            if not capture:
                original = FullGraph.capture
                FullGraph.capture = lambda self: (_ for _ in ()).throw(  # type: ignore[method-assign]
                    RuntimeError("parity: capture disabled")
                )
            try:
                report = train(
                    model,
                    dataclasses.replace(run, train=cfg),
                    stream=ByteStream(data, cfg.batch, cfg.seq_len),
                    device="cuda",
                    log=lambda *_a, **_k: None,
                )
            finally:
                if original is not None:
                    FullGraph.capture = original
            if capture:
                assert report["counters"].get("full_graph_captured") == 1, (
                    f"the graph arm did not capture: {dict(report['counters'])}"
                )
            sd = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
            del model
            _free_gpu()
            pair.append((report, sd))
        worst = max(((pair[0][1][k] - pair[1][1][k]).abs().max().item(), k) for k in pair[0][1])
        results.append(worst)
        print(
            f"[parity {label:2d} step(s)] worst |delta| = {worst[0]:.3e} @ {worst[1]}"
        )
    return results[0][0], results[1][0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/m400b.toml")
    ap.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--modes", default="eager,graph")
    ap.add_argument(
        "--parity",
        type=int,
        default=0,
        metavar="STEPS",
        help="also run graph vs the same arm with capture disabled; 0 is off",
    )
    args = ap.parse_args(argv)

    why = preflight("cuda")
    if why is not None:
        return skip(why)
    if args.steps < 1 or args.reps < 1:
        print("REFUSED: --steps and --reps must be >= 1", file=sys.stderr)
        return 2

    run = _run_config(args)
    data = Path(args.data)
    if not data.is_file():
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_bytes(bytes(range(256)) * 240_000)  # 61 MB covers all reps

    print(
        f"\n--- m400b b{run.train.batch} x s{run.train.seq_len} · "
        f"steps={args.steps} warm={args.warm} reps={args.reps} · "
        f"{'snapshot' if Path(args.snapshot).is_file() else args.config} ---",
        flush=True,
    )
    print(f"{'arm':>7s} {'ms/step':>9s} {'B/s':>12s} {'launches/step':>14s}  note")
    bytes_per_step = run.train.batch * run.train.seq_len
    for mode in args.modes.split(","):
        try:
            ms, launches, report = _arm(
                run,
                mode,
                steps=args.steps,
                warm=args.warm,
                reps=args.reps,
                data=data,
            )
        except LoudError as e:
            print(f"{mode:>7s} {'refused':>9s} {'-':>12s} {'-':>14s}  LOUD: {str(e)[:90]}")
            _free_gpu()
            continue
        except torch.cuda.OutOfMemoryError:
            print(f"{mode:>7s} {'OOM':>9s} {'-':>12s} {'-':>14s}  does not fit this card")
            _free_gpu()
            continue
        counters = report["counters"] if report else {}
        note = ""
        if counters.get("full_graph_captured"):
            note = f"captured={counters['full_graph_captured']}"
        if counters.get("full_graph_fallback"):
            note = "COUNTED fallback -> eager (" + str(getattr(counters, "why", "?"))[:70] + ")"
        rate = bytes_per_step / (ms / 1e3)
        shown = launches if launches is not None else -1
        print(f"{mode:>7s} {ms:9.1f} {rate:12,.0f} {shown:14d}  {note}", flush=True)

    if args.parity > 0:
        first, long = _parity(run, steps=args.parity, data=data)
        print(
            f"\nparity: first replay {first:.3e} (bound 1e-6: "
            f"{'PASS' if first <= 1e-6 else 'FAIL'}); {args.parity}-step drift "
            f"{long:.3e} (reported, not a bound — mHC/Muon float chaos); "
            f"within_1e-6={long <= 1e-6}"
        )
    print("\nThe A/B verdict is NOT here: this table prices the modes, it does not judge a run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
