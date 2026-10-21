"""The two optimizer dtype rungs, measured: the 4-combo micro-bench and the A/B.

    uv run python scripts/bench_opt.py micro
    uv run python scripts/bench_opt.py ab --state bf16
    uv run python scripts/bench_opt.py ab --ns bf16 --out-dir ~/logs/ab_opt

**What is measured, and why these two instruments.**

* `micro` — `Muon.batched_step_from_model` at the b1 preset's OWN Muon shape
  groups (read off the model on `meta`, so a preset change moves the bench with
  it), for the 4 combos of `muon_state_dtype` x `ns_dtype`: median ms over
  `--reps` steps after `--warm` warm steps, plus
  `torch.cuda.max_memory_allocated` and the state's own bytes. The state bytes
  are arithmetic a reader can check by hand, and they are MEASURED here rather
  than asserted: the b1 Muon groups hold 660.8M elements — `[64, 5461] x 1024`
  (the expert bank factors) is 357.9M of them, **1.333 GiB fp32** — so the fp32
  state is **2.4616 GiB** and the `bf16` rung **1.2308 GiB**: a **1.2308 GiB**
  saving at b1. `max_memory_allocated` is the instrument for everything else
  (the NS chain's own temporaries, which the `bf16` NS rung should shrink too).
* `ab` — the training A/B for the ONE combo that wins `micro`, on the repo's
  protocol (`docs/protocols/AB-PROTOCOL.md`, cost note §4 Stage 1): **3 seeds
  per arm, 2 000 steps, `small`, b8 x s512, lr 1e-3, and one eval window of
  81 920 B on every arm and seed** (`eval_batches 20 x batch 8 x seq_len 512`).
  Control = `fp32`/`fp32` (the default, and every number on record); the arm is
  the combo you name, and naming the control is refused (a null A/B). Each
  arm-seed is a SEPARATE PROCESS, because 3 seeds is 3 initialisations. The
  driver prints the verdict table (per-run BPB, warm tokens/s, NaN-skipped
  steps) and the void checks, then states the pre-registered rule — it does not
  apply it: a BPB is a measurement, not a verdict.

**Preflight, and it is the whole reason both modes are safe to run:** one GPU
process at a time (AGENTS §1.5) — no `dormouse` process alive and the card
holding < 2 GiB. On either, the mode prints a SKIP banner naming what it found
and exits 0: nothing was measured, so there is no number to quote. **If it
skips, the numbers are not taken — do not quote a table from a busy card.**

**Owed wiring, named here because this driver works around it:** `train/loop.py`'s
`MixOptimizer(...)` call site does not forward `muon_state_dtype` / `ns_dtype`,
so `--set muon_state_dtype=bf16` would be INERT in a plain `dormouse train` run
(the `SILENT` class, AGENTS §1.1). This driver therefore injects the two kwargs
through a `MixOptimizer` spy — the idiom
`tests/test_muon_fast.py::test_the_trainer_forwards_the_colrow_arm` uses — and
ASSERTS afterwards that the built optimizer carries the rung, so an inert arm
cannot produce a table. Once the call site forwards the keys the injection is a
no-op (`setdefault`) and the banner stops printing.

**What must be re-measured on the card, because this file prices nothing:** the
ms column and the A/B. The cost note's bands, quoted as PREDICTIONS:
`ns_dtype="bf16"` **−100…−200 ms/step** (row 4; the 0.22 bf16 tensor-core rate
is 44.06 vs 20.03 TFLOP/s fp32, AGENTS §2.1). The state rung has NO published ms
prediction — it is a VRAM lever first (1.2308 GiB at b1) and its speed is
whatever `micro` says, which is why the instrument reports both.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight, skip  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

#: The port's data on this box (AGENTS §2.6) — the LOCAL copy (the /mnt drive
#: may not be mounted). The eval reads the no-leak tail split of the SAME file
#: (`data.open_streams`), so the window is `eval_batches x batch x seq_len`.
DATA = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"

#: The protocol constants (cost note §4 Stage 1 / AB-PROTOCOL §1.2).
SEEDS = (1, 2, 3)
STEPS = 2000
BATCH = 8
SEQ = 512
LR = 1e-3
EVAL_BATCHES = 20
EVAL_EVERY = 200
LOG_EVERY = 50
COMBOS = (("fp32", "fp32"), ("bf16", "fp32"), ("fp32", "bf16"), ("bf16", "bf16"))


# --- micro: the 4 combos at the b1 shapes -----------------------------------


def micro(reps: int = 5, warm: int = 2) -> int:
    """Median ms + peak VRAM + state bytes for the 4 combos, at the b1 groups."""
    from test_muon_fast import _b1_params, b1_muon_shapes

    from dormouse.train.optim import Muon

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shapes = b1_muon_shapes()
    elems = sum(c * s[0] * s[1] for s, c in shapes.items())
    print(
        f"b1 Muon groups: {len(shapes)} shapes, {sum(shapes.values())} params, "
        f"{elems / 1e6:.1f}M elements -> state fp32 {elems * 4 / 2**30:.4f} GiB / "
        f"bf16 {elems * 2 / 2**30:.4f} GiB"
    )
    for s, c in sorted(shapes.items(), key=lambda kv: -kv[1] * kv[0][0] * kv[0][1])[:4]:
        print(f"  {str(s):>14s} x{c:<5d} {c * s[0] * s[1] * 4 / 2**20:9.1f} MiB fp32")
    params = _b1_params(shapes, device)
    init = [p.detach().clone() for p in params]
    print(
        f"\n{'state':>5s} {'ns':>5s} {'median ms':>10s} {'min ms':>8s} "
        f"{'peak GiB':>9s} {'state B':>12s}"
    )
    rows = []
    for state, ns in COMBOS:
        for i, p in enumerate(params):
            p.detach().copy_(init[i])
        if device == "cuda":
            torch.cuda.empty_cache()
        muon = Muon(params, lr=1e-3, state_dtype=state, ns_dtype=ns)

        def run(_m=muon) -> None:
            _m.batched_step_from_model(None)

        for _ in range(warm):
            run()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            run()
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
        state_bytes = sum(g.mom.numel() * g.mom.element_size() for g in muon._groups)
        assert muon._groups, "the step built no groups — nothing was measured"
        rows.append((state, ns, statistics.median(times), min(times), peak, state_bytes))
        print(
            f"{state:>5s} {ns:>5s} {rows[-1][2]:10.1f} {rows[-1][3]:8.1f} "
            f"{peak:9.3f} {state_bytes:12,d}"
        )
        del muon
        if device == "cuda":
            torch.cuda.empty_cache()
    if device == "cpu":
        print("\nCPU numbers are a shape check, NOT the measurement: run it on the card.")
    best = min(rows, key=lambda r: r[2])
    print(
        f"\nfastest combo: state={best[0]} ns={best[1]} at {best[2]:.1f} ms "
        f"({best[4]:.3f} GiB peak). A/B it: python scripts/bench_opt.py ab "
        f"--state {best[0]} --ns {best[1]}"
    )
    return 0


# --- ab: the training A/B driver --------------------------------------------


def _arm_name(state: str, ns: str) -> str:
    return f"{state}_{ns}"


def _run_one(args: argparse.Namespace) -> int:
    """One arm, one seed, one process: the trainer, with the rung made live.

    The injection is the OWED WIRING workaround in the module docstring; the
    assertion after `train()` is what makes an inert flag impossible to report.
    """
    import dormouse.train.loop as train_loop
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train import train

    run = load_run(
        "small",
        train_overrides={
            "batch": args.batch,
            "seq_len": args.seq,
            "lr": LR,
            "steps": args.steps,
            "seed": args.seed,
            "eval": True,
            "eval_every": args.eval_every,
            "eval_batches": args.eval_batches,
            "log_every": args.log_every,
            "ckpt_every": 10**9,  # the A/B scores held-out BPB; no weights are written
            "out_dir": str(Path(args.out).parent),
            "ckpt_name": f"{_arm_name(args.state, args.ns)}_s{args.seed}",
            # `timers` is OFF on purpose: `loop.py:666-679` calls
            # `timer.report()` (which RESETS the mark) before the telemetry row
            # reads `timer.window_s()`, so with `--timers` on the JSONL's
            # `tokens_per_s` is the time since that reset (~1e7-1e8 tok/s on
            # any shape). The JSONL rate is this driver's speed instrument, so
            # the flag stays off and the defect is reported, not worked around
            # silently.
            "timers": False,
            "muon_state_dtype": args.state,
            "ns_dtype": args.ns,
            "json_log": str(args.out),
        },
    )
    run.train.data = args.data
    if not Path(run.train.data).is_file():
        print(f"no data file at {run.train.data}: pass --data")
        return 1
    # The run's own config record, next to its JSONL: a number needs its config
    # (AGENTS §1.4), and the JSONL's `run` row does not carry the dtype rungs.
    Path(args.out).with_suffix(".config.toml").write_text(run.snapshot_toml())
    real = train_loop.MixOptimizer
    built: list = []

    def spy(model, **kw):
        if not {"muon_state_dtype", "ns_dtype"} & set(kw):
            print(
                "WIRING GAP: train/loop.py's MixOptimizer call site forwards neither "
                "muon_state_dtype nor ns_dtype, so `--set` alone is INERT. This driver "
                "injects them (see the module docstring); land the two kwargs in the "
                "call site to make the flag the only seam."
            )
        kw.setdefault("muon_state_dtype", args.state)
        kw.setdefault("ns_dtype", args.ns)
        opt = real(model, **kw)
        built.append(opt)
        return opt

    train_loop.MixOptimizer = spy
    device = torch.device(args.device)
    model = DormouseModel(run.model, device=device)
    print(
        f"[run] arm={_arm_name(args.state, args.ns)} seed={args.seed} "
        f"steps={args.steps} b{args.batch} s{args.seq} lr={LR} "
        f"window={args.eval_batches * args.batch * args.seq} B",
        flush=True,
    )
    train(model, run, device=device, log=print)
    assert built and built[0].muon is not None, "no Muon group was built"
    assert built[0].muon.state_dtype == args.state, "the state rung did not reach Muon"
    assert built[0].muon.ns_dtype == args.ns, "the NS rung did not reach Muon"
    return 0


def _read(path: Path, steps: int) -> dict:
    """One run's JSONL -> the numbers the verdict table needs.

    `tok/s` is the median of the JSONL's own window rates over the run's SECOND
    HALF — the protocol's "≥50 warm steps" read as "past the autotune/warmup
    window", and on the real 2 000-step arm that is 950 steps of evidence
    (19 windows of 50, one of which contains an eval; the median absorbs it).
    The instrument is the JSONL row and not the printed `ms/step` line: see the
    `timers=False` note in `_run_one`.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    evals = [r for r in rows if r.get("event") == "eval"]
    logs = [r for r in rows if r.get("event") == "log"]
    warm = [
        r["tokens_per_s"] for r in logs if r.get("step", 0) > steps / 2 and r.get("tokens_per_s")
    ]
    assert evals, f"{path}: no eval row — nothing to compare"
    return {
        "bpb": evals[-1]["bpb"],
        "scored": evals[-1]["scored"],
        "tok_s": statistics.median(warm) if warm else float("nan"),
        "nan": sum(r.get("nan_skipped", 0) for r in logs),
        "loss": evals[-1]["loss"],
    }


def ab(args: argparse.Namespace) -> int:
    """The driver: `--seeds` x 2 arms, separate processes, then the table."""
    if (args.state, args.ns) == ("fp32", "fp32"):
        print("the named combo IS the control (fp32/fp32): a null A/B. Name a rung.")
        return 1
    seeds = tuple(int(s) for s in args.seeds.split(","))
    window = args.eval_batches * args.batch * args.seq
    arms = {"control": ("fp32", "fp32"), "arm": (args.state, args.ns)}
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"A/B: control fp32/fp32 vs arm {_arm_name(args.state, args.ns)} — "
        f"{len(seeds)} seeds x {args.steps} steps, small b{args.batch} s{args.seq} "
        f"lr={LR}, window {window} B. Output: {out_dir}"
    )
    for arm, (state, ns) in arms.items():
        for seed in seeds:
            out = out_dir / f"{arm}_{_arm_name(state, ns)}_s{seed}.jsonl"
            out.unlink(missing_ok=True)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_run",
                "--state",
                state,
                "--ns",
                ns,
                "--seed",
                str(seed),
                "--out",
                str(out),
                "--data",
                args.data,
                "--device",
                args.device,
                "--steps",
                str(args.steps),
                "--batch",
                str(args.batch),
                "--seq",
                str(args.seq),
                "--eval-batches",
                str(args.eval_batches),
                "--eval-every",
                str(args.eval_every),
                "--log-every",
                str(args.log_every),
            ]
            print(f"\n=== {arm} {state}/{ns} seed {seed} -> {out}", flush=True)
            done = subprocess.run(cmd, check=False)
            if done.returncode != 0:
                print(f"the {arm} run (seed {seed}) exited {done.returncode}: no table.")
                return done.returncode

    table = {arm: [] for arm in arms}
    for arm, (state, ns) in arms.items():
        for seed in seeds:
            path = out_dir / f"{arm}_{_arm_name(state, ns)}_s{seed}.jsonl"
            table[arm].append({"seed": seed, **_read(path, args.steps)})
    print(
        f"\n{'arm':>8s} {'seed':>4s} {'bpb':>8s} {'loss':>8s} {'scored B':>9s} "
        f"{'warm tok/s':>10s} {'nan':>4s}"
    )
    for arm, runs in table.items():
        for r in runs:
            print(
                f"{arm:>8s} {r['seed']:>4d} {r['bpb']:8.4f} {r['loss']:8.4f} "
                f"{r['scored']:>9d} {r['tok_s']:10.0f} {r['nan']:>4d}"
            )
    summary = {}
    for arm, runs in table.items():
        bpbs = [r["bpb"] for r in runs]
        summary[arm] = (
            statistics.mean(bpbs),
            max(bpbs) - min(bpbs),
            statistics.median([r["tok_s"] for r in runs]),
        )
        print(
            f"{arm:>8s} mean bpb {summary[arm][0]:.4f}  spread {summary[arm][1]:.4f}  "
            f"median warm tok/s {summary[arm][2]:.0f}"
        )
    # The void checks: a number that fails one of these is not a number.
    for arm, runs in table.items():
        for r in runs:
            assert r["scored"] == window, (
                f"{arm} seed {r['seed']}: window {r['scored']} != {window}"
            )
            assert r["nan"] == 0, f"{arm} seed {r['seed']}: {r['nan']} NaN-skipped steps"
    print(
        f"\nvoid checks: every eval scored {window} B, 0 NaN-skipped steps, "
        f"{len(seeds)} distinct seeds per arm.\npre-registered rule (AGENTS §1.2 / cost "
        f"note §4): keep the rung iff the arm's mean held-out BPB is not worse than the "
        f"control's by more than the CONTROL's own seed spread "
        f"({summary['control'][1]:.4f}) AND the step time shows the saving. A tie is "
        f"not a win."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("micro", help="the 4 combos at the b1 shapes")
    m.add_argument("--reps", type=int, default=5)
    m.add_argument("--warm", type=int, default=2)
    a = sub.add_parser("ab", help="the training A/B driver (seeds x 2 arms)")
    a.add_argument("--state", default="fp32", choices=("fp32", "bf16"))
    a.add_argument("--ns", default="fp32", choices=("fp32", "bf16"))
    a.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a.add_argument("--steps", type=int, default=STEPS)
    a.add_argument("--batch", type=int, default=BATCH)
    a.add_argument("--seq", type=int, default=SEQ)
    a.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    a.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    a.add_argument("--log-every", type=int, default=LOG_EVERY)
    a.add_argument("--data", default=DATA)
    a.add_argument("--device", default="cuda")
    a.add_argument("--out-dir", default="~/logs/ab_opt")
    r = sub.add_parser("_run", help="one arm, one seed (the child mode)")
    r.add_argument("--state", required=True, choices=("fp32", "bf16"))
    r.add_argument("--ns", required=True, choices=("fp32", "bf16"))
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--data", default=DATA)
    r.add_argument("--device", default="cuda")
    r.add_argument("--steps", type=int, default=STEPS)
    r.add_argument("--batch", type=int, default=BATCH)
    r.add_argument("--seq", type=int, default=SEQ)
    r.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    r.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    r.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = ap.parse_args(argv)
    if args.mode == "_run":
        return _run_one(args)
    if args.mode == "micro":
        why = preflight("cuda" if torch.cuda.is_available() else "cpu")
    else:
        why = preflight(args.device)
    if why is not None:
        return skip(why)
    return micro(args.reps, args.warm) if args.mode == "micro" else ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
