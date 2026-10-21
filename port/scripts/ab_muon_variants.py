"""The Muon-variant A/B: control (muon+colrow, WSD) vs normuon vs sf-normuon.

    uv run python scripts/ab_muon_variants.py ab              # 3 seeds x 3 arms, medium
    uv run python scripts/ab_muon_variants.py ab --seeds 1    # one seed, dry check

**Do not run this while the card is busy** (AGENTS §1.5): the preflight refuses
and prints a SKIP banner, and a skipped run has no numbers to quote.

**Arms** (one factor each against the control, `docs/research/2026-10-17-muon-variants.md`
§2 #2 + `docs/research/2026-10-16-sf-muon.md` §4):

* `control`   — `muon_variant=muon`, `muon_sf=false`: the live recipe, WSD on both
                halves. Every number on record.
* `normuon`   — `muon_variant=normuon`: NorMuon's row-wise EMA second moment
                applied the AUTHORS' repo way (`zichongli5/NorMuon`
                `normuon.py:48-56`: row-scale by `1/√(v+1e-10)`, restore the
                pre-normalization Frobenius norm, `× max(1, m/n)^0.5`). It
                REPLACES our ColRow tail rather than composing with it —
                composed, the tail is inert (its row statistic is ColRow's row
                half) — so this arm is "NorMuon instead of ColRow", and against
                the `muon` control its magnitude is not preserved (measured
                `‖ΔW‖` ratios 0.7066 at `[64, 5461]`, 0.1295 at `[2048, 64]`;
                the magnitude invariant holds against the REPO's Muon, i.e.
                without ColRow). Read the verdict as a REPLACE, not a
                direction-only delta.
* `sf-normuon`— `muon_variant=normuon` + `muon_sf=true`: the schedule-free wrapper
                on top. Its Muon-group LR is CONSTANT after warmup while the
                AdamW half keeps WSD — the hybrid deviation `optim.Muon`
                documents. Predicted to TIE at 2k: the published SF gains are
                stated at >=1000 tokens/param, orders past this arm.

**Protocol** (`docs/protocols/AB-PROTOCOL.md`, the cost note's Stage 1): 3 seeds
per arm, 2 000 steps, `medium`, b8 x s512, lr 1e-3, pure CE, and **one eval
window of 81 920 B on every arm and seed** (`eval_batches 20 x batch 8 x seq_len
512`). Each arm-seed is a SEPARATE PROCESS, because 3 seeds is 3 initialisations.
Output: one JSONL per run under `~/logs/ab_muon_variants/`, plus the run's own
`<name>.config.toml` beside it (a number needs its config, AGENTS §1.4).

**The SF arm gets one extra instrument**, because a schedule-free claim judged at
one endpoint is the mistake `2609.04577` names: after the run the driver scores
the held-out BPB at BOTH iterates — `X` (the returned/averaged one) and `Y` (the
gradient-evaluation point) — and writes an `sf_gap` JSONL row. An X-only win is
an X-only win, and that row is how a reader sees it.

**What the driver does NOT do:** apply the verdict. It prints the table, the void
checks and the pre-registered rule (AGENTS §1.2: keep the arm iff its mean BPB
beats the control's mean by more than the CONTROL's own seed spread; a tie
deletes the mechanism). A BPB is a measurement, not a verdict.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight, skip  # noqa: E402

#: The port's data on this box (AGENTS §2.6) — the LOCAL copy, so the driver does
#: not depend on the /mnt drive being mounted.
DATA = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"

SEEDS = (1, 2, 3)
STEPS = 2000
BATCH = 8
SEQ = 512
LR = 1e-3
EVAL_BATCHES = 20
EVAL_EVERY = 200
LOG_EVERY = 50
PRESET = "medium"

#: `(arm name, muon_variant, muon_sf)`. The control is first so the table reads
#: against it.
ARMS = (
    ("control", "muon", False),
    ("normuon", "normuon", False),
    ("sf_normuon", "normuon", True),
)


def _run_one(args: argparse.Namespace) -> int:
    """One arm, one seed, one process: the trainer with the arm made live."""
    import dormouse.train.loop as train_loop
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train import bpb, evaluate, open_streams, train

    run = load_run(
        PRESET,
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
            "ckpt_name": f"{args.arm}_s{args.seed}",
            # `timers` is OFF: `loop.py` calls `timer.report()` (which RESETS the
            # mark) before the telemetry row reads `timer.window_s()`, so with
            # `--timers` on the JSONL's `tokens_per_s` is the time since that
            # reset (~1e7-1e8 tok/s). The JSONL rate is this driver's speed
            # instrument, so the flag stays off and the defect is reported.
            "timers": False,
            "muon_variant": args.variant,
            "muon_sf": args.sf,
            "json_log": str(args.out),
        },
    )
    run.train.data = args.data
    if not Path(run.train.data).is_file():
        print(f"no data file at {run.train.data}: pass --data")
        return 1
    # The run's own config record, next to its JSONL.
    Path(args.out).with_suffix(".config.toml").write_text(run.snapshot_toml())

    real = train_loop.MixOptimizer
    built: list = []

    def spy(model, **kw):
        if not {"muon_variant", "muon_sf"} & set(kw):
            print(
                "WIRING GAP: train/loop.py's MixOptimizer call site forwards neither "
                "muon_variant nor muon_sf, so `--set` alone is INERT and this table "
                "would be a table of the control."
            )
        kw.setdefault("muon_variant", args.variant)
        kw.setdefault("muon_sf", args.sf)
        opt = real(model, **kw)
        built.append(opt)
        return opt

    train_loop.MixOptimizer = spy
    device = torch.device(args.device)
    model = DormouseModel(run.model, device=device)
    window = args.eval_batches * args.batch * args.seq
    print(
        f"[run] arm={args.arm} variant={args.variant} sf={args.sf} seed={args.seed} "
        f"steps={args.steps} b{args.batch} s{args.seq} lr={LR} window={window} B",
        flush=True,
    )
    train(model, run, device=device, log=print)
    assert built and built[0].muon is not None, "no Muon group was built"
    assert built[0].muon.variant == args.variant, "the variant did not reach Muon"
    assert built[0].muon.sf is args.sf, "the schedule-free arm did not reach Muon"

    if args.sf:
        # The X-vs-Y reading (the SF research doc's void check): the run's own
        # evals scored X; Y is where the gradient was taken. One extra held-out
        # pass each, on the same rewound window.
        _train_stream, eval_stream = open_streams(run)
        row: dict = {}
        for mode in ("eval", "train"):
            built[0].sf_switch(mode)
            ce, scored = evaluate(model, eval_stream, run.train.eval_batches, device=device)
            row[f"bpb_{'x' if mode == 'eval' else 'y'}"] = bpb(ce)
            row["scored"] = scored
        with open(args.out, "a", encoding="utf-8") as sink:
            sink.write(json.dumps({"event": "sf_gap", "step": args.steps, **row}) + "\n")
        print(f"[sf gap] {row}", flush=True)
    return 0


def _read(path: Path, steps: int) -> dict:
    """One run's JSONL -> the numbers the verdict table needs.

    `tok/s` is the median of the JSONL's own window rates over the run's SECOND
    HALF — the protocol's ">=50 warm steps" read as "past the warmup window".
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    evals = [r for r in rows if r.get("event") == "eval"]
    logs = [r for r in rows if r.get("event") == "log"]
    warm = [
        r["tokens_per_s"] for r in logs if r.get("step", 0) > steps / 2 and r.get("tokens_per_s")
    ]
    assert evals, f"{path}: no eval row — nothing to compare"
    gap = next((r for r in rows if r.get("event") == "sf_gap"), None)
    return {
        "bpb": evals[-1]["bpb"],
        "scored": evals[-1]["scored"],
        "tok_s": statistics.median(warm) if warm else float("nan"),
        "nan": sum(r.get("nan_skipped", 0) for r in logs),
        "gap": gap,
    }


def ab(args: argparse.Namespace) -> int:
    """The driver: `--seeds` x 3 arms, separate processes, then the table."""
    seeds = tuple(int(s) for s in args.seeds.split(","))
    window = args.eval_batches * args.batch * args.seq
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"A/B: {len(ARMS)} arms x {len(seeds)} seeds x {args.steps} steps, {PRESET} "
        f"b{args.batch} s{args.seq} lr={LR}, window {window} B. Output: {out_dir}"
    )
    for arm, variant, sf in ARMS:
        for seed in seeds:
            out = out_dir / f"{arm}_s{seed}.jsonl"
            out.unlink(missing_ok=True)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_run",
                "--arm",
                arm,
                "--variant",
                variant,
                "--sf" if sf else "--no-sf",
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
            print(f"\n=== {arm} ({variant}, sf={sf}) seed {seed} -> {out}", flush=True)
            done = subprocess.run(cmd, check=False)
            if done.returncode != 0:
                print(f"the {arm} run (seed {seed}) exited {done.returncode}: no table.")
                return done.returncode

    table = {arm: [] for arm, _v, _s in ARMS}
    for arm, _variant, _sf in ARMS:
        for seed in seeds:
            table[arm].append({"seed": seed, **_read(out_dir / f"{arm}_s{seed}.jsonl", args.steps)})
    print(
        f"\n{'arm':>11s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'warm tok/s':>10s} {'nan':>4s}"
    )
    for arm, runs in table.items():
        for r in runs:
            print(
                f"{arm:>11s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['tok_s']:10.0f} {r['nan']:>4d}"
            )
    summary = {}
    for arm, runs in table.items():
        bpbs = [r["bpb"] for r in runs]
        summary[arm] = (statistics.mean(bpbs), max(bpbs) - min(bpbs))
        print(
            f"{arm:>11s} mean bpb {summary[arm][0]:.4f}  spread {summary[arm][1]:.4f}  "
            f"median warm tok/s {statistics.median([r['tok_s'] for r in runs]):.0f}"
        )
    # The void checks: a number that fails one of these is not a number.
    for arm, runs in table.items():
        for r in runs:
            assert r["scored"] == window, (
                f"{arm} seed {r['seed']}: window {r['scored']} != {window}"
            )
            assert r["nan"] == 0, f"{arm} seed {r['seed']}: {r['nan']} NaN-skipped steps"
    for arm, runs in table.items():
        for r in runs:
            if r["gap"] is None:
                continue
            print(
                f"{arm:>11s} seed {r['seed']:>4d} sf gap: bpb(X)={r['gap']['bpb_x']:.4f} "
                f"bpb(Y)={r['gap']['bpb_y']:.4f}"
            )
    print(
        f"\nvoid checks: every eval scored {window} B, 0 NaN-skipped steps, "
        f"{len(seeds)} distinct seeds per arm.\npre-registered rule (AGENTS §1.2 / "
        f"muon-variants §2): keep an arm iff its mean held-out BPB beats the "
        f"control's mean by more than the CONTROL's own seed spread "
        f"({summary['control'][1]:.4f}); a tie deletes the mechanism. The sf arm's "
        f"`sf_gap` row decides whether an X-only win is an X-only win."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ab", help="the A/B driver (seeds x 3 arms)")
    a.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a.add_argument("--steps", type=int, default=STEPS)
    a.add_argument("--batch", type=int, default=BATCH)
    a.add_argument("--seq", type=int, default=SEQ)
    a.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    a.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    a.add_argument("--log-every", type=int, default=LOG_EVERY)
    a.add_argument("--data", default=DATA)
    a.add_argument("--device", default="cuda")
    a.add_argument("--out-dir", default="~/logs/ab_muon_variants")
    r = sub.add_parser("_run", help="one arm, one seed (the child mode)")
    r.add_argument("--arm", required=True)
    r.add_argument("--variant", required=True)
    r.add_argument("--sf", dest="sf", action="store_true", default=False)
    r.add_argument("--no-sf", dest="sf", action="store_false")
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
    # `ab` is the default mode when a subcommand is given by the caller; `main`
    # itself requires one, so `python scripts/ab_muon_variants.py ab` is the form.
    why = preflight(args.device)
    if why is not None:
        return skip(why)
    return ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
