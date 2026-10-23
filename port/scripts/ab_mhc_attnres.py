"""The mHC × AttnRes composition A/B: off / mHC / AttnRes / both, 3 seeds x 600 steps.

    uv run python scripts/ab_mhc_attnres.py ab            # 4 arms x 3 seeds
    uv run python scripts/ab_mhc_attnres.py ab --seeds 1  # one seed, dry check

**THIS DRIVER HAS NOT BEEN RUN.** The card is held by a training run (AGENTS
§1.5: one heavy thing at a time), so every number below is unmeasured and the
pre-registered rule is written BEFORE the first run. A skipped preflight prints
a SKIP banner and no table.

**Arms** — two flags, four corners, everything else the battery's own COMMON:
`medium`, 600 steps, b8 x s512, the `mix_big` plan, bf16 + triton MoE +
`moe_capacity=0` (dropless) + host Engram, tf32 + `ns_kind=triton` +
`opt_graph=true`, one 81 920 B eval window (20 x 8 x 512) on every arm and
seed, `data_local/mix4/web.bin` train and `web_eval.bin` held out:

* `off`   — ReZero, the control: `use_mhc=false, use_attnres=false`;
* `mhc`   — `use_mhc=true,  use_attnres=false` (K=2 lanes, Eq. 3/7/8/9);
* `attnres` — `use_mhc=false, use_attnres=true` (the depth mixture);
* `both`  — `use_mhc=true, use_attnres=true` (`loopblock`'s "THE COMPOSITION":
  `H_pre` mixes the lanes down to the body, AttnRes chooses the write UPDATE,
  `H_post` deposits it, `H_res` mixes, readout = `Σ lanes / K`).

**Pre-registered prediction.** The composition TIES, and the prior is the
singles' own record: attnres is a 3-seed tie (`511d53e`), mHC is a 3-seed tie
(6.3653 vs 6.3433, delta +0.022 against a 0.073 bar), and the AttnRes lever is
exactly the identity at the zero query (`tests/test_mhc_attnres.py` gate 4:
`depth_attend([0, y], scale=2)` is `y` with `torch.equal`, and the composition's
update is the mHC-only update), so there is no step-0 head start to cash. (The
base's mHC init is `b_pre = 10`, whose `H_pre ≈ K`, so equality to the PLAIN
ReZero path is not claimed — gate 4 prints that distance.) A composition whose
two halves each buy nothing at 9M should not buy something because they were
multiplied; if it does, the numbers below say so.

**The rule, fixed before the run.** Each arm is scored by its mean held-out
BPB over the 3 seeds and its own seed spread (AGENTS §1.2). The composition
earns its place iff

    both_mean < min(mhc_mean, attnres_mean) - max(control_spread, single_spread)

i.e. it must beat the BETTER SINGLE by more than either side's seed spread;
anything else is a TIE and DELETES THE COMPOSITION, keeping the better single
(never the pair: one mechanism fewer for the same loss). The two singles are
reported against `off` but not re-adjudicated — they have their own waves.

**Void checks** (a number failing one is not a number): every eval window is
the same 81 920 B, zero NaN-skipped steps, and each arm shows the counter that
proves its mechanism RAN: `off` must show `rezero_write` and no arm counter,
`mhc`/`attnres` their own, `both` `mhc_attnres_forward` once per forward plus
`mhc_attnres_write` per iteration. An arm that runs and writes nothing is the
defect class these counters exist for.

**Protocol** (`docs/protocols/AB-PROTOCOL.md`): 3 seeds per arm, each
arm-seed a SEPARATE PROCESS (3 seeds is 3 initialisations). Output: one JSONL
per run under `~/logs/ab_mhc_attnres/`, the run's own `<name>.config.toml`
beside it (a number needs its config, AGENTS §1.4), and a driver
`event=params` row (the instantiated count, the mHC share, the AttnRes share).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

# `scripts/` is not a package: reuse the byteflow driver's preflight/readers
# rather than a second copy of the same card checks and JSONL parsing.
_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.append(_SCRIPTS)
from ab_byteflow import preflight, read_run, skip, summarize  # noqa: E402

#: The training wave's mix4 plan (owner, 2026-10-22): real code + filtered math,
#: `web` painted as the held-out eval stream. Local paths, not /mnt.
DATA = "/home/sehaxe/dormouse/data_local/mix4/web.bin"
EVAL = "/home/sehaxe/dormouse/data_local/mix4/web_eval.bin"

SEEDS = (1, 2, 3)
STEPS = 600
PRESET = "medium"
BATCH = 8
SEQ = 512
EVAL_BATCHES = 20
EVAL_EVERY = 500
LOG_EVERY = 250
LR = 3e-4  # TrainConfig's default; the battery does not override it

#: `/tmp/opencode/battery.sh`'s COMMON, as `load_run` keyword arguments — the
#: same block `scripts/ab_mhc.py` uses, so the two tables are comparable.
COMMON_TRAIN = dict(
    steps=STEPS,
    batch=BATCH,
    seq_len=SEQ,
    eval=True,
    eval_every=EVAL_EVERY,
    eval_batches=EVAL_BATCHES,
    log_every=LOG_EVERY,
    ckpt_every=10**9,  # the A/B scores held-out BPB; no weights are written
    tf32=True,
    ns_kind="triton",
    opt_graph=True,
    engram_ram=True,
    mix_plan="configs/mix_big.toml",
)

#: The model-side COMMON (`--set` flags in the battery). `use_mhc` and
#: `use_attnres` are the ARM, set per run below; `loom` is pinned false so a
#: preset edit cannot move a third write gate into the comparison.
COMMON_MODEL = dict(
    moe_kernel="triton",
    moe_capacity=0.0,
    bf16=True,
    mhc_streams=2,
    loom=False,
)

#: `(arm name, use_mhc, use_attnres)`. The control is first so the table reads
#: against it; `both` is last so a reader sees the composition as the claim.
ARMS = (
    ("off", False, False),
    ("mhc", True, False),
    ("attnres", False, True),
    ("both", True, True),
)

#: The counters each arm MUST show and the ones it must NOT. Values are the
#: exact predicates `check` applies (`>0`, `==0`).
EXPECT = {
    "off": {"forward": ">0", "rezero_write": ">0", "mhc": "==0", "attnres_write": "==0",
            "mhc_attnres_forward": "==0"},
    "mhc": {"forward": ">0", "mhc": ">0", "rezero_write": "==0", "mhc_attnres_forward": "==0"},
    "attnres": {"forward": ">0", "attnres_write": ">0", "rezero_write": "==0",
                "mhc_attnres_forward": "==0"},
    "both": {"forward": ">0", "mhc": ">0", "attnres_write": ">0",
             "mhc_attnres_forward": ">0", "mhc_attnres_write": ">0", "rezero_write": "==0"},
}


def check(runs: list[dict], window: int, expect: dict[str, str]) -> list[str]:
    """The void checks: a number that fails one is not a number."""
    why: list[str] = []
    for r in runs:
        if r["scored"] != window:
            why.append(f"seed {r['seed']}: eval window {r['scored']} != {window} B")
        if r["nan"]:
            why.append(f"seed {r['seed']}: {r['nan']} NaN-skipped steps")
        for name, want in expect.items():
            got = r["counters"].get(name, 0)
            if want == ">0" and got <= 0:
                why.append(f"seed {r['seed']}: {name}={got} — the arm did not run")
            if want == "==0" and got != 0:
                why.append(f"seed {r['seed']}: {name}={got} — the wrong shape ran")
    return why


def _verdict(arm: str, control: list[dict], runs: list[dict], window: int) -> dict:
    """One arm against the control: VOID, or its mean/spread and a delta."""
    why = check(control, window, EXPECT["off"]) + check(runs, window, EXPECT[arm])
    cmean, cspread = summarize(control)
    amean, aspread = summarize(runs)
    return {
        "verdict": "void" if why else "scored",
        "reason": "; ".join(why),
        "control_mean": cmean,
        "control_spread": cspread,
        "arm_mean": amean,
        "arm_spread": aspread,
        "delta": amean - cmean,
        "ms_control": statistics.median([r["ms_per_step"] for r in control]),
        "ms_arm": statistics.median([r["ms_per_step"] for r in runs]),
    }


def _run_one(args: argparse.Namespace) -> int:
    """One arm, one seed, one process: the trainer with the two flags made live."""
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train import train

    args.use_mhc, args.use_attnres = next(
        (m, a) for name, m, a in ARMS if name == args.arm
    )
    run = load_run(
        PRESET,
        model_overrides={
            **COMMON_MODEL,
            "use_mhc": args.use_mhc,
            "use_attnres": args.use_attnres,
        },
        train_overrides={
            **COMMON_TRAIN,
            "seed": args.seed,
            "lr": LR,
            "out_dir": str(Path(args.out).parent),
            "ckpt_name": f"{args.arm}_s{args.seed}",
            # The JSONL's `tokens_per_s` is this driver's speed instrument and
            # `timers` OFF is what keeps it honest (`timer.report()` resets the
            # mark the telemetry row reads — `train/loop.py`'s own comment).
            "timers": False,
            "json_log": str(args.out),
            "eval_data": args.eval,
        },
    )
    run.train.data = args.data
    if not Path(run.train.data).is_file():
        print(f"no data file at {run.train.data}: pass --data")
        return 1
    Path(args.out).with_suffix(".config.toml").write_text(run.snapshot_toml())

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    import torch

    model = DormouseModel(run.model, device=torch.device(device))
    # The dispatch must match the flags: an arm whose module is not built is a
    # NOT_ENGAGED run wearing the arm's name.
    assert (model.loop.mhc is not None) == args.use_mhc, (
        "loop.mhc is built iff use_mhc — the arm's dispatch does not match its flag"
    )
    assert (model.loop.attnres is not None) == args.use_attnres, (
        "loop.attnres is built iff use_attnres"
    )
    if args.use_mhc:
        assert model.loop.mhc.n_streams == 2, model.loop.mhc.n_streams
    mhc_params = sum(p.numel() for p in model.loop.mhc.parameters()) if args.use_mhc else 0
    ar_params = sum(p.numel() for p in model.loop.attnres.parameters()) if args.use_attnres else 0
    total = sum(p.numel() for p in model.parameters())
    window = args.eval_batches * args.batch * args.seq
    print(
        f"[run] arm={args.arm} use_mhc={run.model.use_mhc} "
        f"use_attnres={run.model.use_attnres} seed={args.seed} "
        f"steps={args.steps} b{args.batch} s{args.seq} lr={LR} window={window} B "
        f"params={total:,} mhc_params={mhc_params:,} ({mhc_params / total:.4%}) "
        f"attnres_params={ar_params:,}",
        flush=True,
    )
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train(model, run, device=torch.device(device), log=print)

    def _row(**kw) -> None:
        with open(args.out, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(kw) + "\n")

    params = {"params": total, "mhc_params": mhc_params, "attnres_params": ar_params}
    if torch.device(device).type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        params["peak_gib"] = peak
        print(f"[peak] {peak:.2f} GiB (torch.cuda.max_memory_allocated)", flush=True)
    _row(event="params", step=args.steps, **params)
    return 0


def ab(args: argparse.Namespace) -> int:
    """The driver: `--seeds` x 4 arms, separate processes, then the table."""
    seeds = tuple(int(s) for s in args.seeds.split(","))
    window = args.eval_batches * args.batch * args.seq
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"A/B: {len(ARMS)} arms x {len(seeds)} seeds x {args.steps} steps, {PRESET} "
        f"b{args.batch} s{args.seq} lr={LR}, window {window} B. Output: {out_dir}\n"
        f"pre-registered: the composition TIES (both singles are ties at 9M and the "
        f"AttnRes lever is the identity at the zero query, so there is no step-0 "
        f"head start to cash)."
    )
    for arm, _mhc, _ar in ARMS:
        for seed in seeds:
            out = out_dir / f"{arm}_s{seed}.jsonl"
            out.unlink(missing_ok=True)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_run",
                "--arm",
                arm,
                "--seed",
                str(seed),
                "--out",
                str(out),
                "--data",
                args.data,
                "--eval",
                args.eval,
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
            print(f"\n=== {arm} seed {seed} -> {out}", flush=True)
            done = subprocess.run(cmd, check=False)
            if done.returncode != 0:
                print(f"the {arm} run (seed {seed}) exited {done.returncode}: no table.")
                return done.returncode

    table = {arm: [] for arm, _m, _a in ARMS}
    for arm, _mhc, _ar in ARMS:
        for seed in seeds:
            row = {"seed": seed, **read_run(out_dir / f"{arm}_s{seed}.jsonl", args.steps)}
            for line in (out_dir / f"{arm}_s{seed}.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r.get("event") == "params":
                    row.update(
                        params=r.get("params"),
                        mhc_params=r.get("mhc_params"),
                        attnres_params=r.get("attnres_params"),
                    )
            table[arm].append(row)
    print(
        f"\n{'arm':>7s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'ms/step':>8s} "
        f"{'tok/s':>8s} {'peak GiB':>8s} {'nan':>4s} {'params':>10s}"
    )
    for arm, runs in table.items():
        for r in runs:
            peak = f"{r['peak_gib']:.2f}" if r["peak_gib"] is not None else "-"
            print(
                f"{arm:>7s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['ms_per_step']:8.1f} {r['tok_s']:8.0f} {peak:>8s} {r['nan']:>4d} "
                f"{r.get('params', '-'):>10}"
            )
    for arm, runs in table.items():
        mean, spread = summarize(runs)
        print(
            f"{arm:>7s} mean bpb {mean:.4f}  spread {spread:.4f}  "
            f"median warm ms/step {statistics.median([r['ms_per_step'] for r in runs]):.1f}  "
            f"mhc_writes={sum(r['counters'].get('mhc', 0) for r in runs)} "
            f"attnres_writes={sum(r['counters'].get('attnres_write', 0) for r in runs)} "
            f"pair_forwards={sum(r['counters'].get('mhc_attnres_forward', 0) for r in runs)}"
        )

    control = table["off"]
    verdicts = {
        arm: _verdict(arm, control, table[arm], window)
        for arm in ("mhc", "attnres", "both")
    }
    for arm, v in verdicts.items():
        print(
            f"\n{arm:>7s} vs off: {v['arm_mean']:.4f} +- {v['arm_spread']:.4f} vs "
            f"{v['control_mean']:.4f} +- {v['control_spread']:.4f} "
            f"(delta {v['delta']:+.4f}) — {v['verdict'].upper()}"
            + (f"\nvoid reasons: {v['reason']}" if v["reason"] else "")
        )

    singles = [verdicts["mhc"], verdicts["attnres"]]
    best = min(singles, key=lambda v: v["arm_mean"])
    both = verdicts["both"]
    bar = max(best["control_spread"], best["arm_spread"], both["arm_spread"])
    if any(v["verdict"] == "void" for v in (*singles, both)):
        comp = "void"
        why = "a single arm or the pair failed its void checks"
    elif both["arm_mean"] < best["arm_mean"] - bar:
        comp = "win"
        why = f"beats the better single by more than the {bar:.4f} spread bar"
    else:
        comp = "tie"
        why = (
            f"does not beat the better single by more than {bar:.4f}; a tie DELETES "
            f"the composition and keeps the better single (one mechanism fewer for "
            f"the same loss)"
        )
    print(
        f"\ncomposition verdict: {comp.upper()} — {why}.\n"
        f"rule (fixed before the run): both_mean < min(single means) - "
        f"max(control spread, single spread); anything else is a tie.\n"
        f"step time (SEPARATE verdict, reported not gated): off "
        f"{both['ms_control']:.1f} -> both {both['ms_arm']:.1f} ms/step "
        f"({both['ms_arm'] - both['ms_control']:+.1f} ms, "
        f"{(both['ms_arm'] - both['ms_control']) / both['ms_control']:+.1%})."
    )
    return 0


def _dry_run() -> int:
    """Resolve each arm's config through the real `load_run` seam; train nothing."""
    from dormouse.config import load_run

    for arm, mhc, attnres in ARMS:
        run = load_run(
            PRESET,
            model_overrides={
                **COMMON_MODEL,
                "use_mhc": mhc,
                "use_attnres": attnres,
            },
        )
        assert run.model.use_mhc is mhc and run.model.use_attnres is attnres
        print(
            f"[dry-run] arm={arm} use_mhc={run.model.use_mhc} "
            f"use_attnres={run.model.use_attnres} mhc_streams={run.model.mhc_streams}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ab", help="the A/B driver (seeds x 4 arms)")
    a.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve the four arms' configs through load_run and exit (no training)",
    )
    a.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a.add_argument("--steps", type=int, default=STEPS)
    a.add_argument("--batch", type=int, default=BATCH)
    a.add_argument("--seq", type=int, default=SEQ)
    a.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    a.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    a.add_argument("--log-every", type=int, default=LOG_EVERY)
    a.add_argument("--data", default=DATA)
    a.add_argument("--eval", default=EVAL)
    a.add_argument("--device", default="cuda")
    a.add_argument("--out-dir", default="~/logs/ab_mhc_attnres")
    r = sub.add_parser("_run", help="one arm, one seed (the child mode)")
    r.add_argument("--arm", required=True, choices=[arm for arm, _m, _a in ARMS])
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--data", default=DATA)
    r.add_argument("--eval", default=EVAL)
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
    if args.dry_run:
        return _dry_run()
    why = preflight(args.device)
    if why is not None:
        return skip(why)
    return ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
