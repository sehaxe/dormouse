"""The ternary FORM A/B: which quantization the release ships, 3 seeds x 3 arms.

    uv run python scripts/ab_ternary_form.py ab            # the whole grid (NOT run)
    uv run python scripts/ab_ternary_form.py ab --seeds 1  # one seed, dry check

**THIS DRIVER HAS NOT BEEN RUN.** The card is busy (AGENTS §1.5: one heavy
thing at a time); the preflight refuses and prints no table. Every number is
unmeasured until it runs.

**Why.** The 2026-10-21 measurement (medium, b8 x s512, 2000 steps, one
81 920 B window, 3 seeds) put `factor_qat=ternary` + `dense_qat=ternary` at
**BPB 3.2733** against the bf16 control's **3.0750** (+0.198, +6.5%). The
coverage list (`tests/test_dense_qat.py`) showed that arm projected the KDA
attention projections (`loop.attn.q/k/v/g/beta`) and `loop.msa.*` — the BitNet
b1.58 rule (arXiv:2402.17764) and our own `act_quant` ("attention is never
below 8 bits") both say those should not be quantized. This driver settles
which form ships after `dense_qat_skip_attn`, now the default, excludes them.

**Arms** — everything but the arm's own flags is the battery's COMMON
(`medium`, 2000 steps, b8 x s512, `mix_big`, bf16 + triton MoE + dropless +
host Engram, tf32 + `ns_kind=triton` + `opt_graph`), so every arm and seed
scores the SAME window `20 x 8 x 512 = 81 920 B`. The release recipe's
500-step factor anneal (`ab_qat.py`'s `qat` arm) is COMMON too: a difference is
the weight form, not the schedule.

* `f` — `factor_qat=ternary` only. The factors-only reference; dense off.
* `b` — `factor_qat=ternary, dense_qat=ternary`, `dense_qat_skip_attn=true`.
  The new default: dense FFN/controller/memory/readout weights projected,
  attention left in bf16. The direct successor to the 3.2733 arm.
* `n` — `b` + `act_quant=int4`. The full release form (ternary weights,
  4-bit activations, attention >= 8 bits at the activation seam too).
* `a` (dropped, deliberately) — factors + dense with attention at int8.
  There is no int8 WEIGHT codec in `dense_qat`: `FACTOR_QATS` is
  ternary/binary/mxfp4/ternary_s3t, and `act_quant`'s `int8` is an activation
  format applied per row/group at a different seam. Reusing it as a weight
  projection needs a new `dense_qat` literal, its absmean/STE contract and its
  own gold test — not cheap, so the arm is dropped rather than approximated.
  `--arms a` refuses with this reason.

**Pre-registered rule** (written BEFORE the first run, AGENTS §1.2): compare
each step against the arm it adds compression to, using the CONTROL arm's own
3-seed spread as the tolerance. `b` is kept iff `mean(b) - mean(f) <=
spread(f)`; `n` is kept iff `mean(n) - mean(b) <= spread(b)`. A step inside the
spread is a TIE and keeps the more compressed form (that is what QAT is for);
a larger loss reverts that step. No number is compared across waves: the
recorded bf16 control (3.0750 +- 0.0084) and the 3.2733 no-skip arm are
printed for context only, not as arms.

VOID (not a number) when an eval window is not 81 920 B, a NaN-skipped step
occurred, or the counters disagree with the arm: `f` must show
`dense_qat_arms=0`/`act_quant_arms=0`, `b` `dense_qat_arms>0`/`act_quant_arms=0`,
`n` both `>0`. A run whose counters say the arm did not engage measured the
bf16 forward wearing a quantized label.

Output: one JSONL per run under `~/logs/ab_ternary_form/`, the run's own
`<name>.config.toml` beside it (a number needs its config, AGENTS §1.4), and a
driver `event=params` row.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# `scripts/` is not a package: reuse the battery's preflight/readers rather
# than a second copy of the same card checks and JSONL parsing.
_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.append(_SCRIPTS)
from ab_byteflow import check, preflight, read_run, skip, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: The port's data on this box (AGENTS §2.6) — the LOCAL copies, so the driver
#: does not depend on the /mnt drive being mounted.
DATA = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"
EVAL = "/home/sehaxe/dormouse/data_local/eval_mix/eval_mix.bin"

SEEDS = (1, 2, 3)
STEPS = 2000
PRESET = "medium"
BATCH = 8
SEQ = 512
EVAL_BATCHES = 20
EVAL_EVERY = 500
LOG_EVERY = 250
#: One eval window: 20 x 8 x 512 = 81 920 B (AGENTS §2.6).
WINDOW = EVAL_BATCHES * BATCH * SEQ

#: `/tmp/opencode/battery.sh`'s COMMON, as `load_run` keyword arguments. Every
#: entry is the battery's, not this driver's; the arm flags are the only
#: difference. `factor_qat_anneal_steps` is the release recipe's ramp
#: (`ab_qat.py`), COMMON so no arm gets a different schedule.
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
    factor_qat_anneal_steps=500,
)

#: The model-side COMMON (`--set` flags in the battery) plus nothing else.
COMMON_MODEL = dict(moe_kernel="triton", moe_capacity=0.0, bf16=True)

#: `(arm, model overrides)`. `f` first: the table reads against it. `b` and `n`
#: spell `dense_qat_skip_attn=true` out, so the run's own config record — not
#: this docstring — proves attention was excluded.
ARMS: dict[str, dict[str, object]] = {
    "f": {"factor_qat": "ternary"},
    "b": {"factor_qat": "ternary", "dense_qat": "ternary", "dense_qat_skip_attn": True},
    "n": {
        "factor_qat": "ternary",
        "dense_qat": "ternary",
        "dense_qat_skip_attn": True,
        "act_quant": "int4",
    },
}

#: The dropped arm and the reason printed when someone asks for it.
DROPPED = {
    "a": "attention at int8 needs an int8 WEIGHT codec (FACTOR_QATS has none; "
    "act_quant's int8 is an activation seam), with its own STE contract and gold "
    "test — not cheap, so the arm is dropped rather than approximated",
}

#: Each arm's counters must agree with its label.
EXPECT = {
    "f": {"dense_qat_arms": "==0", "act_quant_arms": "==0"},
    "b": {"dense_qat_arms": ">0", "act_quant_arms": "==0"},
    "n": {"dense_qat_arms": ">0", "act_quant_arms": ">0"},
}


def resolve(arm: str, seed: int, out: str | None, args: argparse.Namespace):
    """The run config for one arm-seed: the SAME COMMON the child gets."""
    from dormouse.config import load_run

    out_dir = args.out_dir if out is None else str(Path(out).parent)
    run = load_run(
        PRESET,
        model_overrides={**COMMON_MODEL, **ARMS[arm]},
        train_overrides={
            **COMMON_TRAIN,
            "steps": args.steps,
            "batch": args.batch,
            "seq_len": args.seq,
            "eval_batches": args.eval_batches,
            "eval_every": args.eval_every,
            "log_every": args.log_every,
            "seed": seed,
            "out_dir": out_dir,
            "ckpt_name": f"ternary_form_{arm}_s{seed}",
            "data": args.data,
            "eval_data": args.eval,
            "json_log": out,
        },
    )
    run.train.data = args.data
    return run


def _run_one(args: argparse.Namespace) -> int:
    """One arm, one seed, one process: the trainer with the arm's flags live."""
    import torch

    from dormouse.model import DormouseModel
    from dormouse.train import train

    run = resolve(args.arm, args.seed, args.out, args)
    if not Path(run.train.data).is_file():
        print(f"no data file at {run.train.data}: pass --data")
        return 1
    Path(args.out).with_suffix(".config.toml").write_text(run.snapshot_toml())

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    model = DormouseModel(run.model, device=torch.device(device))
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[run] arm={args.arm} seed={args.seed} steps={run.train.steps} "
        f"b{run.train.batch} s{run.train.seq_len} window={WINDOW} B "
        f"factor_qat={run.model.factor_qat} dense_qat={run.model.dense_qat} "
        f"skip_attn={run.model.dense_qat_skip_attn} act_quant={run.model.act_quant} "
        f"params={total:,}",
        flush=True,
    )
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train(model, run, device=torch.device(device), log=print)

    with open(args.out, "a", encoding="utf-8") as sink:
        row = {"event": "params", "step": run.train.steps, "params": total}
        if torch.device(device).type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 2**30
            row["peak_gib"] = peak
            print(f"[peak] {peak:.2f} GiB (torch.cuda.max_memory_allocated)", flush=True)
        sink.write(json.dumps(row) + "\n")
    return 0


def step(control: list[dict], arm: list[dict], window: int, arm_name: str) -> dict:
    """The pre-registered rule over one compression step.

    VOID when a run fails its window/NaN/counter checks. Otherwise KEEP iff the
    arm's mean BPB is within the CONTROL's own seed spread of it; a larger loss
    REVERTS the step. A tie keeps the more compressed form, by design.
    """
    why = check(control, window, EXPECT[control[0]["arm"]]) + check(arm, window, EXPECT[arm_name])
    cmean, cspread = summarize(control)
    amean, aspread = summarize(arm)
    out = {
        "control_mean": cmean,
        "control_spread": cspread,
        "arm_mean": amean,
        "arm_spread": aspread,
        "delta": amean - cmean,
        "reason": "",
    }
    if why:
        out["verdict"] = "void"
        out["reason"] = "; ".join(why)
    elif amean - cmean <= cspread:
        out["verdict"] = "keep"
    else:
        out["verdict"] = "revert"
    return out


def ab(args: argparse.Namespace) -> int:
    """The driver: `--seeds` x arms, separate processes, then the table."""
    seeds = tuple(int(s) for s in args.seeds.split(","))
    arm_names = [a for a in args.arms.split(",") if a]
    for name in arm_names:
        if name in DROPPED:
            print(f"arm {name!r} is dropped: {DROPPED[name]}")
            return 2
        if name not in ARMS:
            print(f"unknown arm {name!r}: choose from {sorted(ARMS)}")
            return 2
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"ternary FORM A/B: {len(arm_names)} arms x {len(seeds)} seeds x "
        f"{args.steps} steps, {PRESET} b{args.batch} s{args.seq}, window {WINDOW} B. "
        f"Output: {out_dir}\n"
        f"pre-registered: keep a compression step iff its mean BPB is within the "
        f"control arm's own seed spread; a tie keeps the more compressed form. "
        f"NOT run yet — no number exists."
    )
    for arm in arm_names:
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
                "--out-dir",
                str(out_dir),
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

    table: dict[str, list[dict]] = {arm: [] for arm in arm_names}
    for arm in arm_names:
        for seed in seeds:
            path = out_dir / f"{arm}_s{seed}.jsonl"
            row = {"seed": seed, "arm": arm, **read_run(path, args.steps)}
            for line in path.read_text().splitlines():
                r = json.loads(line)
                if r.get("event") == "params":
                    row.update(params=r.get("params"))
            table[arm].append(row)

    print(
        f"\n{'arm':>3s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'ms/step':>8s} "
        f"{'nan':>4s} {'dense_qat':>9s} {'act_quant':>9s}"
    )
    for arm, runs in table.items():
        for r in runs:
            print(
                f"{arm:>3s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['ms_per_step']:8.1f} {r['nan']:>4d} "
                f"{r['counters'].get('dense_qat_arms', 0):>9d} "
                f"{r['counters'].get('act_quant_arms', 0):>9d}"
            )
    for arm, runs in table.items():
        mean, spread = summarize(runs)
        print(f"{arm:>3s} mean bpb {mean:.4f}  spread {spread:.4f}  n={len(runs)}")

    print(
        "\ncontext only (separate waves, NOT arms here): bf16 control "
        "3.0750 +- 0.0084; the no-skip dense arm 3.2733 (+0.198, +6.5%)."
    )
    verdicts = {}
    if "b" in table and "f" in table:
        verdicts["b"] = step(table["f"], table["b"], WINDOW, "b")
        print(
            f"dense (attention skipped) vs factors-only: "
            f"{verdicts['b']['delta']:+.4f} BPB, control spread "
            f"{verdicts['b']['control_spread']:.4f} -> {verdicts['b']['verdict'].upper()}"
            + (f" ({verdicts['b']['reason']})" if verdicts["b"]["reason"] else "")
        )
    if "n" in table and "b" in table:
        verdicts["n"] = step(table["b"], table["n"], WINDOW, "n")
        print(
            f"act_quant=int4 on top of b: {verdicts['n']['delta']:+.4f} BPB, "
            f"b's spread {verdicts['n']['control_spread']:.4f} -> "
            f"{verdicts['n']['verdict'].upper()}"
            + (f" ({verdicts['n']['reason']})" if verdicts["n"]["reason"] else "")
        )
    if any(v["verdict"] == "void" for v in verdicts.values()):
        print("\nVOID: at least one arm failed its checks — no form decision from those numbers.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ab", help="the A/B driver (arms x seeds)")
    a.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a.add_argument("--arms", default=",".join(ARMS))
    a.add_argument("--steps", type=int, default=STEPS)
    a.add_argument("--batch", type=int, default=BATCH)
    a.add_argument("--seq", type=int, default=SEQ)
    a.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    a.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    a.add_argument("--log-every", type=int, default=LOG_EVERY)
    a.add_argument("--data", default=DATA)
    a.add_argument("--eval", default=EVAL)
    a.add_argument("--device", default="cuda")
    a.add_argument("--out-dir", default="~/logs/ab_ternary_form")
    r = sub.add_parser("_run", help="one arm, one seed (the child mode)")
    r.add_argument("--arm", required=True, choices=sorted(ARMS))
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--out-dir", required=True)
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
    why = preflight(args.device)
    if why is not None:
        return skip(why)
    return ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
