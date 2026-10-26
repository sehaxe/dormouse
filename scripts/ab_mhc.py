"""The mHC A/B: ReZero (control) vs Manifold-Constrained Hyper-Connections, 3 seeds x 2 arms.

    uv run python scripts/ab_mhc.py ab            # 3 seeds x 2 arms, 2000 steps
    uv run python scripts/ab_mhc.py ab --seeds 1  # one seed, dry check

**THIS DRIVER HAS NOT BEEN RUN.** The card is held by a training run (AGENTS
§1.5: one heavy thing at a time), so every number below is unmeasured and the
pre-registered rule is written BEFORE the first run. A skipped preflight prints
a SKIP banner and no table.

**Arms** — one flag differs (`use_mhc`), everything else is the battery's own
COMMON: `medium`, 2000 steps, b8 x s512, the `mix_big` plan, bf16 + triton MoE +
`moe_capacity=0` (dropless) + host Engram, tf32 + `ns_kind=triton` +
`opt_graph=true`, one 81 920 B eval window (20 x 8 x 512) on every arm and seed:

* `off` — ReZero, the fourth arm's control: `use_mhc=false`, K irrelevant;
* `on` — `use_mhc=true, mhc_streams=2`. **K=2 only** (the φ study's own K,
  2604.21106 §5.2; 2409.19606 Tab. 1's knee). `config.validate` refuses any
  other K, `use_attnres`, `loom` and `use_byteflow` alongside it, so the arms
  differ in ONE statement — the loop-boundary residual write.

**Pre-registered prediction.** The arm TIES at 2 000 steps, for three reasons,
all on the record (`docs/reviews/mhc-2026-09-30.md`):

1. at init the write IS ReZero at scale 1 (`b_res = 0` → uniform mixing on
   replicated lanes, `b_pre = 10` → `H_pre≈1`, `b_post = 0` → `H_post≈1`,
   measured relative distance 5.4e-4 in `tests/test_mhc.py`), so nothing
   happens until training moves the projection;
2. the only mHC evidence is a 27B STABILITY result (Fig. 2/5, loss-surge
   suppression), not a 9M quality win; the paper's 0.021 loss gap is at 27B;
3. the φ = 0.65 claim is UNCONSTRAINED HC at the loop boundary
   (2604.21106 §5.2), a different member of the family, and no published
   number connects mHC to φ.

A WIN has to beat the control's own seed spread (AGENTS §1.2). A tie deletes
the mechanism.

**The second verdict, which is NOT a BPB column** (review §6.1): the step-time
delta at a warm step index, with the fwd/bwd split. The arm's cost is launch
bound (20 Sinkhorn iterations × 2 phases × 2 dims per call, invariant to K), so
the table reports `ms/step` and the driver prints the delta; the review's rule
is that the arm lands only if BPB wins AND that delta is affordable at the
operating depth. This driver prints both and applies neither.

**Protocol** (`docs/protocols/AB-PROTOCOL.md`): 3 seeds per arm, one eval
window of 81 920 B (20 x 8 x 512) on every run, each arm-seed a SEPARATE
PROCESS (3 seeds is 3 initialisations). Output: one JSONL per run under
`~/logs/ab_mhc/`, the run's own `<name>.config.toml` beside it (a number needs
its config, AGENTS §1.4), and a driver `event=params` row (the instantiated
count and the mHC share).
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

#: The port's data on this box (AGENTS §2.6) — the LOCAL copy, so the driver
#: does not depend on the /mnt drive being mounted. The eval file is the
#: battery's own held-out mix, so windows stay comparable to its runs.
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
LR = 3e-4  # TrainConfig's default; the battery does not override it

#: `/tmp/opencode/battery.sh`'s COMMON, as `load_run` keyword arguments. Every
#: entry is the battery's, not this driver's; the one model-side difference is
#: the arm's own flag below.
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

#: The model-side COMMON (`--set` flags in the battery) plus the arm controls.
#: `use_attnres`/`loom` are pinned false so a preset edit cannot move a second
#: residual arm into the comparison.
COMMON_MODEL = dict(
    moe_kernel="triton",
    moe_capacity=0.0,
    bf16=True,
    mhc_streams=2,
    use_attnres=False,
    loom=False,
)

#: `(arm name, use_mhc)`. The control is first so the table reads against it.
ARMS = (
    ("off", False),
    ("on", True),
)

#: The counter each arm MUST show, and the one it must NOT: an arm that runs
#: and writes nothing is the defect class the counters exist for.
EXPECT = {
    "off": {"forward": ">0", "mhc": "==0"},
    "on": {"forward": ">0", "mhc": ">0"},
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


def verdict(control: list[dict], arm: list[dict], window: int) -> dict:
    """AGENTS §1.2 over one control and one arm, plus the review's step-time delta.

    VOID when any run fails a void check. Otherwise WIN iff the arm's mean BPB
    beats the control's mean by MORE than the CONTROL's own seed spread;
    anything else — including a tie — deletes the mechanism. `ms_delta` is
    REPORTED (review §6.1's separate verdict), never folded into the BPB rule.
    """
    why = check(control, window, EXPECT["off"]) + check(arm, window, EXPECT["on"])
    cmean, cspread = summarize(control)
    amean, aspread = summarize(arm)
    cms = statistics.median([r["ms_per_step"] for r in control])
    ams = statistics.median([r["ms_per_step"] for r in arm])
    out = {
        "control_mean": cmean,
        "control_spread": cspread,
        "arm_mean": amean,
        "arm_spread": aspread,
        "delta": amean - cmean,
        "ms_control": cms,
        "ms_arm": ams,
        "ms_delta": ams - cms,
        "reason": "",
    }
    if why:
        out["verdict"] = "void"
        out["reason"] = "; ".join(why)
    elif amean < cmean - cspread:
        out["verdict"] = "win"
    else:
        out["verdict"] = "tie"
    return out


def _run_one(args: argparse.Namespace) -> int:
    """One arm, one seed, one process: the trainer with the flag made live."""
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train import train

    run = load_run(
        PRESET,
        model_overrides={**COMMON_MODEL, "use_mhc": args.arm == "on"},
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
    is_on = model.loop.mhc is not None
    assert is_on == (args.arm == "on"), (
        "the arm's dispatch does not match its flag: loop.mhc is built iff use_mhc"
    )
    if is_on:
        assert model.loop.mhc.n_streams == 2, model.loop.mhc.n_streams
    mhc_params = sum(p.numel() for p in model.loop.mhc.parameters()) if is_on else 0
    total = sum(p.numel() for p in model.parameters())
    window = args.eval_batches * args.batch * args.seq
    print(
        f"[run] arm={args.arm} use_mhc={run.model.use_mhc} seed={args.seed} "
        f"steps={args.steps} b{args.batch} s{args.seq} lr={LR} window={window} B "
        f"params={total:,} mhc_params={mhc_params:,} ({mhc_params / total:.4%})",
        flush=True,
    )
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train(model, run, device=torch.device(device), log=print)

    def _row(**kw) -> None:
        with open(args.out, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(kw) + "\n")

    params = {"params": total, "mhc_params": mhc_params}
    if torch.device(device).type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        params["peak_gib"] = peak
        print(f"[peak] {peak:.2f} GiB (torch.cuda.max_memory_allocated)", flush=True)
    _row(event="params", step=args.steps, **params)
    return 0


def ab(args: argparse.Namespace) -> int:
    """The driver: `--seeds` x 2 arms, separate processes, then the table."""
    seeds = tuple(int(s) for s in args.seeds.split(","))
    window = args.eval_batches * args.batch * args.seq
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"A/B: {len(ARMS)} arms x {len(seeds)} seeds x {args.steps} steps, {PRESET} "
        f"b{args.batch} s{args.seq} lr={LR}, window {window} B. Output: {out_dir}\n"
        f"pre-registered: mHC TIES at 2k steps (at-init write IS ReZero@1; the "
        f"published win is 27B stability, not 9M quality; φ=0.65 is unconstrained "
        f"HC at the loop boundary, not mHC)."
    )
    for arm, _flag in ARMS:
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

    table = {arm: [] for arm, _f in ARMS}
    for arm, _flag in ARMS:
        for seed in seeds:
            row = {"seed": seed, **read_run(out_dir / f"{arm}_s{seed}.jsonl", args.steps)}
            for line in (out_dir / f"{arm}_s{seed}.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r.get("event") == "params":
                    row.update(params=r.get("params"), mhc_params=r.get("mhc_params"))
            table[arm].append(row)
    print(
        f"\n{'arm':>4s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'ms/step':>8s} "
        f"{'tok/s':>8s} {'peak GiB':>8s} {'nan':>4s} {'params':>10s}"
    )
    for arm, runs in table.items():
        for r in runs:
            peak = f"{r['peak_gib']:.2f}" if r["peak_gib"] is not None else "-"
            print(
                f"{arm:>4s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['ms_per_step']:8.1f} {r['tok_s']:8.0f} {peak:>8s} {r['nan']:>4d} "
                f"{r.get('params', '-'):>10}"
            )
    for arm, runs in table.items():
        mean, spread = summarize(runs)
        mhc = sum(r["counters"].get("mhc", 0) for r in runs)
        print(
            f"{arm:>4s} mean bpb {mean:.4f}  spread {spread:.4f}  "
            f"median warm ms/step {statistics.median([r['ms_per_step'] for r in runs]):.1f}  "
            f"mhc_writes={mhc}"
        )
    v = verdict(table["off"], table["on"], window)
    print(
        f"\nvoid checks: every eval scored {window} B, 0 NaN-skipped steps, "
        f"off mhc==0, on mhc>0, {len(seeds)} distinct seeds per arm.\n"
        f"pre-registered rule (AGENTS §1.2): keep the arm iff its mean held-out BPB beats "
        f"the control's mean by more than the CONTROL's own seed spread; a tie deletes the "
        f"mechanism. Verdict: {v['verdict'].upper()} "
        f"(control {v['control_mean']:.4f} +- {v['control_spread']:.4f}, "
        f"mHC {v['arm_mean']:.4f} +- {v['arm_spread']:.4f}, "
        f"delta {v['delta']:+.4f})" + (f"\nvoid reasons: {v['reason']}" if v["reason"] else "")
    )
    print(
        f"step time (review §6.1's SEPARATE verdict, reported not gated): "
        f"median {v['ms_control']:.1f} -> {v['ms_arm']:.1f} ms/step "
        f"({v['ms_delta']:+.1f} ms, {v['ms_delta'] / v['ms_control']:+.1%}); "
        f"the Sinkhorn's launches are invariant to K, so a smaller K is not the lever."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ab", help="the A/B driver (seeds x 2 arms)")
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
    a.add_argument("--out-dir", default="~/logs/ab_mhc")
    r = sub.add_parser("_run", help="one arm, one seed (the child mode)")
    r.add_argument("--arm", required=True, choices=[arm for arm, _f in ARMS])
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
    why = preflight(args.device)
    if why is not None:
        return skip(why)
    return ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
