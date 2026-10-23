"""The prelude/coda A/B: full-stack loop (control) vs unique layers outside it.

    uv run python scripts/ab_prelude.py ab            # 3 seeds x 3 arms, 600 steps
    uv run python scripts/ab_prelude.py ab --seeds 1  # one seed, dry check

**THIS DRIVER HAS NOT BEEN RUN.** The card is held by a training run (AGENTS
§1.5: one heavy thing at a time), so every number below is unmeasured and the
pre-registered rule is written BEFORE the first run. A skipped preflight prints
a SKIP banner and no table.

**Arms** — one pair of flags differs (`prelude_layers`/`coda_layers`),
everything else is the battery's own COMMON (`medium`, b8 x s512, bf16 + triton
MoE + `moe_capacity=0` dropless + host Engram, tf32 + `ns_kind=triton` +
`opt_graph=true`, one 81 920 B eval window = 20 x 8 x 512 on every arm and seed):

* `base(0,0)` — the control, today's full-stack loop: `embed → LoopBlock ×4 →
  RMSNorm → lm_head`;
* `pc(1,1)` — one UNIQUE prelude layer before the loop and one coda after it
  (`arXiv 2610.23036`; `model/prelude.py`), +3 147 776 params at this geometry;
* `pc(2,2)` — two each, +6 295 552 params — the depth check on the topology.

The unique layer is a pre-norm residual SwiGLU FFN at `d_ff = d/2`, NOT a second
KDA: a KDA copy is 4 210 688 params at d=1024, so 1+1 would add 8.4M (outside
the +2-6M envelope) and put a second attention arm in the model. `config.validate`
refuses `use_byteflow` with either stack; the rest of the tree is untouched.

**Pre-registered prediction.** `pc(1,1)` TIES or barely beats the control, and
any win is confounded with +2.2% parameters — the arms do not hold the parameter
count fixed, so the honest reading of a small win is "more capacity", not "the
looped topology". A win has to beat the control's own seed spread (AGENTS §1.2);
a tie deletes the mechanism. `pc(2,2)` is the scaling check: if the gain keeps
growing with the number of unique layers beyond the control's spread, the
mechanism being measured is parameter count.

**Pre-registered verdict rule.**

* VOID when any run fails a void check: eval window != 81 920 B, any NaN-skipped
  step, `base` reports `prelude>0`/`coda>0` (the wrong shape ran), a `pc` arm
  reports either as 0 (the arm did not run), or fewer than 3 seeds per arm.
* KEEP the topology iff `pc(1,1)`'s mean held-out BPB beats the control's mean
  by MORE than the CONTROL's own 3-seed spread — and `pc(2,2)` does not fall
  behind `pc(1,1)` by more than that same spread (a win that only grows with
  added parameters is the capacity confound, not the topology).
* Otherwise TIE, and a tie deletes the mechanism (AGENTS §1.2).
* `ms/step` is REPORTED, never folded into the BPB rule: the prelude/coda add
  ~2-4% parameters and run once per forward, so the step-time question is
  separate from the quality question.

**Protocol** (`docs/protocols/AB-PROTOCOL.md`): 3 seeds per arm, 600 steps, each
arm-seed a SEPARATE PROCESS (3 seeds is 3 initialisations), one eval window of
81 920 B on every run, output one JSONL per run under `~/logs/ab_prelude/` plus
the run's own `<name>.config.toml` (a number needs its config, AGENTS §1.4) and
a driver `event=params` row (the instantiated count and the unique-layer share).

Two deliberate deviations from `ab_mhc.py`'s COMMON, both forced by this
brief and both named here:

1. `mix_plan` is NOT copied. The brief names one data file
   (`data_local/mix4/web.bin`); `mix_plan` REPLACES the single-file stream
   (`train/loop.py`: `if run.train.mix_plan: open_mixture(...)`) and would make
   the named `--data` inert.
2. `eval_every` is 200, not 500: the brief is 600 steps, and a 500-step cadence
   would report the step-500 model as the 600-step result (the driver scores the
   LAST eval row).
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

#: The brief's data (AGENTS §2.6: the LOCAL copy, so the driver does not depend
#: on the /mnt drive being mounted) and its held-out file.
DATA = "/home/sehaxe/dormouse/data_local/mix4/web.bin"
EVAL = "/home/sehaxe/dormouse/data_local/mix4/web_eval.bin"

SEEDS = (1, 2, 3)
STEPS = 600
PRESET = "medium"
BATCH = 8
SEQ = 512
EVAL_BATCHES = 20
EVAL_EVERY = 200
LOG_EVERY = 250
LR = 3e-4  # TrainConfig's default; the battery does not override it

#: `ab_mhc.py`'s COMMON_TRAIN minus `mix_plan` (deviation 1 in the docstring):
#: every other entry is the battery's, and the only model-side difference is the
#: arm's own two fields below.
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

#: `(arm name, prelude_layers, coda_layers)`. The control is first so the table
#: reads against it.
ARMS = (
    ("base", 0, 0),
    ("pc11", 1, 1),
    ("pc22", 2, 2),
)

#: The counters each arm MUST show, and the ones it must NOT: an arm that runs
#: and writes nothing is the defect class the counters exist for.
EXPECT = {
    "base": {"forward": ">0", "prelude": "==0", "coda": "==0"},
    "pc11": {"forward": ">0", "prelude": ">0", "coda": ">0"},
    "pc22": {"forward": ">0", "prelude": ">0", "coda": ">0"},
}


def check(runs: list[dict], window: int, arm: str) -> list[str]:
    """The void checks: a number that fails one is not a number."""
    why: list[str] = []
    for r in runs:
        if r["scored"] != window:
            why.append(f"seed {r['seed']}: eval window {r['scored']} != {window} B")
        if r["nan"]:
            why.append(f"seed {r['seed']}: {r['nan']} NaN-skipped steps")
        for name, want in EXPECT[arm].items():
            got = r["counters"].get(name, 0)
            if want == ">0" and got <= 0:
                why.append(f"seed {r['seed']}: {name}={got} — the arm did not run")
            if want == "==0" and got != 0:
                why.append(f"seed {r['seed']}: {name}={got} — the wrong shape ran")
    return why


def verdict(control: list[dict], arm: list[dict], window: int, arm_name: str) -> dict:
    """AGENTS §1.2 over one control and one arm, plus the reported step-time delta."""
    why = check(control, window, "base") + check(arm, window, arm_name)
    cmean, cspread = summarize(control)
    amean, aspread = summarize(arm)
    cms = statistics.median([r["ms_per_step"] for r in control])
    ams = statistics.median([r["ms_per_step"] for r in arm])
    out = {
        "arm": arm_name,
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


def decide(v11: dict, v22: dict) -> str:
    """The pre-registered rule over the two arms (module docstring)."""
    if v11["verdict"] == "void" or v22["verdict"] == "void":
        return "void"
    if v11["verdict"] != "win":
        return "tie" if v11["verdict"] == "tie" else "lose"
    # A `pc(2,2)` that falls behind `pc(1,1)` by more than the control's own
    # spread is a depth regression; a fall AHEAD is the capacity confound.
    if v22["arm_mean"] > v11["arm_mean"] + v11["control_spread"]:
        return "capacity-confounded"
    return "keep"


def _run_one(args: argparse.Namespace) -> int:
    """One arm, one seed, one process: the trainer with the layer flags made live."""
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train import train

    pre, post = next((p, c) for name, p, c in ARMS if name == args.arm)
    run = load_run(
        PRESET,
        model_overrides={
            **COMMON_MODEL,
            "prelude_layers": pre,
            "coda_layers": post,
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
    assert len(model.prelude) == pre and len(model.coda) == post, (
        "the arm's dispatch does not match its flags: "
        f"built {len(model.prelude)}/{len(model.coda)}, wanted {pre}/{post}"
    )
    pre_params = sum(p.numel() for p in model.prelude.parameters())
    post_params = sum(p.numel() for p in model.coda.parameters())
    total = sum(p.numel() for p in model.parameters())
    window = args.eval_batches * args.batch * args.seq
    print(
        f"[run] arm={args.arm} prelude={pre} coda={post} seed={args.seed} "
        f"steps={args.steps} b{args.batch} s{args.seq} lr={LR} window={window} B "
        f"params={total:,} unique_params={pre_params + post_params:,} "
        f"({(pre_params + post_params) / total:.4%})",
        flush=True,
    )
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train(model, run, device=torch.device(device), log=print)

    def _row(**kw) -> None:
        with open(args.out, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(kw) + "\n")

    params = {
        "params": total,
        "prelude_params": pre_params,
        "coda_params": post_params,
    }
    if torch.device(device).type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        params["peak_gib"] = peak
        print(f"[peak] {peak:.2f} GiB (torch.cuda.max_memory_allocated)", flush=True)
    _row(event="params", step=args.steps, **params)
    return 0


def ab(args: argparse.Namespace) -> int:
    """The driver: `--seeds` x 3 arms, separate processes, then the table."""
    seeds = tuple(int(s) for s in args.seeds.split(","))
    window = args.eval_batches * args.batch * args.seq
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"A/B: {len(ARMS)} arms x {len(seeds)} seeds x {args.steps} steps, {PRESET} "
        f"b{args.batch} s{args.seq} lr={LR}, window {window} B. Output: {out_dir}\n"
        f"pre-registered: pc(1,1) ties or barely beats the control, and any win is "
        f"confounded with +2.2% parameters (the arms do not hold params fixed); "
        f"pc(2,2) is the scaling check. A tie deletes the mechanism."
    )
    for arm, _pre, _post in ARMS:
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

    table = {arm: [] for arm, _p, _c in ARMS}
    for arm, _pre, _post in ARMS:
        for seed in seeds:
            row = {"seed": seed, **read_run(out_dir / f"{arm}_s{seed}.jsonl", args.steps)}
            for line in (out_dir / f"{arm}_s{seed}.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r.get("event") == "params":
                    row.update(
                        params=r.get("params"),
                        prelude_params=r.get("prelude_params"),
                        coda_params=r.get("coda_params"),
                    )
            table[arm].append(row)
    print(
        f"\n{'arm':>5s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'ms/step':>8s} "
        f"{'tok/s':>8s} {'peak GiB':>8s} {'nan':>4s} {'params':>10s} {'unique':>8s}"
    )
    for arm, runs in table.items():
        for r in runs:
            peak = f"{r['peak_gib']:.2f}" if r["peak_gib"] is not None else "-"
            unique = (r.get("prelude_params") or 0) + (r.get("coda_params") or 0)
            print(
                f"{arm:>5s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['ms_per_step']:8.1f} {r['tok_s']:8.0f} {peak:>8s} {r['nan']:>4d} "
                f"{r.get('params', 0):>10} {unique:>8}"
            )
    v11 = verdict(table["base"], table["pc11"], window, "pc11")
    v22 = verdict(table["base"], table["pc22"], window, "pc22")
    for arm, runs in table.items():
        mean, spread = summarize(runs)
        pre = sum(r["counters"].get("prelude", 0) for r in runs)
        post = sum(r["counters"].get("coda", 0) for r in runs)
        print(
            f"{arm:>5s} mean bpb {mean:.4f}  spread {spread:.4f}  "
            f"median warm ms/step {statistics.median([r['ms_per_step'] for r in runs]):.1f}  "
            f"prelude_writes={pre} coda_writes={post}"
        )
    outcome = decide(v11, v22)
    print(
        f"\nvoid checks: every eval scored {window} B, 0 NaN-skipped steps, "
        f"base prelude/coda==0, both pc arms prelude/coda>0, {len(seeds)} distinct "
        f"seeds per arm.\n"
        f"pre-registered rule (AGENTS §1.2 + the capacity confound above): KEEP iff "
        f"pc(1,1)'s mean held-out BPB beats the control's mean by more than the "
        f"CONTROL's own seed spread, and pc(2,2) does not fall behind pc(1,1) by "
        f"more than that spread; otherwise TIE, and a tie deletes the mechanism.\n"
        f"pc(1,1): {v11['verdict'].upper()} "
        f"(control {v11['control_mean']:.4f} +- {v11['control_spread']:.4f}, "
        f"arm {v11['arm_mean']:.4f} +- {v11['arm_spread']:.4f}, "
        f"delta {v11['delta']:+.4f})"
        + (f"\nvoid reasons: {v11['reason']}" if v11["reason"] else "")
        + f"\npc(2,2): {v22['verdict'].upper()} "
        f"({v22['arm_mean']:.4f} +- {v22['arm_spread']:.4f}, "
        f"delta {v22['delta']:+.4f})"
        + (f"\nvoid reasons: {v22['reason']}" if v22["reason"] else "")
        + f"\nTOPOLOGY VERDICT: {outcome.upper()}"
    )
    print(
        f"step time (SEPARATE, reported not gated): pc(1,1) median "
        f"{v11['ms_control']:.1f} -> {v11['ms_arm']:.1f} ms/step "
        f"({v11['ms_delta']:+.1f} ms, {v11['ms_delta'] / v11['ms_control']:+.1%}); "
        f"pc(2,2) {v22['ms_arm']:.1f} ms/step ({v22['ms_delta']:+.1f})."
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
    a.add_argument("--eval", default=EVAL)
    a.add_argument("--device", default="cuda")
    a.add_argument("--out-dir", default="~/logs/ab_prelude")
    r = sub.add_parser("_run", help="one arm, one seed (the child mode)")
    r.add_argument("--arm", required=True, choices=[arm for arm, _p, _c in ARMS])
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
