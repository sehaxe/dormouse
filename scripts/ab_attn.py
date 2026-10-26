"""The attention-arm A/B: compressed-state KDA (control) vs exact causal full, 3 seeds.

    uv run python scripts/ab_attn.py ab            # 3 seeds x 2 arms, 600 steps
    uv run python scripts/ab_attn.py ab --seeds 1  # one seed, dry check
    uv run python scripts/ab_attn.py profile       # ms/step for both arms, no table

**THIS DRIVER HAS NOT BEEN RUN.** The card is held by a training run (AGENTS
§1.5: one heavy thing at a time), so every number below is unmeasured and the
pre-registered rule is written BEFORE the first run. A skipped preflight prints
a SKIP banner and no table.

**The question.** Every number on record ran `attn_kind="kda"` — a LINEAR
attention whose state is compressed before the output is read. The model we
compare against (GPT-2 small, `scripts/bench_hf.py`) runs 12 layers of EXACT
causal attention at 1 024 tokens. `attn_kind="full"` (`model/fullattn.py`)
builds that arm through the same `LoopBlock._attn` seam; this driver prices it
on our own bytes.

**Arms** — ONE flag differs (`attn_kind`), everything else is the battery's
COMMON (`medium`, 600 steps, b8 x s512, bf16 + triton MoE + `moe_capacity=0`
dropless, tf32 + `ns_kind=triton` + `opt_graph=true`, one 81 920 B eval window
= 20 x 8 x 512 on every arm and seed):

* `kda` — the control: `attn_kind="kda"`, the compressed-state arm every
  recorded number ran. Its parameters are `q,k,v,g = 4·d²` plus
  `beta = d·n_heads`; at `medium` 4 210 688.
* `full` — `attn_kind="full"`: exact causal attention, `q,k,v,o = 4·d²`; at
  `medium` 4 194 304. The delta is **-16 384** parameters, i.e. 0.0115% of the
  141.9M model — the two arms are the same size for A/B purposes.

Both arms keep `use_kda=true`: that field is the HARD ON/OFF switch (`false`
is the zero branch, bit-identical to today) and `attn_kind` is what selects
the arm; `config.validate` refuses the pair `use_kda=false + attn_kind="full"`.

**Pre-registered verdict rule (AGENTS §1.2).** Each arm is the mean held-out
BPB over the 3 seeds; a WIN requires the arm's mean to beat the control's mean
by MORE than the CONTROL's own seed spread. Anything else — including a tie —
keeps `kda` (it is the arm every number on record already has, and it is
cheaper by construction). VOID when any run fails a void check: the eval
window is 81 920 B, 0 NaN-skipped steps, `forward>0`, the `full` arm shows
`attn_full_forward>0` and ZERO `kda_fla`/`kda_sdpa`, and the control shows
`attn_full_forward==0`. A void arm is not a number.

**Pre-registered prediction.** `full` scores lower BPB than `kda` at 600
steps — the compressed state is one of the three ranked causes of the GPT-2
gap — but a 600-step, one-file`medium` screen has a seed spread that may be
larger than the effect, so the rule above is the arbiter and no "it looked
better" reading is allowed. COST prediction: `full` is SLOWER per step at
t=512 — exact attention adds `2·t²·d` MAC per iteration on top of the same
`4·t·d²` projections — and `profile` measures that delta instead of guessing.

**`profile` mode.** Runs the SAME trainer for both arms (no eval, no
checkpoint, `log_every=1`) for `--profile-steps` steps, and reports the median
`ms/step` of the second half. It is the same JSONL rate instrument the A/B
uses (`tokens_per_s` at `timers=false`), so the two tables cannot disagree
about what a step costs. This is a cost table, not a verdict.

**Protocol** (`docs/protocols/AB-PROTOCOL.md`): 3 seeds per arm, 600 steps,
each arm-seed a SEPARATE PROCESS (3 seeds is 3 initialisations), one eval
window of 81 920 B on every run, one JSONL per run under `~/logs/ab_attn/`,
the run's own `<name>.config.toml` beside it (a number needs its config,
AGENTS §1.4), and a driver `event=params` row (the instantiated count).

**One deliberate deviation from `ab_mhc.py`'s COMMON, named here:** `mix_plan`
is NOT copied. This brief names one data file (`data_local/mix4/web.bin`);
`mix_plan` REPLACES the single-file stream (`train/loop.py`:
`if run.train.mix_plan: open_mixture(...)`) and would make the named `--data`
inert. `eval_every` is 200, not 500, because the brief is 600 steps: a
500-step cadence would report the step-500 model as the 600-step result (the
driver scores the LAST eval row), and the engram lane's 600-step driver made
the same call.
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

#: The brief's data (AGENTS §2.6: the LOCAL copy) and its held-out file.
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
PROFILE_STEPS = 60

#: `ab_mhc.py`'s COMMON_TRAIN minus `mix_plan` (the docstring's deviation).
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

#: The model-side COMMON (`--set` flags in the battery) plus the two switches
#: this lane holds fixed: `use_kda=true` (the hard on switch — both arms run
#: attention) and the residual arms pinned off so a preset edit cannot move a
#: second mechanism into the comparison.
COMMON_MODEL = dict(
    moe_kernel="triton",
    moe_capacity=0.0,
    bf16=True,
    use_kda=True,
    mhc_streams=2,
    use_attnres=False,
    loom=False,
)

#: The arm names ARE the `attn_kind` values. The control is first so the table
#: reads against it.
ARMS = ("kda", "full")

#: The counters each arm MUST show, and the ones it must NOT: the arm a run
#: names is the one that ran. A full run that also ran KDA is a dispatch bug.
EXPECT = {
    "kda": {"forward": ">0", "attn_full_forward": "==0"},
    "full": {"forward": ">0", "attn_full_forward": ">0", "kda_fla": "==0", "kda_sdpa": "==0"},
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
                why.append(f"seed {r['seed']}: {name}={got} — the wrong arm ran")
    return why


def verdict(control: list[dict], arm: list[dict], window: int) -> dict:
    """The pre-registered rule: WIN iff the arm beats the CONTROL's spread."""
    why = check(control, window, "kda") + check(arm, window, "full")
    cmean, cspread = summarize(control)
    amean, aspread = summarize(arm)
    out = {
        "control_mean": cmean,
        "control_spread": cspread,
        "arm_mean": amean,
        "arm_spread": aspread,
        "delta": amean - cmean,
        "ms_control": statistics.median([r["ms_per_step"] for r in control]),
        "ms_arm": statistics.median([r["ms_per_step"] for r in arm]),
        "reason": "",
    }
    out["ms_delta"] = out["ms_arm"] - out["ms_control"]
    if why:
        out["verdict"] = "void"
        out["reason"] = "; ".join(why)
    elif amean < cmean - cspread:
        out["verdict"] = "win"
    else:
        out["verdict"] = "tie"
    return out


def ms_per_step(tokens_per_s: float, batch: int = BATCH, seq_len: int = SEQ) -> float:
    """`tokens/s` → `ms/step` — the ONE conversion both tables use."""
    return (batch * seq_len) / tokens_per_s * 1e3


def _run_one(args: argparse.Namespace) -> int:
    """One arm, one seed, one process: the trainer with the arm made live."""
    from dormouse.config import load_run
    from dormouse.instrument import loud
    from dormouse.model import DormouseModel
    from dormouse.model.fullattn import FullAttention
    from dormouse.model.kda import KDAAttention, fla_status
    from dormouse.train import train

    run = load_run(
        PRESET,
        model_overrides={**COMMON_MODEL, "attn_kind": args.arm},
        train_overrides={
            **COMMON_TRAIN,
            "seed": args.seed,
            "lr": LR,
            "out_dir": str(Path(args.out).parent),
            "ckpt_name": f"{args.arm}_s{args.seed}",
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

    if args.arm == "kda" and torch.device(device).type == "cuda":
        # The control MUST be the arm every number on record ran. Without FLA
        # it silently becomes the SDPA fallback, a different (slower) algorithm
        # whose BPB would not be the recorded control's.
        ok, why = fla_status(device, recheck=True)
        if not ok:
            loud(
                f"the kda control is not the FLA arm on {device} ({why}). The SDPA "
                f"fallback is a different algorithm, so the table would compare "
                f"exact attention against a fallback. Escape: `uv sync --extra gpu`, "
                f"or --device cpu and read the control as the counted fallback."
            )
    model = DormouseModel(run.model, device=torch.device(device))
    arm_cls = FullAttention if args.arm == "full" else KDAAttention
    assert isinstance(model.loop.attn, arm_cls), (
        f"the arm's dispatch does not match its config: loop.attn is "
        f"{type(model.loop.attn).__name__}, expected {arm_cls.__name__} for "
        f"attn_kind={args.arm!r}"
    )
    attn_params = sum(p.numel() for p in model.loop.attn.parameters())
    total = sum(p.numel() for p in model.parameters())
    window = args.eval_batches * args.batch * args.seq
    print(
        f"[run] arm={args.arm} attn_kind={run.model.attn_kind} seed={args.seed} "
        f"steps={args.steps} b{args.batch} s{args.seq} lr={LR} window={window} B "
        f"params={total:,} attn_params={attn_params:,}",
        flush=True,
    )
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train(model, run, device=torch.device(device), log=print)

    def _row(**kw) -> None:
        with open(args.out, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(kw) + "\n")

    params = {"params": total, "attn_params": attn_params}
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
        f"pre-registered: the full arm lands ONLY if its mean held-out BPB beats the "
        f"control's by more than the CONTROL's own seed spread; a tie keeps kda. "
        f"Cost is reported, not gated."
    )
    for arm in ARMS:
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

    table = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for seed in seeds:
            row = {"seed": seed, **read_run(out_dir / f"{arm}_s{seed}.jsonl", args.steps)}
            for line in (out_dir / f"{arm}_s{seed}.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r.get("event") == "params":
                    row.update(params=r.get("params"), attn_params=r.get("attn_params"))
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
        print(
            f"{arm:>4s} mean bpb {mean:.4f}  spread {spread:.4f}  "
            f"median warm ms/step {statistics.median([r['ms_per_step'] for r in runs]):.1f}  "
            f"attn_params={runs[0].get('attn_params', '-')}"
        )
    v = verdict(table["kda"], table["full"], window)
    print(
        f"\nvoid checks: every eval scored {window} B, 0 NaN-skipped steps, "
        f"kda attn_full_forward==0, full attn_full_forward>0 with no KDA calls, "
        f"{len(seeds)} distinct seeds per arm.\n"
        f"pre-registered rule (AGENTS §1.2): keep the full arm iff its mean held-out "
        f"BPB beats the control's mean by more than the CONTROL's own seed spread; a "
        f"tie keeps the cheaper kda arm. Verdict: {v['verdict'].upper()} "
        f"(control {v['control_mean']:.4f} +- {v['control_spread']:.4f}, "
        f"full {v['arm_mean']:.4f} +- {v['arm_spread']:.4f}, "
        f"delta {v['delta']:+.4f})" + (f"\nvoid reasons: {v['reason']}" if v["reason"] else "")
    )
    print(
        f"step time (reported, NOT folded into the BPB rule): median "
        f"{v['ms_control']:.1f} -> {v['ms_arm']:.1f} ms/step "
        f"({v['ms_delta']:+.1f} ms, {v['ms_delta'] / v['ms_control']:+.1%}); "
        f"`profile` reruns both arms for the warm-step cost."
    )
    return 0


def profile(args: argparse.Namespace) -> int:
    """`ms/step` for both arms: the same trainer, no eval, no table/verdict."""
    out_dir = Path(args.out_dir).expanduser() / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    import torch

    rows: dict[str, float] = {}
    for arm in ARMS:
        from dormouse.config import load_run
        from dormouse.model import DormouseModel
        from dormouse.train import train

        out = out_dir / f"{arm}.jsonl"
        out.unlink(missing_ok=True)
        run = load_run(
            PRESET,
            model_overrides={**COMMON_MODEL, "attn_kind": arm},
            train_overrides={
                **COMMON_TRAIN,
                "steps": args.profile_steps,
                "eval": False,
                "eval_every": 10**9,
                "log_every": 1,
                "ckpt_every": 10**9,
                "ckpt_name": f"profile_{arm}",
                "out_dir": str(out_dir),
                "timers": False,
                "json_log": str(out),
            },
        )
        run.train.data = args.data
        if not Path(run.train.data).is_file():
            print(f"no data file at {run.train.data}: pass --data")
            return 1
        model = DormouseModel(run.model, device=torch.device(device))
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print(f"\n=== profile {arm} ({args.profile_steps} steps, {device})", flush=True)
        train(model, run, device=torch.device(device), log=print)
        rates = [
            r["tokens_per_s"]
            for r in (json.loads(line) for line in out.read_text().splitlines() if line.strip())
            if r.get("event") == "log"
            and r.get("step", 0) > args.profile_steps / 2
            and r.get("tokens_per_s")
        ]
        assert rates, f"{out}: no warm-step rate rows — nothing measured"
        rows[arm] = ms_per_step(statistics.median(rates))
    base = rows["kda"]
    print(f"\n{'arm':>4s} {'ms/step':>10s} {'vs kda':>10s}")
    for arm, ms in rows.items():
        print(f"{arm:>4s} {ms:10.1f} {ms - base:+9.1%}")
    print(
        f"exact attention costs {rows['full'] - base:+.1f} ms/step at "
        f"b{args.batch} s{args.seq} on {device} — measured, not guessed."
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
    a.add_argument("--out-dir", default="~/logs/ab_attn")
    p = sub.add_parser("profile", help="ms/step for both arms (no verdict)")
    p.add_argument("--profile-steps", type=int, default=PROFILE_STEPS)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--seq", type=int, default=SEQ)
    p.add_argument("--data", default=DATA)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="~/logs/ab_attn")
    r = sub.add_parser("_run", help="one arm, one seed (the child mode)")
    r.add_argument("--arm", required=True, choices=ARMS)
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
    return ab(args) if args.mode == "ab" else profile(args)


if __name__ == "__main__":
    raise SystemExit(main())
