"""The four train4 arms: control vs +accum / +ema / +zl / +tie, 3 seeds x 600.

    uv run python scripts/ab_train4.py ab             # 3 seeds x 5 arms
    uv run python scripts/ab_train4.py ab --seeds 1   # one seed, dry check

**THIS DRIVER HAS NOT BEEN RUN.** The task that wrote it named the arms, the
budget and the verdict BEFORE the first run; every number below is unmeasured
until the table exists. A skipped preflight prints a SKIP banner and no table.

**Arms** — exactly ONE statement differs from the control, except `+accum`,
whose graph-arm override is named below:

* `control` — nothing set;
* `+accum` — `grad_accum=2` (effective batch 2x at the SAME 600 optimizer
  steps, so the arm sees twice the tokens: a BPB win is a token-budget result,
  not a free quality win);
* `+ema` — `ema_decay=0.9`; eval reads the shadow, the checkpoint writes it
  under `model_ema`;
* `+zl` — `z_loss=1e-4` (PaLM §5 / Gemma-2 §9's order; the term is
  `coef·mean(logsumexp²)` on the final logits);
* `+tie` — `tie_embeddings=true`; on `medium` (`use_tsct=true`) the tied head
  REPLACES the low-rank TSCT head, which is a second difference a reader of a
  +tie number must know (on `use_tsct=false` tying is the exact head matrix).

**One COMMON deviation, applied to EVERY arm:** `opt_graph=false`. The battery's
COMMON captures the optimizer stage; `grad_accum>1` is a LOUD refusal with that
arm (the graph captures ONE stage, the accumulation hosts k eager backwards
around it), and leaving it on for four arms while turning it off for one would
put the graph in every comparison but `+accum`. All five run the eager
optimizer, so the arms differ in one statement each.

The rest is `ab_mhc.py`'s COMMON: `medium`, 600 steps, b8 x s512, bf16, triton
MoE, `moe_capacity=0.0` (the battery's dropless mode: no row is dropped by
another row's presence, so `+accum`'s micro-forwards are the wide forward's
rows), host Engram, tf32 + triton NS, one 81 920 B eval window (20 x 8 x 512)
on every run.

**Data** — `data_local/mix4/web.bin` and its held-out `web_eval.bin`; the same
two files for every arm and seed, so BPB is comparable within the battery.

**Pre-registered verdict.** All four are predicted TIES at 600 steps:

* `+accum` — the optimizer arithmetic is gated (`tests/test_train4.py`), but
  the same 600 optimizer steps see 2x the tokens; if BPB improves, the honest
  reading is "more tokens", and the arm's case is peak memory at a fixed
  *token* budget, not a quality mechanism;
* `+ema` — a 0.9 shadow averages roughly the last 10 steps; at 600 steps that
  is a small smoothing, expected inside the seed spread;
* `+zl` — a stability regulariser with no claimed quality win at this budget;
* `+tie` — a parameter/throughput lever; a quality claim would need a
  parameter-matched control, which this battery is not.

The rule is `docs/protocols/AB-PROTOCOL.md`/AGENTS §1.2: VOID when any run
misses the window, skips a NaN step, or fails its own arm counter; otherwise
WIN iff an arm's mean held-out BPB beats the CONTROL's mean by more than the
CONTROL's own seed spread. Anything else — a tie included — means the quality
claim is not there. A tie does not delete `+accum`/`+tie` as resource levers;
it deletes any quality claim made for them.

Output: one JSONL per run under `~/logs/ab_train4/` plus the run's own
`<name>.config.toml`, and a driver `event=params` row (instantiated count, the
head's share, `opt_steps`, and which arm wiring was live).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

# `scripts/` is not a package: reuse the battery's preflight/readers and its
# COMMON rather than a second copy of either.
_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.append(_SCRIPTS)
from _lane import preflight, skip  # noqa: E402
from ab_byteflow import read_run, summarize  # noqa: E402
from ab_mhc import COMMON_MODEL, COMMON_TRAIN  # noqa: E402

DATA = "/home/sehaxe/dormouse/data_local/mix4/web.bin"
EVAL = "/home/sehaxe/dormouse/data_local/mix4/web_eval.bin"

SEEDS = (1, 2, 3)
STEPS = 600
PRESET = "medium"
BATCH = 8
SEQ = 512
EVAL_BATCHES = 20
EVAL_EVERY = 300
LOG_EVERY = 100
LR = 3e-4
Z_LOSS = 1e-4
EMA_DECAY = 0.9

#: The battery's train COMMON, minus the graph arm (see the module docstring)
#: and with the single-file mix4 source instead of `mix_big`.
COMMON_TRAIN_LOCAL = dict(
    COMMON_TRAIN,
    opt_graph=False,
    mix_plan="",
    eval_every=EVAL_EVERY,
    log_every=LOG_EVERY,
    steps=STEPS,
    batch=BATCH,
    seq_len=SEQ,
    eval_batches=EVAL_BATCHES,
)

#: The one statement each arm adds. The control is the zero override.
ARMS = ("control", "+accum", "+ema", "+zl", "+tie")

TRAIN_ARM = {
    "control": {},
    "+accum": {"grad_accum": 2},
    "+ema": {"ema_decay": EMA_DECAY},
    "+zl": {},
    "+tie": {},
}
MODEL_ARM = {
    "control": {},
    "+accum": {},
    "+ema": {},
    "+zl": {"z_loss": Z_LOSS},
    "+tie": {"tie_embeddings": True},
}

#: What a void check must see on the counter line (or the params row).
EXPECT = {
    "control": {"z_loss": "==0"},
    "+accum": {"z_loss": "==0"},
    "+ema": {"z_loss": "==0"},
    "+zl": {"z_loss": ">0"},
    "+tie": {"z_loss": "==0"},
}


def void(runs: list[dict], window: int, arm: str) -> list[str]:
    """A run that fails one of these is not a number (the battery's rule)."""
    why: list[str] = []
    for r in runs:
        if r["scored"] != window:
            why.append(f"{arm} seed {r['seed']}: eval window {r['scored']} != {window} B")
        if r["nan"]:
            why.append(f"{arm} seed {r['seed']}: {r['nan']} NaN-skipped steps")
        for name, want in EXPECT[arm].items():
            got = r["counters"].get(name, 0)
            if want == ">0" and got <= 0:
                why.append(f"{arm} seed {r['seed']}: {name}={got} — the arm did not run")
            if want == "==0" and got != 0:
                why.append(f"{arm} seed {r['seed']}: {name}={got} — the wrong shape ran")
        p = r.get("params") or {}
        if arm == "+accum" and p.get("grad_accum") != 2:
            why.append(f"{arm} seed {r['seed']}: grad_accum={p.get('grad_accum')} != 2")
        if arm == "+ema" and not p.get("ema"):
            why.append(f"{arm} seed {r['seed']}: no EMA shadow was written")
        if arm == "+tie" and p.get("lm_head_params") != 0:
            why.append(
                f"{arm} seed {r['seed']}: lm_head still has {p.get('lm_head_params')} params"
            )
        if arm != "+tie" and (p.get("lm_head_params") or 0) <= 0:
            why.append(f"{arm} seed {r['seed']}: the untied head has no params")
    return why


def verdict(control: list[dict], arm: list[dict], window: int, name: str) -> dict:
    """AGENTS §1.2 over the control and ONE arm; the timing delta is reported."""
    why = void(control, window, "control") + void(arm, window, name)
    cmean, cspread = summarize(control)
    amean, aspread = summarize(arm)
    cms = statistics.median([r["ms_per_step"] for r in control])
    ams = statistics.median([r["ms_per_step"] for r in arm])
    out = {
        "arm": name,
        "control_mean": cmean,
        "control_spread": cspread,
        "arm_mean": amean,
        "arm_spread": aspread,
        "delta": amean - cmean,
        "ms_control": cms,
        "ms_arm": ams,
        "ms_delta": ams - cms,
        "reason": "; ".join(why),
    }
    out["verdict"] = "void" if why else ("win" if amean < cmean - cspread else "tie")
    return out


def _run_one(args: argparse.Namespace) -> int:
    """One arm, one seed, one process: the trainer with that arm made live."""
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train import train

    run = load_run(
        PRESET,
        model_overrides={**COMMON_MODEL, **MODEL_ARM[args.arm]},
        train_overrides={
            **COMMON_TRAIN_LOCAL,
            **TRAIN_ARM[args.arm],
            "seed": args.seed,
            "lr": LR,
            "out_dir": str(Path(args.out).parent),
            "ckpt_name": f"{args.arm.lstrip('+')}_s{args.seed}",
            "timers": False,
            "json_log": str(args.out),
            "data": args.data,
            "eval_data": args.eval,
        },
    )
    if not Path(run.train.data).is_file():
        print(f"no data file at {run.train.data}: pass --data")
        return 1
    Path(args.out).with_suffix(".config.toml").write_text(run.snapshot_toml())

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    import torch

    model = DormouseModel(run.model, device=torch.device(device))
    total = sum(p.numel() for p in model.parameters())
    head_params = sum(p.numel() for p in model.lm_head.parameters())
    # The wiring assertions: an arm that is not live in the instantiated model
    # is a void run, not a slow one (ADR-0019).
    assert (run.train.grad_accum == 2) == (args.arm == "+accum"), args.arm
    assert (run.train.ema_decay > 0) == (args.arm == "+ema"), args.arm
    assert (run.model.z_loss > 0) == (args.arm == "+zl"), args.arm
    assert (run.model.tie_embeddings is True) == (args.arm == "+tie"), args.arm
    if args.arm == "+tie":
        assert head_params == 0, head_params
        assert "embed.weight" in model.state_dict()
        assert not any(k.startswith("lm_head.") for k in model.state_dict()), (
            "the tied head must not register a second parameter key"
        )
    window = args.eval_batches * args.batch * args.seq
    print(
        f"[run] arm={args.arm} seed={args.seed} steps={args.steps} "
        f"b{args.batch} s{args.seq} lr={LR} window={window} B params={total:,} "
        f"lm_head_params={head_params:,} grad_accum={run.train.grad_accum} "
        f"ema_decay={run.train.ema_decay:g} z_loss={run.model.z_loss:g}",
        flush=True,
    )
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    report = train(model, run, device=torch.device(device), log=print)

    def _row(**kw) -> None:
        with open(args.out, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(kw) + "\n")

    params = {
        "params": total,
        "lm_head_params": head_params,
        "grad_accum": run.train.grad_accum,
        "ema": bool(run.train.ema_decay > 0),
        "ema_decay": run.train.ema_decay,
        "z_loss": run.model.z_loss,
        "tie": bool(run.model.tie_embeddings),
        "opt_steps": report["opt_steps"],
    }
    if torch.device(device).type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        params["peak_gib"] = peak
        print(f"[peak] {peak:.2f} GiB (torch.cuda.max_memory_allocated)", flush=True)
    _row(event="params", step=args.steps, **params)
    return 0


def ab(args: argparse.Namespace) -> int:
    """The driver: `--seeds` x 5 arms, separate processes, then the table."""
    why = preflight(args.device)
    if why:
        return skip(why)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    window = args.eval_batches * args.batch * args.seq
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"A/B: {len(ARMS)} arms x {len(seeds)} seeds x {args.steps} steps, {PRESET} "
        f"b{args.batch} s{args.seq} lr={LR}, window {window} B. Output: {out_dir}\n"
        f"pre-registered: every arm TIES at 600 steps; +accum sees 2x tokens at "
        f"the same step count, +ema is a ~10-step average, +zl is a stability "
        f"term, +tie is a resource lever. `opt_graph=false` on EVERY arm (the "
        f"battery COMMON's graph is refused with grad_accum>1)."
    )
    for arm in ARMS:
        for seed in seeds:
            out = out_dir / f"{arm.lstrip('+')}_s{seed}.jsonl"
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
            path = out_dir / f"{arm.lstrip('+')}_s{seed}.jsonl"
            row = {"seed": seed, **read_run(path, args.steps)}
            for line in path.read_text().splitlines():
                r = json.loads(line)
                if r.get("event") == "params":
                    row["params"] = {
                        k: r.get(k)
                        for k in (
                            "params",
                            "lm_head_params",
                            "grad_accum",
                            "ema",
                            "ema_decay",
                            "z_loss",
                            "tie",
                            "opt_steps",
                        )
                    }
                    row["peak_gib"] = r.get("peak_gib")
            # `read_run` derives ms/step from `batch·seq_len` (the shared
            # battery's shape); a +accum optimizer step moved `batch·seq·k`
            # tokens, so the ONE number it cannot know is scaled here.
            row["ms_per_step"] *= (row.get("params") or {}).get("grad_accum") or 1
            table[arm].append(row)

    print(
        f"\n{'arm':>7s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'ms/step':>8s} "
        f"{'tok/s':>8s} {'nan':>4s} {'opt_steps':>9s} {'params':>10s} {'head':>8s}"
    )
    for arm in ARMS:
        for r in table[arm]:
            p = r.get("params") or {}
            print(
                f"{arm:>7s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['ms_per_step']:8.1f} {r['tok_s']:8.0f} {r['nan']:>4d} "
                f"{p.get('opt_steps', '-'):>9} {p.get('params', '-'):>10} "
                f"{p.get('lm_head_params', '-'):>8}"
            )
    print()
    results = []
    for arm in ARMS[1:]:
        v = verdict(table["control"], table[arm], window, arm)
        results.append(v)
        print(
            f"{arm:>7s} mean bpb {v['arm_mean']:.4f} +- {v['arm_spread']:.4f}  "
            f"delta {v['delta']:+.4f} vs control {v['control_mean']:.4f} "
            f"+- {v['control_spread']:.4f}  "
            f"ms/step {v['ms_control']:.1f} -> {v['ms_arm']:.1f} "
            f"({v['ms_delta']:+.1f})  VERDICT {v['verdict'].upper()}"
            + (f"\n         void: {v['reason']}" if v["reason"] else "")
        )
    print(
        f"\npre-registered rule (AGENTS §1.2): keep a quality claim iff an arm's "
        f"mean held-out BPB beats the control's mean by MORE than the control's "
        f"own seed spread; a tie deletes the quality claim. +accum/+tie are "
        f"resource levers, so their tie is not a deletion, only a refusal to "
        f"claim quality. {len(seeds)} seeds per arm, one eval window "
        f"({window} B), 0 NaN-skipped steps required."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ab", help="the A/B driver (seeds x 5 arms)")
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
    a.add_argument("--out-dir", default="~/logs/ab_train4")
    a.set_defaults(func=ab)
    r = sub.add_parser("_run", help="one arm x one seed (internal)")
    r.add_argument("--arm", required=True, choices=list(ARMS))
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
    r.set_defaults(func=_run_one)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
