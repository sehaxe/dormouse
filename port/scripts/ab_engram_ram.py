"""The host-RAM Engram bias-correction A/B: VRAM / RAM-old-formula / RAM-corrected.

    uv run python scripts/ab_engram_ram.py ab            # 3 arms x 3 seeds
    uv run python scripts/ab_engram_ram.py ab --seeds 1  # one seed, dry check

**THIS DRIVER HAS NOT BEEN RUN.** The card is held by a training run (AGENTS
§1.5: one heavy thing at a time), so every number below is unmeasured and the
pre-registered rule is written BEFORE the first run. A skipped preflight prints
a SKIP banner and no table.

**The finding this tests.** A 600-step `medium` screen, one seed, everything
else identical (`docs/reviews/2026-10-23-architecture-verdict.md` §3.1):
`engram` in VRAM scored **2.649 BPB**, the same arm with `engram_ram=true`
scored **3.476** (+0.83), and iter3 reproduced the pattern. The mechanism is
the host path's Nesterov look-ahead: for a row's first touch, with
`m = μ·0 + (1-μ)·g`, the applied block is
`μ·m + (1-μ)·g = (1-μ²)·g = 0.0975·g` at `μ = 0.95` — a **10.3× smaller step**
than the gradient. A row needs ~`1/(1-μ) = 20` touches before the block
approaches `g`; a 600-step hash table is mostly rows that never get there. The
formula is burn's verbatim (`crates/dormouse-train/src/offload.rs:154-160`),
tuned for a 48M-row / 300k-step production run, not for our screens.

**The fix under test.** A per-row touch counter and an Adam-style correction of
the look-ahead's transient: on touch `t` the block is divided by
`1 - μ^(t+1)` (the look-ahead contains `t+1` exponentially weighted terms, so
that — not `1-μ^t` — is its total weight). A cold row's first touch is then
exactly `g`; a row with `t ≳ 300` has a divisor of `1.0` in fp32 and is
bit-unchanged. The counter is per ROW, not the global step: a global step
over-corrects a row first touched at step 500.

**Arms** — one pair of flags differs; everything else is the battery's COMMON
(`medium`, 600 steps, b8 x s512, bf16 + triton MoE + `moe_capacity=0` dropless,
tf32 + `ns_kind=triton` + `opt_graph=true`, one 81 920 B eval window = 20 x 8 x
512 on every arm and seed):

* `vram` — the control: `engram_ram=false`, the memory in the parameter table.
  This is the 2.649 arm; the bias correction cannot reach it (it is host-only).
* `ram_old_formula` — `engram_ram=true, engram_bias_correction=false`: burn's
  formula line for line, i.e. the 3.476 arm.
* `ram_corrected` — `engram_ram=true, engram_bias_correction=true` (the new
  default): the same tables, the same transfers, the correction on.

**Pre-registered prediction.** `ram_old_formula` reproduces the gap
(≈3.4-3.5 BPB against ≈2.65 for `vram`), and `ram_corrected` closes it to
within the control's own seed spread. The prediction is the algebra above, not
a hope: the host path's only arithmetic difference from the VRAM path is this
update (the forward reads the same fp32 rows through the same `_read`), so a
remaining gap after the correction falsifies the "cold rows" explanation.

**Pre-registered verdict rule.** Each arm is scored by its mean held-out BPB
over the 3 seeds and its own seed spread (AGENTS §1.2). With
`thr_old = max(vram_spread, old_spread)` and
`thr_new = max(vram_spread, new_spread)`:

* **VOID** when any run fails a void check (eval window, NaN, counters, the
  `engram_ram`/`engram_bias_correction` flags the run actually resolved, or
  fewer than 3 seeds per arm). A void arm is not a number.
* **HELD** iff the gap reproduces (`old_mean > vram_mean + thr_old`), the
  correction closes it (`new_mean <= vram_mean + thr_new`), and the correction
  moves the number (`new_mean < old_mean - max(old_spread, new_spread)`).
  Then the corrected formula ships on, and the host path is trustworthy at this
  budget.
* **INSUFFICIENT** iff the gap reproduces and the correction does not close it:
  cold rows are not the whole story; the host path stays parked for production
  and the engram stays in VRAM at `medium`/`b1` until the remainder is named.
* **NO-EFFECT** iff the gap reproduces, the correction closes it, but the
  correction itself is not what moved it (the two RAM arms tie).
* **NOT-NEEDED** iff the 600-step window does not reproduce the gap at all
  (`old` within spread of `vram`) and `new` is also within spread. The reported
  0.83 BPB then does not transfer to this data/window; the correction stays on
  because it is the algebraically correct transient, but no b1 claim is earned.
* **HARMFUL** iff `new` is worse than `vram` by more than `thr_new` and is not
  better than `old` by more than the spread.

A TIE between `ram_corrected` and `ram_old_formula` deletes the correction
(AGENTS §1.2: a tie deletes the mechanism), and the operator should then
re-examine the premise rather than bank the 600-step number.

**Void checks that are specific to this lane.** `vram` must show
`engram_host_updates == 0` and the two RAM arms `> 0` (the substrate the arm
names is the one that ran); every arm must show `engram_rows > 0` (the memory
was READ — a host run with a zero read counter is the memory-disabled
`engram=0/<n>` class this project retracted a whole arm over, AGENTS §3.2);
and `_run_one` asserts the resolved config flags match the arm, so a driver bug
cannot silently run the same arm under two names.

**What this cannot prove, stated before the run.** It is a 600-step, `medium`,
one-data-file screen on one card. It cannot prove the corrected update is
*stable* over a 100k-step run (the counter grows, `1-μ^(t+1)` saturates, and
only a long run exercises the saturated regime), it cannot price the host bus
(the transfers are unchanged here and their cost is only visible on the b1
shape), and it does not test the b1 case this correction exists for (402M table
params: 12M rows x 32 dim). `medium`'s 524 288 rows/order keep the engram in
VRAM comfortably; the b1 path needs its own run at ~40 GB host RAM.

**Protocol** (`docs/protocols/AB-PROTOCOL.md`): 3 seeds per arm, 600 steps, each
arm-seed a SEPARATE PROCESS (3 seeds is 3 initialisations), one eval window of
81 920 B on every run, output one JSONL per run under `~/logs/ab_engram/` plus
the run's own `<name>.config.toml` (a number needs its config, AGENTS §1.4) and
a driver `event=params` row (the instantiated count and the table's row count).

Two deliberate deviations from `ab_mhc.py`'s COMMON, both forced by this brief
and both named here:

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

#: `ab_mhc.py`'s COMMON_TRAIN minus `mix_plan` (deviation 1 in the docstring).
#: `engram_ram` and `engram_bias_correction` are the ARM and are set per run.
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
)

#: The model-side COMMON (`--set` flags in the battery). The Engram is not a
#: model flag: its substrate is `train.engram_ram` and the arm is
#: `train.engram_bias_correction`.
COMMON_MODEL = dict(
    moe_kernel="triton",
    moe_capacity=0.0,
    bf16=True,
    mhc_streams=2,
    use_attnres=False,
    loom=False,
)

#: `(arm name, engram_ram, engram_bias_correction)`. The control is first so the
#: table reads against it; the two RAM arms differ ONLY in the formula.
ARMS = (
    ("vram", False, True),
    ("ram_old_formula", True, False),
    ("ram_corrected", True, True),
)

#: The counters each arm MUST show, and the ones it must NOT: the substrate a
#: run names is the one that ran. `engram_rows` is the memory READ (both
#: substrates count it in `_read`), `engram_host_updates` the host update.
EXPECT = {
    "vram": {"engram_rows": ">0", "engram_host_updates": "==0"},
    "ram_old_formula": {"engram_rows": ">0", "engram_host_updates": ">0"},
    "ram_corrected": {"engram_rows": ">0", "engram_host_updates": ">0"},
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
                why.append(f"seed {r['seed']}: {name}={got} — the wrong substrate ran")
    return why


def verdict(control: list[dict], old: list[dict], new: list[dict], window: int) -> dict:
    """The pre-registered rule over the three arms (module docstring)."""
    why = (
        check(control, window, "vram")
        + check(old, window, "ram_old_formula")
        + check(new, window, "ram_corrected")
    )
    vram_mean, vram_spread = summarize(control)
    old_mean, old_spread = summarize(old)
    new_mean, new_spread = summarize(new)
    thr_old = max(vram_spread, old_spread)
    thr_new = max(vram_spread, new_spread)
    out = {
        "verdict": "void" if why else "",
        "reason": "; ".join(why),
        "vram_mean": vram_mean,
        "vram_spread": vram_spread,
        "old_mean": old_mean,
        "old_spread": old_spread,
        "new_mean": new_mean,
        "new_spread": new_spread,
        "old_delta": old_mean - vram_mean,
        "new_delta": new_mean - vram_mean,
        "correction_gain": old_mean - new_mean,
        "thr_old": thr_old,
        "thr_new": thr_new,
        "ms_vram": statistics.median([r["ms_per_step"] for r in control]),
        "ms_old": statistics.median([r["ms_per_step"] for r in old]),
        "ms_new": statistics.median([r["ms_per_step"] for r in new]),
    }
    if why:
        return out
    reproduces = old_mean > vram_mean + thr_old
    closed = new_mean <= vram_mean + thr_new
    improved = old_mean - new_mean > max(old_spread, new_spread)
    if not reproduces and closed:
        out["verdict"] = "not-needed"
    elif reproduces and closed and improved:
        out["verdict"] = "held"
    elif reproduces and not closed:
        out["verdict"] = "insufficient"
    elif reproduces and closed and not improved:
        out["verdict"] = "no-effect"
    else:
        out["verdict"] = "harmful"
    return out


def _run_one(args: argparse.Namespace) -> int:
    """One arm, one seed, one process: the trainer with the Engram flags live."""
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.model.engram import table_slots
    from dormouse.train import train

    ram, corrected = next((r, c) for name, r, c in ARMS if name == args.arm)
    run = load_run(
        PRESET,
        model_overrides=COMMON_MODEL,
        train_overrides={
            **COMMON_TRAIN,
            "seed": args.seed,
            "lr": LR,
            "out_dir": str(Path(args.out).parent),
            "ckpt_name": f"{args.arm}_s{args.seed}",
            # The JSONL's `tokens_per_s` is this driver's speed instrument and
            # `timers` OFF is what keeps it honest (`train/loop.py`'s own
            # comment: `timer.report()` resets the mark the telemetry reads).
            "timers": False,
            "json_log": str(args.out),
            "eval_data": args.eval,
            "engram_ram": ram,
            "engram_bias_correction": corrected,
        },
    )
    run.train.data = args.data
    if not Path(run.train.data).is_file():
        print(f"no data file at {run.train.data}: pass --data")
        return 1
    # The snapshot is the arm's record: a number needs its config (AGENTS §1.4).
    Path(args.out).with_suffix(".config.toml").write_text(run.snapshot_toml())

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    import torch

    model = DormouseModel(run.model, device=torch.device(device))
    # The dispatch must match the flags: an arm whose substrate/formula is not
    # the one it names is a NOT_ENGAGED run wearing the arm's name.
    assert run.train.engram_ram == ram, (run.train.engram_ram, ram)
    assert run.train.engram_bias_correction == corrected, (
        run.train.engram_bias_correction,
        corrected,
    )
    total = sum(p.numel() for p in model.parameters())
    table_rows = table_slots(run.model.engram_rows) * len(run.model.engram_orders) if ram else 0
    table_params = table_rows * run.model.engram_dim if ram else 0
    window = args.eval_batches * args.batch * args.seq
    print(
        f"[run] arm={args.arm} engram_ram={ram} bias_correction={corrected} "
        f"seed={args.seed} steps={args.steps} b{args.batch} s{args.seq} lr={LR} "
        f"window={window} B params={total:,} host_table_rows={table_rows:,} "
        f"host_table_params={table_params:,}",
        flush=True,
    )
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train(model, run, device=torch.device(device), log=print)
    # The substrate a run names is the one that must be live AFTER setup: a
    # ram arm that never switched is the NOT_ENGAGED class.
    assert (model.loop.engram.host is not None) == ram, (
        f"after train(), host={model.loop.engram.host is not None} for engram_ram={ram}"
    )

    def _row(**kw) -> None:
        with open(args.out, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(kw) + "\n")

    params = {
        "params": total,
        "host_table_rows": table_rows,
        "host_table_params": table_params,
        "engram_ram": ram,
        "engram_bias_correction": corrected,
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
        f"pre-registered: the old formula reproduces the +0.83 BPB host-RAM gap and "
        f"the per-row bias correction closes it to the control's seed spread."
    )
    for arm, _ram, _corrected in ARMS:
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

    table = {arm: [] for arm, _r, _c in ARMS}
    for arm, _ram, _corrected in ARMS:
        for seed in seeds:
            row = {"seed": seed, **read_run(out_dir / f"{arm}_s{seed}.jsonl", args.steps)}
            for line in (out_dir / f"{arm}_s{seed}.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r.get("event") == "params":
                    row.update(
                        params=r.get("params"),
                        host_table_rows=r.get("host_table_rows"),
                        host_table_params=r.get("host_table_params"),
                    )
            table[arm].append(row)
    print(
        f"\n{'arm':>16s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'ms/step':>8s} "
        f"{'tok/s':>8s} {'peak GiB':>8s} {'nan':>4s} {'host rows':>10s}"
    )
    for arm, runs in table.items():
        for r in runs:
            peak = f"{r['peak_gib']:.2f}" if r["peak_gib"] is not None else "-"
            print(
                f"{arm:>16s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['ms_per_step']:8.1f} {r['tok_s']:8.0f} {peak:>8s} {r['nan']:>4d} "
                f"{r.get('host_table_rows') or 0:>10}"
            )
    v = verdict(table["vram"], table["ram_old_formula"], table["ram_corrected"], window)
    for arm, runs in table.items():
        mean, spread = summarize(runs)
        updates = sum(r["counters"].get("engram_host_updates", 0) for r in runs)
        rows = sum(r["counters"].get("engram_host_update_rows", 0) for r in runs)
        print(
            f"{arm:>16s} mean bpb {mean:.4f}  spread {spread:.4f}  "
            f"median warm ms/step {statistics.median([r['ms_per_step'] for r in runs]):.1f}  "
            f"host_updates={updates} host_rows={rows}"
        )
    print(
        f"\nvoid checks: every eval scored {window} B, 0 NaN-skipped steps, "
        f"vram host_updates==0 and both RAM arms >0, engram_rows>0 on every run, "
        f"{len(seeds)} distinct seeds per arm, the resolved flags printed by "
        f"`[run]` matching the arm.\n"
        f"pre-registered rule: HELD iff old_mean > vram_mean + max(vram_spread, "
        f"old_spread), new_mean <= vram_mean + max(vram_spread, new_spread), and "
        f"old_mean - new_mean > max(old_spread, new_spread); INSUFFICIENT iff the "
        f"gap reproduces and the correction does not close it; NO-EFFECT iff it "
        f"closes but the RAM arms tie; NOT-NEEDED iff the 600-step window does "
        f"not reproduce the gap at all; HARMFUL iff the corrected arm is worse "
        f"than the control and not better than the old formula.\n"
        f"ENG RAM VERDICT: {v['verdict'].upper()}"
        + (f"\nvoid reasons: {v['reason']}" if v["reason"] else "")
        + f"\nvram {v['vram_mean']:.4f} +- {v['vram_spread']:.4f} | "
        f"old {v['old_mean']:.4f} +- {v['old_spread']:.4f} "
        f"({v['old_delta']:+.4f}) | corrected {v['new_mean']:.4f} +- "
        f"{v['new_spread']:.4f} ({v['new_delta']:+.4f}); "
        f"correction gain {v['correction_gain']:+.4f} BPB, "
        f"thresholds old {v['thr_old']:.4f} / new {v['thr_new']:.4f}"
    )
    print(
        f"step time (SEPARATE, reported not gated): vram {v['ms_vram']:.1f} -> "
        f"old {v['ms_old']:.1f} -> corrected {v['ms_new']:.1f} median warm ms/step."
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
    a.add_argument("--out-dir", default="~/logs/ab_engram")
    r = sub.add_parser("_run", help="one arm, one seed (the child mode)")
    r.add_argument("--arm", required=True, choices=[arm for arm, _r, _c in ARMS])
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
