"""The MTP A/B: `k0` (control) vs `k2` and `k4`, 3 seeds x 2 000 steps, medium.

    uv run python scripts/ab_mtp.py ab                 # 3 arms x 3 seeds
    uv run python scripts/ab_mtp.py ab --seeds 1       # one seed, dry check

**THIS DRIVER HAS NOT BEEN RUN.** The card is held by the benchmark chain
(AGENTS §1.5: one heavy thing at a time), so the pre-registered rule is written
here BEFORE the first run so that run cannot move the goalposts. A busy card
makes the driver REFUSE and exit 1; nothing is measured and no table prints.

**The arms** — one model field differs (`mtp_k`), everything else is shared:

* `k0` — the control: `mtp_k=0, mtp_weight=0.0`. The MTP head is not built, the
  parameter set is the recipe's own, and the objective is the recipe's CE.
* `k2` / `k4` — K future-byte modules on the loop's final readout, sharing the
  model's embedding and `lm_head` (DeepSeek-V3 §3.2; `model/mtp.py`, whose
  docstring carries the prior `aux_fb` tie and deletion `39d0485`).

`mtp_weight=1.0` on both arms: λ is NOT under test here (K is), and one value
for both makes the per-K comparison one knob. The gate's gradient fixture uses
the same 1.0, so a K that trains is a K this driver can score.

**Why an A/B at all, given the prior tie.** `aux_fb` was a different head (one
unanchored `d_model -> vocab` readout, horizon 2). MTP's published wins are
large-scale (the paper's own "increasingly useful for larger model sizes";
DeepSeek-V3 at 671B), and `arXiv:2505.22757` is the direct small-scale negative.
At our 9.2M the honest state is "unknown", so the arm is measured, not assumed.

**The flags** are copied from `/tmp/opencode/battery.sh`'s `BASE` + `COMMON`
blocks, the recipe the next run uses: `medium`, b8 x s512, 2 000 steps,
`tf32`, `ns_kind=triton`, `opt_graph`, `moe_kernel=triton`, `moe_capacity=0`,
`engram_ram`, `bf16`, `mix_plan=configs/mix_big.toml`, eval every 500 over
`eval_batches=20` — so every arm and seed scores the SAME window,
`20 x 8 x 512 = 81 920 B` (AGENTS §2.6). The only additions are the run's own
`seed`/`mtp_*`/`ckpt_*`/`json-log`.

**The verdict** (AGENTS §1.2, pre-registered): per arm,
`Δ = mean_arm − mean_control`; **WIN iff `Δ < −control_spread`**, where
`control_spread = max(control) − min(control)` over the three seeds. Anything
else is a TIE, and a tie deletes the mechanism. VOID (not a number) when an
eval window is not 81 920 B, a NaN-skipped step occurred, or the `mtp=` field
is absent on a k2/k4 run / present on a k0 run — the arm must show it ran.

**What the driver does NOT do:** apply the verdict, average across K, or read
the training `loss=` as a quality number (it contains λ·MTP by construction).
Held-out BPB is the only comparison, and the eval forward runs `pure=True`
(`model.py`), so it is the predictive CE on every arm.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight as _lane_preflight  # noqa: E402

# `tests/test_mtp.py` pins this 3-arg shape and its "no data file at" wording:
# the card first, then the cheaper blocker (the missing file, named).
preflight = lambda device, data, eval_data: (  # noqa: E731
    _lane_preflight(device)
    or next((f"no data file at {p}" for p in (data, eval_data) if not Path(p).is_file()), None)
)

ROOT = Path(__file__).resolve().parents[1]

PRESET = "medium"
SEEDS = (1, 2, 3)
STEPS = 2000
#: `(arm name, mtp_k)`. λ is 1.0 for every arm (`mtp_weight` below).
ARMS = (("k0", 0), ("k2", 2), ("k4", 4))
MTP_WEIGHT = 1.0
EVAL_BATCHES = 20
BATCH = 8
SEQ = 512

#: `/tmp/opencode/battery.sh`'s `BASE`: the device and the two data files.
DATA = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"
EVAL_DATA = "/home/sehaxe/dormouse/data_local/eval_mix/eval_mix.bin"
BASE = ("--device", "cuda", "--data", DATA, "--eval", EVAL_DATA)
#: The battery's `COMMON`, verbatim. Do not tune one arm: the whole point is
#: that `mtp_k` is the only difference.
COMMON = (
    "--set",
    "steps=2000",
    "--set",
    "batch=8",
    "--set",
    "seq_len=512",
    "--set",
    "eval=true",
    "--set",
    "eval_every=500",
    "--set",
    "eval_batches=20",
    "--set",
    "log_every=250",
    "--set",
    "ckpt_every=1000000",
    "--set",
    "tf32=true",
    "--set",
    "ns_kind=triton",
    "--set",
    "opt_graph=true",
    "--set",
    "moe_kernel=triton",
    "--set",
    "moe_capacity=0",
    "--set",
    "engram_ram=true",
    "--set",
    "bf16=true",
    "--set",
    "mix_plan=configs/mix_big.toml",
)

#: The one window every run must score. A different number is a VOID arm.
WINDOW = EVAL_BATCHES * BATCH * SEQ


def refuse(why: str) -> int:
    """A LOUD refusal with no table: a busy card measures nothing."""
    print(
        f"\n=== REFUSED: {why}\n"
        f"=== ONE GPU PROCESS AT A TIME (AGENTS §1.5): nothing was measured and no\n"
        f"=== table is printed. Re-run when the card is free.\n",
        flush=True,
    )
    return 1


def read_run(jsonl: Path, log: Path) -> dict:
    """One run -> the numbers the verdict needs, plus the arm's own evidence.

    `bpb` is the LAST eval's held-out score; `scored` is its byte window;
    `nan` is the RUNNING `nan_skipped` count (the last log row, never the sum);
    `mtp_logged` is how many training log lines carry the `mtp=` field — a k2/k4
    run with zero did not run the arm, and a k0 run with one ran a term its
    control must not have.
    """
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    evals = [r for r in rows if r.get("event") == "eval"]
    logs = [r for r in rows if r.get("event") == "log"]
    assert evals, f"{jsonl}: no eval row — nothing to compare"
    assert logs, f"{jsonl}: no log row — nothing to compare"
    text = log.read_text() if log.is_file() else ""
    return {
        "bpb": evals[-1]["bpb"],
        "scored": evals[-1]["scored"],
        "n_evals": len(evals),
        "nan": max((r.get("nan_skipped", 0) for r in logs), default=0),
        "last_step": logs[-1].get("step", 0),
        "mtp_logged": text.count(" mtp="),
    }


def summarize(runs: list[dict]) -> tuple[float, float]:
    """Mean BPB and the seed spread (max - min) of one arm."""
    bpbs = [r["bpb"] for r in runs]
    return statistics.mean(bpbs), max(bpbs) - min(bpbs)


def check(runs: list[dict], window: int, *, mtp: bool) -> list[str]:
    """The void checks: a number that fails one is not a number."""
    why: list[str] = []
    for r in runs:
        if r["scored"] != window:
            why.append(f"seed {r['seed']}: eval window {r['scored']} != {window} B")
        if r["nan"]:
            why.append(f"seed {r['seed']}: {r['nan']} NaN-skipped steps")
        if mtp and not r["mtp_logged"]:
            why.append(f"seed {r['seed']}: no `mtp=` on any log line — the arm did not run")
        if not mtp and r["mtp_logged"]:
            why.append(f"seed {r['seed']}: `mtp=` on a k0 log line — the control ran the arm")
    return why


def verdict(control: list[dict], arm: list[dict], window: int) -> dict:
    """AGENTS §1.2 over one control and one arm: win iff Δ < -control_spread."""
    why = check(control, window, mtp=False) + check(arm, window, mtp=True)
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
    elif amean < cmean - cspread:
        out["verdict"] = "win"
    else:
        out["verdict"] = "tie"
    return out


def run_one(arm: str, k: int, seed: int, out_dir: Path, device: str, steps: int) -> int:
    """One arm, one seed, one CLI process — the battery's own launch shape."""
    jsonl = out_dir / f"{arm}_s{seed}.jsonl"
    log = out_dir / f"{arm}_s{seed}.log"
    jsonl.unlink(missing_ok=True)
    weight = 0.0 if k == 0 else MTP_WEIGHT
    cmd = [
        "uv",
        "run",
        "--extra",
        "gpu",
        "dormouse",
        "train",
        PRESET,
        *BASE,
        *COMMON,
        "--set",
        f"steps={steps}",
        "--set",
        f"seed={seed}",
        "--set",
        f"mtp_k={k}",
        "--set",
        f"mtp_weight={weight}",
        "--set",
        f"ckpt_name=mtp_{arm}_s{seed}",
        "--set",
        f"out_dir={out_dir / 'ckpt'}",
        "--json-log",
        str(jsonl),
    ]
    if device != "cuda":
        cmd[cmd.index("--device") + 1] = device
    print(f"\n=== {arm} seed {seed} -> {jsonl}", flush=True)
    with open(log, "w", encoding="utf-8") as sink:
        done = subprocess.run(cmd, cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT)
    if done.returncode != 0:
        print(f"the {arm} run (seed {seed}) exited {done.returncode}: no table. Log: {log}")
    return done.returncode


def ab(args: argparse.Namespace) -> int:
    """The driver: `len(ARMS)` x seeds separate processes, then the table."""
    seeds = tuple(int(s) for s in args.seeds.split(","))
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"MTP A/B: {len(ARMS)} arms x {len(seeds)} seeds x {args.steps} steps, {PRESET} "
        f"b{BATCH} s{SEQ}, window {WINDOW} B, λ={MTP_WEIGHT}. Output: {out_dir}\n"
        f"pre-registered (AGENTS §1.2): WIN iff the arm's mean held-out BPB is "
        f"below the control's by MORE than the control's own seed spread; a tie "
        f"deletes the mechanism. Prior: the aux_fb future-byte head tied at 3 "
        f"seeds x 2k and was deleted (39d0485)."
    )
    for arm, k in ARMS:
        for seed in seeds:
            rc = run_one(arm, k, seed, out_dir, args.device, args.steps)
            if rc != 0:
                return rc

    table = {arm: [] for arm, _k in ARMS}
    for arm, _k in ARMS:
        for seed in seeds:
            table[arm].append(
                {
                    "seed": seed,
                    **read_run(out_dir / f"{arm}_s{seed}.jsonl", out_dir / f"{arm}_s{seed}.log"),
                }
            )
    print(
        f"\n{'arm':>4s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'evals':>6s} "
        f"{'nan':>4s} {'mtp lines':>9s}"
    )
    for arm, runs in table.items():
        for r in runs:
            print(
                f"{arm:>4s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['n_evals']:>6d} {r['nan']:>4d} {r['mtp_logged']:>9d}"
            )
    for arm, runs in table.items():
        mean, spread = summarize(runs)
        print(f"{arm:>4s} mean bpb {mean:.4f}  spread {spread:.4f}")

    for arm, _k in ARMS[1:]:
        v = verdict(table["k0"], table[arm], WINDOW)
        print(
            f"\n{arm} vs k0: {v['verdict'].upper()} "
            f"(control {v['control_mean']:.4f} +- {v['control_spread']:.4f}, "
            f"{arm} {v['arm_mean']:.4f} +- {v['arm_spread']:.4f}, "
            f"delta {v['delta']:+.4f})" + (f"\nvoid reasons: {v['reason']}" if v["reason"] else "")
        )
    print(
        f"\nvoid checks: every eval scored {WINDOW} B, 0 NaN-skipped steps, "
        f"`mtp=` present on k2/k4 and absent on k0. A WIN clears the control's "
        f"own spread; any arm that does not is a tie, and a tie deletes it."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ab", help="the A/B driver (3 arms x 3 seeds)")
    a.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a.add_argument("--steps", type=int, default=STEPS)
    a.add_argument("--device", default="cuda")
    a.add_argument("--out-dir", default="~/logs/ab_mtp")
    args = ap.parse_args(argv)
    why = preflight(args.device, DATA, EVAL_DATA)
    if why is not None:
        return refuse(why)
    return ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
