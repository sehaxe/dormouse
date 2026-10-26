"""The DEPTH-8 A/B: `mi4` (control) vs `mi8` (`max_iter=8` + `rand_depth`).

    uv run python scripts/ab_depth8.py ab            # 2 arms x 3 seeds
    uv run python scripts/ab_depth8.py ab --seeds 1  # one seed, dry check

**THIS DRIVER HAS NOT BEEN RUN.** The card is held by the benchmark chain
(AGENTS §1.5: one heavy thing at a time), so the verdict is written here BEFORE
the first run so that run cannot move the goalposts. A busy card makes the
driver REFUSE and exit 1; nothing is measured and no table prints.

**Why these two arms.** `rand_depth` trains every depth `1..max_iter` on a pure
`(seed, step)` schedule, but every run on record has `max_iter=4`, so depths
5-8 were never trained and the depth curve stops at 4. The owner's goal is
adaptive depth *up to 8*: 2 iterations on easy input, up to 8 on hard input.
That needs a model that has SEEN 8. `mi8` is `max_iter=8` + `rand_depth=true`;
`mi4` is today's recipe (`max_iter=4`, fixed depth), the control.

**The arms** — one model field differs (`max_iter`) and one training flag
(`rand_depth`), and that pair IS the arm under test:

* `mi4`  — `max_iter=4`, `rand_depth=false`: today's fixed-depth-4 recipe.
* `mi8`  — `max_iter=8`, `rand_depth=true`: the loop samples `T ∈ 1..=8` per
  step (`train/loop.py:_sample_depth`), so every depth is trained, and the loss
  divides by the `T` actually run so the depths are scale-comparable.
* both  — `eval_depths=true`: the eval line carries `depths=1:<fin>,...`, one
  held-out pass per depth, and `fin` is the depth-comparable CE (the headline
  `loss` averages `T+1` terms and is NOT; found 2026-10-16).

**The flags** are copied from `/tmp/opencode/battery.sh`'s `BASE` + `COMMON`
blocks, verbatim — `medium`, b8 x s512, 2 000 steps, `tf32`, `ns_kind=triton`,
`opt_graph`, `moe_kernel=triton`, `moe_capacity=0`, `engram_ram`, `bf16`,
`mix_plan=configs/mix_big.toml`, eval every 500 over `eval_batches=20` — so
every arm and seed scores the SAME window, `20 x 8 x 512 = 81 920 B` (AGENTS
§2.6). The only additions are the run's own
`seed`/`max_iter`/`rand_depth`/`eval_depths`/`ckpt_*`/`json-log`.

**The table** is BPB (`fin / ln 2`, the same conversion the eval line uses) at
depths 1..8 per arm, mean over seeds. `mi4` has no rows for depths 5..8 — that
is the point of the arm, not a missing value: it cannot run them.

**The verdict** (AGENTS §1.2, pre-registered): the two arms are compared at
**matched depth 4**, the deepest depth both trained on, because that isolates
"did training depth 8 change the model" from "what does depth 8 infer to".
`Δ = mean(mi8@4) − mean(mi4@4)`; **WIN iff `Δ < −control_spread`**, where
`control_spread = max(mi4@4) − min(mi4@4)` over the three seeds. Anything else
is a TIE, and a tie deletes the arm. The depths 5..8 of `mi8` are reported —
they answer the open question "does depth 5-8 hold or degrade" — and are NOT
allowed to rescue a tie at depth 4. This driver applies nothing.

VOID (not a number) when an eval window is not 81 920 B, a NaN-skipped step
occurred, `mi4` prints a depth above 4 or `mi8` misses any of 1..8, or the
`depth` counter of a `mi8` run equals the fixed `4 x steps` sum (the random
schedule did not run).

**Wall clock**: 6 runs x ~2 000 steps; `mi8`'s evals each run 8 held-out
passes instead of 4, so it is slower than a control arm at the same step count.
Not measured — the card must be idle before any number is quoted.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import data_missing, preflight  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PRESET = "medium"
SEEDS = (1, 2, 3)
STEPS = 2000
EVAL_BATCHES = 20
BATCH = 8
SEQ = 512

#: `(arm, model overrides, train overrides)`. `mi4` first: the table reads
#: against it. `rand_depth` is a TRAINING field (`TrainConfig`), `max_iter` a
#: MODEL field — the two are set through the same `--set` seam.
ARMS = (
    ("mi4", {"max_iter": 4}, {"rand_depth": False}),
    ("mi8", {"max_iter": 8}, {"rand_depth": True}),
)

#: `/tmp/opencode/battery.sh`'s `BASE`: the device and the two data files.
DATA = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"
EVAL_DATA = "/home/sehaxe/dormouse/data_local/eval_mix/eval_mix.bin"
BASE = ("--device", "cuda", "--data", DATA, "--eval", EVAL_DATA)
#: The battery's `COMMON`, verbatim. Do not tune one arm: the whole point is
#: that the depth pair is the only difference.
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

#: The deepest depth both arms trained on — the matched comparison.
MATCHED_DEPTH = 4


def refuse(why: str) -> int:
    """A LOUD refusal with no table: a busy card measures nothing."""
    print(
        f"\n=== REFUSED: {why}\n"
        f"=== ONE GPU PROCESS AT A TIME (AGENTS §1.5): nothing was measured and no\n"
        f"=== table is printed. Re-run when the card is free.\n",
        flush=True,
    )
    return 1


def read_depths(log: Path) -> dict[int, float]:
    """The LAST eval line's `depths=` field → `{depth: CE(fin)}` (empty if none).

    The eval JSONL carries the headline `loss` only, so the depth curve is read
    from the run's own log text — the same line a reader sees. `fin` is the CE
    of the mean readout at that depth, the depth-comparable number; BPB is
    `fin / ln 2`, the trainer's conversion.
    """
    text = log.read_text() if log.is_file() else ""
    hits = [ln for ln in text.splitlines() if " depths=" in ln]
    if not hits:
        return {}
    payload = hits[-1].split(" depths=", 1)[1].strip()
    out: dict[int, float] = {}
    for item in payload.split(","):
        depth, rest = item.split(":", 1)
        out[int(depth)] = float(rest.split("(", 1)[0])
    return out


def read_run(jsonl: Path, log: Path) -> dict:
    """One run -> the numbers the verdict needs, plus the arm's own evidence."""
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    evals = [r for r in rows if r.get("event") == "eval"]
    logs = [r for r in rows if r.get("event") == "log"]
    assert evals, f"{jsonl}: no eval row — nothing to compare"
    assert logs, f"{jsonl}: no log row — nothing to compare"
    return {
        "bpb": evals[-1]["bpb"],
        "scored": evals[-1]["scored"],
        "n_evals": len(evals),
        "nan": max((r.get("nan_skipped", 0) for r in logs), default=0),
        # The `depth` counter is the SUM of the depths actually run. The JSONL's
        # `counters` are the LOG WINDOW's (they reset every cadence), so the run
        # total is their sum over the log rows. A fixed arm reads `4 x steps`; a
        # random one reads something else. It is the void check that the
        # schedule ran (AGENTS §1.1: count the arm).
        "depth_sum": sum(r.get("counters", {}).get("depth", 0) for r in logs),
        "depths": read_depths(log),
    }


def summarize(runs: list[dict]) -> tuple[float, float]:
    """Mean BPB and the seed spread (max - min) of one arm at the matched depth."""
    bpbs = [r["depths"][MATCHED_DEPTH] / math.log(2) for r in runs]
    return statistics.mean(bpbs), max(bpbs) - min(bpbs)


def mean_bpb(runs: list[dict], depth: int) -> float:
    return statistics.mean([r["depths"][depth] / math.log(2) for r in runs])


def check(runs: list[dict], arm: str, window: int, steps: int) -> list[str]:
    """The void checks: a number that fails one is not a number."""
    why: list[str] = []
    want = {1, 2, 3, 4} if arm == "mi4" else set(range(1, 9))
    for r in runs:
        if r["scored"] != window:
            why.append(f"seed {r['seed']}: eval window {r['scored']} != {window} B")
        if r["nan"]:
            why.append(f"seed {r['seed']}: {r['nan']} NaN-skipped steps")
        if set(r["depths"]) != want:
            why.append(f"seed {r['seed']}: depth curve {sorted(r['depths'])} != {sorted(want)}")
        if arm == "mi4" and r["depth_sum"] != 4 * steps:
            why.append(f"seed {r['seed']}: depth sum {r['depth_sum']} != 4 x {steps}")
        if arm == "mi8" and r["depth_sum"] == 4 * steps:
            why.append(
                f"seed {r['seed']}: depth sum {r['depth_sum']} == 4 x {steps} — "
                f"rand_depth did not sample"
            )
        if arm == "mi8" and not (steps < r["depth_sum"] < steps * 8):
            why.append(f"seed {r['seed']}: depth sum {r['depth_sum']} outside (steps, 8*steps)")
    return why


def verdict(control: list[dict], arm: list[dict], window: int, steps: int) -> dict:
    """AGENTS §1.2 on the MATCHED depth: win iff Δ < -control_spread."""
    why = check(control, "mi4", window, steps) + check(arm, "mi8", window, steps)
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


def _typed(text: str):
    """A `--set k=v` literal back to the Python value the config field wants."""
    if text in ("true", "false"):
        return text == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _snapshot(
    arm: str, model_over: dict, train_over: dict, seed: int, out_dir: Path, steps: int
) -> None:
    """Write the run's own `<name>.config.toml` beside its JSONL.

    `ckpt_every` is 10**6, so the trainer writes no checkpoint and therefore no
    snapshot of its own; without this the number would have no config record
    (AGENTS §1.4). The overrides are resolved from the SAME COMMON tuple the
    child is launched with, so the record cannot drift from the run.
    """
    from dormouse.config import ModelConfig, load_run

    model_keys = {f.name for f in dataclasses.fields(ModelConfig)}
    model_common: dict = {}
    train_common: dict = {}
    items = list(COMMON)
    for i in range(0, len(items), 2):
        assert items[i] == "--set", items[i]
        key, value = items[i + 1].split("=", 1)
        (model_common if key in model_keys else train_common)[key] = _typed(value)
    run = load_run(
        PRESET,
        model_overrides={**model_common, **model_over},
        train_overrides={
            **train_common,
            **train_over,
            "steps": steps,
            "seed": seed,
            "ckpt_name": f"depth8_{arm}_s{seed}",
            "out_dir": str(out_dir / "ckpt"),
            "json_log": str(out_dir / f"{arm}_s{seed}.jsonl"),
            "data": DATA,
            "eval": True,
            "eval_data": EVAL_DATA,
            "eval_depths": True,
        },
    )
    (out_dir / f"{arm}_s{seed}.config.toml").write_text(run.snapshot_toml())


def run_one(
    arm: str, model_over: dict, train_over: dict, seed: int, out_dir: Path, device: str, steps: int
) -> int:
    """One arm, one seed, one CLI process — the battery's own launch shape."""
    jsonl = out_dir / f"{arm}_s{seed}.jsonl"
    log = out_dir / f"{arm}_s{seed}.log"
    jsonl.unlink(missing_ok=True)
    over = {**model_over, **train_over, "eval_depths": True}
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
        f"ckpt_name=depth8_{arm}_s{seed}",
        "--set",
        f"out_dir={out_dir / 'ckpt'}",
        "--json-log",
        str(jsonl),
    ]
    for key, value in over.items():
        cmd += ["--set", f"{key}={_lit(value)}"]
    if device != "cuda":
        cmd[cmd.index("--device") + 1] = device
    print(f"\n=== {arm} seed {seed} -> {jsonl}", flush=True)
    with open(log, "w", encoding="utf-8") as sink:
        done = subprocess.run(cmd, cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT)
    if done.returncode != 0:
        print(f"the {arm} run (seed {seed}) exited {done.returncode}: no table. Log: {log}")
    return done.returncode


def _lit(value: object) -> str:
    """A config value as the `--set k=v` literal the CLI parses back to it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def ab(args: argparse.Namespace) -> int:
    """The driver: `len(ARMS)` x seeds separate processes, then the table."""
    seeds = tuple(int(s) for s in args.seeds.split(","))
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"DEPTH-8 A/B: {len(ARMS)} arms x {len(seeds)} seeds x {args.steps} steps, "
        f"{PRESET} b{BATCH} s{SEQ}, window {WINDOW} B. Output: {out_dir}\n"
        f"pre-registered (AGENTS §1.2): WIN iff mi8's matched-depth-{MATCHED_DEPTH} "
        f"held-out BPB beats mi4's by MORE than mi4's own 3-seed spread; a tie "
        f"deletes the arm. Depths 5-8 are reported, not used to rescue a tie."
    )
    for arm, model_over, train_over in ARMS:
        for seed in seeds:
            _snapshot(arm, model_over, train_over, seed, out_dir, args.steps)
            rc = run_one(arm, model_over, train_over, seed, out_dir, args.device, args.steps)
            if rc != 0:
                return rc

    table = {arm: [] for arm, _m, _t in ARMS}
    for arm, _model, _train in ARMS:
        for seed in seeds:
            table[arm].append(
                {
                    "seed": seed,
                    **read_run(out_dir / f"{arm}_s{seed}.jsonl", out_dir / f"{arm}_s{seed}.log"),
                }
            )

    print(
        f"\n{'arm':>4s} {'seed':>4s} {'bpb@4':>8s} {'scored B':>9s} {'evals':>6s} "
        f"{'nan':>4s} {'depth':>6s}  depths 1..8 (BPB, fin)"
    )
    for arm, runs in table.items():
        for r in runs:
            curve = " ".join(f"{d}:{r['depths'][d] / math.log(2):.3f}" for d in sorted(r["depths"]))
            print(
                f"{arm:>4s} {r['seed']:>4d} "
                f"{r['depths'][MATCHED_DEPTH] / math.log(2):8.4f} {r['scored']:>9d} "
                f"{r['n_evals']:>6d} {r['nan']:>4d} {r['depth_sum']:>6d}  {curve}"
            )
    for arm, runs in table.items():
        mean, spread = summarize(runs)
        print(f"{arm:>4s} mean bpb@{MATCHED_DEPTH} {mean:.4f}  spread {spread:.4f}")

    print(f"\nmean BPB by depth ({len(seeds)} seeds):")
    header = "arm  " + "".join(f"{'d' + str(d):>9s}" for d in range(1, 9))
    print(header)
    for arm, runs in table.items():
        cells = []
        for d in range(1, 9):
            cells.append(
                f"{mean_bpb(runs, d):9.4f}" if all(d in r["depths"] for r in runs) else f"{'-':>9s}"
            )
        print(f"{arm:>4s} " + "".join(cells))

    v = verdict(table["mi4"], table["mi8"], WINDOW, args.steps)
    print(
        f"\nmi8 vs mi4 at depth {MATCHED_DEPTH}: {v['verdict'].upper()} "
        f"(control {v['control_mean']:.4f} +- {v['control_spread']:.4f}, "
        f"mi8 {v['arm_mean']:.4f} +- {v['arm_spread']:.4f}, "
        f"delta {v['delta']:+.4f})" + (f"\nvoid reasons: {v['reason']}" if v["reason"] else "")
    )
    if v["verdict"] != "void":
        mi4_best = min(range(1, 5), key=lambda d: mean_bpb(table["mi4"], d))
        mi8_best = min(range(1, 9), key=lambda d: mean_bpb(table["mi8"], d))
        print(
            f"best depth: mi4 d{mi4_best} {mean_bpb(table['mi4'], mi4_best):.4f} BPB · "
            f"mi8 d{mi8_best} {mean_bpb(table['mi8'], mi8_best):.4f} BPB "
            f"(reported, not a rescue: the verdict is at matched depth "
            f"{MATCHED_DEPTH})"
        )
    print(
        f"\nvoid checks: every eval scored {WINDOW} B, 0 NaN-skipped steps, mi4 "
        f"curve exactly 1..4 with depth sum 4 x steps, mi8 curve exactly 1..8 with "
        f"a depth sum that is not 4 x steps. A WIN clears the control's own "
        f"spread; anything else is a tie, and a tie deletes the arm."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ab", help="the A/B driver (2 arms x 3 seeds)")
    a.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a.add_argument("--steps", type=int, default=STEPS)
    a.add_argument("--device", default="cuda")
    a.add_argument("--out-dir", default="~/logs/ab_depth8")
    args = ap.parse_args(argv)
    why = preflight(args.device) or data_missing(DATA, EVAL_DATA)
    if why is not None:
        return refuse(why)
    return ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
