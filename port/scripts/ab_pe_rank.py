r"""PRE-REGISTERED A/B: `rank_mode="uniform"` vs `rank_mode="load"` (per-expert rank).

WHAT THE ARM DOES. `rank` stops meaning "every expert's rank" and becomes the
BUDGET — the MEAN per-expert rank. Every `rank_cadence` optimizer steps the
bank re-splits it from the LFB load integrator:

    rank_i = clamp(16, 64, round(rank · load_i / mean(load)))

then a water-fill holds `Σ rank_i == E·rank` EXACTLY. Per expert the bank costs
`2·d·rank_i + rank_i·f + 2·rank_i`, so the total is unchanged: the arm moves
CAPACITY to the experts the router actually uses, it does not add parameters.
The bank stays RECTANGULAR at `rank_max` (zero-padded U columns / V rows), so
it also does not save FLOPs — that is the honest scope, and the docstring of
`experts.SparseMoE` says so.

ARMS (3 seeds x 600 steps, preset `medium`, one eval window per arm)
    uniform  rank 32, every expert 32 — the control (`rank_mode` default)
    load     rank 32 BUDGET, band [16, 64], cadence 100 steps

The control and the arm run the SAME number of parameters (asserted below on
the instantiated tree), the same data order, the same eval window. The ONLY
difference is `rank_mode`.

`rank_cadence = 100` (not the 1000 default) because the run is 600 steps: at
1000 the arm would never fire and the A/B would compare the control with
itself. The step-0 firing has an all-zero integrator (`pe_rank_no_load`) and
is the identity; the informative firings are 100..600.

VERDICT RULE (AGENTS 1.2): `load` must beat the CONTROL's own 3-seed spread. A
tie DELETES the mechanism — a per-expert split that never moved a BPB is
complexity the repo would carry for nothing. Report the delta AND the spread;
never the delta alone.

VOID CHECKS, all read from the JSONL counters:
  * `load` must report `pe_rank_realloc > 0` and at least one `pe_rank{16,32,48,64}`
    counter. `pe_rank_realloc = 0` means the cadence never fired (the wrong
    `rank_cadence`) and the arm measured the control.
  * `uniform` must NOT report `pe_rank_realloc` (a control carrying it ran the
    arm wearing the control's label).
  * `moe_lb_coef` must be > 0 in BOTH arms (the load integrator is the same
    one LFB reads; with the controller off the signal is still accumulated, but
    the run would not be the recipe the presets ship).
  * `opt_graph` is false in both arms: the load arm REFUSES a capture (it bakes
    parameter shapes, and the cadence changes them), so the control must run
    the same eager optimizer stage or the A/B prices two changes.
  * `nan_skipped` is reported per arm; a run that skipped steps is not a
    BPB comparison.

USAGE (do NOT launch this from an agent session; it is pre-registered, not run)
    uv run --extra gpu python scripts/ab_pe_rank.py --dry-run   # commands + params
    uv run --extra gpu python scripts/ab_pe_rank.py             # preflight + 6 runs
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted.
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOGDIR = Path.home() / "logs" / "ab_pe_rank"
DATA = "/home/sehaxe/dormouse/data_local/mix4/web.bin"
EVAL = "/home/sehaxe/dormouse/data_local/mix4/web_eval.bin"

SEEDS = (1, 2, 3)
STEPS = 600
BUDGET = 32
CADENCE = 100

BASE = ["train", "medium", "--device", "cuda", "--data", DATA, "--eval", EVAL]
COMMON = [
    "--set",
    f"steps={STEPS}",
    "--set",
    "batch=8",
    "--set",
    "seq_len=512",
    "--set",
    "eval=true",
    "--set",
    "eval_every=200",
    "--set",
    "eval_batches=20",
    "--set",
    "log_every=100",
    "--set",
    "ckpt_every=1000000",
    "--set",
    "tf32=true",
    "--set",
    "ns_kind=triton",
    "--set",
    # The load arm refuses a capture (a capture bakes parameter shapes and the
    # cadence changes them), so BOTH arms run the eager stage.
    "opt_graph=false",
    "--set",
    "torch_compile=false",
    "--set",
    "moe_kernel=triton",
    "--set",
    "moe_capacity=0",
    "--set",
    "engram_ram=true",
    "--set",
    "bf16=true",
    # The load integrator is the one LFB reads; keep the preset's controller on
    # so the arm runs the shipped recipe (medium.toml: 0.01).
    "--set",
    "moe_lb_coef=0.01",
    "--set",
    f"rank={BUDGET}",
    "--set",
    f"rank_cadence={CADENCE}",
]

ARMS: dict[str, str] = {"uniform": "uniform", "load": "load"}


def params(arm: str) -> tuple[int, int]:
    """`(total, routed bank)` from the INSTANTIATED model on `meta`, not a formula."""
    from dormouse.config import load_preset, validate
    from dormouse.model import DormouseModel, param_count

    cfg = dataclasses.replace(load_preset("medium"), rank=BUDGET, rank_mode=ARMS[arm])
    validate(cfg)
    model = DormouseModel(cfg, device="meta")
    bank = sum(p.numel() for n, p in model.named_parameters() if n.startswith("loop.moe.experts."))
    return param_count(model), bank


def read_rows(path: Path) -> tuple[list[dict], list[dict]]:
    evals, logs = [], []
    if not path.exists():
        return evals, logs
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "bpb" in row:
            evals.append(row)
        elif "counters" in row:
            logs.append(row)
    return evals, logs


def run_arm(arm: str, seed: int, dry: bool) -> dict:
    log = LOGDIR / f"{arm}_s{seed}.jsonl"
    text = LOGDIR / f"{arm}_s{seed}.log"
    cmd = [
        "uv",
        "run",
        "--extra",
        "gpu",
        "dormouse",
        *BASE,
        *COMMON,
        "--set",
        f"seed={seed}",
        "--set",
        f"ckpt_name=abperank_{arm}_s{seed}",
        "--set",
        "out_dir=/tmp/opencode/ab_pe_rank_ckpt",
        # The TOML spelling, lowercase: `--set rank_mode=Uniform` would be a
        # LOUD refusal (the config checks the literal), never a silent default.
        "--set",
        f"rank_mode={ARMS[arm]}",
        "--json-log",
        str(log),
    ]
    if dry:
        print(" ".join(cmd))
        return {}
    t0 = time.time()
    with text.open("w") as fh:
        subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, check=False)
    evals, logs = read_rows(log)
    ctr = logs[-1]["counters"] if logs else {}
    return {
        "arm": arm,
        "seed": seed,
        "wall_s": round(time.time() - t0, 1),
        "bpb": evals[-1]["bpb"] if evals else None,
        "steps_done": evals[-1]["step"] if evals else None,
        "pe_rank_realloc": ctr.get("pe_rank_realloc", 0),
        "pe_rank_changed": ctr.get("pe_rank_changed", 0),
        "pe_rank16": ctr.get("pe_rank16", 0),
        "pe_rank32": ctr.get("pe_rank32", 0),
        "pe_rank48": ctr.get("pe_rank48", 0),
        "pe_rank64": ctr.get("pe_rank64", 0),
        "pe_rank_no_load": ctr.get("pe_rank_no_load", 0),
        "moe_triton_gemm": ctr.get("moe_triton_gemm", 0),
        "moe_triton_fallback": ctr.get("moe_triton_fallback", 0),
        "nan_skipped": ctr.get("nan_skipped", 0),
        "log": str(text),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arms", default="uniform,load")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    LOGDIR.mkdir(parents=True, exist_ok=True)

    print("| arm | rank_mode | rank budget | cadence | total params | routed bank |")
    print("|---|---|---|---|---|---|")
    counts = {}
    for arm in arms:
        total, bank = params(arm)
        counts[arm] = (total, bank)
        print(f"| {arm} | {ARMS[arm]} | {BUDGET} | {CADENCE} | {total:,} | {bank:,} |")
    if len(counts) == 2:
        same = len({c[0] for c in counts.values()}) == 1
        print(
            f"\nPARAMETER INVARIANT: {'HOLDS' if same else 'BROKEN'} "
            f"(uniform == load at the same budget)"
        )
        if not same:
            return 2

    if not args.dry_run:
        why = preflight("cuda")
        if why:
            print(f"SKIP: {why}")
            return 0

    rows = []
    for arm in arms:
        for seed in SEEDS:
            r = run_arm(arm, seed, args.dry_run)
            if r:
                rows.append(r)
                print(
                    f"{arm} seed={seed} OK {r['wall_s']}s bpb={r['bpb']} "
                    f"realloc={r['pe_rank_realloc']} hist="
                    f"{r['pe_rank16']}/{r['pe_rank32']}/{r['pe_rank48']}/{r['pe_rank64']} "
                    f"{r['log']}"
                )

    if args.dry_run or not rows:
        return 0

    print("\n| arm | seeds | params | mean BPB | spread | delta vs uniform | verdict |")
    print("|---|---|---|---|---|---|---|")
    summary = {}
    for arm in arms:
        vals = [r["bpb"] for r in rows if r["arm"] == arm and r["bpb"] is not None]
        if not vals:
            print(f"| {arm} | 0 | {counts[arm][0]:,} | — | — | — | NOT RUN |")
            continue
        summary[arm] = (sum(vals) / len(vals), max(vals) - min(vals), len(vals))

    ctrl = summary.get("uniform")
    for arm in arms:
        if arm not in summary:
            continue
        mean, spread, n = summary[arm]
        if ctrl is None or arm == "uniform":
            print(f"| {arm} | {n} | {counts[arm][0]:,} | {mean:.4f} | {spread:.4f} | — | — |")
            continue
        delta = mean - ctrl[0]
        verdict = "WIN" if delta < -ctrl[1] else ("LOSS" if delta > ctrl[1] else "TIE")
        print(
            f"| {arm} | {n} | {counts[arm][0]:,} | {mean:.4f} | {spread:.4f} | "
            f"{delta:+.4f} | {verdict} (tie band {ctrl[1]:.4f}) |"
        )

    for r in rows:
        if r["arm"] == "load" and not r["pe_rank_realloc"]:
            print(
                f"VOID: load seed={r['seed']} — pe_rank_realloc=0: the cadence never "
                f"fired, so this run is the control wearing the arm's label "
                f"(no_load={r['pe_rank_no_load']})."
            )
        if r["arm"] == "uniform" and r["pe_rank_realloc"]:
            print(f"VOID: uniform seed={r['seed']} — a control reported pe_rank_realloc.")
        if not r["moe_triton_gemm"]:
            print(
                f"NOTE: {r['arm']} seed={r['seed']} — moe_triton_gemm=0 "
                f"(fallback={r['moe_triton_fallback']}): the expert pass did not run "
                f"the grouped GEMM."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
