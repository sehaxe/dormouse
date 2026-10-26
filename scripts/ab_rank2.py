r"""Rank-2 in PER-EXPERT SUBSPACES: is a shared dictionary enough to give rank 2 life?

WHY THIS DRIVER EXISTS. `scripts/ab_rank.py` measured the shared-PLANE ladder:
one input basis for the whole bank (`share_u`), each expert keeping only `s` and
`V`. At rank 2 that bank can "see" exactly two directions of a `d`-dim state, and
that is structural — two directions cannot encode a `d`-dim input whatever the
training does. `share_u_mode="dict"` is the refinement that can: ONE
`[d, share_u_rank]` dictionary per site, shared by every expert, and each expert
reads `rank` of its columns (`idx [2, E, rank]`, fixed and seeded, never
learned). Two experts then see DIFFERENT `rank`-dimensional subspaces of the same
dictionary, at the same order of parameter count as the plane.

ARMS (3 seeds x 2000 steps, preset `medium`, one eval window per arm)
    r64   control: rank 64, per-expert U — the bank every number on record has
    r2s   rank 2, plane: ONE shared basis (today's `share_u`, the arm to beat)
    r2d   rank 2, dict:  R=64 dictionary, 2 columns per expert
    r4d   rank 4, dict:  R=64 dictionary, 4 columns per expert
`rank` is MODEL-WIDE (`out_proj` and `lm_head` too, as in the battery's rank
grid), so each arm's param delta includes those two small projections; the bank
arithmetic below is what the mechanism changes.

WHAT EACH COMPARISON ANSWERS
    r2d vs r2s   do PER-EXPERT SUBSPACES help at the same order of params?
                 (1 464 320 vs 1 402 880 of routed bank — 4% apart, so this is
                 the mechanism's own A/B, not a compression claim)
    r2d vs r64   does a 53x smaller bank reach the control? (AGENTS 1.2)
    r4d vs r2d   does the subspace width buy anything at a fixed dictionary?

PARAM ARITHMETIC (medium: d=1024, f=2730, E=256 routed, shared_rank=192)

    per routed expert, private U:      2*(d*r + r) + r*f
    per routed expert, plane:                2*r  + r*f   (u is shared)
    per routed expert, dict:                 2*r  + r*f   (u_dict is shared,
                                                            `r` COLUMNS of it)
    shared basis, plane, once per site:  2*d*r
    shared dictionary, dict, once per site:  d*R

    arm   rank   private-U bank   basis/dict   shared expert   bank total   vs r64
    r64    64       78 315 520          —            917 760     79 233 280   1.000x
    r2s     2                —      4 096            917 760      2 320 640   0.029x
    r2d     2                —     65 536            917 760      2 382 080   0.030x
    r4d     4                —     65 536            917 760      3 780 864   0.048x

    ("bank total" is what the `routed bank` column below prints: the routed
     experts plus the always-on shared expert, which no arm binds — it runs at
     `shared_rank`, a different shape, and E = 1. `r2d` over `r2s` is exactly
     one dictionary minus the two rank-2 planes: 65 536 - 2*1024*2 = 61 440.)

VERDICT RULE (AGENTS 1.2): an arm must beat the CONTROL's 3-seed spread to be a
win, and a tie deletes the mechanism. A smaller bank that lands inside the spread
is a tie for a cheaper mechanism — report it as such, not as a win. The claim
this driver can make is about BPB at a fixed budget; it is NOT a claim that rank 2
is "enough".

VOID CHECK: every arm must report the counter its label names and NO other —
`r2s` → `share_u_forward > 0`, `share_u_dict_forward == 0`; `r2d`/`r4d` → the
reverse; `r64` → neither. A run whose counters disagree measured a different
model than its label, and its BPB is not a number. `moe_triton_gemm == 0` is
reported as a NOTE (the expert pass did not take the grouped GEMM; the dropless
route falls back COUNTED), not silently dropped.

USAGE
    uv run --extra gpu python scripts/ab_rank2.py --dry-run   # params + commands
    uv run --extra gpu python scripts/ab_rank2.py             # preflight + 12 runs
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
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import data_missing, preflight  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOGDIR = Path.home() / "logs" / "ab_rank2"
DATA = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"
EVAL = "/home/sehaxe/dormouse/data_local/eval_mix/eval_mix.bin"

SEEDS = (1, 2, 3)
STEPS = 2000

#: `/tmp/opencode/battery.sh`'s COMMON, verbatim: one recipe, one eval window
#: (81 820 B = 20 x 8 x 512), so a BPB here is comparable to the battery's.
BASE = ["train", "medium", "--device", "cuda", "--data", DATA, "--eval", EVAL]
COMMON = [
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
]

#: The ladder. Every arm spells `share_u_mode` out — the string is the ONE
#: switch (`share_u = true` is its `plane` spelling, and the driver never needs
#: the alias), so a control is unmistakably a control in its own command line.
ARMS: dict[str, dict[str, object]] = {
    "r64": {"rank": 64, "share_u_mode": "off", "share_u_rank": 64},
    "r2s": {"rank": 2, "share_u_mode": "plane", "share_u_rank": 64},
    "r2d": {"rank": 2, "share_u_mode": "dict", "share_u_rank": 64},
    "r4d": {"rank": 4, "share_u_mode": "dict", "share_u_rank": 64},
}

#: The counter each arm must report, and the one it must NOT (`None` = neither).
COUNTERS: dict[str, str | None] = {
    "r64": None,
    "r2s": "share_u_forward",
    "r2d": "share_u_dict_forward",
    "r4d": "share_u_dict_forward",
}


def params(arm: str) -> tuple[int, int]:
    """`(total, routed bank)` for an arm, from the INSTANTIATED model on `meta`.

    Built, not formula-derived: the preset's count is a property of the module
    tree (`param_count`'s own rule), and a header that drifts from the tree is
    the number that reads true and lies.
    """
    from dormouse.config import load_preset, validate
    from dormouse.model import DormouseModel, param_count

    cfg = dataclasses.replace(load_preset("medium"), **ARMS[arm])
    validate(cfg)
    model = DormouseModel(cfg, device="meta")
    bank = sum(
        p.numel()
        for n, p in model.named_parameters()
        if n.startswith("loop.moe.experts.")
        or n in ("loop.moe.gate_u", "loop.moe.up_u", "loop.moe.u_dict")
        or n.startswith("loop.moe.shared.")
    )
    return param_count(model), bank


def _literal(value: object) -> str:
    """A CLI `--set` value in TOML's spelling — lowercase booleans, else `str`."""
    return ("true" if value else "false") if isinstance(value, bool) else str(value)


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
        f"ckpt_name=abrank2_{arm}_s{seed}",
        "--set",
        "out_dir=/tmp/opencode/ab_rank2_ckpt",
        *[
            tok
            for k in ("rank", "share_u_mode", "share_u_rank")
            for tok in ("--set", f"{k}={_literal(ARMS[arm][k])}")
        ],
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
        "share_u_forward": ctr.get("share_u_forward", 0),
        "share_u_dict_forward": ctr.get("share_u_dict_forward", 0),
        "moe_dropless": ctr.get("moe_dropless", 0),
        "moe_triton_gemm": ctr.get("moe_triton_gemm", 0),
        "moe_triton_fallback": ctr.get("moe_triton_fallback", 0),
        "nan_skipped": ctr.get("nan_skipped", 0),
        "log": str(text),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arms", default="r64,r2s,r2d,r4d")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    LOGDIR.mkdir(parents=True, exist_ok=True)

    print("| arm | rank | mode | R | total params | routed bank | vs control bank |")
    print("|---|---|---|---|---|---|---|")
    counts = {}
    for arm in arms:
        total, bank = params(arm)
        counts[arm] = (total, bank)
        ratio = f"{bank / counts['r64'][1]:.3f}x" if "r64" in counts else "—"
        print(
            f"| {arm} | {ARMS[arm]['rank']} | {ARMS[arm]['share_u_mode']} | "
            f"{ARMS[arm]['share_u_rank']} | {total:,} | {bank:,} | {ratio} |"
        )

    if not args.dry_run:
        why = preflight("cuda") or data_missing(DATA, EVAL)
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
                    f"share_u_forward={r['share_u_forward']} "
                    f"share_u_dict_forward={r['share_u_dict_forward']} {r['log']}"
                )

    if args.dry_run or not rows:
        return 0

    print("\n| arm | seeds | params | mean BPB | spread | delta vs r64 | verdict |")
    print("|---|---|---|---|---|---|---|")
    summary = {}
    for arm in arms:
        vals = [r["bpb"] for r in rows if r["arm"] == arm and r["bpb"] is not None]
        if not vals:
            print(f"| {arm} | 0 | {counts[arm][0]:,} | — | — | — | NOT RUN |")
            continue
        summary[arm] = (sum(vals) / len(vals), max(vals) - min(vals), len(vals))

    ctrl = summary.get("r64")
    for arm in arms:
        if arm not in summary:
            continue
        mean, spread, n = summary[arm]
        if ctrl is None or arm == "r64":
            print(f"| {arm} | {n} | {counts[arm][0]:,} | {mean:.4f} | {spread:.4f} | — | — |")
            continue
        delta = mean - ctrl[0]
        verdict = "WIN" if delta < -ctrl[1] else ("LOSS" if delta > ctrl[1] else "TIE")
        print(
            f"| {arm} | {n} | {counts[arm][0]:,} | {mean:.4f} | {spread:.4f} | "
            f"{delta:+.4f} | {verdict} (tie band {ctrl[1]:.4f}) |"
        )

    # The mechanism's own comparison first: same order of params, different
    # subspaces. Printed against the SPREAD of the plane arm, not the control's.
    if "r2s" in summary and "r2d" in summary:
        mean_s, spread_s, _ = summary["r2s"]
        mean_d = summary["r2d"][0]
        print(
            f"\nr2d vs r2s (per-expert subspaces, {counts['r2d'][1] - counts['r2s'][1]:,} more "
            f"bank params): {mean_d - mean_s:+.4f} BPB, plane spread {spread_s:.4f}"
        )

    void = []
    for r in rows:
        want = COUNTERS[r["arm"]]
        got = {
            "share_u_forward": r["share_u_forward"],
            "share_u_dict_forward": r["share_u_dict_forward"],
        }
        if want is None:
            if any(got.values()):
                void.append((r, f"off arm reported {got}"))
        elif not got[want or "share_u_forward"] or any(v for k, v in got.items() if k != want):
            void.append((r, f"labelled {want} but reported {got}"))
    for r, why in void:
        print(f"VOID: {r['arm']} seed={r['seed']} — {why}")
    dead = [r for r in rows if not r["moe_triton_gemm"]]
    for r in dead:
        print(
            f"NOTE: {r['arm']} seed={r['seed']} — moe_triton_gemm=0 "
            f"(dropless={r['moe_dropless']}, fallback={r['moe_triton_fallback']}): "
            f"the expert pass did not run the grouped GEMM"
        )

    with (LOGDIR / "summary.jsonl").open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
