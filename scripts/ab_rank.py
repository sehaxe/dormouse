r"""The rank LADDER with a SHARED input basis: can a rank-2 expert bank work?

WHY THIS DRIVER EXISTS. Every rank measurement so far prices a bank whose experts
own their own `U [d, r]`: rank 16 costs +0.136 BPB against rank 64, rank 8
collapsed, and "compose many weak experts" was already tried and failed (rank 16
with topk 4: 3.2017 vs the 3.0750 control, +0.127). A private rank-2 basis is
two directions of noise per expert, so pure low rank is hopeless BY CONSTRUCTION.
The one mechanism left that can give the bank capacity at rank 2 is SHARING: one
input basis for all experts (`share_u`, `experts.py`), so the capacity lives in a
shared dictionary of input directions and each expert keeps only its scale `s`
and its write `V [r, f]`. This driver MEASURES that; it does not claim rank 2
works.

ARMS (3 seeds x 2000 steps, preset `medium`, one eval window per arm)
    r64   control: rank 64, per-expert U — the bank every number on record has
    r32s  rank 32, shared U
    r8s   rank 8,  shared U
    r2s   rank 2,  shared U
`rank` is MODEL-WIDE (`out_proj` and `lm_head` too, as in the battery's rank
grid), so each arm's param delta includes those two small projections; the
bank arithmetic below is what the mechanism changes.

PARAM ARITHMETIC (medium: d=1024, f=2730, E=256 routed, shared_rank=192)

    per routed expert, private U:  2*(d*r + r) + r*f     (gate_u, up_u, scales, V)
    per routed expert, shared U:        2*r  + r*f       (scales, V; U is shared)
    shared basis, once per site:   2*d*r                 (gate and up, NOT the down
                                                         write: `down` is ONE shared
                                                         nn.Linear already, and the
                                                         always-on shared expert is
                                                         rank 192, so it keeps its
                                                         own basis)

    arm   rank   routed bank (private U)   shared basis   shared expert   bank total
    r64    64          78 315 520                   —          917 760     79 233 280
    r32s   32                   —               65 536         917 760     23 363 840
    r8s     8                   —               16 384         917 760      6 529 280
    r2s     2                   —                4 096         917 760      2 320 640

    i.e. r64 removes 256*2*1024*64 = 33 554 432 param and adds 2*1024*64 = 131 072:
    the bank drops to 57.8% of the control. r2s is 1.4M of routed bank against
    78.3M — a 55.8x compression of the expert bank, and it is the arm this ladder
    is for. (`r64s` — rank 64 shared, the clean "sharing at matched rank"
    isolation — is NOT in this ladder; it is the follow-up that separates the
    sharing effect from the rank effect. This ladder asks the owner's question:
    does SOME shared-basis rank reach the control.)

VERDICT RULE (AGENTS 1.2): an arm must beat the CONTROL's 3-seed spread to be a
win, and a tie deletes the mechanism. A rank arm that lands inside the spread is
a tie for a much smaller bank — report it as such, not as a win.

VOID CHECK: every `s` arm must report `share_u_forward > 0` (the arm's own
counter) and the control must NOT (it owns a per-expert basis). A run whose
counters disagree measured a different model than its label, and its BPB is not
a number.

USAGE
    uv run --extra gpu python scripts/ab_rank.py --dry-run   # params + commands
    uv run --extra gpu python scripts/ab_rank.py             # preflight + 12 runs
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
from _lane import preflight  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOGDIR = Path.home() / "logs" / "ab_rank"
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

#: The ladder. `r64` is the control and spells out BOTH of its switches, so the
#: arm's own line says what it is; the `s` arms differ only in `share_u`.
ARMS: dict[str, dict[str, object]] = {
    "r64": {"rank": 64, "share_u": False},
    "r32s": {"rank": 32, "share_u": True},
    "r8s": {"rank": 8, "share_u": True},
    "r2s": {"rank": 2, "share_u": True},
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
        or n in ("loop.moe.gate_u", "loop.moe.up_u")
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
        f"ckpt_name=abrank_{arm}_s{seed}",
        "--set",
        "out_dir=/tmp/opencode/ab_rank_ckpt",
        # TOML booleans are lowercase. `--set share_u=False` does NOT parse (the
        # CLI's fallback keeps the raw string, which is truthy), so the control
        # would run the SHARED arm wearing the control's label — measured
        # 2026-10-21: `dormouse info medium --set share_u=False` reported
        # 108 487 045 params instead of 141 910 405.
        *[tok for k in ("rank", "share_u") for tok in ("--set", f"{k}={_literal(ARMS[arm][k])}")],
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
        "moe_dropless": ctr.get("moe_dropless", 0),
        "moe_triton_gemm": ctr.get("moe_triton_gemm", 0),
        "moe_triton_fallback": ctr.get("moe_triton_fallback", 0),
        "nan_skipped": ctr.get("nan_skipped", 0),
        "log": str(text),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arms", default="r64,r32s,r8s,r2s")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    LOGDIR.mkdir(parents=True, exist_ok=True)

    print("| arm | rank | shared U | total params | routed bank | vs control bank |")
    print("|---|---|---|---|---|---|")
    counts = {}
    for arm in arms:
        total, bank = params(arm)
        counts[arm] = (total, bank)
        ratio = f"{bank / counts['r64'][1]:.3f}x" if "r64" in counts else "—"
        print(
            f"| {arm} | {ARMS[arm]['rank']} | {ARMS[arm]['share_u']} | {total:,} | "
            f"{bank:,} | {ratio} |"
        )

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
                    f"share_u_forward={r['share_u_forward']} {r['log']}"
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

    void = [
        r
        for r in rows
        if (ARMS[r["arm"]]["share_u"] and not r["share_u_forward"])
        or (not ARMS[r["arm"]]["share_u"] and r["share_u_forward"])
    ]
    for r in void:
        print(f"VOID: {r['arm']} seed={r['seed']} — share_u_forward={r['share_u_forward']}")
    dead = [r for r in rows if not r["moe_triton_gemm"]]
    for r in dead:
        print(
            f"NOTE: {r['arm']} seed={r['seed']} — moe_triton_gemm=0 "
            f"(dropless={r['moe_dropless']}, fallback={r['moe_triton_fallback']}): "
            f"the expert pass did not run the grouped GEMM"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
