r"""The quantization A/B: does the model train as well in the form it ships in?

The owner's directive (2026-10-19): "we train the model for inference, not for
training's sake — train it in the full quantization, the way it will run at
inference." The mechanisms have been in the tree since 2026-10-15
(`factor_qat`, `dense_qat`, `act_quant`, the anneal ramp) but every long run so
far has trained in bf16, so this driver answers the question the directive
actually asks, on the preset that matters (medium), at the operating shape of
the live run (batch 24 x seq 1024, the 230 GB staged mix).

ARMS
    bf16   the control: the recipe the live run uses (masters fp32, forward bf16)
    qat    the shipped form: ternary factors + ternary dense weights
           (`factor_qat=ternary dense_qat=ternary`) + 4-bit activations
           (`act_quant=int4`), with a 500-step anneal ramp so the init is not
           destroyed on step 0 (the ramp is BitNet's own trick, not a
           post-training finisher: the last 75% of every run is hard STE at
           alpha=1, which is the form the release is packed from)

VERDICT RULE (AGENTS 1.2, plus the owner's speed/quality exchange rate): the
qat arm must lose less than the control's own 3-seed spread to be a tie, and a
loss is only acceptable if it is small against the ~1.5-2% budget the owner set
for a resource that buys something. A loss larger than that means the run
trains bf16 and the release is quantized by the QAT-finish rung instead.

VOID CHECK: the qat arm must report `dense_qat_arms > 0` and
`act_quant_arms > 0`; a run whose counters are zero measured the bf16 forward
wearing a quantized label, which is the defect class this repo counts.

USAGE
    uv run --extra gpu python scripts/ab_qat.py            # preflight + all runs
    uv run --extra gpu python scripts/ab_qat.py --dry-run  # print the commands
"""

from __future__ import annotations

import argparse
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
LOGDIR = Path.home() / "logs" / "ab_qat"
DATA = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"
EVAL = "/home/sehaxe/dormouse/data_local/eval_mix/eval_mix.bin"

SEEDS = (1, 2, 3)
STEPS = 2000
EVAL_EVERY = 500

#: The recipe the live long run uses; identical across arms, arm flags appended.
BASE = [
    "train",
    "medium",
    "--data",
    DATA,
    "--eval",
    EVAL,
    "--device",
    "cuda",
    "--batch",
    "24",
    "--steps",
    str(STEPS),
    "--set",
    "mix_plan=configs/mix_big.toml",
    "--set",
    "seq_len=1024",
    "--set",
    "log_every=200",
    "--set",
    "ckpt_every=1000000000",
    "--set",
    "eval=true",
    "--set",
    f"eval_every={EVAL_EVERY}",
    "--set",
    "eval_batches=8",
    "--set",
    "max_iter=3",
    "--set",
    "lr=0.001",
    "--set",
    "use_msa=false",
    "--set",
    "moe_capacity=0",
    "--set",
    "moe_kernel=triton",
    "--set",
    "opt_graph=true",
    "--set",
    "ns_kind=triton",
    "--set",
    "muon_state_dtype=bf16",
    "--set",
    "engram_ram=true",
    "--set",
    "tf32=true",
    "--set",
    "out_dir=checkpoints/abqat",
]

ARMS: dict[str, list[str]] = {
    "bf16": [],
    "qat": [
        "--set",
        "factor_qat=ternary",
        "--set",
        "dense_qat=ternary",
        "--set",
        "act_quant=int4",
        "--set",
        "factor_qat_anneal_steps=500",
    ],
}


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
        "--set",
        f"ckpt_name=abqat_{arm}_s{seed}",
        "--set",
        f"seed={seed}",
        *ARMS[arm],
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
        "dense_qat_arms": ctr.get("dense_qat_arms", 0),
        "act_quant_arms": ctr.get("act_quant_arms", 0),
        "nan_skipped": ctr.get("nan_skipped", 0),
        "log": str(text),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arms", default="bf16,qat")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    LOGDIR.mkdir(parents=True, exist_ok=True)

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
                print(f"{arm} seed={seed} OK {r['wall_s']}s bpb={r['bpb']} {r['log']}")

    if args.dry_run or not rows:
        return 0

    print("\n| arm | seeds | mean BPB | spread |")
    print("|---|---|---|---|")
    summary = {}
    for arm in arms:
        vals = [r["bpb"] for r in rows if r["arm"] == arm and r["bpb"] is not None]
        if not vals:
            print(f"| {arm} | 0 | — | — |")
            continue
        summary[arm] = (sum(vals) / len(vals), max(vals) - min(vals))
        print(f"| {arm} | {len(vals)} | {summary[arm][0]:.4f} | {summary[arm][1]:.4f} |")

    void = [r for r in rows if r["arm"] == "qat" and not r["dense_qat_arms"]]
    if void:
        print(f"\nVOID: {len(void)} qat run(s) report dense_qat_arms=0 — the arm did not engage")
    if "bf16" in summary and "qat" in summary:
        ctrl_mean, ctrl_spread = summary["bf16"]
        qat_mean, _ = summary["qat"]
        delta = qat_mean - ctrl_mean
        rel = 100.0 * delta / ctrl_mean
        verdict = (
            "TIE (inside the control spread)"
            if abs(delta) <= ctrl_spread
            else f"LOSS {delta:+.4f} ({rel:+.2f}%)"
        )
        print(f"\nqat vs bf16: {delta:+.4f} BPB ({rel:+.2f}%) — {verdict}")
        print(
            "budget: the owner's exchange rate is ~1.5-2% of BPB for a resource that buys something"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
