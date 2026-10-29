r"""Does TSCT capacity grow with RANK? — the rank-UP sweep, run correctly.

WHY THIS DRIVER EXISTS. `~/logs/rank_up.sh` asked the one axis never measured
(the ladder had only gone DOWN: rank 64 -> 32 -> 16 cost +0.027 / +0.136 BPB) and
every arm with `rank >= 192` died in ~14 s at KERNEL SELECTION, before step 1:

    AssertionError: bf16 tile BLOCK_N=128 needs 163840 bytes > 101376

That was the tile heuristic's number, not the device's (`moe_kernels.smem_estimate`
charged the GLU's reduction tile with the RANK tile; the compiled kernel measures
98 304 B at rank 192 and fits). The fix is an ordered tile SEARCH
(`moe_kernels.TILE_SEARCH`) plus the corrected estimate; this driver is the sweep
that can now run, and it refuses to start while any arm's tile cannot be selected.

ARMS (constant PARAMETER budget — the whole point: rank trades against expert
count, not against capacity):

    arm       rank  n_experts   bank per expert (share_u on: 2*d*r + r*f)
    r64e256     64        256    305 792   -> 78.28M  (the control)
    r128e128   128        128    611 584   -> 78.28M
    r192e85    192         85    917 376   -> 77.98M  (-0.4%, inside the 1% band)
    r256e64    256         64  1 223 168   -> 78.28M

at `medium`'s d=1024 / f=2730 / topk=2, so MAC/token rises with rank while the
parameters sit still: this sweep measures "same capacity, more compute per
token", which is the question a `rank` knob actually asks.

PROTOCOL. 3 seeds x 1000 steps, batch 8 x seq 1024, one eval window per arm
(`eval_batches=8` -> 65 536 B), the COMMON flags of `~/logs/rank_up.sh` verbatim
(`use_msa=false`, `use_csa=true`, `use_mhc=true`, `use_attnres=true`,
`share_u=true`, `attn_latent_rank=256`, `moe_kernel=triton`, `moe_capacity=0`)
plus `opt_graph=false`. A BPB is comparable only inside one window (AGENTS 2.6),
and the driver prints each arm's resolved config so the comparison can be checked
before it is believed.

PRE-REGISTERED VERDICT RULE (fixed BEFORE the runs, AGENTS 1.2): an arm enters
the recipe only if (a) its measured parameter count is within 1% of the control's
— the budget is constant or the comparison is not the one asked — AND (b) its
BPB is within 2x the control's own 3-seed spread. Ties delete the arm; a bigger
number at constant parameters is a LOSS, not a "close".

VOID CHECKS (a run that measured a different model than its label is not a
number): `moe_triton_gemm > 0` and `moe_triton_fallback == 0` in every arm — on
the pre-fix kernel the rank>=192 arms produced NEITHER (they died), and any
`moe_tile_N*_M*_st*` other than the base tile says the arm paid a tile (a
pipeline stage or half of BLOCK_N) to fit, which is a fact about the shape, not a
reason to discount the BPB.

USAGE
    python scripts/ab_rank_up.py --dry-run   # resolved configs, params, tiles, commands
    python scripts/ab_rank_up.py             # preflight + 12 runs (~unknown: never run)
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
from _lane import data_missing, preflight  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOGDIR = Path.home() / "logs" / "ab_rank_up"
#: `~/logs/rank_up.sh`'s pair, absolute because `data_local/` is not in a worktree.
DATA = "/home/sehaxe/dormouse/data_local/best/fwedu.bin"
EVAL = "/home/sehaxe/dormouse/data_local/mix4/web_eval.bin"

PRESET = "medium"
SEEDS = (1, 2, 3)
STEPS = 1000
#: The constant-budget band: an arm outside it is not at the control's capacity.
BUDGET_TOL = 0.01

ARMS: dict[str, tuple[int, int]] = {
    "r64e256": (64, 256),  # the control: today's recipe
    "r128e128": (128, 128),
    "r192e85": (192, 85),
    "r256e64": (256, 64),
}

#: `~/logs/rank_up.sh`'s COMMON, verbatim (+ `opt_graph=false`): one recipe and
#: one eval window, so a BPB is comparable across the arms AND to that sweep's log.
COMMON = [
    "--set",
    f"steps={STEPS}",
    "--set",
    "batch=8",
    "--set",
    "seq_len=1024",
    "--set",
    "max_seq_len=1024",
    "--set",
    "grad_accum=1",
    "--set",
    "lr=5e-4",
    "--set",
    "warmup_tokens=16384000",
    "--set",
    "decay_frac=0.1",
    "--set",
    "max_iter=3",
    "--set",
    "use_msa=false",
    "--set",
    "use_csa=true",
    "--set",
    "use_mhc=true",
    "--set",
    "use_attnres=true",
    "--set",
    "share_u=true",
    "--set",
    "use_engram=true",
    "--set",
    "engram_ram=false",
    "--set",
    "attn_latent_rank=256",
    "--set",
    "moe_kernel=triton",
    "--set",
    "moe_capacity=0.0",
    "--set",
    "doc_masking=false",
    "--set",
    f"data={DATA}",
    "--set",
    f"eval_data={EVAL}",
    "--set",
    "eval=true",
    "--set",
    "eval_every=500",
    "--set",
    "eval_batches=8",
    "--set",
    "log_every=250",
    "--set",
    "ckpt_every=1000000",
    "--set",
    "opt_graph=false",
]

#: The resolved MODEL fields a reader needs to check that the arms differ ONLY in
#: rank/expert count — the comparison's premise, printed rather than assumed.
SHOWN = (
    "d_model",
    "d_ffn",
    "rank",
    "n_experts",
    "share_u",
    "attn_latent_rank",
    "max_seq_len",
    "max_iter",
    "use_msa",
    "use_csa",
    "use_mhc",
    "use_attnres",
    "use_engram",
    "moe_kernel",
    "moe_capacity",
    "bf16",
    "quant",
)

#: The resolved RUN fields — the protocol a BPB is only comparable inside of
#: (AGENTS 2.6: `eval_batches x batch x seq_len` is the window).
RUN_SHOWN = (
    "steps",
    "batch",
    "seq_len",
    "grad_accum",
    "lr",
    "warmup_tokens",
    "decay_frac",
    "doc_masking",
    "engram_ram",
    "eval",
    "eval_every",
    "eval_batches",
    "log_every",
    "ckpt_every",
    "opt_graph",
)


def _sets(arm: str) -> list[str]:
    """The arm's `k=v` overrides: `COMMON` verbatim plus the two knobs it moves."""
    rank, experts = ARMS[arm]
    return [COMMON[i + 1] for i in range(0, len(COMMON), 2)] + [
        f"rank={rank}",
        f"n_experts={experts}",
    ]


def resolved(arm: str):
    """The arm's resolved `RunConfig` — the CLI's OWN resolver (`load_run`).

    `dormouse info` resolves a run exactly this way, and a driver that restated
    "which `--set` lands on the model" would drift from it: `share_u=true` is a
    MODEL key, so a param count taken without the overrides (`ab_rank.params`'
    shape) counts the private basis and prices the wrong bank.
    """
    from dormouse.cli import _split_sets
    from dormouse.config import load_run

    model_over, train_over = _split_sets(_sets(arm))
    return load_run(PRESET, model_overrides=model_over, train_overrides=train_over)


def params(arm: str) -> tuple[int, int]:
    """`(total, routed bank)` from the INSTANTIATED model on `meta`, not a formula.

    Same rule as `ab_rank.params`: a header that drifts from the module tree is
    the number that reads true and lies. Built from the RESOLVED model config, so
    the bank is the one the run gets (`share_u=true`).
    """
    from dormouse.model import DormouseModel, param_count

    model = DormouseModel(resolved(arm).model, device="meta")
    bank = sum(
        p.numel()
        for n, p in model.named_parameters()
        if n.startswith("loop.moe.experts.")
        or n in ("loop.moe.gate_u", "loop.moe.up_u")
        or n.startswith("loop.moe.shared.")
    )
    return param_count(model), bank


def tile(arm: str) -> str:
    """The tile this arm's shape selects — `?` when the module cannot be imported.

    On the pre-fix kernel this RAISES for the rank>=192 arms (163840 B > 101376),
    which is exactly the crash the sweep died of; the driver refuses to start
    rather than launch a sweep that cannot run.
    """
    import torch

    try:
        from dormouse.model import moe_kernels as mk
    except ImportError:  # a CPU-only wheel: no triton, no tile to name
        return "?"
    cfg = resolved(arm).model
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    return mk.tile_name(dtype, cfg.d_model, cfg.d_ffn, cfg.rank)


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
    rank, experts = ARMS[arm]
    log = LOGDIR / f"{arm}_s{seed}.jsonl"
    text = LOGDIR / f"{arm}_s{seed}.log"
    cmd = [
        sys.executable,
        "-m",
        "dormouse.cli",
        "train",
        PRESET,
        *COMMON,
        "--set",
        f"rank={rank}",
        "--set",
        f"n_experts={experts}",
        "--set",
        f"seed={seed}",
        "--set",
        f"ckpt_name=rankup_{arm}_s{seed}",
        "--set",
        "out_dir=/tmp/opencode/ab_rank_up",
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
        "moe_triton_gemm": ctr.get("moe_triton_gemm", 0),
        "moe_triton_fallback": ctr.get("moe_triton_fallback", 0),
        "moe_tile": next((k for k in ctr if k.startswith("moe_tile_")), "none"),
        "nan_skipped": ctr.get("nan_skipped", 0),
        "log": str(text),
    }


def budget(arms: list[str]) -> tuple[dict, bool]:
    """The parameter table, and whether every arm is inside `BUDGET_TOL`.

    LOUD: an arm outside the band did not measure the constant-budget question
    the sweep exists for, so `main` refuses to launch it as if it had.
    """
    counts = {arm: params(arm) for arm in arms}
    ctrl = counts[arms[0]][0]
    print("| arm | rank | n_experts | total params | routed bank | vs control | tile |")
    print("|---|---|---|---|---|---|---|")
    ok = True
    for arm in arms:
        total, bank = counts[arm]
        drift = (total - ctrl) / ctrl
        ok = ok and abs(drift) <= BUDGET_TOL
        print(
            f"| {arm} | {ARMS[arm][0]} | {ARMS[arm][1]} | {total:,} | {bank:,} | "
            f"{drift:+.3%} | {tile(arm)} |"
        )
    return counts, ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--arms",
        default=",".join(ARMS),
        help="comma list; the CONTROL must come first (it is the 1%% budget reference)",
    )
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    unknown = set(arms) - set(ARMS)
    if unknown:
        print(f"unknown arm(s) {sorted(unknown)}; known: {list(ARMS)}")
        return 2

    counts, ok = budget(arms)
    for arm in arms:
        run = resolved(arm)
        print(f"\n[{arm}] resolved {run.source}")
        print("  model: " + " ".join(f"{k}={getattr(run.model, k)}" for k in SHOWN))
        print("  train: " + " ".join(f"{k}={getattr(run.train, k)}" for k in RUN_SHOWN))
    if not ok:
        print(
            f"\nREFUSED: an arm is outside {BUDGET_TOL:.0%} of the control's "
            f"{counts[arms[0]][0]:,} params — the budget is not constant, so this is "
            f"not the rank-vs-expert-count question. Fix the arm's n_experts."
        )
        return 1
    if not ok:
        print(
            f"\nREFUSED: an arm is outside {BUDGET_TOL:.0%} of the control's "
            f"{counts[arms[0]][0]:,} params — the budget is not constant, so this "
            f"is not the rank-vs-expert-count question. Fix the arm's n_experts."
        )
        return 1

    LOGDIR.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for arm in arms:
            for seed in SEEDS:
                run_arm(arm, seed, dry=True)
        return 0

    why = preflight("cuda") or data_missing(DATA, EVAL)
    if why:
        print(f"SKIP: {why}")
        return 0

    rows = []
    for arm in arms:
        for seed in SEEDS:
            r = run_arm(arm, seed, dry=False)
            rows.append(r)
            print(
                f"{arm} seed={seed} bpb={r['bpb']} steps={r['steps_done']} "
                f"{r['moe_tile']} gemm={r['moe_triton_gemm']} "
                f"fallback={r['moe_triton_fallback']} {r['wall_s']}s {r['log']}"
            )
    if not rows:
        return 0

    summary = {}
    for arm in arms:
        vals = [r["bpb"] for r in rows if r["arm"] == arm and r["bpb"] is not None]
        if vals:
            summary[arm] = (sum(vals) / len(vals), max(vals) - min(vals), len(vals))
    ctrl = summary.get(arms[0])
    print("\n| arm | n | params | mean BPB | 3-seed spread | delta vs control | verdict |")
    print("|---|---|---|---|---|---|---|")
    for arm in arms:
        if arm not in summary:
            print(f"| {arm} | 0 | {counts[arm][0]:,} | — | — | — | NOT RUN |")
            continue
        mean, spread, n = summary[arm]
        if ctrl is None or arm == arms[0]:
            print(f"| {arm} | {n} | {counts[arm][0]:,} | {mean:.4f} | {spread:.4f} | — | — |")
            continue
        band = 2 * ctrl[1]  # PRE-REGISTERED: 2x the control's own 3-seed spread
        delta = mean - ctrl[0]
        verdict = "ENTERS" if delta < -band else ("LOSS" if delta > band else "TIE -> deleted")
        print(
            f"| {arm} | {n} | {counts[arm][0]:,} | {mean:.4f} | {spread:.4f} | "
            f"{delta:+.4f} | {verdict} (band {band:.4f}) |"
        )

    for r in rows:
        if r["moe_triton_fallback"] or not r["moe_triton_gemm"]:
            print(
                f"VOID: {r['arm']} seed={r['seed']} — gemm={r['moe_triton_gemm']} "
                f"fallback={r['moe_triton_fallback']}: the grouped GEMM did not run, "
                f"so this BPB is not the arm's"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
