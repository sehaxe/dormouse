r"""Activation memory: peak/step cost of `iter_checkpoint` and `activation_offload`.

WHAT THIS MEASURES. The two mechanisms are exact by construction (`tests/test_actmem.py`
pins that), so the only open questions are the ones a CPU session cannot answer:
does per-iteration checkpointing actually drop the DEVICE peak, does the pinned
host offload drop it by the budget, and what do they cost in ms/step. The driver
has two instruments because they answer different halves:

* **peak + ms/step** (`speed_row`): the same model/optimizer the trainer builds,
  in-process, `torch.cuda.max_memory_allocated` after a reset and the median of
  `SPEED_STEPS` synchronized steps. Batches 8/16/24 at s1024, one run per
  (arm, batch) — memory and time do not depend on the seed, only BPB does.
* **BPB** (`run_bpb`): the `dormouse train` CLI, 3 seeds per arm, batch 8 only,
  because a BPB needs data and a step count and a per-seed evaluation.

ARMS (`iter_checkpoint`, `activation_offload`):
    off            false  0.0      the control
    iter           true   0.0      per-iteration checkpointing
    offload4       false  4.0 GiB  pinned host offload
    iter+offload4  true   4.0 GiB  both

PRE-REGISTERED VERDICT RULE (fixed BEFORE the runs, AGENTS 1.2 / AB-PROTOCOL):
an arm is KEPT only if (a) its batch-8 peak is at least 30% below the control's,
AND (b) its mean BPB is inside the control's own 3-seed [min, max] spread, AND
(c) no run skipped a step (`nan_skipped == 0`). A tie is a delete; a peak win
that moves BPB outside the spread is a LOSS, not a near miss. `off` is never
"kept" — it is the control.

OUTPUT. One JSONL row per measurement under `~/logs/ab_actmem/rows.jsonl` (plus
one `<arm>_<batch>_s<seed>.jsonl` per BPB run), and a markdown table on stdout.
The JSONL is append/rewrite by this driver only; a re-run rewrites it so a
half-finished sweep cannot be read as two seeds of one arm.

USAGE (written while the card belongs to the LR ladder — NOT run):
    python scripts/ab_actmem.py --dry-run       # plan, no GPU, no torch.cuda
    python scripts/ab_actmem.py                 # preflight, speed rows, BPB runs
    python scripts/ab_actmem.py --speed-only
    python scripts/ab_actmem.py --bpb-only
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

# `scripts/` is not a package: the shared lane guard is a sibling file, so its
# directory goes on the path APPENDED (a script name at the front would shadow a
# package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import CORPUS, EVAL, data_missing, preflight, skip  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOGDIR = Path.home() / "logs" / "ab_actmem"
PRESET = "medium"
#: The measured context: `medium` d1024, max_iter 3, s1024 (the brief's 7.2 GiB
#: control at b8). `bench_step.py`'s override set is the shipped recipe's shape.
MAX_ITER = 3
BATCHES = (8, 16, 24)
SEQ = 1024
#: Peak/step rows: one run per (arm, batch) — memory does not need a seed.
SPEED_SEED = 1
SPEED_STEPS = 20
SPEED_WARM = 3
#: BPB rows: 3 seeds, batch 8 only (the arm's own spread is the control's band).
BPB_SEEDS = (1, 2, 3)
BPB_STEPS = 300
EVAL_BATCHES = 20
#: The keep rule. Fixed here, before any number exists.
PEAK_CUT = 0.30

#: arm -> (iter_checkpoint, activation_offload GiB)
ARMS: dict[str, tuple[bool, float]] = {
    "off": (False, 0.0),
    "iter": (True, 0.0),
    "offload4": (False, 4.0),
    "iter+offload4": (True, 4.0),
}


def sets(arm: str) -> tuple[bool, float]:
    return ARMS[arm]


def common_sets() -> list[str]:
    """The shipped recipe, verbatim where the live LR ladder's COMMON has it.

    One recipe for every arm so the BPB is comparable across seeds (and to the
    ladder's own `medium/max_iter=3` line); only the two mechanism fields move.
    """
    pairs = [
        ("steps", str(BPB_STEPS)),
        ("batch", "8"),
        ("seq_len", str(SEQ)),
        ("max_seq_len", str(SEQ)),
        ("max_iter", str(MAX_ITER)),
        ("grad_accum", "1"),
        ("lr", "5e-4"),
        ("warmup_tokens", "16384000"),
        ("decay_frac", "0.1"),
        ("use_msa", "false"),
        ("use_csa", "true"),
        ("use_mhc", "true"),
        ("use_attnres", "true"),
        ("share_u", "true"),
        ("use_engram", "true"),
        ("engram_ram", "false"),
        ("attn_latent_rank", "256"),
        ("moe_kernel", "triton"),
        ("moe_capacity", "0.0"),
        ("doc_masking", "false"),
        ("data", CORPUS),
        ("eval_data", EVAL),
        ("eval", "true"),
        ("eval_every", str(BPB_STEPS)),
        ("eval_batches", str(EVAL_BATCHES)),
        ("log_every", "50"),
        ("ckpt_every", "1000000"),
        ("opt_graph", "false"),
        ("full_graph", "false"),
    ]
    out: list[str] = []
    for k, v in pairs:
        out += ["--set", f"{k}={v}"]
    return out


def arm_sets(arm: str) -> list[str]:
    it, off = sets(arm)
    return [
        "--set",
        f"iter_checkpoint={'true' if it else 'false'}",
        "--set",
        f"activation_offload={off}",
    ]


def bpb_cmd(arm: str, seed: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "dormouse.cli",
        "train",
        PRESET,
        *common_sets(),
        *arm_sets(arm),
        "--set",
        f"seed={seed}",
        "--set",
        f"ckpt_name=actmem_{arm.replace('+', '_')}_s{seed}",
        "--set",
        f"out_dir={Path('/tmp/opencode/ab_actmem')}",
        "--json-log",
        str(LOGDIR / f"{arm.replace('+', '_')}_s{seed}.jsonl"),
    ]


def read_rows(path: Path) -> tuple[list[dict], list[dict]]:
    evals: list[dict] = []
    logs: list[dict] = []
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


def speed_row(arm: str, batch: int, *, ckpt_dir: str, ckpt_name: str) -> dict:
    """One in-process peak/step measurement — mirrors `bench_step.py`'s shape.

    `torch` is imported HERE, after the parent has passed the lane preflight: a
    `torch.cuda` call in the parent initialises a context the subprocesses
    inherit (`_lane`'s docstring — the bug that killed 33 scripts once).
    """
    import torch

    from dormouse.cli import _device, _model, _snapshot_run
    from dormouse.config import validate, validate_train
    from dormouse.instrument import Counters
    from dormouse.train.loop import load_model_half, load_model_state
    from dormouse.train.optim import MixOptimizer

    run = _snapshot_run(
        Path(ckpt_dir) / ckpt_name,
        ckpt_name,
        eval_file=None,
        eval_batches=None,
    )
    overrides = {
        "batch": batch,
        "seq_len": SEQ,
        "max_seq_len": SEQ,
        "max_iter": MAX_ITER,
        "mm_dtype": "bf16",
        "attn_latent_rank": 256,
        "out_proj_rank": 0,
        "use_csa": True,
        "use_msa": False,
        "use_mhc": True,
        "use_attnres": True,
        "share_u": True,
        "use_engram": True,
        "moe_kernel": "triton",
        "moe_capacity": 0.0,
        "doc_masking": False,
        "eval": False,
        "steps": SPEED_STEPS + SPEED_WARM,
        "iter_checkpoint": sets(arm)[0],
        "activation_offload": sets(arm)[1],
    }
    for key, value in overrides.items():
        setattr(run.train, key, value)
    validate_train(run.train)
    validate(run.model)
    dev = _device("auto")
    model = _model(run).to(dev).train()
    load_model_state(
        model,
        run.train,
        load_model_half(Path(ckpt_dir) / ckpt_name / f"{ckpt_name}.pt"),
    )
    model.configure_actmem(iter_checkpoint=sets(arm)[0], activation_offload=sets(arm)[1])
    c = run.train
    opt = MixOptimizer(
        model,
        lr=c.lr,
        wd=c.wd,
        opt=c.opt,
        ns_steps=c.ns_steps,
        ns_kind=c.ns_kind,
        colrow_3d=c.muon_colrow_3d,
        muon_state_dtype=c.muon_state_dtype,
        ns_dtype=c.ns_dtype,
        muon_variant=c.muon_variant,
        muon_sf=c.muon_sf,
        muon_sf_wd=c.muon_sf_wd,
    )

    b, t = batch, SEQ
    ids = torch.randint(0, 256, (b, t), device=dev)
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    times: list[float] = []
    for step in range(SPEED_STEPS + SPEED_WARM):
        t0 = time.perf_counter()
        _, loss, _ = model(ids, targets=ids, counters=Counters())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize(dev)
        dt = time.perf_counter() - t0
        if step >= SPEED_WARM:
            times.append(dt)
    peak = torch.cuda.max_memory_allocated(dev) / 2**30
    reserved = torch.cuda.max_memory_reserved(dev) / 2**30
    return {
        "kind": "speed",
        "arm": arm,
        "seed": SPEED_SEED,
        "batch": batch,
        "seq": SEQ,
        "max_iter": MAX_ITER,
        "peak_gib": round(peak, 3),
        "reserved_gib": round(reserved, 3),
        "ms_step": round(statistics.median(times) * 1e3, 2),
        "steps": SPEED_STEPS,
    }


def run_bpb(arm: str, seed: int, *, dry: bool) -> dict:
    tag = arm.replace("+", "_")
    log = LOGDIR / f"{tag}_s{seed}.jsonl"
    text = LOGDIR / f"{tag}_s{seed}.log"
    cmd = bpb_cmd(arm, seed)
    if dry:
        print(" ".join(cmd))
        return {}
    t0 = time.time()
    with text.open("w") as fh:
        subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, check=False)
    evals, logs = read_rows(log)
    counters = logs[-1]["counters"] if logs else {}
    return {
        "kind": "bpb",
        "arm": arm,
        "seed": seed,
        "batch": 8,
        "seq": SEQ,
        "max_iter": MAX_ITER,
        "steps": BPB_STEPS,
        "bpb": evals[-1]["bpb"] if evals else None,
        "steps_done": evals[-1]["step"] if evals else None,
        "nan_skipped": counters.get("nan_skipped", 0),
        "iter_checkpoint_saved": counters.get("iter_checkpoint_saved", 0),
        "offload_tensors": counters.get("offload_tensors", 0),
        "offload_gib_x100": counters.get("offload_gib_x100", 0),
        "wall_s": round(time.time() - t0, 1),
        "log": str(text),
    }


def verdict(rows: list[dict]) -> list[dict]:
    """Apply the pre-registered rule to the batch-8 rows, in one place."""
    speed8 = {r["arm"]: r for r in rows if r["kind"] == "speed" and r["batch"] == 8}
    bpb: dict[str, list[float]] = {}
    for r in rows:
        if r["kind"] == "bpb" and r.get("bpb") is not None:
            bpb.setdefault(r["arm"], []).append(r["bpb"])
    if "off" not in speed8 or "off" not in bpb:
        return []
    ctrl_peak = speed8["off"]["peak_gib"]
    ctrl_bpb = bpb["off"]
    lo, hi = min(ctrl_bpb), max(ctrl_bpb)
    out = []
    for arm in ARMS:
        if arm == "off" or arm not in speed8 or arm not in bpb:
            continue
        cut = 1.0 - speed8[arm]["peak_gib"] / ctrl_peak
        mean = statistics.mean(bpb[arm])
        equal = lo <= mean <= hi
        skipped = any(
            r.get("nan_skipped", 0) for r in rows if r["kind"] == "bpb" and r["arm"] == arm
        )
        keep = cut >= PEAK_CUT and equal and not skipped
        out.append(
            {
                "arm": arm,
                "peak_cut": round(cut, 4),
                "bpb_mean": round(mean, 4),
                "control_band": [lo, hi],
                "kept": bool(keep),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--speed-only", action="store_true")
    ap.add_argument("--bpb-only", action="store_true")
    ap.add_argument("--ckpt-dir", default="/home/sehaxe/dormouse/checkpoints")
    ap.add_argument("--ckpt-name", default="longctx1")
    a = ap.parse_args()

    LOGDIR.mkdir(parents=True, exist_ok=True)
    rows_path = LOGDIR / "rows.jsonl"

    if a.dry_run:
        print(f"# {PRESET} b={BATCHES} s={SEQ} max_iter={MAX_ITER}")
        for arm, (it, off) in ARMS.items():
            print(f"\n# {arm}: iter_checkpoint={it} activation_offload={off}")
            for batch in BATCHES:
                print(
                    f"  speed: batch={batch} seed={SPEED_SEED} "
                    f"{SPEED_STEPS} steps (+{SPEED_WARM} warm)"
                )
        if not a.speed_only:
            for arm in ARMS:
                for seed in BPB_SEEDS:
                    print(" ".join(bpb_cmd(arm, seed)))
        print(f"\n# rows -> {rows_path}")
        return 0

    why = preflight("auto") or data_missing(CORPUS, EVAL)
    if why:
        return skip(why)

    rows: list[dict] = []
    if not a.bpb_only:
        for batch in BATCHES:
            for arm in ARMS:
                row = speed_row(arm, batch, ckpt_dir=a.ckpt_dir, ckpt_name=a.ckpt_name)
                rows.append(row)
                print(json.dumps(row), flush=True)
    if not a.speed_only:
        for arm in ARMS:
            for seed in BPB_SEEDS:
                row = run_bpb(arm, seed, dry=False)
                rows.append(row)
                print(json.dumps(row), flush=True)

    rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("\n| arm | peak@8 GiB | ms/step@8 | BPB mean | control band | kept |")
    print("|---|---|---|---|---|---|")
    results = {r["arm"]: r for r in verdict(rows)}
    for arm in ARMS:
        v = results.get(arm, {})
        s8 = next(
            (r for r in rows if r["kind"] == "speed" and r["batch"] == 8 and r["arm"] == arm),
            {},
        )
        print(
            f"| {arm} | {s8.get('peak_gib', '?')} | {s8.get('ms_step', '?')} | "
            f"{v.get('bpb_mean', 'control' if arm == 'off' else '?')} | "
            f"{v.get('control_band', '?')} | {v.get('kept', 'control')} |"
        )
    print(
        f"\nrule: kept iff peak@8 falls >= {PEAK_CUT:.0%} AND mean BPB is inside "
        f"the control's 3-seed [min,max] AND nan_skipped == 0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
