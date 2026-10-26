"""The ENERGY A/B: control vs `energy_weight` 0.05 vs 0.1, 3 seeds.

    uv run python scripts/ab_energy.py ab              # 3 arms x 3 seeds, medium
    uv run python scripts/ab_energy.py ab --seeds 1    # one seed, dry check

**THIS DRIVER HAS NOT BEEN RUN.** The card is busy (AGENTS §1.5: one heavy
thing at a time); the preflight refuses and prints no table, so every number
here is unmeasured. Nothing in this repo claims the energy head improves
quality until this driver and `scripts/rerank.py` have run.

**Arms** — `ModelConfig.energy_weight` (`config.py`; the term in
`model/energy.py`, EBT arXiv:2507.02092). One factor per run, everything else
the battery's COMMON:

* `off`     — `energy_weight=0.0`: no head is built, the control and every
  number on record.
* `energy`  — `energy_weight=0.05`.
* `energy2` — `energy_weight=0.1`.

The weight is SWEPT because the effect is a regulariser's: it adds a
contrastive term to the same hidden state the `lm_head` reads, and the useful
scale is not derivable from the paper (whose measured gain is inference
compute, not a 2 000-step pretraining delta). 0.05 / 0.1 is the same grid
SIGReg's research note names; a single weight would be a tuned constant wearing
an arm's name.

**Protocol** (`docs/protocols/AB-PROTOCOL.md`): 3 seeds per arm, 2 000 steps,
`medium`, b8 x s512, the battery's `BASE` + `COMMON` copied verbatim from
`/tmp/opencode/battery.sh` (the tuple `ab_depth8.py` / `ab_sf_muon.py` carry) —
`tf32`, `ns_kind=triton`, `opt_graph`, `moe_kernel=triton`, `moe_capacity=0`,
`engram_ram`, `bf16`, `mix_plan=configs/mix_big.toml`, eval every 500 over
`eval_batches=20`. Every run scores ONE held-out window of
`20 x 8 x 512 = 81 920 B`; a different `scored` is a VOID run. Each arm-seed is
a SEPARATE PROCESS, launched as the CLI would be, and its own
`<arm>_s<seed>.config.toml` sits beside the JSONL (AGENTS §1.4: a number needs
its config).

**The void check is the counter** (AGENTS §1.1): `counters["energy"]` is bumped
exactly where the term is added (`model.py`), so an `energy`/`energy2` run must
report `steps` additions and `off` must report ZERO. A `counters` window is the
log cadence's, so the run total is the sum over the JSONL's log rows. A non-zero
count on `off` means the arms are not the arms the table claims; zero on an
energy arm means the term never reached the loss and the run measured nothing.

**The divergence flag** (the 2026-10-21 SF-Muon lesson): a seed is DIVERGED
when its BPB is more than 3σ from its arm's median, with σ the MAD-based
estimate `1.4826·median(|b − median|)` and a `2e-3` BPB floor. The sample σ of
three points is inflated by the outlier it should catch, so the robust
estimator is the one used. Diverged seeds are printed with the flag and
excluded from the mean, which prints `n=<kept>/<total>` — a two-of-three mean
can never look like a three-seed mean.

**Verdict** (AGENTS §1.2, pre-registered before the first run): for each energy
arm, `Δ = control_mean − arm_mean` (positive = the arm's BPB is lower). WIN iff
`Δ > control_spread`, where `control_spread = max(control) − min(control)` over
the control's kept seeds; anything else is a tie, and a tie deletes the
mechanism. An arm with fewer than two kept seeds prints "cannot decide" — it is
neither a win nor a loss.

**Pre-registered prediction**: a tie at this scale. EBT's +29% is a gain from
INFERENCE compute on reasoning tasks, not a claim that a 2 000-step pretraining
run at `medium` improves next-byte BPB; the contrastive head is a
regulariser-shaped aux term, and the `aux_fb` future-byte head tied at 3 seeds
and was deleted (`39d0485`). If this ties, the inference reranker is still the
arm to test — the training term and the reranker are separate uses of the same
head, and this driver prices only the training term.

**What the driver does NOT do:** apply the verdict, run the reranker, or quote a
quality number. Wall clock is unmeasured for the same reason (9 x 2 000
`medium` steps; the card must be idle).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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

#: `(arm, energy_weight)`. `off` first: the table and the verdict read against it.
ARMS = (
    ("off", 0.0),
    ("energy", 0.05),
    ("energy2", 0.1),
)

#: `/tmp/opencode/battery.sh`'s `BASE`: the device and the two data files.
DATA = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"
EVAL_DATA = "/home/sehaxe/dormouse/data_local/eval_mix/eval_mix.bin"
BASE = ("--device", "cuda", "--data", DATA, "--eval", EVAL_DATA)
#: The battery's `COMMON`, verbatim. Do not tune one arm: the weight is the
#: only difference the table is allowed to explain.
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

#: The one eval window every run must score. A different number is a VOID run.
WINDOW = EVAL_BATCHES * BATCH * SEQ
#: The robust-divergence floor: the low end of the paired-eval resolution.
SIGMA_FLOOR = 2e-3


def _lit(value: object) -> str:
    """A config value as the `--set k=v` literal the CLI parses back to it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


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


def refuse(why: str) -> int:
    """A LOUD refusal with no table: a busy card measures nothing."""
    print(
        f"\n=== REFUSED: {why}\n"
        f"=== ONE GPU PROCESS AT A TIME (AGENTS §1.5): nothing was measured and no\n"
        f"=== table is printed. Re-run when the card is free.\n",
        flush=True,
    )
    return 1


def _common_dicts() -> tuple[dict, dict]:
    """COMMON split into model keys and train keys, typed exactly as the child sees."""
    from dormouse.config import ModelConfig

    model_keys = {f.name for f in dataclasses.fields(ModelConfig)}
    model: dict = {}
    train: dict = {}
    items = list(COMMON)
    for i in range(0, len(items), 2):
        assert items[i] == "--set", items[i]
        key, value = items[i + 1].split("=", 1)
        (model if key in model_keys else train)[key] = _typed(value)
    return model, train


def _snapshot(arm: str, weight: float, seed: int, out_dir: Path, steps: int) -> None:
    """Write the run's own `<name>.config.toml` beside its JSONL.

    `ckpt_every=10**6`, so the trainer writes no checkpoint and no snapshot of
    its own; without this the number would have no config record (AGENTS §1.4).
    It resolves the SAME COMMON + arm overrides the child gets.
    """
    from dormouse.config import load_run

    model_common, train_common = _common_dicts()
    run = load_run(
        PRESET,
        model_overrides={**model_common, "energy_weight": weight},
        train_overrides={
            **train_common,
            "steps": steps,
            "seed": seed,
            "ckpt_name": f"energy_{arm}_s{seed}",
            "out_dir": str(out_dir / "ckpt"),
            "json_log": str(out_dir / f"{arm}_s{seed}.jsonl"),
            "data": DATA,
            "eval": True,
            "eval_data": EVAL_DATA,
        },
    )
    (out_dir / f"{arm}_s{seed}.config.toml").write_text(run.snapshot_toml())


def run_one(arm: str, weight: float, seed: int, out_dir: Path, device: str, steps: int) -> int:
    """One arm, one seed, one CLI process — the battery's own launch shape."""
    jsonl = out_dir / f"{arm}_s{seed}.jsonl"
    log = out_dir / f"{arm}_s{seed}.log"
    jsonl.unlink(missing_ok=True)
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
        f"energy_weight={_lit(weight)}",
        "--set",
        f"ckpt_name=energy_{arm}_s{seed}",
        "--set",
        f"out_dir={out_dir / 'ckpt'}",
        "--json-log",
        str(jsonl),
    ]
    if device != "cuda":
        cmd[cmd.index("--device") + 1] = device
    print(f"\n=== {arm} (energy_weight={weight}) seed {seed} -> {jsonl}", flush=True)
    with open(log, "w", encoding="utf-8") as sink:
        done = subprocess.run(cmd, cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT)
    if done.returncode != 0:
        print(f"the {arm} run (seed {seed}) exited {done.returncode}: no table. Log: {log}")
    return done.returncode


def read_run(path: Path, steps: int) -> dict:
    """One run's JSONL -> the numbers the table and the void checks need."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    evals = [r for r in rows if r.get("event") == "eval"]
    logs = [r for r in rows if r.get("event") == "log"]
    assert evals, f"{path}: no eval row — nothing to compare"
    return {
        "bpb": float(evals[-1]["bpb"]),
        "scored": int(evals[-1]["scored"]),
        # `nan_skipped` is CUMULATIVE in the JSONL: the last row is the total.
        "nan": max((int(r.get("nan_skipped", 0)) for r in logs), default=0),
        # Counters reset per log window; the sum is the run's number of
        # forwards that added the energy term.
        "energy": sum(int(r.get("counters", {}).get("energy", 0)) for r in logs),
        "steps": steps,
    }


def diverged(bpbs: list[float]) -> list[bool]:
    """One flag per seed: `|b − median| > 3σ`, σ the MAD estimate."""
    assert len(bpbs) >= 2, "an arm needs at least two seeds for a σ"
    med = statistics.median(bpbs)
    mad = statistics.median([abs(b - med) for b in bpbs])
    sigma = max(1.4826 * mad, SIGMA_FLOOR)
    return [abs(b - med) > 3.0 * sigma for b in bpbs]


def kept(bpbs: list[float]) -> list[float]:
    """The non-diverged seeds of one arm."""
    return [b for b, bad in zip(bpbs, diverged(bpbs), strict=True) if not bad]


def verdict(control: list[float], arm: list[float]) -> dict[str, object]:
    """AGENTS §1.2 on the trimmed sets: keep iff the mean beats the spread."""
    assert control and arm, "the verdict needs a kept seed on both sides"
    c_mean = statistics.mean(control)
    a_mean = statistics.mean(arm)
    spread = max(control) - min(control)
    delta = c_mean - a_mean
    if len(control) < 2 or len(arm) < 2:
        return {
            "kept": None,
            "delta": delta,
            "spread": spread,
            "why": "cannot decide: fewer than two kept seeds on one side",
        }
    return {
        "kept": delta > spread,
        "delta": delta,
        "spread": spread,
        "why": "" if delta > spread else "tie: delta does not clear the control spread",
    }


def ab(args: argparse.Namespace) -> int:
    seeds = tuple(int(s) for s in args.seeds.split(","))
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"ENERGY A/B: {len(ARMS)} arms x {len(seeds)} seeds x {args.steps} steps, "
        f"{PRESET} b{BATCH} s{SEQ}, window {WINDOW} B. Output: {out_dir}\n"
        f"pre-registered (AGENTS §1.2): WIN iff the trimmed held-out BPB beats the "
        f"control's by MORE than the control's own kept-seed spread; a tie deletes "
        f"the training term. The reranker is a separate use and is not priced here."
    )
    for arm, weight in ARMS:
        for seed in seeds:
            _snapshot(arm, weight, seed, out_dir, args.steps)
            rc = run_one(arm, weight, seed, out_dir, args.device, args.steps)
            if rc != 0:
                return rc

    table = {arm: [] for arm, _w in ARMS}
    for arm, _weight in ARMS:
        for seed in seeds:
            run = read_run(out_dir / f"{arm}_s{seed}.jsonl", args.steps)
            table[arm].append({"seed": seed, **run})

    print(
        f"\n{'arm':>8s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'nan':>4s} {'energy':>6s}  flag"
    )
    void: dict[str, list[str]] = {arm: [] for arm, _w in ARMS}
    for arm, runs in table.items():
        for r in runs:
            if r["scored"] != WINDOW:
                void[arm].append(f"seed {r['seed']}: window {r['scored']} != {WINDOW} B")
            if r["nan"]:
                void[arm].append(f"seed {r['seed']}: {r['nan']} NaN-skipped steps")
            want = 0 if arm == "off" else args.steps
            if r["energy"] != want:
                void[arm].append(
                    f"seed {r['seed']}: energy counter {r['energy']} != {want} "
                    f"(weight {'0 — the term ran' if arm == 'off' else '>0 — the term never ran'})"
                )
    for arm, runs in table.items():
        flags = diverged([r["bpb"] for r in runs]) if len(runs) >= 2 else [False] * len(runs)
        for r, bad in zip(runs, flags, strict=True):
            mark = "DIVERGED (>3σ from arm median)" if bad else ""
            print(
                f"{arm:>8s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['nan']:>4d} {r['energy']:>6d}  {mark}"
            )

    summary: dict[str, dict[str, object]] = {}
    print("\n=== trimmed per-arm summary (diverged seeds excluded from the mean)")
    for arm, runs in table.items():
        bpbs = [r["bpb"] for r in runs]
        keep = kept(bpbs) if len(bpbs) >= 2 else bpbs
        summary[arm] = {"kept": keep, "n": f"{len(keep)}/{len(bpbs)}"}
        print(
            f"{arm:>8s} n={summary[arm]['n']} mean={statistics.mean(keep):.4f} "
            f"kept={[round(b, 4) for b in keep]}"
        )

    for arm, _weight in ARMS:
        if void[arm]:
            print(f"{arm:>8s} VOID: " + "; ".join(void[arm]) + " — no number.")
    if void["off"]:
        print("\nno control number: no verdict.")
        return 0
    for arm, weight in ARMS[1:]:
        if void[arm]:
            print(f"{arm:>8s} verdict: none — VOID")
            continue
        v = verdict(summary["off"]["kept"], summary[arm]["kept"])
        if v["kept"] is None:
            print(f"{arm:>8s} verdict: {v['why']}")
            continue
        print(
            f"{arm:>8s} (w={weight}) delta vs control {v['delta']:+.4f} BPB "
            f"(control spread {v['spread']:.4f}) -> "
            f"{'WIN (training term kept)' if v['kept'] else 'TIE (deletes the training term)'}"
        )
    print(
        f"\nvoid checks: every eval scored {WINDOW} B, 0 NaN-skipped steps, the energy "
        f"counter == {args.steps} on every armed run and 0 on every control run. "
        f"Divergence: |b − median| > 3 x max(1.4826 x MAD, {SIGMA_FLOOR}). "
        f"Rule: WIN iff the trimmed delta beats the control's own spread; a tie "
        f"deletes the training term. Pre-registered prediction: a tie — EBT's "
        f"+29% is inference compute on reasoning tasks, not a 2k-step BPB delta; "
        f"the reranker (`scripts/rerank.py`) is the other use of this head and is "
        f"not measured by this driver."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ab", help="the A/B driver (3 arms x seeds)")
    a.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a.add_argument("--steps", type=int, default=STEPS)
    a.add_argument("--data", default=DATA)
    a.add_argument("--eval-data", default=EVAL_DATA)
    a.add_argument("--device", default="cuda")
    a.add_argument("--out-dir", default="~/logs/ab_energy")
    args = ap.parse_args(argv)
    why = preflight(args.device) or data_missing(args.data, args.eval_data)
    if why is not None:
        return refuse(why)
    return ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
