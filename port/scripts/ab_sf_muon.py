"""The SF-Muon re-run: 3 arms x 3 seeds, with the arm made LEGAL first.

    uv run python scripts/ab_sf_muon.py ab              # the whole grid (not run yet)
    uv run python scripts/ab_sf_muon.py ab --seeds 1    # one seed, dry check

**This driver has NOT been run.** The card is busy and AGENTS §1.5 allows one
GPU process at a time; the preflight refuses loudly and prints no table. Every
number is therefore unmeasured until it runs.

**Why there is a re-run at all.** The 2026-10-18 wave in
`~/logs/ab_muon_variants/` reported the SF-Muon arm at **+0.045 BPB** against
the control on the two seeds that finished, and one seed diverged to
**BPB 4.0573** — the headline was a trimmed mean with the divergence dropped
and no marker on the table. The arm also sat outside the paper's theorem in
ways the research note had already named (`docs/research/2026-10-16-sf-muon.md`
§3: S1 exact-polar gap, S2 decay at `Z`, S3 retraction). Tasks A and the note
`docs/reviews/2026-10-21-sf-muon-prerequisites.md` made the SF-specific
preconditions explicit; this driver tests the arm those preconditions describe.

**Arms** (the control is first; every other flag is COMMON):

* `muon` — `muon_sf=false`: the live recipe, WSD on both halves, the standing
  control. `muon_sf_wd` is INERT here (the Muon group is built with wd=0.0).
* `sf` — `muon_sf=true`, `muon_sf_wd=0.01` (applied at `Z`, the value the
  theorem needs: λ > 0), `muon_sf_retract=false`. The Muon-group LR is
  constant after warmup, the AdamW half keeps WSD (the hybrid the arm doc
  names), and `engram_ram` host tables keep their own WSD step.
* `sf_lr05` — `sf` at HALF the peak LR. The paper's claim is that SF removes
  the schedule; an SF arm that only matches at one LR is a tuned constant-LR
  arm, not the claim. This arm is the falsifier, and the driver prints both
  deltas side by side.

**Protocol** (`docs/protocols/AB-PROTOCOL.md`): 3 seeds per arm, 2 000 steps,
`medium`, the battery's COMMON (`/tmp/opencode/battery.sh` verbatim, the tuple
`ab_depth8.py`/`ab_rank2.py` carry), b8 x s512, pure CE (the aux weights are
the preset's and are identical across arms), and **one eval window of 81 920 B
on every arm and seed** (`eval_batches 20 x batch 8 x seq_len 512`, the same
window as the 2026-10-18 wave). Each arm-seed is a SEPARATE PROCESS, because 3
seeds is 3 initialisations. Output: one JSONL per run under
`~/logs/ab_sf_muon/`, plus the run's own `<name>.config.toml` beside it (a
number needs its config, AGENTS §1.4).

**The verdict statistic, written BEFORE the first run** (AGENTS §1.2):

* a seed is a **DIVERGENCE** when its BPB is more than 3σ from the arm's own
  median, with σ the MAD-based estimate `1.4826·median(|b − median|)` (the
  sample σ of three points is inflated by the very outlier it should catch —
  on the 2026-10-18 numbers it would have hidden 4.0573) and a `2e-3` BPB floor;
* the mean is over the kept seeds only, and the table prints `n=<kept>/<total>`
  so a two-of-three mean can never look like a three-seed mean;
* `sf` is KEPT iff its trimmed mean beats the control's trimmed mean by more
  than the CONTROL's own seed spread; a tie deletes the mechanism;
* `sf_lr05` is read against `sf`: if the half-LR arm reverses the sign of the
  delta, the arm's result is LR-specific and not the schedule claim;
* an arm left with fewer than two kept seeds prints "cannot decide" — it is not
  a win and not a loss.

**What the driver does NOT do:** apply the verdict, or measure the `X`↔`Y` gap
(the 2026-10-18 JSONLs carry no `sf_gap` row either — the endpoint is what
this wave reports; the curve is printed from the eval rows).
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

#: The port's data on this box (AGENTS §2.6) — the LOCAL copies, so the driver
#: does not depend on the /mnt drive being mounted.
DATA = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"
EVAL_DATA = "/home/sehaxe/dormouse/data_local/eval_mix/eval_mix.bin"

SEEDS = (1, 2, 3)
STEPS = 2000
BATCH = 8
SEQ = 512
#: `TrainConfig`'s own default (3e-4); the battery does not override it
#: (`ab_mhc.py:83`). `sf_lr05` takes half of this.
LR = 3e-4
EVAL_BATCHES = 20
EVAL_EVERY = 500
LOG_EVERY = 250
PRESET = "medium"

#: `/tmp/opencode/battery.sh`'s COMMON, as `--set` keyword pairs: one recipe,
#: one eval window (20 x 8 x 512 = 81 920 B, the same one the 2026-10-18 wave
#: scored); this driver's seq_len is 512, the battery's. Do not tune one arm.
COMMON: dict[str, object] = {
    "steps": STEPS,
    "batch": BATCH,
    "seq_len": SEQ,
    "eval": True,
    "eval_every": EVAL_EVERY,
    "eval_batches": EVAL_BATCHES,
    "log_every": LOG_EVERY,
    "ckpt_every": 10**6,
    "tf32": True,
    "ns_kind": "triton",
    "opt_graph": True,
    "moe_kernel": "triton",
    "moe_capacity": 0,
    "engram_ram": True,
    "bf16": True,
    "mix_plan": "configs/mix_big.toml",
}

#: `(arm, overrides)`. Every SF arm spells the two legality keys out in its own
#: command line, so the config record — not the driver's docstring — is the
#: proof of what ran.
ARMS: dict[str, dict[str, object]] = {
    "muon": {"muon_sf": False, "muon_sf_wd": 0.01, "muon_sf_retract": False},
    "sf": {"muon_sf": True, "muon_sf_wd": 0.01, "muon_sf_retract": False},
    "sf_lr05": {
        "muon_sf": True,
        "muon_sf_wd": 0.01,
        "muon_sf_retract": False,
        "lr": LR / 2,
    },
}

#: The one window every run must score. A different number is a VOID arm.
WINDOW = EVAL_BATCHES * BATCH * SEQ

#: σ below this is measurement noise, not a seed: `AB-PROTOCOL.md` estimates
#: the paired-eval resolution at 0.002-0.005 BPB, so the floor is its LOWER end
#: (the 2026-10-18 control's own 0.0043 spread must not mark a seed).
SIGMA_FLOOR = 2e-3


def _lit(value: object) -> str:
    """A config value as the `--set k=v` literal the CLI parses back to it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def refuse(why: str) -> int:
    """A LOUD refusal with no table: a busy card measures nothing."""
    print(
        f"\n=== REFUSED: {why}\n"
        f"=== ONE GPU PROCESS AT A TIME (AGENTS §1.5): nothing was measured and no\n"
        f"=== table is printed. Re-run when the card is free.\n",
        flush=True,
    )
    return 1


# --- the verdict statistic (pure; gated by tests/test_sf_muon_legal.py) -----


def diverged(bpbs: list[float]) -> list[bool]:
    """One flag per seed: `|b − median| > 3σ`, σ the MAD estimate.

    The sample σ of three points is inflated by the outlier it should catch
    (on the 2026-10-18 arm it reads 1.06 BPB, so 3σ = 3.17 and the 4.0573 seed
    passes). `1.4826·MAD` is the robust σ estimator instead, with a `2e-3`
    floor — the lower end of the paired-eval resolution, below which a
    difference is noise.
    """
    assert len(bpbs) >= 2, "an arm needs at least two seeds for a σ"
    med = statistics.median(bpbs)
    mad = statistics.median([abs(b - med) for b in bpbs])
    sigma = max(1.4826 * mad, SIGMA_FLOOR)
    return [abs(b - med) > 3.0 * sigma for b in bpbs]


def kept(bpbs: list[float]) -> list[float]:
    """The non-diverged seeds of one arm."""
    return [b for b, bad in zip(bpbs, diverged(bpbs), strict=True) if not bad]


def verdict(control: list[float], arm: list[float]) -> dict[str, object]:
    """AGENTS §1.2 on the trimmed sets: keep iff the mean beats the spread.

    `control` and `arm` are already trimmed. The control's spread is the
    max-min of ITS kept seeds — the bar the arm must clear. A tie (delta <=
    spread) deletes the mechanism.
    """
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
        "why": "win beats the control's own seed spread" if delta > spread else "tie deletes",
    }


# --- the run ---------------------------------------------------------------


def _flags(over: dict[str, object]) -> list[str]:
    out: list[str] = []
    for key, value in over.items():
        out += ["--set", f"{key}={_lit(value)}"]
    return out


def _snapshot(arm: str, seed: int, out_dir: Path) -> None:
    """Write the run's own `<name>.config.toml` beside its JSONL.

    `ckpt_every` is 10**6, so the trainer writes no checkpoint and therefore no
    snapshot of its own; without this the number would have no config record
    (AGENTS §1.4). Resolved from the SAME COMMON + overrides the child gets.
    """
    from dormouse.config import ModelConfig, load_run

    model_keys = {f.name for f in dataclasses.fields(ModelConfig)}
    over = {**COMMON, **ARMS[arm], "seed": seed, "ckpt_name": f"ab_sf_muon_{arm}_s{seed}"}
    model_over = {k: v for k, v in over.items() if k in model_keys}
    train_over = {k: v for k, v in over.items() if k not in model_keys}
    run = load_run(
        PRESET,
        model_overrides=model_over,
        train_overrides={
            **train_over,
            "out_dir": str(out_dir / "ckpt"),
            "json_log": str(out_dir / f"{arm}_s{seed}.jsonl"),
        },
    )
    (out_dir / f"{arm}_s{seed}.config.toml").write_text(run.snapshot_toml())


def run_one(arm: str, seed: int, out_dir: Path, device: str, steps: int) -> int:
    """One arm, one seed, one CLI process — the battery's own launch shape."""
    jsonl = out_dir / f"{arm}_s{seed}.jsonl"
    text = out_dir / f"{arm}_s{seed}.log"
    jsonl.unlink(missing_ok=True)
    over = {
        **COMMON,
        **ARMS[arm],
        "steps": steps,
        "seed": seed,
        "ckpt_name": f"ab_sf_muon_{arm}_s{seed}",
        "out_dir": str(out_dir / "ckpt"),
    }
    cmd = [
        "uv",
        "run",
        "--extra",
        "gpu",
        "dormouse",
        "train",
        PRESET,
        "--device",
        device,
        "--data",
        DATA,
        "--eval",
        EVAL_DATA,
        *_flags(over),
        "--json-log",
        str(jsonl),
    ]
    print(f"\n=== {arm} seed {seed} -> {jsonl}", flush=True)
    with open(text, "w", encoding="utf-8") as sink:
        done = subprocess.run(cmd, cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT)
    if done.returncode != 0:
        print(f"the {arm} run (seed {seed}) exited {done.returncode}: no table. Log: {text}")
    return done.returncode


def read_run(path: Path) -> dict[str, object]:
    """One run's JSONL -> the numbers the table needs."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    evals = [r for r in rows if r.get("event") == "eval"]
    logs = [r for r in rows if r.get("event") == "log"]
    assert evals, f"{path}: no eval row — nothing to compare"
    counters = logs[-1].get("counters", {}) if logs else {}
    fallback = any(row.get("counters", {}).get("opt_graph_fallback") for row in logs)
    return {
        "bpb": float(evals[-1]["bpb"]),
        "scored": int(evals[-1]["scored"]),
        "curve": {int(r["step"]): float(r["bpb"]) for r in evals},
        "nan": sum(int(r.get("nan_skipped", 0)) for r in logs),
        "counters": counters,
        "opt_graph_fallback": fallback,
    }


def ab(args: argparse.Namespace) -> int:
    seeds = tuple(int(s) for s in args.seeds.split(","))
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"A/B: {len(ARMS)} arms x {len(seeds)} seeds x {args.steps} steps, {PRESET} "
        f"b{BATCH} s{SEQ}, window {WINDOW} B. Output: {out_dir}"
    )
    for arm in ARMS:
        for seed in seeds:
            _snapshot(arm, seed, out_dir)
            done = run_one(arm, seed, out_dir, args.device, args.steps)
            if done != 0:
                return done

    table: dict[str, list[dict[str, object]]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for seed in seeds:
            row = read_run(out_dir / f"{arm}_s{seed}.jsonl")
            row["seed"] = seed
            table[arm].append(row)

    print(f"\n{'arm':>8s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'nan':>4s}  note")
    for arm, runs in table.items():
        flags = diverged([float(r["bpb"]) for r in runs])
        for row, bad in zip(runs, flags, strict=True):
            print(
                f"{arm:>8s} {row['seed']:>4d} {row['bpb']:8.4f} {row['scored']:>9d} "
                f"{row['nan']:>4d}  {'DIVERGED (>3 sigma from this arm median)' if bad else ''}"
            )

    print("\n=== trimmed per-arm summary (diverged seeds excluded from the mean)")
    summary: dict[str, dict[str, object]] = {}
    for arm, runs in table.items():
        bpbs = [float(r["bpb"]) for r in runs]
        keep = kept(bpbs)
        summary[arm] = {
            "kept": keep,
            "n": f"{len(keep)}/{len(bpbs)}",
            "mean": statistics.mean(keep) if keep else float("nan"),
        }
        print(
            f"{arm:>8s} n={summary[arm]['n']} mean={summary[arm]['mean']:.4f} "
            f"kept={[round(b, 4) for b in keep]}"
        )

    # The curve, because a 2k endpoint alone is the mistake 2609.04577 names.
    steps = sorted({s for runs in table.values() for r in runs for s in r["curve"]})
    print("\n=== held-out BPB curve, mean over each arm's KEPT seeds")
    print(f"{'step':>6s} " + " ".join(f"{arm:>8s}" for arm in ARMS))
    for step in steps:
        cells = []
        for runs in table.values():
            keep = kept([float(r["bpb"]) for r in runs])
            vals = [
                float(r["curve"][step])
                for r in runs
                if float(r["bpb"]) in keep and step in r["curve"]
            ]
            cells.append(f"{statistics.mean(vals):8.4f}" if vals else f"{'-':>8s}")
        print(f"{step:>6d} " + " ".join(cells))

    # void checks: a number that fails one of these is not a number.
    void = []
    for arm, runs in table.items():
        for row in runs:
            if row["scored"] != WINDOW:
                void.append(f"{arm} seed {row['seed']}: window {row['scored']} != {WINDOW}")
            if row["nan"] != 0:
                void.append(f"{arm} seed {row['seed']}: {row['nan']} NaN-skipped steps")
            fallback = row["opt_graph_fallback"]
            if ARMS[arm].get("muon_sf") and not fallback:
                void.append(
                    f"{arm} seed {row['seed']}: no opt_graph_fallback under muon_sf — "
                    f"the graph arm's SF refusal did not run, so this is not the arm's path"
                )
    if void:
        print("\nVOID — not a table:")
        for line in void:
            print(f"  {line}")
        return 1
    print(
        f"\nvoid checks: every eval scored {WINDOW} B, 0 NaN-skipped steps, "
        f"{len(seeds)} distinct seeds per arm, and every SF arm shows the counted "
        f"opt_graph fallback."
    )

    print("\n=== AGENTS §1.2 verdict on the trimmed sets")
    control = summary["muon"]["kept"]
    for arm in ("sf", "sf_lr05"):
        v = verdict(control, summary[arm]["kept"])  # type: ignore[arg-type]
        word = "cannot decide" if v["kept"] is None else ("KEEP" if v["kept"] else "DELETE (tie)")
        print(
            f"{arm:>8s}: delta={v['delta']:+.4f} BPB vs muon, control spread "
            f"{v['spread']:.4f} -> {word} ({v['why']})"
        )
    print(
        "\nThe `sf_lr05` delta is the falsifier: if it reverses the sign of the `sf` "
        "delta, the result is LR-specific and is not the schedule-free claim. A "
        "2k-step tie is the designed outcome (the published gains are stated at "
        ">=1000 tokens/param), so a tie escalates to one 10x-step arm before deletion "
        "(`docs/research/2026-10-16-sf-muon.md` §4)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ab", help="the A/B driver (3 arms x seeds)")
    a.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a.add_argument("--steps", type=int, default=STEPS)
    a.add_argument("--device", default="cuda")
    a.add_argument("--out-dir", default="~/logs/ab_sf_muon")
    args = ap.parse_args(argv)
    why = preflight(args.device) or data_missing(
        DATA, EVAL_DATA, str(ROOT / str(COMMON["mix_plan"]))
    )
    if why is not None:
        return refuse(why)
    return ab(args)


if __name__ == "__main__":
    sys.exit(main())
