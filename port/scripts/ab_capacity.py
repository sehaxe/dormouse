#!/usr/bin/env python3
"""The capacity sweep: what should carry the parameters at 22.2 GB of data?

THE QUESTION, from our own curves. 199M (E=512, top-2, rank-64, bank 134M)
reaches BPB 1.8172 at 0.41 GB seen; 105M reaches 1.806 at the same point. The
bigger model is not better at our data budget. Top-2 of 512 touches 0.4% of the
bank per token: the capacity exists, and this experiment asks whether it is
capacity or dead weight. All five arms are resolved through `config.load_run`
(the real preset/refusal path, not hand-built `ModelConfig`s) and compared at
the same ACTIVE compute per byte.

ARMS (all `configs/m400b.toml`, differing only in the keys named below):

  A  control: E=512, rank 64, f=4096 — today's 199M model.
  B  E=128, rank 64, f re-solved so the active MAC equals A's. The bank drops
     134M -> 34M parameters at the same assignment width; the freed budget is
     moved into `d_ffn` only as far as the compute constraint allows (the rest
     cannot be spent without leaving the equal-compute band — the table prints
     the ceiling, and that is the finding, not a missing knob).
  C  E=64, rank 64, same rule.
  D  dense: `use_tsct=false`, one full-rank expert (`rank=d_model`), the dense
     blend (`n_experts=1`, `moe_topk=0`) and `share_u_mode='off'`; f is solved
     for equal active MAC, so the dense model accepts fewer parameters per
     active FLOP. This is the literal `use_tsct=false` arm, at the largest size
     the compute equality allows.
  E  E=512, rank 32, f re-solved: the rank-32 rule already on record
     (`rank 32` measured +0.027 BPB at fixed width) with the freed rank spent
     on expert width.

THE COMPUTE GATE. `scripts/capacity_math.py` prices the active ledger
(`width.macs_per_token` for the shared body + the routed/shared/router terms).
Before a single step, every arm must be within 5% of A; a miss prints the table
and REFUSES the comparison. Arms B/C/D/E are solved to A's ledger exactly, so
their resolved `d_ffn` is part of the answer, not a hand-typed number.

THE RUN. `dormouse.cli train`, 600 steps, b8 x s1024, eval at step 600 with 8
batches (65 536 scored bytes), `ckpt_every` huge (no checkpoint), a fresh
`ckpt_name`, `out_dir` under `/tmp`, `data` pinned to the v7 plan's first source,
`mix_plan=configs/mix_v7.toml`, and a DIFFERENT seed triple per arm (the seeds
are distinct across the whole sweep, so a synthetic tie cannot come from three
identical inits). Routing health is read from the run's own device counters
(`moe_route_live / moe_route_n`), which is the same `live_experts` definition
`route_hist.summary` reports — measured on the router that ran, not re-probed.

THE DECISION RULE, PRE-REGISTERED HERE BEFORE THE FIRST RUN. An arm wins only if
all three hold:

  1. at least 3 seeds for the control and for the arm;
  2. `mean < control_mean - 2 x control_spread` (the spread is max-min, as in
     `scripts/collect_ab.py`);
  3. `mean <= control_mean - 0.01` (the collect_ab floor: our 600-step eval
     spread is ~0.006-0.02 BPB, so a difference below 0.01 is not a number).

Otherwise it is a LOSS when it is worse than `control_mean + max(2*spread,
0.01)`, and a TIE in the band between. **A tie goes to the SMALLER parameter
count**: the cheaper model wins a tie. A control with fewer than 3 seeds, or a
zero spread, refuses to promote anything (`collect_ab`'s measured failure: a
one-seed control makes `mean - 2*0` promote an exact tie).

USAGE
    python scripts/ab_capacity.py --dry-run                  # configs + compute table, no card
    python scripts/ab_capacity.py --dry-run --arms A,B       # the two arms worth the gap
    CUDA_VISIBLE_DEVICES=0 python scripts/ab_capacity.py --arms A,B   # ~22 min at 22 kB/s
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PORT = Path(__file__).resolve().parents[1]
REPO = PORT.parent
sys.path.append(str(PORT / "scripts"))  # `scripts/` is not a package
if str(PORT / "src") not in sys.path:
    sys.path.insert(0, str(PORT / "src"))

import capacity_math as cm  # noqa: E402

from dormouse.config import ModelConfig, RunConfig, load_preset, load_run  # noqa: E402
from dormouse.instrument import LoudError  # noqa: E402

#: The live m400b recipe, as a path: resolving by name would use `configs/m400b.toml`
#: only if the cwd is the port, and a driver must not depend on the caller's cwd.
BASE_PRESET = PORT / "configs" / "m400b.toml"
#: The v7 plan's first source. With `mix_plan` set the trainer reads the mixture,
#: not this file, but the path is pinned so a `--set mix_plan=` failure cannot
#: silently fall back to a different corpus.
DATA = REPO / "data_local" / "best" / "fwedu.bin"
EVAL = REPO / "data_local" / "eval_mix" / "eval_mix.bin"
MIX_PLAN = PORT / "configs" / "mix_v7.toml"

STEPS = 600
BATCH = 8
SEQ = 1024
EVAL_BATCHES = 8
#: The eval window is `eval_batches x batch x seq_len` (byte count, AGENTS §2.6).
WINDOW = EVAL_BATCHES * BATCH * SEQ
COMPUTE_TOL = 0.05
MIN_SEEDS = 3
MIN_EFFECT = 0.01
#: Huge on purpose: 600 % 1e9 != 0, so the run writes no checkpoint.
CKPT_EVERY = 1_000_000_000
DEFAULT_OUT_ROOT = Path("/tmp/ab_capacity")

#: The one model override every arm shares: the routing-health counters. They
#: are a counter set, not a parameter, so the compute/param table is unchanged.
COMMON_MODEL: dict[str, object] = {"moe_routing_health": True}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    overrides: dict[str, object]
    seeds: tuple[int, ...]
    note: str
    widen: bool = True
    #: A sentinel for D: `rank=None` is resolved to the base preset's `d_model`
    #: (a full-rank expert), so the arm stays dense if the preset's width moves.
    rank_is_d_model: bool = False


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec(
        "A", {}, (1, 2, 3), "control: E=512 top-2 rank-64 f=4096 (199M, bank 134M)", widen=False
    ),
    ArmSpec("B", {"n_experts": 128}, (4, 5, 6), "E=128, rank 64, f solved for equal active MAC"),
    ArmSpec("C", {"n_experts": 64}, (7, 8, 9), "E=64, rank 64, f solved for equal active MAC"),
    ArmSpec(
        "D",
        {
            "use_tsct": False,
            "share_u": False,
            "share_u_mode": "off",
            "n_experts": 1,
            "moe_topk": 0,
        },
        (10, 11, 12),
        "dense: use_tsct=false, one full-rank expert, f solved for equal active MAC",
        rank_is_d_model=True,
    ),
    ArmSpec("E", {"rank": 32}, (13, 14, 15), "E=512, rank 32, f solved for equal active MAC"),
)


@dataclass
class ResolvedArm:
    spec: ArmSpec
    run: RunConfig
    row: dict
    overrides: dict[str, object]


def _resolved(overrides: dict[str, object]) -> RunConfig:
    return load_run(str(BASE_PRESET), model_overrides=overrides)


def _macs_at(overrides: dict[str, object], f: int) -> int:
    return cm.active_macs_per_token(_resolved({**overrides, "d_ffn": f}).model)


def solve_d_ffn(overrides: dict[str, object], target_macs: int) -> int:
    """The `d_ffn` that spends the freed parameters UP TO the compute target.

    The active bill is linear in `f` (`iters*d + topk*rank + shared_rank` per
    unit), so two evaluations solve it exactly. `max(1, ...)` is the schema's
    floor; the caller still runs `assert_equal_compute`, so a non-linear future
    ledger fails loudly instead of being trusted.
    """
    a1 = _macs_at(overrides, 1)
    a2 = _macs_at(overrides, 2)
    coeff = a2 - a1
    if coeff <= 0:
        raise LoudError(
            f"active MAC does not grow with d_ffn (dMAC={coeff}): the ledger this "
            f"driver solves against is not the one the model runs. Escape: fix "
            f"capacity_math.active_macs_per_token before comparing arms."
        )
    f = round(1 + (target_macs - a1) / coeff)
    return max(1, f)


def resolve_arms() -> list[ResolvedArm]:
    """Every arm through `load_run`; no `ModelConfig(...)` is constructed here."""
    base = load_preset(str(BASE_PRESET))
    control = _resolved(COMMON_MODEL)
    target = cm.active_macs_per_token(control.model)
    out: list[ResolvedArm] = []
    for spec in ARMS:
        over: dict[str, object] = {**COMMON_MODEL, **spec.overrides}
        if spec.rank_is_d_model:
            over["rank"] = base.d_model
        if spec.widen:
            over["d_ffn"] = solve_d_ffn(over, target)
        run = _resolved(over)
        row = cm.measure(run, batch=BATCH, seq=SEQ)
        out.append(ResolvedArm(spec=spec, run=run, row=row, overrides=over))
    return out


def assert_equal_compute(rows: list[dict], *, control: str = "A", tol: float = COMPUTE_TOL) -> None:
    """Refuse a comparison at unequal active compute.

    `rows` carry `arm` and `active_macs`. The control row is the denominator;
    a missing control, a non-positive ledger, or any arm outside `tol` raises.
    The table is printed on a miss so the offending arm is visible without a
    second run.
    """
    by_name = {str(r["arm"]): r for r in rows}
    if control not in by_name:
        raise LoudError(
            f"equal-compute gate: no control arm {control!r}. Escape: keep the "
            f"control in the sweep; a comparison needs a denominator."
        )
    base = int(by_name[control]["active_macs"])
    if base <= 0:
        raise LoudError(
            f"equal-compute gate: control {control} has active_macs={base}. "
            f"Escape: fix capacity_math before comparing any arm."
        )
    bad = []
    for r in rows:
        macs = int(r["active_macs"])
        delta = (macs - base) / base
        if abs(delta) > tol:
            bad.append((str(r["arm"]), macs, delta))
    if not bad:
        return
    print(f"\nUNEQUAL ACTIVE COMPUTE (control {control} = {base:,} MAC/byte, tol {tol:.0%})")
    print(f"{'arm':>4s} {'active MAC/byte':>16s} {'vs control':>11s}")
    for name, macs, delta in [(control, base, 0.0)] + sorted(bad):
        print(f"{name:>4s} {macs:16,d} {delta:+11.2%}")
    raise LoudError(
        "the arms do not share active compute, so a BPB difference would price "
        "FLOPs, not capacity. Escape: re-solve d_ffn (solve_d_ffn) or drop the "
        "arm; do not compare at unequal compute."
    )


def summarize(values: list[float]) -> tuple[float, float]:
    """`(mean, max-min spread)`; an empty list is a VOID, not a zero."""
    assert values, "summarize() needs at least one value"
    return statistics.mean(values), max(values) - min(values)


def decide_row(
    control_values: list[float],
    arm_values: list[float],
    *,
    control_params: int,
    arm_params: int,
    min_seeds: int = MIN_SEEDS,
    min_effect: float = MIN_EFFECT,
) -> dict:
    """The pre-registered rule for one arm against the control.

    `WIN` needs `mean < cmean - max(2*cspread, 0.01)`; `LOSS` is worse than
    `cmean + need`; the band between is `TIE`. A control with fewer than
    `min_seeds` values or a zero spread RAISES: the bar would be the mean and an
    exact tie would promote (the `collect_ab` defect).
    """
    ctl = [float(v) for v in control_values]
    arm = [float(v) for v in arm_values]
    if len(ctl) < min_seeds:
        raise LoudError(
            f"control has {len(ctl)} seed(s), need {min_seeds}: a spread of 0 makes "
            f"an exact tie a WIN. Escape: run the control at {min_seeds} seeds."
        )
    cmean, cspread = summarize(ctl)
    if cspread <= 0.0:
        raise LoudError(
            f"control seeds are identical ({ctl}): the spread is 0 and the bar is "
            f"the mean, so a tie would promote. Escape: distinct seeds."
        )
    need = max(2 * cspread, min_effect)
    bar = cmean - need
    if len(arm) < min_seeds:
        return {
            "mean": statistics.mean(arm) if arm else float("nan"),
            "spread": max(arm) - min(arm) if len(arm) > 1 else 0.0,
            "delta": (statistics.mean(arm) - cmean) if arm else float("nan"),
            "need": need,
            "bar": bar,
            "params": arm_params,
            "verdict": "VOID",
        }
    mean, spread = summarize(arm)
    delta = mean - cmean
    verdict = "WIN" if mean < bar else "LOSS" if mean > cmean + need else "TIE"
    return {
        "mean": mean,
        "spread": spread,
        "delta": delta,
        "need": need,
        "bar": bar,
        "params": arm_params,
        "verdict": verdict,
    }


def recommend(rows: list[dict]) -> dict:
    """The winner under the pre-registered rule; a tie goes to fewer params.

    `rows[*]["verdict"]` is `WIN`/`TIE`/`LOSS`/`VOID`; the control row is
    `CONTROL` and participates only in the tie branch. A WIN is a result of its
    own — a cheaper tie does not displace a real improvement.
    """
    control = next((r for r in rows if r["verdict"] == "CONTROL"), None)
    if control is None:
        raise LoudError("recommend(): no control row. Escape: include arm A.")
    wins = [r for r in rows if r["verdict"] == "WIN"]
    if wins:
        best = min(wins, key=lambda r: (r["mean"], r["params"]))
        return {
            "arm": best["arm"],
            "reason": f"WIN: mean {best['mean']:.4f} < bar {best['bar']:.4f}",
        }
    tied = [r for r in rows if r["verdict"] in ("CONTROL", "TIE")]
    best = min(tied, key=lambda r: (r["params"], r["mean"]))
    return {
        "arm": best["arm"],
        "reason": (
            f"tie on BPB; the smaller model wins ({best['params']:,} params). "
            f"No arm cleared mean < {control['mean'] - control['need']:.4f}."
        ),
    }


def read_run(path: Path) -> dict:
    """The last eval BPB and the last log window's routing counters, from JSONL."""
    evals: list[dict] = []
    logs: list[dict] = []
    if not path.exists():
        return {"bpb": None, "scored": None, "live": None, "max_share": None, "nan": None}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "bpb" in row:
            evals.append(row)
        elif "counters" in row:
            logs.append(row)
    bpb = float(evals[-1]["bpb"]) if evals else None
    scored = int(evals[-1]["scored"]) if evals and "scored" in evals[-1] else None
    live = max_share = nan = None
    if logs:
        c = logs[-1].get("counters", {})
        n = float(c.get("moe_route_n", 0) or 0)
        if n > 0:
            live = float(c.get("moe_route_live", 0) or 0) / n
            max_share = float(c.get("moe_route_max_x1000", 0) or 0) / 1000.0
        nan = int(c.get("nan_skipped", 0) or 0)
    return {"bpb": bpb, "scored": scored, "live": live, "max_share": max_share, "nan": nan}


def _literal(value: object) -> str:
    """TOML spelling for `--set`: lowercase booleans; strings stay raw."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def model_diff(run: RunConfig) -> dict[str, object]:
    """The `--set` keys that turn the pinched preset into this resolved arm."""
    base = load_preset(str(BASE_PRESET))
    out: dict[str, object] = {}
    for field in dataclasses.fields(ModelConfig):
        now = getattr(run.model, field.name)
        if now != getattr(base, field.name):
            out[field.name] = now
    return out


def train_overrides(arm: ResolvedArm, seed: int, out_dir: Path, json_log: Path) -> dict:
    return {
        "steps": STEPS,
        "batch": BATCH,
        "seq_len": SEQ,
        "eval": True,
        "eval_every": STEPS,
        "eval_batches": EVAL_BATCHES,
        "log_every": STEPS,
        "ckpt_every": CKPT_EVERY,
        "ckpt_name": f"abcap_{arm.spec.name}_{seed}",
        "out_dir": str(out_dir),
        "seed": seed,
        "data": str(DATA),
        "eval_data": str(EVAL),
        "mix_plan": str(MIX_PLAN),
        "json_log": str(json_log),
    }


def build_command(
    arm: ResolvedArm, seed: int, out_root: Path, device: str = "cuda"
) -> tuple[list[str], Path, Path, Path]:
    """One run's exact argv and the three paths it writes (log, JSONL, snapshot)."""
    out_dir = out_root / f"{arm.spec.name}_s{seed}"
    json_log = out_dir / "run.jsonl"
    stdout = out_dir / "run.log"
    cmd = [
        sys.executable,
        "-m",
        "dormouse.cli",
        "train",
        str(BASE_PRESET),
        "--device",
        device,
    ]
    for key, value in model_diff(arm.run).items():
        cmd += ["--set", f"{key}={_literal(value)}"]
    for key, value in train_overrides(arm, seed, out_dir, json_log).items():
        cmd += ["--set", f"{key}={_literal(value)}"]
    return cmd, out_dir, stdout, json_log


def run_one(arm: ResolvedArm, seed: int, out_root: Path, device: str) -> dict:
    cmd, out_dir, stdout, json_log = build_command(arm, seed, out_root, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The resolved snapshot is evidence, not a checkpoint: `ckpt_every` is huge
    # and no `.pt` is written, but a BPB needs the config it was measured with.
    (out_dir / "resolved.config.toml").write_text(arm.run.snapshot_toml())
    env = dict(os.environ)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    with stdout.open("w") as fh:
        proc = subprocess.run(cmd, cwd=str(PORT), env=env, stdout=fh, stderr=subprocess.STDOUT)
    got = read_run(json_log)
    if got["live"] is None and arm.row["moe_topk"] == 0:
        # The dense blend has no top-k, so no routing counter: every token runs
        # the one expert, which IS 1/1 live. Printing n/a would hide that.
        got["live"] = float(arm.row["n_experts"])
    return {
        "arm": arm.spec.name,
        "seed": seed,
        "returncode": proc.returncode,
        "log": str(stdout),
        **got,
    }


def _preregistration() -> None:
    print(
        "rule (preregistered): WIN iff n>=3 and mean < control_mean - max(2*spread, "
        f"{MIN_EFFECT}); tie -> fewer params; spread=0 refuses."
    )


def _print_compute(arms: list[ResolvedArm]) -> None:
    print(
        f"\n{'arm':>4s} {'params':>14s} {'active MAC/byte':>16s} {'vs A':>7s} "
        f"{'bank':>14s} {'bank%':>6s} {'f':>7s} {'E':>5s} {'rank':>5s} {'topk':>5s} "
        f"{'tok/expert/step':>15s}"
    )
    base = arms[0].row["active_macs"]
    for a in arms:
        r = a.row
        print(
            f"{a.spec.name:>4s} {r['params']:14,d} {r['active_macs']:16,d} "
            f"{r['active_macs'] / base - 1:+7.2%} {r['bank']:14,d} "
            f"{100 * r['bank_share']:5.1f}% {r['d_ffn']:7d} {r['n_experts']:5d} "
            f"{r['rank']:5d} {r['moe_topk']:5d} {r['tokens_per_expert_step']:15.1f}"
        )


def _print_resolved(arms: list[ResolvedArm]) -> None:
    print("\nresolved arms (through config.load_run; the --set diff from m400b)")
    for a in arms:
        print(f"\n[{a.spec.name}] {a.spec.note}")
        print(
            f"  d_model={a.run.model.d_model} d_ffn={a.run.model.d_ffn} "
            f"E={a.run.model.n_experts} rank={a.run.model.rank} "
            f"topk={a.run.model.moe_topk} use_tsct={a.run.model.use_tsct} "
            f"share_u={a.run.model.share_u_mode}"
        )
        diff = model_diff(a.run)
        print("  --set " + " ".join(f"{k}={_literal(v)}" for k, v in diff.items()))


def _print_commands(arms: list[ResolvedArm], out_root: Path, device: str) -> None:
    print("\ncommands")
    for a in arms:
        for seed in a.spec.seeds:
            cmd, *_ = build_command(a, seed, out_root, device)
            print("  " + " ".join(cmd))


def _results_table(arms: list[ResolvedArm], runs: list[dict]) -> list[dict]:
    by_arm: dict[str, list[dict]] = {a.spec.name: [] for a in arms}
    for r in runs:
        by_arm.setdefault(r["arm"], []).append(r)
    control = next(a for a in arms if a.spec.name == "A")
    control_runs = by_arm["A"]
    ctl_bpb = [
        r["bpb"] for r in control_runs if r["bpb"] is not None and math.isfinite(float(r["bpb"]))
    ]
    if len(ctl_bpb) >= MIN_SEEDS:
        cmean, cspread = summarize(ctl_bpb)
        need = max(2 * cspread, MIN_EFFECT)
    else:
        cmean, cspread, need = float("nan"), 0.0, float("nan")

    def _void_reasons(rs: list[dict]) -> str:
        why: list[str] = []
        for r in rs:
            if r.get("returncode") != 0:
                why.append(f"seed {r['seed']}: exit {r.get('returncode')}")
            if r.get("bpb") is None:
                why.append(f"seed {r['seed']}: no eval BPB")
            elif not math.isfinite(float(r["bpb"])):
                why.append(f"seed {r['seed']}: non-finite BPB")
            if r.get("nan"):
                why.append(f"seed {r['seed']}: {r['nan']} NaN-skipped steps")
            if r.get("scored") not in (None, WINDOW):
                why.append(f"seed {r['seed']}: scored {r['scored']} != {WINDOW} B")
        return "; ".join(why)

    rows: list[dict] = [
        {
            "arm": "A",
            "params": control.row["params"],
            "mean": cmean,
            "spread": cspread,
            "need": need,
            "bar": cmean - need,
            "verdict": "CONTROL",
            "live": [r.get("live") for r in control_runs],
            "void": _void_reasons(control_runs),
        }
    ]

    for a in arms:
        if a.spec.name == "A":
            continue
        rs = by_arm[a.spec.name]
        vals = [
            float(r["bpb"]) for r in rs if r["bpb"] is not None and math.isfinite(float(r["bpb"]))
        ]
        void = _void_reasons(rs)
        dec = decide_row(
            ctl_bpb, vals, control_params=control.row["params"], arm_params=a.row["params"]
        )
        if void and dec["verdict"] != "VOID":
            dec["verdict"] = "VOID"
        rows.append(
            {
                "arm": a.spec.name,
                "params": a.row["params"],
                "live": [r.get("live") for r in rs],
                "void": void,
                **dec,
            }
        )
    print(
        f"\n{'arm':>4s} {'params':>14s} {'MAC/byte':>12s} {'mean BPB':>9s} {'spread':>8s} "
        f"{'delta':>8s} {'verdict':>8s} {'live/experts':>18s}"
    )
    for r in rows:
        a = next(x for x in arms if x.spec.name == r["arm"])
        live = r.get("live") or []
        live_txt = ",".join(f"{v:.1f}" for v in live if v is not None) or "n/a"
        delta = r.get("delta", 0.0 if r["arm"] == "A" else float("nan"))
        print(
            f"{r['arm']:>4s} {r['params']:14,d} {a.row['active_macs']:12,d} "
            f"{r['mean']:9.4f} {r['spread']:8.4f} {delta:+8.4f} {r['verdict']:>8s} "
            f"{live_txt + '/' + str(a.row['n_experts']):>18s}"
        )
        if r.get("void"):
            print(f"     VOID: {r['void']}")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dry-run", action="store_true", help="resolve + compute table, no card")
    ap.add_argument("--arms", default="A,B,C,D,E", help="comma-separated arm names")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--device", default="cuda", help="cuda|cpu (cpu for a shape smoke)")
    a = ap.parse_args(argv)

    if a.dry_run:
        # No card, no checkpoint, no CUDA context. `capacity_math` builds on
        # `meta`; no torch.cuda call is made anywhere on this path.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    selected = [x.strip() for x in a.arms.split(",") if x.strip()]
    unknown = [x for x in selected if x not in {s.name for s in ARMS}]
    if unknown:
        raise LoudError(f"unknown arm(s) {unknown}. Known: {[s.name for s in ARMS]}")
    if "A" not in selected:
        raise LoudError(
            "arm A is the control; the decision rule needs it. Escape: include A "
            "in --arms (e.g. --arms A,B)."
        )

    arms = [x for x in resolve_arms() if x.spec.name in selected]
    print(f"preset {BASE_PRESET}")
    print(f"data {DATA}  eval {EVAL}  mix_plan {MIX_PLAN}")
    print(f"steps={STEPS} b{BATCH} s{SEQ} eval={EVAL_BATCHES}x{BATCH}x{SEQ}={WINDOW} B")
    _preregistration()
    _print_compute(arms)
    assert_equal_compute([{"arm": x.spec.name, "active_macs": x.row["active_macs"]} for x in arms])
    _print_resolved(arms)

    if a.dry_run:
        _print_commands(arms, a.out_root, a.device)
        print(f"\ndry-run: no card, no checkpoint; out_root {a.out_root} untouched.")
        return 0

    import _lane  # sibling driver guard: no torch CUDA call in the parent

    why = _lane.data_missing(str(DATA), str(EVAL), str(MIX_PLAN))
    why = why or _lane.preflight(a.device)
    if why:
        return _lane.skip(why)

    print(f"\nruns -> {a.out_root}")
    runs: list[dict] = []
    for arm in arms:
        for seed in arm.spec.seeds:
            print(f"[run] arm={arm.spec.name} seed={seed} params={arm.row['params']:,}", flush=True)
            runs.append(run_one(arm, seed, a.out_root, a.device))
    rows = _results_table(arms, runs)
    pick = recommend(rows)
    print(f"\ndecision: {pick['arm']} — {pick['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
