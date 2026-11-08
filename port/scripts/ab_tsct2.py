#!/usr/bin/env python3
"""TSCT-2's pre-registered A/B: do the three mechanisms bring the bank back?

    python scripts/ab_tsct2.py plan           # resolve arms + the price table; no card
    python scripts/ab_tsct2.py run            # 5 arms x 3 seeds x 2000 steps
    python scripts/ab_tsct2.py report --out ~/logs/ab_tsct2

**THIS DRIVER HAS NOT BEEN RUN.** The card is held by the live m400b run
(AGENTS §1.5: one heavy thing at a time), so no BPB below is measured; `plan`
is the only mode the branch executes.

WHAT IS COMPARED. Base preset `configs/m400b.toml` — the 199 275 551-parameter
bank the 2026-11-05 review lost by 0.046 BPB to a dense 81M model — with
`moe_routing_health=True` on every arm (a counter set, no parameter). The five
arms differ ONLY in the three TSCT-2 flags:

  control   today's bank (`E=512`, top-2, rank 64, plane 64, `v [64, 4096]`).
  +basis    `tsct_expert_basis=64`: widen the routed plane to `shared_rank=192`
            and give each expert `w [64, 192]` (12 288 params). The bank folds
            `gate_u @ w.T` at the rebuild, so the per-token operands and MAC
            are unchanged; the fold itself costs `2·E·d·rank·r_shared` MAC once
            per optimizer step (printed by `plan`, excluded from active MAC).
  +lr       `tsct_expert_lr=1.0`: the full documented multiplier
            `lr_j = lr·(1 + c·(mean(L)/max(L_j,1) − 1))`, clipped [0.2, 5], on
            each expert's `v` only, from the device-side `lfb_counts`.
  +revive   `tsct_expert_revive=100`, `tsct_revive_noise=0.01`: every 100
            optimizer steps the coldest expert takes the hottest's
            `v`/`gate_s`/`up_s` + `σ=0.01·std(v_hot)` and both LFB entries
            reset.
  all       all three.

RECIPE. The 2026-11-05 bank measurement's shape: 2000 steps, `b8 x s1024`,
`mix_v7` over `data_local/best/fwedu.bin`, eval on `data_local/eval_mix` at
step 2000 with 8 batches. The eval window is `8 x 8 x 1024 = 65 536 B` on EVERY
arm and seed; a row whose `scored` is not that is VOID, not a number.

EQUAL ACTIVE MAC. `+basis` does not add per-token compute: the selector is
folded into the effective `[d, rank]` basis at the bank rebuild, so the forward
reads exactly the control's shapes. `+lr` and `+revive` are training-time and
add none. `plan` recomputes `capacity_math.active_macs_per_token` for all five
resolved configs and refuses a mismatch above 0.5 %. The fold is priced
separately because it is a per-STEP build cost, not an active-MAC cost.

PRE-REGISTERED DECISION RULE (written BEFORE the first run):

  1. VOID: any seed with a NaN-skipped step, a missing/short eval window
     (`scored != 65 536 B`), or a non-finite BPB is VOID, full stop.
  2. ROUTING HEALTH GATE: an arm must keep `moe_route_live / moe_route_n >=
     0.70` in EVERY seed's last log window (the bank's "live >= 70 %" rule).
     A failing arm is VOID even if its BPB is good — a bank that wins by
     killing speakers is not the mechanism.
  3. WIN: `mean < control_mean - max(2·control_spread, 0.01)` on the same three
     seeds, where `spread = max - min` of the CONTROL's own seeds. Both bounds
     are required: the 2x-spread half is the seed noise, the 0.01 half is the
     `collect_ab` floor.
  4. LOSS: worse than `control_mean + max(2·control_spread, 0.01)`.
  5. TIE: the band between. **A tie goes to the arm with FEWER parameters**
     (control is 6 684 672 parameters lighter than every TSCT-2 arm; the
     selector arm is +6.68M, the LR/revive arms are +0).
  6. If no arm wins, the bank has not been rescued at 22 GB of data; the
     conclusion in `docs/reviews/2026-11-05-dense-beats-bank.md` stands.

`+basis` is the only arm with a parameter cost, so it is the one a tie deletes.
The LR and revive arms add zero parameters and are deleted only by a LOSS or a
health-gate VOID.

USAGE
    python scripts/ab_tsct2.py plan                 # configs + MAC + param table
    python scripts/ab_tsct2.py run                  # all 15 runs, JSONL under ~/logs/ab_tsct2
    python scripts/ab_tsct2.py report --out PATH
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import subprocess
import sys
from pathlib import Path

PORT = Path(__file__).resolve().parents[1]
REPO = PORT.parent
sys.path.append(str(PORT / "scripts"))
sys.path.insert(0, str(PORT / "src"))

import capacity_math as cm  # noqa: E402
from _lane import data_missing, preflight, skip  # noqa: E402

from dormouse.config import ModelConfig, RunConfig, load_preset, load_run  # noqa: E402
from dormouse.instrument import LoudError  # noqa: E402
from dormouse.model.experts import tsct_basis_param_delta  # noqa: E402

BASE_PRESET = PORT / "configs" / "m400b.toml"
DATA = REPO / "data_local" / "best" / "fwedu.bin"
EVAL = REPO / "data_local" / "eval_mix" / "eval_mix.bin"
MIX_PLAN = PORT / "configs" / "mix_v7.toml"

STEPS = 2000
BATCH = 8
SEQ = 1024
EVAL_BATCHES = 8
EVAL_EVERY = STEPS
LOG_EVERY = 200
WINDOW = EVAL_BATCHES * BATCH * SEQ
SEEDS = (1, 2, 3)
MIN_EFFECT = 0.01
MIN_LIVE = 0.70
COMPUTE_TOL = 0.005
CKPT_EVERY = 1_000_000_000
DEFAULT_OUT = Path("~/logs/ab_tsct2").expanduser()

#: The counter set is the routing-health gate's instrument; it is a counter, not
#: a parameter, so every arm and the control carry it.
COMMON_MODEL: dict[str, object] = {"moe_routing_health": True}

ARMS: tuple[tuple[str, dict[str, object], str], ...] = (
    ("control", {}, "today's bank: E=512, top-2, rank 64, plane 64"),
    (
        "+basis",
        {"tsct_expert_basis": 64},
        "per-expert w [64, 192] inside a [d, 192] shared plane; folded at rebuild",
    ),
    (
        "+lr",
        {"tsct_expert_lr": 1.0},
        "per-expert Muon LR from the device counts, clipped [0.2, 5]",
    ),
    (
        "+revive",
        {"tsct_expert_revive": 100, "tsct_revive_noise": 0.01},
        "coldest <- hottest + seeded noise, every 100 optimizer steps",
    ),
    (
        "all",
        {
            "tsct_expert_basis": 64,
            "tsct_expert_lr": 1.0,
            "tsct_expert_revive": 100,
            "tsct_revive_noise": 0.01,
        },
        "the three together",
    ),
)


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    run: RunConfig
    row: dict
    note: str


def resolve_arms() -> list[Arm]:
    """Every arm through `load_run` (the real preset/refusal path)."""
    out: list[Arm] = []
    for name, over, note in ARMS:
        run = load_run(str(BASE_PRESET), model_overrides={**COMMON_MODEL, **over})
        out.append(Arm(name, run, cm.measure(run, batch=BATCH, seq=SEQ), note))
    return out


def fold_macs(cfg: ModelConfig) -> int:
    """The basin build's extra MAC: `gate_u @ w_j.T` for both sites, per step."""
    return 2 * cfg.n_experts * cfg.d_model * cfg.rank * cfg.shared_rank


def check_equal_active_macs(arms: list[Arm]) -> None:
    base = int(arms[0].row["active_macs"])
    bad = [(a.name, int(a.row["active_macs"])) for a in arms if int(a.row["active_macs"]) != base]
    if bad:
        table = "\n".join(f"  {n:>8s} {m:,}" for n, m in [(arms[0].name, base), *bad])
        raise LoudError(
            f"the TSCT-2 arms do not share active MAC (control {base:,}):\n{table}\n"
            f"Escape: the selector must stay FOLDED into the effective basis; a "
            f"literal per-token `(x @ gate_u) @ w.T` would add {2 * 64 * 192:,} "
            f"MAC/assignment and the BPB difference would price FLOPs."
        )


def model_diff(run: RunConfig) -> dict[str, object]:
    """The `--set` keys that turn the pinched preset into this resolved arm."""
    base = load_preset(str(BASE_PRESET))
    out: dict[str, object] = {}
    for field in dataclasses.fields(ModelConfig):
        now = getattr(run.model, field.name)
        if now != getattr(base, field.name):
            out[field.name] = now
    return out


def _literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_command(arm: Arm, seed: int, out: Path, device: str) -> tuple[list[str], Path]:
    out_dir = out / f"{arm.name.replace('+', 'plus_')}_s{seed}"
    json_log = out_dir / "run.jsonl"
    cmd = [sys.executable, "-m", "dormouse.cli", "train", str(BASE_PRESET), "--device", device]
    for key, value in model_diff(arm.run).items():
        cmd += ["--set", f"{key}={_literal(value)}"]
    train = {
        "steps": STEPS,
        "batch": BATCH,
        "seq_len": SEQ,
        "eval": True,
        "eval_every": EVAL_EVERY,
        "eval_batches": EVAL_BATCHES,
        "log_every": LOG_EVERY,
        "ckpt_every": CKPT_EVERY,
        "ckpt_name": f"abtsct2_{arm.name}_{seed}",
        "out_dir": str(out_dir),
        "seed": seed,
        "data": str(DATA),
        "eval_data": str(EVAL),
        "mix_plan": str(MIX_PLAN),
        "json_log": str(json_log),
    }
    for key, value in train.items():
        cmd += ["--set", f"{key}={_literal(value)}"]
    return cmd, json_log


def plan(args: argparse.Namespace) -> int:
    arms = resolve_arms()
    check_equal_active_macs(arms)
    control = arms[0]
    print(f"preset {BASE_PRESET}  steps {STEPS}  b{BATCH} x s{SEQ}  window {WINDOW:,} B")
    print(
        f"{'arm':>8s} {'params':>12s} {'bank':>12s} {'active MAC/B':>14s} "
        f"{'Δparams':>10s} {'fold MAC/step':>14s}"
    )
    for a in arms:
        cfg = a.run.model
        delta = int(a.row["params"]) - int(control.row["params"])
        fold = fold_macs(cfg) if cfg.tsct_expert_basis else 0
        print(
            f"{a.name:>8s} {int(a.row['params']):12,d} {int(a.row['bank']):12,d} "
            f"{int(a.row['active_macs']):14,d} {delta:+10,d} {fold:14,d}"
        )
        print(f"         {a.note}")
    expected = tsct_basis_param_delta(512, 1536, 64, 192)
    got = int(next(a for a in arms if a.name == "+basis").row["params"]) - int(
        control.row["params"]
    )
    assert got == expected, f"+basis delta {got:,} != tsct_basis_param_delta {expected:,}"
    print(
        f"\n+params (doc): {expected:,} = 512·64·192 selectors + 2·1536·128 wider "
        f"plane; the doc's +6.3M counts only the selectors."
    )
    print(
        f"fold: {fold_macs(control.run.model):,} MAC once per optimizer step "
        f"(not per token; active MAC/B unchanged by construction)."
    )
    print(
        "\nDECISION RULE: arm WINs iff (a) every seed is finite and scored "
        f"{WINDOW:,} B, (b) final-window live/E >= {MIN_LIVE:.0%} in every seed, "
        "(c) mean < control_mean - max(2*control_spread, 0.01). Otherwise TIE "
        "inside the band (fewer parameters wins) or LOSS. A VOID seed voids the arm."
    )
    return 0


def read_run(path: Path) -> dict:
    evals: list[dict] = []
    logs: list[dict] = []
    if not path.exists():
        return {"bpb": None, "scored": None, "live": None, "nan": None}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "bpb" in row:
            evals.append(row)
        elif "counters" in row:
            logs.append(row)
    bpb = float(evals[-1]["bpb"]) if evals and evals[-1].get("bpb") is not None else None
    scored = int(evals[-1]["scored"]) if evals and "scored" in evals[-1] else None
    live = nan = None
    if logs:
        c = logs[-1].get("counters", {})
        n = float(c.get("moe_route_n", 0) or 0)
        if n > 0:
            live = float(c.get("moe_route_live", 0) or 0) / n
        nan = int(c.get("nan_skipped", 0) or 0)
    return {"bpb": bpb, "scored": scored, "live": live, "nan": nan}


def summarize(values: list[float]) -> tuple[float, float]:
    assert values, "summarize() needs at least one value"
    return statistics.mean(values), max(values) - min(values)


def decide(
    control: list[dict], arm: list[dict], *, params: int, cparams: int, e_total: int
) -> dict:
    """One arm against the control under the file's pre-registered rule."""
    ctl = [
        r["bpb"] for r in control if r["bpb"] is not None and r["scored"] == WINDOW and not r["nan"]
    ]
    if len(ctl) < 3:
        raise LoudError(f"control has {len(ctl)} finite {WINDOW}-B seeds, need 3")
    cmean, cspread = summarize(ctl)
    need = max(2.0 * cspread, MIN_EFFECT)
    rows = arm
    healthy = all(r["live"] is not None and r["live"] / e_total >= MIN_LIVE for r in rows)
    void = any(r["bpb"] is None or r["scored"] != WINDOW or r["nan"] for r in rows)
    if void or not healthy:
        return {
            "mean": statistics.mean([r["bpb"] for r in rows if r["bpb"] is not None])
            if any(r["bpb"] is not None for r in rows)
            else float("nan"),
            "spread": 0.0,
            "delta": float("nan"),
            "need": need,
            "verdict": "VOID",
            "why": "non-finite/short window" if void else f"live/E < {MIN_LIVE:.0%}",
        }
    vals = [r["bpb"] for r in rows]
    mean, spread = summarize(vals)
    delta = mean - cmean
    verdict = "WIN" if mean < cmean - need else "LOSS" if mean > cmean + need else "TIE"
    return {
        "mean": mean,
        "spread": spread,
        "delta": delta,
        "need": need,
        "verdict": verdict,
        "why": "",
        "params": params,
        "cparams": cparams,
    }


def report(out: Path) -> int:
    e_total = load_run(str(BASE_PRESET), model_overrides=COMMON_MODEL).model.n_experts
    data: dict[str, list[dict]] = {}
    for name, _, _ in ARMS:
        data[name] = [
            read_run(out / f"{name.replace('+', 'plus_')}_s{s}" / "run.jsonl") for s in SEEDS
        ]
    control = data["control"]
    ctl = [r["bpb"] for r in control if r["bpb"] is not None]
    if len(ctl) < 3:
        raise LoudError(f"report: control has {len(ctl)}/{len(SEEDS)} seeds under {out}")
    cmean, cspread = summarize(ctl)
    need = max(2.0 * cspread, MIN_EFFECT)
    cparams = int(
        cm.measure(load_run(str(BASE_PRESET), model_overrides=COMMON_MODEL), batch=BATCH, seq=SEQ)[
            "params"
        ]
    )
    print(f"control mean={cmean:.4f} spread={cspread:.4f} bar={cmean - need:.4f}")
    print(
        f"{'arm':>8s} {'mean':>8s} {'spread':>8s} {'delta':>8s} {'live':>7s} "
        f"{'params':>12s} {'verdict':>8s}"
    )
    decisions: list[dict] = [
        {
            "arm": "control",
            "mean": cmean,
            "spread": cspread,
            "delta": 0.0,
            "verdict": "CONTROL",
            "params": cparams,
        }
    ]
    for name, _, _ in ARMS[1:]:
        rows = data[name]
        run = load_run(
            str(BASE_PRESET), model_overrides={**COMMON_MODEL, **dict(ARMS_by_name()[name])}
        )
        params = int(cm.measure(run, batch=BATCH, seq=SEQ)["params"])
        d = decide(control, rows, params=params, cparams=cparams, e_total=e_total)
        lives = [r["live"] / e_total for r in rows if r["live"] is not None]
        live = min(lives) if lives else float("nan")
        print(
            f"{name:>8s} {d['mean']:8.4f} {d['spread']:8.4f} {d['delta']:+8.4f} "
            f"{live:7.1%} {params:12,d} {d['verdict']:>8s}"
            + (f"  ({d['why']})" if d["why"] else "")
        )
        d["arm"] = name
        decisions.append(d)
    wins = [d for d in decisions if d["verdict"] == "WIN"]
    if wins:
        best = min(wins, key=lambda d: d["mean"])
        print(f"\nWIN: {best['arm']} mean={best['mean']:.4f} < bar={cmean - need:.4f}")
        return 0
    tied = [d for d in decisions if d["verdict"] in ("CONTROL", "TIE")]
    best = min(tied, key=lambda d: (d["params"], d["mean"]))
    print(
        f"\nno WIN; tie/control goes to fewer parameters: {best['arm']} ({best['params']:,} params)"
    )
    return 0


def ARMS_by_name() -> dict[str, dict[str, object]]:
    return {name: over for name, over, _ in ARMS}


def run(args: argparse.Namespace) -> int:
    why = preflight(args.device)
    if why is not None:
        return skip(why)
    missing = data_missing(DATA, EVAL)
    if missing:
        return skip(missing)
    arms = resolve_arms()
    check_equal_active_macs(arms)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        for seed in SEEDS:
            cmd, json_log = build_command(arm, seed, out, args.device)
            json_log.parent.mkdir(parents=True, exist_ok=True)
            print(f"[run] {arm.name} seed {seed}: {' '.join(cmd)}", flush=True)
            r = subprocess.run(cmd, cwd=PORT)
            if r.returncode != 0:
                raise LoudError(
                    f"{arm.name} seed {seed} died with {r.returncode}; the JSONL is "
                    f"{json_log}. A dead run is VOID, not a slow arm."
                )
    return report(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)
    pl = sub.add_parser("plan", help="resolve the arms, price them, print the rule")
    pl.set_defaults(func=plan)
    r = sub.add_parser("run", help="run all 15 arms/seeds, then report")
    r.add_argument("--device", default="cuda")
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.set_defaults(func=run)
    rp = sub.add_parser("report", help="read an existing run directory")
    rp.add_argument("--out", default=str(DEFAULT_OUT))
    rp.set_defaults(func=lambda a: report(Path(a.out).expanduser()))
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
