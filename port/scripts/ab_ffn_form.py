#!/usr/bin/env python3
"""The FFN-form A/B: does a full-rank FFN beat a compressed TSCT factorisation?

    python scripts/ab_ffn_form.py --dry-run
    python scripts/ab_ffn_form.py --arms dense,tsct_r512,tsct_full
    python scripts/ab_ffn_form.py report --out-root /tmp/ab_ffn_form

THE QUESTION. `DualLatentGLU` always ran the expert through a rank-`rank`
latent, even when `use_tsct = false`, so the "dense" arm of the 2026-11-05
capacity sweep was a partially factorised FFN. This driver holds everything
else fixed (`use_tsct = false` everywhere, `n_experts = 1`, `moe_topk = 0`) and
varies ONLY `ffn_form`:

  dense                one plain GLU: `3·d·f + f + d`, no rank anywhere.
  tsct_r256/512/1024   all three matrices factorised at that rank (gate AND up
                       at `r_g`, down at `r_d = r_g`): compressed, and NOT
                       output-equivalent to dense.
  tsct_eq              `r_g = 1024`, `r_d` solved so the FFN slot's parameter
                       count equals dense's (the equal-budget control).
  tsct_full            `r_g = r_d = min(d, f)`: an exact reparameterisation of
                       the dense GLU (same function class, more parameters).
                       `tests/test_ffn_form.py` proves the equivalence at tiny
                       shape; this arm is the same check at TRAINING scale.

THE BASE. `configs/m400.toml` (d=1536, f=4096, T=3), with `n_experts=1`,
`moe_topk=0`, `share_u_mode='off'` and `use_tsct=false` on every arm. The
`use_tsct=false` pin is deliberate: it is the switch the defect says gives a
dense FFN, and here it changes only the non-FFN linears. A cross with
`use_tsct=true` is a second axis for a later run, not a confound in this one.

THE RUN. 1500 steps, `b8 × s1024`, the same `mix_v7` mixture and eval tail the
capacity sweep used; eval at step 1500 with 8 batches (`65 536` scored bytes)
on every arm and seed. Two seeds per arm, distinct across the sweep. No
checkpoint (`ckpt_every` is huge), so this is a screen, not a recipe.

PRE-REGISTERED RULE (written before the first run):

  1. VOID: any seed with a NaN-skipped step, a missing/short eval window
     (`scored != 65 536 B`), a non-zero exit, or a non-finite BPB.
  2. WIN: `mean < control_mean − max(2·control_spread, 0.01)`, where `spread`
     is max−min over the DENSE control's own seeds and 0.01 is the
     `collect_ab` floor. The rule needs at least 2 control seeds; a zero
     spread does not promote anything.
  3. TIE: inside the band. **A tie goes to the arm with FEWER parameters**
     (the cheap model wins a tie); `tsct_full` loses that tie by construction.
  4. LOSS: worse than `control_mean + max(2·control_spread, 0.01)`.

`--dry-run` resolves every arm through `config.load_run` and prints the
parameter + active-MAC table; no CUDA context, no checkpoint and no card.
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
sys.path.append(str(PORT / "scripts"))
if str(PORT / "src") not in sys.path:
    sys.path.insert(0, str(PORT / "src"))

import capacity_math as cm  # noqa: E402

from dormouse import width  # noqa: E402
from dormouse.config import ModelConfig, RunConfig, load_preset, load_run  # noqa: E402
from dormouse.instrument import LoudError  # noqa: E402

BASE_PRESET = PORT / "configs" / "m400.toml"
DATA = REPO / "data_local" / "best" / "fwedu.bin"
EVAL = REPO / "data_local" / "eval_mix" / "eval_mix.bin"
MIX_PLAN = PORT / "configs" / "mix_v7.toml"

STEPS = 1500
BATCH = 8
SEQ = 1024
EVAL_BATCHES = 8
EVAL_EVERY = STEPS
LOG_EVERY = 150
WINDOW = EVAL_BATCHES * BATCH * SEQ
SEEDS = (1, 2)
MIN_EFFECT = 0.01
CKPT_EVERY = 1_000_000_000
DEFAULT_OUT = Path("/tmp/ab_ffn_form")

#: Every arm is ONE FFN, no bank, and the non-FFN linears are dense: the
#: switch the defect names is held false so the FFN form is the only variable.
COMMON: dict[str, object] = {
    "use_tsct": False,
    "n_experts": 1,
    "moe_topk": 0,
    "share_u_mode": "off",
    # No router exists on either non-bank form; a live LFB step would be inert.
    "moe_lb_coef": 0.0,
}


@dataclass(frozen=True)
class Arm:
    name: str
    over: dict[str, object]
    run: RunConfig
    row: dict
    note: str


def _equal_rank_down(d: int, f: int, rank_gate: int) -> int:
    """`r_d` that makes the TSCT FFN's params equal to dense's (rounded)."""
    dense = width.ffn_params("dense", d, f)
    target = dense - (f + d)  # the biases are not factorised
    rd = round(target / (d + f + 1)) - 2 * rank_gate
    return max(1, min(rd, min(d, f)))


def arm_specs() -> list[tuple[str, dict[str, object], str]]:
    base = load_preset(str(BASE_PRESET))
    d, f, lo = base.d_model, base.d_ffn, min(base.d_model, base.d_ffn)
    rd_eq = _equal_rank_down(d, f, 1024)
    return [
        ("dense", {"ffn_form": "dense"}, "3·d·f + f + d, no rank"),
        (
            "tsct_r256",
            {"ffn_form": "tsct", "ffn_rank_gate": 256, "ffn_rank_down": 256},
            "compressed r_g=r_d=256",
        ),
        (
            "tsct_r512",
            {"ffn_form": "tsct", "ffn_rank_gate": 512, "ffn_rank_down": 512},
            "compressed r_g=r_d=512",
        ),
        (
            "tsct_r1024",
            {"ffn_form": "tsct", "ffn_rank_gate": 1024, "ffn_rank_down": 1024},
            "compressed r_g=r_d=1024",
        ),
        (
            "tsct_eq",
            {"ffn_form": "tsct", "ffn_rank_gate": 1024, "ffn_rank_down": rd_eq},
            f"equal budget r_g=1024, r_d={rd_eq}",
        ),
        (
            "tsct_full",
            {"ffn_form": "tsct", "ffn_rank_gate": lo, "ffn_rank_down": lo},
            f"exact reparameterisation r_g=r_d={lo}",
        ),
    ]


def resolve_arms() -> list[Arm]:
    out: list[Arm] = []
    for name, over, note in arm_specs():
        run = load_run(str(BASE_PRESET), model_overrides={**COMMON, **over})
        out.append(Arm(name, over, run, cm.measure(run, batch=BATCH, seq=SEQ), note))
    return out


def _preregistration() -> None:
    print(
        "rule (preregistered): VOID on NaN/short-window/non-finite; WIN iff "
        "mean < control_mean - max(2*control_spread, "
        f"{MIN_EFFECT}); TIE -> FEWER params; LOSS above the upper band."
    )


def _params_table(arms: list[Arm]) -> None:
    dense = arms[0]
    print(
        f"\n{'arm':>10s} {'params':>13s} {'ffn slot':>12s} {'Δffn':>10s} "
        f"{'active MAC/B':>14s} {'note':>34s}"
    )
    for a in arms:
        cfg = a.run.model
        slot = width.ffn_params(
            cfg.ffn_form,
            cfg.d_model,
            cfg.d_ffn,
            rank=cfg.rank,
            rank_gate=cfg.ffn_rank_gate,
            rank_down=cfg.ffn_rank_down,
            n_experts=cfg.n_experts,
            shared_rank=cfg.shared_rank,
        )
        dslot = slot - width.ffn_params("dense", cfg.d_model, cfg.d_ffn)
        assert slot == int(a.row["ffn_slot"]), (a.name, slot, a.row.get("ffn_slot"))
        print(
            f"{a.name:>10s} {int(a.row['params']):13,d} {slot:12,d} {dslot:+10,d} "
            f"{int(a.row['active_macs']):14,d} {a.note:>34s}"
        )
    # The equal-budget arm's own contract: within 0.1 % of dense, or the arm is
    # mislabelled and the table is the evidence.
    eq = next(a for a in arms if a.name == "tsct_eq")
    got = int(eq.row["ffn_slot"])
    dc = width.ffn_params("dense", dense.run.model.d_model, dense.run.model.d_ffn)
    if abs(got - dc) > max(1, dc // 1000):
        raise LoudError(
            f"tsct_eq FFN slot {got:,} is not equal to dense {dc:,} within 0.1 % "
            f"(Δ={got - dc:+,}). Escape: check _equal_rank_down."
        )


def _literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def model_diff(run: RunConfig) -> dict[str, object]:
    base = load_preset(str(BASE_PRESET))
    out: dict[str, object] = {}
    for field in dataclasses.fields(ModelConfig):
        now = getattr(run.model, field.name)
        if now != getattr(base, field.name):
            out[field.name] = now
    return out


def build_command(arm: Arm, seed: int, out_root: Path, device: str) -> tuple[list[str], Path, Path]:
    out_dir = out_root / f"{arm.name}_s{seed}"
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
        "ckpt_name": f"abffn_{arm.name}_{seed}",
        "out_dir": str(out_dir),
        "seed": seed,
        "data": str(DATA),
        "eval_data": str(EVAL),
        "mix_plan": str(MIX_PLAN),
        "json_log": str(json_log),
    }
    for key, value in train.items():
        cmd += ["--set", f"{key}={_literal(value)}"]
    return cmd, out_dir, json_log


def read_run(path: Path) -> dict:
    if not path.exists():
        return {"bpb": None, "scored": None, "nan": None}
    evals: list[dict] = []
    logs: list[dict] = []
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
    nan = int(logs[-1].get("counters", {}).get("nan_skipped", 0) or 0) if logs else None
    return {"bpb": bpb, "scored": scored, "nan": nan}


def summarize(values: list[float]) -> tuple[float, float]:
    assert values, "summarize() needs at least one value"
    return statistics.mean(values), max(values) - min(values)


def decide_row(control: list[float], arm: list[float], *, cparams: int, aparams: int) -> dict:
    """One arm against the control under the file's pre-registered rule."""
    cmean, cspread = summarize(control)
    need = max(2.0 * cspread, MIN_EFFECT)
    if not control or cspread == 0.0:
        raise LoudError("a zero control spread cannot promote anything")
    mean, spread = summarize(arm)
    delta = mean - cmean
    verdict = "WIN" if mean < cmean - need else "LOSS" if mean > cmean + need else "TIE"
    reason = f"delta={delta:+.4f}, need {need:.4f}"
    if verdict == "TIE" and aparams < cparams:
        reason += f"; tie -> fewer params ({aparams:,} < {cparams:,})"
    return {
        "mean": mean,
        "spread": spread,
        "delta": delta,
        "need": need,
        "verdict": verdict,
        "reason": reason,
    }


def plan(arms: list[Arm], out_root: Path, device: str) -> None:
    print(f"preset {BASE_PRESET}  steps {STEPS}  b{BATCH} x s{SEQ}  window {WINDOW:,} B")
    _preregistration()
    _params_table(arms)
    print("\ncommands")
    for a in arms:
        for seed in SEEDS:
            cmd, *_ = build_command(a, seed, out_root, device)
            print("  " + " ".join(cmd))


def run(arms: list[Arm], out_root: Path, device: str) -> list[dict]:
    import _lane

    why = _lane.data_missing(str(DATA), str(EVAL), str(MIX_PLAN))
    why = why or _lane.preflight(device)
    if why:
        return [{"skip": why}]
    runs: list[dict] = []
    for a in arms:
        for seed in SEEDS:
            cmd, out_dir, json_log = build_command(a, seed, out_root, device)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "resolved.config.toml").write_text(a.run.snapshot_toml())
            env = dict(os.environ)
            if device == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            with (out_dir / "run.log").open("w") as fh:
                proc = subprocess.run(
                    cmd, cwd=str(PORT), env=env, stdout=fh, stderr=subprocess.STDOUT
                )
            runs.append(
                {
                    "arm": a.name,
                    "seed": seed,
                    "params": int(a.row["params"]),
                    "returncode": proc.returncode,
                    **read_run(json_log),
                }
            )
    return runs


def report(arms: list[Arm], runs: list[dict]) -> int:
    if runs and "skip" in runs[0]:
        print(f"SKIP: {runs[0]['skip']}")
        return 0
    by: dict[str, list[dict]] = {a.name: [] for a in arms}
    for r in runs:
        by.setdefault(r["arm"], []).append(r)
    control = by["dense"]
    ctl = [
        float(r["bpb"])
        for r in control
        if r.get("bpb") is not None and math.isfinite(float(r["bpb"]))
    ]
    if len(ctl) < 2:
        print(f"dense control has {len(ctl)} finite seeds, need {len(SEEDS)}")
        return 1
    cmean, cspread = summarize(ctl)
    print(
        f"{'arm':>10s} {'params':>13s} {'mean BPB':>9s} {'spread':>8s} "
        f"{'delta':>8s} {'verdict':>8s} {'reason':>40s}"
    )
    print(
        f"{'dense':>10s} {control[0]['params']:13,d} {cmean:9.4f} {cspread:8.4f} "
        f"{0.0:+8.4f} {'CONTROL':>8s}"
    )
    rows: list[dict] = []
    for a in arms[1:]:
        rs = by[a.name]
        void = []
        vals: list[float] = []
        for r in rs:
            if r.get("returncode") != 0:
                void.append(f"seed {r['seed']}: exit {r['returncode']}")
            if r.get("bpb") is None or not math.isfinite(float(r["bpb"])):
                void.append(f"seed {r['seed']}: no finite BPB")
            elif int(r.get("nan") or 0):
                void.append(f"seed {r['seed']}: {r['nan']} NaN-skipped")
            elif r.get("scored") != WINDOW:
                void.append(f"seed {r['seed']}: scored {r.get('scored')} != {WINDOW}")
            else:
                vals.append(float(r["bpb"]))
        if void:
            print(
                f"{a.name:>10s} {int(a.row['params']):13,d} {'VOID':>9s} "
                f"{'':8s} {'':8s} {'VOID':>8s} {'; '.join(void):>40s}"
            )
            continue
        dec = decide_row(ctl, vals, cparams=int(control[0]["params"]), aparams=int(a.row["params"]))
        print(
            f"{a.name:>10s} {int(a.row['params']):13,d} {dec['mean']:9.4f} "
            f"{dec['spread']:8.4f} {dec['delta']:+8.4f} {dec['verdict']:>8s} "
            f"{dec['reason']:>40s}"
        )
        rows.append({"arm": a.name, "params": int(a.row["params"]), **dec})
    if not rows:
        return 1
    wins = [r for r in rows if r["verdict"] == "WIN"]
    ties = [r for r in rows if r["verdict"] == "TIE"]
    if wins:
        pick = min(wins, key=lambda r: (r["mean"], r["params"]))
    elif ties:
        pick = min(ties, key=lambda r: r["params"])
    else:
        pick = None
    print(
        f"\ndecision: {pick['arm'] if pick else 'dense control'} — "
        f"{(pick['verdict'] + ': ' + pick['reason']) if pick else 'no arm beats control'}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dry-run", action="store_true", help="resolve + table + rule, no card")
    ap.add_argument("--arms", default="", help="comma-separated subset (dense is the control)")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda", help="cuda|cpu (cpu for a shape smoke)")
    a = ap.parse_args(argv)
    if a.dry_run:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    arms = resolve_arms()
    if a.arms:
        keep = {x.strip() for x in a.arms.split(",") if x.strip()}
        unknown = keep - {x.name for x in arms}
        if unknown:
            raise LoudError(f"unknown arm(s) {sorted(unknown)}; known {[x.name for x in arms]}")
        if "dense" not in keep:
            raise LoudError("dense is the control; include it in --arms")
        arms = [x for x in arms if x.name in keep]
    plan(arms, a.out_root, a.device)
    if a.dry_run:
        print(f"\ndry-run: no card, no checkpoint; out_root {a.out_root} untouched.")
        return 0
    return report(arms, run(arms, a.out_root, a.device))


if __name__ == "__main__":
    raise SystemExit(main())
