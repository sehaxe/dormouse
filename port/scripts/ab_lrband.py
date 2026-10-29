"""The LR-band A/B: where the shipped recipe's ceiling is, and what moves it.

    python scripts/ab_lrband.py plan                 # dry run: resolve every (a) arm
    python scripts/ab_lrband.py profile --device cuda  # ms/step for (a), no table
    python scripts/ab_lrband.py run                  # (a), then (b) at the survivor

**THIS DRIVER HAS NOT BEEN RUN.** The card is held by the live flagship
(`AGENTS §1.5`: one heavy thing at a time), so every number it would produce is
unmeasured; `plan` is the only mode this branch executes.

WHY. The measured band on the 600-step battery was 5e-4 stable, 1e-3 a knife
edge (no LayerRoPE: 3.2400 / 3.8238 = spread 0.584) and 3e-3 the wall
(4.3573 / 9.6403). The owner wants the architecture to tolerate a much higher
LR. This driver measures the ceiling on the SHIPPED recipe first, then asks
which of four sourced stabilizers widens it.

RECIPE. `medium` (d=1024, 141.9M), the 600-step battery protocol (b8 x s512,
grad_accum 1), `mix4/web.bin` train and `web_eval.bin` held out (20 x 8 x 512 =
81 920 B on every arm), one JSONL per run under `~/logs/ab_lrband/`, the run's
own `<name>.config.toml` beside it. Model flags are the flagship launcher's
(`~/logs/launch_flagship.sh`): `max_iter=3`, CSA (chunk 8, coarse 256) + mHC +
AttnRes, `share_u`, `attn_latent_rank=256`, MSA off, Engram in VRAM, triton MoE
dropless, LayerRoPE paper form. `opt_graph=false`: LayerRoPE is refused with
both graph arms, so running it on one side would compare two step programs.

(a) LR CEILING. `lr in {1e-3, 2e-3, 3e-3} x warmup_frac in {0.02, 0.08}`,
3 seeds x 600 steps. `warmup_frac` is a fraction of the run's token budget
(`warmup_tokens = frac x steps x batch x seq_len`: 49 152 / 196 608 tokens =
12 / 48 warmup steps), the unit the trainer's WSD schedule already uses.

(b) STABILIZERS, at the highest LR that survives (a), 3 seeds x 600 steps:
`+z_loss 1e-3` (ModelConfig.z_loss), `+adam_eps 1e-6` (AdamW's denominator
floor; ~18.4 -> ~13.8 nats against the fp32 softmax-collapse cliff at 16.6),
`+agc_clip 0.01` (arXiv:2102.06171 AGC, per-tensor `λ‖p‖/max(‖g‖,1e-3)`),
`+softcap 15` (Gemma-2 final-logit cap, every margin <= 30 nats). The control
is the surviving (a) arm itself: (b) reuses its 3 seeds rather than re-running
the same config.

PRE-REGISTERED VERDICT RULES (written BEFORE the first run):

* VOID: an arm with any NaN-skipped step, a missing/short eval window
  (`scored != 81 920 B`), or a non-finite BPB in any seed is VOID — a divergence
  in any seed voids the arm, full stop. It is not a "noisy" number.
* (a) CEILING: an `(lr, warmup_frac)` ARM is VOID under the rule above. The
  highest `lr` with at least one FINITE arm survives; among the finite arms at
  that `lr`, the one with the lower mean BPB is (b)'s control (tie: the smaller
  fraction, 0.02). A warmup fraction that diverges does not delete the other
  fraction's arm at the same LR — it is its own arm.
* (b) WIN: an arm earns its place iff it is finite AND
  `mean_bpb(control) - mean_bpb(arm) > spread(control)` on the same 3 seeds,
  where `spread = max - min` of the control's own seeds. A delta inside the
  control's spread is a TIE and keeps the control's behaviour; a negative delta
  is a LOSS. Nothing here is promoted on one seed or on a mean alone.
* `ms/step` is reported, never folded into the BPB rule.

The profile mode reports `ms/step` per (a) arm from the JSONL's own warm-step
`tokens_per_s` (`log_every=1`, eval off, no verdict) — it answers "what does
this arm cost", not "what does it score".
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

# `scripts/` is not a package: the shared lane guard is a sibling (no torch
# CUDA call in the parent, which is what a forked child inherits).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import data_missing, preflight, skip  # noqa: E402

DATA = "/home/sehaxe/dormouse/data_local/mix4/web.bin"
EVAL = "/home/sehaxe/dormouse/data_local/mix4/web_eval.bin"
PRESET = "medium"
SEEDS = (1, 2, 3)
STEPS = 600
BATCH = 8
SEQ = 512
EVAL_BATCHES = 20
EVAL_EVERY = 300
LOG_EVERY = 100
WINDOW = EVAL_BATCHES * BATCH * SEQ
TOTAL_TOKENS = STEPS * BATCH * SEQ

LRS = (1e-3, 2e-3, 3e-3)
WARMUP_FRACS = (0.02, 0.08)
LR_LABEL = {1e-3: "1e-3", 2e-3: "2e-3", 3e-3: "3e-3"}

#: The flagship launcher's model flags, minus `doc_masking` (train-side) and
#: minus the graph arms (LayerRoPE refuses both).
COMMON_MODEL = dict(
    max_iter=3,
    use_msa=False,
    use_csa=True,
    csa_chunk=8,
    csa_coarse_chunk=256,
    use_mhc=True,
    use_attnres=True,
    share_u=True,
    use_engram=True,
    attn_latent_rank=256,
    moe_kernel="triton",
    moe_capacity=0.0,
    bf16=True,
    layerrope=True,
    layerrope_form="paper",
    layerrope_index="iteration",
    layerrope_init="paper",
    rank=64,
    n_experts=256,
)

COMMON_TRAIN = dict(
    steps=STEPS,
    batch=BATCH,
    seq_len=SEQ,
    eval=True,
    eval_every=EVAL_EVERY,
    eval_batches=EVAL_BATCHES,
    log_every=LOG_EVERY,
    ckpt_every=10**9,  # an A/B scores BPB; no weights are kept
    tf32=True,
    ns_kind="triton",
    opt_graph=False,  # refused with LayerRoPE; both sides stay eager
    engram_ram=False,  # the flagship's VRAM choice
    decay_frac=0.15,
    mix_plan="",
    doc_masking=False,
)

#: name -> (model overrides, train overrides). `control` is not re-run in (b):
#: the surviving (a) arm IS the control (same config, same seeds).
STABILIZERS: dict[str, tuple[dict, dict]] = {
    "z_loss": ({"z_loss": 1e-3}, {}),
    "adam_eps": ({}, {"adam_eps": 1e-6}),
    "agc_clip": ({}, {"agc_clip": 0.01}),
    "softcap": ({"logit_softcap": 15.0}, {}),
}


def warmup_tokens(frac: float) -> int:
    """`frac` of the run's token budget, in the WSD schedule's own unit."""
    return int(round(frac * TOTAL_TOKENS))


def _arm_a(lr: float, frac: float) -> dict:
    return {
        "name": f"a_lr{LR_LABEL[lr]}_w{int(round(frac * 100)):02d}",
        "lr": lr,
        "warmup_frac": frac,
        "model": dict(COMMON_MODEL),
        "train": {**COMMON_TRAIN, "warmup_tokens": warmup_tokens(frac)},
    }


def _arm_b(lr: float, frac: float, stab: str) -> dict:
    model, train = STABILIZERS[stab]
    return {
        "name": f"b_{stab}_lr{LR_LABEL[lr]}",
        "lr": lr,
        "warmup_frac": frac,
        "model": {**COMMON_MODEL, **model},
        "train": {**COMMON_TRAIN, "warmup_tokens": warmup_tokens(frac), **train},
    }


def arms_a() -> list[dict]:
    return [_arm_a(lr, frac) for lr in LRS for frac in WARMUP_FRACS]


def _run_config(
    arm: dict, seed: int, out: Path, *, steps: int, batch: int, seq: int, data: str, eval_file: str
):
    """Resolve ONE arm-seed through the real `load_run` seam (validation included)."""
    from dormouse.config import load_run

    return load_run(
        PRESET,
        model_overrides=arm["model"],
        train_overrides={
            **arm["train"],
            "steps": steps,
            "batch": batch,
            "seq_len": seq,
            "seed": seed,
            "lr": arm["lr"],
            "out_dir": str(out.parent),
            "ckpt_name": arm["name"],
            "json_log": str(out),
            "data": data,
            "eval_data": eval_file,
        },
    )


def plan(args: argparse.Namespace) -> int:
    """Dry run: every (a) arm resolves through `load_run`; print its shape."""
    print(
        f"plan (a): {len(arms_a())} arms x {len(SEEDS)} seeds, {STEPS} steps, "
        f"{PRESET} b{BATCH} s{SEQ}, window {WINDOW} B\n"
        f"           train={DATA}\n           eval ={EVAL}"
    )
    for arm in arms_a():
        out = Path(args.out_dir).expanduser() / f"{arm['name']}_s{SEEDS[0]}.jsonl"
        run = _run_config(
            arm, SEEDS[0], out, steps=STEPS, batch=BATCH, seq=SEQ, data=DATA, eval_file=EVAL
        )
        t, m = run.train, run.model
        print(
            f"  {arm['name']:>16s} lr={arm['lr']:.0e} warmup={t.warmup_tokens} tok "
            f"({arm['warmup_frac']:.2f}) max_iter={m.max_iter} "
            f"csa={m.use_csa} mhc={m.use_mhc} attnres={m.use_attnres} "
            f"share_u={m.share_u} layerrope={m.layerrope}/{m.layerrope_form} "
            f"opt_graph={t.opt_graph} data={Path(t.data).name}"
        )
    print(
        "plan (b): gated on (a) — the stabilizers run at the highest LR that keeps "
        "a finite arm (3 seeds); `run` enforces that rule. Nothing was trained."
    )
    return 0


def _rows(out_dir: Path, name: str, steps: int) -> list[dict]:
    rows = []
    for seed in SEEDS:
        path = out_dir / f"{name}_s{seed}.jsonl"
        if not path.is_file():
            continue
        evals = []
        logs = []
        run_row = None
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("event") == "run":
                run_row = r
            elif r.get("event") == "eval":
                evals.append(r)
            elif r.get("event") == "log":
                logs.append(r)
        rates = [
            r["tokens_per_s"]
            for r in logs
            if r.get("step", 0) > steps / 2 and r.get("tokens_per_s")
        ]
        rows.append(
            {
                "seed": seed,
                "bpb": evals[-1]["bpb"] if evals else float("nan"),
                "scored": evals[-1]["scored"] if evals else 0,
                "nan": max((r.get("nan_skipped", 0) for r in logs), default=0),
                "ms": ((BATCH * SEQ / statistics.median(rates) * 1e3) if rates else float("nan")),
                "run": run_row,
            }
        )
    return rows


def _finite(rows: list[dict]) -> bool:
    return len(rows) == len(SEEDS) and all(
        math.isfinite(r["bpb"]) and r["scored"] == WINDOW and r["nan"] == 0 for r in rows
    )


def _mean(rows: list[dict]) -> float:
    return statistics.fmean(r["bpb"] for r in rows)


def _spread(rows: list[dict]) -> float:
    return max(r["bpb"] for r in rows) - min(r["bpb"] for r in rows)


def _finite_arms(out_dir: Path, arms: list[dict], steps: int) -> dict[str, list[dict]]:
    return {
        arm["name"]: _rows(out_dir, arm["name"], steps)
        for arm in arms
        if _finite(_rows(out_dir, arm["name"], steps))
    }


def select_ceiling(out_dir: Path, steps: int = STEPS) -> tuple[float, float] | None:
    """The pre-registered (a) rule: highest LR with at least one FINITE arm.

    An arm is `(lr, warmup_frac)`: a divergence in any seed VOIDS that arm. The
    highest LR that still has a finite arm survives; among the finite arms at
    that LR the lower mean BPB wins, a tie going to the smaller fraction (0.02).
    """
    finite = _finite_arms(out_dir, arms_a(), steps)
    if not finite:
        return None
    lrs = [arm["lr"] for arm in arms_a() if arm["name"] in finite]
    top = max(lrs)
    candidates = [arm for arm in arms_a() if arm["lr"] == top and arm["name"] in finite]
    best = min(candidates, key=lambda a: (_mean(finite[a["name"]]), a["warmup_frac"]))
    return top, best["warmup_frac"]


def _run_one(args: argparse.Namespace) -> int:
    """One arm-seed in its own process (the child mode)."""
    import torch

    from dormouse.model import DormouseModel
    from dormouse.train import train

    if args.suite == "a":
        arm = _arm_a(args.lr, args.warmup_frac)
    else:
        arm = _arm_b(args.lr, args.warmup_frac, args.stab)
    out = Path(args.out)
    run = _run_config(
        arm,
        args.seed,
        out,
        steps=args.steps,
        batch=args.batch,
        seq=args.seq,
        data=args.data,
        eval_file=args.eval,
    )
    if not Path(args.data).is_file():
        print(f"no data file at {args.data}: pass --data")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".config.toml").write_text(run.snapshot_toml())

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    model = DormouseModel(run.model, device=torch.device(device))
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[run] arm={arm['name']} seed={args.seed} steps={args.steps} "
        f"b{args.batch} s{args.seq} lr={args.lr:.0e} "
        f"window={EVAL_BATCHES * args.batch * args.seq} B params={total:,}",
        flush=True,
    )
    train(model, run, device=torch.device(device), log=print)
    with open(out, "a", encoding="utf-8") as sink:
        sink.write(json.dumps({"event": "params", "step": args.steps, "params": total}) + "\n")
    return 0


def report(out_dir: Path, steps: int = STEPS) -> int:
    """Print (a), select the ceiling, then (b) if its runs exist."""
    print(f"\n{'arm':>18s} {'seed':>4s} {'bpb':>8s} {'scored':>7s} {'nan':>4s} {'ms/step':>9s}")
    for arm in arms_a():
        for r in _rows(out_dir, arm["name"], steps):
            print(
                f"{arm['name']:>18s} {r['seed']:>4d} {r['bpb']:8.4f} "
                f"{r['scored']:>7d} {r['nan']:>4d} {r['ms']:9.1f}"
            )

    print("\n(a) LR ceiling (arm rule: VOID on any divergent seed):")
    for lr in LRS:
        cells = []
        for frac in WARMUP_FRACS:
            arm = _arm_a(lr, frac)
            rows = _rows(out_dir, arm["name"], steps)
            if _finite(rows):
                cell = f"FINITE {_mean(rows):.4f} +- {_spread(rows):.4f}"
            else:
                cell = "VOID"
            cells.append(f"w{int(round(frac * 100)):02d}={cell}")
        print(f"  lr={lr:.0e}  " + "  ".join(cells))

    chosen = select_ceiling(out_dir, steps)
    if chosen is None:
        print("\nno arm finished: nothing to select (no (b) runs).")
        return 0
    lr, frac = chosen
    control = _arm_a(lr, frac)
    crow = _rows(out_dir, control["name"], steps)
    print(
        f"\nselected: lr={lr:.0e} warmup_frac={frac:.2f} "
        f"control mean={_mean(crow):.4f} spread={_spread(crow):.4f}"
    )

    b_arms = [_arm_b(lr, frac, stab) for stab in STABILIZERS]
    present = [arm for arm in b_arms if (out_dir / f"{arm['name']}_s{SEEDS[0]}.jsonl").is_file()]
    if not present:
        print("(b): not run (the stabilizers are gated on a measured (a) ceiling).")
        return 0

    print(f"\n(b) stabilizers at lr={lr:.0e}, control reused from (a):")
    print(f"{'arm':>18s} {'mean':>8s} {'spread':>8s} {'delta':>8s} {'verdict':>10s}")
    for arm in present:
        rows = _rows(out_dir, arm["name"], steps)
        if not _finite(rows):
            print(f"{arm['name']:>18s} {'VOID':>8s} (divergence/short window in a seed)")
            continue
        delta = _mean(crow) - _mean(rows)
        verdict = "WIN" if delta > _spread(crow) else ("LOSS" if delta < 0 else "TIE")
        print(
            f"{arm['name']:>18s} {_mean(rows):8.4f} {_spread(rows):8.4f} "
            f"{delta:+8.4f} {verdict:>10s}"
        )
    print(
        f"\nwin rule: delta > control spread ({_spread(crow):.4f}); "
        f"a divergence in ANY seed voids the arm."
    )
    return 0


def profile(args: argparse.Namespace) -> int:
    """`ms/step` for the (a) arms: same trainer, eval off, no table/verdict."""
    import torch

    from dormouse.model import DormouseModel
    from dormouse.train import train

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    out_dir = Path(args.out_dir).expanduser() / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"profile (a): {PRESET} b{BATCH} s{SEQ}, {args.profile_steps} steps per arm")
    for arm in arms_a():
        out = out_dir / f"{arm['name']}.jsonl"
        out.unlink(missing_ok=True)
        run = _run_config(
            arm,
            SEEDS[0],
            out,
            steps=args.profile_steps,
            batch=BATCH,
            seq=SEQ,
            data=args.data,
            eval_file=args.eval,
        )
        run.train.eval = False
        run.train.log_every = 1
        run.train.timers = False
        if not Path(args.data).is_file():
            print(f"no data file at {args.data}: pass --data")
            return 1
        model = DormouseModel(run.model, device=torch.device(device))
        print(f"\n=== profile {arm['name']} ({args.profile_steps} steps, {device})", flush=True)
        train(model, run, device=torch.device(device), log=print)
        rates = [
            r["tokens_per_s"]
            for r in (json.loads(line) for line in out.read_text().splitlines() if line.strip())
            if r.get("event") == "log"
            and r.get("step", 0) > args.profile_steps / 2
            and r.get("tokens_per_s")
        ]
        assert rates, f"{out}: no warm-step rate rows — nothing measured"
        ms = BATCH * SEQ / statistics.median(rates) * 1e3
        print(f"{arm['name']:>18s} lr={arm['lr']:.0e} {ms:8.1f} ms/step")
    return 0


def run(args: argparse.Namespace) -> int:
    """Preflight, run (a), select the ceiling, run (b), print the verdict."""
    out_dir = Path(args.out_dir).expanduser()
    data = args.data
    why = preflight(args.device) or data_missing(data, args.eval)
    if why and getattr(args, "force_busy", False):
        import subprocess as _sp

        holders = _sp.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        print(f"[force-busy] ignoring preflight: {why}\n[force-busy] holders:\n{holders}")
        why = None
    if why is not None:
        return skip(why)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else SEEDS
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device

    def launch(suite: str, lr: float, frac: float, stab: str, seed: int) -> int:
        arm = _arm_a(lr, frac) if suite == "a" else _arm_b(lr, frac, stab)
        out = out_dir / f"{arm['name']}_s{seed}.jsonl"
        out.unlink(missing_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_run",
            "--suite",
            suite,
            "--lr",
            repr(lr),
            "--warmup-frac",
            repr(frac),
            "--stab",
            stab,
            "--seed",
            str(seed),
            "--out",
            str(out),
            "--data",
            data,
            "--eval",
            args.eval,
            "--device",
            device.split(":")[-1] if ":" in device else device,
            "--steps",
            str(args.steps),
        ]
        print(f"\n=== {arm['name']} seed {seed} -> {out}", flush=True)
        return subprocess.run(cmd, check=False).returncode

    for arm in arms_a():
        for seed in seeds:
            rc = launch("a", arm["lr"], arm["warmup_frac"], "-", seed)
            if rc != 0:
                return rc

    chosen = select_ceiling(out_dir, args.steps)
    if chosen is None:
        print("\nno (a) arm finished: (b) is not run (a ceiling needs a finite LR).")
        return report(out_dir, args.steps)
    lr, frac = chosen
    print(f"\n(a) ceiling: lr={lr:.0e}, warmup_frac={frac:.2f} -> running (b)")
    for stab in STABILIZERS:
        for seed in seeds:
            rc = launch("b", lr, frac, stab, seed)
            if rc != 0:
                return rc
    return report(out_dir, args.steps)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan", help="dry run: resolve every (a) arm through load_run")
    p.add_argument("--out-dir", default="~/logs/ab_lrband")
    pf = sub.add_parser("profile", help="ms/step for the (a) arms; no verdict")
    pf.add_argument("--profile-steps", type=int, default=20)
    pf.add_argument("--device", default="cuda")
    pf.add_argument("--data", default=DATA)
    pf.add_argument("--eval", default=EVAL)
    pf.add_argument("--out-dir", default="~/logs/ab_lrband")
    r = sub.add_parser("run", help="(a) then (b) at the surviving ceiling")
    r.add_argument("--seeds", default=None, help="override the seed list (never for a verdict)")
    r.add_argument("--steps", type=int, default=STEPS)
    r.add_argument("--device", default="cuda")
    r.add_argument("--data", default=DATA)
    r.add_argument("--eval", default=EVAL)
    r.add_argument("--out-dir", default="~/logs/ab_lrband")
    r.add_argument(
        "--force-busy",
        action="store_true",
        help="proceed although another process holds the card; timings are then "
        "meaningless (BPB is not), and the holders are printed first",
    )
    c = sub.add_parser("_run", help="one arm-seed (child mode)")
    c.add_argument("--suite", choices=("a", "b"), required=True)
    c.add_argument("--lr", type=float, required=True)
    c.add_argument("--warmup-frac", type=float, default=WARMUP_FRACS[0])
    c.add_argument("--stab", default="-", choices=("-", *STABILIZERS))
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--data", default=DATA)
    c.add_argument("--eval", default=EVAL)
    c.add_argument("--device", default="cuda")
    c.add_argument("--steps", type=int, default=STEPS)
    c.add_argument("--batch", type=int, default=BATCH)
    c.add_argument("--seq", type=int, default=SEQ)
    args = ap.parse_args(argv)
    if args.mode == "plan":
        return plan(args)
    if args.mode == "profile":
        return profile(args)
    if args.mode == "_run":
        return _run_one(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
