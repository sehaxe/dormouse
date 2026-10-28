"""The LayerRoPE A/B driver (`arXiv:2610.09179`), pre-registered.

    uv run python scripts/ab_layerrope.py plan          # dry run: resolve + print
    uv run python scripts/ab_layerrope.py run --seeds 1 # NOT run by this branch

**THIS DRIVER HAS NOT BEEN RUN.** No GPU work is allowed on this task, so every
number it would produce is unmeasured. `plan` is the only mode this branch
executes: it resolves every arm through the real `load_run` seam (validation
included) and prints the resolved model/train keys, so a config that cannot even
resolve is caught before a card is booked.

Arms (600 steps, b8 x s512, `medium`, data `data_local/mix4/web.bin`, held-out
`data_local/mix4/web_eval.bin`, 20 x 8 x 512 = 81 920 B eval window, one JSONL
per run under `~/logs/ab_layerrope/`):

1. `off_lr{5e-4,1e-3,3e-3}` x 2 seeds — the LR band WITHOUT LayerRoPE.
2. `on_lr{5e-4,1e-3,3e-3}` x 2 seeds — the LR band WITH LayerRoPE.
3. `off` / `on` at the recipe LR (1e-3) x 3 seeds — the quality comparison.
4. `off_mhc0`, `off_mhc1`, `on_mhc0`, `on_mhc1` x 3 seeds — does the arm add
   anything on top of mHC?

Every arm is the battery's COMMON with `opt_graph=false`: the flag is REFUSED
with `layerrope=true` (`config.validate_model_train`), so running it on the OFF
arms only would compare two different step programs. `mix_plan` is not copied
(the brief names one data file, and `mix_plan` replaces the single-file stream).

**Pre-registered verdict rule** (written BEFORE the first run):

* **LR band.** An LR is in the band iff the arm is finite (no NaN-skipped step,
  every eval row present) and its mean held-out BPB is within **2x the best
  arm's loss**. LayerRoPE "widens the band" ONLY if the ON set is STRICTLY
  larger than the OFF set. A tie is not a widening.
* **Quality.** LayerRoPE enters the recipe only if `on`'s mean BPB is within
  **2x the control's 3-seed spread** of `off` (i.e. `|on - off| <= 2*spread`).
  A tie is acceptable ONLY if the LR band widened or the parameter count
  dropped; otherwise the arm is DELETED with a dated verdict in the commit.
* **mHC interaction.** It is a **TIE against mHC** if `on_mhc1` is not better
  than `off_mhc1` by more than the `off_mhc1` 3-seed spread. A tie is still
  worth keeping ONLY for the depth-scaling claim (the paper's mechanism), and
  the report must say exactly that.

**Protocol.** 3 seeds per quality/mHC arm, 2 per LR-sweep arm (the band is a
coarse set, not a quality estimate), 600 steps, each arm-seed a SEPARATE
PROCESS, one eval window on every run, the run's own `<name>.config.toml` beside
its JSONL (a number needs its config). `ms/step` is reported, never folded into
the BPB rule.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.append(_SCRIPTS)
from ab_byteflow import preflight, read_run, skip  # noqa: E402

DATA = "/home/sehaxe/dormouse/data_local/mix4/web.bin"
EVAL = "/home/sehaxe/dormouse/data_local/mix4/web_eval.bin"

STEPS = 600
PRESET = "medium"
BATCH = 8
SEQ = 512
EVAL_BATCHES = 20
EVAL_EVERY = 300
LOG_EVERY = 100
WINDOW = EVAL_BATCHES * BATCH * SEQ
RECIPE_LR = 1e-3
LRS = (5e-4, RECIPE_LR, 3e-3)
LR_LABEL = {5e-4: "5e-4", RECIPE_LR: "1e-3", 3e-3: "3e-3"}

#: The battery's COMMON minus `opt_graph` (refused with the arm) and minus
#: `mix_plan` (the brief names one file). Every arm carries both deviations, so
#: the only per-arm difference is the LR / the mechanism flags.
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
    opt_graph=False,
    engram_ram=True,
)
COMMON_MODEL = dict(
    moe_kernel="triton",
    moe_capacity=0.0,
    bf16=True,
    use_attnres=False,
    loom=False,
)


def _arm(name, *, layerrope, lr, seeds, mhc=False):
    model = {**COMMON_MODEL, "layerrope": layerrope}
    if mhc:
        model["use_mhc"] = True
    return {"name": name, "model": model, "lr": lr, "seeds": seeds, "mhc": mhc}


ARMS = [
    *[_arm(f"off_lr{LR_LABEL[lr]}", layerrope=False, lr=lr, seeds=(1, 2)) for lr in LRS],
    *[_arm(f"on_lr{LR_LABEL[lr]}", layerrope=True, lr=lr, seeds=(1, 2)) for lr in LRS],
    _arm("off", layerrope=False, lr=RECIPE_LR, seeds=(1, 2, 3)),
    _arm("on", layerrope=True, lr=RECIPE_LR, seeds=(1, 2, 3)),
    _arm("off_mhc0", layerrope=False, lr=RECIPE_LR, seeds=(1, 2, 3)),
    _arm("off_mhc1", layerrope=False, lr=RECIPE_LR, seeds=(1, 2, 3), mhc=True),
    _arm("on_mhc0", layerrope=True, lr=RECIPE_LR, seeds=(1, 2, 3)),
    _arm("on_mhc1", layerrope=True, lr=RECIPE_LR, seeds=(1, 2, 3), mhc=True),
]
BY_NAME = {a["name"]: a for a in ARMS}


def _run_config(
    arm: dict,
    seed: int,
    out: Path,
    *,
    steps: int = STEPS,
    batch: int = BATCH,
    seq: int = SEQ,
    data: str = DATA,
    eval_file: str = EVAL,
):
    """Resolve ONE arm-seed through the real seam (validation included)."""
    from dormouse.config import load_run

    return load_run(
        PRESET,
        model_overrides=arm["model"],
        train_overrides={
            **COMMON_TRAIN,
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
    """Dry run: every arm resolves through `load_run`; print the resolved shape."""
    print(
        f"plan: {len(ARMS)} arms, {STEPS} steps, {PRESET} b{BATCH} s{SEQ}, "
        f"window {WINDOW} B, data {DATA}"
    )
    for arm in ARMS:
        out = Path(args.out_dir).expanduser() / f"{arm['name']}_s{arm['seeds'][0]}.jsonl"
        run = _run_config(arm, arm["seeds"][0], out)
        m = run.model
        n_depths = m.prelude_layers + m.max_iter + m.coda_layers + 1
        on = m.layerrope
        print(
            f"  {arm['name']:>10s} lr={arm['lr']:.0e} seeds={arm['seeds']} "
            f"layerrope={on} mhc={m.use_mhc} n_depths={n_depths} "
            f"gain_params={(m.d_model + 2 * n_depths) if on else '-'} "
            f"opt_graph={run.train.opt_graph} steps={run.train.steps} "
            f"batch={run.train.batch} seq_len={run.train.seq_len}"
        )
    print("plan: all arms resolve; nothing was trained.")
    return 0


def _run_one(args: argparse.Namespace) -> int:
    """One arm-seed in its own process (the child mode; not executed here)."""
    import torch

    from dormouse.model import DormouseModel
    from dormouse.train import train

    arm = BY_NAME[args.arm]
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
    assert (model.layerrope is not None) == arm["model"]["layerrope"], arm
    assert (model.loop.mhc is not None) == arm["mhc"], arm
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[run] arm={arm['name']} seed={args.seed} steps={args.steps} "
        f"b{args.batch} s{args.seq} lr={arm['lr']:.0e} window={WINDOW} B "
        f"params={total:,}",
        flush=True,
    )
    train(model, run, device=torch.device(device), log=print)
    with open(out, "a", encoding="utf-8") as sink:
        sink.write(json.dumps({"event": "params", "step": args.steps, "params": total}) + "\n")
    return 0


def _rows(out_dir: Path, arm: str, seeds, steps: int) -> list[dict]:
    rows = []
    for seed in seeds:
        path = out_dir / f"{arm}_s{seed}.jsonl"
        row = {"seed": seed, **read_run(path, steps)}
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("event") == "params":
                row["params"] = r.get("params")
        rows.append(row)
    return rows


def _mean(rows: list[dict]) -> float:
    return statistics.fmean(r["bpb"] for r in rows)


def _spread(rows: list[dict]) -> float:
    return max(r["bpb"] for r in rows) - min(r["bpb"] for r in rows)


def _finite(rows: list[dict]) -> bool:
    return all(math.isfinite(r["bpb"]) and r["scored"] == WINDOW and not r["nan"] for r in rows)


def lr_means(table: dict[str, list[dict]]) -> dict[tuple[str, float], float]:
    """Mean BPB per (side, LR), only for runs that pass the finite checks."""
    return {
        (side, lr): _mean(table[f"{side}_lr{LR_LABEL[lr]}"])
        for side in ("off", "on")
        for lr in LRS
        if _finite(table[f"{side}_lr{LR_LABEL[lr]}"])
    }


def lr_band(means: dict[tuple[str, float], float], side: str) -> set[float]:
    """The pre-registered band: mean BPB <= 2x the best arm's mean (both sides)."""
    if not means:
        return set()
    best = min(means.values())
    return {lr for (s, lr), mean in means.items() if s == side and mean <= 2.0 * best}


def report(out_dir: Path) -> int:
    """Print the table and apply the pre-registered rule (never run here)."""
    table = {arm["name"]: _rows(out_dir, arm["name"], arm["seeds"], STEPS) for arm in ARMS}
    print(f"\n{'arm':>10s} {'seed':>4s} {'bpb':>8s} {'scored':>7s} {'nan':>4s} {'params':>10s}")
    for name, rows in table.items():
        for r in rows:
            print(
                f"{name:>10s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>7d} "
                f"{r['nan']:>4d} {r.get('params', '-'):>10}"
            )
    means = lr_means(table)
    off_band, on_band = lr_band(means, "off"), lr_band(means, "on")
    print(f"\nLR band off={sorted(off_band)} on={sorted(on_band)}")
    widened = on_band > off_band
    off, on = table["off"], table["on"]
    delta = _mean(on) - _mean(off)
    spread = _spread(off)
    quality = _finite(off) and _finite(on) and abs(delta) <= 2.0 * spread
    params_drop = (on[0].get("params") or 0) < (off[0].get("params") or 0)
    print(
        f"quality: on-off={delta:+.4f} off_spread={spread:.4f} "
        f"within_2x_spread={quality} band_widened={widened} params_drop={params_drop}"
    )
    keep = quality and (widened or params_drop)
    print(f"verdict: {'KEEP' if keep else 'DELETE (dated verdict in the commit)'}")
    off_mhc, on_mhc = table["off_mhc1"], table["on_mhc1"]
    mhc_delta = _mean(on_mhc) - _mean(off_mhc)
    print(
        f"mHC: on_mhc1-off_mhc1={mhc_delta:+.4f} spread={_spread(off_mhc):.4f} "
        f"tie={'yes' if mhc_delta <= _spread(off_mhc) else 'no'}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan", help="dry run: resolve every arm through load_run")
    p.add_argument("--out-dir", default="~/logs/ab_layerrope")
    r = sub.add_parser("run", help="the training driver (NOT executed on this branch)")
    r.add_argument("--seeds", default=None, help="override the arm's seed list")
    r.add_argument("--steps", type=int, default=STEPS)
    r.add_argument("--batch", type=int, default=BATCH)
    r.add_argument("--seq", type=int, default=SEQ)
    r.add_argument("--data", default=DATA)
    r.add_argument("--eval", default=EVAL)
    r.add_argument("--device", default="cuda")
    r.add_argument("--out-dir", default="~/logs/ab_layerrope")
    c = sub.add_parser("_run", help="one arm-seed (child mode)")
    c.add_argument("--arm", required=True, choices=[a["name"] for a in ARMS])
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
    if args.mode == "_run":
        return _run_one(args)
    why = preflight(args.device)
    if why is not None:
        return skip(why)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else arm["seeds"]
        for seed in seeds:
            out = out_dir / f"{arm['name']}_s{seed}.jsonl"
            out.unlink(missing_ok=True)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_run",
                "--arm",
                arm["name"],
                "--seed",
                str(seed),
                "--out",
                str(out),
                "--data",
                args.data,
                "--eval",
                args.eval,
                "--device",
                args.device,
                "--steps",
                str(args.steps),
                "--batch",
                str(args.batch),
                "--seq",
                str(args.seq),
            ]
            print(f"\n=== {arm['name']} seed {seed} -> {out}", flush=True)
            done = subprocess.run(cmd, check=False)
            if done.returncode != 0:
                return done.returncode
    return report(out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
