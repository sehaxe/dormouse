"""The LayerRoPE A/B driver (`arXiv:2610.09179`), pre-registered — v2 (paper form).

    uv run python scripts/ab_layerrope.py plan          # dry run: resolve + print
    uv run python scripts/ab_layerrope.py run --seeds 1 # NOT run by this branch

**THIS DRIVER HAS NOT BEEN RUN.** No GPU work is allowed on this task, so every
number it would produce is unmeasured. `plan` is the only mode this branch
executes: it resolves every arm through the real `load_run` seam (validation
included) and prints the resolved model/train keys, so a config that cannot even
resolve is caught before a card is booked.

Arms (600 steps, b8 x s512, `medium`, data `data_local/mix4/web.bin`, held-out
`data_local/mix4/web_eval.bin`, 20 x 8 x 512 = 81 920 B eval window, 3 seeds,
one JSONL per run under `~/logs/ab_layerrope/`):

    no_layerrope_5e-4                              the CONTROL
    v1_5e-4  paper_5e-4  paper_identity_5e-4  paper_pos_5e-4
    v1_1e-3  paper_1e-3  paper_identity_1e-3  paper_pos_1e-3

`paper` = `layerrope_form="paper"`, `layerrope_index="iteration"` (the default);
`paper_pos` = the same with `index="position"` (the paper's looped-model depth
index); `paper_identity` = `init="identity"` (every depth's gain IS `gamma` —
the no-op control for the FORM, it must land on `no_layerrope`). The two LRs are
the band: the v1 A/B's recipe LR (1e-3) and the lower one (5e-4), because the
paper init is NOT an identity and a depth-scaled gain can want a different LR.

Every arm is the battery's COMMON with `opt_graph=false`: the flag is REFUSED
with `layerrope=true` (`config.validate_model_train`), so running it on the
control only would compare two different step programs. `mix_plan` is not
copied (the brief names one data file, and `mix_plan` replaces the single-file
stream).

**Pre-registered verdict rules** (written BEFORE the first run):

* **Form.** The paper form REPLACES v1 at an LR only if
  `mean_bpb(v1) - mean_bpb(paper) > 2 x spread(control)`, where `spread` is the
  max-min of the control's own 3 seeds AT 5e-4. Two LRs are reported; the
  replacement needs the rule at ONE LR and must NOT be worse than v1 by more
  than `2 x spread` at the other. Anything else is a TIE, and a tie keeps v1
  (the arm that already ships) — never the newer form.
* **Depth index.** `paper_pos` vs `paper` (iteration) is decided by the SAME
  rule and the same spread: `|mean(pos) - mean(iter)| > 2 x spread` decides,
  otherwise TIE, and a tie keeps `"iteration"` (v1's rule, the smaller change).
* **LR band width.** For each form, an LR is IN the band iff the arm is finite
  (no NaN-skipped step, every eval row present) and its mean BPB is within
  **2x the best arm's mean** at that LR. `width` = how many LRs are in the band
  (`0..2`). Reported per form; it is not part of the replacement rule.
* A NaN or a short eval window anywhere makes that arm unusable and the
  comparison is reported as UNDECIDED, not as a win.

**Protocol.** 3 seeds per arm, 600 steps, each arm-seed a SEPARATE PROCESS, one
eval window on every run, the run's own `<name>.config.toml` beside its JSONL
(a number needs its config). `ms/step` is reported, never folded into the BPB
rule.
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
SEEDS = (1, 2, 3)
LRS = (5e-4, 1e-3)
LR_LABEL = {5e-4: "5e-4", 1e-3: "1e-3"}
#: The spread that the replacement rule divides by (the control's own seeds).
SPREAD_LR = 5e-4

#: The battery's COMMON minus `opt_graph` (refused with the arm) and minus
#: `mix_plan` (the brief names one file). Every arm carries both deviations, so
#: the only per-arm difference is the LR / the form flags.
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

#: form name -> the model overrides that express it. `None` is the control.
FORMS = {
    "no_layerrope": {"layerrope": False},
    "v1": {"layerrope": True, "layerrope_form": "v1"},
    "paper": {"layerrope": True, "layerrope_form": "paper"},
    "paper_identity": {
        "layerrope": True,
        "layerrope_form": "paper",
        "layerrope_init": "identity",
    },
    "paper_pos": {
        "layerrope": True,
        "layerrope_form": "paper",
        "layerrope_index": "position",
    },
}


def _arm(name: str, form: str, lr: float) -> dict:
    return {"name": name, "form": form, "model": {**COMMON_MODEL, **FORMS[form]}, "lr": lr}


ARMS = [
    _arm("no_layerrope_5e-4", "no_layerrope", 5e-4),
    *[_arm(f"{form}_5e-4", form, 5e-4) for form in ("v1", "paper", "paper_identity", "paper_pos")],
    *[_arm(f"{form}_1e-3", form, 1e-3) for form in ("v1", "paper", "paper_identity", "paper_pos")],
]
BY_NAME = {a["name"]: a for a in ARMS}
#: The five names a comparison is made from, per LR.
FORM_NAMES = ("no_layerrope", "v1", "paper", "paper_identity", "paper_pos")


def _name(form: str, lr: float) -> str:
    return f"{form}_{LR_LABEL[lr]}"


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
        f"plan: {len(ARMS)} arms x {len(SEEDS)} seeds, {STEPS} steps, {PRESET} "
        f"b{BATCH} s{SEQ}, window {WINDOW} B, data {DATA}"
    )
    for arm in ARMS:
        out = Path(args.out_dir).expanduser() / f"{arm['name']}_s{SEEDS[0]}.jsonl"
        run = _run_config(arm, SEEDS[0], out)
        m = run.model
        n_depths = m.prelude_layers + m.max_iter + m.coda_layers + 1
        form = m.layerrope_form if m.layerrope else "-"
        gain = (
            f"4*(d+4)={4 * (m.d_model + 4)}/block"
            if m.layerrope and m.layerrope_form == "paper"
            else (f"d+2*n_depths={m.d_model + 2 * n_depths}" if m.layerrope else "-")
        )
        print(
            f"  {arm['name']:>22s} lr={arm['lr']:.0e} form={form} "
            f"index={m.layerrope_index if m.layerrope else '-'} "
            f"init={m.layerrope_init if m.layerrope else '-'} "
            f"gain_params={gain} opt_graph={run.train.opt_graph} "
            f"steps={run.train.steps} batch={run.train.batch} seq_len={run.train.seq_len}"
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


def lr_band(means: dict[tuple[str, float], float], form: str, lrs=LRS) -> set[float]:
    """The pre-registered band: mean BPB <= 2x the best arm's mean AT THAT LR."""
    band = set()
    for lr in lrs:
        arms = {f: means[(f, lr)] for f in FORM_NAMES if (f, lr) in means}
        if form not in arms or not arms:
            continue
        if arms[form] <= 2.0 * min(arms.values()):
            band.add(lr)
    return band


def report(out_dir: Path) -> int:
    """Print the table and apply the pre-registered rules (never run here)."""
    table = {arm["name"]: _rows(out_dir, arm["name"], SEEDS, STEPS) for arm in ARMS}
    print(f"\n{'arm':>22s} {'seed':>4s} {'bpb':>8s} {'scored':>7s} {'nan':>4s} {'params':>10s}")
    for name, rows in table.items():
        for r in rows:
            print(
                f"{name:>22s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>7d} "
                f"{r['nan']:>4d} {r.get('params', '-'):>10}"
            )
    means = {
        (arm["form"], arm["lr"]): _mean(table[arm["name"]])
        for arm in ARMS
        if _finite(table[arm["name"]])
    }
    control = table[_name("no_layerrope", SPREAD_LR)]
    spread = _spread(control)
    usable = _finite(control)
    print(f"\ncontrol: mean={_mean(control):.4f} spread(3 seeds)={spread:.4f} finite={usable}")
    print("LR band widths: " + " ".join(f"{f}={len(lr_band(means, f))}/{len(LRS)}" for f in FORMS))

    def decider(a: str, b: str, lr: float) -> str:
        """`b` replaces `a` at `lr` iff it wins by > 2x the control spread."""
        ka, kb = (a, lr), (b, lr)
        if ka not in means or kb not in means:
            return "UNDECIDED (a run is missing or not finite)"
        delta = means[ka] - means[kb]  # > 0: b is better (lower BPB)
        if delta > 2.0 * spread:
            return f"{b} REPLACES {a} (delta {delta:+.4f} > 2*spread {2 * spread:.4f})"
        if delta < -2.0 * spread:
            return f"{a} KEEPS (delta {delta:+.4f}; {b} is worse by > 2*spread)"
        return f"TIE (delta {delta:+.4f} within 2*spread {2 * spread:.4f})"

    for lr in LRS:
        print(f"\nlr={lr:.0e}")
        for a, b in (("v1", "paper"), ("paper", "paper_pos"), ("no_layerrope", "paper_identity")):
            print(f"  {a} vs {b}: {decider(a, b, lr)}")
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
        seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else SEEDS
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
