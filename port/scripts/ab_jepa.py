"""The JEPA A/B: `off` (control) vs `jepa` (horizon 1) vs `jepa_h2` (horizon 2).

    uv run python scripts/ab_jepa.py ab              # 3 arms x 3 seeds
    uv run python scripts/ab_jepa.py ab --seeds 1    # one seed, dry check

**THIS DRIVER HAS NOT BEEN RUN.** The card is held by the benchmark chain
(AGENTS §1.5: one heavy thing at a time), so the verdict rule is written here
BEFORE the first run so that run cannot move the goalposts. A busy card makes the
driver REFUSE and exit 1; nothing is measured and no table prints.

**The mechanism.** JEPA is ONE term (`model/jepa.py`), not a predictor plus a
separate regulariser arm:

    loss = CE
         + jepa_weight · [ (1 − cos(pred(h[p]), sg(h[p+h])))
                            + jepa_reg_weight · SIGReg(pred(h[p])) ]

`h` is the loop's own readout from the SAME forward (no teacher, no second
forward, no mask, no EMA); the target is stop-gradient and, in a causal model,
a function of strictly more input than `h[p]`. `jepa_weight = 0` builds no head
and is bit-identical to the control. The internal SIGReg term is what keeps the
predicted latents from degenerating: the 200-step collapse gate in
`tests/test_jepa.py` measures std 0.061 at `jepa_reg_weight=0.1` vs 29 351 (a
diverged latent whose cosine loss goes to ~0) at 0.0, so **0.1 is the value this
driver runs** — not a knob to tune per arm.

**The arms** — the ONLY difference is the predictor's horizon (one model field;
the arm switch and both weights are identical):

* `off`   — `jepa=false, jepa_weight=0.0`: the pure-CE control.
* `jepa`  — horizon 1: predict `h[p+1]` from `h[p]`.
* `jepa_h2` — horizon 2: the same mechanism at a lag of two positions.

Both active arms run `jepa_weight=0.1` and `jepa_reg_weight=0.1`. The grid is
deliberately narrow: the question here is "does JEPA pay at all", not "which λ".

**The flags** are copied from `/tmp/opencode/battery.sh`'s `BASE` + `COMMON`
blocks, verbatim — `medium`, b8 x s512, 2 000 steps, `tf32`, `ns_kind=triton`,
`opt_graph`, `moe_kernel=triton`, `moe_capacity=0`, `engram_ram`, `bf16`,
`mix_plan=configs/mix_big.toml`, eval every 500 over `eval_batches=20` — so every
arm and seed scores the SAME window, `20 x 8 x 512 = 81 920 B` (AGENTS §2.6). The
only additions are the run's own `seed`/`jepa*`/`ckpt_*`/`json-log`. Each arm-seed
is a SEPARATE PROCESS launched as the CLI would be, and a `<arm>_s<seed>.config.toml`
is written beside its JSONL (`ckpt_every=10**6`, so the trainer writes none).

**The verdict** (AGENTS §1.2, pre-registered): per arm,
`Δ = mean_control − mean_arm` over the seeds that did NOT diverge (the MAD-based
flag below); **WIN iff `Δ > control_spread`**, where `control_spread =
max(control) − min(control)` over the control's kept seeds. Anything else is a
TIE, and a tie deletes the arm.

VOID (not a number) when an eval window is not 81 920 B, a NaN-skipped step
occurred, the control reports `counters["jepa"] != 0` (the term ran when the
weight was 0), or an active arm reports `counters["jepa"] == 0` (the arm never
ran — the research doc's metric 2). Diverged seeds are flagged and excluded from
the mean, never averaged in silently.

**What the driver does NOT do:** apply the verdict, tune λ/horizon per seed, or
read the training `loss=` as a quality number (it carries `jepa_weight · term` by
construction). Held-out BPB is the only comparison, and the eval forward runs
`pure=True`, so both sides are the predictive CE.

**Wall clock:** not measured (the card must be idle); 9 runs x 2 000 steps at
`medium` plus evals.
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

#: λ on the whole JEPA term. `0.1` is the burn-era `jepa_weight` scale and the
#: `medium`-compatible middle of the research doc's range; the A/B is not a λ
#: sweep, so one value on every active arm.
JEPA_WEIGHT = 0.1
#: The internal anti-collapse weight. Justified by `tests/test_jepa.py`'s
#: 200-step gate: at 0.1 the predicted latents stay in a bounded band
#: (per-dim std 0.061); at 0.0 they degenerate (29 351). Not tuned here.
JEPA_REG_WEIGHT = 0.1

#: `(arm, model overrides)`. `off` first: the table reads against it.
ARMS = (
    ("off", {"jepa": False, "jepa_weight": 0.0}),
    (
        "jepa",
        {
            "jepa": True,
            "jepa_weight": JEPA_WEIGHT,
            "jepa_horizon": 1,
            "jepa_reg_weight": JEPA_REG_WEIGHT,
        },
    ),
    (
        "jepa_h2",
        {
            "jepa": True,
            "jepa_weight": JEPA_WEIGHT,
            "jepa_horizon": 2,
            "jepa_reg_weight": JEPA_REG_WEIGHT,
        },
    ),
)

#: `/tmp/opencode/battery.sh`'s `BASE`: the device and the two data files.
DATA = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"
EVAL_DATA = "/home/sehaxe/dormouse/data_local/eval_mix/eval_mix.bin"
BASE = ("--device", "cuda", "--data", DATA, "--eval", EVAL_DATA)
#: The battery's `COMMON`, verbatim. Do not tune one arm: the horizon is the
#: only difference between `jepa` and `jepa_h2`.
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

#: The one window every run must score. A different number is a VOID arm.
WINDOW = EVAL_BATCHES * BATCH * SEQ
#: Below this a difference is noise, not a divergence signal.
SIGMA_FLOOR = 2e-3


def refuse(why: str) -> int:
    """A LOUD refusal with no table: a busy card measures nothing."""
    print(
        f"\n=== REFUSED: {why}\n"
        f"=== ONE GPU PROCESS AT A TIME (AGENTS §1.5): nothing was measured and no\n"
        f"=== table is printed. Re-run when the card is free.\n",
        flush=True,
    )
    return 1


# --- the verdict statistic (pure; gated by tests/test_jepa.py) --------------


def diverged(bpbs: list[float]) -> list[bool]:
    """One flag per seed: `|b − median| > 3σ`, σ the MAD estimate.

    Same estimator as `ab_sf_muon.diverged`: `1.4826·MAD`, floored at `2e-3`
    (the lower end of the paired-eval resolution). With three seeds the sample
    σ is inflated by the outlier it is meant to catch; MAD is not.
    """
    assert len(bpbs) >= 2, "an arm needs at least two seeds for a sigma"
    med = statistics.median(bpbs)
    mad = statistics.median([abs(b - med) for b in bpbs])
    sigma = max(1.4826 * mad, SIGMA_FLOOR)
    return [abs(b - med) > 3.0 * sigma for b in bpbs]


def kept(bpbs: list[float]) -> list[float]:
    """The non-diverged seeds of one arm."""
    return [b for b, bad in zip(bpbs, diverged(bpbs), strict=True) if not bad]


def verdict(control: list[float], arm: list[float]) -> dict:
    """AGENTS §1.2 on the trimmed sets: win iff Δ beats the control's spread."""
    if len(control) < 2 or len(arm) < 2:
        return {
            "verdict": "void",
            "delta": float("nan"),
            "spread": float("nan"),
            "why": "fewer than two kept seeds on one side",
        }
    c_mean = statistics.mean(control)
    a_mean = statistics.mean(arm)
    spread = max(control) - min(control)
    delta = c_mean - a_mean
    return {
        "verdict": "win" if delta > spread else "tie",
        "delta": delta,
        "spread": spread,
        "why": (
            "delta beats the control's own seed spread"
            if delta > spread
            else "inside the control's spread — a tie deletes the arm"
        ),
    }


# --- reading one run ---------------------------------------------------------


def read_run(jsonl: Path) -> dict:
    """One run's JSONL -> the numbers the verdict needs, plus the void evidence.

    `jepa` is the max `counters["jepa"]` over the log rows: the window counters
    reset at each log cadence, so a non-zero ANYWHERE is the evidence the term
    was added at least once (the control must read 0).
    """
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    evals = [r for r in rows if r.get("event") == "eval"]
    logs = [r for r in rows if r.get("event") == "log"]
    assert evals, f"{jsonl}: no eval row — nothing to compare"
    warm = [r["tokens_per_s"] for r in logs if r.get("tokens_per_s")]
    return {
        "bpb": evals[-1]["bpb"],
        "scored": evals[-1]["scored"],
        "tok_s": statistics.median(warm) if warm else float("nan"),
        # `nan_skipped` is CUMULATIVE, so the max over rows is the run total.
        "nan": max((r.get("nan_skipped", 0) for r in logs), default=0),
        "jepa": max((r.get("counters", {}).get("jepa", 0) for r in logs), default=0),
    }


# --- launching one run -------------------------------------------------------


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


def _split_common() -> tuple[dict, dict]:
    """COMMON's `--set k=v` pairs -> (model overrides, train overrides)."""
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


def _snapshot(arm: str, model_over: dict, seed: int, out_dir: Path, steps: int) -> None:
    """Write the run's own `<name>.config.toml` beside its JSONL.

    The trainer writes no checkpoint (`ckpt_every=10**6`), hence no snapshot of
    its own; without this the number would have no config record (AGENTS §1.4).
    The overrides are resolved from the SAME COMMON tuple the child is launched
    with, so the record cannot drift from the run.
    """
    from dormouse.config import load_run

    model_common, train_common = _split_common()
    run = load_run(
        PRESET,
        model_overrides={**model_common, **model_over},
        train_overrides={
            **train_common,
            "steps": steps,
            "seed": seed,
            "ckpt_name": f"jepa_{arm}_s{seed}",
            "out_dir": str(out_dir / "ckpt"),
            "json_log": str(out_dir / f"{arm}_s{seed}.jsonl"),
            "data": DATA,
            "eval": True,
            "eval_data": EVAL_DATA,
        },
    )
    (out_dir / f"{arm}_s{seed}.config.toml").write_text(run.snapshot_toml())


def _lit(value: object) -> str:
    """A config value as the `--set k=v` literal the CLI parses back to it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def run_one(arm: str, model_over: dict, seed: int, out_dir: Path, device: str, steps: int) -> int:
    """One arm, one seed, one CLI process — the battery's own launch shape."""
    jsonl = out_dir / f"{arm}_s{seed}.jsonl"
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
        f"ckpt_name=jepa_{arm}_s{seed}",
        "--set",
        f"out_dir={out_dir / 'ckpt'}",
        "--json-log",
        str(jsonl),
    ]
    for key, value in model_over.items():
        cmd += ["--set", f"{key}={_lit(value)}"]
    if device != "cuda":
        cmd[cmd.index("--device") + 1] = device
    print(f"\n=== {arm} seed {seed} -> {jsonl}", flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def ab(args: argparse.Namespace) -> int:
    """The driver: `len(ARMS)` x seeds separate processes, then the table."""
    seeds = tuple(int(s) for s in args.seeds.split(","))
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"JEPA A/B: {len(ARMS)} arms x {len(seeds)} seeds x {args.steps} steps, "
        f"{PRESET} b{BATCH} s{SEQ}, window {WINDOW} B. Output: {out_dir}\n"
        f"jepa_weight={JEPA_WEIGHT} on both active arms, jepa_reg_weight="
        f"{JEPA_REG_WEIGHT} (the collapse gate's value).\n"
        f"pre-registered (AGENTS §1.2): WIN iff an active arm's held-out BPB "
        f"beats the control's by MORE than the control's own kept-seed spread; "
        f"a tie deletes the arm."
    )
    for arm, model_over in ARMS:
        for seed in seeds:
            _snapshot(arm, model_over, seed, out_dir, args.steps)
            rc = run_one(arm, model_over, seed, out_dir, args.device, args.steps)
            if rc != 0:
                print(f"the {arm} run (seed {seed}) exited {rc}: no table.")
                return rc

    table: dict[str, list[dict]] = {arm: [] for arm, _m in ARMS}
    for arm, _model_over in ARMS:
        for seed in seeds:
            table[arm].append({"seed": seed, **read_run(out_dir / f"{arm}_s{seed}.jsonl")})

    print(
        f"\n{'arm':>8s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'warm tok/s':>10s} "
        f"{'jepa':>5s} {'nan':>4s}  note"
    )
    for arm, runs in table.items():
        flags = diverged([float(r["bpb"]) for r in runs])
        for r, bad in zip(runs, flags, strict=True):
            print(
                f"{arm:>8s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['tok_s']:10.0f} {r['jepa']:>5d} {r['nan']:>4d}  "
                f"{'DIVERGED (>3 sigma from this arm median)' if bad else ''}"
            )

    # The void checks: a number that fails one is not a number.
    void: dict[str, list[str]] = {}
    for arm, runs in table.items():
        why: list[str] = []
        for r in runs:
            if r["scored"] != WINDOW:
                why.append(f"seed {r['seed']}: window {r['scored']} != {WINDOW} B")
            if r["nan"]:
                why.append(f"seed {r['seed']}: {r['nan']} NaN-skipped steps")
            if arm == "off" and r["jepa"] != 0:
                why.append(f"seed {r['seed']}: jepa={r['jepa']} with jepa_weight=0")
            if arm != "off" and r["jepa"] == 0:
                why.append(f"seed {r['seed']}: jepa=0 — the term was never added")
        void[arm] = why

    ctrl_kept = kept([float(r["bpb"]) for r in table["off"]])
    print("\n=== trimmed per-arm summary (diverged seeds excluded from the mean)")
    summary: dict[str, list[float]] = {"off": ctrl_kept}
    for arm, runs in table.items():
        bpbs = [float(r["bpb"]) for r in runs]
        k = kept(bpbs)
        summary[arm] = k
        mean = statistics.mean(k) if k else float("nan")
        print(
            f"{arm:>8s} kept={[round(b, 4) for b in k]} n={len(k)}/{len(bpbs)} "
            f"mean={mean:.4f}  median warm tok/s="
            f"{statistics.median([r['tok_s'] for r in runs]):.0f}"
        )
        if void[arm]:
            print(f"{arm:>8s} VOID: {'; '.join(void[arm])}")

    print("\n=== verdicts vs the control (AGENTS §1.2)")
    for arm, _model_over in ARMS[1:]:
        if void[arm] or void["off"]:
            print(f"{arm:>8s} verdict: VOID — no number to compare")
            continue
        v = verdict(ctrl_kept, summary[arm])
        print(
            f"{arm:>8s} delta {v['delta']:+.4f} BPB vs control "
            f"(control spread {v['spread']:.4f}) -> {v['verdict'].upper()} "
            f"({v['why']})"
        )
    print(
        f"\nvoid checks: every eval scored {WINDOW} B, 0 NaN-skipped steps, "
        f"the control reports jepa=0, every active arm reports jepa>0. A WIN "
        f"clears the control's own kept-seed spread; anything else is a tie, "
        f"and a tie deletes the arm.\npre-registered prediction: at medium / "
        f"2 000 steps a tie is the likely outcome (the published latent-prediction "
        f"wins are far longer schedules, and the burn-era 0.082 BPB bundled two "
        f"signals under one weight); the arm is measured, not assumed. This driver "
        f"applies nothing."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ab", help="the A/B driver (3 arms x 3 seeds)")
    a.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a.add_argument("--steps", type=int, default=STEPS)
    a.add_argument("--device", default="cuda")
    a.add_argument("--out-dir", default="~/logs/ab_jepa")
    args = ap.parse_args(argv)
    why = preflight(args.device) or data_missing(
        DATA, EVAL_DATA, str(ROOT / "configs" / "mix_big.toml")
    )
    if why is not None:
        return refuse(why)
    return ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
