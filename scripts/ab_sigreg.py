"""The SIGReg A/B: control (`sigreg_weight=0.0`) vs sigreg (`sigreg_weight=0.05`).

    uv run python scripts/ab_sigreg.py ab              # 3 seeds x 2 arms, small
    uv run python scripts/ab_sigreg.py ab --seeds 1    # one seed, dry check

**Do not run this while the card is busy** (AGENTS §1.5): the preflight refuses
and prints a SKIP banner, and a skipped run has no numbers to quote.

**Arms** — `ModelConfig.sigreg_weight` (`config.py:332`; the term itself in
`model/sigreg.py`, transcription of `burn-jepa/src/losses.rs:33-50` = the direct
form of arXiv:2511.08544). One factor per run, everything else identical:

* `control` — the pure-CE control: `sigreg_weight=0.0`, the preset default.
* `sigreg`  — `sigreg_weight=0.05`, the MIDDLE rung of the research doc's
  suggested grid 0.005 / 0.05 / 0.1 (`config.py:332` names the grid; publish
  which). `loss = CE + 0.05 · (‖mean(z)‖² + ‖cov(z) − I‖²/D)` over the strided
  sample of the state `lm_head` reads — no teacher, no predictor, no mask, no
  second forward, 0 parameters, no detach (`model.py:276-286`). A negative
  weight is refused at load (`config.py:866`); `use_byteflow` + a non-zero
  weight is refused too (`config.py:956`), and `small` runs `use_byteflow =
  false`, so neither refusal is in play here.

**Protocol** (`docs/protocols/AB-PROTOCOL.md`): 3 seeds per arm, 2 000 steps,
`small`, b8 x s512, lr 1e-3, and **one eval window of 81 920 B on every arm and
seed** (`eval_batches 20 x batch 8 x seq_len 512`). The held-out eval runs
`pure=True`, so it scores CE alone on both arms — the comparison is the same
quantity on both sides, never "CE + w·term" against "CE". Each arm-seed is a
SEPARATE PROCESS, launched as the CLI would be
(`uv run --extra gpu dormouse train small ... --set sigreg_weight=<w>`), so the
config a number was taken at is the config the trainer resolved. Output: one
JSONL per run under `~/logs/ab_sigreg/`, plus the run's own
`<name>.config.toml` beside it (a number needs its config, AGENTS §1.4).

**The void check is the counter** (AGENTS §1.1): `sigreg` counts the seam where
the term is added to the loss (`model.py:284`, `if not pure and
sigreg_weight > 0.0`), so it fires once per training forward on the `sigreg` arm
and never on the control. A `sigreg` run whose log rows all report
`counters["sigreg"] == 0` measured nothing — it is flagged **VOID** and excluded
from the mean, never averaged in. Two things a reader must know about the
instrument: (1) the EVAL JSONL row carries no counters (`loop.py:849` writes
`step`/`loss`/`bpb`/`scored` only) AND the eval forward is `pure=True`, so the
term is not even evaluated there — the count is read from the `log` rows'
`counters` dict; (2) the counter proves the term was ADDED, not that its
gradient changed anything — a zero-weight arm and a weight that cancels against
the backbone's own pressure are indistinguishable from a counter alone.

**Pre-registered prediction** (written before the run, AGENTS §1.2): **a tie or
a small change either way at this scale — 9.48M params / 2 000 steps — with step
time unchanged to within a fraction of a percent.** The mechanism: the term is
three small tensor ops on a `[256, 768]` fp32 sample; the step is
HOST-DISPATCH-BOUND (AGENTS §3.1: GPU idle 87% of a warm step), so at most a few
extra launches land on a step whose floor is already ~245 ms. LeJEPA's own gains
are stated under far longer schedules; 2k steps at 9.5M is a filter, not a
measurement of the claim. Under the old rule a tie deletes the mechanism; under
the owner's new rule (2026-10-17) a regulariser has no speed gain to trade, so
its arithmetic can only come out neutral-or-worse — both verdicts are printed,
and neither is applied by this driver.

**Wall clock**: 6 runs x ~4 min ~= **25 min**, plus the preflight.

**What the driver does NOT do:** apply the verdict. It prints the table, the
void checks and BOTH pre-registered rules — AGENTS §1.2's "win iff Δ beats the
control's own 3-seed spread" and the owner's 2026-10-17 trade ("net-positive iff
measured speed gain % x 0.004 BPB > measured BPB loss"; an arm with no measured
speed gain gets the raw verdict only). A BPB is a measurement, not a verdict.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight, skip  # noqa: E402

#: The port's data on this box (AGENTS §2.6) — the LOCAL copy, so the driver does
#: not depend on the /mnt drive being mounted.
DATA = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"

#: `port/`, where `pyproject.toml` lives — `uv run` resolves the `dormouse`
#: console script from there, so every child runs with this cwd.
PORT_DIR = Path(__file__).resolve().parents[1]

SEEDS = (1, 2, 3)
STEPS = 2000
BATCH = 8
SEQ = 512
LR = 1e-3
EVAL_BATCHES = 20
EVAL_EVERY = 200
LOG_EVERY = 50
PRESET = "small"

#: The one factor under test. `control` is first so the table reads against it.
#: 0.05 is the middle rung of the research doc's grid (config.py:332).
ARMS = (
    ("control", 0.0),
    ("sigreg", 0.05),
)

#: The owner's new-rule exchange rate (2026-10-17): a speed gain of X % buys
#: `X * 0.004` BPB of quality loss before the arm stops being net-positive.
SPEED_BPB_EXCHANGE = 0.004


def _overrides(
    arm: str, weight: float, seed: int, out: Path, args: argparse.Namespace
) -> tuple[dict, dict]:
    """The ONE config dict pair this run resolves — read by both consumers.

    The snapshot and the child's `--set` argv are rendered from these, so the
    config a number was taken at cannot drift from the config the trainer ran.
    """
    model_over = {"sigreg_weight": weight}
    train_over = {
        "batch": args.batch,
        "seq_len": args.seq,
        "lr": LR,
        "steps": args.steps,
        "seed": seed,
        "eval": True,
        "eval_every": args.eval_every,
        "eval_batches": args.eval_batches,
        "log_every": args.log_every,
        "ckpt_every": 10**9,  # the A/B scores held-out BPB; no weights are written
        "data": args.data,
        "out_dir": str(Path(args.out_dir).expanduser()),
        "ckpt_name": f"{arm}_s{seed}",
        "json_log": str(out),
    }
    return model_over, train_over


def _lit(v: object) -> str:
    """A config value as the `--set k=v` literal the CLI parses back to it."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _cmd(arm: str, weight: float, seed: int, out: Path, args: argparse.Namespace) -> list[str]:
    """The child: the CLI, exactly as a reader would type it by hand."""
    model_over, train_over = _overrides(arm, weight, seed, out, args)
    sets: list[str] = []
    for k, v in {**model_over, **train_over}.items():
        sets += ["--set", f"{k}={_lit(v)}"]
    return [
        "uv",
        "run",
        "--extra",
        "gpu",
        "dormouse",
        "train",
        PRESET,
        "--device",
        args.device,
        *sets,
    ]


def _snapshot(arm: str, weight: float, seed: int, out: Path, args: argparse.Namespace) -> None:
    """Write the run's own `<name>.config.toml` beside its JSONL.

    `ckpt_every` is 10**9, so the trainer writes no checkpoint and therefore no
    snapshot of its own: without this, the number would have no config record
    (AGENTS §1.4). It resolves the SAME override dicts the child is launched
    with, so the record is the run.
    """
    from dormouse.config import load_run

    model_over, train_over = _overrides(arm, weight, seed, out, args)
    run = load_run(PRESET, model_overrides=model_over, train_overrides=train_over)
    out.with_suffix(".config.toml").write_text(run.snapshot_toml())


def _read(path: Path, steps: int) -> dict:
    """One run's JSONL -> the numbers the verdict table needs.

    `sigreg` is the max `counters["sigreg"]` over the run's log rows — the
    window counters reset at each log cadence, so a non-zero ANYWHERE is the
    evidence the term was added. `tok_s` is the median of the JSONL's own window
    rates over the run's SECOND HALF — the protocol's ">=50 warm steps" read as
    "past the warmup window".
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    evals = [r for r in rows if r.get("event") == "eval"]
    logs = [r for r in rows if r.get("event") == "log"]
    warm = [
        r["tokens_per_s"] for r in logs if r.get("step", 0) > steps / 2 and r.get("tokens_per_s")
    ]
    assert evals, f"{path}: no eval row — nothing to compare"
    return {
        "bpb": evals[-1]["bpb"],
        "scored": evals[-1]["scored"],
        "tok_s": statistics.median(warm) if warm else float("nan"),
        # `nan_skipped` is CUMULATIVE in the JSONL, so the last row's value is
        # the run's total — summing the rows would count it once per cadence.
        "nan": max((r.get("nan_skipped", 0) for r in logs), default=0),
        "sigreg": max((r.get("counters", {}).get("sigreg", 0) for r in logs), default=0),
    }


def _void(arm: str, r: dict) -> bool:
    """A `sigreg` arm that added the term zero times measured nothing."""
    return arm == "sigreg" and r["sigreg"] == 0


def ab(args: argparse.Namespace) -> int:
    """The driver: `--seeds` x 2 arms, separate processes, then the table."""
    seeds = tuple(int(s) for s in args.seeds.split(","))
    window = args.eval_batches * args.batch * args.seq
    if not Path(args.data).is_file():
        print(f"no data file at {args.data}: pass --data")
        return 1
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"A/B: {len(ARMS)} arms x {len(seeds)} seeds x {args.steps} steps, {PRESET} "
        f"b{args.batch} s{args.seq} lr={LR}, window {window} B. "
        f"Output: {out_dir}"
    )
    for arm, weight in ARMS:
        for seed in seeds:
            out = out_dir / f"{arm}_s{seed}.jsonl"
            out.unlink(missing_ok=True)
            _snapshot(arm, weight, seed, out, args)
            cmd = _cmd(arm, weight, seed, out, args)
            print(f"\n=== {arm} (sigreg_weight={weight}) seed {seed} -> {out}", flush=True)
            done = subprocess.run(cmd, cwd=PORT_DIR, check=False)
            if done.returncode != 0:
                print(f"the {arm} run (seed {seed}) exited {done.returncode}: no table.")
                return done.returncode

    table = {arm: [] for arm, _w in ARMS}
    for arm, _weight in ARMS:
        for seed in seeds:
            table[arm].append({"seed": seed, **_read(out_dir / f"{arm}_s{seed}.jsonl", args.steps)})
    print(
        f"\n{'arm':>7s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'warm tok/s':>10s} "
        f"{'sigreg':>6s} {'nan':>4s}"
    )
    for arm, runs in table.items():
        for r in runs:
            mark = " VOID" if _void(arm, r) else ""
            print(
                f"{arm:>7s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['tok_s']:10.0f} {r['sigreg']:>6d} {r['nan']:>4d}{mark}"
            )
    # The void checks: a number that fails one of these is not a number.
    for arm, runs in table.items():
        for r in runs:
            assert r["scored"] == window, (
                f"{arm} seed {r['seed']}: window {r['scored']} != {window}"
            )
            assert r["nan"] == 0, f"{arm} seed {r['seed']}: {r['nan']} NaN-skipped steps"
            if arm == "control":
                # weight 0 must gate the term OFF — a non-zero count here would
                # mean the arms are not the arms the table claims.
                assert r["sigreg"] == 0, (
                    f"control seed {r['seed']}: sigreg={r['sigreg']} with weight 0"
                )
    summary: dict[str, tuple[float, float, int] | None] = {}
    for arm, runs in table.items():
        good = [r for r in runs if not _void(arm, r)]
        if not good:
            summary[arm] = None
            print(
                f"{arm:>7s} NO NUMBER: every run reported sigreg=0 — the term was "
                f"never added, so there is nothing to compare."
            )
            continue
        bpbs = [r["bpb"] for r in good]
        summary[arm] = (statistics.mean(bpbs), max(bpbs) - min(bpbs), len(good))
        print(
            f"{arm:>7s} mean bpb {summary[arm][0]:.4f}  spread {summary[arm][1]:.4f}  "
            f"n={len(good)}  median warm tok/s "
            f"{statistics.median([r['tok_s'] for r in good]):.0f}"
        )
    ctrl = summary["control"]
    if ctrl is None:
        print("\nno control number: no verdict.")
        return 0
    ctrl_tok_s = statistics.median([r["tok_s"] for r in table["control"]])
    for arm, _weight in ARMS[1:]:
        s = summary[arm]
        if s is None:
            print(f"{arm:>7s} verdict: none — VOID")
            continue
        delta = s[0] - ctrl[0]
        verdict = "WIN" if delta < -ctrl[1] else "tie/loss (deletes the mechanism)"
        arm_tok_s = statistics.median([r["tok_s"] for r in table[arm]])
        gain_pct = (arm_tok_s / ctrl_tok_s - 1.0) * 100.0 if ctrl_tok_s > 0 else float("nan")
        if gain_pct > 0:
            bought = gain_pct * SPEED_BPB_EXCHANGE
            new = (
                f"net-positive ({bought:+.4f} BPB bought > {delta:+.4f} loss)"
                if bought > delta
                else f"not net-positive ({bought:+.4f} BPB bought <= {delta:+.4f} loss)"
            )
        else:
            new = f"no measured speed gain ({gain_pct:+.1f}%): raw verdict only"
        print(
            f"{arm:>7s} delta vs control {delta:+.4f} BPB "
            f"(control spread {ctrl[1]:.4f}) -> {verdict}; "
            f"owner's rule: speed {gain_pct:+.1f}%, {new}"
        )
    print(
        f"\nvoid checks: every eval scored {window} B, 0 NaN-skipped steps, "
        f"{len(seeds)} distinct seeds per arm, sigreg>0 on every non-control run, "
        f"sigreg=0 on every control run.\npre-registered rule (AGENTS §1.2): keep an "
        f"arm iff its mean held-out BPB beats the control's mean by more than the "
        f"CONTROL's own seed spread ({ctrl[1]:.4f}); a tie deletes the mechanism. "
        f"Owner's rule (2026-10-17): net-positive iff speed gain % x "
        f"{SPEED_BPB_EXCHANGE} > BPB loss. Pre-registered prediction: a tie at this "
        f"scale, step time unchanged within a fraction of a percent (three small "
        f"tensor ops on a [256, 768] sample against a launch-bound step)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ab", help="the A/B driver (seeds x 2 arms)")
    a.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a.add_argument("--steps", type=int, default=STEPS)
    a.add_argument("--batch", type=int, default=BATCH)
    a.add_argument("--seq", type=int, default=SEQ)
    a.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    a.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    a.add_argument("--log-every", type=int, default=LOG_EVERY)
    a.add_argument("--data", default=DATA)
    a.add_argument("--device", default="cuda")
    a.add_argument("--out-dir", default="~/logs/ab_sigreg")
    args = ap.parse_args(argv)
    why = preflight(args.device)
    if why is not None:
        return skip(why)
    return ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
