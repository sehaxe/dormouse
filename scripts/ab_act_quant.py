"""The activation-quantization A/B: control (`act_quant=none`) vs int8 vs int4 vs fp4.

    uv run python scripts/ab_act_quant.py ab              # 3 seeds x 4 arms, small
    uv run python scripts/ab_act_quant.py ab --seeds 1    # one seed, dry check

**Do not run this while the card is busy** (AGENTS §1.5): the preflight refuses
and prints a SKIP banner, and a skipped run has no numbers to quote.

**Arms** — `ModelConfig.act_quant` (`config.py:392`, the `ACT_QUANTS` rungs
`config.py:93`, formulas in `model/act_quant.py`, transcribed from
`crates/dormouse-core/src/act_quant.rs`). One factor per run, everything else
identical:

* `none` — the control: no activation quantization. Every number on record.
* `int8` — symmetric 8-bit levels (127), per-row scale, STE. The attention path
  is already 8 bits, so this arm quantizes the FFN experts only.
* `int4` — symmetric 4-bit levels (7), same shape. The a4.8 upgrade
  (`act_quant.attn_format`) keeps attention at 8 bits, so no 4-bit attention.
* `fp4`  — e2m1 (the OCP MX FP4 grid, 8 magnitudes), block max onto the format's
  own max 6.0. The 3-level bug of `9b343d3` is what this arm's grid fixed.

**`mxfp8` and `v2_int4`/`v2_int8` are NOT in this A/B, on purpose.** They were
added by the MX/BitNet-v2 lane (`mxfp.fake_quant_mx`, `hadamard.v2_quantize`)
and carry their own scale rules; the sm_120 compute question for them is a
separate lane, and this queue is the a4.8 line (`act_quant.py`'s own table). They
also REFUSE a non-zero `act_group` (`config.py:889`), so a driver that mixed them
in could not hold one `act_group` across arms.

**The companion key** (`config.py:398`): `act_group` — columns per scale,
`0` = one scale per token, else a per-group scale. `0` is the default and
is held across every arm here, so the arms differ in the FORMAT alone. NOTE
(2026-10-18 research, `docs/research/2026-10-18-act4.md` §3.1): the "a4.8 FFN
group recipe" was OUR mis-attribution — the a4.8 paper (`2411.04965`) has no
groups ("group" appears zero times); its scales are γ=max|X| (8-bit) and
β=2·mean|X| (INT4 FFN-down). The training literature's granularity is block
16/32 (NVFP4/MXFP4), never 128. Group scales remain a runnable axis, but it
tests our convention, not a published recipe. A
non-divisor is refused at load (`config.py:911`), not at the first forward.

**Protocol** (`docs/protocols/AB-PROTOCOL.md`): 3 seeds per arm, 2 000 steps,
`small`, b8 x s512, lr 1e-3, pure CE, and **one eval window of 81 920 B on every
arm and seed** (`eval_batches 20 x batch 8 x seq_len 512`). Each arm-seed is a
SEPARATE PROCESS, launched as the CLI would be
(`uv run --extra gpu dormouse train small ... --set act_quant=<arm>`), so the
config a number was taken at is the config the trainer resolved. Output: one
JSONL per run under `~/logs/ab_act_quant/`, plus the run's own
`<name>.config.toml` beside it (a number needs its config, AGENTS §1.4).

**The void check is the counter** (AGENTS §1.1): `act_quant_arms` counts the
seams where the quantizer actually ran (`experts.py:397`, `SparseMoE.forward`,
once per forward over the expert activation). A non-`none` arm whose run reports
`act_quant_arms == 0` measured nothing — it is flagged **VOID** and excluded from
the mean, never averaged in. Two things a reader must know about the instrument:
(1) the EVAL JSONL row carries no counters (`train/loop.py:849` writes
`step`/`loss`/`bpb`/`scored` only), so the count is read from the `log` rows'
`counters` dict — and the eval line prints `engram=`, not this field; (2)
the counter is the EXPERT-BANK seam. `TSCTLinear`'s readout / `lm_head` seam and
the attention upgrade do not count, so `act_quant_arms > 0` proves the experts
were quantized, not that every seam was.

**Pre-registered prediction** (written before the run, AGENTS §1.2): **a tie or
slightly worse for every non-`none` arm at this scale — 9.48M params / 2 000
steps** (the `small` preset as it resolves today; the plan says 9.2M, re-measured
by `dormouse info`) — **with a slightly higher step time.** The mechanism: the
quantizer adds dispatches to a step that is HOST-DISPATCH-BOUND (AGENTS §3.1:
GPU idle 87% of a warm step, ~46 341 `aten::mm` launches per step), so it costs
time before it buys anything. The prior is WEAK and it is a different axis from
the weights-quant A/B, which lost +0.04-0.05 BPB at this scale; activations are
a separate mechanism and a tie here deletes it (a tie deletes the mechanism).

**Wall clock**: 12 runs x ~4 min ~= **50 min**, plus the preflight.

**What the driver does NOT do:** apply the verdict. It prints the table, the void
checks and the pre-registered rule (AGENTS §1.2: keep the arm iff its mean BPB
beats the control's mean by more than the CONTROL's own seed spread; a tie
deletes the mechanism). A BPB is a measurement, not a verdict.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import torch

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

#: The one companion key (`config.py:398`), held at its default across every arm
#: so the arms differ in the FORMAT alone. `0` = one scale per token; a group is
#: the a4.8 FFN recipe and a second axis (see the module docstring).
ACT_GROUP = 0

#: The arms, control first so the table reads against it. `mxfp8` / `v2_*` are
#: deliberately absent — the module docstring names why.
ARMS = ("none", "int8", "int4", "fp4")


def skip(why: str) -> int:
    """A loud SKIP: the reason, and no table. Exit 0 — the code is not at fault."""
    print(
        f"\n=== SKIP: {why}\n"
        f"=== ONE GPU PROCESS AT A TIME (AGENTS §1.5): nothing was measured, so there\n"
        f"=== is no number to quote. Re-run when the card is free.\n",
        flush=True,
    )
    return 0


def preflight(device: str) -> str | None:
    """`None` when the card is ours, else the reason to skip (bench_opt.py's)."""
    if device == "cpu":
        print("[preflight] device=cpu: the card checks do not apply", flush=True)
        return None
    # The bracket keeps `pgrep` from matching this script's own command line.
    busy = subprocess.run(
        ["pgrep", "-af", "dormouse trai[n]"], capture_output=True, text=True
    ).stdout.strip()
    if busy:
        return f"a dormouse process is alive:\n{busy}"
    if not torch.cuda.is_available():
        return "device=cuda but no CUDA device is visible"
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 2**30
    if used >= 2.0:
        return f"the card already holds {used:.2f} GiB (the preflight's ceiling is 2.0)"
    print(f"[preflight] no dormouse process; card holds {used:.2f} GiB", flush=True)
    return None


def _overrides(arm: str, seed: int, out: Path, args: argparse.Namespace) -> tuple[dict, dict]:
    """The ONE config dict pair this run resolves — read by both consumers.

    The snapshot and the child's `--set` argv are rendered from these, so the
    config a number was taken at cannot drift from the config the trainer ran.
    """
    model_over = {"act_quant": arm, "act_group": args.act_group}
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


def _cmd(arm: str, seed: int, out: Path, args: argparse.Namespace) -> list[str]:
    """The child: the CLI, exactly as a reader would type it by hand."""
    model_over, train_over = _overrides(arm, seed, out, args)
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


def _snapshot(arm: str, seed: int, out: Path, args: argparse.Namespace) -> None:
    """Write the run's own `<name>.config.toml` beside its JSONL.

    `ckpt_every` is 10**9, so the trainer writes no checkpoint and therefore no
    snapshot of its own: without this, the number would have no config record
    (AGENTS §1.4). It resolves the SAME override dicts the child is launched
    with, so the record is the run.
    """
    from dormouse.config import load_run

    model_over, train_over = _overrides(arm, seed, out, args)
    run = load_run(PRESET, model_overrides=model_over, train_overrides=train_over)
    out.with_suffix(".config.toml").write_text(run.snapshot_toml())


def _read(path: Path, steps: int) -> dict:
    """One run's JSONL -> the numbers the verdict table needs.

    `arms` is the max `act_quant_arms` over the run's log rows — the window
    counters reset at each log cadence, so a non-zero ANYWHERE is the evidence
    the quantizer ran. `tok_s` is the median of the JSONL's own window rates over
    the run's SECOND HALF — the protocol's ">=50 warm steps" read as "past the
    warmup window".
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
        "arms": max((r.get("counters", {}).get("act_quant_arms", 0) for r in logs), default=0),
    }


def _void(arm: str, r: dict) -> bool:
    """A non-`none` arm that ran the quantizer zero times measured nothing."""
    return arm != "none" and r["arms"] == 0


def ab(args: argparse.Namespace) -> int:
    """The driver: `--seeds` x 4 arms, separate processes, then the table."""
    seeds = tuple(int(s) for s in args.seeds.split(","))
    window = args.eval_batches * args.batch * args.seq
    if not Path(args.data).is_file():
        print(f"no data file at {args.data}: pass --data")
        return 1
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"A/B: {len(ARMS)} arms x {len(seeds)} seeds x {args.steps} steps, {PRESET} "
        f"b{args.batch} s{args.seq} lr={LR}, act_group={args.act_group}, window {window} B. "
        f"Output: {out_dir}"
    )
    for arm in ARMS:
        for seed in seeds:
            out = out_dir / f"{arm}_s{seed}.jsonl"
            out.unlink(missing_ok=True)
            _snapshot(arm, seed, out, args)
            cmd = _cmd(arm, seed, out, args)
            print(f"\n=== {arm} seed {seed} -> {out}", flush=True)
            done = subprocess.run(cmd, cwd=PORT_DIR, check=False)
            if done.returncode != 0:
                print(f"the {arm} run (seed {seed}) exited {done.returncode}: no table.")
                return done.returncode

    table = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for seed in seeds:
            table[arm].append({"seed": seed, **_read(out_dir / f"{arm}_s{seed}.jsonl", args.steps)})
    print(
        f"\n{'arm':>6s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'warm tok/s':>10s} "
        f"{'arms':>9s} {'nan':>4s}"
    )
    for arm, runs in table.items():
        for r in runs:
            mark = " VOID" if _void(arm, r) else ""
            print(
                f"{arm:>6s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['tok_s']:10.0f} {r['arms']:>9d} {r['nan']:>4d}{mark}"
            )
    # The void checks: a number that fails one of these is not a number.
    for arm, runs in table.items():
        for r in runs:
            assert r["scored"] == window, (
                f"{arm} seed {r['seed']}: window {r['scored']} != {window}"
            )
            assert r["nan"] == 0, f"{arm} seed {r['seed']}: {r['nan']} NaN-skipped steps"
    summary: dict[str, tuple[float, float, int] | None] = {}
    for arm, runs in table.items():
        good = [r for r in runs if not _void(arm, r)]
        if not good:
            summary[arm] = None
            print(
                f"{arm:>6s} NO NUMBER: every run reported act_quant_arms=0 — the "
                f"quantizer never ran, so there is nothing to compare."
            )
            continue
        bpbs = [r["bpb"] for r in good]
        summary[arm] = (statistics.mean(bpbs), max(bpbs) - min(bpbs), len(good))
        print(
            f"{arm:>6s} mean bpb {summary[arm][0]:.4f}  spread {summary[arm][1]:.4f}  "
            f"n={len(good)}  median warm tok/s "
            f"{statistics.median([r['tok_s'] for r in good]):.0f}"
        )
    ctrl = summary["none"]
    if ctrl is None:
        print("\nno control number: no verdict.")
        return 0
    for arm in ARMS[1:]:
        s = summary[arm]
        if s is None:
            print(f"{arm:>6s} verdict: none — VOID")
            continue
        delta = s[0] - ctrl[0]
        verdict = "WIN" if delta < -ctrl[1] else "tie/loss (deletes the mechanism)"
        print(
            f"{arm:>6s} delta vs control {delta:+.4f} BPB "
            f"(control spread {ctrl[1]:.4f}) -> {verdict}"
        )
    print(
        f"\nvoid checks: every eval scored {window} B, 0 NaN-skipped steps, "
        f"{len(seeds)} distinct seeds per arm, act_quant_arms>0 on every non-none run.\n"
        f"pre-registered rule (AGENTS §1.2): keep an arm iff its mean held-out BPB "
        f"beats the control's mean by more than the CONTROL's own seed spread "
        f"({ctrl[1]:.4f}); a tie deletes the mechanism. Pre-registered prediction: "
        f"tie or slightly worse for every non-none arm at this scale, step time "
        f"slightly higher (the quantizer adds dispatches to a dispatch-bound step)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ab", help="the A/B driver (seeds x 4 arms)")
    a.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a.add_argument("--steps", type=int, default=STEPS)
    a.add_argument("--batch", type=int, default=BATCH)
    a.add_argument("--seq", type=int, default=SEQ)
    a.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    a.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    a.add_argument("--log-every", type=int, default=LOG_EVERY)
    a.add_argument("--act-group", type=int, default=ACT_GROUP)
    a.add_argument("--data", default=DATA)
    a.add_argument("--device", default="cuda")
    a.add_argument("--out-dir", default="~/logs/ab_act_quant")
    args = ap.parse_args(argv)
    why = preflight(args.device)
    if why is not None:
        return skip(why)
    return ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
