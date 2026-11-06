#!/usr/bin/env python3
"""The MoD A/B: control (12L, capacity 1.0) vs MoD (capacity 0.5), 3 seeds.

    uv run python scripts/ab_mod.py ab                  # 3 seeds x 2 arms, 600 steps
    uv run python scripts/ab_mod.py ab --seeds 1        # one seed, dry check
    CUDA_VISIBLE_DEVICES="" uv run python scripts/ab_mod.py ab --device cpu --steps 20

**THIS DRIVER HAS NOT BEEN RUN.** No number below is measured; the rule is
written BEFORE the first run so the first run cannot move the goalposts. A
skipped preflight prints a SKIP banner and no table.

**Arms** (the SAME `configs/mirror_control.toml` 12-layer dumpy the 2026-11-06
control used, bf16 compute, 3 seeds, equal steps and equal bytes):

* `control` — `mod_capacity=1.0`: no router is built, the plain transformer.
  13 478 448 parameters.
* `mod05` — `mod_capacity=0.5, mod_aux_weight=0.01`: one `d`-wide router per
  block (the paper's §3.4 linear projection), top-`round(0.5·T)` tokens per
  sequence through each block, `r_i·f_i + x_i` on them, the rest through the
  residual. 13 482 096 parameters (+3 648, +0.027%: the routers; every shared
  block tensor is bit-identical at the same seed because the router's constant
  init draws no RNG).

**THE CONFOUND, stated before the run.** At EQUAL STEPS the MoD arm does the
less work the ledger prices (0.63x at capacity 0.5 under the conservative
accounting, `scripts/mod_ledger.py`); so a step-time win at equal steps is
expected even if the mechanism is worthless. The fair SPEED comparison is
equal FLOPs, i.e. MoD running ≈`1/0.63` times as many steps (and therefore
seeing more bytes) — this driver deliberately does NOT run that arm, because it
would confound the quality comparison with a data-budget difference. Read the
step-time column as the cost of the arm's schedule, not as the mechanism's
efficiency.

**Pre-registered verdict rule** (written before the first run):

* VOID when any run fails a void check: eval window != 196 608 B, a different
  pinned `eval_offset`, any NaN-skipped step, `mod_block=0` on the arm or
  `mod_block>0` on the control, a params row whose router delta is not
  `n_layers·d_model`, or fewer than 3 seeds per arm.
* KEEP iff the arm's mean held-out `bpb_fin` is within **+0.02** of the
  control's mean **AND** the arm's median warm step time is **<= 0.85x** the
  control's.
* Otherwise DROP: the mechanism does not buy the FLOPs at our scale.

`bpb_fin` is the HEADLINE (the final mean readout, `e2a0542`); `bpb_mean` is the
depth-averaged objective and is never the comparison. The window is pinned by
`eval_bytes=196608` and `eval_offset` (default 0), printed on every eval row.

**Protocol** (`docs/protocols/AB-PROTOCOL.md`): 3 seeds per arm, 600 steps, one
separate process per arm-seed, output one JSONL per run under
`~/logs/ab_mod/`, the run's own `<name>.config.toml` beside it (a number needs
its config, AGENTS §1.4), and a driver `event=params` row with the instantiated
count and the router share.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.append(_SCRIPTS)
from _lane import preflight, skip  # noqa: E402
from ab_byteflow import read_run, summarize  # noqa: E402

PRESET = "mirror_control"
SEEDS = (1, 2, 3)
STEPS = 600
BATCH = 8
SEQ = 1024
EVAL_BYTES = 196_608
EVAL_OFFSET = 0
EVAL_EVERY = 100
LOG_EVERY = 50
LR = 3e-4  # TrainConfig's default; the battery does not override it

#: `(arm, mod_capacity, mod_aux_weight)`. The control first so the table reads
#: against it. The aux weight is the arm's own (the paper's §3.5 BCE).
ARMS = (
    ("control", 1.0, 0.0),
    ("mod05", 0.5, 0.01),
)

COMMON_TRAIN = dict(
    steps=STEPS,
    batch=BATCH,
    seq_len=SEQ,
    eval=True,
    eval_every=EVAL_EVERY,
    eval_bytes=EVAL_BYTES,
    eval_offset=EVAL_OFFSET,
    eval_batches=None,  # derive from eval_bytes and batch*seq_len
    log_every=LOG_EVERY,
    ckpt_every=10**9,  # the A/B scores held-out BPB; no weights are written
    timers=False,  # the JSONL's tokens_per_s is the speed instrument
)


def _ledger_ratio() -> float | None:
    """The arm's conservative FLOPs ratio, for the header and the table."""
    try:
        import mod_ledger

        from dormouse.config import load_preset

        cfg = load_preset(PRESET)
        one = mod_ledger.ledger(cfg, 1.0, seq_len=SEQ)
        half = mod_ledger.ledger(cfg, 0.5, seq_len=SEQ)
        return half.total / one.total
    except Exception as exc:  # noqa: BLE001 - a header line must not kill the driver
        print(f"[ledger] unavailable ({exc})")
        return None


def _checks(runs: list[dict], window: int, offset: int, *, arm: str) -> list[str]:
    """The void checks: a number that fails one is not a number."""
    why: list[str] = []
    for r in runs:
        if r["scored"] != window:
            why.append(f"{arm} seed {r['seed']}: window {r['scored']} != {window} B")
        if r.get("eval_offset") != offset:
            why.append(f"{arm} seed {r['seed']}: eval_offset {r.get('eval_offset')} != {offset}")
        if r["nan"]:
            why.append(f"{arm} seed {r['seed']}: {r['nan']} NaN-skipped steps")
        if not r["ms_per_step"] == r["ms_per_step"]:
            why.append(f"{arm} seed {r['seed']}: no warm step-time window")
        blocks = r["counters"].get("mod_block", 0)
        if arm == "control" and blocks != 0:
            why.append(f"{arm} seed {r['seed']}: mod_block={blocks} — the control routed")
        if arm != "control" and blocks <= 0:
            why.append(f"{arm} seed {r['seed']}: mod_block=0 — the arm did not run")
        if r.get("router_params") is None:
            why.append(f"{arm} seed {r['seed']}: no params row")
        elif arm == "control" and r["router_params"] != 0:
            why.append(f"{arm} seed {r['seed']}: control built {r['router_params']} routers")
        elif arm != "control" and r["router_params"] != 12 * 304:
            why.append(f"{arm} seed {r['seed']}: router_params={r['router_params']} != 3648")
    return why


def verdict(control: list[dict], arm: list[dict], *, window: int, offset: int) -> dict:
    """The pre-registered rule, over one control and one arm."""
    why = _checks(control, window, offset, arm="control") + _checks(
        arm, window, offset, arm="mod05"
    )
    cmean, cspread = summarize(control)
    amean, aspread = summarize(arm)
    cms = statistics.median([r["ms_per_step"] for r in control])
    ams = statistics.median([r["ms_per_step"] for r in arm])
    out = {
        "control_mean": cmean,
        "control_spread": cspread,
        "arm_mean": amean,
        "arm_spread": aspread,
        "delta": amean - cmean,
        "ms_control": cms,
        "ms_arm": ams,
        "ms_ratio": ams / cms if cms and cms == cms else float("nan"),
        "reason": "",
    }
    if why or len(control) < 3 or len(arm) < 3:
        out["verdict"] = "void"
        out["reason"] = "; ".join(why) or "fewer than 3 seeds"
    elif out["delta"] <= 0.02 and out["ms_ratio"] <= 0.85:
        out["verdict"] = "keep"
    else:
        out["verdict"] = "drop"
    return out


def _run_one(args: argparse.Namespace) -> int:
    """One arm, one seed, one process."""
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train import train

    capacity, aux = next((c, a) for name, c, a in ARMS if name == args.arm)
    run = load_run(
        PRESET,
        model_overrides={"mod_capacity": capacity, "mod_aux_weight": aux},
        train_overrides={
            **COMMON_TRAIN,
            "steps": args.steps,
            "batch": args.batch,
            "seq_len": args.seq,
            "eval_bytes": args.eval_bytes,
            "eval_every": min(EVAL_EVERY, args.steps),
            "log_every": min(LOG_EVERY, args.steps),
            "seed": args.seed,
            "lr": LR,
            "out_dir": str(Path(args.out).parent),
            "ckpt_name": f"{args.arm}_s{args.seed}",
            "json_log": str(args.out),
            "eval_data": args.eval,
        },
    )
    run.train.data = args.data
    if not Path(run.train.data).is_file():
        print(f"no data file at {run.train.data}: pass --data")
        return 1
    Path(args.out).with_suffix(".config.toml").write_text(run.snapshot_toml())

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    import torch

    model = DormouseModel(run.model, device=torch.device(device))
    total = sum(p.numel() for p in model.parameters())
    router_params = sum(p.numel() for n, p in model.named_parameters() if n.endswith(".router"))
    window = args.eval_bytes
    print(
        f"[run] arm={args.arm} capacity={capacity} aux={aux} seed={args.seed} "
        f"steps={args.steps} b{args.batch} s{args.seq} lr={LR} "
        f"window={window} B offset={EVAL_OFFSET} params={total:,} "
        f"router_params={router_params:,}",
        flush=True,
    )
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train(model, run, device=torch.device(device), log=print)

    def _row(**kw) -> None:
        with open(args.out, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(kw) + "\n")

    params = {
        "params": total,
        "router_params": router_params,
        "mod_capacity": capacity,
        "mod_aux_weight": aux,
    }
    if torch.device(device).type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        params["peak_gib"] = peak
        print(f"[peak] {peak:.2f} GiB (torch.cuda.max_memory_allocated)", flush=True)
    _row(event="params", step=args.steps, **params)
    return 0


def ab(args: argparse.Namespace) -> int:
    """The driver: `--seeds` x 2 arms, separate processes, then the table."""
    reason = preflight(args.device)
    if reason:
        skip(reason)
        return 0
    seeds = tuple(int(s) for s in args.seeds.split(","))
    window = args.eval_bytes
    ratio = _ledger_ratio()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"A/B: {len(ARMS)} arms x {len(seeds)} seeds x {args.steps} steps, "
        f"{PRESET} b{args.batch} s{args.seq} lr={LR}, eval window {window} B "
        f"at offset {EVAL_OFFSET}. Output: {out_dir}\n"
        f"pre-registered: KEEP iff MoD's mean bpb_fin is within +0.02 of the "
        f"control AND its median step time is <= 0.85x the control; otherwise "
        f"DROP.\n"
        f"CONFOUND: at equal steps the MoD arm sees "
        f"{'?' if ratio is None else f'{ratio:.2f}x'} the control's ledger FLOPs, "
        f"so the step-time column prices the arm's SCHEDULE, not its efficiency. "
        f"The fair speed comparison is equal FLOPs, i.e. ~"
        f"{'?' if ratio is None else f'{1 / ratio:.2f}x'} more MoD steps — not "
        f"run here, because it changes the data budget."
    )
    for arm, _capacity, _aux in ARMS:
        for seed in seeds:
            out = out_dir / f"{arm}_s{seed}.jsonl"
            out.unlink(missing_ok=True)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_run",
                "--arm",
                arm,
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
                "--eval-bytes",
                str(args.eval_bytes),
            ]
            print(f"\n=== {arm} seed {seed} -> {out}", flush=True)
            done = subprocess.run(cmd, check=False)
            if done.returncode != 0:
                print(f"the {arm} run (seed {seed}) exited {done.returncode}: no table.")
                return done.returncode

    table: dict[str, list[dict]] = {arm: [] for arm, _c, _a in ARMS}
    for arm, _capacity, _aux in ARMS:
        for seed in seeds:
            path = out_dir / f"{arm}_s{seed}.jsonl"
            row = {"seed": seed, **read_run(path, args.steps)}
            for line in path.read_text().splitlines():
                r = json.loads(line)
                if r.get("event") == "params":
                    row.update(
                        params=r.get("params"),
                        router_params=r.get("router_params"),
                        mod_capacity=r.get("mod_capacity"),
                        mod_aux_weight=r.get("mod_aux_weight"),
                    )
                if r.get("event") == "eval":
                    row["eval_offset"] = r.get("eval_offset")
            table[arm].append(row)

    print(
        f"\n{'arm':>8s} {'seed':>4s} {'bpb_fin':>8s} {'scored B':>9s} {'off':>6s} "
        f"{'ms/step':>8s} {'tok/s':>9s} {'params':>10s} {'routers':>8s} {'nan':>4s}"
    )
    for arm, runs in table.items():
        for r in runs:
            print(
                f"{arm:>8s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r.get('eval_offset', -1):>6d} {r['ms_per_step']:8.1f} "
                f"{r['tok_s']:9.0f} {r.get('params', 0):>10} "
                f"{r.get('router_params', 0):>8} {r['nan']:>4d}"
            )
    for arm, runs in table.items():
        mean, spread = summarize(runs)
        print(
            f"{arm:>8s} mean bpb_fin {mean:.4f}  spread {spread:.4f}  "
            f"median warm ms/step {statistics.median([r['ms_per_step'] for r in runs]):.1f}"
        )
    v = verdict(table["control"], table["mod05"], window=window, offset=EVAL_OFFSET)
    print(
        f"\nvoid checks: window {window} B, offset {EVAL_OFFSET}, 0 NaN-skipped "
        f"steps, mod_block on the arm and not on the control, 3 seeds per arm.\n"
        f"control {v['control_mean']:.4f} +- {v['control_spread']:.4f} | "
        f"mod05 {v['arm_mean']:.4f} +- {v['arm_spread']:.4f} | "
        f"delta {v['delta']:+.4f} (rule: <= +0.02) | "
        f"median ms/step ratio {v['ms_ratio']:.3f} (rule: <= 0.85)\n"
        f"VERDICT: {v['verdict'].upper()}"
        + (f"\nvoid reasons: {v['reason']}" if v["reason"] else "")
    )
    return 0


def _args() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ab", "_run"):
        p = sub.add_parser(name)
        p.add_argument("--seeds", default="1,2,3")
        p.add_argument("--steps", type=int, default=STEPS)
        p.add_argument("--batch", type=int, default=BATCH)
        p.add_argument("--seq", type=int, default=SEQ)
        p.add_argument("--eval-bytes", type=int, default=EVAL_BYTES)
        p.add_argument("--data", default="/home/sehaxe/dormouse/data_local/mix4/web.bin")
        p.add_argument("--eval", default="/home/sehaxe/dormouse/data_local/mix4/web_eval.bin")
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--out-dir", default="~/logs/ab_mod")
        if name == "_run":
            p.add_argument("--arm", required=True)
            p.add_argument("--seed", type=int, required=True)
            p.add_argument("--out", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _args().parse_args(argv)
    if args.cmd == "_run":
        return _run_one(args)
    return ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
