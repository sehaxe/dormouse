#!/usr/bin/env -S uv run --script
"""The document-masking A/B: flat concatenation (control) vs one document per row.

    uv run python scripts/ab_docmask.py ab                 # 3 seeds x 2 arms, 600 steps
    uv run python scripts/ab_docmask.py ab --seeds 1       # one seed, dry check
    uv run python scripts/ab_docmask.py ab --long          # 200 steps at seq_len=4096

**THIS DRIVER HAS NOT BEEN RUN.** The brief forbids running the arms; every
number below is unmeasured and the pre-registered rule is written BEFORE the
first run. A skipped preflight prints a SKIP banner and no table.

**Arms** — ONE flag differs (`doc_masking`), everything else is the battery's
COMMON (`medium`, 600 steps, b8 x s512, bf16 + triton MoE + `moe_capacity=0`
dropless, tf32 + `ns_kind=triton` + `opt_graph=true`, one 81 920 B eval window
= 20 x 8 x 512 on every arm and seed):

* `off` — `doc_masking=false`: the flat byte concatenation, every number on
  record. A `batch x (seq_len+1)` window is cut from one contiguous byte range,
  so a window that crosses a document boundary lets the model see the previous
  document's bytes.
* `on` — `doc_masking=true`: one document per row (`train/data.py::
  DocPackedStream`), padded at the tail; padding and cross-document targets are
  excluded from the loss (`masked_ce_mean`). The eval stream stays flat, so the
  held-out BPB is the same instrument on both arms.

**Why the arm is not whole-document PACKING.** The KDA arm is a recurrence
carried through the fused `chunk_kda` over the whole window, and there is no
state reset at a mid-window boundary. Gluing two documents into one row (the
usual packing) would still let document B attend to document A. One document
per row needs no reset anywhere.

**Data prerequisite.** The `on` arm needs `web.bin.docs.json` beside the
`.bin`; the driver refuses to start without it and prints the builder command.
The sidecar is produced by `scripts/build_doc_sidecar.py` from the domain's
source `*.txt` directory (the `web` domain is 373 whole-shard documents over
2 350 670 235 B; a `.bin` alone cannot say where a document ends).

**Pre-registered verdict rule (AGENTS §1.2).** Each arm is the mean held-out
BPB over the 3 seeds; a WIN requires the arm's mean to beat the control's mean
by MORE than the CONTROL's own seed spread. Anything else — including a tie —
keeps `off` (it is the arm every number on record already has, and it wastes no
padding). VOID when any run fails a void check: the eval window is 81 920 B, 0
NaN-skipped steps, `forward>0`, the `on` arm reports `docmask_windows>0` and
`docmask_docs>0` with `docmask_scored>0`, and the control reports no document
counters. A void arm is not a number.

**Pre-registered prediction: a TIE on `web.bin`.** The `web` documents are
~6.3 MB each; a b8 x s512 window is 4 096 B, so the chance that a flat window
straddles a boundary is ~4096 / 6.3e6 ≈ 0.065 % per window — roughly 3 rows in
the whole 600-step A/B. The control therefore almost never exercises the
mechanism this arm removes, and the measured delta is predicted to sit inside
the seed spread. **This A/B is a correctness/cost check on `web`, not an
effect-size measurement.** The corpora where the arm has leverage are the
short-document ones (`code`, `instruct`, `qa`) and the 8k-32k windows the
concatenation artifact dominates; neither is this brief's data, so neither is
claimed here.

**`--long`** runs the same two arms for 200 steps at `seq_len=4096` (batch 1,
2 x 1 x 4096 = 8 192 B eval window) as a validity/cost probe, not a verdict:
it proves the boundary path runs at 4k without a state reset and reports the
step cost and peak VRAM. The same tie is predicted, for the same reason.
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
from _lane import preflight  # noqa: E402
from ab_byteflow import read_run, summarize  # noqa: E402

#: The brief's data (AGENTS §2.6: the LOCAL copy) and its held-out file.
DATA = "/home/sehaxe/dormouse/data_local/mix4/web.bin"
EVAL = "/home/sehaxe/dormouse/data_local/mix4/web_eval.bin"
SIDECAR = DATA + ".docs.json"

SEEDS = (1, 2, 3)
STEPS = 600
LONG_STEPS = 200
PRESET = "medium"
BATCH = 8
SEQ = 512
EVAL_BATCHES = 20
EVAL_EVERY = 200
LOG_EVERY = 250
LR = 3e-4  # TrainConfig's default; the battery does not override it

#: `ab_mhc.py`'s COMMON minus `mix_plan` (it would make the named `--data`
#: inert, exactly as `ab_attn.py` and `ab_csa.py` document) because this brief
#: names ONE data file. `moe_capacity=0` is the dropless route the doc arm
#: requires (`config._validate_doc`: a capacity-padded dispatch reads the
#: padding rows when it decides which real rows an expert keeps).
COMMON_TRAIN = dict(
    steps=STEPS,
    batch=BATCH,
    seq_len=SEQ,
    eval=True,
    eval_every=EVAL_EVERY,
    eval_batches=EVAL_BATCHES,
    log_every=LOG_EVERY,
    ckpt_every=10**9,  # the A/B scores held-out BPB; no weights are written
    tf32=True,
    ns_kind="triton",
    opt_graph=True,
    engram_ram=True,
)

COMMON_MODEL = dict(
    moe_kernel="triton",
    moe_capacity=0.0,
    bf16=True,
    mhc_streams=2,
    use_attnres=False,
    loom=False,
)

#: `(arm name, doc_masking)`. The control is first so the table reads against it.
ARMS = (
    ("off", False),
    ("on", True),
)

EXPECT = {
    "off": {"forward": ">0", "docmask_windows": "==0"},
    "on": {"forward": ">0", "docmask_windows": ">0", "docmask_docs": ">0"},
}


def check(runs: list[dict], window: int, arm: str) -> list[str]:
    """The void checks: a number that fails one is not a number."""
    why: list[str] = []
    for r in runs:
        if r["scored"] != window:
            why.append(f"seed {r['seed']}: eval window {r['scored']} != {window} B")
        if r["nan"]:
            why.append(f"seed {r['seed']}: {r['nan']} NaN-skipped steps")
        for name, want in EXPECT[arm].items():
            got = int(r["doc"].get(name, r["counters"].get(name, 0)))
            if want == ">0" and got <= 0:
                why.append(f"seed {r['seed']}: {name}={got} — the arm did not run")
            if want == "==0" and got != 0:
                why.append(f"seed {r['seed']}: {name}={got} — the wrong shape ran")
        if arm == "on" and int(r["doc"].get("docmask_scored", 0)) <= 0:
            why.append(f"seed {r['seed']}: docmask_scored=0 — no position was scored")
    return why


def verdict(control: list[dict], arm: list[dict], window: int) -> dict:
    """The pre-registered rule: WIN iff the arm beats the CONTROL's spread."""
    why = check(control, window, "off") + check(arm, window, "on")
    cmean, cspread = summarize(control)
    amean, aspread = summarize(arm)
    out = {
        "control_mean": cmean,
        "control_spread": cspread,
        "arm_mean": amean,
        "arm_spread": aspread,
        "delta": amean - cmean,
        "reason": "",
    }
    out["ms_control"] = statistics.median([r["ms_per_step"] for r in control])
    out["ms_arm"] = statistics.median([r["ms_per_step"] for r in arm])
    out["ms_delta"] = out["ms_arm"] - out["ms_control"]
    if why:
        out["verdict"] = "void"
        out["reason"] = "; ".join(why)
    elif amean < cmean - cspread:
        out["verdict"] = "win"
    else:
        out["verdict"] = "tie"
    return out


def ms_per_step(tokens_per_s: float, batch: int = BATCH, seq_len: int = SEQ) -> float:
    """`tokens/s` → `ms/step` — the ONE conversion both tables use."""
    return (batch * seq_len) / tokens_per_s * 1e3


def _run_one(args: argparse.Namespace) -> int:
    """One arm, one seed, one process: the trainer with the arm made live."""
    import torch

    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train import train

    if args.arm == "on" and not Path(args.sidecar).is_file():
        print(
            f"doc_masking=on needs {args.sidecar}. Build it first:\n"
            f"  uv run scripts/build_doc_sidecar.py --bin {args.data} "
            f"--source-dir <the web *.txt shard directory>"
        )
        return 2
    run = load_run(
        PRESET,
        model_overrides=COMMON_MODEL,
        train_overrides={
            **COMMON_TRAIN,
            "steps": args.steps,
            "batch": args.batch,
            "seq_len": args.seq,
            "eval_batches": args.eval_batches,
            "eval_every": args.eval_every,
            "log_every": args.log_every,
            "seed": args.seed,
            "lr": LR,
            "doc_masking": args.arm == "on",
            "out_dir": str(Path(args.out).parent),
            "ckpt_name": f"{args.arm}_s{args.seed}",
            "timers": False,
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
    model = DormouseModel(run.model, device=torch.device(device))
    total = sum(p.numel() for p in model.parameters())
    window = args.eval_batches * args.batch * args.seq
    print(
        f"[run] arm={args.arm} doc_masking={run.train.doc_masking} seed={args.seed} "
        f"steps={args.steps} b{args.batch} s{args.seq} lr={LR} window={window} B "
        f"params={total:,}",
        flush=True,
    )
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    report = train(model, run, device=torch.device(device), log=print)

    def _row(**kw) -> None:
        with open(args.out, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(kw, default=str) + "\n")

    doc = report.get("doc_counts") or {}
    params = {"params": total, "doc_counts": doc}
    if torch.device(device).type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        params["peak_gib"] = peak
        print(f"[peak] {peak:.2f} GiB (torch.cuda.max_memory_allocated)", flush=True)
    _row(event="params", step=args.steps, **params)
    _row(event="docmask", step=args.steps, **doc)
    return 0


def ab(args: argparse.Namespace) -> int:
    """The driver: `--seeds` x 2 arms, separate processes, then the table."""
    seeds = tuple(int(s) for s in args.seeds.split(","))
    window = args.eval_batches * args.batch * args.seq
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not Path(args.data).is_file():
        print(f"no data file at {args.data}: pass --data")
        return 1
    if not Path(args.sidecar).is_file():
        print(
            f"SKIP: {args.sidecar} is missing — the `on` arm cannot run without "
            f"document boundaries. Build it with `scripts/build_doc_sidecar.py`."
        )
        return 2
    reason = preflight(args.device)
    if reason is not None:
        print(f"SKIP: {reason}")
        return 0
    mode = "LONG 4096-byte probe" if args.long else "A/B"
    print(
        f"{mode}: 2 arms x {len(seeds)} seeds x {args.steps} steps, {PRESET} "
        f"b{args.batch} s{args.seq} lr={LR}, window {window} B. Output: {out_dir}\n"
        f"pre-registered: `on` lands ONLY if its mean held-out BPB beats the "
        f"control's by more than the CONTROL's own seed spread; a tie keeps the "
        f"flat control. Prediction: TIE (web documents are ~6.3 MB; a 4 KiB window "
        f"almost never straddles one)."
    )
    for arm, _on in ARMS:
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
                "--sidecar",
                args.sidecar,
                "--device",
                args.device,
                "--steps",
                str(args.steps),
                "--batch",
                str(args.batch),
                "--seq",
                str(args.seq),
                "--eval-batches",
                str(args.eval_batches),
                "--eval-every",
                str(args.eval_every),
                "--log-every",
                str(args.log_every),
            ]
            print(f"\n=== {arm} seed {seed} -> {out}", flush=True)
            done = subprocess.run(cmd, check=False)
            if done.returncode != 0:
                print(f"the {arm} run (seed {seed}) exited {done.returncode}: no table.")
                return done.returncode

    table: dict[str, list[dict]] = {arm: [] for arm, _ in ARMS}
    for arm, _on in ARMS:
        for seed in seeds:
            path = out_dir / f"{arm}_s{seed}.jsonl"
            row = {"seed": seed, **read_run(path, args.steps)}
            for line in path.read_text().splitlines():
                r = json.loads(line)
                if r.get("event") == "params":
                    row["params"] = r.get("params")
                if r.get("event") == "docmask":
                    row["doc"] = r
            row.setdefault("doc", {})
            table[arm].append(row)
    print(
        f"\n{'arm':>4s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'ms/step':>8s} "
        f"{'tok/s':>8s} {'peak GiB':>8s} {'nan':>4s} {'docs':>5s} {'masked':>7s}"
    )
    for arm, runs in table.items():
        for r in runs:
            peak = f"{r['peak_gib']:.2f}" if r["peak_gib"] is not None else "-"
            print(
                f"{arm:>4s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['ms_per_step']:8.1f} {r['tok_s']:8.0f} {peak:>8s} {r['nan']:>4d} "
                f"{int(r['doc'].get('docmask_docs', 0)):>5d} "
                f"{int(r['doc'].get('docmask_masked', 0)):>7d}"
            )
    for arm, runs in table.items():
        mean, spread = summarize(runs)
        print(
            f"{arm:>4s} mean bpb {mean:.4f}  spread {spread:.4f}  "
            f"median warm ms/step {statistics.median([r['ms_per_step'] for r in runs]):.1f}"
        )
    v = verdict(table["off"], table["on"], window)
    print(
        f"\nvoid checks: every eval scored {window} B, 0 NaN-skipped steps, "
        f"off docmask_windows==0, on docmask_windows>0 + docs>0 + scored>0, "
        f"{len(seeds)} distinct seeds per arm.\n"
        f"pre-registered rule (AGENTS §1.2): keep `on` iff its mean held-out "
        f"BPB beats the control's mean by more than the CONTROL's own seed spread; "
        f"a tie keeps the flat control. Verdict: {v['verdict'].upper()} "
        f"(control {v['control_mean']:.4f} +- {v['control_spread']:.4f}, "
        f"on {v['arm_mean']:.4f} +- {v['arm_spread']:.4f}, "
        f"delta {v['delta']:+.4f})"
        + (f"\nvoid reasons: {v['reason']}" if v["reason"] else "")
    )
    print(
        f"step time (reported, NOT folded into the BPB rule): median "
        f"{v['ms_control']:.1f} -> {v['ms_arm']:.1f} ms/step "
        f"({v['ms_delta']:+.1f} ms, {v['ms_delta'] / v['ms_control']:+.1%}). "
        f"At 600 steps on `web.bin` this is a null check by construction: the "
        f"effect this arm removes is ~0.065% of windows here."
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
    a.add_argument("--eval", default=EVAL)
    a.add_argument("--sidecar", default=SIDECAR)
    a.add_argument("--device", default="cuda")
    a.add_argument("--out-dir", default="~/logs/ab_docmask")
    a.add_argument(
        "--long",
        action="store_true",
        help="the 200-step seq_len=4096 probe (overrides steps/batch/seq/eval)",
    )
    r = sub.add_parser("_run", help="one arm, one seed (the child mode)")
    r.add_argument("--arm", required=True, choices=[name for name, _ in ARMS])
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--eval", required=True)
    r.add_argument("--sidecar", required=True)
    r.add_argument("--device", default="cuda")
    r.add_argument("--steps", type=int, default=STEPS)
    r.add_argument("--batch", type=int, default=BATCH)
    r.add_argument("--seq", type=int, default=SEQ)
    r.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    r.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    r.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = ap.parse_args(argv)
    if args.mode == "ab" and args.long:
        args.steps = LONG_STEPS
        args.batch = 1
        args.seq = 4096
        args.eval_batches = 2
        args.eval_every = 100
        args.log_every = 50
        if args.out_dir == "~/logs/ab_docmask":
            args.out_dir = "~/logs/ab_docmask/long"
    return ab(args) if args.mode == "ab" else _run_one(args)


if __name__ == "__main__":
    sys.exit(main())
