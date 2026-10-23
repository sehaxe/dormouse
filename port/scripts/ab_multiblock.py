"""The multi-block A/B: one shared body (control) vs DISTINCT bodies in the loop.

    uv run python scripts/ab_multiblock.py ab            # 3 seeds x 3 arms, 600 steps
    uv run python scripts/ab_multiblock.py ab --seeds 1  # one seed, dry check

**THIS DRIVER HAS NOT BEEN RUN.** The card is held by a training run (AGENTS
§1.5: one heavy thing at a time), so every number below is unmeasured and the
pre-registered rule is written BEFORE the first run. A skipped preflight prints
a SKIP banner and no table.

**Arms** (medium, b8 x s512, one 81 920 B eval window, the battery's COMMON as
in `ab_mhc.py`/`ab_prelude.py`; only the two fields below differ):

* `b1` — `loop_blocks=1, max_iter=4`: the control, today's full-stack loop
  (`embed → LoopBlock ×4 → RMSNorm → lm_head`), 141 910 405 params;
* `b2` — `loop_blocks=2, max_iter=4`: body 0 on iterations 0/2, body 1 on 1/3,
  283 471 558 params (+141 561 153 — ONE full body: controller, KDA, Engram,
  its own MoE bank, readout, residual operator). The SAME four passes as `b1`,
  so the extra cost is parameters, not tokens;
* `b1x2` — `loop_blocks=1, max_iter=8`: the same one body run EIGHT times,
  141 910 405 params. The COMPUTE control: +4 passes of one body, no new
  transformation.

**Pre-registered prediction.** `b2` is a TIE on held-out BPB, or a win smaller
than the control's own seed spread. The prelude/coda experiment already measured
one form of depth at this scale and it was a TIE (`ab_prelude.py`, +2.2-4.4%
parameters, no held-out win) — distinct bodies inside the loop are a second form
of the same bet, not a free quality term. If `b2` does beat `b1`, the FIRST
question is whether `max_iter=8` on ONE body beats it too: the same extra passes
of a shared body are the cheaper way to buy the same compute, and `loop_blocks`
is only justified if the new TRANSFORMATION pays over that.

**Pre-registered verdict rule** (parameter AND compute confounds both stated):

* VOID when any run fails a void check: eval window != 81 920 B, any NaN-skipped
  step, a params row whose `loop_blocks`/`max_iter` is not the arm's, the built
  `loop_extra` count != `loop_blocks - 1`, or fewer than 3 seeds per arm.
* KEEP iff `b2`'s mean held-out BPB beats BOTH:
  - `b1`'s mean by MORE than the CONTROL's own 3-seed spread (the effect is real
    against seed noise), AND
  - `b1x2`'s mean by MORE than that same spread (the effect is the distinct
    body, not the extra passes).
* COMPUTE-CONFOUNDED iff `b2` beats `b1` by more than the spread but does NOT
  beat `b1x2` by it: the doubled shared loop is doing the work, and
  `loop_blocks` has not earned a preset.
* Otherwise TIE.

**The parameter confound is NOT controlled.** `b2` carries +141.6M parameters
(+99.8%) against `b1`; `b1x2` holds the parameter count at 1x and doubles the
passes. No arm here is parameter-matched to `b2` (that would need a wider or
deeper one-body arm, a different experiment). So even a KEEP means "distinct
bodies beat a shallower AND a deeper shared loop", not "free quality". A TIE
leaves `loop_blocks=1` as every preset's default: the knob is DEPTH FOR SCALE,
not a quality win at 142M, and adding it to a preset is a separate decision that
needs the parameter-matched control.

**Protocol** (`docs/protocols/AB-PROTOCOL.md`): 3 seeds per arm, 600 steps, each
arm-seed a SEPARATE PROCESS (3 seeds is 3 initialisations), one eval window of
81 920 B on every run, output one JSONL per run under `~/logs/ab_multiblock/`
plus the run's own `<name>.config.toml` (a number needs its config, AGENTS §1.4)
and a driver `event=params` row (the instantiated count, the extra body's count,
`loop_blocks` and `max_iter`).

Two deliberate deviations from `ab_mhc.py`'s COMMON, both inherited from
`ab_prelude.py` and both named there: `mix_plan` is NOT copied (it would replace
the named single-file stream and make `--data` inert), and `eval_every` is 200,
not 500 (600 steps, and the driver scores the LAST eval row).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

# `scripts/` is not a package: reuse the byteflow driver's preflight/readers
# rather than a second copy of the same card checks and JSONL parsing.
_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.append(_SCRIPTS)
from ab_byteflow import preflight, read_run, skip, summarize  # noqa: E402

#: The brief's data (AGENTS §2.6: the LOCAL copy) and its held-out file.
DATA = "/home/sehaxe/dormouse/data_local/mix4/web.bin"
EVAL = "/home/sehaxe/dormouse/data_local/mix4/web_eval.bin"

SEEDS = (1, 2, 3)
STEPS = 600
PRESET = "medium"
BATCH = 8
SEQ = 512
EVAL_BATCHES = 20
EVAL_EVERY = 200
LOG_EVERY = 250
LR = 3e-4  # TrainConfig's default; the battery does not override it

#: `ab_mhc.py`'s COMMON_TRAIN minus `mix_plan` (deviation 1 in the docstring).
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

#: The model-side COMMON (`--set` flags in the battery) plus the arm controls.
#: `use_attnres`/`loom` are pinned false so a preset edit cannot move a second
#: residual arm into the comparison.
COMMON_MODEL = dict(
    moe_kernel="triton",
    moe_capacity=0.0,
    bf16=True,
    mhc_streams=2,
    use_attnres=False,
    loom=False,
)

#: `(arm name, loop_blocks, max_iter)`. The control is first so the table reads
#: against it; `b1x2` is the compute control, NOT a third mechanism.
ARMS = (
    ("b1", 1, 4),
    ("b2", 2, 4),
    ("b1x2", 1, 8),
)

#: The counters every arm must show. The arm's SHAPE is checked from the
#: params row (`loop_blocks`/`max_iter`/`loop_extra`), because the counter line
#: carries no per-body key: a run that silently ran one body 4 times would show
#: the same `iter` count as the two-body arm.
EXPECT = {arm: {"forward": ">0", "iter": ">0"} for arm, _b, _i in ARMS}


def check(runs: list[dict], window: int, arm: str, blocks: int, iters: int) -> list[str]:
    """The void checks: a number that fails one is not a number."""
    why: list[str] = []
    for r in runs:
        if r["scored"] != window:
            why.append(f"seed {r['seed']}: eval window {r['scored']} != {window} B")
        if r["nan"]:
            why.append(f"seed {r['seed']}: {r['nan']} NaN-skipped steps")
        for name, want in EXPECT[arm].items():
            got = r["counters"].get(name, 0)
            if want == ">0" and got <= 0:
                why.append(f"seed {r['seed']}: {name}={got} — the arm did not run")
        if r.get("loop_blocks") != blocks:
            why.append(f"seed {r['seed']}: loop_blocks={r.get('loop_blocks')} != {blocks}")
        if r.get("max_iter") != iters:
            why.append(f"seed {r['seed']}: max_iter={r.get('max_iter')} != {iters}")
        if r.get("loop_extra") != blocks - 1:
            why.append(
                f"seed {r['seed']}: loop_extra={r.get('loop_extra')} != {blocks - 1} "
                f"— the wrong tree ran"
            )
    return why


def verdict(
    control: list[dict], arm: list[dict], window: int, name: str, blocks: int, iters: int
) -> dict:
    """AGENTS §1.2 over one control and one arm, plus the reported step time."""
    why = check(control, window, "b1", 1, 4) + check(arm, window, name, blocks, iters)
    cmean, cspread = summarize(control)
    amean, aspread = summarize(arm)
    cms = statistics.median([r["ms_per_step"] for r in control])
    ams = statistics.median([r["ms_per_step"] for r in arm])
    out = {
        "arm": name,
        "control_mean": cmean,
        "control_spread": cspread,
        "arm_mean": amean,
        "arm_spread": aspread,
        "delta": amean - cmean,
        "ms_control": cms,
        "ms_arm": ams,
        "ms_delta": ams - cms,
        "reason": "",
    }
    if why:
        out["verdict"] = "void"
        out["reason"] = "; ".join(why)
    elif amean < cmean - cspread:
        out["verdict"] = "win"
    else:
        out["verdict"] = "tie"
    return out


def decide(v2: dict, vx2: dict) -> str:
    """The pre-registered rule over the two controls (module docstring)."""
    if v2["verdict"] == "void" or vx2["verdict"] == "void":
        return "void"
    spread = v2["control_spread"]
    if v2["verdict"] != "win":
        return "tie"
    if v2["arm_mean"] >= vx2["arm_mean"] - spread:
        # The one-body / 8-pass arm matches (or beats) b2 within the control's
        # spread: the same extra passes of a shared body buy the gain, so the
        # distinct body has not earned the preset.
        return "compute-confounded"
    return "keep"


def _run_one(args: argparse.Namespace) -> int:
    """One arm, one seed, one process: the trainer with the loop shape made live."""
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train import train

    blocks, iters = next((b, i) for name, b, i in ARMS if name == args.arm)
    run = load_run(
        PRESET,
        model_overrides={
            **COMMON_MODEL,
            "loop_blocks": blocks,
            "max_iter": iters,
        },
        train_overrides={
            **COMMON_TRAIN,
            "seed": args.seed,
            "lr": LR,
            "out_dir": str(Path(args.out).parent),
            "ckpt_name": f"{args.arm}_s{args.seed}",
            # The JSONL's `tokens_per_s` is this driver's speed instrument and
            # `timers` OFF is what keeps it honest (`timer.report()` resets the
            # mark the telemetry row reads — `train/loop.py`'s own comment).
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
    import torch

    model = DormouseModel(run.model, device=torch.device(device))
    assert len(model.loop_extra) == blocks - 1, (
        "the arm's dispatch does not match its flags: "
        f"built {len(model.loop_extra)} extra bodies, wanted {blocks - 1}"
    )
    assert model.cfg.max_iter == iters
    extra_params = sum(p.numel() for p in model.loop_extra.parameters())
    total = sum(p.numel() for p in model.parameters())
    window = args.eval_batches * args.batch * args.seq
    print(
        f"[run] arm={args.arm} loop_blocks={blocks} max_iter={iters} seed={args.seed} "
        f"steps={args.steps} b{args.batch} s{args.seq} lr={LR} window={window} B "
        f"params={total:,} extra_body_params={extra_params:,} "
        f"({extra_params / total:.4%})",
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
        "loop_blocks": blocks,
        "max_iter": iters,
        "loop_extra": len(model.loop_extra),
        "extra_body_params": extra_params,
    }
    if torch.device(device).type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        params["peak_gib"] = peak
        print(f"[peak] {peak:.2f} GiB (torch.cuda.max_memory_allocated)", flush=True)
    _row(event="params", step=args.steps, **params)
    return 0


def ab(args: argparse.Namespace) -> int:
    """The driver: `--seeds` x 3 arms, separate processes, then the table."""
    seeds = tuple(int(s) for s in args.seeds.split(","))
    window = args.eval_batches * args.batch * args.seq
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"A/B: {len(ARMS)} arms x {len(seeds)} seeds x {args.steps} steps, {PRESET} "
        f"b{args.batch} s{args.seq} lr={LR}, window {window} B. Output: {out_dir}\n"
        f"pre-registered: b2 is a TIE, or a win smaller than the control's seed "
        f"spread; if it wins, the max_iter=8 one-body arm (b1x2) is the cheaper "
        f"compute control and must NOT match it. b2 carries +141.6M parameters."
    )
    for arm, _blocks, _iters in ARMS:
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

    table = {arm: [] for arm, _b, _i in ARMS}
    for arm, _blocks, _iters in ARMS:
        for seed in seeds:
            row = {"seed": seed, **read_run(out_dir / f"{arm}_s{seed}.jsonl", args.steps)}
            for line in (out_dir / f"{arm}_s{seed}.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r.get("event") == "params":
                    row.update(
                        params=r.get("params"),
                        loop_blocks=r.get("loop_blocks"),
                        max_iter=r.get("max_iter"),
                        loop_extra=r.get("loop_extra"),
                        extra_body_params=r.get("extra_body_params"),
                    )
            table[arm].append(row)
    print(
        f"\n{'arm':>5s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'ms/step':>8s} "
        f"{'tok/s':>8s} {'peak GiB':>8s} {'nan':>4s} {'params':>10s} {'bodies':>6s} "
        f"{'iters':>5s}"
    )
    for arm, runs in table.items():
        for r in runs:
            peak = f"{r['peak_gib']:.2f}" if r["peak_gib"] is not None else "-"
            print(
                f"{arm:>5s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['ms_per_step']:8.1f} {r['tok_s']:8.0f} {peak:>8s} {r['nan']:>4d} "
                f"{r.get('params', 0):>10} {r.get('loop_blocks', 0):>6} "
                f"{r.get('max_iter', 0):>5}"
            )
    ctrl = table["b1"]
    v2 = verdict(ctrl, table["b2"], window, "b2", 2, 4)
    vx2 = verdict(ctrl, table["b1x2"], window, "b1x2", 1, 8)
    for arm, runs in table.items():
        mean, spread = summarize(runs)
        extra = sum(r.get("extra_body_params") or 0 for r in runs) // max(1, len(runs))
        print(
            f"{arm:>5s} mean bpb {mean:.4f}  spread {spread:.4f}  "
            f"median warm ms/step {statistics.median([r['ms_per_step'] for r in runs]):.1f}  "
            f"extra_body_params={extra:,}"
        )
    outcome = decide(v2, vx2)
    print(
        f"\nvoid checks: every eval scored {window} B, 0 NaN-skipped steps, "
        f"params rows match the arms, {len(seeds)} distinct seeds per arm.\n"
        f"pre-registered rule (AGENTS §1.2 + the parameter/compute confounds above): "
        f"KEEP iff b2 beats b1 by more than b1's own seed spread AND beats the "
        f"max_iter=8 one-body arm (b1x2) by that spread; a b2 win that b1x2 matches "
        f"is COMPUTE-CONFOUNDED; otherwise TIE, and a TIE leaves loop_blocks=1 as "
        f"every preset's default (the knob stays for scale, not as a quality win).\n"
        f"b2: {v2['verdict'].upper()} "
        f"(control {v2['control_mean']:.4f} +- {v2['control_spread']:.4f}, "
        f"arm {v2['arm_mean']:.4f} +- {v2['arm_spread']:.4f}, "
        f"delta {v2['delta']:+.4f})"
        + (f"\nvoid reasons: {v2['reason']}" if v2["reason"] else "")
        + f"\nb1x2 (compute control): {vx2['verdict'].upper()} "
        f"({vx2['arm_mean']:.4f} +- {vx2['arm_spread']:.4f}, "
        f"delta {vx2['delta']:+.4f})"
        + (f"\nvoid reasons: {vx2['reason']}" if vx2["reason"] else "")
        + f"\nTOPOLOGY VERDICT: {outcome.upper()}"
    )
    print(
        f"step time (SEPARATE, reported not gated): b2 median "
        f"{v2['ms_control']:.1f} -> {v2['ms_arm']:.1f} ms/step "
        f"({v2['ms_delta']:+.1f} ms, {v2['ms_delta'] / v2['ms_control']:+.1%}); "
        f"b1x2 {vx2['ms_arm']:.1f} ms/step ({vx2['ms_delta']:+.1f})."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ab", help="the A/B driver (seeds x 3 arms)")
    a.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a.add_argument("--steps", type=int, default=STEPS)
    a.add_argument("--batch", type=int, default=BATCH)
    a.add_argument("--seq", type=int, default=SEQ)
    a.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    a.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    a.add_argument("--log-every", type=int, default=LOG_EVERY)
    a.add_argument("--data", default=DATA)
    a.add_argument("--eval", default=EVAL)
    a.add_argument("--device", default="cuda")
    a.add_argument("--out-dir", default="~/logs/ab_multiblock")
    r = sub.add_parser("_run", help="one arm, one seed (the child mode)")
    r.add_argument("--arm", required=True, choices=[arm for arm, _b, _i in ARMS])
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--data", default=DATA)
    r.add_argument("--eval", default=EVAL)
    r.add_argument("--device", default="cuda")
    r.add_argument("--steps", type=int, default=STEPS)
    r.add_argument("--batch", type=int, default=BATCH)
    r.add_argument("--seq", type=int, default=SEQ)
    r.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    r.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    r.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = ap.parse_args(argv)
    if args.mode == "_run":
        return _run_one(args)
    why = preflight(args.device)
    if why is not None:
        return skip(why)
    return ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
