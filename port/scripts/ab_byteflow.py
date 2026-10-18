"""The ByteFlow A/B: plain byte model (control) vs the byteflow channel, 3 seeds x 2 arms.

    uv run python scripts/ab_byteflow.py ab              # 3 seeds x 2 arms, 2000 steps
    uv run python scripts/ab_byteflow.py ab --seeds 1    # one seed, dry check

**THIS DRIVER HAS NOT BEEN RUN.** The card is held by a training run (AGENTS
§1.5: one heavy thing at a time), so every number below is unmeasured, and the
pre-registered rule is written here BEFORE the first run so the first run cannot
move the goalposts. A skipped preflight prints a SKIP banner and no table.

**Arms** — one flag differs (`use_byteflow`), everything else is shared:

* `control` — the plain byte stack: `use_byteflow=false` at the same geometry,
  so the loop runs over T=512 byte positions. `use_kda=true` (the preset's own
  arm), `bf16=false`, `use_engram=false` — the last two are the loud refusals
  the byteflow arm carries (`config._validate_byteflow`), and running the control
  in bf16 against an fp32-only arm would compare dtypes, not mechanisms.
* `channel` — Option B (`byteflow.py`): `use_byteflow=true` with `use_kda=true`
  arms `BfChannel` (`compat_armed`), the front stage compresses the 512-byte
  window into K=128 patch latents, THIS model's loop runs on the 128 patches,
  and the back stage lifts them back to 512 byte logits. The net's own global
  blocks are built empty (`g_layers=0`) — the loop IS the global stage.

**The exact legal combinations, and the refusals found** (read off
`config._validate_byteflow` and `model.DormouseModel`):

* `use_byteflow=true` REFUSES `bf16=true`, `use_engram=true`, `sigreg_weight>0`,
  `dspark_weight>0`, `segment_len>0`, `max_seq_len>byteflow_max_bytes`,
  `byteflow_k_tokens>max_seq_len`, `byteflow_k_tokens<1`, `byteflow_eps2<=0`,
  `byteflow_bins<1`, a width that does not tile its head count, and
  `--engram-ram` — each with its own message. The smoke's geometry below keeps
  every one of them legal.
* Channel mode (Option B) is legal iff ANY of `use_kda` / `use_attnres` /
  `moe_topk>0` is on (`use_engram` cannot be, above). Standalone net mode
  (Option A) is legal iff ALL of them are off — a third legal arm this driver
  does NOT run: Option A has no loop at all, so it is a different comparison
  (the net alone vs the stack), not the compression A/B.
* The geometry is the smoke's (`scripts/smoke_byteflow.py`, verified against
  `model.py`'s dispatch): `small` at d_model 256 / n_heads 4, byteflow widths
  d_local 96 / d_global 256 / E=2 / w_local 256 / K=128 over a 512-byte window —
  burn's `byteflow_9m` build (schema.rs:751-794).

**Pre-registered prediction.** The channel arm LOSES on held-out BPB at 2k
steps, and the reason is in the record (`byteflow.py`'s docstring): the burn A/B
measured the compression itself as the win at equal params (canon 3.466 vs plain
bytes 3.947 BPB), while the full-arm dormouse `small` lost at 6.498
(`docs/reviews/byteflow-ab-2026-10-02.md`); and the probe (`port/scripts/patch_rate.py`)
shows why this polygon is the weak form of the bet — at K=128 over T=512 the
mean patch is T/K = 4 bytes, the SAME as BPE's ~4 bytes/token, so the loop's
128 patch positions buy no sequence compression, while the channel pays for a
second encoder/decoder stack the control does not have. A WIN here would be the
first positive reading for the arm, and it would have to beat the control's own
seed spread to count (AGENTS §1.2).

**Protocol** (`docs/protocols/AB-PROTOCOL.md`): 3 seeds per arm, 2000 steps,
`small` + the geometry above, b4 x s512, lr 1e-3, pure CE (aux is refused with
byteflow anyway), and **one eval window of 40 960 B on every arm and seed**
(`eval_batches 20 x batch 4 x seq_len 512`). Each arm-seed is a SEPARATE
PROCESS, because 3 seeds is 3 initialisations. Output: one JSONL per run under
`~/logs/ab_byteflow/`, plus the run's own `<name>.config.toml` beside it (a
number needs its config, AGENTS §1.4), plus two driver rows per run:

* `event=peak` — `torch.cuda.max_memory_allocated()` after the run, GiB (only on
  a CUDA device; on CPU there is no row and the table prints `-`);
* `event=patch` — the run's own patch-length distribution, measured by wrapping
  the channel's `encode_chunks` and reading its `sel` output (mean/median/p90,
  histogram) — the mechanism's compute economics, not a proxy.

**What the driver does NOT do:** apply the verdict. It prints the table, the
void checks and the pre-registered rule (AGENTS §1.2: keep the arm iff its mean
BPB beats the control's mean by more than the CONTROL's own seed spread; a tie
deletes the mechanism; a void run is not a number). A BPB is a measurement, not
a verdict.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

# `scripts/` is not a package: the script's own directory must be importable
# for the probe's statistics. APPENDED, not inserted: a script name at the
# FRONT of `sys.path` would shadow a package of the same name (this file is
# also loaded by `tests/test_ab_byteflow.py`, inside the pytest process).
_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.append(_SCRIPTS)
from patch_rate import patch_lengths, patch_stats  # noqa: E402

#: The port's data on this box (AGENTS §2.6) — the LOCAL copy, so the driver
#: does not depend on the /mnt drive being mounted.
DATA = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"

SEEDS = (1, 2, 3)
STEPS = 2000
#: b4 s512 is the only shape this arm has ever run (the 145.8 ms/step smoke,
#: `byteflow.py`'s docstring); both arms share it, so the eval window matches.
BATCH = 4
SEQ = 512
LR = 1e-3
EVAL_BATCHES = 20
EVAL_EVERY = 200
LOG_EVERY = 50
PRESET = "small"

#: The A/B polygon's geometry, shared by both arms so `use_byteflow` is the only
#: flag that differs (`scripts/smoke_byteflow.py`; burn's `byteflow_9m` widths).
GEOM = dict(
    use_engram=False,
    bf16=False,
    d_model=256,
    n_heads=4,
    byteflow_k_tokens=128,
    byteflow_d_local=96,
    byteflow_d_global=256,
    byteflow_heads_local=4,
    byteflow_heads_global=4,
    byteflow_e_layers=2,
    byteflow_w_local=256,
    byteflow_d_ff_local=256,
    byteflow_max_bytes=512,
)

#: `(arm name, use_byteflow)`. The control is first so the table reads against it.
ARMS = (
    ("control", False),
    ("channel", True),
)

#: The counter each arm MUST show, and the one it must NOT (`counters.get`,
#: 0 when absent). A run whose counters disagree with its arm is VOID: an arm
#: that runs and trains nothing is the defect class the counters exist for.
EXPECT = {
    "control": {"forward": ">0", "bf_channel": "==0", "bf_net": "==0"},
    "channel": {"forward": ">0", "bf_channel": ">0", "bf_net": "==0"},
}


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
    """`None` when the card is ours, else the reason to skip (ab_muon_variants')."""
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


def read_run(path: Path, steps: int) -> dict:
    """One run's JSONL -> the numbers the verdict table needs.

    `ms/step` and `tok/s` come from the JSONL's own `tokens_per_s` windows (the
    trainer's rate instrument; `timers` is OFF in the child for exactly this
    reason — `timer.report()` resets the mark the telemetry reads). `tok/s` is
    the median of the run's SECOND HALF — the protocol's ">=50 warm steps" read
    as "past the warmup window". `nan` is the last log row's `nan_skipped` (read
    with `max`): the trainer keeps it as a RUNNING total, so summing the rows
    would multiply it.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    run = next((r for r in rows if r.get("event") == "run"), None)
    assert run is not None, f"{path}: no run row — the JSONL is not this driver's"
    evals = [r for r in rows if r.get("event") == "eval"]
    logs = [r for r in rows if r.get("event") == "log"]
    assert evals, f"{path}: no eval row — nothing to compare"
    warm = [
        r["tokens_per_s"] for r in logs if r.get("step", 0) > steps / 2 and r.get("tokens_per_s")
    ]
    rate = statistics.median(warm) if warm else float("nan")
    per_step = run["batch"] * run["seq_len"]
    counters: dict[str, int] = {}
    for r in logs:
        for k, v in (r.get("counters") or {}).items():
            counters[k] = counters.get(k, 0) + v
    return {
        "bpb": evals[-1]["bpb"],
        "scored": evals[-1]["scored"],
        "n_evals": len(evals),
        "tok_s": rate,
        "ms_per_step": (per_step / rate * 1e3) if rate == rate and rate > 0 else float("nan"),
        "nan": max((r.get("nan_skipped", 0) for r in logs), default=0),
        "counters": counters,
        "peak_gib": next((r["peak_gib"] for r in rows if r.get("event") == "peak"), None),
        "patch": next((r for r in rows if r.get("event") == "patch"), None),
    }


def summarize(runs: list[dict]) -> tuple[float, float]:
    """Mean BPB and the seed spread (max − min) of one arm."""
    bpbs = [r["bpb"] for r in runs]
    return statistics.mean(bpbs), max(bpbs) - min(bpbs)


def check(runs: list[dict], window: int, expect: dict[str, str]) -> list[str]:
    """The void checks: a number that fails one is not a number."""
    why: list[str] = []
    for r in runs:
        if r["scored"] != window:
            why.append(f"seed {r['seed']}: eval window {r['scored']} != {window} B")
        if r["nan"]:
            why.append(f"seed {r['seed']}: {r['nan']} NaN-skipped steps")
        for name, want in expect.items():
            got = r["counters"].get(name, 0)
            if want == ">0" and got <= 0:
                why.append(f"seed {r['seed']}: {name}={got} — the arm did not run")
            if want == "==0" and got != 0:
                why.append(f"seed {r['seed']}: {name}={got} — the wrong shape ran")
    return why


def verdict(control: list[dict], arm: list[dict], window: int) -> dict:
    """The pre-registered rule (AGENTS §1.2) over one control and one arm.

    VOID when any run fails a void check (window, NaN, counters). Otherwise WIN
    iff the arm's mean BPB beats the control's mean by MORE than the CONTROL's
    own seed spread; anything else — including a tie — deletes the mechanism.
    """
    why = check(control, window, EXPECT["control"]) + check(arm, window, EXPECT["channel"])
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
    if why:
        out["verdict"] = "void"
        out["reason"] = "; ".join(why)
    elif amean < cmean - cspread:
        out["verdict"] = "win"
    else:
        out["verdict"] = "tie"
    return out


def _run_one(args: argparse.Namespace) -> int:
    """One arm, one seed, one process: the trainer with the flag made live."""
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train import train

    run = load_run(
        PRESET,
        model_overrides={**GEOM, "use_byteflow": args.arm == "channel"},
        train_overrides={
            "batch": args.batch,
            "seq_len": args.seq,
            "lr": LR,
            "steps": args.steps,
            "seed": args.seed,
            "eval": True,
            "eval_every": args.eval_every,
            "eval_batches": args.eval_batches,
            "log_every": args.log_every,
            "ckpt_every": 10**9,  # the A/B scores held-out BPB; no weights are written
            "out_dir": str(Path(args.out).parent),
            "ckpt_name": f"{args.arm}_s{args.seed}",
            # The JSONL's `tokens_per_s` is this driver's speed instrument and
            # `timers` OFF is what keeps it honest: `timer.report()` resets the
            # mark the telemetry row reads (loop.py's own comment).
            "timers": False,
            "json_log": str(args.out),
        },
    )
    run.train.data = args.data
    if not Path(run.train.data).is_file():
        print(f"no data file at {run.train.data}: pass --data")
        return 1
    # The run's own config record, next to its JSONL.
    Path(args.out).with_suffix(".config.toml").write_text(run.snapshot_toml())

    device = torch.device(args.device)
    model = DormouseModel(run.model, device=device)
    assert (model.bf is not None) == (args.arm == "channel"), (
        "the arm's dispatch does not match its flag: model.bf is built iff "
        "use_byteflow arms the channel"
    )
    # The patch statistics: wrap the channel's OWN chunker and read its `sel`.
    # One stash per forward, summarized after the run — the number the compute
    # economics turn on, measured on the arm that ran, not on a probe.
    sels: list[torch.Tensor] = []
    if model.bf is not None:
        real = model.bf.net.encode_chunks

        def spy(bytes_: torch.Tensor, k: int):
            h, z, sel = real(bytes_, k)
            sels.append(sel.detach().to("cpu"))
            return h, z, sel

        model.bf.net.encode_chunks = spy  # type: ignore[method-assign]

    window = args.eval_batches * args.batch * args.seq
    print(
        f"[run] arm={args.arm} use_byteflow={run.model.use_byteflow} seed={args.seed} "
        f"steps={args.steps} b{args.batch} s{args.seq} lr={LR} window={window} B "
        f"d_model={run.model.d_model} K={run.model.byteflow_k_tokens} "
        f"bf_channel={int(model.bf is not None)} "
        f"params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train(model, run, device=device, log=print)

    def _row(**kw) -> None:
        with open(args.out, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(kw) + "\n")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        _row(event="peak", step=args.steps, peak_gib=peak)
        print(f"[peak] {peak:.2f} GiB (torch.cuda.max_memory_allocated)", flush=True)
    if sels:
        lengths = torch.cat([patch_lengths(s, args.seq).reshape(-1) for s in sels])
        stats = patch_stats(lengths)
        _row(event="patch", step=args.steps, forwards=len(sels), **stats)
        print(
            f"[patch] {stats['n']:,} patches over {len(sels)} forwards: "
            f"mean {stats['mean']:.3f} median {stats['median']:.1f} p90 {stats['p90']:.1f} "
            f"histogram {stats['histogram']}",
            flush=True,
        )
    return 0


def ab(args: argparse.Namespace) -> int:
    """The driver: `--seeds` x 2 arms, separate processes, then the table."""
    seeds = tuple(int(s) for s in args.seeds.split(","))
    window = args.eval_batches * args.batch * args.seq
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"A/B: {len(ARMS)} arms x {len(seeds)} seeds x {args.steps} steps, {PRESET} "
        f"b{args.batch} s{args.seq} lr={LR}, window {window} B. Output: {out_dir}\n"
        f"pre-registered: the channel arm is predicted to LOSE or TIE at this polygon "
        f"(mean patch = T/K = {args.seq}/{GEOM['byteflow_k_tokens']} = "
        f"{args.seq / GEOM['byteflow_k_tokens']:.1f} bytes, BPE's own ~4)."
    )
    for arm, _flag in ARMS:
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

    table = {arm: [] for arm, _f in ARMS}
    for arm, _flag in ARMS:
        for seed in seeds:
            table[arm].append(
                {"seed": seed, **read_run(out_dir / f"{arm}_s{seed}.jsonl", args.steps)}
            )
    print(
        f"\n{'arm':>8s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'ms/step':>8s} "
        f"{'tok/s':>8s} {'peak GiB':>8s} {'nan':>4s} {'patch med':>9s}"
    )
    for arm, runs in table.items():
        for r in runs:
            med = r["patch"]["median"] if r["patch"] else float("nan")
            peak = f"{r['peak_gib']:.2f}" if r["peak_gib"] is not None else "-"
            print(
                f"{arm:>8s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['ms_per_step']:8.1f} {r['tok_s']:8.0f} {peak:>8s} {r['nan']:>4d} "
                f"{med:9.1f}"
            )
    for arm, runs in table.items():
        mean, spread = summarize(runs)
        print(
            f"{arm:>8s} mean bpb {mean:.4f}  spread {spread:.4f}  "
            f"median warm ms/step {statistics.median([r['ms_per_step'] for r in runs]):.1f}  "
            f"bf_channel={sum(r['counters'].get('bf_channel', 0) for r in runs)}"
        )
    v = verdict(table["control"], table["channel"], window)
    print(
        f"\nvoid checks: every eval scored {window} B, 0 NaN-skipped steps, "
        f"control bf_channel==0, channel bf_channel>0, {len(seeds)} distinct seeds per arm.\n"
        f"pre-registered rule (AGENTS §1.2): keep the arm iff its mean held-out BPB beats "
        f"the control's mean by more than the CONTROL's own seed spread; a tie deletes the "
        f"mechanism. Verdict: {v['verdict'].upper()} "
        f"(control {v['control_mean']:.4f} +- {v['control_spread']:.4f}, "
        f"channel {v['arm_mean']:.4f} +- {v['arm_spread']:.4f}, "
        f"delta {v['delta']:+.4f})" + (f"\nvoid reasons: {v['reason']}" if v["reason"] else "")
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
    a.add_argument("--out-dir", default="~/logs/ab_byteflow")
    r = sub.add_parser("_run", help="one arm, one seed (the child mode)")
    r.add_argument("--arm", required=True, choices=[arm for arm, _f in ARMS])
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--data", default=DATA)
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
