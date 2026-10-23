"""The CSA A/B: KDA control vs compressed-selectable KV memory, 3 seeds x 600.

    uv run python scripts/ab_csa.py ab            # 4 arms x 3 seeds, 600 steps
    uv run python scripts/ab_csa.py longctx --control-dir DIR --control off_s1 \\
        --csa-dir DIR --csa csa8+coarse256_s1   # 4096 and 16384-byte windows

**THIS DRIVER HAS NOT BEEN RUN.** The card is free, but the brief forbids
training arms; every number below is unmeasured and the pre-registered rule is
written BEFORE the first run. A skipped preflight prints a SKIP banner.

**Arms** — one flag family differs, everything else is `ab_mhc.py`'s COMMON
(with ONE deviation, named: `mix_plan` is dropped because the brief names one
data file, exactly as `ab_attn.py` does, and `ckpt_every=STEPS` so `longctx`
has a checkpoint to load):

* `off` — `use_csa=false`, the KDA control;
* `csa8` — `use_csa=true, csa_chunk=8, csa_kv_dim=64, csa_topk=32` (FINE only);
* `csa8+coarse256` — the same fine level plus the COARSE dense level
  (`csa_coarse_chunk=256, csa_coarse_kv_dim=64`): 4096 dense entries at 1M,
  1.0 MiB of extra state, 0.52M MAC/byte at 1M;
* `csa16` — `use_csa=true, csa_chunk=16`, same width/top-k.

`medium`, b8 x s512, 600 steps, 3 seeds, one 81 920 B eval window (20 x 8 x 512)
on every arm and seed, JSONL to `~/logs/ab_csa/`.

**Pre-registered prediction.** On the 512-byte training window every arm is
predicted to **TIE** (`off`), for two reasons, both checkable before the run:

1. the branch sees only *strictly earlier* chunks; at `csa_chunk=8` a position
   in the first eight bytes has no earlier chunk at all, and the deepest chunk
   of a 512-byte window has at most 63 candidates against the top-32 cut — the
   window is already mostly reachable to KDA at this length;
2. the long-range detail this arm exists for is beyond 512 bytes, so a 512-byte
   held-out BPB cannot price it. The 4096/16384-byte `longctx` table is the
   primary read for the mechanism; the 600-step A/B is the cost/slope filter.

The verdict rule is AGENTS §1.2, applied to every arm vs the control: **WIN**
iff the arm's mean held-out BPB beats the control's mean by more than the
**control's own seed spread**; a tie (including a longctx win with no 512-byte
win) does NOT land a default — it lands the `--longctx` claim only, and the
call is reported, not silently rounded.

**`longctx`** evaluates each checkpoint on held-out **4096 AND 16384-byte**
windows, and — for a coarse checkpoint — with the coarse level ON and OFF
(`coarse_enabled`), so the coarse level's long-context contribution is visible
at both lengths rather than inferred from one. It prints BPB per (window,
variant) and the delta against the control; the rule is the same (beat the
control's own window spread). `segment_len=1024` is used for the control; the
CSA arm runs `segment_len=0` because `config.validate` refuses the pair loudly
(the boundary rule for a half-full chunk is undecided) — the driver prints that
deviation rather than hiding it.
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
from ab_byteflow import preflight, read_run, skip, summarize  # noqa: E402
from ab_mhc import COMMON_MODEL, COMMON_TRAIN  # noqa: E402

#: The brief's data (AGENTS §2.6, the LOCAL copy) and its held-out file.
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
LR = 3e-4  # TrainConfig's default

#: `ab_mhc.py`'s COMMON, minus `mix_plan` (it would make `--data` inert) and
#: plus a checkpoint at the end so `longctx` can load the arm it scored.
COMMON = dict(COMMON_TRAIN)
COMMON.pop("mix_plan", None)
COMMON["ckpt_every"] = STEPS

#: One entry per arm; the control is first so the table reads against it.
ARMS = (
    ("off", {"use_csa": False}),
    ("csa8", {"use_csa": True, "csa_chunk": 8}),
    (
        "csa8+coarse256",
        {"use_csa": True, "csa_chunk": 8, "csa_coarse_chunk": 256},
    ),
    ("csa16", {"use_csa": True, "csa_chunk": 16}),
)

#: The counter each arm MUST show (and the control must NOT).
EXPECT = {
    "off": {"iter": ">0", "csa_forward": "==0", "csa_coarse_forward": "==0"},
    "csa8": {"iter": ">0", "csa_forward": ">0", "csa_coarse_forward": "==0"},
    "csa8+coarse256": {
        "iter": ">0",
        "csa_forward": ">0",
        "csa_coarse_forward": ">0",
    },
    "csa16": {"iter": ">0", "csa_forward": ">0", "csa_coarse_forward": "==0"},
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
            got = r["counters"].get(name, 0)
            if want == ">0" and got <= 0:
                why.append(f"seed {r['seed']}: {name}={got} — the arm did not run")
            if want == "==0" and got != 0:
                why.append(f"seed {r['seed']}: {name}={got} — the wrong arm ran")
    return why


def verdict(control: list[dict], arm: list[dict], window: int, name: str) -> dict:
    """AGENTS §1.2 over one control and one arm, plus the reported step delta."""
    why = check(control, window, "off") + check(arm, window, name)
    cmean, cspread = summarize(control)
    amean, aspread = summarize(arm)
    out = {
        "control_mean": cmean,
        "control_spread": cspread,
        "arm_mean": amean,
        "arm_spread": aspread,
        "delta": amean - cmean,
        "ms_control": statistics.median([r["ms_per_step"] for r in control]),
        "ms_arm": statistics.median([r["ms_per_step"] for r in arm]),
        "reason": "",
    }
    out["ms_delta"] = out["ms_arm"] - out["ms_control"]
    if why:
        out["verdict"] = "void"
        out["reason"] = "; ".join(why)
    elif amean < cmean - cspread:
        out["verdict"] = "win"
    else:
        out["verdict"] = "tie"
    return out


def _run_one(args: argparse.Namespace) -> int:
    """One arm, one seed, one process: the trainer with the arm's flags live."""
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train import train

    flag = dict(ARMS_DICT[args.arm])
    run = load_run(
        PRESET,
        model_overrides={**COMMON_MODEL, **flag},
        train_overrides={
            **COMMON,
            "seed": args.seed,
            "lr": LR,
            "ckpt_every": args.steps,
            "out_dir": str(Path(args.out).parent / "ckpts"),
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
    import torch

    model = DormouseModel(run.model, device=torch.device(device))
    is_on = model.loop.csa is not None
    assert is_on == (args.arm != "off"), (
        "the arm's dispatch does not match its flag: loop.csa is built iff use_csa"
    )
    if is_on:
        assert model.loop.csa.chunk == run.model.csa_chunk, model.loop.csa.chunk
        assert model.loop.csa.coarse_chunk == run.model.csa_coarse_chunk, (
            model.loop.csa.coarse_chunk
        )
    csa_params = sum(p.numel() for p in model.loop.csa.parameters()) if is_on else 0
    total = sum(p.numel() for p in model.parameters())
    window = args.eval_batches * args.batch * args.seq
    print(
        f"[run] arm={args.arm} use_csa={run.model.use_csa} "
        f"chunk={run.model.csa_chunk} topk={run.model.csa_topk} "
        f"coarse_chunk={run.model.csa_coarse_chunk} "
        f"coarse_kv={run.model.csa_coarse_kv_dim} seed={args.seed} "
        f"steps={args.steps} b{args.batch} s{args.seq} lr={LR} window={window} B "
        f"params={total:,} csa_params={csa_params:,} ({csa_params / total:.4%})",
        flush=True,
    )
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train(model, run, device=torch.device(device), log=print)

    def _row(**kw) -> None:
        with open(args.out, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(kw) + "\n")

    params = {"params": total, "csa_params": csa_params}
    if torch.device(device).type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        params["peak_gib"] = peak
        print(f"[peak] {peak:.2f} GiB (torch.cuda.max_memory_allocated)", flush=True)
    _row(event="params", step=args.steps, **params)
    return 0


def ab(args: argparse.Namespace) -> int:
    seeds = tuple(int(s) for s in args.seeds.split(","))
    window = args.eval_batches * args.batch * args.seq
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"A/B: {len(ARMS)} arms x {len(seeds)} seeds x {args.steps} steps, {PRESET} "
        f"b{args.batch} s{args.seq} lr={LR}, window {window} B. Output: {out_dir}\n"
        f"pre-registered: the 512-byte window TIES (strictly-earlier chunks cap the "
        f"reachable past at 63 chunks, and the detail this arm adds lives beyond 512 B); "
        f"the 4096-byte longctx table is the primary read, the A/B is the cost filter."
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

    table = {arm: [] for arm, _f in ARMS}
    for arm, _flag in ARMS:
        for seed in seeds:
            row = {"seed": seed, **read_run(out_dir / f"{arm}_s{seed}.jsonl", args.steps)}
            for line in (out_dir / f"{arm}_s{seed}.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r.get("event") == "params":
                    row.update(params=r.get("params"), csa_params=r.get("csa_params"))
            table[arm].append(row)
    print(
        f"\n{'arm':>5s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'ms/step':>8s} "
        f"{'tok/s':>8s} {'peak GiB':>8s} {'nan':>4s} {'params':>10s}"
    )
    for arm, runs in table.items():
        for r in runs:
            peak = f"{r['peak_gib']:.2f}" if r["peak_gib"] is not None else "-"
            print(
                f"{arm:>5s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['ms_per_step']:8.1f} {r['tok_s']:8.0f} {peak:>8s} {r['nan']:>4d} "
                f"{r.get('params', '-'):>10}"
            )
    for arm, runs in table.items():
        mean, spread = summarize(runs)
        csa = sum(r["counters"].get("csa_forward", 0) for r in runs)
        print(
            f"{arm:>5s} mean bpb {mean:.4f}  spread {spread:.4f}  "
            f"median warm ms/step {statistics.median([r['ms_per_step'] for r in runs]):.1f}  "
            f"csa_forward={csa}"
        )
    for arm, _flag in ARMS[1:]:
        v = verdict(table["off"], table[arm], window, arm)
        print(
            f"\narm {arm}: {v['verdict'].upper()} "
            f"(control {v['control_mean']:.4f} +- {v['control_spread']:.4f}, "
            f"arm {v['arm_mean']:.4f} +- {v['arm_spread']:.4f}, "
            f"delta {v['delta']:+.4f}, step {v['ms_control']:.1f} -> "
            f"{v['ms_arm']:.1f} ms {v['ms_delta']:+.1f})"
            + (f"\nvoid reasons: {v['reason']}" if v["reason"] else "")
        )
    print(
        "\npre-registered rule: keep a DEFAULT only if an arm beats the control's "
        "mean by more than the CONTROL's own seed spread on this 512-byte window; a "
        "tie deletes the default. A longctx-only win is a long-context claim, not a "
        "default change — run `longctx` on the same checkpoints before believing it."
    )
    return 0


def _longctx_bpb(model, eval_path: str, window: int, batches: int, seg: int, device):
    """Held-out BPB over `batches` windows of `window` bytes (batch 1)."""
    import torch

    from dormouse.train import bpb
    from dormouse.train.data import ByteStream

    stream = ByteStream(eval_path, batch=1, seq_len=window)
    model.eval()
    total = 0.0
    segment = seg if model.loop.csa is None else 0
    with torch.no_grad():
        for _ in range(batches):
            x, y = stream.next_batch()
            x, y = x.to(device), y.to(device)
            _, loss, _ = model(x, y, pure=True, segment_len=segment)
            assert loss is not None
            total += float(loss)
    return bpb(total / max(batches, 1)), segment


def _score_windows(model, eval_path, windows, batches, seg, device):
    """BPB per window per active level; a coarse model is scored ON and OFF."""
    csa = model.loop.csa
    has_coarse = csa is not None and getattr(csa, "coarse_chunk", 0) > 0
    variants = [("coarse", True), ("fine", False)] if has_coarse else [("csa", None)]
    out: dict[tuple[int, str], tuple[float, int]] = {}
    for window in windows:
        for label, enabled in variants:
            if enabled is not None:
                csa.coarse_enabled = enabled
            score, segment = _longctx_bpb(model, eval_path, window, batches, seg, device)
            out[(window, label)] = (score, segment)
    return out


def longctx(args: argparse.Namespace) -> int:
    """Held-out BPB at 4096 AND 16384 bytes: control vs CSA, coarse ON and OFF."""
    import torch

    from dormouse.serve import load_checkpoint

    windows = tuple(int(w) for w in args.windows.split(",") if w.strip())
    assert windows, "--windows is empty"
    dev = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    print(
        f"longctx: windows={windows} B, {args.batches} held-out windows each, "
        f"segment_len={args.seg} (control) / 0 (CSA, the pair is refused), device={dev}"
    )
    rows: dict[str, dict[tuple[int, str], tuple[float, int]]] = {}
    labels: dict[str, tuple[str, ...]] = {}
    for name, base, ckpt in (
        ("off", args.control_dir, args.control),
        ("csa", args.csa_dir, args.csa),
    ):
        model, run, loaded = load_checkpoint(Path(base).expanduser(), ckpt, str(dev))
        rows[name] = _score_windows(
            model, args.eval, windows, args.batches, args.seg, loaded
        )
        labels[name] = tuple(sorted({label for _w, label in rows[name]}))
        print(
            f"{name}: use_csa={run.model.use_csa}, chunk={run.model.csa_chunk}, "
            f"topk={run.model.csa_topk}, "
            f"coarse_chunk={run.model.csa_coarse_chunk}, "
            f"coarse_kv={run.model.csa_coarse_kv_dim}, variants={labels[name]}"
        )
        del model
    print(
        f"\n{'window':>7s} {'model':>4s} {'variant':>7s} {'bpb':>8s} {'seg':>4s} "
        f"{'scored B':>9s}"
    )
    for window in windows:
        for name in ("off", "csa"):
            for (w, label), (score, segment) in sorted(rows[name].items()):
                if w != window:
                    continue
                print(
                    f"{window:>7d} {name:>4s} {label:>7s} {score:8.4f} "
                    f"{segment:>4d} {args.batches * window:>9d}"
                )
    print()
    for window in windows:
        control = rows["off"][(window, "csa")][0]
        for label in labels["csa"]:
            delta = rows["csa"][(window, label)][0] - control
            print(
                f"window {window:>5d} csa/{label:<6s} - off: {delta:+.4f} BPB "
                f"({'CSA better' if delta < 0 else 'control better'})"
            )
    print(
        "\npre-registered: the coarse level is a long-context claim only if it "
        "beats the control by more than the control's own window spread, AND the "
        "same checkpoint's coarse row beats its fine-only row at the same window; "
        "one checkpoint cannot meet a seed-spread rule, so this table is REPORTED, "
        "not a verdict — run the A/B's 3 seeds to judge it."
    )
    return 0


ARMS_DICT = dict(ARMS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ab", help="the A/B driver (seeds x arms)")
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
    a.add_argument("--out-dir", default="~/logs/ab_csa")
    lc = sub.add_parser(
        "longctx", help="4096 and 16384-byte held-out BPB, control vs CSA"
    )
    lc.add_argument("--control-dir", required=True)
    lc.add_argument("--control", required=True)
    lc.add_argument("--csa-dir", required=True)
    lc.add_argument("--csa", required=True)
    lc.add_argument("--eval", default=EVAL)
    lc.add_argument("--windows", default="4096,16384")
    lc.add_argument("--batches", type=int, default=4)
    lc.add_argument("--seg", type=int, default=1024)
    lc.add_argument("--device", default="cuda")
    r = sub.add_parser("_run", help="one arm, one seed (the child mode)")
    r.add_argument("--arm", required=True, choices=[arm for arm, _f in ARMS])
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
    if args.mode == "longctx":
        return longctx(args)
    why = preflight(args.device)
    if why is not None:
        return skip(why)
    return ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
