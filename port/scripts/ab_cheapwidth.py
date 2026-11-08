"""The cheap-width A/B: dense control vs latent / GQA / low-rank readout.

    uv run python scripts/ab_cheapwidth.py ab              # 3 seeds x 5 arms, 600 steps
    uv run python scripts/ab_cheapwidth.py ab --profile    # ms/step + tok/s, no training
    uv run python scripts/ab_cheapwidth.py ab --seeds 1    # one seed, dry check

**THIS DRIVER HAS NOT BEEN RUN.** The card is held by a training run (AGENTS
§1.5: one heavy thing at a time), so every number below is unmeasured and the
pre-registered rule is written BEFORE the first run. A skipped preflight prints
a SKIP banner and no table.

**Arms** — every arm is `medium` with the battery's COMMON, and the ONLY keys
that differ are the three width flags:

* `dense` — all three off (the control; every number on record).
* `latent` — `attn_latent_rank = d/4 = 256`: one `d→r` projection, four `r→d`
  ups for q/k/v/g.
* `gqa` — `n_kv_heads = 2` at `n_heads = 16`: k/v project to `2·64 = 128`
  channels and expand `repeat_interleave(8)` before the kernel.
* `latent+gqa` — both; the k/v ups are then `r → 128`.
* `all` — both plus `out_proj_rank = d/4 = 256`.

The predicted dense bill at this shape (`iters=3`, `f=d_ffn=2730`,
`head_dim=64`), from `dormouse.width`:

    dense      24.12M MAC/token   cut  0.0%
    latent     15.46M             cut 35.9%
    gqa        18.61M             cut 22.8%
    latent+gqa 14.09M             cut 41.6%
    all        12.52M             cut 48.1%

**One deviation from `ab_mhc`'s COMMON, and why**: `use_tsct = false`. The
default TSCT readout is ALREADY a rank-64 factorization (`2·d·64` MAC), so
`out_proj_rank = 256` would GROW it; the flag names the dense `d²` readout, and
this A/B prices exactly that. Every arm, control included, runs the dense
readout, so the comparison is still one variable per row. (The attention flags
do not depend on the rung: their projections are dense `nn.Linear`s either way.)

**Data**: the mix4 `web.bin` corpus and its own held-out `web_eval.bin`, one
81 920 B eval window (20 x 8 x 512) on every arm and seed. The eval/log cadence
is scaled to the 600-step budget (300/100 instead of the battery's 500/250); the
window itself is the battery's, so BPB stays comparable.

**Pre-registered prediction** (600 steps, 3 seeds, short on purpose):

* `gqa` ADOPTS: 22.8% of the dense bill removed for a k/v projection that is
  still 2 heads wide, and GQA is a published quality-neutral change at this
  scale (Ainslie et al. 2023). It is the safest flag.
* `latent` and `latent+gqa` ADOPT: a rank-256 latent against `d=1024` keeps a
  quarter of the width and is the inference-cheapness MLA is known for; at 600
  steps its BPB should sit inside the control's seed spread.
* `all` is the ARM MOST LIKELY TO LOSE: it stacks the two compressions with a
  rank-256 readout on a 600-step budget. If it loses, the follow-up is
  `latent+gqa` at `out_proj_rank = 0`, not a different corpus.

**The verdict, pre-registered before the first run** (this is the rule the
driver applies, not a description of a result):

1. **VOID** if any run has a NaN-skipped step, an eval window that is not
   `eval_batches x batch x seq_len`, or fewer than the requested seeds.
2. **LOSS** if the arm's mean held-out BPB is worse than the control's mean by
   more than `max(control seed spread, 0.01)` BPB.
3. **TIE** if the arm is quality-safe but its predicted MAC/token cut is below
   **20%** — the mechanism is then not the lever and the wiring is not worth it.
4. **ADOPT** otherwise: quality inside the control's own spread and a >= 20%
   dense-MAC cut. `ms/step` and `tok/s` are REPORTED beside the BPB verdict
   (review §6.1's separate verdict), never folded into it.

Output: one JSONL per run under `~/logs/ab_cheapwidth/`, the run's own
`<name>.config.toml` beside it (a number needs its config, AGENTS §1.4), and a
driver `event=params` row with the instantiated count and the arm's delta.
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

from dormouse.width import heads, macs_per_token, param_delta  # noqa: E402

#: The port's data on this box (AGENTS §2.6) — the LOCAL mix4 copy.
DATA = "/home/sehaxe/dormouse/data_local/mix4/web.bin"
EVAL = "/home/sehaxe/dormouse/data_local/mix4/web_eval.bin"

SEEDS = (1, 2, 3)
STEPS = 600
PRESET = "medium"
BATCH = 8
SEQ = 512
EVAL_BATCHES = 20
EVAL_EVERY = 300
LOG_EVERY = 100
LR = 3e-4  # TrainConfig's default; the battery does not override it

WINDOW = EVAL_BATCHES * BATCH * SEQ
D = 1024  # medium
H = heads(D)

#: `/tmp/opencode/battery.sh`'s COMMON, as `load_run` keyword arguments, plus
#: the one documented deviation below (`use_tsct=false` — the dense readout the
#: `out_proj_rank` arm prices).
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
    mix_plan="configs/mix_big.toml",
)

#: The model-side COMMON (`--set` flags in the battery) plus the arm controls.
#: `use_tsct=false` is the deviation named in the module docstring; it is in
#: EVERY arm, so it cannot be attributed to any single comparison.
COMMON_MODEL = dict(
    moe_kernel="triton",
    moe_capacity=0.0,
    bf16=True,
    use_tsct=False,
    use_attnres=False,
    loom=False,
)

#: `(arm name, the three width keys)`. `dense` first: the table reads against it.
ARMS = (
    ("dense", {"attn_latent_rank": 0, "n_kv_heads": 0, "out_proj_rank": 0}),
    ("latent", {"attn_latent_rank": D // 4, "n_kv_heads": 0, "out_proj_rank": 0}),
    ("gqa", {"attn_latent_rank": 0, "n_kv_heads": 2, "out_proj_rank": 0}),
    ("latent+gqa", {"attn_latent_rank": D // 4, "n_kv_heads": 2, "out_proj_rank": 0}),
    ("all", {"attn_latent_rank": D // 4, "n_kv_heads": 2, "out_proj_rank": D // 4}),
)

#: The pure-arithmetic prediction each row is judged against.
BASE_MACS = macs_per_token(D, iters=3, n_heads=H)
MAC_CUT = {
    arm: 1.0 - macs_per_token(D, iters=3, n_heads=H, **flags) / BASE_MACS for arm, flags in ARMS
}
PARAMS_DELTA = {arm: param_delta(D, n_heads=H, **flags) for arm, flags in ARMS}


def check(runs: list[dict], window: int) -> list[str]:
    """The void checks: a number that fails one is not a number."""
    why: list[str] = []
    if len(runs) < 2:
        why.append(f"{len(runs)} seed(s): a spread needs at least two")
    for r in runs:
        if r["scored"] != window:
            why.append(f"seed {r['seed']}: eval window {r['scored']} != {window} B")
        if r["nan"]:
            why.append(f"seed {r['seed']}: {r['nan']} NaN-skipped steps")
    return why


def verdict(control: list[dict], arm: list[dict], window: int, mac_cut: float) -> dict:
    """The pre-registered rule: quality inside the control's spread AND >= 20% cut.

    VOID / LOSS / TIE / ADOPT, in that order. The 0.01 BPB floor on the quality
    window is there because a 3-seed spread can be zero on a deterministic run,
    and a zero-width window would call any noise a loss.
    """
    why = check(control, window) + check(arm, window)
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
        "mac_cut": mac_cut,
        "ms_control": cms,
        "ms_arm": ams,
        "ms_delta": ams - cms,
        "reason": "",
    }
    if why:
        out["verdict"] = "void"
        out["reason"] = "; ".join(why)
    elif amean > cmean + max(cspread, 0.01):
        out["verdict"] = "loss"
        out["reason"] = (
            f"mean BPB {amean:.4f} is worse than the control {cmean:.4f} by more "
            f"than max(spread {cspread:.4f}, 0.01)"
        )
    elif mac_cut < 0.20:
        out["verdict"] = "tie"
        out["reason"] = f"MAC cut {mac_cut:.1%} < the pre-registered 20%"
    else:
        out["verdict"] = "adopt"
    return out


def _run_one(args: argparse.Namespace) -> int:
    """One arm, one seed, one process: the trainer with the flags made live."""
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.model.tsct import DenseLowRank
    from dormouse.train import train

    arm = dict(ARMS)[args.arm]
    run = load_run(
        PRESET,
        model_overrides={**COMMON_MODEL, **arm},
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
    # The structural proof that the arm's dispatch matches its flag (there is
    # no per-flag counter: the flags change a projection's SHAPE, and the shape
    # is what these asserts read).
    attn = model.loop.attn
    assert (attn.latent_down is not None) == bool(arm["attn_latent_rank"]), arm
    assert attn.kv_heads == (arm["n_kv_heads"] or run.model.n_heads), arm
    if arm["out_proj_rank"]:
        if run.model.use_tsct:
            assert model.loop.out_proj.rank == arm["out_proj_rank"]
        else:
            assert isinstance(model.loop.out_proj, DenseLowRank)
            assert model.loop.out_proj.rank == arm["out_proj_rank"]
    total = sum(p.numel() for p in model.parameters())
    want_delta = PARAMS_DELTA[args.arm]
    print(
        f"[run] arm={args.arm} seed={args.seed} steps={args.steps} "
        f"b{args.batch} s{args.seq} lr={LR} window={WINDOW} B params={total:,} "
        f"delta_vs_dense_predicted={want_delta:+,d} mac_cut={MAC_CUT[args.arm]:.1%}",
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
        "params_delta_predicted": want_delta,
        "mac_per_token": BASE_MACS * (1.0 - MAC_CUT[args.arm]),
        "mac_cut": MAC_CUT[args.arm],
    }
    if torch.device(device).type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        params["peak_gib"] = peak
        print(f"[peak] {peak:.2f} GiB (torch.cuda.max_memory_allocated)", flush=True)
    _row(event="params", step=args.steps, **params)
    return 0


def profile(args: argparse.Namespace) -> int:
    """`--profile`: ms/step and tok/s per arm, forward+backward+step, no training.

    A SHAPE measurement, not a quality one: synthetic ids of `b x s`, 10 warmup
    steps, 20 timed, median. It answers "does the cheaper projection actually
    buy wall clock at this width", which the BPB table cannot.
    """
    import torch

    from dormouse.config import load_run
    from dormouse.model import DormouseModel

    if not torch.cuda.is_available():
        print("SKIP: --profile needs CUDA (the trainer's real device); 0 arms timed.")
        return 0
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(1)
    ids = torch.randint(0, 256, (args.batch, args.seq), generator=g, device=dev)
    print(f"profile: {PRESET} b{args.batch} s{args.seq}, 10 warmup + 20 timed, ms/step")
    print(f"{'arm':>10s} {'ms/step':>9s} {'tok/s':>10s} {'vs dense':>9s} {'mac cut':>8s}")
    base_ms = None
    for name, flags in ARMS:
        run = load_run(PRESET, model_overrides={**COMMON_MODEL, **flags})
        model = DormouseModel(run.model, device=dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        tokens = args.batch * args.seq

        def step(model=model, opt=opt) -> None:
            _, loss, _ = model(ids, ids)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        for _ in range(10):
            step()
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        ms = statistics.median(times)
        if base_ms is None:
            base_ms = ms
        print(
            f"{name:>10s} {ms:9.2f} {tokens / (ms / 1e3):10.0f} "
            f"{ms / base_ms:9.2%} {MAC_CUT[name]:8.1%}"
        )
        del model, opt
        torch.cuda.empty_cache()
    return 0


def ab(args: argparse.Namespace) -> int:
    """The driver: `--seeds` x 5 arms, separate processes, then the table."""
    seeds = tuple(int(s) for s in args.seeds.split(","))
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"A/B: {len(ARMS)} arms x {len(seeds)} seeds x {args.steps} steps, {PRESET} "
        f"b{args.batch} s{args.seq} lr={LR}, window {WINDOW} B. Output: {out_dir}\n"
        f"pre-registered: gqa, latent and latent+gqa ADOPT (quality inside the "
        f"control's spread at 600 steps); `all` is the arm most likely to LOSE "
        f"(three compressions at once). The rule: ADOPT iff mean held-out BPB is "
        f"within max(control seed spread, 0.01) AND the predicted dense MAC cut "
        f"is >= 20%."
    )
    for arm, _flags in ARMS:
        for seed in seeds:
            out = out_dir / f"{arm.replace('+', '_')}_s{seed}.jsonl"
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

    table: dict[str, list[dict]] = {arm: [] for arm, _f in ARMS}
    for arm, _flags in ARMS:
        for seed in seeds:
            stem = arm.replace("+", "_")
            row = {"seed": seed, **read_run(out_dir / f"{stem}_s{seed}.jsonl", args.steps)}
            for line in (out_dir / f"{stem}_s{seed}.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r.get("event") == "params":
                    row.update(params=r.get("params"), params_delta=r.get("params_delta_predicted"))
            table[arm].append(row)
    print(
        f"\n{'arm':>10s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'ms/step':>8s} "
        f"{'tok/s':>8s} {'nan':>4s} {'params':>10s} {'mac cut':>8s}"
    )
    for arm, runs in table.items():
        for r in runs:
            print(
                f"{arm:>10s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['ms_per_step']:8.1f} {r['tok_s']:8.0f} {r['nan']:>4d} "
                f"{r.get('params', '-'):>10} {MAC_CUT[arm]:8.1%}"
            )
    for arm, runs in table.items():
        mean, spread = summarize(runs)
        print(
            f"{arm:>10s} mean bpb {mean:.4f}  spread {spread:.4f}  "
            f"median warm ms/step {statistics.median([r['ms_per_step'] for r in runs]):.1f}  "
            f"predicted delta params {PARAMS_DELTA[arm]:+,d}"
        )
    print("\nverdicts (dense is the control):")
    control = table["dense"]
    for arm, _flags in ARMS[1:]:
        v = verdict(control, table[arm], WINDOW, MAC_CUT[arm])
        print(
            f"  {arm:>10s}: {v['verdict'].upper():>6s} "
            f"(bpb {v['arm_mean']:.4f} +- {v['arm_spread']:.4f} vs "
            f"{v['control_mean']:.4f} +- {v['control_spread']:.4f}, "
            f"delta {v['delta']:+.4f}; mac cut {v['mac_cut']:.1%}; "
            f"ms/step {v['ms_control']:.1f} -> {v['ms_arm']:.1f} "
            f"{v['ms_delta']:+.1f})" + (f" — {v['reason']}" if v["reason"] else "")
        )
    print(
        "\nms/step is the SEPARATE verdict (review §6.1): reported, never folded into the BPB rule."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ab", help="the A/B driver (seeds x 5 arms)")
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
    a.add_argument("--out-dir", default="~/logs/ab_cheapwidth")
    a.add_argument("--profile", action="store_true", help="ms/step + tok/s per arm, no training")
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
    why = preflight(args.device)
    if why is not None:
        return skip(why)
    if getattr(args, "profile", False):
        return profile(args)
    return ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
