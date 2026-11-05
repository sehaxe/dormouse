"""The optimizer-region A/B: bf16 state, bf16 NS, ns_steps 5 vs 3, chunk cap.

    # the 100-step CPU gate (no card; prints loss delta, seed spread, param ulp)
    uv run python scripts/ab_opt_region.py gate --steps 100

    # the card A/B at the next checkpoint gap (quality + warm tok/s)
    uv run --extra gpu python scripts/ab_opt_region.py ab \
        --arms base,state_bf16,ns3 --steps 2000 --preset m400 --batch 8 --seq 1024

**The four arms**, one factor each against `base` (the clean-main recipe):

* `base`          — `muon_state_dtype=fp32`, `ns_dtype=fp32`, `ns_steps=5`,
                    `muon_chunk_elems=1<<25`; every number on record.
* `state_bf16`    — the 2026-11-05 preset recipe: bf16 momentum state, same NS
                    chain, `muon_chunk_elems=1<<26` (a bit-identical split).
* `ns3`           — `state_bf16` with `ns_steps=3`: the measured-parity arm.
                    Compare it to `state_bf16`, NOT to `base` (two factors).
* `state_ns_bf16` — `state_bf16` plus the bf16 NS chain (`ns_dtype=bf16`), if
                    the card run wants to price the second precision rung.

`state_bf16` vs `ns3` is the 5-vs-3 parity gate: same data, same seed, same
state/chunk recipe, one iteration-count knob. The driver prints both arms'
held-out BPB, the per-arm seed range, and the delta. The pre-registered rule is
the repo's (AGENTS 1.2 / AB-PROTOCOL): keep `ns3` iff its mean BPB beats
`state_bf16`'s by more than `state_bf16`'s own seed spread; a tie deletes the
arm, not the default. The book on 3-vs-5 is a tie prediction (the cited 3-seed
study is non-monotone with a <0.003 BPB spread), so a tie here is the expected
outcome and the win is the 40% shorter chain.

**Protocol.** The task's shape: `m400`, b8 x s1024, 2 000 steps, 1 seed minimum
(`--seeds 1,2,3` for a verdict), one eval window on every arm. Each arm-seed is
a separate PROCESS (a fresh optimizer/model), writes its own JSONL and
`<name>.config.toml`. `grad_clip`, the NaN firewall, the EMA and the checkpoint
cadence are the run's own; `ckpt_every` is pinned away from any write.

**Safety.** The `ab` mode needs the card and refuses while another process holds
it (`scripts/_lane.preflight`). The `gate` mode is CPU-only and touches no
card.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight, skip  # noqa: E402

DATA = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"
PRESET = "m400"
STEPS = 2000
BATCH = 8
SEQ = 1024
LR = 1e-3
SEEDS = (1,)
EVAL_BATCHES = 20
EVAL_EVERY = 250
LOG_EVERY = 50

#: `(state_dtype, ns_dtype, ns_steps, chunk_elems)`.
ARMS: dict[str, tuple[str, str, int, int]] = {
    "base": ("fp32", "fp32", 5, 1 << 25),
    "state_bf16": ("bf16", "fp32", 5, 1 << 26),
    "ns3": ("bf16", "fp32", 3, 1 << 26),
    "state_ns_bf16": ("bf16", "bf16", 5, 1 << 26),
}


def _run_overrides(arm: str, args: argparse.Namespace) -> dict:
    state, ns, steps, chunk = ARMS[arm]
    return {
        "batch": args.batch,
        "seq_len": args.seq,
        "lr": LR,
        "steps": args.steps,
        "seed": args.seed,
        "eval": True,
        "eval_every": args.eval_every,
        "eval_batches": args.eval_batches,
        "log_every": args.log_every,
        "ckpt_every": 10**9,
        "out_dir": str(Path(args.out).parent),
        "ckpt_name": f"{args.arm}_s{args.seed}",
        "timers": False,
        "muon_state_dtype": state,
        "ns_dtype": ns,
        "ns_steps": steps,
        "muon_chunk_elems": chunk,
        "json_log": str(args.out),
    }


def _run_one(args: argparse.Namespace) -> int:
    """One arm, one seed, one process: the trainer with the arm made live."""
    import dormouse.train.loop as train_loop
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train import train

    run = load_run(args.preset, train_overrides=_run_overrides(args.arm, args))
    run.train.data = args.data
    if not Path(run.train.data).is_file():
        print(f"no data file at {run.train.data}: pass --data")
        return 1
    Path(args.out).with_suffix(".config.toml").write_text(run.snapshot_toml())

    real = train_loop.MixOptimizer
    built: list = []

    def spy(model, **kw):
        # The wiring check, the same shape `ab_muon_variants.py` uses: a config
        # key the call site does not forward is INERT and the table would be a
        # table of the control.
        kw.setdefault("muon_state_dtype", ARMS[args.arm][0])
        kw.setdefault("ns_dtype", ARMS[args.arm][1])
        kw.setdefault("ns_steps", ARMS[args.arm][2])
        kw.setdefault("muon_chunk_elems", ARMS[args.arm][3])
        opt = real(model, **kw)
        built.append(opt)
        return opt

    train_loop.MixOptimizer = spy
    device = torch.device(args.device)
    model = DormouseModel(run.model, device=device)
    print(
        f"[run] arm={args.arm} state={ARMS[args.arm][0]} ns={ARMS[args.arm][1]} "
        f"ns_steps={ARMS[args.arm][2]} chunk={ARMS[args.arm][3]} seed={args.seed} "
        f"steps={args.steps} b{args.batch} s{args.seq}",
        flush=True,
    )
    train(model, run, device=device, log=print)
    assert built and built[0].muon is not None, "no Muon group was built"
    assert built[0].muon.state_dtype == ARMS[args.arm][0], "state dtype did not reach Muon"
    assert built[0].muon.ns_dtype == ARMS[args.arm][1], "ns dtype did not reach Muon"
    assert built[0].muon.chunk_elems == ARMS[args.arm][3], "chunk cap did not reach Muon"
    assert built[0].muon.param_groups[0]["ns_steps"] == ARMS[args.arm][2], (
        "ns_steps did not reach the Muon param group"
    )
    return 0


def _read(path: Path, steps: int) -> dict:
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
        "nan": sum(r.get("nan_skipped", 0) for r in logs),
    }


def ab(args: argparse.Namespace) -> int:
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        print(f"unknown arm(s) {bad}; known {sorted(ARMS)}")
        return 2
    seeds = tuple(int(s) for s in args.seeds.split(","))
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    window = args.eval_batches * args.batch * args.seq
    print(
        f"optimizer-region A/B: {len(arms)} arms x {len(seeds)} seeds x {args.steps} "
        f"steps, {args.preset} b{args.batch} s{args.seq} lr={LR}, window {window} B. "
        f"Output: {out_dir}"
    )
    for arm in arms:
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
                "--preset",
                args.preset,
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

    table = {
        arm: [{"seed": s, **_read(out_dir / f"{arm}_s{s}.jsonl", args.steps)} for s in seeds]
        for arm in arms
    }
    print(
        f"\n{'arm':>15s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} "
        f"{'warm tok/s':>10s} {'nan':>4s}"
    )
    for arm, runs in table.items():
        for r in runs:
            print(
                f"{arm:>15s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['tok_s']:10.0f} {r['nan']:>4d}"
            )
    summary = {}
    for arm, runs in table.items():
        bpbs = [r["bpb"] for r in runs]
        summary[arm] = (statistics.mean(bpbs), max(bpbs) - min(bpbs))
        print(
            f"{arm:>15s} mean bpb {summary[arm][0]:.4f}  spread {summary[arm][1]:.4f}  "
            f"median warm tok/s {statistics.median([r['tok_s'] for r in runs]):.0f}"
        )
    for arm, runs in table.items():
        for r in runs:
            assert r["scored"] == window, f"{arm} s{r['seed']}: window {r['scored']} != {window}"
            assert r["nan"] == 0, f"{arm} s{r['seed']}: {r['nan']} NaN-skipped steps"
    if "state_bf16" in summary and "ns3" in summary:
        delta = summary["ns3"][0] - summary["state_bf16"][0]
        verdict = (
            "TIE within the control spread"
            if abs(delta) <= summary["state_bf16"][1]
            else "SEPARATED"
        )
        print(
            f"\n5-vs-3 parity: ns3 - state_bf16 = {delta:+.4f} BPB "
            f"(state_bf16's seed spread {summary['state_bf16'][1]:.4f}). "
            f"{verdict} — the driver prints both; it does not apply the verdict."
        )
    return 0


# --- the CPU gate ------------------------------------------------------------


def _gate_curve(preset: str, rung: str, seed: int, steps: int, batch: int, seq: int):
    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train.optim import MUON_CHUNK_ELEMS, MixOptimizer, clip_grads

    torch.manual_seed(seed)
    run = load_run(
        preset,
        train_overrides={
            "steps": steps,
            "batch": batch,
            "seq_len": seq,
            "muon_state_dtype": rung,
            "ns_dtype": "fp32",
            "ns_steps": 5,
            "muon_chunk_elems": MUON_CHUNK_ELEMS,
        },
    )
    model = DormouseModel(run.model, device=torch.device("cpu")).train()
    opt = MixOptimizer(
        model,
        lr=1e-3,
        wd=0.0,
        ns_steps=5,
        muon_state_dtype=rung,
        ns_dtype="fp32",
        muon_chunk_elems=MUON_CHUNK_ELEMS,
    )
    gen = torch.Generator().manual_seed(1234)
    losses = []
    for _ in range(steps):
        # A fresh batch every step (a fixed one hits ~0 loss and the relative
        # gate becomes 0/0 noise); both arms see the same stream.
        x = torch.randint(0, 256, (batch, seq), generator=gen)
        y = torch.randint(0, 256, (batch, seq), generator=gen)
        opt.zero_grad()
        _logits, loss, _counters = model(x, y)
        loss.backward()
        if run.train.grad_clip > 0:
            clip_grads(model.parameters(), run.train.grad_clip)
        opt.step()
        losses.append(float(loss.detach()))
    return losses, {n: p.detach().clone() for n, p in model.named_parameters()}


def gate(args: argparse.Namespace) -> int:
    fp32, fp32_params = _gate_curve(args.preset, "fp32", 0, args.steps, args.batch, args.seq)
    bf16, bf16_params = _gate_curve(args.preset, "bf16", 0, args.steps, args.batch, args.seq)
    fp32_other, _ = _gate_curve(args.preset, "fp32", 1, args.steps, args.batch, args.seq)
    loss_rel = max(abs(a - b) / max(abs(a), 1e-12) for a, b in zip(fp32, bf16, strict=True))
    seed_spread = max(
        abs(a - b) / max(abs(a), 1e-12) for a, b in zip(fp32, fp32_other, strict=True)
    )
    eps = torch.finfo(torch.bfloat16).eps
    tiny = torch.finfo(torch.bfloat16).tiny
    param_ulp = max(
        float((p - bf16_params[n]).abs().max()) / (eps * max(float(p.abs().max()), tiny))
        for n, p in fp32_params.items()
    )
    print(
        f"\n[gate] {args.preset} {args.steps} steps: "
        f"max rel loss {loss_rel:.6g}; fp32 seed spread {seed_spread:.6g}; "
        f"worst parameter delta {param_ulp:.4g} bf16 ulp at scale"
    )
    print(f"[gate] losses fp32[0,-1]={fp32[0]:.6f},{fp32[-1]:.6f}  "
          f"bf16[0,-1]={bf16[0]:.6f},{bf16[-1]:.6f}")
    if loss_rel < seed_spread:
        print("[gate] VERDICT: the state rung is inside the fp32 seed spread (a quality tie).")
        return 0
    print("[gate] VERDICT: the state rung moved the curve further than a seed change — NOT a tie.")
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("gate", help="the 100-step CPU fp32-vs-bf16-state gate")
    g.add_argument("--preset", default="tiny")
    g.add_argument("--steps", type=int, default=100)
    g.add_argument("--batch", type=int, default=2)
    g.add_argument("--seq", type=int, default=64)

    a = sub.add_parser("ab", help="the card A/B driver")
    a.add_argument("--arms", default="base,state_bf16,ns3")
    a.add_argument("--preset", default=PRESET)
    a.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a.add_argument("--steps", type=int, default=STEPS)
    a.add_argument("--batch", type=int, default=BATCH)
    a.add_argument("--seq", type=int, default=SEQ)
    a.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    a.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    a.add_argument("--log-every", type=int, default=LOG_EVERY)
    a.add_argument("--data", default=DATA)
    a.add_argument("--device", default="cuda")
    a.add_argument("--out-dir", default="~/logs/ab_opt_region")

    r = sub.add_parser("_run", help="one arm, one seed (the child mode)")
    r.add_argument("--arm", required=True, choices=sorted(ARMS))
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--preset", default=PRESET)
    r.add_argument("--data", default=DATA)
    r.add_argument("--device", default="cuda")
    r.add_argument("--steps", type=int, default=STEPS)
    r.add_argument("--batch", type=int, default=BATCH)
    r.add_argument("--seq", type=int, default=SEQ)
    r.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    r.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    r.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = ap.parse_args(argv)
    if args.mode == "gate":
        return gate(args)
    if args.mode == "_run":
        return _run_one(args)
    why = preflight(args.device)
    if why is not None:
        return skip(why)
    return ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
