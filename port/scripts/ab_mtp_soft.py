"""The Soft-MTP A/B: `ntp` (control) vs `hard_mtp` vs `soft_mtp`, 3 seeds x 600.

    uv run --extra gpu python scripts/ab_mtp_soft.py ab
    uv run --extra gpu python scripts/ab_mtp_soft.py ab --probe   # + per-head CE / acceptance
    uv run --extra gpu python scripts/ab_mtp_soft.py probe --ckpt-dir ... --ckpt-name ...

**THIS DRIVER HAS NOT BEEN RUN.** The card is held (AGENTS §1.5: one heavy
thing at a time), so the keep rule is written here BEFORE the first run. A busy
card makes the driver REFUSE and exit 1; nothing is measured and no table
prints.

**The arms** — one model field differs (`mtp_soft`), everything else is shared:

* `ntp`      — `mtp_k=0, mtp_weight=0.0`: the control, no MTP head at all.
* `hard_mtp` — `mtp_k=4, mtp_weight=1.0, mtp_soft=false`: DeepSeek-V3 §3.2
  future-byte CE (`model/mtp.py`, `forward`).
* `soft_mtp` — `mtp_k=4, mtp_weight=1.0, mtp_soft=true`: the same CE VALUE, with
  the NTP CE's gradient at each head's ALIGNED position routed into the head
  (`forward_soft`, arXiv:2610.38331). The total loss value is bit-identical to
  `hard_mtp` (the coupling is `x - x`); only the backward differs.

K=4 because the probe reports acceptance for k=1..4 and one K makes the arms a
single knob. λ=1.0 for both MTP arms, the value `ab_mtp.py` uses, so the
per-head scale is the same one its gates pin.

**The recipe** is `/tmp/opencode/battery.sh`'s `BASE` + `COMMON` (the same
fields every other port A/B uses), with `mix_v7` as the train plan and the
shared `eval_mix.bin` as the held-out file: `medium`, b8 x s512, **600 steps**,
`tf32`, `ns_kind=triton`, `opt_graph`, `moe_kernel=triton`, `moe_capacity=0`,
`engram_ram`, `bf16`, eval every 200 over `eval_batches=20` — every arm and
seed scores the SAME window, `20 x 8 x 512 = 81 920 B` (AGENTS §2.6). The only
additions are the run's own `seed`/`mtp_*`/`ckpt_*`/`json-log`. `ckpt_every`
is `steps`, so the `--probe` pass has a final checkpoint to load; without
`--probe` the checkpoint is still written (600 steps is one checkpoint, not a
cadence).

**The pre-registered keep rule** (the reason this file exists before a run):

    soft_mtp ENTERS the recipe iff ALL THREE hold
      1. mean held-out BPB(soft_mtp) < mean held-out BPB(hard_mtp)   (strictly),
      2. mean held-out BPB(soft_mtp) <= mean BPB(ntp) + 2 * control_spread,
         where control_spread = max(ntp seeds) - min(ntp seeds),
      3. mean draft acceptance over k=1..4 is HIGHER for soft_mtp than for
         hard_mtp (the `--probe` measurement; without it the arm is VOID).

Anything else is a DROP. The second clause is the "no worse than the control
beyond twice its own seed noise" guard: a tie with `ntp` on BPB is not a reason
to carry a head, and a 2x-spread deficit is a regression even if it edges out
`hard_mtp`. The mechanism is deleted if it cannot clear all three (AGENTS
§1.2); it is not kept "to be safe".

VOID (not a number) when an eval window is not 81 920 B, a NaN-skipped step
occurred, `mtp=` is absent on a hard/soft run or present on `ntp`, or the
acceptance probe did not run on the two MTP arms. A VOID arm is not a keep.

**What the driver does NOT do:** apply the verdict, tune λ/K per seed, or read
the training `loss=` as a quality number (it carries `λ·MTP` by construction).
Held-out BPB is the only comparison, and the eval forward runs `pure=True`
(`model.py`), so both sides are the predictive CE.

**The probe** (written, not run here): loads the final checkpoint, runs the
eval batches with `breakdown=True` (the MTP block active), and reports
`per_head_ce[k]` plus two cheap draft diagnostics on the real next byte:
`accept[k] = mean min(1, p_ntp(x)/q_mtp(x))` (the speculative acceptance
probability for the observed byte) and `greedy[k] = argmax(q_mtp) == x`. It is
the model's OWN heads, not the paper's sampling harness; the label says so.
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
from _lane import preflight as _lane_preflight  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRESET = "medium"
SEEDS = (1, 2, 3)
STEPS = 600
BATCH = 8
SEQ = 512
EVAL_BATCHES = 20
EVAL_EVERY = 200
MTP_K = 4
MTP_WEIGHT = 1.0
WINDOW = EVAL_BATCHES * BATCH * SEQ

#: `(arm, mtp_k, mtp_weight, mtp_soft)`.
ARMS = (
    ("ntp", 0, 0.0, False),
    ("hard_mtp", MTP_K, MTP_WEIGHT, False),
    ("soft_mtp", MTP_K, MTP_WEIGHT, True),
)

#: The v7 mix plan's train side; `data` is still passed because the eval half's
#: stream builder is shared (the trainer opens `eval_data` explicitly).
DATA = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"
EVAL_DATA = "/home/sehaxe/dormouse/data_local/eval_mix/eval_mix.bin"
MIX_PLAN = "configs/mix_v7.toml"
BASE = ("--device", "cuda", "--data", DATA, "--eval", EVAL_DATA)
COMMON = (
    "--set",
    f"steps={STEPS}",
    "--set",
    f"batch={BATCH}",
    "--set",
    f"seq_len={SEQ}",
    "--set",
    "eval=true",
    "--set",
    f"eval_every={EVAL_EVERY}",
    "--set",
    f"eval_batches={EVAL_BATCHES}",
    "--set",
    "log_every=100",
    "--set",
    f"ckpt_every={STEPS}",
    "--set",
    "ckpt_name=mtpsoft",
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
    f"mix_plan={MIX_PLAN}",
)


def preflight(device: str) -> str | None:
    """The card, then the two files this driver names (the cheaper blocker)."""
    why = _lane_preflight(device)
    if why is not None:
        return why
    return next((f"no data file at {p}" for p in (DATA, EVAL_DATA) if not Path(p).is_file()), None)


def refuse(why: str) -> int:
    print(
        f"\n=== REFUSED: {why}\n"
        f"=== ONE GPU PROCESS AT A TIME (AGENTS §1.5): nothing was measured and no\n"
        f"=== table is printed. Re-run when the card is free.\n",
        flush=True,
    )
    return 1


# --- the A/B's pure logic ---------------------------------------------------


def read_run(jsonl: Path, log: Path) -> dict:
    """One run -> BPB, its byte window, NaN count, and the `mtp=` evidence."""
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    evals = [r for r in rows if r.get("event") == "eval"]
    logs = [r for r in rows if r.get("event") == "log"]
    assert evals, f"{jsonl}: no eval row — nothing to compare"
    assert logs, f"{jsonl}: no log row — nothing to compare"
    text = log.read_text() if log.is_file() else ""
    return {
        "bpb": evals[-1]["bpb"],
        "scored": evals[-1]["scored"],
        "n_evals": len(evals),
        "nan": max((r.get("nan_skipped", 0) for r in logs), default=0),
        "mtp_logged": text.count(" mtp="),
    }


def summarize(runs: list[dict]) -> tuple[float, float]:
    bpbs = [r["bpb"] for r in runs]
    return statistics.mean(bpbs), max(bpbs) - min(bpbs)


def check(runs: list[dict], window: int, *, mtp: bool) -> list[str]:
    why: list[str] = []
    for r in runs:
        if r["scored"] != window:
            why.append(f"seed {r['seed']}: eval window {r['scored']} != {window} B")
        if r["nan"]:
            why.append(f"seed {r['seed']}: {r['nan']} NaN-skipped steps")
        if mtp and not r["mtp_logged"]:
            why.append(f"seed {r['seed']}: no `mtp=` on any log line — the arm did not run")
        if not mtp and r["mtp_logged"]:
            why.append(f"seed {r['seed']}: `mtp=` on ntp — the control ran the head")
    return why


def verdict(
    ntp: list[dict],
    hard: list[dict],
    soft: list[dict],
    window: int,
    *,
    accept_hard: float | None,
    accept_soft: float | None,
) -> dict:
    """The pre-registered three-clause rule in the module docstring."""
    why = (
        check(ntp, window, mtp=False)
        + check(hard, window, mtp=True)
        + check(soft, window, mtp=True)
    )
    if accept_hard is None or accept_soft is None:
        why.append("no acceptance probe on the two MTP arms — clause 3 cannot be read")
    nmean, nspread = summarize(ntp)
    hmean, _hspread = summarize(hard)
    smean, _sspread = summarize(soft)
    out = {
        "ntp_mean": nmean,
        "control_spread": nspread,
        "hard_mean": hmean,
        "soft_mean": smean,
        "soft_minus_hard": smean - hmean,
        "soft_minus_ntp": smean - nmean,
        "accept_hard": accept_hard,
        "accept_soft": accept_soft,
        "reason": "",
    }
    if why:
        out["verdict"] = "void"
        out["reason"] = "; ".join(why)
    elif (
        smean < hmean
        and smean <= nmean + 2.0 * nspread
        and accept_soft is not None
        and accept_hard is not None
        and accept_soft > accept_hard
    ):
        out["verdict"] = "keep"
    else:
        out["verdict"] = "drop"
    return out


# --- the train half ---------------------------------------------------------


def run_one(
    arm: str,
    k: int,
    weight: float,
    soft: bool,
    seed: int,
    out_dir: Path,
    device: str,
    steps: int,
) -> int:
    """One arm, one seed, one CLI process — the battery's own launch shape."""
    name = f"mtpsoft_{arm}_s{seed}"
    jsonl = out_dir / f"{arm}_s{seed}.jsonl"
    log = out_dir / f"{arm}_s{seed}.log"
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
        f"mtp_k={k}",
        "--set",
        f"mtp_weight={weight}",
        "--set",
        f"mtp_soft={'true' if soft else 'false'}",
        "--set",
        f"ckpt_name={name}",
        "--set",
        f"out_dir={out_dir / 'ckpt'}",
        "--json-log",
        str(jsonl),
    ]
    if device != "cuda":
        cmd[cmd.index("--device") + 1] = device
    print(f"\n=== {arm} seed {seed} -> {jsonl}", flush=True)
    with open(log, "w", encoding="utf-8") as sink:
        done = subprocess.run(cmd, cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT)
    if done.returncode != 0:
        print(f"the {arm} run (seed {seed}) exited {done.returncode}: no table. Log: {log}")
    return done.returncode


# --- the probe (written, not run here) --------------------------------------


def probe(ckpt_dir: Path, ckpt_name: str, device: str) -> dict:
    """Per-head CE and draft acceptance on the held-out window.

    Loads the final checkpoint through the ONE loader (`load_model_state`), runs
    the model's TRAIN forward (`pure=False`, MTP active) under `no_grad` so the
    MTP block runs, and reads `last_breakdown["mtp_per_k"]` (a probe; the
    per-forward `.item()` ban is a training-path rule).
    """
    import torch

    from dormouse.cli import _snapshot_run
    from dormouse.model import DormouseModel
    from dormouse.train.data import open_streams
    from dormouse.train.loop import load_model_half, load_model_state

    run = _snapshot_run(ckpt_dir, ckpt_name)
    model = DormouseModel(run.model).to(device)
    load_model_state(model, run.train, load_model_half(ckpt_dir / f"{ckpt_name}.pt"))
    model.eval()
    if model.mtp is None:
        return {"per_head_ce": [], "accept": [], "greedy": [], "accept_mean": None}
    xs: list[torch.Tensor] = []

    def keep(_m, _inp, out):
        xs.append(out.detach())

    handles = [p.register_forward_hook(keep) for p in model.mtp.projs]
    _train, ev = open_streams(run, eval_only=True)
    assert ev is not None, "eval_data is set but open_streams returned no eval stream"
    ev.rewind()
    per_ce: list[list[float]] = [[] for _ in range(model.mtp.k)]
    per_acc: list[list[float]] = [[] for _ in range(model.mtp.k)]
    per_greedy: list[list[float]] = [[] for _ in range(model.mtp.k)]
    with torch.no_grad():
        for _ in range(run.train.eval_batches):
            x, y = ev.next_batch()
            x, y = x.to(device), y.to(device)
            xs.clear()
            logits, _loss, _counters = model(x, y, breakdown=True)
            bd = getattr(model, "last_breakdown", None) or {}
            for j, value in enumerate(bd.get("mtp_per_k", [])):
                per_ce[j].append(float(value))
            p_ntp = torch.softmax(logits.reshape(x.shape[0], x.shape[1], -1), dim=-1)
            for j, z in enumerate(xs):
                shift = j + 1
                q = torch.softmax(model.lm_head(z.reshape(-1, z.shape[-1])), dim=-1)
                n = z.shape[1]
                q = q.reshape(x.shape[0], n, -1)
                p = p_ntp[:, shift : shift + n]
                target = y[:, shift : shift + n]
                p_x = p.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                q_x = q.gather(-1, target.unsqueeze(-1)).squeeze(-1).clamp_min(1e-30)
                per_acc[j].append(float(torch.minimum(torch.ones_like(p_x), p_x / q_x).mean()))
                per_greedy[j].append(float((q.argmax(-1) == target).float().mean()))
    for handle in handles:
        handle.remove()
    acc = [statistics.mean(v) if v else float("nan") for v in per_acc]
    return {
        "per_head_ce": [statistics.mean(v) if v else float("nan") for v in per_ce],
        "accept": acc,
        "greedy": [statistics.mean(v) if v else float("nan") for v in per_greedy],
        "accept_mean": statistics.mean(acc) if acc else None,
    }


# --- the driver -------------------------------------------------------------


def ab(args: argparse.Namespace) -> int:
    seeds = tuple(int(s) for s in args.seeds.split(","))
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Soft-MTP A/B: 3 arms x {len(seeds)} seeds x {args.steps} steps, {PRESET} "
        f"b{BATCH} s{SEQ}, window {WINDOW} B, K={MTP_K}, λ={MTP_WEIGHT}. "
        f"Output: {out_dir}\n"
        f"pre-registered: KEEP iff BPB(soft) < BPB(hard) AND BPB(soft) <= "
        f"BPB(ntp) + 2*control_spread AND acceptance(soft) > acceptance(hard); "
        f"otherwise DROP. A VOID arm is not a keep. The probe is "
        f"{'ON' if args.probe else 'OFF (a keep is impossible without it)'}."
    )
    for arm, k, weight, soft in ARMS:
        for seed in seeds:
            rc = run_one(arm, k, weight, soft, seed, out_dir, args.device, args.steps)
            if rc != 0:
                return rc

    table = {arm: [] for arm, *_ in ARMS}
    for arm, *_ in ARMS:
        for seed in seeds:
            table[arm].append(
                {
                    "seed": seed,
                    **read_run(out_dir / f"{arm}_s{seed}.jsonl", out_dir / f"{arm}_s{seed}.log"),
                }
            )
    probes: dict[tuple[str, int], dict] = {}
    if args.probe:
        for arm, k, _weight, _soft in ARMS:
            if k == 0:
                continue
            for seed in seeds:
                name = f"mtpsoft_{arm}_s{seed}"
                probes[(arm, seed)] = probe(out_dir / "ckpt", name, args.device)

    print(
        f"\n{'arm':>9s} {'seed':>4s} {'bpb':>8s} {'scored B':>9s} {'evals':>6s} "
        f"{'nan':>4s} {'mtp lines':>9s}"
    )
    for arm, runs in table.items():
        for r in runs:
            print(
                f"{arm:>9s} {r['seed']:>4d} {r['bpb']:8.4f} {r['scored']:>9d} "
                f"{r['n_evals']:>6d} {r['nan']:>4d} {r['mtp_logged']:>9d}"
            )
    for arm, runs in table.items():
        mean, spread = summarize(runs)
        print(f"{arm:>9s} mean bpb {mean:.4f}  spread {spread:.4f}")

    if args.probe:
        print(
            f"\n{'arm':>9s} {'seed':>4s} {'CE k=1..4':>26s} {'accept k=1..4':>26s} {'greedy':>20s}"
        )
        for (arm, seed), p in probes.items():
            ce = " ".join(f"{v:.4f}" for v in p["per_head_ce"])
            acc = " ".join(f"{v:.3f}" for v in p["accept"])
            gr = " ".join(f"{v:.3f}" for v in p["greedy"])
            print(f"{arm:>9s} {seed:>4d} {ce:>26s} {acc:>26s} {gr:>20s}")

    accept_hard = accept_soft = None
    if args.probe:
        hard = [p["accept_mean"] for (arm, _s), p in probes.items() if arm == "hard_mtp"]
        soft = [p["accept_mean"] for (arm, _s), p in probes.items() if arm == "soft_mtp"]
        accept_hard = statistics.mean(hard) if hard else None
        accept_soft = statistics.mean(soft) if soft else None

    v = verdict(
        table["ntp"],
        table["hard_mtp"],
        table["soft_mtp"],
        WINDOW,
        accept_hard=accept_hard,
        accept_soft=accept_soft,
    )
    print(
        f"\nsoft_mtp vs hard_mtp: {v['verdict'].upper()} "
        f"(ntp {v['ntp_mean']:.4f} +- {v['control_spread']:.4f}, "
        f"hard {v['hard_mean']:.4f}, soft {v['soft_mean']:.4f}; "
        f"soft-hard {v['soft_minus_hard']:+.4f}, soft-ntp {v['soft_minus_ntp']:+.4f}; "
        f"accept {v['accept_hard']} -> {v['accept_soft']})"
        + (f"\nvoid reasons: {v['reason']}" if v["reason"] else "")
    )
    print(
        "\nkeep rule: better than hard_mtp AND no worse than ntp by more than "
        "2x the control spread AND a higher acceptance. Anything else drops it."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ab", help="the A/B driver (3 arms x 3 seeds)")
    a.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a.add_argument("--steps", type=int, default=STEPS)
    a.add_argument("--device", default="cuda")
    a.add_argument(
        "--probe",
        action="store_true",
        help="load each final checkpoint and measure per-head CE/acceptance",
    )
    a.add_argument("--out-dir", default="~/logs/ab_mtp_soft")
    p = sub.add_parser("probe", help="per-head CE + acceptance for one checkpoint")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--ckpt-name", required=True)
    p.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    why = preflight(args.device) if args.mode == "ab" else _lane_preflight(args.device)
    if why is not None:
        return refuse(why)
    if args.mode == "probe":
        print(
            json.dumps(
                probe(Path(args.ckpt_dir).expanduser(), args.ckpt_name, args.device),
                indent=2,
            )
        )
        return 0
    return ab(args)


if __name__ == "__main__":
    raise SystemExit(main())
