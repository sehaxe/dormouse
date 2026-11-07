"""The long-context plan: does the stack go FLAT from 32 K to 256 K bytes?

    uv run python scripts/ab_longctx.py plan            # resolve + print (no run)
    uv run python scripts/ab_longctx.py run             # the 8 runs (NOT run yet)

**THIS DRIVER HAS NOT BEEN RUN.** It is written, and `plan` is its dry run: it
resolves every config through `load_run` (so the refusals in `config.py` are
exercised, not assumed), prints the per-arm memory bill the dense selection
would allocate at 262 144 bytes, and prints the pre-registered rule below. The
card belongs to the owner, so no arm has a number yet.

**THE QUESTION, pre-registered BEFORE any run.** `attn_window` and the CSA
levels are supposed to make cost FLAT in the context length: the window's bill
is `2·W·d` MAC/byte regardless of `t`, and CSA's state is a ring. So for each
arm, comparing its OWN two lengths (32 768 -> 262 144, 8x the bytes):

* **FLAT** iff `ms(262144) / ms(32768) <= 1.25` **and**
  `peak(262144) - peak(32768) <= 1.5 GiB`;
* **FAILURE** otherwise, and an arm that cannot produce the 262 144 number at
  all (the bill below does not fit) is a failure reported as such — not a
  skipped row.

The threshold is 1.25 rather than 1.0 because 8x the bytes at a fixed batch
still costs the extra KDA launches and the 8x data read; 1.5 GiB is one model
copy at `medium`. Both are written here, before the numbers exist, so the
verdict cannot be chosen after the fact.

**Arms** (one flag family differs; everything else is `ab_mhc.py`'s COMMON, the
battery's recipe, plus the brief's pins):

* `kda_only` — the control: KDA's O(1) state, no extra arm;
* `kda_window32k` — `attn_window=32768, attn_window_query_chunk=4096`: the exact
  local band beside KDA. Its score bill is `2·W·d = 67.1M` MAC/byte at
  `d=1024`, flat in `t`;
* `kda_window32k_csa2` — the same band plus the shipped two-level CSA
  (`csa_chunk=8, csa_kv_dim=64, csa_topk=32, csa_coarse_chunk=256`);
* `kda_window32k_csa3hier` — the same plus the third level and the containment
  chain (`csa_mid_chunk=512, csa_hierarchical=true`).

`medium`, batch 1, 300 steps, seed 1, `bf16`, `sdpa_mem_efficient=true`,
`moe_capacity=0` (dropless — the only MoE mode that is prefix-causal),
`data=/home/sehaxe/dormouse/data_local/best/fwedu.bin`. The two lengths are two
separate driver calls (32 768 first, then 262 144): one process per length
because `max_seq_len` is a model-shape field and the cubecl autotune cache is
per shape.

**`segment_len` is NOT 32 768 everywhere, and the reason is a refusal, not a
choice** (both printed per row):

* at 32 768 bytes `segment_len=32768 == seq_len` is refused as pointless
  (`_validate_segment`), so it is 0;
* `attn_window` with `segment_len>0` is refused: a cut hides the bytes before
  it, so the first positions of a segment would read a NARROWER band than the
  one-shot forward and the segmentation would change the model;
* `use_csa` with `segment_len>0` is now ALLOWED (2026-11-08: the partial chunk
  is carried and completed by the next segment — `tests/test_segment_csa.py`);
  every CSA arm in this table also sets `attn_window`, which is the refusal
  that keeps its row at 0, and a CSA-only arm would now be segmented;
* only `kda_only` at 262 144 keeps `segment_len=32768`.

The flattening claim is therefore WITHIN an arm, across its two lengths, which
is what the rule above measures; comparing `kda_only`'s segmented 256 K row to
a window arm's one-shot row would compare two memory strategies, and the table
does not do it.

**The bill that decides "can it run at all"** (`_temp_gib`, b=1, printed per
row): the largest single per-position temporary of the arm at that length —
`t·n_fine` floats for the flat CSA scan, `t·n_level` for a dense level,
`t·K·kv` for the hierarchical fine gather (`K = topk·coarse/fine`), and
`C·(C+W-1)·4` for one window query block. Anything over `FIT_GIB` is reported
as WILL NOT FIT *before* it is launched, because a 34 TB score matrix is a
property of the design, not of the card.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.append(_SCRIPTS)
from ab_byteflow import preflight, read_run, skip  # noqa: E402
from ab_mhc import COMMON_MODEL, COMMON_TRAIN  # noqa: E402

DATA = "/home/sehaxe/dormouse/data_local/best/fwedu.bin"
PRESET = "medium"
WINDOW = 32768
QUERY_CHUNK = 4096
LENGTHS = (32768, 262144)
SEGMENT = 32768
STEPS = 300
SEED = 1
BATCH = 1
LR = 1e-3
LOG_EVERY = 50
EVAL_BATCHES = 4

#: The pre-registered flatten rule — see the module docstring. Written here so
#: no later edit can move it after the numbers land.
FLAT_MS_RATIO = 1.25
FLAT_GIB_DELTA = 1.5
#: Above this a single temporary does not fit the 16 GB card, whatever the pool
#: does with it.
FIT_GIB = 12.0

#: Pinned OFF for every arm: the `medium` preset ships `use_msa=true`, and the
#: block-sparse arm is (a) a THIRD attention branch that would confound the
#: window/CSA comparison and (b) refused beside `segment_len>0`, which would
#: turn the control's 262 144 row into a different memory strategy. Named in
#: the plan output like every other deviation.
BASE = {"use_msa": False}

#: `(arm, model flags)`; the control is first so the table reads against it.
ARMS = (
    ("kda_only", {}),
    (
        "kda_window32k",
        {"attn_window": WINDOW, "attn_window_query_chunk": QUERY_CHUNK},
    ),
    (
        "kda_window32k_csa2",
        {
            "attn_window": WINDOW,
            "attn_window_query_chunk": QUERY_CHUNK,
            "use_csa": True,
            "csa_chunk": 8,
            "csa_kv_dim": 64,
            "csa_topk": 32,
            "csa_coarse_chunk": 256,
            "csa_coarse_kv_dim": 64,
        },
    ),
    (
        "kda_window32k_csa3hier",
        {
            "attn_window": WINDOW,
            "attn_window_query_chunk": QUERY_CHUNK,
            "use_csa": True,
            "csa_chunk": 8,
            "csa_kv_dim": 64,
            "csa_topk": 32,
            "csa_coarse_chunk": 256,
            "csa_coarse_kv_dim": 64,
            "csa_mid_chunk": 512,
            "csa_hierarchical": True,
        },
    ),
)
ARMS_DICT = dict(ARMS)

#: The counters that name WHICH arm ran (the control must show none of them).
EXPECT = {
    "kda_only": {"iter": ">0", "attn_window_forward": "==0", "csa_forward": "==0"},
    "kda_window32k": {"iter": ">0", "attn_window_forward": ">0", "csa_forward": "==0"},
    "kda_window32k_csa2": {
        "iter": ">0",
        "attn_window_forward": ">0",
        "csa_forward": ">0",
        "csa_coarse_forward": ">0",
        "csa_mid_forward": "==0",
    },
    "kda_window32k_csa3hier": {
        "iter": ">0",
        "attn_window_forward": ">0",
        "csa_forward": ">0",
        "csa_coarse_forward": ">0",
        "csa_mid_forward": ">0",
        "csa_hier_forward": ">0",
        "csa_sel_fine": ">0",
    },
}


def flags(arm: str) -> dict:
    """The arm's model overrides on top of `BASE`."""
    return {**BASE, **ARMS_DICT[arm]}


def _arm_only(arm: str) -> bool:
    """True when the arm overrides anything `BASE` does not pin (it is not bare)."""
    return any(k not in BASE for k in ARMS_DICT[arm])


def segment_len(arm: str, length: int) -> int:
    """The largest segment_len the arm and the length ALLOW (0 when refused)."""
    if length <= SEGMENT:
        return 0
    if _arm_only(arm):
        return 0
    return SEGMENT


def segment_why(arm: str, length: int) -> str:
    """The refusal that made `segment_len` 0, in the config's own words."""
    if length <= SEGMENT:
        return f"segment_len={SEGMENT} >= seq_len={length}: refused as pointless"
    if _arm_only(arm):
        return "this arm overrides attn_window (refused with segment_len) or another flag"
    return f"segment_len={SEGMENT}: the arm adds no branch, and KDA carries O(1) state"


def _temp_gib(arm: str, length: int, d_model: int = 1024) -> dict[str, float]:
    """The largest per-position temporaries the arm allocates at `length`, GiB.

    b=1. `q @ kᵀ` is `[t, n]` floats at fp32; the hierarchical gather is
    `[t, K, kv]`; one window query block holds `[C, C+W-1]` scores plus its
    boolean mask. The list is the arm's, not the allocator's, so it is an
    ESTIMATE with the formula printed beside it.
    """
    gib = 2**30
    out: dict[str, float] = {}
    arm_flags = flags(arm)
    if "attn_window" in arm_flags:
        c = min(QUERY_CHUNK, length)
        out["window block C*(C+W-1)"] = c * (c + WINDOW - 1) * 4 / gib
    if arm_flags.get("use_csa"):
        fine = -(-length // arm_flags["csa_chunk"])
        coarse = -(-length // arm_flags["csa_coarse_chunk"])
        if arm_flags.get("csa_hierarchical"):
            mid = -(-length // arm_flags["csa_mid_chunk"])
            ratio = arm_flags["csa_coarse_chunk"] // arm_flags["csa_chunk"]
            keep = min(arm_flags["csa_topk"], coarse) * ratio
            out["hier mid scores t*n_mid"] = length * mid * 4 / gib
            out["hier coarse scores t*n_coarse"] = length * coarse * 4 / gib
            out["hier fine gather t*K*kv"] = length * keep * arm_flags["csa_kv_dim"] * 4 / gib
        else:
            out["flat fine scores t*n_fine"] = length * fine * 4 / gib
        # The coarse score matrix is the dense branch's AND the selection stage's
        # tensor (computed once), so it is priced once.
        out["coarse scores t*n_coarse (shared)"] = length * coarse * 4 / gib
    if not out:
        out["kda state (O(1) per slot)"] = 0.0
    return out


def _resolve(arm: str, length: int):
    """The resolved run for one arm at one length (`load_run` validates it)."""
    from dormouse.config import load_run

    arm_flags = dict(flags(arm))
    return load_run(
        PRESET,
        model_overrides={
            **COMMON_MODEL,
            **arm_flags,
            "sdpa_mem_efficient": True,
            "max_seq_len": length,
        },
        train_overrides={
            **COMMON_TRAIN,
            "steps": STEPS,
            "batch": BATCH,
            "seq_len": length,
            "seed": SEED,
            "lr": LR,
            "eval": False,
            "log_every": LOG_EVERY,
            "ckpt_every": 10**9,
            "segment_len": segment_len(arm, length),
            "out_dir": str(Path("~/logs/ab_longctx").expanduser()),
            "ckpt_name": f"{arm}_{length}",
        },
    )


def plan(args: argparse.Namespace) -> int:
    """Resolve every (arm, length) and print what a run would cost — no training."""
    from dormouse.instrument import LoudError

    print(
        f"plan: {len(ARMS)} arms x {len(LENGTHS)} lengths x {STEPS} steps, {PRESET} "
        f"b{BATCH}, lr={LR}, data={args.data}\n"
        f"pre-registered FLAT rule: ms ratio <= {FLAT_MS_RATIO} and peak GiB delta "
        f"<= {FLAT_GIB_DELTA}; an arm that cannot produce the 262 144 number is a "
        f"FAILURE, not a skipped row.\n"
        f"pinned OFF for every arm: use_msa (the preset ships it ON; it is a third "
        f"attention branch AND it is refused with segment_len>0)."
    )
    bad = 0
    for length in LENGTHS:
        for arm, _flags in ARMS:
            bill = _temp_gib(arm, length)
            fit = max(bill.values()) <= FIT_GIB
            try:
                run = _resolve(arm, length)
            except LoudError as exc:
                bad += 1
                print(f"\n{arm:>24s} @ {length:>7,d}: REFUSED — {exc}")
                continue
            csa = (
                "off"
                if not run.model.use_csa
                else f"{run.model.csa_chunk}/{run.model.csa_coarse_chunk}/{run.model.csa_mid_chunk}"
            )
            print(
                f"\n{arm:>24s} @ {length:>7,d}: segment_len={run.train.segment_len} "
                f"({segment_why(arm, length)}), window={run.model.attn_window} "
                f"chunk={run.model.attn_window_query_chunk}, csa={csa}, "
                f"hier={run.model.csa_hierarchical}, "
                f"sdpa_mem_efficient={run.model.sdpa_mem_efficient}"
            )
            for name, gib in sorted(bill.items(), key=lambda kv: -kv[1]):
                print(f"    {gib:9.3f} GiB  {name}")
            if not fit:
                print(
                    f"    WILL NOT FIT: the largest temporary is "
                    f"{max(bill.values()):.1f} GiB > {FIT_GIB} GiB — this arm cannot "
                    f"produce the {length:,}-byte row on this card; the per-position "
                    f"selection would need the query blocking `attn_window_query_chunk` "
                    f"gives the WINDOW only."
                )
            else:
                print("    fits")
    print(
        f"\n{len(ARMS) * len(LENGTHS) - bad}/{len(ARMS) * len(LENGTHS)} configs resolve "
        f"through load_run (refusals exercised, not assumed); "
        f"`run` launches them one process at a time"
    )
    return 1 if bad else 0


def _run_one(args: argparse.Namespace) -> int:
    """One arm, one length, one process: the trainer with the arm's flags live."""
    import torch

    from dormouse.config import load_run
    from dormouse.model import DormouseModel
    from dormouse.train import train

    arm_flags = dict(flags(args.arm))
    run = load_run(
        PRESET,
        model_overrides={
            **COMMON_MODEL,
            **arm_flags,
            "sdpa_mem_efficient": True,
            "max_seq_len": args.length,
        },
        train_overrides={
            **COMMON_TRAIN,
            "steps": args.steps,
            "batch": BATCH,
            "seq_len": args.length,
            "seed": args.seed,
            "lr": LR,
            "eval": False,
            "log_every": LOG_EVERY,
            "ckpt_every": 10**9,
            "segment_len": segment_len(args.arm, args.length),
            "out_dir": str(Path(args.out).parent / "ckpts"),
            "ckpt_name": f"{args.arm}_{args.length}",
            "timers": False,
            "json_log": str(args.out),
        },
    )
    run.train.data = args.data
    if not Path(run.train.data).is_file():
        print(f"no data file at {run.train.data}: pass --data")
        return 1
    Path(args.out).with_suffix(".config.toml").write_text(run.snapshot_toml())
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    dev = torch.device(device)

    model = DormouseModel(run.model, device=dev)
    has_window = model.loop.window_attn is not None
    assert has_window == bool(run.model.attn_window), "attn_window dispatch drifted"
    assert (model.loop.csa is not None) == run.model.use_csa, "CSA dispatch drifted"
    window = run.train.eval_batches * BATCH * args.length
    print(
        f"[run] arm={args.arm} length={args.length} segment_len={run.train.segment_len} "
        f"window={run.model.attn_window}/{run.model.attn_window_query_chunk} "
        f"csa={run.model.use_csa} hier={run.model.csa_hierarchical} "
        f"b{BATCH} steps={args.steps} lr={LR} eval_window={window} B "
        f"params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train(model, run, device=dev, log=print)
    peak = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else None
    with open(args.out, "a", encoding="utf-8") as sink:
        sink.write(json.dumps({"event": "params", "peak_gib": peak}) + "\n")
    print(f"[peak] {peak if peak is None else f'{peak:.2f}'} GiB", flush=True)
    return 0


def run(args: argparse.Namespace) -> int:
    why = preflight()
    if why:
        skip(why)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for length in LENGTHS:
        for arm, _flags in ARMS:
            out = out_dir / f"{arm}_{length}.jsonl"
            out.unlink(missing_ok=True)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_run",
                "--arm",
                arm,
                "--length",
                str(length),
                "--seed",
                str(SEED),
                "--steps",
                str(args.steps),
                "--out",
                str(out),
                "--data",
                args.data,
                "--device",
                args.device,
            ]
            print(f"\n=== {arm} @ {length} -> {out}", flush=True)
            done = subprocess.run(cmd, check=False)
            if done.returncode != 0:
                print(f"the {arm} run @ {length} exited {done.returncode}: no table.")
                return done.returncode
    table = _table(out_dir, args.steps)
    _report(table)
    return 0


def _table(out_dir: Path, steps: int) -> dict[tuple[str, int], dict]:
    table: dict[tuple[str, int], dict] = {}
    for length in LENGTHS:
        for arm, _flags in ARMS:
            path = out_dir / f"{arm}_{length}.jsonl"
            row = {"arm": arm, "length": length, "peak_gib": None}
            if path.exists():
                row.update(read_run(path, steps))
                for line in path.read_text().splitlines():
                    event = json.loads(line)
                    if event.get("event") == "params":
                        row["peak_gib"] = event.get("peak_gib")
            table[(arm, length)] = row
    return table


def _report(table: dict[tuple[str, int], dict]) -> None:
    """ms/step, peak GiB, the selection counters, and the FLAT verdict per arm."""
    print(
        f"\n{'arm':>24s} {'length':>8s} {'ms/step':>9s} {'peak GiB':>9s} "
        f"{'nan':>4s} {'sel mid':>8s} {'sel coarse':>11s} {'sel fine':>9s}"
    )

    def per(counters: dict, name: str) -> str:
        return f"{counters.get(name, 0):,}" if name in counters else "-"

    for (arm, length), r in table.items():
        c = r.get("counters", {})
        ms = "-" if "ms_per_step" not in r else f"{r['ms_per_step']:.1f}"
        peak = "-" if r["peak_gib"] is None else f"{r['peak_gib']:.2f}"
        print(
            f"{arm:>24s} {length:>8,d} {ms:>9s} {peak:>9s} {r.get('nan', 0):>4d} "
            f"{per(c, 'csa_sel_mid'):>8s} {per(c, 'csa_sel_coarse'):>11s} "
            f"{per(c, 'csa_sel_fine'):>9s}"
        )
    print()
    for arm, _flags in ARMS:
        lo, hi = table[(arm, 32768)], table[(arm, 262144)]
        if "ms_per_step" not in lo or "ms_per_step" not in hi:
            print(f"{arm:>24s}: FAILURE — a length produced no ms/step (no number)")
            continue
        ratio = hi["ms_per_step"] / lo["ms_per_step"]
        dgib = (
            None
            if lo["peak_gib"] is None or hi["peak_gib"] is None
            else hi["peak_gib"] - lo["peak_gib"]
        )
        flat = ratio <= FLAT_MS_RATIO and dgib is not None and dgib <= FLAT_GIB_DELTA
        print(
            f"{arm:>24s}: ms {lo['ms_per_step']:.1f} -> {hi['ms_per_step']:.1f} "
            f"(x{ratio:.2f}, bar {FLAT_MS_RATIO}); peak "
            f"{lo['peak_gib']} -> {hi['peak_gib']} "
            f"({f'delta {dgib:.2f} GiB' if dgib is not None else 'unmeasured'}, "
            f"bar {FLAT_GIB_DELTA}) => {'FLAT' if flat else 'FAILURE'}"
        )
        for name, want in EXPECT[arm].items():
            got = hi.get("counters", {}).get(name, 0)
            if (want == ">0" and got <= 0) or (want == "==0" and got != 0):
                print(f"    VOID: {name}={got} at {hi['length']:,} (wanted {want})")
    print(
        "\npre-registered: FLAT is the goal, per arm and across ITS two lengths; an "
        "arm that is not flat (or cannot produce the 262 144 row) is a FAILURE and is "
        "reported as one. The counters name the selection actually run."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan", help="resolve + print the bill and the rule (no run)")
    p.add_argument("--data", default=DATA)
    r = sub.add_parser("run", help="the 8 runs, one process at a time")
    r.add_argument("--steps", type=int, default=STEPS)
    r.add_argument("--data", default=DATA)
    r.add_argument("--device", default="cuda")
    r.add_argument("--out-dir", default="~/logs/ab_longctx")
    one = sub.add_parser("_run", help="one arm, one length (the child mode)")
    one.add_argument("--arm", required=True, choices=[arm for arm, _f in ARMS])
    one.add_argument("--length", type=int, required=True, choices=LENGTHS)
    one.add_argument("--seed", type=int, required=True)
    one.add_argument("--steps", type=int, default=STEPS)
    one.add_argument("--out", required=True)
    one.add_argument("--data", default=DATA)
    one.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if args.mode == "plan":
        return plan(args)
    if args.mode == "run":
        return run(args)
    return _run_one(args)


if __name__ == "__main__":
    sys.exit(main())
