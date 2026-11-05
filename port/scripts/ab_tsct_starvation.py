#!/usr/bin/env python3
"""The starvation experiment: does the TSCT bank lose because its experts are starved?

THE QUESTION. Two measured facts are consistent with each other and nobody has
separated them (2026-11-05):

    MoE E=512 rank64 199M  2.3655 BPB @2000 steps  31 347 B/s  189/512 experts alive
    dense 81M              2.3195 BPB @2000 steps  46 106 B/s

    equal-active-MAC 600-step sweep, 3 seeds:
    E=512 2.5387 | E=128 2.5482 | E=64 2.5382 | dense 2.4714

The candidate causes are (a) TOKEN STARVATION — at b8 x s1024 with top-2 the
bank sees `8192*2/512 = 32` assignments per expert per step, so a `[rank, f]`
write is trained on 32 rows while a dense FFN sees 2048 — and (b) the rank-64
FACTORIZATION caps what one expert can represent. This experiment prices (a):
hold the model, the data and the active assignment width fixed, spend 3x the
tokens per step, and ask how the bank-vs-dense GAP moves.

ARMS (all resolved through `config.load_run` on `configs/m400b.toml`; the
`--set` difference is the keys named here and nothing else):

    A  E=512  b24 x s1024   96 tokens/expert/step   (the starvation arm)
    B  E=128  b24 x s1024  384 tokens/expert/step   (same active MAC/byte, narrower bank)
    C  E=512  b8  x s1024   32 tokens/expert/step   (TODAY'S CONTROL)
    D  dense  b24 x s1024      -                   (the same-bytes dense control)
    E  dense  b8  x s1024      -                   (the same-bytes dense control)

Same bytes/step WITHIN each tier by construction (A/B/D share `b24 x s1024`;
C/E share `b8 x s1024`). "Same active MAC" holds exactly inside the bank arms:
with a constant top-2 every token runs two `[rank, f]` writes, so E changes how
many copies exist, not what one token computes (`capacity_math.active_macs_per_token`
is printed per arm). The dense arm cannot equal the bank's active bill at the
recorded 81M recipe; it is printed, and the DIFFERENCE-IN-GAPS design cancels
that constant advantage: the question is how `gap = BPB(E=512) - BPB(dense)`
moves between the two byte tiers, not what the gap is at one tier.

THE b24 x s1024 MEMORY FALLBACK. The 2026-10-24 measurement (P-FULLGRAPH) put
that step's eager peak at **14.8 of the card's 15.6 GiB**, so it does not fit
with headroom; the driver therefore resolves the high tier to **b24 x s512**
(12 288 B/step, `48` tokens/expert at E=512) unless `--seq 1024 --vram-gib 24`
is passed. That is a weaker contrast than the pre-registered 32 -> 96 and the
output says so: with the fallback the verdict is DIRECTIONAL, not the 3x test.

THE RUN. 1500 steps, 2 seeds per arm, the same eval (`eval_every=500`, 8
batches), `dormouse.cli train`, no checkpoint (`ckpt_every` huge), a fresh
`ckpt_name`, `data`/`eval`/`mix_plan` pinned exactly as `scripts/ab_capacity.py`
does. The only difference between arms is the model keys listed above and the
batch/seq of their tier.

THE DECISION RULE, PRE-REGISTERED HERE BEFORE THE FIRST RUN (and printed by
`--dry-run`):

    gap32 = mean_BPB(A=C arm, E=512) - mean_BPB(E, dense)   # 32 tokens/expert
    gapHI = mean_BPB(A, E=512)       - mean_BPB(D, dense)   # 96 (or 48) tokens/expert
    spread = max(E_seeds) - min(E_seeds)                    # dense-b8 seed spread

    STARVATION IS TRUE  iff gapHI <= 0.5 * gap32.
    STARVATION IS FALSE iff |gapHI - gap32| <= 2 * spread.
    Otherwise the result is INCONCLUSIVE (it shrank, but by less than half, and
    the move is outside the noise band): neither the starvation claim nor the
    "unchanged" claim is earned.

A tie (`gapHI == gap32`) is FALSE by construction — this is the gate the test
pins. A `gap32 <= 0` (the bank already matches or beats dense at 32) makes the
ratio meaningless and is reported as `NO GAP TO SHRINK`, not TRUE.

USAGE
    python scripts/ab_tsct_starvation.py --dry-run       # configs + tokens/expert + table
    python scripts/ab_tsct_starvation.py --dry-run --seq 1024 --vram-gib 24
    python scripts/ab_tsct_starvation.py --run               # the 10 runs (NOT run by this task)
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PORT = Path(__file__).resolve().parents[1]
REPO = PORT.parent
sys.path.append(str(PORT / "scripts"))  # `scripts/` is not a package
if str(PORT / "src") not in sys.path:
    sys.path.insert(0, str(PORT / "src"))

import capacity_math as cm  # noqa: E402

from dormouse.config import RunConfig, load_run  # noqa: E402

BASE_PRESET = PORT / "configs" / "m400b.toml"
DATA = REPO / "data_local" / "best" / "fwedu.bin"
EVAL = REPO / "data_local" / "eval_mix" / "eval_mix.bin"
MIX_PLAN = PORT / "configs" / "mix_v7.toml"

STEPS = 1500
SEEDS = (1, 2)
EVAL_EVERY = 500
EVAL_BATCHES = 8
MIN_EFFECT = 0.0  # the rule's band is 2*spread; no absolute floor is added here
#: The measured b24 x s1024 eager peak (P-FULLGRAPH, 2026-10-24) and the
#: headroom below which the high tier is resolved to s512.
B24_S1024_GIB = 14.8
CARD_GIB = 15.6
HEADROOM_GIB = 1.0
CKPT_EVERY = 1_000_000_000
DEFAULT_OUT_ROOT = Path("/tmp/ab_tsct_starvation")

#: The bank recipe, minus the keys an arm varies. `moe_routing_health` is a
#: counter set (no parameter, no compute): it makes the live-expert count
#: readable on the same log line the BPB comes from.
COMMON: dict[str, object] = {"moe_routing_health": True}
DENSE: dict[str, object] = {
    # The review's dense recipe: `f=4096, share_u off`, one full-rank expert.
    # Resolved on THIS tree's m400b it is 77 464 991 parameters, not the review's
    # 81 126 404 (that run's exact override set is not committed); the arm is
    # "the bank removed, full rank", which is what the control needs to be.
    "use_tsct": False,
    "share_u": False,
    "share_u_mode": "off",
    "n_experts": 1,
    "moe_topk": 0,
    "rank": 1536,
}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    batch: int
    tier: str  # "hi" (b24) or "lo" (b8)
    overrides: dict[str, object]
    note: str


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("A", 24, "hi", {}, "E=512: the starvation arm (96 tokens/expert at s1024)"),
    ArmSpec("B", 24, "hi", {"n_experts": 128}, "E=128: same active MAC/byte, 3x per-expert tokens"),
    ArmSpec("C", 8, "lo", {}, "E=512 at b8: TODAY'S CONTROL (32 tokens/expert)"),
    ArmSpec("D", 24, "hi", DENSE, "dense: the same-bytes control for A"),
    ArmSpec("E", 8, "lo", DENSE, "dense: the same-bytes control for C"),
)


def tokens_per_expert(batch: int, seq: int, topk: int, n_experts: int) -> float:
    """Assignments per expert per step: `b*s*k/E` (byte-level: 1 token = 1 byte)."""
    return batch * seq * topk / n_experts


def high_seq(vram_gib: float = CARD_GIB) -> int:
    """The high tier's `seq`: s1024 when it fits with headroom, else s512."""
    return 1024 if B24_S1024_GIB + HEADROOM_GIB <= vram_gib else 512


def resolve(*, seq_hi: int = 1024) -> list[dict]:
    """Resolve every arm through `load_run` and price it with `capacity_math`.

    No card is touched: `measure` builds a `meta` model and the token counts are
    arithmetic. `tokens/expert` is reported both from `measure` and from the
    direct `b*s*k/E` formula, and the two are asserted equal — a formula that
    drifts from the ledger would make the whole experiment a unit error.
    """
    base_e = load_run(str(BASE_PRESET)).model.n_experts
    rows: list[dict] = []
    for spec in ARMS:
        seq = seq_hi if spec.tier == "hi" else 1024
        run: RunConfig = load_run(
            str(BASE_PRESET), model_overrides={**COMMON, **spec.overrides}
        )
        row = cm.measure(run, batch=spec.batch, seq=seq)
        topk = run.model.moe_topk if run.model.moe_topk > 0 else run.model.n_experts
        direct = tokens_per_expert(spec.batch, seq, topk, run.model.n_experts)
        assert math.isclose(
            float(row["tokens_per_expert_step"]), direct, rel_tol=1e-9
        ), f"{spec.name}: capacity_math and b*s*k/E disagree"
        rows.append(
            {
                "arm": spec.name,
                "note": spec.note,
                "run": run,
                "row": row,
                "batch": spec.batch,
                "seq": seq,
                "tier": spec.tier,
                "tokens_per_expert": direct,
                "base_e": base_e,
            }
        )
    return rows


def starvation_verdict(
    gap_control: float,
    gap_high: float,
    spread_control: float,
    *,
    control_tokens: float = 32.0,
    high_tokens: float = 96.0,
) -> dict:
    """The pre-registered rule, as a pure function (the tie gate lives here).

    `gap_*` are `mean_BPB(E=512) - mean_BPB(dense)` at the SAME bytes/step;
    `spread_control` is the dense control's seed spread (max-min). The order of
    the branches is the rule: a tie is FALSE before the ratio is ever formed.
    """
    if gap_control <= MIN_EFFECT:
        return {
            "verdict": "NO GAP TO SHRINK",
            "reason": (
                f"gap at {control_tokens:g} tokens/expert is {gap_control:+.4f} BPB "
                f"(<= {MIN_EFFECT:g}): the bank does not lose at the control, so "
                f"there is no deficit for more tokens to close."
            ),
            "gap_control": gap_control,
            "gap_high": gap_high,
            "half": 0.5 * gap_control,
            "band": 2.0 * spread_control,
        }
    band = 2.0 * spread_control
    if abs(gap_high - gap_control) <= band:
        return {
            "verdict": "NOT STARVATION",
            "reason": (
                f"the gap moved {gap_high - gap_control:+.4f} BPB from "
                f"{control_tokens:g} -> {high_tokens:g} tokens/expert, inside the "
                f"2x control-spread band ({band:.4f}): unchanged."
            ),
            "gap_control": gap_control,
            "gap_high": gap_high,
            "half": 0.5 * gap_control,
            "band": band,
        }
    if gap_high <= 0.5 * gap_control:
        return {
            "verdict": "STARVATION",
            "reason": (
                f"the gap shrank {gap_control:+.4f} -> {gap_high:+.4f} BPB "
                f"(<= half at {0.5 * gap_control:+.4f}) when tokens/expert went "
                f"{control_tokens:g} -> {high_tokens:g}: the bank's deficit is "
                f"token supply, not the rank-64 form."
            ),
            "gap_control": gap_control,
            "gap_high": gap_high,
            "half": 0.5 * gap_control,
            "band": band,
        }
    return {
        "verdict": "INCONCLUSIVE",
        "reason": (
            f"the gap shrank {gap_control:+.4f} -> {gap_high:+.4f} BPB, outside "
            f"the noise band ({band:.4f}) but by less than half "
            f"({0.5 * gap_control:+.4f}): neither claim is earned."
        ),
        "gap_control": gap_control,
        "gap_high": gap_high,
        "half": 0.5 * gap_control,
        "band": band,
    }


def print_table(rows: list[dict], *, seq_hi: int) -> None:
    print("\nRESOLVED ARMS (config.load_run on configs/m400b.toml; no card touched)")
    header = (
        f"{'arm':>3s} {'E':>4s} {'b':>3s} {'seq':>4s} {'bytes/step':>11s} "
        f"{'k':>2s} {'tok/expert':>10s} {'active MAC/B':>13s} {'params':>12s}  note"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        m = r["run"].model
        topk = m.moe_topk if m.moe_topk > 0 else m.n_experts
        print(
            f"{r['arm']:>3s} {m.n_experts:>4d} {r['batch']:>3d} {r['seq']:>4d} "
            f"{r['batch'] * r['seq']:>11,d} {topk:>2d} "
            f"{r['tokens_per_expert']:>10.1f} {r['row']['active_macs']:>13,d} "
            f"{r['row']['params']:>12,d}  {r['note']}"
        )
    print(f"\nSAME-BYTES PAIRS: (C, E) at b8 x s1024; (A, B, D) at b24 x s{seq_hi}.")
    print("SAME ACTIVE MAC/BYTE: A and B differ only by the router GEMM's E term")
    print("(2.1% at E=512 vs 128); the dense arms run 1 full-rank expert and are")
    print("priced, not equalised — the difference-in-gaps test cancels their")
    print("constant advantage between the two byte tiers.")
    if seq_hi != 1024:
        print(
            f"\nFALLBACK: b24 x s1024 peaks {B24_S1024_GIB:.1f} GiB of the card's "
            f"{CARD_GIB:.1f} GiB (measured 2026-10-24) -> the high tier is b24 x s512 "
            f"= {24 * seq_hi:,d} B/step and {tokens_per_expert(24, seq_hi, 2, 512):g} "
            f"tokens/expert at E=512, NOT 96. The verdict is then DIRECTIONAL, not "
            f"the 3x contrast; pass --seq 1024 --vram-gib 24 to run the primary arm."
        )


RULE = """\
PRE-REGISTERED DECISION RULE (never changes after the first run):
  gap32 = mean(BPB | E=512, 32 tok/expert) - mean(BPB | dense at the same bytes/step)
  gapHI = mean(BPB | E=512, HI tok/expert) - mean(BPB | dense at the same bytes/step)
  spread = max(dense-b8 seeds) - min(dense-b8 seeds)
  STARVATION is TRUE  iff gapHI <= 0.5 * gap32
  STARVATION is FALSE iff |gapHI - gap32| <= 2 * spread   (a tie is FALSE)
  otherwise INCONCLUSIVE (it shrank, but not by half, and not within noise)."""


def _argv(arm: dict, seed: int, out_root: Path) -> list[str]:
    m = arm["run"].model
    out_dir = out_root / f"{arm['arm']}-seed{seed}"
    sets = [
        f"seq_len={arm['seq']}",
        f"data={DATA}",
        f"eval_data={EVAL}",
        f"mix_plan={MIX_PLAN}",
        f"eval_every={EVAL_EVERY}",
        f"eval_batches={EVAL_BATCHES}",
        f"ckpt_every={CKPT_EVERY}",
        f"out_dir={out_dir}",
        f"ckpt_name={arm['arm'].lower()}",
        f"json_log={out_dir / 'run.jsonl'}",
        f"seed={seed}",
    ]
    if m.n_experts != arm["base_e"]:
        sets.append(f"n_experts={m.n_experts}")
    if not m.use_tsct:
        sets += [
            "use_tsct=false",
            "share_u=false",
            "share_u_mode=off",
            "moe_topk=0",
            f"rank={m.rank}",
        ]
    return [
        sys.executable,
        "-m",
        "dormouse.cli",
        "train",
        str(BASE_PRESET),
        "--batch",
        str(arm["batch"]),
        "--steps",
        str(STEPS),
        *[x for s in sets for x in ("--set", s)],
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    seq_hi = high_seq(args.vram_gib)
    if args.seq is not None:
        seq_hi = args.seq
    rows = resolve(seq_hi=seq_hi)
    print(f"\nSTEPS={STEPS}  SEEDS={list(SEEDS)}  EVAL_EVERY={EVAL_EVERY} "
          f"EVAL_BATCHES={EVAL_BATCHES}")
    print_table(rows, seq_hi=seq_hi)
    print("\n" + RULE)
    if args.dry_run or not args.run:
        print("\n[dry-run] nothing trained. Pass --run to execute the table above.")
        return 0
    if not DATA.is_file() or not EVAL.is_file() or not MIX_PLAN.is_file():
        raise SystemExit(f"missing data: {DATA} / {EVAL} / {MIX_PLAN}")
    for arm in rows:
        for seed in SEEDS:
            cmd = _argv(arm, seed, args.out_root)
            out_dir = args.out_root / f"{arm['arm']}-seed{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print("+", " ".join(cmd))
            with (out_dir / "run.log").open("w") as fh:
                subprocess.run(cmd, cwd=str(PORT), stdout=fh, stderr=subprocess.STDOUT, check=True)
    return 0


BASE_E_ROW = "base_e"


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="print configs + table, train nothing")
    p.add_argument("--run", action="store_true", help="execute the 5 arms x 2 seeds")
    p.add_argument("--seq", type=int, default=None, help="force the high tier's seq (1024|512)")
    p.add_argument("--vram-gib", type=float, default=CARD_GIB)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
