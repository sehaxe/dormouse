#!/usr/bin/env -S uv run --script
"""Per-domain held-out BPB from ONE checkpoint and ONE procedure.

A single BPB on one English window hides the four numbers a company needs:
English, code, math and Russian, each scored the way the trainer scores its own
eval window (`dormouse.train.loop.evaluate` over `eval_batches_for` windows from
`open_streams`), each from a PINNED offset, each on a file no training source
reads.

    cd port
    uv run scripts/eval_by_domain.py --ckpt-dir ../checkpoints/csa8k_prop \
        --ckpt-name csa8k_prop --mix-plan configs/mix_refetch.toml

The window for a plan source is `<stem>_eval.bin` beside it, with a `_train`
suffix dropped: `best/fwedu_train.bin -> best/fwedu_eval.bin`,
`mix/code.bin -> mix/code_eval.bin`. `scripts/carve_eval_tails.py` makes those
files once; `docs/reviews/2026-11-08-domain-eval.md` records the offsets and
SHA-256s.

The overlap gate is `eval_window.py`'s path rule: a window that IS a training
source of the run is refused, not scored. A COPY of training text under another
name cannot be detected by a path comparison, and this says so rather than
pretending otherwise.

`bytes trained` is the EXACT per-source draw of the mixture replayed to the
checkpoint's step (`mixture.replay_bytes`, the same pure function a resume
uses), not a weight x budget estimate. It counts bytes READ (each window
carries the successor byte), so it is the stream's number, not the scored
count.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from dormouse.cli import _device, _model, _snapshot_run
from dormouse.instrument import loud
from dormouse.train.data import eval_batches_for, eval_window_bytes, open_streams
from dormouse.train.loop import evaluate, load_model_half, load_model_state
from dormouse.train.mixture import load_mix_plan, replay_bytes

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_window import refuse_training_overlap  # noqa: E402

LN2 = math.log(2)


@dataclass(frozen=True)
class DomainScore:
    """One row of the table: a domain's window, its two BPBs, and its fuel."""

    domain: str
    scored: int
    bpb_fin: float
    bpb_mean: float
    trained: int | None


def eval_path_for(train: Path) -> Path:
    """`<stem>_eval.bin` beside a plan source, `_train` dropped from the stem.

    `best/fwedu_train.bin -> best/fwedu_eval.bin` and
    `mix/code.bin -> mix/code_eval.bin` — the same naming the fwedu recovery
    used, so a source and its window always sit together.
    """
    stem = train.stem
    if stem.endswith("_train"):
        stem = stem[: -len("_train")]
    return train.with_name(f"{stem}_eval.bin")


def resolve_plan(value: str | Path) -> Path:
    """A snapshot's `configs/...` is relative to `port/`, where the trainer runs."""
    p = Path(value)
    for cand in (p, Path(__file__).resolve().parents[1] / p):
        if cand.is_file():
            return cand.resolve()
    loud(
        f"mix plan {value} does not exist (tried {p} and "
        f"{Path(__file__).resolve().parents[1] / p}). Escape: pass --mix-plan "
        f"with the plan the checkpoint trained on."
    )


def checkpoint_step(weights: Path) -> int:
    """The checkpoint's 0-based last completed step, from the file itself.

    `save_ckpt` writes `step` = the index of the step it just finished; a
    resume continues at `step + 1`, so the number of completed optimizer steps
    is `step + 1`. Read with `mmap=True`: only the small records are touched,
    the tensors are not paged in until the model load.
    """
    ck = torch.load(weights, map_location="cpu", weights_only=False, mmap=True)
    step = ck.get("step")
    if step is None:
        loud(
            f"{weights} carries no `step`. Escape: pass --step with the completed "
            f"optimizer-step count (the log's `ckpt step=` is 1-based)."
        )
    return int(step)


def trained_bytes(plan, cfg, done_steps: int) -> dict[str, int]:
    """Exact bytes each source served in `done_steps` optimizer steps.

    The trainer draws `grad_accum` micro-batches per optimizer step; the
    mixture's unit is the micro-batch, so the replay takes
    `done_steps x grad_accum`. The seed is the run's (`open_mixture` passes
    `run.train.seed`), not the plan's.
    """
    total = cfg.steps * cfg.grad_accum
    return replay_bytes(
        plan,
        steps=total,
        batch=cfg.batch,
        seq_len=cfg.seq_len,
        upto=done_steps * cfg.grad_accum,
        seed=cfg.seed,
    )


def select_domains(plan, names: str) -> list[str]:
    """The plan's sources in plan order, or the comma-separated subset asked for."""
    if not names:
        return list(plan.sources)
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    unknown = sorted(set(wanted) - set(plan.sources))
    if unknown:
        loud(
            f"--domains names {unknown}, which are not in the plan "
            f"({', '.join(plan.sources)}). Escape: use a source name, or drop "
            f"--domains to score every source."
        )
    return wanted


def windows_for(plan, run, names: str, allow_overlap: bool) -> list[tuple[str, Path]]:
    """`(domain, eval file)` for every selected source, refusing an absent or
    training-owned window.

    The path rule is `eval_window.refuse_training_overlap`: a window that is
    the training file (or any mix source) is not held out and is refused here,
    before a forward pass. A copy of the same text under another name passes —
    a path comparison cannot see content, and the doc says so.
    """
    out: list[tuple[str, Path]] = []
    for name in select_domains(plan, names):
        window = eval_path_for(Path(plan.sources[name]))
        if not window.is_file():
            loud(
                f"domain {name!r}: no held-out window at {window}. Escape: carve "
                f"one with `uv run scripts/carve_eval_tails.py --plan "
                f"{run.train.mix_plan}`, or select the domains you have with "
                f"--domains."
            )
        refuse_training_overlap(run, window, allow_overlap)
        out.append((name, window))
    return out


def window_cfg(train, path: Path, window: int | None, offset: int):
    """The run's config with ONE domain's window in place of the recorded one.

    `window=None` keeps the snapshot's own geometry, including a legacy
    `eval_batches` — that is the window the trainer actually scored, and the
    comparable per-domain number. An explicit `--bytes` replaces both window
    keys and goes through the usual multiple/floor checks.
    """
    over: dict[str, object] = {
        "eval": True,
        "eval_data": str(path),
        "eval_offset": offset,
    }
    if window is not None:
        over["eval_bytes"] = window
        over["eval_batches"] = None
    return dataclasses.replace(train, **over)


def render_table(rows: list[DomainScore]) -> str:
    """The markdown table: one row per domain, the two BPBs side by side."""
    lines = [
        "| domain | bytes scored | bpb_fin | bpb_mean | bytes trained |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        trained = "n/a" if r.trained is None else f"{r.trained:,}"
        lines.append(
            f"| {r.domain} | {r.scored:,} | {r.bpb_fin:.4f} | {r.bpb_mean:.4f} | {trained} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="the run's checkpoint directory (default: checkpoints/<ckpt-name>)",
    )
    ap.add_argument("--ckpt-name", required=True)
    ap.add_argument(
        "--mix-plan",
        default=None,
        help="the plan to score (default: the checkpoint snapshot's own)",
    )
    ap.add_argument(
        "--bytes",
        type=int,
        default=None,
        help="scored bytes PER domain (default: the checkpoint's own eval window)",
    )
    ap.add_argument("--offset", type=int, default=0, help="pinned byte offset in each window")
    ap.add_argument("--step", type=int, default=None, help="completed optimizer steps")
    ap.add_argument("--domains", default="", help="comma list (default: every source)")
    ap.add_argument(
        "--check-windows",
        action="store_true",
        help="resolve and gate the windows, then stop before loading the model",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--allow-train-overlap",
        action="store_true",
        help="score a window that lives in the training corpus (the number is then not held out)",
    )
    a = ap.parse_args()

    ckpt_dir = a.ckpt_dir or Path("checkpoints") / a.ckpt_name
    run = _snapshot_run(ckpt_dir, a.ckpt_name)
    plan_path = resolve_plan(a.mix_plan or run.train.mix_plan or "")
    plan = load_mix_plan(plan_path)
    run.train = dataclasses.replace(run.train, mix_plan=str(plan_path))

    window = a.bytes
    window_display = eval_window_bytes(run.train) if window is None else window
    done = a.step if a.step is not None else checkpoint_step(ckpt_dir / f"{a.ckpt_name}.pt") + 1
    trained = trained_bytes(plan, run.train, done)

    windows = windows_for(plan, run, a.domains, a.allow_train_overlap)
    if a.allow_train_overlap:
        print("WARNING: --allow-train-overlap: these windows are NOT held out", flush=True)
    if a.check_windows:
        for name, window_path in windows:
            cfg = window_cfg(run.train, window_path, window, a.offset)
            batches = eval_batches_for(cfg)
            # the same constructor the score uses: it refuses a window that
            # does not fit or that is not a whole number of rows
            open_streams(dataclasses.replace(run, train=cfg), eval_only=True)
            print(
                f"{name}\t{window_path}\t{window_path.stat().st_size:,} B file"
                f"\tover {batches * cfg.batch * cfg.seq_len:,} B ({batches} x "
                f"{cfg.batch} x {cfg.seq_len})"
            )
        return 0

    dev = _device(a.device)
    model = _model(run).to(dev)
    load_model_state(model, run.train, load_model_half(ckpt_dir / f"{a.ckpt_name}.pt"))

    rows: list[DomainScore] = []
    for name, window_path in windows:
        cfg = window_cfg(run.train, window_path, window, a.offset)
        batches = eval_batches_for(cfg)
        _, stream = open_streams(dataclasses.replace(run, train=cfg), eval_only=True)
        if stream is None:
            loud(f"domain {name!r}: open_streams built no eval stream for {window_path}")
        loss, scored, fin, _per_iter = evaluate(
            model,
            stream,
            batches,
            device=None if dev.type == "cpu" else dev,
        )
        rows.append(
            DomainScore(
                domain=name,
                scored=scored,
                bpb_fin=fin / LN2,
                bpb_mean=loss / LN2,
                trained=trained.get(name),
            )
        )

    print(
        f"checkpoint={a.ckpt_name} step={done} window={window_display:,} B/domain "
        f"off={a.offset} plan={plan_path} device={dev.type}",
        flush=True,
    )
    print(render_table(rows), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
