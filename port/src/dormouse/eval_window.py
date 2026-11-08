"""Which files a run TRAINS on — the held-out check, as pure path logic.

`eval_data` and `mix_plan` are trusted to name a different file from `data`
and from every plan source, and the trainer's own gate
(`train.data.eval_train_overlap`) verifies it by inode at startup: two paths
can name one file, and a BPB scored on bytes the run read is not held out
(`docs/reviews/2026-11-08-*` — the `off=100,000,000` window was scored while
the run's cursor had already passed 108.9 MB).

This module is that predicate without the streams: a path comparison, shared
by `scripts/eval_window.py` (the scoring tool) and `dormouse diagnose` (which
has a snapshot, not an open stream). It cannot see the same text under another
name; it says so rather than guessing.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def training_sources(data: str | None, mix_plan: str | None) -> set[Path]:
    """Every file the run trains on: the flat `data` file plus each plan source.

    A relative `[sources]` path resolves beside the PLAN — the trainer's own
    rule (`train/mixture.py::load_mix_plan`), not the caller's cwd. A named
    plan that cannot be read is `FileNotFoundError`: silently dropping it would
    turn a mixed run into a single-source one and a real overlap into a pass.
    Callers that want to report that instead of raising catch it.
    """
    out: set[Path] = set()
    if data:
        out.add(Path(data).resolve())
    if mix_plan:
        plan = Path(mix_plan)
        with open(plan, "rb") as fh:
            table = tomllib.load(fh)
        for value in (table.get("sources") or {}).values():
            p = Path(value)
            out.add((p if p.is_absolute() else plan.parent / p).resolve())
    return out


def training_paths(run) -> set[Path]:
    """`training_sources` for a resolved `RunConfig` (the script's entry)."""
    return training_sources(run.train.data, run.train.mix_plan)


def training_overlap(data: str | None, mix_plan: str | None, eval_path: Path) -> Path | None:
    """The training source the eval file IS, or `None` when it is separate.

    Path identity only: a copy of a source under another name passes this and
    is not held out, which is the stated ceiling of the cheap detector.
    """
    target = eval_path.resolve()
    for source in training_sources(data, mix_plan):
        if source == target:
            return source
    return None
