"""CRR fork-replay advantage — the 2609.33875 shape, on a group of rollouts.

`docs/research/2026-10-15-agentic-post-training.md` §1 row CRR: "restore a state,
take an alternative action, roll forward; replace the advantage at selected steps
with (real return − counterfactual return). No human process labels, no learned
process RM." At group granularity that is:

- a **passing** trace's advantage is `+1`;
- a **failing** trace inherits `−1` from the BEST failed rollout's fork point
  history — the counterfactual baseline is the best failed rollout itself, not a
  learned reward model;
- a failing trace with an **empty** trajectory has no fork history and carries
  `0` (there is nothing to attribute).

`fork_depth` is the best failed rollout's fork point: the number of strokes the
furthest-failing rollout survived. Nothing here updates a weight; the trainer is
downstream of this function, per the plan's state 1.
"""

from __future__ import annotations

from dataclasses import dataclass

from .rollouts import Rollout

__all__ = ["Advantage", "crr_advantage"]


@dataclass(frozen=True)
class Advantage:
    """One rollout's advantage and where it came from."""

    index: int
    value: float
    source: str  # "pass" | "fork_history" | "empty"
    fork_depth: int


def crr_advantage(group: list[Rollout]) -> list[Advantage]:
    """Group-level fork-replay advantages, aligned with `group` by index."""
    failed = [r for r in group if not r.passed and len(r.trace) > 0]
    fork_depth = max((len(r.trace) for r in failed), default=0)
    out: list[Advantage] = []
    for i, r in enumerate(group):
        if r.passed:
            out.append(Advantage(i, 1.0, "pass", fork_depth))
        elif len(r.trace) == 0:
            out.append(Advantage(i, 0.0, "empty", fork_depth))
        else:
            out.append(Advantage(i, -1.0, "fork_history", fork_depth))
    return out
