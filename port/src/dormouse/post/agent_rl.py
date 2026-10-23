"""Trajectory-level credit assignment: one weight per supervised span.

The single-turn CRR arm (`post/train_steps.py`) forks a ROLLOUT: a passing trace
inherits `+1`, a failing one inherits `−1` from the best failed rollout's fork
point. A multi-turn trajectory has a fork worth naming per turn: the transcript
PREFIX is the state, the turn is the action, and the counterfactual is what
would have happened if the model had continued differently from that point.
That is fork-replay at the transcript level, and the estimator is the CRR shape
the lane already ships (`docs/research/2026-10-15-agentic-post-training.md` §1
row CRR):

    advantage(turn t) = pass(actual continuation)
                      − mean(pass(replayed continuations from turn t's prefix))

Every replayed continuation resamples turn `t` (the caller runs the replays; this
module does not invent them) and is scored by the SAME deterministic verifier the
actual rollout was, so the baseline is a return, not a learned model.

Three properties are deliberate:

* **One weight per SUPERVISED span**, from `sft.scan_ranges`: a turn that both
  spoke and called contributes two spans (its text and its call bytes) with the
  same weight; the observations are never in that list. The reward is assigned
  where the model wrote bytes and nowhere else.
* **A turn with no call is not a fork** — no counterfactual was sampled, so its
  weight is exactly 0 and the source says `no_call`. This is what makes a
  two-turn trajectory in which only turn 2 matters leave turn 1 at zero.
* **No replays is a LOUD refusal**, not a silent zero: without a replay the
  "advantage" would be the raw pass flag, i.e. a different estimator wearing the
  fork's name (`RLConfig.min_replays` is the floor).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..instrument import loud
from .agent import Trajectory
from .rl import RLConfig

__all__ = ["SpanAdvantage", "trajectory_advantages"]


@dataclass(frozen=True)
class SpanAdvantage:
    """One supervised span's weight. `span` is INCLUSIVE, over the transcript bytes."""

    turn: int
    span: tuple[int, int]
    weight: float
    #: "fork" (a call-bearing turn with replays) | "no_call" (not a fork, weight 0).
    source: str


def trajectory_advantages(
    trajectory: Trajectory,
    replays: Mapping[int, Sequence[bool]],
    cfg: RLConfig | None = None,
) -> list[SpanAdvantage]:
    """Fork-replay weights, one per supervised span of `trajectory`.

    `replays[t]` is the pass/fail of each replayed CONTINUATION of the turn-`t`
    fork (the transcript prefix before turn `t`, then `t` resampled). `cfg` is
    the lane's `RLConfig`; `cfg.min_replays` is the number of counterfactuals a
    fork needs before its mean is believed.
    """
    cfg = RLConfig() if cfg is None else cfg
    turns = list(trajectory.turns)
    by_turn = {t.index: t for t in turns}
    for turn in replays:
        if turn not in by_turn:
            loud(
                f"trajectory_advantages: replays for turn {turn}, but the trajectory "
                f"has {len(turns)} turn(s) ({sorted(by_turn)}). Escape: one entry per "
                f"turn, or drop it."
            )
        if by_turn[turn].call is None:
            loud(
                f"trajectory_advantages: turn {turn} made no tool call, so it is not a "
                f"fork and has nothing to replay. Escape: replays only for call turns."
            )
    actual = 1.0 if trajectory.passed else 0.0
    out: list[SpanAdvantage] = []
    for t in turns:
        if t.call is None:
            out.extend(SpanAdvantage(t.index, span, 0.0, "no_call") for span in t.supervised)
            continue
        rs = replays.get(t.index)
        if rs is None or len(rs) < cfg.min_replays:
            loud(
                f"trajectory_advantages: turn {t.index} called {t.call.name!r} and has "
                f"{0 if rs is None else len(rs)} replayed continuation(s); "
                f"min_replays={cfg.min_replays}. Without a replay the 'advantage' is "
                f"the raw pass flag. Escape: replay the turn, or lower min_replays."
            )
        baseline = sum(1.0 for r in rs if r) / len(rs)
        weight = actual - baseline
        out.extend(SpanAdvantage(t.index, span, weight, "fork") for span in t.supervised)
    return out
