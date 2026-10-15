"""The post-training lane, state 1: verifier + tasks + rollouts + CRR advantage.

The plan (`docs/architecture/post-training.md`, and the sweep in
`docs/research/2026-10-15-agentic-post-training.md`) orders this lane verifier-
first: nothing else can be measured without a reward that cannot lie. This
package is that floor and nothing above it — no trainer, no CLI, no LLM judge.

- [`tasks`] — the seedable synthetic task builder (code, math).
- [`verifiers`] — the deterministic reward (`code` exec jail, `math` sympy).
- [`rollouts`] — the byte-budgeted rollout runner.
- [`train_steps`] — the CRR fork-replay advantage over a group.
"""

from .rollouts import GREEDY_TEMP, STROKE_BYTES, Rollout, rollout
from .tasks import TASKS_PER_CLASS, Task, build_tasks
from .train_steps import Advantage, crr_advantage
from .verifiers import TIMEOUT_S, Verdict, sympy_equivalent, verify

__all__ = [
    "GREEDY_TEMP",
    "STROKE_BYTES",
    "TASKS_PER_CLASS",
    "TIMEOUT_S",
    "Advantage",
    "Rollout",
    "Task",
    "Verdict",
    "build_tasks",
    "crr_advantage",
    "rollout",
    "sympy_equivalent",
    "verify",
]
