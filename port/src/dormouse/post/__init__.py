"""The post-training lane: verifier + tasks + rollouts + CRR + the RL stepper.

The plan (`docs/architecture/post-training.md`, and the sweep in
`docs/research/2026-10-15-agentic-post-training.md`) orders this lane verifier-
first: nothing else can be measured without a reward that cannot lie. This
package is that floor plus the RL loop above it — and no LLM judge anywhere: at
this scale the judge would be the network under training (notes §3.1).

- [`tasks`] — the seedable synthetic task builder (code, math).
- [`verifiers`] — the deterministic reward (`code` exec jail, `math` sympy).
- [`rollouts`] — the byte-budgeted rollout runner.
- [`train_steps`] — the CRR fork-replay advantage over a group.
- [`rl`] — the MiMo reward, the two advantage arms, the policy-gradient step.
- [`rl_train`] — the loop: one prompt per step, G rollouts, one step of PG.
"""

from .rl import RLConfig
from .rl_train import rl_train
from .rollouts import GREEDY_TEMP, STROKE_BYTES, Rollout, rollout
from .tasks import TASKS_PER_CLASS, Task, build_tasks
from .train_steps import Advantage, crr_advantage
from .verifiers import TIMEOUT_S, Verdict, sympy_equivalent, verify

__all__ = [
    "GREEDY_TEMP",
    "RLConfig",
    "STROKE_BYTES",
    "TASKS_PER_CLASS",
    "TIMEOUT_S",
    "Advantage",
    "Rollout",
    "Task",
    "Verdict",
    "build_tasks",
    "crr_advantage",
    "rl_train",
    "rollout",
    "sympy_equivalent",
    "verify",
]
