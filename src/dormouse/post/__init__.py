"""The post-training lane: verifier + tasks + rollouts + CRR + the RL stepper.

The plan (`docs/architecture/post-training.md`, and the sweep in
`docs/research/2026-10-15-agentic-post-training.md`) orders this lane verifier-
first: nothing else can be measured without a reward that cannot lie. This
package is that floor plus the RL loop above it — and no LLM judge anywhere: at
this scale the judge would be the network under training (notes §3.1).

- [`tasks`] — the seedable synthetic task builder (code, math) and the
  `scripts/build_rl_tasks.py` JSONL loader.
- [`verifiers`] — the deterministic reward (`code` exec jail, `math` sympy).
- [`rollouts`] — the byte-budgeted rollout runner.
- [`train_steps`] — the CRR fork-replay advantage over a group.
- [`rl`] — the MiMo reward, the two advantage arms, the policy-gradient step.
- [`rl_train`] — the loop: one prompt per step, G rollouts, one step of PG.
- [`agent`] — the multi-turn rollout: generate, parse a call, run it in the
  verifier's jail, append the observation, continue (off by default; no import
  from this `__init__`, so a single-turn lane carries no multi-turn cost).
- [`agent_rl`] — trajectory-level fork-replay: one weight per supervised span.
"""

from .rl import RLConfig
from .rl_train import rl_train
from .rollouts import GREEDY_TEMP, STROKE_BYTES, Rollout, rollout
from .tasks import SFT_TASK_OFFSET, TASKS_PER_CLASS, Task, build_tasks, load_rl_tasks
from .train_steps import Advantage, crr_advantage
from .verifiers import TIMEOUT_S, Verdict, sympy_equivalent, verify

__all__ = [
    "GREEDY_TEMP",
    "RLConfig",
    "STROKE_BYTES",
    "SFT_TASK_OFFSET",
    "TASKS_PER_CLASS",
    "TIMEOUT_S",
    "Advantage",
    "Rollout",
    "Task",
    "Verdict",
    "build_tasks",
    "load_rl_tasks",
    "crr_advantage",
    "rl_train",
    "rollout",
    "sympy_equivalent",
    "verify",
]
