"""Self-play pretraining with zero data — arXiv:2609.30063, CPU core.

The paper's loop, one module per piece of it:

- [`utm`] — the Brainf*ck-like universal machine `U` (Appendix E): byte tape,
  loops, read/emit, random input tape, and the totality property that makes
  **every** instruction string executable.
- [`reward`] — Eq. 2's preconditioned gradient-alignment reward, forward mode.
- [`pool`] — Appendix G's program pool (fresh ∪ mutations ∪ replay), the bank,
  Eq. 4's stored sampling probabilities and Eq. 5's EI weights.
- [`loop`] — §2.2's round: generate → execute → learner step + generator step.

The learner and the generator are both plain byte-level language models
(`dormouse.model.DormouseModel`); nothing here touches the trainer, and the
whole package is CPU-runnable.
"""

from .loop import (
    RoundConfig,
    RoundStats,
    generator_step,
    learner_step,
    program_logprobs,
    round,
    sample_from_generator,
)
from .pool import (
    RHO_CLIP,
    BankEntry,
    Pool,
    PoolConfig,
    PoolItem,
    ProgramPool,
    advantages,
    ei_weights,
    importance_ratio,
    mutate,
    prior_logprob,
    sample_prior,
)
from .reward import (
    DEFAULT_EPS,
    OUT_PREFIX,
    PROGRAM_PREFIX,
    adamw_preconditioner,
    gradient_alignment_rewards,
    lookback,
    output_batch,
    sequence_ce,
)
from .utm import (
    ALPHABET,
    END,
    INSTRUCTIONS,
    MACROS,
    MODULUS,
    PROGRAM_TOKENS,
    TAPE_SIZE,
    run,
)

__all__ = [
    "ALPHABET",
    "DEFAULT_EPS",
    "END",
    "INSTRUCTIONS",
    "MACROS",
    "MODULUS",
    "OUT_PREFIX",
    "PROGRAM_PREFIX",
    "PROGRAM_TOKENS",
    "RHO_CLIP",
    "TAPE_SIZE",
    "BankEntry",
    "Pool",
    "PoolConfig",
    "PoolItem",
    "ProgramPool",
    "RoundConfig",
    "RoundStats",
    "adamw_preconditioner",
    "advantages",
    "ei_weights",
    "generator_step",
    "gradient_alignment_rewards",
    "importance_ratio",
    "learner_step",
    "lookback",
    "mutate",
    "output_batch",
    "prior_logprob",
    "program_logprobs",
    "round",
    "run",
    "sample_from_generator",
    "sample_prior",
    "sequence_ce",
]
