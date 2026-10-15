"""dormouse — PyTorch port of the burn trainer.

Day one carries only the three mechanisms nothing in the ecosystem provides
(TSCT low-rank experts, hashed n-gram memory, the byte stream) plus the
instrument layer and the trainer skeleton. Everything else is a dependency:
FLA for the gated-delta attention (optional, LOUD when absent; SDPA is the
COUNTED fallback and the CPU test path).

The port's design record is `docs/decisions/2026-10-15-1b-readiness-plan.md`
§7 (the port lane) and `docs/architecture/bf16-plan.md` (precision).
"""

from .config import ModelConfig, RunConfig, TrainConfig, load_preset, load_run
from .instrument import Counters, LoudError
from .model import DormouseModel, param_count

__all__ = [
    "Counters",
    "DormouseModel",
    "LoudError",
    "ModelConfig",
    "RunConfig",
    "TrainConfig",
    "load_preset",
    "load_run",
    "param_count",
]
