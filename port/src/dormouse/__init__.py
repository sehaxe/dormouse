"""dormouse — PyTorch port of the burn trainer.

Day one carries only the three mechanisms nothing in the ecosystem provides
(TSCT low-rank experts, hashed n-gram memory, the byte stream) plus the
instrument layer and the trainer skeleton. Everything else is a dependency:
FLA for the gated-delta attention (optional, LOUD when absent; SDPA is the
COUNTED fallback and the CPU test path).

The port's design record is `docs/decisions/2026-10-15-1b-readiness-plan.md`
§7 (the port lane) and `docs/architecture/bf16-plan.md` (precision).
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from .config import ModelConfig, RunConfig, TrainConfig, load_preset, load_run
from .instrument import Counters, LoudError
from .model import DormouseModel, param_count

#: The ONE version: `pyproject.toml` declares it and the installed metadata
#: carries it; a bare source tree (no metadata) reports `0.0.0+unknown` rather
#: than inventing a version.
try:
    __version__ = _dist_version("dormouse")
except PackageNotFoundError:  # `PYTHONPATH=src` without an install
    __version__ = "0.0.0+unknown"

__all__ = [
    "Counters",
    "DormouseModel",
    "LoudError",
    "ModelConfig",
    "RunConfig",
    "TrainConfig",
    "__version__",
    "load_preset",
    "load_run",
    "param_count",
]
