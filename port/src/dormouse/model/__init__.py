"""The model: TSCT linear, dual-latent MoE experts, KDA, Engram, loop block,
ByteFlow (the five-stage net and the channel that runs the loop on its latents)."""

from .byteflow import (
    VOCAB,
    BfChannel,
    ByteFlowConfig,
    ByteFlowNet,
    CanonLayer,
    RateMode,
    RatePatcher,
    bf_for_model,
    byteflow_config,
    coding_rate_exact,
    compat_armed,
    effective_bins,
    marginal_gains_exact,
    marginal_gains_l2,
    select_positions,
)
from .engram import Engram, fnv, raw_keys, table_slots
from .experts import DualLatentGLU, SparseMoE
from .kda import KDAAttention, fla_status
from .loopblock import LoopBlock, memory_floor_mix
from .model import DormouseModel, param_count
from .tsct import DenseLinear, TSCTLinear, make_projection

__all__ = [
    "VOCAB",
    "BfChannel",
    "ByteFlowConfig",
    "ByteFlowNet",
    "CanonLayer",
    "DenseLinear",
    "DormouseModel",
    "DualLatentGLU",
    "Engram",
    "KDAAttention",
    "LoopBlock",
    "RateMode",
    "RatePatcher",
    "SparseMoE",
    "TSCTLinear",
    "bf_for_model",
    "byteflow_config",
    "coding_rate_exact",
    "compat_armed",
    "effective_bins",
    "fla_status",
    "fnv",
    "make_projection",
    "marginal_gains_exact",
    "marginal_gains_l2",
    "memory_floor_mix",
    "param_count",
    "raw_keys",
    "select_positions",
    "table_slots",
]
