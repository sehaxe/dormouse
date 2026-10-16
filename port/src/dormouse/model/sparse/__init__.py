"""MSA — the block-sparse second attention arm (readiness-plan §3.5 item 4).

Two files, two tracks:

* `sparse.FlexSparseSelector` — the block INDEXER: MQA (4 q-heads, 1 shared
  k-head), keys avg-pooled into blocks of `r`, block-causal ReLU scores, top-K
  per query. Pure torch; its product is the selected BLOCK ADDRESSES.
* `attention.BlockSparseAttention` — the ARM: a config-driven, off-by-default
  alternative to KDA, whose aggregator runs dense attention over the gathered
  keys of the selected blocks. Kernel interface is pure torch in / torch out.
"""

from .attention import MSA_BLOCK, MSA_TOPK, BlockSparseAttention
from .sparse import MSA_Q_HEADS, FlexSparseSelector

__all__ = [
    "MSA_BLOCK",
    "MSA_Q_HEADS",
    "MSA_TOPK",
    "BlockSparseAttention",
    "FlexSparseSelector",
]
