//! # burn-msa - Qwen Sparse Attention (QSA), the re-entry ADR-0014 named
//!
//! | Source | Part | What |
//! |--------|------|------|
//! | `docs/papers/tech_report-qwen3.8-flash-next.pdf` §QSA | Eq. 12-15 | compressed MQA block indexer |
//! | same | Eq. 16 | top-`KB` block selection, tail tokens always included |
//! | same | Eq. 17-18 | dense-distillation KL into the indexer |
//!
//! **One word about why this crate is new and not the ADR-0014 one.** The old
//! `burn-msa` was deleted (ADR-0014) because its indexer emitted garbage block
//! indices and every gather went out of bounds. This crate is written against
//! the same constraints its re-entry conditions name: indices in range by
//! construction and under the vendored `cubek-reduce` fix (ADR-0015; the
//! open "partial row" defect means a `kb`-top over fewer visible blocks may
//! repeat a visible column — in range, CAUSAL, and priced in the A/B
//! findings doc, never out of bounds and never a future token), and a
//! dense-contract gate — sparse attention over ALL blocks must equal dense
//! attention bit-near `sparse_attention(models..., kb = B)` — ahead of any
//! A/B.
//!
//! Structure notes:
//!
//! - The **indexer** is a real burn `Module`: it carries the learned Wq/Wk
//!   the distillation trains. The core attention's QKV projections stay the
//!   model's own `LinearLike`, so padding and the quant seams keep one owner.
//! - The **core sparse attention** is a pure function over already-projected
//!   q/k/v: gather the selected blocks' keys/values, attention over the
//!   assembled window, one softmax. No per-query loop; no dynamic 4D
//!   slicing (the sm_120 rule) — the window is a static `W = kb*r + tail`
//!   rank.
//! - The **teacher** is the same projected q/k/v under a plain causal mask;
//!   distillation (Eq. 17-18) max-pools the head-summed distribution into
//!   blocks and minimizes the KL against the indexer's scores.
//!
//! `kb` counts COMPLETE blocks and is a caller-owned budget (the paper's
//! `KB = ceil(K/r)`); the caller refuses `kb >= B` LOUDLY in config, so this
//! crate may treat `kb < B` as its working invariant and the all-blocks
//! call (`kb == B`) as the dense-equivalence gate, not a setting.
#![cfg_attr(test, allow(deprecated))]

pub mod distill;
pub mod indexer;
pub mod sparse;

pub use indexer::{IndexerConfig, IndexerModule, indexer_scores, select_blocks};
pub use sparse::{NEG, dense_attention, sparse_attention, sparse_mask};
pub use distill::{distill_loss, pool_teacher};

/// Block geometry of a full sequence: `B = t / r` complete blocks (the ones
/// the indexer scores) and `tail = t - B*r` tokens of the final incomplete
/// block, which are always attended (Eq. 19).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MsaInfo {
    /// Block compression ratio `r` (tokens per block).
    pub block_r: usize,
    /// Complete blocks the indexer scores: `B = t / r` (floor).
    pub blocks: usize,
    /// Tokens of the final incomplete block, always in the window.
    pub tail: usize,
}

pub fn blocks(t: usize, block_r: usize) -> MsaInfo {
    let blocks = t / block_r;
    MsaInfo {
        block_r,
        blocks,
        tail: t - blocks * block_r,
    }
}
