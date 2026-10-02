//! dormouse-core - all-bf16 mini Aria on burn-fused kernels (CUDA)
//!
//! # What this crate is
//!
//! The model: [`DormouseModel`] = Embedding -> [`LoopBlock`] (weight-shared,
//! run `max_iter` times) -> RMSNorm -> `lm_head`, every projection a
//! [`LinearLike`] (spectral low-rank TSCT factors, optional factor
//! quantization). Auxiliary heads ([`aux`]), the residual arms ([`gr`]) and
//! the per-arm execution counters ([`probe`]) are here too. Configuration is a
//! flat TOML schema ([`config`]); nothing in this crate loads weights or touches
//! the filesystem except `config::load_config`.
//!
//! # Contracts a reader of this crate must know
//!
//! These are the rules the rest of the project states as laws (AGENTS.md §1);
//! the code they are about says so at the item.
//!
//! * **Loud, counted, or silent — never silent** (ADR-0011/0019). Every
//!   degradation of a value, kernel or file carries one of three marks, and
//!   only two of them are legal. A fallback is not excused by having a
//!   defensible meaning; it is excused by the caller being able to tell which
//!   arm ran. Read [`probe`] before believing a fused path engaged.
//! * **Zero host-device synchronization** (ADR-0018 rule 2). No
//!   `into_scalar`, no `try_into_scalar`, no `blocking_read`, no host-side
//!   branch on a device value, in any forward in this crate. Every
//!   host-visible quantity is a device counter read at the caller's declared
//!   cadence. A forward that must branch on a device value branches *on the
//!   device*.
//! * **A claim names its evidence** (ADR-0018 rule 1, ADR-0020). Where this
//!   crate's arithmetic departs from the paper it cites, the doc comment at
//!   that item says which source it follows and which measurement is ours.
//!   "Verified" appears only with the file it was compared against.
//! * **A/B or delete** (ADR-0002). An arm that has not beaten its own removal
//!   on held-out BPB, 3 seeds per arm, is not a result. Doc comments mark the
//!   arms in that state as unmeasured rather than letting the name imply more.
//!
//! # Cost, in one paragraph
//!
//! This workload is launch-bound, not compute-bound (AGENTS.md §3.1: the GPU
//! idles through most of a warm step). GEMMs are a small fraction of a step,
//! and the per-step fixed cost — the retraction, the optimizer, and above all
//! the attention arm's per-iteration allocations — does not amortize with
//! batch size. So a change here is measured in launches and bytes of scratch,
//! not in FLOPs.
//!
//! # Gate
//!
//! `#![warn(missing_docs)]` and `#![warn(rustdoc::broken_intra_doc_links)]` are
//! on, and `RUSTFLAGS="-D warnings" cargo doc --no-deps -p dormouse-core -p
//! dormouse-data` is green. A public item without a doc comment fails CI.
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

pub mod act_quant;
pub mod attention;
pub mod aux;
pub mod config;
#[cfg(test)]
mod dspark_oracle;
pub mod future_byte;
pub mod gr;
pub mod loop_block;
pub mod mixture_probe;
pub mod model;
pub mod moe;
pub mod mor;
pub mod param;
pub mod probe;
pub mod routing;

pub use attention::{fused_seam_counts, kda_seam_counts, AdaptiveAttention};
pub use aux::{AuxHeads, TEACHER_MOMENTUM};
pub use config::{ActQuant, DormouseConfig};
pub use loop_block::{ExpertFFN, LoopBlock};
pub use model::DormouseModel;
pub use param::{LinearLike, TsctDiag};
pub use routing::{param_paths, Group, GroupCounts, Role, Routed, Routing};

/// FNV-1a 64-bit digest of a byte slice.
///
/// Present here for the crate's own tests and for anything that needs to agree
/// with the hashed memory's key derivation. **It is NOT the key derivation.**
/// The Engram keys are derived by `dormouse_data::raw_keys`, which takes the
/// low 31 bits of a 32-bit FNV-1a of each n-gram context — and dormouse-core
/// cannot depend on dormouse-data, so a second copy of that derivation here
/// would be exactly the drift this crate is not allowed to introduce (the same
/// reason there is deliberately no `forward_bytes` in `model.rs`: a bytes->logits
/// entry point here had to hand the memory branch `hashed_ids = None`, and
/// shipped `generate`/`serve` a network whose memory arm contributed literal
/// zeros, unlogged).
///
/// So: this is a utility with no callers in the model's hot path, it is not the
/// memory's hash, and a reader who needs the memory's hash must go to
/// `dormouse-data::raw_keys`.
pub fn fnv_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Int, Tensor};
    #[test]
    fn fnv_deterministic() {
        assert_eq!(fnv_hash(b"hello"), fnv_hash(b"hello"));
        assert_ne!(fnv_hash(b"hello"), fnv_hash(b"world"));
    }

    /// The in-loop L_Rec formula (gather the target log-prob) must equal the
    /// retired one-hot product row-for-row — the CE swap of 2026-09-04 was a
    /// bandwidth optimization, not a semantics change.
    #[test]
    fn gather_ce_matches_one_hot() {
        let dev = burn::tensor::Device::flex();
        let (n, v): (usize, usize) = (128, 256);
        let logits = Tensor::<2>::random([n, v], Distribution::Normal(0.0, 4.0), &dev);
        let tgt: Vec<i64> = (0..n).map(|i| (i * 7 + 3) as i64 % v as i64).collect();
        let lp = burn::tensor::activation::log_softmax(logits.clone(), 1);
        // new: gather; old: one-hot elementwise product
        let idx: Tensor<2, Int> =
            Tensor::from_data(burn::tensor::TensorData::new(tgt.clone(), [n, 1]), &dev);
        let gathered = lp.clone().gather(1, idx);
        let one_hot =
            Tensor::<1, Int>::from_data(burn::tensor::TensorData::new(tgt.clone(), [n]), &dev)
                .one_hot::<2>(v)
                .cast(burn::tensor::FloatDType::F32);
        let product = (lp * one_hot).sum_dim(1);
        let diff = (gathered - product).abs().max().into_scalar::<f32>();
        assert!(
            diff < 1e-4,
            "gather CE diverges from one-hot CE: {diff:.2e}"
        );
    }
}
