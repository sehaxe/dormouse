//! attention - KDA (burn-kda), the only attention arm.
//!
//! The MSA (block-sparse top-k) arm was cut with its crate (ADR-0014): broken
//! on the pre.4 stack and off in every preset since ADR-0012. The wrapper
//! struct stays for one reason only - it keeps the checkpoint param prefix
//! `loop_block.shared_attn.gdn2.*` stable across the cut.
use burn::tensor::Device;
use burn_kda::KdaModule;

/// Fused-path accounting for the two library seams this crate's arms cross,
/// as one line a training log can print (ADR-0019: a fused kernel that falls
/// back to tensor ops returns the RIGHT answer, so nothing but a counter
/// distinguishes "ran" from "fell back").
///
/// `(kda_fused_forward, kda_fused_backward)` are launch counters the library
/// already keeps (`burn_gdn2::fused_calls`); `(norm_asked, norm_skipped)` is
/// the fused RMSNorm. `norm_asked == norm_skipped` means the norm kernel never
/// engaged - the state on the trainer's autodiff backend, where the fused
/// kernel cannot take an autodiff tensor.
pub fn fused_seam_counts() -> (u64, u64, u64, u64) {
    #[cfg(all(feature = "cuda", feature = "std"))]
    let (kda_f, kda_b) = burn_gdn2::fused_calls();
    #[cfg(not(all(feature = "cuda", feature = "std")))]
    let (kda_f, kda_b) = (0, 0);
    let (norm_asked, norm_skipped) = burn_rmsnorm::fused::calls();
    (kda_f, kda_b, norm_asked, norm_skipped)
}

/// The gdn2 seam in FULL: `(asked, fused_fwd, fused_bwd, declined, ops_path,
/// custom_node_bwd)`.
///
/// The four-tuple above cannot answer the question that matters, and asking it
/// is how a day went missing: `fused kda=30/0` reads as "the fast path is
/// slow", when the truth was that the op was returning a LEAF and the arm
/// trained nothing. `asked` separates "never asked" from "asked and correctly
/// declined", `ops_path` is the arm that actually carries a gradient, and
/// `custom_node_backward` is the one number that separates "correct gradients"
/// from "no gradient at all" - zero while `ops_path` is positive means burn's
/// own graph did the backward.
///
/// Separate from `fused_seam_counts` so the trainer can print it without
/// changing the signature every other caller holds.
#[cfg(all(feature = "cuda", feature = "std"))]
pub fn kda_seam_counts() -> (u64, u64, u64, u64, u64, u64) {
    burn_gdn2::seam_counts()
}

#[cfg(not(all(feature = "cuda", feature = "std")))]
pub fn kda_seam_counts() -> (u64, u64, u64, u64, u64, u64) {
    (0, 0, 0, 0, 0, 0)
}

#[derive(Debug, burn::module::Module)]
pub struct AdaptiveAttention {
    pub gdn2: KdaModule,
}

impl AdaptiveAttention {
    pub fn new(d_model: usize, n_heads: usize, head_dim: usize, device: &Device) -> Self {
        let kda_cfg = burn_kda::KdaConfig {
            hidden_size: d_model,
            num_heads: n_heads,
            head_dim,
            // Report §2.1.1 wants the short causal conv, but on this box it
            // made fp32+AdamW NaN at ~step 60 (measured 2026-08-29; without
            // it the same recipe ran 150+ steps clean). Keep off until the
            // instability is traced inside burn-kda.
            use_short_conv: false,
            chunk_size: 16,
            ..Default::default()
        };
        Self {
            gdn2: KdaModule::new(&kda_cfg, 0.0, device),
        }
    }
}
