//! attention - KDA (dormouse-kda), the only attention arm.
//!
//! The MSA (block-sparse top-k) arm was cut with its crate (ADR-0014): broken
//! on the pre.4 stack and off in every preset since ADR-0012. The wrapper
//! struct stays for one reason only - it keeps the checkpoint param prefix
//! `loop_block.shared_attn.gdn2.*` stable across the cut.
//!
//! # Invariants and failure modes
//!
//! * **Everything measurable here is a counter, not a value.** Both functions
//!   in this module return launch counts. A count of 0 is not "fast", it is
//!   "did not run", and the pair that matters is `asked` vs `fused_fwd` — a
//!   fused forward with a fused backward of 0 is a forward that returned a
//!   leaf, which trained nothing and looked healthy (the `8fa5d4c` defect).
//! * **No host sync.** Neither function reads a tensor; they read atomics the
//!   library incremented on the device. Safe to call in a hot loop's log path
//!   (AGENTS.md §1.3).
//! * **The fused path is not reachable on the trainer's backend**, and the
//!   counters are how a reader finds out. `burn-dispatch` refuses the
//!   autodiff→bare demotion unless the dispatch autodiff context is
//!   `Disabled`, so every training forward and every eval takes the tensor-ops
//!   route. `norm_asked == norm_skipped` is the same statement about
//!   `dormouse-rmsnorm`, and it held on every reading in this project's history
//!   (1560 asks per 500 steps, 0 runs).
//! * **A fused forward that runs and is discarded is a defect**, not a speedup.
//!   That was found on 2026-09-30 and is still open; it is why the counters
//!   are per-direction rather than a single "ran" flag.
use burn::tensor::Device;
use dormouse_kda::KdaModule;

/// Fused-path accounting for the two library seams this crate's arms cross,
/// as one line a training log can print (ADR-0019: a fused kernel that falls
/// back to tensor ops returns the RIGHT answer, so nothing but a counter
/// distinguishes "ran" from "fell back").
///
/// `(kda_fused_forward, kda_fused_backward)` are launch counters the library
/// already keeps (`dormouse_gdn2::fused_calls`); `(norm_asked, norm_skipped)` is
/// the fused RMSNorm. `norm_asked == norm_skipped` means the norm kernel never
/// engaged - the state on the trainer's autodiff backend, where the fused
/// kernel cannot take an autodiff tensor.
pub fn fused_seam_counts() -> (u64, u64, u64, u64) {
    #[cfg(all(feature = "cuda", feature = "std"))]
    let (kda_f, kda_b) = dormouse_gdn2::fused_calls();
    #[cfg(not(all(feature = "cuda", feature = "std")))]
    let (kda_f, kda_b) = (0, 0);
    let (norm_asked, norm_skipped) = dormouse_rmsnorm::fused::calls();
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
    dormouse_gdn2::seam_counts()
}

/// The gdn2 seam in FULL, on a build with no CUDA: six zeros.
///
/// The same counter set as the CUDA build, so a reader of a non-CUDA training
/// log sees `0/0/0/...` and can conclude "this backend has no fused path",
/// rather than having the field absent.
#[cfg(not(all(feature = "cuda", feature = "std")))]
pub fn kda_seam_counts() -> (u64, u64, u64, u64, u64, u64) {
    (0, 0, 0, 0, 0, 0)
}

/// The attention wrapper: KDA (Kimi Delta Attention) as the crate's one and
/// only attention arm.
///
/// Deliberately a one-field struct rather than a re-export of
/// [`dormouse_kda::KdaModule`], and that is the whole reason it exists: the
/// checkpoint parameter prefix `loop_block.shared_attn.gdn2.*` is part of the
/// on-disk format, so the path from the model's attribute to the file's name
/// has to survive the MSA arm's deletion (ADR-0014). Renaming this field would
/// make every existing checkpoint refuse to load.
///
/// Cost, from a warm-step measurement (2026-09-30, `d8fa449`): the KDA
/// backward is ~205 ms of a ~480 ms step at 9.2M params, depth 2 — the single
/// largest term, and it is per-sequence-length work, not per-parameter.
#[derive(Debug, burn::module::Module)]
pub struct AdaptiveAttention {
    /// The KDA module. Named for the checkpoint prefix, not for its contents
    /// (dormouse-kda's gated-delta kernel). Do not rename.
    pub gdn2: KdaModule,
}

impl AdaptiveAttention {
    /// Build the arm at `(d_model, n_heads, head_dim)`.
    ///
    /// Two fields are set by this repo's measurement rather than by the paper:
    /// `use_short_conv` is OFF (report §2.1.1 asks for it; it made fp32 +
    /// AdamW NaN at ~step 60 on this box on 2026-08-29, while the same recipe
    /// without it ran 150+ steps clean — the instability is untraced inside
    /// dormouse-kda, so it stays off), and `chunk_size` is 16.
    ///
    /// `KdaModule::new`'s second argument is the dropout rate, 0.0: dormouse
    /// has no dropout anywhere.
    pub fn new(d_model: usize, n_heads: usize, head_dim: usize, device: &Device) -> Self {
        let kda_cfg = dormouse_kda::KdaConfig {
            hidden_size: d_model,
            num_heads: n_heads,
            head_dim,
            // Report §2.1.1 wants the short causal conv, but on this box it
            // made fp32+AdamW NaN at ~step 60 (measured 2026-08-29; without
            // it the same recipe ran 150+ steps clean). Keep off until the
            // instability is traced inside dormouse-kda.
            use_short_conv: false,
            chunk_size: 16,
            ..Default::default()
        };
        Self {
            gdn2: KdaModule::new(&kda_cfg, 0.0, device),
        }
    }
}
