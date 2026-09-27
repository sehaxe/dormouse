//! attention - KDA (burn-kda), the only attention arm.
//!
//! The MSA (block-sparse top-k) arm was cut with its crate (ADR-0014): broken
//! on the pre.4 stack and off in every preset since ADR-0012. The wrapper
//! struct stays for one reason only - it keeps the checkpoint param prefix
//! `loop_block.shared_attn.gdn2.*` stable across the cut.
use burn::tensor::Device;
use burn_kda::KdaModule;

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
