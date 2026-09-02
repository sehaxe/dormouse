//! dormouse config - all-bf16 scales, no fp32 master
use crate::act_quant::ActFormat;

/// BitNet a4.8-style activation quantization policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActQuant {
    /// e2m1 fp4 (the paper's a4).
    Fp4,
    /// Symmetric int with `bits` levels (attention path runs at max(bits, 8)).
    Int(u32),
}

impl From<ActQuant> for ActFormat {
    fn from(q: ActQuant) -> Self {
        match q {
            ActQuant::Fp4 => ActFormat::Fp4,
            ActQuant::Int(b) => ActFormat::Int(b),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DormouseConfig {
    /// bf16 storage mode: activations stored in bf16, compute in fp32 (this
    /// stack NaNs on mixed-dtype ops). Runtime cast policy only - weights and
    /// checkpoints are unaffected.
    pub bf16: bool,
    /// Some(policy) quantizes FFN activations (attention path at
    /// max(bits, 8)); None keeps them fp32.
    pub act_quant: Option<ActQuant>,
    /// Activation-quant scale group size (0 = per-token).
    pub act_group: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub d_ffn: usize,
    pub vocab: usize,
    pub max_seq_len: usize,
    pub max_iter: usize,
    pub rank: usize,
    pub halt_theta: f32,
    pub ponder_w: f32,
    pub rec_w: f32,
    pub guard_w: f32,
    pub norm_eps: f32,
    pub rope_base: f64,
    pub msa_topk: usize,
    pub msa_block: usize,
    /// Sparse-attention arm (burn-msa). A/B switch: pre.3 leaked autodiff
    /// nodes (~74 tensors/step) - re-enable after the leak is fixed.
    pub use_msa: bool,
    /// Gated-delta attention arm (burn-kda). A/B switch: its chunk-WY
    /// backward dominates step time (~3x at batch 12/s512).
    pub use_kda: bool,
    /// Engram n-gram memory arm (disable for bisecting).
    pub use_engram: bool,
    /// Gated Residual (Qwen3.8-Flash-Next §2.2) instead of pre-norm + ReZero:
    /// 4-branch residual stream, sigmoid-gated read, scalar writes. Off by
    /// default (checkpoint-compatible); the report's stability/loss wins come
    /// from this, so flip it on once the A/B shows it on this box.
    pub use_gr: bool,
    pub keep_frac: f32,
    pub dropout: f32,
    pub n_experts: usize,
    /// PonderNet KL weight β (Banino et al. 2021).
    pub ponder_beta: f32,
    /// JEPA auxiliary weight (data2vec 2.0, burn-jepa): the EMA-teacher
    /// latent is predicted at span-masked positions. 0 = off. Pure helper -
    /// CE stays the primary objective; causality forbids the identity
    /// shortcut, so the latent must encode predictive abstractions.
    pub jepa_weight: f32,
    /// JEPA masked-position fraction (span-dilated masks, burn-jepa).
    pub jepa_mask_frac: f32,
    /// JEPA contiguous mask span length.
    pub jepa_mask_span: usize,
    /// DSpark auxiliary weight (DeepSeek draft head, used instead of MTP):
    /// corrects frozen backbone logits into the next-K tokens. 0 = off.
    pub dspark_weight: f32,
    /// DSpark draft depth K (tokens per anchor window).
    pub dspark_k: usize,
    /// DSpark anchor stride in bytes.
    pub dspark_stride: usize,
    /// PonderNet geometric-prior parameter λ_p (sets expected halting steps ≈ 1/λ_p).
    pub ponder_prior: f32,
}

impl DormouseConfig {
    pub fn small() -> Self {
        Self {
            d_model: 768,
            n_heads: 12,
            head_dim: 64,
            d_ffn: 2048,
            vocab: 256,
            max_seq_len: 512,
            max_iter: 8,
            rank: 64,
            halt_theta: 0.9,
            ponder_w: 0.05,
            rec_w: 0.5,
            guard_w: 0.01,
            norm_eps: 1e-3,
            rope_base: 10000.0,
            msa_topk: 8,
            msa_block: 32,
            bf16: false,
            act_quant: None,
            act_group: 0,
            use_msa: true,
            use_kda: true,
            use_engram: true,
            use_gr: false,
            keep_frac: 0.5,
            dropout: 0.0,
            n_experts: 3,
            ponder_beta: 0.01,
            ponder_prior: 2.0 / 9.0,
            jepa_weight: 0.05,
            jepa_mask_frac: 0.15,
            jepa_mask_span: 8,
            dspark_weight: 0.1,
            dspark_k: 4,
            dspark_stride: 16,
        }
    }
    pub fn base() -> Self {
        Self {
            d_model: 1024,
            n_heads: 16,
            head_dim: 64,
            d_ffn: 2816,
            vocab: 256,
            max_seq_len: 1024,
            max_iter: 8,
            rank: 64,
            halt_theta: 0.9,
            ponder_w: 0.05,
            rec_w: 0.5,
            guard_w: 0.01,
            norm_eps: 1e-3,
            rope_base: 10000.0,
            msa_topk: 8,
            msa_block: 32,
            bf16: false,
            act_quant: None,
            act_group: 0,
            use_msa: true,
            use_kda: true,
            use_engram: true,
            use_gr: false,
            keep_frac: 0.5,
            dropout: 0.0,
            n_experts: 3,
            ponder_beta: 0.01,
            ponder_prior: 2.0 / 9.0,
            jepa_weight: 0.05,
            jepa_mask_frac: 0.15,
            jepa_mask_span: 8,
            dspark_weight: 0.1,
            dspark_k: 4,
            dspark_stride: 16,
        }
    }
    /// 1B config - d=2048, 24 layers equivalent via loop 8*3 experts
    pub fn one_b() -> Self {
        Self {
            d_model: 2048,
            n_heads: 32,
            head_dim: 64,
            d_ffn: 5632,
            vocab: 256,
            max_seq_len: 2048,
            max_iter: 12,
            rank: 64,
            halt_theta: 0.9,
            ponder_w: 0.05,
            rec_w: 0.5,
            guard_w: 0.01,
            norm_eps: 1e-3,
            rope_base: 10000.0,
            msa_topk: 8,
            msa_block: 32,
            bf16: false,
            act_quant: None,
            act_group: 0,
            use_msa: true,
            use_kda: true,
            use_engram: true,
            use_gr: false,
            keep_frac: 0.5,
            dropout: 0.0,
            n_experts: 4,
            ponder_beta: 0.01,
            ponder_prior: 2.0 / 13.0,
            jepa_weight: 0.05,
            jepa_mask_frac: 0.15,
            jepa_mask_span: 8,
            dspark_weight: 0.1,
            dspark_k: 4,
            dspark_stride: 16,
        }
    }
}
