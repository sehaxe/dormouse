//! dormouse config - all-bf16 scales, no fp32 master
#[derive(Debug, Clone)]
pub struct DormouseConfig {
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
    /// burn-msa on pre.3 leaks autodiff nodes (~74 tensors/step): a long run
    /// OOMs regardless of pool mode. Default off; env DM_NO_MSA / DM_MSA=1
    /// override. Re-enable once burn-msa is fixed.
    pub use_msa: bool,
    /// burn-kda chunk-WY backward dominates step time (~3x slowdown at
    /// batch 12/s512). Default off for speed; DM_NO_KDA / DM_KDA=1 override.
    pub use_kda: bool,
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
            use_msa: true,
            use_kda: true,
            use_gr: false,
            keep_frac: 0.5,
            dropout: 0.0,
            n_experts: 3,
            ponder_beta: 0.01,
            ponder_prior: 2.0 / 9.0,
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
            use_msa: true,
            use_kda: true,
            use_gr: false,
            keep_frac: 0.5,
            dropout: 0.0,
            n_experts: 3,
            ponder_beta: 0.01,
            ponder_prior: 2.0 / 9.0,
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
            use_msa: true,
            use_kda: true,
            use_gr: false,
            keep_frac: 0.5,
            dropout: 0.0,
            n_experts: 4,
            ponder_beta: 0.01,
            ponder_prior: 2.0 / 13.0,
        }
    }
}
