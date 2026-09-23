#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MsaConfig {
    pub d_model: usize,
    pub n_heads_q: usize,
    pub n_heads_kv: usize,
    pub d_head: usize,
    pub d_idx: usize,
    pub block_size: usize,
    pub topk: usize,
    pub force_local_block: bool,
    pub causal: bool,
    pub use_kl_loss: bool,
    pub warmup_steps: usize,
    pub kl_coeff: f64,
    pub gradient_detach: bool,
    /// Apply rotary position embeddings to q/k (MSA paper: proxy branch
    /// selects blocks, real attention still needs position info).
    pub use_rope: bool,
    pub rope_base: f64,
    pub rope_max_seq_len: usize,
}

impl Default for MsaConfig {
    fn default() -> Self {
        Self {
            d_model: 1152,
            n_heads_q: 18,
            n_heads_kv: 1,
            d_head: 64,
            d_idx: 32,
            block_size: 128,
            topk: 16,
            force_local_block: true,
            causal: true,
            use_kl_loss: true,
            warmup_steps: 1000,
            kl_coeff: 0.1,
            gradient_detach: true,
            use_rope: false,
            rope_base: 10000.0,
            rope_max_seq_len: 32768,
        }
    }
}

impl MsaConfig {
    pub fn new(
        d_model: usize,
        n_heads_q: usize,
        n_heads_kv: usize,
        d_head: usize,
        d_idx: usize,
    ) -> Self {
        Self {
            d_model,
            n_heads_q,
            n_heads_kv,
            d_head,
            d_idx,
            ..Self::default()
        }
    }

    pub fn qhead_per_kv(&self) -> usize {
        self.n_heads_q / self.n_heads_kv
    }
    pub fn max_attended_tokens(&self) -> usize {
        self.topk * self.block_size
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_heads_q == 0 || !self.n_heads_q.is_multiple_of(self.n_heads_kv) {
            return Err(format!(
                "n_heads_q ({}) must be divisible by n_heads_kv ({})",
                self.n_heads_q, self.n_heads_kv
            ));
        }
        if self.d_model == 0
            || self.d_head == 0
            || self.d_idx == 0
            || self.block_size == 0
            || self.topk == 0
        {
            return Err("all dimensions must be > 0".into());
        }
        if !self.d_model.is_multiple_of(self.d_head) {
            return Err(format!(
                "d_model ({}) must be divisible by d_head ({})",
                self.d_model, self.d_head
            ));
        }
        Ok(())
    }
}
