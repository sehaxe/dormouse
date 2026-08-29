//! model - Embedding -> LoopBlock -> RMSNorm -> lm_head, all-bf16 capable
use burn::backend::{Backend, DispatchKindConversion};
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::{Device, DispatchTensor, FloatDType, Int, Tensor, TensorData};
use burn_rmsnorm::RMSNorm;

use crate::config::DormouseConfig;
use crate::loop_block::LoopBlock;
use crate::param::LinearLike;

#[derive(Module, Debug)]
pub struct DormouseModel {
    pub embedding: Embedding,
    pub loop_block: LoopBlock,
    pub norm: RMSNorm,
    pub lm_head: LinearLike,
    #[module(skip)]
    pub vocab_size: usize,
    #[module(skip)]
    pub d_model: usize,
    #[module(skip)]
    pub ponder_beta: f32,
    #[module(skip)]
    pub ponder_prior: f32,
}

impl DormouseModel {
    pub fn new(cfg: &DormouseConfig, device: &Device) -> Self {
        let d = cfg.d_model;
        let v = cfg.vocab;
        Self {
            embedding: EmbeddingConfig::new(v, d).init(device),
            loop_block: LoopBlock::new(cfg, device),
            norm: RMSNorm::new(d, cfg.norm_eps, device),
            lm_head: LinearLike::new(d, v, cfg.rank.min(d).min(v), device),
            vocab_size: v,
            d_model: d,
            ponder_beta: cfg.ponder_beta,
            ponder_prior: cfg.ponder_prior,
        }
    }

    pub fn forward<B: Backend>(
        &self,
        input_ids: Tensor<2, Int>,
        hashed_ids: Option<Tensor<3, Int>>,
    ) -> Tensor<3>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        self.forward_with_hidden::<B>(input_ids, hashed_ids, None, None).0
    }

    /// Returns (logits, L_Rec [1], p_dist [b,N], kda). When `targets` is Some,
    /// the per-step reconstruction loss is accumulated inside the loop block so
    /// the model never slices a 4D autodiff tensor (cubecl/sm_120 stability).
    /// `host_rows` carries pre-gathered n-gram rows `[b,t,3*32]` for the
    /// RAM-offload path (see dormouse-train offload); when None, `hashed_ids`
    /// drives the in-model tables.
    pub fn forward_with_hidden<B: Backend>(
        &self,
        input_ids: Tensor<2, Int>,
        hashed_ids: Option<Tensor<3, Int>>,
        host_rows: Option<Tensor<3>>,
        targets: Option<Tensor<2, Int>>,
    ) -> (Tensor<3>, Tensor<1>, Tensor<2>, Tensor<4>)
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let x = self.embedding.forward(input_ids);
        let x = if crate::param::bf16_on() {
            x.cast(FloatDType::BF16)
        } else {
            x
        };
        let [b, t, _d] = x.dims();
        let v = self.vocab_size;
        let tgt = targets.map(|tg| tg.reshape([b * t]).one_hot::<2>(v).cast(FloatDType::F32));
        let (out_acc, rec, p_dist, kda) =
            self.loop_block
                .forward_full_state::<B>(x, hashed_ids, host_rows, None, tgt, &self.lm_head);
        // loop activations may be bf16; the final norm+head compute in fp32
        // (bf16 logits make the softmax/CE numerically unstable -> NaN).
        let h = if crate::param::bf16_on() {
            self.norm.forward(out_acc.cast(FloatDType::F32))
        } else {
            self.norm.forward(out_acc)
        };
        let logits = self
            .lm_head
            .forward::<B>(h.reshape([b * t, self.d_model]))
            .reshape([b, t, self.vocab_size]);
        (logits, rec, p_dist, kda)
    }

    /// PonderNet loss (Banino et al. 2021): L = L_Rec + β·KL(p || Geom(λ_p)).
    pub fn loss<B: Backend>(&self, rec_ce: Tensor<1>, p_dist: Tensor<2>) -> Tensor<1>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let kl = self.ponder_kl(p_dist, self.ponder_prior); // [1]
        rec_ce + kl.mul_scalar(self.ponder_beta)
    }

    /// KL(p_dist || truncated-geometric(λ_p)); p_dist is [b, N].
    fn ponder_kl(&self, p_dist: Tensor<2>, lambda_p: f32) -> Tensor<1> {
        let n = p_dist.dims()[1];
        let b = p_dist.dims()[0];
        let dev = p_dist.device();
        // Truncated-geometric prior pmf over the N halting steps, renormalized
        // so the support sums to 1.
        let mut prior = Vec::with_capacity(n);
        let mut mass = 1.0f32;
        let mut total = 0.0f32;
        for _ in 0..n {
            let prob = lambda_p * mass;
            prior.push(prob);
            total += prob;
            mass *= 1.0 - lambda_p;
        }
        let inv = 1.0 / total;
        let prior_v: Vec<f32> = prior.iter().map(|x| x * inv).collect();
        let prior_t = Tensor::<1>::from_data(TensorData::new(prior_v, [n]), &dev).log();
        let log_p = p_dist.clone().log();
        let term = p_dist * (log_p - prior_t.unsqueeze_dim::<2>(0)); // [b, N]
        // Mean over batch and steps -> [1]. The double sum + reshape is robust
        // to whether sum_dim keeps the reduced dimension.
        term.sum_dim(1).sum_dim(0).reshape([1]).div_scalar((b * n) as f32)
    }

    /// Apply a quantization format to every TSCT factor in the model
    /// (experts, readout/router projections, lm_head).
    pub fn set_quant_all(&mut self, quant: burn_spectral::QuantFormat) {
        self.loop_block.set_quant_all(quant);
        self.loop_block.out_proj.set_quant(quant);
        self.loop_block.shared_attn.router.set_quant(quant);
        self.lm_head.set_quant(quant);
    }

    /// Polar-retract every TSCT factor U/V to orthonormal (on device, keeps
    /// autodiff tracking). Call every step during training: without it the
    /// factors drift and the quantized forward degrades into NaN.
    pub fn retract_tsct(&mut self, iters: usize) {
        self.loop_block.retract_tsct(iters);
        self.lm_head.retract(iters);
    }

    /// Worst orthonormality error across all TSCT factors (syncs the device;
    /// monitor at cadence, not per step).
    pub fn max_ortho(&self) -> f32 {
        self.loop_block.max_ortho().max(self.lm_head.max_ortho())
    }

    /// Inference: bytes -> last-token logits [vocab].
    pub fn forward_bytes<B: Backend>(&self, bytes: &[u8]) -> Vec<f32>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let device = self.embedding.weight.device();
        let ids: Vec<i64> = bytes.iter().map(|&b| b as i64).collect();
        let x: Tensor<2, Int> = Tensor::from_data(TensorData::new(ids, [1, bytes.len().max(1)]), &device);
        let logits = self.forward::<B>(x, None);
        let [_, t, v] = logits.dims();
        logits.slice([0..1, t - 1..t, 0..v]).reshape([v]).into_data().try_to_vec().unwrap_or_else(|_| vec![0.0; v])
    }

    pub fn max_seq_len(&self) -> usize {
        4096
    }
}