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
        self.forward_with_hidden::<B>(input_ids, hashed_ids).0
    }

    /// Returns (logits, per-step hidden states [N,b,t,d], p_dist [b,N], kda).
    pub fn forward_with_hidden<B: Backend>(
        &self,
        input_ids: Tensor<2, Int>,
        hashed_ids: Option<Tensor<3, Int>>,
    ) -> (Tensor<3>, Tensor<4>, Tensor<2>, Tensor<4>)
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let x = self.embedding.forward(input_ids);
        let x = if crate::param::bf16_on() {
            x.cast(FloatDType::BF16)
        } else {
            x
        };
        let (out_acc, step_hiddens, p_dist, kda) =
            self.loop_block.forward_full_state::<B>(x, hashed_ids, None);
        // loop activations may be bf16; the final norm+head compute in fp32
        // (bf16 logits make the softmax/CE numerically unstable -> NaN).
        let h = if crate::param::bf16_on() {
            self.norm.forward(out_acc.cast(FloatDType::F32))
        } else {
            self.norm.forward(out_acc)
        };
        let b = h.dims()[0];
        let t = h.dims()[1];
        let logits = self
            .lm_head
            .forward::<B>(h.reshape([b * t, self.d_model]))
            .reshape([b, t, self.vocab_size]);
        (logits, step_hiddens, p_dist, kda)
    }

    /// PonderNet loss (Banino et al. 2021): L = Σ_n p_n·CE(ŷ_n, y) + β·KL(p || Geom(λ_p)).
    pub fn loss<B: Backend>(
        &self,
        step_hiddens: Tensor<4>,
        p_dist: Tensor<2>,
        targets: Tensor<2, Int>,
    ) -> Tensor<1>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let [n, b, t, d] = step_hiddens.dims();
        let v = self.vocab_size;
        let tgt = targets.reshape([b * t]).one_hot::<2>(v).cast(FloatDType::F32);
        // L_Rec: expectation of the reconstruction loss over halting steps.
        let mut rec = Tensor::<1>::zeros([1], &step_hiddens.device());
        for i in 0..n {
            let h_n = step_hiddens
                .clone()
                .slice([i..i + 1, 0..b, 0..t, 0..d])
                .reshape([b * t, d]);
            let logits_n = self.lm_head.forward::<B>(h_n).reshape([b * t, v]);
            let ce = burn::tensor::loss::cross_entropy_with_logits(logits_n, tgt.clone())
                .reshape([b, t])
                .mean_dim(1); // [b]
            let pn = p_dist.clone().slice([0..b, i..i + 1]).reshape([b, 1]); // [b,1]
            rec = rec + (pn * ce.reshape([b, 1])).sum_dim(0).reshape([1]);
        }
        let rec = rec.div_scalar(b as f32); // mean over batch
        let kl = self.ponder_kl(p_dist, self.ponder_prior); // [1]
        rec + kl.mul_scalar(self.ponder_beta)
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