//! model - Embedding -> LoopBlock -> RMSNorm -> lm_head, all-bf16 capable
use burn::backend::DispatchKindConversion;
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::{Device, DispatchTensor, FloatDType, Int, Tensor, TensorData};
use burn_rmsnorm::RMSNorm;

use crate::aux::AuxHeads;
use crate::config::DormouseConfig;
use crate::loop_block::LoopBlock;
use crate::param::LinearLike;

#[derive(Module, Debug)]
pub struct DormouseModel {
    pub embedding: Embedding,
    pub loop_block: LoopBlock,
    pub norm: RMSNorm,
    pub lm_head: LinearLike,
    pub aux: AuxHeads,
    #[module(skip)]
    pub vocab_size: usize,
    #[module(skip)]
    pub d_model: usize,
    #[module(skip)]
    pub bf16: bool,
    #[module(skip)]
    pub jepa_weight: f32,
    #[module(skip)]
    pub jepa_mask_frac: f32,
    #[module(skip)]
    pub jepa_mask_span: usize,
    #[module(skip)]
    pub dspark_weight: f32,
    #[module(skip)]
    pub dspark_k: usize,
    #[module(skip)]
    pub dspark_stride: usize,
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
            aux: AuxHeads::new(d, v, cfg.rank, device),
            vocab_size: v,
            d_model: d,
            bf16: cfg.bf16,
            jepa_weight: cfg.jepa_weight,
            jepa_mask_frac: cfg.jepa_mask_frac,
            jepa_mask_span: cfg.jepa_mask_span,
            dspark_weight: cfg.dspark_weight,
            dspark_k: cfg.dspark_k,
            dspark_stride: cfg.dspark_stride,
            ponder_beta: cfg.ponder_beta,
            ponder_prior: cfg.ponder_prior,
        }
    }

    pub fn forward<B: burn::backend::AutodiffBackend>(
        &self,
        input_ids: Tensor<2, Int>,
        hashed_ids: Option<Tensor<3, Int>>,
    ) -> Tensor<3>
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        self.forward_with_hidden::<B>(input_ids, hashed_ids, None, None, None).0
    }

    /// Teacher pass: the pre-head accumulated latent `out_acc` only (no
    /// lm_head, no losses). Used for the EMA JEPA teacher.
    pub fn forward_latent<B: burn::backend::AutodiffBackend>(
        &self,
        input_ids: Tensor<2, Int>,
        hashed_ids: Option<Tensor<3, Int>>,
        host_rows: Option<Tensor<3>>,
    ) -> Tensor<3>
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        let x = self.embedding.forward(input_ids);
        let x = if self.bf16 {
            x.cast(FloatDType::BF16)
        } else {
            x
        };
        let (out_acc, _rec, _p, _kda) = self.loop_block.forward_full_state::<B>(
            x, hashed_ids, host_rows, None, None, &self.lm_head,
        );
        out_acc
    }

    /// Returns (logits, L_Rec [1], p_dist [b,N], kda, aux Option<[1]>). When
    /// `targets` is Some, the per-step reconstruction loss is accumulated
    /// inside the loop block so the model never slices a 4D autodiff tensor
    /// (cubecl/sm_120 stability). `host_rows` carries pre-gathered n-gram
    /// rows `[b,t,3*32]` for the RAM-offload path; when None, `hashed_ids`
    /// drives the in-model tables. `teacher` (the EMA copy) enables the JEPA
    /// term; `aux` is the weight-combined auxiliary loss (JEPA + DSpark),
    /// None when every aux weight is 0 or `targets` is None.
    pub fn forward_with_hidden<B: burn::backend::AutodiffBackend>(
        &self,
        input_ids: Tensor<2, Int>,
        hashed_ids: Option<Tensor<3, Int>>,
        host_rows: Option<Tensor<3>>,
        targets: Option<Tensor<2, Int>>,
        teacher: Option<&Self>,
    ) -> (Tensor<3>, Tensor<1>, Tensor<2>, Tensor<4>, Option<Tensor<1>>)
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        let ids_raw = targets.clone();
        let x = self.embedding.forward(input_ids);
        let x = if self.bf16 {
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
        let h = if self.bf16 {
            self.norm.forward(out_acc.clone().cast(FloatDType::F32))
        } else {
            self.norm.forward(out_acc.clone())
        };
        let logits = self
            .lm_head
            .forward::<B>(h.clone().reshape([b * t, self.d_model]))
            .reshape([b, t, self.vocab_size]);
        let aux = self.aux_loss::<B>(&out_acc, teacher, ids_raw, &h, &logits);
        (logits, rec, p_dist, kda, aux)
    }

    /// Weight-combined auxiliary loss (JEPA + DSpark). None when every aux
    /// weight is 0, when there are no targets, or when JEPA is on but no
    /// teacher was supplied.
    fn aux_loss<B: burn::backend::AutodiffBackend>(
        &self,
        student_latent: &Tensor<3>,
        teacher: Option<&Self>,
        ids: Option<Tensor<2, Int>>,
        h: &Tensor<3>,
        logits: &Tensor<3>,
    ) -> Option<Tensor<1>>
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        if (self.jepa_weight <= 0.0 && self.dspark_weight <= 0.0) || self.dspark_k == 0 {
            return None;
        }
        let ids = ids?;
        let dev = h.device();
        let mut total: Option<Tensor<1>> = None;
        if self.jepa_weight > 0.0 {
            if let Some(t) = teacher {
                // Teacher latent over the same inputs; the student predicts
                // its own out_acc (pre-head accumulation) against it.
                let tl = t.forward_latent::<B>(ids.clone(), None, None);
                let j = crate::aux::jepa_aux_loss(
                    &self.aux.jepa_pred,
                    student_latent.clone(),
                    tl,
                    self.jepa_mask_frac,
                    self.jepa_mask_span,
                );
                total = Some(j.mul_scalar(self.jepa_weight)
                    + total.unwrap_or_else(|| Tensor::zeros([1], &dev)));
            }
        }
        if self.dspark_weight > 0.0 {
            let d = crate::aux::dspark_aux_loss(
                &self.aux.dspark,
                &self.aux.conf,
                h.clone(),
                logits.clone(),
                ids,
                self.dspark_k,
                self.dspark_stride,
            );
            total = Some(d.mul_scalar(self.dspark_weight)
                + total.unwrap_or_else(|| Tensor::zeros([1], &dev)));
        }
        total
    }

    /// PonderNet loss (Banino et al. 2021): L = L_Rec + β·KL(p || Geom(λ_p)).
    pub fn loss<B: burn::backend::AutodiffBackend>(&self, rec_ce: Tensor<1>, p_dist: Tensor<2>) -> Tensor<1>
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
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

    /// Toggle the bf16 matmul path (fp32 graph, tensor-core forward) on
    /// every TSCT factor. Call under the model's BF16 mode.
    pub fn set_bf16_compute(&mut self, on: bool) {
        self.loop_block.set_bf16_compute(on);
        self.lm_head.set_bf16_compute(on);
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
    pub fn forward_bytes<B: burn::backend::AutodiffBackend>(&self, bytes: &[u8]) -> Vec<f32>
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
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