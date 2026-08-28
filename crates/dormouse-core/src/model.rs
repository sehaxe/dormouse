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

    /// Returns (logits, ponder_term).
    pub fn forward_with_hidden<B: Backend>(
        &self,
        input_ids: Tensor<2, Int>,
        hashed_ids: Option<Tensor<3, Int>>,
    ) -> (Tensor<3>, Tensor<1>)
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let x = self.embedding.forward(input_ids);
        let x = if crate::param::bf16_on() {
            x.cast(FloatDType::BF16)
        } else {
            x
        };
        let (h, ponder, _mod, _kda) = self.loop_block.forward_full_state::<B>(x, hashed_ids, None);
        // loop activations may be bf16; the final norm+head compute in fp32
        // (bf16 logits make the softmax/CE numerically unstable -> NaN).
        let h = if crate::param::bf16_on() {
            self.norm.forward(h.cast(FloatDType::F32))
        } else {
            self.norm.forward(h)
        };
        let b = h.dims()[0];
        let t = h.dims()[1];
        let logits = self.lm_head.forward::<B>(h.reshape([b * t, self.d_model])).reshape([b, t, self.vocab_size]);
        (logits, ponder)
    }

    pub fn loss<B: Backend>(&self, logits: Tensor<3>, targets: Tensor<2, Int>, ponder: Tensor<1>) -> Tensor<1>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let [b, t, v] = logits.dims();
        let flat = logits.reshape([b * t, v]);
        let target_probs = targets.reshape([b * t]).one_hot::<2>(v).cast(FloatDType::F32);
        let ce = burn::tensor::loss::cross_entropy_with_logits(flat, target_probs);
        ce.mean() + ponder.mul_scalar(0.05)
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