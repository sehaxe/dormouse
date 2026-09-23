//! # burn-mtp - Multi-Token Prediction for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! | Reference | What |
//! |-----------|------|
//! | Gloeckle et al. 2024 | k heads predicting the next k tokens (equal weights) |
//! | DeepSeek-V3 2024 | MTP with uniform per-depth weight λ |
//!
//! > Paper: [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737).
//!
//! Per the paper: head `i` predicts token `t+i+1` from the shared backbone
//! hidden state `h[..t]`; all heads share one unembedding matrix `f_u`
//! (paper method.tex: `softmax(f_u(f_h_i(z_{t:1})))`). Loss is the cross
//! entropy over the vocab. Gloeckle et al. weight all depths EQUALLY;
//! DeepSeek-V3 applies a uniform λ per depth. The crate's default is the
//! cited paper's equal weighting; `with_lambda` switches to the V3 scheme.
//!
//! Heads are the paper's documented "linear probing" variant
//! (appendix.tex:23-24): LayerNorm → Linear(D,D) → shared f_u → vocab.
use burn::module::Module;
use burn::nn::Initializer;
use burn::tensor::{activation, Device, Int, Tensor};

/// Multi-token prediction heads with a shared unembedding matrix.
///
/// `forward` returns per-head vocab logits `[B, T, vocab]`; `loss` sums the
/// per-head cross entropy over the `k` future-token predictions.
#[derive(Module, Debug)]
pub struct MtpHeads {
    pub head_norms: Vec<burn::nn::LayerNorm>,
    pub head_linears: Vec<burn::nn::Linear>,
    /// Shared unembedding `f_u: D -> vocab` (paper method.tex:23).
    pub unembed: burn::nn::Linear,
    #[module(skip)]
    pub n_heads: usize,
    /// Uniform per-depth weight (DeepSeek-V3 style); 1.0 = equal weights
    /// (Gloeckle et al.).
    #[module(skip)]
    pub lambda: f64,
}

impl MtpHeads {
    pub fn new(d_model: usize, vocab_size: usize, k: usize, device: &Device) -> Self {
        let init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        let heads = (0..k)
            .map(|_| {
                burn::nn::LinearConfig::new(d_model, d_model)
                    .with_bias(false)
                    .with_initializer(init.clone())
                    .init(device)
            })
            .collect();
        let norms = (0..k)
            .map(|_| burn::nn::LayerNormConfig::new(d_model).init(device))
            .collect();
        Self {
            head_norms: norms,
            head_linears: heads,
            unembed: burn::nn::LinearConfig::new(d_model, vocab_size)
                .with_bias(false)
                .with_initializer(init)
                .init(device),
            n_heads: k,
            lambda: 1.0,
        }
    }

    /// DeepSeek-V3-style uniform per-depth weight (default 0.3).
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Per-head vocab logits: `Vec<[B, T, vocab]>`, head `i` predicts `t+i+1`.
    pub fn forward(&self, h: Tensor<3>) -> Vec<Tensor<3>> {
        self.head_linears
            .iter()
            .zip(self.head_norms.iter())
            .map(|(lin, norm)| {
                let head_out = lin.forward(norm.forward(h.clone()));
                self.unembed.forward(head_out)
            })
            .collect()
    }

    /// Multi-token prediction loss: `lambda * sum_i CE_i` over the k heads.
    ///
    /// `h`: shared backbone hidden states `[B, T, D]`.
    /// `targets`: token ids `[B, T, Int]`.
    /// Head `i` predicts `targets[:, i+1 ..]` from `h[:, .. T-i-1]`
    /// (alignment: position `j` predicts token `j+i+1`).
    pub fn loss(&self, h: Tensor<3>, targets: Tensor<2, Int>) -> Tensor<1> {
        let [b, t, _d] = h.dims();
        let dev = h.device();
        let mut total = Tensor::zeros([1], &dev);
        for (i, (lin, norm)) in self
            .head_linears
            .iter()
            .zip(self.head_norms.iter())
            .enumerate()
        {
            let offset = i + 1;
            if t <= offset {
                break;
            }
            let h_in = h.clone().slice([0..b, 0..(t - offset)]);
            let logits = self.unembed.forward(lin.forward(norm.forward(h_in))); // [B, T-offset, V]
            let target = targets.clone().slice([0..b, offset..t]); // [B, T-offset]
            let logp = activation::log_softmax(logits, 2);
            let idx = target.unsqueeze_dim::<3>(2);
            let token_logp: Tensor<2> = logp
                .gather(2, idx.expand([b, t - offset, 1]))
                .reshape([b, t - offset]);
            total = total + token_logp.neg().mean().mul_scalar(self.lambda as f32);
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    fn dev() -> Device {
        Device::ndarray()
    }

    fn to_f32(t: Tensor<1>) -> f32 {
        f32::from_le_bytes(t.into_data().bytes[..4].try_into().unwrap())
    }

    #[test]
    fn module_loss_finite() {
        let m = MtpHeads::new(32, 100, 3, &dev());
        let h = Tensor::<3>::random([1, 16, 32], Distribution::Default, &dev());
        let tgt = Tensor::<2, Int>::random([1, 16], Distribution::Uniform(0.0, 100.0), &dev());
        let loss = m.loss(h, tgt);
        assert!(to_f32(loss).is_finite());
    }

    #[test]
    fn module_loss_short_sequence() {
        // T=3 < k=4: only 2 valid predictions, no panic
        let m = MtpHeads::new(16, 50, 4, &dev());
        let h = Tensor::<3>::random([1, 3, 16], Distribution::Default, &dev());
        let tgt = Tensor::<2, Int>::random([1, 3], Distribution::Uniform(0.0, 50.0), &dev());
        let loss = m.loss(h, tgt);
        assert!(to_f32(loss).is_finite());
    }

    #[test]
    fn logits_are_vocab_dim() {
        let m = MtpHeads::new(16, 64, 2, &dev());
        let h = Tensor::<3>::random([2, 8, 16], Distribution::Default, &dev());
        let logits = m.forward(h);
        assert_eq!(logits.len(), 2);
        assert_eq!(logits[0].dims(), [2, 8, 64]);
    }

    #[test]
    fn ce_equals_manual_gather() {
        // The loss's CE must equal hand-computed -log(softmax) at the targets
        let m = MtpHeads::new(16, 32, 1, &dev());
        let h = Tensor::<3>::random([1, 4, 16], Distribution::Default, &dev());
        let tgt =
            Tensor::<1, Int>::from_ints(vec![5i64, 3, 7, 9].as_slice(), &dev()).reshape([1, 4]);
        let loss = to_f32(m.loss(h.clone(), tgt.clone()));
        let logits = m.forward(h)[0].clone();
        let logp = activation::log_softmax(logits, 2);
        let vals: Vec<f32> = logp
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let t: Vec<i64> = tgt
            .into_data()
            .bytes
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        // head 0: logits position j predicts token j+1 -> t[j+1].
        // vals are LOG probabilities; CE = -logp at the target.
        let manual = -vals[t[1] as usize] - vals[32 + t[2] as usize] - vals[64 + t[3] as usize];
        let manual = manual / 3.0;
        assert!(
            (loss - manual).abs() < 1e-3,
            "loss {loss} != manual CE {manual}"
        );
    }
}
