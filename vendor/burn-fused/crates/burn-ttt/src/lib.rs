//! # burn-ttt - Test-Time Training for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! | arXiv | Function | What |
//! |-------|----------|------|
//! | [2512.23675](https://arxiv.org/abs/2512.23675) | `ttt_ce_loss`, `token_log_probs` | End-to-End TTT: masked next-token CE at test time |
//! | [2407.04620](https://arxiv.org/abs/2407.04620) | `ttt_latent_loss` | Original TTT: MSE on the next latent state |
//!
//! TTT-E2E (Tandon et al., 2025): continue learning at test time via
//! next-token prediction on the given context, compressing the context
//! into model weights. Constant latency regardless of context length.
//! Matched to the official implementation
//! [test-time-training/e2e](https://github.com/test-time-training/e2e)
//! (`ttt/model/loss.py`: masked `log_softmax` gather, loss averaged over
//! valid text length).
use burn::tensor::{Int, Tensor};

/// Token-level negative log-likelihood `[B, L]` from logits `[B, L, V]`
/// (TTT-E2E `token_log_probs`, masked positions zeroed).
pub fn token_log_probs(logits: Tensor<3>, targets: Tensor<2, Int>) -> Tensor<2> {
    let lp = burn::tensor::activation::log_softmax(logits, 2);
    lp.gather(2, targets.unsqueeze_dim::<3>(2))
        .squeeze_dim::<2>(2)
}

/// Masked next-token cross-entropy loss for test-time training
/// (TTT-E2E `cross_entropy_loss_and_accuracy`).
///
/// `logits`: `[B, L, V]` - model predictions for the context.
/// `targets`: `[B, L]` - ground truth (shifted by 1).
/// `mask`: `[B, L]` - 1.0 for valid positions, 0.0 for padding.
///
/// Returns the mean over valid text length (each sequence's loss is divided
/// by its valid length, then averaged over the batch).
pub fn ttt_ce_loss(logits: Tensor<3>, targets: Tensor<2, Int>, mask: Tensor<2>) -> Tensor<1> {
    let nll = -token_log_probs(logits, targets); // [B, L]
    let zeros = Tensor::zeros_like(&nll);
    let nll = nll.mask_where(mask.clone().equal_elem(0.0), zeros);
    let valid = mask.sum_dim(1).clamp_min(1e-10); // [B, 1]
    nll.sum_dim(1).div(valid).mean()
}

/// Masked MSE loss for test-time training on the next latent state
/// (original TTT paper, "learning to predict the next latent").
///
/// `pred`: `[B, L, D]` - model predictions for the next latent
/// `target`: `[B, L, D]` - ground truth latents (shifted by 1)
/// `mask`: `[B, L]` - 1.0 for valid positions, 0.0 for padding
///
/// Returns scalar loss suitable for a single SGD step at test time.
pub fn ttt_latent_loss(pred: Tensor<3>, target: Tensor<3>, mask: Tensor<2>) -> Tensor<1> {
    let se = (pred - target)
        .powf_scalar(2.0)
        .mean_dim(2)
        .squeeze_dim::<2>(2);
    (se * mask.clone()).sum().div(mask.sum().clamp_min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Distribution};

    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn ce_loss_zero_when_perfect() {
        // one-hot logits: correct token has logit 10, others 0
        let logits = Tensor::<3>::from_data(
            burn::tensor::TensorData::new(
                vec![
                    10.0f32, 0.0, 0.0, 0.0, //
                    0.0, 10.0, 0.0, 0.0, //
                    0.0, 0.0, 10.0, 0.0,
                ],
                [1, 3, 4],
            ),
            &dev(),
        );
        let targets: Tensor<2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(vec![0i64, 1, 2], [1, 3]),
            &dev(),
        );
        let mask = Tensor::<2>::ones([1, 3], &dev());
        let l = ttt_ce_loss(logits, targets, mask);
        let v: f32 = l.into_scalar();
        // softmax(10, 0, 0, 0) gives ~1 - 4.5e-5, so CE ~ 4.5e-5
        assert!(v < 1e-3, "perfect logits should have ~0 CE, got {v}");
    }

    #[test]
    fn ce_loss_masked_positions_ignored() {
        let logits = Tensor::<3>::zeros([1, 4, 8], &dev());
        let targets: Tensor<2, Int> = Tensor::zeros([1, 4], &dev());
        let mask = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(vec![1.0f32, 1.0, 1.0, 0.0], [1, 4]),
            &dev(),
        );
        let l = ttt_ce_loss(logits, targets, mask);
        let v: f32 = l.into_scalar();
        // uniform CE = ln(8) ~ 2.079
        assert!((v - 8f32.ln()).abs() < 1e-3, "got {v}");
    }

    #[test]
    fn latent_loss_zero_when_equal() {
        let p = Tensor::<3>::ones([2, 8, 32], &dev());
        let m = Tensor::<2>::ones([2, 8], &dev());
        let l = ttt_latent_loss(p.clone(), p, m);
        let v: f32 = l.into_scalar();
        assert!(v.abs() < 1e-4);
    }

    #[test]
    fn latent_loss_masked() {
        let p = Tensor::<3>::random([1, 16, 64], Distribution::Default, &dev());
        let t = Tensor::<3>::random([1, 16, 64], Distribution::Default, &dev());
        let m = Tensor::<2>::ones([1, 16], &dev());
        let v: f32 = ttt_latent_loss(p, t, m).into_scalar();
        assert!(v.is_finite() && v >= 0.0);
    }
}
