//! # dormouse-jepa - JEPA for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! data2vec 2.0-style self-supervised learning on top of a student encoder:
//! an EMA teacher (momentum-updated, stop-grad) produces target latents, the
//! student predicts the latents of MASKED positions with a shared predictor
//! head, trained with masked L1 loss. Plus LeJEPA (isotropic Gaussian) and
//! KoLeo (uniformity) auxiliary losses.
//!
//! | arXiv | Component | What |
//! |-------|-----------|------|
//! | [2212.07525](https://arxiv.org/abs/2212.07525) | `EmaTarget` / `JepaPredictor` / `mask_indices` / `jepa_l1_loss` | data2vec 2.0: EMA teacher + masked latent prediction |
//! | [2511.08544](https://arxiv.org/abs/2511.08544) | `lejepa_loss` | Isotropic Gaussian - ∥mean∥² + ∥cov-I∥²/D |
//! | [2304.07193](https://arxiv.org/abs/2304.07193) | `koleo_loss` | KoLeo uniformity - -mean(log(min_dist)) |

/// data2vec 2.0 hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JepaConfig {
    /// EMA teacher momentum, typically 0.999+ (ramp recommended).
    pub momentum: f64,
    /// Fraction of input positions to mask, e.g. 0.15.
    pub mask_frac: f32,
    /// Contiguous span length of each mask, e.g. 8.
    pub mask_span: usize,
    /// Predictor head dimension (usually d_model).
    pub predictor_dim: usize,
    /// Loss variant.
    pub loss: JepaLoss,
    /// Weight of the JEPA loss in the total objective.
    pub lambda: f64,
}

impl JepaConfig {
    pub fn new(momentum: f64, mask_frac: f32, mask_span: usize, predictor_dim: usize) -> Self {
        Self {
            momentum,
            mask_frac,
            mask_span,
            predictor_dim,
            loss: JepaLoss::L1,
            lambda: 1.0,
        }
    }
}

/// Masked latent prediction loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JepaLoss {
    /// Mean absolute error between predicted and teacher latents.
    L1,
}

mod ema;
mod losses;
mod mask;
mod predictor;

pub use ema::EmaTarget;
pub use losses::{jepa_l1_loss, koleo_loss, lejepa_loss};
pub use mask::mask_indices;
pub use predictor::JepaPredictor;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Device, Distribution, Tensor};
    fn dev() -> Device {
        Device::ndarray()
    }
    fn as_scalar(t: Tensor<1>) -> f32 {
        f32::from_le_bytes(t.into_data().bytes[..4].try_into().unwrap())
    }
    fn as_vec(t: Tensor<1>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
    fn count_true(mask: Tensor<1, Bool>) -> usize {
        as_scalar(mask.float().sum()) as usize
    }

    #[test]
    fn ema_update_tracks_student() {
        let s = Tensor::<1>::from_floats([2.0f32, 4.0], &dev());
        let mut m0 = EmaTarget::new(0.0, &dev());
        m0.update(s.clone());
        let v = as_vec(m0.val());
        assert_eq!(v, vec![2.0, 4.0]);

        let mut m05 = EmaTarget::new(0.5, &dev());
        m05.update(s.clone());
        let v = as_vec(m05.val());
        assert!(
            (v[0] - 1.0).abs() < 0.01 && (v[1] - 2.0).abs() < 0.01,
            "{v:?}"
        );
    }

    #[test]
    fn ema_momentum_decays() {
        let s = Tensor::<1>::from_floats([2.0f32, 2.0], &dev());
        let mut t = EmaTarget::new(0.9, &dev());
        for _ in 0..100 {
            t.update(s.clone());
        }
        let v = as_vec(t.val());
        assert!(
            (v[0] - 2.0).abs() < 1e-3 && (v[1] - 2.0).abs() < 1e-3,
            "{v:?}"
        );
        assert!((v[0] - 2.0).abs() > 1e-6, "should still approach, not snap");
    }

    #[test]
    fn mask_rate_within_tolerance() {
        // span-8 blocks correlate neighbours (effective samples ~ t/8), so
        // average 8 independent draws to shrink the rate variance.
        let mut sum = 0usize;
        for _ in 0..8 {
            let mask = mask_indices(4096, 0.15, 8, &dev());
            sum += count_true(mask);
        }
        let rate = sum as f32 / (4096.0 * 8.0);
        assert!(
            (0.13..=0.17).contains(&rate),
            "mask rate {rate} outside [0.13, 0.17]"
        );
    }

    #[test]
    fn mask_span_dilates() {
        // span-8 masking must produce contiguous runs; with ~83 expected
        // starts in 4096 positions a run of >= 8 is a near-certainty.
        let bytes: Vec<bool> = mask_indices(4096, 0.15, 8, &dev())
            .into_data()
            .try_to_vec()
            .unwrap();
        let mut run = 0usize;
        let mut max_run = 0usize;
        for b in bytes {
            if b {
                run += 1;
                max_run = max_run.max(run);
            } else {
                run = 0;
            }
        }
        assert!(max_run >= 8, "no span-8 run found (max {max_run})");
    }

    #[test]
    fn jepa_l1_masked_mean() {
        let pred = Tensor::<3>::from_floats(
            [
                [[1.0f32], [2.0], [3.0], [4.0]],
                [[1.0], [2.0], [3.0], [4.0]],
            ],
            &dev(),
        );
        let target = Tensor::<3>::from_floats(
            [
                [[1.0f32], [0.0], [0.0], [4.0]],
                [[0.0], [0.0], [3.0], [8.0]],
            ],
            &dev(),
        );
        let mask =
            Tensor::<2>::from_floats([[1.0f32, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0]], &dev())
                .bool();
        // masked diffs: 0, 3, 1, 2, 4 -> mean = 2.0
        let loss = jepa_l1_loss(pred, target, mask);
        let loss = as_scalar(loss);
        assert!((loss - 2.0).abs() < 1e-4, "loss {loss}");
    }

    #[test]
    fn jepa_l1_averages_over_mask_elements_not_positions() {
        // d=3 regression: old formula divided by the number of masked
        // POSITIONS, inflating the loss by a factor of d.
        let pred = Tensor::<3>::ones([2, 4, 3], &dev());
        let target = Tensor::<3>::zeros([2, 4, 3], &dev());
        let mask =
            Tensor::<2>::from_floats([[1.0f32, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0]], &dev())
                .bool();
        // 5 masked positions x 3 dims = 15 masked elements, all diff 1.0
        let loss = as_scalar(jepa_l1_loss(pred, target, mask));
        assert!((loss - 1.0).abs() < 1e-4, "loss {loss}, expected 1.0");
    }

    #[test]
    fn jepa_l1_zero_when_equal() {
        let h = Tensor::<3>::random([4, 16, 8], Distribution::Default, &dev());
        let mask =
            Tensor::<2>::from_floats([[1.0f32; 16], [0.0; 16], [1.0; 16], [1.0; 16]], &dev())
                .bool();
        assert!(as_scalar(jepa_l1_loss(h.clone(), h, mask)).abs() < 1e-5);
    }

    #[test]
    fn jepa_l1_empty_mask_is_zero() {
        // All-false mask: no masked positions -> loss must be 0.0, not NaN.
        let pred = Tensor::<3>::random([2, 8, 4], Distribution::Default, &dev());
        let target = Tensor::<3>::random([2, 8, 4], Distribution::Default, &dev());
        let mask = Tensor::<2>::zeros([2, 8], &dev()).bool();
        let loss = as_scalar(jepa_l1_loss(pred, target, mask));
        assert!(loss == 0.0, "empty mask must give 0.0, got {loss}");
    }

    #[test]
    fn predictor_shape() {
        let pred = JepaPredictor::new(16, &dev());
        let h = Tensor::<3>::random([2, 8, 16], Distribution::Default, &dev());
        assert_eq!(pred.forward(h).dims(), [2, 8, 16]);
    }

    #[test]
    fn koleo_loss_finite() {
        let z = Tensor::<2>::random([128, 64], Distribution::Default, &dev());
        assert!(as_scalar(koleo_loss(z)).is_finite());
    }

    #[test]
    fn lejepa_loss_finite() {
        let z = Tensor::<2>::random([64, 32], Distribution::Default, &dev());
        assert!(as_scalar(lejepa_loss(z)).is_finite());
        let z2 = Tensor::<2>::ones([32, 16], &dev());
        assert!(as_scalar(lejepa_loss(z2)).is_finite());
    }

    #[test]
    fn koleo_subsamples() {
        let z = Tensor::<2>::random([512, 64], Distribution::Default, &dev());
        assert!(as_scalar(koleo_loss(z)).is_finite());
    }

    #[test]
    fn koleo_duplicate_rows_finite() {
        // Half the rows are exact duplicates -> min_dist = 0 -> must clamp,
        // not produce NaN (regression: sqrt of negative 2-2*dot rounding).
        let half = Tensor::<2>::random([256, 64], Distribution::Default, &dev());
        let z = Tensor::cat(vec![half.clone(), half], 0);
        assert!(as_scalar(koleo_loss(z)).is_finite());
    }

    #[test]
    fn koleo_all_same_rows_finite() {
        // All rows identical: unit-sphere rows coincide exactly -> dist 0.
        let z = Tensor::<2>::ones([256, 16], &dev());
        assert!(as_scalar(koleo_loss(z)).is_finite());
    }
}
