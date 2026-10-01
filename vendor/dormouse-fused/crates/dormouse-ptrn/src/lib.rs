//! # dormouse-ptrn — Probabilistic Tiny Recursive Model (PTRM)
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! Test-time scaling for recursive (looped, parameter-shared) models from
//! [PTRM (arXiv 2605.19943)](https://arxiv.org/abs/2605.19943).
//!
//! The recipe needs NO retraining of the loop weights:
//!
//! 1. **Recurrent Gaussian noise** — inject `ε ~ N(0, σ²I)` into the latent at
//!    every recursion step. Noise-only (no gradient term) beat Q-guided
//!    Langevin in the paper's ablations; it lets ~8% of rollouts escape the
//!    "bad basins" failed trajectories get stuck in.
//! 2. **Best-Q@K selection** — run K parallel rollouts (width scaling: stronger
//!    and more practical than depth scaling, fully parallelizable), score each
//!    with a learned Q-head, and take the highest-scoring trajectory.
//! 3. **Joint Q-head training** — `L_step = CE(f_O(y), y_true) + BCE(q̂, 1[ŷ = y_true])`
//!    trains the value head alongside the next-token head, so Q separates
//!    correct/incorrect trajectories at convergence.
//!
//! All primitives are pure tensor ops — no host branching, no `.into_data()`
//! in the forward/rollout path.
//!
//! ## Usage
//!
//! ```ignore
//! use burn_ndarray::NdArray;
//! use dormouse_ptrn::{PtrnConfig, QHead, add_recurrent_noise, best_of_k, correctness_target};
//!
//! let config = PtrnConfig::new(); // σ=0.5, K=16, τ=1.0
//! let device = NdArrayDevice::default();
//! let q = QHead::<NdArray>::new(64, &device);
//!
//! // inside the loop, per recursion step:
//! z = add_recurrent_noise(z, config.noise_sigma, &device);
//! // ... z = loop_step(z) ...
//!
//! // after K rollouts, select best:
//! let (best, idx) = best_of_k(q_logits, rollouts); // rollouts: [K,B,T,D]
//!
//! // training: correctness flag feeds the Q loss
//! let target = correctness_target(pred_ids, truth_ids); // [B,T] Int 0/1
//! let loss = q.q_loss(q.logit(h), target, Some(mask));
//! ```

mod qhead;
mod scaling;

pub use qhead::QHead;
pub use scaling::{
    add_recurrent_noise, best_of_k, best_of_k_sampled, correctness_target, PtrnConfig,
};

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Distribution, Int, Tensor, TensorData};
    fn dev() -> Device {
        Device::ndarray()
    }

    fn as_scalar(t: Tensor<1>) -> f32 {
        f32::from_le_bytes(t.into_data().bytes[..4].try_into().unwrap())
    }

    #[test]
    fn noise_adds_variance() {
        let z = Tensor::<3>::zeros([4, 32, 64], &dev());
        let noised = add_recurrent_noise(z.clone(), 1.0, &dev());
        let diff = (noised - z).powf_scalar(2.0).mean();
        let rmse = as_scalar(diff).sqrt();
        assert!((rmse - 1.0).abs() < 0.15, "rmse = {rmse}");
    }

    #[test]
    fn q_head_shape_and_range() {
        let q = QHead::new(32, &dev());
        let h = Tensor::<3>::zeros([2, 8, 32], &dev());
        assert_eq!(q.logit(h.clone()).dims(), [2, 8, 1]);
        let v: Vec<f32> = q
            .prob(h)
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(v.iter().all(|x| (0.0..=1.0).contains(x)), "probs {v:?}");
    }

    #[test]
    fn q_loss_zero_when_perfect() {
        let q = QHead::new(4, &dev());
        let logits = Tensor::<3>::ones([2, 4, 1], &dev()).mul_scalar(10.0);
        let target: Tensor<2, Int> = Tensor::ones([2, 4], &dev());
        let loss = as_scalar(q.q_loss(logits, target, None));
        assert!(loss.abs() < 1e-3, "loss = {loss}");
    }

    #[test]
    fn q_loss_penalizes_wrong() {
        let q = QHead::new(4, &dev());
        let logits = Tensor::<3>::ones([2, 4, 1], &dev()).mul_scalar(-10.0);
        let target: Tensor<2, Int> = Tensor::ones([2, 4], &dev());
        let loss = as_scalar(q.q_loss(logits, target, None));
        assert!(loss > 0.5, "loss = {loss}");
    }

    #[test]
    fn q_loss_respects_mask() {
        let q = QHead::new(4, &dev());
        let logits = Tensor::<3>::ones([2, 4, 1], &dev()).mul_scalar(-10.0);
        let target: Tensor<2, Int> = Tensor::ones([2, 4], &dev());
        let mask = Tensor::<2>::zeros([2, 4], &dev());
        let loss = as_scalar(q.q_loss(logits, target, Some(mask)));
        assert!(loss == 0.0, "loss = {loss}");
    }

    #[test]
    #[allow(clippy::approx_constant)] // 0.6931... is -ln(0.5), also ≈ ln 2
    fn q_loss_partial_mask_excludes_unmasked() {
        let q = QHead::new(4, &dev());
        // B=2, T=3: logit 0 → bce = -ln(0.5) ≈ 0.6931; logit 10, target 0 → bce ≈ 10.0
        let logits = Tensor::<3>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 10.0, 0.0, 0.0, 10.0], [2, 3, 1]),
            &dev(),
        );
        let target: Tensor<2, Int> = Tensor::zeros([2, 3], &dev());
        // batch 0 masks the two low-bce positions, batch 1 only the high-bce one:
        // masked sum = 0.6931 + 0.6931 + 10.0 = 11.3863 over 3 masked positions
        let mask = Tensor::<2>::from_data(
            TensorData::new(vec![1.0f32, 1.0, 0.0, 0.0, 0.0, 1.0], [2, 3]),
            &dev(),
        );
        let loss = as_scalar(q.q_loss(logits, target, Some(mask)));
        let expected = (2.0 * 0.693_147_2 + 10.0) / 3.0; // ≈ 3.7954
        assert!(
            (loss - expected).abs() < 1e-2,
            "loss = {loss}, expected {expected}"
        );
    }

    #[test]
    fn correctness_target_marks_equal() {
        let pred: Tensor<2, Int> =
            Tensor::from_data(TensorData::new(vec![1i32, 2, 3, 4], [2, 2]), &dev());
        let truth: Tensor<2, Int> =
            Tensor::from_data(TensorData::new(vec![1i32, 9, 3, 9], [2, 2]), &dev());
        let flags = correctness_target(pred, truth).into_data();
        assert_eq!(
            flags.bytes,
            burn::tensor::Bytes::from_elems(vec![1i64, 0, 1, 0])
        );
    }

    #[test]
    fn best_of_k_picks_max() {
        let k: usize = 4;
        let b: usize = 2;
        let t: usize = 3;
        let d: usize = 5;
        // q_logits[k][b][t][0] = qk[k] → best index is 2 for every batch
        let qk = [0.0f32, 1.0, 3.0, 2.0];
        let mut qs: Vec<f32> = Vec::new();
        for &q in qk.iter() {
            for _ in 0..b * t {
                qs.push(q);
            }
        }
        let q_logits = Tensor::<4>::from_data(TensorData::new(qs, [k, b, t, 1]), &dev());
        // rollouts[k][b][t][d] = k → best rollout should be all 2.0
        let mut rs: Vec<f32> = Vec::new();
        for kk in 0..k {
            for _ in 0..b * t * d {
                rs.push(kk as f32);
            }
        }
        let rollouts = Tensor::<4>::from_data(TensorData::new(rs, [k, b, t, d]), &dev());
        let (best, idx) = best_of_k(q_logits, rollouts);
        assert_eq!(best.dims(), [b, t, d]);
        assert_eq!(
            idx.into_data().bytes,
            burn::tensor::Bytes::from_elems(vec![2i64, 2])
        );
        let best_data = best.into_data();
        assert!(best_data
            .bytes
            .chunks_exact(4)
            .all(|b| f32::from_le_bytes(b.try_into().unwrap()) == 2.0));
    }

    #[test]
    fn noise_is_finite() {
        let z = Tensor::<3>::zeros([2, 16, 32], &dev());
        let noised = add_recurrent_noise(z, 0.5, &dev());
        let s = as_scalar(noised.mean());
        assert!(s.is_finite());
    }

    #[test]
    fn best_of_k_sampled_tau_zero_is_argmax() {
        // τ < 1e-5 must short-circuit to the deterministic argmax path.
        let k: usize = 4;
        let (b, t, d) = (2usize, 3usize, 5usize);
        let qk = [0.0f32, 1.0, 3.0, 2.0];
        let mut qs = Vec::new();
        for &q in qk.iter() {
            for _ in 0..b * t {
                qs.push(q);
            }
        }
        let q_logits = Tensor::<4>::from_data(TensorData::new(qs, [k, b, t, 1]), &dev());
        let mut rs = Vec::new();
        for kk in 0..k {
            for _ in 0..b * t * d {
                rs.push(kk as f32);
            }
        }
        let rollouts = Tensor::<4>::from_data(TensorData::new(rs, [k, b, t, d]), &dev());
        let (best, idx) = best_of_k_sampled(q_logits.clone(), rollouts.clone(), 0.0);
        let (best_ref, idx_ref) = best_of_k(q_logits, rollouts);
        assert_eq!(idx.into_data().bytes, idx_ref.into_data().bytes);
        let diff: f32 = (best - best_ref).abs().max().into_scalar();
        assert!(diff == 0.0);
    }

    #[test]
    fn best_of_k_sampled_stays_in_range() {
        let k: usize = 4;
        let (b, t, d) = (3usize, 2usize, 4usize);
        let q_logits = Tensor::<4>::random([k, b, t, 1], Distribution::Default, &dev());
        let rollouts = Tensor::<4>::random([k, b, t, d], Distribution::Default, &dev());
        let (_, idx) = best_of_k_sampled(q_logits, rollouts, 1.5);
        let v: Vec<i64> = idx.into_data().try_to_vec().unwrap();
        assert!(v.iter().all(|&x| (0..k as i64).contains(&x)), "{v:?}");
    }

    #[test]
    fn config_defaults() {
        let c = PtrnConfig::new();
        assert_eq!(c.noise_sigma, 0.5);
        assert_eq!(c.num_rollouts, 16);
        assert_eq!(c.tau, 1.0);
    }
}
