//! Test-time scaling primitives: recurrent noise injection, deterministic
//! and temperature-scaled Best-Q@K selection, correctness targets.

use burn::tensor::{activation, Device, Distribution, Int, Tensor};

/// PTRM test-time scaling hyperparameters.
#[derive(Clone, Debug, PartialEq)]
pub struct PtrnConfig {
    /// Standard deviation of the recurrent Gaussian noise per step.
    /// Paper: σ ≈ 0.2–1.0 by task.
    pub noise_sigma: f64,
    /// Number of parallel rollouts (width K). Higher K = more compute at test time.
    pub num_rollouts: usize,
    /// Temperature for Q-based selection (τ = 1.0 → argmax / pure softmax).
    pub tau: f64,
}

impl PtrnConfig {
    pub fn new() -> Self {
        Self {
            noise_sigma: 0.5,
            num_rollouts: 16,
            tau: 1.0,
        }
    }
}

impl Default for PtrnConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Inject recurrent Gaussian noise: `z ← z + σ·N(0, I)`.
///
/// Called once per recursion step before the loop cell. Noise-only per paper
/// (no gradient term). Pure tensor op.
pub fn add_recurrent_noise(z: Tensor<3>, sigma: f64, device: &Device) -> Tensor<3> {
    let eps = Tensor::random(z.shape(), Distribution::Normal(0.0, 1.0), device);
    z + eps.mul_scalar(sigma as f32)
}

/// Best-Q@K selection: pick the rollout with the highest mean Q score per batch.
///
/// `q_logits`: `[K,B,T,1]` — `rollouts`: `[K,B,T,D]`.
/// Returns `(best rollout [B,T,D], best index [B] Int)` via argmax over K.
/// Pure tensor ops.
pub fn best_of_k(q_logits: Tensor<4>, rollouts: Tensor<4>) -> (Tensor<3>, Tensor<1, Int>) {
    let [_, b, t, d] = rollouts.dims();
    let q = q_logits.squeeze_dim::<3>(3); // [K,B,T]
    let q_mean = q.mean_dim(2); // [K,B,1]
    let idx = q_mean.squeeze_dim::<2>(2).argmax(0).squeeze_dim::<1>(0); // [B] Int
    let idx_exp = idx.clone().reshape([1, b, 1, 1]).expand([1, b, t, d]);
    let best = rollouts.gather(0, idx_exp).squeeze_dim::<3>(0); // [B,T,D]
    (best, idx)
}

/// Binary correctness flag: 1 where `pred == truth`, 0 elsewhere.
///
/// `[B,T]` Int in, `[B,T]` Int out. Pure tensor op (host-free).
pub fn correctness_target(pred: Tensor<2, Int>, truth: Tensor<2, Int>) -> Tensor<2, Int> {
    pred.equal(truth).int()
}

/// Temperature-scaled Best-Q@K selection (the stochastic counterpart to
/// [`best_of_k`]): sample one rollout per batch element from
/// `softmax(q_mean / τ)` over the K candidates.
///
/// `τ → 0` degenerates to [`best_of_k`] (argmax); larger τ flattens the
/// distribution toward uniform exploration. `τ < 1e-5` short-circuits to the
/// deterministic path without ever building a softmax.
///
/// `q_logits`: `[K,B,T,1]` — `rollouts`: `[K,B,T,D]`.
/// Returns `(best rollout [B,T,D], sampled index [B] Int)`.
pub fn best_of_k_sampled(
    q_logits: Tensor<4>,
    rollouts: Tensor<4>,
    tau: f32,
) -> (Tensor<3>, Tensor<1, Int>) {
    let [k, b, t, d] = rollouts.dims();
    let [kq, bq, tq, _] = q_logits.dims();
    assert!(
        (kq, bq, tq) == (k, b, t),
        "q_logits [K,B,T,1] must match rollouts [K,B,T,D]"
    );
    assert!(tau.is_finite() && tau >= 0.0, "tau must be finite and >= 0");
    if tau < 1e-5 {
        return best_of_k(q_logits, rollouts);
    }
    let q = q_logits.squeeze_dim::<3>(3); // [K,B,T]
    let q_mean = q.mean_dim(2).squeeze_dim::<2>(2); // [K,B]
                                                    // per-batch distribution over the K candidates
    let probs = activation::softmax(q_mean.swap_dims(0, 1).div_scalar(tau), 1); // [B,K]
    let idx = probs.categorical(1).reshape([b]); // [B] Int
    let idx_exp = idx.clone().reshape([1, b, 1, 1]).expand([1, b, t, d]);
    let best = rollouts.gather(0, idx_exp).squeeze_dim::<3>(0); // [B,T,D]
    (best, idx)
}
