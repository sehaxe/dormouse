//! # burn-es - Evolution Strategies for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! | arXiv | Function | What |
//! |-------|----------|------|
//! | [1703.03864](https://arxiv.org/abs/1703.03864) | `antithetic_pair`, `openai_rank_utilities`, `es_gradient` | OpenAI ES: mirrored sampling, log-shaped rank utilities, gradient estimator |
//! | [2511.16652](https://arxiv.org/abs/2511.16652) | `eggroll_mutate` | Low-rank perturbation - 100x speedup |
//!
//! Weight mutation, fitness shaping, and the ES gradient estimator
//! `grad = (1/(N*sigma)) * sum_i u_i * eps_i` for black-box optimization
//! of neural network parameters.
//!
//! NOTE: this crate keeps a convenience `eggroll_mutate` whose perturbation
//! scales as `σ·A·Bᵀ` (entry std ≈ σ·√rank); the dedicated
//! [`burn-eggroll`](https://docs.rs/burn-eggroll) crate implements the paper's
//! normalized form `(σ/√r)·A·Bᵀ` (entry std ≈ σ exactly), plus antithetic
//! batched updates and seed-regeneration API. Prefer burn-eggroll for real
//! ES loops; the function here stays for quick experiments and its test.
use burn::tensor::{Device, Distribution, Tensor};

/// Gaussian noise mutation: `w + eps`, `eps ~ N(0, sigma^2)`.
pub fn gaussian_mutate(w: Tensor<2>, sigma: f32, device: &Device) -> Tensor<2> {
    let noise = Tensor::random(w.dims(), Distribution::Normal(0.0, sigma as f64), device);
    w + noise
}

/// Ternary quantization with dead zone: `sign(w) * mean(|w|)` where
/// `|w| > 0.7 * mean(|w|)`, else 0 — the BitNet b1.58 convention
/// (threshold = 0.7 * absmean). Produces `{-scale, 0, +scale}`.
pub fn ternarize(w: Tensor<2>) -> Tensor<2> {
    // one abs pass instead of three: absmean is both the scale and (x0.7)
    // the dead-zone threshold
    let mag = w.clone().abs();
    let mean = mag.clone().mean().unsqueeze_dims(&[0, 0]);
    let keep = mag.greater(mean.clone().mul_scalar(0.7)).int().float();
    w.sign().mul(mean).mul(keep)
}

/// Ternary weight mutation: perturb + re-ternarize (hill-climbing on
/// `{-scale, 0, +scale}` weights).
pub fn ternary_mutate(w: Tensor<2>, sigma: f32, device: &Device) -> Tensor<2> {
    let perturbed = gaussian_mutate(w, sigma, device);
    ternarize(perturbed)
}

/// Low-rank perturbation `Delta W = sigma * A * B^T` (EGGROLL, 2511.16652).
///
/// `A ~ N(0,1) in R^{d_out x rank}`, `B ~ N(0,1) in R^{d_in x rank}`; the
/// perturbation is scaled by `sigma` at use (paper: `x @ B @ A.T * sigma`),
/// so each entry of Delta W has variance `sigma^2 * rank`.
/// rank << min(d_out, d_in) - 100x fewer random numbers than full-rank.
pub fn eggroll_mutate(w: Tensor<2>, rank: usize, sigma: f32, device: &Device) -> Tensor<2> {
    let [d_out, d_in] = w.dims();
    let a = Tensor::random([d_out, rank], Distribution::Normal(0.0, 1.0), device);
    let b = Tensor::random([d_in, rank], Distribution::Normal(0.0, 1.0), device);
    w + a.matmul(b.transpose()).mul_scalar(sigma)
}

/// Antithetic sampling: paired +e/-e mutations (OpenAI ES, 1703.03864).
///
/// Returns `(w + noise, w - noise)`. Correlating errors reduces
/// gradient variance by ~2x without extra population members.
pub fn antithetic_pair(w: Tensor<2>, sigma: f32, device: &Device) -> (Tensor<2>, Tensor<2>) {
    let noise = Tensor::random(w.dims(), Distribution::Normal(0.0, sigma as f64), device);
    (w.clone().add(noise.clone()), w.sub(noise))
}

/// Log-shaped rank utilities, NORMALIZED VARIANT: `u/sum(u) - 1/N`.
///
/// Same shape as [`openai_rank_utilities_paper`] but with a different
/// centering/scaling — sums to zero like the paper form yet carries a
/// 1/sum(u) factor in its magnitude (an effective-LR change). Kept for
/// backward compatibility of A/B baselines; `es_gradient` uses the
/// paper-exact form.
///
/// ```text
/// u_i = max(0, ln(N/2 + 1) - ln(N - i))      i = rank, 0 = worst
/// u <- u / sum(u) - 1/N                       (centered, sums to 0)
/// ```
/// The log shape emphasizes the top of the ranking; centering removes the
/// common-mode term from the gradient estimator.
pub fn openai_rank_utilities(rewards: &[f32]) -> Vec<f32> {
    let n = rewards.len();
    if n <= 1 {
        return rewards.to_vec();
    }
    let mut sorted: Vec<f32> = rewards.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut u: Vec<f32> = (0..n)
        .map(|i| {
            let rank = i as f32 + 1.0; // 1 = worst
            let v = (n as f32 / 2.0 + 1.0).ln() - (n as f32 + 1.0 - rank).ln();
            v.max(0.0)
        })
        .collect();
    let sum: f32 = u.iter().sum();
    if sum > 0.0 {
        for v in u.iter_mut() {
            *v = *v / sum - 1.0 / n as f32;
        }
    }
    // Map back to original order (u[rank] belongs to the rank-th sorted reward)
    let mut pairs: Vec<(usize, f32)> = rewards.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0.0f32; n];
    for (rank, (orig_idx, _)) in pairs.iter().enumerate() {
        out[*orig_idx] = u[rank];
    }
    out
}

/// Rank-based fitness shaping exactly as in arXiv 1703.03864 §2:
/// `u_i = max(0, ln(N/2 + 1) − ln(k_i))` with `k_i` the rank (1 = best),
/// then centered by subtracting the mean (`Σu = 0`). This is the utility
/// vector the paper's estimator consumes.
pub fn openai_rank_utilities_paper(rewards: &[f32]) -> Vec<f32> {
    let n = rewards.len();
    if n <= 1 {
        return vec![0.0; n];
    }
    let mut indexed: Vec<(usize, f32)> = rewards.iter().copied().enumerate().collect();
    // sort ascending: index 0 = worst
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for (rank, (orig, _)) in indexed.iter().enumerate() {
        // k_i in the paper counts from BEST = 1
        let k_best = (n - rank) as f32;
        let v = ((n as f32) / 2.0 + 1.0).ln() - k_best.ln();
        let v = v.max(0.0);
        out[*orig] = v;
        sum += v;
    }
    let mean = sum / n as f32;
    for v in out.iter_mut() {
        *v -= mean;
    }
    out
}

/// Rank-based fitness normalization (linear): `(rank_i - 1) / (N - 1)`.
///
/// Robust to reward scale outliers and non-stationarity. Returns values in
/// `[0, 1]`; ties get their average rank.
pub fn rank_fitness(rewards: &[f32]) -> Vec<f32> {
    let n = rewards.len();
    if n <= 1 {
        return rewards.to_vec();
    }
    let mut indexed: Vec<(usize, f32)> = rewards.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0f32; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j + 1 < n && (indexed[j + 1].1 - indexed[i].1).abs() < 1e-8 {
            j += 1;
        }
        let avg_rank = (i + j) as f32 / 2.0;
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank / (n - 1).max(1) as f32;
        }
        i = j + 1;
    }
    ranks
}

/// ES gradient estimator (OpenAI ES, 1703.03864).
///
/// ```text
/// grad = (1 / (N * sigma)) * sum_i u_i * eps_i
/// ```
/// where `eps_i = population[i] - base` are the perturbations and `u_i` the
/// centered rank utilities of `rewards`. Add `lr * grad` to `base` to take
/// an ES step.
pub fn es_gradient(
    base: Tensor<2>,
    population: &[Tensor<2>],
    rewards: &[f32],
    sigma: f32,
) -> Tensor<2> {
    let n = population.len();
    assert!(
        n > 0 && n == rewards.len(),
        "population and rewards must match"
    );
    // paper-exact fitness shaping (1703.03864 §2)
    let utilities = openai_rank_utilities_paper(rewards);
    let mut grad = Tensor::<2>::zeros(base.dims(), &base.device());
    for (eps, u) in population.iter().zip(utilities.iter()) {
        let perturb = eps.clone().sub(base.clone());
        grad = grad.add(perturb.mul_scalar(*u));
    }
    grad.mul_scalar(1.0 / (n as f32 * sigma))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn ternary_shape() {
        let w = Tensor::<2>::random([64, 32], Distribution::Default, &dev());
        assert_eq!(ternary_mutate(w, 0.1, &dev()).dims(), [64, 32]);
    }
    #[test]
    fn ternarize_has_zero_level() {
        // With a dead zone, small-magnitude weights map to 0: {-s, 0, +s}.
        let w = Tensor::<1>::from_floats(vec![-2.0f32, 0.01, 0.0, 3.0].as_slice(), &dev())
            .reshape([2, 2]);
        let t = ternarize(w);
        let vals: Vec<f32> = t
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(vals.contains(&0.0), "must contain zeros: {vals:?}");
        assert!(
            vals.iter().all(|&v| v == 0.0 || v.abs() > 1.0),
            "nonzero entries must be +/-scale: {vals:?}"
        );
    }
    #[test]
    fn eggroll_shape() {
        let w = Tensor::<2>::random([32, 16], Distribution::Default, &dev());
        assert_eq!(eggroll_mutate(w, 4, 0.1, &dev()).dims(), [32, 16]);
    }
    #[test]
    fn eggroll_sigma_scales_perturbation() {
        // Delta W = sigma * A B^T: entries have std ~ sigma*sqrt(rank).
        let w = Tensor::<2>::zeros([64, 64], &dev());
        let mut diffs = Vec::new();
        for _ in 0..8 {
            let p = eggroll_mutate(w.clone(), 4, 1.0, &dev());
            let d: Vec<f32> = p
                .into_data()
                .bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            diffs.extend(d);
        }
        let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
        let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / diffs.len() as f32;
        let std = var.sqrt();
        assert!(
            (std - 2.0).abs() < 0.5,
            "sigma=1, rank=4 -> std ~ sqrt(4)=2, got {std}"
        );
    }
    #[test]
    fn antithetic_shape() {
        let w = Tensor::<2>::ones([8, 4], &dev());
        let (p, n) = antithetic_pair(w, 0.1, &dev());
        assert_eq!(p.dims(), [8, 4]);
        assert_eq!(n.dims(), [8, 4]);
        // (w+e) + (w-e) = 2w
        let sum: Vec<f32> = (p + n)
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(sum.iter().all(|&v| (v - 2.0).abs() < 1e-4));
    }
    #[test]
    fn rank_fitness_range() {
        let r = rank_fitness(&[5.0, 1.0, 3.0, 10.0]);
        assert!(r.iter().all(|x| (0.0..=1.0).contains(x)));
        assert!((r[3] - 1.0).abs() < 0.01); // best = 1.0
        assert!((r[1] - 0.0).abs() < 0.01); // worst = 0.0
    }
    #[test]
    fn openai_paper_utilities_center_and_rank() {
        // 1703.03864 §2 exact form: log-shaped, centered by subtracting the
        // mean; best reward gets the largest utility.
        let u = openai_rank_utilities_paper(&[1.0, 2.0, 100.0]);
        assert!((u.iter().sum::<f32>()).abs() < 1e-5, "must sum to 0");
        assert!(u[2] > u[1] && u[1] > u[0], "monotone in reward");
        // worst rank clips at 0 before centering -> strictly negative utility
        assert!(u[0] < 0.0);
        // scale differs from the normalized variant (effective-LR check)
        let un = openai_rank_utilities(&[1.0, 2.0, 100.0]);
        assert!(
            u.iter().zip(un.iter()).any(|(a, b)| (a - b).abs() > 1e-3),
            "paper and normalized variants should not coincide"
        );
    }

    #[test]
    fn openai_utilities_center_and_rank() {
        let u = openai_rank_utilities(&[1.0, 2.0, 100.0]);
        assert!(
            (u.iter().sum::<f32>()).abs() < 1e-5,
            "utilities must sum to 0"
        );
        assert!(
            u[2] > u[1] && u[1] > u[0],
            "best reward gets highest utility"
        );
    }
    #[test]
    fn es_gradient_points_up_hill() {
        // rewards correlate with sum(w): grad should point along the
        // direction that increases the sum.
        let base = Tensor::<2>::zeros([4, 4], &dev());
        let pop: Vec<Tensor<2>> = (0..6)
            .map(|i| {
                let v = (i + 1) as f32;
                Tensor::<1>::from_floats(vec![v; 16].as_slice(), &dev()).reshape([4, 4])
            })
            .collect();
        let rewards: Vec<f32> = pop
            .iter()
            .map(|t| {
                let vals: Vec<f32> = t
                    .clone()
                    .into_data()
                    .bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                vals.iter().sum()
            })
            .collect();
        let grad = es_gradient(base.clone(), &pop, &rewards, 1.0);
        let gv: Vec<f32> = grad
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mean_g = gv.iter().sum::<f32>() / 16.0;
        assert!(
            mean_g > 0.0,
            "gradient should point toward higher rewards, got {mean_g}"
        );
    }
}
