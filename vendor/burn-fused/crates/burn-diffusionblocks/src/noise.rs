//! Noise schedule for block-wise diffusion training.
//!
//! VE (variance-exploding) noise with a log-normal distribution
//! `log σ ~ N(p_mean, p_std²)` (defaults `p_mean = -1.2`, `p_std = 1.2`,
//! `σ_min = 0.002`, `σ_max = 80` — "unless otherwise specified" in the
//! paper, Appendix E preamble). The range `[σ_min, σ_max]` is partitioned
//! into `B` intervals of equal probability mass under the log-normal, so
//! each block sees the same "amount" of noise: `q_b = q_min + (b/B)·
//! (q_max - q_min)` with `q_min/max = Φ((ln σ_min/max - p_mean)/p_std)`
//! and `σ_b = exp(p_mean + p_std·Φ⁻¹(q_b))`. Blocks are indexed from low
//! noise (block 0, nearest `σ_min`) to high noise (block `B-1`, nearest
//! `σ_max`).

use burn::tensor::Tensor;

use crate::normcdf::{inv_normal_cdf, normal_cdf};

/// Default log-normal location `p_mean` (paper default, App. E preamble).
pub const DEFAULT_P_MEAN: f64 = -1.2;
/// Default log-normal scale `p_std` (paper default, App. E preamble).
pub const DEFAULT_P_STD: f64 = 1.2;
/// Default minimum noise level.
pub const DEFAULT_SIGMA_MIN: f64 = 0.002;
/// Default maximum noise level.
pub const DEFAULT_SIGMA_MAX: f64 = 80.0;
/// Default data variance scale `σ_data` of the EDM weighting (App. C.2:
/// "with σ_data = 0.5 for all experiments").
pub const DEFAULT_SIGMA_DATA: f64 = 0.5;

/// Log-normal VE noise schedule `log σ ~ N(p_mean, p_std²)` restricted to
/// `[σ_min, σ_max]`, with equi-probability partitioning into blocks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseSchedule {
    /// Mean of `log σ` (natural log).
    pub p_mean: f64,
    /// Std of `log σ`.
    pub p_std: f64,
    /// Lowest noise level (inclusive).
    pub sigma_min: f64,
    /// Highest noise level (inclusive).
    pub sigma_max: f64,
    /// Data standard deviation used by the EDM weighting (App. C.2).
    pub sigma_data: f64,
}

impl Default for NoiseSchedule {
    fn default() -> Self {
        Self {
            p_mean: DEFAULT_P_MEAN,
            p_std: DEFAULT_P_STD,
            sigma_min: DEFAULT_SIGMA_MIN,
            sigma_max: DEFAULT_SIGMA_MAX,
            sigma_data: DEFAULT_SIGMA_DATA,
        }
    }
}

impl NoiseSchedule {
    /// Normal-CDF quantile `q_b` of the block boundaries,
    /// `q_b = q_min + (b/B)·(q_max - q_min)`, `len = n_blocks + 1`, ascending.
    pub(crate) fn q_edges(&self, n_blocks: usize) -> Vec<f64> {
        assert!(n_blocks > 0, "n_blocks must be > 0");
        let q_min = normal_cdf((self.sigma_min.ln() - self.p_mean) / self.p_std);
        let q_max = normal_cdf((self.sigma_max.ln() - self.p_mean) / self.p_std);
        (0..=n_blocks)
            .map(|b| q_min + (b as f64) * (q_max - q_min) / (n_blocks as f64))
            .collect()
    }

    pub(crate) fn sigma_of_q(&self, q: f64) -> f64 {
        (self.p_mean + self.p_std * inv_normal_cdf(q)).exp()
    }

    /// Block `block`'s noise range `(σ_high, σ_low)` — the `B` blocks tile
    /// `[σ_min, σ_max]` contiguously, each holding `1/B` of the log-normal
    /// probability mass. Block 0 is the lowest-noise interval.
    pub fn partition(&self, block: usize, n_blocks: usize) -> (f64, f64) {
        let edges = self.q_edges(n_blocks);
        assert!(
            block < n_blocks,
            "block {block} out of range [0, {n_blocks})"
        );
        (
            self.sigma_of_q(edges[block + 1]),
            self.sigma_of_q(edges[block]),
        )
    }

    /// Sample a noise level from block `block`'s restricted log-normal.
    ///
    /// Inverse-CDF sampling: one uniform draw `u ~ U[0, 1)`, mapped onto the
    /// block's quantile interval `q = q_lo + u·(q_hi - q_lo)`. Exact given
    /// [`inv_normal_cdf`] (no rejection), so `P(σ ≤ σ') = Φ((ln σ' - p_mean)/
    /// p_std)` restricted to the block's range.
    pub fn sample_sigma(&self, block: usize, n_blocks: usize, rng: &mut fastrand::Rng) -> f32 {
        let edges = self.q_edges(n_blocks);
        assert!(
            block < n_blocks,
            "block {block} out of range [0, {n_blocks})"
        );
        let (q_lo, q_hi) = (edges[block], edges[block + 1]);
        let q = q_lo + rng.f64() * (q_hi - q_lo);
        self.sigma_of_q(q) as f32
    }

    /// Sample a noise level from the full log-normal `p_σ`, restricted only
    /// to `[σ_min, σ_max]`. Used by recurrent-depth training (Appendix E.5:
    /// no partitioning — the whole network is one denoiser, "sampling
    /// different noise levels σ at each training step").
    pub fn sample_sigma_full(&self, rng: &mut fastrand::Rng) -> f32 {
        self.sample_sigma(0, 1, rng)
    }

    /// EDM loss weight `w(σ) = (σ² + σ_data²)/(σ·σ_data)²`, `σ_data = 0.5`.
    ///
    /// Appendix C.2 (verbatim): "we use the EDM weighting function
    /// `w(σ) = (σ² + σ_data²)/(σ·σ_data)²` with `σ_data = 0.5` for all
    /// experiments". In the `σ ≫ σ_data` regime this is the `1/σ²`
    /// score-matching weight (Vincent 2011) that balances the `σ²` growth
    /// of the optimal denoiser's error; as `σ → 0` it stays finite at
    /// `1/σ_data²`, so the lowest-noise samples are not over-weighted.
    pub fn weight(&self, sigma: f32) -> f32 {
        let s = f64::from(sigma);
        ((s * s + self.sigma_data * self.sigma_data) / (s * self.sigma_data).powi(2)) as f32
    }

    /// One Euler step of the probability-flow ODE `dz/dσ = (z - D(z, σ))/σ`:
    /// `z' = z + ((σ_next - σ_prev)/σ_prev)·(z - D)`, moving from the
    /// current noise level `σ_prev` down to `σ_next`.
    ///
    /// For a perfect denoiser (`D = y`) this maps `y + σ_prev·ε` exactly
    /// onto `y + σ_next·ε`. Composition walks blocks from high to low noise:
    /// start with `z = x` at `σ_max` and step with `sigma_prev =
    /// σ_edges[i+1]`, `sigma_next = σ_edges[i]` for `i = B-1 .. 0`.
    pub fn euler_step(
        z: Tensor<2>,
        denoised: Tensor<2>,
        sigma_prev: f32,
        sigma_next: f32,
    ) -> Tensor<2> {
        let scale = (sigma_next - sigma_prev) / sigma_prev;
        z.clone().add(z.sub(denoised).mul_scalar(scale))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Distribution};

    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn partition_covers_range_and_equal_mass() {
        let schedule = NoiseSchedule::default();
        let n_blocks = 8;
        let mut prev_high = None;
        for b in 0..n_blocks {
            let (sigma_high, sigma_low) = schedule.partition(b, n_blocks);
            assert!(sigma_low < sigma_high, "range must be ascending");
            // contiguous tiling: this block's low edge is the previous
            // block's high edge (block b covers [σ_b, σ_{b+1}])
            if let Some(prev_high) = prev_high {
                assert_eq!(prev_high, sigma_low, "blocks must tile contiguously");
            }
            prev_high = Some(sigma_high);
            // equal mass: the log-normal CDF mass of the interval is 1/B
            let mass = normal_cdf((sigma_high.ln() - schedule.p_mean) / schedule.p_std)
                - normal_cdf((sigma_low.ln() - schedule.p_mean) / schedule.p_std);
            let expected = 1.0 / n_blocks as f64;
            assert!(
                (mass - expected).abs() < 1e-2,
                "block {b} mass {mass} != 1/B = {expected}"
            );
        }
        // the tiling covers [σ_min, σ_max]
        let (_, first_low) = schedule.partition(0, n_blocks);
        assert!(
            (first_low - schedule.sigma_min).abs() < 1e-6,
            "must start at σ_min"
        );
        let (last_high, _) = schedule.partition(n_blocks - 1, n_blocks);
        assert!(
            (last_high - schedule.sigma_max).abs() < 1e-6,
            "must end at σ_max"
        );
    }

    #[test]
    fn sample_sigma_stays_in_block_range() {
        let schedule = NoiseSchedule::default();
        let n_blocks = 8;
        let mut rng = fastrand::Rng::with_seed(42);
        for b in 0..n_blocks {
            let (sigma_high, sigma_low) = schedule.partition(b, n_blocks);
            for _ in 0..200 {
                let s = f64::from(schedule.sample_sigma(b, n_blocks, &mut rng));
                assert!(
                    s >= sigma_low * (1.0 - 1e-5) && s <= sigma_high * (1.0 + 1e-5),
                    "sample {s} outside [{sigma_low}, {sigma_high}]"
                );
            }
        }
    }

    #[test]
    fn sample_sigma_full_covers_sigma_range() {
        let schedule = NoiseSchedule::default();
        let mut rng = fastrand::Rng::with_seed(42);
        for _ in 0..1000 {
            let s = f64::from(schedule.sample_sigma_full(&mut rng));
            assert!(
                s >= schedule.sigma_min && s <= schedule.sigma_max,
                "sample {s} outside [{}, {}]",
                schedule.sigma_min,
                schedule.sigma_max
            );
        }
    }

    #[test]
    fn inv_normal_cdf_accuracy() {
        // roundtrip through the CDF (used by the equal-mass test). In the
        // bulk (|x| <= 3) the roundtrip is limited by f64 CDF rounding;
        // at the extreme tail |Φ(±6)| ~ 1e-9 the CDF's absolute error
        // (~2e-16) blows up to ~1e-7 in x-space - still 5 orders below the
        // 1e-2 partition tolerance, and irrelevant for the default schedule
        // (whose q range maps to x in [-4.2, 4.7]).
        for x in [-3.0, -1.0, 0.0, 1.0, 3.0] {
            let err = (inv_normal_cdf(normal_cdf(x)) - x).abs();
            assert!(err < 1e-9, "roundtrip error {err} at x = {x}");
        }
        for x in [-6.0, 6.0] {
            let err = (inv_normal_cdf(normal_cdf(x)) - x).abs();
            assert!(err < 1e-7, "roundtrip error {err} at x = {x}");
        }
        assert!(inv_normal_cdf(0.5).abs() < 1e-15);
    }

    #[test]
    fn euler_step_reduces_sigma() {
        // perfect denoiser: D(z) = y, so z' = y + σ_next·ε exactly
        let y = Tensor::<2>::random([4, 8], Distribution::Default, &dev());
        let eps = Tensor::<2>::random([4, 8], Distribution::Normal(0.0, 1.0), &dev());
        let (sigma_prev, sigma_next) = (1.0f32, 0.4f32);
        let z = y.clone().add(eps.clone().mul_scalar(sigma_prev));
        let z_new = NoiseSchedule::euler_step(z.clone(), y.clone(), sigma_prev, sigma_next);
        let expected = y.clone().add(eps.mul_scalar(sigma_next));
        let max_diff: f32 = z_new
            .clone()
            .sub(expected.clone())
            .abs()
            .max()
            .into_scalar();
        assert!(
            max_diff < 1e-4,
            "perfect denoiser must hit σ_next exactly, diff {max_diff}"
        );
        // and it is strictly closer to y than the pre-step state
        let dist_old: f32 = z.sub(y.clone()).abs().max().into_scalar();
        let dist_new: f32 = expected.sub(y).abs().max().into_scalar();
        assert!(dist_new < dist_old, "noise must decrease");
    }

    #[test]
    fn weight_matches_edm_formula() {
        // Appendix C.2 verbatim: w(σ) = (σ² + σ_data²)/(σ·σ_data)² with
        // σ_data = 0.5 for all experiments.
        let schedule = NoiseSchedule::default();
        assert_eq!(schedule.sigma_data, 0.5);
        for sigma in [0.01f32, 0.1, 0.5, 1.0, 10.0, 80.0] {
            let s = f64::from(sigma);
            let expected = ((s * s + 0.25) / (s * 0.5).powi(2)) as f32;
            let got = schedule.weight(sigma);
            assert!(
                (got - expected).abs() <= expected.abs() * 1e-6 + 1e-7,
                "w({sigma}) = {got} != EDM {expected}"
            );
        }
    }

    #[test]
    fn weight_is_finite_and_decreasing() {
        let schedule = NoiseSchedule::default();
        let w_min = schedule.weight(schedule.sigma_min as f32);
        let w_max = schedule.weight(schedule.sigma_max as f32);
        assert!(w_min.is_finite() && w_max.is_finite());
        assert!(w_max < w_min, "weight must penalize high-σ samples less");
    }
}
