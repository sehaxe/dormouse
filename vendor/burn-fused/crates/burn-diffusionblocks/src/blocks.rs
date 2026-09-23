//! [`BlockPartition`] — precomputed block noise ranges.
//!
//! The same equi-probability partition as [`NoiseSchedule::partition`], but
//! the quantile boundaries are computed once at construction instead of per
//! call. The actual model partitioning is the user's job (the crate provides
//! schedule + objective); see the crate docs for the "recurrent-depth"
//! usage sketch.

use crate::NoiseSchedule;

/// Precomputed equi-probability partition of `[σ_min, σ_max]` into
/// `n_blocks` intervals of equal log-normal mass.
///
/// Blocks are indexed from low noise (block 0, nearest `σ_min`) to high
/// noise (block `n_blocks - 1`, nearest `σ_max`); `range_for(b)` returns
/// `(σ_high, σ_low)`.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockPartition {
    schedule: NoiseSchedule,
    n_blocks: usize,
    /// Normal-CDF quantiles `q_b` of the boundaries, `len = n_blocks + 1`,
    /// ascending (`q_0 = Φ((ln σ_min − p_mean)/p_std)`).
    q_edges: Vec<f64>,
    /// `σ_b = exp(p_mean + p_std·q_b)`, `len = n_blocks + 1`, ascending.
    sigma_edges: Vec<f64>,
}

impl BlockPartition {
    /// Precompute the partition of `schedule` into `n_blocks` blocks.
    ///
    /// # Panics
    /// If `n_blocks == 0`.
    pub fn new(schedule: NoiseSchedule, n_blocks: usize) -> Self {
        assert!(n_blocks > 0, "n_blocks must be > 0");
        let q_edges = schedule.q_edges(n_blocks);
        let sigma_edges = q_edges.iter().map(|&q| schedule.sigma_of_q(q)).collect();
        Self {
            schedule,
            n_blocks,
            q_edges,
            sigma_edges,
        }
    }

    /// The schedule this partition was built from.
    pub fn schedule(&self) -> NoiseSchedule {
        self.schedule
    }

    /// Number of blocks.
    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    /// Block `block`'s noise range `(σ_high, σ_low)`.
    ///
    /// # Panics
    /// If `block >= n_blocks`.
    pub fn range_for(&self, block: usize) -> (f64, f64) {
        assert!(
            block < self.n_blocks,
            "block {block} out of range [0, {})",
            self.n_blocks
        );
        (self.sigma_edges[block + 1], self.sigma_edges[block])
    }

    /// Sample a noise level from block `block`'s restricted log-normal
    /// (inverse-CDF, see [`NoiseSchedule::sample_sigma`]).
    ///
    /// # Panics
    /// If `block >= n_blocks`.
    pub fn sample_sigma(&self, block: usize, rng: &mut fastrand::Rng) -> f32 {
        assert!(
            block < self.n_blocks,
            "block {block} out of range [0, {})",
            self.n_blocks
        );
        let q = self.q_edges[block] + rng.f64() * (self.q_edges[block + 1] - self.q_edges[block]);
        self.schedule.sigma_of_q(q) as f32
    }

    /// The `n_blocks + 1` noise boundaries, ascending from `σ_min` to
    /// `σ_max`.
    pub fn sigma_edges(&self) -> &[f64] {
        &self.sigma_edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_edges_match_schedule() {
        let schedule = NoiseSchedule::default();
        let partition = BlockPartition::new(schedule, 6);
        assert_eq!(partition.sigma_edges().len(), 7);
        for b in 0..6 {
            let (hi, lo) = partition.range_for(b);
            let (hi_ref, lo_ref) = schedule.partition(b, 6);
            assert_eq!(hi, hi_ref);
            assert_eq!(lo, lo_ref);
            assert_eq!(lo, partition.sigma_edges()[b]);
            assert_eq!(hi, partition.sigma_edges()[b + 1]);
        }
        // sampling stays inside the cached range
        let mut rng = fastrand::Rng::with_seed(1);
        for b in 0..6 {
            let (hi, lo) = partition.range_for(b);
            for _ in 0..100 {
                let s = f64::from(partition.sample_sigma(b, &mut rng));
                assert!(s >= lo * (1.0 - 1e-5) && s <= hi * (1.0 + 1e-5));
            }
        }
    }
}
