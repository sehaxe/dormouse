//! The streaming coding-rate patcher — the paper's §3.2 chunker as STATE.
//!
//! The batch path ([`crate::chunk::marginal_gains_exact`] /
//! [`crate::chunk::marginal_gains_l2`] with [`crate::chunk::select_positions`])
//! scores a whole window at once: it needs the local encoder's hidden states
//! as one `[B, T, d]` tensor. This module carries the SAME quantities as
//! running state, so a byte stream can be patched position by position:
//!
//! - [`RatePatcher::push`] consumes ONE hidden row and returns its marginal
//!   coding rate `ΔR_t = R(h_1:t) − R(h_1:t−1)` (paper eq. 12);
//! - [`RatePatcher::borders`] returns the Top-K boundary set over the window
//!   pushed so far: `{position 0 (BOS)} ∪ top-(K−1) by ΔR`, sorted
//!   chronologically (paper §3.2);
//! - [`RatePatcher::reset`] starts the next window.
//!
//! Window state is the d×d Gram matrix (LogDet mode) or one running sum of
//! squared norms (L2 mode) — never a T-length buffer of rows, so a stream of
//! any length patches in O(d²) memory. A window is ONE sequence `h_1:W` in the
//! paper's sense: the rate is per-window, and `reset()` begins the next one,
//! which is exactly the semantics the batch path gives a teacher-forced
//! training window.
//!
//! The two modes are the paper's own pair: LogDet is eq. (11) computed
//! incrementally (per-push Cholesky of the accumulated Gram through the
//! Sylvester folding — the same route the batch path takes), and L2 is
//! Appendix B's streaming approximation (`R ∝ ‖H‖₂`), which is what the paper
//! itself offers "for streaming applications where quick local decisions are
//! required".
//!
//! ponytail: LogDet push is O(d³) (a fresh Cholesky per position) — fine for
//! analysis and CPU patching at d_local ≈ 96; a maintained rank-1 Cholesky
//! update would make it O(d²) if a profile ever says it matters. The L2 push
//! is O(d) and is the intended streaming path.

use crate::chunk::{logdet_spd, RateMode};

/// The streaming coding-rate patcher over one window of hidden states.
#[derive(Clone, Debug)]
pub struct RatePatcher {
    mode: RateMode,
    d: usize,
    /// `d_local / ε²` — the constant inside eq. (11)'s determinant.
    c: f64,
    /// Accumulated Gram `HᵀH` (row-major `[d, d]`), LogDet mode only.
    gram: Vec<f64>,
    /// Running `Σ ‖h‖²`, L2 mode only.
    sum_sq: f64,
    /// `R(h_1:t)` of the current window.
    rate: f64,
    /// `R(h_1:t−1)` — the value the next ΔR subtracts.
    prev_rate: f64,
    /// `ΔR` per pushed position, in push order (window-local).
    gains: Vec<f64>,
}

impl RatePatcher {
    /// A patcher over `d`-dimensional hidden states at noise variance `eps2`.
    pub fn new(d: usize, eps2: f64, mode: RateMode) -> Self {
        assert!(d >= 1, "hidden width must be >= 1");
        assert!(eps2 > 0.0, "eps2 must be > 0 (it divides d inside the logdet)");
        Self {
            mode,
            d,
            c: d as f64 / eps2,
            gram: vec![0.0; d * d],
            sum_sq: 0.0,
            rate: 0.0,
            prev_rate: 0.0,
            gains: Vec::new(),
        }
    }

    /// Feed ONE hidden state row; returns `ΔR_t` for its position in the
    /// window. The first pushed position has ΔR = R(h_1:1) ≥ 0.
    ///
    /// Panics when `h.len() != d` — a shape error a long way from the caller
    /// that caused it is the silent-crop class, not a clamp (AGENTS §1.1).
    pub fn push(&mut self, h: &[f32]) -> f64 {
        assert_eq!(h.len(), self.d, "hidden row width must equal the patcher's d");
        match self.mode {
            RateMode::LogDet => {
                let d = self.d;
                // Same accumulation as chunk::prefix_logdets: f64 products of
                // the f32 rows, row-major, zero rows skipped (a no-op add).
                if h.iter().any(|&v| v != 0.0) {
                    for (i, &hi32) in h.iter().enumerate() {
                        let hi = hi32 as f64;
                        if hi == 0.0 {
                            continue;
                        }
                        for (j, &hv) in h.iter().enumerate() {
                            self.gram[i * d + j] += hi * hv as f64;
                        }
                    }
                }
                let m: Vec<f64> = (0..d * d)
                    .map(|ix| {
                        if ix / d == ix % d {
                            1.0 + self.c * self.gram[ix]
                        } else {
                            self.c * self.gram[ix]
                        }
                    })
                    .collect();
                self.rate = 0.5 * logdet_spd(&m, d).expect("I + c·HᵀH must be SPD");
            }
            RateMode::L2 => {
                // Appendix B: R ∝ ‖h_1:t‖₂, i.e. the running sqrt-sum-of-squares.
                self.sum_sq += h.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
                self.rate = self.sum_sq.sqrt();
            }
        }
        let dr = self.rate - self.prev_rate;
        self.prev_rate = self.rate;
        self.gains.push(dr);
        dr
    }

    /// The running coding rate `R(h_1:t)` of the current window.
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// The marginal rates pushed so far, in window order.
    pub fn gains(&self) -> &[f64] {
        &self.gains
    }

    /// Positions pushed this window.
    pub fn len(&self) -> usize {
        self.gains.len()
    }

    /// Whether nothing has been pushed yet.
    pub fn is_empty(&self) -> bool {
        self.gains.is_empty()
    }

    /// Begin the next window: the gain history and the rate state clear. The
    /// paper's chunker scores one sequence h_1:T; a stream of windows is a
    /// stream of sequences.
    pub fn reset(&mut self) {
        self.gram.iter_mut().for_each(|v| *v = 0.0);
        self.sum_sq = 0.0;
        self.rate = 0.0;
        self.prev_rate = 0.0;
        self.gains.clear();
    }

    /// Top-K patch borders over the window pushed so far: `{position 0 (BOS)}
    /// ∪ top-(K−1)` of positions `1..T` by ΔR, sorted chronologically — the
    /// paper §3.2 selection rule. Ties break on position (ascending), so the
    /// result is a pure function of the gains.
    ///
    /// `k = 1` is the BOS boundary alone; `k ≥ T` selects every position.
    pub fn borders(&self, k: usize) -> Vec<usize> {
        assert!(k >= 1, "k must be >= 1");
        let t = self.gains.len();
        assert!(t >= 1, "borders on an empty window: push at least one position");
        if k >= t {
            return (0..t).collect();
        }
        let mut idx: Vec<usize> = (1..t).collect();
        idx.sort_by(|&a, &b| {
            self.gains[b]
                .partial_cmp(&self.gains[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        idx.truncate(k - 1);
        let mut out = vec![0usize];
        out.extend(idx);
        out.sort_unstable();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny deterministic LCG so the fixtures do not depend on burn's RNG
    /// state or on device-side sampling.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
        fn row(&mut self, d: usize, scale: f32) -> Vec<f32> {
            (0..d).map(|_| self.next_f32() * scale).collect()
        }
    }

    #[test]
    fn l2_push_matches_the_batch_cumsum() {
        // The streaming L2 state is Appendix B's running norm: the gains must
        // reproduce the batch path's √cumsum − shift to float tolerance.
        let mut rng = Lcg(7);
        let (t, d) = (17usize, 5usize);
        let rows: Vec<Vec<f32>> = (0..t).map(|_| rng.row(d, 1.0)).collect();
        let mut p = RatePatcher::new(d, 0.5, RateMode::L2);
        let got: Vec<f64> = rows.iter().map(|r| p.push(r)).collect();
        // Batch reference, computed by hand: cumsum of squared norms.
        let mut s = 0.0f64;
        let mut prev = 0.0f64;
        for (i, r) in rows.iter().enumerate() {
            s += r.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
            let rate = s.sqrt();
            assert!((rate - prev - got[i]).abs() < 1e-12, "gain {i}");
            prev = rate;
        }
        assert_eq!(p.len(), t);
        // Telescope: the gains sum to the running rate (eq. 12's identity).
        let total: f64 = got.iter().sum();
        assert!((total - p.rate()).abs() < 1e-12);
    }

    #[test]
    fn reset_starts_a_fresh_sequence() {
        let mut p = RatePatcher::new(3, 0.5, RateMode::L2);
        p.push(&[1.0, 2.0, 3.0]);
        let r1 = p.rate();
        p.push(&[4.0, 5.0, 6.0]);
        assert!(p.rate() > r1);
        p.reset();
        assert_eq!(p.rate(), 0.0);
        assert!(p.is_empty());
        // The restarted window must reproduce the first window's rate exactly:
        // the state is the sum, not a carry.
        p.push(&[1.0, 2.0, 3.0]);
        assert_eq!(p.rate(), r1);
    }
}
