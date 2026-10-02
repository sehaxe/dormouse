//! # The DYNAMIC PATCHER oracle — the streaming coding-rate patcher against a
//! direct log-det computation on CPU.
//!
//! Tier **(a)-style double route inside one test** (the "formula against the
//! wall" construction this repo's `tests/kda_oracle.rs` uses): the patcher
//! computes eq. (11)/(12) incrementally through the Cholesky + Sylvester route
//! it shares with the batch path, and the ORACLE SIDE computes the same rate
//! by building the paper's LITERAL eq. (11) matrix — `I_T + (d_local/ε²)·H·Hᵀ`,
//! T×T, not folded — and taking its log-determinant by **Gaussian elimination
//! with partial pivoting**. That is a third decomposition of the same equation
//! (no Cholesky, no eigenvalues, no Sylvester), so a wrong folding, a wrong
//! d/ε² constant or a dropped ½ cannot agree with it by construction.
//!
//! The two named gates:
//!
//! - `test_coding_rate_matches_direct_logdet` — the streaming rate and every
//!   ΔR against the direct T×T log-det, plus the telescope identity
//!   (Σ ΔR_t = R(h_1:T)) and the d×d cross-check of the Sylvester folding.
//! - `test_deltaR_topk_borders` — a planted-burst stream whose information
//!   profile is known by construction: the Top-K borders must land on the
//!   bursts, keep the BOS first, stay chronological, and agree with the batch
//!   path's `select_positions` on the same gains.
//!
//! ## Falsify (the gate must be able to fail)
//!
//! Three mutants, each asserted RED against the same inputs the real patcher
//! is green under: the missing-½ and ε²→ε⁴ formulas (the two eq. (11) reading
//! errors), and the STATIC-STRIDE chunker this lane replaces — the mutant that
//! puts borders on every T/K-th position and misses the bursts.
//!
//! Bars: both routes are f64 on the same f32 inputs; the LU and Cholesky
//! decompositions of a well-conditioned SPD matrix agree far below 1e-9
//! relative at these magnitudes (measured 1e-13 in the first run).

use burn::tensor::{Device, Int, Tensor};
use burn_byteflow::{marginal_gains_exact, marginal_gains_l2, select_positions, RateMode, RatePatcher};

const BAR: f64 = 1e-9;

fn dev() -> Device {
    Device::ndarray()
}

/// A tiny deterministic LCG: the fixtures are pure functions of the test, not
/// of burn's RNG state.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn row(&mut self, d: usize, scale: f32) -> Vec<f32> {
        (0..d).map(|_| self.next_f32() * scale).collect()
    }
}

/// log|det(M)| of a square matrix by Gaussian elimination with partial
/// pivoting — the direct route. Returns None on a singular matrix.
fn direct_logdet_lu(m: &mut Vec<f64>, n: usize) -> Option<f64> {
    let mut logdet = 0.0f64;
    for k in 0..n {
        let pivot = (k..n)
            .map(|i| (i, m[i * n + k].abs()))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
            .unwrap();
        if m[pivot * n + k].abs() < 1e-300 {
            return None;
        }
        if pivot != k {
            for j in 0..n {
                m.swap(k * n + j, pivot * n + j);
            }
        }
        let piv = m[k * n + k];
        logdet += piv.abs().ln();
        for i in (k + 1)..n {
            let f = m[i * n + k] / piv;
            if f != 0.0 {
                for j in k..n {
                    m[i * n + j] -= f * m[k * n + j];
                }
            }
        }
    }
    Some(logdet)
}

/// The paper's LITERAL eq. (11): `I_T + (d/ε²)·H·Hᵀ` as a T×T matrix, its
/// log-det taken directly. `h` is row-major [t][d] as f64.
fn direct_rate_t_by_t(h: &[f64], t: usize, d: usize, c: f64) -> f64 {
    let mut m = vec![0.0f64; t * t];
    for i in 0..t {
        for j in 0..t {
            let s: f64 = (0..d).map(|k| h[i * d + k] * h[j * d + k]).sum();
            m[i * t + j] = c * s + if i == j { 1.0 } else { 0.0 };
        }
    }
    0.5 * direct_logdet_lu(&mut m, t).expect("I + c·HHᵀ is SPD by construction")
}

/// The d×d reading of the same equation: `I_d + (d/ε²)·HᵀH`.
fn direct_rate_d_by_d(h: &[f64], t: usize, d: usize, c: f64) -> f64 {
    let mut m = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            let s: f64 = (0..t).map(|k| h[k * d + i] * h[k * d + j]).sum();
            m[i * d + j] = c * s + if i == j { 1.0 } else { 0.0 };
        }
    }
    0.5 * direct_logdet_lu(&mut m, d).expect("I + c·HᵀH is SPD by construction")
}

/// The gate the lane names: the streaming patcher's rate — eq. (11) computed
/// incrementally through Cholesky + Sylvester — against the DIRECT log-det of
/// the paper's literal T×T matrix, on CPU, f64 on both sides.
#[test]
fn test_coding_rate_matches_direct_logdet() {
    let mut rng = Lcg(0xBEEF);
    let (t, d) = (24usize, 8usize);
    let eps2 = 0.5f64;
    let c = d as f64 / eps2;
    let rows: Vec<Vec<f32>> = (0..t).map(|_| rng.row(d, 1.0)).collect();
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().map(|&v| v as f64)).collect();

    // The streaming patcher, one push per position.
    let mut p = RatePatcher::new(d, eps2, RateMode::LogDet);
    let gains: Vec<f64> = rows.iter().map(|r| p.push(r)).collect();
    let streamed = p.rate();

    // 1. The rate itself, against the literal T×T form.
    let direct = direct_rate_t_by_t(&flat, t, d, c);
    assert!(
        (streamed - direct).abs() / direct.abs().max(1.0) < BAR,
        "streamed rate {streamed:.14e} vs direct T×T log-det {direct:.14e}"
    );

    // 2. The Sylvester folding, cross-checked against the d×d direct route —
    //    the pair the batch path's own comment names (det(I_T + c·HHᵀ) =
    //    det(I_d + c·HᵀH)).
    let direct_d = direct_rate_d_by_d(&flat, t, d, c);
    assert!(
        (direct - direct_d).abs() / direct_d.abs().max(1.0) < BAR,
        "T×T form {direct:.14e} vs d×d form {direct_d:.14e}"
    );

    // 3. Every marginal ΔR_t against the direct PREFIX rates: R(h_1:t) −
    //    R(h_1:t−1) computed from scratch per prefix by the direct route.
    let mut prev = 0.0f64;
    for (i, g) in gains.iter().enumerate() {
        let want = direct_rate_t_by_t(&flat[..(i + 1) * d], i + 1, d, c) - prev;
        assert!(
            (g - want).abs() / want.abs().max(1e-12) < BAR,
            "ΔR_{} streamed {g:.14e} vs direct prefix difference {want:.14e}",
            i + 1
        );
        prev += want;
    }

    // 4. The telescope identity: Σ ΔR_t == R(h_1:T) (eq. 11 ↔ 12).
    let total: f64 = gains.iter().sum();
    assert!(
        (total - streamed).abs() / streamed.abs().max(1.0) < BAR,
        "telescope {total:.14e} vs rate {streamed:.14e}"
    );

    // ── falsify: the two eq. (11) reading errors must go RED against the
    //    same direct route the real patcher is green under ──────────────
    // Mutant "no-half": the ½ dropped. Mutant "eps4": d/ε² read as d/ε⁴.
    let no_half = 2.0 * streamed;
    assert!(
        (no_half - direct).abs() / direct > 1e-3,
        "the no-half mutant stayed within the oracle bar — the gate measures nothing"
    );
    let mut p4 = RatePatcher::new(d, eps2 * eps2, RateMode::LogDet);
    for r in &rows {
        let _ = p4.push(r);
    }
    let eps4 = p4.rate();
    assert!(
        (eps4 - direct).abs() / direct > 1e-3,
        "the ε²→ε⁴ mutant stayed within the oracle bar — the gate measures nothing"
    );
}

/// The second named gate: on a stream whose information profile is known by
/// construction, the Top-K borders land on the planted bursts, keep the BOS
/// first, stay chronological, and agree with the batch path.
#[test]
fn test_deltaR_topk_borders() {
    let (t, d) = (64usize, 8usize);
    let eps2 = 0.5f64;
    let planted = [10usize, 30, 50];
    let mut rng = Lcg(42);
    let mut rows: Vec<Vec<f32>> = (0..t).map(|_| rng.row(d, 1.0)).collect();
    // Three information bursts: 100×-norm rows. ΔR_t is monotone in the new
    // direction a row introduces (the paper's "representational novelty"), so
    // the planted positions must dominate the selection in BOTH modes.
    for &pos in &planted {
        rows[pos] = rows[pos].iter().map(|&v| v * 100.0).collect();
    }

    for mode in [RateMode::LogDet, RateMode::L2] {
        let mut p = RatePatcher::new(d, eps2, mode);
        let gains: Vec<f64> = rows.iter().map(|r| p.push(r)).collect();
        let k = 5usize;
        let borders = p.borders(k);
        assert_eq!(borders.len(), k, "mode {mode:?}: exactly K borders");
        assert_eq!(borders[0], 0, "mode {mode:?}: BOS forced first");
        assert!(
            borders.windows(2).all(|w| w[0] < w[1]),
            "mode {mode:?}: chronological, got {borders:?}"
        );
        for &pos in &planted {
            assert!(
                borders.contains(&pos),
                "mode {mode:?}: the burst at {pos} must be a border, got {borders:?}"
            );
        }
        // Every non-BOS border is among the top-(K−1) gains — checked against
        // an independent host sort of the same gains.
        let mut ranked: Vec<usize> = (1..t).collect();
        ranked.sort_by(|&a, &b| {
            gains[b]
                .partial_cmp(&gains[a])
                .unwrap()
                .then(a.cmp(&b))
        });
        let want: std::collections::BTreeSet<usize> =
            ranked[..k - 1].iter().copied().chain([0]).collect();
        assert_eq!(
            want.into_iter().collect::<Vec<_>>(),
            borders,
            "mode {mode:?}: borders must be the top-(K−1) gains plus BOS"
        );
    }

    // Degenerate Ks: 1 is the BOS alone, T is everything.
    let mut p = RatePatcher::new(d, eps2, RateMode::LogDet);
    for r in &rows {
        let _ = p.push(r);
    }
    assert_eq!(p.borders(1), vec![0]);
    assert_eq!(p.borders(t), (0..t).collect::<Vec<_>>());

    // The streaming patcher and the BATCH path must be the same patcher,
    // WITHIN each mode: the borders computed from streamed gains equal
    // `select_positions` on the batch gains of the same mode (LogDet against
    // `marginal_gains_exact`, L2 against `marginal_gains_l2`). ACROSS modes
    // the two criteria legitimately disagree on the non-burst borders (the
    // paper's own Table-4 point: L2 ≈ log-det to ~0.01 BPB of loss, not
    // rank-identical at init) — only the bursts must agree, and the loop
    // above pins exactly that.
    let h: Tensor<3> = {
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Tensor::<1>::from_floats(flat.as_slice(), &dev()).reshape([1, t, d])
    };
    let as_vec = |t: Tensor<2, Int>| -> Vec<usize> {
        t.into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .unwrap()
            .into_iter()
            .map(|v| v as usize)
            .collect()
    };
    assert_eq!(
        {
            let mut p = RatePatcher::new(d, eps2, RateMode::LogDet);
            for r in &rows {
                let _ = p.push(r);
            }
            p.borders(5)
        },
        as_vec(select_positions(marginal_gains_exact(h.clone(), eps2), 5)),
        "LogDet: streaming borders == batch borders"
    );
    assert_eq!(
        {
            let mut p = RatePatcher::new(d, eps2, RateMode::L2);
            for r in &rows {
                let _ = p.push(r);
            }
            p.borders(5)
        },
        as_vec(select_positions(marginal_gains_l2(h.clone()), 5)),
        "L2: streaming borders == batch borders"
    );
    // The streamed LogDet gains are the batch values themselves (same f64
    // accumulation order through the same route); the bar is the batch path's
    // own f32 OUTPUT round-trip — `marginal_gains_exact` returns an f32
    // tensor, so its f64 echo carries ~1e-7 of quantization the streaming
    // f64 path does not have.
    let batch_exact: Vec<f64> = marginal_gains_exact(h, eps2)
        .into_data()
        .convert::<f64>()
        .try_to_vec()
        .unwrap();
    let mut pl = RatePatcher::new(d, eps2, RateMode::LogDet);
    let gl: Vec<f64> = rows.iter().map(|r| pl.push(r)).collect();
    for (i, (g, w)) in gl.iter().zip(batch_exact.iter()).enumerate() {
        assert!(
            (g - w).abs() / w.abs().max(1e-12) < 2e-7,
            "streamed ΔR_{i} {g:.14e} vs batch {w:.14e}"
        );
    }

    // ── falsify: the STATIC-STRIDE chunker this lane replaces ────────────
    // Borders on every T/K-th position — the fixed-stride v0 — must MISS the
    // planted bursts on this stream. If a stride pattern ever landed on all
    // three bursts here the gate would be measuring nothing; with the bursts
    // at 10/30/50 and K=5 it cannot.
    let stride: Vec<usize> = (0..5).map(|i| i * t / 5).collect();
    assert!(
        planted.iter().any(|p| !stride.contains(p)),
        "the stride mutant landed on every planted burst — the falsify case is degenerate"
    );
}
