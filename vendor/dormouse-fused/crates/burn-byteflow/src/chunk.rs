//! Coding-rate chunking: the downsampling stage of ByteFlow Net (paper §3.2).
//!
//! Positions with high marginal coding rate `ΔR_t` carry more information and
//! are promoted to the global level; the rest is compressed away. Selection is
//! Top-K over a fixed K, which keeps the computation graph static (the paper's
//! argument against global-threshold chunking).

use burn::tensor::{Int, Tensor, TensorData};

/// Selection scoring mode for the chunker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateMode {
    /// L2 streaming approximation of the coding rate, paper Appendix B
    /// (`R ∝ ‖H‖₂`). Default fast path: pure tensor ops, no host sync.
    L2,
    /// Exact lossy coding rate, paper eq. (11). Host-side Cholesky per prefix.
    LogDet,
}

/// Cholesky log-determinant of a symmetric positive-definite matrix.
fn logdet_spd(m: &[f64], n: usize) -> Option<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = m[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = if i == j {
                if s <= 0.0 {
                    return None;
                }
                s.sqrt()
            } else {
                s / l[j * n + j]
            };
        }
    }
    Some(2.0 * (0..n).map(|i| l[i * n + i].ln()).sum::<f64>())
}

/// Gram matrix HᵀH accumulated from row-major `rows` ([t][d] at offset row*t*d).
///
/// Returns log det(I_d + c·HᵀH) for every prefix length 1..=t (Sylvester:
/// det(I_T + cHHᵀ) = det(I_d + cHᵀH)).
fn prefix_logdets(rows: &[f64], t: usize, d: usize, c: f64) -> Vec<f64> {
    let mut gram = vec![0.0f64; d * d];
    let mut out = Vec::with_capacity(t);
    for p in 0..t {
        let off = p * d;
        for i in 0..d {
            let hi = rows[off + i];
            if hi != 0.0 {
                for j in 0..d {
                    gram[i * d + j] += hi * rows[off + j];
                }
            }
        }
        let m: Vec<f64> = (0..d * d)
            .map(|ix| {
                if ix / d == ix % d {
                    1.0 + c * gram[ix]
                } else {
                    c * gram[ix]
                }
            })
            .collect();
        out.push(0.5 * logdet_spd(&m, d).expect("I + c·HᵀH must be SPD"));
    }
    out
}

/// Marginal coding rates `ΔR_t = R(h_1:t) − R(h_1:t−1)` via the exact lossy
/// coding rate (paper eq. 11–12), `[B, T]`.
///
/// ponytail: host-synced O(T·d³) per row — analysis and small-scale validation
/// only; training uses [`RateMode::L2`] (within 0.01 BPB of log-det in the
/// paper's Table 4).
pub fn marginal_gains_exact(h: Tensor<3>, eps2: f64) -> Tensor<2> {
    let device = h.device();
    let [b, t, d] = h.dims();
    let c = d as f64 / eps2;
    let rows: Vec<f64> = h.into_data().convert::<f64>().try_to_vec().unwrap();
    let mut gains = vec![0.0f64; b * t];
    for row in 0..b {
        let ld = prefix_logdets(&rows[row * t * d..(row + 1) * t * d], t, d, c);
        let mut prev = 0.0;
        for (p, r) in ld.iter().enumerate() {
            gains[row * t + p] = r - prev;
            prev = *r;
        }
    }
    let rates: Vec<f32> = gains.into_iter().map(|v| v as f32).collect();
    Tensor::from_data(TensorData::new(rates, [b, t]), &device)
}

/// Full-sequence lossy coding rate `R_ε(h_1:T)` (paper eq. 11), `[B]`.
///
/// Same host-sync caveat as [`marginal_gains_exact`].
pub fn coding_rate_exact(h: Tensor<3>, eps2: f64) -> Tensor<1> {
    let device = h.device();
    let [b, t, d] = h.dims();
    let c = d as f64 / eps2;
    let rows: Vec<f64> = h.into_data().convert::<f64>().try_to_vec().unwrap();
    let rates: Vec<f32> = (0..b)
        .map(|row| {
            let last = prefix_logdets(&rows[row * t * d..(row + 1) * t * d], t, d, c);
            last[t - 1] as f32
        })
        .collect();
    Tensor::from_data(TensorData::new(rates, [b]), &device)
}

/// Marginal gains under the Appendix B L2 approximation, `[B, T]`.
///
/// `R_t ∝ √S_t` with running sum of squared norms `S_t = Σ_{i≤t} ‖h_i‖²`, so
/// `ΔR_t = √S_t − √S_{t−1}` — monotone in each position's contribution and a
/// telescoping decomposition of the total rate. Pure tensor ops.
pub fn marginal_gains_l2(h: Tensor<3>) -> Tensor<2> {
    let [b, t, _d] = h.dims();
    let sq_norms = h.powf_scalar(2.0).sum_dim(2); // [B,T,1]
    let cum_rate = sq_norms.cumsum(1).sqrt().reshape([b, t]); // ∝ R(h_1:t)
    let shifted = Tensor::cat(
        vec![
            Tensor::<2>::zeros([b, 1], &cum_rate.device()),
            cum_rate.clone().slice([0..b, 0..t - 1]),
        ],
        1,
    );
    cum_rate.sub(shifted)
}

/// Top-K selection of positions to promote to the global level.
///
/// `S = {position 0 (BOS)} ∪ top-(K−1)` of positions `1..T` by gain, sorted
/// chronologically (paper §3.2). Returns int indices `[B, K]` ascending.
pub fn select_positions(gains: Tensor<2>, k: usize) -> Tensor<2, Int> {
    let [b, t] = gains.dims();
    assert!(k >= 1, "k must be >= 1");
    let device = gains.device();
    if k >= t {
        // Everything is selected; identity order.
        return Tensor::<1, Int>::arange(0..t as i64, &device)
            .reshape([1, t])
            .expand([b, t]);
    }
    // Candidates exclude position 0: it is always kept as the BOS boundary.
    // argtopk indices are relative to the candidate slice ⇒ shift by 1.
    let cand = gains.slice([0..b, 1..t]);
    let idx = cand.argtopk(k - 1, 1).add_scalar(1i64); // [B,k-1] descending by gain
                                                       // Chronological order: argsort the picked index values ascending.
    let perm = idx.clone().argsort(1);
    let picked = idx.gather(1, perm);
    let bos = Tensor::<2, Int>::zeros([b, 1], &device);
    Tensor::cat(vec![bos, picked], 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Distribution};

    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn exact_rate_matches_closed_form_orthonormal() {
        // Orthonormal rows: HHᵀ = I_t ⇒ R = ½·t·ln(1 + d/ε²).
        // h = first three unit basis vectors in d=4.
        let h = Tensor::<1>::from_floats(
            [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            &dev(),
        )
        .reshape([1, 3, 4]);
        let eps2 = 0.5f64;
        let expected = 0.5 * 3.0 * (1.0 + 4.0 / eps2).ln();
        let got: f32 = coding_rate_exact(h, eps2).into_scalar();
        assert!(
            (got as f64 - expected).abs() < 1e-5,
            "exact rate {got} vs closed form {expected}"
        );
    }

    #[test]
    fn exact_rate_finite_on_rank_deficient() {
        // Duplicated rows collapse rank; the rate must stay finite (log of the
        // clamped eigenvalues, never of zero).
        let h = Tensor::<1>::from_floats([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], &dev()).reshape([1, 3, 2]);
        let r: f32 = coding_rate_exact(h, 1e-2).into_scalar();
        assert!(r.is_finite() && r > 0.0);
    }

    #[test]
    fn l2_gains_telescope_to_total_rate() {
        // ΣΔR_t = R(T) − R(0) = √S_T exactly, for any input.
        let dev = dev();
        let h = Tensor::<3>::random([3, 17, 5], Distribution::Default, &dev);
        let gains = marginal_gains_l2(h.clone());
        let total: Vec<f32> = h
            .powf_scalar(2.0)
            .sum_dim(2)
            .sum_dim(1)
            .sqrt()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        let summed: Vec<f32> = gains
            .sum_dim(1)
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        for (s, g) in total.iter().zip(summed.iter()) {
            assert!((s - g).abs() < 1e-4, "telescope {g} vs {s}");
        }
    }

    #[test]
    fn constant_stream_prefers_early_positions_outlier_wins() {
        let dev = dev();
        // All-identical representations: ΔR_t = √(tc)−√((t−1)c) strictly
        // decreases ⇒ top picks are the earliest positions.
        let ones = vec![1.0f32; 16 * 4];
        let flat = Tensor::<1>::from_floats(ones.as_slice(), &dev).reshape([1, 16, 4]);
        let sel_const = select_positions(marginal_gains_l2(flat), 4);
        let sv: Vec<i64> = sel_const.into_data().convert::<i64>().to_vec().unwrap();
        assert_eq!(sv, vec![0, 1, 2, 3], "constant stream should pick early");

        // Inject one huge-norm position: its ΔR dominates and must be selected.
        let mut mixed = vec![1.0f32; 16 * 4];
        for v in &mut mixed[10 * 4..11 * 4] {
            *v = 100.0;
        }
        let flat = Tensor::<1>::from_floats(mixed.as_slice(), &dev).reshape([1, 16, 4]);
        let sel = select_positions(marginal_gains_l2(flat), 4);
        let sv: Vec<i64> = sel.into_data().convert::<i64>().to_vec().unwrap();
        assert!(
            sv.contains(&10),
            "outlier position must be selected, got {sv:?}"
        );
    }

    #[test]
    fn selection_is_bos_first_chronological() {
        let dev = dev();
        let gains = Tensor::<2>::random([4, 33], Distribution::Default, &dev);
        let sel = select_positions(gains, 7);
        assert_eq!(sel.dims(), [4, 7]);
        for row in sel
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .unwrap()
            .chunks(7)
        {
            assert_eq!(row[0], 0, "position 0 (BOS) forced");
            assert!(
                row.windows(2).all(|w: &[i64]| w[0] < w[1]),
                "must ascend: {row:?}"
            );
        }
    }

    #[test]
    fn exact_marginal_gains_sum_to_full_rate() {
        // Consistency between exact prefix rates and their differences.
        let dev = dev();
        let h = Tensor::<3>::random([1, 9, 6], Distribution::Default, &dev);
        let full: f32 = coding_rate_exact(h.clone(), 0.25).into_scalar();
        let total: f32 = marginal_gains_exact(h, 0.25).sum_dim(1).into_scalar();
        assert!((full - total).abs() < 1e-4, "{total} vs {full}");
    }
}
