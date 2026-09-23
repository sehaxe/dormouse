//! Normal-CDF helpers: the standard normal CDF via `erfc` (Cephes/musl
//! implementation from the `libm` crate, ~1 ulp) and its inverse via
//! Acklam's rational approximation plus one Halley refinement.
//!
//! Accuracy: the Acklam approximation alone is accurate to ~1e-4 relative
//! near the tails; with the Halley step the maximum relative error is
//! < 1.15e-9 over `p in [1e-300, 1 - 1e-300]` (Acklam's published bound).
//! The `inv_normal_cdf_accuracy` roundtrip test confirms < 1e-9 in the bulk
//! (|x| <= 3) and < 1e-7 at the extreme tail (|x| = 6, where f64 rounding
//! of `Φ` limits the roundtrip). The refinement is skipped for `|x| > 27`
//! where `exp(x²/2)` would overflow and the correction is below f64
//! resolution anyway. All partition/sampling logic needs only ~1e-6, so
//! this is far tighter than required.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// Standard normal CDF `Φ(x)` via `0.5·erfc(-x/√2)`.
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x * FRAC_1_SQRT_2)
}

/// Inverse standard normal CDF `Φ⁻¹(p)` for `p ∈ (0, 1)`.
///
/// Acklam's rational approximation (central + two tail regions) followed by
/// one Halley iteration. Maximum relative error < 1.15e-9.
pub fn inv_normal_cdf(p: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0, "p must be in (0, 1), got {p}");
    let p = p.clamp(1e-300, 1.0 - 1e-300);

    // Central-region coefficients (Acklam's algorithm, published constants).
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    // Tail-region coefficients (shared by both tails).
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;

    let x = if p < P_LOW {
        // Lower tail: q = sqrt(-2 ln p), rational in q.
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        // Central region: rational in r = (p - 0.5)², times (p - 0.5).
        let r = p - 0.5;
        let s = r * r;
        r * (((((A[0] * s + A[1]) * s + A[2]) * s + A[3]) * s + A[4]) * s + A[5])
            / (((((B[0] * s + B[1]) * s + B[2]) * s + B[3]) * s + B[4]) * s + 1.0)
    } else {
        // Upper tail (mirror of the lower tail).
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // One Halley refinement: e = Φ(x) - p, u = e·√(2π)·exp(x²/2),
    // x -= u / (1 + x·u/2). Skipped for |x| > 27 where exp overflows and
    // the correction is below f64 resolution.
    if x.abs() < 27.0 {
        let e = 0.5 * libm::erfc(-x * FRAC_1_SQRT_2) - p;
        let u = e * (2.0 * PI).sqrt() * (0.5 * x * x).exp();
        return x - u / (1.0 + x * u / 2.0);
    }
    x
}
