//! # burn-parcae — stable looping via spectral retention
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! Implementation of the Parcae stability mechanism from
//! [Parcae: Scaling Laws For Stable Looped Language Models](https://arxiv.org/abs/2604.12946)
//! (Prairie, Novack, Berg-Kirkpatrick, Fu).
//!
//! Parcae recasts looping as a nonlinear time-variant dynamical system over
//! the residual stream:
//!
//! ```text
//! h_{t+1} = Ā·h_t + B̄·e + R̄(h_t, e)
//! ```
//!
//! where `Ā` is state retention, `B̄` input injection and `R̄` the transformer
//! nonlinearities. Instability comes from large spectral norms of the
//! injection parameters. Parcae constrains `Ā` by discretizing a continuous-time
//! negative-diagonal parameterization:
//!
//! ```text
//! A  := diag(-exp(a))            a ∈ R^d, learnable, per-channel
//! Ā  := exp(Δ·A)                 Δ stored raw, read as |δ| + 1e-8
//! ```
//!
//! `A` has strictly negative diagonal and `Δ` is made positive at read time
//! (`|δ| + 1e-8`), so `Δ·A` has strictly negative entries and `exp(Δ·A)` has
//! all eigenvalues in `[0, 1)` — guaranteed contraction, bounded residual
//! dynamics for ANY loop count `T`, by construction for ANY optimizer (no
//! clipping, no post-hoc normalization, no constraint on the raw `delta`
//! param).
//!
//! `B̄ = Δ·B` with unconstrained `B` (LayerNorm applied to the input keeps
//! `e` bounded).
//!
//! # Usage
//!
//! ```rust,ignore
//! use burn::backend::NdArray;
//! use burn::tensor::{Device, Distribution, Tensor};
//! use burn_parcae::SpectralRetention;
//!
//! type Backend = NdArray<f32>;
//!
//! let retention = SpectralRetention::<Backend>::new(64, &Default::default());
//! let h = Tensor::<Backend, 3>::random([2, 32, 64], Distribution::Default, &Default::default());
//! let e = Tensor::<Backend, 3>::random([2, 32, 64], Distribution::Default, &Default::default());
//!
//! for _ in 0..10 {
//!     // h stays bounded for ANY number of loop iterations (contraction by construction).
//!     let h = retention.forward(h, e.clone());
//! }
//! ```

use burn::module::{Module, Param};
use burn::tensor::{Device, Tensor};

/// Parcae spectral retention configuration.
///
/// The paper uses a full `d×d` injection matrix `B`; the parameter-efficient
/// variant keeps `B` diagonal (`d` params instead of `d²`). Both compute the
/// same map `B̄ = Δ·B` and guarantee the same retention bounds.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SpectralRetentionConfig {
    /// Paper's full `d×d` injection matrix `B` (`false` = diagonal variant).
    pub full_b: bool,
}

/// Discretized retention matrix from the Parcae parameterization.
///
/// `A := diag(-exp(a))`, `Ā := exp(Δ·A)`; the step size is made positive at
/// read time (`Δ := |δ| + 1e-8`), so every entry of `Ā` is
/// `exp(-Δ_i · exp(a_i)) ∈ [0, 1)` — all eigenvalues in `[0, 1)` by
/// construction, for ANY optimizer (SGD can push a raw `delta` param ≤ 0;
/// `abs + eps` keeps the contraction guarantee with gradient flow intact).
pub fn retention_matrix(a: Tensor<1>, delta: Tensor<1>) -> Tensor<1> {
    delta.abs().add_scalar(1e-8).mul(a.exp()).neg().exp()
}

/// Parcae spectral retention: stable looping state update `h ← Ā⊙h + B̄·e`.
///
/// # Invariants
/// - `retention()` is elementwise in `[0, 1)` for any `a`, any raw `delta` —
///   `Δ` is read as `|δ| + 1e-8` (never ≤ 0), so the residual stream is
///   contractive for any loop count `T` by construction, no clipping or
///   post-hoc normalization.
/// - `B̄ = Δ·B` (diag-Δ scaling of the unconstrained injection `B`); the input
///   is expected to be LayerNorm'd, keeping the injection term bounded.
#[derive(Module, Debug)]
pub struct SpectralRetention {
    /// Log-rate parameter `a` (zeros init → `Ā = exp(-1·Δ)` contraction).
    pub a: Param<Tensor<1>>,
    /// Raw per-channel step size (ones init), unconstrained in memory — read
    /// as `|δ| + 1e-8` so `Δ > 0` holds for any optimizer.
    pub delta: Param<Tensor<1>>,
    /// Diagonal injection `B` (`[d]`) — parameter-efficient variant of the
    /// paper's full matrix (paper uses full `B`; see `SpectralRetentionConfig::full_b`).
    pub b_diag: Option<Param<Tensor<1>>>,
    /// Full `d×d` injection `B` (paper) — used when `full_b = true`.
    pub b_full: Option<Param<Tensor<2>>>,
    #[module(skip)]
    pub full_b: bool,
    #[module(skip)]
    pub d_model: usize,
}

impl SpectralRetention {
    /// Diagonal variant with defaults (`full_b = false`).
    pub fn new(d_model: usize, device: &Device) -> Self {
        Self::from_config(&SpectralRetentionConfig::default(), d_model, device)
    }

    /// Builds from a [`SpectralRetentionConfig`].
    pub fn from_config(cfg: &SpectralRetentionConfig, d_model: usize, device: &Device) -> Self {
        let (b_diag, b_full) = if cfg.full_b {
            (None, Some(Param::from_tensor(Tensor::eye(d_model, device))))
        } else {
            (
                Some(Param::from_tensor(Tensor::ones([d_model], device))),
                None,
            )
        };
        Self {
            a: Param::from_tensor(Tensor::zeros([d_model], device)),
            delta: Param::from_tensor(Tensor::ones([d_model], device)),
            b_diag,
            b_full,
            full_b: cfg.full_b,
            d_model,
        }
    }

    /// Positive, epsilon-floored per-channel step size `Δ := |δ| + 1e-8`.
    ///
    /// Applied at READ time so the `Δ > 0` invariant holds for any optimizer:
    /// a raw `delta` that SGD pushed to 0 or negative is re-positivized with
    /// gradient flow intact (`d|x|/dx = ±1`, never 0).
    fn delta_pos(&self) -> Tensor<1> {
        self.delta.val().abs().add_scalar(1e-8)
    }

    /// Retention diagonal `Ā = exp(-Δ·exp(a))` as `[d]` — every entry in
    /// `[0, 1)` by construction (Δ read as `|δ| + 1e-8` > 0, exp > 0).
    pub fn retention(&self) -> Tensor<1> {
        retention_matrix(self.a.val(), self.delta.val())
    }

    /// Injection scaling `B̄ = Δ·B`: `[d]` in the diagonal variant, the full
    /// `d×d` matrix in the paper variant.
    pub fn inject_scale(&self) -> Tensor<2> {
        let delta = self.delta_pos();
        match &self.b_diag {
            Some(b) => b.val().mul(delta).reshape([1, self.d_model]),
            None => {
                let b = self.b_full.as_ref().expect("b_full set");
                b.val().mul(delta.reshape([self.d_model, 1]))
            }
        }
    }

    /// Input injection `B̄·e` (`[B, T, D] → [B, T, D]`).
    pub fn inject(&self, e: Tensor<3>) -> Tensor<3> {
        let delta = self.delta_pos();
        match &self.b_diag {
            Some(b) => {
                let scale = b.val().mul(delta).reshape([1, 1, self.d_model]);
                e.mul(scale)
            }
            None => {
                let b = self.b_full.as_ref().expect("b_full set");
                let scaled = b.val().transpose().mul(delta.reshape([1, self.d_model]));
                e.matmul(scaled.unsqueeze())
            }
        }
    }

    /// One Parcae step: `h ← Ā⊙h + B̄·e` (`[B, T, D] → [B, T, D]`).
    pub fn forward(&self, h: Tensor<3>, e: Tensor<3>) -> Tensor<3> {
        let ret = self.retention().reshape([1, 1, self.d_model]);
        h.mul(ret).add(self.inject(e))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    fn dev() -> Device {
        Device::ndarray()
    }

    fn to_vec(t: Tensor<1>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn retention_in_unit_interval() {
        // Bounded draws: a ∈ [-3, 3] and delta ∈ [0.5, 2] give the exponent
        // delta·exp(a) ∈ [0.025, 40.2], so exp(-x) ∈ (0.975, 2.8e-18) for ANY
        // draw — never the f32 underflow (x > 103 → exactly 0.0) that
        // unbounded Normal(0, 2) a produced (~1/3 of runs, the old flake),
        // never a round-to-1.0 from delta ≈ 0.
        let d = 64;
        let a = Tensor::<1>::random([d], Distribution::Uniform(-3.0, 3.0), &dev());
        let delta = Tensor::<1>::random([d], Distribution::Uniform(0.5, 2.0), &dev());
        let r = to_vec(retention_matrix(a, delta));
        assert!(
            r.iter().all(|&v| v > 0.0 && v < 1.0),
            "retention out of (0,1): sample {:?}",
            &r[..8]
        );
    }

    #[test]
    fn retention_in_unit_interval_extreme() {
        // a=100, delta=-100: a raw delta pushed negative by SGD plus an
        // overflowing exp(a). The read-time clamp (|delta| + 1e-8) must still
        // keep retention in [0, 1) — never NaN, never >= 1. exp(-100·e^100)
        // underflows to exactly 0.0 in f32, which is still contractive.
        let a = Tensor::<1>::from_floats([100.0, 0.0], &dev());
        let delta = Tensor::<1>::from_floats([-100.0, -100.0], &dev());
        let r = to_vec(retention_matrix(a, delta));
        assert!(
            r.iter().all(|&v| v.is_finite() && (0.0..1.0).contains(&v)),
            "retention out of [0,1): sample {:?}",
            r
        );
        // Same negative delta with a = 0 gives strictly positive retention
        // (exp(-100.00000001) ≈ 3.7e-44), so the clamp doesn't deaden the map.
        assert!(r[1] > 0.0, "retention must stay positive for moderate a");
    }

    #[test]
    fn retention_decreases_with_a() {
        // Larger a -> smaller retention (monotonic in a), fixed positive delta.
        let d = 64;
        let a1 = Tensor::<1>::random([d], Distribution::Default, &dev());
        let a2 = a1.clone().add_scalar(1.0);
        let delta = Tensor::<1>::ones([d], &dev());
        let r1 = to_vec(retention_matrix(a1, delta.clone()));
        let r2 = to_vec(retention_matrix(a2, delta));
        assert!(
            r1.iter().zip(&r2).all(|(&x, &y)| x > y),
            "retention must decrease with a"
        );
    }

    #[test]
    fn forward_contracts() {
        // e = 0: after 50 steps the residual must not grow (contractive Ā).
        let d = 32;
        let sr = SpectralRetention::new(d, &dev());
        let mut h = Tensor::<3>::random([1, 4, d], Distribution::Default, &dev());
        let e = Tensor::<3>::zeros([1, 4, d], &dev());
        let norm = |t: &Tensor<3>| {
            let v: f32 = t.clone().powf_scalar(2.0).sum().into_scalar();
            v.sqrt()
        };
        let first = norm(&h);
        for _ in 0..50 {
            h = sr.forward(h, e.clone());
        }
        let last = norm(&h);
        assert!(last <= first, "residual grew: first {first} -> last {last}");
    }

    #[test]
    fn forward_shape() {
        let sr = SpectralRetention::new(64, &dev());
        let h = Tensor::<3>::random([2, 8, 64], Distribution::Default, &dev());
        let e = Tensor::<3>::random([2, 8, 64], Distribution::Default, &dev());
        assert_eq!(sr.forward(h, e).dims(), [2, 8, 64]);
    }

    #[test]
    fn injection_scale_shape() {
        let sr = SpectralRetention::new(64, &dev());
        assert_eq!(sr.inject_scale().dims(), [1, 64]);
    }

    #[test]
    fn default_init_is_contractive() {
        // a = 0, delta = 1 -> Ā = exp(-1) ≈ 0.3679 everywhere.
        let sr = SpectralRetention::new(16, &dev());
        let r = to_vec(sr.retention());
        let expected = (-1.0f32).exp();
        assert!(
            r.iter().all(|&v| (v - expected).abs() < 1e-6),
            "default retention must be e^-1, sample {:?}",
            &r[..4]
        );
    }

    #[test]
    fn full_b_matches_diagonal() {
        // Identity-init full B == ones-init diagonal B: identical inject.
        let cfg = SpectralRetentionConfig { full_b: true };
        let full = SpectralRetention::from_config(&cfg, 32, &dev());
        let diag = SpectralRetention::new(32, &dev());
        let e = Tensor::<3>::random([1, 4, 32], Distribution::Default, &dev());
        let f: f32 = (full.inject(e.clone()) - diag.inject(e))
            .powf_scalar(2.0)
            .mean()
            .into_scalar();
        assert!(
            f < 1e-6,
            "full B (identity) vs diagonal (ones) mismatch {f}"
        );
    }
}
