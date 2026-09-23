//! # burn-rope - Rotary Position Embedding for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! | Feature | Reference | What |
//! |---------|-----------|------|
//! | `RoPE` | Su et al. 2021 | Rotary position encoding via complex rotation |
//! | `YaRN` | Peng et al. 2023 | Context-length extrapolation with frequency scaling |
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss, clippy::many_single_char_names)]

#[cfg(any(feature = "cuda", feature = "autodiff"))]
pub mod rope_cuda;

/// Precompute plain RoPE cos/sin frequency tables `[max_seq_len, head_dim/2]`.
mod freqs;
mod rotate;

pub use freqs::{precompute_freqs, precompute_freqs_yarn};
pub use rotate::{apply_rope_3d, apply_rope_4d, RotaryEmbedding};

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor};
    use burn_ndarray::NdArray;
    fn dev() -> Device {
        Device::ndarray()
    }

    fn to_vec(x: Tensor<2>) -> Vec<f32> {
        x.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn freqs_identity_at_position_zero() {
        let (cos, _sin) = precompute_freqs(64, 256, 10000.0, &dev());
        let half = 32usize;
        let v = to_vec(cos);
        for i in 0..half {
            assert!(
                (v[i] - 1.0).abs() < 1e-5,
                "cos[0,{i}] should be 1, got {}",
                v[i]
            );
        }
    }

    #[test]
    fn freqs_in_range() {
        let (cos, sin) = precompute_freqs(64, 256, 10000.0, &dev());
        for t in [cos, sin] {
            for v in to_vec(t) {
                assert!((-1.0..=1.0).contains(&v), "value {v} outside [-1,1]");
            }
        }
    }

    #[test]
    fn rope_preserves_shape() {
        let x = Tensor::<3>::random([2, 16, 64], burn::tensor::Distribution::Default, &dev());
        let (cos, sin) = precompute_freqs(16, 32, 10000.0, &dev());
        let y = apply_rope_3d::<NdArray>(x, cos, sin, 4);
        assert_eq!(y.dims(), [2, 16, 64]);
    }

    #[test]
    fn rope_pos_zero_is_identity() {
        let x = Tensor::<3>::ones([1, 4, 16], &dev());
        let (cos, sin) = precompute_freqs(4, 4, 10000.0, &dev());
        let y = apply_rope_3d::<NdArray>(x.clone(), cos, sin, 4);
        let d: Vec<f32> = (x.slice([0..1, 0..1]) - y.slice([0..1, 0..1]))
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        for &v in &d {
            assert!(v.abs() < 1e-4, "RoPE at pos 0 should be identity");
        }
    }

    #[test]
    fn yarn_scale1_equals_plain_rope() {
        let (c1, s1) = precompute_freqs(16, 64, 10000.0, &dev());
        let (cy, sy) = precompute_freqs_yarn(16, 64, 10000.0, 1.0, 4096, 32.0, 1.0, &dev());
        let d: Vec<f32> = (c1.clone() - cy.clone())
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(
            d.iter().all(|&v| v.abs() < 1e-5),
            "YaRN scale=1 must reduce to plain RoPE"
        );
        let d2: Vec<f32> = (s1 - sy)
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(d2.iter().all(|&v| v.abs() < 1e-5));
    }

    #[test]
    fn yarn_ramp_matches_documented_formula() {
        // Pins the implementation to the documented YaRN ramp:
        //   r(d) = L·θ_d/(2π), γ = clamp((r−βs)/(βf−βs), 0, 1),
        //   f = ((1−γ)·θ/s + γ·θ)·temp,  temp = 0.1·ln(s)+1.
        //
        // The reference is computed here in f64 from those lines and compared
        // in the COS domain (wrap-safe, no acos∘cos round-trip: below ~3e-4
        // rad an f32 cos collapses to 1.0, which once let an inverted ramp
        // ship with green tests). Parameters are chosen so the table
        // exercises all three regimes: dim 0 lands mid-ramp, dim 1 fully
        // interpolated. An inverted r (= L/(2πθ)) would flip dim 1 to
        // "keep" and fail this comparison massively.
        let (base, s, orig_len, beta_fast, beta_slow) =
            (1000.0f64, 16.0f64, 64.0f64, 32.0f64, 1.0f64);
        let temp = 0.1 * s.ln() + 1.0;
        let (head_dim, max_seq) = (4usize, 1024usize);

        let (cy, sy) = precompute_freqs_yarn(
            head_dim,
            max_seq,
            base,
            s,
            orig_len as usize,
            beta_fast,
            beta_slow,
            &ndarray_dev(),
        );
        let got_c = to_vec(cy);
        let got_s = to_vec(sy);

        for i in 0..(head_dim / 2) {
            let theta = base.powf(-2.0 * i as f64 / head_dim as f64);
            let r = orig_len * theta / (2.0 * std::f64::consts::PI);
            let gamma = ((r - beta_slow) / (beta_fast - beta_slow)).clamp(0.0, 1.0);
            let want_f = ((1.0 - gamma) * theta / s + gamma * theta) * temp;
            // Positions stay ≤ 64 so the table angle stays ≤ ~28 rad: an f32
            // angle there quantizes to ~2e-6, inside the tolerance below.
            // (Large positions are wrap-safe to COMPARE but their f32 angle
            // storage alone injects ~3e-5 of noise.) The discriminating
            // signal is dim 1, where the inverted ramp differs by ~2.6 rad.
            for pos in [1usize, 17, 64] {
                let want_c = (want_f * pos as f64).cos() as f32;
                let want_s = (want_f * pos as f64).sin() as f32;
                let gc = got_c[pos * (head_dim / 2) + i];
                let gs = got_s[pos * (head_dim / 2) + i];
                assert!(
                    (gc - want_c).abs() < 1e-5,
                    "dim {i} pos {pos}: cos {gc} vs {want_c} (γ={gamma})"
                );
                assert!(
                    (gs - want_s).abs() < 1e-5,
                    "dim {i} pos {pos}: sin {gs} vs {want_s} (γ={gamma})"
                );
            }
        }
    }

    /// The precision reference above needs correctly-rounded cos/sin: the
    /// ambient default backend may dispatch to a kernel whose transcendentals
    /// are only ~1e-4-accurate, so pin this test to ndarray.
    fn ndarray_dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn apply_rope_4d_matches_3d() {
        let x = Tensor::<3>::random([2, 8, 32], burn::tensor::Distribution::Default, &dev());
        let (cos, sin) = precompute_freqs(8, 16, 10000.0, &dev());
        let y3 = apply_rope_3d::<NdArray>(x.clone(), cos.clone(), sin.clone(), 4);
        let x4 = x.reshape([2, 8, 4, 8]);
        let y4 = apply_rope_4d::<NdArray>(x4, cos, sin).reshape([2, 8, 32]);
        let d: Vec<f32> = (y3 - y4)
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(d.iter().all(|&v| v.abs() < 1e-4), "3d and 4d should match");
    }

    #[test]
    fn module_forward_works() {
        let rope = RotaryEmbedding::new(64, 4, 32, 10000.0, &dev());
        let x = Tensor::<3>::random([1, 8, 64], burn::tensor::Distribution::Default, &dev());
        let y = rope.forward::<NdArray>(x);
        assert_eq!(y.dims(), [1, 8, 64]);
    }

    #[test]
    fn module_qk_works() {
        let rope = RotaryEmbedding::new(32, 2, 16, 10000.0, &dev());
        let q = Tensor::<3>::ones([1, 4, 32], &dev());
        let k = Tensor::<3>::ones([1, 4, 32], &dev());
        let (q_r, k_r) = rope.forward_qk::<NdArray>(q, k);
        assert_eq!(q_r.dims(), [1, 4, 32]);
        assert_eq!(k_r.dims(), [1, 4, 32]);
    }
}
