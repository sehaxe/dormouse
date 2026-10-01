//! # dormouse-eggroll — EGGROLL low-rank evolutionary strategies
//!
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//! Rank-1 evolutionary-strategy perturbations for non-differentiable
//! decisions (e.g. top-k routing in LLMs), per
//! [arXiv:2511.16652](https://arxiv.org/abs/2511.16652)
//! "Evolution Strategies at the Hyperscale".
//!
//! Instead of backprop: perturb a weight matrix `M ∈ R^{m×n}` with rank-1
//! Gaussian noise `E = (1/√r)·A·Bᵀ`, `A ~ N(0,1) [m,r]`, `B ~ N(0,1) [n,r]`,
//! evaluate the model on `M + σE` and `M − σE` (antithetic pair), and update
//! `M ← M + (α/√r)·Σᵢ Eᵢ·fᵢ` where `fᵢ = sign(sᵢ⁺ − sᵢ⁻) ∈ {−1,0,1}`
//! (the `1/N` population average is the caller's job — `α` already includes it).
//!
//! The paper shows rank 1 is as effective as full-rank noise while being
//! ~100× cheaper in random numbers and storage. Updates never materialize E:
//! `Σᵢ fᵢ·Aᵢ·Bᵢᵀ` is computed as `(f·A)ᵀ·B` batched matmul.
//!
//! All tensor code is pure `B: Backend` ops — zero host branching except the
//! inherently scalar ES fitness function [`antithetic_sign`].
mod sampling;
mod update;

pub use sampling::{eggroll_mutate, sample_a, sample_b};
pub use update::{antithetic_sign, perturb, update, update_batched, EggrollConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Distribution, Tensor};

    fn dev() -> Device {
        Device::ndarray()
    }

    fn to_vec(t: Tensor<2>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    #[test]
    fn sample_a_shape_and_finite() {
        let a = sample_a(8, 1, 42, &dev());
        assert_eq!(a.dims(), [8, 1]);
        assert!(to_vec(a).iter().all(|v| v.is_finite()));
        let b = sample_b(6, 3, 42, &dev());
        assert_eq!(b.dims(), [6, 3]);
        assert!(to_vec(b).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn perturb_scales_with_sigma() {
        // ‖perturb(M,σ) − M‖ must grow ~linearly with σ (same A, B).
        let m = Tensor::<2>::zeros([16, 16], &dev());
        let a = sample_a(16, 1, 1, &dev());
        let b = sample_b(16, 1, 2, &dev());
        let d1 = perturb(&m, &a, &b, 1e-3).sub(m.clone());
        let d2 = perturb(&m, &a, &b, 2e-3).sub(m.clone());
        let v1 = to_vec(d1);
        let v2 = to_vec(d2);
        let mean1 = v1.iter().map(|x| x * x).sum::<f32>() / v1.len() as f32;
        let mean2 = v2.iter().map(|x| x * x).sum::<f32>() / v2.len() as f32;
        let ratio = (mean2 / mean1).sqrt();
        assert!(
            (ratio - 2.0).abs() < 0.05,
            "double sigma -> double perturbation norm, got ratio {ratio}"
        );
    }

    #[test]
    fn perturb_rank1_variance_bounded() {
        // Deterministic A/B, no RNG: provably non-flaky. r=1 => E = σ·A·Bᵀ.
        let sigma = 0.5f64;
        let m = Tensor::<2>::zeros([64, 64], &dev());
        let one = Tensor::<2>::ones([64, 1], &dev());

        // A = B = 1: E ≡ σ constant -> every entry exactly σ, std exactly 0.
        let e = to_vec(perturb(&m, &one, &one, sigma));
        assert!(e.iter().all(|v| (v - sigma as f32).abs() < 1e-6));

        // A = 1, B = [1,-1,1,-1,...]: E alternates ±σ -> std exactly σ.
        let alt: Vec<f32> = (0..64)
            .map(|j| if j % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let b = Tensor::<1>::from_floats(alt.as_slice(), &dev()).reshape([64, 1]);
        let e = to_vec(perturb(&m, &one, &b, sigma));
        let mean = e.iter().sum::<f32>() / e.len() as f32;
        let var = e.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / e.len() as f32;
        let std = var.sqrt() as f64;
        assert!(
            (std - sigma).abs() < 1e-6,
            "rank-1 alternating std should be exactly sigma={sigma}, got {std}"
        );
    }

    #[test]
    fn antithetic_sign_ternary() {
        assert_eq!(antithetic_sign(0.9, 0.1), 1.0);
        assert_eq!(antithetic_sign(0.1, 0.9), -1.0);
        assert_eq!(antithetic_sign(0.5, 0.5), 0.0);
        assert_eq!(antithetic_sign(-1.0, -1.0), 0.0);
    }

    #[test]
    fn update_moves_toward_fitness() {
        // Dot(ΔM, A·Bᵀ) > 0 for positive fitness, < 0 for negative.
        let m = Tensor::<2>::zeros([8, 8], &dev());
        let a = sample_a(8, 1, 9, &dev());
        let b = sample_b(8, 1, 11, &dev());
        let e = a.clone().matmul(b.clone().transpose());
        let up = update(&m, &a, &b, 1.0, 0.1, 1);
        let down = update(&m, &a, &b, -1.0, 0.1, 1);
        let ev = to_vec(e);
        let uv = to_vec(up);
        let dv = to_vec(down);
        let dot_plus: f32 = ev.iter().zip(uv.iter()).map(|(e, u)| e * u).sum();
        let dot_minus: f32 = ev.iter().zip(dv.iter()).map(|(e, d)| e * d).sum();
        assert!(
            dot_plus > 0.0,
            "positive fitness must move along E: {dot_plus}"
        );
        assert!(
            dot_minus < 0.0,
            "negative fitness must move against E: {dot_minus}"
        );
    }

    #[test]
    fn update_batched_equals_loop() {
        let m = Tensor::<2>::zeros([8, 6], &dev());
        let n = 5;
        let a = Tensor::<3>::random([n, 8, 1], Distribution::Normal(0.0, 1.0), &dev());
        let b = Tensor::<3>::random([n, 6, 1], Distribution::Normal(0.0, 1.0), &dev());
        let f = Tensor::<1>::from_floats([0.7f32, -0.3, 0.0, 0.2, 0.9].as_slice(), &dev());

        let batched = update_batched(&m, &a, &b, &f, 0.1, 1);

        let mut looped = m.clone();
        let av: Vec<Tensor<2>> = (0..n)
            .map(|i| a.clone().narrow(0, i, 1).reshape([8, 1]))
            .collect();
        let bv: Vec<Tensor<2>> = (0..n)
            .map(|i| b.clone().narrow(0, i, 1).reshape([6, 1]))
            .collect();
        let fv: Vec<f32> = f.clone().into_data().try_to_vec::<f32>().unwrap();
        for i in 0..n {
            looped = update(&looped, &av[i], &bv[i], fv[i], 0.1, 1);
        }

        let diff = batched.sub(looped);
        let max_abs = to_vec(diff).iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-5,
            "batched update diverges from per-member loop: {max_abs}"
        );
    }

    #[test]
    fn eggroll_mutate_roundtrip() {
        let w = Tensor::<2>::ones([4, 3], &dev());
        let p = eggroll_mutate(&w, 2, 0.01, &dev());
        assert_eq!(p.dims(), [4, 3]);
        assert!(to_vec(p).iter().all(|v| v.is_finite()));

        // Deterministic roundtrip: A = B = 1, r = 2 ->
        // E = (σ/√2)·(1·1 + 1·1) = σ√2 exactly.
        let a = Tensor::<2>::ones([4, 2], &dev());
        let b = Tensor::<2>::ones([3, 2], &dev());
        let p = to_vec(perturb(&w, &a, &b, 0.01));
        let expected = 1.0f32 + 0.01f32 * std::f32::consts::SQRT_2;
        assert!(
            p.iter().all(|v| (v - expected).abs() < 1e-6),
            "deterministic perturb should land exactly on {expected}"
        );
    }
}
