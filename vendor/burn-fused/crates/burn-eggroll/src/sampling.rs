//! Gaussian factor sampling for rank-r perturbations.

use burn::tensor::{Device, Distribution, Tensor};

use crate::update::perturb;

/// Gaussian left factor `A ~ N(0,1)` of shape `[m, r]`.
///
/// `seed` is part of the API for deterministic noise regeneration
/// (paper §2: store seeds, regenerate A/B on demand). Burn 0.21's
/// `Tensor::random` has no seed parameter, so the seed is currently
/// accepted but unused — pending a seeded backend.
pub fn sample_a(m: usize, r: usize, _seed: u64, device: &Device) -> Tensor<2> {
    Tensor::random([m, r], Distribution::Normal(0.0, 1.0), device)
}

/// Gaussian right factor `B ~ N(0,1)` of shape `[n, r]`.
pub fn sample_b(n: usize, r: usize, _seed: u64, device: &Device) -> Tensor<2> {
    Tensor::random([n, r], Distribution::Normal(0.0, 1.0), device)
}

/// Convenience: fresh random A, B, return perturbed `W = M + σE`.
pub fn eggroll_mutate(w: &Tensor<2>, rank: usize, sigma: f64, device: &Device) -> Tensor<2> {
    assert!(rank > 0, "rank must be >= 1");
    let [rows, cols] = w.dims();
    let a = Tensor::<2>::random([rows, rank], Distribution::Normal(0.0, 1.0), device);
    let b = Tensor::<2>::random([cols, rank], Distribution::Normal(0.0, 1.0), device);
    perturb(w, &a, &b, sigma)
}
