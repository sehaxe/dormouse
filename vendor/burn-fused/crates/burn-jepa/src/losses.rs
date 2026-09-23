//! JEPA objectives: masked L1, LeJEPA SIGReg (direct form), KoLeo uniformity.

use burn::tensor::activation::softmax;
use burn::tensor::{Bool, Int, Tensor};

/// Masked L1 loss between predicted and teacher latents.
///
/// `|pred - target|` averaged over masked elements only (mask `[B, T]`,
/// expanded over the `D` dim). An all-false mask yields `0.0` (the count is
/// `clamp_min(1)`ed so the division never produces NaN).
pub fn jepa_l1_loss(pred: Tensor<3>, target: Tensor<3>, mask: Tensor<2, Bool>) -> Tensor<1> {
    let [b, t, d] = pred.dims();
    let count = mask.clone().float().sum().clamp_min(1.0);
    let mf: Tensor<3> = mask.unsqueeze_dim::<3>(2).expand::<3, _>([b, t, d]).float();
    (pred - target)
        .abs()
        .mul(mf)
        .sum()
        .div(count.mul_scalar(d as f32))
}

/// LeJEPA: isotropic Gaussian regularization (arXiv 2511.08544,
/// Balestriero & LeCun 2025 — verified).
///
/// Constrains embeddings toward an isotropic Gaussian N(0, I). This is the
/// direct (non-sketched) form of the paper's SIGReg objective; the paper's
/// sketching is a scalability device and does not change the fixed point.
/// Single trade-off hyperparameter: weight of this loss against MSE.
///
/// ```text
/// L = ∥mean(z)∥² + ∥cov(z) - I∥² / D
/// ```
pub fn lejepa_loss(z: Tensor<2>) -> Tensor<1> {
    let [n, d] = z.dims();
    let dev = z.device();
    let mean = z.clone().mean_dim(0);
    let mean_loss = mean.clone().powf_scalar(2.0).sum();
    // `mean` is [1, d] (mean_dim keeps rank in burn 0.21); expand explicitly
    // to [n, d] for the broadcast subtraction (`.unsqueeze()` was a
    // rank-preserving no-op here).
    let z_c = z.clone() - mean.expand([n, d]);
    let cov = z_c
        .clone()
        .transpose()
        .matmul(z_c)
        .div_scalar((n.max(2) - 1) as f32);
    let eye = Tensor::eye(d, &dev);
    let cov_loss = (cov - eye).powf_scalar(2.0).sum().div_scalar(d as f32);
    (mean_loss + cov_loss).unsqueeze()
}

/// KoLeo: uniformity regularizer (Caron et al., DINOv2, 2023).
///
/// Encourages embeddings to spread uniformly on the unit sphere.
/// Subsamples to ~256 tokens for O(n²) efficiency. Subsample is uniform
/// by stride, NOT norm-topk: burn-cuda 0.21 has no GPU sort, so
/// `topk_with_indices` falls back to a host read that fails inside
/// autodiff (and its panic poisons the CUDA context -> illegal address).
///
/// ```text
/// L = -mean(log(nn_dist))    nn_dist = soft-min of pairwise squared dists (τ=0.25)
/// ```
pub fn koleo_loss(z: Tensor<2>) -> Tensor<1> {
    let [n, d] = z.dims();
    let dev = z.device();
    let m = 256usize.min(n);
    let (z_sub, m) = if m < n {
        // Strided pick: 0, step, 2*step, ...; materialized repeat-indices +
        // gather-by-dim (the MSA-safe pattern; gather_nd is buggy on CUDA).
        let step = n.div_ceil(m);
        let count = n.div_ceil(step);
        let idx = Tensor::<1, Int>::arange(0..count as i64, &dev).mul_scalar(step as i64);
        let idx_2d = idx.unsqueeze_dim::<2>(1).repeat_dim(1, d);
        (z.gather(0, idx_2d), count)
    } else {
        (z, m)
    };
    let z_sq = z_sub.clone().powf_scalar(2.0).sum_dim(1).clamp_min(1e-12);
    let z_norm = z_sub / z_sq.sqrt();
    let dots = z_norm.clone().matmul(z_norm.transpose());
    // Squared distance on the unit sphere: 2-2*dot. Clamp BEFORE sqrt:
    // float rounding can push it slightly negative for near-identical rows
    // -> sqrt(NaN) -> NaN loss -> NaN grads. sqrt afterwards keeps the
    // Euclidean distance of the DINOv2 reference.
    let dists = dots.neg().mul_scalar(2.0).add_scalar(2.0).clamp_min(0.0)
        + Tensor::eye(m, &dev).mul_scalar(1e6);
    // Soft-min surrogate for the row-min: cubek-reduce 0.2's ArgMin/ArgMax
    // (coordinate) reduce triggers a latent OOB on burn-cuda 0.21 inside
    // autodiff (CUDA_ERROR_ILLEGAL_ADDRESS, flaky, in-training only —
    // verified: arg-reduce variants crash, soft-min is clean). τ→0 recovers
    // DINOv2 KoLeo exactly; the backward is elementwise + sum (no arg-reduce,
    // no scatter). The eye keeps self-pairs out (weight ~ e^-1e6).
    let tau = 0.25f32;
    let w = softmax(dists.clone().mul_scalar(-1.0 / tau), 1);
    let nn_sq = dists.mul(w).sum_dim(1).clamp_min(1e-12);
    let nn_dists = nn_sq.sqrt().clamp_min(1e-8);
    nn_dists.log().neg().mean().unsqueeze()
}
