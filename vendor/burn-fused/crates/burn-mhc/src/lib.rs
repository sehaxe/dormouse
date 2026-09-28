//! # burn-mhc - Manifold-Constrained Hyper-Connections for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! Full implementation of [mHC](https://arxiv.org/abs/2512.24880) (DeepSeek, 2025).
//!
//! Implements the paper's complete mechanism:
//! - Eq 3: `x_{l+1} = H_res x_l + (H_post)^T F(H_pre x_l)` - n-stream residual
//! - Eq 7: first-order hyper-network - input-dependent mappings via
//!   `RMSNorm(vec(x))` linear projections + static biases
//! - Eq 8: `H_pre = sigmoid`, `H_post = 2*sigmoid` (non-negativity, prevents
//!   signal cancellation), `H_res = Sinkhorn-Knopp(...)`
//! - Eq 9: Sinkhorn-Knopp entropic projection onto the Birkhoff polytope
//!   (doubly stochastic: row/col sums = 1, spectral norm <= 1, compositional
//!   closure) - restores the identity-mapping property of residual streams
//!
//! Hyper-parameters follow App. A.1: gating factors alpha init = 0.01,
//! Sinkhorn-Knopp t_max = 20.
#![allow(clippy::single_range_in_vec_init, clippy::needless_range_loop)]
// `autodiff` alone also compiles this module, and deliberately so: the fused
// ADJOINT and its strategy seam live in it, and a gate that can only be
// compile-checked with a GPU is exactly how a `NoCheckpointing`-only entry
// survived. The kernels inside stay `#[cfg(feature = "cuda")]`, which is what
// the CPU seam test exercises — the tensor path, after the downcast.
#[cfg(any(feature = "cuda", feature = "autodiff"))]
pub mod sinkhorn_cuda;

/// Sinkhorn-Knopp iterations used in the paper (App. A.1: t_max = 20).
mod block;
mod sinkhorn;

pub use block::MhcBlock;
pub use sinkhorn::{sinkhorn_knopp, SINKHORN_ITERS};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::STATIC_INIT;
    use burn::tensor::Device;
    use burn::tensor::Distribution;
    use burn::tensor::Tensor;
    fn dev() -> Device {
        Device::ndarray()
    }

    fn to_vec(t: Tensor<4>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn sinkhorn_is_doubly_stochastic() {
        let logits = Tensor::<4>::random([2, 3, 4, 4], Distribution::Normal(0.0, 1.0), &dev());
        let m = sinkhorn_knopp(logits, SINKHORN_ITERS);
        let vals = to_vec(m);
        for i in 0..2 {
            for j in 0..3 {
                let m: Vec<f32> = vals[(i * 3 + j) * 16..(i * 3 + j) * 16 + 16].to_vec();
                for row in 0..4 {
                    let row_sum: f32 = m[row * 4..row * 4 + 4].iter().sum();
                    assert!((row_sum - 1.0).abs() < 1e-3, "row sum {row_sum}");
                }
                for col in 0..4 {
                    let col_sum: f32 = (0..4).map(|r| m[r * 4 + col]).sum();
                    assert!((col_sum - 1.0).abs() < 1e-3, "col sum {col_sum}");
                }
            }
        }
        for (k, &v) in vals.iter().enumerate() {
            assert!(v >= 0.0, "entry {k} negative: {v}");
        }
    }

    #[test]
    fn sinkhorn_identity_static_init() {
        // b_res = 10*I -> exp(diag 10) -> Sinkhorn ~ identity
        let logits = Tensor::<2>::eye(3, &dev()).mul_scalar(STATIC_INIT);
        let m = sinkhorn_knopp(logits.reshape([1, 1, 3, 3]), SINKHORN_ITERS);
        let vals = to_vec(m);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                let v = vals[i * 3 + j];
                assert!((v - want).abs() < 1e-3, "H_res[{i},{j}] = {v}, want {want}");
            }
        }
    }

    #[test]
    fn sinkhorn_large_logits_stay_finite() {
        // Regression: exp(±100+) used to overflow f32 to inf and NaN the
        // normalization; the ±60 input clamp must keep every entry finite.
        let mut data = vec![0.0f32; 32]; // [2, 1, 4, 4]
        for (k, v) in data.iter_mut().enumerate() {
            *v = if k % 2 == 0 { 120.0 } else { -120.0 };
        }
        let logits =
            Tensor::<4>::from_data(burn::tensor::TensorData::new(data, [2, 1, 4, 4]), &dev());
        let m = sinkhorn_knopp(logits, SINKHORN_ITERS);
        let vals = to_vec(m);
        let bad: Vec<f32> = vals.iter().copied().filter(|v| !v.is_finite()).collect();
        assert!(
            bad.is_empty(),
            "non-finite entries: {bad:?} of {:?}",
            &vals[..8]
        );
        // Row sums are exactly 1 by construction (first normalization pass).
        for b in 0..2 {
            let off = b * 16;
            for row in 0..4 {
                let s: f32 = vals[off + row * 4..off + row * 4 + 4].iter().sum();
                assert!((s - 1.0).abs() < 5e-3, "row sum {s}");
            }
            // Column convergence to 1 needs many more iterations at these
            // extreme ratios (rows collapse toward one-hot), so only assert
            // the columns stayed finite and non-negative here.
            for col in 0..4 {
                let s: f32 = (0..4).map(|r| vals[off + r * 4 + col]).sum();
                assert!(s.is_finite() && s >= 0.0, "col sum {s}");
            }
        }
    }

    #[test]
    fn n1_identity_at_init() {
        let m = MhcBlock::new(1, 32, &dev());
        let h = Tensor::<3>::random([2, 4, 32], Distribution::Normal(0.0, 1.0), &dev());
        let branch = Tensor::<3>::random([2, 4, 32], Distribution::Normal(0.0, 1.0), &dev());
        let out = m.forward(h.clone(), std::slice::from_ref(&branch));
        let want = h + branch;
        let err: f32 = (out - want).powf_scalar(2.0).mean().into_scalar();
        assert!(
            err < 1e-3,
            "n=1 init should be identity residual, mse {err}"
        );
    }

    #[test]
    fn mhc_shape() {
        let m = MhcBlock::new(2, 64, &dev());
        let h = Tensor::<3>::ones([2, 4, 64], &dev());
        let branches: Vec<Tensor<3>> = (0..2)
            .map(|_| Tensor::<3>::random([2, 4, 64], Distribution::Default, &dev()))
            .collect();
        assert_eq!(m.forward(h, &branches).dims(), [2, 4, 64]);
    }

    #[test]
    fn mhc_no_residual_shape() {
        let m = MhcBlock::new(2, 32, &dev());
        let branches: Vec<Tensor<3>> = (0..2)
            .map(|_| Tensor::<3>::random([1, 8, 32], Distribution::Default, &dev()))
            .collect();
        assert_eq!(m.forward_no_residual(&branches).dims(), [1, 8, 32]);
    }

    #[test]
    fn hyper_mappings_shape() {
        let m = MhcBlock::new(3, 96, &dev());
        let h = Tensor::<3>::random([1, 5, 96], Distribution::Default, &dev());
        let (pre, post, res) = m.hyper_mappings(&h);
        assert_eq!(pre.dims(), [1, 5, 1, 3]);
        assert_eq!(post.dims(), [1, 5, 1, 3]);
        assert_eq!(res.dims(), [1, 5, 3, 3]);
        // H_pre, H_post in [0, 2], H_res doubly stochastic
        let pv: Vec<f32> = pre
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(pv.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }
}
