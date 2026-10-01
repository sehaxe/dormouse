//! # dormouse-mhc - Manifold-Constrained Hyper-Connections for Burn
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
//! Sinkhorn-Knopp t_max = 20, expansion rate n = 4, Layer Norm eps = 1e-20
//! (Table 5). Those four are the paper's. The static-bias initialisation
//! (`b_pre = b_res = 10`, `b_post = 0`, `block.rs:7-8`) is OURS: the paper
//! states no bias init, and 10 is chosen so `sigma(10) ~ 1` and
//! `Sinkhorn(exp(diag 10)) ~ I`.
//!
//! Two readings of the paper matter, both decided in favour of the PRINTED
//! equation:
//!
//! - **Eq. 7 has no `tanh`.** HC's Eq. 5 parameterises its mappings as
//!   `alpha * tanh(theta x') + b`; mHC's Eq. 7 - and the kernel section's
//!   Eq. 14-16, which is independent of the prose - is a plain linear
//!   projection `alpha * (x' phi) + b`. The prose says "we follow the original
//!   HC formulation"; the equations carry no `tanh`; this crate implements the
//!   equations. The consequence is named rather than absorbed: `phi_*` here are
//!   unbounded linear maps where HC's are `tanh`-squashed, which is why the
//!   Sinkhorn input is unbounded and why `sinkhorn.rs` has to run in the log
//!   domain at all.
//! - **`H_pre` is not applied inside `forward`.** Eq. 3 is
//!   `x' = H_res x + H_post^T F(H_pre x, W)` and `H_pre x` is
//!   `[1,n] [n,C] = [1,C]`: it feeds the block's own function `F`, which lives
//!   in the caller. `MhcBlock::forward` therefore implements Eq. 3's
//!   `H_res x + H_post^T F` terms with `F`'s output handed in, and
//!   `hyper_mappings` returns the `H_pre` a caller that CAN apply it needs.
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

    // ---- the fixture that separates Eq. 3 from every wrong reading of it --

    /// The block the hand computation below describes, with the DYNAMIC part
    /// switched off so the whole operator is a function of the static biases:
    /// `alpha_* = 0` kills the `x' phi` term of Eq. 7, leaving
    /// `H_post = 2 sigma(b_post)` (Eq. 8) and `H_res = Sinkhorn(b_res)`
    /// (Eq. 8-9) exactly.
    fn hand_fixture(n: usize, b_post: [f32; 2], b_res: [[f32; 2]; 2]) -> MhcBlock {
        let mut m = MhcBlock::new(n, 2 * n, &dev());
        m.alpha_pre = burn::module::Param::from_tensor(Tensor::<1>::zeros([1], &dev()));
        m.alpha_post = burn::module::Param::from_tensor(Tensor::<1>::zeros([1], &dev()));
        m.alpha_res = burn::module::Param::from_tensor(Tensor::<1>::zeros([1], &dev()));
        // `from_floats([b_post])` is rank 2, not rank 1 - burn's
        // `From<[f32; N]> for TensorData` infers the rank from the nesting, so
        // wrapping a `[f32; 2]` in another `[..]` builds dims `[1, 2]` and dies
        // in `from_data` with "Given dimensions differ from the tensor rank".
        // `TensorData::new` is the spelling that carries the shape explicitly.
        m.b_post = burn::module::Param::from_tensor(Tensor::<1>::from_data(
            burn::tensor::TensorData::new(b_post.to_vec(), [2]),
            &dev(),
        ));
        m.b_res = burn::module::Param::from_tensor(Tensor::<2>::from_floats(b_res, &dev()));
        m
    }

    /// EQ. 3, ON NUMBERS A HUMAN CAN CHECK. `n = 2` streams of width `C = 2`,
    /// so `D = 4` and the whole residual is four numbers.
    ///
    /// `b_res = [[0, -ln4], [-ln4, 0]]` -> `M = exp(b_res) = [[1, 1/4], [1/4, 1]]`.
    /// Sinkhorn's alternating normalization preserves the cross ratio
    /// `M00 M11 / (M01 M10) = 16`, and a 2x2 doubly stochastic matrix
    /// `[[p, 1-p], [q, 1-q]]` has cross ratio `p(1-q) / ((1-p) q)`, which is 16
    /// exactly at `p = q = 4/5`. 20 iterations (the paper's `t_max`) reach it
    /// to the last bit of f32, so the expected `H_res` is the exact literal
    /// `[[0.8, 0.2], [0.2, 0.8]]` and there is no tolerance to argue about.
    ///
    /// `b_post = [-0.5, 1.5]` gives `H_post = 2 sigma(b_post) =
    /// [0.75508134, 1.63514895]`. Eq. 3 then says, for each stream `j` and
    /// each column `c`: `out[j,c] = sum_i H_res[j,i] x[i,c] + H_post[j] y[j,c]`,
    /// with `x = [[1,2],[3,4]]` and `y = [[10,20],[30,40]]`. All four numbers,
    /// row-major (which is what `reshape([b,t,n,C])` gives back):
    ///
    /// | | `c = 0` | `c = 1` |
    /// |---|---|---|
    /// | `j = 0` | `0.8·1 + 0.2·3 + 0.75508134·10` = **8.95081338** | `0.8·2 + 0.2·4 + 0.75508134·20` = 17.50162675 |
    /// | `j = 1` | `0.2·1 + 0.8·3 + 1.63514895·30` = 51.65446857 | `0.2·2 + 0.8·4 + 1.63514895·40` = 69.00595810 |
    ///
    /// WHY A FIXTURE AND NOT "the output moved" (the AttnRes lesson, 511daa5:
    /// their first gate was blind to a `d^-0.5` scale and an L2 norm together,
    /// a d-fold logit compression that every "did it change?" assertion
    /// passed). The wrong readings of Eq. 3 that a plausible transcription
    /// makes, on `out[0,0]`:
    ///
    /// | reading of Eq. 3 | `out[0,0]` | gap | relative |
    /// |---|---|---|---|
    /// | **as printed** | **8.950813** | — | — |
    /// | `H_post = sigma(.)`, the factor 2 of Eq. 8 dropped | 5.175407 | −3.775406 | 42.2 % |
    /// | `H_res = b_res`, the Sinkhorn of Eq. 9 dropped | 3.391930 | −5.558883 | 62.1 % |
    /// | `H_post` indexed off by one stream | 17.751490 | +8.800677 | 98.3 % |
    ///
    /// On an output of order 10 that is 42%, 62% and 98% relative - two orders
    /// of magnitude above any tolerance a test could be accused of choosing to
    /// fit. A gate that only asserted "the readout moved" is green under all
    /// four. The last row is not decoration: it is the mistake THIS FILE made
    /// in its own expected values, which is why the four literals are written
    /// out as the arithmetic above rather than as bare numbers.
    ///
    /// **What this fixture provably cannot separate**, so nobody reads more
    /// into it: `H_res` transposed. The fixture's `b_res` is symmetric, so
    /// `H_res^T = H_res` and the wrong reading is the right answer. Catching a
    /// transposed `H_res` needs a non-symmetric `b_res`, which changes the
    /// hand-arithmetic and is a second fixture, not a tweak of this one.
    #[test]
    fn eq3_residual_is_the_papers_equation_on_a_hand_computed_fixture() {
        let ln4 = 4f32.ln();
        let m = hand_fixture(2, [-0.5, 1.5], [[0.0, -ln4], [-ln4, 0.0]]);
        let h = Tensor::<3>::from_floats([[[1.0, 2.0, 3.0, 4.0]]], &dev());
        let y = Tensor::<3>::from_floats([[[10.0, 20.0, 30.0, 40.0]]], &dev());
        let out: Vec<f32> = m
            .forward(h, std::slice::from_ref(&y))
            .into_data()
            .try_to_vec()
            .expect("readable [1,1,4]");
        for (i, want) in [
            8.950813375962909f32,
            17.50162675192582,
            51.65446857161862,
            69.00595809549148,
        ]
        .iter()
        .enumerate()
        {
            assert!(
                (out[i] - want).abs() < 1e-5,
                "Eq. 3 out[{i}] = {}, hand-computed {want}",
                out[i]
            );
        }
        // And the mapping itself, not just its effect: the fixture's whole
        // point is that H_res is the PROJECTION, so assert the projection.
        let (_, _, res) = m.hyper_mappings(&Tensor::<3>::ones([1, 1, 4], &dev()));
        let r: Vec<f32> = res.into_data().try_to_vec().expect("[1,1,2,2]");
        for (i, want) in [0.8f32, 0.2, 0.2, 0.8].iter().enumerate() {
            assert!(
                (r[i] - want).abs() < 1e-5,
                "H_res[{i}] = {}, want {want} - the Sinkhorn of Eq. 9 did not run",
                r[i]
            );
        }
    }
}
