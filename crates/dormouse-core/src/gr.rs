//! Gated Residual (Qwen3.8-Flash-Next §2.2), implemented as its Eq. 30-34 are
//! written:
//!
//! ```text
//! Eq. 30  Rhat_i = RMSNorm(R_i; gamma_i)                                  i = 1..nr
//! Eq. 31  G     = unvec(sigma(W_u SiLU((1/nr) W_d vec(Rhat))))   in R^{nr x d}
//! Eq. 32  x     = (1/nr) sum_i G_i . Rhat_i
//! Eq. 33  s     = 2 sigma((1/nr) W_w vec(Rhat))                   in R^nr
//! Eq. 34  R_i'  = R_i + s_i y
//! ```
//!
//! The `2 sigma` in Eq. 33 is the whole stability argument, and it is worth
//! being exact about what it buys: every `s_i` is in (0, 2) and POSITIVE, so a
//! block's deposit can reinforce the widened stream or damp it to half, and it
//! can never subtract it. A block cannot cancel the branch it wrote. The
//! report's *measurement* of that (no loss spikes at 4x LR) is theirs, at 276B
//! tokens over 28 layers; it has not been measured here.
//!
//! The read normalizes, so it also replaces the block's pre-norm - the
//! report's "Eq. (24) loses its Norm". `loop_block.rs` skips `self.norm` when
//! this module is present. There is no branch-mixing operator (Hres): the
//! report's ablation found it worth nothing once read and write are
//! expressive, and it costs a full extra read of the residual state.
//!
//! What is OURS, stated so it is not read back as the paper: the report puts a
//! separate GR module on the attention block and the MLP block of every layer
//! of a 56-sublayer stack. We run ONE weight-shared block recursively, so this
//! module is read and written once per loop ITERATION and the loop's own
//! iteration embedding is added to the read (a depth signal the report does not
//! need and does not have). That is a transposition of the report's operator,
//! not the report's placement.
//!
//! Nothing here has been A/B'd. The verdict owed is GR vs ReZero at a matched
//! budget (`docs/archive/audit-2026-09-25.md:31`); until it runs, the numbers in the
//! report are not ours to spend.
//!
//! All slicing happens on 2D/3D tensors only: dynamic slicing of a 4D
//! autodiff tensor crashes cubecl on sm_120 (see AGENTS.md).

use burn::backend::DispatchKindConversion;
use burn::module::Module;
use burn::tensor::{activation, Device, DispatchTensor, Tensor};
use burn_rmsnorm::RMSNorm;

use crate::param::LinearLike;

/// Number of residual branches (report: nr = 4).
pub const GR_BRANCHES: usize = 4;

#[derive(Module, Debug)]
pub struct GatedResidual {
    pub branch_norms: Vec<RMSNorm>,
    /// Read bottleneck: `W_d` [nr*d, r] (r = d/8), `W_u` [r, nr*d] (Eq. 31).
    pub wd: LinearLike,
    pub wu: LinearLike,
    /// Write scalars: `W_w` [nr*d, nr] (Eq. 33).
    pub ww: LinearLike,
}

/// Per-branch normalized views, reused by read and write.
pub struct GrState {
    pub normed: Vec<Tensor<3>>,
}

impl GatedResidual {
    pub fn new(d: usize, device: &Device) -> Self {
        let nd = GR_BRANCHES * d;
        let r = (d / 8).max(4);
        Self {
            branch_norms: (0..GR_BRANCHES)
                .map(|_| RMSNorm::new(d, 1e-3, device))
                .collect(),
            wd: LinearLike::new(nd, r, r.min(nd), device),
            wu: LinearLike::new(r, nd, r.min(nd), device),
            ww: LinearLike::new(nd, GR_BRANCHES, GR_BRANCHES.min(nd), device),
        }
    }

    /// Read: the block input, Eq. 31-32.
    ///
    /// Eq. 31's `1/nr` sits INSIDE the SiLU, on the bottleneck output. It is
    /// load-bearing, not a scale convention: `vec(Rhat)` is nr branches
    /// concatenated, so without it the gate pre-activation is nr times too
    /// large and a fresh init lands the gate saturated at 0 or 1 instead of
    /// near 0.5.
    pub fn read<B: burn::backend::AutodiffBackend>(&self, branches: &[Tensor<3>]) -> (Tensor<3>, GrState)
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        let [b, t, d] = branches[0].dims();
        let normed: Vec<Tensor<3>> = branches
            .iter()
            .enumerate()
            .map(|(i, r)| self.branch_norms[i].forward(r.clone()))
            .collect();
        let stacked = Tensor::cat(normed.clone(), 2).reshape([b * t, GR_BRANCHES * d]);
        let nr = GR_BRANCHES as f32;
        // G = unvec(sigmoid(Wu SiLU((1/nr) Wd vec(R)))): one [b,t,d] gate per branch.
        let g_flat = activation::sigmoid(self.wu.forward::<B>(activation::silu(
            self.wd.forward::<B>(stacked).mul_scalar(1.0 / nr),
        ))); // [b*t, nr*d]
        let mut x = Tensor::zeros([b, t, d], &branches[0].device());
        for i in 0..GR_BRANCHES {
            let gi = g_flat
                .clone()
                .slice([0..b * t, i * d..(i + 1) * d])
                .reshape([b, t, d]);
            x = x + gi.mul(normed[i].clone());
        }
        (x.div_scalar(nr), GrState { normed })
    }

    /// Write: deposit the block output into every branch, Eq. 33-34.
    /// `branches` are the pre-read branches: R_i' = R_i + s_i y, and `s` is
    /// Eq. 33's `2 sigmoid((1/nr) W_w vec(Rhat))` - the normalized branches of
    /// the SAME read, which is why `state` is the read's return value.
    pub fn write<B: burn::backend::AutodiffBackend>(
        &self,
        branches: &[Tensor<3>],
        state: &GrState,
        y: Tensor<3>,
    ) -> Vec<Tensor<3>>
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        let [b, t, d] = y.dims();
        let stacked = Tensor::cat(state.normed.clone(), 2).reshape([b * t, GR_BRANCHES * d]);
        // Eq. 33: s = 2 * sigmoid((1/nr) * W_w * vec(R)), one scalar per branch.
        let s_flat = activation::sigmoid(
            self.ww
                .forward::<B>(stacked)
                .mul_scalar(1.0 / GR_BRANCHES as f32),
        )
        .mul_scalar(2.0)
        .reshape([b, t, GR_BRANCHES]);
        let mut out = Vec::with_capacity(GR_BRANCHES);
        for i in 0..GR_BRANCHES {
            let si = s_flat
                .clone()
                .slice([0..b, 0..t, i..i + 1]);
            // Store back in the branch dtype (bf16 under --bf16): the sum
            // itself is computed in fp32 (mixed-dtype ops NaN on sm_120).
            out.push(branches[i].clone() + si.mul(y.clone()).cast(branches[i].dtype()));
        }
        out
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::param::LinearLikeInner;
    use burn::backend::Autodiff;
    use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
    use burn::tensor::TensorData;

    // The CPU backend (burn-flex). Either backend works: the arithmetic here is
    // backend-independent and the point is the EQUATION, not the device.
    type B = Autodiff<burn::backend::Flex, BalancedCheckpointing>;

    const D: usize = 8;
    const R: usize = 4; // (D/8).max(4)
    const B_: usize = 1;
    const T: usize = 3;
    const ND: usize = GR_BRANCHES * D;

    /// Dense linears: the TSCT path's u/s/v factorization is not what these
    /// tests are about, and a host reference needs one [out, in] matrix.
    fn device() -> Device {
        Device::flex().autodiff()
    }

    fn gr(device: &Device) -> GatedResidual {
        GatedResidual {
            branch_norms: (0..GR_BRANCHES)
                .map(|_| RMSNorm::new(D, 1e-3, device))
                .collect(),
            wd: LinearLike::dense(ND, R, device),
            wu: LinearLike::dense(R, ND, device),
            ww: LinearLike::dense(ND, GR_BRANCHES, device),
        }
    }

    /// (weight row-major, bias, in, out) read back to the host. burn's
    /// `nn::Linear` weight is [in, out], which is also the report's
    /// orientation: W_d in R^{r x nr d}, W_u in R^{nr d x r}, W_w in
    /// R^{nr x nr d}.
    fn dense_params(l: &LinearLike) -> (Vec<f32>, Vec<f32>, usize, usize) {
        let LinearLikeInner::Dense(m) = &l.inner else {
            panic!("test wants the dense linear");
        };
        let w = m.weight.val();
        let [inp, out] = w.dims();
        let wv = w.into_data().try_to_vec::<f32>().expect("weight to host");
        let bv = m
            .bias
            .as_ref()
            .map(|b| b.val().into_data().try_to_vec::<f32>().expect("bias to host"))
            .unwrap_or_default();
        assert_eq!(wv.len(), out * inp, "weight shape");
        (wv, bv, inp, out)
    }

    /// `y = W x + b` for one column, host, f64.
    fn affine(x: &[f64], w: &[f32], b: &[f32], inp: usize, out: usize) -> Vec<f64> {
        (0..out)
            .map(|j| {
                let mut acc: f64 = b.get(j).copied().unwrap_or(0.0) as f64;
                for k in 0..inp {
                    acc += w[k * out + j] as f64 * x[k];
                }
                acc
            })
            .collect()
    }

    /// `vec(Rhat)` for one (b, t) position, as f64.
    fn stacked64(normed: &[Vec<f32>], pos: usize) -> Vec<f64> {
        stacked(normed, pos).iter().map(|v| *v as f64).collect()
    }

    /// Four deterministic, non-degenerate branches [B_, T, D] (RMSNorm of a
    /// constant row is 0/0; a zero row must not reach the gate).
    fn branches(device: &Device) -> Vec<Tensor<3>> {
        (0..GR_BRANCHES)
            .map(|i| {
                let v: Vec<f32> = (0..B_ * T * D)
                    .map(|n| ((n * 7 + i * 13) % 11) as f32 * 0.37 - 1.1)
                    .collect();
                Tensor::from_data(TensorData::new(v, [B_, T, D]), device)
            })
            .collect()
    }

    fn host3(x: &Tensor<3>) -> Vec<f32> {
        x.clone().into_data().try_to_vec().expect("tensor to host")
    }

    fn silu(v: f64) -> f64 {
        v / (1.0 + (-v).exp())
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// `vec(Rhat)` for one (b, t) position: branch-major concatenation.
    fn stacked(normed: &[Vec<f32>], pos: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(ND);
        for r in normed {
            v.extend_from_slice(&r[pos * D..pos * D + D]);
        }
        v
    }

    /// Eq. 31 + Eq. 32 on the host, from the module's own weights and its own
    /// Rhat (the read's return value). The `1/nr` INSIDE the SiLU is the
    /// whole point: a copy that dropped it passed a golden-of-itself test
    /// because the golden was the copy.
    #[test]
    fn read_is_eq_31_and_eq_32() {
        let dev = device();
        let gr = gr(&dev);
        let br = branches(&dev);
        let (x, state) = gr.read::<B>(&br);
        let normed: Vec<Vec<f32>> = state.normed.iter().map(host3).collect();
        let got = host3(&x);

        let (wd, bd, wd_in, wd_out) = dense_params(&gr.wd);
        let (wu, bu, wu_in, wu_out) = dense_params(&gr.wu);
        assert_eq!((wd_in, wd_out), (ND, R), "W_d in R^(r x nr*d)");
        assert_eq!((wu_in, wu_out), (R, ND), "W_u in R^(nr*d x r)");
        let nr = GR_BRANCHES as f64;

        for pos in 0..B_ * T {
            let v = stacked64(&normed, pos);
            let h: Vec<f64> = affine(&v, &wd, &bd, ND, R)
                .into_iter()
                .map(|z| silu(z / nr))
                .collect();
            let g = affine(&h, &wu, &bu, R, ND)
                .into_iter()
                .map(sigmoid)
                .collect::<Vec<f64>>();
            for di in 0..D {
                let want: f64 = (0..GR_BRANCHES)
                    .map(|i| g[i * D + di] * normed[i][pos * D + di] as f64)
                    .sum::<f64>()
                    / nr;
                let have = got[pos * D + di] as f64;
                assert!(
                    (want - have).abs() < 2e-4,
                    "Eq. 31/32 at pos {pos} dim {di}: equation {want:.6}, code {have:.6}"
                );
            }
        }
    }

    /// Eq. 33 + Eq. 34 on the host. `s = 2 sigma((1/nr) W_w vec(Rhat))` is
    /// POSITIVE and bounded by 2, which is the property that stops a block
    /// from cancelling the branch it just wrote.
    #[test]
    fn write_is_eq_33_and_eq_34_and_s_is_positive() {
        let dev = device();
        let gr = gr(&dev);
        let br = branches(&dev);
        let (_, state) = gr.read::<B>(&br);
        let normed: Vec<Vec<f32>> = state.normed.iter().map(host3).collect();
        let y: Vec<f32> = (0..B_ * T * D).map(|n| ((n * 3) % 7) as f32 * 0.5 - 0.9).collect();
        let y = Tensor::from_data(TensorData::new(y, [B_, T, D]), &dev);
        let out = gr.write::<B>(&br, &state, y.clone());

        let (ww, bw, ww_in, ww_out) = dense_params(&gr.ww);
        assert_eq!(
            (ww_in, ww_out),
            (ND, GR_BRANCHES),
            "W_w in R^(nr x nr*d)"
        );
        let nr = GR_BRANCHES as f64;

        let yv = host3(&y);
        let before: Vec<Vec<f32>> = br.iter().map(host3).collect();
        for pos in 0..B_ * T {
            let v = stacked64(&normed, pos);
            let s: Vec<f64> = affine(&v, &ww, &bw, ND, GR_BRANCHES)
                .into_iter()
                .map(|z| 2.0 * sigmoid(z / nr))
                .collect();
            for (i, si) in s.iter().enumerate() {
                assert!(
                    *si > 0.0 && *si < 2.0,
                    "Eq. 33: s_{i} = {si} left (0, 2) - the write can cancel the stream"
                );
                let have = host3(&out[i]);
                for di in 0..D {
                    let k = pos * D + di;
                    let want = before[i][k] as f64 + si * yv[k] as f64;
                    assert!(
                        (want - have[k] as f64).abs() < 2e-4,
                        "Eq. 34 at pos {pos} branch {i} dim {di}: equation {want:.6}, code {:.6}",
                        have[k]
                    );
                }
            }
        }
    }
}
