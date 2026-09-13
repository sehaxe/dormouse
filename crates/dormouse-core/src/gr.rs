//! Gated Residual (Qwen3.8-Flash-Next §2.2): widened residual stream with an
//! elementwise sigmoid-gated read and per-branch scalar writes.
//!
//! The block input is a gated average of per-branch RMSNorm outputs (read,
//! Eq. 30-32 of the report); the block output is deposited into every branch
//! through one data-dependent scalar per branch (write, Eq. 33-34). There is
//! no branch-mixing operator (Hres): it costs a full read of the residual
//! state and buys nothing once read/write are expressive.
//!
//! Replaces the pre-norm + ReZero pair: the read already normalizes, and the
//! sigmoid gates bound the write magnitudes, which is the report's stability
//! mechanism (zero loss spikes at 4x LR, no qk-clip needed).
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
    /// Read bottleneck: [nr*d, r] (r = d/8), [r, nr*d], gate [nr*d, nr*d].
    pub wd: LinearLike,
    pub wu: LinearLike,
    /// Write scalars: [nr*d, nr].
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

    /// Read: gate-average the branches into the block input (report Eq. 32).
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
        // G = unvec(sigmoid(Wu SiLU(Wd vec(R)))): one [b,t,d] gate per branch.
        let g_flat = activation::sigmoid(
            self.wu
                .forward::<B>(activation::silu(self.wd.forward::<B>(stacked.clone()))),
        ); // [b*t, nr*d]
        let mut x = Tensor::zeros([b, t, d], &branches[0].device());
        for i in 0..GR_BRANCHES {
            let gi = g_flat
                .clone()
                .slice([0..b * t, i * d..(i + 1) * d])
                .reshape([b, t, d]);
            x = x + gi.mul(normed[i].clone());
        }
        (x.div_scalar(GR_BRANCHES as f32), GrState { normed })
    }

    /// Write: deposit the block output into every branch (report Eq. 33-34).
    /// `branches` are the pre-read branches: Ri' = Ri + si·y.
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
        // s = 2*sigmoid((1/nr) Ww vec(R)): one scalar per branch.
        let s_flat = self
            .ww
            .forward::<B>(stacked)
            .mul_scalar(1.0 / GR_BRANCHES as f32)
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