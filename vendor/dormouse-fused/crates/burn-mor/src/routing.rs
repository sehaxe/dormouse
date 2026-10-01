//! Expert-choice active/inactive token split and the dense gather/
//! scatter moves around the shared recursion block.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, IndexingUpdateOp, Int, Tensor};

use crate::topk::topk_indices;

/// Configuration for integrating MoR routing into a looped model.
///
/// The same shared recursion block is applied up to `max_iter` times; at every
/// recursion step `keep_frac` of the still-active tokens are selected by the
/// router to keep computing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoRConfig {
    /// Fraction of active tokens routed into the recursion block at each step.
    pub keep_frac: f32,
    /// KV-sharing cache block size (tokens), used by the cache integration.
    pub block_size: usize,
    /// Weight of the load-balancing auxiliary loss in the total loss.
    pub aux_weight: f64,
    /// Detach router scores before gating the block output, so the main task
    /// loss trains the router only through the auxiliary loss.
    pub gradient_detach: bool,
}

impl MoRConfig {
    /// Create a config with a given `keep_frac` and the defaults
    /// (`block_size = 128`, `aux_weight = 0.01`, `gradient_detach = true`).
    pub fn new(keep_frac: f32) -> Self {
        Self {
            keep_frac,
            block_size: 128,
            aux_weight: 0.01,
            gradient_detach: true,
        }
    }
}

/// Lightweight per-token importance scorer: a single linear map `d → 1`
/// without bias (the paper's "linear router" with sigmoid/tanh normalization
/// applied downstream if needed).
#[derive(Module, Debug)]
pub struct MoRRouter {
    /// Linear projection from hidden dimension to a scalar score.
    pub proj: Linear,
}

impl MoRRouter {
    /// Create a router projecting `d_model`-dimensional states to one score.
    pub fn new(d_model: usize, device: &Device) -> Self {
        Self {
            proj: LinearConfig::new(d_model, 1).init(device),
        }
    }

    /// Router scores `[B, T, 1]` — token "importance". Higher is more important.
    pub fn scores(&self, h: Tensor<3>) -> Tensor<3> {
        self.proj.forward(h)
    }
}

/// Split tokens into the `top-k` (active) and the remaining (inactive) set for
/// one recursion step.
///
/// `k = round(keep_frac * T)`, clamped to `[1, T-1]` so both sets are non-empty;
/// `T <= 1` degenerates to the single token active, inactive empty
/// (a token with depth 1 still passes the first recursion, per the paper).
///
/// Returns `(active_idx, inactive_idx)` of shape `[B, K]` / `[B, T-K]`, in
/// descending score order. Pure tensor ops: no `gather_nd`, no host branching.
///
/// When the inactive set is not needed (the recursion loop drops it), use
/// [`select_active_only`] — it skips the mask + second top-k (~half the ops).
pub fn select_active(
    scores: Tensor<3>,
    keep_frac: f32,
    device: &Device,
) -> (Tensor<2, Int>, Tensor<2, Int>) {
    let [b, t, _] = scores.dims();
    if t <= 1 {
        // Degenerate: with <= 1 token there is no split — the token is active
        // (a depth-1 token still passes the first recursion per the paper),
        // inactive is empty. (round(frac*t).clamp(1, t-1) would panic on
        // clamp(1, 0).)
        let active = Tensor::<1, Int>::arange(0..t as i64, device)
            .reshape([1, t])
            .expand([b, t]);
        let inactive = Tensor::<2, Int>::empty([b, 0], device);
        return (active, inactive);
    }
    let k = (((t as f32) * keep_frac).round() as usize).clamp(1, t.saturating_sub(1));

    // `topk_indices` = GPU `argtopk` on cubecl backends (burn-cuda 0.21's
    // `topk_with_indices` falls back to a host sort whose read can fail
    // inside autodiff under async load, panicking and poisoning the CUDA
    // context — subsequent kernels fail with CUDA_ERROR_ILLEGAL_ADDRESS).
    // Only the indices are needed; the values are discarded.
    let flat = scores.clone().reshape([b, t]);
    let active_idx = topk_indices(flat, k, 1);

    // Mask the active positions so the second top-k recovers exactly the
    // inactive set: Add-scatter -1e30 onto the active indices. -1e30 sits
    // below every real router score (Linear outputs are O(1); the original
    // masked-argmax loop used the same constant), so active tokens can never
    // be re-picked. The data-dependent min-1 mask guarded against routers
    // with unbounded negative outputs; nothing reaches -1e30, and the
    // scatter-add costs 3 launches instead of 7.
    let idx3 = active_idx.clone().unsqueeze_dim::<3>(2); // [B, K, 1]
    let neg = Tensor::<3>::full([b, k, 1], -1e30_f32, device);
    let mask = Tensor::<3>::zeros([b, t, 1], device).scatter(1, idx3, neg, IndexingUpdateOp::Add);
    let masked = scores.add(mask);
    let inactive_idx = topk_indices(masked.reshape([b, t]), t - k, 1);

    (active_idx, inactive_idx)
}

/// Active set only — skips the inactive computation of [`select_active`].
///
/// Same `(active_idx, inactive_idx)` signature; `inactive_idx` is an empty
/// `[B, 0]` tensor. For the common loop pattern that drops the inactive set
/// this saves the mask + second top-k per recursion step.
pub fn select_active_only(
    scores: Tensor<3>,
    keep_frac: f32,
    device: &Device,
) -> (Tensor<2, Int>, Tensor<2, Int>) {
    let [b, t, _] = scores.dims();
    let inactive = Tensor::<2, Int>::empty([b, 0], device);
    if t <= 1 {
        let active = Tensor::<1, Int>::arange(0..t as i64, device)
            .reshape([1, t])
            .expand([b, t]);
        return (active, inactive);
    }
    let k = (((t as f32) * keep_frac).round() as usize).clamp(1, t.saturating_sub(1));
    let flat = scores.reshape([b, t]);
    let active_idx = topk_indices(flat, k, 1);
    (active_idx, inactive)
}

/// Gather the rows of `h [B, T, D]` selected by `idx [B, K]` → `[B, K, D]`.
///
/// Uses `gather(dim, indices)` with the indices expanded along the last
/// dimension — `gather_nd` is avoided (illegal-address bugs on CUDA in burn 0.21).
pub fn gather_active(h: Tensor<3>, idx: Tensor<2, Int>) -> Tensor<3> {
    let [b, k] = idx.dims();
    let [_, _, d] = h.dims();
    let idx3 = idx.unsqueeze_dim::<3>(2).expand([b, k, d]); // [B, K, D]
    h.gather(1, idx3)
}

/// Place `values [B, K, D]` (the block output of active tokens) back into the
/// full `[B, T, D]` tensor `out` at the `idx` positions.
///
/// With `out = zeros` this yields exactly `values` at the active positions and
/// zeros elsewhere (dropped tokens pass through the residual only). With a
/// non-zero `out` the values are accumulated (`IndexingUpdateOp::Add` — the
/// only scatter mode burn 0.21 implements).
pub fn scatter_active(
    out: Tensor<3>,
    idx: Tensor<2, Int>,
    values: Tensor<3>,
    _device: &Device,
) -> Tensor<3> {
    let [b, k] = idx.dims();
    let [_, _, d] = values.dims();
    let idx3 = idx.unsqueeze_dim::<3>(2).expand([b, k, d]); // [B, K, D]
    out.scatter(1, idx3, values, IndexingUpdateOp::Add)
}
