//! Optional router-balancing auxiliary loss (see provenance note):
//! the paper pairs this balancer with token-choice routing; here it
//! regularizes the expert-choice split (aux_weight=0 for strict paper).

use burn::tensor::activation::sigmoid;
use burn::tensor::{Int, Tensor};

use crate::routing::gather_active;

/// Auxiliary load-balancing loss for the router.
///
/// PROVENANCE (verified against arXiv 2507.10524v2, §4.2): the paper pairs a
/// balancing auxiliary loss with its **token-choice** routing ("auxiliary
/// loss and linear router yield the best performance"), while expert-choice
/// routing needs none. This crate implements the expert-choice split; the
/// loss below applies that token-choice-style balancer on top as an optional
/// regularizer against router collapse. Set `MoRConfig::aux_weight = 0` for
/// the strict expert-choice-only recipe.
///
/// Form: penalizes deviation of per-token active usage from the uniform budget
/// `n / T` (each token should be equally likely to be routed deeper, so the
/// router does not collapse onto a fixed token subset):
///
/// ```text
/// counts = scatter of ones at `idx_active`        // hard active mask [B, T, 1]
/// loss   = mean( (sigmoid(scores) - n/T)^2 * counts )
/// ```
///
/// The sigmoid soft-count keeps the loss differentiable through `scores`; the
/// hard `counts` mask (scatter of ones) restricts the penalty to tokens the
/// router actually committed to computing this step.
pub fn load_balancing_loss(scores: Tensor<3>, idx_active: Tensor<2, Int>, n: usize) -> Tensor<1> {
    let [_, t, _] = scores.dims();
    let k = idx_active.dims()[1];
    let frac = (n as f32) / (t as f32);

    // Gather the active rows instead of scattering a [B,T,1] mask: mean over
    // active tokens of (sigmoid(s_active) - n/T)^2 scaled by K/T equals the old
    // scatter-mask form mean_{B,T,1}((soft-frac)^2 * counts).
    let soft_active = gather_active(sigmoid(scores), idx_active); // [B, K, 1]
    soft_active
        .sub_scalar(frac)
        .powf_scalar(2.0)
        .mean()
        .mul_scalar((k as f32) / (t as f32))
}
