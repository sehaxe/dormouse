//! Capacity routing mechanics: top-k selection, gather/scatter around the
//! block, Eq.1 scaling and the sampling-oriented BCE auxiliary loss.

use burn::tensor::activation;
use burn::tensor::{Device, Int, IntDType, Tensor};

use crate::router::ModRouter;

/// Select the `k` highest-weight token indices per batch row (expert-choice
/// routing, paper section 3.3), `k = round(capacity_frac * T)` clamped to
/// `[1, T-1]`. Descending by weight; ties resolve to the lower index.
///
/// Rounds of native `argtopk(take <= 16)` with Add-scatter masking between
/// rounds (same trick as burn-mor): cubecl's ArgTopK compiles a take-unrolled
/// insertion sort, so take <= 16 stays ~1s of NVRTC while larger takes cost
/// ~50s, and the old per-pick masked-argmax loop needed ~3 launches *per
/// token* selected. ponytail note: upgrade to plain `topk_with_indices` when
/// cubecl fixes its host-fallback sync.
pub fn select_topk(weights: Tensor<3>, capacity_frac: f32, device: &Device) -> Tensor<2, Int> {
    let [b, t, _] = weights.dims();
    let k = ((t as f32) * capacity_frac).round() as usize;
    let k = k.clamp(1, t.saturating_sub(1));

    // flat [B, T] view: indices along dim 1
    let mut work = weights.reshape([b, t]);
    let mut parts: Vec<Tensor<2, Int>> = Vec::with_capacity(k.div_ceil(16));
    let mut remaining = k;
    while remaining > 0 {
        let take = remaining.min(16);
        let idx = work.clone().argtopk(take, 1).cast(IntDType::I64); // [B, take]
        parts.push(idx.clone());
        remaining -= take;
        if remaining > 0 {
            // knock out the picked positions for the next round: -1e30 sits
            // below every realistic router weight (Linear output)
            let neg = Tensor::<2>::full([b, take], -1e30_f32, device);
            work = work.scatter(1, idx, neg, burn::tensor::IndexingUpdateOp::Add);
        }
    }
    Tensor::cat(parts, 1)
}

/// Gather the rows of `h [B, T, D]` selected by `idx [B, K]` -> `[B, K, D]`
/// (expanded-index gather; `gather_nd` is avoided, illegal-address bugs on
/// CUDA in burn 0.21).
pub fn gather_selected(h: Tensor<3>, idx: Tensor<2, Int>) -> Tensor<3> {
    let [b, k] = idx.dims();
    let [_, _, d] = h.dims();
    let idx3 = idx.unsqueeze_dim::<3>(2).expand([b, k, d]);
    h.gather(1, idx3)
}

/// Apply the paper's routing step (eq. 1) around a block `f`:
///
/// 1. router weights `r [B, T, 1]` are computed from `x`,
/// 2. the `k` highest-weighted tokens are selected and run through `f`
///    (self-attention + MLP, applied to the reduced set),
/// 3. the block output is multiplied by its token's router weight and
///    scattered back, so routed tokens get `x + r*f(x~)` and the rest `x`.
///
/// `block` receives `[B, K, D]` and returns `[B, K, D]`.
pub fn route_block<F>(
    x: Tensor<3>,
    router: &ModRouter,
    capacity_frac: f32,
    block: F,
    device: &Device,
) -> Tensor<3>
where
    F: FnOnce(Tensor<3>) -> Tensor<3>,
{
    let r = router.weights(x.clone()); // [B, T, 1]
    let idx = select_topk(r.clone(), capacity_frac, device); // [B, K]
    let x_active = gather_selected(x.clone(), idx.clone()); // [B, K, D]
    let out_active = block(x_active); // [B, K, D]
    let r_active = gather_selected(r, idx.clone()); // [B, K, 1]
    let scaled = out_active.mul(r_active); // eq. 1: r_i * f(X~)
    let scattered = scatter_selected_full(idx, scaled, device, &x); // [B, T, D]
    if std::env::var("MOD_DEBUG").is_ok() {
        let sv: Vec<f32> = scattered.clone().into_data().try_to_vec().unwrap();
        println!("scattered={sv:?}");
    }
    x.add(scattered)
}

/// Scatter `values [B, K, D]` at `idx` into a zeros `[B, T, D]` tensor whose
/// `T` is taken from `x` (the pre-routing input).
fn scatter_selected_full(
    idx: Tensor<2, Int>,
    values: Tensor<3>,
    device: &Device,
    x: &Tensor<3>,
) -> Tensor<3> {
    let [b, k] = idx.dims();
    let [_, t, d] = x.dims();
    let idx3 = idx.unsqueeze_dim::<3>(2).expand([b, k, d]);
    Tensor::<3>::zeros([b, t, d], device).scatter(
        1,
        idx3,
        values,
        burn::tensor::IndexingUpdateOp::Add,
    )
}

/// Binary-cross-entropy auxiliary loss (paper section 3.5, "Sampling"):
/// `sigmoid(router weight)` is trained to match the top-k selection, so the
/// model can be sampled autoregressively by routing on `weight > 0.5`.
///
/// ```text
/// loss = BCE(sigmoid(r), 1[r in top-k])
/// ```
pub fn bce_aux_loss(weights: Tensor<3>, idx: Tensor<2, Int>, device: &Device) -> Tensor<1> {
    let [b, t, _] = weights.dims();
    let idx3 = idx.unsqueeze_dim::<3>(2); // [B, K, 1]
    let ones = Tensor::<3>::ones(idx3.shape(), device);
    let target = Tensor::<3>::zeros([b, t, 1], device).scatter(
        1,
        idx3,
        ones,
        burn::tensor::IndexingUpdateOp::Add,
    );
    let p = activation::sigmoid(weights);
    let eps = 1e-7f32;
    let loss = target.clone().mul(p.clone().add_scalar(eps).log()).add(
        target
            .neg()
            .add_scalar(1.0)
            .mul(p.neg().add_scalar(1.0 + eps).log()),
    );
    loss.mean().neg()
}
