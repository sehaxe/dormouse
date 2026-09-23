//! Bernoulli-start span-dilated masking with exact expected rate.

use burn::tensor::{Bool, Device, Distribution, Tensor};

/// Random mask over `t` positions: Bernoulli starts dilated causally into
/// contiguous spans of `mask_span`. The start rate is inverted so the
/// expected masked fraction equals `mask_frac` exactly:
/// `p = 1 - (1 - mask_frac)^(1/span)`. Pure tensor ops, no host data.
pub fn mask_indices(
    t: usize,
    mask_frac: f32,
    mask_span: usize,
    device: &Device,
) -> Tensor<1, Bool> {
    let span = mask_span.max(1);
    let rate = 1.0 - (1.0 - mask_frac).powf(1.0 / span as f32);
    let base = Tensor::<1>::random([t], Distribution::Uniform(0.0, 1.0), device).lower_elem(rate);
    if span <= 1 {
        return base;
    }
    let c = base.clone().float().cumsum(0);
    let zeros = Tensor::<1>::zeros([span], device);
    let c_shifted = Tensor::cat(vec![zeros, c.clone()], 0).slice_dim(0, 0..t);
    c.sub(c_shifted).greater_elem(0.0)
}
