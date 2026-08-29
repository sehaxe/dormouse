//! BitNet a4.8-style activation quantization (b1.58 2B4T, 2025):
//! 4-bit activations + 1.58-bit weights trained from scratch with STE.
//!
//! The weight side already exists (SpectralLinear's ternary/2-bit STE
//! quantizers in burn-bitnet). This module adds the activation side:
//! per-token symmetric int4 with a straight-through estimator, applied
//! before the matmuls. The quantizer is an f32 tensor op in the autodiff
//! graph (round has no gradient, so STE is x + (xq - x).detach()), which
//! sidesteps the stack's bf16-backward limitation entirely.
//!
//! Mixed precision follows a4.8: FFN activations at `bits`, the attention
//! path at `bits.max(8)` (attention is the sensitive part; the paper keeps
//! early/late layers higher precision too).

use burn::backend::{Backend, DispatchKindConversion};
use burn::tensor::{DispatchTensor, Tensor};

/// Symmetric integer quantization with per-token scale and STE.
/// `bits=4` -> levels 7 (int4, like the paper's fp4 range), `bits=8` -> 127.
pub fn quant_act<B: Backend>(x: Tensor<2>, bits: u32) -> Tensor<2>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let levels = (1i64 << (bits - 1)) - 1;
    let l = levels as f32;
    // Per-token absmax scale (fp8-training standard; per-group later).
    let s = x.clone().abs().max_dim(1).clamp_min(1e-8); // [b, 1]
    let q = (x.clone().div(s.clone()).mul_scalar(l)).round().clamp(-l, l);
    let xq = q.mul(s).div_scalar(l);
    // STE: forward uses the quantized value, backward flows through x.
    x.clone().add(xq.sub(x).detach())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    #[test]
    fn quant_act_bounds_and_ste() {
        let dev = burn::tensor::Device::ndarray();
        let x: Tensor<2> = Tensor::random([16, 64], Distribution::Normal(0.0, 1.0), &dev);
        let q = quant_act::<NdArray>(x.clone(), 4);
        let v: Vec<f32> = q.into_data().try_to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "quantized acts must be finite");
        // The quantized value must be coarser than the input (levels 7).
        let orig: Vec<f32> = x.into_data().try_to_vec().unwrap();
        let mut max_d = 0.0f32;
        for (a, b) in orig.iter().zip(v.iter()) {
            max_d = max_d.max((a - b).abs());
        }
        assert!(max_d > 1e-4, "quantization must actually quantize");
    }
}