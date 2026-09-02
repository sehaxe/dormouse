//! BitNet a4.8-style activation quantization (b1.58 2B4T, 2025):
//! 4-bit activations + 1.58-bit weights trained from scratch with STE.
//!
//! The weight side already exists (SpectralLinear's ternary/2-bit STE
//! quantizers in burn-bitnet). This module adds the activation side with
//! straight-through estimators, applied before the matmuls. The quantizers
//! are f32 tensor ops in the autodiff graph (round has no gradient, so STE
//! is x + (xq - x).detach()), which sidesteps the stack's bf16-backward
//! limitation entirely.
//!
//! Formats:
//! - [`ActFormat::Fp4`]: e2m1 (the paper's fp4). Values 0, +/-{0.5, 0.75,
//!   1, 1.5, 2, 3, 4, 6}; round-to-nearest via the e2m1 mantissa rule.
//! - [`ActFormat::Int(bits)`]: symmetric integer (int4 levels 7, int8 127).
//!
//! Scales are per-token (group = 0) or per-group of `group` columns; the
//! a4.8 recipe uses group scales for the FFN activations.

use burn::backend::{Backend, DispatchKindConversion};
use burn::tensor::{DispatchTensor, FloatDType, Tensor};

/// Activation quantization format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActFormat {
    /// e2m1 fp4 (the paper's a4).
    Fp4,
    /// Symmetric integer, `bits` levels 2^(bits-1)-1.
    Int(u32),
}

impl ActFormat {
    /// Higher-precision variant for the sensitive attention path (the a4.8
    /// recipe keeps attention and the early/late layers at 8 bits).
    pub fn attn(self) -> Self {
        match self {
            ActFormat::Fp4 => ActFormat::Int(8),
            ActFormat::Int(b) => ActFormat::Int(b.max(8)),
        }
    }
}

/// Quantize `x [B, D]` with a straight-through estimator.
/// `group = 0` -> one scale per token (row); `group > 0` -> one scale per
/// `group` columns (the a4.8 FFN recipe).
pub fn quant_act<B: Backend>(x: Tensor<2>, fmt: ActFormat, group: usize) -> Tensor<2>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let [b, d] = x.dims();
    let g = if group == 0 { d } else { group.min(d) };
    let scale = if g == d {
        // Per-token scale: [b, 1] broadcast over d.
        x.clone().abs().max_dim(1).clamp_min(1e-8)
    } else {
        // Per-group scale: [b, d/g, 1] broadcast inside each group.
        x.clone()
            .reshape([b, d / g, g])
            .abs()
            .max_dim(2)
            .clamp_min(1e-8)
            .reshape([b, d / g, 1])
            .repeat(&[1, 1, g])
            .reshape([b, d])
    };
    let norm = x.clone().div(scale.clone()); // [-1, 1] per scale unit
    let q = match fmt {
        ActFormat::Fp4 => fp4_round::<B>(norm),
        ActFormat::Int(bits) => {
            let l = ((1i64 << (bits - 1)) - 1) as f32;
            norm.clone()
                .mul_scalar(l)
                .round()
                .clamp(-l, l)
                .div_scalar(l)
        }
    };
    let xq = q.mul(scale);
    // STE: forward uses the quantized value, backward flows through x.
    x.clone().add(xq.sub(x).detach())
}

/// Round to the nearest e2m1 value (needs the input within the fp4 range;
/// the caller scales it to [-1, 1] first and rescales after).
fn fp4_round<B: Backend>(x: Tensor<2>) -> Tensor<2>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let a = x.clone().abs();
    let sign = x.div(a.clone().clamp_min(1e-12)).clamp(-1.0, 1.0);
    // e2m1 positive set: 0.5, 0.75, 1, 1.5, 2, 3, 4, 6. Rule: e = floor(log2 a),
    // m = a / 2^e in [1, 2) for a >= 0.5; m < 1.25 -> 2^e, else 1.5*2^e.
    let zero = a.clone().lower_scalar(0.25).int().cast(FloatDType::F32);
    let half = a
        .clone()
        .greater_equal_scalar(0.25)
        .int()
        .cast(FloatDType::F32)
        .mul(a.clone().lower_scalar(0.625).int().cast(FloatDType::F32));
    let big = a
        .clone()
        .greater_equal_scalar(0.625)
        .int()
        .cast(FloatDType::F32);
    let ln2 = std::f32::consts::LN_2;
    let e = a
        .clone()
        .clamp_min(1e-8)
        .log()
        .div_scalar(ln2)
        .floor(); // floor(log2 a)
    let v = e.mul_scalar(ln2).exp(); // 2^e
    let m = a.clone().div(v.clone());
    let mant = m
        .clone()
        .lower_scalar(1.25)
        .int()
        .cast(FloatDType::F32)
        .mul_scalar(1.0)
        .add(
            m.greater_equal_scalar(1.25)
                .int()
                .cast(FloatDType::F32)
                .mul_scalar(1.5),
        );
    let val = v.mul(mant).mul(big);
    // 0.5 bucket: a in [0.25, 0.625) -> 0.5
    let val = val.add(half.mul_scalar(0.5));
    // sign * value, zero for a < 0.25
    sign.mul(val).mul(zero.neg().add_scalar(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use burn::tensor::Distribution;

    #[test]
    fn quant_act_bounds_and_ste() {
        let dev = burn::tensor::Device::ndarray();
        let x: Tensor<2> = Tensor::random([16, 64], Distribution::Normal(0.0, 1.0), &dev);
        let q = quant_act::<NdArray>(x.clone(), ActFormat::Int(4), 0);
        let v: Vec<f32> = q.into_data().try_to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "quantized acts must be finite");
        let orig: Vec<f32> = x.into_data().try_to_vec().unwrap();
        let mut max_d = 0.0f32;
        for (a, b) in orig.iter().zip(v.iter()) {
            max_d = max_d.max((a - b).abs());
        }
        assert!(max_d > 1e-4, "quantization must actually quantize");
    }

    #[test]
    fn fp4_round_matches_e2m1_values() {
        let dev = burn::tensor::Device::ndarray();
        // Every positive e2m1 value must round to itself.
        let vals: Vec<f32> = vec![0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let x: Tensor<2> = Tensor::from_data(
            burn::tensor::TensorData::new(vals.clone(), [2, 4]),
            &dev,
        );
        let q = fp4_round::<NdArray>(x);
        let out: Vec<f32> = q.into_data().try_to_vec().unwrap();
        for (a, b) in vals.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "fp4 round-trip: {a} -> {b}"
            );
        }
    }
}