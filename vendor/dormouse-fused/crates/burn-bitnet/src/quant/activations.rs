//! Activation quantization: b1.58 absmax 8-bit and the BitNet v2
//! Hadamard-rotated 4-bit path (arXiv 2504.18415), plus the backend-generic
//! tensor fallback used by the autodiff CUDA ops.

use burn::tensor::Tensor;

use crate::fwt::fast_walsh_hadamard;

/// b1.58 activation quantizer: per-token absmax scale to int8-like values.
pub fn activation_quant_8bit(x: Tensor<3>) -> Tensor<3> {
    let [b, t, d] = x.dims();
    let flat = x.clone().reshape([b * t, d]);
    let scale = flat.clone().abs().max_dim(1).clamp_min(1e-5);
    let y = flat
        .div(scale.clone())
        .mul_scalar(127.0)
        .round()
        .clamp(-128.0, 127.0)
        .div_scalar(127.0)
        .mul(scale)
        .reshape([b, t, d]);
    let base = x.clone().detach();
    x.add(y.sub(base))
}

/// BitNet v2 joint scheme: Hadamard rotation (outlier suppression) then
/// quantize — 8-bit absmax or 4-bit per-token absmean — and rotate back.
/// Straight-through via the residual trick; the fused CUDA paths return the
/// same values with one kernel when available.
pub fn bitnet_v2_quantize(x: Tensor<3>, bits: usize) -> Tensor<3> {
    if bits >= 16 {
        return x;
    }
    let [b, t, d] = x.dims();
    let flat = x.clone().reshape([b * t, d]);
    let rotated = fast_walsh_hadamard(flat);
    let deq_rot = if bits >= 8 {
        #[cfg(all(feature = "cuda", feature = "autodiff"))]
        {
            type CudaBare = burn_cubecl::CubeBackend;
            if let Some(quant) = crate::fwt_cuda::quant_autodiff::<CudaBare>(rotated.clone(), 8) {
                return x.clone().add(
                    fast_walsh_hadamard(quant)
                        .reshape([b, t, d])
                        .sub(x.clone().detach()),
                );
            }
        }
        #[cfg(feature = "cuda")]
        {
            if let Some(quant) = crate::fwt_cuda::quant_cuda(&rotated, 8) {
                return x.clone().add(
                    fast_walsh_hadamard(quant)
                        .reshape([b, t, d])
                        .sub(x.clone().detach()),
                );
            }
        }
        let scale = rotated.clone().abs().max_dim(1).clamp_min(1e-12);
        let q = rotated
            .clone()
            .div(scale.clone())
            .mul_scalar(127.0)
            .round()
            .clamp(-128.0, 127.0);
        fast_walsh_hadamard(q.clone().mul(scale.clone()).div_scalar(127.0))
    } else {
        // BitNet v2: 4-bit uses per-token ABSMEAN scale (the Hadamard rotation
        // suppresses outliers so absmean is safe). q = round(x/scale) already
        // lands in [-8,7] (no x127 pre-scaling — that's the absmax convention).
        #[cfg(all(feature = "cuda", feature = "autodiff"))]
        {
            type CudaBare = burn_cubecl::CubeBackend;
            if let Some(quant) = crate::fwt_cuda::quant_autodiff::<CudaBare>(rotated.clone(), 4) {
                return x.clone().add(
                    fast_walsh_hadamard(quant)
                        .reshape([b, t, d])
                        .sub(x.clone().detach()),
                );
            }
        }
        #[cfg(feature = "cuda")]
        {
            if let Some(quant) = crate::fwt_cuda::quant_cuda(&rotated, 4) {
                return x.clone().add(
                    fast_walsh_hadamard(quant)
                        .reshape([b, t, d])
                        .sub(x.clone().detach()),
                );
            }
        }
        let scale = rotated.clone().abs().mean_dim(1).clamp_min(1e-12);
        let q = rotated.clone().div(scale.clone()).round().clamp(-8.0, 7.0);
        fast_walsh_hadamard(q.mul(scale))
    };
    let y = deq_rot.reshape([b, t, d]);
    let base = x.clone().detach();
    x.add(y.sub(base))
}

/// Pure tensor-path elementwise quantize (autodiff fallback): absmax/absmean
/// scale then round/clamp/rescale. Straight-through is applied by callers.
pub fn quantize_tensor<B: burn::backend::Backend>(x: Tensor<2>, bits: usize) -> Tensor<2>
where
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    let scale = if bits == 8 {
        x.clone().abs().max_dim(1).clamp_min(1e-12)
    } else {
        x.clone().abs().mean_dim(1).clamp_min(1e-12)
    };
    if bits == 8 {
        x.div(scale.clone())
            .mul_scalar(127.0)
            .round()
            .clamp(-128.0, 127.0)
            .div_scalar(127.0)
            .mul(scale)
    } else {
        x.div(scale.clone()).round().clamp(-8.0, 7.0).mul(scale)
    }
}

pub fn quantize_4bit(x: Tensor<3>) -> Tensor<3> {
    bitnet_v2_quantize(x, 4)
}
pub fn quantize_8bit(x: Tensor<3>) -> Tensor<3> {
    bitnet_v2_quantize(x, 8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    fn dev() -> burn::tensor::Device {
        burn::tensor::Device::ndarray()
    }

    #[test]
    fn activation_8bit_shape() {
        assert_eq!(
            activation_quant_8bit(Tensor::<3>::random(
                [2, 8, 64],
                Distribution::Normal(0.0, 1.0),
                &dev()
            ))
            .dims(),
            [2, 8, 64]
        );
    }

    #[test]
    fn v2_4bit_roundtrip() {
        let x = Tensor::<3>::random([2, 8, 32], Distribution::Normal(0.0, 1.0), &dev());
        let q = quantize_4bit(x.clone());
        let rel_err: f32 = (x - q).powf_scalar(2.0).mean().into_scalar();
        assert!(
            rel_err < 0.1,
            "4-bit roundtrip mse {rel_err} (scale must multiply, not divide)"
        );
    }

    #[test]
    fn v2_8bit_roundtrip() {
        let x = Tensor::<3>::random([2, 16, 128], Distribution::Normal(0.0, 1.0), &dev());
        let q = quantize_8bit(x.clone());
        let rel_err: f32 = (x - q).powf_scalar(2.0).mean().into_scalar();
        assert!(
            rel_err < 0.002,
            "8-bit roundtrip mse {rel_err} (scale must multiply, not divide)"
        );
    }

    #[test]
    fn v2_pass_through() {
        let x = Tensor::<3>::random([1, 4, 16], Distribution::Normal(0.0, 1.0), &dev());
        let q = bitnet_v2_quantize(x.clone(), 16);
        let d: Vec<f32> = (x - q)
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(d.iter().all(|&v| v.abs() < 1e-5));
    }
}
