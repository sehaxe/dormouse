//! # burn-rmsnorm - RMS Normalization for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! | Reference | What |
//! |-----------|------|
//! | Zhang & Sennrich 2019 | Root mean square layer normalization |
//!
//! > Paper: [RMSNorm](https://arxiv.org/abs/1910.07467).
//! > Used in LLaMA, Mistral, Qwen, DeepSeek, Gemma.
use burn::module::{Module, Param, ParamId};
use burn::tensor::{Device, Tensor};

/// RMS normalization layer: `x / RMS(x) * weight`.
#[derive(Module, Debug)]
pub struct RMSNorm {
    pub weight: Param<Tensor<1>>,
    eps: f32,
}

impl RMSNorm {
    pub fn new(d_model: usize, eps: f32, device: &Device) -> Self {
        Self {
            // require_grad is mandatory: burn 0.22-pre.3's Param::initialized
            // inherits the tensor's flag and Tensor::ones defaults to false,
            // which froze the weight at 1.0 for the whole training (found by
            // dormouse's model_seam gradient_flow test, 2026-09-21).
            weight: Param::initialized(
                ParamId::new(),
                Tensor::ones([d_model], device).require_grad(),
            ),
            eps,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [_, _, d] = x.dims();
        #[cfg(feature = "cuda")]
        {
            let [b, t, _] = x.dims();
            if let Some(out) =
                crate::fused::rmsnorm_cuda::<burn_cubecl::CubeBackend>(
                    x.clone().reshape([b * t, d]),
                    self.weight.val().clone(),
                    self.eps,
                )
            {
                return out.reshape([b, t, d]);
            }
        }
        // eps inside the sqrt (LLaMA/HF convention): x / sqrt(mean(x^2) + eps)
        let rms = x
            .clone()
            .powf_scalar(2.0)
            .mean_dim(2)
            .add_scalar(self.eps)
            .sqrt();
        (x / rms) * self.weight.val().reshape([1, 1, d])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn forward_shape() {
        let norm = RMSNorm::new(64, 1e-5, &dev());
        let x = Tensor::<3>::random([2, 16, 64], Distribution::Default, &dev());
        assert_eq!(norm.forward(x).dims(), [2, 16, 64]);
    }

    #[test]
    fn unit_variance() {
        let norm = RMSNorm::new(32, 1e-5, &dev());
        let x = Tensor::<3>::random([2, 16, 32], Distribution::Default, &dev());
        let y = norm.forward(x);
        let mean_sq: f32 = y.clone().powf_scalar(2.0).mean().into_scalar();
        // weight is ones: RMS-normalized output has unit mean-square per row
        assert!((mean_sq - 1.0).abs() < 1e-2, "mean_sq={mean_sq}");
    }
}
pub mod fused;

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use burn::tensor::{Distribution, Tensor};

    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn fused_matches_tensor() {
        let norm = RMSNorm::new(128, 1e-5, &dev());
        let x = Tensor::<3>::random([2, 16, 128], Distribution::Default, &dev());
        let y_fused = norm.forward(x.clone());
        // reference: recompute via the pure tensor path
        let w = norm.weight.val();
        let rms = x
            .clone()
            .powf_scalar(2.0)
            .mean_dim(2)
            .add_scalar(1e-5)
            .sqrt();
        let y_ref = (x / rms) * w.reshape([1, 1, 128]);
        let diff: f32 = (y_fused - y_ref).abs().max().into_scalar();
        assert!(diff < 1e-4, "fused vs tensor diff {diff}");
    }
}
