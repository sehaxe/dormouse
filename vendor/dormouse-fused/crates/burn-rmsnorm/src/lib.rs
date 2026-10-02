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
            // pre.3 required require_grad here (Param::initialized inherits
            // the flag; without it the weight froze — model_seam
            // gradient_flow, 2026-09-21). pre.4 panics on require_grad for
            // non-autodiff devices (raw-launch modules), and still needs the
            // flag on autodiff devices — so branch on the device context.
            weight: Param::initialized(
                ParamId::new(),
                if device.is_autodiff() {
                    Tensor::ones([d_model], device).require_grad()
                } else {
                    Tensor::ones([d_model], device)
                },
            ),
            eps,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [_, _, d] = x.dims();
        #[cfg(any(feature = "cuda", feature = "wgpu"))]
        {
            let [b, t, _] = x.dims();
            if let Some(out) = crate::fused::rmsnorm_cuda::<burn_cubecl::CubeBackend>(
                x.clone().reshape([b * t, d]),
                self.weight.val().clone(),
                self.eps,
            ) {
                return out.reshape([b, t, d]);
            }
            // The tensor path below is CORRECT, so the only honest way to see
            // a dead fused kernel is a counter (ADR-0019): on dormouse's
            // training backend this line is the normal case, not an error.
            crate::fused::SKIPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

    // BURN_DEVICE=cuda (with the cuda feature) routes these at the fused
    // kernel; the CPU default is the tensor path. The old cuda_tests module
    // returned ndarray unconditionally, so the fused branch at :46-54 was
    // dead in the one test that claimed to cover it.
    fn dev() -> Device {
        #[cfg(feature = "cuda")]
        if std::env::var("BURN_DEVICE").as_deref() == Ok("cuda") {
            return burn::tensor::Device::cuda(0);
        }
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

    #[test]
    fn forward_matches_scalar_reference() {
        // The oracle is a scalar f64 loop over host memory — an INDEPENDENT
        // formulation of RMSNorm, not the tensor chain at :61-67 restated.
        // That restatement is what this file used to assert, on ndarray, where
        // `forward` IS the tensor path: the test compared the code to itself
        // and could not fail. A scalar loop can disagree about eps inside vs
        // outside the sqrt, the mean axis, the weight broadcast, and about
        // the fused kernel's own reduction.
        let (b, t, d) = (3usize, 4usize, 8usize);
        let dev = dev();
        let mut norm = RMSNorm::new(d, 1e-5, &dev);
        // a non-constant gain: with all-ones a broken weight broadcast is
        // invisible.
        let w: Vec<f32> = (0..d).map(|i| 0.5 + 0.25 * i as f32).collect();
        norm.weight = Param::from_tensor(Tensor::<1>::from_floats(w.as_slice(), &dev));
        let xs: Vec<f32> = (0..b * t * d)
            .map(|i| (i as f32 * 0.37).sin() * 3.0)
            .collect();
        let x = Tensor::<3>::from_data(burn::tensor::TensorData::new(xs.clone(), [b, t, d]), &dev);
        let got: Vec<f32> = norm
            .forward(x)
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        for r in 0..b * t {
            let row = &xs[r * d..(r + 1) * d];
            let ss: f64 = row.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
            let rms = (ss / d as f64 + 1e-5).sqrt();
            for (j, &x) in row.iter().enumerate() {
                let want = (x as f64 / rms * f64::from(w[j])) as f32;
                // f32 leaves ~1e-7 relative here; the bound is 1e-4 relative,
                // i.e. three orders of head-room, and still far tighter than
                // any real disagreement.
                let err = (got[r * d + j] - want).abs();
                assert!(
                    err <= 1e-4 * want.abs().max(1.0),
                    "row {r} col {j}: got {} want {want} (err {err})",
                    got[r * d + j]
                );
            }
        }
    }
}
pub mod fused;
