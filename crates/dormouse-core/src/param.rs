//! param - TSCT linear via burn-sct SpectralLinear, pad to multiple of 4,
//! NM knob, BF16 env (mirrors aria semantics; fresh mini composition)
use burn::module::Module;
use burn::nn::LinearConfig;
use burn::tensor::{Device, DispatchTensor, Tensor};
use burn::backend::DispatchKindConversion;
use burn_spectral::SpectralLinear;
use burn_sct::SctLinear;

/// bf16 mode env flag: activations stream in bf16, weights fp32 cast per
/// forward (burn-ndarray has no bf16 dtype -> CPU tests stay fp32).
/// Cached: this sits in the per-iteration hot path and `env::var` takes a
/// process-wide lock on every call.
pub fn bf16_on() -> bool {
    static BF16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *BF16.get_or_init(|| std::env::var("BF16").map(|v| v != "0").unwrap_or(false))
}

/// DM_QUANT_DEBUG, read once per process (queried by every LinearLike
/// forward otherwise).
fn quant_debug_on() -> bool {
    static DBG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DBG.get_or_init(|| std::env::var("DM_QUANT_DEBUG").is_ok())
}

#[derive(Module, Debug)]
pub struct LinearLike {
    pub inner: LinearLikeInner,
    #[module(skip)]
    pub out_features: usize,
}

#[derive(Module, Debug)]
pub enum LinearLikeInner {
    Tsct(SpectralLinear),
    Sct(SctLinear),
    Dense(burn::nn::Linear),
}

impl LinearLike {
    pub fn new(in_features: usize, out_features: usize, rank: usize, device: &Device) -> Self {
        // Pad to multiple of 4: cubek matmul vectorization needs N%4==0
        // (aria probe: N=2,3,5,6,7 dirty under initcheck).
        let padded = if !out_features.is_multiple_of(4) && out_features != 1 {
            out_features.next_multiple_of(4)
        } else {
            out_features
        };
        Self {
            inner: LinearLikeInner::Tsct(SpectralLinear::new(in_features, padded, rank, device)),
            out_features,
        }
    }

    pub fn forward<B: burn::backend::AutodiffBackend>(&self, x: Tensor<2>) -> Tensor<2>
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        let y = match &self.inner {
            LinearLikeInner::Tsct(l) => {
                if quant_debug_on() {
                    println!("[ll] quant={:?} out={} bf16_compute={}", l.quant, l.out_features, l.bf16_compute);
                }
                if l.bf16_compute {
                    // bf16 matmuls (tensor cores) with the fp32 autodiff
                    // graph; the factor quant follows the configured format.
                    // The custom op is CUDA-only; elsewhere fall back.
                    #[cfg(feature = "cuda")]
                    {
                        l.forward_quant_bf16::<B>(x)
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        if l.quant != burn_spectral::QuantFormat::Fp32 {
                            l.forward_quant::<B>(x)
                        } else {
                            l.forward(x)
                        }
                    }
                } else if l.quant != burn_spectral::QuantFormat::Fp32 {
                    l.forward_quant::<B>(x)
                } else {
                    l.forward(x)
                }
            }
            LinearLikeInner::Sct(l) => l.forward::<B>(x),
            LinearLikeInner::Dense(l) => l.forward(x),
        };
        // slice back to the real out_features when padded (extra columns are
        // never read downstream and would be the float4 tail).
        if y.dims()[1] != self.out_features {
            let n = y.dims()[0];
            y.slice([0..n, 0..self.out_features])
        } else {
            y
        }
    }

    /// Set the quantized factor format on the TSCT arm (no-op otherwise).
    pub fn set_quant(&mut self, quant: burn_spectral::QuantFormat) {
        if let LinearLikeInner::Tsct(l) = &mut self.inner {
            l.set_quant(quant);
        }
    }

    /// Enable the bf16 matmul path (fp32 graph, tensor-core forward).
    pub fn set_bf16_compute(&mut self, on: bool) {
        if let LinearLikeInner::Tsct(l) = &mut self.inner {
            l.bf16_compute = on;
        }
    }

    /// Polar-retract the TSCT masters U/V to orthonormal (burn-spectral NS,
    /// on device, keeps autodiff tracking). No-op for dense/sct variants.
    /// Without periodic retract the factors drift and the quantized forward
    /// degrades (bf16_KERNEL_PLAN: retract every 1 step, monitor max_ortho).
    pub fn retract(&mut self, iters: usize) {
        if let LinearLikeInner::Tsct(l) = &mut self.inner {
            l.retract(iters);
        }
    }

    /// Worst-case orthonormality error of the TSCT masters (0 for dense/sct).
    /// Syncs the device (reads back); call at monitor cadence, not per step.
    pub fn max_ortho(&self) -> f32 {
        match &self.inner {
            LinearLikeInner::Tsct(l) => {
                let u = burn_spectral::ortho_error(&l.u.val());
                let v = burn_spectral::ortho_error(&l.v.val());
                u.max(v)
            }
            _ => 0.0,
        }
    }
}

impl LinearLike {
    pub fn dense(in_features: usize, out_features: usize, device: &Device) -> Self {
        let padded = if !out_features.is_multiple_of(4) && out_features != 1 {
            out_features.next_multiple_of(4)
        } else {
            out_features
        };
        Self {
            inner: LinearLikeInner::Dense(
                LinearConfig::new(in_features, padded).with_bias(false).init(device),
            ),
            out_features,
        }
    }
}