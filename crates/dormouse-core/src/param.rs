//! param - TSCT linear via burn-sct SpectralLinear, pad to multiple of 4,
//! NM knob, BF16 env (mirrors aria semantics; fresh mini composition)
use burn::module::Module;
use burn::nn::LinearConfig;
use burn::tensor::{Device, DispatchTensor, Tensor};
use burn::backend::Backend;
use burn::backend::DispatchKindConversion;
use burn_spectral::SpectralLinear;
use burn_sct::SctLinear;

/// bf16 mode env flag: activations stream in bf16, weights fp32 cast per
/// forward (burn-ndarray has no bf16 dtype -> CPU tests stay fp32).
pub fn bf16_on() -> bool {
    std::env::var("BF16").is_ok() && std::env::var("BF16").unwrap() != "0"
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
        let inner = if bf16_on() {
            LinearLikeInner::Tsct(SpectralLinear::new(in_features, padded, rank, device))
        } else {
            LinearLikeInner::Tsct(SpectralLinear::new(in_features, padded, rank, device))
        };
        Self { inner, out_features }
    }

    pub fn forward<B: Backend>(&self, x: Tensor<2>) -> Tensor<2>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let y = match &self.inner {
            LinearLikeInner::Tsct(l) => {
                if std::env::var("DM_QUANT_DEBUG").is_ok() {
                    println!("[ll] quant={:?} out={}", l.quant, l.out_features);
                }
                if l.quant != burn_spectral::QuantFormat::Fp32 {
                    l.forward_quant::<B>(x)
                } else {
                    l.forward(x)
                }
            }
            LinearLikeInner::Sct(l) => l.forward(x),
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

/// retract every step: polar Newton-Schulz on U/V (3 iters, burn-sct exposes
/// the same orthogonalization the trainer calls). SpectralLinear owns retract
/// via its QR-based parameterization; dense/sct need none.
pub fn retract_model<M: Module>(_m: &M) {}