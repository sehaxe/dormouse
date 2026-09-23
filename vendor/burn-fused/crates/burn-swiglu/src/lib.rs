//! # burn-swiglu - SiLU-Gated Linear Unit for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! | Reference | What |
//! |-----------|------|
//! | Shazeer 2020 | `SiLU(x·W_gate) * (x·W_up)` - standard Transformer FFN |
//!
//! > Paper: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (Shazeer, 2020).
//! > Used in LLaMA, Mistral, Qwen, DeepSeek, Gemma.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{activation, Device, Tensor};

/// Apply gating: `SiLU(gate) * up` from a `[B, T, 2*hidden]` tensor.
///
/// Left half = gate, right half = value. Used by `SwiGLU.forward`.
pub fn swiglu_gate(gu: Tensor<3>) -> Tensor<3> {
    let [b, t, d2] = gu.dims();
    let hidden = d2 / 2;
    #[cfg(feature = "cuda")]
    {
        if let Some(out) = crate::fused::swiglu_cuda::<
            burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>,
        >(gu.clone().reshape([b * t, d2]), hidden)
        {
            return out.reshape([b, t, hidden]);
        }
    }
    let g = activation::silu(gu.clone().slice([0..b, 0..t, 0..hidden]));
    let u = gu.slice([0..b, 0..t, hidden..d2]);
    g.mul(u)
}

/// Standard SwiGLU feed-forward network.
///
/// ```text
/// x → Linear(d, 2h) → SwiGLU gate → Linear(h, d) → output
/// ```
///
/// LLaMA-style defaults: no bias on either projection (paper convention).
#[derive(Module, Debug)]
pub struct SwiGLU {
    pub gate_up: Linear,
    pub down: Linear,
    pub hidden: usize,
}

/// Configuration for [`SwiGLU`].
#[derive(Clone, Debug)]
pub struct SwiGLUConfig {
    pub d_model: usize,
    pub hidden: usize,
    pub bias: bool,
}

impl SwiGLUConfig {
    pub fn new(d_model: usize, hidden: usize) -> Self {
        Self {
            d_model,
            hidden,
            bias: false,
        }
    }

    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    pub fn init(&self, device: &Device) -> SwiGLU {
        SwiGLU {
            gate_up: LinearConfig::new(self.d_model, 2 * self.hidden)
                .with_bias(self.bias)
                .init(device),
            down: LinearConfig::new(self.hidden, self.d_model)
                .with_bias(self.bias)
                .init(device),
            hidden: self.hidden,
        }
    }
}

impl SwiGLU {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.down.forward(swiglu_gate(self.gate_up.forward(x)))
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
    fn gate_shape() {
        let gu = Tensor::<3>::random([2, 8, 128], Distribution::Default, &dev());
        assert_eq!(swiglu_gate(gu).dims(), [2, 8, 64]);
    }

    #[test]
    fn forward_shape() {
        let layer = SwiGLUConfig::new(64, 128).init(&dev());
        let x = Tensor::<3>::random([2, 8, 64], Distribution::Default, &dev());
        assert_eq!(layer.forward(x).dims(), [2, 8, 64]);
    }
}
pub mod fused;
