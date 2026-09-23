//! Expert-choice token router (paper section 3.4): scalar weight per token.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor};

/// Token-level router: a single linear projection `d -> 1` producing the
/// scalar weight `r_i = w_theta^T x_i` (paper section 3.4).
#[derive(Module, Debug)]
pub struct ModRouter {
    pub proj: Linear,
}

impl ModRouter {
    pub fn new(d_model: usize, device: &Device) -> Self {
        Self {
            proj: LinearConfig::new(d_model, 1).with_bias(false).init(device),
        }
    }

    /// Router weights `[B, T, 1]`, unbounded scalars.
    pub fn weights(&self, h: Tensor<3>) -> Tensor<3> {
        self.proj.forward(h)
    }
}
