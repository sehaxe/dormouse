//! Shared predictor head: LayerNorm -> Linear (data2vec 2.0 style).

use burn::module::Module;
use burn::nn::LayerNorm;
use burn::nn::Linear;
use burn::tensor::{Device, Tensor};

/// Lightweight predictor head (norm -> linear), shared across layers
/// (data2vec 2.0 uses a single head, unlike v1's per-layer heads).
#[derive(Module, Debug)]
pub struct JepaPredictor {
    pub proj: Linear,
    pub norm: LayerNorm,
}

impl JepaPredictor {
    pub fn new(d_model: usize, device: &Device) -> Self {
        let proj = burn::nn::LinearConfig::new(d_model, d_model)
            .with_bias(false)
            .init(device);
        let norm = burn::nn::LayerNormConfig::new(d_model).init(device);
        Self { proj, norm }
    }

    pub fn forward(&self, h: Tensor<3>) -> Tensor<3> {
        let h = self.norm.forward(h);
        self.proj.forward(h)
    }
}
