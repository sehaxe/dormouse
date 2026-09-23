//! Hard activation interventions and the lightweight probing head.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{activation, Device, Tensor};

/// Hard intervention in the style of the paper (`apply_scaling`): scale the
/// activations of `neuron_indices` by `scale` (0.0 ablate) and leave the
/// rest untouched. `x`: `[B, T, D]`, indices in `[0, D)`.
pub fn intervene(x: Tensor<3>, neuron_indices: &[usize], scale: f32) -> Tensor<3> {
    let [b, t, d] = x.dims();
    let device = x.device();
    let mut mask = vec![1.0f32; d];
    for &i in neuron_indices {
        mask[i] = scale;
    }
    let m = Tensor::<3>::from_data(burn::tensor::TensorData::new(mask, [1, 1, d]), &device)
        .expand([b, t, d]);
    x * m
}

/// Lightweight hallucination probing head.
///
/// Maps hidden states to per-token hallucination probability via a learned
/// linear projection: `sigmoid(h · W) ∈ [0, 1]`.
#[derive(Module, Debug)]
pub struct HallDetector {
    pub probe: Linear,
}

impl HallDetector {
    pub fn new(d_model: usize, device: &Device) -> Self {
        Self {
            probe: LinearConfig::new(d_model, 1).with_bias(false).init(device),
        }
    }

    pub fn logit(&self, h: Tensor<3>) -> Tensor<3> {
        self.probe.forward(h)
    }

    pub fn prob(&self, h: Tensor<3>) -> Tensor<3> {
        activation::sigmoid(self.probe.forward(h))
    }
}
