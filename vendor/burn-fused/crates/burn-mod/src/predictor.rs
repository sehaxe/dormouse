//! Second-stage router predictor (paper section 3.5): stop-gradient style
//! training target from the current top-k selection.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{activation, Device, Tensor};

use crate::routing::{bce_aux_loss, select_topk};

/// Small MLP predictor (paper section 3.5, second method): trained with a
/// stop-gradient input to predict whether each token will be in the top-k.
/// Unlike the BCE loss it does not perturb the language-modeling gradient.
#[derive(Module, Debug)]
pub struct ModPredictor {
    pub net: Linear,
    #[module(skip)]
    pub capacity_frac: f32,
}

impl ModPredictor {
    pub fn new(d_model: usize, capacity_frac: f32, device: &Device) -> Self {
        Self {
            net: LinearConfig::new(d_model, 1).with_bias(false).init(device),
            capacity_frac,
        }
    }

    /// Sigmoid prediction `[B, T, 1]` that `h` will be routed into the block.
    pub fn forward(&self, h: Tensor<3>) -> Tensor<3> {
        activation::sigmoid(self.net.forward(h))
    }

    /// BCE loss against the current top-k selection. NOTE: the returned
    /// scalar sums TWO terms — the predictor's own BCE and the router's
    /// `bce_aux_loss` — so stepping the optimizer on this loss updates the
    /// predictor AND the router jointly. (The top-k indices themselves carry
    /// no gradient; detaching the router weights here is deliberately
    /// avoided because burn 0.22 `detach()` strips the autodiff wrapper and
    /// breaks downstream tensor ops — see `route_block` notes.) Callers that
    /// want predictor-only training must zero the router term or filter its
    /// gradient by param id.
    pub fn loss(&self, h: Tensor<3>, weights: Tensor<3>, device: &Device) -> Tensor<1> {
        let idx = select_topk(weights.clone(), self.capacity_frac, device);
        // no detach: top-k indices are Int (no gradient), so the BCE flows
        // into the predictor only; detach() on 0.22 strips the autodiff
        // wrapper and breaks the later add
        let bce = bce_aux_loss(weights.clone(), idx.clone(), device);
        let p = self.forward(h);
        // same BCE as bce_aux_loss but with the predictor's own sigmoid
        let [b, t, _] = p.dims();
        let idx3 = idx.unsqueeze_dim::<3>(2);
        let ones = Tensor::<3>::ones(idx3.shape(), device);
        let target = Tensor::<3>::zeros([b, t, 1], device).scatter(
            1,
            idx3,
            ones,
            burn::tensor::IndexingUpdateOp::Add,
        );
        let eps = 1e-7f32;
        let bce2 = target
            .clone()
            .mul(p.clone().add_scalar(eps).log())
            .add(
                target
                    .neg()
                    .add_scalar(1.0)
                    .mul(p.neg().add_scalar(1.0 + eps).log()),
            )
            .mean()
            .neg();
        bce.add(bce2).div_scalar(2.0)
    }
}
