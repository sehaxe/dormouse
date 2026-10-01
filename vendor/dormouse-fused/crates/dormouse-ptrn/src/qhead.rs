//! Learned value head Q(h): sigmoid score used for Best-Q@K selection.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{activation, Device, Int, Tensor};

/// Learned Q-head (value head): `h → Linear(d, 1) → logit → sigmoid → P(correct)`.
///
/// Trained with `q_loss` against a binary correctness target; at inference the
/// per-rollout Q scores drive `best_of_k` selection.
#[derive(Module, Debug)]
pub struct QHead {
    proj: Linear,
}

impl QHead {
    pub fn new(d_model: usize, device: &Device) -> Self {
        Self {
            proj: LinearConfig::new(d_model, 1).init(device),
        }
    }

    /// Raw pre-sigmoid score: `[B,T,d] → [B,T,1]`.
    pub fn logit(&self, h: Tensor<3>) -> Tensor<3> {
        self.proj.forward(h)
    }

    /// Calibrated correctness probability: `[B,T,1] ∈ [0,1]`.
    pub fn prob(&self, h: Tensor<3>) -> Tensor<3> {
        activation::sigmoid(self.logit(h))
    }

    /// Masked BCE between `sigmoid(logits)` and the binary correctness target.
    ///
    /// `logits`: `[B,T,1]` — `target`: `[B,T]` Int (1 = prediction matched truth).
    /// `mask`: `[B,T]` (1.0 = supervised position). Returns mean BCE over
    /// masked positions, scalar `[1]`.
    #[allow(clippy::unused_self)]
    pub fn q_loss(
        &self,
        logits: Tensor<3>,
        target: Tensor<2, Int>,
        mask: Option<Tensor<2>>,
    ) -> Tensor<1> {
        let p = activation::sigmoid(logits)
            .squeeze_dim::<2>(2)
            .clamp(1e-7, 1.0 - 1e-7);
        let t = target.float();
        let bce = -(t.clone() * p.clone().log()
            + t.clone().neg().add_scalar(1.0) * p.clone().neg().add_scalar(1.0).log()); // [B,T]
        match mask {
            // masked mean: average bce over supervised positions only
            Some(m) => (bce * m.clone()).sum().div(m.sum().clamp_min(1.0)),
            None => bce.mean(),
        }
    }
}
