//! Learned per-neuron hallucination suppression gates (static / domain /
//! context-adaptive).

use burn::module::{Module, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{activation, Device, Tensor};

/// Per-neuron hallucination suppressor with three levels of adaptation.
///
/// ```text
/// forward(x)               → x * sigmoid(gates)                    static
/// forward_domain(x, d)     → x * sigmoid(gates + d·W_d)            domain-aware
/// forward_adaptive(x, d, c) → x * sigmoid(gates + d·W_d + c·W_c)   context-adaptive
/// ```
///
/// Gates initialized at `sigmoid(2.0) ≈ 0.88` - minimal suppression
/// at start to preserve fluency. Neurons drift toward zero during training
/// only when causal evidence warrants it.
#[derive(Module, Debug)]
pub struct HallSuppressor {
    pub gates: Param<Tensor<1>>,
    domain_proj: Option<Linear>,
    context_proj: Option<Linear>,
    hidden: usize,
}

impl HallSuppressor {
    pub fn new(hidden: usize, device: &Device) -> Self {
        Self {
            gates: Param::initialized(ParamId::new(), Tensor::full([hidden], 2.0, device)),
            domain_proj: None,
            context_proj: None,
            hidden,
        }
    }

    pub fn with_domain_proj(mut self, domain_dim: usize, device: &Device) -> Self {
        self.domain_proj = Some(
            LinearConfig::new(domain_dim, self.hidden)
                .with_bias(false)
                .init(device),
        );
        self
    }

    pub fn with_context_proj(mut self, device: &Device) -> Self {
        self.context_proj = Some(
            LinearConfig::new(self.hidden, 1)
                .with_bias(false)
                .init(device),
        );
        self
    }

    /// Static per-neuron suppression: `x * sigmoid(gates)`.
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let gate = activation::sigmoid(self.gates.val());
        x * gate.reshape([1, 1, self.hidden])
    }

    /// Domain-conditioned: `x * sigmoid(gates + domain·W_d)`.
    ///
    /// `domain_emb`: `[B, D_domain]` - projects to per-neuron gate offsets,
    /// broadcast over the time axis (any batch size).
    pub fn forward_domain(&self, x: Tensor<3>, domain_emb: Tensor<2>) -> Tensor<3> {
        let base = self.gates.val().reshape([1, self.hidden]);
        let dm = self
            .domain_proj
            .as_ref()
            .expect("call with_domain_proj() before forward_domain()")
            .forward(domain_emb);
        // sigmoid(base + dm) is [B, hidden]; broadcast over T via unsqueeze.
        // (A reshape to [1, 1, H] here would panic for B > 1.)
        let gate = activation::sigmoid(base + dm).unsqueeze_dim::<3>(1);
        x * gate
    }

    /// Context-adaptive: `x * sigmoid(gates + domain·W_d + context·W_c)`.
    ///
    /// `domain_emb`: `[B, D_domain]`, `context`: `[B, T, hidden]`.
    pub fn forward_adaptive(
        &self,
        x: Tensor<3>,
        domain_emb: Tensor<2>,
        context: Tensor<3>,
    ) -> Tensor<3> {
        let base = self.gates.val().reshape([1, 1, self.hidden]);
        let dm = self
            .domain_proj
            .as_ref()
            .expect("call with_domain_proj() before forward_adaptive()")
            .forward(domain_emb)
            .unsqueeze_dim::<3>(1);
        let ct = self
            .context_proj
            .as_ref()
            .expect("call with_context_proj() before forward_adaptive()")
            .forward(context);
        x * activation::sigmoid(base + dm + ct)
    }
}
