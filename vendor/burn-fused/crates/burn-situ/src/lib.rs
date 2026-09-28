//! # burn-situ - SiTU-GLU Activation for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! Sigmoid-Tanh Unit gated linear - bounded GLU variant from Kimi K3.
//! Applies `softcap(x, beta) = beta * tanh(x / beta)` to gate and up
//! branches before gating, preventing activation explosion.
//!
//! | Paper | What |
//! |-------|------|
//! | [Kimi K3](https://arxiv.org/abs/2607.24653) (Moonshot, 2026) | SiTU-GLU bounded GLU with tanh soft-caps |
//!
//! Benchmarked in Kimi K3 (2.8T MoE, 104B active): better stability
//! than SwiGLU for deep models and low-bit quantization (MXFP4).
use burn::tensor::{activation, Tensor};

// `autodiff` alone also compiles this module, deliberately: the fused ADJOINT
// and its strategy seam live here, and a gate that can only be compile-checked
// with a GPU is how a `NoCheckpointing`-only entry survived review. The
// kernels inside stay `#[cfg(feature = "cuda")]`. `pub` because the seam
// counters are the arm's only externally visible evidence (ADR-0019).
#[cfg(any(feature = "cuda", feature = "autodiff"))]
pub mod fused_situ;

/// Soft-cap: `beta * tanh(x / beta)`
///
/// Bounded near ±beta, linear near 0. From Kimi K3.
pub fn softcap(x: Tensor<2>, beta: f64) -> Tensor<2> {
    x.div_scalar(beta).tanh().mul_scalar(beta)
}

/// SiTU-GLU: bounded gated linear unit.
///
/// ```text
/// gate, up = split(gate_up)
/// gate_cap = softcap(gate, beta_gate)
/// up_cap   = softcap(up, beta_up)
/// gate_act = gate_cap * sigmoid(gate)   // Swish factor on the RAW pre-activation
/// return gate_act * up_cap
/// ```
///
/// Per Kimi K3 Eq (12): `beta1*tanh(Wg x/beta1) ⊙ sigmoid(Wg x) ⊙ beta2*tanh(Wu x/beta2)`.
/// The sigmoid must act on the uncapped gate so the negative tail vanishes
/// (Swish-like), which is the paper's stated design goal.
///
/// `gate_up`: `[N, 2*hidden]` - concatenated gate and up projections
/// `hidden`: size of each half
/// `beta_gate`, `beta_up`: soft-cap thresholds (default: 1.0)
pub fn situ_glu(gate_up: Tensor<2>, hidden: usize, beta_gate: f64, beta_up: f64) -> Tensor<2> {
    let [n, _d2] = gate_up.dims();

    #[cfg(all(feature = "autodiff", feature = "cuda"))]
    {
        // Probe BOTH checkpointing strategies. `Autodiff<Inner>`'s second type
        // parameter defaults to `NoCheckpointing` and the downcast compares the
        // whole backend type, so probing only that one silently sent a
        // `BalancedCheckpointing` caller (dormouse's backend) to the ~7-pass
        // tensor path: the right answer, slowly, with nothing counting it.
        use burn_autodiff::checkpoint::strategy::{BalancedCheckpointing, NoCheckpointing};
        type CudaBare = burn_cubecl::CubeBackend;
        if let Some(out) = crate::fused_situ::situ_glu_autodiff_s::<CudaBare, NoCheckpointing>(
            gate_up.clone(),
            hidden,
            beta_gate,
            beta_up,
        ) {
            return out;
        }
        if let Some(out) = crate::fused_situ::situ_glu_autodiff_s::<
            CudaBare,
            BalancedCheckpointing,
        >(gate_up.clone(), hidden, beta_gate, beta_up)
        {
            return out;
        }
    }
    #[cfg(feature = "cuda")]
    if let Some(out) = crate::fused_situ::situ_glu_cuda(&gate_up, hidden, beta_gate, beta_up) {
        return out;
    }

    let gate = gate_up.clone().slice([0..n, 0..hidden]);
    let up = gate_up.slice([0..n, hidden..2 * hidden]);

    let gate_cap = softcap(gate.clone(), beta_gate);
    let up_cap = softcap(up, beta_up);
    gate_cap.mul(activation::sigmoid(gate)).mul(up_cap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Distribution};
    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn softcap_bounded() {
        let vals = vec![100.0f32, -100.0f32, 0.5f32, -0.5f32];
        let x = Tensor::<1>::from_floats(vals.as_slice(), &dev()).reshape([4, 1]);
        let capped = softcap(x, 1.0);
        let vals: Vec<f32> = capped
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(vals[0].abs() < 1.1, "tanh(100) should be ~1");
        assert!(vals[2] > 0.0, "near zero should be linear");
    }
    #[test]
    fn situ_shape() {
        let gu = Tensor::<2>::random([16, 128], Distribution::Default, &dev());
        assert_eq!(situ_glu(gu, 64, 1.0, 1.0).dims(), [16, 64]);
    }
    #[test]
    fn situ_finite() {
        let gu = Tensor::<2>::random([32, 256], Distribution::Default, &dev());
        let out = situ_glu(gu, 128, 1.0, 1.0);
        let vals: Vec<f32> = out
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(vals.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn situ_negative_gate_tail_vanishes() {
        // Paper Eq (12): sigmoid acts on the RAW gate, so a large negative
        // gate pre-activation makes the output -> 0 (Swish tail), instead of
        // sigmoid(capped) which leaves a ~ -0.27 residual at beta=1.
        let gu = Tensor::<1>::from_floats(vec![-50.0f32, 50.0, 50.0, 50.0].as_slice(), &dev())
            .reshape([2, 2]);
        let out = situ_glu(gu, 1, 1.0, 1.0);
        let vals: Vec<f32> = out
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(
            vals[0].abs() < 1e-3,
            "gate=-50 should vanish (Swish tail), got {}",
            vals[0]
        );
        assert!(
            (vals[1] - 1.0).abs() < 1e-3,
            "gate=+50 should saturate near +1, got {}",
            vals[1]
        );
    }
}
