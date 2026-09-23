//! # burn-antihall - Anti-Hallucination for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! Hallucination suppression (per-neuron sigmoid gates, domain-conditioned,
//! context-adaptive) and detection (probing head).
//!
//! | arXiv | Title |
//! |-------|-------|
//! | [2512.01797](https://arxiv.org/abs/2512.01797) | H-Neurons: <0.1% FFN neurons cause hallucinations |
//!
//! The official implementation (thunlp/H-Neurons) intervenes by scaling the
//! columns of the FFN `down_proj` for classifier-identified neurons
//! (`apply_scaling`, scale factor 0 = hard ablation). `HallSuppressor` is the
//! learned variant: per-neuron sigmoid gates, optionally conditioned on a
//! domain embedding and the context, trained end-to-end. `intervene` applies
//! the paper's hard scaling to a fixed set of neuron indices.
//! | [2604.19765](https://arxiv.org/abs/2604.19765) | Cross-domain transfer fails |
//! | [2607.00158](https://arxiv.org/abs/2607.00158) | Medical: readable but not controllable |
//! | [2512.18623](https://arxiv.org/abs/2512.18623) | LLM-CAS: dynamic perturbation (AAAI 2026) |
//! | [s41598-026-42981-3](https://doi.org/10.1038/s41598-026-42981-3) | HALL-OPT: detection at 94.3% (Nature 2026) |

mod detect;
mod suppress;

pub use detect::{intervene, HallDetector};
pub use suppress::HallSuppressor;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::activation;
    use burn::tensor::{Device, Distribution, Tensor};

    fn dev() -> Device {
        Device::ndarray()
    }

    fn flatten(x: Tensor<3>) -> Vec<f32> {
        x.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn flatten1(x: Tensor<1>) -> Vec<f32> {
        x.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn suppressor_shape() {
        let s = HallSuppressor::new(128, &dev());
        let x = Tensor::<3>::random([2, 16, 128], Distribution::Default, &dev());
        assert_eq!(s.forward(x).dims(), [2, 16, 128]);
    }

    #[test]
    fn gates_in_range() {
        let s = HallSuppressor::new(64, &dev());
        for v in flatten1(activation::sigmoid(s.gates.val())) {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn init_near_one() {
        let s = HallSuppressor::new(256, &dev());
        let vals = flatten1(activation::sigmoid(s.gates.val()));
        let avg = vals.iter().sum::<f32>() / vals.len() as f32;
        assert!(avg > 0.8, "init gate avg {avg}, expected >0.8");
    }

    #[test]
    fn domain_forward() {
        let s = HallSuppressor::new(64, &dev()).with_domain_proj(32, &dev());
        let x = Tensor::<3>::ones([1, 4, 64], &dev());
        let d = Tensor::<2>::ones([1, 32], &dev());
        let y = s.forward_domain(x, d);
        assert_eq!(y.dims(), [1, 4, 64]);
        assert!(flatten(y).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn domain_forward_batched() {
        // Regression: the gate used to reshape [B, H] to [1, 1, H], which
        // panics for B > 1. Batched input must broadcast cleanly.
        let s = HallSuppressor::new(64, &dev()).with_domain_proj(32, &dev());
        let x = Tensor::<3>::ones([4, 3, 64], &dev());
        let d = Tensor::<2>::ones([4, 32], &dev());
        let y = s.forward_domain(x, d);
        assert_eq!(y.dims(), [4, 3, 64]);
        let v = flatten(y);
        assert!(v.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn adaptive_forward() {
        let s = HallSuppressor::new(64, &dev())
            .with_domain_proj(32, &dev())
            .with_context_proj(&dev());
        let x = Tensor::<3>::ones([2, 4, 64], &dev());
        let d = Tensor::<2>::ones([2, 32], &dev());
        let c = Tensor::<3>::ones([2, 4, 64], &dev());
        let y = s.forward_adaptive(x, d, c);
        assert_eq!(y.dims(), [2, 4, 64]);
        assert!(flatten(y).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn detector_range() {
        let d = HallDetector::new(32, &dev());
        let h = Tensor::<3>::random([1, 8, 32], Distribution::Default, &dev());
        for v in flatten(d.prob(h)) {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn detector_shape() {
        let d = HallDetector::new(64, &dev());
        let h = Tensor::<3>::random([1, 16, 64], Distribution::Default, &dev());
        assert_eq!(d.logit(h).dims(), [1, 16, 1]);
    }
}
#[cfg(test)]
mod intervene_tests {
    use super::*;
    use burn::tensor::{Device, Tensor};
    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn intervene_scales_only_selected() {
        let x = Tensor::<3>::ones([2, 3, 4], &dev());
        let y = intervene(x, &[1], 0.0);
        let v: Vec<f32> = y.into_data().try_to_vec().unwrap();
        // every row: [1, 0, 1, 1]
        for row in v.chunks_exact(4) {
            assert_eq!(row, &[1.0, 0.0, 1.0, 1.0]);
        }
    }

    #[test]
    fn intervene_scale_factor() {
        let x = Tensor::<3>::ones([1, 1, 3], &dev());
        let y = intervene(x, &[0, 2], 0.5);
        let v: Vec<f32> = y.into_data().try_to_vec().unwrap();
        assert_eq!(v, vec![0.5, 1.0, 0.5]);
    }
}
