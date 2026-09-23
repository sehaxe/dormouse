//! # burn-mod - Mixture-of-Depths for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! Port of [Mixture-of-Depths: Dynamically allocating compute in
//! transformer-based language models](https://arxiv.org/abs/2404.02258)
//! (Raposo et al., Google DeepMind 2024).
//!
//! MoD caps the number of tokens that may enter a block's computation:
//! the router emits a scalar weight per token, the `k` highest-weighted
//! tokens (expert-choice routing) go through self-attention + MLP, the rest
//! pass through a residual connection. The block output is multiplied by
//! the (unbounded) router weight, putting the router on the gradient path
//! (paper eq. 1):
//!
//! ```text
//! x_i^{l+1} = r_i * f(X~^l) + x_i   if r_i > P_beta(R^l)
//! x_i^{l+1} = x_i                   otherwise
//! ```
//!
//! Expert-choice top-k needs no auxiliary balancing loss. Since top-k is
//! non-causal, the paper adds a binary-cross-entropy auxiliary loss that
//! centers `sigmoid(r)` around 0.5 so autoregressive sampling can route on
//! `r > 0.5` alone, or a small MLP predictor (second router) trained on the
//! same targets with a stop-gradient input (section 3.5, "Sampling").

/// Mixture-of-Depths configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModConfig {
    /// Fraction of tokens that may enter the block's computation
    /// (paper: capacity C = round(capacity_frac * T), T = seq len).
    pub capacity_frac: f32,
}

impl Default for ModConfig {
    fn default() -> Self {
        // paper: 12.5% worked best with routing every other block
        Self {
            capacity_frac: 0.125,
        }
    }
}

mod predictor;
mod router;
mod routing;

pub use predictor::ModPredictor;
pub use router::ModRouter;
pub use routing::{bce_aux_loss, gather_selected, route_block, select_topk};

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Param;
    use burn::tensor::{Device, Int, Tensor};
    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn select_topk_picks_highest() {
        let w = Tensor::<3>::from_data(
            burn::tensor::TensorData::new(
                vec![0.1f32, 0.9, 0.5, 0.8, 0.2, 0.7, 0.3, 0.4],
                [2, 4, 1],
            ),
            &dev(),
        );
        let idx = select_topk(w, 0.5, &dev());
        let v: Vec<i64> = idx.into_data().to_vec().unwrap();
        // row 0: top 2 of {0.1,0.9,0.5,0.8} -> {1,3}; row 1: {0.2,0.7,0.3,0.4} -> {1,3}
        assert_eq!(&v[0..2], &[1, 3]);
        assert_eq!(&v[2..4], &[1, 3]);
    }

    #[test]
    fn route_block_residual_and_scaling() {
        // router weights set so token 1 is routed; block = identity * 2
        let x = Tensor::<3>::ones([1, 3, 2], &dev());
        let router = ModRouter::new(2, &dev());
        // force weights: proj is linear w/o bias; set weights manually via
        // a zero-weight projection + scaled input is awkward, so test the
        // routing mechanics with a big margin instead: build a router whose
        // proj has known weights by replacing the tensor.
        // Simpler: construct with linear and override param values.
        let mut router = router;
        // weight [1, 2]: [10.0, 0.0] -> r = 10*x0, so token with x0=1 wins
        router.proj.weight = Param::from_tensor(Tensor::<2>::from_data(
            burn::tensor::TensorData::new(vec![10.0f32, 0.0], [2, 1]),
            &dev(),
        ));
        // ndarray scatter sanity: one index repeated across D must write
        // once (burn-mor relies on the same pattern on CUDA)
        let one =
            Tensor::<2, Int>::from_data(burn::tensor::TensorData::new(vec![0i64], [1, 1]), &dev());
        let idx3 = one.unsqueeze_dim::<3>(2).expand([1, 1, 2]);
        let vals = Tensor::<3>::ones([1, 1, 2], &dev()).mul_scalar(20.0);
        let sc = Tensor::<3>::zeros([1, 3, 2], &dev()).scatter(
            1,
            idx3,
            vals,
            burn::tensor::IndexingUpdateOp::Add,
        );
        let sv: Vec<f32> = sc.into_data().to_vec().unwrap();
        assert_eq!(
            &sv[0..2],
            &[20.0, 20.0],
            "scatter should hit position 0 only, got {sv:?}"
        );
        assert_eq!(&sv[2..4], &[0.0, 0.0]);
        // trace internals
        let r = router.weights(x.clone());
        let idx = select_topk(r.clone(), 1.0 / 3.0, &dev());
        let xa = gather_selected(x.clone(), idx.clone());
        let iv: Vec<i64> = idx.into_data().to_vec().unwrap();
        let rv: Vec<f32> = r.into_data().to_vec().unwrap();
        let xav: Vec<f32> = xa.into_data().to_vec().unwrap();
        println!("idx={iv:?} r={rv:?} xa={xav:?}");
        let out = route_block(x.clone(), &router, 1.0 / 3.0, |a| a.mul_scalar(2.0), &dev());
        let v: Vec<f32> = out.into_data().to_vec().unwrap();
        // token 0 (d=2): routed, out = 1 + 10*(1*2) = 21
        // tokens 1,2: not routed, out = 1
        assert!((v[0] - 21.0).abs() < 1e-4, "token0 {}", v[0]);
        assert!((v[1] - 21.0).abs() < 1e-4, "v = {v:?}");
        assert!((v[2] - 1.0).abs() < 1e-4);
        assert!((v[3] - 1.0).abs() < 1e-4);
        assert!((v[4] - 1.0).abs() < 1e-4);
        assert!((v[5] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn bce_loss_penalizes_mismatch() {
        let dev = dev();
        let w = Tensor::<3>::from_data(
            burn::tensor::TensorData::new(vec![0.1f32, 0.9, 0.5, 0.8], [1, 4, 1]),
            &dev,
        );
        let idx = select_topk(w.clone(), 0.5, &dev);
        let l = bce_aux_loss(w, idx, &dev);
        let v: f32 = l.into_scalar();
        assert!(v.is_finite() && v > 0.0);
    }

    #[test]
    fn predictor_learns_targets() {
        // No autodiff needed: nudge the predictor's weight toward "routed"
        // (the top-k token) and verify the loss drops. The BCE is minimized
        // when p -> 1 for the routed token, so increasing the weight (and
        // with it sigmoid(p)) must reduce the loss. (Autodiff + scatter of
        // Int indices is broken on 0.22, so no backward here.)
        let dev = dev();
        let mut predictor: ModPredictor = ModPredictor::new(4, 0.5, &dev);
        let w = Tensor::<3>::from_data(
            burn::tensor::TensorData::new(vec![0.9f32, 0.1], [1, 2, 1]),
            &dev,
        );
        // h differs per row, so the first weight row only affects the
        // routed token (row 0): raising it must raise p_0 -> target 1 and
        // lower the BCE, while p_1 stays put.
        let h = Tensor::<3>::from_data(
            burn::tensor::TensorData::new(
                vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1, 2, 4],
            ),
            &dev,
        );
        let l0: f32 = predictor.loss(h.clone(), w.clone(), &dev).into_scalar();
        let mut w_row = predictor.net.weight.val();
        let add = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(vec![1.0f32, 0.0, 0.0, 0.0], [4, 1]),
            &dev,
        );
        w_row = w_row.add(add);
        predictor.net.weight = Param::from_tensor(w_row);
        let l1: f32 = predictor.loss(h, w, &dev).into_scalar();
        assert!(
            l1 < l0,
            "loss must drop when p grows toward the routed target: {l0} -> {l1}"
        );
    }
}
