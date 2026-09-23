//! Mixture-of-Recursions (MoR) routing primitives for Burn.
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! Implements the expert-choice routing mechanism from
//! *"Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive
//! Token-Level Computation"* (Bae et al., 2025, [arXiv:2507.10524](https://arxiv.org/abs/2507.10524)).
//!
//! MoR reuses a single shared stack of layers across recursion steps while a
//! lightweight router assigns each token its own recursion depth. This crate
//! provides the pure-tensor routing pieces:
//!
//! - [`MoRRouter`] — linear d→1 importance scorer.
//! - [`select_active`] — top-k active / inactive token split for one recursion
//!   step (expert-choice routing, hierarchical filtering per the paper).
//! - [`gather_active`] / [`scatter_active`] — move the hidden states of active
//!   tokens into a dense active batch and place the block output back, so the
//!   quadratic attention and FFN run only on active tokens.
//! - [`load_balancing_loss`] — auxiliary loss that keeps the router from
//!   collapsing (all tokens assigned the same depth).
//!
//! All functions are pure tensor ops (no host branching, no `into_data`), so
//! they work on any `Backend` and compose with `Autodiff`.
//!
//! # Usage sketch
//!
//! ```
//! use burn::tensor::Tensor;
//! use burn_mor::{MoRConfig, MoRRouter, gather_active, load_balancing_loss, scatter_active, select_active};
//!
//! fn mor_step(
//!     block: impl Fn(Tensor<3>) -> Tensor<3>,
//!     router: &MoRRouter,
//!     h: Tensor<3>,
//!     config: MoRConfig,
//! ) -> (Tensor<3>, Tensor<1>) {
//!     let device = h.device();
//!     let scores = router.scores(h.clone());
//!     let (active, _inactive) = select_active(scores.clone(), config.keep_frac, &device);
//!     let n = active.dims()[1];
//!
//!     // Heavy computation (attention + FFN) runs only on active tokens.
//!     let out_active = block(gather_active(h.clone(), active.clone()));
//!
//!     // Dropped tokens get zero block output -> residual-only pass-through.
//!     let out = scatter_active(Tensor::zeros_like(&h), active.clone(), out_active, &device);
//!     let h_next = h.add(out);
//!
//!     let aux = load_balancing_loss(scores, active, n).mul_scalar(config.aux_weight);
//!     (h_next, aux)
//! }
//! ```

mod loss;
mod routing;
mod topk;

pub use loss::load_balancing_loss;
pub use routing::{
    gather_active, scatter_active, select_active, select_active_only, MoRConfig, MoRRouter,
};
pub use topk::topk_indices;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::activation::sigmoid;
    use burn::tensor::{Device, Distribution, IndexingUpdateOp, Tensor};

    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn router_scores_shape_and_finite() {
        let r = MoRRouter::new(64, &dev());
        let h = Tensor::<3>::random([2, 16, 64], Distribution::Default, &dev());
        let s = r.scores(h);
        assert_eq!(s.dims(), [2, 16, 1]);
        let v: Vec<f32> = s.into_data().to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "router scores not finite");
    }

    #[test]
    fn topk_indices_descending_exact() {
        // Distinct values: exact top-k set, descending order, per row.
        let x = Tensor::<2>::from_floats(
            [[1.0, 5.0, 3.0, 9.0, 2.0], [8.0, 0.0, 7.0, 4.0, 6.0]],
            &dev(),
        );
        let idx = topk_indices(x, 3, 1);
        let v: Vec<i64> = idx.into_data().to_vec().unwrap();
        assert_eq!(v, vec![3, 1, 2, 0, 2, 4]);
        // k = 0 -> empty, k = n-1 -> all but the minimum.
        let e = topk_indices(Tensor::<2>::from_floats([[2.0, 1.0]], &dev()), 0, 1);
        assert_eq!(e.dims(), [1, 0]);
        let all = topk_indices(Tensor::<2>::from_floats([[2.0, 1.0, 3.0]], &dev()), 2, 1);
        let a: Vec<i64> = all.into_data().to_vec().unwrap();
        assert_eq!(a, vec![2, 0]);
        // Wide row beyond the old argtopk take<=16 cap: exact top-k set,
        // descending order (all values distinct, no tie-order dependence).
        let vals: Vec<f32> = (0..64).map(|i| (63 - i) as f32).collect(); // [63, 62, .., 0]
        let big = Tensor::<2>::from_data(burn::tensor::TensorData::new(vals, [1, 64]), &dev());
        let idx = topk_indices(big.clone(), 40, 1);
        let v: Vec<i64> = idx.into_data().to_vec().unwrap();
        assert_eq!(v, (0..40).collect::<Vec<i64>>()); // top-40 = 63..24
        let idx2 = topk_indices(big, 5, 1);
        let v2: Vec<i64> = idx2.into_data().to_vec().unwrap();
        assert_eq!(v2, (0..5).collect::<Vec<i64>>());
    }

    #[test]
    fn select_active_partitions_all_tokens() {
        // active + inactive must be exactly 0..T, disjoint (k clamped to [1, T-1]).
        let scores = Tensor::<3>::random([1, 8, 1], Distribution::Default, &dev());
        let (active, inactive) = select_active(scores, 0.5, &dev());
        assert_eq!(active.dims(), [1, 4]);
        assert_eq!(inactive.dims(), [1, 4]);
        let a: Vec<i64> = active.into_data().to_vec().unwrap();
        let i: Vec<i64> = inactive.into_data().to_vec().unwrap();
        let mut both = a.clone();
        both.extend(i.clone());
        both.sort();
        assert_eq!(
            both,
            (0..8).collect::<Vec<i64>>(),
            "active+inactive != all tokens"
        );
        assert!(a.iter().all(|x| !i.contains(x)), "overlap between sets");
    }

    #[test]
    fn select_active_single_token() {
        // t=1 used to panic (clamp(1, t-1) = clamp(1, 0)); the single token
        // must be active, inactive empty
        let scores = Tensor::<3>::random([2, 1, 1], Distribution::Default, &dev());
        let (active, inactive) = select_active(scores, 0.5, &dev());
        assert_eq!(active.dims(), [2, 1]);
        assert_eq!(inactive.dims(), [2, 0]);
        let a: Vec<i64> = active.into_data().to_vec().unwrap();
        assert_eq!(a, vec![0, 0]);
        let i: Vec<i64> = inactive.into_data().to_vec().unwrap();
        assert!(i.is_empty(), "inactive should be empty");
    }

    #[test]
    fn gather_scatter_roundtrip() {
        let (b, t, d, k) = (2usize, 8usize, 4usize, 3usize);
        let h = Tensor::<3>::random([b, t, d], Distribution::Default, &dev());
        let scores = Tensor::<3>::random([b, t, 1], Distribution::Default, &dev());
        let active = topk_indices(scores.reshape([b, t]), k, 1);
        let gathered = gather_active(h.clone(), active.clone());
        assert_eq!(gathered.dims(), [b, k, d]);
        let out = scatter_active(
            Tensor::zeros_like(&h),
            active.clone(),
            gathered.clone(),
            &dev(),
        );
        // Re-gathering the scattered output at the active positions recovers
        // the gathered rows exactly (scatter into zeros is a pure placement).
        let back = gather_active(out, active);
        let diff: f32 = (back - gathered).abs().max().into_scalar();
        assert!(diff < 1e-6, "scatter/gather roundtrip mismatch {diff}");
    }

    #[test]
    fn load_balancing_loss_finite_nonneg() {
        let scores = Tensor::<3>::random([2, 8, 1], Distribution::Default, &dev());
        let active = topk_indices(scores.clone().reshape([2, 8]), 3, 1);
        let loss = load_balancing_loss(scores, active, 3);
        let v: f32 = loss.into_scalar();
        assert!(v.is_finite() && v >= 0.0, "loss {v} not finite/nonneg");
    }

    #[test]
    fn load_balancing_loss_equals_scatter_mask() {
        let scores = Tensor::<3>::random([2, 8, 1], Distribution::Default, &dev());
        let active = topk_indices(scores.clone().reshape([2, 8]), 3, 1);

        let new = load_balancing_loss(scores.clone(), active.clone(), 3);

        let [b, t, _] = scores.dims();
        let frac = 3.0_f32 / (t as f32);
        let idx3 = active.unsqueeze_dim::<3>(2); // [B, K, 1]
        let ones = Tensor::<3>::ones(idx3.shape(), &dev());
        let counts =
            Tensor::<3>::zeros([b, t, 1], &dev()).scatter(1, idx3, ones, IndexingUpdateOp::Add);
        let old = sigmoid(scores)
            .sub_scalar(frac)
            .mul(counts)
            .powf_scalar(2.0)
            .mean();

        let (n, o): (f32, f32) = (new.into_scalar(), old.into_scalar());
        assert!((n - o).abs() < 1e-5, "gather loss {n} != scatter loss {o}");
    }
}
