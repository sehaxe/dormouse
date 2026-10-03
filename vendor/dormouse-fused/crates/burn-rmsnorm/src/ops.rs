//! The fused RMSNorm as ONE tracked autodiff node (the `burn-gdn2`
//! `chunk_wy_forward_autodiff_s` pattern, 2 parents).
//!
//! Why this exists: burn's dispatch layer refuses to hand a bare primitive to a
//! tensor whose autodiff context is enabled (burn-dispatch
//! `src/tensor.rs:481-487`), so [`crate::fused::rmsnorm_cuda`] — correct on the
//! bare backend since `34c5631` — has never produced a number on ANY training
//! forward (`norm=0/N` in every trainer log). The way through is NOT relaxing
//! that guard: a bare result wrapped back into the autodiff graph would be a
//! LEAF, receive no gradient, and every loss curve would still look healthy —
//! the `8fa5d4c` defect class. The way through is the node: lift the caller's
//! tensors to their autodiff primitives (legal — we ask for the SAME backend
//! the context names), strip to bare for the kernel, and register the output
//! with an explicit backward. The backward here is tensor ops on the bare
//! backend (~10 launches); replacing it with a fused adjoint kernel is the
//! follow-up, and the honest gate for it is a step-time A/B, not this file.
//!
//! Gate (ADR-0019): `crate::fused::calls()`'s `asked - skipped` counts this
//! arm's launches the same as the bare arm's, because the module counts the
//! ask and this path falls through to the SAME skip line on any decline.

use burn::backend::{Backend, DispatchKindConversion};
use burn::tensor::{DispatchTensor, Tensor};
use burn_autodiff::checkpoint::base::Checkpointer;
use burn_autodiff::checkpoint::strategy::{
    BalancedCheckpointing, CheckpointStrategy, NoCheckpointing,
};
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};
use burn_autodiff::{Autodiff, NodeId};

use crate::cuda_dispatch::{autodiff_node, bare_from_node};

const N_PARENTS: usize = 2;

/// Backward of `y = x / sqrt(mean(x²) + eps) * w` over the [rows, d] view:
/// `dx_i = w_i·inv·gy_i − x_i·inv³/d · Σ_j w_j·gy_j·x_j` (inv per row),
/// `dw_i = Σ_rows gy_i·x_i·inv`. Pinned against burn's own autograd over the
/// tensor path in `tests/autodiff_node_cuda.rs` at 1e-5 relative.
#[derive(Debug)]
struct RmsNormFused;

impl<B: Backend> Backward<B, N_PARENTS> for RmsNormFused
where
    DispatchTensor: DispatchKindConversion<B>,
{
    type State = ([Option<NodeId>; N_PARENTS], f32);

    fn backward(
        self,
        ops: Ops<Self::State, N_PARENTS>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        let (ids, eps) = ops.state;
        // Both parents are checkpointed unconditionally, so the retrieval is
        // total and the expects name a broken invariant, not a possible state.
        let x = Tensor::<2>::from_primitive::<B>(
            checkpointer.retrieve_node_output(ids[0].expect("x is checkpointed")),
        );
        let w = Tensor::<1>::from_primitive::<B>(
            checkpointer.retrieve_node_output(ids[1].expect("w is checkpointed")),
        );
        let d_out = Tensor::<2>::from_primitive::<B>(grads.consume::<B>(&ops.node));

        // The backward runs on B = the BARE inner backend: `finish` builds the
        // node around a `FloatTensor<B>`, so every gradient registered below
        // must be a bare primitive — re-wrapping as `Autodiff<B, _>` is the
        // refusal `chunk_wy` measured ("Expected concrete Cube backend with
        // disabled autodiff context").
        let [_, d] = x.shape().dims::<2>();
        let inv = x
            .clone()
            .powf_scalar(2.0)
            .mean_dim(1)
            .add_scalar(eps as f64)
            .sqrt()
            .recip(); // [rows, 1]

        let gy_w = d_out.clone().mul(w.clone().reshape([1, d])); // gy ⊙ w
        let dot = gy_w.clone().mul(x.clone()).sum_dim(1); // [rows, 1]
        let dx = gy_w.mul(inv.clone())
            - x.clone()
                .mul(dot)
                .mul(inv.clone().powf_scalar(3.0).div_scalar(d as f64));
        let dw = d_out.mul(x).mul(inv).sum_dim(0).reshape([d]);

        if let Some(node) = ops.parents[0].clone() {
            grads.register::<B>(node.id, dx.try_into_primitive::<B>().unwrap());
        }
        if let Some(node) = ops.parents[1].clone() {
            grads.register::<B>(node.id, dw.try_into_primitive::<B>().unwrap());
        }
    }
}

/// The fused kernel as one tracked node, for a caller on `Autodiff<Inner, S>`.
///
/// `None` = decline: not this backend, not this strategy, nothing tracked, or
/// the kernel itself refused (d < [`crate::fused::MIN_FUSED_D`], non-f32, not
/// CUDA). The caller falls to the bare arm / tensor path and its counters say
/// so (ADR-0019) — a decline here is COUNTED, never silent.
pub fn rmsnorm_node_autodiff_s<Inner: Backend, S: CheckpointStrategy>(
    x: Tensor<2>,
    weight: Tensor<1>,
    eps: f32,
) -> Option<Tensor<2>>
where
    DispatchTensor: DispatchKindConversion<Autodiff<Inner, S>> + DispatchKindConversion<Inner>,
{
    let x_node = autodiff_node::<Inner, S, 2>(&x)?;
    let w_node = autodiff_node::<Inner, S, 1>(&weight)?;
    // NOT `is_require_grad()`: that is `requirement == Grad`, the strict
    // requirement of a LEAF, and the trainer's norm input is always the output
    // of a projection (`GradInBackward`). The same decision, read from the
    // same API burn reads it from — see the long note in
    // `burn-gdn2/src/autodiff.rs::chunk_wy_forward_autodiff_s`.
    if !(x_node.is_tracked() || w_node.is_tracked()) {
        return None;
    }

    let x_bare = bare_from_node::<Inner, S, 2>(&x_node);
    let w_bare = bare_from_node::<Inner, S, 1>(&w_node);
    let out = crate::fused::rmsnorm_launch::<Inner>(x_bare, w_bare, eps)?;
    let out_prim = out.try_into_primitive::<Inner>().unwrap();

    let nodes = [x_node.node(), w_node.node()];
    let prep = RmsNormFused.prepare::<S>(nodes);
    let out_adt = match prep.compute_bound().stateful() {
        OpsKind::Tracked(mut prep) => {
            let ids = [Some(prep.checkpoint(&x_node)), Some(prep.checkpoint(&w_node))];
            prep.finish((ids, eps), out_prim)
        }
        // Nothing tracked was declined above, so this arm is reachable only
        // through a race with a drop of the graph — the output is still the
        // correct fused value, there is just no backward to serve.
        OpsKind::UnTracked(prep) => prep.finish(out_prim),
    };
    Some(Tensor::from_primitive::<Autodiff<Inner, S>>(out_adt))
}

/// [`rmsnorm_node_autodiff_s`] for the TWO strategies a training stack can
/// hold. The strategy lives in the TYPE (`AutodiffTensor` does not carry
/// it — the same reason `chunk_wy_forward_autodiff_s` takes `S`), and a module
/// as deep as RMSNorm cannot know its caller's strategy, so both are tried;
/// a failed try is a type-level context compare, not a kernel launch.
pub fn rmsnorm_node_autodiff<Inner: Backend>(x: Tensor<2>, weight: Tensor<1>, eps: f32) -> Option<Tensor<2>>
where
    DispatchTensor: DispatchKindConversion<Autodiff<Inner, BalancedCheckpointing>>
        + DispatchKindConversion<Autodiff<Inner, NoCheckpointing>>
        + DispatchKindConversion<Inner>,
{
    rmsnorm_node_autodiff_s::<Inner, BalancedCheckpointing>(x.clone(), weight.clone(), eps)
        .or_else(|| rmsnorm_node_autodiff_s::<Inner, NoCheckpointing>(x, weight, eps))
}
