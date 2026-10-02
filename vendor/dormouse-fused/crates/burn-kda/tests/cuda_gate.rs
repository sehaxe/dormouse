//! The production wiring: `KdaModule::forward_train_state` on the trainer's
//! backend (`Autodiff<CudaBare, BalancedCheckpointing>`) must produce a
//! GRADIENT, and the seam counters must say which arm produced it.
//!
//! # What changed on 2026-09-28, and why the assertions moved
//!
//! This test used to assert `fused_calls().0 > 0` - "the fused kernels ran" -
//! and it now asserts the opposite of that on this fixture, on purpose.
//!
//! The trainer hands the chunk op seven tensors that are all RESULTS of
//! projections, activations, a permute and a repeat (`KdaModule::project`).
//! burn reports `is_require_grad()` only for a LEAF marked `require_grad`
//! (`float_is_require_grad` = `matches!(requirement, Requirement::Grad)`), and
//! every one of those seven is a `GradInBackward` intermediate. So
//! `chunk_wy_forward_autodiff_s` declines - correctly, because the custom node
//! it would otherwise build comes back `UnTracked` and its output is a LEAF -
//! and `chunk_dispatch` takes the ops path, where `chunk_wy_forward_impl` runs
//! on the INCOMING tensors and burn builds the graph itself.
//!
//! Asserting the fused kernels on this fixture was asserting the BUG: for the
//! whole history of this project the attention arm ran thousands of fused
//! forwards whose output nothing downstream could send a gradient back to, and
//! `fused kda=<f>/0` read as "the fast path is slow" for a day. The
//! declaration under test now is the one that was actually broken: **the
//! module's parameters get a gradient**, and the counters name the arm.
//!
//! The fused path is still reachable - `require_grad()`'d leaves make the op
//! `Tracked` - and that is gated in burn-gdn2
//! (`tests/autodiff_cuda_gate.rs::fused_kernels_run_from_a_balanced_graph`,
//! red for a real reason: its `d_k`/`d_g` are wrong past one chunk).
#![cfg(all(feature = "cuda", feature = "autodiff"))]

use burn::tensor::{Device, Distribution, Tensor};
use burn_autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn_autodiff::Autodiff;
use burn_gdn2::CudaBare;
use burn_kda::{DecayFn, KdaConfig, KdaModule};

type AdBal = Autodiff<CudaBare, BalancedCheckpointing>;

fn rel_diff<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> f32 {
    let a = a.into_data();
    let b = b.into_data();
    let mut max_abs = 0.0f32;
    let mut scale = 0.0f32;
    for (x, y) in a.bytes.chunks_exact(4).zip(b.bytes.chunks_exact(4)) {
        let x = f32::from_le_bytes(x.try_into().unwrap());
        let y = f32::from_le_bytes(y.try_into().unwrap());
        max_abs = max_abs.max((x - y).abs());
        scale = scale.max(x.abs()).max(y.abs());
    }
    max_abs / scale.max(1e-30)
}

#[test]
fn kda_forward_train_state_produces_a_gradient_on_the_trainers_backend() {
    let dev = Device::cuda(0);
    let cfg = KdaConfig {
        hidden_size: 128,
        num_heads: 4,
        head_dim: 32,
        num_v_heads: Some(4),
        use_short_conv: false,
        decay_fn: DecayFn::Sigmoid,
        chunk_size: 16,
        ..Default::default()
    };
    // The module's PARAMETERS have to live on the autodiff device: a `Param` on
    // the bare backend is a leaf with no graph node, so `param.grad(&grads)` is
    // `None` however correct the backward is. This fixture built them on the bare
    // device, which is why it could never have read a parameter gradient - the
    // thing it was written to check.
    let ad = dev.clone().autodiff();
    let km = KdaModule::new(&cfg, 0.9, &ad);
    let x = Tensor::<3>::random([2, 64, 128], Distribution::Normal(0.0, 1.0), &ad).require_grad();

    burn_gdn2::reset_fused_calls();
    let (y, s) = km.forward_train_state::<AdBal>(x.clone(), None);
    let fwd_counts = burn_gdn2::seam_counts();
    println!(
        "seam after forward: asked={} fused_fwd={} fused_bwd={} declined={} ops_path={} \
         custom_node_bwd={}",
        fwd_counts.0, fwd_counts.1, fwd_counts.2, fwd_counts.3, fwd_counts.4, fwd_counts.5
    );

    // The arm is declared by what it produces, not by which kernel ran. The
    // `d(loss)/d(param)` arithmetic of this seam is pinned against central
    // finite differences in `tests/ops_grad_cuda.rs`; what this file adds is
    // the trainer's own wiring - the real module, the real backend, a real
    // state carried out of the op.
    let loss = y
        .clone()
        .powf_scalar(2.0)
        .sum()
        .add(s.powf_scalar(2.0).sum());
    let grads = loss.backward();
    let counts = burn_gdn2::seam_counts();
    println!(
        "seam after backward: asked={} fused_fwd={} fused_bwd={} declined={} ops_path={} \
         custom_node_bwd={}",
        counts.0, counts.1, counts.2, counts.3, counts.4, counts.5
    );
    assert!(
        counts.4 > 0,
        "the chunk op did not take the ops path ({counts:?}): the trainer's inputs \
         are all intermediates, so the fused node is declined and the gradient \
         must come from burn's own graph"
    );
    assert_eq!(
        counts.5, 0,
        "the custom node's backward ran ({counts:?}) while the op was declined: the \
         fused forward engaged after all, and this fixture no longer describes \
         the trainer"
    );
    for (name, t) in [
        ("q_proj.weight", &km.q_proj.weight),
        ("v_proj.weight", &km.v_proj.weight),
        ("o_proj.weight", &km.o_proj.weight),
    ] {
        let g = t
            .grad(&grads)
            .unwrap_or_else(|| panic!("{name} got NO gradient: the arm is frozen"));
        assert!(
            g.abs().max().into_scalar::<f32>() > 0.0,
            "{name} got an all-zero gradient"
        );
    }
    assert!(
        x.grad(&grads).unwrap().abs().max().into_scalar::<f32>() > 0.0,
        "no gradient reached the input"
    );

    // The numbers: the chunked WY form against the exact per-token recurrence.
    // Same function, different f32 reassociation.
    let mut st = None;
    let y_ref = km.forward_recurrent(x, &mut st, true);
    let err = rel_diff(y, y_ref);
    println!("KDA chunk vs per-token reference: rel_diff = {err:.3e}");
    assert!(err < 1e-3, "chunked drift vs reference: {err:.3e}");
}
