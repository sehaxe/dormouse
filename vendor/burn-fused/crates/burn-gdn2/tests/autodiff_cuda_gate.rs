//! The gate test: a tensor that came out of an autodiff graph must still reach
//! the fused kernels, on the checkpointing strategy the trainer actually uses.
//!
//! Every other fused test in this library builds its tensors on the BARE
//! backend, so the production gate was never exercised. `TypeId::of::<B>()`
//! against `CudaBare` is false for `Autodiff<CudaBare, BalancedCheckpointing>`
//! (dormouse's trainer backend) and false for every non-default strategy, so
//! the fused path silently fell back to the tensor-ops chunk loop for every
//! training step. The runtime half of the same bug: burn's dispatch layer
//! REFUSES to hand a bare primitive to a tensor whose autodiff context is
//! enabled, so even a correctly-typed gate got `None` on a training tensor.
//!
//! Numbers cannot catch that — the fallback computes the same function. So
//! these tests assert the fused LAUNCH COUNTERS moved, and then measure the
//! numerics of what actually ran.
#![cfg(all(feature = "cuda", feature = "autodiff"))]
#![allow(deprecated)]

use burn::backend::{AutodiffBackend, BackendTypes, NdArray};
use burn::tensor::{Device, Distribution, Tensor};
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::strategy::{BalancedCheckpointing, NoCheckpointing};
use burn_gdn2::{
    backend_matches, chunk_dispatch, chunk_wy_forward, chunk_wy_forward_autodiff, fused_calls,
    reset_fused_calls, CudaBare, Fused,
};

type AdBal = Autodiff<CudaBare, BalancedCheckpointing>;
type AdNo = Autodiff<CudaBare, NoCheckpointing>;

/// Max |a-b| / max|a| — the measure the crate's other fused tests use.
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

fn rnd<const D: usize>(device: &Device, shape: [usize; D], mean: f64) -> Tensor<D> {
    Tensor::<D>::random(shape, Distribution::Normal(mean, 0.3), device)
}

/// Lift a bare CUDA tensor onto `Autodiff<CudaBare, $strategy>` — what a
/// training graph hands the module. The lift IS the point: a tensor that
/// never went through it carries a DISABLED autodiff context, which is what
/// every fused test in this library used to build. (A macro, not a generic
/// fn: burn keeps `IntoGradientCheckpointingStrategy` private, so only a
/// concrete strategy can be named at a conversion.)
macro_rules! lift {
    ($strategy:ty, $t:expr) => {{
        let bare = $t
            .clone()
            .try_into_primitive::<CudaBare>()
            .expect("bare cuda tensor");
        let node = <Autodiff<CudaBare, $strategy> as AutodiffBackend>::from_inner(bare);
        Tensor::from_primitive::<Autodiff<CudaBare, $strategy>>(node).require_grad()
    }};
}

/// The seam's decision, for every backend this project runs. A string
/// comparison is only as good as the string, so the test prints it.
#[test]
fn backend_gate_decides_right_for_every_backend_we_run() {
    let cuda = <CudaBare as BackendTypes>::Device::default();
    let nd = <NdArray as BackendTypes>::Device::default();
    println!(
        "Backend::name: AdBal={} AdNo={} Cuda={} NdArray={}",
        <AdBal as burn::backend::Backend>::name(&cuda),
        <AdNo as burn::backend::Backend>::name(&cuda),
        <CudaBare as burn::backend::Backend>::name(&cuda),
        <NdArray as burn::backend::Backend>::name(&nd),
    );
    assert!(
        backend_matches::<AdBal>(),
        "the trainer's backend fell off the fused path"
    );
    assert!(backend_matches::<AdNo>(), "default strategy fell off");
    assert!(backend_matches::<CudaBare>(), "bare CUDA fell off");
    assert!(
        !backend_matches::<NdArray>(),
        "NdArray must stay on tensor ops"
    );
}

/// The production gate: from a Balanced-checkpointed graph the fused forward
/// AND adjoint kernels must launch, and the numbers must still be the chunked
/// WY form.
///
/// # What this proves
///
/// - `fused_calls()` counts kernel LAUNCHES and nothing else. Both counters
///   are incremented past every gate (backend, divisibility, dtype, kernel
///   limits, contiguity) and immediately before the first `launch_unchecked`,
///   so `(1, 0)` after the forward and `(1, 1)` after the backward mean one
///   fused forward and one fused adjoint really ran. Each counter used to be
///   the first statement of its entry point, which counted the CALLER's
///   interest: the backward one moved even when the adjoint returned `None` on
///   the next line, and this test's `bwd > 0` was green on that lie
///   (ADR-0019).
/// - The fused ADJOINT produces the same gradients as the tensor adjoint, to
///   the same tolerance the forward already had to meet. A counter cannot tell
///   a right adjoint from a wrong one, and the adjoint is the half of the fused
///   path that had never run inside a trainer.
///
/// # What this still cannot prove
///
/// - Anything on CPU: the kernels, both counters and this whole file are behind
///   `feature = "cuda"`. `tests/autodiff_nested_balanced.rs` is the CPU twin for
///   the graph SHAPE (nested parents, `BalancedCheckpointing`, real fwd+bwd) and
///   cannot see the kernels, the counters, or this gradient comparison.
/// - That the fused fwd+bwd is FASTER, or even that it is usable end to end in
///   a training step. This test asserts nothing about time. Until a step-time
///   measurement exists on a quiet GPU, only the fused FORWARD is established
///   as live; a 1.57x fwd+bwd number measured through the node may already have
///   included the tensor adjoint, and nobody has checked.
/// - That the trainer's own graph works: the inputs here are contiguous and the
///   graph is flat. The strided-input reproducer is
///   `tests/fused_permuted_view.rs`.
///
/// # `#[ignore]`d: this is the gate on the fused ADJOINT, and it has never passed
///
/// Measured on hardware 2026-09-28 — before `277b442` — it panicked at
/// `src/autodiff.rs:195`, where the adjoint closure refused, because the gate it
/// asked was the dead `TypeId` test on the autodiff backend. `277b442` rewrote
/// that closure so the strip-to-bare / run / rebuild happens where `Inner` is
/// nameable. **Nothing has been re-measured since, on either this test or
/// `fused_chunk_verify.rs`, so the refusal may or may not still happen**; what
/// is certain is that `bwd > 0` below has never been observed. The counter
/// placement from `f737710` stands: it sits after the gate, so a `bwd == 0`
/// today is the honest reading and not a counter lying about a refused call.
///
/// Un-ignoring this is not a formality: it is the measurement that decides
/// whether the fused backward is usable, and it must go green before any claim
/// about fused fwd+bwd or any training arm that relies on it.
///
/// Run it on demand:
/// `cargo test -p burn-gdn2 --release --features cuda,autodiff --test autodiff_cuda_gate -- --ignored --exact fused_kernels_run_from_a_balanced_graph --nocapture`
#[test]
#[ignore = "the gate on the fused adjoint, never run green; its old reason (a refusal at autodiff.rs:195 through the dead TypeId gate) was fixed in 277b442 and has not been re-measured, and the gradient comparison has never run on hardware"]
fn fused_kernels_run_from_a_balanced_graph() {
    let device = Device::cuda(0);
    let (batch, heads, time, k_dim, v_dim) = (2usize, 2usize, 64usize, 32usize, 32usize);
    let scale = 32f64.powf(-0.5);
    let chunk = 16usize;
    let inp = [
        lift!(BalancedCheckpointing, rnd(&device, [batch, heads, time, k_dim], 0.0)),
        lift!(BalancedCheckpointing, rnd(&device, [batch, heads, time, k_dim], 0.0)),
        lift!(BalancedCheckpointing, rnd(&device, [batch, heads, time, v_dim], 0.0)),
        // negative gates (decay), like the model produces
        lift!(BalancedCheckpointing, rnd(&device, [batch, heads, time, k_dim], -0.5)),
        lift!(BalancedCheckpointing, rnd(&device, [batch, heads, time, k_dim], 0.0)),
        lift!(BalancedCheckpointing, rnd(&device, [batch, heads, time, v_dim], 0.0)),
        lift!(BalancedCheckpointing, rnd(&device, [batch, heads, k_dim, v_dim], 0.0)),
    ];

    reset_fused_calls();
    let (out, state) = match chunk_dispatch::<AdBal>(
        inp[0].clone(),
        inp[1].clone(),
        inp[2].clone(),
        inp[3].clone(),
        inp[4].clone(),
        inp[5].clone(),
        inp[6].clone(),
        scale,
        chunk,
    ) {
        Fused::Fused(r) => r,
        Fused::Fallback(why) => panic!("the fused path did not run: {why:?}"),
    };
    let (fwd, bwd_after_fwd) = fused_calls();
    assert_eq!(
        (fwd, bwd_after_fwd),
        (1, 0),
        "one fused forward and no adjoint yet; anything else means a counter is \
         counting a gate that refused, or the graph ran the op twice"
    );

    // Backward: the fused adjoint kernel, not the tensor-ops adjoint. Ops AFTER
    // the op, so the adjoint is reached through a chain.
    let loss = out
        .clone()
        .powf_scalar(2.0)
        .sum()
        .add(state.clone().powf_scalar(2.0).sum())
        .add(out.clone().sum().mul_scalar(0.5));
    let grads = loss.backward();
    let (fwd, bwd) = fused_calls();
    assert_eq!(fwd, 1, "the fused forward ran more than once");
    assert_eq!(
        bwd, 1,
        "the fused adjoint kernel never launched: the gate is closed again, and \
         the counter now says so instead of counting the call"
    );
    for (i, t) in inp.iter().enumerate() {
        let g = t
            .grad(&grads)
            .unwrap_or_else(|| panic!("input {i} got no gradient"));
        assert!(
            g.abs().max().into_scalar::<f32>() > 0.0,
            "input {i} got a zero gradient"
        );
    }

    // Numerics: the fused chunk path against the tensor-ops chunk path, which
    // is what the trainer ran until this gate was fixed. Different algorithm
    // (f32 reassociation over chunks), same function.
    let (plain_out, plain_state) = chunk_wy_forward(
        inp[0].clone(),
        inp[1].clone(),
        inp[2].clone(),
        inp[3].clone(),
        inp[4].clone(),
        inp[5].clone(),
        inp[6].clone(),
        scale,
        chunk,
    );
    let d_out = rel_diff(out.clone(), plain_out);
    let d_state = rel_diff(state.clone(), plain_state);
    println!("fused vs tensor-ops chunk path: out {d_out:.3e}, state {d_state:.3e}");
    assert!(d_out < 1e-4, "forward drift vs tensor path: {d_out:.3e}");
    assert!(d_state < 1e-4, "state drift vs tensor path: {d_state:.3e}");

    // The adjoint must be the SAME function as the tensor adjoint's, or the
    // fused path is a different optimizer rather than a faster one. Same
    // values, same graph, tensor path instead of the fused node.
    let (ref_out, ref_state) = chunk_wy_forward(
        inp[0].clone(),
        inp[1].clone(),
        inp[2].clone(),
        inp[3].clone(),
        inp[4].clone(),
        inp[5].clone(),
        inp[6].clone(),
        scale,
        chunk,
    );
    let ref_loss = ref_out
        .clone()
        .powf_scalar(2.0)
        .sum()
        .add(ref_state.powf_scalar(2.0).sum())
        .add(ref_out.sum().mul_scalar(0.5));
    let ref_grads = ref_loss.backward();
    let d_loss = (loss.clone().into_scalar::<f32>() - ref_loss.into_scalar::<f32>()).abs();
    println!("fused graph vs tensor graph: loss {d_loss:.3e}");
    assert!(d_loss < 1e-3, "loss differs between the fused and the tensor graph");
    for (i, t) in inp.iter().enumerate() {
        let fused_g = t.grad(&grads).unwrap().clone();
        let ref_g = t
            .grad(&ref_grads)
            .unwrap_or_else(|| panic!("input {i} got no gradient on the tensor path"));
        let d = rel_diff(fused_g.clone(), ref_g.clone());
        println!("input {i} gradient, fused adjoint vs tensor adjoint: rel {d:.3e}");
        assert!(
            d < 1e-3,
            "input {i}: the fused adjoint disagrees with the tensor adjoint \
             (rel={d:.2e}) — the fused backward would be a different function"
        );
    }
}

/// The strategy is a parameter of the op, not a hardcoded
/// `NoCheckpointing`: the default-strategy graph reaches the kernels too, and
/// the default-strategy-only entry point stays (correctly) blind to a
/// Balanced graph — that blindness WAS the bug, so both directions are
/// asserted.
#[test]
fn the_checkpointing_strategy_is_a_parameter_not_a_constant() {
    let device = Device::cuda(0);
    let (batch, heads, time, k_dim, v_dim) = (1usize, 2usize, 32usize, 16usize, 16usize);
    let shapes = [
        [batch, heads, time, k_dim],
        [batch, heads, time, k_dim],
        [batch, heads, time, v_dim],
        [batch, heads, time, k_dim],
        [batch, heads, time, k_dim],
        [batch, heads, time, v_dim],
        [batch, heads, k_dim, v_dim],
    ];
    let mut no = Vec::new();
    let mut bal = Vec::new();
    for (i, s) in shapes.iter().enumerate() {
        let mean = if i == 3 { -0.5 } else { 0.0 };
        no.push(lift!(NoCheckpointing, rnd(&device, *s, mean)));
        bal.push(lift!(BalancedCheckpointing, rnd(&device, *s, mean)));
    }
    let seven = |v: &[Tensor<4>]| {
        (
            v[0].clone(),
            v[1].clone(),
            v[2].clone(),
            v[3].clone(),
            v[4].clone(),
            v[5].clone(),
            v[6].clone(),
        )
    };

    let nn = seven(&no);
    assert!(
        chunk_wy_forward_autodiff::<CudaBare>(nn.0, nn.1, nn.2, nn.3, nn.4, nn.5, nn.6, 0.125, 16)
            .is_some(),
        "a NoCheckpointing graph must reach the op"
    );
    let bb = seven(&bal);
    assert!(
        chunk_wy_forward_autodiff::<CudaBare>(bb.0, bb.1, bb.2, bb.3, bb.4, bb.5, bb.6, 0.125, 16)
            .is_none(),
        "the default-strategy entry point must stay blind to a Balanced graph"
    );

    reset_fused_calls();
    let nn = seven(&no);
    assert!(
        chunk_dispatch::<AdNo>(nn.0, nn.1, nn.2, nn.3, nn.4, nn.5, nn.6, 0.125, 16).is_fused(),
        "NoCheckpointing graph fell off the fused path"
    );
    assert!(fused_calls().0 > 0, "no fused forward on NoCheckpointing");
    reset_fused_calls();
    let bb = seven(&bal);
    assert!(
        chunk_dispatch::<AdBal>(bb.0, bb.1, bb.2, bb.3, bb.4, bb.5, bb.6, 0.125, 16).is_fused(),
        "Balanced graph fell off the fused path"
    );
    assert!(fused_calls().0 > 0, "no fused forward on Balanced");
}
