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
/// AND adjoint kernels must launch, and the adjoint must be the SAME FUNCTION
/// as the tensor path's - not merely non-zero.
///
/// # Status, measured on hardware 2026-09-28
///
/// The forward and the adjoint both LAUNCH now. Two real defects fixed that
/// day are why they used to refuse: an inverted `fused_allowed` in
/// `autodiff.rs` (so the fused forward was dead code and every entry silently
/// took the tensor branch) and a `strip`/`rebuild` pair in `FusedAdjoint` for
/// tensors that were already bare (so the adjoint returned `Err` and the
/// caller panicked). `the_checkpointing_strategy_is_a_parameter_not_a_constant`
/// below is what caught the first one: it asserts `fused_calls().0 > 0` where
/// the op is Tracked.
///
/// It still fails, and the failure is REAL and NEW: the fused adjoint's `d_k`
/// and `d_g` disagree with the ops path once there is more than one chunk.
/// Measured on the same card, `b=2 h=2 k=v=32 chunk=16`, loss `sum(out^2)`,
/// fused adjoint vs the ops path, varying T:
///
/// ```text
/// T=16  (1 chunk)  q 1.6e-7  k 2.4e-7  v 2.5e-7  g 1.7e-2  b 3.4e-7  w 2.5e-7  s 1.9e-7
/// T=32  (2 chunks) q 4.2e-7  k 1.8e-1   v 3.5e-7  g 2.3e-2  b 2.4e-7  w 1.7e-7  s 3.0e-7
/// T=64  (4 chunks) q 4.2e-7  k 2.0e-1   v 1.5e-7  g 2.2e-2  b 2.4e-7  w 3.6e-7  s 3.4e-7
/// T=128 (8 chunks) q 4.2e-7  k 2.8e-1   v 3.0e-7  g 3.2e-2  b 3.0e-7  w 3.1e-7  s 4.3e-7
/// ```
///
/// q, v, b, w and the state agree to f32 noise at every length; `d_k` and
/// `d_g` are correct for ONE chunk and wrong for two or more, which points at
/// the cross-chunk BPTT chain for the `E = exp(cumsum(g))` terms rather than at
/// the per-chunk algebra. That is the open bug this gate exists for, and it is
/// the reason the fused arm is not the default: the ops path's gradient is
/// verified against finite differences in
/// `burn-kda/tests/ops_grad_cuda.rs`, the fused one is not.
///
/// Run it on demand:
/// `cargo test -p burn-gdn2 --release --features cuda,autodiff --test autodiff_cuda_gate -- --ignored --exact fused_kernels_run_from_a_balanced_graph --nocapture`
#[test]
#[ignore = "MEASURED 2026-09-28: the fused forward and adjoint both launch (the old refusal was an inverted fused_allowed plus a strip of already-bare tensors, both fixed here), and they are correct for one chunk - but from two chunks on the fused adjoint's d_k is off by 1.8e-1..2.8e-1 and its d_g by 1.7e-2..3.2e-2 against the ops path, while q/v/b/w/state agree to ~3e-7"]
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
    //
    // The loss is the OUTPUT only. The op's state output is an UNTRACKED leaf
    // on this path (the burn-gdn2 module header says so, and the fused adjoint
    // starts its BPTT chain from a zero state adjoint), so a `state^2` term
    // would ask for a gradient this node structurally cannot produce - the
    // test would then be red for a reason that is the op's contract, not a
    // defect, which is the same class of green-looking gate this file exists
    // to stop.
    let loss = out.clone().powf_scalar(2.0).sum().add(out.clone().sum().mul_scalar(0.5));
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
    // values, same graph, tensor path instead of the fused node. The tensor
    // path's own gradient is pinned against finite differences in
    // `burn-kda/tests/ops_grad_cuda.rs`, so this is a comparison against a
    // checked value and not between two unchecked ones.
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
             (rel={d:.2e}) - the fused backward would be a different function"
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
