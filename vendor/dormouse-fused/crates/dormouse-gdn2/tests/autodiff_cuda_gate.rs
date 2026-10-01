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
use burn_autodiff::checkpoint::strategy::{BalancedCheckpointing, NoCheckpointing};
use burn_autodiff::Autodiff;
use dormouse_gdn2::{
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

/// Exclusive access to the library's launch counters.
///
/// `fused_calls()` is process-global (`static AtomicU64` in `cuda_dispatch.rs`),
/// and `cargo test` runs a binary's tests on parallel threads by default. So two
/// tests in THIS file that each `reset_fused_calls()` and then assert a count
/// are racing over one counter: measured 2026-09-28 as
/// `the_op_reaches_the_kernels_through_a_projection_graph_not_just_lifted_leaves`
/// passing alone and failing in the suite. Every test that reads the counters
/// takes this first.
///
/// A mutex rather than `--test-threads=1` in the runner: the suite must be
/// correct the way CI invokes it, and a test that only passes under a flag
/// nobody sets is a test that will be reported as green having asserted less.
static COUNTERS: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn exclusive_counters() -> std::sync::MutexGuard<'static, ()> {
    COUNTERS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The same numbers as `src`, on FRESH leaves of the same backend.
///
/// Two graphs that share leaves share graph nodes: a leaf is one node, and the
/// second backward walks every op that lists it as a parent. Comparing a fused
/// forward against a tensor forward on shared leaves therefore compares a clean
/// gradient against a doubled one - which is what made this test report the KDA
/// gradient off by rel 1.04 with the two forward values agreeing to 2e-7.
fn same_values_as(src: &[Tensor<4>; 7], dev: &Device) -> [Tensor<4>; 7] {
    std::array::from_fn(|i| {
        lift!(
            BalancedCheckpointing,
            Tensor::<4>::from_data(src[i].clone().into_data(), dev)
        )
    })
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
/// # GREEN on hardware, 2026-09-28
///
/// The three defects that kept this red, all measured rather than inferred,
/// and all in this library rather than in the test:
///
/// 1. the fused FORWARD branch was gated on `fused_forced_off()`
///    (`src/autodiff.rs`), so the kernels ran only when the kill switch said
///    the fused path was off - and the kill switch had already declined three
///    lines earlier. Both arms dead: `fused_calls()` read `(0, 0)` and the
///    sibling test `the_checkpointing_strategy_is_a_parameter_not_a_constant`
///    was RED in this tree.
/// 2. the adjoint's strip-to-bare asked the dispatch layer for an autodiff
///    context on a tensor that arrives with `DispatchAutodiffContext::Disabled`
///    ("Expected BalancedCheckpointing autodiff context, got Disabled"), and
///    the registration then re-wrapped each gradient as `Autodiff<Inner, S>`
///    where `Backward`'s `B` is the BARE backend ("Expected concrete Cube
///    backend with disabled autodiff context, got Enabled(Balanced)").
/// 3. this test's own comparison walked two graphs that SHARED the same seven
///    leaves, so the second backward also walked the fused op - a clean
///    gradient against a doubled one, which is where its "rel 1.04" came from.
///    `same_values_as` gives the ops side its own leaves now.
///
/// It is no longer `#[ignore]`d: it passes, and the numbers it prints (launch
/// counters 1/1 and every gradient at rel < 1e-6 against the ops path) are the
/// evidence. `tests/fused_adjoint_vs_ops.rs` is the kernel-level twin, with no
/// autodiff node in the way at all.
#[test]
fn fused_kernels_run_from_a_balanced_graph() {
    let _exclusive = exclusive_counters();
    let device = Device::cuda(0);
    let (batch, heads, time, k_dim, v_dim) = (2usize, 2usize, 64usize, 32usize, 32usize);
    let scale = 32f64.powf(-0.5);
    let chunk = 16usize;
    let inp = [
        lift!(
            BalancedCheckpointing,
            rnd(&device, [batch, heads, time, k_dim], 0.0)
        ),
        lift!(
            BalancedCheckpointing,
            rnd(&device, [batch, heads, time, k_dim], 0.0)
        ),
        lift!(
            BalancedCheckpointing,
            rnd(&device, [batch, heads, time, v_dim], 0.0)
        ),
        // negative gates (decay), like the model produces
        lift!(
            BalancedCheckpointing,
            rnd(&device, [batch, heads, time, k_dim], -0.5)
        ),
        lift!(
            BalancedCheckpointing,
            rnd(&device, [batch, heads, time, k_dim], 0.0)
        ),
        lift!(
            BalancedCheckpointing,
            rnd(&device, [batch, heads, time, v_dim], 0.0)
        ),
        lift!(
            BalancedCheckpointing,
            rnd(&device, [batch, heads, k_dim, v_dim], 0.0)
        ),
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
    // The loss carries the OUTPUT term only, on both sides, and the reason is
    // a contract difference rather than a convenience: `ChunkWy` returns its
    // state as an untracked leaf (`from_inner`, `src/autodiff.rs`), so the
    // fused adjoint's BPTT chain starts from a ZERO state adjoint, while the
    // ops path's state is an ordinary graph tensor. Put `state^2` into both
    // losses and the two paths compute different functions - measured on CUDA
    // 2026-09-28 with independent leaves, KDA's gradient rel 9.5e-1 apart,
    // while every other input agreed to 1.2e-7. Asserted just below, so the
    // exclusion is a measured fact rather than a quiet narrowing of the
    // comparison.
    let loss = out
        .clone()
        .powf_scalar(2.0)
        .sum()
        .add(out.clone().sum().mul_scalar(0.5));
    let grads = loss.backward();
    println!(
        "the fused op's state output requires grad: {} (an untracked leaf)",
        state.is_require_grad(),
    );
    assert!(
        !state.is_require_grad(),
        "the fused op's state output is now a tracked graph tensor: the BPTT \
         chain contract changed, and this test's output-only loss is no longer \
         the whole of what the two paths compute"
    );
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
    let (plain_out, _ops_state_probe) = chunk_wy_forward(
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
    let d_state = rel_diff(state.clone(), _ops_state_probe.clone());
    println!("fused vs tensor-ops chunk path: out {d_out:.3e}, state {d_state:.3e}");
    assert!(d_out < 1e-4, "forward drift vs tensor path: {d_out:.3e}");
    assert!(d_state < 1e-4, "state drift vs tensor path: {d_state:.3e}");

    // The adjoint must be the SAME function as the tensor adjoint's, or the
    // fused path is a different optimizer rather than a faster one. Same loss
    // terms, and - the point of `same_values_as` - INDEPENDENT leaves, so the
    // ops graph does not also walk the fused op that ran above on these nodes.
    let ref_inp = same_values_as(&inp, &device);
    let (ref_out, _ref_state_unused) = chunk_wy_forward(
        ref_inp[0].clone(),
        ref_inp[1].clone(),
        ref_inp[2].clone(),
        ref_inp[3].clone(),
        ref_inp[4].clone(),
        ref_inp[5].clone(),
        ref_inp[6].clone(),
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
    assert!(
        d_loss < 1e-3,
        "loss differs between the fused and the tensor graph"
    );
    for i in 0..inp.len() {
        let fused_g = inp[i].grad(&grads).unwrap().clone();
        let ref_g = ref_inp[i]
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

/// THE PRODUCTION QUESTION, and it is not the one the other tests ask.
///
/// Every other test in this file lifts bare leaves with `require_grad()`. The
/// trainer does not: `KdaModule::project` builds the seven inputs out of
/// matmuls, silus, sigmoids and a log, so each arrives with requirement
/// `GradInBackward`, not `Grad`. `Tensor::is_require_grad()` is
/// `matches!(requirement, Grad)` — the strict leaf requirement — so it is FALSE
/// for all seven of the trainer's inputs, and any gate written on it declines on
/// the trainer's own graph while passing on every fixture built from lifted
/// leaves.
///
/// That is not hypothetical: `chunk_wy_forward_autodiff_s` gated on exactly
/// that until 2026-09-28, which is why the op was "unreachable in production"
/// and why `tests/autodiff_nested_balanced.rs` (leaf -> `mul_scalar` ->
/// `permute`, the trainer's construction) was RED in this tree.
///
/// So this test builds its inputs the way the module does and asserts the
/// launch counters, which is the only thing that cannot tell a fallback from a
/// computation. `is_tracked()` — `!requirement.is_none()`, the same predicate
/// `OpsPrep::prepare` uses — is the correct question; `chunk_wy_forward_autodiff_s`
/// now asks it.
#[test]
fn the_op_reaches_the_kernels_through_a_projection_graph_not_just_lifted_leaves() {
    let _exclusive = exclusive_counters();
    let device = Device::cuda(0);
    device.seed(17);
    let (batch, heads, time, k_dim, v_dim) = (1usize, 2usize, 64usize, 32usize, 32usize);
    let scale = 32f64.powf(-0.5);
    let chunk = 16usize;

    // The leaves, as a Parameter would arrive: `require_grad()`.
    // The state is [B,H,K,V]; the six token-side tensors are [B,H,T,D], the
    // layout the op takes (the module's `to_4d` view, already applied).
    let leaves: [Tensor<4>; 7] = std::array::from_fn(|i| {
        let mean = if i == 3 { -1.0 } else { 0.0 };
        let d = if i == 2 || i == 5 || i == 6 {
            v_dim
        } else {
            k_dim
        };
        let shape = if i == 6 {
            [batch, heads, k_dim, v_dim]
        } else {
            [batch, heads, time, d]
        };
        lift!(
            BalancedCheckpointing,
            Tensor::<4>::random(shape, Distribution::Normal(mean, 0.3), &device)
        )
    });

    // ...and the op's inputs, as the module's `project` + `to_4d` build them:
    // an elementwise activation per leaf. Each now carries
    // `Requirement::GradInBackward`, which is the whole point.
    // an elementwise op of the kind `project` applies: enough to move the
    // requirement from `Grad` to `GradInBackward`, which is the point
    let act = |i: usize| leaves[i].clone().powf_scalar(2.0);
    let inputs = [
        act(0),
        act(1),
        act(2),
        // the decay gate through a log, like `alpha.log()`: negative
        leaves[3].clone().exp().log(),
        act(4),
        act(5),
        leaves[6].clone().mul_scalar(0.1),
    ];
    for (i, t) in inputs.iter().enumerate() {
        assert!(
            !t.is_require_grad(),
            "input {i} still has the LEAF requirement: the fixture is not \
             reproducing the trainer's graph, so it cannot catch the gate \
             regression this test exists for"
        );
    }

    reset_fused_calls();
    let (out, _state) = match chunk_dispatch::<AdBal>(
        inputs[0].clone(),
        inputs[1].clone(),
        inputs[2].clone(),
        inputs[3].clone(),
        inputs[4].clone(),
        inputs[5].clone(),
        inputs[6].clone(),
        scale,
        chunk,
    ) {
        Fused::Fused(r) => r,
        Fused::Fallback(why) => panic!("the fused path did not run: {why:?}"),
    };
    let (fwd, bwd) = fused_calls();
    println!("through a projection graph: fused launches {fwd}/{bwd}");
    assert_eq!(
        (fwd, bwd),
        (1, 0),
        "the fused forward must launch exactly once on the trainer's own graph \
         shape; a different count means it either fell back to the ops path or \
         ran twice."
    );

    // And the backward, so "reachable" is not just "the forward half".
    let grads = out
        .clone()
        .powf_scalar(2.0)
        .sum()
        .add(out.sum().mul_scalar(0.5))
        .backward();
    let (_, bwd) = fused_calls();
    assert_eq!(
        bwd, 1,
        "the fused adjoint never launched through a projection graph: the op \
         declined in the backward, which on the trainer means the attention arm \
         trains nothing"
    );
    for (i, t) in leaves.iter().enumerate() {
        let g = t
            .grad(&grads)
            .unwrap_or_else(|| panic!("leaf {i} got no gradient"));
        assert!(
            g.abs().max().into_scalar::<f32>() > 0.0,
            "leaf {i} got a zero gradient"
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
    let _exclusive = exclusive_counters();
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
