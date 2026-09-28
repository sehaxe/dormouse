#![cfg(all(feature = "cuda", feature = "autodiff"))]
#![allow(deprecated)]
//! The regression for the stack overflow that made the fused KDA op unusable
//! in the trainer, in its OWN test binary: a stack overflow aborts the process,
//! so it must not share a binary with tests that have assertions to report.
//!
//! ## What broke
//!
//! `KdaModule::project` builds every op input as `[B,T,H*D]` reshaped and
//! `.permute([0,2,1,3])`-ed to `[B,H,T,D]` (burn-kda `lib.rs:407-410`), which is
//! a metadata-only view: a STRIDED tensor. The seam's `cube_of` materialized
//! such a view with `t.mul_scalar(1.0)` and then called ITSELF on the result.
//! On burn 0.22 that op is not a materializer: `launch_scalar_binop` routes
//! through `in_memory_order`, which allocates the output dense in the operand's
//! *memory* order and then permutes the metadata back (burn-cubecl
//! `kernel/memory_order.rs:38`). The "copy" therefore came back with the very
//! strides it was asked to remove, at every shape, and `cube_of` recursed until
//! the stack died — `thread 'main' has overflowed its stack` before step 0 of
//! any training run. `RUST_MIN_STACK` does not help because the trainer's
//! warmup runs on the MAIN thread, whose stack is the OS limit, and because the
//! recursion is unbounded either way.
//!
//! ## Why every other test missed it
//!
//! No test in this crate ever handed the seam a strided view. `Tensor::random`
//! and every hand-built input are freshly allocated and contiguous, so
//! `cube_of`'s non-contiguous branch never ran — including
//! `tests/autodiff_cuda_gate.rs`, which is on the trainer's backend but flat.
//! The property under test is the LAYOUT, not the shape: this file transposes
//! and the shape stays small, so a fix that only handled large tensors, or only
//! contiguous ones, fails here.
//!
//! ## The fixture, and how it was got wrong once
//!
//! The first version of this file built its views as
//! `swap_dims(1,2).permute([0,2,1,3])` — a permutation followed by its own
//! inverse, which returns the strides to where they started. It reported
//! `0 copies` on the GPU, because the inputs were contiguous and the
//! branch under test never ran: a test that does not test. The fixture is now
//! the trainer's own construction, and the CPU twin
//! `tests/autodiff_nested_balanced.rs::the_trainers_view_is_a_real_transposition`
//! proves the difference with VALUES (the trainer's permute reads the data back
//! in a different order; the round trip reads it back identical), so the mistake
//! cannot come back unnoticed. The control arm below closes the same gap from
//! the other side: contiguous inputs must produce ZERO materializations, or a
//! count of 6 on strided inputs would mean nothing.
//!
//! ## DEFERRED — needs a free GPU
//!
//! Not run while a training run held the card. It allocates device memory and
//! launches kernels:
//!
//! ```sh
//! cd vendor/burn-fused
//! cargo test -p burn-gdn2 --release --features cuda,autodiff --test fused_permuted_view -- --nocapture
//! ```

use burn::backend::AutodiffBackend;
use burn::tensor::{Bytes, Device, Distribution, Tensor};
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn_gdn2::alloc_trace::{contiguous_copies, reset_contiguous};
use burn_gdn2::{chunk_dispatch, chunk_wy_forward, fused_calls, reset_fused_calls, CudaBare, Fused};

type AdBal = Autodiff<CudaBare, BalancedCheckpointing>;

/// Lift a bare CUDA tensor onto `Autodiff<CudaBare, BalancedCheckpointing>` —
/// what a training graph hands the module. A macro, not a generic fn: burn
/// keeps `IntoGradientCheckpointingStrategy` private, so only a concrete
/// strategy can be named at a conversion.
macro_rules! lift {
    ($t:expr) => {{
        let bare = $t
            .clone()
            .try_into_primitive::<CudaBare>()
            .expect("bare cuda tensor");
        let node = <AdBal as AutodiffBackend>::from_inner(bare);
        Tensor::from_primitive::<AdBal>(node).require_grad()
    }};
}

fn rel_diff<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> f32 {
    let a = a.clone().into_data();
    let b = b.clone().into_data();
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

/// The trainer's own input construction, copied from `KdaModule::project`'s
/// `to_4d` (burn-kda `src/lib.rs:407-410`): a `reshape([B,T,H,D])` of the
/// projection followed by `.permute([0,2,1,3])`. THAT LINE is the only reason
/// this fixture and the bug it guards exist — the trainer hands the fused seam
/// a strided tensor on every KDA call and nowhere else in the project does.
///
/// Applied to a lifted, tracked tensor, so the parent the op sees is a real
/// permute node in the graph, exactly as in the trainer.
fn project_to_4d(t: Tensor<3>, heads: usize, d: usize) -> Tensor<4> {
    let [b, tokens, hk] = t.shape().dims::<3>();
    assert_eq!(hk, heads * d, "the projection must be [B,T,H*D]");
    t.reshape([b, tokens, heads, d])
        .permute([0, 2, 1, 3])
}

/// The fixture's three pieces. `inputs` is what the op sees; the two leaf
/// fields are where a gradient can be READ. Keeping them apart is not tidiness:
/// burn materializes a gradient only for a LEAF marked `require_grad`
/// (`NodeRef::clone_if_require_grad`), so `t.grad(&grads)` on a permute output
/// is `None` even when the gradient reached the graph perfectly. The op's
/// inputs here are `permute(mul_scalar(leaf))` - all intermediates - so
/// checking THEM for a gradient is a check that cannot pass, and its failure
/// says nothing about the seam. `tests/autodiff_nested_balanced.rs` keeps the
/// same separation for the same reason.
struct Strided {
    /// The six `[B,T,H*D]` leaves behind the transposed views, in input order.
    token_leaves: Vec<Tensor<3>>,
    /// The `[B,H,K,V]` state, itself a leaf.
    state_leaf: Tensor<4>,
    /// What the op is handed.
    inputs: [Tensor<4>; 7],
}

/// One `[B,T,H*D]` leaf and the `[B,H,T,D]` view `project()` builds from it.
/// The leaf is returned so a gradient can be read where burn keeps one.
fn leaf_and_view(
    device: &Device,
    batch: usize,
    heads: usize,
    time: usize,
    d: usize,
    mean: f64,
) -> (Tensor<3>, Tensor<4>) {
    let l = lift!(rnd(device, [batch, time, heads * d], mean));
    let view = project_to_4d(l.clone().mul_scalar(0.5), heads, d);
    (l, view)
}

fn strided_inputs(
    device: &Device,
    batch: usize,
    heads: usize,
    time: usize,
    k: usize,
    v: usize,
) -> Strided {
    // The activation comes first, like `project()` does (silu / sigmoid / log),
    // and the view ops last, so the op's parent is a node over a strided buffer.
    let (l0, q) = leaf_and_view(device, batch, heads, time, k, 0.0);
    let (l1, kq) = leaf_and_view(device, batch, heads, time, k, 0.0);
    let (l2, vv) = leaf_and_view(device, batch, heads, time, v, 0.0);
    // negative, like a log decay the model produces
    let l3 = lift!(rnd(device, [batch, time, heads * k], -1.0));
    let g = project_to_4d(l3.clone().exp().log(), heads, k);
    let (l4, b) = leaf_and_view(device, batch, heads, time, k, 0.0);
    let (l5, w) = leaf_and_view(device, batch, heads, time, v, 0.0);
    // The state is contiguous, like the trainer's zeros, and IS a leaf (4-D).
    let state_leaf: Tensor<4> = lift!(rnd(device, [batch, heads, k, v], 0.0));
    Strided {
        token_leaves: vec![l0, l1, l2, l3, l4, l5],
        state_leaf: state_leaf.clone(),
        inputs: [q, kq, vv, g, b, w, state_leaf],
    }
}

/// The same seven shapes with NO transposition: the control. A materializer
/// that fires here is firing on data that is already row-major, which would
/// make the strided arm's count meaningless.
fn contiguous_inputs(
    device: &Device,
    batch: usize,
    heads: usize,
    time: usize,
    k: usize,
    v: usize,
) -> [Tensor<4>; 7] {
    [
        lift!(rnd(device, [batch, heads, time, k], 0.0)),
        lift!(rnd(device, [batch, heads, time, k], 0.0)),
        lift!(rnd(device, [batch, heads, time, v], 0.0)),
        lift!(rnd(device, [batch, heads, time, k], -1.0)),
        lift!(rnd(device, [batch, heads, time, k], 0.0)),
        lift!(rnd(device, [batch, heads, time, v], 0.0)),
        lift!(rnd(device, [batch, heads, k, v], 0.0)),
    ]
}

fn run_op(
    inp: &[Tensor<4>; 7],
    scale: f64,
    chunk: usize,
) -> (Tensor<4>, Tensor<4>, &'static str) {
    match chunk_dispatch::<AdBal>(
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
        Fused::Fused((o, s)) => (o, s, "fused"),
        Fused::Fallback(why) => panic!("the fused path did not run: {why:?}"),
    }
}

/// CONTIGUOUS inputs must not be materialized. Without this arm, a count of 0
/// on the strided arm would be ambiguous between "the fixture is contiguous"
/// (the bug this file already had once) and "the seam never materializes
/// anything" (a wrong-answer bug). With it, only the second can pass.
#[test]
fn contiguous_inputs_are_left_alone() {
    let device = Device::cuda(0);
    let (batch, heads, time, k, v, chunk) = (1usize, 2usize, 64usize, 32usize, 32usize, 16usize);
    let inp = contiguous_inputs(&device, batch, heads, time, k, v);
    reset_fused_calls();
    reset_contiguous();
    let _ = run_op(&inp, (k as f64).powf(-0.5), chunk);
    assert!(
        fused_calls().0 > 0,
        "the fused forward kernel never launched on contiguous inputs"
    );
    let (copies, fallbacks) = contiguous_copies();
    assert_eq!(
        (copies, fallbacks),
        (0, 0),
        "contiguous inputs were copied or refused: the materializer must only \
         fire on a strided view, or the strided arm's count proves nothing"
    );
}

/// The reported failure, end to end: a Balanced-checkpointed graph whose op
/// inputs are the trainer's transposed views must complete a real fwd+bwd,
/// with the fused forward engaged, the views materialized once each, and the
/// numbers unchanged.
///
/// # Status, measured on hardware 2026-09-28
///
/// The forward half runs green: `copies >= 6`, zero fallbacks, the fused
/// launch, the caller's tensors unmutated, the output equal to the tensor
/// path. Two things were wrong with the backward half, and both were the
/// TEST's, not the seam's:
///
/// 1. The fixture's "inputs" are `permute(mul_scalar(leaf))` — graph
///    intermediates — and burn materializes a gradient only for a LEAF marked
///    `require_grad`. `t.grad(&grads)` on a permute output is therefore `None`
///    whatever the seam did, and the test panicked with "input 0 got no
///    gradient" instead of measuring anything. The fixture now returns the
///    leaves and the gradients are read there.
/// 2. The fused path no longer REFUSES. Two real defects fixed on 2026-09-28
///    (an inverted `fused_allowed` and a `strip` of tensors that were already
///    bare) are what used to make it refuse; with those gone the fused forward
///    and the fused adjoint both launch here, and the backward completes.
///
/// What this test still cannot assert is that the fused adjoint is the right
/// derivative. Measured on the same card: at one chunk every input agrees
/// with the ops path to ~2e-7, and from two chunks on `d_k` diverges by
/// 1.8e-1…2.8e-1 and `d_g` by 1.7e-2…3.2e-2. The assertion below is
/// therefore the weak "non-zero" one on purpose — it is what this file is
/// about (the LAYOUT reaching the kernels, not the adjoint's arithmetic).
/// `tests/autodiff_cuda_gate.rs::fused_kernels_run_from_a_balanced_graph` is
/// where that disagreement is a red gate.
///
/// Run it on demand:
/// `cargo test -p burn-gdn2 --release --features cuda,autodiff --test fused_permuted_view -- --ignored --exact a_strided_input_reaches_the_fused_kernels_and_completes_a_backward --nocapture`
#[test]
fn a_strided_input_reaches_the_fused_kernels_and_completes_a_backward() {
    let device = Device::cuda(0);
    // Small: the overflow was shape-independent, so the reproducer must be too.
    let (batch, heads, time, k, v, chunk) = (1usize, 2usize, 64usize, 32usize, 32usize, 16usize);
    let scale = (k as f64).powf(-0.5);
    let fx = strided_inputs(&device, batch, heads, time, k, v);
    let inp = &fx.inputs;
    // Keep the values: a materialization that wrote in memory order instead of
    // logical order would leave the inputs alone and corrupt what the kernel
    // reads, so the numbers below are the check.
    let before: Vec<Bytes> = inp.iter().map(|t| t.clone().into_data().bytes).collect();

    reset_fused_calls();
    reset_contiguous();
    let (out, _state, arm) = run_op(&inp, scale, chunk);
    assert_eq!(arm, "fused");
    let (fwd, _) = fused_calls();
    assert!(fwd > 0, "the fused forward kernel never launched");
    let (copies, fallbacks) = contiguous_copies();
    assert!(
        copies >= 6,
        "the trainer's transposed views were not materialized ({copies} copies, \
         fallbacks {fallbacks}): either the fixture is contiguous (see \
         the_trainers_view_is_a_real_transposition) or the seam read the views \
         as if they were row-major, which is a wrong-answer bug"
    );
    assert_eq!(
        fallbacks, 0,
        "a materialization did not come back row-major: {fallbacks} fallbacks"
    );

    // A real fwd+bwd, with ops after the op so the backward reaches it through
    // a chain — and a recompute of the transposed parents. The loss is the
    // OUTPUT only: on the fused path the op's state output is an untracked
    // leaf (the module header says so), so a state term in the loss would
    // contribute a gradient the op does not and cannot produce.
    let loss = out
        .clone()
        .powf_scalar(2.0)
        .sum()
        .add(out.clone().sum().mul_scalar(0.5));
    let grads = loss.backward();
    for (i, t) in fx.token_leaves.iter().enumerate() {
        let g = t
            .grad(&grads)
            .unwrap_or_else(|| panic!("token leaf {i} got no gradient"));
        assert!(
            g.abs().max().into_scalar::<f32>() > 0.0,
            "token leaf {i} got a zero gradient"
        );
    }
    let gs = fx
        .state_leaf
        .grad(&grads)
        .expect("the state leaf got no gradient");
    assert!(
        gs.abs().max().into_scalar::<f32>() > 0.0,
        "the state leaf got a zero gradient"
    );

    // The materialization must not have disturbed the caller's tensors.
    for (i, t) in inp.iter().enumerate() {
        assert_eq!(
            t.clone().into_data().bytes,
            before[i],
            "input {i} was mutated by the contiguity materialization"
        );
    }

    // And the fused forward must still be the chunked WY function, not a
    // differently-ordered copy of it: same numbers as the tensor path on THE
    // SAME transposed inputs. A freshly drawn fixture would compare two
    // different functions - `Tensor::random` is unseeded here, so a second
    // call is a different set of values, and this assertion used to be
    // unreachable so nothing caught it.
    let (plain_out, _) = chunk_wy_forward(
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
    let d = rel_diff(out.clone(), plain_out);
    assert!(
        d < 1e-4,
        "fused vs tensor path on a transposed input: rel={d:.2e}"
    );
}
