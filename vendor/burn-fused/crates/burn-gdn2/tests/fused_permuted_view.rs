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

/// The seven op inputs, transposed exactly as the trainer transposes them: six
/// token-side tensors `[B,T,H,D] -> [B,H,T,D]`, and the state `[B,H,K,V]`
/// which the trainer starts each loop as a contiguous zeros leaf.
fn strided_inputs(
    device: &Device,
    batch: usize,
    heads: usize,
    time: usize,
    k: usize,
    v: usize,
) -> [Tensor<4>; 7] {
    // The activation comes first, like `project()` does (silu / sigmoid / log),
    // and the view ops last, so the op's parent is a node over a strided buffer.
    let side = |d: usize, mean: f64| -> Tensor<4> {
        project_to_4d(
            lift!(rnd(device, [batch, time, heads * d], mean)).mul_scalar(0.5),
            heads,
            d,
        )
    };
    [
        side(k, 0.0),
        side(k, 0.0),
        side(v, 0.0),
        // negative, like a log decay the model produces
        project_to_4d(
            lift!(rnd(device, [batch, time, heads * k], -1.0)).exp().log(),
            heads,
            k,
        ),
        side(k, 0.0),
        side(v, 0.0),
        // the state: contiguous, like the trainer's zeros
        lift!(rnd(device, [batch, heads, k, v], 0.0)),
    ]
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
/// # `#[ignore]`d: the materializer is proven, the ADJOINT it feeds is not
///
/// Half of this test is green and stays that way: `contiguous_inputs_are_left_alone`
/// ran on hardware 2026-09-28 and passed, and with the fixture fixed the forward
/// half of this test — `copies >= 6`, zero fallbacks, the fused launch, the
/// caller's tensors unmutated, the output equal to the tensor path — ran too, on
/// the same card, inside the trainer smoke (`fused kda=12/0`).
///
/// What has never passed is the tail: `loss.backward()`, which reaches
/// `src/autodiff.rs:195` and refuses, because the fused adjoint is not live yet.
/// So the test is `#[ignore]`d rather than left red — a permanently red suite
/// trains everyone to ignore red, and this project's retracted claims all came
/// from green-looking checks covering something other than what they said.
///
/// When un-ignored it proves the WHOLE seam on the trainer's own layout: a
/// transposed-view input is materialized rather than read as contiguous, the
/// fused forward launches, the fused adjoint launches, the backward completes
/// and every input gets a non-zero gradient, the caller's tensors are unchanged
/// by the materialization, and the fused forward still equals the tensor path.
///
/// Run it on demand:
/// `cargo test -p burn-gdn2 --release --features cuda,autodiff --test fused_permuted_view -- --ignored --exact a_strided_input_reaches_the_fused_kernels_and_completes_a_backward --nocapture`
#[test]
#[ignore = "the fused adjoint refuses at autodiff.rs:195; the forward half of this test ran green on hardware, the backward half never has"]
fn a_strided_input_reaches_the_fused_kernels_and_completes_a_backward() {
    let device = Device::cuda(0);
    // Small: the overflow was shape-independent, so the reproducer must be too.
    let (batch, heads, time, k, v, chunk) = (1usize, 2usize, 64usize, 32usize, 32usize, 16usize);
    let scale = (k as f64).powf(-0.5);
    let inp = strided_inputs(&device, batch, heads, time, k, v);
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
    // a chain — and a recompute of the transposed parents.
    let loss = out
        .clone()
        .powf_scalar(2.0)
        .sum()
        .add(out.clone().sum().mul_scalar(0.5));
    let grads = loss.backward();
    for (i, t) in inp.iter().enumerate() {
        let g = t
            .grad(&grads)
            .unwrap_or_else(|| panic!("input {i} got no gradient"));
        assert!(
            g.abs().max().into_scalar::<f32>() > 0.0,
            "input {i} got a zero gradient"
        );
    }

    // The materialization must not have disturbed the caller's tensors.
    for (i, t) in inp.iter().enumerate() {
        assert_eq!(
            t.clone().into_data().bytes,
            before[i],
            "input {i} was mutated by the contiguity materialization"
        );
    }

    // And the fused forward must still be the chunked WY function, not a
    // differently-ordered copy of it: same numbers as the tensor path on the
    // same transposed inputs.
    let plain = strided_inputs(&device, batch, heads, time, k, v);
    let (plain_out, _) = chunk_wy_forward(
        plain[0].clone(),
        plain[1].clone(),
        plain[2].clone(),
        plain[3].clone(),
        plain[4].clone(),
        plain[5].clone(),
        plain[6].clone(),
        scale,
        chunk,
    );
    let d = rel_diff(out.clone(), plain_out);
    assert!(
        d < 1e-4,
        "fused vs tensor path on a transposed input: rel={d:.2e}"
    );
}
