#![cfg(all(feature = "cuda", feature = "autodiff"))]
#![allow(deprecated)]
//! The regression for the stack overflow that made the fused KDA op unusable
//! in the trainer, in its OWN test binary: a stack overflow aborts the process,
//! so it must not share a binary with tests that have assertions to report.
//!
//! ## What broke
//!
//! `KdaModule::project` builds every op input as `[B,T,H*D]` reshaped and
//! `.permute([0,2,1,3])`-ed to `[B,H,T,D]` (burn-kda `lib.rs:409`), which is a
//! metadata-only view: a STRIDED tensor. The seam's `cube_of` materialized such
//! a view with `t.mul_scalar(1.0)` and then called ITSELF on the result. On
//! burn 0.22 that op is not a materializer: `launch_scalar_binop` routes
//! through `in_memory_order`, which allocates the output dense in the
//! operand's *memory* order and then permutes the metadata back
//! (burn-cubecl `kernel/memory_order.rs:38`). The "copy" therefore came back
//! with the very strides it was asked to remove, at every shape, and
//! `cube_of` recursed until the stack died — `thread 'main' has overflowed its
//! stack` before step 0 of any training run. `RUST_MIN_STACK` does not help
//! because the trainer's warmup runs on the MAIN thread, whose stack is the
//! OS limit, and because the recursion is unbounded either way.
//!
//! ## Why every other test missed it
//!
//! No test in this crate ever handed the seam a strided view. `Tensor::random`
//! and every hand-built input are freshly allocated and contiguous, so
//! `cube_of`'s non-contiguous branch never ran — including
//! `tests/autodiff_cuda_gate.rs`, which is on the trainer's backend but flat.
//! The property under test is the LAYOUT, not the shape: this file permutes
//! and the shape stays small, so a fix that only handled large tensors, or
//! only contiguous ones, fails here.
//!
//! ## DEFERRED — needs a free GPU
//!
//! Not run while a training run held the card. It allocates device memory and
//! launches kernels:
//!
//! ```sh
//! cd vendor/burn-fused
//! cargo test -p burn-gdn2 --features cuda,autodiff --test fused_permuted_view -- --nocapture
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

fn rel_diff<const D: usize>(a: &Tensor<D>, b: &Tensor<D>) -> f32 {
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

/// The trainer's seven op inputs, each a STRIDED view of a contiguous buffer:
/// `project()`'s `[B,T,H,D] -> [B,H,T,D]` permute, which leaves the logical
/// shape unchanged and the strides non-nesting.
fn strided_inputs(
    device: &Device,
    batch: usize,
    heads: usize,
    time: usize,
    k: usize,
    v: usize,
) -> [Tensor<4>; 7] {
    let permute = |t: Tensor<4>| t.swap_dims(1, 2).permute([0, 2, 1, 3]);
    let s = [batch, heads, time];
    [
        lift!(permute(rnd(device, [s[0], s[1], s[2], k], 0.0))),
        lift!(permute(rnd(device, [s[0], s[1], s[2], k], 0.0))),
        lift!(permute(rnd(device, [s[0], s[1], s[2], v], 0.0))),
        lift!(permute(rnd(device, [s[0], s[1], s[2], k], -1.0))),
        lift!(permute(rnd(device, [s[0], s[1], s[2], k], 0.0))),
        lift!(permute(rnd(device, [s[0], s[1], s[2], v], 0.0))),
        lift!(rnd(device, [batch, heads, k, v], 0.0)),
    ]
}

/// The reported failure, end to end: a Balanced-checkpointed graph whose op
/// inputs are permuted views must complete a real fwd+bwd, with the fused
/// forward engaged and no fallback.
#[test]
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
    let (out, _state) = match chunk_dispatch::<AdBal>(
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
    let (fwd, _) = fused_calls();
    assert!(fwd > 0, "the fused forward kernel never launched");
    let (copies, fallbacks) = contiguous_copies();
    assert!(
        copies >= 6,
        "the strided views were not materialized ({copies} copies): the seam \
         read them as if they were contiguous"
    );
    assert_eq!(
        fallbacks, 0,
        "a materialization did not come back row-major: {fallbacks} fallbacks"
    );

    // A real fwd+bwd, with ops after the op so the backward reaches it through
    // a chain — and a recompute of the permuted parents.
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
    // differently-ordered copy of it: same numbers as the tensor path.
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
    let d = rel_diff(&out, &plain_out);
    assert!(d < 1e-4, "fused vs tensor path on a strided input: rel={d:.2e}");
}
