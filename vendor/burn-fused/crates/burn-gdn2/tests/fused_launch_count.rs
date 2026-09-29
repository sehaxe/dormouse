#![cfg(all(feature = "cuda", feature = "autodiff"))]
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
//! How many CUDA kernels does one fused chunked forward actually issue?
//!
//! The crate's only counter before this was `cuda_dispatch::fused_calls()`, which
//! counts CALLS. A call count cannot see a call that grew an extra kernel, and
//! `fused_chunk_forward` (the module path) used to route through
//! `fused_chunk_forward_scratch` and throw the scratch away — paying for the
//! trajectory-export kernel, a second full-size copy of the initial state, and
//! two over-allocated export buffers on every forward, for an input the backward
//! adjoint is the only consumer of.
//!
//! The reference is two hand-written kernels
//! (`research/papers/spec-flashkda.md` §1: `fwd_kernel1.cuh` 587 lines +
//! `fwd_kernel2.cuh` 840 lines), so three was one too many on the module path.
//!
//! Run: cargo test --release --features "cuda,autodiff" --test fused_launch_count -- --nocapture

use burn::tensor::{Device, Distribution, Tensor};
use burn_gdn2::alloc_trace::{
    fused_launch_log, fused_launches, fused_launches_per_call, reset_fused_launches,
};
use burn_gdn2::kernel::chunk_cube::cuda::{fused_chunk_forward, fused_chunk_forward_scratch};
use burn_gdn2::CudaBare;

/// `(batch, heads, time, K, V, chunk)`. chunk 16 because the fused kernels
/// decline anything larger (the `K/exp(cumsum g)` factor underflows f32 past
/// chunk ~17 at the K3 decay floor) — so this is the shape the module path
/// actually takes.
const SHAPE: (usize, usize, usize, usize, usize, usize) = (2, 4, 256, 64, 64, 16);

fn inputs(
    b: usize,
    h: usize,
    t: usize,
    k: usize,
    v: usize,
    dev: &Device,
) -> (Tensor<4>, Tensor<4>, Tensor<4>, Tensor<4>, Tensor<4>, Tensor<4>, Tensor<4>) {
    let r4 = |shape: [usize; 4], m: f64, s: f64| {
        Tensor::<4>::random(shape, Distribution::Normal(m, s), dev)
    };
    (
        r4([b, h, t, k], 0.0, 0.1),
        r4([b, h, t, k], 0.0, 0.1),
        r4([b, h, t, v], 0.0, 0.1),
        r4([b, h, t, k], -0.05, 0.1),
        Tensor::<4>::random([b, h, t, k], Distribution::Uniform(0.0, 0.1), dev),
        r4([b, h, t, v], 0.5, 0.1),
        r4([b, h, k, v], 0.0, 0.1),
    )
}

/// The module path: TWO launches, intra then inter. The export kernel is the
/// backward's input and must not appear here.
#[test]
fn module_forward_issues_two_launches() {
    type B = CudaBare;
    let dev: Device = Default::default();
    dev.seed(7);
    let (b, h, t, k, v, c) = SHAPE;
    let (q, kk, vv, g, bb, w, st) = inputs(b, h, t, k, v, &dev);
    let scale = (k as f64).powf(-0.5);

    reset_fused_launches();
    let r = fused_chunk_forward::<B>(q, kk, vv, g, bb, w, st.clone(), scale, c);
    let (launches, calls) = fused_launches();
    let log = fused_launch_log();
    let per = fused_launches_per_call();

    println!("module path: {launches} launches over {calls} engaged call(s) = {per:.1}/call");
    println!("  order: {log:?}");

    assert!(r.is_some(), "the fused forward declined its own test shape");
    assert_eq!(calls, 1, "exactly one engaged call expected");
    assert_eq!(
        launches, 2,
        "the module forward must issue 2 launches, issued {launches}: {log:?}"
    );
    assert_eq!(
        log,
        vec!["gdn2_chunk_intra_kernel", "gdn2_chunk_inter_kernel"],
        "the launch set and its order are the contract, not just the count"
    );
}

/// The scratch path — the backward's entry — still issues THREE, because the
/// export is what it is for. This is the arm that must not be "optimised" the
/// same way: dropping its export would break the adjoint silently.
#[test]
fn the_scratch_path_still_exports() {
    type B = CudaBare;
    let dev: Device = Default::default();
    dev.seed(7);
    let (b, h, t, k, v, c) = SHAPE;
    let (q, kk, vv, g, bb, w, st) = inputs(b, h, t, k, v, &dev);
    let scale = (k as f64).powf(-0.5);

    reset_fused_launches();
    let r = fused_chunk_forward_scratch::<B>(q, kk, vv, g, bb, w, st, scale, c);
    let (launches, calls) = fused_launches();
    let log = fused_launch_log();
    let per = fused_launches_per_call();

    println!("scratch path: {launches} launches over {calls} engaged call(s) = {per:.1}/call");
    println!("  order: {log:?}");

    assert!(r.is_some(), "the fused forward declined its own test shape");
    assert_eq!(calls, 1, "exactly one engaged call expected");
    assert_eq!(
        launches, 3,
        "the scratch path must keep its export kernel, issued {launches}: {log:?}"
    );
    assert_eq!(
        log,
        vec![
            "gdn2_chunk_intra_kernel",
            "gdn2_chunk_inter_kernel",
            "gdn2_chunk_trajectory_export_kernel",
        ],
        "the export must come last: the backward reads buffers the inter kernel writes"
    );
}

/// The two entry points must return the SAME forward answer. Splitting them is
/// only free if the extra arguments change nothing about `out` or `state`, and
/// that is exactly the kind of claim a launch-count test does not make.
#[test]
fn the_two_entry_points_agree_on_the_forward() {
    type B = CudaBare;
    let dev: Device = Default::default();
    dev.seed(7);
    let (b, h, t, k, v, c) = SHAPE;
    let (q, kk, vv, g, bb, w, st) = inputs(b, h, t, k, v, &dev);
    let scale = (k as f64).powf(-0.5);

    let (out_a, state_a) =
        fused_chunk_forward::<B>(q.clone(), kk.clone(), vv.clone(), g.clone(), bb.clone(), w.clone(), st.clone(), scale, c)
            .expect("fused forward declined");
    let (out_b, state_b, _io) =
        fused_chunk_forward_scratch::<B>(q, kk, vv, g, bb, w, st, scale, c)
            .expect("fused forward declined");

    let d_out = (out_a.clone() - out_b.clone())
        .abs()
        .max()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .fold(0.0f32, f32::max);
    let d_st = (state_a - state_b)
        .abs()
        .max()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .fold(0.0f32, f32::max);
    println!("module vs scratch entry: out {d_out:.3e}, state {d_st:.3e}");
    assert_eq!(d_out, 0.0, "out differs between the two entry points: {d_out:.3e}");
    assert_eq!(d_st, 0.0, "state differs between the two entry points: {d_st:.3e}");
}
