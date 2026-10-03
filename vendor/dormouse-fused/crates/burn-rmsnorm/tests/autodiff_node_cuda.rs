#![cfg(feature = "cuda")]
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
//! The gate the dispatch guard made impossible until the node existed: the
//! fused RMSNorm kernel must RUN on the trainer's backend
//! (`Autodiff<Cuda, BalancedCheckpointing>`) — through ONE tracked node, with
//! gradients that match the tensor path, and the counter that says the arm
//! ran (ADR-0019: a fallback is excused only when the reader can tell).
//!
//! The fixture is trainer-shaped on purpose: parents at least one op away
//! from their leaves (`GradInBackward`, not `Grad` — asking `is_require_grad`
//! instead of `is_tracked` is the decline that kept the KDA op off the
//! trainer's own graph), under `BalancedCheckpointing`, the strategy the
//! trainer runs.

use burn::backend::{Autodiff, AutodiffBackend};
use burn::tensor::{Device, Distribution, Tensor, TensorData};
use burn_autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn_rmsnorm::ops::rmsnorm_node_autodiff_s;
use burn_rmsnorm::RMSNorm;
use std::sync::Mutex;

type Inner = burn_cubecl::CubeBackend;
type AdBal = Autodiff<Inner, BalancedCheckpointing>;

/// The seam counters are process globals; the module test and the node test
/// both touch them, so parallel test threads race the snapshots. Same pattern
/// as `fused_kernel_gate.rs`'s SEAM.
static SEAM: Mutex<()> = Mutex::new(());

const EPS: f32 = 1e-5;
const ROWS: usize = 8;
const D: usize = 32;

fn bare_dev() -> Device {
    Device::cuda(0)
}

fn trainer_dev() -> Device {
    Device::cuda(0).autodiff().gradient_checkpointing()
}

fn host(d: TensorData) -> Vec<f32> {
    let mut v = Vec::new();
    for c in d.bytes.chunks_exact(4) {
        v.push(f32::from_le_bytes(c.try_into().unwrap()));
    }
    v
}

/// Lift a bare value onto `AdBal` as a tracked LEAF.
fn lift<const N: usize>(raw: &Tensor<N>) -> Tensor<N> {
    let bare = raw
        .clone()
        .try_into_primitive::<Inner>()
        .expect("bare cuda tensor");
    Tensor::from_primitive::<AdBal>(<AdBal as AutodiffBackend>::from_inner(bare)).require_grad()
}

/// One op away from the leaf, exactly like the trainer's norm input (the
/// output of a projection, so its requirement is `GradInBackward`).
fn project<const N: usize>(t: Tensor<N>) -> Tensor<N> {
    t.mul_scalar(0.5)
}

/// Gradients are read at the `require_grad()` LEAVES, not at the projected
/// intermediates: `Gradients::consume` REMOVES a `GradInBackward` node's
/// gradient when its own step runs (burn-autodiff `src/grads.rs:103-107`), so
/// `x.grad()` on the projection output is `None` after a healthy backward —
/// the first run of this fixture panicked there while the node was working.
/// `Grad` leaves are read, not removed (`grads.rs:98-102`).

fn draw() -> (Tensor<2>, Tensor<1>) {
    let dev = bare_dev();
    // DETERMINISTIC data: the device RNG is never seeded (the §3.3 seed saga),
    // and a gate whose worst-case error moves between runs is a gate nobody
    // can compare a regression against. Same construction as lib.rs's
    // `forward_matches_scalar_reference`.
    let xs: Vec<f32> = (0..ROWS * D).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
    let x = Tensor::<2>::from_data(TensorData::new(xs, [ROWS, D]), &dev);
    let w: Vec<f32> = (0..D).map(|i| 0.5 + 0.25 * i as f32).collect();
    let w = Tensor::<1>::from_data(TensorData::new(w, [D]), &dev);
    (x, w)
}

/// d(sum(y²))/d(x), d(sum(y²))/d(w) through the NODE arm; the loss value too.
fn node_arm(raw_x: &Tensor<2>, raw_w: &Tensor<1>) -> (Vec<f32>, Vec<f32>, f32) {
    let x_leaf = lift(raw_x);
    let w_leaf = lift(raw_w);
    let x = project(x_leaf.clone());
    let w = project(w_leaf.clone());
    let y = rmsnorm_node_autodiff_s::<Inner, BalancedCheckpointing>(x.clone(), w.clone(), EPS)
        .expect("the node arm must engage on the trainer's backend");
    let lval = y.clone().powf_scalar(2.0).sum().into_scalar();
    let grads = y.powf_scalar(2.0).sum().backward();
    let dx = x_leaf
        .grad(&grads)
        .expect("x leaf got no gradient: the node is disconnected from the graph");
    let dw = w_leaf
        .grad(&grads)
        .expect("w leaf got no gradient: the node is disconnected from the graph");
    (host(dx.into_data()), host(dw.into_data()), lval)
}

/// The same gradients through burn's own autograd over the tensor path.
fn tensor_arm(raw_x: &Tensor<2>, raw_w: &Tensor<1>) -> (Vec<f32>, Vec<f32>, f32) {
    let x_leaf = lift(raw_x);
    let w_leaf = lift(raw_w);
    let x = project(x_leaf.clone());
    let w = project(w_leaf.clone());
    let rms = x
        .clone()
        .powf_scalar(2.0)
        .mean_dim(1)
        .add_scalar(EPS)
        .sqrt(); // [rows, 1]
    let y = x.clone().div(rms).mul(w.clone().reshape([1, D]));
    let lval = y.clone().powf_scalar(2.0).sum().into_scalar();
    let grads = y.powf_scalar(2.0).sum().backward();
    let dx = x_leaf.grad(&grads).expect("x leaf got no gradient");
    let dw = w_leaf.grad(&grads).expect("w leaf got no gradient");
    (host(dx.into_data()), host(dw.into_data()), lval)
}

#[test]
fn the_node_runs_on_the_trainer_backend_and_gradients_match_the_tensor_path() {
    let _g = SEAM.lock().unwrap_or_else(|e| e.into_inner());
    let (raw_x, raw_w) = draw();

    let before = burn_rmsnorm::fused::arm_counts();
    let (dx_n, dw_n, l_n) = node_arm(&raw_x, &raw_w);
    let after = burn_rmsnorm::fused::arm_counts();

    // THE COUNTER GATE, scoped honestly: ASKED/NODE_RAN are the MODULE's seam
    // (`RMSNorm::forward` counts the ask; `ops.rs` counts nothing) and the
    // module test below pins that pair at (1, 0, 1). This fixture calls the
    // node DIRECTLY to isolate it, so the only claim the counters can carry
    // here is that nothing else in the process touched the seam.
    assert_eq!(
        (after.0 - before.0, after.1 - before.1, after.2 - before.2),
        (0, 0, 0),
        "the seam counters moved on a direct node call"
    );

    let (dx_t, dw_t, l_t) = tensor_arm(&raw_x, &raw_w);

    // Same function, same values, both arms. The loss equality is the
    // forward's claim; the gradient closeness is the backward's.
    //
    // The bar is SCALE-RELATIVE (max |a−b| / max |reference|), not
    // per-element: a f32 backward's error is bounded by eps × the scale of
    // the values that produced it, and a near-zero entry's per-element ratio
    // carries none of that bound (measured: abs diff 4.8e-7 on a 1.6e-3
    // entry against a 47-magnitude tensor = 1e-8 scale-relative, while the
    // same pair reads 3e-4 per-element). 1e-6 is ~4000× under f32 eps and
    // ~10⁸× over the measured disagreement: a wrong sign, a missing term or
    // a consumed-then-absent gradient all cross it at once.
    let rel_l = (l_n - l_t).abs() / l_t.abs().max(1e-30);
    assert!(rel_l < 1e-5, "loss disagrees: node {l_n} tensor {l_t}");
    let scale_rel = |a: &[f32], b: &[f32]| {
        let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
            / scale.max(1e-30)
    };
    let rel_dx = scale_rel(&dx_n, &dx_t);
    let rel_dw = scale_rel(&dw_n, &dw_t);
    assert!(
        rel_dx < 1e-6,
        "dx disagrees (scale-relative): {rel_dx:e} (node[0..4]={:?} tensor[0..4]={:?})",
        &dx_n[..4],
        &dx_t[..4]
    );
    assert!(
        rel_dw < 1e-6,
        "dw disagrees (scale-relative): {rel_dw:e} (node[0..4]={:?} tensor[0..4]={:?})",
        &dw_n[..4],
        &dw_t[..4]
    );
    // Nonzero and finite: a gradient of zeros PASSES a closeness check and
    // still means a frozen arm.
    assert!(dx_n.iter().cloned().fold(0.0f32, f32::max) > 0.0);
    assert!(dw_n.iter().cloned().fold(0.0f32, f32::max) > 0.0);
    assert!(dx_n.iter().all(|v| v.is_finite()));
    assert!(dw_n.iter().all(|v| v.is_finite()));
    eprintln!("the rmsnorm NODE arm: loss rel {rel_l:e}, dx rel {rel_dx:e}, dw rel {rel_dw:e}");
}

#[test]
fn the_module_forward_engages_the_node_on_the_trainer_device() {
    let _g = SEAM.lock().unwrap_or_else(|e| e.into_inner());
    let dev = trainer_dev();
    let norm = RMSNorm::new(D, EPS, &dev);
    let raw = Tensor::<3>::random([2, 16, D], Distribution::Default, &bare_dev());
    let x = lift(&raw);

    let before = burn_rmsnorm::fused::arm_counts();
    let y = norm.forward(x.clone());
    let after = burn_rmsnorm::fused::arm_counts();

    assert_eq!(after.0 - before.0, 1, "one ask");
    assert_eq!(after.1 - before.1, 0, "nothing skipped");
    assert_eq!(after.2 - before.2, 1, "the module must take the NODE arm");

    // And the graph the node built is live: backward reaches the input leaf.
    let grads = y.powf_scalar(2.0).sum().backward();
    let gx = x
        .grad(&grads)
        .expect("no gradient through the module's fused node");
    let g = host(gx.into_data());
    assert!(g.iter().all(|v| v.is_finite()));
    assert!(g.iter().cloned().fold(0.0f32, f32::max) > 0.0);
}

#[test]
fn the_kill_switch_declines_both_arms_and_still_counts() {
    // The A/B of the node against the tensor path is one binary and an env
    // var — but the switch must be visible in the counters, not silent
    // (ADR-0019). The var is checked at process start, so this arm only
    // asserts when the harness ran with DM_RMSNORM_FUSED=0 and otherwise
    // reports that it was skipped.
    if std::env::var("DM_RMSNORM_FUSED").as_deref() != Ok("0") {
        eprintln!("kill-switch arm skipped: run with DM_RMSNORM_FUSED=0 to exercise it");
        return;
    }
    // This test reads the process-global counters too — without the lock it
    // races the two siblings, which is the failure this pattern exists to end.
    let _g = SEAM.lock().unwrap_or_else(|e| e.into_inner());
    let dev = trainer_dev();
    let norm = RMSNorm::new(D, EPS, &dev);
    let raw = Tensor::<3>::random([2, 16, D], Distribution::Default, &bare_dev());
    let x = lift(&raw);
    let before = burn_rmsnorm::fused::arm_counts();
    let _ = norm.forward(x);
    let after = burn_rmsnorm::fused::arm_counts();
    assert_eq!(after.0 - before.0, 1, "the ask is still counted");
    assert_eq!(after.1 - before.1, 1, "the skip is counted");
    assert_eq!(after.2 - before.2, 0, "no node launch under the switch");
}
