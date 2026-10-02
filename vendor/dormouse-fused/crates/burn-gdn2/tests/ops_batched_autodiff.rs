// The custom node's analytic backward, on a RAGGED tail, for both arms of the
// chunked-WY forward. CPU (NdArray) on purpose: the question here is not about
// CUDA, it is about a shape contract.
//
// ## The contract
//
// `chunk_wy_forward_batched` hands the backward a scratch whose last chunk is
// zero-padded to `chunk_size`, while the loop arm hands it a scratch trimmed to
// the chunk's real length. `Backward for ChunkWy` reads the real length from
// `time`/`chunk_size` (`c_real`) and pads its own re-derived inputs to whatever
// `c_pad` the scratch carries, so both are consumable — but nothing asserted
// that, and the pre-existing custom-node gradient tests use a `T` that divides
// evenly, where the two arms' scratches have the same shape and the difference
// cannot show up.
//
// A ragged tail is therefore the case that would catch a padded scratch the
// backward cannot read, and it is checked here three ways: the custom node's
// analytic gradient against central differences of the plain path, for each arm
// separately, and the two arms' gradients against each other.
//
// Run: cargo test -p burn-gdn2 --features autodiff --test ops_batched_autodiff -- --nocapture
//
// ## KNOWN RUNNER FLAKE — re-run before believing a red here
//
// On identical main code this test flipped ok / FAIL across CI runner
// allocations with BYTE-IDENTICAL failing numbers every time: `loss
// 77.005417, autodiff -1.002479e-2, fd -1.068115e-2, rel 6.15e-2` (runs
// 36948985425 ok / 36950763703 FAIL / 36951111160 ok / 36972953116 ok /
// 36979996862 FAIL / 36982262271 FAIL, 2026-10-02). The axis is
// burn-ndarray's `blas-openblas` DYNAMIC_ARCH kernel selection per host CPU —
// the runner's host is part of the arithmetic, and a ~2-ulp kernel-order
// difference at loss ~77 amplified by the fd contrast's 1/2h = 50 lands right
// on the 5e-2 bar. `OPENBLAS_NUM_THREADS=1` pinns the scheduler, NOT the host
// kernel, and did not stop the flip.
//
// The bar (5e-2) therefore measures the runner's host on this coordinate, and
// was NOT widened (a proposed 5e-2 -> 0.25 was rejected — the reviewed
// widening lives in wt/kdaci d53229d, not merged). Real contract breaks are
// O(1)-wrong or shape-panic, orders above this noise. IF THIS GOES RED: run
// the same test once more before investigating; a repeated red with a
// DIFFERENT number, or a red that survives a local re-run on your own host,
// is a real defect — name `num_diff`'s worst coordinate in the report.
#![cfg(feature = "autodiff")]
#![allow(deprecated)]

use burn::backend::{Autodiff, NdArray};
use burn::tensor::{Device, Distribution, Tensor, TensorData};
use burn_gdn2::{chunk_path, chunk_wy_forward, chunk_wy_forward_autodiff, ChunkPath};

/// Central-difference step, from `burn-kda/tests/ops_grad_cuda.rs`: truncation
/// O(h^2) against f32 round-off O(eps/h).
const H: f32 = 1e-2;
/// The bar from the same file.
const REL_BAR: f32 = 5e-2;
const N_COORDS: usize = 6;

// 40 = 2 full chunks of 16 + one of 8. The ragged tail is the point.
const B: usize = 1;
const NH: usize = 2;
const T: usize = 40;
const K: usize = 16;
const V: usize = 16;
const CHUNK: usize = 16;
const SHAPE: [usize; 4] = [B, NH, T, K];

#[derive(Clone)]
struct Arms {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    b: Vec<f32>,
    w: Vec<f32>,
    state: Vec<f32>,
}

fn draw(dev: &Device) -> Arms {
    dev.seed(5);
    let f = |t: Tensor<4>| -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let kraw = Tensor::<4>::random(SHAPE, Distribution::Normal(0.0, 1.0), dev);
    let k = kraw.clone() / kraw.powf_scalar(2.0).sum_dim(3).sqrt();
    Arms {
        q: f(Tensor::<4>::random(
            SHAPE,
            Distribution::Normal(0.0, 1.0),
            dev,
        )),
        k: f(k),
        v: f(Tensor::<4>::random(
            SHAPE,
            Distribution::Normal(0.0, 1.0),
            dev,
        )),
        g: f(Tensor::<4>::random(
            SHAPE,
            Distribution::Uniform(-5.0, -0.01),
            dev,
        )),
        b: f(Tensor::<4>::random(
            SHAPE,
            Distribution::Uniform(0.0, 1.0),
            dev,
        )),
        w: f(Tensor::<4>::random(
            SHAPE,
            Distribution::Uniform(0.5, 1.0),
            dev,
        )),
        state: f(Tensor::<4>::random(
            [B, NH, K, V],
            Distribution::Normal(0.0, 0.5),
            dev,
        )),
    }
}

fn t(data: &[f32], shape: [usize; 4], dev: &Device, grad: bool) -> Tensor<4> {
    let t = Tensor::<4>::from_data(TensorData::new(data.to_vec(), shape.to_vec()), dev);
    if grad {
        t.require_grad()
    } else {
        t
    }
}

/// `q_override` exists so a caller can hold the SAME `q` tensor it will ask
/// for the gradient of. Building it twice makes two nodes, and the backward
/// then registers against the one that is not the one being read — which is
/// what "q got no gradient" means.
fn call(
    a: &Arms,
    dev: &Device,
    node: bool,
    q_override: Option<Tensor<4>>,
) -> (Tensor<4>, Tensor<4>) {
    let args = || {
        (
            q_override
                .clone()
                .unwrap_or_else(|| t(&a.q, SHAPE, dev, true)),
            t(&a.k, SHAPE, dev, true),
            t(&a.v, SHAPE, dev, true),
            t(&a.g, SHAPE, dev, true),
            t(&a.b, SHAPE, dev, true),
            t(&a.w, SHAPE, dev, true),
            t(&a.state, [B, NH, K, V], dev, true),
        )
    };
    let (q, k, v, g, b, w, s) = args();
    let scale = (K as f64).powf(-0.5);
    if node {
        chunk_wy_forward_autodiff::<NdArray>(q, k, v, g, b, w, s, scale, CHUNK)
            .expect("the custom node declined on graph tensors: the arm under test never ran")
    } else {
        chunk_wy_forward(q, k, v, g, b, w, s, scale, CHUNK)
    }
}

fn loss_of(a: &Arms, dev: &Device, node: bool) -> Tensor<1> {
    let (o, s) = call(a, dev, node, None);
    o.powf_scalar(2.0).sum().add(s.powf_scalar(2.0).sum())
}

fn host(t: &Tensor<4>) -> Vec<f32> {
    t.clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// d(loss)/dq through the custom node, and the loss value.
fn analytic(a: &Arms, dev: &Device) -> (Vec<f32>, f32) {
    let q = t(&a.q, SHAPE, dev, true);
    let (o, s) = call(a, dev, true, Some(q.clone()));
    let l = o.powf_scalar(2.0).sum().add(s.powf_scalar(2.0).sum());
    let lval = l.clone().into_scalar::<f32>();
    let grads = l.backward();
    (
        host(
            &q.grad(&grads)
                .expect("q got no gradient: the arm is frozen"),
        ),
        lval,
    )
}

/// Central differences of the same loss, no autodiff in the loop.
fn fd(a: &Arms, dev: &Device, idx: usize, h: f32) -> f32 {
    let at = |q: Vec<f32>| -> f32 {
        let mut b = a.clone();
        b.q = q;
        loss_of(&b, dev, false).into_scalar::<f32>()
    };
    let mut up = a.q.clone();
    up[idx] += h;
    let mut dn = a.q.clone();
    dn[idx] -= h;
    (at(up) - at(dn)) / (2.0 * h)
}

#[test]
fn the_custom_node_backward_reads_a_padded_scratch() {
    let dev = Device::ndarray().autodiff();
    let a = draw(&dev);
    assert_eq!(T % CHUNK, CHUNK / 2, "this test is about a ragged tail");

    let mut grads = Vec::new();
    for path in [ChunkPath::Batched, ChunkPath::Loop] {
        burn_gdn2::set_chunk_path(path);
        let (g, lval) = analytic(&a, &dev);
        assert!(
            g.iter().fold(0.0f32, |m, v| m.max(v.abs())) > 0.0,
            "{path:?}: d(loss)/dq is all zero"
        );
        grads.push(g);
        let step = a.q.len() / N_COORDS;
        let mut worst = 0.0f32;
        println!("--- {path:?} (loss {lval:.6}) ---");
        for j in 0..N_COORDS {
            let idx = j * step;
            let ga = grads.last().unwrap()[idx];
            assert!(ga.abs() > 0.0, "{path:?}: q[{idx}] is zero — vacuous check");
            let f = fd(&a, &dev, idx, H);
            let rel = (f - ga).abs() / f.abs().max(1e-30);
            worst = worst.max(rel);
            println!("  q[{idx:4}]: autodiff {ga:+.6e}  fd {f:+.6e}  rel {rel:.2e}");
            assert!(
                rel < REL_BAR,
                "{path:?}: d(loss)/dq[{idx}] = {ga:+.6e} by the custom node's analytic adjoint, \
                 {f:+.6e} by central differences — relative {rel:.2e}, over the {REL_BAR} bar"
            );
        }
        println!(
            "  worst {rel_bar_line}",
            rel_bar_line = format!("{worst:.2e}")
        );
    }
    burn_gdn2::set_chunk_path(ChunkPath::Batched);
    assert_ne!(chunk_path(), ChunkPath::Loop, "the arm did not take");

    let (x, y) = (&grads[0], &grads[1]);
    let d = x
        .iter()
        .zip(y.iter())
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f32, f32::max);
    let scale = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    println!("arm-vs-arm max |dgdq| {d:.3e} (scale {scale:.3e})");
    assert!(
        d / scale.max(1e-30) < 1e-3,
        "the two arms' custom-node gradients differ by {} relative on a ragged tail",
        d / scale.max(1e-30)
    );
}
