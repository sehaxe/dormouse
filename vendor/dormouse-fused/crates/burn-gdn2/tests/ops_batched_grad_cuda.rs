// The batched chunk arm's GRADIENT, on the trainer's backend, against central
// differences of the same forward. Run:
//   cargo test -p burn-gdn2 --features cuda,autodiff --test ops_batched_grad_cuda -- --nocapture
//
// ## Method: copied, not reinvented
//
// This is `burn-kda/tests/ops_grad_cuda.rs`'s method, unchanged: the reference
// is central finite differences of the SAME forward on the SAME inputs, on
// `Autodiff<CudaBare, BalancedCheckpointing>` — the trainer's backend, the
// strategy whose checkpoint leaves once made this whole arm gradient-free. Same
// step `H = 1e-2`, same bar `REL_BAR = 5e-2`, same `N_COORDS = 8` coordinates
// spread across the tensor, same "print h and h/10 so the residual's dependence
// on the step is visible" convention. Two definitions of "the gradient is
// right" is one too many, and the existing one has a rationale attached.
//
// The difference from that file is only the object being differentiated and the
// extra arm: this test drives the chunk forward DIRECTLY with graph tensors, so
// each arm's graph is its own, and both arms are checked against the same
// finite-difference reference. The KDA-module-level check (a real `q_proj`
// weight) is `burn-kda/tests/ops_grad_batched_cuda.rs`; this one is the cheaper
// half that says which arm the graph ran.
//
// ## Why a gradient test at all
//
// The forward agreeing is half a claim. The trainer differentiates this op, and
// a batched arm is a different graph: fewer, larger nodes, different
// checkpointing decisions, and (in the custom-node arm) a different `m_inv`
// feeding an analytic adjoint. A wrong gradient is invisible to the forward
// comparison and is exactly the failure this project already shipped once.

#![cfg(all(feature = "cuda", feature = "autodiff"))]

use burn::backend::AutodiffBackend;
use burn::tensor::{Device, Distribution, Tensor, TensorData};
use burn_autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn_autodiff::Autodiff;
use burn_gdn2::{ChunkPath, CudaBare};

type AdBal = Autodiff<CudaBare, BalancedCheckpointing>;
// burn 0.22's Tensor is backend-agnostic (the backend is a runtime dispatch
// marker), so these names carry no type parameter — the SAME type on the bare
// and the autodiff backend, which is what makes this file a test of the graph
// rather than of a type alias.
type T4 = Tensor<4>;
type T1 = Tensor<1>;

/// Central-difference step. From ops_grad_cuda.rs: truncation O(h^2) balances
/// f32 round-off O(eps/h) near h = 1e-2.
const H: f32 = 1e-2;
/// The bar for |analytic - fd| / |fd|. From ops_grad_cuda.rs.
const REL_BAR: f32 = 5e-2;
/// Coordinates checked per input. From ops_grad_cuda.rs.
const N_COORDS: usize = 8;

/// [batch, heads, time, k], v_dim == k, chunk 16 — small enough that the
/// finite-difference forward passes are quick on a shared GPU, the same chunk
/// the trainer runs, and long enough for several chunks plus a ragged tail.
const SHAPE: [usize; 4] = [1, 2, 40, 16];
const CHUNK: usize = 16;

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
    dev.seed(3);
    let [nb, nh, _nt, nk] = SHAPE;
    let f = |t: Tensor<4>| -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    // unit keys (the delta rule needs them) and b in [0,1], g in K3's range
    let kraw = Tensor::<4>::random(SHAPE, Distribution::Normal(0.0, 1.0), dev);
    let k = kraw.clone() / kraw.powf_scalar(2.0).sum_dim(3).sqrt();
    let q = Tensor::<4>::random(SHAPE, Distribution::Normal(0.0, 1.0), dev);
    let v = Tensor::<4>::random(SHAPE, Distribution::Normal(0.0, 1.0), dev);
    let b = Tensor::<4>::random(SHAPE, Distribution::Uniform(0.0, 1.0), dev);
    let w = Tensor::<4>::random(SHAPE, Distribution::Uniform(0.5, 1.0), dev);
    let g = Tensor::<4>::random(SHAPE, Distribution::Uniform(-5.0, -0.01), dev);
    let state = Tensor::<4>::random([nb, nh, nk, nk], Distribution::Normal(0.0, 0.5), dev);
    Arms {
        q: f(q),
        k: f(k),
        v: f(v),
        g: f(g),
        b: f(b),
        w: f(w),
        state: f(state),
    }
}

fn t4(data: &[f32], shape: [usize; 4], dev: &Device) -> T4 {
    let bare = Tensor::<4>::from_data(TensorData::new(data.to_vec(), shape.to_vec()), dev);
    let node = <AdBal as AutodiffBackend>::from_inner(
        bare.try_into_primitive::<CudaBare>()
            .expect("bare cuda tensor"),
    );
    Tensor::from_primitive::<AdBal>(node).require_grad()
}

fn t4s(data: &[f32], shape: [usize; 4], dev: &Device) -> T4 {
    // the STATE is a constant of the loss, not a differentiated input
    let bare = Tensor::<4>::from_data(TensorData::new(data.to_vec(), shape.to_vec()), dev);
    Tensor::from_primitive::<AdBal>(<AdBal as AutodiffBackend>::from_inner(
        bare.try_into_primitive::<CudaBare>()
            .expect("bare cuda tensor"),
    ))
}

/// A scalar loss over the output AND the output state, so the arm that only
/// reaches one of them cannot pass.
///
/// `q_override` exists so a caller can hold the SAME `q` tensor it will ask
/// for the gradient of. Building it twice makes two nodes, and the backward
/// registers against the one that is not the one being read — which is what
/// "q got no gradient" means.
fn loss(a: &Arms, dev: &Device, q_override: Option<Tensor<4>>) -> T1 {
    let (o, s) = burn_gdn2::forward::chunk_wy_forward(
        q_override.unwrap_or_else(|| t4(&a.q, SHAPE, dev)),
        t4(&a.k, SHAPE, dev),
        t4(&a.v, SHAPE, dev),
        t4(&a.g, SHAPE, dev),
        t4(&a.b, SHAPE, dev),
        t4(&a.w, SHAPE, dev),
        t4s(&a.state, [SHAPE[0], SHAPE[1], SHAPE[3], SHAPE[3]], dev),
        (SHAPE[2] as f64).powf(-0.5),
        CHUNK,
    );
    o.powf_scalar(2.0).sum().add(s.powf_scalar(2.0).sum())
}

fn host(t: &T4) -> Vec<f32> {
    t.clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// d(loss)/d(q) of one arm, plus the loss value, on the trainer's backend.
fn analytic(a: &Arms, dev: &Device) -> (Vec<f32>, f32) {
    let q = t4(&a.q, SHAPE, dev);
    let l = loss(a, dev, Some(q.clone()));
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

/// Central differences of the same loss w.r.t. one coordinate of q. No autodiff
/// anywhere in here — that is the whole point of the comparison.
fn fd(a: &Arms, dev: &Device, idx: usize, h: f32) -> f32 {
    let at = |q: Vec<f32>| -> f32 {
        let mut b = a.clone();
        b.q = q;
        loss(&b, dev, None).into_scalar::<f32>()
    };

    let mut up = a.q.clone();
    up[idx] += h;
    let mut dn = a.q.clone();
    dn[idx] -= h;
    (at(up) - at(dn)) / (2.0 * h)
}

/// Is this coordinate one central differences in f32 can speak about?
///
/// `fd(h) = (L(+h) - L(-h)) / 2h`, and `L` is an f32 scalar. The subtraction
/// cannot resolve a perturbation smaller than a few ulps of `L`, so a
/// coordinate is checkable only when `2h*|dL/dq|` clears that floor:
///
///     |dL/dq| >= FD_FLOOR_ULPS * f32::EPSILON * |L| / h
///
/// A coordinate below the floor is not a failing gradient, it is an
/// unmeasurable one — the finite-difference reference returns noise (or zero)
/// there and any bar would be a bar on noise. The MEASURED example: on this
/// shape, `q[640]` has dL/dq = 2.3e-4 on a loss of 45.79, so at h = 1e-2 the
/// perturbation is 4.7e-7 = 0.085 ulp of the loss, and at h/10 it is 0.0085
/// ulp and the difference comes back exactly 0. That is the round-off-dominated
/// signature ops_grad_cuda.rs tells you to look for ("a round-off-dominated
/// residual rises" as h falls), not a disagreement between the arms.
///
/// So coordinates are chosen from the RESOLVABLE set, deterministically, and
/// the test still fails if too few of them exist — a shrinking resolvable set
/// is the instrument degrading, and that must not pass quietly.
const FD_FLOOR_ULPS: f32 = 8.0;

#[test]
fn both_arms_produce_the_same_gradient_of_the_same_function() {
    let dev = Device::cuda(0);
    let a = draw(&dev);

    // The analytic gradient of the batched arm first: it decides which
    // coordinates the finite-difference reference is allowed to judge.
    burn_gdn2::set_chunk_path(ChunkPath::Batched);
    let (g_batched, lval) = analytic(&a, &dev);
    assert!(
        g_batched.iter().fold(0.0f32, |m, v| m.max(v.abs())) > 0.0,
        "Batched: d(loss)/dq is all zero"
    );
    let floor = FD_FLOOR_ULPS * f32::EPSILON * lval.abs() / H;
    let resolvable: Vec<usize> = (0..a.q.len())
        .filter(|i| g_batched[*i].abs() >= floor)
        .collect();
    println!(
        "loss {lval:.6}; fd floor {floor:.2e}; {} of {} coordinates resolvable in f32",
        resolvable.len(),
        a.q.len()
    );
    assert!(
        resolvable.len() >= a.q.len() / 4,
        "only {} of {} coordinates clear the f32 finite-difference floor ({floor:.2e}) — the \
         reference cannot judge this shape, and a test that skips most of its \
         coordinates is not a test",
        resolvable.len(),
        a.q.len()
    );
    // deterministic spread through the resolvable set
    let coords: Vec<usize> = (0..N_COORDS)
        .map(|j| resolvable[j * resolvable.len() / N_COORDS])
        .collect();

    let mut worst = [0.0f32; 2];
    let mut grads = vec![g_batched];
    for (i, path) in [ChunkPath::Batched, ChunkPath::Loop]
        .into_iter()
        .enumerate()
    {
        burn_gdn2::set_chunk_path(path);
        if i == 0 {
            // already computed above; only the printed loss is missing
            println!("--- {path:?} (loss {lval:.6}) ---");
        } else {
            let (g, lval) = analytic(&a, &dev);
            assert!(
                g.iter().fold(0.0f32, |m, v| m.max(v.abs())) > 0.0,
                "{path:?}: d(loss)/dq is all zero"
            );
            grads.push(g);
            println!("--- {path:?} (loss {lval:.6}) ---");
        }
        for &idx in &coords {
            let ga = grads[i][idx];
            assert!(
                ga.abs() > 0.0,
                "{path:?}: q[{idx}] is zero — the finite-difference check would be vacuous"
            );
            let f = fd(&a, &dev, idx, H);
            let f10 = fd(&a, &dev, idx, H / 10.0);
            let rel = (f - ga).abs() / f.abs().max(1e-30);
            let rel10 = (f10 - ga).abs() / f10.abs().max(1e-30);
            worst[i] = worst[i].max(rel);
            println!(
                "  q[{idx:5}]: autodiff {ga:+.6e}  fd(h) {f:+.6e}  fd(h/10) {f10:+.6e}  \
                 rel {rel:.2e}  rel(h/10) {rel10:.2e}"
            );
            assert!(
                rel < REL_BAR,
                "{path:?}: d(loss)/dq[{idx}] = {ga:+.6e} by autodiff, {f:+.6e} by central \
                 differences (h={H}) — relative {rel:.2e}, over the {REL_BAR} bar"
            );
        }
    }

    // And the two arms against each other, on the same graph: the batched arm
    // must not be a different function's gradient that happens to be a gradient.
    let d = grads[0]
        .iter()
        .zip(grads[1].iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let scale = grads[0].iter().fold(0.0f32, |m, v| m.max(v.abs()));
    println!(
        "worst relative deviation: batched {:.2e}, loop {:.2e} (bar {REL_BAR})",
        worst[0], worst[1]
    );
    println!("arm-vs-arm max |dgdq|: {d:.3e} (scale {scale:.3e})");
    assert!(
        d / scale.max(1e-30) < 1e-3,
        "the two arms' gradients differ by {} relative — they are not the same function",
        d / scale.max(1e-30)
    );
    burn_gdn2::set_chunk_path(ChunkPath::Batched);
}
