#![cfg(all(feature = "cuda", feature = "autodiff"))]
//! The strong gradient gate: on CUDA, on the trainer's backend, the chunk op
//! must produce the derivative of a real KDA **parameter** — and it must agree
//! with a value derived independently of burn's autodiff.
//!
//! ## Why this file exists
//!
//! For the whole history of this project the attention arm had NO gradient.
//! `chunk_wy_forward_autodiff_s` ran the forward on the bare backend and
//! wrapped the result in one hand-rolled node; under
//! `BalancedCheckpointing` that node was `UnTracked`, so its output was a
//! LEAF and nothing downstream could send a gradient back. A real run printed
//! `fused kda=30/0` — 30 fused forwards, 0 adjoints — while every other arm
//! trained normally, so the loss curve looked healthy.
//!
//! The CUDA tests that guarded the old path all assert the same weak thing:
//! that some gradient tensor is non-zero. A **wrong** gradient, a gradient
//! through the wrong input, and a gradient of a different function all pass
//! that. And a numerical comparison between the fused forward and the ops
//! forward cannot catch a *missing* gradient either: they compute the SAME
//! function, so it compares a thing to itself.
//!
//! So the reference here is **central finite differences of the same forward
//! on the same weights**, evaluated on this box: no autodiff, no second
//! implementation, no reference file. Same function, different method of
//! differentiation. A zero gradient fails it. A wrong gradient fails it. A
//! gradient of a different function fails it.
//!
//! The parameter differentiated is `q_proj.weight` — a real `nn::Linear`
//! weight, i.e. exactly the thing that was frozen. Coordinates are picked
//! deterministically across the matrix and every one is printed, so the
//! measurement is reproducible rather than one summary number.
//!
//! ## What it does NOT prove
//!
//! That the fused adjoint kernel is correct. The strong statement it does
//! make is about *whichever* arm the seam took, and the arm is printed:
//! `ops_path` counts dispatches that fell through to `chunk_wy_forward_impl`
//! on the incoming tensors (burn builds that graph itself, so the gradient is
//! burn's), `fused_bwd` counts launches of the hand-written adjoint kernel.
//!
//! Run it:
//! `cargo test -p burn-kda --release --features cuda,autodiff --test ops_grad_cuda -- --nocapture`

use burn::module::Param;
use burn::tensor::{Device, Distribution, Tensor, TensorData};
use burn_autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn_autodiff::Autodiff;
use burn_gdn2::CudaBare;
use burn_kda::{DecayFn, KdaConfig, KdaModule};

/// The trainer's backend: `Autodiff<CudaBare, BalancedCheckpointing>`.
type AdBal = Autodiff<CudaBare, BalancedCheckpointing>;

/// How many coordinates of `q_proj.weight` are checked. Eight, spread across
/// the matrix: enough that a bug which only touches one region (a chunk
/// boundary, the first or last row) cannot hide behind a lucky draw, few
/// enough that the 2 x N forward passes stay cheap.
const N_COORDS: usize = 8;

/// Central-difference step. Chosen, not tuned to a number: f32 here gives a
/// truncation error O(h^2) and a round-off error O(eps_f32/h) ~ 6e-6/h, so
/// the two balance near h = 1e-2 for a function whose third derivative is the
/// same size as its first. The test prints the same comparison at h AND h/10,
/// so the residual's dependence on h is visible instead of assumed — a
/// truncation-dominated residual falls when h falls, a round-off-dominated one
/// rises.
const H: f32 = 1e-2;

/// The bar for |analytic - finite-difference| / |finite-difference|.
///
/// A BAR, not a measured value: it is what central differences at [`H`] can
/// deliver in f32 with TF32 matmuls on this GPU. Whoever runs this first
/// should paste the printed value into the commit message — the value, not a
/// widened bar. If the measured value sits at the bar, that is a finding about
/// the loss surface and the fix is a better reference step, never a looser
/// assertion.
const REL_BAR: f32 = 5e-2;

fn cfg() -> KdaConfig {
    KdaConfig {
        hidden_size: 64,
        num_heads: 2,
        head_dim: 16,
        num_v_heads: Some(2),
        use_short_conv: false,
        decay_fn: DecayFn::Sigmoid,
        chunk_size: 16,
        ..Default::default()
    }
}

/// Host f32s of a tensor.
fn host<const D: usize>(t: &Tensor<D>) -> Vec<f32> {
    t.clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `d(loss)/d(q_proj.weight)` on the trainer's backend, against central
/// differences of the same forward. Prints every coordinate; asserts the
/// relative deviation against [`REL_BAR`].
#[test]
fn the_chunk_op_produces_a_real_parameter_gradient_on_cuda() {
    let dev = Device::cuda(0);
    dev.seed(11); // unseeded draws make the FD comparison irreproducible
    let ad = dev.clone().autodiff();
    let c = cfg();
    let km = KdaModule::new(&c, 0.9, &ad);
    // The trainer's own shape of thing: a graph tensor in, `Balanced` out.
    let x = Tensor::<3>::random([1, 64, c.hidden_size], Distribution::Normal(0.0, 1.0), &ad)
        .require_grad();

    burn_gdn2::reset_fused_calls();
    let (y, s) = km.forward_train_state::<AdBal>(x.clone(), None);
    println!("seam after forward: {}", show(burn_gdn2::seam_counts()));

    let loss = y.powf_scalar(2.0).sum().add(s.powf_scalar(2.0).sum());
    let grads = loss.backward();
    let counts = burn_gdn2::seam_counts();
    println!("seam after backward: {}", show(counts));

    // The arm must not be the one that stayed frozen. q_proj's weight is
    // upstream of everything the chunk recurrence reads.
    let gw = km
        .q_proj
        .weight
        .grad(&grads)
        .expect("q_proj.weight got NO gradient: the attention arm is still frozen")
        .clone();
    let analytic = host(&gw);
    assert!(
        analytic.iter().fold(0.0f32, |m, v| m.max(v.abs())) > 0.0,
        "q_proj.weight got an all-zero gradient"
    );

    // The reference: the SAME forward on the SAME weights with ONE coordinate
    // of the weight moved by +-h, and no backward anywhere in the comparison.
    // The state is rebuilt from zeros exactly as the forward above does.
    let base = km.clone();
    let w0 = host(&base.q_proj.weight.val());
    let shape = [c.hidden_size, c.num_heads * c.head_dim];
    assert_eq!(w0.len(), shape[0] * shape[1], "unexpected q_proj.weight shape");
    let loss_at = |w: Vec<f32>| -> f32 {
        let mut m = base.clone();
        m.q_proj.weight = Param::from_tensor(Tensor::<2>::from_data(TensorData::new(w, shape), &ad));
        let (yy, ss) = m.forward_train_state::<AdBal>(x.clone(), None);
        yy.powf_scalar(2.0)
            .sum()
            .add(ss.powf_scalar(2.0).sum())
            .into_scalar::<f32>()
    };
    let fd_at = |idx: usize, h: f32| -> f32 {
        let mut up = w0.clone();
        up[idx] += h;
        let mut dn = w0.clone();
        dn[idx] -= h;
        (loss_at(up) - loss_at(dn)) / (2.0 * h)
    };

    let step = w0.len() / N_COORDS;
    let mut worst_rel = 0.0f32;
    for i in 0..N_COORDS {
        let idx = i * step;
        let g = analytic[idx];
        assert!(
            g.abs() > 0.0,
            "q_proj.weight[{idx}] is zero: the finite-difference check would be vacuous"
        );
        let fd = fd_at(idx, H);
        let fd10 = fd_at(idx, H / 10.0);
        let rel = (fd - g).abs() / fd.abs().max(1e-30);
        let rel10 = (fd10 - g).abs() / fd10.abs().max(1e-30);
        worst_rel = worst_rel.max(rel);
        println!(
            "q_proj.weight[{idx:5}] (w={:+.5}): autodiff {g:+.6e}  fd(h) {fd:+.6e}  \
             fd(h/10) {fd10:+.6e}  rel {rel:.2e}  rel(h/10) {rel10:.2e}",
            w0[idx]
        );
        assert!(
            rel < REL_BAR,
            "q_proj.weight[{idx}]: d(loss)/dw = {g:+.6e} by autodiff, {fd:+.6e} by central \
             differences (h={H}) - relative {rel:.2e}, over the {REL_BAR} bar. The gradient \
             does not match the function."
        );
    }
    println!("worst relative deviation {worst_rel:.3e}");

    // Which arm produced that gradient. `ops_path` next to a zero
    // `custom_node_backward` is the statement that burn's own graph carried
    // the backward; `fused_bwd` is the hand-written adjoint kernel. Either is
    // a gradient — this test's claim is that the SEAM's gradient matches the
    // function, and it prints which arm it checked.
    assert!(
        counts.4 > 0 || counts.2 > 0,
        "neither arm ran a backward: {}. The seam is not differentiable at all.",
        show(counts)
    );
    println!(
        "arm: {}",
        if counts.2 > 0 {
            "FUSED adjoint kernel"
        } else {
            "OPS path (burn's own graph)"
        }
    );
}

fn show(c: (u64, u64, u64, u64, u64, u64)) -> String {
    format!(
        "asked={} fused_fwd={} fused_bwd={} declined={} ops_path={} custom_node_bwd={}",
        c.0, c.1, c.2, c.3, c.4, c.5
    )
}
