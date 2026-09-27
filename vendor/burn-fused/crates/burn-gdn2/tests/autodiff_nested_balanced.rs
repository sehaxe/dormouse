#![cfg(feature = "autodiff")]
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
//! The op in the shape the TRAINER runs it: parents that are NOT leaves, on
//! `Autodiff<_, BalancedCheckpointing>`, with a real fwd+bwd.
//!
//! Every other autodiff test in this crate (`tests/autodiff_chunk.rs`) hands
//! the op seven random leaves. That is the one shape in which the op's
//! `checkpointer.retrieve_node_output(id)` calls are trivial, so it is the
//! shape in which a broken checkpointing contract cannot show up. The trainer
//! does the opposite: `KdaModule::project` builds every input through
//! projections, activations, a permute and a `repeat` (burn-kda
//! `lib.rs:412-463`), so all seven parents are nodes that burn may DROP and
//! must RE-EXECUTE through its retro-forward during the backward.
//! `BalancedCheckpointing` is what drops them; `NoCheckpointing` saves
//! everything and the recompute path never runs.
//!
//! So the whole graph runs twice, on the SAME values, with the two strategies,
//! and the gradients must agree. A recompute that returns the wrong value — or
//! a parent whose node the strategy never checkpointed — shows up as a number
//! and not as a silent fallback. This is the test the flat graph could not be.

use burn::backend::{Autodiff, AutodiffBackend, DispatchKindConversion, NdArray};
use burn::tensor::activation::sigmoid;
use burn::tensor::{Device, DispatchTensor, Distribution, Tensor};
use burn_autodiff::checkpoint::strategy::{BalancedCheckpointing, CheckpointStrategy, NoCheckpointing};
use burn_gdn2::chunk_wy_forward_autodiff_s;

const RELATOL: f32 = 1e-3;

/// The trainer's device: autodiff with gradient checkpointing enabled.
fn balanced_device() -> Device {
    Device::ndarray().autodiff().gradient_checkpointing()
}

/// The same graph with checkpointing off, which is what the crate's other
/// autodiff tests run: nothing is dropped, so nothing is recomputed.
fn plain_device() -> Device {
    Device::ndarray().autodiff()
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

/// The seven op inputs as random values on the BARE backend. Drawn once and
/// shared by both arms: two `Tensor::random` calls do not agree, and a
/// comparison across different values measures nothing.
fn raw_inputs(batch: usize, heads: usize, time: usize, k: usize, v: usize) -> [Tensor<4>; 7] {
    let device = Device::ndarray();
    let r = |shape: [usize; 4], mean: f64| {
        Tensor::<4>::random(shape, Distribution::Normal(mean, 0.3), &device)
    };
    [
        r([batch, heads, time, k], 0.0),
        r([batch, heads, time, k], 0.0),
        r([batch, heads, time, v], 0.0),
        // negative, like a log decay the model produces
        r([batch, heads, time, k], -1.0),
        r([batch, heads, time, k], 0.0),
        r([batch, heads, time, v], 0.0),
        r([batch, heads, k, v], 0.0),
    ]
}

/// Lift a bare value into a tracked leaf on `Autodiff<NdArray, S>`, on `device`.
fn lift<S: CheckpointStrategy>(raw: &Tensor<4>, device: &Device) -> Tensor<4>
where
    DispatchTensor: DispatchKindConversion<Autodiff<NdArray, S>> + DispatchKindConversion<NdArray>,
{
    let bare = raw
        .clone()
        .try_into_primitive::<NdArray>()
        .expect("a bare ndarray tensor");
    let node = <Autodiff<NdArray, S> as AutodiffBackend>::from_inner(bare);
    let t = Tensor::from_primitive::<Autodiff<NdArray, S>>(node).require_grad();
    assert_eq!(t.device(), *device, "the lift must land on the caller's device");
    t
}

/// The op's seven inputs, each at least one op away from its leaf, shaped like
/// `KdaModule::project` builds them.
struct Nested {
    leaves: Vec<Tensor<4>>,
    inputs: [Tensor<4>; 7],
}

fn nested<S: CheckpointStrategy>(device: &Device, raw: &[Tensor<4>; 7]) -> Nested
where
    DispatchTensor: DispatchKindConversion<Autodiff<NdArray, S>> + DispatchKindConversion<NdArray>,
{
    let l = |i: usize| lift::<S>(&raw[i], device);
    // The model reaches the op through a projection that lands [B,T,H,D] and
    // is then `.permute([0,2,1,3])`-ed to [B,H,T,D] — a metadata-only view, so
    // the op's input is a STRIDED tensor of unchanged logical shape. That is
    // the layout the CUDA seam has to materialize, and this is its twin.
    let to_4d = |t: Tensor<4>| -> Tensor<4> { t.swap_dims(1, 2).permute([0, 2, 1, 3]) };

    let q_l = l(0);
    let k_l = l(1);
    let v_l = l(2);
    let g_l = l(3);
    let b_l = l(4);
    let w_l = l(5);
    let s_l = l(6);

    let inputs = [
        to_4d(q_l.clone().mul_scalar(0.5)),
        to_4d(sigmoid(k_l.clone()).mul_scalar(0.5)),
        to_4d(v_l.clone().mul_scalar(0.5)),
        // the decay gate through a log, like `alpha.log()`
        to_4d(g_l.clone().mul_scalar(0.5).powf_scalar(2.0).log()),
        to_4d(sigmoid(b_l.clone())),
        to_4d(sigmoid(w_l.clone())),
        s_l.clone().mul_scalar(0.1),
    ];
    Nested {
        leaves: vec![q_l, k_l, v_l, g_l, b_l, w_l, s_l],
        inputs,
    }
}

/// Run the whole nested graph under one strategy: the op in the middle, a
/// chain of ops after it, a real backward. Returns the loss and the gradient
/// of every leaf.
fn run<S: CheckpointStrategy>(
    device: &Device,
    raw: &[Tensor<4>; 7],
    chunk: usize,
) -> (f32, Vec<Tensor<4>>)
where
    DispatchTensor: DispatchKindConversion<Autodiff<NdArray, S>> + DispatchKindConversion<NdArray>,
{
    let g = nested::<S>(device, raw);
    let (out, _state) = chunk_wy_forward_autodiff_s::<NdArray, S>(
        g.inputs[0].clone(),
        g.inputs[1].clone(),
        g.inputs[2].clone(),
        g.inputs[3].clone(),
        g.inputs[4].clone(),
        g.inputs[5].clone(),
        g.inputs[6].clone(),
        1.0,
        chunk,
    )
    .expect("the op must accept a graph on the caller's own checkpointing strategy");

    // Ops AFTER the op, so the backward reaches it through a chain instead of
    // consuming its output directly. Nothing outside the op contributes, so a
    // zero gradient on a leaf means the op's backward did not reach it.
    let loss = out
        .clone()
        .powf_scalar(2.0)
        .sum()
        .add(out.sum().mul_scalar(0.5));
    let value: f32 = loss.clone().into_scalar();
    let grads = loss.backward();
    let leaves = g
        .leaves
        .iter()
        .map(|t| {
            t.grad(&grads)
                .unwrap_or_else(|| panic!("a leaf got no gradient"))
                .clone()
        })
        .collect();
    (value, leaves)
}

/// The op on a nested graph under `BalancedCheckpointing` — the trainer's
/// configuration, where the backward must re-execute the dropped parents —
/// must give the same loss and the same gradients as the same graph under
/// `NoCheckpointing`, which recomputes nothing.
#[test]
fn nested_balanced_graph_matches_no_checkpointing() {
    let (batch, heads, time, k, v, chunk) = (2usize, 2usize, 32usize, 4usize, 3usize, 16usize);
    let raw = raw_inputs(batch, heads, time, k, v);

    let (loss_bal, grads_bal) = {
        let device = balanced_device();
        run::<BalancedCheckpointing>(&device, &raw, chunk)
    };
    let (loss_no, grads_no) = {
        let device = plain_device();
        run::<NoCheckpointing>(&device, &raw, chunk)
    };

    // The forward cannot depend on the strategy, so this also proves the two
    // arms really did run on the same values.
    let d = (loss_bal - loss_no).abs() / loss_no.abs().max(1e-30);
    assert!(
        d < RELATOL,
        "loss differs between strategies: balanced={loss_bal} no_ckpt={loss_no} rel={d:.2e}"
    );
    for (i, (a, b)) in grads_bal.iter().zip(grads_no.iter()).enumerate() {
        let d = rel_diff(a, b);
        assert!(
            d < RELATOL,
            "input {i} gradient differs between strategies: rel={d:.2e} \
             (a recomputed parent, or a parent nobody checkpointed)"
        );
        assert!(
            a.clone().abs().max().into_scalar::<f32>() > 0.0,
            "input {i} got a zero gradient through the op"
        );
    }
}

/// The nested graph must reach the op at all: a strategy-blind op returns
/// `None` here, which is the silent fallback that kept the fused path out of
/// production. Asserted apart from the numbers so a `None` is a clear failure.
#[test]
fn the_op_accepts_a_nested_balanced_graph() {
    let raw = raw_inputs(1, 2, 32, 4, 3);
    let device = balanced_device();
    let g = nested::<BalancedCheckpointing>(&device, &raw);
    let balanced = chunk_wy_forward_autodiff_s::<NdArray, BalancedCheckpointing>(
        g.inputs[0].clone(),
        g.inputs[1].clone(),
        g.inputs[2].clone(),
        g.inputs[3].clone(),
        g.inputs[4].clone(),
        g.inputs[5].clone(),
        g.inputs[6].clone(),
        1.0,
        16,
    );
    assert!(
        balanced.is_some(),
        "the op refused a Balanced graph: it is still strategy-blind"
    );
    // The same values through the default-strategy entry point must stay
    // blind: that blindness is what `chunk_dispatch` probes around, and it
    // must not be "fixed" by making the two indistinguishable.
    let default_entry = burn_gdn2::chunk_wy_forward_autodiff::<NdArray>(
        g.inputs[0].clone(),
        g.inputs[1].clone(),
        g.inputs[2].clone(),
        g.inputs[3].clone(),
        g.inputs[4].clone(),
        g.inputs[5].clone(),
        g.inputs[6].clone(),
        1.0,
        16,
    );
    assert!(
        default_entry.is_none(),
        "the NoCheckpointing entry point accepted a Balanced graph: \
         the strategy is no longer part of the conversion"
    );
}
