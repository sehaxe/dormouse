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
        // The six token-side leaves are laid out [B,T,H,D], the layout a
        // projection leaves them in; `project_to_4d` transposes to [B,H,T,D].
        r([batch, time, heads, k], 0.0),
        r([batch, time, heads, k], 0.0),
        r([batch, time, heads, v], 0.0),
        // negative, like a log decay the model produces
        r([batch, time, heads, k], -1.0),
        r([batch, time, heads, k], 0.0),
        r([batch, time, heads, v], 0.0),
        // the state is [B,H,K,V] and contiguous, exactly like the zeros the
        // trainer starts each loop with
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

/// The trainer's own input construction, copied: `KdaModule::project`'s
/// `to_4d` (burn-kda `src/lib.rs:407-410`) is a `reshape([B,T,H,D])` of a
/// CONTIGUOUS projection followed by `.permute([0,2,1,3])` — a metadata-only
/// transposition. The leaf here is already split into heads, so the reshape is
/// the identity view and only the permute acts; the strides that reach the op
/// are the same either way.
///
/// The detail that matters, and that a fixture can get wrong: it must be a
/// reshape-then-permute, NOT a `swap_dims(1,2)` followed by its own inverse
/// `permute([0,2,1,3])` — that round trip returns the strides to where they
/// started and yields a CONTIGUOUS tensor, so a fixture built that way
/// exercises no strided view at all. [`the_trainers_view_is_a_real_transposition`]
/// is that mistake, turned into a runnable check.
fn project_to_4d(t: Tensor<4>, heads: usize, d: usize) -> Tensor<4> {
    let [b, tokens, h, dd] = t.shape().dims::<4>();
    assert_eq!((h, dd), (heads, d), "the leaf must be [B,T,H,D]");
    let _ = (b, tokens);
    t.permute([0, 2, 1, 3])
}

fn nested<S: CheckpointStrategy>(device: &Device, raw: &[Tensor<4>; 7]) -> Nested
where
    DispatchTensor: DispatchKindConversion<Autodiff<NdArray, S>> + DispatchKindConversion<NdArray>,
{
    let l = |i: usize| lift::<S>(&raw[i], device);
    // heads and dims come from the leaves: a [B,T,H,D] leaf, so its last two
    // axes ARE the head count and the head width.
    let [b, time, heads, k] = raw[0].shape().dims::<4>();
    let v = raw[2].shape().dims::<4>()[3];
    let _ = (b, time);

    let q_l = l(0);
    let k_l = l(1);
    let v_l = l(2);
    let g_l = l(3);
    let b_l = l(4);
    let w_l = l(5);
    let s_l = l(6);

    let inputs = [
        project_to_4d(q_l.clone().mul_scalar(0.5), heads, k),
        project_to_4d(sigmoid(k_l.clone()).mul_scalar(0.5), heads, k),
        project_to_4d(v_l.clone().mul_scalar(0.5), heads, v),
        // the decay gate through a log, like `alpha.log()`
        project_to_4d(
            g_l.clone().mul_scalar(0.5).powf_scalar(2.0).log(),
            heads,
            k,
        ),
        project_to_4d(sigmoid(b_l.clone()), heads, k),
        project_to_4d(sigmoid(w_l.clone()), heads, v),
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
    // The op DECLINES here, and that is the fixed behaviour, not a regression:
    // the seven inputs are `permute(act(leaf))` - all `GradInBackward`
    // intermediates - so the custom node this function would build comes back
    // `UnTracked`, its output is a LEAF, and no gradient could reach the arm.
    // Declining routes the call to the ops path on the same tensors, which is
    // what burn's own graph then differentiates - so this file still measures
    // the thing it was written for (the same graph under two checkpointing
    // strategies), through the arm that actually runs.
    let (out, _state) = match chunk_wy_forward_autodiff_s::<NdArray, S>(
        g.inputs[0].clone(),
        g.inputs[1].clone(),
        g.inputs[2].clone(),
        g.inputs[3].clone(),
        g.inputs[4].clone(),
        g.inputs[5].clone(),
        g.inputs[6].clone(),
        1.0,
        chunk,
    ) {
        Some(r) => r,
        None => burn_gdn2::chunk_wy_forward(
            g.inputs[0].clone(),
            g.inputs[1].clone(),
            g.inputs[2].clone(),
            g.inputs[3].clone(),
            g.inputs[4].clone(),
            g.inputs[5].clone(),
            g.inputs[6].clone(),
            1.0,
            chunk,
        ),
    };

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

/// The fixture must really be a TRANSPOSITION, because a fixture that is not
/// is a test that does not test — the exact failure that let the fused seam's
/// infinite recursion ship (`Tensor::random` is contiguous, so no test ever
/// reached the non-contiguous branch), and the one that made
/// `tests/fused_permuted_view.rs` report `0 copies` on its first GPU run.
///
/// Proved with VALUES, not with an assumption about strides, so it holds on any
/// backend: a `[B,T,H,D]` tensor whose `permute([0,2,1,3])` reads the same
/// elementwise is a no-op permutation, i.e. contiguous. The trainer's view
/// does not, and the inverse-pair mistake does.
#[test]
fn the_trainers_view_is_a_real_transposition() {
    let device = Device::ndarray();
    let (b, t, h, d) = (2usize, 6usize, 3usize, 4usize);
    let src = Tensor::<4>::random([b, t, h, d], Distribution::Uniform(0.0, 1.0), &device);

    // The trainer's construction: a real transposition, so the [B,H,T,D] view
    // holds the data in a different order than a row-major buffer of that
    // shape — i.e. it is strided and must be materialized.
    let trainer_view = project_to_4d(src.clone(), h, d);
    assert_eq!(trainer_view.shape().dims::<4>(), [b, h, t, d]);
    assert_ne!(
        trainer_view.clone().into_data().bytes,
        src.clone().into_data().bytes,
        "the trainer's [B,T,H,D]->[B,H,T,D] permute was a no-op: the fixture \
         would be contiguous and no strided path would be exercised"
    );

    // The mistake: swap_dims(1,2) followed by its own inverse permute returns
    // the strides to where they started, so the data reads back identical.
    let round_trip = src.clone().swap_dims(1, 2).permute([0, 2, 1, 3]);
    assert_eq!(round_trip.shape().dims::<4>(), [b, t, h, d]);
    assert_eq!(
        round_trip.into_data().bytes,
        src.into_data().bytes,
        "swap_dims followed by its inverse permute stopped being the identity; \
         if this ever fails the strided-fixture argument above needs rechecking"
    );
}

/// The op on a nested graph under `BalancedCheckpointing` — the trainer's
/// configuration, where the backward must re-execute the dropped parents —
/// must give the same loss and the same gradients as the same graph under
/// `NoCheckpointing`, which recomputes nothing.
#[test]
fn nested_balanced_graph_matches_no_checkpointing() {    let (batch, heads, time, k, v, chunk) = (2usize, 2usize, 32usize, 4usize, 3usize, 16usize);
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

/// The nested graph must reach the op, and the op's output must be a TRACKED
/// node — a node that can send a gradient back — rather than a leaf.
///
/// Asserted apart from the numbers, so a silent change of arm is a clear
/// failure rather than a slightly different set of gradients.
///
/// # This assertion was INVERTED once, and the reason it is written this way
///
/// `8fa5d4c` diagnosed the right disease and prescribed the wrong cure. The
/// disease was real: under `BalancedCheckpointing` the op's inputs are
/// checkpoint leaves, so `OpsPrep::prepare` returned `UnTracked`, the custom
/// node's output was a LEAF, and no gradient could reach the arm. The cure was
/// to DECLINE (`return None`) whenever no input "requires grad", so
/// `chunk_dispatch` fell through to the ops path. That works — and it is
/// precisely why this test was written to assert the decline.
///
/// It stopped being the right cure at `2a430cc`, which changed the gate from
/// `Tensor::is_require_grad()` (literally `requirement == Grad`, the strict
/// requirement of a LEAF) to `AutodiffTensor::is_tracked()` (literally
/// `!requirement.is_none()`, which is what `OpsPrep::prepare` reads). Every one
/// of the trainer's seven inputs is a projection output with requirement
/// `GradInBackward`, so the old gate declined on the trainer's own graph while
/// every fixture built from lifted `require_grad()` leaves passed — the exact
/// shape of the defect `8fa5d4c` was written about, re-created one commit later.
///
/// So "the op declined" stopped being the intended behaviour, and this assertion
/// went red against a DELIBERATE change rather than against a regression. The
/// replacement therefore does not invert it: it asserts the PROPERTY the decline
/// was standing in for, which is the thing that was actually broken —
///
///   * the op builds a node on a projection graph (it does not decline), and
///   * that node is TRACKED, i.e. its output carries a gradient, and
///   * the loss reaches every leaf through it, with the ops path's numbers.
///
/// A fixture that stops being non-leaf would stop testing the thing, so the
/// fixture is checked for that too — the mistake `8fa5d4c` made was in the
/// GATE, and the mistake this test could make is in the FIXTURE.
#[test]
fn the_op_builds_a_tracked_node_on_a_projection_graph_and_the_gradient_reaches_every_leaf() {
    let raw = raw_inputs(1, 2, 32, 4, 3);
    let device = balanced_device();
    let g = nested::<BalancedCheckpointing>(&device, &raw);
    for (i, t) in g.inputs.iter().enumerate() {
        assert!(
            !t.is_require_grad(),
            "input {i} is a require_grad leaf: this fixture is supposed to be all \
             intermediates, and a lifted leaf would let a gate that asks the wrong \
             question pass"
        );
    }
    let args = [
        g.inputs[0].clone(),
        g.inputs[1].clone(),
        g.inputs[2].clone(),
        g.inputs[3].clone(),
        g.inputs[4].clone(),
        g.inputs[5].clone(),
        g.inputs[6].clone(),
    ];
    let r = chunk_wy_forward_autodiff_s::<NdArray, BalancedCheckpointing>(
        args[0].clone(),
        args[1].clone(),
        args[2].clone(),
        args[3].clone(),
        args[4].clone(),
        args[5].clone(),
        args[6].clone(),
        1.0,
        16,
    );
    let (out, _state) = r.expect(
        "the op declined on the trainer's own graph shape (leaf -> mul_scalar -> permute). \
         That is `8fa5d4c`'s behaviour, not `2a430cc`'s: the gate must be \
         `AutodiffTensor::is_tracked()` = `!requirement.is_none()`, because every one of \
         these inputs is a projection output whose requirement is `GradInBackward`. If this \
         fires, the gate is asking `is_require_grad()` again and the attention arm is \
         training nothing.",
    );
    // Tracked-ness is `!requirement.is_none()` on the primitive, which is
    // literally what `OpsPrep::prepare` reads.  A second, INDEPENDENT witness
    // that the output is a node and not a leaf: a leaf's `requirement` is the
    // strict `Grad`, and a tracked node's output is not — so `is_require_grad`
    // being FALSE here is the observable signature of "this is a node, and burn
    // will run its backward".  The first is the mechanism, the second is what a
    // caller can see, and a defect that only moves one of them is still a
    // defect.
    let node = out
        .clone()
        .try_into_primitive::<Autodiff<NdArray, BalancedCheckpointing>>()
        .expect("the op's output is not an autodiff tensor at all");
    assert!(
        node.is_tracked(),
        "the op built a node but its output is not TRACKED: the backward cannot reach the \
         arm through it. This is the `8fa5d4c` defect itself (OpsPrep returned UnTracked), \
         and a leaf output is silent — the forward runs, its counter climbs, and no \
         gradient is produced."
    );
    assert!(
        !out.is_require_grad(),
        "the op's output reports the strict LEAF requirement. A node's output carries \
         `GradInBackward`, not `Grad`; if this is true the output is a leaf and the whole \
         assertion above is vacuous."
    );

    // NOT probed through `chunk_wy_forward_autodiff` (the NoCheckpointing entry
    // point) here, and the reason is worth recording because the previous
    // version of this test got it backwards.  `Autodiff<Inner, S>`'s float
    // primitive is `AutodiffTensor<Inner>` and `S` does not appear in the type,
    // so the two entry points are not interchangeable on the same tensors: the
    // strategy lives in burn's runtime dispatch context, and asking for
    // `NoCheckpointing` on a tensor whose context is `BalancedCheckpointing`
    // makes every conversion return `None`.  That is a TYPE-level fact, not the
    // tracked-ness gate, and asserting on it measured nothing about the
    // backward.  What the strategy-blindness actually needs is covered where it
    // is meaningful: `nested_balanced_graph_matches_no_checkpointing` runs the
    // same graph under both strategies and requires the gradients to agree.

    // And the gradient is real and it is the SAME gradient the ops path gives.
    // This is the assertion that would have failed on a frozen arm, and
    // comparing the two is what makes "it is not zero" more than a smoke test.
    //
    // TWO graphs, not one graph walked twice: burn's autodiff tape is consumed
    // by the first `backward()` and `retain_graph` does not exist, so a second
    // call on the same nodes panics with "graph tape has already been
    // consumed".  `nested` rebuilds every tensor, so the two graphs are
    // independent and the comparison is still value-for-value.
    let g_ops = nested::<BalancedCheckpointing>(&device, &raw);
    let (ops_out, _) = burn_gdn2::chunk_wy_forward(
        g_ops.inputs[0].clone(),
        g_ops.inputs[1].clone(),
        g_ops.inputs[2].clone(),
        g_ops.inputs[3].clone(),
        g_ops.inputs[4].clone(),
        g_ops.inputs[5].clone(),
        g_ops.inputs[6].clone(),
        1.0,
        16,
    );
    let ops_grads = ops_out
        .clone()
        .powf_scalar(2.0)
        .sum()
        .add(ops_out.sum().mul_scalar(0.5))
        .backward();
    let grads = out.clone().powf_scalar(2.0).sum().add(out.sum().mul_scalar(0.5)).backward();
    let mut worst = 0.0f32;
    let mut mags = Vec::new();
    for (i, t) in g.leaves.iter().enumerate() {
        let d = t
            .grad(&grads)
            .unwrap_or_else(|| panic!("leaf {i} got no gradient at all: the arm is frozen"))
            .clone();
        let m = d.clone().abs().max().into_scalar::<f32>();
        assert!(m > 0.0, "leaf {i} got a ZERO gradient through the op");
        mags.push(m);
        let r_ops = g_ops.leaves[i]
            .grad(&ops_grads)
            .expect("the ops path left a leaf without a gradient")
            .clone();
        worst = worst.max(rel_diff(&d, &r_ops));
    }
    println!(
        "BalancedCheckpointing, all-intermediate inputs: the op built a TRACKED node; \
         leaf gradient |max| = {mags:?}; worst disagreement with the ops path {worst:.2e} \
         (bar {RELATOL:.0e})"
    );
    assert!(
        worst < RELATOL,
        "the node's gradient differs from the ops path's by {worst:.2e}: the tracked node is \
         carrying a gradient, but not the gradient of this function"
    );
}
