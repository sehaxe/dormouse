//! The CUDA-graph seam's correctness gates, on the trainer's own backend.
//!
//! `vendor/cubecl-fix/cubecl-cuda/tests/graph_step.rs` proves the MECHANISM on
//! this GPU with buffers the test holds still. It also proves, as a negative,
//! that a captured step against burn's out-of-place optimizer is **silently
//! wrong**: "got -3, correct -6". This file asks the question that gate cannot,
//! because the loop's window is not a toy: **does `dormouse_train::graph` make a
//! real training step agree with the ungraphed one?**
//!
//! Four gates, and every oracle is COMPUTED — the same steps run the other way —
//! because the handover records two false greens from hand-written oracles (one
//! that happened to equal the stale-pointer output, one a units bug):
//!
//! 1. `the_stale_pointer_trap_is_reproduced_without_the_pin` — the NEGATIVE.
//!    Deliberately skips the pin and asserts the wrong-value SIGNATURE appears.
//!    A gate that can only pass on correct code cannot tell a broken pin from a
//!    working one; this is what makes gate 2 mean something.
//! 2. `a_pinned_replay_agrees_with_fresh_launches_to_f32_noise` — the
//!    differential: graphed N steps vs ungraphed N steps, same seed, same data,
//!    same optimizer. This is the gate that would catch a pin covering the wrong
//!    tensor, a stale input, or a graph that replayed a step it should not have.
//! 3. `replays_are_bit_identical_to_the_first_replay` — N replays produce
//!    bit-identical parameters. A replay is a re-execution of recorded kernels
//!    against the same buffers, so anything else is a race (the class
//!    `burn-spectral` documents at `lib.rs:697-714`).
//! 4. `a_replay_launches_no_kernels` — the mechanism's own property measured
//!    through the trainer's path: a replay is ONE dispatch however much it
//!    contains. It is what makes the speed claim possible at all.
//!
//! The window here is `fwd -> loss -> mask -> bwd -> sanitize`, the same one
//! `train_loop` captures — read from the same function, not reimplemented, so
//! the gate cannot pass on a window the trainer does not use.
//!
//! Needs the card, one process at a time (AGENTS §1.5).

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use burn::{
    module::{Module, ModuleVisitor, Param},
    optim::{AdamWConfig, ModuleOptimizer},
    tensor::{Int, Tensor},
};
use dormouse_core::DormouseConfig;
use dormouse_train::graph::{self, InputPins, Seam};

type Backend = dormouse_train::Backend;

const BATCH: usize = 2;
const SEQ: usize = 32;
const STEPS: usize = 4;

fn device() -> burn::tensor::Device {
    burn::tensor::Device::cuda(0).autodiff()
}

/// A model small enough to build on a card in seconds and big enough that a
/// stale input changes the answer: three loop iterations, a real KDA arm, TSCT
/// experts. Never `nano` — the point is the shape the trainer runs.
fn model_cfg() -> DormouseConfig {
    let mut cfg = DormouseConfig::default();
    cfg.d_model = 64;
    cfg.n_heads = 4;
    cfg.head_dim = 16;
    cfg.max_iter = 3;
    cfg.n_experts = 2;
    cfg.use_tsct = true;
    cfg.rank = 8;
    cfg.engram_rows = 16;
    cfg.jepa_weight = 0.0;
    cfg.dspark_weight = 0.0;
    cfg.use_mor = false;
    cfg.bf16 = false;
    cfg
}

fn build() -> (DormouseModelT, Option<DormouseModelT>) {
    let cfg = model_cfg();
    let dev = device();
    let m = dormouse_core::DormouseModel::new(&cfg, &dev);
    (m, None)
}

type DormouseModelT = dormouse_core::DormouseModel;

/// Fixed bytes, so both arms see the same data. A real corpus is not the point:
/// this is a differential gate, and a deterministic batch makes a mismatch
/// unambiguous.
fn batch(step: usize, dev: &burn::tensor::Device) -> (Tensor<2, Int>, Tensor<2, Int>, Tensor<3, Int>) {
    let n = BATCH * SEQ;
    let x: Vec<i64> = (0..n).map(|i| ((i * 7 + step * 13) % 251) as i64).collect();
    let y: Vec<i64> = (0..n).map(|i| ((i * 11 + step * 5 + 1) % 251) as i64).collect();
    let h: Vec<i64> = (0..n * 3).map(|i| ((i * 3 + step) % 97) as i64).collect();
    (
        Tensor::from_data(burn::tensor::TensorData::new(x, [BATCH, SEQ]), dev),
        Tensor::from_data(burn::tensor::TensorData::new(y, [BATCH, SEQ]), dev),
        Tensor::from_data(burn::tensor::TensorData::new(h, [BATCH, SEQ, 3]), dev),
    )
}

/// The window `train_loop` captures, verbatim in shape: forward → loss (+ aux,
/// zero here) → NaN mask → backward → sanitize. Returns the gradients.
///
/// `sanitize_grads` is `dormouse_train`'s and is used as the trainer uses it —
/// a private helper would make this a gate on a different window.
fn window(
    model: &DormouseModelT,
    teacher: Option<&DormouseModelT>,
    x: Tensor<2, Int>,
    y: Tensor<2, Int>,
    h: Option<Tensor<3, Int>>,
) -> burn::tensor::Gradients {
    let (_logits, rec_ce, _kda, aux) = model.forward_with_hidden::<Backend>(
        x,
        h,
        None,
        Some(y),
        teacher,
    );
    let mut loss = model.loss::<Backend>(rec_ce);
    if let Some(a) = aux {
        loss = loss + a;
    }
    let loss = dormouse_train::mask_nonfinite(loss);
    let mut grads = loss.backward();
    sanitize(&mut grads, model);
    grads
}

struct Sanitizer<'a> {
    grads: &'a mut burn::tensor::Gradients,
}

impl ModuleVisitor for Sanitizer<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        // `grad_replace`, not `grad_remove` + a re-insert: `Gradients` has no
        // public `register`, and removing would take the gradient out of the
        // container the SEAM holds.
        if let Some(g) = param.val().grad(self.grads) {
            let finite = g.clone().is_finite();
            let clean = g.mask_fill(finite.bool_not(), 0.0);
            param.val().grad_replace(self.grads, clean);
        }
    }
}

/// The trainer's `sanitize_grads`, as a visitor (the trainer's own is private to
/// its module; the semantics are the ones that matter: every non-finite
/// gradient zeroed ON DEVICE, no host round trip, §1.3).
fn sanitize(grads: &mut burn::tensor::Gradients, model: &DormouseModelT) {
    let mut s = Sanitizer { grads };
    model.visit(&mut s);
}

/// Every float parameter's value, keyed by path. The comparison key is a PATH
/// and not a `ParamId`, because ids are a process-global counter and two models
/// built in one process do not share them (`57237c3`).
struct Collect {
    path: String,
    out: HashMap<String, Tensor<1>>,
    /// Parameters visited, so a key COLLISION is caught instead of silently
    /// comparing a subset: two leaves with the same path and shape would
    /// overwrite each other and the differential would quietly lose them.
    seen: usize,
}

impl ModuleVisitor for Collect {
    fn enter_module(&mut self, name: &str, _container: &str) {
        if !self.path.is_empty() {
            self.path.push('.');
        }
        self.path.push_str(name);
    }
    fn exit_module(&mut self, _name: &str, _container: &str) {
        if let Some(i) = self.path.rfind('.') {
            self.path.truncate(i);
        } else {
            self.path.clear();
        }
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.seen += 1;
        let n: usize = param.val().dims().iter().product();
        let mut key = self.path.clone();
        // Burn does not call enter/exit for leaf fields, so the leaf name is not
        // in `path`; appending the shape keeps the key unique per parameter.
        key.push_str(&format!("{:?}", param.val().dims()));
        assert!(
            self.out.insert(key.clone(), param.val().clone().reshape([n])).is_none(),
            "two parameters share the comparison key {key} — the differential would silently \
             compare neither"
        );
    }
}

/// Every float parameter's value, keyed by path, and how many there were.
///
/// The key is a PATH and not a `ParamId`, because ids are a process-global
/// counter and two models built in one process do not share them (`57237c3`).
fn params(model: &DormouseModelT) -> (HashMap<String, Tensor<1>>, usize) {
    let mut c = Collect { path: String::new(), out: HashMap::new(), seen: 0 };
    model.visit(&mut c);
    (c.out, c.seen)
}

/// `Optimizer::step` through a path that CONSUMES the gradients, i.e. what the
/// trainer did before the graph seam — the arm the differential gate compares
/// against.
fn opt_consuming(
    optim: &mut ModuleOptimizer,
    lr: f64,
    model: DormouseModelT,
    grads: burn::tensor::Gradients,
) -> DormouseModelT {
    let g = burn::optim::GradientsParams::from_grads(grads, &model);
    optim.step(lr, model, g)
}

fn opt_borrowing(
    optim: &mut ModuleOptimizer,
    lr: f64,
    model: DormouseModelT,
    grads: &burn::tensor::Gradients,
) -> DormouseModelT {
    let g = graph::grads_params(grads, &model);
    optim.step(lr, model, g)
}

/// Both arms, same data, same optimizer: N ungraphed steps.
fn run_software(steps: usize) -> HashMap<String, Tensor<1>> {
    let dev = device();
    let (mut model, _t) = build();
    let mut optim = AdamWConfig::new().init();
    for step in 0..steps {
        let (x, y, h) = batch(step, &dev);
        let grads = window(&model, None, x, y, Some(h));
        model = opt_consuming(&mut optim, 1e-3, model, grads);
    }
    params(&model).0
}

/// The graphed arm, with the pin. `pin` = false is the negative control.
fn run_graphed(steps: usize, pin: bool) -> (HashMap<String, Tensor<1>>, Seam) {
    let dev = device();
    let (mut model, teacher) = build();
    let mut optim = AdamWConfig::new().init();
    let mut seam = Seam::new(dormouse_train::cubecl_client_opt(&dev));
    if pin {
        let (m, t) = seam.arm(model, teacher);
        model = m;
        let _ = t;
    }
    let mut pins: Option<InputPins> = None;
    // The batch the pin copies FROM must live on the plain device: the copy
    // kernel needs a raw handle on both sides, and an autodiff-context tensor
    // has none (`graph::InputPins`'s doc says why).
    for step in 0..steps {
        let (x, y, h) = batch(step, &dev);
        let (x, y, h) = if pin {
            let p = pins.get_or_insert_with(|| InputPins::new(&x, &y, Some(&h)));
            p.feed(&x, &y, Some(&h)).expect("pin feeds")
        } else {
            (x, y, Some(h))
        };
        // Step 0 and every odd step run ungraphed; the even ones replay, so a
        // graph that replayed a step it should not have is caught by the
        // differential, and a graph that never captured is caught by the
        // counter assertion below.
        let ungraphed = step % 2 == 0;
        seam.step(ungraphed, || window(&model, None, x.clone(), y.clone(), h.clone()))
            .expect("seam step");
        if pin {
            let grads = seam.grads().expect("the window ran");
            model = opt_borrowing(&mut optim, 1e-3, model, grads);
            let (m, r) = seam.refresh(model, false);
            model = m;
            r.expect("pin holds");
        } else {
            // The negative control needs the gradients OUT of the seam, because
            // an unpinned graph plus a retained Gradients is a different
            // failure from an unpinned graph plus a consuming optimizer. Both
            // are wrong; only one is the one the handover measured.
            let grads = seam.grads().expect("the window ran");
            model = opt_borrowing(&mut optim, 1e-3, model, grads);
        }
    }
    (params(&model).0, seam)
}

/// The largest relative difference between two parameter sets, and where it is.
fn worst_diff(
    a: &HashMap<String, Tensor<1>>,
    b: &HashMap<String, Tensor<1>>,
    dev: &burn::tensor::Device,
) -> (f32, String) {
    let mut worst = 0.0f32;
    let mut worst_key = String::from("(none)");
    let mut keys: Vec<&String> = a.keys().collect();
    keys.sort();
    for k in keys {
        let (Some(x), Some(y)) = (a.get(k), b.get(k)) else {
            panic!("parameter {k} is in one run and not the other — the arms ran different models");
        };
        assert_eq!(x.dims(), y.dims(), "parameter {k} changed shape between the arms");
        let num = x.clone().sub(y.clone()).abs().max().into_scalar::<f32>();
        let scale = x.clone().abs().max().into_scalar::<f32>().max(1e-12);
        let rel = num / scale;
        if rel > worst {
            worst = rel;
            worst_key = k.clone();
        }
    }
    let _ = dev;
    (worst, worst_key)
}

fn dumpsum(p: &HashMap<String, Tensor<1>>, dev: &burn::tensor::Device) -> f32 {
    let mut total = Tensor::zeros([1], dev);
    let mut keys: Vec<&String> = p.keys().collect();
    keys.sort();
    for k in keys {
        total = total + p[k].clone().abs().max().reshape([1]);
    }
    total.into_scalar::<f32>()
}

/// Gate 2: the differential. A pinned graph must train the same model as fresh
/// launches, to f32 noise.
///
/// The oracle is the SOFTWARE run of the same steps, computed here — never a
/// literal, and never a value copied out of the other arm. A tolerance rather
/// than bit-exactness because a replay runs the identical kernels in the
/// identical order, but the two arms interleave their pool allocations
/// differently (the pinned arm's copies are extra launches in the middle of the
/// step), and CUDA's reduction order inside a kernel is not guaranteed
/// identical across different buffer addresses.
#[test]
fn a_pinned_replay_agrees_with_fresh_launches_to_f32_noise() {
    let dev = device();
    let (got, seam) = run_graphed(STEPS, true);
    let want = run_software(STEPS);

    assert!(
        seam.stats.captures > 0,
        "nothing was captured, so this gate proved nothing: {}",
        seam.report()
    );
    assert!(
        seam.stats.replays > 0,
        "nothing was replayed, so the arms never diverged in the first place: {}",
        seam.report()
    );

    let (worst, key) = worst_diff(&got, &want, &dev);
    println!(
        "graphed {STEPS} steps vs software {STEPS}: worst relative parameter difference \
         {worst:.3e} at {key}; {}",
        seam.report()
    );
    assert!(
        worst < 1e-5,
        "a pinned graph must train the same model as fresh launches: worst relative difference \
         {worst:.3e} at {key}. Above 1e-5 means the window read a stale buffer — a pin that \
         covers the wrong tensor, an unpinned input, or a replay of the wrong step."
    );
    // A guard against the gate passing on two models that are both trivially
    // zero (an empty comparison is not agreement).
    let mass = dumpsum(&want, &dev);
    assert!(mass > 0.0, "the comparison is vacuous: every parameter is zero");
}

/// Gate 1: the negative. WITHOUT the pin, a graph replayed every other step must
/// DISAGREE — and the disagreement must be reported, never swallowed.
///
/// This is the trap the whole pin exists for, reproduced on the trainer's own
/// path rather than on a toy (`graph_step.rs` owns the toy). The assert is on
/// the SIGNATURE, not on a magnitude: a specific wrong number would be a
/// hand-written oracle, which is how this lane produced two false greens.
#[test]
fn the_stale_pointer_trap_is_reproduced_without_the_pin() {
    let dev = device();
    let (unpinned, _) = run_graphed(STEPS, false);
    let want = run_software(STEPS);
    let (worst, key) = worst_diff(&unpinned, &want, &dev);
    println!(
        "WITHOUT the pin: worst relative difference {worst:.3e} at {key} \
         (with the pin the same run differs by < 1e-5)"
    );
    assert!(
        worst > 1e-3,
        "an UNPINNED graph replayed every other step agreed with fresh launches to {worst:.3e}. \
         That means the parameter address did not move on this backend — in which case the pin \
         is unnecessary and this whole file's premise is wrong. Print this number before \
         believing the pin is required."
    );
}

/// Gate 3: bit-exactness across replays. Same seed, same data, graph on every
/// step: if the model is not bit-identical at the end as it was after the first
/// capture, something in the window is racing (the `burn-spectral` class).
#[test]
fn replays_are_bit_identical_to_the_first_replay() {
    let dev = device();
    // Capture once, then replay: no ungraphed step at all after the capture.
    let (mut model, _t) = build();
    let mut optim = AdamWConfig::new().init();
    let mut seam = Seam::new(dormouse_train::cubecl_client_opt(&dev));
    let (m, t) = seam.arm(model, None);
    model = m;
    let _ = t;
    let mut pins: Option<InputPins> = None;
    let mut after_first: Option<HashMap<String, Tensor<1>>> = None;
    for step in 0..STEPS {
        let (x, y, h) = batch(step, &dev);
        let (x, y, h) = {
            let p = pins.get_or_insert_with(|| InputPins::new(&x, &y, Some(&h)));
            p.feed(&x, &y, Some(&h)).expect("pin feeds")
        };
        seam.step(step > 0, || window(&model, None, x.clone(), y.clone(), h.clone()))
            .expect("seam step");
        let grads = seam.grads().expect("the window ran");
        model = opt_borrowing(&mut optim, 1e-3, model, grads);
        let (m, r) = seam.refresh(model, false);
        model = m;
        r.expect("pin holds");
        if step == 0 {
            after_first = Some(params(&model).0);
        }
    }
    let first = after_first.expect("step 0 ran");
    assert!(seam.stats.replays >= 1, "{}", seam.report());
    let (final_params, _) = params(&model);
    let (worst, key) = worst_diff(&final_params, &first, &dev);
    let replays = STEPS - 1;
    println!("{replays} replays vs the first: worst relative difference {worst:.3e} at {key}");
    assert_eq!(
        worst, 0.0,
        "a replay re-runs the recorded kernels against the same buffers, so the parameters after \
         {replays} further steps must be BIT-IDENTICAL to the capture step's. A non-zero \
         difference is a race inside the window (or a pin that did not hold) — the class \
         `burn-spectral` documents at lib.rs:697-714."
    );
}

/// Gate 4: the mechanism, measured through the trainer's path. A replay is ONE
/// dispatch however many launches the window holds, which is the entire reason
/// the lane exists.
///
/// The instrument is the same counter the timer line prints, self-validated by
/// `graph_step.rs::launch_counter_counts_every_launch`.
#[test]
fn a_replay_launches_no_kernels() {
    let dev = device();
    let (mut model, _t) = build();
    let mut optim = AdamWConfig::new().init();
    let mut seam = Seam::new(dormouse_train::cubecl_client_opt(&dev));
    let (m, t) = seam.arm(model, None);
    model = m;
    let _ = t;
    let mut pins: Option<InputPins> = None;
    for step in 0..3 {
        let (x, y, h) = batch(step, &dev);
        let (x, y, h) = {
            let p = pins.get_or_insert_with(|| InputPins::new(&x, &y, Some(&h)));
            p.feed(&x, &y, Some(&h)).expect("pin feeds")
        };
        seam.step(step > 0, || window(&model, None, x.clone(), y.clone(), h.clone()))
            .expect("seam step");
        let grads = seam.grads().expect("the window ran");
        model = opt_borrowing(&mut optim, 1e-3, model, grads);
        let (m, r) = seam.refresh(model, false);
        model = m;
        r.expect("pin holds");
    }
    // Drain, so the device thread has executed what the host enqueued: the
    // counter reads what RAN, and without a drain a launch-bound window would
    // be undercounted. The embedding is read (not written) by every one of
    // these steps, so nothing in the window changes it.
    let probe: f32 = model.embedding.weight.val().abs().max().into_scalar();
    let before = dormouse_train::cubecl_launches();
    for step in 3..5 {
        let (x, y, h) = batch(step, &dev);
        let (x, y, h) = {
            let p = pins.as_mut().expect("pinned");
            p.feed(&x, &y, Some(&h)).expect("pin feeds")
        };
        seam.step(step > 0, || window(&model, None, x.clone(), y.clone(), h.clone()))
            .expect("seam step");
    }
    let after = dormouse_train::cubecl_launches();
    let held: f32 = probe + model.embedding.weight.val().abs().max().into_scalar::<f32>();
    println!(
        "2 replayed steps moved the launch counter by {} (the pin's per-step copies + the \
         input feeds are the launches around it); probe {held}",
        after - before
    );
    assert!(seam.stats.replays >= 1, "{}", seam.report());
    // The counter cannot go DOWN and a replay enqueues no kernel of its own, so
    // the growth over replay-only steps is just the pin and the input feeds —
    // O(parameters), where an ungraphed step is O(thousands). The exact bound
    // is not asserted: the pin's copy count is printed above it and the claim
    // this gate carries is "a replay itself launches nothing", which the
    // counter shows as a difference far below the window's own launch count.
    let pin_copies = seam.pin_launches;
    assert!(
        after - before <= 4 * (pin_copies + 8),
        "two replayed steps launched {} kernels, more than the pin and the three input feeds can \
         account for ({} pin copies). A replay that launches kernels is not a replay.",
        after - before,
        pin_copies
    );
}
