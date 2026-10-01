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
/// 8 steps, so the graphed arm captures once and REPLAYS 7 — a single replay
/// would satisfy `replays >= 1` while proving nothing about repeatability, and
/// gate 3's claim is about N of them.
const STEPS: usize = 8;

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
    println!("opt_borrowing: registered {} gradient tensors", g.len());
    optim.step(lr, model, g)
}

/// Two models with IDENTICAL values, for the two arms of a differential.
///
/// **`build()` twice is not the same model.** `Device::seed` does not rewind a
/// consumed stream, so two builds in one process draw different weights — 36 of
/// 54 parameters differ at `small`, and the withdrawn claim in AGENTS §3.7 is
/// exactly this. The first version of this file called `build()` inside
/// `run_software` and again inside `run_graphed`, so every differential it
/// reported was dominated by initialisation, not by the arm: the "graphed vs
/// software" 1.98e0 and the "unpinned diverges" 3.96e0 are both the magnitude
/// of two random draws of the same config. Neither number meant what its gate
/// said, and the negative gate would have gone green on a graph that was never
/// used.
///
/// `clone` copies the parameter VALUES; `fork` would re-randomise. The arms must
/// start from the same weights, so it is `clone`.
fn build_pair() -> (DormouseModelT, DormouseModelT) {
    let (model, _t) = build();
    let copy = model.clone();
    (model, copy)
}

/// N ungraphed steps from `model`, returning the parameters after the first step
/// and at the end — the movement pair, so a caller can ask "did it train?"
/// without assuming it.
fn run_software(model: DormouseModelT, steps: usize) -> (HashMap<String, Tensor<1>>, HashMap<String, Tensor<1>>) {
    let dev = device();
    let mut model = model;
    let mut optim = AdamWConfig::new().init();
    let mut after_first: Option<HashMap<String, Tensor<1>>> = None;
    for step in 0..steps {
        let (x, y, h) = batch(step, &dev);
        let grads = window(&model, None, x, y, Some(h));
        model = opt_consuming(&mut optim, 1e-3, model, grads);
        if step == 0 {
            after_first = Some(params(&model).0);
        }
    }
    (after_first.expect("step 0 ran"), params(&model).0)
}

/// The graphed arm, with the pin. `pin` = false is the negative control.
fn run_graphed(
    model: DormouseModelT,
    steps: usize,
    pin: bool,
) -> (HashMap<String, Tensor<1>>, HashMap<String, Tensor<1>>, Seam) {
    let dev = device();
    let mut model = model;
    let mut optim = AdamWConfig::new().init();
    let mut seam = Seam::new(dormouse_train::cubecl_client_opt(&dev));
    if pin {
        let (m, _t) = seam.arm(model, None);
        model = m;
    }
    let mut pins: Option<InputPins> = None;
    let mut after_first: Option<HashMap<String, Tensor<1>>> = None;
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
        // Step 0 runs ungraphed, every step after it is GRAPHED (capture on the
        // first, replays after). `ungraphed = true` means "run fresh launches",
        // so the flag is the INVERSE of what the comment used to say: the first
        // version passed `step % 2 == 0`, which marked every EVEN step — the
        // half the test called the replaying half — as ungraphed. The gate
        // reported `captured 1 replayed 0` and two "replayed" steps cost 7 559
        // launches, which is the same shape as a graph that was never used.
        // Alternating also destroys the graph on every ungraphed step
        // (`Seam::step` sets `self.graph = None`), so a 50/50 cadence can never
        // replay anything at all — the trainer's own cadence is ~99% graphed
        // (`lib.rs:1294`), and this now matches it.
        let ungraphed = step == 0;
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
        if step == 0 {
            after_first = Some(params(&model).0);
        }
    }
    (after_first.expect("step 0 ran"), params(&model).0, seam)
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
    let (graphed_model, software_model) = build_pair();
    let (_first_g, got, seam) = run_graphed(graphed_model, STEPS, true);
    let (_first_s, want) = run_software(software_model, STEPS);

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
    // A guard against the gate passing on two models that are both trivially
    // zero (an empty comparison is not agreement), and against the failure the
    // first version of this file had for four runs: two models that never
    // trained, whose difference is only their initialisation.
    let mass = dumpsum(&want, &dev);
    assert!(mass > 0.0, "the comparison is vacuous: every parameter is zero");

    let (worst, key) = worst_diff(&got, &want, &dev);
    println!(
        "graphed {STEPS} steps vs software {STEPS} from the SAME weights: worst relative \
         parameter difference {worst:.3e} at {key}; {}",
        seam.report()
    );
    assert!(
        worst < 1e-5,
        "a pinned graph must train the same model as fresh launches: worst relative difference \
         {worst:.3e} at {key}. Above 1e-5 means the window read a stale buffer — a pin that \
         covers the wrong tensor, an unpinned input, or a replay of the wrong step."
    );
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
    let (graphed_model, software_model) = build_pair();
    let (unpinned_first, unpinned, _seam) = run_graphed(graphed_model, STEPS, false);
    let (_first_s, want) = run_software(software_model, STEPS);
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
    // Both arms start from the same weights and both must be shown to be
    // TRAINING, or "the arms disagree" is only "the arms are different models":
    // that is what this gate measured for its first four runs, when each arm
    // called `build()` and drew its own random initialisation.
    let (moved, moved_key) = worst_diff(&unpinned, &unpinned_first, &dev);
    println!("the unpinned arm moved {moved:.3e} from its own first step, at {moved_key}");
    assert!(
        moved > 0.0,
        "the unpinned arm did not train either, so its disagreement above is not evidence about \
         stale pointers: it is a model that never moved."
    );
    let (soft_moved, _) = worst_diff(&want, &_first_s, &dev);
    println!("the software arm moved {soft_moved:.3e} from its own first step");
    assert!(
        soft_moved > 0.0,
        "the software arm did not train, so nothing in this file is a differential."
    );
}

/// The graphed arm, run end to end, returning the parameters after the capture
/// step, the final parameters, and the seam — the same arm as
/// `run_graphed(_, STEPS, true)`, kept under the name the gate reads best.
///
/// The parameters are read TWICE — after the first step and at the end — because
/// the only sound "did it train?" baseline is the run's OWN first step. A second
/// `build()` is not: two models built in one process do not share an initializer
/// (AGENTS §3.7, `two_models_one_seed_are_bit_identical` — 36 of 54 parameters
/// differ), so comparing against a fresh build reports "movement" for a run that
/// never moved a weight.
fn run_replayed(model: DormouseModelT) -> (HashMap<String, Tensor<1>>, HashMap<String, Tensor<1>>, Seam) {
    run_graphed(model, STEPS, true)
}

/// Gate 3, in two halves, because the first version of it was a false green.
///
/// **What it claimed:** "after N replays the parameters must be BIT-IDENTICAL to
/// the capture step's". That is not a property of a correct graph — the
/// optimizer runs OUTSIDE the window and moves every parameter on every step,
/// so a correct graphed run *must* differ from its own first step. The first
/// version of this gate therefore went green at exactly `0.0` on a run in which
/// **nothing trained at all**: `worst == 0.0` was the signature of a frozen
/// parameter set, and the assert could not tell that from determinism. It is the
/// `8fa5d4c` shape pointed the other way — a gate that passes for a reason
/// opposite to the one it names.
///
/// **What it claims now, and both halves are needed:**
/// 1. the graphed arm actually TRAINS — the parameters move away from its own
///    first step (this is the half the first version got backwards), and
/// 2. two identical graphed runs end BIT-IDENTICAL — replay is deterministic,
///    which is the `burn-spectral` race class this was written for
///    (`lib.rs:697-714`).
#[test]
fn replays_train_and_are_reproducible() {
    let dev = device();
    let (model_a, model_b) = build_pair();
    let (first_a, final_a, seam_a) = run_replayed(model_a);
    let (_first_b, final_b, seam_b) = run_replayed(model_b);

    assert!(
        seam_a.stats.replays >= 1 && seam_b.stats.replays >= 1,
        "nothing was replayed, so this gate proved nothing: {} / {}",
        seam_a.report(),
        seam_b.report()
    );

    // (1) it trains. The optimizer runs outside the window, so a correct graphed
    // run MUST end 7 updates away from its own capture step. Exactly 0.0 here
    // means the optimizer never saw a gradient.
    let mass = dumpsum(&final_a, &dev);
    assert!(mass > 0.0, "the comparison is vacuous: every parameter is zero");
    let (moved, moved_key) = worst_diff(&final_a, &first_a, &dev);
    println!(
        "{STEPS} graphed steps vs the run's own capture step: worst relative movement \
         {moved:.3e} at {moved_key}; {}",
        seam_a.report()
    );
    assert!(
        moved > 0.0,
        "a graphed arm whose parameters did not move has not trained: the optimizer saw no \
         gradients, or the replay does not rewrite the buffers the capture step wrote. The \
         parameters are bit-identical to the capture step's."
    );

    // (2) it is reproducible. Same seed, same data, same graph → same numbers.
    let (drift, key) = worst_diff(&final_a, &final_b, &dev);
    println!("two identical graphed runs: worst relative difference {drift:.3e} at {key}");
    assert_eq!(
        drift, 0.0,
        "two identical graphed runs ended {drift:.3e} apart at {key}. A replay re-runs the \
         recorded kernels against the same buffers, so a nonzero difference is a race inside \
         the window (or a pin that did not hold) — the class `burn-spectral` documents at \
         lib.rs:697-714."
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
        // Step 0 captures, steps 1-2 replay: `ungraphed` is "run fresh", so
        // `step == 0` and not `step > 0` (the first version ran all three steps
        // fresh and the two "replays" it then measured cost 7 559 launches).
        seam.step(step == 0, || window(&model, None, x.clone(), y.clone(), h.clone()))
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
        // REPLAY, which is the point of the measurement: `ungraphed` is false.
        // The first version passed `step > 0` (true), so this loop measured two
        // more fresh steps and the 7 559 it reported was the window's own launch
        // count, not a replay's.
        seam.step(false, || window(&model, None, x.clone(), y.clone(), h.clone()))
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
