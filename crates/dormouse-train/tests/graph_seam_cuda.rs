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
    // The in-VRAM engram is OFF, as in every graph-mode run this lane measures
    // (--no-engram). Its lookup is atomics-class and its params dominated the
    // differential's failures (3.163e4 at engram.value_proj, 1.448e1 at the
    // Tsct.v it feeds) while embedding/iter_embed agreed to six decimals —
    // filing the capture-compatibility question is a follow-up; gating the
    // measured path is what this file is for. graph::check does not refuse the
    // in-VRAM engram yet (it refuses --engram-ram): that gap is the follow-up.
    cfg.use_engram = false;
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
fn run_software(
    model: DormouseModelT,
    steps: usize,
) -> (Mat, Mat) {
    let dev = device();
    let mut model = model;
    let mut optim = AdamWConfig::new().init();
    let mut after_first: Option<Mat> = None;
    for step in 0..steps {
        let (x, y, h) = batch(step, &dev);
        let grads = window(&model, None, x, y, Some(h));
        model = opt_consuming(&mut optim, 1e-6, model, grads);
        if step == 0 {
            after_first = Some(params_materialized(&model).0);
        }
    }
    (after_first.expect("step 0 ran"), params_materialized(&model).0)
}

/// The graphed arm, with the pin. `pin` = false is the negative control.
fn run_graphed(
    model: DormouseModelT,
    steps: usize,
    pin: bool,
) -> (Mat, Mat, Seam) {
    let dev = device();
    let mut model = model;
    let mut optim = AdamWConfig::new().init();
    let mut seam = Seam::new(dormouse_train::cubecl_client_opt(&dev));
    if pin {
        let (m, _t) = seam.arm(model, None);
        model = m;
    }
    let mut pins: Option<InputPins> = None;
    let mut after_first: Option<Mat> = None;
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
        // Step 0 is ALWAYS plain: the trainer's own cadence never captures on
        // its first step, and a COLD capture (the pool has not seen the
        // window) produced an arm that did not train at all — measured
        // 2026-10-02, gate 2's one-step arm: |iter_embed| stayed at its
        // initial value while the software arm moved ~1e-3. The warm capture
        // (after one plain step) trains, which is the only path the trainer
        // runs.
        let ungraphed = step == 0;
        seam.step(ungraphed, || window(&model, None, x.clone(), y.clone(), h.clone()))
            .expect("seam step");
        if pin {
            let grads = seam.grads().expect("the window ran");
            model = opt_borrowing(&mut optim, 1e-6, model, grads);
            let (m, r) = seam.refresh(model, false);
            model = m;
            r.expect("pin holds");
        } else {
            // The negative control needs the gradients OUT of the seam, because
            // an unpinned graph plus a retained Gradients is a different
            // failure from an unpinned graph plus a consuming optimizer. Both
            // are wrong; only one is the one the handover measured.
            let grads = seam.grads().expect("the window ran");
            model = opt_borrowing(&mut optim, 1e-6, model, grads);
        }
        if step == 0 {
            after_first = Some(params_materialized(&model).0);
        }
    }
    (after_first.expect("step 0 ran"), params_materialized(&model).0, seam)
}

/// The largest relative difference between two MATERIALIZED parameter sets,
/// and where it is.
fn worst_diff(a: &Mat, b: &Mat) -> (f32, String) {
    let mut worst = 0.0f32;
    let mut worst_key = String::from("(none)");
    let mut keys: Vec<&String> = a.keys().collect();
    keys.sort();
    for k in keys {
        let (Some(x), Some(y)) = (a.get(k), b.get(k)) else {
            panic!("parameter {k} is in one run and not the other — the arms ran different models");
        };
        // TENSOR-level scale (max|x| over the whole tensor), not per-entry: a
        // parameter's near-zero entries make per-entry relative differences
        // meaningless (an entry at 1e-8 moving 1e-9 reads as 0.1). A read
        // failure is INFINITE disagreement: the arm corrupted the device.
        let (Ok(x), Ok(y)) = (x, y) else {
            if worst < f32::INFINITY {
                worst = f32::INFINITY;
                worst_key = k.clone();
            }
            continue;
        };
        let num = x
            .iter()
            .zip(y.iter())
            .fold(0.0f32, |m, (xv, yv)| m.max((xv - yv).abs()));
        let scale = x.iter().fold(1e-12f32, |m, xv| m.max(xv.abs()));
        let rel = num / scale;
        if rel > worst {
            worst = rel;
            worst_key = k.clone();
        }
    }
    (worst, worst_key)
}

/// Parameter values MATERIALIZED to host memory at read time. The handle
/// clones `params` returns alias the live master buffers, so a "movement"
/// comparison between two of them reads the same buffer twice and reports
/// exactly 0.0 by construction — the false green this file's gate 3 carried
/// for its whole first life.
type Mat = HashMap<String, Result<Vec<f32>, String>>;

/// Parameter values materialized to host memory at read time. A read can FAIL
/// — the unpinned arm's stale-pointer writes corrupt the device, and the next
/// sync dies with CUDA_ERROR_ILLEGAL_ADDRESS — and the NEGATIVE gate must
/// report that as maximal disagreement instead of dying before its assert.
fn params_materialized(model: &DormouseModelT) -> (Mat, usize) {
    let (tensors, seen) = params(model);
    let mut out = HashMap::new();
    for (k, t) in tensors {
        out.insert(k, t.into_data().try_to_vec::<f32>().map_err(|e| e.to_string()));
    }
    (out, seen)
}

fn dumpsum(p: &Mat) -> f32 {
    let mut total = 0.0f32;
    for v in p.values() {
        if let Ok(v) = v {
            total += v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        }
    }
    total
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
/// Gate 2: the differential, AT THE LEVEL WHERE IT CAN BE EXACT.
///
/// "A pinned graph trains the same model as fresh launches" decomposes into:
/// (a) the window's GRADIENTS are bit-identical whether the step was replayed
/// or run fresh — asserted here, over every entry of every parameter, for a
/// replayed step AND a twice-replayed step AND a fresh step on the same batch;
/// (b) the optimizer is the same deterministic function of those gradients —
/// burn's, unmodified, in both arms. Identical gradients through an identical
/// deterministic optimizer ARE identical training; comparing trained
/// PARAMETERS instead is unfalsifiable on this stack, because this AdamW
/// applies ~lr-scale updates to near-zero tensors (iter_embed lives at ~2e-6)
/// and any last-ulp difference amplifies to O(1) within two steps — the seven
/// random 1e0-scale readings this gate's parameter form produced across its
/// runs (1.416e0, 1.187e0, 8.823e-1, 1.086e0, 1.303e0, 1.628e0, 1.000e0) were
/// that chaos, measured one final time before this rewrite.
#[test]
fn a_pinned_replays_gradients_are_bit_identical_to_fresh_launches() {
    let dev = device();
    let (mut model, _t) = build();
    let mut seam = Seam::new(dormouse_train::cubecl_client_opt(&dev));
    let (m, _t2) = seam.arm(model, None);
    model = m;
    assert!(seam.armed(), "the pin did not arm");
    let mut pins: Option<InputPins> = None;

    struct GradCollect<'a> {
        grads: &'a burn::tensor::Gradients,
        out: Mat,
    }
    impl ModuleVisitor for GradCollect<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            if !param.is_require_grad() {
                return;
            }
            if let Some(g) = param.val().grad(self.grads) {
                let n: usize = g.dims().iter().product();
                let key = format!("{:?}x{}", g.dims(), self.out.len());
                let v = g
                    .clone()
                    .reshape([n])
                    .into_data()
                    .try_to_vec::<f32>()
                    .map_err(|e| e.to_string());
                self.out.insert(key, v);
            }
        }
    }
    fn grad_map(model: &DormouseModelT, grads: &burn::tensor::Gradients) -> Mat {
        let mut c = GradCollect { grads, out: HashMap::new() };
        model.visit(&mut c);
        c.out
    }
    fn worst(a: &Mat, b: &Mat) -> (f32, String) {
        let mut w = 0.0f32;
        let mut wk = String::from("(none)");
        for k in a.keys() {
            let rel = match (a.get(k), b.get(k)) {
                (Some(Ok(x)), Some(Ok(y))) => x
                    .iter()
                    .zip(y.iter())
                    .fold(0.0f32, |m, (xv, yv)| m.max((xv - yv).abs())),
                _ => f32::INFINITY,
            };
            if rel > w {
                w = rel;
                wk = k.clone();
            }
        }
        (w, wk)
    }

    // plain(batch0) -> capture(batch1) -> replay(batch2) -> replay(batch2
    // again) -> fresh(batch2). The two batch2 replays measure replay
    // determinism; the fresh step measures replay fidelity against the
    // ungraphed window on the SAME inputs.
    let mut sets: Vec<(String, Mat)> = Vec::new();
    let plan: [(usize, bool, &str); 5] = [
        (0, true, "plain-batch0"),
        (1, false, "capture-batch1"),
        (2, false, "replay1-batch2"),
        (2, false, "replay2-batch2"),
        (2, true, "fresh-batch2"),
    ];
    for (step, ungraphed, label) in plan {
        let (x, y, h) = batch(step, &dev);
        let (x, y, h) = {
            let p = pins.get_or_insert_with(|| InputPins::new(&x, &y, Some(&h)));
            p.feed(&x, &y, Some(&h)).expect("pin feeds")
        };
        seam.step(ungraphed, || {
            window(&model, None, x.clone(), y.clone(), h.clone())
        })
        .expect("seam step");
        let grads = seam.grads().expect("the window ran");
        sets.push((label.to_string(), grad_map(&model, grads)));
    }
    assert!(
        seam.stats.captures > 0 && seam.stats.replays >= 3,
        "the plan must capture once and replay at least three times (the post-capture \
         execution plus two measured replays): {}",
        seam.report()
    );
    let rep1 = &sets[2].1;
    let rep2 = &sets[3].1;
    let fresh = &sets[4].1;
    let (d_replays, k_r) = worst(rep1, rep2);
    println!("two replays of the same batch: worst absolute difference {d_replays:.3e} at {k_r}");
    assert_eq!(
        d_replays, 0.0,
        "two replays of the SAME batch on the SAME buffers differed by {d_replays:.3e} at \
         {k_r} - a replay is not even deterministic against itself"
    );
    let (d_fresh, k_f) = worst(rep1, fresh);
    println!(
        "replayed gradients vs fresh launches on the same batch: worst absolute difference \
         {d_fresh:.3e} at {k_f}; {}",
        seam.report()
    );
    assert_eq!(
        d_fresh, 0.0,
        "a replayed window's gradients differ from a fresh window's by {d_fresh:.3e} (absolute, \
         {k_f}) - the replay computes something else"
    );
}

#[test]
fn the_stale_pointer_trap_is_reproduced_without_the_pin() {
    let dev = device();
    let (graphed_model, software_model) = build_pair();
    // THREE steps, not STEPS: one replay, then read. The stale-pointer writes
    // CORRUPT the device (cuEventCreate status 700 in the fence, measured) and
    // the corruption compounds per replay - eight replays can kill the process
    // before the assert runs, which turns the trap's strongest evidence into a
    // crash instead of a number. One replay already diverges; this gate also
    // runs as its own process (see tools' invocation) so a corruption death
    // cannot take the other gates' results with it.
    let (unpinned_first, unpinned, _seam) = run_graphed(graphed_model, 3, false);
    let (_first_s, want) = run_software(software_model, 3);
    let (worst, key) = worst_diff(&unpinned, &want);
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
    let (moved, moved_key) = worst_diff(&unpinned, &unpinned_first);
    println!("the unpinned arm moved {moved:.3e} from its own first step, at {moved_key}");
    assert!(
        moved > 0.0,
        "the unpinned arm did not train either, so its disagreement above is not evidence about \
         stale pointers: it is a model that never moved."
    );
    let (soft_moved, _) = worst_diff(&want, &_first_s);
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
fn run_replayed(model: DormouseModelT) -> (Mat, Mat, Seam) {
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
    // means the optimizer never saw a gradient — or, as in the first nine runs
    // of this gate, that `first_a` aliased the live master buffers and read
    // them twice (movement 0.0 BY CONSTRUCTION; the maps are materialized to
    // host memory at capture time to make the comparison mean anything).
    let mass = dumpsum(&final_a);
    assert!(mass > 0.0, "the comparison is vacuous: every parameter is zero");
    let (moved, moved_key) = worst_diff(&final_a, &first_a);
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
    // The old second half here asserted two graphed RUNS end bit-identical.
    // Measured 2026-10-02: false for any training arm of this stack - this
    // AdamW moves near-zero tensors (iter_embed ~2e-6) by ~lr per step, so
    // last-ulp address noise between two runs amplifies to O(1) within the
    // eight steps (readings 7.538e3, 5.265e1, 1.001e0). Replay determinism is
    // asserted where it is well-defined: same step, same buffers, bit-exact -
    // in `a_pinned_replays_gradients_are_bit_identical_to_fresh_launches`.
    let _ = final_b;
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
        // Step 0 runs plain (no graph yet), step 1 captures (and replays once
        // to execute the recording), step 2 replays: `ungraphed` is "run
        // fresh", so `step == 0` (the first version ran all three steps fresh
        // and the two "replays" it then measured cost 7 559 launches).
        seam.step(step == 0, || window(&model, None, x.clone(), y.clone(), h.clone()))
            .expect("seam step");
        let grads = seam.grads().expect("the window ran");
        model = opt_borrowing(&mut optim, 1e-6, model, grads);
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

/// MEASUREMENT, not a gate: which step's gradients are zero?
///
/// The first gate run (2026-10-02, /tmp/opencode/graph3_gates.out) showed the
/// pinned arm registering 38 gradients on every step yet moving nothing — gate
/// 3's movement was exactly 0.0 — so the registered gradients must be
/// zero-valued somewhere between the capture step and the replays. This test
/// reads one gradient back per step (a host sync, which is why it is a probe
/// and the gates above never do this) and prints the loss the window saw.
///
/// The loss is read AFTER `seam.step` returns, never inside the closure: a
/// host sync inside a capture window faults the stream (cuEventSynchronize
/// 907, first probe run), and `try_into_scalar` turns a fault into a printed
/// value instead of a panic.
#[test]
fn probe_which_steps_produce_zero_gradients() {
    use std::cell::RefCell;
    let dev = device();
    let (mut model, _t) = build();
    let mut optim = AdamWConfig::new().init();
    let mut seam = Seam::new(dormouse_train::cubecl_client_opt(&dev));
    let (m, _t2) = seam.arm(model, None);
    model = m;
    let mut pins: Option<InputPins> = None;
    fn rd(t: burn::tensor::Tensor<1>) -> String {
        match t.try_into_scalar::<f32>() {
            Ok(v) => format!("{v:.7}"),
            Err(e) => format!("ERR {}", e.to_string().chars().take(60).collect::<String>()),
        }
    }
    // iter_embed is the watched parameter: values ~0.03 (ulp ~2e-9), so an
    // AdamW delta of ~2e-7 is ~100 ulp - VISIBLE. The embedding's 0.5-scale
    // entries round a 1e-7 update away, which is what hid the answer in v3.
    fn ie(model: &DormouseModelT) -> String {
        rd(model.loop_block.iter_embed.val().abs().max().reshape([1]))
    }
    fn ie_grad(model: &DormouseModelT, grads: &burn::tensor::Gradients) -> String {
        match model.loop_block.iter_embed.val().grad(grads) {
            Some(g) => rd(g.abs().max().reshape([1])),
            None => "grad-missing".into(),
        }
    }
    for step in 0..4 {
        let (x, y, h) = batch(step, &dev);
        let (x, y, h) = {
            let p = pins.get_or_insert_with(|| InputPins::new(&x, &y, Some(&h)));
            p.feed(&x, &y, Some(&h)).expect("pin feeds")
        };
        let kind = if step == 0 { "plain" } else if step == 1 { "capture" } else { "replay" };
        seam.step(step == 0, || {
            window(&model, None, x.clone(), y.clone(), h.clone())
        })
        .expect("seam step");
        let grads = seam.grads().expect("the window ran");
        println!(
            "step {step} ({kind}): pre={} d[ie]={}",
            ie(&model),
            ie_grad(&model, grads)
        );
        model = opt_borrowing(&mut optim, 1e-6, model, grads);
        let after_opt = ie(&model);
        let (m, r) = seam.refresh(model, false);
        model = m;
        println!(
            "step {step} ({kind}): after_opt={after_opt} after_refresh={} refresh={}",
            ie(&model),
            match r { Ok(_) => "ok".into(), Err(e) => format!("ERR {e}") },
        );
    }
}

#[test]
fn probe_replay_matches_fresh_window() {
    use std::cell::RefCell;
    let dev = device();
    let (mut model, _t) = build();
    let mut seam = Seam::new(dormouse_train::cubecl_client_opt(&dev));
    let (m, _t2) = seam.arm(model, None);
    model = m;
    let mut pins: Option<InputPins> = None;

    // Full per-parameter gradient maps, materialized. The max-abs comparison
    // this probe used before read ONE entry per tensor and called two
    // distributions equal; this reads every entry.
    struct GradCollect<'a> {
        grads: &'a burn::tensor::Gradients,
        out: Mat,
    }
    impl ModuleVisitor for GradCollect<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            if !param.is_require_grad() {
                return;
            }
            if let Some(g) = param.val().grad(self.grads) {
                let n: usize = g.dims().iter().product();
                let key = format!("{:?}x{}", g.dims(), self.out.len());
                let v = g
                    .clone()
                    .reshape([n])
                    .into_data()
                    .try_to_vec::<f32>()
                    .map_err(|e| e.to_string());
                self.out.insert(key, v);
            }
        }
    }
    fn grad_map(model: &DormouseModelT, grads: &burn::tensor::Gradients) -> Mat {
        let mut c = GradCollect { grads, out: HashMap::new() };
        model.visit(&mut c);
        c.out
    }
    fn rel_stats(a: &Mat, b: &Mat) -> (f32, f32, String) {
        let mut rels: Vec<(f32, String)> = vec![];
        for k in a.keys() {
            match (a.get(k), b.get(k)) {
                (Some(Ok(x)), Some(Ok(y))) => {
                    let num = x
                        .iter()
                        .zip(y.iter())
                        .fold(0.0f32, |m, (xv, yv)| m.max((xv - yv).abs()));
                    let scale = x.iter().fold(1e-12f32, |m, xv| m.max(xv.abs()));
                    rels.push((num / scale, k.clone()));
                }
                _ => rels.push((f32::INFINITY, k.clone())),
            }
        }
        rels.sort_by(|p, q| p.0.total_cmp(&q.0));
        let worst = rels.last().map(|r| r.0).unwrap_or(0.0);
        let med = rels.get(rels.len() / 2).map(|r| r.0).unwrap_or(0.0);
        let key = rels.last().map(|r| r.1.clone()).unwrap_or_default();
        (worst, med, key)
    }

    let mut sets: Vec<(String, Mat)> = Vec::new();
    let plan: [(usize, bool, &str); 5] = [
        (0, true, "plain-batch0"),
        (1, false, "capture-batch1"),
        (2, false, "replay-batch2"),
        (2, true, "fresh-batch2"),
        (2, true, "fresh2-batch2"),
    ];
    for (step, ungraphed, label) in plan {
        let (x, y, h) = batch(step, &dev);
        let (x, y, h) = {
            let p = pins.get_or_insert_with(|| InputPins::new(&x, &y, Some(&h)));
            p.feed(&x, &y, Some(&h)).expect("pin feeds")
        };
        seam.step(ungraphed, || {
            window(&model, None, x.clone(), y.clone(), h.clone())
        })
        .expect("seam step");
        let grads = seam.grads().expect("the window ran");
        sets.push((label.to_string(), grad_map(&model, grads)));
    }
    let cap = &sets[1].1;
    let rep = &sets[2].1;
    let fresh1 = &sets[3].1;
    let fresh2 = &sets[4].1;
    let (w1, m1, k1) = rel_stats(fresh1, fresh2);
    println!("fresh-vs-fresh (same batch, no graphs): worst={w1:.3e} median={m1:.3e} at {k1}");
    let (w2, m2, k2) = rel_stats(rep, fresh1);
    println!("replay-vs-fresh:                         worst={w2:.3e} median={m2:.3e} at {k2}");
    let (w3, m3, k3) = rel_stats(rep, cap);
    println!("replay-vs-capture:                       worst={w3:.3e} median={m3:.3e} at {k3}");
    assert!(
        w1 < 1e-5,
        "the window itself is nondeterministic: two FRESH runs on the same batch differ by {w1:.3e} at {k1} - the differential gates are measuring the window, not the graphs"
    );
}
