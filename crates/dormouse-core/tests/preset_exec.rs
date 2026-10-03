//! Every shipped preset, RUN: what it resolves to, what it costs, and what
//! actually EXECUTED.
//!
//! A preset is a promise. This file is the part of the project that holds the
//! model to it, per preset, on the CPU backend so the suite stays runnable
//! without a GPU.
//!
//! Three things are asserted, in this order, because they fail in this order:
//!
//! 1. **Resolved config** - the preset parses, validates, and the fields that
//!    decide execution are the ones the TOML says.
//! 2. **Cost** - the real parameter count, split into memory rows and
//!    computation, so a config's price is visible where the config is chosen
//!    and not discovered at OOM. The counts come from a REAL instantiated
//!    model, not a closed form: a formula that drifts from the module tree is
//!    exactly the kind of number that reads true and lies.
//! 3. **Execution** - the branch counters in [`dormouse_core::probe`]. Each
//!    arm the preset declares ON must have been ENTERED, the loop must have
//!    run its configured number of iterations, and the aux objectives must
//!    have run with their configured weights.
//!
//! Sizing: the forward runs a WIDTH-SHRUNK fixture of the preset, because a
//! p150 forward is not a CPU test. Everything that decides *what executes* is
//! the preset's own value and is asserted equal to it; only the widths and
//! the table capacity shrink. That equality IS the check (`execution_fields`,
//! asserted at the top of `executes`) — keep the fixture and that list
//! together, or the fixture stops being the preset.
//!
//! Run: `cargo test -p dormouse-core --test preset_exec`
//! The CUDA half: `cargo test -p dormouse-core --features cuda --test preset_exec -- --ignored`

use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn::backend::autodiff::Autodiff;
use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::{Device, Int, Tensor, TensorData};
use dormouse_core::aux::ema_update;
use dormouse_core::config::load_config;
use dormouse_core::probe;
use dormouse_core::{fnv_hash, DormouseConfig, DormouseModel};

#[allow(deprecated)] // the alias the train crate uses for `--features cpu`
type B = Autodiff<burn::backend::Flex, BalancedCheckpointing>;

#[allow(deprecated)]
fn device() -> Device {
    Device::flex().autodiff()
}

/// The repo's `configs/`, so a preset added tomorrow is in scope today.
fn configs_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs")
}

/// Every shipped preset, sorted. Read from disk, not a hand-kept list.
fn shipped_presets() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(configs_dir())
        .expect("configs/ exists")
        .map(|e| e.expect("readable dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .filter_map(|p| p.file_stem()?.to_str().map(str::to_string))
        .collect();
    v.sort();
    v
}

fn preset(name: &str) -> DormouseConfig {
    let path = configs_dir().join(format!("{name}.toml"));
    let c =
        load_config(path.to_str().expect("utf-8 path")).unwrap_or_else(|e| panic!("{name}: {e}"));
    dormouse_core::config::validate(&c).unwrap_or_else(|e| panic!("{name} must validate: {e}"));
    c
}

/// The preset with its WIDTHS shrunk. Everything that decides what executes
/// (arms, depth, expert count, aux weights, quant format, MoR, GR) is the
/// preset's own value; only `d_model`/`n_heads`/`head_dim`/`d_ffn`/`rank`/
/// `max_seq_len`, the n-gram table capacity and — for a byteflow preset —
/// `byteflow_k_tokens` shrink.
///
/// The first group is what [`execution_fields`] forbids touching, and it is
/// deliberately a list of ARM switches. `byteflow_k_tokens` is not on it: it
/// is a capacity, in the same class as `max_seq_len` (the chunker asserts
/// `K <= T` in `encode_chunks`, the fixture's window is `SEQ` = 32 bytes, and
/// the preset ships K = 128 for a 512-byte window), and shrinking it does not
/// decide which mechanism runs. Do not add an arm switch here — add it to
/// `execution_fields`, or the fixture stops being the preset.
fn fixture(c: &DormouseConfig) -> DormouseConfig {
    DormouseConfig {
        d_model: 64,
        n_heads: 4,
        head_dim: 16,
        d_ffn: 128,
        rank: 16,
        max_seq_len: 64,
        engram_rows: 512,
        byteflow_k_tokens: if c.use_byteflow {
            8
        } else {
            c.byteflow_k_tokens
        },
        ..c.clone()
    }
}

/// The fields a fixture is NOT allowed to change. Every arm switch, the loop
/// depth, the expert count, the aux weights, the quant format and the memory
/// floor. This is the list that makes "a mini preset" honest: a fixture that
/// shrank one of these would be testing a model the preset never declared.
fn execution_fields(c: &DormouseConfig) -> Vec<(&'static str, String)> {
    vec![
        ("bf16", c.bf16.to_string()),
        ("act_quant", format!("{:?}", c.act_quant)),
        ("use_kda", c.use_kda.to_string()),
        ("use_tsct", c.use_tsct.to_string()),
        ("use_engram", c.use_engram.to_string()),
        ("use_gr", c.use_gr.to_string()),
        ("use_mor", c.use_mor.to_string()),
        ("mor_k", c.mor_k.to_string()),
        ("mor_bce_weight", c.mor_bce_weight.to_string()),
        ("max_iter", c.max_iter.to_string()),
        ("n_experts", c.n_experts.to_string()),
        // The sparse-routing arm decides which expert each token uses on each
        // pass, so it changes what the model IS, not how it is run: a config
        // that differed only here would train a different network and this
        // tuple would call it the same run.
        ("moe_topk", c.moe_topk.to_string()),
        ("moe_lb_coef", c.moe_lb_coef.to_string()),
        ("engram_lam_max", c.engram_lam_max.to_string()),
        ("engram_dim", c.engram_dim.to_string()),
        ("jepa_weight", c.jepa_weight.to_string()),
        ("jepa_mask_frac", c.jepa_mask_frac.to_string()),
        ("jepa_mask_span", c.jepa_mask_span.to_string()),
        ("dspark_weight", c.dspark_weight.to_string()),
        ("dspark_k", c.dspark_k.to_string()),
        ("dspark_stride", c.dspark_stride.to_string()),
        ("vocab", c.vocab.to_string()),
    ]
}

const STEPS: usize = 3;
const BATCH: usize = 2;
const SEQ: usize = 32;

/// The shared body of every per-preset test.
fn executes(name: &str) {
    let t0 = std::time::Instant::now();
    let cfg = preset(name);
    // The fixture must be the preset.
    let fx = fixture(&cfg);
    assert_eq!(
        execution_fields(&fx),
        execution_fields(&cfg),
        "{name}: the test fixture changed an execution-deciding field"
    );

    let dev = device();
    let model = DormouseModel::new(&fx, &dev);
    let teacher = ema_update(model.clone(), &model, 0.0);
    let bytes = batch_bytes(name.len() as u64, BATCH * SEQ);
    let x = input_ids(&bytes, &dev);
    let h = hashed_ids(&bytes, &dev);
    let y = targets(&bytes, &dev);

    // ---- EXECUTION -------------------------------------------------------
    probe::reset();
    let mut last = None;
    for _ in 0..STEPS {
        last = Some(model.forward_with_hidden::<B>(
            x.clone(),
            Some(h.clone()),
            None,
            Some(y.clone()),
            Some(&teacher),
        ));
    }
    let (_logits, rec, kda, aux) = last.expect("three steps ran");
    let finite = |t: &Tensor<1>| {
        let v: Vec<f32> = t
            .clone()
            .into_data()
            .try_to_vec()
            .expect("readable [1] tensor");
        v.iter().all(|x| x.is_finite())
    };
    assert!(finite(&rec), "{name}: loss is not finite");

    // The loop ran `max_iter` iterations per forward, TWICE per step when a
    // teacher is attached (the JEPA teacher is a second full forward - a
    // mechanism that silently stopped running would halve this number, which
    // is why it is counted and not inferred).
    let iters_per_step = 2 * fx.max_iter * STEPS;
    assert_eq!(
        probe::count(probe::ITER),
        iters_per_step as u64,
        "{name}: the loop did not run max_iter={} iterations per pass ({iters_per_step} expected)\n  counters: {:?}",
        fx.max_iter,
        probe::counts()
    );

    // Every arm: entered iff the preset declares it, and once per iteration
    // when it does.
    let arm = |on: bool, counter: usize| {
        let want = if on { iters_per_step as u64 } else { 0 };
        assert_eq!(
            probe::count(counter),
            want,
            "{name}: {} counter is {} but the preset declares it {} (expected {want})\n  counters: {:?}",
            probe::NAMES[counter],
            probe::count(counter),
            if on { "ON" } else { "off" },
            probe::counts()
        );
    };
    arm(cfg.use_kda, probe::KDA);
    arm(cfg.use_engram, probe::ENGRAM);
    // The memory branch: entered AND read. An arm that is entered with no
    // keys is inert - the difference between a memory and a table nobody
    // consults, and invisible in the output.
    arm(cfg.use_engram, probe::ENGRAM_KEYS);
    arm(cfg.use_mor, probe::MOR);
    arm(cfg.use_gr, probe::GR);
    arm(cfg.use_attnres, probe::ATTNRES);
    arm(fx.act_quant.is_some(), probe::ACT_QUANT);
    // The sparse-routing arm. `moe_topk > 0` must show the selection ran once
    // per executed iteration: a top-k that could not fill computes the dense
    // blend, so the arm trains the control while its config says otherwise.
    arm(fx.moe_topk > 0, probe::MOE_ROUTE);
    // THE COMPAT CHANNEL: entered iff the model BUILT one. The condition is
    // `compat_armed` and not `use_byteflow`, because the standalone byteflow
    // preset has the flag with `model.bf = None` (the trainer sends it down
    // another dispatch before a model exists) — counting on the flag would
    // demand a channel that model never had. ONE entry per step, not
    // `iters_per_step`: the JEPA teacher's `forward_latent` is the byte-level
    // latent path and goes through the plain loop, not the channel, and there
    // is nothing for it to run through — every aux weight is refused with the
    // arm, so its latent is discarded before it can matter.
    let bf_on = dormouse_core::bf::compat_armed(&cfg);
    let bf_want = if bf_on { STEPS as u64 } else { 0 };
    assert_eq!(
        probe::count(probe::BF_CHANNEL),
        bf_want,
        "{name}: the compat channel ran {} times, {bf_want} expected (armed = {bf_on})\n  counters: {:?}",
        probe::count(probe::BF_CHANNEL),
        probe::counts()
    );
    // The KDA state: with the arm on it is a real [b, heads, k, v] state; with
    // it off the loop substitutes a [b,1,1,1] placeholder. Shape, not a
    // counter, because it is the returned value the trainer persists.
    if cfg.use_kda {
        assert_eq!(kda.dims()[0], BATCH, "{name}: kda state batch dim");
        assert!(
            kda.dims().iter().skip(1).any(|d| *d > 1),
            "{name}: kda state is the placeholder shape {:?}",
            kda.dims()
        );
    } else {
        assert_eq!(
            kda.dims(),
            [BATCH, 1, 1, 1],
            "{name}: the KDA arm is off, so there is no state"
        );
    }

    // ---- AUX OBJECTIVES: ran, with the configured weights ----------------
    // A weight of 0 must mean the term did not run at all (a head that
    // computed a loss nobody added is a cost with no effect), and `aux` is
    // then None unless the MoR BCE is on.
    let jepa_on = cfg.jepa_weight > 0.0;
    let dspark_on = cfg.dspark_weight > 0.0 && cfg.dspark_k > 0;
    assert_eq!(
        probe::count(probe::JEPA),
        STEPS as u64 * jepa_on as u64,
        "{name}: JEPA ran with weight {}",
        cfg.jepa_weight
    );
    assert_eq!(
        probe::count(probe::DSPARK),
        STEPS as u64 * dspark_on as u64,
        "{name}: DSpark ran with weight {}",
        cfg.dspark_weight
    );
    assert_eq!(
        probe::count(probe::MOR_BCE),
        STEPS as u64 * cfg.use_mor as u64,
        "{name}: the MoR BCE ran with weight {}",
        cfg.mor_bce_weight
    );
    // The routing balancer, counted where the term is ADDED. A non-zero
    // coefficient with no selection is refused by `validate`, so this is a
    // check that the term reached the objective rather than that a flag was
    // spelled right.
    let moe_on = fx.moe_lb_coef > 0.0;
    assert_eq!(
        probe::count(probe::MOE_LB),
        STEPS as u64 * moe_on as u64,
        "{name}: the routing balancer ran with weight {}",
        fx.moe_lb_coef
    );
    let any_aux = jepa_on || dspark_on || cfg.use_mor || moe_on;
    match (any_aux, &aux) {
        (true, Some(a)) => assert!(finite(a), "{name}: aux is not finite"),
        (true, None) => panic!("{name}: every aux weight is on but aux is None"),
        (false, Some(_)) => panic!("{name}: every aux weight is 0 but an aux loss was returned"),
        (false, None) => {}
    }

    if dspark_on {
        // The DSpark term is deterministic given the inputs (no RNG, no
        // dropout), so its configured weight is observable EXACTLY: with
        // JEPA muted, aux is `weight * L_dspark`. Two extra forwards.
        let mut m = model.clone();
        m.jepa_weight = 0.0;
        m.mor_bce_weight = 0.0;
        m.dspark_weight = 1.0;
        let unit = m
            .forward_with_hidden::<B>(x.clone(), Some(h.clone()), None, Some(y.clone()), None)
            .3
            .expect("dspark only")
            .into_scalar::<f32>();
        m.dspark_weight = cfg.dspark_weight;
        let scaled = m
            .forward_with_hidden::<B>(x.clone(), Some(h.clone()), None, Some(y.clone()), None)
            .3
            .expect("dspark only")
            .into_scalar::<f32>();
        assert!(
            (scaled - cfg.dspark_weight * unit).abs() < 1e-4 * unit.abs().max(1.0),
            "{name}: dspark_weight={} is not the multiplier applied (unit {unit:.6}, scaled {scaled:.6})",
            cfg.dspark_weight
        );
    }

    if jepa_on {
        // The JEPA mask is drawn fresh per call (that is the point of it), so
        // its exact multiplier is not observable from two forwards. What IS
        // observable, exactly: with a teacher the aux total is the
        // no-teacher total PLUS a non-negative JEPA term. A sign error, a
        // detached target or a term added to the wrong side all fail here;
        // "how much" is a masked quantity by construction and is not claimed.
        // BOTH forwards run on `ref_m`, so the ONLY difference is the teacher.
        // (Comparing `model` against the clone would compare two different
        // objectives: JEPA-only against DSpark-only.) On a clone, so the probe
        // counts asserted above are untouched.
        let mut ref_m = model.clone();
        ref_m.dspark_weight = 1.0;
        let with = ref_m
            .forward_with_hidden::<B>(
                x.clone(),
                Some(h.clone()),
                None,
                Some(y.clone()),
                Some(&teacher),
            )
            .3
            .expect("jepa on")
            .into_scalar::<f32>();
        // "No teacher" leaves DSpark's term, not "JEPA alone". Every preset
        // ships dspark_weight = 0.0 (DeepSeek's own MTP ablation reports the
        // head bits-per-byte neutral, and our protocol measures BPB), so on a
        // shipped preset the no-teacher forward has NOTHING to sum and `aux`
        // is correctly `None` - `any` at model.rs:254 needs a non-zero weight
        // plus dspark_k > 0. Hence `ref_m` above, which enables the arm for
        // BOTH sides so the subtraction isolates the teacher.
        let without = ref_m
            .forward_with_hidden::<B>(x.clone(), Some(h.clone()), None, Some(y.clone()), None)
            .3
            .expect("dspark only")
            .into_scalar::<f32>();
        assert!(
            with >= without - 1e-4,
            "{name}: the JEPA term is not an addition ({with:.6} with teacher vs {without:.6} without)"
        );
        println!(
            "{name}: jepa contributes {:.6} (mask drawn per call, not pinnable)",
            with - without
        );
    }

    // ---- DEPTH: the loop ran the configured number of iterations ----------
    // Without a teacher, one forward is exactly `max_iter` iterations - the
    // cleanest read on the number. And the depth override (--eval-depths,
    // --rand-depth) must actually truncate the loop, not just be accepted.
    probe::reset();
    let _ = model.forward::<B>(x.clone(), Some(h.clone()));
    assert_eq!(
        probe::count(probe::ITER),
        fx.max_iter as u64,
        "{name}: one teacher-free forward must be max_iter iterations"
    );
    let mut shallow = model.clone();
    shallow.set_loop_depth(Some(2));
    probe::reset();
    let _ = shallow.forward::<B>(x, Some(h));
    assert_eq!(
        probe::count(probe::ITER),
        2,
        "{name}: set_loop_depth(2) did not truncate the loop"
    );
    probe::reset();
    println!("{name}: ok, max_iter={} experts={} arms=kda:{} engram:{} gr:{} mor:{} jepa:{} dspark:{} ({} ms)",
        fx.max_iter, fx.n_experts, cfg.use_kda, cfg.use_engram, cfg.use_gr, cfg.use_mor,
        cfg.jepa_weight, cfg.dspark_weight, t0.elapsed().as_millis());
}

// --- one test per shipped preset -----------------------------------------

#[test]
fn nano_executes() {
    executes("nano");
}
#[test]
fn small_executes() {
    executes("small");
}
#[test]
fn base_executes() {
    executes("base");
}
#[test]
fn swift50_executes() {
    executes("swift50");
}
#[test]
fn one_b_executes() {
    executes("one_b");
}
#[test]
fn p150_executes() {
    executes("p150");
}
/// A preset whose forward CANNOT run on this backend, and why. The class test
/// below skips these by name (it would otherwise pass vacuously: a panicking
/// forward enters nothing, and "entered nothing" is indistinguishable from
/// "entered nothing because every arm is off"). The failure is real and it is
/// not in this file: `burn_mor::topk_indices` returns an index tensor whose
/// storage is I64 while `Tensor<D, Int>` is i32 on burn-flex, and `mor::route`
///'s `equal` against the slot arange refuses to mix them ("expected I32, got
/// I64", burn-flex tensor.rs:170). So `mor` cannot execute a single step here
/// - a finding about the MoR arm, fixed where the indices are built.
const UNVERIFIABLE: &[&str] = &["mor"];

#[test]
#[ignore = "the MoR arm panics on the CPU backend: burn-mor topk_indices is I64 where Tensor<D, Int> is i32"]
fn mor_executes() {
    executes("mor");
}
#[test]
fn nano_fused_executes() {
    executes("nano-fused");
}
/// The standalone ByteFlow arm: `use_byteflow` with NO dormouse arm on, so
/// `model.bf` is `None` and this forward is the plain dormouse loop. That is
/// NOT the path the trainer runs for this preset — `train_loop` dispatches it
/// to `byteflow::train_loop` (ByteFlowNet itself) before a `DormouseModel`
/// exists — so what this test holds is the preset's own promises: it parses,
/// it validates, and it declares no arm. The channel below is the other
/// dispatch.
#[test]
fn byteflow_executes() {
    executes("byteflow");
}
/// THE COMPAT CHANNEL (lane bf-compat, `crates/dormouse-core/src/bf.rs`):
/// `use_byteflow` + `use_kda` keeps the model. ByteFlow's front stage packs
/// the window into K patch latents, the loop runs KDA over those K positions,
/// the back stage lifts them to byte logits, and the objective is the
/// decoder's byte CE. `executes` proves the pair resolves and validates, the
/// KDA arm entered on PATCH input, the channel counter moved, and the byte CE
/// came back finite.
#[test]
fn byteflow_kda_executes() {
    executes("byteflow_kda");
}

/// The presets that have an execution test above. A preset added to
/// `configs/` without one is the exact hole this file exists to close, so the
/// list is checked against the directory, not trusted.
const COVERED: &[&str] = &[
    "base",
    "byteflow",
    "byteflow_kda",
    "mor",
    "nano",
    "nano-fused",
    "one_b",
    "p150",
    "small",
    "swift50",
];

#[test]
fn every_shipped_preset_has_an_execution_test() {
    let shipped = shipped_presets();
    assert!(
        !shipped.is_empty(),
        "no presets found in {}",
        configs_dir().display()
    );
    let missing: Vec<&String> = shipped
        .iter()
        .filter(|p| !COVERED.contains(&p.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "presets with no execution test: {missing:?} - add a `fn <name>_executes() {{ executes(\"<name>\") }}` \
         and add the name to COVERED, or delete the preset"
    );
}

/// The class test, named for the failure it prevents: a preset DECLARES an
/// arm and never enters it.
///
/// The fused gated-delta kernels were dead for a year behind exactly this
/// shape - the run was correct, the fallback computed the same function, and
/// nothing in the config said the fast path had been skipped. A config is a
/// claim about what ran; this asserts the claim against the counters. It
/// fails the moment an arm's branch stops being reachable, whatever the
/// reason (a renamed field, a gate that closed, a branch that moved).
#[test]
fn a_preset_that_declares_an_arm_it_never_enters_fails_here() {
    for name in shipped_presets() {
        if UNVERIFIABLE.contains(&name.as_str()) {
            println!("{name}: SKIPPED - its forward cannot run on this backend (see mor_executes)");
            continue;
        }
        let cfg = preset(&name);
        let fx = fixture(&cfg);
        let dev = device();
        let model = DormouseModel::new(&fx, &dev);
        let teacher = ema_update(model.clone(), &model, 0.0);
        let bytes = batch_bytes(7, BATCH * SEQ);
        let x = input_ids(&bytes, &dev);
        let h = hashed_ids(&bytes, &dev);
        let y = targets(&bytes, &dev);
        probe::reset();
        let _ = model.forward_with_hidden::<B>(x, Some(h), None, Some(y), Some(&teacher));

        // (does the config declare it ON?, the counter that proves it ran)
        let declared: Vec<(bool, usize)> = vec![
            (cfg.use_kda, probe::KDA),
            (cfg.use_engram, probe::ENGRAM),
            (cfg.use_mor, probe::MOR),
            (cfg.use_gr, probe::GR),
            (cfg.use_attnres, probe::ATTNRES),
            (fx.act_quant.is_some(), probe::ACT_QUANT),
            (cfg.jepa_weight > 0.0, probe::JEPA),
            (cfg.dspark_weight > 0.0 && cfg.dspark_k > 0, probe::DSPARK),
            (cfg.use_mor, probe::MOR_BCE),
            // The compat channel (lane bf-compat): the model built one iff
            // `compat_armed`, and it must have been ENTERED — a channel that
            // exists and never runs is the same defect as any other dead arm.
            (dormouse_core::bf::compat_armed(&cfg), probe::BF_CHANNEL),
        ];
        let never_entered: Vec<&str> = declared
            .iter()
            .filter(|(on, counter)| *on && probe::count(*counter) == 0)
            .map(|(_, counter)| probe::NAMES[*counter])
            .collect();
        assert!(
            never_entered.is_empty(),
            "{name}: the preset declares ON {never_entered:?} and the run entered NONE of them.\n  \
             A declared arm that is never entered is the defect this test exists for: the run is \
             correct and the mechanism is dead. counters: {:?}",
            probe::counts()
        );
    }
}

// --- cost: what the preset actually costs ---------------------------------

/// Sums elements per parameter, split by which subtree the parameter is in.
/// `memory` is the n-gram table (`loop_block.engram.memory`): rows, not
/// computation. The split is the number that matters - a preset whose count is
/// mostly rows is a lookup table with a model attached. The Engram's key and
/// value PROJECTIONS are compute; only the table is memory.
#[derive(Default, Debug)]
struct Cost {
    compute: usize,
    memory: usize,
    groups: usize,
    stack: Vec<String>,
}

impl ModuleVisitor for Cost {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.stack.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.stack.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let n: usize = param.val().clone().dims().iter().product();
        // `loop_block.engram.memory` is the table; `engram.key_projs` /
        // `engram.value_proj` are projections of the hidden state onto it.
        let is_table = self
            .stack
            .windows(2)
            .any(|w| w[0] == "engram" && w[1] == "memory");
        if is_table {
            self.memory += n;
        } else {
            self.compute += n;
        }
        self.groups += 1;
    }
}

struct CostSplit {
    compute: usize,
    memory: usize,
    groups: usize,
}

fn cost_of(model: &DormouseModel) -> CostSplit {
    let mut c = Cost::default();
    model.visit(&mut c);
    CostSplit {
        compute: c.compute,
        memory: c.memory,
        groups: c.groups,
    }
}

/// THE COST of one preset, on the real instantiated model, with every claim
/// this file makes about it checked. A config's price should be visible where
/// the config is chosen.
///
/// The monopoly check is the point. `small` used to ship 48M memory rows
/// against a 7.5M backbone - 86% of the model, and the reason nearly every
/// run in the project's history was rescued by `--no-engram` rather than by a
/// config. That shape must be impossible to reintroduce silently, so it is an
/// assertion with the number in the message.
fn cost_of_preset(name: &str) -> CostSplit {
    let t0 = std::time::Instant::now();
    let cfg = preset(name);
    let dev = device();
    let model = DormouseModel::new(&cfg, &dev);
    let c = cost_of(&model);
    let total = c.compute + c.memory;
    // The table count is the config's budget rounded up to a power of two
    // (the in-model read masks the hash), so the measured row count must be
    // AT LEAST the config's - if it is not, the config is not the capacity and
    // the cost line below is fiction.
    let (tables, _mask) =
        dormouse_core::loop_block::engram_tables(cfg.engram_rows, cfg.engram_orders.len());
    let min_rows: usize = tables.iter().sum::<usize>() * cfg.engram_dim;
    assert!(
        c.memory >= min_rows,
        "{name}: measured {} memory params, below the config's own {min_rows} ({} rows x {} dim)",
        c.memory,
        tables[0],
        cfg.engram_dim
    );
    let share = c.memory as f64 / total as f64;
    println!(
        "{name:>11}: total {total:>12}  compute {:>12}  memory-rows {:>12}  {:>5.1}% of the model  {} param tensors  ({} ms)",
        k(c.compute), k(c.memory), share * 100.0, c.groups, t0.elapsed().as_millis()
    );
    assert!(
        total > 0 && c.groups > 10,
        "{name}: the visitor found {} params over {total} elements - the model is not what this test thinks",
        c.groups
    );
    assert!(
        share <= 0.5,
        "{name}: {:.1}% of its {} parameters are n-gram memory rows ({} rows vs {} compute) - \
         that is the monopoly shape: a lookup table with a model attached, and the reason a run \
         gets rescued by a flag instead of a config",
        share * 100.0,
        k(total),
        k(c.memory),
        k(c.compute)
    );
    // Every parameter of every shipped preset is declared to an optimizer
    // group, exactly once - and the counts follow the preset's topology, not
    // literals.
    let r = dormouse_core::routing::routing(&model, false);
    let g = r
        .check(&model)
        .unwrap_or_else(|e| panic!("{name}: routing: {e}"));
    assert_eq!(
        g.muon,
        4 * cfg.n_experts + 3,
        "{name}: Muon+ group must follow n_experts"
    );
    assert_eq!(
        g.qk, 2,
        "{name}: the head-wise Q/K group is the KDA q and k"
    );
    assert_eq!(g.tables, 1, "{name}: the n-gram table is one parameter");
    assert_eq!(
        g.muon + g.qk + g.tables + g.rest,
        c.groups,
        "{name}: the groups must cover every parameter"
    );
    c
}

/// The presets whose instantiation is affordable in a default CPU test run.
/// `mor` is here for a reason worth stating: it is `small`'s geometry field
/// for field, so it costs the same, and `mor_costs_what_small_costs` below
/// proves that from the configs instead of from a five-minute build.
///
/// The rest are in the slow test: a full-width build on the pure-Rust CPU
/// backend costs MINUTES (measured: 4-11 min for the presets here), and a
/// suite nobody runs is a suite nobody reads.
const CHEAP_PRESETS: &[&str] = &[
    "base",
    "byteflow",
    "byteflow_kda",
    "mor",
    "nano",
    "nano-fused",
    "small",
];

#[test]
fn every_preset_states_its_cost_and_no_preset_is_a_lookup_table() {
    for name in CHEAP_PRESETS {
        cost_of_preset(name);
    }
    // Nothing ships that is on neither list, or a new preset would silently
    // escape the cost gate.
    for name in shipped_presets() {
        assert!(
            CHEAP_PRESETS.contains(&name.as_str()) || WIDE_PRESETS.contains(&name.as_str()),
            "{name} is in neither the cheap nor the wide cost list - cost it, or the preset's price is invisible"
        );
    }
    mor_differs_from_small_in_the_three_mor_lines_only();
}

/// `mor` is the A/B control pair with `small`: the same geometry and the same
/// memory budget, so the two runs differ in the three MoR scalars and nothing
/// else. Asserted from the resolved configs rather than assumed - the file
/// shipped `engram_rows = 500_000` until 2026-09-27, which made the A/B
/// "MoR plus an 11x memory table" against "no MoR, 25_000 rows" and cost
/// 50.3M memory rows against a 7.05M backbone (84% of the model, the exact
/// monopoly shape every other preset was re-priced away from). If a future
/// edit gives `mor` its own width or budget, this fails and the cost goes
/// back to the slow test.
fn mor_differs_from_small_in_the_three_mor_lines_only() {
    let small = preset("small");
    let mut mor = preset("mor");
    assert!(
        !small.use_mor && mor.use_mor,
        "the A/B pair: small off, mor on"
    );
    mor.use_mor = small.use_mor;
    assert_eq!(small, mor, "configs/mor.toml must differ from configs/small.toml in use_mor, mor_k and mor_bce_weight ONLY");
}

/// The presets whose full width costs minutes of CPU to instantiate. Same
/// gate, same assertions, `#[ignore]`d by policy like the other slow CPU gates
/// in this repo: the assertion is here and
/// `cargo test -p dormouse-core --test preset_exec -- --ignored` prints the
/// numbers.
const WIDE_PRESETS: &[&str] = &["swift50", "one_b", "p150"];

#[test]
#[ignore = "slow: instantiating these widths on the CPU backend takes minutes each"]
fn the_wide_presets_cost_what_they_say_they_cost() {
    for name in WIDE_PRESETS {
        cost_of_preset(name);
    }
}

/// `1234567` -> `1_234_567`: Rust's format has no thousands grouping, and this
/// line is read by a person deciding which preset to run.
fn k(n: usize) -> String {
    let s = n.to_string();
    s.chars()
        .rev()
        .collect::<Vec<_>>()
        .chunks(3)
        .map(|c| c.iter().collect::<String>())
        .collect::<Vec<_>>()
        .join("_")
        .chars()
        .rev()
        .collect()
}

// --- the fields no preset sets: proven live, not assumed ------------------

/// Three config fields are in the schema, in no preset, and would be a false
/// promise in a TOML file if they did not work. Set them on `small` and
/// assert each one CHANGES what runs: `act_quant` counts and perturbs the
/// forward, `use_gr` swaps ReZero for the gated residual, `use_tsct = false`
/// replaces the spectral experts with dense ones and still routes every
/// parameter.
#[test]
fn fields_no_preset_ships_still_take_effect() {
    let dev = device();
    let base = preset("small");

    // act_quant: the counter moves and the logits change.
    let plain = fixture(&base);
    let model = DormouseModel::new(&plain, &dev);
    let bytes = batch_bytes(11, BATCH * SEQ);
    let x = input_ids(&bytes, &dev);
    let h = hashed_ids(&bytes, &dev);
    probe::reset();
    let l0 = model.forward::<B>(x.clone(), Some(h.clone()));
    assert_eq!(
        probe::count(probe::ACT_QUANT),
        0,
        "no preset ships act_quant; the default must be off"
    );
    let quant = DormouseConfig {
        act_quant: Some(dormouse_core::ActQuant::Int(8)),
        ..plain.clone()
    };
    let qmodel = DormouseModel::new(&quant, &dev);
    probe::reset();
    let l1 = qmodel.forward::<B>(x.clone(), Some(h.clone()));
    assert_eq!(
        probe::count(probe::ACT_QUANT),
        plain.max_iter as u64,
        "act_quant=8 must quantize every iteration's FFN input"
    );
    let d = (l0.clone() - l1).abs().max().into_scalar::<f32>();
    assert!(
        d > 1e-6,
        "act_quant reached no logits: max |dlogit| = {d:.3e}"
    );

    // use_gr: entered every iteration, and the readout differs.
    let gr = DormouseConfig {
        use_gr: true,
        ..plain.clone()
    };
    let gmodel = DormouseModel::new(&gr, &dev);
    probe::reset();
    let l2 = gmodel.forward::<B>(x, Some(h));
    assert_eq!(
        probe::count(probe::GR),
        plain.max_iter as u64,
        "use_gr must enter the gated residual every iteration"
    );
    let d = (l0 - l2).abs().max().into_scalar::<f32>();
    assert!(d > 1e-6, "use_gr changed no logits: max |dlogit| = {d:.3e}");

    // use_attnres: replaces the residual accumulation. Entered once per
    // iteration (the aggregation is per-iteration, one query per slot), and
    // the readout must MOVE - an AttnRes that ran and changed nothing would be
    // the same class of defect as the GR arm was in 9b343d3.
    let ar = DormouseConfig {
        use_attnres: true,
        ..plain.clone()
    };
    let amodel = DormouseModel::new(&ar, &dev);
    let bytes = batch_bytes(11, BATCH * SEQ);
    let x = input_ids(&bytes, &dev);
    let h = hashed_ids(&bytes, &dev);
    probe::reset();
    let l0 = amodel.forward::<B>(x.clone(), Some(h.clone()));
    assert_eq!(
        probe::count(probe::ATTNRES),
        plain.max_iter as u64,
        "use_attnres must aggregate every iteration"
    );
    let p0 = DormouseModel::new(&plain, &dev);
    let l1 = p0.forward::<B>(x, Some(h));
    let d = (l0 - l1).abs().max().into_scalar::<f32>();
    assert!(
        d > 1e-6,
        "use_attnres changed no logits: max |dlogit| = {d:.3e}"
    );

    // use_tsct = false: dense experts, no TSCT factors anywhere, and every
    // parameter still declared to exactly one group.
    let dense = DormouseConfig {
        use_tsct: false,
        ..plain.clone()
    };
    let dmodel = DormouseModel::new(&dense, &dev);
    let r = dormouse_core::routing::routing(&dmodel, false);
    let g = r
        .check(&dmodel)
        .expect("the dense arm must be fully declared too");
    assert!(
        g.rest > 0,
        "the dense arm must route the expert weights to the fallback"
    );
    let spectral = dormouse_core::routing::routing(&model, false)
        .check(&model)
        .expect("declared");
    assert!(
        g.muon < spectral.muon,
        "use_tsct=false left {} params on Muon+ (spectral: {}) - the expert factors are the point of the group",
        g.muon,
        spectral.muon
    );
    assert_eq!(
        g.qk, 2,
        "the head-wise Q/K group is the KDA q and k either way"
    );
}

// --- the CUDA half: the fused gate, at the model --------------------------

/// THE YEAR OF SILENCE, on the only backend where it happened.
///
/// A CPU test cannot see this: the fused gated-delta gate is a CUDA-only
/// phenomenon, and the tensor-ops fallback computes the same function, so the
/// run is correct and only a launch counter distinguishes "ran" from "fell
/// back". `fused_seam_counts` is the library's own counter pair; this asserts
/// that a forward through a SHIPPED preset's attention arm moves it.
///
/// `#[ignore]`d by policy: this repo does not put a GPU in CI. It is here so
/// the assertion exists, and so anyone with a box runs it before believing a
/// training log's `fused kda=` line.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU: cargo test -p dormouse-core --features cuda --test preset_exec -- --ignored"]
fn on_cuda_a_preset_attention_arm_actually_reaches_the_fused_kernels() {
    let dev = Device::cuda(0).autodiff();
    let cfg = preset("small");
    assert!(
        cfg.use_kda,
        "the preset under test must declare the KDA arm"
    );
    let fx = fixture(&cfg);
    let model = DormouseModel::new(&fx, &dev);
    let bytes = batch_bytes(3, BATCH * SEQ);
    let x: Tensor<2, Int> = {
        let v: Vec<i64> = bytes.iter().map(|&b| b as i64).collect();
        Tensor::from_data(TensorData::new(v, [BATCH, SEQ]), &dev)
    };
    dormouse_core::probe::reset();
    let before = dormouse_core::fused_seam_counts();
    let logits = model.forward::<B>(x, None);
    assert_eq!(logits.dims(), [BATCH, SEQ, fx.vocab]);
    let after = dormouse_core::fused_seam_counts();
    assert_eq!(
        probe::count(probe::KDA),
        fx.max_iter as u64,
        "the KDA arm was declared ON and never entered"
    );
    let (kda_f, kda_b, norm_asked, norm_skipped) = after;
    assert!(
        kda_f > before.0,
        "the KDA arm RAN ({:>4} entries) and not one fused kernel launched: the dispatch gate is \
         closed and the run is silently on the tensor-ops chunk path. counts: {:?} -> {:?}",
        probe::count(probe::KDA),
        before,
        after
    );
    println!(
        "fused kda fwd/bwd {kda_f}/{kda_b}; fused rmsnorm engaged {}/{} ({})",
        norm_asked.saturating_sub(norm_skipped),
        norm_asked,
        if norm_asked == norm_skipped {
            "never engaged"
        } else {
            "engaged"
        }
    );
    assert!(
        norm_skipped <= norm_asked,
        "the RMSNorm seam counter is inconsistent: {norm_asked}/{norm_skipped}"
    );
}

// --- fixtures ------------------------------------------------------------

/// Fixed-seed LCG (Knuth's MMIX constants) - no RNG dependency.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 33) as u8
    }
}

fn batch_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut rng = Lcg(seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
    (0..n).map(|_| rng.byte()).collect()
}

fn input_ids(bytes: &[u8], dev: &Device) -> Tensor<2, Int> {
    let v: Vec<i64> = bytes.iter().map(|&x| x as i64).collect();
    Tensor::from_data(TensorData::new(v, [BATCH, SEQ]), dev)
}

/// Next-byte targets per row (the train loop's shift-by-one).
fn targets(bytes: &[u8], dev: &Device) -> Tensor<2, Int> {
    let mut v = Vec::with_capacity(BATCH * SEQ);
    for r in 0..BATCH {
        let row = &bytes[r * SEQ..(r + 1) * SEQ];
        v.extend(row.iter().skip(1).map(|&x| x as i64));
        v.push(row[0] as i64);
    }
    Tensor::from_data(TensorData::new(v, [BATCH, SEQ]), dev)
}

/// FNV-hashed n-gram ids `[b, t, 3]`, RAW (not reduced) - the model masks the
/// slot index against its own table size (`hash & mask`), so the model config
/// is the only copy of the capacity.
///
/// The hash is truncated to the backend's `Int` width on the way in. On
/// burn-flex (and on cuda) `Tensor<D, Int>` is i32, and the data crate's
/// `hashes_raw` emits `fnv as u32` - a value that does NOT fit. Truncating to
/// i32 selects the SAME row: the slot mask is a power of two below 2^31
/// (`engram_tables` refuses anything larger), so the low 31 bits carry all of
/// it. See the report: the trainer's own `bytes_to_tensors` feeds the
/// untruncated u32 and panics in `Tensor::from_data` on the same conversion.
fn hashed_ids(bytes: &[u8], dev: &Device) -> Tensor<3, Int> {
    let mut v = Vec::with_capacity(BATCH * SEQ * 3);
    for r in 0..BATCH {
        let row = &bytes[r * SEQ..(r + 1) * SEQ];
        for p in 0..SEQ {
            let e = p + 1;
            for &n in [2usize, 3, 4].iter() {
                v.push((fnv_hash(&row[e.saturating_sub(n)..e]) as u32) as i32 as i64);
            }
        }
    }
    Tensor::from_data(TensorData::new(v, [BATCH, SEQ, 3]), dev)
}
