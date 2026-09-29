//! Seam tests for `DormouseModel` on the CPU (NdArray) backend: the public
//! forward/loss/aux interface the train loop and CLI build on, pinned
//! without CUDA hardware. fp32 only - the CPU backend has no bf16.
//!
//! Sizing: the always-on tests run a literal mini config cut from the nano
//! preset (same max_iter / n_experts / vocab / arms / aux weights, narrower
//! widths) because full nano costs tens of minutes per test in the default
//! dev-profile test build on this backend (burn-ndarray unoptimized; the
//! train crate's own roundtrip test shrank its config for the same reason).
//! The full-nano seam is covered by the `#[ignore]`d long-sequence test.
//!
//! Default run: `cargo test -p dormouse-core --test model_seam`
//! Slow tests:  `cargo test -p dormouse-core --test model_seam -- --ignored`

use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn::backend::autodiff::Autodiff;
use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::{Device, Int, Tensor, TensorData};
use dormouse_core::aux::{ema_update, TEACHER_MOMENTUM};
use dormouse_core::{fnv_hash, DormouseConfig, DormouseModel};

/// Same alias the train crate uses for `--features cpu`: proven to satisfy
/// the `DispatchKindConversion` bounds `forward_with_hidden` carries.
type B = Autodiff<burn::backend::Flex, BalancedCheckpointing>;

fn device() -> Device {
    Device::flex().autodiff()
}

/// Nano's shape at a debug-build-friendly width: everything that defines
/// the seam (loop iterations, expert count, aux weights, KDA+Engram
/// arms, vocab 256) stays nano; only the widths shrink. use_gr stays false
/// and bf16 stays off (CPU is fp32-only anyway).
fn nano_cfg() -> DormouseConfig {
    dormouse_core::config::load_config(concat!(env!("CARGO_MANIFEST_DIR"), "/../../configs/nano.toml"))
        .expect("configs/nano.toml loads")
}

fn mini_nano() -> DormouseConfig {
    DormouseConfig {
        d_model: 128,
        n_heads: 4,
        head_dim: 32,
        d_ffn: 256,
        rank: 32,
        // Capacity is not what this seam tests, and 500_000 rows is 50M
        // in-model params per fixture. The shipped budget is
        // `DormouseConfig::default().engram_rows`, pinned in loop_block's
        // `capacity_budget_is_the_measured_optimum`.
        engram_rows: 4096,
        // DSpark is OFF in every shipped preset as of 2026-09-29 (DeepSeek's
        // own MTP ablation reports it bits-per-byte neutral, and our protocol
        // measures BPB). These tests check that the aux HEADS exist, work and
        // carry a gradient - not that the default turns them on. So the head
        // is switched on here explicitly; otherwise three tests silently stop
        // testing it and pass for the wrong reason.
        dspark_weight: 0.1,
        ..nano_cfg()
    }
}

/// Fixed-seed LCG (Knuth's MMIX constants) - no RNG dependency.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
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
    let mut rng = Lcg::new(seed);
    (0..n).map(|_| rng.byte()).collect()
}

fn input_ids(bytes: &[u8], b: usize, t: usize, dev: &Device) -> Tensor<2, Int> {
    let v: Vec<i64> = bytes.iter().map(|&x| x as i64).collect();
    Tensor::from_data(TensorData::new(v, [b, t]), dev)
}

/// Next-byte targets per row (the train loop's shift-by-one).
fn targets(bytes: &[u8], b: usize, t: usize, dev: &Device) -> Tensor<2, Int> {
    let mut v = Vec::with_capacity(b * t);
    for r in 0..b {
        let row = &bytes[r * t..(r + 1) * t];
        v.extend(row.iter().skip(1).map(|&x| x as i64));
        v.push(row[0] as i64);
    }
    Tensor::from_data(TensorData::new(v, [b, t]), dev)
}

/// FNV-hashed n-gram ids `[b, t, 3]`, RAW (not reduced) exactly as
/// `dormouse_data::hashes_raw` emits them for the in-VRAM path: the model
/// masks the slot index against its own table size, so the model config is
/// the only copy of the capacity. Windows are the shipped `ORDERS` 2/3/4.
fn hashed_ids(bytes: &[u8], b: usize, t: usize, dev: &Device) -> Tensor<3, Int> {
    let mut v = Vec::with_capacity(b * t * 3);
    for r in 0..b {
        let row = &bytes[r * t..(r + 1) * t];
        for p in 0..t {
            let e = p + 1;
            for &n in [2usize, 3, 4].iter() {
                // `& 0x7fff_ffff`, exactly as `dormouse_data::raw_keys` does it.
                // The tensor is `Int` = i32 on this backend; an unmasked u32
                // above `i32::MAX` panics on Flex ("Element cannot be
                // represented in the target type") while CUDA wraps silently.
                // The model masks the key against its own table size, so the
                // high bit was never load-bearing.
                v.push(((fnv_hash(&row[e.saturating_sub(n)..e]) as u32) & 0x7fff_ffff) as i64);
            }
        }
    }
    Tensor::from_data(TensorData::new(v, [b, t, 3]), dev)
}

fn assert_all_finite<const D: usize>(name: &str, t: &Tensor<D>) {
    let v: Vec<f32> = t
        .clone()
        .into_data()
        .try_to_vec()
        .unwrap_or_else(|_| panic!("{name}: unreadable data"));
    let bad = v.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(bad, 0, "{name}: {bad} non-finite of {} values", v.len());
}

/// Pre-clip L2 norm of all parameter gradients (one scalar read).
fn grad_norm(model: &DormouseModel, grads: &burn::tensor::Gradients) -> f32 {
    struct NormVisitor<'a> {
        grads: &'a burn::tensor::Gradients,
        acc: Option<Tensor<1>>,
    }
    impl ModuleVisitor for NormVisitor<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            if let Some(g) = param.grad(self.grads) {
                let sq = g.powf_scalar(2.0).sum();
                self.acc = Some(match self.acc.take() {
                    Some(a) => a + sq,
                    None => sq,
                });
            }
        }
    }
    let mut v = NormVisitor { grads, acc: None };
    model.visit(&mut v);
    v.acc
        .map(|t| t.sqrt().into_scalar::<f32>())
        .unwrap_or(0.0)
}

/// Mini-nano, batch 2 s128 random bytes: the full forward seam runs, every
/// returned tensor has the documented shape, and loss/aux are finite.
#[test]
fn forward_smoke() {
    let t0 = std::time::Instant::now();
    let dev = device();
    let cfg = mini_nano();
    let model = DormouseModel::new(&cfg, &dev);
    let (b, s) = (2, 128);
    let bytes = batch_bytes(0xA11CE, b * s);
    let x = input_ids(&bytes, b, s, &dev);
    let h = hashed_ids(&bytes, b, s, &dev);
    let y = targets(&bytes, b, s, &dev);

    // No teacher: the JEPA term drops (needs the EMA copy) but DSpark keeps
    // `aux` alive at nano's dspark_weight = 0.1.
    let (logits, rec, kda, aux) =
        model.forward_with_hidden::<B>(x, Some(h), None, Some(y), None);

    assert_eq!(logits.dims(), [b, s, cfg.vocab]);
    assert_eq!(rec.dims(), [1]);
    assert_eq!(kda.dims().len(), 4, "kda state must be a Tensor<4>");
    assert_eq!(kda.dims()[0], b, "kda state batch dim");
    assert_all_finite("logits", &logits);
    assert_all_finite("rec", &rec);
    assert_all_finite("kda", &kda);
    let aux = aux.expect("aux must be Some (dspark_weight > 0, targets given)");
    assert_all_finite("aux", &aux);
    let loss = model.loss::<B>(rec);
    assert_all_finite("loss", &loss);
    println!(
        "forward_smoke ok: loss={:.4} aux={:.4} ({} ms)",
        loss.into_scalar::<f32>(),
        aux.into_scalar::<f32>(),
        t0.elapsed().as_millis()
    );
}

/// Same model, same inputs, twice: the forward is bit-stable on the CPU
/// backend (fp32, no dropout, no quantization in this config).
#[test]
fn determinism() {
    let t0 = std::time::Instant::now();
    let dev = device();
    let cfg = mini_nano();
    let model = DormouseModel::new(&cfg, &dev);
    let (b, s) = (2, 128);
    let bytes = batch_bytes(0xBEEF, b * s);
    let x = input_ids(&bytes, b, s, &dev);
    let h = hashed_ids(&bytes, b, s, &dev);

    let l1 = model.forward::<B>(x.clone(), Some(h.clone()));
    let l2 = model.forward::<B>(x, Some(h));
    assert_eq!(l1.dims(), l2.dims());
    let d = (l1 - l2).abs().max().into_scalar::<f32>();
    println!("determinism: max |dlogit| = {d:.3e}");
    assert!(d < 1e-6, "two identical forwards diverged: {d:.3e}");
    println!("determinism ok ({} ms)", t0.elapsed().as_millis());
}

/// TWO MODELS, ONE SEED. The `determinism` test above builds one model and
/// runs it twice, so it cannot see an init that draws fresh entropy - which
/// is exactly the defect it was cited as covering.
///
/// The claim this pins is `device.seed(s)` -> the same weights, which is what
/// ADR-0002's "3 seeds per arm" needs and what `4b42b6d` asserted when it
/// added the `device.seed(cfg.seed)` call. That commit shipped the call and a
/// comment; it shipped no test, so "the seed governs the init" has been prose
/// since. The number usually quoted for what is left - 409 043 differing
/// values, ~4% of the model - appears ONLY in docs/AB-PROTOCOL.md:93 and
/// docs/PLAN-2026-09-29.md:198. It was never measured in a test, and a reader
/// has no way to check it. This test is the measurement, and it runs on CPU in
/// seconds.
///
/// It also pins the OTHER half, which is what makes a seed a run: two
/// different seeds must NOT give the same model, or the knob does nothing.
#[test]
fn two_models_one_seed_are_bit_identical() {
    let cfg = mini_nano();
    let (b, s) = (2, 128);
    let bytes = batch_bytes(0xC0FFEE, b * s);
    let x = input_ids(&bytes, b, s, &device());
    let h = hashed_ids(&bytes, b, s, &device());

    let fwd = |d: &Device| {
        d.seed(7u64);
        let m = DormouseModel::new(&cfg, d);
        // A forward touches every arm's parameters, so a difference anywhere
        // shows up in the logits rather than needing per-param plumbing.
        m.forward::<B>(x.clone(), Some(h.clone())).into_data()
    };
    let (a, c) = (fwd(&device()), fwd(&device()));

    let diff = a
        .bytes
        .iter()
        .zip(c.bytes.iter())
        .filter(|(x, y)| x != y)
        .count();
    // Report the VALUE count too, since the standing claim is phrased in
    // values: differing bytes are not differing values, and conflating them
    // is how a 4% figure and a 0.9% figure end up describing one number.
    let vals = (0..a.bytes.len() / 4)
        .filter(|i| a.bytes[i * 4..i * 4 + 4] != c.bytes[i * 4..i * 4 + 4])
        .count();
    assert_eq!(
        diff, 0,
        "same seed, two builds: {diff} differing bytes / {vals} differing f32 values \
         out of {} - init is not a pure function of the seed",
        a.bytes.len() / 4
    );

    // A seed that changes nothing is not a seed. Cheap, and it stops a
    // "fix" that hardcodes every initializer from passing this test.
    let d2 = device();
    d2.seed(8u64);
    let other = DormouseModel::new(&cfg, &d2)
        .forward::<B>(x, Some(h))
        .into_data();
    assert_ne!(
        a.bytes, other.bytes,
        "seed 7 and seed 8 produced the same model - the seed does nothing"
    );
}

/// Full nano preset, single sequence s8192, forward AND backward through
/// the KDA recurrence: no NaN/Inf in any returned tensor or in the gradient
/// norm. This is the seam the 16 GB box cannot probe cheaply on CUDA.
#[test]
#[ignore = "slow: full nano at s8192, fp32 fwd+bwd on CPU (~minutes)"]
fn kda_long_stability() {
    let t0 = std::time::Instant::now();
    let dev = device();
    let mut cfg = nano_cfg();
    cfg.max_seq_len = 8192; // nano ships 512; raise to cover the sequence
    let model = DormouseModel::new(&cfg, &dev);
    let (b, s) = (1, 8192);
    let bytes = batch_bytes(0x1D3A, b * s);
    let x = input_ids(&bytes, b, s, &dev);
    let h = hashed_ids(&bytes, b, s, &dev);
    let y = targets(&bytes, b, s, &dev);

    let (logits, rec, kda, aux) =
        model.forward_with_hidden::<B>(x, Some(h), None, Some(y), None);
    assert_all_finite("logits", &logits);
    assert_all_finite("rec", &rec);
    assert_all_finite("kda", &kda);
    let mut loss = model.loss::<B>(rec);
    if let Some(a) = aux {
        assert_all_finite("aux", &a);
        loss = loss + a;
    }
    assert_all_finite("loss", &loss);
    let grads = loss.backward();
    let gn = grad_norm(&model, &grads);
    println!("kda_long_stability: grad norm = {gn:.4}");
    assert!(
        gn.is_finite() && gn > 0.0,
        "gradient norm after s8192 fwd+bwd: {gn}"
    );
    println!("kda_long_stability ok ({} ms)", t0.elapsed().as_millis());
}

/// Backward through the full loss (CE + JEPA + DSpark): every named float
/// parameter on a live grad path receives a finite gradient; the controller
/// must receive a gradient (routing stays connected).
///
/// Two documented exceptions - anything else missing fails the test:
/// - the two RMSNorm gains (`loop_block.norm.weight`, `norm.weight`):
///   burn-rmsnorm builds them with `Param::initialized(.., Tensor::ones)`
///   instead of `Param::from_tensor`, so the param inherits the tensor's
///   `require_grad = false` and can never receive a gradient. That is an
///   upstream bug this test pins: gains stay at 1.0 forever.
///
/// The zero-norm count is informational: at init ReZero's residual scale is
/// 0, so the block body (attention / Engram / experts) sits on the graph but
/// its gradients are exactly zero - they switch on once the scale trains off
/// zero (see `engram_host_rows`).
#[test]
fn gradient_flow() {
    let t0 = std::time::Instant::now();
    let dev = device();
    let cfg = mini_nano();
    let mut model = DormouseModel::new(&cfg, &dev);
    // EMA teacher at momentum 0 = exact copy (train loop's init), so the
    // JEPA term is on and jepa_pred sits on the live grad path.
    let teacher = ema_update(model.clone(), &model, 0.0);
    let (b, s) = (2, 128);
    let bytes = batch_bytes(0x6BAD, b * s);
    let x = input_ids(&bytes, b, s, &dev);
    let h = hashed_ids(&bytes, b, s, &dev);
    let y = targets(&bytes, b, s, &dev);

    let (_logits, rec, _kda, aux) = model.forward_with_hidden::<B>(
        x.clone(),
        Some(h.clone()),
        None,
        Some(y.clone()),
        Some(&teacher),
    );
    let mut loss = model.loss::<B>(rec);
    if let Some(a) = aux {
        loss = loss + a;
    }
    let grads = loss.backward();

    struct Probe<'a> {
        stack: Vec<String>,
        grads: &'a burn::tensor::Gradients,
        rows: Vec<(String, bool, f32)>,
    }
    impl ModuleVisitor for Probe<'_> {
        fn enter_module(&mut self, name: &str, _container: &str) {
            self.stack.push(name.to_string());
        }
        fn exit_module(&mut self, _name: &str, _container: &str) {
            self.stack.pop();
        }
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let path = self.stack.join(".");
            match param.grad(self.grads) {
                None => self.rows.push((path, false, f32::NAN)),
                Some(g) => {
                    let sq = g.powf_scalar(2.0).sum().into_scalar::<f32>();
                    self.rows.push((path, true, sq.sqrt()));
                }
            }
        }
    }
    let mut probe = Probe {
        stack: Vec::new(),
        grads: &grads,
        rows: Vec::new(),
    };
    model.visit(&mut probe);
    assert!(
        !probe.rows.is_empty(),
        "model reports no float parameters"
    );
    let missing: Vec<&str> = probe
        .rows
        .iter()
        .filter(|(_, has, _)| !has)
        .map(|(p, _, _)| p.as_str())
        .collect();
    // Documented grad-free params (see the test doc): the RMSNorm gains
    // (burn-rmsnorm require_grad bug). Anything else missing is a regression.
    //
    // The MoR router joins them by DESIGN, not by defect: `LoopBlock.mor_router`
    // is documented as "always present (769 params, routed to AdamW...);
    // `use_mor` decides whether it is read" (loop_block.rs:100-103). With
    // `use_mor = false` the forward never reads it, so it cannot carry a
    // gradient. 769 of 9.2M params, and constructing it conditionally would
    // break every checkpoint that has one.
    const KNOWN_GRAD_FREE: &[&str] = &[
        "loop_block.norm.weight",
        "norm.weight",
        "loop_block.mor_router.proj.weight",
        "loop_block.mor_router.proj.bias",
    ];
    let unexpected: Vec<&str> = missing
        .iter()
        .filter(|p| !KNOWN_GRAD_FREE.contains(p))
        .map(|p| *p)
        .collect();
    assert!(
        unexpected.is_empty(),
        "{} params without gradient: {unexpected:#?}",
        unexpected.len()
    );
    println!(
        "gradient_flow: {} params grad-free (2 RMSNorm gains: burn-rmsnorm require_grad bug)",
        missing.len()
    );
    let nonfinite: Vec<&str> = probe
        .rows
        .iter()
        .filter(|(p, has, n)| *has && !n.is_finite() && !KNOWN_GRAD_FREE.contains(&p.as_str()))
        .map(|(p, _, _)| p.as_str())
        .collect();
    assert!(
        nonfinite.is_empty(),
        "{} params with non-finite grads: {nonfinite:#?}",
        nonfinite.len()
    );
    // Every parameter that HAS a gradient must have a NON-ZERO one. This is
    // the arbiter that should have caught the real bug: with the ReZero scale
    // initialized to 0 (its value until 2026-09-27) dL/dy = 0, so the KDA arm,
    // every expert FFN and both controller gates started with no gradient at
    // all and the model was a linear map of the byte embedding. The old test
    // bumped the scale by hand and only asserted on the controller; the
    // initialization is now 1.0 (identity, as ReZero specifies) and nothing
    // needs the workaround.
    // The JEPA predictor is exempt BY CONSTRUCTION, not by convenience: its
    // loss is an L1 over a RANDOM span mask (2% start rate, 128 positions), and
    // when no start lands the mask is empty, the masked loss is identically
    // zero and the head's gradient is exactly zero. That is a property of the
    // objective, not starvation - and it bites ~7.6% of steps, so the old
    // catch-all assert flaked.
    const RANDOM_MASK_EXEMPT: &[&str] = &[
        "aux.jepa_pred.proj.weight",
        "aux.jepa_pred.norm.gamma",
        "aux.jepa_pred.norm.beta",
    ];
    let starved: Vec<&str> = probe
        .rows
        .iter()
        .filter(|(p, has, n)| *has && n.abs() < 1e-12 && !RANDOM_MASK_EXEMPT.contains(&p.as_str()))
        .map(|(p, _, _)| p.as_str())
        .collect();
    assert!(
        starved.is_empty(),
        "{} parameters have a zero gradient at init (starved, not just small): {starved:#?}",
        starved.len()
    );
    let zeros: Vec<(&str, f32)> = probe
        .rows
        .iter()
        .filter(|(_, _, n)| *n < 1e-12)
        .map(|(p, _, n)| (p.as_str(), *n))
        .collect();
    println!(
        "gradient_flow: {} params, {} zero-norm grads",
        probe.rows.len(),
        zeros.len()
    );
    for (p, n) in &zeros {
        println!("  zero-grad-norm: {p} ({n:.2e})");
    }
    // The controller's gates multiply the arms, so it is the last thing to
    // light up; a second backward with the scale forced to 1 asserts the
    // wiring independently of whatever the init happens to be.
    let head_norm = |rows: &[(String, bool, f32)], want: &str| {
        rows.iter()
            .find(|(p, _, _)| p == want)
            .unwrap_or_else(|| panic!("{want} not found in module tree"))
            .2
    };
    let ctrl0 = head_norm(&probe.rows, "loop_block.controller.weight");
    println!("gradient_flow at init: controller norm={ctrl0:.3e}");

    model.loop_block.residual_scale =
        burn::module::Param::from_tensor(Tensor::<1>::ones([1], &dev));
    let (_l, rec, _k, aux) =
        model.forward_with_hidden::<B>(x, Some(h), None, Some(y), Some(&teacher));
    let mut loss = model.loss::<B>(rec);
    if let Some(a) = aux {
        loss = loss + a;
    }
    let grads1 = loss.backward();
    let mut probe1 = Probe {
        stack: Vec::new(),
        grads: &grads1,
        rows: Vec::new(),
    };
    model.visit(&mut probe1);
    let ctrl1 = head_norm(&probe1.rows, "loop_block.controller.weight");
    println!("gradient_flow at scale=1: controller norm={ctrl1:.3e}");
    assert!(
        ctrl1 > 0.0,
        "controller grad norm stays zero with residual scale on: {ctrl1}"
    );
    println!("gradient_flow ok ({} ms)", t0.elapsed().as_millis());
}

/// JEPA + DSpark aux seam: with both weights > 0 and an EMA teacher, aux is
/// finite, and `ema_update` advances the teacher without producing NaNs.
#[test]
fn aux_heads() {
    let t0 = std::time::Instant::now();
    let dev = device();
    let cfg = mini_nano();
    assert!(cfg.jepa_weight > 0.0 && cfg.dspark_weight > 0.0);
    let model = DormouseModel::new(&cfg, &dev);
    let mut teacher = ema_update(model.clone(), &model, 0.0);
    let (b, s) = (2, 128);
    let bytes = batch_bytes(0xA0C0, b * s);
    let x = input_ids(&bytes, b, s, &dev);
    let h = hashed_ids(&bytes, b, s, &dev);
    let y = targets(&bytes, b, s, &dev);

    let (_l1, _rec, _k, aux) = model.forward_with_hidden::<B>(
        x.clone(),
        Some(h.clone()),
        None,
        Some(y.clone()),
        Some(&teacher),
    );
    let aux = aux.expect("aux must be Some with a teacher and both weights > 0");
    assert_all_finite("aux", &aux);

    // Advance the EMA (the train loop does this after every optimizer step)
    // and confirm the teacher stays finite and usable as a JEPA target.
    teacher = ema_update(teacher, &model, TEACHER_MOMENTUM);
    let tw: Vec<f32> = teacher
        .embedding
        .weight
        .val()
        .into_data()
        .try_to_vec()
        .expect("teacher embedding readable");
    assert!(
        tw.iter().all(|v| v.is_finite()),
        "teacher params went non-finite after ema_update"
    );
    let (_l2, rec2, _k2, aux2) =
        model.forward_with_hidden::<B>(x, Some(h), None, Some(y), Some(&teacher));
    assert_all_finite("rec(2nd)", &rec2);
    let aux2 = aux2.expect("aux must still be Some after the EMA advance");
    assert_all_finite("aux(2nd)", &aux2);
    println!(
        "aux_heads ok: aux={:.4} aux_after_ema={:.4} ({} ms)",
        aux.into_scalar::<f32>(),
        aux2.into_scalar::<f32>(),
        t0.elapsed().as_millis()
    );
}

/// The RAM-offload seam: `host_rows` `[b, t, 96]` actually feeds the Engram
/// (the train loop passes hashed_ids = None when rows are pre-gathered), so
/// zero rows vs random rows must produce different logits.
///
/// ReZero's residual scale starts at 0, which keeps the whole block body
/// (attention / Engram / experts) out of the logits at init - the two
/// forwards would be bit-identical no matter what the rows contain. The
/// scale is set to 1 through its public field first (what one optimizer
/// step does in training) so the seam is observable.
#[test]
fn engram_host_rows() {
    let t0 = std::time::Instant::now();
    let dev = device();
    let cfg = mini_nano();
    assert!(cfg.use_engram, "nano must run the Engram arm");
    let mut model = DormouseModel::new(&cfg, &dev);
    model.loop_block.residual_scale =
        burn::module::Param::from_tensor(Tensor::<1>::ones([1], &dev));
    let (b, s) = (2, 128);
    // LoopBlock builds the Engram with row dim 32 x 3 tables.
    const ROW_DIM: usize = 3 * 32;
    let bytes = batch_bytes(0xE0, b * s);
    let x = input_ids(&bytes, b, s, &dev);

    let zeros = Tensor::<3>::zeros([b, s, ROW_DIM], &dev);
    let (lz, _rec, _k, _aux) =
        model.forward_with_hidden::<B>(x.clone(), None, Some(zeros), None, None);
    assert_all_finite("logits(zero rows)", &lz);

    let mut rng = Lcg::new(0x5EED);
    let rows: Vec<f32> = (0..b * s * ROW_DIM)
        .map(|_| (rng.next() % 2000) as f32 / 1000.0 - 1.0)
        .collect();
    let rnd = Tensor::<3>::from_data(TensorData::new(rows, [b, s, ROW_DIM]), &dev);
    let (lr, _rec, _k, _aux) =
        model.forward_with_hidden::<B>(x, None, Some(rnd), None, None);
    assert_all_finite("logits(random rows)", &lr);

    let d = (lz - lr).abs().max().into_scalar::<f32>();
    println!("engram_host_rows: max |dlogit| = {d:.3e}");
    assert!(
        d > 1e-5,
        "host_rows does not reach the logits: max |dlogit| = {d:.3e}"
    );
    println!("engram_host_rows ok ({} ms)", t0.elapsed().as_millis());
}

/// The backbone's gradient path THROUGH the memory branch. Two parameters
/// have to be on the live graph for the arm to be a memory and not a sink:
///
/// - `loop_block.engram.key_projs.0.weight` - the key projection, the one
///   backbone parameter on the addressing path (`3*engram_dim -> d_model`).
/// - `loop_block.mem_dense.weight` - the `(1 - lam)` half of the convex
///   mixture (FwPKM eq. 12's dense value path), which is what makes the
///   floor a gradient path and not just a coefficient.
///
/// Honest limit, stated here because the test cannot fix it: the SLOT INDEX
/// is a fixed FNV digest, so the key projection can only re-weight which of
/// the three fixed tables to trust - it cannot learn to address. Learned
/// addressing is product keys (1907.05242) / learned sub-keys, a different
/// mechanism and a separate decision; this pins the cheap route instead of
/// pretending the hash is trainable.
#[test]
fn engram_addressing_path_receives_gradient() {
    let dev = device();
    let mut cfg = mini_nano();
    // KDA off so the only non-FFN branch on the graph is the memory: a
    // nonzero grad here is unambiguously the memory's.
    cfg.use_kda = false;
    cfg.jepa_weight = 0.0;
    cfg.dspark_weight = 0.0;
    let model = DormouseModel::new(&cfg, &dev);
    let (b, s) = (2, 64);
    let bytes = batch_bytes(0xADD, b * s);
    let x = input_ids(&bytes, b, s, &dev);
    let h = hashed_ids(&bytes, b, s, &dev);
    let y = targets(&bytes, b, s, &dev);

    let (_logits, rec, _kda, aux) =
        model.forward_with_hidden::<B>(x, Some(h), None, Some(y), None);
    let mut loss = model.loss::<B>(rec);
    if let Some(a) = aux {
        loss = loss + a;
    }
    let grads = loss.backward();

    struct Probe<'a> {
        stack: Vec<String>,
        grads: &'a burn::tensor::Gradients,
        want: Vec<String>,
        found: Vec<(String, f32)>,
    }
    impl ModuleVisitor for Probe<'_> {
        fn enter_module(&mut self, name: &str, _c: &str) {
            self.stack.push(name.to_string());
        }
        fn exit_module(&mut self, _name: &str, _c: &str) {
            self.stack.pop();
        }
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let path = self.stack.join(".");
            if self.want.iter().any(|w| path == *w) {
                let g = param
                    .grad(self.grads)
                    .map(|g| g.powf_scalar(2.0).sum().into_scalar::<f32>())
                    .unwrap_or(0.0);
                self.found.push((path, g));
            }
        }
    }
    let want = vec![
        "loop_block.engram.key_projs.0.weight".to_string(),
        "loop_block.mem_dense.weight".to_string(),
    ];
    let mut p = Probe { stack: vec![], grads: &grads, want: want.clone(), found: vec![] };
    model.visit(&mut p);
    assert_eq!(p.found.len(), want.len(), "visitor missed a param: {:?} vs {want:?}", p.found.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>());
    for (name, sq) in p.found {
        assert!(
            sq > 1e-12,
            "{name} has no gradient through the memory branch (sum sq {sq:.3e}) - \
             the arm is a sink, not a memory"
        );
        println!("{name}: sum sq grad = {sq:.3e}");
    }
}
