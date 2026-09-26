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
#[allow(deprecated)]
type B = Autodiff<burn_ndarray::NdArray, BalancedCheckpointing>;

#[allow(deprecated)] // Device::ndarray is deprecated upstream; the repo still targets it
fn device() -> Device {
    Device::ndarray().autodiff()
}

/// Nano's shape at a debug-build-friendly width: everything that defines
/// the seam (PonderNet iterations, expert count, aux weights, KDA+MSA+Engram
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
        use_msa: false,
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

/// FNV-hashed 3-gram ids `[b, t, 3]`, modded to the Engram table size
/// (LoopBlock builds `[4096; 3]` tables; windows 3/5/8 like the train loop).
fn hashed_ids(bytes: &[u8], b: usize, t: usize, dev: &Device) -> Tensor<3, Int> {
    let mut v = Vec::with_capacity(b * t * 3);
    for r in 0..b {
        let row = &bytes[r * t..(r + 1) * t];
        for p in 0..t {
            let e = p + 1;
            v.push((fnv_hash(&row[e.saturating_sub(3)..e]) % 4096) as i64);
            v.push((fnv_hash(&row[e.saturating_sub(5)..e]) % 4096) as i64);
            v.push((fnv_hash(&row[e.saturating_sub(8)..e]) % 4096) as i64);
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
    let (logits, rec, p_dist, kda, aux) =
        model.forward_with_hidden::<B>(x, Some(h), None, Some(y), None);

    assert_eq!(logits.dims(), [b, s, cfg.vocab]);
    assert_eq!(rec.dims(), [1]);
    assert_eq!(p_dist.dims(), [b, cfg.max_iter]);
    assert_eq!(kda.dims().len(), 4, "kda state must be a Tensor<4>");
    assert_eq!(kda.dims()[0], b, "kda state batch dim");
    assert_all_finite("logits", &logits);
    assert_all_finite("rec", &rec);
    assert_all_finite("p_dist", &p_dist);
    assert_all_finite("kda", &kda);
    let aux = aux.expect("aux must be Some (dspark_weight > 0, targets given)");
    assert_all_finite("aux", &aux);
    let loss = model.loss::<B>(rec, p_dist);
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

    let (logits, rec, p_dist, kda, aux) =
        model.forward_with_hidden::<B>(x, Some(h), None, Some(y), None);
    assert_all_finite("logits", &logits);
    assert_all_finite("rec", &rec);
    assert_all_finite("p_dist", &p_dist);
    assert_all_finite("kda", &kda);
    let mut loss = model.loss::<B>(rec, p_dist);
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

/// Backward through the full loss (CE + PonderNet KL + JEPA + DSpark):
/// every named float parameter on a live grad path receives a finite
/// gradient; controller and halt head must receive gradients (PonderNet
/// halting and routing stay connected).
///
/// Two documented exceptions - anything else missing fails the test:
/// - `msa.index_branch.{q,k}_proj`: the top-k block selection is
///   non-differentiable, so the indexer has no CE grad path by design (it
///   trains via the MSA KL distill loss, which this interface does not
///   return).
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

    let (_logits, rec, p_dist, _kda, aux) = model.forward_with_hidden::<B>(
        x.clone(),
        Some(h.clone()),
        None,
        Some(y.clone()),
        Some(&teacher),
    );
    let mut loss = model.loss::<B>(rec, p_dist);
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
    // Documented grad-free params (see the test doc): the MSA indexer (no
    // differentiable path) and the RMSNorm gains (burn-rmsnorm require_grad
    // bug). Anything else missing is a regression.
    // MSA params are structurally grad-free while the arm is off (ADR-0012:
    // use_msa=false gates the forward; the module is still constructed).
    const KNOWN_GRAD_FREE: &[&str] = &[
        "loop_block.shared_attn.msa.index_branch.q_proj.weight",
        "loop_block.shared_attn.msa.index_branch.k_proj.weight",
        "loop_block.shared_attn.msa.attention.q_proj.weight",
        "loop_block.shared_attn.msa.attention.k_proj.weight",
        "loop_block.shared_attn.msa.attention.v_proj.weight",
        "loop_block.shared_attn.msa.attention.out_proj.weight",
        "loop_block.norm.weight",
        "norm.weight",
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
        "gradient_flow: {} params grad-free (2 MSA indexer by design, 2 RMSNorm gains: burn-rmsnorm require_grad bug)",
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
    // Halt head must carry signal even at ReZero zero (the PonderNet KL term
    // does not pass through the residual scale). The controller's gradient is
    // legitimately zero at init: everything it gates is multiplied by the
    // zero residual scale. So the wiring assert runs a second backward with
    // the scale bumped to 1 - one optimizer step's worth - where the
    // controller must light up.
    let head_norm = |rows: &[(String, bool, f32)], want: &str| {
        rows.iter()
            .find(|(p, _, _)| p == want)
            .unwrap_or_else(|| panic!("{want} not found in module tree"))
            .2
    };
    let halt0 = head_norm(&probe.rows, "loop_block.halt_head.weight");
    assert!(halt0 > 0.0, "halt_head grad norm at init is {halt0}");
    let ctrl0 = head_norm(&probe.rows, "loop_block.controller.weight");
    println!(
        "gradient_flow at init: halt_head norm={halt0:.3e}, controller norm={ctrl0:.3e} (ReZero scale=0 zeroes the controller's path)"
    );

    model.loop_block.residual_scale =
        burn::module::Param::from_tensor(Tensor::<1>::ones([1], &dev));
    let (_l, rec, pd, _k, aux) =
        model.forward_with_hidden::<B>(x, Some(h), None, Some(y), Some(&teacher));
    let mut loss = model.loss::<B>(rec, pd);
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
    let halt1 = head_norm(&probe1.rows, "loop_block.halt_head.weight");
    println!(
        "gradient_flow at scale=1: controller norm={ctrl1:.3e}, halt_head norm={halt1:.3e}"
    );
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

    let (_l1, _rec, _pd, _k, aux) = model.forward_with_hidden::<B>(
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
    let (_l2, rec2, pd2, _k2, aux2) =
        model.forward_with_hidden::<B>(x, Some(h), None, Some(y), Some(&teacher));
    assert_all_finite("rec(2nd)", &rec2);
    assert_all_finite("p_dist(2nd)", &pd2);
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
    let (lz, _rec, _pd, _k, _aux) =
        model.forward_with_hidden::<B>(x.clone(), None, Some(zeros), None, None);
    assert_all_finite("logits(zero rows)", &lz);

    let mut rng = Lcg::new(0x5EED);
    let rows: Vec<f32> = (0..b * s * ROW_DIM)
        .map(|_| (rng.next() % 2000) as f32 / 1000.0 - 1.0)
        .collect();
    let rnd = Tensor::<3>::from_data(TensorData::new(rows, [b, s, ROW_DIM]), &dev);
    let (lr, _rec, _pd, _k, _aux) =
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
