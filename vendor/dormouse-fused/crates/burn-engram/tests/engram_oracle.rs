//! Golden-vector tests against the OFFICIAL Engram reference.
//!
//! Oracle: `deepseek-ai/Engram@main`, `engram_demo_v1.py`, fetched and run by
//! `tests/oracle/gen_engram_oracle.py` (which vendors a copy of that file next
//! to itself). Nothing in this file restates a formula from the reference; every
//! expected number is a value the reference produced.
//!
//! # THE CLAIM, and it is deliberately not "bit-for-bit"
//!
//! `the_claim` prints it and checks itself. In short:
//!
//! * burn-engram agrees with the reference to `TOL_GATE` on the gate,
//!   `TOL_MODULE` on the full `EngramModule` forward, `TOL_CONV` on the
//!   depthwise conv, and EXACTLY (integer equality) on the prime table ladder.
//! * It does **not** agree on the gate's `clamp_min`/`add` token
//!   (`gate_is_on_the_clamp_min_side` is RED and says which way), on the RMS
//!   eps, on the hash VALUES (different RNG), or on the parameter inventory.
//!
//! Every tolerance is a sum of two MEASURED quantities:
//!
//! * `TOL_GATE` / `TOL_MODULE` / `TOL_CONV` - the disagreement observed between
//!   burn-on-ndarray and the reference-on-torch;
//! * the reference's OWN f32 answer's sensitivity, which the generator measures
//!   per row by re-running the reference's own formula with the 64-term dot
//!   product summed back to front (`gate.row.f32_error`). Same mathematics,
//!   different order, and at |s| = 1.5e-8 the answer moves by 5.0e-4. Any
//!   comparison against that row is a comparison against rounding, so those
//!   rows are excluded by measurement rather than by a tolerance wide enough to
//!   hide them.
//!
//! The fixture reaches the `clamp_min` vs `add` distinction because
//! `compute_gate`'s own reachable range forces it to: `s = <k, q>/sqrt(D)` with
//! `k, q` RMS-normalised, so `|s| <= sqrt(D) = 8` and the only region where
//! `|s| + 1e-6` and `max(|s|, 1e-6)` differ by more than a few ulp is just
//! above 1e-6. `the_claim` checks the fixture's smallest gap in that band
//! against the tolerance instead of trusting that.
//!
//! # What is deliberately NOT pinned
//!
//! The hash VALUES. The reference draws its odd multipliers from numpy PCG64
//! and burn-engram from splitmix64 (`hasher.rs:11-12`), so the two hashers
//! cannot agree bit-for-bit and the fixture does not pretend otherwise. What IS
//! pinned is the part that carries no RNG: the prime table ladder, which the
//! reference derives from `engram_vocab_size` alone.
//!
//! Two structural gaps are asserted as GAPS, not numerically (see
//! `reference_has_learnable_norm_gains_and_burn_has_none`): the reference's
//! `nn.RMSNorm` gains and `nn.Linear` biases have no burn-engram counterpart, so
//! the generator sets them to 1 and 0 and the numeric comparison stays exact.

#![allow(deprecated)] // Device::ndarray: the backend the fixtures were made on

use std::collections::HashMap;

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::{Device, Tensor, TensorData};

use burn_engram::hasher::NgramHasher;
use burn_engram::{compute_gate, depthwise_conv_1d, EngramModule};

const FIXTURE: &str = include_str!("fixtures/engram_oracle.txt");

/// Two-backend disagreement on a gate value, over the rows the reference can
/// itself resolve. MEASURED: the green gate test prints the observed maximum
/// (5.4e-6); this is 1e-5, about 2x.
const TOL_GATE: f64 = 1e-5;
/// `forward_embeds` vs the reference's `Engram.forward`: two 192-term fp32 dot
/// products per head plus a conv, so this is a sum bound, not a formula bound.
const TOL_MODULE: f64 = 2e-5;
/// `depthwise_conv_1d` vs the reference's `ShortConv.conv` at hc_mult = 1:
/// 4 taps, no dot product, so this is a rounding bound.
const TOL_CONV: f64 = 1e-5;
/// How much of a row's discriminating power the tolerances must leave unused.
/// Every "must match column X, not column Y" assertion demands the X gap
/// clear `DISCRIM_FACTOR * tol_row` and the Y gap fall inside `tol_row`;
/// `the_claim` checks the fixture's weakest band row clears the same bar, so
/// the two numbers cannot drift apart.
const DISCRIM_FACTOR: f64 = 2.0;
/// Below this |s| the reference's own f32 answer is a property of the dot
/// product's summation order, not of the input. Measured, not assumed: see
/// `the_reference_gate_is_sign_unstable_below_1e_minus_7`.
const RESOLVABLE_S: f64 = 1e-6;

fn dev() -> Device {
    Device::ndarray()
}

/// Parse the flat fixture: `key [chunk]: v v v ...`, `#` comments.
fn fixture() -> HashMap<String, Vec<f64>> {
    let mut out: HashMap<String, Vec<f64>> = HashMap::new();
    for line in FIXTURE.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, body) = line.split_once(':').expect("fixture line without ':'");
        // A chunk suffix ("key 3:") continues a key that is ALREADY in the map.
        // Testing for the base key first is what keeps a key whose NAME ends in a
        // number (meta.module_shape) from being read as a chunk of the same key
        // minus its last digit.
        let key = match key.rsplit_once(' ') {
            Some((k, n)) if n.parse::<usize>().is_ok() && out.contains_key(k) => k.to_string(),
            _ => key.to_string(),
        };
        let entry = out.entry(key).or_default();
        entry.extend(
            body.split_whitespace()
                .map(|v| v.parse::<f64>().unwrap_or_else(|_| panic!("bad number {v:?}"))),
        );
    }
    out
}

struct Fx {
    map: HashMap<String, Vec<f64>>,
}

impl Fx {
    fn load() -> Self {
        Self { map: fixture() }
    }
    fn get(&self, key: &str) -> &[f64] {
        self.map
            .get(key)
            .unwrap_or_else(|| panic!("fixture key {key:?} missing"))
    }
    fn one(&self, key: &str) -> f64 {
        let v = self.get(key);
        assert_eq!(v.len(), 1, "{key} should be a scalar, got {} values", v.len());
        v[0]
    }
    fn at(&self, key: &str, i: usize) -> f64 {
        self.get(key)[i]
    }
    fn ints(&self, key: &str) -> Vec<i64> {
        self.get(key).iter().map(|v| *v as i64).collect()
    }
    /// `[b, l, d]` from a flat list.
    fn t3<const D: usize>(&self, key: &str, dims: [usize; D]) -> Tensor<D> {
        let v = self.get(key);
        assert_eq!(
            v.len(),
            dims.iter().product::<usize>(),
            "{key}: fixture has {} values, shape {dims:?} wants {}",
            v.len(),
            dims.iter().product::<usize>()
        );
        Tensor::from_data(
            TensorData::new(v.iter().map(|x| *x as f32).collect::<Vec<_>>(), dims),
            &dev(),
        )
    }
    /// Tolerance for one gate row: the two-backend disagreement PLUS how far
    /// the REFERENCE'S OWN f32 answer moves when the 64-term dot is summed the
    /// other way round. Both terms are measured, per row.
    fn gate_tol(&self, i: usize) -> f64 {
        TOL_GATE + self.get("gate.row.f32_error")[i]
    }
    fn resolvable(&self, i: usize) -> bool {
        self.get("gate.row.s")[i].abs() >= RESOLVABLE_S
    }
    /// Is row `i` in the band where `clamp_min` and `add` are separable?
    fn in_band(&self, i: usize) -> bool {
        let (lo, hi) = (self.at("meta.gate_band", 0), self.at("meta.gate_band", 1));
        (lo..=hi).contains(&self.get("gate.row.s")[i].abs())
    }
    fn rows(&self) -> usize {
        self.get("gate.row.s").len()
    }
}

fn flat<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    // The in-repo readback pattern (lib.rs:275-278).
    t.into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn abs_diff(a: f32, b: f64) -> f64 {
    (a as f64 - b).abs()
}

fn max_abs_diff(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| abs_diff(*x, *y)).fold(0.0, f64::max)
}

fn our_gate(fx: &Fx) -> Vec<f32> {
    let d = fx.one("meta.gate_d") as usize;
    let rows = fx.rows();
    let key = fx.t3::<3>("gate.key", [1, rows, d]);
    let query = fx.t3::<3>("gate.query", [1, rows, d]);
    flat(compute_gate(key, query, d))
}

// ─── 1. The prime ladder: exact integers, no float, no RNG ────────────────

/// The one part of the hasher that carries no RNG.
/// `calculate_vocab_size_across_layers` (reference l.235-260) walks the next
/// primes above `engram_vocab_size`, and `NgramHasher::new` (hasher.rs:94-106)
/// walks the same walk, so for a single layer they must produce the SAME
/// integers. Not "close": equal.
#[test]
fn prime_ladder_matches_the_reference_exactly() {
    let fx = Fx::load();
    let mut checked = 0;
    for i in 0.. {
        let vkey = format!("ladder.{i}.vocab_size");
        if !fx.map.contains_key(&vkey) {
            break;
        }
        let vocab = fx.one(&vkey) as usize;
        let max_ngram = fx.one(&format!("ladder.{i}.max_ngram")) as usize;
        let n_head = fx.one(&format!("ladder.{i}.n_head")) as usize;
        let primes = fx.ints(&format!("ladder.{i}.primes"));

        let ours = NgramHasher::new(vocab, 2, max_ngram, n_head, 0, 0).table_sizes();
        assert_eq!(
            ours,
            primes.iter().map(|p| *p as usize).collect::<Vec<_>>(),
            "vocab {vocab}, max_ngram {max_ngram}, n_head {n_head}"
        );
        assert_eq!(ours.len(), primes.len());
        assert!(
            ours.iter().all(|p| *p > vocab),
            "every slot must be addressed by a prime ABOVE the vocabulary, which \
             is what a power-of-two table cannot be"
        );
        checked += 1;
    }
    assert!(checked >= 4, "only {checked} ladders checked; fixture looks truncated");
}

/// The reference shares ONE `seen_primes` set across every layer it hashes
/// (l.236, threaded through `calculate_vocab_size_across_layers`), so a second
/// layer's ladder skips the first layer's primes. `NgramHasher` has no
/// equivalent: each instance starts from an empty list (hasher.rs:95).
///
/// A CHARACTERISATION, not a bug report: it asserts the two DIFFER, so the
/// moment `NgramHasher` grows the shared set this goes red and the comment gets
/// rewritten. If it ever passes, the claim is stale.
#[test]
fn multi_layer_ladder_diverges_from_the_reference_and_this_is_recorded() {
    let fx = Fx::load();
    let vocab = fx.one("ladder.3.vocab_size") as usize;
    let max_ngram = fx.one("ladder.3.max_ngram") as usize;
    let n_head = fx.one("ladder.3.n_head") as usize;
    let layer0 = fx.ints("ladder.3.primes");
    let layer1 = fx.ints("ladder.3.layer_last_primes");
    assert_ne!(
        layer0, layer1,
        "the reference's second layer shares `seen_primes` with the first"
    );
    let ours = NgramHasher::new(vocab, 2, max_ngram, n_head, 0, 0).table_sizes();
    assert_eq!(ours, layer0.iter().map(|p| *p as usize).collect::<Vec<_>>());
}

// ─── 2. The gate ─────────────────────────────────────────────────────────

/// RED until the port is fixed, and it is supposed to be red. The reference is
/// `gate.abs().clamp_min(1e-6).sqrt() * gate.sign()` (engram_demo_v1.py:372);
/// burn-engram computes `dot.abs().add_scalar(1e-6).sqrt().mul(dot.sign())`
/// (lib.rs:234) - the `add` variant. The fixture's band rows sit in the only
/// region where the two differ by more than a few ulp, and there the difference
/// is 1e-4, a hundred times this tolerance.
///
/// The next test is the green half: everything else about the gate IS the
/// reference's, so the fix is one token and not a rewrite.
#[test]
fn gate_is_on_the_clamp_min_side() {
    let fx = Fx::load();
    let ours = our_gate(&fx);
    let clamp = fx.get("gate.out.gate_eps1e5");
    let add = fx.get("gate.out.gate_add_eps1e5");
    let mut band = 0usize;
    for i in 0..ours.len() {
        if !fx.in_band(i) {
            continue;
        }
        band += 1;
        let to_clamp = abs_diff(ours[i], clamp[i]);
        let to_add = abs_diff(ours[i], add[i]);
        eprintln!(
            "band row {i} (|s| = {:e}): ours {:.9} | reference clamp_min \
             {:.9} ({to_clamp:e} away) | reference add {:.9} ({to_add:e} away)",
            fx.get("gate.row.s")[i].abs(),
            ours[i],
            clamp[i],
            add[i]
        );
        assert!(
            to_clamp <= fx.gate_tol(i),
            "gate row {i} (|s| = {:e}): ours {} is {to_clamp:e} from the \
             reference's clamp_min gate and only {to_add:e} from its `add` gate, \
             i.e. lib.rs:234 uses add_scalar(1e-6) where engram_demo_v1.py:372 \
             uses clamp_min(1e-6). Tolerance {:e}.",
            fx.get("gate.row.s")[i].abs(),
            ours[i],
            fx.gate_tol(i)
        );
    }
    assert!(band >= 4, "only {band} band rows; the fixture no longer discriminates");
}

/// The green half of the same comparison: everything about the gate except that
/// one token is the reference's, to within the reference's own measured
/// sensitivity plus the two-backend disagreement.
///
/// This is what BOUNDS the fix, and since 2026-09-29 it is the fix's own gate.
///
/// It was written to compare us against the reference's `add` column, on the
/// stated expectation that switching lib.rs:234 to `clamp_min(1e-6)` would
/// leave it passing. It did not: the correct column is `gate.out.gate_eps1e5`
/// (the fixture's generator says so in as many words at
/// `oracle/gen_engram_oracle.py:254-255` - "gate_eps1e5_add is the column it
/// is WRONG to match and gate_eps1e5 is the one it is right to"). Comparing a
/// correct implementation to the column it must not match is a test that
/// pins the bug, so both this and `the_claim` now compare against `clamp_min`.
#[test]
fn the_gate_matches_the_reference_up_to_the_add_divergence() {
    let fx = Fx::load();
    let ours = our_gate(&fx);
    // The reference's OWN gate: `gate.abs().clamp_min(1e-6).sqrt() * sign`.
    let clamp = fx.get("gate.out.gate_eps1e5");
    // The formula we must NOT match, kept so the separation is measured rather
    // than assumed - if the two columns ever converged this test would be
    // vacuous, which is what DISCRIM_FACTOR in `the_claim` exists to catch.
    let add = fx.get("gate.out.gate_add_eps1e5");
    let mut checked = 0usize;
    let mut worst: (f64, usize) = (0.0, 0);
    for i in 0..ours.len() {
        if !fx.resolvable(i) {
            continue;
        }
        checked += 1;
        let diff = abs_diff(ours[i], clamp[i]);
        if diff > worst.0 {
            worst = (diff, i);
        }
        assert!(
            diff <= fx.gate_tol(i),
            "gate row {i} (|s| = {:e}): ours {} vs the reference's `clamp_min` \
             gate {:.9} = {diff:e}, tolerance {:e} = TOL_GATE + the reference's \
             own measured sensitivity {:e}. (The `add` variant there is {:.9}.)",
            fx.get("gate.row.s")[i].abs(),
            ours[i],
            clamp[i],
            fx.gate_tol(i),
            fx.get("gate.row.f32_error")[i],
            add[i]
        );
    }
    assert!(checked >= 14, "only {checked} resolvable rows; fixture looks truncated");
    let (worst_d, worst_i) = worst;
    eprintln!(
        "gate vs the reference's `clamp_min` gate, {checked} resolvable rows: max \
         {worst_d:e} at row {worst_i} (TOL_GATE {TOL_GATE:e} plus per-row sensitivity)"
    );
}

/// The reference's gate is NOT numerically stable at small |s|, which is why
/// the rows below `RESOLVABLE_S` are excluded by measurement rather than
/// covered by a tolerance wide enough to hide them.
///
/// `clamp_min(1e-6)` floors the radicand, so the gate is `sigmoid(+-1e-3)` all
/// the way down to s = 0 - it depends on the SIGN of a quantity that, for
/// |s| < 1e-7, is f32 cancellation noise in a 64-term dot product. The
/// generator measures this by re-running the reference's own f32 formula with
/// the dot summed back to front: the answers differ by up to 5.0e-4 at
/// |s| = 1.5e-8, straddling 0.5. That is a property of the published formula,
/// and no port can preserve it.
#[test]
fn the_reference_gate_is_sign_unstable_below_1e_minus_7() {
    let fx = Fx::load();
    let s = fx.get("gate.row.s");
    let err = fx.get("gate.row.f32_error");
    let gates = fx.get("gate.out.gate_eps1e5");
    let stable = err
        .iter()
        .zip(s)
        .filter(|(_, m)| m.abs() > 1e-5)
        .map(|(e, _)| *e)
        .fold(0.0f64, f64::max);
    let unstable = err
        .iter()
        .zip(s)
        .filter(|(_, m)| m.abs() < RESOLVABLE_S)
        .map(|(e, _)| *e)
        .fold(0.0f64, f64::max);
    eprintln!(
        "the reference's own f32 answer moves by {unstable:.2e} for |s| < \
         {RESOLVABLE_S:e} (dot summed back to front) and by {stable:.2e} for \
         |s| > 1e-5"
    );
    assert!(
        unstable > 1e-4,
        "no row below |s| = {RESOLVABLE_S:e} reproduces the sign instability; \
         this test's name is no longer supported by the fixture"
    );
    let flip = err.iter().position(|e| *e > 1e-4).expect("a row with the instability");
    let g = gates[flip];
    assert!((g - 0.5).abs() > 1e-5, "the unstable row should sit off 0.5; it is {g:.9}");
    eprintln!("unstable row {flip}: |s| {:e}, the reference's gate {g:.9}", s[flip].abs());
}

/// The RMS eps IS a divergence, and this sizes it instead of hiding it.
///
/// The reference's `nn.RMSNorm(hidden_size)` has `eps=None`, which torch
/// resolves to `torch.finfo(f32).eps` = 1.19e-7 - MEASURED by the generator
/// from the reference's own output, not assumed. burn-engram hardcodes 1e-5
/// (lib.rs:219, 225, 204). At |s| ~ 1e-6 that moves the gate by less than the
/// clamp/add signal; at |s| ~ 0.5 it moves it by 6.3e-5, five times MORE.
#[test]
fn the_rms_eps_divergence_is_real() {
    let fx = Fx::load();
    let shipped = fx.one("meta.rms_eps_shipped");
    assert!(
        (shipped - 1.1920929e-7).abs() < 1e-12,
        "the reference's resolved eps moved: {shipped:e}"
    );
    assert_eq!(fx.one("meta.rms_eps_burn"), 1e-5, "burn-engram's eps moved");
    let worst = max_abs_diff(
        &fx.get("gate.out.gate_eps1e5")
            .iter()
            .map(|v| *v as f32)
            .collect::<Vec<_>>(),
        fx.get("gate.out.gate_shipped"),
    );
    eprintln!(
        "the reference's own two eps values move the gate by up to {worst:e} \
         (1e-5 vs a measured 1.19e-7)"
    );
    assert!(
        worst > DISCRIM_FACTOR * TOL_GATE,
        "the eps difference moved the gate by only {worst:e}; the divergence is \
         no longer measurable at this tolerance"
    );
}

// ─── 3. The short conv ───────────────────────────────────────────────────

/// `depthwise_conv_1d` at hc_mult = 1, where the two implementations are the
/// same operator - up to the tap order, which the next test pins.
///
/// The reference's `nn.Conv1d` pairs `w[:, i]` with `x[t - (k-1-i)*dilation]`
/// (tap 0 is the most delayed); `depthwise_conv_1d` (lib.rs:96-101) pairs
/// `w[:, i]` with `x[t - i*dilation]` (tap 0 is the CURRENT sample). Both are
/// causal; they are time-reverses of one another. The weight is learnable, so
/// this is a reparameterisation rather than a correctness bug - but weight
/// transferred from the reference lands reversed, and the recorded init means
/// something different, so it is pinned here rather than left as a footnote.
#[test]
fn depthwise_conv_is_the_time_reverse_of_the_reference_kernel() {
    let fx = Fx::load();
    let (k, dil) = (
        fx.at("meta.conv_kernel", 0) as usize,
        fx.at("meta.conv_kernel", 1) as usize,
    );
    let x = fx.t3::<4>("conv.x_hc1", [2, 6, 1, 64]);
    let w = fx.t3::<2>("conv.w_hc1", [64, k]);
    let ours = flat(depthwise_conv_1d(x, &w, dil));

    let fwd = max_abs_diff(&ours, fx.get("conv.y_hc1"));
    let rev = max_abs_diff(&ours, fx.get("conv.y_hc1_reversed"));
    eprintln!(
        "conv hc1: vs the reference's kernel {fwd:e}, vs its time-reverse {rev:e} \
         (output scale 1.5)"
    );
    assert!(
        rev <= TOL_CONV,
        "depthwise_conv_1d should match the reference's conv with the taps \
         reversed ({rev:e})"
    );
    assert!(
        fwd > DISCRIM_FACTOR * TOL_CONV,
        "depthwise_conv_1d now matches the reference's tap order directly \
         ({fwd:e}); the reversal claim at lib.rs:96-101 is stale"
    );
}

// ─── 4. The whole module ─────────────────────────────────────────────────

/// Replaces float params in traversal order with fixture values. Traversal
/// order is `EngramModule`'s field order; the count is asserted below, so a
/// reordered or added field reddens this instead of silently misaligning it.
struct Loader {
    queue: std::collections::VecDeque<(String, Vec<f32>)>,
}

impl ModuleMapper for Loader {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (name, vals) = self
            .queue
            .pop_front()
            .expect("fixture has more params than EngramModule has tensors");
        let dims = param.val().dims();
        assert_eq!(
            vals.len(),
            dims.iter().product::<usize>(),
            "param {name}: fixture {} values vs shape {dims:?}",
            vals.len()
        );
        let n: usize = dims.iter().product();
        let t = Tensor::<1>::from_data(TensorData::new(vals, [n]), &dev()).reshape(dims);
        let (id, _old, mapper) = param.consume();
        Param::from_mapped_value(id, t, mapper)
    }
}

fn oracle_module(fx: &Fx) -> (EngramModule, Tensor<3>, Tensor<4>) {
    let hidden = fx.at("meta.module_shape", 0) as usize;
    let hc = fx.at("meta.module_shape", 1) as usize;
    let embed = fx.at("meta.module_shape", 2) as usize;
    let tables = fx.at("meta.module_shape", 3) as usize;
    let (k, dil) = (
        fx.at("meta.conv_kernel", 0) as usize,
        fx.at("meta.conv_kernel", 1) as usize,
    );
    assert_eq!(tables * embed, 192, "total_embed must match the fixture's Linears");

    let table_sizes: Vec<usize> = fx
        .ints("module.table_sizes")
        .iter()
        .map(|v| *v as usize)
        .collect();
    assert_eq!(table_sizes.len(), tables);
    let base = EngramModule::new(&table_sizes, embed, hidden, hc, &dev())
        .with_short_conv(k, dil, &dev());

    // Params are queued in `EngramModule`'s DECLARATION order, which is the
    // derive's traversal order: memory, key_projs[0..hc), value_proj,
    // conv_weight. `conv_weight_group0` is the [D, k] slice of the reference's
    // [hc*D, k] kernel - the shape burn-engram stores.
    let mut order: Vec<String> = vec!["memory.weight".into()];
    for i in 0..hc {
        order.push(format!("key_projs[{i}].weight"));
    }
    order.push("value_proj.weight".into());
    order.push("conv_weight_burn".into());
    for n in &order {
        assert!(
            fx.map.contains_key(&format!("module.param.{n}")),
            "fixture is missing module.param.{n}"
        );
    }
    let named: Vec<String> = fx
        .map
        .keys()
        .filter_map(|key| key.strip_prefix("module.param.").map(str::to_string))
        .filter(|n| !n.contains(' '))
        .collect();
    // `conv_weight` (the reference's full [hc*D, k] kernel) and
    // `conv_weight_group0` (its group-0 slice) are records, not loads - the
    // load is `conv_weight_burn`, the same kernel in burn's tap order. Nothing
    // else may be hiding.
    assert_eq!(
        named.len(),
        order.len() + 2,
        "the fixture has params this loader does not account for: {named:?}"
    );

    let mut queue: std::collections::VecDeque<(String, Vec<f32>)> =
        std::collections::VecDeque::new();
    for n in &order {
        queue.push_back((
            n.clone(),
            fx.get(&format!("module.param.{n}"))
                .iter()
                .map(|v| *v as f32)
                .collect(),
        ));
    }
    let module = base.map(&mut Loader { queue });

    let embeds = fx.t3::<3>("module.embeds", [2, 6, 192]);
    let hidden_states = fx.t3::<4>("module.hidden", [2, 6, hc, hidden]);
    (module, embeds, hidden_states)
}

/// The reference's `Engram.forward` on loaded weights and the SAME embeddings:
/// value_proj -> per-head gated fusion -> SiLU(depthwise conv) + residual.
///
/// The eps-matched column is the one to compare against, because burn-engram's
/// RMS eps is 1e-5 and the reference's is 1.19e-7; the shipped-eps column is
/// the same run with the reference's own eps, and the gap between the two
/// columns IS that divergence, measured at module level.
#[test]
fn forward_embeds_matches_the_reference() {
    let fx = Fx::load();
    let (module, embeds, hidden_states) = oracle_module(&fx);
    let ours = flat(module.forward_embeds(embeds, hidden_states));
    let diff = max_abs_diff(&ours, fx.get("module.out_eps1e5"));
    eprintln!("forward_embeds vs the reference (eps 1e-5): max abs diff {diff:e}");
    assert!(
        diff <= TOL_MODULE,
        "forward_embeds disagrees with the reference by {diff:e} (> {TOL_MODULE:e})"
    );
    // The eps divergence IS visible at module level, but only about 3x the
    // tolerance - which is why the operator-level claim lives on the gate,
    // where it is two orders of magnitude clearer.
    let shipped_diff = max_abs_diff(&ours, fx.get("module.out_shipped"));
    eprintln!(
        "forward_embeds vs the reference (its own eps 1.19e-7): {shipped_diff:e} \
         = {:.1}x TOL_MODULE",
        shipped_diff / TOL_MODULE
    );
    assert!(
        shipped_diff > diff,
        "the eps difference did not move the module output at all; the module \
         fixture is not sensitive to it"
    );
}

// ─── 5. The structural gaps, asserted as gaps ───────────────────────────

/// The reference's three `nn.ModuleList`s of `nn.RMSNorm` (l.355, l.356, and
/// `ShortConv`'s l.148-151) default to `elementwise_affine=True`, i.e. a
/// learnable gain per element; its `nn.Linear`s (l.351-354) default to
/// `bias=True`. burn-engram has neither: its gate norms are
/// `x / sqrt(mean(x^2) + eps)` (lib.rs:214-228) and its Linears are
/// `.with_bias(false)` (lib.rs:136-146).
///
/// The numeric fixtures above are made comparable by setting the reference's
/// gains to 1 and biases to 0, so they CANNOT detect these. This is the test
/// that records the gap, with the count derived from the reference's OWN shipped
/// configuration (hc_mult = 4, hidden = 1024).
#[test]
fn reference_has_learnable_norm_gains_and_linear_biases_and_burn_has_none() {
    let fx = Fx::load();
    let gains = fx.one("meta.ref_norm_gain_params_shipped") as usize;
    let bias = fx.one("meta.ref_bias_params_shipped") as usize;
    assert_eq!(
        (gains, bias),
        (12_288, 5_120),
        "the reference's parameter inventory changed; recount from l.148-151, \
         l.355-356 and l.351-354 with hc_mult = 4, hidden = 1024"
    );
    assert_eq!(fx.one("meta.ref_norm_count"), 9.0, "3 norm groups x 3 heads");

    // burn-engram's own inventory, counted the same way. `EngramModule` has no
    // norm parameters and no Linear biases, so both are 0; if anyone adds
    // either, this reddens and the numeric fixtures above stop being
    // comparable-by-construction.
    let (module, _, _) = oracle_module(&fx);
    let mut n_params = 0usize;
    module.map(&mut Counter { n: &mut n_params });
    let expected = fx.get("module.param.memory.weight").len()
        + fx.get("module.param.value_proj.weight").len()
        + fx.get("module.param.conv_weight_burn").len()
        + fx.get("module.param.key_projs[0].weight").len() * 3;
    assert_eq!(n_params, expected, "EngramModule's tensor count moved");
    assert_eq!(
        expected,
        2194 * 24 + 3 * 192 * 64 + 192 * 64 + 64 * 4,
        "EngramModule's parameter count moved"
    );
    // The short conv is the third structural gap: the reference stores
    // `hc_mult * D` independent channel kernels (l.138-146), burn-engram stores
    // `D` and broadcasts them across the groups.
    assert_eq!(
        fx.get("module.param.conv_weight").len(),
        fx.get("module.param.conv_weight_group0").len() * 3,
        "the reference stores hc_mult x our conv parameters at hc_mult = 3"
    );
    let g0 = fx.get("module.param.conv_weight_group0");
    let burn = fx.get("module.param.conv_weight_burn");
    let k = fx.at("meta.conv_kernel", 0) as usize;
    for ch in 0..g0.len() / k {
        for tap in 0..k {
            assert_eq!(
                burn[ch * k + tap],
                g0[ch * k + (k - 1 - tap)],
                "channel {ch} tap {tap}: the burn-tap-order kernel must be the \
                 exact per-channel time-reverse of the reference's, or the two \
                 convolutions compute different functions"
            );
        }
    }
}

/// Counts float params in traversal order; returns them untouched.
struct Counter<'a> {
    n: &'a mut usize,
}

impl ModuleMapper for Counter<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        *self.n += param.val().dims().iter().product::<usize>();
        param
    }
}

// ─── 6. The claim, and the guard on the claim ────────────────────────────

/// One place that states what this file claims and checks the claim against
/// itself. A tolerance wider than the fixture's discriminating power makes
/// every "must be on this side" assertion vacuous; that is the failure this
/// catches, and it is why the numbers are named constants instead of inlined.
#[test]
fn the_claim() {
    let fx = Fx::load();
    let ours = our_gate(&fx);
    let add = fx.get("gate.out.gate_add_eps1e5");
    let clamp = fx.get("gate.out.gate_eps1e5");

    // (1) Every resolvable row sits on the `clamp_min` side - the reference's
    //     own gate - inside its budget. Until 2026-09-29 this asserted the
    //     `add` side, which is the bug `gate_is_on_the_clamp_min_side` has
    //     been red about since 2026-09-28: the claim and the gate named the
    //     same file and contradicted each other, and the gate was the wrong
    //     one. Both were fixed together or neither is meaningful.
    let mut checked = 0usize;
    let mut max_residual: f64 = 0.0;
    // (2) The band is inside the resolvable set, so the distinction is
    //     meaningful there, and the fixture's weakest band row still separates
    //     the two columns by DISCRIM_FACTOR tolerances.
    let mut band = 0usize;
    let mut min_ratio = f64::INFINITY;
    for i in 0..ours.len() {
        if fx.resolvable(i) {
            checked += 1;
            max_residual = max_residual.max(abs_diff(ours[i], clamp[i]) - fx.gate_tol(i));
        }
        if fx.in_band(i) {
            assert!(
                fx.resolvable(i),
                "band row {i} (|s| = {:e}) is not resolvable, so the band is \
                 not where the distinction can be drawn",
                fx.get("gate.row.s")[i].abs()
            );
            band += 1;
            let sep = (clamp[i] - add[i]).abs();
            min_ratio = min_ratio.min(sep / fx.gate_tol(i));
        }
    }
    assert!(max_residual <= 0.0, "a resolvable gate row exceeded its tolerance");
    assert!(checked >= 14, "only {checked} resolvable rows");
    assert!(band >= 4, "the discriminating band is down to {band} rows");
    assert!(
        min_ratio > DISCRIM_FACTOR,
        "the fixture's weakest band row separates clamp_min from add by only \
         {min_ratio:.1}x its own tolerance; DISCRIM_FACTOR is {DISCRIM_FACTOR}"
    );
    eprintln!(
        "CLAIM: burn-engram agrees with deepseek-ai/Engram@main \
         engram_demo_v1.py to {TOL_GATE:e} on the gate (absolute, f32, over \
         {checked} of {} rows the reference can itself resolve; {band} of those \
         sit in the band where clamp_min and add are separable), {TOL_MODULE:e} \
         on the full EngramModule forward, {TOL_CONV:e} on the depthwise conv, \
         and EXACTLY on the prime ladder.\n\
         IT DOES NOT AGREE on: the clamp_min/add token (lib.rs:234 uses add; \
         see gate_is_on_the_clamp_min_side, which is RED), the RMS eps (1e-5 vs \
         a measured 1.19e-7), the hash VALUES (splitmix64 vs numpy PCG64), or \
         the parameter inventory (no norm gains, no Linear biases, \
         group-shared conv kernels).\n\
         The fixture's weakest band row separates the two gate columns by \
         {min_ratio:.1}x its tolerance, so the clamp_min/add distinction is \
         decided with that much margin and cannot be widened away.",
        fx.rows()
    );
}
