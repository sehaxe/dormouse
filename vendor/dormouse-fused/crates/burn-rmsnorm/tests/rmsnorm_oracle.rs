//! # RMSNorm against upstream's OWN code, run.
//!
//! **Tier (a) — the first one in this tree that is not a self-comparison.**
//!
//! ## What produced every expected number
//!
//! Two upstreams, both executed, neither of them written here and neither of
//! them an arm of ours:
//!
//! | # | what was run | shipped as | where the bytes are pinned |
//! |---|---|---|---|
//! | 1 | `torch.nn.functional.rms_norm` → `torch.rms_norm`, PyTorch's own ATen operator | `torch==2.14.0+cpu`, wheel from download.pytorch.org | `oracle/upstream/torch_rms_norm.py` (the two Python entry points, verbatim); the arithmetic is compiled C++ (`aten/src/ATen/native/layer_norm.cpp::rms_norm` in [pytorch/pytorch](https://github.com/pytorch/pytorch) at tag `v2.14.0`) and is not quotable in this file |
//! | 2 | `fla.modules.layernorm.rms_norm_ref`, flash-linear-attention's own torch-level reference | `flash-linear-attention==0.5.2` on PyPI, [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | `oracle/upstream/fla_rms_norm_ref.py`, verbatim, with `sha256(fla/modules/layernorm.py) = e78b729b…c6d6f` in the fixture header |
//!
//! Fetched and run 2026-09-29 on this box, CPU only. The generator is
//! `oracle/gen_rmsnorm_oracle.py`; the values are in
//! `fixtures/rmsnorm_oracle.txt`; the test needs no network.
//!
//! **Revision reproducibility, stated rather than implied.** `torch` is pinned
//! by release tag (`v2.14.0`), which *is* reproducible. `flash-linear-attention`
//! is a PyPI wheel and carries **no git revision**, so the sha256 above is the
//! only handle: the same wheel at a different release will have a different
//! hash, and this fixture is tied to that hash rather than to a commit I can
//! cite. Neither citation is the paper's authors.
//!
//! ## What that does NOT license
//!
//! arXiv:1910.07467 (Zhang & Sennrich) ships **no code**, so "the authors' own
//! implementation of *this paper*" does not exist and cannot be run. PyTorch and
//! FLA are *implementations of the mechanism*, chosen because they are the two
//! public reference implementations the field's own linear-attention stack
//! uses, and because they are independent of each other. So: this file is
//! tier (a) against **PyTorch's and FLA's RMSNorm**, and it is **not** evidence
//! that either one is a faithful transcription of the paper. A shared
//! misreading of Zhang & Sennrich survives it — which is a smaller version of
//! the risk `docs/protocols/ORACLE.md` §3 takes for tier (c), not the same thing, because
//! here neither side was written by the person who wrote the Rust.
//!
//! Measured and worth recording: on all 12 cases the two upstreams agree to
//! **0.0** relative (bit-identical f32), so the fixture's correct column is not
//! a compromise between two opinions. The generator refuses to write a fixture
//! on which they disagree by more than 1e-6.
//!
//! ## The tolerances, and the arithmetic behind them
//!
//! `TOL_REL = 1e-5` relative. The bound it has to cover, in f32:
//!
//! * the denominator is a mean over `d ≤ 13` terms. A tree reduction of `d`
//!   terms has relative error `≤ log2(d)·u` with `u = 2⁻²⁴ = 5.96e-8`, so
//!   `≤ 4·5.96e-8 = 2.4e-7`;
//! * burn computes `x / rms`, both upstreams compute `x · (1/rms)`. A division
//!   and a reciprocal-then-multiply differ by up to 2 ulp = 1.2e-7;
//! * the weight multiply and the fixture's own 9-significant-digit encoding add
//!   ~1e-7.
//!
//! Sum: **≈ 5e-7**. `1e-5` is ~20× that head-room and, per `the_claim`, still
//! more than three orders of magnitude below the smallest margin the fixture
//! actually has on the distinction being claimed.
//!
//! ## The tolerance is checked against the fixture, not trusted
//!
//! A tolerance wider than the fixture's discriminating power makes every
//! "must be on this side" assertion vacuous — the defect class
//! `vendor/dormouse-fused/crates/burn-engram/tests/engram_oracle.rs` exists to
//! catch, and it caught one. `the_claim` here does the same job: it measures,
//! per case and per wrong column, how far the correct column sits from the
//! wrong one, and asserts that the cases which are supposed to decide the eps
//! question decide it by `DISCRIM_FACTOR × TOL_REL`, and that `big` (which
//! provably cannot) is excluded by that same measurement rather than by hand.
//!
//! **Measured on the committed fixture, 2026-09-29**: the worst eps-decisive
//! case against the worst wrong column is `4.45e-1`, i.e. **44 538× TOL_REL**
//! and 22× the 2e-5 the guard demands. The undecidable end is measured too —
//! `big` (`r ~ 7.9e3`) sits at `9.9e-8` for both eps columns, `1.0e-2` of the
//! tolerance, so it is *provably* unable to decide eps and the test says so
//! with a number rather than asserting it in prose. `main` (`r ~ 2.13`) is the
//! everyday case and is likewise undecidable at `3.8e-6` — it is in the
//! fixture precisely to show the tolerance is not being asked to carry eps.
//!
//! ## Three defects the guards found, and what they cost
//!
//! The guards in this file were red on arrival. All three were FIXTURE
//! defects, not guard defects, and each is now checked at generation time so it
//! cannot come back:
//!
//! 1. **the eps-decisive cases were not eps-decisive.** `zero_row` and `d1`
//!    sat at `r ~ 2` and `r ~ 5.3` against `eps = 1e-5`, where the eps-outside
//!    gap is `~e/r ≈ 4.6e-6` — *below* `TOL_REL`. The guard reported 0.3× TOL
//!    and was right to. Fixed by putting them at `r ~ 2e-3` and `r ~ 5e-5`
//!    (see `EPS_DECISIVE`'s doc comment for the arithmetic).
//! 2. **`d1`'s gain was constant** — a `d == 1` gain is one element *by
//!    shape*, so no per-feature broadcast exists to break. The guard could not
//!    be satisfied by any fixture at `d == 1`; it is now skipped by that
//!    derivation and `d2` (`d == 2`, gains `[-2.5, 3.0]`) was added because
//!    `d == 2` is the smallest axis where the broadcast IS observable.
//! 3. **`micro_eps0` was a different dataset from `micro`.** The generator
//!    drew a second random tensor for it, so
//!    `eps_is_a_parameter_of_the_call_on_both_sides` was measuring the
//!    difference between two datasets and calling it the eps difference. It
//!    passed. The generator now draws once, and the test asserts the two `x`
//!    rows are bit-identical before it measures anything.
//!
//! ## The oracle has been SEEN RED, on five wrong kernels
//!
//! A test that has only ever been green is not evidence that it can fail. Five
//! mutants of `RMSNorm::forward`'s tensor path, run by
//! `tests/oracle/mutate_kernel.sh` on 2026-09-29, with the kernel restored after
//! each (md5 checked). **Measured, not asserted**:
//!
//! | # | perturbation | `forward_matches…` | `eps_is_a_parameter…` | `the_claim` | `gains_are_not_constant` |
//! |---|---|---|---|---|---|
//! | — | none (baseline) | ok | ok | ok | ok |
//! | M1 | eps moved OUTSIDE the sqrt | **FAILED** 8.15e-1 on `tiny` | ok | **FAILED** 9.1e-8 vs 2e-5 | ok |
//! | M2 | eps dropped | **FAILED** 8.24e-1 on `tiny` | **FAILED** gap `0e0` | **FAILED** `0e0` vs 2e-5 | ok |
//! | M3 | gain replaced by its mean | **FAILED** 9.98e-1 on `main` | ok | **FAILED** | ok |
//! | M4 | `mean_dim(2)` → `mean_dim(1)` | **FAILED** 1.04e0 on `main` | ok | **FAILED** | ok |
//! | M5 | eps hardcoded to 1.1920929e-7 | **FAILED** 7.99e-1 on `tiny` | **FAILED** gap `0e0` | ok | ok |
//!
//! Three things in that table are worth reading rather than skimming:
//!
//! * **the comparison test, the one that was green and had never been seen
//!   red, catches every one of the five** — by 8 to 10 orders of magnitude
//!   past its own tolerance, not marginally.
//! * `the_claim` stays GREEN under M5. That is correct and is the point of the
//!   split: `the_claim` measures the FIXTURE's discriminating power, and under
//!   M5 the fixture is unchanged and just as discriminating. It is
//!   `eps_is_a_parameter…`, which compares two runs of the KERNEL, that owns
//!   "eps reached the implementation". Two guards with different jobs; neither
//!   is asked to do the other's.
//! * `the_fixture_gains_are_not_constant` stays GREEN under M3, for the same
//!   reason — it asserts the fixture cannot go vacuous, not that the kernel
//!   applies the gain. M3 is caught by the comparison at 9.98e-1 on `main`.
//!
//! Re-run it: `bash tests/oracle/mutate_kernel.sh`. It restores `src/lib.rs`
//! on every exit path and verifies the md5 at the end.
//!
//! ## The fused CUDA kernel is NOT covered by any of this
//!
//! The arithmetic below is the **tensor path**. The fused kernel has its own
//! file, `rmsnorm_kernel_cuda.rs`: the same fixture, against the kernel's own
//! output on a bare device, plus the gate on the trainer's `norm=0/N`. The
//! kernel never engages on the trainer's backend because an autodiff tensor
//! cannot be handed a bare kernel (`docs/protocols/ORACLE.md` / AGENTS.md §3.3; the eval
//! line prints `norm=0/N`) — so it is that refusal, and not a missing test,
//! that keeps the trainer's numerics on the path measured here.
//!
//! Run: `cargo test -p burn-rmsnorm --test rmsnorm_oracle`

#![allow(deprecated)] // Device::ndarray: the backend the fixture was made on

use std::collections::HashMap;

use burn::module::Param;
use burn::tensor::{Device, Tensor, TensorData};

use burn_rmsnorm::RMSNorm;

const FIXTURE: &str = include_str!("fixtures/rmsnorm_oracle.txt");

/// See the module docs for the arithmetic. ~20x the f32 sum bound of ~5e-7.
const TOL_REL: f64 = 1e-5;
/// A "must be on this side" assertion must beat the tolerance by this factor,
/// or it is not a claim. Same constant as the engram oracle.
const DISCRIM_FACTOR: f64 = 2.0;
/// Cases the fixture must be able to decide the EPS question on.
///
/// **Why a scale is the only lever**, with `r = sqrt(mean(x^2))` the row's rms
/// and `e = eps` — the correct answer divides by `sqrt(r^2 + e)`, so:
///
/// | wrong formula | its denominator | gap to the right one | as `r` grows |
/// |---|---|---|---|
/// | eps OUTSIDE | `r + e` | `\|(r+e)/sqrt(r^2+e) - 1\| ~ e/r` | falls as `1/r` |
/// | eps DROPPED | `sqrt(r^2)` | `sqrt(r^2+e)/r - 1 ~ e/(2r^2)` | falls as `1/r^2` |
///
/// On `main` (`r ~ 2.1`) those are `4.8e-6` and `1.1e-6` — at or below any
/// honest f32 tolerance, which is exactly why "close enough" cannot decide this
/// question. The cases below sit at `r ~ 5e-5 … 2e-3`, where both gaps are
/// O(0.1 … 1). Measured on the regenerated fixture, the WORST of these five
/// against the WORST of the three wrong columns is **4.45e-1 = 44 538x
/// TOL_REL** — four orders of magnitude above the `2e-5` this guard demands,
/// and `the_claim` re-measures it and prints it rather than trusting it here.
///
/// `micro` is the weakest of them against `out_layernorm` (1.96e-2) and always
/// will be: its rows are zero-mean noise, and `out_layernorm` centres the row,
/// so on a row whose mean is already ~0 the two answers nearly coincide. That
/// is 1962x TOL and is stated in the generator rather than discovered here.
const EPS_DECISIVE: [&str; 5] = ["tiny", "micro", "zero_row", "d1", "d2"];
/// The wrong columns, and the case that must separate from each of them.
const WRONG_COLUMNS: [&str; 4] = ["out_eps_outside", "out_no_eps", "out_layernorm", "out_axis1"];

fn dev() -> Device {
    Device::ndarray()
}

fn fixture() -> HashMap<String, Vec<f64>> {
    let mut out: HashMap<String, Vec<f64>> = HashMap::new();
    for line in FIXTURE.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, body) = line
            .split_once(':')
            .unwrap_or_else(|| panic!("fixture line without ':' — {line:?}"));
        // `meta.*` rows are provenance (a version string, a path, a sha256),
        // not numbers; the numeric columns are read by name. Skipping them
        // here rather than in the generator keeps the fixture human-readable
        // and keeps the claim reproducible from the file on disk.
        if key.starts_with("meta.") {
            continue;
        }
        out.entry(key.to_string()).or_default().extend(
            body.split_whitespace()
                .map(|v| v.parse::<f64>().unwrap_or_else(|_| panic!("bad number {v:?} in {key:?}"))),
        );
    }
    out
}

struct Fx {
    map: HashMap<String, Vec<f64>>,
    /// `meta.*` rows verbatim: provenance (versions, paths, sha256) and the
    /// case list, none of which is a number. Kept as text so the fixture on
    /// disk stays the thing a human reads.
    meta: HashMap<String, String>,
}

impl Fx {
    fn load() -> Self {
        let map = fixture();
        let mut meta: HashMap<String, String> = HashMap::new();
        for line in FIXTURE.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once(':') {
                if k.starts_with("meta.") {
                    meta.insert(k.to_string(), v.trim().to_string());
                }
            }
        }
        // Nine significant digits round-trip an f32 exactly. The bug this
        // guards is a generator writing six (`f"{v:g}"`), which silently
        // truncated every column to 0.49975 and reddened a green test.
        assert!(
            map.keys().any(|k| k.ends_with("out_torch")),
            "the fixture has no expected columns at all"
        );
        Self { map, meta }
    }
    fn cases(&self) -> Vec<String> {
        self.meta
            .get("meta.cases")
            .unwrap_or_else(|| panic!("fixture has no meta.cases"))
            .split_whitespace()
            .map(str::to_string)
            .collect()
    }
    fn text(&self, key: &str) -> &str {
        self.meta
            .get(key)
            .map(String::as_str)
            .unwrap_or_else(|| panic!("fixture has no {key:?}"))
    }
    fn get(&self, key: &str) -> &[f64] {
        self.map
            .get(key)
            .unwrap_or_else(|| panic!("fixture key {key:?} missing"))
    }
    fn one(&self, key: &str) -> f64 {
        let v = self.get(key);
        assert_eq!(v.len(), 1, "{key} should be a scalar, got {}", v.len());
        v[0]
    }
    fn dims(&self, case: &str) -> Vec<usize> {
        self.get(&format!("case.{case}.dims"))
            .iter()
            .map(|v| *v as usize)
            .collect()
    }
    /// Max |a − b| / max(|b|, 1) over a case. Non-finite in EITHER column
    /// counts as maximally separated: a `no_eps` column that is NaN on the
    /// zero row is a *stronger* discriminator than a large number, and it must
    /// not be silently skipped by an `is_finite()` guard.
    fn rel_diff(&self, case: &str, ours: &[f32], col: &str) -> f64 {
        let want = self.get(&format!("case.{case}.{col}"));
        assert_eq!(ours.len(), want.len(), "{case}/{col}: {} vs {}", ours.len(), want.len());
        ours.iter()
            .zip(want)
            .map(|(a, b)| {
                if !a.is_finite() || !b.is_finite() {
                    return f64::INFINITY;
                }
                (f64::from(*a) - b).abs() / b.abs().max(1.0)
            })
            .fold(0.0f64, f64::max)
    }
}

/// Load the fixture's inputs into a module. `weight` is a real gain, never
/// all-ones, so a broken broadcast cannot hide.
fn run_case(fx: &Fx, case: &str) -> Vec<f32> {
    let dims = fx.dims(case);
    assert_eq!(dims.len(), 3, "RMSNorm::forward takes a rank-3 tensor");
    let eps = fx.one(&format!("case.{case}.eps")) as f32;
    let w = fx.get(&format!("case.{case}.w"));
    let xs = fx.get(&format!("case.{case}.x"));
    assert_eq!(xs.len(), dims.iter().product::<usize>());

    let mut norm = RMSNorm::new(dims[2], eps, &dev());
    norm.weight = Param::from_tensor(Tensor::<1>::from_data(
        TensorData::new(w.iter().map(|v| *v as f32).collect::<Vec<_>>(), [dims[2]]),
        &dev(),
    ));
    let x = Tensor::<3>::from_data(
        TensorData::new(xs.iter().map(|v| *v as f32).collect::<Vec<_>>(), dims),
        &dev(),
    );
    norm.forward(x)
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

// ── 1. the comparison ──────────────────────────────────────────────────────

/// Every case, against BOTH upstreams. One green number per column.
#[test]
fn forward_matches_pytorch_and_flash_linear_attention() {
    let fx = Fx::load();
    let cases = fx.cases();
    assert!(cases.len() >= 10, "fixture has {} cases", cases.len());

    let (mut worst_torch, mut worst_fla) = (0.0f64, 0.0f64);
    for case in &cases {
        let ours = run_case(&fx, case);
        for (col, worst) in [("out_torch", &mut worst_torch), ("out_fla", &mut worst_fla)] {
            let d = fx.rel_diff(case, &ours, col);
            assert!(
                d <= TOL_REL,
                "case {case}: RMSNorm::forward differs from the reference's `{col}` \
                 column by {d:e} relative (> {TOL_REL:e}). Fixture: \
                 tests/fixtures/rmsnorm_oracle.txt",
            );
            *worst = worst.max(d);
        }
    }
    eprintln!(
        "RMSNorm vs upstream, {} cases: max rel diff {worst_torch:e} against \
         torch.nn.functional.rms_norm (torch.rms_norm, ATen) and {worst_fla:e} \
         against fla.modules.layernorm.rms_norm_ref (flash-linear-attention \
         {}), both at TOL_REL {TOL_REL:e}",
        cases.len(),
        fx.text("meta.upstream_fla_version"),
    );
    assert!(worst_torch <= TOL_REL && worst_fla <= TOL_REL);
}

/// The eps is a PARAMETER, not a constant on either side. `micro` at eps = 1e-5
/// and `micro_eps0` at eps = 0 are the SAME tensor; if either implementation
/// baked its eps in, the two would be identical and the whole file would be
/// comparing constants.
///
/// **The fixture shipped with a second random tensor for `micro_eps0`,** so
/// this comparison was measuring the difference between two datasets and
/// calling it the eps difference — it passed for a reason that had nothing to
/// do with eps. The check below is what catches that class: the two `x` rows
/// must be bit-identical, so any difference in the outputs is attributable to
/// eps and to nothing else.
#[test]
fn eps_is_a_parameter_of_the_call_on_both_sides() {
    let fx = Fx::load();
    let (xs, xs0) = (fx.get("case.micro.x"), fx.get("case.micro_eps0.x"));
    assert_eq!(
        xs, xs0,
        "`micro` and `micro_eps0` are documented as the same tensor at eps = 0 \
         and to 9 significant digits; they are not. A different dataset here \
         makes every gap below a measurement of the DATA, not of eps."
    );
    assert_ne!(
        fx.get("case.micro.eps")[0],
        fx.get("case.micro_eps0.eps")[0],
        "the two cases are meant to differ ONLY in eps"
    );

    let a = run_case(&fx, "micro");
    let b = run_case(&fx, "micro_eps0");
    let zero_row = fx.get("case.micro_eps0.out_no_eps");
    // Sanity on the fixture itself: at eps = 0 the reference's `eps_inside`
    // column and its `no_eps` column must COINCIDE, or the two wrong columns
    // are not the two formulas they claim to be.
    let coincident = fx
        .get("case.micro_eps0.out_torch")
        .iter()
        .zip(zero_row)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    eprintln!(
        "fixture check: at eps = 0 the reference's eps-inside and no-eps columns \
         coincide to {coincident:e}"
    );
    assert!(
        coincident <= TOL_REL,
        "at eps = 0 the fixture's eps-inside and no_eps columns differ by \
         {coincident:e}; one of the two wrong columns is not the formula it names"
    );
    let gap = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs() / f64::from(*y).abs().max(1.0))
        .fold(0.0f64, f64::max);
    eprintln!(
        "eps 1e-5 vs eps 0 on the SAME `micro` tensor moves the output by \
         {gap:e} relative (analytic sqrt(1 + e/r^2) - 1 with r = 9.08e-5: \
         33.8, i.e. the answer is dominated by eps)"
    );
    assert!(
        gap > DISCRIM_FACTOR * TOL_REL,
        "changing eps from 1e-5 to 0 moved the output by only {gap:e}; the eps \
         is not reaching the implementation (or the fixture's two cases are \
         not the same shape)"
    );
}

/// The gain is applied per FEATURE. A constant gain makes a broken broadcast
/// invisible, so every case must carry a gain whose range is real — EXCEPT the
/// cases where the shape makes a range impossible.
///
/// The exemption is derived, not waived: `RMSNorm::new(d, ...)` holds a gain of
/// exactly `d` entries, so a `d == 1` case has a one-element gain and
/// `hi - lo == 0` BY CONSTRUCTION. There is no per-feature axis on such a row
/// for a broadcast to be wrong about. `d1` earns its place for a different
/// reason (the degenerate `mean(x^2) = x^2` reduction, at a magnitude where eps
/// decides), and `d2` is here because `d == 2` is the smallest axis on which a
/// collapsed gain IS observable — the assertion below is skipped for `d == 1`,
/// and the skip is only allowed to stand because a `d == 2` case exists.
#[test]
fn the_fixture_gains_are_not_constant() {
    let fx = Fx::load();
    let (mut constant, mut broadcastable) = (Vec::new(), 0usize);
    for case in fx.cases() {
        let w = fx.get(&format!("case.{case}.w"));
        let d = fx.dims(&case)[2];
        assert_eq!(
            w.len(),
            d,
            "case {case}: the gain has {} entries for a feature axis of {d}; a \
             gain that does not match the axis is not a per-feature gain",
            w.len()
        );
        if d == 1 {
            constant.push(case);
            continue;
        }
        broadcastable += 1;
        let lo = w.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (hi - lo) > 0.5,
            "case {case}: the gain ranges over [{lo}, {hi}]; with a constant \
             gain a broken per-feature broadcast is invisible"
        );
    }
    assert!(
        broadcastable > 0,
        "every case is d == 1, so nothing in the fixture tests a gain broadcast"
    );
    assert!(
        fx.cases().iter().any(|c| fx.dims(c)[2] == 2),
        "no d == 2 case: d == 1 is exempt by shape, so d == 2 is the smallest \
         axis on which the loop above actually decides anything"
    );
    eprintln!(
        "fixture gains: {broadcastable} case(s) carry a non-constant gain; \
         {constant:?} are d == 1 and exempt by shape (one gain entry, no \
         per-feature axis to break), which is why a d == 2 case must exist"
    );
}

// ── 2. the claim, and the guard on the claim ───────────────────────────────

/// States what this file claims, and checks the claim against itself.
///
/// The failure mode this exists for: a tolerance wider than the fixture's
/// discriminating power turns every "must be on this side" assertion into
/// noise. So it is measured, per case, per wrong column — including the cases
/// that are supposed to be UNDECIDABLE, which are excluded by that measurement
/// rather than by an assertion about them written by hand.
#[test]
fn the_claim() {
    let fx = Fx::load();
    let cases = fx.cases();

    // (1) The two upstreams agree with each other, exactly, on every case.
    let mut worst_pair: (f64, String) = (0.0, String::new());
    for case in &cases {
        let t = fx.get(&format!("case.{case}.out_torch"));
        let f = fx.get(&format!("case.{case}.out_fla"));
        let scale = t.iter().cloned().fold(0.0f64, f64::max).max(1.0);
        let gap = t
            .iter()
            .zip(f)
            .map(|(x, y)| (x - y).abs() / scale)
            .fold(0.0f64, f64::max);
        if gap > worst_pair.0 {
            worst_pair = (gap, case.clone());
        }
    }
    assert!(
        worst_pair.0 <= 1e-6,
        "torch and fla disagree on case {} by {}e relative; the fixture is \
         recording a conflict, not an answer",
        worst_pair.1, worst_pair.0
    );

    // (2) The cases that must decide the eps question, do — by a wide margin.
    // (3) The cases that cannot, are excluded BY the measurement, and the
    //     worst of them is still below the tolerance (so they are not
    //     accidentally relying on a wrong answer).
    let mut margins: HashMap<&str, f64> = HashMap::new();
    for col in WRONG_COLUMNS {
        let mut decisive = f64::INFINITY;
        for case in &cases {
            // `out_axis1` only exists for a cube whose last two axes match.
            if !fx.map.contains_key(&format!("case.{case}.{col}")) {
                continue;
            }
            let ours = run_case(&fx, case);
            let sep = fx.rel_diff(case, &ours, col);
            if EPS_DECISIVE.contains(&case.as_str()) && col != "out_axis1" {
                decisive = decisive.min(sep);
            }
        }
        assert!(
            decisive > DISCRIM_FACTOR * TOL_REL,
            "the eps-decisive cases only separate the correct column from `{col}` \
             by {decisive:.3}x TOL_REL ({decisive:e} vs {TOL_REL:e}); DISCRIM_FACTOR \
             is {DISCRIM_FACTOR} and the eps claim is not being decided",
        );
        margins.insert(col, decisive);
    }

    // (4) The `big` case is where eps provably cannot matter (mean(x^2) ~ 6.3e7
    //     against eps = 1e-5, a ~1e-13 relative effect). If it ever DID
    //     separate, the reasoning above is stale and must be rewritten.
    let big = run_case(&fx, "big");
    let big_sep = fx.rel_diff("big", &big, "out_no_eps");
    eprintln!(
        "the `big` case: eps-inside vs no-eps is {big_sep:e} relative \
         ({:.1e} x TOL_REL) -- measured, not assumed",
        big_sep / TOL_REL
    );
    assert!(
        big_sep <= TOL_REL,
        "the `big` case now separates eps-inside from no_eps by {big_sep:e}; the \
         module docs' claim that it cannot is stale"
    );

    // (5) The reduction axis is decided on a square cube, where mean-over-dim-1
    //     is well-formed and is not our answer.
    let cube = run_case(&fx, "axis_confusion");
    let axis_gap = fx.rel_diff("axis_confusion", &cube, "out_axis1");
    assert!(
        axis_gap > 1e3 * TOL_REL,
        "reducing over the wrong axis is only {axis_gap:e} away (> 1e3 x TOL_REL \
         required); the axis claim is not being decided"
    );

    eprintln!(
        "CLAIM: burn-rmsnorm's tensor path agrees with torch.nn.functional.rms_norm\n\
         (torch=={}, ATen `torch::rms_norm`) and with\n\
         fla.modules.layernorm.rms_norm_ref (flash-linear-attention {}, sha256 {})\n\
         to {TOL_REL:e} relative, over {} cases. The two upstreams agree with EACH\n\
         OTHER to {:.1}e over the same cases.\n\
         On the smallest eps-decisive case each wrong formula is this far from\n\
\
         the correct one, in units of the tolerance: eps-outside {:.0}x,\n\
         no-eps {:.0}x, LayerNorm-style centring {:.0}x. Reducing over the time\n\
         axis instead of the feature axis is {:.0}x on a square cube.\n\
         IT DOES NOT COVER: the fused CUDA kernel (it never engages on the\n\
         trainer's backend — AGENTS.md §3.3, eval line `norm=0/N`), and whether\n\
         either upstream is a faithful transcription of arXiv:1910.07467, whose\n\
         authors ship no code.",
        fx.text("meta.upstream_torch_version"),
        fx.text("meta.upstream_fla_version"),
        &fx.text("meta.upstream_fla_sha256")[..8],
        cases.len(),
        worst_pair.0,
        margins["out_eps_outside"] / TOL_REL,
        margins["out_no_eps"] / TOL_REL,
        margins["out_layernorm"] / TOL_REL,
        axis_gap / TOL_REL,
    );
}
