//! Tier-(a) oracle: the expected values come from the Muon+ AUTHORS' own code,
//! named in their paper's own abstract.
//!
//! ```text
//! repo   https://github.com/K1seki221/MuonPlus
//! commit 8a9ace123afedaab8ba75ea0b19315594ae1da7c  (HEAD; last push 2026-02-26)
//! files  utils/optim/muon_plus.py, utils/optim/polar_express.py
//! paper  arXiv:2602.21545 v1/v2 abstract: "We provide our code here:
//!        https://github.com/K1seki221/MuonPlus."
//! ```
//!
//! Both files are pinned byte-identically under `tests/oracle/upstream/`, and
//! `tests/oracle/gen_oracle.py` asserts their sha256 before importing anything,
//! so the pin cannot rot silently. That script PRODUCES `muon_oracle.bin` by
//! running the authors' `apply_post_polar_norm`; this file consumes it.
//!
//! ## What is gated, and what is not, and why
//!
//! **Gated at zero tolerance: the Newton-Schulz coefficients.** `NS_COEFFS` is
//! compared against the decimal literal in the authors' source
//! (`muon_plus.py:56`), parsed at test time. A four-significant-figure constant
//! copied by hand is the actual transcription risk here, and a literal
//! comparison catches a single wrong digit with no tolerance at all.
//!
//! **Not gated numerically: the Newton-Schulz iteration itself.** The authors
//! run it in bfloat16 (`X = G.bfloat16()`, `muon_plus.py:55`); we run f32.
//! Measured on this box, 2026-09-30 (`tests/oracle/transcript.txt`):
//!
//! | quantity | value |
//! |---|---|
//! | max abs, authors' bf16 vs the same arithmetic in f32 | 2.3e-2 … 2.1e-1 |
//! | max abs, f32 with `a` perturbed by 1e-4 RELATIVE | 1.5e-4 … 5.8e-4 |
//! | **signal / noise** | **0.002 … 0.02** |
//!
//! The dtype difference the authors' code carries is 50x to 5000x LARGER than
//! the defect a numeric gate would be hunting, so any bar that passes is above
//! the signal and catches nothing. A loose bar here would be a green line over
//! nothing - the exact failure this file exists to prevent. The constant is
//! gated instead; the iteration's *shape* invariants are pinned by
//! `orient_and_normalize_is_never_taller_than_wide` and the orthogonality
//! check in `self_checks.rs`. This is a real ceiling, not a gap for later.
//!
//! **Gated numerically at 3e-5: `normalize`, all four directions.** This is
//! the layer that catches the axis and composition-order class. See
//! `TOL_NORMALIZE` for where 3e-5 comes from; it is not a number fitted to a
//! pass.
#![allow(deprecated)]

use burn::tensor::{Device, Tensor, TensorData};
use dormouse_muon_plus::{MuonPlusConfig, NormDir, NS_COEFFS};

const UPSTREAM_MUON_PLUS: &str = include_str!("oracle/upstream/utils/optim/muon_plus.py");
const UPSTREAM_POLAR_EXPRESS: &str = include_str!("oracle/upstream/utils/optim/polar_express.py");
const FIXTURE: &[u8] = include_bytes!("oracle/muon_oracle.bin");

/// Numeric bar for `normalize` against the authors' implementation.
///
/// The two implementations differ in exactly two places, and only two.
///
/// **1. Epsilon placement — a small term here, and a large one elsewhere.**
/// The authors write `sqrt(v² + 1e-7)`, the epsilon INSIDE the root, hardcoded
/// at `muon_plus.py:105,116,118` (their `norm_eps` argument is accepted and
/// never read - its own finding, in `FINDINGS-muon-oracle.md`). We write
/// `max(v, 1e-7)`, a floor on the divisor, outside the root. At divisor
/// `v > eps` the two differ by `1 - (1 + eps/v²)^(-1/2) ~ eps/(2v²)` RELATIVE.
/// On the fixture's axes that is at most `eps/(2*0.76) = 6.6e-8` relative,
/// i.e. below 1e-7 absolute. Where the floor BINDS it is a different story and
/// a different function: at `v = 1e-6` the two differ by 5e4 relative, so no
/// single bar can be both tight enough to catch a transposed axis and loose
/// enough to admit a tiny-norm input. Inputs with a tiny axis norm are
/// therefore EXCLUDED from this gate. That exclusion is stated, and
/// `falsify.sh` step C2 measures it: moving the floor from 1e-7 to 1e-4 leaves
/// this suite green, because no fixture axis norm falls in that range.
/// **The epsilon is not gated. Nothing here claims it is.**
///
/// **2. f32 reassociation in `sum_dim` — the term that sets the bar.** The
/// two implementations sum the same squares in a different order relative to
/// the divisions around them. With `u = 2^-24 = 5.96e-8`, a naive reduction
/// over `k` terms bounds the relative error at `k*u`; the fixture's longest
/// reduction is 32 terms (the 128x64 case reduces over 64) and two
/// normalizations compound, so the bound lands near 1e-6 - and the measured
/// worst is 1.97e-6, which is that term.
///
/// `16 * k * u` with `k = 32` is 3.05e-5, rounded to **3e-5**. The measured
/// worst over the 32-case fixture is **1.97e-6**, i.e. 15x inside the bar. The
/// defect this bar exists to catch is far above it, and
/// `the_bar_is_far_below_the_defect_it_exists_to_catch` measures that rather
/// than leaving it to this comment.
const TOL_NORMALIZE: f32 = 3e-5;

fn dev() -> Device {
    Device::ndarray().autodiff()
}

// ── the pinned source, parsed ──────────────────────────────────────────────

/// Every `(a, b, c)` float triple inside `src[from..to]`, in order.
fn triples(src: &str, from: usize, to: usize) -> Vec<(f64, f64, f64)> {
    let body = &src[from..to];
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(open) = rest.find('(') {
        let Some(close) = rest[open..].find(')') else {
            break;
        };
        let inner = &rest[open + 1..open + close];
        let nums: Vec<f64> = inner
            .split(',')
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .filter_map(|p| p.parse::<f64>().ok())
            .collect();
        if nums.len() == 3 {
            out.push((nums[0], nums[1], nums[2]));
        }
        rest = &rest[open + close + 1..];
    }
    out
}

/// The first `a, b, c = (...)` in the authors' Newton-Schulz function.
fn authors_ns_coeffs() -> (f64, f64, f64) {
    let at = UPSTREAM_MUON_PLUS
        .find("a, b, c = (")
        .unwrap_or_else(|| panic!("no `a, b, c = (` in the pinned muon_plus.py"));
    let all = triples(UPSTREAM_MUON_PLUS, at, UPSTREAM_MUON_PLUS.len());
    assert!(
        !all.is_empty(),
        "pinned muon_plus.py has no float triple after `a, b, c = (`"
    );
    all[0]
}

/// The PolarExpress coefficient schedule as the authors' repo LITERALS it,
/// taken from the FIRST `coeffs_list` assignment (the second one re-derives the
/// list by dividing by the safety factor and is not the schedule).
fn authors_polar_coeffs() -> Vec<(f64, f64, f64)> {
    let at = UPSTREAM_POLAR_EXPRESS
        .find("coeffs_list = [")
        .unwrap_or_else(|| panic!("no `coeffs_list = [` in the pinned polar_express.py"));
    let end = at
        + UPSTREAM_POLAR_EXPRESS[at..]
            .find(']')
            .expect("`coeffs_list = [` is never closed");
    triples(UPSTREAM_POLAR_EXPRESS, at, end)
}

// ── 1. the constant, at zero tolerance ─────────────────────────────────────

/// `NS_COEFFS` IS the authors' literal. No tolerance: our f32 must be the f32
/// nearest to the decimal literal in `muon_plus.py:56`, so a wrong digit on
/// either side fails. This is the gate that covers the Newton-Schulz
/// polynomial, because the numeric one is not available (see the module docs
/// for the measured signal-to-noise that rules it out).
#[test]
fn ns_coeffs_are_the_authors_literal() {
    let (a, b, c) = authors_ns_coeffs();
    assert_eq!(
        NS_COEFFS,
        (a as f32, b as f32, c as f32),
        "NS_COEFFS {:?} is not the f32 nearest to the authors' literal ({a}, {b}, {c}) \
         at K1seki221/MuonPlus@8a9ace12 utils/optim/muon_plus.py:56",
        NS_COEFFS
    );
    // And the literal is the paper's, quoted: 2602.21545 v3 App. D.1
    // "In [15], the coefficients are set to (a, b, c) = (3.4445, -4.7750, 2.0315)."
    assert_eq!(
        (a, b, c),
        (3.4445, -4.7750, 2.0315),
        "the pinned source no longer carries the paper's App. D.1 triple"
    );
}

/// `src/lib.rs:186-188` claims App. D.3's PolarExpress schedule "terminates at
/// `(1.875, -1.25, 0.375)`". That is a claim about a table in a paper, and
/// this makes it checkable against the Muon+ authors' own copy of the list.
///
/// NOTE, and it is a finding not a gate: the PolarExpress AUTHORS' repository
/// (`NoahAmsel/PolarExpress`) never contained this literal. At every one of its
/// four revisions `polar_express.py` computes the list at import time
/// (`optimal_composition(...)`), and running the current one on this box gives
/// a different 10-entry list converging to `(1.8564, -1.2132, 0.3568)`, with no
/// setting of `l` / `safety_factor_eps` / `cushion` reproducing the published
/// table. So the D.3 list is verifiable against the Muon+ authors' copy and
/// against the paper, and NOT against the algorithm's own authors' code.
/// Recorded in `FINDINGS-muon-oracle.md`; the citation is not weakened by it
/// because the crate cites v3 App. D.3, which does print the table.
#[test]
fn polarexpress_schedule_ends_where_our_doc_comment_says() {
    let coeffs = authors_polar_coeffs();
    assert_eq!(
        coeffs.len(),
        8,
        "the pinned schedule has {} entries, not the 8 the paper's App. D.3 prints",
        coeffs.len()
    );
    assert_eq!(
        coeffs[6],
        (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
        "App. D.3's 7th entry changed: {:?}",
        coeffs[6]
    );
    assert_eq!(
        coeffs[7],
        (1.875, -1.25, 0.375),
        "the schedule no longer terminates at (1.875, -1.25, 0.375), which is \
         what src/lib.rs:186-188 says App. D.3 prints"
    );
}

// ── 2. the normalization, against the authors' run output ─────────────────

struct Case {
    m: usize,
    n: usize,
    mode: NormDir,
    input: Vec<f32>,
    want: Vec<f32>,
}

fn read_fixture() -> Vec<Case> {
    assert_eq!(&FIXTURE[..4], b"BMUO", "fixture magic");
    let u32at = |o: usize| u32::from_le_bytes(FIXTURE[o..o + 4].try_into().unwrap()) as usize;
    assert_eq!(u32at(4), 1, "fixture version");
    let count = u32at(8);
    let mut off = 12;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let (m, n, mode) = (u32at(off), u32at(off + 4), u32at(off + 8));
        assert_eq!(u32at(off + 12), 0, "unknown tolerance class in the fixture");
        off += 16;
        let k = m * n;
        let f = |o: usize| -> Vec<f32> {
            FIXTURE[o..o + 4 * k]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        let input = f(off);
        off += 4 * k;
        let want = f(off);
        off += 4 * k;
        out.push(Case {
            m,
            n,
            mode: match mode {
                0 => NormDir::Col,
                1 => NormDir::Row,
                2 => NormDir::ColRow,
                3 => NormDir::RowCol,
                other => panic!("unknown mode {other} in the fixture"),
            },
            input,
            want,
        });
    }
    assert_eq!(off, FIXTURE.len(), "fixture has trailing bytes");
    out
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// `MuonPlus::normalize` against the output the Muon+ authors' own
/// `apply_post_polar_norm` produced, for all four directions and eight shapes.
#[test]
fn normalize_matches_the_authors_implementation() {
    let device = dev();
    let cases = read_fixture();
    assert_eq!(cases.len(), 32, "fixture case count changed");
    let mut worst = 0.0f32;
    for c in &cases {
        let muon = MuonPlusConfig::new().with_norm_dir(Some(c.mode)).build();
        let data = TensorData::new(c.input.clone(), [c.m, c.n]);
        let got: Vec<f32> = muon
            .normalize(Tensor::<2>::from_data(data, &device))
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let d = maxdiff(&got, &c.want);
        worst = worst.max(d);
        assert!(
            d <= TOL_NORMALIZE,
            "{:?} on {}x{}: max|ours - K1seki221/MuonPlus@8a9ace12 \
             apply_post_polar_norm| = {d:e}, bar {TOL_NORMALIZE:e} \
             (derivation: 16*k*u, k=32, u=2^-24, the f32 reassociation term)",
            c.mode,
            c.m,
            c.n
        );
    }
    println!(
        "normalize: worst deviation over {} cases = {worst:e}",
        cases.len()
    );
    assert!(
        worst < TOL_NORMALIZE / 4.0,
        "the worst case ({worst:e}) has eaten most of the bar ({TOL_NORMALIZE:e}); \
         the f32 reassociation term it was derived from has grown"
    );
}

/// The bar is only meaningful if the defect it exists to catch is far above
/// it. `ColRow` and `RowCol` are different functions and the paper names which
/// is which (Eq. (7) col-then-row, Eq. (8) row-then-col), so a swap is a silent
/// loss of a different update. Measured against the AUTHORS' output for both,
/// so the number cannot be inflated by a bug we share.
#[test]
fn the_bar_is_far_below_the_defect_it_exists_to_catch() {
    let cases = read_fixture();
    let pick = |mode: NormDir| {
        cases
            .iter()
            .find(|c| c.m == 16 && c.n == 8 && c.mode == mode)
            .unwrap_or_else(|| panic!("the 16x8 {mode:?} case is in the fixture"))
    };
    let (cr, rc) = (pick(NormDir::ColRow), pick(NormDir::RowCol));
    assert_eq!(cr.input, rc.input, "the two fixture cases share an input");
    let d = maxdiff(&cr.want, &rc.want);
    assert!(
        d > 100.0 * TOL_NORMALIZE,
        "col_row and row_col differ by only {d:e} on the fixture input, so a bar \
         of {TOL_NORMALIZE:e} cannot distinguish them and the composition order \
         is not actually pinned by the comparison above"
    );
    println!(
        "order signal |col_row - row_col| on 16x8 = {d:e}, {:.0}x the bar {TOL_NORMALIZE:e}",
        d / TOL_NORMALIZE
    );
}
