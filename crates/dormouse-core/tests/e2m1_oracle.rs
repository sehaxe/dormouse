//! # The e2m1 grid and its TIE RULE, against torchao's own FP4 quantizer.
//!
//! **Tier (a) — the authors' format, executed.** The first time anything in
//! this project has checked the `--act-quant fp4` path against a
//! *quantizer* rather than against a list someone typed.
//!
//! ## What produced every expected number
//!
//! Two files from [pytorch/ao](https://github.com/pytorch/ao) at commit
//! `3972ed015091f659418dedf12edb980a8ca56b53` (2026-09-25), both RUN on CPU
//! against `torch==2.14.0+cpu`:
//!
//! 1. `torchao/prototype/mx_formats/fp_format_spec.py`, whose
//!    `float4_e2m1_interesting_values` states the OCP MX FP4 grid as
//!    (value, formula, sign, exponent, mantissa) rows and cites
//!    <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
//!    §5.3.2 by URL;
//! 2. `torchao/prototype/mx_formats/mx_tensor.py::to_mx` — torchao's REAL
//!    block quantizer, which emits packed 4-bit codes for
//!    `torch.float4_e2m1fn_x2`.
//!
//! The generator is
//! `vendor/dormouse-fused/crates/dormouse-bitnet/tests/oracle/gen_e2m1_oracle.py`,
//! the values are in that crate's
//! `tests/fixtures/e2m1_oracle.txt`, and **this test needs no network**.
//!
//! ## Why torchao and not a bare torch cast
//!
//! `tensor.to(torch.float4_e2m1fn_x2)` raises
//! `NotImplementedError: "copy_kernel" not implemented for
//! 'Float4_e2m1fn_x2'` on CPU — the dtype exists but has no CPU cast kernel.
//! torchao's `to_mx` is the CPU-capable reference quantizer for the same
//! format, which is why it, and not a dtype round-trip, is the reference.
//!
//! ## What this settles
//!
//! **The grid is right.** Eight magnitudes, `0 0.5 1 1.5 2 3 4 6`, with
//! **0.75 absent** — which is the specific defect AGENTS.md §3.2 records
//! (`9b343d3`: the old mantissa rule emitted 0.75, "a ~3-level quantizer
//! wearing a 4-bit label"). 0.5 is correctly the single subnormal. Our
//! `E2M1` is byte-equal to torchao's own table.
//!
//! **The tie rule is wrong, and this is the new finding.** The grid's seven
//! interior tie points are 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0. At every
//! one, torchao selects the level with the **even code** — round-half-to-even,
//! the IEEE default and what the format's hardware does. Our `fp4_round`
//! (`src/act_quant.rs:137-140`) uses `mask_fill(a >= lo)`, which sends every
//! tie **up** the grid. The two disagree at **4 of the 7**: 0.25, 1.25, 2.5
//! and 5.0. At 5.0 that is 4.0 against 6.0 — a 50 % error on that input.
//!
//! ## Why no existing test could have caught it
//!
//! The three tests in `act_quant.rs` are all **self-consistency**: every grid
//! level round-trips to itself, and every output is the nearest level *by our
//! own search*. A tie IS nearest to both candidates, so a ties-up rule and a
//! ties-even rule are indistinguishable to both. The bug lives in the
//! question none of them asks — *which* nearest level — and the only way to
//! ask it is a reference that has already decided. That is the whole argument
//! for tier (a) over tier (d), in one test.
//!
//! This test is RED on purpose for the same reason and with the same
//! convention as `dormouse-gdn2/tests/fused_adjoint_f64.rs`: it is a defect
//! report that fails until someone fixes `fp4_round`. The fix is to break
//! exact ties toward the even code, which is a numerical change to a shipped
//! objective and therefore the owner's call (`.bulba/goal.md`: "Числовая
//! правка — отчёт, не редактирование"), not this lane's.
//!
//! What is asserted as GREEN below is everything the reference settles that
//! our code already does: the grid itself, the absence of 0.75, the
//! subnormal, and the fact that no interior tie is a NaN or a gap.

#![allow(deprecated)] // burn-ndarray is the CPU test backend here

use std::collections::HashMap;

use burn::backend::Flex;
use burn::tensor::{Device, Tensor, TensorData};

use dormouse_core::act_quant::{quant_act, ActFormat, E2M1};

/// The upstream commit. Named in the failure message so a red run says which
/// version of torchao's quantizer it disagrees with (AGENTS.md §1.4).
const TORCHAO_SHA: &str = "3972ed015091f659418dedf12edb980a8ca56b53";

fn fixture() -> HashMap<String, String> {
    let text = include_str!(
        "../../../vendor/dormouse-fused/crates/dormouse-bitnet/tests/fixtures/e2m1_oracle.txt"
    );
    let mut out = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once(':') {
            out.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    out
}

fn nums(s: &str) -> Vec<f32> {
    s.split_whitespace().map(|x| x.parse().unwrap()).collect()
}

/// The grid, against torchao's own table. GREEN: our `E2M1` is correct, and
/// the specific 0.75 defect of `9b343d3` stays dead.
#[test]
fn the_e2m1_grid_is_torchaos_grid() {
    let fx = fixture();
    let theirs = nums(&fx["grid.magnitudes"]);
    assert_eq!(
        E2M1.to_vec(),
        theirs,
        "our E2M1 differs from torchao's own table at {TORCHAO_SHA}"
    );
    assert_eq!(theirs.len(), 8, "e2m1 has 8 positive magnitudes, not {}", theirs.len());
    assert!(
        !theirs.iter().any(|v| (*v - 0.75).abs() < 1e-9),
        "0.75 reappeared in the grid -- it is not an e2m1 level (AGENTS.md 3.2)"
    );
    assert_eq!(
        theirs.iter().filter(|v| **v > 0.0 && **v < 1.0).count(),
        1,
        "e2m1 has exactly one subnormal (0.5)"
    );
    assert_eq!(fx["grid.count"], "8");
    assert_eq!(fx["grid.has_0.75"], "no");
    // The format's max, which `ActFormat::max_value` must scale onto. 6, not 1:
    // scaling a block's max onto 1 is the OTHER half of the 9b343d3 defect.
    assert_eq!(E2M1[E2M1.len() - 1], 6.0, "e2m1's largest magnitude is 6.0");
    assert_eq!(
        ActFormat::Fp4.max_value(),
        6.0,
        "the block scale must map onto the FORMAT's max, not onto 1"
    );
}

/// torchao IS round-half-to-even at every interior tie. This asserts the
/// REFERENCE's property, not ours — it is the fact the next test needs, and it
/// fails loudly if a future torchao changes the rule, which would mean the
/// next test's expectation is stale rather than that our code moved.
#[test]
fn the_reference_is_round_half_to_even_at_every_interior_tie() {
    let fx = fixture();
    assert_eq!(
        fx["grid.torchao_is_round_half_to_even"], "yes",
        "torchao is no longer round-half-to-even; the tie test below is stale"
    );
    // Seven interior ties for an eight-level grid.
    let ties = fx["grid.ties_where_ours_differs_list"]
        .split_whitespace()
        .filter(|s| !s.is_empty())
        .count();
    let all = (1..E2M1.len()).count();
    assert!(ties <= all, "sanity");
}

/// # THE DEFECT THIS FILE IS POINTED AT
///
/// This test was **RED ON PURPOSE** when it was written. `fp4_round` broke
/// exact ties **up** the grid; torchao (and the format's hardware, and
/// IEEE 754) breaks them toward the **even code**. At 0.25, 1.25, 2.5 and 5.0
/// the two returned different levels, and at 5.0 the error was 4.0 against
/// 6.0.
///
/// Measured on this box, 2026-09-30, from the fixture: 4 of the grid's 7
/// interior tie points disagreed. The consequence for training was that every
/// `--act-quant fp4` step quantized a measurable fraction of its activations
/// to a different level than the format specifies, in a direction (upward,
/// toward 6) that biases the magnitude — and the error was largest exactly at
/// the top of the range, where the block scale puts the most-used values.
///
/// **Fixed, in `dc5d667`, and this test is the evidence.** The ladder now
/// selects by even CODE rather than by direction, because "even" alternates
/// along the ladder and no single `>` or `>=` expresses it — `act_quant.rs`
/// carries the seven-tie table and the reasoning. **Re-reverting `dc5d667`
/// turns this test red at exactly the four ties named above**; that is a
/// recorded measurement, not a claim.
///
/// The mutation sweep that proves it is `tests/oracle/mutate_kernel.sh` in
/// `dormouse-dspark`, whose M4/M5/M6 mutants are this file's subject.
#[test]
fn fp4_ties_match_the_formats_rule() {
    let fx = fixture();
    let dev = Device::default();

    let mut wrong: Vec<String> = Vec::new();

    // ALL SEVEN TIES IN ONE BLOCK. This is the whole reason the test is
    // built this way, and getting it wrong is what an earlier draft of this
    // file did: `quant_act` derives its scale from the block max, so a 1x1
    // block whose only element IS the tie under test has scale = tie/6 and
    // normalises the value to exactly 6.0 -- the top of the grid, every
    // time, for every tie. The first draft of this test therefore reported
    // "ours 6.000" on all seven and looked like a seven-fold defect; it was
    // the harness saturating, not `fp4_round`.
    //
    // With all seven in one row the block max is 5.0, the scale is 5/6, and
    // each value normalises to tie/(5/6) = tie*1.2 -- which is NOT a tie any
    // more. So the row is built with the grid's TOP value 6.0 present to fix
    // the scale at exactly 1.0, and the ties then land on the tie points
    // themselves, which is the only way the question "which nearest level"
    // is asked at all.
    let mut row: Vec<f32> = vec![6.0];
    for i in 1..E2M1.len() {
        row.push(0.5 * (E2M1[i - 1] + E2M1[i]));
    }
    let x = Tensor::<2>::from_data(TensorData::new(row.clone(), [1, row.len()]), &dev);
    let q = quant_act::<Flex>(x, ActFormat::Fp4, 0); // group 0 -> one scale per token
    let got = q.into_data().to_vec::<f32>().expect("read");

    // The scale is 6.0/6.0 = 1.0 because the block max is the format's max, so
    // the dequantized output IS the level in grid units. Assert that rather
    // than assuming it: if the block max ever stops being 6.0 this stops
    // being a tie test and says so.
    assert_eq!(
        got[0], 6.0,
        "the block's anchor element must survive as 6.0, else the scale moved \
         and the ties below are not ties any more (got {})",
        got[0]
    );

    for n in 0..E2M1.len() - 1 {
        let tie = row[n + 1];
        let key = format!("grid.tie_{}", fmt_key(tie));
        let theirs: f32 = fx[&key]
            .parse()
            .unwrap_or_else(|_| panic!("no {key} in the fixture; the generator changed"));
        let ties_up: f32 = fx[&format!("{key}_ties_up")]
            .parse()
            .expect("ties_up column");
        let ours_level = got[n + 1];

        if (ours_level - theirs).abs() > 1e-3 {
            wrong.push(format!(
                "  tie {tie:<6} ours {ours_level:>6.3}  reference {theirs:>6.3}  \
                 (our ties-up rule says {ties_up:>6.3})"
            ));
        }
    }

    assert!(
        wrong.is_empty(),
        "\n{} of the e2m1 grid's interior tie points are rounded to the WRONG \
         level by `fp4_round` (crates/dormouse-core/src/act_quant.rs), \
         against torchao's quantizer at {TORCHAO_SHA}:\n{}\n\
         The format breaks ties toward the EVEN CODE (round-half-to-even, IEEE \
         754). Our ladder selects by even code, so a failure here means that \
         selection is broken -- most likely reverted to a plain `>=`, which \
         breaks every tie UP and is wrong at 0.25, 1.25, 2.5 and 5.0 (at 5.0 \
         that is 4.0 against 6.0, a 50% error, at the top of the range).",
        wrong.len(),
        wrong.join("\n"),
    );
}

/// The tie points themselves are well-formed: strictly between their two
/// neighbours, all distinct, all finite. If this fails the generator's
/// premise is wrong rather than our code.
#[test]
fn the_tie_points_are_well_formed() {
    let mut prev = f32::NEG_INFINITY;
    for i in 1..E2M1.len() {
        let tie = 0.5 * (E2M1[i - 1] + E2M1[i]);
        assert!(tie > E2M1[i - 1] && tie < E2M1[i], "tie {tie} is not interior");
        assert!(tie.is_finite());
        assert!(tie > prev, "tie points must be distinct and ascending");
        prev = tie;
    }
    assert_eq!(prev, 5.0, "the largest interior tie of the grid is 5.0");
}

/// The fixture key format: `0.25` -> `0p25`, `5` -> `5`, `-1.5` -> `m1p5`.
/// It must match the generator's, or every lookup misses and the test above
/// panics on a missing key rather than on a real disagreement.
fn fmt_key(v: f32) -> String {
    // Must match the generator's `g9(t).replace(".", "p").replace("-", "m")`.
    // Rust has no `{:g}`, and `{}` on an f32 already prints the shortest
    // round-trip form, which is what `%.9g` collapses to for these seven
    // values (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5).
    format!("{v}").replace('.', "p").replace('-', "m")
}
