//! The sanity ceiling's arithmetic, pinned.
//!
//! The 0.22 corruption produces FINITE garbage (1e3..1e38), which the NaN
//! firewall cannot see. The ceiling is the guard for that class, so its one
//! property matters: it must sit far above anything a healthy run reaches and
//! far below anything the corruption produces. Both sides are arithmetic here,
//! not judgement: uniform over 256 byte classes is ln(256) nats, and the
//! smallest garbage value ever observed on this box was 1e3.

use dormouse_train::{is_corrupt_ce, CE_CEILING};

#[test]
fn the_ceiling_sits_between_a_healthy_run_and_the_garbage() {
    // A healthy model starts near uniform: ln(256) = 5.545 nats, plus the
    // initialization's own excess. The ceiling has to clear a 4x-LR stress
    // spike and stay absurd to a real loss - and, separately, it must sit below
    // the SMALLEST corruption this box has produced, which is 206 (2026-10-10,
    // the bench200 batch) with 234 next. 100 is 18x uniform and half of 206, so
    // both sides hold with a factor to spare, which is the only honest way to
    // pick a single constant from two different populations.
    let uniform = 256.0f64.ln() as f32;
    let smallest_observed_corruption = 206.0f32;
    assert!(
        CE_CEILING > 10.0 * uniform,
        "the ceiling must clear a stress spike: {CE_CEILING} vs {uniform}"
    );
    assert!(
        CE_CEILING < smallest_observed_corruption / 2.0,
        "the ceiling must sit below the smallest observed corruption: {CE_CEILING}"
    );
}

#[test]
fn only_finite_values_above_the_ceiling_are_a_violation() {
    // The NaN firewall owns the non-finite class, so this guard must not claim
    // it: a NaN step is a masked step, not a corrupted-run stop.
    assert!(!is_corrupt_ce(f32::NAN));
    assert!(!is_corrupt_ce(f32::INFINITY));
    assert!(!is_corrupt_ce(f32::NEG_INFINITY));
    assert!(!is_corrupt_ce(5.57));
    assert!(!is_corrupt_ce(10.0 * 5.545));
    // and the observed boundary itself is a violation, not a grey area
    assert!(is_corrupt_ce(206.0));
    assert!(!is_corrupt_ce(100.0));
    assert!(is_corrupt_ce(100.001));
    assert!(is_corrupt_ce(1e8));
    assert!(is_corrupt_ce(1e38));
}
