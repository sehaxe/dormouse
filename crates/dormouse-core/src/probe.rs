//! Execution counters: what RAN, not what a config declared.
//!
//! ADR-0011's loud-failures doctrine: a mechanism that silently does not run
//! is the cardinal sin here, and the shape of the bug is always the same - a
//! branch is taken, a fallback computes the same function, and the run is
//! correct but a year slower. The fused gated-delta kernels were dead behind
//! exactly such a gate. Numbers in a TOML file cannot catch that; a counter
//! can.
//!
//! Plain integers, incremented at the branch, thread-local so a parallel test
//! run cannot cross-contaminate. No device tensor, no sync, no abstraction
//! layer: `note` at the branch, `count` in the test.

use std::cell::Cell;

/// One loop iteration's body was entered.
pub const ITER: usize = 0;
/// The shared attention arm (KDA) was entered.
pub const KDA: usize = 1;
/// The memory branch was entered.
pub const ENGRAM: usize = 2;
/// ... and actually read a row (entered with keys, not inert).
pub const ENGRAM_KEYS: usize = 3;
/// The MoR router scored the iteration slots.
pub const MOR: usize = 4;
/// Gated Residual read + write.
pub const GR: usize = 5;
/// Activation quantization was applied to the FFN input.
pub const ACT_QUANT: usize = 6;
/// The JEPA term was added.
pub const JEPA: usize = 7;
/// The DSpark term was added.
pub const DSPARK: usize = 8;
/// The MoR router BCE was added.
pub const MOR_BCE: usize = 9;
pub const N_ARMS: usize = 10;
/// Arm name per index, for assertion messages that name the thing.
pub const NAMES: [&str; N_ARMS] = [
    "iterations",
    "kda",
    "engram",
    "engram_keys",
    "mor_router",
    "gated_residual",
    "act_quant",
    "jepa",
    "dspark",
    "mor_bce",
];

thread_local! {
    static COUNTS: [Cell<u64>; N_ARMS] = [const { Cell::new(0) }; N_ARMS];
}

/// Count one entry into `arm`. Host-side, branch-local, never on device.
#[inline]
pub fn note(arm: usize) {
    COUNTS.with(|c| c[arm].set(c[arm].get() + 1));
}

/// How many times `arm` was entered on this thread since the last [`reset`].
pub fn count(arm: usize) -> u64 {
    COUNTS.with(|c| c[arm].get())
}

/// Zero every counter on this thread.
pub fn reset() {
    COUNTS.with(|c| c.iter().for_each(|x| x.set(0)));
}

/// `(name, count)` for every arm, zeroed ones included - a printout that shows
/// the arms that did NOT run is the point.
pub fn counts() -> Vec<(&'static str, u64)> {
    COUNTS.with(|c| NAMES.iter().zip(c.iter()).map(|(n, x)| (*n, x.get())).collect())
}
