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
/// Attention Residuals replaced the residual accumulation (arXiv:2603.15031):
/// one entry per loop iteration that ran the softmax-over-depth aggregation
/// instead of the ReZero write. COUNTED, not implied: a config that says
/// `use_attnres = true` and a loop that never reaches the arm are the same
/// run as far as the loss curve is concerned, and nothing else on the eval
/// line would say so.
pub const ATTNRES: usize = 10;
/// TSCT retraction, per-factor arm: one host-syncing `polar_orthogonalize`
/// per factor. This is what ran in production until 2026-09-29 and it is
/// still the default.
pub const RETRACT_FACTOR: usize = 11;
/// TSCT retraction, grouped arm: `burn_spectral::retract_batched`, sync-free
/// per shape group (`--retract-batched`). Counted because a fallback here is
/// invisible in every other number: the batched arm computes the same
/// matrices, so a run that silently took the slow arm prints the same loss.
pub const RETRACT_BATCHED: usize = 12;
/// The future-byte auxiliary term was COMPUTED and added. Distinct from
/// [`FUTURE_BYTE_ASKED`] because a horizon at or past the sequence length
/// leaves zero valid positions: the arm is asked for, produces a real zero,
/// and the printed `fb=<ran>/<asked>` is the only place a reader learns a
/// 2k-step A/B was 2k steps of nothing.
pub const FUTURE_BYTE: usize = 13;
/// The future-byte arm was ENTERED with a non-zero weight. The `asked` half of
/// `fb=<ran>/<asked>`; ADR-0019 COUNTED, so `0/<n>` on the eval line means the
/// head was never reached rather than "the arm is off".
pub const FUTURE_BYTE_ASKED: usize = 14;
pub const N_ARMS: usize = 15;
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
    "attnres",
    "retract_factor",
    "retract_batched",
    "future_byte",
    "future_byte_asked",
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
