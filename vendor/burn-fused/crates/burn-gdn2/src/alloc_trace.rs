//! Per-site device-allocation accounting for the chunked WY paths.
//!
//! Off by default: `note` is a relaxed atomic load and returns immediately, so
//! an untraced run pays one predictable-branch per site. Set
//! `GDN2_ALLOC_TRACE=1` to record `(site, bytes)` for every fresh buffer the
//! chunked forward/backward materialises, then `dump()` the table.
//!
//! The bytes come from the shapes of the tensors actually allocated, so the
//! table is a measurement, not a hardcoded model of the code. Cross-check it
//! against the delta of `cubecl_runtime::Client::memory_usage()` (measured in
//! `tests/alloc_probe.rs`) — the two must agree to within the few buffers the
//! fused kernels allocate outside the traced sites.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::sync::OnceLock;

static ENABLED: AtomicBool = AtomicBool::new(false);
static SITES: OnceLock<Mutex<Vec<(&'static str, u64)>>> = OnceLock::new();
/// Chunk-loop iterations actually EXECUTED, not logically needed. Under
/// `BalancedCheckpointing` burn's retro-forward re-runs the ops whose saved
/// values it dropped, and every op of the chunk loop is ours — so sampling
/// this after the forward and again after the backward is a direct read of
/// "did the backward re-execute the forward, and how many times".
static CHUNK_ITERATIONS: AtomicU64 = AtomicU64::new(0);

/// Strided views copied into a row-major buffer so the fused kernels can read
/// them linearly. The trainer's KDA inputs are all permutes, so this is the
/// normal path, not an anomaly — and a copy is a real cost, so it is counted.
static CONTIG_COPIES: AtomicU64 = AtomicU64::new(0);
/// Materializations that did NOT come back row-major (a pitched allocation).
/// A fallback, so it is counted rather than silent.
static CONTIG_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// One strided input materialized into a contiguous buffer.
#[inline]
pub fn note_contiguous_copy() {
    CONTIG_COPIES.fetch_add(1, Ordering::Relaxed);
}

/// One input that could not be made row-major: the caller fell back.
#[inline]
pub fn note_contiguous_fallback() {
    CONTIG_FALLBACKS.fetch_add(1, Ordering::Relaxed);
}

/// `(materialized, fell_back)` since [`reset_contiguous`].
pub fn contiguous_copies() -> (u64, u64) {
    (
        CONTIG_COPIES.load(Ordering::Relaxed),
        CONTIG_FALLBACKS.load(Ordering::Relaxed),
    )
}

/// Zero both contiguous counters.
pub fn reset_contiguous() {
    CONTIG_COPIES.store(0, Ordering::Relaxed);
    CONTIG_FALLBACKS.store(0, Ordering::Relaxed);
}

/// One executed iteration of the tensor-ops chunk loop.
#[inline]
pub fn chunk_iteration() {
    CHUNK_ITERATIONS.fetch_add(1, Ordering::Relaxed);
}

/// Executed chunk-loop iterations so far.
pub fn chunk_iterations() -> u64 {
    CHUNK_ITERATIONS.load(Ordering::Relaxed)
}

/// Reset the iteration counter.
pub fn reset_iterations() {
    CHUNK_ITERATIONS.store(0, Ordering::Relaxed);
}

/// `(declines, last reason)` since [`reset_batched_declined`] — the count of
/// calls the batched chunk arm could not express (chunk_size > TILE, or a
/// caller-supplied `m_invs`), with the reason of the most recent one. COUNTED,
/// not silent: a reader asking whether the batched arm ran should not have to
/// infer it from the config.
static BATCHED_DECLINED: AtomicU64 = AtomicU64::new(0);
static BATCHED_REASON: Mutex<&'static str> = Mutex::new("");

#[inline]
pub fn note_batched_declined(why: &'static str) {
    BATCHED_DECLINED.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut slot) = BATCHED_REASON.lock() {
        *slot = why;
    }
}

pub fn batched_declined() -> (u64, &'static str) {
    (
        BATCHED_DECLINED.load(Ordering::Relaxed),
        BATCHED_REASON.lock().map(|s| *s).unwrap_or(""),
    )
}

pub fn reset_batched_declined() {
    BATCHED_DECLINED.store(0, Ordering::Relaxed);
}

fn sites() -> &'static Mutex<Vec<(&'static str, u64)>> {
    SITES.get_or_init(|| Mutex::new(Vec::new()))
}

/// `true` once `GDN2_ALLOC_TRACE` has been seen. The env read happens at most
/// once, so the hot-path check is a relaxed atomic load.
pub fn enabled() -> bool {
    if ENABLED.load(Ordering::Relaxed) {
        return true;
    }
    if std::env::var("GDN2_ALLOC_TRACE").is_ok() {
        ENABLED.store(true, Ordering::Relaxed);
        true
    } else {
        false
    }
}

/// Bytes of a tensor's payload (4 bytes/element, the only dtype these paths
/// run in; the fused kernels are f32-only and fall back below that).
pub fn bytes_of<const D: usize>(t: &burn::tensor::Tensor<D>) -> u64 {
    let n: u64 = t.shape().dims::<D>().iter().map(|d| *d as u64).product();
    n * 4
}

/// Record that `label` allocated a fresh buffer of `bytes`.
#[inline]
pub fn note(label: &'static str, bytes: u64) {
    if !enabled() {
        return;
    }
    if let Ok(mut g) = sites().lock() {
        g.push((label, bytes));
    }
}

/// Drop everything recorded so far (call between measured regions).
pub fn reset() {
    if let Ok(mut g) = sites().lock() {
        g.clear();
    }
}

/// `(site, calls, total_bytes)` per site, sorted by total bytes descending.
pub fn table() -> Vec<(&'static str, u64, u64)> {
    let Ok(g) = sites().lock() else {
        return Vec::new();
    };
    let mut agg: Vec<(&'static str, u64, u64)> = Vec::new();
    for (label, bytes) in g.iter() {
        match agg.iter_mut().find(|(l, _, _)| l == label) {
            Some(e) => {
                e.1 += 1;
                e.2 += bytes;
            }
            None => agg.push((label, 1, *bytes)),
        }
    }
    agg.sort_by(|a, b| b.2.cmp(&a.2));
    agg
}

/// Print the per-site table to stdout.
pub fn dump(what: &str) {
    let t = table();
    let total: u64 = t.iter().map(|e| e.2).sum();
    let calls: u64 = t.iter().map(|e| e.1).sum();
    println!("── {what}: {calls} fresh buffers, {:.1} MB", total as f64 / 1e6);
    println!("{:>6} {:>10} {:>12}  site", "calls", "MB", "MB/call");
    for (label, n, bytes) in &t {
        println!(
            "{:>6} {:>10.2} {:>12.4}  {label}",
            n,
            *bytes as f64 / 1e6,
            *bytes as f64 / 1e6 / (*n).max(1) as f64
        );
    }
}
