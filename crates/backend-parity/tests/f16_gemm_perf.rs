//! Does the `SizedType`-for-`FP16Type` patch actually buy tensor cores, or only
//! a green test? (ADR-0016 bug 3.)
//!
//! The defect: pliron's `builtin.fp16` implements `FloatTypeInterface` and not
//! the `SizedType` that the shared-memory sizing queries ask for
//! (`cubecl-opt/src/lib.rs:36-40` -> `SmemAllocation::end`). An f16 matmul is
//! the one matmul that stages f16 tiles, so f16 is the one element type that
//! walks into that query - and it panicked. The autotuner catches a failed
//! candidate and falls back to a non-accelerated routine, so the answer was
//! right and the cost was invisible.
//!
//! The patch (`vendor/cubecl-fix/cubecl-ir/src/types/scalar.rs`) is ten lines
//! and its own unit test in that crate is the deterministic gate: a
//! `FP16Type` handle must answer `size`/`align` with 2, the same numbers
//! `Float16Type` already reported. That test is green. **This file is the
//! performance gate and it has NOT been run against the patch** - the GPU was
//! needed for a multi-hour training run, so both tests below are `#[ignore]`d
//! and the patch is UNVERIFIED on hardware. Nothing in this file may be cited
//! as a speedup until someone runs it.
//!
//! ## What is measured, and what is only projected
//!
//! Pre-patch, on a quiet card, through burn's `matmul` at `[5120,2048]x[2048,8192]`:
//!
//! | | ms | TFLOP/s | how |
//! |---|---|---|---|
//! | fp32 | 16.60 | 10.35 | `fp32_gemm_baseline`, run |
//! | f16 | 20.02 | 8.58 | `f16_gemm_speed`, run - and it logged the `SizedType` panic while doing it |
//!
//! So on this stack, unpatched, f16 is not merely "un-accelerated", it is
//! **slower than fp32** - a fallback doing the same work in the wrong dtype.
//!
//! Two cautions on those numbers, because they are the only two and they are
//! load-bearing:
//!
//! * The f16 run is indicative, not clean: the candidate-compile panic is in
//!   the log, which is the defect itself, and a process that has panicked its
//!   device runner once already gave a nonsense 0.04 ms / 3900 TFLOP/s for fp32
//!   in an earlier run. Re-measure both arms after the patch.
//! * The 47.7 ms fp32 figure quoted in the task brief does **not** reproduce
//!   here: the same shape through the same call came back at 16.60 ms. Either
//!   the brief's number is a different shape or a different build; the
//!   projection below uses neither, it uses these two rows.
//!
//! ## The projection, and it is a projection
//!
//! cuBLAS f16-in/fp32-accumulate measures 43.7 TFLOP/s on this GPU
//! (`.bulba/memory.md:17`). This shape is 2*5120*2048*8192 = 1.718e11 FLOP, so
//! that ceiling is 3.93 ms, and against the 16.60 ms fp32 baseline a
//! *hypothetical* f16 path that reached the cuBLAS ceiling would be **4.2x**.
//! Against the brief's 47.7 ms it would be **12.1x**.
//!
//! The patch cannot deliver either number by itself. It removes the reason the
//! tensor-core CANDIDATES never compiled; whether a candidate then WINS the
//! autotune, and how close it lands to 43.7, is what the deferred run answers.
//! A number in the 4-12x range is the projection, not a result.
//!
//! ## One dtype per process, and that is not a style choice
//!
//! Unpatched, the f16 warmup panics the cubecl device runner, the client is
//! left broken, and every op queued afterwards returns without executing -
//! which is how a first version of this file reported fp32 at 0.04 ms. So each
//! dtype is its own `#[test]`, and the run passes `--exact` to select one:
//!
//! ```
//! cargo test --release -p backend-parity --features cuda --test f16_gemm_perf \
//!   -- --nocapture --ignored --exact fp32_gemm_baseline
//! cargo test --release -p backend-parity --features cuda --test f16_gemm_perf \
//!   -- --nocapture --ignored --exact f16_gemm_speed
//! ```
//!
//! Method, and the two things that make the numbers mean something:
//!
//! * **Drain inside the timed region.** `Device::cuda` is asynchronous (the
//!   cubecl server owns its own stream and a thread), so timing the enqueue
//!   alone measures launch overhead and every kernel looks the same. N
//!   back-to-back matmuls are enqueued and one cheap reduction plus a 4-byte
//!   scalar read closes the region, so the number covers execution, not
//!   submission.
//! * **A cheap drain, not `into_data()`.** The output is 5120x8192, so a full
//!   D2H of it is 168 MB - roughly 15 ms of PCIe, a third of the fp32 baseline,
//!   added to both arms. `sum().into_scalar()` is a device reduction and a
//!   4-byte read.
//!
//! The warmup runs before the clock, which is where JIT and autotune land. The
//! figure is the MINIMUM of three repeats: the robust estimator when something
//! else may be touching the GPU, and the only one that does not punish a
//! throughput number for an unrelated process.

#![cfg(feature = "cuda")]

use burn::tensor::{Device, FloatDType, Tensor, TensorData};
use std::time::Instant;

/// The production shape, and the one the 47.7 ms fp32 baseline was taken on.
const PROD: (usize, usize, usize) = (5120, 2048, 8192);

const ITERS: usize = 10;

/// Deterministic host data, so the two dtypes see identical bits and the two
/// processes are comparable. A LCG rather than `Tensor::random` because the f16
/// result is checked against the fp32 result of the *same* inputs.
fn lcg(n: usize, scale: f32) -> Vec<f32> {
    let mut s = 0x2545_f491u32;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / 8_388_608.0 - 1.0) * scale
        })
        .collect()
}

fn operands(device: &Device) -> (Tensor<2>, Tensor<2>, Tensor<2>, Tensor<2>) {
    let (m, k, n) = PROD;
    let a = Tensor::<2>::from_data(TensorData::new(lcg(m * k, 1.0), [m, k]), device);
    let b = Tensor::<2>::from_data(TensorData::new(lcg(k * n, 0.05), [k, n]), device);
    let a16 = a.clone().cast(FloatDType::F16);
    let b16 = b.clone().cast(FloatDType::F16);
    (a, b, a16, b16)
}

/// Enqueue `ITERS` back-to-back matmuls, then drain and stop the clock.
fn ms_of(f: impl Fn() -> Tensor<2>) -> f64 {
    // One call before the clock: the first-touch allocation would otherwise
    // land in iteration 1 alone. Its own drain, so the value is not the one
    // the timed loop consumes.
    let warm = f();
    std::hint::black_box(warm.sum().into_scalar::<f32>());
    let t = Instant::now();
    let mut last = f();
    for _ in 1..ITERS {
        last = f();
    }
    // The drain: a device reduction plus a 4-byte D2H. It also forces every
    // queued matmul to have finished, so elapsed time is execution time.
    let s: f32 = last.sum().into_scalar();
    std::hint::black_box(s);
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn tflops(ms: f64) -> f64 {
    let (m, k, n) = PROD;
    2.0 * m as f64 * k as f64 * n as f64 / (ms * 1e-3) / 1e12
}

/// The baseline this whole claim is measured against. Separate from the f16
/// test because the unpatched f16 run breaks the device runner for the rest of
/// the process (see the module docs).
///
/// `#[ignore]`d: measured once pre-patch (16.60 ms / 10.35 TFLOP/s, recorded
/// above) and deferred after that, because the GPU was needed for a training
/// run. Re-run it to re-establish the comparison; the number is in the module
/// docs, not in an assert, precisely so it cannot go stale silently.
#[test]
#[ignore = "GPU measurement deferred: the card was needed for a training run"]
fn fp32_gemm_baseline() {
    let device = Device::cuda(0);
    let (a, b, ..) = operands(&device);
    // One call outside the clock pays JIT + autotune.
    std::hint::black_box(a.clone().matmul(b.clone()).sum().into_scalar::<f32>());
    let ms = (0..3)
        .map(|_| ms_of(|| a.clone().matmul(b.clone())))
        .fold(f64::INFINITY, f64::min);
    let (m, k, n) = PROD;
    println!("\n== fp32 {m}x{k}x{n} ({ITERS} iters/repeat, min of 3) ==\n  {ms:.2} ms  {:.2} TFLOP/s", tflops(ms));
    assert!(ms.is_finite() && ms > 0.0, "no fp32 baseline was measured");
    // Same poisoned-runner guard as the f16 arm: a broken client reports
    // microseconds, which is a measurement of nothing.
    assert!(ms > 1.0, "implausibly fast ({ms:.3} ms) - the device runner is broken");
}

/// f16 has to be both CORRECT and FASTER, and before the patch it was correct
/// and SLOWER than fp32 (20.02 vs 16.60 ms). The correctness half of that is
/// already gated for the small shapes by `backend_parity.rs`; this is the
/// production shape and the clock.
///
/// f16 eps is 2^-11 = 4.9e-4, so 1e-2 relative on a K=2048 dot is the honest
/// band - the same band `backend_parity.rs` uses.
///
/// `#[ignore]`d: **this is the test that decides whether the patch is real, and
/// it has not been run against the patch.** A green unit test in cubecl-ir
/// proves the type answers the size query; it does not prove a tensor-core
/// candidate now compiles, wins the autotune, or is fast. Unpatched, the f16
/// run also panics the device runner, so a clean run of this is itself part of
/// the evidence.
#[test]
#[ignore = "GPU measurement deferred: the card was needed for a training run"]
fn f16_gemm_speed() {
    let device = Device::cuda(0);
    let (a, b, a16, b16) = operands(&device);

    // Correctness, on the production shape, against the fp32 result of the
    // same product - before the clock, so a wrong answer is not reported as a
    // fast one.
    let want: f32 = a
        .clone()
        .matmul(b.clone())
        .sum()
        .into_scalar();
    let got: f32 = a16
        .clone()
        .matmul(b16.clone())
        .cast(FloatDType::F32)
        .sum()
        .into_scalar();
    assert!(
        got.is_finite() && want.is_finite() && (got - want).abs() <= 1e-2 * want.abs().max(1.0),
        "f16 GEMM is not within 1e-2 of the fp32 result: {got} vs {want}"
    );

    // Warmup: JIT + autotune.
    std::hint::black_box(a16.clone().matmul(b16.clone()).sum().into_scalar::<f32>());
    let ms = (0..3)
        .map(|_| ms_of(|| a16.clone().matmul(b16.clone())))
        .fold(f64::INFINITY, f64::min);
    let (m, k, n) = PROD;
    println!("\n== f16 {m}x{k}x{n} ({ITERS} iters/repeat, min of 3) ==\n  {ms:.2} ms  {:.2} TFLOP/s", tflops(ms));

    // The comparison is against the sibling test's fp32 number, 16.60 ms as
    // measured pre-patch (module docs). It is deliberately NOT an assert: a
    // hardcoded time is a flaky test on a shared GPU and a wrong one on a
    // different card, and this file's job is to produce the number for a human
    // to compare, not to encode a guess made before the card was free. What it
    // does assert is that the clock produced a real measurement at all - a
    // poisoned device runner reports microseconds.
    assert!(ms.is_finite() && ms > 0.0, "no f16 time was measured");
    assert!(ms > 1.0, "implausibly fast ({ms:.3} ms) - the device runner is broken");
}
