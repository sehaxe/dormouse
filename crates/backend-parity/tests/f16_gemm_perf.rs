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
//! performance gate.** Pre-patch numbers, quiet card, through burn's `matmul`
//! at `[5120,2048]x[2048,8192]`:
//!
//! | | ms | TFLOP/s | how |
//! |---|---|---|---|
//! | fp32 | 16.60 | 10.35 | `fp32_gemm_baseline`, run |
//! | f16 | 20.02 | 8.58 | `f16_gemm_speed`, run - and it logged the `SizedType` panic while doing it |
//!
//! So on this stack, unpatched, f16 is not merely "un-accelerated", it is
//! **slower than fp32** - a fallback doing the same work in the wrong dtype.
//! (The 47.7 ms fp32 figure quoted in the original task brief does not
//! reproduce here: the same shape through the same call came back at 16.60 ms.)
//!
//! The patch cannot deliver a speedup by itself. It removes the reason the
//! tensor-core CANDIDATES never compiled; whether a candidate then WINS the
//! autotune, and how close it lands to cuBLAS, is what running these tests
//! answers.
//!
//! ## Reading a run
//!
//! Three signals decide "accelerated vs silent fallback", in order of trust:
//!
//! 1. **TFLOP/s** - the f16 arm has to clearly beat the fp32 arm on the same
//!    shape. Pre-patch it LOST (20.02 vs 16.60 ms).
//! 2. **stderr** - a candidate that dies at compile time now leaves a
//!    `autotune: candidate '...' failed and was skipped` warn (vendored
//!    cubecl-runtime `Candidate::fail`, ADR-0019 COUNTED) and/or the original
//!    `Expected type builtin.fp16 to implement dyn SizedType` panic text. A
//!    test binary installs no logger, so the file installs a stderr one at
//!    Warn level. Run with `RUST_LOG=warn` or not at all - the logger is
//!    unconditional.
//! 3. **The poisoned-runner guard** - a TFLOP/s ceiling (60 f16 / 25 fp32;
//!    cuBLAS on this card measured 43.7/13.2), because an unpatched f16 warmup
//!    panic once made later ops report 0.04 ms. The first version of this
//!    guard was a wall-clock floor (`ms > 1.0`) written pre-patch; it panicked
//!    on the genuine 0.474 ms win the day the patch worked - a gate that
//!    fails on success is the same defect as one that cannot fail.
//!
//! ## Shapes
//!
//! * `PROD` = 5120x2048x8192 - the big shape both pre-patch rows above were
//!   taken on; big enough that launch overhead cannot hide a tensor-core win.
//! * `MODEL` = 5120x768x2048 - the d_model=768 shape the cuBLAS comparison was
//!   taken on (`docs/research/2026-09-27-cublas-integration-poc.md`: burn fp32
//!   7.1, burn f16 5.6, cuBLAS f16-in/fp32-acc 41.4 TFLOP/s, same GPU, same
//!   process). That 5.6 is the number to beat.
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
//! Method, and the things that make the numbers mean something:
//!
//! * **Drain inside the timed region.** `Device::cuda` is asynchronous (the
//!   cubecl server owns its own stream and a thread), so timing the enqueue
//!   alone measures launch overhead and every kernel looks the same. N
//!   back-to-back matmuls are enqueued and one cheap reduction plus a 4-byte
//!   scalar read closes the region, so the number covers execution, not
//!   submission.
//! * **A cheap drain, not `into_data()`.** The big shape's output is
//!   5120x8192, so a full D2H of it is 168 MB - roughly 15 ms of PCIe, a third
//!   of the fp32 baseline, added to both arms. `sum().into_scalar()` is a
//!   device reduction and a 4-byte read.
//! * **The figure is the MINIMUM of three repeats**: the robust estimator when
//!   something else may be touching the GPU, and the only one that does not
//!   punish a throughput number for an unrelated process.
//! * **A step-indexed reading rule carries over from the step-time incident**
//!   (AGENTS.md 3.1): the warmup outside the clock is where JIT + autotune
//!   land; these are warm numbers by construction.

#![cfg(feature = "cuda")]

use burn::tensor::{Device, FloatDType, Tensor, TensorData};
use std::time::Instant;

/// The big shape, and the one both pre-patch rows in the module docs were taken on.
const PROD: (usize, usize, usize) = (5120, 2048, 8192);

/// The d_model=768 shape the cuBLAS 41.4 / burn-f16 5.6 comparison was taken on.
const MODEL: (usize, usize, usize) = (5120, 768, 2048);

const ITERS: usize = 10;

/// cubecl's autotune fallback is a `log::warn!` (vendored cubecl-runtime
/// `Candidate::fail`); a test binary installs no logger, so without this the
/// fallback line goes nowhere and a fallback looks like an ordinary slow win.
struct WarnLogger;

impl log::Log for WarnLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Warn
    }
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            eprintln!("[cubecl {}] {}", record.level(), record.args());
        }
    }
    fn flush(&self) {}
}

fn install_warn_logger() {
    // `set_boxed_logger` sits behind log's `alloc` feature, which the resolved
    // feature set does not turn on; the static form needs no allocator.
    static WARN_LOGGER: WarnLogger = WarnLogger;
    let _ = log::set_logger(&WARN_LOGGER);
    log::set_max_level(log::LevelFilter::Warn);
}

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

fn operands(
    device: &Device,
    (m, k, n): (usize, usize, usize),
) -> (Tensor<2>, Tensor<2>, Tensor<2>, Tensor<2>) {
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

fn tflops((m, k, n): (usize, usize, usize), ms: f64) -> f64 {
    2.0 * m as f64 * k as f64 * n as f64 / (ms * 1e-3) / 1e12
}

/// One dtype, one shape, one clock. Returns the ms of the best repeat.
///
/// The f16 arm first checks correctness against the fp32 result of the same
/// product (f16 eps is 2^-11 = 4.9e-4, so 1e-2 relative is the honest band -
/// the same band `backend_parity.rs` uses), because a wrong answer is not a
/// fast one.
fn run_arm(shape: (usize, usize, usize), dtype: &str) -> f64 {
    install_warn_logger();
    let device = Device::cuda(0);
    let (a, b, a16, b16) = operands(&device, shape);
    let (m, k, n) = shape;

    if dtype == "f16" {
        let want: f32 = a.clone().matmul(b.clone()).sum().into_scalar();
        let got: f32 = a16
            .clone()
            .matmul(b16.clone())
            .cast(FloatDType::F32)
            .sum()
            .into_scalar();
        assert!(
            got.is_finite()
                && want.is_finite()
                && (got - want).abs() <= 1e-2 * want.abs().max(1.0),
            "f16 GEMM is not within 1e-2 of the fp32 result: {got} vs {want}"
        );
    }

    let (lhs, rhs) = if dtype == "f16" { (a16, b16) } else { (a, b) };

    // Warmup outside the clock: JIT + autotune land here. This is also where a
    // dying candidate prints its warn/panic - read stderr, not just the number.
    std::hint::black_box(
        lhs.clone()
            .matmul(rhs.clone())
            .sum()
            .into_scalar::<f32>(),
    );

    let ms = (0..3)
        .map(|_| ms_of(|| lhs.clone().matmul(rhs.clone())))
        .fold(f64::INFINITY, f64::min);

    println!(
        "\n== {dtype} {m}x{k}x{n} ({ITERS} iters/repeat, min of 3) ==\n  {ms:.2} ms  {:.2} TFLOP/s",
        tflops(shape, ms)
    );

    // The poisoned-runner guard, stated as a physical ceiling rather than a
    // wall-clock floor. The old form (`ms > 1.0`) was written pre-patch, when
    // f16 was the SLOW arm; the day the patch worked it panicked on a genuine
    // 0.474 ms / 34 TFLOP/s - a gate that fails on success. The ceilings:
    // cuBLAS on this GPU measured 43.7 TFLOP/s f16 and 13.2 fp32, and burn
    // has never beaten cuBLAS, so anything above 60/25 is a broken runner
    // reporting without executing (the f16 correctness assert above is the
    // other half: a poisoned runner returns a wrong answer and dies there).
    let ceiling = if dtype == "f16" { 60.0 } else { 25.0 };
    let rate = tflops(shape, ms);
    assert!(ms.is_finite() && ms > 0.0, "no {dtype} time was measured");
    assert!(
        rate < ceiling,
        "implausible {rate:.1} TFLOP/s (ceiling {ceiling}) - the device runner is broken"
    );
    ms
}

/// The big-shape fp32 baseline. Pre-patch: 16.60 ms / 10.35 TFLOP/s (module docs).
#[test]
#[ignore = "GPU measurement; run with --ignored --exact on a free card"]
fn fp32_gemm_baseline() {
    run_arm(PROD, "fp32");
}

/// The big-shape f16 arm - the test that decides whether the patch is real.
/// Pre-patch: 20.02 ms / 8.58 TFLOP/s, i.e. LOST to fp32, with the SizedType
/// panic in the log (module docs). Accelerated would be a clear win over the
/// sibling baseline, toward the cuBLAS ceiling of ~3.9 ms / 43.7 TFLOP/s.
#[test]
#[ignore = "GPU measurement; run with --ignored --exact on a free card"]
fn f16_gemm_speed() {
    run_arm(PROD, "f16");
}

/// The model-shaped fp32 baseline. Pre-patch reference for the same shape:
/// 7.1 TFLOP/s (`docs/research/2026-09-27-cublas-integration-poc.md`).
#[test]
#[ignore = "GPU measurement; run with --ignored --exact on a free card"]
fn fp32_model_shape_baseline() {
    run_arm(MODEL, "fp32");
}

/// The model-shaped f16 arm. Unpatched this shape measured 5.6 TFLOP/s - the
/// number to beat (module docs). cuBLAS f16-in/fp32-acc does 41.4 on this
/// shape; that is the ceiling, not the claim.
#[test]
#[ignore = "GPU measurement; run with --ignored --exact on a free card"]
fn f16_model_shape_speed() {
    run_arm(MODEL, "f16");
}
