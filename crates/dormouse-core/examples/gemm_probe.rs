//! GEMM probe: is a bf16 tensor-core matmul actually FASTER than the fp32
//! path on this box, once the casts are paid for?
//!
//! Context: LLMQ (arXiv 2512.15306) reports on an RTX 5060 Ti - our exact
//! GPU - 3.9k tok/s at 78% MFU for 1.5B in BF16 and 13.0k at 85% for 0.5B.
//! We run 2.7k tok/s on 7.5M params = 0.12 TFLOP/s, ~325x off that. The
//! candidates for the gap are (a) no tensor cores at all (fp32 CUDA cores)
//! and (b) per-op overhead. This probe answers (a) with numbers instead of an
//! estimate, on the shapes the `small` preset actually multiplies
//! (b=10, t=512, d=768, d_ffn=2048, vocab=256) plus one 1B-scale FFN shape.
//!
//! Three variants per shape:
//!   f32   - the current path
//!   bf16  - bf16_matmul (bf16 GEMM on tensor cores, fp32 out, fp32 bwd)
//!   cast  - the two input casts alone, so GEMM speedup and conversion
//!           overhead can be told apart
//!
//! Sync without naming the inner device type: reading one scalar back drains
//! the ordered stream, so the timed region covers every queued kernel.
//!
//! Run: cargo run --release -p dormouse-core --example gemm_probe --features cuda

#[cfg(feature = "cuda")]
fn main() {
    use burn::tensor::{Device, Distribution, FloatDType, Tensor};
    use burn_spectral::bf16_ops::bf16_matmul;

    type Bare = burn_cuda::Cuda;
    let device = Device::cuda(0).autodiff();
    let iters = 20;
    let warmup = 3;

    println!(
        "{:<26} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "shape", "f32 ms", "bf16 ms", "cast ms", "speedup", "f32 TF/s"
    );
    for (name, m, k, n) in [
        ("attn proj  5120x768x768", 5120usize, 768usize, 768usize),
        ("ffn up     5120x768x2048", 5120, 768, 2048),
        ("lm_head    5120x768x256", 5120, 768, 256),
        ("kda qkv    5120x768x2304", 5120, 768, 2304),
        ("1B ffn     5120x2048x8192", 5120, 2048, 8192),
    ] {
        let a: Tensor<2> =
            Tensor::random([m, k], Distribution::Normal(0.0, 1.0), &device).require_grad();
        let w: Tensor<2> =
            Tensor::random([k, n], Distribution::Normal(0.0, 0.02), &device).require_grad();

        let f32_ms = time_ms(iters, warmup, || a.clone().matmul(w.clone()));
        // bf16 tensor-core path: BROKEN on pre.4 + cuda (its own tests in
        // burn-spectral fail at burn-cubecl ops/tensor.rs:150), so it is not
        // timed here - see docs/PLAN.md OPTIMIZATION BLOCKERS. Kept behind a
        // flag so the probe documents the blocker next to the numbers.
        let bf16_ms: Option<f64> = if std::env::var("GEMM_PROBE_BF16").is_ok() {
            Some(time_ms(iters, warmup, || bf16_matmul::<Bare>(a.clone(), w.clone())))
        } else {
            None
        };
        // Conversion overhead alone (no matmul): bf16_matmul pays two input
        // casts plus one output cast, measured here as the round trips of the
        // two inputs (the smaller output cast is not included).
        let cast_a_ms = time_ms(iters, warmup, || {
            a.clone().cast(FloatDType::BF16).cast(FloatDType::F32)
        });
        let cast_w_ms = time_ms(iters, warmup, || {
            w.clone().cast(FloatDType::BF16).cast(FloatDType::F32)
        });
        let cast_ms = cast_a_ms + cast_w_ms;

        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        println!(
            "{:<26} {:>9.3} {:>9.3} {:>9.3} {:>9} {:>9.1}",
            name,
            f32_ms,
            bf16_ms.unwrap_or(f64::NAN),
            cast_ms,
            match bf16_ms { Some(b) => format!("{:.2}x", f32_ms / b), None => "broken".into() },
            flops / (f32_ms * 1e-3) / 1e12
        );
    }
}

/// Mean ms per call. The final `into_scalar` drains the stream, so the timed
/// region really covers every kernel the loop queued.
#[cfg(feature = "cuda")]
fn time_ms<F>(iters: usize, warmup: usize, mut f: F) -> f64
where
    F: FnMut() -> burn::tensor::Tensor<2>,
{
    for _ in 0..warmup {
        std::hint::black_box(f());
    }
    let t = std::time::Instant::now();
    let mut last = f();
    for _ in 1..iters {
        last = f();
    }
    let v: f32 = last.sum().try_into_scalar().unwrap_or(0.0);
    std::hint::black_box(v);
    t.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("gemm_probe needs --features cuda (it measures the tensor cores)");
}
