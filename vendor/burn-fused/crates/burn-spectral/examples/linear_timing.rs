//! SpectralLinear fused training kernels vs dense at real sizes: forward and
//! forward+backward wall-clock at B=16384, plus per-kernel ms (TSCT_TIMING=1).
//! Run: cargo run --release -p burn-spectral --features cuda --example linear_timing
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Distribution, Tensor};
use burn_spectral::SpectralLinear;

fn bench(label: &str, m: usize, n: usize, k: usize, b: usize, iters: usize) {
    let dev = Device::cuda(0).autodiff();
    let mut s = SpectralLinear::new(m, n, k, &dev);
    let d = LinearConfig::new(m, n).init(&dev);
    let x = Tensor::<2>::random([b, m], Distribution::Normal(0.0, 1.0), &dev);
    // warmup both paths, drained: the one-time kernel JIT compile must not
    // land inside the timed loops
    let y = s.forward(x.clone());
    let g = y.powf_scalar(2.0).sum().backward();
    let _: f32 = s.u.grad(&g).unwrap().sum().into_scalar();
    let _ = d.forward(x.clone());

    let t = std::time::Instant::now();
    for _ in 0..iters {
        let y = s.forward(x.clone());
        let _: f32 = y.sum().into_scalar();
    }
    let fused_fwd = t.elapsed().as_secs_f64() / iters as f64;

    let t = std::time::Instant::now();
    for _ in 0..iters {
        let y = s.forward(x.clone());
        let grads = y.powf_scalar(2.0).sum().backward();
        let _: f32 = s.u.grad(&grads).unwrap().sum().into_scalar();
    }
    let fused_fb = t.elapsed().as_secs_f64() / iters as f64;

    s.set_fused(false);
    let t = std::time::Instant::now();
    for _ in 0..iters {
        let y = s.forward(x.clone());
        let _: f32 = y.sum().into_scalar();
    }
    let old_fwd = t.elapsed().as_secs_f64() / iters as f64;
    s.set_fused(true);

    let t = std::time::Instant::now();
    for _ in 0..iters {
        let y = d.forward(x.clone());
        let _: f32 = y.sum().into_scalar();
    }
    let dense = t.elapsed().as_secs_f64() / iters as f64;

    let t = std::time::Instant::now();
    for _ in 0..iters {
        let y = d.forward(x.clone());
        let grads = y.powf_scalar(2.0).sum().backward();
        let _: f32 = d.weight.grad(&grads).unwrap().sum().into_scalar();
    }
    let dense_fb = t.elapsed().as_secs_f64() / iters as f64;

    println!("== {label}: m={m} k={k} n={n} B={b} ==");
    println!(
        "  fused fwd:     {:8.4} ms   fused fwd+bwd: {:8.4} ms",
        fused_fwd * 1e3,
        fused_fb * 1e3
    );
    println!(
        "  dense fwd:     {:8.4} ms   dense fwd+bwd: {:8.4} ms",
        dense * 1e3,
        dense_fb * 1e3
    );
    println!(
        "  old path fwd:  {:8.4} ms (TSCT tensor ops)",
        old_fwd * 1e3
    );
    println!(
        "  fused fwd {:.2}x faster than dense; fused fwd+bwd {:.2}x faster than dense fwd",
        dense / fused_fwd,
        dense / fused_fb
    );
    if std::env::var("TSCT_TIMING").map_or(false, |v| v == "1") {
        // drain everything since process start: the list below must cover
        // exactly the one measured step (8 kernels, not ~21 accumulated)
        burn_spectral::fused::clear_step_timing();
        let y = s.forward(x.clone());
        let _ = y.powf_scalar(2.0).sum().backward();
        let times = burn_spectral::fused::take_step_timing();
        let total: f32 = times.iter().map(|(_, ms)| ms).sum();
        for (name, ms) in &times {
            println!("    kernel {name}: {ms:.4} ms");
        }
        println!("    kernels total (fwd+bwd): {total:.4} ms");
    }
}

fn main() {
    let iters = 10;
    let b = 16384usize;
    bench("plan dims", 512, 4096, 32, b, iters);
    bench("mission dims", 768, 3072, 32, b, iters);
}
