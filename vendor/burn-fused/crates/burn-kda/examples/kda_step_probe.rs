//! Where does a dormouse training step actually go? The KDA path, timed end
//! to end on the real module.
//!
//! `kda_bench` measured the fused FORWARD kernel at 0.23 ms on the production
//! shape, and the conclusion drawn from it was "KDA is free (0.016% of a
//! step)". That was wrong - it timed one kernel and ignored the backward. The
//! ablation says otherwise: `--no-kda` steps in 188 ms where the full model
//! takes 956 ms, so the KDA backward is ~80% of a step.
//!
//! This harness times the paths the model actually uses, on the production
//! shape (b=10, t=512, d=768, 12 heads, K=V=64, chunk 16):
//!   forward_train      - the chunked delta-rule training forward
//!   backward           - the full backward through it
//!   forward_train_fused - the fused-kernel variant of the same
//!   forward            - the inference path (what generation pays)
//!
//! Run: cargo run --release -p burn-kda --example kda_step_probe --features cuda

#[cfg(feature = "cuda")]
fn main() {
    use burn::backend::autodiff::Autodiff;
    use burn::module::Module;
    use burn::tensor::Tensor;
    use burn_cuda::Cuda;
    use burn_kda::{KdaConfig, KdaModule};

    type Ad = Autodiff<Cuda>;
    let dev = burn::tensor::Device::cuda(0);
    // pre.4 idiom (as in burn-spectral's own tests): float tensors live on the
    // default backend, created on an AUTODIFF device, with require_grad. A
    // concrete `Tensor<_, Autodiff<..>>` does not satisfy BasicOps.
    let adev = dev.clone().autodiff();

    // The `small` preset geometry.
    let (b, t, d) = (10usize, 512usize, 768usize);
    let cfg = KdaConfig {
        hidden_size: d,
        num_heads: 12,
        head_dim: 64,
        use_short_conv: false, // the preset disables it (it NaN'd on this box)
        chunk_size: 16,
        ..Default::default()
    };
    let m = KdaModule::new(&cfg, 0.0, &dev);

    let mut st = 0x243F6A88_5A30_8D3Du64;
    let mut next = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        (st % 10000) as f32 / 10000.0 - 0.5
    };
    let data: Vec<f32> = (0..b * t * d).map(|_| next()).collect();
    let x = || -> Tensor<3> {
        Tensor::from_data(burn::tensor::TensorData::new(data.clone(), [b, t, d]), &adev).require_grad()
    };

    let iters = 10;
    let warmup = 3;

    let time = |f: &mut dyn FnMut()| -> f32 {
        for _ in 0..warmup {
            f();
        }
        let t = std::time::Instant::now();
        for _ in 0..iters {
            f();
        }
        t.elapsed().as_secs_f32() * 1000.0 / iters as f32
    };

    // Backward: one graph, backwarded repeatedly is invalid (the graph is
    // consumed), so rebuild per iteration and time the pair separately.
    let mut fwd;
    let mut bwd = f32::NAN;
    {
        for _ in 0..warmup {
            let y = m.forward_train::<Ad>(x());
            let _ = y.sum().backward();
        }
        let mut fwd_ms = 0f32;
        let mut bwd_ms = 0f32;
        for _ in 0..iters {
            let xi = x();
            let t0 = std::time::Instant::now();
            let y = m.forward_train::<Ad>(xi);
            let loss = y.sum();
            fwd_ms += t0.elapsed().as_secs_f32() * 1000.0;
            let t1 = std::time::Instant::now();
            let _ = loss.backward();
            bwd_ms += t1.elapsed().as_secs_f32() * 1000.0;
        }
        fwd = fwd_ms / iters as f32;
        bwd = bwd_ms / iters as f32;
        bwd = bwd_ms / iters as f32;
    }

    #[cfg(feature = "autodiff")]
    let fused = time(&mut || {
        let _ = m.forward_train_fused::<Ad>(x());
    });
    #[cfg(not(feature = "autodiff"))]
    let fused = f32::NAN;
    let infer = time(&mut || {
        let mut st = None;
        let _ = m.forward::<Ad>(x(), &mut st, false);
    });

    println!("KDA production shape: b={b} t={t} d={d} heads=12 K=V=64 chunk=16 (iters={iters})");
    println!("  forward_train (chunked)   {fwd:9.3} ms   (from the timed fwd+bwd pair)");
    println!("  backward through it       {bwd:9.3} ms   <- the 80% of the step");
    println!("  forward_train_fused       {fused:9.3} ms");
    println!("  forward (inference path)  {infer:9.3} ms");
    println!("  params in module: {}", m.num_params());
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("kda_step_probe needs --features cuda");
}
