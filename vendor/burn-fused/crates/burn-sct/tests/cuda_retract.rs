//! CUDA retraction must match the CPU reference exactly (feature "cuda").
#![cfg(feature = "cuda")]
use burn::module::Param;
use burn::tensor::{Device, Distribution, Tensor};
use burn_sct::qr::qr_cpu;
use burn_sct::qr_cuda::{forward_cuda, CudaBare};
use burn_sct::{SctConfig, SctLinear};
use std::time::Instant;

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn cuda_dev() -> Device {
    Device::cuda(0)
}

#[test]
fn gpu_retract_matches_cpu() {
    let dev = cuda_dev();
    let (m, k) = (1024usize, 128usize);
    let x = Tensor::<2>::random([m, k], Distribution::Normal(0.0, 1.0), &dev);

    // GPU path through the layer (kernels, no host round-trip in between).
    let mut layer = SctLinear::new(&SctConfig::new(m, m * 2, k), &dev);
    layer.u = Param::from_tensor(x.clone());
    layer.retract::<CudaBare>();
    let gpu: Vec<f32> = layer
        .u
        .val()
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();

    // CPU reference on the same data.
    let data: Vec<f32> = x.clone().into_data().try_to_vec::<f32>().unwrap();
    let (cpu, _) = qr_cpu(&data, m, k);
    let diff = max_diff(&gpu, &cpu);
    assert!(diff < 1e-4, "GPU vs CPU retract max_diff {diff:.3e}");
    println!("gpu retract matches cpu: max_diff {diff:.3e}");
}

#[test]
fn gpu_retract_speed() {
    let dev = cuda_dev();
    let (m, n, k) = (2048usize, 8192usize, 128usize);
    let mut layer = SctLinear::new(&SctConfig::new(m, n, k), &dev);
    for _ in 0..10 {
        layer.retract::<CudaBare>(); // warm up the kernel JIT
    }
    let _ = layer.u.val().clone().into_data();
    let iters = 50;
    let t0 = Instant::now();
    for _ in 0..iters {
        layer.retract::<CudaBare>();
    }
    let _ = layer.u.val().clone().into_data(); // sync the queue once
    let t = t0.elapsed().as_secs_f64() / iters as f64;
    println!("gpu retract {m}x{n} k={k}: {:.4} ms/step", t * 1e3);
}

/// Non-pow2 widths: cubecl's pitched allocator pads 2D row strides
/// (300*4B=1200B -> pitch 1536B = 384 elems; 24*4B=96B -> pitch 128B =
/// 32 elems). The fused GEMMs take the row strides as comptime args, so
/// the raw path must agree with the tensor path instead of reading padding.
#[test]
fn gpu_forward_matches_tensor_path_non_pow2() {
    let dev = cuda_dev();
    for (in_f, out_f, rank) in [
        (300usize, 200usize, 8usize),  // in=300 non-pow2, rank pow2
        (300usize, 200usize, 24usize), // both non-pow2
        (512usize, 128usize, 12usize), // in pow2, rank=12 non-pow2
    ] {
        let layer = SctLinear::new(&SctConfig::new(in_f, out_f, rank), &dev);
        let x = Tensor::<2>::random([64, in_f], Distribution::Normal(0.0, 1.0), &dev);
        let reference = x
            .clone()
            .matmul(layer.u.val())
            .mul(layer.s.val().unsqueeze_dims(&[0]))
            .matmul(layer.v.val().transpose());
        // The raw path must RUN (not fall back) on these shapes.
        let fused =
            forward_cuda::<CudaBare>(x.clone(), layer.u.val(), layer.s.val(), layer.v.val())
                .expect("raw forward should run for non-pow2 in/rank");
        let d = (fused.clone() - reference.clone())
            .abs()
            .max()
            .into_scalar::<f32>();
        assert!(
            d < 1e-3,
            "in={in_f} out={out_f} r={rank}: fused vs tensor max {d:.3e}"
        );
        // Whole layer (public entry) agrees too.
        let y = layer.forward::<CudaBare>(x);
        let d2 = (y - reference).abs().max().into_scalar::<f32>();
        assert!(
            d2 < 1e-3,
            "in={in_f} r={rank}: layer vs tensor max {d2:.3e}"
        );
    }
}
