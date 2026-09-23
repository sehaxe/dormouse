#![cfg(all(feature = "cuda", feature = "autodiff"))]
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
//! Fused chunk kernels vs tensor path: numerical equivalence on CUDA.
//! Run: cargo test --release --features "cuda" --test fused_chunk_verify -- --nocapture

use burn::backend::{Backend, NdArray};
use burn::tensor::{Device, Distribution, Tensor};
use burn_gdn2::kernel::chunk_cube::cuda::fused_chunk_forward;
use burn_gdn2::{chunk_wy_forward, CudaBare};

fn max_rel(a: &Tensor<4>, b: &Tensor<4>) -> f32 {
    let a = a.clone().into_data();
    let b = b.clone().into_data();
    let mut max_abs = 0.0f32;
    let mut scale = 0.0f32;
    for (x, y) in a.bytes.chunks_exact(4).zip(b.bytes.chunks_exact(4)) {
        let x = f32::from_le_bytes(x.try_into().unwrap());
        let y = f32::from_le_bytes(y.try_into().unwrap());
        max_abs = max_abs.max((x - y).abs());
        scale = scale.max(x.abs()).max(y.abs());
    }
    max_abs / scale.max(1e-30)
}

#[test]
fn fused_chunk_matches_tensor_path() {
    type B = CudaBare;
    let dev: Device = Default::default();
    dev.seed(42); // deterministic — the k=128 config is fp32-vs-TF32 sensitive, unseeded draws flake
                  // chunk <= 16: the fused kernels' numerical range (K3 design). Larger
                  // chunks fall back to the 16-tile tensor path (checked separately).
    for (batch, heads, time, k_dim, v_dim, cs) in [
        (1usize, 4usize, 64usize, 64usize, 64usize, 16usize),
        (1usize, 4usize, 256usize, 64usize, 64usize, 16usize),
        (2, 8, 2048, 64, 64, 16),
        (1, 8, 4096, 128, 128, 16),
    ] {
        let q = Tensor::<4>::random(
            [batch, heads, time, k_dim],
            Distribution::Normal(0.0, 0.1),
            &dev,
        );
        let k = Tensor::<4>::random(
            [batch, heads, time, k_dim],
            Distribution::Normal(0.0, 0.1),
            &dev,
        );
        let v = Tensor::<4>::random(
            [batch, heads, time, v_dim],
            Distribution::Normal(0.0, 0.1),
            &dev,
        );
        let g = Tensor::<4>::random(
            [batch, heads, time, k_dim],
            Distribution::Normal(-0.05, 0.1),
            &dev,
        );
        let b = Tensor::<4>::random(
            [batch, heads, time, k_dim],
            Distribution::Uniform(0.0, 0.1),
            &dev,
        );
        let w = Tensor::<4>::random(
            [batch, heads, time, v_dim],
            Distribution::Uniform(0.0, 0.1),
            &dev,
        );
        let s = Tensor::<4>::random(
            [batch, heads, k_dim, v_dim],
            Distribution::Normal(0.0, 0.1),
            &dev,
        );
        let scale = (k_dim as f64).powf(-0.5);

        let (ref_out, ref_s) = chunk_wy_forward(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            b.clone(),
            w.clone(),
            s.clone(),
            scale,
            cs,
        );
        let (f_out, f_s) = fused_chunk_forward::<B>(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            b.clone(),
            w.clone(),
            s.clone(),
            scale,
            cs,
        )
        .expect("fused path should dispatch");

        let do_out = max_rel(&f_out, &ref_out);
        let d_s = max_rel(&f_s, &ref_s);
        println!(
            "B={batch} H={heads} T={time} k={k_dim} v={v_dim}: out_rel={do_out:.2e} state_rel={d_s:.2e}"
        );
        if k_dim == 128 {
            let fo: Vec<f32> = f_out
                .clone()
                .into_data()
                .bytes
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect();
            let ro: Vec<f32> = ref_out
                .clone()
                .into_data()
                .bytes
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect();
            println!("k128 out fused [0..8] {fo:?}");
            println!("k128 out ref   [0..8] {ro:?}");
            println!("k128 out fused [4096..4104] {fo:?}");
        }
        assert!(do_out < 1e-3, "out mismatch: {do_out:.2e}");
        assert!(d_s < 1e-3, "state mismatch: {d_s:.2e}");
    }
    let _ = NdArray::name(&Default::default());
}

/// The fused-op backward with the kernel-exported M^-1 must match the
/// tensor-path gradients on CUDA (within fp32 noise from the kernel).
#[test]
fn fused_op_grads_match_tensor_path_cuda() {
    use burn::tensor::Device as D;
    let plain: D = Default::default();
    plain.seed(42);
    let dev = burn::tensor::Device::autodiff(plain);
    let (batch, heads, time, k_dim, v_dim, cs) =
        (1usize, 2usize, 128usize, 32usize, 32usize, 16usize);
    let scale = (k_dim as f64).powf(-0.5);
    let mk = |shape: [usize; 4], dist: Distribution| {
        Tensor::<4>::random(shape, dist, &dev).require_grad()
    };
    let q = mk([batch, heads, time, k_dim], Distribution::Normal(0.0, 0.1));
    let k = mk([batch, heads, time, k_dim], Distribution::Normal(0.0, 0.1));
    let v = mk([batch, heads, time, v_dim], Distribution::Normal(0.0, 0.1));
    let g = mk([batch, heads, time, k_dim], Distribution::Normal(-0.5, 0.2));
    let b = mk([batch, heads, time, k_dim], Distribution::Uniform(0.0, 0.1));
    let w = mk([batch, heads, time, v_dim], Distribution::Uniform(0.0, 0.1));
    let s = mk([batch, heads, k_dim, v_dim], Distribution::Normal(0.0, 0.1));

    let (out, _) = chunk_wy_forward(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        b.clone(),
        w.clone(),
        s.clone(),
        scale,
        cs,
    );
    let grads_ref = out.powf_scalar(2.0).sum().backward();
    let (out2, _) = burn_gdn2::chunk_wy_forward_autodiff::<CudaBare>(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        b.clone(),
        w.clone(),
        s.clone(),
        scale,
        cs,
    )
    .expect("op should dispatch");
    let grads_f = out2.powf_scalar(2.0).sum().backward();

    for (name, t) in [
        ("q", &q),
        ("k", &k),
        ("v", &v),
        ("g", &g),
        ("b", &b),
        ("w", &w),
        ("s", &s),
    ] {
        let gr = t.grad(&grads_ref).unwrap().clone();
        let gf = t.grad(&grads_f).unwrap().clone();
        let a = gr.clone().into_data();
        let b = gf.clone().into_data();
        let mut max_abs = 0.0f32;
        let mut scale_v = 0.0f32;
        for (x, y) in a.bytes.chunks_exact(4).zip(b.bytes.chunks_exact(4)) {
            let x = f32::from_le_bytes(x.try_into().unwrap());
            let y = f32::from_le_bytes(y.try_into().unwrap());
            max_abs = max_abs.max((x - y).abs());
            scale_v = scale_v.max(x.abs()).max(y.abs());
        }
        let rel = max_abs / scale_v.max(1e-30);
        println!("{name}: grads rel={rel:.2e}");
        // k's chunk-boundary rows divide by E≈glast (~1e-3 here), which
        // amplifies fp32 path noise to a few percent (data-dependent);
        // everything else is 1e-3 or better.
        let tol = if name == "k" { 1e-1 } else { 1e-2 };
        assert!(rel < tol, "{name}: grads mismatch rel={rel:.2e}");
    }
}

/// Zero-key regression: a key row that is exactly 0.0 must not NaN the fused
/// backward. The old `k·glast/kgd` E-reconstruction divides 0/0 in that case.
#[test]
fn fused_zero_key_row_grads_finite() {
    use burn::tensor::TensorData;
    let plain: burn::tensor::Device = Default::default();
    plain.seed(7);
    let dev = burn::tensor::Device::autodiff(plain);
    let (batch, heads, time, k_dim, v_dim, cs) =
        (1usize, 2usize, 16usize, 32usize, 32usize, 16usize);
    let scale = (k_dim as f64).powf(-0.5);
    let mk = |shape: [usize; 4], dist: Distribution| {
        Tensor::<4>::random(shape, dist, &dev).require_grad()
    };
    let q = mk([batch, heads, time, k_dim], Distribution::Normal(0.0, 0.1));
    let v = mk([batch, heads, time, v_dim], Distribution::Normal(0.0, 0.1));
    let g = mk([batch, heads, time, k_dim], Distribution::Normal(-0.5, 0.2));
    let b = mk([batch, heads, time, k_dim], Distribution::Uniform(0.0, 0.1));
    let w = mk([batch, heads, time, v_dim], Distribution::Uniform(0.0, 0.1));
    let s = mk([batch, heads, k_dim, v_dim], Distribution::Normal(0.0, 0.1));

    // k with row 0 of head 0 exactly zero (everything else nonzero).
    let k = {
        let mut data = vec![0.0f32; batch * heads * time * k_dim];
        let mut val = 0.1f32;
        for x in data.iter_mut() {
            *x = val;
            val = val * 1.001 + 0.001;
        }
        for x in data.iter_mut().take(k_dim) {
            *x = 0.0; // head 0, time 0
        }
        Tensor::<4>::from_data(
            TensorData::new(data, [batch, heads, time, k_dim].to_vec()),
            &dev,
        )
        .require_grad()
    };

    let (out, _) = burn_gdn2::chunk_wy_forward_autodiff::<CudaBare>(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        b.clone(),
        w.clone(),
        s.clone(),
        scale,
        cs,
    )
    .expect("op should dispatch");
    let grads = out.powf_scalar(2.0).sum().backward();

    for (name, t) in [
        ("q", &q),
        ("k", &k),
        ("v", &v),
        ("g", &g),
        ("b", &b),
        ("w", &w),
        ("s", &s),
    ] {
        let gr = t.grad(&grads).unwrap().clone();
        let d = gr.into_data();
        let (mut n_nan, mut n_inf) = (0usize, 0usize);
        for bytes in d.bytes.chunks_exact(4) {
            let x = f32::from_le_bytes(bytes.try_into().unwrap());
            if x.is_nan() {
                n_nan += 1;
            }
            if x.is_infinite() {
                n_inf += 1;
            }
        }
        println!("{name}: NaN={n_nan} inf={n_inf}");
        assert_eq!(n_nan, 0, "{name}: {n_nan} NaN grads");
        assert_eq!(n_inf, 0, "{name}: {n_inf} inf grads");
    }
}
