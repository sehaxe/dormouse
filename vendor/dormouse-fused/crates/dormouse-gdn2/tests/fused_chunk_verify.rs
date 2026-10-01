#![cfg(all(feature = "cuda", feature = "autodiff"))]
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
//! Fused chunk kernels vs tensor path: numerical equivalence on CUDA.
//! Run: cargo test --release --features "cuda" --test fused_chunk_verify -- --nocapture

use burn::backend::{Backend, NdArray};
use burn::tensor::{Device, Distribution, Tensor};
use dormouse_gdn2::kernel::chunk_cube::cuda::fused_chunk_forward;
use dormouse_gdn2::{chunk_wy_forward, CudaBare};

/// Relative (to the largest magnitude in the tensor) bound for a fused-vs-tensor
/// gradient comparison.
///
/// The `1e-2` this replaced was called "a bar, not a known-achievable value",
/// which was true: the comparison had never run against a kernel. It has now.
/// Measured 2026-09-28 on CUDA at b=1 h=2 t=128 k=v=32 chunk=16, worst input
/// `g` at **2.4e-7**; the CPU twin of the same comparison
/// (`autodiff_chunk.rs::fused_grads_match_tensor_path`, tensor adjoint against
/// per-op autograd) is held to 1e-2 with its own worst case at ~1e-3.
///
/// So 1e-3: four times the measured worst, and four times tighter than the
/// tensor adjoint's own bound on the same function. fp32 eps is 1.2e-7 and a
/// f32 reassociation over 8 chunk boundaries lands where these land, so there
/// is no headroom between 1e-3 and 1e-2 that means anything - and a wrong
/// adjoint that is wrong by a few percent still fails both. See
/// `tests/fused_adjoint_vs_ops.rs` for the same comparison with the two shapes
/// that localise a wrong term.
const GRAD_REL_TOL: f32 = 1e-3;

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
///
/// # GREEN on hardware, 2026-09-28 — and it is no longer a tautology
///
/// This test compared `ChunkWy::backward` against the tensor path, and for its
/// whole life both sides of that pair were the TENSOR adjoint: the fused
/// forward branch was gated on `fused_forced_off()`, so the fused kernels ran
/// only when the kill switch said the fused path was off. Any `rel` it printed
/// was tensor-vs-tensor, which is why the tolerance could be described as a bar
/// nobody had measured and still be committed.
///
/// Three things changed, all in the library, all measured (see
/// `autodiff_cuda_gate.rs` for the same three with a git history):
///   1. the fused-forward gate was inverted, so the kernels ran in neither
///      direction;
///   2. the adjoint's strip-to-bare asked the dispatch layer for an autodiff
///      context on a tensor that arrives `Disabled`, and the registration
///      re-wrapped gradients as `Autodiff<Inner, S>` where `Backward`'s `B` is
///      the bare backend;
///   3. two real arithmetic defects the comparison was in no position to see,
///      because it was comparing the tensor adjoint with itself: a spurious
///      factor of `E` in BK1's `bk` (d_g wrong by 3.5e-1 on a single chunk) and
///      a `reshape` standing in for a transpose of `d_s` in the BPTT glue
///      (d_k wrong by 3.3e-1 whenever there is more than one chunk to carry).
///
/// Printed values now, worst input first: g 2.4e-7, s 2.9e-7, q 2.7e-7.
/// The fused adjoint is `#[cfg(feature = "cuda")]` and its kernels are
/// `#[cfg(feature = "cuda")]` on the bare `CubeBackend`, so this file and
/// `autodiff_cuda_gate.rs` are the only places the question can be asked —
/// which is why the tolerance below is a measured number.
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
    let (out2, _) = dormouse_gdn2::chunk_wy_forward_autodiff::<CudaBare>(
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
        assert!(rel < GRAD_REL_TOL, "{name}: grads mismatch rel={rel:.2e}");
    }
}

/// Zero-key regression: a key row that is exactly 0.0 must not NaN the fused
/// backward. The old `k·glast/kgd` E-reconstruction divides 0/0 in that case.
///
/// # GREEN on hardware, 2026-09-28
///
/// The `#[ignore]` this replaces named the adjoint's `strip(k)` refusal, which
/// was real and is gone. The gate was never reachable before: the fused
/// forward branch was gated on `fused_forced_off()`, so the op's backward ran
/// the tensor branch and the fused adjoint's kernels were not what produced the
/// gradients. With the branch fixed and the two arithmetic defects repaired
/// (see `fused_op_grads_match_tensor_path_cuda`), a key row of exactly 0.0
/// leaves every gradient finite: the old `k·glast/kgd` E-reconstruction divided
/// 0/0, and the forward now exports `E` directly.
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

    let (out, _) = dormouse_gdn2::chunk_wy_forward_autodiff::<CudaBare>(
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
