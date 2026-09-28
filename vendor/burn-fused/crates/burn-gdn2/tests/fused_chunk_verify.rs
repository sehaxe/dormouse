#![cfg(all(feature = "cuda", feature = "autodiff"))]
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
//! Fused chunk kernels vs tensor path: numerical equivalence on CUDA.
//! Run: cargo test --release --features "cuda" --test fused_chunk_verify -- --nocapture

use burn::backend::{Backend, NdArray};
use burn::tensor::{Device, Distribution, Tensor};
use burn_gdn2::kernel::chunk_cube::cuda::fused_chunk_forward;
use burn_gdn2::{chunk_wy_forward, CudaBare};

/// Relative (to the largest magnitude in the tensor) bound for a fused-vs-tensor
/// gradient comparison. See the note at its only use: this is a BAR chosen to
/// be tight, not a value anybody has measured the fused adjoint against.
const GRAD_REL_TOL: f32 = 1e-2;

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
/// # Still `#[ignore]`d - and now for a MEASURED reason, not a stale one
///
/// It was ignored because the op's backward refused: an inverted
/// `fused_allowed` in `autodiff.rs` had made the fused forward inside the node
/// dead code, and a `strip`/`rebuild` pair in `FusedAdjoint` had been asked to
/// strip tensors that were already bare, so the adjoint returned `Err` and the
/// caller panicked. Both fixed 2026-09-28, and with them the refusal went away
/// - the fused forward and the fused adjoint both launch now.
///
/// What the hardware then showed is that the gate should stay closed. On this
/// fixture (`b=1 h=2 T=128 K=V=32 chunk=16`, loss `sum(out^2)`, `g ~ N(-0.5,
/// 0.2)`, `b, w ~ U(0, 0.1)`) the seven printed relative deviations are:
///
/// ```text
/// q 2.65e-7   k 8.69e-3   v 1.04e-7   g 3.41e-3   b 5.65e-8   w 1.96e-7   s 2.86e-7
/// ```
///
/// i.e. k and g sit at 8.7e-3 and 3.4e-3 against a 1e-2 bar - a coin flip, not
/// a margin. Measured the same day, running this file's three tests together
/// (they share one process RNG, so the draw differs) the SAME test FAILS at
/// that assertion. And on a deliberate different draw - `b, w ~ N(0, 0.3)`
/// instead of `U(0, 0.1)`, `b=2 h=2`, `T >= 32` - the fused adjoint's `d_k` is
/// off by 1.8e-1..2.8e-1 and its `d_g` by 1.7e-2..3.2e-2 against the ops path,
/// while `q, v, b, w, s` stay at ~3e-7 (table in
/// `tests/autodiff_cuda_gate.rs::fused_kernels_run_from_a_balanced_graph`).
///
/// So this file's bar cannot distinguish a correct adjoint from a wrong one,
/// and a test that flips on the RNG is worse than a red one. Un-ignoring it
/// means fixing the adjoint's cross-chunk BPTT terms for `E = exp(cumsum(g))`
/// first, or replacing the bar with a reference that can see the difference.
/// `tests/ops_grad_cuda.rs` is that reference for the arm the trainer takes:
/// central finite differences, agreeing to 1.4e-2 worst case and ~1e-4 typical.
///
/// (The zero-key NaN test below IS un-ignored: it asserts finiteness, which is
/// draw-independent, and it runs.)
///
/// Measured on this fixture (`b=1 h=2 T=128 K=V=32 chunk=16`, loss
/// `sum(out^2)`, `g ~ N(-0.5, 0.2)`, `b, w ~ U(0, 0.1)`), the seven printed
/// relative deviations:
///
/// ```text
/// q 2.65e-7   k 8.69e-3   v 1.04e-7   g 3.41e-3   b 5.65e-8   w 1.96e-7   s 2.86e-7
/// ```
///
/// Under the 1e-2 bar, and the values are pasted here because the comment above
/// the bar asked for exactly that.
///
/// # What the green does NOT mean
///
/// `k` and `g` sit at 8.7e-3 and 3.4e-3 - two to three orders of magnitude
/// above the other five, and inside the noise of a fixture-dependent bar. On a
/// DIFFERENT draw of the same shapes they are not: with `b, w ~ N(0, 0.3)`
/// instead of `U(0, 0.1)`, at `b=2 h=2 T>=32`, the fused adjoint's `d_k` is off
/// by 1.8e-1..2.8e-1 and its `d_g` by 1.7e-2..3.2e-2 against the ops path,
/// while `q, v, b, w, s` stay at ~3e-7 (the table is in
/// `tests/autodiff_cuda_gate.rs::fused_kernels_run_from_a_balanced_graph`).
/// So this test passing is evidence about ONE fixture, not about the adjoint:
/// the cross-chunk BPTT terms for `E = exp(cumsum(g))` are the suspect, and
/// this file's bar cannot see them. The gate above can, and is red.
///
/// The fused adjoint also cannot be checked anywhere else: it is
/// `#[cfg(feature = "cuda")]` and gated on the bare `CubeBackend`, so there is
/// no CPU device on which these kernels run at all.
///
/// Run it on demand:
/// `cargo test -p burn-gdn2 --release --features cuda,autodiff --test fused_chunk_verify -- --nocapture`
#[test]
#[ignore = "MEASURED 2026-09-28: the fused adjoint now RUNS (its old refusal was an inverted fused_allowed plus a strip of already-bare tensors, both fixed), and it is correct for one chunk - but past one chunk its d_k is off by 1.8e-1..2.8e-1 and its d_g by 1.7e-2..3.2e-2 against the ops path. This fixture squeaks under the 1e-2 bar at k=8.7e-3 g=3.4e-3 and FAILS on another draw, so the bar cannot see the defect"]
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
        // One bar for every input, 1e-2 relative to the largest magnitude in
        // the tensor. This used to be `if name == "k" { 1e-1 } else { 1e-2 }`,
        // with a comment claiming k's chunk-boundary rows divide by
        // E≈glast (~1e-3) and "amplify fp32 path noise to a few percent".
        //
        // That claim is not a measurement. The tolerance was never exercised
        // against the fused adjoint: the test is `#[ignore]`d, and until
        // `277b442` the "fused" side was the tensor adjoint too, so any `rel`
        // it printed was tensor-vs-tensor. A 10% allowance on one of seven
        // gradients is a green light for a wrong adjoint, and it was the
        // largest number in the file.
        //
        // 1e-2 is therefore a BAR, not a known-achievable value: it is what
        // the other six inputs were already held to, and a first run that
        // fails it is information, not a reason to widen. If k's gradient
        // turns out to be ill-conditioned through the `k·glast/kgd`
        // reconstruction, fix the conditioning. Whoever runs this first:
        // paste the seven printed `rel` values into the commit message — they
        // are the deliverable, and until they exist nobody knows whether the
        // fused adjoint is right, only that it has never been asked.
        assert!(rel < GRAD_REL_TOL, "{name}: grads mismatch rel={rel:.2e}");
    }
}

/// Zero-key regression: a key row that is exactly 0.0 must not NaN the fused
/// backward. The old `k*glast/kgd` E-reconstruction divides 0/0 in that case.
///
/// # Run green on hardware 2026-09-28 - and un-ignored
///
/// `#[ignore]`d for the same cause as the test above: the op's backward refused
/// before it reached a single assertion, so the NaN gate had never been
/// exercised. With the refusal fixed it runs, and all seven gradients come back
/// finite (0 NaN, 0 inf each).
///
/// It stays un-ignored while the test above does not, because its claim is
/// FINITENESS, which a wrong-but-finite gradient still satisfies and which no
/// draw has yet broken. It says nothing about the adjoint being correct; the
/// test above and `tests/autodiff_cuda_gate.rs` are for that.
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
