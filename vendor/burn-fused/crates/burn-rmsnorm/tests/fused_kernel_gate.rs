//! The fused CUDA kernel, against a scalar f64 definition — the gate that did
//! not exist while the kernel was broken.
//!
//! # WHY THIS FILE
//!
//! `34c5631` fixed a real defect: `Shared::new_slice` sized by a runtime value
//! instead of a `#[comptime]` one, so the lowered module failed LLVM
//! verification (`Expected operand type llvm.ptr, but found builtin.integer`) and
//! **the kernel had never produced a number on any device in the life of the
//! project**. The fix is right — it is the `#[comptime]` pattern every working
//! kernel in this tree uses — and re-reverting it turned **zero** tests red,
//! because nothing executed the kernel: `BURN_DEVICE` appeared in no test under
//! this crate and its `Cargo.toml` had zero `required-features`.
//!
//! A fix no test can see is half a fix. This is the other half.
//!
//! # WHAT IT ASSERTS
//!
//! 1. On a BARE CUDA device the fused arm is **TAKEN** — asserted from the seam
//!    counter, not assumed — and its answer matches a scalar f64 definition to
//!    1e-5 relative, at mean-squares spanning 1e-4 .. 4 so the eps question is
//!    decided (`the_claim` in `rmsnorm_oracle.rs` measures that separation at
//!    44538x the tolerance).
//! 2. The same input on the **autodiff** device takes the tensor path, with the
//!    counter incremented on the fallback side. That is `norm=0/N` from the
//!    trainer's logs, turned into a gate. If it ever inverts, the arm runs on a
//!    device where it has **no backward**, returns a leaf, receives **no
//!    gradient**, and the loss curve still looks healthy — the `8fa5d4c` defect
//!    verbatim. This test is the thing standing between that and a training run.
//!
//! # WHAT IT CANNOT DO, stated up front
//!
//! - **No backward.** `rmsnorm_cuda` returns a fresh `Tensor::<2>::empty` with
//!   raw handles written in, so the result carries no graph. This file pins the
//!   guard that prevents the backward-less path from being reached; it cannot
//!   supply the backward that would make reaching it safe.
//! - CUDA-only. `tools/lib_gate.sh` runs the ndarray cell and does not build
//!   this; the vendor GPU gate does.
#![cfg(feature = "cuda")]

use burn::module::Param;
use burn::tensor::{Device, Distribution, Tensor};
use burn_rmsnorm::RMSNorm;

/// (d, mean of x^2, one x) — mean-square chosen so eps=1e-5 is decisive at the
/// bottom and provably invisible at the top, which is what makes the eps
/// question decidable at all.
const ROWS: [(usize, f32, f32); 6] = [
    (8, 1.0, 1.4),
    (8, 4.0, 2.8),
    (4, 1.0, -1.0),
    (16, 0.01, 0.14),
    (16, 0.0001, 0.014),
    (32, 100.0, 14.0),
];

/// `x / sqrt(mean(x^2) + eps) * gain`, f64 on the host. The definition written
/// out, not a transcription of the crate, so it is independent in the way that
/// matters.
fn scalar_ref(x: f32, mean_sq: f64, gain: f32, eps: f64) -> f64 {
    (f64::from(x) / (mean_sq + eps).sqrt()) * f64::from(gain)
}

/// A non-constant gain, or a broken per-feature broadcast is invisible — the
/// same reason the unit test in `src/lib.rs` uses one.
fn gain_at(i: usize) -> f32 {
    0.5 + 0.25 * (i % 3) as f32
}

fn build_row(d: usize, mean_sq: f32) -> Vec<f32> {
    // Exact mean-square: x_i = sqrt(mean_sq) * u_i with mean(u^2) == 1.
    let us: Vec<f32> = (0..d).map(|i| ((i * 2 + 1) as f32 * std::f32::consts::PI / d as f32).sin()).collect();
    let m = us.iter().map(|u| u * u).sum::<f32>() / us.len() as f32;
    us.iter().map(|u| (mean_sq * u * u / m).sqrt()).collect()
}

#[test]
fn the_fused_kernel_is_taken_on_cuda_and_matches_the_scalar_definition() {
    let cuda = Device::cuda(0);
    let (asked0, skipped0) = burn_rmsnorm::fused::calls();

    let mut checked = 0usize;
    let mut worst = 0.0f64;
    for (d, mean_sq, x_target) in ROWS {
        let mut norm = RMSNorm::new(d, 1e-5, &cuda);
        let w: Vec<f32> = (0..d).map(gain_at).collect();
        norm.weight = Param::from_tensor(Tensor::<1>::from_floats(w.as_slice(), &cuda));

        let row = build_row(d, mean_sq);
        let got: Vec<f32> = norm
            .forward(Tensor::from_floats(
                burn::tensor::TensorData::new(row.clone(), [1, 1, d]),
                &cuda,
            ))
            .into_data()
            .try_to_vec()
            .unwrap();
        let take = |i: usize| *got.get(i).expect("row too short for d");

        // Only the positions the fixed vector actually carries; `d` may be < 32.
        for (i, &x) in row.iter().take(d).enumerate() {
            let want = scalar_ref(x, f64::from(mean_sq), gain_at(i), 1e-5);
            let rel = (f64::from(got[i]) - want).abs() / want.abs().max(1.0);
            worst = worst.max(rel);
            checked += 1;
        }
    }

    let (asked, skipped) = burn_rmsnorm::fused::calls();
    let takes = asked - asked0;
    assert_eq!(
        takes,
        skipped - skipped0,
        "the fused arm DECLINED on a bare CUDA device, so this test measured the \\
         tensor path and called it the kernel. took {} asked, {} skipped",
        takes,
        skipped - skipped0
    );
    assert_eq!(takes, ROWS.len() as u64, "the fused arm ran {} times, not {}", takes, ROWS.len());
    assert!(checked > 20, "only {checked} outputs compared");
    assert!(
        worst < 1e-5,
        "the fused kernel disagrees with the scalar definition by {worst:e} relative over \\
         {checked} outputs"
    );

    // CONTROL: the same input through the tensor path on ndarray must agree with
    // the same f64 reference, so a pass above cannot come from the reference
    // merely agreeing with itself. `burn::tensor` has NO rms_norm of its own -
    // this crate IS the implementation - so the control is this crate on CPU.
    let cpu = Device::ndarray();
    let (d, mean_sq) = (8usize, 1.0f32);
    let mut norm = RMSNorm::new(d, 1e-5, &cpu);
    let w: Vec<f32> = (0..d).map(gain_at).collect();
    norm.weight = Param::from_tensor(Tensor::<1>::from_floats(w.as_slice(), &cpu));
    let row = build_row(d, mean_sq);
    let ctl: Vec<f32> = norm
        .forward(Tensor::from_floats(
            burn::tensor::TensorData::new(row.clone(), [1, 1, d]),
            &cpu,
        ))
        .into_data()
        .try_to_vec()
        .unwrap();
    for (i, &x) in row.iter().take(d).enumerate() {
        let want = scalar_ref(x, f64::from(mean_sq), gain_at(i), 1e-5);
        let rel = (f64::from(ctl[i]) - want).abs() / want.abs().max(1.0);
        assert!(rel < 1e-5, "the CONTROL is wrong, not the kernel: i={i} rel={rel:e}");
    }
}

/// The dispatch guard. On an AUTODIFF device the kernel must DECLINE, because it
/// has no backward and would return a leaf — `norm=0/N` from the trainer's own
/// logs, as a gate rather than a doc string.
#[test]
fn the_fused_kernel_declines_on_an_autodiff_device() {
    let dev = Device::cuda(0).autodiff();
    let before = burn_rmsnorm::fused::calls();
    let t: Tensor<3> = Tensor::random([1, 4, 64], Distribution::Default, &dev);
    let out = RMSNorm::new(64, 1e-5, &dev).forward(t);
    let after = burn_rmsnorm::fused::calls();

    assert_eq!(
        after.1,
        before.1 + 1,
        "the fused arm was TAKEN on an autodiff device. It has no backward - \\
         rmsnorm_cuda returns a fresh Tensor::empty with raw handles written in, so \\
         the result carries no graph and the arm would receive NO GRADIENT while \\
         the loss curve still looked healthy."
    );
    assert_eq!(out.dims(), [1, 4, 64], "the fallback must still produce the right shape");
}
