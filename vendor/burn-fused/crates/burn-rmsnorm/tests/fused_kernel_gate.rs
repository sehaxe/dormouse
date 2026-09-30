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

use std::sync::Mutex;

use burn::module::Param;
use burn::tensor::{Device, Distribution, Tensor};
use burn_rmsnorm::RMSNorm;

/// (d, mean of x^2, one x) — mean-square chosen so eps=1e-5 is decisive at the
/// bottom and provably invisible at the top, which is what makes the eps
/// question decidable at all.
/// `fused::calls()` is PROCESS-GLOBAL (`src/fused.rs` statics), and cargo runs
/// the tests of one binary on several threads. Without this lock the two tests
/// interleave their asks and both counter deltas are noise — measured 2026-09-30
/// as `took 7 asked, 1 skipped` for a test that had done 6 forwards, because the
/// sibling's ask landed inside this test's window. Every test that reads the
/// counters holds this.
static SEAM: Mutex<()> = Mutex::new(());

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
    let _g = SEAM.lock().unwrap_or_else(|e| e.into_inner());
    let cuda = Device::cuda(0);
    let (asked0, skipped0) = burn_rmsnorm::fused::calls();

    let mut checked = 0usize;
    let mut worst = 0.0f64;
    for (d, mean_sq, _x_target) in ROWS {
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
    // THE ASSERTION HERE USED TO BE `takes == skipped - skipped0`, and it was
    // WRONG: that demands one skip per ask, which is the shape of a DEAD kernel.
    // It could only ever have passed while this kernel never ran, which is why
    // `34c5631` and `9ac0377` could both land with it "passing" for a reason
    // that had nothing to do with the kernel working. Its own failure message
    // said the opposite of what the code demanded ("the fused arm DECLINED on a
    // bare CUDA device"), so message and assertion disagreed - the ADR-0020
    // defect, in a test. Measured, 2026-09-30: with the kernel fixed, one bare
    // device gives asked=6 skipped=0 over ROWS, and the old assertion rejected
    // exactly that. What "the arm was taken" means is: nothing declined, and
    // every one of the ROWS asks took the fused path.
    assert_eq!(
        skipped - skipped0,
        0,
        "the fused arm DECLINED on a bare CUDA device - {} of {} asks fell back - so \
         this test measured the tensor path and called it the kernel",
        skipped - skipped0,
        takes
    );
    assert_eq!(takes, ROWS.len() as u64, "the fused arm ran {} times, not {}", takes, ROWS.len());
    assert!(checked > 20, "only {checked} outputs compared");
    assert!(
        worst < 1e-5,
        "the fused kernel disagrees with the scalar definition by {worst:e} relative over \\
         {checked} outputs"
    );
    // The number is printed, not only asserted. A gate that reports "ok" and no
    // magnitude cannot be compared against a later run, and 1e-7 and 9e-6 are
    // the same verdict here and not the same measurement.
    eprintln!(
        "the FUSED kernel: asked {takes}, skipped {skips}, vs a scalar f64 definition - \
         worst {worst:e} relative over {checked} outputs, {rows} rows at d in {{4,8,16,32}} \
         and mean-square spanning 1e-4..1e2, tolerance 1e-5",
        skips = skipped - skipped0,
        rows = ROWS.len()
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
    let _g = SEAM.lock().unwrap_or_else(|e| e.into_inner());
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

/// The `d < 4` refusal in `fused.rs`, as a gate rather than a hope.
///
/// Measured 2026-09-30: at `d` of 2 or 3 the trailing cubes of the launch never
/// execute, so the fused arm returns a PARTLY UNWRITTEN tensor while the seam
/// counters report that it ran. The fixture case `d2` (`dims 1 2 2`) is off by
/// 5.245e-1 relative on the fused arm and 7.29e-8 on the tensor path. That is a
/// SILENT wrong answer, which is the worst class in this repo (ADR-0019), so
/// `rmsnorm_cuda` now declines `d < MIN_FUSED_D` and the caller's tensor path
/// answers instead.
///
/// This test asserts BOTH halves of that: the arm DECLINED (so a future change
/// that removes the guard goes red here rather than returning garbage), and the
/// answer that came back is RIGHT (so the fallback is pinned as a correct one,
/// not merely as a decline).
#[test]
fn the_fused_arm_declines_the_narrow_shapes_and_the_fallback_is_correct() {
    let _g = SEAM.lock().unwrap_or_else(|e| e.into_inner());
    let cuda = Device::cuda(0);

    for d in [1usize, 2, 3] {
        let rows = 4usize;
        let w: Vec<f32> = (0..d).map(gain_at).collect();
        let x: Vec<f32> = (0..rows * d)
            .map(|i| (i as f32 * 0.37).sin() * 3.0)
            .collect();

        // A non-constant gain, or a broken per-feature broadcast is invisible.
        let mut norm = RMSNorm::new(d, 1e-5, &cuda);
        norm.weight = Param::from_tensor(Tensor::<1>::from_floats(w.as_slice(), &cuda));

        let (asked0, skipped0) = burn_rmsnorm::fused::calls();
        let got: Vec<f32> = norm
            .forward(Tensor::from_floats(
                burn::tensor::TensorData::new(x.clone(), [1, rows, d]),
                &cuda,
            ))
            .into_data()
            .try_to_vec()
            .expect("nothing came back at all");
        let (asked, skipped) = burn_rmsnorm::fused::calls();

        assert_eq!(
            asked - asked0,
            1,
            "the fused arm was not asked once at d={d}"
        );
        assert_eq!(
            skipped - skipped0,
            1,
            "the fused arm RAN at d={d}, which it must not: the launch's trailing \
             cubes do not execute at d < {min}, so the result is a partially \
             unwritten tensor and the counters would have said 'ran'. Either fix \
             the cubecl defect that `tests/d2_isolate.rs` reproduces, or leave \
             the guard alone — do not just drop it.",
            min = burn_rmsnorm::fused::MIN_FUSED_D
        );

        // And the fallback is a CORRECT answer, not merely a decline: the same
        // f64 definition the other test uses, on this row set.
        for r in 0..rows {
            let row = &x[r * d..(r + 1) * d];
            let ss: f64 = row.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
            for i in 0..d {
                let want = f64::from(row[i]) / (ss / d as f64 + 1e-5).sqrt() * f64::from(gain_at(i));
                let rel =
                    (f64::from(got[r * d + i]) - want).abs() / want.abs().max(1.0);
                assert!(rel < 1e-5, "d={d} r={r} i={i}: rel {rel:e} from the FALLBACK");
            }
        }
    }
    eprintln!(
        "d in 1..3: the fused arm declined every one (COUNTED, not silent) and the \
         tensor path answered all of them correctly"
    );
}
