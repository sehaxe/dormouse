//! The cross-vendor gate: the fused RMSNorm kernel lowered to **WGSL** by
//! cubecl-wgpu and run on a CPU-adapter Vulkan device (lavapipe), where no AMD
//! or Intel card is attached. Software Vulkan is fine here — the gate asserts
//! CORRECTNESS and that the fused arm was TAKEN, not speed.
//!
//! What it pins, in ONE test (the seam counters are process-global and cargo
//! runs `#[test]`s in parallel threads, so any two readers of these counters in
//! one binary race — hence a single function, two phases):
//!
//! 1. the kernel-direct call `rmsnorm_cuda::<CubeBackend>` on a wgpu device
//!    INCREMENTS `ASKED` without `SKIPPED` — the downcast seam
//!    (`try_into_primitive` → `CubeTensor`) holds on wgpu exactly as on CUDA,
//!    and the WGSL-lowered kernel answers to 1e-5 relative against a scalar
//!    f64 definition;
//! 2. `RMSNorm::forward` on a wgpu device routes through the same fused arm
//!    (the cfg seam in `lib.rs`), not the tensor path.
//!
//! A fix no test can see is half a fix; without this file, reverting the
//! launch to a non-lowering shape turns zero wgpu-side tests red because no
//! wgpu-side test exists.
//!
//! Needs a CPU-capable Vulkan ICD (Arch: `pacman -S vulkan-swrast`); without
//! one the adapter request fails LOUDLY — this test never skips.

use burn::tensor::{Device, DeviceKind, Tensor, TensorData};
use burn_cubecl::CubeBackend;
use burn_rmsnorm::fused;

/// (d, mean of x^2, one x) — the same eps-deciding fixture the CUDA gate uses.
const ROWS: [(usize, f32, f32); 3] = [(8, 1.0, 1.4), (16, 4.0, 2.8), (32, 0.25, 0.7)];

fn dev() -> Device {
    Device::wgpu(DeviceKind::Cpu)
}

/// Scalar f64 definition — an INDEPENDENT formulation, not the tensor path
/// restated (the same oracle discipline as `fused_kernel_gate.rs`).
fn reference(x: &[f32], w: &[f32], d: usize, eps: f64) -> Vec<f64> {
    x.chunks_exact(d)
        .flat_map(|row| {
            let ms: f64 = row
                .iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                / d as f64;
            let inv = (ms + eps).sqrt().recip();
            row.iter()
                .zip(w.iter())
                .map(move |(&x, &w)| f64::from(x) * inv * f64::from(w))
        })
        .collect()
}

#[test]
fn rmsnorm_kernel_runs_on_wgpu() {
    let dev = dev();
    let eps = 1e-5_f32;
    let (asked0, skipped0) = fused::calls();
    let mut asked = asked0;
    let mut skipped = skipped0;

    for &(d, ms, x0) in ROWS.iter() {
        let rows = 3usize;
        // scale so the row mean-square hits the fixture target
        let norm0 = (x0 * x0) as f64;
        let scale = (f64::from(ms) / norm0).sqrt();
        let xs: Vec<f32> = (0..rows * d)
            .map(|i| ((i % d) as f32 * x0).sin() * scale as f32)
            .collect();
        let w: Vec<f32> = (0..d).map(|i| 0.5 + 0.25 * (i % 7) as f32).collect();

        let out = fused::rmsnorm_cuda::<CubeBackend>(
            Tensor::<2>::from_data(TensorData::new(xs.clone(), [rows, d]), &dev),
            Tensor::<1>::from_data(TensorData::new(w.clone(), [d]), &dev),
            eps,
        )
        .expect("the fused arm must be TAKEN on a wgpu device");
        let (a, s) = fused::calls();
        asked += 1;
        assert_eq!(a, asked, "d={d}: the seam did not count the ask");
        assert_eq!(
            s, skipped,
            "d={d}: the arm was SKIPPED — the downcast seam failed on wgpu"
        );
        skipped = s;

        let got: Vec<f64> = out
            .into_data()
            .try_to_vec::<f32>()
            .expect("readback")
            .iter()
            .map(|v| f64::from(*v))
            .collect();
        let want = reference(&xs, &w, d, f64::from(eps));
        for (r, (g, wnt)) in got.iter().zip(want.iter()).enumerate() {
            let rel = (g - wnt).abs() / wnt.abs().max(1.0);
            assert!(
                rel < 1e-5,
                "d={d} elem {r}: got {g} want {wnt} (rel {rel:.3e})"
            );
        }
    }

    // Phase 2, same single sequence: the module forward must route through the
    // SAME fused arm on a wgpu device (the cfg seam in lib.rs). One function,
    // because the counters are process-global and cargo runs #[test]s in
    // parallel threads — two readers of these counters in one binary race.
    let (b, t, d) = (3usize, 4usize, 8usize);
    let mut norm = burn_rmsnorm::RMSNorm::new(d, 1e-5, &dev);
    let w: Vec<f32> = (0..d).map(|i| 0.5 + 0.25 * i as f32).collect();
    norm.weight = burn::module::Param::from_tensor(Tensor::<1>::from_floats(w.as_slice(), &dev));
    let xs: Vec<f32> = (0..b * t * d)
        .map(|i| (i as f32 * 0.37).sin() * 3.0)
        .collect();

    let got: Vec<f32> = norm
        .forward(Tensor::<3>::from_data(
            TensorData::new(xs.clone(), [b, t, d]),
            &dev,
        ))
        .into_data()
        .try_to_vec::<f32>()
        .expect("readback");
    let (a, s) = fused::calls();
    assert_eq!(
        a,
        asked + 1,
        "module forward did not ask the fused arm — the cfg seam in lib.rs is closed on wgpu"
    );
    assert_eq!(s, skipped, "module forward SKIPPED to the tensor path");

    let want = reference(&xs, &w, d, 1e-5);
    for (r, (g, wnt)) in got.iter().zip(want.iter()).enumerate() {
        let rel = (f64::from(*g) - wnt).abs() / wnt.abs().max(1.0);
        assert!(rel < 1e-4, "elem {r}: got {g} want {wnt} (rel {rel:.3e})");
    }
}
