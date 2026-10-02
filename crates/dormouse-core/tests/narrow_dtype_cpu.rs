//! The reason the CPU backend is burn-flex and not burn-ndarray: **it does
//! f16 and bf16.**
//!
//! burn-ndarray declared `DType::F16 | DType::BF16` an empty usage set
//! (`burn-ndarray/src/backend.rs:97`) and refused both, so every CPU test in
//! this crate ran fp32-only and the CPU suite could not gate a half-precision
//! bug at all — which is why ADR-0016's bugs 2 and 3 were CUDA-only by
//! construction. burn-flex declares both `DTypeUsage::Storage |
//! DTypeUsage::Arithmetic` (`burn-flex/src/backend.rs:140`).
//!
//! So the claim is not "the crate is in the lockfile", it is "a narrow-dtype
//! matmul on the CPU backend produces the right numbers". That is what this
//! file asserts, and it runs with no GPU and no native BLAS.
//!
//! This is deliberately NOT a copy of `backend-parity`'s f16/bf16 gate: that
//! crate tests the backends themselves and is still on burn-ndarray for its
//! CPU arm. This one pins the property of the backend *this crate's* CPU tests
//! now run on, so a regression here fails the ordinary `cargo test` run.
//!
//! Run: `cargo test -p dormouse-core --test narrow_dtype_cpu`

use burn::tensor::{DType, Device, FloatDType, Tensor, TensorData};

/// 2.0 / 0.5 / -1.5 for sign and rounding, a 0 and a -0.0 (a kernel that
/// normalises zero still has to agree on the dot), and an exact 1.0.
const K: usize = 8;
#[rustfmt::skip]
const K_VALUES: [f32; K] = [1.0, 2.0, 0.5, -1.5, 3.25, 0.0, -0.0, 1.0];

/// f16 eps is 2^-11 = 4.9e-4, bf16 eps is 2^-8 = 3.9e-3. Measured against
/// fp32 on the SAME device, 1e-2 / 5e-2 relative is the honest band for an
/// 8-64 term dot; see `backend-parity` for the same constants and why.
fn narrow_matmul_matches_fp32(cast: FloatDType, eps: f32) {
    let device = Device::flex();
    let a = Tensor::<2>::from_data(TensorData::new(K_VALUES.to_vec(), [1, K]), &device);
    let b = Tensor::<2>::from_data(TensorData::new(K_VALUES.to_vec(), [K, 1]), &device);
    let reference: f32 = a
        .clone()
        .cast(FloatDType::F32)
        .matmul(b.clone().cast(FloatDType::F32))
        .into_scalar();

    let an = a.cast(cast);
    // The dtype claim itself: burn-ndarray refused here, so assert the cast
    // produced the requested dtype rather than silently keeping f32.
    assert_eq!(
        an.dtype(),
        DType::from(cast),
        "cast produced the wrong dtype"
    );
    let got: f32 = an.matmul(b.cast(cast)).cast(FloatDType::F32).into_scalar();
    assert!(
        (got - reference).abs() <= eps * reference.abs().max(1.0),
        "{cast:?} 1xK @ Kx1 matmul on the CPU backend: {got} vs fp32 {reference}"
    );
}

/// The shape that actually runs in a trainer, where vectorization and tiling
/// differ from the dot above — the case that would expose a narrow-dtype path
/// that only handles small products.
fn narrow_gemm_matches_fp32(cast: FloatDType, eps: f32) {
    let device = Device::flex();
    let (m, k, n) = (64usize, 96usize, 48usize);
    let av: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
    let bv: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) / 5.0).collect();
    let a = Tensor::<2>::from_data(TensorData::new(av, [m, k]), &device);
    let b = Tensor::<2>::from_data(TensorData::new(bv, [k, n]), &device);
    let reference: Vec<f32> = a
        .clone()
        .cast(FloatDType::F32)
        .matmul(b.clone().cast(FloatDType::F32))
        .into_data()
        .try_to_vec_as()
        .unwrap();
    let got: Vec<f32> = a
        .cast(cast)
        .matmul(b.cast(cast))
        .cast(FloatDType::F32)
        .into_data()
        .try_to_vec_as()
        .unwrap();
    assert_eq!(got.len(), reference.len());
    for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
        assert!(
            (g - r).abs() <= eps * r.abs().max(1.0),
            "{cast:?} gemm[{i}]: {g} vs fp32 {r}"
        );
    }
}

/// The capability, in one test. Not `#[ignore]`d and not a TODO: this is the
/// property the backend swap was made for, and it holds on this box today.
#[test]
fn cpu_backend_does_f16_and_bf16() {
    narrow_matmul_matches_fp32(FloatDType::F16, 1e-2);
    narrow_matmul_matches_fp32(FloatDType::BF16, 5e-2);
    narrow_gemm_matches_fp32(FloatDType::F16, 1e-2);
    narrow_gemm_matches_fp32(FloatDType::BF16, 5e-2);
}
