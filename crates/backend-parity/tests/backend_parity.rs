//! CPU/CUDA parity gate for the three backend precision bugs of ADR-0016.
//!
//! Every bug here was invisible to a CPU-only suite, so CPU/CUDA parity is
//! the deliverable, not a nicety:
//!
//! 1. `bool_tensor.float()` returned 0.0 for `true` on cubecl (the NaN
//!    firewall's counter read 0.0 every step and killed a healthy run).
//! 2. bf16 matmul is not offered by the CUDA backend at all.
//! 3. f16 matmul dies in the compiler with `builtin.fp16 to implement
//!    dyn SizedType`.
//!
//! On this stack the backend is not a type parameter - `Tensor`/`Device` are
//! backend-agnostic and the device picks the backend at runtime - so parity
//! here means ONE body of assertions run against two devices in one process:
//! `Device::ndarray()` and `Device::cuda(0)`. The `#[test]`s below call the same
//! functions; there is no second copy of an assertion to drift.
//!
//! Bugs 2/3 cannot be checked against ndarray at all: it declares
//! `DType::F16 | DType::BF16` an empty usage set
//! (`burn-ndarray/src/backend.rs:97`) and refuses the dtype. So their gate is
//! CUDA-only and self-referential - the narrow-dtype result must match the
//! **fp32 result of the same matmul on the same backend** to within that
//! dtype's own epsilon. (`burn-flex` DOES support both dtypes, so this could
//! widen to true two-backend parity; it is not used here because enabling
//! burn's `flex` feature re-fingerprints the entire burn stack for every crate
//! in the workspace. Do that as a standalone change, not from this crate.)
//!
//! Run: `cargo test -p backend-parity --features cuda --test backend_parity`
//! (the cuda half is `#[cfg(feature = "cuda")]` so a default run never
//! touches the GPU - and a default run proves nothing about this bug class).

use burn::tensor::{Bool, DType, Device, FloatDType, Tensor, TensorData};

/// `true` must read back as exactly 1.0 and `false` as exactly 0.0, on every
/// backend. The adversarial set is the one the firewall used in anger: a
/// single step, a whole log window, and the sizes that make cubecl pick a
/// different vector width (`max_vector_size` keys off the storage type and
/// the length, so 1/3/4/15/16/17 straddle every tile boundary).
#[track_caller]
fn assert_bool_to_float(device: &Device, data: &[bool]) {
    let n = data.len();
    let b = Tensor::<1, Bool>::from_data(TensorData::new(data.to_vec(), [n]), device);
    let f: Vec<f32> = b.float().into_data().try_to_vec_as().unwrap();
    let want: Vec<f32> = data.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect();
    assert_eq!(f, want, "bool.float() over {n} elems");
}

#[track_caller]
fn bool_to_float_parity(device: &Device) {
    // all-false and all-true: a cast that collapses to zero passes a mixed
    // test only if the "true" side is right, and the other way round.
    for &n in &[1usize, 3, 4, 8, 15, 16, 17, 64, 1000] {
        assert_bool_to_float(device, &vec![true; n]);
        assert_bool_to_float(device, &vec![false; n]);
        let mixed: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
        assert_bool_to_float(device, &mixed);
    }
    // A mask that came from a COMPARISON, not from from_bool: the cast has to
    // agree with the tensor the mask was derived from.
    let lhs = Tensor::<1>::from_data(
        TensorData::new(vec![-1.0f32, 0.0, 1.0, 2.0, -3.0, 4.0, 0.5, -0.5], [8]),
        device,
    );
    let mask = lhs.greater_elem(0.0);
    let got: Vec<f32> = mask.clone().float().into_data().try_to_vec_as().unwrap();
    assert_eq!(got, vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0], "comparison mask .float()");
    // and the negation of one, the shape the firewall's `bool_not` used
    let notted: Vec<f32> = mask
        .bool_not()
        .float()
        .into_data()
        .try_to_vec_as()
        .unwrap();
    assert_eq!(
        notted,
        vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        "bool_not().float()"
    );

    // 2-D, because every cubecl cast kernel is shaped by rank (a LinearView
    // over [m, n] is not the same launch as one over [m]).
    let m = Tensor::<2, Bool>::from_bool([[true, false], [false, true]], device);
    let f: Vec<f32> = m.float().into_data().try_to_vec_as().unwrap();
    assert_eq!(f, vec![1.0, 0.0, 0.0, 1.0], "2-D bool.float()");

    // The sibling op that shares the kernel: bool -> int. Reported broken in
    // the same breath as the float cast; cheap to keep honest.
    let i: Vec<i64> = Tensor::<1, Bool>::from_data(
        TensorData::new(vec![true, false, true, false, true], [5]),
        device,
    )
    .int()
    .into_data()
    .try_to_vec_as()
    .unwrap();
    assert_eq!(i, vec![1, 0, 1, 0, 1], "bool.int()");
}

#[test]
fn bool_to_float_ndarray() {
    bool_to_float_parity(&Device::ndarray());
}

#[cfg(feature = "cuda")]
#[test]
fn bool_to_float_cuda() {
    bool_to_float_parity(&Device::cuda(0));
}

/// The same body again through the **autodiff (dispatch) tensor type** - which is
/// the path the trainer actually runs, and the one the production NaN firewall
/// was built on. A raw-backend pass here would prove nothing: the raw and the
/// dispatch paths are different code in burn (`Dispatch::bool_into_float` vs
/// `BoolTensorOps::bool_into_float`) and only the second one is a `Tensor` the
/// optimizer ever sees.
#[cfg(feature = "cuda")]
#[test]
fn bool_to_float_cuda_autodiff() {
    bool_to_float_parity(&Device::cuda(0).autodiff());
}

#[cfg(feature = "cuda")]
#[test]
fn bool_to_float_ndarray_autodiff() {
    bool_to_float_parity(&Device::ndarray().autodiff());
}

// ---------------------------------------------------------------------------
// What the cast was BLAMED for
// ---------------------------------------------------------------------------
//
// `bool.float()` is correct on both backends (all four tests above), so the
// 2026-09-27 production miscount has to come from the expression built ON TOP
// of it. `.bulba/memory.md:15` blames the cast; `.bulba/memory.md:20` names
// something else and these two tests are the check for it. Both run on both
// backends, because both are pure burn-API semantics, not CUDA quirks.

/// The device-side counter the firewall actually used: a 0/1 float indicator
/// made by `zeros_like().mask_fill(mask, 1.0)`. It must count `true` and
/// nothing else - and the "fed into `add`" half matters too, because that is
/// where a wrong indicator stops being a number and becomes a step skip.
#[track_caller]
fn device_side_bool_indicator(device: &Device) {
    for n in [1usize, 7, 16, 33, 1000] {
        let flags: Vec<bool> = (0..n).map(|i| i % 5 == 0).collect();
        let expect = flags.iter().filter(|&&b| b).count() as f32;
        let mask = Tensor::<1, Bool>::from_data(TensorData::new(flags.clone(), [n]), device);
        // `zeros_like(mask).mask_fill(mask, 1.0)`: note mask_fill takes the
        // MASK's shape, not the tensor's - a [1] zeros with a [7] mask comes
        // back [7]. Build it the way the trainer did, same shape.
        let indicator = Tensor::<1>::zeros([n], device).mask_fill(mask, 1.0);
        let v: Vec<f32> = indicator.clone().into_data().try_to_vec_as().unwrap();
        assert_eq!(v, flags.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect::<Vec<_>>());
        // ... and the accumulation, which is how it was read.
        assert_eq!(
            indicator.sum().into_scalar::<f32>(),
            expect,
            "counter sum at n={n}"
        );
    }
}

/// The aliasing half. `clone()` on a CUDA tensor SHARES the device buffer, and
/// `mask_fill` writes in place, so a "raw" copy taken before a mask is applied
/// can read the masked value - which is exactly how a run logged `ce=0.000`
/// with `best` stuck at 0 forever (memory.md:20). A clone must be a snapshot.
#[track_caller]
fn clone_is_a_snapshot_before_mask_fill(device: &Device) {
    let x = Tensor::<1>::from_data(TensorData::new(vec![3.0f32, -1.0, 7.5, 0.0], [4]), device);
    let raw = x.clone();
    let mask = Tensor::<1, Bool>::from_bool([false, true, false, false], device);
    let masked = x.mask_fill(mask, 0.0);
    let raw_after: Vec<f32> = raw.into_data().try_to_vec_as().unwrap();
    let masked_v: Vec<f32> = masked.into_data().try_to_vec_as().unwrap();
    assert_eq!(
        raw_after,
        vec![3.0, -1.0, 7.5, 0.0],
        "a clone must not see a later in-place mask_fill"
    );
    assert_eq!(masked_v, vec![3.0, 0.0, 7.5, 0.0]);
}

#[test]
fn device_side_bool_indicator_ndarray() {
    device_side_bool_indicator(&Device::ndarray());
    clone_is_a_snapshot_before_mask_fill(&Device::ndarray());
}

#[cfg(feature = "cuda")]
#[test]
fn device_side_bool_indicator_cuda() {
    device_side_bool_indicator(&Device::cuda(0));
    clone_is_a_snapshot_before_mask_fill(&Device::cuda(0));
}

// ---------------------------------------------------------------------------
// bugs 2 + 3: bf16 / f16 matmul (CUDA only - the CPU backend has neither)
// ---------------------------------------------------------------------------

/// Values that break a naive "it compiled, so it works" reading: an exact 1.0
/// (identity), 2.0 / 0.5 / -1.5 (sign and rounding), a zero and a -0.0.
const K: usize = 8;
#[rustfmt::skip]
const K_VALUES: [f32; K] = [1.0, 2.0, 0.5, -1.5, 3.25, 0.0, -0.0, 1.0];

/// f16 eps is 2^-11 = 4.9e-4 and bf16 eps is 2^-8 = 3.9e-3, so 1e-2 and
/// 5e-2 relative are the honest bands for an 8-64 term dot.
#[track_caller]
fn narrow_matmul_matches_fp32(device: &Device, cast: FloatDType, eps: f32) {
    let a = Tensor::<2>::from_data(TensorData::new(K_VALUES.to_vec(), [1, K]), device);
    let b = Tensor::<2>::from_data(TensorData::new(K_VALUES.to_vec(), [K, 1]), device);
    let reference: f32 = a
        .clone()
        .cast(FloatDType::F32)
        .matmul(b.clone().cast(FloatDType::F32))
        .into_scalar();

    let an = a.cast(cast);
    assert_eq!(an.dtype(), DType::from(cast), "cast produced the wrong dtype");
    let got: f32 = an.matmul(b.cast(cast)).cast(FloatDType::F32).into_scalar();
    assert!(
        (got - reference).abs() <= eps * reference.abs().max(1.0),
        "{cast:?} 1xK @ Kx1 matmul: {got} vs fp32 {reference}"
    );
}

/// The round that actually runs in a trainer: a [m, k] @ [k, n] GEMM in the
/// narrow dtype, where vectorization and tiling differ from the dot above.
#[track_caller]
fn narrow_gemm_matches_fp32(device: &Device, cast: FloatDType, eps: f32) {
    let (m, k, n) = (64usize, 96usize, 48usize);
    let av: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
    let bv: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) / 5.0).collect();
    let a = Tensor::<2>::from_data(TensorData::new(av, [m, k]), device);
    let b = Tensor::<2>::from_data(TensorData::new(bv, [k, n]), device);
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

/// f16 matmul is CORRECT on this stack (verified: matches the fp32 result of the
/// same product to 1e-2, both a 1xK@Kx1 dot and a 64x96x48 GEMM). What is broken
/// is the *tensor-core* candidate: it dies at kernel-compile time with
/// `Expected type builtin.fp16  to implement dyn SizedType` and the autotuner
/// quietly falls back to a non-accelerated routine. So this test is the guard
/// against that fallback ever becoming WRONG, and the perf cost of it is
/// ADR-0016's open item, not this test's business.
#[cfg(feature = "cuda")]
#[test]
fn f16_matmul_cuda() {
    narrow_matmul_matches_fp32(&Device::cuda(0), FloatDType::F16, 1e-2);
}

#[cfg(feature = "cuda")]
#[test]
fn f16_gemm_cuda() {
    narrow_gemm_matches_fp32(&Device::cuda(0), FloatDType::F16, 1e-2);
}

/// bf16 matmul CANNOT work on this backend and that is not fixable here
/// (ADR-0016 bug 2): the LLVM dialect cubecl lowers through has no bf16 type, so
/// the failure is not even in the matmul - the f32 readback of a bf16 buffer
/// dies first, with `Type cube.bf16 does not have a conversion to LLVM type
/// implemented`.
///
/// `#[ignore]`d BY DESIGN, the same convention as dormouse-spectral's own bf16
/// probe: this is a report, not a requirement, and a permanently red test is
/// not a gate. Run it when you want the proof the gap is still there:
/// `cargo test -p backend-parity --features cuda --test backend_parity -- --ignored`
/// It flips to red-green the day someone lands a bf16 type in the dialect.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "design gap, ADR-0016 bug 2: the LLVM dialect has no bf16 type"]
fn bf16_matmul_cuda() {
    narrow_matmul_matches_fp32(&Device::cuda(0), FloatDType::BF16, 5e-2);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "design gap, ADR-0016 bug 2: the LLVM dialect has no bf16 type"]
fn bf16_gemm_cuda() {
    narrow_gemm_matches_fp32(&Device::cuda(0), FloatDType::BF16, 5e-2);
}
