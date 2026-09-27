//! The EMA JEPA teacher's input is a STRUCTURAL property, not a numeric one:
//! the teacher must consume the same tensors the student consumed, with the
//! same arms live. A latent built from anything else is still finite, still
//! plausible, still passes every "is it close / is it finite" check - which is
//! why the online teacher ran on the LABEL sequence with an inert Engram arm
//! (model.rs, `forward_with_hidden`) on every step of every run since the aux
//! objectives shipped, with no test able to see it.
//!
//! Two independent pins, neither a tolerance:
//!  1. `teacher_target_is_the_student_latent` - the live path's aux value must
//!     equal the aux value built from `forward_latent(x, Some(h), None)`, the
//!     tensor `precompute_jepa_targets` (dormouse-train) stores offline. Fed
//!     the label sequence instead, it differs by ~1e0 against a ~1e-5
//!     tolerance, and the test asserts that margin, so it cannot be passed by
//!     loosening a bound.
//!  2. `teacher_engram_arm_runs` - ADR-0011's integer counter: the memory
//!     branch must run TWICE per loop iteration with a teacher attached, once
//!     per network. An integer has no tolerance to widen.
//!
//! The mask is pinned to `frac = 1.0` - the start rate is then 1
//! (`1 - (1-frac)^(1/span)`), so every position is masked and the value does
//! not depend on the mask source at all.

use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn::backend::Autodiff;
use burn::tensor::{Device, Int, Tensor, TensorData};
use dormouse_core::{probe, DormouseConfig, DormouseModel};

type B = Autodiff<burn::backend::Flex, BalancedCheckpointing>;

fn device() -> Device {
    Device::flex().autodiff()
}

/// nano at a test-build width: every arm and aux weight that defines the
/// teacher/student seam stays nano's; only the widths shrink (the same cut as
/// tests/model_seam.rs, for the same reason - burn-flex in a dev-profile test
/// build is slow at full width).
fn mini() -> DormouseConfig {
    let nano = dormouse_core::config::load_config(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../configs/nano.toml"
    ))
    .expect("configs/nano.toml loads");
    DormouseConfig {
        d_model: 128,
        n_heads: 4,
        head_dim: 32,
        d_ffn: 256,
        rank: 32,
        engram_rows: 4096,
        // The JEPA term alone: `aux` must then BE the JEPA value, so an exact
        // comparison is possible. dspark_k stays > 0 because the aux gate is
        // `(jepa || dspark) && dspark_k > 0`.
        jepa_weight: 1.0,
        dspark_weight: 0.0,
        ..nano
    }
}

fn ids(bytes: &[u8], b: usize, t: usize, dev: &Device) -> Tensor<2, Int> {
    let v: Vec<i64> = bytes.iter().map(|&x| x as i64).collect();
    Tensor::from_data(TensorData::new(v, [b, t]), dev)
}

/// FNV-hashed n-gram keys `[b, t, 3]`, RAW as `dormouse_data` emits them,
/// narrowed to 30 bits: the full u32 does not fit this backend's checked
/// i64->i32 cast (burn-std `cast.rs:218` panics above i32::MAX, where CUDA
/// wraps). The low 12 bits - the 4096-row slot mask the model applies - are
/// untouched, so the row this test's model reads is the row a GPU run reads.
fn hashed(bytes: &[u8], b: usize, t: usize, dev: &Device) -> Tensor<3, Int> {
    let mut v = Vec::with_capacity(b * t * 3);
    for r in 0..b {
        let row = &bytes[r * t..(r + 1) * t];
        for p in 0..t {
            let e = p + 1;
            for n in [2usize, 3, 4] {
                let raw = dormouse_core::fnv_hash(&row[e.saturating_sub(n)..e]) as u32 as i64;
                v.push(raw & 0x3FFF_FFFF);
            }
        }
    }
    Tensor::from_data(TensorData::new(v, [b, t, 3]), dev)
}

fn aux_of(v: Option<burn::tensor::Tensor<1>>) -> f32 {
    v.expect("aux must be Some with jepa_weight > 0")
        .try_into_scalar()
        .expect("aux scalar")
}

/// Teacher and student are the same network (EMA at momentum 0, the train
/// loop's own init), and the student consumed `x`; so the only tensor a
/// correct JEPA target can be is the student's own latent. Anything else - the
/// label sequence, the Engram-less view - is a different value, and the test
/// pins how much of that difference it is willing to forgive.
#[test]
fn teacher_target_is_the_student_latent() {
    let dev = device();
    let cfg = mini();
    let mut model = DormouseModel::new(&cfg, &dev);
    // All-true mask (start rate 1), so the value carries no mask randomness.
    model.jepa_mask_frac = 1.0;
    let teacher = dormouse_core::aux::ema_update(model.clone(), &model, 0.0);

    let (b, s) = (2, 64);
    // `x` and `y` are maximally different sequences, so "fed the labels
    // instead of the input" is a large, unmistakable change - not a tolerance
    // question. The shift need not be real: what is under test is WHICH tensor
    // the teacher consumes, not what the labels happen to be.
    let x = ids(&vec![0u8; b * s], b, s, &dev);
    let y = ids(&vec![255u8; b * s], b, s, &dev);
    let h = hashed(&vec![7u8; b * s], b, s, &dev);
    // The offline target's definition, verbatim from `precompute_jepa_targets`.
    let reference = |t: Tensor<3>| -> f32 {
        aux_of(
            model
                .forward_with_jepa_targets::<B>(x.clone(), Some(h.clone()), None, Some(y.clone()), Some(t))
                .3,
        )
    };

    let correct = reference(model.forward_latent::<B>(x.clone(), Some(h.clone()), None));
    // The bug's value, through the same loss: labels instead of the input.
    let wrong = reference(teacher.forward_latent::<B>(y.clone(), Some(h.clone()), None));
    // ... and the input with the Engram arm inert. NOT asserted here: how much
    // the memory branch moves the value at init is a random draw (4x-800x the
    // tolerance over runs of this fixture), which is exactly why
    // `teacher_engram_arm_runs` pins that defect on an integer counter.
    let no_keys = reference(teacher.forward_latent::<B>(x.clone(), None, None));

    let got = aux_of(
        model
            .forward_with_hidden::<B>(x.clone(), Some(h.clone()), None, Some(y.clone()), Some(&teacher))
            .3,
    );

    let tol = 1e-5 * correct.abs().max(1.0);
    assert!(
        (got - correct).abs() <= tol,
        "the live teacher's latent must BE the student's latent: online {got:.6} \
         vs forward_latent(x, Some(h), None) {correct:.6} (tol {tol:.2e})"
    );
    // Teeth: the assertion above is worth nothing if the defect it exists for
    // moves the value less than the tolerance. A teacher fed the label
    // sequence moves it by ~1e0 against a ~1e-5 tolerance.
    assert!(
        (wrong - correct).abs() > 1e3 * tol,
        "fixture stopped discriminating: a teacher fed the label sequence moves \
         the JEPA value by only {:.2e} (tol {tol:.2e})",
        (wrong - correct).abs()
    );
    println!(
        "teacher_target_is_the_student_latent: online {got:.6} == correct {correct:.6} \
         (labels move it {:.2e}, inert engram {:.2e}, tol {tol:.2e})",
        (wrong - correct).abs(),
        (no_keys - correct).abs()
    );
}

/// ADR-0011 counter on the seam: one loop iteration of the memory branch is ONE
/// `engram_keys` per live network. The student alone is the baseline; the
/// teacher's own run is the delta. Integer equality - nothing to widen.
#[test]
fn teacher_engram_arm_runs() {
    let dev = device();
    let cfg = mini();
    let model = DormouseModel::new(&cfg, &dev);
    let teacher = dormouse_core::aux::ema_update(model.clone(), &model, 0.0);
    let (b, s) = (2, 64);
    let x = ids(&vec![0u8; b * s], b, s, &dev);
    let y = ids(&vec![255u8; b * s], b, s, &dev);
    let h = hashed(&vec![7u8; b * s], b, s, &dev);

    probe::reset();
    let _ = model.forward_with_hidden::<B>(x.clone(), Some(h.clone()), None, Some(y.clone()), None);
    let student = probe::count(probe::ENGRAM_KEYS);
    assert_eq!(
        student,
        cfg.max_iter as u64,
        "the student's Engram arm must read a row per iteration, else this \
         test proves nothing (arm off, or the keys were dropped)"
    );

    probe::reset();
    let _ = model.forward_with_hidden::<B>(
        x.clone(),
        Some(h.clone()),
        None,
        Some(y.clone()),
        Some(&teacher),
    );
    let both = probe::count(probe::ENGRAM_KEYS);
    assert_eq!(
        both,
        2 * student,
        "the teacher must run the SAME arms as the student: the memory branch \
         ran {student}x for the student alone and {both}x with the teacher \
         attached (a teacher fed `None` keys is inert and gives {student})"
    );
}
