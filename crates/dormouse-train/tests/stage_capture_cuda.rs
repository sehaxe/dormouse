//! The stage seam's correctness gate, on the trainer's own backend (gate b of
//! the graph-stage lane).
//!
//! The claim `--graph-stage` makes: the captured teacher forward, REPLAYED,
//! writes the same latents the uncaptured teacher forward computes on the same
//! inputs — bit-for-bit, new batches included (a replay reads its input buffers
//! at replay time, so feeding a fresh batch through the pin and replaying must
//! produce THAT batch's latents, not the capture step's). This file is the
//! whole of the gate: a stage that is not bit-exact has no business in an A/B,
//! because it would silently change the JEPA objective.
//!
//! One capture at a time per device (`CU_STREAM_CAPTURE_MODE_GLOBAL` aborts on
//! any concurrent unsafe call), so this file holds exactly one capture test.
//! Needs the card (AGENTS §1.5).

#![cfg(feature = "cuda")]

use burn::tensor::{Int, Tensor};
use dormouse_core::DormouseConfig;
use dormouse_train::stage::StageSeam;

type Backend = dormouse_train::Backend;

const BATCH: usize = 2;
const SEQ: usize = 32;

fn device() -> burn::tensor::Device {
    burn::tensor::Device::cuda(0).autodiff()
}

/// The same shape the trainer runs, small enough to build in seconds: real KDA
/// arm, TSCT experts, in-VRAM engram OFF (the stage refuses the keyed arm).
fn model_cfg() -> DormouseConfig {
    let mut cfg = DormouseConfig::default();
    cfg.d_model = 64;
    cfg.n_heads = 4;
    cfg.head_dim = 16;
    cfg.max_iter = 3;
    cfg.n_experts = 2;
    cfg.use_tsct = true;
    cfg.rank = 8;
    cfg.use_engram = false;
    cfg.engram_rows = 16;
    cfg.jepa_weight = 0.05;
    cfg.dspark_weight = 0.0;
    cfg.use_mor = false;
    cfg.bf16 = false;
    cfg
}

/// Fixed bytes per step, so the differential is unambiguous.
fn batch(step: usize, dev: &burn::tensor::Device) -> (Tensor<2, Int>, Tensor<3, Int>) {
    let n = BATCH * SEQ;
    let x: Vec<i64> = (0..n).map(|i| ((i * 7 + step * 13) % 251) as i64).collect();
    let h: Vec<i64> = (0..n * 3).map(|i| ((i * 3 + step) % 97) as i64).collect();
    (
        Tensor::from_data(burn::tensor::TensorData::new(x, [BATCH, SEQ]), dev),
        Tensor::from_data(burn::tensor::TensorData::new(h, [BATCH, SEQ, 3]), dev),
    )
}

/// The teacher's forward, verbatim in shape from what `train_loop` hands the
/// stage: the pinned batch, the step's hashed keys, no host rows.
fn teacher_forward(
    teacher: &dormouse_core::DormouseModel,
    x: Tensor<2, Int>,
    h: Option<Tensor<3, Int>>,
) -> Tensor<3> {
    teacher.forward_latent::<Backend>(x, h, None)
}

/// f32 bit patterns, so "bit-comparable" is a comparison of bytes and not of
/// an epsilon a reviewer could argue with.
fn bits(t: Tensor<3>) -> Vec<u32> {
    t.into_data().try_to_vec::<f32>().expect("f32 latents").iter().map(|f| f.to_bits()).collect()
}

#[test]
fn a_captured_stage_replays_bit_identical_latents_for_new_batches() {
    let cfg = model_cfg();
    let dev = device();
    let model = dormouse_core::DormouseModel::new(&cfg, &dev);
    // The EMA teacher at momentum 0: same values as the model, params
    // no_grad-frozen — exactly what the trainer feeds the stage.
    let teacher =
        dormouse_core::aux::ema_update(model.clone(), &model, 0.0);

    // (x1, h1) is the capture step's batch; (x2, h2) arrives later, as a real
    // step would.
    let (x1, h1) = batch(1, &dev);
    let (x2, h2) = batch(2, &dev);

    // Uncaptured reference on the capture step's batch.
    let ref1 = bits(teacher_forward(&teacher, x1.clone(), Some(h1.clone())));

    // Arm and capture: the first graphed step.
    let client = dormouse_train::cubecl_client_opt(&dev)
        .expect("a CUDA device has a cubecl client");
    let mut seam = StageSeam::new(Some(client));
    let teacher = seam.arm(teacher);
    assert!(seam.armed(), "every parameter of this model has a pin representation");

    let latents1 = seam
        .step(false, &x1, |xin| teacher_forward(&teacher, xin, Some(h1.clone())))
        .expect("the capture must succeed on a clean card");
    assert_eq!(seam.stats.captures, 1, "one capture");
    assert_eq!(seam.stats.replays, 1, "the capture replays immediately: a recording does not execute");

    // GATE (b), half one: the replayed stage against the uncaptured forward,
    // same batch, same teacher.
    let got1 = bits(latents1);
    assert_eq!(got1.len(), ref1.len(), "same latent shape");
    let mismatches1: Vec<usize> =
        got1.iter().zip(ref1.iter()).enumerate().filter(|(_, (a, b))| a != b).map(|(i, _)| i).collect();
    assert!(
        mismatches1.is_empty(),
        "the replayed stage must be BIT-identical to the uncaptured forward on the capture \
         batch: {} of {} elements differ, first at {} ({:e} vs {:e})",
        mismatches1.len(),
        got1.len(),
        mismatches1.first().map_or(usize::MAX, |i| *i),
        f32::from_bits(got1[*mismatches1.first().unwrap_or(&0)]),
        f32::from_bits(ref1[*mismatches1.first().unwrap_or(&0)]),
    );

    // GATE (b), half two: a NEW batch through the pin, replayed — against the
    // uncaptured forward on the same new batch. This is the input-rewrite
    // proof: a replay that returned the capture step's latents would pass half
    // one and fail here.
    let ref2 = bits(teacher_forward(&teacher, x2.clone(), Some(h2.clone())));
    let latents2 = seam
        .step(false, &x2, |xin| teacher_forward(&teacher, xin, Some(h2.clone())))
        .expect("the replay must succeed");
    assert_eq!(seam.stats.captures, 1, "the stage must NOT re-capture for a new batch");
    assert_eq!(seam.stats.replays, 2);
    let got2 = bits(latents2);
    let mismatches2: Vec<usize> =
        got2.iter().zip(ref2.iter()).enumerate().filter(|(_, (a, b))| a != b).map(|(i, _)| i).collect();
    assert!(
        mismatches2.is_empty(),
        "the replayed stage must be BIT-identical to the uncaptured forward on a NEW batch: \
         {} of {} elements differ, first at {}",
        mismatches2.len(),
        got2.len(),
        mismatches2.first().map_or(usize::MAX, |i| *i),
    );
    // And the two batches really are different inputs — a stale graph would
    // otherwise pass by returning latents1 twice.
    assert_ne!(ref1, ref2, "the fixture must vary the batch, or the gate is blind");
}
