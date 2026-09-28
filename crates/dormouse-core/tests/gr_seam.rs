//! The Gated Residual seam: what the LOOP does with `read`/`write`, which is
//! where the operator stops being the report's Eq. 30-34.
//!
//! `gr.rs`'s own tests pin the EQUATIONS (a host reference computed from the
//! module's own weights). This file pins the loop around them, and the thing it
//! pins is ORDERING.
//!
//! Before 2026-09-28 `step_out` was taken from `h = h_ctx` - the state BEFORE
//! `gr.write` - and `h` was reset to `h_ctx` at the end of every iteration. So
//! iteration n's readout never saw iteration n's own deposit: GR was a
//! depth-(iters-1) model, and at `max_iter = 1` the entire block body (KDA, the
//! memory arm, every expert FFN, the controller) was computed and discarded,
//! leaving `out_proj(read(h0) + e_0)`.
//!
//! The assertion is a DEPENDENCE, not a value: the output must change when
//! the block body changes. Every comparison here is one model against a
//! MUTATED COPY OF ITSELF, so the only difference is the body - two separately
//! constructed models differ in their random weights and the test would pass
//! for the wrong reason (it did, until this was rewritten).
//!
//! Run: `cargo test -p dormouse-core --test gr_seam`

use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn::backend::Autodiff;
use burn::module::Param;
use burn::tensor::{Device, Int, Tensor, TensorData};
use dormouse_core::loop_block::ExpertFFN;
use dormouse_core::param::LinearLikeInner;
use dormouse_core::{fnv_hash, DormouseConfig, DormouseModel};

// NdArray, like the rest of this crate's CPU tests. Nothing here is
// device-specific: what is under test is loop ORDER.
type B = Autodiff<burn::backend::NdArray, BalancedCheckpointing>;

const BATCH: usize = 2;
const SEQ: usize = 16;

fn device() -> Device {
    Device::ndarray().autodiff()
}

/// A GR model small enough for a CPU forward. `use_tsct = false` so the
/// expert FFNs are plain dense linears whose weights a test may zero - that
/// is the only mutation needed to switch the block body off without touching
/// a config field, which is what keeps the two arms' weights identical.
fn gr_cfg() -> DormouseConfig {
    DormouseConfig {
        d_model: 32,
        n_heads: 2,
        head_dim: 16,
        d_ffn: 64,
        rank: 8,
        max_seq_len: SEQ,
        engram_rows: 256,
        n_experts: 2,
        use_gr: true,
        use_tsct: false,
        // The aux heads are a separate objective and a second forward.
        jepa_weight: 0.0,
        dspark_weight: 0.0,
        ..DormouseConfig::default()
    }
}

struct Batch {
    x: Tensor<2, Int>,
    h: Tensor<3, Int>,
}

fn batch(dev: &Device) -> Batch {
    // Deterministic, non-degenerate bytes: a constant row makes an
    // RMSNorm'd branch degenerate and a constant target makes CE meaningless.
    let bytes: Vec<u8> = (0..BATCH * SEQ)
        .map(|i| ((i * 37 + 11) % 251) as u8)
        .collect();
    let ids: Vec<i64> = bytes.iter().map(|&b| b as i64).collect();
    let mut hashes = Vec::with_capacity(BATCH * SEQ * 3);
    for r in 0..BATCH {
        let row = &bytes[r * SEQ..(r + 1) * SEQ];
        for p in 0..SEQ {
            let e = p + 1;
            for n in [2usize, 3, 4] {
                hashes.push((fnv_hash(&row[e.saturating_sub(n)..e]) as u32) as i32 as i64);
            }
        }
    }
    Batch {
        x: Tensor::from_data(TensorData::new(ids, [BATCH, SEQ]), dev),
        h: Tensor::from_data(TensorData::new(hashes, [BATCH, SEQ, 3]), dev),
    }
}

fn logits(model: &DormouseModel, b: &Batch) -> Vec<f32> {
    model
        .forward::<B>(b.x.clone(), Some(b.h.clone()))
        .into_data()
        .try_to_vec()
        .expect("logits to host")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Switch the expert FFN half of the block body off: every `down` projection
/// becomes zero, so the FFN term of `y = attn + memory + ffn` vanishes and
/// nothing else in the model moves. This is the mutation the ordering test
/// needs - it changes `y` and only `y`.
fn silence_expert_ffn(model: &mut DormouseModel, dev: &Device) {
    for e in 0..model.loop_block.expert_ffns.len() {
        let ExpertFFN { down, .. } = &mut model.loop_block.expert_ffns[e];
        let LinearLikeInner::Dense(l) = &mut down.inner else {
            panic!("gr_seam needs use_tsct = false to zero an expert weight");
        };
        let [inp, out] = l.weight.val().dims();
        l.weight = Param::initialized(l.weight.id, Tensor::zeros([inp, out], dev));
        l.bias = None;
    }
}

/// The core ordering property: the readout of iteration n must contain
/// iteration n's OWN block body, at every depth - and depth 1 is where the
/// old ordering lost it completely.
#[test]
fn the_readout_contains_this_iterations_block_body_at_every_depth() {
    let dev = device();
    let b = batch(&dev);
    for depth in 1..=3 {
        let mut model = DormouseModel::new(&gr_cfg(), &dev);
        model.loop_block.set_depth(Some(depth));
        let with_body = logits(&model, &b);
        silence_expert_ffn(&mut model, &dev);
        let without_body = logits(&model, &b);
        let d = max_abs_diff(&with_body, &without_body);
        assert!(
            d > 1e-6,
            "GR at depth {depth}: silencing the expert FFN moved the logits by {d:.3e} - \
             the readout is not seeing iteration {depth}'s own block body"
        );
    }
}

/// The same property on the ReZero arm, so a future edit cannot satisfy the
/// GR test by breaking the shared loop body, and so the two arms are known to
/// agree on WHERE the body enters, differing only in the residual operator.
#[test]
fn the_rezero_readout_also_contains_the_body_at_every_depth() {
    let dev = device();
    let b = batch(&dev);
    for depth in 1..=3 {
        let cfg = DormouseConfig { use_gr: false, ..gr_cfg() };
        let mut model = DormouseModel::new(&cfg, &dev);
        model.loop_block.set_depth(Some(depth));
        let with_body = logits(&model, &b);
        silence_expert_ffn(&mut model, &dev);
        let without_body = logits(&model, &b);
        let d = max_abs_diff(&with_body, &without_body);
        assert!(
            d > 1e-6,
            "ReZero at depth {depth}: the readout lost the block body \
             (max |dlogit| = {d:.3e})"
        );
    }
}

/// Depth 1 must be a real model of depth 1, not a re-parameterized embedding.
/// Under the old ordering the GR readout at depth 1 was
/// `out_proj(read(h0) + e_0)` - the depth-2 average contained the same term
/// plus one more copy of it, so the two depths differed only by the
/// iteration-embedding row, which is a much weaker claim than "iteration 2
/// ran". One model, two depths, via `set_depth`, so the weights are identical.
#[test]
fn depth_one_is_not_depth_one_iteration_of_depth_two() {
    let dev = device();
    let b = batch(&dev);
    let mut model = DormouseModel::new(&gr_cfg(), &dev);
    model.loop_block.set_depth(Some(1));
    let d1 = logits(&model, &b);
    model.loop_block.set_depth(Some(2));
    let d2 = logits(&model, &b);
    let d = max_abs_diff(&d1, &d2);
    assert!(
        d > 1e-6,
        "GR depth 1 and depth 2 agree (max |dlogit| = {d:.3e}); \
         the second iteration is not reaching the readout"
    );
}
