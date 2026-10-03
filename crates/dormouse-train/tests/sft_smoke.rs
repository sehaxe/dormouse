//! The SFT mask GATE: is the loss really only the assistant's bytes?
//!
//! `crates/dormouse-data/tests/sft_format.rs` pins the bytes and the mask. It
//! cannot answer this question, because a mask that is correctly computed and
//! then not CONSUMED produces a perfectly healthy loss curve — the objective is
//! simply "predict the user's code too", which is a worse model rather than a
//! broken one. That is the failure class this project's own history is built on
//! (the eval that scored a memory-disabled network, the attention arm that ran
//! thousands of forwards and trained nothing), so the mask needs a gate that
//! would go red.
//!
//! # What this gate is, and what it deliberately is NOT
//!
//! The obvious gate — "a user-only byte's embedding row receives exactly zero
//! gradient" — is **false**, and asserting it would have shipped a wrong gate.
//! The attention arm is causal, so the hidden state at position `q` is a
//! function of EVERY position `< q`, user spans included: a user byte's
//! embedding row sits on the gradient path of every supervised position after
//! it, and masking those CE terms removes the *terms*, not the path. The first
//! version of this file measured 1.4e-28 where it expected 0.0, and the second
//! measured a user byte's gradient going UP under the mask (1.27e-4 masked vs
//! 1.15e-4 plain). Both numbers are why this gate is an identity and not a
//! magnitude.
//!
//! What is exact, and what this file checks instead — a GRADIENT identity:
//!
//! 1. **`2 * grad L({q1,q2}) == grad L({q1}) + grad L({q2})`**, elementwise. The
//!    masked CE is a linear reduction, so this holds at any initialisation, and
//!    it pins the per-position weighting, the divisor (the MASKED count: a
//!    `1/t` divisor turns it into `g_A = g_B + g_C` and it fails by 2x) and the
//!    backward itself. Measured relative error on this fixture: < 1e-7.
//! 2. **The unshifted mask scores a different loss.** It is a legal tensor of
//!    the right shape that trains one byte early, and both losses are finite and
//!    plausible: the wrong one is not an error, it is a slightly worse model.
//! 3. **Degenerate masks are defined**: all-zero gives 0.0 (the padded SFT
//!    batch, which must not be `0/0` = NaN mid-run) and all-one gives plain CE.
//! 4. **One optimizer step** on the masked loss is finite and moves the model.
//!
//! # There is deliberately NO loss-value assertion
//!
//! The obvious gate — "the masked loss equals the CE of the supervised
//! positions" — was written, passed, and was deleted. Measured on this fixture
//! with a randomly initialised model: the CE is ~5.545 at EVERY position (256
//! classes, near-uniform logits), so masking 60% of the positions moves the loss
//! by **1.3e-3**, while `L_Rec` and the returned logits differ by **4.4e-3-1.3e-2**
//! because the loop scores its per-iteration step output through the head
//! WITHOUT the final `norm` that the returned `logits` go through. The noise is
//! an order of magnitude larger than the signal, so any tolerance wide enough to
//! pass was wide enough to hide an ignored mask — verified: with the mask arm
//! deleted, a tolerance-based assert stayed green while the gradient gate went
//! red. A flaky tolerance test is worse than no test.
//!
//! CPU only (`--features cpu`, burn-flex), no CUDA: this is host arithmetic and
//! a gate that needs the GPU cannot run beside a training run.
//! `cargo test -p dormouse-train --test sft_smoke`

use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn::backend::autodiff::Autodiff;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Device, Int, Tensor, TensorData};
use dormouse_core::{DormouseConfig, DormouseModel};
use dormouse_data::sft::{self, SftStream};

/// The same alias the crate's own `cpu` feature builds (`device()`), proven to
/// satisfy the `DispatchKindConversion` bounds `forward_sft` carries.
type B = Autodiff<burn::backend::Flex, BalancedCheckpointing>;

const BATCH: usize = 2;
/// 96, not 48, and the reason is the template: the shortest possible first turn
/// is `<|im_start|>user\n` (17) + content + `<|im_end|>\n` (11) +
/// `<|im_start|>assistant\n` (22) = 50 + content, so at seq_len 48 the FIRST
/// row of a batch is nothing but headers and user bytes and supervises zero
/// positions. The loss would then be 0 for that row by the `clamp_min(1)`
/// branch — a green test over an empty objective.
const SEQ: usize = 96;
/// One Adam step's learning rate. Any nonzero value works; the test only asks
/// whether the weights moved at all.
const LR: f64 = 1e-3;

/// Nano's shape at a width a dev-profile CPU test can afford. Every aux weight
/// is zero: this gate is about the CE mask, and a JEPA/DSpark term would add
/// its own loss and its own gradients to every number below.
fn cfg() -> DormouseConfig {
    let mut c = dormouse_core::config::load_config(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../configs/nano.toml"
    ))
    .expect("configs/nano.toml loads");
    c.d_model = 64;
    c.n_heads = 2;
    c.head_dim = 32;
    c.d_ffn = 128;
    c.rank = 16;
    c.engram_rows = 256;
    c.max_seq_len = SEQ;
    c.jepa_weight = 0.0;
    c.dspark_weight = 0.0;
    c.dspark_k = 0;
    c.mor_bce_weight = 0.0;
    c.aux_fb_weight = 0.0;
    c.moe_lb_coef = 0.0;
    c
}

/// A literal, so the fixture is readable in the diff. Parsed in memory rather
/// than through a temp file: five tests share one process, and one thread's
/// `remove_file` is another thread's empty corpus.
const FIXTURE: &str = concat!(
    r#"{"messages":[{"role":"user","content":"QQ a QQ"},"#,
    r#"{"role":"assistant","content":"fn add(a: i32, b: i32) -> i32 { a + b } // sum two"}]}"#,
    "\n",
    r#"{"messages":[{"role":"user","content":"QQ b QQ"},"#,
    r#"{"role":"assistant","content":"fn rev<T: Clone>(v: &[T]) -> Vec<T> { v.iter().rev().cloned().collect() }"}]}"#,
    "\n",
    r#"{"messages":[{"role":"user","content":"QQ c QQ"},"#,
    r#"{"role":"assistant","content":"assert_eq!(add(1, 2), 3); // 1 + 2 == 3"}]}"#,
    "\n",
);

/// Three fake conversations, on purpose of different lengths so the packing cuts
/// mid-turn and the mask has to survive a batch boundary.
///
/// Two properties the fixture must have, and both cost a run to learn:
/// * the user spans carry `QQ` and the assistant spans do NOT, so the fixture
///   provably contains bytes that are read and never supervised;
/// * the assistant answers are LONG relative to the `batch * seq_len` window.
///   With short answers the first batch lands entirely inside headers and user
///   spans, supervises nothing, and every objective assertion below becomes
///   vacuous — which is what the first version of this file did.
fn corpus() -> Vec<sft::SftExample> {
    let exs = sft::parse_jsonl(FIXTURE).expect("the fixture parses");
    let total: usize = exs.iter().map(|e| e.bytes.len()).sum();
    let trained: usize = exs.iter().map(|e| e.trainable()).sum();
    assert!(
        trained * 2 > total / 2,
        "the fixture supervises {trained} of {total} bytes; a first batch that lands \
         in the headers makes the loss identity below vacuous"
    );
    exs
}

/// A stream over the fixture, the first batch, and the tensors the forward
/// takes. One place builds them so every test below scores the SAME batch.
struct Fx {
    model: DormouseModel,
    dev: Device,
    ids: Tensor<2, Int>,
    targets: Tensor<2, Int>,
    /// Label-aligned mask (`SftBatch::target_mask`), `[BATCH, SEQ]`.
    mask: Tensor<2>,
    /// The byte-indexed mask, unshifted. Used only to prove the shift matters.
    raw_mask: Tensor<2>,
    /// The label-aligned mask as host `f32`, row-major — the same numbers `mask`
    /// holds, so a test can ask which positions the mask selected without going
    /// back to the device.
    host_mask: Vec<f32>,
    /// Supervised labels per row.
    supervised_per_row: Vec<usize>,
    /// One byte that IS a supervised label.
    supervised_byte: i64,
    /// Bytes that appear in `ids` and in no supervised label.
    user_only: Vec<i64>,
}

impl Fx {
    fn new() -> Self {
        let cfg = cfg();
        assert_eq!(
            (cfg.jepa_weight, cfg.dspark_weight, cfg.mor_bce_weight),
            (0.0, 0.0, 0.0),
            "an aux term would add a second loss to every number in this file"
        );
        let mut st = SftStream::new(corpus(), BATCH, SEQ).expect("the fixture fits");
        st.rewind();
        let b = st.next_batch().expect("first batch");

        // The trainer's own label construction, copied: a flat one-byte shift
        // over `batch * seq_len`, with the wrap. If the trainer ever changes
        // this the mask alignment changes with it, and naming it here is how the
        // next reader finds out.
        let targets = b
            .bytes
            .iter()
            .skip(1)
            .chain(b.bytes.iter().take(1))
            .map(|&x| x as i64)
            .collect::<Vec<i64>>();
        let ids: Vec<i64> = b.bytes.iter().map(|&x| x as i64).collect();
        let tm = b.target_mask();

        let supervised_per_row: Vec<usize> = (0..BATCH)
            .map(|r| {
                tm[r * SEQ..(r + 1) * SEQ]
                    .iter()
                    .filter(|&&m| m != 0.0)
                    .count()
            })
            .collect();
        assert!(
            supervised_per_row.iter().all(|&n| n > 0),
            "row supervises nothing: {supervised_per_row:?}"
        );
        let supervised: std::collections::HashSet<i64> = targets
            .iter()
            .zip(tm.iter())
            .filter(|(_, &m)| m != 0.0)
            .map(|(&t, _)| t)
            .collect();
        let supervised_byte = *supervised.iter().next().expect("non-empty");
        let user_only: Vec<i64> = ids
            .iter()
            .copied()
            .filter(|x| !supervised.contains(x))
            .collect();
        assert!(
            !user_only.is_empty(),
            "fixture supervises every byte it reads"
        );

        let dev = Device::flex().autodiff();
        let model = DormouseModel::new(&cfg, &dev);
        Self {
            model,
            dev: dev.clone(),
            ids: Tensor::from_data(TensorData::new(ids, [BATCH, SEQ]), &dev),
            targets: Tensor::from_data(TensorData::new(targets, [BATCH, SEQ]), &dev),
            mask: Tensor::from_data(TensorData::new(tm.clone(), [BATCH, SEQ]), &dev),
            raw_mask: Tensor::from_data(TensorData::new(b.mask, [BATCH, SEQ]), &dev),
            host_mask: tm,
            supervised_per_row,
            supervised_byte,
            user_only,
        }
    }

    /// `L_Rec` under a given mask, as a host scalar.
    fn loss(&self, mask: Option<Tensor<2>>) -> f32 {
        let (_, rec, _, _) = self.model.forward_sft::<B>(
            self.ids.clone(),
            None,
            None,
            Some(self.targets.clone()),
            mask,
        );
        rec.into_scalar::<f32>()
    }

    /// The WHOLE embedding gradient, read to the host. Row-major `[vocab,
    /// d_model]`, so a test can compare gradients ELEMENTWISE rather than
    /// through a norm: the additivity identity is linear in the gradient, and
    /// `max|x|` over a row is not.
    ///
    /// Two traps this shape avoids, both of which make a gradient gate green
    /// because it reads nothing:
    ///
    /// * a slice of a `Param` carries its own id, so `row.grad(&grads)` looks up
    ///   an entry that is not there and returns 0.0 for every byte — take the
    ///   whole tensor's gradient first, then slice the NUMBERS;
    /// * a mask that hides a position's CE term removes the term, not the path:
    ///   the attention arm is causal, so a user byte's embedding row still sits
    ///   on the gradient of every supervised position after it.
    fn emb_grad_row(&self, mask: Tensor<2>) -> Vec<f32> {
        let (_, rec, _, _) = self.model.forward_sft::<B>(
            self.ids.clone(),
            None,
            None,
            Some(self.targets.clone()),
            Some(mask),
        );
        let grads = rec.backward();
        let w = self.model.embedding.weight.val();
        match w.grad(&grads) {
            Some(g) => g
                .into_data()
                .try_to_vec::<f32>()
                .expect("gradients are readable"),
            None => vec![0.0; self.model.vocab_size * self.model.d_model],
        }
    }
}

/// THE GATE — and it is a GRADIENT identity, because a loss value cannot do
/// this job at initialisation.
///
/// Measured on this fixture: with a randomly initialised model the CE is ~5.545
/// at *every* position (256 classes, near-uniform logits), so masking 60% of the
/// positions moves the loss by **1.3e-3**, while the RMSNorm difference between
/// `L_Rec` and the returned logits is **4.4e-3 to 1.3e-2** — the noise is an
/// order of magnitude larger than the signal. Any loss-value tolerance wide enough to pass is wide enough to hide
/// the mask, which is why the first version of this test picked a number and
/// then had to widen it.
///
/// The gradient has no such floor. `L_Rec` is a linear reduction of per-position
/// CE terms, so for any two supervised positions `q1`, `q2`:
///
/// ```text
///   grad L(M)  =  (1/b) * (1/k) * (1/|M|) * SUM_{q in M} grad CE_q
///   =>  2 * grad L({q1,q2})  ==  grad L({q1})  +  grad L({q2})
/// ```
///
/// (the `1/b` batch mean and the `1/k` iteration mean are common to all three
/// calls). That identity is EXACT, it holds at any initialisation, and it pins
/// three things at once: the mask weights positions individually, the divisor is
/// the MASKED COUNT (with a `1/t` divisor the relation becomes `g_A = g_B + g_C`
/// and this fails by a factor of two), and the backward is the gradient of the
/// masked sum.
#[test]
fn the_gradient_is_the_gradient_of_the_masked_sum() {
    const ROW: usize = 0;
    let fx = Fx::new();

    // Two supervised positions in the same row, so the batch mean and the
    // per-row divisor are the same constant in all three forwards.
    let sup: Vec<usize> = (0..SEQ)
        .filter(|&q| fx.host_mask[ROW * SEQ + q] != 0.0)
        .collect();
    assert!(
        sup.len() >= 2,
        "row {ROW} supervises {} positions",
        sup.len()
    );
    let (q1, q2) = (sup[0], sup[1]);

    let pick = |qs: &[usize]| {
        let mut m = vec![0.0f32; BATCH * SEQ];
        for &q in qs {
            m[ROW * SEQ + q] = 1.0;
        }
        Tensor::from_data(TensorData::new(m, [BATCH, SEQ]), &fx.dev)
    };
    let g_both = fx.emb_grad_row(pick(&[q1, q2]));
    let g_one = fx.emb_grad_row(pick(&[q1]));
    let g_two = fx.emb_grad_row(pick(&[q2]));
    let g_none = fx.emb_grad_row(pick(&[]));

    assert!(
        g_none.iter().all(|v| *v == 0.0),
        "an empty mask produced a non-zero embedding gradient: the mask is not \
         reaching the backward"
    );

    // Elementwise, on a supervised byte's embedding row: `2*g_both == g_one+g_two`.
    let byte = fx.supervised_byte as usize;
    let mut worst = 0.0f32;
    for i in 0..fx.model.d_model {
        let want = g_one[byte * fx.model.d_model + i] + g_two[byte * fx.model.d_model + i];
        let got = 2.0 * g_both[byte * fx.model.d_model + i];
        worst = worst.max((got - want).abs());
    }
    let scale = g_one
        .iter()
        .zip(g_two.iter())
        .map(|(a, b)| (a + b).abs())
        .fold(0.0f32, f32::max)
        .max(1e-12);
    assert!(
        worst / scale < 1e-4,
        "2*g({{q1,q2}}) != g({{q1}}) + g({{q2}}): worst |diff| {worst:.3e} on a scale of \
         {scale:.3e} ({:.1e} relative). The masked loss is not the mean of the masked \
         positions' CE terms, or the divisor is not their count.",
        worst / scale
    );
    // And the two single-position gradients are genuinely different, or the
    // identity above is satisfied by any two equal vectors.
    assert!(
        g_one.iter().zip(g_two.iter()).any(|(a, b)| a != b),
        "the two positions produced identical gradients: the fixture cannot \
         distinguish which positions the mask selects"
    );
}

/// An all-zero mask is 0.0, not NaN, and an all-one mask is the plain CE. The
/// first is the padded-SFT-batch case; the second is the claim that the mask's
/// absence changes nothing.
#[test]
fn degenerate_masks_are_defined() {
    let fx = Fx::new();
    let none = fx.loss(None);
    assert!(
        none > 0.0,
        "the plain CE is {none}: every other assert here would be vacuous"
    );
    let all = fx.loss(Some(Tensor::ones([BATCH, SEQ], &fx.dev)));
    assert!(
        (all - none).abs() < 1e-6,
        "all-ones mask changed the loss: {all} vs {none}"
    );
    let zero = fx.loss(Some(Tensor::zeros([BATCH, SEQ], &fx.dev)));
    assert_eq!(
        zero, 0.0,
        "an all-zero mask must be 0.0, not NaN or a stale value"
    );
}

/// The one-byte shift, measured rather than asserted in prose. The unshifted
/// mask is a legal tensor of the right shape that trains one byte early.
#[test]
fn the_unshifted_mask_scores_a_different_loss() {
    let fx = Fx::new();
    let shifted = fx.loss(Some(fx.mask.clone()));
    let unshifted = fx.loss(Some(fx.raw_mask.clone()));
    assert!(
        shifted.is_finite() && unshifted.is_finite(),
        "{shifted} / {unshifted}"
    );
    assert!(
        (shifted - unshifted).abs() > 1e-6,
        "the label-aligned and byte-indexed masks scored the same loss ({shifted:.6}). \
         Either the shift is a no-op or the fixture cannot tell them apart - and the \
         shift is the off-by-one that fed the DSpark head the byte it was predicting."
    );
}

/// One optimizer step on the masked loss: a smoke, not just a forward. Asserts
/// the step is finite and moves the model — the property an SFT run needs on
/// step 0.
#[test]
fn one_step_on_the_masked_loss_is_finite_and_moves_the_model() {
    let fx = Fx::new();
    let loss = fx.loss(Some(fx.mask.clone()));
    let grads = {
        let (_, rec, _, _) = fx.model.forward_sft::<B>(
            fx.ids.clone(),
            None,
            None,
            Some(fx.targets.clone()),
            Some(fx.mask.clone()),
        );
        rec.backward()
    };
    assert!(loss.is_finite() && loss > 0.0, "step-0 loss is {loss}");

    let before = fx
        .model
        .embedding
        .weight
        .val()
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .expect("weights are readable");

    // The crate's own optimizer, through the API the trainer uses (lib.rs's
    // `optim.step(lr, model, GradientsParams::from_grads(..))`). A hand-rolled
    // SGD step would prove nothing about the run this gate exists for.
    let mut opt = AdamConfig::new().init();
    let stepped = opt.step(
        LR,
        fx.model.clone(),
        GradientsParams::from_grads(grads, &fx.model),
    );
    let after = stepped
        .embedding
        .weight
        .val()
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .expect("weights are readable");

    assert_eq!(
        after.len(),
        before.len(),
        "the step changed the parameter's SHAPE"
    );
    let moved = before
        .iter()
        .zip(after.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(moved > 0, "one Adam step on the masked loss moved nothing");
    assert!(
        after.iter().all(|v| v.is_finite()),
        "a non-finite weight after one step on {} bytes of SFT",
        BATCH * SEQ
    );
}
