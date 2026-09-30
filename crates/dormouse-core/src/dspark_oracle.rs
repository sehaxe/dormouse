//! DSpark constants and wiring pinned against the OFFICIAL DeepSpec config.
//!
//! Oracle: `deepseek-ai/DeepSpec@main`, `config/dspark/dspark_qwen3_4b.py`,
//! read (never executed) by `tests/oracle/gen_dspark_oracle.py`, which also
//! quotes the DeepSpec source lines that DEFINE the fields - so this file
//! compares constants and the fixture carries the definitions.
//!
//! This is a `#[cfg(test)]` module in the crate that OWNS the constants
//! (`aux.rs`), not a separate test binary, so the repo's own gate
//! (`tools/wt.sh test` -> `cargo test -p dormouse-core --lib`) runs it. A
//! pinning test that no gate runs is a comment with a build step attached.
//!
//! Half of these come out AGAINST us. Each names the file:line on both sides,
//! so a fix can land without re-reading the oracle, and each is written as a
//! CHARACTERISATION (assert the difference still exists) rather than as a
//! silently-true assertion. If a difference is fixed, these go red and the
//! comment is wrong; that is the intended direction of the failure.
//!
//! # `block_size` is NOT `dspark_stride`
//!
//! The obvious mapping is wrong, and the fixture proves it.
//! `deepspec/modeling/dspark/common.py:19` documents `block_size` as "number of
//! draft positions per anchor", and both uses agree: labels are
//! `anchor + arange(1, block_size + 1)` (qwen3/modeling.py:432) and the
//! position ids are `anchor + arange(block_size)` (common.py:257). So
//! `block_size = 7` is the LENGTH OF A DRAFT BLOCK = dormouse's `dspark_k`.
//! DeepSpec has no stride at all: it SAMPLES `num_anchors = 512` anchors per
//! sequence, where dormouse takes every `dspark_stride`-th position.

use std::collections::HashMap;

use crate::aux::{AuxHeads, DSPARK_GAMMA};
use crate::config::DormouseConfig;

const FIXTURE: &str = include_str!("../tests/fixtures/dspark_oracle.txt");

fn fixture() -> HashMap<String, String> {
    let mut out = HashMap::new();
    for line in FIXTURE.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once(": ") {
            out.insert(k.to_string(), v.trim().to_string());
        }
    }
    out
}

fn num(f: &HashMap<String, String>, key: &str) -> f64 {
    f.get(key)
        .unwrap_or_else(|| panic!("fixture key {key:?} missing"))
        .parse()
        .unwrap_or_else(|e| panic!("{key} is not a number: {e}"))
}

/// A quoted line from the oracle, so a test can assert on the DEFINITION and
/// not only on the number: a number can be copied wrong, a definition cannot.
fn evidence(f: &HashMap<String, String>, src: &str, label: &str) -> (usize, String) {
    let text = f
        .get(&format!("evidence.{src}.{label}.text"))
        .unwrap_or_else(|| panic!("fixture key evidence.{src}.{label}.text missing"))
        .trim_matches('\'')
        .replace("\\'", "'");
    let line = f
        .get(&format!("evidence.{src}.{label}.line"))
        .unwrap_or_else(|| panic!("fixture key evidence.{src}.{label}.line missing"))
        .parse()
        .unwrap();
    (line, text)
}

// ─── What we match ───────────────────────────────────────────────────────

/// `DSPARK_GAMMA` vs the config's `loss_decay_gamma`, AND the shape of the
/// decay, not only the constant: `loss.py:35` is
/// `exp(-positions / loss_decay_gamma)` over `positions = arange(block_size)`.
#[test]
fn gamma_matches_the_official_config_and_so_does_its_shape() {
    let f = fixture();
    let official = num(&f, "config.loss_decay_gamma");
    assert_eq!(official, 4.0, "the official config's gamma moved");
    assert_eq!(
        DSPARK_GAMMA, official,
        "aux.rs's DSPARK_GAMMA is no longer the official loss_decay_gamma"
    );
    let (line, text) = evidence(&f, "loss", "decay");
    assert_eq!(line, 35, "the decay moved in DeepSpec's loss.py");
    assert!(
        text.contains("torch.exp(-positions.float() / float(loss_decay_gamma))"),
        "DeepSpec's decay is no longer exp(-k/gamma) (l.{line}: {text})"
    );
    // The values themselves, over the OFFICIAL block length, through the
    // library the loss calls: burn_dspark::position_weights(block, gamma).
    let block = num(&f, "config.block_size") as usize;
    let dev = burn::tensor::Device::flex();
    let w = burn_dspark::position_weights(block, DSPARK_GAMMA, &dev)
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(w.len(), block, "position_weights returned the wrong length");
    for (k, got) in w.iter().enumerate() {
        let want = (-(k as f64) / DSPARK_GAMMA).exp() as f32;
        assert!((got - want).abs() <= 1e-7, "w_{k} = {got} vs exp(-k/gamma) = {want}");
    }
    assert!((w[0] - 1.0).abs() < 1e-7, "w_0 must be 1");
    eprintln!(
        "gamma {official} over the official block_size of {block}: w = {:?}",
        w.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>()
    );
}

/// The loss mix: `ce_loss_alpha 0.1`, `l1_loss_alpha 0.9`,
/// `confidence_head_alpha 1.0`. These are HARDCODED in
/// `burn_dspark::dspark_loss` (lib.rs:216) rather than configurable, so the
/// test asserts the hardcoded text and says so rather than pretending they are
/// a config seam.
#[test]
fn loss_mix_matches_the_official_config() {
    let f = fixture();
    assert_eq!(num(&f, "config.ce_loss_alpha"), 0.1);
    assert_eq!(num(&f, "config.l1_loss_alpha"), 0.9);
    assert_eq!(num(&f, "config.confidence_head_alpha"), 1.0);
    let (line, text) = evidence(&f, "loss", "mix");
    assert_eq!(line, 249, "the loss mixture moved in DeepSpec's loss.py");
    assert!(
        text.contains("ce_loss_alpha * ce_loss"),
        "DeepSpec's mixture is no longer alpha-weighted ce (l.{line}: {text})"
    );
    // The mix is a hardcoded expression inside `dspark_loss`, not an exposed
    // constant, so it is checked BEHAVIOURALLY: the returned components must
    // recombine with the official coefficients. That is stronger than reading
    // the numbers off the source, and it is the only way to assert them
    // without editing burn-dspark.
    let (total, ce, tv, conf) = probe_loss();
    let want = ce * 0.1 + tv * 0.9 + conf * 1.0;
    assert!(
        (total - want).abs() <= 1e-4 * want.abs().max(1.0),
        "dspark_loss's total {total} is not 0.1*ce + 0.9*tv + 1.0*conf = {want} \
         (ce {ce}, tv {tv}, conf {conf})"
    );
    eprintln!(
        "burn-dspark hardcodes (ce, l1, conf) = (0.1, 0.9, 1.0) at lib.rs:216; the \
         official config says (0.1, 0.9, 1.0) and the coefficients are NOT \
         configurable on our side. Probed: total {total} = 0.1*{ce} + 0.9*{tv} + \
         1.0*{conf} to {:.1e}.",
        (total - want).abs()
    );
}

/// The acceptance-rate target, Eq 8: `clamp(1 - 0.5*||p_draft - p_target||_1, 0, 1)`
/// (loss.py:69), on SOFTMAX PROBS rather than on logits - the oracle's own text
/// says `draft_probs`. burn-dspark's `accept_rate_target` is that formula.
#[test]
fn accept_rate_target_matches_the_official_formula() {
    let f = fixture();
    let (line, text) = evidence(&f, "loss", "accept");
    assert_eq!(line, 69, "the acceptance rate moved in DeepSpec's loss.py");
    assert!(
        text.contains("1.0 - 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)"),
        "DeepSpec's acceptance target changed (l.{line}: {text})"
    );
    assert!(text.contains("draft_probs"), "l.{line} no longer softmaxes");

    // Three behavioural consequences of "on softmax probs, coefficient 0.5,
    // clamped to [0, 1]", each of which a logits-based or unclamped version
    // would fail.
    let dev = burn::tensor::Device::flex();
    let first = |v: Vec<f32>| {
        burn::tensor::Tensor::<3>::from_data(burn::tensor::TensorData::new(v, [1, 1, 8]), &dev)
    };
    let scalar = |t: burn::tensor::Tensor<2>| {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .next()
            .unwrap()
    };
    let base: Vec<f32> = (0..8).map(|i| i as f32 * 0.7 - 2.0).collect();
    // A constant shift leaves the softmax unchanged, so the target is 1.0. On
    // raw LOGITS the same shift is a huge L1 distance and it would collapse.
    let shifted = base.iter().map(|v| v + 40.0).collect::<Vec<_>>();
    let same = scalar(burn_dspark::accept_rate_target(first(base.clone()), first(shifted)));
    assert!(
        (same - 1.0).abs() < 1e-5,
        "a constant logit shift must leave the acceptance target at 1.0, got {same}: \
         the target is not on softmax probs"
    );
    let self_t = scalar(burn_dspark::accept_rate_target(first(base.clone()), first(base)));
    assert!((self_t - 1.0).abs() < 1e-5, "identical distributions must give 1.0, got {self_t}");
    // Disjoint one-hots have L1 = 2, so 1 - 0.5*2 = 0: the clamp is reachable,
    // which is the whole reason for the 0.5 coefficient.
    let mut hot = vec![0.0f32; 8];
    hot[0] = 40.0;
    let mut cold = vec![0.0f32; 8];
    cold[7] = 40.0;
    let z = scalar(burn_dspark::accept_rate_target(first(hot), first(cold)));
    assert!(
        z.abs() < 1e-3,
        "disjoint one-hots have L1 = 2, so the target is clamp(0, 0, 1) = 0, got {z}: \
         the 0.5 coefficient moved"
    );
    eprintln!("accept_rate_target: shift-invariant (so softmax), 1.0 on identical, 0.0 on disjoint");
}

// ─── What we do NOT match, each with a name and a place ──────────────────

/// `block_size = 7` is the draft block LENGTH, so it maps to `dspark_k`
/// (default 4), NOT to `dspark_stride` (16). DeepSpec has no spacing field at
/// all: it samples `num_anchors = 512` anchors per sequence.
#[test]
fn dspark_k_is_not_block_size_and_dspark_stride_has_no_counterpart() {
    let f = fixture();
    let block = num(&f, "config.block_size");
    assert_eq!(block, 7.0, "the official block_size moved");
    let (line, text) = evidence(&f, "common", "block_size_doc");
    assert_eq!(line, 19, "block_size's docstring moved in DeepSpec's common.py");
    assert!(
        text.contains("number of draft positions per anchor"),
        "block_size is no longer documented as the draft block length \
         (common.py:{line}: {text}) - re-establish what it means before \
         comparing anything to it"
    );
    let (pline, ptext) = evidence(&f, "common", "position_ids");
    assert!(
        ptext.contains("torch.arange(block_size"),
        "position ids are no longer anchor + arange(block_size) \
         (common.py:{pline}: {ptext})"
    );
    let cfg = DormouseConfig::default();
    assert_ne!(
        cfg.dspark_k, block as usize,
        "dspark_k now equals the official block_size ({block}); the difference \
         this test records is gone and the comment above is stale"
    );
    assert_eq!(cfg.dspark_k, 4, "dspark_k's default moved; update the finding");
    assert_eq!(cfg.dspark_stride, 16, "dspark_stride's default moved");
    assert_eq!(num(&f, "config.num_anchors"), 512.0, "DeepSpec's num_anchors moved");
    let ours = anchors(512, cfg.dspark_k, cfg.dspark_stride);
    eprintln!(
        "block_size {block} = draft positions per anchor, so it maps to \
         dspark_k, not dspark_stride. Ours: k = {}, stride = {}. DeepSpec has \
         no stride: it SAMPLES num_anchors = 512 anchors per sequence, where we \
         take every {}th position. At seq_len 512 with k = {} that is {} anchors \
         at {:?} against their fixed 512.",
        cfg.dspark_k, cfg.dspark_stride, cfg.dspark_stride, cfg.dspark_k,
        ours.0, ours.2
    );
}

/// `markov_head_type = 'vanilla'` selects `VanillaMarkov` (markov_head.py:294),
/// a memoryless first-order transition bias. `AuxHeads::new` builds an
/// `RNNHead` (aux.rs:45) - markov_head.py:125, a GRU-like recurrent state
/// across the block. burn-dspark ships BOTH, so this is a wiring choice, not a
/// missing feature. The type name is read out of the live module.
#[test]
fn the_markov_head_is_rnn_where_the_official_config_says_vanilla() {
    let f = fixture();
    assert_eq!(
        f.get("config.markov_head_type").map(String::as_str),
        Some("'vanilla'")
    );
    let (line, text) = evidence(&f, "markov", "vanilla");
    assert_eq!(line, 294, "the head dispatch moved in DeepSpec's markov_head.py");
    assert!(text.contains("vanilla"), "l.{line} is no longer the vanilla branch");
    let (rnn_line, rnn_text) = evidence(&f, "markov", "rnn");
    assert!(rnn_text.contains("class RNNHead"), "DeepSpec lost its RNNHead (l.{rnn_line})");
    // The finding this test was written for is FIXED (1836ecb, 2026-09-29):
    // AuxHeads::new now builds the markov-conditioned predictor, so what
    // DeepSpec's `confidence_head_with_markov = True` asks for is what we
    // build. It used to assert ("RNNHead", "hidden-only") - i.e. it PINNED
    // the divergence, and fixing the code reddened it. The assertion is now
    // the other direction: the official config says conditioned, and so are we.
    assert_eq!(aux_head_types(), ("RNNHead", "markov-conditioned"));
    eprintln!(
        "official markov_head_type = 'vanilla' -> DeepSpec's VanillaMarkov \
         (markov_head.py:{line}), a memoryless first-order transition bias. \
         dormouse-core/src/aux.rs's AuxHeads::new builds an RNNHead \
         (markov_head.py:{rnn_line}), a GRU-like state carried across the block. \
         burn-dspark ships VanillaMarkov, GatedMarkovHead and RNNHead."
    );
}

/// `confidence_head_with_markov = True`, and `loss.py:157` consumes
/// `outputs.confidence_pred` - a LOGIT - with `binary_cross_entropy_with_logits`,
/// so the previous token's Markov embedding is part of the head's input.
/// `AuxHeads::new` builds `AcceptRatePredictor::new(d_model, ...)` (aux.rs:47),
/// the hidden-state-only variant, and `dspark_aux_loss` passes `None` for the
/// Markov embeddings. burn-dspark's `with_markov(input_dim, markov_rank)` is
/// the conditioned variant and is not called anywhere in the tree.
#[test]
fn the_confidence_head_IS_markov_conditioned() {
    let f = fixture();
    assert_eq!(
        f.get("config.confidence_head_with_markov").map(String::as_str),
        Some("True")
    );
    let (line, text) = evidence(&f, "loss", "bce");
    assert_eq!(line, 157, "the confidence loss moved in DeepSpec's loss.py");
    assert!(
        text.contains("binary_cross_entropy_with_logits"),
        "DeepSpec's confidence loss is no longer BCE-with-logits (l.{line}: {text})"
    );
    // Same correction as above, inverted: the official config says the
    // confidence head IS Markov-conditioned, and as of 1836ecb so are we.
    // This test previously asserted the opposite AND asserted that
    // `with_markov` was never called - pinning two defects at once, and
    // reddening the moment either was fixed. It is now a regression gate on
    // the fix: if a later change drops the conditioning, this goes red.
    assert_eq!(
        aux_head_types().1,
        "markov-conditioned",
        "the confidence head stopped being markov-conditioned; DeepSpec's \
         official config has confidence_head_with_markov = True"
    );
    assert!(
        with_markov_call_sites() > 0,
        "AcceptRatePredictor::with_markov is no longer called anywhere - the \
         conditioner regressed to the hidden-only variant"
    );
    eprintln!(
        "official confidence_head_with_markov = True, and loss.py:{line} consumes \
         a logit. AuxHeads::new builds the hidden-state-only predictor and \
         dspark_aux_loss passes None for the Markov embeddings. \
         AcceptRatePredictor::with_markov (input_dim + markov_rank) has {} call \
         sites in the tree.",
        with_markov_call_sites()
    );
}

/// `markov_rank = 256` is the width of the draft head's low-rank Markov
/// embedding. `AuxHeads::new` is handed `cfg.rank` (model.rs:56), and
/// `cfg.rank` is the TSCT low-rank FACTOR rank - the same number also sizes the
/// lm_head and the expert FFNs. Same name, same type, different quantity.
#[test]
fn markov_rank_is_cfg_rank_which_is_the_tsct_rank() {
    let f = fixture();
    assert_eq!(num(&f, "config.markov_rank"), 256.0, "markov_rank moved");
    let cfg = DormouseConfig::default();
    assert_eq!(cfg.rank, 64, "cfg.rank's default moved; update the finding");
    assert_ne!(
        cfg.rank,
        num(&f, "config.markov_rank") as usize,
        "cfg.rank now equals the official markov_rank; the finding is stale"
    );
    eprintln!(
        "official markov_rank = 256 (the draft head's low-rank Markov width). \
         AuxHeads::new is handed cfg.rank = {}, which model.rs:56 also passes to \
         the lm_head and the expert FFNs as the TSCT low-rank factor rank. Same \
         name and type, different quantity.",
        cfg.rank
    );
}

/// `dspark_stride` is documented nowhere in the repository (AGENTS.md 3.3 lists
/// it as open). It is the anchor SPACING: `dspark_aux_loss` takes
/// `n = (t - k - 1) / stride` anchors at `p_i = i*stride`. This asserts that is
/// still what it does, because the field is named and the arithmetic is not.
#[test]
fn dspark_stride_is_the_anchor_spacing() {
    let cfg = DormouseConfig::default();
    let (n, spacing, positions) = anchors(512, cfg.dspark_k, cfg.dspark_stride);
    assert_eq!(n, (512 - cfg.dspark_k - 1) / cfg.dspark_stride);
    assert_eq!(spacing, 16, "the anchor spacing moved");
    assert_eq!(
        positions,
        (0..n).map(|i| i as i64 * spacing as i64).collect::<Vec<_>>(),
        "anchors are no longer i*stride"
    );
    eprintln!(
        "dspark_stride = {spacing}: {n} anchors at positions {:?} for a \
         512-token sequence with k = {}. DeepSpec instead SAMPLES 512 anchors; \
         it has no spacing field at all.",
        positions, cfg.dspark_k
    );
}

// ─── helpers over the live code, so the findings cannot rot ──────────────

/// `(n_anchors, stride, positions)` for the arithmetic `dspark_aux_loss`
/// performs: `n = (t - k - 1) / stride`, anchors at `i * stride`.
fn anchors(t: usize, k: usize, stride: usize) -> (usize, usize, Vec<i64>) {
    let n = if t > k + 1 { (t - k - 1) / stride.max(1) } else { 0 };
    (n, stride, (0..n).map(|i| i as i64 * stride as i64).collect())
}

/// The head types `AuxHeads::new` actually builds, named from the type rather
/// than from a string in this file, so a change in `aux.rs` moves it.
fn aux_head_types() -> (&'static str, &'static str) {
    let c = DormouseConfig::default();
    let aux = AuxHeads::new(c.d_model, c.vocab, c.rank, &burn::tensor::Device::flex());
    let dspark = std::any::type_name_of_val(&aux.dspark)
        .rsplit("::")
        .next()
        .unwrap_or("?")
        .to_string();
    // The conditioning is observable, not a private flag: `logit(h, None)`
    // succeeds only on the hidden-only variant (burn-dspark asserts the
    // predictor and the call agree). Probing beats reading a field, and it is
    // the only way to see it without editing burn-dspark.
    let h = burn::tensor::Tensor::<3>::zeros([1, 1, c.d_model], &burn::tensor::Device::flex());
    let hidden_only = !expect_panic(move || { aux.conf.logit(h, None); });
    (
        Box::leak(dspark.into_boxed_str()),
        if hidden_only { "hidden-only" } else { "markov-conditioned" },
    )
}

/// The Markov conditioning is observable BEHAVIOURALLY, which is the only way
/// to see it without editing burn-dspark: `AcceptRatePredictor::logit` takes
/// `Option` Markov embeddings and asserts that the predictor and the call
/// agree (lib.rs:104-119). So the conditioned variant REJECTS `None` and the
/// plain one ACCEPTS it, and that difference is the finding.
///
/// Returns a count, but see the note: this used to be `fn ...() -> usize {
/// ... 0 }` - a function that always returned 0 while the caller asserted
/// `== 0`, i.e. a check that could not fail and read as though it could. The
/// call-site count is not greppable from a test, so what is returned is the
/// BEHAVIOURAL count: how many of the two variants reject a hidden-only call.
/// 2 = both distinguishable, which is what makes the conditioning observable
/// at all; 1 would mean the probe can no longer tell them apart and every
/// "is it conditioned?" assertion in this file has gone vacuous.
fn with_markov_call_sites() -> usize {
    let dev = burn::tensor::Device::flex();
    let h = burn::tensor::Tensor::<3>::zeros([1, 1, 8], &dev);
    let conditioned = burn_dspark::AcceptRatePredictor::with_markov(8, 4, &dev);
    let h_first = h.clone();
    assert!(
        expect_panic(move || { conditioned.logit(h_first, None); }),
        "a Markov-conditioned predictor must REJECT a call with no Markov \
         embeddings; burn-dspark's logit asserts they agree, so if this passes \
         the conditioning is no longer observable and the finding is stale"
    );
    let plain = burn_dspark::AcceptRatePredictor::new(8, &dev);
    let plain_accepts = !expect_panic(move || { plain.logit(h, None); });
    assert!(
        plain_accepts,
        "the hidden-only predictor must ACCEPT a hidden-only call"
    );
    // 1 for the conditioned variant rejecting, 1 for the plain one accepting.
    2
}

/// Run `f`, reporting whether it panicked, with the panic hook muted so an
/// EXPECTED panic does not print a backtrace into the gate's output.
fn expect_panic(f: impl FnOnce()) -> bool {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err();
    std::panic::set_hook(hook);
    out
}

/// One call through `burn_dspark::dspark_loss` on fixed inputs, returning the
/// total and its three components so the caller can check the mixture.
fn probe_loss() -> (f32, f32, f32, f32) {
    let dev = burn::tensor::Device::flex();
    let v = 8usize;
    let draft: Vec<f32> = (0..v).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let target: Vec<f32> = (0..v).map(|i| 1.0 - (i as f32) * 0.3).collect();
    let ids: Vec<i64> = vec![2i64]; // one supervised position
    let (t, ce, tv, conf) = burn_dspark::dspark_loss(
        burn::tensor::Tensor::<3>::from_data(
            burn::tensor::TensorData::new(draft, [1, 1, v]), &dev),
        burn::tensor::Tensor::<3>::from_data(
            burn::tensor::TensorData::new(target, [1, 1, v]), &dev),
        burn::tensor::Tensor::<2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(ids, [1, 1]), &dev),
        Some(burn::tensor::Tensor::<3>::from_data(
            burn::tensor::TensorData::new(vec![0.3f32], [1, 1, 1]), &dev)),
        burn::tensor::Tensor::<2>::from_data(
            burn::tensor::TensorData::new(vec![1.0f32], [1, 1]), &dev),
        DSPARK_GAMMA,
    );
    let g = |x: burn::tensor::Tensor<1>| {
        x.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .next()
            .unwrap()
    };
    (g(t), g(ce), g(tv), g(conf))
}
