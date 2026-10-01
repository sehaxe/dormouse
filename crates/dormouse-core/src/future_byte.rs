//! The FUTURE-BYTE auxiliary head: predict the byte at `t + k`, in CE.
//!
//! # What this is, and what it is not
//!
//! **Our adaptation of arXiv 2404.19737's per-horizon multi-token heads to a
//! byte-level autoregressive model.** The paper's external reference is its
//! own §3 (a separate, untied output head per lookahead horizon, each trained
//! with plain CE on the token that far ahead); the claim reused here is
//! *induction-head gains at 1-30M scale appear when the model is trained to
//! predict ahead*, not the paper's exact shape, which it does not prescribe
//! for a 256-symbol byte vocabulary on a 9M-parameter recurrent block. Nothing
//! in this file is "verified against 2404.19737" - the mechanism has never
//! been run here, and the A/B that would say whether it earns its 197 120
//! parameters is queue row 1b.
//!
//! It is also NOT the JEPA retune, and the choice between them is recorded in
//! `docs/decisions/2026-09-30-do-we-need-jepa.md`: CE against a real future
//! byte preserves the ambiguity of the continuation (a byte has one right
//! answer but many reasonable ones, and CE scores the whole 256-way
//! distribution), where a latent-target regression L1s against one EMA
//! teacher's guess. And it is not DSpark: no weight tying, no shared `W1`, no
//! draft window, no acceptance head - DSpark is disabled project-wide and
//! carries its own bug history, and a head that shares machinery with a
//! disabled, once-broken arm inherits the part you cannot see.
//!
//! # The head
//!
//! One `LinearLike`, `d_model -> vocab`, **untied** from `lm_head`. Untied for
//! the paper's reason (separate heads per horizon, so each horizon's head can
//! specialise) and for ours: a tied head would put the future-byte gradient
//! into `lm_head`, and the arm's whole claim is that it shapes the BACKBONE
//! (through the hidden state) without spending the main head's capacity. The
//! gradient must never touch `lm_head`; `the_gradient_does_not_reach_the_lm_head`
//! is that claim, measured.
//!
//! It reads the same hidden state the main head reads: `h = norm(out_acc)`,
//! the T-averaged loop readout, `[b, t, d]`. The per-ITERATION states
//! (`[T, b, t, d]`) are a different and more expensive arm - they are what
//! would let the head see how the loop's answer refined - and they are
//! deliberately not this one. One readout, two heads over it, so the
//! comparison the A/B makes is "an extra CE on a future byte" and nothing else.
//!
//! # Which labels are read, and why the last one is not
//!
//! `targets` is the next-byte label sequence: `targets[q]` is the byte at
//! `q + 1`. The trainer builds it as `bytes[1..] ++ [bytes[0]]`
//! (`dormouse-train/src/lib.rs:1117`), so the LAST entry is a wraparound - the
//! first byte of the batch again, which the model at position `t - 1` has no
//! reason to predict. So the label for position `q` at horizon `k` is
//! `targets[q + k]`, and the largest index this head may read is `t - 2`:
//!
//! ```text
//! n = t - k - 1        valid positions, q = 0 .. n-1
//! labels = targets[:, k .. t-1]        (width t-1-k = n)
//! ```
//!
//! No wraparound, no padding invention, no invented bytes: a shorter sequence
//! contributes fewer positions and the loss is the MEAN over the ones that
//! exist, so a `t` barely above `k` contributes few terms at full weight rather
//! than a term padded with made-up labels. `n = 0` (a sequence no longer than
//! the horizon) returns a real zero and bumps NO `FUTURE_BYTE` count, against
//! the `FUTURE_BYTE_ASKED` it does bump - COUNTED, not silent (ADR-0019),
//! because a horizon longer than the sequence is a config that trains for
//! thousands of steps and produces no objective at all.
//!
//! `future_byte_loss_uses_only_shifted_targets` pins that index set one entry
//! at a time, because the DSpark window read the label sequence where it needed
//! the consumed one, was handed the answer one step early, and invalidated
//! every DSpark number this project has ever recorded (2026-09-29). The
//! failure mode is not exotic; the gate is the vaccine.

use burn::backend::DispatchKindConversion;
use burn::tensor::{DispatchTensor, Int, Tensor, activation::log_softmax};

use crate::param::LinearLike;

/// Number of parameters the head adds on a `small`-shaped model, quoted in
/// [`crate::config::schema::DormouseConfig::aux_fb_weight`]. Dense
/// `nn::Linear(d_model, vocab)` with a bias, no low-rank factorisation: the
/// head is a readout like `lm_head` and is routed to `Group::Rest`
/// (`routing::Role::Head` is `lm_head`'s; an unclaimed parameter lands in Rest
/// via `rest_of`, and `Routing::check` is what would say so out loud).
/// `197 120 = 256*768 + 256` against `small`'s measured 9 197 390 is **2.1%** -
/// the price of the arm, quoted so the A/B row is not a surprise.
pub const FB_PARAMS_AT_SMALL: usize = 256 * 768 + 256;

/// The valid-position count for a sequence of `t` at horizon `k`: `t - k - 1`.
/// The `- 1` is the wraparound entry at `targets[t-1]`; see the module docs.
pub fn valid_positions(t: usize, k: usize) -> usize {
    t.saturating_sub(k.saturating_add(1))
}

/// CE of an independent `d_model -> vocab` head at horizon `k`.
///
/// `h` is the T-averaged readout `[b, t, d]`, `targets` the next-byte label
/// sequence `[b, t]`. Returns `[1]`: the mean over the `b * n` valid
/// (position, label) pairs, or an exact zero when `n == 0`.
///
/// Gradients flow from this term into `h` (so into the backbone) and into the
/// head, and into nothing else: no `lm_head`, no teacher, no embedding.
pub fn future_byte_loss<B: burn::backend::AutodiffBackend>(
    head: &LinearLike,
    h: Tensor<3>,
    targets: Tensor<2, Int>,
    k: usize,
) -> Tensor<1>
where
    DispatchTensor: DispatchKindConversion<B>
        + DispatchKindConversion<B::InnerBackend>
        + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
{
    let [b, t, d] = h.dims();
    let n = valid_positions(t, k);
    if n == 0 {
        // The arm was ASKED for; it produced nothing. The caller has already
        // bumped `FUTURE_BYTE_ASKED`, and the `fb=<ran>/<asked>` field on the
        // eval line is what makes that difference visible. Not an error: a
        // short sequence is a legal input, a horizon longer than every
        // sequence is a config mistake `config::validate` cannot see (it does
        // not know `seq_len` at the config layer... it does, but the horizon
        // is allowed to exceed it legitimately for a short-sequence sweep).
        return Tensor::zeros([1], &h.device());
    }
    crate::probe::note(crate::probe::FUTURE_BYTE);
    // `targets[:, k .. t-1]`: the shifted labels, last (wraparound) entry out.
    let labels = targets.slice([0..b, k..t - 1]).reshape([b * n, 1]);
    // `h[:, 0 .. n]`: the same rows. Sliced BEFORE the head so the head runs on
    // n rows, not t, and so the two slices are visibly the same arithmetic.
    let logits = head.forward::<B>(h.slice([0..b, 0..n, 0..d]).reshape([b * n, d]));
    // Gathered log-prob, not one-hot (the same 2026-09-04 swap the in-loop CE
    // made): the loss is `-mean(log_softmax(logits)[label])`.
    log_softmax(logits, 1).gather(1, labels).neg().mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::param::LinearLikeInner;
    use crate::{DormouseConfig, DormouseModel};
    use burn::backend::autodiff::Autodiff;
    use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
    use burn::module::{Module, ModuleVisitor, Param};
    use burn::tensor::{Device, TensorData};

    /// The alias the train crate uses for `--features cpu`.
    type B = Autodiff<burn::backend::Flex, BalancedCheckpointing>;

    /// A tiny fixture with NOTHING else on: no KDA, no Engram, no TSCT, so the
    /// only thing the gates below can be measuring is the future-byte path.
    /// `t = 12, k = 2` gives `n = 9` valid positions and three labels the head
    /// must NOT read (`targets[0]`, `targets[1]`, `targets[11]`).
    fn mini(t: usize) -> DormouseConfig {
        DormouseConfig {
            d_model: 8,
            n_heads: 2,
            head_dim: 4,
            d_ffn: 16,
            rank: 4,
            max_iter: 1,
            n_experts: 1,
            max_seq_len: t,
            use_kda: false,
            use_tsct: false,
            use_engram: false,
            jepa_weight: 0.0,
            dspark_weight: 0.0,
            aux_fb_weight: 0.1,
            aux_fb_horizon: 2,
            ..DormouseConfig::default()
        }
    }

    /// Knuth MMIX LCG, the same generator the sibling aux tests use: no RNG
    /// dependency, so a failure is reproducible and a golden is pinnable.
    fn bytes(seed: u64, n: usize) -> Vec<i64> {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) % 256) as i64
            })
            .collect()
    }

    /// A hidden state and a label tensor on the `vocab = 16` sub-vocabulary the
    /// gates mutate, so `(v + 37) % 16` is a different byte and not an
    /// out-of-range gather.
    fn hidden(t: usize, d: usize, dev: &Device) -> Tensor<3> {
        let v: Vec<f32> = bytes(0xB7, t * d).iter().map(|&x| x as f32 * 0.01 - 1.0).collect();
        Tensor::from_data(TensorData::new(v, [1, t, d]), dev)
    }

    /// The trainer's labelling: `y[q] = x[q+1]`, last entry a wraparound.
    fn labels(x: &[i64]) -> Vec<i64> {
        x[1..].iter().chain(std::iter::once(&x[0])).copied().collect()
    }

    fn dense(head: &LinearLike) -> &burn::nn::Linear {
        match &head.inner {
            LinearLikeInner::Dense(l) => l,
            LinearLikeInner::Tsct(_) => panic!("the future-byte head is a dense linear, not TSCT"),
        }
    }

    /// #1 - THE VACCINE. Exactly which entries of `targets` the term reads,
    /// one entry at a time: `k ..= t-2` must move the loss, everything else
    /// must leave it BITWISE equal. Enumerated rather than sampled, because a
    /// sampled version cannot tell "the gate passes" from "the gate did not
    /// look".
    ///
    /// The DSpark window bug (2026-09-29) is what this is for: the draft head
    /// was handed `targets` where it needed the consumed sequence, so its step
    /// `s` was fed the byte `logits[p+s]` had just predicted and supervised
    /// toward the one after, and EVERY DSpark number this project ever recorded
    /// measured that objective. The invariance direction is the assertion that
    /// would have caught it: a term fed the labels it is asked to predict
    /// moves when the labels move.
    #[test]
    fn future_byte_loss_uses_only_shifted_targets() {
        let dev = Device::flex().autodiff();
        let (t, k, d) = (12usize, 2usize, 8usize);
        let cfg = mini(t);
        let head = DormouseModel::new(&cfg, &dev).aux.fb.expect("weight > 0 builds the head");
        let h = hidden(t, d, &dev);
        let y0 = labels(&bytes(0x2C, t));
        let run = |y: Vec<i64>| -> f32 {
            future_byte_loss::<B>(&head, h.clone(), Tensor::from_data(TensorData::new(y, [1, t]), &dev), k)
                .into_scalar::<f32>()
        };
        let base = run(y0.clone());
        assert!(base.is_finite() && base > 0.0, "the term must be a real loss, got {base}");
        let mut read: Vec<(usize, f32)> = Vec::new();
        let mut skipped: Vec<(usize, f32)> = Vec::new();
        for i in 0..t {
            let mut y = y0.clone();
            y[i] = (y[i] + 37) % 16; // the 16-symbol sub-vocabulary `hidden`/`labels` use
            let got = run(y);
            let moved = (got - base).abs();
            if i >= k && i <= t - 2 {
                read.push((i, moved));
                assert!(moved > 1e-6, "targets[{i}] IS a label at k={k} (read at q={i}) but the loss ignored it");
            } else {
                skipped.push((i, moved));
                assert_eq!(got.to_bits(), base.to_bits(), "targets[{i}] must not be read at all");
            }
        }
        println!(
            "labels targets[k..=t-2] moved the loss by {:.3e}..{:.3e}; the {} entries outside that \
             range moved it by at most {:.3e} (bitwise 0)",
            read.iter().map(|x| f64::from(x.1)).fold(f64::INFINITY, f64::min),
            read.iter().map(|x| f64::from(x.1)).fold(0.0, f64::max),
            skipped.len(),
            skipped.iter().map(|x| f64::from(x.1)).fold(0.0, f64::max),
        );
        // The map itself, not just the two halves: exactly n = t - k - 1 labels
        // are read, and the entries outside are `targets[0..k]` (a label only at
        // k = 0) plus the wraparound at `targets[t-1]`.
        assert_eq!(read.len(), valid_positions(t, k), "the number of labels read is t - k - 1");
        assert_eq!(skipped, vec![(0, 0.0), (1, 0.0), (t - 1, 0.0)], "the skipped set is the documented one");
    }

    /// The same property at the MODEL level, plus the two things only the model
    /// can be wrong about: that the term reaches the returned `aux` at all,
    /// and that `aux_fb_weight` is exactly the multiplier applied.
    ///
    /// The label side is a single end-to-end assertion here (the exhaustive map
    /// is the loss-level gate above): mutate a label the head must not read and
    /// the returned aux is bitwise unchanged. The other three are about the
    /// inputs the term is allowed to see - the consumed sequence moves it (it
    /// reads the hidden state, so it is not a constant), a future label moves
    /// it, and the EMA teacher does NOT (there is no teacher in this arm, and
    /// an arm that accidentally read one would be measuring JEPA).
    #[test]
    fn future_byte_loss_is_a_function_of_input_and_logits_only() {
        let dev = Device::flex().autodiff();
        let (t, k) = (12usize, 2usize);
        let cfg = mini(t);
        let model = DormouseModel::new(&cfg, &dev);
        let teacher = crate::aux::ema_update(model.clone(), &model, 0.0);
        let x = bytes(0x51, t);
        let y = labels(&x);
        let run = |m: &DormouseModel, ids: &[i64], labels: &[i64], teacher: Option<&DormouseModel>| -> f32 {
            m.forward_with_hidden::<B>(
                Tensor::from_data(TensorData::new(ids.to_vec(), [1, t]), &dev),
                None,
                None,
                Some(Tensor::from_data(TensorData::new(labels.to_vec(), [1, t]), &dev)),
                teacher,
            )
            .3
            .expect("fb_weight > 0 with labels must return an aux term")
            .into_scalar::<f32>()
        };
        let a = run(&model, &x, &y, None);
        assert!(a.is_finite() && a > 0.0, "the term is vacuous at {a}, so every check below is free");
        // (1) The multiplier is the configured weight, EXACTLY. No RNG in this
        // term (no mask, no sampling), so unlike the JEPA gate this one is a
        // bitwise observation, not an inequality.
        let mut unit = model.clone();
        unit.aux_fb_weight = 1.0;
        let one = run(&unit, &x, &y, None);
        assert!(
            (a - cfg.aux_fb_weight * one).abs() < 1e-6 * one.abs().max(1.0),
            "aux_fb_weight {} is not the multiplier applied ({a} vs {} x {one})",
            cfg.aux_fb_weight,
            cfg.aux_fb_weight * one
        );
        // (2) A label the head must NOT read leaves the returned aux bitwise
        // equal - `targets[0]`, and the wraparound `targets[t-1]`.
        for i in [0usize, t - 1] {
            let mut other = y.clone();
            other[i] = (other[i] + 91) % 16;
            let got = run(&model, &x, &other, None);
            assert_eq!(
                got.to_bits(),
                a.to_bits(),
                "targets[{i}] is not a label at k={k} but it moved the returned aux: {a} -> {got}"
            );
        }
        // (3) A label the head IS asked for moves it, or (1) and (2) are both
        // satisfied by a constant.
        let mut fut = y.clone();
        fut[k] = (fut[k] + 91) % 16;
        assert!(
            (run(&model, &x, &fut, None) - a).abs() > 1e-6,
            "the term ignores the future byte it exists to predict"
        );
        // (4) The consumed sequence moves it: the term reads the hidden state,
        // so it is a function of the model's prediction and not a constant.
        let x2: Vec<i64> = x.iter().map(|&v| (v + 3) % 16).collect();
        assert!(
            (run(&model, &x2, &y, None) - a).abs() > 1e-6,
            "the term ignores the consumed sequence: it is not reading the hidden state"
        );
        // (5) The EMA teacher does not appear in it. This arm has no teacher;
        // a term that read one would be JEPA wearing a different name.
        assert_eq!(
            run(&model, &x, &y, Some(&teacher)).to_bits(),
            a.to_bits(),
            "the future-byte term read the EMA teacher"
        );
    }

    /// #2 - RANGE HANDLING against an f64 HOST reference, on a fixture small
    /// enough to compute by hand.
    ///
    /// The host computes the same quantity in f64 from the head's OWN weights,
    /// read back off the device: `n = t - k - 1` positions, each an f64
    /// log-softmax of an f64 matmul, averaged. The tolerance is 1e-5 on a loss
    /// of order 2, i.e. 5e-6 RELATIVE: f32 over `d = 8` accumulations plus a
    /// log-sum-exp is ~1e-6 here, so the bound has about a decade of margin and
    /// is not fitted to the observed error. (f64 is the reference, not f32: the
    /// point is that the DEVICE is not being compared against itself.)
    ///
    /// The length boundary is the same test: `t = k + 1` has no valid position
    /// and the term is an exact zero that bumps `FUTURE_BYTE_ASKED` and not
    /// `FUTURE_BYTE`, so `fb=<ran>/<asked>` shows `0/1` rather than a run that
    /// looks healthy and trained nothing.
    #[test]
    fn future_byte_loss_matches_an_f64_host_reference_at_the_length_boundary() {
        let dev = Device::flex().autodiff();
        let (b, t, d, v) = (2usize, 9usize, 8usize, 256usize);
        let cfg = DormouseConfig { d_model: d, vocab: v, ..mini(t) };
        let model = DormouseModel::new(&cfg, &dev);
        let head = model.aux.fb.expect("cfg.aux_fb_weight > 0 must build the head");
        // Two batches, so the `[b, n]` reshape and the per-row mean are actually
        // exercised (b = 1 would hide a transposition in either).
        let hv: Vec<f32> = bytes(0x6B, b * t * d).iter().map(|&x| x as f32 * 0.01 - 1.0).collect();
        let h = Tensor::from_data(TensorData::new(hv.clone(), [b, t, d]), &dev);
        let x = bytes(0x71, b * t);
        let mut y: Vec<i64> = Vec::with_capacity(b * t);
        for r in 0..b {
            y.extend(labels(&x[r * t..(r + 1) * t]));
        }

        // f64 reference, from the head's OWN parameters read back off the device.
        let lin = dense(&head);
        let w: Vec<f32> = lin.weight.val().clone().into_data().try_to_vec().expect("readable weight");
        let bias: Vec<f32> =
            lin.bias.as_ref().expect("the dense head has a bias").val().clone().into_data().try_to_vec().expect("readable bias");
        // burn's `nn::Linear` default layout is `Row` with the weight stored
        // `[d_input, d_output]` and applied as `x @ W`, so element `j * v + c`
        // multiplies `h[j]` into logit `c`. Pinned here because getting it
        // backwards is a SILENT 6.5e-3 disagreement in this very test - the
        // numbers look like noise until you find the transposition, and the
        // first draft of this gate had exactly that bug.
        assert_eq!(lin.weight.dims(), [d, v], "the weight is [d_input, d_output]");
        assert_eq!(bias.len(), v, "one bias per output class");
        let hf: Vec<f64> = hv.iter().map(|&x| f64::from(x)).collect();
        let n = valid_positions(t, cfg.aux_fb_horizon);
        let k = cfg.aux_fb_horizon;
        let mut acc = 0.0f64;
        for r in 0..b {
            for q in 0..n {
                let label = y[r * t + q + k] as usize;
                let z: Vec<f64> = (0..v)
                    .map(|c| {
                        (0..d).map(|j| f64::from(w[j * v + c]) * hf[(r * t + q) * d + j]).sum::<f64>()
                            + f64::from(bias[c])
                    })
                    .collect();
                let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let lse = m + z.iter().map(|&e| (e - m).exp()).sum::<f64>().ln();
                acc += -(z[label] - lse);
            }
        }
        let want = acc / (b * n) as f64;
        let got = future_byte_loss::<B>(&head, h, Tensor::from_data(TensorData::new(y, [b, t]), &dev), k)
            .into_scalar::<f32>();
        let rel = ((f64::from(got) - want) / want).abs();
        assert!(rel < 1e-5, "device {got:.8} vs the f64 host reference {want:.8}: rel {rel:.2e}");
        println!("future_byte_loss vs f64 host reference: {got:.8} vs {want:.8}, rel {rel:.2e} (n = {n})");

        // The length boundary: a sequence of exactly `k` bytes has no valid
        // position, and the term is then an exact zero that bumps NO
        // `FUTURE_BYTE` - so the eval line's `fb=<ran>/<asked>` reads `0/1` and
        // a 2k-step A/B at `seq_len <= k` is visibly a 2k-step A/B of nothing.
        let short = k;
        let ran_before = crate::probe::count(crate::probe::FUTURE_BYTE);
        assert!(ran_before > 0, "the t = 9 term above must have counted itself");
        let xs = bytes(0x11, b * short);
        let ys: Vec<i64> = (0..b).flat_map(|r| labels(&xs[r * short..(r + 1) * short])).collect();
        let zero = future_byte_loss::<B>(
            &head,
            Tensor::<3>::zeros([b, short, d], &dev),
            Tensor::from_data(TensorData::new(ys, [b, short]), &dev),
            k,
        )
        .into_scalar::<f32>();
        assert_eq!(zero.to_bits(), 0.0f32.to_bits(), "t = k must return an exact zero, not a small term");
        assert_eq!(
            crate::probe::count(crate::probe::FUTURE_BYTE),
            ran_before,
            "a zero-length term must not bump `ran`; `fb=<ran>/<asked>` is how that is visible"
        );
    }

    /// ZERO-WEIGHT IDENTITY, in the two forms that can actually be checked.
    ///
    /// (1) The head does not EXIST at weight 0 (`AuxHeads::fb` is `None`), so
    /// the parameter set and the checkpoint are today's, byte for byte - which
    /// is what makes queue row 1 (pure CE) a valid control without a
    /// re-baseline, and what keeps every measured preset parameter count
    /// (`tests/preset_exec.rs`) true.
    /// (2) With the arm on, the MAIN loss is bitwise unchanged by it: the
    /// future-byte term is added to the total and the loop's own CE `rec` is
    /// not touched. This is the claim "the A/B arm is the arm plus a term",
    /// and a head that leaked into the loop's CE would move `rec` here.
    #[test]
    fn zero_weight_adds_no_parameters_and_the_main_loss_is_untouched() {
        let dev = Device::flex().autodiff();
        let t = 12usize;
        let on = DormouseModel::new(&mini(t), &dev);
        // The default is the off position, and it is the DEFAULT that must be
        // off: a recipe nobody edited must not carry an untrained head.
        assert_eq!(DormouseConfig::default().aux_fb_weight, 0.0, "the default must be OFF");
        assert_eq!(DormouseConfig::default().aux_fb_horizon, 2, "the default horizon is 2");
        let cfg_off = DormouseConfig { aux_fb_weight: 0.0, ..mini(t) };
        let plain = DormouseModel::new(&cfg_off, &dev);
        assert!(plain.aux.fb.is_none(), "weight 0 must not build a head: it would be dead parameters");
        assert!(on.aux.fb.is_some(), "weight > 0 must build it");
        // The head is the WHOLE parameter difference between the two models -
        // nothing else moved, which is the byte-identical claim in its
        // strongest form this crate can check.
        let head_params = cfg_off.vocab * cfg_off.d_model + cfg_off.vocab;
        assert_eq!(count_params(&plain) + head_params, count_params(&on), "the head is the whole difference");
        // ... and the 197 120 the schema comment quotes is that arithmetic at
        // `small`'s widths, not a number from a different place.
        assert_eq!(FB_PARAMS_AT_SMALL, 256 * 768 + 256, "the quoted price of the arm on `small`");

        // (2) the main CE, bitwise, with the arm on and then off - on the SAME
        // model, so the parameter draw is identical and any difference is the
        // arm's.
        let model = on;
        let x = bytes(0x3F, t);
        let y = labels(&x);
        let step = |m: &DormouseModel| -> (f32, Option<f32>) {
            let (_, r, _, a) = m
                .forward_with_hidden::<B>(
                    Tensor::from_data(TensorData::new(x.clone(), [1, t]), &dev),
                    None,
                    None,
                    Some(Tensor::from_data(TensorData::new(y.clone(), [1, t]), &dev)),
                    None,
                );
            (r.into_scalar::<f32>(), a.map(|a| a.into_scalar::<f32>()))
        };
        crate::probe::reset();
        let (ce_on, aux_on) = step(&model);
        assert!(aux_on.is_some_and(|a| a > 0.0), "weight > 0 must return a real term");
        assert_eq!(crate::probe::count(crate::probe::FUTURE_BYTE), 1, "the term ran once");
        assert_eq!(crate::probe::count(crate::probe::FUTURE_BYTE_ASKED), 1, "and it was asked for once");
        let mut muted = model.clone();
        muted.aux_fb_weight = 0.0;
        let (ce_off, aux_off) = step(&muted);
        assert_eq!(ce_on.to_bits(), ce_off.to_bits(), "the arm moved the MAIN CE: {ce_on} vs {ce_off}");
        assert_eq!(aux_off, None, "weight 0 on a labelled forward must return no aux at all");
        // A muted arm is not a silently-running one: the counters do not move.
        assert_eq!(crate::probe::count(crate::probe::FUTURE_BYTE), 1, "the muted forward must not bump `ran`");
        assert_eq!(crate::probe::count(crate::probe::FUTURE_BYTE_ASKED), 1, "nor `asked`");
    }

    /// The claim in the module docs, measured through the LIVE model forward:
    /// backward the auxiliary total alone, then read every parameter's gradient
    /// by path. The head trains, the BACKBONE is shaped (that is the arm's
    /// entire point), and `lm_head` is never on the graph - "untied" and "the
    /// gradient path must not touch the main lm_head" are the same sentence.
    ///
    /// `None` is the strong half: not a zero gradient but no entry at all,
    /// because the main head's output (`logits`) is simply not in the aux
    /// subgraph. A test that accepted `max |grad| == 0` would also pass if the
    /// loss had been multiplied by 0.
    #[test]
    fn the_gradient_reaches_the_backbone_and_never_the_lm_head() {
        let dev = Device::flex().autodiff();
        let t = 12usize;
        let cfg = mini(t);
        let model = DormouseModel::new(&cfg, &dev);
        let x = bytes(0x4D, t);
        let y = labels(&x);
        let aux = model
            .forward_with_hidden::<B>(
                Tensor::from_data(TensorData::new(x, [1, t]), &dev),
                None,
                None,
                Some(Tensor::from_data(TensorData::new(y, [1, t]), &dev)),
                None,
            )
            .3
            .expect("fb only");
        let grads = aux.backward();
        struct Rows<'a> {
            stack: Vec<String>,
            grads: &'a burn::tensor::Gradients,
            out: Vec<(String, Option<f32>)>,
        }
        impl ModuleVisitor for Rows<'_> {
            fn enter_module(&mut self, name: &str, _c: &str) {
                self.stack.push(name.to_string());
            }
            fn exit_module(&mut self, _n: &str, _c: &str) {
                self.stack.pop();
            }
            fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
                let n = p.grad(self.grads).map(|g| g.abs().max().into_scalar::<f32>());
                self.out.push((self.stack.join("."), n));
            }
        }
        let mut rows = Rows { stack: Vec::new(), grads: &grads, out: Vec::new() };
        model.visit(&mut rows);
        let find = |prefix: &str| -> Vec<(&str, Option<f32>)> {
            rows.out.iter().filter(|(p, _)| p.starts_with(prefix)).map(|(p, n)| (p.as_str(), *n)).collect()
        };
        let head_rows = find("aux.fb");
        assert!(!head_rows.is_empty(), "the head must own parameters under aux.fb: {:?}", rows.out);
        assert!(
            head_rows.iter().any(|(_, n)| n.is_some_and(|v| v > 0.0)),
            "the future-byte head receives no gradient: {head_rows:?}"
        );
        // `None`, not zero: the main head is not in the subgraph at all.
        let lm = find("lm_head");
        assert!(!lm.is_empty(), "the fixture must have a main head to be wrong about");
        assert!(lm.iter().all(|(_, n)| n.is_none()), "the future-byte term reached lm_head: {lm:?}");
        // The backbone. `residual_scale` is 0 at init (ReZero), so the block
        // body sits on the graph with a zero gradient; what must be non-zero is
        // at least one parameter of the block or of the readout above it.
        let block = find("loop_block");
        assert!(
            block.iter().any(|(_, n)| n.is_some_and(|v| v > 0.0)),
            "the term did not reach the loop block at all: {block:?}"
        );
        println!(
            "future-byte grads: head {}/{} with one, loop_block {}/{} with a non-zero one, \
             lm_head {}/{} with one",
            head_rows.iter().filter(|(_, n)| n.is_some()).count(),
            head_rows.len(),
            block.iter().filter(|(_, n)| n.is_some_and(|v| v > 0.0)).count(),
            block.len(),
            lm.iter().filter(|(_, n)| n.is_some()).count(),
            lm.len(),
        );
    }

    /// The one checkpoint direction this lane could get wrong, checked because
    /// it decides whether the A/B can RESUME: an off-arm record loaded into a
    /// model built with `aux_fb_weight > 0`.
    ///
    /// `AuxHeads::fb` is an `Option`, and an `Option` in a burn record is
    /// exactly the shape that can come back as a fresh random head instead of
    /// an error. A silently re-initialized head is the `8fa5d4c` failure
    /// class one level down: the run trains a parameter the checkpoint never
    /// carried and reports no problem, and the first 2 000 steps of the arm are
    /// measuring that. The other direction (an on-arm record into an on-arm
    /// model) is the ordinary case and is covered by the repo's
    /// `tests/ckpt_roundtrip.rs`.
    #[test]
    fn an_off_arm_record_refuses_to_load_into_an_on_arm_model() {
        let dev = Device::flex();
        let t = 12usize;
        let off = DormouseModel::new(&DormouseConfig { aux_fb_weight: 0.0, ..mini(t) }, &dev);
        let record = off.into_record();
        let on = DormouseModel::new(&mini(t), &dev);
        match on.try_load_record(record) {
            Err(e) => {
                let msg = e.to_string();
                println!("refused loudly, as it must: {}", &msg[..msg.len().min(200)])
            }
            // A silent fill-in is acceptable ONLY if the loaded head is visibly
            // a new one, which the caller cannot see - so this branch is a
            // recorded defect, not a pass.
            Ok(loaded) => panic!(
                "an off-arm record loaded into an on-arm model (head built: {}) and the \
                 future-byte head was silently created. Either refuse the load (loud) or the \
                 loader has to report a re-initialized parameter, because a resume would then \
                 train a head the checkpoint never carried. This is the `8fa5d4c` failure class: \
                 correct numbers, wrong network.",
                loaded.aux.fb.is_some()
            ),
        }
    }

    fn count_params(m: &DormouseModel) -> usize {
        struct N(usize);
        impl ModuleVisitor for N {
            fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
                self.0 += p.val().dims().iter().product::<usize>();
            }
        }
        let mut n = N(0);
        m.visit(&mut n);
        n.0
    }
}
