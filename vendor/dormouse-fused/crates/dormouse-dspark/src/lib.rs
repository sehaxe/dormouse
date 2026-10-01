//! # dormouse-dspark - DSpark Speculative Decoding for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! Building blocks from [DSpark](https://arxiv.org/abs/2607.05147) (DeepSeek AI, 2026).
//!
//! `dspark_loss` and `accept_rate_loss` are **matched against the official
//! [DeepSpec](https://github.com/deepseek-ai/DeepSpec) implementation**, which
//! is the strongest tier this crate has: `tests/dspark_loss_oracle.rs` compares
//! against DeepSeek's `compute_dspark_loss` executed on CPU at commit
//! `005e03b81cec38b7da6399833d609ee89a2587f2`, with the OFFICIAL alphas from
//! DeepSpec's own config — not transcribed, run. It is that gate which found
//! and fixed three defects in the confidence term (a wrong numerical space, a
//! target that was not detached, and a kink in the BCE spelling), so the older
//! claim that these tests "cannot detect a wrong loss term" is no longer true
//! of them: it is exactly what they now do. The rest of the crate below is
//! still hand-derived from the paper and has no reference of any kind.
//!
//! - `VanillaMarkov` — low-rank first-order transition bias `B = W1[x]W2` (Eq 5)
//! - `GatedMarkovHead` — gated variant: `gate * W1[x]` with `sigmoid(W_g[h; W1[x]])`
//! - `RNNHead` — GRU-like recurrent head with joint `[s; W1[x]; h]` projection (Eq 6)
//! - `AcceptRatePredictor` — confidence head `sigmoid(w^T[h_k; W1[x_{k-1}]])` (Eq 7)
//! - `sample_tokens` — temperature sampling / argmax, `sample_residual`
//! - `dspark_loss` — CE + TV (L1) + confidence BCE, position-weighted `w_k = exp(-k/gamma)`
//! - `sts_calibrate` — Sequential Temperature Scaling (paper 3.2.1)
//!
//! Loss formula (paper Eq 9-12, DeepSpec `loss.py`):
//! ```text
//! w_k    = exp(-k / gamma)                   (k = 0-based position)
//! L_ce   = sum_k w_k * CE(p_k^d, x_k*)
//! L_tv   = sum_k w_k * ||p_k^d - p_k^t||_1   (softmax probs)
//! L_conf = sum_k w_k * BCE(c_k, 1 - 0.5*||p_d - p_t||_1)
//! L      = 0.1*L_ce + 0.9*L_tv + 1.0*L_conf
//! ```
//!
//! ```toml
//! dormouse-dspark = "0.1"                    # inference
//! dormouse-dspark = { version = "0.1", features = ["training"] }  # + loss fns
//! ```
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
#[cfg(feature = "training")]
use burn::tensor::Int;
use burn::tensor::{activation, Device, Tensor};

pub mod markov;
pub mod sampling;

pub use markov::{greedy_draft, GatedMarkovHead, RNNHead, VanillaMarkov};
pub use sampling::{sample_residual, sample_tokens};

// ─── Confidence predictor ─────────────────────────────────────────────

/// Per-position acceptance probability estimator (paper Eq 7).
///
/// ```text
/// [h_k; W1[x_{k-1}]] -> Linear(1) -> sigmoid -> P(accept) in [0,1]
/// ```
/// The Markov embedding of the previous token is concatenated to the
/// backbone hidden state; the projection input is then
/// `input_dim + markov_rank`. Construct with [`Self::with_markov`] to size
/// the layer for that; `new` builds the plain hidden-state-only variant.
#[derive(Module, Debug)]
pub struct AcceptRatePredictor {
    proj: Linear,
    #[module(skip)]
    with_markov: bool,
    /// Markov embedding rank the input was sized for (0 = hidden-only).
    #[module(skip)]
    markov_rank: usize,
}

impl AcceptRatePredictor {
    fn build(input_dim: usize, device: &Device) -> Linear {
        LinearConfig::new(input_dim, 1)
            .with_bias(false)
            .init(device)
    }

    /// Hidden-state-only predictor: `h_k -> logit`.
    pub fn new(input_dim: usize, device: &Device) -> Self {
        Self {
            proj: Self::build(input_dim, device),
            with_markov: false,
            markov_rank: 0,
        }
    }

    /// Markov-conditioned variant: input is `[h_k; W1[x_{k-1}]]` of width
    /// `input_dim + markov_rank` (DeepSpec: `input_dim + markov_rank`).
    pub fn with_markov(input_dim: usize, markov_rank: usize, device: &Device) -> Self {
        assert!(
            markov_rank > 0,
            "markov_rank must be > 0 for the conditioned variant"
        );
        Self {
            proj: Self::build(input_dim + markov_rank, device),
            with_markov: true,
            markov_rank,
        }
    }

    /// Logit `[B, L, 1]` for the given hidden states and (optionally) the
    /// Markov embeddings of the previous tokens.
    ///
    /// Panics when `prev_embeddings` is passed to a predictor built without
    /// it (or vice versa) — a silent shape mismatch would hide a wiring bug.
    pub fn logit(&self, hidden_states: Tensor<3>, prev_embeddings: Option<Tensor<3>>) -> Tensor<3> {
        let features = match prev_embeddings {
            Some(pe) => {
                assert!(
                    self.with_markov,
                    "predictor was built hidden-only; rebuild with AcceptRatePredictor::with_markov"
                );
                Tensor::cat(vec![hidden_states, pe], 2)
            }
            None => {
                assert!(
                    !self.with_markov,
                    "predictor expects Markov embeddings; pass prev_embeddings"
                );
                hidden_states
            }
        };
        let [_, _, in_w] = features.dims();
        // burn 0.22 stores Linear weight as [in_features, out_features]
        let expected = self.proj.weight.dims()[0];
        assert_eq!(
            in_w, expected,
            "feature width {in_w} does not match the projection input {expected}"
        );
        self.proj.forward(features)
    }

    /// Probability `[B, L, 1]`.
    pub fn prob(&self, hidden_states: Tensor<3>, prev_embeddings: Option<Tensor<3>>) -> Tensor<3> {
        activation::sigmoid(self.logit(hidden_states, prev_embeddings))
    }
}

// ─── Training losses (behind "training" feature) ─────────────────────

/// Position decay weights `w_k = exp(-k/gamma)` (DeepSpec `loss.py`),
/// k = 0-based position within the draft block.
pub fn position_weights(block_size: usize, gamma: f64, device: &Device) -> Tensor<1> {
    let vals: Vec<f32> = (0..block_size)
        .map(|k| (-(k as f64) / gamma).exp() as f32)
        .collect();
    Tensor::from_data(burn::tensor::TensorData::new(vals, [block_size]), device)
}

/// Per-position analytical acceptance rate (paper Eq 8, DeepSpec
/// `_compute_accept_rate_3d`):
///
/// ```text
/// c_k* = clamp(1 - 0.5 * ||p_draft - p_target||_1, 0, 1)
/// ```
/// `draft_logits` / `target_logits`: `[B, L, vocab]`, returns `[B, L]`.
pub fn accept_rate_target(draft_logits: Tensor<3>, target_logits: Tensor<3>) -> Tensor<2> {
    let pd = activation::softmax(draft_logits, 2);
    let pt = activation::softmax(target_logits, 2);
    let tv = (pd - pt).abs().sum_dim(2).squeeze_dim::<2>(2); // [B, L]
    tv.mul_scalar(0.5).neg().add_scalar(1.0).clamp(0.0, 1.0)
}

/// DSpark training objective (paper Eq 9-12; DeepSpec `compute_dspark_loss`).
///
/// `draft_logits` `[B, L, vocab]` — drafter logits after the sequential head.
/// `target_logits` `[B, L, vocab]` — aligned frozen target logits.
/// `target_ids` `[B, L]` — ground-truth next-token ids.
/// `confidence_logits` `[B, L, 1]` — confidence head logits (Eq 7).
/// `mask` `[B, L]` — 1.0 = supervised, 0.0 = padding.
/// `gamma` — position decay (DeepSpec default 4.0).
///
/// Returns `(total, ce, tv, conf)` so the caller can log the components.
#[cfg(feature = "training")]
#[allow(clippy::too_many_arguments)]
/// `confidence_logits` is `Option` because DeepSeek's own
/// `compute_dspark_loss` takes `confidence_pred=None` and **skips the term
/// entirely** in that case. Our previous signature took a mandatory
/// `Tensor<3>`, so "no head" could only be spelled as a zero tensor — which the
/// logit form correctly scores as p = 0.5 and charges 0.693 for. Measured
/// against the official loss: ours 2.063 against DeepSeek's 1.370, rel 3.4e-1
/// (`tests/dspark_loss_oracle.rs`, case `no_confidence_head`).
///
/// With no head the term is 0 and the others are unchanged, which is the
/// reference's behaviour and the only way to express it honestly.
pub fn dspark_loss(
    draft_logits: Tensor<3>,
    target_logits: Tensor<3>,
    target_ids: Tensor<2, Int>,
    confidence_logits: Option<Tensor<3>>,
    mask: Tensor<2>,
    gamma: f64,
) -> (Tensor<1>, Tensor<1>, Tensor<1>, Tensor<1>) {
    let [b, l, v] = draft_logits.dims();
    let device = draft_logits.device();
    let w = position_weights(l, gamma, &device)
        .unsqueeze_dim::<2>(0)
        .expand([b, l]);
    let wm = w.clone() * mask.clone();
    let den = wm.clone().sum().clamp_min(1.0);

    // CE (Eq 9): -sum_k w_k log p_k(x_k*)
    let flat_logits = draft_logits.clone().reshape([b * l, v]);
    let flat_ids = target_ids.reshape([b * l]);
    let lp = burn::tensor::activation::log_softmax(flat_logits, 1);
    let ce_per = -lp.gather(1, flat_ids.unsqueeze_dim::<2>(1)).reshape([b, l]);
    let ce = (ce_per * wm.clone()).sum().div(den.clone());

    // TV (Eq 10): sum_k w_k ||p_d - p_t||_1
    let pd = activation::softmax(draft_logits.clone(), 2);
    let pt = activation::softmax(target_logits.clone(), 2);
    let tv_per = (pd - pt).abs().sum_dim(2).squeeze_dim::<2>(2); // [B, L]
    let tv = (tv_per * wm.clone()).sum().div(den.clone());

    // Confidence BCE (Eq 11): target = analytical acceptance rate (Eq 8)
    //
    // LOGIT FORM, and this is a fix rather than a style choice.
    // DeepSeek's `loss.py` calls `F.binary_cross_entropy_with_logits(conf,
    // c_star)`. The old code did `sigmoid(x)` and then `log(p + 1e-7)`, which
    // puts a hard CEILING of ln(1e7) = 16.1 on the term no matter how wrong the
    // head is. Measured against the official loss at `DeepSpec@005e03b8`
    // (`tests/dspark_loss_oracle.rs`, run on CPU): the `saturated_conf` case
    // gave ours 9.309 against DeepSeek's 21.445. The CE and L1/TV terms agreed
    // on all eight cases, so this one term was the whole disagreement. At
    // |x|=40 the official form returns 40.00; the clamped one could not exceed
    // 16.1 at any x.
    //
    // AND THE TARGET IS DETACHED, which is the same reference and a second
    // fix. `loss.py:145` is `confidence_targets = accept_rate_3d.detach()`: the
    // analytical acceptance rate is a LABEL derived from the drafter's own
    // output, not a second objective for it, so upstream back-propagates it
    // nowhere. We built `c_star` from `draft_logits` on a live graph, which
    // pulls `-cl * d(c_star)/d(draft_logits)` into the drafter's gradient -
    // a gradient path that exists only to move a loss DOWN by making the
    // drafter's own output look more like the target, and which has no
    // counterpart in the paper's Eq. 11 or in the reference implementation.
    //
    // The FORWARD VALUE IS BIT-IDENTICAL either way, which is why the
    // value-level oracle above was blind to it. It is a gradient-only defect
    // and it is not small. Measured against the reference's own autograd on the
    // fixture's eight inputs, the L2 norm of d(total)/d(draft_logits):
    //
    //   case              |g| no-detach   |g| detached    ratio
    //   saturated_conf       1.227131       0.073771      16.6x too LARGE
    //   block7_exact         0.118244       0.087681       1.35x
    //   tiny_mask            0.131755       0.153709       1.17x
    //   main                 0.056004       0.054848       1.02x
    //   aligned_identical    0.034983       0.034983       1.00x  (accept = 1
    //   big_logits           0.048648       0.048648       1.00x   clamps to a
    //   all_masked_off        0.0            0.0          -         constant)
    //   no_confidence_head   0.057351       0.057351       1.00x  (no head)
    //
    // Four of eight cases move, and `confidence_head_alpha` is the LARGEST of
    // the three weights (1.0, against 0.1 and 0.9). The gate is
    // `dspark_loss_gradients_agree_with_the_official_loss`, the only comparison
    // in the crate that can see it: with the `.detach()` removed it reports
    // 3270x to 279147x the tolerance on those same four cases, and every value
    // test beside it stays green.
    //
    // AND THE SPELLING IS `(1-c)*x + softplus(-x)`, which is a THIRD fix and
    // the only one the value oracle could never have found. The intermediate
    // version wrote BCE the way it is usually written out,
    // `max(x,0) - x*c + log(1 + exp(-|x|))`, which is correct as a VALUE and
    // correct as a gradient everywhere except the two kinks it introduces
    // itself. At `x == 0` exactly, burn's `relu_backward` zeroes the `relu`
    // term (its mask is `output <= 0`, `burn-backend/src/backend/ops/
    // activation.rs:55`) and `abs`'s subgradient at 0 is 0, so the derivative
    // collapses to `-c` where the true derivative - and
    // `binary_cross_entropy_with_logits`' - is `sigmoid(x) - c`. Measured on
    // the fixture's `aligned_identical` case, which has `confidence_pred` all
    // zeros: d/d(conf[0]) came out at -0.1723984 against DeepSeek's
    // -0.08619919, exactly 2x, on 10 of 14 coordinates. It is a measure-zero
    // set in training and a hard disagreement on a fixture row, and the
    // gradient gate found it on the first run - which is the argument for
    // having the gate.
    //
    // `(1-c)*x + softplus(-x)` is the algebra `binary_cross_entropy_with_logits`
    // itself is written in (`logsigmoid` rearranged), so it cannot have a
    // kink: it is smooth, its derivative is `(1-c) - sigmoid(-x) = sigmoid(x)
    // - c` identically, and `softplus` is already the stable primitive (it
    // switches to the identity above its threshold, so |x| = 40 gives 4.5e-18
    // for a right sign and 40.00 for a wrong one). Three terms become two.
    let conf = match confidence_logits {
        None => Tensor::zeros([1], &draft_logits.device()),
        Some(cl) => {
            // `detach()`: see the block comment above. Every other tensor in
            // this function stays attached; this one is a label.
            let c_star = accept_rate_target(draft_logits, target_logits).detach(); // [B, L]
            let cl = cl.reshape([b, l]);
            let one_minus_c = c_star.neg().add_scalar(1.0);
            let bce_per = cl.clone().mul(one_minus_c).add(activation::softplus(cl.neg(), 1.0));
            (bce_per * wm).sum().div(den)
        }
    };

    // Eq 12: 0.1*L_ce + 0.9*L_tv + 1.0*L_conf
    let total = ce.clone().mul_scalar(0.1) + tv.clone().mul_scalar(0.9) + conf.clone();
    (total, ce, tv, conf)
}

/// Calibration MSE for the confidence head.
///
/// `probs`: `[B, L]` - predicted P(accept)
/// `target`: `[B, L]` - analytical acceptance rate (Eq 8)
/// `mask`: `[B, L]` - valid positions
#[cfg(feature = "training")]
pub fn accept_rate_loss(probs: Tensor<2>, target: Tensor<2>, mask: Tensor<2>) -> Tensor<1> {
    let se = (probs - target).powf_scalar(2.0) * mask.clone();
    se.sum().div(mask.sum().clamp_min(1.0))
}

/// Sequential Temperature Scaling (paper section 3.2.1).
///
/// Calibrates confidence scores `c: [B, L]` position by position from left
/// to right. At each position a 1D grid search over temperature `t`
/// minimizes the Expected Calibration Error of the cumulative product
/// `prod_{i<=k} c_i` against binary empirical acceptances `accepted: [B, L]`,
/// keeping previously calibrated scores fixed. Host-side helper: the search
/// runs on plain `f32` data.
#[cfg(feature = "training")]
pub fn sts_calibrate(c: &[f32], accepted: &[f32], b: usize, l: usize, grid: &[f32]) -> Vec<f32> {
    assert_eq!(c.len(), b * l);
    assert_eq!(accepted.len(), b * l);
    let mut cal = c.to_vec();
    for k in 0..l {
        let mut best_t = 1.0f32;
        let mut best_ece = f32::MAX;
        for &t in grid {
            // recalibrate position k only; the prefix stays fixed
            let mut ece = 0.0;
            for i in 0..b {
                let mut cum = 1.0f32;
                for j in 0..=k {
                    let v = if j == k {
                        (c[i * l + j] / t).clamp(0.0, 1.0)
                    } else {
                        cal[i * l + j]
                    };
                    cum *= v;
                }
                let target = accepted[i * l + k];
                ece += (cum - target).abs();
            }
            ece /= b as f32;
            if ece < best_ece {
                best_ece = ece;
                best_t = t;
            }
        }
        for i in 0..b {
            cal[i * l + k] = (c[i * l + k] / best_t).clamp(0.0, 1.0);
        }
    }
    cal
}

#[cfg(test)]
mod tests {
    use super::*;
    // the top-level Int import is behind the "training" feature
    use burn::tensor::{Device, Int};
    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn accept_rate_predictor_hidden_only() {
        let p = AcceptRatePredictor::new(16, &dev());
        let h = Tensor::<3>::zeros([2, 4, 16], &dev());
        assert_eq!(p.prob(h, None).dims(), [2, 4, 1]);
    }

    #[test]
    fn accept_rate_predictor_markov_conditioned() {
        // Regression: the conditioned variant used to size its Linear for
        // `input_dim` while logit() concatenated `[h; W1[x]]` of width
        // `input_dim + rank`, making the documented path uncallable.
        const RANK: usize = 8;
        let markov = VanillaMarkov::new(64, RANK, &dev());
        let p = AcceptRatePredictor::with_markov(16, RANK, &dev());
        let h = Tensor::<3>::zeros([2, 4, 16], &dev());
        let prev: Tensor<2, Int> = Tensor::zeros([2, 4], &dev());
        let pe = markov.get_prev_embeddings(prev);
        let out = p.prob(h, Some(pe));
        assert_eq!(out.dims(), [2, 4, 1]);
        let v: Vec<f32> = out
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(v.iter().all(|x| (0.0..=1.0).contains(x)));
    }

    #[test]
    #[should_panic(expected = "built hidden-only")]
    fn accept_rate_predictor_rejects_missing_markov_embeddings() {
        let p = AcceptRatePredictor::new(16, &dev());
        let h = Tensor::<3>::zeros([1, 2, 16], &dev());
        let _ = p.logit(h, Some(Tensor::<3>::zeros([1, 2, 8], &dev())));
    }

    /// Eq. 7 is `sigmoid(w^T[h_k ; W1[x_{k-1}]])`, and the PREVIOUS TOKEN is
    /// the mechanism, so this asserts the mechanism: hold the hidden state
    /// fixed, change the token, and the acceptance logit must move.
    ///
    /// The projection is BIAS-FREE and the hidden state here is ZERO, so the
    /// number that comes out is exactly `w_markov . W1[x]` - the markov block
    /// isolated by construction, with no surgery on the weights. The
    /// comparison is the second half: the hidden-only head on the same
    /// features is a constant 0.5, because it cannot read a token at all.
    /// Same inputs, different function - which is what `prob(h, None)` was
    /// doing in production, and why `logit` REFUSES `None` on a markov head
    /// rather than quietly degrading to `w^T[h_k]` (ADR-0019).
    #[test]
    fn accept_rate_predictor_reads_the_previous_token() {
        const D: usize = 8;
        const R: usize = 4;
        let markov = VanillaMarkov::new(64, R, &dev());
        let p = AcceptRatePredictor::with_markov(D, R, &dev());
        let h = Tensor::<3>::zeros([1, 3, D], &dev());
        let token = |i: i64| -> Tensor<2, Int> {
            Tensor::from_data(burn::tensor::TensorData::new(vec![i, 1 + i, 2 + i], [1, 3]), &dev())
        };
        let logit = |t: i64| -> Vec<f32> {
            p.logit(h.clone(), Some(markov.get_prev_embeddings(token(t))))
                .into_data()
                .try_to_vec()
                .unwrap()
        };
        let (a, b) = (logit(1), logit(9));
        let moved: f32 = (0..a.len()).map(|i| (a[i] - b[i]).abs()).fold(0.0, f32::max);
        assert!(moved > 1e-4, "the markov head ignores W1[x]: {a:?} vs {b:?}");

        let hidden_only: Vec<f32> = AcceptRatePredictor::new(D, &dev())
            .prob(h, None)
            .into_data()
            .try_to_vec()
            .unwrap();
        assert!(
            hidden_only.iter().all(|x| (x - 0.5).abs() < 1e-6),
            "a bias-free hidden-only head on a ZERO hidden state is sigmoid(0): {hidden_only:?}"
        );
    }

    #[test]
    fn position_weights_decay() {
        let w: Vec<f32> = position_weights(4, 4.0, &dev())
            .into_data()
            .try_to_vec()
            .unwrap();
        assert!((w[0] - 1.0).abs() < 1e-5);
        assert!(w[0] > w[1] && w[1] > w[2] && w[2] > w[3]);
        assert!(w[3] > 0.0);
    }

    #[test]
    fn accept_rate_target_bounds() {
        // identical distributions -> acceptance 1.0
        let l = accept_rate_target(
            Tensor::<3>::zeros([1, 4, 8], &dev()),
            Tensor::<3>::zeros([1, 4, 8], &dev()),
        );
        let v: Vec<f32> = l.into_data().try_to_vec().unwrap();
        assert!(v.iter().all(|x| (x - 1.0).abs() < 1e-4));
        // one-hot vs uniform -> 1 - 0.5*(1-1/V)*2 = 1/V
        let oh = Tensor::<3>::from_data(
            burn::tensor::TensorData::new(vec![10.0f32, 0.0, 0.0, 0.0], [1, 1, 4]),
            &dev(),
        );
        let uni = Tensor::<3>::zeros([1, 1, 4], &dev());
        let r = accept_rate_target(oh, uni);
        let x: f32 = r.squeeze_dim::<1>(1).reshape([1]).into_scalar();
        assert!((x - 0.25).abs() < 1e-3, "got {x}");
    }

    #[cfg(feature = "training")]
    #[test]
    fn dspark_loss_finite_and_components() {
        let (b, l, v) = (2, 6, 32);
        let dl = Tensor::<3>::random([b, l, v], burn::tensor::Distribution::Default, &dev());
        let tl = Tensor::<3>::random([b, l, v], burn::tensor::Distribution::Default, &dev());
        let ids: Tensor<2, Int> = Tensor::random(
            [b, l],
            burn::tensor::Distribution::Uniform(0.0, v as f64),
            &dev(),
        );
        let conf = Tensor::<3>::random([b, l, 1], burn::tensor::Distribution::Default, &dev());
        let mask = Tensor::<2>::ones([b, l], &dev());
        let (total, ce, tv, c) = dspark_loss(dl, tl, ids, Some(conf), mask, 4.0);
        assert!(as_f32(total).is_finite());
        assert!(as_f32(ce) > 0.0);
        assert!(as_f32(tv) > 0.0);
        assert!(as_f32(c) > 0.0);
    }

    #[cfg(feature = "training")]
    #[test]
    fn sts_reduces_ece() {
        // badly calibrated: c = 0.9 everywhere, accepted ~ 0.3
        let (b, l) = (64, 4);
        let c = vec![0.9f32; b * l];
        let accepted: Vec<f32> = (0..b * l)
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let grid: Vec<f32> = (5..=20).map(|t| t as f32 / 10.0).collect();
        let cal = sts_calibrate(&c, &accepted, b, l, &grid);
        // calibrated cumulative products must be closer to empirical rates
        let ece_before = cum_ece(&c, &accepted, b, l);
        let ece_after = cum_ece(&cal, &accepted, b, l);
        assert!(
            ece_after <= ece_before + 1e-6,
            "{ece_before} -> {ece_after}"
        );
    }

    #[cfg(feature = "training")]
    fn cum_ece(c: &[f32], accepted: &[f32], b: usize, l: usize) -> f32 {
        let mut ece = 0.0;
        for i in 0..b {
            let mut cum = 1.0f32;
            for j in 0..l {
                cum *= c[i * l + j];
                ece += (cum - accepted[i * l + j]).abs();
            }
        }
        ece / (b * l) as f32
    }

    #[cfg(feature = "training")]
    fn as_f32(t: Tensor<1>) -> f32 {
        t.into_scalar()
    }
}
