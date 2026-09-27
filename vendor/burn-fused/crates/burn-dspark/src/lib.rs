//! # burn-dspark - DSpark Speculative Decoding for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! Building blocks from [DSpark](https://arxiv.org/abs/2607.05147) (DeepSeek AI, 2026).
//! NOT yet matched against the official
//! [DeepSpec](https://github.com/deepseek-ai/DeepSpec) implementation — the
//! 7 tests here are hand-derived and cannot detect a wrong loss term, a wrong
//! `gamma`, or L1-on-logits instead of L1-on-probs:
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
//! burn-dspark = "0.1"                    # inference
//! burn-dspark = { version = "0.1", features = ["training"] }  # + loss fns
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
pub fn dspark_loss(
    draft_logits: Tensor<3>,
    target_logits: Tensor<3>,
    target_ids: Tensor<2, Int>,
    confidence_logits: Tensor<3>,
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
    let c_star = accept_rate_target(draft_logits, target_logits); // [B, L]
    let cp = activation::sigmoid(confidence_logits.reshape([b, l]));
    let eps = 1e-7f32;
    let bce_per = c_star
        .clone()
        .mul(cp.clone().add_scalar(eps).log())
        .add(
            c_star
                .neg()
                .add_scalar(1.0)
                .mul(cp.neg().add_scalar(1.0 + eps).log()),
        )
        .neg();
    let conf = (bce_per * wm).sum().div(den);

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
        let (total, ce, tv, c) = dspark_loss(dl, tl, ids, conf, mask, 4.0);
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
