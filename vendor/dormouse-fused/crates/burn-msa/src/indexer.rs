//! The compressed block indexer (tech-report §QSA Eq. 12-16).
//!
//! `H` MQA query heads (paper: 4), one shared key head; keys are average-
//! pooled into non-overlapping blocks of `r` tokens *before* the norm and
//! the positional encoding (Eq. 13's explicit ordering), so every block
//! speaks as one content vector at its first token's position. Scores are
//! ReLU'd per head and summed over heads (Eq. 15), block-causal, masked
//! with `-1e30` rather than `-inf` (ADR-0015: -inf reaches the empty-slot
//! comparison as a TRUE comparison).
//!
//! Selection is [`select_blocks`]. The vendored `argtopk` (ADR-0015)
//! guarantees in-range indices; its remaining "partial row" defect means a
//! row with fewer visible blocks than `kb` may name a masked column —
//! in range, and CAUSALLY KILLED at the layer by the window mask
//! mask, which is the gate that proves the leak impossible. `kb == B` does
//! NOT go through topk at all (burn's `argtopk` asserts `k < n`, and the
//! all-blocks call is the dense-equivalence gate, not a selection):
//! `select_blocks` emits the identity there, which is also what makes the
//! gate meaningful — the ONLY difference between sparse(all) and dense is
//! the gather/pack path itself.

use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Int, Tensor};

use burn_rope::precompute_freqs;

use crate::blocks;

/// Indexer geometry. `head_dim` is BOTH the per-query-head width and the
/// shared key-head width (paper keeps them equal); the model's own
/// core-attention head dims stay separate.
#[derive(Debug, Clone)]
pub struct IndexerConfig {
    /// Input width (the model's `d_model`).
    pub d_model: usize,
    /// MQA query heads `H` (paper: 4).
    pub q_heads: usize,
    /// Head width and shared key-head width.
    pub head_dim: usize,
    /// Compression ratio `r`: tokens per block (paper: 4).
    pub block_r: usize,
    /// Partial-RoPE width: only the first `rope_frac` dims of each head
    /// rotate (paper: 64 of 128). Even, `<= head_dim`.
    pub rope_frac: usize,
    /// Longest sequence the indexer's RoPE tables cover.
    pub max_seq_len: usize,
}

/// The compressed MQA indexer — Eq. 12-15 as one module. Learnables live
/// here; this is what the two-stage distillation trains (stage (a): ONLY
/// these parameters, high LR — paper 1e-3).
///
/// "Zero-Centered RMSNorm" is wired as the library's own RMS formula (the
/// Zhang&Sennrich mean form the whole model uses), applied in-place here
/// rather than through a second module type — see [`rms`].
#[derive(Module, Debug)]
pub struct IndexerModule {
    /// `W_q`: `d -> q_heads * head_dim`, no bias (Eq. 12).
    pub q_proj: Linear,
    /// `W_k`: `d -> head_dim`, no bias (Eq. 12).
    pub k_proj: Linear,
    /// Per-head RMSNorm gain, applied to the projected split heads (Eq. 12:
    /// `b^h_i = RMSNorm(W^h_q x_i)` — a norm per head, NOT one over the
    /// fused width).
    pub q_gain: Param<Tensor<1>>,
    /// Key RMSNorm gain, applied AFTER the average pool (Eq. 13).
    pub k_gain: Param<Tensor<1>>,
    /// RoPE tables `[max_seq_len, rope_frac/2]` (cos, sin).
    pub rope_cos: Param<Tensor<2>>,
    pub rope_sin: Param<Tensor<2>>,
    /// Block geometry the module scores at: `r` tokens per block.
    pub block_r: usize,
    pub q_heads: usize,
    pub head_dim: usize,
    pub rope_frac: usize,
    pub eps: f32,
}

impl IndexerModule {
    pub fn new(config: IndexerConfig, device: &Device) -> Self {
        let hd = config.head_dim;
        assert!(
            config.rope_frac % 2 == 0 && config.rope_frac <= hd && config.rope_frac >= 2,
            "IndexerConfig.rope_frac = {} must be even and 2 <= rope_frac <= head_dim {hd}",
            config.rope_frac
        );
        assert!(config.q_heads >= 1, "at least one MQA query head");
        assert!(config.block_r >= 1, "block_r is a token count, not a ratio");
        let (rope_cos, rope_sin) =
            precompute_freqs(config.rope_frac, config.max_seq_len, 10000.0, device);
        Self {
            q_proj: LinearConfig::new(config.d_model, config.q_heads * hd)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(config.d_model, hd)
                .with_bias(false)
                .init(device),
            q_gain: Param::from_tensor(Tensor::ones([hd], device)),
            k_gain: Param::from_tensor(Tensor::ones([hd], device)),
            rope_cos: Param::from_tensor(rope_cos),
            rope_sin: Param::from_tensor(rope_sin),
            block_r: config.block_r,
            q_heads: config.q_heads,
            head_dim: hd,
            rope_frac: config.rope_frac,
            eps: 1e-6,
        }
    }
}

/// The library's RMS formula (Zhang & Sennrich `x / sqrt(mean(x^2) + eps)`
/// with a learned gain) on the last axis of any rank `D` — one
/// implementation, not a second module.
fn rms<const D: usize>(x: Tensor<D>, gain: &Param<Tensor<1>>, eps: f32) -> Tensor<D> {
    let norm = x
        .clone()
        .powf_scalar(2.0)
        .mean_dim(D - 1)
        .add_scalar(eps)
        .sqrt()
        .recip();
    // A rank-1 gain joined onto the LAST axis: [.., hd] * [1..1(hd)].
    let mut shape = [1usize; D];
    shape[D - 1] = gain.val().dims()[0];
    (x * norm) * gain.val().clone().reshape(shape)
}

/// Partial RoPE on the last axis of `x [n, m, rf_even]`: half-split rotate.
/// `cos`/`sin` `[n, rf/2]` have their rows named by the SAME first axis, so
/// the caller lays its positions along it. Query (per-token rows) and
/// keys-bar (per-block rows, Eq. 13's block-level positions) both go
/// through this one function; being self-consistent inside the indexer is
/// what makes "partial RoPE" faithful where it can differ (only `rope_frac
/// < head_dim` dims rotate).
fn rope_rotate3(x: Tensor<3>, cos: Tensor<2>, sin: Tensor<2>) -> Tensor<3> {
    let [n, m, rf] = x.dims();
    let h = rf / 2;
    // Rows named by the SECOND axis; the first axis broadcasts (the query
    // batch axis b never appears in the position tables).
    let c = cos.unsqueeze_dim::<3>(0).expand([n, m, h]);
    let s = sin.unsqueeze_dim::<3>(0).expand([n, m, h]);
    let x1 = x.clone().slice([0..n, 0..m, 0..h]);
    let x2 = x.slice([0..n, 0..m, h..rf]);
    burn::tensor::Tensor::cat(
        vec![x1.clone() * c.clone() - x2.clone() * s.clone(), x2 * c + x1 * s],
        2,
    )
}

/// RoPE rows selected by position: `cos [m, rf/2]` for `positions [m]`.
fn rope_at(m: &IndexerModule, positions: Tensor<1, Int>) -> (Tensor<2>, Tensor<2>) {
    (
        m.rope_cos.val().select(0, positions.clone()),
        m.rope_sin.val().select(0, positions),
    )
}

/// RoPE rows for per-token queries: `cos rows [t, rf/2]` repeated per
/// query head and folded into the `[t*qh, rf/2]` rows `rope_rotate3`
/// expects (row order `t*qh + h`, the same order the caller reshapes the
/// query into). One helper so the call sites cannot disagree about the
/// layout.
fn rope_rows_per_token(m: &IndexerModule, t: usize, q_heads: usize) -> (Tensor<2>, Tensor<2>) {
    let rf2 = m.rope_frac / 2;
    let cos = m
        .rope_cos
        .val()
        .slice([0..t, 0..rf2])
        .unsqueeze_dim::<3>(1)
        .expand([t, q_heads, rf2]);
    let sin = m
        .rope_sin
        .val()
        .slice([0..t, 0..rf2])
        .unsqueeze_dim::<3>(1)
        .expand([t, q_heads, rf2]);
    (
        cos.reshape([t * q_heads, rf2]),
        sin.reshape([t * q_heads, rf2]),
    )
}

/// The indexer's block scores I (Eq. 15): `[b, t, B]` with
/// `B = t / block_r` complete blocks, `-1e30` where `r*b + r - 1 > i`
/// (block-causal: only fully observed blocks are scored; Eq. 15's
/// "otherwise"). The B columns are EXACTLY the block space the caller
/// packs into the window: block b holds tokens `r*b .. r*(b+1)`.
pub fn indexer_scores(m: &IndexerModule, x: Tensor<3>) -> Tensor<3> {
    let [b, t, _d] = x.dims();
    let info = blocks(t, m.block_r);
    let b_n = info.blocks;
    assert!(
        b_n >= 1,
        "seq_len {t} shorter than one complete block of {} - the indexer cannot score anything",
        m.block_r
    );

    // Eq. 12 (keys): one shared k head, raw projection per token.
    let k = m.k_proj.forward(x.clone()); // [b, t, hd]
    // Eq. 13: AvgPool over the complete blocks, THEN the norm.
    let k_bar = k
        .slice([0..b, 0..b_n * m.block_r, 0..m.head_dim])
        .reshape([b, b_n, m.block_r, m.head_dim])
        .mean_dim(2)
        .reshape([b, b_n, m.head_dim]); // [b, B, hd]
    let k_bar = rms::<3>(k_bar, &m.k_gain, m.eps);
    // Eq. 14: block-level position p_b = r*b - the RoPE rows at those starts.
    let dev = k_bar.device();
    let pb = Tensor::<1, Int>::arange(0..b_n as i64, &dev).mul_scalar(m.block_r as i64);
    let (cos, sin) = rope_at(m, pb);
    let k_hd = k_bar.dims()[2];
    let rf = m.rope_frac;
    let k_bar = if rf == k_hd {
        rope_rotate3(k_bar, cos, sin)
    } else {
        // Partial RoPE. The range slice is STATIC (a module constant), not
        // the dynamically-computed 4D slicing sm_120 refuses.
        let anchor = k_bar.clone().slice([0..b, 0..b_n, rf..k_hd]);
        let rot = rope_rotate3(k_bar.slice([0..b, 0..b_n, 0..rf]), cos, sin);
        burn::tensor::Tensor::cat(vec![rot, anchor], 2)
    };

    // Eq. 12 (queries): project, a norm PER HEAD, then RoPE at position i.
    let q = m
        .q_proj
        .forward(x)
        .reshape([b, t, m.q_heads, k_hd]); // [b, t, qh, hd]
    let q = rms::<4>(q, &m.q_gain, m.eps);
    let (cos_q, sin_q) = rope_rows_per_token(m, t, m.q_heads);
    let q = if rf == k_hd {
        let q3 = q.reshape([b, t * m.q_heads, k_hd]);
        rope_rotate3(q3, cos_q, sin_q).reshape([b, t, m.q_heads, k_hd])
    } else {
        let anchor = q.clone().slice([0..b, 0..t, 0..m.q_heads, rf..k_hd]);
        let rot = rope_rotate3(
            q.slice([0..b, 0..t, 0..m.q_heads, 0..rf])
                .reshape([b, t * m.q_heads, rf]),
            cos_q,
            sin_q,
        )
        .reshape([b, t, m.q_heads, rf]);
        burn::tensor::Tensor::cat(vec![rot, anchor], 3)
    };

    // Eq. 15: per-head ReLU(q_h . kbar_b) summed over heads.
    let per_head = q
        .swap_dims(1, 2) // [b, qh, t, hd]
        .matmul(
            k_bar
                .swap_dims(1, 2) // [b, hd, B]
                .unsqueeze_dim::<4>(1)
                .repeat_dim(1, m.q_heads), // [b, qh, hd, B]
        ); // [b, qh, t, B]
    let dots = burn::tensor::activation::relu(per_head)
        .sum_dim(1) // [b, 1, t, B]
        .reshape([b, t, b_n]);

    // The mask (Eq. 15's condition): block b visible to query i iff
    // r*b + r - 1 <= i. mask_fill is the bool consumer - no numeric is
    // BUILT from a bool on device (AGENTS.md 1.3).
    let pos = Tensor::<1, Int>::arange(0..t as i64, &dev)
        .unsqueeze_dim::<2>(1)
        .unsqueeze_dim::<3>(0); // [1, t, 1]
    let end = Tensor::<1, Int>::arange(0..b_n as i64, &dev)
        .mul_scalar(m.block_r as i64)
        .add_scalar((m.block_r as i64) - 1)
        .reshape([1, 1, b_n]); // [1, 1, B]
    let invisible = end.greater(pos); // [1, t, B]
    dots.mask_fill(invisible, -1e30)
}

/// The `KB` blocks each query attends (Eq. 16): the top-`kb` of the
/// indexer's row of scores. `kb == B` short-circuits to the identity (see
/// the crate root). Returns `[b, q(=t), kb]`, Int, in `[0, B)`.
pub fn select_blocks(scores: Tensor<3>, kb: usize) -> Tensor<3, Int> {
    let [b, q, n] = scores.dims();
    assert!(kb >= 1, "the empty selection is a lost arm, not an option");
    if kb >= n {
        let dev = scores.device();
        return Tensor::<1, Int>::arange(0..n as i64, &dev)
            .reshape([1, 1, n])
            .expand([b, q, n]); // [b, q, n], identity
    }
    scores.argtopk(kb, 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> Device {
        Device::ndarray()
    }

    fn cfg() -> IndexerConfig {
        IndexerConfig {
            d_model: 16,
            q_heads: 4,
            head_dim: 8,
            block_r: 4,
            rope_frac: 8,
            max_seq_len: 32,
        }
    }

    fn to_f32(x: Tensor<3>) -> Vec<f32> {
        x.into_data().bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
    }

    fn to_i64(x: Tensor<3, Int>) -> Vec<i64> {
        x.into_data()
            .convert::<i64>()
            .bytes
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn scores_are_block_causal() {
        let dev = dev();
        let m = IndexerModule::new(cfg(), &dev);
        let x = Tensor::<3>::random([2, 16, 16], burn::tensor::Distribution::Default, &dev);
        let s = indexer_scores(&m, x); // [2, 16, 4]
        let v = to_f32(s);
        // Block 3 spans tokens 12..15; Eq. 15's condition (r*b + r - 1 <= i)
        // makes it visible from query 15 ALONE: 15 <= i.
        for row in 0..15usize {
            assert!(
                v[row * 4 + 3] == -1e30,
                "query {row} scored the unobserved block 3 at {}",
                v[row * 4 + 3]
            );
        }
        for row in [15usize] {
            assert!(
                v[row * 4 + 3] > -1e29,
                "query {row} must see block 3, saw {}",
                v[row * 4 + 3]
            );
        }
    }

    #[test]
    fn select_block_ids_in_range_and_identity_when_full() {
        let dev = dev();
        let m = IndexerModule::new(cfg(), &dev);
        let x = Tensor::<3>::random([1, 16, 16], burn::tensor::Distribution::Default, &dev);
        let s = indexer_scores(&m, x);
        let all = to_i64(select_blocks(s.clone(), 4));
        // kb == B is the identity: every query sees blocks 0..3.
        for row in 0..16usize {
            for (slot, want) in [(0usize, 0), (1, 1), (2, 2), (3, 3)] {
                assert_eq!(all[row * 4 + slot], want, "kb == B must be the identity");
            }
        }
        // kb = 2: in range, exactly 2 picks per row.
        let half = to_i64(select_blocks(s, 2));
        for &idx in &half {
            assert!((0..4).contains(&idx), "index {idx} out of [0,4)");
        }
        assert_eq!(half.len(), 32, "a kb-2 call returns exactly 2 per row");
    }
}
