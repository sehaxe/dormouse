//! # burn-fastblt - Byte-Level BLT + FastBLT for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! | arXiv | Component | What |
//! |-------|-----------|------|
//! | [2412.09871](https://arxiv.org/abs/2412.09871) | `byte_patch` | Entropy-based dynamic byte patching (BLT, Meta 2024) |
//! | [2412.09871](https://arxiv.org/abs/2412.09871) | `byte_group_hash_ids`, `HashEmbeddings` | Byte n-gram hash embeddings (BLT, Meta 2024) |
//! | [2605.08044](https://arxiv.org/abs/2605.08044) | `bltd_loss` | Block-wise masked diffusion loss (FastBLT Eq 5-7) |
//! | [2605.08044](https://arxiv.org/abs/2605.08044) | `verify_draft` | Greedy draft verification (FastBLT Alg. 2) |
//! | [2605.08044](https://arxiv.org/abs/2605.08044) | `unmask_*` | Confidence / entropy-bounded block unmasking (FastBLT 3.1.2) |
pub mod hash;

use burn::tensor::{activation, Bool, IndexingUpdateOp, Int, Tensor};

/// Dynamic byte patching by next-byte entropy (BLT, 2412.09871 §2.3).
///
/// Faithful to Meta's `bytelatent/data/patcher.py`: the entropy of a byte is
/// `-sum p * ln p` over its next-byte softmax (natural log). The first byte
/// always opens a patch; every byte whose entropy exceeds `entropy_threshold`
/// also opens one. Patches longer than `max_patch_size` are split into chunks
/// of at most that many bytes (the reference `max_patch_length`).
///
/// Returns `(patch_start_ids, patch_lengths)`:
/// - `patch_start_ids[b, k]` - byte index (into `bytes`) where patch `k` starts
/// - `patch_lengths[b, k]` - number of bytes in patch `k`; trailing `0`s are padding
///
/// Per row `sum(patch_lengths) == seq_len`. Pure tensor ops, zero CPU sync.
pub fn byte_patch(
    bytes: Tensor<2, Int>,
    logits: Tensor<3>,
    entropy_threshold: f32,
    max_patch_size: usize,
) -> (Tensor<2, Int>, Tensor<2, Int>) {
    let device = bytes.device();
    let [b, t] = bytes.dims();
    let m = max_patch_size.max(1) as i64;

    // next-byte entropy, natural log (Meta patcher.entropy)
    let log_probs = activation::log_softmax(logits, 2);
    let probs = log_probs.clone().exp();
    let entropy = -(log_probs * probs).sum_dim(2).squeeze_dim(2); // [b, t]

    // positions 0..=t; position `t` is the end sentinel closing the last patch
    let pos = Tensor::<1, Int>::arange(0..(t as i64 + 1), &device)
        .unsqueeze_dim::<2>(0)
        .expand([b, t + 1]);

    // patch starts: first byte always, then bytes above the entropy threshold
    let is_start = entropy
        .greater_elem(entropy_threshold)
        .bool_or(pos.clone().slice([0..b, 0..t]).equal_elem(0_i64));
    let is_start = Tensor::cat(
        vec![
            is_start,
            Tensor::<2, Bool>::zeros([b, 1], &device).bool_not(),
        ],
        1,
    );

    // enforce max_patch_size: a byte a multiple of `m` past its patch start
    // also opens a chunk
    let (starts, _lens) = starts_and_lengths(is_start.clone(), &pos);
    let rank = is_start.clone().int().cumsum(1) - 1;
    let prev_start = starts.gather(1, rank);
    let is_chunk = is_start.bool_or(
        (pos.clone() - prev_start)
            .remainder_scalar(m)
            .equal_elem(0_i64),
    );

    starts_and_lengths(is_chunk, &pos)
}

/// Patch start positions and lengths for a start mask over positions `0..=t`.
fn starts_and_lengths(
    is_start: Tensor<2, Bool>,
    pos: &Tensor<2, Int>,
) -> (Tensor<2, Int>, Tensor<2, Int>) {
    let device = pos.device();
    let [b, n] = is_start.dims();
    let t = n - 1;

    let rank = is_start.clone().int().cumsum(1) - 1; // patch index of each position
    let values = pos.clone().mask_fill(is_start.clone().bool_not(), 0_i64);
    // scatter is a sum-reduction, but only one position per column contributes
    let starts =
        Tensor::<2, Int>::zeros([b, n], &device).scatter(1, rank, values, IndexingUpdateOp::Add); // column k = byte index of the k-th start
    let counts = is_start.int().sum_dim(1); // [b, 1] starts per row
    let starts = starts.mask_fill(pos.clone().lower(counts).bool_not(), t as i64);

    let next = Tensor::cat(
        vec![
            starts.clone().slice([0..b, 1..n]),
            Tensor::full([b, 1], t as i64, &device),
        ],
        1,
    );
    let lengths = next - starts.clone();
    (starts, lengths)
}

/// Block-wise masked diffusion loss (FastBLT, 2605.08044, Eq 6).
///
/// Absorbing discrete diffusion: each byte is masked with probability `t`,
/// the decoder predicts the original byte at masked positions, and the loss
/// is the masked next-byte CE scaled by `1/t` (the simplified ELBO):
///
/// ```text
/// L_mask = -(1/t) * sum_i 1[x_i masked] * log p(x_i | x^t, prefix)
/// ```
///
/// `logits`: `[B, L, V]` - decoder predictions for the corrupted block
/// `targets`: `[B, L]` - original byte ids
/// `mask`: `[B, L]` - 1.0 = masked position (diffused), 0.0 = clean
/// `t`: the diffusion timestep used to corrupt the block.
pub fn bltd_loss(logits: Tensor<3>, targets: Tensor<2, Int>, mask: Tensor<2>, t: f32) -> Tensor<1> {
    let [b, l, v] = logits.dims();
    let flat = logits.reshape([b * l, v]);
    let lp = activation::log_softmax(flat, 1);
    let nll = -lp
        .gather(1, targets.reshape([b * l]).unsqueeze_dim::<2>(1))
        .reshape([b, l]);
    let zeros = Tensor::zeros_like(&nll);
    let nll = nll.mask_where(mask.clone().equal_elem(0.0), zeros);
    let t = t.max(1e-4);
    nll.sum().div(mask.sum().clamp_min(1.0)).div_scalar(t)
}

/// Greedy verification of a drafted byte block (FastBLT Algorithm 2,
/// BLT-S / BLT-DV shared procedure).
///
/// `draft`: `[B, K]` - drafted byte ids (K bytes beyond the prefix)
/// `verified_next`: `[B, K + 1]` - greedy next-byte predictions of the full
/// model for positions after each draft byte (the last entry is the free
/// bonus byte after the draft).
///
/// Accepts the draft up to the first mismatch, replaces the first mismatch
/// with the model's prediction, and appends the free byte when the whole
/// draft matches. Returns the accepted bytes `[B, K + 1]` (may contain
/// trailing padding zeros that the caller truncates).
pub fn verify_draft(draft: Tensor<2, Int>, verified_next: Tensor<2, Int>) -> Tensor<2, Int> {
    let [b, k] = draft.dims();
    let device = draft.device();
    // eq[b, j]: the model's prediction at j matches the draft byte.
    let eq = draft
        .clone()
        .equal(verified_next.clone().slice([0..b, 0..k])); // [B, k]
                                                           // accepted[b] = length of the all-true prefix of eq: a position's
                                                           // cumsum equals its 1-based index iff every earlier position matched.
    let pos = Tensor::<1, Int>::arange(1..(k as i64 + 1), &device)
        .unsqueeze_dim::<2>(0)
        .expand([b, k]);
    let accepted = eq.clone().int().cumsum(1).equal(pos).int().sum_dim(1); // [B, 1]
                                                                           // out[b, j] = draft[b, j] for j < accepted, verified_next[b, j] for
                                                                           // j == accepted (the replacement, or the free bonus byte when the whole
                                                                           // draft matches), 0 otherwise. All device-side: no host roundtrip.
    let jj = Tensor::<1, Int>::arange(0..(k as i64 + 1), &device)
        .unsqueeze_dim::<2>(0)
        .expand([b, k + 1]);
    let take_draft = jj.clone().lower(accepted.clone()); // [B, k+1]
    let take_vn = jj.equal(accepted); // [B, k+1]
    let draft_pad = Tensor::cat(vec![draft, Tensor::<2, Int>::zeros([b, 1], &device)], 1);
    draft_pad
        .mul(take_draft.int())
        .add(verified_next.mul(take_vn.int()))
}

/// Confidence-based block unmasking (FastBLT 3.1.2): unmask all masked
/// positions whose max predicted probability exceeds `alpha`; if none do,
/// unmask the single most confident position.
///
/// `probs`: `[B, L, V]` - per-position byte distributions
/// `masked`: `[B, L]` - 1.0 = still masked
/// Returns the updated mask (0.0 = unmasked this step).
pub fn unmask_confidence(probs: Tensor<3>, masked: Tensor<2>, alpha: f32) -> Tensor<2> {
    let [_, l, _] = probs.dims();
    let conf = probs.max_dim(2).squeeze_dim::<2>(2); // [B, L]
    let above = conf.clone().greater_elem(alpha).float().mul(masked.clone());
    let any_above = above
        .clone()
        .sum_dim(1)
        .squeeze_dim::<1>(1)
        .greater_elem(0.5)
        .float()
        .unsqueeze_dim::<2>(1); // [B, 1]
                                // fallback: the single most confident masked position (zeroed when the
                                // threshold already unmasked something)
    let best = conf.clone().argmax(1); // [B]
    let fallback = best
        .clone()
        .one_hot::<3>(l)
        .float()
        .squeeze_dim::<2>(1)
        .mul(masked.clone())
        .mul(any_above.clone().neg().add_scalar(1.0)); // [B, L]
    let unmask = above.add(fallback).clamp(0.0, 1.0);
    masked.sub(unmask).clamp(0.0, 1.0)
}

/// Entropy-bounded sampling (FastBLT 3.1.2): sort masked positions by
/// ascending entropy and unmask the largest subset whose cumulative entropy
/// does not exceed `gamma`; if no masked position fits, unmask the
/// lowest-entropy MASKED one (so every call makes progress).
///
/// `probs`: `[B, L, V]`, `masked`: `[B, L]` (1.0 = masked).
/// Returns the updated mask (0.0 = unmasked this step).
pub fn unmask_entropy_bounded(probs: Tensor<3>, masked: Tensor<2>, gamma: f32) -> Tensor<2> {
    let [b, l, _] = probs.dims();
    let device = probs.device();
    let lp = activation::log_softmax(probs.clone(), 2);
    let p = lp.clone().exp();
    let entropy = -(lp * p).sum_dim(2).squeeze_dim::<2>(2); // [B, L]
                                                            // Sort key: masked positions keep their entropy; CLEAN positions get a
                                                            // large sentinel so they always sort last. Without this, zero-entropy
                                                            // clean positions would sort first and both the budget check and the
                                                            // "at least one" fallback could pick already-clean positions (making no
                                                            // progress on a fully-over-budget row). 1e9 per element cannot overflow
                                                            // an f32 cumsum for any realistic sequence length.
    const SENTINEL: f32 = 1e9;
    let h = entropy.mul(masked.clone()) + masked.clone().neg().add_scalar(1.0).mul_scalar(SENTINEL);
    // ascending key order: lowest-entropy masked positions first, clean last
    let order = h.clone().argsort(1); // [B, L] indices into the row
    let sorted = h.gather(1, order.clone()); // [B, L]
    let cumsum = sorted.cumsum(1); // [B, L]
    let fits = cumsum.lower_equal_elem(gamma).float(); // 1.0 = within budget
    let mut selected = fits;
    // Guarantee forward progress: always unmask the single lowest-entropy
    // position of the row. It is now guaranteed to be a MASKED one.
    let col0 = selected
        .clone()
        .slice([0..b, 0..1])
        .add_scalar(1.0)
        .clamp(0.0, 1.0);
    selected = Tensor::cat(vec![col0, selected.slice([0..b, 1..l])], 1);
    // scatter back to original positions: for each row, positions `order[r, j]`
    // get `selected[r, j]`; scatter with Add over a zeros tensor accumulates
    let ones = selected.reshape([b, l]);
    let idx = order.unsqueeze_dim::<3>(2).expand([b, l, 1]);
    let unmask = Tensor::<3>::zeros([b, l, 1], &device)
        .scatter(1, idx, ones.unsqueeze_dim::<3>(2), IndexingUpdateOp::Add)
        .squeeze_dim::<2>(2)
        .clamp(0.0, 1.0);
    // clamp keeps already-clean rows untouched (their mask entry is 0)
    masked.sub(unmask).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::ndarray()
    }
    fn ints(t: Tensor<2, Int>) -> Vec<i64> {
        t.into_data().try_to_vec::<i64>().unwrap()
    }

    #[test]
    fn byte_patch_splits_long_patches() {
        // confident logits (argmax class 0, entropy ~0) except uniform rows 0
        // and 30 (entropy ln 256 > 2.5) which open patches at those bytes
        let bytes = Tensor::<2, Int>::zeros([1, 64], &dev());
        let rows = Tensor::<1, Int>::arange(0..64, &dev())
            .unsqueeze_dim::<2>(0)
            .expand([1, 64]);
        let uniform = rows
            .clone()
            .equal_elem(0_i64)
            .bool_or(rows.equal_elem(30_i64));
        let class0 = Tensor::<1, Int>::arange(0..256, &dev()).equal_elem(0_i64);
        let hot = uniform
            .bool_not()
            .unsqueeze_dim::<3>(2)
            .expand([1, 64, 256])
            .bool_and(
                class0
                    .unsqueeze_dim::<2>(0)
                    .unsqueeze_dim::<3>(1)
                    .expand([1, 64, 256]),
            );
        let logits = hot.float().mul_scalar(10.0);

        let (starts, lens) = byte_patch(bytes, logits, 2.5, 16);
        assert_eq!(&ints(lens.clone())[..6], &[16, 14, 16, 16, 2, 0]);
        assert_eq!(&ints(starts)[..6], &[0, 16, 30, 46, 62, 64]);
        // trailing columns are zero-length padding; all bytes are covered
        let lens = ints(lens);
        assert_eq!(lens.iter().sum::<i64>(), 64);
        assert!(lens[6..].iter().all(|&l| l == 0));
    }

    #[test]
    fn bltd_loss_finite_and_scaled() {
        let logits = Tensor::<3>::zeros([2, 16, 64], &dev());
        let targets: Tensor<2, Int> = Tensor::zeros([2, 16], &dev());
        let m = Tensor::<2>::ones([2, 16], &dev());
        let l1: f32 = bltd_loss(logits.clone(), targets.clone(), m.clone(), 1.0).into_scalar();
        let l05: f32 = bltd_loss(logits, targets, m, 0.5).into_scalar();
        assert!(l1.is_finite() && l05.is_finite());
        // scaling by 1/t: t=0.5 doubles the loss (ln 64 * 2)
        assert!((l05 - 2.0 * l1).abs() < 1e-3, "{l1} vs {l05}");
    }

    #[test]
    fn unmask_confidence_threshold() {
        // confident at position 0 (p_max 0.9), uncertain elsewhere
        let mut data = vec![0.1f32; 2 * 4 * 8];
        data[0] = 10.0; // [0,0]
        data[32] = 10.0; // [1,0] confident
        let probs = Tensor::<3>::from_data(burn::tensor::TensorData::new(data, [2, 4, 8]), &dev());
        let masked = Tensor::<2>::ones([2, 4], &dev());
        let out = unmask_confidence(probs, masked, 0.7);
        let v: Vec<f32> = out.into_data().to_vec().unwrap();
        // only position 0 of each row is unmasked
        assert_eq!(&v[0..4], &[0.0, 1.0, 1.0, 1.0]);
        assert_eq!(&v[4..8], &[0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn verify_draft_accepts_prefix() {
        // draft [3, 5, 7], model predicts [3, 5, 9, 1] -> accept 3,5, replace 9, bonus 1
        let draft: Tensor<2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(vec![3i64, 5, 7], [1, 3]),
            &dev(),
        );
        let vn: Tensor<2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(vec![3i64, 5, 9, 1], [1, 4]),
            &dev(),
        );
        let out = verify_draft(draft, vn);
        let v: Vec<i64> = out.into_data().to_vec().unwrap();
        // first mismatch at j=2: accept 3,5, replace with 9, no bonus byte
        assert_eq!(v, vec![3, 5, 9, 0]);
    }

    #[test]
    fn verify_draft_all_match() {
        let draft: Tensor<2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(vec![1i64, 2, 3], [1, 3]),
            &dev(),
        );
        let vn: Tensor<2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(vec![1i64, 2, 3, 4], [1, 4]),
            &dev(),
        );
        let out = verify_draft(draft, vn);
        let v: Vec<i64> = out.into_data().to_vec().unwrap();
        assert_eq!(v, vec![1, 2, 3, 4]);
    }
}
