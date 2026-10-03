//! The dense-distillation loss (tech-report §QSA Eq. 17-18): the
//! head-summed L1-normalized token distribution of the dense teacher,
//! MAX-pooled into complete blocks (the preserving aggregation),
//! L1-normalized again — then the KL against the indexer's own block
//! scores, mean over queries. Only the indexer's learnables see a gradient
//! by construction: the teacher is the caller's DETACHED dense
//! distribution and the scores are the only tracked input. (Stage (a)'s
//! "train ONLY the indexer" lives in the optimizer groups, not here.)
use burn::tensor::{Device, Tensor};

use burn::tensor::activation;

/// Eq. 17's max-pool into the complete block space: `a [b, n, t]` -> a max
/// over the `r` tokens of each complete block -> `[b, n, B]`, then L1-
/// normalized per query (`ahat`). Zero future-probability blocks fall out
/// of the KL on their own — a zero-weight term of a finite log stays zero
/// (the log is guarded below, NOT the input: an all-zero teacher ROW means
/// a query whose hot block is never visible, and its KL term is 0 exactly).
pub fn pool_teacher(a: Tensor<3>, block_r: usize) -> Tensor<3> {
    let [b, q, n] = a.dims(); // [batch, query rows, TOKENS last]
    let info = crate::blocks(n, block_r);
    assert!(
        info.blocks >= 1,
        "teacher seq {n} has no complete block of {block_r}"
    );
    let pooled = a
        .slice([0..b, 0..q, 0..info.blocks * block_r])
        .reshape([b, q, info.blocks, block_r])
        .max_dim(3)
        .reshape([b, q, info.blocks]);
    // The L1 normalization of Eq. 17. A query's head-summed distribution has
    // at least one positive token at or before itself, so the divisor is > 0
    // structurally; the epsilon guards an f32 softmax underflow on a long
    // sequence - an f32 guard of 1e-20, not a repair of a broken input.
    let den = pooled.clone().sum_dim(2).add_scalar(1e-20);
    pooled / den
}

/// Eq. 18: `mean_i KL(ahat_i || softmax(I_i))`. `scores [b, t, B]` from
/// [`crate::indexer_scores`] (masked columns are -1e30 so their softmax is
/// ~0, and only a zero-weight teacher term pairs with them); the teacher is
/// [`pool_teacher`]'s output, and the caller DETACHES it (stage (a) trains
/// the indexer against a frozen teacher, never a self-KL loop).
pub fn distill_loss(scores: Tensor<3>, teacher: Tensor<3>, block_r: usize) -> Tensor<1> {
    let [_, t, b_n] = scores.dims();
    assert_eq!(
        teacher.dims(),
        scores.dims(),
        "teacher and indexer scores must share the complete-block space ({b_n})"
    );
    // "Only complete key blocks are included in the KL loss for each
    // query" - which is the indexer's OWN block-causal visibility (Eq.
    // 15's condition, and the same grid the scores came masked by). The
    // query's own partial block must not appear in the KL: the teacher
    // mass there is real (it was max-pooled) but the scorer is masked, so
    // an untrimmed teacher puts logp(-1e30) terms inside the loss.
    let dev = scores.device();
    let pos = Tensor::<1, burn::tensor::Int>::arange(0..t as i64, &dev)
        .unsqueeze_dim::<2>(1)
        .unsqueeze_dim::<3>(0); // [1, t, 1]
    let end = Tensor::<1, burn::tensor::Int>::arange(0..b_n as i64, &dev)
        .mul_scalar(block_r as i64)
        .add_scalar((block_r as i64) - 1)
        .reshape([1, 1, b_n]);
    let invisible = end.greater(pos); // [1, t, B]
    let teacher2 = teacher.mask_fill(invisible, 0.0f32); // zero the invisible
    let den = teacher2.clone().sum_dim(2).add_scalar(1e-20);
    let teacher = teacher2 / den; // renormalize over the VISIBLE blocks
    let logp = activation::log_softmax(scores, 2);
    let kl = (teacher.clone() * ((teacher.clone() + 1e-30).log() - logp)).sum_dim(2); // [b, t, 1]
    kl.mean_dim(2).mean_dim(1).mean_dim(0).reshape([1]) // mean over queries AND batch
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32s(x: Tensor<3>) -> Vec<f32> {
        x.into_data().bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    #[test]
    fn pool_l1_normalizes_per_query() {
        // All-zero input: stays all-zero (n / 1e-20 -> 0), no NaN.
        let dev = Device::ndarray();
        let a = Tensor::<3>::from_data(burn::tensor::TensorData::new(vec![0.0f32; 16], [1, 4, 4]), &dev);
        let v = f32s(pool_teacher(a, 2));
        assert!(v.iter().all(|&x| x.abs() < 1e-12), "all-zero stays all-zero: {v:?}");
    }

    #[test]
    fn distill_loss_is_zero_when_the_scorer_matches_the_teacher() {
        // A CAUSAL teacher (a real head-summed softmax: future tokens are
        // 0), pooled and L1'd; scores = log(a-hat) on the visible complete
        // blocks and -1e30 beyond - reproduce the teacher, so KL = 0.
        // Composition pinned: the pool's L1 over complete blocks, the
        // distill's visible-only zero+renorm, the scorer's masked softmax.
        let dev = Device::ndarray();
        let (t, r, b_n) = (4usize, 2usize, 2usize);
        let din: Vec<f32> = (0..t)
            .flat_map(|i| (0..t).map(move |j| ((j <= i) as usize as f32) / (i + 1) as f32))
            .collect();
        let teacher = pool_teacher(
            Tensor::<3>::from_data(burn::tensor::TensorData::new(din, [1, t, t]), &dev),
            r,
        );
        let av: Vec<f32> = teacher.clone().into_data().bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // scores: log(a-hat) where the block is visible, -1e30 where not.
        let mut s = vec![-1e30f32; t * b_n];
        for i in 0..t {
            for b in 0..b_n {
                let visible = b * r + r - 1 <= i;
                if visible {
                    s[i * b_n + b] = av[i * b_n + b].ln();
                }
            }
        }
        let scores =
            Tensor::<3>::from_data(burn::tensor::TensorData::new(s, [1, t, b_n]), &dev);
        let l = distill_loss(scores, teacher, r);
        let v = l.into_data().bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert!(
            v[0].abs() < 1e-5,
            "matched scorer's KL must be ~0, got {}",
            v[0]
        );
    }
}
