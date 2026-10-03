//! The core sparse attention: the micro-block-mask path (the paper's own
//! tensor stage — "these indices are expanded into a microblock sparse
//! attention mask for sparse core attention") followed by ONE dense-shaped
//! softmax. The efficiency layer (a fused kernel) comes later; this tensor
//! path is the correctness reference every kernel must match, and the
//! exact form of GATE 1:
//!
//! with `picks = identity(B)` the mask is the plain causal tril, so
//! sparse(all) == dense bit-near. THE gate of the re-entry: without it
//! nothing downstream is a number.
//!
//! Shapes: `q`/`k`/`v` `[b, t, h, hd]` — ALREADY projected (the caller owns
//! its projections; a `LinearLike` in the model's case, so padding and the
//! quant seams keep one owner). Returns `[b, t, h*hd]`. No per-query loop,
//! no dynamic 4D slice (the sm_120 rule): the mask is a rank-4 Bool built
//! from the pick Int tensor by EQUALITY/comparison ops only — no bool->
//! float bridge (AGENTS.md 1.3), no gather — and the attention itself is
//! one 3D matmul chain over the head-folded batch.
//!
//! DELIVERABLE NOTE: the lane asked for "gather -> attention -> scatter".
//! The mask path IS the paper's tensor-level formulation, and the ADR-0015
//! topk->gather primitive cannot select a query-varying key set from a
//! shared source (burn's gather needs the candidate axis last with every
//! non-gathered dim equal — the shared key matrix cannot vary per query
//! without replication). Mask now; the fused kernel that gathers blocks
//! outright is stage (b)'s GPU work, and gate 1 is what it must reproduce
//! bit-near.

use burn::tensor::{Bool, Device, Int, Tensor};

pub const NEG: f32 = -1e30;

/// The micro-block sparse mask (Eq. 19): `[b, 1, t(query), t(token)]`.
/// `allowed[i, j]` = `j <= i` (causal) AND (`block(j)` is one of query i's
/// picks OR `j` is in the final incomplete block, which is ALWAYS attended).
/// Built from COMPARISONS only — no bool->float bridge (AGENTS.md 1.3), no
/// gather.
pub fn sparse_mask(
    b: usize,
    t: usize,
    block_r: usize,
    picks: &Tensor<3, Int>, // [b, t(query), kb] block ids in [0, B)
    device: &Device,
) -> Tensor<4, Bool> {
    let info = crate::blocks(t, block_r);
    let kb = info.blocks;
    let [pb, q, pk] = picks.dims();
    assert_eq!(pb, b, "picks batch {pb} vs q/kv batch {b}");
    assert_eq!(q, t, "one pick row per query: picks {q} vs seq {t}");
    assert!(
        pk <= info.blocks,
        "KB selection {pk} exceeds complete block count {} - raise msa_kb below it,",
        info.blocks
    );

    // Token -> block id.
    let j = Tensor::<1, Int>::arange(0..t as i64, device);
    let j_block = j.clone().div_scalar(block_r as i64); // [t] in [0, B)

    // Selected-block membership: equality per (b, q, slot, token), ANY
    // over the slots -> sel [b, q, t]. NO gather consumes the picks - the
    // ADR-0015 topk->gather primitive is not needed for the tensor path.
    let picks4 = picks
        .clone()
        .unsqueeze_dim::<4>(3)
        .expand([b, t, pk.max(1), t]);
    let jb = j_block
        .clone()
        .reshape([1, 1, 1, t])
        .expand([b, t, pk.max(1), t]);
    let sel = picks4.equal(jb).any_dim(2).reshape([b, t, t]);

    // The final incomplete block's tokens are ALWAYS members (Eq. 19).
    let tail = j
        .clone()
        .greater_equal_scalar((kb * block_r) as i64) // tokens >= B*r = the
                                                    // final incomplete block's first token
        .reshape([1, 1, t])
        .repeat_dim(0, b)
        .repeat_dim(1, t); // [b, t, t]
    let member = sel.bool_or(tail); // [b, q, t]

    // mask_fill fills where TRUE, so the mask IS the EXCLUDED set: the
    // future grid (tril_mask(0) is true where j > i - byteflow's comment
    // pins it) plus everything the selection and tail didn't admit.
    let future = Tensor::<4, Bool>::tril_mask([1, 1, t, t], 0, device).repeat_dim(0, b);
    let member4 = member.unsqueeze_dim::<4>(1); // [b, 1, q, t]
    member4.bool_not().bool_or(future)
}

/// Sparse attention: `q . k` -> the mask above -> ONE softmax -> weighted
/// sum -> the same out-layout the dense path returns.
pub fn sparse_attention(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    block_r: usize,
    picks: &Tensor<3, Int>,
) -> Tensor<3> {
    let [b, t, h, hd] = q.dims();
    assert_eq!(k.dims(), q.dims(), "k must carry q's projector layout");
    assert_eq!(v.dims(), q.dims(), "v must carry q's projector layout");
    let dev = q.device();

    // Fold the heads into the batch axis: every rank below stays 3.
    let qh = q.swap_dims(1, 2).reshape([b * h, t, hd]); // [bh, t, hd]
    let kh = k.swap_dims(1, 2).reshape([b * h, t, hd]);
    let vh = v.swap_dims(1, 2).reshape([b * h, t, hd]);

    let scores = qh
        .matmul(kh.swap_dims(1, 2)) // [bh, t, t]
        .mul_scalar((hd as f64).powf(-0.5) as f32);
    let mask = sparse_mask(b, t, block_r, picks, &dev); // [b, 1, t, t]
    let scores = scores
        .reshape([b, h, t, t])
        .mask_fill(mask, NEG)
        .reshape([b * h, t, t]);
    let mixed = burn::tensor::activation::softmax(scores, 2)
        .matmul(vh) // [bh, t, hd]
        .reshape([b, h, t, hd])
        .swap_dims(1, 2) // [b, t, h, hd]
        .reshape([b, t, h * hd]);
    mixed
}

/// The dense teacher for stage (a): plain causal attention over the SAME
/// projected q/k/v, plus its head-summed attention distribution (Eq. 17's
/// `a_i` raw material). Spare the fused-QSA caveat: this is the path the
/// TRAINED backsparse replaces, run only during distillation — running it
/// in stage (b) would price the whole mechanism twice.
pub fn dense_attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> (Tensor<3>, Tensor<3>) {
    let [b, t, h, hd] = q.dims();
    assert_eq!(k.dims(), q.dims(), "k must carry q's projector layout");
    assert_eq!(v.dims(), q.dims(), "v must carry q's projector layout");
    let qh = q.swap_dims(1, 2).reshape([b * h, t, hd]);
    let kh = k.swap_dims(1, 2).reshape([b * h, t, hd]);
    let vh = v.swap_dims(1, 2).reshape([b * h, t, hd]);

    let mask = Tensor::<4, Bool>::tril_mask([1, 1, t, t], 0, &qh.device());
    let scores = qh
        .matmul(kh.swap_dims(1, 2)) // [bh, t, t]
        .mul_scalar((hd as f64).powf(-0.5) as f32)
        .mask_fill(mask.expand([b * h, 1, t, t]).reshape([b * h, t, t]), NEG);
    let soft = burn::tensor::activation::softmax(scores, 2); // [bh, t, t]
    let out = soft.clone()
        .matmul(vh) // [bh, t, hd]
        .reshape([b, h, t, hd])
        .swap_dims(1, 2)
        .reshape([b, t, h * hd]);
    let dist = soft.reshape([b, h, t, t]).sum_dim(1).reshape([b, t, t]);
    (out, dist)
}
