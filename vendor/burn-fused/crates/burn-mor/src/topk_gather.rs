//! The top-k-indices -> gather sequence, as one primitive.
//!
//! Every sparse-attention block indexer, product-key lookup and mixture-of-
//! recursions router is this function: score the `n` candidates along an axis,
//! take the `k` best, and pull the matching rows out of a value tensor. It was
//! never available as a working primitive because cubecl's `argtopk` emitted
//! out-of-range indices for masked score rows, and a `gather` with one of those
//! reads out of bounds (`cuEventCreate 700` / `CUDA_ERROR_ILLEGAL_ADDRESS`).
//! ADR-0012 and ADR-0014 cut a whole attention arm over it; ADR-0013 rejected
//! MoR over it.
//!
//! The index kernel is fixed in the vendored `cubek-reduce` (ADR-0015), so the
//! guarantee this module makes is the one that was missing: **every index
//! `argtopk` hands back is in `[0, n)`**, whatever the scores contain.

use burn::tensor::{Int, Tensor};

/// Indices of the `k` largest elements along `axis` of `scores`, plus the rows
/// of `values` they select.
///
/// `scores [b, q, n]` and `values [b, q, n, m]` give `[b, q, k, m]`: `n`
/// candidate columns, `m` features per column, `k` of them kept. Both are
/// indexed by the candidate, so a caller lays out its key/value tensors the way
/// it scores them and picks from them in one step.
///
/// A row of `scores` that holds fewer than `k` values above `-inf` still returns
/// `k` in-range indices — the surplus repeats an index rather than pointing past
/// the axis, so the gather cannot fault. Callers that need `k` *distinct* picks
/// must therefore pad their score rows instead of masking with `-inf`; that is
/// the only remaining sharp edge, and it is a property of "the k largest of n",
/// not of the kernel.
///
/// `k` must be smaller than `n` — burn's `argtopk` asserts it, and a `k == n`
/// call would panic there. The clamp below is only so a caller that ignores that
/// gets a working selection instead of a panic from inside a kernel launch.
///
/// Ties are not ordered: the `k`-th largest is well defined, which of `k`
/// elements share that value is not. Every caller here consumes the picked set
/// (attention, a lookup, a routed batch), never its order.
pub fn topk_gather(scores: Tensor<3>, values: Tensor<4>, k: usize) -> Tensor<4> {
    let [b, q, n] = scores.dims();
    let [vb, vq, vn, m] = values.dims();
    assert_eq!(
        (vb, vq, vn),
        (b, q, n),
        "scores and values must agree on the candidate axis: {n} scores, {vn} value rows"
    );
    let k = k.min(n.saturating_sub(1));
    let idx = scores
        .argtopk(k, 2)
        .unsqueeze_dim::<4>(3)
        .expand([b, q, k, m]);
    values.gather(2, idx)
}

/// Indices of the `k` largest elements along the last axis of `scores`,
/// `[b, q, n]` -> `[b, q, k]`. The index half of [`topk_gather`], for callers
/// that gather from more than one tensor with the same picks. Same `k < n`
/// contract.
pub fn topk_indices_last(scores: Tensor<3>, k: usize) -> Tensor<3, Int> {
    let [_, _, n] = scores.dims();
    scores.argtopk(k.min(n.saturating_sub(1)), 2)
}
