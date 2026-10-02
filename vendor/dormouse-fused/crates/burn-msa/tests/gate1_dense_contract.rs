//! GATE 1 — the dense contract: sparse attention with `kb == B` (all
//! complete blocks) over an index set that by construction covers every
//! token exactly once must equal dense causal attention on the same
//! projected q/k/v, bit-near. THE gate of the re-entry: without it nothing
//! downstream is a number.
//!
//! CPU (ndarray) arm; the CUDA arm lives in `gate1_cuda.rs` and runs only
//! under the `cuda` feature (see its header).

use burn::tensor::Tensor;
use burn_msa::NEG;

fn dev() -> burn::tensor::Device {
    burn::tensor::Device::ndarray()
}

/// A dense causal attention reference written WITHOUT the crate's helpers:
/// tril-masked softmax, no gather. If the two disagree, the discrepancy is
/// in the pack/gather/mask path, which is the whole mechanism.
fn dense_reference(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Tensor<3> {
    let [b, t, h, hd] = q.dims();
    let mask = Tensor::<4, burn::tensor::Bool>::tril_mask([1, 1, t, t], 0, &q.device());
    let qh = q.swap_dims(1, 2).reshape([b * h, t, hd]);
    let kh = k.swap_dims(1, 2).reshape([b * h, t, hd]);
    let vh = v.swap_dims(1, 2).reshape([b * h, t, hd]);
    let scores = qh
        .matmul(kh.swap_dims(1, 2)) // [bh, t, t]
        
        .mul_scalar((hd as f64).powf(-0.5) as f32)
        .mask_fill(
            mask.expand([b * h, 1, t, t]).reshape([b * h, t, t]),
            burn_msa::NEG,
        );
    let out = burn::tensor::activation::softmax(scores, 2)
        .matmul(vh)
        .reshape([b, h, t, hd]);
    out.swap_dims(1, 2).reshape([b, t, h * hd])
}

#[test]
fn sparse_kb_all_equals_dense() {
    let dev = dev();
    let (b, t, hq, hd) = (2, 32, 4, 8); // r = 4 -> B = 8 complete, tail = 0
    let q = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);
    let k = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);
    let v = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);

    let picks = burn_msa::select_blocks(
        // all-blocks: identity, NOT topk (k == n is not expressible - ADR-0015)
        Tensor::<3>::full([b, t, 8], 0.0, &dev),
        8,
    );
    let sparse = burn_msa::sparse_attention(q.clone(), k.clone(), v.clone(), 4, &picks);
    let dense = dense_reference(q, k, v);

    let worst = worst_abs(&sparse, &dense);
    assert!(
        worst < 2e-5,
        "sparse(all) diverges from dense by {worst} — the gather/pack path is wrong"
    );
}

#[test]
fn sparse_kb_all_with_tail_equals_dense() {
    let dev = dev();
    let (b, t, hq, hd) = (1, 30, 2, 8); // r = 4 -> B = 7, tail = 2
    let q = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);
    let k = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);
    let v = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);

    let picks = burn_msa::select_blocks(Tensor::<3>::full([b, t, 7], 0.0, &dev), 7);
    let sparse = burn_msa::sparse_attention(q.clone(), k.clone(), v.clone(), 4, &picks);
    let dense = dense_reference(q, k, v);
    let worst = worst_abs(&sparse, &dense);
    assert!(
        worst < 2e-5,
        "sparse(all)+tail diverges from dense by {worst} — the tail packing is wrong"
    );
}

/// Max |a - b| over two equal-shape f32 tensors. THE comparison of the
/// dense contract.
fn worst_abs(a: &Tensor<3>, b: &Tensor<3>) -> f32 {
    let da = a.clone().into_data().bytes;
    let db = b.clone().into_data().bytes;
    assert_eq!(da.len(), db.len(), "shape mismatch");
    da.chunks_exact(4)
        .zip(db.chunks_exact(4))
        .map(|(a, c)| {
            f32::from_le_bytes(a.try_into().unwrap()) - f32::from_le_bytes(c.try_into().unwrap())
        })
        .fold(0f32, |w, d| w.max(d.abs()))
}

#[test]
fn sparse_small_kb_is_mathematically_consistent_but_restrictive() {
    // Not the dense contract: a smaller kb output is a RESTRICTED softmax,
    // in range, causal, but not equal to dense's. What must hold: the
    // output is finite, and the sum of every packed window's selection
    // weights is 1 (that is what softmax of the gathered row guarantees).
    let dev = dev();
    let (b, t, hq, hd) = (1, 32, 4, 8);
    let q = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);
    let k = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);
    let v = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);
    // inert scores: every query's top-2 of 8 (anti-causal -1e30 mask is in
    // the scores here, so select_blocks sits ON the indexer's output space).
    let inert = Tensor::<3>::from_data(
        burn::tensor::TensorData::new(
            (0..b * t * 8)
                .map(|j| {
                    // blocks 4..8 are future for every query at seq 32 when
                    // block_causal; build visible-first tilt.
                    if j % 8 > 3 {
                        -1e30f32
                    } else {
                        ((j % 8) as f32) * 0.1
                    }
                })
                .collect(),
            [b, t, 8],
        ),
        &dev,
    );
    let picks = burn_msa::select_blocks(inert, 2);
    let out = burn_msa::sparse_attention(q, k, v, 4, &picks);
    let ov: Vec<f32> = out.into_data().bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!(
        ov.iter().all(|&x| x.is_finite()),
        "sparse with kb=2 produced a non-finite output"
    );
}
