//! GATE 1's CUDA arm: the same dense contract against the trainer's
//! backend class (bare `cubecl` CUDA, and an autodiff wrapper so the
//! autodiff path of every op (softmax under a mask, the gather) is in the
//! comparison too).
//!
//! `required-features = ["cuda"]` — cargo skips the target without the
//! feature, it does not fail (tools/test_targets.py checks the
//! declarations); the GPU gate `vendor/dormouse-fused/tools/gpu-gate.sh` is
//! where this runs on the shared card.

use burn::tensor::Tensor;
use burn_msa::NEG;

fn bare_dev() -> burn::tensor::Device {
    burn::tensor::Device::cuda(0)
}
fn ad_dev() -> burn::tensor::Device {
    burn::tensor::Device::cuda(0).autodiff()
}

fn dense_reference(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Tensor<3> {
    let [b, t, h, hd] = q.dims();
    let mask = Tensor::<4, burn::tensor::Bool>::tril_mask([1, 1, t, t], 0, &q.device());
    let qh = q.swap_dims(1, 2).reshape([b * h, t, hd]);
    let kh = k.swap_dims(1, 2).reshape([b * h, t, hd]);
    let vh = v.swap_dims(1, 2).reshape([b * h, t, hd]);
    let scores = qh
        .matmul(kh.swap_dims(1, 2)) // [bh, t, t]
        
        .mul_scalar((hd as f64).powf(-0.5) as f32)
        .mask_fill(mask.expand([b * h, 1, t, t]).reshape([b * h, t, t]), NEG);
    burn::tensor::activation::softmax(scores, 2)
        .unsqueeze_dim::<4>(2)
        .matmul(vh.unsqueeze_dim::<4>(2))
        .reshape([b, h, t, hd])
        .swap_dims(1, 2)
        .reshape([b, t, h * hd])
}

fn body(dev: burn::tensor::Device) {
    let (b, t, hq, hd) = (1, 128, 4, 32); // the indexer-passing shape: B = 32, kb = 32
    let q = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);
    let k = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);
    let v = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);
    let picks = burn_msa::select_blocks(Tensor::<3>::full([b, t, 32], 0.0, &dev), 32);
    let sparse = burn_msa::sparse_attention(q.clone(), k.clone(), v.clone(), 4, &picks);
    let dense = dense_reference(q, k, v);
    let da = dense.into_data().bytes;
    let sa = sparse.into_data().bytes;
    let mut worst = 0f32;
    for (a, c) in da.chunks_exact(4).zip(sa.chunks_exact(4)) {
        let af = f32::from_le_bytes(a.try_into().unwrap());
        let cf = f32::from_le_bytes(c.try_into().unwrap());
        worst = worst.max((af - cf).abs());
    }
    assert!(
        worst < 2e-5,
        "CUDA sparse(all) diverges from CUDA dense by {worst}"
    );
}

#[test]
fn sparse_kb_all_equals_dense_cuda_bare() {
    body(bare_dev());
}

#[test]
fn sparse_kb_all_equals_dense_cuda_autodiff() {
    body(ad_dev());
}
