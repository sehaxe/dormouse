//! GATE 1 on the trainer's CPU backend: sparse attention over ALL complete
//! blocks must equal dense causal attention bit-near, on `Device::flex()`
//! (the trainer's CPU). The ndarray arm lives in
//! `vendor/dormouse-fused/crates/burn-msa/tests/gate1_dense_contract.rs`.

use burn::tensor::Tensor;

fn dev() -> burn::tensor::Device {
    burn::tensor::Device::flex()
}

fn dense_reference(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Tensor<3> {
    let [b, t, h, hd] = q.dims();
    let mask = Tensor::<4, burn::tensor::Bool>::tril_mask([1, 1, t, t], 0, &q.device());
    let qh = q.swap_dims(1, 2).reshape([b * h, t, hd]);
    let kh = k.swap_dims(1, 2).reshape([b * h, t, hd]);
    let vh = v.swap_dims(1, 2).reshape([b * h, t, hd]);
    let scores = qh
        .matmul(kh.swap_dims(1, 2))
        .mul_scalar((hd as f64).powf(-0.5) as f32)
        .mask_fill(
            mask.expand([b * h, 1, t, t]).reshape([b * h, t, t]),
            burn_msa::NEG,
        );
    burn::tensor::activation::softmax(scores, 2)
        .matmul(vh)
        .reshape([b, h, t, hd])
        .swap_dims(1, 2)
        .reshape([b, t, h * hd])
}

fn worst_abs(a: &Tensor<3>, b: &Tensor<3>) -> f32 {
    let da = a.clone().into_data().bytes;
    let db = b.clone().into_data().bytes;
    da.chunks_exact(4)
        .zip(db.chunks_exact(4))
        .map(|(a, c)| f32::from_le_bytes(a.try_into().unwrap()) - f32::from_le_bytes(c.try_into().unwrap()))
        .fold(0f32, |w, d| w.max(d.abs()))
}

#[test]
fn sparse_kb_all_equals_dense_flex() {
    let dev = dev();
    let (b, t, hq, hd) = (2, 32, 4, 8);
    let q = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);
    let k = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);
    let v = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);
    let picks = burn_msa::select_blocks(Tensor::<3>::full([b, t, 8], 0.0, &dev), 8);
    let sparse = burn_msa::sparse_attention(q.clone(), k.clone(), v.clone(), 4, &picks);
    let dense = dense_reference(q, k, v);
    let worst = worst_abs(&sparse, &dense);
    assert!(worst < 2e-5, "sparse(all) diverges from dense by {worst} on flex");
}

#[test]
fn sparse_kb_all_with_tail_equals_dense_flex() {
    let dev = dev();
    let (b, t, hq, hd) = (1, 30, 2, 8);
    let q = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);
    let k = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);
    let v = Tensor::<4>::random([b, t, hq, hd], burn::tensor::Distribution::Default, &dev);
    let picks = burn_msa::select_blocks(Tensor::<3>::full([b, t, 7], 0.0, &dev), 7);
    let sparse = burn_msa::sparse_attention(q.clone(), k.clone(), v.clone(), 4, &picks);
    let dense = dense_reference(q, k, v);
    let worst = worst_abs(&sparse, &dense);
    assert!(worst < 2e-5, "sparse(all)+tail diverges from dense by {worst} on flex");
}
