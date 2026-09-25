//! Two-pass gather micro-repro: pre.4 poisons on the second gather round.
use burn::backend::Autodiff as Ad;
use burn::prelude::*;

type CudaBare = burn_cubecl::CubeBackend;
type B = Ad<CudaBare, burn_autodiff::checkpoint::strategy::BalancedCheckpointing>;

fn pass(n: usize) {
    let device = Default::default();
    // source: [B, H, S*d] like block_scores' k
    let base: Tensor<2> = Tensor::from_floats([[1.0f32, 2.0]], &device);
    let k: Tensor<3> = base
        .repeat_dim(0, (10 * 3 * 512 * 64 / 2))
        .reshape([10, 3, 512 * 64]);
    // indices: [B, H, S, bs*d] in [0, S*d) — same construction as block_scores
    // idx values = (t % 512)*64 + 0..8, in [0, 512*64)
    let t: Tensor<1, Int> = Tensor::arange(0..(10 * 3 * 512 * 8) as i64, &device);
    let blk = (t.clone() / 8) % 512;
    let off = t.clone() % 8;
    let idx_flat: Tensor<1, Int> = blk * 64 + off;
    let idx = idx_flat.reshape([10, 3, 512, 8]);
    let _ = n;
    let g = k.gather(2, idx.reshape([10, 3, 512 * 8]));
    let s: f32 = g.sum().into_scalar();
    println!("gather pass {n}: OK sum={s}");
}

fn main() {
    pass(0);
    pass(1);
    println!("GATHER-REPRO-OK");
}
