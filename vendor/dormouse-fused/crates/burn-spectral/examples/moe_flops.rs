//! FLOPs accounting for the spectral MoE: router + experts vs dense FFN.
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
//! Run: cargo run -p burn-spectral --release --example moe_flops
use burn_spectral::SpectralMoE;

fn main() {
    let (m, n) = (512usize, 4096usize);
    let dev = burn::tensor::Device::ndarray();

    // dense FFN
    let dense_flops = m * n;

    for (c, e, k) in [
        (128usize, 32usize, 2usize),
        (64, 32, 2),
        (32, 16, 2),
        (128, 64, 4),
    ] {
        let moe = SpectralMoE::new(m, n, c, e, k, 4, &dev);
        // MACs, not params: router is three dense bias-free projections;
        // experts are k rank-4 patterns, so k * rank * (m + n) MACs per token
        let p = moe.proj_dim;
        let router = m * p + p * c + p * e;
        let expert_flops = k * moe.rank * (m + n);
        let total = router + expert_flops;
        assert_eq!(moe.flops(), total, "moe.flops() must match the split");
        let masters = moe.param_count();
        println!(
            "clusters={c:3} per={e:2} top={k}  experts={:5}  FLOPs: router={router:6} + exp={expert_flops:6} = {total:7}  ({:.0}x fewer vs dense)  masters={} ({:.0} MB VRAM, fp32x3)",
            moe.num_experts(),
            dense_flops as f64 / total as f64,
            masters,
            masters as f64 * 12.0 / 1e6
        );
    }
    println!("dense FFN FLOPs: {dense_flops}");
}
