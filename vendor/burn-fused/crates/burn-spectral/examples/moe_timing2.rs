//! Per-op timing inside SpectralMoE::forward on CUDA.
//! Run: cargo run --release -p burn-spectral --features cuda --example moe_timing2
use burn::tensor::{activation, Device, Distribution, Int, Tensor};
use burn_spectral::SpectralMoE;

fn main() {
    let dev = Device::cuda(0).autodiff();
    let m: SpectralMoE = SpectralMoE::new(64, 256, 8, 32, 2, 4, &dev);
    let x = Tensor::<2>::random([32, 64], Distribution::Normal(0.0, 1.0), &dev);

    let t = std::time::Instant::now();
    let (c_logits, e_logits) = m.router_logits(x.clone());
    let v: f32 = c_logits.clone().sum().into_scalar();
    println!("router_logits: {:.3}s ({v})", t.elapsed().as_secs_f64());

    let t = std::time::Instant::now();
    let c_idx = c_logits.clone().argmax(1);
    let v: f32 = c_idx.clone().float().sum().into_scalar();
    println!("argmax: {:.3}s ({v})", t.elapsed().as_secs_f64());

    let t = std::time::Instant::now();
    let pos = m.topk_indices(e_logits.clone());
    let v: f32 = pos.clone().float().sum().into_scalar();
    println!("topk_indices: {:.3}s ({v})", t.elapsed().as_secs_f64());

    let t = std::time::Instant::now();
    let u_t = burn_spectral::ste_ternary(m.u.val());
    let vv: f32 = u_t.clone().sum().into_scalar();
    println!(
        "ste_ternary u [64,1024]: {:.3}s ({vv})",
        t.elapsed().as_secs_f64()
    );

    let k = m.top_k;
    let r = m.rank;
    let idx = c_idx
        .expand([32, k])
        .mul_scalar(m.experts_per_cluster as i64)
        .add(pos)
        .unsqueeze_dim::<3>(2)
        .mul_scalar(r as i64);
    let ar = Tensor::<1, Int>::arange(0..(r as i64), &x.device());
    let idx_r = idx
        .add(
            ar.unsqueeze_dim::<2>(0)
                .unsqueeze_dim::<3>(1)
                .expand([32, k, r]),
        )
        .reshape([32, k * r]);
    let vv: f32 = idx_r.clone().float().sum().into_scalar();
    println!("idx_r build: {:.3}s ({vv})", t.elapsed().as_secs_f64());

    let t = std::time::Instant::now();
    let u_g = u_t
        .transpose()
        .gather(
            0,
            idx_r
                .clone()
                .reshape([32 * k * r, 1])
                .expand([32 * k * r, 64]),
        )
        .reshape([32, k * r, 64]);
    let vv: f32 = u_g.clone().sum().into_scalar();
    println!("gather u: {:.3}s ({vv})", t.elapsed().as_secs_f64());

    let t = std::time::Instant::now();
    let proj = (x.clone().unsqueeze_dim::<3>(1) * u_g)
        .sum_dim(2)
        .squeeze_dim::<2>(2);
    let vv: f32 = proj.clone().sum().into_scalar();
    println!("proj: {:.3}s ({vv})", t.elapsed().as_secs_f64());
    println!("ALL OK");
}
