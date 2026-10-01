use burn::tensor::{Device, Distribution, Tensor};
use std::time::Instant;
fn main() {
    let dev = Device::cuda(0);
    // GEMM1: [64x4096] @ [4096x128]
    let x = Tensor::<2>::random([64, 4096], Distribution::Normal(0.0, 1.0), &dev);
    let u = Tensor::<2>::random([4096, 128], Distribution::Normal(0.0, 1.0), &dev);
    for _ in 0..5 {
        let y = x.clone().matmul(u.clone());
        let _ = y.clone().into_data();
    }
    let t0 = Instant::now();
    for _ in 0..50 {
        let y = x.clone().matmul(u.clone());
        let _ = y.clone().into_data();
    }
    println!(
        "gemm1 [64x4096]@[4096x128]: {:.4} ms",
        t0.elapsed().as_secs_f64() * 1000.0 / 50.0
    );
    // GEMM2: [64x128] @ [128x4096]
    let t = Tensor::<2>::random([64, 128], Distribution::Normal(0.0, 1.0), &dev);
    let v = Tensor::<2>::random([128, 4096], Distribution::Normal(0.0, 1.0), &dev);
    for _ in 0..5 {
        let y = t.clone().matmul(v.clone());
        let _ = y.clone().into_data();
    }
    let t0 = Instant::now();
    for _ in 0..50 {
        let y = t.clone().matmul(v.clone());
        let _ = y.clone().into_data();
    }
    println!(
        "gemm2 [64x128]@[128x4096]: {:.4} ms",
        t0.elapsed().as_secs_f64() * 1000.0 / 50.0
    );
    // GEMM3: [64x128] @ [128x11008]
    let v2 = Tensor::<2>::random([128, 11008], Distribution::Normal(0.0, 1.0), &dev);
    for _ in 0..5 {
        let y = t.clone().matmul(v2.clone());
        let _ = y.clone().into_data();
    }
    let t0 = Instant::now();
    for _ in 0..50 {
        let y = t.clone().matmul(v2.clone());
        let _ = y.clone().into_data();
    }
    println!(
        "gemm3 [64x128]@[128x11008]: {:.4} ms",
        t0.elapsed().as_secs_f64() * 1000.0 / 50.0
    );
}
