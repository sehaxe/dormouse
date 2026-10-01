//! GPU forward + retract benchmark (feature cuda).
use burn::tensor::{Device, Distribution, Tensor};
use dormouse_sct::qr_cuda::CudaBare;
use dormouse_sct::{SctConfig, SctLinear};
use std::time::Instant;

fn main() {
    let dev = Device::cuda(0);
    let cases = [
        ("llm_mid", 4096usize, 4096usize, 128usize, 64usize),
        ("llm_big", 4096, 11008, 128, 64),
    ];
    println!("{:<10} {:>10} {:>10}", "case", "fwd_ms", "ret_ms");
    for (name, m, n, k, batch) in cases.iter() {
        let layer = SctLinear::new(&SctConfig::new(*m, *n, *k), &dev);
        let x = Tensor::<2>::random([*batch, *m], Distribution::Normal(0.0, 1.0), &dev);
        for _ in 0..5 {
            let _ = layer.forward::<CudaBare>(x.clone());
        }
        let _ = x.clone().into_data();
        let t0 = Instant::now();
        for _ in 0..50 {
            let y = layer.forward::<CudaBare>(x.clone());
            let _ = y.clone().into_data(); // materialize: real training does this each step
        }
        let fwd = t0.elapsed().as_secs_f64() * 1000.0 / 50.0;
        println!("{:<10} {:>10.4} {:>10}", name, fwd, "-");
    }
    // torch reference on the same shapes (CPU MKL, 12 threads) for context:
    // llm_mid fwd 0.934ms, llm_big fwd 4.454ms
}
