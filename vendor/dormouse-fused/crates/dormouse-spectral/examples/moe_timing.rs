//! Timing: SpectralMoE forward/backward on CUDA.
//! Run: cargo run -p dormouse-spectral --features cuda --example moe_timing
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::{activation, Device, Distribution, Tensor};
use dormouse_spectral::SpectralMoE;

fn main() {
    let dev = Device::cuda(0).autodiff();
    let m: SpectralMoE = SpectralMoE::new(64, 256, 8, 32, 2, 4, &dev);
    let x = Tensor::<2>::random([32, 64], Distribution::Normal(0.0, 1.0), &dev);
    let t = std::time::Instant::now();
    let y = m.forward(x.clone());
    println!("forward: {:.3}s", t.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    let loss = y.powf_scalar(2.0).sum();
    let lv: f32 = loss.clone().into_scalar();
    println!("loss+sync: {:.3}s ({lv})", t.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    let grads = loss.backward();
    println!("backward: {:.3}s", t.elapsed().as_secs_f64());
    let _ = m.u.grad(&grads).unwrap();
    let mut opt = AdamWConfig::new().init();
    let grads = GradientsParams::from_grads(grads, &m);
    let t = std::time::Instant::now();
    // `ModuleOptimizer::step` takes `impl Into<ModuleLearningRate>`, so a bare
    // `1e-3.into()` leaves `Self` unconstrained (E0283). The lr is a bare f64
    // (`LearningRate` is a type alias for it), so pass it straight.
    let m2 = opt.step(1e-3, m.clone(), grads);
    println!("opt.step: {:.3}s", t.elapsed().as_secs_f64());
    let _ = m2;
    println!("ALL OK");
}
