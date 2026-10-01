//! GPU from_dense: accuracy vs CPU path + speed.
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
use burn::tensor::{Device, Distribution, Tensor};
use dormouse_sct::qr_cuda::CudaBare;
use dormouse_sct::SctLinear;
fn main() {
    let dev_cpu = Device::ndarray();
    let dev_gpu = Device::cuda(0);
    // medium size first (CPU path is slow at llm scale)
    let (n, m, k) = (1024usize, 512usize, 32usize);
    let w = Tensor::<2>::random([n, m], Distribution::Normal(0.0, 1.0), &dev_cpu);
    let wg = w.clone().to_device(&dev_gpu);
    let lc = SctLinear::from_dense::<burn_ndarray::NdArray>(w.clone(), k);
    let lg = SctLinear::from_dense_with_iters::<CudaBare>(wg, k, 60);
    // compare reconstructions
    let rc =
        lc.v.val()
            .clone()
            .mul(lc.s.val().clone().unsqueeze_dims(&[0]))
            .matmul(lc.u.val().clone().transpose());
    let rg =
        lg.v.val()
            .clone()
            .mul(lg.s.val().clone().unsqueeze_dims(&[0]))
            .matmul(lg.u.val().clone().transpose());
    let vc: Vec<f32> = rc.into_data().try_to_vec::<f32>().unwrap();
    let vg: Vec<f32> = rg.into_data().try_to_vec::<f32>().unwrap();
    let maxd: f32 = vc
        .iter()
        .zip(vg.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("recon max_diff: {maxd:.6e} lens {} {}", vc.len(), vg.len());
    println!(
        "gpu rank {} cpu rank {} gpu s[0..4] {:?} cpu s[0..4] {:?}",
        lg.rank,
        lc.rank,
        &lg.s.val().clone().into_data().try_to_vec::<f32>().unwrap()[0..4],
        &lc.s.val().clone().into_data().try_to_vec::<f32>().unwrap()[0..4]
    );
    let vw: Vec<f32> = w.clone().into_data().try_to_vec::<f32>().unwrap();
    let e_cpu: f32 = vc
        .iter()
        .zip(vw.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let e_gpu: f32 = vg
        .iter()
        .zip(vw.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("truncation err: cpu {e_cpu:.3e} gpu {e_gpu:.3e}");
    // speed at llm-ish
    let (n2, m2, k2) = (2048usize, 2048usize, 128usize);
    let w2 = Tensor::<2>::random([n2, m2], Distribution::Normal(0.0, 1.0), &dev_gpu);
    let t0 = std::time::Instant::now();
    let lg2 = SctLinear::from_dense::<CudaBare>(w2, k2);
    println!(
        "from_dense gpu [{n2}x{m2} k={k2}]: {:.3} s",
        t0.elapsed().as_secs_f64()
    );
    let _ = lg2;
}
