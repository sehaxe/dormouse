// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
use burn::tensor::{Device, Distribution, Tensor};
use dormouse_sct::qr_cuda::CudaBare;
use dormouse_sct::SctLinear;
fn main() {
    let dev_cpu = Device::ndarray();
    let dev_gpu = Device::cuda(0);
    let (n, m, k) = (512usize, 256usize, 32usize);
    let w = Tensor::<2>::random([n, m], Distribution::Normal(0.0, 1.0), &dev_cpu);
    let lc = SctLinear::from_dense::<burn_ndarray::NdArray>(w.clone(), k);
    let wg = w.clone().to_device(&dev_gpu);
    let lg = SctLinear::from_dense_with_iters::<CudaBare>(wg, k, 60);
    let uc: Vec<f32> = lc.u.val().clone().into_data().try_to_vec::<f32>().unwrap();
    let ug: Vec<f32> = lg.u.val().clone().into_data().try_to_vec::<f32>().unwrap();
    let vc: Vec<f32> = lc.v.val().clone().into_data().try_to_vec::<f32>().unwrap();
    let vg: Vec<f32> = lg.v.val().clone().into_data().try_to_vec::<f32>().unwrap();
    let sc: Vec<f32> = lc.s.val().clone().into_data().try_to_vec::<f32>().unwrap();
    let sg: Vec<f32> = lg.s.val().clone().into_data().try_to_vec::<f32>().unwrap();
    let du: f32 = uc
        .iter()
        .zip(ug.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let dv: f32 = vc
        .iter()
        .zip(vg.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let ds: f32 = sc
        .iter()
        .zip(sg.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!(
        "u diff {du:.3e} v diff {dv:.3e} s diff {ds:.3e} lens u {} v {}",
        uc.len(),
        vc.len()
    );
    println!("s cpu {:?}", &sc[0..3]);
    println!("s gpu {:?}", &sg[0..3]);
}
