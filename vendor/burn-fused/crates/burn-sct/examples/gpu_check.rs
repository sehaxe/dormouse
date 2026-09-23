// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
use burn::module::Param;
use burn::tensor::{Device, Tensor};
use burn_sct::qr_cuda::CudaBare;
use burn_sct::{SctConfig, SctLinear};
fn main() {
    let dev_cpu = Device::ndarray();
    let dev_gpu = Device::cuda(0);
    let (m, n, k, batch) = (4096usize, 4096usize, 128usize, 64usize);
    let lc = SctLinear::new(&SctConfig::new(m, n, k), &dev_cpu);
    let mut lg = SctLinear::new(&SctConfig::new(m, n, k), &dev_gpu);
    // same params on both
    let u: Vec<f32> = lc.u.val().clone().into_data().try_to_vec::<f32>().unwrap();
    let s: Vec<f32> = lc.s.val().clone().into_data().try_to_vec::<f32>().unwrap();
    let v: Vec<f32> = lc.v.val().clone().into_data().try_to_vec::<f32>().unwrap();
    lg.u = Param::from_tensor(
        burn::tensor::Tensor::<1>::from_floats(u.as_slice(), &dev_gpu).reshape([m, k]),
    );
    lg.s = Param::from_tensor(burn::tensor::Tensor::<1>::from_floats(
        s.as_slice(),
        &dev_gpu,
    ));
    lg.v = Param::from_tensor(
        burn::tensor::Tensor::<1>::from_floats(v.as_slice(), &dev_gpu).reshape([n, k]),
    );
    let x: Vec<f32> = (0..batch * m).map(|i| ((i as f32) * 0.001).sin()).collect();
    let xc = Tensor::<1>::from_floats(x.as_slice(), &dev_cpu).reshape([batch, m]);
    let xg = Tensor::<1>::from_floats(x.as_slice(), &dev_gpu).reshape([batch, m]);
    let yc: Vec<f32> = lc
        .forward::<burn_ndarray::NdArray>(xc)
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let yg: Vec<f32> = lg
        .forward::<CudaBare>(xg)
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let maxd: f32 = yc
        .iter()
        .zip(yg.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("forward gpu vs cpu: max_diff {maxd:.3e}");
}
