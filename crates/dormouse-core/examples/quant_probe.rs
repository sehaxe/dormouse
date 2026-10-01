//! quant fidelity probe: does quantize_tensor(bits=4/8) actually quantize
//! orthonormal TSCT factors? Prints max/mean quantization error on U.
//! cargo run --release -p dormouse-core --features cuda --example quant_probe
use burn::tensor::Device;
use burn_spectral::SpectralLinear;

/// Same autodiff backend the trainer uses (forward_with_hidden needs an
/// AutodiffBackend; plain Cuda is not one).
type B = burn::backend::autodiff::Autodiff<
    burn_cuda::Cuda,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;

fn main() {
    let device = Device::cuda(0);
    let lin = SpectralLinear::new(768, 2048, 64, &device);
    let u = lin.u.val();
    for bits in [4usize, 8] {
        let q = burn_bitnet::quantize_tensor::<burn_cuda::Cuda>(u.clone(), bits);
        let vu: Vec<f32> = u.clone().into_data().try_to_vec().unwrap();
        let vq: Vec<f32> = q.into_data().try_to_vec().unwrap();
        let mut max_d = 0.0f32;
        let mut sum = 0.0f32;
        let mut nz = 0usize;
        for i in 0..vu.len() {
            let d = (vu[i] - vq[i]).abs();
            if d > max_d {
                max_d = d;
            }
            sum += d;
            if vq[i] != 0.0 {
                nz += 1;
            }
        }
        println!(
            "bits={bits}: max|d|={max_d:.6} mean|d|={:.8} nonzero={}/{} u_scale={:.4}",
            sum / vu.len() as f32,
            nz,
            vu.len(),
            vu.iter().map(|x| x.abs()).sum::<f32>() / vu.len() as f32
        );
    }
    // STE roundtrip used by forward_quant: w + (q - w).detach()
    let q = burn_bitnet::quantize_tensor::<burn_cuda::Cuda>(u.clone(), 4);
    let ste = u.clone() + (q - u.clone()).detach();
    let vste: Vec<f32> = ste.into_data().try_to_vec().unwrap();
    let vu: Vec<f32> = u.into_data().try_to_vec().unwrap();
    let mut max_d = 0.0f32;
    for i in 0..vu.len() {
        let d = (vu[i] - vste[i]).abs();
        if d > max_d {
            max_d = d;
        }
    }
    println!("STE roundtrip fp4: max|d|={max_d:.6}");

    // end-to-end: forward_quant vs forward on one layer
    let x = burn::tensor::Tensor::<2>::random(
        [4, 768],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let y_plain = lin.forward(x.clone());
    let mut linq = lin.clone();
    linq.set_quant(burn_spectral::QuantFormat::Fp4);
    let y_q = linq.forward_quant::<burn_cuda::Cuda>(x);
    let vp: Vec<f32> = y_plain.into_data().try_to_vec().unwrap();
    let vq: Vec<f32> = y_q.into_data().try_to_vec().unwrap();
    let md = vp
        .iter()
        .zip(&vq)
        .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
    println!("layer fwd plain-vs-quant fp4: max|d|={md:.6}");

    // full model loss delta: fp32 vs fp4 on same batch
    let cfg = dormouse_core::DormouseConfig::default();
    let m_plain = dormouse_core::DormouseModel::new(&cfg, &device);
    let mut m_quant = m_plain.clone();
    m_quant
        .loop_block
        .set_quant_all(burn_spectral::QuantFormat::Fp4);
    let ids: Vec<i64> = (0..(2 * 64)).map(|i| (i % 250) as i64).collect();
    let xb: burn::tensor::Tensor<2, burn::tensor::Int> = burn::tensor::Tensor::from_data(
        burn::tensor::TensorData::new(ids.clone(), [2, 64]),
        &device,
    );
    let yb: burn::tensor::Tensor<2, burn::tensor::Int> =
        burn::tensor::Tensor::from_data(burn::tensor::TensorData::new(ids, [2, 64]), &device);
    let (_lp, rec_p, _k, _ap) =
        m_plain.forward_with_hidden::<B>(xb.clone(), None, None, Some(yb.clone()), None);
    let l_plain: f32 = m_plain.loss::<B>(rec_p).try_into_scalar().unwrap();
    let (_lq, rec_q, _k, _aq) = m_quant.forward_with_hidden::<B>(xb, None, None, Some(yb), None);
    let l_quant: f32 = m_quant.loss::<B>(rec_q).try_into_scalar().unwrap();
    println!(
        "model loss: fp32={l_plain:.4} fp4={l_quant:.4} delta={:.4}",
        (l_plain - l_quant).abs()
    );
}
