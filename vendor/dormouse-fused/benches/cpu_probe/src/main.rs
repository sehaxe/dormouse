use burn::tensor::{Device, Distribution, Tensor};
fn main() {
    type B = burn_cubecl::CubeBackend;
    let dev = Device::cpu();
    let (b, t, nh, hd) = (2usize, 16, 4, 32);
    let x = Tensor::<4>::random([b, t, nh, hd], Distribution::Normal(0.0, 1.0), &dev);
    let (cos, sin) = dormouse_rope::precompute_freqs(hd, t, 10000.0, &dev);
    let _ = dormouse_rope::apply_rope_4d::<B>(x.clone(), cos.clone(), sin.clone());
    let half = hd / 2;
    let x1 = x.clone().slice([0..b, 0..t, 0..nh, 0..half]);
    let x2 = x.clone().slice([0..b, 0..t, 0..nh, half..hd]);
    let c = cos.clone().slice([0..t, 0..half]).reshape([1, t, 1, half]);
    let sn = sin.clone().slice([0..t, 0..half]).reshape([1, t, 1, half]);
    let ref_out = Tensor::cat(
        vec![
            x1.clone().mul(c.clone()).sub(x2.clone().mul(sn.clone())),
            x1.mul(sn).add(x2.mul(c)),
        ],
        3,
    );
    let fused_out = dormouse_rope::apply_rope_4d::<B>(x.clone(), cos, sin);
    let diff: f32 = (fused_out - ref_out).abs().max().into_scalar();
    println!(
        "cubecl-cpu rope: diff = {:.3e} {}",
        diff,
        if diff < 1e-3 { "OK" } else { "FAIL" }
    );

    // --- situ on cubecl-cpu ---
    {
        let (n, h) = (64usize, 32);
        let gu = Tensor::<2>::random([n, 2 * h], Distribution::Normal(0.0, 1.0), &dev);
        let f = dormouse_situ::situ_glu(gu.clone(), h, 1.0, 1.5);
        let g = gu.clone().slice([0..n, 0..h]);
        let u = gu.slice([0..n, h..2 * h]);
        let ref_s = g
            .clone()
            .div_scalar(1.0)
            .tanh()
            .mul(burn::tensor::activation::sigmoid(g))
            .mul(u.clone().div_scalar(1.5).tanh().mul_scalar(1.5));
        let diff: f32 = (f - ref_s).abs().max().into_scalar();
        println!(
            "cubecl-cpu situ: diff = {:.3e} {}",
            diff,
            if diff < 1e-3 { "OK" } else { "FAIL" }
        );
    }

    // --- rmsnorm fused on cubecl-cpu ---
    {
        let norm = dormouse_rmsnorm::RMSNorm::new(32, 1e-5, &dev);
        let x = Tensor::<3>::random([1, 8, 32], Distribution::Default, &dev);
        let y = norm.forward(x.clone());
        let w = norm.weight.val();
        let rms = x
            .clone()
            .powf_scalar(2.0)
            .mean_dim(2)
            .add_scalar(1e-5)
            .sqrt();
        let ref_y = (x / rms) * w.reshape([1, 1, 32]);
        let diff: f32 = (y - ref_y).abs().max().into_scalar();
        println!(
            "cubecl-cpu rmsnorm: diff = {:.3e} {}",
            diff,
            if diff < 1e-3 { "OK" } else { "FAIL" }
        );
    }

    // --- swiglu fused on cubecl-cpu ---
    {
        let gu = Tensor::<3>::random([1, 8, 64], Distribution::Default, &dev);
        let y = dormouse_swiglu::swiglu_gate(gu.clone());
        let g = gu.clone().slice([0..1, 0..8, 0..32]);
        let u = gu.slice([0..1, 0..8, 32..64]);
        let ref_y = burn::tensor::activation::silu(g).mul(u);
        let diff: f32 = (y - ref_y).abs().max().into_scalar();
        println!(
            "cubecl-cpu swiglu: diff = {:.3e} {}",
            diff,
            if diff < 1e-3 { "OK" } else { "FAIL" }
        );
    }
}
