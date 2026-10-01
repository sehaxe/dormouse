//! GPU kernel check: correctness + speed vs the tensor path.
//! Run: cargo run -p burn-tsct --release --features cuda --example gpu_check
#[cfg(feature = "cuda")]
use burn::tensor::{Device, Distribution, Tensor};
#[cfg(feature = "cuda")]
use dormouse_spectral::gpu::{pack_u32, tsct_linear_cuda};

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use dormouse_spectral::gpu::{pack_u32, tsct_linear_cuda};

    fn main() {
        let dev = Device::cuda(0);
        let (m, k, n, b) = (512usize, 8usize, 4096usize, 4usize);

        // random ternary U, V and scales
        let mut u = vec![0i8; m * k];
        let mut v = vec![0i8; n * k];
        for (i, x) in u.iter_mut().enumerate() {
            *x = match i % 5 {
                0 => -1,
                1 => 1,
                _ => 0,
            };
        }
        for (i, x) in v.iter_mut().enumerate() {
            *x = match i % 3 {
                0 => -1,
                2 => 1,
                _ => 0,
            };
        }
        let s: Vec<f32> = (0..k).map(|i| 0.5 + i as f32 * 0.05).collect();
        let up = pack_u32(&u, m, k);
        let vp = pack_u32(&v, n, k);

        let x = Tensor::<2>::random([b, m], Distribution::Normal(0.0, 1.0), &dev);

        // reference: tensor path y = (x @ U) * s @ V^T
        let u_t = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(u.iter().map(|&v| v as f32).collect::<Vec<_>>(), [m, k]),
            &dev,
        );
        let v_t = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(v.iter().map(|&v| v as f32).collect::<Vec<_>>(), [n, k]),
            &dev,
        );
        let s_t = Tensor::<1>::from_data(burn::tensor::TensorData::new(s.clone(), [k]), &dev);
        let ref_out = x
            .clone()
            .matmul(u_t.clone())
            .mul(s_t.clone().unsqueeze_dims(&[0]))
            .matmul(v_t.clone().transpose());

        // CPU reference: same layout, proven correct in infer_bench
        let x_host: Vec<f32> = x.clone().into_data().to_vec().unwrap();
        let u_host: Vec<f32> = u.iter().map(|&v| v as f32).collect();
        let v_host: Vec<f32> = v.iter().map(|&v| v as f32).collect();
        let mut cpu_out = vec![0.0f32; b * n];
        for bi in 0..b {
            for j in 0..n {
                let mut acc = 0.0;
                for i in 0..k {
                    let mut zi = 0.0;
                    for mm in 0..m {
                        zi += x_host[bi * m + mm] * u_host[mm * k + i];
                    }
                    acc += zi * s[i] * v_host[j * k + i];
                }
                cpu_out[bi * n + j] = acc;
            }
        }
        let cpu_t = Tensor::<2>::from_data(burn::tensor::TensorData::new(cpu_out, [b, n]), &dev);
        // gpu kernel
        let gpu_out = tsct_linear_cuda(&x, &up, &vp, &s, m, k, n, b).expect("cuda kernel");
        let diff_cpu: f32 = (gpu_out.clone() - cpu_t.clone()).abs().max().into_scalar();
        let diff_ref: f32 = (ref_out.clone() - cpu_t.clone()).abs().max().into_scalar();
        println!("gpu vs cpu: {diff_cpu:.6}, tensor vs cpu: {diff_ref:.6}");
        assert!(diff_cpu < 1e-3, "kernel vs cpu mismatch: {diff_cpu}");
        assert!(diff_ref < 0.05, "kernel vs tensor mismatch: {diff_ref}");

        // speed: kernel vs tensor path, 100 iters
        let t0 = std::time::Instant::now();
        for _ in 0..100 {
            let _ = tsct_linear_cuda(&x, &up, &vp, &s, m, k, n, b).unwrap();
        }
        let t_gpu = t0.elapsed().as_secs_f64() / 100.0;

        let t0 = std::time::Instant::now();
        for _ in 0..100 {
            let _ = x
                .clone()
                .matmul(u_t.clone())
                .mul(s_t.clone().unsqueeze_dims(&[0]))
                .matmul(v_t.clone().transpose());
        }
        let t_ref = t0.elapsed().as_secs_f64() / 100.0;

        println!(
            "tensor path: {:.3} ms, gpu kernel: {:.3} ms, speedup: {:.1}x",
            t_ref * 1e3,
            t_gpu * 1e3,
            t_ref / t_gpu
        );
    }
}
