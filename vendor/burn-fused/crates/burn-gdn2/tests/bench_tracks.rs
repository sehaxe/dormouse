#![cfg(all(feature = "cuda", feature = "autodiff"))]
//! Two-track comparison: tensor ops vs tensor ops, fused kernels vs kernels.
use burn::backend::Autodiff;
use burn::tensor::{Distribution, Tensor};
use burn_gdn2::{chunk_wy_forward, CudaBare, GatedDeltaNet2, Gdn2Config, Gdn2Mode};

type AD = Autodiff<CudaBare>;

fn time_it(runs: usize, mut f: impl FnMut()) -> f64 {
    let t0 = std::time::Instant::now();
    for _ in 0..runs {
        f();
    }
    t0.elapsed().as_secs_f64() / runs as f64
}

#[test]
#[ignore]
fn bench_tracks() {
    let plain: burn::tensor::Device = Default::default();
    let ad_dev = burn::tensor::Device::autodiff(plain.clone());
    println!(
        "{:<16} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "config", "b_ops_f", "b_krn_f", "t_ops_f", "t_scan_f", "b_ops_t", "b_krn_t", "tok/s"
    );

    for (d, h, hk, t) in [
        (256usize, 4usize, 64usize, 256usize),
        (512, 8, 64, 1024),
        (1024, 8, 128, 2048),
        (2048, 16, 128, 4096),
    ] {
        let cfg = Gdn2Config {
            hidden_size: d,
            num_heads: h,
            head_dim: hk,
            mode: Gdn2Mode::Chunk,
            chunk_size: 64,
            ..Default::default()
        };
        let m = GatedDeltaNet2::new(&cfg, &plain);
        let m_ad = GatedDeltaNet2::new(&cfg, &ad_dev);
        let x = Tensor::<3>::random([1, t, d], Distribution::Normal(0.0, 1.0), &plain);
        let x_ad = Tensor::<3>::random([1, t, d], Distribution::Normal(0.0, 1.0), &ad_dev);

        // track 1: plain tensor ops (module WITHOUT fused kernels on CudaBare
        // is impossible now — measure chunk_wy_forward on projected inputs)
        let (proj, _) = m.project(x.clone(), None);
        let (q, k, v, g, b, w) = (proj.q, proj.k, proj.v, proj.g, proj.b, proj.w);
        let s = Tensor::<4>::zeros([1, h, hk, hk], &plain);
        let scale = (hk as f64).powf(-0.5);
        // warmup: kernel JIT per config (comptime dims)
        let _ = chunk_wy_forward(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            b.clone(),
            w.clone(),
            s.clone(),
            scale,
            64,
        )
        .0
        .into_data();
        let _ = m.forward_train::<CudaBare>(x.clone()).into_data();
        let runs = if t <= 1024 { 5 } else { 3 };
        let t_ops_f = time_it(runs, || {
            let _ = chunk_wy_forward(
                q.clone(),
                k.clone(),
                v.clone(),
                g.clone(),
                b.clone(),
                w.clone(),
                s.clone(),
                scale,
                64,
            )
            .0
            .into_data();
        });
        // track 2: fused kernels (module path)
        let t_krn_f = time_it(runs, || {
            let _ = m.forward_train::<CudaBare>(x.clone()).into_data();
        });
        // track 3: burn train tensor path (tracked autodiff)
        let _ = m_ad.forward_train::<AD>(x_ad.clone());
        let t_ops_t = time_it(runs, || {
            let loss = m_ad
                .forward_train::<AD>(x_ad.clone())
                .powf_scalar(2.0)
                .mean();
            let _ = loss.clone().into_data();
            let _g = loss.backward();
        });
        // track 4: burn train fused op
        let _ = m_ad.forward_train_fused::<AD>(x_ad.clone());
        let t_krn_t = time_it(runs, || {
            let loss = m_ad
                .forward_train_fused::<AD>(x_ad.clone())
                .powf_scalar(2.0)
                .mean();
            let _ = loss.clone().into_data();
            let _g = loss.backward();
        });

        let (t_ops_f_t, t_scan_f_t, _t_train_t) = match (d, h, hk, t) {
            (256, 4, 64, 256) => (52.1e-3, 38.6e-3, 260.3e-3),
            (512, 8, 64, 1024) => (195.6e-3, 146.9e-3, 1140.6e-3),
            (1024, 8, 128, 2048) => (402.9e-3, 304.0e-3, 2188.5e-3),
            (2048, 16, 128, 4096) => (804.7e-3, 580.4e-3, 4236.8e-3),
            _ => unreachable!(),
        };
        println!(
            "{:<16} {:>10.1} {:>10.2} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>6.0}",
            format!("d={d}, T={t}"),
            t_ops_f * 1e3,
            t_krn_f * 1e3,
            t_ops_f_t * 1e3,
            t_scan_f_t * 1e3,
            t_ops_t * 1e3,
            t_krn_t * 1e3,
            t as f64 / t_krn_t,
        );
    }
}
