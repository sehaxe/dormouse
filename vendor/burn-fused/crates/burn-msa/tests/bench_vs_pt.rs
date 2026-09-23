#![cfg(feature = "cuda")]
#![allow(missing_docs, non_snake_case, dead_code, unused_imports)]

use burn::tensor::{Distribution, Tensor};
use burn_cubecl::CubeBackend;
use burn_msa::{MsaConfig, MsaModule};
use std::time::Instant;

type B = CubeBackend<cubecl::cuda::CudaRuntime>;

fn time_it(runs: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..(runs.min(5)) {
        f();
    }
    let start = Instant::now();
    for _ in 0..runs {
        f();
    }
    start.elapsed().as_secs_f64() / runs as f64
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn bench_cuda() {
    let dev = Default::default();

    #[allow(clippy::type_complexity)] // bench config table
    let configs: &[(
        &str,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    )] = &[
        ("tiny", 64, 2, 1, 32, 16, 16, 4, 1, 1, 64),
        ("small", 128, 4, 2, 32, 16, 32, 4, 1, 1, 256),
        ("small_b", 128, 4, 2, 32, 16, 32, 4, 4, 1, 256),
        ("med", 256, 8, 2, 32, 32, 64, 4, 1, 256, 1024),
        ("med_b", 256, 8, 2, 32, 32, 64, 4, 4, 256, 1024),
        ("large", 512, 8, 2, 64, 64, 64, 8, 1, 256, 4096),
        ("large_b", 512, 8, 2, 64, 64, 64, 8, 4, 256, 4096),
        ("xl", 768, 12, 4, 64, 64, 128, 16, 1, 512, 8192),
        ("xl_b", 768, 12, 4, 64, 64, 128, 16, 2, 512, 8192),
        ("xxl", 1024, 16, 4, 64, 64, 128, 16, 1, 512, 16384),
        ("decode_4k", 512, 8, 2, 64, 64, 64, 8, 1, 1, 4096),
        ("decode_16k", 512, 8, 2, 64, 64, 64, 8, 1, 1, 16384),
        ("decode_64k", 512, 8, 2, 64, 64, 64, 8, 1, 1, 65536),
        ("prefill_8k", 512, 8, 2, 64, 64, 64, 8, 1, 512, 8192),
        ("prefill_16k", 512, 8, 2, 64, 64, 64, 8, 1, 512, 16384),
        ("gqa_8_1", 512, 8, 1, 64, 64, 64, 8, 1, 256, 4096),
        ("gqa_16_2", 512, 16, 2, 64, 64, 64, 8, 1, 256, 4096),
        ("gqa_16_1", 512, 16, 1, 64, 64, 64, 8, 1, 256, 4096),
        ("bs_32", 512, 8, 2, 64, 64, 32, 16, 1, 256, 4096),
        ("bs_128", 512, 8, 2, 64, 64, 128, 8, 1, 256, 4096),
        ("bs_256", 512, 8, 2, 64, 64, 256, 4, 1, 256, 4096),
    ];

    println!("\n{:=^100}", "");
    println!("  burn-msa CUDA BENCHMARK (optimized) - RTX 5060 Ti");
    println!("{:=^100}", "");
    println!(
        "{:<16} {:>9} {:>9} {:>9} {:>9} {:>8} {:>6} {:>9}",
        "config", "idx_us", "full_us", "attn_us", "dense_us", "sp/de", "B*Sq", "tok/s"
    );

    for &(name, dm, nq, nk, dh, di, bs, tk, batch, Sq, Sk) in configs {
        let cfg = MsaConfig {
            d_model: dm,
            n_heads_q: nq,
            n_heads_kv: nk,
            d_head: dh,
            d_idx: di,
            block_size: bs,
            topk: tk,
            causal: false,
            force_local_block: false,
            use_rope: false,
            rope_base: 10000.0,
            rope_max_seq_len: 32768,
            use_kl_loss: false,
            warmup_steps: 0,
            kl_coeff: 0.0,
            gradient_detach: true,
        };
        let runs = if batch * Sq <= 256 {
            100
        } else if batch * Sq <= 4096 {
            50
        } else {
            10
        };

        // 1. Index+topk
        let m1 = MsaModule::new(&cfg, &dev);
        let q1 = Tensor::<3>::random([batch, Sq, dm], Distribution::Normal(0.0, 1.0), &dev);
        let kv1 = Tensor::<3>::random([batch, Sk, dm], Distribution::Normal(0.0, 1.0), &dev);
        let di_c = di;
        let bs_c = bs;
        let tk_c = tk;
        let nk_c = nk;
        let t_idx = time_it(runs, move || {
            let idx_i = q1.clone().detach();
            let kv_i = kv1.clone().detach();
            let (qi, ki) = m1.index_branch.forward(idx_i, kv_i, nk_c, di_c);
            let bscores =
                m1.index_branch
                    .compute_block_scores(qi, ki, bs_c, (di_c as f64).sqrt(), false);
            let _ = burn_msa::TopKSelector::new(tk_c, bs_c, false).select::<B>(bscores);
        });

        // 2. Full sparse
        let m2 = MsaModule::new(&cfg, &dev);
        let q2 = Tensor::<3>::random([batch, Sq, dm], Distribution::Normal(0.0, 1.0), &dev);
        let kv2 = Tensor::<3>::random([batch, Sk, dm], Distribution::Normal(0.0, 1.0), &dev);
        let t_full = time_it(runs, move || {
            let _ = m2.forward_cross::<B>(q2.clone(), kv2.clone());
        });

        // 3. Attention-only
        let m3 = MsaModule::new(&cfg, &dev);
        let q3 = Tensor::<3>::random([batch, Sq, dm], Distribution::Normal(0.0, 1.0), &dev);
        let kv3 = Tensor::<3>::random([batch, Sk, dm], Distribution::Normal(0.0, 1.0), &dev);
        let idx_i = q3.clone().detach();
        let kv_i = kv3.clone().detach();
        let (qi, ki) = m3.index_branch.forward(idx_i, kv_i, nk, di);
        let bscores = m3
            .index_branch
            .compute_block_scores(qi, ki, bs, (di as f64).sqrt(), false);
        let bi_pre = burn_msa::TopKSelector::new(tk, bs, false).select::<B>(bscores);

        let (qp, kp, vp) = {
            let a = &m3.attention;
            (
                a.q_proj.forward(q3),
                a.k_proj.forward(kv3.clone()),
                a.v_proj.forward(kv3),
            )
        };
        let t_attn = time_it(runs, move || {
            let _ = m3.attention.forward_sparse::<B>(
                qp.clone(),
                kp.clone(),
                vp.clone(),
                bi_pre.clone(),
            );
        });

        // 4. Dense
        let m4 = MsaModule::new(&cfg, &dev);
        let q4 = Tensor::<3>::random([batch, Sq, dm], Distribution::Normal(0.0, 1.0), &dev);
        let kv4 = Tensor::<3>::random([batch, Sk, dm], Distribution::Normal(0.0, 1.0), &dev);
        let (qp_d, kp_d, vp_d) = {
            let a = &m4.attention;
            (
                a.q_proj.forward(q4),
                a.k_proj.forward(kv4.clone()),
                a.v_proj.forward(kv4),
            )
        };
        let t_dense = time_it(runs / 2, move || {
            let _ = m4
                .attention
                .forward_dense::<B>(qp_d.clone(), kp_d.clone(), vp_d.clone());
        });

        let ratio = t_full / t_dense;
        let tok_s = (batch * Sq) as f64 / t_full;

        println!(
            "{:<16} {:>9.0} {:>9.0} {:>9.0} {:>9.0} {:>7.2}x {:>6} {:>9.0}",
            name,
            t_idx * 1e6,
            t_full * 1e6,
            t_attn * 1e6,
            t_dense * 1e6,
            ratio,
            batch * Sq,
            tok_s
        );
    }
    println!();
}
