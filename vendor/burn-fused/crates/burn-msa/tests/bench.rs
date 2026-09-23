// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
#![allow(missing_docs)]
use burn::backend::Backend;
use burn::tensor::{Device, Distribution, Int, Tensor};
use burn_msa::{softmax, sparse_attn_batched_gqa, MsaConfig, MsaModule};
use std::time::Instant;

fn cfg(
    d_model: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    d_idx: usize,
    topk: usize,
    block_size: usize,
) -> MsaConfig {
    MsaConfig {
        d_model,
        d_idx,
        n_heads_q,
        n_heads_kv,
        d_head: d_model / n_heads_q,
        block_size,
        topk,
        causal: false,
        force_local_block: false,
        use_kl_loss: false,
        warmup_steps: 0,
        kl_coeff: 0.0,
        gradient_detach: true,
        use_rope: false,
        rope_base: 10000.0,
        rope_max_seq_len: 1024,
    }
}

fn time_it<B: Backend>(runs: usize, mut f: impl FnMut()) -> f64
where
    B::Device: Default,
{
    // All tensors in this bench live on the backend's default device; sync so
    // the measured time covers the enqueued work instead of just the launches.
    let sync = || {
        let _ = B::sync(&B::Device::default());
    };
    for _ in 0..(runs.min(5)) {
        f();
        sync();
    }
    let start = Instant::now();
    for _ in 0..runs {
        f();
        sync();
    }
    start.elapsed().as_secs_f64() / runs as f64
}

#[test]
#[ignore]
fn bench_ndarray() {
    run_backend::<burn_ndarray::NdArray>("NdArray (CPU)");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn bench_cuda() {
    run_backend::<burn_cuda::Cuda>("CUDA (GPU)");
}

#[allow(clippy::needless_range_loop)]
fn run_backend<B: burn::backend::Backend>(label: &str)
where
    B::Device: Clone + Default,
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    let dev = Device::default();
    println!("\n{:=^80}", "");
    println!("  BURN-MSA BENCHMARKS - {label}");
    println!("{:=^80}", "");

    println!("\n── 1. TIMING WRAPPER VERIFICATION ──");
    let t0 = time_it::<B>(100, || {
        std::hint::spin_loop();
    });
    println!("  empty-loop: {:.2} ns/step (expected < 1 us)", t0 * 1e9);

    // ========================================================================
    // SECTION 1: Throughput matrix (sparse vs dense)
    // ========================================================================
    println!("\n── 2. THROUGHPUT: sparse vs dense ──");
    println!(
        "{:>8} {:>8} {:>8} {:>12} {:>12} {:>8}",
        "dm", "seq_q", "seq_kv", "sparse(us)", "dense(us)", "ratio"
    );
    for &(dm, sq, sk) in &[
        (256, 1, 4096),
        (256, 256, 4096),
        (512, 1, 4096),
        (512, 256, 4096),
        (512, 512, 8192),
        (768, 1, 4096),
        (768, 256, 4096),
    ] {
        let nq = 8.min(dm / 64);
        let nk = 2.min(nq);
        let c = cfg(dm, nq, nk, 64, 8, 64);
        let runs = if sq <= 8 { 50 } else { 20 };

        let m = MsaModule::new(&c, &dev);
        let q = Tensor::<3>::random([1, sq, dm], Distribution::Normal(0.0, 1.0), &dev);
        let kv = Tensor::<3>::random([1, sk, dm], Distribution::Normal(0.0, 1.0), &dev);

        let m2 = m.clone();
        let q2 = q.clone();
        let kv2 = kv.clone();
        let t_sp = time_it::<B>(runs, move || {
            let _ = m2.forward_cross::<B>(q2.clone(), kv2.clone());
        });

        let (qp, kp, vp) = {
            let a = &m.attention;
            (
                a.q_proj.forward(q),
                a.k_proj.forward(kv.clone()),
                a.v_proj.forward(kv),
            )
        };
        let t_de = time_it::<B>(runs / 2, move || {
            let _ = m
                .attention
                .forward_dense::<B>(qp.clone(), kp.clone(), vp.clone());
        });

        println!(
            "{:>8} {:>8} {:>8} {:>12.1} {:>12.1} {:>7.1}x",
            dm,
            sq,
            sk,
            t_sp * 1e6,
            t_de * 1e6,
            t_de / t_sp
        );
    }

    // ========================================================================
    // SECTION 2: Long context scaling
    // ========================================================================
    println!("\n── 3. LONG CONTEXT SCALING ──");
    println!(
        "{:>10} {:>10} {:>12} {:>12} {:>7} {:>12}",
        "seq_kv", "attended", "spare(us)", "dense(us)", "speedup", "tok/s_sp"
    );
    let c512 = cfg(512, 8, 2, 64, 8, 64);
    for &sk in &[1024usize, 2048, 4096, 8192, 16384] {
        let m = MsaModule::new(&c512, &dev);
        let q = Tensor::<3>::random([1, 256, 512], Distribution::Normal(0.0, 1.0), &dev);
        let kv = Tensor::<3>::random([1, sk, 512], Distribution::Normal(0.0, 1.0), &dev);
        let (qp, kp, vp) = {
            let a = &m.attention;
            (
                a.q_proj.forward(q.clone()),
                a.k_proj.forward(kv.clone()),
                a.v_proj.forward(kv.clone()),
            )
        };
        let t_de = time_it::<B>(3, move || {
            let _ = m
                .attention
                .forward_dense::<B>(qp.clone(), kp.clone(), vp.clone());
        });

        let m2 = MsaModule::new(&c512, &dev);
        let t_sp = time_it::<B>(5, move || {
            let _ = m2.forward_cross::<B>(q.clone(), kv.clone());
        });
        println!(
            "{:>10} {:>10} {:>12.1} {:>12.1} {:>6.1}x {:>12.0}",
            sk,
            512,
            t_sp * 1e6,
            t_de * 1e6,
            t_de / t_sp,
            sk as f64 / t_sp
        );
    }

    // ========================================================================
    // SECTION 3: Block size sweep
    // ========================================================================
    println!("\n── 4. BLOCK SIZE SWEEP ──");
    println!(
        "{:>12} {:>10} {:>12} {:>12} {:>7}",
        "block_size", "attended", "sparse(us)", "dense(us)", "speedup"
    );
    let dm = 512;
    let sq = 256;
    let sk = 4096;
    let q = Tensor::<3>::random([1, sq, dm], Distribution::Normal(0.0, 1.0), &dev);
    let kv = Tensor::<3>::random([1, sk, dm], Distribution::Normal(0.0, 1.0), &dev);
    for &bs in &[16usize, 32, 64, 128, 256] {
        let c = cfg(dm, 8, 2, 64, 8, bs);
        let m = MsaModule::new(&c, &dev);
        let m2 = m.clone();
        let q2 = q.clone();
        let kv2 = kv.clone();
        let t_sp = time_it::<B>(10, move || {
            let _ = m2.forward_cross::<B>(q2.clone(), kv2.clone());
        });
        let (qp, kp, vp) = {
            let a = &m.attention;
            (
                a.q_proj.forward(q.clone()),
                a.k_proj.forward(kv.clone()),
                a.v_proj.forward(kv.clone()),
            )
        };
        let t_de = time_it::<B>(5, move || {
            let _ = m
                .attention
                .forward_dense::<B>(qp.clone(), kp.clone(), vp.clone());
        });
        println!(
            "{:>12} {:>10} {:>12.1} {:>12.1} {:>6.1}x",
            bs,
            8 * bs,
            t_sp * 1e6,
            t_de * 1e6,
            t_de / t_sp
        );
    }

    // ========================================================================
    // SECTION 5: GQA ratio sweep
    // ========================================================================
    println!("\n── 5. GQA RATIO SCALING ──");
    println!(
        "{:>8} {:>8} {:>6} {:>12} {:>12} {:>7}",
        "n_q", "n_kv", "ratio", "sparse(us)", "dense(us)", "speedup"
    );
    let q = Tensor::<3>::random([1, 256, 512], Distribution::Normal(0.0, 1.0), &dev);
    let kv = Tensor::<3>::random([1, 4096, 512], Distribution::Normal(0.0, 1.0), &dev);
    for &(nq, nk) in &[(4, 1), (4, 2), (8, 1), (8, 2), (8, 4), (16, 2)] {
        let c = cfg(512, nq, nk, 64, 8, 64);
        let m = MsaModule::new(&c, &dev);
        let m2 = m.clone();
        let q2 = q.clone();
        let kv2 = kv.clone();
        let t_sp = time_it::<B>(10, move || {
            let _ = m2.forward_cross::<B>(q2.clone(), kv2.clone());
        });
        let (qp, kp, vp) = {
            let a = &m.attention;
            (
                a.q_proj.forward(q.clone()),
                a.k_proj.forward(kv.clone()),
                a.v_proj.forward(kv.clone()),
            )
        };
        let t_de = time_it::<B>(5, move || {
            let _ = m
                .attention
                .forward_dense::<B>(qp.clone(), kp.clone(), vp.clone());
        });
        println!(
            "{:>8} {:>8} {:>5}:1 {:>12.1} {:>12.1} {:>6.1}x",
            nq,
            nk,
            nq / nk,
            t_sp * 1e6,
            t_de * 1e6,
            t_de / t_sp
        );
    }

    // ========================================================================
    // SECTION 6: Decode mode
    // ========================================================================
    println!("\n── 6. DECODE MODE (seq_q=1) ──");
    println!(
        "{:>10} {:>12} {:>12} {:>7} {:>12}",
        "seq_kv", "sparse(us)", "dense(us)", "speedup", "tok/s_de"
    );
    for &sk in &[1024usize, 4096, 8192, 16384, 32768, 65536] {
        let c = cfg(512, 8, 2, 64, 8, 64);
        let m = MsaModule::new(&c, &dev);
        let q = Tensor::<3>::random([1, 1, 512], Distribution::Normal(0.0, 1.0), &dev);
        let kv = Tensor::<3>::random([1, sk, 512], Distribution::Normal(0.0, 1.0), &dev);
        let m2 = m.clone();
        let q2 = q.clone();
        let kv2 = kv.clone();
        let t_sp = time_it::<B>(100, move || {
            let _ = m2.forward_cross::<B>(q2.clone(), kv2.clone());
        });
        let (qp, kp, vp) = {
            let a = &m.attention;
            (
                a.q_proj.forward(q),
                a.k_proj.forward(kv.clone()),
                a.v_proj.forward(kv),
            )
        };
        let t_de = time_it::<B>(50, move || {
            let _ = m
                .attention
                .forward_dense::<B>(qp.clone(), kp.clone(), vp.clone());
        });
        println!(
            "{:>10} {:>12.1} {:>12.1} {:>6.1}x {:>12.0}",
            sk,
            t_sp * 1e6,
            t_de * 1e6,
            t_de / t_sp,
            sk as f64 / t_de
        );
    }

    // ========================================================================
    // SECTION 7: Per-op profiling
    // ========================================================================
    println!("\n── 7. PER-OP PROFILING BREAKDOWN ──");
    let dh = 64;
    let nq = 8;
    let nk = 2;
    let bs = 64;
    let sq = 256;
    let sk = 4096;
    let tk = 8;
    let scale = (dh as f64).sqrt();
    let att = tk * bs;
    let runs = 10;

    let q4 = Tensor::<4>::random([1, nq, sq, dh], Distribution::Default, &dev);
    let k4 = Tensor::<4>::random([1, nk, sk, dh], Distribution::Default, &dev);
    let v4 = Tensor::<4>::random([1, nk, sk, dh], Distribution::Default, &dev);
    let nb = (sk as f64 / bs as f64).ceil() as usize;
    let bi = Tensor::<4, Int>::random(
        [1, nk, sq, tk],
        Distribution::Uniform(0.0, (nb - 1) as f64),
        &dev,
    );

    let qc = q4.clone();
    let kc = k4.clone();
    let vc = v4.clone();
    let bic = bi.clone();
    let t_full = time_it::<B>(runs, move || {
        let _ = sparse_attn_batched_gqa(
            qc.clone(),
            kc.clone(),
            vc.clone(),
            bic.clone(),
            scale,
            bs,
            nk,
            nq,
            false,
        );
    });

    let offsets = Tensor::<1, Int>::arange(0..bs as i64, &dev).reshape::<5, _>([1, 1, 1, 1, bs]);
    let all_idx = bi
        .clone()
        .mul_scalar(bs as i64)
        .unsqueeze_dim::<5>(4)
        .add(offsets)
        .reshape::<4, _>([1, nk, sq, att]);

    let ai = all_idx.clone();
    let k4c = k4.clone();
    let v4c = v4.clone();
    let t_gather = time_it::<B>(runs, move || {
        let gi = ai
            .clone()
            .reshape::<4, _>([1, nk, sq * att, 1])
            .repeat(&[1, 1, 1, dh])
            .clamp_max((sk - 1) as i64);
        let _ = k4c
            .clone()
            .gather(2, gi.clone())
            .reshape::<5, _>([1, nk, sq, att, dh]);
        let _ = v4c
            .clone()
            .gather(2, gi)
            .reshape::<5, _>([1, nk, sq, att, dh]);
    });

    let gi = all_idx
        .clone()
        .reshape::<4, _>([1, nk, sq * att, 1])
        .repeat(&[1, 1, 1, dh])
        .clamp_max((sk - 1) as i64);
    let k_g = k4
        .clone()
        .gather(2, gi)
        .reshape::<5, _>([1, nk, sq, att, dh]);
    let q_gqa = q4
        .clone()
        .reshape::<5, _>([1, nk, nq / nk, sq, dh])
        .swap_dims(2, 3);
    let k_t = k_g.clone().swap_dims(3, 4);

    let qc2 = q_gqa.clone();
    let kc2 = k_t.clone();
    let t_score = time_it::<B>(runs, move || {
        let _ = qc2.clone().matmul(kc2.clone()).div_scalar(scale);
    });

    let scores = q_gqa.clone().matmul(k_t.clone()).div_scalar(scale);
    let sc = scores.clone();
    let t_sm = time_it::<B>(runs, move || {
        let _ = softmax(sc.clone(), 4);
    });

    let attn = softmax(scores, 4);
    let ac = attn.clone();
    let kc3 = k_g.clone();
    let t_out = time_it::<B>(runs, move || {
        let _ = ac
            .clone()
            .matmul(kc3.clone())
            .swap_dims(2, 3)
            .reshape::<4, _>([1, nq, sq, dh]);
    });

    println!("{:>22} | {:>12} {:>7}", "operation", "time(us)", "%");
    let pct = |t| (t / t_full * 100.0) as usize;
    println!(
        "  {:<20} | {:>10.1} us {:>6}%",
        "gather K+V",
        t_gather * 1e6,
        pct(t_gather)
    );
    println!(
        "  {:<20} | {:>10.1} us {:>6}%",
        "score matmul (5D)",
        t_score * 1e6,
        pct(t_score)
    );
    println!(
        "  {:<20} | {:>10.1} us {:>6}%",
        "softmax",
        t_sm * 1e6,
        pct(t_sm)
    );
    println!(
        "  {:<20} | {:>10.1} us {:>6}%",
        "output matmul",
        t_out * 1e6,
        pct(t_out)
    );
    println!(
        "  {:<20} | {:>10.1} us {:>6}%",
        "TOTAL accounted",
        (t_gather + t_score + t_sm + t_out) * 1e6,
        pct(t_gather + t_score + t_sm + t_out)
    );
    println!(
        "  {:<20} | {:>10.1} us {:>6}%",
        "full sparse",
        t_full * 1e6,
        100
    );

    // ========================================================================
    // SECTION 8: TopK overhead
    // ========================================================================
    println!("\n── 8. INDEX+TOPK OVERHEAD ──");
    let dm = 512;
    for &sk in &[1024usize, 4096, 8192, 16384] {
        let c = cfg(dm, 8, 2, 64, 8, 64);
        let sq = 256;
        let m = MsaModule::new(&c, &dev);
        let q = Tensor::<3>::random([1, sq, dm], Distribution::Normal(0.0, 1.0), &dev);
        let kv = Tensor::<3>::random([1, sk, dm], Distribution::Normal(0.0, 1.0), &dev);
        let runs = if sk > 4096 { 5 } else { 10 };

        let m2 = m.clone();
        let q2 = q.clone();
        let kv2 = kv.clone();
        let t_full = time_it::<B>(runs, move || {
            let _ = m2.forward_cross::<B>(q2.clone(), kv2.clone());
        });

        let idx = Tensor::<4, Int>::random(
            [1, 2, sq, 8],
            Distribution::Uniform(0.0, (sk / 64 - 1) as f64),
            &dev,
        );
        let (qp, kp, vp) = {
            let a = &m.attention;
            (
                a.q_proj.forward(q),
                a.k_proj.forward(kv.clone()),
                a.v_proj.forward(kv),
            )
        };
        let t_attn = time_it::<B>(runs, move || {
            let _ =
                m.attention
                    .forward_sparse::<B>(qp.clone(), kp.clone(), vp.clone(), idx.clone());
        });
        println!(
            "  kv={:<6} total={:.1}us  attn={:.1}us ({:.0}%)  index={:.1}us ({:.0}%)",
            sk,
            t_full * 1e6,
            t_attn * 1e6,
            t_attn / t_full * 100.0,
            (t_full - t_attn) * 1e6,
            (t_full - t_attn) / t_full * 100.0
        );
    }

    // ========================================================================
    // SECTION 9: Causal mask overhead
    // ========================================================================
    println!("\n── 9. CAUSAL MASK OVERHEAD ──");
    let dh = 64;
    let bs = 64;
    let tk = 8;
    let nk = 2;
    let nq = 8;
    let sq = 256;
    let s = (dh as f64).sqrt();
    for &sk in &[256usize, 1024, 4096] {
        let q = Tensor::<4>::random([1, nq, sq, dh], Distribution::Default, &dev);
        let k = Tensor::<4>::random([1, nk, sk, dh], Distribution::Default, &dev);
        let v = Tensor::<4>::random([1, nk, sk, dh], Distribution::Default, &dev);
        let nb = (sk as f64 / bs as f64).ceil() as usize;
        let bi = Tensor::<4, Int>::random(
            [1, nk, sq, tk.min(nb)],
            Distribution::Uniform(0.0, (nb - 1).max(1) as f64),
            &dev,
        );
        let qc = q.clone();
        let kc = k.clone();
        let vc = v.clone();
        let bic = bi.clone();
        let t_nc = time_it::<B>(10, move || {
            let _ = sparse_attn_batched_gqa(
                qc.clone(),
                kc.clone(),
                vc.clone(),
                bic.clone(),
                s,
                bs,
                nk,
                nq,
                false,
            );
        });
        let t_c = time_it::<B>(10, move || {
            let _ = sparse_attn_batched_gqa(
                q.clone(),
                k.clone(),
                v.clone(),
                bi.clone(),
                s,
                bs,
                nk,
                nq,
                true,
            );
        });
        println!(
            "  kv={:<6} nc={:.1}us  c={:.1}us  overhead={:.1}%",
            sk,
            t_nc * 1e6,
            t_c * 1e6,
            (t_c / t_nc - 1.0) * 100.0
        );
    }

    // ========================================================================
    // SECTION 10: Batch scaling
    // ========================================================================
    println!("\n── 10. BATCH SCALING ──");
    println!(
        "{:>8} {:>12} {:>12} {:>7}",
        "batch", "sparse(ms)", "dense(ms)", "speedup"
    );
    let c = cfg(512, 8, 2, 64, 8, 64);
    for &batch in &[1usize, 2, 4, 8] {
        let m = MsaModule::new(&c, &dev);
        let q = Tensor::<3>::random([batch, 256, 512], Distribution::Normal(0.0, 1.0), &dev);
        let kv = Tensor::<3>::random([batch, 4096, 512], Distribution::Normal(0.0, 1.0), &dev);
        let m2 = m.clone();
        let q2 = q.clone();
        let kv2 = kv.clone();
        let t_sp = time_it::<B>(if batch <= 2 { 10 } else { 5 }, move || {
            let _ = m2.forward_cross::<B>(q2.clone(), kv2.clone());
        });
        let (qp, kp, vp) = {
            let a = &m.attention;
            (
                a.q_proj.forward(q),
                a.k_proj.forward(kv.clone()),
                a.v_proj.forward(kv),
            )
        };
        let t_de = time_it::<B>(3, move || {
            let _ = m
                .attention
                .forward_dense::<B>(qp.clone(), kp.clone(), vp.clone());
        });
        println!(
            "{:>8} {:>12.4} {:>12.4} {:>6.1}x",
            batch,
            t_sp * 1e3,
            t_de * 1e3,
            t_de / t_sp
        );
    }

    println!("{:-^80}\n", "");
}
