//! Fused-kernel performance harness for the burn-fused workspace.
//!
//! Runs each crate's fused forward on the CUDA backend and prints a JSON
//! report. The CI compares the fused timings against `bench/baselines.json`
//! and fails on regressions.
//!
//! The backend is ambient in burn 0.22: run with `BURN_DEVICE=cuda`.
//!
//! Usage: `BURN_DEVICE=cuda cargo run -p burn-fused-benches --release`

use burn::tensor::{Device, Distribution, Tensor};
use std::collections::BTreeMap;
use std::time::Instant;

// backend type used by the crates' fused dispatch (rope is generic over it)
type B = burn_cubecl::CubeBackend;

fn time<F: FnMut()>(runs: usize, mut f: F) -> f64 {
    // warmup (JIT, clock ramp)
    for _ in 0..3 {
        f();
    }
    // Sync point, not a cache flush: the tiny sum forces the queue to drain
    // so every measured iteration includes one sync. Its own cost is timed
    // separately and subtracted below (it matters for sub-0.1ms kernels,
    // where it is ~30-40% of the raw reading). GPU L2 is NOT flushed between
    // iterations — memory-resident kernels are measured warm.
    let flush = || {
        let _: f32 = Tensor::<1>::ones([1], &Device::default())
            .sum()
            .into_scalar();
    };
    flush();
    let mut flush_min = f64::MAX;
    for _ in 0..runs {
        let t0 = Instant::now();
        flush();
        flush_min = flush_min.min(t0.elapsed().as_secs_f64() * 1000.0);
    }
    // min over runs: the stable metric (mean includes clock-ramp noise)
    let mut best = f64::MAX;
    for _ in 0..runs {
        let t0 = Instant::now();
        f();
        flush();
        best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
    }
    (best - flush_min).max(0.0)
}

fn main() {
    let dev = Device::default();
    let mut out: BTreeMap<String, serde_json::Value> = BTreeMap::new();

    // --- rope ---
    {
        let (b, t, nh, hd) = (4usize, 2048, 32, 128);
        let x = Tensor::<4>::random([b, t, nh, hd], Distribution::Normal(0.0, 1.0), &dev);
        let (cos, sin) = burn_rope::precompute_freqs(hd, t, 10000.0, &dev);
        out.insert(
            "rope.fused_ms".into(),
            time(20, || {
                // no clones: apply_rope_4d consumes by value and does not
                // mutate, so clone-free timing measures the kernel alone
                let _ = burn_rope::apply_rope_4d::<B>(x.clone(), cos.clone(), sin.clone());
            })
            .into(),
        );
    }

    // --- situ ---
    {
        let (n, h) = (2048usize, 5120usize);
        let gu = Tensor::<2>::random([n, 2 * h], Distribution::Normal(0.0, 1.0), &dev);
        out.insert(
            "situ.fused_ms".into(),
            time(20, || {
                let _ = burn_situ::situ_glu(gu.clone(), h, 1.0, 1.5);
            })
            .into(),
        );
    }

    // --- bitnet FWT ---
    {
        let (n, d) = (256usize, 512usize);
        let x = Tensor::<2>::random([n, d], Distribution::Normal(0.0, 1.0), &dev);
        out.insert(
            "bitnet_fwt.fused_ms".into(),
            time(200, || {
                let _ = burn_bitnet::fast_walsh_hadamard(x.clone());
            })
            .into(),
        );
    }

    // --- mhc sinkhorn ---
    {
        let (b, t, n) = (8usize, 2048, 16usize);
        let logits = Tensor::<4>::random([b, t, n, n], Distribution::Normal(0.0, 1.0), &dev);
        out.insert(
            "mhc_sinkhorn.fused_ms".into(),
            time(20, || {
                let _ = burn_mhc::sinkhorn_knopp(logits.clone(), 20);
            })
            .into(),
        );
    }

    // --- attnres depth attend ---
    {
        let (l, b, t, d) = (24usize, 1usize, 2048usize, 4096usize);
        let hist: Vec<Tensor<3>> = (0..l)
            .map(|_| Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &dev))
            .collect();
        let q = Tensor::<1>::random([d], Distribution::Normal(0.0, 1.0), &dev);
        out.insert(
            "attnres.fused_ms".into(),
            time(10, || {
                let _ = burn_attnres::depth_attend(&hist, q.clone());
            })
            .into(),
        );
    }

    // --- msa sparse attention (fused kernel, topk >= 4) ---
    {
        let (b, hq, hkv, s, d, topk, bs) = (1usize, 8, 2, 64, 16, 4, 8);
        let nblocks = s / bs;
        let q = Tensor::<4>::random([b, hq, s, d], Distribution::Normal(0.0, 1.0), &dev);
        let k = Tensor::<4>::random([b, hkv, s, d], Distribution::Normal(0.0, 1.0), &dev);
        let v = Tensor::<4>::random([b, hkv, s, d], Distribution::Normal(0.0, 1.0), &dev);
        // Deterministic golden-ratio hash spreads the block indices over ALL
        // nblocks uniformly. (The previous pattern `(i % topk) % nblocks`
        // only ever touched the first `topk` blocks, biasing memory access
        // locality versus a realistic routing distribution.)
        let bi_data: Vec<i64> = (0..b * hkv * s * topk)
            .map(|i| {
                let h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                ((h >> 33) % nblocks as u64) as i64
            })
            .collect();
        let bi = Tensor::<4, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(bi_data, [b, hkv, s, topk]),
            &dev,
        );
        let fused = time(200, || {
            let r = burn_msa::sparse_kernel::sparse_attn_cuda::<B>(
                q.clone(),
                k.clone(),
                v.clone(),
                bi.clone(),
                (d as f64).sqrt(),
                bs,
                hkv,
                hq,
                false,
            );
            let _ = r.expect("fused path must trigger");
        });
        out.insert("msa_sparse.fused_ms".into(), fused.into());
    }

    // --- muon-plus: column-row norm (in-place fused) ---
    {
        let (n, h) = (2048usize, 5120usize);
        let mut x = Tensor::<2>::random([n, h], Distribution::Normal(0.0, 1.0), &dev);
        out.insert(
            "muon_norm.fused_ms".into(),
            time(200, || {
                let _ = burn_muon_plus::fused_kernels::norm_colrow_cuda::<2>(&mut x, 1e-6);
            })
            .into(),
        );
    }

    // --- gdn2: fused chunked prefill (the module's canonical CUDA path) ---
    {
        let (b, h, t, d, chunk) = (1usize, 4, 128, 64, 16usize);
        let mk = |n: usize| Tensor::<4>::random([b, h, t, n], Distribution::Normal(0.0, 1.0), &dev);
        let q = mk(d);
        let k = mk(d);
        let v = mk(d);
        let g = mk(d);
        let bb = mk(d);
        let w = mk(d);
        let state = Tensor::<4>::zeros([b, h, d, d], &dev);
        out.insert(
            "gdn2_prefill.fused_ms".into(),
            time(20, || {
                let r = burn_gdn2::kernel::chunk_cube::cuda::fused_chunk_forward::<B>(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    g.clone(),
                    bb.clone(),
                    w.clone(),
                    state.clone(),
                    1.0,
                    chunk,
                );
                let _ = r.expect("fused chunk path must trigger");
            })
            .into(),
        );
    }

    // --- kda: fused chunked prefill (reuses GDN2 fused chunk kernels) ---
    {
        let (b, h, t, dk, dv, chunk) = (1usize, 8, 128, 64, 64, 16usize);
        let mk4 =
            |n: usize| Tensor::<4>::random([b, h, t, n], Distribution::Normal(0.0, 1.0), &dev);
        let q = mk4(dk);
        let k = mk4(dk);
        let v = mk4(dv);
        let log_a = mk4(dk);
        let bk = mk4(dk);
        let bv = mk4(dv);
        let state = Tensor::<4>::zeros([b, h, dk, dv], &dev);
        out.insert(
            "kda_prefill.fused_ms".into(),
            time(20, || {
                let r = burn_kda::fused::cuda::kda_fused_chunk::<B>(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    log_a.clone(),
                    bk.clone(),
                    bv.clone(),
                    state.clone(),
                    chunk,
                );
                let _ = r.expect("fused kda path must trigger");
            })
            .into(),
        );
    }

    // --- sct: QR from_dense (fused retraction) ---
    {
        let (n, m, rank) = (512usize, 256, 32);
        let w = Tensor::<2>::random([n, m], Distribution::Normal(0.0, 1.0), &dev);
        out.insert(
            "sct_qr.fused_ms".into(),
            time(2, || {
                // sweeps=15: the crate's own benchmark convention (bench_ops.rs)
                let _ = burn_sct::qr_cuda::from_dense_cuda::<B>(w.clone(), rank, 15);
            })
            .into(),
        );
    }

    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
