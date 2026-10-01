//! KDA fused-kernel benchmark on the production training shape.
//!   cargo run --release -p dormouse-kda --example kda_bench --features cuda
#![cfg(feature = "cuda")]

use burn::prelude::*;
use dormouse_kda::fused::cuda::{kda_fused_chunk, CudaBare};

fn load4(dev: &Device, name: &str) -> Tensor<4> {
    let dims: Vec<usize> = std::fs::read_to_string(format!("/tmp/opencode/kda_bfb/{name}.shape"))
        .expect("shape")
        .split_whitespace()
        .map(|d| d.parse().expect("dim"))
        .collect();
    let bytes = std::fs::read(format!("/tmp/opencode/kda_bfb/{name}.bin")).expect("bin");
    let v: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    Tensor::<4>::from_data(
        burn::tensor::TensorData::new(v, [dims[0], dims[1], dims[2], dims[3]]),
        dev,
    )
}

fn main() {
    let dev = burn::tensor::Device::cuda(0);
    // production training shape via the dumped projections
    let q = load4(&dev, "q").repeat(&[1, 1, 32, 1]); // [2,2,16,32] -> [2,2,512,32]
    let k = load4(&dev, "k").repeat(&[1, 1, 32, 1]);
    let v = load4(&dev, "v").repeat(&[1, 1, 32, 1]);
    let g = load4(&dev, "g").repeat(&[1, 1, 32, 1]);
    let b_k = load4(&dev, "b_k").repeat(&[1, 1, 32, 1]);
    let b_v = load4(&dev, "b_v").repeat(&[1, 1, 32, 1]);
    let _ = (q.shape(), k.shape());
    // simpler: synthetic production shape directly
    let (b, t, h, hk, vd) = (10usize, 512usize, 12usize, 64usize, 64usize);
    let rand = |n: usize, lo: f32, hi: f32| -> Vec<f32> {
        let mut st = 0x243F6A8885A308D3u64;
        (0..n)
            .map(|_| {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                lo + (st % 10000) as f32 / 10000.0 * (hi - lo)
            })
            .collect()
    };
    let mk = |n: usize, dims: [usize; 4]| -> Tensor<4> {
        Tensor::<4>::from_data(burn::tensor::TensorData::new(rand(n, -1.0, 1.0), dims), &dev)
    };
    let q = mk(b * t * h * hk, [b, h, t, hk]);
    let k = mk(b * t * h * hk, [b, h, t, hk]);
    let v = mk(b * t * h * vd, [b, h, t, vd]);
    let g = mk(b * t * h * hk, [b, h, t, hk]).neg(); // log-decay: negative
    let b_k = mk(b * t * h * hk, [b, h, t, hk]);
    let b_v = mk(b * t * h * vd, [b, h, t, vd]);
    let _ = (q.shape(), k.shape());
    let st0 = Tensor::<4>::zeros([b, h, hk, vd], &dev);

    // correctness anchor: the fused path must run (no silent fallback)
    let (_o, _s) = kda_fused_chunk::<CudaBare>(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        b_k.clone(),
        b_v.clone(),
        st0.clone(),
        16,
    )
    .expect("fused must run");

    for _ in 0..10 {
        let _ = kda_fused_chunk::<CudaBare>(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            b_k.clone(),
            b_v.clone(),
            st0.clone(),
            16,
        )
        .expect("fused must run");
    }
    let n = 200;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let (o, s) = kda_fused_chunk::<CudaBare>(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            b_k.clone(),
            b_v.clone(),
            st0.clone(),
            16,
        )
        .expect("fused must run");
        std::hint::black_box(o.shape());
        std::hint::black_box(s.shape());
    }
    let dt = t0.elapsed();
    println!(
        "kda_fused b={b} t={t} h={h} hk={hk} vd={vd} chunk=16: {n} iters in {dt:?} => {:.3} ms/iter",
        dt.as_secs_f64() * 1000.0 / n as f64
    );
    let state_mib = (b * h * hk * vd * 4) as f64 / 1024.0 / 1024.0;
    println!("recurrent state: {state_mib:.2} MiB (constant, context-independent)");
}
