//! Tensor-dump harness (CUDA): the FUSED chunk kernel vs the per-token
//! reference, both on the CUDA device, both on the EXACT projections the CPU
//! harness dumped (loads q,k,v,g,b_k,b_v from /tmp/opencode/kda_bfb/ so no
//! RNG can drift between backends). Dumps f32-LE + .shape to
//! /tmp/opencode/kda_bfb_cuda/ and prints max-abs diffs.
//!
//! NOT a bit-for-bit comparison, despite this binary's name: the per-token
//! reference is `burn_kda`'s own `forward_recurrent`, so both sides are ours and
//! there is no external reference. The printed max-abs diff is the number to
//! read; the assertion lives in tests/fused_cuda.rs. See docs/protocols/ORACLE.md.
//!   cargo run --release -p burn-kda --example bitforbit_cuda --features cuda
#![cfg(feature = "cuda")]

use burn::prelude::*;
use burn_kda::fused::cuda::{kda_fused_chunk, CudaBare};

const IN: &str = "/tmp/opencode/kda_bfb";
const OUT: &str = "/tmp/opencode/kda_bfb_cuda";

fn read_f32(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(format!("{IN}/{name}.bin")).expect("bin");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn load4(dev: &Device, name: &str) -> Tensor<4> {
    let dims: Vec<usize> = std::fs::read_to_string(format!("{IN}/{name}.shape"))
        .expect("shape")
        .split_whitespace()
        .map(|d| d.parse().expect("dim"))
        .collect();
    Tensor::<4>::from_data(
        burn::tensor::TensorData::new(read_f32(name), [dims[0], dims[1], dims[2], dims[3]]),
        dev,
    )
}

fn write_f32(name: &str, v: &[f32]) {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    std::fs::write(format!("{OUT}/{name}.bin"), bytes).expect("write");
}

fn dump4(t: &Tensor<4>, name: &str) {
    let dims = t.shape().dims::<4>();
    std::fs::write(
        format!("{OUT}/{name}.shape"),
        dims.iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(" "),
    )
    .expect("shape");
    write_f32(
        name,
        &t.clone().into_data().try_to_vec::<f32>().expect("f32"),
    );
}

fn dmax(a: &[f32], b: &[f32], what: &str) -> f32 {
    let m = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!("{what}: max_abs = {m:.3e}");
    m
}

fn main() {
    std::fs::create_dir_all(OUT).expect("dir");
    let dev = Device::cuda(0);

    // Identical inputs: the exact projections the CPU harness dumped.
    let q = load4(&dev, "q");
    let k = load4(&dev, "k");
    let v = load4(&dev, "v");
    let g = load4(&dev, "g");
    let b_k = load4(&dev, "b_k");
    let b_v = load4(&dev, "b_v");
    let [b, h, t, hk] = q.shape().dims::<4>();
    let hv = v.shape().dims::<4>()[1];
    let vd = v.shape().dims::<4>()[3];
    let chunk = 8usize; // bitforbit.rs KdaConfig.chunk_size
    println!("inputs from {IN}: [b={b} h={h} t={t} hk={hk}] hv={hv} vd={vd} chunk={chunk}");

    // ── path F: fused CUDA chunk kernel ──────────────────────────────
    let st0 = Tensor::<4>::zeros([b, hv, hk, vd], &dev);
    let (out_fused, state_fused) = kda_fused_chunk::<CudaBare>(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        b_k.clone(),
        b_v,
        st0,
        chunk,
    )
    .expect("fused path must apply on CudaBare");

    // ── path R: per-token reference on CUDA (forward_recurrent's loop) ──
    let mut s = Tensor::<4>::zeros([b, hv, hk, vd], &dev);
    let mut outs = Vec::with_capacity(t);
    for tt in 0..t {
        let q_t = q.clone().slice_dim(2, tt..tt + 1);
        let k_t = k.clone().slice_dim(2, tt..tt + 1);
        let v_t = v.clone().slice_dim(2, tt..tt + 1);
        let d_t = g.clone().slice_dim(2, tt..tt + 1).exp();
        let beta_t = b_k.clone().slice_dim(2, tt..tt + 1);
        s = s * d_t.swap_dims(2, 3);
        let erased = (s.clone() * k_t.clone().swap_dims(2, 3))
            .sum_dim(2)
            .mul(beta_t.clone());
        s = s - k_t.clone().swap_dims(2, 3) * erased;
        s = s + k_t.swap_dims(2, 3) * v_t.mul(beta_t);
        let out = (s.clone() * q_t.swap_dims(2, 3)).sum_dim(2);
        outs.push(out);
    }
    let state_ref = s;
    let out_ref = Tensor::cat(outs, 2);

    dump4(&out_fused, "out_fused");
    dump4(&state_fused, "state_fused");
    dump4(&out_ref, "out_ref_cuda");
    dump4(&state_ref, "state_ref_cuda");

    // ── compare: fused vs CUDA reference vs CPU reference ────────────
    let of = out_fused
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .expect("f32");
    let or = out_ref
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .expect("f32");
    let sf = state_fused
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .expect("f32");
    let sr = state_ref
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .expect("f32");
    dmax(&of, &or, "out   fused vs ref_cuda");
    dmax(&sf, &sr, "state fused vs ref_cuda");

    // CPU counterparts: only same-shape/same-stage tensors are comparable —
    // out_rec/out_chunk are post-output-stage [B,T,D], the fused/ref outputs
    // here are the RAW [B,H,T,V] chunk outputs.
    let cpu_o_raw = read_f32("o_raw");
    let cpu_s_chunk = read_f32("state_chunk");
    let cpu_s_rec = read_f32("state_rec");
    dmax(&of, &cpu_o_raw, "out   fused vs CPU o_raw (same WY inputs)");
    dmax(&sf, &cpu_s_chunk, "state fused vs CPU state_chunk");
    dmax(&sf, &cpu_s_rec, "state fused vs CPU state_rec");
    dmax(&sr, &cpu_s_rec, "state ref_cuda vs CPU state_rec");

    println!("dumped to {OUT}");
}
