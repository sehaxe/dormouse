//! Bit-for-bit dump harness (CPU, deterministic): one seeded input through
//! the per-token reference and the chunked tensor path; every tensor dumped
//! as f32-LE + a shape file for the python/FLA cross-check. Also dumps the
//! raw chunked output (before output()) and the output-stage weights.
//!   cargo run --release -p burn-kda --example bitforbit

use burn::prelude::*;
use burn_gdn2::chunk_wy_forward;
use burn_kda::{DecayFn, GateMode, KdaConfig, KdaModule};

fn dev() -> Device {
    Device::ndarray()
}

fn write_f32(name: &str, v: &[f32]) {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    std::fs::write(format!("/tmp/opencode/kda_bfb/{name}.bin"), bytes).expect("write");
}

fn dump(t: &Tensor<4>, name: &str) {
    let dims = t.shape().dims::<4>();
    std::fs::write(
        format!("/tmp/opencode/kda_bfb/{name}.shape"),
        format!("{}", dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" ")),
    )
    .expect("write shape");
    write_f32(name, &t.clone().into_data().to_vec::<f32>().expect("f32"));
}

fn dump3(t: &Tensor<3>, name: &str) {
    let dims = t.shape().dims::<3>();
    std::fs::write(
        format!("/tmp/opencode/kda_bfb/{name}.shape"),
        format!("{}", dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" ")),
    )
    .expect("write shape");
    write_f32(name, &t.clone().into_data().to_vec::<f32>().expect("f32"));
}

fn main() {
    std::fs::create_dir_all("/tmp/opencode/kda_bfb").expect("dir");
    let dev = dev();
    let cfg = KdaConfig {
        hidden_size: 64,
        num_heads: 2,
        head_dim: 32,
        num_v_heads: None,
        expand_v: 1.0,
        use_short_conv: false,
        rank: 16,
        decay_fn: DecayFn::Sigmoid,
        g_min: 0.0, // selects the fixed G_MIN floor
        gate: GateMode::FullRank,
        chunk_size: 8,
        norm_eps: 1e-5,
    };
    let km = KdaModule::new(&cfg, 0.9, &dev);

    let (b, t, dm) = (2usize, 16usize, 64usize);
    let x: Vec<f32> = (0..b * t * dm)
        .map(|i| ((i % 97) as f32 / 97.0 - 0.5))
        .collect();
    let x = Tensor::<3>::from_data(burn::tensor::TensorData::new(x, [b, t, dm]), &dev);

    let (q, k, v, g, b_k, b_v, gate) = km.project_for_test(x.clone());
    dump(&q, "q");
    dump(&k, "k");
    dump(&v, "v");
    dump(&g, "g");
    dump(&b_k, "b_k");
    dump(&b_v, "b_v");
    dump(&gate, "gate");

    // path 1: per-token reference
    let mut st: Option<Tensor<4>> = None;
    let o_rec = km.forward_recurrent(x.clone(), &mut st, true);
    dump3(&o_rec, "out_rec");
    if let Some(s) = &st {
        dump(s, "state_rec");
    }

    // path 2: chunked tensor path, RAW (before the output norm/gate/proj)
    let (q2, k2, v2, g2, bk2, bv2, _) = km.project_for_test(x.clone());
    let st0 = Tensor::<4>::zeros([b, cfg.num_heads, cfg.head_dim, cfg.head_dim], &dev)
        .cast(q2.dtype());
    let (o_raw, state_chunk) =
        chunk_wy_forward(q2, k2, v2, g2, bk2, bv2, st0, 1.0, cfg.chunk_size);
    dump(&o_raw, "o_raw");
    dump(&state_chunk, "state_chunk");

    // path 3: the full path (with output()) for end-to-end comparison
    let st3: Option<Tensor<4>> = None;
    let (o_chunk, state_chunk3) =
        km.forward_train_state::<burn::backend::NdArray>(x.clone(), st3);
    dump3(&o_chunk, "out_chunk");
    dump(&state_chunk3, "state_chunk3");

    // output-stage weights for the python replay
    write_f32(
        "o_norm_w",
        &km.o_norm_w.val().to_data().to_vec::<f32>().expect("w"),
    );
    let wd = km.o_proj.weight.val().to_data().to_vec::<f32>().expect("w");
    std::fs::write(
        "/tmp/opencode/kda_bfb/o_proj_weight.shape",
        format!("2"),
    )
    .expect("write shape");
    write_f32("o_proj_weight", &wd);

    println!("dumped to /tmp/opencode/kda_bfb");
}
