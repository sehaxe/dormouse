#![cfg(feature = "cuda")]
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
#![allow(
    missing_docs,
    non_snake_case,
    dead_code,
    unused_imports,
    unused_variables
)]
//! dormouse-gdn2 CUDA benchmark - projections + chunk forward.
//! Run: cargo test --release --features cuda -p dormouse-gdn2 --test bench_cuda -- --ignored --nocapture

#[cfg(feature = "cuda")]
use burn::backend::NdArray;
#[cfg(feature = "cuda")]
use burn::module::Module;
#[cfg(feature = "cuda")]
use burn::tensor::{Distribution, Tensor};
#[cfg(feature = "cuda")]
use burn_cuda::Cuda;
use dormouse_gdn2::forward::chunk_wy_forward;
#[cfg(feature = "cuda")]
use dormouse_gdn2::kernel::fused_recurrent_cube::cuda::fused_step;
#[cfg(feature = "cuda")]
use dormouse_gdn2::{fused_recurrent_forward, GatedDeltaNet2, Gdn2Config, Gdn2Mode, Gdn2State};
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
type B = Cuda;

#[cfg(feature = "cuda")]
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
fn cfg(hs: usize, nh: usize, hk: usize, ev: f32, nvh: Option<usize>, cs: usize) -> Gdn2Config {
    Gdn2Config {
        hidden_size: hs,
        num_heads: nh,
        head_dim: hk,
        expand_v: ev,
        num_v_heads: nvh,
        chunk_size: cs,
        mode: Gdn2Mode::Chunk,
        ..Default::default()
    }
}

#[cfg(feature = "cuda")]
type BenchCfg = (
    &'static str,
    usize,
    usize,
    usize,
    f32,
    Option<usize>,
    usize,
    usize,
    usize,
    usize,
);

#[cfg(feature = "cuda")]
const CFGS: &[BenchCfg] = &[
    ("tiny", 64, 2, 32, 1.0, None, 64, 1, 1, 64),
    ("small", 128, 4, 32, 1.0, None, 64, 1, 1, 256),
    ("med", 256, 4, 64, 1.0, None, 64, 1, 128, 1024),
    ("med_f", 256, 4, 64, 1.0, None, 64, 1, 256, 1024),
    ("large", 512, 8, 64, 1.0, None, 64, 1, 256, 2048),
    ("xl", 1024, 8, 128, 1.0, None, 64, 1, 512, 4096),
    ("gva_4x", 512, 4, 64, 1.0, Some(16), 64, 1, 256, 2048),
    ("exp_v2", 512, 8, 64, 2.0, None, 64, 1, 256, 2048),
];

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn bench_cuda() {
    let dev = Default::default();
    println!("\n{:=^80}", "");
    println!("  dormouse-gdn2 CUDA BENCHMARK - RTX 5060 Ti");
    println!("{:=^80}", "");

    println!("\n── 1. PROJECTIONS ONLY ──");
    println!("{:<12} {:>10} {:>10}", "config", "project_us", "B*Sq");

    for &(name, hs, nh, hk, ev, nvh, _cs, B, Sq, Sk) in CFGS {
        let c = cfg(hs, nh, hk, ev, nvh, 64);
        let m = GatedDeltaNet2::new(&c, &dev);
        let x = Tensor::<3>::random([B, Sk, hs], Distribution::Normal(0.0, 1.0), &dev);

        let proj_time = time_it(50, {
            let m = m.clone();
            let x = x.clone();
            move || {
                let _ = m.project(x.clone(), None).0.q.into_data();
            }
        });

        println!("{:<12} {:>10.0} {:>10}", name, proj_time * 1e6, B * Sk);
    }

    println!("\n── 2. CHUNK FORWARD ──");
    println!(
        "{:<12} {:>10} {:>10} {:>6}",
        "config", "chunk_us", "seq", "tok/s"
    );
    for &(name, hs, nh, hk, ev, nvh, cs, B, Sq, Sk) in CFGS {
        if Sk < 128 {
            continue;
        }
        let c = cfg(hs, nh, hk, ev, nvh, cs);
        let m = GatedDeltaNet2::new(&c, &dev);
        let x = Tensor::<3>::random([B, Sk, hs], Distribution::Normal(0.0, 1.0), &dev);

        let runs = if Sk <= 1024 { 20 } else { 5 };
        let t = time_it(runs, {
            let m = m.clone();
            let x = x.clone();
            move || {
                let _ = m.forward_train::<Bare>(x.clone()).into_data();
            }
        });

        let tok_s = (B * Sk) as f64 / t;
        println!("{:<12} {:>10.0} {:>10} {:>8.0}", name, t * 1e6, Sk, tok_s);
    }

    println!("\n── 3. FUSED RECURRENT FORWARD ──");
    println!(
        "{:<12} {:>10} {:>10} {:>6}",
        "config", "fused_us", "seq", "tok/s"
    );
    for &(name, hs, nh, hk, ev, nvh, cs, B, _Sq, Sk) in CFGS {
        if Sk < 128 {
            continue;
        }
        let c = cfg(hs, nh, hk, ev, nvh, cs);
        c.validate();
        let mut c2 = c.clone();
        c2.mode = Gdn2Mode::FusedRecurrent;
        let m = GatedDeltaNet2::new(&c2, &dev);
        let x = Tensor::<3>::random([B, Sk, hs], Distribution::Normal(0.0, 1.0), &dev);

        let runs = if Sk <= 1024 { 5 } else { 2 };
        let t = time_it(runs, {
            let m = m.clone();
            let x = x.clone();
            move || {
                let _ = m.forward_train::<Bare>(x.clone()).into_data();
            }
        });

        let tok_s = (B * Sk) as f64 / t;
        println!("{:<12} {:>10.0} {:>10} {:>8.0}", name, t * 1e6, Sk, tok_s);
    }

    println!("\n── 4. DECODE (single token, fused kernel) ──");
    println!(
        "{:<12} {:>10} {:>8} {:>6}",
        "config", "step_us", "tokens", "tok/s"
    );
    type Bare = dormouse_gdn2::CudaBare;
    for &(name, hs, nh, hk, ev, nvh, cs, B, _Sq, _Sk) in CFGS {
        let c = cfg(hs, nh, hk, ev, nvh, cs);
        let mut c2 = c.clone();
        c2.mode = Gdn2Mode::FusedRecurrent;
        let m = GatedDeltaNet2::new(&c2, &dev);
        let x = Tensor::<3>::random([B, 1, hs], Distribution::Normal(0.0, 1.0), &dev);

        let steps = 200;
        let t = time_it(steps, {
            let m = m.clone();
            let x = x.clone();
            move || {
                let mut state: Option<Gdn2State> = None;
                let _ = m.forward::<Bare>(x.clone(), &mut state, true).into_data();
            }
        });

        let tok_s = B as f64 / t;
        println!("{:<12} {:>10.0} {:>8} {:>8.0}", name, t * 1e6, "1", tok_s);
    }

    println!();
}

/// The fused decode kernel must reproduce the token-by-token tensor path.
///
/// Runs the whole projected sequence through both paths per token and compares
/// the state and outputs. This exercises `fused_step` directly (not just the
/// dispatch), so a silent fallback cannot mask a broken kernel.
#[cfg(feature = "cuda")]
#[test]
fn fused_kernel_matches_tensor_path() {
    type Bare = dormouse_gdn2::CudaBare;
    let dev = Default::default();
    let m = GatedDeltaNet2::new(&cfg(256, 4, 32, 1.0, None, 64), &dev);
    let seq = 16;
    let x = Tensor::<3>::random([1, seq, 256], Distribution::Normal(0.0, 1.0), &dev);
    let (proj, _) = m.project(x.clone(), None);

    let hv = proj.hv;
    let hk = 32;
    let v_head = 32;
    let scale = (hk as f64).powf(-0.5);

    // Reference: full-sequence tensor scan.
    let (ref_outs, ref_state) = fused_recurrent_forward(
        proj.q.clone(),
        proj.k.clone(),
        proj.v.clone(),
        proj.g.clone(),
        proj.b.clone(),
        proj.w.clone(),
        Tensor::<4>::zeros([1, hv, hk, v_head], &dev),
        scale,
    );

    // Kernel path: token-by-token fused_step.
    let mut state = Tensor::<4>::zeros([1, hv, hk, v_head], &dev);
    for t in 0..seq {
        let sl = |tt: Tensor<4>| tt.slice_dim(2, t..t + 1).mul_scalar(1.0);
        let (o, ns) = fused_step::<Bare>(
            sl(proj.q.clone()),
            sl(proj.k.clone()),
            sl(proj.v.clone()),
            sl(proj.g.clone()),
            sl(proj.b.clone()),
            sl(proj.w.clone()),
            state.clone(),
            scale,
        )
        .expect("kernel should run on cuda");
        state = ns;
        let ref_o = ref_outs.clone().slice_dim(2, t..t + 1);
        let max_err = (o - ref_o)
            .abs()
            .max()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0];
        assert!(
            max_err < 1e-3,
            "kernel output diverges from tensor path at token {t}: {max_err}",
        );
    }
    let state_err = (state - ref_state)
        .abs()
        .max()
        .into_data()
        .try_to_vec::<f32>()
        .unwrap()[0];
    assert!(
        state_err < 1e-3,
        "kernel state diverges from tensor path: {state_err}",
    );
    println!("fused kernel vs tensor path: OK");
}
