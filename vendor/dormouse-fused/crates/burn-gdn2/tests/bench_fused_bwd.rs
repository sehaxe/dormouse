#![cfg(all(feature = "cuda", feature = "autodiff"))]
//! Fused-backward attention-core bench (chunk 16): op train vs plain autodiff.
//!
//! # Read the seam line before quoting a number from here
//!
//! This bench times the CUSTOM NODE's forward and its `tt - tf` difference, so
//! what it prices is a backward that a real training step may never take:
//! `chunk_wy_forward_autodiff_s` DECLINES whenever no input is a
//! `require_grad` leaf, which is the case for every input the trainer produces
//! (`KdaModule::project` output is a `GradInBackward` intermediate), and then
//! `chunk_dispatch` takes the ops path and burn builds the graph itself. The
//! bench feeds `.require_grad()` leaves, so the node is built and the numbers
//! are real for THAT arm.
//!
//! Worse, and measured: that arm is not a correct gradient. On hardware
//! 2026-09-28 the fused adjoint agrees with the ops path to ~3e-7 for q, v, b,
//! w and the state at any length, and to 2.4e-7 for k at ONE chunk - and from
//! two chunks on its `d_k` is off by 1.8e-1..2.8e-1 and its `d_g` by
//! 1.7e-2..3.2e-2 (table in `tests/autodiff_cuda_gate.rs`). **A speed number
//! for a wrong gradient is not a speed number.** The bench now prints the seam
//! counters on every shape so a reader can see which arm it timed, and the
//! `#[ignore]` reason says the rest.

use burn::tensor::{Distribution, Tensor};
use burn_gdn2::CudaBare;

fn time_it(runs: usize, mut f: impl FnMut()) -> f64 {
    let t0 = std::time::Instant::now();
    for _ in 0..runs {
        f();
    }
    t0.elapsed().as_secs_f64() / runs as f64
}

#[test]
#[ignore = "prices the custom node's backward, which the trainer never takes (chunk_wy_forward_autodiff_s declines on every real input) AND whose d_k/d_g are wrong past one chunk; kept because it is the only instrument for the raw-kernel cost"]
fn bench_fused_bwd() {
    let dev: burn::tensor::Device = Default::default();
    let ad_dev = burn::tensor::Device::autodiff(dev.clone());
    println!("── fused backward (chunk 16) attention core: op train ──");
    for (d, h, hk, t) in [
        (512usize, 4usize, 64usize, 1024usize),
        (1024, 8, 64, 2048),
        (2048, 8, 128, 4096),
        (4096, 8, 128, 8192),
    ] {
        let mk =
            |shape: [usize; 4]| Tensor::<4>::random(shape, Distribution::Normal(0.0, 0.1), &ad_dev);
        let q = mk([1, h, t, hk]).require_grad();
        let k = mk([1, h, t, hk]).require_grad();
        let v = mk([1, h, t, hk]).require_grad();
        let g = mk([1, h, t, hk]).require_grad();
        let b = mk([1, h, t, hk]).require_grad();
        let w = mk([1, h, t, hk]).require_grad();
        let s = mk([1, h, hk, hk]).require_grad();

        // warmup
        let (o, _) = burn_gdn2::chunk_wy_forward_autodiff::<CudaBare>(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            b.clone(),
            w.clone(),
            s.clone(),
            1.0,
            16,
        )
        .unwrap();
        let loss = o.powf_scalar(2.0).sum();
        let gs = loss.backward();
        let _ = gs;
        let runs = if t <= 2048 { 5 } else { 3 };
        burn_gdn2::reset_fused_calls();
        let tf = time_it(runs, || {
            let (o2, _) = burn_gdn2::chunk_wy_forward_autodiff::<CudaBare>(
                q.clone(),
                k.clone(),
                v.clone(),
                g.clone(),
                b.clone(),
                w.clone(),
                s.clone(),
                1.0,
                16,
            )
            .unwrap();
            let _ = o2;
        });
        let tt = time_it(runs, || {
            let (o2, _) = burn_gdn2::chunk_wy_forward_autodiff::<CudaBare>(
                q.clone(),
                k.clone(),
                v.clone(),
                g.clone(),
                b.clone(),
                w.clone(),
                s.clone(),
                1.0,
                16,
            )
            .unwrap();
            let l = o2.powf_scalar(2.0).sum();
            let g2 = l.backward();
            let _ = g2;
        });
        let counts = burn_gdn2::seam_counts();
        println!(
            "d={d} h={h} hk={hk} T={t}: fwd {:.3} ms  train {:.3} ms  bwd {:.3} ms  \
             [seam asked={} fused_fwd={} fused_bwd={} declined={} ops_path={} custom_node_bwd={} \
             - this is the CUSTOM NODE, not the trainer's arm]",
            tf * 1e3,
            tt * 1e3,
            (tt - tf) * 1e3,
            counts.0,
            counts.1,
            counts.2,
            counts.3,
            counts.4,
            counts.5
        );
        // raw kernels: fused_chunk_forward_scratch + fused_chunk_backward
        {
            use burn::tensor::Tensor;
            let mk = |tt: Tensor<4>| -> Tensor<4> {
                Tensor::<4>::from_data(tt.clone().into_data(), &dev)
            };
            let qc = mk(q.clone());
            let kc = mk(k.clone());
            let vc = mk(v.clone());
            let gc = mk(g.clone());
            let bc = mk(b.clone());
            let wc = mk(w.clone());
            let sc = mk(s.clone());
            let (ro, _, io) =
                burn_gdn2::kernel::chunk_cube::cuda::fused_chunk_forward_scratch::<CudaBare>(
                    qc.clone(),
                    kc.clone(),
                    vc.clone(),
                    gc.clone(),
                    bc.clone(),
                    wc.clone(),
                    sc.clone(),
                    1.0,
                    16,
                )
                .unwrap();
            let do_c = Tensor::<4>::from_data(ro.clone().into_data(), &dev);
            let tf_raw = time_it(runs, || {
                let (ro2, _, io2) =
                    burn_gdn2::kernel::chunk_cube::cuda::fused_chunk_forward_scratch::<CudaBare>(
                        qc.clone(),
                        kc.clone(),
                        vc.clone(),
                        gc.clone(),
                        bc.clone(),
                        wc.clone(),
                        sc.clone(),
                        1.0,
                        16,
                    )
                    .unwrap();
                let _ = ro2;
                let fwd2 = burn_gdn2::kernel::chunk_adjoint_cube::cuda::FusedBackwardInputs {
                    m_inv: io2.m_inv,
                    aqk: io2.aqk,
                    qgt: io2.qgt,
                    glast: io2.glast,
                    v_new: io2.v_new,
                    states: io2.states,
                    w: io2.w,
                    u: io2.u,
                    gexp: io2.gexp,
                };
                let _ =
                    burn_gdn2::kernel::chunk_adjoint_cube::cuda::fused_chunk_backward::<CudaBare>(
                        &fwd2, &kc, &vc, &bc, &wc, &do_c, 1.0, 16,
                    )
                    .unwrap();
            });
            println!("    raw fwd+bwd: {:.3} ms", tf_raw * 1e3);
            let _ = io;
        }
    }
}
