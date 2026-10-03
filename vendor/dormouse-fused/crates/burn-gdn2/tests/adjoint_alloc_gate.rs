#![cfg(all(feature = "cuda", feature = "autodiff"))]
//! Alloc-traffic probe for the fused KDA adjoint (lane wt/adjoint-tune).
//!
//! One fused forward + one fused backward at the trainer's shape
//! (b=8, h=12, t=512, k=64, v=64, chunk=16), twice: the wall of the second
//! backward (autotune warm) and the live-bytes delta of `memory_usage()`
//! across it. Prints, never asserts — the number is the record.

use burn::tensor::{Distribution, Tensor};
use burn_gdn2::kernel::chunk_adjoint_cube::cuda::{fused_chunk_backward, FusedBackwardInputs};
use burn_gdn2::kernel::chunk_cube::cuda::fused_chunk_forward_scratch;
use burn_gdn2::CudaBare;
use std::any::Any;

fn client() -> burn_cubecl::cubecl::client::Client {
    let dev: burn::tensor::Device = Default::default();
    let t = Tensor::<1>::zeros([1], &dev);
    let prim = t
        .try_into_primitive::<CudaBare>()
        .ok()
        .expect("cuda tensor");
    let cube: &burn_cubecl::tensor::CubeTensor =
        (&prim as &dyn Any).downcast_ref().expect("CubeTensor");
    cube.client.clone()
}

#[test]
fn adjoint_alloc_and_time_probe() {
    let (b, h, t, k, v) = (8usize, 12usize, 512usize, 64usize, 64usize);
    let scale = (k as f64).powf(-0.5);
    let bare: burn::tensor::Device = Default::default();
    bare.seed(11);
    let r = |shape: [usize; 4], m: f64, s: f64| {
        Tensor::<4>::random(shape, Distribution::Normal(m, s), &bare)
    };
    let q = r([b, h, t, k], 0.0, 0.4);
    let kk = r([b, h, t, k], 0.0, 0.4);
    let vv = r([b, h, t, v], 0.0, 0.4);
    let g = r([b, h, t, k], -0.5, 0.2);
    let bb = r([b, h, t, k], 0.0, 0.3);
    let w = r([b, h, t, v], 0.0, 0.3);
    let state = r([b, h, k, v], 0.0, 0.2);
    let d_out = r([b, h, t, v], 0.0, 1.0);
    let c = client();

    let run_once = |tag: &str| {
        let (_fused_out, _s, io) = fused_chunk_forward_scratch::<CudaBare>(
            q.clone(),
            kk.clone(),
            vv.clone(),
            g.clone(),
            bb.clone(),
            w.clone(),
            state.clone(),
            scale,
            16,
        )
        .expect("fused forward");
        let fbi = FusedBackwardInputs {
            m_inv: io.m_inv,
            aqk: io.aqk,
            qgt: io.qgt,
            glast: io.glast,
            v_new: io.v_new,
            states: io.states,
            w: io.w,
            u: io.u,
            gexp: io.gexp,
        };
        let t0 = std::time::Instant::now();
        let fused = fused_chunk_backward::<CudaBare>(&fbi, &kk, &vv, &bb, &w, &d_out, scale, 16)
            .expect("fused adjoint");
        let _ = fused;
        // a sync submit: memory_usage blocks
        let u = c.memory_usage();
        println!(
            "{tag}: backward {:?}; allocs={} in_use={:.1}MB reserved={:.1}MB",
            t0.elapsed(),
            u.number_allocs,
            u.bytes_in_use as f64 / 1e6,
            u.bytes_reserved as f64 / 1e6
        );
    };

    run_once("warm-up");
    run_once("measured");
}
