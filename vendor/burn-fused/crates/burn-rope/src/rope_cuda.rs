//! Fused CUDA RoPE rotation kernel (burn 0.22 / cubecl 0.11).
//!
//! One thread per (i, h) pair inside a cube per (b, t/8): computes both
//! halves of the rotation (x1·c − x2·s, x1·s + x2·c) in one pass. Replaces
//! the tensor path's 4 muls + 2 adds + a [.., hd] cat per forward.

use burn::backend::Backend;
use burn::tensor::Tensor;
use std::any::Any;

#[cfg(feature = "cuda")]
use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn rope_kernel<F: Float>(
    x: &[F],       // [B, T, NH, HD]
    cos: &[F],     // [T, HD/2]
    sin: &[F],     // [T, HD/2]
    out: &mut [F], // [B, T, NH, HD]
    #[comptime] t_chunk: u32,
    #[comptime] t_len: u32,
    #[comptime] half: u32,
    #[comptime] nh: u32,
    #[comptime] h_chunk: u32,
) {
    // Cube per (b, t/t_chunk): UNIT_POS_X = rotation pair, UNIT_POS_Y = head
    // within h_chunk; loops over the t_chunk rows and the head groups inside.
    // The original one-cube-per-(b,t,nh) grid (262k cubes at b=4, t=2048,
    // nh=32) was launch-bound at ~14.7 ms; this grid is b * ceil(t/8) cubes.
    // Keep t_chunk comptime: unrolling beats the runtime loop here
    // (12.5 vs 18.2 ms on the bench config).
    let gc = CUBE_POS_X as usize; // b * (t / t_chunk) + t_group
    let i = UNIT_POS_X as usize;
    let h = UNIT_POS_Y as usize;
    let t_len = t_len as usize;
    let half = half as usize;
    let nh = nh as usize;
    let h_chunk = h_chunk as usize;
    let t_groups = t_len.div_ceil(t_chunk as usize);

    let b = gc / t_groups;
    let t0 = (gc % t_groups) * (t_chunk as usize);

    for tg in 0..(t_chunk as usize) {
        let tt = t0 + tg;
        if tt < t_len {
            let cbase = tt * half;
            let mut hc = h;
            while hc < nh {
                let row = (b * t_len + tt) * nh + hc;
                let base = row * (2 * half);
                if i < half {
                    let a = x[base + i];
                    let bb = x[base + half + i];
                    let c = cos[cbase + i];
                    let s = sin[cbase + i];
                    out[base + i] = a * c - bb * s;
                    out[base + half + i] = a * s + bb * c;
                }
                hc += h_chunk;
            }
        }
    }
}

#[cube(launch_unchecked)]
fn rope_backward_kernel<F: Float>(
    dout: &[F],   // [B, T, NH, HD]
    cos: &[F],    // [T, HD/2]
    sin: &[F],    // [T, HD/2]
    dx: &mut [F], // [B, T, NH, HD]
    #[comptime] t_chunk: u32,
    #[comptime] t_len: u32,
    #[comptime] half: u32,
    #[comptime] nh: u32,
    #[comptime] h_chunk: u32,
) {
    // Mirror of the forward kernel: cube per (b, t/t_chunk).
    let gc = CUBE_POS_X as usize; // b * (t / t_chunk) + t_group
    let i = UNIT_POS_X as usize;
    let h = UNIT_POS_Y as usize;
    let t_len = t_len as usize;
    let half = half as usize;
    let nh = nh as usize;
    let h_chunk = h_chunk as usize;
    let t_groups = t_len.div_ceil(t_chunk as usize);

    let b = gc / t_groups;
    let t0 = (gc % t_groups) * (t_chunk as usize);

    for tg in 0..(t_chunk as usize) {
        let tt = t0 + tg;
        if tt < t_len {
            let cbase = tt * half;
            let mut hc = h;
            while hc < nh {
                let row = (b * t_len + tt) * nh + hc;
                let base = row * (2 * half);
                if i < half {
                    let c = cos[cbase + i];
                    let s = sin[cbase + i];
                    let d1 = dout[base + i];
                    let d2 = dout[base + half + i];
                    dx[base + i] = d1 * c + d2 * s;
                    dx[base + half + i] = F::new(0.0_f32) - d1 * s + d2 * c;
                }
                hc += h_chunk;
            }
        }
    }
}

/// Fused RoPE backward on the bare CUDA backend.
#[cfg(all(feature = "cuda", feature = "autodiff"))]
fn rope_backward_cuda<B: Backend>(
    d_out: &Tensor<4>,
    cos: &Tensor<2>,
    sin: &Tensor<2>,
) -> Option<Tensor<4>>
where
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    use burn_cubecl::tensor::CubeTensor;
    let cube = |t: &Tensor<4>| -> Option<CubeTensor> {
        let prim = t.clone().try_into_primitive::<B>().ok()?;
        let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
        Some(c.clone())
    };
    let cube2 = |t: &Tensor<2>| -> Option<CubeTensor> {
        let prim = t.clone().try_into_primitive::<B>().ok()?;
        let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
        Some(c.clone())
    };
    let [b, t, nh, hd] = d_out.dims();
    let half = hd / 2;
    let dc = cube(d_out)?;
    let cos_c = cube2(cos)?;
    let sin_c = cube2(sin)?;
    let client = dc.client.clone();

    let dx = Tensor::<4>::empty([b, t, nh, hd], &d_out.device());
    let dxc = cube(&dx)?;

    let h_chunk = ((nh as u32).min(1024 / (half as u32).max(1))).max(1);
    let t_chunk = 8u32; // cubes: b * ceil(t/8)
    let t_groups = t.div_ceil(t_chunk as usize);
    let cube_dim = CubeDim::new_3d(half as u32, h_chunk, 1);
    let cube_count = CubeCount::Static((b * t_groups) as u32, 1, 1);
    unsafe {
        rope_backward_kernel::launch_unchecked::<f32>(
            &client,
            cube_count,
            cube_dim,
            BufferArg::from_raw_parts(dc.handle, b * t * nh * hd),
            BufferArg::from_raw_parts(cos_c.handle, t * half),
            BufferArg::from_raw_parts(sin_c.handle, t * half),
            BufferArg::from_raw_parts(dxc.handle, b * t * nh * hd),
            t_chunk,
            t as u32,
            half as u32,
            nh as u32,
            h_chunk,
        );
    }
    Some(dx)
}

/// Fused RoPE on the bare CUDA backend. Returns `None` otherwise.
pub fn rope_cuda<B: Backend>(x: Tensor<4>, cos: Tensor<2>, sin: Tensor<2>) -> Option<Tensor<4>>
where
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    use burn_cubecl::tensor::CubeTensor;
    let cube = |t: &Tensor<4>| -> Option<CubeTensor> {
        let prim = t.clone().try_into_primitive::<B>().ok()?;
        let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
        Some(c.clone())
    };
    let cube2 = |t: &Tensor<2>| -> Option<CubeTensor> {
        let prim = t.clone().try_into_primitive::<B>().ok()?;
        let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
        Some(c.clone())
    };
    let [b, t, nh, hd] = x.dims();
    let half = hd / 2;
    let x_c = cube(&x)?;
    let cos_c = cube2(&cos)?;
    let sin_c = cube2(&sin)?;
    let client = x_c.client.clone();

    let out = Tensor::<4>::empty([b, t, nh, hd], &x.device());
    let out_c = cube(&out)?;

    let h_chunk = ((nh as u32).min(1024 / (half as u32).max(1))).max(1);
    let t_chunk = 8u32; // cubes: b * ceil(t/8)
    let t_groups = t.div_ceil(t_chunk as usize);
    let cube_dim = CubeDim::new_3d(half as u32, h_chunk, 1);
    let cube_count = CubeCount::Static((b * t_groups) as u32, 1, 1);
    unsafe {
        rope_kernel::launch_unchecked::<f32>(
            &client,
            cube_count,
            cube_dim,
            BufferArg::from_raw_parts(x_c.handle, b * t * nh * hd),
            BufferArg::from_raw_parts(cos_c.handle, t * half),
            BufferArg::from_raw_parts(sin_c.handle, t * half),
            BufferArg::from_raw_parts(out_c.handle, b * t * nh * hd),
            t_chunk,
            t as u32,
            half as u32,
            nh as u32,
            h_chunk,
        );
    }
    Some(out)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::{apply_rope_4d, precompute_freqs};
    use burn::tensor::{Distribution, Tensor};
    use burn_cuda::Cuda;
    use std::time::Instant;

    fn rope_ref<B: Backend>(x: Tensor<4>, cos: Tensor<2>, sin: Tensor<2>) -> Tensor<4>
    where
        burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
    {
        let [b, t, nh, hd] = x.dims();
        let half = hd / 2;
        let x1 = x.clone().slice([0..b, 0..t, 0..nh, 0..half]);
        let x2 = x.slice([0..b, 0..t, 0..nh, half..hd]);
        let c = cos.slice([0..t, 0..half]).reshape([1, t, 1, half]);
        let s = sin.slice([0..t, 0..half]).reshape([1, t, 1, half]);
        let rot1 = x1.clone().mul(c.clone()).sub(x2.clone().mul(s.clone()));
        let rot2 = x1.mul(s).add(x2.mul(c));
        Tensor::cat(vec![rot1, rot2], 3).reshape([b, t, nh, hd])
    }

    #[test]
    #[ignore = "needs a live CUDA context: BURN_DEVICE=cuda cargo test -p burn-rope --features cuda -- --ignored"]
    fn rope_matches_ref() {
        // NOT a silent skip. The old `return` here reported PASS with zero
        // assertions on every CPU run, which is how a broken kernel stayed
        // green (ADR-0011). Now the test is `#[ignore]`d, so `cargo test`
        // prints `ignored` with the reason, and asking for it explicitly
        // without a GPU FAILS instead of passing.
        assert_eq!(
            std::env::var("BURN_DEVICE").as_deref(),
            Ok("cuda"),
            "rope_matches_ref needs BURN_DEVICE=cuda; without it there is no fused kernel to compare"
        );
        let device = Default::default();
        for hd in [64usize, 128, 256] {
            for t in [1usize, 17, 64] {
                let b = 2;
                let nh = 4;
                let x: Tensor<4> =
                    Tensor::<4>::random([b, t, nh, hd], Distribution::Normal(0.0, 1.0), &device);
                let (cos, sin) = precompute_freqs(hd, t, 10000.0, &device);
                let r = rope_ref::<Cuda>(x.clone(), cos.clone(), sin.clone());
                let k = apply_rope_4d::<Cuda>(x, cos, sin);
                let diff: f32 = (r - k).abs().max().into_scalar();
                assert!(diff < 1e-3, "hd={hd} t={t} max diff {diff}");
            }
        }
    }

    #[test]
    #[ignore]
    fn rope_bench() {
        let device = Default::default();
        let (b, t, nh, hd) = (4usize, 2048, 32, 128);
        let x: Tensor<4> =
            Tensor::<4>::random([b, t, nh, hd], Distribution::Normal(0.0, 1.0), &device);
        let (cos, sin) = precompute_freqs(hd, t, 10000.0, &device);
        for _ in 0..5 {
            let _ = apply_rope_4d::<Cuda>(x.clone(), cos.clone(), sin.clone());
        }
        let t0 = Instant::now();
        for _ in 0..50 {
            let out = apply_rope_4d::<Cuda>(x.clone(), cos.clone(), sin.clone());
            let _: f32 = out.sum().into_scalar(); // sync: kernel launch is async
        }
        let tk = t0.elapsed() / 50;
        let t0 = Instant::now();
        for _ in 0..10 {
            let out = rope_ref::<Cuda>(x.clone(), cos.clone(), sin.clone());
            let _: f32 = out.sum().into_scalar();
        }
        let tr = t0.elapsed() / 10;
        println!(
            "b={b} t={t} nh={nh} hd={hd}: kernel {:?} vs tensor {:?} ({:.0}x)",
            tk,
            tr,
            tr.as_secs_f64() / tk.as_secs_f64()
        );
    }
}

// ---- seam counters (ADR-0019) ----
//
// ENTRY is incremented AFTER the strategy downcasts — the gate that was
// hardcoded to `NoCheckpointing`, so dormouse's
// `Autodiff<CudaBare, BalancedCheckpointing>` never got here. A counter
// before that gate would count interest, not arrivals (`f737710`).
#[cfg(feature = "autodiff")]
static ENTRY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(all(feature = "autodiff", feature = "cuda"))]
static FWD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(all(feature = "autodiff", feature = "cuda"))]
static BWD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(entry, fused_forward, fused_backward)` since [`reset_seam_counts`].
pub fn seam_counts() -> Option<(u64, u64, u64)> {
    #[cfg(feature = "autodiff")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        #[cfg(feature = "cuda")]
        return Some((ENTRY.load(Relaxed), FWD.load(Relaxed), BWD.load(Relaxed)));
        #[cfg(not(feature = "cuda"))]
        return Some((ENTRY.load(Relaxed), 0, 0));
    }
    #[cfg(not(feature = "autodiff"))]
    None
}

/// Zero the seam counters.
pub fn reset_seam_counts() {
    #[cfg(feature = "autodiff")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        ENTRY.store(0, Relaxed);
        #[cfg(feature = "cuda")]
        {
            FWD.store(0, Relaxed);
            BWD.store(0, Relaxed);
        }
    }
}

#[cfg(feature = "autodiff")]
fn note_entry_reached() {
    ENTRY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "cuda")]
fn note_fused_forward() {
    FWD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "cuda")]
fn note_fused_backward() {
    BWD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "autodiff")]
mod ad {
    use super::*;
    use burn::backend::DispatchKindConversion;
    use burn::tensor::{DispatchTensor, Tensor};
    use burn_autodiff::checkpoint::base::Checkpointer;
    use burn_autodiff::checkpoint::strategy::{CheckpointStrategy, NoCheckpointing};
    use burn_autodiff::grads::Gradients;
    use burn_autodiff::ops::{Backward, Ops, OpsKind};
    use burn_autodiff::Autodiff;

    // Only x is a tracked parent: cos/sin are precomputed freqs and treated as
    // constants (fused path). Learnable freqs fall back to the tensor path.
    const N_PARENTS: usize = 1;

    #[derive(Debug)]
    struct RopeOp;

    impl<B: Backend> Backward<B, N_PARENTS> for RopeOp
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        type State = ();

        fn backward(
            self,
            ops: Ops<Self::State, N_PARENTS>,
            grads: &mut Gradients,
            checkpointer: &mut Checkpointer,
        ) {
            let x = Tensor::<4>::from_primitive::<B>(checkpointer.retrieve_node_output(
                ops.parents[0].as_ref().expect("rope input checkpointed").id,
            ));
            let d_out = Tensor::from_primitive::<B>(grads.consume::<B>(&ops.node));
            let [b, t, nh, hd] = x.dims();
            let half = hd / 2;

            // cos/sin recomputed identically to the forward (precompute_freqs
            // is deterministic on the same device).
            let (cos, sin) = crate::precompute_freqs(hd, t, 10000.0, &x.device());

            #[cfg(not(feature = "cuda"))]
            let _ = (&x, &d_out);

            #[cfg(feature = "cuda")]
            {
                type CudaBare = burn_cubecl::CubeBackend;
                // `B` is the INNER backend here (`OpsPrep::finish` returns
                // `AutodiffTensor<B>`; the result goes to
                // `Tensor::from_primitive::<Autodiff<Inner, S>>`), so this
                // gate asks the right question. The entry below was the one
                // pinned to `NoCheckpointing`.
                if std::any::TypeId::of::<B>() == std::any::TypeId::of::<CudaBare>() {
                    if let Some(dx) = super::rope_backward_cuda::<B>(&d_out, &cos, &sin) {
                        note_fused_backward();
                        grads.register::<B>(
                            ops.parents[0].clone().unwrap().id,
                            dx.try_into_primitive::<B>().unwrap(),
                        );
                        return;
                    }
                }
            }

            let d1 = d_out.clone().slice([0..b, 0..t, 0..nh, 0..half]);
            let d2 = d_out.slice([0..b, 0..t, 0..nh, half..hd]);
            let x1 = x.clone().slice([0..b, 0..t, 0..nh, 0..half]);
            let x2 = x.clone().slice([0..b, 0..t, 0..nh, half..hd]);
            let _ = (&x1, &x2);
            let c = cos.slice([0..t, 0..half]).reshape([1, t, 1, half]);
            let s = sin.slice([0..t, 0..half]).reshape([1, t, 1, half]);
            let dx1 = d1.clone() * c.clone() + d2.clone() * s.clone();
            let dx2 = d1.neg() * s + d2 * c;
            let dx = Tensor::cat(vec![dx1, dx2], 3).reshape([b, t, nh, hd]);
            grads.register::<B>(
                ops.parents[0].clone().unwrap().id,
                dx.try_into_primitive::<B>().unwrap(),
            );
        }
    }

    /// Fused RoPE with exact backward on `Autodiff<Inner>`.
    pub fn rope_autodiff_s<Inner: Backend, S: CheckpointStrategy>(
        x: Tensor<4>,
        cos: Tensor<2>,
        sin: Tensor<2>,
    ) -> Option<Tensor<4>>
    where
        DispatchTensor: DispatchKindConversion<Autodiff<Inner, S>> + DispatchKindConversion<Inner>,
    {
        let xa = x.try_into_primitive::<Autodiff<Inner, S>>().ok()?;
        let ca = cos.try_into_primitive::<Autodiff<Inner, S>>().ok()?;
        let sa = sin.try_into_primitive::<Autodiff<Inner, S>>().ok()?;
        note_entry_reached();

        let x_t = Tensor::<4>::from_primitive::<Inner>(xa.primitive().clone());
        let c_t = Tensor::<2>::from_primitive::<Inner>(ca.primitive().clone());
        let s_t = Tensor::<2>::from_primitive::<Inner>(sa.primitive().clone());

        let out_t = {
            #[cfg(feature = "cuda")]
            {
                type CudaBare = burn_cubecl::CubeBackend;
                if std::any::TypeId::of::<Inner>() == std::any::TypeId::of::<CudaBare>() {
                    if let Some(o) =
                        super::rope_cuda::<Inner>(x_t.clone(), c_t.clone(), s_t.clone())
                    {
                        note_fused_forward();
                        o
                    } else {
                        super::rope_tensor::<Inner>(x_t.clone(), c_t.clone(), s_t.clone())
                    }
                } else {
                    super::rope_tensor::<Inner>(x_t, c_t, s_t)
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                super::rope_tensor::<Inner>(x_t, c_t, s_t)
            }
        };

        let out_prim = out_t.try_into_primitive::<Inner>().unwrap();
        let nodes = [xa.node()];
        let prep = RopeOp.prepare::<S>(nodes);
        let out_adt = match prep.compute_bound().stateful() {
            OpsKind::Tracked(mut prep) => {
                let _ids = [Some(prep.checkpoint(&xa))];
                prep.finish((), out_prim)
            }
            OpsKind::UnTracked(prep) => prep.finish(out_prim),
        };
        Some(Tensor::from_primitive::<Autodiff<Inner, S>>(out_adt))
    }

    /// [`rope_autodiff_s`] on the default (no-checkpointing) strategy.
    pub fn rope_autodiff<Inner: Backend>(
        x: Tensor<4>,
        cos: Tensor<2>,
        sin: Tensor<2>,
    ) -> Option<Tensor<4>>
    where
        DispatchTensor: DispatchKindConversion<Autodiff<Inner>> + DispatchKindConversion<Inner>,
    {
        rope_autodiff_s::<Inner, NoCheckpointing>(x, cos, sin)
    }
}

#[cfg(feature = "autodiff")]
pub use ad::{rope_autodiff, rope_autodiff_s};

/// Pure tensor-path RoPE (shared forward for the autodiff fallback).
pub fn rope_tensor<B: Backend>(x: Tensor<4>, cos: Tensor<2>, sin: Tensor<2>) -> Tensor<4>
where
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    let [b, t, nh, hd] = x.dims();
    let half = hd / 2;
    let x1 = x.clone().slice([0..b, 0..t, 0..nh, 0..half]);
    let x2 = x.slice([0..b, 0..t, 0..nh, half..hd]);
    let c = cos.slice([0..t, 0..half]).reshape([1, t, 1, half]);
    let s = sin.slice([0..t, 0..half]).reshape([1, t, 1, half]);
    let rot1 = x1.clone().mul(c.clone()).sub(x2.clone().mul(s.clone()));
    let rot2 = x1.mul(s).add(x2.mul(c));
    Tensor::cat(vec![rot1, rot2], 3).reshape([b, t, nh, hd])
}

#[cfg(all(test, feature = "autodiff", feature = "cuda"))]
mod seam_tests {
    //! The proof that the strategy gate is strategy-AGNOSTIC. It needs the
    //! `cuda` feature to COMPILE (this file's kernels are not feature-gated
    //! individually) but uses only the ndarray backend at RUNTIME: no device,
    //! no kernel launch. It asserts a caller on
    //! `Autodiff<Inner, BalancedCheckpointing>` — dormouse's backend — gets
    //! PAST the seam downcast, and that the old `NoCheckpointing`-only
    //! spelling does not. Revert `rope_autodiff_s` to `Autodiff<Inner>` and the
    //! first assertion goes red: it is the assertion, not the comment.
    use super::*;
    use burn::backend::DispatchKindConversion;
    use burn::tensor::{Device, DispatchTensor};
    use burn_autodiff::Autodiff as Ad;
    use burn_autodiff::checkpoint::strategy::{
        BalancedCheckpointing, CheckpointStrategy, NoCheckpointing,
    };

    type Nd = burn_ndarray::NdArray;

    #[test]
    fn balanced_checkpointing_reaches_the_seam_and_the_legacy_entry_does_not() {
        fn reach<S: CheckpointStrategy>(x: &Tensor<4>, c: &Tensor<2>, s: &Tensor<2>)
        -> Option<Tensor<4>>
        where
            DispatchTensor: DispatchKindConversion<Ad<Nd, S>> + DispatchKindConversion<Nd>,
        {
            rope_autodiff_s::<Nd, S>(x.clone(), c.clone(), s.clone())
        }

        let dev = Device::ndarray().autodiff().gradient_checkpointing();
        let x = Tensor::<4>::ones([1, 8, 2, 4], &dev);
        let (c, s) = crate::precompute_freqs(4, 8, 10000.0, &dev);

        reset_seam_counts();
        let base = seam_counts().expect("autodiff feature is on in this test").0;

        assert!(reach::<BalancedCheckpointing>(&x, &c, &s).is_some());
        assert_eq!(
            seam_counts().expect("counters").0,
            base + 1,
            "a BalancedCheckpointing caller must get past the seam downcast"
        );

        assert!(
            rope_autodiff::<Nd>(x.clone(), c.clone(), s.clone()).is_none(),
            "on a Balanced tensor the NoCheckpointing entry must refuse"
        );
        assert_eq!(
            seam_counts().expect("counters").0,
            base + 1,
            "the refusing entry must not have counted a reach"
        );

        // The default strategy still works, and the cross-check refuses, so
        // the gate is real rather than always-true.
        let plain = Device::ndarray().autodiff();
        let xp = Tensor::<4>::ones([1, 8, 2, 4], &plain);
        let (cp, sp) = crate::precompute_freqs(4, 8, 10000.0, &plain);
        assert!(reach::<NoCheckpointing>(&xp, &cp, &sp).is_some());
        assert!(reach::<BalancedCheckpointing>(&xp, &cp, &sp).is_none());
    }
}

#[cfg(all(test, feature = "autodiff", feature = "cuda"))]
mod ad_tests {
    use super::*;
    use crate::precompute_freqs;
    use burn::tensor::{Device, Distribution, Tensor};

    type CudaBare = burn_cubecl::CubeBackend;

    fn to_host(t: Tensor<4>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn fused_backward_matches_tensor_backward() {
        let dev = Device::default().autodiff();
        let (b, t, nh, hd) = (2usize, 7usize, 4usize, 64usize);
        let x = Tensor::<4>::random([b, t, nh, hd], Distribution::Normal(0.0, 1.0), &dev);
        let (cos, sin) = precompute_freqs(hd, t, 10000.0, &dev);

        // fused op graph
        let xf = x.clone().require_grad();
        let outf = rope_autodiff::<CudaBare>(xf.clone(), cos.clone(), sin.clone()).unwrap();
        let loss_f = outf.powf_scalar(2.0).sum();
        let grads_f = loss_f.backward();
        let dxf = xf.grad(&grads_f).unwrap();

        // plain tensor path graph (through the dispatched lib on AD device)
        let xt = x.clone().require_grad();
        let outt = crate::apply_rope_4d::<burn_autodiff::Autodiff<CudaBare>>(
            xt.clone(),
            cos.clone(),
            sin.clone(),
        );
        let loss_t = outt.powf_scalar(2.0).sum();
        let grads_t = loss_t.backward();
        let dxt = xt.grad(&grads_t).unwrap();

        let md = maxdiff(&to_host(dxf), &to_host(dxt));
        assert!(md < 1e-3, "dx maxdiff {md}");
    }
}

#[cfg(all(test, feature = "autodiff", feature = "cuda"))]
mod fd_tests {
    use super::*;
    use crate::precompute_freqs;
    use burn::tensor::{Device, Distribution, Tensor};

    type CudaBare = burn_cubecl::CubeBackend;

    fn to_host<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn rope_loss(x: &Tensor<4>, cos: &Tensor<2>, sin: &Tensor<2>) -> f32 {
        let [b, t, nh, hd] = x.dims();
        let p = hd.next_power_of_two().max(2);
        let r = crate::apply_rope_4d::<CudaBare>(x.clone(), cos.clone(), sin.clone());
        r.powf_scalar(2.0).sum().into_scalar::<f32>()
    }

    #[test]
    fn rope_grad_matches_finite_difference() {
        let dev = Device::default();
        let adev = Device::default().autodiff();
        let (b, t, nh, hd) = (1usize, 3usize, 2usize, 8usize);
        let x = Tensor::<4>::random([b, t, nh, hd], Distribution::Normal(0.0, 1.0), &dev);
        let (cos, sin) = precompute_freqs(hd, t, 10000.0, &dev);
        let data = x.clone().into_data();

        let xf = Tensor::<4>::from_data(data.clone(), &adev).require_grad();
        // cos/sin must live on the autodiff device for the fused dispatch
        let cos_a = Tensor::<2>::from_data(cos.clone().into_data(), &adev);
        let sin_a = Tensor::<2>::from_data(sin.clone().into_data(), &adev);
        let out = crate::apply_rope_4d::<CudaBare>(xf.clone(), cos_a, sin_a);
        let grads = out.powf_scalar(2.0).sum().backward();
        let analytic = to_host(xf.grad(&grads).unwrap());

        let eps = 1e-3f32;
        let total = b * t * nh * hd;
        let mut fd = vec![0.0f32; total];
        for i in 0..total {
            let mut plus = data.clone();
            let pv = f32::from_le_bytes(plus.bytes[i * 4..i * 4 + 4].try_into().unwrap()) + eps;
            plus.bytes[i * 4..i * 4 + 4].copy_from_slice(&pv.to_le_bytes());
            let mut minus = data.clone();
            let mv = f32::from_le_bytes(minus.bytes[i * 4..i * 4 + 4].try_into().unwrap()) - eps;
            minus.bytes[i * 4..i * 4 + 4].copy_from_slice(&mv.to_le_bytes());
            let xp = Tensor::<4>::from_data(plus, &dev);
            let xm = Tensor::<4>::from_data(minus, &dev);
            fd[i] = (rope_loss(&xp, &cos, &sin) - rope_loss(&xm, &cos, &sin)) / (2.0 * eps);
        }
        for i in 0..total {
            // pass on absolute OR relative error: the rel metric blows up when
            // a gradient element is near zero (random data), which made this
            // test flaky
            let abs = (analytic[i] - fd[i]).abs();
            let rel = abs / (fd[i].abs() + 1e-6);
            assert!(
                abs < 1e-2 || rel < 5e-2,
                "idx {i}: analytic {} vs fd {} (rel {rel})",
                analytic[i],
                fd[i]
            );
        }
    }
}
