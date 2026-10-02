//! Fused CUDA Fast Walsh-Hadamard Transform + quantize (BitNet v2), burn 0.22.
//!
//! One cube per input row; the log-p butterfly rounds run in shared memory
//! with `sync_cube()` between rounds (comptime loop: the round stride is a
//! power of two, so the butterfly index arithmetic folds to shifts — integer
//! division by non-powers of two is broken in the cubecl CUDA codegen).
//!
//! The normalized Hadamard matrix is symmetric and orthogonal, so the
//! backward of the FWT is the same transform on the gradient. The quantize
//! kernels use straight-through (identity) backward.

use burn::tensor::Tensor;
use std::any::Any;

#[cfg(feature = "cuda")]
use {burn_cubecl::tensor::CubeTensor, cubecl::prelude::*};

#[cfg(feature = "autodiff")]
use {
    burn::backend::{Backend, DispatchKindConversion},
    burn::tensor::DispatchTensor,
    burn_autodiff::checkpoint::base::Checkpointer,
    burn_autodiff::checkpoint::strategy::{CheckpointStrategy, NoCheckpointing},
    burn_autodiff::grads::Gradients,
    burn_autodiff::ops::{Backward, Ops, OpsKind},
    burn_autodiff::Autodiff,
};

// ---- seam counters (ADR-0019: "if you add an arm, add its counter") ----
//
// ENTRY is the one that matters. It is incremented AFTER the strategy
// downcast succeeded, i.e. AFTER the gate that used to be hardcoded to
// `NoCheckpointing`, so it answers "did a caller on a non-default
// checkpointing strategy get past the seam?" — a number that is 0 forever if
// the hardcode is still there. FWD/BWD count actual kernel launches.
#[cfg(feature = "autodiff")]
static ENTRY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "cuda")]
static FWD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "cuda")]
static BWD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(entry, fused_forward, fused_backward)` since [`reset_seam_counts`].
/// `None` when the `autodiff` feature is off: there is no seam to count then.
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

/// Fused zero-pad + Fast Walsh-Hadamard Transform: reads `[n, d]`, zero-pads
/// each row to `p` inside shared memory, runs the butterfly on the padded
/// row and writes `[n, p]` to a fresh buffer (the caller's tensor is never
/// aliased). Same math as the separate pad + in-place transform it replaces.
#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn fwt_pad_kernel<F: Float>(
    x: &[F],       // [n, d] input
    out: &mut [F], // [n, p] zero-padded, transformed output
    d: u32,
    #[comptime] p: u32,
    #[comptime] log2p: u32,
    #[comptime] threads: u32,
) {
    let row = CUBE_POS_X as usize;
    let t = UNIT_POS_X as usize;
    let p = p as usize;
    let d = d as usize;
    let threads = threads as usize;
    let sl = p / threads;
    let mut sh = Shared::<[F]>::new_slice(p);

    let base = row * p;
    for j in 0..sl {
        let col = t * sl + j;
        sh[col] = if col < d {
            x[row * d + col]
        } else {
            F::new(0.0_f32)
        };
    }

    for k in 0..log2p {
        let h = (1u32 << k) as usize;
        sync_cube();
        let start = t * sl;
        for j in 0..sl {
            let i = start + j;
            if (i & ((h << 1) - 1)) < h {
                let a = sh[i];
                let b = sh[i + h];
                sh[i] = a + b;
                sh[i + h] = a - b;
            }
        }
    }

    sync_cube();
    let inv = F::new(1.0_f32) / F::cast_from((p as f32).sqrt());
    for j in 0..sl {
        out[base + t * sl + j] = sh[t * sl + j] * inv;
    }
}

#[cfg(feature = "cuda")]
/// Elementwise quantize: q = clamp(round(x/scale), lo, hi)·scale (absmax 8-bit
/// rescales by 1/127). Row-per-cube grid (no integer division).
#[cube(launch_unchecked)]
fn quant_kernel<F: Float>(
    x: &[F],       // [n, p]
    scale: &[F],   // [n]
    out: &mut [F], // [n, p]
    #[comptime] p: u32,
    #[comptime] p_chunks: u32,
    #[comptime] bits: u32,
) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let p = p as usize;
    let base = row * p;
    let mut s = scale[row];
    if s < F::new(1.0e-12_f32) {
        s = F::new(1.0e-12_f32);
    }
    let inv = if bits == 8 {
        F::new(127.0_f32) / s
    } else {
        F::new(1.0_f32) / s
    };
    for chunk in 0..p_chunks {
        let col = (chunk as usize) * 256 + tid;
        if col < p {
            let mut q = (x[base + col] * inv).round();
            if bits == 8 {
                if q < F::new(-128.0_f32) {
                    q = F::new(-128.0_f32);
                }
                if q > F::new(127.0_f32) {
                    q = F::new(127.0_f32);
                }
            } else {
                if q < F::new(-8.0_f32) {
                    q = F::new(-8.0_f32);
                }
                if q > F::new(7.0_f32) {
                    q = F::new(7.0_f32);
                }
            }
            out[base + col] = if bits == 8 {
                q * s * (F::new(1.0_f32) / F::new(127.0_f32))
            } else {
                q * s
            };
        }
    }
}

#[cfg(feature = "cuda")]
type CudaBare = burn_cubecl::CubeBackend;

#[cfg(feature = "cuda")]
fn cube_of<const D: usize>(t: &Tensor<D>) -> Option<CubeTensor> {
    let prim = t.clone().try_into_primitive::<CudaBare>().ok()?;
    let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
    Some(c.clone())
}

#[cfg(feature = "cuda")]
fn cube_of_1(t: &Tensor<1>) -> Option<CubeTensor> {
    cube_of(t)
}

/// Pad `[n, d]` to `[n, p]` (p = next power of two) with a plain copy kernel —
/// burn's `Tensor::cat` is pathologically slow on CUDA. Used by the quantize
/// path; the FWT path folds the pad into `fwt_pad_kernel` instead.
#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn pad_kernel<F: Float>(
    x: &[F],       // [n, d]
    out: &mut [F], // [n, p]
    d: u32,
    #[comptime] p: u32,
    #[comptime] p_chunks: u32,
) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize;
    let p = p as usize;
    let base = row * p;
    for chunk in 0..p_chunks {
        let col = (chunk as usize) * 256 + tid;
        if col < d {
            out[base + col] = x[row * d + col];
        } else if col < p {
            out[base + col] = F::new(0.0_f32);
        }
    }
}

#[cfg(feature = "cuda")]
fn pad_cuda(x: &Tensor<2>, p: usize) -> Option<CubeTensor> {
    let [n, d] = x.dims();
    // always copy to a fresh buffer: the caller's tensor must never be aliased
    let xc = cube_of(x)?;
    let out = Tensor::<2>::empty([n, p], &x.device());
    let oc = cube_of(&out)?;
    let client = xc.client.clone();
    let chunks = (p as u32).div_ceil(256);
    unsafe {
        pad_kernel::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(n as u32, 1, 1),
            CubeDim::new_3d(256, 1, 1),
            BufferArg::from_raw_parts(xc.handle.clone(), n * d),
            BufferArg::from_raw_parts(oc.handle.clone(), n * p),
            d as u32,
            p as u32,
            chunks,
        );
    }
    Some(oc)
}

/// Fused zero-pad + FWT on the bare CUDA backend, one launch. `p` = padded
/// row width.
#[cfg(feature = "cuda")]
pub fn fwt_cuda(x: &Tensor<2>, p: usize) -> Option<Tensor<2>> {
    let [n, d] = x.dims();
    // shared-memory ceiling: 48 KB (RTX 3090) / 4 bytes per f32
    if p > 12000 || !p.is_power_of_two() {
        return None;
    }
    let xc = cube_of(x)?;
    let out = Tensor::<2>::empty([n, p], &x.device());
    let oc = cube_of(&out)?;
    let client = xc.client.clone();
    // burn's graph defers fill ops (random/zeros) until consumption; a raw
    // launch_unchecked bypasses the graph, so sync before the kernel reads x.
    {
        use burn::backend::Backend as _;
        let _ = <burn_cubecl::CubeBackend as burn::backend::Backend>::sync(&xc.device);
    }
    let threads = (p as u32).min(256);
    let log2p = p.ilog2();
    let cube_dim = CubeDim::new_3d(threads, 1, 1);
    let cube_count = CubeCount::Static(n as u32, 1, 1);
    unsafe {
        fwt_pad_kernel::launch_unchecked::<f32>(
            &client,
            cube_count,
            cube_dim,
            BufferArg::from_raw_parts(xc.handle.clone(), n * d),
            BufferArg::from_raw_parts(oc.handle.clone(), n * p),
            d as u32,
            p as u32,
            log2p,
            threads,
        );
    }
    let out = Tensor::from_primitive::<CudaBare>(oc);
    if p != d {
        Some(out.slice([0..n, 0..d]))
    } else {
        Some(out)
    }
}

/// Fused FWT backward = the same transform (normalized Hadamard is symmetric
/// and orthogonal).
#[cfg(all(feature = "cuda", feature = "autodiff"))]
fn fwt_backward_cuda(d_out: &Tensor<2>, p: usize) -> Option<Tensor<2>> {
    let [n, d] = d_out.dims();
    if p > 12000 || !p.is_power_of_two() {
        return None;
    }
    let xc = cube_of(d_out)?;
    let out = Tensor::<2>::empty([n, p], &d_out.device());
    let oc = cube_of(&out)?;
    let client = xc.client.clone();
    let threads = (p as u32).min(256);
    let log2p = p.ilog2();
    unsafe {
        fwt_pad_kernel::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(n as u32, 1, 1),
            CubeDim::new_3d(threads, 1, 1),
            BufferArg::from_raw_parts(xc.handle.clone(), n * d),
            BufferArg::from_raw_parts(oc.handle.clone(), n * p),
            d as u32,
            p as u32,
            log2p,
            threads,
        );
    }
    let dx = Tensor::from_primitive::<CudaBare>(oc);
    if p != d {
        Some(dx.slice([0..n, 0..d]))
    } else {
        Some(dx)
    }
}

/// Fused 8-bit / 4-bit quantize on the bare CUDA backend.
#[cfg(feature = "cuda")]
pub fn quant_cuda(x: &Tensor<2>, bits: usize) -> Option<Tensor<2>> {
    let [n, d] = x.dims();
    let p = d.next_power_of_two();
    let xc = pad_cuda(x, p)?;
    let scale = if bits == 8 {
        x.clone().abs().max_dim(1).clamp_min(1e-12).reshape([n])
    } else {
        x.clone().abs().mean_dim(1).clamp_min(1e-12).reshape([n])
    };
    let sc = cube_of_1(&scale)?;
    let client = xc.client.clone();
    let out = Tensor::<2>::empty([n, p], &x.device());
    let oc = cube_of(&out)?;
    let chunks = (p as u32).div_ceil(256);
    unsafe {
        quant_kernel::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(n as u32, 1, 1),
            CubeDim::new_3d(256, 1, 1),
            BufferArg::from_raw_parts(xc.handle.clone(), n * p),
            BufferArg::from_raw_parts(sc.handle.clone(), n),
            BufferArg::from_raw_parts(oc.handle.clone(), n * p),
            p as u32,
            chunks,
            bits as u32,
        );
    }
    let out = Tensor::from_primitive::<CudaBare>(oc);
    if p != d {
        Some(out.slice([0..n, 0..d]))
    } else {
        Some(out)
    }
}

#[cfg(feature = "autodiff")]
mod ad {
    use super::*;

    // ---- FWT: backward = the same transform ----

    #[derive(Debug)]
    struct FwtOp;

    impl<B: Backend> Backward<B, 1> for FwtOp
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        type State = usize;

        fn backward(
            self,
            ops: Ops<Self::State, 1>,
            grads: &mut Gradients,
            _checkpointer: &mut Checkpointer,
        ) {
            let p = ops.state;
            let d_out = Tensor::from_primitive::<B>(grads.consume::<B>(&ops.node));
            #[cfg(feature = "cuda")]
            {
                // NOT the gdn2 bug. `B` here is the *inner* backend, not the
                // autodiff one: `OpsPrep::finish` takes `FloatTensor<B>` and
                // returns `AutodiffTensor<B>`, and the result is handed to
                // `Tensor::from_primitive::<Autodiff<Inner, S>>`, so the
                // compiler pins `B = Inner`. (Typing this impl
                // `B: AutodiffBackend` instead fails with "required for
                // `FwtOp` to implement `Backward<Inner, 1>`" — that error IS
                // the proof.) The test therefore asks the right question; what
                // was dead was the ENTRY above, which could only build the
                // node on `NoCheckpointing`.
                if std::any::TypeId::of::<B>() == std::any::TypeId::of::<CudaBare>() {
                    if let Some(dx) = super::fwt_backward_cuda(&d_out, p) {
                        note_fused_backward();
                        grads.register::<B>(
                            ops.parents[0].clone().unwrap().id,
                            dx.try_into_primitive::<B>().unwrap(),
                        );
                        return;
                    }
                }
            }
            // tensor-path backward: H is symmetric, so apply the forward again
            let dx = crate::fast_walsh_hadamard_tensor::<B>(d_out, p);
            grads.register::<B>(
                ops.parents[0].clone().unwrap().id,
                dx.try_into_primitive::<B>().unwrap(),
            );
        }
    }

    // ---- quantize: straight-through (identity backward) ----

    #[derive(Debug)]
    struct QuantOp;

    impl<B: Backend> Backward<B, 1> for QuantOp
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        type State = ();

        fn backward(
            self,
            ops: Ops<Self::State, 1>,
            grads: &mut Gradients,
            _checkpointer: &mut Checkpointer,
        ) {
            let d_out = grads.consume::<B>(&ops.node);
            grads.register::<B>(ops.parents[0].clone().unwrap().id, d_out);
        }
    }

    /// Fused FWT with exact backward on `Autodiff<Inner, S>`, for ANY
    /// checkpointing strategy.
    ///
    /// The `_s` suffix is the same one gdn2 uses, and for the same reason: the
    /// un-suffixed spelling below says `Autodiff<Inner>`, which is
    /// `Autodiff<Inner, NoCheckpointing>` because that is `Autodiff`'s default
    /// type parameter. The strategy is not a type the caller can elide past —
    /// `try_into_primitive` compares the whole backend type — so a
    /// `NoCheckpointing`-only entry returns `None` on dormouse's
    /// `Autodiff<CudaBare, BalancedCheckpointing>` and the tensor path runs
    /// instead, SILENTLY and CORRECTLY.
    pub fn fwt_autodiff_s<Inner: Backend, S: CheckpointStrategy>(
        x: Tensor<2>,
        p: usize,
    ) -> Option<Tensor<2>>
    where
        DispatchTensor: DispatchKindConversion<Autodiff<Inner, S>> + DispatchKindConversion<Inner>,
    {
        let xa = x.try_into_primitive::<Autodiff<Inner, S>>().ok()?;
        note_entry_reached();
        let x_t = Tensor::from_primitive::<Inner>(xa.primitive().clone());
        let out_t = {
            #[cfg(feature = "cuda")]
            {
                if std::any::TypeId::of::<Inner>() == std::any::TypeId::of::<CudaBare>() {
                    if let Some(o) = super::fwt_cuda(&x_t, p) {
                        note_fused_forward();
                        o
                    } else {
                        crate::fast_walsh_hadamard_tensor::<Inner>(x_t, p)
                    }
                } else {
                    crate::fast_walsh_hadamard_tensor::<Inner>(x_t, p)
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                crate::fast_walsh_hadamard_tensor::<Inner>(x_t, p)
            }
        };
        let out_prim = out_t.try_into_primitive::<Inner>().unwrap();
        let nodes = [xa.node()];
        let prep = FwtOp.prepare::<S>(nodes);
        let out_adt = match prep.compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish(p, out_prim),
            OpsKind::UnTracked(prep) => prep.finish(out_prim),
        };
        Some(Tensor::from_primitive::<Autodiff<Inner, S>>(out_adt))
    }

    /// [`fwt_autodiff_s`] on the default (no-checkpointing) strategy. Kept so
    /// existing callers keep compiling; it is NOT the strategy dormouse runs.
    pub fn fwt_autodiff<Inner: Backend>(x: Tensor<2>, p: usize) -> Option<Tensor<2>>
    where
        DispatchTensor: DispatchKindConversion<Autodiff<Inner>> + DispatchKindConversion<Inner>,
    {
        fwt_autodiff_s::<Inner, NoCheckpointing>(x, p)
    }

    /// Fused quantize (straight-through backward) on `Autodiff<Inner, S>`.
    pub fn quant_autodiff_s<Inner: Backend, S: CheckpointStrategy>(
        x: Tensor<2>,
        bits: usize,
    ) -> Option<Tensor<2>>
    where
        DispatchTensor: DispatchKindConversion<Autodiff<Inner, S>> + DispatchKindConversion<Inner>,
    {
        let xa = x.try_into_primitive::<Autodiff<Inner, S>>().ok()?;
        note_entry_reached();
        let x_t = Tensor::from_primitive::<Inner>(xa.primitive().clone());
        let out_t = {
            #[cfg(feature = "cuda")]
            {
                if std::any::TypeId::of::<Inner>() == std::any::TypeId::of::<CudaBare>() {
                    if let Some(o) = super::quant_cuda(&x_t, bits) {
                        note_fused_forward();
                        o
                    } else {
                        crate::quantize_tensor::<Inner>(x_t, bits)
                    }
                } else {
                    crate::quantize_tensor::<Inner>(x_t, bits)
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                crate::quantize_tensor::<Inner>(x_t, bits)
            }
        };
        let out_prim = out_t.try_into_primitive::<Inner>().unwrap();
        let nodes = [xa.node()];
        let prep = QuantOp.prepare::<S>(nodes);
        let out_adt = match prep.compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish((), out_prim),
            OpsKind::UnTracked(prep) => prep.finish(out_prim),
        };
        Some(Tensor::from_primitive::<Autodiff<Inner, S>>(out_adt))
    }

    /// [`quant_autodiff_s`] on the default (no-checkpointing) strategy.
    pub fn quant_autodiff<Inner: Backend>(x: Tensor<2>, bits: usize) -> Option<Tensor<2>>
    where
        DispatchTensor: DispatchKindConversion<Autodiff<Inner>> + DispatchKindConversion<Inner>,
    {
        quant_autodiff_s::<Inner, NoCheckpointing>(x, bits)
    }
}

#[cfg(feature = "autodiff")]
pub use ad::{fwt_autodiff, fwt_autodiff_s, quant_autodiff, quant_autodiff_s};

#[cfg(all(test, feature = "autodiff"))]
mod seam_tests {
    //! The proof that the strategy gate is strategy-AGNOSTIC, and it runs on
    //! CPU. No GPU, no kernel: it asserts that a caller on
    //! `Autodiff<Inner, BalancedCheckpointing>` — dormouse's backend — gets
    //! PAST the seam downcast, and that the old `NoCheckpointing`-only
    //! spelling does not. Revert `fwt_autodiff_s` to `Autodiff<Inner>` and the
    //! first assertion goes red: it is the assertion, not the comment.
    use super::*;
    use burn::backend::DispatchKindConversion;
    use burn::tensor::Device;
    use burn_autodiff::checkpoint::strategy::{
        BalancedCheckpointing, CheckpointStrategy, NoCheckpointing,
    };
    use burn_autodiff::Autodiff as Ad;

    type Nd = burn_ndarray::NdArray;

    #[test]
    fn balanced_checkpointing_reaches_the_seam_and_the_legacy_entry_does_not() {
        fn reach<S: CheckpointStrategy>(x: &Tensor<2>) -> Option<Tensor<2>>
        where
            DispatchTensor: DispatchKindConversion<Ad<Nd, S>> + DispatchKindConversion<Nd>,
        {
            fwt_autodiff_s::<Nd, S>(x.clone(), 16)
        }

        // dormouse's own backend, on the CPU inner backend so the test needs no
        // device. The strategy is the only variable under test.
        let dev = Device::ndarray().autodiff().gradient_checkpointing();
        let x = Tensor::<2>::ones([4, 12], &dev);

        reset_seam_counts();
        let base = seam_counts()
            .expect("autodiff feature is on in this test")
            .0;

        // 1. the strategy-generic entry reaches the seam on Balanced.
        assert!(reach::<BalancedCheckpointing>(&x).is_some());
        assert_eq!(
            seam_counts().expect("counters").0,
            base + 1,
            "a BalancedCheckpointing caller must get past the seam downcast"
        );

        // 2. and the legacy NoCheckpointing-only spelling does NOT, which is
        //    precisely what used to happen to dormouse: a silent `None`, a
        //    correct tensor answer, and a fused arm that never ran.
        assert!(
            fwt_autodiff::<Nd>(x.clone(), 16).is_none(),
            "on a Balanced tensor the NoCheckpointing entry must refuse"
        );
        assert_eq!(
            seam_counts().expect("counters").0,
            base + 1,
            "the refusing entry must not have counted a reach"
        );

        // 3. the default strategy still works, so the probe is not a
        //    replacement but an addition. Its own device, because that is the
        //    point: the strategy is a property of the tensor, not of the call.
        let plain = Device::ndarray().autodiff();
        let xp = Tensor::<2>::ones([4, 12], &plain);
        assert!(reach::<NoCheckpointing>(&xp).is_some());
        assert!(
            reach::<BalancedCheckpointing>(&xp).is_none(),
            "and the cross-check must refuse too, or the gate is not real"
        );
    }
}

#[cfg(all(test, feature = "autodiff", feature = "cuda"))]
mod ad_tests {
    use super::*;
    use burn::tensor::{Device, Distribution, Tensor};

    fn to_host(t: Tensor<2>) -> Vec<f32> {
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
    fn fwt_fused_backward_matches_tensor() {
        let dev = Device::default().autodiff();
        // d=12: non-power-of-2, exercises the fused pad branch (p=16)
        let (n, d) = (4usize, 12usize);
        let p = 16usize;
        let x = Tensor::<2>::random([n, d], Distribution::Normal(0.0, 1.0), &dev);

        // fused op graph
        let xf = x.clone().require_grad();
        let outf = fwt_autodiff::<CudaBare>(xf.clone(), p).unwrap();
        let loss_f = outf.powf_scalar(2.0).sum();
        let grads_f = loss_f.backward();
        let dxf = xf.grad(&grads_f).unwrap();

        // independent reference: same values on the ndarray device (CUDA
        // matmul would run TF32; ndarray f32 matmul is near-exact). The
        // butterfly tensor path would dispatch to this same fused kernel.
        let cpu = Device::ndarray().autodiff();
        let xt = Tensor::<2>::from_data(x.clone().into_data(), &cpu).require_grad();
        let outt = crate::fwt_reference_naive(&xt.clone());
        let loss_t = outt.powf_scalar(2.0).sum();
        let grads_t = loss_t.backward();
        let dxt = xt.grad(&grads_t).unwrap();

        let md = maxdiff(&to_host(dxf), &to_host(dxt));
        assert!(md < 1e-3, "fwt grad maxdiff {md}");
    }

    #[test]
    fn quant_fused_straight_through() {
        // The quant op's backward must be identity (straight-through).
        let dev = Device::default().autodiff();
        let (n, d) = (4usize, 16usize);
        let x = Tensor::<2>::random([n, d], Distribution::Normal(0.0, 1.0), &dev);
        let xf = x.clone().require_grad();
        let q = quant_autodiff::<CudaBare>(xf.clone(), 4).unwrap();
        let loss = q.powf_scalar(2.0).sum();
        let grads = loss.backward();
        let dx = xf.grad(&grads).unwrap();
        // straight-through: d loss/dx = upstream grad = 2*q (identity through
        // the quantize); burn's autodiff of round/clamp is 0, so compare
        // against the analytic 2*q directly
        let q_plain = crate::quantize_tensor::<CudaBare>(x, 4);
        let expected = to_host(q_plain.mul_scalar(2.0));
        let md = maxdiff(&to_host(dx), &expected);
        assert!(md < 1e-3, "straight-through grad maxdiff {md}");
    }
}

#[cfg(all(test, feature = "cuda"))]
mod bench {
    use super::*;
    use burn::tensor::{Device, Distribution, Tensor};

    #[test]
    #[ignore]
    fn bitnet_bench() {
        let dev = Device::default();
        let (n, d) = (256usize, 512usize);
        let p = d.next_power_of_two();
        let x = Tensor::<2>::random([n, d], Distribution::Normal(0.0, 1.0), &dev);
        // warmup
        for _ in 0..3 {
            let _ = fwt_cuda(&x, p).unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..50 {
            let _ = fwt_cuda(&x, p).unwrap();
        }
        let _: f32 = fwt_cuda(&x, p).unwrap().sum().into_scalar();
        let tf = t0.elapsed() / 50;
        // tensor-path baseline on the SAME CUDA device (an earlier revision
        // compared against an NdArray CPU baseline, i.e. CUDA-vs-CPU, which
        // is not a GPU-vs-GPU comparison)
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            let r = crate::fast_walsh_hadamard_tensor::<CudaBare>(x.clone(), p);
            let _: f32 = r.sum().into_scalar();
        }
        let tt = t0.elapsed() / 10;
        println!(
            "[{n}x{d}] fwt fused {:?} tensor-path {:?} ({:.0}x)",
            tf,
            tt,
            tt.as_secs_f64() / tf.as_secs_f64()
        );
        // quant
        for _ in 0..3 {
            let _ = quant_cuda(&x, 8).unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..50 {
            let _ = quant_cuda(&x, 8).unwrap();
        }
        let _: f32 = quant_cuda(&x, 8).unwrap().sum().into_scalar();
        let tq = t0.elapsed() / 50;
        println!("[{n}x{d}] quant8 fused {tq:?}");
    }
}
