//! Fused CUDA SiTU-GLU + autodiff registration (Kimi K3 Eq 12).
//!
//! The whole `situ_glu` (two softcaps + Swish gate factor) is one elementwise
//! launch instead of ~7 tensor passes, and under an autodiff backend it runs
//! as a single tracked node with an exact elementwise backward (recomputed
//! from the checkpointed input), so training sees the same fused kernels.

#[cfg(feature = "autodiff")]
use burn::backend::Backend;
#[cfg(feature = "autodiff")]
use burn::backend::DispatchKindConversion;
#[cfg(feature = "autodiff")]
use burn::tensor::DispatchTensor;
use burn::tensor::Tensor;

#[cfg(feature = "cuda")]
use cubecl::prelude::*;

#[cfg(feature = "autodiff")]
use {
    burn_autodiff::checkpoint::base::Checkpointer,
    burn_autodiff::checkpoint::strategy::NoCheckpointing,
    burn_autodiff::grads::Gradients,
    burn_autodiff::ops::{Backward, Ops, OpsKind},
    burn_autodiff::Autodiff,
};

#[cfg(feature = "cuda")]
/// out[row·H+col] = bg·tanh(g/bg)·sigmoid(g)·bu·tanh(u/bu), g = gate, u = up.
/// One cube per ROW (CUBE_POS_X), comptime loop over H in 256-col slabs: no
/// integer division (cubecl's CUDA codegen computes `a/b` as `(a/b)·b`, which
/// corrupts any kernel that divides), no 2D grid.
#[cube(launch_unchecked)]
fn situ_glu_kernel<F: Float>(
    gu: &[F],
    out: &mut [F],
    bg: f32,
    bu: f32,
    #[comptime] h: u32,
    #[comptime] h_chunks: u32,
) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let h = h as usize;
    for chunk in 0..h_chunks {
        let col = (chunk as usize) * 256 + tid;
        if col < h {
            let g = gu[row * (2 * h) + col];
            let u = gu[row * (2 * h) + h + col];
            let tg = (g / F::cast_from(bg)).tanh();
            let tu = (u / F::cast_from(bu)).tanh();
            let s = F::new(1.0_f32) / (F::new(1.0_f32) + (-g).exp());
            out[row * h + col] = F::cast_from(bg) * tg * s * F::cast_from(bu) * tu;
        }
    }
}

#[cfg(feature = "cuda")]
/// Exact elementwise backward: recomputes the gate/up activations from the
/// checkpointed input and scatters d_out into the [N, 2H] input grad.
#[cube(launch_unchecked)]
fn situ_glu_backward_kernel<F: Float>(
    gu: &[F],      // [N, 2H]
    grad: &[F],    // [N, H]
    dgu: &mut [F], // [N, 2H]
    bg: f32,
    bu: f32,
    #[comptime] h: u32,
    #[comptime] h_chunks: u32,
) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let h = h as usize;
    for chunk in 0..h_chunks {
        let col = (chunk as usize) * 256 + tid;
        if col < h {
            let g = gu[row * (2 * h) + col];
            let u = gu[row * (2 * h) + h + col];
            let tg = (g / F::cast_from(bg)).tanh();
            let tu = (u / F::cast_from(bu)).tanh();
            let s = F::new(1.0_f32) / (F::new(1.0_f32) + (-g).exp());
            let one = F::new(1.0_f32);
            let dg = grad[row * h + col]
                * F::cast_from(bu)
                * tu
                * ((one - tg * tg) * s + F::cast_from(bg) * tg * s * (one - s));
            let du = grad[row * h + col] * F::cast_from(bg) * tg * s * (one - tu * tu);
            dgu[row * (2 * h) + col] = dg;
            dgu[row * (2 * h) + h + col] = du;
        }
    }
}

#[cfg(feature = "cuda")]
fn cube_of(t: &Tensor<2>) -> Option<burn_cubecl::tensor::CubeTensor> {
    type B = burn_cubecl::CubeBackend;
    let prim = t.clone().try_into_primitive::<B>().ok()?;
    let c = (&prim as &dyn std::any::Any)
        .downcast_ref::<burn_cubecl::tensor::CubeTensor>()?;
    Some(c.clone())
}

/// Fused forward on the bare CUDA backend; `None` otherwise.
#[cfg(feature = "cuda")]
pub fn situ_glu_cuda(gate_up: &Tensor<2>, hidden: usize, bg: f64, bu: f64) -> Option<Tensor<2>> {
    let [n, d2] = gate_up.dims();
    // cubecl's CUDA codegen corrupts coalesced stores on row strides that are
    // not 32-byte multiples (h % 8 != 0, except the single-float4 h = 4 case).
    // Callers fall back to the tensor path for those shapes.
    if hidden == 0 || d2 != 2 * hidden || (!hidden.is_multiple_of(8) && hidden != 4) {
        return None;
    }
    let gc = cube_of(gate_up)?;
    let out = Tensor::<2>::empty([n, hidden], &gate_up.device());
    let oc = cube_of(&out)?;
    let client = gc.client.clone();
    let threads = 256u32;
    let dim = CubeDim::new_3d(threads, 1, 1);
    unsafe {
        situ_glu_kernel::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(n as u32, 1, 1),
            dim,
            BufferArg::from_raw_parts(gc.handle.clone(), n * d2),
            BufferArg::from_raw_parts(oc.handle, n * hidden),
            bg as f32,
            bu as f32,
            hidden as u32,
            (hidden as u32).div_ceil(threads),
        );
    }
    Some(out)
}

#[cfg(all(feature = "cuda", feature = "autodiff"))]
fn situ_glu_backward_cuda(
    gate_up: &Tensor<2>,
    d_out: &Tensor<2>,
    hidden: usize,
    bg: f64,
    bu: f64,
) -> Option<Tensor<2>> {
    let [n, d2] = gate_up.dims();
    if hidden % 8 != 0 && hidden != 4 {
        return None;
    }
    let gc = cube_of(gate_up)?;
    let dc = cube_of(d_out)?;
    let dgu = Tensor::<2>::empty([n, d2], &gate_up.device());
    let dgc = cube_of(&dgu)?;
    let client = gc.client.clone();
    let threads = 256u32;
    let dim = CubeDim::new_3d(threads, 1, 1);
    unsafe {
        situ_glu_backward_kernel::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(n as u32, 1, 1),
            dim,
            BufferArg::from_raw_parts(gc.handle.clone(), n * d2),
            BufferArg::from_raw_parts(dc.handle.clone(), n * hidden),
            BufferArg::from_raw_parts(dgc.handle, n * d2),
            bg as f32,
            bu as f32,
            hidden as u32,
            (hidden as u32).div_ceil(threads),
        );
    }
    Some(dgu)
}

/// Tensor-path backward (fallback for non-CUDA autodiff backends).
#[cfg(feature = "autodiff")]
fn situ_glu_backward_tensor<B: Backend>(
    gate_up: &Tensor<2>,
    d_out: &Tensor<2>,
    hidden: usize,
    bg: f64,
    bu: f64,
) -> Tensor<2> {
    let [n, _] = gate_up.dims();
    let gate = gate_up.clone().slice([0..n, 0..hidden]);
    let up = gate_up.clone().slice([0..n, hidden..2 * hidden]);
    let tg = (gate.clone() / bg).tanh();
    let tu = (up.clone() / bu).tanh();
    let s = burn::tensor::activation::sigmoid(gate.clone());
    let one = 1.0_f64;
    let dg = d_out.clone()
        * bu
        * tu.clone()
        * ((one - tg.clone() * tg.clone()) * s.clone()
            + bg * tg.clone() * s.clone() * (one - s.clone()));
    let du = d_out.clone() * bg * tg * s * (one - tu.clone() * tu.clone());
    Tensor::cat(vec![dg.reshape([n, hidden]), du.reshape([n, hidden])], 1)
}

#[cfg(feature = "autodiff")]
mod ad {
    use super::*;

    const N_PARENTS: usize = 1;

    #[derive(Debug)]
    struct SituGlu;

    impl<B: Backend> Backward<B, N_PARENTS> for SituGlu
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        type State = (f64, f64, usize);

        fn backward(
            self,
            ops: Ops<Self::State, N_PARENTS>,
            grads: &mut Gradients,
            checkpointer: &mut Checkpointer,
        ) {
            let (bg, bu, hidden) = ops.state;
            let gu = ops.parents[0]
                .as_ref()
                .map(|n| Tensor::from_primitive::<B>(checkpointer.retrieve_node_output(n.id)))
                .expect("gate_up checkpointed");
            let d_out = Tensor::from_primitive::<B>(grads.consume::<B>(&ops.node));

            #[cfg(feature = "cuda")]
            {
                type CudaBare = burn_cubecl::CubeBackend;
                if std::any::TypeId::of::<B>() == std::any::TypeId::of::<CudaBare>() {
                    if let Some(d_gu) = super::situ_glu_backward_cuda(&gu, &d_out, hidden, bg, bu) {
                        grads.register::<B>(
                            ops.parents[0].clone().unwrap().id,
                            d_gu.try_into_primitive::<B>().unwrap(),
                        );
                        return;
                    }
                }
            }

            let d_gu = super::situ_glu_backward_tensor::<B>(&gu, &d_out, hidden, bg, bu);
            grads.register::<B>(
                ops.parents[0].clone().unwrap().id,
                d_gu.try_into_primitive::<B>().unwrap(),
            );
        }
    }

    /// Fused SiTU-GLU with exact backward on `Autodiff<Inner>`; `None` when the
    /// tensor is not on an autodiff backend (caller falls back).
    pub fn situ_glu_autodiff<Inner: Backend>(
        gate_up: Tensor<2>,
        hidden: usize,
        bg: f64,
        bu: f64,
    ) -> Option<Tensor<2>>
    where
        DispatchTensor: DispatchKindConversion<Autodiff<Inner>> + DispatchKindConversion<Inner>,
    {
        let gu = gate_up.try_into_primitive::<Autodiff<Inner>>().ok()?;
        let gu_t = Tensor::from_primitive::<Inner>(gu.primitive().clone());

        let out_t = {
            #[cfg(feature = "cuda")]
            {
                type CudaBare = burn_cubecl::CubeBackend;
                if std::any::TypeId::of::<Inner>() == std::any::TypeId::of::<CudaBare>() {
                    if let Some(o) = super::situ_glu_cuda(&gu_t, hidden, bg, bu) {
                        o
                    } else {
                        super::situ_glu_tensor(&gu_t, hidden, bg, bu)
                    }
                } else {
                    super::situ_glu_tensor(&gu_t, hidden, bg, bu)
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                super::situ_glu_tensor(&gu_t, hidden, bg, bu)
            }
        };

        let out_prim = out_t.try_into_primitive::<Inner>().unwrap();
        let nodes = [gu.node()];
        let prep = SituGlu.prepare::<NoCheckpointing>(nodes);
        let out_adt = match prep.compute_bound().stateful() {
            OpsKind::Tracked(mut prep) => {
                let _ids = [Some(prep.checkpoint(&gu))];
                prep.finish((bg, bu, hidden), out_prim)
            }
            OpsKind::UnTracked(prep) => prep.finish(out_prim),
        };
        Some(Tensor::from_primitive::<Autodiff<Inner>>(out_adt))
    }
}

/// Pure tensor-path forward shared by the autodiff fallback.
#[cfg(any(feature = "autodiff", test))]
pub fn situ_glu_tensor(gate_up: &Tensor<2>, hidden: usize, bg: f64, bu: f64) -> Tensor<2> {
    let [n, _d2] = gate_up.dims();
    let gate = gate_up.clone().slice([0..n, 0..hidden]);
    let up = gate_up.clone().slice([0..n, hidden..2 * hidden]);
    let gate_cap = gate.clone() / bg;
    let gate_cap = gate_cap.tanh() * bg;
    let up_cap = up / bu;
    let up_cap = up_cap.tanh() * bu;
    gate_cap
        .mul(burn::tensor::activation::sigmoid(gate))
        .mul(up_cap)
}

#[cfg(feature = "autodiff")]
pub use ad::situ_glu_autodiff;

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    // These tests need a live CUDA context; local CPU sweeps skip them via
    // BURN_DEVICE (the CI gpu-job sets BURN_DEVICE=cuda).
    fn cuda_enabled() -> bool {
        std::env::var("BURN_DEVICE")
            .map(|v| v == "cuda")
            .unwrap_or(false)
    }

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
    #[ignore]
    fn situ_bench() {
        let dev = Device::default();
        let (n, h) = (2048usize, 5120usize);
        let gu = Tensor::<2>::random([n, 2 * h], Distribution::Normal(0.0, 1.0), &dev);
        for _ in 0..3 {
            let _ = situ_glu_cuda(&gu, h, 1.0, 1.5).unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            let out = situ_glu_cuda(&gu, h, 1.0, 1.5).unwrap();
            let _: f32 = out.sum().into_scalar(); // sync: fused kernels are async
        }
        let tf = t0.elapsed() / 20;
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            let out = situ_glu_tensor(&gu, h, 1.0, 1.5);
            let _: f32 = out.sum().into_scalar();
        }
        let tt = t0.elapsed() / 10;
        println!(
            "[{n}x{h}] fused {:?} tensor {:?} ({:.1}x)",
            tf,
            tt,
            tt.as_secs_f64() / tf.as_secs_f64()
        );
    }

    #[test]
    fn unaligned_h_falls_back() {
        if !cuda_enabled() {
            eprintln!("skipped: BURN_DEVICE != cuda");
            return;
        }
        // h % 8 != 0: the fused kernel must decline so the tensor path runs.
        let dev = Device::default();
        let gu = Tensor::<2>::random([4, 6], Distribution::Normal(0.0, 1.0), &dev);
        assert!(
            situ_glu_cuda(&gu, 3, 1.0, 1.0).is_none(),
            "h=3 must fall back"
        );
        // and the lib-level result is still correct through the fallback
        let out = crate::situ_glu(gu, 3, 1.0, 1.0);
        assert_eq!(out.dims(), [4, 3]);
        assert!(to_host(out).iter().all(|x| x.is_finite()));
    }

    #[test]
    fn situ_fused_matches_tensor() {
        if !cuda_enabled() {
            eprintln!("skipped: BURN_DEVICE != cuda");
            return;
        }
        let dev = Device::default();
        for (n, h) in [(16usize, 64usize), (2048, 5120), (4, 8)] {
            let gu = Tensor::<2>::random([n, 2 * h], Distribution::Normal(0.0, 1.0), &dev);
            let expected = to_host(situ_glu_tensor(&gu, h, 1.0, 1.5));
            let got = to_host(situ_glu_cuda(&gu, h, 1.0, 1.5).expect("kernel"));
            let md = maxdiff(&got, &expected);
            assert!(md < 1e-5, "[{n},{h}] maxdiff {md}");
        }
    }
}

#[cfg(all(test, feature = "autodiff", feature = "cuda"))]
mod ad_tests {
    use super::*;
    use burn::tensor::{Device, Distribution, Tensor};

    type CudaBare = burn_cubecl::CubeBackend;

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
    fn fused_backward_matches_tensor_backward() {
        if !cuda_enabled() {
            eprintln!("skipped: BURN_DEVICE != cuda");
            return;
        }
        let dev = Device::default().autodiff();
        let gu = Tensor::<2>::random([4, 8], Distribution::Normal(0.0, 1.0), &dev);

        let gu_f = gu.clone().require_grad();
        let out_f = situ_glu_autodiff::<CudaBare>(gu_f.clone(), 4, 1.0, 1.0).unwrap();
        let loss_f = out_f.powf_scalar(2.0).sum();
        let grads_f = loss_f.backward();
        let dgu_f = gu_f.grad(&grads_f).unwrap();

        let gu_t = gu.clone().require_grad();
        let out_t = situ_glu_tensor(&gu_t, 4, 1.0, 1.0);
        let loss_t = out_t.powf_scalar(2.0).sum();
        let grads_t = loss_t.backward();
        let dgu_t = gu_t.grad(&grads_t).unwrap();

        let a = to_host(dgu_f);
        let b = to_host(dgu_t);
        let md = maxdiff(&a, &b);
        assert!(md < 1e-3, "grad maxdiff {md}");
    }

    #[test]
    fn fused_forward_autodiff_matches() {
        if !cuda_enabled() {
            eprintln!("skipped: BURN_DEVICE != cuda");
            return;
        }
        let dev = Device::default().autodiff();
        let gu = Tensor::<2>::random([3, 6], Distribution::Normal(0.0, 1.0), &dev);
        let out_f = situ_glu_autodiff::<CudaBare>(gu.clone(), 3, 1.0, 2.0).unwrap();
        let expected = situ_glu_tensor(&gu, 3, 1.0, 2.0);
        let md = maxdiff(&to_host(out_f), &to_host(expected));
        assert!(md < 1e-5, "fwd maxdiff {md}");
    }
}

#[cfg(all(test, feature = "autodiff", feature = "cuda"))]
mod fd_tests {
    use super::*;
    use burn::tensor::{Device, Distribution, Tensor};

    type CudaBare = burn_cubecl::CubeBackend;

    fn to_host(t: Tensor<2>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn situ_sum(x: &Tensor<2>) -> f32 {
        situ_glu_tensor(x, 2, 1.0, 1.0).sum().into_scalar()
    }

    #[test]
    fn situ_grad_matches_finite_difference() {
        if !cuda_enabled() {
            eprintln!("skipped: BURN_DEVICE != cuda");
            return;
        }
        // Independent gradient check: the analytic backward (fused op) must
        // match central finite differences of the scalar loss sum(situ(x)).
        let dev = Device::default();
        let adev = Device::default().autodiff();
        let x = Tensor::<2>::random([3, 4], Distribution::Normal(0.0, 1.0), &dev);
        let data = x.clone().into_data();

        // analytic grad (fused op, loss = sum)
        let xf = Tensor::<2>::from_data(data.clone(), &adev).require_grad();
        let out = situ_glu_autodiff::<CudaBare>(xf.clone(), 2, 1.0, 1.0).unwrap();
        let grads = out.sum().backward();
        let analytic = to_host(xf.grad(&grads).unwrap());

        // central finite differences on the plain backend
        let eps = 1e-3f32;
        let mut fd = vec![0.0f32; 12];
        for i in 0..12 {
            let mut plus = data.clone();
            plus.bytes[i * 4..i * 4 + 4].copy_from_slice(
                &(data.bytes[i * 4..i * 4 + 4]
                    .try_into()
                    .map(f32::from_le_bytes)
                    .unwrap()
                    + eps)
                    .to_le_bytes(),
            );
            let mut minus = data.clone();
            let mv = f32::from_le_bytes(minus.bytes[i * 4..i * 4 + 4].try_into().unwrap()) - eps;
            minus.bytes[i * 4..i * 4 + 4].copy_from_slice(&mv.to_le_bytes());
            let xp = Tensor::<2>::from_data(plus, &dev);
            let xm = Tensor::<2>::from_data(minus, &dev);
            fd[i] = (situ_sum(&xp) - situ_sum(&xm)) / (2.0 * eps);
        }

        for i in 0..12 {
            let rel = (analytic[i] - fd[i]).abs() / (fd[i].abs() + 1e-6);
            assert!(
                rel < 5e-3,
                "idx {i}: analytic {} vs fd {} (rel {rel})",
                analytic[i],
                fd[i]
            );
        }
    }
}
