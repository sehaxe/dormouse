//! Fused CUDA Sinkhorn-Knopp (mHC Eq 9). One cube per (batch, time) matrix;
//! the full `iters` of alternating row/column normalizations run in one
//! launch (the row sums need the previous iteration's column-normalized rows,
//! so a `sync_cube()` separates the two phases each iteration).

use burn::tensor::Tensor;
use cubecl::prelude::*;
use std::any::Any;

#[cube(launch_unchecked)]
fn sinkhorn_kernel<F: Float>(
    m: &mut [F], // [B, T, n, n]
    iters: u32,
    #[comptime] n: u32,
) {
    let bt = CUBE_POS_X as usize; // b·T + t
    let i = UNIT_POS_X as usize;
    let n = n as usize;
    let base = bt * n * n;

    let mut it = 0u32;
    while it < iters {
        // row i sum + normalize to sum 1 (row is contiguous)
        let mut rs = F::new(0.0_f32);
        let mut j = 0;
        while j < n {
            rs += m[base + i * n + j];
            j += 1;
        }
        let mut rn = rs;
        if rn < F::new(1e-7_f32) {
            rn = F::new(1e-7_f32);
        }
        let mut j = 0;
        while j < n {
            m[base + i * n + j] /= rn;
            j += 1;
        }
        sync_cube();

        // column i sum + normalize to sum 1 (column is strided)
        let mut cs = F::new(0.0_f32);
        let mut j = 0;
        while j < n {
            cs += m[base + j * n + i];
            j += 1;
        }
        let mut cn = cs;
        if cn < F::new(1e-7_f32) {
            cn = F::new(1e-7_f32);
        }
        let mut j = 0;
        while j < n {
            m[base + j * n + i] /= cn;
            j += 1;
        }
        sync_cube();
        it += 1;
    }
}

#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn sinkhorn_backward_kernel<F: Float>(
    logits: &[F],      // [B, T, n, n]
    dout: &[F],        // [B, T, n, n]
    dlogits: &mut [F], // [B, T, n, n]
    #[comptime] n: u32,
    #[comptime] iters: u32,
) {
    let bt = CUBE_POS_X as usize;
    let t = UNIT_POS_X as usize;
    let n = n as usize;
    let iters = iters as usize;
    let base = bt * n * n;
    let mut m = Shared::<[F]>::new_slice(n * n);
    let mut d = Shared::<[F]>::new_slice(n * n);
    let mut sums = Shared::<[F]>::new_slice(iters * 2 * n);

    for j in 0..n {
        m[t * n + j] = logits[base + t * n + j].exp();
        d[t * n + j] = dout[base + t * n + j];
    }
    sync_cube();

    for it in 0..iters {
        let mut rs = F::new(0.0_f32);
        for j in 0..n {
            rs += m[t * n + j];
        }
        sums[(it * 2) * n + t] = rs;
        sync_cube();
        for j in 0..n {
            m[t * n + j] /= rs;
        }
        sync_cube();
        let mut cs = F::new(0.0_f32);
        for i in 0..n {
            cs += m[i * n + t];
        }
        sums[(it * 2 + 1) * n + t] = cs;
        sync_cube();
        for i in 0..n {
            m[i * n + t] /= cs;
        }
        sync_cube();
    }

    for k in 0..(iters * 2) {
        let r = iters * 2 - 1 - k;
        let s = sums[r * n + t];
        if r.is_multiple_of(2) {
            // row normalization (forward summed over dim 3): the correction is
            // the WEIGHTED row sum of d with m_pre = m·s: Σ_j d·m_pre
            let mut sd = F::new(0.0_f32);
            for j in 0..n {
                sd += d[t * n + j] * m[t * n + j] * s;
            }
            for j in 0..n {
                let mp = m[t * n + j] * s;
                d[t * n + j] = d[t * n + j] / s - sd / (s * s);
                m[t * n + j] = mp;
            }
        } else {
            // column normalization (forward summed over dim 2)
            let mut sd = F::new(0.0_f32);
            for i in 0..n {
                sd += d[i * n + t] * m[i * n + t] * s;
            }
            for i in 0..n {
                let mp = m[i * n + t] * s;
                d[i * n + t] = d[i * n + t] / s - sd / (s * s);
                m[i * n + t] = mp;
            }
        }
        sync_cube();
    }

    for j in 0..n {
        dlogits[base + t * n + j] = d[t * n + j] * logits[base + t * n + j].exp();
    }
}

#[cfg(all(feature = "cuda", feature = "autodiff"))]
fn sinkhorn_backward_cuda(
    logits: &Tensor<4>,
    d_out: &Tensor<4>,
    iters: usize,
) -> Option<Tensor<4>> {
    let [b, t, n, n2] = logits.dims();
    if n != n2 || iters == 0 || iters > 20 {
        return None;
    }
    // shared ceiling (m + d + sums) <= 48 KB (RTX 3090)
    if n * n * 2 + iters * 2 * n > 12000 {
        return None;
    }
    let cube =
        |x: &Tensor<4>| -> Option<burn_cubecl::tensor::CubeTensor<cubecl::cuda::CudaRuntime>> {
            type B = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
            let prim = x.clone().try_into_primitive::<B>().ok()?;
            let c = (&prim as &dyn std::any::Any)
                .downcast_ref::<burn_cubecl::tensor::CubeTensor<cubecl::cuda::CudaRuntime>>()?;
            Some(c.clone())
        };
    let lc = cube(logits)?;
    let dc = cube(d_out)?;
    let dlogits = Tensor::<4>::empty([b, t, n, n], &logits.device());
    let oc = cube(&dlogits)?;
    let client = lc.client.clone();
    unsafe {
        sinkhorn_backward_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
            &client,
            CubeCount::Static((b * t) as u32, 1, 1),
            CubeDim::new_3d(n as u32, 1, 1),
            BufferArg::from_raw_parts(lc.handle, b * t * n * n),
            BufferArg::from_raw_parts(dc.handle, b * t * n * n),
            BufferArg::from_raw_parts(oc.handle, b * t * n * n),
            n as u32,
            iters as u32,
        );
    }
    Some(dlogits)
}

/// Run the fused Sinkhorn on the bare CUDA backend. Returns false otherwise.
pub fn sinkhorn_cuda(x: &mut Tensor<4>, iters: usize) -> bool {
    let dims = x.dims();
    let (b, t, n) = (dims[0], dims[1], dims[2]);
    if n != dims[3] {
        return false;
    }
    type B = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
    let prim = x.clone().try_into_primitive::<B>().ok();
    let Some(prim) = prim else {
        return false;
    };
    let Some(xc) = (&prim as &dyn Any)
        .downcast_ref::<burn_cubecl::tensor::CubeTensor<cubecl::cuda::CudaRuntime>>()
    else {
        return false;
    };
    let client = xc.client.clone();
    let dim = CubeDim::new_3d(n as u32, 1, 1);
    unsafe {
        sinkhorn_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
            &client,
            CubeCount::Static((b * t) as u32, 1, 1),
            dim,
            BufferArg::from_raw_parts(xc.handle.clone(), b * t * n * n),
            iters as u32,
            n as u32,
        );
    }
    true
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use burn::tensor::{Device, Distribution, Tensor};

    #[test]
    fn sinkhorn_matches_tensor() {
        let dev = Device::default();
        let x: Tensor<4> = Tensor::random([2, 64, 8, 8], Distribution::Normal(0.0, 1.0), &dev);
        // tensor reference
        let mut r = x.clone().exp();
        for _ in 0..20 {
            r = r.clone() / r.clone().sum_dim(3);
            r = r.clone() / r.clone().sum_dim(2);
        }
        let mut k = x.exp();
        assert!(sinkhorn_cuda(&mut k, 20), "kernel should run");
        let diff: f32 = (r - k.clone()).abs().max().into_scalar::<f32>();
        assert!(diff < 1e-3, "sinkhorn diff {diff}");

        // doubly stochastic check
        let rowsum: f32 = (k.clone().sum_dim(3) - 1.0)
            .abs()
            .max()
            .into_scalar::<f32>();
        let colsum: f32 = (k.clone().sum_dim(2) - 1.0)
            .abs()
            .max()
            .into_scalar::<f32>();
        assert!(rowsum < 1e-2, "row sums {rowsum}");
        assert!(colsum < 1e-2, "col sums {colsum}");
    }

    #[test]
    #[ignore]
    fn sinkhorn_bench() {
        let dev = Device::default();
        for (b, t, n) in [(8usize, 2048, 16usize), (4, 1024, 64)] {
            let x: Tensor<4> = Tensor::random([b, t, n, n], Distribution::Normal(0.0, 1.0), &dev);
            for _ in 0..3 {
                let mut k = x.clone();
                let _ = sinkhorn_cuda(&mut k, 20);
            }
            let t0 = std::time::Instant::now();
            for _ in 0..20 {
                let mut k = x.clone();
                let _ = sinkhorn_cuda(&mut k, 20);
            }
            let tf = t0.elapsed() / 20;
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                let mut m = x.clone().exp();
                for _ in 0..20 {
                    m = m.clone() / m.clone().sum_dim(3);
                    m = m.clone() / m.clone().sum_dim(2);
                }
            }
            let tt = t0.elapsed() / 10;
            println!(
                "[{b}x{t}x{n}] fused {:?} tensor {:?} ({:.1}x)",
                tf,
                tt,
                tt.as_secs_f64() / tf.as_secs_f64()
            );
        }
    }
}

#[cfg(feature = "autodiff")]
mod ad {
    use burn::backend::{Backend, DispatchKindConversion};
    use burn::tensor::{DispatchTensor, Tensor};
    use burn_autodiff::checkpoint::base::Checkpointer;
    use burn_autodiff::checkpoint::strategy::NoCheckpointing;
    use burn_autodiff::grads::Gradients;
    use burn_autodiff::ops::{Backward, Ops, OpsKind};
    use burn_autodiff::Autodiff;

    #[derive(Debug)]
    struct SinkhornOp;

    impl<B: Backend> Backward<B, 1> for SinkhornOp
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        type State = usize;

        fn backward(
            self,
            ops: Ops<Self::State, 1>,
            grads: &mut Gradients,
            checkpointer: &mut Checkpointer,
        ) {
            let iters = ops.state;
            let logits = Tensor::<4>::from_primitive::<B>(
                checkpointer
                    .retrieve_node_output(ops.parents[0].as_ref().expect("logits checkpointed").id),
            );
            let d_out = Tensor::from_primitive::<B>(grads.consume::<B>(&ops.node));
            #[cfg(feature = "cuda")]
            {
                type CudaBare = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
                if std::any::TypeId::of::<B>() == std::any::TypeId::of::<CudaBare>() {
                    if let Some(dl) =
                        crate::sinkhorn_cuda::sinkhorn_backward_cuda(&logits, &d_out, iters)
                    {
                        let _ = &d_out;
                        grads.register::<B>(
                            ops.parents[0].clone().unwrap().id,
                            dl.try_into_primitive::<B>().unwrap(),
                        );
                        return;
                    } else if std::env::var("SKD3").is_ok() {
                        eprintln!("[bwd] fused downcast fail -> tensor fallback");
                    }
                }
            }
            let d_logits = crate::sinkhorn_cuda::sinkhorn_backward_tensor(&logits, &d_out, iters);
            grads.register::<B>(
                ops.parents[0].clone().unwrap().id,
                d_logits.try_into_primitive::<B>().unwrap(),
            );
        }
    }

    /// Fused Sinkhorn forward with exact (recomputed-sums) backward.
    pub fn sinkhorn_autodiff<Inner: Backend>(logits: Tensor<4>, iters: usize) -> Option<Tensor<4>>
    where
        DispatchTensor: DispatchKindConversion<Autodiff<Inner>> + DispatchKindConversion<Inner>,
    {
        let la = logits.try_into_primitive::<Autodiff<Inner>>().ok()?;
        let l_t = Tensor::from_primitive::<Inner>(la.primitive.clone());

        let out_t = {
            let mut m = l_t.exp();
            // The fused kernel already runs all `iters` normalizations in one
            // launch; only fall back to the tensor loop when it did not run
            // (non-CUDA backend or shape outside the kernel's guards).
            let mut fused = false;
            #[cfg(feature = "cuda")]
            {
                type CudaBare = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
                if std::any::TypeId::of::<Inner>() == std::any::TypeId::of::<CudaBare>() {
                    fused = crate::sinkhorn_cuda::sinkhorn_cuda(&mut m, iters);
                }
            }
            if !fused {
                let mut it = 0;
                while it < iters {
                    m = m.clone() / m.clone().sum_dim(3);
                    m = m.clone() / m.clone().sum_dim(2);
                    it += 1;
                }
            }
            m
        };

        let out_prim = out_t.try_into_primitive::<Inner>().unwrap();
        let nodes = [la.node.clone()];
        let prep = SinkhornOp.prepare::<NoCheckpointing>(nodes);
        let out_adt = match prep.compute_bound().stateful() {
            OpsKind::Tracked(mut prep) => {
                let _ids = [Some(prep.checkpoint(&la))];
                prep.finish(iters, out_prim)
            }
            OpsKind::UnTracked(prep) => prep.finish(out_prim),
        };
        Some(Tensor::from_primitive::<Autodiff<Inner>>(out_adt))
    }
}

#[cfg(feature = "autodiff")]
pub use ad::sinkhorn_autodiff;

/// Exact Sinkhorn backward: reverses the 2·iters row/column normalizations.
/// Only the per-step normalization sums are recomputed (tiny), the
/// intermediates are recovered as `m_pre = m_post · s`, so no matrix
/// checkpoints are needed.
pub fn sinkhorn_backward_tensor(logits: &Tensor<4>, d_out: &Tensor<4>, iters: usize) -> Tensor<4> {
    let mut m = logits.clone().exp();
    let mut sums = Vec::with_capacity(2 * iters);
    let mut i = 0;
    while i < iters {
        let rs = m.clone().sum_dim(3);
        sums.push(rs.clone());
        m = m.clone() / rs;
        let cs = m.clone().sum_dim(2);
        sums.push(cs.clone());
        m = m.clone() / cs;
        i += 1;
    }
    let mut d = d_out.clone();
    let mut m_post = m;
    let mut j = sums.len();
    while j > 0 {
        j -= 1;
        let s = &sums[j];
        // m_pre = m_post·s; correction = Σ d·m_pre over the normalized axis
        let pre = m_post.clone() * s.clone();
        let weighted = (d.clone() * pre.clone()).sum_dim(if j % 2 == 0 { 3 } else { 2 });
        d = d.clone() / s.clone() - weighted / (s.clone() * s.clone());
        m_post = pre;
    }
    d * logits.clone().exp()
}

#[cfg(all(test, feature = "autodiff", feature = "cuda"))]
mod ad_tests {
    use super::*;
    use burn::tensor::{Device, Distribution, Tensor};

    type CudaBare = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;

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
    #[ignore]
    fn sinkhorn_backward_bench() {
        let dev = Device::default();
        let (b, t, n, iters) = (8usize, 2048usize, 16usize, 20usize);
        let logits = Tensor::<4>::random([b, t, n, n], Distribution::Normal(0.0, 1.0), &dev);
        let dout = Tensor::<4>::random([b, t, n, n], Distribution::Normal(0.0, 1.0), &dev);
        for _ in 0..3 {
            let _ = sinkhorn_backward_cuda(&logits, &dout, iters).unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            let _ = sinkhorn_backward_cuda(&logits, &dout, iters).unwrap();
        }
        let _: f32 = sinkhorn_backward_cuda(&logits, &dout, iters)
            .unwrap()
            .sum()
            .into_scalar();
        let tf = t0.elapsed() / 20;
        let t0 = std::time::Instant::now();
        for _ in 0..3 {
            let dl = crate::sinkhorn_cuda::sinkhorn_backward_tensor(&logits, &dout, iters);
            let _: f32 = dl.sum().into_scalar();
        }
        let tt = t0.elapsed() / 3;
        println!(
            "[{b}x{t}x{n}] fused bwd {:?} tensor bwd {:?} ({:.0}x)",
            tf,
            tt,
            tt.as_secs_f64() / tf.as_secs_f64()
        );
    }

    #[test]
    fn sinkhorn_fused_backward_matches_tensor() {
        let dev = Device::default().autodiff();
        let (b, t, n, iters) = (2usize, 4usize, 8usize, 6usize);
        let logits = Tensor::<4>::random([b, t, n, n], Distribution::Normal(0.0, 1.0), &dev);

        // fused op graph
        let lf = logits.clone().require_grad();
        let outf = sinkhorn_autodiff::<CudaBare>(lf.clone(), iters).unwrap();
        let loss_f = outf.powf_scalar(2.0).sum();
        let grads_f = loss_f.backward();
        let dlf = lf.grad(&grads_f).unwrap();

        // plain tensor path graph
        let lt = logits.clone().require_grad();
        let outt = crate::sinkhorn_knopp(lt.clone(), iters);
        let loss_t = outt.powf_scalar(2.0).sum();
        let grads_t = loss_t.backward();
        let dlt = lt.grad(&grads_t).unwrap();

        let md = maxdiff(&to_host(dlf), &to_host(dlt));
        assert!(md < 1e-3, "sinkhorn grad maxdiff {md}");
    }
}

#[cfg(all(test, feature = "autodiff", feature = "cuda"))]
mod fd_tests {
    use super::*;
    use burn::tensor::{Device, Distribution, Tensor};

    type CudaBare = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;

    fn to_host(t: Tensor<4>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn sinkhorn_grad_matches_finite_difference() {
        // Independent check of the 40-step reverse unroll: the analytic grad
        // of sum(sinkhorn(logits)^2) must match central finite differences.
        let dev = Device::default();
        let adev = Device::default().autodiff();
        let (n, iters) = (4usize, 6usize);
        let logits = Tensor::<4>::random([1, 1, n, n], Distribution::Normal(0.0, 1.0), &dev);
        let data = logits.clone().into_data();

        // analytic grad (lib dispatch -> fused op)
        let lf = Tensor::<4>::from_data(data.clone(), &adev).require_grad();
        let out = crate::sinkhorn_knopp(lf.clone(), iters);
        let grads = out.powf_scalar(2.0).sum().backward();
        let analytic = to_host(lf.grad(&grads).unwrap());

        // central finite differences
        let eps = 1e-3f32;
        let total = n * n;
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
            let sp = crate::sinkhorn_knopp(xp, iters)
                .powf_scalar(2.0)
                .sum()
                .into_scalar::<f32>();
            let sm = crate::sinkhorn_knopp(xm, iters)
                .powf_scalar(2.0)
                .sum()
                .into_scalar::<f32>();
            fd[i] = (sp - sm) / (2.0 * eps);
        }
        for i in 0..total {
            let rel = (analytic[i] - fd[i]).abs() / (fd[i].abs() + 1e-6);
            assert!(
                rel < 2e-2,
                "idx {i}: analytic {} vs fd {} (rel {rel})",
                analytic[i],
                fd[i]
            );
        }
    }
}
