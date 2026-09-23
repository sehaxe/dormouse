//! Fused RMSNorm CUDA kernel: x / sqrt(mean(x^2) + eps) * w in one launch
//! (vs ~5 tensor passes). One cube per row, 256 threads, shared reduction.

#[cfg(feature = "cuda")]
use burn::backend::Backend;
#[cfg(feature = "cuda")]
use burn::tensor::Tensor;
#[cfg(feature = "cuda")]
use std::any::Any;

#[cfg(feature = "cuda")]
use cubecl::prelude::*;

#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn rmsnorm_kernel<F: Float>(
    x: &[F],       // [B*T, D]
    w: &[F],       // [D]
    out: &mut [F], // [B*T, D]
    eps: f32,
    #[comptime] d: u32,
) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize;
    let threads = 256usize;
    let base = row * d;
    let mut partial = Shared::<[F]>::new_slice(threads);

    let mut sum = F::new(0.0_f32);
    let mut i = tid;
    while i < d {
        let v = x[base + i];
        sum += v * v;
        i += threads;
    }
    partial[tid] = sum;
    sync_cube();
    // Binary tree reduction over the shared partials (log2(256) = 8 steps)
    // instead of a serial accumulation by thread 0: every step halves the
    // live lanes, so the reduction cost drops from O(T) serialized reads to
    // O(log T) parallel passes.
    let mut s = threads / 2;
    while s > 0 {
        if tid < s {
            // Read into a temp first: cubecl shared-memory indexing is plain
            // Index/IndexMut, so `partial[a] += partial[b]` would alias a
            // mutable and an immutable borrow of the same slice.
            let other = partial[tid + s];
            partial[tid] += other;
        }
        sync_cube();
        s /= 2;
    }
    if tid == 0 {
        // mean + eps inside the sqrt (LLaMA/HF convention)
        partial[0] = (partial[0] / F::cast_from(d as f32) + F::cast_from(eps)).sqrt();
    }
    sync_cube();
    let inv = F::new(1.0_f32) / partial[0];
    let mut i = tid;
    while i < d {
        out[base + i] = x[base + i] * inv * w[i];
        i += threads;
    }
}

/// Fused RMSNorm on the bare CUDA backend. Returns `None` when the tensor is
/// not on CUDA (caller falls back to the tensor path).
#[cfg(feature = "cuda")]
pub fn rmsnorm_cuda<B: Backend>(x: Tensor<2>, weight: Tensor<1>, eps: f32) -> Option<Tensor<2>>
where
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    use burn_cubecl::tensor::CubeTensor;
    let cube = |t: Tensor<2>| -> Option<CubeTensor<cubecl::cuda::CudaRuntime>> {
        let prim = t.clone().try_into_primitive::<B>().ok()?;
        let c = (&prim as &dyn Any).downcast_ref::<CubeTensor<cubecl::cuda::CudaRuntime>>()?;
        Some(c.clone())
    };
    let cube1 = |t: Tensor<1>| -> Option<CubeTensor<cubecl::cuda::CudaRuntime>> {
        let prim = t.clone().try_into_primitive::<B>().ok()?;
        let c = (&prim as &dyn Any).downcast_ref::<CubeTensor<cubecl::cuda::CudaRuntime>>()?;
        Some(c.clone())
    };
    let [rows, d] = x.dims();
    if d == 0 || rows == 0 {
        return None;
    }
    let x_c = cube(x.clone())?;
    let w_c = cube1(weight.clone())?;
    // The kernel is hard-coded to f32 lanes (`launch_unchecked::<f32, ..>`):
    // a bf16/f16 buffer reinterpreted as f32 would produce garbage silently,
    // so bail out and let the caller's tensor path handle other dtypes.
    if x_c.dtype != burn::tensor::DType::F32 || w_c.dtype != burn::tensor::DType::F32 {
        return None;
    }
    let client = x_c.client.clone();
    let out = Tensor::<2>::empty([rows, d], &x.device());
    let out_c = cube(out.clone())?;
    unsafe {
        rmsnorm_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
            &client,
            CubeCount::Static(rows as u32, 1, 1),
            CubeDim::new_3d(256, 1, 1),
            BufferArg::from_raw_parts(x_c.handle, rows * d),
            BufferArg::from_raw_parts(w_c.handle, d),
            BufferArg::from_raw_parts(out_c.handle, rows * d),
            eps,
            d as u32,
        );
    }
    Some(out)
}
