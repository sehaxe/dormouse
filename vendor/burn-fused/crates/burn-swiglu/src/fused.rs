//! Fused SiLU-gated linear unit kernel: out[i] = silu(g[i]) * u[i] in one
//! launch (vs ~4 tensor passes: slice, silu, slice, mul).

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
fn swiglu_kernel<F: Float>(
    gu: &[F],      // [B*T, 2H]
    out: &mut [F], // [B*T, H]
    #[comptime] h: u32,
    #[comptime] threads: u32,
) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let h = h as usize;
    let threads = threads as usize;
    let base = row * (2 * h);
    let mut i = tid;
    while i < h {
        // silu(x) = x * sigmoid(x)
        let g = gu[base + i];
        let sg = F::new(1.0_f32) / (F::new(1.0_f32) + (-g).exp());
        out[row * h + i] = g * sg * gu[base + h + i];
        i += threads;
    }
}

/// Fused SwiGLU gate on the bare CUDA backend. Returns `None` when not on CUDA.
#[cfg(feature = "cuda")]
pub fn swiglu_cuda<B: Backend>(gu: Tensor<2>, h: usize) -> Option<Tensor<2>>
where
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    use burn_cubecl::tensor::CubeTensor;
    let cube = |t: Tensor<2>| -> Option<CubeTensor<cubecl::cuda::CudaRuntime>> {
        let prim = t.clone().try_into_primitive::<B>().ok()?;
        let c = (&prim as &dyn Any).downcast_ref::<CubeTensor<cubecl::cuda::CudaRuntime>>()?;
        Some(c.clone())
    };
    let [rows, d2] = gu.dims();
    if d2 != 2 * h || rows == 0 || h == 0 {
        return None;
    }
    let gu_c = cube(gu.clone())?;
    // f32-only kernel (see burn-rmsnorm/src/fused.rs for the rationale):
    // fall back to the tensor path for any other dtype.
    if gu_c.dtype != burn::tensor::DType::F32 {
        return None;
    }
    let client = gu_c.client.clone();
    let out = Tensor::<2>::empty([rows, h], &gu.device());
    let out_c = cube(out.clone())?;
    unsafe {
        swiglu_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
            &client,
            CubeCount::Static(rows as u32, 1, 1),
            CubeDim::new_3d(256, 1, 1),
            BufferArg::from_raw_parts(gu_c.handle, rows * d2),
            BufferArg::from_raw_parts(out_c.handle, rows * h),
            h as u32,
            256u32,
        );
    }
    Some(out)
}
