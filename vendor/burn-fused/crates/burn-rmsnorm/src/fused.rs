//! Fused RMSNorm CUDA kernel: x / sqrt(mean(x^2) + eps) * w in one launch
//! (vs ~5 tensor passes). One cube per row, 256 threads, shared reduction.
//!
//! The seam accounting lives here (ADR-0019): `rmsnorm_cuda` returning `None`
//! is a CORRECT tensor-ops answer, so a kernel that never engages looks
//! exactly like one that does - on the trainer's backend it does not engage
//! (the model's norm input is an autodiff tensor, not a bare `CubeTensor`).
//! `(asked, skipped)` is the pair that says so out loud; the trainer prints it
//! on the eval line.

/// `(times the fused kernel was asked for, times it was skipped for the
/// tensor path)`. `asked == skipped` is a DEAD kernel; `asked == 0` is a
/// build without the cuda feature. Both are incremented on the two sides of
/// the one `if let` in [`crate::RMSNorm::forward`], so they cannot disagree.
#[cfg(feature = "cuda")]
pub fn calls() -> (u64, u64) {
    (
        ASKED.load(std::sync::atomic::Ordering::Relaxed),
        SKIPPED.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Seam counters (ADR-0019): one increment per ask, one per fallback.
#[cfg(feature = "cuda")]
pub static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "cuda")]
pub static SKIPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// No fused path exists without the cuda feature, so nothing was asked and
/// nothing was skipped. The trainer's eval line prints this as `0/0`.
#[cfg(not(feature = "cuda"))]
pub fn calls() -> (u64, u64) {
    (0, 0)
}

#[cfg(feature = "cuda")]
use burn::backend::Backend;
#[cfg(feature = "cuda")]
use burn::tensor::Tensor;
#[cfg(feature = "cuda")]
use std::any::Any;

#[cfg(feature = "cuda")]
use cubecl::prelude::*;

#[cfg(feature = "cuda")]
/// One source of truth for the lane count. It is a `#[comptime]` PARAMETER, not
/// a local: `Shared::new_slice` sizes the shared-memory allocation, and on this
/// backend a runtime-sized one fails LLVM lowering with
/// `Expected operand type llvm.ptr, but found builtin.integer` before a byte is
/// written. `34c5631` did not fix this - it left `let threads = 256usize;` in
/// the body, so `new_slice` still saw a runtime value, and
/// `tests/fused_kernel_gate.rs` reproduced the identical failure. Every working
/// kernel in this tree takes the width as `#[comptime]`
/// (`burn-attnres/src/fused_attnres.rs:127`, `burn-gdn2/src/kernel/
/// chunk_adjoint_cube.rs:61`).
pub const THREADS: u32 = 256;

#[cube(launch_unchecked)]
fn rmsnorm_kernel<F: Float>(
    x: &[F],       // [B*T, D]
    w: &[F],       // [D]
    out: &mut [F], // [B*T, D]
    eps: f32,
    #[comptime] d: u32,
    #[comptime] threads: u32,
) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize;
    let threads = threads as usize;
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
    ASKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let cube = |t: Tensor<2>| -> Option<CubeTensor> {
        let prim = t.clone().try_into_primitive::<B>().ok()?;
        let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
        Some(c.clone())
    };
    let cube1 = |t: Tensor<1>| -> Option<CubeTensor> {
        let prim = t.clone().try_into_primitive::<B>().ok()?;
        let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
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
        rmsnorm_kernel::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(rows as u32, 1, 1),
            CubeDim::new_3d(THREADS, 1, 1),
            BufferArg::from_raw_parts(x_c.handle, rows * d),
            BufferArg::from_raw_parts(w_c.handle, d),
            BufferArg::from_raw_parts(out_c.handle, rows * d),
            eps,
            d as u32,
            THREADS,
        );
    }
    Some(out)
}
