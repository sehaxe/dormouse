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

/// Threads per cube. `#[comptime]` because `Shared::new_slice` sizes the
/// shared-memory allocation from it, and the reduction width and `CubeDim` must
/// not be able to disagree. **Keeping it `#[comptime]` was NOT the fix**, though
/// it was twice claimed to be (`34c5631`, then `9ac0377`) — the measurement is in
/// `tests/lower_probe.rs` and the loop header below is the fix.
#[cfg(feature = "cuda")]
pub const THREADS: u32 = 256;
/// `log2(THREADS)`, carried as its own `#[comptime]` parameter because the
/// reduction's trip count has to be comptime — see the loop in
/// [`rmsnorm_kernel`]. `fused_attnres.rs` carries the same quantity as
/// `log_threads` for the same reason.
#[cfg(feature = "cuda")]
pub const LOG_THREADS: u32 = THREADS.trailing_zeros();

/// The narrowest feature width the fused arm will accept, and the reason is a
/// measured defect rather than a limitation of the algorithm — see the `d < 4`
/// guard in [`rmsnorm_cuda`]. `d >= MIN_FUSED_D` is the only class measured to
/// write all its cubes.
#[cfg(feature = "cuda")]
pub const MIN_FUSED_D: usize = 4;


#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn rmsnorm_kernel<F: Float>(
    x: &[F],       // [B*T, D]
    w: &[F],       // [D]
    out: &mut [F], // [B*T, D]
    eps: f32,
    #[comptime] d: u32,
    #[comptime] threads: u32,
    #[comptime] log_threads: u32,
) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize;
    let threads = threads as usize;
    let log_threads = log_threads as usize;
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
    // Binary tree reduction over the shared partials (log2(threads) steps)
    // instead of a serial accumulation by thread 0: every step halves the
    // live lanes, so the reduction cost drops from O(T) serialized reads to
    // O(log T) parallel passes.
    //
    // **THE `for` OVER THE `while` IS THE WHOLE FIX, and it is load-bearing.**
    // This loop was `while s > 0 { if tid < s { ... } sync_cube(); s /= 2; }`
    // and the kernel did not lower: `the lowered module does not verify:
    // Expected operand type llvm.ptr, but found builtin.integer`. A cubecl
    // shared-memory read-modify-write at an index derived from a LOOP-CARRIED
    // runtime value does not survive CubeToLLVM; `tests/lower_probe.rs` is the
    // bisect, and it is a permanent test rather than a story:
    //
    //   p2  shared alloc + one write + one barrier + one read   LOWERS
    //   p4  a `sync_cube()` inside a runtime-bounded `while`      LOWERS
    //       (so it is not "a barrier in a loop", and
    //        chunk_cube.rs / sinkhorn_cuda.rs do exactly that)
    //   p6  the same kernel, reduction SERIALISED by thread 0      LOWERS, 1.7e-7
    //   p9  the halving `while` with NO barrier anywhere          FAILS
    //       (so it is not the barrier either)
    //   p5  the halving `while` with the barrier                  FAILS
    //   p10 THIS loop, `for k in 0..log_threads`                  LOWERS, 1.7e-7
    //
    // p10 is the same reduction, the same eight steps, the same addresses and
    // the same barrier per step — only the loop header differs, and the trip
    // count is `#[comptime]`, so cubecl unrolls the loop and every barrier
    // lands in straight-line code. This is the shape every other working
    // reduction in this tree already had: `dormouse-attnres/src/fused_attnres.rs:
    // 351-357` is the same eight lines, written this way, and it runs on every
    // training step.
    for k in 0..log_threads {
        let s = threads >> (k + 1);
        if tid < s {
            // Read into a temp first: cubecl shared-memory indexing is plain
            // Index/IndexMut, so `partial[a] += partial[b]` would alias a
            // mutable and an immutable borrow of the same slice.
            let other = partial[tid + s];
            partial[tid] += other;
        }
        sync_cube();
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
///
/// **THERE IS NO BACKWARD, and that is the whole story of this function's
/// reach.** `out` is a fresh `Tensor::empty` and the launch writes raw handles
/// into it, so the result carries no graph: were this reached from an autodiff
/// input, the arm would hand back a leaf, receive no gradient, and every loss
/// curve would still look healthy — the `dormouse_gdn2` defect. The only thing
/// standing between the trainer and that is `try_into_primitive`'s refusal,
/// because `DispatchKindConversion` rejects any dispatch tensor whose `autodiff`
/// field is not `Disabled` (burn-dispatch `src/tensor.rs:481-487`). So: wiring
/// this into a training forward means writing the adjoint first, not relaxing
/// a `?`. Measured both ways in `tests/rmsnorm_kernel_cuda.rs`.
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
    // **REFUSE d < 4. This is a guard against a defect that is NOT MINE, and
    // it is here because the alternative is a SILENT WRONG ANSWER.**
    //
    // Measured 2026-09-30 on this box, GPU, the kernel as it now stands
    // (lowering fixed): for `d` of 2 or 3 the trailing CUBES NEVER RUN, so
    // `out` comes back partly unwritten while the seam counter says the arm
    // ran. The fixture case `d2` (`dims 1 2 2`) is off by **5.245e-1**
    // relative, where the tensor path on the same input is 7.29e-8.
    //
    // What was ruled out by measurement, not by argument:
    //   * NOT the kernel body. A four-line kernel with no shared memory, no
    //     barrier and no comptime arithmetic (`out[CUBE_POS_X] = 1.0`) drops
    //     cubes the same way.
    //   * NOT the grid. `CubeCount::Static(4,1,1)` is honoured for some output
    //     shapes and not others: with the same kernel and the same 4-cube
    //     grid, output shape [8,2] ran 2 cubes and [16,1] ran 4. The count of
    //     cubes that execute is a function of the OUTPUT TENSOR'S SHAPE.
    //   * NOT the block size. `CubeDim` 32, 64, 128, 256, 512, 1024 all
    //     truncate identically.
    //   * NOT a readback race. Four consecutive `into_data()` reads agree, and
    //     an explicit `client.sync()` before the last one changes nothing.
    //
    // So the rule lives inside cubecl's launch path and is NOT isolated here;
    // `tests/d2_isolate.rs` is the minimal reproducer and the shape table.
    // What IS established is the safe envelope: every `d >= 4` measured
    // (d in {4, 8, 13, 16, 32}, rows 1..=12) writes all its cubes and matches
    // the reference, and every `d < 4` measured with `rows > 1` does not. This
    // guard therefore declines the whole uncertain class rather than
    // enumerating the ones that happen to work — the tensor path is CORRECT
    // and `SKIPPED` is incremented, so the fallback is COUNTED, not silent
    // (ADR-0019).
    if d < MIN_FUSED_D {
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
            LOG_THREADS,
        );
    }
    Some(out)
}
