//! CUDA QR retraction kernels (feature `cuda`).
//!
//! Same math as [`crate::qr::qr_cpu`] (safe_qr of the paper: orthonormal Q
//! with non-negative R diagonal), computed as:
//!   1. Gram matrix G = A^T·A via the backend's matmul (cuBLAS-class, ~0.1
//!      ms for LLM shapes; a hand-written scalar kernel reached only
//!      ~40 GFLOP/s on this cubecl stack).
//!   2. Cholesky G = R^T·R on the host: k x k is tiny (64 KB at k=128), one
//!      thread finishes in microseconds. R's diagonal is positive by
//!      construction, which is exactly torch's sign(diag(R)) convention.
//!   3. `sct_qr_qsolve_kernel`: Q = A·R^-1 by forward substitution with a
//!      four-wide column block (independent FMAs hide latency); rows of Q
//!      are independent, so m threads each walk their own row with no
//!      cross-thread synchronization, and Q is written row-major directly.
//!
//! Accuracy: G = A^T·A squares the condition number; for retraction inputs
//! (near-orthonormal, kappa ~ 1.1-2) the f32 error stays ~1e-6, far inside
//! the 1e-4 reference tolerance. Results agree with the CPU Householder
//! path to ~1e-6 (verified by tests/cuda_retract.rs).
//!
//! On the bare CUDA `CubeBackend` this replaces the CPU path entirely
//! (no host round-trip for the data itself); every other backend falls back
//! to `qr_cpu`.
//!
//! Pitch rule: cubecl's `PitchedMemoryLayoutPolicy` pads 2D row widths to
//! `next_pow2(width_bytes).clamp(16, 512)` bytes, so a buffer whose last
//! dim is non-pow2 is row-pitched (e.g. 300-wide rows stride 384 elements).
//! Every kernel here takes each operand's row stride as a comptime arg
//! (`cube_of` guarantees row-major-with-pitch layout; the host passes
//! `meta.strides()[0]`), so non-pow2 in/rank shapes are read correctly.

use burn::backend::{Backend, DispatchKindConversion};
use burn::tensor::{DispatchTensor, Tensor};
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::CubeBackend;
use cubecl::prelude::*;
use std::any::Any;
use std::any::TypeId;

/// The bare (non-fusion) CUDA backend the kernels target.
pub type CudaBare = CubeBackend<cubecl::cuda::CudaRuntime>;

pub fn is_cuda<B: Backend>() -> bool {
    TypeId::of::<B>() == TypeId::of::<CudaBare>()
}

/// Owned copy of the underlying `CubeTensor` of `t` (see burn-gdn2's
/// `cube_of`). `None` when `B` is not the bare CUDA backend, or the layout
/// is not row-major with per-row pitch: the kernels index `(i, j)` as
/// `i*row_stride + j`, which needs element stride 1 and non-overlapping
/// rows. Row-pitched buffers (cubecl pads 2D row widths to
/// `next_pow2(width_bytes).clamp(16, 512)`, so a 300-wide row strides 384)
/// are fine — the row stride is passed to the kernels as a comptime arg.
pub fn cube_of<B: Backend>(t: &Tensor<2>) -> Option<CubeTensor<cubecl::cuda::CudaRuntime>>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    if !is_cuda::<B>() {
        return None;
    }
    let prim = t.clone().try_into_primitive::<B>().ok()?;
    let cube = (&prim as &dyn Any).downcast_ref::<CubeTensor<cubecl::cuda::CudaRuntime>>()?;
    let shape = cube.meta.shape().dims::<2>();
    let strides = cube.meta.strides().to_vec();
    if strides[1] != 1 || (shape[0] > 1 && strides[0] < shape[1]) {
        return None; // transposed/broadcast view, use the CPU path
    }
    Some(cube.clone())
}

/// Row stride (elements) of a 2D tensor accepted by [`cube_of`].
pub(crate) fn row_stride(cube: &CubeTensor<cubecl::cuda::CudaRuntime>) -> u32 {
    cube.meta.strides()[0] as u32
}

/// Like [`cube_of`] for an int tensor (counter buffers).
pub fn cube_of_int<B: Backend>(
    t: &Tensor<2, burn::tensor::Int>,
) -> Option<CubeTensor<cubecl::cuda::CudaRuntime>>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    if !is_cuda::<B>() {
        return None;
    }
    let prim = t.clone().try_into_primitive::<B>().ok()?;
    let cube = (&prim as &dyn Any).downcast_ref::<CubeTensor<cubecl::cuda::CudaRuntime>>()?;
    Some(cube.clone())
}

/// Gram matrix G = A^T·A of a row-major `[rows, cols]` matrix, one thread
/// per (i, j) pair; generic over the element type (f32 for the retract,
/// f64 for from_dense).
#[cube(launch_unchecked)]
fn sct_qr_gram_kernel<F: Float>(
    a: &[F],     // [rows, cols] row-major input
    g: &mut [F], // [cols, cols] Gram matrix
    #[comptime] rows: u32,
    #[comptime] cols: u32,
    _epoch: u32,
) {
    let i = (CUBE_POS_X * 16 + UNIT_POS_X) as usize;
    let j = (CUBE_POS_Y * 16 + UNIT_POS_Y) as usize;
    let cc = cols as usize;
    let rr = rows as usize;
    if i <= j && i < cc && j < cc {
        let mut acc = F::new(0.0_f32);
        let mut r = 0;
        while r < rr {
            acc += a[r * cc + i] * a[r * cc + j];
            r += 1;
        }
        g[i * cc + j] = acc + F::cast_from(_epoch) * F::new(0.0_f32);
        g[j * cc + i] = acc;
    }
}

/// Q = A·R^-1 by forward substitution, four columns at a time (the four
/// independent FMAs hide memory latency and give nvcc ILP to play with).
/// Each thread owns one row of Q, so no cross-thread synchronization is
/// needed; Q is written row-major directly.
#[cube(launch_unchecked)]
fn sct_qr_qsolve_kernel<F: Float>(
    a: &[F],     // [m, k] row-major input
    r: &[F],     // [k, k] upper-triangular factor
    q: &mut [F], // [m, k] row-major Q
    #[comptime] m: u32,
    #[comptime] k: u32,
    #[comptime] sa: u32,
    #[comptime] sr: u32,
    #[comptime] sq: u32,
) {
    let row = (CUBE_POS_Y * 256 + UNIT_POS_Y) as usize;
    let kk = k as usize;
    let mm = m as usize;
    if row < mm {
        let base = row * sq as usize;
        let abase = row * sa as usize;
        let mut j = 0;
        while j < kk {
            let j1 = j + 1;
            let j2 = j + 2;
            let j3 = j + 3;
            let w1 = j1 < kk;
            let w2 = j2 < kk;
            let w3 = j3 < kk;
            let mut acc0 = a[abase + j];
            let mut acc1 = if w1 { a[abase + j1] } else { F::new(0.0_f32) };
            let mut acc2 = if w2 { a[abase + j2] } else { F::new(0.0_f32) };
            let mut acc3 = if w3 { a[abase + j3] } else { F::new(0.0_f32) };
            let mut i = 0;
            while i < j {
                let qi = q[base + i];
                acc0 -= r[i * sr as usize + j] * qi;
                if w1 {
                    acc1 -= r[i * sr as usize + j1] * qi;
                }
                if w2 {
                    acc2 -= r[i * sr as usize + j2] * qi;
                }
                if w3 {
                    acc3 -= r[i * sr as usize + j3] * qi;
                }
                i += 1;
            }
            // intra-block corrections, sequentially
            let q0 = acc0 / r[j * sr as usize + j];
            q[base + j] = q0;
            if w1 {
                let q1 = (acc1 - r[j * sr as usize + j1] * q0) / r[j1 * sr as usize + j1];
                q[base + j1] = q1;
                if w2 {
                    let q2 = (acc2 - r[j * sr as usize + j2] * q0 - r[j1 * sr as usize + j2] * q1)
                        / r[j2 * sr as usize + j2];
                    q[base + j2] = q2;
                    if w3 {
                        let q3 = (acc3
                            - r[j * sr as usize + j3] * q0
                            - r[j1 * sr as usize + j3] * q1
                            - r[j2 * sr as usize + j3] * q2)
                            / r[j3 * sr as usize + j3];
                        q[base + j3] = q3;
                    }
                }
            }
            j += 4;
        }
    }
}

/// Tiled GEMM `C = A·B` (+ optional per-column scale on the output), one
/// 16x16 tile per block, one output element per thread, float4 accumulation
/// along K. A-loads are float4 (row-major, K multiple of 4); B-loads are
/// coalesced across the tile's j direction. Written because cubecl 0.11's
/// matmul collapses to ~20 GFLOP/s for skinny shapes (m=64, k=4096: 4.8 ms
/// vs ~50 us here) and is used by both the fused forward and the retract
/// Gram matrix.
#[cube(launch_unchecked)]
fn sct_gemm_kernel<F: Float>(
    a: &[F],     // [M, K] row-major
    b: &[F],     // [K, N] row-major
    c: &mut [F], // [M, N] row-major output
    scale: &[F], // [N] optional per-column scale (all-ones when unused)
    #[comptime] m: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
    #[comptime] scaled: bool,
    #[comptime] sa: u32,
    #[comptime] sb: u32,
    #[comptime] sc: u32,
) {
    let ti = UNIT_POS_X as usize;
    let tj = UNIT_POS_Y as usize;
    let i = CUBE_POS_X as usize * 16 + ti;
    let j = CUBE_POS_Y as usize * 16 + tj;
    let mm = m as usize;
    let nn = n as usize;
    let kk = k as usize;
    let mut acc = F::new(0.0_f32);
    if i < mm && j < nn {
        let mut kk4 = 0;
        while kk4 + 4 <= kk {
            let a0 = a[i * sa as usize + kk4];
            let a1 = a[i * sa as usize + kk4 + 1];
            let a2 = a[i * sa as usize + kk4 + 2];
            let a3 = a[i * sa as usize + kk4 + 3];
            acc += a0 * b[kk4 * sb as usize + j];
            acc += a1 * b[(kk4 + 1) * sb as usize + j];
            acc += a2 * b[(kk4 + 2) * sb as usize + j];
            acc += a3 * b[(kk4 + 3) * sb as usize + j];
            kk4 += 4;
        }
        while kk4 < kk {
            acc += a[i * sa as usize + kk4] * b[kk4 * sb as usize + j];
            kk4 += 1;
        }
        if scaled {
            c[i * sc as usize + j] = acc * scale[j];
        } else {
            c[i * sc as usize + j] = acc;
        }
    }
}

/// GEMM with the right operand transposed: `C[i][j] = sum_k A[i][k]·B[j][k]`
/// with both `A [M, K]` and `B [N, K]` row-major. Both operands stream as
/// float4 along K (coalesced), so no transpose copy is needed: this computes
/// `t·V^T` directly from the stored `V [n, k]`.
#[cube(launch_unchecked)]
fn sct_gemm_t_kernel<F: Float>(
    a: &[F],     // [M, K] row-major
    b: &[F],     // [N, K] row-major (the transposed operand, stored untransposed)
    c: &mut [F], // [M, N] row-major output
    #[comptime] m: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
    #[comptime] sa: u32,
    #[comptime] sb: u32,
    #[comptime] sc: u32,
) {
    let ti = UNIT_POS_X as usize;
    let tj = UNIT_POS_Y as usize;
    let i = CUBE_POS_X as usize * 16 + ti;
    let j = CUBE_POS_Y as usize * 16 + tj;
    let mm = m as usize;
    let nn = n as usize;
    let kk = k as usize;
    let mut acc = F::new(0.0_f32);
    if i < mm && j < nn {
        let mut kk4 = 0;
        while kk4 + 4 <= kk {
            let a0 = a[i * sa as usize + kk4];
            let a1 = a[i * sa as usize + kk4 + 1];
            let a2 = a[i * sa as usize + kk4 + 2];
            let a3 = a[i * sa as usize + kk4 + 3];
            let b0 = b[j * sb as usize + kk4];
            let b1 = b[j * sb as usize + kk4 + 1];
            let b2 = b[j * sb as usize + kk4 + 2];
            let b3 = b[j * sb as usize + kk4 + 3];
            acc += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            kk4 += 4;
        }
        while kk4 < kk {
            acc += a[i * sa as usize + kk4] * b[j * sb as usize + kk4];
            kk4 += 1;
        }
        c[i * sc as usize + j] = acc;
    }
}

/// Fused forward `y = (x@U)·s @ V^T` with the two custom GEMM kernels above
/// (one launch each; the scale is folded into the first GEMM's output write).
/// `None` when the backend is not the bare CUDA one or the shapes are
/// outside the kernel limits, in which case the caller falls back to tensor
/// ops. Beats cubecl's matmul by 50-100x on skinny LLM shapes (m=64 rows).
pub fn forward_cuda<B: Backend>(
    x: Tensor<2>,
    u: Tensor<2>,
    s: Tensor<1>,
    v: Tensor<2>,
) -> Option<Tensor<2>>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let [b, m] = x.dims();
    let k = u.dims()[1];
    let n = v.dims()[0];
    if b == 0
        || m == 0
        || k == 0
        || n == 0
        || !b.is_multiple_of(16)
        || !k.is_multiple_of(4)
        || !m.is_multiple_of(4)
    {
        return None;
    }
    let xc = cube_of::<B>(&x)?;
    let uc = cube_of::<B>(&u)?;
    let sc = cube_of::<B>(&s.unsqueeze_dims(&[0]))?;
    let vc = cube_of::<B>(&v)?;
    let client = xc.client.clone();
    let device = x.device();

    let t = Tensor::<2>::zeros([b, k], &device);
    let y = Tensor::<2>::zeros([b, n], &device);
    let tc = cube_of::<B>(&t)?;
    let yc = cube_of::<B>(&y)?;

    unsafe {
        sct_gemm_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
            &client,
            CubeCount::Static(b.div_ceil(16) as u32, k.div_ceil(16) as u32, 1),
            CubeDim::new_3d(16, 16, 1),
            BufferArg::from_raw_parts(xc.handle.clone(), b * m),
            BufferArg::from_raw_parts(uc.handle.clone(), m * k),
            BufferArg::from_raw_parts(tc.handle.clone(), b * k),
            BufferArg::from_raw_parts(sc.handle.clone(), k),
            b as u32,
            k as u32,
            m as u32,
            true,
            row_stride(&xc),
            row_stride(&uc),
            row_stride(&tc),
        );
        sct_gemm_t_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
            &client,
            CubeCount::Static(b.div_ceil(16) as u32, n.div_ceil(16) as u32, 1),
            CubeDim::new_3d(16, 16, 1),
            BufferArg::from_raw_parts(tc.handle.clone(), b * k),
            BufferArg::from_raw_parts(vc.handle.clone(), n * k),
            BufferArg::from_raw_parts(yc.handle.clone(), b * n),
            b as u32,
            n as u32,
            k as u32,
            row_stride(&tc),
            row_stride(&vc),
            row_stride(&yc),
        );
    }
    // No host sync: the raw launches and every later burn op on `y` share
    // this client, whose server executes tasks in FIFO order, so the kernels
    // read x/u/s/v after their producers and any consumer of `y` runs after
    // the kernels. retract_cuda provides the one per-step barrier (retraction
    // runs once per training step); a per-forward sync would serialize every
    // layer of the pipeline.
    //
    // TODO(gpu): re-verify on device; if burn defers op materialization again
    // (0.22-pre.1 did), restore a single block_on(client.sync()) here.
    Some(y)
}

/// One round of the round-robin one-sided Jacobi: one thread per pair,
/// disjoint pairs per round, a-rotations applied immediately, V rotations
/// appended to deferred lists (u32 p/q, f64 c/s). Launched once per round;
/// the queue serializes rounds, so no in-kernel barrier is needed.
#[cube(launch_unchecked)]
fn sct_jacobi_round_kernel<F: Float>(
    a: &mut [F], // [m*m] column-major
    rotp: &mut [u32],
    rotq: &mut [u32],
    rotc: &mut [F],
    rots: &mut [F],
    #[comptime] m: u32,
    round: u32,
) {
    let tid = (CUBE_POS_X * CUBE_DIM + UNIT_POS_X) as usize;
    let mm = m as usize;
    let npairs = mm / 2;
    if tid < npairs {
        let r = round as usize;
        let (p, q) = if tid < npairs - 1 {
            ((r + tid) % (mm - 1), (r + mm - 2 - tid) % (mm - 1))
        } else {
            ((r + npairs - 1) % (mm - 1), mm - 1)
        };
        let mut alpha = F::new(0.0_f32);
        let mut beta = F::new(0.0_f32);
        let mut gamma = F::new(0.0_f32);
        let mut rr = 0;
        while rr < mm {
            let pv = a[p * mm + rr];
            let qv = a[q * mm + rr];
            alpha += pv * pv;
            beta += qv * qv;
            gamma += pv * qv;
            rr += 1;
        }
        if gamma.abs() > F::new(1e-12_f32) * (alpha * beta).sqrt() {
            let zeta = (beta - alpha) / (F::new(2.0_f32) * gamma);
            let sg = if gamma < F::new(0.0_f32) {
                F::new(-1.0_f32)
            } else {
                F::new(1.0_f32)
            };
            let t = sg / (zeta.abs() + (F::new(1.0_f32) + zeta * zeta).sqrt());
            let c = F::new(1.0_f32) / (F::new(1.0_f32) + t * t).sqrt();
            let s = t * c;
            rr = 0;
            while rr < mm {
                let ap = a[p * mm + rr];
                let aq = a[q * mm + rr];
                a[p * mm + rr] = c * ap + s * aq;
                a[q * mm + rr] = c * aq - s * ap;
                rr += 1;
            }
            rotp[tid] = p as u32;
            rotq[tid] = q as u32;
            rotc[tid] = c;
            rots[tid] = s;
        } else {
            rotp[tid] = u32::MAX;
            rotq[tid] = u32::MAX;
            rotc[tid] = F::new(0.0_f32);
            rots[tid] = F::new(0.0_f32);
        }
    }
}

/// Deferred V application: one thread per row of V, walks the rotation
/// lists and applies the rotations touching its row, in list order.
#[cube(launch_unchecked)]
fn sct_jacobi_vpass_kernel<F: Float>(
    rotp: &[u32],
    rotq: &[u32],
    rotc: &[F],
    rots: &[F],
    v: &mut [F], // [m*m] row-major V accumulator
    #[comptime] m: u32,
) {
    let row = (CUBE_POS_Y * 256 + UNIT_POS_Y) as usize;
    let mm = m as usize;
    let npairs = mm / 2;
    if row < mm {
        let mut i = 0;
        while i < npairs {
            let p = rotp[i] as usize;
            let q = rotq[i] as usize;
            if p != u32::MAX as usize && (row == p || row == q) {
                let other = if row == p { q } else { p };
                let c = rotc[i];
                let s = rots[i];
                let mut rr = 0;
                while rr < mm {
                    let vp = v[row * mm + rr];
                    let vq = v[other * mm + rr];
                    v[row * mm + rr] = c * vp + s * vq;
                    v[other * mm + rr] = c * vq - s * vp;
                    rr += 1;
                }
            }
            i += 1;
        }
    }
}

/// Cast f32 -> f64, one thread per element.
#[cube(launch_unchecked)]
fn sct_cast_f64_kernel(a: &[f32], out: &mut [f64], #[comptime] n: u32) {
    let i = (CUBE_POS_X * CUBE_DIM + UNIT_POS_X) as usize;
    if i < n as usize {
        out[i] = f64::cast_from(a[i]);
    }
}

/// Generic transpose of a row-major `[rows, cols]` matrix into row-major
/// `[cols, rows]`.
#[cube(launch_unchecked)]
fn sct_transpose_kernel<F: Float>(
    a: &[F],
    out: &mut [F],
    #[comptime] rows: u32,
    #[comptime] cols: u32,
) {
    let i = (CUBE_POS_X * CUBE_DIM + UNIT_POS_X) as usize;
    if i < (rows * cols) as usize {
        let r = i / cols as usize;
        let c = i % cols as usize;
        out[c * rows as usize + r] = a[i];
    }
}

/// GPU retraction of `matrix` on the bare CUDA backend. `None` when the
/// backend is not CUDA or the shape is outside the kernel limits (then the
/// CPU path in [`crate::orthogonalize`] is used).
/// Full GPU truncated SVD for `from_dense` (tall case n >= m), mirroring the
/// CPU QR-reduced path but in f64 end to end (Gram -> Cholesky -> Q = A·R^-1
/// -> one-sided Jacobi), so the reconstruction is exact to ~1e-12 and the
/// one-sided Jacobi runs on the GPU's bandwidth instead of the CPU's
/// (~650 s -> ~5-10 s at LLM scale; torch's gesdd takes ~60 s).
///
/// Returns the flat layer factors `(u_flat [m,k], s [k], v_flat [n,k])` in
/// the same layout `from_dense` expects, or `None` for wide matrices or
/// shapes outside the kernel limits (the caller falls back to the CPU path).
pub fn from_dense_cuda<B: Backend>(
    dense_weight: Tensor<2>,
    rank: usize,
    sweeps: usize,
) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let [n, m] = dense_weight.dims();
    let k = rank.min(m).min(n);
    if n < m || k == 0 || m % 2 != 0 || m > 4096 {
        return None;
    }
    // A to the host once, then the vendored Golub-Kahan + dbdsqr SVD in f32
    // with the crate's own sweep convention (matching the burn svd PR perf
    // table: 15 sweeps, f32). The previous pipeline (Gram kernel + host
    // Cholesky + 255x15 one-sided Jacobi round launches + V pass + 4 gemms)
    // was launch-dispatch bound on cubecl 0.11-pre: ~0.15 ms per round
    // launch. The host path is O(m n^2) scalar math, deterministic, and
    // orders of magnitude fewer round trips.
    let a_host = dense_weight.clone().into_data();
    let a_f32 = a_host.try_to_vec::<f32>().ok()?;

    // svd_host requires m >= n: A is [n, m] with n >= m (checked above).
    let sweeps = sweeps.max(15);
    let (u, sigma, vt) = crate::host_svd::svd_host(&a_f32, n, m, 1, sweeps, false);

    // top-k extraction: U [n, k], S [k], V [m, k] as f32, column-major-ish
    // layout matching the previous Jacobi output (u_flat[r*k+i], v_flat[r*k+i])
    let mut u_flat = vec![0.0f32; n * k];
    let mut v_flat = vec![0.0f32; m * k];
    let mut s_flat = vec![0.0f32; k];
    for i in 0..k {
        s_flat[i] = sigma[i];
        for r in 0..n {
            u_flat[r * k + i] = u[r * m + i];
        }
        for r in 0..m {
            v_flat[r * k + i] = vt[i * m + r];
        }
    }
    Some((u_flat, s_flat, v_flat))
}

pub fn retract_cuda<B: Backend>(matrix: Tensor<2>) -> Option<Tensor<2>>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let a_cube = cube_of::<B>(&matrix)?;
    let [m, k] = matrix.dims();
    if m == 0 || k == 0 || k > 1024 || m < 256 || k < 16 {
        return None; // too small: the host round trip costs more than qr_cpu
    }
    let client = a_cube.client.clone();
    let device = matrix.device();

    // 1. Gram matrix via the backend matmul (lazy tensor op; into_data
    //    materializes it and copies the 64 KB result to the host).
    let g = matrix.clone().transpose().matmul(matrix.clone());
    let g_data = g.into_data();
    let g_flat = g_data.try_to_vec::<f32>().ok()?;

    // 2. Cholesky on the host (microseconds for k <= 256).
    let r_flat = crate::qr::cholesky_host(&g_flat, k);
    let r_tensor = Tensor::<1>::from_floats(r_flat.as_slice(), &device).reshape([k, k]);

    // 3. Q = A·R^-1, one thread per row, four columns at a time.
    let r_cube = cube_of::<B>(&r_tensor)?;
    let q = Tensor::<2>::zeros([m, k], &device);
    let q_cube = cube_of::<B>(&q)?;
    let (sa, sr, sq) = (
        row_stride(&a_cube),
        row_stride(&r_cube),
        row_stride(&q_cube),
    );
    unsafe {
        sct_qr_qsolve_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
            &client,
            CubeCount::Static(1, m.div_ceil(256) as u32, 1),
            CubeDim::new_3d(1, 256, 1),
            BufferArg::from_raw_parts(a_cube.handle, m * k),
            BufferArg::from_raw_parts(r_cube.handle.clone(), k * k),
            BufferArg::from_raw_parts(q_cube.handle.clone(), m * k),
            m as u32,
            k as u32,
            sa,
            sr,
            sq,
        );
    }
    // Block on the server queue: burn 0.22 tensors are lazy, and a
    // raw-handle launch would otherwise be dropped or deferred past the
    // caller's next read. One sync per retract is acceptable: retraction
    // runs once per training step anyway.
    let _ = futures_lite::future::block_on(client.sync());

    Some(q)
}
