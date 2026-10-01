//! Fused CUDA elementwise kernels for the Muon+ hot path.
//!
//! The optimizer's per-step elementwise passes (momentum, NS polynomial
//! combine, row/col normalization, final decay+update) dominate on CUDA —
//! ~45 tensor passes over [r, c] per step vs ~15 cuBLAS matmuls. Fusing them
//! cuts the launches and memory traffic several-fold.

use burn::tensor::Tensor;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use std::any::Any;

/// x = a·x + b·t1 + c·t2 (NS polynomial combine; one launch vs 5 passes).
#[cube(launch_unchecked)]
fn ns_combine_kernel<F: Float>(
    x: &mut [F],
    t1: &[F],
    t2: &[F],
    a: f32,
    b: f32,
    c: f32,
    total: u32,
    #[comptime] cols: u32,
    #[comptime] s0: u32,
    #[comptime] s1: u32,
) {
    let idx = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    if idx < total {
        // Logical (i, j) -> physical i*s0 + j*s1: buffers are row-pitched
        // (s0 >= cols), so a flat walk over r*c would read padding zeros.
        let i = idx / cols;
        let j = idx - i * cols;
        let pos = (i * s0 + j * s1) as usize;
        let v = x[pos];
        x[pos] = v * F::cast_from(a) + t1[pos] * F::cast_from(b) + t2[pos] * F::cast_from(c);
    }
}

/// m = μ·m + (1−μ)·g (momentum update; one launch vs 3 passes).
#[cube(launch_unchecked)]
fn momentum_kernel<F: Float>(
    m: &mut [F],
    g: &[F],
    mu: f32,
    total: u32,
    #[comptime] cols: u32,
    #[comptime] s0: u32,
    #[comptime] s1: u32,
) {
    let idx = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    if idx < total {
        let i = idx / cols;
        let j = idx - i * cols;
        let pos = (i * s0 + j * s1) as usize;
        m[pos] = m[pos] * F::cast_from(mu) + g[pos] * F::cast_from(1.0 - mu);
    }
}

/// out = (x − η·u)·(1−wd) fused (final decay + update; one launch vs 5 passes).
#[cube(launch_unchecked)]
fn finalize_kernel<F: Float>(
    x: &mut [F],
    u: &[F],
    eta: f32,
    wd: f32,
    total: u32,
    #[comptime] cols: u32,
    #[comptime] s0: u32,
    #[comptime] s1: u32,
) {
    let idx = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    if idx < total {
        let i = idx / cols;
        let j = idx - i * cols;
        let pos = (i * s0 + j * s1) as usize;
        x[pos] = x[pos] * F::cast_from(1.0 - wd) - u[pos] * F::cast_from(eta);
    }
}

fn cube_of<const D: usize>(t: &Tensor<D>) -> Option<CubeTensor> {
    type B = burn_cubecl::CubeBackend;
    let prim = t.clone().try_into_primitive::<B>().ok()?;
    let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
    Some(c.clone())
}

/// Fused NS polynomial combine: `x ← a·x + b·t1 + c·t2`. Returns false when
/// the tensors are not on the bare CUDA backend (caller falls back).
pub fn ns_combine_cuda<const D: usize>(
    x: &mut Tensor<D>,
    t1: &Tensor<D>,
    t2: &Tensor<D>,
    a: f32,
    b: f32,
    c: f32,
) -> bool {
    let dims = x.dims();
    let cols = dims[1];
    let n = dims[0] * cols;
    if n == 0 {
        return false;
    }
    if let (Some(xc), Some(t1c), Some(t2c)) = (cube_of(x), cube_of(t1), cube_of(t2)) {
        let client = xc.client.clone();
        let st = xc.meta.strides();
        let (s0, s1) = (st[0] as u32, st[1] as u32);
        let threads = 256u32;
        let cubes = (n as u32).div_ceil(threads);
        let dim = CubeDim::new_3d(threads, 1, 1);
        let count = CubeCount::Static(cubes, 1, 1);
        unsafe {
            ns_combine_kernel::launch_unchecked::<f32>(
                &client,
                count,
                dim,
                BufferArg::from_raw_parts(xc.handle, n),
                BufferArg::from_raw_parts(t1c.handle, n),
                BufferArg::from_raw_parts(t2c.handle, n),
                a,
                b,
                c,
                n as u32,
                cols as u32,
                s0,
                s1,
            );
        }
        true
    } else {
        false
    }
}

/// Fused momentum update: `m ← μ·m + (1−μ)·g`.
pub fn momentum_cuda<const D: usize>(m: &mut Tensor<D>, g: &Tensor<D>, mu: f32) -> bool {
    let dims = m.dims();
    let cols = dims[1];
    let n = dims[0] * cols;
    if n == 0 {
        return false;
    }
    if let (Some(mc), Some(gc)) = (cube_of(m), cube_of(g)) {
        let client = mc.client.clone();
        let st = mc.meta.strides();
        let (s0, s1) = (st[0] as u32, st[1] as u32);
        let threads = 256u32;
        let cubes = (n as u32).div_ceil(threads);
        let dim = CubeDim::new_3d(threads, 1, 1);
        let count = CubeCount::Static(cubes, 1, 1);
        unsafe {
            momentum_kernel::launch_unchecked::<f32>(
                &client,
                count,
                dim,
                BufferArg::from_raw_parts(mc.handle, n),
                BufferArg::from_raw_parts(gc.handle, n),
                mu,
                n as u32,
                cols as u32,
                s0,
                s1,
            );
        }
        true
    } else {
        false
    }
}

/// Fused final step: `x ← x·(1−wd) − η·u`.
pub fn finalize_cuda<const D: usize>(x: &mut Tensor<D>, u: &Tensor<D>, eta: f32, wd: f32) -> bool {
    let dims = x.dims();
    let cols = dims[1];
    let n = dims[0] * cols;
    if n == 0 {
        return false;
    }
    if let (Some(xc), Some(uc)) = (cube_of(x), cube_of(u)) {
        let client = xc.client.clone();
        let st = xc.meta.strides();
        let (s0, s1) = (st[0] as u32, st[1] as u32);
        let threads = 256u32;
        let cubes = (n as u32).div_ceil(threads);
        let dim = CubeDim::new_3d(threads, 1, 1);
        let count = CubeCount::Static(cubes, 1, 1);
        unsafe {
            finalize_kernel::launch_unchecked::<f32>(
                &client,
                count,
                dim,
                BufferArg::from_raw_parts(xc.handle, n),
                BufferArg::from_raw_parts(uc.handle, n),
                eta,
                wd,
                n as u32,
                cols as u32,
                s0,
                s1,
            );
        }
        true
    } else {
        false
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use burn::tensor::{Device, Distribution, Tensor};

    fn cuda_dev() -> Device {
        Device::default()
    }

    #[test]
    fn norm_colrow_match() {
        let dev = cuda_dev();
        for (r, c) in [(512usize, 256), (500, 256), (500, 300), (33, 260), (1, 64)] {
            let mut x: Tensor<2> = Tensor::random([r, c], Distribution::Normal(0.0, 1.0), &dev);
            let x_ref = {
                let cn = x.clone().mul(x.clone()).sum_dim(0).sqrt().clamp_min(1e-7);
                let y = x.clone().div(cn);
                let rn = y.clone().mul(y.clone()).sum_dim(1).sqrt().clamp_min(1e-7);
                y.div(rn)
            };
            assert!(norm_colrow_cuda(&mut x, 1e-7), "norm kernel should run");
            let diff: f32 = (x - x_ref).abs().max().into_scalar::<f32>();
            assert!(diff < 1e-4, "norm_colrow diff {diff} [{r}x{c}]");
        }
    }

    #[test]
    #[ignore]
    fn ortho_bench() {
        let dev = cuda_dev();
        for (r, c) in [(4096usize, 4096usize), (8192, 1024)] {
            let g: Tensor<2> = Tensor::random([r, c], Distribution::Normal(0.0, 1.0), &dev);
            for _ in 0..3 {
                let _ = ortho_fused(g.clone(), true);
                let _ = ortho_fused(g.clone(), false);
            }
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                let _ = ortho_fused(g.clone(), true);
            }
            let tf = t0.elapsed() / 10;
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                let _ = ortho_fused(g.clone(), false);
            }
            let tt = t0.elapsed() / 10;
            println!(
                "[{r}x{c}] fused {:?} tensor {:?} ({:.1}x)",
                tf,
                tt,
                tt.as_secs_f64() / tf.as_secs_f64()
            );
        }
    }

    fn full_fused(g: Tensor<2>, fused: bool) -> Tensor<2> {
        let mut x = g;
        let norm = x.clone().mul(x.clone()).sum().sqrt().clamp_min(1e-7);
        x = x.div(norm.unsqueeze());
        for _ in 0..5 {
            let xt = x.clone().swap_dims(0, 1);
            let xx = x.clone().matmul(xt);
            let t1 = xx.clone().matmul(x.clone());
            let t2 = xx.matmul(t1.clone());
            if fused {
                let _ = ns_combine_cuda(&mut x, &t1, &t2, 3.25, -2.0, 0.25);
            } else {
                x = x
                    .clone()
                    .mul_scalar(3.25)
                    .add(t1.mul_scalar(-2.0))
                    .add(t2.mul_scalar(0.25));
            }
        }
        if fused {
            let _ = norm_colrow_cuda(&mut x, 1e-7);
        } else {
            let cn = x.clone().mul(x.clone()).sum_dim(0).sqrt().clamp_min(1e-7);
            let y = x.div(cn);
            let rn = y.clone().mul(y.clone()).sum_dim(1).sqrt().clamp_min(1e-7);
            x = y.div(rn);
        }
        x
    }

    #[test]
    #[ignore]
    fn step_bench() {
        let dev = cuda_dev();
        for (r, c) in [(4096usize, 4096usize), (8192, 1024)] {
            let g: Tensor<2> = Tensor::random([r, c], Distribution::Normal(0.0, 1.0), &dev);
            for _ in 0..3 {
                let _ = full_fused(g.clone(), true);
                let _ = full_fused(g.clone(), false);
            }
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                let _ = full_fused(g.clone(), true);
            }
            let tf = t0.elapsed() / 10;
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                let _ = full_fused(g.clone(), false);
            }
            let tt = t0.elapsed() / 10;
            println!(
                "[{r}x{c}] full fused {:?} tensor {:?} ({:.1}x)",
                tf,
                tt,
                tt.as_secs_f64() / tf.as_secs_f64()
            );
        }
    }

    fn ortho_fused(g: Tensor<2>, fused: bool) -> Tensor<2> {
        let mut x = g;
        let norm = x.clone().mul(x.clone()).sum().sqrt().clamp_min(1e-7);
        x = x.div(norm.unsqueeze());
        for _ in 0..5 {
            let xt = x.clone().swap_dims(0, 1);
            let xx = x.clone().matmul(xt);
            let t1 = xx.clone().matmul(x.clone());
            let t2 = xx.matmul(t1.clone());
            if fused {
                let _ = ns_combine_cuda(&mut x, &t1, &t2, 3.25, -2.0, 0.25);
            } else {
                x = x
                    .clone()
                    .mul_scalar(3.25)
                    .add(t1.mul_scalar(-2.0))
                    .add(t2.mul_scalar(0.25));
            }
        }
        x
    }

    #[test]
    fn fused_match_tensor() {
        let dev = cuda_dev();
        for (r, c) in [(1024usize, 1024usize), (1024, 300)] {
            // 300-col tensors are row-pitched (stride 384) — the fused path
            // must index with the real strides, not the logical col count.
            let mut x: Tensor<2> = Tensor::random([r, c], Distribution::Normal(0.0, 1.0), &dev);
            let t1: Tensor<2> = Tensor::random([r, c], Distribution::Normal(0.0, 1.0), &dev);
            let t2: Tensor<2> = Tensor::random([r, c], Distribution::Normal(0.0, 1.0), &dev);
            let x_ref = x
                .clone()
                .mul_scalar(3.25)
                .add(t1.clone().mul_scalar(-2.0))
                .add(t2.clone().mul_scalar(0.25));
            assert!(
                ns_combine_cuda(&mut x, &t1, &t2, 3.25, -2.0, 0.25),
                "kernel should run"
            );
            let diff: f32 = (x - x_ref).abs().max().into_scalar::<f32>();
            assert!(diff < 1e-5, "ns_combine diff {diff} [{r}x{c}]");

            // momentum
            let mut m: Tensor<2> = Tensor::random([r, c], Distribution::Normal(0.0, 1.0), &dev);
            let g: Tensor<2> = Tensor::random([r, c], Distribution::Normal(0.0, 1.0), &dev);
            let m_ref = m.clone().mul_scalar(0.95).add(g.clone().mul_scalar(0.05));
            assert!(
                momentum_cuda(&mut m, &g, 0.95),
                "momentum kernel should run"
            );
            let diff: f32 = (m - m_ref).abs().max().into_scalar::<f32>();
            assert!(diff < 1e-5, "momentum diff {diff} [{r}x{c}]");

            // finalize
            let mut p: Tensor<2> = Tensor::random([r, c], Distribution::Normal(0.0, 1.0), &dev);
            let u: Tensor<2> = Tensor::random([r, c], Distribution::Normal(0.0, 1.0), &dev);
            let p_ref = p.clone().mul_scalar(0.99).sub(u.clone().mul_scalar(0.001));
            assert!(
                finalize_cuda(&mut p, &u, 0.001, 0.01),
                "finalize kernel should run"
            );
            let diff: f32 = (p - p_ref).abs().max().into_scalar::<f32>();
            assert!(diff < 1e-5, "finalize diff {diff} [{r}x{c}]");
        }
    }
}

/// Col-wise L2 normalize: each column divided by sqrt(Σ_r x²). Two-pass
/// coalesced reduction: pass 1 sums x² over 32-row slabs (consecutive threads
/// read consecutive columns of the same row), pass 2 applies the norms.
#[cube(launch_unchecked)]
fn norm_col_partials_kernel<F: Float>(
    x: &[F],
    partials: &mut [F], // [ceil(r/32), c]
    #[comptime] r: u32,
    #[comptime] c: u32,
    #[comptime] s0: u32,
    #[comptime] s1: u32,
    #[comptime] ps0: u32,
    #[comptime] ps1: u32,
) {
    let slab = CUBE_POS_X as usize;
    let cj = CUBE_POS_Y as usize;
    let tid = UNIT_POS_X as usize;
    let r0 = slab * 32;
    let rmax = (r as usize).min(r0 + 32);
    let c0 = cj * 256;
    let col = c0 + tid;
    if col < c as usize {
        let mut sum = F::new(0.0_f32);
        let mut i = r0;
        while i < rmax {
            let v = x[i * (s0 as usize) + col * (s1 as usize)];
            sum += v * v;
            i += 1;
        }
        partials[slab * (ps0 as usize) + col * (ps1 as usize)] = sum;
    }
}

/// Pass 2: divide each 32x256 tile by its column norm, computed from the
/// per-slab partials into shared memory.
#[cube(launch_unchecked)]
fn norm_col_apply_kernel<F: Float>(
    x: &mut [F],
    partials: &[F], // [ceil(r/32), c]
    eps: f32,
    #[comptime] r: u32,
    #[comptime] c: u32,
    #[comptime] s0: u32,
    #[comptime] s1: u32,
    #[comptime] ps0: u32,
    #[comptime] ps1: u32,
) {
    let slab = CUBE_POS_X as usize;
    let cj = CUBE_POS_Y as usize;
    let tid = UNIT_POS_X as usize;
    let r0 = slab * 32;
    let rmax = (r as usize).min(r0 + 32);
    let c0 = cj * 256;
    let col = c0 + tid;
    let mut norms = Shared::<[F]>::new_slice(256usize);
    if col < c as usize {
        let mut cs = F::new(0.0_f32);
        let mut s = 0;
        let n_slabs = (r as usize).div_ceil(32);
        while s < n_slabs {
            cs += partials[s * (ps0 as usize) + col * (ps1 as usize)];
            s += 1;
        }
        let mut nrm = cs.sqrt();
        if nrm < F::cast_from(eps) {
            nrm = F::cast_from(eps);
        }
        norms[tid] = nrm;
    }
    sync_cube();
    if col < c as usize {
        let mut i = r0;
        while i < rmax {
            let pos = i * (s0 as usize) + col * (s1 as usize);
            x[pos] /= norms[tid];
            i += 1;
        }
    }
}

/// Row-wise L2 normalize: each row divided by sqrt(Σ_c x²). One cube per row,
/// 256 threads do the column reduction via shared-memory partials.
#[cube(launch_unchecked)]
fn norm_row_kernel_v2<F: Float>(
    x: &mut [F],
    eps: f32,
    #[comptime] c: u32,
    #[comptime] s0: u32,
    #[comptime] s1: u32,
) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let threads = 256usize;
    let mut partial = Shared::<[F]>::new_slice(threads);
    let mut sum = F::new(0.0_f32);
    let mut j = tid;
    while j < (c as usize) {
        let v = x[row * (s0 as usize) + j * (s1 as usize)];
        sum += v * v;
        j += threads;
    }
    partial[tid] = sum;
    sync_cube();
    if tid == 0 {
        let active = threads.min(c as usize);
        let mut total = F::new(0.0_f32);
        let mut t = 0;
        while t < active {
            total += partial[t];
            t += 1;
        }
        let mut nrm = total.sqrt();
        if nrm < F::cast_from(eps) {
            nrm = F::cast_from(eps);
        }
        partial[0] = nrm;
    }
    sync_cube();
    let nrm = partial[0];
    let mut j = tid;
    while j < (c as usize) {
        let pos = row * (s0 as usize) + j * (s1 as usize);
        x[pos] /= nrm;
        j += threads;
    }
}

/// ColRow post-polar normalization on the bare CUDA backend (2 launches vs
/// ~10 tensor passes). Returns false when not on CUDA.
pub fn norm_colrow_cuda<const D: usize>(x: &mut Tensor<D>, eps: f32) -> bool {
    let dims = x.dims();
    let (r, c) = (dims[D - 2], dims[D - 1]);
    if r == 0 || c == 0 {
        return false;
    }
    if let Some(xc) = cube_of(x) {
        let client = xc.client.clone();
        let dim = CubeDim::new_3d(256, 1, 1);
        // Buffers are row-pitched: logical (i, j) lives at i*s0 + j*s1, not
        // i*c + j. Index with the metadata strides so non-256-col shapes
        // (e.g. [500, 300]) read real data instead of padding.
        let st = xc.meta.strides();
        let (s0, s1) = (st[0] as u32, st[1] as u32);
        // Coalesced two-pass column normalize: pass 1 accumulates x² per
        // column over 32-row slabs (consecutive threads read consecutive
        // columns of a row, instead of the old stride-r walk per thread),
        // pass 2 divides each tile by the column norm.
        let n_slabs = (r as u32).div_ceil(32);
        let n_tiles = (c as u32).div_ceil(256);
        let partials = Tensor::<2>::zeros([n_slabs as usize, c], &x.device());
        let Some(pc) = cube_of(&partials) else {
            return false;
        };
        let pst = pc.meta.strides();
        let (ps0, ps1) = (pst[0] as u32, pst[1] as u32);
        let x_len = (xc.handle.size_in_used() / 4) as usize;
        let p_len = (pc.handle.size_in_used() / 4) as usize;
        let count = CubeCount::Static(n_slabs, n_tiles, 1);
        unsafe {
            norm_col_partials_kernel::launch_unchecked::<f32>(
                &client,
                count.clone(),
                dim,
                BufferArg::from_raw_parts(xc.handle.clone(), x_len),
                BufferArg::from_raw_parts(pc.handle.clone(), p_len),
                r as u32,
                c as u32,
                s0,
                s1,
                ps0,
                ps1,
            );
            norm_col_apply_kernel::launch_unchecked::<f32>(
                &client,
                count.clone(),
                dim,
                BufferArg::from_raw_parts(xc.handle.clone(), x_len),
                BufferArg::from_raw_parts(pc.handle, p_len),
                eps,
                r as u32,
                c as u32,
                s0,
                s1,
                ps0,
                ps1,
            );
            norm_row_kernel_v2::launch_unchecked::<f32>(
                &client,
                CubeCount::Static(r as u32, 1, 1),
                dim,
                BufferArg::from_raw_parts(xc.handle, x_len),
                eps,
                c as u32,
                s0,
                s1,
            );
        }
        true
    } else {
        false
    }
}
