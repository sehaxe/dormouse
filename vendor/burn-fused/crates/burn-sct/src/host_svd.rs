//! Host-side SVD (Golub-Kahan bidiagonalization + LAPACK-style dbdsqr).
//!
//! Vendored from the burn `linalg::svd` PR (same author) to keep the SCT
//! from_dense decomposition off the GPU: the one-sided Jacobi path was
//! launch-dispatch bound on cubecl 0.11-pre (~0.15 ms per round kernel,
//! 255 rounds x 15 sweeps). This runs in f64 on the host instead.

#![allow(clippy::assign_op_pattern)]

use num_traits::float::Float;

/// Shared raw access for the parallel bidiag. All pointer arithmetic lives
/// inside methods: a closure that builds raw-pointer temporaries directly
/// (e.g. `s.0.add(i)`) is considered non-Send by current rustc, while method
/// calls on a Send+Sync guard are fine. The barrier protocol makes the
/// sharing race-free by construction.
#[derive(Clone, Copy)]
struct SharedF32(*mut f32);
unsafe impl Send for SharedF32 {}
unsafe impl Sync for SharedF32 {}
impl SharedF32 {
    #[inline(always)]
    unsafe fn get(&self, i: usize) -> f32 {
        *self.0.add(i)
    }
    #[inline(always)]
    unsafe fn set(&self, i: usize, v: f32) {
        *self.0.add(i) = v;
    }
    #[inline(always)]
    unsafe fn sub_assign(&self, i: usize, v: f32) {
        *self.0.add(i) -= v;
    }
}

pub fn svd_host<F: Float + Copy>(
    a: &[F],
    m: usize,
    n: usize,
    batch: usize,
    max_sweeps: usize,
    swap: bool,
) -> (Vec<F>, Vec<F>, Vec<F>) {
    let mut u = vec![F::zero(); batch * m * n];
    let mut sigma = vec![F::zero(); batch * n];
    let mut vt = vec![F::zero(); batch * n * n];
    let mut d = vec![F::zero(); n];
    let mut e = vec![F::zero(); n.saturating_sub(1)];
    let mut givens: Vec<(usize, F, F, F, F)> = Vec::new();

    for b in 0..batch {
        let (mut u1t, bv, mut v1t) = bidiag_host(&a[b * m * n..(b + 1) * m * n], m, n);
        for i in 0..n {
            d[i] = bv[i * n + i];
        }
        for i in 0..n.saturating_sub(1) {
            e[i] = bv[i * n + i + 1];
        }
        givens.clear();
        let sigma_b = dbdsqr(&mut d, &mut e, &mut givens, max_sweeps);

        // U = U1 @ (product of left Givens rotations): u1 is stored
        // transposed ([n, m] row-major), so the column pair of the original
        // is a contiguous row pair here.
        for &(k, cl, sl, _, _) in &givens {
            for i in 0..m {
                let (a0, b0) = (u1t[k * m + i], u1t[(k + 1) * m + i]);
                u1t[k * m + i] = cl * a0 + sl * b0;
                u1t[(k + 1) * m + i] = -sl * a0 + cl * b0;
            }
        }
        // Vt = (V1 @ (product of right Givens rotations))^T: v1t rows are the
        // original v1 columns, contiguous.
        for &(k, _, _, cr, sr) in &givens {
            for i in 0..n {
                let (a0, b0) = (v1t[k * n + i], v1t[(k + 1) * n + i]);
                v1t[k * n + i] = cr * a0 + sr * b0;
                v1t[(k + 1) * n + i] = -sr * a0 + cr * b0;
            }
        }
        // Absorb the signs of the diagonal into U.
        for k in 0..n {
            if d[k] < F::zero() {
                for i in 0..m {
                    u1t[k * m + i] = -u1t[k * m + i];
                }
            }
        }
        for i in 0..m {
            for j in 0..n {
                u[b * m * n + i * n + j] = u1t[j * m + i];
            }
        }
        for i in 0..n {
            for j in 0..n {
                vt[b * n * n + i * n + j] = v1t[j * n + i];
            }
            sigma[b * n + i] = sigma_b[i];
        }
        // Sort the singular values descending and permute the factors, on the
        // host: deterministic (stable sort), independent of backend
        // gather/argsort kernels (which are view-based or nondeterministic on
        // fused CUDA). Mask numerical zeros relative to sigma_max, with a
        // dtype-derived threshold (10 * machine epsilon, same relative
        // tolerance as the dbdsqr deflation test) instead of a fixed 1e-6.
        let smax = sigma[b * n..(b + 1) * n]
            .iter()
            .fold(F::zero(), |s, &x| s.max(x.abs()));
        let zero_tol = smax * (F::epsilon() * F::from(10.0).unwrap());
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| {
            sigma[b * n + j]
                .partial_cmp(&sigma[b * n + i])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        let mut pu = vec![F::zero(); m * n];
        let mut pvt = vec![F::zero(); n * n];
        let mut sorted = vec![F::zero(); n];
        for (t, &src) in order.iter().enumerate() {
            for i in 0..m {
                pu[i * n + t] = u[b * m * n + i * n + src];
            }
            for i in 0..n {
                pvt[t * n + i] = vt[b * n * n + src * n + i];
            }
            // read from the untouched slot: sigma is rewritten in place below
            let v = sigma[b * n + src];
            sorted[t] = if v.abs() <= zero_tol { F::zero() } else { v };
        }
        for i in 0..m * n {
            u[b * m * n + i] = pu[i];
        }
        for i in 0..n * n {
            vt[b * n * n + i] = pvt[i];
        }
        for i in 0..n {
            sigma[b * n + i] = sorted[i];
        }
    }
    if swap {
        // The SVD was computed on A^T ([n, m]); the factors for the original
        // wide A = Vt^T S U^T, so return u = Vt^T and vt = U^T (already
        // permuted consistently with the sorted sigma).
        let mut uf = vec![F::zero(); batch * n * n];
        let mut vf = vec![F::zero(); batch * n * m];
        for b in 0..batch {
            for i in 0..n {
                for j in 0..n {
                    uf[(b * n + i) * n + j] = vt[(b * n + j) * n + i];
                }
            }
            for i in 0..n {
                for j in 0..m {
                    vf[(b * n + i) * m + j] = u[b * m * n + j * n + i];
                }
            }
        }
        (uf, sigma, vf)
    } else {
        (u, sigma, vt)
    }
}

/// Golub-Kahan bidiagonalization on the host: `A = U1 B V1^T` with `B` upper
/// bidiagonal (row-major `[m, n]`), using Householder reflections on
/// shrinking submatrices, mirroring the tensor-op version operation for
/// operation.
pub fn bidiag_host<F: Float + Copy>(a: &[F], m: usize, n: usize) -> (Vec<F>, Vec<F>, Vec<F>) {
    // f32 specialization: concrete loops autovectorize (the generic F: Float
    // versions hover at ~3.7 GFLOP/s here; f32 reaches ~2-3x that)
    if core::mem::size_of::<F>() == 4 && m >= n {
        let (u1t, bv, v1t) = bidiag_host_f32(f32_view(a), m, n);
        return (f32_vec_to(u1t), f32_vec_to(bv), f32_vec_to(v1t));
    }
    /// f32-specialized bidiagonalization: identical algorithm, concrete loops
    /// (the generic `F: Float` version stalls the autovectorizer; f32 reaches
    /// ~2-3x its throughput on the O(m n^2) reflector passes).
    fn bidiag_host_f32(a: &[f32], m: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut v1 = vec![0.0f32; n * n];
        for i in 0..n {
            v1[i * n + i] = 1.0;
        }
        let mut ws: Vec<f32> = vec![0.0; m * n];
        let mut taus = vec![0.0f32; n];
        let mut a = a.to_vec();
        let mut aw = vec![0.0f32; m];

        // Parallelize the O(m n^2) right-reflection passes (aw, row update, v1)
        // and the deferred u1t assembly across cores. The right reflector
        // (norm, w, tau) is O(n) per step and recomputed identically by every
        // worker, so no shared stash or merge is needed; the aw/update/v1 passes
        // split over an index whose rows are written by exactly one worker.
        let nt = std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1)
            .min(6);
        let barrier = std::sync::Barrier::new(nt);

        // SAFETY: all workers share `a`, `v1`, `aw`, `ws`, `taus` through raw
        // pointers; every phase is separated by a barrier, and within a phase
        // each worker only writes its own index range (or, for the left
        // reflection on thread 0, the whole matrix while the others wait on the
        // barrier). This is race-free by construction.
        let a_s = SharedF32(a.as_mut_ptr());
        let v1_s = SharedF32(v1.as_mut_ptr());
        let aw_s = SharedF32(aw.as_mut_ptr());
        let ws_s = SharedF32(ws.as_mut_ptr());
        let taus_s = SharedF32(taus.as_mut_ptr());
        std::thread::scope(|scope| {
            let barrier = &barrier;
            for t in 0..nt {
                let (klo, khi) = split_range(0, m, nt, t);
                let (ilo, ihi) = split_range(0, n, nt, t);
                scope.spawn(move || {
                    // all access goes through raw pointer arithmetic: creating
                    // &mut slices from raw inside a scoped-spawn closure makes it
                    // non-Send on current rustc
                    for i in 0..n {
                        // ---- left reflection: thread 0, workers wait ----
                        if t == 0 {
                            let mut scale = 0.0f32;
                            for k in i..m {
                                scale = scale.max(unsafe { a_s.get(k * n + i) }.abs());
                            }
                            let norm = if scale == 0.0 {
                                0.0
                            } else {
                                let mut acc = 0.0f32;
                                for k in i..m {
                                    let x = unsafe { a_s.get(k * n + i) } / scale;
                                    acc += x * x;
                                }
                                scale * acc.sqrt()
                            };
                            let x0 = unsafe { a_s.get(i * n + i) };
                            let sign = if x0 >= 0.0 { -1.0 } else { 1.0 };
                            let u0 = x0 - norm * sign;
                            let tau = if norm == 0.0 {
                                0.0
                            } else {
                                -u0 / (norm * sign)
                            };
                            if norm != 0.0 {
                                for k in (i + 1)..m {
                                    unsafe {
                                        ws_s.set(i * m + k, *a_s.0.add(k * n + i) / u0);
                                    }
                                }
                                unsafe {
                                    ws_s.set(i * m + i, 1.0);
                                }
                                unsafe {
                                    taus_s.set(i, tau);
                                }
                                let mut wta = vec![0.0f32; n];
                                for j in i..n {
                                    wta[j] = unsafe { a_s.get(i * n + j) };
                                }
                                for k in (i + 1)..m {
                                    let wk = unsafe { ws_s.get(i * m + k) };
                                    for j in i..n {
                                        wta[j] += wk * unsafe { a_s.get(k * n + j) };
                                    }
                                }
                                for j in i..n {
                                    unsafe {
                                        a_s.sub_assign(i * n + j, tau * wta[j]);
                                    }
                                }
                                for k in (i + 1)..m {
                                    let wk = tau * unsafe { ws_s.get(i * m + k) };
                                    for j in i..n {
                                        unsafe {
                                            a_s.sub_assign(k * n + j, wk * wta[j]);
                                        }
                                    }
                                }
                            }
                        }
                        barrier.wait();

                        // ---- right reflection: every worker recomputes the
                        // O(n) reflector, then works its chunks ----
                        if i + 1 < n - 1 {
                            let mut scale = 0.0f32;
                            for j in (i + 1)..n {
                                scale = scale.max(unsafe { a_s.get(i * n + j) }.abs());
                            }
                            let norm = if scale == 0.0 {
                                0.0
                            } else {
                                let mut acc = 0.0f32;
                                for j in (i + 1)..n {
                                    let x = unsafe { a_s.get(i * n + j) } / scale;
                                    acc += x * x;
                                }
                                scale * acc.sqrt()
                            };
                            let y0 = unsafe { a_s.get(i * n + i + 1) };
                            let sign = if y0 >= 0.0 { -1.0 } else { 1.0 };
                            let u0 = y0 - norm * sign;
                            let tau = if norm == 0.0 {
                                0.0
                            } else {
                                -u0 / (norm * sign)
                            };
                            if norm != 0.0 {
                                let mut w = vec![0.0f32; n];
                                w[i + 1] = 1.0;
                                for j in (i + 2)..n {
                                    w[j] = unsafe { a_s.get(i * n + j) } / u0;
                                }
                                for k in klo..khi {
                                    let mut acc = 0.0f32;
                                    for j in (i + 1)..n {
                                        acc += unsafe { a_s.get(k * n + j) } * w[j];
                                    }
                                    unsafe {
                                        aw_s.set(k, acc);
                                    }
                                }
                                for k in klo..khi {
                                    let coef = tau * unsafe { aw_s.get(k) };
                                    for j in (i + 1)..n {
                                        unsafe {
                                            a_s.sub_assign(k * n + j, coef * w[j]);
                                        }
                                    }
                                }
                                for i2 in ilo..ihi {
                                    let mut vw = 0.0f32;
                                    for j in (i + 1)..n {
                                        vw += unsafe { v1_s.get(i2 * n + j) } * w[j];
                                    }
                                    let coef = tau * vw;
                                    for j in (i + 1)..n {
                                        unsafe {
                                            v1_s.sub_assign(i2 * n + j, coef * w[j]);
                                        }
                                    }
                                }
                            }
                        }
                        barrier.wait();
                    }
                });
            }
        });

        // deferred left reflectors -> u1t, parallel over rows j: each worker
        // gets its own non-overlapping &mut row slice (safe Rust, no raw access)
        let mut u1t = vec![0.0f32; n * m];
        for i in 0..n {
            u1t[i * m + i] = 1.0;
        }
        std::thread::scope(|scope| {
            let ws = &ws;
            let taus = &taus;
            // split u1t into non-overlapping row chunks up front (safe Rust)
            let mut u1t = u1t.as_mut_slice();
            for t in 0..nt {
                let (jlo, jhi) = split_range(0, n, nt, t);
                let (rows, rest) = u1t.split_at_mut((jhi - jlo) * m);
                u1t = rest;
                let (ws, taus) = (ws, taus);
                let _ = jlo;
                scope.spawn(move || {
                    for i in (0..n).rev() {
                        let tau = taus[i];
                        if tau != 0.0 {
                            let wrow = i * m;
                            for row in rows.chunks_exact_mut(m) {
                                let mut uw = 0.0f32;
                                for k in i..m {
                                    uw += ws[wrow + k] * row[k];
                                }
                                let coef = tau * uw;
                                for k in i..m {
                                    row[k] -= coef * ws[wrow + k];
                                }
                            }
                        }
                    }
                });
            }
        });

        // transpose v1 -> v1t
        let mut v1t = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                v1t[j * n + i] = v1[i * n + j];
            }
        }
        (u1t, a, v1t)
    }

    /// [lo, hi) chunk `t` of `total` split into `nt` pieces.
    fn split_range(lo: usize, hi: usize, nt: usize, t: usize) -> (usize, usize) {
        let len = hi - lo;
        let base = len / nt;
        let rem = len % nt;
        let s = lo + t * base + t.min(rem);
        let e = s + base + usize::from(t < rem);
        (s, e)
    }

    /// Reinterpret `&[F]` as `&[f32]` (caller guarantees F == f32).
    fn f32_view<F: Float + Copy>(a: &[F]) -> &[f32] {
        unsafe { core::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) }
    }

    /// Reinterpret a `Vec<f32>` as `Vec<F>` (caller guarantees F == f32).
    fn f32_vec_to<F: Float + Copy>(v: Vec<f32>) -> Vec<F> {
        let mut v = core::mem::ManuallyDrop::new(v);
        unsafe { Vec::from_raw_parts(v.as_mut_ptr() as *mut F, v.len(), v.capacity()) }
    }

    // Everything transposed: U1 as [n, m] row-major (U1 columns = u1t rows)
    // and V1 as [n, n] row-major (V1 rows = v1t columns), so every reflector
    // and Givens application is a contiguous row update (SIMD-friendly).
    // The scalar row-major formulation paid a strided column access per
    // element on the O(m n^2) reflector passes; the transposed layout makes
    // the hot loops contiguous.
    // V1 kept in the ORIGINAL orientation during the bidiagonalization (the
    // right-reflector updates are then contiguous row passes); it is
    // transposed to v1t once at the end for the contiguous Givens assembly.
    let mut v1 = vec![F::zero(); n * n];
    for i in 0..n {
        v1[i * n + i] = F::one();
    }
    let mut ws: Vec<F> = vec![F::zero(); m * n];
    let mut taus = vec![F::zero(); n];
    let mut a = a.to_vec();
    let mut wta = vec![F::zero(); n];
    let mut aw = vec![F::zero(); m];

    for i in 0..n {
        // Left reflection: annihilate the subdiagonal of column i (scaled
        // norm like LAPACK dlarfg; the column reads stay strided but are
        // O(m) per step, dwarfed by the row-wise update passes below).
        let scale = (i..m).map(|k| a[k * n + i].abs()).fold(F::zero(), F::max);
        let norm = if scale == F::zero() {
            F::zero()
        } else {
            scale
                * (i..m)
                    .map(|k| {
                        let t = a[k * n + i] / scale;
                        t * t
                    })
                    .fold(F::zero(), |s, x| s + x)
                    .sqrt()
        };
        let x0 = a[i * n + i];
        let sign = if x0 >= F::zero() { -F::one() } else { F::one() };
        let u0 = x0 - norm * sign;
        let tau = if norm == F::zero() {
            F::zero()
        } else {
            -u0 / (norm * sign)
        };
        if norm != F::zero() {
            // w[i] = 1, w[k] = a[k][i] / u0
            for k in (i + 1)..m {
                ws[i * m + k] = a[k * n + i] / u0;
            }
            ws[i * m + i] = F::one();
            taus[i] = tau;
            // wta[j] = w^T a[:, j] (row-wise accumulation, contiguous j)
            for j in i..n {
                wta[j] = a[i * n + j];
            }
            for k in (i + 1)..m {
                let wk = ws[i * m + k];
                for j in i..n {
                    wta[j] = wta[j] + wk * a[k * n + j];
                }
            }
            // a_new = a - tau w wta (row-wise update, contiguous j)
            for j in i..n {
                a[i * n + j] = a[i * n + j] - tau * wta[j];
            }
            for k in (i + 1)..m {
                let wk = tau * ws[i * m + k];
                for j in i..n {
                    a[k * n + j] = a[k * n + j] - wk * wta[j];
                }
            }
        }

        // Right reflection: annihilate row i right of the superdiagonal.
        if i + 1 < n - 1 {
            let scale = ((i + 1)..n)
                .map(|j| a[i * n + j].abs())
                .fold(F::zero(), F::max);
            let norm = if scale == F::zero() {
                F::zero()
            } else {
                scale
                    * ((i + 1)..n)
                        .map(|j| {
                            let t = a[i * n + j] / scale;
                            t * t
                        })
                        .fold(F::zero(), |s, x| s + x)
                        .sqrt()
            };
            let y0 = a[i * n + i + 1];
            let sign = if y0 >= F::zero() { -F::one() } else { F::one() };
            let u0 = y0 - norm * sign;
            let tau = if norm == F::zero() {
                F::zero()
            } else {
                -u0 / (norm * sign)
            };
            if norm != F::zero() {
                let mut w = vec![F::zero(); n];
                w[i + 1] = F::one();
                for j in (i + 2)..n {
                    w[j] = a[i * n + j] / u0;
                }
                // aw[k] = a[k, :] w (contiguous j), then a = a - tau aw w^T
                for k in i..m {
                    let mut acc = F::zero();
                    for j in (i + 1)..n {
                        acc = acc + a[k * n + j] * w[j];
                    }
                    aw[k] = acc;
                }
                for k in i..m {
                    let coef = tau * aw[k];
                    for j in (i + 1)..n {
                        a[k * n + j] = a[k * n + j] - coef * w[j];
                    }
                }
                // V1 = V1 (I - tau w w^T): in the transposed layout the
                // original rows are the v1t columns, so this pass is strided;
                // it is O(n^2) per step and the row-major alternative (build
                // V1 untransposed) costs the same total, so keep one layout.
                for i2 in 0..n {
                    let mut vw = F::zero();
                    for j in (i + 1)..n {
                        vw = vw + v1[i2 * n + j] * w[j];
                    }
                    // unconditional: the zero-coef no-op is cheaper than a
                    // per-row branch, and keeps the update loop vectorizable
                    let coef = tau * vw;
                    for j in (i + 1)..n {
                        v1[i2 * n + j] = v1[i2 * n + j] - coef * w[j];
                    }
                }
            }
        }
    }
    // U1 (transposed) from the deferred left reflectors: the original column
    // updates u1[k][j] become contiguous u1t[j][k] row updates.
    let mut u1t = vec![F::zero(); n * m];
    for i in 0..n {
        u1t[i * m + i] = F::one();
    }
    for i in (0..n).rev() {
        let tau = taus[i];
        if tau != F::zero() {
            for j in 0..n {
                let base = j * m;
                let wrow = i * m;
                let mut uw = F::zero();
                let mut k = i;
                while k + 3 < m {
                    uw = uw
                        + ws[wrow + k] * u1t[base + k]
                        + ws[wrow + k + 1] * u1t[base + k + 1]
                        + ws[wrow + k + 2] * u1t[base + k + 2]
                        + ws[wrow + k + 3] * u1t[base + k + 3];
                    k += 4;
                }
                while k < m {
                    uw = uw + ws[wrow + k] * u1t[base + k];
                    k += 1;
                }
                let coef = tau * uw;
                let mut k = i;
                while k + 3 < m {
                    u1t[base + k] = u1t[base + k] - coef * ws[wrow + k];
                    u1t[base + k + 1] = u1t[base + k + 1] - coef * ws[wrow + k + 1];
                    u1t[base + k + 2] = u1t[base + k + 2] - coef * ws[wrow + k + 2];
                    u1t[base + k + 3] = u1t[base + k + 3] - coef * ws[wrow + k + 3];
                    k += 4;
                }
                while k < m {
                    u1t[base + k] = u1t[base + k] - coef * ws[wrow + k];
                    k += 1;
                }
            }
        }
    }
    // transpose V1 -> v1t (rows of v1 = columns of v1t)
    let mut v1t = vec![F::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            v1t[j * n + i] = v1[i * n + j];
        }
    }
    (u1t, a, v1t)
}

/// LAPACK dbdsqr-style shifted QR iteration on an upper bidiagonal matrix
/// (main diagonal `d`, superdiagonal `e`). Returns the singular values and
/// logs the Givens rotations (k, cosl, sinl, cosr, sinr) in application
/// order so the caller can rebuild the singular vectors.
pub fn dbdsqr<F: Float + Copy>(
    d: &mut [F],
    e: &mut [F],
    givens: &mut Vec<(usize, F, F, F, F)>,
    max_sweeps: usize,
) -> Vec<F> {
    let n = d.len();
    let eps = F::epsilon();
    let tol = eps * F::from(10.0).unwrap();
    let mut smax = F::zero();
    for &x in d.iter().chain(e.iter()) {
        smax = smax.max(x.abs());
    }
    if smax == F::zero() {
        return d.iter().map(|x| x.abs()).collect();
    }
    // Perturb exact zeros on the DIAGONAL only (never the superdiagonal): a
    // zero d[i] inside an active block makes the sweep stall (dlartg(0, 0)),
    // and perturbing e instead would swamp small-but-real singular values on
    // large-scale inputs (e.g. diag(1e38, 1, 1) in f32 would get e-floors of
    // 1e31 and diverge to NaN). e zeros deflate naturally via the check below.
    let floor = eps * smax;
    for (i, x) in d.iter_mut().enumerate() {
        if *x == F::zero() {
            *x = if i % 2 == 0 { floor } else { -floor };
        }
    }
    let mut m = n;
    let mut iters = 0;
    while m > 1 {
        // Find the lowest split: the block is [ll..m).
        let mut ll = 0;
        for k in (0..m - 1).rev() {
            if e[k].abs() <= tol * d[k].abs().max(d[k + 1].abs()) {
                e[k] = F::zero();
                ll = k + 1;
                break;
            }
        }
        if m - ll == 1 {
            m = ll;
            continue;
        }
        // Wilkinson-style shift from the bottom 2x2 block of B^T B. For f32
        // inputs the closed form carries ~1e-5 error, which exceeds the
        // deflation threshold (10 eps) and stalls 2x2 blocks on rounding
        // boundaries; computing the shift in f64 keeps it exact.
        let shift = if core::mem::size_of::<F>() == 4 {
            let d1 = d[m - 2].to_f64().unwrap();
            let e1 = e[m - 2].to_f64().unwrap();
            let d2 = d[m - 1].to_f64().unwrap();
            let t = d1 * d1 + d2 * d2 + e1 * e1;
            let disc = (t * t - 4.0 * d1 * d1 * d2 * d2).max(0.0).sqrt();
            F::from(((t + disc) / 2.0).max(0.0).sqrt()).unwrap()
        } else {
            dlas2_smax(d[m - 2], e[m - 2], d[m - 1])
        };
        // One QR sweep over the block. The starting value is the first column
        // of (B - shift I) scaled for stability; with d[ll] exactly zero the
        // formula diverges, so take its finite proxy (-shift, direction of
        // the limit for positive d).
        let mut f = if d[ll] == F::zero() {
            -shift
        } else {
            (d[ll].abs() - shift) * (d[ll].signum() + shift / d[ll])
        };
        let mut g = e[ll];
        for i in ll..m - 1 {
            let (cr, sr, r) = dlartg(f, g);
            if i > ll {
                e[i - 1] = r;
            }
            f = cr * d[i] + sr * e[i];
            e[i] = cr * e[i] - sr * d[i];
            g = sr * d[i + 1];
            d[i + 1] = cr * d[i + 1];
            let (cl, sl, r) = dlartg(f, g);
            d[i] = r;
            f = cl * e[i] + sl * d[i + 1];
            d[i + 1] = cl * d[i + 1] - sl * e[i];
            if i < m - 2 {
                g = sl * e[i + 1];
                e[i + 1] = cl * e[i + 1];
            }
            givens.push((i, cl, sl, cr, sr));
        }
        e[m - 2] = f;
        iters += 1;
        if iters > max_sweeps * n {
            break;
        }
    }
    d.iter().map(|x| x.abs()).collect()
}

/// Largest singular value of the 2x2 block [[d1, e1], [0, d2]]. Scaled like
/// LAPACK dlas2: no overflow or underflow on the intermediate squares.
fn dlas2_smax<F: Float + Copy>(d1: F, e1: F, d2: F) -> F {
    let t = d1 * d1 + d2 * d2 + e1 * e1;
    if t.is_finite() {
        let disc = (t * t - F::from(4.0).unwrap() * d1 * d1 * d2 * d2)
            .max(F::zero())
            .sqrt();
        ((t + disc) / F::from(2.0).unwrap()).max(F::zero()).sqrt()
    } else {
        let scale = d1.abs().max(d2.abs()).max(e1.abs());
        if scale == F::zero() {
            return F::zero();
        }
        let (a, b, c) = (d1 / scale, d2 / scale, e1 / scale);
        let t = a * a + b * b + c * c;
        let disc = (t * t - F::from(4.0).unwrap() * a * a * b * b).max(F::zero());
        scale * ((t + disc.sqrt()) / F::from(2.0).unwrap()).sqrt()
    }
}

/// Givens rotation annihilating g: (f, g) -> (r, 0). Scaled like LAPACK
/// dlartg: r = sqrt(f^2 + g^2) computed without overflow or underflow.
fn dlartg<F: Float + Copy>(f: F, g: F) -> (F, F, F) {
    let r2 = f * f + g * g;
    if r2.is_finite() && r2 > F::zero() {
        let r = r2.sqrt();
        (f / r, g / r, r)
    } else {
        let scale = f.abs().max(g.abs());
        if scale == F::zero() {
            (F::one(), F::zero(), F::zero())
        } else {
            let (sf, sg) = (f / scale, g / scale);
            let r = scale * (sf * sf + sg * sg).sqrt();
            (f / r, g / r, r)
        }
    }
}
