//! Fused chunked-forward kernels for the WY path (single launch per chunk batch).
//!
//! Replaces the ~40 tensor ops × per-chunk serial loop of `chunk_wy_forward`
//! (251 ms for d=512, T=2048 on a 5060 Ti) with two cubecl launches:
//!   - `gdn2_chunk_intra_kernel`: all chunks in parallel. Computes the
//!     chunk-local cumulative decay, the causal Q-K (`aqk`) and strict-lower
//!     key-key (`akk` = T) score matrices, the WY inverse A = (I + T)^{-1} by
//!     forward substitution, and the pseudo-key / pseudo-value blocks
//!     `w = A·(b⊙k⊙g_exp)`, `u = A·(w_gate⊙v)` plus `qg`, `kg_decay`, `g_last`.
//!   - `gdn2_chunk_inter_kernel`: sequential over chunks (the state recurrence
//!     is inherently sequential), one cube per (batch·head, value tile),
//!     holds the state slice in shared memory, computes `v_new`, intra/inter
//!     outputs, and the state update. This mirrors the reference Triton design.
//!
//! Only active on the bare CUDA `CubeBackend` (feature `cuda`); every other
//! backend falls back to the tensor ops in `super::forward::chunk_wy_forward`.
//!
//! Kernel limits (fall back to tensor ops beyond them): chunk_size ≤ 64,
//! key/value dims ≤ 256, sequence length a multiple of the chunk size.
//! The math matches `chunk_wy_forward` exactly, including the per-row
//! forward substitution, so results agree with the tensor path to ~1e-4.

use cubecl::prelude::*;

/// Intra-chunk kernel: one cube per (batch·head, chunk). See module docs.
#[allow(clippy::manual_div_ceil)] // cubecl can't expand `.div_ceil()` on runtime values
#[cube(launch_unchecked)]
fn gdn2_chunk_intra_kernel<F: Float>(
    q: &[F],             // [BH*NT, C, K]
    k: &[F],             // [BH*NT, C, K]
    g: &[F],             // [BH*NT, C, K]
    b: &[F],             // [BH*NT, C, K]
    v: &[F],             // [BH*NT, C, V]
    wg: &[F],            // [BH*NT, C, V]
    gexp: &mut [F],      // [BH*NT, C, K]  exp(chunk-local cumulative g)
    kgt: &mut [F],       // [BH*NT, K, C]  k / g_exp, transposed
    qgt: &mut [F],       // [BH*NT, K, C]  q ⊙ g_exp, transposed
    bkt: &mut [F],       // [BH*NT, K, C]  b ⊙ k ⊙ g_exp, transposed
    wvt: &mut [F],       // [BH*NT, V, C]  w_gate ⊙ v, transposed
    aqk: &mut [F],       // [BH*NT, C, C]  causal, scaled
    akk: &mut [F],       // [BH*NT, C, C]  strict lower (the T matrix)
    m_inv_out: &mut [F], // [BH*NT, C, C]  A = (I + T)^-1 (exported for backward)
    w: &mut [F],         // [BH*NT, C, K]  pseudo-key  = A @ (b⊙k⊙g_exp)
    u: &mut [F],         // [BH*NT, C, V]  pseudo-value = A @ (w_gate⊙v)
    kgd: &mut [F],       // [BH*NT, C, K]  k ⊙ g_exp[last] / g_exp
    glast: &mut [F],     // [BH*NT, K]     g_exp[last row]
    scale: f32,
    #[comptime] chunk_c: u32,
    #[comptime] k_dim: u32,
    #[comptime] v_dim: u32,
    #[comptime] stages: u32,
) {
    let block = CUBE_POS_X as usize;
    let r = UNIT_POS_X as usize;
    let grp = UNIT_POS_Y as usize;
    let c = chunk_c as usize;
    let kd = k_dim as usize;
    let vd = v_dim as usize;
    let nthr = CUBE_DIM as usize;
    let gy = nthr / c;
    let kt = (kd + gy - 1) / gy;
    let mut a_sh = Shared::<[F]>::new_slice(c * (c + 1));
    let ac = c + 1; // padded row stride: kills 32-way shared bank conflicts
    let tid = grp * c + r;
    if r < c && stages & 1 != 0 {
        // Phase 1: gexp[r][c] = exp(sum_{j<=r} g[j][c]). Each thread re-sums its
        // prefix serially (redundant work, but avoids a shared-memory scan and
        // the write-to-read hazard that comes with one). Flat index mapping so
        // a warp spans consecutive channels (coalesced global reads) instead of
        // stride-`kd` reads.
        let base = block * c * kd;
        let mut idx = tid;
        while idx < c * kd {
            let cc = idx % kd;
            let rr = idx / kd;
            // Kahan-compensated prefix sum: the plain serial accumulation
            // drifts ~1e-4 relative vs the cuBLAS-matmulled cumsum of the
            // tensor path once exp() amplifies it; compensation keeps the two
            // paths within fp32 noise of each other.
            let mut acc = F::new(0.0_f32);
            let mut comp = F::new(0.0_f32);
            let mut j = 0;
            while j <= rr {
                let y = g[base + j * kd + cc] - comp;
                let t = acc + y;
                comp = (t - acc) - y;
                acc = t;
                j += 1;
            }
            gexp[base + idx] = F::exp(acc);
            idx += nthr;
        }
        sync_storage();
    }

    if r < c && stages & 16 != 0 {
        // Phase 1.5: precompute qg/bk/kg/wv once per element into TRANSPOSED
        // buffers ([cc][r] layout) so the matmul phases read coalesced rows
        // instead of column-strided [r][cc] global loads. No divisions remain
        // in the matmul phases either. Flat index mapping: a warp spans
        // consecutive channels, so the q/k/g/b/gexp/v/wg reads are coalesced;
        // only the four transposed-buffer writes end up strided.
        let base = block * c * kd;
        let mut idx = tid;
        while idx < c * kd {
            let cc = idx % kd;
            let rr = idx / kd;
            let gexp_r = gexp[base + idx];
            let gexp_l = gexp[base + (c - 1) * kd + cc];
            let kr = k[base + idx];
            let to = (block * kd + cc) * c;
            kgt[to + rr] = kr / gexp_r;
            bkt[to + rr] = b[base + idx] * kr * gexp_r;
            qgt[to + rr] = q[base + idx] * gexp_r;
            // qg == qgt (same layout/data); kgd and g_last are derived here
            // so P4 only has to do the A@bk / A@wv matmuls.
            kgd[base + idx] = kr * gexp_l / gexp_r;
            if rr == 0 {
                glast[block * kd + cc] = gexp_l;
            }
            idx += nthr;
        }
        let vbase = block * c * vd;
        let mut idx = tid;
        while idx < c * vd {
            let vv = idx % vd;
            let rr = idx / vd;
            wvt[(block * vd + vv) * c + rr] = wg[vbase + idx] * v[vbase + idx];
            idx += nthr;
        }
        sync_storage();
    }

    if r < c && stages & 2 != 0 {
        // Phase 2: score matrices, row r per thread, 4 s-values per grp with
        // 8 independent accumulators (ILP). Reads are coalesced transposed
        // precomputes; no divisions.
        //   aqk[r][s] = scale·sum_c qg[r,c]·kg[s,c],      s ≤ r
        //   akk[r][s] = sum_c bk[r,c]·kg[s,c],           s < r
        // Each grp owns a contiguous slice of the s-range so the block is
        // correct for any cube height (16 or 8 grp lanes).
        let s_per = c / gy;
        let mut sbase = grp * s_per;
        while sbase < (grp + 1) * s_per {
            let s0 = sbase;
            let s1 = s0 + 1;
            let s2 = s0 + 2;
            let s3 = s0 + 3;
            let s1ok = s1 < c;
            let s2ok = s2 < c;
            let s3ok = s3 < c;
            // `% c` keeps reads in-bounds even when c is not a multiple of 4;
            // accumulators for invalid s are discarded by the write guards.
            let s1r = s1 % c;
            let s2r = s2 % c;
            let s3r = s3 % c;
            let mut aqk0 = F::new(0.0_f32);
            let mut aqk1 = F::new(0.0_f32);
            let mut aqk2 = F::new(0.0_f32);
            let mut aqk3 = F::new(0.0_f32);
            let mut akk0 = F::new(0.0_f32);
            let mut akk1 = F::new(0.0_f32);
            let mut akk2 = F::new(0.0_f32);
            let mut akk3 = F::new(0.0_f32);
            let mut cc = 0;
            while cc < kd {
                let qg_r = qgt[(block * kd + cc) * c + r];
                let bk_g = bkt[(block * kd + cc) * c + r];
                let k0 = kgt[(block * kd + cc) * c + s0];
                let k1 = kgt[(block * kd + cc) * c + s1r];
                let k2 = kgt[(block * kd + cc) * c + s2r];
                let k3 = kgt[(block * kd + cc) * c + s3r];
                aqk0 += qg_r * k0;
                akk0 += bk_g * k0;
                aqk1 += qg_r * k1;
                akk1 += bk_g * k1;
                aqk2 += qg_r * k2;
                akk2 += bk_g * k2;
                aqk3 += qg_r * k3;
                akk3 += bk_g * k3;
                cc += 1;
            }
            let sc = F::cast_from(scale);
            let o = block * c * c + r * c;
            if s0 <= r {
                aqk[block * c * c + s0 * c + r] = aqk0 * sc;
            } else {
                aqk[block * c * c + s0 * c + r] = F::new(0.0_f32);
            }
            akk[o + s0] = if s0 < r { akk0 } else { F::new(0.0_f32) };
            if s1ok {
                if s1 <= r {
                    aqk[block * c * c + s1 * c + r] = aqk1 * sc;
                } else {
                    aqk[block * c * c + s1 * c + r] = F::new(0.0_f32);
                }
                akk[o + s1] = if s1 < r { akk1 } else { F::new(0.0_f32) };
            }
            if s2ok {
                if s2 <= r {
                    aqk[block * c * c + s2 * c + r] = aqk2 * sc;
                } else {
                    aqk[block * c * c + s2 * c + r] = F::new(0.0_f32);
                }
                akk[o + s2] = if s2 < r { akk2 } else { F::new(0.0_f32) };
            }
            if s3ok {
                if s3 <= r {
                    aqk[block * c * c + s3 * c + r] = aqk3 * sc;
                } else {
                    aqk[block * c * c + s3 * c + r] = F::new(0.0_f32);
                }
                akk[o + s3] = if s3 < r { akk3 } else { F::new(0.0_f32) };
            }
            sbase += 4;
        }
        sync_storage();
    }

    if r < c && stages & 4 != 0 {
        // Phase 3: A = (I + T)^{-1} in shared memory, forward substitution.
        // Thread r solves column r of A row by row; the kk-sum over
        // T[row][*]·A[*][r] is split across the 16 `grp` lanes (partial sums in
        // shared, one reduction per row), so all 1024 threads stay busy
        // instead of just grp==0.
        let mut p_sh = Shared::<[F]>::new_slice(c * 17); // max 16 grp lanes, padded
        let p_stride = gy + 1;
        let mut row = 0;
        while row < c {
            if row < r {
                a_sh[row * ac + r] = F::new(0.0_f32);
            } else if row == r {
                a_sh[row * ac + r] = F::new(1.0_f32);
            } else {
                let mut acc = F::new(0.0_f32);
                let mut kk = r + 1 + grp;
                while kk < row {
                    acc += akk[block * c * c + row * c + kk] * a_sh[kk * ac + r];
                    kk += gy;
                }
                p_sh[r * p_stride + grp] = acc;
            }
            sync_cube();
            if row > r && grp == 0 {
                let mut tot = F::new(0.0_f32);
                let mut g = 0;
                while g < gy {
                    tot += p_sh[r * p_stride + g];
                    g += 1;
                }
                a_sh[row * ac + r] = F::new(0.0_f32) - akk[block * c * c + row * c + r] - tot;
            }
            sync_cube();
            row += 1;
        }
        // All threads must wait: P4 reads the full A matrix from shared.
        sync_cube();
        // Export A (= M^-1) so the training backward can reuse it and skip
        // re-inverting: the autodiff op saves this and the recompute feeds it
        // straight into W = A@rhs, d_rhs = A^T@d_*.
        let mut row = 0;
        while row < c {
            m_inv_out[block * c * c + row * c + r] = a_sh[row * ac + r];
            row += 1;
        }
        sync_cube();
    }

    if r < c && stages & 8 != 0 {
        // Phase 4: w = A@bk, u = A@wv. qg/qgt, kgd and g_last come from P1.5.
        let mut jj = 0;
        while jj < kt {
            let c_idx = grp * kt + jj;
            if c_idx < kd {
                let mut wa = F::new(0.0_f32);
                let mut wb = F::new(0.0_f32);
                let mut s = 0;
                while s + 1 < c {
                    wa += a_sh[r * ac + s] * bkt[(block * kd + c_idx) * c + s];
                    wb += a_sh[r * ac + s + 1] * bkt[(block * kd + c_idx) * c + s + 1];
                    s += 2;
                }
                while s < c {
                    wa += a_sh[r * ac + s] * bkt[(block * kd + c_idx) * c + s];
                    s += 1;
                }
                w[block * kd * c + c_idx * c + r] = wa + wb;
            }
            jj += 1;
        }
        let vt = (vd + gy - 1) / gy;
        let mut jj = 0;
        while jj < vt {
            let v_idx = grp * vt + jj;
            if v_idx < vd {
                let mut ua = F::new(0.0_f32);
                let mut ub = F::new(0.0_f32);
                let mut s = 0;
                while s + 1 < c {
                    ua += a_sh[r * ac + s] * wvt[(block * vd + v_idx) * c + s];
                    ub += a_sh[r * ac + s + 1] * wvt[(block * vd + v_idx) * c + s + 1];
                    s += 2;
                }
                while s < c {
                    ua += a_sh[r * ac + s] * wvt[(block * vd + v_idx) * c + s];
                    s += 1;
                }
                u[block * vd * c + v_idx * c + r] = ua + ub;
            }
            jj += 1;
        }
    }
}

/// Inter-chunk recurrence + output kernel. One cube per (batch·head, value
/// tile of 16 channels); sequential over chunks. State slice `[K, 16]` and
/// `v_new` `[C, 16]` live in shared memory. Mutates `state` in place.
#[cube(launch_unchecked)]
fn gdn2_chunk_inter_kernel<F: Float>(
    aqk: &[F],       // [BH*NT, C, C]
    w: &[F],         // [BH*NT, C, K]
    u: &[F],         // [BH*NT, C, V]
    qg: &[F],        // [BH*NT, C, K]
    kgd: &[F],       // [BH*NT, C, K]
    glast: &[F],     // [BH*NT, K]
    state: &mut [F], // [BH, K, V], updated in place
    out: &mut [F],   // [BH*NT, C, V]
    scale: f32,
    nt: u32,
    #[comptime] chunk_c: u32,
    #[comptime] k_dim: u32,
    #[comptime] v_dim: u32,
    #[comptime] vtile: u32,
    #[comptime] stages: u32, // bit0: v_new, bit1: out, bit2: state update
) {
    let bh = CUBE_POS_X as usize;
    let vt = CUBE_POS_Y as usize;
    let r = UNIT_POS_X as usize;
    let grp = UNIT_POS_Y as usize;
    let c = chunk_c as usize;
    let kd = k_dim as usize;
    let vd = v_dim as usize;
    let vtile = vtile as usize;
    let n_vp = 2;
    let vs = vt * vtile;
    // Prefetch w/kgd through shared, always. The no-prefetch path (reading
    // w/kgd straight from global inside the hot loop) races on Blackwell:
    // `BAR.SYNC.DEFER_BLOCKING` defers the block barrier's flush, and the
    // interleaved global loads let the state reads (s_sh/vn_sh) slip ahead of
    // the previous iteration's writes — a WAR on the shared state that flips the
    // result ~25-40% of runs (see fused_chunk_verify k=128). Prefetching keeps
    // the hot loop pure-shared so the barrier is scheduled correctly. Shared
    // budget: w_sh+kgd_sh = 2·kd·c floats ≤ 32 KiB for kd ≤ 256, fine per block.
    let pf = comptime![true];
    if r < c {
        let mut s_sh = Shared::<[F]>::new_slice(kd * vtile);
        let mut vn_sh = Shared::<[F]>::new_slice(c * vtile);
        let mut w_sh = Shared::<[F]>::new_slice(if pf { kd * c } else { 0 });
        let mut kgd_sh = Shared::<[F]>::new_slice(if pf { kd * c } else { 0 });
        let nthr = CUBE_DIM as usize;
        let tid = grp * c + r;

        // Load the state slice into shared ONCE. Each cube owns a distinct
        // `vtile` column slice of `state`, so the whole recurrence can run in
        // shared memory; we write the slice back to global only after the last
        // chunk.
        let mut jj = 0;
        while jj < n_vp {
            let vv = vs + grp * n_vp + jj;
            if vv < vd {
                let mut kk = r;
                while kk < kd {
                    s_sh[kk * vtile + grp * n_vp + jj] = state[bh * kd * vd + kk * vd + vv];
                    kk += c;
                }
            }
            jj += 1;
        }
        sync_cube();

        // Prefetch chunk 0's w and kgd into shared (skipped when pf = false).
        if pf {
            let base0 = (bh * (nt as usize)) * c * kd;
            let mut idx = tid;
            while idx < kd * c {
                w_sh[idx] = w[base0 + idx];
                kgd_sh[idx] = kgd[base0 + idx];
                idx += nthr;
            }
            sync_cube();
        }

        let mut t = 0;
        while t < nt as usize {
            let cb = bh * (nt as usize) + t;
            let cbase = cb * c;

            let vv0 = vs + grp * n_vp;
            let vv1 = vs + grp * n_vp + 1;
            let v0 = grp * n_vp;
            let v1 = grp * n_vp + 1;

            // v_new[r][v] = u[r][v] - sum_kk w[r][kk]·S[kk][v] (w read from shared)
            let mut vn_a = F::new(0.0_f32);
            let mut vn_b = F::new(0.0_f32);
            let mut vn_c = F::new(0.0_f32);
            let mut vn_d = F::new(0.0_f32);
            let mut vn_a1 = F::new(0.0_f32);
            let mut vn_b1 = F::new(0.0_f32);
            let mut vn_c1 = F::new(0.0_f32);
            let mut vn_d1 = F::new(0.0_f32);
            let mut kk = 0;
            while kk + 3 < kd {
                let w0 = if pf {
                    w_sh[kk * c + r]
                } else {
                    w[cbase * kd + kk * c + r]
                };
                let w1 = if pf {
                    w_sh[(kk + 1) * c + r]
                } else {
                    w[cbase * kd + (kk + 1) * c + r]
                };
                let w2 = if pf {
                    w_sh[(kk + 2) * c + r]
                } else {
                    w[cbase * kd + (kk + 2) * c + r]
                };
                let w3 = if pf {
                    w_sh[(kk + 3) * c + r]
                } else {
                    w[cbase * kd + (kk + 3) * c + r]
                };
                vn_a += w0 * s_sh[kk * vtile + v0];
                vn_a1 += w0 * s_sh[kk * vtile + v1];
                vn_b += w1 * s_sh[(kk + 1) * vtile + v0];
                vn_b1 += w1 * s_sh[(kk + 1) * vtile + v1];
                vn_c += w2 * s_sh[(kk + 2) * vtile + v0];
                vn_c1 += w2 * s_sh[(kk + 2) * vtile + v1];
                vn_d += w3 * s_sh[(kk + 3) * vtile + v0];
                vn_d1 += w3 * s_sh[(kk + 3) * vtile + v1];
                kk += 4;
            }
            while kk < kd {
                let w0 = if pf {
                    w_sh[kk * c + r]
                } else {
                    w[cbase * kd + kk * c + r]
                };
                vn_a += w0 * s_sh[kk * vtile + v0];
                vn_a1 += w0 * s_sh[kk * vtile + v1];
                kk += 1;
            }
            let vn0 = u[cbase * vd + vv0 * c + r] - (vn_a + vn_b + vn_c + vn_d);
            let vn1 = if vv1 < vd {
                u[cbase * vd + vv1 * c + r] - (vn_a1 + vn_b1 + vn_c1 + vn_d1)
            } else {
                F::new(0.0_f32)
            };
            if stages & 1 != 0 && vv0 < vd {
                vn_sh[r * vtile + v0] = vn0;
                vn_sh[r * vtile + v1] = vn1;
            }
            sync_cube();

            // In-loop output phase: out[r][v] = aqk[:,r]·v_new[:,v]
            //                         + scale·qg[:,r]·S[:,v].
            // Runs after v_new is in shared but BEFORE the state update, so it
            // reads the pre-update state. Reads v_new / state from shared.
            if stages & 2 != 0 && vv0 < vd {
                let sc = F::cast_from(scale);
                let mut ia = F::new(0.0_f32);
                let mut ib = F::new(0.0_f32);
                let mut ic = F::new(0.0_f32);
                let mut id = F::new(0.0_f32);
                let mut ia1 = F::new(0.0_f32);
                let mut ib1 = F::new(0.0_f32);
                let mut ic1 = F::new(0.0_f32);
                let mut id1 = F::new(0.0_f32);
                let mut ja = F::new(0.0_f32);
                let mut jb = F::new(0.0_f32);
                let mut jc = F::new(0.0_f32);
                let mut jd = F::new(0.0_f32);
                let mut ja1 = F::new(0.0_f32);
                let mut jb1 = F::new(0.0_f32);
                let mut jc1 = F::new(0.0_f32);
                let mut jd1 = F::new(0.0_f32);
                let mut s = 0;
                while s + 3 < c {
                    let a0 = aqk[cbase * c + s * c + r];
                    let a1 = aqk[cbase * c + (s + 1) * c + r];
                    let a2 = aqk[cbase * c + (s + 2) * c + r];
                    let a3 = aqk[cbase * c + (s + 3) * c + r];
                    ia += a0 * vn_sh[s * vtile + v0];
                    ia1 += a0 * vn_sh[s * vtile + v1];
                    ib += a1 * vn_sh[(s + 1) * vtile + v0];
                    ib1 += a1 * vn_sh[(s + 1) * vtile + v1];
                    ic += a2 * vn_sh[(s + 2) * vtile + v0];
                    ic1 += a2 * vn_sh[(s + 2) * vtile + v1];
                    id += a3 * vn_sh[(s + 3) * vtile + v0];
                    id1 += a3 * vn_sh[(s + 3) * vtile + v1];
                    s += 4;
                }
                while s < c {
                    let a0 = aqk[cbase * c + s * c + r];
                    ia += a0 * vn_sh[s * vtile + v0];
                    ia1 += a0 * vn_sh[s * vtile + v1];
                    s += 1;
                }
                let mut kk = 0;
                while kk + 3 < kd {
                    let q0 = qg[cbase * kd + kk * c + r];
                    let q1 = qg[cbase * kd + (kk + 1) * c + r];
                    let q2 = qg[cbase * kd + (kk + 2) * c + r];
                    let q3 = qg[cbase * kd + (kk + 3) * c + r];
                    ja += q0 * s_sh[kk * vtile + grp * n_vp];
                    ja1 += q0 * s_sh[kk * vtile + grp * n_vp + 1];
                    jb += q1 * s_sh[(kk + 1) * vtile + grp * n_vp];
                    jb1 += q1 * s_sh[(kk + 1) * vtile + grp * n_vp + 1];
                    jc += q2 * s_sh[(kk + 2) * vtile + grp * n_vp];
                    jc1 += q2 * s_sh[(kk + 2) * vtile + grp * n_vp + 1];
                    jd += q3 * s_sh[(kk + 3) * vtile + grp * n_vp];
                    jd1 += q3 * s_sh[(kk + 3) * vtile + grp * n_vp + 1];
                    kk += 4;
                }
                while kk < kd {
                    let q0 = qg[cbase * kd + kk * c + r];
                    ja += q0 * s_sh[kk * vtile + grp * n_vp];
                    ja1 += q0 * s_sh[kk * vtile + grp * n_vp + 1];
                    kk += 1;
                }
                out[cbase * vd + r * vd + vv0] = (ia + ib + ic + id) + (ja + jb + jc + jd) * sc;
                if vv1 < vd {
                    out[cbase * vd + r * vd + vv1] =
                        (ia1 + ib1 + ic1 + id1) + (ja1 + jb1 + jc1 + jd1) * sc;
                }
            }
            sync_cube();

            // State update in shared:
            // S[kk][v] = S[kk][v]·g_last[kk] + sum_r kgd[r][kk]·v_new[r][v].
            let mut jj = 0;
            while jj < n_vp {
                let vv = vs + grp * n_vp + jj;
                if vv < vd {
                    let mut kk = r;
                    while kk < kd {
                        let mut acc_a = F::new(0.0_f32);
                        let mut acc_b = F::new(0.0_f32);
                        let mut acc_c = F::new(0.0_f32);
                        let mut acc_d = F::new(0.0_f32);
                        let mut r2 = 0;
                        while r2 + 3 < c {
                            let g0 = if pf {
                                kgd_sh[r2 * kd + kk]
                            } else {
                                kgd[cbase * kd + r2 * kd + kk]
                            };
                            let g1 = if pf {
                                kgd_sh[(r2 + 1) * kd + kk]
                            } else {
                                kgd[cbase * kd + (r2 + 1) * kd + kk]
                            };
                            let g2 = if pf {
                                kgd_sh[(r2 + 2) * kd + kk]
                            } else {
                                kgd[cbase * kd + (r2 + 2) * kd + kk]
                            };
                            let g3 = if pf {
                                kgd_sh[(r2 + 3) * kd + kk]
                            } else {
                                kgd[cbase * kd + (r2 + 3) * kd + kk]
                            };
                            acc_a += g0 * vn_sh[r2 * vtile + grp * n_vp + jj];
                            acc_b += g1 * vn_sh[(r2 + 1) * vtile + grp * n_vp + jj];
                            acc_c += g2 * vn_sh[(r2 + 2) * vtile + grp * n_vp + jj];
                            acc_d += g3 * vn_sh[(r2 + 3) * vtile + grp * n_vp + jj];
                            r2 += 4;
                        }
                        while r2 < c {
                            let g0 = if pf {
                                kgd_sh[r2 * kd + kk]
                            } else {
                                kgd[cbase * kd + r2 * kd + kk]
                            };
                            acc_a += g0 * vn_sh[r2 * vtile + grp * n_vp + jj];
                            r2 += 1;
                        }
                        if stages & 4 != 0 {
                            s_sh[kk * vtile + grp * n_vp + jj] = s_sh[kk * vtile + grp * n_vp + jj]
                                * glast[cb * kd + kk]
                                + (acc_a + acc_b + acc_c + acc_d);
                        }
                        kk += c;
                    }
                }
                jj += 1;
            }
            sync_cube();

            // Prefetch the next chunk's w and kgd. Safe to overwrite the staging
            // buffers: the v_new loop consumed w_sh and the state update consumed
            // kgd_sh before this point.
            if pf {
                if t + 1 < nt as usize {
                    let base1 = (bh * (nt as usize) + t + 1) * c * kd;
                    let mut idx = tid;
                    while idx < kd * c {
                        w_sh[idx] = w[base1 + idx];
                        kgd_sh[idx] = kgd[base1 + idx];
                        idx += nthr;
                    }
                }
                sync_cube();
            }

            t += 1;
        }
        // Write the state slice back to global once, after all chunks.
        let mut jj = 0;
        while jj < n_vp {
            let vv = vs + grp * n_vp + jj;
            if vv < vd {
                let mut kk = r;
                while kk < kd {
                    state[bh * kd * vd + kk * vd + vv] = s_sh[kk * vtile + grp * n_vp + jj];
                    kk += c;
                }
            }
            jj += 1;
        }
    }
}

#[cfg(feature = "cuda")]

/// Re-runs the chunked trajectory from the exported `u`/`w`/`kgd`/`glast` and
/// the initial state, dumping `v_new` and the pre-chunk states for the fused
/// backward. A separate launch: the inter kernel's hot loop stays untouched
/// (the in-kernel dumps were flaky on some configs).
#[cube(launch_unchecked)]
fn gdn2_chunk_trajectory_export_kernel<F: Float>(
    u_buf: &[F],          // [nblk, V, C] transposed U[v][r]
    w_buf: &[F],          // [nblk, K, C] transposed W[k][r]
    kgd: &[F],            // [nblk, C, K]
    glast: &[F],          // [nblk, K]
    state_in: &[F],       // [bh, K, V]
    v_new_out: &mut [F],  // [nblk, C, V]
    states_out: &mut [F], // [nblk, K, V]
    nt: u32,
    #[comptime] chunk_c: u32,
    #[comptime] k_dim: u32,
    #[comptime] v_dim: u32,
    #[comptime] vtile: u32,
) {
    let bh = CUBE_POS_X as usize;
    let vt = CUBE_POS_Y as usize;
    let r = UNIT_POS_X as usize;
    let grp = UNIT_POS_Y as usize;
    let c = chunk_c as usize;
    let kd = k_dim as usize;
    let vd = v_dim as usize;
    let vtile = vtile as usize;
    let n_vp = 2;
    let vs = vt * vtile;
    let slot = grp * n_vp;

    if r < c {
        let mut s_sh = Shared::<[F]>::new_slice(kd * vtile);
        let mut vn_sh = Shared::<[F]>::new_slice(c * vtile);
        let mut jj = 0;
        while jj < n_vp {
            let vv = vs + slot + jj;
            if vv < vd {
                let mut kk = r;
                while kk < kd {
                    s_sh[kk * vtile + slot + jj] = state_in[bh * kd * vd + kk * vd + vv];
                    kk += c;
                }
            }
            jj += 1;
        }
        sync_cube();

        let mut t = 0;
        while t < nt as usize {
            let cb = bh * (nt as usize) + t;
            let uo = cb * vd * c;
            let wo = cb * kd * c;
            let ko = cb * c * kd;
            // v_new[r][v] = u[r][v] - Σ_k w[r][k]·S[k][v]
            let mut jj = 0;
            while jj < n_vp {
                let vv = vs + slot + jj;
                if vv < vd {
                    let mut acc = F::new(0.0_f32);
                    let mut kk = 0;
                    while kk + 3 < kd {
                        acc += w_buf[wo + kk * c + r] * s_sh[kk * vtile + slot + jj];
                        acc += w_buf[wo + (kk + 1) * c + r] * s_sh[(kk + 1) * vtile + slot + jj];
                        acc += w_buf[wo + (kk + 2) * c + r] * s_sh[(kk + 2) * vtile + slot + jj];
                        acc += w_buf[wo + (kk + 3) * c + r] * s_sh[(kk + 3) * vtile + slot + jj];
                        kk += 4;
                    }
                    while kk < kd {
                        acc += w_buf[wo + kk * c + r] * s_sh[kk * vtile + slot + jj];
                        kk += 1;
                    }
                    let vn = u_buf[uo + vv * c + r] - acc;
                    vn_sh[r * vtile + slot + jj] = vn;
                    v_new_out[cb * c * vd + r * vd + vv] = vn;
                }
                jj += 1;
            }
            sync_cube();
            // dump the pre-chunk state
            let mut jj = 0;
            while jj < n_vp {
                let vv = vs + slot + jj;
                if vv < vd {
                    let mut kk = r;
                    while kk < kd {
                        states_out[cb * kd * vd + kk * vd + vv] = s_sh[kk * vtile + slot + jj];
                        kk += c;
                    }
                }
                jj += 1;
            }
            // S[kk][v] = S[kk][v]·glast[kk] + Σ_r kgd[r][kk]·v_new[r][v]
            let mut jj = 0;
            while jj < n_vp {
                let vv = vs + slot + jj;
                if vv < vd {
                    let mut kk = r;
                    while kk < kd {
                        let mut acc = F::new(0.0_f32);
                        let mut r2 = 0;
                        while r2 + 3 < c {
                            acc += kgd[ko + r2 * kd + kk] * vn_sh[r2 * vtile + slot + jj];
                            acc +=
                                kgd[ko + (r2 + 1) * kd + kk] * vn_sh[(r2 + 1) * vtile + slot + jj];
                            acc +=
                                kgd[ko + (r2 + 2) * kd + kk] * vn_sh[(r2 + 2) * vtile + slot + jj];
                            acc +=
                                kgd[ko + (r2 + 3) * kd + kk] * vn_sh[(r2 + 3) * vtile + slot + jj];
                            r2 += 4;
                        }
                        while r2 < c {
                            acc += kgd[ko + r2 * kd + kk] * vn_sh[r2 * vtile + slot + jj];
                            r2 += 1;
                        }
                        s_sh[kk * vtile + slot + jj] =
                            s_sh[kk * vtile + slot + jj] * glast[cb * kd + kk] + acc;
                        kk += c;
                    }
                }
                jj += 1;
            }
            sync_cube();
            t += 1;
        }
    }
}

pub mod cuda {
    use super::*;
    use burn::backend::{Backend, DispatchKindConversion};
    use burn::tensor::{DispatchTensor, Tensor};
    use burn_cubecl::tensor::CubeTensor;
    use burn_cubecl::CubeBackend;
    use std::any::TypeId;

    /// The bare (non-fusion) CUDA backend the fused kernels target.
    pub type CudaBare = CubeBackend;

    pub(crate) fn is_cuda<B: Backend>() -> bool {
        TypeId::of::<B>() == TypeId::of::<CudaBare>()
    }

    /// Owned copy of the underlying `CubeTensor` of `t`, if `B` is the bare
    /// CUDA `CubeBackend` and the buffer is row-major contiguous. A strided
    /// view (the `project()` [B,T,H,K] -> [B,H,T,K] permute) is materialized
    /// once, never recursed into — see
    /// [`crate::kernel::contiguous_cube_of`].
    fn cube_of<B: Backend, const D: usize>(
        t: &Tensor<D>,
    ) -> Option<CubeTensor>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        crate::kernel::contiguous_cube_of::<B, D>(t)
    }

    /// Run the fused chunked forward on CUDA. Returns `None` when the backend
    /// is not the bare CUDA `CubeBackend`, the sequence length is not a
    /// multiple of the chunk size, or the dimensions exceed the kernel limits;
    /// the caller then falls back to the tensor-ops path.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_chunk_forward_scratch<B: Backend>(
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        g: Tensor<4>,
        b: Tensor<4>,
        w: Tensor<4>,
        state: Tensor<4>,
        scale: f64,
        chunk_size: usize,
    ) -> Option<(Tensor<4>, Tensor<4>, IntraOut)>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let [batch, heads, time, k_dim] = q.shape().dims::<4>();
        let v_dim = v.shape().dims::<4>()[3];
        let c = chunk_size;
        if time % c != 0 || time == 0 {
            return None;
        }
        // The fused kernels are compiled for f32 only (launch_unchecked
        // hardcodes f32); bf16 inputs would make them read twice the bytes
        // and fault with ILLEGAL_ADDRESS. Fall back to the tensor path until
        // the kernels gain an F-generic launch.
        use burn::tensor::DType;
        if !matches!(q.dtype(), DType::F32) {
            return None;
        }
        // Numerical limit, not a hardware one: the naive K/exp(cumsum(g))
        // factor underflows f32 once cumsum(g) < -88, i.e. at chunk > 17 with
        // the K3 floor g = -5. FlashKDA picked chunk 16 for exactly this
        // reason; larger chunks go through the K3 16-tile tensor path.
        if c > 16 || k_dim > 256 || v_dim > 256 {
            return None; // kernel limits, fall back
        }
        let nt = time / c;
        let bh = batch * heads;
        let device = state.device();

        let q = cube_of::<B, 4>(&q)?;
        let k = cube_of::<B, 4>(&k)?;
        let v = cube_of::<B, 4>(&v)?;
        let g = cube_of::<B, 4>(&g)?;
        let b = cube_of::<B, 4>(&b)?;
        let w = cube_of::<B, 4>(&w)?;
        // The inter kernel mutates the state buffer in place; give it a
        // private copy so the caller's tensor (and any autodiff checkpoint
        // holding the same handle) is never corrupted.
        let state = state.clone() * 1.0;
        let state_initial = state.clone() * 1.0;
        let state_cube = cube_of::<B, 4>(&state).expect("backend mismatch");

        let nblk = bh * nt;
        let client = state_cube.client.clone();

        let mk = |shape: [usize; 3]| -> Tensor<3> { Tensor::<3>::empty(shape, &device) };
        let gexp = mk([nblk, c, k_dim]);
        let kgt = mk([nblk, k_dim, c]);
        let qgt = mk([nblk, k_dim, c]);
        let bkt = mk([nblk, k_dim, c]);
        let aqk = mk([nblk, c, c]);
        let akk = mk([nblk, c, c]);
        let m_inv = mk([nblk, c, c]);
        let w_blk = mk([nblk, c, k_dim]);
        let u_blk = mk([nblk, c, v_dim]);
        let kgd = mk([nblk, c, k_dim]);
        let wvt = mk([nblk, v_dim, c]);
        let glast = Tensor::<2>::empty([nblk, k_dim], &device);
        let out = Tensor::<4>::empty([batch, heads, time, v_dim], &device);
        let m_inv_c = cube_of::<B, 3>(&m_inv).expect("backend mismatch");
        let gexp_c = cube_of::<B, 3>(&gexp).expect("backend mismatch");
        let kgt_c = cube_of::<B, 3>(&kgt).expect("backend mismatch");
        let qgt_c = cube_of::<B, 3>(&qgt).expect("backend mismatch");
        let bkt_c = cube_of::<B, 3>(&bkt).expect("backend mismatch");
        let aqk_c = cube_of::<B, 3>(&aqk).expect("backend mismatch");
        let akk_c = cube_of::<B, 3>(&akk).expect("backend mismatch");
        let w_c = cube_of::<B, 3>(&w_blk).expect("backend mismatch");
        let u_c = cube_of::<B, 3>(&u_blk).expect("backend mismatch");
        let kgd_c = cube_of::<B, 3>(&kgd).expect("backend mismatch");
        let wvt_c = cube_of::<B, 3>(&wvt).expect("backend mismatch");
        let glast_c = cube_of::<B, 2>(&glast).expect("backend mismatch");
        let out_c = cube_of::<B, 4>(&out).expect("backend mismatch");
        let nk = nblk * c * k_dim;
        let nv = nblk * c * v_dim;
        let ncc = nblk * c * c;
        // Prefetch path needs fat cubes to amortize the w/kgd staging buffers,
        // but the no-prefetch (tiny-cube) variant raced (see the `pf` comment in
        // the inter kernel), so every k_dim prefetches; kd>64 keeps the tiny
        // cubes (more blocks → better latency hiding), which wins even with the
        // longer per-block prefetch loop.
        let vtile = if k_dim <= 64 { 8usize } else { 2usize };
        let iy = if k_dim <= 64 { 4u32 } else { 1u32 };

        let cube_dim = CubeDim::new_3d(c as u32, 8, 1);
        let cube_count = CubeCount::Static(nblk as u32, 1, 1);
        // The counter means "a kernel launch happened", so it is incremented
        // HERE, past every gate above (backend, divisibility, dtype, kernel
        // limits, contiguity). Before the gates it counted the CALLER's
        // interest, and `fused_calls()` then reported a fused run that fell
        // back to the tensor path — a green test on a lie (ADR-0019).
        crate::cuda_dispatch::note_fused_forward();
        unsafe {
            gdn2_chunk_intra_kernel::launch_unchecked::<f32>(
                &client,
                cube_count,
                cube_dim,
                BufferArg::from_raw_parts(q.handle, nk),
                BufferArg::from_raw_parts(k.handle, nk),
                BufferArg::from_raw_parts(g.handle, nk),
                BufferArg::from_raw_parts(b.handle, nk),
                BufferArg::from_raw_parts(v.handle, nv),
                BufferArg::from_raw_parts(w.handle, nv),
                BufferArg::from_raw_parts(gexp_c.handle, nk),
                BufferArg::from_raw_parts(kgt_c.handle, nk),
                BufferArg::from_raw_parts(qgt_c.handle.clone(), nk),
                BufferArg::from_raw_parts(bkt_c.handle, nk),
                BufferArg::from_raw_parts(wvt_c.handle, nv),
                BufferArg::from_raw_parts(aqk_c.handle.clone(), ncc),
                BufferArg::from_raw_parts(akk_c.handle, ncc),
                BufferArg::from_raw_parts(m_inv_c.handle.clone(), ncc),
                BufferArg::from_raw_parts(w_c.handle.clone(), nk),
                BufferArg::from_raw_parts(u_c.handle.clone(), nv),
                BufferArg::from_raw_parts(kgd_c.handle.clone(), nk),
                BufferArg::from_raw_parts(glast_c.handle.clone(), nblk * k_dim),
                scale as f32,
                c as u32,
                k_dim as u32,
                v_dim as u32,
                31,
            );
        }

        let vt = v_dim.div_ceil(vtile);
        let cube_dim2 = CubeDim::new_3d(c as u32, iy, 1);
        let cube_count2 = CubeCount::Static(bh as u32, vt as u32, 1);
        unsafe {
            gdn2_chunk_inter_kernel::launch_unchecked::<f32>(
                &client,
                cube_count2,
                cube_dim2,
                BufferArg::from_raw_parts(aqk_c.handle.clone(), ncc),
                BufferArg::from_raw_parts(w_c.handle.clone(), nk),
                BufferArg::from_raw_parts(u_c.handle.clone(), nv),
                BufferArg::from_raw_parts(qgt_c.handle.clone(), nk),
                BufferArg::from_raw_parts(kgd_c.handle.clone(), nk),
                BufferArg::from_raw_parts(glast_c.handle.clone(), nblk * k_dim),
                BufferArg::from_raw_parts(state_cube.handle, bh * k_dim * v_dim),
                BufferArg::from_raw_parts(out_c.handle, bh * time * v_dim),
                scale as f32,
                nt as u32,
                c as u32,
                k_dim as u32,
                v_dim as u32,
                vtile as u32,
                7,
            );
        }

        // Export buffers padded to a unique size.
        let v_new_out = Tensor::<1>::empty([nblk * c * v_dim + 262144], &device);
        let states_out = Tensor::<1>::empty([nblk * k_dim * v_dim + 262144], &device);
        let v_new_c = cube_of::<B, 1>(&v_new_out).expect("backend mismatch");
        let states_c = cube_of::<B, 1>(&states_out).expect("backend mismatch");

        {
            let state_initial_c = cube_of::<B, 4>(&state_initial).expect("backend mismatch");
            let cube_dim3 = CubeDim::new_3d(c as u32, (vtile / 2) as u32, 1);
            let cube_count3 = CubeCount::Static(bh as u32, vt as u32, 1);
            unsafe {
                gdn2_chunk_trajectory_export_kernel::launch_unchecked::<f32>(
                    &client,
                    cube_count3,
                    cube_dim3,
                    BufferArg::from_raw_parts(u_c.handle.clone(), nblk * v_dim * c),
                    BufferArg::from_raw_parts(w_c.handle.clone(), nblk * k_dim * c),
                    BufferArg::from_raw_parts(kgd_c.handle.clone(), nblk * c * k_dim),
                    BufferArg::from_raw_parts(glast_c.handle.clone(), nblk * k_dim),
                    BufferArg::from_raw_parts(state_initial_c.handle, bh * k_dim * v_dim),
                    BufferArg::from_raw_parts(v_new_c.handle.clone(), nblk * c * v_dim),
                    BufferArg::from_raw_parts(states_c.handle.clone(), nblk * k_dim * v_dim),
                    nt as u32,
                    c as u32,
                    k_dim as u32,
                    v_dim as u32,
                    vtile as u32,
                );
            }
        }

        Some((
            out.clone(),
            state,
            IntraOut {
                aqk,
                w: w_blk,
                u: u_blk,
                kgd,
                glast,
                qgt,
                wvt,
                m_inv,
                v_new: v_new_out
                    .clone()
                    .slice(0..nblk * c * v_dim)
                    .reshape([nblk, c, v_dim]),
                states: states_out
                    .clone()
                    .slice(0..nblk * k_dim * v_dim)
                    .reshape([nblk, k_dim, v_dim]),
                out,
                gexp,
            },
        ))
    }

    /// Fused chunked forward without the exported intermediates (module path).
    #[allow(clippy::too_many_arguments)]
    pub fn fused_chunk_forward<B: Backend>(
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        g: Tensor<4>,
        b: Tensor<4>,
        w: Tensor<4>,
        state: Tensor<4>,
        scale: f64,
        chunk_size: usize,
    ) -> Option<(Tensor<4>, Tensor<4>)>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        fused_chunk_forward_scratch::<B>(q, k, v, g, b, w, state, scale, chunk_size)
            .map(|(out, state, _io)| (out, state))
    }

    /// Inter kernel only, with a tunable `vtile`/`y_dim` (diagnostics).
    /// `vtile` must equal `y_dim * 2` (the kernel's `n_vp`).
    #[allow(clippy::too_many_arguments)]
    pub fn inter_launch_raw<B: Backend>(
        aqk: Tensor<3>,
        w: Tensor<3>,
        u: Tensor<3>,
        qgt: Tensor<3>,
        kgd: Tensor<3>,
        glast: Tensor<2>,
        state: Tensor<4>,
        out: Tensor<4>,
        scale: f64,
        chunk_size: usize,
        nt: usize,
        bh: usize,
        vtile: usize,
        y_dim: usize,
        stages: u32,
    ) -> Tensor<4>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let k_dim = w.shape().dims::<3>()[2];
        let v_dim = u.shape().dims::<3>()[2];
        let c = chunk_size;
        let _device = state.device();
        let aqk_c = cube_of::<B, 3>(&aqk).expect("backend mismatch");
        let w_c = cube_of::<B, 3>(&w).expect("backend mismatch");
        let u_c = cube_of::<B, 3>(&u).expect("backend mismatch");
        let qgt_c = cube_of::<B, 3>(&qgt).expect("backend mismatch");
        let kgd_c = cube_of::<B, 3>(&kgd).expect("backend mismatch");
        let glast_c = cube_of::<B, 2>(&glast).expect("backend mismatch");
        let state_cube = cube_of::<B, 4>(&state).expect("backend mismatch");
        let out_c = cube_of::<B, 4>(&out).expect("backend mismatch");
        let nblk = bh * nt;
        let client = state_cube.client.clone();
        let nk = nblk * c * k_dim;
        let nv = nblk * c * v_dim;
        let ncc = nblk * c * c;
        let vt = v_dim.div_ceil(vtile);
        let cube_dim2 = CubeDim::new_3d(c as u32, y_dim as u32, 1);
        let cube_count2 = CubeCount::Static(bh as u32, vt as u32, 1);
        unsafe {
            gdn2_chunk_inter_kernel::launch_unchecked::<f32>(
                &client,
                cube_count2,
                cube_dim2,
                BufferArg::from_raw_parts(aqk_c.handle, ncc),
                BufferArg::from_raw_parts(w_c.handle, nk),
                BufferArg::from_raw_parts(u_c.handle, nv),
                BufferArg::from_raw_parts(qgt_c.handle, nk),
                BufferArg::from_raw_parts(kgd_c.handle, nk),
                BufferArg::from_raw_parts(glast_c.handle, nblk * k_dim),
                BufferArg::from_raw_parts(state_cube.handle, bh * k_dim * v_dim),
                BufferArg::from_raw_parts(out_c.handle, bh * nt * c * v_dim),
                scale as f32,
                nt as u32,
                c as u32,
                k_dim as u32,
                v_dim as u32,
                vtile as u32,
                stages,
            );
        }
        out
    }

    /// Intermediate buffers produced by the intra kernel and consumed by the
    /// inter kernel (diagnostics).
    pub struct IntraOut {
        pub aqk: Tensor<3>,
        pub w: Tensor<3>,
        pub u: Tensor<3>,
        pub kgd: Tensor<3>,
        pub glast: Tensor<2>,
        pub qgt: Tensor<3>,
        pub wvt: Tensor<3>,
        pub m_inv: Tensor<3>,
        pub v_new: Tensor<3>,
        pub states: Tensor<3>,
        pub out: Tensor<4>,
        /// exp(cumsum(g)) per chunk, `[nblk, c, k]`.
        pub gexp: Tensor<3>,
    }

    /// Intra kernel only, with a comptime phase mask (diagnostics).
    #[allow(clippy::too_many_arguments)]
    pub fn intra_launch_raw<B: Backend>(
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        g: Tensor<4>,
        b: Tensor<4>,
        w: Tensor<4>,
        state: Tensor<4>,
        scale: f64,
        chunk_size: usize,
        stages: u32,
        y_dim: usize,
    ) -> IntraOut
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let [batch, heads, time, k_dim] = q.shape().dims::<4>();
        let v_dim = v.shape().dims::<4>()[3];
        let c = chunk_size;
        let nt = time / c;
        let bh = batch * heads;
        let device = state.device();
        let q = cube_of::<B, 4>(&q).expect("backend mismatch");
        let k = cube_of::<B, 4>(&k).expect("backend mismatch");
        let v = cube_of::<B, 4>(&v).expect("backend mismatch");
        let g = cube_of::<B, 4>(&g).expect("backend mismatch");
        let b = cube_of::<B, 4>(&b).expect("backend mismatch");
        let w = cube_of::<B, 4>(&w).expect("backend mismatch");
        let state_cube = cube_of::<B, 4>(&state).expect("backend mismatch");
        let nblk = bh * nt;
        let client = state_cube.client.clone();
        let mk = |shape: [usize; 3]| -> Tensor<3> { Tensor::<3>::empty(shape, &device) };
        let gexp = mk([nblk, c, k_dim]);
        let kgt = mk([nblk, k_dim, c]);
        let qgt = mk([nblk, k_dim, c]);
        let bkt = mk([nblk, k_dim, c]);
        let aqk = mk([nblk, c, c]);
        let akk = mk([nblk, c, c]);
        let w_blk = mk([nblk, c, k_dim]);
        let u_blk = mk([nblk, c, v_dim]);
        let kgd = mk([nblk, c, k_dim]);
        let wvt = mk([nblk, v_dim, c]);
        let glast = Tensor::<2>::empty([nblk, k_dim], &device);
        let m_inv = mk([nblk, c, c]);
        let v_new_out = mk([nblk, c, v_dim]);
        let states_out = mk([nblk, k_dim, v_dim]);
        let out = Tensor::<4>::empty([batch, heads, time, v_dim], &device);
        let _v_new_c = cube_of::<B, 3>(&v_new_out).expect("backend mismatch");
        let _states_c = cube_of::<B, 3>(&states_out).expect("backend mismatch");
        let m_inv_c = cube_of::<B, 3>(&m_inv).expect("backend mismatch");
        let gexp_c = cube_of::<B, 3>(&gexp).expect("backend mismatch");
        let kgt_c = cube_of::<B, 3>(&kgt).expect("backend mismatch");
        let qgt_c = cube_of::<B, 3>(&qgt).expect("backend mismatch");
        let bkt_c = cube_of::<B, 3>(&bkt).expect("backend mismatch");
        let aqk_c = cube_of::<B, 3>(&aqk).expect("backend mismatch");
        let akk_c = cube_of::<B, 3>(&akk).expect("backend mismatch");
        let w_c = cube_of::<B, 3>(&w_blk).expect("backend mismatch");
        let u_c = cube_of::<B, 3>(&u_blk).expect("backend mismatch");
        let kgd_c = cube_of::<B, 3>(&kgd).expect("backend mismatch");
        let wvt_c = cube_of::<B, 3>(&wvt).expect("backend mismatch");
        let glast_c = cube_of::<B, 2>(&glast).expect("backend mismatch");
        let _out_c = cube_of::<B, 4>(&out).expect("backend mismatch");
        let nk = nblk * c * k_dim;
        let nv = nblk * c * v_dim;
        let ncc = nblk * c * c;
        let cube_dim = CubeDim::new_3d(c as u32, y_dim as u32, 1);
        let cube_count = CubeCount::Static(nblk as u32, 1, 1);
        unsafe {
            gdn2_chunk_intra_kernel::launch_unchecked::<f32>(
                &client,
                cube_count,
                cube_dim,
                BufferArg::from_raw_parts(q.handle, nk),
                BufferArg::from_raw_parts(k.handle, nk),
                BufferArg::from_raw_parts(g.handle, nk),
                BufferArg::from_raw_parts(b.handle, nk),
                BufferArg::from_raw_parts(v.handle, nv),
                BufferArg::from_raw_parts(w.handle, nv),
                BufferArg::from_raw_parts(gexp_c.handle, nk),
                BufferArg::from_raw_parts(kgt_c.handle, nk),
                BufferArg::from_raw_parts(qgt_c.handle.clone(), nk),
                BufferArg::from_raw_parts(bkt_c.handle, nk),
                BufferArg::from_raw_parts(wvt_c.handle, nv),
                BufferArg::from_raw_parts(aqk_c.handle.clone(), ncc),
                BufferArg::from_raw_parts(akk_c.handle, ncc),
                BufferArg::from_raw_parts(m_inv_c.handle.clone(), ncc),
                BufferArg::from_raw_parts(w_c.handle.clone(), nk),
                BufferArg::from_raw_parts(u_c.handle.clone(), nv),
                BufferArg::from_raw_parts(kgd_c.handle.clone(), nk),
                BufferArg::from_raw_parts(glast_c.handle.clone(), nblk * k_dim),
                scale as f32,
                c as u32,
                k_dim as u32,
                v_dim as u32,
                stages,
            );
        }
        IntraOut {
            aqk,
            w: w_blk,
            u: u_blk,
            kgd,
            glast,
            qgt,
            wvt,
            m_inv,
            v_new: v_new_out,
            states: states_out,
            out,
            gexp,
        }
    }
}
