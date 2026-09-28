//! Fused chunked backward (adjoint) kernels for the WY delta rule.
//!
//! Mirrors `chunk_cube.rs`: BK1 is the token-parallel per-chunk adjoint,
//! BK2 is the sequential reverse recurrence (BPTT through the state). The
//! forward kernels export `M⁻¹`, `aqk`, `qE`, decay, `v_new` and the per-chunk
//! states, so the backward never re-runs the forward; only a few elementwise
//! glue ops run on the tensor path in the wrapper.
//!
//! Layouts (same as the forward kernels): aqk `[s][r] = score(q_r,k_s)`,
//! w/U `[k|v][r]` (transposed), qgt `[k][r]`, everything else `[r][·]`; all
//! `[nblk, ·, ·]` with `nblk = bh·nt`.
//!
//! Thread mapping: cube `(c, gy)`; thread `(r, grp)` owns ROW r of the chunk
//! and computes the row-local terms; cross-row dependencies (the `d_qk`
//! column, the `d_W`/`d_v_new` columns for the `M⁻ᵀ` solves, the `d_e`
//! column for the reverse cumsum) are staged through shared memory. c is
//! small (16), so each thread's serial work stays a few microseconds.

use cubecl::prelude::*;

/// Token-parallel intra-chunk adjoint (BK1). One cube per chunk.
#[cube(launch_unchecked)]
fn gdn2_chunk_intra_adjoint_kernel<F: Float>(
    d_out: &[F],    // [nblk, C, V]
    v_new: &[F],    // [nblk, C, V]
    qgt: &[F],      // [nblk, K, C]  qE transposed [k][r]
    gexp: &[F],     // [nblk, C, K]
    m_inv: &[F],    // [nblk, C, C]  M^-1 [row][col]
    w_buf: &[F],    // [nblk, K, C]  W transposed
    u_buf: &[F],    // [nblk, V, C]  U transposed
    k_buf: &[F],    // [nblk, C, K]
    b_buf: &[F],    // [nblk, C, K]
    v_buf: &[F],    // [nblk, C, V]
    wg_buf: &[F],   // [nblk, C, V]
    s_before: &[F], // [nblk, K, V]
    d_v_new: &[F],  // [nblk, C, V]  BK2 output (incl. BPTT part)
    d_k_bptt: &[F], // [nblk, C, K]  BK2 output
    d_e_bptt: &[F], // [nblk, C, K]  BK2 output (d_e_decay)
    d_e_last: &[F], // [nblk, K]     BK2 output
    d_q: &mut [F],  // [nblk, C, K]
    d_k: &mut [F],  // [nblk, C, K]
    d_b: &mut [F],  // [nblk, C, K]
    d_g: &mut [F],  // [nblk, C, K]
    d_v: &mut [F],  // [nblk, C, V]
    d_w: &mut [F],  // [nblk, C, V]
    scale: f32,
    #[comptime] chunk_c: u32,
    #[comptime] k_dim: u32,
    #[comptime] v_dim: u32,
) {
    let block = CUBE_POS_X as usize;
    let r = UNIT_POS_X as usize;
    let c = chunk_c as usize;
    let kd = k_dim as usize;
    let vd = v_dim as usize;
    let base = block * c * c;
    let kb = block * c * kd;
    let vb = block * c * vd;
    let sb = block * kd * vd;

    let mut d_qk_sh = Shared::<[F]>::new_slice(c * c);
    let mut d_w_sh = Shared::<[F]>::new_slice(c * kd);
    let mut d_vn_sh = Shared::<[F]>::new_slice(c * vd);
    let mut d_qe_sh = Shared::<[F]>::new_slice(c * kd);
    let mut d_kg_sh = Shared::<[F]>::new_slice(c * kd);
    let mut d_rhs_k_sh = Shared::<[F]>::new_slice(c * kd);
    let mut d_rhs_v_sh = Shared::<[F]>::new_slice(c * vd);
    let mut d_akk_sh = Shared::<[F]>::new_slice(c * c);
    let mut d_e_sh = Shared::<[F]>::new_slice(c * kd);

    if r < c {
        // 1) d_qk row: d_aqk[r][s]·scale·causal, stage the full d_qk.
        for s in 0..c {
            let mut a = F::new(0.0_f32);
            for vv in 0..vd {
                a += d_out[vb + r * vd + vv] * v_new[vb + s * vd + vv];
            }
            let causal = if s <= r {
                F::new(1.0_f32)
            } else {
                F::new(0.0_f32)
            };
            d_qk_sh[r * c + s] = a * F::cast_from(scale) * causal;
        }
        sync_cube();

        // 2) d_qe, d_k_g (first term), d_W rows.
        for k in 0..kd {
            let mut qe = F::new(0.0_f32);
            let mut kg1 = F::new(0.0_f32);
            for s in 0..c {
                let kg_s = k_buf[kb + s * kd + k] / gexp[kb + s * kd + k];
                qe += d_qk_sh[r * c + s] * kg_s;
                kg1 += d_qk_sh[s * c + r] * qgt[(block * kd + k) * c + s];
            }
            let mut qe_s = F::new(0.0_f32);
            let mut wd = F::new(0.0_f32);
            for vv in 0..vd {
                let s_b = s_before[sb + k * vd + vv];
                qe_s += d_out[vb + r * vd + vv] * s_b;
                wd += d_v_new[vb + r * vd + vv] * s_b;
            }
            d_qe_sh[r * kd + k] = qe + qe_s * F::cast_from(scale);
            d_kg_sh[r * kd + k] = kg1;
            d_w_sh[r * kd + k] = -wd;
        }
        for vv in 0..vd {
            d_vn_sh[r * vd + vv] = d_v_new[vb + r * vd + vv];
        }
        sync_cube();

        // 3) M⁻ᵀ solves: d_rhs_k[r][k] = Σ_s M⁻¹[s][r]·d_W[s][k],
        //                 d_rhs_v[r][v] = Σ_s M⁻¹[s][r]·d_v_new[s][v].
        for k in 0..kd {
            let mut acc = F::new(0.0_f32);
            for s in 0..c {
                acc += m_inv[base + s * c + r] * d_w_sh[s * kd + k];
            }
            d_rhs_k_sh[r * kd + k] = acc;
        }
        for vv in 0..vd {
            let mut acc = F::new(0.0_f32);
            for s in 0..c {
                acc += m_inv[base + s * c + r] * d_vn_sh[s * vd + vv];
            }
            d_rhs_v_sh[r * vd + vv] = acc;
        }
        sync_cube();

        // 4) d_akk row; then d_bkE, the d_k_g second term and all row-local
        //    elementwise gradients; assemble d_e.
        for s in 0..c {
            let mut a1 = F::new(0.0_f32);
            for k in 0..kd {
                a1 += d_rhs_k_sh[r * kd + k] * w_buf[(block * kd + k) * c + s];
            }
            let mut a2 = F::new(0.0_f32);
            for vv in 0..vd {
                a2 += d_rhs_v_sh[r * vd + vv] * u_buf[(block * vd + vv) * c + s];
            }
            let strict = if s < r {
                F::new(1.0_f32)
            } else {
                F::new(0.0_f32)
            };
            d_akk_sh[r * c + s] = -(a1 + a2) * strict;
        }
        sync_cube();

        for k in 0..kd {
            let mut bk_e = F::new(0.0_f32);
            let mut kg2 = F::new(0.0_f32);
            for s in 0..c {
                let kg_s = k_buf[kb + s * kd + k] / gexp[kb + s * kd + k];
                bk_e += d_akk_sh[r * c + s] * kg_s;
                // rhs_k[s][k] = b[s][k]·k[s][k]·E[s][k]
                let rhs_s = b_buf[kb + s * kd + k] * k_buf[kb + s * kd + k] * gexp[kb + s * kd + k];
                kg2 += d_akk_sh[s * c + r] * rhs_s;
            }
            let e = gexp[kb + r * kd + k];
            let kk = k_buf[kb + r * kd + k];
            let bb = b_buf[kb + r * kd + k];
            let bk = (bb * kk) * e;
            let d_rhs_k = d_rhs_k_sh[r * kd + k];
            let d_qe_k = d_qe_sh[r * kd + k];
            let d_k_g = d_kg_sh[r * kd + k] + kg2;
            let d_bk = (d_rhs_k + bk_e) * e;

            d_q[kb + r * kd + k] = d_qe_k * e;
            d_k[kb + r * kd + k] = d_bk * bb + d_k_g / e + d_k_bptt[kb + r * kd + k];
            d_b[kb + r * kd + k] = d_bk * kk;

            let e_rhsk = d_rhs_k * bk;
            let e_kg = -(d_k_g * kk / (e * e));
            // q = qE/E = qgt[k][r]/E[r][k] (q is not checkpointed)
            let q_val = qgt[(block * kd + k) * c + r] / e;
            let e_qe = d_qe_k * q_val;
            let e_bke = bk_e * bk;
            d_e_sh[r * kd + k] = e_rhsk + e_kg + e_qe + e_bke + d_e_bptt[kb + r * kd + k];
            d_kg_sh[r * kd + k] = d_k_g;
        }
        if r == c - 1 {
            for k in 0..kd {
                d_e_sh[r * kd + k] += d_e_last[block * kd + k];
            }
        }
        for vv in 0..vd {
            let d_rhs_v = d_rhs_v_sh[r * vd + vv];
            d_v[vb + r * vd + vv] = d_rhs_v * wg_buf[vb + r * vd + vv];
            d_w[vb + r * vd + vv] = d_rhs_v * v_buf[vb + r * vd + vv];
        }
        sync_cube();

        // 5) d_g row: reverse cumsum over t of d_e[t][k]·E[t][k], t = r..c.
        if r < c {
            for k in 0..kd {
                let mut acc = F::new(0.0_f32);
                let mut t = r;
                while t < c {
                    acc += d_e_sh[t * kd + k] * gexp[kb + t * kd + k];
                    t += 1;
                }
                d_g[kb + r * kd + k] = acc;
            }
        }
    }
}

/// Sequential inter-chunk adjoint (BK2): the BPTT chain over the state.
///
/// One cube per (bh, value tile). Iterates the chunks in reverse; per chunk:
///   d_v_new = aqk^T·d_out + (k·decay)·d_state_acc
///   d_s_c   = -W^T·d_v_new + scale·qE^T·d_out + g_last ⊙ d_state_acc
///   d_state_acc = d_s_c
///   d_K̂ = v_new·d_state_acc^T;  d_k_bptt = d_K̂⊙decay;  d_decay = d_K̂⊙k
///   d_e_last = Σ_t d_decay/E + Σ_v d_state_acc⊙S_before
///   d_e_bptt = -d_decay·decay/E
/// Writes d_v_new (full), d_k_bptt, d_e_bptt, d_e_last per chunk and the
/// state adjoint at chunk 0 as `d_s`.
#[cube(launch_unchecked)]
fn gdn2_chunk_inter_adjoint_kernel<F: Float>(
    d_out: &[F],           // [nblk, C, V]
    v_new: &[F],           // [nblk, C, V]
    aqk: &[F],             // [nblk, C, C] transposed [s][r]
    qgt: &[F],             // [nblk, K, C] qE transposed
    w_buf: &[F],           // [nblk, K, C] W transposed
    glast: &[F],           // [nblk, K]
    gexp: &[F],            // [nblk, C, K]
    k_buf: &[F],           // [nblk, C, K]
    s_before: &[F],        // [nblk, K, V]
    state_in: &[F],        // [bh, K, V] initial state adjoint (zeros)
    d_v_new_out: &mut [F], // [nblk, C, V]
    d_s: &mut [F],         // [nblk, K, V] state adjoint per chunk
    scale: f32,
    nt: u32,
    #[comptime] chunk_c: u32,
    #[comptime] k_dim: u32,
    #[comptime] v_dim: u32,
    #[comptime] vtile: u32,
) {
    let bh = CUBE_POS_X as usize;
    let vt = CUBE_POS_Y as usize;
    let r = UNIT_POS_X as usize;
    let c = chunk_c as usize;
    let kd = k_dim as usize;
    let vd = v_dim as usize;
    let vtile = vtile as usize;
    let vs = vt * vtile;
    let n_vp = 2;

    let mut vn_sh = Shared::<[F]>::new_slice(c * vtile);
    let mut d_sh = Shared::<[F]>::new_slice(kd * vtile);

    if r < c {
        // Load the initial state adjoint slice.
        {
            let mut jj = 0;
            while jj < n_vp {
                let vv = vs + (grp_id() as usize) * n_vp + jj;
                if vv < vd {
                    let mut kk = r;
                    while kk < kd {
                        d_sh[kk * vtile + (grp_id() as usize) * n_vp + jj] =
                            state_in[bh * kd * vd + kk * vd + vv];
                        kk += c;
                    }
                }
                jj += 1;
            }
        }
        sync_cube();

        let mut t = nt as usize;
        while t > 0 {
            t -= 1;
            let cb = bh * (nt as usize) + t;
            let cbase = cb * c;

            // d_v_new[r][v] = Σ_s aqk[s][r]·d_out[s][v] + Σ_k K̂[r][k]·d_sh[k][v]
            // (d_out is indexed by the GLOBAL v; d_sh by the LOCAL slot)
            let v0 = (grp_id() as usize) * n_vp;
            let vv0 = vs + v0;
            let mut acc_a = F::new(0.0_f32);
            let mut acc_b = F::new(0.0_f32);
            let mut s = 0;
            while s < c {
                // d_v_new[r][v] = Σ_s score(q_s,k_r)·d_out[s][v]; the export stores
                // aqk[s][r] = score(q_r,k_s), so the correct entry is [r][s].
                let a = aqk[cbase * c + r * c + s];
                acc_a += a * d_out[cbase * vd + s * vd + vv0];
                if vv0 + 1 < vd {
                    acc_b += a * d_out[cbase * vd + s * vd + vv0 + 1];
                }
                s += 1;
            }
            let mut kk = 0;
            while kk < kd {
                // K̂ = k·decay = k[r][k]·glast[k]/E[r][k]
                let khat = k_buf[cbase * kd + r * kd + kk] * glast[cb * kd + kk]
                    / gexp[cbase * kd + r * kd + kk];
                acc_a += khat * d_sh[kk * vtile + v0];
                if v0 + 1 < vd {
                    acc_b += khat * d_sh[kk * vtile + v0 + 1];
                }
                kk += 1;
            }
            vn_sh[r * vtile + v0] = acc_a;
            if v0 + 1 < vd {
                vn_sh[r * vtile + v0 + 1] = acc_b;
            }
            sync_cube();

            // d_s_c[k][v] = -Σ_r W[r][k]·d_v_new[r][v] + scale·Σ_r qE[r][k]·d_out[r][v]
            //               + g_last[k]·d_sh[k][v]
            let mut jj = 0;
            while jj < n_vp {
                let vv = vs + (grp_id() as usize) * n_vp + jj;
                if vv < vd {
                    let mut kk = r;
                    while kk < kd {
                        let mut acc1 = F::new(0.0_f32);
                        let mut acc2 = F::new(0.0_f32);
                        let mut rr = 0;
                        while rr < c {
                            acc1 += w_buf[(cb * kd + kk) * c + rr]
                                * vn_sh[rr * vtile + (grp_id() as usize) * n_vp + jj];
                            acc2 += qgt[(cb * kd + kk) * c + rr] * d_out[cbase * vd + rr * vd + vv];
                            rr += 1;
                        }
                        let new_d = -acc1
                            + acc2 * F::cast_from(scale)
                            + glast[cb * kd + kk]
                                * d_sh[kk * vtile + (grp_id() as usize) * n_vp + jj];
                        d_sh[kk * vtile + (grp_id() as usize) * n_vp + jj] = new_d;
                        kk += c;
                    }
                }
                jj += 1;
            }
            sync_cube();

            // d_v_new output + the per-chunk state adjoint (d_s_c).
            let mut jj = 0;
            while jj < n_vp {
                let vv = vs + (grp_id() as usize) * n_vp + jj;
                if vv < vd {
                    d_v_new_out[cbase * vd + r * vd + vv] =
                        vn_sh[r * vtile + (grp_id() as usize) * n_vp + jj];
                    let mut kk = r;
                    while kk < kd {
                        d_s[cb * kd * vd + kk * vd + vv] =
                            d_sh[kk * vtile + (grp_id() as usize) * n_vp + jj];
                        kk += c;
                    }
                }
                jj += 1;
            }
            // d_K̂[r][k] = Σ_v v_new[r][v]·d_sh[k][v]  (only the tile's v-slice;
            // the full sum needs all tiles — accumulate into shared, reduced by
            // the wrapper? — per-tile partials are wrong. Instead: each (bh, vt)
            // cube sums its OWN v-slice; the k-bptt/e-last terms need the FULL
            // sum over v. Handle with an atomics-free two-phase: stage the
            // per-(r,k) partial into a global accumulator and reduce on the
            // host? — ponytail: compute d_K̂ in the wrapper's glue via the
            // exported v_new/d_s buffers (cheap tensor ops), and only d_e_bptt /
            // d_k_bptt per-tile are written here for their own v-slices... — NO:
            // d_k_bptt and d_e_bptt also need the full d_K̂.
            //
            // Simplest correct split: this kernel writes only d_v_new and the
            // state chain (d_s); the BPTT k/e terms (d_K̂, d_k_bptt, d_decay,
            // d_e_last, d_e_bptt) are computed by the wrapper with a few tensor
            // ops from the exported v_new/d_s/decay/k/E — still ~10 launches
            // total, not per-chunk.
            let _ = v_new;
            let _ = s_before;
        }
    }
}

/// Y-lane id (the forward inter kernel names it `grp`).
#[cube]
fn grp_id() -> u32 {
    UNIT_POS_Y
}

/// Fused chunked backward: launch plumbing + the small tensor glue.
#[cfg(feature = "cuda")]
pub mod cuda {
    use super::*;
    use burn::backend::{Backend, DispatchKindConversion};
    use burn::tensor::{DispatchTensor, Tensor};
    use burn_cubecl::tensor::CubeTensor;
    use burn_cubecl::CubeBackend;
    use std::any::TypeId;

    /// The bare (non-fusion) CUDA backend the fused kernels target.
    pub type CudaBare = CubeBackend;

    pub fn is_cuda<B: Backend>() -> bool {
        TypeId::of::<B>() == TypeId::of::<CudaBare>()
    }

    /// Owned copy of the underlying `CubeTensor` of `t`, if `B` is the bare
    /// CUDA `CubeBackend` and the buffer is row-major contiguous. A strided
    /// view is materialized once, never recursed into — see
    /// [`crate::kernel::contiguous_cube_of`].
    fn cube_of<B: Backend, const D: usize>(
        t: &Tensor<D>,
    ) -> Option<CubeTensor>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        crate::kernel::contiguous_cube_of::<B, D>(t)
    }

    /// Exported forward buffers consumed by the fused backward.
    #[derive(Clone, Debug)]
    pub struct FusedBackwardInputs {
        /// M^-1, `[nblk, c, c]`.
        pub m_inv: Tensor<3>,
        /// aqk (transposed `[s][r]`), `[nblk, c, c]`.
        pub aqk: Tensor<3>,
        /// qE transposed `[k][r]`, `[nblk, k, c]`.
        pub qgt: Tensor<3>,
        /// exp(cumsum(g)) last row, `[nblk, k]`.
        pub glast: Tensor<2>,
        /// v_new, `[nblk, c, v]`.
        pub v_new: Tensor<3>,
        /// state before each chunk, `[nblk, k, v]`.
        pub states: Tensor<3>,
        /// W transposed `[k][r]`, `[nblk, k, c]`.
        pub w: Tensor<3>,
        /// U transposed `[v][r]`, `[nblk, v, c]`.
        pub u: Tensor<3>,
        /// exp(cumsum(g)) per chunk, `[nblk, c, k]`.
        pub gexp: Tensor<3>,
    }

    /// Gradient tensors of the chunked forward, full sequence shape.
    pub struct FusedBackwardOutput {
        pub d_q: Tensor<4>,
        pub d_k: Tensor<4>,
        pub d_v: Tensor<4>,
        pub d_g: Tensor<4>,
        pub d_b: Tensor<4>,
        pub d_w: Tensor<4>,
        pub d_s: Tensor<4>,
    }

    /// Run the fused backward. Returns `None` when the backend is not the
    /// bare CUDA `CubeBackend`, in which case the caller falls back to the
    /// tensor-path adjoint.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_chunk_backward<B: Backend>(
        fwd: &FusedBackwardInputs,
        k: &Tensor<4>,
        v: &Tensor<4>,
        b: &Tensor<4>,
        wg: &Tensor<4>,
        d_out: &Tensor<4>,
        scale: f64,
        chunk_size: usize,
    ) -> Option<FusedBackwardOutput>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let [batch, heads, time, k_dim] = d_out.shape().dims::<4>();
        let v_dim = d_out.shape().dims::<4>()[3];
        let c = chunk_size;
        let nt = time / c;
        let nblk = batch * heads * nt;
        let bh = batch * heads;
        let device = d_out.device();

        let k_c = cube_of::<B, 4>(k)?;
        let v_c = cube_of::<B, 4>(v)?;
        let b_c = cube_of::<B, 4>(b)?;
        let wg_c = cube_of::<B, 4>(wg)?;
        let do_c = cube_of::<B, 4>(d_out)?;
        let m_inv_c = cube_of::<B, 3>(&fwd.m_inv)?;
        let aqk_c = cube_of::<B, 3>(&fwd.aqk)?;
        let qgt_c = cube_of::<B, 3>(&fwd.qgt)?;
        let glast_c = cube_of::<B, 2>(&fwd.glast)?;
        let v_new_c = cube_of::<B, 3>(&fwd.v_new)?;
        let states_c = cube_of::<B, 3>(&fwd.states)?;
        let w_c = cube_of::<B, 3>(&fwd.w)?;
        let u_c = cube_of::<B, 3>(&fwd.u)?;
        let client = m_inv_c.client.clone();

        // output buffers
        let mk3 = |shape: [usize; 3]| -> Tensor<3> { Tensor::<3>::zeros(shape, &device) };
        let mk4 = |shape: [usize; 4]| -> Tensor<4> { Tensor::<4>::zeros(shape, &device) };
        let d_v_new = mk3([nblk, c, v_dim]);
        let d_s_flat = mk3([nblk, k_dim, v_dim]);

        let d_k_bptt = mk3([nblk, c, k_dim]);
        let d_e_bptt = mk3([nblk, c, k_dim]);
        let d_e_last = Tensor::<2>::empty([nblk, k_dim], &device);
        let d_q = mk4([batch, heads, time, k_dim]);
        let d_k = mk4([batch, heads, time, k_dim]);
        let d_b = mk4([batch, heads, time, k_dim]);
        let d_g = mk4([batch, heads, time, k_dim]);
        let d_v = mk4([batch, heads, time, v_dim]);
        let d_w = mk4([batch, heads, time, v_dim]);
        let d_s = mk4([batch, heads, k_dim, v_dim]);

        let d_v_new_c = cube_of::<B, 3>(&d_v_new).expect("backend mismatch");
        let d_s_c = cube_of::<B, 3>(&d_s_flat).expect("backend mismatch");
        let d_k_bptt_c = cube_of::<B, 3>(&d_k_bptt).expect("backend mismatch");
        let d_e_bptt_c = cube_of::<B, 3>(&d_e_bptt).expect("backend mismatch");
        let d_e_last_c = cube_of::<B, 2>(&d_e_last).expect("backend mismatch");
        let d_q_c = cube_of::<B, 4>(&d_q).expect("backend mismatch");
        let d_k_c = cube_of::<B, 4>(&d_k).expect("backend mismatch");
        let d_b_c = cube_of::<B, 4>(&d_b).expect("backend mismatch");
        let d_g_c = cube_of::<B, 4>(&d_g).expect("backend mismatch");
        let d_v_c = cube_of::<B, 4>(&d_v).expect("backend mismatch");
        let d_w_c = cube_of::<B, 4>(&d_w).expect("backend mismatch");
        let d_s_out_c = cube_of::<B, 4>(&d_s).expect("backend mismatch");
        for (l, b) in [
            ("bwd:d_v_new", crate::alloc_trace::bytes_of(&d_v_new)),
            ("bwd:d_s trajectory", crate::alloc_trace::bytes_of(&d_s_flat)),
            ("bwd:d_q", crate::alloc_trace::bytes_of(&d_q)),
            ("bwd:d_k", crate::alloc_trace::bytes_of(&d_k)),
            ("bwd:d_b", crate::alloc_trace::bytes_of(&d_b)),
            ("bwd:d_g", crate::alloc_trace::bytes_of(&d_g)),
            ("bwd:d_v", crate::alloc_trace::bytes_of(&d_v)),
            ("bwd:d_w", crate::alloc_trace::bytes_of(&d_w)),
            ("bwd:d_s", crate::alloc_trace::bytes_of(&d_s)),
        ] {
            crate::alloc_trace::note(l, b);
        }
        for (l, b) in [
            ("bwd:d_k_bptt (discarded)", crate::alloc_trace::bytes_of(&d_k_bptt)),
            ("bwd:d_e_bptt (discarded)", crate::alloc_trace::bytes_of(&d_e_bptt)),
            ("bwd:d_e_last (discarded)", crate::alloc_trace::bytes_of(&d_e_last)),
        ] {
            crate::alloc_trace::note(l, b);
        }
        let state_in = Tensor::<3>::zeros([bh, k_dim, v_dim], &device);
        crate::alloc_trace::note("bwd:state_in", crate::alloc_trace::bytes_of(&state_in));
        let state_in_c = cube_of::<B, 3>(&state_in).expect("backend mismatch");

        // E = exp(cumsum(g)) exported directly by the forward kernel. The old
        // k·glast/kgd reconstruction 0/0-NaNs when a key row is exactly 0.
        let e_flat = fwd.gexp.clone().reshape([nblk, c, k_dim]);
        let e_flat_c = cube_of::<B, 3>(&e_flat).expect("backend mismatch");

        // BK2: the sequential BPTT chain.
        let vtile = 8usize;
        let vt = v_dim.div_ceil(vtile);
        // the vtile column is covered by y·n_vp lanes (same as the forward inter)
        let cube_dim = CubeDim::new_3d(c as u32, (vtile / 2) as u32, 1);
        let cube_count = CubeCount::Static(bh as u32, vt as u32, 1);
        // The counter means "a kernel launch happened", so it is incremented
        // HERE, past every gate above (the `is_cuda` TypeId, the `cube_of`
        // conversions, and their contiguity materializations). It used to be
        // the FIRST statement, so `fused_calls().1` counted a fused backward
        // that had returned `None` on the next line — which is why
        // `tests/autodiff_cuda_gate.rs`'s `bwd > 0` was green while the fused
        // adjoint had never run (ADR-0019).
        crate::cuda_dispatch::note_fused_backward();
        unsafe {
            gdn2_chunk_inter_adjoint_kernel::launch_unchecked::<f32>(
                &client,
                cube_count,
                cube_dim,
                BufferArg::from_raw_parts(do_c.handle.clone(), nblk * c * v_dim),
                BufferArg::from_raw_parts(v_new_c.handle.clone(), nblk * c * v_dim),
                BufferArg::from_raw_parts(aqk_c.handle.clone(), nblk * c * c),
                BufferArg::from_raw_parts(qgt_c.handle.clone(), nblk * k_dim * c),
                BufferArg::from_raw_parts(w_c.handle.clone(), nblk * k_dim * c),
                BufferArg::from_raw_parts(glast_c.handle.clone(), nblk * k_dim),
                BufferArg::from_raw_parts(e_flat_c.handle.clone(), nblk * c * k_dim),
                BufferArg::from_raw_parts(k_c.handle.clone(), nblk * c * k_dim),
                BufferArg::from_raw_parts(states_c.handle.clone(), nblk * k_dim * v_dim),
                BufferArg::from_raw_parts(state_in_c.handle, bh * k_dim * v_dim),
                BufferArg::from_raw_parts(d_v_new_c.handle.clone(), nblk * c * v_dim),
                BufferArg::from_raw_parts(d_s_c.handle.clone(), nblk * k_dim * v_dim),
                scale as f32,
                nt as u32,
                c as u32,
                k_dim as u32,
                v_dim as u32,
                vtile as u32,
            );
        }

        // E = exp(cumsum(g)) exported by the forward kernel (no 0/0).
        let k_r = k.clone().reshape([batch, heads, nt, c, k_dim]);
        let e_full = fwd.gexp.clone().reshape([batch, heads, nt, c, k_dim]);
        let v_new_r = fwd.v_new.clone().reshape([batch, heads, nt, c, v_dim]);
        let d_s_r = d_s_flat.clone().reshape([batch, heads, nt, k_dim, v_dim]);
        // d_K̂[i] = v_new[i]·dSacc(i+1)^T — the state adjoint AFTER chunk i
        // (the shifted chain); the last chunk's dSacc is zero.
        let d_s_shift = Tensor::cat(
            vec![
                d_s_r
                    .clone()
                    .slice([0..batch, 0..heads, 1..nt, 0..k_dim, 0..v_dim]),
                Tensor::<5>::zeros([batch, heads, 1, k_dim, v_dim], &device),
            ],
            2,
        );
        // batched [c,v]@[v,k] over b1 = B·H·nt (explicit 3D: the 5D matmul
        // path reshapes internally in a way that breaks non-contiguous RHS)
        let b1 = batch * heads * nt;
        let v_new_3 = v_new_r.clone().reshape([b1, c, v_dim]);
        let d_s_3 = d_s_shift.clone().reshape([b1, v_dim, k_dim]);
        let d_hat = v_new_3.matmul(d_s_3).reshape([batch, heads, nt, c, k_dim]); // [B,H,nt,c,k]
        let decay = fwd
            .glast
            .clone()
            .reshape([batch, heads, nt, 1, k_dim])
            .repeat(&[1, 1, 1, c, 1])
            .div(e_full.clone());
        let d_k_bptt_r = d_hat.clone() * decay.clone();
        let d_decay = d_hat * k_r.clone();
        let d_e_bptt_r = -(d_decay.clone() * decay.clone()) / e_full.clone();
        let d_e_last_3 = (d_decay.clone() / e_full.clone()).reshape([b1, c, k_dim]);
        let d_e_last_2 = d_e_last_3.sum_dim(1);
        let d_e_last_r = d_e_last_2.reshape([batch, heads, nt, k_dim])
            + d_s_shift
                .clone()
                .mul(fwd.states.clone().reshape([batch, heads, nt, k_dim, v_dim]))
                .reshape([b1, k_dim, v_dim])
                .sum_dim(2)
                .reshape([batch, heads, nt, k_dim]);

        // push the glue results into the flat buffers
        let d_k_bptt_flat = d_k_bptt_r.reshape([nblk, c, k_dim]);
        let d_e_bptt_flat = d_e_bptt_r.reshape([nblk, c, k_dim]);
        let d_e_last_flat = d_e_last_r.reshape([nblk, k_dim]);
        let _ = d_k_bptt;
        let _ = d_e_bptt;
        let _ = d_e_last;
        let d_k_bptt = d_k_bptt_flat;
        let d_e_bptt = d_e_bptt_flat;
        let d_e_last = d_e_last_flat;
        let d_k_bptt_c2 = cube_of::<B, 3>(&d_k_bptt).expect("backend mismatch");
        let d_e_bptt_c2 = cube_of::<B, 3>(&d_e_bptt).expect("backend mismatch");
        let d_e_last_c2 = cube_of::<B, 2>(&d_e_last).expect("backend mismatch");
        let _ = d_k_bptt_c;
        let _ = d_e_bptt_c;
        let _ = d_e_last_c;

        let e_flat = e_full.clone().reshape([nblk, c, k_dim]);
        let e_flat_c = cube_of::<B, 3>(&e_flat).expect("backend mismatch");
        // BK1: the token-parallel intra-chunk adjoint.
        // Materialize d_v_new: the output buffers share the allocator's
        // address space and can alias the BK2 scratch buffer otherwise.
        let d_v_new_fresh = d_v_new.clone().mul_scalar(1.0);
        let d_v_new_fresh_c = cube_of::<B, 3>(&d_v_new_fresh).expect("backend mismatch");
        let cube_dim1 = CubeDim::new_3d(c as u32, 8, 1);
        let cube_count1 = CubeCount::Static(nblk as u32, 1, 1);
        unsafe {
            gdn2_chunk_intra_adjoint_kernel::launch_unchecked::<f32>(
                &client,
                cube_count1,
                cube_dim1,
                BufferArg::from_raw_parts(do_c.handle.clone(), nblk * c * v_dim),
                BufferArg::from_raw_parts(v_new_c.handle.clone(), nblk * c * v_dim),
                BufferArg::from_raw_parts(qgt_c.handle.clone(), nblk * k_dim * c),
                BufferArg::from_raw_parts(e_flat_c.handle.clone(), nblk * c * k_dim),
                BufferArg::from_raw_parts(m_inv_c.handle.clone(), nblk * c * c),
                BufferArg::from_raw_parts(w_c.handle, nblk * k_dim * c),
                BufferArg::from_raw_parts(u_c.handle, nblk * v_dim * c),
                BufferArg::from_raw_parts(k_c.handle, nblk * c * k_dim),
                BufferArg::from_raw_parts(b_c.handle, nblk * c * k_dim),
                BufferArg::from_raw_parts(v_c.handle, nblk * c * v_dim),
                BufferArg::from_raw_parts(wg_c.handle, nblk * c * v_dim),
                BufferArg::from_raw_parts(states_c.handle, nblk * k_dim * v_dim),
                BufferArg::from_raw_parts(d_v_new_fresh_c.handle, nblk * c * v_dim),
                BufferArg::from_raw_parts(d_k_bptt_c2.handle, nblk * c * k_dim),
                BufferArg::from_raw_parts(d_e_bptt_c2.handle, nblk * c * k_dim),
                BufferArg::from_raw_parts(d_e_last_c2.handle, nblk * k_dim),
                BufferArg::from_raw_parts(d_q_c.handle, nblk * c * k_dim),
                BufferArg::from_raw_parts(d_k_c.handle, nblk * c * k_dim),
                BufferArg::from_raw_parts(d_b_c.handle, nblk * c * k_dim),
                BufferArg::from_raw_parts(d_g_c.handle, nblk * c * k_dim),
                BufferArg::from_raw_parts(d_v_c.handle, nblk * c * v_dim),
                BufferArg::from_raw_parts(d_w_c.handle, nblk * c * v_dim),
                scale as f32,
                c as u32,
                k_dim as u32,
                v_dim as u32,
            );
        }

        // d_s: the state adjoint at chunk 0 (BK2's first-chunk slice)
        let d_s_5 = d_s_flat.clone().reshape([batch, heads, nt, k_dim, v_dim]);
        let d_s_final = d_s_5
            .slice([0..batch, 0..heads, 0..1, 0..k_dim, 0..v_dim])
            .reshape([batch, heads, k_dim, v_dim]);
        let _ = d_s_out_c;

        let _ = d_v_new;
        Some(FusedBackwardOutput {
            d_q,
            d_k,
            d_v,
            d_g,
            d_b,
            d_w,
            d_s: d_s_final,
        })
    }
}
