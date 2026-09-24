//! Fused token-choice MoE forward+backward on CUDA (T2.2).
//!
//! One kernel computes the per-token rank-`r` expert projection WITHOUT ever
//! materializing `[B, k*r, in]` / `[B, k*r, out]` (the OOM source), and one
//! kernel runs the exact backward (scatter-add into the masters via float
//! atomics, mirrors burn-msa / `fused.rs`). The router itself (projections,
//! argmax/argtopk, softmax, gates) is also one kernel per direction
//! (`moe_router_kernel` + `moe_router_bwd_kernel`), killing the ~11 tensor
//! graph nodes the tensor-path router adds per module; the tensor router stays
//! as the fallback.
//!
//! Math (identical to `SpectralMoE::forward` token-choice path). For token
//! `t`, expert `j`, column `col = idx[t,j]*r + jj`:
//!   `p[t,col] = Σ_i x[t,i]·u[i,col]`
//!   `y[t,o] = Σ_{j,jj} gates[t,j]·s[col]·p[t,col]·v[o,col]`
//! Backward (with `w[t,col] = Σ_o d_out[t,o]·v[o,col]`):
//!   `d_x[t,i] = Σ_{col} gates·s·u[i,col]·w`
//!   `d_u[i,col] += gates·s·x[t,i]·w`   (scatter-add, atomics)
//!   `d_v[o,col] += gates·s·p·d_out[t,o]` (scatter-add, atomics)
//!   `d_s[col] += gates·p·w`            (scatter-add, atomics)
//!   `d_gates[t,j] = Σ_jj s·p·w`
//!
//! `u`/`v` are the already-ternarized masters (`ste_ternary` pre-pass, so the
//! STE flows back to the raw masters through the tensor graph); `idx` are
//! global expert ids (`cluster*E + pos`), no gradient.

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};

use burn::backend::{Backend, DispatchKindConversion};
use burn::tensor::{activation, Device, DispatchTensor, Int, Tensor};
use burn_autodiff::checkpoint::base::Checkpointer;
use burn_autodiff::checkpoint::strategy::NoCheckpointing;
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};
use burn_autodiff::Autodiff;
use burn_cubecl::tensor::CubeTensor;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;

type CB = burn_cubecl::CubeBackend;
type CAd = Autodiff<CB>;

static MOE_FWD_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Number of fused MoE forward launches since process start (tests assert the
/// fused path engaged instead of silently falling back to the tensor path).
pub fn moe_fused_forward_count() -> usize {
    MOE_FWD_COUNT.load(Ordering::Relaxed)
}

/// Force a dense row-major tensor (see burn-msa `sparse_kernel::dense4`).
/// cubecl pitches 2D rows to `next_pow2(width_bytes).clamp(16, 512)`; the
/// fused kernels index flat row-major, so every operand must be exactly dense.
fn dense<const D: usize, K>(t: Tensor<D, K>) -> Tensor<D, K>
where
    K: burn::tensor::kind::Basic,
{
    let dims = t.dims();
    let n = dims.iter().product::<usize>();
    t.reshape::<1, _>([n]).reshape::<D, _>(dims)
}

/// Dense (never row-pitched) output allocation: a fresh 1D buffer is
/// contiguous and the reshape back is a lazy dense view.
fn empty_dense<const D: usize>(dims: [usize; D], device: &Device) -> Tensor<D> {
    let n = dims.iter().product::<usize>();
    Tensor::<1>::empty([n], device).reshape::<D, _>(dims)
}

/// Dense zero-initialized allocation (the backward kernels scatter-add, so
/// the buffers must start at zero).
fn zeros_dense<const D: usize>(dims: [usize; D], device: &Device) -> Tensor<D> {
    let n = dims.iter().product::<usize>();
    Tensor::<1>::zeros([n], device).reshape::<D, _>(dims)
}

fn cube_of2(t: &Tensor<2>) -> Option<CubeTensor> {
    let prim = t.clone().try_into_primitive::<CB>().ok()?;
    let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
    Some(c.clone())
}

fn cube_of1(t: &Tensor<1>) -> Option<CubeTensor> {
    let prim = t.clone().try_into_primitive::<CB>().ok()?;
    let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
    Some(c.clone())
}

fn cube_int2(t: &Tensor<2, Int>) -> Option<CubeTensor> {
    let prim = t.clone().try_into_primitive::<CB>().ok()?;
    let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
    Some(c.clone())
}

/// Forward: one warp per token (32-lane cubes). Stage 1 splits the `k*r`
/// projection columns across the lanes (lane `a` owns columns `a`, `a+32`,
/// ...), each accumulated serially over `dim_in` in the same per-column
/// order as the single-thread version, so the results are bit-identical.
/// The projections are staged in shared memory for stage 2, which processes
/// `dim_out` in 32-row chunks: the `v` block of each chunk is staged in
/// shared memory with coalesced loads, and lane `a` computes row `a` of the
/// chunk. Per-token work is `k*r*(dim_in + dim_out)` MACs but spread over
/// 32 lanes, so the serial load chain of the old one-thread-per-token
/// kernel (latency-bound on the `v` reads) is gone.
#[cube(launch_unchecked)]
fn moe_fwd_kernel<F: Float>(
    x: &[F],     // [b, dim_in] dense
    u: &[F],     // [dim_in, m] dense ternary
    v: &[F],     // [dim_out, m] dense ternary
    s: &[F],     // [m]
    idx: &[i32], // [b, k] global expert ids
    gates: &[F], // [b, k]
    z: &mut [F], // [b, k*r] projection (checkpoint for backward)
    y: &mut [F], // [b, dim_out] dense
    b: u32,
    dim_in: u32,
    m: u32,
    dim_out: u32,
    #[comptime] k: u32,
    #[comptime] r: u32,
) {
    let t = CUBE_POS_X as usize;
    let a = UNIT_POS_X as usize;
    let dim_in = dim_in as usize;
    let m = m as usize;
    let dim_out = dim_out as usize;
    let k = k as usize;
    let r = r as usize;
    let kk = k * r;

    if (t as u32) < b {
        let mut shared_p = Shared::new_slice(kk);
        // stage 1: p[t,col] = x[t] . u[:, col], columns split across lanes;
        // lane `a` serially accumulates columns `a`, `a+32`, ... over the
        // full `dim_in` (identical per-column FMA order to the tensor path)
        let mut c = a;
        while c < kk {
            let j = c / r;
            let jj = c % r;
            let col = idx[t * k + j] as usize * r + jj;
            let mut acc = F::new(0.0_f32);
            let mut i = 0usize;
            while i < dim_in {
                acc += x[t * dim_in + i] * u[i * m + col];
                i += 1usize;
            }
            shared_p[c] = acc;
            z[t * kk + c] = acc;
            c += 32usize;
        }
        sync_cube();
        // stage 2: y[t,o] = sum over (j,jj) of g*s*p*v, rows split across
        // lanes. The `v` block for a 32-row chunk is staged in shared memory
        // with coalesced loads (lanes span the `r` contiguous columns of a
        // row) and conflict-free reads (rows padded to an odd stride); the
        // direct per-lane strided reads cost ~16x more L1 line fetches.
        // The per-row (j, jj) FMA order is unchanged, so results stay
        // bit-identical to the single-thread version.
        let mut shared_v = Shared::new_slice(32 * (kk | 1));
        let mut o0 = 0usize;
        while o0 < dim_out {
            // load: lane `a` fills rows o0 + a/r + q*(32/r) (all 32 chunk
            // rows for any r in 1..16; duplicate writes are same-valued)
            #[unroll]
            for j in 0..k {
                let e = idx[t * k + j] as usize;
                let mut row = o0 + a / r;
                while row < o0 + 32usize && row < dim_out {
                    shared_v[(row - o0) * (kk | 1) + j * r + (a % r)] =
                        v[row * m + e * r + (a % r)];
                    row += 32usize / r;
                }
            }
            sync_cube();
            let o = o0 + a;
            if o < dim_out {
                let mut acc = F::new(0.0_f32);
                #[unroll]
                for j in 0..k {
                    let e = idx[t * k + j] as usize;
                    let g = gates[t * k + j];
                    #[unroll]
                    for jj in 0..r {
                        acc += g
                            * s[e * r + jj]
                            * shared_p[j * r + jj]
                            * shared_v[(o - o0) * (kk | 1) + j * r + jj];
                    }
                }
                y[t * dim_out + o] = acc;
            }
            sync_cube();
            o0 += 32usize;
        }
    }
}

/// Backward: one warp per token (32-lane cubes), mirroring the forward's
/// tiling. Stage 1 walks `dim_out` in 32-row chunks: the `v` block of each
/// chunk is staged in shared memory with coalesced loads (row stride `kk|1`
/// odd, conflict-free), lane `a` computes row `o0+a` and accumulates its
/// partial `w[t,col] = Σ_{o≡a} d_out[t,o]·v[o,col]` in registers plus the
/// `d_v` scatter-add. Stage 2 sums the 32 partials through shared memory into
/// the full `w`. Stage 3 walks `dim_in` in 32-row chunks with the `u` block
/// staged the same way, lane `a` computing `d_x` row `i0+a` and the `d_u`
/// scatter-add. Stage 4 folds `d_s` (columns split across lanes) and
/// `d_gates` (one expert per lane). Scatter-adds stay float atomics into the
/// zeroed buffers (same non-determinism burn-msa tolerates).
#[allow(clippy::unnecessary_cast)]
#[cube(launch_unchecked)]
fn moe_bwd_kernel<F: Float>(
    d_out: &[F],           // [b, dim_out] dense
    x: &[F],               // [b, dim_in] dense
    u: &[F],               // [dim_in, m] dense
    v: &[F],               // [dim_out, m] dense
    s: &[F],               // [m]
    idx: &[i32],           // [b, k]
    gates: &[F],           // [b, k]
    z: &[F],               // [b, k*r]
    d_x: &mut [F],         // [b, dim_in]
    d_u: &mut [Atomic<F>], // [dim_in, m] zeroed
    d_v: &mut [Atomic<F>], // [dim_out, m] zeroed
    d_s: &mut [Atomic<F>], // [m] zeroed
    d_gates: &mut [F],     // [b, k]
    b: u32,
    dim_in: u32,
    m: u32,
    dim_out: u32,
    #[comptime] k: u32,
    #[comptime] r: u32,
) {
    let t = CUBE_POS_X as usize;
    let a = UNIT_POS_X as usize;
    let dim_in = dim_in as usize;
    let m = m as usize;
    let dim_out = dim_out as usize;
    let k = k as usize;
    let r = r as usize;
    let kk = k * r;

    if (t as u32) < b {
        // stage 1: w-partial[t,col] = Σ_{o≡a (mod 32)} d_out[t,o]·v[o,col],
        // plus the d_v scatter-add, in 32-row chunks of dim_out with the `v`
        // block staged in shared (exactly the forward kernel's stage-2 load)
        let mut w = Array::<F>::new(kk);
        #[unroll]
        for c in 0..kk {
            w[c] = F::new(0.0_f32);
        }
        let mut shared_v = Shared::new_slice(32 * (kk | 1));
        let mut o0 = 0usize;
        while o0 < dim_out {
            #[unroll]
            for j in 0..k {
                let e = idx[t * k + j] as usize;
                let mut row = o0 + a / r;
                while row < o0 + 32usize && row < dim_out {
                    shared_v[(row - o0) * (kk | 1) + j * r + (a % r)] =
                        v[row * m + e * r + (a % r)];
                    row += 32usize / r;
                }
            }
            sync_cube();
            let o = o0 + a;
            if o < dim_out {
                let dvo = d_out[t * dim_out + o];
                #[unroll]
                for j in 0..k {
                    let e = idx[t * k + j] as usize;
                    let g = gates[t * k + j];
                    #[unroll]
                    for jj in 0..r {
                        let col = e * r + jj;
                        w[j * r + jj] += dvo * shared_v[(o - o0) * (kk | 1) + j * r + jj];
                        d_v[o * m + col].fetch_add(g * s[col] * z[t * kk + j * r + jj] * dvo);
                    }
                }
            }
            sync_cube();
            o0 += 32usize;
        }
        // stage 2: full w = Σ_lane partials. The 32 partials go to shared
        // slots [lane*kk .. lane*kk+kk), then lane `a` reduces column groups
        // `a`, `a+32`, ... (kk <= 64 -> at most two columns per lane; each
        // final slot is written by exactly one lane, its own reader)
        let mut shared_w = Shared::new_slice(32 * kk + kk);
        #[unroll]
        for c in 0..kk {
            shared_w[a * kk + c] = w[c];
        }
        sync_cube();
        let mut c = a;
        while c < kk {
            let mut acc = F::new(0.0_f32);
            #[unroll]
            for aa in 0..32 {
                acc += shared_w[aa * kk + c];
            }
            shared_w[32 * kk + c] = acc;
            c += 32usize;
        }
        sync_cube();
        // stage 3: d_x[t,i] and the d_u scatter-add over dim_in in 32-row
        // chunks with the `u` block staged like the forward's `v` block
        let mut shared_u = Shared::new_slice(32 * (kk | 1));
        let mut i0 = 0usize;
        while i0 < dim_in {
            #[unroll]
            for j in 0..k {
                let e = idx[t * k + j] as usize;
                let mut row = i0 + a / r;
                while row < i0 + 32usize && row < dim_in {
                    shared_u[(row - i0) * (kk | 1) + j * r + (a % r)] =
                        u[row * m + e * r + (a % r)];
                    row += 32usize / r;
                }
            }
            sync_cube();
            let i = i0 + a;
            if i < dim_in {
                let xv = x[t * dim_in + i];
                let mut acc = F::new(0.0_f32);
                #[unroll]
                for j in 0..k {
                    let e = idx[t * k + j] as usize;
                    let g = gates[t * k + j];
                    #[unroll]
                    for jj in 0..r {
                        let col = e * r + jj;
                        acc += g
                            * s[col]
                            * shared_u[(i - i0) * (kk | 1) + j * r + jj]
                            * shared_w[32 * kk + j * r + jj];
                        d_u[i * m + col]
                            .fetch_add(g * s[col] * xv * shared_w[32 * kk + j * r + jj]);
                    }
                }
                d_x[t * dim_in + i] = acc;
            }
            sync_cube();
            i0 += 32usize;
        }
        // stage 4: d_s[col] += g·p·w (columns split across lanes) and
        // d_gates[t,j] = Σ_jj s·p·w (expert j to lane j, k <= 16)
        let mut c = a;
        while c < kk {
            let j = c / r;
            let e = idx[t * k + j] as usize;
            let col = e * r + (c % r);
            d_s[col].fetch_add(gates[t * k + j] * z[t * kk + c] * shared_w[32 * kk + c]);
            c += 32usize;
        }
        sync_cube();
        if a < k {
            let e = idx[t * k + a] as usize;
            let mut acc = F::new(0.0_f32);
            #[unroll]
            for jj in 0..r {
                let col = e * r + jj;
                acc += s[col] * z[t * kk + a * r + jj] * shared_w[32 * kk + a * r + jj];
            }
            d_gates[t * k + a] = acc;
        }
    }
}

/// Fused router forward: one launch for the whole router. One warp per
/// token (lane `a` computes `h[a]`, staged through shared memory), which
/// keeps the register pressure low and the occupancy high: the previous
/// one-thread-per-token version left ~5% of the GPU busy on the serial
/// `in·p` MAC chain. Computes `c = h·W_cluster` with argmax over `C` (lowest
/// index wins ties) and the cluster gate `softmax(c)[c_idx]`; then
/// `e = h·W_expert` with top-`k` over `E` (descending values, ties to the
/// lowest position — the exact cubek-reduce total order), softmax over ALL
/// `E` positions, and `g = softmax(e)[pos]·cluster_gate`,
/// `idx = c_idx·E + pos` — op-for-op `SpectralMoE::forward`.
#[cube(launch_unchecked)]
fn moe_router_kernel<F: Float>(
    x: &[F],           // [b, in] dense
    w_proj: &[F],      // [in, p] dense
    w_cluster: &[F],   // [p, C] dense
    w_expert: &[F],    // [p, E] dense
    c_idx: &mut [i32], // [b]
    idx: &mut [i32],   // [b, k] global expert ids
    gates: &mut [F],   // [b, k]
    b: u32,
    in_features: u32,
    #[comptime] p: u32,
    #[comptime] c: u32,
    #[comptime] e: u32,
    #[comptime] k: u32,
) {
    let token = CUBE_POS_X as usize;
    let a = UNIT_POS_X as usize;
    let in_features = in_features as usize;
    let p = p as usize;
    let c = c as usize;
    let e = e as usize;
    let k = k as usize;
    if (token as u32) < b {
        // h[a] = x[token] . w_proj[:, a]; lane a owns element a
        let mut shared_h = Shared::new_slice(p);
        let mut acc = F::new(0.0_f32);
        let mut j = 0usize;
        while j < in_features {
            acc += x[token * in_features + j] * w_proj[j * p + a];
            j += 1usize;
        }
        shared_h[a] = acc;
        sync_cube();

        // c = h . w_cluster : [C]; argmax keeps the lowest index on ties
        let mut c_scores = Array::<F>::new(c);
        let mut c_best = F::min_value();
        let mut c_best_idx = 0i32;
        #[unroll]
        for i in 0..c {
            let mut s = F::new(0.0_f32);
            #[unroll]
            for aa in 0..p {
                s += shared_h[aa] * w_cluster[aa * c + i];
            }
            c_scores[i] = s;
            if s > c_best {
                c_best = s;
                c_best_idx = i as i32;
            }
        }
        let mut c_sum = F::new(0.0_f32);
        #[unroll]
        for i in 0..c {
            c_sum += (c_scores[i] - c_best).exp();
        }
        let cw = (c_scores[c_best_idx as usize] - c_best).exp() / c_sum;

        // e = h . w_expert : [E]; top-k via the k selection rounds over the
        // (value desc, index asc) total order (identical to the cubek
        // ArgTopK order)
        let mut e_scores = Array::<F>::new(e);
        #[unroll]
        for i in 0..e {
            let mut s = F::new(0.0_f32);
            #[unroll]
            for aa in 0..p {
                s += shared_h[aa] * w_expert[aa * e + i];
            }
            e_scores[i] = s;
        }
        let mut top_vals = Array::<F>::new(k);
        let mut top_pos = Array::<i32>::new(k);
        #[unroll]
        for s in 0..k {
            let mut best_v = F::min_value();
            let mut best_p = 0i32;
            let mut found = false;
            #[unroll]
            for i in 0..e {
                let mut already = false;
                #[unroll]
                for j in 0..s {
                    if top_pos[j] == (i as i32) {
                        already = true;
                    }
                }
                if !already
                    && (!found
                        || e_scores[i] > best_v
                        || (e_scores[i] == best_v && (i as i32) < best_p))
                {
                    best_v = e_scores[i];
                    best_p = i as i32;
                    found = true;
                }
            }
            top_vals[s] = best_v;
            top_pos[s] = best_p;
        }
        let e_max = top_vals[0];
        // softmax over ALL E positions, then gather the top-k (the tensor
        // path does softmax(e, 1).gather(1, pos))
        let mut e_sum = F::new(0.0_f32);
        #[unroll]
        for i in 0..e {
            e_sum += (e_scores[i] - e_max).exp();
        }

        if a == 0 {
            c_idx[token] = c_best_idx;
            #[unroll]
            for s in 0..k {
                let pos = top_pos[s] as usize;
                idx[token * k + s] = (c_best_idx as usize * e + pos) as i32;
                gates[token * k + s] = (e_scores[pos] - e_max).exp() / e_sum * cw;
            }
        }
    }
}

/// Fused router backward: one thread per token. Recomputes the forward
/// scores/softmaxes from the checkpointed inputs (deterministic, so the
/// values match the forward bit-for-bit), pushes `d_gates` through the two
/// softmaxes with the exact formulas, and scatters the weight gradients with
/// float atomics (mirrors `moe_bwd_kernel`):
///   `d_c[i] = dcw·smc[i]·(δ_{i,c_idx} − smc[c_idx])`, `dcw = Σ_s dg[s]·sme[pos[s]]`
///   `d_e[i] = sme[i]·(dsme[i] − Σ_j dsme[j]·sme[j])`, `dsme[pos[s]] = dg[s]·cw`
///   `d_h = d_c·W_c + d_e·W_e`; `d_x = d_h·W_p`; `dW = x/h/d_h outer products`
#[allow(clippy::unnecessary_cast)]
#[cube(launch_unchecked)]
fn moe_router_bwd_kernel<F: Float>(
    x: &[F],                       // [b, in] dense
    w_proj: &[F],                  // [in, p] dense
    w_cluster: &[F],               // [p, C] dense
    w_expert: &[F],                // [p, E] dense
    d_gates: &[F],                 // [b, k]
    h_buf: &mut [F],               // [b, p] scratch: h (the atomics re-read it from here)
    d_h_buf: &mut [F],             // [b, p] scratch: dh (streamed by the second kernel)
    d_w_cluster: &mut [Atomic<F>], // [p, C] zeroed
    d_w_expert: &mut [Atomic<F>],  // [p, E] zeroed
    b: u32,
    in_features: u32,
    #[comptime] p: u32,
    #[comptime] c: u32,
    #[comptime] e: u32,
    #[comptime] k: u32,
) {
    let token = CUBE_POS_X as usize;
    let a = UNIT_POS_X as usize;
    let in_features = in_features as usize;
    let p = p as usize;
    let c = c as usize;
    let e = e as usize;
    let k = k as usize;
    if (token as u32) < b {
        // recompute h (identical math to the forward kernel), staged through
        // shared so every lane re-derives the scores redundantly
        let mut shared_h = Shared::new_slice(p);
        let mut acc = F::new(0.0_f32);
        let mut j = 0usize;
        while j < in_features {
            acc += x[token * in_features + j] * w_proj[j * p + a];
            j += 1usize;
        }
        shared_h[a] = acc;
        h_buf[token * p + a] = acc;
        sync_cube();

        let mut c_scores = Array::<F>::new(c);
        let mut c_best = F::min_value();
        let mut c_best_idx = 0i32;
        #[unroll]
        for i in 0..c {
            let mut s = F::new(0.0_f32);
            #[unroll]
            for aa in 0..p {
                s += shared_h[aa] * w_cluster[aa * c + i];
            }
            c_scores[i] = s;
            if s > c_best {
                c_best = s;
                c_best_idx = i as i32;
            }
        }
        let mut c_sum = F::new(0.0_f32);
        #[unroll]
        for i in 0..c {
            c_sum += (c_scores[i] - c_best).exp();
        }

        let mut e_scores = Array::<F>::new(e);
        #[unroll]
        for i in 0..e {
            let mut s = F::new(0.0_f32);
            #[unroll]
            for aa in 0..p {
                s += shared_h[aa] * w_expert[aa * e + i];
            }
            e_scores[i] = s;
        }
        let mut top_vals = Array::<F>::new(k);
        let mut top_pos = Array::<i32>::new(k);
        #[unroll]
        for s in 0..k {
            let mut best_v = F::min_value();
            let mut best_p = 0i32;
            let mut found = false;
            #[unroll]
            for i in 0..e {
                let mut already = false;
                #[unroll]
                for j in 0..s {
                    if top_pos[j] == (i as i32) {
                        already = true;
                    }
                }
                if !already
                    && (!found
                        || e_scores[i] > best_v
                        || (e_scores[i] == best_v && (i as i32) < best_p))
                {
                    best_v = e_scores[i];
                    best_p = i as i32;
                    found = true;
                }
            }
            top_vals[s] = best_v;
            top_pos[s] = best_p;
        }
        let e_max = top_vals[0];
        let mut e_sum = F::new(0.0_f32);
        #[unroll]
        for i in 0..e {
            e_sum += (e_scores[i] - e_max).exp();
        }

        // cluster path: dcw = Σ_s dg[s]·sme[top_pos[s]]
        let smc_ci = (c_scores[c_best_idx as usize] - c_best).exp() / c_sum;
        let mut dcw = F::new(0.0_f32);
        #[unroll]
        for s in 0..k {
            let pos = top_pos[s] as usize;
            dcw += d_gates[token * k + s] * (e_scores[pos] - e_max).exp() / e_sum;
        }
        // d_c[i] = dcw·smc_i·(δ_{i,c_idx} − smc_ci): fold the delta into a
        // post-add to avoid a select on the runtime index comparison
        let mut d_c = Array::<F>::new(c);
        #[unroll]
        for i in 0..c {
            let smc_i = (c_scores[i] - c_best).exp() / c_sum;
            d_c[i] = -dcw * smc_i * smc_ci;
        }
        d_c[c_best_idx as usize] += dcw * smc_ci;

        // expert path: dsme[pos] = dg·cw; d_e[i] = sme_i·(dsme_i − e_dot).
        // dsme is only nonzero at the k picked positions, so re-derive it
        // from top_pos instead of keeping an [e] array alive
        let cw = smc_ci;
        let mut e_dot = F::new(0.0_f32);
        #[unroll]
        for s in 0..k {
            let pos = top_pos[s] as usize;
            e_dot += d_gates[token * k + s] * cw * (e_scores[pos] - e_max).exp() / e_sum;
        }
        let mut d_e = Array::<F>::new(e);
        #[unroll]
        for i in 0..e {
            let sme_i = (e_scores[i] - e_max).exp() / e_sum;
            let mut dsme_i = F::new(0.0_f32);
            #[unroll]
            for s in 0..k {
                if top_pos[s] == (i as i32) {
                    dsme_i = d_gates[token * k + s] * cw;
                }
            }
            d_e[i] = sme_i * (dsme_i - e_dot);
        }

        // lane a scatters row a of the weight grads (h[a] is its own value)
        #[unroll]
        for i in 0..c {
            d_w_cluster[a * c + i].fetch_add(d_c[i] * shared_h[a]);
        }
        #[unroll]
        for i in 0..e {
            d_w_expert[a * e + i].fetch_add(d_e[i] * shared_h[a]);
        }
        // dh[a] = Σ_i d_c[i]·w_cluster[a·c+i] + Σ_i d_e[i]·w_expert[a·e+i]
        let mut dh_a = F::new(0.0_f32);
        #[unroll]
        for i in 0..c {
            dh_a += d_c[i] * w_cluster[a * c + i];
        }
        #[unroll]
        for i in 0..e {
            dh_a += d_e[i] * w_expert[a * e + i];
        }
        d_h_buf[token * p + a] = dh_a;
    }
}

/// Router backward, part 2: streams the staged `dh [b, p]` over `w_proj` to
/// produce `d_x [b, in]` and scatter `d_w_proj` (atomics). A separate kernel
/// so part 1 does not hold `dh` in registers across the whole `in`-loop (the
/// unrolled [p] arrays would spill to local memory).
#[allow(clippy::unnecessary_cast)]
#[cube(launch_unchecked)]
fn moe_router_bwd2_kernel<F: Float>(
    w_proj: &[F],  // [in, p] dense
    d_h: &[F],     // [b, p] scratch from part 1
    d_x: &mut [F], // [b, in]
    b: u32,
    in_features: u32,
    #[comptime] p: u32,
) {
    let token = CUBE_POS_X as usize;
    let a = UNIT_POS_X as usize;
    let in_features = in_features as usize;
    let p = p as usize;
    if (token as u32) < b {
        let mut shared_dh = Shared::new_slice(p);
        shared_dh[a] = d_h[token * p + a];
        sync_cube();
        // lane a computes columns j ≡ a (mod p): 512/32 = 16 per lane
        let mut j = a;
        while j < in_features {
            let mut acc = F::new(0.0_f32);
            #[unroll]
            for aa in 0..p {
                acc += shared_dh[aa] * w_proj[j * p + aa];
            }
            d_x[token * in_features + j] = acc;
            j += p;
        }
    }
}

/// Router backward, part 3: `d_w_proj [in, p]` scatter-add. One thread per
/// (j, a) cell accumulates `Σ_t d_h[t,a]·x[t,j]` over the tokens in registers
/// and performs a SINGLE atomic per cell — the per-token loop of part 2 would
/// make every token hammer the same 32 cells per j-step (2048-way atomic
/// contention, ~10ms per call at b=2048; this is ~contention-free). Threads
/// with consecutive `a` read consecutive `d_h` (coalesced) and the same `x`
/// column (broadcast).
#[allow(clippy::unnecessary_cast)]
#[cube(launch_unchecked)]
fn moe_router_bwd3_kernel<F: Float>(
    x: &[F],                    // [b, in] dense
    d_h: &[F],                  // [b, p] scratch from part 1
    d_w_proj: &mut [Atomic<F>], // [in, p] zeroed
    b: u32,
    in_features: u32,
    #[comptime] p: u32,
) {
    let cell = CUBE_POS_X as usize * 64usize + UNIT_POS_X as usize;
    let p = p as usize;
    let in_features = in_features as usize;
    let cells = in_features * p;
    if cell < cells {
        let j = cell / p;
        let a = cell % p;
        let mut acc = F::new(0.0_f32);
        let mut t = 0usize;
        while t < (b as usize) {
            acc += d_h[t * p + a] * x[t * in_features + j];
            t += 1usize;
        }
        d_w_proj[cell].fetch_add(acc);
    }
}

/// Checkpointed router state: the dense inputs (captured at forward time so
/// the backward recomputes the scores from the exact values).
#[derive(Debug, Clone)]
struct RoutState {
    x: Tensor<2>,
    w_proj: Tensor<2>,
    w_cluster: Tensor<2>,
    w_expert: Tensor<2>,
    b: usize,
    in_features: usize,
    p: usize,
    c: usize,
    e: usize,
    k: usize,
}

#[derive(Debug)]
struct RoutFwdOp;

impl<B: Backend> Backward<B, 4> for RoutFwdOp
where
    DispatchTensor: DispatchKindConversion<B>,
{
    type State = RoutState;

    fn backward(
        self,
        ops: Ops<Self::State, 4>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let st = &ops.state;
        let d_gates = Tensor::<2>::from_primitive::<B>(grads.consume::<B>(&ops.node));

        #[cfg(feature = "cuda")]
        {
            if std::any::TypeId::of::<B>() == std::any::TypeId::of::<CB>() {
                if let Some((dx, dwp, dwc, dwe)) = router_backward_cuda(st, &d_gates) {
                    if let Some(node) = ops.parents[0].clone() {
                        grads.register::<B>(node.id, dx.try_into_primitive::<B>().unwrap());
                    }
                    if let Some(node) = ops.parents[1].clone() {
                        grads.register::<B>(node.id, dwp.try_into_primitive::<B>().unwrap());
                    }
                    if let Some(node) = ops.parents[2].clone() {
                        grads.register::<B>(node.id, dwc.try_into_primitive::<B>().unwrap());
                    }
                    if let Some(node) = ops.parents[3].clone() {
                        grads.register::<B>(node.id, dwe.try_into_primitive::<B>().unwrap());
                    }
                    return;
                }
            }
        }

        let (dx, dwp, dwc, dwe) = router_backward_tensor(st, &d_gates);
        if let Some(node) = ops.parents[0].clone() {
            grads.register::<B>(node.id, dx.try_into_primitive::<B>().unwrap());
        }
        if let Some(node) = ops.parents[1].clone() {
            grads.register::<B>(node.id, dwp.try_into_primitive::<B>().unwrap());
        }
        if let Some(node) = ops.parents[2].clone() {
            grads.register::<B>(node.id, dwc.try_into_primitive::<B>().unwrap());
        }
        if let Some(node) = ops.parents[3].clone() {
            grads.register::<B>(node.id, dwe.try_into_primitive::<B>().unwrap());
        }
    }
}

/// Fused router backward on the bare CUDA backend.
#[allow(clippy::too_many_arguments)]
fn router_backward_cuda(
    st: &RoutState,
    d_gates: &Tensor<2>,
) -> Option<(Tensor<2>, Tensor<2>, Tensor<2>, Tensor<2>)> {
    let xc = cube_of2(&st.x)?;
    let wpc = cube_of2(&st.w_proj)?;
    let wcc = cube_of2(&st.w_cluster)?;
    let wec = cube_of2(&st.w_expert)?;
    // d_gates arrives from the autodiff graph: a fresh 2D buffer may be
    // row-pitched, and the kernel reads it flat, so force exact density
    let d_gates = dense(d_gates.clone());
    let dgc = cube_of2(&d_gates)?;
    let client = xc.client.clone();

    let dev = st.x.device();
    let dx = empty_dense([st.b, st.in_features], &dev);
    let dwp = zeros_dense([st.in_features, st.p], &dev);
    let dwc = zeros_dense([st.p, st.c], &dev);
    let dwe = zeros_dense([st.p, st.e], &dev);
    let h_buf = empty_dense([st.b, st.p], &dev);
    let d_h_buf = empty_dense([st.b, st.p], &dev);
    let dxc = cube_of2(&dx)?;
    let dwpc = cube_of2(&dwp)?;
    let dwcc = cube_of2(&dwc)?;
    let dwec = cube_of2(&dwe)?;
    let h_bufc = cube_of2(&h_buf)?;
    let d_h_bufc = cube_of2(&d_h_buf)?;

    unsafe {
        moe_router_bwd_kernel::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(st.b as u32, 1, 1),
            CubeDim::new_3d(32, 1, 1),
            BufferArg::from_raw_parts(xc.handle.clone(), st.b * st.in_features),
            BufferArg::from_raw_parts(wpc.handle.clone(), st.in_features * st.p),
            BufferArg::from_raw_parts(wcc.handle.clone(), st.p * st.c),
            BufferArg::from_raw_parts(wec.handle.clone(), st.p * st.e),
            BufferArg::from_raw_parts(dgc.handle.clone(), st.b * st.k),
            BufferArg::from_raw_parts(h_bufc.handle.clone(), st.b * st.p),
            BufferArg::from_raw_parts(d_h_bufc.handle.clone(), st.b * st.p),
            BufferArg::from_raw_parts(dwcc.handle.clone(), st.p * st.c),
            BufferArg::from_raw_parts(dwec.handle.clone(), st.p * st.e),
            st.b as u32,
            st.in_features as u32,
            st.p as u32,
            st.c as u32,
            st.e as u32,
            st.k as u32,
        );
        moe_router_bwd2_kernel::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(st.b as u32, 1, 1),
            CubeDim::new_3d(32, 1, 1),
            BufferArg::from_raw_parts(wpc.handle.clone(), st.in_features * st.p),
            BufferArg::from_raw_parts(d_h_bufc.handle.clone(), st.b * st.p),
            BufferArg::from_raw_parts(dxc.handle.clone(), st.b * st.in_features),
            st.b as u32,
            st.in_features as u32,
            st.p as u32,
        );
        moe_router_bwd3_kernel::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(((st.in_features * st.p) as u32).div_ceil(64), 1, 1),
            CubeDim::new_3d(64, 1, 1),
            BufferArg::from_raw_parts(xc.handle.clone(), st.b * st.in_features),
            BufferArg::from_raw_parts(d_h_bufc.handle.clone(), st.b * st.p),
            BufferArg::from_raw_parts(dwpc.handle.clone(), st.in_features * st.p),
            st.b as u32,
            st.in_features as u32,
            st.p as u32,
        );
    }
    Some((dx, dwp, dwc, dwe))
}

/// Tensor-path router backward (reference; dead in practice since the op is
/// only built for `Autodiff<CubeBackend>`).
fn router_backward_tensor(
    st: &RoutState,
    dg: &Tensor<2>,
) -> (Tensor<2>, Tensor<2>, Tensor<2>, Tensor<2>) {
    let x = &st.x;
    let wp = &st.w_proj;
    let wc = &st.w_cluster;
    let we = &st.w_expert;
    let device = x.device();
    let h = x.clone().matmul(wp.clone()); // [B, p]
    let c_l = h.clone().matmul(wc.clone()); // [B, C]
    let e_l = h.clone().matmul(we.clone()); // [B, E]
    let c_idx = c_l.clone().argmax(1); // [B, 1]
    let smc = activation::softmax(c_l, 1); // [B, C]
    let cw = smc.clone().gather(1, c_idx.clone()); // [B, 1]
    let sme = activation::softmax(e_l.clone(), 1); // [B, E]
                                                   // k <= 16 is enforced by the kernel limits, so this is the single-round
                                                   // argtopk of `topk_indices_generic`
    let pos = e_l.clone().argtopk(st.k, 1); // [B, k]
    let b = st.b;

    // d_smc scattered at c_idx; softmax backward: d_c = smc·(d_smc − Σ d_smc·smc)
    let dcw = dg
        .clone()
        .mul(sme.clone().gather(1, pos.clone()))
        .sum_dim(1); // [B, 1]
    let onehot_c = Tensor::<1, Int>::arange(0..st.c as i64, &device)
        .unsqueeze_dim::<2>(0)
        .expand([b, st.c])
        .equal(c_idx.clone())
        .float();
    let d_smc = onehot_c.mul(dcw); // [B, C]
    let sum_c = d_smc.clone().mul(smc.clone()).sum_dim(1); // [B, 1]
    let d_c = smc.clone().mul(d_smc.sub(sum_c)); // [B, C]

    // d_sme scattered at pos; full softmax backward
    let onehot_e = Tensor::<1, Int>::arange(0..st.e as i64, &device)
        .unsqueeze_dim::<2>(0)
        .unsqueeze_dim::<3>(1)
        .expand([b, st.k, st.e])
        .equal(pos.clone().unsqueeze_dim::<3>(2))
        .float(); // [B, k, E]
    let d_sme = onehot_e
        .mul(dg.clone().mul(cw).unsqueeze_dim::<3>(2))
        .sum_dim(1)
        .squeeze_dim::<2>(1); // [B, E]
    let sum_e = d_sme.clone().mul(sme.clone()).sum_dim(1); // [B, 1]
    let d_e = sme.clone().mul(d_sme.sub(sum_e)); // [B, E]

    let d_h = d_c
        .clone()
        .matmul(wc.clone().transpose())
        .add(d_e.clone().matmul(we.clone().transpose())); // [B, p]
    let dx = d_h.clone().matmul(wp.clone().transpose());
    let dwp = d_h.clone().transpose().matmul(x.clone()).transpose(); // [in, p]
    let dwc = d_c.transpose().matmul(h.clone()).transpose(); // [p, C]
    let dwe = d_e.transpose().matmul(h).transpose(); // [p, E]
    (dx, dwp, dwc, dwe)
}

/// Shared CUDA router launch: one kernel produces `c_idx [B,1]`, `idx [B,k]`,
/// `gates [B,k]`. Returns `None` when the shapes fall outside the kernel
/// limits (caller falls back to the tensor-path router).
#[allow(clippy::too_many_arguments)]
fn router_cuda(
    x: &Tensor<2>,
    w_proj: &Tensor<2>,
    w_cluster: &Tensor<2>,
    w_expert: &Tensor<2>,
    p: usize,
    c: usize,
    e: usize,
    k: usize,
) -> Option<(Tensor<2, Int>, Tensor<2, Int>, Tensor<2>)> {
    let [b, in_features] = x.dims();
    // kernel limits: p/c/e small enough for the comptime-unrolled register
    // arrays; k < e because the tensor path switches to argsort at k >= e
    // (not replicated here); k <= 16 (unroll bound, same as the tensor path's
    // argtopk round cap)
    if b == 0
        || p == 0
        || c == 0
        || e == 0
        || k == 0
        || k >= e
        || k > 16
        || p > 256
        || c > 256
        || e > 256
    {
        return None;
    }
    if w_proj.dims() != [in_features, p] || w_cluster.dims() != [p, c] || w_expert.dims() != [p, e]
    {
        return None;
    }
    let xc = cube_of2(x)?;
    let wpc = cube_of2(w_proj)?;
    let wcc = cube_of2(w_cluster)?;
    let wec = cube_of2(w_expert)?;
    let client = xc.client.clone();
    let dev = x.device();
    let c_idx = empty_dense_int([b, 1], &dev);
    let idx = empty_dense_int([b, k], &dev);
    let g = empty_dense([b, k], &dev);
    let c_idx_c = cube_int2(&c_idx)?;
    let idx_c = cube_int2(&idx)?;
    let gc = cube_of2(&g)?;
    unsafe {
        moe_router_kernel::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(b as u32, 1, 1),
            CubeDim::new_3d(32, 1, 1),
            BufferArg::from_raw_parts(xc.handle.clone(), b * in_features),
            BufferArg::from_raw_parts(wpc.handle.clone(), in_features * p),
            BufferArg::from_raw_parts(wcc.handle.clone(), p * c),
            BufferArg::from_raw_parts(wec.handle.clone(), p * e),
            BufferArg::from_raw_parts(c_idx_c.handle.clone(), b),
            BufferArg::from_raw_parts(idx_c.handle.clone(), b * k),
            BufferArg::from_raw_parts(gc.handle.clone(), b * k),
            b as u32,
            in_features as u32,
            p as u32,
            c as u32,
            e as u32,
            k as u32,
        );
    }
    Some((c_idx, idx, g))
}

/// Fused router on `Autodiff<CubeBackend>`: runs the kernels on
/// the inner backend and wraps only the gates in a tracked op (parents:
/// x, w_proj, w_cluster, w_expert; `c_idx`/`idx` are plain byproducts — they
/// never carry gradients).
#[allow(clippy::too_many_arguments)]
fn router_fused_autodiff(
    x: Tensor<2>,
    w_proj: Tensor<2>,
    w_cluster: Tensor<2>,
    w_expert: Tensor<2>,
    p: usize,
    c: usize,
    e: usize,
    k: usize,
) -> Option<(Tensor<2, Int>, Tensor<2, Int>, Tensor<2>)> {
    let xa = x.try_into_primitive::<CAd>().ok()?;
    let wpa = w_proj.try_into_primitive::<CAd>().ok()?;
    let wca = w_cluster.try_into_primitive::<CAd>().ok()?;
    let wea = w_expert.try_into_primitive::<CAd>().ok()?;
    let x_t = Tensor::<2>::from_primitive::<CB>(xa.primitive().clone());
    let wp_t = Tensor::<2>::from_primitive::<CB>(wpa.primitive().clone());
    let wc_t = Tensor::<2>::from_primitive::<CB>(wca.primitive().clone());
    let we_t = Tensor::<2>::from_primitive::<CB>(wea.primitive().clone());
    let [b, in_features] = x_t.dims();
    let x_d = dense(x_t);
    let wp_d = dense(wp_t);
    let wc_d = dense(wc_t);
    let we_d = dense(we_t);
    let (c_idx, idx, g) = router_cuda(&x_d, &wp_d, &wc_d, &we_d, p, c, e, k)?;
    let g_prim = g.try_into_primitive::<CB>().unwrap();
    let nodes = [
        xa.node(),
        wpa.node(),
        wca.node(),
        wea.node(),
    ];
    let prep = RoutFwdOp.prepare::<NoCheckpointing>(nodes);
    let g_adt = match prep.compute_bound().stateful() {
        OpsKind::Tracked(mut prep) => {
            let _ids = [
                Some(prep.checkpoint(&xa)),
                Some(prep.checkpoint(&wpa)),
                Some(prep.checkpoint(&wca)),
                Some(prep.checkpoint(&wea)),
            ];
            prep.finish(
                RoutState {
                    x: x_d,
                    w_proj: wp_d,
                    w_cluster: wc_d,
                    w_expert: we_d,
                    b,
                    in_features,
                    p,
                    c,
                    e,
                    k,
                },
                g_prim,
            )
        }
        OpsKind::UnTracked(prep) => prep.finish(g_prim),
    };
    Some((c_idx, idx, Tensor::from_primitive::<CAd>(g_adt)))
}

/// Fused router on the bare CUDA backend (no autodiff, e.g. inference).
#[allow(clippy::too_many_arguments)]
fn router_fused_plain(
    x: Tensor<2>,
    w_proj: Tensor<2>,
    w_cluster: Tensor<2>,
    w_expert: Tensor<2>,
    p: usize,
    c: usize,
    e: usize,
    k: usize,
) -> Option<(Tensor<2, Int>, Tensor<2, Int>, Tensor<2>)> {
    let x_d = dense(x);
    let wp_d = dense(w_proj);
    let wc_d = dense(w_cluster);
    let we_d = dense(w_expert);
    let xc = cube_of2(&x_d)?;
    let client = xc.client.clone();
    let _ = futures_lite::future::block_on(client.sync());
    router_cuda(&x_d, &wp_d, &wc_d, &we_d, p, c, e, k)
}

/// Dense zero-initialized Int allocation.
fn empty_dense_int<const D: usize>(dims: [usize; D], device: &Device) -> Tensor<D, Int> {
    let n = dims.iter().product::<usize>();
    Tensor::<1, Int>::empty([n], device).reshape::<D, _>(dims)
}

/// Fused CUDA router: one kernel produces `c_idx [B,1]`, the global expert
/// ids `idx [B,k]` and gates `g [B,k]` from `x [B,in]` and the three router
/// weights (`w_proj [in,p]`, `w_cluster [p,C]`, `w_expert [p,E]`), replacing
/// the ~11 tensor-path ops of `SpectralMoE::forward` with one launch (and one
/// backward kernel). Returns `None` when the backend is not CUDA or the
/// shapes fall outside the kernel limits; the caller keeps the tensor-path
/// router as the fallback.
#[allow(clippy::too_many_arguments)]
pub fn router_fused(
    x: Tensor<2>,
    w_proj: Tensor<2>,
    w_cluster: Tensor<2>,
    w_expert: Tensor<2>,
    p: usize,
    c: usize,
    e: usize,
    k: usize,
) -> Option<(Tensor<2, Int>, Tensor<2, Int>, Tensor<2>)> {
    if let Some(r) = router_fused_autodiff(
        x.clone(),
        w_proj.clone(),
        w_cluster.clone(),
        w_expert.clone(),
        p,
        c,
        e,
        k,
    ) {
        return Some(r);
    }
    router_fused_plain(x, w_proj, w_cluster, w_expert, p, c, e, k)
}

/// Checkpointed state: the projection and the dense inputs (captured at
/// forward time so the backward reads the exact values even when a parent is
/// pruned by retract+optimizer).
#[derive(Debug, Clone)]
struct MoeState {
    z: Handle,
    idx: Tensor<2, Int>,
    x: Tensor<2>,
    u: Tensor<2>,
    v: Tensor<2>,
    s: Tensor<1>,
    gates: Tensor<2>,
    b: usize,
    dim_in: usize,
    dim_out: usize,
    m: usize,
    k: usize,
    r: usize,
}

#[derive(Debug)]
struct MoeFwdOp;

impl<B: Backend> Backward<B, 5> for MoeFwdOp
where
    DispatchTensor: DispatchKindConversion<B>,
{
    type State = MoeState;

    fn backward(
        self,
        ops: Ops<Self::State, 5>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let st = &ops.state;
        let d_out = Tensor::<2>::from_primitive::<B>(grads.consume::<B>(&ops.node));

        #[cfg(feature = "cuda")]
        {
            if std::any::TypeId::of::<B>() == std::any::TypeId::of::<CB>() {
                if let Some((dx, du, dv, ds, dg)) = moe_backward_cuda(st, &d_out) {
                    if let Some(node) = ops.parents[0].clone() {
                        grads.register::<B>(node.id, dx.try_into_primitive::<B>().unwrap());
                    }
                    if let Some(node) = ops.parents[1].clone() {
                        grads.register::<B>(node.id, du.try_into_primitive::<B>().unwrap());
                    }
                    if let Some(node) = ops.parents[2].clone() {
                        grads.register::<B>(node.id, dv.try_into_primitive::<B>().unwrap());
                    }
                    if let Some(node) = ops.parents[3].clone() {
                        grads.register::<B>(node.id, ds.try_into_primitive::<B>().unwrap());
                    }
                    if let Some(node) = ops.parents[4].clone() {
                        grads.register::<B>(node.id, dg.try_into_primitive::<B>().unwrap());
                    }
                    return;
                }
            }
        }

        let (dx, du, dv, ds, dg) =
            moe_backward_tensor(&st.x, &st.u, &st.v, &st.s, &st.gates, &st.idx, &d_out, st.r);
        if let Some(node) = ops.parents[0].clone() {
            grads.register::<B>(node.id, dx.try_into_primitive::<B>().unwrap());
        }
        if let Some(node) = ops.parents[1].clone() {
            grads.register::<B>(node.id, du.try_into_primitive::<B>().unwrap());
        }
        if let Some(node) = ops.parents[2].clone() {
            grads.register::<B>(node.id, dv.try_into_primitive::<B>().unwrap());
        }
        if let Some(node) = ops.parents[3].clone() {
            grads.register::<B>(node.id, ds.try_into_primitive::<B>().unwrap());
        }
        if let Some(node) = ops.parents[4].clone() {
            grads.register::<B>(node.id, dg.try_into_primitive::<B>().unwrap());
        }
    }
}

/// Fused backward on the bare CUDA backend.
fn moe_backward_cuda(
    st: &MoeState,
    d_out: &Tensor<2>,
) -> Option<(Tensor<2>, Tensor<2>, Tensor<2>, Tensor<1>, Tensor<2>)> {
    let xc = cube_of2(&st.x)?;
    let uc = cube_of2(&st.u)?;
    let vc = cube_of2(&st.v)?;
    let sc = cube_of1(&st.s)?;
    let gc = cube_of2(&st.gates)?;
    let idxc = cube_int2(&st.idx)?;
    let d_out = dense(d_out.clone());
    let doc = cube_of2(&d_out)?;
    let client = xc.client.clone();
    let _ = futures_lite::future::block_on(client.sync());

    let b = st.b;
    let dim_in = st.dim_in;
    let m = st.m;
    let dim_out = st.dim_out;
    let k = st.k;
    let r = st.r;
    let dev = st.x.device();

    let dx = empty_dense([b, dim_in], &dev);
    let du = zeros_dense([dim_in, m], &dev);
    let dv = zeros_dense([dim_out, m], &dev);
    let ds = zeros_dense([m], &dev);
    let dg = empty_dense([b, k], &dev);
    let dxc = cube_of2(&dx)?;
    let duc = cube_of2(&du)?;
    let dvc = cube_of2(&dv)?;
    let dsc = cube_of1(&ds)?;
    let dgc = cube_of2(&dg)?;

    unsafe {
        moe_bwd_kernel::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(b as u32, 1, 1),
            CubeDim::new_3d(32, 1, 1),
            BufferArg::from_raw_parts(doc.handle.clone(), b * dim_out),
            BufferArg::from_raw_parts(xc.handle.clone(), b * dim_in),
            BufferArg::from_raw_parts(uc.handle.clone(), dim_in * m),
            BufferArg::from_raw_parts(vc.handle.clone(), dim_out * m),
            BufferArg::from_raw_parts(sc.handle.clone(), m),
            BufferArg::from_raw_parts(idxc.handle.clone(), b * k),
            BufferArg::from_raw_parts(gc.handle.clone(), b * k),
            BufferArg::from_raw_parts(st.z.clone(), b * k * r),
            BufferArg::from_raw_parts(dxc.handle.clone(), b * dim_in),
            BufferArg::from_raw_parts(duc.handle.clone(), dim_in * m),
            BufferArg::from_raw_parts(dvc.handle.clone(), dim_out * m),
            BufferArg::from_raw_parts(dsc.handle.clone(), m),
            BufferArg::from_raw_parts(dgc.handle.clone(), b * k),
            b as u32,
            dim_in as u32,
            m as u32,
            dim_out as u32,
            k as u32,
            r as u32,
        );
    }
    Some((dx, du, dv, ds, dg))
}

/// Tensor-path backward (reference; dead in practice since the op is only
/// built for `Autodiff<CubeBackend>`). Re-materializes the
/// gather/intermediates, so it is NOT used for the memory win.
#[allow(clippy::too_many_arguments)]
fn moe_backward_tensor(
    x: &Tensor<2>,
    u: &Tensor<2>,
    v: &Tensor<2>,
    s: &Tensor<1>,
    gates: &Tensor<2>,
    idx: &Tensor<2, Int>,
    d_out: &Tensor<2>,
    rank: usize,
) -> (Tensor<2>, Tensor<2>, Tensor<2>, Tensor<1>, Tensor<2>) {
    let [b, dim_in] = x.dims();
    let dim_out = v.dims()[0];
    let m = u.dims()[1];
    let k = idx.dims()[1];
    let r = rank;
    let kk = k * r;
    let device = x.device();

    let ar = Tensor::<1, Int>::arange(0..r as i64, &device);
    let cols = idx
        .clone()
        .unsqueeze_dim::<3>(2)
        .mul_scalar(r as i64)
        .add(
            ar.unsqueeze_dim::<2>(0)
                .unsqueeze_dim::<3>(1)
                .expand([b, k, r]),
        )
        .reshape([b, kk]);
    let g_r = gates
        .clone()
        .unsqueeze_dim::<3>(2)
        .expand([b, k, r])
        .reshape([b, kk]);

    let u_g = u
        .clone()
        .transpose()
        .gather(
            0,
            cols.clone().reshape([b * kk, 1]).expand([b * kk, dim_in]),
        )
        .reshape([b, kk, dim_in]);
    let v_g = v
        .clone()
        .transpose()
        .gather(
            0,
            cols.clone().reshape([b * kk, 1]).expand([b * kk, dim_out]),
        )
        .reshape([b, kk, dim_out]);
    let s_g = s
        .clone()
        .gather(0, cols.clone().reshape([b * kk]))
        .reshape([b, kk]);

    let p = (x.clone().unsqueeze_dim::<3>(1) * u_g.clone())
        .sum_dim(2)
        .squeeze_dim::<2>(2);
    let w = (d_out.clone().unsqueeze_dim::<3>(1) * v_g.clone())
        .sum_dim(2)
        .squeeze_dim::<2>(2);
    let dz = g_r.clone().mul(s_g.clone()).mul(w.clone());

    let dx = (dz.clone().unsqueeze_dim::<3>(2) * u_g)
        .sum_dim(1)
        .squeeze_dim::<2>(1);

    let du = Tensor::<2>::zeros([dim_in, m], &device).scatter(
        1,
        cols.clone()
            .reshape([b * kk])
            .unsqueeze_dim::<2>(0)
            .expand([dim_in, b * kk]),
        x.clone()
            .transpose()
            .matmul(dz.clone())
            .reshape([dim_in, b * kk]),
        burn::tensor::IndexingUpdateOp::Add,
    );
    let dv = Tensor::<2>::zeros([dim_out, m], &device).scatter(
        1,
        cols.clone()
            .reshape([b * kk])
            .unsqueeze_dim::<2>(0)
            .expand([dim_out, b * kk]),
        d_out
            .clone()
            .transpose()
            .matmul(g_r.clone().mul(s_g.clone()).mul(p.clone()))
            .reshape([dim_out, b * kk]),
        burn::tensor::IndexingUpdateOp::Add,
    );
    let ds = Tensor::<1>::zeros([m], &device).scatter(
        0,
        cols.reshape([b * kk]),
        g_r.mul(p.clone()).mul(w.clone()).reshape([b * kk]),
        burn::tensor::IndexingUpdateOp::Add,
    );
    let dg = s_g
        .mul(p)
        .mul(w)
        .reshape([b, k, r])
        .sum_dim(2)
        .squeeze_dim::<2>(2);
    (dx, du, dv, ds, dg)
}

/// Fused forward on the bare CUDA backend (shared by the autodiff and plain
/// paths). Returns `None` when a tensor is not on the CUDA runtime or the
/// shape falls outside the kernel limits.
#[allow(clippy::too_many_arguments)]
fn moe_fwd(
    client: &Client,
    x: &Tensor<2>,
    u: &Tensor<2>,
    v: &Tensor<2>,
    s: &Tensor<1>,
    idx: &Tensor<2, Int>,
    gates: &Tensor<2>,
    b: usize,
    dim_in: usize,
    m: usize,
    dim_out: usize,
    k: usize,
    r: usize,
) -> Option<(Tensor<2>, MoeState)> {
    if b == 0 || k > 16 || r > 16 || k * r > 64 {
        return None;
    }
    let xc = cube_of2(x)?;
    let uc = cube_of2(u)?;
    let vc = cube_of2(v)?;
    let sc = cube_of1(s)?;
    let idxc = cube_int2(idx)?;
    let gc = cube_of2(gates)?;
    let dev = x.device();
    let kk = k * r;
    let z = client.empty(b * kk * 4);
    let y = empty_dense([b, dim_out], &dev);
    let yc = cube_of2(&y)?;
    unsafe {
        moe_fwd_kernel::launch_unchecked::<f32>(
            client,
            CubeCount::Static(b as u32, 1, 1),
            CubeDim::new_3d(32, 1, 1),
            BufferArg::from_raw_parts(xc.handle.clone(), b * dim_in),
            BufferArg::from_raw_parts(uc.handle.clone(), dim_in * m),
            BufferArg::from_raw_parts(vc.handle.clone(), dim_out * m),
            BufferArg::from_raw_parts(sc.handle.clone(), m),
            BufferArg::from_raw_parts(idxc.handle.clone(), b * k),
            BufferArg::from_raw_parts(gc.handle.clone(), b * k),
            BufferArg::from_raw_parts(z.clone(), b * kk),
            BufferArg::from_raw_parts(yc.handle.clone(), b * dim_out),
            b as u32,
            dim_in as u32,
            m as u32,
            dim_out as u32,
            k as u32,
            r as u32,
        );
    }
    MOE_FWD_COUNT.fetch_add(1, Ordering::Relaxed);
    Some((
        y,
        MoeState {
            z,
            idx: idx.clone(),
            x: x.clone(),
            u: u.clone(),
            v: v.clone(),
            s: s.clone(),
            gates: gates.clone(),
            b,
            dim_in,
            dim_out,
            m,
            k,
            r,
        },
    ))
}

/// Fused forward on `Autodiff<CubeBackend>`: runs the kernels on
/// the inner backend and wraps the result in a tracked op (parents:
/// x, u, v, s, gates).
#[allow(clippy::too_many_arguments)]
fn moe_fused_autodiff(
    x: Tensor<2>,
    u: Tensor<2>,
    v: Tensor<2>,
    s: Tensor<1>,
    idx: Tensor<2, Int>,
    gates: Tensor<2>,
    rank: usize,
) -> Option<Tensor<2>> {
    let xa = x.try_into_primitive::<CAd>().ok()?;
    let ua = u.try_into_primitive::<CAd>().ok()?;
    let va = v.try_into_primitive::<CAd>().ok()?;
    let sa = s.try_into_primitive::<CAd>().ok()?;
    let ga = gates.try_into_primitive::<CAd>().ok()?;
    let x_t = Tensor::<2>::from_primitive::<CB>(xa.primitive().clone());
    let u_t = Tensor::<2>::from_primitive::<CB>(ua.primitive().clone());
    let v_t = Tensor::<2>::from_primitive::<CB>(va.primitive().clone());
    let s_t = Tensor::<1>::from_primitive::<CB>(sa.primitive().clone());
    let g_t = Tensor::<2>::from_primitive::<CB>(ga.primitive().clone());
    let idx_t: Tensor<2, Int> = {
        let prim = idx.clone().try_into_primitive::<CB>().ok()?;
        let prim: <CB as burn::backend::BackendTypes>::IntTensorPrimitive = prim;
        Tensor::from_primitive::<CB>(prim)
    };
    let [b, dim_in] = x_t.dims();
    let dim_out = v_t.dims()[0];
    let m = u_t.dims()[1];
    let k = idx_t.dims()[1];

    let x_d = dense(x_t.clone());
    let u_d = dense(u_t.clone());
    let v_d = dense(v_t.clone());
    let s_d = dense(s_t.clone());
    let g_d = dense(g_t.clone());
    let idx_d = dense(idx_t.clone());
    let xc = cube_of2(&x_d)?;
    let client = xc.client.clone();
    // materialize the parents' pending graph work before the raw launches
    let _ = futures_lite::future::block_on(client.sync());
    let (out_t, state) = moe_fwd(
        &client, &x_d, &u_d, &v_d, &s_d, &idx_d, &g_d, b, dim_in, m, dim_out, k, rank,
    )?;
    let out_prim = out_t.try_into_primitive::<CB>().unwrap();
    let nodes = [
        xa.node(),
        ua.node(),
        va.node(),
        sa.node(),
        ga.node(),
    ];
    let prep = MoeFwdOp.prepare::<NoCheckpointing>(nodes);
    let out_adt = match prep.compute_bound().stateful() {
        OpsKind::Tracked(mut prep) => {
            let _ids = [
                Some(prep.checkpoint(&xa)),
                Some(prep.checkpoint(&ua)),
                Some(prep.checkpoint(&va)),
                Some(prep.checkpoint(&sa)),
                Some(prep.checkpoint(&ga)),
            ];
            prep.finish(state, out_prim)
        }
        OpsKind::UnTracked(prep) => prep.finish(out_prim),
    };
    Some(Tensor::from_primitive::<CAd>(out_adt))
}

/// Fused forward on the bare CUDA backend (no autodiff, e.g. inference).
#[allow(clippy::too_many_arguments)]
fn moe_fused_plain(
    x: Tensor<2>,
    u: Tensor<2>,
    v: Tensor<2>,
    s: Tensor<1>,
    idx: Tensor<2, Int>,
    gates: Tensor<2>,
    rank: usize,
) -> Option<Tensor<2>> {
    let [b, dim_in] = x.dims();
    let dim_out = v.dims()[0];
    let m = u.dims()[1];
    let k = idx.dims()[1];
    let x_d = dense(x);
    let u_d = dense(u);
    let v_d = dense(v);
    let s_d = dense(s);
    let g_d = dense(gates);
    let idx_d = dense(idx);
    let xc = cube_of2(&x_d)?;
    let client = xc.client.clone();
    let _ = futures_lite::future::block_on(client.sync());
    let (out, _st) = moe_fwd(
        &client, &x_d, &u_d, &v_d, &s_d, &idx_d, &g_d, b, dim_in, m, dim_out, k, rank,
    )?;
    Some(out)
}

/// Fused token-choice MoE forward. `u`/`v` are the ternary masters
/// (`ste_ternary` output), `idx` the `[B, k]` global expert ids, `gates` the
/// `[B, k]` router gates. Returns `None` when the backend is not CUDA or the
/// shapes fall outside the kernel limits (caller falls back to the tensor
/// path). `rank` is the per-expert rank `r`.
#[allow(clippy::too_many_arguments)]
pub fn forward_moe_fused(
    x: Tensor<2>,
    u: Tensor<2>,
    v: Tensor<2>,
    s: Tensor<1>,
    idx: Tensor<2, Int>,
    gates: Tensor<2>,
    rank: usize,
) -> Option<Tensor<2>> {
    if let Some(y) = moe_fused_autodiff(
        x.clone(),
        u.clone(),
        v.clone(),
        s.clone(),
        idx.clone(),
        gates.clone(),
        rank,
    ) {
        return Some(y);
    }
    moe_fused_plain(x, u, v, s, idx, gates, rank)
}

#[cfg(all(test, feature = "cuda"))]
mod moe_tests {
    use super::*;
    use crate::SpectralMoE;
    use burn::tensor::Distribution;

    fn to_host<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn to_host_int<const D: usize>(t: Tensor<D, Int>) -> Vec<i32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    fn relmaxdiff(a: &[f32], b: &[f32]) -> f32 {
        let scale = b.iter().fold(0.0_f32, |m, x| m.max(x.abs()));
        maxdiff(a, b) / scale.max(1e-6)
    }

    /// One fused run vs one tensor-path run on identical (seeded) MoE
    /// instances; returns (fwd, du, dv, ds, dx) relative deviations.
    fn moe_fused_vs_raw(
        b: usize,
        c: usize,
        e: usize,
        k: usize,
        r: usize,
        seed: u64,
    ) -> (f32, f32, f32, f32, f32) {
        let dim_in = 512usize;
        let dim_out = 512usize;
        Device::cuda(0).seed(seed);
        let adev = Device::cuda(0).autodiff();
        let x =
            Tensor::<2>::random([b, dim_in], Distribution::Normal(0.0, 1.0), &adev).require_grad();
        let mut mf = SpectralMoE::new(dim_in, dim_out, c, e, k, r, &adev);
        mf.set_fused(true);
        let mut mr = mf.clone();
        mr.set_fused(false);

        let yf = mf.forward(x.clone());
        let fwd = {
            let of = to_host(Tensor::<2>::from_data(
                yf.clone().into_data(),
                &Device::cuda(0),
            ));
            let yr = mr.forward(x.clone());
            let or = to_host(Tensor::<2>::from_data(yr.into_data(), &Device::cuda(0)));
            relmaxdiff(&of, &or)
        };
        let grads_f = yf.powf_scalar(2.0).sum().backward();
        let yr = mr.forward(x.clone());
        let grads_r = yr.powf_scalar(2.0).sum().backward();
        let du = relmaxdiff(
            &to_host(mf.u.grad(&grads_f).unwrap()),
            &to_host(mr.u.grad(&grads_r).unwrap()),
        );
        let dv = relmaxdiff(
            &to_host(mf.v.grad(&grads_f).unwrap()),
            &to_host(mr.v.grad(&grads_r).unwrap()),
        );
        let ds = relmaxdiff(
            &to_host(mf.s.grad(&grads_f).unwrap()),
            &to_host(mr.s.grad(&grads_r).unwrap()),
        );
        let dx = relmaxdiff(
            &to_host(x.grad(&grads_f).unwrap()),
            &to_host(x.grad(&grads_r).unwrap()),
        );
        (fwd, du, dv, ds, dx)
    }

    #[test]
    fn moe_fused_forward_matches_tensor_path() {
        // TALL config (m = c*e*r = 256 < dim_in = 512), realistic case
        let (fwd, du, dv, ds, dx) = moe_fused_vs_raw(64, 8, 8, 2, 4, 42);
        println!(
            "moe fused vs tensor: fwd {fwd:.6}, du {du:.6}, dv {dv:.6}, ds {ds:.6}, dx {dx:.6}"
        );
        assert!(fwd < 1e-3, "fused forward mismatch: {fwd}");
        assert!(du < 1e-2, "du mismatch: {du}");
        assert!(dv < 1e-2, "dv mismatch: {dv}");
        assert!(ds < 1e-2, "ds mismatch: {ds}");
        assert!(dx < 1e-2, "dx mismatch: {dx}");
    }

    #[test]
    fn moe_fused_grads_router() {
        // d_gates correctness: the router grads must match too (the gate
        // gradient flows through softmax -> expert/cluster logits)
        let dim_in = 256usize;
        let dim_out = 256usize;
        let (c, e, k, r) = (4usize, 4usize, 2usize, 4usize);
        Device::cuda(0).seed(7);
        let adev = Device::cuda(0).autodiff();
        let x =
            Tensor::<2>::random([64, dim_in], Distribution::Normal(0.0, 1.0), &adev).require_grad();
        let mut mf = SpectralMoE::new(dim_in, dim_out, c, e, k, r, &adev);
        mf.set_fused(true);
        let mut mr = mf.clone();
        mr.set_fused(false);
        let yf = mf.forward(x.clone());
        let grads_f = yf.powf_scalar(2.0).sum().backward();
        let yr = mr.forward(x.clone());
        let grads_r = yr.powf_scalar(2.0).sum().backward();
        // the gate feeds the expert logits -> expert_key, and c_w -> cluster_key
        let ek = relmaxdiff(
            &to_host(mf.expert_key.weight.grad(&grads_f).unwrap()),
            &to_host(mr.expert_key.weight.grad(&grads_r).unwrap()),
        );
        let ck = relmaxdiff(
            &to_host(mf.cluster_key.weight.grad(&grads_f).unwrap()),
            &to_host(mr.cluster_key.weight.grad(&grads_r).unwrap()),
        );
        println!("moe router grads: expert_key {ek:.6}, cluster_key {ck:.6}");
        assert!(ek < 1e-2, "expert_key grad mismatch: {ek}");
        assert!(ck < 1e-2, "cluster_key grad mismatch: {ck}");
    }

    #[test]
    fn router_fused_matches_tensor_path() {
        // task-specified config: in=512, p=32, C=4, E=8, k=2, B=64
        let (b, p, c, e, k) = (64usize, 32usize, 4usize, 8usize, 2usize);
        let dim_in = 512usize;
        Device::cuda(0).seed(2026);
        let dev = Device::cuda(0);
        let x = Tensor::<2>::random([b, dim_in], Distribution::Normal(0.0, 1.0), &dev);
        let w_proj = Tensor::<2>::random([dim_in, p], Distribution::Normal(0.0, 1.0), &dev);
        let w_cluster = Tensor::<2>::random([p, c], Distribution::Normal(0.0, 1.0), &dev);
        let w_expert = Tensor::<2>::random([p, e], Distribution::Normal(0.0, 1.0), &dev);

        let (c_idx, idx, g) = router_fused(
            x.clone(),
            w_proj.clone(),
            w_cluster.clone(),
            w_expert.clone(),
            p,
            c,
            e,
            k,
        )
        .expect("bare cuda router must engage");

        // tensor-path router, op-for-op what SpectralMoE::forward does
        let h = x.clone().matmul(w_proj.clone()); // [B, p]
        let c_l = h.clone().matmul(w_cluster.clone()); // [B, C]
        let e_l = h.clone().matmul(w_expert.clone()); // [B, E]
        let c_idx_t = c_l.clone().argmax(1);
        let c_w = activation::softmax(c_l, 1).gather(1, c_idx_t.clone());
        let pos_t = e_l.clone().argtopk(k, 1);
        let idx_t = c_idx_t
            .clone()
            .expand([b, k])
            .mul_scalar(e as i64)
            .add(pos_t.clone())
            .reshape([b, k]);
        let g_t = activation::softmax(e_l.clone(), 1)
            .gather(1, pos_t)
            .mul(c_w.expand([b, k]));

        let ci = to_host_int(c_idx);
        let ci_t = to_host_int(c_idx_t);
        let ix = to_host_int(idx);
        let ix_t = to_host_int(idx_t);
        assert_eq!(
            ci, ci_t,
            "cluster ids must match exactly (tie-break included)"
        );
        assert_eq!(
            ix, ix_t,
            "expert ids must match exactly (tie-break included)"
        );
        let gh = to_host(g);
        let gt = to_host(g_t);

        // The GPU tensor path runs its matmuls in TF32 (cubek-matmul converts
        // f32 inputs on tf32-capable hardware), so its scores differ from
        // strict fp32 by ~1e-3 relative and the softmax amplifies that to a
        // few % on the gates. The kernel is the exact fp32 formula; the
        // meaningful equivalence check is against a host fp32 replication of
        // the same formula, which must hold to 1e-4. The GPU comparison only
        // sanity-checks the tf32 floor.
        let xh = to_host(x.clone());
        let wph = to_host(w_proj.clone());
        let wch = to_host(w_cluster.clone());
        let weh = to_host(w_expert.clone());
        let mut hh = vec![vec![0f32; p]; b];
        for t in 0..b {
            for a in 0..p {
                let mut acc = 0f32;
                for j in 0..dim_in {
                    acc += xh[t * dim_in + j] * wph[j * p + a];
                }
                hh[t][a] = acc;
            }
        }
        let mut ch = vec![vec![0f32; c]; b];
        let mut eh = vec![vec![0f32; e]; b];
        for t in 0..b {
            for i in 0..c {
                let mut acc = 0f32;
                for a in 0..p {
                    acc += hh[t][a] * wch[a * c + i];
                }
                ch[t][i] = acc;
            }
            for i in 0..e {
                let mut acc = 0f32;
                for a in 0..p {
                    acc += hh[t][a] * weh[a * e + i];
                }
                eh[t][i] = acc;
            }
        }
        let mut gh_ref = vec![0f32; b * k];
        for t in 0..b {
            let (mut c_best, mut c_idx_h) = (ch[t][0], 0usize);
            for i in 1..c {
                if ch[t][i] > c_best {
                    c_best = ch[t][i];
                    c_idx_h = i;
                }
            }
            let mut c_sum = 0f32;
            for i in 0..c {
                c_sum += (ch[t][i] - c_best).exp();
            }
            let cw = 1.0 / c_sum;
            let mut e_max = f32::MIN;
            for i in 0..e {
                e_max = e_max.max(eh[t][i]);
            }
            let mut e_sum = 0f32;
            for i in 0..e {
                e_sum += (eh[t][i] - e_max).exp();
            }
            for s in 0..k {
                let pos = ix_t[t * k + s] - c_idx_h as i32 * e as i32;
                gh_ref[t * k + s] = (eh[t][pos as usize] - e_max).exp() / e_sum * cw;
            }
        }
        let g_ref_diff = relmaxdiff(&gh, &gh_ref);
        let g_gpu_diff = relmaxdiff(&gh, &gt);
        println!(
            "router fused vs tensor: c_idx exact, idx exact, g vs host-fp32 relmax {g_ref_diff:.6}, vs gpu-tf32 relmax {g_gpu_diff:.6}"
        );
        assert!(
            g_ref_diff < 1e-4,
            "gate mismatch vs exact fp32 formula: {g_ref_diff}"
        );
        assert!(
            g_gpu_diff < 0.1,
            "gate mismatch vs tf32 tensor path: {g_gpu_diff}"
        );
    }

    #[test]
    fn router_bwd_kernel_matches_host() {
        // direct backward-kernel check: synthetic dense dg (all ones) vs the
        // exact fp32 formulas on host
        let (b, p, c, e, k) = (8usize, 32usize, 4usize, 8usize, 2usize);
        let dim_in = 64usize;
        Device::cuda(0).seed(5);
        let dev = Device::cuda(0);
        let x = Tensor::<2>::random([b, dim_in], Distribution::Normal(0.0, 1.0), &dev);
        let w_proj = Tensor::<2>::random([dim_in, p], Distribution::Normal(0.0, 1.0), &dev);
        let w_cluster = Tensor::<2>::random([p, c], Distribution::Normal(0.0, 1.0), &dev);
        let w_expert = Tensor::<2>::random([p, e], Distribution::Normal(0.0, 1.0), &dev);
        let dg = dense(Tensor::<2>::ones([b, k], &dev));
        let st = RoutState {
            x: dense(x.clone()),
            w_proj: dense(w_proj.clone()),
            w_cluster: dense(w_cluster.clone()),
            w_expert: dense(w_expert.clone()),
            b,
            in_features: dim_in,
            p,
            c,
            e,
            k,
        };
        let (dx, dwp, dwc, dwe) = router_backward_cuda(&st, &dg).expect("cuda");
        let xh = to_host(x.clone());
        let wph = to_host(w_proj.clone());
        let wch = to_host(w_cluster.clone());
        let weh = to_host(w_expert.clone());
        let ix = {
            let (_, idx, _) = router_fused(
                x.clone(),
                w_proj.clone(),
                w_cluster.clone(),
                w_expert.clone(),
                p,
                c,
                e,
                k,
            )
            .expect("fwd");
            to_host_int(idx)
        };
        let mut dxh = vec![vec![0f32; dim_in]; b];
        let mut dwch = vec![vec![0f32; c]; p];
        let mut dweh = vec![vec![0f32; e]; p];
        let mut dwph = vec![vec![0f32; p]; dim_in];
        for t in 0..b {
            let mut hh = vec![0f32; p];
            for a in 0..p {
                let mut acc = 0f32;
                for j in 0..dim_in {
                    acc += xh[t * dim_in + j] * wph[j * p + a];
                }
                hh[a] = acc;
            }
            let mut ch = vec![0f32; c];
            for i in 0..c {
                let mut acc = 0f32;
                for a in 0..p {
                    acc += hh[a] * wch[a * c + i];
                }
                ch[i] = acc;
            }
            let mut eh = vec![0f32; e];
            for i in 0..e {
                let mut acc = 0f32;
                for a in 0..p {
                    acc += hh[a] * weh[a * e + i];
                }
                eh[i] = acc;
            }
            let (mut c_best, mut c_idx) = (ch[0], 0usize);
            for i in 1..c {
                if ch[i] > c_best {
                    c_best = ch[i];
                    c_idx = i;
                }
            }
            let mut c_sum = 0f32;
            for i in 0..c {
                c_sum += (ch[i] - c_best).exp();
            }
            let smc: Vec<f32> = (0..c).map(|i| (ch[i] - c_best).exp() / c_sum).collect();
            let mut e_max = f32::MIN;
            for i in 0..e {
                e_max = e_max.max(eh[i]);
            }
            let mut e_sum = 0f32;
            for i in 0..e {
                e_sum += (eh[i] - e_max).exp();
            }
            let sme: Vec<f32> = (0..e).map(|i| (eh[i] - e_max).exp() / e_sum).collect();
            let pos: Vec<usize> = (0..k)
                .map(|s| (ix[t * k + s] - c_idx as i32 * e as i32) as usize)
                .collect();
            let cw = smc[c_idx];
            let dcw = (0..k).map(|s| 1.0 * sme[pos[s]]).sum::<f32>();
            let mut d_c = vec![0f32; c];
            for i in 0..c {
                let delta = if i == c_idx { 1.0 } else { 0.0 };
                d_c[i] = dcw * smc[i] * (delta - smc[c_idx]);
            }
            let mut e_dot = 0f32;
            for s in 0..k {
                e_dot += 1.0 * cw * sme[pos[s]];
            }
            let mut d_e = vec![0f32; e];
            for i in 0..e {
                let mut dsme = 0f32;
                for s in 0..k {
                    if pos[s] == i {
                        dsme = 1.0 * cw;
                    }
                }
                d_e[i] = sme[i] * (dsme - e_dot);
            }
            let mut dh = vec![0f32; p];
            for a in 0..p {
                let mut acc = 0f32;
                for i in 0..c {
                    acc += d_c[i] * wch[a * c + i];
                }
                for i in 0..e {
                    acc += d_e[i] * weh[a * e + i];
                }
                dh[a] = acc;
            }
            for a in 0..p {
                for i in 0..c {
                    dwch[a][i] += d_c[i] * hh[a];
                }
                for i in 0..e {
                    dweh[a][i] += d_e[i] * hh[a];
                }
            }
            for j in 0..dim_in {
                let mut acc = 0f32;
                for a in 0..p {
                    acc += dh[a] * wph[j * p + a];
                }
                dxh[t][j] = acc;
                for a in 0..p {
                    dwph[j][a] += dh[a] * xh[t * dim_in + j];
                }
            }
        }
        // tiny grads make relmaxdiff scale-inflated (dwe max entry ~6e-5),
        // so assert absolute diffs: the kernel must match the exact fp32
        // formulas to ~1e-5
        let dxd = maxdiff(
            &to_host(dx),
            &dxh.iter().flatten().copied().collect::<Vec<_>>(),
        );
        let dwpd = maxdiff(
            &to_host(dwp),
            &dwph.iter().flatten().copied().collect::<Vec<_>>(),
        );
        let dwcd = maxdiff(
            &to_host(dwc),
            &dwch.iter().flatten().copied().collect::<Vec<_>>(),
        );
        let dwed = maxdiff(
            &to_host(dwe),
            &dweh.iter().flatten().copied().collect::<Vec<_>>(),
        );
        println!(
            "bwd kernel vs host (abs): dx {dxd:.7}, dwp {dwpd:.7}, dwc {dwcd:.7}, dwe {dwed:.7}"
        );
        assert!(dxd < 1e-4, "d_x mismatch: {dxd}");
        assert!(dwpd < 1e-4, "d_wp mismatch: {dwpd}");
        assert!(dwcd < 1e-4, "d_wc mismatch: {dwcd}");
        assert!(dwed < 1e-4, "d_we mismatch: {dwed}");
    }

    #[test]
    fn router_fused_ties_break_lowest_index() {
        // identical cluster/expert weight rows -> every score ties; argmax
        // and argtopk must pick the lowest indices, exactly like the tensor
        // path (cubek-reduce keeps the lower coordinate on ties)
        let (b, p, c, e, k) = (16usize, 8usize, 4usize, 8usize, 3usize);
        let dim_in = 16usize;
        Device::cuda(0).seed(77);
        let dev = Device::cuda(0);
        let x = Tensor::<2>::random([b, dim_in], Distribution::Normal(0.0, 1.0), &dev);
        let w_proj = Tensor::<2>::random([dim_in, p], Distribution::Normal(0.0, 1.0), &dev);
        let w_cluster = Tensor::<2>::full([p, c], 0.5, &dev);
        let w_expert = Tensor::<2>::full([p, e], -0.25, &dev);

        let (c_idx, idx, g) = router_fused(
            x.clone(),
            w_proj.clone(),
            w_cluster.clone(),
            w_expert.clone(),
            p,
            c,
            e,
            k,
        )
        .expect("bare cuda router must engage");

        let h = x.matmul(w_proj);
        let c_l = h.clone().matmul(w_cluster);
        let e_l = h.matmul(w_expert);
        let c_idx_t = c_l.clone().argmax(1);
        let pos_t = e_l.clone().argtopk(k, 1);
        let idx_t = c_idx_t
            .clone()
            .expand([b, k])
            .mul_scalar(e as i64)
            .add(pos_t.clone())
            .reshape([b, k]);
        let g_t = activation::softmax(e_l, 1)
            .gather(1, pos_t)
            .mul(activation::softmax(c_l, 1).gather(1, c_idx_t.clone()));

        assert_eq!(to_host_int(c_idx), to_host_int(c_idx_t));
        assert_eq!(to_host_int(idx), to_host_int(idx_t.clone()));
        // all scores equal: lowest indices must win on both sides
        assert!(
            to_host_int(idx_t).chunks(k).all(|row| row == [0, 1, 2]),
            "argtopk ties must pick positions 0..k"
        );
        let gd = relmaxdiff(&to_host(g), &to_host(g_t));
        println!("router tie-break: idx exact, g relmax {gd:.6}");
        assert!(gd < 1e-4, "tie gate mismatch: {gd}");
    }

    #[test]
    fn router_fused_grads_match_tensor_path() {
        // d_gates -> router weight grads through the fused backward kernel
        // must match the exact fp32 formula. The GPU tensor path's matmuls
        // run in TF32 (cubek-matmul on tf32 hardware), so its backward is
        // TF32-noisy too and cannot be the reference at tight tolerances;
        // the reference is a host fp32 replication of the softmax/gather
        // backward instead (the full-model router-grad equivalence is covered
        // by moe_fused_grads_router at the tf32 floor).
        let (b, p, c, e, k) = (32usize, 32usize, 4usize, 8usize, 2usize);
        let dim_in = 256usize;
        Device::cuda(0).seed(9);
        let dev = Device::cuda(0).autodiff();
        let x =
            Tensor::<2>::random([b, dim_in], Distribution::Normal(0.0, 1.0), &dev).require_grad();
        let w_proj =
            Tensor::<2>::random([dim_in, p], Distribution::Normal(0.0, 1.0), &dev).require_grad();
        let w_cluster =
            Tensor::<2>::random([p, c], Distribution::Normal(0.0, 1.0), &dev).require_grad();
        let w_expert =
            Tensor::<2>::random([p, e], Distribution::Normal(0.0, 1.0), &dev).require_grad();

        let (_, idx, g) = router_fused(
            x.clone(),
            w_proj.clone(),
            w_cluster.clone(),
            w_expert.clone(),
            p,
            c,
            e,
            k,
        )
        .expect("bare cuda router must engage");
        let grads_f = g.powf_scalar(2.0).sum().backward();

        // host fp32 reference: forward scores/softmaxes, then the exact
        // backward (softmax formulas + outer products)
        let xh = to_host(x.clone());
        let wph = to_host(w_proj.clone());
        let wch = to_host(w_cluster.clone());
        let weh = to_host(w_expert.clone());
        let ix = to_host_int(idx.clone());
        let mut hh = vec![vec![0f32; p]; b];
        let mut ch = vec![vec![0f32; c]; b];
        let mut eh = vec![vec![0f32; e]; b];
        for t in 0..b {
            for a in 0..p {
                let mut acc = 0f32;
                for j in 0..dim_in {
                    acc += xh[t * dim_in + j] * wph[j * p + a];
                }
                hh[t][a] = acc;
            }
            for i in 0..c {
                let mut acc = 0f32;
                for a in 0..p {
                    acc += hh[t][a] * wch[a * c + i];
                }
                ch[t][i] = acc;
            }
            for i in 0..e {
                let mut acc = 0f32;
                for a in 0..p {
                    acc += hh[t][a] * weh[a * e + i];
                }
                eh[t][i] = acc;
            }
        }
        let mut dxh = vec![vec![0f32; dim_in]; b];
        let mut dwph = vec![vec![0f32; p]; dim_in];
        let mut dwch = vec![vec![0f32; c]; p];
        let mut dweh = vec![vec![0f32; e]; p];
        for t in 0..b {
            let (mut c_best, mut c_idx) = (ch[t][0], 0usize);
            for i in 1..c {
                if ch[t][i] > c_best {
                    c_best = ch[t][i];
                    c_idx = i;
                }
            }
            let mut c_sum = 0f32;
            for i in 0..c {
                c_sum += (ch[t][i] - c_best).exp();
            }
            let smc: Vec<f32> = (0..c).map(|i| (ch[t][i] - c_best).exp() / c_sum).collect();
            let mut e_max = f32::MIN;
            for i in 0..e {
                e_max = e_max.max(eh[t][i]);
            }
            let mut e_sum = 0f32;
            for i in 0..e {
                e_sum += (eh[t][i] - e_max).exp();
            }
            let sme: Vec<f32> = (0..e).map(|i| (eh[t][i] - e_max).exp() / e_sum).collect();
            let pos: Vec<usize> = (0..k)
                .map(|s| (ix[t * k + s] - c_idx as i32 * e as i32) as usize)
                .collect();
            // dg = d(g^2)/dg = 2g = 2·sme[pos]·cw
            let cw = smc[c_idx];
            let dg: Vec<f32> = pos.iter().map(|&p| 2.0 * sme[p] * cw).collect();
            let dcw = (0..k).map(|s| dg[s] * sme[pos[s]]).sum::<f32>();
            let mut d_c = vec![0f32; c];
            for i in 0..c {
                let delta = if i == c_idx { 1.0 } else { 0.0 };
                d_c[i] = dcw * smc[i] * (delta - smc[c_idx]);
            }
            let mut d_e = vec![0f32; e];
            let mut e_dot = 0f32;
            for s in 0..k {
                e_dot += dg[s] * cw * sme[pos[s]];
            }
            for i in 0..e {
                let mut dsme = 0f32;
                for s in 0..k {
                    if pos[s] == i {
                        dsme = dg[s] * cw;
                    }
                }
                d_e[i] = sme[i] * (dsme - e_dot);
            }
            let mut dh = vec![0f32; p];
            for a in 0..p {
                let mut acc = 0f32;
                for i in 0..c {
                    acc += d_c[i] * wch[a * c + i];
                }
                for i in 0..e {
                    acc += d_e[i] * weh[a * e + i];
                }
                dh[a] = acc;
            }
            for a in 0..p {
                for i in 0..c {
                    dwch[a][i] += d_c[i] * hh[t][a];
                }
                for i in 0..e {
                    dweh[a][i] += d_e[i] * hh[t][a];
                }
            }
            for j in 0..dim_in {
                let mut acc = 0f32;
                for a in 0..p {
                    acc += dh[a] * wph[j * p + a];
                }
                dxh[t][j] = acc;
                for a in 0..p {
                    dwph[j][a] += dh[a] * xh[t * dim_in + j];
                }
            }
        }

        let grads = grads_f;
        let dx = relmaxdiff(
            &to_host(x.grad(&grads).unwrap()),
            &dxh.iter().flatten().copied().collect::<Vec<_>>(),
        );
        let dwp = relmaxdiff(
            &to_host(w_proj.grad(&grads).unwrap()),
            &dwph.iter().flatten().copied().collect::<Vec<_>>(),
        );
        let dwc = relmaxdiff(
            &to_host(w_cluster.grad(&grads).unwrap()),
            &dwch.iter().flatten().copied().collect::<Vec<_>>(),
        );
        let dwe = relmaxdiff(
            &to_host(w_expert.grad(&grads).unwrap()),
            &dweh.iter().flatten().copied().collect::<Vec<_>>(),
        );
        println!(
            "router grads fused vs host-fp32: dx {dx:.6}, dwp {dwp:.6}, dwc {dwc:.6}, dwe {dwe:.6}"
        );
        assert!(dx < 1e-3, "d_x mismatch: {dx}");
        assert!(dwp < 1e-3, "d_w_proj mismatch: {dwp}");
        assert!(dwc < 1e-3, "d_w_cluster mismatch: {dwc}");
        assert!(dwe < 1e-3, "d_w_expert mismatch: {dwe}");
    }

    #[test]
    fn moe_fused_falls_back_off_cuda_shapes() {
        // verify a fused-eligible config actually engaged the expert kernel
        let dim_in = 128usize;
        let dim_out = 128usize;
        let (c, e, k, r) = (4usize, 4usize, 2usize, 2usize);
        Device::cuda(0).seed(3);
        let adev = Device::cuda(0).autodiff();
        let x = Tensor::<2>::random([32, dim_in], Distribution::Normal(0.0, 1.0), &adev);
        let before = moe_fused_forward_count();
        let mut m = SpectralMoE::new(dim_in, dim_out, c, e, k, r, &adev);
        m.set_fused(true);
        let y = m.forward(x);
        assert_eq!(y.dims(), [32, dim_out]);
        assert!(
            moe_fused_forward_count() > before,
            "the fused path must engage for this config"
        );
    }

    #[test]
    fn moe_fused_square_master_matches_tensor_path() {
        // SQUARE case (m = c*e*r = 512 == dim_in == dim_out): the polar fix
        // (spectral-norm pre-scale) must keep the [512, 512] master sane, and
        // the fused path must agree with the tensor path on it.
        let (fwd, du, dv, ds, dx) = moe_fused_vs_raw(8, 8, 8, 2, 8, 1337);
        println!("moe fused square vs tensor: fwd {fwd:.6}, du {du:.6}, dv {dv:.6}, ds {ds:.6}, dx {dx:.6}");
        assert!(fwd < 1e-3, "square fused forward mismatch: {fwd}");
        assert!(du < 1e-2, "square du mismatch: {du}");
        assert!(dv < 1e-2, "square dv mismatch: {dv}");
        assert!(ds < 1e-2, "square ds mismatch: {ds}");
        assert!(dx < 1e-2, "square dx mismatch: {dx}");
    }

    #[test]
    fn moe_fused_no_oom_large_batch() {
        // TALL config, B=4096 and B=8192: the fused path never materializes
        // [B, k*r, in] / [B, k*r, out], so both must run in a few MB of
        // output (the tensor path's [8192, 8, 512] gather would be 128MB).
        let dim_in = 512usize;
        let dim_out = 512usize;
        let (c, e, k, r) = (8usize, 8usize, 2usize, 4usize);
        Device::cuda(0).seed(11);
        let adev = Device::cuda(0).autodiff();
        let mut m = SpectralMoE::new(dim_in, dim_out, c, e, k, r, &adev);
        m.set_fused(true);
        let before = moe_fused_forward_count();
        for b in [4096usize, 8192] {
            let x = Tensor::<2>::random([b, dim_in], Distribution::Normal(0.0, 1.0), &adev);
            let y = m.forward(x);
            assert_eq!(y.dims(), [b, dim_out]);
            let _: f32 = y.mean().into_scalar();
        }
        // the global counter is shared with the other moe_fused tests when
        // the suite runs in parallel, so only assert the fused path engaged
        // (>= 2: one per batch; exact-2 would be racy under --test-threads>1)
        assert!(
            moe_fused_forward_count() >= before + 2,
            "both batches must use the fused kernel"
        );
    }

    #[test]
    #[ignore = "perf smoke: run manually with --ignored"]
    fn router_kernel_perf_smoke() {
        // aria scale: B=2048, in=512, p=32, C=4, E=8, k=2
        let (b, p, c, e, k) = (2048usize, 32usize, 4usize, 8usize, 2usize);
        let dim_in = 512usize;
        Device::cuda(0).seed(99);
        let dev = Device::cuda(0);
        let x = Tensor::<2>::random([b, dim_in], Distribution::Normal(0.0, 1.0), &dev);
        let w_proj = Tensor::<2>::random([dim_in, p], Distribution::Normal(0.0, 1.0), &dev);
        let w_cluster = Tensor::<2>::random([p, c], Distribution::Normal(0.0, 1.0), &dev);
        let w_expert = Tensor::<2>::random([p, e], Distribution::Normal(0.0, 1.0), &dev);
        let dg = dense(Tensor::<2>::ones([b, k], &dev));
        let st = RoutState {
            x: dense(x.clone()),
            w_proj: dense(w_proj.clone()),
            w_cluster: dense(w_cluster.clone()),
            w_expert: dense(w_expert.clone()),
            b,
            in_features: dim_in,
            p,
            c,
            e,
            k,
        };
        // warm up (kernel compile + allocator); into_data forces execution
        for _ in 0..3 {
            let (_, _, g) = router_fused(
                x.clone(),
                w_proj.clone(),
                w_cluster.clone(),
                w_expert.clone(),
                p,
                c,
                e,
                k,
            )
            .expect("fwd");
            let _ = g.into_data();
        }
        let iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let (_, _, g) = router_fused(
                x.clone(),
                w_proj.clone(),
                w_cluster.clone(),
                w_expert.clone(),
                p,
                c,
                e,
                k,
            )
            .expect("fwd");
            let d = g.into_data();
            assert!(!d.bytes.is_empty(), "kernel produced no output");
        }
        let tf = t0.elapsed().as_secs_f32() / iters as f32;
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let (dx, dwp, dwc, dwe) = router_backward_cuda(&st, &dg).expect("bwd");
            let _ = (
                dx.into_data(),
                dwp.into_data(),
                dwc.into_data(),
                dwe.into_data(),
            );
        }
        let tb = t1.elapsed().as_secs_f32() / iters as f32;
        println!("router fwd {tf:.3} ms, bwd {tb:.3} ms @B={b}");
    }

    /// The pre-tiling forward kernel (1 thread per token, 64-token cubes),
    /// kept as the microbench baseline. Must stay bit-identical to the
    /// tiled `moe_fwd_kernel` (the bench asserts maxdiff == 0.0).
    #[cube(launch_unchecked)]
    fn moe_fwd_kernel_scalar<F: Float>(
        x: &[F],
        u: &[F],
        v: &[F],
        s: &[F],
        idx: &[i32],
        gates: &[F],
        z: &mut [F],
        y: &mut [F],
        b: u32,
        dim_in: u32,
        m: u32,
        dim_out: u32,
        #[comptime] k: u32,
        #[comptime] r: u32,
    ) {
        let t = CUBE_POS_X as usize * 64usize + UNIT_POS_X as usize;
        let dim_in = dim_in as usize;
        let m = m as usize;
        let dim_out = dim_out as usize;
        let k = k as usize;
        let r = r as usize;
        let kk = k * r;

        if (t as u32) < b {
            let mut p = Array::<F>::new(kk);
            #[unroll]
            for c in 0..kk {
                p[c] = F::new(0.0_f32);
            }
            let mut i = 0usize;
            while i < dim_in {
                let xv = x[t * dim_in + i];
                #[unroll]
                for j in 0..k {
                    let e = idx[t * k + j] as usize;
                    #[unroll]
                    for jj in 0..r {
                        p[j * r + jj] += xv * u[i * m + e * r + jj];
                    }
                }
                i += 1usize;
            }
            #[unroll]
            for c in 0..kk {
                z[t * kk + c] = p[c];
            }
            let mut o = 0usize;
            while o < dim_out {
                let mut acc = F::new(0.0_f32);
                #[unroll]
                for j in 0..k {
                    let e = idx[t * k + j] as usize;
                    let g = gates[t * k + j];
                    #[unroll]
                    for jj in 0..r {
                        acc += g * s[e * r + jj] * p[j * r + jj] * v[o * m + e * r + jj];
                    }
                }
                y[t * dim_out + o] = acc;
                o += 1usize;
            }
        }
    }

    /// The pre-tiling backward kernel (1 thread per token, 64-token cubes),
    /// kept as the microbench baseline. Bit-equality with the tiled
    /// `moe_bwd_kernel` does not hold (the w reduction reorders FMAs and the
    /// scatter-adds are atomic), so the bench asserts a tight relative
    /// tolerance; production-path grad equivalence is covered by
    /// `moe_fused_forward_matches_tensor_path`.
    #[allow(clippy::unnecessary_cast)]
    #[cube(launch_unchecked)]
    fn moe_bwd_kernel_scalar<F: Float>(
        d_out: &[F],           // [b, dim_out] dense
        x: &[F],               // [b, dim_in] dense
        u: &[F],               // [dim_in, m] dense
        v: &[F],               // [dim_out, m] dense
        s: &[F],               // [m]
        idx: &[i32],           // [b, k]
        gates: &[F],           // [b, k]
        z: &[F],               // [b, k*r]
        d_x: &mut [F],         // [b, dim_in]
        d_u: &mut [Atomic<F>], // [dim_in, m] zeroed
        d_v: &mut [Atomic<F>], // [dim_out, m] zeroed
        d_s: &mut [Atomic<F>], // [m] zeroed
        d_gates: &mut [F],     // [b, k]
        b: u32,
        dim_in: u32,
        m: u32,
        dim_out: u32,
        #[comptime] k: u32,
        #[comptime] r: u32,
    ) {
        let t = CUBE_POS_X as usize * 64usize + UNIT_POS_X as usize;
        let dim_in = dim_in as usize;
        let m = m as usize;
        let dim_out = dim_out as usize;
        let k = k as usize;
        let r = r as usize;
        let kk = k * r;

        if (t as u32) < b {
            let mut w = Array::<F>::new(kk);
            #[unroll]
            for c in 0..kk {
                w[c] = F::new(0.0_f32);
            }
            let mut o = 0usize;
            while o < dim_out {
                let dvo = d_out[t * dim_out + o];
                #[unroll]
                for j in 0..k {
                    let e = idx[t * k + j] as usize;
                    let g = gates[t * k + j];
                    #[unroll]
                    for jj in 0..r {
                        let col = e * r + jj;
                        w[j * r + jj] += dvo * v[o * m + col];
                        d_v[o * m + col].fetch_add(g * s[col] * z[t * kk + j * r + jj] * dvo);
                    }
                }
                o += 1usize;
            }
            let mut i = 0usize;
            while i < dim_in {
                let xv = x[t * dim_in + i];
                let mut acc = F::new(0.0_f32);
                #[unroll]
                for j in 0..k {
                    let e = idx[t * k + j] as usize;
                    let g = gates[t * k + j];
                    #[unroll]
                    for jj in 0..r {
                        let col = e * r + jj;
                        acc += g * s[col] * u[i * m + col] * w[j * r + jj];
                        d_u[i * m + col].fetch_add(g * s[col] * xv * w[j * r + jj]);
                    }
                }
                d_x[t * dim_in + i] = acc;
                i += 1usize;
            }
            #[unroll]
            for j in 0..k {
                let e = idx[t * k + j] as usize;
                let g = gates[t * k + j];
                let mut acc = F::new(0.0_f32);
                #[unroll]
                for jj in 0..r {
                    let col = e * r + jj;
                    let pv = z[t * kk + j * r + jj];
                    acc += s[col] * pv * w[j * r + jj];
                    d_s[col].fetch_add(g * pv * w[j * r + jj]);
                }
                d_gates[t * k + j] = acc;
            }
        }
    }

    /// Raw launch of either the tiled or the scalar forward kernel on the
    /// bare CUDA client (no tensor graph, no autodiff) for the microbench.
    #[allow(clippy::too_many_arguments)]
    fn launch_fwd_raw(
        client: &Client,
        scalar: bool,
        xc: &CubeTensor,
        uc: &CubeTensor,
        vc: &CubeTensor,
        sc: &CubeTensor,
        idxc: &CubeTensor,
        gc: &CubeTensor,
        z: &Handle,
        yc: &CubeTensor,
        b: u32,
        dim_in: u32,
        m: u32,
        dim_out: u32,
        k: u32,
        r: u32,
    ) {
        macro_rules! run {
            ($kernel:ident, $count:expr, $dim:expr) => {
                unsafe {
                    $kernel::launch_unchecked::<f32>(
                        client,
                        $count,
                        $dim,
                        BufferArg::from_raw_parts(xc.handle.clone(), (b * dim_in) as usize),
                        BufferArg::from_raw_parts(uc.handle.clone(), (dim_in * m) as usize),
                        BufferArg::from_raw_parts(vc.handle.clone(), (dim_out * m) as usize),
                        BufferArg::from_raw_parts(sc.handle.clone(), m as usize),
                        BufferArg::from_raw_parts(idxc.handle.clone(), (b * k) as usize),
                        BufferArg::from_raw_parts(gc.handle.clone(), (b * k) as usize),
                        BufferArg::from_raw_parts(z.clone(), (b * k * r) as usize),
                        BufferArg::from_raw_parts(yc.handle.clone(), (b * dim_out) as usize),
                        b,
                        dim_in,
                        m,
                        dim_out,
                        k,
                        r,
                    );
                }
            };
        }
        if scalar {
            run!(
                moe_fwd_kernel_scalar,
                CubeCount::Static(b.div_ceil(64), 1, 1),
                CubeDim::new_3d(64, 1, 1)
            );
        } else {
            run!(
                moe_fwd_kernel,
                CubeCount::Static(b, 1, 1),
                CubeDim::new_3d(32, 1, 1)
            );
        }
    }

    #[test]
    #[ignore = "kernel microbench: run manually with --ignored"]
    fn moe_fused_fwd_kernel_bench() {
        // gate_up-like: in=512, out=4096, k=2, r=16 (k*r=32 <= 64)
        let (b, dim_in, dim_out, m) = (1024usize, 512usize, 4096usize, 256usize);
        let (k, r) = (2usize, 16usize);
        let kk = k * r;
        let dev = Device::cuda(0);
        let x = Tensor::<2>::random([b, dim_in], Distribution::Normal(0.0, 1.0), &dev);
        let u = Tensor::<2>::random([dim_in, m], Distribution::Normal(0.0, 1.0), &dev);
        let v = Tensor::<2>::random([dim_out, m], Distribution::Normal(0.0, 1.0), &dev);
        let s = Tensor::<1>::random([m], Distribution::Normal(1.0, 0.1), &dev);
        let gates = Tensor::<2>::random([b, k], Distribution::Uniform(0.1, 0.9), &dev);
        let idx: Tensor<2, Int> = {
            let mut raw = vec![0i32; b * k];
            for t in 0..b {
                for j in 0..k {
                    raw[t * k + j] = ((t + 3 * j) % (m / r)) as i32;
                }
            }
            Tensor::from_data(burn::tensor::TensorData::new(raw, [b, k]), &dev)
        };
        let xc = cube_of2(&x).unwrap();
        let uc = cube_of2(&u).unwrap();
        let vc = cube_of2(&v).unwrap();
        let sc = cube_of1(&s).unwrap();
        let idxc = cube_int2(&idx).unwrap();
        let gc = cube_of2(&gates).unwrap();
        let client = xc.client.clone();
        let z = client.empty(b * kk * 4);
        let ya = empty_dense([b, dim_out], &dev);
        let yb = empty_dense([b, dim_out], &dev);
        let yac = cube_of2(&ya).unwrap();
        let ybc = cube_of2(&yb).unwrap();

        // correctness: the tiled kernel must be bit-identical to the scalar
        // kernel AND to the production moe_fwd launch (proven against the
        // tensor path by moe_fused_forward_matches_tensor_path); into_data
        // forces real device execution, so equal outputs prove the raw
        // launches actually ran
        let (yp, _st) = moe_fwd(
            &client, &x, &u, &v, &s, &idx, &gates, b, dim_in, m, dim_out, k, r,
        )
        .expect("fwd");
        for _ in 0..2 {
            launch_fwd_raw(
                &client,
                true,
                &xc,
                &uc,
                &vc,
                &sc,
                &idxc,
                &gc,
                &z,
                &yac,
                b as u32,
                dim_in as u32,
                m as u32,
                dim_out as u32,
                k as u32,
                r as u32,
            );
            launch_fwd_raw(
                &client,
                false,
                &xc,
                &uc,
                &vc,
                &sc,
                &idxc,
                &gc,
                &z,
                &ybc,
                b as u32,
                dim_in as u32,
                m as u32,
                dim_out as u32,
                k as u32,
                r as u32,
            );
        }
        let sa = to_host(ya.clone());
        let sb = to_host(yb.clone());
        let sp = to_host(yp);
        let md = maxdiff(&sa, &sb);
        let md_p = maxdiff(&sb, &sp);
        println!("tiled vs scalar fwd maxdiff: {md:.3e}, raw vs production: {md_p:.3e}");
        assert_eq!(md, 0.0, "tiled kernel diverged from the scalar kernel");
        assert_eq!(
            md_p, 0.0,
            "raw launches diverged from the production launch"
        );

        // timing loop: wall-clock here measures host enqueue only (into_data /
        // into_scalar / client.sync are async queueing in this cubecl
        // version); run with CUBECL_DEBUG_OPTION=profile for authoritative
        // per-kernel CUDA-event times
        let iters = 50;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            launch_fwd_raw(
                &client,
                true,
                &xc,
                &uc,
                &vc,
                &sc,
                &idxc,
                &gc,
                &z,
                &yac,
                b as u32,
                dim_in as u32,
                m as u32,
                dim_out as u32,
                k as u32,
                r as u32,
            );
            let _ = ya.clone().slice([0..1, 0..32]).into_data().bytes;
        }
        let ts = t0.elapsed().as_secs_f32() / iters as f32;
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            launch_fwd_raw(
                &client,
                false,
                &xc,
                &uc,
                &vc,
                &sc,
                &idxc,
                &gc,
                &z,
                &ybc,
                b as u32,
                dim_in as u32,
                m as u32,
                dim_out as u32,
                k as u32,
                r as u32,
            );
            let _ = yb.clone().slice([0..1, 0..32]).into_data().bytes;
        }
        let tt = t1.elapsed().as_secs_f32() / iters as f32;
        println!("fwd kernel scalar {ts:.3} ms vs tiled {tt:.3} ms @B={b} in={dim_in} out={dim_out} k={k} r={r} (host-enqueue only; use CUBECL_DEBUG_OPTION=profile for GPU times)");
    }

    /// Raw launch of either the tiled or the scalar backward kernel on the
    /// bare CUDA client (no tensor graph, no autodiff) for the microbench.
    #[allow(clippy::too_many_arguments)]
    fn launch_bwd_raw(
        client: &Client,
        scalar: bool,
        doc: &CubeTensor,
        xc: &CubeTensor,
        uc: &CubeTensor,
        vc: &CubeTensor,
        sc: &CubeTensor,
        idxc: &CubeTensor,
        gc: &CubeTensor,
        z: &Handle,
        dxc: &CubeTensor,
        duc: &CubeTensor,
        dvc: &CubeTensor,
        dsc: &CubeTensor,
        dgc: &CubeTensor,
        b: u32,
        dim_in: u32,
        m: u32,
        dim_out: u32,
        k: u32,
        r: u32,
    ) {
        macro_rules! run {
            ($kernel:ident, $count:expr, $dim:expr) => {
                unsafe {
                    $kernel::launch_unchecked::<f32>(
                        client,
                        $count,
                        $dim,
                        BufferArg::from_raw_parts(doc.handle.clone(), (b * dim_out) as usize),
                        BufferArg::from_raw_parts(xc.handle.clone(), (b * dim_in) as usize),
                        BufferArg::from_raw_parts(uc.handle.clone(), (dim_in * m) as usize),
                        BufferArg::from_raw_parts(vc.handle.clone(), (dim_out * m) as usize),
                        BufferArg::from_raw_parts(sc.handle.clone(), m as usize),
                        BufferArg::from_raw_parts(idxc.handle.clone(), (b * k) as usize),
                        BufferArg::from_raw_parts(gc.handle.clone(), (b * k) as usize),
                        BufferArg::from_raw_parts(z.clone(), (b * k * r) as usize),
                        BufferArg::from_raw_parts(dxc.handle.clone(), (b * dim_in) as usize),
                        BufferArg::from_raw_parts(duc.handle.clone(), (dim_in * m) as usize),
                        BufferArg::from_raw_parts(dvc.handle.clone(), (dim_out * m) as usize),
                        BufferArg::from_raw_parts(dsc.handle.clone(), m as usize),
                        BufferArg::from_raw_parts(dgc.handle.clone(), (b * k) as usize),
                        b,
                        dim_in,
                        m,
                        dim_out,
                        k,
                        r,
                    );
                }
            };
        }
        if scalar {
            run!(
                moe_bwd_kernel_scalar,
                CubeCount::Static(b.div_ceil(64), 1, 1),
                CubeDim::new_3d(64, 1, 1)
            );
        } else {
            run!(
                moe_bwd_kernel,
                CubeCount::Static(b, 1, 1),
                CubeDim::new_3d(32, 1, 1)
            );
        }
    }

    #[test]
    #[ignore = "kernel microbench: run manually with --ignored"]
    fn moe_fused_bwd_kernel_bench() {
        // gate_up-like: in=512, out=4096, k=2, r=16 (k*r=32 <= 64)
        let (b, dim_in, dim_out, m) = (1024usize, 512usize, 4096usize, 256usize);
        let (k, r) = (2usize, 16usize);
        let kk = k * r;
        let dev = Device::cuda(0);
        let x = Tensor::<2>::random([b, dim_in], Distribution::Normal(0.0, 1.0), &dev);
        let u = Tensor::<2>::random([dim_in, m], Distribution::Normal(0.0, 1.0), &dev);
        let v = Tensor::<2>::random([dim_out, m], Distribution::Normal(0.0, 1.0), &dev);
        let s = Tensor::<1>::random([m], Distribution::Normal(1.0, 0.1), &dev);
        let gates = Tensor::<2>::random([b, k], Distribution::Uniform(0.1, 0.9), &dev);
        let d_out = Tensor::<2>::random([b, dim_out], Distribution::Normal(0.0, 1.0), &dev);
        let idx: Tensor<2, Int> = {
            let mut raw = vec![0i32; b * k];
            for t in 0..b {
                for j in 0..k {
                    raw[t * k + j] = ((t + 3 * j) % (m / r)) as i32;
                }
            }
            Tensor::from_data(burn::tensor::TensorData::new(raw, [b, k]), &dev)
        };
        let xc = cube_of2(&x).unwrap();
        let uc = cube_of2(&u).unwrap();
        let vc = cube_of2(&v).unwrap();
        let sc = cube_of1(&s).unwrap();
        let idxc = cube_int2(&idx).unwrap();
        let gc = cube_of2(&gates).unwrap();
        let doc = cube_of2(&d_out).unwrap();
        let client = xc.client.clone();
        let z = client.empty(b * kk * 4);

        // correctness: tiled vs scalar on fresh zeroed buffers; the w
        // reduction reorders FMAs and the atomics are non-deterministic, so
        // the bound is a tight relative tolerance (production-path grad
        // equivalence is covered by moe_fused_forward_matches_tensor_path)
        let dxa = empty_dense([b, dim_in], &dev);
        let dua = zeros_dense([dim_in, m], &dev);
        let dva = zeros_dense([dim_out, m], &dev);
        let dsa = zeros_dense([m], &dev);
        let dga = empty_dense([b, k], &dev);
        let dxb = empty_dense([b, dim_in], &dev);
        let dub = zeros_dense([dim_in, m], &dev);
        let dvb = zeros_dense([dim_out, m], &dev);
        let dsb = zeros_dense([m], &dev);
        let dgb = empty_dense([b, k], &dev);
        let dxac = cube_of2(&dxa).unwrap();
        let duac = cube_of2(&dua).unwrap();
        let dvac = cube_of2(&dva).unwrap();
        let dsac = cube_of1(&dsa).unwrap();
        let dgac = cube_of2(&dga).unwrap();
        let dxbc = cube_of2(&dxb).unwrap();
        let dubc = cube_of2(&dub).unwrap();
        let dvbc = cube_of2(&dvb).unwrap();
        let dsbc = cube_of1(&dsb).unwrap();
        let dgbc = cube_of2(&dgb).unwrap();
        launch_bwd_raw(
            &client,
            true,
            &doc,
            &xc,
            &uc,
            &vc,
            &sc,
            &idxc,
            &gc,
            &z,
            &dxac,
            &duac,
            &dvac,
            &dsac,
            &dgac,
            b as u32,
            dim_in as u32,
            m as u32,
            dim_out as u32,
            k as u32,
            r as u32,
        );
        launch_bwd_raw(
            &client,
            false,
            &doc,
            &xc,
            &uc,
            &vc,
            &sc,
            &idxc,
            &gc,
            &z,
            &dxbc,
            &dubc,
            &dvbc,
            &dsbc,
            &dgbc,
            b as u32,
            dim_in as u32,
            m as u32,
            dim_out as u32,
            k as u32,
            r as u32,
        );
        let dx = relmaxdiff(&to_host(dxa), &to_host(dxb));
        let du = relmaxdiff(&to_host(dua), &to_host(dub));
        let dv = relmaxdiff(&to_host(dva), &to_host(dvb));
        let ds = relmaxdiff(&to_host(dsa), &to_host(dsb));
        let dg = relmaxdiff(&to_host(dga.clone()), &to_host(dgb.clone()));
        println!("tiled vs scalar bwd relmax: dx {dx:.3e}, du {du:.3e}, dv {dv:.3e}, ds {ds:.3e}, dg {dg:.3e}");
        assert!(
            dx < 1e-3 && du < 1e-3 && dv < 1e-3 && ds < 1e-3 && dg < 1e-3,
            "tiled kernel diverged from the scalar kernel: dx {dx:.3e} du {du:.3e} dv {dv:.3e} ds {ds:.3e} dg {dg:.3e}"
        );

        // timing loop: wall-clock here measures host enqueue only (into_data /
        // into_scalar / client.sync are async queueing in this cubecl
        // version); run with CUBECL_DEBUG_OPTION=profile for authoritative
        // per-kernel CUDA-event times
        let iters = 50;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            launch_bwd_raw(
                &client,
                true,
                &doc,
                &xc,
                &uc,
                &vc,
                &sc,
                &idxc,
                &gc,
                &z,
                &dxac,
                &duac,
                &dvac,
                &dsac,
                &dgac,
                b as u32,
                dim_in as u32,
                m as u32,
                dim_out as u32,
                k as u32,
                r as u32,
            );
            let _ = dga.clone().slice([0..1, 0..k]).into_data().bytes;
        }
        let ts = t0.elapsed().as_secs_f32() / iters as f32;
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            launch_bwd_raw(
                &client,
                false,
                &doc,
                &xc,
                &uc,
                &vc,
                &sc,
                &idxc,
                &gc,
                &z,
                &dxbc,
                &dubc,
                &dvbc,
                &dsbc,
                &dgbc,
                b as u32,
                dim_in as u32,
                m as u32,
                dim_out as u32,
                k as u32,
                r as u32,
            );
            let _ = dgb.clone().slice([0..1, 0..k]).into_data().bytes;
        }
        let tb = t1.elapsed().as_secs_f32() / iters as f32;
        println!("bwd kernel scalar {ts:.3} ms vs tiled {tb:.3} ms @B={b} in={dim_in} out={dim_out} k={k} r={r} (host-enqueue only; use CUBECL_DEBUG_OPTION=profile for GPU times)");
    }

    #[test]
    #[ignore = "perf smoke: run manually with --ignored"]
    fn moe_fused_perf_smoke() {
        let dim_in = 512usize;
        let dim_out = 512usize;
        let (c, e, k, r) = (8usize, 8usize, 2usize, 4usize);
        let b = 1024usize;
        Device::cuda(0).seed(99);
        let adev = Device::cuda(0).autodiff();
        let x = Tensor::<2>::random([b, dim_in], Distribution::Normal(0.0, 1.0), &adev);
        let mut mf = SpectralMoE::new(dim_in, dim_out, c, e, k, r, &adev);
        mf.set_fused(true);
        let mut mr = mf.clone();
        mr.set_fused(false);
        // warm up both paths (JIT + allocators); the mean forces the lazy
        // graph to actually execute on the device (both sides fully sync)
        let _: f32 = mf.forward(x.clone()).mean().into_scalar();
        let _: f32 = mr.forward(x.clone()).mean().into_scalar();
        let iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _: f32 = mf.forward(x.clone()).mean().into_scalar();
        }
        let tf = t0.elapsed().as_secs_f32() / iters as f32;
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let _: f32 = mr.forward(x.clone()).mean().into_scalar();
        }
        let tr = t1.elapsed().as_secs_f32() / iters as f32;
        println!("fused {tf:.3} ms vs tensor {tr:.3} ms per forward @B={b}");
    }
}
