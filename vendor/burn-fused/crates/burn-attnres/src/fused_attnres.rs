//! Fused CUDA kernels for Attention Residuals (Kimi K3 §2.2).
//!
//! `depth_attend` is the full AttnRes hot path: RMS-norm per (layer, b, t),
//! score = q·h_l·scale, online softmax over the depth axis, weighted sum —
//! ~8 tensor passes over `[L, B, T, D]` on the tensor path, one launch here
//! (one cube per (b, t), threads split D into slabs, shared-memory tree
//! reductions for the per-layer dot).
//!
//! `source_score` and the online-softmax `merge` collapse the streaming
//! BlockAttnRes step's ~15 launches per layer output to 2.

use burn::tensor::Tensor;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use std::any::Any;
use std::cell::RefCell;

use crate::ScoreForm;

const THREADS: u32 = 256;

/// Layers per chunk in the chunked `depth_attend` (paper's N ≈ 8). Peak
/// memory scales as `(G+2)·B·T·D`; G=8 balances the running-state traffic
/// against the chunk stack size.
pub const CHUNK_G: usize = 8;

// ponytail: single-slot cache for the internal [G,B,T,D] chunk stack plus the
// per-call scores/max/sum state. A fresh CUDA allocation costs ~15ms of
// first-touch page faults (default allocator); none of these buffers escape
// depth_attend_cuda, so one extra allocation per distinct chunk shape and
// device. `out` is the only per-call allocation (it is returned to the caller).
struct ChunkState {
    device: burn::tensor::Device,
    key: (usize, usize, usize, usize),
    chunk: CubeTensor,
    scores: CubeTensor,
    max_s: CubeTensor,
    sum_e: CubeTensor,
}

thread_local! {
    static STATE_CACHE: RefCell<Option<ChunkState>> = const { RefCell::new(None) };
}

fn cached_state(
    g: usize,
    b: usize,
    t: usize,
    d: usize,
    device: &burn::tensor::Device,
) -> ChunkState {
    STATE_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if let Some(s) = &*c {
            if s.key == (g, b, t, d) && s.device == *device {
                return ChunkState {
                    device: s.device.clone(),
                    key: s.key,
                    chunk: s.chunk.clone(),
                    scores: s.scores.clone(),
                    max_s: s.max_s.clone(),
                    sum_e: s.sum_e.clone(),
                };
            }
        }
        let st = ChunkState {
            device: device.clone(),
            key: (g, b, t, d),
            chunk: cube_of(&Tensor::<4>::empty([g, b, t, d], device)).expect("bare CUDA chunk"),
            scores: cube_of(&Tensor::<2>::empty([g, b * t], device)).expect("bare CUDA scores"),
            max_s: cube_of(&Tensor::<2>::empty([b, t], device)).expect("bare CUDA max_s"),
            sum_e: cube_of(&Tensor::<2>::empty([b, t], device)).expect("bare CUDA sum_e"),
        };
        *c = Some(ChunkState {
            device: st.device.clone(),
            key: st.key,
            chunk: st.chunk.clone(),
            scores: st.scores.clone(),
            max_s: st.max_s.clone(),
            sum_e: st.sum_e.clone(),
        });
        st
    })
}

fn cube_of<const D: usize>(t: &Tensor<D>) -> Option<CubeTensor> {
    type B = burn_cubecl::CubeBackend;
    let prim = t.clone().try_into_primitive::<B>().ok()?;
    let c = (&prim as &dyn Any).downcast_ref::<CubeTensor>()?;
    Some(c.clone())
}

fn cube_of_1(t: &Tensor<1>) -> Option<CubeTensor> {
    cube_of(t)
}

/// Chunked Full AttnRes (Kimi K3 §2.2, exact math, bounded memory).
///
/// The history is processed in chunks of `G` layers: per-layer score launches
/// (RMS norm + `q·h` in one read pass, tree reductions) fill the `[G,B,T,D]`
/// chunk stack as a by-product, and one chunk kernel per chunk folds the
/// chunk into a running online-softmax state `(acc, max, sum)` — `acc` keeps
/// every seen layer's contribution in one `[B,T,D]`, so peak memory is
/// `(G+2)·B·T·D` instead of `(L+1)·B·T·D`. The final division recovers the
/// exact full-depth softmax weights (no Block approximation).
#[cube(launch_unchecked)]
fn attnres_scores_kernel<F: Float>(
    h: &[F],          // [B, T, D] one layer
    q: &[F],          // [D]
    scores: &mut [F], // [G, B, T] row `row`
    chunk: &mut [F],  // [G, B, T, D] row `row`
    row: u32,
    scale: f32,
    // Multiplier on `sum(h^2)` inside the norm's root: `1/d` for the paper's
    // RMSNorm (a mean, `ScoreForm::Paper`), `1` for a plain L2 norm
    // (`ScoreForm::SqrtD`). A RUNTIME argument like `scale` on purpose - both
    // are constants of the `ScoreForm`, and a kernel that only knew one of
    // the two forms would compute a different function from the tensor path
    // for the same declared form, with nothing to say so.
    norm_m: f32,
    #[comptime] bt_count: u32, // B * T
    #[comptime] d: u32,
    #[comptime] threads: u32,
    #[comptime] per: u32,
    #[comptime] log_threads: u32,
) {
    let bt = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let bt_count = bt_count as usize;
    let d = d as usize;
    let threads = threads as usize;
    let per = per as usize;
    let lg = log_threads as usize;
    let base = bt * d;

    let mut red = Shared::<[F]>::new_slice(threads);

    let mut psq = F::new(0.0_f32);
    let mut pdot = F::new(0.0_f32);
    for j in 0..per {
        let col = j * threads + tid;
        if col < d {
            let v = h[base + col];
            psq += v * v;
            pdot += q[col] * v;
            chunk[(row as usize) * bt_count * d + base + col] = v;
        }
    }
    red[tid] = psq;
    sync_cube();
    for k in 0..lg {
        let stride = threads >> (k + 1);
        if tid < stride {
            red[tid] = red[tid] + red[tid + stride];
        }
        sync_cube();
    }
    let sq = red[0];
    sync_cube();
    red[tid] = pdot;
    sync_cube();
    for k in 0..lg {
        let stride = threads >> (k + 1);
        if tid < stride {
            red[tid] = red[tid] + red[tid + stride];
        }
        sync_cube();
    }
    let dot = red[0];

    if tid == 0 {
        scores[(row as usize) * bt_count + bt] =
            dot * F::cast_from(scale) / (sq * F::cast_from(norm_m) + F::new(1e-5_f32)).sqrt();
    }
}

/// Fold one chunk into the running online-softmax state.
///
/// Per (b,t): chunk max m_c and exp-sum, running merge
/// m' = max(m, m_c); acc' = acc·e^(m−m') + Σ_g e^(s_g−m')·h_g;
/// sum' = sum·e^(m−m') + sumexp_c·e^(m_c−m'); the last chunk writes
/// out = acc'/clamp(sum', 1e-12). The per-thread state lives in registers and
/// shared memory, and a `sync_cube()` guards the read-then-write of
/// `max_s`/`sum_e` (see the note in `merge_kernel`).
#[cube(launch_unchecked)]
fn attnres_chunk_kernel<F: Float>(
    scores: &[F],    // [G, B, T] current chunk (rows >= ga stale)
    chunk: &[F],     // [G, B, T, D] current chunk
    acc: &mut [F],   // [B, T, D] running weighted sum
    max_s: &mut [F], // [B, T] running max
    sum_e: &mut [F], // [B, T] running exp sum
    out: &mut [F],   // [B, T, D] written on the last chunk
    #[comptime] g: u32,
    #[comptime] ga: u32, // actual layer count of this chunk (tail < G)
    #[comptime] bt_count: u32,
    #[comptime] d: u32,
    #[comptime] threads: u32,
    #[comptime] per: u32,
    #[comptime] first: bool,
    #[comptime] last: bool,
) {
    let bt = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let g = g as usize;
    let bt_count = bt_count as usize;
    let d = d as usize;
    let threads = threads as usize;
    let per = per as usize;
    let base = bt * d;

    // comptime loops with a runtime `ga` guard: the tail chunk reuses the
    // scores/chunk buffers and stale rows above ga must not contribute
    let mut m_c = F::new(-3.0e38_f32);
    for li in 0..g {
        if (li as u32) < ga {
            let s = scores[li * bt_count + bt];
            if s > m_c {
                m_c = s;
            }
        }
    }
    let mut sumexp_c = F::new(0.0_f32);
    for li in 0..g {
        if (li as u32) < ga {
            sumexp_c += (scores[li * bt_count + bt] - m_c).exp();
        }
    }

    // First chunk: cached state buffers may hold stale data; start from the
    // identity values (m = -inf, sum = 0) so they need no per-call fill.
    let m_old = if first {
        F::new(-3.0e38_f32)
    } else {
        max_s[bt]
    };
    let sum_old = if first { F::new(0.0_f32) } else { sum_e[bt] };
    let mut m_new = m_old;
    if m_c > m_old {
        m_new = m_c;
    }
    let rescale = (m_old - m_new).exp();
    let sum_new = sum_old * rescale + sumexp_c * (m_c - m_new).exp();

    let mut ws = Shared::<[F]>::new_slice(g);
    for li in 0..g {
        if (li as u32) < ga {
            ws[li] = (scores[li * bt_count + bt] - m_new).exp();
        } else {
            ws[li] = F::new(0.0_f32);
        }
    }

    for j in 0..per {
        let col = j * threads + tid;
        if col < d {
            let mut a = F::new(0.0_f32);
            for li in 0..g {
                if (li as u32) < ga {
                    a += ws[li] * chunk[li * bt_count * d + base + col];
                }
            }
            let acc_new = if first {
                a
            } else {
                acc[base + col] * rescale + a
            };
            acc[base + col] = acc_new;
            if last {
                let mut den = sum_new;
                if den < F::new(1e-12_f32) {
                    den = F::new(1e-12_f32);
                }
                out[base + col] = acc_new / den;
            }
        }
    }
    // Same barrier requirement as `merge_kernel`: every thread read
    // `max_s[bt]`/`sum_e[bt]` (and filled `ws`) above, only `tid == 0`
    // writes them below. See the note there.
    sync_cube();
    if tid == 0 {
        max_s[bt] = m_new;
        sum_e[bt] = sum_new;
    }
}

/// Copy one `[B, T, D]` layer into `stack[li]` (row-per-cube, no division).
#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn stack_copy_kernel<F: Float>(
    h: &[F],         // [B, T, D]
    stack: &mut [F], // [L, B, T, D] row `li`
    li: u32,
    #[comptime] bt_count: u32,
    #[comptime] d: u32,
    #[comptime] chunks: u32,
) {
    let bt = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize;
    let base = bt * d;
    let row = (li as usize) * (bt_count as usize) * d + base;
    for c in 0..chunks {
        let col = (c as usize) * 256 + tid;
        if col < d {
            stack[row + col] = h[base + col];
        }
    }
}

/// Fused full-depth-attention backward. Per (b, t) cube: recompute the RMS
/// scores over L, softmax, then d_h_l (per-cube row across all L, no race) and
/// the d_q partial (reduced to [D] by the caller).
#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn depth_attend_backward_kernel<F: Float>(
    h: &[F],          // [L, B, T, D] stacked
    q: &[F],          // [D]
    dout: &[F],       // [B, T, D]
    dh: &mut [F],     // [L, B, T, D]
    dqpart: &mut [F], // [B, T, D]
    scale: f32,
    // Multiplier on `sum(h^2)` inside the norm's root: `1/d` for the paper's
    // RMSNorm (a mean, `ScoreForm::Paper`), `1` for a plain L2 norm
    // (`ScoreForm::SqrtD`). A RUNTIME argument like `scale` on purpose - both
    // are constants of the `ScoreForm`, and a kernel that only knew one of
    // the two forms would compute a different function from the tensor path
    // for the same declared form, with nothing to say so.
    norm_m: f32,
    #[comptime] l: u32,
    #[comptime] bt_count: u32,
    #[comptime] d: u32,
    #[comptime] threads: u32,
    #[comptime] per: u32,
    #[comptime] log_threads: u32,
) {
    let bt = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let l = l as usize;
    let bt_count = bt_count as usize;
    let d = d as usize;
    let threads = threads as usize;
    let per = per as usize;
    let lg = log_threads as usize;
    let hbt = bt_count * d;
    let base = bt * d;

    let mut red = Shared::<[F]>::new_slice(threads);
    let mut sq = Shared::<[F]>::new_slice(l);
    let mut qh = Shared::<[F]>::new_slice(l);
    let mut w = Shared::<[F]>::new_slice(l);
    let mut dw = Shared::<[F]>::new_slice(l);
    let mut dq = Shared::<[F]>::new_slice(threads * per);

    // sq_l, qh_l
    for li in 0..l {
        let mut p = F::new(0.0_f32);
        for j in 0..per {
            let col = j * threads + tid;
            if col < d {
                let v = h[li * hbt + base + col];
                p += v * v;
            }
        }
        red[tid] = p;
        sync_cube();
        for k in 0..lg {
            let stride = threads >> (k + 1);
            if tid < stride {
                red[tid] = red[tid] + red[tid + stride];
            }
            sync_cube();
        }
        sq[li] = red[0];
        sync_cube();
        let mut p = F::new(0.0_f32);
        for j in 0..per {
            let col = j * threads + tid;
            if col < d {
                p += q[col] * h[li * hbt + base + col];
            }
        }
        red[tid] = p;
        sync_cube();
        for k in 0..lg {
            let stride = threads >> (k + 1);
            if tid < stride {
                red[tid] = red[tid] + red[tid + stride];
            }
            sync_cube();
        }
        qh[li] = red[0];
        sync_cube();
    }

    // scores + softmax over l
    let mut max_s = F::new(-3.0e38_f32);
    for li in 0..l {
        let s = qh[li] * F::cast_from(scale) / (sq[li] * F::cast_from(norm_m) + F::new(1e-5_f32)).sqrt();
        if s > max_s {
            max_s = s;
        }
    }
    let mut sum_e = F::new(0.0_f32);
    for li in 0..l {
        let s = qh[li] * F::cast_from(scale) / (sq[li] * F::cast_from(norm_m) + F::new(1e-5_f32)).sqrt();
        let e = (s - max_s).exp();
        w[li] = e;
        sum_e += e;
    }
    let inv_sum = F::new(1.0_f32) / sum_e;
    for li in 0..l {
        w[li] *= inv_sum;
    }

    // d_w_l and the softmax-backward sum
    for li in 0..l {
        let mut p = F::new(0.0_f32);
        for j in 0..per {
            let col = j * threads + tid;
            if col < d {
                p += dout[base + col] * h[li * hbt + base + col];
            }
        }
        red[tid] = p;
        sync_cube();
        for k in 0..lg {
            let stride = threads >> (k + 1);
            if tid < stride {
                red[tid] = red[tid] + red[tid + stride];
            }
            sync_cube();
        }
        dw[li] = red[0];
        sync_cube();
    }
    let mut wd_sum = F::new(0.0_f32);
    for li in 0..l {
        wd_sum += w[li] * dw[li];
    }

    // d_h_l + d_q partial
    for j in 0..per {
        dq[tid * per + j] = F::new(0.0_f32);
    }
    for li in 0..l {
        let dsc = w[li] * (dw[li] - wd_sum);
        let inv = F::new(1.0_f32) / (sq[li] * F::cast_from(norm_m) + F::new(1e-5_f32)).sqrt();
        let inv3 = inv * inv * inv;
        let m = F::cast_from(norm_m);
        for j in 0..per {
            let col = j * threads + tid;
            if col < d {
                let hl = h[li * hbt + base + col];
                // d/dh of `scale·(q·h)·inv`, `inv = (m‖h‖²+ε)^(-1/2)`:
                //   `scale·(q_j·inv − m·h_j·(q·h)·inv³)`
                // The `m` on the second term is `d(inv)/dh`'s factor 2m/2 and
                // it is NOT optional: with the L2 norm (`m = 1`) it is a
                // no-op, which is why the formula was written without it and
                // every test stayed green. The paper's RMSNorm has
                // `m = 1/d`, and the finite-difference gate
                // (`depth_attend_grad_matches_finite_difference`) went red at
                // rel 1.47 the moment `ScoreForm::Paper` was wired in.
                let norm =
                    dsc * F::cast_from(scale) * (q[col] * inv - m * hl * qh[li] * inv3);
                dh[li * hbt + base + col] = w[li] * dout[base + col] + norm;
                dq[tid * per + j] += dsc * F::cast_from(scale) * hl * inv;
            }
        }
    }
    for j in 0..per {
        let col = j * threads + tid;
        if col < d {
            dqpart[base + col] = dq[tid * per + j];
        }
    }
}

/// Fused depth-attention backward on the bare CUDA backend.
#[cfg(all(feature = "cuda", any(feature = "autodiff", test)))]
pub fn depth_attend_backward_cuda(
    history: &[Tensor<3>],
    query: &Tensor<1>,
    d_out: &Tensor<3>,
    form: ScoreForm,
) -> Option<(Vec<Tensor<3>>, Tensor<1>)> {
    use burn_cubecl::tensor::CubeTensor;
    type CudaBare = burn_cubecl::CubeBackend;
    let cube = |t: &Tensor<3>| -> Option<CubeTensor> {
        let prim = t.clone().try_into_primitive::<CudaBare>().ok()?;
        let c = (&prim as &dyn std::any::Any)
            .downcast_ref::<CubeTensor>()?;
        Some(c.clone())
    };
    let cube_any = |t: &burn::tensor::Tensor<3, burn::tensor::Float>| -> Option<CubeTensor> {
        let prim = t.clone().try_into_primitive::<CudaBare>().ok()?;
        let c = (&prim as &dyn std::any::Any)
            .downcast_ref::<CubeTensor>()?;
        Some(c.clone())
    };
    let cube1 = |t: &Tensor<1>| -> Option<CubeTensor> {
        let prim = t.clone().try_into_primitive::<CudaBare>().ok()?;
        let c = (&prim as &dyn std::any::Any)
            .downcast_ref::<CubeTensor>()?;
        Some(c.clone())
    };
    let l = history.len();
    let [b, t, d] = history[0].dims();
    let bt = b * t;
    let hc: Vec<CubeTensor> =
        history.iter().map(cube).collect::<Option<_>>()?;
    let qc = cube1(query)?;
    let dc = cube(d_out)?;
    let dev = &history[0].device();
    let stack = Tensor::<4>::empty([l, b, t, d], dev);
    let sc = cube_any(&stack.reshape([l * b * t, 1, d]))?;
    let dh_t = Tensor::<4>::empty([l, b, t, d], dev);
    let dhc = cube_any(&dh_t.reshape([l * b * t, 1, d]))?;
    let dqp_t = Tensor::<3>::empty([b, t, d], dev);
    let dqpc = cube(&dqp_t)?;
    let client = hc[0].client.clone();
    let chunks = (d as u32).div_ceil(256);
    let dim = CubeDim::new_3d(256, 1, 1);
    let count = CubeCount::Static(bt as u32, 1, 1);
    let per = (d as u32).div_ceil(256);
    unsafe {
        for (li, h) in hc.iter().enumerate() {
            stack_copy_kernel::launch_unchecked::<f32>(
                &client,
                count.clone(),
                dim,
                BufferArg::from_raw_parts(h.handle.clone(), bt * d),
                BufferArg::from_raw_parts(sc.handle.clone(), l * bt * d),
                li as u32,
                bt as u32,
                d as u32,
                chunks,
            );
        }
        depth_attend_backward_kernel::launch_unchecked::<f32>(
            &client,
            count,
            dim,
            BufferArg::from_raw_parts(sc.handle.clone(), l * bt * d),
            BufferArg::from_raw_parts(qc.handle, d),
            BufferArg::from_raw_parts(dc.handle, bt * d),
            BufferArg::from_raw_parts(dhc.handle.clone(), l * bt * d),
            BufferArg::from_raw_parts(dqpc.handle.clone(), bt * d),
            form.scale(d),
            form.norm_m(d),
            l as u32,
            bt as u32,
            d as u32,
            256,
            per,
            8,
        );
    }
    // dq = reduce [B,T,D] over (b, t); dh split into per-layer slices
    let dh_full = Tensor::<4>::from_primitive::<CudaBare>(dhc);
    let dq_part = Tensor::<3>::from_primitive::<CudaBare>(dqpc);
    let dq = dq_part.sum_dim(0).sum_dim(1).reshape([d]);
    let dh_full4 = dh_full.reshape([l, b, t, d]);
    let dhs: Vec<Tensor<3>> = (0..l)
        .map(|li| {
            dh_full4
                .clone()
                .slice([li..li + 1, 0..b, 0..t, 0..d])
                .reshape([b, t, d])
        })
        .collect();
    Some((dhs, dq))
}

// ---- dispatch (bare CUDA backend only, callers fall back to tensor path) ----

/// `source_score` (streaming path): RMS-norm + q·src·scale -> [B, T, 1].
#[cube(launch_unchecked)]
fn source_score_kernel<F: Float>(
    h: &[F],       // [B, T, D]
    q: &[F],       // [D]
    out: &mut [F], // [B, T]
    scale: f32,
    // Multiplier on `sum(h^2)` inside the norm's root: `1/d` for the paper's
    // RMSNorm (a mean, `ScoreForm::Paper`), `1` for a plain L2 norm
    // (`ScoreForm::SqrtD`). A RUNTIME argument like `scale` on purpose - both
    // are constants of the `ScoreForm`, and a kernel that only knew one of
    // the two forms would compute a different function from the tensor path
    // for the same declared form, with nothing to say so.
    norm_m: f32,
    #[comptime] d: u32,
    #[comptime] threads: u32,
    #[comptime] per: u32,
    #[comptime] log_threads: u32,
) {
    let bt = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize;
    let threads = threads as usize;
    let per = per as usize;
    let lg = log_threads as usize;
    let base = bt * d;

    let mut red = Shared::<[F]>::new_slice(threads);

    let mut psq = F::new(0.0_f32);
    let mut pdot = F::new(0.0_f32);
    for j in 0..per {
        let col = j * threads + tid;
        if col < d {
            let v = h[base + col];
            psq += v * v;
            pdot += q[col] * v;
        }
    }
    red[tid] = psq;
    sync_cube();
    for k in 0..lg {
        let stride = threads >> (k + 1);
        if tid < stride {
            red[tid] = red[tid] + red[tid + stride];
        }
        sync_cube();
    }
    let sq = red[0];
    sync_cube();
    red[tid] = pdot;
    sync_cube();
    for k in 0..lg {
        let stride = threads >> (k + 1);
        if tid < stride {
            red[tid] = red[tid] + red[tid + stride];
        }
        sync_cube();
    }
    let dot = red[0];

    if tid == 0 {
        out[bt] = dot * F::cast_from(scale) / (sq * F::cast_from(norm_m) + F::new(1e-5_f32)).sqrt();
    }
}

/// Online-softmax merge (streaming path): fold `src` with score `s` into the
/// running state and emit the attended output in one launch.
///
/// m' = max(m, s); acc' = acc·e^(m−m') + src·e^(s−m');
/// sum' = sum·e^(m−m') + e^(s−m'); out = acc'/clamp(sum', 1e-12).
#[cube(launch_unchecked)]
fn merge_kernel<F: Float>(
    acc: &mut [F],     // [B, T, D] in/out
    max_s: &mut [F],   // [B, T] in/out
    sum_exp: &mut [F], // [B, T] in/out
    src: &[F],         // [B, T, D]
    s: &[F],           // [B, T]
    out: &mut [F],     // [B, T, D]
    #[comptime] d: u32,
    #[comptime] threads: u32,
    #[comptime] per: u32,
) {
    let bt = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize;
    let threads = threads as usize;
    let per = per as usize;
    let base = bt * d;

    let m_old = max_s[bt];
    let sval = s[bt];
    let mut m_new = m_old;
    if sval > m_old {
        m_new = sval;
    }
    let rescale = (m_old - m_new).exp();
    let w = (sval - m_new).exp();
    let sum_new = sum_exp[bt] * rescale + w;

    for j in 0..per {
        let col = j * threads + tid;
        if col < d {
            let i = base + col;
            let a = acc[i] * rescale + src[i] * w;
            acc[i] = a;
            let mut den = sum_new;
            if den < F::new(1e-12_f32) {
                den = F::new(1e-12_f32);
            }
            out[i] = a / den;
        }
    }
    // `max_s[bt]`/`sum_exp[bt]` are read by every thread above (into the
    // thread-local `m_old`/`rescale`/`sum_new`) and written by `tid == 0`
    // below. Without this barrier a slower warp can reach the read *after*
    // tid 0's write landed, silently computing rescale = 1.
    sync_cube();
    if tid == 0 {
        max_s[bt] = m_new;
        sum_exp[bt] = sum_new;
    }
}

pub fn depth_attend_cuda(
    history: &[Tensor<3>],
    query: &Tensor<1>,
    form: ScoreForm,
) -> Option<Tensor<3>> {
    let l = history.len();
    if l == 0 {
        return None;
    }
    let [b, t, d] = history[0].dims();
    if d == 0 {
        return None;
    }
    let qc = cube_of(query)?;
    let hc: Vec<CubeTensor> =
        history.iter().map(cube_of).collect::<Option<_>>()?;
    // chunked: peak memory (G+2)*B*T*D instead of (L+1)*B*T*D; the chunk
    // stack is cached (never escapes), out is written once by the last chunk
    let dev = &history[0].device();
    let g = CHUNK_G.min(l);
    // scores/max_s/sum_e are call-internal state, cached across calls like
    // the chunk stack; `out` escapes (returned) so it stays a fresh alloc.
    let st = cached_state(g, b, t, d, dev);
    let out = Tensor::<3>::empty([b, t, d], dev);
    let oc = cube_of(&out)?;
    let client = hc[0].client.clone();
    let per = (d as u32).div_ceil(THREADS);
    let dim = CubeDim::new_3d(THREADS, 1, 1);
    let scale = form.scale(d);
    let norm_m = form.norm_m(d);
    let bt = (b * t) as u32;
    let chunks = l.div_ceil(g);
    unsafe {
        for c in 0..chunks {
            let first = c == 0;
            let last = c == chunks - 1;
            for gi in 0..g {
                let li = c * g + gi;
                if li >= l {
                    break;
                }
                let h = &hc[li];
                attnres_scores_kernel::launch_unchecked::<f32>(
                    &client,
                    CubeCount::Static(bt, 1, 1),
                    dim,
                    BufferArg::from_raw_parts(h.handle.clone(), b * t * d),
                    BufferArg::from_raw_parts(qc.handle.clone(), d),
                    BufferArg::from_raw_parts(st.scores.handle.clone(), g * b * t),
                    BufferArg::from_raw_parts(st.chunk.handle.clone(), g * b * t * d),
                    gi as u32,
                    scale,
                    norm_m,
                    bt,
                    d as u32,
                    THREADS,
                    per,
                    THREADS.ilog2(),
                );
            }
            let ga = g.min(l - c * g);
            attnres_chunk_kernel::launch_unchecked::<f32>(
                &client,
                CubeCount::Static(bt, 1, 1),
                dim,
                BufferArg::from_raw_parts(st.scores.handle.clone(), g * b * t),
                BufferArg::from_raw_parts(st.chunk.handle.clone(), g * b * t * d),
                BufferArg::from_raw_parts(oc.handle.clone(), b * t * d),
                BufferArg::from_raw_parts(st.max_s.handle.clone(), b * t),
                BufferArg::from_raw_parts(st.sum_e.handle.clone(), b * t),
                BufferArg::from_raw_parts(oc.handle.clone(), b * t * d),
                g as u32,
                ga as u32,
                bt,
                d as u32,
                THREADS,
                per,
                first,
                last,
            );
        }
    }
    Some(out)
}

pub fn source_score_cuda(
    query: &Tensor<1>,
    src: &Tensor<3>,
    form: ScoreForm,
) -> Option<Tensor<3>> {
    let [b, t, d] = src.dims();
    if d == 0 {
        return None;
    }
    let qc = cube_of_1(query)?;
    let sc = cube_of(src)?;
    let out = Tensor::<3>::zeros([b, t, 1], &src.device());
    let oc = cube_of(&out)?;
    let client = sc.client.clone();
    let per = (d as u32).div_ceil(THREADS);
    let dim = CubeDim::new_3d(THREADS, 1, 1);
    let scale = form.scale(d);
    let norm_m = form.norm_m(d);
    unsafe {
        source_score_kernel::launch_unchecked::<f32>(
            &client,
            CubeCount::Static((b * t) as u32, 1, 1),
            dim,
            BufferArg::from_raw_parts(sc.handle, b * t * d),
            BufferArg::from_raw_parts(qc.handle, d),
            BufferArg::from_raw_parts(oc.handle, b * t),
            scale,
            norm_m,
            d as u32,
            THREADS,
            per,
            THREADS.ilog2(),
        );
    }
    Some(out)
}

/// Folds `src` into the online state and returns the attended output.
pub fn merge_cuda(
    acc: &mut Tensor<3>,
    max_s: &mut Tensor<3>,
    sum_exp: &mut Tensor<3>,
    src: &Tensor<3>,
    s: &Tensor<3>,
) -> Option<Tensor<3>> {
    let [b, t, d] = src.dims();
    if d == 0 {
        return None;
    }
    let accc = cube_of(acc)?;
    let mx = cube_of(max_s)?;
    let se = cube_of(sum_exp)?;
    let sc = cube_of(src)?;
    let s1 = cube_of(s)?;
    let out = Tensor::<3>::zeros([b, t, d], &src.device());
    let oc = cube_of(&out)?;
    let client = accc.client.clone();
    let per = (d as u32).div_ceil(THREADS);
    let dim = CubeDim::new_3d(THREADS, 1, 1);
    unsafe {
        merge_kernel::launch_unchecked::<f32>(
            &client,
            CubeCount::Static((b * t) as u32, 1, 1),
            dim,
            BufferArg::from_raw_parts(accc.handle, b * t * d),
            BufferArg::from_raw_parts(mx.handle, b * t),
            BufferArg::from_raw_parts(se.handle, b * t),
            BufferArg::from_raw_parts(sc.handle, b * t * d),
            BufferArg::from_raw_parts(s1.handle, b * t),
            BufferArg::from_raw_parts(oc.handle, b * t * d),
            d as u32,
            THREADS,
            per,
        );
    }
    Some(out)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::{depth_attend, depth_attend_form, BlockAttnRes, ScoreForm};
    use burn::module::{Param, ParamId};
    use burn::tensor::{activation, Device, Distribution, Tensor};

    fn to_host<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn stack(history: &[Tensor<3>]) -> Tensor<4> {
        let stacked: Vec<Tensor<4>> = history
            .iter()
            .map(|h| h.clone().unsqueeze_dim::<4>(0))
            .collect();
        Tensor::cat(stacked, 0)
    }

    fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Raw-op depth_attend on the same device as the fused call, in the
    /// declared `ScoreForm`. The reference used to hard-code `d^-0.5` and an
    /// L2 norm, which made every parity test below blind to a divergence from
    /// Eq. 2: kernel and reference were the same transcription of the wrong
    /// formula, so they agreed. It now takes the form as an argument, and the
    /// tests run BOTH forms, so a kernel that ignored `norm_m` or `scale`
    /// would be red rather than consistently wrong.
    fn ref_depth_attend(
        history: &[Tensor<3>],
        query: &Tensor<1>,
        form: ScoreForm,
    ) -> Tensor<3> {
        let n = history.len();
        let [b, t, d] = history[0].dims();
        let scale = form.scale(d);
        let h_stack = stack(history);
        let h_norm_sq = h_stack
            .clone()
            .powf_scalar(2.0)
            .sum_dim(3)
            .mul_scalar(form.norm_m(d))
            .add_scalar(1e-5);
        let h_norm = h_stack.clone() / h_norm_sq.sqrt().reshape([n, b, t, 1usize]);
        let q = query.clone().reshape([1, 1, 1, d]);
        let scores = (q * h_norm).sum_dim(3).mul_scalar(scale);
        let weights = activation::softmax(scores, 0);
        h_stack
            .mul(weights.reshape([n, b, t, 1usize]))
            .sum_dim(0)
            .reshape([b, t, d])
    }

    #[test]
    fn depth_attend_fused_matches_tensor() {
        let cdev = Device::default();
        for (l, b, t, d) in [
            (12usize, 2usize, 8usize, 512usize),
            (3, 1, 4, 32),
            (40, 2, 4, 2048),
        ] {
            let hist: Vec<Tensor<3>> = (0..l)
                .map(|_| Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cdev))
                .collect();
            let q = Tensor::<1>::random([d], Distribution::Normal(0.0, 1.0), &cdev);
            // BOTH forms, not one: the two conventions give outputs ~d apart
            // in logit temperature, so a kernel that quietly applied the
            // wrong one cannot pass both.
            for form in [ScoreForm::Paper, ScoreForm::SqrtD] {
                let expected = to_host(ref_depth_attend(&hist, &q, form));
                assert!(
                    depth_attend_cuda(&hist, &q, form).is_some(),
                    "fused dispatch should engage on bare CUDA"
                );
                let got = to_host(depth_attend_form(&hist, q.clone(), form));
                let md = maxdiff(&got, &expected);
                assert!(md < 1e-4, "[{l},{b},{t},{d}] {form:?} maxdiff {md}");
            }
        }
    }

    #[test]
    fn source_score_fused_matches_tensor() {
        let cdev = Device::default();
        let (b, t, d) = (2usize, 8usize, 256usize);
        let src = Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cdev);
        let q = Tensor::<1>::random([d], Distribution::Normal(0.0, 1.0), &cdev);
        for form in [ScoreForm::Paper, ScoreForm::SqrtD] {
            let norm = src.clone()
                / src
                    .clone()
                    .powf_scalar(2.0)
                    .sum_dim(2)
                    .mul_scalar(form.norm_m(d))
                    .add_scalar(1e-5)
                    .sqrt();
            let expected = to_host(
                (q.clone().reshape([1, 1, d]) * norm)
                    .sum_dim(2)
                    .mul_scalar(form.scale(d)),
            );
            let got = source_score_cuda(&q, &src, form).expect("kernel");
            let md = maxdiff(&to_host(got), &expected);
            assert!(md < 1e-4, "source_score {form:?} maxdiff {md}");
        }
    }

    #[test]
    fn merge_fused_matches_tensor() {
        let cdev = Device::default();
        let (b, t, d) = (2usize, 8usize, 128usize);
        let acc = Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cdev);
        let max_s = Tensor::<3>::random([b, t, 1], Distribution::Normal(0.0, 1.0), &cdev);
        let sum_e = Tensor::<3>::random([b, t, 1], Distribution::Uniform(0.5, 1.5), &cdev);
        let src = Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cdev);
        let s = Tensor::<3>::random([b, t, 1], Distribution::Normal(0.0, 1.0), &cdev);
        // raw-op reference: m' = max(m,s); rescale; out = acc'/clamp(sum',1e-12)
        let m_new = (max_s.clone() + s.clone() + (max_s.clone() - s.clone()).abs()).div_scalar(2.0);
        let rescale = (max_s.clone() - m_new.clone()).exp();
        let w = (s.clone() - m_new.clone()).exp();
        let acc_r = acc.clone() * rescale.clone() + src.clone() * w.clone();
        let sum_r = sum_e.clone() * rescale + w;
        let expected = to_host(acc_r.clone() / sum_r.clamp_min(1e-12));

        let mut acc_c = acc.clone();
        let mut mx = max_s.clone();
        let mut se = sum_e.clone();
        let got = merge_cuda(&mut acc_c, &mut mx, &mut se, &src, &s).expect("kernel");
        let md = maxdiff(&to_host(got), &expected);
        assert!(md < 1e-4, "merge out maxdiff {md}");
        let md = maxdiff(&to_host(acc_c), &to_host(acc_r));
        assert!(md < 1e-4, "acc state mismatch {md}");
    }

    #[test]
    fn streaming_fused_matches_tensor_path() {
        // CPU (burn-cpu, tensor path) vs CUDA (fused kernels): the same module
        // semantics must agree at every step. The comparison is on the online
        // state as well as the output: `acc`/`max_score`/`sum_exp` diverge
        // first if the kernel's in/out buffers are the problem, and the output
        // diverges with the state intact if the kernel's maths is — the panic
        // message names which, so a red run says where to look.
        let cdev = Device::default();
        let cpu_dev = Device::cpu();
        let (b, t, d, bs) = (1usize, 3usize, 64usize, 2usize);
        let q_cpu = Tensor::<1>::random([d], Distribution::Normal(0.0, 1.0), &cpu_dev);
        let mut cpu = BlockAttnRes::new(d, bs, &cpu_dev);
        cpu.query = Param::initialized(ParamId::new(), q_cpu.clone());
        let mut cuda = BlockAttnRes::new(d, bs, &cdev);
        cuda.query = Param::initialized(
            ParamId::new(),
            Tensor::<1>::from_data(q_cpu.clone().into_data(), &cdev),
        );

        let mut st_cpu = cpu.init_state(b, t, d, &cpu_dev);
        let mut st_cuda = cuda.init_state(b, t, d, &cdev);
        for step in 0..8 {
            let h = Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cpu_dev);
            let out_cpu = cpu.step(h.clone(), &mut st_cpu);
            let hc = Tensor::<3>::from_data(h.into_data(), &cdev);
            let out_cuda = cuda.step(hc, &mut st_cuda);
            let md = maxdiff(&to_host(out_cuda.clone()), &to_host(out_cpu));
            let md_acc = maxdiff(&to_host(st_cuda.acc.clone()), &to_host(st_cpu.acc.clone()));
            let md_ms = maxdiff(&to_host(st_cuda.max_score.clone()), &to_host(st_cpu.max_score.clone()));
            let md_se = maxdiff(&to_host(st_cuda.sum_exp.clone()), &to_host(st_cpu.sum_exp.clone()));
            assert!(
                md < 1e-4 && md_acc < 1e-4 && md_ms < 1e-4 && md_se < 1e-4,
                "step {step}: out {md} acc {md_acc} max_score {md_ms} sum_exp {md_se} \
                 (started {} vs {})",
                st_cuda.started,
                st_cpu.started
            );
        }
    }

    /// `cube_of` (and every `BufferArg::from_raw_parts` below it) takes the
    /// raw allocation handle and a length — it does **not** read the tensor's
    /// strides or offset. A strided view (`permute`, `reshape().permute()`,
    /// `slice` on the cubecl backend) therefore hands the kernel the wrong
    /// memory while looking like a perfectly ordinary `[B,T,D]` tensor. Every
    /// other test here feeds `Tensor::random`/`from_data`/elementwise results,
    /// which are contiguous, so none of them can see this. `#[ignore]`d as a
    /// probe: it is a claim to be falsified on a free GPU, not a green test.
    #[test]
    #[ignore]
    fn source_score_noncontiguous_matches_tensor() {
        let cdev = Device::default();
        let (b, t, d) = (2usize, 8usize, 64usize);
        // [B, D, T] view of a [B, T, D] allocation: same shape as the fused
        // kernels expect, non-row-major strides.
        let view = Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cdev)
            .permute([0, 2, 1]);
        let q = Tensor::<1>::random([d], Distribution::Normal(0.0, 1.0), &cdev);
        let expected = to_host(crate::source_score(&q, &view, ScoreForm::Paper));
        let md = maxdiff(
            &to_host(source_score_cuda(&q, &view, ScoreForm::Paper).unwrap()),
            &expected,
        );
        assert!(md < 1e-4, "strided input: maxdiff {md}");
    }

    /// The online-softmax state machine, over a chain of merges.
    ///
    /// `merge_cuda` mutates the caller's `acc`/`max_score`/`sum_exp` **in
    /// place** through a raw `BufferArg`, so its only proof that the write
    /// landed is a read-back. `merge_fused_matches_tensor` checks `out` and
    /// `acc` once; this chains 64 merges per shape and checks all four values
    /// at every step against a plain-host reference evaluated from the
    /// PRE-merge state (read before the launch, never a tensor handed across
    /// backends), which is what a single merge cannot tell you.
    ///
    /// Honest scope: this does NOT reliably catch the missing-barrier race in
    /// `merge_kernel` — that race is timing-dependent, and this test passes
    /// with the barrier deleted (measured 2026-09-29). It is the coverage, not
    /// the gate, for that defect; the gate that failed is
    /// `streaming_fused_matches_tensor_path`, which is what found it.
    #[test]
    fn merge_state_writeback_matches_host_reference() {
        let cdev = Device::default();
        let (b, t) = (1usize, 3usize);
        for d in [64usize, 128, 256] {
            let q = Tensor::<1>::random([d], Distribution::Normal(0.0, 1.0), &cdev);
            let mut acc = Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cdev);
            let mut mx = Tensor::<3>::ones([b, t, 1], &cdev).add_scalar(0.3);
            let mut se = Tensor::<3>::ones([b, t, 1], &cdev).mul_scalar(1.7);
            let bt = b * t;
            for i in 0..64 {
                let src = Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cdev);
                let s = source_score_cuda(&q, &src, ScoreForm::Paper).unwrap();
                // The PRE-merge state and inputs, on the host, before the launch.
                let f = to_host(acc.clone());
                let m_old = to_host(mx.clone());
                let se_old = to_host(se.clone());
                let sv = to_host(s.clone());
                let srcv = to_host(src.clone());
                let got = to_host(merge_cuda(&mut acc, &mut mx, &mut se, &src, &s).unwrap());
                // Reference in plain host f32 off those pre-merge values, so a
                // stale in-kernel read of the state shows up as a state
                // divergence and not only as a wrong output. No tensor crosses
                // backends: every value is read back once, so the reference
                // cannot itself be the racy thing.
                let mut want_acc = vec![0f32; bt * d];
                let mut want_out = vec![0f32; bt * d];
                let mut want_m = vec![0f32; bt];
                let mut want_se = vec![0f32; bt];
                for p in 0..bt {
                    let m_new = (m_old[p] + sv[p] + (m_old[p] - sv[p]).abs()) / 2.0;
                    let rescale = (m_old[p] - m_new).exp();
                    let w = (sv[p] - m_new).exp();
                    want_m[p] = m_new;
                    want_se[p] = se_old[p] * rescale + w;
                    for c in 0..d {
                        let a = f[p * d + c] * rescale + srcv[p * d + c] * w;
                        want_acc[p * d + c] = a;
                        want_out[p * d + c] = a / want_se[p].max(1e-12);
                    }
                }
                // state first: the state write is the root, the output the symptom
                let md = maxdiff(&to_host(mx.clone()), &want_m);
                assert!(md < 1e-5, "d{d} merge {i}: max_score state maxdiff {md}");
                let md = maxdiff(&to_host(se.clone()), &want_se);
                let se_scale = want_se.iter().fold(1.0f32, |m, v| m.max(v.abs()));
                assert!(
                    md < 1e-5 * se_scale,
                    "d{d} merge {i}: sum_exp state maxdiff {md:.3e} at |sum_exp| ~{se_scale:.3}"
                );
                let md = maxdiff(&to_host(acc.clone()), &want_acc);
                // SCALE-RELATIVE, and this is a measured correction. The
                // tolerance was `1e-5` ABSOLUTE on `acc` after 64 chained f32
                // merges, and `acc`'s scale depends on the score convention:
                // the paper's unscaled logit is ~sqrt(d) times larger, so the
                // online-softmax weights are sharper and the accumulator lands
                // at |acc| ~ 3-4 where the old form drifted less. Measured on
                // this box, identical code, 4 runs: 3 red at 1.0-1.7e-5 and 1
                // green, i.e. 2.5e-6 RELATIVE - about 21x f32 epsilon after
                // 64 accumulations, which is the arithmetic, not the kernel.
                // The absolute form was a coin flip on the random draw
                // (unseeded, ADR-0021). The defect this cell exists for is the
                // missing-barrier race, whose signature is `rescale = 1` and
                // therefore O(1) - four orders above this bound.
                let scale = want_acc.iter().fold(1.0f32, |m, v| m.max(v.abs()));
                assert!(
                    md < 1e-5 * scale,
                    "d{d} merge {i}: acc state maxdiff {md:.3e} at |acc| ~{scale:.3} \
                     (bound {:e})",
                    1e-5 * scale
                );
                let md = maxdiff(&got, &want_out);
                let out_scale = want_out.iter().fold(1.0f32, |m, v| m.max(v.abs()));
                assert!(
                    md < 1e-5 * out_scale,
                    "d{d} merge {i}: out maxdiff {md:.3e} at |out| ~{out_scale:.3}"
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn attnres_backward_bench() {
        let cdev = Device::default();
        let (l, b, t, d) = (24usize, 1usize, 2048usize, 4096usize);
        let hist: Vec<Tensor<3>> = (0..l)
            .map(|_| Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cdev))
            .collect();
        let q = Tensor::<1>::random([d], Distribution::Normal(0.0, 1.0), &cdev);
        let dout = Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cdev);
        for _ in 0..2 {
            let _ = depth_attend_backward_cuda(&hist, &q, &dout, ScoreForm::Paper).unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            let r = depth_attend_backward_cuda(&hist, &q, &dout, ScoreForm::Paper).unwrap();
            let _: f32 = r.0[0].clone().sum().into_scalar();
        }
        let tf = t0.elapsed() / 10;
        let t0 = std::time::Instant::now();
        for _ in 0..3 {
            let (dhs, dq) =
                crate::fused_attnres::ad::depth_attend_backward_tensor(&hist, &q, &dout, ScoreForm::Paper);
            let _: f32 = (dhs[0].clone().sum() + dq.clone().sum()).into_scalar();
        }
        let tt = t0.elapsed() / 3;
        println!(
            "[L{l} b{b} t{t} d{d}] fused bwd {:?} tensor bwd {:?} ({:.1}x)",
            tf,
            tt,
            tt.as_secs_f64() / tf.as_secs_f64()
        );
    }

    #[test]
    #[ignore]
    fn attnres_bench() {
        let cdev = Device::default();
        for (l, b, t, d) in [(24usize, 1usize, 2048usize, 4096usize), (8, 2, 2048, 5120)] {
            let hist: Vec<Tensor<3>> = (0..l)
                .map(|_| Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cdev))
                .collect();
            let q = Tensor::<1>::random([d], Distribution::Normal(0.0, 1.0), &cdev);
            for _ in 0..3 {
                let _ = depth_attend(&hist, q.clone());
            }
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                let r = depth_attend(&hist, q.clone());
                let _: f32 = r.clone().sum().into_scalar(); // flush
            }
            let tf = t0.elapsed() / 10;
            // probe: 24 trivial launches (one per layer) to isolate launch overhead
            let hc3: Vec<_> = hist.iter().map(cube_of).collect::<Option<_>>().unwrap();
            let client3 = hc3[0].client.clone();
            let qc = cube_of(&q).unwrap();
            let nidle = Tensor::<1>::zeros([l * b * t], &cdev);
            let nc = cube_of(&nidle).unwrap();
            let dim3 = CubeDim::new_3d(THREADS, 1, 1);
            let per = (d as u32).div_ceil(THREADS);
            for _ in 0..2 {
                for h in &hc3 {
                    unsafe {
                        attnres_scores_kernel::launch_unchecked::<f32>(
                            &client3,
                            CubeCount::Static((b * t) as u32, 1, 1),
                            dim3,
                            BufferArg::from_raw_parts(h.handle.clone(), b * t * d),
                            BufferArg::from_raw_parts(qc.handle.clone(), d),
                            BufferArg::from_raw_parts(nc.handle.clone(), l * b * t),
                            BufferArg::from_raw_parts(nc.handle.clone(), l * b * t * d),
                            0u32,
                            ScoreForm::Paper.scale(d),
                            ScoreForm::Paper.norm_m(d),
                            (b * t) as u32,
                            d as u32,
                            THREADS,
                            per,
                            THREADS.ilog2(),
                        );
                    }
                }
                let _: f32 = nidle.clone().sum().into_scalar();
            }
            let t0 = std::time::Instant::now();
            for h in &hc3 {
                unsafe {
                    attnres_scores_kernel::launch_unchecked::<f32>(
                        &client3,
                        CubeCount::Static((b * t) as u32, 1, 1),
                        dim3,
                        BufferArg::from_raw_parts(h.handle.clone(), b * t * d),
                        BufferArg::from_raw_parts(qc.handle.clone(), d),
                        BufferArg::from_raw_parts(nc.handle.clone(), l * b * t),
                        BufferArg::from_raw_parts(nc.handle.clone(), l * b * t * d),
                        0u32,
                        ScoreForm::Paper.scale(d),
                        ScoreForm::Paper.norm_m(d),
                        (b * t) as u32,
                        d as u32,
                        THREADS,
                        per,
                        THREADS.ilog2(),
                    );
                }
            }
            let _: f32 = nidle.clone().sum().into_scalar();
            println!("    {} scores launches+work {:?}", l, t0.elapsed());
            let hs = stack(&hist);
            let t0 = std::time::Instant::now();
            for _ in 0..3 {
                let hn = hs.clone().powf_scalar(2.0).sum_dim(3).add_scalar(1e-5);
                let hnm = hs.clone() / hn.sqrt().reshape([l, b, t, 1usize]);
                let sc = (q.clone().reshape([1, 1, 1, d]) * hnm)
                    .sum_dim(3)
                    .mul_scalar(ScoreForm::Paper.scale(d));
                let w = activation::softmax(sc, 0);
                let r = hs
                    .clone()
                    .mul(w.reshape([l, b, t, 1usize]))
                    .sum_dim(0)
                    .reshape([b, t, d]);
                let _: f32 = r.clone().sum().into_scalar(); // flush async queue
            }
            let tt = t0.elapsed() / 3;
            println!(
                "[L{l} b{b} t{t} d{d}] fused {:?} tensor {:?} ({:.1}x)",
                tf,
                tt,
                tt.as_secs_f64() / tf.as_secs_f64()
            );
        }
    }
}

// ---- seam counters (ADR-0019) ----
//
// ENTRY is incremented AFTER the strategy downcasts — the gate that was
// hardcoded to `NoCheckpointing`, so dormouse's
// `Autodiff<CudaBare, BalancedCheckpointing>` never got here. A counter
// before that gate would count interest, not arrivals (`f737710`).
#[cfg(any(feature = "autodiff", test))]
static ENTRY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "cuda")]
static FWD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "cuda")]
static BWD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(entry, fused_forward, fused_backward)` since [`reset_seam_counts`].
pub fn seam_counts() -> Option<(u64, u64, u64)> {
    #[cfg(feature = "autodiff")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        #[cfg(feature = "cuda")]
        return Some((ENTRY.load(Relaxed), FWD.load(Relaxed), BWD.load(Relaxed)));
        #[cfg(not(feature = "cuda"))]
        return Some((ENTRY.load(Relaxed), 0, 0));
    }
    #[cfg(not(feature = "autodiff"))]
    None
}

/// Zero the seam counters.
pub fn reset_seam_counts() {
    #[cfg(feature = "autodiff")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        ENTRY.store(0, Relaxed);
        #[cfg(feature = "cuda")]
        {
            FWD.store(0, Relaxed);
            BWD.store(0, Relaxed);
        }
    }
}

#[cfg(any(feature = "autodiff", test))]
fn note_entry_reached() {
    ENTRY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "cuda")]
fn note_fused_forward() {
    FWD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "cuda")]
fn note_fused_backward() {
    BWD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(any(feature = "autodiff", test))]
#[allow(dead_code)] // items are wired by the autodiff dispatch / ad tests
mod ad {
    use burn::backend::{Backend, DispatchKindConversion};
    use burn::tensor::{DispatchTensor, Tensor};
    use burn_autodiff::checkpoint::base::Checkpointer;
    use burn_autodiff::checkpoint::strategy::{CheckpointStrategy, NoCheckpointing};
    use burn_autodiff::grads::Gradients;
    use burn_autodiff::ops::{Backward, Ops, OpsKind};
    use burn_autodiff::Autodiff;
    use crate::ScoreForm;
    #[cfg(any(feature = "autodiff", test))]
    use crate::fused_attnres::note_entry_reached;
    #[cfg(feature = "cuda")]
    use crate::fused_attnres::{note_fused_backward, note_fused_forward};

    #[derive(Debug)]
    struct AttnResOp;

    impl<B: Backend, const N: usize> Backward<B, N> for AttnResOp
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        // L (number of history layers) AND the score convention: the backward
        // has to differentiate the SAME function the forward computed, and a
        // fused backward that re-derived `d^-0.5` internally while the forward
        // ran the paper's unscaled score would be a correct derivative of the
        // wrong function.
        type State = (usize, ScoreForm);

        fn backward(
            self,
            ops: Ops<Self::State, N>,
            grads: &mut Gradients,
            checkpointer: &mut Checkpointer,
        ) {
            let (l, form) = ops.state;
            let node = |i: usize| ops.parents[i].as_ref().expect("attnres input checkpointed");
            let q = Tensor::<1>::from_primitive::<B>(checkpointer.retrieve_node_output(node(l).id));
            let mut hs: Vec<Tensor<3>> = Vec::with_capacity(l);
            for i in 0..l {
                hs.push(Tensor::from_primitive::<B>(
                    checkpointer.retrieve_node_output(node(i).id),
                ));
            }
            let d_out = Tensor::from_primitive::<B>(grads.consume::<B>(&ops.node));
            #[cfg(feature = "cuda")]
            {
                type CudaBare = burn_cubecl::CubeBackend;
                // `B` is the INNER backend here (`OpsPrep::finish` returns
                // `AutodiffTensor<B>`; the result goes to
                // `Tensor::from_primitive::<Autodiff<Inner, S>>`), so this
                // gate asks the right question. The entry below was the one
                // pinned to `NoCheckpointing`.
                if std::any::TypeId::of::<B>() == std::any::TypeId::of::<CudaBare>() {
                    if let Some((dhs, dq)) = super::depth_attend_backward_cuda(&hs, &q, &d_out, form) {
                        note_fused_backward();
                        for (i, dh) in dhs.into_iter().enumerate() {
                            grads.register::<B>(
                                ops.parents[i].clone().unwrap().id,
                                dh.try_into_primitive::<B>().unwrap(),
                            );
                        }
                        grads.register::<B>(
                            ops.parents[l].clone().unwrap().id,
                            dq.try_into_primitive::<B>().unwrap(),
                        );
                        return;
                    }
                }
            }
            let (dhs, dq) = depth_attend_backward_tensor(&hs, &q, &d_out, form);
            for (i, dh) in dhs.into_iter().enumerate() {
                grads.register::<B>(
                    ops.parents[i].clone().unwrap().id,
                    dh.try_into_primitive::<B>().unwrap(),
                );
            }
            grads.register::<B>(
                ops.parents[l].clone().unwrap().id,
                dq.try_into_primitive::<B>().unwrap(),
            );
        }
    }

    /// Exact full-attention backward over the depth axis (tensor path).
    /// scores_l = q·h_l·scale/√(m·Σh²+ε); w = softmax(scores, 0);
    /// out = Σ w_l·h_l. Returns (d_h per layer, d_q).
    pub fn depth_attend_backward_tensor(
        history: &[Tensor<3>],
        query: &Tensor<1>,
        d_out: &Tensor<3>,
        form: ScoreForm,
    ) -> (Vec<Tensor<3>>, Tensor<1>) {
        let l = history.len();
        let [b, t, d] = history[0].dims();
        let scale = form.scale(d);
        let m_norm = form.norm_m(d);
        let h_stack = crate::fused_attnres::stack_ad(history);
        let q = query.clone().reshape([1, 1, 1, d]);
        let s2 = h_stack
            .clone()
            .powf_scalar(2.0)
            .sum_dim(3)
            .mul_scalar(m_norm)
            .add_scalar(1e-5); // [L,B,T,1]
        let inv = s2.clone().powf_scalar(-0.5);
        let scores = (q.clone() * h_stack.clone())
            .sum_dim(3)
            .mul(inv.clone()) // [L,B,T,1]
            .mul_scalar(scale);
        let w = burn::tensor::activation::softmax(scores.squeeze_dim::<3>(3), 0); // [L,B,T]

        // d_w_l = Σ_d d_out·h_l; d_scores = w·(d_w − Σ_l w·d_w)
        let d_out4 = d_out.clone().unsqueeze_dim::<4>(0); // [1,B,T,D]
        let d_w = (d_out4.clone() * h_stack.clone())
            .sum_dim(3)
            .squeeze_dim::<3>(3); // [L,B,T]
        let d_scores = w.clone() * (d_w.clone() - (w.clone() * d_w).sum_dim(0));

        // d_h = w_l·d_out + scale·d_scores·(q/√s − m·h_l·(q·h_l)/s^(3/2))
        // The `m` is the same factor as in the fused kernel above: with the
        // L2 norm (`m = 1`) it vanishes, which is exactly why it was missing
        // from both implementations until `ScoreForm::Paper` (m = 1/d) made
        // the finite-difference gate red.
        let d_scores4 = d_scores.unsqueeze_dim::<4>(3); // [L,B,T,1]
        let qh = (q.clone() * h_stack.clone()).sum_dim(3); // [L,B,T,1]
        let dh_attn = w.unsqueeze_dim::<4>(3) * d_out4.clone(); // [L,B,T,D]
        let dh_norm = d_scores4.clone()
            * (q.clone() * inv.clone()
                - h_stack.clone() * qh * s2.clone().powf_scalar(-1.5) * m_norm)
            * scale;
        let dh_stack = dh_attn + dh_norm;

        // d_q = Σ_l d_scores·scale·h_l/√s
        let dq = (d_scores4 * h_stack * inv * scale)
            .sum_dim(0)
            .sum_dim(1)
            .sum_dim(2)
            .reshape([d]);

        let dhs: Vec<Tensor<3>> = (0..l)
            .map(|i| {
                dh_stack
                    .clone()
                    .slice([i..i + 1, 0..b, 0..t, 0..d])
                    .reshape([b, t, d])
            })
            .collect();
        (dhs, dq)
    }

    /// Fused chunked depth_attend with exact backward on `Autodiff<Inner>`.
    pub fn depth_attend_autodiff_s<Inner: Backend, S: CheckpointStrategy, const N: usize>(
        history: &[Tensor<3>],
        query: Tensor<1>,
        form: ScoreForm,
    ) -> Option<Tensor<3>>
    where
        DispatchTensor: DispatchKindConversion<Autodiff<Inner, S>> + DispatchKindConversion<Inner>,
    {
        let l = history.len();
        if l + 1 > N {
            return None;
        }
        let qa = query.try_into_primitive::<Autodiff<Inner, S>>().ok()?;
        let has: Vec<_> = history
            .iter()
            .map(|h| h.clone().try_into_primitive::<Autodiff<Inner, S>>().ok())
            .collect::<Option<_>>()?;
        note_entry_reached();
        let q_t = Tensor::<1>::from_primitive::<Inner>(qa.primitive().clone());
        let hs_t: Vec<Tensor<3>> = has
            .iter()
            .map(|h| Tensor::<3>::from_primitive::<Inner>(h.primitive().clone()))
            .collect();

        let out_t = {
            #[cfg(feature = "cuda")]
            {
                type CudaBare = burn_cubecl::CubeBackend;
                if std::any::TypeId::of::<Inner>() == std::any::TypeId::of::<CudaBare>() {
                    if let Some(o) = super::depth_attend_cuda(&hs_t, &q_t, form) {
                        note_fused_forward();
                        o
                    } else {
                        super::depth_attend_tensor_ad(&hs_t, q_t, form)
                    }
                } else {
                    super::depth_attend_tensor_ad(&hs_t, q_t, form)
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                super::depth_attend_tensor_ad(&hs_t, q_t, form)
            }
        };

        let out_prim = out_t.try_into_primitive::<Inner>().unwrap();
        let mut nodes: Vec<_> = Vec::with_capacity(N);
        for h in &has {
            nodes.push(h.node());
        }
        nodes.push(qa.node());
        while nodes.len() < N {
            nodes.push(qa.node());
        }
        let nodes: [_; N] = nodes.try_into().unwrap();
        let prep = AttnResOp.prepare::<S>(nodes);
        let out_adt = match prep.compute_bound().stateful() {
            OpsKind::Tracked(mut prep) => {
                for h in &has {
                    let _ = prep.checkpoint(h);
                }
                let _ = prep.checkpoint(&qa);
                prep.finish((l, form), out_prim)
            }
            OpsKind::UnTracked(prep) => prep.finish(out_prim),
        };
        Some(Tensor::from_primitive::<Autodiff<Inner, S>>(out_adt))
    }

    /// [`depth_attend_autodiff_s`] on the default (no-checkpointing) strategy.
    pub fn depth_attend_autodiff<Inner: Backend, const N: usize>(
        history: &[Tensor<3>],
        query: Tensor<1>,
        form: ScoreForm,
    ) -> Option<Tensor<3>>
    where
        DispatchTensor: DispatchKindConversion<Autodiff<Inner>> + DispatchKindConversion<Inner>,
    {
        depth_attend_autodiff_s::<Inner, NoCheckpointing, N>(history, query, form)
    }
}

#[cfg(feature = "autodiff")]
pub use ad::{depth_attend_autodiff, depth_attend_autodiff_s};

/// Stack the history into [L,B,T,D] (tensor path for the autodiff fallback).
#[cfg(any(feature = "autodiff", test))]
pub fn stack_ad(history: &[Tensor<3>]) -> Tensor<4> {
    let stacked: Vec<Tensor<4>> = history
        .iter()
        .map(|h| h.clone().unsqueeze_dim::<4>(0))
        .collect();
    Tensor::cat(stacked, 0)
}

/// Pure tensor-path depth_attend (autodiff fallback).
#[cfg(any(feature = "autodiff", test))]
#[allow(dead_code)] // compiled-but-dead under test+cuda without autodiff; used by the ad dispatch
pub fn depth_attend_tensor_ad(
    history: &[Tensor<3>],
    query: Tensor<1>,
    form: ScoreForm,
) -> Tensor<3> {
    let n = history.len();
    let [b, t, d] = history[0].dims();
    let scale = form.scale(d);
    let h_stack = stack_ad(history);
    let h_norm_sq = h_stack
        .clone()
        .powf_scalar(2.0)
        .sum_dim(3)
        .mul_scalar(form.norm_m(d))
        .add_scalar(1e-5);
    let h_norm = h_stack.clone() / h_norm_sq.sqrt().reshape([n, b, t, 1usize]);
    let q = query.reshape([1, 1, 1, d]);
    let scores = (q * h_norm).sum_dim(3).mul_scalar(scale);
    let weights = burn::tensor::activation::softmax(scores, 0);
    h_stack
        .mul(weights.reshape([n, b, t, 1usize]))
        .sum_dim(0)
        .reshape([b, t, d])
}

#[cfg(all(test, feature = "autodiff", feature = "cuda"))]
mod seam_tests {
    //! The proof that the strategy gate is strategy-AGNOSTIC. It needs the
    //! `cuda` feature to COMPILE but uses only the ndarray backend at RUNTIME:
    //! no device, no kernel launch. It asserts a caller on
    //! `Autodiff<Inner, BalancedCheckpointing>` — dormouse's backend — gets
    //! PAST the seam downcasts, and that the old `NoCheckpointing`-only
    //! spelling does not. Revert `depth_attend_autodiff_s` to `Autodiff<Inner>`
    //! and the first assertion goes red: it is the assertion, not the comment.
    use super::*;
    use burn::backend::DispatchKindConversion;
    use burn::tensor::{Device, DispatchTensor};
    use burn_autodiff::Autodiff as Ad;
    use burn_autodiff::checkpoint::strategy::{
        BalancedCheckpointing, CheckpointStrategy, NoCheckpointing,
    };

    type Nd = burn_ndarray::NdArray;

    #[test]
    fn balanced_checkpointing_reaches_the_seam_and_the_legacy_entry_does_not() {
        fn reach<S: CheckpointStrategy>(h: &[Tensor<3>], q: &Tensor<1>) -> Option<Tensor<3>>
        where
            DispatchTensor: DispatchKindConversion<Ad<Nd, S>> + DispatchKindConversion<Nd>,
        {
            depth_attend_autodiff_s::<Nd, S, 8>(h, q.clone(), ScoreForm::Paper)
        }

        let dev = Device::ndarray().autodiff().gradient_checkpointing();
        let h = vec![Tensor::<3>::ones([1, 4, 8], &dev); 2];
        let q = Tensor::<1>::ones([8], &dev);

        reset_seam_counts();
        let base = seam_counts().expect("autodiff feature is on in this test").0;

        assert!(reach::<BalancedCheckpointing>(&h, &q).is_some());
        assert_eq!(
            seam_counts().expect("counters").0,
            base + 1,
            "a BalancedCheckpointing caller must get past the seam downcast"
        );

        assert!(
            depth_attend_autodiff::<Nd, 8>(&h, q.clone(), ScoreForm::Paper).is_none(),
            "on a Balanced tensor the NoCheckpointing entry must refuse"
        );
        assert_eq!(
            seam_counts().expect("counters").0,
            base + 1,
            "the refusing entry must not have counted a reach"
        );

        // The default strategy still works, and the cross-check refuses, so
        // the gate is real rather than always-true.
        let plain = Device::ndarray().autodiff();
        let hp = vec![Tensor::<3>::ones([1, 4, 8], &plain); 2];
        let qp = Tensor::<1>::ones([8], &plain);
        assert!(reach::<NoCheckpointing>(&hp, &qp).is_some());
        assert!(reach::<BalancedCheckpointing>(&hp, &qp).is_none());
    }
}

#[cfg(all(test, feature = "autodiff", feature = "cuda"))]
mod ad_tests {
    use super::*;
    use burn::tensor::{Device, Distribution, Tensor};

    type CudaBare = burn_cubecl::CubeBackend;

    fn to_host<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn depth_attend_fused_backward_matches_tensor() {
        let dev = Device::default().autodiff();
        let (l, b, t, d) = (4usize, 2usize, 3usize, 16usize);
        let hist: Vec<Tensor<3>> = (0..l)
            .map(|_| Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &dev))
            .collect();
        let q = Tensor::<1>::random([d], Distribution::Normal(0.0, 1.0), &dev);

        // fused op graph
        let hf: Vec<Tensor<3>> = hist.iter().map(|h| h.clone().require_grad()).collect();
        let qf = q.clone().require_grad();
        let outf =
            crate::fused_attnres::depth_attend_autodiff::<CudaBare, 64>(&hf, qf.clone(), ScoreForm::Paper)
                .unwrap();
        let loss_f = outf.powf_scalar(2.0).sum();
        let grads_f = loss_f.backward();
        let dhf: Vec<Tensor<3>> = hf.iter().map(|h| h.grad(&grads_f).unwrap()).collect();
        let dqf = qf.grad(&grads_f).unwrap();

        // tensor path graph
        let ht: Vec<Tensor<3>> = hist.iter().map(|h| h.clone().require_grad()).collect();
        let qt = q.clone().require_grad();
        let outt = crate::depth_attend(&ht, qt.clone());
        let loss_t = outt.powf_scalar(2.0).sum();
        let grads_t = loss_t.backward();
        let dht: Vec<Tensor<3>> = ht.iter().map(|h| h.grad(&grads_t).unwrap()).collect();
        let dqt = qt.grad(&grads_t).unwrap();

        for i in 0..l {
            let md = maxdiff(&to_host(dhf[i].clone()), &to_host(dht[i].clone()));
            assert!(md < 1e-1, "dh[{i}] maxdiff {md}");
        }
        let md = maxdiff(&to_host(dqf), &to_host(dqt));
        assert!(md < 1e-2, "dq maxdiff {md}");
    }
}

#[cfg(all(test, feature = "autodiff", feature = "cuda"))]
mod fd_tests {
    use super::*;
    use burn::tensor::{Device, Distribution, Tensor};

    type CudaBare = burn_cubecl::CubeBackend;

    fn to_host<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn raw_depth_attend(hs: &[Tensor<3>], q: &Tensor<1>, form: ScoreForm) -> Tensor<3> {
        let n = hs.len();
        let [b, t, d] = hs[0].dims();
        let scale = form.scale(d);
        let h_stack = stack_ad(hs);
        let hn = h_stack
            .clone()
            .powf_scalar(2.0)
            .sum_dim(3)
            .mul_scalar(form.norm_m(d))
            .add_scalar(1e-5)
            .sqrt();
        let hnm = h_stack.clone() / hn.reshape([n, b, t, 1usize]);
        let sc = (q.clone().reshape([1, 1, 1, d]) * hnm)
            .sum_dim(3)
            .mul_scalar(scale);
        let w = burn::tensor::activation::softmax(sc.squeeze_dim::<3>(3), 0);
        h_stack
            .clone()
            .mul(w.reshape([n, b, t, 1usize]))
            .sum_dim(0)
            .reshape([b, t, d])
    }

    #[test]
    fn fused_backward_matches_burn_autodiff() {
        let dev = Device::default();
        let adev = Device::default().autodiff();
        let (l, b, t, d) = (3usize, 1usize, 2usize, 8usize);
        let hist: Vec<Tensor<3>> = (0..l)
            .map(|li| {
                let v: Vec<f32> = (0..b * t * d)
                    .map(|i| ((i * 7 + li * 13) % 17) as f32 / 17.0 - 0.5)
                    .collect();
                Tensor::<3>::from_data(burn::tensor::TensorData::new(v, [b, t, d]), &dev)
            })
            .collect();
        let qv: Vec<f32> = (0..d).map(|i| ((i * 5) % 11) as f32 / 11.0 - 0.5).collect();
        let q = Tensor::<1>::from_data(burn::tensor::TensorData::new(qv, [d]), &dev);
        // fused op (lib dispatch)
        let hf: Vec<Tensor<3>> = hist
            .iter()
            .map(|h| Tensor::<3>::from_data(h.clone().into_data(), &adev).require_grad())
            .collect();
        let qf = Tensor::<1>::from_data(q.clone().into_data(), &adev).require_grad();
        let out = crate::depth_attend(&hf, qf.clone());
        let grads = out.powf_scalar(2.0).sum().backward();
        let fused: Vec<f32> = hf[0]
            .grad(&grads)
            .unwrap()
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        // raw burn autodiff
        let hr: Vec<Tensor<3>> = hist
            .iter()
            .map(|h| Tensor::<3>::from_data(h.clone().into_data(), &adev).require_grad())
            .collect();
        let qr = Tensor::<1>::from_data(q.clone().into_data(), &adev).require_grad();
        let outr = raw_depth_attend(&hr, &qr, ScoreForm::Paper);
        let grads = outr.powf_scalar(2.0).sum().backward();
        let raw: Vec<f32> = hr[0]
            .grad(&grads)
            .unwrap()
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut worst = 0.0f32;
        for i in 0..fused.len() {
            worst = worst.max((fused[i] - raw[i]).abs());
        }
        assert!(worst < 1e-4, "fused vs burn raw autodiff worst={worst}");
    }

    #[test]
    fn depth_attend_grad_matches_finite_difference() {
        let dev = Device::default();
        let adev = Device::default().autodiff();
        let (l, b, t, d) = (3usize, 1usize, 2usize, 8usize);
        let hist: Vec<Tensor<3>> = (0..l)
            .map(|li| {
                let v: Vec<f32> = (0..b * t * d)
                    .map(|i| ((i * 7 + li * 13) % 17) as f32 / 17.0 - 0.5)
                    .collect();
                Tensor::<3>::from_data(burn::tensor::TensorData::new(v, [b, t, d]), &dev)
            })
            .collect();
        let qv: Vec<f32> = (0..d).map(|i| ((i * 5) % 11) as f32 / 11.0 - 0.5).collect();
        let q = Tensor::<1>::from_data(burn::tensor::TensorData::new(qv, [d]), &dev);
        let data_h: Vec<_> = hist.iter().map(|h| h.clone().into_data()).collect();
        let data_q = q.clone().into_data();

        // analytic grads (lib dispatch -> fused op)
        let hf: Vec<Tensor<3>> = data_h
            .iter()
            .map(|dt| Tensor::<3>::from_data(dt.clone(), &adev).require_grad())
            .collect();
        let qf = Tensor::<1>::from_data(data_q.clone(), &adev).require_grad();
        let out = crate::depth_attend(&hf, qf.clone());
        let grads = out.powf_scalar(2.0).sum().backward();
        let dhs: Vec<Vec<f32>> = hf
            .iter()
            .map(|h| to_host(h.grad(&grads).unwrap()))
            .collect();
        let dq = to_host(qf.grad(&grads).unwrap());

        // CENTRAL finite differences, eps = 1e-3 (the 2026-09-29 audit's
        // Test F prescription; the old 1e-4 put the difference quotient's own
        // f32 noise at ~5e-3, which is larger than the gradient components
        // this test is trying to check).
        let eps = 1e-3f32;
        let loss = |hs: &[Tensor<3>], q: &Tensor<1>| -> f32 {
            crate::depth_attend(hs, q.clone())
                .powf_scalar(2.0)
                .sum()
                .into_scalar::<f32>()
        };
        // The difference quotient's noise floor: an f32 loss of this magnitude
        // carries ~EPSILON of rounding, and dividing by 2*eps turns that into
        // a per-component uncertainty. Components below it are not measurable
        // by finite differences, so asserting a RELATIVE error on them is
        // asserting noise. The floor is computed from the loss, not guessed.
        let plain: Vec<Tensor<3>> = data_h
            .iter()
            .map(|dt| Tensor::<3>::from_data(dt.clone(), &dev))
            .collect();
        let loss_scale = loss(&plain, &Tensor::<1>::from_data(data_q.clone(), &dev)).abs();
        let fd_floor = 8.0 * f32::EPSILON * loss_scale / (2.0 * eps);
        let total = b * t * d;
        for li in 0..l {
            let mut fd = vec![0.0f32; total];
            for i in 0..total {
                let mut plus = data_h[li].clone();
                let pv = f32::from_le_bytes(plus.bytes[i * 4..i * 4 + 4].try_into().unwrap()) + eps;
                plus.bytes[i * 4..i * 4 + 4].copy_from_slice(&pv.to_le_bytes());
                let mut minus = data_h[li].clone();
                let mv =
                    f32::from_le_bytes(minus.bytes[i * 4..i * 4 + 4].try_into().unwrap()) - eps;
                minus.bytes[i * 4..i * 4 + 4].copy_from_slice(&mv.to_le_bytes());
                let mut hs_p: Vec<Tensor<3>> = data_h
                    .iter()
                    .map(|x| Tensor::<3>::from_data(x.clone(), &dev))
                    .collect();
                hs_p[li] = Tensor::<3>::from_data(plus, &dev);
                let mut hs_m: Vec<Tensor<3>> = data_h
                    .iter()
                    .map(|x| Tensor::<3>::from_data(x.clone(), &dev))
                    .collect();
                hs_m[li] = Tensor::<3>::from_data(minus, &dev);
                let qd = Tensor::<1>::from_data(data_q.clone(), &dev);
                fd[i] = (loss(&hs_p, &qd) - loss(&hs_m, &qd)) / (2.0 * eps);
            }
            // Per component, with the noise floor added: the missing `m` in
            // the norm's derivative (the defect this cell caught) moved
            // components by 0.1-1.2, four orders above the floor, so the gate
            // still has all of its teeth and none of its false reds.
            for i in 0..total {
                let diff = (dhs[li][i] - fd[i]).abs();
                let rel = diff / (fd[i].abs() + fd_floor);
                assert!(
                    rel < 1e-1,
                    "layer {li} idx {i}: analytic {} vs fd {} (rel {rel}, diff {diff:.3e}, \
                     fd noise floor {fd_floor:.3e})",
                    dhs[li][i],
                    fd[i]
                );
            }
            // And the aggregate, which is the tolerance-meaningful statement:
            // per-component relative error on a near-zero component is not.
            let num: f64 = dhs[li]
                .iter()
                .zip(fd.iter())
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let den: f64 = fd.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
            assert!(
                num / den < 1e-2,
                "layer {li}: ||analytic - fd||/||fd|| = {:.4} ({} vs {})",
                num / den,
                num,
                den
            );
        }
        // dq (same eps and floor as dh: the loss and the step are the same)
        let mut fd = vec![0.0f32; d];
        for i in 0..d {
            let mut plus = data_q.clone();
            let pv = f32::from_le_bytes(plus.bytes[i * 4..i * 4 + 4].try_into().unwrap()) + eps;
            plus.bytes[i * 4..i * 4 + 4].copy_from_slice(&pv.to_le_bytes());
            let mut minus = data_q.clone();
            let mv = f32::from_le_bytes(minus.bytes[i * 4..i * 4 + 4].try_into().unwrap()) - eps;
            minus.bytes[i * 4..i * 4 + 4].copy_from_slice(&mv.to_le_bytes());
            let hs: Vec<Tensor<3>> = data_h
                .iter()
                .map(|x| Tensor::<3>::from_data(x.clone(), &dev))
                .collect();
            let qp = Tensor::<1>::from_data(plus, &dev);
            let qm = Tensor::<1>::from_data(minus, &dev);
            fd[i] = (loss(&hs, &qp) - loss(&hs, &qm)) / (2.0 * eps);
        }
        for i in 0..d {
            let diff = (dq[i] - fd[i]).abs();
            let rel = diff / (fd[i].abs() + fd_floor);
            assert!(
                rel < 1e-1,
                "dq idx {i}: analytic {} vs fd {} (rel {rel}, diff {diff:.3e})",
                dq[i],
                fd[i]
            );
        }
    }
}
