use burn::tensor::{Device, Tensor};
use std::sync::atomic::{AtomicU8, Ordering};

/// Chunk length above which the score path switches from the plain factorized
/// form to the K3 16-tile log-space scheme. Shared by both implementations so
/// the routing decision has one authority.
pub const TILE: usize = 16;

/// Which implementation of the `c <= TILE` chunk path runs.
///
/// Two arms, one switch, because this is a numerical rewrite of a recurrence:
/// the loop arm is the reference (it is the implementation that was in
/// production) and the batched arm is the candidate. The comparison is a
/// differential test, not a claim.
///
/// `DM_GDN2_OPS=batched|loop` picks the default arm for a whole process (the
/// trainer's A/B needs both arms out of one build — a second build of this
/// workspace is a 30-minute thing to ask for twice). [`set_chunk_path`] sets it
/// in-process, which is how the tests compare the two arms in one run. An
/// unrecognised value is a loud panic: a typo must not silently measure the
/// reference twice and call it a win.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChunkPath {
    /// All chunks of the whole sequence in one batched set of tensor ops.
    Batched,
    /// The original per-chunk loop, row-by-row triangular inversion.
    Loop,
}

static PATH: AtomicU8 = AtomicU8::new(2);

/// The arm that runs, reading `DM_GDN2_OPS` once and caching it.
pub fn chunk_path() -> ChunkPath {
    match PATH.load(Ordering::Relaxed) {
        0 => return ChunkPath::Batched,
        1 => return ChunkPath::Loop,
        _ => {}
    }
    let p = match std::env::var("DM_GDN2_OPS").as_deref() {
        Err(_) => ChunkPath::Batched,
        Ok("batched") | Ok("") => ChunkPath::Batched,
        Ok("loop") => ChunkPath::Loop,
        Ok(other) => panic!(
            "DM_GDN2_OPS={other:?} is not an arm of this switch. The accepted values are \
             `batched` (default) and `loop` (the reference implementation)."
        ),
    };
    set_chunk_path(p);
    p
}

/// Set the arm in-process. Tests use this; the trainer uses the env var.
pub fn set_chunk_path(p: ChunkPath) {
    PATH.store(
        match p {
            ChunkPath::Batched => 0,
            ChunkPath::Loop => 1,
        },
        Ordering::Relaxed,
    )
}

/// Whether the batched arm can express this call at all.
///
/// It cannot: (a) `chunk_size > TILE` needs the K3 tile scheme, which the
/// batched arm does not implement, and (b) a caller-supplied `m_invs` is a
/// per-chunk list the batched arm has no layout for. Neither is a degraded
/// answer to a question the loop cannot answer — the loop is the complete
/// implementation of both, so routing is not a fallback. It is COUNTED
/// anyway, because a reader asking "did the batched arm run?" should never have
/// to infer it from the config.
pub fn batched_applies(chunk_size: usize, m_invs: Option<&[Tensor<4>]>) -> bool {
    if chunk_size > TILE {
        crate::alloc_trace::note_batched_declined("chunk_size > TILE");
        return false;
    }
    if m_invs.is_some() {
        crate::alloc_trace::note_batched_declined("caller-supplied m_invs");
        return false;
    }
    true
}

fn tril_matrix(t: usize, device: &Device) -> Tensor<4> {
    let n = t * t;
    let mut d = vec![0.0f32; n];
    for i in 0..t {
        for j in 0..=i {
            d[i * t + j] = 1.0;
        }
    }
    Tensor::<1>::from_floats(d.as_slice(), device)
        .reshape([t, t])
        .unsqueeze_dims(&[0, 0])
}

pub(crate) fn chunk_masks(c: usize, device: &Device) -> (Tensor<4>, Tensor<4>) {
    let n = c * c;
    let mut causal = vec![0.0f32; n];
    let mut strict = vec![0.0f32; n];
    for i in 0..c {
        for j in 0..=i {
            causal[i * c + j] = 1.0;
            if j < i {
                strict[i * c + j] = 1.0;
            }
        }
    }
    (
        Tensor::<1>::from_floats(causal.as_slice(), device).reshape([1, 1, c, c]),
        Tensor::<1>::from_floats(strict.as_slice(), device).reshape([1, 1, c, c]),
    )
}

/// Intermediates of one chunk needed by the exact backward adjoint.
///
/// The adjoint re-derives gradients from these values without re-differentiating
/// the token loop: the WY solve is differentiated via the triangular adjoint
/// `d_rhs = M^-T d_*`, `d_akk = tril_strict(-M^-T d_* W^T)`.
#[derive(Clone, Debug)]
pub struct ChunkScratch {
    /// exp(cumsum(g)) over the chunk, `[B, H, c, k]`.
    pub g_exp: Tensor<4>,
    /// q * g_exp, `[B, H, c, k]`.
    pub q_gated: Tensor<4>,
    /// `(q·E)(k/E)^T * scale * causal`, `[B, H, c, c]`.
    pub aqk: Tensor<4>,
    /// `M^-1` with `M = I + L`: W = M⁻¹·rhs_k, backward solves via M⁻ᵀ·d_*.
    pub m_inv: Tensor<4>,
}

#[derive(Clone, Debug)]
pub struct ChunkWyScratch {
    pub chunks: Vec<ChunkScratch>,
}

/// The chunked-WY forward, on the arm [`chunk_path`] selects.
#[allow(clippy::too_many_arguments)]
pub fn chunk_wy_forward_impl(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<4>,
    b: Tensor<4>,
    w_gate: Tensor<4>,
    state: Tensor<4>,
    scale: f64,
    chunk_size: usize,
    m_invs: Option<&[Tensor<4>]>,
) -> (Tensor<4>, Tensor<4>, ChunkWyScratch) {
    if chunk_path() == ChunkPath::Batched && batched_applies(chunk_size, m_invs) {
        chunk_wy_forward_batched(q, k, v, g, b, w_gate, state, scale, chunk_size)
    } else {
        chunk_wy_forward_loop(q, k, v, g, b, w_gate, state, scale, chunk_size, m_invs)
    }
}

/// The batched arm: `chunk_size <= TILE`, every chunk of the whole sequence in
/// one set of tensor ops, and the state recurrence left sequential.
///
/// ## What is batched, and why that split
///
/// `v_new` for chunk `i` needs the state left by chunk `i-1`, so the state
/// recurrence is irreducibly sequential and stays a loop. Everything else —
/// the scores, the decay, the WY right-hand sides and the triangular solve —
/// depends only on that chunk's own inputs, so it is done for all `n` chunks
/// at once on a rank-5 `[B, H, n, c, *]` view: ONE reshape, and the whole
/// forward is ~30 tensor ops for the scores and solves plus 2(c-1) for the
/// inversion, against ~110 per chunk before.
///
/// The head axis and the chunk axis stay separate on purpose: the backward's
/// per-chunk scratch is then a contiguous slice on axis 2 instead of a strided
/// one. (`tests/b5_seam_probe.rs` measures the rank-5 ops this relies on.)
///
/// ## Ragged tails
///
/// `T` need not be a multiple of `chunk_size`. The last chunk is zero-padded
/// to `chunk_size` and the padding is then exactly the loop's answer, not an
/// approximation: `g` padded with 0 makes `E` constant over the padded rows,
/// so `k/E = 0` there and every padded row of `akk`, `aqk`, `rhs_k`, `rhs_v`,
/// `q_gated` and the output is 0, and `k*decay_last` is 0 so the padded rows
/// cannot reach the state either. That is what lets one batched set of ops
/// cover both the full and the ragged case; the loop's `chunk_masks(c)` /
/// `eye(c)` special case is not needed here.
///
/// ## The inversion
///
/// `akk` is masked strictly lower triangular, so with `L = akk` and
/// `M = I + L`, `L^c = 0` EXACTLY for a `c x c` matrix, and the Neumann series
/// `M^-1 = I - L + L^2 - ... + (-L)^(c-1)` is a finite identity, not a
/// truncated one. That is 2(c-1) batched matmuls for every chunk at once,
/// against the loop's 15 iterations of (clone, 2 slices, tiny matmul,
/// slice-assign) per chunk.
#[allow(clippy::too_many_arguments)]
pub fn chunk_wy_forward_batched(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<4>,
    b: Tensor<4>,
    w_gate: Tensor<4>,
    mut state: Tensor<4>,
    scale: f64,
    chunk_size: usize,
) -> (Tensor<4>, Tensor<4>, ChunkWyScratch) {
    assert!(
        chunk_size <= TILE,
        "the batched chunk arm is the c <= TILE path: chunk_size={chunk_size} needs the K3 \
         16-tile decay scheme, which lives in the loop arm. Route it with ChunkPath::Loop \
         (or leave DM_GDN2_OPS unset and let the dispatcher route it)."
    );
    let [batch, heads, time, k_dim] = q.shape().dims::<4>();
    let v_dim = v.shape().dims::<4>()[3];
    let device = q.device();
    let n_chunks = time.div_ceil(chunk_size);
    let pad = n_chunks * chunk_size - time;

    // The module feeds permuted [B,H,T,K] views here; cubecl ops would copy
    // them per-op. Materialize once so every later op sees contiguous buffers.
    // Then pad the tail and fold the time axis into (chunk, position) — the
    // one reshape that turns the whole forward into batched ops.
    let fold = |t: Tensor<4>, d: usize| -> Tensor<5> {
        let t = t.mul_scalar(1.0);
        let t = if pad > 0 {
            Tensor::cat(vec![t, Tensor::zeros([batch, heads, pad, d], &device)], 2)
        } else {
            t
        };
        t.reshape([batch, heads, n_chunks, chunk_size, d])
    };
    let q5 = fold(q, k_dim);
    let k5 = fold(k, k_dim);
    let v5 = fold(v, v_dim);
    let g5 = fold(g, k_dim);
    let b5 = fold(b, k_dim);
    let w5 = fold(w_gate, v_dim);

    // masks, hoisted with the scale folded in: the batched body pays one mul
    let (scale_causal, strict) = chunk_masks(chunk_size, &device);
    let scale_causal = scale_causal
        .mul_scalar(scale)
        .reshape([1, 1, 1, chunk_size, chunk_size]);
    let strict = strict.reshape([1, 1, 1, chunk_size, chunk_size]);
    let eye = Tensor::<2>::eye(chunk_size, &device).reshape([1, 1, 1, chunk_size, chunk_size]);

    // --- phase A: scores / decay / right-hand sides, all chunks at once
    let g_cumsum = g5.cumsum(3);
    let g_exp = g_cumsum.clone().exp();
    let k_over_gamma = k5.clone() / g_exp.clone();
    let k_over_gamma_t = k_over_gamma.clone().swap_dims(3, 4);
    let q_gated = q5 * g_exp.clone();
    let aqk = q_gated.clone().matmul(k_over_gamma_t.clone()) * scale_causal;
    let bk = b5 * k5.clone();
    let akk = (bk.clone() * g_exp.clone()).matmul(k_over_gamma_t) * strict;

    // --- phase B: the triangular solve, all chunks at once
    let m_inv = neumann_inverse(akk, chunk_size, eye);
    let w_wy = m_inv.clone().matmul(bk * g_exp.clone());
    let u = m_inv.clone().matmul(w5 * v5);

    // --- phase C: the state recurrence, which is irreducibly sequential
    let g_last = g_exp.clone().slice([
        0..batch,
        0..heads,
        0..n_chunks,
        chunk_size - 1..chunk_size,
        0..k_dim,
    ]);
    let decay_last = (g_cumsum.clone().slice([
        0..batch,
        0..heads,
        0..n_chunks,
        chunk_size - 1..chunk_size,
        0..k_dim,
    ]) - g_cumsum)
        .exp();
    let k_dec = k5 * decay_last;

    let mut outputs = Vec::with_capacity(n_chunks);
    for ci in 0..n_chunks {
        // Executed (not logical) iterations: a retro-forward replay of this
        // loop lands here too, which is how the checkpointer recompute is
        // counted. See `alloc_trace::chunk_iterations`.
        crate::alloc_trace::chunk_iteration();
        let cut = |t: Tensor<5>, d: usize| -> Tensor<4> {
            t.slice([0..batch, 0..heads, ci..ci + 1, 0..chunk_size, 0..d])
                .squeeze_dim::<4>(2)
        };
        let s_before = state.clone();
        let v_new = cut(u.clone(), v_dim) - cut(w_wy.clone(), k_dim).matmul(s_before.clone());
        let out_c = cut(aqk.clone(), chunk_size).matmul(v_new.clone())
            + cut(q_gated.clone(), k_dim)
                .matmul(s_before.clone())
                .mul_scalar(scale);
        outputs.push(out_c);
        state = s_before * cut(g_last.clone(), k_dim).swap_dims(2, 3)
            + cut(k_dec.clone(), k_dim).swap_dims(2, 3).matmul(v_new);
    }

    // The same four values per chunk the custom node's analytic backward needs.
    // Free on the ops path, which discards them: cubecl's `slice` with aligned
    // offsets is a handle offset, not a kernel (f32 and a chunk-axis offset are
    // always aligned, so the copy fallback does not fire here), and a slice
    // costs a node only on the autodiff backend, which is the arm that reads
    // them.
    let mut chunks = Vec::with_capacity(n_chunks);
    for ci in 0..n_chunks {
        let cut = |t: &Tensor<5>, d: usize| -> Tensor<4> {
            t.clone()
                .slice([0..batch, 0..heads, ci..ci + 1, 0..chunk_size, 0..d])
                .squeeze_dim::<4>(2)
        };
        chunks.push(ChunkScratch {
            g_exp: cut(&g_exp, k_dim),
            q_gated: cut(&q_gated, k_dim),
            aqk: cut(&aqk, chunk_size),
            m_inv: cut(&m_inv, chunk_size),
        });
    }

    (
        Tensor::cat(outputs, 2).slice([0..batch, 0..heads, 0..time, 0..v_dim]),
        state,
        ChunkWyScratch { chunks },
    )
}

/// `(I + L)^-1` for a strictly lower triangular `L` (here `akk`), batched over
/// every leading axis: `sum_{j=0}^{c-1} (-L)^j`, exact because `L^c = 0`.
///
/// 2(c-1) matmuls and adds for ALL chunks at once, against the loop's c-1
/// iterations of a rank-4 matmul each. The diagonal of the seed is what carries
/// the identity; the strictly-lower mask on `akk` is what makes `L^c` zero
/// rather than merely small, so this is an identity and not a truncation.
fn neumann_inverse(akk: Tensor<5>, c: usize, eye: Tensor<5>) -> Tensor<5> {
    let neg_l = akk.mul_scalar(-1.0);
    let mut acc = eye.clone();
    let mut p = neg_l.clone();
    for _ in 1..c {
        acc = acc + p.clone();
        p = p.matmul(neg_l.clone());
    }
    acc
}

/// The reference implementation: one pass over the chunks, the triangular
/// inversion row by row. Unmodified — it is the arm the differential test
/// compares against, so a "cleanup" here would move the goalposts.
#[allow(clippy::too_many_arguments)]
pub fn chunk_wy_forward_loop(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<4>,
    b: Tensor<4>,
    w_gate: Tensor<4>,
    mut state: Tensor<4>,
    scale: f64,
    chunk_size: usize,
    m_invs: Option<&[Tensor<4>]>,
) -> (Tensor<4>, Tensor<4>, ChunkWyScratch) {
    let [batch, heads, time, k_dim] = q.shape().dims::<4>();
    let v_dim = v.shape().dims::<4>()[3];
    let device = q.device();
    let mut outputs = Vec::with_capacity(time.div_ceil(chunk_size));
    // The module feeds permuted [B,H,T,K] views here; cubecl ops would copy
    // them per-op. Materialize once so every later op sees contiguous buffers.
    let q = q.mul_scalar(1.0);
    let k = k.mul_scalar(1.0);
    let v = v.mul_scalar(1.0);
    let g = g.mul_scalar(1.0);
    let b = b.mul_scalar(1.0);
    let w_gate = w_gate.mul_scalar(1.0);
    let mut scratch = ChunkWyScratch { chunks: Vec::new() };
    let tril_full = tril_matrix(chunk_size, &device);
    let (_causal_full, strict_full) = chunk_masks(chunk_size, &device);
    // scale folded into the causal mask, eye pre-broadcast: hoisted so the
    // chunk loop pays one mul / one eye instead of two / three ops.
    let scale_causal_full = tril_full.clone() * scale;
    let eye_full = Tensor::<2>::eye(chunk_size, &device)
        .reshape([1, 1, chunk_size, chunk_size])
        .repeat(&[batch, heads, 1, 1]);

    // Two score paths:
    // - c <= 16: the fast matmul form (q·E)(k/E)^T — numerically safe for the
    //   K3 floor g = -5 (16·(-5) = -80 > -88, the f32 exp underflow point)
    //   and for the bounded decay ranges of the GDN-2/KDA gates.
    // - c > 16: the K3 16-tile log-space decay scheme (ported from the
    //   burn-0.21 line): decay split into tile-local factors (cumsum minus
    //   the tile boundary, bounded by 16·g_min = -80) and an inter-tile
    //   factor exp(G_p - G_q) of boundary differences (always <= 0), so no
    //   exp argument overflows and the chunk length is unbounded.
    // `TILE` is the module-level constant both arms route on.
    let pad_to = |t: Tensor<4>, c_pad: usize, d: usize| -> Tensor<4> {
        let cc = t.shape().dims::<4>()[2];
        if cc == c_pad {
            t
        } else {
            Tensor::cat(
                vec![t, Tensor::zeros([batch, heads, c_pad - cc, d], &device)],
                2,
            )
        }
    };

    for chunk_start in (0..time).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(time);
        let c = chunk_end - chunk_start;
        if c == 0 {
            continue;
        }
        // Executed (not logical) iterations: a retro-forward replay of this
        // loop lands here too, which is how the checkpointer recompute is
        // counted. See `alloc_trace::chunk_iterations`.
        crate::alloc_trace::chunk_iteration();

        let q_c = q
            .clone()
            .slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let k_c = k
            .clone()
            .slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let v_c = v
            .clone()
            .slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let g_c = g
            .clone()
            .slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let b_c = b
            .clone()
            .slice([0..batch, 0..heads, chunk_start..chunk_end]);
        let w_c = w_gate
            .clone()
            .slice([0..batch, 0..heads, chunk_start..chunk_end]);

        // scores / rhs / decay, with the effective (possibly padded) length
        let (aqk, akk, gamma, g_cumsum, q_gated, rhs_k, rhs_v, k_upd, c_eff, m_eye) = if c <= TILE {
            let (scale_causal, strict_mask) = if c == chunk_size {
                (scale_causal_full.clone(), strict_full.clone())
            } else {
                let (cau, str) = chunk_masks(c, &device);
                (cau * scale, str)
            };
            let m_eye = if c == chunk_size {
                eye_full.clone()
            } else {
                Tensor::<2>::eye(c, &device)
                    .reshape([1, 1, c, c])
                    .repeat(&[batch, heads, 1, 1])
            };
            let g_cumsum = g_c.clone().cumsum(2);
            let g_exp = g_cumsum.clone().exp();
            let k_over_gamma = k_c.clone() / g_exp.clone();
            let aqk = (q_c.clone() * g_exp.clone()).matmul(k_over_gamma.clone().swap_dims(2, 3))
                * scale_causal;
            let bk = b_c.clone() * k_c.clone();
            let akk =
                (bk.clone() * g_exp.clone()).matmul(k_over_gamma.swap_dims(2, 3)) * strict_mask;
            let rhs_k = bk * g_exp.clone();
            let rhs_v = w_c * v_c;
            let q_gated = q_c.clone() * g_exp.clone();
            (
                aqk, akk, g_exp, g_cumsum, q_gated, rhs_k, rhs_v, k_c, c, m_eye,
            )
        } else {
            let c_pad = c.div_ceil(TILE) * TILE;
            let n_t = c_pad / TILE;
            let g_p = pad_to(g_c, c_pad, k_dim);
            let q_p = pad_to(q_c, c_pad, k_dim);
            let k_p = pad_to(k_c, c_pad, k_dim);
            let v_p = pad_to(v_c, c_pad, v_dim);
            let b_p = pad_to(b_c, c_pad, k_dim);
            let w_p = pad_to(w_c, c_pad, v_dim);

            let (scale_causal, strict_mask) = if c_pad == chunk_size {
                (scale_causal_full.clone(), strict_full.clone())
            } else {
                let (cau, str) = chunk_masks(c_pad, &device);
                (cau * scale, str)
            };
            let m_eye = if c_pad == chunk_size {
                eye_full.clone()
            } else {
                Tensor::<2>::eye(c_pad, &device)
                    .reshape([1, 1, c_pad, c_pad])
                    .repeat(&[batch, heads, 1, 1])
            };

            // full cumulative log-decay over the (padded) chunk
            let g_cumsum = g_p.clone().cumsum(2); // [B,H,c_pad,k]
                                                  // exclusive prefix of tile sums: decay accumulated before each tile
            let g_bound_prev = g_cumsum
                .clone()
                .reshape([batch, heads, n_t, TILE, k_dim])
                .slice([0..batch, 0..heads, 0..n_t - 1, TILE - 1..TILE, 0..k_dim])
                .reshape([batch, heads, n_t - 1, k_dim]);
            let g_bound = Tensor::cat(
                vec![
                    Tensor::zeros([batch, heads, 1, k_dim], &device),
                    g_bound_prev,
                ],
                2,
            ); // [B,H,n_t,k]

            // tile-local decay: cumsum minus the boundary accumulated before
            // the tile (at most 16·g_min = -80 -> exp <= e^-80, no underflow)
            let g_bound_full = g_bound
                .clone()
                .reshape([batch, heads, n_t, 1, k_dim])
                .repeat(&[1, 1, 1, TILE, 1])
                .reshape([batch, heads, c_pad, k_dim]);
            let g_rel_log = g_cumsum.clone() - g_bound_full;
            let g_rel_exp = g_rel_log.exp();
            let gamma = g_cumsum.clone().exp();

            // inter-tile decay exp(G_p - G_q) per channel, clamped to 1 above
            // the diagonal so upper blocks are killed by the causal mask
            // Batched over channels (roadmap #4b): per-channel matmuls become
            // one 5D broadcast product aqk[i,j] = sum_k A[i,k]·B[j,k]·E[i,j,k]
            // (no per-channel loop, no permutes — burn-ndarray 0.22-pre.2
            // corrupts swap_dims on 5D and reshapes of permuted views, so the
            // k-batched matmul form [B,H,k,c,c] is out of reach on ndarray).
            // k is chunked to bound the [B,H,c_pad,c_pad,KK] intermediate.
            // gb_full = repeat_interleave of g_bound: the TILE×TILE ones-block
            // Kronecker expansion falls out of the broadcast itself.
            let gb_full = g_bound
                .clone()
                .unsqueeze_dim::<5>(3) // [B,H,n_t,1,k]
                .repeat_dim(3, TILE) // [B,H,n_t,TILE,k]
                .reshape([batch, heads, c_pad, k_dim]); // [B,H,c_pad,k]
            let a5 = q_p.clone().mul(g_rel_exp.clone()).unsqueeze_dim::<5>(3); // [B,H,c,1,k]
            let b5 = k_p.clone().div(g_rel_exp.clone()).unsqueeze_dim::<5>(2); // [B,H,1,c,k]
            let bk5 = b_p
                .clone()
                .mul(k_p.clone())
                .mul(g_rel_exp)
                .unsqueeze_dim::<5>(3); // [B,H,c,1,k]
            let k_chunk = (4_000_000 / (batch * heads * c_pad * c_pad)).max(1);
            let mut aqk = Tensor::zeros([batch, heads, c_pad, c_pad], &device);
            let mut akk = Tensor::zeros([batch, heads, c_pad, c_pad], &device);
            for kc in (0..k_dim).step_by(k_chunk) {
                let ke = (kc + k_chunk).min(k_dim);
                let e5 = gb_full
                    .clone()
                    .slice([0..batch, 0..heads, 0..c_pad, kc..ke]) // [B,H,c,KK]
                    .unsqueeze_dim::<5>(3) // [B,H,c,1,KK]
                    .sub(
                        gb_full
                            .clone()
                            .slice([0..batch, 0..heads, 0..c_pad, kc..ke])
                            .unsqueeze_dim::<5>(2), // [B,H,1,c,KK]
                    )
                    .clamp_max(0.0)
                    .exp(); // [B,H,c,c,KK]
                let ak = a5
                    .clone()
                    .slice([0..batch, 0..heads, 0..c_pad, 0..1, kc..ke]);
                let bk = b5
                    .clone()
                    .slice([0..batch, 0..heads, 0..1, 0..c_pad, kc..ke]);
                let bkk = bk5
                    .clone()
                    .slice([0..batch, 0..heads, 0..c_pad, 0..1, kc..ke]);
                aqk = aqk
                    + ak.mul(bk.clone())
                        .mul(e5.clone())
                        .sum_dim(4)
                        .squeeze_dim::<4>(4);
                akk = akk + bkk.mul(bk).mul(e5).sum_dim(4).squeeze_dim::<4>(4);
            }
            let aqk = aqk * scale_causal;
            let akk = akk * strict_mask;

            let rhs_k = b_p.clone() * k_p.clone() * gamma.clone();
            let rhs_v = w_p * v_p;
            let q_gated = q_p * gamma.clone();
            (
                aqk,
                akk,
                gamma.clone(),
                g_cumsum,
                q_gated,
                rhs_k,
                rhs_v,
                k_p,
                c_pad,
                m_eye,
            )
        };

        // M = I + L (unit lower triangular, L = strict-lower(akk)). Invert M
        // once per chunk (row i needs only rows < i), then all four solves —
        // forward W = M⁻¹·rhs_k, U = M⁻¹·rhs_v and the two backward solves
        // d_rhs = M⁻ᵀ·d_* — become single matmuls instead of per-row loops.
        // m_inv is seeded with the identity so the diagonal e_i survives the
        // strict-lower slice_assign.
        let mut m_inv = m_eye;
        if let Some(saved) = m_invs {
            // M^-1 exported by the fused kernel (or the previous forward):
            // skip the row-by-row inversion entirely.
            m_inv = saved[scratch.chunks.len()].clone();
        } else {
            for i in 1..c_eff {
                let akk_row = akk.clone().slice([0..batch, 0..heads, i..i + 1, 0..i]);
                let m_prev = m_inv.clone().slice([0..batch, 0..heads, 0..i, 0..c_eff]);
                let row = -(akk_row.matmul(m_prev)).slice([0..batch, 0..heads, 0..1, 0..i]);
                m_inv = m_inv.slice_assign([0..batch, 0..heads, i..i + 1, 0..i], row);
            }
        }

        let w_wy = m_inv.clone().matmul(rhs_k.clone());
        let u = m_inv.clone().matmul(rhs_v.clone());

        let state_before = state.clone();
        let v_new = u.clone() - w_wy.clone().matmul(state_before.clone());
        let intra = aqk.clone().matmul(v_new.clone());
        let inter = q_gated.clone().matmul(state_before) * scale;
        let out_c = (intra + inter).slice([0..batch, 0..heads, 0..c, 0..v_dim]);
        outputs.push(out_c);

        // Only the four values the backward cannot cheaply re-derive are
        // kept (W/U/v_new/rhs/kG/akk all recompute from these + the input
        // checkpoints in ~8 ops per chunk); the fused kernels export the same
        // four, so the training path never re-runs the forward.
        let g_last = gamma.clone().slice([0..batch, 0..heads, c - 1..c]);
        scratch.chunks.push(ChunkScratch {
            g_exp: gamma,
            q_gated,
            aqk,
            m_inv,
        });

        // state update in log-space differences: decay = exp(G_last - G_t) <= 1
        let g_last_log = g_cumsum.clone().slice([0..batch, 0..heads, c - 1..c]);
        let decay_last = (g_last_log - g_cumsum).exp();
        state = state * g_last.swap_dims(2, 3) + (k_upd * decay_last).swap_dims(2, 3).matmul(v_new);
    }

    (
        Tensor::cat(outputs, 2),
        state,
        ChunkWyScratch {
            chunks: scratch.chunks,
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn chunk_wy_forward(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<4>,
    b: Tensor<4>,
    w_gate: Tensor<4>,
    state: Tensor<4>,
    scale: f64,
    chunk_size: usize,
) -> (Tensor<4>, Tensor<4>) {
    if crate::alloc_trace::enabled() {
        let [b, h, t, kd] = q.shape().dims::<4>();
        let vd = v.shape().dims::<4>()[3];
        let (decl, why) = crate::alloc_trace::batched_declined();
        println!(
            "[gdn2] TENSOR-OP chunk path (fused kernels NOT engaged): b={b} h={h} t={t} \
             K={kd} V={vd} chunk={chunk_size} n_chunks={} arm={:?} batched_declined={decl}{}",
            t.div_ceil(chunk_size),
            crate::forward::chunk_path(),
            if decl == 0 {
                String::new()
            } else {
                format!(" ({why})")
            },
        );
    }
    let (output, new_state, _scratch) =
        chunk_wy_forward_impl(q, k, v, g, b, w_gate, state, scale, chunk_size, None);
    (output, new_state)
}
