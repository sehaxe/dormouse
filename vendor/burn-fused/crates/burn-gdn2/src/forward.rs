use burn::tensor::{Device, Tensor};

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

#[allow(clippy::too_many_arguments)]
pub fn chunk_wy_forward_impl(
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
    const TILE: usize = 16;
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
    let (output, new_state, _scratch) =
        chunk_wy_forward_impl(q, k, v, g, b, w_gate, state, scale, chunk_size, None);
    (output, new_state)
}
