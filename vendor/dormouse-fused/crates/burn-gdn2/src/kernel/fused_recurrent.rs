use burn::backend::{Backend, DispatchKindConversion};
use burn::tensor::{DispatchTensor, Tensor};

/// Fused recurrent forward - token-by-token scan.
///
/// Reference implementation. Each token applies:
///   S ← S * exp(g_t)                             - channel-wise decay
///   v_new ← w_t * v_t - (b_t * k_t)^T @ S       - erase + write
///   S ← S + k_t @ v_new^T                        - rank-1 state update
///   o_t ← q_t^T @ S * scale                      - output read
#[allow(clippy::too_many_arguments)]
pub fn fused_recurrent_forward(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<4>,
    b: Tensor<4>,
    w: Tensor<4>,
    mut state: Tensor<4>,
    scale: f64,
) -> (Tensor<4>, Tensor<4>) {
    let time = q.shape().dims::<4>()[2];
    let mut outputs = Vec::with_capacity(time);

    // Pre-expand gates to avoid per-token exp calls
    let g_exp_acc = g.clone().exp();

    for t in 0..time {
        let q_t = q.clone().slice_dim(2, t..t + 1);
        let k_t = k.clone().slice_dim(2, t..t + 1);
        let v_t = v.clone().slice_dim(2, t..t + 1);
        let g_t = g_exp_acc.clone().slice_dim(2, t..t + 1);
        let b_t = b.clone().slice_dim(2, t..t + 1);
        let w_t = w.clone().slice_dim(2, t..t + 1);

        // S ← S * exp(g_tᵀ) - decay state along K dimension
        state = state * g_t.swap_dims(2, 3);

        // v_new = w * v - (b * k)^T @ S
        let bk_t = (b_t * k_t.clone()).swap_dims(2, 3); // [B, HV, K, 1]
        let erased = (state.clone() * bk_t).sum_dim(2);
        let v_new = w_t * v_t - erased;

        // S ← S + k_tᵀ @ v_new  (rank-1 update)
        state = state + k_t.swap_dims(2, 3) * v_new;

        // o = q_t^T @ S * scale  (output read)
        let out = (state.clone() * q_t.swap_dims(2, 3))
            .sum_dim(2)
            .mul_scalar(scale);

        outputs.push(out);
    }

    (Tensor::cat(outputs, 2), state)
}

/// Token-by-token recurrent forward, accepts optional state.
///
/// When `update_state` is true, the state is modified in-place (normal
/// autoregressive decoding). When false, the state is read-only (prefill).
///
/// With the `cuda` feature, single-token steps (autoregressive decoding) are
/// executed by one fused cubecl kernel instead of ~8 tensor ops.
#[allow(clippy::too_many_arguments)]
pub fn fused_recurrent_gdn2<B: Backend>(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<4>,
    b: Tensor<4>,
    w: Tensor<4>,
    state: Option<Tensor<4>>,
    scale: f64,
    update_state: bool,
) -> (Tensor<4>, Option<Tensor<4>>)
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let [batch, hv, _time, _] = q.shape().dims::<4>();
    let [_, _, _, k_dim] = k.shape().dims::<4>();
    let v_dim = v.shape().dims::<4>()[3];

    let s = state.unwrap_or_else(|| Tensor::zeros([batch, hv, k_dim, v_dim], &q.device()));

    // Autoregressive decoding (one token): single fused kernel launch.
    #[cfg(feature = "cuda")]
    if update_state && _time == 1 {
        if let Some((o, s)) = crate::kernel::fused_recurrent_cube::cuda::fused_step::<B>(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            b.clone(),
            w.clone(),
            s.clone(),
            scale,
        ) {
            return (o, Some(s));
        }
    }

    let (output, new_state) = if update_state {
        fused_recurrent_forward(q, k, v, g, b, w, s, scale)
    } else {
        let mem_clone = s.clone();
        let out = q.matmul(mem_clone).mul_scalar(scale).permute([0, 2, 1, 3]);
        (out, s)
    };

    (output, Some(new_state))
}
