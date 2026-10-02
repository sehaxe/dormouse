//! Fused CUDA chunked KDA forward.
//!
//! KDA's chunkwise form (Kimi Linear Eqs 6-9) is algebraically identical to
//! the GDN-2 WY chunk algorithm under the mapping:
//!   g (log decay)  = log(alpha)        (channel-wise retention factors)
//!   b              = beta_k            (erase strength, key channels)
//!   w_gate         = beta_v            (write strength: KDA's pseudo-value is
//!                                       U = (I+T')^{-1}(β⊙V), NOT V)
//!   scale          = head_k_dim**-0.5  (FLA's default, both upstreams)
//! With those inputs, `chunk_wy_forward` computes exactly
//!   A  = Tril((Q⊙Γ)(K/Γ)^T)
//!   W  = (I + T')^{-1}(β⊙Γ⊙K),  U = (I + T')^{-1}(β⊙V)
//!   S += Diag(Γ^C)S + (Γ⊙K)^T(U - W S),  O = (Γ⊙Q)S + A(U - W S)
//! so the existing GDN-2 fused kernels are reused as-is.
//!
//! The kernel is only used on the bare CUDA `CubeBackend`; everything else
//! falls back to the tensor-ops chunk path.
//!
//! # `scale = head_k_dim**-0.5`, FIXED 2026-10-02
//!
//! This file used to hard-code `scale = 1.0` and claimed "no softmax scale in
//! KDA". That claim was **false against both upstreams**:
//!
//! - `fla/ops/kda/chunk.py:474-475` and `fla/ops/kda/fused_recurrent.py:261-262`
//!   both contain `if scale is None: scale = K ** -0.5`;
//! - `fla/layers/kda.py:262-278` calls `chunk_kda(...)` with **no `scale`
//!   argument**, so the official KDA layer runs at `head_k_dim**-0.5`;
//! - `fla/ops/kda/naive.py:57` folds it into `q` before the recurrence loop.
//!
//! `burn-gdn2` itself uses `d_k**-0.5` (`gdn2/src/module.rs:299`); burn-kda
//! was the only place in the library that overrode it to `1.0`. FIXED
//! owner-approved, class B (was two RED-ON-PURPOSE oracle tests, then
//! `#[ignore]`d on branch wt/kdaci — never merged): the constant now lives in
//! ONE function, `crate::fla_read_scale` (src/lib.rs), the dispatch arms below
//! derive it from `q`'s own K, and the doc-divergence history is in
//! `docs/reviews/kdafix-2026-10-02.md`.
//!
//! WHY THE MODEL HAS NOT NOTICED: the factor enters only the read `o = q·S`, so
//! it is one constant on the attention output, and the very next thing this
//! crate does is `KdaModule::output`'s RMSNorm, which is invariant to a
//! constant rescale — `c·o / sqrt(mean((c·o)²) + eps) = o / sqrt(mean(o²) +
//! eps/c²)` — so it is absorbed to `O(eps / mean(o²))` ≈ `O(1e-5)`. That is
//! exactly why no loss curve, seed comparison or in-crate test can see it, and
//! also why it is still real: anything reading the raw attention output sees a
//! tensor `head_k_dim**0.5` too large (2.83× at K=8, 8× at K=64), but the
//! module's own RMSNorm absorbs it. Anything reading the RAW attention output
//! (an external scorer, a probe) sees the corrected scale from 2026-10-02 on.
//!
//! The mechanism was never missing — `chunk_wy_forward` honours the scale when
//! asked (`tests/kda_oracle.rs::chunked_wy_applies_the_fla_read_scale`,
//! former `chunked_wy_honours_the_read_scale_when_asked`), and the oracle
//! tests `(read_scale_matches_fla_reference`,
//! `chunked_wy_applies_the_fla_read_scale`) have been green as a pair since
//! the fix. `docs/reviews/2026-09-30-kda-formula-audit.md` §3.2 is the audit
//! this fix closes.

#[cfg(feature = "cuda")]
pub mod cuda {
    use burn::backend::{Backend, DispatchKindConversion};
    use burn::tensor::{DispatchTensor, Tensor};
    use std::any::TypeId;

    pub type CudaBare = burn_gdn2::CudaBare;

    /// `true` when `B` IS the bare CUDA backend (no autodiff wrapper). A
    /// `TypeId` is the right test HERE and only here: a bare backend has
    /// exactly one identity. It is NOT the test for "CUDA underneath" — an
    /// autodiff wrapper is a different type for every checkpointing strategy,
    /// which is what kept the fused path out of every training step
    /// (`Autodiff<CudaBare, BalancedCheckpointing>` is dormouse's trainer
    /// backend). For that see [`burn_gdn2::backend_matches`].
    fn is_bare_cuda<B: Backend>() -> bool {
        TypeId::of::<B>() == TypeId::of::<CudaBare>()
    }

    /// `q,k,v`: `[B, H, T, K/V]` projected (L2-normed q/k, Swish v).
    /// `log_alpha`: `[B, H, T, K]` log retention factors.
    /// `beta_k`: `[B, H, T, K]` write strength over key channels (erase term).
    /// `beta_v`: `[B, H, T, V]` write strength over value channels (the KDA
    /// pseudo-value is `U = (I+T')^-1 (beta ⊙ V)`, so the value gate input is
    /// `beta`, not 1).
    /// `state`: `[B, H, K, V]`.
    /// Returns `(out [B,H,T,V], state)` if the fused path applies.
    #[allow(clippy::too_many_arguments)]
    pub fn kda_fused_chunk<B: Backend>(
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        log_alpha: Tensor<4>,
        beta_k: Tensor<4>,
        beta_v: Tensor<4>,
        state: Tensor<4>,
        chunk_size: usize,
    ) -> Option<(Tensor<4>, Tensor<4>)>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        kda_fused_chunk_reported::<B>(q, k, v, log_alpha, beta_k, beta_v, state, chunk_size)
            .into_option()
    }

    /// The production gate, and it SAYS which arm ran.
    ///
    /// Bare CUDA backend: the kernels take the tensors as they are. Autodiff
    /// wrapper over CUDA, ANY checkpointing strategy: the seam in
    /// [`burn_gdn2::cuda_dispatch`] strips the autodiff context, runs the
    /// fused kernels on the bare backend inside ONE autodiff node, and
    /// rebuilds the node on the caller's own strategy. Everything else
    /// (NdArray, ...) reports [`Fallback::NotCuda`] and the caller runs the
    /// tensor-ops chunk path.
    ///
    /// `Fused::Fused` means the single-node op ran; the launch counters
    /// ([`burn_gdn2::fused_calls`]) are the ground truth for whether the
    /// kernels inside it engaged or it used its own tensor fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn kda_fused_chunk_reported<B: Backend>(
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        log_alpha: Tensor<4>,
        beta_k: Tensor<4>,
        beta_v: Tensor<4>,
        state: Tensor<4>,
        chunk_size: usize,
    ) -> burn_gdn2::Fused<(Tensor<4>, Tensor<4>)>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        use burn_gdn2::{Fallback, Fused};
        // The read scale FLA's official KDA layer runs at, folded the way the
        // gdn2 kernels fold it (causal mask + read) — see `fla_read_scale`
        // (src/lib.rs). `q` is `[B, H, T, K]`, so K is its last dim.
        let [_, _, _, k_dim] = q.shape().dims::<4>();
        let scale = crate::fla_read_scale(k_dim);
        if is_bare_cuda::<B>() {
            // Numerical limit of the reused GDN-2 kernel: K/exp(cumsum(g))
            // underflows f32 once cumsum(g) < -88, i.e. chunk > 17 at the K3
            // floor g = -5. FlashKDA picked chunk 16 for exactly this reason.
            if chunk_size > 16 {
                return Fused::Fallback(Fallback::KernelLimits);
            }
            return match burn_gdn2::kernel::chunk_cube::cuda::fused_chunk_forward::<B>(
                q, k, v, log_alpha, beta_k, beta_v, state, scale, chunk_size,
            ) {
                Some(r) => Fused::Fused(r),
                None => Fused::Fallback(Fallback::KernelLimits),
            };
        }
        #[cfg(feature = "autodiff")]
        {
            burn_gdn2::chunk_dispatch::<B>(
                q, k, v, log_alpha, beta_k, beta_v, state, scale, chunk_size,
            )
        }
        #[cfg(not(feature = "autodiff"))]
        {
            let _ = (q, k, v, log_alpha, beta_k, beta_v, state, chunk_size, k_dim);
            Fused::Fallback(Fallback::NotCuda)
        }
    }
}
