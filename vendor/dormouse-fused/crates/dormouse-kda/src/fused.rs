//! Fused CUDA chunked KDA forward.
//!
//! KDA's chunkwise form (Kimi Linear Eqs 6-9) is algebraically identical to
//! the GDN-2 WY chunk algorithm under the mapping:
//!   g (log decay)  = log(alpha)        (channel-wise retention factors)
//!   b              = beta_k            (erase strength, key channels)
//!   w_gate         = beta_v            (write strength: KDA's pseudo-value is
//!                                       U = (I+T')^{-1}(β⊙V), NOT V)
//!   scale          = 1                 (see the OPEN note below)
//! With those inputs, `chunk_wy_forward` computes exactly
//!   A  = Tril((Q⊙Γ)(K/Γ)^T)
//!   W  = (I + T')^{-1}(β⊙Γ⊙K),  U = (I + T')^{-1}(β⊙V)
//!   S += Diag(Γ^C)S + (Γ⊙K)^T(U - W S),  O = (Γ⊙Q)S + A(U - W S)
//! so the existing GDN-2 fused kernels are reused as-is.
//!
//! The kernel is only used on the bare CUDA `CubeBackend`; everything else
//! falls back to the tensor-ops chunk path.
//!
//! # `scale = 1` here is OPEN, and the reason this file used to give for it was FALSE
//!
//! This file said "no softmax scale in KDA". That is **false against both
//! upstreams**:
//!
//! - `fla/ops/kda/chunk.py:474-475` and `fla/ops/kda/fused_recurrent.py:261-262`
//!   both contain `if scale is None: scale = K ** -0.5`;
//! - `fla/layers/kda.py:262-278` calls `chunk_kda(...)` with **no `scale`
//!   argument**, so the official KDA layer runs at `head_k_dim**-0.5`;
//! - `fla/ops/kda/naive.py:57` folds it into `q` before the recurrence loop.
//!
//! `dormouse-gdn2` itself does use `d_k**-0.5` (`gdn2/src/module.rs:299`); dormouse-kda
//! is the only place in the library that overrides it to `1.0`.
//!
//! WHY THE MODEL HAS NOT NOTICED: the factor enters only the read `o = q·S`, so
//! it is one constant on the attention output, and the very next thing this
//! crate does is `KdaModule::output`'s RMSNorm, which is invariant to a
//! constant rescale — `c·o / sqrt(mean((c·o)²) + eps) = o / sqrt(mean(o²) +
//! eps/c²)` — so it is absorbed to `O(eps / mean(o²))` ≈ `O(1e-5)`. That is
//! exactly why no loss curve, seed comparison or in-crate test can see it, and
//! also why it is still real: anything reading the raw attention output sees a
//! tensor `head_k_dim**0.5` too large (2.83× at K=8, 8× at K=64).
//!
//! The mechanism is not missing — `chunk_wy_forward` honours the scale when
//! asked, and `tests/kda_oracle.rs::chunked_wy_honours_the_read_scale_when_asked`
//! is green against FLA's own default-scale row. Only the ARGUMENT is. Two
//! tests carry this as RED ON PURPOSE (`read_scale_matches_fla_reference`,
//! `chunked_wy_applies_no_read_scale`). Changing `1.0` moves every number
//! derived from this crate, so it is the owner's call.
//! `docs/reviews/2026-09-30-kda-formula-audit.md` §3.2.

#[cfg(feature = "cuda")]
pub mod cuda {
    use burn::backend::{Backend, DispatchKindConversion};
    use burn::tensor::{DispatchTensor, Tensor};
    use std::any::TypeId;

    pub type CudaBare = dormouse_gdn2::CudaBare;

    /// `true` when `B` IS the bare CUDA backend (no autodiff wrapper). A
    /// `TypeId` is the right test HERE and only here: a bare backend has
    /// exactly one identity. It is NOT the test for "CUDA underneath" — an
    /// autodiff wrapper is a different type for every checkpointing strategy,
    /// which is what kept the fused path out of every training step
    /// (`Autodiff<CudaBare, BalancedCheckpointing>` is dormouse's trainer
    /// backend). For that see [`dormouse_gdn2::backend_matches`].
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
    /// [`dormouse_gdn2::cuda_dispatch`] strips the autodiff context, runs the
    /// fused kernels on the bare backend inside ONE autodiff node, and
    /// rebuilds the node on the caller's own strategy. Everything else
    /// (NdArray, ...) reports [`Fallback::NotCuda`] and the caller runs the
    /// tensor-ops chunk path.
    ///
    /// `Fused::Fused` means the single-node op ran; the launch counters
    /// ([`dormouse_gdn2::fused_calls`]) are the ground truth for whether the
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
    ) -> dormouse_gdn2::Fused<(Tensor<4>, Tensor<4>)>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        use dormouse_gdn2::{Fallback, Fused};
        if is_bare_cuda::<B>() {
            // Numerical limit of the reused GDN-2 kernel: K/exp(cumsum(g))
            // underflows f32 once cumsum(g) < -88, i.e. chunk > 17 at the K3
            // floor g = -5. FlashKDA picked chunk 16 for exactly this reason.
            if chunk_size > 16 {
                return Fused::Fallback(Fallback::KernelLimits);
            }
            return match dormouse_gdn2::kernel::chunk_cube::cuda::fused_chunk_forward::<B>(
                q, k, v, log_alpha, beta_k, beta_v, state, 1.0, chunk_size,
            ) {
                Some(r) => Fused::Fused(r),
                None => Fused::Fallback(Fallback::KernelLimits),
            };
        }
        #[cfg(feature = "autodiff")]
        {
            dormouse_gdn2::chunk_dispatch::<B>(
                q, k, v, log_alpha, beta_k, beta_v, state, 1.0, chunk_size,
            )
        }
        #[cfg(not(feature = "autodiff"))]
        {
            let _ = (q, k, v, log_alpha, beta_k, beta_v, state, chunk_size);
            Fused::Fallback(Fallback::NotCuda)
        }
    }
}
