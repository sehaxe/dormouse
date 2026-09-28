//! Fused CUDA chunked KDA forward.
//!
//! KDA's chunkwise form (Kimi Linear Eqs 6-9) is algebraically identical to
//! the GDN-2 WY chunk algorithm under the mapping:
//!   g (log decay)  = log(alpha)        (channel-wise retention factors)
//!   b              = beta_k            (erase strength, key channels)
//!   w_gate         = beta_v            (write strength: KDA's pseudo-value is
//!                                       U = (I+T')^{-1}(β⊙V), NOT V)
//!   scale          = 1                 (no softmax scale in KDA)
//! With those inputs, `chunk_wy_forward` computes exactly
//!   A  = Tril((Q⊙Γ)(K/Γ)^T)
//!   W  = (I + T')^{-1}(β⊙Γ⊙K),  U = (I + T')^{-1}(β⊙V)
//!   S += Diag(Γ^C)S + (Γ⊙K)^T(U - W S),  O = (Γ⊙Q)S + A(U - W S)
//! so the existing GDN-2 fused kernels are reused as-is.
//!
//! The kernel is only used on the bare CUDA `CubeBackend`; everything else
//! falls back to the tensor-ops chunk path.

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
        if is_bare_cuda::<B>() {
            // Numerical limit of the reused GDN-2 kernel: K/exp(cumsum(g))
            // underflows f32 once cumsum(g) < -88, i.e. chunk > 17 at the K3
            // floor g = -5. FlashKDA picked chunk 16 for exactly this reason.
            if chunk_size > 16 {
                return Fused::Fallback(Fallback::KernelLimits);
            }
            return match burn_gdn2::kernel::chunk_cube::cuda::fused_chunk_forward::<B>(
                q, k, v, log_alpha, beta_k, beta_v, state, 1.0, chunk_size,
            ) {
                Some(r) => Fused::Fused(r),
                None => Fused::Fallback(Fallback::KernelLimits),
            };
        }
        #[cfg(feature = "autodiff")]
        {
            burn_gdn2::chunk_dispatch::<B>(
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
