//! Sinkhorn-Knopp entropic projection onto the Birkhoff polytope (Eq 9).

use burn::tensor::Tensor;

/// Sinkhorn-Knopp iterations used in the paper (App. A.1: t_max = 20).
pub const SINKHORN_ITERS: usize = 20;

/// Sinkhorn-Knopp entropic projection (Eq 9) onto the Birkhoff polytope.
///
/// `logits`: `[..., n, n]` - `M^(0) = exp(logits)` followed by `iters`
/// alternating row/column normalizations so rows and columns sum to 1.
///
/// Runs in the LOG domain: each normalization becomes `m -= logsumexp(m)`,
/// which is shift-invariant and never overflows. The naive exp-then-divide
/// form breaks twice at extreme logits (the hyper-network output
/// `alpha*x_norm*phi_res + b_res` is unbounded): entries above exp(88)
/// overflow f32 outright, and even bounded inputs whose within-row dynamic
/// range exceeds f32's exponent span flush to exact zeros (FTZ), after which
/// a zero column sum makes the column pass emit inf and the next pass NaN.
/// The final `.exp()` maps the converged log-matrix back to a doubly
/// stochastic (up to fp) matrix.
pub fn sinkhorn_knopp(logits: Tensor<4>, iters: usize) -> Tensor<4> {
    #[cfg(all(feature = "cuda", feature = "autodiff"))]
    {
        type CudaBare = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
        if let Some(out) =
            crate::sinkhorn_cuda::sinkhorn_autodiff::<CudaBare>(logits.clone(), iters)
        {
            return out;
        }
    }
    // `max_dim`/`sum_dim` keep the reduced axis (size 1), broadcasting below.
    let mut m = logits;
    for _ in 0..iters {
        // row normalization in log-space (last dim)
        let max_r = m.clone().max_dim(3);
        let lse_r = m
            .clone()
            .sub(max_r.clone())
            .exp()
            .sum_dim(3)
            .log()
            .add(max_r);
        m = m.sub(lse_r);
        // column normalization in log-space (second-to-last dim)
        let max_c = m.clone().max_dim(2);
        let lse_c = m
            .clone()
            .sub(max_c.clone())
            .exp()
            .sum_dim(2)
            .log()
            .add(max_c);
        m = m.sub(lse_c);
    }
    m.exp()
}
