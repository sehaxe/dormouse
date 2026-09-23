//! RoPE application to hidden states and the precomputed module.

use burn::backend::Backend;
use burn::module::Module;
use burn::tensor::{Device, Tensor};

use crate::freqs::{precompute_freqs, precompute_freqs_yarn};

/// Apply `RoPE` to `[B, T, D]` tensor. Reshapes to `[B, T, NH, HD]` internally.
pub fn apply_rope_3d<B: Backend>(
    x: Tensor<3>,
    cos: Tensor<2>,
    sin: Tensor<2>,
    n_heads: usize,
) -> Tensor<3>
where
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    let [b, t, d] = x.dims();
    let hd = d / n_heads;
    let out = apply_rope_4d::<B>(x.reshape([b, t, n_heads, hd]), cos, sin);
    let [b2, t2, _, _] = out.dims();
    out.reshape([b2, t2, d])
}

/// Apply `RoPE` to `[B, T, NH, HD]` tensor.
/// `cos`, `sin` must have shape `[max_seq_len, HD/2]`.
pub fn apply_rope_4d<B: Backend>(x: Tensor<4>, cos: Tensor<2>, sin: Tensor<2>) -> Tensor<4>
where
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    #[cfg(all(feature = "cuda", feature = "autodiff"))]
    {
        type CudaBare = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
        if let Some(r) =
            crate::rope_cuda::rope_autodiff::<CudaBare>(x.clone(), cos.clone(), sin.clone())
        {
            return r;
        }
    }
    #[cfg(feature = "cuda")]
    {
        if let Some(r) = crate::rope_cuda::rope_cuda::<B>(x.clone(), cos.clone(), sin.clone()) {
            return r;
        }
    }
    let [b, t, nh, hd] = x.dims();
    let half = hd / 2;
    let x1 = x.clone().slice([0..b, 0..t, 0..nh, 0..half]);
    let x2 = x.slice([0..b, 0..t, 0..nh, half..hd]);
    let c = cos.slice([0..t, 0..half]).reshape([1, t, 1, half]);
    let s = sin.slice([0..t, 0..half]).reshape([1, t, 1, half]);
    let rot1 = x1.clone().mul(c.clone()).sub(x2.clone().mul(s.clone()));
    let rot2 = x1.mul(s).add(x2.mul(c));
    Tensor::cat(vec![rot1, rot2], 3).reshape([b, t, nh, hd])
}

/// Precomputed rotary position embedding module.
#[derive(Module, Debug)]
pub struct RotaryEmbedding {
    pub cos: Tensor<2>,
    pub sin: Tensor<2>,
    pub n_heads: usize,
}

impl RotaryEmbedding {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        max_seq_len: usize,
        base: f64,
        device: &Device,
    ) -> Self {
        let head_dim = d_model / n_heads;
        let (cos, sin) = precompute_freqs(head_dim, max_seq_len, base, device);
        Self { cos, sin, n_heads }
    }

    /// YaRN-extended embedding (NTK-by-parts + temperature, see
    /// [`precompute_freqs_yarn`]).
    #[allow(clippy::too_many_arguments)]
    pub fn yarn(
        d_model: usize,
        n_heads: usize,
        max_seq_len: usize,
        base: f64,
        scale: f64,
        orig_len: usize,
        beta_fast: f64,
        beta_slow: f64,
        device: &Device,
    ) -> Self {
        let head_dim = d_model / n_heads;
        let (cos, sin) = precompute_freqs_yarn(
            head_dim,
            max_seq_len,
            base,
            scale,
            orig_len,
            beta_fast,
            beta_slow,
            device,
        );
        Self { cos, sin, n_heads }
    }

    /// Apply `RoPE` to a single `[B, T, D]` tensor.
    pub fn forward<B: Backend>(&self, x: Tensor<3>) -> Tensor<3>
    where
        burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
    {
        apply_rope_3d::<B>(x, self.cos.clone(), self.sin.clone(), self.n_heads)
    }

    /// Apply `RoPE` to query and key `[B, T, D]` tensors.
    pub fn forward_qk<B: Backend>(&self, q: Tensor<3>, k: Tensor<3>) -> (Tensor<3>, Tensor<3>)
    where
        burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
    {
        (
            apply_rope_3d::<B>(q, self.cos.clone(), self.sin.clone(), self.n_heads),
            apply_rope_3d::<B>(k, self.cos.clone(), self.sin.clone(), self.n_heads),
        )
    }
}
