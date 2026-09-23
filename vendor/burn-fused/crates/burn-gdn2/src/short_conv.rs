use burn::tensor::activation::silu;
use burn::tensor::Tensor;

/// Number of taps of the short convolution (kernel_size=4).
pub const SHORT_CONV_KERNEL: usize = 4;
/// Width of the causal context cache: `kernel_size - 1`.
pub const SHORT_CONV_CACHE: usize = 3;

/// 1D causal depthwise convolution with kernel_size=4 and SiLU activation.
///
/// Returns `(output, cache)` where `cache` carries the last
/// `SHORT_CONV_CACHE` input values `[B, 3, C]`. Pass the cache from a previous
/// call to continue a sequence (incremental decoding); pass `None` for a fresh
/// prefill (the beginning of the sequence is padded by replicating the first
/// token). The returned cache makes token-by-token decoding produce exactly
/// the same outputs as one forward pass over the full sequence.
///
/// # Panics
///
/// Panics if `weight` does not have shape `[C, SHORT_CONV_KERNEL]` or if
/// `cache` does not have shape `[B, SHORT_CONV_CACHE, C]`.
pub fn short_conv_1d(
    x: Tensor<3>,
    weight: Tensor<2>,
    cache: Option<&Tensor<3>>,
) -> (Tensor<3>, Tensor<3>) {
    let [b, t, c] = x.shape().dims();
    debug_assert_eq!(weight.shape().dims::<2>(), [c, SHORT_CONV_KERNEL]);

    let (x_pad, new_cache) = match cache {
        Some(prev) => {
            let [cb, cc, _] = prev.shape().dims::<3>();
            debug_assert_eq!([cb, cc], [b, SHORT_CONV_CACHE]);
            let combined = Tensor::cat(vec![prev.clone(), x.clone()], 1);
            (
                combined.clone(),
                combined
                    .clone()
                    .slice([0..b, t..t + SHORT_CONV_CACHE, 0..c]),
            )
        }
        None => {
            let pad = x
                .clone()
                .slice([0..b, 0..1, 0..c])
                .repeat(&[1, SHORT_CONV_CACHE, 1]);
            let combined = Tensor::cat(vec![pad, x], 1);
            (
                combined.clone(),
                combined
                    .clone()
                    .slice([0..b, t..t + SHORT_CONV_CACHE, 0..c]),
            )
        }
    };

    let device = x_pad.device();
    let mut out = Tensor::zeros([b, t, c], &device);
    for i in 0..SHORT_CONV_KERNEL {
        let x_slice = x_pad.clone().slice([0..b, i..i + t, 0..c]);
        let w_i = weight.clone().slice([0..c, i..i + 1]).reshape([1, 1, c]);
        out = out + x_slice * w_i;
    }
    (silu(out), new_cache)
}
