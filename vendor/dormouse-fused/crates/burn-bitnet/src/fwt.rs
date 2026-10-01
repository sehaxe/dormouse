//! Fast Walsh–Hadamard Transform (FWHT), BitNet v2's outlier-suppressing
//! rotation (arXiv 2504.18415): butterfly tensor path, a generic-backend
//! entry for the autodiff fallback, and an explicit Hadamard-matrix
//! reference used by the CUDA-gated equivalence tests.

use burn::tensor::Tensor;

/// FWT dispatch: fused CUDA kernel when available, butterfly tensor path
/// otherwise. Orthogonal (normalized by 1/sqrt(p)), so applying it twice is
/// the identity.
///
/// The autodiff arm probes BOTH checkpointing strategies, because the node it
/// builds lives on the caller's strategy and `Autodiff<Inner>`'s default type
/// parameter is `NoCheckpointing` — probing only that one silently sent
/// dormouse (`Autodiff<CudaBare, BalancedCheckpointing>`) to the tensor path.
#[cfg(all(feature = "cuda", feature = "autodiff"))]
macro_rules! fwt_probe {
    ($x:expr, $p:expr) => {{
        use burn::tensor::DispatchTensor;
        use burn_autodiff::checkpoint::strategy::{
            BalancedCheckpointing, CheckpointStrategy, NoCheckpointing,
        };
        type CudaBare = burn_cubecl::CubeBackend;
        // Both conversions must exist for the probe to compile; assert it in
        // the bound rather than discovering it as a silent `None`.
        fn assert_conv<S: CheckpointStrategy>()
        where
            DispatchTensor: burn::backend::DispatchKindConversion<AutodiffAlias<S>>,
        {
        }
        type AutodiffAlias<S> = burn_autodiff::Autodiff<CudaBare, S>;
        assert_conv::<NoCheckpointing>();
        assert_conv::<BalancedCheckpointing>();
        if let Some(r) =
            crate::fwt_cuda::fwt_autodiff_s::<CudaBare, NoCheckpointing>($x.clone(), $p)
        {
            Some(r)
        } else {
            crate::fwt_cuda::fwt_autodiff_s::<CudaBare, BalancedCheckpointing>($x.clone(), $p)
        }
    }};
}

pub fn fast_walsh_hadamard(x: Tensor<2>) -> Tensor<2> {
    let [_, d] = x.dims();
    let _p = d.next_power_of_two();
    #[cfg(all(feature = "cuda", feature = "autodiff"))]
    {
        if let Some(r) = fwt_probe!(x, _p) {
            return r;
        }
    }
    #[cfg(feature = "cuda")]
    {
        if let Some(r) = crate::fwt_cuda::fwt_cuda(&x, _p) {
            return r;
        }
    }
    fast_walsh_hadamard_tensor_body(x)
}

/// Pure tensor-path FWT (shared by the autodiff fallback). `p` is ignored: the
/// row width's next power of two is derived from the tensor itself.
pub fn fast_walsh_hadamard_tensor<B: burn::backend::Backend>(x: Tensor<2>, _p: usize) -> Tensor<2>
where
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    fast_walsh_hadamard_tensor_body(x)
}

/// Independent FWT reference: explicit Hadamard matrix via Kronecker
/// products, `y = x_pad @ H_p / sqrt(p)`. No butterfly recursion, so it can
/// catch a broken fused kernel (the tensor butterfly and the CUDA kernel
/// share the same structure). Used by the CUDA-gated tests.
pub fn fwt_reference_naive(x: &Tensor<2>) -> Tensor<2> {
    let [n, d] = x.dims();
    let p = d.next_power_of_two();
    let dev = x.device();
    let xp = if p != d {
        Tensor::cat(vec![x.clone(), Tensor::zeros([n, p - d], &dev)], 1)
    } else {
        x.clone()
    };
    // H_1 = [1]; H_{2k} = [[H_k, H_k], [H_k, -H_k]]
    let mut h = Tensor::<2>::ones([1, 1], &dev);
    let mut k = 1;
    while k < p {
        let top = Tensor::cat(vec![h.clone(), h.clone()], 1);
        let bot = Tensor::cat(vec![h.clone(), h.clone().mul_scalar(-1.0)], 1);
        h = Tensor::cat(vec![top, bot], 0);
        k *= 2;
    }
    let y = xp.matmul(h).div_scalar((p as f32).sqrt());
    if p != d {
        y.slice([0..n, 0..d])
    } else {
        y
    }
}

fn fast_walsh_hadamard_tensor_body(x: Tensor<2>) -> Tensor<2> {
    let [n, d] = x.dims();
    let p = d.next_power_of_two();
    let dev = x.device();
    let (x_pad, was_padded) = if p != d {
        (
            Tensor::cat(vec![x, Tensor::zeros([n, p - d], &dev)], 1),
            true,
        )
    } else {
        (x, false)
    };
    let mut h = 1usize;
    let mut out = x_pad;
    while h < p {
        let step = 2 * h;
        let r = out.reshape([n, p / step, 2, h]);
        let l = r
            .clone()
            .slice([0..n, 0..(p / step), 0..1, 0..h])
            .squeeze_dim::<3>(2);
        let rt = r
            .slice([0..n, 0..(p / step), 1..2, 0..h])
            .squeeze_dim::<3>(2);
        out = Tensor::cat(
            vec![
                (l.clone() + rt.clone()).unsqueeze_dim::<4>(2),
                (l - rt).unsqueeze_dim::<4>(2),
            ],
            2,
        )
        .reshape([n, p]);
        h = step;
    }
    let out = out.div_scalar((p as f32).sqrt());
    if was_padded {
        out.slice([0..n, 0..d])
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Tensor};

    fn dev() -> burn::tensor::Device {
        burn::tensor::Device::ndarray()
    }

    #[test]
    fn hadamard_roundtrip() {
        let x = Tensor::<2>::ones([4, 8], &dev());
        let h2x = fast_walsh_hadamard(fast_walsh_hadamard(x));
        let v: Vec<f32> = h2x
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        for (i, val) in v.iter().enumerate() {
            assert!((val - 1.0).abs() < 0.1, "idx {i}: {val}");
        }
    }

    #[test]
    fn hadamard_non_power_of_two() {
        let h2x = fast_walsh_hadamard(fast_walsh_hadamard(Tensor::<2>::ones([2, 7], &dev())));
        assert_eq!(h2x.dims(), [2, 7]);
    }

    #[test]
    fn fwt_reference_matches_butterfly() {
        // naive matrix reference vs the butterfly tensor path on ndarray
        // (f32 matmul is near-exact there; CUDA matmul would run TF32).
        // Covers the pad branch (d=12 -> p=16) and power-of-two (d=16).
        let dev = burn::tensor::Device::ndarray();
        for d in [12usize, 16] {
            let x = Tensor::<2>::random([4, d], Distribution::Normal(0.0, 1.0), &dev);
            let a = fwt_reference_naive(&x);
            let b = fast_walsh_hadamard(x);
            let diff: f32 = (a - b).abs().max().into_scalar();
            assert!(diff < 1e-4, "d={d}: naive vs butterfly maxdiff {diff}");
        }
    }
}
