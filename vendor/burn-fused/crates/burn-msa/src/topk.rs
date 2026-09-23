use burn::backend::Backend;
use burn::tensor::{Device, Int, Tensor};

/// Block id for each token position: `t / block_size`, computed without
/// division. cubecl's integer division returns `(t / bs) * bs`, which corrupts
/// any kernel that divides, so build `[0,0,..,1,1,..]` from a repeated arange.
/// Returns `[seq]`.
pub(crate) fn block_of_token(seq: usize, block_size: usize, device: &Device) -> Tensor<1, Int> {
    let n_blocks = seq.div_ceil(block_size);
    Tensor::<1, Int>::arange(0..n_blocks as i64, device)
        .unsqueeze_dim::<2>(1)
        .repeat_dim(1, block_size)
        .reshape::<1, _>([n_blocks * block_size])
        .slice(0..seq)
}

/// Indices of the `k` largest elements along `dim` (last axis) of a 3D tensor.
///
/// `topk_with_indices` must NOT be used on cubecl — it falls back to a
/// synchronous host sort (`sort_with_indices`) whose read can fail inside
/// autodiff under async load, panicking and poisoning the CUDA context
/// (subsequent kernels fail with CUDA_ERROR_ILLEGAL_ADDRESS). Host backends
/// (ndarray) implement no `argtopk`, so this stays generic.
fn argtopk3(x: Tensor<3>, k: usize, dim: usize) -> Tensor<3, Int> {
    // ponytail: cubecl's ArgTopK reduce returns garbage indices for some
    // output rows (verified on 0.11.0-pre.1) which then OOB a downstream
    // scatter and kill the CUDA context. ArgMax is correct, so pick the k
    // maxima by k masked-argmax passes instead (k is tiny, <=3).
    #[cfg(feature = "cuda")]
    {
        if dim == 2 {
            if let Some(idx) = argtopk3_cuda(&x, k) {
                return idx;
            }
        }
    }
    let [b, s, nb] = x.dims();
    // Run the masked-argmax loop on the bare inner backend. The one-hot mask
    // is rebuilt from Int data on the bare device (`one_hot` round-trips
    // through to_data/from_data, and Int tensors are pass-through in
    // autodiff), so a tracked `work` would mix an Autodiff float with a bare
    // float in `work.sub(oh)` and panic at dispatch. `no_grad()` strips the
    // wrapper (a no-op on bare tensors); `detach()` would keep the Autodiff
    // kind and still panic. Indices carry no gradient, so running bare is
    // numerically identical, and Int results stay valid on the caller's
    // backend.
    let mut work = x.no_grad();
    let mut parts: Vec<Tensor<3, Int>> = Vec::with_capacity(k);
    for _ in 0..k {
        // cubecl reduce ops return I32; burn's Int contract is i64, so cast
        // each pick back (downstream math and the tests rely on i64 values).
        let idx = work.clone().argmax(dim).cast(burn::tensor::IntDType::I64); // [b, s, 1]
        let oh = idx.clone().one_hot::<4>(nb).float(); // [b, s, 1, nb]
        let [_, _, _, n2] = oh.dims();
        work = work.sub(oh.reshape([b, s, n2]).mul_scalar(1e30_f32));
        parts.push(idx);
    }
    Tensor::cat(parts, 2)
}

/// CUDA fast path for `argtopk3(dim = 2)`: the fused `exp_free_topk_kernel`
/// picks the k largest per row in one launch (per-warp heap + warp reduce)
/// instead of k masked-argmax passes over [B, S, nb]. Returns `None` when the
/// tensor is not on the bare CUDA backend or the shape is outside the
/// kernel's contract; the caller then falls back to the masked-argmax loop.
#[cfg(feature = "cuda")]
fn argtopk3_cuda(x: &Tensor<3>, k: usize) -> Option<Tensor<3, Int>> {
    // ponytail: exp_free_topk_kernel has a race on heap building for
    // n_blocks >8 (shared heap updated by 32 lanes without sync) and
    // produces -1 indices that OOB the downstream gather (t>256 crash).
    // Disable fast path until kernel is fixed; fallback masked-argmax is
    // correct and cheap (k<=8, 8 passes).
    return None;
    #[allow(unreachable_code)]
    {
    use burn_cubecl::tensor::CubeTensor;
    use std::any::Any;

    type CudaBare = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
    let [b, s, nb] = x.dims();
    let n_rows = b * s;
    if n_rows == 0 || nb == 0 || k == 0 || k > nb {
        return None;
    }
    // The scores can be a permuted (non-contiguous) view or a row-pitched
    // buffer (cubecl pads 2D row widths: nb=6 -> 24B rows -> pitch 32B,
    // stride 8). Reshaping to 1D materializes a dense copy whenever the
    // tensor is not already contiguous (burn-std `reshape_action`: rank
    // shrink -> Recompute -> copy_into), so the kernel's flat `row*nb+j`
    // indexing only ever sees a dense [n_rows, nb] buffer.
    let flat = x.clone().reshape::<1, _>([n_rows * nb]);
    let prim = flat.try_into_primitive::<CudaBare>().ok()?;
    let xc = (&prim as &dyn Any).downcast_ref::<CubeTensor<cubecl::cuda::CudaRuntime>>()?;
    let indices = Tensor::<1, Int>::empty([n_rows * k], &x.device());
    let ip = indices.clone().try_into_primitive::<CudaBare>().ok()?;
    let ic = (&ip as &dyn Any).downcast_ref::<CubeTensor<cubecl::cuda::CudaRuntime>>()?;
    let client = xc.client.clone();
    unsafe {
        crate::kernel::topk_select::launch_exp_free_topk(
            &client,
            &xc.handle,
            &ic.handle,
            n_rows as u32,
            nb as u32,
            k as u32,
        );
    }
    Some(
        indices
            .reshape::<3, _>([b, s, k])
            .cast(burn::tensor::IntDType::I64),
    )
    }
}

/// GPU-sync-free top-K block selector. Now accepts 4D tensor directly.
#[derive(Debug)]
pub struct TopKSelector {
    pub topk: usize,
    pub block_size: usize,
    pub force_local_block: bool,
}

impl TopKSelector {
    pub fn new(topk: usize, block_size: usize, force_local_block: bool) -> Self {
        Self {
            topk,
            block_size,
            force_local_block,
        }
    }

    /// `block_scores`: [batch, n_heads_kv, seq_q, n_blocks]
    /// Returns: [batch, n_heads_kv, seq_q, topk_actual]
    pub fn select<B: Backend>(&self, block_scores: Tensor<4>) -> Tensor<4, Int> {
        let [batch, h, seq_q, n_blocks] = block_scores.dims();
        let topk = self.topk.min(n_blocks);
        let device = block_scores.device();

        // Flatten batch and heads for topk
        let scores_3d = block_scores
            .permute([0, 1, 3, 2]) // [B, H, Nb, Sq]
            .reshape::<3, _>([batch * h, n_blocks, seq_q])
            .permute([0, 2, 1]); // [B*H, Sq, Nb]

        // `argtopk` is a GPU reduce (burn-cuda 0.21 implements ArgTopK);
        // `topk_with_indices` must NOT be used here — it falls back to a
        // synchronous host sort whose read can fail inside autodiff under
        // async load, panicking and poisoning the CUDA context (subsequent
        // kernels fail with CUDA_ERROR_ILLEGAL_ADDRESS).
        let indices_3d = if topk >= n_blocks {
            // k == size: every block is selected — no topk needed at all. This
            // shortcut also hides the argtopk3 fallback path: the fallback
            // mixes tracked scores with a bare one-hot mask on autodiff
            // backends and panics, but only n_blocks > topk reaches it (e.g.
            // seq > 256 at block_size 32 with topk 8), so small configs never
            // crash.
            Tensor::<1, Int>::arange(0..n_blocks as i64, &device)
                .reshape::<3, _>([1, 1, n_blocks])
                .repeat_dim(0, batch * h)
                .repeat_dim(1, seq_q)
        } else if self.force_local_block && seq_q > 0 && n_blocks > 0 && topk > 1 {
            let local_block =
                block_of_token(seq_q, self.block_size, &device).reshape::<2, _>([seq_q, 1]);
            let block_idx = Tensor::<1, Int>::arange(0..n_blocks as i64, &device)
                .reshape::<2, _>([1, n_blocks]);
            let bias = local_block
                .equal(block_idx)
                .float()
                .mul_scalar(1e4)
                .unsqueeze::<3>();
            argtopk3(scores_3d.add(bias), topk, 2)
        } else {
            argtopk3(scores_3d, topk, 2)
        };

        let actual_topk = indices_3d.dims()[2];
        indices_3d.reshape::<4, _>([batch, h, seq_q, actual_topk])
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    fn host_i64(t: Tensor<3, Int>) -> Vec<i64> {
        t.into_data()
            .bytes
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn topk_cuda_matches_fallback() {
        let dev = Device::default();
        for (b, s, nb, k) in [
            (2usize, 4, 13, 3),
            (1, 1, 257, 2),
            (3, 7, 64, 5),
            (1, 2, 4, 4),
            // Non-pow2 nb (24B/48B rows -> pitch 32B/64B): pitched buffers
            // must be densified before the raw launch. nb=6 is exactly
            // seq=768 / block=128.
            (2, 3, 6, 2),
            (1, 1, 12, 3),
            (2, 4, 8, 3), // pow2 control
        ] {
            let x = Tensor::<3>::random([b, s, nb], Distribution::Normal(0.0, 1.0), &dev);
            let fast = argtopk3_cuda(&x, k).expect("cuda topk fast path should run");
            let slow = argtopk3(x, k, 2);
            let fv = host_i64(fast);
            let sv = host_i64(slow);
            for row in 0..b * s {
                let mut got: Vec<i64> = fv[row * k..row * k + k].to_vec();
                let mut want: Vec<i64> = sv[row * k..row * k + k].to_vec();
                got.sort_unstable();
                want.sort_unstable();
                assert_eq!(got, want, "row {row} of [{b},{s},{nb}] k={k}");
            }
        }
    }

    #[test]
    fn topk_cuda_permuted_view_matches_fallback() {
        let dev = Device::default();
        // nb=6 (pitched to stride 8) behind a dim-permuted view: the fast
        // path must materialize a dense copy instead of reading the
        // pitched/permuted buffer with flat indexing.
        let x = Tensor::<3>::random([2, 4, 6], Distribution::Normal(0.0, 1.0), &dev);
        let v = x.permute([1, 0, 2]); // [4, 2, 6], non-dense view, last dim 6
        let fast = argtopk3_cuda(&v, 2).expect("cuda topk fast path should run");
        let slow = argtopk3(v, 2, 2);
        let fv = host_i64(fast);
        let sv = host_i64(slow);
        for row in 0..8 {
            let mut got: Vec<i64> = fv[row * 2..row * 2 + 2].to_vec();
            let mut want: Vec<i64> = sv[row * 2..row * 2 + 2].to_vec();
            got.sort_unstable();
            want.sort_unstable();
            assert_eq!(got, want, "row {row}");
        }
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn topk_autodiff_cuda_fallback_matches_host() {
        use burn::backend::autodiff::Autodiff;

        type CudaBare = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
        type CudaAd = Autodiff<CudaBare>;

        let adev = Device::default().autodiff();
        // n_blocks (16) > topk (8): the `topk >= n_blocks` shortcut does not
        // apply, so argtopk3's masked-argmax fallback runs. On an autodiff
        // CUDA tensor that fallback used to panic, mixing the tracked scores
        // with the bare one-hot mask.
        let (b, h, sq, n_blocks, topk) = (1usize, 1usize, 512usize, 16usize, 8usize);
        let sel = TopKSelector::new(topk, 32, false);
        let x = Tensor::<4>::random([b, h, sq, n_blocks], Distribution::Normal(0.0, 1.0), &adev)
            .require_grad();
        let idx = sel.select::<CudaAd>(x.clone());
        assert_eq!(idx.dims(), [b, h, sq, topk]);
        // host reference: the k largest block ids per row of the same data
        let hx: Vec<f32> = x
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let iv = host_i64_4d(idx);
        for row in 0..b * h * sq {
            let mut v: Vec<(f32, usize)> =
                (0..n_blocks).map(|j| (hx[row * n_blocks + j], j)).collect();
            v.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut want: Vec<i64> = v[..topk].iter().map(|(_, j)| *j as i64).collect();
            let mut got: Vec<i64> = iv[row * topk..row * topk + topk].to_vec();
            got.sort_unstable();
            want.sort_unstable();
            assert_eq!(got, want, "row {row}");
        }
    }

    // Only the autodiff-gated test above calls this helper; without that
    // feature it would be dead code and fail `-D warnings`.
    #[cfg(feature = "autodiff")]
    fn host_i64_4d(t: Tensor<4, Int>) -> Vec<i64> {
        t.into_data()
            .bytes
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
}
