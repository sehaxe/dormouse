//! Sparse-BitNet (arXiv 2603.05168): dynamic N:M semi-structured sparsity +
//! ternary quantization with Dual straight-through estimators (STE).
//!
//! The paper builds on the b1.58 quantizer `W_q = RoundClip(W/γ, -1, +1) * γ`
//! with `γ = mean(|W|)` (arXiv 2402.17764; implemented exactly in
//! [`crate::weight_quant_b158`]). This crate's mask path composes with the
//! sign-centered variant [`crate::weight_quant_ternary`] (the historical
//! default here); Sparse-BitNet adds a per-block
//! top-N magnitude mask (computed from the continuous master weights, not the
//! quantized ones — ties would destabilize) and applies it quant-then-mask so
//! the discrete weights are exactly N:M sparse for hardware layout.
//!
//! The key insight is Dual-STE: gradients flow straight through BOTH the
//! quantizer AND the mask (`dL/dW = dL/dW_eff`). Masked (pruned) weights still
//! receive gradients so they can grow back into the top-N; this differs from
//! gated-gradient pruning, where the mask multiplication would zero them.

use burn::module::{Module, Param};
use burn::tensor::{Distribution, IndexingUpdateOp, Int, Tensor};

/// Per-block N:M sparsity mask: for every contiguous group of `m` elements
/// along the last dim (the input dim), keep the `n` largest `|w|` as 1 and
/// zero the rest. Returns `{0, 1}` floats in the input's shape.
///
/// The mask is read off the **continuous** master weights (the paper: ties in
/// a quantized `W_q` would destabilize the top-N selection).
///
/// Non-divisible last dim: the row is padded with zeros to a multiple of `m`
/// (zero can't rank above a nonzero `|w|`, so real entries are unchanged;
/// padding is only ever selected in blocks with fewer than `n` nonzeros, which
/// is measure-zero on continuous weights), then the padding is sliced off.
pub fn compute_nm_mask(w: Tensor<2>, n: usize, m: usize) -> Tensor<2> {
    assert!(m > 0, "compute_nm_mask: m must be > 0");
    assert!(n <= m, "compute_nm_mask: n must be <= m");
    let [out, dim] = w.dims();
    if n >= m {
        // keep everything (N == M): `argtopk` asserts k < block size
        return Tensor::ones([out, dim], &w.device());
    }
    let blocks = dim.div_ceil(m);
    let padded = blocks * m;
    let w_p = if padded != dim {
        Tensor::cat(
            vec![w.clone(), Tensor::zeros([out, padded - dim], &w.device())],
            1,
        )
    } else {
        w.clone()
    };
    // [out, blocks, m] -> argtopk along the block dim -> [out, blocks, n]
    let idx = w_p.abs().reshape([out, blocks, m]).argtopk(n, 2);
    // Add on zeros: indices within a block are unique for continuous weights
    // (the mask deliberately reads the master, not the quantized weights, so
    // ties are measure-zero), giving exactly {0, 1}. Add is the only scatter
    // update every backend (incl. CubeCL) implements.
    let mask = Tensor::<3, Int>::zeros([out, blocks, m], &w.device()).scatter(
        2,
        idx,
        Tensor::<3, Int>::ones([out, blocks, n], &w.device()),
        IndexingUpdateOp::Add,
    );
    mask.float().reshape([out, padded]).slice([0..out, 0..dim])
}

/// Ternary quant + N:M mask as ONE autodiff node with the Dual-STE backward:
/// forward `ternary(w) * mask(w)`, backward `g_out` (unchanged) for `w`.
///
/// CUDA autodiff takes the fused op ([`crate::sparse::ad`]); every other
/// backend falls back to [`weight_quant_masked_tensor`], which yields the same
/// values and the same identity gradient through native autodiff.
pub fn weight_quant_masked(w: Tensor<2>, n: usize, m: usize) -> Tensor<2> {
    #[cfg(all(feature = "cuda", feature = "autodiff"))]
    {
        type CudaBare = burn_cubecl::CubeBackend;
        if let Some(r) = ad::weight_quant_masked_autodiff::<CudaBare>(w.clone(), n, m) {
            return r;
        }
    }
    weight_quant_masked_tensor(w, n, m)
}

/// Pure-tensor Dual-STE fallback (any backend, incl. non-CUDA autodiff).
///
/// Native autodiff of `t*m` would multiply the gradient by the mask (gated
/// pruning); the residual trick makes the gradient identity without a custom
/// op: `out = t*m + (w - detach(w)) * (1 - m)` has value `t*m` (the second
/// term is exactly zero) and gradient `m + (1 - m) = 1`, i.e. straight through
/// both the quantizer and the mask.
///
/// The ternary `t` is rebuilt with detached mean/scale so its native gradient
/// is *exactly* identity: `crate::weight_quant_ternary`'s mean/scale are
/// functions of `w` and would leak ~`1/N` cross-terms into the gradient.
fn weight_quant_masked_tensor(w: Tensor<2>, n: usize, m: usize) -> Tensor<2> {
    let wd = w.clone().detach();
    let scale = w.clone().abs().mean().unsqueeze_dims(&[0, 0]).detach();
    // t = RoundClip(w/(scale+eps), -1, +1) * scale — the b1.58/v2 Eq.1 form
    // this crate's Sparse-BitNet builds on (arXiv 2603.05168 references it).
    // The detached wrapper `w + (u - w_det)` gives value u and gradient
    // exactly 1 (Dual-STE straight-through).
    let u = wd
        .clone()
        .div(scale.clone())
        .round()
        .clamp(-1.0, 1.0)
        .mul(scale);
    let t = w.clone().add(u.sub(wd.clone()));
    let mask = compute_nm_mask(wd.clone(), n, m);
    t.mul(mask.clone())
        .add(w.sub(wd).mul(mask.neg().add_scalar(1.0)))
}

/// Sparse-BitLinear: `y = activation_quant(x) @ weight_quant_masked(w)`.
///
/// The weight is a ternary-quantized, N:M-sparse master; both the quantizer
/// and the mask use straight-through gradients (Dual-STE), so the full-precision
/// master keeps learning the pruned positions. Activations are quantized with
/// either the b1.58 8-bit absmax scheme ([`crate::activation_quant_8bit`],
/// the default) or the BitNet v2 4-bit Hadamard scheme ([`crate::quantize_4bit`],
/// arXiv 2504.18415), selected by [`Self::activation_bits`].
#[derive(Module, Debug)]
pub struct SparseBitLinear {
    /// Full-precision master `[out, in]` (quantized + masked per forward).
    pub weight: Param<Tensor<2>>,
    /// Keep the `n` largest `|w|` per block of `m` along the input dim.
    #[module(skip)]
    pub n: usize,
    /// N:M block size.
    #[module(skip)]
    pub m: usize,
    /// Activation quantization width: `4` = BitNet v2 Hadamard
    /// ([`crate::quantize_4bit`]), `>= 8` = b1.58 8-bit absmax. Any other
    /// value panics in [`Self::forward`].
    #[module(skip)]
    pub activation_bits: usize,
}

impl SparseBitLinear {
    /// Create a layer with an `N(0, 0.02)` master (BitNet-style small init).
    pub fn new(
        in_features: usize,
        out_features: usize,
        n: usize,
        m: usize,
        device: &burn::tensor::Device,
    ) -> Self {
        let weight = Param::from_tensor(Tensor::<2>::random(
            [out_features, in_features],
            Distribution::Normal(0.0, 0.02),
            device,
        ));
        Self {
            weight,
            n,
            m,
            activation_bits: 8,
        }
    }

    /// Select the activation quantization width: `4` switches to the BitNet v2
    /// Hadamard 4-bit path ([`crate::quantize_4bit`]); anything `>= 8` keeps
    /// the default b1.58 8-bit absmax path.
    pub fn with_activation_bits(mut self, bits: usize) -> Self {
        self.activation_bits = bits;
        self
    }

    /// Forward on `[b, t, in]`, returns `(y: [b, t, out], w_eff: [out, in])`.
    /// `w_eff` is the masked ternary weight, returned for diagnostics.
    ///
    /// The activation matmul runs in [`crate::precision::kernel_precision()`]
    /// (`KERNEL_PRECISION` env; default fp32 = bit-identical to the original
    /// path). The selector only affects the matmul: the activation is already
    /// quantized (int8-like for 8-bit, dequantized fp32 with int4-like values
    /// for the 4-bit Hadamard path), so its precision choice is moot. In bf16
    /// mode both operands are cast to bf16 (burn's matmul requires matching
    /// dtypes): `w_eff` is {−1,0,+1}·scale, so bf16 rounding only rounds the
    /// global scale and the ternary structure survives exactly. The matmul
    /// sits AFTER the Dual-STE autodiff op, so its backward is burn's normal
    /// bf16 matmul backward and the identity gradient still reaches the master
    /// weight.
    pub fn forward(&self, x: Tensor<3>) -> (Tensor<3>, Tensor<2>) {
        let w_eff = weight_quant_masked(self.weight.val(), self.n, self.m);
        let [b, t, d] = x.dims();
        let [out, _] = w_eff.dims();
        // Activation quant: 4 bits uses BitNet v2's Hadamard scheme
        // (crate::quantize_4bit, STE + inverse-Hadamard already inside), >= 8
        // uses the b1.58 absmax path. 4-bit outputs a dequantized fp32 tensor,
        // so the precision selector's bf16 cast below applies unchanged.
        let x_q = match self.activation_bits {
            4 => crate::quantize_4bit(x).reshape([b * t, d]),
            bits if bits >= 8 => crate::activation_quant_8bit(x).reshape([b * t, d]),
            other => panic!("SparseBitLinear: unsupported activation_bits {other} (use 4 or >= 8)"),
        };
        let y = match crate::precision::kernel_precision() {
            crate::precision::Precision::Fp32 => x_q.matmul(w_eff.clone().transpose()),
            crate::precision::Precision::Bf16 => x_q
                .cast(burn::tensor::FloatDType::BF16)
                .matmul(
                    w_eff
                        .clone()
                        .transpose()
                        .cast(burn::tensor::FloatDType::BF16),
                )
                .cast(burn::tensor::FloatDType::F32),
            crate::precision::Precision::Fp8 => {
                crate::precision::warn_fp8_unavailable();
                x_q.matmul(w_eff.clone().transpose())
            }
        };
        (y.reshape([b, t, out]), w_eff)
    }
}

#[cfg(feature = "autodiff")]
pub mod ad {
    //! Fused autodiff node for masked ternary quant (Dual-STE), mirroring the
    //! crate's `fwt_cuda` pattern: `try_into_primitive` fast path + tensor
    //! fallback, `Ops/Backward/NoCheckpointing`, `OpsKind::Tracked`/`UnTracked`.

    use super::*;
    use burn::backend::{Backend, DispatchKindConversion};
    use burn::tensor::DispatchTensor;
    use burn_autodiff::checkpoint::base::Checkpointer;
    use burn_autodiff::checkpoint::strategy::NoCheckpointing;
    use burn_autodiff::grads::Gradients;
    use burn_autodiff::ops::{Backward, Ops, OpsKind};
    use burn_autodiff::Autodiff;

    #[derive(Debug)]
    struct WeightQuantMaskedOp;

    impl<B: Backend> Backward<B, 1> for WeightQuantMaskedOp
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        type State = ();

        fn backward(
            self,
            ops: Ops<Self::State, 1>,
            grads: &mut Gradients,
            _checkpointer: &mut Checkpointer,
        ) {
            // Dual-STE: straight through both the quantizer and the mask. Do
            // NOT multiply by the mask and do NOT zero masked entries — pruned
            // weights must keep their gradient to regrow into the top-N.
            let d_out = grads.consume::<B>(&ops.node);
            grads.register::<B>(ops.parents[0].clone().unwrap().id, d_out);
        }
    }

    /// Fused masked ternary quant (identity backward) on `Autodiff<Inner>`.
    pub fn weight_quant_masked_autodiff<Inner: Backend>(
        w: Tensor<2>,
        n: usize,
        m: usize,
    ) -> Option<Tensor<2>>
    where
        DispatchTensor: DispatchKindConversion<Autodiff<Inner>> + DispatchKindConversion<Inner>,
    {
        let wa = w.try_into_primitive::<Autodiff<Inner>>().ok()?;
        let w_t = Tensor::from_primitive::<Inner>(wa.primitive().clone());
        let out_t = weight_quant_masked_tensor(w_t, n, m);
        let out_prim = out_t.try_into_primitive::<Inner>().unwrap();
        let nodes = [wa.node()];
        let prep = WeightQuantMaskedOp.prepare::<NoCheckpointing>(nodes);
        let out_adt = match prep.compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish((), out_prim),
            OpsKind::UnTracked(prep) => prep.finish(out_prim),
        };
        Some(Tensor::from_primitive::<Autodiff<Inner>>(out_adt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::ndarray()
    }

    fn to_host<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn nm_mask_6_8_keeps_largest_per_block() {
        // one row, values chosen so the top-6 of the block are unambiguous:
        // 7 1 2 6 3 5 4 0.5 -> keep 7 6 5 4 3 2 (indices 0 3 5 6 4 2)
        let w = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(vec![7.0, 1.0, 2.0, 6.0, 3.0, 5.0, 4.0, 0.5], [1, 8]),
            &dev(),
        );
        let mask = to_host(compute_nm_mask(w, 6, 8));
        assert_eq!(mask, vec![1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn nm_mask_2_4_per_row() {
        // two rows, each with a distinct top-2
        let w = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(vec![10.0, 1.0, 9.0, 2.0, 0.5, 4.0, 3.0, 8.0], [2, 4]),
            &dev(),
        );
        let mask = to_host(compute_nm_mask(w, 2, 4));
        assert_eq!(mask, vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn nm_mask_non_divisible_pads_and_slices() {
        // dim 10 is not a multiple of m=4: padded to 12, mask over the real 10
        // values must match the unpadded computation, and the output is [1, 10]
        let values = vec![5.0, 1.0, 4.0, 2.0, 3.0, 9.0, 0.5, 8.0, 7.0, 6.0];
        let w = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(values.clone(), [1, 10]),
            &dev(),
        );
        let mask = compute_nm_mask(w, 2, 4);
        assert_eq!(mask.dims(), [1, 10]);
        let m = to_host(mask);
        // blocks: [5 1 4 2] top2 = 5,4 -> [1,0,1,0]; [3 9 .5 8] top2 = 9,8 ->
        // [0,1,0,1]; [7 6 | pad pad] top2 = 7,6 -> [1,1,0,0]
        assert_eq!(m, vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn weight_quant_masked_is_ternary_and_sparse() {
        let w = Tensor::<2>::random([4, 16], Distribution::Normal(0.0, 0.5), &dev());
        // gamma = mean(|w|) of the master: the RoundClip scale
        let s = to_host(w.clone().abs()).iter().sum::<f32>() / 64.0;
        // reference mask read off the same master (what hardware would store)
        let mask = to_host(compute_nm_mask(w.clone(), 6, 8));
        let q = weight_quant_masked(w, 6, 8);
        let v = to_host(q);
        // (a) every value is ternary at scale s: {-s, 0, +s}; zeros come from
        // BOTH quantization (|w| < gamma/2 under Eq.1) and the mask.
        for val in &v {
            assert!(
                val.abs() < 0.01 || (val.abs() - s).abs() < 0.01,
                "{val} not ternary at s={s}"
            );
        }
        // (b) nonzeros only at mask-on positions; at most n per m-block
        // (exactly n selected, some may quantize to 0 under Eq.1).
        for (bi, blk) in v.chunks_exact(8).enumerate() {
            let mk = &mask[bi * 8..(bi + 1) * 8];
            let nz = blk.iter().filter(|x| x.abs() > 0.01).count();
            assert!(nz <= 6, "block {bi}: {nz} nonzeros > N");
            for (j, (&val, &m)) in blk.iter().zip(mk).enumerate() {
                assert!(
                    val.abs() < 0.01 || m > 0.5,
                    "block {bi} pos {j}: nonzero outside the N:M mask"
                );
            }
        }
    }

    #[test]
    fn nm_knob_off_matches_plain_ternary() {
        let w = Tensor::<2>::random([4, 16], Distribution::Normal(0.0, 0.5), &dev());
        let a = to_host(crate::weight_quant_ternary_nm(w.clone(), 0, 0));
        let b = to_host(crate::weight_quant_ternary(w.clone()));
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6);
        }
        // knob on produces sparsity
        let sparse = to_host(crate::weight_quant_ternary_nm(w, 6, 8));
        assert!(sparse.iter().filter(|x| x.abs() < 0.01).count() > 0);
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn dual_ste_gradient_flows_to_masked_weights() {
        let adev = dev().autodiff();
        let w = Tensor::<2>::random([2, 8], Distribution::Normal(0.0, 0.5), &adev);
        let wf = w.clone().require_grad();
        let loss = weight_quant_masked(wf.clone(), 6, 8).sum();
        let grads = loss.backward();
        let g = to_host(wf.grad(&grads).unwrap());
        // Dual-STE backward is identity: d loss/d W = d loss/d W_eff = 1 for
        // EVERY entry — including masked (pruned) ones. A gated-gradient
        // implementation would zero them here (and this test fails).
        assert!(g.iter().all(|x| (x - 1.0).abs() < 1e-3), "grads {g:?}");
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn sparse_bit_linear_forward_shapes_and_finite() {
        let layer = SparseBitLinear::new(8, 4, 6, 8, &dev());
        let x = Tensor::<3>::random([2, 5, 8], Distribution::Normal(0.0, 1.0), &dev());
        let (y, w_eff) = layer.forward(x);
        assert_eq!(y.dims(), [2, 5, 4]);
        assert_eq!(w_eff.dims(), [4, 8]);
        for v in to_host(y) {
            assert!(v.is_finite());
        }
        // w_eff is exactly N:M sparse
        for block in to_host(w_eff).chunks_exact(8) {
            assert_eq!(block.iter().filter(|x| x.abs() > 0.01).count(), 6);
        }
    }

    #[test]
    fn activation_bits_4_uses_hadamard_and_stays_close_to_8bit() {
        // d=64/512: d=8 is degenerate (output scale ~0.04, quantization-noise-
        // dominated, rel diff swings 0.25-0.31 between seeds)
        for d in [64usize, 512] {
            let layer8 = SparseBitLinear::new(d, d / 2, 6, 8, &dev());
            let layer4 = layer8.clone().with_activation_bits(4);
            let x = Tensor::<3>::random([2, 8, d], Distribution::Normal(0.0, 1.0), &dev());
            let [b, t, dd] = x.dims();
            let [out, _] = layer4.weight.val().dims();
            let w_eff = weight_quant_masked(layer4.weight.val(), 6, 8);
            // The 4-bit path must be exactly BitNet v2's scheme: it equals a manual
            // composition with crate::quantize_4bit (whose Hadamard semantics are
            // guarded by the lib.rs v2_4bit_roundtrip / FWT tests).
            let y_manual = crate::quantize_4bit(x.clone())
                .reshape([b * t, dd])
                .matmul(w_eff.transpose())
                .reshape([b, t, out]);
            let (y4, w4) = layer4.forward(x.clone());
            assert_eq!(y4.dims(), [2, 8, out]);
            assert_eq!(w4.dims(), [out, d]);
            assert!(to_host(y4.clone()).iter().all(|v| v.is_finite()));
            let md = maxdiff(&to_host(y4.clone()), &to_host(y_manual));
            assert!(
                md < 1e-4,
                "4-bit path must use quantize_4bit (Hadamard), maxdiff {md}"
            );
            // quantize_4bit really rotates: it differs from a plain (non-Hadamard)
            // per-token absmean 4-bit quant of the same input
            let flat = x.clone().reshape([b * t, dd]);
            let scale = flat.clone().abs().mean_dim(1).clamp_min(1e-12);
            let no_had = flat
                .div(scale.clone())
                .round()
                .clamp(-8.0, 7.0)
                .mul(scale)
                .reshape([b, t, dd]);
            let had_diff = maxdiff(&to_host(crate::quantize_4bit(x.clone())), &to_host(no_had));
            assert!(
                had_diff > 1e-3,
                "quantize_4bit must apply the Hadamard rotation (maxdiff {had_diff})"
            );
            // 4-bit is coarser than 8-bit, but on the same weight the layer output
            // stays within ~0.3 relative (rmse). This is the dequantized-activation
            // relative error too: the matmul is linear, so output rmse_rel ==
            // activation rmse_rel. ~0.1 is impossible — the 4-bit Hadamard scheme's
            // own quant error is ~23% (v2_4bit_roundtrip guards mse < 0.1); 0.3 only
            // guards against a broken/mis-scaled path.
            let (y8, _) = layer8.forward(x.clone());
            let y4 = to_host(y4);
            let y8 = to_host(y8);
            let rmse = (y4
                .iter()
                .zip(&y8)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                / y8.len() as f32)
                .sqrt();
            let rms = (y8.iter().map(|v| v * v).sum::<f32>() / y8.len() as f32).sqrt();
            assert!(
                rmse / rms < 0.3,
                "4-bit vs 8-bit forward rmse rel {rmse}/{rms}"
            );
        }
    }
}

#[cfg(all(test, feature = "autodiff", feature = "cuda"))]
mod ad_tests {
    use super::*;
    use burn::tensor::Device;

    fn to_host<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// The fused op (CUDA autodiff) must produce the same forward as the pure
    /// tensor path and the Dual-STE identity gradient.
    #[test]
    fn fused_masked_quant_forward_and_dual_ste() {
        let dev = Device::ndarray();
        let adev = dev.clone().autodiff();
        let w = Tensor::<2>::random([4, 16], Distribution::Normal(0.0, 0.5), &dev);
        // dispatch on `Autodiff<CudaBare>` takes the fused op path
        let wf = Tensor::<2>::from_data(w.clone().into_data(), &adev).require_grad();
        let q = weight_quant_masked(wf.clone(), 6, 8);
        let q_ref =
            weight_quant_masked_tensor(Tensor::<2>::from_data(w.clone().into_data(), &dev), 6, 8);
        let md = maxdiff(&to_host(q.clone()), &to_host(q_ref));
        assert!(md < 1e-5, "fused vs tensor forward maxdiff {md}");
        // Dual-STE: identity backward -> d loss/d W = 1 for every entry,
        // masked (pruned) ones included
        let grads = q.sum().backward();
        let g = to_host(wf.grad(&grads).unwrap());
        assert!(g.iter().all(|x| (x - 1.0).abs() < 1e-3), "grads {g:?}");
    }

    /// SparseBitLinear::forward in bf16 mode vs fp32 on the same input:
    /// rel maxdiff < 1e-2 (bf16 ~3 decimal digits) and a finite output.
    /// bf16 matmul needs a CUDA backend (NdArray has no bf16 support).
    #[test]
    fn bf16_forward_matches_fp32() {
        let dev = Device::cuda(0);
        let layer = SparseBitLinear::new(512, 512, 6, 8, &dev);
        let x = Tensor::<3>::random([2, 8, 512], Distribution::Normal(0.0, 1.0), &dev);
        let (y32, _) = layer.forward(x.clone());
        std::env::set_var("KERNEL_PRECISION", "bf16");
        let (y16, _) = layer.forward(x);
        std::env::remove_var("KERNEL_PRECISION");
        let y32 = to_host(y32);
        let y16 = to_host(y16);
        assert!(y16.iter().all(|v| v.is_finite()), "bf16 output not finite");
        let max_abs: f32 = y32
            .iter()
            .zip(&y16)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let scale: f32 = y32.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!(
            max_abs / scale < 1e-2,
            "fp32 vs bf16 forward rel maxdiff {max_abs}/{scale}"
        );
    }

    /// Gradients through the bf16 matmul path match the fp32 path: the
    /// Dual-STE identity backward sits AFTER the matmul, so bf16 rounding is
    /// the only difference between the two runs (weight_quant_masked is
    /// deterministic on the same master).
    #[test]
    fn bf16_gradients_match_fp32() {
        let dev = Device::cuda(0).autodiff();
        let layer = SparseBitLinear::new(64, 64, 6, 8, &dev);
        let x = Tensor::<3>::random([2, 8, 64], Distribution::Normal(0.0, 1.0), &dev);
        let (y32, _) = layer.forward(x.clone());
        let g32 = to_host(layer.weight.grad(&y32.sum().backward()).unwrap());
        std::env::set_var("KERNEL_PRECISION", "bf16");
        let (y16, _) = layer.forward(x);
        let g16 = to_host(layer.weight.grad(&y16.sum().backward()).unwrap());
        std::env::remove_var("KERNEL_PRECISION");
        assert!(g16.iter().all(|v| v.is_finite()), "bf16 grads not finite");
        let md = maxdiff(&g32, &g16);
        let scale = g32.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!(
            md / scale < 1e-2,
            "fp32 vs bf16 gradient rel maxdiff {md}/{scale}"
        );
    }

    /// SparseBitLinear 4-bit (BitNet v2 Hadamard) forward on CUDA: the fused
    /// FWT path runs, output is finite, and KERNEL_PRECISION=bf16 still works
    /// on the dequantized fp32 activations (rel maxdiff < 1e-2 like the 8-bit
    /// path).
    #[test]
    fn four_bit_forward_cuda_finite_and_bf16() {
        let dev = Device::cuda(0);
        let layer = SparseBitLinear::new(512, 512, 6, 8, &dev).with_activation_bits(4);
        let x = Tensor::<3>::random([2, 8, 512], Distribution::Normal(0.0, 1.0), &dev);
        let (y32, _) = layer.forward(x.clone());
        assert!(to_host(y32.clone()).iter().all(|v| v.is_finite()));
        std::env::set_var("KERNEL_PRECISION", "bf16");
        let (y16, _) = layer.forward(x);
        std::env::remove_var("KERNEL_PRECISION");
        let y16 = to_host(y16);
        assert!(
            y16.iter().all(|v| v.is_finite()),
            "bf16 4-bit output not finite"
        );
        let y32 = to_host(y32);
        let md = maxdiff(&y32, &y16);
        let scale = y32.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!(
            md / scale < 1e-2,
            "fp32 vs bf16 4-bit forward rel maxdiff {md}/{scale}"
        );
    }
}
