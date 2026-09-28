//! Weight quantizers of the BitNet family (ternary b1.58/v2, legacy sign
//! variant, 2-bit, and the N:M-sparse dispatch into [`crate::sparse`]).

use burn::tensor::Tensor;

/// BitNet ternary weight quantizer — verbatim arXiv 2504.18415 (BitNet v2,
/// Eq. 1), same scheme as b1.58 (2402.17764) and the 2B4T tech report
/// (2504.12285): `W̃ = α·RoundClip(W/(α+ε), −1, +1)`, `α = mean(|W|)`
/// per-tensor absmean; |w| < α/2 maps to 0 naturally and magnitudes above
/// that survive instead of collapsing to ±α. Straight-through gradient via
/// the residual trick (forward value = W̃, backward identity).
pub fn weight_quant_ternary(w: Tensor<2>) -> Tensor<2> {
    weight_quant_b158(w)
}

/// The BitNet b1.58 / v2 weight quantizer verbatim:
/// `Q_w(W) = α·RoundClip(W/(α+ε), −1, +1)`, `α = mean(|W|)` (2504.18415
/// Eq. 1; identical scheme described in 2407.09527 §3).
pub fn weight_quant_b158(w: Tensor<2>) -> Tensor<2> {
    let wd = w.clone().detach();
    // γ = mean(|W|): mean() yields rank-1, unsqueeze to [1,1] for broadcast
    let scale = wd
        .clone()
        .abs()
        .mean()
        .unsqueeze_dims(&[0, 0])
        .clamp_min(1e-5); // γ + ε guard from the papers
    let q = wd
        .clone()
        .div(scale.clone())
        .round()
        .clamp(-1.0, 1.0)
        .mul(scale);
    w.add(q.sub(wd))
}

/// LEGACY (llama.cpp-style): mean-centered sign variant
/// `sign(w − mean(w)) · mean(|w|)`. NOT a BitNet-paper form — kept only for
/// historical A/B baselines; do not use for new paper-matched work.
pub fn weight_quant_sign_centered(w: Tensor<2>) -> Tensor<2> {
    let scale = w.clone().abs().mean().unsqueeze_dims(&[0, 0]);
    let mean = w.clone().mean().unsqueeze_dims(&[0, 0]);
    let u = w.clone().sub(mean).sign().mul(scale);
    let base = w.clone().detach();
    w.add(u.sub(base))
}

/// BitNet-style 2-bit weight quantizer with straight-through gradient.
///
/// Five levels: `q = round(clamp(w / s, -2, 2)) * s` with `s` the per-row
/// (output-channel) absmean over the input dim, i.e. values in
/// `{-2s, -s, 0, s, 2s}` — the "2.3-bit" scheme used by several 2-bit works.
/// Five levels instead of four ({-2,-1,1,2}): zero stays representable (the
/// direct 2-bit superset of [`weight_quant_ternary`], keeping A/Bs against
/// the ternary apples-to-apples) and the ±2 levels absorb outliers at no
/// forward cost. Storage packing to 2-3 real bits is a deployment concern,
/// not a training one — masters stay fp32.
///
/// STE: forward value is the dequantized `q`, backward is exactly identity.
/// The per-row scale is detached so the gradient is exactly 1 (the ternary's
/// non-detached mean/scale would leak ~1/N cross-terms, see
/// [`crate::sparse::weight_quant_masked_tensor`]).
pub fn weight_quant_2bit(w: Tensor<2>) -> Tensor<2> {
    let wd = w.clone().detach();
    let scale = w.clone().abs().mean_dim(1).clamp_min(1e-12).detach();
    let q = wd
        .clone()
        .div(scale.clone())
        .round()
        .clamp(-2.0, 2.0)
        .mul(scale);
    w.add(q.sub(wd))
}

/// Ternary quant with optional N:M sparsity (Sparse-BitNet).
///
/// `n == 0 || m == 0` disables sparsity and is identical to
/// [`weight_quant_ternary`] (paper-exact RoundClip form).
pub fn weight_quant_ternary_nm(w: Tensor<2>, n: usize, m: usize) -> Tensor<2> {
    if n == 0 || m == 0 {
        weight_quant_ternary(w)
    } else {
        crate::sparse::weight_quant_masked(w, n, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Tensor};

    fn dev() -> burn::tensor::Device {
        burn::tensor::Device::ndarray()
    }

    fn to_host<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn b158_roundclip_is_nearest_of_three_levels() {
        // The oracle is a scalar loop over host f64, NOT a restatement of the
        // tensor chain at :19-33 (which is what this test used to be: a
        // hand-copy of the implementation moves with the bug). The property
        // is the DEFINING one of Eq. 1 — every weight lands on the level in
        // {-1,0,+1} nearest to W/gamma, scaled by gamma — stated as
        // argmin over the level set, which is a different computation from
        // round().clamp() and disagrees with it if gamma is taken over the
        // wrong axis or the clamp is applied before the round.
        // Input crafted so every branch is exercised:
        // |w| < gamma/2 -> 0, mid -> +-gamma, large -> clip keeps +-gamma.
        let xs = [
            0.1f32, -0.2, 0.6, -0.9, //
            1.4, 2.0, -3.0, 0.05,
        ];
        let w = Tensor::<2>::from_data(burn::tensor::TensorData::new(xs.to_vec(), [2, 4]), &dev());
        let q = weight_quant_b158(w.clone());
        let v: Vec<f32> = to_host(q);
        let gamma = xs.iter().map(|x| f64::from(*x).abs()).sum::<f64>() / xs.len() as f64;
        for (got, x) in v.iter().zip(xs.iter()) {
            let r = f64::from(*x) / gamma;
            // nearest level in {-1,0,+1}: distance, not rounding
            let level = [(-1.0f64, r + 1.0), (0.0, r.abs()), (1.0, (r - 1.0).abs())]
                .iter()
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("three levels")
                .0;
            let want = (level * gamma) as f32;
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
        // weight_quant_ternary IS the paper form now (delegates to b158):
        let t3: Vec<f32> = to_host(weight_quant_ternary(w.clone()));
        assert_eq!(t3, v, "ternary must be the b1.58/v2 RoundClip form");
        // while the legacy sign-centered variant must DIFFER here (guards
        // against silently aliasing the two schemes):
        let s: Vec<f32> = to_host(weight_quant_sign_centered(w));
        assert!(
            s.iter().zip(v.iter()).any(|(a, b)| (a - b).abs() > 1e-3),
            "roundclip and sign variants should diverge here"
        );
    }

    #[test]
    fn b158_exactly_three_levels_bounded_by_gamma() {
        // Properties of ANY correct b1.58 quantizer, on random input: the code
        // book is exactly {-gamma, 0, +gamma}, gamma is the absmean of the
        // INPUT, and nothing exceeds it. A copied expected value cannot see
        // the level count or the bound.
        let w = Tensor::<2>::random([8, 64], Distribution::Normal(0.0, 1.0), &dev());
        let master = to_host(w.clone());
        let gamma = to_host(w.clone().abs().mean())[0];
        let v = to_host(weight_quant_b158(w));
        for q in &v {
            let on_level = (*q - gamma).abs() < 1e-6 || (*q + gamma).abs() < 1e-6 || q.abs() < 1e-6;
            assert!(on_level, "{q} is not one of {{-gamma, 0, +gamma}}");
            assert!(q.abs() <= gamma + 1e-6, "{q} exceeds gamma {gamma}");
        }
        assert!(
            v.iter().filter(|q| q.abs() < 1e-6).count() > 0,
            "a 512-element normal sample must contain zeroed weights"
        );
        // sign is preserved unless the weight is zeroed (|w| < gamma/2). The
        // band starts at 0.55, not 0.5: exactly at the threshold the f32
        // absmean and the implementation's gamma can disagree on which side of
        // it a weight sits, and that is a rounding artefact, not a bug.
        for (q, m) in v.iter().zip(master.iter()) {
            if m.abs() >= gamma * 0.55 {
                assert_eq!(q.signum(), m.signum(), "sign flipped at w={m}");
            }
        }
    }

    #[test]
    fn b158_ste_gradient_is_exactly_identity() {
        // The straight-through claim (doc :11) is about the BACKWARD, so no
        // forward-value oracle can check it. dQ/dW must be exactly 1 per
        // element: every path from w to q runs through `wd = w.detach()`.
        let dev = burn::tensor::Device::ndarray().autodiff();
        let xs = [0.1f32, -0.2, 0.6, -0.9, 1.4, 2.0, -3.0, 0.05];
        let w = Tensor::<2>::from_data(burn::tensor::TensorData::new(xs.to_vec(), [2, 4]), &dev)
            .require_grad();
        // any non-constant scalar function of q; sum keeps the expected grad
        // at exactly 1 so a wrong derivative is unmissable.
        let grads = weight_quant_b158(w.clone()).sum().backward();
        let d = to_host(w.grad(&grads).expect("weight grad"));
        for (i, g) in d.iter().enumerate() {
            assert!((g - 1.0).abs() < 1e-6, "dQ/dW[{i}] = {g}, STE says 1");
        }
    }

    #[test]
    fn weight_ternary_values() {
        let w = Tensor::<2>::random([4, 16], Distribution::Normal(0.0, 0.5), &dev());
        // RoundClip form: every value is exactly one of {-gamma, 0, +gamma},
        // gamma = mean(|W|) of the master input (paper Eq. 1).
        let gamma = to_host(w.clone().abs().mean())[0];
        let v: Vec<f32> = to_host(weight_quant_ternary(w));
        for val in v {
            let k = (val / gamma).round();
            assert!(
                ((val - k * gamma).abs() < 1e-4) && k.abs() <= 1.0,
                "{val} not ternary at gamma={gamma}"
            );
        }
    }

    #[test]
    fn weight_2bit_values() {
        // exact check on inputs whose scales divide cleanly in fp32:
        // row 0 absmean 6/4 = 1.5, row 1 absmean 12/4 = 3.0
        let w = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(vec![2.0, 2.0, 0.0, -2.0, 6.0, 2.0, -2.0, -2.0], [2, 4]),
            &dev(),
        );
        let q = weight_quant_2bit(w);
        let v: Vec<f32> = to_host(q);
        assert_eq!(v, vec![1.5, 1.5, 0.0, -1.5, 6.0, 3.0, -3.0, -3.0]);
        // randomized: every value lands in {-2s, -s, 0, s, 2s} of its row's
        // absmean scale (per-row, unlike the global-scale ternary)
        let w = Tensor::<2>::random([4, 16], Distribution::Normal(0.0, 0.5), &dev());
        let scales = to_host(w.clone().abs().mean_dim(1));
        let v = to_host(weight_quant_2bit(w));
        for (i, val) in v.iter().enumerate() {
            let s = scales[i / 16];
            let ok = (0..=2).any(|k| (val.abs() - k as f32 * s).abs() < 0.01);
            assert!(ok, "row {i}: {val} not in {{-2s,-s,0,s,2s}} with s={s}");
        }
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn weight_2bit_ste_gradient_is_identity() {
        // STE: forward = quantized value, backward = identity. Loss = sum(q)
        // so the master gradient must be exactly 1 for every entry.
        let adev = dev().autodiff();
        let wf = Tensor::<2>::random([4, 16], Distribution::Normal(0.0, 0.5), &adev).require_grad();
        let grads = weight_quant_2bit(wf.clone()).sum().backward();
        let g = to_host(wf.grad(&grads).unwrap());
        assert!(
            g.iter().all(|x| (x - 1.0).abs() < 1e-3),
            "STE gradient must be identity, got {g:?}"
        );
    }
}
