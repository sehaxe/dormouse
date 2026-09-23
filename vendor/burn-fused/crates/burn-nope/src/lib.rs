//! # burn-nope - No Positional Encoding for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! Pure content-based attention without positional embeddings
//! ([Kimi K3](https://arxiv.org/abs/2607.24653), Moonshot 2026).
//! Q·K^T attention without RoPE rotation - relies on causal masking
//! and content structure rather than explicit position encoding.
//!
//! Works best with recurrent architectures (KDA/GDN2) where temporal
//! order is captured through state updates rather than positional encoding.
use burn::tensor::{Bool, Tensor};

/// Content-based scaled dot-product attention without position encoding.
///
/// `q, k, v`: `[B, T, NH, HD]`
/// `causal`: if true, mask future positions.
///
/// Returns `[B, T, NH, HD]` - attention output without RoPE rotation.
pub fn nope_attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, causal: bool) -> Tensor<4> {
    let [_b, t, _nh, hd] = q.dims();
    let scale = (hd as f64).powf(-0.5);
    let device = q.device();

    let scores = q
        .clone()
        .swap_dims(1, 2)
        .matmul(k.swap_dims(1, 2).swap_dims(2, 3))
        .mul_scalar(scale);

    let scores = if causal {
        // Pure GPU mask - no host work. burn's triu_mask(off=1) yields true
        // for j <= i; invert to mask strictly-future positions (j > i).
        let mask = Tensor::<4, Bool>::triu_mask([1, 1, t, t], 1, &device).bool_not();
        scores.mask_fill(mask, -1e9)
    } else {
        scores
    };

    let attn = burn::tensor::activation::softmax(scores, 3);
    attn.matmul(v.swap_dims(1, 2)).swap_dims(1, 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;
    use burn::tensor::Distribution;

    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn nope_shape_causal() {
        let q = Tensor::<4>::random([1, 8, 2, 32], Distribution::Default, &dev());
        let k = Tensor::<4>::random([1, 8, 2, 32], Distribution::Default, &dev());
        let v = Tensor::<4>::random([1, 8, 2, 32], Distribution::Default, &dev());
        assert_eq!(nope_attention(q, k, v, true).dims(), [1, 8, 2, 32]);
    }

    #[test]
    fn nope_shape_non_causal() {
        let q = Tensor::<4>::random([2, 4, 4, 16], Distribution::Default, &dev());
        let k = Tensor::<4>::random([2, 4, 4, 16], Distribution::Default, &dev());
        let v = Tensor::<4>::random([2, 4, 4, 16], Distribution::Default, &dev());
        assert_eq!(nope_attention(q, k, v, false).dims(), [2, 4, 4, 16]);
    }

    #[test]
    fn causal_row_t_only_sees_prefix() {
        // Row t of the causal output must equal bidirectional attention
        // restricted to the prefix [0..=t].
        let (b, t, nh, hd) = (1usize, 3usize, 1usize, 8usize);
        let q = Tensor::<4>::random([b, t, nh, hd], Distribution::Default, &dev());
        let k = Tensor::<4>::random([b, t, nh, hd], Distribution::Default, &dev());
        let v = Tensor::<4>::random([b, t, nh, hd], Distribution::Default, &dev());
        let causal_out = nope_attention(q.clone(), k.clone(), v.clone(), true);
        let causal_vals: Vec<f32> = causal_out
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        // Compare row 1: causal = attend over tokens 0..=1.
        let q_pref = q.clone().slice([0..b, 0..2, 0..nh, 0..hd]);
        let k_pref = k.clone().slice([0..b, 0..2, 0..nh, 0..hd]);
        let v_pref = v.slice([0..b, 0..2, 0..nh, 0..hd]);
        let ref_out = nope_attention(q_pref, k_pref, v_pref, false);
        let ref_vals: Vec<f32> = ref_out
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        for i in 0..hd {
            let a = causal_vals[nh * hd + i];
            let b = ref_vals[nh * hd + i];
            assert!((a - b).abs() < 1e-4, "causal row 1 mismatch: {a} vs {b}");
        }
    }
}
