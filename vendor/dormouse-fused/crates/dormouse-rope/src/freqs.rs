//! RoPE frequency tables: plain (Su et al. 2021) and YaRN NTK-by-parts
//! (Peng et al. 2023) with the attention-temperature correction.

use burn::tensor::{Device, Tensor};

/// Precompute plain RoPE cos/sin frequency tables `[max_seq_len, head_dim/2]`.
pub fn precompute_freqs(
    head_dim: usize,
    max_seq_len: usize,
    base: f64,
    device: &Device,
) -> (Tensor<2>, Tensor<2>) {
    let half = head_dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (1.0_f64 / base.powf(2.0 * i as f64 / head_dim as f64)) as f32)
        .collect();
    let tv: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
    let t_t: Tensor<1> = Tensor::from_floats(tv.as_slice(), device);
    let f_t: Tensor<1> = Tensor::from_floats(freqs.as_slice(), device);
    let angles = t_t.reshape([max_seq_len, 1]).matmul(f_t.reshape([1, half]));
    (angles.clone().cos(), angles.sin())
}

/// YaRN (arXiv:2309.00071v3) frequency tables with NTK-by-parts interpolation.
///
/// Per-dimension ramp (Def. 2, Eqs 10-13):
/// ```text
/// r(d) = L / (2*pi * base^(2d/|D|))          - original-context radius of dim d
/// gamma(r) = clamp((r - beta_slow) / (beta_fast - beta_slow), 0, 1)
/// h(theta_d) = (1 - gamma) * theta_d / s + gamma * theta_d
/// ```
/// plus the attention temperature correction (Eqs 14-15):
/// `sqrt(1/t) = 0.1 * ln(s) + 1`, folded into the frequencies.
///
/// `scale`: `s = L'/L` (extended / original context ratio).
/// `orig_len`: original context length `L`.
/// `beta_fast`/`beta_slow`: ramp bounds (paper defaults 32 / 1 for Llama-4096).
///
/// With `scale = 1.0` this reduces exactly to plain RoPE.
#[allow(clippy::too_many_arguments)]
pub fn precompute_freqs_yarn(
    head_dim: usize,
    max_seq_len: usize,
    base: f64,
    scale: f64,
    orig_len: usize,
    beta_fast: f64,
    beta_slow: f64,
    device: &Device,
) -> (Tensor<2>, Tensor<2>) {
    let half = head_dim / 2;
    let temp = 0.1 * scale.ln() + 1.0;
    let freqs: Vec<f32> = (0..half)
        .map(|i| {
            let theta = base.powf(-2.0 * i as f64 / head_dim as f64);
            // Original-context radius of this dimension (YaRN Def. 2):
            // r = L/λ_d = L·θ_d/(2π). High-frequency dims (large θ) get a
            // large r and keep their frequency; low-frequency dims get
            // interpolated down to θ/s. (An earlier revision divided by θ,
            // which inverted the ramp and silently disabled interpolation.)
            let r = orig_len as f64 * theta / (2.0 * std::f64::consts::PI);
            let gamma = ((r - beta_slow) / (beta_fast - beta_slow)).clamp(0.0, 1.0);
            let h = (1.0 - gamma) * theta / scale + gamma * theta;
            (h * temp) as f32
        })
        .collect();
    let tv: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
    let t_t: Tensor<1> = Tensor::from_floats(tv.as_slice(), device);
    let f_t: Tensor<1> = Tensor::from_floats(freqs.as_slice(), device);
    let angles = t_t.reshape([max_seq_len, 1]).matmul(f_t.reshape([1, half]));
    (angles.clone().cos(), angles.sin())
}
