//! # burn-engram - Conditional Memory for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! N-gram hash embedding with gated memory fusion, based on
//! [Engram](https://arxiv.org/abs/2601.07372) (DeepSeek AI, 2026).
//!
//! Key results (27B scale): MMLU +3.4, BBH +5.0, ARC +3.7, HumanEval +3.0, MATH +2.4,
//! long-context NIAH 84.2 → 97.0.
//!
//! ## Architecture
//!
//! ```text
//! hashed_ids → MultiHashEmbedding → concat → value_proj
//!                                   │
//!                                   └→ key_projs[N] ─┐
//! hidden_states [B,L,HC,D] → RMSNorm(query) ─────────┤
//!                                                      ├→ gate[N] → out [B,L,HC,D]
//!                                                      │
//!                                        value ←───────┘
//! out = out + depthwise_conv_1d(out)     (residual short conv)
//! ```
pub mod hasher;

use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig};
use burn::tensor::{activation, Device, Int, Tensor};

/// Multiple embedding tables with offset addressing.
#[derive(Module, Debug)]
pub struct MultiHashEmbedding {
    embedding: Embedding,
    #[module(skip)]
    offsets: Vec<i32>,
    #[module(skip)]
    embed_dim: usize,
    #[module(skip)]
    num_tables: usize,
}

impl MultiHashEmbedding {
    pub fn new(table_sizes: &[usize], embed_dim: usize, device: &Device) -> Self {
        let total: usize = table_sizes.iter().sum();
        let mut offsets = Vec::with_capacity(table_sizes.len());
        let mut acc = 0usize;
        for &s in table_sizes {
            offsets.push(acc as i32);
            acc += s;
        }
        Self {
            embedding: EmbeddingConfig::new(total, embed_dim).init(device),
            offsets,
            embed_dim,
            num_tables: table_sizes.len(),
        }
    }

    pub fn forward(&self, hashed_ids: Tensor<3, Int>) -> Tensor<3> {
        let [b, l, t] = hashed_ids.dims();
        let total = self.num_tables * self.embed_dim;
        // Single fused lookup: pre-offset all table indices and gather once
        // over the concatenated table (was t slices + t embedding gathers + a cat).
        let offsets = self.offsets.iter().map(|&o| o as i64).collect::<Vec<_>>();
        let off = Tensor::<1, Int>::from_ints(offsets.as_slice(), &hashed_ids.device());
        let flat = hashed_ids.add(off.reshape([1, 1, t])).reshape([b * l * t]);
        let w = self.embedding.weight.val();
        let [_, d] = w.dims();
        let idx = flat.unsqueeze_dim::<2>(1).expand([b * l * t, d]);
        w.gather(0, idx).reshape([b, l, total])
    }
}

/// Depthwise causal 1D convolution (paper's ShortConv).
///
/// x: `[B, L, HC, D]`, weight: `[D, kernel_size]` - depthwise per-channel.
/// Left-padded with zeros for causal masking.
///
/// NOTE: the per-tap pad+slice pattern below is the roadmap's #13b fused-kernel
/// candidate. It stays tensor-path deliberately: this crate has no cubecl
/// dependency or kernel infrastructure, and `conv_weight` is a trainable
/// `Param`, so a fused kernel would need fwd + bwd kernels plus an autodiff
/// `Backward` impl (the burn-msa template is ~300 lines) plus CUDA/ndarray
/// runtime gating - a redesign, not a clean addition. Tensor-path `slice`
/// consumes self, so each tap clone is required; the cost is one copy of the
/// padded input per tap. Revisit with a shared fused-conv kernel crate.
pub fn depthwise_conv_1d(x: Tensor<4>, weight: &Tensor<2>, dilation: usize) -> Tensor<4> {
    let [b, l, hc, d] = x.dims();
    let k = weight.dims()[1];
    let pad_left = (k - 1) * dilation;
    if pad_left == 0 {
        return x * weight.clone().slice([0..d, 0..1]).reshape([1, 1, 1, d]);
    }
    let zeros = Tensor::zeros([b, pad_left, hc, d], &x.device());
    let x_pad = Tensor::cat(vec![zeros, x.clone()], 1);
    let mut out = Tensor::zeros([b, l, hc, d], &x.device());
    for i in 0..k {
        let offset = (k - 1 - i) * dilation;
        let x_slice = x_pad.clone().slice([0..b, offset..offset + l, 0..hc, 0..d]);
        let w = weight.clone().slice([0..d, i..i + 1]).reshape([1, 1, 1, d]);
        out = out + x_slice * w;
    }
    out
}

/// Full Engram module: hash lookup → key/value proj → multi-head gated fusion → short conv.
///
/// `hc_mult` parallels the backbone's hyper-connection multiplier.
/// Set `hc_mult=1` for a standard single-head module.
#[derive(Module, Debug)]
pub struct EngramModule {
    memory: MultiHashEmbedding,
    key_projs: Vec<Linear>,
    value_proj: Linear,
    conv_weight: Option<Param<Tensor<2>>>,
    #[module(skip)]
    hidden_size: usize,
    #[module(skip)]
    hc_mult: usize,
    #[module(skip)]
    conv_kernel: usize,
    #[module(skip)]
    conv_dilation: usize,
}

impl EngramModule {
    pub fn new(
        table_sizes: &[usize],
        embed_dim: usize,
        hidden_size: usize,
        hc_mult: usize,
        device: &Device,
    ) -> Self {
        let total_embed = table_sizes.len() * embed_dim;
        let key_projs = (0..hc_mult)
            .map(|_| {
                LinearConfig::new(total_embed, hidden_size)
                    .with_bias(false)
                    .init(device)
            })
            .collect();
        Self {
            memory: MultiHashEmbedding::new(table_sizes, embed_dim, device),
            key_projs,
            value_proj: LinearConfig::new(total_embed, hidden_size)
                .with_bias(false)
                .init(device),
            conv_weight: None,
            hidden_size,
            hc_mult,
            conv_kernel: 0,
            conv_dilation: 1,
        }
    }

    pub fn with_short_conv(mut self, kernel_size: usize, dilation: usize, device: &Device) -> Self {
        self.conv_kernel = kernel_size;
        self.conv_dilation = dilation;
        // Paper §4.1: conv is ZERO-initialized so the short-conv path is the
        // identity (Y = Ṽ) at the start of training.
        self.conv_weight = Some(Initializer::Zeros.init([self.hidden_size, kernel_size], device));
        self
    }

    /// Multi-head gated memory fusion.
    ///
    /// `hashed_ids`: `[B, L, num_tables, Int]`
    /// `hidden_states`: `[B, L, HC, D]` (4D multi-head) or `[B, L, D]` (3D single-head)
    ///
    /// Returns memory-augmented representations, same shape as `hidden_states`.
    pub fn forward(&self, hashed_ids: Tensor<3, Int>, hidden_states: Tensor<4>) -> Tensor<4> {
        let embeds = self.memory.forward(hashed_ids);
        self.forward_embeds(embeds, hidden_states)
    }

    /// Same gated fusion as [`forward`](Self::forward), but the per-token
    /// embeddings arrive pre-assembled `[B, L, num_tables*embed_dim]` instead
    /// of being looked up from the in-model tables. This is the RAM-offload
    /// path (Qwen3.8-Flash-Next §2.3): the tables live in host memory, only
    /// the batch's rows reach the GPU.
    pub fn forward_embeds(&self, embeds: Tensor<3>, hidden_states: Tensor<4>) -> Tensor<4> {
        let [b, l, hc, d] = hidden_states.dims();
        let value = self.value_proj.forward(embeds.clone());

        let mut gates = Vec::with_capacity(hc);
        for head in 0..hc {
            let key = self.key_projs[head].forward(embeds.clone());
            let query = hidden_states
                .clone()
                .slice([0..b, 0..l, head..head + 1])
                .reshape([b, l, d]);
            let gate = compute_gate(key, query, self.hidden_size);
            gates.push(gate.unsqueeze_dim::<4>(2));
        }
        let gate = Tensor::cat(gates, 2); // [B, L, HC, 1]
        let mut out = value.unsqueeze_dim::<4>(2).mul(gate); // [B, L, HC, D]

        if let Some(ref w) = self.conv_weight {
            // Paper Eq 5: Y = SiLU(Conv1D(RMSNorm(Ṽ))) + Ṽ
            let conv_in = out.clone()
                / out
                    .clone()
                    .powf_scalar(2.0)
                    .mean_dim(3)
                    .add_scalar(1e-5)
                    .sqrt();
            let conv_out = depthwise_conv_1d(conv_in, &w.val(), self.conv_dilation);
            out = out + activation::silu(conv_out);
        }

        out
    }
}

pub fn compute_gate(key: Tensor<3>, query: Tensor<3>, d: usize) -> Tensor<3> {
    let key_rms = key
        .clone()
        .powf_scalar(2.0)
        .mean_dim(2)
        .add_scalar(1e-5)
        .sqrt();
    let query_rms = query
        .clone()
        .powf_scalar(2.0)
        .mean_dim(2)
        .add_scalar(1e-5)
        .sqrt();
    let key = key / key_rms;
    let query = query / query_rms;
    let scale = (d as f32).sqrt();
    let dot = (key * query).sum_dim(2).div_scalar(scale);
    // Official reference (deepseek-ai/Engram, engram_demo_v1.py:371-373, the
    // copy committed under this crate's tests/oracle/):
    //   gate = sigmoid(sqrt(|s| clamped to >= 1e-6) * sign(s)), s = dot/sqrt(d)
    // (the plain sigmoid(s) variant diverges for |s| > ~1)
    //
    // CLAMP, NOT ADD. `clamp_min(1e-6)`, not `+ 1e-6` - and the two differ
    // only just above 1e-6, which is reachable: |s| is bounded by sqrt(D), so
    // the band is not a fixture choice, it is the only place the formulas are
    // separable. `add` raises every value below 1e-6 (a negative shift, so it
    // even moves the sign-adjacent values the wrong way) and leaves everything
    // above untouched, where `clamp_min` is the identity. Found by
    // `tests/engram_oracle.rs::gate_is_on_the_clamp_min_side`, which had been
    // RED since 2026-09-28 and outside the gate until 2026-09-29.
    let g = dot.clone().abs().clamp_min(1e-6).sqrt().mul(dot.sign());
    activation::sigmoid(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn multi_hash_embed_shape() {
        let m = MultiHashEmbedding::new(&[1000, 1000], 32, &dev());
        assert_eq!(
            m.forward(Tensor::<3, Int>::zeros([1, 4, 2], &dev())).dims(),
            [1, 4, 64]
        );
    }
    #[test]
    fn engram_module_shape() {
        let m = EngramModule::new(&[1000, 1000, 1000], 64, 256, 1, &dev());
        let ids = Tensor::<3, Int>::zeros([2, 8, 3], &dev());
        let h = Tensor::<4>::zeros([2, 8, 1, 256], &dev());
        assert_eq!(m.forward(ids, h).dims(), [2, 8, 1, 256]);
    }
    #[test]
    fn engram_module_multi_head() {
        let m = EngramModule::new(&[1000, 1000], 32, 128, 4, &dev());
        let ids = Tensor::<3, Int>::zeros([1, 4, 2], &dev());
        let h = Tensor::<4>::zeros([1, 4, 4, 128], &dev());
        assert_eq!(m.forward(ids, h).dims(), [1, 4, 4, 128]);
    }
    #[test]
    fn gate_in_range() {
        let m = EngramModule::new(&[1000, 1000], 32, 64, 1, &dev());
        let ids = Tensor::<3, Int>::zeros([1, 4, 2], &dev());
        let h = Tensor::<4>::zeros([1, 4, 1, 64], &dev());
        let out = m.forward(ids, h).reshape([1, 4, 64]);
        let v: Vec<f32> = out
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(v.iter().all(|x| x.is_finite()));
    }
    #[test]
    fn short_conv_shape() {
        let m = EngramModule::new(&[1000, 1000], 32, 64, 2, &dev()).with_short_conv(4, 3, &dev());
        let ids = Tensor::<3, Int>::zeros([1, 8, 2], &dev());
        let h = Tensor::<4>::zeros([1, 8, 2, 64], &dev());
        assert_eq!(m.forward(ids, h).dims(), [1, 8, 2, 64]);
    }
}

#[cfg(test)]
mod gate_tests {
    use super::*;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn gate_matches_reference_formula() {
        // key = query = [1, 1, 4] constant -> dot/sqrt(4) = s; the reference
        // gate is sigmoid(sqrt(|s|+1e-6)*sign(s)); with s = 4 the plain
        // sigmoid would give 0.982, the reference gives sigmoid(2) = 0.881
        let key = Tensor::<3>::ones([1, 1, 4], &dev()).mul_scalar(2.0);
        let q = Tensor::<3>::ones([1, 1, 4], &dev()).mul_scalar(2.0);
        let g = compute_gate(key, q, 4);
        let v: Vec<f32> = g
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // RMSNorm maps both to ~1, so s = (1*1)*4 / sqrt(4) = 2 and the
        // reference gate is sigmoid(sqrt(2.000001)) = 0.804; the plain
        // sigmoid(2) = 0.881 differs enough to catch the wrong formula
        let expected = 1.0 / (1.0 + (-(2.0f32 + 1e-6).sqrt()).exp());
        assert!(
            (v[0] - expected).abs() < 1e-3,
            "gate {} vs ref {}",
            v[0],
            expected
        );
        let plain = 1.0 / (1.0 + (-2.0f32).exp());
        assert!(
            (v[0] - plain).abs() > 1e-3,
            "gate must use the sqrt compression, got {} (plain sigmoid {})",
            v[0],
            plain
        );
    }
}
