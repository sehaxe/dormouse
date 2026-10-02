//! ByteFlow Net: the five-stage hierarchical architecture (paper §3, Fig. 1).
//!
//! bytes → local encoder (SWA+Canon) → coding-rate Top-K chunking → global
//! transformer (full causal attention) → multi-linear upsampling with large
//! residual → decoder (SWA+Canon) → next-byte logits.

use crate::chunk::{marginal_gains_exact, marginal_gains_l2, select_positions, RateMode};

use burn::module::{Module, Param};
use burn::nn::{
    Embedding, EmbeddingConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::tensor::{Bool, Device, IndexingUpdateOp, Int, Tensor};
use burn_swiglu::{SwiGLU, SwiGLUConfig};

/// Pure UTF-8 byte vocabulary (256). The BOS boundary is position 0 of the
/// sequence itself, not a vocabulary entry.
pub const VOCAB: usize = 256;

/// Canon layer (Allen-Zhu 2025): causal depthwise conv k=4 with learned
/// per-channel gates, paper eq. (10):
/// `Canon(h_t) = w0⊙h_t + w1⊙h_{t−1} + w2⊙h_{t−2} + w3⊙h_{t−3}`.
#[derive(Module, Debug)]
pub struct CanonLayer {
    /// Gates `[4, d]`, row k weights the tap `h_{t−k}`.
    pub gates: Param<Tensor<2>>,
}

impl CanonLayer {
    pub fn new(d: usize, device: &Device) -> Self {
        // Identity init (w0=1, rest 0): the layer starts as a no-op so the
        // pre-norm block trains stably from step one.
        let mut gates = vec![0.0f32; 4 * d];
        gates[..d].fill(1.0);
        Self {
            gates: Param::from_tensor(
                Tensor::<1>::from_floats(gates.as_slice(), device).reshape([4, d]),
            ),
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, t, d] = x.dims();
        // Left-pad by 3 so window taps h_{t−3}..h_t without state shifting.
        let pad = Tensor::<3>::zeros([b, 3, d], &x.device());
        let padded = Tensor::cat(vec![pad, x], 1);
        let gates = self.gates.val();
        let mut acc: Option<Tensor<3>> = None;
        for k in 0..4usize {
            let w = gates.clone().slice([k..k + 1, 0..d]).reshape([1, 1, d]);
            let tap = padded
                .clone()
                .slice([0..b, (3 - k)..(3 - k + t), 0..d])
                .mul(w);
            acc = Some(match acc {
                None => tap,
                Some(a) => a.add(tap),
            });
        }
        acc.unwrap()
    }
}

/// Precomputed rotary table (half-split convention), applied per head to q/k.
///
// TODO(бумага): Table 5 lists BFlowNet positional handling only as
// "Multi-level" heads; RoPE θ=500000 follows the hierarchical-family spec in
// Appendix C.3.2 (stated for AU-Net).
#[derive(Module, Debug)]
pub struct RopeTable {
    pub cos: Tensor<2>, // [max_seq_len, head_dim/2]
    pub sin: Tensor<2>,
    pub n_heads: usize,
}

impl RopeTable {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        max_seq_len: usize,
        base: f64,
        device: &Device,
    ) -> Self {
        let hd = d_model / n_heads;
        let half = hd / 2;
        let mut angles = vec![0.0f32; max_seq_len * half];
        for p in 0..max_seq_len {
            for i in 0..half {
                let inv = base.powf(-((2 * i) as f64) / hd as f64);
                angles[p * half + i] = (p as f64 * inv) as f32;
            }
        }
        let ang = Tensor::<1>::from_floats(angles.as_slice(), device).reshape([max_seq_len, half]);
        Self {
            cos: ang.clone().cos(),
            sin: ang.sin(),
            n_heads,
        }
    }

    /// Rotate `[B, T, D]` (heads split along D, halves split per head).
    pub fn apply(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, t, d] = x.dims();
        let nh = self.n_heads;
        let hd = d / nh;
        let half = hd / 2;
        let x4 = x.reshape([b, t, nh, hd]);
        let x1 = x4.clone().slice([0..b, 0..t, 0..nh, 0..half]);
        let x2 = x4.slice([0..b, 0..t, 0..nh, half..hd]);
        let c = self
            .cos
            .clone()
            .slice([0..t, 0..half])
            .reshape([1, t, 1, half]);
        let s = self
            .sin
            .clone()
            .slice([0..t, 0..half])
            .reshape([1, t, 1, half]);
        let o1 = x1.clone().mul(c.clone()).sub(x2.clone().mul(s.clone()));
        let o2 = x1.mul(s).add(x2.mul(c));
        Tensor::cat(vec![o1, o2], 3).reshape([b, t, d])
    }
}

/// Boolean mask (`true` = masked out): strictly future positions plus, with a
/// sliding window, everything farther than `w − 1` positions into the past.
fn causal_mask(t: usize, window: Option<usize>, device: &Device) -> Tensor<4, Bool> {
    // tril_mask(o) is true where j > i + o; offset 0 ⇒ strictly future.
    let mask = Tensor::<4, Bool>::tril_mask([1, 1, t, t], 0, device);
    match window {
        None => mask,
        Some(w) => {
            // Far past: j ≤ i − w ⇔ j < i − w + 1 ⇒ triu offset 1 − w.
            let far_past = Tensor::<4, Bool>::triu_mask([1, 1, t, t], -(w as i64) + 1, device);
            mask | far_past
        }
    }
}

/// Causal attention with optional sliding window and RoPE over a fused QKV
/// projection (the `X_Q, X_K, X_V` semantics of paper eq. (7)).
#[derive(Module, Debug)]
pub struct FlowAttention {
    pub qkv: Linear,
    pub out: Linear,
    pub rope: RopeTable,
    pub n_heads: usize,
    /// Attend only the last `window` positions incl. current; `None` = full
    /// causal (global stage).
    pub window: Option<usize>,
}

impl FlowAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        n_heads: usize,
        window: Option<usize>,
        max_seq_len: usize,
        rope_base: f64,
        device: &Device,
    ) -> Self {
        Self {
            qkv: LinearConfig::new(d_model, 3 * d_model)
                .with_bias(false)
                .init(device),
            out: LinearConfig::new(d_model, d_model)
                .with_bias(false)
                .init(device),
            rope: RopeTable::new(d_model, n_heads, max_seq_len, rope_base, device),
            n_heads,
            window,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [bsz, seq, d] = x.dims();
        let hd = d / self.n_heads;
        let device = x.device();
        let qkv = self.qkv.forward(x);
        let q = self
            .rope
            .apply(qkv.clone().slice([0..bsz, 0..seq, 0..d]))
            .reshape([bsz, seq, self.n_heads, hd])
            .swap_dims(1, 2);
        let k = self
            .rope
            .apply(qkv.clone().slice([0..bsz, 0..seq, d..2 * d]))
            .reshape([bsz, seq, self.n_heads, hd])
            .swap_dims(1, 2);
        let v = qkv
            .slice([0..bsz, 0..seq, 2 * d..3 * d])
            .reshape([bsz, seq, self.n_heads, hd])
            .swap_dims(1, 2);

        let scores = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar((hd as f64).powf(-0.5))
            .mask_fill(causal_mask(seq, self.window, &device), -1e9);
        let mixed = burn::tensor::activation::softmax(scores, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([bsz, seq, d]);
        self.out.forward(mixed)
    }
}

/// One pre-norm block, paper eq. (6)–(9):
/// `ĥ = Canon(h + Attn(LN(h)))`, `h' = Canon(ĥ + SwiGLU(LN(ĥ)))`.
#[derive(Module, Debug)]
pub struct FlowBlock {
    pub norm1: LayerNorm,
    pub attn: FlowAttention,
    pub canon1: CanonLayer,
    pub norm2: LayerNorm,
    pub ffn: SwiGLU,
    pub canon2: CanonLayer,
}

impl FlowBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        n_heads: usize,
        window: Option<usize>,
        d_ff: usize,
        max_seq_len: usize,
        rope_base: f64,
        device: &Device,
    ) -> Self {
        Self {
            norm1: LayerNormConfig::new(d_model).init(device),
            attn: FlowAttention::new(d_model, n_heads, window, max_seq_len, rope_base, device),
            canon1: CanonLayer::new(d_model, device),
            norm2: LayerNormConfig::new(d_model).init(device),
            ffn: SwiGLUConfig::new(d_model, d_ff).init(device),
            canon2: CanonLayer::new(d_model, device),
        }
    }

    pub fn forward(&self, h: Tensor<3>) -> Tensor<3> {
        let u = self.norm1.forward(h.clone());
        let h_hat = self.canon1.forward(h.add(self.attn.forward(u)));
        let v = self.norm2.forward(h_hat.clone());
        self.canon2.forward(h_hat.add(self.ffn.forward(v)))
    }
}

/// Architecture hyperparameters. Defaults follow BFlowNet-600M from paper
/// Appendix C.2, Table 5 (layers `[6, 20]`, dims `[512, 1536]`,
/// 8192→3200→8192).
#[derive(Clone, Debug)]
pub struct ByteFlowConfig {
    /// Local width `d_local`.
    pub d_local: usize,
    /// Global width `d_global` (`≫ d_local`, §3.3).
    pub d_global: usize,
    /// Global sequence length K after chunking.
    pub k_tokens: usize,
    /// Encoder depth E; reused by the decoder (§3.5 "identical architecture").
    pub e_layers: usize,
    /// Global transformer depth G.
    pub g_layers: usize,
    // TODO(бумага): Table 5 says "Multi-level" heads without a split; 64-dim
    // heads chosen to match the Llama baseline head size in C.3.1.
    pub n_heads_local: usize,
    pub n_heads_global: usize,
    /// Sliding-window size `w_local`.
    // TODO(бумага): BFlowNet's window is not stated; 512 mirrors the
    // hierarchical-family local window in Appendix C.3.2 ([512, 4096]).
    pub w_local: usize,
    // TODO(бумага): FFN widths are not itemized per stage; LLaMA-style
    // ≈8/3·d used here.
    pub d_ff_local: usize,
    pub d_ff_global: usize,
    /// Shared upsampling bins B (paper default 16, §3.4).
    pub bins: usize,
    /// Noise variance ε² of the coding rate, eq. (11).
    // TODO(бумага): value not given in the paper text.
    pub eps2: f64,
    /// Chunker scoring mode; [`RateMode::L2`] is the paper default fast path.
    pub rate_mode: RateMode,
    /// Byte-sequence length T the model is built for (RoPE table size).
    pub max_bytes: usize,
    /// RoPE base θ.
    pub rope_base: f64,
}

impl Default for ByteFlowConfig {
    fn default() -> Self {
        Self {
            d_local: 512,
            d_global: 1536,
            k_tokens: 3200,
            e_layers: 6,
            g_layers: 20,
            n_heads_local: 8,
            n_heads_global: 12,
            w_local: 512,
            d_ff_local: 1408,
            d_ff_global: 4096,
            bins: 16,
            eps2: 0.5,
            rate_mode: RateMode::L2,
            max_bytes: 8192,
            rope_base: 500_000.0,
        }
    }
}

/// Effective shared-bin count for byte length `t`: the absolute-position
/// grouping (`bin(t) = ⌊t·B/T⌋`, eq. 15) needs `t % B == 0` for the static
/// batched-GEMM reshape. Teacher-forced training windows keep the full B;
/// arbitrary inference lengths fall back to the largest divisor of `t` not
/// exceeding it (1 always divides — a single-bin linear upsample).
fn effective_bins(bins: usize, t: usize) -> usize {
    let bins = bins.max(1);
    if t.is_multiple_of(bins) {
        return bins.min(t.max(1));
    }
    (1..=bins.min(t))
        .rev()
        .find(|&b| t.is_multiple_of(b))
        .expect("1 divides every t")
}

/// ByteFlow Net (paper §3).
#[derive(Module, Debug)]
pub struct ByteFlowNet {
    pub embedding: Embedding,
    pub encoder: Vec<FlowBlock>,
    pub proj: Linear,
    pub global: Vec<FlowBlock>,
    /// Shared per-bin upsampling matrices, stacked `[B_bins, d_global, d_local]`.
    pub upsample_w: Param<Tensor<3>>,
    pub decoder: Vec<FlowBlock>,
    pub out: Linear,
    // Plain-value knobs (registered as constants by the Module derive):
    pub k_tokens: usize,
    pub bins: usize,
    pub eps2: f64,
    pub use_logdet_rate: bool,
    pub max_bytes: usize,
}

impl ByteFlowNet {
    pub fn init(config: ByteFlowConfig, device: &Device) -> Self {
        assert!(
            config.d_global.is_multiple_of(config.n_heads_global)
                && config.d_local.is_multiple_of(config.n_heads_local),
            "widths must divide evenly by heads"
        );
        let block =
            |d_model: usize, n_heads: usize, window: Option<usize>, d_ff: usize, max_seq: usize| {
                FlowBlock::new(
                    d_model,
                    n_heads,
                    window,
                    d_ff,
                    max_seq,
                    config.rope_base,
                    device,
                )
            };
        Self {
            embedding: EmbeddingConfig::new(VOCAB, config.d_local).init(device),
            encoder: (0..config.e_layers)
                .map(|_| {
                    block(
                        config.d_local,
                        config.n_heads_local,
                        Some(config.w_local),
                        config.d_ff_local,
                        config.max_bytes,
                    )
                })
                .collect(),
            proj: LinearConfig::new(config.d_local, config.d_global)
                .with_bias(false)
                .init(device),
            global: (0..config.g_layers)
                .map(|_| {
                    block(
                        config.d_global,
                        config.n_heads_global,
                        None,
                        config.d_ff_global,
                        config.k_tokens,
                    )
                })
                .collect(),
            upsample_w: Initializer::KaimingUniform {
                gain: 1.0,
                fan_out_only: false,
            }
            .init_with(
                [config.bins, config.d_global, config.d_local],
                Some(config.d_global),
                Some(config.d_local),
                device,
            ),
            decoder: (0..config.e_layers)
                .map(|_| {
                    block(
                        config.d_local,
                        config.n_heads_local,
                        Some(config.w_local),
                        config.d_ff_local,
                        config.max_bytes,
                    )
                })
                .collect(),
            out: LinearConfig::new(config.d_local, VOCAB).init(device),
            k_tokens: config.k_tokens,
            bins: config.bins,
            eps2: config.eps2,
            use_logdet_rate: config.rate_mode == RateMode::LogDet,
            max_bytes: config.max_bytes,
        }
    }

    /// Next-byte logits `[B, T, 256]` for raw byte ids `[B, T]`. Causal end to
    /// end; the chunker sees the full-sequence importance profile exactly as
    /// in teacher-forced training (paper §3.2).
    pub fn forward(&self, bytes: Tensor<2, Int>) -> Tensor<3> {
        let (h, z, sel) = self.encode_chunks(bytes, self.k_tokens);
        let g = self.global.iter().fold(z, |z, b| b.forward(z));
        self.decode_chunks(g, h, &sel)
    }

    /// Front stage of the hierarchy, split out for host architectures whose
    /// global stage is NOT this net's `global` blocks (Aria's UniversalLoop):
    /// local encoding + coding-rate Top-K chunking + projection. Returns
    /// `(h_local [B,T,d_local], z [B,K,d_global], sel [B,K] ascending)`.
    /// `sel` is a discrete argtopk product (no gradient path); pair with
    /// [`Self::decode_chunks`], passing back BOTH `z`-stage output and `h`.
    pub fn encode_chunks(
        &self,
        bytes: Tensor<2, Int>,
        k: usize,
    ) -> (Tensor<3>, Tensor<3>, Tensor<2, Int>) {
        let [bsz, t] = bytes.dims();
        assert!(t <= self.max_bytes, "T={t} exceeds RoPE table");
        assert!(k >= 1 && k <= t, "K={k} must be in 1..={t}");
        let h = self.embedding.forward(bytes);
        let h = self.encoder.iter().fold(h, |h, b| b.forward(h));

        // Discrete selection: scores feed argtopk only, no gradient path.
        let gains = if self.use_logdet_rate {
            marginal_gains_exact(h.clone(), self.eps2)
        } else {
            marginal_gains_l2(h.clone())
        };
        let sel = select_positions(gains, k); // [B,K] ascending
        let d_local = h.dims()[2];
        // torch-style gather needs a full-rank index tile.
        let sel3 = sel.clone().reshape([bsz, k, 1]).repeat_dim(2, d_local);
        let z = self.proj.forward(h.clone().gather(1, sel3));
        (h, z, sel)
    }

    /// Back stage paired with [`Self::encode_chunks`]: multi-linear upsampling
    /// with large residual (`s = h + ũ(g)`), decoder blocks, byte logits
    /// `[B, T, VOCAB]`. `g` holds the post-global latents `[B,K,d_global]`,
    /// `h`/`sel` come straight from the encode call.
    pub fn decode_chunks(&self, g: Tensor<3>, h: Tensor<3>, sel: &Tensor<2, Int>) -> Tensor<3> {
        let [bsz, t, _] = h.dims();
        let bins = effective_bins(self.bins, t);
        let s = h.add(self.upsample(&g, sel, t, bsz, bins));
        let d = self.decoder.iter().fold(s, |s, b| b.forward(s));
        self.out.forward(d)
    }

    /// Multi-linear reconstruction with large residual, paper eq. (14)–(17).
    ///
    /// `chunk(t)` counts selected boundaries ≤ t (position 0 is always one, so
    /// the count never dips below 1); `bin(t) = ⌊t·B/T⌋` groups contiguous
    /// positions so `g_chunk(t) @ W_bin(t)` runs as B batched GEMMs instead of
    /// materializing a per-position weight gather.
    fn upsample(
        &self,
        g: &Tensor<3>,
        sel: &Tensor<2, Int>,
        t: usize,
        bsz: usize,
        bins: usize,
    ) -> Tensor<3> {
        let [_, k, dg] = g.dims();
        let dl = self.upsample_w.dims()[2];
        let device = g.device();

        // Boundaries → chunk index per position via inclusive prefix count.
        // Target starts at zeros and indices are unique (ascending sel), so
        // Add is exactly a write here (Add is burn's only scatter update op).
        let bound = Tensor::<2, Int>::zeros([bsz, t], &device)
            .scatter(
                1,
                sel.clone(),
                Tensor::<2, Int>::ones([bsz, k], &device),
                IndexingUpdateOp::Add,
            )
            .cumsum(1); // #selected ≤ t, ≥ 1 because position 0 is selected
        let chunk_idx = bound.sub_scalar(1);

        // Absolute-position bin: ⌊t·B/T⌋ (eq. 15), contiguous ranges — the
        // static reshape below realizes the grouping without an index tensor.
        let per = t / bins;

        let g_expanded = g
            .clone()
            .gather(1, chunk_idx.reshape([bsz, t, 1]).repeat_dim(2, dg))
            .reshape([bsz, t, dg]); // [B,T,Dg]
        let grouped = g_expanded.reshape([bsz, bins, per, dg]);
        // Fallback-bin lengths use a PREFIX of the shared bin weights (the
        // absolute-position mapping covers exactly [0, bins)); divisible
        // lengths slice nothing.
        let w = self
            .upsample_w
            .val()
            .slice([0..bins, 0..dg, 0..dl])
            .reshape([1, bins, dg, dl]);
        grouped // s̃_t = g_chunk(t)·W_bin(t), then reshape back to T positions
            .matmul(w)
            .reshape([bsz, t, dl])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::RateMode;
    use burn::module::ModuleVisitor;
    use burn::tensor::Distribution;

    fn dev() -> Device {
        Device::ndarray()
    }

    fn tiny_config(rate_mode: RateMode) -> ByteFlowConfig {
        ByteFlowConfig {
            d_local: 16,
            d_global: 32,
            k_tokens: 6,
            e_layers: 2,
            g_layers: 2,
            n_heads_local: 2,
            n_heads_global: 2,
            w_local: 8,
            d_ff_local: 32,
            d_ff_global: 48,
            bins: 8,
            eps2: 0.5,
            rate_mode,
            max_bytes: 64,
            ..ByteFlowConfig::default()
        }
    }

    #[test]
    fn output_shape_is_static_across_input_lengths() {
        // Fixed [B, K, ...] internals mean the graph shape depends only on the
        // input length — any T divisible by bins works with the same model.
        let device = dev();
        let net = ByteFlowNet::init(tiny_config(RateMode::L2), &device);
        for t in [16usize, 32, 64] {
            let x = Tensor::<2, Int>::zeros([2, t], &device);
            assert_eq!(net.forward(x).dims(), [2, t, VOCAB], "T={t}");
        }
    }

    #[test]
    fn swa_block_respects_the_window() {
        // Perturbing a byte beyond the window + canon kernel reach must not
        // change earlier block outputs.
        let device = dev();
        let blk = FlowBlock::new(16, 2, Some(4), 32, 32, 10_000.0, &device);
        let x = Tensor::<3>::random([1, 16, 16], Distribution::Default, &device);
        let base = blk.forward(x.clone());
        let mut perturbed = x;
        let bump =
            Tensor::<1>::from_floats(vec![5.0f32; 16].as_slice(), &device).reshape([1, 1, 16]);
        perturbed = perturbed.slice_assign([0..1, 15..16, 0..16], bump);
        let changed = blk.forward(perturbed);
        let a: Vec<f32> = base
            .slice([0..1, 0..8, 0..16])
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        let b: Vec<f32> = changed
            .slice([0..1, 0..8, 0..16])
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        for (u, v) in a.iter().zip(b.iter()) {
            assert!((u - v).abs() < 1e-5, "future leaked into past");
        }
    }

    /// Collects one finite-ness verdict per parameter grad.
    struct GradAudit<'g> {
        grads: &'g burn::tensor::Gradients,
        path: Vec<String>,
        checked: usize,
        names: Vec<String>,
        bad: Vec<String>,
    }
    impl ModuleVisitor for GradAudit<'_> {
        fn enter_module(&mut self, name: &str, _container_type: &str) {
            self.path.push(name.to_string());
        }
        fn exit_module(&mut self, _name: &str, _container_type: &str) {
            self.path.pop();
        }
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let name = self.path.join(".");
            self.checked += 1;
            self.names.push(name.clone());
            match param.grad(self.grads) {
                None => self.bad.push(format!("{name}:missing")),
                Some(g) => {
                    if g.into_data()
                        .convert::<f32>()
                        .try_to_vec::<f32>()
                        .unwrap()
                        .iter()
                        .any(|v| !v.is_finite())
                    {
                        self.bad.push(format!("{name}:nonfinite"));
                    }
                }
            }
        }
    }

    #[test]
    fn backward_reaches_every_parameter_finitely() {
        // Includes Canon gates, W_proj and the stacked upsample weights: every
        // param must receive a finite gradient.
        let device = dev().autodiff();
        let net = ByteFlowNet::init(tiny_config(RateMode::L2), &device);
        let bytes = Tensor::<1, Int>::from_ints(
            [
                7u32, 3, 200, 41, 90, 5, 12, 88, 60, 1, 250, 17, 33, 76, 99, 4,
            ]
            .as_slice(),
            &device,
        )
        .reshape([1, 16]);
        let grads = net
            .forward(bytes)
            .powf_scalar(2.0)
            .sum_dim(2)
            .mean()
            .backward();
        let mut audit = GradAudit {
            grads: &grads,
            path: Vec::new(),
            checked: 0,
            names: Vec::new(),
            bad: Vec::new(),
        };
        net.visit(&mut audit);
        assert!(
            audit.checked >= 20,
            "expected to visit all params, saw {}",
            audit.checked
        );
        assert!(
            audit.names.iter().any(|p| p.contains("canon")),
            "canon gates not visited"
        );
        assert!(
            audit.names.iter().any(|p| p.contains("proj.weight")),
            "W_proj not visited"
        );
        assert!(
            audit.names.iter().any(|p| p.contains("upsample_w")),
            "upsample weights not visited"
        );
        assert!(audit.bad.is_empty(), "bad grads: {:?}", audit.bad);
    }

    #[test]
    fn forward_is_deterministic() {
        let device = dev();
        let net = ByteFlowNet::init(tiny_config(RateMode::L2), &device);
        let bytes = Tensor::<2, Int>::zeros([1, 24], &device);
        let a = net.forward(bytes.clone()).into_data();
        let b = net.forward(bytes).into_data();
        assert_eq!(a, b);
    }

    /// Arbitrary inference lengths: the upsample falls back to the largest
    /// divisor of T not exceeding the configured bins (training windows that
    /// divide evenly keep the full B).
    #[test]
    fn non_multiple_lengths_upsample_with_fallback_bins() {
        assert_eq!(effective_bins(16, 512), 16, "divisible: full B");
        assert_eq!(effective_bins(16, 24), 12, "largest divisor of 24 ≤ 16");
        assert_eq!(effective_bins(8, 11), 1, "prime above bins: single bin");
        assert_eq!(effective_bins(8, 7), 7, "t below bins: per-position bins");

        let device = dev();
        let net = ByteFlowNet::init(tiny_config(RateMode::L2), &device);
        // tiny_config bins=8; 12 does not divide → still produces logits.
        let x = Tensor::<2, Int>::zeros([2, 12], &device);
        assert_eq!(net.forward(x).dims(), [2, 12, VOCAB]);
    }

    /// The host-architecture split (encode_chunks → arbitrary global stage →
    /// decode_chunks) is bit-identical to the monolithic forward when the
    /// "arbitrary stage" IS the net's own global blocks.
    #[test]
    fn encode_decode_split_matches_monolithic_forward() {
        let device = dev();
        let net = ByteFlowNet::init(tiny_config(RateMode::L2), &device);
        let bytes = Tensor::<2, Int>::zeros([2, 16], &device);
        let mono = net.forward(bytes.clone());
        let (h, z, sel) = net.encode_chunks(bytes, net.k_tokens);
        let g = net.global.iter().fold(z, |z, b| b.forward(z));
        let split = net.decode_chunks(g, h, &sel);
        assert_eq!(mono.into_data(), split.into_data());
    }

    #[test]
    fn logdet_rate_mode_runs_finite() {
        let device = dev();
        let net = ByteFlowNet::init(tiny_config(RateMode::LogDet), &device);
        let bytes = Tensor::<2, Int>::zeros([1, 16], &device);
        let logits: Vec<f32> = net
            .forward(bytes)
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn canon_layer_matches_reference_formula() {
        let device = dev();
        let layer = CanonLayer::new(2, &device);
        // Non-identity gates so taps actually contribute.
        let gates_init: Vec<f32> = layer
            .gates
            .val()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        assert_eq!(
            gates_init,
            vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "identity init"
        );
        let h = Tensor::<1>::from_floats(
            [
                1.0, 10.0, //
                2.0, 20.0, //
                3.0, 30.0, //
                4.0, 40.0,
            ],
            &device,
        )
        .reshape([1, 4, 2]);
        // Identity init ⇒ out == h exactly.
        let got: Vec<f32> = layer
            .forward(h.clone())
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        let want: Vec<f32> = h.clone().into_data().convert::<f32>().to_vec().unwrap();
        assert_eq!(got, want, "identity init must be a no-op");

        // Set w1 = 2 per channel: out_t = h_t + 2·h_{t−1} (t≥1), h_0 unchanged.
        let mut gates = vec![0.0f32; 8];
        gates[..2].fill(1.0);
        gates[2..4].fill(2.0);
        let mut layer2 = layer;
        layer2.gates =
            Param::from_tensor(Tensor::<1>::from_floats(gates.as_slice(), &device).reshape([4, 2]));
        let got: Vec<f32> = layer2
            .forward(h.clone())
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        let hv = h.into_data().convert::<f32>().try_to_vec::<f32>().unwrap();
        let expect = |t: usize, c: usize| -> f32 {
            hv[t * 2 + c]
                + if t >= 1 {
                    2.0 * hv[(t - 1) * 2 + c]
                } else {
                    0.0
                }
        };
        for t in 0..4 {
            for c in 0..2 {
                let g = got[t * 2 + c];
                let e = expect(t, c);
                assert!((g - e).abs() < 1e-5, "[{t},{c}] {g} vs {e}");
            }
        }
    }
}
