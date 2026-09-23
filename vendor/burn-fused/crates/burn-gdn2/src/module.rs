#[cfg(feature = "autodiff")]
use burn::backend::AutodiffBackend;
use burn::backend::Backend;
use burn::backend::DispatchKindConversion;
use burn::module::{Initializer, Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, DispatchTensor, Distribution as TensorDistribution, Tensor};

use burn::tensor::activation::{sigmoid, silu, softplus};

use crate::config::{Gdn2Config, Gdn2Mode};
use crate::forward::chunk_wy_forward;
use crate::kernel::fused_recurrent::fused_recurrent_gdn2;
use crate::l2norm::l2_normalize_4d;
use crate::short_conv::{short_conv_1d, SHORT_CONV_CACHE};

/// Chunked WY forward that dispatches to the two fused CUDA chunk kernels on
/// `CudaBare` (2 launches per chunk) and falls back to the tensor path
/// everywhere else.
///
/// Chunked WY forward variant selectable at runtime (plain tensor path, the
/// fused CUDA kernels or the fused autodiff op), so the training core stays
/// backend-agnostic.
type ChunkFn = fn(
    Tensor<4>,
    Tensor<4>,
    Tensor<4>,
    Tensor<4>,
    Tensor<4>,
    Tensor<4>,
    Tensor<4>,
    f64,
    usize,
) -> (Tensor<4>, Tensor<4>);

#[allow(clippy::too_many_arguments)]
fn chunk_wy_dispatch<B: Backend>(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<4>,
    b: Tensor<4>,
    w: Tensor<4>,
    state: Tensor<4>,
    scale: f64,
    chunk_size: usize,
) -> (Tensor<4>, Tensor<4>)
where
    DispatchTensor: DispatchKindConversion<B>,
{
    #[cfg(feature = "cuda")]
    if crate::kernel::chunk_cube::cuda::is_cuda::<B>() {
        if let Some(r) = crate::kernel::chunk_cube::cuda::fused_chunk_forward::<B>(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            b.clone(),
            w.clone(),
            state.clone(),
            scale,
            chunk_size,
        ) {
            return r;
        }
    }
    chunk_wy_forward(q, k, v, g, b, w, state, scale, chunk_size)
}

/// Gain used for Xavier-uniform init of all linear layers, matching the
/// reference implementation (`gain = 2^-2.5`).
/// Decay to the reference: sequences of at most this length always use the
/// fused-recurrent path at inference, longer ones fall back to `mode`.
const FUSED_RECURRENT_MAX_SEQ: usize = 64;

/// Recurrent + short-convolution state of a [`GatedDeltaNet2`] layer.
///
/// Carries everything needed to continue generation token by token after a
/// prefill pass:
/// - `recurrent`: the matrix state `S [B, HV, K, V]` of the delta rule,
/// - `conv_q`/`conv_k`/`conv_v`: the last `SHORT_CONV_CACHE` projected values
///   of each short convolution `[B, SHORT_CONV_CACHE, K/V]`.
///
/// The convolution cache matters: without it, decoding token `t` would pad the
/// conv with the *current* token instead of the previous ones and drift from
/// the prefill trajectory.
#[derive(Debug, Clone)]
pub struct Gdn2State {
    /// Recurrent matrix state `[B, HV, K, V]`.
    pub recurrent: Tensor<4>,
    /// Short-conv context for q `[B, SHORT_CONV_CACHE, K]`.
    pub conv_q: Tensor<3>,
    /// Short-conv context for k `[B, SHORT_CONV_CACHE, K]`.
    pub conv_k: Tensor<3>,
    /// Short-conv context for v `[B, SHORT_CONV_CACHE, V]`.
    pub conv_v: Tensor<3>,
}

impl Gdn2State {
    /// Create a zero-initialized state (used as a fresh prefill/decoding state).
    ///
    /// `recurrent` is per-head `[B, HV, head_dim, head_v_dim]`; the conv
    /// caches carry the projected channel dims `[B, SHORT_CONV_CACHE, K/V]`.
    pub fn zeros(
        device: &Device,
        batch: usize,
        hv: usize,
        hk: usize,
        v_head: usize,
        kd: usize,
        vd: usize,
    ) -> Self {
        Self {
            recurrent: Tensor::zeros([batch, hv, hk, v_head], device),
            conv_q: Tensor::zeros([batch, SHORT_CONV_CACHE, kd], device),
            conv_k: Tensor::zeros([batch, SHORT_CONV_CACHE, kd], device),
            conv_v: Tensor::zeros([batch, SHORT_CONV_CACHE, vd], device),
        }
    }
}

/// Updated short-convolution context returned by [`GatedDeltaNet2::project`]:
/// the last `SHORT_CONV_CACHE` projected values of q, k and v.
pub type ConvCache = (Tensor<3>, Tensor<3>, Tensor<3>);

/// GDN-2 (Gated DeltaNet 2) token-mixing layer.
///
/// GDN-2 decouples the erase and write operations of the gated delta rule
/// into two independent channel-wise gates:
///   - `b` (erase gate, `[0,1]^{d_k}`) - how much of each key channel to erase
///   - `w` (write gate, `[0,1]^{d_v}`) - how much of each value channel to write
///
/// The per-token recurrence on the matrix state `S ∈ R^{d_k × d_v}` is:
///
///   S_t = (I - k_t (b_t ⊙ k_t)ᵀ) diag(α_t) S_{t-1} + k_t (w_t ⊙ v_t)ᵀ
///
/// Two forward paths are available:
/// - `FusedRecurrent`: token-by-token scan (inference, state-passing)
/// - `Chunk`: chunked WY representation (training, cuBLAS-friendly matmuls)
#[derive(Module, Debug)]
pub struct GatedDeltaNet2 {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub f_proj_0: Linear,
    pub f_proj_1: Linear,
    pub b_proj: Linear,
    pub w_proj: Linear,
    pub g_proj_0: Linear,
    pub g_proj_1: Linear,
    pub a_log: Param<Tensor<1>>,
    pub dt_bias: Param<Tensor<1>>,
    pub o_norm_weight: Param<Tensor<1>>,
    pub o_proj: Linear,
    pub q_conv_w: Param<Tensor<2>>,
    pub k_conv_w: Param<Tensor<2>>,
    pub v_conv_w: Param<Tensor<2>>,
    #[module(skip)]
    pub config: Gdn2Config,
    pub decay_factors: Option<Param<Tensor<2>>>,
}

impl GatedDeltaNet2 {
    /// Create a new GDN-2 layer from a configuration.
    ///
    /// # Panics
    ///
    /// Panics if the configuration is invalid, see [`Gdn2Config::validate`].
    pub fn new(cfg: &Gdn2Config, device: &Device) -> Self {
        cfg.validate();
        let d = cfg.hidden_size;
        let h = cfg.num_heads;
        let hk = cfg.head_dim;
        let hv = cfg.num_v_heads.unwrap_or(h);
        let kd = h * hk;
        let v_head = (hk as f32 * cfg.expand_v) as usize;
        let vd = hv * v_head;

        let xavier = || Initializer::XavierUniform {
            gain: 2f64.powf(-2.5),
        };
        let lin = |i, o| {
            LinearConfig::new(i, o)
                .with_bias(false)
                .with_initializer(xavier())
                .init(device)
        };
        let lin_b = |i, o| {
            let mut lin = LinearConfig::new(i, o)
                .with_bias(true)
                .with_initializer(xavier())
                .init(device);
            // Reference code zero-initializes all biases.
            lin.bias = Some(Param::from_tensor(Tensor::zeros([o], device)));
            lin
        };

        // Short-conv weights: U(-0.5, 0.5), matching the reference conv init
        // (kaiming_uniform with a=sqrt(5), fan_in = kernel_size = 4).
        let rand_w = |c: usize| -> Param<Tensor<2>> {
            let w =
                Tensor::<2>::random([c, 4], TensorDistribution::Uniform(-0.5f64, 0.5f64), device);
            Param::from_tensor(w)
        };

        let a_init = Tensor::<1>::random(
            [h],
            TensorDistribution::Uniform(1.0f64.ln(), 16.0f64.ln()),
            device,
        );

        let dt = {
            let raw = Tensor::<1>::random(
                [kd],
                TensorDistribution::Uniform(0.001f64.ln(), 0.1f64.ln()),
                device,
            )
            .exp()
            .clamp(1e-4, f32::MAX);
            raw.clone() + (-raw.clone()).exp().neg().add_scalar(1.0).log()
        };

        Self {
            q_proj: lin(d, kd),
            k_proj: lin(d, kd),
            v_proj: lin(d, vd),
            f_proj_0: lin(d, v_head),
            f_proj_1: lin(v_head, kd),
            b_proj: lin(d, kd),
            w_proj: lin(d, vd),
            g_proj_0: lin(d, v_head),
            g_proj_1: lin_b(v_head, vd),
            a_log: Param::from_tensor(a_init),
            dt_bias: Param::from_tensor(dt),
            o_norm_weight: Param::from_tensor(Tensor::ones([v_head], device)),
            o_proj: lin(vd, d),
            q_conv_w: rand_w(kd),
            k_conv_w: rand_w(kd),
            v_conv_w: rand_w(vd),
            decay_factors: cfg
                .min_decay
                .map(|_| Param::from_tensor(Tensor::ones([h, hk], device).mul_scalar(2.0))),
            config: cfg.clone(),
        }
    }

    /// Validate input dimensions match the model configuration.
    ///
    /// # Panics
    ///
    /// Panics if `hidden_states` last dimension does not equal `config.hidden_size`,
    /// or if `state` is provided with mismatched dimensions.
    #[inline]
    fn validate(&self, hidden_states: &Tensor<3>) {
        let d = hidden_states.shape().dims::<3>()[2];
        assert_eq!(
            d, self.config.hidden_size,
            "GatedDeltaNet2: expected hidden_states with last dim {} but got {}. \
             Check that the model hidden_size matches your input.",
            self.config.hidden_size, d,
        );
    }

    /// Inference forward: one sequence pass with state management.
    ///
    /// Short sequences (≤ 64 tokens) always use the token-by-token
    /// `FusedRecurrent` path; longer ones use the configured `mode` (the
    /// `Chunk` path is substantially faster for long sequences). When
    /// `update_state` is true, the recurrent and short-convolution state is
    /// updated (normal autoregressive decoding). When false, the state is read
    /// without modification (useful for prefill).
    ///
    /// The conv cache in `state` is what makes token-by-token decoding
    /// equivalent to a single forward pass over the full sequence.
    ///
    /// # Panics
    ///
    /// Panics if `hidden_states` last dimension does not equal `config.hidden_size`.
    pub fn forward<B: Backend>(
        &self,
        hidden_states: Tensor<3>,
        state: &mut Option<Gdn2State>,
        update_state: bool,
    ) -> Tensor<3>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        self.validate(&hidden_states);
        let [batch, tokens, _] = hidden_states.shape().dims::<3>();

        let conv_in = state.as_ref().map(|s| (&s.conv_q, &s.conv_k, &s.conv_v));
        let (projected, conv_out) = self.project(hidden_states, conv_in);

        let hk = self.config.head_dim;
        let hv = projected.hv;
        let vd = projected.vd;
        let kd = self.config.num_heads * hk;
        let v_head = vd / hv;
        let scale = (hk as f64).powf(-0.5);
        let device = projected.q.device();

        let output = if update_state {
            let mode = if tokens <= FUSED_RECURRENT_MAX_SEQ {
                Gdn2Mode::FusedRecurrent
            } else {
                self.config.mode
            };
            let (o, new_rec) = match mode {
                Gdn2Mode::FusedRecurrent => {
                    let rec = state
                        .as_ref()
                        .map(|s| s.recurrent.clone())
                        .unwrap_or_else(|| Tensor::zeros([batch, hv, hk, v_head], &device));
                    let (o, ns) = fused_recurrent_gdn2::<B>(
                        projected.q,
                        projected.k,
                        projected.v,
                        projected.g,
                        projected.b,
                        projected.w,
                        Some(rec),
                        scale,
                        true,
                    );
                    (o, ns.expect("fused path always returns a state"))
                }
                Gdn2Mode::Chunk => {
                    let rec = state
                        .as_ref()
                        .map(|s| s.recurrent.clone())
                        .unwrap_or_else(|| Tensor::zeros([batch, hv, hk, v_head], &device));
                    chunk_wy_dispatch::<B>(
                        projected.q,
                        projected.k,
                        projected.v,
                        projected.g,
                        projected.b,
                        projected.w,
                        rec,
                        scale,
                        self.config.chunk_size,
                    )
                }
            };

            let st = state
                .get_or_insert_with(|| Gdn2State::zeros(&device, batch, hv, hk, v_head, kd, vd));
            st.recurrent = new_rec;
            if let Some((cq, ck, cv)) = conv_out {
                st.conv_q = cq;
                st.conv_k = ck;
                st.conv_v = cv;
            }
            o
        } else {
            // Read-only: consume the recurrent state without modifying it.
            let mem = state
                .as_ref()
                .map(|s| s.recurrent.clone())
                .unwrap_or_else(|| Tensor::zeros([batch, hv, hk, v_head], &projected.q.device()));
            projected
                .q
                .matmul(mem)
                .mul_scalar(scale)
                .permute([0, 2, 1, 3])
        };

        let out_4d = output.permute([0, 2, 1, 3]);
        let out_norm =
            rms_norm_gate_per_head(out_4d, projected.gate, projected.o_norm, projected.eps);
        projected
            .o_proj
            .forward(out_norm.reshape([batch, tokens, vd]))
    }

    /// Training forward: full-sequence forward, selects mode based on config.
    ///
    /// `FusedRecurrent` mode: token-by-token scan for small models.
    /// `Chunk` mode: chunked WY representation for training at scale.
    ///
    /// # Panics
    ///
    /// Panics if `hidden_states` last dimension does not equal `config.hidden_size`.
    pub fn forward_train<B: Backend>(&self, hidden_states: Tensor<3>) -> Tensor<3>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        self.forward_train_core::<B>(hidden_states, chunk_wy_dispatch::<B>)
    }

    /// Training forward with the fused autodiff chunk op.
    ///
    /// Same result as [`forward_train`](Self::forward_train) (verified by
    /// `tests/autodiff_chunk.rs`)
    /// but the whole chunked WY forward runs as a single autodiff node with an
    /// exact matrix-level backward — the per-op autodiff graph and its
    /// gradient bookkeeping are skipped.
    #[cfg(feature = "autodiff")]
    pub fn forward_train_fused<B: AutodiffBackend>(&self, hidden_states: Tensor<3>) -> Tensor<3>
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn_autodiff::Autodiff<B::InnerBackend>>,
    {
        self.forward_train_core::<B>(
            hidden_states,
            crate::autodiff::chunk_autodiff_or_plain::<B::InnerBackend>,
        )
    }

    fn forward_train_core<B: Backend>(&self, hidden_states: Tensor<3>, chunk: ChunkFn) -> Tensor<3>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        self.validate(&hidden_states);
        let [batch, tokens, _] = hidden_states.shape().dims::<3>();
        let (projected, _conv_out) = self.project(hidden_states, None);
        let hk = self.config.head_dim;
        let hv = projected.hv;
        let vd = projected.vd;
        let v_head = vd / hv;
        let scale = (hk as f64).powf(-0.5);
        let device = projected.q.device();
        let state = Tensor::zeros([batch, hv, hk, v_head], &device);

        let output = match self.config.mode {
            Gdn2Mode::FusedRecurrent => {
                let (o, _s) = fused_recurrent_gdn2::<B>(
                    projected.q,
                    projected.k,
                    projected.v,
                    projected.g,
                    projected.b,
                    projected.w,
                    Some(state),
                    scale,
                    true,
                );
                o
            }
            Gdn2Mode::Chunk => {
                let (o, _s) = chunk(
                    projected.q,
                    projected.k,
                    projected.v,
                    projected.g,
                    projected.b,
                    projected.w,
                    state,
                    scale,
                    self.config.chunk_size,
                );
                o
            }
        };

        let out_4d = output.permute([0, 2, 1, 3]);
        let out_norm =
            rms_norm_gate_per_head(out_4d, projected.gate, projected.o_norm, projected.eps);
        projected
            .o_proj
            .forward(out_norm.reshape([batch, tokens, vd]))
    }

    /// Shared projection pipeline: input → Q/K/V/B/W + gates.
    ///
    /// `conv_cache` optionally carries the short-convolution context from a
    /// previous call; returns the updated context as the second element.
    pub fn project(
        &self,
        hidden_states: Tensor<3>,
        conv_cache: Option<(&Tensor<3>, &Tensor<3>, &Tensor<3>)>,
    ) -> (ProjectedInputs<'_>, Option<ConvCache>) {
        let [batch, tokens, _] = hidden_states.shape().dims::<3>();
        let h = self.config.num_heads;
        let hk = self.config.head_dim;
        let hv = self.config.num_v_heads.unwrap_or(h);
        let v_head = (hk as f32 * self.config.expand_v) as usize;
        let vd = hv * v_head;
        let kd = h * hk;
        let use_sc = self.config.use_short_conv;

        let to_4d = |t: Tensor<3>, n: usize, d: usize| -> Tensor<4> {
            let [b, tt, _] = t.shape().dims::<3>();
            t.reshape([b, tt, n, d]).permute([0, 2, 1, 3])
        };

        let q_raw = self.q_proj.forward(hidden_states.clone());
        let k_raw = self.k_proj.forward(hidden_states.clone());
        let v_raw = self.v_proj.forward(hidden_states.clone());

        let (q_act, k_act, v_act, conv_out) = if use_sc {
            let cache = conv_cache.map(|(q, k, v)| (q.clone(), k.clone(), v.clone()));
            let (q_act, cq) =
                short_conv_1d(q_raw, self.q_conv_w.val(), cache.as_ref().map(|c| &c.0));
            let (k_act, ck) =
                short_conv_1d(k_raw, self.k_conv_w.val(), cache.as_ref().map(|c| &c.1));
            let (v_act, cv) =
                short_conv_1d(v_raw, self.v_conv_w.val(), cache.as_ref().map(|c| &c.2));
            (q_act, k_act, v_act, Some((cq, ck, cv)))
        } else {
            (silu(q_raw), silu(k_raw), silu(v_raw), None)
        };

        let f_hid = self.f_proj_0.forward(hidden_states.clone());
        let f_out = self.f_proj_1.forward(f_hid);
        let dt_b = self.dt_bias.val().reshape([1, 1, kd]);
        let a_exp = self
            .a_log
            .val()
            .exp()
            .reshape([1, h, 1])
            .repeat(&[1, 1, hk])
            .reshape([1, 1, kd]);
        let g_pre = softplus(f_out + dt_b, 1.0);
        let g = -a_exp * g_pre;

        let g = if let (Some(df), Some(min_d)) = (&self.decay_factors, self.config.min_decay) {
            let alpha = (1.0 - min_d) as f32;
            let factor = sigmoid(df.val())
                .mul_scalar(alpha)
                .reshape([1, h, hk])
                .reshape([1, 1, kd]);
            let d_lower_bounded = factor.mul(g.exp()).add_scalar(min_d as f32);
            d_lower_bounded.log()
        } else {
            g
        };

        let b_gate = sigmoid(self.b_proj.forward(hidden_states.clone()));
        let w_gate = sigmoid(self.w_proj.forward(hidden_states.clone()));

        let mut q_4d = to_4d(q_act, h, hk);
        let mut k_4d = to_4d(k_act, h, hk);
        let v_4d = to_4d(v_act, hv, v_head);
        let mut g_4d = to_4d(g, h, hk);
        let mut b_4d = to_4d(b_gate, h, hk);
        let w_4d = to_4d(w_gate, hv, v_head);

        // L2-normalize q and k per head (matching the reference kernels).
        q_4d = l2_normalize_4d(q_4d, 1e-6);
        k_4d = l2_normalize_4d(k_4d, 1e-6);

        // Repeat key-side tensors for grouped value attention (GVA)
        if hv > h {
            let rep = hv / h;
            let r = |t: Tensor<4>| -> Tensor<4> {
                t.unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, rep, 1])
                    .reshape([batch, hv, tokens, hk])
            };
            q_4d = r(q_4d);
            k_4d = r(k_4d);
            g_4d = r(g_4d);
            b_4d = r(b_4d);
        }

        if self.config.allow_neg_eigval {
            b_4d = b_4d.mul_scalar(2.0);
        }

        let g_hid = self.g_proj_0.forward(hidden_states);
        let gate_signal = self.g_proj_1.forward(g_hid);
        let gate_4d = gate_signal.reshape([batch, tokens, hv, v_head]);

        (
            ProjectedInputs {
                q: q_4d,
                k: k_4d,
                v: v_4d,
                g: g_4d,
                b: b_4d,
                w: w_4d,
                gate: gate_4d,
                o_norm: self.o_norm_weight.val(),
                eps: self.config.norm_eps,
                o_proj: &self.o_proj,
                hv,
                vd,
            },
            conv_out,
        )
    }
}

/// Per-head RMS norm with SiLU gate.
///
/// x: `[B, T, HV, V]`, gate: `[B, T, HV, V]`, weight: `[V]`
pub fn rms_norm_gate_per_head(
    x: Tensor<4>,
    gate: Tensor<4>,
    weight: Tensor<1>,
    eps: f64,
) -> Tensor<4> {
    let v = x.shape().dims::<4>()[3];
    let rms = x
        .clone()
        .powf_scalar(2.0)
        .mean_dim(3)
        .add_scalar(eps)
        .sqrt();
    let normed = x / rms;
    let w = weight.reshape([1, 1, 1, v]);
    normed * w * silu(gate)
}

pub struct ProjectedInputs<'a> {
    pub q: Tensor<4>,
    pub k: Tensor<4>,
    pub v: Tensor<4>,
    pub g: Tensor<4>,
    pub b: Tensor<4>,
    pub w: Tensor<4>,
    pub gate: Tensor<4>,
    pub o_norm: Tensor<1>,
    pub eps: f64,
    pub o_proj: &'a Linear,
    pub hv: usize,
    pub vd: usize,
}
