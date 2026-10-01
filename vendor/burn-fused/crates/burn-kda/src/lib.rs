//! # burn-kda - Kimi Delta Attention for Burn
//!
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//! Complete implementation of KDA from [Kimi Linear](https://arxiv.org/abs/2510.26692)
//! (fine-grained gating, chunkwise WY form) and [Kimi K3](https://arxiv.org/abs/2607.24653)
//! (§2.1.1, Eqs 1-6 - lower-bounded decay, full-rank output gate).
//!
//! Core recurrence (Eq 1):
//! ```text
//! S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
//! o_t = S_t^T q_t
//! ```
//! with data-dependent per-head write strength `beta_t^h = Sigmoid(W_beta^h x_t)`
//! (K3 Eq 2) and channel-wise decay `alpha_t = exp(g_t)` from a low-rank logit
//! `z_t = W_alpha^down(W_alpha^up x_t) + b_alpha`:
//! - Kimi K3 (**the running branch**): `g_t = g_min * Sigmoid(exp(A_h) z_t)`,
//!   fixed `g_min = -5`. **Verified** against FLA's own executed reference
//!   (`fla/ops/kda/gate.py::naive_kda_lowerbound_gate`, commit `9f38d249`) by
//!   `tests/kda_oracle.rs::k3_bounded_decay_matches_fla_reference`.
//! - Kimi Linear: `g_t = -exp(A_h) * Softplus(z_t)` (unbounded below). **This
//!   is what the sources say and it is NOT what [`DecayFn::Softplus`]
//!   computes** — that branch puts `exp(A_h)` *inside* the softplus. The two
//!   are different functions; see the branch's own comment and
//!   `tests/kda_oracle.rs::kimi_linear_softplus_decay_matches_fla_reference`,
//!   which is RED ON PURPOSE. The branch is not the default and the fix is A/B
//!   queue arm 5, so it is reported rather than changed.
//!
//! Training uses the chunked WY form, which is the same algebra as GDN-2 under
//! the mapping `b = beta` (key channels), `g = log(alpha)` (key channels),
//! `w_gate = beta` (value channels) and `scale = 1.0`; see [`crate::fused`] for
//! why `w_gate` is `beta` and not `1` (it was documented as `1` here until
//! 2026-09-30, and the code was right and the comment was wrong). Decoding uses
//! the exact chunked WY form (`forward_recurrent` is the exact per-token
//! reference). A fused CUDA kernel path (`feature = "cuda"`) reuses the GDN-2
//! chunked kernels through the same WY mapping.
//!
//! # `scale = 1.0` here, `head_k_dim**-0.5` in both references — OPEN
//!
//! FLA's `chunk_kda` and `fused_recurrent_kda` both default `scale = K ** -0.5`
//! (`chunk.py:474`, `fused_recurrent.py:261`) and `fla/layers/kda.py:262` calls
//! `chunk_kda` **without** a `scale` argument, so the official KDA layer runs
//! at `head_k_dim**-0.5`. This crate passes `1.0`. The factor enters only the
//! read `o = q·S`, never the state update, and the RMSNorm in [`KdaModule::output`]
//! is invariant to a constant rescale of its input, so it is absorbed to
//! `O(eps / mean(o²))` ≈ `O(1e-5)` in the running model — which is exactly why
//! no test, loss curve or seed comparison here can see it. It is still a real
//! divergence, and `tests/kda_oracle.rs` carries it as two RED-ON-PURPOSE tests
//! (`read_scale_matches_fla_reference`, `chunked_wy_applies_no_read_scale`)
//! with a green twin proving the mechanism already honours the scale when asked.
//! Changing `1.0` moves every number derived from this crate, so it is the
//! owner's call. Arithmetic and citations:
//! `docs/reviews/2026-09-30-kda-formula-audit.md` §3.2.
//!
//! # Decay init: `a_log = -3`, `b_alpha = +1` is OURS, and it is not a citation
//!
//! This pair has been quoted as "the Moonshot/FLA recipe". It is not in any of
//! the three sources, and each says something different:
//!
//! | source | what it says |
//! |---|---|
//! | Kimi K3 §2.1.1, arXiv:2607.24653v2 | "We initialize `A_h = 0`", and "each bias `b_alpha^h` is initialized following [57, 24, 139]" — i.e. following Kimi Linear / GDN / Mamba-2, not Moonshot's own constant |
//! | FLA `fla/layers/kda.py:176` (v0.5.2) | `A_log = zeros` **only under `safe_gate=True`** — the K3 lower-bounded branch, the one this crate runs as [`DecayFn::Sigmoid`] |
//! | FLA `fla/layers/kda.py:178` | otherwise `A_log = log(U(1, 16))` |
//! | FLA `fla/layers/kda.py:180-184` | `dt_bias = inv_dt = dt + log(-expm1(-dt))`, `dt ~ logU(0.001, 0.1)` ⇒ `inv_dt ∈ [-6.91, -2.25]`, **negative** |
//! | FlashKDA | there is no `kda.py` in the repo (master tree `7afb9f4`); `tests/torch_ref.py` takes `A_log`/`dt_bias` as arguments and initialises nothing; its tests use `torch.rand` and `torch.full(0.0)` |
//!
//! So `-3` / `+1.0` are this project's own choice, kept on its own
//! measurement: the decay init **plus** the `a_log` clamp extended the
//! NaN-free window to 110+ steps under heavy overfit (2026-08-29, dormouse
//! `AGENTS.md` §2.3). That record does not isolate `a_log` from the clamp or
//! from `b_alpha`, so it does not attribute the value to a paper either. An
//! honest "we chose this and measured it" is the whole claim.
//!
//! # The open discrepancy: the bias SIGN caps retention, not `A_h`
//!
//! Under [`DecayFn::Sigmoid`], `g = g_min·sigmoid(exp(A_h)·z)` and `z` starts at
//! `b_alpha`. `sigmoid(u) > 1/2` for `u > 0`, so **any non-negative `z` forces
//! `g < g_min/2 = -2.5`, i.e. `alpha < e^{g_min/2} = 0.0821`.** With
//! `b_alpha = +1` the init sits at `alpha = 0.0771`, 94% of that ceiling, and
//! no `A_h` lifts it: `A=+3 → 0.0067`, `A=0 → 0.0259`, `A=-10` (the clamp
//! floor) `→ 0.0821`. The reference family reaches `alpha ≈ 0.62-0.995`
//! because its bias is negative.
//!
//! | init | `alpha` | `1/(1-alpha)`, steps |
//! |---|---|---|
//! | ours (`A=-3`, `b=+1`) | 0.0771 | 1.08 |
//! | the paper's `A` with our bias (`A=0`, `b=+1`) | 0.0259 | 1.03 |
//! | our `A` with FLA's mean bias (`A=-3`, `b=-4.60`) | 0.1092 | 1.12 |
//! | `A=0`, `b=FLA mean inv_dt` — **FLA's `safe_gate` branch** | 0.9515 | 20.6 |
//!
//! 12x on `alpha`, 19x on effective memory. Neither knob alone reaches it: `A`
//! is not the lever, and the sign of the bias is not the whole lever — the pair
//! has to move together, which is exactly what the paper's recipe does. Changing
//! either changes initialisation and every number derived from it: that is an
//! A/B, not a patch. Note the same pair is *benign* under [`DecayFn::Softplus`]
//! (`alpha = 0.937`): there `exp(A_h)` is a plain multiplier on `softplus(z)`,
//! `A = -3` damps by 20x, and nothing caps retention — and that is also the
//! branch the `a_log` overflow clamp below exists for. Under the K3 sigmoid the
//! midpoint is 0 rather than `softplus(0) = 0.693`, so the same pair lands at
//! the floor. Which branch the pair was chosen for is not recoverable from this
//! repo's history; "ours, measured" is the claim it can carry.
//!
//! # GVA: the decay is parameterised on the KEY-head axis
//!
//! `a_log` is `[num_heads, 1]` and `b_alpha` is `[num_heads * head_dim]` —
//! [`KdaDecay::new`] is called with the key-head count, not the value-head one.
//! FLA parameterises on the value-head axis: `gate_dim = num_v_heads *
//! head_k_dim` and `A_log = zeros(num_v_heads)` "per value-head for native GVA
//! support" (`fla/layers/kda.py:166-167, 174-178`). So under
//! `num_v_heads = rep · num_heads` this crate holds `rep`× too few decay
//! parameters and `KdaModule::project` copies the key-head decay across the
//! group (`g_4d = r(g_4d)`). The forward is self-consistent, which is why a
//! shape-only check calls it a match; the **parameter** is the finding. Dormouse
//! runs `hv == h` (`dormouse-core/src/attention.rs:60` leaves `num_v_heads:
//! None`), so it is dormant there, but both crates ship GVA. Parameter shapes are
//! checkpoint format: changing them invalidates every burn-kda checkpoint.
/// Paper's fixed log-space decay floor (K3 Eq 5: `g_min = -5`, alpha > e^-5).
pub const G_MIN: f64 = -5.0;

/// Decay logit -> per-step log-decay mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecayFn {
    /// Kimi Linear: `g = -exp(A_h) * Softplus(z)`, alpha in (0, 1) - unbounded below.
    Softplus,
    /// Kimi K3: `g = g_min * Sigmoid(exp(A_h) z)`, alpha in (e^g_min, 1) - bounded.
    Sigmoid,
}

/// Output gate parameterization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateMode {
    /// Kimi Linear: `Sigmoid(W_g^down (W_g^up x))` - low rank, parameter-fair.
    LowRank,
    /// Kimi K3: `Sigmoid(W_g x)` - full rank.
    FullRank,
}

/// KDA layer configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct KdaConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_v_heads: Option<usize>,
    pub expand_v: f32,
    pub use_short_conv: bool,
    /// Low-rank logit rank (both papers use `rank = head_dim`).
    pub rank: usize,
    /// Decay logit mapping (Kimi Linear vs K3).
    pub decay_fn: DecayFn,
    /// Log-space decay floor (K3); pass `ln(min_decay)` to reinterpret a
    /// decay-space floor, `<= 0` uses the fixed `G_MIN = -5`.
    pub g_min: f64,
    /// Output gate parameterization.
    pub gate: GateMode,
    /// Apply rotary position embedding to q/k **before** the L2 norm.
    ///
    /// OFF by default and **not** in either paper: FLA's official KDA layer
    /// has no rope at all (`fla/layers/kda.py` at `9f38d249` contains zero
    /// `rotary`/`rope` occurrences; GatedDeltaNet likewise has no `use_rope`).
    /// Position in a Kimi-Linear-style hybrid comes from the *interleaved
    /// full-attention* layers (`fla/layers/attn.py:83,125`,
    /// `fla/models/hybrid.py:17-23`), which is a cross-family transplant into a
    /// pure-KDA loop. Required for post-training, not for pretrain parity
    /// (Qwen3.8 playbook, AGENTS.md §3.5 item 5: NoPE breaks SFT/RLVR).
    ///
    /// Adds **zero parameters** — rotary is a function of position, and FLA
    /// holds only non-persistent buffers (`fla/modules/rotary.py:379,384`), so
    /// no checkpoint format changes either.
    pub use_rope: bool,
    pub chunk_size: usize,
    pub norm_eps: f64,
}

impl Default for KdaConfig {
    fn default() -> Self {
        Self {
            hidden_size: 128,
            num_heads: 4,
            head_dim: 32,
            num_v_heads: None,
            expand_v: 1.0,
            use_short_conv: true,
            rank: 0,
            decay_fn: DecayFn::Sigmoid,
            g_min: G_MIN,
            gate: GateMode::FullRank,
            use_rope: false,
            chunk_size: 16,
            norm_eps: 1e-5,
        }
    }
}

impl KdaConfig {
    /// From a `Gdn2Config` (drop-in for the old wrapper) with K3 defaults.
    pub fn from_gdn2(cfg: &Gdn2Config) -> Self {
        Self {
            hidden_size: cfg.hidden_size,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim,
            num_v_heads: cfg.num_v_heads,
            expand_v: cfg.expand_v,
            use_short_conv: cfg.use_short_conv,
            rank: cfg.head_dim,
            decay_fn: DecayFn::Sigmoid,
            g_min: G_MIN,
            gate: GateMode::FullRank,
            use_rope: false,
            chunk_size: cfg.chunk_size,
            norm_eps: cfg.norm_eps,
        }
    }
}
#[cfg(feature = "autodiff")]
use burn::backend::AutodiffBackend;
use burn::backend::Backend;
use burn::backend::DispatchKindConversion;
use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::tensor::{activation, Device, DispatchTensor, Tensor, TensorData};
use burn_gdn2::{chunk_wy_forward, l2_normalize, short_conv_1d, Gdn2Config};

pub mod fused;

/// RoPE base. Upstream's own default, from two places: the rotary module takes
/// it as `RotaryEmbedding(dim, base=...)` (`fla/modules/rotary.py:327`) and the
/// hybrid spec defaults it — `rope_theta = normalized.get('rope_theta', 10000.)`
/// (`fla/models/hybrid.py:103`). A constant, not a knob, until something
/// measures otherwise.
pub const ROPE_THETA: f64 = 10000.0;

/// Rotary embedding on `[B, T, H*D]`: one full-head rotation per head, at
/// positions `0..T-1`.
///
/// This is `rotary_embedding_ref` (`fla/modules/rotary.py:30-36`) with
/// `interleaved = False` and a full-head rotation (`ro_dim = D`), which reduces
/// to `x1 = x[..D/2]`, `x2 = x[D/2..]`,
/// `out = [x1*cos - x2*sin, x2*cos + x1*sin]` at
/// `angle[t, i] = t * theta^(-2i/D)` — the frequencies of
/// `fla/modules/rotary.py:410-414`. Positions start at 0 and there is no
/// `seqlen_offset`: KDA carries a recurrent **state**, not a KV cache, so there
/// is no cache length to offset by.
///
/// **Placed before the L2 norm, and at a full-head rotation the order is
/// provably free.** FLA normalizes q/k *inside* the kernel
/// (`fla/ops/kda/chunk.py:56-60`), so a layer-level rope necessarily lands
/// ahead of the norm. The rotation is orthogonal, so `l2(rope(x)) == rope(l2(x))`
/// exactly and the choice costs nothing — pinned by
/// `tests/kda_oracle.rs::rope_commutes_with_l2norm`. A **partial** rotation is
/// not free (upstream's reference carries an un-rotated tail, and an L2 norm
/// over all head dims then mixes rotated and un-rotated coordinates), so this
/// takes no fraction: a future partial rope must re-derive the order rather
/// than inherit it.
///
/// Public because it is the oracle's seam: `tests/kda_oracle.rs` compares this
/// against FLA's own `rotary_embedding_ref` tensor for tensor.
pub fn apply_rope(x: Tensor<3>, n_heads: usize, head_dim: usize) -> Tensor<3> {
    let [b, t, d] = x.shape().dims::<3>();
    assert_eq!(d, n_heads * head_dim, "rope expects [B,T,H*HD]");
    assert!(head_dim % 2 == 0, "rope needs an even head dim");
    let dev = x.device();
    let half = head_dim / 2;

    // freqs = outer(positions, inv_freq), then cos/sin of it — upstream's own
    // `_update_cos_sin_cache` with `scale = None` (the default,
    // `fla/modules/rotary.py:419-447`) and its `_compute_inv_freq`
    // (`:410-414`).
    let mut cos_v = Vec::with_capacity(t * half);
    let mut sin_v = Vec::with_capacity(t * half);
    for p in 0..t {
        for i in 0..half {
            let a = p as f32 * (ROPE_THETA as f32).powf(-(2.0 * i as f32) / head_dim as f32);
            cos_v.push(a.cos());
            sin_v.push(a.sin());
        }
    }
    let cos = Tensor::<2>::from_data(TensorData::new(cos_v, [t, half]), &dev)
        .reshape([1, 1, t, half])
        .expand([b, n_heads, t, half]);
    let sin = Tensor::<2>::from_data(TensorData::new(sin_v, [t, half]), &dev)
        .reshape([1, 1, t, half])
        .expand([b, n_heads, t, half]);

    // [B,T,H*HD] -> [B,H,T,HD], then the two halves the rotation mixes.
    let x = x.reshape([b, t, n_heads, head_dim]).permute([0, 2, 1, 3]);
    let x1 = x.clone().slice([0..b, 0..n_heads, 0..t, 0..half]);
    let x2 = x.slice([0..b, 0..n_heads, 0..t, half..head_dim]);
    let o1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
    let o2 = x2 * cos + x1 * sin;
    Tensor::cat(vec![o1, o2], 3)
        .permute([0, 2, 1, 3])
        .reshape([b, t, d])
}

// ─── Data-dependent decay (Eq 2/5) ────────────────────────────────────

/// Data-dependent channel-wise decay (K3 report Eq 2/5).
///
/// ```text
/// z_t    = W^down(W^up(x_t)) + b_alpha            (Eq 2, low-rank logit)
/// g_t    = g_min * sigmoid(e^{A_h} * z_t)          (Eq 5, per-head log scale)
/// alpha_t = exp(g_t)  in (e^{g_min}, 1)
/// ```
#[derive(Module, Debug)]
pub struct KdaDecay {
    pub w_up: Linear,
    pub w_down: Linear,
    /// Decay-logit bias, shape `[n_heads * head_dim]`. KEY-head axis: under GVA
    /// (`num_v_heads > num_heads`) one bias is shared across a group, where FLA
    /// gives each value head its own (`fla/layers/kda.py:166-167, 174-178`).
    /// See the module docs, "GVA: the decay is parameterised on the KEY-head axis".
    pub b_alpha: Param<Tensor<1>>,
    /// Per-head log-scale `A_h` (Eq 5), shape `[n_heads, 1]`. Key-head axis, same
    /// GVA caveat as [`Self::b_alpha`].
    pub a_log: Param<Tensor<2>>,
    #[module(skip)]
    pub g_min: f64,
    #[module(skip)]
    pub decay_fn: DecayFn,
}

impl KdaDecay {
    /// `rank`: low-rank logit dimension `r` (both papers use `rank = head_dim`).
    /// `g_min`: log-space decay floor; K3 fixes it at -5; pass
    /// `ln(min_decay)` to reinterpret a decay-space floor.
    pub fn new(
        d_model: usize,
        n_heads: usize,
        head_dim: usize,
        rank: usize,
        g_min: f64,
        decay_fn: DecayFn,
        device: &Device,
    ) -> Self {
        let init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        Self {
            w_up: LinearConfig::new(d_model, rank)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            w_down: LinearConfig::new(rank, n_heads * head_dim)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            // OURS, deliberately not from any reference — the module docs
            // ("Decay init", "The open discrepancy") carry the sources this
            // was once misattributed to and the alpha each of them gives.
            // Keep the numbers; changing them re-initialises the model and
            // invalidates every checkpoint.
            b_alpha: Param::from_tensor(Tensor::ones([n_heads * head_dim], device)),
            a_log: Param::from_tensor(Tensor::full([n_heads, 1], -3.0, device)),
            g_min,
            decay_fn,
        }
    }

    /// `x`: `[B, T, D]` hidden states. Returns `alpha [B, T, H, HD]`.
    pub fn forward(&self, x: Tensor<3>) -> Tensor<4> {
        let z = self.w_down.forward(self.w_up.forward(x));
        let [b2, t2, hd2] = z.shape().dims::<3>();
        let z = z.add(self.b_alpha.val().clone().reshape([1, 1, hd2]));
        let n_heads = self.a_log.val().shape().dims::<2>()[0];
        let head_dim = hd2 / n_heads;
        let z_h = z.reshape([b2, t2, n_heads, head_dim]);
        // Clamp A before exp: an unconstrained A > 88 overflows fp32
        // (exp(A) = inf), and `inf * 0` at z == 0 poisons the whole decay
        // with NaN. A <= 20 keeps exp(A) ~ 4.8e8: decay still saturates to
        // alpha = 0/1 at the extremes, NaN becomes impossible. (Measured on
        // 5060 Ti 2026-08-29: unclamped NaN episodes at heavy overfit.)
        //
        // The clamp is OURS: no source has one, so the tier-(a) fixture
        // deliberately contains no A outside [-10, 20] and cannot test it
        // (`gen_kda_oracle.py`, case `A_clamp_endpoints`, says so in place).
        //
        // `a` below is A_h ITSELF after the clamp, not exp(A_h) -- the
        // exponential is applied per branch. `falsify.sh`'s B1 got this wrong on
        // its third attempt (it broadcast `mul(a)` where `mul(a.exp())` was
        // meant) and the mutant COMPILED and flipped every sign. Recorded
        // because a decision that looks fine and is not the reference's is the
        // exact shape of the two divergences this crate currently carries.
        let a = self
            .a_log
            .val()
            .clone()
            .reshape([1, 1, n_heads, 1])
            .clamp(-10.0, 20.0);
        let scaled = z_h.mul(a.exp());
        let g = match self.decay_fn {
            // Kimi Linear, AS THE SOURCES STATE IT: g = -exp(A_h) * Softplus(z).
            //
            // DIVERGES FROM THAT, and did so until 2026-09-30 when this comment
            // was corrected to match the code rather than the reverse:
            // `exp(A_h)` is INSIDE the softplus here, `-Softplus(exp(A_h) * z)`.
            // FLA has it outside, in two independent transcriptions in one file
            // -- the executed reference `naive_kda_gate` (`gate.py:50`) and the
            // triton twin (`gate.py:167`, `b_yg = -exp(b_A) * softplus(b_g)`).
            // The two are different functions, not a reparameterisation: they
            // agree only at A = 0, and at this crate's own init (A = -3, z = +1)
            // upstream gives alpha = 0.574 and this gives 0.512.
            //
            // No test in this crate can see it, because every comparison here is
            // arm-vs-arm and both arms read this same function. It is caught by
            // `tests/kda_oracle.rs::kimi_linear_softplus_decay_matches_fla_reference`,
            // which is RED ON PURPOSE; `tests/oracle/falsify.sh` mutant B1 is
            // the candidate fix and turns that red green.
            //
            // NOT FIXED HERE. This branch is not the default (`DecayFn::Sigmoid`
            // is, and every checkpoint in the tree was trained with it), but it
            // is the subject of A/B queue arm 5 and a numerical change to a
            // shipped objective is the owner's call.
            DecayFn::Softplus => activation::softplus(scaled, 1.0).neg(),
            // Kimi K3, the RUNNING branch: g = g_min * Sigmoid(exp(A_h) z), with
            // `g_min = -5` fixed. VERIFIED against FLA's executed
            // `naive_kda_lowerbound_gate` (gate.py:81) including the bound:
            // `tests/kda_oracle.rs::k3_bounded_decay_matches_fla_reference`.
            DecayFn::Sigmoid => activation::sigmoid(scaled).mul_scalar(self.g_min as f32),
        };
        g.exp()
    }
}

// ─── Exact single-step recurrence (Eq 1) ──────────────────────────────

/// One KDA step (Eq 1): decay -> erase -> write -> read.
///
/// `state`: `[B, H, DK, DV]`, `decay`: `[H, DK]`, `q/k`: `[B, H, DK]`,
/// `v`: `[B, H, DV]`, `beta`: scalar write strength in `(0, 1)`.
///
/// Returns `(state, out [B, H, DV])`.
pub fn kda_step<B: Backend>(
    state: Tensor<4>,
    decay: Tensor<2>,
    q: Tensor<3>,
    k: Tensor<3>,
    v: Tensor<3>,
    beta: f64,
) -> (Tensor<4>, Tensor<3>) {
    let [b, h, dk, dv] = state.shape().dims::<4>();

    let state = state.mul(decay.clone().reshape([1, h, dk, 1]));
    let v_hat = state
        .clone()
        .swap_dims(2, 3)
        .matmul(k.clone().reshape([b, h, dk, 1]))
        .reshape([b, h, dv]);
    let delta = v.clone().sub(v_hat).mul_scalar(beta);
    let state = state + k.reshape([b, h, dk, 1]).mul(delta.reshape([b, h, 1, dv]));
    let out = q
        .reshape([b * h, 1, dk])
        .matmul(state.clone().reshape([b * h, dk, dv]))
        .reshape([b, h, dv]);
    (state, out)
}

// ─── Full KDA layer ───────────────────────────────────────────────────

/// Kimi Delta Attention layer (K3 report §2.1.1): delta rule with
/// data-dependent lower-bounded decay.
///
/// Block design (K3): `q,k = L2Norm(SiLU(ShortConv(Linear(x))))`,
/// `v = SiLU(ShortConv(Linear(x)))`; decay from Eq 2/5; scalar `beta`;
/// output gate `Sigmoid(W_g x) * RMSNorm(o)` before `o_proj`.
#[derive(Module, Debug)]
pub struct KdaModule {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub q_conv_w: Option<Param<Tensor<2>>>,
    pub k_conv_w: Option<Param<Tensor<2>>>,
    pub v_conv_w: Option<Param<Tensor<2>>>,
    pub decay: KdaDecay,
    /// Data-dependent write strength (K3 Eq 2): `beta_t^h = Sigmoid(W_beta^h x_t)`.
    pub beta_proj: Linear,
    /// Output gate: full-rank `W_g` (K3 Eq 6) or low-rank pair (Kimi Linear
    /// Eq 10): `Sigmoid(W_g^down(W_g^up x))`.
    pub o_gate: Option<Linear>,
    pub o_gate_up: Option<Linear>,
    pub o_gate_down: Option<Linear>,
    #[module(skip)]
    pub gate: GateMode,
    pub o_norm_w: Param<Tensor<1>>,
    pub o_proj: Linear,
    #[module(skip)]
    pub d_model: usize,
    #[module(skip)]
    pub n_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
    #[module(skip)]
    pub n_v_heads: usize,
    #[module(skip)]
    pub v_head_dim: usize,
    #[module(skip)]
    pub use_short_conv: bool,
    /// RoPE on q/k before the L2 norm. Off by default; see
    /// [`KdaConfig::use_rope`]. Not a parameter and not a buffer — rotary
    /// carries no weights, so this field is skipped by the derive and no
    /// checkpoint is affected by the flag.
    #[module(skip)]
    pub use_rope: bool,
    #[module(skip)]
    pub chunk_size: usize,
    #[module(skip)]
    pub norm_eps: f64,
}

impl KdaModule {
    /// Builds a KDA layer from a [`KdaConfig`].
    /// `min_decay`: decay-space floor, reinterpreted as the log-space
    /// `g_min = ln(min_decay)`; `min_decay <= 0` uses the config's `g_min`.
    pub fn new(cfg: &KdaConfig, min_decay: f64, device: &Device) -> Self {
        let d = cfg.hidden_size;
        let h = cfg.num_heads;
        let hk = cfg.head_dim;
        let hv = cfg.num_v_heads.unwrap_or(h);
        let v_head = (hk as f32 * cfg.expand_v) as usize;
        let init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        let rank = if cfg.rank > 0 { cfg.rank } else { hk };
        let g_min = if min_decay > 0.0 {
            min_decay.ln()
        } else {
            cfg.g_min
        };
        let conv = |dim: usize| -> Option<Param<Tensor<2>>> {
            cfg.use_short_conv.then(|| {
                Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                }
                .init([dim, 4], device)
            })
        };
        Self {
            q_proj: LinearConfig::new(d, h * hk)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            k_proj: LinearConfig::new(d, h * hk)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            v_proj: LinearConfig::new(d, hv * v_head)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            q_conv_w: conv(h * hk),
            k_conv_w: conv(h * hk),
            v_conv_w: conv(hv * v_head),
            decay: KdaDecay::new(d, h, hk, rank, g_min, cfg.decay_fn, device),
            beta_proj: LinearConfig::new(d, h)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            o_gate: (cfg.gate == GateMode::FullRank).then(|| {
                LinearConfig::new(d, hv * v_head)
                    .with_bias(false)
                    .with_initializer(init.clone())
                    .init(device)
            }),
            o_gate_up: (cfg.gate == GateMode::LowRank).then(|| {
                LinearConfig::new(d, hk)
                    .with_bias(false)
                    .with_initializer(init.clone())
                    .init(device)
            }),
            o_gate_down: (cfg.gate == GateMode::LowRank).then(|| {
                LinearConfig::new(hk, hv * v_head)
                    .with_bias(false)
                    .with_initializer(init.clone())
                    .init(device)
            }),
            gate: cfg.gate,
            o_norm_w: Initializer::Ones.init([v_head], device),
            o_proj: LinearConfig::new(hv * v_head, d)
                .with_bias(false)
                .with_initializer(init)
                .init(device),
            d_model: d,
            n_heads: h,
            head_dim: hk,
            n_v_heads: hv,
            v_head_dim: v_head,
            use_short_conv: cfg.use_short_conv,
            use_rope: cfg.use_rope,
            chunk_size: cfg.chunk_size,
            norm_eps: cfg.norm_eps,
        }
    }

    /// Projections + decay (K3 block design). Returns `(q, k, v, log_decay,
    /// beta, gate_signal)` with heads expanded for grouped value attention.
    /// Public wrapper of [`Self::project`] (testing / custom chunk paths).
    #[allow(clippy::type_complexity)]
    pub fn project_for_test(
        &self,
        x: Tensor<3>,
    ) -> (
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
    ) {
        self.project(x)
    }

    #[allow(clippy::type_complexity)]
    fn project(
        &self,
        x: Tensor<3>,
    ) -> (
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
        Tensor<4>,
    ) {
        let [batch, tokens, _] = x.shape().dims::<3>();
        let h = self.n_heads;
        let hk = self.head_dim;
        let hv = self.n_v_heads;
        let vd = self.v_head_dim;
        let to_4d = |t: Tensor<3>, n: usize, d: usize| -> Tensor<4> {
            let [b, tt, _] = t.shape().dims::<3>();
            t.reshape([b, tt, n, d]).permute([0, 2, 1, 3])
        };

        let q_raw = self.q_proj.forward(x.clone());
        let k_raw = self.k_proj.forward(x.clone());
        let v_raw = self.v_proj.forward(x.clone());

        let q_act = if self.use_short_conv {
            short_conv_1d(q_raw, self.q_conv_w.as_ref().unwrap().val(), None).0
        } else {
            activation::silu(q_raw)
        };
        let k_act = if self.use_short_conv {
            short_conv_1d(k_raw, self.k_conv_w.as_ref().unwrap().val(), None).0
        } else {
            activation::silu(k_raw)
        };
        let v_act = if self.use_short_conv {
            short_conv_1d(v_raw, self.v_conv_w.as_ref().unwrap().val(), None).0
        } else {
            activation::silu(v_raw)
        };

        // RoPE before the L2 norm, matching FLA's dataflow: its layer hands raw
        // q/k to the kernel and the kernel normalizes them
        // (`fla/ops/kda/chunk.py:56-60`), so anything the layer adds to q/k sits
        // ahead of the norm. See `apply_rope` for why the order is free at a
        // full-head rotation. Off by default — FLA's official KDA is NoPE.
        let (q_rot, k_rot) = if self.use_rope {
            (apply_rope(q_act, h, hk), apply_rope(k_act, h, hk))
        } else {
            (q_act, k_act)
        };

        let q_norm = l2_normalize(q_rot, 1e-6);
        let k_norm = l2_normalize(k_rot, 1e-6);

        let alpha = self.decay.forward(x.clone()); // [B, T, H, HD]
        let [_, _, _, hd] = alpha.shape().dims::<4>();
        let log_decay = alpha.log().permute([0, 2, 1, 3]); // [B, H, T, HD]
        let _ = hd;

        // Eq 2: beta_t^h = Sigmoid(W_beta^h x_t), per-head scalar in (0,1),
        // repeated over key channels (erase term) and value channels (write).
        let beta = activation::sigmoid(self.beta_proj.forward(x.clone())); // [B, T, H]
        let beta_h = beta.permute([0, 2, 1]).reshape([batch, h, tokens, 1]);
        let mut b_k = beta_h.clone().repeat(&[1, 1, 1, hk]);
        let mut b_v = beta_h.repeat(&[1, 1, 1, vd]);

        let mut q_4d = to_4d(q_norm, h, hk);
        let mut k_4d = to_4d(k_norm, h, hk);
        let v_4d = to_4d(v_act, hv, vd);
        let mut g_4d = log_decay;

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
            b_k = r(b_k);
            b_v = b_v
                .unsqueeze_dim::<5>(3)
                .repeat(&[1, 1, 1, rep, 1])
                .reshape([batch, hv, tokens, vd]);
        }

        let gate_logit = match self.gate {
            GateMode::FullRank => self.o_gate.as_ref().unwrap().forward(x),
            GateMode::LowRank => {
                let up = self.o_gate_up.as_ref().unwrap().forward(x);
                self.o_gate_down.as_ref().unwrap().forward(up)
            }
        };
        let gate_4d = activation::sigmoid(gate_logit)
            .reshape([batch, tokens, hv, vd])
            .permute([0, 2, 1, 3]);

        (q_4d, k_4d, v_4d, g_4d, b_k, b_v, gate_4d)
    }

    fn output(&self, attn_out: Tensor<4>, gate: Tensor<4>) -> Tensor<3> {
        let [b, hv, t, vd] = attn_out.shape().dims::<4>();
        let rms = attn_out
            .clone()
            .powf_scalar(2.0)
            .mean_dim(3)
            .add_scalar(self.norm_eps)
            .sqrt();
        let normed = attn_out / rms;
        let gated = normed
            .mul(gate)
            .mul(self.o_norm_w.val().reshape([1, 1, 1, vd]));
        self.o_proj
            .forward(gated.permute([0, 2, 1, 3]).reshape([b, t, hv * vd]))
    }

    /// Training forward: chunked delta-rule (WY algebra, `b = beta` on the key
    /// channels, `w_gate = beta` on the value channels) over a zero-initialized
    /// state.
    ///
    /// This said `w = 1` until 2026-09-30 and was **wrong about the code**: the
    /// call passes `b_v` in the `w_gate` position, and it has to. GDN-2's
    /// chunked form forms `U = (I + L)^{-1} (w_gate ⊙ V)`
    /// (`burn_gdn2/src/forward.rs:259`), and Eq 1's write term is `beta k vᵀ`, so
    /// `w_gate = 1` would drop beta from the write entirely and compute a
    /// different model. `src/fused.rs` had it right all along.
    ///
    /// Delegates to [`forward_train_state`](Self::forward_train_state) with
    /// `state: None` — identical behavior to the pre-carry implementation.
    pub fn forward_train<B: Backend>(&self, x: Tensor<3>) -> Tensor<3>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        self.forward_train_state::<B>(x, None).0
    }

    /// Training forward over `state` (chunked delta-rule, WY algebra) with
    /// state carry — the truncated-BPTT path for long-context training.
    ///
    /// Processes `x` as one chunk starting from `state` (zeros when `None`)
    /// and returns the updated state alongside the output. The caller threads
    /// the returned state into the next chunk's call. Truncated BPTT: the
    /// caller MUST `detach()` the returned state before passing it to the next
    /// chunk so gradient does not flow across chunk boundaries — on the tensor
    /// path (non-CUDA autodiff) the returned state is a graph node, so without
    /// the detach the graph grows with every chunk and backward slows/OOMs;
    /// on the fused autodiff path the state comes back untracked, making the
    /// detach a no-op there but still required for backend portability.
    ///
    /// Dispatch matches [`forward_train`](Self::forward_train): fused chunked
    /// kernels on the bare CUDA backend, ONE autodiff node over the fused
    /// kernels on `Autodiff<Cuda>`, the tensor-ops chunk path everywhere else.
    /// All paths compute the same WY form; the state shape `[B, H, K, V]` is
    /// independent of the token count, so `x` may be a routed active subset.
    pub fn forward_train_state<B: Backend>(
        &self,
        x: Tensor<3>,
        state: Option<Tensor<4>>,
    ) -> (Tensor<3>, Tensor<4>)
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let [batch, _t, _] = x.shape().dims::<3>();
        let (q, k, v, g, b_k, b_v, gate) = self.project(x);
        let [_, hv, _, _] = v.shape().dims::<4>();
        let dev = q.device();
        // State follows the compute dtype of the inputs: bf16 under the
        // model-wide BF16 mode (Moonshot FlashKDA trains bf16 state), fp32
        // otherwise (NdArray has no bf16).
        let state = state.unwrap_or_else(|| {
            Tensor::<4>::zeros([batch, hv, self.head_dim, self.v_head_dim], &dev).cast(q.dtype())
        });
        let (out, new_state) = {
            #[cfg(feature = "cuda")]
            {
                // The fused chunk kernels, on the bare CUDA backend or from
                // ANY autodiff wrapper of it (any checkpointing strategy).
                // The else arm is the tensor-ops chunk path.
                if let Some((o, s)) = fused::cuda::kda_fused_chunk_reported::<B>(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    g.clone(),
                    b_k.clone(),
                    b_v.clone(),
                    state.clone(),
                    self.chunk_size,
                )
                .into_option()
                {
                    (o, s)
                } else {
                    chunk_wy_forward(
                        q,
                        k,
                        v,
                        g,
                        b_k.clone(),
                        b_v,
                        state,
                        1.0,
                        self.chunk_size,
                    )
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                chunk_wy_forward(q, k, v, g, b_k.clone(), b_v, state, 1.0, self.chunk_size)
            }
        };
        (self.output(out, gate), new_state)
    }

    /// Training forward through the fused autodiff op of burn-gdn2: the whole
    /// chunked WY recurrence runs as ONE autodiff node with an exact
    /// matrix-level backward (the fused CUDA kernels run the forward on
    /// `CudaBare`; the backward never re-runs the forward). Same result as
    /// [`forward_train`](Self::forward_train), much faster on
    /// `Autodiff<B>` backends.
    #[cfg(feature = "autodiff")]
    pub fn forward_train_fused<B: AutodiffBackend>(&self, x: Tensor<3>) -> Tensor<3>
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn_autodiff::Autodiff<B::InnerBackend>>,
    {
        let [batch, _t, _] = x.shape().dims::<3>();
        let (q, k, v, g, b_k, b_v, gate) = self.project(x);
        let [_, hv, _, _] = v.shape().dims::<4>();
        let state = Tensor::<4>::zeros([batch, hv, self.head_dim, self.v_head_dim], &q.device())
            .cast(q.dtype());
        let (o, _) = burn_gdn2::chunk_autodiff_or_plain::<B::InnerBackend>(
            q,
            k,
            v,
            g,
            b_k,
            b_v,
            state,
            1.0,
            self.chunk_size,
        );
        self.output(o, gate)
    }

    /// Decode forward: chunked WY form of the per-token recurrence (Eq 1).
    /// One (fused) call per `chunk_size` tokens does the state update and
    /// output together, instead of ~13 tensor launches per token. Same
    /// algebra as [`forward_train`](Self::forward_train); the exact per-token
    /// reference is [`forward_recurrent`](Self::forward_recurrent).
    ///
    /// `update_state=true`: state is decayed/updated per token (autoregressive
    /// decoding). `update_state=false`: read-only prefill over the state.
    pub fn forward<B: Backend>(
        &self,
        x: Tensor<3>,
        state: &mut Option<Tensor<4>>,
        update_state: bool,
    ) -> Tensor<3>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let [batch, tokens, _] = x.shape().dims::<3>();
        let (q, k, v, g, b_k, b_v, gate) = self.project(x.clone());
        let [_, hv, _, _] = v.shape().dims::<4>();
        let dev = q.device();
        let s = state
            .take()
            .unwrap_or_else(|| {
                Tensor::<4>::zeros([batch, hv, self.head_dim, self.v_head_dim], &dev)
                    .cast(q.dtype())
            });

        let (out_4d, new_state) = if update_state {
            let mut outs = Vec::with_capacity(tokens.div_ceil(self.chunk_size));
            let mut s = s;
            let mut t = 0;
            while t < tokens {
                let e = (t + self.chunk_size).min(tokens);
                let sl = [0..batch, 0..hv, t..e];
                let (o_c, s_c) = {
                    #[cfg(feature = "cuda")]
                    {
                        if let Some(r) = fused::cuda::kda_fused_chunk::<B>(
                            q.clone().slice(sl.clone()),
                            k.clone().slice(sl.clone()),
                            v.clone().slice(sl.clone()),
                            g.clone().slice(sl.clone()),
                            b_k.clone().slice(sl.clone()),
                            b_v.clone().slice(sl.clone()),
                            s.clone(),
                            self.chunk_size,
                        ) {
                            r
                        } else {
                            chunk_wy_forward(
                                q.clone().slice(sl.clone()),
                                k.clone().slice(sl.clone()),
                                v.clone().slice(sl.clone()),
                                g.clone().slice(sl.clone()),
                                b_k.clone().slice(sl.clone()),
                                b_v.clone().slice(sl.clone()),
                                s,
                                1.0,
                                self.chunk_size,
                            )
                        }
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        chunk_wy_forward(
                            q.clone().slice(sl.clone()),
                            k.clone().slice(sl.clone()),
                            v.clone().slice(sl.clone()),
                            g.clone().slice(sl.clone()),
                            b_k.clone().slice(sl.clone()),
                            b_v.clone().slice(sl.clone()),
                            s,
                            1.0,
                            self.chunk_size,
                        )
                    }
                };
                outs.push(o_c);
                s = s_c;
                t = e;
            }
            (Tensor::cat(outs, 2), s)
        } else {
            let out = q.matmul(s.clone()).permute([0, 2, 1, 3]);
            (out, s)
        };
        *state = Some(new_state);
        self.output(out_4d, gate)
    }

    /// Exact per-token decode (Eq 1). Reference for the chunked
    /// [`forward`](Self::forward) path: same recurrence, different f32
    /// rounding. Use [`forward`](Self::forward) for decoding.
    pub fn forward_recurrent(
        &self,
        x: Tensor<3>,
        state: &mut Option<Tensor<4>>,
        update_state: bool,
    ) -> Tensor<3> {
        let [batch, tokens, _] = x.shape().dims::<3>();
        // `b_v`, not `b_k`. Both are the same per-head scalar (Eq 2) repeated
        // over a channel axis, so they are bit-identical whenever
        // `head_dim == v_head_dim` — which is every configuration that runs
        // today, `expand_v` defaulting to 1.0 and dormouse never setting it. But
        // the two multiplications below are on the VALUE axis and the shapes
        // only line up by that coincidence: `erased` is `[B, H, 1, DV]` and
        // `v_t` is `[B, H, 1, DV]`, while `b_k` is `[B, H, 1, DK]`. Under
        // `expand_v != 1.0` the old code raised a broadcast error rather than
        // returning a wrong number, so this is a latent break and not a silent
        // one — but it made the crate's own exact-per-token *reference* unusable
        // one config away, and every chunk-path test compares against it.
        //
        // FLA gets the same thing right for the same reason: `naive.py:64`
        // applies `b_i` to `k_i` AND to `v_i - (k_i * S).sum(-2)`, with `b_i`
        // indexed by VALUE head.
        let (q, k, v, g, _b_k, b_v, gate) = self.project(x.clone());
        let [_, hv, _, _] = v.shape().dims::<4>();
        let dev = q.device();
        let s = state
            .take()
            .unwrap_or_else(|| {
                Tensor::<4>::zeros([batch, hv, self.head_dim, self.v_head_dim], &dev)
                    .cast(q.dtype())
            });

        // The per-head scalar of Eq 2, repeated over the VALUE channels, which
        // is the axis both multiplications below live on.
        let beta = b_v; // [B, H, T, DV]
        let (out_4d, new_state) = if update_state {
            let mut s = s;
            let mut outs = Vec::with_capacity(tokens);
            for t in 0..tokens {
                let q_t = q.clone().slice_dim(2, t..t + 1);
                let k_t = k.clone().slice_dim(2, t..t + 1);
                let v_t = v.clone().slice_dim(2, t..t + 1);
                let d_t = g.clone().slice_dim(2, t..t + 1).exp();
                // S <- Diag(alpha_t) S  (Eq 1). `g` is on the KEY axis, so
                // `swap_dims(2, 3)` is what scales S's key rows.
                let beta_t = beta.clone().slice_dim(2, t..t + 1); // [B, H, 1, DV]
                s = s * d_t.swap_dims(2, 3);
                // erase: S <- S - beta k (k^T S)
                let erased = (s.clone() * k_t.clone().swap_dims(2, 3))
                    .sum_dim(2)
                    .mul(beta_t.clone());
                s = s - k_t.clone().swap_dims(2, 3) * erased;
                // write: S <- S + beta k v^T
                s = s + k_t.swap_dims(2, 3) * v_t.mul(beta_t);
                // read: o = q^T S
                let out = (s.clone() * q_t.swap_dims(2, 3)).sum_dim(2);
                outs.push(out);
            }
            (Tensor::cat(outs, 2), s)
        } else {
            let out = q.matmul(s.clone()).permute([0, 2, 1, 3]);
            (out, s)
        };
        *state = Some(new_state);
        self.output(out_4d, gate)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Device;
    use burn::tensor::Distribution;
    fn dev() -> Device {
        Device::ndarray()
    }
    fn cfg(hidden: usize, heads: usize, head_dim: usize, decay_fn: DecayFn) -> KdaConfig {
        KdaConfig {
            hidden_size: hidden,
            num_heads: heads,
            head_dim,
            use_short_conv: false,
            decay_fn,
            ..Default::default()
        }
    }

    #[test]
    fn kda_decay_bounds() {
        // K3 Eq 5: alpha = exp(g_min*sigmoid(...)) in (e^{g_min}, 1)
        let dec = KdaDecay::new(32, 4, 16, 16, G_MIN, DecayFn::Sigmoid, &dev());
        let x = Tensor::<3>::random([2, 8, 32], Distribution::Default, &dev());
        let alpha = dec.forward(x.clone());
        let vals: Vec<f32> = alpha
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let lo = G_MIN.exp() as f32;
        assert!(
            vals.iter().all(|&v| v >= lo - 1e-6 && v <= 1.0 + 1e-6),
            "alpha out of (e^g_min, 1): lo={lo}, sample {:?}",
            &vals[..8]
        );
        // Kimi Linear: alpha in (0, 1) - unbounded below
        let dec2 = KdaDecay::new(32, 4, 16, 16, G_MIN, DecayFn::Softplus, &dev());
        let alpha2 = dec2.forward(x.clone());
        let vals2: Vec<f32> = alpha2
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(
            vals2.iter().all(|&v| v > 0.0 && v <= 1.0 + 1e-6),
            "softplus alpha must be in (0, 1), sample {:?}",
            &vals2[..8]
        );
    }

    #[test]
    fn kda_decay_is_data_dependent() {
        // Eq 2: the logit is a function of x - two different inputs give
        // different decays. Inputs are scaled up so the projection term
        // dominates the b_alpha = 1.0 anchor (our init, see module docs).
        let dec = KdaDecay::new(32, 2, 16, 16, G_MIN, DecayFn::Sigmoid, &dev());
        let x1 = Tensor::<3>::ones([1, 4, 32], &dev()).mul_scalar(50.0);
        let x2 = Tensor::<3>::ones([1, 4, 32], &dev()).mul_scalar(-50.0);
        let a1 = dec.forward(x1);
        let a2 = dec.forward(x2);
        let d: Vec<f32> = (a1 - a2)
            .abs()
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let max_d = d.iter().fold(0.0f32, |a, &x| a.max(x));
        assert!(max_d > 1e-3, "decay must depend on the input (Eq 2)");
    }

    #[test]
    fn kda_step_shapes() {
        let s = Tensor::<4>::zeros([2, 4, 32, 64], &dev());
        let d = Tensor::<2>::ones([4, 32], &dev()).mul_scalar(0.95);
        let q = Tensor::<3>::random([2, 4, 32], Distribution::Default, &dev());
        let k = Tensor::<3>::random([2, 4, 32], Distribution::Default, &dev());
        let v = Tensor::<3>::random([2, 4, 64], Distribution::Default, &dev());
        let (ns, o) = kda_step::<NdArray>(s, d, q, k, v, 0.5);
        assert_eq!(ns.dims(), [2, 4, 32, 64]);
        assert_eq!(o.dims(), [2, 4, 64]);
    }

    #[test]
    fn kda_module_forward() {
        let km = KdaModule::new(&cfg(64, 2, 32, DecayFn::Sigmoid), 0.9, &dev());
        let x = Tensor::<3>::random([1, 4, 64], Distribution::Default, &dev());
        assert_eq!(km.forward_train::<NdArray>(x).dims(), [1, 4, 64]);
    }

    #[test]
    fn chunked_decode_matches_recurrent() {
        // Decode fusion: the chunked WY path (forward) must equal the exact
        // per-token recurrence (forward_recurrent) in output AND carried
        // state, with T not a multiple of the chunk size.
        let km = KdaModule::new(&cfg(64, 2, 16, DecayFn::Sigmoid), 0.9, &dev());
        let x = Tensor::<3>::random([1, 40, 64], Distribution::Default, &dev());
        let mut s_c: Option<Tensor<4>> = None;
        let mut s_r: Option<Tensor<4>> = None;
        let out_c = km.forward::<NdArray>(x.clone(), &mut s_c, true);
        let out_r = km.forward_recurrent(x, &mut s_r, true);
        let diff: f32 = (out_c - out_r).powf_scalar(2.0).mean().into_scalar();
        assert!(
            diff < 1e-4,
            "chunked vs recurrent decode mismatch mse {diff}"
        );
        let st: f32 = (s_c.unwrap() - s_r.unwrap())
            .powf_scalar(2.0)
            .mean()
            .into_scalar();
        assert!(st < 1e-4, "carried state mismatch mse {st}");
    }

    #[test]
    fn chunk_matches_decode() {
        // The chunked training path must equal the per-token recurrence.
        let km = KdaModule::new(&cfg(64, 2, 16, DecayFn::Sigmoid), 0.9, &dev());
        let x = Tensor::<3>::random([1, 17, 64], Distribution::Default, &dev());
        let chunk_out = km.forward_train::<NdArray>(x.clone());
        let mut state: Option<Tensor<4>> = None;
        let decode_out = km.forward_recurrent(x, &mut state, true);
        let diff: f32 = (chunk_out - decode_out)
            .powf_scalar(2.0)
            .mean()
            .into_scalar();
        assert!(diff < 1e-4, "chunk vs decode mismatch mse {diff}");
    }

    #[test]
    fn chunk64_matches_decode() {
        // K3-paper chunk size: the tensor chunk path (burn-gdn2 0.5.1 tile
        // scheme) must stay exact for chunk 64 under weak decay too.
        let mut c = cfg(64, 2, 16, DecayFn::Softplus);
        c.chunk_size = 64;
        let km = KdaModule::new(&c, 0.9, &dev());
        let x = Tensor::<3>::random([1, 128, 64], Distribution::Default, &dev());
        let chunk_out = km.forward_train::<NdArray>(x.clone());
        let mut state: Option<Tensor<4>> = None;
        let decode_out = km.forward_recurrent(x, &mut state, true);
        let diff: f32 = (chunk_out - decode_out)
            .powf_scalar(2.0)
            .mean()
            .into_scalar();
        assert!(diff < 1e-4, "chunk64 vs decode mismatch mse {diff}");
    }

    #[test]
    fn beta_stays_in_unit_interval() {
        let km = KdaModule::new(&cfg(32, 2, 16, DecayFn::Sigmoid), 0.0, &dev());
        let x = Tensor::<3>::random([1, 4, 32], Distribution::Default, &dev());
        let beta = activation::sigmoid(km.beta_proj.forward(x));
        let vals: Vec<f32> = beta
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(
            vals.iter().all(|&v| v > 0.0 && v < 1.0),
            "beta must be in (0,1), sample {:?}",
            &vals[..8]
        );
    }

    #[test]
    fn low_rank_gate_matches_full_rank_shapes() {
        // Both gate modes produce the same output shape.
        let km_low = KdaModule::new(
            &KdaConfig {
                gate: GateMode::LowRank,
                ..cfg(64, 2, 32, DecayFn::Sigmoid)
            },
            0.9,
            &dev(),
        );
        let km_full = KdaModule::new(
            &KdaConfig {
                gate: GateMode::FullRank,
                ..cfg(64, 2, 32, DecayFn::Sigmoid)
            },
            0.9,
            &dev(),
        );
        let x = Tensor::<3>::random([1, 4, 64], Distribution::Default, &dev());
        assert_eq!(
            km_low.forward_train::<NdArray>(x.clone()).dims(),
            km_full.forward_train::<NdArray>(x).dims()
        );
    }

    #[test]
    fn softplus_chunk_matches_decode() {
        // Kimi Linear decay mapping: chunk (WY) == per-token recurrence.
        let km = KdaModule::new(&cfg(64, 2, 16, DecayFn::Softplus), 0.0, &dev());
        let x = Tensor::<3>::random([1, 17, 64], Distribution::Default, &dev());
        let chunk_out = km.forward_train::<NdArray>(x.clone());
        let mut state: Option<Tensor<4>> = None;
        let decode_out = km.forward_recurrent(x, &mut state, true);
        let diff: f32 = (chunk_out - decode_out)
            .powf_scalar(2.0)
            .mean()
            .into_scalar();
        assert!(diff < 1e-4, "softplus chunk vs decode mismatch mse {diff}");
    }

    #[test]
    fn beta_is_data_dependent() {
        // K3 Eq 2: beta_t^h = Sigmoid(W_beta^h x_t) - different inputs must
        // give different write strengths (not a global scalar).
        let km = KdaModule::new(&cfg(32, 2, 16, DecayFn::Sigmoid), 0.0, &dev());
        let x1 = Tensor::<3>::ones([1, 4, 32], &dev());
        let x2 = Tensor::<3>::ones([1, 4, 32], &dev()).mul_scalar(-3.0);
        let b1 = activation::sigmoid(km.beta_proj.forward(x1));
        let b2 = activation::sigmoid(km.beta_proj.forward(x2));
        let d: Vec<f32> = (b1 - b2)
            .abs()
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let max_d = d.iter().fold(0.0f32, |a, &x| a.max(x));
        assert!(
            max_d > 1e-3,
            "beta must depend on x (Eq 2), max diff {max_d}"
        );
    }

    /// Truncated-BPTT carry: 4 x [1, 1024] state-carry calls must equal one
    /// [1, 4096] training forward. The WY recurrence is grouping-independent
    /// (chunk boundaries are just state transfers, same per-token algebra —
    /// the existing `chunk_matches_decode` test pins the same equivalence at
    /// the 16-token tile level), so the only difference is f32 rounding
    /// accumulated over 4096 tokens: tolerance 1e-3 mse (vs 1e-4 at 17-40
    /// tokens) and documented for exactly that reason.
    #[test]
    fn forward_train_state_chunked_matches_full() {
        let km = KdaModule::new(&cfg(64, 2, 16, DecayFn::Sigmoid), 0.9, &dev());
        let x = Tensor::<3>::random([1, 4096, 64], Distribution::Default, &dev());
        let full = km.forward_train::<NdArray>(x.clone());
        let mut state: Option<Tensor<4>> = None;
        let mut chunk_outs = Vec::with_capacity(4);
        for c in 0..4 {
            let (o, s) = km.forward_train_state::<NdArray>(
                x.clone().slice([0..1, c * 1024..(c + 1) * 1024]),
                state,
            );
            chunk_outs.push(o);
            state = Some(s);
        }
        let chunked = Tensor::cat(chunk_outs, 1);
        let diff: f32 = (full - chunked).powf_scalar(2.0).mean().into_scalar();
        assert!(
            diff < 1e-3,
            "chunked state-carry vs one-shot forward mismatch mse {diff}"
        );
    }

    /// Gradient contract of the truncated-BPTT carry: with the state detached
    /// at the chunk boundary, the FIRST chunk's input x1 must receive ZERO
    /// gradient (its only path into the loss is the state, which the detach
    /// severs) while the SECOND chunk's params get nonzero, finite grads.
    /// The undetached control (state threaded without detach) must show x1
    /// receiving gradient — proving the detach — not some other mechanism —
    /// is what truncates. The carry must also change the second chunk's
    /// output (it is not a no-op).
    #[cfg(feature = "autodiff")]
    #[test]
    fn forward_train_state_truncated_bptt_grads() {
        type NdAD = burn::backend::autodiff::Autodiff<
            burn_ndarray::NdArray,
            burn::backend::autodiff::checkpoint::strategy::NoCheckpointing,
        >;
        let device = dev().autodiff();
        let km = KdaModule::new(&cfg(64, 2, 16, DecayFn::Sigmoid), 0.9, &device);
        // require_grad: a probeable leaf (random tensors are untracked in burn
        // 0.22-pre.2, so without this x1 can never receive a grad to read).
        let x1 = Tensor::<3>::random([1, 32, 64], Distribution::Default, &device).require_grad();
        let x2 = Tensor::<3>::random([1, 32, 64], Distribution::Default, &device);
        let target = Tensor::<3>::random([1, 32, 64], Distribution::Default, &device);
        let mse = |out: Tensor<3>| -> Tensor<1> { (out - target.clone()).powf_scalar(2.0).mean() };
        let max_grad = |km: &KdaModule, grads: &burn::tensor::Gradients| -> f32 {
            km.q_proj
                .weight
                .grad(grads)
                .map(|t| {
                    t.into_data()
                        .try_to_vec::<f32>()
                        .unwrap()
                        .into_iter()
                        .fold(0.0f32, |m, x| m.max(x.abs()))
                })
                .unwrap_or(f32::NAN)
        };
        let max_x1 = |grads: &burn::tensor::Gradients| -> f32 {
            x1.grad(grads)
                .map(|t| {
                    t.into_data()
                        .try_to_vec::<f32>()
                        .unwrap()
                        .into_iter()
                        .fold(0.0f32, |m, x| m.max(x.abs()))
                })
                .unwrap_or(0.0)
        };

        // (a) The training contract: 2-chunk carry, state detached at the
        // boundary. Loss touches only chunk 2's output.
        let (_, s1) = km.forward_train_state::<NdAD>(x1.clone(), None);
        let (out2, _) = km.forward_train_state::<NdAD>(x2.clone(), Some(s1.detach()));
        let grads = mse(out2.clone()).backward();
        let ga = max_grad(&km, &grads);
        assert!(
            ga.is_finite() && ga > 1e-10,
            "second-chunk grads dead: {ga}"
        );
        let x1_detached = max_x1(&grads);
        assert!(
            x1_detached < 1e-8,
            "truncated BPTT broken: chunk-1 input got grad {x1_detached} through the detached state"
        );

        // (b) Undetached control on the same weights: the state is threaded
        // without detach, so chunk 2's gradient must flow back into x1.
        let (_, s1b) = km.forward_train_state::<NdAD>(x1.clone(), None);
        let (out2b, _) = km.forward_train_state::<NdAD>(x2.clone(), Some(s1b));
        let grads_b = mse(out2b.clone()).backward();
        let x1_undetached = max_x1(&grads_b);
        assert!(
            x1_undetached > 1e-10,
            "control: without detach the chunk-1 input must receive grad, got {x1_undetached}"
        );

        // The carry must actually change the output: out2 reads a nonzero
        // state written by chunk 1 (decay alpha >= e^-5 keeps it alive),
        // so it differs from the zero-state run over x2 alone.
        let (out2c, _) = km.forward_train_state::<NdAD>(x2.clone(), None);
        let carry_effect: f32 = (out2 - out2c).powf_scalar(2.0).mean().into_scalar();
        assert!(
            carry_effect > 1e-8,
            "carried state had no effect on the second chunk output: {carry_effect}"
        );
    }
}
