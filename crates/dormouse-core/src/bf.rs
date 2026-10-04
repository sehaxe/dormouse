//! The ByteFlow × dormouse compat channel (lane bf-compat:
//! `docs/reviews/bf-compat-2026-10-03.md`).
//!
//! ByteFlow's front stage (embedding → local SWA/Canon encoder → coding-rate
//! Top-K chunker → `d_local → d_global` projection) compresses the byte
//! window `[B, T]` into patch latents `z [B, K, d_global]` (K =
//! `byteflow_k_tokens`, the 3–5× compression the 1M plan prices); the
//! DORMOUSE LOOP (`LoopBlock`, its KDA attention arm driving over K
//! positions) runs on those latents; and ByteFlow's own decode_chunks
//! (multi-linear upsample with large residual → the SWA/Canon decoder blocks
//! → the byte head) lifts the loop's K outputs back to `[B, T, 256]` logits.
//! Two `Linear`s pair the widths (`d_global ↔ d_model`); no other glue
//! exists.
//!
//! That is exactly the host-architecture split `ByteFlowNet` ships for
//! "whose global stage is NOT this net's global blocks"
//! (`ByteFlowNet::encode_chunks` / `decode_chunks`), and the crate pins the
//! split against its monolithic forward, so the front/back math here is the
//! crate's own, not a re-spelling.
//!
//! NOT THE 1M WINDOW. The lane brief excludes it: `max_seq_len >
//! byteflow_max_bytes` keeps its refusal, `T <= max_bytes` keeps the forward
//! assert, and the recurrent local window stays unwritten. The delivered
//! window is 512 bytes → 128 patches at the preset's settings.
//!
//! # Loss and shape contracts
//!
//! * **The objective is the BYTE CE from the decoder**, over all T
//!   positions: `forward_channel` runs the loop with `targets = None` and
//!   returns `mean CE(logits[T, 256], labels)` as the `rec` the trainer
//!   makes the loss. Two reasons — the decoder is the only exit, so a loss
//!   taken at K patch starts would leave `upsample_w`/`decoder`/`out` with
//!   no gradient, and the held-out eval scores `[B,T,256]` logits against
//!   all T bytes, so anything else would be a train CE that is not the
//!   quantity on the eval line. (The loop's own `L_Rec` path, and with it
//!   `lm_head`, is unused here.)
//! * The chunker's `sel` is a discrete argtopk product: no gradient through
//!   the SELECTION — the same treatment the paper's standalone net gives it.
//!   The encoder/proj behind it and everything after `z` train normally.
//! * The final norm + `lm_head` (the byte-level dormouse head) and the byte
//!   `embedding` are UNUSED on this path — the `[B,K,d_model]` latent leaves
//!   through `out_proj` and the byteflow decoder produces the logits — but
//!   stay in the module tree and the checkpoint, because the module tree IS
//!   the checkpoint format (the same decision the Engram tables made,
//!   `loop_block.rs::engram`). A missing gradient is not an error here: a
//!   pure-CE run's aux heads have none either.
//! * `validate` refuses every aux weight with `use_byteflow`; the assert in
//!   `forward_channel` holds that refusal at the runtime seam (the model
//!   constructors do not call `validate`), and the channel returns
//!   `aux = None` unconditionally rather than mis-shaping a DSpark window
//!   built for `[B,T]` ids against a `[B,K]` hidden.
//! * The channel runs fp32 end to end. `byteflow::check` refuses `--bf16`
//!   for both byteflow modes (ByteFlowNet has no bf16 handling, and §2.1 is
//!   why that would be slower than fp32 here anyway).

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Device;
use burn_byteflow::{ByteFlowConfig, ByteFlowNet};

use crate::config::DormouseConfig;

/// A `DormouseConfig`' byteflow half → the crate's own config. The dormouse
/// side reads `d_model` off the config as the loop width.
pub fn byteflow_config(c: &DormouseConfig) -> ByteFlowConfig {
    ByteFlowConfig {
        d_local: c.byteflow_d_local,
        d_global: c.byteflow_d_global,
        k_tokens: c.byteflow_k_tokens,
        e_layers: c.byteflow_e_layers,
        g_layers: c.byteflow_g_layers,
        n_heads_local: c.byteflow_heads_local,
        n_heads_global: c.byteflow_heads_global,
        w_local: c.byteflow_w_local,
        d_ff_local: c.byteflow_d_ff_local,
        d_ff_global: c.byteflow_d_ff_global,
        bins: c.byteflow_bins,
        eps2: c.byteflow_eps2,
        max_bytes: c.byteflow_max_bytes,
        ..ByteFlowConfig::default() // rate_mode/rope_base: crate defaults
    }
}

/// True when ANY dormouse arm is on: `use_byteflow` then runs the channel
/// (`model.bf`) instead of replacing the model. THE ONE LIST: the trainer's
/// dispatch calls this, and `config::validate` refuses the same arms by name
/// (everything here except `use_kda` and `moe_topk`, which it refuses
/// separately), so a config that reaches the dispatch is a config the
/// validator already ruled on. `use_kda` is the arm this lane admits.
pub fn compat_armed(c: &DormouseConfig) -> bool {
    c.use_byteflow
        && (c.use_kda
            || c.use_plain_attn
            || c.use_engram
            || c.use_mor
            || c.use_msa
            || c.use_gr
            || c.use_attnres
            || c.use_mhc
            || c.use_situ
            || c.moe_topk > 0)
}

/// Owned front/back stages + the two pairing Linears. `net.global` is BUILT
/// EMPTY here (`g_layers = 0`): the net module tree is otherwise untouched,
/// an Empty-`Vec` writes no record, so this channel's byteflow side stays
/// byte-identical to the standalone net's checkpoint — and ByteFlowNet's own
/// `forward` (never called on this path) would refuse K=0 loudly before it
/// could be mistaken as wired.
#[derive(Module, Debug)]
pub struct BfChannel {
    /// ByteFlow's own params: embed, encoder, proj, upsample_w, decoder, out.
    pub net: ByteFlowNet,
    /// `d_global → d_model`, no bias.
    pub in_proj: Linear,
    /// `d_model → d_global`, no bias.
    pub out_proj: Linear,
    /// `byteflow_k_tokens` (a `#[module(skip)]` plain field; the forward's
    /// loop shapes are a config constant, like `norm_eps` everywhere here).
    #[module(skip)]
    pub k: usize,
}

impl BfChannel {
    /// The run's channel, from a VALIDATED config (validate refuses the
    /// incompatible arms; this constructor skips shapes better than the
    /// forward asserts would). `net.global` carries NO parameters, so this
    /// never touches the optimizer and never lives in the checkpoint record.
    pub fn new(cfg: &DormouseConfig, device: &Device) -> Self {
        let bfc = byteflow_config(cfg);
        let (dg, dm, k) = (bfc.d_global, cfg.d_model, bfc.k_tokens);
        let net = ByteFlowNet::init(ByteFlowConfig { g_layers: 0, ..bfc }, device);
        Self {
            net,
            in_proj: LinearConfig::new(dg, dm).with_bias(false).init(device),
            out_proj: LinearConfig::new(dm, dg).with_bias(false).init(device),
            k,
        }
    }
}

/// The model's byteflow knob, from the config: `Some` iff the compat channel
/// is armed. The trainer and this function agree because both read
/// [`compat_armed`].
pub fn bf_for_model(c: &DormouseConfig, device: &Device) -> Option<BfChannel> {
    compat_armed(c).then(|| BfChannel::new(c, device))
}
