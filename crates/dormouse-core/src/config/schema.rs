//! The one flat model schema (ADR-0005): every default is written exactly
//! once, in the per-field serde defaults below; `Default` derives from them
//! by deserializing an empty table. There is no sectioned mirror - preset
//! TOMLs are flat tables of these same fields.
//!
//! # Invariants
//!
//! * **`DormouseConfig::default() == small.toml`**, and that is a test
//!   (`default_equals_small_preset`), checked in both directions. The `d_*`
//!   functions in this file are the only written copy of a default.
//! * **A field here always changes the network.** Step/schedule/optimizer
//!   knobs live in `dormouse-train`'s config seam, because they are exempt
//!   from the resume config-equality check (ADR-0005/0021): extending a run is
//!   a legal resume, so putting `steps` here would forbid it.
//! * **Fields that add parameters default OFF**, and they build their module
//!   with an `Option` or behind a flag, so an off-arm model is
//!   byte-identical to a build from before the arm existed. That is what keeps
//!   the pure-CE control a valid control and the measured preset parameter
//!   counts in `tests/preset_exec.rs` true. The arms: `use_gr`,
//!   `use_attnres`, `use_mhc`, `use_situ`, `use_mor`, and
//!   `aux_fb_weight > 0`.
//! * **Parsing is single-sourced.** [`ActQuant`] has one `FromStr`, and serde,
//!   `--set` and the CLI all route through it, so a value that is illegal in a
//!   flag is illegal in a TOML file.
//!
//! # What this module does NOT check
//!
//! Cross-field legality lives in [`super::validation`], not here. This module
//! parses and defaults; it does not refuse. An `Err` from `toml::from_str`
//! means the text is not a config at all.

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::{self, Visitor}};
use std::str::FromStr;
use crate::act_quant::ActFormat;

/// Activation quantization format, as the CONFIG spells it.
///
/// Config-level, so it is a two-variant enum rather than
/// [`ActFormat`]: the model's enum carries the four shipped formats and the
/// `None`-means-auto decision lives in the trainer. Only what a TOML file or a
/// `--set` value can say is here, which is why `bf16` is absent — the trainer
/// picks bf16-vs-fp8-vs-fp32 from the device and `--quant`.
///
/// One parser ([`FromStr`], ADR-0005): `fp4` | `4` | `8`, case-insensitive,
/// `int4`/`int8` accepted as aliases. Anything else is `Err` naming what was
/// expected — a silent fp32 default here would be a run that trained in fp32
/// while its config said it did not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActQuant {
    /// The e2m1 grid (8 magnitudes: 0, .5, 1, 1.5, 2, 3, 4, 6), scaled per
    /// block onto the FORMAT's max. Note what this does NOT cover: the
    /// attention path is unconditionally upgraded to at least 8 bits
    /// ([`crate::act_quant::ActFormat::attn`]), so `--act-quant fp4` has never
    /// run 4-bit attention — the FFN path only. Every pre-2026-09-28 fp4
    /// number is also invalid for a second, independent reason: `fp4` was not
    /// e2m1 at all then (it emitted a nonexistent 0.75 level and scaled blocks
    /// onto 1 instead of 6, making ~3 of 8 levels reachable).
    Fp4,
    /// Uniform integer quantization at this width in bits, 4 or 8. BitNet
    /// a4.8-style: the forward path quantizes with a straight-through
    /// estimator and the graph stays f32, so the arithmetic is unchanged and
    /// only the values are rounded.
    Int(u32),
}
impl From<ActQuant> for ActFormat {
    fn from(q: ActQuant) -> Self { match q { ActQuant::Fp4 => ActFormat::Fp4, ActQuant::Int(b) => ActFormat::Int(b) } }
}

impl FromStr for ActQuant {
    type Err = String;
    /// The one act-quant parser (ADR-0005): serde deserialization, --set and
    /// the CLI all route through here. Accepts fp4 | 4 | 8 (int4/int8 kept
    /// as aliases; case-insensitive).
    fn from_str(v: &str) -> Result<Self, Self::Err> {
        match v.to_ascii_lowercase().as_str() {
            "fp4" => Ok(ActQuant::Fp4),
            "4" | "int4" => Ok(ActQuant::Int(4)),
            "8" | "int8" => Ok(ActQuant::Int(8)),
            _ => Err(format!("act_quant expected fp4 | 4 | 8, got {v:?}")),
        }
    }
}

impl Serialize for ActQuant {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self { ActQuant::Fp4 => s.serialize_str("fp4"), ActQuant::Int(b) => s.serialize_str(&b.to_string()) }
    }
}
impl<'de> Deserialize<'de> for ActQuant {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = ActQuant;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("act_quant fp4/4/8") }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                v.parse::<ActQuant>().map_err(E::custom)
            }
            fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> { self.visit_str(&v) }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> { self.visit_str(&v.to_string()) }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> { self.visit_str(&v.to_string()) }
        }
        d.deserialize_any(V)
    }
}

// The schema defaults, written once. They mirror the `small` preset (the
// documented default preset): `DormouseConfig::default() == small`.
fn d_bf16() -> bool { false }
fn d_act_group() -> usize { 0 }
fn d_d_model() -> usize { 768 }
fn d_n_heads() -> usize { 12 }
fn d_head_dim() -> usize { 64 }
fn d_d_ffn() -> usize { 2048 }
fn d_vocab() -> usize { 256 }
fn d_max_seq_len() -> usize { 512 }
fn d_max_iter() -> usize { 4 }
fn d_rank() -> usize { 64 }
fn d_norm_eps() -> f32 { 0.001 }
fn d_true() -> bool { true }
fn d_n_experts() -> usize { 3 }
fn d_jepa_weight() -> f32 { 0.05 }
fn d_jepa_mask_frac() -> f32 { 0.15 }
fn d_jepa_mask_span() -> usize { 8 }
fn d_dspark_weight() -> f32 { 0.0 }
fn d_dspark_k() -> usize { 4 }
fn d_dspark_stride() -> usize { 16 }
fn d_aux_fb_weight() -> f32 { 0.0 }
fn d_aux_fb_horizon() -> usize { 2 }
fn d_engram_rows() -> usize { 25_000 }
fn d_engram_orders() -> Vec<usize> { vec![2, 3, 4] }
fn d_engram_dim() -> usize { 32 }
fn d_engram_lam_max() -> f32 { 0.5 }
fn d_false() -> bool { false }
fn d_mor_k() -> usize { 2 }
fn d_mor_bce_weight() -> f32 { 0.05 }

/// The flat model schema: every knob that changes the network, in one table.
///
/// # What this type is NOT
///
/// It is not the run's configuration. `steps`, `log_every`, `ckpt_every`,
/// `lr`, `eval`, `eval_every` and the optimizer choice live in
/// `dormouse-train`'s config seam, and that split is load-bearing: those keys
/// are exempt from the resume config-equality check, because extending a run
/// is a legal resume and a config here would be compared against the snapshot
/// on every restart (ADR-0005/0021).
///
/// # The three ways a field here can be set
///
/// A preset TOML (the normal way), a `--set key=value` override (which wins,
/// and resolves through the same parsers), or the default. The defaults are
/// the `small` preset — `default_equals_small_preset` fails if they drift.
///
/// # Legality
///
/// Deserialization here does not refuse anything cross-field. Every "these two
/// cannot both be on" and "this value would silently compute a different
/// objective" rule is in [`super::validation`], called once after the config
/// is fully resolved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DormouseConfig {
    /// Store and compute activations in bfloat16. **Slower than fp32 on this
    /// box**, not faster: the LLVM dialect has no bf16 type, so the backend
    /// deletes bf16 from its advertised element types and strips the bf16
    /// tensor-core families — `--bf16` stores bf16 and computes fp32 through
    /// cast copies (AGENTS.md §2.1, ADR-0016 bug 2). Do not "fix" a slow
    /// `--bf16` run by turning this off without saying so in the log: the
    /// comparison changes meaning.
    #[serde(default = "d_bf16")] pub bf16: bool,
    /// Activation quantization format, or `None` for auto. `None` is NOT
    /// "off": the trainer resolves it (bf16 mode -> Bf16; sm_120 -> Fp8;
    /// else Fp32), and `--quant` is the way to pin it. See [`ActQuant`].
    #[serde(default)] pub act_quant: Option<ActQuant>,
    /// Block size for activation quantization, in elements along `d_model`.
    /// `0` means one scale per token (a single group covering the whole row).
    /// Anything else must DIVIDE `d_model`: the quantizer reshapes `[b*t, d]`
    /// to `[b, d/g, g]`, so a non-divisor fails the reshape on the first
    /// forward — a shape panic a long way from the flag. Checked by
    /// [`super::validate`].
    #[serde(default = "d_act_group")] pub act_group: usize,
    /// Residual-stream width. Must be divisible by `n_heads`, by `act_group`
    /// when that is set, and by `mhc_streams` when mHC is on — each of those
    /// reshapes the hidden state, so the check belongs at config time rather
    /// than at first forward.
    #[serde(default = "d_d_model")] pub d_model: usize,
    /// Attention heads in the KDA arm. `d_model = n_heads * head_dim` is the
    /// relation that holds on every shipped preset (768 = 12 * 64).
    #[serde(default = "d_n_heads")] pub n_heads: usize,
    /// Per-head width in the KDA arm, independent of `d_model` so the two can
    /// be swept without changing the stream width.
    #[serde(default = "d_head_dim")] pub head_dim: usize,
    /// Expert FFN inner width. With `use_tsct` on this is the rank space's
    /// output width, and the FFN's parameter cost binds at `r * d_ffn` per
    /// expert rather than `d_model * d_ffn`, which is why `use_situ` costs
    /// only +4.28% on `small` when it widens the projection to `2 * d_ffn`.
    #[serde(default = "d_d_ffn")] pub d_ffn: usize,
    /// Token vocabulary. **256, and it is bytes**: dormouse is a byte-level
    /// LM with no tokenizer, so this is fixed by the task and not a tuning
    /// knob. It is a field because `serve`/`generate` need it, not because
    /// anyone should change it.
    #[serde(default = "d_vocab")] pub vocab: usize,
    /// The longest sequence the model will be asked about, in BYTE positions.
    /// Used for RoPE table sizing and by the trainer's shape checks. A batch
    /// longer than this is a shape error, not a truncation.
    #[serde(default = "d_max_seq_len")] pub max_seq_len: usize,
    /// **Loop depth**: how many times the weight-shared block runs. This is the
    /// model's compute knob and the cheapest big lever in the A/B queue (depth
    /// 2 vs 4). There is no halt head and no PonderNet: depth is fixed, and
    /// `L_Rec` is the honest unweighted mean CE over all iterations
    /// (ADR-0013). `--rand-depth` samples it per step and `--set
    /// use_mor=true` routes per position instead; both are un-A/B'd.
    #[serde(default = "d_max_iter")] pub max_iter: usize,
    /// TSCT spectral rank: every [`crate::param::LinearLike`] stores `U[d, r]`
    /// and `V[r, f]` instead of a `[d, f]` matrix, so `rank` sets the
    /// compression ratio of the whole model. `rank` must stay small enough
    /// that `r*f < d*f` or the "spectral" path is more expensive than the dense
    /// one it replaces.
    #[serde(default = "d_rank")] pub rank: usize,
    /// RMSNorm epsilon. 1e-3 is the shipped value, three orders above the
    /// f32 default, which is a deliberate stability choice for the fp32 runs
    /// on this box rather than a paper number.
    #[serde(default = "d_norm_eps")] pub norm_eps: f32,
    /// The KDA attention arm. Default ON, and `--no-kda` is the bisect flag:
    /// turning it off gives the model with NO attention arm at all, which is
    /// the one configuration in the archive whose held-out BPB is a number for
    /// the network that produced it (AGENTS.md §3.1).
    #[serde(default = "d_true")] pub use_kda: bool,
    /// Spectral (low-rank TSCT) linears vs plain dense ones. The spectral
    /// path is what makes the FFN 2048-wide on `small` (9 197 390 params,
    /// MEASURED on the instantiated model - the 7.5M in older comments here
    /// predates the 2026-09-27 memory re-pricing and is retracted); turning it
    /// off gives the A/B that decides whether it earns its retraction, quant
    /// machinery and ~1000 lines (at a matched param budget the FFN gets
    /// narrower, which is the comparison that actually means something).
    #[serde(default = "d_true")] pub use_tsct: bool,
    /// The hashed n-gram memory (Engram, arXiv 2601.07372): FNV-hashed 2/3/4-gram
    /// keys into per-order tables, one shared value projection over all orders,
    /// and a hard floor `min(w_mem, engram_lam_max)` on the memory branch's
    /// share of the block output.
    ///
    /// Default ON, `--no-engram` is the bisect. **This arm was switched off in
    /// the A/B queue's first control for a capacity reason, not a quality
    /// reason** — at 8M rows/order it was 99% of the model and monopolized the
    /// loss — see [`Self::engram_rows`] for the numbers and for the
    /// disagreement about the right capacity that is written down rather than
    /// averaged away.
    #[serde(default = "d_true")] pub use_engram: bool,
    /// Gated Residual (arXiv Qwen3.8-Flash-Next, Eq 31-34): the residual
    /// stream is read through a low-rank sigmoid gate and written with one
    /// scalar per branch, replacing ReZero's additive scale.
    ///
    /// OFF by default (owner decision, checkpoint compatibility). The arm has
    /// **never been A/B'd**, and the four equation defects the audit found on
    /// 2026-09-28 (a missing sigmoid and factor 2 on the write, a missing
    /// `1/nr` inside the SiLU on the read, and a readout taken one iteration
    /// behind) were all fixed in the same commit — so there is no number, good
    /// or bad, in this project's history. Placement is ours, not the report's:
    /// one GR per loop ITERATION of a weight-shared block against the report's
    /// one GR per sublayer of a 56-sublayer stack, which is a transposition, so
    /// the report's -0.026 does not transfer unmeasured.
    ///
    /// Mutually exclusive with `use_attnres` and `use_mhc` — all three replace
    /// the same statement in the loop, and two on would mean one is silently
    /// ignored. Refused by [`super::validate`].
    #[serde(default)] pub use_gr: bool,
    /// Attention Residuals (arXiv:2603.15031), the learned replacement for the
    /// loop's residual accumulation: the state after iteration `n` becomes a
    /// softmax mixture over the token embedding and the block-body outputs of
    /// iterations `0..=n`, with one learned pseudo-query per iteration slot
    /// (`w_l`, §5). OFF by default, like `use_gr`, for checkpoint
    /// compatibility - the arm's `d_model`-wide query vectors only exist when
    /// it is on, so every existing checkpoint loads unchanged.
    ///
    /// Score convention: the paper's, `q . RMSNorm(k)` with no temperature -
    /// `burn_attnres::ScoreForm::default()` is `Paper`. A `SqrtD` option
    /// exists in the crate so the temperature is a nameable A/B rather than an
    /// accident, but it is not wired to a config field: the first A/B this
    /// arm needs is AttnRes vs ReZero, not AttnRes vs AttnRes.
    #[serde(default)] pub use_attnres: bool,
    /// multi-hyper-connections (arXiv 2409.19606): the residual stream is
    /// split into `mhc_streams` streams that are mixed by a learned matrix
    /// instead of being added, replacing the loop's ReZero write.
    ///
    /// OFF by default, like `use_gr` and `use_attnres`, and mutually exclusive
    /// with both — one check over all three lives in [`super::validate`] so a
    /// fourth arm cannot leave a pair unguarded. Note this arm's shipped
    /// `mhc_streams` default of 2 is the K the phi study (2604.21106v3 §5.2)
    /// ran; the base paper's own n=4 is `--set mhc_streams=4`.
    #[serde(default)] pub use_mhc: bool,
    /// Streams the residual stream is split into when `use_mhc` is on. **Must
    /// divide `d_model`** and be `>= 1`: `MhcBlock` reshapes the hidden state
    /// to `[b, t, n, d_model/n]` in its first forward, so a non-divisor is a
    /// shape panic a long way from the flag — the same class as `act_group`,
    /// and the same reason it is a loud check rather than a clamp.
    #[serde(default = "d_mhc_streams")] pub mhc_streams: usize,
    /// SiTU-GLU in the expert FFN (arXiv:2607.24653v2 Eq 12, Kimi K3):
    /// `beta1*tanh(Wg x/beta1) * Sigmoid(Wg x) * beta2*tanh(Wu x/beta2)` at
    /// K3's own `beta1 = 4`, `beta2 = 25` (`burn_situ::K3_GATE_BETA` /
    /// `K3_UP_BETA`, which are constants - beta is fixed in the paper, not
    /// learned and not swept).
    ///
    /// OFF by default. **This is not a pure activation swap**: Eq 12 reads
    /// `Wg x` and `Wu x` separately, so `ExpertFFN::gate_up` becomes
    /// `d_model -> 2*d_ffn` when the flag is on (a `d -> f` projection that was
    /// named after two projections finally being two). On `small` that is
    /// +393 216 parameters, **+4.28%**, because TSCT binds (`r*f` per expert,
    /// not `d*f`); the honest A/B therefore needs a width-matched control,
    /// which is SwiGLU at the same `2f` - the paper's own comparison in
    /// §2.3.2. Off is bitwise identical: see `situ_off_is_bitwise_the_old_model`
    /// and the unchanged `preset_exec` counts.
    #[serde(default)] pub use_situ: bool,
    /// Experts selected per token in the sparse-routing FFN arm
    /// (Switch-style). **`0` is the dense default** — the whole bank is blended,
    /// which is the arm's own removal and therefore its control. A top-k
    /// larger than `n_experts` is REFUSED rather than clamped, because the
    /// silent result is the dense blend wearing the arm's label: the run would
    /// print `moe=` counters for a model that never routed anything.
    #[serde(default = "d_moe_topk")] pub moe_topk: usize,
    /// Switch §3.3 load-balancing coefficient. **`0.0` by default, and 0 with
    /// `moe_topk > 0` is DELIBERATELY LEGAL** — it is the arm without its
    /// balancer, which §1.2 requires in order to measure the arm at all, and
    /// the external review found every published coefficient three orders of
    /// magnitude too weak at our token count. The risk is countable rather than
    /// refusable: `probe::MOE_LB` prints `moe=<lb>/<sel>` = `0/<n>` for a
    /// routed run with no balancer. Setting it `> 0` with `moe_topk = 0` IS
    /// refused: a term with no selection to balance is a field that reads like
    /// an objective and contributes nothing.
    #[serde(default = "d_moe_lb_coef")] pub moe_lb_coef: f32,
    /// Experts in the FFN bank. Every expert carries its own TSCT factors, so
    /// this is a direct multiplier on the FFN's parameter count and on the
    /// routing decision's width. `moe_topk` must not exceed it.
    #[serde(default = "d_n_experts")] pub n_experts: usize,
    /// Weight of the JEPA auxiliary term (masked latent prediction against an
    /// EMA teacher + KoLeo), on top of the CE. **ON by default at 0.05.**
    /// `0.0` is the pure-CE baseline and queue row 1; the measured verdict is
    /// JEPA+KoLeo over pure CE, 3/3 seeds (`d8062d1`) — but every A/B
    /// conclusion from before 2026-09-27 is retracted, because the teacher was
    /// being fed the label sequence.
    #[serde(default = "d_jepa_weight")] pub jepa_weight: f32,
    /// Expected fraction of masked positions for the JEPA span mask, in
    /// `(0, 1)`. The span rule inverts the per-position start rate so the
    /// EXPECTED FRACTION is exactly this, so the field means what it says.
    #[serde(default = "d_jepa_mask_frac")] pub jepa_mask_frac: f32,
    /// Length of each masked span, in byte positions. A start masks itself and
    /// the `span - 1` positions after it, which is what makes a masked
    /// position unpredictable from a masked neighbour — the point of masking.
    /// A span of 8 at a 0.15 fraction is data2vec's ballpark.
    #[serde(default = "d_jepa_mask_span")] pub jepa_mask_span: usize,
    /// Weight of the DSpark draft-head term. **0.0 in the shipped default**,
    /// and the shipped production recipe also sets it 0 — the first DSpark
    /// measurement this project has ever made, because the window was one
    /// position shifted until 2026-09-29 (it read the byte it was asked to
    /// predict). No DSpark number survives that, so nothing here is a
    /// continuation of a previous one.
    #[serde(default = "d_dspark_weight")] pub dspark_weight: f32,
    /// How many future bytes the draft head corrects at once (`k`). The term's
    /// cost scales with it, and `k` must leave room for at least one anchor
    /// inside the sequence: the anchor count is `floor((t - k - 1) / stride)`.
    #[serde(default = "d_dspark_k")] pub dspark_k: usize,
    /// The DSpark window's anchor SPACING, in byte positions — a sampling
    /// density, not a count. `1` is a window at every position; the default
    /// `16` is ~6% of them, giving 31 anchors at `k=4`, `seq_len=512`. `0` is
    /// REFUSED: it collapses the k-step window into k copies of one CE.
    ///
    /// Why striding and not the paper's random sampling: sampling is a second
    /// RNG stream, and burn's global RNG is never seeded, which is the exact
    /// defect that forced the JEPA mask to be rebuilt from `(seed, step)`
    /// (ADR-0021) — two runs of one config drew different anchor sets, so no
    /// A/B was reproducible and a resume changed the objective. This is a
    /// determinism fix and has **never been A/B'd against sampling**; the
    /// honest claim is "deterministic and documented", not "equivalent".
    #[serde(default = "d_dspark_stride")] pub dspark_stride: usize,
    /// Weight of the FUTURE-BYTE auxiliary head (our adaptation of arXiv
    /// 2404.19737's per-horizon multi-token heads to a byte-level AR model).
    ///
    /// **0.0 - OFF by default, and the head does not EXIST at 0.0**
    /// (`AuxHeads::fb` is an `Option`, built iff this is non-zero, the same
    /// `cfg.use_gr.then(...)` shape `LoopBlock::gr` uses). Two consequences,
    /// both deliberate: a zero-weight run's parameter set, and therefore its
    /// checkpoint, is byte-identical to a build from before this field existed
    /// - so queue row 1 (pure CE) needs no re-baseline of its own - and an
    /// off-arm model carries no 197 120-parameter head that no loss ever
    /// touches, which would move every measured preset parameter count.
    /// `0.0` is the A/B's off position and `--set aux_fb_weight=0.1` is its
    /// on position; nothing in the shipped recipe turns it on.
    #[serde(default = "d_aux_fb_weight")] pub aux_fb_weight: f32,
    /// How far ahead the head predicts, in BYTE POSITIONS.
    ///
    /// `targets[q]` is the byte at `q + 1`, so the label for position `q` at
    /// horizon `k` is `targets[q + k]`: the byte `k + 1` positions after the
    /// one the main CE already asks for. `k = 1` would be a second copy of the
    /// main CE through an independent head - the same target, two parameter
    /// sets - so the default is **2**, and 0 is refused loudly by `validate`.
    /// ONE horizon, not a set: a head per horizon is what 2404.19737 does and
    /// what a capacity ladder would A/B, and it is not this arm. Horizon 4 is
    /// a later row in the queue, not a field with a list in it.
    #[serde(default = "d_aux_fb_horizon")] pub aux_fb_horizon: usize,

    // --- hashed n-gram memory (Engram): the arm's capacity budget ---
    /// Rows per n-gram order. A CAPACITY BUDGET, not a tuning knob: the arm
    /// was switched off on 2026-09-27 because at 8M rows/order it was 99% of
    /// the model (3 x 8M x 32 = 768M memory params against `small`'s measured
    /// 6.05M compute params, 9.2M total) and monopolized the loss (rec ->
    /// 0.005, held-out frozen at exactly uniform 8.000 BPB). Two pieces of
    /// evidence set this number, and they disagree at our scale - so the
    /// smaller one wins and the disagreement is written down rather than
    /// averaged away:
    ///
    /// 1. The measured slot-count curve at iso-parameter (arXiv 2601.16531v2,
    ///    Table 3, slots per order per head): Hash-300K 4.4825 / Hash-500K
    ///    **4.4809** / Hash-800K 4.4961 - 500K is the best point and 800K is
    ///    ~1.9 sigma worse (0.0152 / the one std that is reported for a Hash
    ///    row, 0.0082). 300K-500K is FLAT: 0.0016 against that same 0.0082.
    ///    Only 500K has a std (0.0082; the 300K/800K rows report "-"), and
    ///    the paper's largest point is 800K - there is no measured 8M, so
    ///    "16x past the knee" was OUR extrapolation, not its data.
    /// 2. The allocation ratio, which is what "monopoly" means. The curve
    ///    above was measured on a **~185M** GPT-2-arch backbone carrying a
    ///    128M Engram (Table 1; total 313 567 232) - so the paper's own
    ///    memory-to-compute density is 0.69x, which the paper itself calls
    ///    "likely much higher than in practical large-scale deployments".
    ///    At `small`'s real size (6.05M compute) 3 x 500_000 x 32 = 48M nominal is
    ///    7.9x the compute side, and the in-VRAM rounding takes it to
    ///    3 x 524_288 x 32 = 50.3M = 8.3x - 12x denser than the paper's own
    ///    config, and **89% of the model**: the exact shape that failed.
    ///
    ///    The allocation ratio this comment used to cite came from the WRONG
    ///    source: "196B Engram against a 552B backbone" is DeepSeek-V4.1-
    ///    Flash's shipped shape (196.6B Engram tables in layers 1 and 14,
    ///    ~384M rows x 256 dims, against a 552B backbone = 26.2% of total,
    ///    0.356x - model card / vLLM recipes, NOT arXiv 2601.07372, which has
    ///    no "552" and no "196" anywhere in it: its largest model is
    ///    Engram-40B at 39.5B total / 18.5B Engram). The ratio is real; the
    ///    citation was not.
    ///
    /// What the shipped number actually costs, and the two denominators it was
    /// previously quoted under (all of it measured on the instantiated models,
    /// `cargo test -p dormouse-core --test preset_exec -- --nocapture`, this
    /// box 2026-09-29 at 8ac7942 - re-derive by running it):
    ///
    /// | preset | memory rows | compute | total | share of total | x compute |
    /// |--------|------------:|--------:|------:|---------------:|----------:|
    /// | small  |     3 145 728 | 6 051 662 | 9 197 390 | **34.2%** | 0.52x |
    /// | nano   |     3 145 728 | 4 047 178 | 7 192 906 | **43.7%** | 0.78x |
    /// | base   |     3 145 728 | 9 920 082 | 13 065 810 | 24.1% | 0.32x |
    ///
    /// "share of total" is memory/(memory+compute) - the denominator
    /// `loop_block::tests::capacity_budget_is_a_minority_of_the_model` uses;
    /// "x compute" is memory/compute. The old comment quoted 24%, 0.32x and
    /// 29% as if they were one number: 24% and 29% were share-of-total with
    /// the NOMINAL 25 000 rows (2.4M) and with the rounded table (3.1M)
    /// respectively, and 0.32x was the other denominator, all three computed
    /// against the retracted 7.5M backbone.
    ///
    /// So the honest containment claim is not "inside 20-25%": at 34.2%
    /// (`small`) and 43.7% (`nano`) the shipped share is ABOVE that band, and
    /// what it IS inside is the range the Engram paper actually ships -
    /// 5.7B Engram in a 26.7B Engram-27B (21% of total, 0.27x a 21.0B compute
    /// side) up to 18.5B in a 39.5B Engram-40B (47%, 0.88x). The paper's
    /// "20-25%" is a fraction of the SPARSE (inactive) budget at its optimum
    /// rho ~ 80% (Sec. 3.1), not a fraction of the model, and this comment
    /// used to read it as the latter. This is the capacity decision the owner
    /// owns; the numbers above are the input to it.
    ///
    /// In-VRAM tables round UP to a power of two (the slot index is masked on
    /// device, not divided): 25_000 -> 32_768 rows, 3 145 728 params, which is
    /// the 3.1M in the table above. The host-RAM path (`--engram-ram
    /// --engram-slots`) takes its row count from that flag; the two are the
    /// same knob until the trainer's plumbing is unfrozen. The 500K point AT
    /// THIS SCALE is untested: it is the first rung of the capacity ladder the
    /// A/B queue should run next, and it is only worth running as a single
    /// seed, because at 500K the arm is the monopoly again (89%).
    #[serde(default = "d_engram_rows")] pub engram_rows: usize,
    /// N-gram orders, one table each, smallest first. At 8M rows only n=3
    /// had per-key support (the corpus exhausts the 16.8M 3-gram space 2750x
    /// over) while n=5 and n=8 were ~5775-way averages - 512M dead
    /// parameters, 2/3 of the table. 2/3/4 is DeepSeek's own shipped set
    /// over compressed tokens (V4.1-Flash n in 2,3,4; Engram-27B 2,3)
    /// and the deepest order whose key space (256^4 = 4.3e9) a 46 GB byte
    /// corpus can populate; n>=5 spaces (1.1e12) are hopeless. The VALUES
    /// are what `dormouse_data::ORDERS` hashes; the COUNT is what the model
    /// sizes its tables from, and `validate` pins the count to 3 (the
    /// trainer's hash tensor is `[b, t, 3]`).
    #[serde(default = "d_engram_orders")] pub engram_orders: Vec<usize>,
    /// Columns per memory row. One shared value projection over all orders
    /// (arXiv 2601.07372 Sec. 2.4: "a single sparse embedding table and a
    /// Value projection matrix W_V are shared across all M branches"; its
    /// eq. 6 is the branch GATE, not the sharing), so a row is a lookup, not
    /// a per-order model.
    #[serde(default = "d_engram_dim")] pub engram_dim: usize,
    /// HARD ceiling on the memory branch's share of the block output. The
    /// branch is `lam * memory + (1 - lam) * dense(normed)` with
    /// `lam = min(w_mem, engram_lam_max)`, so the backbone's share of that
    /// branch is never below `1 - engram_lam_max` - a floor, not a learned
    /// value (kNN-LM eq. 3, FwPKM eq. 12).
    #[serde(default = "d_engram_lam_max")] pub engram_lam_max: f32,
    /// MoR - Mixture-of-Recursions (arXiv 2507.10524) routing on the loop's
    /// iteration slots: per position, a shared linear router ranks the
    /// `max_iter` slots and the top `mor_k` of them feed the readout and the
    /// CE. Default OFF so the fixed-depth arm stays the default; it is
    /// mutually exclusive with the random-depth arm (`--rand-depth`), which
    /// buys the same depth robustness the other way.
    #[serde(default = "d_false")] pub use_mor: bool,
    /// Selected slots per position. Fixed capacity: the set always fills, and
    /// the floor of 1 recursion is structural (`mor::route` refuses k=0).
    #[serde(default = "d_mor_k")] pub mor_k: usize,
    /// Weight of the MoR BCE auxiliary, whose label is the router's own top-k
    /// recomputed on the current batch every step.
    #[serde(default = "d_mor_bce_weight")] pub mor_bce_weight: f32,
}

impl Default for DormouseConfig {
    /// The serde defaults above are the single written copy; Default (and
    /// anything building on it) just deserializes an empty table. Every
    /// field has a default, so this is infallible - the expect is a schema
    /// invariant, not a runtime path.
    fn default() -> Self {
        toml::from_str("").expect("DormouseConfig serde defaults are complete")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defaults mirror the `small` preset exactly (the comment on the
    /// default fns promises this); a drift in either direction fails here.
    #[test]
    fn default_equals_small_preset() {
        let d = DormouseConfig::default();
        let manifest = format!("{}/../../configs/small.toml", env!("CARGO_MANIFEST_DIR"));
        let s: DormouseConfig = toml::from_str(
            &std::fs::read_to_string(&manifest).expect("configs/small.toml exists"),
        )
        .expect("configs/small.toml parses");
        assert_eq!(d, s);
    }

    /// Missing fields in a flat TOML fall back to the schema defaults.
    #[test]
    fn partial_toml_fills_schema_defaults() {
        let c: DormouseConfig = toml::from_str("d_model = 512").unwrap();
        assert_eq!(c.d_model, 512);
        assert_eq!(c.vocab, 256);
        assert!(c.use_kda);
        assert_eq!(c.dspark_k, 4);
        assert_eq!(c.act_quant, None);
    }

    /// The memory-row count the `engram_rows` comment quotes. The comment
    /// carries three numbers and they disagreed; this pins the one that is
    /// pure arithmetic on the shipped default, so a change to `engram_rows`,
    /// `engram_dim` or the order count cannot leave the prose stale without
    /// failing here. The shares that comment also quotes come off the
    /// INSTANTIATED models (`cargo test -p dormouse-core --test preset_exec
    /// -- --nocapture`), which is where they are measured, not here.
    #[test]
    fn shipped_memory_rows_are_what_the_comment_says() {
        use crate::loop_block::engram_tables;
        let c = DormouseConfig::default();
        assert_eq!((c.engram_rows, c.engram_dim, c.engram_orders.len()), (25_000, 32, 3));
        let (tables, _mask) = engram_tables(c.engram_rows, c.engram_orders.len());
        // In-VRAM rounds UP to a power of two, so 25 000 buys 32 768 rows.
        assert_eq!(tables, vec![32_768; 3]);
        let mem: usize = tables.iter().sum::<usize>() * c.engram_dim;
        assert_eq!(mem, 3_145_728, "the comment's 3 145 728 / 3.1M");
        // And the nominal figure it must NOT be confused with: 2.4M is what
        // `engram_rows` says, before the rounding. Two different numbers for
        // the same table is the whole reason the old comment read as three.
        assert_eq!(c.engram_orders.len() * c.engram_rows * c.engram_dim, 2_400_000);
    }

    /// The one act-quant parser: strings, ints, and the error naming the input.
    #[test]
    fn act_quant_from_str_forms() {
        assert_eq!("fp4".parse::<ActQuant>(), Ok(ActQuant::Fp4));
        assert_eq!("4".parse::<ActQuant>(), Ok(ActQuant::Int(4)));
        assert_eq!("8".parse::<ActQuant>(), Ok(ActQuant::Int(8)));
        assert_eq!("FP4".parse::<ActQuant>(), Ok(ActQuant::Fp4));
        assert_eq!("int8".parse::<ActQuant>(), Ok(ActQuant::Int(8)));
        assert!("9".parse::<ActQuant>().is_err());
        assert!("nope".parse::<ActQuant>().is_err());
        // TOML integer values deserialize through the same parser.
        let c: DormouseConfig = toml::from_str("act_quant = 8").unwrap();
        assert_eq!(c.act_quant, Some(ActQuant::Int(8)));
    }
}

fn d_mhc_streams() -> usize { 2 }
fn d_moe_topk() -> usize { 0 }
fn d_moe_lb_coef() -> f32 { 0.0 }
