//! The one flat model schema (ADR-0005): every default is written exactly
//! once, in the per-field serde defaults below; `Default` derives from them
//! by deserializing an empty table. There is no sectioned mirror - preset
//! TOMLs are flat tables of these same fields.

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::{self, Visitor}};
use std::str::FromStr;
use crate::act_quant::ActFormat;

/// Activation quantization for the FFN branch (BitNet a4.8 style, straight
/// through the estimator: the graph stays f32, the VALUES are quantized at
/// `core/src/act_quant.rs`). Not a weight format - `--quant` is that.
///
/// One parser for the whole project ([`FromStr`], below), so a preset TOML,
/// `--set act_quant=…` and `--act-quant …` cannot disagree about what a
/// spelling means (ADR-0005).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActQuant {
    /// FP4 in the REAL e2m1 grid: magnitudes `{0, .5, 1, 1.5, 2, 3, 4, 6}`,
    /// 16 codes, and the block scale maps onto the FORMAT's max (6) rather
    /// than onto 1.
    ///
    /// Until 2026-09-28 this variant was not e2m1 at all: the mantissa rule
    /// emitted 0.75, which is not a level of the format, and the caller
    /// scaled each block's max onto 1. Only `{0, 0.5, 0.75, 1}` of the eight
    /// magnitudes were reachable - a ~3-level quantizer wearing a 4-bit label.
    /// **Every `--act-quant fp4` number before `9b343d3` is invalid**; it
    /// measured the 3-level thing.
    Fp4,
    /// Uniform INT with `b` bits, symmetric range `-2^(b-1) … 2^(b-1)-1`. The
    /// shipped values are 4 and 8, and both are bit-identical across the whole
    /// e2m1 history (`int4_levels_are_the_whole_symmetric_range` pins that).
    ///
    /// THE ATTENTION PATH IS UPGRADED UNCONDITIONALLY, and that is
    /// load-bearing, not an oversight: `ActFormat::attn()` maps `Fp4 -> Int(8)`
    /// and `Int(b) -> Int(b.max(8))`. So `--act-quant fp4` has never run
    /// 4-bit attention, and no number claiming otherwise is a measurement of
    /// the FFN path alone.
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

/// The whole model configuration: one flat table, no sections (ADR-0005).
///
/// Every default is written EXACTLY ONCE, in the `d_*` serde functions at the
/// top of this file; [`Default`] derives from them by deserializing an empty
/// table, and those defaults are `small`, so
/// `DormouseConfig::default() == load_config("small")` is a test
/// (`default_equals_small_preset`). A preset TOML is a flat table of these same
/// fields, which is why there is no "model section" anywhere.
///
/// # Reading the fields
///
/// The WIDTHS (`d_model`, `n_heads`, `head_dim`, `d_ffn`, `vocab`,
/// `max_seq_len`, `rank`, `norm_eps`, `n_experts`) are the model's geometry and
/// need no per-field note: the names are the geometry, and the constraints
/// between them are [`super::validate`]'s job, not this struct's. The fields
/// that DO carry a doc are the ones whose VALUE is a decision - an arm, a
/// budget, or a convention someone could get wrong - and those are the ones a
/// reader of a preset is actually asking about.
///
/// # What this struct is not responsible for
///
/// Coherence (`super::validate`), the merge order (`dormouse_train::resolve`),
/// and the run-level knobs that are not the model's - learning rate, batch,
/// checkpoint cadence - which live in `dormouse_train::TrainCfg`. A field here
/// changes what the network IS; a field there changes how it is trained.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DormouseConfig {
    /// bf16 STORAGE, fp32 compute. On this box that is a cast-copy per forward
    /// and no bf16 tensor cores (ADR-0016: the LLVM dialect has no bf16 type),
    /// so `--bf16` is SLOWER than fp32 here. Off by default; a CPU run must
    /// leave it off (burn-flex has no bf16 kernel to run).
    #[serde(default = "d_bf16")] pub bf16: bool,
    /// Activation-quant format, or `None` for none. See [`ActQuant`] for the
    /// two forms and for the unconditional attention upgrade that makes
    /// `Fp4` a statement about the FFN path only.
    #[serde(default)] pub act_quant: Option<ActQuant>,
    /// Columns per quantizer scale group. `0` = one scale per token. Otherwise
    /// it MUST divide `d_model`, because the quantizer reshapes `[b*t, d]` to
    /// `[b, d/g, g]` and a non-divisor fails the first forward - checked by
    /// [`super::validate`], which says so in the error.
    #[serde(default = "d_act_group")] pub act_group: usize,
    /// The residual stream's width: the embedding, `lm_head`, every norm and
    /// every arm's input and output. The single number the parameter budget is
    /// argued from, and the one the VRAM estimate is `params x 4 B` of.
    ///
    /// Must be divisible by [`DormouseConfig::n_heads`]
    /// ([`super::validate`]) and, when the arms that reshape it are on, by
    /// `act_group` and `mhc_streams`.
    #[serde(default = "d_d_model")] pub d_model: usize,
    /// Attention heads. `d_model / n_heads` is the head width, so this is a
    /// SHAPE factor rather than a capacity one: the total attention width is
    /// `d_model` either way, and the split decides the softmax granularity.
    #[serde(default = "d_n_heads")] pub n_heads: usize,
    /// Width of ONE head, when it is set independently. The trainer
    /// re-derives head-wise Muon routing from `n_heads`
    /// (`dormouse_train::cfg`), so changing it without changing `n_heads`
    /// moves the optimizer's grouping and nothing else - which is why the
    /// presets set both or neither.
    #[serde(default = "d_head_dim")] pub head_dim: usize,
    /// FFN inner width. Doubled under `use_situ` (that arm's Eq. 12 reads
    /// `Wg x` and `Wu x` separately, so the projection becomes `d -> 2f`).
    #[serde(default = "d_d_ffn")] pub d_ffn: usize,
    /// **256, fixed by the byte-level task, not a tuning knob.** It is a field
    /// because `generate` and `serve` need to read it out of a config; a
    /// reader who sees a `vocab` field assumes it is swept, and it is not.
    #[serde(default = "d_vocab")] pub vocab: usize,
    /// Longest sequence the model will take. The decode seam stops when the
    /// context passes it, and `--seq-len` is a TRAINING-batch length: a run
    /// whose `seq_len` exceeds this is training on sequences the model will
    /// refuse to serve.
    #[serde(default = "d_max_seq_len")] pub max_seq_len: usize,
    /// THE LOOP DEPTH: how many times the weight-shared block runs. There is
    /// no halt head and no PonderNet - depth is fixed, and the per-iteration CE
    /// is an unweighted mean (ADR-0013, the KL machinery is deleted). Two arms
    /// buy depth robustness and both are OFF: `use_mor` (rank the slots per
    /// position) and `--rand-depth` (sample `T` per step), mutually exclusive
    /// and refused together.
    #[serde(default = "d_max_iter")] pub max_iter: usize,
    /// Inner rank of the TSCT low-rank factors (`d x r` and `r x out`). The
    /// cheapest capacity knob in the schema and the reason the retraction is
    /// affordable: orthogonality is maintained in the FACTORED form, so fp32
    /// Newton-Schulz over `[768, 768]` (~40 s/step, the reason Muon+ stays off
    /// every `[d,d]` projection) becomes milliseconds over `[d, 64]` and
    /// `[64, f]`.
    #[serde(default = "d_rank")] pub rank: usize,
    /// Epsilon inside every RMSNorm. Small and fixed by convention (1e-3 here,
    /// not 1e-5 or 1e-6): it is the only thing standing between a zero-norm row
    /// and a division by zero, and a larger epsilon is a slightly less peaked
    /// normal at a cost nobody here has measured. Must be `> 0`.
    #[serde(default = "d_norm_eps")] pub norm_eps: f32,
    /// The gated-delta (KDA) attention arm. The ONLY attention arm: the MSA
    /// top-k sparse arm and its crate were cut (ADR-0014). On by default; the
    /// bisect flag is `--no-kda` (an A/B, not a preset).
    ///
    /// Whether it actually TRAINS is a separate question from whether it runs,
    /// and it is a real one: for the whole history of this project it ran
    /// thousands of forwards with no backward (`8fa5d4c`), and the counter
    /// `fused kda=<fwd>/<bwd>` on the eval line is how a reader tells the two
    /// apart. `0/0` means the fused path is not on this backend, NOT that the
    /// arm is slow.
    #[serde(default = "d_true")] pub use_kda: bool,
    /// Spectral (low-rank TSCT) linears vs plain dense ones. The spectral
    /// path is what makes the FFN 2048-wide on `small` (9 197 390 params,
    /// MEASURED on the instantiated model - the 7.5M in older comments here
    /// predates the 2026-09-27 memory re-pricing and is retracted); turning it
    /// off gives the A/B that decides whether it earns its retraction, quant
    /// machinery and ~1000 lines (at a matched param budget the FFN gets
    /// narrower, which is the comparison that actually means something).
    #[serde(default = "d_true")] pub use_tsct: bool,
    /// The hashed n-gram memory arm (FNV tables over 2/3/4-grams as input
    /// features). On by default; the capacity budget is [`DormouseConfig::engram_rows`],
    /// which is 34.2% of `small`'s parameters. The bisect flag is
    /// `--no-engram`.
    ///
    /// THE FIELD THAT OTHERS GET WRONG: with this on and the memory held in
    /// host RAM (`--engram-ram`), the in-model tables are never read and are
    /// squeezed to one row per order, so a model exported from such a run
    /// returns THE SAME ROW for every key - a memory arm that is constant and
    /// was never trained. `dormouse_train::decode::refuse_unservable_memory`
    /// is the LOUD check for it, and `generate`/`serve` call it at startup.
    #[serde(default = "d_true")] pub use_engram: bool,
    /// Gated Residual (Qwen3.8-Flash-Next §3.5, Eq. 31-34) in place of the
    /// ReZero scale. Off by default for checkpoint compatibility, and mutually
    /// exclusive with [`DormouseConfig::use_attnres`] and
    /// [`DormouseConfig::use_mhc`]: all three REPLACE the loop's residual
    /// accumulation, so only one can run. Never A/B'd - the report's -0.026
    /// does not transfer to one GR per loop ITERATION of a weight-shared block.
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
    /// Manifold-Constrained Hyper-Connections: a learned, per-layer rate
    /// between the residual stream and the block body, replacing ReZero.
    /// Off by default, mutually exclusive with `use_gr` / `use_attnres`
    /// (same replacement, one statement of the loop), and unreachable from
    /// `--set` - a preset TOML is currently the only way to turn it on
    /// (`config::override`'s module docs name this).
    #[serde(default)] pub use_mhc: bool,
    /// How many streams the residual is split into. Must divide `d_model` or
    /// the first forward's reshape fails, which is why
    /// [`super::validate`] checks it with the divisor in the message. `2` is
    /// the K the phi study ran; the base paper's own `n = 4` is
    /// `--set mhc_streams=4` - except that this field is not in the `--set`
    /// match, so today it is a preset edit.
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
    /// Experts selected per position by the sparse router. **`0` is the dense
    /// blend, not "no experts"** - the run trains a dense MoE and the log
    /// still prints `moe=...`. A value above `n_experts` is REFUSED rather
    /// than clamped, because clamping is exactly that silent collapse.
    #[serde(default = "d_moe_topk")] pub moe_topk: usize,
    /// Load-balancing coefficient (Switch §3.3). A POSITIVE value with
    /// `moe_topk = 0` is refused: there is no selection to balance, so the
    /// term would be ignored while the config reads like an objective. The
    /// combination "routing on, balancer off" is deliberately LEGAL - it is the
    /// arm's own removal under the A/B rule, and `probe::MOE_LB` reads `0` on
    /// the eval line so the state is visible.
    #[serde(default = "d_moe_lb_coef")] pub moe_lb_coef: f32,
    /// Size of the expert bank, each member a low-rank TSCT FFN. This is the
    /// BANK, not the selection: how many of them a position uses is
    /// [`DormouseConfig::moe_topk`], and with `moe_topk = 0` (the default)
    /// every expert is built and the controller blends them densely. A bank
    /// with nothing selecting from it is a flat cost, which is why the MoE arm
    /// ships off.
    #[serde(default = "d_n_experts")] pub n_experts: usize,
    /// JEPA weight on top of CE: masked latent prediction against an EMA
    /// teacher, plus KoLeo. `0` turns the term off; the head still costs a
    /// second full forward per step, which is what OOMs `base` at batch 6.
    /// ON by default, and the A/B that would justify it is queue row 1
    /// (pure CE).
    #[serde(default = "d_jepa_weight")] pub jepa_weight: f32,
    /// Fraction of positions the JEPA mask hides. The span is built from
    /// `(seed, step)` and nothing else, because burn's global RNG is never
    /// seeded - a sampled mask would make two runs of one config different
    /// experiments (ADR-0021).
    #[serde(default = "d_jepa_mask_frac")] pub jepa_mask_frac: f32,
    /// Span length of the JEPA mask in BYTE POSITIONS.
    #[serde(default = "d_jepa_mask_span")] pub jepa_mask_span: usize,
    /// DSpark weight: a DeepSeek-style draft head that corrects the frozen
    /// logits into the next K bytes, used INSTEAD of MTP. **`0.0` in the schema
    /// and in every preset** - it ships off.
    ///
    /// No DSpark number in this project's history survives 2026-09-29: the
    /// window was built from the LABEL sequence, so step `s` was fed the very
    /// byte `logits[p+s]` had just predicted, and the head was handed the
    /// answer one step early. The next reading is a first measurement, not a
    /// continuation.
    #[serde(default = "d_dspark_weight")] pub dspark_weight: f32,
    /// Draft depth K, in bytes.
    #[serde(default = "d_dspark_k")] pub dspark_k: usize,
    /// ANCHOR SPACING of the DSpark window, in byte positions - a sampling
    /// density, not a token count and not a fraction. Windows start at
    /// `p = 0, stride, 2*stride, …`, so the default `16` covers ~6% of
    /// positions and `1` covers every one. `0` is REFUSED: it would put every
    /// anchor on position 0 and collapse the K-step window to K copies of one
    /// cross-entropy. Never A/B'd against the paper's random anchor sampling -
    /// "deterministic and documented" is the claim, not "equivalent".
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
    /// over compressed tokens (V4.1-Flash n in {2,3,4}, Engram-27B \[2,3\])
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
