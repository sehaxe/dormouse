//! Diagnostic: does TSCT actually learn? Mini-GPT (d=64, 4 layers) on a
//! char-level task, dense vs SpectralLinear vs SpectralMoE, same params budget.
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![allow(deprecated)]
//! Attention is swappable via ATTN=dense|kda (dense MHA+RoPE is the default
//! baseline; kda = burn-kda linear attention - see AttnImpl).
//!
//! Run: cargo run -p burn-tsct --example tsct_diag
use burn::backend::{Backend, DispatchKindConversion};
use burn::module::{Module, ParamGroup};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::DispatchTensor;
use burn::tensor::{activation, Bool, Device, Distribution, Int, Tensor};
use burn_kda::{KdaConfig, KdaModule};
use burn_muon_plus::{MuonPlusConfig, NormDir};
use burn_optim::lr_scheduler::module_lr_scheduler::{ModuleLrScheduler, ModuleLrSchedulerConfig};
use burn_optim::GradientsParams;
use burn_spectral::{SpectralLinear, SpectralMoE};

#[cfg(feature = "cuda")]
type SctBackend = burn_cubecl::CubeBackend;
#[cfg(not(feature = "cuda"))]
type SctBackend = burn_ndarray::NdArray;

const D: usize = 64;
const VOCAB: usize = 64;
const SEQ: usize = 32;
const LAYERS: usize = 4;
const N_HEADS: usize = 4;

// real-text corpus: a small embedded English text (char-level, 63 chars)
const TEXT: &str = "to be or not to be that is the question whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune or to take arms against a sea of troubles and by opposing end them to die to sleep no more and by a sleep to say we end the heartache and the thousand natural shocks that flesh is heir to tis a consummation devoutly to be wished to die to sleep to sleep perchance to dream ay theres the rub for in that sleep of death what dreams may come when we have shuffled off this mortal coil must give us pause theres the respect that makes calamity of so long life for who would bear the whips and scorns of time the oppressors wrong the proud mans contumely the pangs of despised love the laws delay the insolence of office and the spurns that patient merit of the unworthy takes when he himself might his quietus make with a bare bodkin who would fardels bear to grunt and sweat under a weary life but that the dread of something after death the undiscovered country from whose bourn no traveller returns puzzles the will and makes us rather bear those ills we have than fly to others that we know not of thus conscience does make cowards of us all and thus the native hue of resolution is sicklied oer with the pale cast of thought and enterprises of great pith and moment with this regard their currents turn awry and lose the name of action";

fn make_corpus(n: usize) -> Vec<u8> {
    let chars: Vec<u8> = TEXT.bytes().map(|b| b % 63).collect();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(chars[i % chars.len()]);
    }
    out
}

/// Holdout from an env-var path: read as text, map to the 63-char set.
/// `None` when the env var is unset (caller falls back to embedded text).
fn corpus_from_env(key: &str) -> Option<Vec<u8>> {
    let path = std::env::var(key).ok()?;
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{key}={path}: cannot read holdout file: {e}"));
    Some(text.bytes().map(|b| b % 63).collect())
}

fn window(corpus: &[u8], i: usize, seq: usize) -> Vec<i64> {
    corpus[i..i + seq].iter().map(|&c| c as i64).collect()
}

/// One MLP layer (a, b) in one of several parameterizations.
#[derive(Module, Debug)]
struct Mixture {
    dense_a: Option<Linear>,
    dense_b: Option<Linear>,
    sct_a: Option<burn_sct::SctLinear>,
    sct_b: Option<burn_sct::SctLinear>,
    tsct_a: Option<SpectralLinear>,
    tsct_b: Option<SpectralLinear>,
    tsct_c: Option<SpectralLinear>,
    tsct_d: Option<SpectralLinear>,
    tsct_e: Option<SpectralLinear>,
    tsct_f: Option<SpectralLinear>,
    moe_a: Option<SpectralMoE>,
    moe_b: Option<SpectralMoE>,
    situ: bool,
    /// 2-bit-quantized dense (BitNet STE): dense_a/b weights are quantized
    /// on the forward path with weight_quant_2bit (masters stay fp32).
    bitnet: bool,
}

impl Mixture {
    fn new(kind: &str, d: usize, device: &Device) -> Self {
        match kind {
            "sct8" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: Some(burn_sct::SctLinear::new(
                    &burn_sct::SctConfig::new(d, 4 * d, 8),
                    device,
                )),
                sct_b: Some(burn_sct::SctLinear::new(
                    &burn_sct::SctConfig::new(4 * d, d, 8),
                    device,
                )),
                tsct_a: None,
                tsct_b: None,
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            "sct16" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: Some(burn_sct::SctLinear::new(
                    &burn_sct::SctConfig::new(d, 4 * d, 16),
                    device,
                )),
                sct_b: Some(burn_sct::SctLinear::new(
                    &burn_sct::SctConfig::new(4 * d, d, 16),
                    device,
                )),
                tsct_a: None,
                tsct_b: None,
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            "tsct_cascade" => Self {
                sct_a: None,
                sct_b: None,
                // cascade: rank-1 + rank-2 + rank-4 + rank-8 in parallel,
                // (1+2+4+8)*(64+256) = 4800 params vs dense 16K (3.4x less)
                dense_a: None,
                dense_b: None,
                tsct_a: Some(SpectralLinear::new(d, 4 * d, 1, device)),
                tsct_b: Some(SpectralLinear::new(4 * d, d, 1, device)),
                tsct_c: Some(SpectralLinear::new(d, 4 * d, 2, device)),
                tsct_d: Some(SpectralLinear::new(4 * d, d, 2, device)),
                tsct_e: Some(SpectralLinear::new(d, 4 * d, 4, device)),
                tsct_f: Some(SpectralLinear::new(4 * d, d, 4, device)),
                moe_a: Some(SpectralMoE::new(d, 4 * d, 8, 8, 2, 4, device)),
                moe_b: Some(SpectralMoE::new(4 * d, d, 8, 8, 2, 4, device)),
                situ: false,
                bitnet: false,
            },
            "dense" => Self {
                dense_a: Some(LinearConfig::new(d, 4 * d).init(device)),
                dense_b: Some(LinearConfig::new(4 * d, d).init(device)),
                sct_a: None,
                sct_b: None,
                tsct_a: None,
                tsct_b: None,
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            // BitNet 2-bit (the "cool dense" A/B): the plain dense MLP whose
            // weights are 2-bit quantized with straight-through gradients
            // (burn_bitnet::weight_quant_2bit). Same shapes/FLOPs/params as
            // dense; fp32 masters train, quantized only on the forward path.
            "bitnet2" => Self {
                dense_a: Some(LinearConfig::new(d, 4 * d).init(device)),
                dense_b: Some(LinearConfig::new(4 * d, d).init(device)),
                sct_a: None,
                sct_b: None,
                tsct_a: None,
                tsct_b: None,
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: true,
            },
            // SiTU-GLU FFN (burn-situ): gate+up of width 4*D need a
            // 2*(4*D) input, so dense_a is twice as wide; output is
            // [N, 4*D] like the dense branch, so dense_b is unchanged.
            "situ" => Self {
                dense_a: Some(LinearConfig::new(d, 2 * 4 * d).init(device)),
                dense_b: Some(LinearConfig::new(4 * d, d).init(device)),
                sct_a: None,
                sct_b: None,
                tsct_a: None,
                tsct_b: None,
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: true,
                bitnet: false,
            },
            "tsct4" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some(SpectralLinear::new(d, 4 * d, 4, device)),
                tsct_b: Some(SpectralLinear::new(4 * d, d, 4, device)),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            "tsct2_stoch" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some({
                    let mut l = SpectralLinear::new(d, 4 * d, 2, device);
                    l.set_stochastic(true);
                    l
                }),
                tsct_b: Some({
                    let mut l = SpectralLinear::new(4 * d, d, 2, device);
                    l.set_stochastic(true);
                    l
                }),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            "tsct2_percol" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some({
                    let mut l = SpectralLinear::new(d, 4 * d, 2, device);
                    l.set_per_column(true);
                    l
                }),
                tsct_b: Some({
                    let mut l = SpectralLinear::new(4 * d, d, 2, device);
                    l.set_per_column(true);
                    l
                }),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            "tsct2_combo2" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some({
                    let mut l = SpectralLinear::new(d, 4 * d, 2, device);
                    l.set_stochastic(true);
                    l.set_per_column(true);
                    l
                }),
                tsct_b: Some({
                    let mut l = SpectralLinear::new(4 * d, d, 2, device);
                    l.set_stochastic(true);
                    l.set_per_column(true);
                    l
                }),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            "tsct2_combo" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some({
                    let mut l = SpectralLinear::new(d, 4 * d, 2, device);
                    l.set_stochastic(true);
                    l.set_per_column(true);
                    l
                }),
                tsct_b: Some({
                    let mut l = SpectralLinear::new(4 * d, d, 2, device);
                    l.set_stochastic(true);
                    l.set_per_column(true);
                    l
                }),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            "tsct2" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some(SpectralLinear::new(d, 4 * d, 2, device)),
                tsct_b: Some(SpectralLinear::new(4 * d, d, 2, device)),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            // rank-2 2-bit: same factors, {-2,-1,0,1,2}*row-scale STE instead
            // of ternary on both U and V (quality A/B, weights stay fp32)
            "tsct2_2bit" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some({
                    let mut l = SpectralLinear::new(d, 4 * d, 2, device);
                    l.set_2bit(true);
                    l
                }),
                tsct_b: Some({
                    let mut l = SpectralLinear::new(4 * d, d, 2, device);
                    l.set_2bit(true);
                    l
                }),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            // asymmetric: only U is quantized, V stays fp32 (A/B which
            // factor carries the ternary)
            "tsct2_asym" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some({
                    let mut l = SpectralLinear::new(d, 4 * d, 2, device);
                    l.set_asym(true);
                    l
                }),
                tsct_b: Some({
                    let mut l = SpectralLinear::new(4 * d, d, 2, device);
                    l.set_asym(true);
                    l
                }),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            // tsct2 + master error feedback: after each optimizer step the
            // harness pulls the masters toward their ternary projections
            // (EFB_ETA, default 0.1) - the layer itself is plain tsct2
            "tsct2_efb" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some(SpectralLinear::new(d, 4 * d, 2, device)),
                tsct_b: Some(SpectralLinear::new(4 * d, d, 2, device)),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            "tsct1" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some(SpectralLinear::new(d, 4 * d, 1, device)),
                tsct_b: Some(SpectralLinear::new(4 * d, d, 1, device)),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            "tsct8" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some(SpectralLinear::new(d, 4 * d, 8, device)),
                tsct_b: Some(SpectralLinear::new(4 * d, d, 8, device)),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            // rank-8 2-bit: 5-level STE factors (quality A/B vs tsct8)
            "tsct8_2bit" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some({
                    let mut l = SpectralLinear::new(d, 4 * d, 8, device);
                    l.set_2bit(true);
                    l
                }),
                tsct_b: Some({
                    let mut l = SpectralLinear::new(4 * d, d, 8, device);
                    l.set_2bit(true);
                    l
                }),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            "tsct_anneal" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some(SpectralLinear::new(d, 4 * d, 16, device)),
                tsct_b: Some(SpectralLinear::new(4 * d, d, 16, device)),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            "tsct_dual" => Self {
                // complement: rank-2 + rank-8 in parallel (LOST-style)
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some(SpectralLinear::new(d, 4 * d, 2, device)),
                tsct_b: Some(SpectralLinear::new(4 * d, d, 2, device)),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: Some(SpectralMoE::new(d, 4 * d, 8, 8, 2, 4, device)),
                moe_b: Some(SpectralMoE::new(4 * d, d, 8, 8, 2, 4, device)),
                situ: false,
                bitnet: false,
            },
            "tsct_iso" => Self {
                // iso-param vs dense: rank 50 -> 50*(64+256) = 16K params
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some(SpectralLinear::new(d, 4 * d, 50, device)),
                tsct_b: Some(SpectralLinear::new(4 * d, d, 50, device)),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            "tsct32" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some(SpectralLinear::new(d, 4 * d, 32, device)),
                tsct_b: Some(SpectralLinear::new(4 * d, d, 32, device)),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            "tsct" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some(SpectralLinear::new(d, 4 * d, 16, device)),
                tsct_b: Some(SpectralLinear::new(4 * d, d, 16, device)),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            "tsctmoe" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: None,
                tsct_b: None,
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: Some(SpectralMoE::new(d, 4 * d, 8, 32, 2, 4, device)),
                moe_b: Some(SpectralMoE::new(4 * d, d, 8, 32, 2, 4, device)),
                situ: false,
                bitnet: false,
            },
            "tsctmoe_wide" => Self {
                // effective rank 32 (rank 8 x top-4) = same as tsct32, but
                // adaptive: 16 clusters x 32 experts, top-4 of 512 patterns
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: None,
                tsct_b: None,
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: Some(SpectralMoE::new(d, 4 * d, 16, 32, 4, 8, device)),
                moe_b: Some(SpectralMoE::new(4 * d, d, 16, 32, 4, 8, device)),
                situ: false,
                bitnet: false,
            },
            "tsctmoe_ec" => Self {
                // Expert-Choice routing: same config as tsctmoe, but each
                // expert picks its top tokens per cluster (every expert
                // trains every step; no [B, k*r, in/out] materialization).
                // The FLOPs/step print adds the EC top-k selection cost
                // (comparisons, SpectralMoE::topk_cost) on the same line.
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: None,
                tsct_b: None,
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: Some({
                    let mut m = SpectralMoE::new(d, 4 * d, 8, 32, 2, 4, device);
                    m.set_expert_choice(true);
                    m
                }),
                moe_b: Some({
                    let mut m = SpectralMoE::new(4 * d, d, 8, 32, 2, 4, device);
                    m.set_expert_choice(true);
                    m
                }),
                situ: false,
                bitnet: false,
            },
            // rank-scaled tsctmoe: same clusters/experts/topk as tsctmoe
            // (8, 32, 2), only the expert rank changes - d=512 probe for the
            // fixed-rank-4 weakness (ranks ~ d/8 = 64)
            "tsctmoe_r16" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: None,
                tsct_b: None,
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: Some(SpectralMoE::new(d, 4 * d, 8, 32, 2, 16, device)),
                moe_b: Some(SpectralMoE::new(4 * d, d, 8, 32, 2, 16, device)),
                situ: false,
                bitnet: false,
            },
            "tsctmoe_r32" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: None,
                tsct_b: None,
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: Some(SpectralMoE::new(d, 4 * d, 8, 32, 2, 32, device)),
                moe_b: Some(SpectralMoE::new(4 * d, d, 8, 32, 2, 32, device)),
                situ: false,
                bitnet: false,
            },
            "tsctmoe_r64" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: None,
                tsct_b: None,
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: Some(SpectralMoE::new(d, 4 * d, 8, 32, 2, 64, device)),
                moe_b: Some(SpectralMoE::new(4 * d, d, 8, 32, 2, 64, device)),
                situ: false,
                bitnet: false,
            },
            // iso-ish plain TSCT at d=512: rank 32 both factors (d/16)
            "tsct2_r32" => Self {
                dense_a: None,
                dense_b: None,
                sct_a: None,
                sct_b: None,
                tsct_a: Some(SpectralLinear::new(d, 4 * d, 32, device)),
                tsct_b: Some(SpectralLinear::new(4 * d, d, 32, device)),
                tsct_c: None,
                tsct_d: None,
                tsct_e: None,
                tsct_f: None,
                moe_a: None,
                moe_b: None,
                situ: false,
                bitnet: false,
            },
            _ => unreachable!(),
        }
    }

    fn fwd<B: Backend>(&self, x: Tensor<2>) -> Tensor<2>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let cascade = self.tsct_c.is_some();
        let both = self.tsct_a.is_some() && self.moe_a.is_some();
        let sct = self.sct_a.is_some();
        let h = if sct {
            activation::gelu(self.sct_a.as_ref().unwrap().forward::<B>(x.clone()))
        } else if cascade {
            let mut acc = self.tsct_a.as_ref().unwrap().forward(x.clone());
            acc = acc.add(self.tsct_c.as_ref().unwrap().forward(x.clone()));
            acc = acc.add(self.tsct_e.as_ref().unwrap().forward(x.clone()));
            acc = acc.add(self.moe_a.as_ref().unwrap().forward(x.clone()));
            activation::gelu(acc)
        } else if let Some(a) = &self.dense_a {
            // "dense" keeps GELU (baseline preserved); "situ" is a new
            // kind for A/B, implemented in the same branch.
            if self.situ {
                burn_situ::situ_glu(a.forward(x.clone()), a.weight.val().dims()[1] / 2, 1.0, 1.0)
            } else if self.bitnet {
                // 2-bit quantized dense (BitNet STE): the quantized weight is
                // computed inside the autodiff graph so grads reach the fp32
                // master; same FLOPs as dense (quant is elementwise, free per
                // the paper's framing, not counted in the flops() print).
                let w_q = burn_bitnet::weight_quant_2bit(a.weight.val());
                activation::gelu(burn::tensor::module::linear(
                    x.clone(),
                    w_q,
                    a.bias.as_ref().map(|b| b.val()),
                ))
            } else {
                activation::gelu(a.forward(x.clone()))
            }
        } else if both {
            // complement: rank-2 TSCT + 8 rank-1 MoE experts in parallel
            let t = self.tsct_a.as_ref().unwrap().forward(x.clone());
            let m = self.moe_a.as_ref().unwrap().forward(x.clone());
            activation::gelu(t.add(m))
        } else if let Some(a) = &self.tsct_a {
            activation::gelu(a.forward(x.clone()))
        } else {
            activation::gelu(self.moe_a.as_ref().unwrap().forward(x.clone()))
        };
        if sct {
            self.sct_b.as_ref().unwrap().forward::<B>(h)
        } else if cascade {
            let mut acc = self.tsct_b.as_ref().unwrap().forward(h.clone());
            acc = acc.add(self.tsct_d.as_ref().unwrap().forward(h.clone()));
            acc = acc.add(self.tsct_f.as_ref().unwrap().forward(h.clone()));
            acc = acc.add(self.moe_b.as_ref().unwrap().forward(h));
            acc
        } else if let Some(b) = &self.dense_b {
            b.forward(h)
        } else if both {
            let t = self.tsct_b.as_ref().unwrap().forward(h.clone());
            let m = self.moe_b.as_ref().unwrap().forward(h);
            t.add(m)
        } else if let Some(b) = &self.tsct_b {
            b.forward(h)
        } else {
            self.moe_b.as_ref().unwrap().forward(h)
        }
    }
}

/// Causal multi-head self-attention with RoPE, shared by every kind
/// (fairness: the harness itself must contain real sequence modeling).
#[derive(Module, Debug)]
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    rope: burn_rope::RotaryEmbedding,
}

impl Attention {
    fn new(d: usize, device: &Device) -> Self {
        Self {
            q: LinearConfig::new(d, d).init(device),
            k: LinearConfig::new(d, d).init(device),
            v: LinearConfig::new(d, d).init(device),
            o: LinearConfig::new(d, d).init(device),
            rope: burn_rope::RotaryEmbedding::new(d, N_HEADS, 1024, 10000.0, device),
        }
    }

    fn forward<B: Backend>(&self, x: Tensor<3>) -> Tensor<3>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let [b, t, d] = x.dims();
        let hd = d / N_HEADS;
        let q = self.q.forward(x.clone());
        let k = self.k.forward(x.clone());
        let v = self.v.forward(x.clone());
        let (q, k) = self.rope.forward_qk::<B>(q, k);
        let q = q.reshape([b, t, N_HEADS, hd]).swap_dims(1, 2);
        let k = k.reshape([b, t, N_HEADS, hd]).swap_dims(1, 2);
        let v = v.reshape([b, t, N_HEADS, hd]).swap_dims(1, 2);
        let scores = q.matmul(k.transpose()).div_scalar((hd as f64).sqrt());
        let mask = Tensor::<2, Bool>::tril_mask([t, t], 0, &x.device())
            .reshape([1, 1, t, t])
            .expand([b, N_HEADS, t, t]);
        let scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        let out = activation::softmax(scores, 3).matmul(v);
        self.o.forward(out.swap_dims(1, 2).reshape([b, t, d]))
    }
}

/// Attention implementation per the ATTN knob: dense MHA+RoPE (baseline,
/// byte-identical to the pre-knob harness) or burn-kda linear attention.
#[derive(Module, Debug)]
enum AttnImpl {
    Dense(Attention),
    Kda(KdaModule),
}

fn kda_module(d: usize, device: &Device) -> KdaModule {
    // K3-style: no short conv; chunked WY for training, exact per-token
    // recurrence for decode. min_decay = 0.0 -> fixed g_min = -5.
    let cfg = KdaConfig {
        hidden_size: d,
        num_heads: N_HEADS,
        head_dim: d / N_HEADS,
        use_short_conv: false,
        ..Default::default()
    };
    KdaModule::new(&cfg, 0.0, device)
}

impl AttnImpl {
    fn new(mode: &str, d: usize, _seq: usize, device: &Device) -> Self {
        match mode {
            "dense" => AttnImpl::Dense(Attention::new(d, device)),
            "kda" => AttnImpl::Kda(kda_module(d, device)),
            other => panic!("ATTN must be dense|kda, got {other}"),
        }
    }

    fn forward<B: Backend>(&self, x: Tensor<3>) -> Tensor<3>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        match self {
            AttnImpl::Dense(a) => a.forward::<B>(x),
            // fused autodiff op: the per-op chunk-WY autodiff yields NaN
            // grads under AdamW (delta-rule solve); the fused node has an
            // exact matrix-level backward (burn-gdn2 autodiff.rs)
            AttnImpl::Kda(k) => k.forward_train_fused::<burn_autodiff::Autodiff<SctBackend>>(x),
        }
    }
}

// Attention MACs per token per layer, counted from the actual ops:
//   dense:  q,k,v,o proj 4*d*d; scores+attn 2*NH*hd*seq per token
//   kda:    q,k,v proj 3*d*d; full-rank gate d*d; o_proj d*d;
//           decay w_up/w_down 2*d*hd (rank = hd); beta d*NH;
//           decode-loop state ops ~5*NH*hd*hd per token (decay apply,
//           erase k^T S + outer, write k v^T, read q^T S) - counted from
//           KdaModule::forward(update_state=true)
fn dense_attn_flops(d: usize, seq: usize) -> usize {
    4 * d * d + 2 * N_HEADS * (d / N_HEADS) * seq
}
fn kda_attn_flops(k: &KdaModule, d: usize) -> usize {
    let h = k.n_heads;
    let hd = k.head_dim;
    5 * d * d + 2 * d * hd + d * h + 5 * h * hd * hd
}
/// MACs per token per layer for the layer's attention impl.
fn attn_flops(mode: &AttnImpl, d: usize, seq: usize) -> usize {
    match mode {
        AttnImpl::Dense(_) => dense_attn_flops(d, seq),
        AttnImpl::Kda(k) => kda_attn_flops(k, d),
    }
}

/// NM env knob (Sparse-BitNet 2603.05168): "6:8" | "2:4" applies N:M
/// sparsity to every SpectralLinear factor; absent = off (default behavior
/// unchanged). Requires 0 < n <= m.
fn nm_knob() -> Option<(usize, usize)> {
    let v = std::env::var("NM").ok()?;
    let (n, m) = v.split_once(':')?;
    let (n, m): (usize, usize) = (n.parse().ok()?, m.parse().ok()?);
    assert!(
        n > 0 && m > 0 && n <= m,
        "NM={v} must be n:m with 0 < n <= m"
    );
    Some((n, m))
}

#[derive(Module, Debug)]
struct MiniGPT {
    emb: Embedding,
    heads: Vec<Mixture>,
    attns: Vec<AttnImpl>,
    lns: Vec<LayerNorm>,
    ln: LayerNorm,
    head: Linear,
}

impl MiniGPT {
    #[cfg(test)]
    fn new(kind: &str, device: &Device) -> Self {
        let attn = std::env::var("ATTN").unwrap_or_else(|_| "dense".into());
        Self::new_with_dim(kind, D, &attn, SEQ, device)
    }

    fn new_with_dim(kind: &str, d: usize, attn: &str, seq: usize, device: &Device) -> Self {
        // deep variants: same layer but 12/24 of them (spend the saved
        // weight budget on depth instead of width). Mapped here so
        // Mixture::new never sees a kind it can't build (tsct2_deep used
        // to fall through to `_ => unreachable!()` and panic).
        let (base, extra) = match kind {
            "tsct2_deep" => ("tsct2", 8),
            "tsct2_deeper" => ("tsct2", 20),
            "tsctmoe_deep" => ("tsctmoe", 20),
            "tsctmoe_mid" => ("tsctmoe", 8),
            other => (other, 0),
        };
        let n_layers = LAYERS + extra;
        let mut heads: Vec<Mixture> = (0..n_layers)
            .map(|_| Mixture::new(base, d, device))
            .collect();
        // NM knob: N:M sparsity on every SpectralLinear factor (see nm_knob)
        if let Some((n, m)) = nm_knob() {
            let mut count = 0usize;
            for h in &mut heads {
                for l in [
                    &mut h.tsct_a,
                    &mut h.tsct_b,
                    &mut h.tsct_c,
                    &mut h.tsct_d,
                    &mut h.tsct_e,
                    &mut h.tsct_f,
                ]
                .into_iter()
                .flatten()
                {
                    l.set_nm(n, m);
                    count += 1;
                }
            }
            println!("  nm {n}:{m} applied to {count} spectral factors");
        }
        Self {
            emb: EmbeddingConfig::new(VOCAB, d).init(device),
            heads,
            attns: (0..n_layers)
                .map(|_| AttnImpl::new(attn, d, seq, device))
                .collect(),
            lns: (0..2 * n_layers)
                .map(|_| LayerNormConfig::new(d).init(device))
                .collect(),
            ln: LayerNormConfig::new(d).init(device),
            head: LinearConfig::new(d, VOCAB).init(device),
        }
    }

    fn forward<B: Backend>(&self, ids: Tensor<2, Int>) -> Tensor<3>
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let [b, t] = ids.dims();
        let mut h = self.emb.forward(ids);
        let [_, _, d] = h.dims();
        for i in 0..self.heads.len() {
            // pre-norm attention (even ln) + residual
            let h_attn = self.lns[2 * i].forward(h.clone());
            h = h.add(self.attns[i].forward::<B>(h_attn));
            // pre-norm FFN (odd ln) + residual
            let flat = self.lns[2 * i + 1].forward(h.clone()).reshape([b * t, d]);
            let out = self.heads[i].fwd::<B>(flat);
            // BitNet v2: 4-bit Hadamard quantized residual (2504.18415),
            // gated by ACT=4 env (isolate its effect)
            let out3 = out.reshape([b, t, d]);
            if std::env::var("ACT").map(|v| v == "4").unwrap_or(false) {
                h = h + burn_bitnet::quantize_4bit(out3);
            } else {
                h = h + out3;
            }
        }
        let h = self.ln.forward(h);
        self.head
            .forward(h.reshape([b * t, d]))
            .reshape([b, t, VOCAB])
    }
}

fn make_val_corpus(n: usize) -> Vec<u8> {
    // hold-out: different segment of the same text (last 20% of chars,
    // cycled), so the model must generalize character statistics
    let chars: Vec<u8> = TEXT.bytes().map(|b| b % 63).collect();
    let seg = chars.len() * 4 / 5;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(chars[(seg + i) % chars.len()]);
    }
    out
}

/// Param path markers for the FFN LR group (substring match): the
/// spectral/ternary factors (tsct_*, moe_a/b, sct_a/b) and the dense MLP
/// weights (dense_a/b). Everything else (emb, attns, lns, ln, head) keeps
/// the base LR. Per-component LR is the SCT 2604.00733 fix for its ~3-loss
/// gap vs dense: FFN factors want a higher LR than the dense parts.
const FFN_PATH_MARKERS: [&str; 7] = [
    "dense_a", "dense_b", "sct_a", "sct_b", "tsct_", "moe_a", "moe_b",
];

/// True when a module param path belongs to the FFN group (spectral/
/// ternary/dense MLP factors), false for the dense group (embeddings,
/// attention, norms, head). Param paths are module field names joined by
/// ".", e.g. "heads.0.tsct_a.u".
#[cfg(test)]
fn is_ffn_param(path: &str) -> bool {
    FFN_PATH_MARKERS.iter().any(|m| path.contains(m))
}

fn ffn_param_group() -> ParamGroup {
    ParamGroup::from_any_predicates(FFN_PATH_MARKERS.to_vec())
}

/// Param path markers for the OPT_MIX Muon+ group: the 2D spectral masters
/// (tsct_*/sct_* u/v, moe_* u/v) and the MoE router linears (proj,
/// cluster_key, expert_key all live under moe_a/b). The 1D scale vector `s`
/// is excluded so it stays on the base AdamW (Muon+ would run its own AdamW
/// fallback on it, but the plan keeps every 1D param on the base optimizer).
/// Plain dense MLP weights (dense_a/b) are 2D but not spectral: AdamW is
/// fine for them and they stay in the fallback.
const MUON_PATH_MARKERS: [&str; 10] = [
    "tsct_a", "tsct_b", "tsct_c", "tsct_d", "tsct_e", "tsct_f", "sct_a", "sct_b", "moe_a", "moe_b",
];

/// True when a module param path belongs to the Muon+ group (2D spectral
/// factors + MoE router), false for the AdamW fallback. Mirrors
/// [`muon_param_group`] predicate-for-predicate.
#[cfg(test)]
fn is_muon_param(path: &str) -> bool {
    MUON_PATH_MARKERS.iter().any(|m| path.contains(m)) && !path.ends_with(".s")
}

fn muon_param_group() -> ParamGroup {
    // Exclude the 1D scale leaf (path ends in ".s"): keep it on AdamW. A
    // plain ".s" predicate would also hit "heads.3.sct_*" (module separator
    // dot before the s), so the exclude is end-anchored.
    ParamGroup::from_any_predicates(MUON_PATH_MARKERS.to_vec())
        .exclude(ParamGroup::from_regex(r"\.s$").expect("valid regex"))
}

fn train(kind: &str, steps: usize, lr: f64) -> (f32, f32) {
    let device = if std::env::var("DEVICE").as_deref() == Ok("cuda") {
        #[cfg(feature = "cuda")]
        {
            Device::cuda(0).autodiff()
        }
        #[cfg(not(feature = "cuda"))]
        {
            panic!("DEVICE=cuda requires --features cuda");
        }
    } else {
        Device::ndarray().autodiff()
    };
    // SEED env: fixed init + shuffle seed. A/B comparisons use 3 fixed
    // seeds and report mean±std.
    if let Ok(s) = std::env::var("SEED") {
        let seed: u64 = s.parse().expect("SEED must be an integer");
        device.seed(seed);
    }
    let batch: usize = std::env::var("BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let seq: usize = std::env::var("SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(SEQ);
    assert!(seq <= 1024, "SEQ={seq} exceeds the RoPE max_seq_len=1024");
    let d: usize = std::env::var("DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(D);
    // ATTN knob: dense (default) | kda - swaps the attention implementation
    // of every layer (see AttnImpl)
    let attn_mode = std::env::var("ATTN").unwrap_or_else(|_| "dense".into());
    // TRAIN_FILE/VAL_FILE: true holdout corpora (any text file; 63-char
    // set). Default mid-scale val source: /mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/datasets/wikipedia/books_00495.txt (6.3MB, exists).
    let corpus = corpus_from_env("TRAIN_FILE").unwrap_or_else(|| make_corpus(200_000));
    let val = corpus_from_env("VAL_FILE").unwrap_or_else(|| make_val_corpus(20_000));
    let mut model = MiniGPT::new_with_dim(kind, d, &attn_mode, seq, &device);
    // optimizer: OPT=muon = single Muon+ for all params; OPT=mix or
    // OPT_MIX=1 = Muon+ ColRow on the 2D spectral factors + MoE router,
    // AdamW on everything else (1D s/norms/emb/head, dense MLP weights);
    // default = plain AdamW. The mixed build starts from AdamW (which
    // matches ALL params as fallback, ParamGroup::all()) and adds the Muon+
    // group last, so last-match-wins routes the spectral 2D params to Muon+.
    let opt_env = std::env::var("OPT").unwrap_or_default();
    let use_muon = opt_env == "muon";
    let use_mix = opt_env == "mix" || std::env::var("OPT_MIX").map(|v| v == "1").unwrap_or(false);
    let muon_plus = || {
        MuonPlusConfig::new()
            .with_norm_dir(Some(NormDir::ColRow))
            .with_weight_decay(0.01)
    };
    let mut opt = if use_muon {
        muon_plus().init()
    } else if use_mix {
        burn_optim::AdamWConfig::new()
            .with_weight_decay(0.01)
            .init()
            .with_group(muon_param_group(), muon_plus().build(), None)
    } else {
        burn_optim::AdamWConfig::new()
            .with_weight_decay(0.01)
            .init()
    };
    if use_muon {
        println!("  opt: Muon+ ColRow (all params)");
    } else if use_mix {
        println!("  opt: AdamW (1D s, norms, emb, head, dense) + Muon+ ColRow (tsct_*/sct_*/moe_* u/v, MoE router)");
    } else {
        println!("  opt: AdamW (all params)");
    }
    // Per-component LR (SCT 2604.00733): LR_DENSE for attention/embeddings/
    // norms/head, LR_FFN for the spectral/ternary FFN factors. Both default
    // to LR, so behavior is unchanged when the knobs are unset (single-LR
    // step, identical to pre-change).
    let lr_dense: f64 = std::env::var("LR_DENSE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(lr);
    let lr_ffn: f64 = std::env::var("LR_FFN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(lr);
    let mut lr_sched: Option<ModuleLrScheduler> = if lr_dense != lr_ffn {
        Some(
            ModuleLrSchedulerConfig::new(lr_dense.into())
                .with_group(ffn_param_group(), lr_ffn)
                .init()
                .expect("constant-LR policy is always valid"),
        )
    } else {
        None
    };
    println!("  {kind}: lr: dense {lr_dense} / ffn {lr_ffn}");
    // training memory: masters + Adam m/v (3 fp32 tensors per param)
    if kind != "dense" && kind != "situ" && kind != "bitnet2" {
        let mut n = 0usize;
        for h in &model.heads {
            if let Some(a) = &h.tsct_a {
                n += a.param_count();
            }
            if let Some(b) = &h.tsct_b {
                n += b.param_count();
            }
            if let Some(a) = &h.moe_a {
                n += a.param_count();
            }
            if let Some(b) = &h.moe_b {
                n += b.param_count();
            }
        }
        println!(
            "  {kind}: trainable factors = {n} ({:.2} MB with Adam, 3x fp32)",
            n as f64 * 12.0 / 1e6
        );
    }

    let mut flops_token = 0usize;
    // EC top-k selection is not MACs: counted separately in comparisons
    // (one pass over [E, n_c] per expert pick, see SpectralMoE::topk_cost)
    let mut topk_per_step = 0usize;
    let rows = batch * seq; // rows the FFN/MoE actually sees per step
    for h in &model.heads {
        for l in [&h.dense_a, &h.dense_b].into_iter().flatten() {
            let d = l.weight.val().dims();
            flops_token += d[0] * d[1];
        }
        for l in [&h.sct_a, &h.sct_b].into_iter().flatten() {
            flops_token += l.rank * (l.in_features + l.out_features);
        }
        for l in [
            &h.tsct_a, &h.tsct_b, &h.tsct_c, &h.tsct_d, &h.tsct_e, &h.tsct_f,
        ]
        .into_iter()
        .flatten()
        {
            flops_token += l.flops();
        }
        for m in [&h.moe_a, &h.moe_b].into_iter().flatten() {
            // batch-aware: EC k_e depends on the rows the MoE sees
            flops_token += m.flops_n(rows);
            topk_per_step += m.topk_cost(rows);
        }
    }
    // attention cost per mode (see attn_flops for the counted formulas)
    let attn_flops = model
        .attns
        .iter()
        .map(|a| attn_flops(a, d, seq))
        .sum::<usize>();
    flops_token += attn_flops;
    let attn_note = format!("attn={attn_mode} {attn_flops} MACs/token");
    println!(
        "  {kind}: FLOPs/step = {} MACs ({} MACs/token over {} layers x {} tokens/step, {attn_note}){}",
        flops_token * rows + topk_per_step,
        flops_token,
        model.heads.len(),
        rows,
        if topk_per_step > 0 {
            format!(" +{topk_per_step} comparisons/step EC top-k")
        } else {
            String::new()
        }
    );

    #[cfg(feature = "cuda")]
    let (mut vram_in_use, mut vram_reserved) = (0u64, 0u64);
    #[cfg(feature = "cuda")]
    let on_cuda = std::env::var("DEVICE").as_deref() == Ok("cuda");

    // val on a bounded, deterministic sample of windows (full xv was
    // ~2062 forwards per call; with the EC forward that was hours per
    // curve point). VAL_N env knob: number of windows, default 24.
    let val_n: usize = std::env::var("VAL_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    let val_every: usize = std::env::var("VAL_EVERY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let nv = val.len() - seq;
    let val_loss = |model: &MiniGPT| -> f32 {
        let mut tot = 0.0f32;
        let mut cnt = 0usize;
        for i in (0..val_n).map(|i| (i * 97) % nv) {
            let xt = Tensor::<2, Int>::from_data(
                burn::tensor::TensorData::new(window(&val, i, seq), [1, seq]),
                &device,
            );
            let yt = Tensor::<2, Int>::from_data(
                burn::tensor::TensorData::new(window(&val, (i + 1) % nv, seq), [1, seq]),
                &device,
            );
            let logits = model.forward::<SctBackend>(xt);
            let lp = activation::log_softmax(logits.reshape([seq, VOCAB]), 1);
            let loss = -lp.gather(1, yt.reshape([seq]).unsqueeze_dim::<2>(1)).mean();
            tot += loss.into_scalar::<f32>();
            cnt += 1;
        }
        tot / cnt as f32
    };

    let mut first = 0.0f32;
    let mut curve = Vec::new();
    let n = corpus.len() - seq;
    let t_step = std::time::Instant::now();
    for step in 0..steps {
        let t0 = std::time::Instant::now();
        let i = (step * 17) % n;
        let mut xf = Vec::with_capacity(batch * seq);
        let mut yf = Vec::with_capacity(batch * seq);
        for j in 0..batch {
            xf.extend_from_slice(&window(&corpus, (i + j) % n, seq));
            yf.extend_from_slice(&window(&corpus, (i + j + 1) % n, seq));
        }
        let xt =
            Tensor::<2, Int>::from_data(burn::tensor::TensorData::new(xf, [batch, seq]), &device);
        let yt =
            Tensor::<2, Int>::from_data(burn::tensor::TensorData::new(yf, [batch, seq]), &device);
        let logits = model.forward::<SctBackend>(xt);
        let lp = activation::log_softmax(logits.reshape([batch * seq, VOCAB]), 1);
        let loss = -lp
            .gather(1, yt.reshape([batch * seq]).unsqueeze_dim::<2>(1))
            .mean();
        let lv: f32 = loss.clone().into_scalar();
        #[cfg(feature = "cuda")]
        if on_cuda {
            if let Some((iu, rv)) = vram_usage(&loss) {
                vram_in_use = vram_in_use.max(iu);
                vram_reserved = vram_reserved.max(rv);
            }
        }
        if step == 0 {
            println!("  [dbg] forward+loss ok {lv}");
            first = lv;
        }
        if step % val_every == 0 {
            curve.push(val_loss(&model));
        }
        let grads = loss.backward();
        if step == 0 {
            println!("  [dbg] backward ok");
        }
        let grads = GradientsParams::from_grads(grads, &model);
        model = if let Some(sched) = &mut lr_sched {
            opt.step(sched.step(), model, grads)
        } else {
            opt.step(lr, model, grads)
        };
        if step == 0 {
            println!("  [dbg] opt.step ok");
        }
        // master error feedback (LQE direction): pull each ternary master
        // toward its quantized value, so the master tracks the ternary the
        // forward actually used. Order: opt.step -> efb pull -> polar
        // retract every 50 (retract re-orthonormalizes AFTER the pull; at
        // eta=0.1 the pull barely breaks orthonormality). EFB_ETA knob,
        // default 0.1; applies to any kind ending in "_efb".
        if kind.ends_with("_efb") {
            let efb_eta: f32 = std::env::var("EFB_ETA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.1);
            for h in &mut model.heads {
                for t in [&mut h.tsct_a, &mut h.tsct_b].into_iter().flatten() {
                    // consume/from_mapped_value like retract: post-opt.step
                    // tensors are non-leaf (GradInBackward), from_tensor panics
                    let (id_u, u, map_u) = t.u.clone().consume();
                    let u_tern = burn_spectral::ternarize(u.clone());
                    t.u = burn::module::Param::from_mapped_value(
                        id_u,
                        u.clone().add(u_tern.sub(u).mul_scalar(efb_eta)),
                        map_u,
                    );
                    let (id_v, v, map_v) = t.v.clone().consume();
                    let v_tern = burn_spectral::ternarize(v.clone());
                    t.v = burn::module::Param::from_mapped_value(
                        id_v,
                        v.clone().add(v_tern.sub(v).mul_scalar(efb_eta)),
                        map_v,
                    );
                }
            }
        }
        // soft-to-hard annealing for the combo variant: alpha 0->1 over
        // the first 200 steps, then hard ternary (stochastic stays on)
        if kind == "tsct2_combo" {
            let alpha = (step as f32 / 200.0).clamp(0.0, 1.0);
            for h in &mut model.heads {
                for t in [&mut h.tsct_a, &mut h.tsct_b].into_iter().flatten() {
                    t.set_alpha(alpha);
                }
            }
        }
        // rank annealing: zero the tail of s (16 -> 8 -> 4 -> 2)
        if kind == "tsct_anneal" && step % 150 == 149 {
            for h in &mut model.heads {
                for t in [&mut h.tsct_a, &mut h.tsct_b].into_iter().flatten() {
                    let k = t.rank;
                    let keep = match step / 150 {
                        0 => 16,
                        1 => 8,
                        2 => 4,
                        _ => 2,
                    };
                    let dev = t.s.val().device();
                    let mut vals = vec![1.0f32; k];
                    for v in vals.iter_mut().skip(keep) {
                        *v = 0.0;
                    }
                    t.s = burn::module::Param::from_tensor(Tensor::<1>::from_data(
                        burn::tensor::TensorData::new(vals, [k]),
                        &dev,
                    ));
                }
            }
        }
        // polar retraction every 50 steps, gated by POLAR=1 (isolate)
        if std::env::var("POLAR").map(|v| v == "1").unwrap_or(false)
            && step % 50 == 49
            && kind.starts_with("tsct")
        {
            for h in &mut model.heads {
                if let Some(a) = &mut h.tsct_a {
                    a.retract(3);
                }
                if let Some(b) = &mut h.tsct_b {
                    b.retract(3);
                }
                if let Some(m) = &mut h.moe_a {
                    m.retract(3);
                }
                if let Some(m) = &mut h.moe_b {
                    m.retract(3);
                }
            }
        }
        // SCT: its own QR retraction (SCT requires it every step; here
        // every 50 steps to keep the comparison fair on step cost)
        if step % 50 == 49 && kind.starts_with("sct") {
            for h in &mut model.heads {
                if let Some(a) = &mut h.sct_a {
                    a.retract::<SctBackend>();
                }
                if let Some(b) = &mut h.sct_b {
                    b.retract::<SctBackend>();
                }
            }
        }
        if step % 50 == 0 {
            println!(
                "  {kind} step {step}: loss {lv:.4} (step {:.2}s, total {:.1}s)",
                t0.elapsed().as_secs_f64(),
                t_step.elapsed().as_secs_f64()
            );
        }
    }
    println!("  val curve (val windows: {val_n}, every {val_every} steps): {curve:?}");
    #[cfg(feature = "cuda")]
    if on_cuda {
        println!(
            "  {kind} VRAM peak: {:.1} MB reserved / {:.1} MB in use (gap {:.1} MB)",
            vram_reserved as f64 / 1e6,
            vram_in_use as f64 / 1e6,
            (vram_reserved - vram_in_use) as f64 / 1e6
        );
    }
    let result = (first, val_loss(&model));
    if std::env::var("GEN").map(|v| v == "1").unwrap_or(false) {
        let temp: f32 = std::env::var("GEN_TEMP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.8);
        let n: usize = std::env::var("GEN_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(600);
        let seed_text = std::env::var("GEN_SEED_TEXT").unwrap_or_else(|_| "the ".into());
        let seed: Vec<i64> = seed_text.bytes().map(|b| (b % 63) as i64).collect();
        let toks = generate::<SctBackend>(&model, &seed, temp, n, &device);
        let text: String = toks.iter().map(|&v| (v as u8 + 63) as char).collect();
        println!("[{kind} sample]\n{text}");
    }
    if std::env::var("PROBE").map(|v| v == "1").unwrap_or(false) {
        // one-time per-component timing at 16384 tokens; AFTER val_loss so
        // the per-step timing in the main loop stays undistorted
        let x = Tensor::<2>::random([16384, d], Distribution::Default, &device);
        let mut parts: Vec<String> = Vec::new();
        // (a) ste_ternary on the first layer's masters (tsct_a, else moe_a, else sct_a)
        let uv = model.heads[0]
            .tsct_a
            .as_ref()
            .map(|a| (a.u.val(), a.v.val()))
            .or_else(|| {
                model.heads[0]
                    .moe_a
                    .as_ref()
                    .map(|a| (a.u.val(), a.v.val()))
            })
            .or_else(|| {
                model.heads[0]
                    .sct_a
                    .as_ref()
                    .map(|a| (a.u.val(), a.v.val()))
            });
        if let Some((u, v)) = uv {
            let t0 = std::time::Instant::now();
            for _ in 0..3 {
                let _ = burn_spectral::ste_ternary(u.clone());
                let _ = burn_spectral::ste_ternary(v.clone());
            }
            parts.push(format!(
                "ste_ternary={:.2}",
                t0.elapsed().as_secs_f64() / 6.0 * 1e3
            ));
        }
        // (b) full first-layer forward
        let t0 = std::time::Instant::now();
        for _ in 0..3 {
            let _ = model.heads[0].fwd::<SctBackend>(x.clone());
        }
        parts.push(format!(
            "layer_fwd={:.2}",
            t0.elapsed().as_secs_f64() / 3.0 * 1e3
        ));
        // (c) MoE router
        if let Some(m) = &model.heads[0].moe_a {
            let t0 = std::time::Instant::now();
            for _ in 0..3 {
                let _ = m.router_logits(x.clone());
            }
            parts.push(format!(
                "router={:.2}",
                t0.elapsed().as_secs_f64() / 3.0 * 1e3
            ));
        }
        // (d) polar retract(3) on the first layer's spectral/moe masters
        if model.heads[0].tsct_a.is_some() || model.heads[0].moe_a.is_some() {
            let t0 = std::time::Instant::now();
            for _ in 0..3 {
                if let Some(a) = &mut model.heads[0].tsct_a {
                    a.retract(3);
                }
                if let Some(a) = &mut model.heads[0].moe_a {
                    a.retract(3);
                }
            }
            parts.push(format!(
                "retract3={:.2}",
                t0.elapsed().as_secs_f64() / 3.0 * 1e3
            ));
        }
        // (e) full optimizer step on 16384 tokens (one shot, ms total)
        let ids: Vec<i64> = (0..16384).map(|i| ((i * 7919) % VOCAB) as i64).collect();
        let xt =
            Tensor::<2, Int>::from_data(burn::tensor::TensorData::new(ids, [128, 128]), &device);
        let yt = Tensor::<1, Int>::zeros([16384], &device);
        let t0 = std::time::Instant::now();
        let logits = model.forward::<SctBackend>(xt);
        let lp = activation::log_softmax(logits.reshape([16384, VOCAB]), 1);
        let loss = -lp.gather(1, yt.unsqueeze_dim::<2>(1)).mean();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        let _ = if let Some(sched) = &mut lr_sched {
            opt.step(sched.step(), model, grads)
        } else {
            opt.step(lr, model, grads)
        };
        parts.push(format!("opt_step={:.2}", t0.elapsed().as_secs_f64() * 1e3));
        println!("  [probe] {kind} ms/call: {}", parts.join(" "));
    }
    result
}

/// Current CUDA allocator usage in bytes (in_use, reserved) from a live
/// tensor's client. `None` on non-CUDA backends (silent skip).
#[cfg(feature = "cuda")]
fn vram_usage<const R: usize>(t: &Tensor<R>) -> Option<(u64, u64)> {
    type CB = burn_cubecl::CubeBackend;
    let prim = t
        .clone()
        .try_into_primitive::<burn_autodiff::Autodiff<CB>>()
        .ok()?;
    let client = prim.primitive.client.clone();
    let m = client.memory_usage().ok()?;
    Some((m.bytes_in_use, m.bytes_reserved))
}

/// Greedy-free sampling: Gumbel-max over logits/temp (one tensor op per
/// token, no host roundtrips). Seed truncated to the last SEQ ids; the
/// window slides so the model only ever sees the last SEQ tokens.
fn generate<B: Backend>(
    model: &MiniGPT,
    seed: &[i64],
    temp: f32,
    n: usize,
    device: &Device,
) -> Vec<i64>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let seq: usize = std::env::var("SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(SEQ);
    let greedy = std::env::var("GEN_GREEDY")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut ids: Vec<i64> = seed.iter().rev().take(seq).copied().rev().collect();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut padded = vec![0i64; seq - ids.len()];
        padded.extend_from_slice(&ids);
        let xt =
            Tensor::<2, Int>::from_data(burn::tensor::TensorData::new(padded, [1, seq]), device);
        let logits = model
            .forward::<B>(xt)
            .narrow(1, seq - 1, 1)
            .reshape([VOCAB])
            .div_scalar(temp);
        let tok = if greedy {
            logits.clone().argmax(0).into_scalar::<i64>()
        } else {
            let u = Tensor::<1>::random([VOCAB], Distribution::Uniform(0.0, 1.0), device)
                .clamp(f32::MIN_POSITIVE, 1.0); // Gumbel u in (0, 1]
            let g = u.log().neg().log().neg(); // -log(-log u)
            logits.clone().add(g).argmax(0).into_scalar::<i64>()
        };
        // burn-autodiff checkpoints every Linear's input until backward
        // (checkpoint/builder.rs Computed action holds the buffer handle),
        // so a generation loop without backward leaks the whole forward
        // per token: d=256 crashes ~300 tokens (VRAM), d=64 slows
        // quadratically (SlicedPool::try_reserve scans all pages). The
        // sample is already read; backward here is pure cleanup (~1x fwd).
        let _ = logits.sum().backward();
        out.push(tok);
        ids.push(tok);
        ids.drain(0..ids.len().saturating_sub(seq));
    }
    out
}

fn main() {
    let steps: usize = std::env::var("STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(600);
    let lr = std::env::var("LR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-3);
    for kind in std::env::var("KINDS")
        .map(|v| v.split(',').map(|s| s.to_string()).collect::<Vec<_>>())
        .unwrap_or_else(|_| vec!["dense".into(), "tsctmoe".into(), "tsctmoe_wide".into()])
    {
        println!("=== {kind} (lr={lr}) ===");
        let (f, l) = train(&kind, steps, lr);
        println!("{kind}: {f:.4} -> {l:.4}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // burn-ndarray's RNG is process-global (static SEED), so parallel
    // tests interleave each other's random draws. Serialize them here or
    // the seeded reproducibility check is racy.
    static RNG: Mutex<()> = Mutex::new(());

    fn lock_rng() -> std::sync::MutexGuard<'static, ()> {
        RNG.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn attention_shapes() {
        let _g = lock_rng();
        let d = dev();
        let model = MiniGPT::new("dense", &d);
        let ids = Tensor::<2, Int>::zeros([1, SEQ], &d);
        let out = model.forward::<SctBackend>(ids);
        assert_eq!(out.dims(), [1, SEQ, VOCAB]);
    }

    #[test]
    fn situ_kind_shapes() {
        let _g = lock_rng();
        let d = dev();
        let model = MiniGPT::new("situ", &d);
        let ids = Tensor::<2, Int>::zeros([1, SEQ], &d);
        let out = model.forward::<SctBackend>(ids);
        assert_eq!(out.dims(), [1, SEQ, VOCAB]);
    }

    #[test]
    fn bitnet2_kind_shapes() {
        let _g = lock_rng();
        let d = dev();
        let model = MiniGPT::new("bitnet2", &d);
        let ids = Tensor::<2, Int>::zeros([1, SEQ], &d);
        let out = model.forward::<SctBackend>(ids);
        assert_eq!(out.dims(), [1, SEQ, VOCAB]);
    }

    #[test]
    fn two_bit_kind_shapes() {
        let _g = lock_rng();
        let d = dev();
        for kind in ["tsct2_2bit", "tsct8_2bit"] {
            let model = MiniGPT::new(kind, &d);
            let ids = Tensor::<2, Int>::zeros([1, SEQ], &d);
            let out = model.forward::<SctBackend>(ids);
            assert_eq!(out.dims(), [1, SEQ, VOCAB], "{kind} shape");
        }
    }

    #[test]
    fn asym_kind_shapes() {
        let _g = lock_rng();
        let d = dev();
        let model = MiniGPT::new("tsct2_asym", &d);
        let ids = Tensor::<2, Int>::zeros([1, SEQ], &d);
        let out = model.forward::<SctBackend>(ids);
        assert_eq!(out.dims(), [1, SEQ, VOCAB]);
    }

    #[test]
    fn efb_kind_shapes() {
        let _g = lock_rng();
        let d = dev();
        let model = MiniGPT::new("tsct2_efb", &d);
        let ids = Tensor::<2, Int>::zeros([1, SEQ], &d);
        let out = model.forward::<SctBackend>(ids);
        assert_eq!(out.dims(), [1, SEQ, VOCAB]);
    }

    #[test]
    fn attn_modes_shapes() {
        // every ATTN mode keeps [B, T, VOCAB] through the full model
        let _g = lock_rng();
        // autodiff device: the kda arm runs the fused autodiff op
        let d = Device::ndarray().autodiff();
        for mode in ["dense", "kda"] {
            let model = MiniGPT::new_with_dim("dense", D, mode, SEQ, &d);
            let ids = Tensor::<2, Int>::zeros([1, SEQ], &d);
            let out = model.forward::<SctBackend>(ids);
            assert_eq!(out.dims(), [1, SEQ, VOCAB]);
        }
    }

    #[test]
    fn seeded_reproducible() {
        let _g = lock_rng();
        std::env::set_var("SEED", "42");
        // train returns (first, last); step-0 losses must be bit-identical
        // under a fixed seed (same init, same first batch, same loss).
        let (f1, _) = train("dense", 2, 1e-3);
        let (f2, _) = train("dense", 2, 1e-3);
        assert_eq!(f1, f2, "SEED=42 must reproduce the step-0 loss exactly");
    }

    #[test]
    fn ffn_lr_group_assignment() {
        // FFN group: spectral/ternary factors and dense MLP weights.
        for p in [
            "heads.0.tsct_a.u",
            "heads.0.tsct_f.s",
            "heads.1.moe_a.u",
            "heads.2.moe_b.v",
            "heads.3.sct_a.weight",
            "heads.3.sct_b.weight",
            "heads.0.dense_a.weight",
            "heads.1.dense_b.weight",
            // bitnet2 is a quantized dense: same dense_a/dense_b paths, so its
            // params land in the FFN group via the dense_a/dense_b markers.
            "heads.0.dense_a.weight",
            "heads.2.dense_b.weight",
        ] {
            assert!(is_ffn_param(p), "{p} must be FFN-LR");
        }
        // Dense group: embeddings, attention, norms, head.
        for p in [
            "emb.weight",
            "attns.0.q.weight",
            "attns.0.o.weight",
            "attns.0.gate.proj.weight",
            "attns.0.kda.q_proj.weight",
            "attns.1.kda.v_proj.weight",
            "lns.0.weight",
            "lns.0.bias",
            "ln.weight",
            "head.weight",
        ] {
            assert!(!is_ffn_param(p), "{p} must be dense-LR");
        }
    }

    #[test]
    fn opt_mix_group_assignment() {
        // OPT_MIX groups: Muon+ for the 2D spectral masters and the MoE
        // router linears; AdamW for everything else (1D scale vectors,
        // embeddings, attention, norms, head, plain dense MLP weights).
        for p in [
            "heads.0.tsct_a.u",
            "heads.0.tsct_f.v",
            "heads.1.moe_a.u",
            "heads.2.moe_b.v",
            "heads.3.sct_a.u",
            "heads.3.sct_b.v",
            // MoE router linears are 2D and live under moe_a/b
            "heads.0.moe_a.proj.weight",
            "heads.1.moe_b.cluster_key.weight",
            "heads.2.moe_a.expert_key.weight",
        ] {
            assert!(is_muon_param(p), "{p} must be Muon+");
        }
        for p in [
            // 1D scale vectors: excluded from the Muon+ group -> AdamW
            "heads.0.tsct_a.s",
            "heads.0.tsct_f.s",
            "heads.1.moe_a.s",
            "heads.3.sct_b.s",
            // dense MLP weights, embeddings, attention, norms, head
            "heads.0.dense_a.weight",
            "heads.1.dense_b.weight",
            "emb.weight",
            "attns.0.q.weight",
            "attns.0.o.weight",
            "lns.0.weight",
            "ln.weight",
            "head.weight",
        ] {
            assert!(!is_muon_param(p), "{p} must be AdamW");
        }
    }
}
