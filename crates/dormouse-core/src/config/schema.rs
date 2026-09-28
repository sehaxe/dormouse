//! The one flat model schema (ADR-0005): every default is written exactly
//! once, in the per-field serde defaults below; `Default` derives from them
//! by deserializing an empty table. There is no sectioned mirror - preset
//! TOMLs are flat tables of these same fields.

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::{self, Visitor}};
use std::str::FromStr;
use crate::act_quant::ActFormat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActQuant { Fp4, Int(u32) }
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
fn d_dspark_weight() -> f32 { 0.1 }
fn d_dspark_k() -> usize { 4 }
fn d_dspark_stride() -> usize { 16 }
fn d_engram_rows() -> usize { 25_000 }
fn d_engram_orders() -> Vec<usize> { vec![2, 3, 4] }
fn d_engram_dim() -> usize { 32 }
fn d_engram_lam_max() -> f32 { 0.5 }
fn d_false() -> bool { false }
fn d_mor_k() -> usize { 2 }
fn d_mor_bce_weight() -> f32 { 0.05 }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DormouseConfig {
    #[serde(default = "d_bf16")] pub bf16: bool,
    #[serde(default)] pub act_quant: Option<ActQuant>,
    #[serde(default = "d_act_group")] pub act_group: usize,
    #[serde(default = "d_d_model")] pub d_model: usize,
    #[serde(default = "d_n_heads")] pub n_heads: usize,
    #[serde(default = "d_head_dim")] pub head_dim: usize,
    #[serde(default = "d_d_ffn")] pub d_ffn: usize,
    #[serde(default = "d_vocab")] pub vocab: usize,
    #[serde(default = "d_max_seq_len")] pub max_seq_len: usize,
    #[serde(default = "d_max_iter")] pub max_iter: usize,
    #[serde(default = "d_rank")] pub rank: usize,
    #[serde(default = "d_norm_eps")] pub norm_eps: f32,
    #[serde(default = "d_true")] pub use_kda: bool,
    /// Spectral (low-rank TSCT) linears vs plain dense ones. The spectral
    /// path is what makes the FFN 2048-wide on 7.5M params; turning it off
    /// gives the A/B that decides whether it earns its retraction, quant
    /// machinery and ~1000 lines (at a matched param budget the FFN gets
    /// narrower, which is the comparison that actually means something).
    #[serde(default = "d_true")] pub use_tsct: bool,
    #[serde(default = "d_true")] pub use_engram: bool,
    #[serde(default)] pub use_gr: bool,
    #[serde(default = "d_n_experts")] pub n_experts: usize,
    #[serde(default = "d_jepa_weight")] pub jepa_weight: f32,
    #[serde(default = "d_jepa_mask_frac")] pub jepa_mask_frac: f32,
    #[serde(default = "d_jepa_mask_span")] pub jepa_mask_span: usize,
    #[serde(default = "d_dspark_weight")] pub dspark_weight: f32,
    #[serde(default = "d_dspark_k")] pub dspark_k: usize,
    #[serde(default = "d_dspark_stride")] pub dspark_stride: usize,

    // --- hashed n-gram memory (Engram): the arm's capacity budget ---
    /// Rows per n-gram order. A CAPACITY BUDGET, not a tuning knob: the arm
    /// was switched off on 2026-09-27 because at 8M rows/order it was 99% of
    /// the model (768M memory params against a 7.5M backbone) and
    /// monopolized the loss (rec -> 0.005, held-out frozen at exactly
    /// uniform 8.000 BPB). Two pieces of evidence set this number, and they
    /// disagree at our scale - so the smaller one wins and the disagreement
    /// is written down rather than averaged away:
    ///
    /// 1. The measured saturation curve at iso-parameter (arXiv 2601.16531:
    ///    125M backbone, 128M Engram, slots per order): Hash-300K 4.4825 /
    ///    Hash-500K **4.4809** / Hash-800K 4.4961 - 500K is the optimum, 800K
    ///    is ~2 sigma WORSE, and 8M was 16x past the knee. That curve is FLAT
    ///    from 300K to 500K (delta 0.0016 against sigma 0.008-0.012) and says
    ///    nothing below 300K.
    /// 2. The allocation ratio, which is what "monopoly" means. The curve
    ///    above was measured on a 125M backbone; `small` has 7.5M, so the
    ///    same slot count is 16x more memory per backbone parameter and
    ///    3 x 500_000 x 32 = 48M params = **86% of the model** - the exact
    ///    shape that failed. DeepSeek's shipped operating point is 196B Engram
    ///    against a 552B backbone (26% of total, 0.36x) and its stated law is
    ///    20-25% of the sparse budget; 25_000 rows/order puts this preset at
    ///    2.4M memory params against 7.5M = **24% of the model, 0.32x** -
    ///    the published operating point, not a guess.
    ///
    /// In-VRAM tables round UP to a power of two (the slot index is masked on
    /// device, not divided): 25_000 -> 32_768 rows, 3.1M params, 29% of the
    /// model. The host-RAM path (`--engram-ram --engram-slots`) takes its row
    /// count from that flag; the two are the same knob until the trainer's
    /// plumbing is unfrozen. The 500K point AT THIS SCALE is untested: it is
    /// the first rung of the capacity ladder the A/B queue should run next,
    /// and it is only worth running as a single seed, because at 500K the arm
    /// is the monopoly again.
    #[serde(default = "d_engram_rows")] pub engram_rows: usize,
    /// N-gram orders, one table each, smallest first. At 8M rows only n=3
    /// had per-key support (the corpus exhausts the 16.8M 3-gram space 2750x
    /// over) while n=5 and n=8 were ~5775-way averages - 512M dead
    /// parameters, 2/3 of the table. 2/3/4 is DeepSeek's own shipped set
    /// over compressed tokens (V4.1-Flash n in {2,3,4}, Engram-27B [2,3])
    /// and the deepest order whose key space (256^4 = 4.3e9) a 46 GB byte
    /// corpus can populate; n>=5 spaces (1.1e12) are hopeless. The VALUES
    /// are what `dormouse_data::ORDERS` hashes; the COUNT is what the model
    /// sizes its tables from, and `validate` pins the count to 3 (the
    /// trainer's hash tensor is `[b, t, 3]`).
    #[serde(default = "d_engram_orders")] pub engram_orders: Vec<usize>,
    /// Columns per memory row. One shared value projection over all orders
    /// (arXiv 2601.07372 Sec. 2.4 eq. 6), so a row is a lookup, not a
    /// per-order model.
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
