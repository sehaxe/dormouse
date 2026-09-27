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
fn d_msa_topk() -> usize { 8 }
fn d_msa_block() -> usize { 32 }
fn d_true() -> bool { true }
fn d_false() -> bool { false }
fn d_n_experts() -> usize { 3 }
fn d_jepa_weight() -> f32 { 0.05 }
fn d_jepa_mask_frac() -> f32 { 0.15 }
fn d_jepa_mask_span() -> usize { 8 }
fn d_dspark_weight() -> f32 { 0.1 }
fn d_dspark_k() -> usize { 4 }
fn d_dspark_stride() -> usize { 16 }

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
    #[serde(default = "d_msa_topk")] pub msa_topk: usize,
    #[serde(default = "d_msa_block")] pub msa_block: usize,
    #[serde(default = "d_false")] pub use_msa: bool,
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
