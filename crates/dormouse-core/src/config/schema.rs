use serde::{Deserialize, Deserializer, Serialize, Serializer, de::{self, Visitor}};
use crate::act_quant::ActFormat;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActQuant { Fp4, Int(u32) }
impl From<ActQuant> for ActFormat {
    fn from(q: ActQuant) -> Self { match q { ActQuant::Fp4 => ActFormat::Fp4, ActQuant::Int(b) => ActFormat::Int(b) } }
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
                match v.to_ascii_lowercase().as_str() {
                    "fp4" => Ok(ActQuant::Fp4), "4"|"int4" => Ok(ActQuant::Int(4)), "8"|"int8" => Ok(ActQuant::Int(8)),
                    _ => v.parse::<u32>().map(ActQuant::Int).map_err(|_| E::custom(format!("unknown act_quant {v:?}"))),
                }
            }
            fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> { self.visit_str(&v) }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                match v { 4|8 => Ok(ActQuant::Int(v as u32)), _ => Err(E::custom(format!("act_quant int {v}")))}
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> { self.visit_u64(v as u64) }
        }
        d.deserialize_any(V)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DormouseConfig {
    pub bf16: bool, pub act_quant: Option<ActQuant>, pub act_group: usize,
    pub d_model: usize, pub n_heads: usize, pub head_dim: usize, pub d_ffn: usize,
    pub vocab: usize, pub max_seq_len: usize, pub max_iter: usize, pub rank: usize,
    pub halt_theta: f32, pub ponder_w: f32, pub rec_w: f32, pub guard_w: f32,
    pub norm_eps: f32, pub rope_base: f64, pub msa_topk: usize, pub msa_block: usize,
    pub use_msa: bool, pub use_kda: bool, pub use_engram: bool, pub use_gr: bool,
    pub keep_frac: f32, pub dropout: f32, pub n_experts: usize,
    pub ponder_beta: f32, pub ponder_prior: f32,
    pub jepa_weight: f32, pub jepa_mask_frac: f32, pub jepa_mask_span: usize,
    pub dspark_weight: f32, pub dspark_k: usize, pub dspark_stride: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSection {
    pub d_model: usize, pub n_heads: usize, pub head_dim: usize, pub d_ffn: usize,
    pub vocab: usize, pub max_seq_len: usize, pub max_iter: usize, pub rank: usize,
    pub msa_topk: usize, pub msa_block: usize, pub n_experts: usize,
    pub norm_eps: f32, pub rope_base: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PonderSection {
    pub halt_theta: f32, pub ponder_w: f32, pub rec_w: f32, pub guard_w: f32,
    pub ponder_beta: f32, pub ponder_prior: f32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuxSection {
    pub jepa_weight: f32, pub jepa_mask_frac: f32, pub jepa_mask_span: usize,
    pub dspark_weight: f32, pub dspark_k: usize, pub dspark_stride: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeSection {
    #[serde(default)] pub bf16: bool,
    #[serde(default)] pub act_quant: Option<ActQuant>,
    #[serde(default)] pub act_group: usize,
    #[serde(default="default_true")] pub use_msa: bool,
    #[serde(default="default_true")] pub use_kda: bool,
    #[serde(default="default_true")] pub use_engram: bool,
    #[serde(default)] pub use_gr: bool,
    #[serde(default="default_keep")] pub keep_frac: f32,
    #[serde(default)] pub dropout: f32,
}
fn default_true() -> bool { true }
fn default_keep() -> f32 { 0.5 }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileConfig {
    pub model: ModelSection, pub ponder: PonderSection, pub aux: AuxSection,
    #[serde(default)] pub runtime: RuntimeSection,
}
impl From<FileConfig> for DormouseConfig {
    fn from(f: FileConfig) -> Self {
        Self {
            bf16: f.runtime.bf16, act_quant: f.runtime.act_quant, act_group: f.runtime.act_group,
            d_model: f.model.d_model, n_heads: f.model.n_heads, head_dim: f.model.head_dim, d_ffn: f.model.d_ffn,
            vocab: f.model.vocab, max_seq_len: f.model.max_seq_len, max_iter: f.model.max_iter, rank: f.model.rank,
            halt_theta: f.ponder.halt_theta, ponder_w: f.ponder.ponder_w, rec_w: f.ponder.rec_w, guard_w: f.ponder.guard_w,
            norm_eps: f.model.norm_eps, rope_base: f.model.rope_base, msa_topk: f.model.msa_topk, msa_block: f.model.msa_block,
            use_msa: f.runtime.use_msa, use_kda: f.runtime.use_kda, use_engram: f.runtime.use_engram, use_gr: f.runtime.use_gr,
            keep_frac: f.runtime.keep_frac, dropout: f.runtime.dropout, n_experts: f.model.n_experts,
            ponder_beta: f.ponder.ponder_beta, ponder_prior: f.ponder.ponder_prior,
            jepa_weight: f.aux.jepa_weight, jepa_mask_frac: f.aux.jepa_mask_frac, jepa_mask_span: f.aux.jepa_mask_span,
            dspark_weight: f.aux.dspark_weight, dspark_k: f.aux.dspark_k, dspark_stride: f.aux.dspark_stride,
        }
    }
}
impl From<DormouseConfig> for FileConfig {
    fn from(c: DormouseConfig) -> Self {
        Self {
            model: ModelSection { d_model: c.d_model, n_heads: c.n_heads, head_dim: c.head_dim, d_ffn: c.d_ffn, vocab: c.vocab, max_seq_len: c.max_seq_len, max_iter: c.max_iter, rank: c.rank, msa_topk: c.msa_topk, msa_block: c.msa_block, n_experts: c.n_experts, norm_eps: c.norm_eps, rope_base: c.rope_base },
            ponder: PonderSection { halt_theta: c.halt_theta, ponder_w: c.ponder_w, rec_w: c.rec_w, guard_w: c.guard_w, ponder_beta: c.ponder_beta, ponder_prior: c.ponder_prior },
            aux: AuxSection { jepa_weight: c.jepa_weight, jepa_mask_frac: c.jepa_mask_frac, jepa_mask_span: c.jepa_mask_span, dspark_weight: c.dspark_weight, dspark_k: c.dspark_k, dspark_stride: c.dspark_stride },
            runtime: RuntimeSection { bf16: c.bf16, act_quant: c.act_quant, act_group: c.act_group, use_msa: c.use_msa, use_kda: c.use_kda, use_engram: c.use_engram, use_gr: c.use_gr, keep_frac: c.keep_frac, dropout: c.dropout },
        }
    }
}
