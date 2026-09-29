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
fn d_dspark_weight() -> f32 { 0.0 }
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
    /// over compressed tokens (V4.1-Flash n in {2,3,4}, Engram-27B [2,3])
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
