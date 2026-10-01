//! `--set key=value`: the CLI's seam into the schema.
//!
//! # Why a hand-written match and not a serde round-trip
//!
//! Applied after the TOML, so a flag beats a preset; and routed through the
//! SAME parsers as deserialization (`ActQuant::from_str`, `str::parse`,
//! `parse_bool`), so `--set` cannot accept a value a TOML file would have
//! rejected. Both properties are the reason this file exists rather than
//! `toml::from_str(&format!("{k} = {v}"))`.
//!
//! # LOUD, both ways
//!
//! * An unparsable value is `Err("<key>: <parse error>")`, never a silent
//!   fallback to the default.
//! * An UNKNOWN key is `Err("unknown config key ...")`, never a no-op. That
//!   direction matters more than it looks: a misspelled flag that quietly
//!   changed nothing is a run whose config differs from what the log says, and
//!   the log is the only record.
//! * Keys are matched on their LAST dotted segment and lowercased, so both
//!   `--set use_gr=true` and `--set model.use_gr=true` reach `use_gr`. A
//!   prefix is therefore decorative — there is no namespacing to get wrong.
//!
//! # Known coverage gap (2026-10-01)
//!
//! `use_situ`, `use_attnres`, `use_mhc` and `mhc_streams` have NO arm here, so
//! `--set use_situ=true` is refused as an unknown key. It is a LOUD refusal, not
//! a silent no-op, so nothing trains differently from what was asked — but those
//! four fields are reachable only from a preset TOML, which is an inconsistency
//! with every other field. Recorded in
//! `docs/reviews/doc-coverage-2026-10-01.md`; not fixed here (docs lane).

use super::schema::ActQuant;
use super::schema::DormouseConfig;
use std::str::FromStr;

/// One parsed `--set key=value`, still both strings.
///
/// Deliberately untyped: parsing happens in [`apply_overrides`], against the
/// live config, so a value's type is the field's type and there is no second
/// type table to drift from [`DormouseConfig`].
#[derive(Debug, Clone)]
pub struct Override {
    /// The key as written, trimmed, original case and any dotted prefix
    /// intact. Only [`apply_overrides`] interprets it.
    pub key: String,
    /// The value as written, trimmed. Whitespace around `=` is stripped;
    /// whitespace inside is kept and will usually fail to parse, loudly.
    pub value: String,
}

/// Split each `key=value` argument, LOUDly.
///
/// `Err` names the offending argument: a missing `=` (`--set expected
/// key=value, got "..."`) or an empty key. Both are refused rather than
/// skipped, because a `--set` that applies nothing is a run whose config is
/// not the one on the command line.
pub fn parse_overrides(raw: &[String]) -> Result<Vec<Override>, String> {
    let mut out = Vec::new();
    for s in raw {
        let (k, v) = s.split_once('=').ok_or_else(|| format!("--set expected key=value, got {s:?}"))?;
        let key = k.trim().to_string();
        let value = v.trim().to_string();
        if key.is_empty() { return Err(format!("empty key in --set {s:?}")); }
        out.push(Override { key, value });
    }
    Ok(out)
}

/// `true`/`1`/`yes`/`on` and `false`/`0`/`no`/`off`, case-insensitive.
/// Anything else is `Err` naming what was expected — a bool flag must not
/// silently read as false.
fn parse_bool(v: &str) -> Result<bool, String> {
    match v.to_ascii_lowercase().as_str() { "true"|"1"|"yes"|"on" => Ok(true), "false"|"0"|"no"|"off" => Ok(false), _ => Err(format!("bool expected true/false, got {v:?}")) }
}

/// Apply `--set` overrides to `cfg`, in order.
///
/// Mutates in place and returns `Err` on the FIRST bad override, having
/// already applied the ones before it — so on `Err` the config is
/// partially-overridden and must be discarded by the caller. It is a
/// startup-time path that aborts the process, so there is no rollback to
/// write; the doc is the warning.
///
/// NOT idempotent and NOT validating: it does not call [`super::validate`].
/// A caller that overrides a field into an illegal combination (for example
/// `use_situ` with a width the gate_up projection cannot take) must validate
/// afterwards, and the loader does.
///
/// A `None` for `act_quant` is spelled `none`/`null`/`off`/empty — there is no
/// way to say "leave it alone", because this is the last writer.
pub fn apply_overrides(cfg: &mut DormouseConfig, ov: &[Override]) -> Result<(), String> {
    for o in ov {
        let key = o.key.split('.').last().unwrap_or(&o.key).to_ascii_lowercase();
        let v = o.value.as_str();
        match key.as_str() {
            "d_model" => cfg.d_model = v.parse().map_err(|e| format!("d_model: {e}"))?,
            "n_heads" => cfg.n_heads = v.parse().map_err(|e| format!("n_heads: {e}"))?,
            "head_dim" => cfg.head_dim = v.parse().map_err(|e| format!("head_dim: {e}"))?,
            "d_ffn" => cfg.d_ffn = v.parse().map_err(|e| format!("d_ffn: {e}"))?,
            "vocab" => cfg.vocab = v.parse().map_err(|e| format!("vocab: {e}"))?,
            "max_seq_len" => cfg.max_seq_len = v.parse().map_err(|e| format!("max_seq_len: {e}"))?,
            "max_iter" => cfg.max_iter = v.parse().map_err(|e| format!("max_iter: {e}"))?,
            "rank" => cfg.rank = v.parse().map_err(|e| format!("rank: {e}"))?,
            "n_experts" => cfg.n_experts = v.parse().map_err(|e| format!("n_experts: {e}"))?,
            "act_group" => cfg.act_group = v.parse().map_err(|e| format!("act_group: {e}"))?,
            "jepa_mask_span" => cfg.jepa_mask_span = v.parse().map_err(|e| format!("jepa_mask_span: {e}"))?,
            "dspark_k" => cfg.dspark_k = v.parse().map_err(|e| format!("dspark_k: {e}"))?,
            "dspark_stride" => cfg.dspark_stride = v.parse().map_err(|e| format!("dspark_stride: {e}"))?,
            "aux_fb_horizon" => cfg.aux_fb_horizon = v.parse().map_err(|e| format!("aux_fb_horizon: {e}"))?,
            "mor_k" => cfg.mor_k = v.parse().map_err(|e| format!("mor_k: {e}"))?,
            "mor_bce_weight" => cfg.mor_bce_weight = v.parse().map_err(|e| format!("mor_bce_weight: {e}"))?,
            "moe_topk" => cfg.moe_topk = v.parse().map_err(|e| format!("moe_topk: {e}"))?,
            "moe_lb_coef" => cfg.moe_lb_coef = v.parse().map_err(|e| format!("moe_lb_coef: {e}"))?,
            "norm_eps" => cfg.norm_eps = v.parse().map_err(|e| format!("norm_eps: {e}"))?,
            "engram_rows" => cfg.engram_rows = v.parse().map_err(|e| format!("engram_rows: {e}"))?,
            "engram_dim" => cfg.engram_dim = v.parse().map_err(|e| format!("engram_dim: {e}"))?,
            "engram_lam_max" => cfg.engram_lam_max = v.parse().map_err(|e| format!("engram_lam_max: {e}"))?,
            // Comma-separated, e.g. `--set engram_orders=2,3,4`.
            "engram_orders" => cfg.engram_orders = v
                .split(',')
                .map(|p| p.trim().parse::<usize>().map_err(|e| format!("engram_orders: {e}")))
                .collect::<Result<Vec<usize>, String>>()?,            "jepa_weight" => cfg.jepa_weight = v.parse().map_err(|e| format!("jepa_weight: {e}"))?,
            "jepa_mask_frac" => cfg.jepa_mask_frac = v.parse().map_err(|e| format!("jepa_mask_frac: {e}"))?,
            "dspark_weight" => cfg.dspark_weight = v.parse().map_err(|e| format!("dspark_weight: {e}"))?,
            "aux_fb_weight" => cfg.aux_fb_weight = v.parse().map_err(|e| format!("aux_fb_weight: {e}"))?,
            "bf16" => cfg.bf16 = parse_bool(v)?,
            "use_tsct" => cfg.use_tsct = parse_bool(v)?,
            "use_kda" => cfg.use_kda = parse_bool(v)?,
            "use_engram" => cfg.use_engram = parse_bool(v)?,
            "use_gr" => cfg.use_gr = parse_bool(v)?,
            "use_mor" => cfg.use_mor = parse_bool(v)?,
            "act_quant" => cfg.act_quant = match v.to_ascii_lowercase().as_str() {
                "none" | "null" | "off" | "" => None,
                // The one act-quant parser: ActQuant::from_str (ADR-0005).
                _ => Some(ActQuant::from_str(v).map_err(|e| format!("act_quant: {e}"))?),
            },
            _ => return Err(format!("unknown config key {:?}", o.key)),
        }
    }
    Ok(())
}
