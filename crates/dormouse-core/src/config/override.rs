//! `--set key=value`: the third layer of the merge, and the only one that is
//! per-run rather than per-file.
//!
//! WHY A THIRD LAYER. The merge order is defaults -> preset TOML -> `--set` ->
//! typed flags -> validate, and the ordering is the feature: an A/B is one
//! command line over a shared preset, so the arm's value must be the last
//! thing that touches the field. Presets are data and cannot encode "and on
//! this run, only this run".
//!
//! WHAT A KEY MAY BE. The `match` in [`apply_overrides`] is the whole
//! enumeration of reachable fields, and it is checked against the schema by
//! the type system: an unknown key is a loud `Err` naming it, never a silently
//! ignored flag.
//!
//! **FOUR SCHEMA FIELDS ARE UNREACHABLE FROM `--set`, recorded not fixed.**
//! Arm coverage note (2026-10-01): use_situ/use_attnres/use_mhc/mhc_streams
//! and moe_topk/moe_lb_coef all have arms; an unknown key still returns
//! `Err("unknown config key ...")` - LOUD, not silent, so nothing
//! trains differently from what was asked. But those four are then reachable
//! ONLY from a preset TOML, which is an inconsistency with every other field
//! and a trap for anyone doing an A/B from the command line: SiTU-GLU, AttnRes
//! and mHC are all queue rows and all three currently need a file edit under
//! `configs/`. Fix is four arms in the match, ~6 lines. Not done here (docs
//! lane).
//!
//! COST. One `split_once` and one `match` per `--set`; a run passes a handful.

use super::schema::ActQuant;
use super::schema::DormouseConfig;
use std::str::FromStr;

/// One `--set key=value` pair, already split and trimmed. Both halves are
/// strings: the value is parsed by the arm that knows the field's type, so a
/// bad value is reported by the field rather than by a generic parser.
#[derive(Debug, Clone)]
pub struct Override {
    /// The key as written, before normalisation. Kept verbatim because the
    /// error messages quote it: a user who typed `d-model=512` should be told
    /// about `d-model`, not about the key this resolved to.
    pub key: String,
    /// The right-hand side, trimmed, still unparsed.
    pub value: String,
}

/// Split `key=value` strings into [`Override`]s, in the order given.
///
/// LOUD on the two shapes that are certainly a mistake: no `=` at all, and an
/// empty key (`=4`, ` =4`). Order is preserved because two `--set`s for the
/// same key are last-wins and that has to be the order the command line reads.
pub fn parse_overrides(raw: &[String]) -> Result<Vec<Override>, String> {
    let mut out = Vec::new();
    for s in raw {
        let (k, v) = s
            .split_once('=')
            .ok_or_else(|| format!("--set expected key=value, got {s:?}"))?;
        let key = k.trim().to_string();
        let value = v.trim().to_string();
        if key.is_empty() {
            return Err(format!("empty key in --set {s:?}"));
        }
        out.push(Override { key, value });
    }
    Ok(out)
}

/// `true`/`false` for every bool field, spelled four ways each. Case
/// insensitive. The set is deliberately wider than Rust's `bool::from_str`
/// (which takes only two) because a shell user types `on`.
fn parse_bool(v: &str) -> Result<bool, String> {
    match v.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(format!("bool expected true/false, got {v:?}")),
    }
}

/// Write the overrides onto `cfg`, in order, so a later `--set` for the same
/// key wins. Keys are matched on the LAST dot-separated segment and
/// lowercased, so `model.d_model` and `D_MODEL` both reach `d_model`: the
/// section prefix is accepted and discarded because the trainer's flags carry
/// a prefix the model config does not.
///
/// Every unknown key is a LOUD `Err` naming the key as written. That is the
/// point of the exhaustive `match` rather than a `getattr`-style lookup: a typo
/// in an A/B command line must not run the control and call it the arm.
///
/// `act_quant` is the one field whose parser lives elsewhere - `ActQuant`'s
/// `FromStr` (ADR-0005) - so serde, `--set` and the CLI cannot disagree about
/// what `fp4` means. It additionally accepts `none`/`null`/`off`/empty, which
/// mean "the arm is off", because the field is an `Option` and there is no
/// other way to write the absence from a command line.
pub fn apply_overrides(cfg: &mut DormouseConfig, ov: &[Override]) -> Result<(), String> {
    for o in ov {
        let key = o
            .key
            .split('.')
            .next_back()
            .unwrap_or(&o.key)
            .to_ascii_lowercase();
        let v = o.value.as_str();
        match key.as_str() {
            "d_model" => cfg.d_model = v.parse().map_err(|e| format!("d_model: {e}"))?,
            "n_heads" => cfg.n_heads = v.parse().map_err(|e| format!("n_heads: {e}"))?,
            "head_dim" => cfg.head_dim = v.parse().map_err(|e| format!("head_dim: {e}"))?,
            "d_ffn" => cfg.d_ffn = v.parse().map_err(|e| format!("d_ffn: {e}"))?,
            "vocab" => cfg.vocab = v.parse().map_err(|e| format!("vocab: {e}"))?,
            "max_seq_len" => {
                cfg.max_seq_len = v.parse().map_err(|e| format!("max_seq_len: {e}"))?
            }
            "max_iter" => cfg.max_iter = v.parse().map_err(|e| format!("max_iter: {e}"))?,
            "rank" => cfg.rank = v.parse().map_err(|e| format!("rank: {e}"))?,
            "n_experts" => cfg.n_experts = v.parse().map_err(|e| format!("n_experts: {e}"))?,
            "act_group" => cfg.act_group = v.parse().map_err(|e| format!("act_group: {e}"))?,
            "jepa_mask_span" => {
                cfg.jepa_mask_span = v.parse().map_err(|e| format!("jepa_mask_span: {e}"))?
            }
            "dspark_k" => cfg.dspark_k = v.parse().map_err(|e| format!("dspark_k: {e}"))?,
            "dspark_stride" => {
                cfg.dspark_stride = v.parse().map_err(|e| format!("dspark_stride: {e}"))?
            }
            "aux_fb_horizon" => {
                cfg.aux_fb_horizon = v.parse().map_err(|e| format!("aux_fb_horizon: {e}"))?
            }
            "mor_k" => cfg.mor_k = v.parse().map_err(|e| format!("mor_k: {e}"))?,
            "mor_bce_weight" => {
                cfg.mor_bce_weight = v.parse().map_err(|e| format!("mor_bce_weight: {e}"))?
            }
            "moe_topk" => cfg.moe_topk = v.parse().map_err(|e| format!("moe_topk: {e}"))?,
            "moe_lb_coef" => {
                cfg.moe_lb_coef = v.parse().map_err(|e| format!("moe_lb_coef: {e}"))?
            }
            "norm_eps" => cfg.norm_eps = v.parse().map_err(|e| format!("norm_eps: {e}"))?,
            "engram_rows" => {
                cfg.engram_rows = v.parse().map_err(|e| format!("engram_rows: {e}"))?
            }
            "engram_dim" => cfg.engram_dim = v.parse().map_err(|e| format!("engram_dim: {e}"))?,
            "engram_lam_max" => {
                cfg.engram_lam_max = v.parse().map_err(|e| format!("engram_lam_max: {e}"))?
            }
            // Comma-separated, e.g. `--set engram_orders=2,3,4`.
            "engram_orders" => {
                cfg.engram_orders = v
                    .split(',')
                    .map(|p| {
                        p.trim()
                            .parse::<usize>()
                            .map_err(|e| format!("engram_orders: {e}"))
                    })
                    .collect::<Result<Vec<usize>, String>>()?
            }
            "jepa_weight" => {
                cfg.jepa_weight = v.parse().map_err(|e| format!("jepa_weight: {e}"))?
            }
            "jepa_mask_frac" => {
                cfg.jepa_mask_frac = v.parse().map_err(|e| format!("jepa_mask_frac: {e}"))?
            }
            "dspark_weight" => {
                cfg.dspark_weight = v.parse().map_err(|e| format!("dspark_weight: {e}"))?
            }
            "aux_fb_weight" => {
                cfg.aux_fb_weight = v.parse().map_err(|e| format!("aux_fb_weight: {e}"))?
            }
            "bf16" => cfg.bf16 = parse_bool(v)?,
            "use_tsct" => cfg.use_tsct = parse_bool(v)?,
            "use_kda" => cfg.use_kda = parse_bool(v)?,
            "use_engram" => cfg.use_engram = parse_bool(v)?,
            "use_gr" => cfg.use_gr = parse_bool(v)?,
            "use_mor" => cfg.use_mor = parse_bool(v)?,
            "use_situ" => cfg.use_situ = parse_bool(v)?,
            "use_attnres" => cfg.use_attnres = parse_bool(v)?,
            "use_mhc" => cfg.use_mhc = parse_bool(v)?,
            "mhc_streams" => {
                cfg.mhc_streams = v.parse().map_err(|e| format!("mhc_streams: {e}"))?
            }
            "act_quant" => {
                cfg.act_quant = match v.to_ascii_lowercase().as_str() {
                    "none" | "null" | "off" | "" => None,
                    // The one act-quant parser: ActQuant::from_str (ADR-0005).
                    _ => Some(ActQuant::from_str(v).map_err(|e| format!("act_quant: {e}"))?),
                }
            }
            _ => return Err(format!("unknown config key {:?}", o.key)),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::schema::DormouseConfig;
    use super::{apply_overrides, parse_overrides};

    fn one(set: &str) -> Result<DormouseConfig, String> {
        let mut c = DormouseConfig::default();
        let ov = parse_overrides(&[set.to_string()])?;
        apply_overrides(&mut c, &ov)?;
        Ok(c)
    }

    /// The queue's A/B arms were reachable only from a preset TOML — every
    /// other field resolves from --set, these four must too
    /// (docs/reviews/doc-coverage-2026-10-01.md §3.1).
    #[test]
    fn arm_flags_resolve_from_set() {
        assert!(one("use_situ=true").unwrap().use_situ);
        assert!(one("use_attnres=true").unwrap().use_attnres);
        assert!(one("use_mhc=true").unwrap().use_mhc);
        assert_eq!(one("mhc_streams=4").unwrap().mhc_streams, 4);
    }
}
