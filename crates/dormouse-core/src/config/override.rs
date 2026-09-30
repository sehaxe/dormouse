use super::schema::ActQuant;
use super::schema::DormouseConfig;
use std::str::FromStr;

#[derive(Debug, Clone)]
pub struct Override { pub key: String, pub value: String }

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

fn parse_bool(v: &str) -> Result<bool, String> {
    match v.to_ascii_lowercase().as_str() { "true"|"1"|"yes"|"on" => Ok(true), "false"|"0"|"no"|"off" => Ok(false), _ => Err(format!("bool expected true/false, got {v:?}")) }
}

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
