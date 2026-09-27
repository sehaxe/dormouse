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
            "msa_topk" => cfg.msa_topk = v.parse().map_err(|e| format!("msa_topk: {e}"))?,
            "msa_block" => cfg.msa_block = v.parse().map_err(|e| format!("msa_block: {e}"))?,
            "n_experts" => cfg.n_experts = v.parse().map_err(|e| format!("n_experts: {e}"))?,
            "act_group" => cfg.act_group = v.parse().map_err(|e| format!("act_group: {e}"))?,
            "jepa_mask_span" => cfg.jepa_mask_span = v.parse().map_err(|e| format!("jepa_mask_span: {e}"))?,
            "dspark_k" => cfg.dspark_k = v.parse().map_err(|e| format!("dspark_k: {e}"))?,
            "dspark_stride" => cfg.dspark_stride = v.parse().map_err(|e| format!("dspark_stride: {e}"))?,
            "norm_eps" => cfg.norm_eps = v.parse().map_err(|e| format!("norm_eps: {e}"))?,
            "jepa_weight" => cfg.jepa_weight = v.parse().map_err(|e| format!("jepa_weight: {e}"))?,
            "jepa_mask_frac" => cfg.jepa_mask_frac = v.parse().map_err(|e| format!("jepa_mask_frac: {e}"))?,
            "dspark_weight" => cfg.dspark_weight = v.parse().map_err(|e| format!("dspark_weight: {e}"))?,
            "bf16" => cfg.bf16 = parse_bool(v)?,
            "use_msa" => cfg.use_msa = parse_bool(v)?,
            "use_kda" => cfg.use_kda = parse_bool(v)?,
            "use_engram" => cfg.use_engram = parse_bool(v)?,
            "use_gr" => cfg.use_gr = parse_bool(v)?,
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
