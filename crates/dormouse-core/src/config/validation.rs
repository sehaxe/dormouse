use super::schema::DormouseConfig;

pub fn validate(c: &DormouseConfig) -> Result<(), String> {
    macro_rules! gt0 {
        ($v:expr, $name:expr) => { if $v == 0 { return Err(format!("{} must be >0", $name)); } };
    }
    gt0!(c.d_model, "d_model"); gt0!(c.n_heads, "n_heads"); gt0!(c.head_dim, "head_dim");
    gt0!(c.d_ffn, "d_ffn"); gt0!(c.vocab, "vocab"); gt0!(c.max_seq_len, "max_seq_len");
    gt0!(c.max_iter, "max_iter"); gt0!(c.rank, "rank"); gt0!(c.msa_topk, "msa_topk");
    gt0!(c.msa_block, "msa_block"); gt0!(c.n_experts, "n_experts");
    if c.norm_eps <= 0.0 { return Err("norm_eps must be >0".into()); }
    if c.rope_base <= 0.0 { return Err("rope_base must be >0".into()); }
    if !(0.0..=1.0).contains(&c.halt_theta) { return Err("halt_theta must be 0..1".into()); }
    if !(0.0..=1.0).contains(&c.keep_frac) { return Err("keep_frac must be 0..1".into()); }
    if !(0.0..=1.0).contains(&c.dropout) { return Err("dropout must be 0..1".into()); }
    if c.ponder_w < 0.0 || c.rec_w < 0.0 || c.guard_w < 0.0 { return Err("ponder/rec/guard weights >=0".into()); }
    if c.ponder_beta < 0.0 { return Err("ponder_beta >=0".into()); }
    if c.ponder_prior <= 0.0 || c.ponder_prior > 1.0 { return Err("ponder_prior must be 0<..<=1".into()); }
    if c.jepa_weight < 0.0 || c.dspark_weight < 0.0 { return Err("jepa/dspark_weight >=0".into()); }
    if c.jepa_mask_frac < 0.0 || c.jepa_mask_frac > 1.0 { return Err("jepa_mask_frac 0..1".into()); }
    if c.d_ffn % 2 != 0 { /* SwiGLU needs even */ }
    if c.d_model % c.n_heads != 0 { return Err("d_model must be divisible by n_heads".into()); }
    Ok(())
}
