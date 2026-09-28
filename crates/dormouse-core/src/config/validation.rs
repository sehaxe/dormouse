use super::schema::DormouseConfig;
#[cfg(test)]
use super::schema::ActQuant;

pub fn validate(c: &DormouseConfig) -> Result<(), String> {
    macro_rules! gt0 {
        ($v:expr, $name:expr) => { if $v == 0 { return Err(format!("{} must be >0", $name)); } };
    }
    gt0!(c.d_model, "d_model"); gt0!(c.n_heads, "n_heads"); gt0!(c.head_dim, "head_dim");
    gt0!(c.d_ffn, "d_ffn"); gt0!(c.vocab, "vocab"); gt0!(c.max_seq_len, "max_seq_len");
    gt0!(c.max_iter, "max_iter"); gt0!(c.rank, "rank");
    gt0!(c.n_experts, "n_experts");
    if c.norm_eps <= 0.0 { return Err("norm_eps must be >0".into()); }
    if c.jepa_weight < 0.0 || c.dspark_weight < 0.0 { return Err("jepa/dspark_weight >=0".into()); }
    if c.jepa_mask_frac < 0.0 || c.jepa_mask_frac > 1.0 { return Err("jepa_mask_frac 0..1".into()); }
    // DSpark anchors. `dspark_aux_loss` sizes the anchor COUNT with
    // `stride.max(1)` but multiplies the anchor POSITIONS by the raw stride,
    // so stride = 0 puts every one of the n = t-k-1 anchors on position 0 and
    // the k-step draft window becomes k copies of one CE: the objective still
    // computes, still descends, and is no longer the one the config names.
    // LOUD here rather than clamped in the loss wrapper (ADR-0019).
    if c.dspark_stride == 0 {
        return Err(
            "dspark_stride must be >= 1: 0 places every draft anchor at position 0, so the \
             k-step DSpark window collapses to k copies of one CE. Use 1 for every position, \
             or dspark_weight = 0 to turn the term off."
                .into(),
        );
    }
    if c.mor_bce_weight < 0.0 { return Err("mor_bce_weight >=0".into()); }
    // The floor of one recursion is structural, so mor_k=0 is a config error
    // (loud, not clamped), and mor_k > max_iter is a set that cannot fill.
    if c.mor_k < 1 { return Err("mor_k must be >= 1 (floor of one recursion)".into()); }
    if c.use_mor && c.mor_k > c.max_iter {
        return Err(format!("mor_k {} > max_iter {}: the top-k set cannot fill", c.mor_k, c.max_iter));
    }
    if c.d_model % c.n_heads != 0 { return Err("d_model must be divisible by n_heads".into()); }
    // The act-quant group size. `quant_act` takes `g = min(act_group, d)` and
    // reshapes the [b*t, d] activations to [b, d/g, g], so a `g` that does not
    // divide `d` makes d/g*g != d and the reshape fails on the FIRST forward
    // (act_quant.rs:82) - a shape error a long way from the flag that caused
    // it. `act_group` is a free-form `--set` value, so it is checked here.
    if c.act_quant.is_some() && c.act_group != 0 && c.d_model % c.act_group != 0 {
        return Err(format!(
            "act_group {} must divide d_model {} (or be 0 for one scale per token): the \
             quantizer groups the activations in blocks of act_group, so a non-divisor fails \
             the reshape on the first forward",
            c.act_group, c.d_model
        ));
    }
    // Engram capacity. The order COUNT is load-bearing beyond the model: the
    // trainer puts one hash column per order into a `[b, t, 3]` tensor, so a
    // different count is a shape mismatch there, not a silent relayout.
    gt0!(c.engram_rows, "engram_rows");
    gt0!(c.engram_dim, "engram_dim");
    if c.engram_orders.len() != 3 {
        return Err(format!(
            "engram_orders must have 3 entries (the trainer's hash tensor is [b,t,3]), got {:?}",
            c.engram_orders
        ));
    }
    if c.engram_orders.iter().any(|&n| n == 0 || n > 32) {
        return Err(format!("engram_orders entries must be 1..=32, got {:?}", c.engram_orders));
    }
    if !(c.engram_lam_max > 0.0 && c.engram_lam_max <= 1.0) {
        return Err("engram_lam_max must be in (0, 1]: 0 would delete the arm, >1 is not a floor".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{apply_overrides, parse_overrides};

    /// The real path a bad value arrives on: `--set key=value` into the
    /// defaults, then `validate` - the same three steps `resolve` runs.
    fn set_then_validate(sets: &[&str]) -> Result<(), String> {
        let mut c = DormouseConfig::default();
        let raw: Vec<String> = sets.iter().map(|s| s.to_string()).collect();
        apply_overrides(&mut c, &parse_overrides(&raw)?)?;
        validate(&c)
    }

    /// ADR-0019 LOUD: `--set dspark_stride=0` is a value that computes the
    /// WRONG objective without saying so. `dspark_aux_loss` sizes the anchor
    /// count with `stride.max(1)` but scales the anchor positions by the raw
    /// stride, so all n = t-k-1 anchors land on position 0 and the k-step
    /// draft window is k copies of one CE. Loud at startup, naming the field,
    /// the value and the escape.
    #[test]
    fn dspark_stride_zero_is_refused() {
        let err = set_then_validate(&["dspark_stride=0"]).expect_err("stride 0 must be refused");
        assert!(err.contains("dspark_stride"), "the error must name the field: {err}");
        // The shipped value and its neighbours are untouched: this check
        // refuses the collapse, it does not shrink the legal range.
        assert!(set_then_validate(&["dspark_stride=1"]).is_ok(), "stride 1 is every position");
        assert!(set_then_validate(&["dspark_stride=16"]).is_ok(), "the preset value");
        assert!(set_then_validate(&[]).is_ok(), "the defaults");
    }

    /// Same class, different file: `--set act_group=N` is free-form and
    /// `quant_act` reshapes to `[b, d/g, g]`, so a `g` that does not divide
    /// `d_model` dies on the FIRST forward instead of at startup.
    #[test]
    fn act_group_must_divide_d_model_when_act_quant_is_on() {
        let e = set_then_validate(&["act_quant=8", "act_group=100"])
            .expect_err("a non-divisor group must be refused");
        assert!(e.contains("act_group"), "the error must name the field: {e}");
        // 0 is the documented per-token spelling and stays legal; so does a
        // real divisor, and so does a group with the quantizer off (the field
        // is unread then - refusing it would reject a config that trains).
        assert!(set_then_validate(&["act_quant=8", "act_group=0"]).is_ok(), "0 = per token");
        assert!(set_then_validate(&["act_quant=8", "act_group=64"]).is_ok(), "64 divides 768");
        assert!(set_then_validate(&["act_group=100"]).is_ok(), "inert without act_quant");
    }

    /// Every shipped preset must still validate: a check that rejects a
    /// config the project ships is a wrong check, and the honest answer is to
    /// say so rather than narrow the check to fit.
    #[test]
    fn every_shipped_preset_still_validates() {
        for name in ["nano", "small", "base", "swift50", "one_b", "p150", "mor", "nano-fused"] {
            let c = crate::config::load_config(name).unwrap_or_else(|e| panic!("{name}: {e}"));
            validate(&c).unwrap_or_else(|e| panic!("{name} must validate: {e}"));
        }
    }

    /// The act-quant field is a `config` value, not a struct built by hand:
    /// pin the spelling the CLI parses, so a future ActQuant variant cannot
    /// make this test pass for the wrong reason.
    #[test]
    fn act_group_check_sees_the_parsed_format() {
        let mut c = DormouseConfig { act_quant: Some(ActQuant::Fp4), act_group: 100, ..Default::default() };
        assert!(validate(&c).is_err(), "fp4 takes the same grouping path");
        c.act_group = 128;
        assert!(validate(&c).is_ok(), "128 divides 768");
    }
}
