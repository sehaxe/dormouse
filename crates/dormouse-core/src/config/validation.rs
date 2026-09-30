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
    // The future-byte head (queue v2 item 5). Both checks are LOUD because both
    // wrong values compute a DIFFERENT objective without saying so, which is
    // the ADR-0019 class: a negative weight is a sign flip on the aux term, and
    // `aux_fb_horizon = 0` is the subtle one - the label for position q would
    // be `targets[q]`, the byte the MAIN CE already predicts, so the term is a
    // second CE on the next byte through an independent head. It trains, it
    // descends, it costs a step, and it is not multi-token prediction at all.
    if c.aux_fb_weight < 0.0 { return Err("aux_fb_weight >= 0".into()); }
    if c.aux_fb_horizon < 1 {
        return Err(format!(
            "aux_fb_horizon {} must be >= 1: the label for position q is targets[q + k], so k = 0 \
             supervises the head on the byte the MAIN CE already predicts - the term still computes \
             and still descends, and it is not multi-token prediction. Use 2 (the default), or \
             aux_fb_weight = 0 to turn the arm off.",
            c.aux_fb_horizon
        ));
    }
    // The floor of one recursion is structural, so mor_k=0 is a config error
    // (loud, not clamped), and mor_k > max_iter is a set that cannot fill.
    if c.mor_k < 1 { return Err("mor_k must be >= 1 (floor of one recursion)".into()); }
    if c.use_mor && c.mor_k > c.max_iter {
        return Err(format!("mor_k {} > max_iter {}: the top-k set cannot fill", c.mor_k, c.max_iter));
    }
    // Sparse expert routing over the FFN branch. Two LOUD checks, both the
    // class where a wrong value computes a DIFFERENT objective without saying
    // so (ADR-0019):
    //
    // 1. A top-k larger than the bank silently becomes the dense blend. That
    //    is the control, wearing the arm's label - and the run's log would
    //    print `moe=...` for a model that never routed anything. Refused,
    //    naming the escape (`moe_topk = 0` or a larger `n_experts`).
    // 2. A load-balance coefficient with routing OFF is a term with nothing to
    //    balance. Ignored, it would be a config field that reads as an
    //    objective and contributes nothing - the `aux_fb_weight`-without-a-head
    //    defect, and the one shape of this this repo has already been bitten by
    //    (`probe::JEPA` was bumped nowhere, so the field was a no-op while the
    //    loss curve looked healthy).
    if c.moe_topk > c.n_experts {
        return Err(format!(
            "moe_topk {} > n_experts {}: the top-k set cannot fill, and topk > n_experts silently \
             runs the DENSE blend - the control wearing the arm's label. Use moe_topk = 0 (the \
             dense default), moe_topk <= n_experts, or a larger n_experts.",
            c.moe_topk, c.n_experts
        ));
    }
    if c.moe_lb_coef < 0.0 {
        return Err("moe_lb_coef >= 0".into());
    }
    if c.moe_lb_coef > 0.0 && c.moe_topk == 0 {
        return Err(format!(
            "moe_lb_coef {} with moe_topk = 0: the load-balancing term has no selection to balance, \
             so it would be silently ignored while the config reads like an objective. Set \
             moe_topk >= 1, or moe_lb_coef = 0.",
            c.moe_lb_coef
        ));
    }
    // DELIBERATELY NOT a refusal: `moe_topk > 0` with `moe_lb_coef == 0`.
    //
    // The risk is real - Switch §3.3 is the whole reason the term exists, and a
    // collapsed router means every pass picks the same expert while the loss
    // curve stays healthy. So this configuration is DELIVERABLE, and it is
    // legal, for three reasons:
    //
    // 1. It is the arm's OWN REMOVAL under the A/B rule (AGENTS.md 1.2: every
    //    mechanism beats its own removal). A routing arm that cannot be run
    //    without its balancer cannot be measured against one.
    // 2. Refusing it would force the operator to pass SOME coefficient, and
    //    the sweep (research/reviews/moe-routing-2026-10-01.md §5) measures
    //    that every value in the published range is three orders of magnitude
    //    too weak at our token count. A refusal here would manufacture exactly
    //    the "copied a large-MoE number" defect the external review named.
    // 3. The risk is already VISIBLE without a refusal: `probe::MOE_ROUTE`
    //    counts the selections and `probe::MOE_LB` counts the balancer terms
    //    added, so the eval line's `moe=<lb>/<sel>` reads `0/<n>` for a
    //    routed run with no balancer behind it. That is the repo's own
    //    COUNTED mark for "a fallback happened and a reader can tell", which
    //    is the correct instrument here - not a loud refusal of a legal run.
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
    // AttnRes, Gated Residual and mHC all REPLACE the residual accumulation, in
    // the same statement of the loop (the `else` chain in `forward_full_state`).
    // Any two of them on is not a configuration with an interpretation: one of
    // them would be silently ignored, and the reader of the run's log would
    // have no way to tell which. Refused here, where the escape can be named
    // (ADR-0011). One check over the three, so adding a fourth arm cannot
    // leave a pair unguarded.
    let residual_arms = [
        ("use_attnres", c.use_attnres),
        ("use_gr", c.use_gr),
        ("use_mhc", c.use_mhc),
    ];
    let on: Vec<&str> = residual_arms
        .iter()
        .filter(|(_, is_on)| *is_on)
        .map(|(name, _)| *name)
        .collect();
    if on.len() > 1 {
        return Err(format!(
            "{} all replace the loop's residual accumulation and only one can run: \
             set all but one to false. Each is a one-flag swap against ReZero, so the \
             comparison that means something is the arm against the additive residual \
             (use_attnres / use_gr / use_mhc = false), not two arms against each other.",
            on.join(" and ")
        ));
    }
    // `MhcBlock` reshapes the hidden state to `[b, t, n, D/n]` in its first
    // forward, so a non-divisor `n` is a shape panic a long way from the flag
    // that caused it. The same class as `act_group` above, and the same
    // reason for a loud check instead of a clamp: a clamped `n` trains a
    // mechanism the config does not name.
    if c.use_mhc {
        if c.mhc_streams == 0 {
            return Err("mhc_streams must be >= 1".into());
        }
        if c.d_model % c.mhc_streams != 0 {
            return Err(format!(
                "mhc_streams {} must divide d_model {}: the residual stream is read as \
                 n streams of width d_model/n, so a non-divisor fails the reshape in the \
                 first forward. Default 2 is the K the phi study (2604.21106v3 5.2) ran \
                 and the base paper's own n=4 (2409.19606 App. Tab. 1) is --set \
                 mhc_streams=4.",
                c.mhc_streams, c.d_model
            ));
        }
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

    /// ADR-0019 LOUD, the subtle half of the pair: `--set aux_fb_horizon=0`
    /// supervises the head on `targets[q]` - the byte the MAIN CE already
    /// predicts - so the arm trains, descends, costs a step, and is not
    /// multi-token prediction at all. It is a second CE on the next byte
    /// through an independent parameter set, which is the most expensive way to
    /// buy nothing. A horizon at or past `max_seq_len` is legal and returns an
    /// exact zero (COUNTED: `fb=0/<n>` on the eval line), because a
    /// short-sequence sweep is a legitimate thing to want.
    #[test]
    fn aux_fb_horizon_zero_is_refused() {
        let err = set_then_validate(&["aux_fb_horizon=0"]).expect_err("horizon 0 must be refused");
        assert!(err.contains("aux_fb_horizon"), "the error must name the field: {err}");
        assert!(err.contains("targets[q + k]"), "the error must name the arithmetic: {err}");
        assert!(set_then_validate(&["aux_fb_weight=-0.1"]).is_err(), "a negative weight is a sign flip on the term");
        // The legal range is untouched: the check refuses the collapse, it does
        // not shrink the range.
        assert!(set_then_validate(&[]).is_ok(), "the defaults (weight 0, horizon 2)");
        assert!(set_then_validate(&["aux_fb_horizon=1"]).is_ok(), "k = 1 is a second CE on the next byte - legal, and the doc says so");
        assert!(set_then_validate(&["aux_fb_horizon=4", "aux_fb_weight=0.1"]).is_ok(), "the later arm's horizon");
        // A horizon at or past the sequence length is NOT refused: it is a
        // zero term with a counter, not a config error.
        assert!(set_then_validate(&["aux_fb_horizon=600", "max_seq_len=512"]).is_ok(), "seq_len is the trainer's, not the config's business here");
    }

    /// ADR-0019 LOUD for the routing arm, both halves. The dangerous one is
    /// `moe_topk > n_experts`: `topk_blend` would clamp to the whole bank and
    /// compute the DENSE blend, so the run trains the control while its config,
    /// its counters and its `moe=` field all say the arm is on. The other is a
    /// balancer with no selection to balance.
    #[test]
    fn moe_topk_that_cannot_fill_is_refused_and_the_defaults_are_off() {
        // The default is OFF, which is what keeps every preset's parameters
        // and checkpoint byte-identical to a build from before this field.
        let d = DormouseConfig::default();
        assert_eq!((d.moe_topk, d.moe_lb_coef), (0, 0.0), "the routing arm is off by default");

        // `topk > bank` is the collapse: refuse it, naming the field and the
        // escape.
        let err = set_then_validate(&["moe_topk=5"])
            .expect_err("a top-k larger than the expert bank must be refused");
        assert!(err.contains("moe_topk"), "the error must name the field: {err}");
        assert!(err.contains("DENSE"), "the error must name what it would silently run: {err}");

        // A balancer with no selection: refuse rather than ignore.
        let err = set_then_validate(&["moe_lb_coef=0.01"])
            .expect_err("a balancer with routing off must be refused");
        assert!(err.contains("moe_lb_coef"), "the error must name the field: {err}");
        assert!(set_then_validate(&["moe_lb_coef=-0.1"]).is_err(), "a negative coefficient is a sign flip");

        // ...and the collapse the other way: routing with NO balancer. This is
        // LEGAL, and the reason it is not a refusal is the argument in the
        // function body: it is the arm's own removal (the A/B control), and a
        // refusal would force a coefficient the sweep measures as useless at
        // our token count. The collapse stays VISIBLE through
        // `probe::MOE_LB == 0` while `probe::MOE_ROUTE > 0`.
        assert!(
            set_then_validate(&["n_experts=4", "moe_topk=1"]).is_ok(),
            "routing with no balancer is the arm's own removal and must be runnable"
        );

        // The legal range is untouched: this refuses the two collapses that are
        // silently different networks, not the configurations that are merely
        // bad ideas.
        assert!(set_then_validate(&["n_experts=4", "moe_topk=1", "moe_lb_coef=0.01"]).is_ok(), "the FIRST configuration: 4 experts, top-1, balanced");
        assert!(set_then_validate(&["n_experts=4", "moe_topk=2", "moe_lb_coef=0.01"]).is_ok(), "the SECOND configuration: top-2 at the same active FFN compute");
        assert!(set_then_validate(&["n_experts=4", "moe_topk=4", "moe_lb_coef=0.01"]).is_ok(), "top-k == n_experts is the dense blend spelled out loud");
        assert!(set_then_validate(&[]).is_ok(), "the shipped defaults");
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
