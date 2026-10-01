//! The coherence check: every rule a merged [`DormouseConfig`] has to satisfy
//! before a model is built from it (ADR-0005).
//!
//! ONE FUNCTION, and that is the design. The rules are cross-field (a
//! top-k that must fit the bank, a horizon that must fit the window, a
//! capacity that must stay a minority of the model), so they cannot live on
//! the fields, and splitting them across call sites is how a rule ends up
//! checked on the training path and not on the eval path. `resolve` calls
//! [`validate`] once, at the end of the merge.
//!
//! WHAT THE RULES ARE FOR. Nearly all of them exist because the wrong value
//! computes a DIFFERENT NETWORK and says nothing about it (ADR-0019). The four
//! shapes, with the check that catches each:
//!
//! 1. **A shape error a long way from the flag.** `act_group` and
//!    `mhc_streams` must divide `d_model`, or the first forward's reshape
//!    fails; `engram_orders` must have exactly 3 entries, or the trainer's
//!    `[b, t, 3]` hash tensor is a shape mismatch. These are LOUD rather than
//!    clamped because a clamped divisor trains a mechanism the config does not
//!    name.
//! 2. **A silent collapse into the control.** `moe_topk > n_experts` makes the
//!    top-k set the dense blend, and `dspark_stride = 0` makes the k-step
//!    draft window k copies of one cross-entropy: the run trains, the loss
//!    descends, and the config no longer describes the objective. Refused,
//!    naming the escape.
//! 3. **A term with nothing to act on.** `moe_lb_coef > 0` with
//!    `moe_topk = 0` is a load-balancing loss with no selection to balance -
//!    a field that reads like an objective and contributes nothing, the
//!    `probe::JEPA`-never-bumped shape this repo has already been bitten by.
//! 4. **Two arms claiming one statement.** AttnRes, Gated Residual and mHC all
//!    REPLACE the loop's residual accumulation, in the same `else` chain, so
//!    any two on means one is silently ignored. One check over all three, so
//!    adding a fourth cannot leave a pair unguarded.
//!
//! WHAT IS DELIBERATELY NOT A REFUSAL, and why: `moe_topk > 0` with
//! `moe_lb_coef == 0`. A collapsed router is a real risk, but that
//! configuration IS the arm's own removal under the A/B rule (AGENTS.md §1.2),
//! the published coefficient range is three orders of magnitude too weak at our
//! token count (so a refusal would manufacture a "copied a large-MoE number"
//! defect), and `probe::MOE_ROUTE`/`MOE_LB` already make the state VISIBLE on
//! the eval line as `moe=<lb>/<sel>` - the repo's own COUNTED mark, which is the
//! correct instrument for a legal run.
//!
//! NOT CHECKED HERE, and each for a reason: values the loop can adapt to
//! (`max_iter`, `rank`), values with a per-layer effect and no config-level
//! rule (`lr`, `wd`, `grad_clip` - the train layer's, not the model's), and
//! anything the filesystem owns (a preset that does not exist - that is
//! [`super::load_config`]'s error).
//!
//! COST. ~30 scalar comparisons, no allocation: nothing next to a 0.8 s step,
//! and it is paid once per run.

use super::schema::DormouseConfig;
#[cfg(test)]
use super::schema::ActQuant;

/// Check every cross-field rule the model cannot enforce for itself. `Ok(())`
/// means the config is coherent, not that it is GOOD - validation has no
/// opinion on a learning rate.
///
/// Every `Err` names the field and, where a wrong value would compute a
/// different objective rather than crash, the escape. Called once by
/// `dormouse_train::resolve` after the merge; the model constructors do not
/// call it, because a library that validates its own input on every call
/// cannot be used to reproduce a run that predates a new rule.
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
    //    the sweep (docs/reviews/moe-routing-2026-10-01.md §5) measures
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
