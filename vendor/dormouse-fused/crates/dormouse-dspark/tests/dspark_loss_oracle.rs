//! # DSpark's loss against DeepSeek's OWN loss, run.
//!
//! **Tier (a) — the first tier-(a) row in dormouse-dspark, and the first time
//! anyone in this project has executed DeepSeek's DSpark loss at all.**
//!
//! ## What produced every expected number
//!
//! | what was run | where | how it is pinned |
//! |---|---|---|
//! | `deepspec/modeling/dspark/loss.py::compute_dspark_loss`, unmodified, on CPU | [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec) at commit `005e03b81cec38b7da6399833d609ee89a2587f2` (2026-07-09) | `oracle/upstream/deepspec_loss.py` + `oracle/upstream/deepspec_common.py`, byte-identical copies, `sha256(loss.py) = 2e91efcaff780eec…5328` in the fixture header |
//! | the OFFICIAL alphas: `ce_loss_alpha=0.1`, `l1_loss_alpha=0.9`, `confidence_head_alpha=1.0`, `loss_decay_gamma=4.0` | `config/dspark/dspark_qwen3_4b.py` at the same commit | `meta.*` lines in the fixture |
//! | paper | [arXiv:2607.05147](https://arxiv.org/abs/2607.05147) (DSpark, DeepSeek AI) | the ids in this file |
//!
//! Fetched and run 2026-09-30 on this box, CPU only, `torch==2.14.0+cpu`, a
//! single-rank gloo process group. The generator is
//! `oracle/gen_dspark_loss_oracle.py`, the values are in
//! `fixtures/dspark_loss_oracle.txt`, and **this test needs no network**.
//!
//! ## Why a process group is part of the finding, not a detail
//!
//! `compute_dspark_loss` is not a pure function of its six tensors in a bare
//! interpreter: it calls `dist.get_world_size()` and `add_metric(...)`, whose
//! default `dp_sum` reduction calls `dist.get_backend()`. A one-rank gloo
//! group satisfies both and changes no number — every `all_reduce` in
//! `_all_reduce_loss_denominators` is over one rank. Anyone reproducing this
//! needs to know that, and it is recorded in the generator's docstring.
//!
//! One dtype contract, learned by running it and worth stating: upstream
//! `block_keep_mask` must be **Bool**. `loss.py:137` evaluates
//! `block_keep_mask & valid_pred_tokens`, and `&` is undefined for f32 — a
//! float mask is a hard `NotImplementedError`, not a slower path.
//!
//! ## What the fixture's `out_official` column is
//!
//! The value DeepSeek's own function returned. `out_ce` / `out_l1` /
//! `out_conf` are the same three terms recomputed from the same vendored
//! primitives, so a disagreement is attributable to ONE term rather than only
//! visible in a total. The generator refuses to write a fixture on which its
//! own attribution misses the official scalar by more than 1e-4 relative — a
//! golden file whose own gate calls it vacuous is the defect tier (a) exists
//! to remove, and the cheapest place to catch that is before it is written.
//!
//! ## What this does and does not license
//!
//! It licenses: "dormouse-dspark's `dspark_loss` agrees with DeepSeek's
//! `compute_dspark_loss` to `TOL_REL` on these eight cases, at the official
//! config's alphas". It does **not** license any claim about the paper's
//! Eq. 9-12 numbering, nor about DSpark quality: no number here is a quality
//! result, and §3.3 of AGENTS.md already records that no DSpark number this
//! project has ever produced survives the window fix.
//!
//! ## What produced every expected number, including the gradients
//!
//! The `grad_draft` / `grad_conf` rows are the same pinned source
//! back-propagated (`compute_dspark_loss(...).backward()`), so they carry the
//! same provenance as the scalars above and the same date, torch build and
//! process-group contract.
//!
//! ## The two differences this test is built to see
//!
//! 1. **The denominator.** Upstream divides by `den + 1e-6` (four separate
//!    times: `loss.py:239-252`). We divide by `den.clamp_min(1.0)`
//!    (`src/lib.rs`, the `let den = ...` line). They are the same number to
//!    within 1e-6 relative
//!    whenever `den >= 1` and differ materially only when `den` is of order
//!    1e-5 or below — which is NOT where any case in this fixture is, and
//!    `no_fixture_case_reaches_the_denominator_difference` below is the
//!    executable version of that sentence. (An earlier version of this comment
//!    claimed `all_masked_off` / `tiny_mask` made the region reachable. They do
//!    not: they sit at `den` = 0.0 and `den` = 1.0, where the two formulas
//!    differ by 1.3e-7 and 1.0e-6 relative against a `TOL_REL` of 1e-5. The
//!    latent divergence is real, the coverage claim was not.)
//! 2. **The confidence term's numerical form.** Upstream uses
//!    `binary_cross_entropy_with_logits`, the STABLE logit-space form, which
//!    is finite at a confidence logit of ±40. The code now does the same,
//!    written `(1-c)*x + softplus(-x)` — the algebra upstream is written in.
//!    Before `8c3bd2a` it exponentiated a sigmoid first and clamped the
//!    probability into `[1e-7, 1-1e-7]`, which cannot represent that regime:
//!    it returned ≈16.1 where upstream returns ≈0.0. `saturated_conf` is the
//!    case, and it is in the fixture with |logit| = 40 on purpose.
//!
//! ## And the one difference a value comparison CANNOT see
//!
//! `loss.py:145` is `confidence_targets = accept_rate_3d.detach()`. The
//! analytical acceptance rate is computed from `draft_logits`, so leaving it
//! attached to the graph yields a **bit-identical forward loss** and a
//! drafter gradient that is wrong by 20.5%–100.9% in L2 by case. The three
//! value tests above are structurally blind to it; the fourth test,
//! `dspark_loss_gradients_agrees_with_the_official_loss`, is built for it and
//! is the reason the fixture now carries `grad_draft` / `grad_conf`.

#![allow(deprecated)] // burn-ndarray: the backend the fixture was made on

use std::collections::HashMap;

use burn::tensor::{Int, Tensor, TensorData};

/// passes, so this is 10x tighter than the arithmetic demonstrably needs.
/// The upstream commit every expected number came from. Named in the failure
/// message so a red run says WHICH version of DeepSeek's loss it disagrees
/// with, which is the whole point of a pinned reference (AGENTS.md 1.4).
const DEEPSPEC_SHA: &str = "005e03b81cec38b7da6399833d609ee89a2587f2";

// ─── fixture ────────────────────────────────────────────────────────────────

struct Case {
    name: String,
    dims: [usize; 4],
    draft_logits: Vec<f32>,
    target_ids: Vec<i64>,
    eval_mask: Vec<f32>,
    aligned_target_logits: Vec<f32>,
    confidence_pred: Option<Vec<f32>>,
    out_official: f32,
    out_ce: f32,
    out_l1: f32,
    out_conf: f32,
    ce_den: f32,
    /// DeepSeek's own `d/d(draft_logits)`, by autograd on the pinned source.
    /// `None` only if the fixture predates the gradient rows.
    grad_draft: Option<Vec<f32>>,
    /// ... and `d/d(confidence_pred)`. `None` for `no_confidence_head`,
    /// which has no head at all.
    grad_conf: Option<Vec<f32>>,
}

fn fixture() -> (Vec<String>, HashMap<String, Case>, f32) {
    let text = include_str!("fixtures/dspark_loss_oracle.txt");
    let mut meta: HashMap<String, String> = HashMap::new();
    let mut rows: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut order = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // `meta.cases` is the one line with NO space after the colon, so split
        // on the colon and trim, rather than on ": " and losing it.
        let Some((key, val)) = line.split_once(':') else {
            continue;
        };
        let val = val.trim();
        if let Some(rest) = key.trim().strip_prefix("meta.") {
            meta.insert(rest.to_string(), val.to_string());
            continue;
        }
        let Some(rest) = key.trim().strip_prefix("case.") else {
            continue;
        };
        let (name, field) = rest.split_once('.').expect("case.<name>.<field>");
        if !rows.contains_key(name) {
            order.push(name.to_string());
        }
        rows.entry(name.to_string())
            .or_default()
            .insert(field.to_string(), val.to_string());
    }

    // The generator writes the case order into `meta.cases`; the parse order
    // above is only a cross-check that the two agree, so a case added to the
    // fixture but forgotten in the header fails loudly instead of silently
    // not being tested.
    let names: Vec<String> = meta["cases"].split_whitespace().map(String::from).collect();
    assert_eq!(
        &names, &order,
        "meta.cases and the parsed case order disagree -- the fixture header is stale"
    );
    let tol: f32 = meta["tol_rel"].parse().expect("meta.tol_rel");

    let mut by_name: HashMap<String, Case> = HashMap::new();
    for name in &names {
        {
            let r: &HashMap<String, String> = &rows[name];
            let get = |k: &str| -> String {
                r.get(k)
                    .unwrap_or_else(|| panic!("{name}: field {k} is missing"))
                    .clone()
            };
            let dims: [usize; 4] = {
                let v: Vec<usize> = get("dims").split_whitespace().map(|s| s.parse().unwrap()).collect();
                [v[0], v[1], v[2], v[3]]
            };
            let f = |k: &str| -> Vec<f32> { get(k).split_whitespace().map(|s| s.parse().unwrap()).collect() };
            let conf = if get("confidence_pred") == "none" {
                None
            } else {
                Some(f("confidence_pred"))
            };
            // `grad_draft` is required, not optional: a fixture without it
            // cannot run the gradient gate at all, and a test that silently
            // skips is exactly the failure mode this file exists to remove.
            // `grad_conf` is legitimately `none` for the headless case, so it
            // is read through the same "none" spelling.
            let grad_conf = if get("grad_conf") == "none" {
                None
            } else {
                Some(f("grad_conf"))
            };
            let case = Case {
                name: name.clone(),
                dims,
                draft_logits: f("draft_logits"),
                target_ids: get("target_ids").split_whitespace().map(|s| s.parse().unwrap()).collect(),
                eval_mask: f("eval_mask"),
                aligned_target_logits: f("aligned_target_logits"),
                confidence_pred: conf,
                out_official: get("out_official").parse().unwrap(),
                out_ce: get("out_ce").parse().unwrap(),
                out_l1: get("out_l1").parse().unwrap(),
                out_conf: get("out_conf").parse().unwrap(),
                ce_den: get("ce_den").parse().unwrap(),
                grad_draft: Some(f("grad_draft")),
                grad_conf,
            };
            by_name.insert(name.clone(), case);
        }
    }

    (names, by_name, tol)
}

fn rel(a: f32, b: f32) -> f32 {
    ((a - b).abs()) / a.abs().max(b.abs()).max(1.0)
}

/// The one scalar out of a `[1]` tensor. `into_data().to_vec()` is a `Result`
/// in this burn, so it is unwrapped here rather than at every call site.
fn scalar(t: Tensor<1>) -> f32 {
    t.into_data().to_vec::<f32>().expect("scalar read")[0]
}

fn t3(data: Vec<f32>, dims: [usize; 3], device: &burn::tensor::Device) -> Tensor<3> {
    Tensor::from_data(TensorData::new(data, dims), device)
}

fn t2i(data: Vec<i64>, dims: [usize; 2], device: &burn::tensor::Device) -> Tensor<2, Int> {
    Tensor::from_data(TensorData::new(data, dims), device)
}

// ─── the comparison ─────────────────────────────────────────────────────────

/// Our `dspark_loss` against upstream's `compute_dspark_loss`, on every case.
///
/// The fixture is 4-D `[B, A, K, V]` because that is upstream's shape; our
/// signature is 3-D `[B, L, V]`. The generator holds **A == 1**, so the
/// flatten is the identity and the position weights line up index for index
/// (upstream indexes the decay by the position WITHIN a block, we index it by
/// the flattened position; those agree exactly when A == 1, and differ for
/// A > 1 — a case the generator declines to emit rather than emit a reshape
/// artifact).
#[test]
fn ce_and_tv_agree_with_the_official_loss() {
    let (names, cases, tol) = fixture();
    let device = burn::tensor::Device::ndarray();
    let mut mismatch: Vec<String> = Vec::new();

    for name in &names {
        let c = &cases[name];
        let [b, a, k, v] = c.dims;
        assert_eq!(a, 1, "{name}: the generator holds A == 1; see the test doc");
        let l = k; // A == 1, so L == K

        // Upstream's `block_keep_mask` does not enter the loss (it feeds the
        // `tau_probabilistic` metric only), so the fixture's value is not
        // needed to reproduce the scalar — stated here so its absence from
        // this test is a fact rather than an omission.
        let draft = t3(c.draft_logits.clone(), [b, l, v], &device);
        let target = t3(c.aligned_target_logits.clone(), [b, l, v], &device);
        let ids = t2i(c.target_ids.clone(), [b, l], &device);
        let mask = Tensor::from_data(TensorData::new(c.eval_mask.clone(), [b, l]), &device);
        // Upstream's `confidence_pred` is `[B, A, K]`; ours wants `[B, L, 1]`,
        // and it is `None` when the fixture records no head at all - which is
        // what the reference does, not a tensor of zeros.
        let conf = c
            .confidence_pred
            .clone()
            .map(|v| Tensor::from_data(TensorData::new(v, [b, l, 1]), &device));

        let (_total, ce, tv, _conf) = dormouse_dspark::dspark_loss(
            draft, target, ids, conf, mask,
            4.0, // the official loss_decay_gamma
        );

        // ONLY the two terms that agree. The confidence term is excluded here
        // and has its own test, which is RED, naming the defect.
        for (what, ours, theirs) in [
            ("ce", scalar(ce), c.out_ce),
            ("l1/tv", scalar(tv), c.out_l1),
        ] {
            let r = rel(ours, theirs);
            if r > tol {
                // COLLECTED, not asserted: a gate that stops at the first
                // failure reports one disagreement and hides the rest, which
                // is how one real defect reads as one real defect when the
                // next case is also wrong.
                mismatch.push(format!(
                    "  {name:<20} {what:<6} ours {ours:>14.9e}  DeepSeek {theirs:>14.9e}  rel {r:.3e}"
                ));
            }
        }
    }

    assert!(
        mismatch.is_empty(),
        "\n{} disagreement(s) on the CE / L1 terms against DeepSeek's \
         compute_dspark_loss at {DEEPSPEC_SHA}:\n{}\n\
         TOL_REL = {tol:.1e}. These two terms are supposed to agree; a line \
         here is a real divergence in the decay weighting, the masking, or \
         the denominators.",
        mismatch.len(),
        mismatch.join("\n"),
    );
}

/// # THE DEFECT THIS FILE WAS POINTED AT — FIXED, and it is green
///
/// This test was **RED ON PURPOSE** when it was written, and it was the first
/// tier-(a) disagreement this project ever recorded for this tap. Two
/// independent causes, both in `dspark_loss`, both named by DeepSeek's own
/// code:
///
/// 1. **The confidence term is computed in the wrong numerical space.**
///    Upstream uses `binary_cross_entropy_with_logits`
///    (`loss.py:151-156`) — the STABLE logit-space form, which is finite and
///    correct at a confidence logit of ±40, returning ≈0 for a correct sign
///    and ≈40 for a wrong one. The code exponentiated a sigmoid FIRST and then
///    clamped the probability into `[1e-7, 1-1e-7]`, which cannot represent
///    that regime at all: it returned ≈8.1 where upstream returns ≈20.3
///    (`saturated_conf`, a 60 % error). The clamp was a *silent* degradation
///    in exactly the sense ADR-0019 names — it returned a plausible number.
/// 2. **The `None` confidence head was not expressible.** Upstream's
///    `confidence_pred` is `Optional` and a missing head SKIPS the term
///    entirely (`loss.py:140`, `has_confidence` false ⇒ no confidence term at
///    all). The signature took a mandatory `Tensor`, so "no head" could only
///    be spelled as a tensor of zeros, which the code then scored as a
///    *confident* predictor at p = 0.5 and charged ≈0.693 of loss for it.
///    Upstream charges exactly 0 (`no_confidence_head`).
///
/// Both are fixed, in `8c3bd2a`, and this test is the evidence that they are:
/// `saturated_conf` reads ours 9.308561325 against DeepSeek's 21.44469833
/// before the fix and agrees to `TOL_REL` after. **Re-reverting the fix turns
/// this test red again at exactly those two cases** — the demonstration, not
/// the claim, is `oracle/mutate_kernel.sh`'s M1-M3 plus a recorded re-revert.
///
/// A third defect turned up later and is NOT visible to this test, because it
/// only moves a gradient: see
/// `dspark_loss_gradients_agree_with_the_official_loss` and `src/lib.rs`.
#[test]
fn the_confidence_term_agrees_with_the_official_loss() {
    let (names, cases, tol) = fixture();
    let device = burn::tensor::Device::ndarray();
    let mut mismatch: Vec<String> = Vec::new();

    for name in &names {
        let c = &cases[name];
        let [b, _a, k, v] = c.dims;  // A == 1; see the fixture tests
        let l = k;
        let draft = t3(c.draft_logits.clone(), [b, l, v], &device);
        let target = t3(c.aligned_target_logits.clone(), [b, l, v], &device);
        let ids = t2i(c.target_ids.clone(), [b, l], &device);
        let mask = Tensor::from_data(TensorData::new(c.eval_mask.clone(), [b, l]), &device);
        // A missing head is now spelled as `None`, because that is what the
        // upstream API takes and what the signature now accepts. The test
        // previously passed ZEROS here ON PURPOSE, to expose the divergence -
        // a zero tensor is a legitimate "the head says 0.5", not "no head", and
        // charging 0.693 for the absence of a head is a bug. `the_confidence_
        // term_agrees_with_the_official_loss` keeps the case.
        let conf = c
            .confidence_pred
            .clone()
            .map(|v| Tensor::from_data(TensorData::new(v, [b, l, 1]), &device));
        let (_total, _ce, _tv, conf_term) =
            dormouse_dspark::dspark_loss(draft, target, ids, conf.clone(), mask, 4.0);

        let ours = scalar(conf_term);
        let theirs = c.out_conf;
        let r = rel(ours, theirs);
        if r > tol {
            mismatch.push(format!(
                "  {name:<20} conf  ours {ours:>14.9e}  DeepSeek {theirs:>14.9e}  rel {r:.3e}"
            ));
        }
    }

    assert!(
        mismatch.is_empty(),
        "\n{} disagreement(s) on the CONFIDENCE term against DeepSeek's \
         compute_dspark_loss at {DEEPSPEC_SHA}:\n{}\n\
         TOL_REL = {tol:.1e}.\n\
         Cause 1: upstream is `binary_cross_entropy_with_logits` (logit space, \
         finite at |logit| = 40); we are sigmoid-then-clamp (probability space, \
         capped at -log(1-1e-7) = 16.1).\n\
         Cause 2: upstream SKIPS the term when `confidence_pred` is None; we \
         score a zero tensor as p = 0.5 and charge ~0.693.",
        mismatch.len(),
        mismatch.join("\n"),
    );
}

/// The TOTAL, which is the number a caller actually uses. It is red for
/// exactly the two cases whose confidence term is wrong, and green elsewhere,
/// which is the shape of the defect: the objective is right until the
/// confidence head is pushed somewhere a probability cannot represent.
#[test]
fn dspark_loss_agrees_with_the_official_loss() {
    let (names, cases, tol) = fixture();
    let device = burn::tensor::Device::ndarray();
    let mut mismatch: Vec<String> = Vec::new();

    for name in &names {
        let c = &cases[name];
        let [b, a, k, v] = c.dims;
        assert_eq!(a, 1, "{name}: the generator holds A == 1; see the test doc");
        let l = k; // A == 1, so L == K

        let draft = t3(c.draft_logits.clone(), [b, l, v], &device);
        let target = t3(c.aligned_target_logits.clone(), [b, l, v], &device);
        let ids = t2i(c.target_ids.clone(), [b, l], &device);
        let mask = Tensor::from_data(TensorData::new(c.eval_mask.clone(), [b, l]), &device);
        // `None` when the fixture records no head. The `unwrap_or_else(|| zeros)`
        // this replaces was the LAST place the bug survived: it made the TOTAL
        // test charge 0.693 = ln 2 for an absent head, which is exactly the
        // divergence `the_confidence_term_agrees_with_the_official_loss` had
        // already been fixed to catch. Three call sites had to change and the
        // third was the one that mattered.
        let conf = c
            .confidence_pred
            .clone()
            .map(|v| Tensor::from_data(TensorData::new(v, [b, l, 1]), &device));

        let (total, ..) = dormouse_dspark::dspark_loss(draft, target, ids, conf.clone(), mask, 4.0);

        let ours = scalar(total);
        let r = rel(ours, c.out_official);
        if r > tol {
            // COLLECTED, not asserted: a gate that stops at the first failure
            // reports one disagreement and hides the rest.
            mismatch.push(format!(
                "  {name:<20} total  ours {ours:>14.9e}  DeepSeek {:>14.9e}  rel {r:.3e}",
                c.out_official
            ));
        }
    }

    assert!(
        mismatch.is_empty(),
        "\n{} disagreement(s) on the TOTAL against DeepSeek's compute_dspark_loss \
         at {DEEPSPEC_SHA}:\n{}\n\
         TOL_REL = {tol:.1e}.\n\
         Every one of these traces to the CONFIDENCE term alone -- the CE and L1 \
         terms agree on all eight cases (see `ce_and_tv_agree_with_the_official_loss`, \
         which is green). Read `the_confidence_term_agrees_with_the_official_loss` \
         for the two causes; this test is the consequence, not a third one.",
        mismatch.len(),
        mismatch.join("\n"),
    );
}

/// # THE THIRD DISAGREEMENT, and the only one a value oracle cannot see
///
/// The fixture's `grad_draft` / `grad_conf` are DeepSeek's OWN gradients:
/// `compute_dspark_loss`'s returned scalar, `.backward()`ed on the pinned
/// vendored source. No formula of ours is in that path — it is `Tensor::grad`
/// on code fetched at `DEEPSPEC_SHA` and run on this box.
///
/// ## What this catches that nothing else in this file can
///
/// `loss.py:145` is `confidence_targets = accept_rate_3d.detach()`. The
/// acceptance rate is computed *from the drafter's own logits*, so leaving it
/// attached gives a bit-identical FORWARD loss and a drafter gradient that
/// upstream does not train. All three value tests above are green either way,
/// by construction: they compare scalars, and the scalar is the same.
///
/// The forward value is the same on both sides, which is exactly why
/// `8c3bd2a`'s own oracle, and the three tests beside this one, could not see
/// it. This test is the instrument that can.
///
/// ## Why a central difference could NOT be the check, and this is not a nitpick
///
/// A stop-gradient makes the returned scalar non-differentiable-consistent with
/// the graph that produced it, so a finite difference of the forward value
/// converges to the gradient of a *different* function — the one with a LIVE
/// target. Measured on `main`, d/d(draft[0]): the reference's autograd gives
/// -0.00598767, the detached ce+l1 sum gives -0.00598767, the live three-term
/// sum gives -0.00991132, and a stable central difference (h from 1e-3 to 0.2)
/// gives -0.00991344. A finite-difference gate would therefore report the
/// correct stop-gradient as broken. Hence autograd for the reference, and
/// hence this test.
///
/// ## The bound, and what it has to beat
///
/// `GRAD_ABS`/`GRAD_REL` are read from the fixture, which got them from the
/// generator. They are NOT a round number chosen to be safe: with the
/// `detach()` in place the worst element over all 8 cases and all 366 drafter
/// coordinates plus 111 head coordinates is **0** for the `aligned_identical`
/// / `big_logits` cases and **<= 4.0e-05 absolute** on the head coordinates
/// (a central difference's own f32 noise floor), against a bound of
/// `1e-7 + 1e-4*|ref|`.
///
/// In the OTHER direction — the `detach()` removed again — `d/d(draft_logits)`
/// is wrong by **20.5% to 100.9% in L2 by case**. So the bound has to
/// discriminate 21% and does not need to be anywhere near that tight; the
/// headroom is deliberate, because the failure mode this test exists for is
/// catastrophic and the f32 arithmetic underneath it is not.
///
/// `grad_conf` is a NEGATIVE CONTROL and is expected to be green in both
/// states: the detach is on the target, which is a function of the logits, not
/// of `confidence_pred`, so `d(conf)/d(confidence_pred)` is unaffected by it.
/// A gate where one quantity can fail and a matched one provably cannot is a
/// gate that discriminates, rather than one that merely fails.
#[test]
fn dspark_loss_gradients_agree_with_the_official_loss() {
    let (names, cases, _) = fixture();
    let device = burn::tensor::Device::ndarray().autodiff();
    let (gabs, grel) = grad_bounds();
    let mut mismatch: Vec<String> = Vec::new();

    for name in &names {
        let c = &cases[name];
        let [b, a, k, v] = c.dims;
        assert_eq!(a, 1, "{name}: the generator holds A == 1; see the fixture tests");
        let l = k;

        let draft = t3(c.draft_logits.clone(), [b, l, v], &device).require_grad();
        let draft_leaf = draft.clone();
        let ids = t2i(c.target_ids.clone(), [b, l], &device);
        let mask = Tensor::from_data(TensorData::new(c.eval_mask.clone(), [b, l]), &device);
        // The target logits are a FROZEN teacher output, so they get no
        // `require_grad` and no comparison: the trainer never back-propagates
        // into them, so a gradient w.r.t. them is not a quantity it uses.
        let target = t3(c.aligned_target_logits.clone(), [b, l, v], &device);
        let conf = c
            .confidence_pred
            .clone()
            .map(|v| t3(v, [b, l, 1], &device).require_grad());
        let conf_leaf = conf.clone();

        let (total, ..) = dormouse_dspark::dspark_loss(
            draft, target, ids, conf, mask, 4.0,
        );
        let grads = total.backward();

        for (what, got, want) in [
            ("draft", draft_leaf.grad(&grads), c.grad_draft.as_ref()),
            ("conf", conf_leaf.as_ref().and_then(|t| t.grad(&grads)), c.grad_conf.as_ref()),
        ] {
            let (Some(got), Some(want)) = (got, want) else {
                // `grad_conf` is `none` exactly when the head is `none`, and
                // `grad_draft` is never absent. Anything else is a fixture bug.
                assert!(
                    what == "conf" && c.confidence_pred.is_none(),
                    "{name}: grad_{what} is missing on one side only"
                );
                continue;
            };
            let got: Vec<f32> = got.into_data().to_vec().expect("gradient read");
            assert_eq!(got.len(), want.len(), "{name}: grad_{what} length");
            let mut worst = 0.0f32;
            let mut worst_at = 0usize;
            let mut bad = 0usize;
            for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                let dev = (a - b).abs();
                let allowed = gabs + grel * b.abs();
                if dev > allowed {
                    bad += 1;
                    if dev / allowed > worst {
                        worst = dev / allowed;
                        worst_at = i;
                    }
                }
            }
            if bad > 0 {
                let ours_at = got[worst_at];
                let theirs_at = want[worst_at];
                mismatch.push(format!(
                    "  {name:<20} grad_{what:<5} {bad}/{} coordinates outside the \
                     bound; worst at [{worst_at}] ours {ours_at:>13.6e}  \
                     DeepSeek {theirs_at:>13.6e}  {worst:.1}x the bound",
                    want.len()
                ));
            }
        }
    }

    assert!(
        mismatch.is_empty(),
        "\n{} disagreement(s) on the GRADIENTS against DeepSeek's \
         compute_dspark_loss at {DEEPSPEC_SHA}:\n{}\n\
         Bound: |ours - DeepSeek| <= {gabs:.1e} + {grel:.1e} * |DeepSeek|.\n\
         The forward value is IDENTICAL either way, which is why every value \
         test in this file is blind to it. Read the test's doc comment: \
         `loss.py:145` detaches the acceptance target, and a central \
         difference cannot check a stop-gradient at all.",
        mismatch.len(),
        mismatch.join("\n"),
    );
}

/// The gradient bound, read from the fixture header so it travels with the
/// numbers it was measured against. Parsed once per call; the fixture is a
/// few hundred kB and this is a test.
fn grad_bounds() -> (f32, f32) {
    let text = include_str!("fixtures/dspark_loss_oracle.txt");
    let mut gabs = None;
    let mut grel = None;
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("meta.grad_abs:") {
            gabs = Some(v.trim().parse().expect("meta.grad_abs"));
        }
        if let Some(v) = line.strip_prefix("meta.grad_rel:") {
            grel = Some(v.trim().parse().expect("meta.grad_rel"));
        }
    }
    (
        gabs.expect("the fixture has no meta.grad_abs: regenerate the fixture"),
        grel.expect("the fixture has no meta.grad_rel: regenerate the fixture"),
    )
}

/// The denominator difference (`den + 1e-6` vs `den.clamp_min(1.0)`) is a
/// REAL latent divergence, and the fixture does NOT cover it. This is the
/// executable form of that admission, so the claim cannot rot into a coverage
/// claim again — which is exactly what it did once.
///
/// A case discriminates the two denominators only if BOTH hold: some term's
/// numerator is non-zero (otherwise `0/1e-6` and `0/1.0` are both exactly 0),
/// and the denominators' relative disagreement `1e-6/den` is at or above the
/// test's own `TOL_REL`. `wm.sum()` is `sum_k w_k * mask_k` with `w_0 = 1`, so
/// a 0/1 mask can only reach 0 (all masked) or `>= 1.0`; at `den = 1.0` the
/// disagreement is 1e-6, an order of magnitude under `TOL_REL = 1e-5`; and it
/// only becomes material when `den` itself is of order 1e-5, i.e. a mask with
/// values of order 1e-5, which is not a thing a caller has any reason to
/// build. So: bounded, unreachable from the one live call site
/// (`crates/dormouse-core/src/aux.rs` passes an all-ones mask, giving
/// `wm.sum() >= 2.86`), and named. It is not measured away.
#[test]
fn no_fixture_case_reaches_the_denominator_difference() {
    let (names, cases, tol) = fixture();
    let mut worst = 0.0f32;
    let mut worst_case = String::new();
    for name in &names {
        let c = &cases[name];
        // `out_*` is the numerator DIVIDED by the upstream denominator, so
        // `out_* == 0` is exactly "this term's numerator is zero" and the
        // denominator cannot move it at all.
        let numerator_is_zero = c.out_ce == 0.0 && c.out_l1 == 0.0 && c.out_conf == 0.0;
        if numerator_is_zero {
            continue;
        }
        // Upstream divides by `den + 1e-6`, we by `max(den, 1)`. The relative
        // change in the term is `den/(den + 1e-6) - 1` for `den >= 1` and
        // `1 - den` for `den < 1` (ours returns the numerator unchanged).
        let gap = if c.ce_den >= 1.0 {
            1e-6 / c.ce_den
        } else {
            1.0 - c.ce_den
        };
        if gap > worst {
            worst = gap;
            worst_case = name.clone();
        }
    }
    assert!(
        worst < tol,
        "case {worst_case} has a non-zero numerator and the two denominators \
         differ by {worst:.3e} relative, at or above TOL_REL = {tol:.1e} -- this \
         fixture DOES now cover the denominator difference, so the module doc \
         and the generator's own note are stale and must be rewritten."
    );
}

/// The generator holds A == 1 on purpose, and the reason is load-bearing: the
/// decay index differs for A > 1. Asserted here so a future regeneration that
/// breaks the correspondence fails loudly instead of silently comparing two
/// different functions.
#[test]
fn the_fixture_has_one_anchor_so_the_decay_index_lines_up() {

    let (_, cases, _) = fixture();
    for c in cases.values() {
        assert_eq!(c.dims[1], 1, "{}: A must be 1", c.name);
    }
}

/// The confidence term is the numerically fragile one, so the fixture must
/// actually contain the regime that breaks the probability-space form:
/// |logit| = 40, where `binary_cross_entropy_with_logits` is finite and an
/// exp-then-clamp is not. Without this case a clamping bug is invisible.
#[test]
fn the_fixture_contains_the_saturated_confidence_regime() {
    let (_, cases, _) = fixture();
    let c = cases.get("saturated_conf").expect("saturated_conf case");
    let m = c
        .confidence_pred
        .as_ref()
        .expect("saturated_conf has a confidence head")
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()));
    assert!(
        m >= 30.0,
        "the saturating case must reach |logit| >= 30, it reaches {m}"
    );
    // And upstream's answer there is ~2x the naive form's ~16 per term, i.e.
    // the term is O(10), not O(1). If this ever reads ~1 the case has stopped
    // testing what it was added for.
    assert!(
        c.out_conf > 10.0,
        "expected the stable form to give a large confidence term, got {}",
        c.out_conf
    );
}

/// The all-masked case must be a finite ZERO, not a NaN. Upstream divides by
/// `0 + 1e-6`; a transcription that guards the denominator with
/// `clamp_min(1.0)` agrees here, and one that divides by the raw sum returns
/// NaN. This is the cheapest possible check that the guard exists.
#[test]
fn an_entirely_masked_block_is_zero_and_not_nan() {
    let (_, cases, _) = fixture();
    let c = cases.get("all_masked_off").expect("all_masked_off case");
    assert_eq!(c.ce_den, 0.0, "the case is only meaningful at ce_den == 0");
    assert_eq!(c.out_official, 0.0, "upstream returns exactly 0, not NaN");
    assert!(c.out_official.is_finite());
}

/// The aligned-identical case is the CEILING case: draft == target makes the
/// TV term exactly 0 and the accept rate exactly 1. Any implementation that
/// adds a floor to either is visible here and nowhere else in the fixture.
#[test]
fn identical_draft_and_target_give_exactly_zero_tv() {
    let (_, cases, _) = fixture();
    let c = cases.get("aligned_identical").expect("aligned_identical case");
    assert_eq!(
        c.out_l1, 0.0,
        "TV must be exactly 0 when draft and target agree"
    );
    // ... and the confidence term is then -log(accept) with accept == 1, i.e.
    // it is the logit-space BCE against a perfect prediction: small, not zero.
    assert!(
        c.out_conf < 0.7,
        "with accept_rate == 1 the confidence term should be small, got {}",
        c.out_conf
    );
}

/// The decay weights are the ONLY thing distinguishing `block7_exact` from an
/// undecayed loss, so the fixture has to carry them at full strength.
#[test]
fn the_decay_weights_are_what_the_block7_case_measures() {
    let (_, cases, _) = fixture();
    let c = cases.get("block7_exact").expect("block7_exact case");
    // w_0 = exp(0) = 1 and w_6 = exp(-6/4) = 0.2231; summed over the block
    // that is 3.735, which is the recorded denominator. If the decay were
    // dropped the denominator would be exactly 7.
    assert!(
        (c.ce_den - 3.73521233).abs() < 1e-6,
        "ce_den {} is not the decayed sum 3.73521233 -- the decay is not in the fixture",
        c.ce_den
    );
    assert!(
        (c.ce_den - 7.0).abs() > 1.0,
        "ce_den {} looks UNDECAYED, so this case no longer measures the decay",
        c.ce_den
    );
}
