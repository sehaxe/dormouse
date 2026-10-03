//! Optimizer policy: Muon+ (burn-fused) with the Qwen3.8-Flash-Next §3.1
//! param routing.
//!
//! **The policy lives in [`dormouse_core::routing`]**, next to the arms it
//! names, and this file only ASSEMBLES the optimizer. It used to keep a
//! SECOND copy here: a table of path markers (`"expert_ffns."`,
//! `"gdn2.q_proj"`, ...) re-derived from the module field names. The two
//! copies did disagree, and the disagreement was load-bearing - the marker
//! set excluded `inner.Dense.bias` but not `inner.Dense.weight`, so with
//! `--set use_tsct=false` a dense expert fed its full `[d,d]` weight to Muon+
//! and fp32 Newton-Schulz, the ~40 s/step case the declaration exists to avoid.
//! The only count in the log was the marker's own, so nothing said so.
//!
//! That copy is GONE (`831e3a0`, 2026-09-28) — grep finds no marker symbol in
//! the tree. An older note here said the path markers were what ran and that
//! `routing.rs` was "exercised only by tests"; that was true before the cut
//! and is false now, in this direction and in every document that repeats it.
//! `docs/reviews/dedup-optimizer-2026-10-01.md`.
//!
//! What is here now:
//! - [`optimizer_groups`] turns the declaration into the three
//!   [`ParamGroup`]s the optimizer installs, by id. A rename cannot move a
//!   parameter between groups, and a parameter nothing declared is a startup
//!   error ([`routing::Routing::check`]).
//! - [`check_installed`] runs the INSTALLED groups against the live module
//!   tree: a parameter in no group, in two, or a 1D parameter in Muon+ is a
//!   loud error naming the path. This is the check with teeth, because it
//!   asks the objects the optimizer actually holds instead of re-deriving the
//!   policy a third time.
//!
//! Which group is which (report §3.1):
//! - **Muon+ ColRow** on the small matrices that genuinely act as linear maps:
//!   the Engram key projections and the low-rank TSCT `u`/`v` factors.
//! - **Head-wise Muon+** on the attention Q/K weights (report §3.1: split qkv
//!   per head BEFORE orthogonalization - fusing mixes singular directions):
//!   each `[head_dim, d]` block gets its own NS preconditioner
//!   ([`HeadWiseMuon`]). Installed only when `qk_heads` is set (the train
//!   loop derives it from the preset's `n_heads`); unset, Q/K train on the
//!   base optimizer.
//! - **Plain Adam, weight decay disabled** on the n-gram tables (§2.3).
//! - **The base optimizer** (AdamW / Adan per `--opt`) on everything else:
//!   embeddings, output head, routers/scorers, per-head scalar producers
//!   (KDA decay/β gates), conv kernels, and every dense `[m,n]` leaf - a dense
//!   linear IS the expensive NS case.
//!
//! Selection via `--opt` (runtime switch, no rebuild):
//! - `mix` (default): the groups above (AdamW fallback).
//! - `mix-adan`: same groups, but the fallback group runs Adan
//!   (arXiv 2208.06677: Nesterov momentum + gradient-difference correction,
//!   the "AdamW+" already shipped in burn-optim).
//! - `adan`: Adan on every param.
//! - `adamw`: legacy AdamW on every param (A/B baseline).
//! - `muon`: Muon+ on every param (its own AdamW fallback for 1D); debug
//!   mode, not the report recipe. No groups are installed - the mode already
//!   says "Muon on everything".
//!
//! `--factors-fallback` additionally drops the expert TSCT factors from the
//! Muon+ group. That is a [`routing::group_of`] branch, not a group to filter
//! here: the declaration has to be the only place a group is decided.

use burn::{
    grad_clipping::GradientClippingConfig,
    module::ParamGroup,
    optim::{AdamConfig, AdamWConfig, AdanConfig, LearningRate, Optimizer},
    tensor::{Device, ElementConversion, Tensor},
};
use burn_muon_plus::{MuonPlus, MuonPlusConfig, MuonPlusState, NormDir};
use dormouse_core::routing::{self, Group};
// Re-exported, not re-declared: core owns this type and its field docs (see
// `GroupCounts` there), and `lib.rs` re-exports it from this module, so it has
// to stay public through here.
pub use dormouse_core::routing::GroupCounts;
use dormouse_core::DormouseModel;
use std::sync::atomic::Ordering::Relaxed;

use crate::{Optim, TrainCfg};

/// How many times a fused Muon+ CUDA kernel was asked for and answered with
/// the tensor-ops path instead, as `(momentum, finalize)` summed over BOTH
/// implementations of the Muon update: [`HeadWiseMuon`] here, and
/// `MuonPlus::step` in the crate.
///
/// The sum is the point. This counter used to live only in `HeadWiseMuon`, so
/// the `muon_skipped=mom/finalize` field on the eval line counted the two Q/K
/// weights and said nothing about the much larger `Group::Muon` population
/// (the TSCT factors and the Engram key projections), whose fused path could
/// have been dead for an entire run with a log that read the same. Both
/// implementations now increment the same statics, and the field is the
/// optimizer's, not one group's.
///
/// `false` from `momentum_cuda`/`finalize_cuda` means "not a bare cubecl
/// tensor" or "empty", and the tensor path computes the same function (ADR-0019;
/// the head-wise Q/K group is ON in every run, `qk_heads` is always resolved).
pub fn fused_kernels_skipped() -> (u64, u64) {
    (
        SKIPPED_MOMENTUM.load(Relaxed) + burn_muon_plus::fused_skipped().0,
        SKIPPED_FINALIZE.load(Relaxed) + burn_muon_plus::fused_skipped().1,
    )
}
/// Present on every build so [`fused_kernels_skipped`] has one answer: with
/// no cuda feature the fused kernels are never asked, and `(0, 0)` says so.
static SKIPPED_MOMENTUM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SKIPPED_FINALIZE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Newton-Schulz iterations for Muon+ orthogonalization.
///
/// **8, and the paper says 5.** 2602.21545 §3, verbatim: "Note that, all the
/// experiments in this paper use **5 iterations** in `Ortho(·)`". §3.4
/// repeats "all with 5 iterations" for the polar-method ablation and Table 7's
/// caption repeats it again, so 5 is not a default that drifted - it is the
/// one value every table in the paper was produced with. There is no 8 in
/// 2602.21545.
///
/// 8 is attributed to the Qwen3.8-Flash-Next report §3.1 (`ns_steps=8` as its
/// stability choice), which is a different document, not a reference
/// implementation, so it does not discharge a bit-for-bit check against the
/// paper.
///
/// What the paper does supply is the *direction*: §2.1, "In principle, one
/// may mitigate the imbalance by running a large number of Newton–Schulz
/// iterations, but this is computationally infeasible in LLM pre-training",
/// and Fig. 2(a) shows the variance still amplified at 5. So 8 is "unrun",
/// not "unsupported" — the paper argues for more iterations and declines to
/// pay for them.
///
/// CHANGING THIS IS THE OWNER'S CALL, not a code cleanup, and it is not free:
/// the optimizer is ~19% of a warm 245 ms step at 9.2M params, 8 of which are
/// this loop, so 8 → 5 is ~-15% of `opt` and ~-2.8% of a step; and it
/// invalidates the only provenance the routing policy has, because
/// `routing.rs:78-80` justifies keeping Muon+ off every `[d,d]` projection
/// with "8 NS iters on the `[d,d]` projections added ~40 s/step" — a number
/// with no config, date or commit on it (ADR-0020 rule 1) that is entangled
/// with this constant. Re-measuring what a change here costs: the step-time
/// row in `benches/history.tsv` at a stated step index, plus a held-out BPB A/B
/// at one batch size, because a different NS depth is a different optimizer,
/// not a speed knob.
pub const MUON_NS_STEPS: usize = 8;
/// Post-polar normalization direction: column-then-row, 2602.21545 Eq. (7).
///
/// One of the paper's two orders, not "the paper's best": §3.4 claims only
/// that bi-directional beats single-directional, and Table 6's two orders are
/// within noise of each other with the winner flipping by model. The paper's
/// own code default is the narrower `Norm_(col)` (App. C, `d="col"`).
pub const MUON_NORM_DIR: Option<NormDir> = Some(NormDir::ColRow);

fn muon_plus_cfg(cfg: &TrainCfg) -> MuonPlusConfig {
    MuonPlusConfig::new()
        .with_norm_dir(MUON_NORM_DIR)
        .with_ns_steps(MUON_NS_STEPS)
        .with_weight_decay(cfg.wd)
}

/// Head-wise Muon+ for the attention Q/K projections (Qwen3.8-Flash-Next
/// §3.1: "split fused params per head BEFORE orthogonalization - fusing
/// mixes singular directions"). The `[n_heads*head_dim, d]` weight is
/// sliced into `n_heads` `[head_dim, d]` blocks, each gets its own
/// Newton-Schulz preconditioner + ColRow normalization, and the blocks are
/// concatenated back. The split also cuts NS cost ~n_heads-fold, which is
/// what makes routing Q/K to Muon affordable on this box (fp32 NS on
/// [768,768] was ~40 s/step; 12 blocks of [64,768] are milliseconds).
///
/// Not in 2602.21545 at all — the paper orthogonalizes each weight matrix as
/// a whole, so this is a deviation, attributed to the Qwen report, not to
/// Muon+.
///
/// This reimplements the Muon 2D branch, which is how it came to miss the
/// zero-gradient rule: `MuonPlus::step` gates its update on
/// [`burn_muon_plus::signal_mask`] so a masked (NaN-firewall) step moves
/// nothing, and this must apply the *same* mask from the *same* function. It
/// did not, and `qk_heads` is always resolved (`cfg.rs:54`), so the Q/K
/// weights took a full-magnitude step in the stale momentum direction on
/// every masked step while the rest of the model correctly did not.
/// Pinned by `headwise_zero_gradient_does_not_move_the_parameter`.
///
/// Q/K here are separate Linears (burn-kda gdn2), so the per-head split is
/// unambiguous - there is no fused `[d, 3d]` qkv to disambiguate.
#[derive(Clone)]
pub struct HeadWiseMuon {
    muon: MuonPlus,
    n_heads: usize,
    momentum: f64,
    weight_decay: f64,
}

impl HeadWiseMuon {
    /// Build from the shared Muon+ config with the given head count.
    pub fn new(cfg: &TrainCfg, n_heads: usize) -> Self {
        let muon_cfg = muon_plus_cfg(cfg);
        Self {
            momentum: muon_cfg.momentum,
            muon: muon_cfg.build(),
            n_heads,
            weight_decay: cfg.wd,
        }
    }
}

impl Optimizer for HeadWiseMuon {
    type State<const D: usize> = MuonPlusState<D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        let state = state.unwrap_or_else(|| MuonPlusState::new(None, None, None, None));
        // Q/K weights are matrices; anything else (never expected with the
        // current routing) falls through to Muon+'s own 1D AdamW branch.
        if D != 2 {
            return self.muon.step(lr, tensor, grad, Some(state));
        }
        let dims = tensor.dims();
        let (rows, cols) = (dims[0], dims[1]);
        assert!(
            rows >= self.n_heads && rows % self.n_heads == 0,
            "head-wise muon: weight [{rows}, {cols}] does not split into {} heads \
             (stale qk_heads after a config change?)",
            self.n_heads
        );

        // Momentum, identical to the MuonPlus 2D branch: M = mu*M + (1-mu)*G.
        let mu = self.momentum.elem::<f32>();
        let momentum = match &state.mu_momentum {
            Some(m) => {
                let mut mm = m.clone();
                #[cfg(feature = "cuda")]
                {
                    if !burn_muon_plus::fused_kernels::momentum_cuda(&mut mm, &grad, mu) {
                        SKIPPED_MOMENTUM.fetch_add(1, Relaxed); // tensor path, counted
                        mm = mm
                            .clone()
                            .mul_scalar(mu)
                            .add(grad.clone().mul_scalar(1.0 - mu));
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    mm = mm
                        .clone()
                        .mul_scalar(mu)
                        .add(grad.clone().mul_scalar(1.0 - mu));
                }
                mm
            }
            None => grad.clone().mul_scalar(1.0 - mu),
        };

        // The zero-gradient rule, from the crate that owns it: NS normalizes
        // the DECAYED momentum back to unit Frobenius norm, so without this
        // gate a step the trainer masked as a no-op still moved Q/K by a full
        // `lr_scaled` in the stale direction. Decoupled weight decay is NOT
        // gated, exactly as in `MuonPlus::step` - a zero gradient is a no-op
        // on the update, not on the decay.
        let g_active = burn_muon_plus::signal_mask(&grad);
        // Batched per-head NS: `[n_heads, dh, cols]` runs the SAME per-head
        // arithmetic (batched matmul is math per slice - no singular
        // directions mix; per-slice Frobenius norm and per-slice ColRow, same
        // clamp rule as the fused kernel) in one launch set instead of
        // `n_heads` sequential `[dh, cols]` calls. That was ~2 066 of the
        // optimizer stage's 3 921 launches/step at `small` (the single
        // largest opt item, launch-atlas 2026-10-02); the batched form is
        // ~60. Pinned bit-for-bit against the old per-head loop on CPU by
        // `headwise_batched_matches_per_head_loop`.
        //
        // What the change gives up: the fused `norm_colrow_cuda` kernel
        // (2-D strides only) no longer participates in the Q/K arm - that
        // is 2 launches per head per step it no longer runs, and the eval
        // line's `muon_skipped=` field is unaffected (the fused kernel is
        // not ASKED, so nothing counts a skip that did not happen). The
        // momentum and finalize fused kernels still run on the full 2-D
        // momentum/update, unchanged.
        let dh = rows / self.n_heads;
        let heads = self.n_heads;
        let per_head = self.muon.orthogonalize_batched(
            momentum.clone().reshape([heads, dh, cols]),
        );
        let per_head = self.muon.normalize(per_head);
        // Rank cast back to the generic `D`: the branch above verified the
        // RUNTIME rank is 2, but the compiler still holds the generic const.
        // Unsqueeze adds the size-1 dim, the squeeze removes it - the
        // two-step no-copy cast is the whole ritual. Batch layout
        // `[heads, dh, cols]` is row-major contiguous, so the reshape is the
        // same value sequence the per-head loop produced with
        // `Tensor::cat(parts, 0)` - no compute either way.
        let update = per_head
            .reshape([rows, cols])
            .mul(g_active.unsqueeze())
            .reshape::<3, _>([1, rows, cols])
            .squeeze_dim::<D>(0);

        // Same tail as MuonPlus 2D. The Bernstein factor uses the FULL
        // dims: per-head NS output carries the same total Frobenius norm
        // as a full-matrix NS (NS pins every singular value near 1 in both
        // cases), so the per-param step size is what full Muon would pick.
        // `max(1, m/n)^0.5` vs the paper's `sqrt(m/n)` is inert here: the Q/K
        // weight is `[n_heads*head_dim, d]`, square for every preset, so both
        // forms are 1.0. See `burn-muon-plus`'s `lr_scaled` comment.
        let (m, n) = (rows as f64, cols as f64);
        let lr_scaled = lr * (m / n).max(1.0).sqrt();
        let wd = (self.weight_decay as f32 * lr_scaled as f32).min(0.999);
        let mut updated = tensor.clone();
        #[cfg(feature = "cuda")]
        {
            if !burn_muon_plus::fused_kernels::finalize_cuda(
                &mut updated,
                &update,
                lr_scaled as f32,
                wd,
            ) {
                SKIPPED_FINALIZE.fetch_add(1, Relaxed); // tensor path, counted
                updated = updated
                    .clone()
                    .mul_scalar(1.0 - wd)
                    .sub(update.mul_scalar(lr_scaled as f32));
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            updated = updated
                .clone()
                .mul_scalar(1.0 - wd)
                .sub(update.mul_scalar(lr_scaled as f32));
        }
        (
            updated,
            Some(MuonPlusState::new(Some(momentum), None, None, None)),
        )
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device) -> Self::State<D> {
        state.mu_momentum = state.mu_momentum.map(|t| t.to_device(device));
        state.ad_moment_1 = state.ad_moment_1.map(|t| t.to_device(device));
        state.ad_moment_2 = state.ad_moment_2.map(|t| t.to_device(device));
        state
    }
}

/// The optimizer groups the trainer INSTALLS, built by id from the
/// declaration. This is the only place a group is assembled; the decision of
/// which group a parameter is in lives in [`routing::group_of`].
pub struct Installed {
    /// The low-rank TSCT `u`/`v` factors and the Engram key projections —
    /// the "small matrices that act as linear maps" group (module docs).
    /// Muon+ ColRow, NS 8; the group is never empty when the arms exist,
    /// which `Routing::check` (not this file) refuses loudly.
    pub muon: ParamGroup,
    /// `None` when head-wise Q/K routing is off: those two parameters then
    /// train on the base optimizer, so the group is not installed and must
    /// not be counted.
    pub qk: Option<ParamGroup>,
    /// The n-gram tables. Plain Adam, weight decay disabled (report §2.3):
    /// rows are trained by sparse per-key noise, and orthogonalizing a
    /// lookup table's rows against each other has no meaning.
    pub table: ParamGroup,
}

impl Installed {
    fn new(r: &routing::Routing, qk_heads: Option<usize>) -> Self {
        Self::of(
            r.group(Group::Muon),
            qk_heads.map(|_| r.group(Group::QkHeadWise)),
            r.group(Group::Table),
        )
    }

    /// An install built by hand. Exists so the loud gate
    /// ([`check_installed`]) can be tested against a group set that LIES -
    /// a group that claims nothing, everything, or a 1D leaf. That is the
    /// only way to prove the gate fires, and the previous string-based
    /// validator had exactly this test while the thing it did not test - the
    /// groups the optimizer actually holds - drifted.
    pub fn of(muon: ParamGroup, qk: Option<ParamGroup>, table: ParamGroup) -> Self {
        Self { muon, qk, table }
    }

    /// The group `id` must be INSTALLED into, as the declaration sees it.
    /// `None` means "no group": the base optimizer, which is correct for
    /// [`Group::Rest`] and for Q/K when `qk_heads` is unset.
    fn expected_group(&self, r: &routing::Routing, id: &burn::module::ParamId) -> Option<Group> {
        match r.group_of_id(id)? {
            Group::Rest => None,
            Group::QkHeadWise if self.qk.is_none() => None,
            g => Some(g),
        }
    }
}

/// The groups for `model`, from the declaration, verified against the live
/// module tree. The one call the trainer makes; it fails loudly rather than
/// installing groups that do not cover the model.
pub fn optimizer_groups(model: &DormouseModel, cfg: &TrainCfg) -> Result<Installed, String> {
    let r = routing::routing(model, cfg.factors_fallback);
    // Every parameter declared, in exactly one group (the declaration is
    // total, so this is the "a new arm cannot slip in unnoticed" gate).
    r.check(model)?;
    let g = Installed::new(&r, cfg.qk_heads);
    let _ = check_installed(model, &r, &g)?;
    Ok(g)
}

/// Run the INSTALLED groups against the live module tree and require that
/// they REPRODUCE the declaration, parameter by parameter. A parameter in two
/// groups, a 1D parameter in a Muon+ group, and a parameter the install puts
/// somewhere the policy does not are all loud errors naming the path.
///
/// `Rest` is the base optimizer and installs no group, so "claimed by
/// nothing" is not on its own an error - "claimed by nothing WHEN THE POLICY
/// SAYS MUON+" is. That distinction is the check; the previous version tested
/// only the string copy of the rules against itself.
///
/// This is the loud gate (ADR-0011/ADR-0019), and it reads the `ParamGroup`s
/// the optimizer holds, not a copy of the rules.
pub fn check_installed(
    model: &DormouseModel,
    r: &routing::Routing,
    g: &Installed,
) -> Result<GroupCounts, String> {
    let params = param_paths(model);
    let mut counts = GroupCounts::default();
    for (path, id, rank) in &params {
        let muon = g.muon.matches(id, Some(path));
        let qk = g.qk.as_ref().is_some_and(|q| q.matches(id, Some(path)));
        let table = g.table.matches(id, Some(path));
        let hits = [muon, qk, table].iter().filter(|hit| **hit).count();
        if hits > 1 {
            return Err(format!(
                "{path}: matches multiple installed optimizer groups ({hits})"
            ));
        }
        // A 1D leaf has no shared linear structure to orthogonalize; the TSCT
        // scale leaf's being 1D is the whole reason the factors are in the
        // group and `s` is not.
        if (muon || qk) && *rank != 2 {
            return Err(format!("{path}: 1D param routed to a Muon+ group"));
        }
        let declared = g.expected_group(r, id);
        let installed = muon
            .then_some(Group::Muon)
            .or(qk.then_some(Group::QkHeadWise))
            .or(table.then_some(Group::Table));
        if declared != installed {
            return Err(format!(
                "{path}: the installed optimizer group ({installed:?}) is not the declared one \
                 ({declared:?}) - the group sets must be built from the declaration, not from a \
                 second copy of the rules"
            ));
        }
        if muon {
            counts.muon += 1;
        } else if qk {
            counts.qk += 1;
        } else if table {
            counts.tables += 1;
        } else {
            counts.rest += 1;
        }
    }
    if params.is_empty() {
        return Err("the live model has no float parameters to route".to_string());
    }
    // A rule that routes NOTHING is the reachable form of a stale marker:
    // the groups are ids, so a marker cannot "match nothing" any more, but
    // an arm that stopped being built leaves its group empty and the run
    // would train that arm on the base optimizer without saying so. The
    // Engram key projection and the n-gram table are covered by
    // `Routing::check`; the Q/K group is only supposed to exist when it is
    // installed, so it is covered here.
    if g.qk.is_some() && counts.qk == 0 {
        return Err(
            "the head-wise Q/K group is installed (qk_heads is set) but claims no parameter - the \
             KDA q/k projections are gone"
                .to_string(),
        );
    }
    Ok(counts)
}

/// Every float parameter of `model` as `(path, id, rank)`. Paths are for
/// error messages and for tests that want to say WHICH parameter a group
/// decision is about - no routing decision is ever made from one, which is
/// the property the deleted marker table violated.
///
/// One collector for the whole repo: it lives in `dormouse-core::routing`,
/// next to the declaration that walks the same tree, because two
/// `ModuleVisitor`s doing the identical stack-then-join walk is the twin this
/// lane was opened to cut. Re-exported rather than reimplemented so a third
/// walk cannot appear.
pub use dormouse_core::routing::param_paths;

/// Build the optimizer for `mode`, the pure core of [`build_optim`].
pub(crate) fn build_optim_mode(model: &DormouseModel, cfg: &TrainCfg, mode: &str) -> Optim {
    let clip = (cfg.grad_clip > 0.0).then_some(GradientClippingConfig::Norm(cfg.grad_clip as f32));
    let groups = optimizer_groups(model, cfg)
        .unwrap_or_else(|e| panic!("optimizer routing check failed: {e}"));
    let muon_plus = || muon_plus_cfg(cfg).build();
    let mut opt = match mode {
        // Fallback optimizer candidates for the "rest" group.
        "adamw" => AdamWConfig::new().with_weight_decay(cfg.wd as f32).init(),
        "adan" => AdanConfig::new().with_weight_decay(cfg.wd as f32).init(),
        "muon" => muon_plus_cfg(cfg).init(),
        "mix-adan" => {
            let mut o = AdanConfig::new().with_weight_decay(cfg.wd as f32).init();
            o = o.with_group(groups.muon.clone(), muon_plus(), None);
            o = o.with_group(groups.table.clone(), AdamConfig::new().build(), None);
            with_qk_groups(o, cfg, &groups)
        }
        // mix (default): the report §3.1 recipe.
        _ => {
            let mut o = AdamWConfig::new().with_weight_decay(cfg.wd as f32).init();
            o = o.with_group(groups.muon.clone(), muon_plus(), None);
            o = o.with_group(groups.table.clone(), AdamConfig::new().build(), None);
            with_qk_groups(o, cfg, &groups)
        }
    };
    if let Some(c) = clip {
        // ModuleOptimizer::with_grad_clipping applies to the first optimizer.
        opt = opt.with_grad_clipping(c.init());
    }
    opt
}

/// Install the head-wise Q/K group when `qk_heads` is set (the preset's
/// `n_heads`). Unset keeps Q/K on the base fallback optimizer, which is why
/// the group is not installed rather than installed empty.
fn with_qk_groups(opt: Optim, cfg: &TrainCfg, groups: &Installed) -> Optim {
    let (Some(qk), Some(heads)) = (&groups.qk, cfg.qk_heads) else {
        return opt;
    };
    opt.with_group(qk.clone(), HeadWiseMuon::new(cfg, heads), None)
}

/// Build the optimizer per OPT env:
/// `mix` (default) | `mix-adan` | `adamw` | `adan` | `muon`.
/// Grad clipping stays on the base (first) optimizer; the report's Muon
/// recipe does not clip (gates bound the activations, pre-clip norms stay
/// low), and Muon+ has no clipping hook.
///
/// `model` is the LIVE instance, not a config: the groups are built from its
/// parameter ids, so this must be the model that will be trained.
pub fn build_optim(model: &DormouseModel, cfg: &TrainCfg) -> Optim {
    build_optim_mode(model, cfg, &cfg.opt)
}

/// The startup check, and the counts the banner prints. Reads the same
/// declaration the optimizer is built from and verifies the installed groups
/// against the live tree, so this cannot disagree with the install by
/// construction - which is the whole point: the previous version re-derived
/// the policy from path strings and disagreed with it on dense experts.
///
/// `factors_fallback` and `qk_heads` must match the values the optimizer was
/// built with.
pub fn validate_routing(
    model: &DormouseModel,
    factors_fallback: bool,
    qk_heads: Option<usize>,
) -> Result<GroupCounts, String> {
    let r = routing::routing(model, factors_fallback);
    r.check(model)?;
    let g = Installed::new(&r, qk_heads);
    check_installed(model, &r, &g)
}

#[cfg(test)]
mod tests {
    use super::*;
    // `model.visit` below comes from the Module trait; the trait has no
    // non-test caller in this crate, so a top-level import reads unused on
    // the lib build (newer rustc flags it, CI clippy runs one).
    use burn::module::Module as _;
    use burn::tensor::{Distribution, TensorData};

    /// The weight the trainer installs Q/K into is `[n_heads*head_dim, d]`.
    /// Square, as the real one is.
    const ROWS: usize = 64;
    const COLS: usize = 64;
    const HEADS: usize = 4;

    fn headwise() -> HeadWiseMuon {
        HeadWiseMuon::new(
            &TrainCfg {
                wd: 0.0,
                ..Default::default()
            },
            HEADS,
        )
    }

    /// Deterministic weights, so a failing assertion names a value.
    fn w() -> Tensor<2> {
        let vals: Vec<f32> = (0..ROWS * COLS)
            .map(|i| ((i.wrapping_mul(2654435761) % 997) as f32 / 498.0) - 1.0)
            .collect();
        Tensor::<2>::from_data(TensorData::new(vals, [ROWS, COLS]), &crate::device())
    }

    /// The batched per-head NS must equal the per-head loop it replaced,
    /// slice for slice, on the CPU backend (ndarray's batched matmul is a
    /// per-slice loop, so the gate is BITWISE here). The reference
    /// reconstructs the step's own momentum the same way `step` built it
    /// from a fresh state (μ from the config, `M = G·(1-μ)`), then runs the
    /// old `[dh, cols]`-slice path; the step runs the `[heads, dh, cols]`
    /// batched path. A divergence here is a silent change of WHOSE singular
    /// directions get orthogonalized.
    #[test]
    fn headwise_batched_matches_per_head_loop() {
        let dev = crate::device();
        let g = w(); // deterministic gradient
        let opt = headwise();
        let (updated, _) = opt.step(1e-3, w(), g.clone(), None);

        // Reference: the momentum `step` computed (fresh state), then the
        // per-head slice loop, then the same tail `step` applies.
        let mu_f: f32 = opt.momentum.elem();
        let momentum = g.clone().mul_scalar(1.0 - mu_f);
        let mut parts = Vec::with_capacity(HEADS);
        let dh = ROWS / HEADS;
        for h in 0..HEADS {
            let block = momentum.clone().slice([h * dh..(h + 1) * dh, 0..COLS]);
            parts.push(opt.muon.normalize(opt.muon.orthogonalize(block)));
        }
        let g_active = burn_muon_plus::signal_mask(&g);
        let ref_update = Tensor::cat(parts, 0).mul(g_active.unsqueeze());
        let (m, n) = (ROWS as f64, COLS as f64);
        let lr_scaled = 1e-3 * (m / n).max(1.0).sqrt();
        let wd = (opt.weight_decay as f32 * lr_scaled as f32).min(0.999);
        let mut ref_updated = w().clone();
        ref_updated = ref_updated
            .clone()
            .mul_scalar(1.0 - wd)
            .sub(ref_update.mul_scalar(lr_scaled as f32));

        // ndarary's batched matmul reduces per slice but through the generic
        // axis op, so the dot's summation order can differ from the singleton
        // shape's BLAS call: measured worst-case 1.5e-6 absolute at singular
        // values ~1 (probe_batched_tmp, this box). The gate is therefore
        // scale-relative, not bitwise.
        let maxdiff: f32 = updated
            .clone()
            .sub(ref_updated.clone())
            .abs()
            .max()
            .into_scalar();
        let scale: f32 = ref_updated.clone().abs().max().into_scalar();
        assert!(
            maxdiff <= 5e-6 * scale.max(1.0),
            "the batched per-head NS diverged from the per-head loop: {maxdiff:.3e} (scale {scale:.3e})"
        );
    }

    /// THE BUG. The NaN firewall zeroes every gradient on device, so a masked
    /// step must be a no-op for the head-wise Q/K group exactly as it is for
    /// the Muon+ group (`burn-muon-plus/tests/zero_grad.rs`). It was not: the
    /// momentum decays to `mu·M`, `orthogonalize` normalizes it back to unit
    /// Frobenius norm, and the weight moved by `lr_scaled` in the stale
    /// direction. `qk_heads` is always resolved, so this was every run.
    ///
    /// wd=0 so the only thing that can move the weight is the update.
    ///
    /// The first step takes a REAL gradient, and that is load-bearing: a
    /// zero-gradient step from a fresh state seeds `momentum = 0`, and
    /// `orthogonalize(0)` is 0, so the second step is a no-op with or without
    /// the gate. Written the other way round this test passes green with the
    /// mask deleted, which is exactly what happened to the sibling test in
    /// `burn-muon-plus` (see the note in that file). The leak needs a LIVE
    /// momentum, and two masked steps so the decayed-and-still-nonzero case is
    /// covered too.
    #[test]
    fn headwise_zero_gradient_does_not_move_the_parameter() {
        let dev = crate::device();
        let seed = Tensor::<2>::random([ROWS, COLS], Distribution::Default, &dev);
        let zero = Tensor::<2>::zeros([ROWS, COLS], &dev);
        let opt = headwise();
        let (w1, s1) = opt.step(1e-3, w(), seed, None);
        let after_seed = w1.clone().into_data();
        let (w2, s2) = opt.step(1e-3, w1, zero.clone(), s1);
        let (w3, _) = opt.step(1e-3, w2, zero, s2);
        assert_eq!(
            after_seed.into_bytes(),
            w3.into_data().into_bytes(),
            "a zero gradient moved the head-wise Q/K weight: the stale momentum \
             was rescaled to a full step (the NaN firewall's no-op contract)"
        );
    }

    /// THE SECOND BUG, in the same family: the eval line's
    /// `muon_skipped=mom/finalize` field used to count only the two Q/K
    /// weights this file steps, and nothing in `MuonPlus::step` — so the much
    /// larger `Group::Muon` population could have been running the tensor path
    /// for a whole run behind a log that read the same (ADR-0019: a fused arm
    /// must be able to show it ran).
    ///
    /// A CPU build cannot make the fused kernels answer, so this cannot assert
    /// the CUDA outcome. It pins the part that is decidable here: the number
    /// the trainer reads is the SUM over both implementations, so it cannot
    /// regress to counting one group. (The counters are global, so this
    /// asserts on the total rather than on a per-group delta.)
    #[test]
    fn the_eval_counter_covers_both_muon_implementations() {
        let dev = crate::device();
        let before = fused_kernels_skipped();
        // One step of the `Group::Muon` optimizer — the implementation that
        // was uncounted.
        let muon = MuonPlusConfig::new()
            .with_momentum(0.0)
            .with_norm_dir(MUON_NORM_DIR)
            .with_ns_steps(MUON_NS_STEPS)
            .with_weight_decay(0.0)
            .build();
        let g = Tensor::<2>::random([ROWS, COLS], Distribution::Default, &dev);
        let _ = muon.step(1e-3, Tensor::<2>::zeros([ROWS, COLS], &dev), g, None);
        // And one of the head-wise Q/K steps, the implementation that was
        // counted.
        let _ = headwise().step(1e-3, w(), Tensor::<2>::ones([ROWS, COLS], &dev), None);
        let after = fused_kernels_skipped();
        // Neither number can move on a CPU build (the fused kernels are never
        // asked), so the assertable property is that reading the seam is
        // side-effect-free and stable: a run that never touches CUDA must not
        // accumulate skips, or the eval line reports a fallback that never
        // happened.
        assert_eq!(
            before, after,
            "a CPU build must not accumulate fused-kernel skips: {before:?} -> {after:?}"
        );
        // And the aggregate is genuinely two sums, not one alias: the crate's
        // own accessor is what makes the second term exist at all.
        assert_eq!(
            (after.0, after.1),
            (
                SKIPPED_MOMENTUM.load(Relaxed) + burn_muon_plus::fused_skipped().0,
                SKIPPED_FINALIZE.load(Relaxed) + burn_muon_plus::fused_skipped().1,
            ),
            "the seam must be the sum of HeadWiseMuon's and MuonPlus::step's"
        );
    }

    /// The same run with a real gradient must still move it, so the gate above
    /// is not just "HeadWiseMuon stopped optimizing".
    #[test]
    fn headwise_non_zero_gradient_still_moves_the_parameter() {
        let dev = crate::device();
        let vals: Vec<f32> = (0..ROWS * COLS)
            .map(|i| ((i.wrapping_mul(40503) % 1013) as f32 / 506.0) - 1.0)
            .collect();
        let g = Tensor::<2>::from_data(TensorData::new(vals, [ROWS, COLS]), &dev);
        let before = w().into_data();
        let (after, _) = headwise().step(1e-3, w(), g, None);
        assert_ne!(
            before.into_bytes(),
            after.into_data().into_bytes(),
            "a non-zero gradient did not move the head-wise Q/K weight"
        );
    }

    /// The dimensional step-size factor, and why its deviation from the paper
    /// is inert here. `lr_scaled = lr·max(1, m/n)^0.5` (Jordan's `muon.py`)
    /// against the paper's `lr·sqrt(m/n)` (Eq. (4), Alg. 1 line 10).
    ///
    /// The factor is measured off the weight, not off the formula: with `wd=0`
    /// a zero weight and a fresh state, `W ← 0 − lr_scaled·O_t`, so with `μ=0`
    /// (which makes the momentum exactly `G`, no `(1-μ)` scale to guess at)
    /// and `O_t` recomputed here by the same two calls the optimizer makes,
    /// every entry of the result must equal `−O_t[i]·lr·factor`.
    ///
    /// The second half is what gives the test teeth: where the two candidate
    /// factors DIFFER (a wide `m < n`), the observed step must match Jordan's
    /// and must *not* match the paper's by more than a rounding error's worth.
    /// That is a ratio, so it is independent of `lr`.
    #[test]
    fn step_size_factor_is_jordans_max_one_not_the_papers_sqrt() {
        let dev = crate::device();
        // μ=0 so the momentum is exactly the gradient; wd=0 so the decay cannot
        // move a zero weight; ColRow so the live direction is the one tested.
        let muon = MuonPlusConfig::new()
            .with_momentum(0.0)
            .with_norm_dir(MUON_NORM_DIR)
            .with_ns_steps(MUON_NS_STEPS)
            .with_weight_decay(0.0)
            .build();
        // A large lr, so f32 rounding is negligible next to the factor gap
        // (which is a fixed ratio, ~1.41 on the wide shape) rather than a fixed
        // absolute tolerance that would have to be guessed.
        let lr = 1.0f32;
        for (m, n) in [(ROWS, 2 * ROWS), (2 * ROWS, ROWS)] {
            let vals: Vec<f32> = (0..m * n)
                .map(|i| ((i.wrapping_mul(2654435761) % 997) as f32 / 498.0) - 1.0)
                .collect();
            let g = Tensor::<2>::from_data(TensorData::new(vals, [m, n]), &dev);
            let o = muon.normalize(muon.orthogonalize(g.clone())).into_data();
            let o = o.as_slice::<f32>().unwrap().to_vec();
            let (updated, _) = muon.step(lr as f64, Tensor::<2>::zeros([m, n], &dev), g, None);
            let got = updated.into_data();
            let got = got.as_slice::<f32>().unwrap();

            let ratio = m as f64 / n as f64;
            let max_one = (lr as f64 * ratio.max(1.0).sqrt()) as f32;
            let paper = (lr as f64 * ratio.sqrt()) as f32;
            let err = |factor: f32| -> f32 {
                o.iter()
                    .zip(got)
                    .map(|(a, b)| (a * -factor - b).abs())
                    .fold(0.0f32, f32::max)
            };
            let worst = err(max_one);
            assert!(
                worst < 1e-4 * max_one.abs().max(1.0),
                "[{m}x{n}]: max|observed + lr·max(1,m/n)^0.5·O| = {worst:e}"
            );

            if (paper - max_one).abs() > 1e-6 {
                let worst_paper = err(paper);
                assert!(
                    worst_paper > 100.0 * worst,
                    "[{m}x{n}]: the wide shape must tell the two factors apart \
                     ({max_one} vs {paper}), but the step fits the paper's \
                     sqrt(m/n) as well as ours ({worst_paper:e} vs {worst:e}) - \
                     the test cannot see the difference it exists to pin"
                );
            }
        }
    }

    /// ...and on the shapes the trainer ACTUALLY installs, the two forms are the
    /// same number, so the D3 deviation cannot reach a run.
    ///
    /// Measured off the live model's routed groups, not off a hand-written
    /// shape list: a list would keep passing after a preset widened a factor,
    /// which is the whole failure this claim has. The declared groups are
    /// `Group::Muon` (TSCT factors + the Engram key projections) and
    /// `Group::QkHeadWise` (the Q/K weights). Every one of them must satisfy
    /// `m ≥ n`, which is where `max(1, m/n)^0.5` and `sqrt(m/n)` coincide.
    ///
    /// A wide member is not a failure of the optimizer, it is a failure of this
    /// argument: it would mean the `lr_scaled` comment in `burn-muon-plus` (and
    /// the `D3` deviation being inert) is no longer true, and the deviation
    /// would have to be re-adjudicated rather than assumed.
    #[test]
    fn no_routed_parameter_is_wide() {
        let mut seen = 0;
        for use_tsct in [true, false] {
            let model = DormouseModel::new(&test_model_cfg(use_tsct), &crate::device());
            let r = routing::routing(&model, false);
            for (path, id, rank) in param_paths(&model) {
                if rank != 2 || !matches!(r.group_of_id(&id), Some(Group::Muon | Group::QkHeadWise))
                {
                    continue;
                }
                seen += 1;
                let (m, n) = param_dims(&model, &id);
                assert!(
                    m >= n,
                    "use_tsct={use_tsct}: {path} is {m}x{n} and routed to Muon+ - \
                     a WIDE matrix in a Muon+ group changes the step size under \
                     the paper's sqrt(m/n), so the max(1,·) deviation stops \
                     being inert and D3 has to be re-adjudicated"
                );
            }
        }
        assert!(
            seen > 0,
            "no rank-2 parameter landed in a Muon+ group, so this proves nothing \
             about the shapes that are actually routed"
        );
    }

    /// Small but every arm present: rank, experts, Engram key projections, and
    /// the KDA q/k the head-wise group claims.
    fn test_model_cfg(use_tsct: bool) -> dormouse_core::DormouseConfig {
        dormouse_core::DormouseConfig {
            use_tsct,
            d_model: 32,
            n_heads: 2,
            head_dim: 16,
            d_ffn: 64,
            max_iter: 2,
            n_experts: 2,
            rank: 8,
            engram_rows: 64,
            ..dormouse_core::DormouseConfig::default()
        }
    }

    /// `(rows, cols)` of the rank-2 parameter with this id, read off the model.
    fn param_dims(model: &DormouseModel, id: &burn::module::ParamId) -> (usize, usize) {
        struct Dims<'a> {
            want: &'a burn::module::ParamId,
            out: Option<(usize, usize)>,
        }
        impl burn::module::ModuleVisitor for Dims<'_> {
            fn enter_module(&mut self, _n: &str, _c: &str) {}
            fn exit_module(&mut self, _n: &str, _c: &str) {}
            fn visit_float<const D: usize>(&mut self, p: &burn::module::Param<Tensor<D>>) {
                if p.id == *self.want {
                    let d = p.val().dims();
                    self.out = Some((d[D - 2], d[D - 1]));
                }
            }
        }
        let mut v = Dims {
            want: id,
            out: None,
        };
        model.visit(&mut v);
        v.out.expect("id came from this model, so it must be in it")
    }
}
