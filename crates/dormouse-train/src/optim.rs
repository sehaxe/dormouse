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
    module::{Module, ModuleVisitor, Param, ParamGroup},
    optim::{AdanConfig, AdamConfig, AdamWConfig, LearningRate, Optimizer},
    tensor::{Device, ElementConversion, Tensor},
};
use burn_muon_plus::{MuonPlus, MuonPlusConfig, MuonPlusState, NormDir};
use dormouse_core::routing::{self, Group};
use dormouse_core::DormouseModel;
use std::sync::atomic::Ordering::Relaxed;

use crate::{Optim, TrainCfg};

/// How many times a fused Muon+ CUDA kernel was asked for and answered with
/// the tensor-ops path instead. `false` from `momentum_cuda`/`finalize_cuda`
/// means "not a bare cubecl tensor" or "empty", and the tensor path computes
/// the same function - so without a counter the fused optimizer kernel can be
/// dead for a whole run and the log looks identical (ADR-0019; the head-wise
/// Q/K group this guards is ON in every run, `qk_heads` is always resolved).
pub fn fused_kernels_skipped() -> (u64, u64) {
    (SKIPPED_MOMENTUM.load(Relaxed), SKIPPED_FINALIZE.load(Relaxed))
}
/// Present on every build so [`fused_kernels_skipped`] has one answer: with
/// no cuda feature the fused kernels are never asked, and `(0, 0)` says so.
static SKIPPED_MOMENTUM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SKIPPED_FINALIZE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Newton-Schulz iterations for Muon+ orthogonalization. Report §3.1: 8 for
/// stability (the Muon+ paper default is 5).
pub const MUON_NS_STEPS: usize = 8;
/// Post-polar normalization direction (2602.21545: column-then-row is the
/// paper's best combination).
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

        // One NS preconditioner per head (2D slices keep the fused muon+
        // kernels - which assume a 2D layout - on their fast path), then
        // concatenate back.
        let dh = rows / self.n_heads;
        let mut parts = Vec::with_capacity(self.n_heads);
        for h in 0..self.n_heads {
            let block = momentum.clone().slice([h * dh..(h + 1) * dh, 0..cols]);
            parts.push(self.muon.normalize(self.muon.orthogonalize(block)));
        }
        let update = Tensor::cat(parts, 0);

        // Same tail as MuonPlus 2D. The Bernstein factor uses the FULL
        // dims: per-head NS output carries the same total Frobenius norm
        // as a full-matrix NS (NS pins every singular value near 1 in both
        // cases), so the per-param step size is what full Muon would pick.
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
    pub muon: ParamGroup,
    /// `None` when head-wise Q/K routing is off: those two parameters then
    /// train on the base optimizer, so the group is not installed and must
    /// not be counted.
    pub qk: Option<ParamGroup>,
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
            return Err(format!("{path}: matches multiple installed optimizer groups ({hits})"));
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

/// Count parameters per optimizer group (Muon+ / head-wise Q/K / Adam tables
/// / base-optimizer rest), counted off the INSTALLED groups. Used by the
/// startup banner and the routing tests.
#[derive(Default, Debug)]
pub struct GroupCounts {
    pub muon: usize,
    pub qk: usize,
    pub tables: usize,
    pub rest: usize,
}

/// Every float param of the live model: `(path, id, rank)`. Paths are for
/// error messages only - no routing decision is ever made from one.
#[derive(Default)]
struct PathCollector {
    stack: Vec<String>,
    out: Vec<(String, burn::module::ParamId, usize)>,
}

impl ModuleVisitor for PathCollector {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.stack.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.stack.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.out.push((self.stack.join("."), param.id, D));
    }
}

/// Every float parameter of `model` as `(path, id, rank)`. Paths are for
/// error messages and for tests that want to say WHICH parameter a group
/// decision is about - no routing decision is ever made from one, which is
/// the property the marker table violated.
pub fn param_paths(model: &DormouseModel) -> Vec<(String, burn::module::ParamId, usize)> {
    let mut c = PathCollector::default();
    model.visit(&mut c);
    c.out
}

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
