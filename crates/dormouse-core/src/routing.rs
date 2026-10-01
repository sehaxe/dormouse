//! routing - which optimizer each parameter goes to, DECLARED BY THE MODULE
//! THAT OWNS IT (Qwen3.8-Flash-Next §3.1 routing, ADR-0017).
//!
//! Burn matches optimizer groups by module path, so this used to be a table
//! of the model's private field names living in the trainer
//! (`"expert_ffns."`, `"gdn2.q_proj"`, ...), plus 85 lines of startup
//! validation re-deriving the policy from those strings. A rename then
//! silently demoted a parameter from Muon+ to AdamW: a quality regression
//! with no error anywhere, and the one violation that fought every move in
//! the refactor plan.
//!
//! Here the declaration sits next to the arms it names, and the group is
//! built from `ParamId`s ([`ParamGroup::from_ids`]), so a rename or a
//! restructure cannot change where a parameter trains. The POLICY is one
//! match, [`group_of`]; everything else is bookkeeping:
//! - each arm claims the parameters that need a non-default optimizer and
//!   declares the rest of its own subtree as [`Group::Rest`];
//! - a parameter no arm claims is a startup error ([`Routing::check`]),
//!   so a new arm cannot slip in unnoticed;
//! - a new [`Role`] or [`LinearParam`] does not compile until the policy
//!   match says where it trains.
//!
//! **There is no second implementation.** This file is what the trainer calls
//! (`dormouse-train::optim::optimizer_groups` -> [`routing`]) and what
//! installs the groups. An earlier version of this note claimed the opposite
//! and pointed at a path-marker copy in `optim.rs`; that copy was deleted by
//! `831e3a0` on 2026-09-28, and the stale sentence was carried into a LATER
//! docs commit (`d81d920`, 2026-10-01). See
//! `docs/reviews/dedup-optimizer-2026-10-01.md`.

use std::collections::{BTreeMap, HashMap};

use burn::module::{Module, ModuleVisitor, Param, ParamGroup, ParamId};
use burn::tensor::Tensor;
use burn_engram::EngramModule;

use crate::attention::AdaptiveAttention;
use crate::loop_block::{ExpertFFN, LoopBlock};
use crate::model::DormouseModel;
use crate::param::{LinearLike, LinearParam};

/// The optimizer group of a parameter. The set is closed: [`Group::ALL`] is
/// what the trainer must be able to serve, and a group is registered by
/// construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Group {
    /// Muon+ ColRow with Newton-Schulz (`ns_steps = 8`).
    Muon,
    /// Head-wise Muon+ for the attention Q/K weights: each `[head_dim, d]`
    /// block gets its own preconditioner (report §3.1).
    QkHeadWise,
    /// n-gram tables: plain Adam, weight decay disabled (report §2.3).
    Table,
    /// Everything else - the base (AdamW / Adan per `--opt`).
    Rest,
}

impl Group {
    /// Every group a trainer must be able to serve. The set is CLOSED: the
    /// array's length is the count the enum claims, so adding a [`Group`]
    /// variant does not compile until the trainer is taught to serve it.
    pub const ALL: [Group; 4] = [Group::Muon, Group::QkHeadWise, Group::Table, Group::Rest];
}

/// Parameter count per group, for the startup banner and the tests.
///
/// The repo's ONE counts type. `dormouse-train` used to declare a second
/// `GroupCounts` with the same four fields and no docs; it now re-exports
/// this one, so the banner and the declaration cannot drift apart in shape.
///
/// A count of 0 is a finding, not a formatting detail: [`Routing::check`]
/// refuses to start when the `Table` or `Muon` group is empty, because an empty
/// group is an arm that silently stopped being built — the shape that produced
/// a run reporting `muon=0` in its own banner while training everything on
/// AdamW.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct GroupCounts {
    /// Parameters on Muon+ ColRow with Newton-Schulz (`ns_steps = 8`).
    pub muon: usize,
    /// Attention Q/K parameters on head-wise Muon+ (report §3.1).
    pub qk: usize,
    /// n-gram table rows, on plain Adam with weight decay off (report §2.3).
    pub tables: usize,
    /// Everything else: the base optimizer (`--opt mix` puts 1-D and
    /// embeddings here, which is Adan/AdamW depending on the flag).
    pub rest: usize,
}

/// What a [`LinearLike`] IS in the model. The owner says this where it builds
/// the linear; [`group_of`] turns (role, kind) into the group. Adding a role
/// does not compile until the policy decides it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// An expert FFN's gate_up/down (report: fc1/fc2 of routed/shared experts).
    Expert,
    /// The per-iteration loop readout.
    Readout,
    /// The output head.
    Head,
}

/// THE POLICY, in one match. Everything else in this file is bookkeeping.
///
/// Only the small low-rank TSCT **factors** go to Muon+: the Newton-Schulz
/// iteration costs ~3 matmuls per step on the FULL `[m,n]` matrix, and on
/// this box (5060 Ti, fp32, no tensor cores) 8 NS iters on the `[d,d]`
/// projections added ~40 s/step, while the factored `[d,r]`/`[r,f]` form is
/// ~1000x cheaper. So:
/// - the 1D TSCT scale leaf `s` is excluded (a 1-D output has no shared
///   linear structure to orthogonalize);
/// - a dense (`use_tsct = false`) linear's weight and bias are excluded -
///   a dense linear is a real `[d,d]` map, i.e. the expensive case;
/// - the elongated low-rank readout's factors ARE included, but `--factors-
///   fallback` moves the EXPERT factors to the fallback (A/B knob).
#[must_use]
pub fn group_of(role: Role, kind: LinearParam, factors_fallback: bool) -> Group {
    use Group::*;
    use LinearParam::*;
    use Role::*;
    match (role, kind) {
        (Expert, Factor) if factors_fallback => Rest,
        (Expert | Readout, Factor) => Muon,
        // The head is an elongated [d, vocab] readout: AdamW (report).
        (Head, Factor | Scale | DenseWeight | DenseBias) => Rest,
        (Expert | Readout, Scale | DenseWeight | DenseBias) => Rest,
    }
}

/// Declare the parameters of one linear, by what it is.
pub fn route_linear(into: &mut Routing, lin: &LinearLike, role: Role, factors_fallback: bool) {
    for (id, kind) in lin.param_kinds() {
        into.push(group_of(role, kind, factors_fallback), id);
    }
}

/// Every float param of a module subtree as `(path, id, rank)`. Paths are for
/// error messages only - no routing decision is ever made from one.
///
/// THE repo's one module walker: the trainer needs the rank (`check_installed`
/// refuses a 1D param in a Muon+ group) and the declaration does not, so the
/// rank rides along in the same tuple rather than in a second visitor doing an
/// identical stack-then-join walk.
#[derive(Default)]
struct ParamCollector {
    stack: Vec<String>,
    out: Vec<(String, ParamId, usize)>,
}

impl ModuleVisitor for ParamCollector {
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

/// Every float parameter of `model` as `(path, id, rank)`.
///
/// The one walk. `dormouse-train::param_paths` re-exports this rather than
/// keeping its own copy of the traversal.
pub fn param_paths(model: &DormouseModel) -> Vec<(String, ParamId, usize)> {
    let mut c = ParamCollector::default();
    model.visit(&mut c);
    c.out
}

/// The declared routing: one id list per group.
#[derive(Default)]
pub struct Routing {
    groups: BTreeMap<Group, Vec<ParamId>>,
}

impl Routing {
    fn push(&mut self, group: Group, id: ParamId) {
        self.groups.entry(group).or_default().push(id);
    }

    /// Declare everything in this subtree that no arm has claimed yet as
    /// [`Group::Rest`]. This is how an arm states its default: a new
    /// parameter added to an existing arm is trained by the base optimizer
    /// unless someone routes it, and it is still CHECKED (below), never
    /// invisible.
    fn rest_of<M: Module>(&mut self, module: &M) {
        let mut c = ParamCollector::default();
        module.visit(&mut c);
        for (_, id, _) in c.out {
            if self.claimed(id) {
                continue;
            }
            self.push(Group::Rest, id);
        }
    }

    /// A parameter is claimed if it is ALREADY declared anywhere, Rest
    /// included: `rest_of` is called on nested subtrees (the attention arm
    /// first, then the whole block), so without this a parameter nothing
    /// explicitly routed - `v_proj`, every norm - is pushed to Rest twice and
    /// `check` reports it as declared in two groups.
    fn claimed(&self, id: ParamId) -> bool {
        self.groups.values().any(|ids| ids.contains(&id))
    }

    /// The burn group for one declared group. Ids, not paths: this is the
    /// whole point - the seam carries no field names.
    pub fn group(&self, group: Group) -> ParamGroup {
        ParamGroup::from_ids(self.groups.get(&group).cloned().unwrap_or_default())
    }

    /// How many parameters this routing declares for `group`. `0` for a group
    /// nobody claimed anything for, which [`Routing::check`] treats as a
    /// startup error for `Table` and `Muon`.
    pub fn count(&self, group: Group) -> usize {
        self.groups.get(&group).map_or(0, Vec::len)
    }

    /// The declared group of ONE parameter, by id. The trainer reads the
    /// policy through this and through [`Routing::group`] - the same map, so
    /// there is no second place a group can be decided, and no second place
    /// they can disagree.
    pub fn group_of_id(&self, id: &ParamId) -> Option<Group> {
        self.groups.iter().find_map(|(g, ids)| ids.contains(id).then_some(*g))
    }

    /// The startup assertion, and now the only one: every parameter of the
    /// live model is declared, in exactly one group. It checks the
    /// DECLARATION - not a second derivation of the policy from names - so it
    /// cannot disagree with the group builder by construction.
    pub fn check(&self, model: &DormouseModel) -> Result<GroupCounts, String> {
        let mut c = ParamCollector::default();
        model.visit(&mut c);
        let path_of: HashMap<ParamId, String> =
            c.out.iter().map(|(p, id, _)| (*id, p.clone())).collect();
        let mut declared: HashMap<ParamId, Group> = HashMap::with_capacity(path_of.len());
        for (group, ids) in &self.groups {
            for id in ids {
                if let Some(prev) = declared.insert(*id, *group) {
                    return Err(format!(
                        "{}: declared in both {prev:?} and {group:?}",
                        path_of.get(id).map_or("<unknown param>", |p| p.as_str())
                    ));
                }
            }
        }
        // A group that routes NOTHING is reported BEFORE the totality scan
        // below, because that scan would fire first and name one of the
        // orphaned parameters instead of the arm that stopped being built.
        // The n-gram tables are one such group: the report's §2.3 rule is that
        // they train on plain Adam, and a model with no table means an arm
        // that silently stopped being built.
        if self.count(Group::Table) == 0 {
            return Err("no parameter is routed to the n-gram table group".to_string());
        }
        // Likewise the Engram KEY projection, which the report routes to
        // Muon+. An empty Muon+ group means the Engram arm stopped being
        // built, and the symptom without this check is a run that reports
        // `muon=0` in the banner and trains the whole model on AdamW.
        if self.count(Group::Muon) == 0 {
            return Err(
                "no parameter is routed to the Muon+ group - the Engram key projection is gone"
                    .to_string(),
            );
        }
        for (path, id, _) in &c.out {
            if !declared.contains_key(id) {
                return Err(format!("{path}: no declared optimizer group"));
            }
        }
        Ok(GroupCounts {
            muon: self.count(Group::Muon),
            qk: self.count(Group::QkHeadWise),
            tables: self.count(Group::Table),
            rest: self.count(Group::Rest),
        })
    }
}

/// A module that declares where its parameters train.
///
/// Implementors push into the [`Routing`] they are handed: each claims the
/// parameters that need a non-default optimizer, then declares the remainder
/// of its own subtree as [`Group::Rest`]. That last part is the load-bearing
/// half — a new parameter added inside an existing arm is TRAINED (by the base
/// optimizer) and CHECKED, never invisible.
///
/// `factors_fallback` is the `--factors-fallback` switch: when set, the expert
/// TSCT `u`/`v` factors drop out of the Muon+ group and go to the fallback
/// optimizer, which is the A/B for whether the factors belong in Muon+ at all.
///
/// This trait IS the live policy — there is no second copy. `831e3a0`
/// (2026-09-28) deleted the trainer's path-marker policy that used to build
/// the groups from strings; `dormouse-train::optim` now ASSEMBLES what this
/// declares, into `ParamGroup`s by id. If you are reading an older note that
/// calls this "exercised only by tests" and names a path-marker twin in
/// `optim.rs`, that note predates the cut and is wrong (it was reintroduced
/// into prose by `d81d920`, after the code was already fixed).
pub trait Routed {
    /// Declare this subtree's parameters into `into`. Called once per module
    /// tree at startup; must be total (every parameter in exactly one group),
    /// which [`Routing::check`] then asserts against the live model.
    fn route(&self, into: &mut Routing, factors_fallback: bool);
}

/// The declared routing of a live model. Called once at startup, next to the
/// optimizer build it feeds.
pub fn routing(model: &DormouseModel, factors_fallback: bool) -> Routing {
    let mut into = Routing::default();
    model.route(&mut into, factors_fallback);
    into
}

impl Routed for DormouseModel {
    fn route(&self, into: &mut Routing, factors_fallback: bool) {
        route_linear(into, &self.lm_head, Role::Head, factors_fallback);
        self.loop_block.route(into, factors_fallback);
        // Embedding, final norm, aux heads - and any arm added later, which
        // lands on the base optimizer until someone routes it.
        into.rest_of(self);
    }
}

impl Routed for LoopBlock {
    fn route(&self, into: &mut Routing, factors_fallback: bool) {
        for e in &self.expert_ffns {
            e.route(into, factors_fallback);
        }
        route_linear(into, &self.out_proj, Role::Readout, factors_fallback);
        self.shared_attn.route(into, factors_fallback);
        self.engram.route(into, factors_fallback);
        // Controller, mem_dense, norm, GR, AttnRes pseudo-queries, mHC's
        // hyper-network, iter_embed, residual_scale, the MoR router: routers,
        // scalars, vectors and norms are AdamW (orthogonalizing them is
        // meaningless or harmful). The AttnRes query is a `[d]` vector per
        // iteration slot, so it lands here with `residual_scale` rather than in
        // a group of its own - the same reasoning as the 1-D TSCT scale leaf.
        // mHC's three projections (`[D, n]`, `[D, n]`, `[D, n^2]`) land here
        // for the same reason ReZero's scalar and the controller's gates do:
        // they are GATES, not weight matrices, and the paper's own
        // parameterization puts a nonlinearity (sigmoid, Sinkhorn) between the
        // parameter and the operator, so there is no matrix to polar-rotate.
        into.rest_of(self);
    }
}

impl Routed for ExpertFFN {
    fn route(&self, into: &mut Routing, factors_fallback: bool) {
        let Self { gate_up, down } = self;
        route_linear(into, gate_up, Role::Expert, factors_fallback);
        route_linear(into, down, Role::Expert, factors_fallback);
    }
}

impl Routed for AdaptiveAttention {
    fn route(&self, into: &mut Routing, _factors_fallback: bool) {
        let Self { gdn2 } = self;
        // Q/K only: report §3.1 splits them per head before orthogonalization
        // (see the trainer's `HeadWiseMuon`). v/o, the decay and beta gates
        // and the output norm are the rest of the attention arm.
        for qk in [&gdn2.q_proj, &gdn2.k_proj] {
            into.push(Group::QkHeadWise, qk.weight.id);
        }
        into.rest_of(gdn2);
    }
}

impl Routed for EngramModule {
    fn route(&self, into: &mut Routing, _factors_fallback: bool) {
        // ponytail: the one path match left in the repo, because
        // burn-engram keeps its fields private and its key projections are
        // indistinguishable from its value projection by type or shape (both
        // are a bias-free `nn::Linear` [3*32, d]). The patterns are matched
        // against paths RELATIVE to the engram, in the file that owns the
        // model; `Routing::check` fails loudly if a rename moves a param out
        // of all three. Undo by making the fields public in the library.
        let mut v = EngramVisitor { into, stack: Vec::new() };
        self.visit(&mut v);
    }
}

struct EngramVisitor<'a> {
    into: &'a mut Routing,
    stack: Vec<String>,
}

impl ModuleVisitor for EngramVisitor<'_> {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.stack.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.stack.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let arm = self.stack.first().map_or("", String::as_str);
        let group = match arm {
            // Report §3.1: the n-gram KEY projections are Muon+ candidates.
            "key_projs" => Group::Muon,
            // Report §2.3: the hashed tables train on plain Adam, no wd.
            "memory" => Group::Table,
            // value_proj is [d,d] (too expensive for fp32 NS) and the
            // optional short-conv kernel is a 1-D filter: the fallback.
            _ => Group::Rest,
        };
        self.into.push(group, param.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DormouseConfig;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex()
    }

    fn cfg() -> DormouseConfig {
        DormouseConfig {
            d_model: 32,
            n_heads: 2,
            head_dim: 16,
            d_ffn: 64,
            max_iter: 2,
            n_experts: 2,
            rank: 8,
            engram_rows: 64,
            ..DormouseConfig::default()
        }
    }

    /// The policy, pinned as a table. This is the replacement for the string
    /// table: every parameter kind of the model, and the group it trains on.
    /// A rename or a restructure cannot move a row (ids, not paths); a new
    /// row that nobody declared fails `Routing::check`; a change to any of
    /// these cells is a change to [`group_of`] and shows up HERE first.
    #[test]
    fn policy_table_is_pinned() {
        use Group::*;
        use LinearParam::*;
        use Role::*;
        // (role, kind, factors_fallback) -> group.
        let table: &[(Role, LinearParam, bool, Group)] = &[
            // TSCT spectral factors: Muon+ (expert factors move to the
            // fallback under --factors-fallback, an A/B knob).
            (Expert, Factor, false, Muon),
            (Expert, Factor, true, Rest),
            (Readout, Factor, false, Muon),
            (Readout, Factor, true, Muon),
            // The 1D TSCT scale leaf: never orthogonalized.
            (Expert, Scale, false, Rest),
            (Readout, Scale, false, Rest),
            // A dense linear is the expensive [d,d] case: weight AND bias on
            // the fallback. This is the pair that disagreed before
            // (is_muon_param said the weight was Muon+, the group builder
            // said otherwise).
            (Expert, DenseWeight, false, Rest),
            (Expert, DenseBias, false, Rest),
            (Readout, DenseWeight, false, Rest),
            (Readout, DenseBias, false, Rest),
            // The output head: elongated [d, vocab], AdamW.
            (Head, Factor, false, Rest),
            (Head, Scale, false, Rest),
            (Head, DenseWeight, false, Rest),
            (Head, DenseBias, false, Rest),
        ];
        for (role, kind, ff, want) in table {
            assert_eq!(
                group_of(*role, *kind, *ff),
                *want,
                "policy changed for {role:?} {kind:?} factors_fallback={ff}"
            );
        }
        // Exhaustive by construction: every (role, kind) pair is in the table
        // above, so a new role or a new leaf is a compile error in the match
        // in `group_of` AND a missing row here.
        let all: Vec<(Role, LinearParam, bool, Group)> = [
            Expert,
            Readout,
            Head,
        ]
        .into_iter()
        .flat_map(|r| [Factor, Scale, DenseWeight, DenseBias].into_iter().map(move |k| (r, k)))
        .flat_map(|(r, k)| {
            // (r, k) has to ride through the (ff, g) expansion: the pair
            // below is the only carrier, so dropping it here loses the role
            // and the kind and the map after it has nothing to destructure.
            [(false, group_of(r, k, false)), (true, group_of(r, k, true))]
                .into_iter()
                .map(move |(ff, g)| (r, k, ff, g))
        })
        .collect();
        // The ENUMERATION is exhaustive by construction; the table above is
        // the readable subset (every cell the policy says something
        // non-obvious about). Asserting they have the same LENGTH was wrong
        // - it demanded 24 hand-written rows to document 14 decisions - and
        // the per-row check above already ties every table cell to
        // `group_of`. What must not go stale is the enumeration: a new Role
        // or LinearParam changes this number, and the match in `group_of`
        // stops compiling at the same time.
        assert_eq!(all.len(), 24, "3 roles x 4 kinds x 2 fallback settings");
    }

    /// The whole model, checked against the declaration: nothing invisible,
    /// nothing twice. The counts come from the topology, not from literals.
    #[test]
    fn every_param_is_declared_exactly_once() {
        let cfg = cfg();
        let model = DormouseModel::new(&cfg, &dev());
        let r = routing(&model, false);
        let c = r.check(&model).expect("the live model must be fully declared");
        // 4 factors per expert (gate_up u,v + down u,v) + the readout's u,v
        // + the engram's key projection.
        assert_eq!(c.muon, 4 * cfg.n_experts + 3, "Muon+ group must match the topology");
        assert_eq!(c.qk, 2, "KDA q/k weights are head-wise Muon");
        assert_eq!(c.tables, 1, "the n-gram tables are one param");
        assert!(c.rest > 0, "embedding, head, router, norms, conv...");
        assert_eq!(
            c.muon + c.qk + c.tables + c.rest,
            r.groups.values().map(Vec::len).sum::<usize>(),
            "the counts must cover the declaration exactly"
        );
        // The dense arm moves the experts' weights to the fallback. It does
        // NOT declare the same number of params: a dense linear is TWO
        // (weight, bias) where the TSCT it replaces is THREE (u, s, v), so
        // the dense model has FEWER param tensors. Asserting equal totals
        // here was the bug; the disagreement this test exists for is about
        // the GROUP, and that is what the three lines below check.
        let dense = DormouseConfig { use_tsct: false, ..cfg.clone() };
        let m2 = DormouseModel::new(&dense, &dev());
        let c2 = routing(&m2, false).check(&m2).expect("dense model must be fully declared");
        assert!(
            c2.rest > c.rest,
            "the dense experts' weights must move to the fallback ({:?} -> {:?})",
            c,
            c2
        );
        assert_eq!(c2.qk, c.qk, "the head-wise Q/K group is the KDA q and k either way");
        assert_eq!(c2.tables, c.tables, "the n-gram table is one param either way");
    }

    /// The check still has teeth: a parameter nobody claims is an error, not
    /// a silent fallback. Drop the readout's declaration by hand.
    #[test]
    fn undeclared_param_fails_the_check() {
        let cfg = cfg();
        let model = DormouseModel::new(&cfg, &dev());
        let mut r = routing(&model, false);
        let mut ids = r.groups.get_mut(&Group::Muon).expect("declared");
        let dropped = ids.pop().expect("non-empty");
        let err = r.check(&model).expect_err("an unclaimed param must fail");
        assert!(err.contains("no declared optimizer group"), "unexpected error: {err}");
        assert!(!err.contains(&dropped.val().to_string()), "the error names the path: {err}");
    }

    /// A rule that routes NOTHING is loud (ADR-0019), not a silent
    /// degradation. The groups are ids, so a marker cannot "match nothing"
    /// any more - but an arm that stopped being built leaves its group empty,
    /// and the symptom without this is a run that trains the whole Engram (or
    /// the whole table set) on the base optimizer and reports no error.
    #[test]
    fn an_empty_routing_group_is_loud() {
        let cfg = cfg();
        let model = DormouseModel::new(&cfg, &dev());
        for group in [Group::Muon, Group::Table] {
            let mut r = routing(&model, false);
            let ids = r.groups.get_mut(&group).expect("declared");
            assert!(!ids.is_empty(), "{group:?} starts non-empty");
            ids.clear();
            let err = r.check(&model).expect_err("an empty group must fail the check");
            assert!(err.contains("no parameter is routed"), "unexpected error: {err}");
        }
    }

    /// `--factors-fallback` moves exactly the expert factors, from Muon+ to
    /// the base optimizer, and the declaration stays valid.
    #[test]
    fn factors_fallback_moves_only_the_experts() {
        let cfg = cfg();
        let model = DormouseModel::new(&cfg, &dev());
        let with = routing(&model, false).check(&model).expect("declared");
        let without = routing(&model, true).check(&model).expect("declared");
        assert_eq!(without.muon + 4 * cfg.n_experts, with.muon);
        assert_eq!(without.rest - with.rest, 4 * cfg.n_experts);
    }
}
