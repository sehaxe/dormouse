//! Optimizer policy: Muon+ (burn-fused) with the Qwen3.8-Flash-Next §3.1
//! param routing.
//!
//! Policy (report §3.1, validated by [`validate_routing`]):
//! - **Muon+ ColRow** on matrices that genuinely act as linear maps: KDA q/k/v,
//!   sparse-attention core q/k/v/out, expert TSCT u/v factors, Engram
//!   key/value projections.
//! - **Plain Adam, weight decay disabled** on the n-gram tables (§2.3).
//! - **AdamW** on everything else: embeddings, output head, routers/scorers
//!   (controller, attention blend, MSA indexer, halt head), per-head scalar
//!   producers (KDA decay/β gates), elongated low-rank readouts (out_proj),
//!   conv kernels and all 1D leaves. Orthogonalization is meaningless or
//!   harmful there (e.g. a 1-D output like the router score has no shared
//!   linear structure to exploit).
//!
//! Burn matches parameter groups by module path (field names joined by ".",
//! Vec entries as indices, e.g. `expert_ffns.0.gate_up.inner.u`), so the
//! policy is declared as path markers. Path strings are the framework's
//! mechanism; what keeps them honest is [`validate_routing`], which re-checks
//! the policy invariants against the live module tree at startup and fails
//! loudly when a marker goes stale (module renamed) instead of silently
//! falling back to AdamW.
//!
//! Selection via `OPT` env (runtime switch, no rebuild):
//! - `mix` (default): the policy above (AdamW fallback).
//! - `mix-adan`: same policy, but the fallback group runs Adan
//!   (arXiv 2208.06677: Nesterov momentum + gradient-difference correction,
//!   the "AdamW+" already shipped in burn-optim).
//! - `adan`: Adan on every param.
//! - `adamw`: legacy AdamW on every param (A/B baseline).
//! - `muon`: Muon+ on every param (its own AdamW fallback for 1D); debug
//!   mode, not the report recipe.
//!
//! `DM_FACTORS_FALLBACK=1` additionally drops the expert TSCT factors from
//! the Muon+ group (see [`effective_muon_markers`]).

use burn::{
    grad_clipping::GradientClippingConfig,
    module::{Module, ModuleVisitor, Param, ParamGroup},
    optim::{AdanConfig, AdamConfig, AdamWConfig},
    tensor::Tensor,
};
use burn_muon_plus::{MuonPlusConfig, NormDir};
use dormouse_core::DormouseModel;

use crate::{Optim, TrainCfg};

/// Newton-Schulz iterations for Muon+ orthogonalization. Report §3.1: 8 for
/// stability (the Muon+ paper default is 5).
pub const MUON_NS_STEPS: usize = 8;
/// Post-polar normalization direction (2602.21545: column-then-row is the
/// paper's best combination).
pub const MUON_NORM_DIR: Option<NormDir> = Some(NormDir::ColRow);

/// Muon+ group markers. Each marker must match at least one param of the
/// live model (checked by [`validate_routing`]).
///
/// Only SMALL matrices go to Muon+: the Newton-Schulz orthogonalization
/// costs ~3 matmuls per iteration on the FULL [m,n] matrix, and on this box
/// (5060 Ti, fp32, no tensor cores) 8 NS iters on [768,768] projections
/// added ~40 s per step. The low-rank TSCT factors are [d,64]/[64,f]:
/// orthogonalize in the factored [64,64] form, ~1000x cheaper. The dense
/// [d,d] attention projections stay on the fallback until the bf16 compute
/// path (tensor cores) is fixed.
pub(crate) const MUON_PATH_MARKERS: &[&str] = &[
    // Expert TSCT factors (report: fc1/fc2 of routed and shared experts).
    // The 1D scale leaf `s` is excluded from the group.
    "expert_ffns.",
    // Engram key projection (report: n-gram key/value projections on Muon;
    // value_proj is [d,d] here, excluded for the same cost reason).
    "engram.key_projs",
    // Loop readout projection (low-rank [d,64]/[64,d]).
    "out_proj.inner",
];

/// n-gram tables: plain Adam with weight decay disabled (report §2.3).
const ENGRAM_TABLE_MARKER: &str = "engram.memory";

/// True when a module param path belongs to the Muon+ group (2D only; the 1D
/// TSCT scale leaf `s` is excluded, end-anchored so `inner.s` is caught but
/// `inner.u`/`inner.v` are not).
pub fn is_muon_param(path: &str) -> bool {
    MUON_PATH_MARKERS.iter().any(|m| path.contains(m)) && !path.ends_with(".s")
}

/// True when a module param path is an n-gram table (plain Adam, no wd).
pub fn is_engram_table_param(path: &str) -> bool {
    path.contains(ENGRAM_TABLE_MARKER)
}

fn engram_table_group() -> ParamGroup {
    ParamGroup::from_predicate(ENGRAM_TABLE_MARKER)
}

fn muon_plus_cfg(cfg: &TrainCfg) -> MuonPlusConfig {
    MuonPlusConfig::new()
        .with_norm_dir(MUON_NORM_DIR)
        .with_ns_steps(MUON_NS_STEPS)
        .with_weight_decay(cfg.wd)
}

/// The Muon+ marker list in effect. `factors_fallback` drops the expert TSCT
/// factors from the Muon+ group (they are very elongated [d, r]/[r, f]
/// shapes; the report found elongated low-rank projections did better with
/// AdamW - A/B knob against the default Muon+ routing).
fn effective_muon_markers(factors_fallback: bool) -> Vec<&'static str> {
    let mut markers = MUON_PATH_MARKERS.to_vec();
    if factors_fallback {
        markers.retain(|m| *m != "expert_ffns.");
    }
    markers
}

/// Build the optimizer for `mode`, the pure core of [`build_optim`].
pub(crate) fn build_optim_mode(cfg: &TrainCfg, mode: &str) -> Optim {
    let clip = (cfg.grad_clip > 0.0).then_some(GradientClippingConfig::Norm(cfg.grad_clip as f32));
    let markers = effective_muon_markers(cfg.factors_fallback);
    let muon_group = || {
        ParamGroup::from_any_predicates(markers.clone())
            .exclude(ParamGroup::from_regex(r"\.s$").expect("valid regex"))
    };
    let muon_plus = || muon_plus_cfg(cfg).build();
    let mut opt = match mode {
        // Fallback optimizer candidates for the "rest" group.
        "adamw" => AdamWConfig::new().with_weight_decay(cfg.wd as f32).init(),
        "adan" => AdanConfig::new().with_weight_decay(cfg.wd as f32).init(),
        "muon" => muon_plus_cfg(cfg).init(),
        "mix-adan" => {
            let mut o = AdanConfig::new().with_weight_decay(cfg.wd as f32).init();
            o = o.with_group(muon_group(), muon_plus(), None);
            o = o.with_group(engram_table_group(), AdamConfig::new().build(), None);
            o
        }
        // mix (default): the report §3.1 recipe.
        _ => {
            let mut o = AdamWConfig::new().with_weight_decay(cfg.wd as f32).init();
            o = o.with_group(muon_group(), muon_plus(), None);
            o = o.with_group(engram_table_group(), AdamConfig::new().build(), None);
            o
        }
    };
    if let Some(c) = clip {
        // ModuleOptimizer::with_grad_clipping applies to the first optimizer.
        opt = opt.with_grad_clipping(c.init());
    }
    opt
}

/// Build the optimizer per OPT env:
/// `mix` (default) | `mix-adan` | `adamw` | `adan` | `muon`.
/// Grad clipping stays on the base (first) optimizer; the report's Muon
/// recipe does not clip (gates bound the activations, pre-clip norms stay
/// low), and Muon+ has no clipping hook.
pub fn build_optim(cfg: &TrainCfg) -> Optim {
    build_optim_mode(cfg, &cfg.opt)
}

/// Count parameters per optimizer group (Muon+ / Adam tables / AdamW rest).
/// Used by the startup banner and the routing tests.
#[derive(Default, Debug)]
pub struct GroupCounts {
    pub muon: usize,
    pub tables: usize,
    pub rest: usize,
    stack: Vec<String>,
}

impl ModuleVisitor for GroupCounts {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.stack.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.stack.pop();
    }
    fn visit_float<const D: usize>(&mut self, _param: &Param<Tensor<D>>) {
        let path = self.stack.join(".");
        if is_muon_param(&path) {
            self.muon += 1;
        } else if is_engram_table_param(&path) {
            self.tables += 1;
        } else {
            self.rest += 1;
        }
    }
}

/// Collect every float param path with its tensor rank.
#[derive(Default)]
struct PathCollector {
    stack: Vec<String>,
    paths: Vec<(String, usize)>,
}

impl ModuleVisitor for PathCollector {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.stack.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.stack.pop();
    }
    fn visit_float<const D: usize>(&mut self, _param: &Param<Tensor<D>>) {
        self.paths.push((self.stack.join("."), D));
    }
}

/// Re-check the routing policy against the live module tree. Fails loudly on
/// any violation of the policy invariants or on a marker that matches nothing
/// (stale after a module rename). Call once at startup, before training.
/// `factors_fallback` must match the value the optimizer was built with.
pub fn validate_routing(model: &DormouseModel, factors_fallback: bool) -> Result<GroupCounts, String> {
    let markers = effective_muon_markers(factors_fallback);
    validate_routing_with(model, &markers, ENGRAM_TABLE_MARKER)
}

pub(crate) fn validate_routing_with(
    model: &DormouseModel,
    markers: &[&str],
    table_marker: &str,
) -> Result<GroupCounts, String> {
    let is_muon = |p: &str| markers.iter().any(|m| p.contains(m)) && !p.ends_with(".s");
    let is_table = |p: &str| p.contains(table_marker);
    let mut collector = PathCollector::default();
    model.visit(&mut collector);
    let mut counts = GroupCounts::default();
    for (path, rank) in &collector.paths {
        let muon = is_muon(path);
        let table = is_table(path);
        if muon && table {
            return Err(format!("{path}: matches both Muon+ and table groups"));
        }
        if muon && *rank == 1 {
            return Err(format!("{path}: 1D param routed to Muon+"));
        }
        if muon {
            counts.muon += 1;
        } else if table {
            counts.tables += 1;
        } else {
            counts.rest += 1;
        }
        // Policy invariants: expert TSCT factors are always Muon+ (unless the
        // DM_FACTORS_FALLBACK knob removed them), n-gram tables always plain
        // Adam.
        if markers.iter().any(|m| *m == "expert_ffns.")
            && path.contains("expert_ffns.")
            && (path.ends_with(".u") || path.ends_with(".v"))
        {
            if !muon {
                return Err(format!("{path}: expert TSCT factor must be Muon+"));
            }
        }
        if path.contains(table_marker) && !table {
            return Err(format!("{path}: n-gram table must be on plain Adam"));
        }
    }
    // No dead markers: every marker must match at least one live param, so a
    // module rename surfaces here instead of silently degrading to AdamW.
    for marker in markers {
        if !collector.paths.iter().any(|(p, _)| p.contains(marker)) {
            return Err(format!("Muon+ marker {marker:?} matches no param (renamed?)"));
        }
    }
    if !collector.paths.iter().any(|(p, _)| p.contains(table_marker)) {
        return Err(format!("table marker {table_marker:?} matches no param (renamed?)"));
    }
    Ok(counts)
}