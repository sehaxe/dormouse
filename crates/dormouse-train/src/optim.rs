//! Optimizer policy: Muon+ (burn-fused) with the Qwen3.8-Flash-Next §3.1
//! param routing.
//!
//! Policy (report §3.1, validated by [`validate_routing`]):
//! - **Muon+ ColRow** on matrices that genuinely act as linear maps: KDA q/k/v,
//!   sparse-attention core q/k/v/out, expert TSCT u/v factors, Engram
//!   key/value projections.
//! - **Head-wise Muon+** on the attention Q/K weights (report §3.1: split
//!   qkv per head BEFORE orthogonalization - fusing mixes singular
//!   directions): each `[head_dim, d]` block gets its own NS
//!   preconditioner ([`HeadWiseMuon`]). Enabled by `qk_heads` (set from the
//!   preset's n_heads in the train loop).
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
//! Selection via `--opt` (runtime switch, no rebuild):
//! - `mix` (default): the policy above (AdamW fallback).
//! - `mix-adan`: same policy, but the fallback group runs Adan
//!   (arXiv 2208.06677: Nesterov momentum + gradient-difference correction,
//!   the "AdamW+" already shipped in burn-optim).
//! - `adan`: Adan on every param.
//! - `adamw`: legacy AdamW on every param (A/B baseline).
//! - `muon`: Muon+ on every param (its own AdamW fallback for 1D); debug
//!   mode, not the report recipe.
//!
//! `--factors-fallback` additionally drops the expert TSCT factors from
//! the Muon+ group (see [`effective_muon_markers`]).

use burn::{
    grad_clipping::GradientClippingConfig,
    module::{Module, ModuleVisitor, Param, ParamGroup},
    optim::{AdanConfig, AdamConfig, AdamWConfig, LearningRate, Optimizer},
    tensor::{Device, ElementConversion, Tensor},
};
use burn_muon_plus::{MuonPlus, MuonPlusConfig, MuonPlusState, NormDir};
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

/// Q/K paths routed to the head-wise Muon group (per-head preconditioner).
/// The k-side of the sparse attention has GQA heads (`n_kv = n_heads/4`),
/// so it forms its own group with a different head count. The MQA indexer
/// (msa.index_branch) stays on the fallback: 4 q-heads / 1 shared k-head,
/// tiny matrices with ambiguous per-head semantics.
pub(crate) const QK_HEAD_MARKERS: &[&str] = &[
    "gdn2.q_proj",
    "gdn2.k_proj",
    "msa.attention.q_proj",
];
pub(crate) const QK_KV_MARKER: &str = "msa.attention.k_proj";

/// True when a param path belongs to a head-wise Q/K group (2D only).
pub fn is_qk_param(path: &str) -> bool {
    QK_HEAD_MARKERS.iter().any(|m| path.contains(m)) || path.contains(QK_KV_MARKER)
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
/// Q/K here are separate Linears (burn-kda gdn2, burn-msa attention), so
/// the per-head split is unambiguous - there is no fused [d, 3d] qkv to
/// disambiguate.
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
            o = with_qk_groups(o, cfg);
            o
        }
        // mix (default): the report §3.1 recipe.
        _ => {
            let mut o = AdamWConfig::new().with_weight_decay(cfg.wd as f32).init();
            o = o.with_group(muon_group(), muon_plus(), None);
            o = o.with_group(engram_table_group(), AdamConfig::new().build(), None);
            o = with_qk_groups(o, cfg);
            o
        }
    };
    if let Some(c) = clip {
        // ModuleOptimizer::with_grad_clipping applies to the first optimizer.
        opt = opt.with_grad_clipping(c.init());
    }
    opt
}

/// Route the attention Q/K projections to head-wise Muon when `qk_heads`
/// is set (the preset's `n_heads`); the GQA k-side gets the reduced head
/// count. Unset keeps Q/K on the base fallback optimizer.
fn with_qk_groups(opt: Optim, cfg: &TrainCfg) -> Optim {
    let Some(h) = cfg.qk_heads else { return opt };
    let n_kv = (h / 4).max(1); // must mirror AdaptiveAttention::new
    let mut o = opt;
    o = o.with_group(
        ParamGroup::from_any_predicates(QK_HEAD_MARKERS.to_vec()),
        HeadWiseMuon::new(cfg, h),
        None,
    );
    o = o.with_group(
        ParamGroup::from_any_predicates(vec![QK_KV_MARKER]),
        HeadWiseMuon::new(cfg, n_kv),
        None,
    );
    o
}

/// Build the optimizer per OPT env:
/// `mix` (default) | `mix-adan` | `adamw` | `adan` | `muon`.
/// Grad clipping stays on the base (first) optimizer; the report's Muon
/// recipe does not clip (gates bound the activations, pre-clip norms stay
/// low), and Muon+ has no clipping hook.
pub fn build_optim(cfg: &TrainCfg) -> Optim {
    build_optim_mode(cfg, &cfg.opt)
}

/// Count parameters per optimizer group (Muon+ / head-wise Q/K / Adam
/// tables / AdamW rest). Used by the startup banner and the routing tests.
#[derive(Default, Debug)]
pub struct GroupCounts {
    pub muon: usize,
    pub qk: usize,
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
        } else if is_qk_param(&path) {
            self.qk += 1;
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
/// `factors_fallback` must match the value the optimizer was built with;
/// `qk_heads` must be `Some(n_heads)` iff the head-wise Q/K groups were
/// enabled.
pub fn validate_routing(
    model: &DormouseModel,
    factors_fallback: bool,
    qk_heads: Option<usize>,
) -> Result<GroupCounts, String> {
    let markers = effective_muon_markers(factors_fallback);
    validate_routing_with(model, &markers, ENGRAM_TABLE_MARKER, qk_heads)
}

pub(crate) fn validate_routing_with(
    model: &DormouseModel,
    markers: &[&str],
    table_marker: &str,
    qk_heads: Option<usize>,
) -> Result<GroupCounts, String> {
    let is_muon = |p: &str| markers.iter().any(|m| p.contains(m)) && !p.ends_with(".s");
    let is_table = |p: &str| p.contains(table_marker);
    // The Q/K groups exist only when head-wise routing is on; otherwise the
    // params stay in the base fallback group like before.
    let is_qk = |p: &str| qk_heads.is_some() && is_qk_param(p);
    let mut collector = PathCollector::default();
    model.visit(&mut collector);
    let mut counts = GroupCounts::default();
    for (path, rank) in &collector.paths {
        let muon = is_muon(path);
        let table = is_table(path);
        let qk = is_qk(path);
        if muon && table {
            return Err(format!("{path}: matches both Muon+ and table groups"));
        }
        if muon && qk || table && qk {
            return Err(format!("{path}: matches multiple optimizer groups"));
        }
        if muon && *rank == 1 {
            return Err(format!("{path}: 1D param routed to Muon+"));
        }
        if qk && *rank != 2 {
            return Err(format!("{path}: non-2D param routed to the Q/K head-wise group"));
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
        // Policy invariants: expert TSCT factors are always Muon+ (unless
        // --factors-fallback removed them), n-gram tables always plain Adam.
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
    if qk_heads.is_some() {
        for marker in QK_HEAD_MARKERS.iter().chain(std::iter::once(&QK_KV_MARKER)) {
            if !collector.paths.iter().any(|(p, _)| p.contains(marker)) {
                return Err(format!("Q/K marker {marker:?} matches no param (renamed?)"));
            }
        }
    }
    Ok(counts)
}