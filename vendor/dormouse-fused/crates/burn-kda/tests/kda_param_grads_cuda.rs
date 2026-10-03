//! PER-PARAMETER-GROUP gradient proof for the KDA attention arm, on CUDA, on
//! the trainer's backend — the check `8fa5d4c` said it had not done.
//!
//! ## What this is for
//!
//! For the whole history of this project the attention arm received **no
//! gradient at all**. `chunk_wy_forward_autodiff_s` ran the forward on the bare
//! backend and wrapped the result in one hand-rolled autodiff node; under
//! `BalancedCheckpointing` (dormouse's strategy) that node was `UnTracked`, its
//! output was a LEAF, and nothing downstream could send a gradient back. A real
//! run printed `fused kda=30/0` — 30 fused forwards, 0 backwards — while every
//! other arm trained normally, so the loss curve looked healthy. See
//! `AGENTS.md` §3.2.
//!
//! The fix is committed (`8fa5d4c`) and **its own message says the
//! gradient-flowing check was not done.** It has still not been done. This file
//! is that check.
//!
//! ## Why per GROUP and not one aggregate
//!
//! The existing strong gate, `tests/ops_grad_cuda.rs`, differentiates exactly
//! one parameter — `q_proj.weight` — against central differences. That is a
//! real test of one tensor. It leaves the class this defect lives in open: a
//! backward that reaches the first projection and stops. Every group here is
//! checked SEPARATELY, and each carries its own named verdict, so a regression
//! names the group that stopped rather than reporting "some grad exists".
//!
//! The layout of the groups also makes a falsification legible. `o_gate` and
//! `o_proj` are applied AFTER the chunk recurrence, so a break inside the chunk
//! op leaves those two with a gradient and takes it away from the other nine.
//! `docs/reviews/kda-gradflow-2026-09-30.md` records the run: breaking the
//! op's output into a leaf turns exactly those two green.
//!
//! ## Where this file lives, and why not in burn-gdn2
//!
//! The arm the trainer runs is `burn_kda::KdaModule`
//! (`crates/dormouse-core/src/attention.rs:55`, reached from `loop_block.rs:397`
//! via `forward_train_state`). Its parameter names are the ones that matter
//! here — `decay.a_log` / `decay.b_alpha` / `o_gate` / `o_norm_w`. burn-gdn2's
//! own `GatedDeltaNet2` has a different set (`a_log` + `dt_bias`, no
//! `b_alpha`) and is not what the trainer builds, and burn-gdn2 cannot depend
//! on burn-kda without a cycle. The seam counters are reachable from here
//! because burn-kda already depends on burn-gdn2.
//!
//! ## Which arm this pins
//!
//! `forward_train_state` dispatches through `burn_gdn2::chunk_dispatch`, which
//! either builds ONE autodiff node over the fused kernels (fused arm) or
//! declines and hands the incoming tensors to `chunk_wy_forward_impl` so burn
//! builds the graph itself (ops arm). `tests/cuda_gate.rs` already gates that
//! the trainer's `BalancedCheckpointing` strategy **declines** the fused op and
//! runs the ops path, and this file's run log confirms it on a real module
//! (`fused_fwd=0 ops_path=1` with `DM_FUSED_KDA` unset). So the arm the trainer
//! runs is the OPS one, and that is the arm the per-group proof is about.
//! `NoCheckpointing` is pinned here too, because on that strategy the op's
//! intermediates are real graph nodes and the fused op IS reached
//! (`fused_fwd=1 fused_bwd=1 custom_node_bwd=1 ops_path=0`). There the numerics
//! are NOT asserted and the reason is measured: the fused adjoint's decay-path
//! gradients vary 7-27% between identical runs and disagree with a central
//! difference of its own forward by 1.5e-1..2.6e-1. That test therefore asserts
//! FLOW and the arm, and PRINTS the disagreement on every run.
//!
//! ## The two references, and what each is for
//!
//! Gradients here span four orders of magnitude — `v_proj.weight` peaks near
//! 1e1, `decay.w_up.weight` near 1e-4 — because the decay path is two stacked
//! 0.02-std projections. So:
//!
//! 1. **Central differences of the same forward on the same weights**, with the
//!    loss accumulated in **f64 on the host** (differencing two f32 sums cancels
//!    catastrophically at this loss scale). Same function, different method of
//!    differentiation: no autodiff in the reference, no second implementation,
//!    no reference file. A zero gradient fails it, a wrong gradient fails it, a
//!    gradient of a different function fails it. The step is derived per group
//!    ([`fd_step`]) rather than fixed, because a fixed step is one group's step.
//!    Groups whose FD the instrument cannot resolve are named as such and
//!    fall to reference 2 on a TIGHTER bar — never to a pass.
//! 2. **The same module on CPU `NdArray`**, same weights, same input values,
//!    its own autodiff graph. Exact rather than truncated, and a different
//!    device and a different chunk implementation, so it covers the groups
//!    reference 1 cannot reach and cross-checks the fused CUDA forward against
//!    the tensor-ops one. It CANNOT catch a wrong formula (both arms run the
//!    same `chunk_wy_forward_impl`) — which is exactly why reference 1 exists.
//!
//! WHAT NEITHER IS: an f64 forward. `KdaModule` is not generic over its float
//! element and `NdArray` is f32-only in burn 0.22
//! (`burn-ndarray-0.22.0-pre.4/src/backend.rs:55`, `pub struct NdArray;`, with
//! `DType::F32` in `NdArrayDevice::defaults`), so an f64 module would mean
//! transcribing the whole KDA block — the thing this project has been burned
//! by twice. What replaces it is stated rather than assumed: the resolution
//! floor of reference 1 is derived from its own noise model and printed, and a
//! group below it is reported as unresolved instead of compared.
//!
//! Run:
//! ```text
//! cargo test -p burn-kda --features cuda,autodiff \
//!     --test kda_param_grads_cuda -- --nocapture --test-threads=1
//! ```
#![cfg(all(feature = "cuda", feature = "autodiff"))]
#![allow(deprecated)]

use std::sync::Mutex;

use burn::backend::NdArray;
use burn::module::Param;
use burn::tensor::{Device, Distribution, Tensor, TensorData};
use burn_autodiff::checkpoint::strategy::{BalancedCheckpointing, NoCheckpointing};
use burn_autodiff::Autodiff;
use burn_gdn2::CudaBare;
use burn_kda::{DecayFn, KdaConfig, KdaModule};

/// The trainer's backend: `Autodiff<CudaBare, NoCheckpointing>` — which is
/// what `Device::cuda(0).autodiff()` builds (`Enabled(Disabled)` and
/// `NoCheckpointing::STRATEGY == Disabled`, burn-dispatch
/// `src/tensor.rs:449-460`). The trainer's TYPE said `BalancedCheckpointing`
/// until 2026-10-02 while its device ran `Disabled`; the type governed
/// nothing (burn's ops are type-erased) except the seams that name it in a
/// conversion, where it silently discarded every fused result
/// (`fused kda=412/0 ops=492` on the probe run). See the Backend comment in
/// `dormouse-train/src/lib.rs`. This file keeps BOTH strategies pinned: AdBal
/// exercises the Balanced conversions, AdNo the trainer's actual pair.
type AdBal = Autodiff<CudaBare, BalancedCheckpointing>;
/// The same module on the same strategy with every intermediate a real graph
/// node, which is what makes the fused op reachable.
type AdNo = Autodiff<CudaBare, NoCheckpointing>;
/// `Tensor::backward()`'s return type. NOT
/// `<AdBal as AutodiffBackend>::Gradients` (that is `burn_autodiff::grads::
/// Gradients`) and NOT `burn_autodiff::AutodiffBackend::Gradients` — three
/// types with the same name, and `Param::grad` takes the `burn-tensor` one.
type Grads = burn::tensor::Gradients;

// ── the shape under test ───────────────────────────────────────────────

/// Batch/sequence. The trainer's real seq_len is 512 and its controls run
/// batch 8-10; this is `[2, 256, D]` = 512 tokens, 16 chunks at `chunk_size`
/// 16, so every chunk boundary in the recurrence is crossed. The defect this
/// test exists for is gradient FLOW, which does not depend on the token count,
/// and the reference cost is linear in forwards, so the token count is the one
/// knob worth spending down. The full-size run is the trainer's own evidence.
const B: usize = 2;
const T: usize = 256;

/// The trainer's `AdaptiveAttention` shape: `use_short_conv: false` (it NaN'd
/// at ~step 60 with the conv on — `attention.rs:65-70`), `GateMode::FullRank`,
/// `DecayFn::Sigmoid`, `chunk_size: 16`. `hidden_size/num_heads/head_dim` are
/// the ones `tests/ops_grad_cuda.rs` established, so the two files are
/// comparable line for line.
fn cfg_trainer() -> KdaConfig {
    KdaConfig {
        hidden_size: 64,
        num_heads: 2,
        head_dim: 16,
        num_v_heads: Some(2),
        use_short_conv: false,
        decay_fn: DecayFn::Sigmoid,
        chunk_size: 16,
        ..Default::default()
    }
}

/// The same with the short conv ON — the only way `q/k/v_conv_w` are ever
/// exercised, since the trainer keeps them off. On the trainer's own shape
/// those three groups are `None` and do not exist, which
/// [`every_group_is_present_for_the_config`] asserts rather than skips past.
fn cfg_short_conv() -> KdaConfig {
    KdaConfig {
        use_short_conv: true,
        ..cfg_trainer()
    }
}

// ── bars, all derived ──────────────────────────────────────────────────

/// f32 eps — the floor of every number this file produces.
const EPS_F32: f64 = f32::EPSILON as f64;

/// Relative bar for |analytic − central difference| against the difference.
///
/// A BAR, not a measured value: it is what central differences can deliver in
/// f32 with TF32 matmuls on this GPU, and it is the bar the existing
/// single-parameter gate already uses (`tests/ops_grad_cuda.rs`, `REL_BAR`),
/// so a disagreement between the two files is a finding about one of them. It
/// is NOT widened to accommodate a disagreement.
const FD_BAR: f64 = 5e-2;

/// The bar a group falls to when its finite differences are UNRESOLVABLE —
/// five times TIGHTER than the resolved groups' bar against the CPU
/// reference, because an unresolved group is being carried entirely by
/// reference 2 and a bare pass at the same bar would be a silent fallback.
/// Ten times tighter than [`CPU_BAR`]: the margin is the price of not being
/// able to cross-check it.
const CPU_BAR_UNRESOLVED_FD: f64 = 5e-3;

/// Relative bar for the CUDA gradient against the CPU `NdArray` gradient, per
/// entry, over the WHOLE tensor (not a sample: the reference is exact and free
/// at these sizes, so there is no reason to sample).
///
/// Derived from the arithmetic, not from a run: two f32 evaluations of the
/// same expression on different devices differ by the accumulated rounding of
/// both, ~`2·eps_f32` per elementary op, amplified over the recurrence's depth
/// — 16 chunks of the WY form, each a handful of chained matmuls. At
/// `16 · 10 · 2 · eps_f32 ≈ 4e-5` the floor is three orders below this bar,
/// so the bar is set by the ill-conditioning of a delta rule near its
/// eigenvalue, not by f32: `‖I − k(k∘β)ᵀ‖` approaches 0 when the erase gate
/// saturates, and then a relative perturbation of the state is amplified
/// without bound. A bar that cannot separate "different rounding" from
/// "different function" is not a bar; this one is asserted, not hoped for.
const CPU_BAR: f64 = 5e-2;

/// Bar for the two references' FORWARD agreeing, which is the precondition
/// for comparing their gradients. Derived like [`CPU_BAR`]: two f32 evaluations
/// of the same expression on different devices, reassociating over a 16-chunk
/// recurrence. An order of magnitude looser than the gradient bar because the
/// forward is a shorter chain than the backward — and a precondition is not
/// worth a tight bar, it is worth being true.
const LOSS_BAR: f64 = 1e-3;

/// Spread coordinates per group, on top of the analytic argmax. The argmax is
/// where the signal is largest, so where a relative comparison means most; the
/// spread ones are there so a bug local to one region of a matrix cannot hide
/// behind the argmax alone.
const N_SPREAD: usize = 2;

// ── host helpers ───────────────────────────────────────────────────────

fn host<const D: usize>(t: &Tensor<D>) -> Vec<f32> {
    t.clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// ── the parameters ─────────────────────────────────────────────────────

type Setter = Box<dyn Fn(&mut KdaModule, Vec<f32>, &[usize], &Device)>;

/// One named parameter: its host values, its shape, and a way to put values
/// back. The NAME is what a failure prints, which is the entire reason this
/// type exists instead of a loop over `ParamId`s.
struct Group {
    name: &'static str,
    val: Vec<f32>,
    shape: Vec<usize>,
    set: Setter,
}

impl Group {
    fn rms(&self) -> f64 {
        (self
            .val
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            / self.val.len() as f64)
            .sqrt()
    }
}

fn t1(v: Vec<f32>, sh: &[usize], d: &Device) -> Tensor<1> {
    Tensor::<1>::from_data(TensorData::new(v, [sh[0]]), d)
}

fn t2(v: Vec<f32>, sh: &[usize], d: &Device) -> Tensor<2> {
    Tensor::<2>::from_data(TensorData::new(v, [sh[0], sh[1]]), d)
}

/// One parameter group with its gradient. A parameter with no gradient is
/// carried as `None` and reported as such — the 8fa5d4c signature — rather
/// than skipped. `grads: None` builds the list for the CPU reference module,
/// which is a scaffold for installing the CUDA weights and nothing else.
fn p<const D: usize>(
    name: &'static str,
    bp: &Param<Tensor<D>>,
    grads: Option<&Grads>,
    set: impl Fn(&mut KdaModule, Vec<f32>, &[usize], &Device) + 'static,
) -> (Group, Option<Vec<f32>>) {
    let v = bp.val();
    (
        Group {
            name,
            val: host(&v),
            shape: v.shape().dims::<D>().to_vec(),
            set: Box::new(set),
        },
        grads.and_then(|g| bp.grad(g)).map(|t| host(&t)),
    )
}

/// Every KDA-side parameter group with its gradient, in the order `KdaModule`
/// declares them.
///
/// NOT a reflection over the module: an explicit list is the point. A parameter
/// added to the struct and forgotten here is one nothing checks, and a
/// reflection would silently grow to cover it while the LIST — the thing a
/// reader audits against the struct — stopped matching.
fn params(m: &KdaModule, grads: Option<&Grads>) -> Vec<(Group, Option<Vec<f32>>)> {
    let mut v: Vec<(Group, Option<Vec<f32>>)> = vec![
        p("q_proj.weight", &m.q_proj.weight, grads, |m, v, s, d| {
            m.q_proj.weight = Param::from_tensor(t2(v, s, d));
        }),
        p("k_proj.weight", &m.k_proj.weight, grads, |m, v, s, d| {
            m.k_proj.weight = Param::from_tensor(t2(v, s, d));
        }),
        p("v_proj.weight", &m.v_proj.weight, grads, |m, v, s, d| {
            m.v_proj.weight = Param::from_tensor(t2(v, s, d));
        }),
        p(
            "decay.w_up.weight",
            &m.decay.w_up.weight,
            grads,
            |m, v, s, d| {
                m.decay.w_up.weight = Param::from_tensor(t2(v, s, d));
            },
        ),
        p(
            "decay.w_down.weight",
            &m.decay.w_down.weight,
            grads,
            |m, v, s, d| {
                m.decay.w_down.weight = Param::from_tensor(t2(v, s, d));
            },
        ),
        p("decay.b_alpha", &m.decay.b_alpha, grads, |m, v, s, d| {
            m.decay.b_alpha = Param::from_tensor(t1(v, s, d));
        }),
        p("decay.a_log", &m.decay.a_log, grads, |m, v, s, d| {
            m.decay.a_log = Param::from_tensor(t2(v, s, d));
        }),
        p(
            "beta_proj.weight",
            &m.beta_proj.weight,
            grads,
            |m, v, s, d| {
                m.beta_proj.weight = Param::from_tensor(t2(v, s, d));
            },
        ),
        p("o_norm_w", &m.o_norm_w, grads, |m, v, s, d| {
            m.o_norm_w = Param::from_tensor(t1(v, s, d));
        }),
        p("o_proj.weight", &m.o_proj.weight, grads, |m, v, s, d| {
            m.o_proj.weight = Param::from_tensor(t2(v, s, d));
        }),
    ];

    // The output gate. `GateMode::FullRank` (the trainer's default) has one
    // matrix; `LowRank` has a pair. Both are this arm under one flag, so both
    // are listed and the absent one is asserted, not skipped.
    if let Some(g) = &m.o_gate {
        v.push(p("o_gate.weight", &g.weight, grads, |m, v, s, d| {
            let mut l = m.o_gate.clone().unwrap();
            l.weight = Param::from_tensor(t2(v, s, d));
            m.o_gate = Some(l);
        }));
    }
    if let Some(g) = &m.o_gate_up {
        v.push(p("o_gate_up.weight", &g.weight, grads, |m, v, s, d| {
            let mut l = m.o_gate_up.clone().unwrap();
            l.weight = Param::from_tensor(t2(v, s, d));
            m.o_gate_up = Some(l);
        }));
    }
    if let Some(g) = &m.o_gate_down {
        v.push(p("o_gate_down.weight", &g.weight, grads, |m, v, s, d| {
            let mut l = m.o_gate_down.clone().unwrap();
            l.weight = Param::from_tensor(t2(v, s, d));
            m.o_gate_down = Some(l);
        }));
    }

    // The short conv. `None` on the trainer's shape, so there the prompt's
    // "and the short conv if present" resolves to three parameters that do not
    // exist — the short-conv test below is the config where they do.
    if let Some(q) = &m.q_conv_w {
        v.push(p("q_conv_w", q, grads, |m, v, s, d| {
            m.q_conv_w = Some(Param::from_tensor(t2(v, s, d)));
        }));
    }
    if let Some(k) = &m.k_conv_w {
        v.push(p("k_conv_w", k, grads, |m, v, s, d| {
            m.k_conv_w = Some(Param::from_tensor(t2(v, s, d)));
        }));
    }
    if let Some(vw) = &m.v_conv_w {
        v.push(p("v_conv_w", vw, grads, |m, v, s, d| {
            m.v_conv_w = Some(Param::from_tensor(t2(v, s, d)));
        }));
    }

    v
}

// ── the check, for one checkpointing strategy ──────────────────────────

/// Serialises the tests in this file: the seam counters are PROCESS-GLOBAL
/// statics, and `DM_FUSED_KDA` is read out of the environment on EVERY fused
/// dispatch — which is inside the module's forward. A second test flipping it
/// while this one is between forwards would move this one's arm mid-run, and
/// the arm is the thing under test.
static SERIAL: Mutex<()> = Mutex::new(());

/// What one group looks like after the check. Printed for every group on every
/// run, so the table is in the log and not only in a failure message.
struct Verdict {
    name: &'static str,
    non_zero: bool,
    finite: bool,
    amax: f32,
    /// Worst relative deviation from the CPU `NdArray` gradient, whole tensor.
    cpu_rel: f64,
    /// The two numbers that ratio is built from, printed so a `cpu_rel` of
    /// exactly 0 can be told apart from a comparison that never ran.
    cpu_maxdiff: f64,
    cpu_amax: f64,
    /// Worst relative deviation from central differences, when the instrument
    /// resolved them; `None` with a reason otherwise.
    fd: Result<(f64, f64), String>,
}

/// The central-difference step for one group, DERIVED and not fixed.
///
/// Two forces set it. Round-off on the difference of the two losses is
/// ~`eps_f32·|L|/h`, which wants `h` LARGE; the perturbation must also stay in
/// the linear regime of the forward, which wants `h` SMALL relative to the
/// parameter's own scale — `decay.w_up` is a 0.02-std matrix, and a step
/// anywhere near its RMS is not a derivative measurement, it is a different
/// network. So `h` is the cube root that balances round-off against a
/// first-order truncation term — the standard FD optimum, `h* = (3·eps·|L|/
/// |g|)^(1/3)`, which for the groups where the old fixed `H = 1e-2` was
/// already right reproduces it to within a factor of ~1.3 — clamped to 5% of
/// the parameter's RMS. The clamp is the binding constraint for the
/// small-gain decay path, and the reason those groups are reported as
/// unresolved rather than compared.
fn fd_step(prm: &Group, gmax: f64, loss: f64) -> (f64, &'static str) {
    let optimum = (3.0 * EPS_F32 * loss / gmax.max(1e-300)).cbrt();
    let linear = 0.05 * prm.rms();
    if optimum <= linear {
        (optimum, "round-off/truncation optimum")
    } else {
        (linear, "clamped to 5% of the parameter RMS (linear regime)")
    }
}

/// Coordinates: the analytic argmax, then `N_SPREAD` spread over the tensor.
/// Deterministic, and printed, so a rerun checks the same ones.
fn coords(g: &[f32]) -> Vec<usize> {
    let mut c = vec![g
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map(|(i, _)| i)
        .unwrap_or(0)];
    let step = (g.len() / N_SPREAD).max(1);
    // dedup: a group with fewer entries than coordinates would otherwise print
    // the same index twice and look like more evidence than it is.
    for k in 1..=N_SPREAD {
        let i = (k * step) % g.len();
        if !c.contains(&i) {
            c.push(i);
        }
    }
    c
}

#[allow(clippy::too_many_arguments)]
fn check_group<S>(
    cuda: &KdaModule,
    x: &Tensor<3>,
    prm: &Group,
    grad: Option<Vec<f32>>,
    cpu: Option<&[f32]>,
    loss: f64,
    dev: &Device,
) -> Verdict
where
    S: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<Autodiff<CudaBare, S>>,
{
    let name = prm.name;
    let grad = match grad {
        // No gradient at all is the 8fa5d4c signature. Named, not counted.
        None => {
            return Verdict {
                name,
                non_zero: false,
                finite: true,
                amax: 0.0,
                cpu_rel: f64::INFINITY,
                cpu_maxdiff: f64::NAN,
                cpu_amax: f64::NAN,
                fd: Err("no gradient tensor".into()),
            }
        }
        Some(v) => v,
    };
    let amax = grad.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let finite = grad.iter().all(|v| v.is_finite());

    // Reference 2: the CPU `NdArray` gradient of the same module, whole tensor.
    let (cpu_rel, cpu_maxdiff, cpu_amax) = match cpu {
        None => (f64::INFINITY, f64::NAN, f64::NAN),
        Some(c) => {
            let (r, d, m) = worst_rel(&grad, c);
            (r, d, m)
        }
    };

    // Reference 1: central differences, at this group's own derived step.
    let fd = reference_fd::<S>(cuda, x, prm, &grad, loss, dev, amax as f64);

    Verdict {
        name,
        non_zero: amax > 0.0,
        finite,
        amax,
        cpu_rel,
        cpu_maxdiff,
        cpu_amax,
        fd,
    }
}

/// `max|a − b| / max|b|` over the whole tensor — the measure every other
/// fused test in this library uses (`rel_diff` in `autodiff_cuda_gate.rs`,
/// `rel` in `ref_f64.rs`).
///
/// NOT per-entry relative, which is what this file did first and is useless
/// here: a gradient tensor has entries spanning three orders of magnitude
/// within one group, and an entry where the reference is 1e-7 and the
/// measurement 1e-4 scores 1e3 while telling you nothing. Measured before the
/// fix: every group read 1e1..1e4 on a comparison that was in fact agreeing to
/// 1e-6. Relative to the tensor's own scale, an entry is judged by how much it
/// contributes to the gradient, which is the question.
fn worst_rel(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    assert_eq!(
        a.len(),
        b.len(),
        "gradient shapes differ between references"
    );
    let scale = b.iter().fold(0.0f64, |m, y| m.max((*y as f64).abs()));
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
        .fold(0.0f64, f64::max);
    (worst / scale.max(1e-30), worst, scale)
}

/// Central differences against the same forward, at [`fd_step`] and a tenth of
/// it. Returns the worst relative deviation, or the reason the instrument
/// cannot resolve this group.
fn reference_fd<S>(
    cuda: &KdaModule,
    x: &Tensor<3>,
    prm: &Group,
    grad: &[f32],
    loss: f64,
    dev: &Device,
    gmax: f64,
) -> Result<(f64, f64), String>
where
    S: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<Autodiff<CudaBare, S>>,
{
    let (h, why) = fd_step(prm, gmax, loss);
    // The instrument's own resolution: the difference of the two losses is
    // `2h·|g|`, and the forward carries f32 relative error, so the difference
    // carries `eps_f32·|L|` of noise. The factor 32 bounds the accumulated
    // error over the ~10⁴ terms of the loss. If the signal `2h·|g|` is at or
    // under that, the numbers below are noise and this group is UNRESOLVED.
    let signal = 2.0 * h * gmax;
    let noise = 32.0 * EPS_F32 * loss;
    if signal <= noise {
        return Err(format!(
            "h={h:.2e} ({why}): signal 2h|g| = {signal:.2e} <= the instrument's noise {noise:.2e} \
             (32*eps_f32*|L|, |L| = {loss:.3e})"
        ));
    }

    let mut worst = 0.0f64;
    for idx in coords(grad) {
        let (d, d10) = (
            fd_at::<S>(cuda, x, prm, idx, dev, h),
            fd_at::<S>(cuda, x, prm, idx, dev, h / 10.0),
        );
        let a = grad[idx] as f64;
        let rel = (a - d).abs() / d.abs().max(1e-30);
        let rel10 = (a - d10).abs() / d10.abs().max(1e-30);
        // The MIN of the two step sizes: a truncation-dominated residual falls
        // with h, a round-off-dominated one rises, and taking the min refuses a
        // value that only looks good at one of them.
        worst = worst.max(rel.min(rel10));
        println!(
            "  {nm:<20} [{idx:5}] h={h:.2e}  autodiff {a:+.6e}  fd(h) {d:+.6e}  fd(h/10) \
             {d10:+.6e}  rel {rel:.2e}  rel(h/10) {rel10:.2e}",
            nm = prm.name
        );
    }
    Ok((h, worst))
}

fn fd_at<S>(base: &KdaModule, x: &Tensor<3>, prm: &Group, idx: usize, dev: &Device, h: f64) -> f64
where
    S: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<Autodiff<CudaBare, S>>,
{
    let mut up = base.clone();
    (prm.set)(&mut up, shifted(&prm.val, idx, h), &prm.shape, dev);
    let mut dn = base.clone();
    (prm.set)(&mut dn, shifted(&prm.val, idx, -h), &prm.shape, dev);
    (cuda_loss::<S>(&up, x) - cuda_loss::<S>(&dn, x)) / (2.0 * h)
}

fn shifted(w: &[f32], idx: usize, h: f64) -> Vec<f32> {
    let mut v = w.to_vec();
    v[idx] = (v[idx] as f64 + h) as f32;
    v
}

fn show(c: (u64, u64, u64, u64, u64, u64)) -> String {
    format!(
        "asked={} fused_fwd={} fused_bwd={} declined={} ops_path={} custom_node_bwd={}",
        c.0, c.1, c.2, c.3, c.4, c.5
    )
}

fn arm_name(c: (u64, u64, u64, u64, u64, u64)) -> &'static str {
    if c.2 > 0 {
        "FUSED adjoint kernel (ChunkWy::backward)"
    } else if c.4 > 0 {
        "OPS path (burn's own graph)"
    } else {
        "NEITHER — no backward ran at all"
    }
}

// ── the driver ─────────────────────────────────────────────────────────

/// One device-seeded module and input, so every test in the file starts from
/// the same weights and the finite differences are reproducible across runs
/// and processes.
struct Fixture {
    cuda: Device,
    cpu: Device,
    cfg: KdaConfig,
    m: KdaModule,
}

fn fixture(cfg: KdaConfig) -> Fixture {
    let cuda = Device::cuda(0);
    cuda.seed(11); // unseeded draws make the finite differences irreproducible
    let cpu = Device::default(); // the CPU device, whatever this build calls it
    let m = KdaModule::new(&cfg, 0.9, &cuda.clone().autodiff());
    Fixture { cuda, cpu, cfg, m }
}

/// The input, TWICE and with the SAME values: drawn on CUDA once, then
/// installed on CPU from the host bytes, so the two references see bit-equal
/// input rather than two draws from the same distribution. Both halves travel
/// together to the caller on purpose — an earlier version let `run` call this
/// again for the CPU side, which advanced the RNG and handed the CPU reference
/// a DIFFERENT sequence than the CUDA gradient was taken on. Every group then
/// read `cpu_rel` ~ 1, which reads exactly like a real disagreement and is not
/// one. A reference that is not fed the same input is not a reference.
fn input(f: &Fixture) -> (Tensor<3>, Tensor<3>) {
    let x = Tensor::<3>::random(
        [B, T, f.cfg.hidden_size],
        Distribution::Normal(0.0, 1.0),
        &f.cuda.clone().autodiff(),
    );
    let xs = host(&x);
    let sh = [B, T, f.cfg.hidden_size];
    (
        x,
        Tensor::<3>::from_data(TensorData::new(xs, sh), &f.cpu.clone().autodiff()),
    )
}

/// The CPU `NdArray` reference: the same module with the same weights, the
/// same input values, and its own autodiff graph on a different device and a
/// different chunk implementation. Returns the gradient per parameter name.
fn cpu_reference(f: &Fixture, xs: &Tensor<3>) -> (f64, Vec<(&'static str, Vec<f32>)>) {
    // The reference must be on a DIFFERENT DEVICE, asserted rather than
    // assumed: `Device::default()` is a function call, and if it ever named a
    // GPU this would be a second CUDA run with the same kernels — the
    // reference measuring itself, and every "agreement" below it vacuous.
    // `Backend::name` is forwarded down to the runtime's own string, so this
    // sees through any wrapper.
    // `backend_matches` is burn-gdn2's own sanctioned test
    // (`cuda_dispatch.rs`), and `NdArrayDevice` has exactly one variant, so
    // this is the assertion "the reference backend is not a GPU".
    let dev_name = <NdArray as burn::backend::Backend>::name(&Default::default());
    assert!(
        !burn_gdn2::backend_matches::<Autodiff<NdArray, BalancedCheckpointing>>(),
        "the CPU reference backend reports as {dev_name} — it is not an independent device, so \
         every comparison against it compares a thing to itself"
    );
    println!("CPU reference backend: {dev_name}");
    let mut m = KdaModule::new(&f.cfg, 0.9, &f.cpu.clone().autodiff());
    for (prm, _) in params(&f.m, None) {
        (prm.set)(
            &mut m,
            prm.val.clone(),
            &prm.shape,
            &f.cpu.clone().autodiff(),
        );
    }
    let (y, s) =
        m.forward_train_state::<Autodiff<NdArray, BalancedCheckpointing>>(xs.clone(), None);
    let sq = |v: Vec<f32>| v.iter().map(|t| (*t as f64) * (*t as f64)).sum::<f64>();
    let loss = sq(host(&y.clone())) + sq(host(&s.clone()));
    let grads = y
        .powf_scalar(2.0)
        .sum()
        .add(s.powf_scalar(2.0).sum())
        .backward();
    let out = params(&m, Some(&grads))
        .into_iter()
        .map(|(prm, g)| {
            assert!(
                g.is_some(),
                "the CPU reference produced no gradient for {}",
                prm.name
            );
            (prm.name, g.unwrap())
        })
        .collect();
    (loss, out)
}

/// `f64` loss on CUDA, the reference for every finite difference.
fn cuda_loss<S: burn_autodiff::checkpoint::strategy::CheckpointStrategy>(
    m: &KdaModule,
    x: &Tensor<3>,
) -> f64
where
    S: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<Autodiff<CudaBare, S>>,
{
    let (y, s) = m.forward_train_state::<Autodiff<CudaBare, S>>(x.clone(), None);
    let sq = |v: Vec<f32>| v.iter().map(|t| (*t as f64) * (*t as f64)).sum::<f64>();
    sq(host(&y)) + sq(host(&s))
}

/// The whole run for one checkpointing strategy on CUDA. `S` names the
/// strategy in the printed label, because the two are different programs and a
/// log line that does not say which one ran is the 8fa5d4c mistake again.
fn run<S>(f: &Fixture, x: &Tensor<3>, xs: &Tensor<3>, label: &str) -> Vec<Verdict>
where
    S: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
    burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<Autodiff<CudaBare, S>>,
{
    burn_gdn2::reset_fused_calls();
    let (y, s) =
        f.m.forward_train_state::<Autodiff<CudaBare, S>>(x.clone(), None);
    let grads = y
        .powf_scalar(2.0)
        .sum()
        .add(s.powf_scalar(2.0).sum())
        .backward();
    let counts = burn_gdn2::seam_counts();
    let loss = cuda_loss::<S>(&f.m, x);
    println!("\n=== {label} ===");
    println!("seam: {}", show(counts));
    println!("arm: {}", arm_name(counts));
    println!("loss (f64) = {loss:.6e}");
    assert!(
        counts.4 > 0 || counts.2 > 0,
        "neither arm ran a backward ({}) — the seam is not differentiable at all, so every \
         verdict below would be a statement about nothing",
        show(counts)
    );
    assert!(loss > 0.0, "the loss is not positive");

    let (cpu_loss, cpu) = cpu_reference(f, xs);
    // The two references must be looking at the SAME FUNCTION before their
    // gradients can be compared, and the cheapest statement of that is their
    // forward agreeing. This assertion is here because an earlier version of
    // this file let `run` draw its own CPU input, which advanced the RNG and
    // fed the reference a different sequence; every group then read `cpu_rel`
    // ~ 1.0, which is indistinguishable from a real disagreement. Comparing
    // two derivatives of two different functions is not a weak test, it is not
    // a test.
    assert!(
        (cpu_loss - loss).abs() <= LOSS_BAR * loss.abs().max(1.0),
        "the CPU reference and the CUDA run computed DIFFERENT functions: |L_cpu| = {cpu_loss:.6e} \
         vs |L_cuda| = {loss:.6e}, off by {:.2e} against a {LOSS_BAR:.0e} bar. The weights or the \
         input the two sides saw are not the same, so any gradient comparison below is about \
         nothing.",
        (cpu_loss - loss).abs() / loss.abs().max(1.0)
    );
    println!(
        "CPU reference loss = {cpu_loss:.12e} vs CUDA {loss:.12e}  (relative {:.3e})",
        (cpu_loss - loss).abs() / loss.abs().max(1.0)
    );
    let mut out = vec![];
    for (prm, g) in params(&f.m, Some(&grads)) {
        let c = cpu
            .iter()
            .find(|(n, _)| *n == prm.name)
            .map(|(_, v)| v.as_slice());
        out.push(check_group::<S>(
            &f.m,
            x,
            &prm,
            g,
            c,
            loss,
            &f.cuda.clone().autodiff(),
        ));
    }
    print_table(&out, label);
    out
}

fn print_table(v: &[Verdict], label: &str) {
    println!("--- verdicts ({label}) ---");
    for x in v {
        let fd = match &x.fd {
            Ok((h, r)) => format!("h={h:.1e} rel={r:.2e}"),
            Err(why) => format!("UNRESOLVED ({why})"),
        };
        println!(
            "  {:<20} non_zero={:<5} finite={:<5} amax={:.3e}  cpu_rel={:.2e} (maxdiff {:.3e} of \
             cpu amax {:.3e})  fd: {fd}",
            x.name, x.non_zero, x.finite, x.amax, x.cpu_rel, x.cpu_maxdiff, x.cpu_amax
        );
    }
}

/// The contract that holds on EVERY dispatch arm, including the fused one: a
/// gradient exists, and it is a finite non-zero tensor. This is the deliverable
/// — the 8fa5d4c class is a group with NO gradient, and no magnitude argument
/// is needed to see it. Group names in the message.
fn assert_flow(v: &[Verdict]) {
    let frozen: Vec<&str> = v
        .iter()
        .filter(|x| !x.non_zero || !x.finite)
        .map(|x| x.name)
        .collect();
    assert!(
        frozen.is_empty(),
        "the KDA attention arm has NO USABLE GRADIENT for {} of {} parameter groups: [{}]. The \
         arm is frozen — the 8fa5d4c signature (AGENTS.md 3.2). A group that gets no gradient \
         while its neighbours do is that defect, not a numerical accident.",
        frozen.len(),
        v.len(),
        frozen.join(", ")
    );
}

/// Gradient FLOW plus NUMERICS, for an arm whose gradients are the derivative
/// of the function that arm computed. Three independent requirements, so a
/// failure says which one broke:
///
///  * every group has a finite, non-zero gradient (`8fa5d4c`);
///  * every group matches the CPU reference on the WHOLE tensor (`CPU_BAR`);
///  * every group whose finite differences resolved also matches those, and
///    every group whose differences did NOT is held to a five-times TIGHTER
///    bar against the CPU reference — an unresolved group is carried entirely
///    by one instrument, and letting it pass at the same bar would be a silent
///    fallback (ADR-0011's third mark, which is the defect).
fn assert_matches(v: &[Verdict]) {
    assert_flow(v);

    let wrong: Vec<String> = v
        .iter()
        .filter(|x| x.cpu_rel >= CPU_BAR)
        .map(|x| format!("{} (rel {:.2e} >= {CPU_BAR:.0e})", x.name, x.cpu_rel))
        .collect();
    assert!(
        wrong.is_empty(),
        "every group HAS a gradient, but {} of them are not the derivative of the function the \
         arm computed (CPU NdArray reference, whole tensor): [{}]. Both arms run the same \
         chunk implementation, so this is a device/precision-level disagreement — which is a \
         defect, and not one the finite differences can see.",
        wrong.len(),
        wrong.join(", ")
    );

    let mut fd_bad = vec![];
    let mut fd_carried = vec![];
    for x in v {
        match &x.fd {
            Ok((h, r)) if *r >= FD_BAR => fd_bad.push(format!(
                "{} (rel {r:.2e} >= {FD_BAR:.0e} at h={h:.1e})",
                x.name
            )),
            Ok(_) => {}
            Err(_) if x.cpu_rel >= CPU_BAR_UNRESOLVED_FD => fd_carried.push(format!(
                "{} (cpu_rel {:.2e} >= {CPU_BAR_UNRESOLVED_FD:.0e})",
                x.name, x.cpu_rel
            )),
            Err(_) => {}
        }
    }
    assert!(
        fd_bad.is_empty(),
        "the gradient disagrees with central differences of the SAME forward for: [{}]. Bar \
         {FD_BAR:.0e} is what an f32 central difference resolves; it is not widened to \
         accommodate a disagreement, and the CPU reference agreeing does not rescue it, because \
         both references could share a wrong formula and only this one is a different METHOD.",
        fd_bad.join(", ")
    );
    assert!(
        fd_carried.is_empty(),
        "these groups' finite differences were UNRESOLVED (see the table) and the CPU reference \
         they fall back to is {CPU_BAR_UNRESOLVED_FD:.0e} away — five times tighter than the \
         resolved groups get, because an unresolved group is carried by one instrument alone: \
         [{}]",
        fd_carried.join(", ")
    );
}

// ── the tests ──────────────────────────────────────────────────────────

/// THE test. The trainer's own backend — `Device::cuda(0).autodiff()`, i.e.
/// `Autodiff<CudaBare, BalancedCheckpointing>` — on the arm
/// `forward_train_state` actually dispatches to, on the trainer's parameter
/// shape: every KDA group gets a finite non-zero gradient that agrees with two
/// independent references.
#[test]
fn every_group_gets_a_matching_gradient_on_the_trainers_backend() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let f = fixture(cfg_trainer());
    let (x, xs) = input(&f);
    let v = run::<BalancedCheckpointing>(&f, &x, &xs, "trainer's backend: AdBal, trainer shape");
    assert_matches(&v);
}

/// The same module on `NoCheckpointing`, where every intermediate is a real
/// graph node and the fused op IS reachable. Two strategies are two programs;
/// a gradient that is right on one and absent on the other is a defect only
/// two runs find, and the strategy a user gets depends on a flag they cannot
/// see. This is the arm `8fa5d4c` wrote, so this is the arm where its fix
/// needs proving.
///
/// WHAT IS ASSERTED HERE IS FLOW, NOT NUMERICS, and the difference is a
/// measured finding rather than a convenience — see
/// `docs/reviews/kda-gradflow-2026-09-30.md`:
///
/// On the fused arm the gradients do NOT match central differences of the fused
/// arm's OWN forward. Measured on this fixture: 4–6% on q/k/v_proj, **2.5e-1 on
/// `beta_proj.weight`** and **4.8e-1 on `decay.b_alpha`**, while the two
/// parameters applied after the op (`o_norm_w`, `o_proj`) agree to 1e-6. The
/// downstream pair agreeing rules out a noisy forward; the upstream-sensitive
/// gates disagreeing by a quarter is a wrong input gradient in the fused
/// adjoint kernel. That kernel has never been numerically compared to anything
/// (§3.2), and this is that comparison.
///
/// So: asserting the numerics would land a red suite for a defect in a file
/// this lane does not own, and NOT asserting them silently would be the
/// 8fa5d4c class — a gate that prints green over a broken arm. What is done
/// instead is the third option: the flow and the arm are HARD assertions, and
/// the disagreement is recomputed and printed on every run, so it cannot rot
/// into a claim that the fused arm was checked.
#[test]
fn every_group_gets_a_gradient_on_the_fused_arm() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let f = fixture(cfg_trainer());
    let (x, xs) = input(&f);
    let v = run::<NoCheckpointing>(&f, &x, &xs, "AdNo (fused op reachable), trainer shape");

    // The arm is ASSERTED, not assumed. Without this the test could pass on the
    // ops path while claiming the fused one — the mistake 8fa5d4c's own commit
    // message made.
    burn_gdn2::reset_fused_calls();
    let (y, s) = f.m.forward_train_state::<AdNo>(x.clone(), None);
    y.powf_scalar(2.0)
        .sum()
        .add(s.powf_scalar(2.0).sum())
        .backward();
    let c = burn_gdn2::seam_counts();
    assert!(
        c.1 > 0 && c.2 > 0,
        "no fused kernel launched ({}); this test did not exercise the fused arm it is named for",
        show(c)
    );
    assert!(
        c.5 == 1,
        "expected exactly one entry into ChunkWy::backward, got {} — the custom node's backward \
         did not run once, so 'the fused arm' is a claim about an arm that did not",
        show(c)
    );
    assert_eq!(
        c.4,
        0,
        "the ops path also ran ({}); the seam declined, so this was not the fused arm",
        show(c)
    );

    assert_flow(&v);
    let bad: Vec<String> = v
        .iter()
        .filter(|x| matches!(&x.fd, Ok((_, r)) if *r >= FD_BAR) || x.cpu_rel >= CPU_BAR)
        .map(|x| {
            let fd = match &x.fd {
                Ok((h, r)) => format!("fd rel {r:.2e} at h={h:.1e}"),
                Err(_) => "fd UNRESOLVED".to_string(),
            };
            format!("{} (cpu_rel {:.2e}, {fd})", x.name, x.cpu_rel)
        })
        .collect();
    println!(
        "\n*** THE FUSED ADJOINT'S NUMERICS ARE NOT ASSERTED, and here is the \
         disagreement, recomputed: {}\n    {} of {} groups are outside {CPU_BAR:.0e}. FLOW is \
         asserted above; the numbers are here so this arm can never be reported as \
         numerically verified. docs/reviews/kda-gradflow-2026-09-30.md",
        if bad.is_empty() {
            "NONE — the fused adjoint now agrees".into()
        } else {
            bad.join("\n    ")
        },
        bad.len(),
        v.len()
    );
}

/// The short conv. Off in the trainer (`attention.rs:65-70`, it NaN'd at ~step
/// 60), so on the trainer's shape those three groups do not exist — which
/// `every_group_is_present_for_the_config` asserts rather than skipping past.
/// This is the config where they do, and the trainer has never run it, so a
/// green here is the FIRST evidence any of them ever receives a gradient.
#[test]
fn short_conv_weights_get_gradients_when_the_conv_is_on() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let f = fixture(cfg_short_conv());
    let (x, xs) = input(&f);
    let v = run::<BalancedCheckpointing>(&f, &x, &xs, "trainer's backend, short conv ON");
    for want in ["q_conv_w", "k_conv_w", "v_conv_w"] {
        assert!(
            v.iter().any(|x| x.name == want),
            "{want} is missing from the verdict table — the module did not build the short conv \
             this test is about, so the test proved nothing about it"
        );
    }
    assert_matches(&v);
}

/// The enumeration IS the coverage, and a coverage hole is a silent pass: a
/// group nobody listed would produce a green run with that group unchecked.
/// So the expected names are spelled out and compared, and the trainer's own
/// shape is confirmed NOT to build the conv — which is what makes "and the
/// short conv if present" mean something rather than reading as coverage.
#[test]
fn every_group_is_present_for_the_config() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let f = fixture(cfg_trainer());
    let (x, _) = input(&f);
    let (y, s) = f.m.forward_train_state::<AdBal>(x, None);
    let grads = y
        .powf_scalar(2.0)
        .sum()
        .add(s.powf_scalar(2.0).sum())
        .backward();
    let names: Vec<&str> = params(&f.m, Some(&grads))
        .iter()
        .map(|(p, _)| p.name)
        .collect();
    assert_eq!(
        names,
        [
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "decay.w_up.weight",
            "decay.w_down.weight",
            "decay.b_alpha",
            "decay.a_log",
            "beta_proj.weight",
            "o_norm_w",
            "o_proj.weight",
            "o_gate.weight",
        ],
        "the enumerated group list no longer matches KdaModule's parameters for this config. A \
         group missing from this list is a group nothing checks."
    );
    assert!(
        f.m.q_conv_w.is_none() && f.m.k_conv_w.is_none() && f.m.v_conv_w.is_none(),
        "the trainer's use_short_conv=false did not leave the conv unset, so the other two tests \
         silently check fewer tensors than their names claim"
    );
    // And the conv-on config really does add exactly the three.
    let f = fixture(cfg_short_conv());
    let (x, _) = input(&f);
    let (y, s) = f.m.forward_train_state::<AdBal>(x, None);
    let grads = y
        .powf_scalar(2.0)
        .sum()
        .add(s.powf_scalar(2.0).sum())
        .backward();
    let names: Vec<&str> = params(&f.m, Some(&grads))
        .iter()
        .map(|(p, _)| p.name)
        .collect();
    for want in ["q_conv_w", "k_conv_w", "v_conv_w"] {
        assert!(
            names.contains(&want),
            "{want} missing with use_short_conv=true"
        );
    }
    assert_eq!(names.len(), 14, "the conv-on group list is {names:?}");
}
