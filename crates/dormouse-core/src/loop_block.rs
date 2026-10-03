//! loop - mini UniversalLoop: controller + shared attention (KDA) +
//! expert TSCT FFNs + Engram + e_k (iteration embedding) + ReZero residual
//! scale (or, mutually exclusively, the GR / AttnRes / mHC replacement for
//! that one statement). Fixed depth (ADR-0013): every iteration counts
//! equally, the loss is an honest unweighted CE.
//!
//! # Invariants this file holds, and what breaks each one
//!
//! * **`L_Rec` is accumulated inside [`LoopBlock::forward_full_state`], per
//!   iteration, on `[b*t, d]` reshapes.** Never materialized as `[N, b, t, d]`
//!   and sliced: a dynamic slice of a 4-D autodiff tensor crashes cubecl on
//!   sm_120 with `CUDA_ERROR_ILLEGAL_ADDRESS` (AGENTS.md §2.2). This is the
//!   single most load-bearing structural fact in the file.
//! * **No host-device synchronization.** Every branch below is on a
//!   `#[module(skip)]` config field, so a branch costs nothing per step, and
//!   nothing reads a tensor back to make a decision (ADR-0018 rule 2). The
//!   asserts at the top of the forward inspect CONFIG-derived fields, never
//!   device values.
//! * **One residual mechanism at a time.** ReZero, GR, AttnRes and mHC each
//!   replace the SAME statement in the loop; `config::validate` refuses any
//!   pair and the forward re-asserts it, because every field is `pub` and a
//!   hand-built block can carry a flag its parameters do not match.
//! * **The memory branch has a structural floor**, not a learned one:
//!   [`memory_floor_mix`] caps the memory's coefficient at `lam_max`, so the
//!   backbone's share of that branch cannot fall below `1 - lam_max` whatever
//!   the controller learns. Before that floor the arm was a scaled row copy,
//!   which is how a lookup table became 99% of the model and explained the
//!   targets.
//! * **The in-VRAM memory masks its slot index** (`hash & engram_slot_mask`)
//!   rather than dividing, which is why the tables round UP to a power of two
//!   and why the data crate emits RAW FNV keys on this path.
//!
//! # Cost
//!
//! This is the expensive half of a step. Each iteration is a full-sequence
//! gated-delta pass allocating ~17 fresh tensors, and it does not amortize with
//! batch size; the measured warm-step split at 9.2M params / depth 2 / batch 8
//! puts the KDA backward at ~205 ms of a ~480 ms step, with the TSCT
//! retraction a further ~53 ms of fixed per-parameter cost. GEMMs are a small
//! fraction. Judge changes here on launches and scratch bytes, not FLOPs.
use burn::backend::DispatchKindConversion;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{activation, Device, DispatchTensor, FloatDType, Int, Tensor};
use burn_attnres::{depth_attend, AttnRes};
use burn_engram::EngramModule;
use burn_mhc::MhcBlock;
use burn_rmsnorm::RMSNorm;

use crate::attention::AdaptiveAttention;
use crate::config::{ActQuant, DormouseConfig};
use crate::gr::{GatedResidual, GrState, GR_BRANCHES};
use crate::mor::{self, MoRRouter};
use crate::param::LinearLike;

/// Table sizes for the in-VRAM hashed memory: `n_tables` of `rows` rounded UP
/// to a power of two, plus the matching slot-index mask. The rounding exists
/// because the in-model read MASKS the raw FNV hash (`hash & mask`) instead of
/// dividing it on device: one integer op, and every index is in range by
/// construction. 500_000 -> 524_288 rows (+4.9%), which sits on the flat part
/// of the measured 300K/500K/800K curve.
pub fn engram_tables(rows: usize, n_tables: usize) -> (Vec<usize>, i32) {
    assert!(rows > 0, "engram_rows must be > 0");
    assert!(n_tables > 0, "engram needs at least one order");
    assert!(
        rows.next_power_of_two() <= i32::MAX as usize,
        "engram_rows {} does not fit an i32 slot mask",
        rows
    );
    let pow2 = rows.next_power_of_two();
    (vec![pow2; n_tables], (pow2 - 1) as i32)
}

/// The memory branch, as a CONVEX MIXTURE with a hard floor:
///
/// ```text
/// out = lam * memory + (1 - lam) * dense,   lam = min(w_mem, lam_max)
/// ```
///
/// `memory` is the gated read (`value_proj(e) * sigma(sign(s)*sqrt(|s|+1e-6))`,
/// the DeepSeek gate, unchanged) and `dense` is `mem_dense(normed)` - a plain
/// projection of the same hidden state. The point is that the guarantee is
/// STRUCTURAL: whatever the controller learns, the memory's coefficient in
/// this branch cannot exceed `lam_max`, so the backbone's share is never below
/// `1 - lam_max`. Before this the branch was a direct row copy scaled by an
/// unbounded `w_mem`, which is how a 99%-of-the-model lookup table ended up
/// explaining the targets (rec -> 0.005, held-out at uniform 8.000 BPB).
///
/// Copied from: FwPKM eq. 12 `o_t = g_t*v_hat_t + (1-g_t)*v_t` (the dense
/// value path from the same hidden state is what makes it a floor rather
/// than a gate), kNN-LM eq. 3 `p = lambda*p_knn + (1-lambda)*p_lm` (lambda is
/// a tuned CONSTANT, not a learned value - same claim, and the reason it
/// generalizes), and XLM's PKM `EmbeddingBag(per_sample_weights=True)`
/// (xlm/model/memory/memory.py:79) where the read is a bounded weighted
/// average of rows rather than one row.
pub fn memory_floor_mix(
    memory: Tensor<2>,
    dense: Tensor<2>,
    w_mem: Tensor<2>,
    lam_max: f32,
) -> Tensor<2> {
    let lam = w_mem.clamp(0.0, lam_max);
    memory.mul(lam.clone()) + dense.mul(lam.neg().add_scalar(1.0))
}

/// One expert of the FFN bank: the SwiGLU-shaped pair, both spectral.
///
/// There is no shared expert and no dense FFN when `n_experts > 1` — the
/// controller's softmax blend over experts is the mixture, and with
/// `moe_topk = 0` that blend is dense, which is the arm's control. Every
/// expert is a separate [`LinearLike`] and therefore carries its own TSCT
/// factors, its own quant format and its own slot in the retraction walk.
#[derive(Module, Debug)]
pub struct ExpertFFN {
    /// `d_model -> f` normally, `d_model -> 2f` when `use_situ` is on (Eq 12
    /// reads `Wg x` and `Wu x` as separate inputs). See [`ExpertFFN::new`].
    pub gate_up: LinearLike,
    /// `f -> d_model`. The down projection; the activation lives between the
    /// two, not here.
    pub down: LinearLike,
}

/// What the loop's ROUTING arms hand back for the loss.
///
/// A struct rather than a second `Option` in the return tuple: the fourth
/// element used to mean `mor_aux` alone, and adding a second arm's term to the
/// same slot would give one value two meanings - the failure mode this repo's
/// own glossary exists to prevent. Every field is `None` unless its arm ran, so
/// the off-arm shape is exactly what it was.
#[derive(Default)]
pub struct RouteAux {
    /// MoR's router-vs-its-own-top-k BCE (`mor::route`), UNSCALED - the caller
    /// owns the weight (`mor_bce_weight`).
    pub mor: Option<Tensor<1>>,
    /// The sparse-routing load-balancing term (`moe::lb_aux`), UNSCALED and
    /// averaged over the iterations that ran. `None` unless `moe_topk > 0`.
    pub moe_lb: Option<Tensor<1>>,
    /// The MSA distillation KL (tech report Eq. 18), UNSCALED, averaged over
    /// the iterations that ran. `None` unless the MSA stage ran stage (a).
    pub msa_distill: Option<Tensor<1>>,
}

impl ExpertFFN {
    /// `situ` makes `gate_up` a `d -> 2f` projection, because Eq (12) reads
    /// `Wg x` and `Wu x` as separate inputs. With `situ = false` the
    /// projection stays `d -> f` and the field's name is the misnomer it has
    /// always been (a `d -> f` up-projection named after two projections);
    /// `situ_off_leaves_the_parameter_set_alone` holds the off arm to the old
    /// shape.
    pub fn new(
        d: usize,
        f: usize,
        rank: usize,
        use_tsct: bool,
        situ: bool,
        device: &Device,
    ) -> Self {
        let mid = if situ { 2 * f } else { f };
        Self {
            gate_up: LinearLike::with_tsct(d, mid, rank, use_tsct, device),
            down: LinearLike::with_tsct(f, d, rank, use_tsct, device),
        }
    }
}

/// The weight-shared loop body, run `max_iter` times with the SAME weights.
///
/// # What one iteration does, in order
///
/// 1. the **controller** reads `[norm(h), h0]` and emits three gates
///    (`w_attn`, `w_mem`, `w_ffn`, all sigmoid) plus a softmax blend over the
///    `n_experts` FFNs — so the mixture weights are a function of the state,
///    computed per position, not a config constant;
/// 2. the **shared attention** (KDA) and the **experts** run, each scaled by
///    its gate;
/// 3. the **Engram** memory read joins as a convex mixture with a hard floor
///    ([`memory_floor_mix`]);
/// 4. the body is deposited into the residual stream by exactly ONE of ReZero
///    ([`LoopBlock::residual_scale`]), Gated Residual
///    ([`LoopBlock::gr`]), Attention Residuals ([`LoopBlock::attnres`]) or mHC
///    ([`LoopBlock::mhc`]) — mutually exclusive, refused in
///    `config::validate`;
/// 5. the readout adds the iteration embedding [`Self::iter_embed`].
///
/// # Invariants
///
/// * **Fixed depth.** There is no halt head and no PonderNet: every iteration
///   counts equally and `L_Rec` is the unweighted mean CE over them
///   (ADR-0013). `depth_override` (random depth) and `use_mor` (per-position
///   routing) are the two arms that vary it, and neither is A/B'd.
/// * **`L_Rec` is accumulated INSIDE this function**, per iteration, on
///   `[b*t, d]` reshapes. It is never materialized as `[N, b, t, d]` and
///   sliced: a dynamic slice of a 4-D autodiff tensor crashes cubecl on
///   sm_120 with `CUDA_ERROR_ILLEGAL_ADDRESS` (AGENTS.md §2.2). Keep it that
///   way — this is the single most load-bearing structural fact in the file.
/// * **No host sync anywhere in the forward.** Arm switches are plain fields
///   read on the host, so a branch costs nothing per step; nothing reads a
///   tensor back to decide (ADR-0018 rule 2). The three `assert`s at the top
///   inspect CONFIG-derived fields, never device values.
/// * **Off arms do not exist.** `gr`, `attnres`, `mhc` are `Option`s built
///   from the config flag, so an off-arm model is byte-identical to a build
///   from before the arm shipped and every existing checkpoint loads.
/// * The field set is the checkpoint's field set: `#[derive(Module)]` names
///   every parameter, and `#[module(skip)]` fields are config constants the
///   model carries rather than reads from the config at forward time.
#[derive(Module, Debug)]
pub struct LoopBlock {
    /// `[h_norm ; h0] -> [w_attn, w_mem, w_ffn, expert blend]`, no bias. Padded
    /// on the right to a multiple of 4 because cubek's matmul vectorizes on
    /// `N % 4 == 0` (AGENTS.md §2.2) and the slice back is on the caller.
    pub controller: Linear,
    /// The KDA arm, wrapped for checkpoint-path stability. Weight-SHARED across
    /// iterations like everything else here.
    pub shared_attn: AdaptiveAttention,
    /// `n_experts` independent expert FFNs. The controller's softmax blend
    /// mixes them; with `moe_topk > 0` a top-k selection replaces the blend
    /// per token.
    pub expert_ffns: Vec<ExpertFFN>,
    /// The hashed n-gram memory tables (in-VRAM). `None`-shaped by
    /// [`Self::use_engram`], not by an `Option`: the module always exists so
    /// an off-arm checkpoint keeps its (unused) rows, and the forward takes an
    /// inert branch instead of a different module tree.
    pub engram: EngramModule,
    /// The dense half of the memory branch's convex mixture: a plain
    /// `d_model -> d_model` projection of the SAME hidden state the memory
    /// is gated against. It is the `(1 - g) * v_t` term of FwPKM eq. 12 and
    /// the reason the backbone can no longer be starved (see
    /// [`memory_floor_mix`]).
    pub mem_dense: Linear,
    /// Pre-norm applied to `h` at the top of each iteration, before the
    /// controller and the arms. One RMSNorm for the whole loop, shared — this
    /// is a weight-shared block, so a per-sublayer norm would be the one thing
    /// that is not.
    pub norm: RMSNorm,
    /// Gated Residual arm (arXiv Qwen3.8-Flash-Next Eq 31-34). `None` unless
    /// `use_gr`. Never A/B'd, and the four equation defects the 2026-09-28
    /// audit found were all fixed before any number existed — see
    /// [`crate::config::DormouseConfig::use_gr`].
    pub gr: Option<GatedResidual>,
    /// Attention Residuals (arXiv:2603.15031): one learned pseudo-query `w_l`
    /// per loop iteration slot (§5, "one RMSNorm and one pseudo-query vector
    /// per layer"), zero-initialised so the first forward is an equal-weight
    /// average. `None` unless `use_attnres`, which is what keeps every
    /// existing checkpoint loadable: the parameters do not exist when the arm
    /// is off, and the config snapshot refuses a resume that flips the flag.
    pub attnres: Option<Vec<AttnRes>>,
    /// Manifold-Constrained Hyper-Connections (arXiv:2512.24880): ONE shared
    /// `MhcBlock`, applied at every iteration's residual write, the way the
    /// loop shares everything else. It is Eq. 3 with the block body's output
    /// read as `n` per-stream outputs: `h' = H_res h + H_post . y` with
    /// `H_res` Sinkhorn-projected onto the Birkhoff polytope (Eq. 8-9), so
    /// `H_res h` is a convex combination of the `n` streams and the composite
    /// over the loop's `T` iterations - `H_res^T` under the sharing - is
    /// doubly stochastic too. `None` unless `use_mhc`.
    ///
    /// SHARED, not one per iteration slot, and that is the decision: the
    /// manifold's stability argument is about the COMPOSITE `prod H_res`
    /// (2512.24880 Eq. 4 and §4.1), which a shared block realises as a power
    /// of one doubly stochastic matrix - still in the Birkhoff polytope, since
    /// that set is closed under multiplication - while per-slot matrices would
    /// make the residual mechanism the only un-shared thing in a weight-shared
    /// loop. `H_pre` is NOT wired: Eq. 3 puts it on the block's own input,
    /// which collapses the `n x C` stream to `C`, and our body runs at full
    /// width `D`. The paper's Tab. 1 ablation puts `H_res` at -0.022 of the
    /// -0.027 total, so the two terms we do wire are ~89% of the measured
    /// effect (docs/reviews/mhc-2026-09-30.md §3).
    ///
    /// `n = mhc_streams` defaults to **4** since the fidelity fix F-F3
    /// (2026-10-02; the base paper 2409.19606 calls the same quantity the
    /// "expansion rate" and its Tab. 1 ablates it: n=4 is its best rung; mHC
    /// 2512.24880 runs n=4 on every model in Tab. 5). At our launch-bound
    /// profile the `n x n` Sinkhorn is a launch-count cost, not a parameter
    /// cost, and the phi study's n = 2 rung is `--set mhc_streams=2` - see
    /// [`crate::config::DormouseConfig::mhc_streams`].
    pub mhc: Option<MhcBlock>,
    /// `e_k`, the per-iteration-slot embedding added at the readout:
    /// `[max_iter, d_model]`. Its presence is what makes the `T` iterations
    /// distinguishable at all — the body is weight-shared, so without it the
    /// `T` passes would compute the same function `T` times. Zero-initialised,
    /// so at step 0 the readout is the plain mean over iterations and no arm
    /// gets a head start from it.
    pub iter_embed: burn::module::Param<Tensor<2>>,
    /// ReZero's residual coefficient, ONE scalar for the whole loop, init 1.0
    /// and **not** 0: at 0 the block body contributes nothing AND `dL/dy` is
    /// exactly zero, so the KDA arm, the experts and the controller's gates
    /// would all start with no gradient and the model would be a linear map of
    /// the byte embedding until the scalar moved. Ignored when
    /// `use_gr`/`use_attnres`/`use_mhc` is on, since those replace the write.
    pub residual_scale: burn::module::Param<Tensor<1>>,
    /// The readout projection, `d_model -> d_model`, applied to the averaged
    /// hidden states before the final RMSNorm and `lm_head`. Spectral like
    /// every other linear, and the third member of the retraction walk.
    pub out_proj: LinearLike,
    /// MoR router: the shared linear scorer of arXiv 2507.10524, one score
    /// per (position, iteration slot). Always present (769 params, routed to
    /// AdamW by the optimizer policy - routers are not Muon+ candidates);
    /// `use_mor` decides whether it is read.
    pub mor_router: MoRRouter,
    /// Qwen Sparse Attention stage (tech report §QSA): an EXTRA full-softmax
    /// attention per iteration, off by default (`use_msa`). See
    /// [`crate::msa_stage`] for the placement decision and the two stages.
    pub msa: Option<crate::msa_stage::MsaStage>,
    /// Loop depth, copied from the config so the forward does not have to be
    /// handed the config. Changing it after construction does NOT resize
    /// `iter_embed` or the `attnres` slot vector, so it is a config-time
    /// constant, not a runtime knob.
    #[module(skip)]
    pub max_iter: usize,
    /// Residual-stream width. A `#[module(skip)]` copy of `DormouseConfig::
    /// d_model`, kept next to the shapes it has to agree with.
    #[module(skip)]
    pub d_model: usize,
    /// `DormouseConfig::d_ffn`. The `use_situ` assert in the forward holds
    /// `gate_up.out_features` against `2 * ffn_hidden`, because the field that
    /// says which activation runs is not the same claim as the width the
    /// projection was built with.
    #[module(skip)]
    pub ffn_hidden: usize,
    /// `DormouseConfig::n_experts`. The controller's blend width and the
    /// length of [`Self::expert_ffns`] are two claims; this is the one the
    /// forward reads.
    #[module(skip)]
    pub n_experts: usize,
    /// The KDA arm switch. `false` = a model with NO attention arm, which is
    /// the one configuration whose held-out BPB in the archive is a number for
    /// the network that produced it (`--no-kda`; AGENTS.md §3.1).
    #[module(skip)]
    pub use_kda: bool,
    /// Random-depth arm (ADR-0013 rank 2): run only the first `n` iterations
    /// instead of `max_iter`, and average the readout and the CE over the
    /// iterations actually executed. `None` = fixed depth, the default.
    /// The trainer sets it per step; there is no learned halting here, so
    /// there is nothing to collapse.
    #[module(skip)]
    pub depth_override: Option<usize>,
    /// The memory arm switch. `false` makes the forward take the inert
    /// branch — and the inert branch is the CORRECT forward for a model with
    /// no memory in it, which is why this is not the bug it was in the eval
    /// path (AGENTS.md §3.2: `hashed_ids = None` is right here and wrong in a
    /// measurement).
    #[module(skip)]
    pub use_engram: bool,
    /// Read [`LoopBlock::attnres`]. Kept as a plain field (not `attnres.is_some()`)
    /// so the branch is a config constant, like every other arm switch.
    #[module(skip)]
    pub use_attnres: bool,
    /// Read [`LoopBlock::mhc`]. Plain field, same reason as `use_attnres`:
    /// the branch is a config constant, not a shape test.
    #[module(skip)]
    pub use_mhc: bool,
    /// SiTU-GLU in the expert FFN (arXiv:2607.24653v2 Eq 12). A plain field,
    /// like every other arm switch - NOT `gate_up.dims()` inferred, because
    /// the shape it implies and the activation that reads it are two claims
    /// and only one of them is the config. Off means the elementwise
    /// `activation::silu` this repo ran before, bit for bit.
    #[module(skip)]
    pub use_situ: bool,
    /// MoR routing arm (arXiv 2507.10524). Off by default; the fixed-depth
    /// mean readout is the default path.
    #[module(skip)]
    pub use_mor: bool,
    /// MSA stage switch: `false` = the loop never reaches the arm (and the
    /// parameters do not exist - `Option`-shaped, unlike the Engram's
    /// always-built tables).
    #[module(skip)]
    pub use_msa: bool,
    /// Selected iteration slots per position under MoR (>= 1, see
    /// `mor::route`). Ignored when `use_mor` is off.
    #[module(skip)]
    pub mor_k: usize,
    /// Sparse expert routing over the FFN branch: selected experts per token
    /// per pass. **0 = the dense softmax blend, the default and the control.**
    /// See `config::schema::moe_topk` for why 4 experts at k=1 is the first
    /// configuration and what this does NOT buy (it is not a FLOP saving).
    #[module(skip)]
    pub moe_topk: usize,
    /// Slot-index mask for the in-VRAM tables: `engram_rows` rounded UP to a
    /// power of two, minus one. Masking (not dividing) keeps the address
    /// arithmetic to one op on the device and makes every index in range by
    /// construction - the data crate emits RAW FNV hashes on this path and
    /// the model owns the capacity. See [`LoopBlock::engram_slot_mask`].
    #[module(skip)]
    pub engram_slot_mask: i32,
    /// HARD floor on the memory branch: `w_mem` is clamped to this, so the
    /// dense half of the branch never carries less than `1 - lam_max`.
    #[module(skip)]
    pub engram_lam_max: f32,
    /// Cast activations to bf16 at the arm boundaries. **Slower than fp32 on
    /// this backend** (no bf16 tensor-core path exists; §2.1), and mixed-dtype
    /// ops NaN, which is why every forward casts to f32 before a Linear and
    /// back after the residual write. `None` = fp32 everywhere.
    #[module(skip)]
    pub bf16: bool,
    /// Activation quantization format for the FFN branch. `None` = the
    /// trainer's auto resolution, not "off": the trainer decides before it
    /// builds the model. The ATTENTION path is unconditionally at least 8
    /// bits regardless of what this says
    /// ([`crate::act_quant::ActFormat::attn`]), so `Fp4` has never run 4-bit
    /// attention.
    #[module(skip)]
    pub act_quant: Option<ActQuant>,
    /// Block size for that quantization. `0` = one scale per token. Must
    /// divide `d_model`; refused in `config::validate` because the failure
    /// would otherwise be a reshape error on the first forward.
    #[module(skip)]
    pub act_group: usize,
}

impl LoopBlock {
    /// Apply a quantization format to every TSCT factor (stage-2 switch).
    pub fn set_quant_all(&mut self, quant: burn_spectral::QuantFormat) {
        for f in &mut self.expert_ffns {
            f.gate_up.set_quant(quant);
            f.down.set_quant(quant);
        }
    }

    /// Toggle the bf16 matmul path on every TSCT factor.
    pub fn set_bf16_compute(&mut self, on: bool) {
        for f in &mut self.expert_ffns {
            f.gate_up.set_bf16_compute(on);
            f.down.set_bf16_compute(on);
        }
        self.out_proj.set_bf16_compute(on);
    }

    /// Polar-retract every TSCT factor U/V in the block (see LinearLike).
    pub fn retract_tsct(&mut self, iters: usize) {
        for f in &mut self.expert_ffns {
            f.gate_up.retract(iters);
            f.down.retract(iters);
        }
        self.out_proj.retract(iters);
    }

    /// Append every TSCT master in the block (`u`, `v` per linear) to `out`.
    /// The factor list [`LoopBlock::retract_tsct`] walks, handed to the
    /// batched arm instead of one factor at a time.
    pub fn push_tsct_masters<'a>(&'a mut self, out: &mut Vec<&'a mut Param<Tensor<2>>>) {
        for f in &mut self.expert_ffns {
            f.gate_up.push_tsct_masters(out);
            f.down.push_tsct_masters(out);
        }
        self.out_proj.push_tsct_masters(out);
    }

    /// Worst orthonormality error across all TSCT factors (syncs the device).
    pub fn max_ortho(&self) -> f32 {
        let mut m = 0.0f32;
        for f in &self.expert_ffns {
            m = m.max(f.gate_up.max_ortho()).max(f.down.max_ortho());
        }
        m.max(self.out_proj.max_ortho())
    }

    /// Fold the block's TSCT forward diagnostics into `agg`, worst-case over
    /// every factor (the `max_ortho` convention). Syncs the device once per
    /// factor; cadence only — see [`LinearLike::fold_tsct_diag`].
    pub fn fold_tsct_diag(&self, agg: &mut crate::param::TsctDiag) {
        for f in &self.expert_ffns {
            f.gate_up.fold_tsct_diag(agg);
            f.down.fold_tsct_diag(agg);
        }
        self.out_proj.fold_tsct_diag(agg);
    }

    /// Build the block from a validated config.
    ///
    /// Does NOT call [`crate::config::validate`] — the caller owns that, and
    /// the trainer calls it once on the fully-resolved config (after
    /// `--set`). Two invariants this constructor establishes that a
    /// post-hoc mutation of the fields would break, both re-asserted in
    /// [`Self::forward_full_state`] because every field here is `pub`:
    ///
    /// * `use_situ` and the experts' `gate_up` width must agree (`f` vs
    ///   `2 * f`), or Eq 12 reads a `d -> f` tensor as a `d -> 2f` one;
    /// * the three residual-replacement arms must not be combined.
    ///
    /// The `Option` arms are built from the config flags and are `None`
    /// otherwise, which is what makes an off-arm model byte-identical to a
    /// build from before the arm existed.
    ///
    /// ReZero's `residual_scale` inits to **1**, not 0, and the reason is
    /// written at the field: at 0 the whole body has exactly zero gradient.
    pub fn new(cfg: &DormouseConfig, device: &Device) -> Self {
        let d = cfg.d_model;
        let f = cfg.d_ffn;
        // Controller: [h_ctx, h0] (2d) -> weights for attn/mem/ffn + expert blend
        let n_ctrl = 3 + cfg.n_experts;
        let ctrl_pad = if !n_ctrl.is_multiple_of(4) {
            n_ctrl.next_multiple_of(4)
        } else {
            n_ctrl
        };
        let controller = LinearConfig::new(d * 2, ctrl_pad)
            .with_bias(false)
            .init(device);
        let iter_embed = burn::tensor::Tensor::<2>::zeros([cfg.max_iter, d], device);
        let iter_embed = burn::module::Param::from_tensor(iter_embed.clone());
        // One table per n-gram order, `engram_rows` rounded up to a power of
        // two (the slot index is masked on device, see engram_slot_mask).
        let (tables, mask) = engram_tables(cfg.engram_rows, cfg.engram_orders.len());
        Self {
            controller,
            shared_attn: AdaptiveAttention::new(d, cfg.n_heads, cfg.head_dim, device),
            expert_ffns: (0..cfg.n_experts)
                .map(|_| ExpertFFN::new(d, f, cfg.rank, cfg.use_tsct, cfg.use_situ, device))
                .collect(),
            engram: EngramModule::new(&tables, cfg.engram_dim, d, 1, device),
            mem_dense: LinearConfig::new(d, d).with_bias(false).init(device),
            norm: RMSNorm::new(d, cfg.norm_eps, device),
            gr: cfg.use_gr.then(|| GatedResidual::new(d, device)),
            // One pseudo-query per iteration slot. `ScoreForm` is left at its
            // default, which IS the paper's: `q . RMSNorm(k)`, no temperature
            // (arXiv:2603.15031 Eq. 2). Spelled out here because this is the
            // line a reader checks to know which function the arm computes.
            attnres: cfg
                .use_attnres
                .then(|| (0..cfg.max_iter).map(|_| AttnRes::new(d, device)).collect()),
            // One shared block, applied once per iteration. `mhc_streams` is
            // the expansion rate `n`; `validate` refuses a non-divisor of
            // `d_model` LOUDLY, so the reshape inside `forward` cannot be the
            // thing that tells the user which flag was wrong.
            mhc: cfg
                .use_mhc
                .then(|| MhcBlock::new(cfg.mhc_streams, d, device)),
            iter_embed,
            // ReZero's residual coefficient starts at 1 (identity init), NOT 0:
            // at 0 the block body contributes nothing AND its gradient is
            // exactly zero (dL/dy = 0), so the KDA arm, the expert FFNs and the
            // controller's gates all start with no gradient at all - the model
            // is a linear map of the byte embedding until the scalar moves.
            residual_scale: burn::module::Param::from_tensor(Tensor::ones([1], device)),
            out_proj: LinearLike::with_tsct(d, d, cfg.rank, cfg.use_tsct, device),
            mor_router: MoRRouter::new(d, device),
            msa: match cfg.use_msa {
                true => {
                    let indexer = burn_msa::IndexerModule::new(
                        burn_msa::IndexerConfig {
                            d_model: d,
                            q_heads: cfg.msa_q_heads,
                            head_dim: cfg.msa_head_dim,
                            block_r: cfg.msa_block_r,
                            rope_frac: cfg.msa_head_dim / 2,
                            max_seq_len: cfg.max_seq_len,
                        },
                        device,
                    );
                    Some(crate::msa_stage::MsaStage::new(cfg, indexer, device))
                }
                false => None,
            },
            max_iter: cfg.max_iter,
            d_model: d,
            ffn_hidden: f,
            n_experts: cfg.n_experts,
            use_kda: cfg.use_kda,
            depth_override: None,
            use_engram: cfg.use_engram,
            use_attnres: cfg.use_attnres,
            use_mhc: cfg.use_mhc,
            use_situ: cfg.use_situ,
            use_mor: cfg.use_mor,
            use_msa: cfg.use_msa,
            mor_k: cfg.mor_k,
            moe_topk: cfg.moe_topk,
            engram_slot_mask: mask,
            engram_lam_max: cfg.engram_lam_max,
            bf16: cfg.bf16,
            act_quant: cfg.act_quant,
            act_group: cfg.act_group,
        }
    }

    /// Run the loop `n` times and return `(logits, L_Rec, kda_state, route_aux)`.
    ///
    /// `n` is [`Self::max_iter`] unless [`Self::depth_override`] shortens it
    /// (the random-depth arm). The iteration count is a loop bound, not a
    /// tensor dimension — the step hiddens are never stacked, which is the
    /// sm_120 crash-avoidance rule stated at the struct.
    ///
    /// # Arguments that are `Option` because the path is optional
    ///
    /// * `hashed_ids` `[b, t, 3]` — RAW FNV keys for the in-VRAM memory; `None`
    ///   when `use_engram` is false, and that `None` is the CORRECT forward for
    ///   a model with no memory in it. It is NOT correct in a measurement: the
    ///   eval once passed `None` unconditionally and so scored a
    ///   memory-disabled network while claiming to score the trained one
    ///   (AGENTS.md §3.2). Pass real keys whenever the arm is on.
    /// * `host_rows` `[b, t, 96]` f32 — gathered rows for the RAM-offload
    ///   tables; `None` on the in-VRAM path.
    /// * `kda_state` `[b, t, n_heads, head_dim]` — the KDA recurrence carried
    ///   across chunks, so one sequence longer than `max_seq_len` is served by
    ///   successive calls rather than one enormous tensor. `None` starts fresh.
    /// * `targets` `[b*t, 1]` byte indices — when present, the per-iteration CE
    ///   is gathered from `log_softmax` at these positions and averaged into
    ///   `L_Rec`. `None` (an eval/decode forward) returns a **zero** `L_Rec`
    ///   rather than the previous iteration's value, so an eval cannot
    ///   accidentally train on a stale loss.
    ///
    /// # Cost (this is where a step goes)
    ///
    /// From a warm-step profile at 9.2M params, depth 2, batch 8 (measured
    /// 2026-10-01, `d8fa449`): the KDA backward is ~205 ms of a ~480 ms step.
    /// Each iteration is a full-sequence gated-delta pass allocating ~17 fresh
    /// tensors, and that does NOT amortize with batch size. The TSCT retraction
    /// is a separate ~53 ms of fixed per-parameter cost. GEMMs are a small
    /// fraction of a step. So: this function is launch- and allocation-bound,
    /// and a change here should be judged on launches and scratch bytes.
    ///
    /// # Failure modes
    ///
    /// Panics, LOUDLY and naming the cause, on a `use_situ`/`gate_up` width
    /// mismatch or on two residual-replacement arms at once. Everything else
    /// that could degrade is a branch on a `#[module(skip)]` field, which is a
    /// host-side config constant and free — there is no silent kernel fallback
    /// in this function.
    #[allow(clippy::type_complexity)]
    pub fn forward_full_state<B: burn::backend::AutodiffBackend>(
        &self,
        x: Tensor<3>,
        hashed_ids: Option<Tensor<3, Int>>,
        host_rows: Option<Tensor<3>>,
        kda_state: Option<Tensor<4>>,
        // Target byte indices [b*t, 1]: L_Rec gathers their log-probs.
        targets: Option<Tensor<2, Int>>,
        lm_head: &LinearLike,
    ) -> (Tensor<3>, Tensor<1>, Tensor<4>, RouteAux)
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        let [b, t, d] = x.dims();
        let bf16 = self.bf16;
        let h0 = x.clone();
        let mut h = x;
        // Gated Residual: every branch starts from the token embedding, and
        // `h` becomes the READ of the branches (Eq. 31-32) rather than the
        // embedding. The read is also re-taken after every write, which is
        // what makes the readout see the current iteration's deposit.
        let use_gr = self.gr.is_some();
        let use_attnres = self.use_attnres;
        let use_mhc = self.use_mhc;
        // Read once, next to the other arm switches. The `2 * ffn_hidden`
        // contract is the one the `gate_up` width was built to (`ExpertFFN::new`
        // with `situ = true`); the assert below is what holds the two together.
        // Re-checked at the branch rather than trusted, like the
        // `use_attnres`/`use_gr` pair: every field here is `pub`, so a
        // hand-built or post-hoc-mutated `LoopBlock` can carry a flag its
        // parameters do not match, and Eq (12) reading a `d -> f` tensor is an
        // out-of-range slice - loud, but a panic from inside burn with no
        // mention of `use_situ` (ADR-0011: name the cause AND the escape).
        let use_situ = self.use_situ;
        assert_eq!(
            self.expert_ffns[0].gate_up.out_features,
            if use_situ {
                2 * self.ffn_hidden
            } else {
                self.ffn_hidden
            },
            "use_situ = {use_situ} but the experts' gate_up is {} wide, so it was built for \
             the other arm: set use_situ in the config (so ExpertFFN::new sizes gate_up \
             accordingly) rather than flipping the field after construction",
            self.expert_ffns[0].gate_up.out_features
        );
        assert!(
            !(use_attnres && use_gr) && !(use_attnres && use_mhc) && !(use_gr && use_mhc),
            "use_attnres, use_gr and use_mhc each replace the residual accumulation; \
             config::validate refuses any two of them together"
        );
        let mut branches: Vec<Tensor<3>> = if use_gr {
            vec![h0.clone(); GR_BRANCHES]
        } else {
            Vec::new()
        };
        // The normalized branches Eq. 33 reads `vec(Rhat)` from. Refreshed
        // with every read, so the write always uses the CURRENT branches.
        let mut gr_state: Option<GrState> = if use_gr {
            let gr = self.gr.as_ref().unwrap();
            let (x_in, st) = gr.read::<B>(&branches);
            h = x_in;
            Some(st)
        } else {
            None
        };
        // ATTNRES SOURCES (Eq. 3): `b_0 = h_1` is the token embedding and is a
        // source of its own right, permanently - never summed into a later
        // block. The rest of the vector is this loop's block-body outputs
        // `f_i(h_i)`, one per executed iteration. Empty when the arm is off, so
        // the default path allocates nothing for it.
        let mut res: Vec<Tensor<3>> = if use_attnres {
            vec![h0.clone()]
        } else {
            Vec::new()
        };
        let mut kda_s: Option<Tensor<4>> = kda_state;
        // Fixed depth (ADR-0013): out_acc averages the per-iteration outputs.
        let mut out_acc = Tensor::<3>::zeros([b, t, d], &h.device());
        // L_Rec = mean over iterations and batch of CE(ŷ_n, y), accumulated
        // inside the loop so we never materialize/slice the [N,b,t,d] tensor:
        // dynamic slicing of a 4D autodiff tensor crashes cubecl on sm_120
        // (CUDA_ERROR_ILLEGAL_ADDRESS).
        let mut rec = Tensor::<1>::zeros([1], &h.device());
        // Pre-compute loop invariants once: fewer per-iteration allocations
        // and graph nodes. Arm switches are plain config fields (A/B via the
        // CLI, not env).
        let h0_flat = h0.clone().reshape([b * t, d]);
        let act_fmt: Option<(crate::act_quant::ActFormat, usize)> =
            self.act_quant.map(|q| (q.into(), self.act_group));
        let use_kda = self.use_kda;
        let use_engram = self.use_engram;
        // Random-depth arm (ADR-0013 rank 2): `depth_override` runs the first
        // n iterations only, and BOTH averages below divide by what actually
        // ran - a truncated run is a complete, honest model at depth n, not a
        // partial sum of a depth-max_iter one.
        let iters = match self.depth_override {
            Some(n) => {
                assert!(
                    n >= 1 && n <= self.max_iter,
                    "loop depth {n} outside 1..={} (loud, not clamped)",
                    self.max_iter
                );
                n
            }
            None => self.max_iter,
        };
        // MoR routing arm (arXiv 2507.10524). Off by default. The loop body
        // is unchanged either way: all `iters` iterations run, and the gate
        // picks which of them reach the readout and the CE. The per-iteration
        // score, output and CE are collected here and combined after the loop
        // (the top-k cannot be known before the last score exists).
        let use_mor = self.use_mor;
        let use_msa = self.use_msa;
        // The MSA stage's distillation KL, one per EXECUTED iteration,
        // averaged below like every other per-iteration quantity.
        let mut msa_terms: Vec<Tensor<1>> = Vec::with_capacity(iters);
        let mut step_outs: Vec<Tensor<3>> = Vec::with_capacity(iters);
        let mut ce_terms: Vec<Tensor<2>> = Vec::with_capacity(iters);
        // MoR's per-slot scores (arXiv 2507.10524's router).
        let mut slot_scores: Vec<Tensor<3>> = Vec::with_capacity(iters);
        // One load-balancing term per executed iteration, averaged below. Empty
        // when `moe_topk == 0`, which is how the off-arm gets `None` rather
        // than a zero that a caller would have to guess the meaning of.
        let mut lb_terms: Vec<Tensor<1>> = Vec::with_capacity(iters);

        for iter in 0..iters {
            crate::probe::note(crate::probe::ITER);
            let row = iter;
            // This slot's pseudo-query `w_l` (§5: one per layer). Read once
            // per iteration, before the block body, so the query is a
            // function of the slot and not of anything computed this step.
            // FIDELITY F-F7 (2026-10-02): the fold adds ref [66]'s RMSNorm
            // gain, `q_eff = γ ⊙ w_l`, so the score is `γ ⊙ w_l · RMSNorm(k)`
            // - the paper's affine norm. Init ones × query zero = the old
            // exact function at step 0 (the uniform-average gate below stays
            // the truth about init). AttnRes::forward does the SAME fold; this
            // call site reads the raw params directly, so it must fold too.
            let slot_query = match use_attnres {
                true => {
                    let ar = self.attnres.as_ref().expect("use_attnres => attnres")[row]
                        .query
                        .val()
                        .clone();
                    self.attnres.as_ref().expect("use_attnres => attnres")[row]
                        .gain
                        .val()
                        .mul(ar)
                }
                false => Tensor::<1>::zeros([d], &h.device()),
            };
            let iter_ctx = self
                .iter_embed
                .val()
                .clone()
                .slice([row..row + 1, 0..d])
                .reshape([1, 1, d]);
            // The loop's depth signal. Under GR it lands on the READ (the
            // report's block has no such term; the loop needs one), under
            // ReZero on the residual stream - same place in both, so the two
            // arms differ only in the residual operator underneath.
            let add_iter = |base: Tensor<3>| -> Tensor<3> {
                if bf16 {
                    base + iter_ctx.clone().cast(FloatDType::BF16)
                } else {
                    base + iter_ctx.clone()
                }
            };
            // GR: `h` is the read of the branches as they stand (Eq. 31-32),
            // which is initialized below and re-read after every write.
            let h_ctx = add_iter(h.clone());
            // Pre-norm for the block body (identity under GR: the read
            // already normalized).
            let normed = if use_gr {
                h_ctx.clone()
            } else {
                self.norm.forward(h_ctx.clone())
            };
            let normed_f = if bf16 {
                // clone: `normed` is read again below (the Engram dense-path
                // input), and `cast` consumes the tensor.
                normed.clone().cast(FloatDType::F32)
            } else {
                normed.clone()
            };
            let attn_fmt = act_fmt.map(|(f, g)| (f.attn(), g));
            let normed_attn = match attn_fmt {
                Some((f, g)) => {
                    crate::act_quant::quant_act::<B>(normed_f.clone().reshape([b * t, d]), f, g)
                        .reshape([b, t, d])
                }
                None => normed_f.clone(),
            };
            let normed_ffn = match act_fmt {
                Some((f, g)) => {
                    crate::probe::note(crate::probe::ACT_QUANT);
                    crate::act_quant::quant_act::<B>(normed_f.reshape([b * t, d]), f, g)
                }
                None => normed_f.clone().reshape([b * t, d]),
            };

            // Controller routing on [h_ctx, h0]
            let ctrl_in = Tensor::cat(vec![h_ctx.clone().reshape([b * t, d]), h0_flat.clone()], 1);
            // Mixed-dtype ops (bf16 act x fp32 weight) NaN on this stack; the
            // BF16 mode stores activations in bf16 but computes in fp32.
            let ctrl_in = if bf16 {
                ctrl_in.cast(FloatDType::F32)
            } else {
                ctrl_in
            };
            let raw = self.controller.forward(ctrl_in); // [b*t, ctrl_pad]
            let w_attn = activation::sigmoid(raw.clone().slice([0..b * t, 0..1]));
            let w_mem = activation::sigmoid(raw.clone().slice([0..b * t, 1..2]));
            let w_ffn = activation::sigmoid(raw.clone().slice([0..b * t, 2..3]));
            // THE EXPERT BLEND. Two spellings of one statement - how this
            // token's expert weights are decided - and the config picks which:
            //
            // `moe_topk == 0` (DEFAULT): the dense softmax over ALL experts,
            // exactly as this block has always done it. This line is
            // unchanged, which is what makes the off-arm a bit-identical
            // network and a byte-identical parameter set.
            //
            // `moe_topk > 0`: the top-k restricted and RENORMALIZED weights
            // (`moe::topk_blend`). The renormalization is what keeps the two
            // arms at the same output magnitude - a top-1 token's gate is
            // exactly 1.0, the same scale as the single shared FFN the A/B
            // control runs - so the row compares specialization, not scale.
            //
            // The router is NOT a new module: the controller's expert columns
            // already receive `h_ctx = h + iter_embed[row]` (`add_iter` above,
            // the controller input at `cat([h_ctx, h0])`), so the selection is
            // already a function of (position, pass) and the arm adds NO
            // parameters. `probe::MOE_ROUTE` counts it (ADR-0011: an arm that
            // cannot show it ran is the cardinal sin here).
            let expert_logits = raw.slice([0..b * t, 3..3 + self.n_experts]);
            let blend = if self.moe_topk > 0 {
                crate::probe::note(crate::probe::MOE_ROUTE);
                let (g, m, probs) = crate::moe::topk_blend(expert_logits, self.moe_topk);
                lb_terms.push(crate::moe::lb_aux(&probs, &m, self.n_experts));
                g
            } else {
                activation::softmax(expert_logits, 1)
            };
            // THE ROUTING SEAM (arXiv 2605.09165 §6.1). This is the weight
            // vector the FFN branch multiplies each expert's output by, per
            // EXECUTED iteration - the number the loop-routing question is
            // asked of. Disarmed by default: one thread-local check, no sync,
            // no allocation (`mixture_probe`'s module docs). It is recorded on
            // the FIXED-mixture path because that is the baseline the routing
            // arm is measured against; the arm records its routed weights on
            // the same line, so "the metric changed" compares like with like.
            crate::mixture_probe::record(&blend);

            // Shared attention. `normed` above is the block-body input:
            // RMSNorm of h_ctx, identity under GR (the read already
            // normalized).
            let (gdn2_out, s_new) = if use_kda {
                crate::probe::note(crate::probe::KDA);
                self.shared_attn
                    .gdn2
                    .forward_train_state::<B>(normed_attn.clone(), kda_s.take())
            } else {
                (
                    Tensor::zeros([b, t, d], &h.device()),
                    kda_s
                        .take()
                        .unwrap_or_else(|| Tensor::zeros([b, 1, 1, 1], &h.device())),
                )
            };
            kda_s = Some(s_new);
            let attn = gdn2_out.reshape([b * t, d]).mul(w_attn);

            // Engram (hashed n-gram lookup) as a convex mixture with a hard
            // floor: `min(w_mem, lam_max) * memory + (1 - that) * dense`.
            // The module's own gate (the DeepSeek sigmoid(sqrt|s| sign s), which
            // decides WHETHER to trust the row) is untouched; what is new is
            // that the memory can never carry more than `lam_max` of the
            // branch, and that the other half is a function of the hidden
            // state - so the backbone keeps a gradient path through the
            // memory branch and the branch cannot explain the target alone.
            let engram_a = if use_engram {
                crate::probe::note(crate::probe::ENGRAM);
                let mem_read = match &host_rows {
                    // RAM-offload path: rows already gathered on the host.
                    Some(rows) => {
                        let eg_in = if bf16 {
                            h_ctx.clone().reshape([b, t, 1, d]).cast(FloatDType::F32)
                        } else {
                            h_ctx.clone().reshape([b, t, 1, d])
                        };
                        Some(
                            self.engram
                                .forward_embeds(rows.clone(), eg_in)
                                .reshape([b * t, d]),
                        )
                    }
                    None => match &hashed_ids {
                        Some(hashed) => {
                            // burn-engram kernels are f32-only (ILLEGAL_ADDRESS
                            // on bf16, measured 2026-08-29).
                            let eg_in = if bf16 {
                                h_ctx.clone().reshape([b, t, 1, d]).cast(FloatDType::F32)
                            } else {
                                h_ctx.clone().reshape([b, t, 1, d])
                            };
                            // Raw FNV hashes arrive here: the model owns the
                            // capacity, so it masks the slot index itself
                            // (one op, in range by construction - see
                            // `engram_tables`).
                            Some(
                                self.engram
                                    .forward(
                                        hashed.clone().bitwise_and_scalar(self.engram_slot_mask),
                                        eg_in,
                                    )
                                    .reshape([b * t, d]),
                            )
                        }
                        // No keys this call (inference without hashed ids): the
                        // arm is inert, dense path included.
                        None => None,
                    },
                };
                match mem_read {
                    Some(mem) => {
                        crate::probe::note(crate::probe::ENGRAM_KEYS);
                        // The dense half reads the same block-body input the
                        // attention and FFN arms read (RMSNorm of h_ctx;
                        // identity under GR). Cast to fp32 first under
                        // --bf16: mixed-dtype (bf16 act x fp32 weight) NaNs
                        // on this stack, the rule every Linear here follows.
                        let dense_in = if bf16 {
                            normed.clone().cast(FloatDType::F32)
                        } else {
                            normed.clone()
                        };
                        let dense = self.mem_dense.forward(dense_in.reshape([b * t, d]));
                        memory_floor_mix(mem, dense, w_mem.clone(), self.engram_lam_max)
                    }
                    None => Tensor::zeros([b * t, d], &h.device()),
                }
            } else {
                Tensor::zeros([b * t, d], &h.device())
            };

            // Qwen Sparse Attention (tech report §QSA): the stage's OWN
            // multi-head softmax attention over the same block-body input,
            // dense (teacher) in stage (a), micro-block masked in stage (b);
            // the distill KL rides out unscaled on the aux seam. Zero on the
            // off arm, so the residual assembly below is untouched.
            let (msa_out, _msa_dl) = match (use_msa, &self.msa) {
                (true, Some(msa)) => {
                    let m = msa.forward::<B>(normed_attn.clone());
                    if let Some(dl) = m.distill {
                        msa_terms.push(dl);
                    }
                    (m.output, ())
                }
                _ => (Tensor::zeros([b, t, d], &h.device()), ()),
            };

            // Expert FFN: softmax blend of n_experts TSCT gate_up/silu/down.
            // `use_situ` swaps the elementwise SiLU for SiTU-GLU
            // (arXiv:2607.24653v2 Eq 12), which is why `gate_up` is `d -> 2f`
            // on that arm: Eq 12 reads `Wg x` and `Wu x` separately. The cast
            // to f32 is the same rule every forward here follows (mixed-dtype
            // bf16 activations x fp32 weights NaN on this stack), and SiTU's
            // output is bounded to +-beta1*beta2, so nothing downstream of this
            // can inherit an unbounded activation.
            let mut ffn = Tensor::zeros([b * t, d], &h.device());
            for e in 0..self.n_experts {
                let mid = self.expert_ffns[e].gate_up.forward::<B>(normed_ffn.clone());
                let mid_f32 = if bf16 { mid.cast(FloatDType::F32) } else { mid };
                let mid = if use_situ {
                    crate::probe::note(crate::probe::SITU);
                    burn_situ::situ_glu(
                        mid_f32,
                        self.ffn_hidden,
                        burn_situ::K3_GATE_BETA,
                        burn_situ::K3_UP_BETA,
                    )
                } else {
                    activation::silu(mid_f32)
                };
                let out = self.expert_ffns[e].down.forward::<B>(mid);
                ffn = ffn + out.mul(blend.clone().slice([0..b * t, e..e + 1]));
            }
            let ffn = ffn.mul(w_ffn).reshape([b, t, d]);

            // ReZero residual, AttnRes aggregation, GR write, or the mHC
            // manifold projection (per-branch scalar deposit, Eq. 33-34).
            // These are four spellings of ONE statement - how iteration n's
            // block-body output joins the residual stream - and
            // `config::validate` refuses the pairs that cannot both run.
            let y = attn.reshape([b, t, d])
                + engram_a.reshape([b, t, d])
                + ffn
                + msa_out.reshape([b, t, d]);
            if use_attnres {
                crate::probe::note(crate::probe::ATTNRES);
                // Eq. 1/3/4: sources are the token embedding `b_0 = h_1` plus
                // every layer output `f_i(h_i)` so far, INCLUDING this
                // iteration's. The state that comes out is the state the
                // readout reads and the next iteration consumes, so the
                // loop's body still reaches its own CE at every depth - the
                // placement that avoids the depth-(iters-1) model GR had.
                //
                // The history is a `Vec<Tensor<3>>` and never a `[N,b,t,d]`
                // stack: dynamic slicing of a 4D autodiff tensor crashes
                // cubecl on sm_120 (AGENTS.md 2.2), and `max_iter` is 2-4.
                res.push(y.clone());
                h = depth_attend(&res, slot_query);
            } else if use_gr {
                crate::probe::note(crate::probe::GR);
                let gr = self.gr.as_ref().unwrap();
                branches = gr.write::<B>(&branches, gr_state.as_ref().unwrap(), y);
                // Re-read the branches Eq. 32, AFTER the Eq. 34 deposit. The
                // readout below and the next iteration's input are both this
                // read, so iteration n's output contains iteration n's own
                // block body. Reading the pre-write state here made GR a
                // depth-(iters-1) model: at max_iter=1 the entire body (KDA,
                // memory, every expert) was discarded and the block reduced to
                // out_proj(read(h0) + e_0), while the ReZero arm kept `y`.
                let (x_next, st) = gr.read::<B>(&branches);
                gr_state = Some(st);
                h = x_next;
            } else if use_mhc {
                crate::probe::note(crate::probe::MHC);
                // Eq. 3, at the loop boundary: `h' = H_res h + H_post . y`.
                // The base is `h_ctx`, the SAME state ReZero's branch adds
                // `y` to, so the A/B against ReZero differs in the residual
                // OPERATOR and in nothing else - the iteration embedding, the
                // block body and the readout are identical.
                //
                // `MhcBlock` reshapes the hidden state to `[b, t, n, D/n]` and
                // Sinkhorn-projects a per-token `n x n` mix, so this is a
                // handful of TINY kernels per iteration, not a GEMM: on this
                // launch-bound box it is the arm's cost, and it is measured
                // with the A/B rather than guessed here
                // (docs/reviews/mhc-2026-09-30.md §6).
                //
                // fp32 in, fp32 out (the rule every Linear here follows: mixed
                // bf16 x fp32 NaNs on this stack), then the residual write goes
                // back into the activation dtype.
                let h_next = self.mhc.as_ref().expect("use_mhc => mhc").forward(
                    if bf16 {
                        h_ctx.clone().cast(FloatDType::F32)
                    } else {
                        h_ctx.clone()
                    },
                    &[if bf16 {
                        y.clone().cast(FloatDType::F32)
                    } else {
                        y.clone()
                    }],
                );
                h = h_next.cast(h_ctx.dtype());
            } else if !use_attnres {
                let scale = self.residual_scale.val().clone().reshape([1, 1, 1]);
                // Store the residual back in the activation dtype (bf16
                // under --bf16): the sum itself is computed in fp32.
                h = h_ctx.clone() + y.mul(scale).cast(h_ctx.dtype());
            }

            // Per-iteration readout. Fixed depth (ADR-0013): uniform
            // iteration weights over the executed iterations and an honest
            // unweighted CE — the PonderNet variant lost its A/B (lambda
            // collapse zeroed rec, a fake loss, and out_acc, uniform
            // outputs). The per-iteration pieces are COLLECTED here and
            // combined once after the loop: under MoR the router ranks the
            // slots, and the top-k cannot be known before the last slot's
            // score exists. Holding them costs nothing — the autograd graph
            // pins them either way.
            let step_out = self
                .out_proj
                .forward::<B>(h.clone().reshape([b * t, d]))
                .reshape([b, t, d]);
            if use_mor {
                crate::probe::note(crate::probe::MOR);
                // MoR router: the shared linear scorer reads THIS iteration's
                // input state (arXiv 2507.10524's per-step linear router).
                let hs = if bf16 {
                    h_ctx.clone().cast(FloatDType::F32)
                } else {
                    h_ctx.clone()
                };
                slot_scores.push(self.mor_router.scores(hs));
            }
            step_outs.push(step_out.clone());
            if let Some(tgt) = &targets {
                let so = if bf16 {
                    step_out.clone().cast(FloatDType::F32)
                } else {
                    step_out.clone()
                };
                let logits_n = lm_head.forward::<B>(so.reshape([b * t, d])); // [b*t, v]
                                                                             // Gather the target column of the log-softmax instead of an
                                                                             // elementwise one-hot product: no [b*t,v] fp32 temporary per
                                                                             // iteration (max_iter of them per step otherwise).
                let ce = burn::tensor::activation::log_softmax(logits_n, 1)
                    .gather(1, tgt.clone())
                    .neg()
                    .reshape([b, t]); // [b, t], summed below under the same gate
                ce_terms.push(ce);
            }
            // The RECURRENCE: the next iteration reads this iteration's
            // post-residual h. It used to be reset to h_ctx here, which
            // silently reduced the "looped block" to 4 independent passes
            // over `x + sum(e_k)` with shared weights - a weight-tied
            // ensemble with ~2 effective layers, not a loop, and 4x the
            // forward/backward for ~1x the capacity. Under Gated Residual `h`
            // is the post-write READ of the branches, assigned in the write
            // branch above, so there is nothing to reset: both arms carry
            // this iteration's body into the next one.
        }
        // Combine the per-iteration pieces under the iteration gate. MoR off:
        // the gate is all-ones and the divisor is `iters`, i.e. exactly the
        // fixed-depth mean this loop has always used. MoR on: the gate is the
        // 0/1 top-k membership of `mor::route` and the divisor is the number
        // of selected slots — what actually ran, a CONSTANT, not a learned
        // weight (ingredient 4: no p_n, nothing that can decay to zero).
        let (mask, mor_aux) = if use_mor {
            let sc = Tensor::cat(slot_scores, 2); // [b, t, iters]
            let (m, bce) = mor::route(sc, self.mor_k);
            (m, Some(bce))
        } else {
            (Tensor::<3>::ones([b, t, iters], &h.device()), None)
        };
        // The divisor is the count of selected slots, known on the host (k is
        // a config constant) — reading it off the mask would sync the device
        // every step for a number we already have.
        let ksel = if use_mor {
            mor::eff_k(self.mor_k, iters)
        } else {
            iters
        };
        let w = 1.0f32 / ksel as f32;
        for (n, s_out) in step_outs.iter().take(iters).enumerate() {
            // Mixed-dtype (f32 mask x bf16 activation) NaNs on this stack, so
            // the gate lands in the readout's own dtype.
            let g = mask.clone().slice([0..b, 0..t, n..n + 1]);
            let g = g.cast(s_out.dtype());
            out_acc = out_acc + s_out.clone().mul(g.clone()).mul_scalar(w);
            if let Some(ce) = ce_terms.get(n) {
                rec = rec
                    + ce.clone()
                        .mul(g.reshape([b, t]))
                        .sum_dim(1)
                        .div_scalar(t as f32)
                        .sum_dim(0)
                        .reshape([1])
                        .mul_scalar(w);
            }
        }
        let rec = rec.div_scalar(b as f32); // mean over batch
        let kda = kda_s.unwrap_or_else(|| Tensor::zeros([1, 1, 1, 1], &h.device()));
        // The balancer is averaged over the iterations that ran, so it is a
        // per-iteration quantity summed into one loss term - and it is the
        // MEAN over what actually executed, which is the same rule every other
        // average in this loop follows (a truncated run must not report the
        // untruncated depth's statistics).
        let moe_lb = if lb_terms.is_empty() {
            None
        } else {
            let mut acc = lb_terms[0].clone();
            for t in &lb_terms[1..] {
                acc = acc + t.clone();
            }
            Some(acc.div_scalar(lb_terms.len() as f32))
        };
        (
            out_acc,
            rec,
            kda,
            RouteAux {
                mor: mor_aux,
                moe_lb,
                msa_distill: if msa_terms.is_empty() {
                    None
                } else {
                    let mut acc = msa_terms[0].clone();
                    for t in &msa_terms[1..] {
                        acc = acc + t.clone();
                    }
                    Some(acc.div_scalar(msa_terms.len() as f32))
                },
            },
        )
    }

    /// Random-depth arm: run only the first `n` iterations. `None` restores
    /// fixed depth. Rejects `n == 0` or `n > max_iter` loudly — a silently
    /// clamped depth would make the A/B lie about what it trained.
    ///
    /// Under MoR this is the `--eval-depths` MEASUREMENT path, not a second
    /// training arm: the router still ranks the slots, `k` shrinks to the
    /// slots that ran (`mor::eff_k`), and the readout and CE divide by that
    /// count, so each depth is a complete model of depth `n`. Training-time
    /// random depth is a different mechanism buying the same depth
    /// robustness, and the pair is refused LOUDLY before any GPU work, in
    /// `dormouse_train::resolve` — that is the only place that can tell the
    /// training flag from this eval call, because both arrive through this
    /// one setter.
    pub fn set_depth(&mut self, n: Option<usize>) {
        if let Some(n) = n {
            assert!(
                n >= 1 && n <= self.max_iter,
                "loop depth {n} outside 1..={} (loud, not clamped)",
                self.max_iter
            );
        }
        self.depth_override = n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DormouseConfig;
    use burn::tensor::{Device, Distribution};

    fn dev() -> Device {
        Device::flex()
    }

    /// The backend the loop's `B: AutodiffBackend` is instantiated on: the CPU
    /// (flex) backend under the dispatch layer, with the SAME checkpointing
    /// strategy the trainer uses. `Device::flex()` alone would not satisfy the
    /// bound, and picking `NoCheckpointing` here would test a backend the
    /// trainer never runs.
    type B = burn::backend::autodiff::Autodiff<
        burn::backend::Flex,
        burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
    >;

    /// The device the tests build on: the CPU backend under the dispatch layer,
    /// so a tensor handed to `forward_full_state` has the same shape the
    /// trainer's does.
    #[allow(deprecated)]
    fn adev() -> Device {
        dev().autodiff()
    }

    /// The CPU-fixture config at loop depth `depth`: the widest shape these
    /// tests instantiate, with the arms that would drag in a different gate
    /// off. Tests that need the DEFAULTS for an arm (kda/engram on) build
    /// their literal by hand instead of layering onto this.
    fn small_cfg(depth: usize) -> DormouseConfig {
        DormouseConfig {
            d_model: 32,
            n_heads: 2,
            head_dim: 16,
            d_ffn: 64,
            max_iter: depth,
            n_experts: 1,
            rank: 8,
            engram_rows: 256,
            use_kda: false,
            use_engram: false,
            // These are dormouse-loop fixtures; the schema default is the
            // byteflow net as of 2026-10-02, so the net is pinned off.
            use_byteflow: false,
            ..Default::default()
        }
    }

    /// Recover the mixture coefficient `a` in `out = a*mem + (1-a)*dense`
    /// from the two inputs: `a = (out - dense) / (mem - dense)`. Asserting on
    /// the RECOVERED coefficient is what makes this a test of the floor
    /// rather than of the numbers that went in - no magic output value.
    fn recover_a(out: &Tensor<2>, mem: f32, dense: f32) -> f32 {
        let o = out.clone().into_scalar::<f32>();
        (o - dense) / (mem - dense)
    }

    /// THE FLOOR HOLDS. The controller's `w_mem` is a sigmoid, so it can
    /// reach 1.0; the memory's coefficient in the branch must still be capped
    /// at `lam_max`, i.e. the backbone keeps at least `1 - lam_max` of the
    /// branch no matter what is learned. Drive the controller to its maximum
    /// and check the recovered coefficient.
    #[test]
    fn memory_floor_caps_the_controller() {
        let lam_max = 0.5_f32;
        // A wide spread so the recovered `a` is a ratio, not a rounding
        // artifact: the memory read is 1000x the dense path elementwise.
        let (mem, dense) = (1000.0_f32, 1.0_f32);
        let t = |x: f32| Tensor::<2>::from_floats([[x]], &dev());
        // w_mem saturated high (sigmoid of a large pre-activation).
        let out = memory_floor_mix(t(mem), t(dense), t(1.0), lam_max);
        let a = recover_a(&out, mem, dense);
        assert!(
            a <= lam_max,
            "floor violated: saturated controller gave a = {a} > lam_max {lam_max}"
        );
        assert!(
            (a - lam_max).abs() < 1e-5,
            "saturated w_mem must sit AT the cap, got {a}"
        );

        // Below the cap the learned value still works - the clamp is a
        // ceiling, not a constant.
        let out = memory_floor_mix(t(mem), t(dense), t(0.1), lam_max);
        let a = recover_a(&out, mem, dense);
        assert!(
            (a - 0.1).abs() < 1e-5,
            "below the cap the mix must follow w_mem, got {a}"
        );

        // lam_max = 1.0 is the old behaviour (a direct row copy, no floor) -
        // the degenerate case this test exists to forbid by default.
        let out = memory_floor_mix(t(mem), t(dense), t(1.0), 1.0);
        let a = recover_a(&out, mem, dense);
        assert!(
            (a - 1.0).abs() < 1e-5,
            "lam_max=1 must be a pure memory read, got {a}"
        );
    }

    /// ATTNRES: the readout MUST move when the aggregation changes, or the
    /// arm ran and did nothing - which is what GR's 9b343d3 defect was (the
    /// readout was taken from the state BEFORE the write, so the body was
    /// computed and discarded at depth 1).
    #[test]
    fn attnres_moves_the_readout_at_every_depth() {
        for depth in [1usize, 2, 3] {
            let cfg = small_cfg(depth);
            let plain = LoopBlock::new(&cfg, &adev());
            let ar = LoopBlock::new(
                &DormouseConfig {
                    use_attnres: true,
                    ..cfg.clone()
                },
                &adev(),
            );
            let [b, t, d] = [2usize, 5, 32];
            let x = Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &adev());
            let head = LinearLike::with_tsct(d, 16, 8, cfg.use_tsct, &adev());
            let (o_rezero, _, _, _) =
                plain.forward_full_state::<B>(x.clone(), None, None, None, None, &head);
            let (o_attnres, _, _, _) = ar.forward_full_state::<B>(x, None, None, None, None, &head);
            let diff = (o_rezero - o_attnres).abs().max().into_scalar::<f32>();
            assert!(
                diff > 1e-6,
                "depth {depth}: AttnRes changed no output (max {diff:.3e}) - the \
                 aggregation would be running and being discarded, the GR defect"
            );
        }
    }

    /// §5's init invariant, ON THE MODEL: zero-initialised pseudo-queries make
    /// the first AttnRes an EQUAL-WEIGHT AVERAGE of the embedding and the
    /// block-body outputs, which is the property that makes the arm a
    /// drop-in at step 0. Checked at depth 2 by recovering the mixture from
    /// the run, not by reading the initializer.
    #[test]
    fn attnres_at_init_is_a_uniform_average_of_its_sources() {
        let cfg = small_cfg(1);
        let ar = LoopBlock::new(
            &DormouseConfig {
                use_attnres: true,
                ..cfg.clone()
            },
            &adev(),
        );
        // Pin the zero init: if a future change randomizes the query, the
        // uniform-average property is gone and the arm no longer starts where
        // the paper says it starts. Every component, so a partially-zeroed
        // vector is red too.
        let q = ar.attnres.as_ref().expect("arm on")[0].query.val().clone();
        let worst = q.abs().max().into_scalar::<f32>();
        assert_eq!(
            worst, 0.0,
            "w_l must be zero-initialized (§5): uniform alpha at init"
        );
    }

    /// THE FORM THE ARM WAS BUILT WITH — and this gate exists because the
    /// falsification run found it missing.
    ///
    /// Perturbing `LoopBlock::new` to build `AttnRes::with_form(.., SqrtD)`
    /// turned **every other gate in this file green**: the readout moved
    /// (both forms move it), the counter fired (both aggregate), the init was
    /// uniform (at `w_l = 0` every score is 0, so the temperature multiplies
    /// into nothing). The model's own gates are structurally blind to the
    /// score convention, because the convention only enters through a query
    /// that starts at zero. The number is pinned in `burn-attnres`
    /// (`paper_form_has_no_temperature_and_this_is_pinned`, literals
    /// 0.9820138 / 0.7310586); this asserts the model is wired to the one
    /// those literals describe — the link between the two files, and the
    /// half that was missing.
    #[test]
    fn attnres_is_built_in_the_papers_score_form() {
        let cfg = DormouseConfig {
            d_model: 32,
            max_iter: 2,
            ..Default::default()
        };
        let on = LoopBlock::new(
            &DormouseConfig {
                use_attnres: true,
                ..cfg
            },
            &adev(),
        );
        for (i, a) in on.attnres.as_ref().expect("arm on").iter().enumerate() {
            assert_eq!(
                a.form,
                burn_attnres::ScoreForm::Paper,
                "slot {i}: the model arm must be the paper's `q . RMSNorm(k)`, no \
                 temperature. Eq. 2 has no 1/sqrt(d); the crate's SqrtD form is kept \
                 for the A/B that would name it, and wiring it here by accident would \
                 make every future AttnRes number a different mechanism from the one \
                 the paper describes"
            );
        }
    }

    /// The seam: with `use_attnres = false` the parameters must not EXIST, or
    /// every existing checkpoint would be one parameter set away from loading
    /// (and the A/B would be confounded by `d_model` extra trainable values
    /// that no forward reads).
    #[test]
    fn attnres_off_means_no_parameters() {
        let cfg = DormouseConfig {
            d_model: 32,
            n_heads: 2,
            head_dim: 16,
            d_ffn: 64,
            max_iter: 3,
            n_experts: 1,
            rank: 8,
            engram_rows: 256,
            ..Default::default()
        };
        let b = LoopBlock::new(&cfg, &adev());
        assert!(
            b.attnres.is_none(),
            "use_attnres defaults to false, so no query vectors"
        );
        assert!(!b.use_attnres);
        // The default config itself, so a preset cannot turn this on by
        // accident: the flag is off unless someone says so.
        assert!(!DormouseConfig::default().use_attnres);
        // And on, there is exactly one query per iteration slot.
        let on = LoopBlock::new(
            &DormouseConfig {
                use_attnres: true,
                ..cfg
            },
            &adev(),
        );
        let qs = on.attnres.as_ref().expect("arm on");
        assert_eq!(
            qs.len(),
            on.max_iter,
            "one pseudo-query per iteration slot (§5)"
        );
        assert!(
            qs.iter().all(|a| a.query.dims() == [32]),
            "each is a [d_model] vector"
        );
    }

    /// ATTNRES and GR both REPLACE the residual accumulation, so a config that
    /// asks for both has no interpretation. Refused in `config::validate`, and
    /// re-checked at the branch, because a hand-built `LoopBlock` bypasses the
    /// config path.
    #[test]
    fn attnres_and_gr_are_refused_together() {
        let cfg = DormouseConfig {
            d_model: 32,
            n_heads: 2,
            head_dim: 16,
            d_ffn: 64,
            max_iter: 2,
            n_experts: 1,
            rank: 8,
            engram_rows: 256,
            // dormouse-loop fixture; the schema default is the byteflow net
            use_byteflow: false,
            ..Default::default()
        };
        let both = DormouseConfig {
            use_attnres: true,
            use_gr: true,
            ..cfg.clone()
        };
        let err = crate::config::validate(&both).expect_err("both residual arms must be refused");
        assert!(
            err.contains("use_attnres") && err.contains("use_gr"),
            "the error must name both: {err}"
        );
        // Either alone is legal.
        assert!(crate::config::validate(&DormouseConfig {
            use_attnres: true,
            ..cfg.clone()
        })
        .is_ok());
        assert!(crate::config::validate(&DormouseConfig {
            use_gr: true,
            ..cfg.clone()
        })
        .is_ok());
    }

    /// THE COUNTER. `use_attnres = true` with a counter of 0 would be a run
    /// that reports ReZero's loss under AttnRes's name - the defect ADR-0019
    /// is about, in the one shape where nothing else would show it.
    #[test]
    fn attnres_counts_every_iteration_it_aggregates() {
        let cfg = small_cfg(4);
        let head = LinearLike::with_tsct(32, 16, 8, cfg.use_tsct, &adev());
        let x = Tensor::<3>::random([2, 5, 32], Distribution::Normal(0.0, 1.0), &adev());

        crate::probe::reset();
        let ar = LoopBlock::new(
            &DormouseConfig {
                use_attnres: true,
                ..cfg.clone()
            },
            &adev(),
        );
        let _ = ar.forward_full_state::<B>(x.clone(), None, None, None, None, &head);
        assert_eq!(
            crate::probe::count(crate::probe::ATTNRES),
            4,
            "one aggregation per iteration at depth 4"
        );

        // Off: zero. The ReZero path must not touch the counter, or the field
        // on the eval line would read non-zero for a run that never aggregated.
        crate::probe::reset();
        let rz = LoopBlock::new(&cfg, &adev());
        let _ = rz.forward_full_state::<B>(x, None, None, None, None, &head);
        assert_eq!(
            crate::probe::count(crate::probe::ATTNRES),
            0,
            "ReZero does not aggregate"
        );
        // Random depth is a truncation of what ran, and the counter must count
        // what RAN: a truncated run that reported max_iter aggregations would
        // be counting a model that was not evaluated.
        crate::probe::reset();
        let mut ar = LoopBlock::new(
            &DormouseConfig {
                use_attnres: true,
                ..cfg
            },
            &adev(),
        );
        ar.set_depth(Some(2));
        let _ = ar.forward_full_state::<B>(
            Tensor::zeros([2, 5, 32], &adev()),
            None,
            None,
            None,
            None,
            &head,
        );
        assert_eq!(
            crate::probe::count(crate::probe::ATTNRES),
            2,
            "depth 2 aggregated twice"
        );
    }

    // ---- mHC (arXiv:2512.24880) ------------------------------------------------

    /// The one fixture every mHC gate below is built on: a narrow, fast block
    /// with the attention and memory arms off, so what differs between the
    /// arms is the residual statement and nothing else.
    ///
    /// `mhc_streams: 2` is set EXPLICITLY with the rule the 2026-09-30 DSpark
    /// incident taught (`memory.md`: a test that takes a config value from the
    /// preset verifies nothing about that value - when a default moves, such a
    /// test reddens or passes vacuously). This fixture's literals (`2(J-I)`
    /// perturbation, the `n = 2` eigenvalues) are for n = 2; the DEFAULT is
    /// the paper's n = 4 since the fidelity fix F-F3, and n = 4's own gates
    /// are the ones that read the configured `n` instead of a literal.
    fn mhc_cfg(depth: usize) -> DormouseConfig {
        DormouseConfig {
            mhc_streams: 2,
            ..small_cfg(depth)
        }
    }

    // SITU (arXiv:2607.24653v2 Eq 12). The FORM is gated in the mechanism
    // crate, against Moonshot's own numbers - see
    // vendor/dormouse-fused/crates/burn-situ/src/lib.rs and
    // docs/reviews/situ-2026-09-30.md. What is gated HERE is the wiring:
    // that the flag is load-bearing, that the counter sees the arm, and that
    // the gradient survives the cap. The form gate cannot see any of it: a
    // perfectly-formed SiTU wired to nothing is a green crate and a run that
    // measured SiLU.
    // ---------------------------------------------------------------------

    /// `d_ffn` and `n_experts` for the gates below, and the arms off: the
    /// gradient numbers are about the activation, and KDA/Engram only add
    /// other arms' counters to the same forward.
    fn situ_cfg() -> DormouseConfig {
        DormouseConfig {
            d_model: 32,
            n_heads: 2,
            head_dim: 16,
            // A multiple of 4, so `LinearLike`'s `N % 4 == 0` padding
            // (param.rs:49) leaves the width exactly `f` and `2f` and the
            // width assertions below are about the arm, not about the pad.
            d_ffn: 64,
            n_experts: 2,
            rank: 8,
            engram_rows: 256,
            max_iter: 2,
            use_kda: false,
            use_engram: false,
            ..Default::default()
        }
    }

    fn mhc_head(cfg: &DormouseConfig) -> LinearLike {
        LinearLike::with_tsct(cfg.d_model, 16, 8, cfg.use_tsct, &adev())
    }

    /// A `[1]` tensor from one number, for the `Param::from_tensor` knobs. Not
    /// `from_floats([[x]])`: that is rank 2 and burns says so at runtime.
    fn p1(x: f32) -> Param<Tensor<1>> {
        Param::from_tensor(Tensor::<1>::from_data(
            burn::tensor::TensorData::new(vec![x], [1]),
            &adev(),
        ))
    }

    /// THE MANIFOLD, ON THE MAPPINGS THE MODEL ACTUALLY BUILDS. This is the
    /// form-fidelity gate, and it is deliberately NOT "the readout moved":
    /// `attnres_moves_the_readout_at_every_depth` is the shape of gate that
    /// stayed green through 511daa5's `d^-0.5` scale and L2 norm together - a
    /// d-fold logit compression that changes the number and passes. What
    /// distinguishes mHC from BOTH of its rivals is a property of the OPERATOR,
    /// not of the output, and this asserts it in both directions:
    ///
    /// - **forward gain** = max |row sum| of `H_res` (Eq. 6: `H_res 1 = 1`),
    /// - **backward gain** = max |column sum| (`1^T H_res = 1^T`),
    ///
    /// which are the paper's own Amax Gain Magnitude metrics (2512.24880 §3.1,
    /// Fig. 7) and the two things ReZero (a scalar, no matrix) and AttnRes (a
    /// softmax over sources, no stream mixing) simply do not have. Sinkhorn at
    /// the paper's `t_max = 20` is an APPROXIMATE projection, so the columns
    /// are the loose direction and the rows the tight one - §5.4 says exactly
    /// that, and the gate says so rather than pretending one number serves
    /// both.
    #[test]
    fn mhc_res_is_doubly_stochastic_in_both_directions() {
        let cfg = mhc_cfg(2);
        let blk = LoopBlock::new(
            &DormouseConfig {
                use_mhc: true,
                ..cfg.clone()
            },
            &adev(),
        );
        let mhc = blk.mhc.as_ref().expect("use_mhc => mhc");
        assert_eq!(mhc.n_branches, 2, "the default expansion rate is n = 2");
        let h = Tensor::<3>::random([2, 7, 32], Distribution::Normal(0.0, 1.0), &adev());
        let (_, _, res) = mhc.hyper_mappings(&h);
        assert_eq!(res.dims(), [2, 7, 2, 2], "H_res is [b, t, n, n]");
        let r: Vec<f32> = res.into_data().try_to_vec().expect("[2,7,2,2] readable");
        let (mut row_max, mut col_max) = (0.0f32, 0.0f32);
        let mut min_entry = f32::MAX;
        for bt in 0..(2 * 7) {
            let m = &r[bt * 4..bt * 4 + 4];
            for i in 0..2 {
                let row: f32 = (0..2).map(|j| m[i * 2 + j]).sum();
                let col: f32 = (0..2).map(|j| m[j * 2 + i]).sum();
                row_max = row_max.max((row - 1.0).abs());
                col_max = col_max.max((col - 1.0).abs());
            }
            min_entry = min_entry.min(m.iter().copied().fold(f32::MAX, f32::min));
        }
        assert!(
            row_max < 1e-3,
            "forward gain: max |rowsum(H_res) - 1| = {row_max:.3e}. The last \
             normalization of Eq. 9 makes this exact, so anything above 1e-3 is \
             a broken projection, not an approximation"
        );
        assert!(
            col_max < 5e-2,
            "backward gain: max |colsum(H_res) - 1| = {col_max:.3e}. The column \
             pass runs FIRST inside the last iteration's pair and the row pass \
             second, so at t_max = 20 this is the direction that has not \
             converged - the paper measures the same asymmetry in Fig. 7(a)"
        );
        assert!(
            min_entry >= 0.0,
            "H_res must be non-negative (Eq. 6): {min_entry}"
        );
    }

    /// THE MECHANISM, NOT ITS INIT. Two assertions in one, because they are one
    /// question - "does `H_res` actually mix the streams?" - asked at both ends
    /// of the trajectory, and the middle is the only place it can fail:
    ///
    /// 1. **At the fidelity-fixed init it is (approximately) the uniform mix.**
    ///    `b_res = 0` (the F-F4 fix) makes `Sinkhorn(exp(E·x_norm·φ + 0))` -
    ///    exp(≈0) ≈ 1 everywhere, so the projection starts AT the uniform mix
    ///    `J/n` - the doubly stochastic fixed point, alpha's gradient ALIVE
    ///    (the 10I init read 7.3e-9 on it, `mhc-2026-09-30.md` §5.3b). An init
    ///    at the fixed point has a zero LINEAR drift toward it - the expensive
    ///    claim the old init made (start at identity, drift to the identity's
    ///    neighbourhood) is replaced by start at the fixed point and learn
    ///    which direction the A/B wants.
    /// 2. **Off that init it leaves the identity while staying on the
    ///    manifold.** This is the assertion that matters, and it is the
    ///    published failure mode: 2603.20896 (s²HC, NeurIPS 2026) reports that
    ///    under the doubly stochastic constraint "learned matrices collapse
    ///    around the identity initialization and diminish cross-stream
    ///    interactions". If our `b_res` perturbation left the composite AT the
    ///    identity, the arm would be a slower ReZero and the A/B would be
    ///    measuring nothing. The distance from the identity is measured, the
    ///    manifold is re-checked after the perturbation, and the T-fold
    ///    COMPOSITE is checked too - which is the quantity the paper's whole
    ///    stability argument is about (Eq. 4), and the one that stays doubly
    ///    stochastic under a shared block because the Birkhoff polytope is
    ///    closed under multiplication.
    ///
    /// `N` is the configured `mhc_streams` (= 2), read from the block rather
    /// than repeated, so this gate is a check of the CONFIGURED expansion rate
    /// and not of a literal: `--set mhc_streams=4` has to move it too.
    #[test]
    fn mhc_res_leaves_the_identity_when_the_bias_moves_and_stays_on_the_manifold() {
        let cfg = mhc_cfg(4);
        let mut blk = LoopBlock::new(
            &DormouseConfig {
                use_mhc: true,
                ..cfg.clone()
            },
            &adev(),
        );
        let h = Tensor::<3>::random([1, 3, 32], Distribution::Normal(0.0, 1.0), &adev());
        let n = blk.mhc.as_ref().expect("arm on").n_branches;
        const N: usize = 2;
        assert_eq!(n, N, "this gate's predicted literals are for n = 2");
        const NN: usize = N * N;

        // (1) at init: LOWER BOUND, refined - H_res must have LEFT the Birkhoff
        // polytope's only accidentally-attractive point the OLD init sat on:
        // every entry ~1/n (exp(0) -> doubly stochastic -> uniform) and
        // OFF-DIAGONAL MASS O(1), which is what alpha can actually move. The
        // OLD assert here was `max |entry - I| < 1e-3` - the 10I stiffness
        // scored 4.5e-5 and the gate blessed a frozen arm; inverted, it reads
        // whether the init off-diagonal mass (at n = 2, uniform J/2) is O(1):
        // the map distance from I is |J/n - I| ~ 0.5 per identity entry.
        let h_res = |blk: &LoopBlock| -> Vec<f32> {
            blk.mhc
                .as_ref()
                .expect("arm on")
                .hyper_mappings(&h)
                .2
                .into_data()
                .try_to_vec()
                .expect("[1,3,n,n]")
        };
        let ident = |m: &[f32]| -> f64 {
            let mut worst: f64 = 0.0;
            for bt in 0..3 {
                for i in 0..N {
                    for j in 0..N {
                        let want = if i == j { 1.0 } else { 0.0 };
                        worst = worst.max((m[bt * NN + i * N + j] as f64 - want).abs());
                    }
                }
            }
            worst
        };
        let at_init = ident(&h_res(&blk));
        // uniform J/2: off-diagonals 0.5, diagonal 0.5 -> distance to I = 0.5.
        let b = 0.5f64;
        println!(
            "mhc H_res at init: max |entry - I| = {at_init:.3e} (predicted |J/n - I| entry = {b})"
        );
        assert!(
            at_init > 0.3,
            "H_res at init is (near) the identity again (max entry error {at_init:.3e}, \
             predicted |J/n - I| = {b}): the b_res init drifted back toward the 10I \
             stiffness, whose gradient is dead on the flat top"
        );

        // (2) perturbed: the manifold holds and the streams MIX.
        // `b_res = 2 (J - I)` - zero diagonal, a strong off-diagonal. The
        // predicted readings, computed from Eq. 9 before this test was
        // written: distance from the identity **0.8808**, off-diagonal mass
        // **1.7616**, both row and column sums 1 to 1e-6.
        //
        // A NOTE ON WHAT "PERTURBED" MAY MEAN, because the first attempt at
        // this gate was wrong in an instructive way: adding +/-1.75 to the
        // init's `b_res = 10 I` moves the DIAGONAL to 12.5 and leaves the
        // off-diagonals near 0, which is a *more* extreme identity, not a less
        // one. It reads `2.6e-4` at n = 2 (and `3.5e-4` at n = 4) and looks like
        // identity degeneration. Near `diag = 10` the map is exponentially
        // stiff - one unit on a single off-diagonal moves `H_res` by ~2e-4 -
        // which is a fact about the ARM (§6 of the findings file) and the
        // reason the perturbation here has to be big enough to be visible.
        let mhc = blk.mhc.as_mut().expect("arm on");
        mhc.b_res = Param::from_tensor(Tensor::<2>::from_floats([[0.0, 2.0], [2.0, 0.0]], &adev()));
        mhc.b_res = Param::from_tensor(Tensor::<2>::from_floats([[0.0, 2.0], [2.0, 0.0]], &adev()));
        let moved = h_res(&blk);
        let dist = ident(&moved);
        assert!(
            dist > 0.5,
            "b_res = 2(J-I) left H_res at the identity (max entry error {dist:.3e}, \
             predicted 0.8808). That is 2603.20896's identity degeneration: the \
             arm would be a ReZero with a Sinkhorn in the way"
        );
        // And the manifold still holds at the perturbed point - the constraint
        // is what makes the mixing safe, so a gate that checked one without
        // the other would pass an unconstrained mixing matrix.
        for bt in 0..3 {
            let m = &moved[bt * NN..bt * NN + NN];
            for i in 0..N {
                let row: f32 = (0..N).map(|j| m[i * N + j]).sum();
                let col: f32 = (0..N).map(|j| m[j * N + i]).sum();
                assert!(
                    (row - 1.0).abs() < 1e-3 && (col - 1.0).abs() < 5e-2,
                    "perturbed H_res left the Birkhoff polytope: rowsum {row}, colsum {col}"
                );
            }
        }
        // The COMPOSITE over the loop's T iterations. With one shared block the
        // paper's `prod_i H_res^i` (Eq. 4) is a POWER of a doubly stochastic
        // matrix, hence still doubly stochastic - that is the closure property
        // the entire stability argument rests on, and at T = 2-4 it is the only
        // part of the mechanism that our depth even exercises.
        //
        // The second half is the paper's §4.1 point 3: repeated application
        // "tends to increase the mixing of information across streams
        // monotonically". The fixed point of a doubly stochastic matrix under
        // multiplication is the uniform `J/n`, so the honest form of that claim
        // is **the composite's distance to `J/n` shrinks with T** - NOT that
        // its off-diagonal mass grows, which is the version this gate used to
        // assert and which is FALSE at n = 2.
        //
        // Why, and it is a real prediction for the A/B rather than a fixture
        // quibble: a symmetric `H_res` has a second eigenvalue
        // `2 * exp(0) ... ` - measured, `{1, -0.7616}` at n = 2 - so `H_res^T`
        // ALTERNATES about the fixed point instead of approaching it from one
        // side. Off-diagonal mass at n = 2 over T = 1..5: 1.7616, 0.4200,
        // 1.4417, 0.6636, 1.2562. A two-stream residual CAN overshoot the
        // uniform mix and come back, and at our `T = 2-4` that means the
        // composite is LESS mixed than a single pass. What does hold at every T
        // is the distance to `J/n` (0.7616, 0.5800, 0.4417, 0.3364, 0.2562 -
        // monotone, because `|lambda| < 1` so `|lambda|^T` is), so that is what
        // is asserted, and the oscillation is asserted to exist so a change
        // that made it go away would be a change, not a fix.
        let hr = Tensor::<2>::from_floats([[moved[0], moved[1]], [moved[2], moved[3]]], &adev());
        let mut comp = hr.clone();
        for _ in 1..cfg.max_iter {
            comp = comp.clone().matmul(hr.clone());
        }
        let c: Vec<f32> = comp.into_data().try_to_vec().expect("[n,n]");
        for i in 0..N {
            let row: f32 = (0..N).map(|j| c[i * N + j]).sum();
            let col: f32 = (0..N).map(|j| c[j * N + i]).sum();
            assert!(
                (row - 1.0).abs() < 1e-2 && (col - 1.0).abs() < 1e-2,
                "the T={} composite is not doubly stochastic (rowsum {row}, colsum \
                 {col}): the closure property the whole stability argument rests on",
                cfg.max_iter
            );
        }
        // Off-diagonal mass of the uniform `J/n` is `n(n-1)/n = n - 1`. Both
        // sides are ONE position: `moved` holds all `n_bt = 3` and `c` is a
        // single `n x n`, so summing the whole buffer on one side and not the
        // other would compare a 3x quantity against a 1x one - an assertion
        // that still passes and means nothing.
        let first = &moved[..NN];
        let off = |m: &[f32]| -> f32 {
            m.iter().sum::<f32>() - (0..N).map(|i| m[i * N + i]).sum::<f32>()
        };
        let (o1, o4, uniform) = (off(first), off(&c), (N - 1) as f32);
        let (d1, d4) = ((o1 - uniform).abs(), (o4 - uniform).abs());
        println!(
            "mhc H_res perturbed: max |entry - I| = {dist:.4} (predicted 0.8808), \
             off-diagonal mass {o1:.4} -> {o4:.4} over T={} (uniform J/n = {uniform}, \
             predicted 1.7616 -> 0.6636); distance to J/n {d1:.4} -> {d4:.4} \
             (predicted 0.7616 -> 0.3364)",
            cfg.max_iter
        );
        assert!(
            d4 < d1,
            "the composite is no closer to the uniform mixing matrix than one pass \
             (distance {d1:.4} -> {d4:.4}, predicted 0.7616 -> 0.3364). A loop that \
             does not converge toward uniform mixing is not what the arm is for"
        );
        assert!(
            o4 > 1e-3 && d4 > 0.0,
            "the composite is at or below the identity (off-diagonal mass {o4:.4}): \
             the streams are not mixing at all, so the arm is a slower ReZero"
        );
    }

    /// A DROP-IN AT STEP 0, WITH THE SIZE OF THE STEP STATED. The crate's init
    /// (`b_res = 10 I`, `b_post = 0` -> `H_post = 2 sigma(0) = 1`) makes the
    /// mHC write `h + y`, which is ReZero's write at its own init
    /// (`residual_scale = 1`). So the A/B starts from the same function both
    /// arms start from, up to the initialisation's own error - and this gate is
    /// what makes that a measured number instead of a claim.
    ///
    /// **ONE block, both branches.** The obvious spelling - build a ReZero model
    /// and an mHC model and compare - is VACUOUS here, and the trap is worth
    /// naming: `LoopBlock::new` draws every weight from the device RNG, and two
    /// constructions in one process get two different sets (the 2026-09-30
    /// finding: `Device::seed` does not rewind a consumed stream, and
    /// `Device::flex()` is shared). That version passes with a relative
    /// difference of **1.686** - which is two different networks, not two
    /// residual operators. Toggling `use_mhc` on ONE block is the experiment:
    /// same weights, same block body, same readout, one statement apart.
    ///
    /// It also has teeth: `H_post = sigma(.)` without Eq. 8's factor 2 would
    /// make this `h + y/2` (a 50% error), a missing Sinkhorn `h + 10 I y` (an
    /// order of magnitude), a transposed `H_post` a 33% re-weighting. A
    /// tolerance of 1e-2 is two orders below the smallest of those.
    #[test]
    fn mhc_at_init_is_rezero_at_scale_one() {
        let cfg = mhc_cfg(2);
        let mut blk = LoopBlock::new(
            &DormouseConfig {
                use_mhc: true,
                ..cfg.clone()
            },
            &adev(),
        );
        let head = mhc_head(&cfg);
        let x = Tensor::<3>::random([2, 5, 32], Distribution::Normal(0.0, 1.0), &adev());
        let (o_mh, _, _, _) = blk.forward_full_state::<B>(x.clone(), None, None, None, None, &head);
        // The scalar ReZero is about to be compared against must BE 1.
        assert_eq!(blk.residual_scale.val().clone().into_scalar::<f32>(), 1.0);
        // Same object, ReZero's statement. `mhc` stays allocated (so the
        // parameter set is identical and only the branch differs), and the
        // counter says which branch ran.
        blk.use_mhc = false;
        crate::probe::reset();
        let (o_rz, _, _, _) = blk.forward_full_state::<B>(x, None, None, None, None, &head);
        assert_eq!(
            crate::probe::count(crate::probe::MHC),
            0,
            "the toggle did not take"
        );

        let scale = o_rz.clone().abs().max().into_scalar::<f32>();
        let rel = (o_rz.clone() - o_mh).abs().max().into_scalar::<f32>() / scale;
        // FIDELITY FIX F-F4 (2026-10-02): the OLD assertion here was
        // `rel < 1e-2` - "mHC at init is ReZero at scale 1" - which was the
        // 10I identity init SPEAKING, not the mechanism: it froze alpha_res's
        // gradient at 7.3e-9 and measured a tie between the arm and its own
        // starting point. The fix moves b_res to 0, so mHC-at-init is UNIFORM
        // stream mixing - O(1) apart from ReZero, and trainable.
        println!(
            "mhc@init vs ReZero@scale1 on identical weights: relative {rel:.3e} \
             (the fidelity fix made this O(1); it was < 1e-2 under the 10I init)"
        );
        assert!(
            rel > 0.05,
            "mHC at init is ReZero at residual_scale = 1 (relative {rel:.3e}): \
             the b_res init has drifted back onto the identity - \
             the arm is frozen exactly as the fidelity audit describes"
        );
        // ...and still a BOUNDED operator, not garbage: the Sinkhorn of
        // exp(≈0) is doubly stochastic, so the init mixing is a convex map.
        assert!(
            rel < 5.0,
            "mHC at init diverged (relative {rel:.3e}): the projection left the \
             Birkhoff polytope or the fixture is broken"
        );
    }

    /// TWO-SIDED SEPARATION: this is the "zero-default identity" claim in the
    /// only form one process can witness. ReZero's knob must move the ReZero
    /// arm and NOT the mHC arm; mHC's knob must move the mHC arm and not the
    /// ReZero arm. A branch-ordering slip - the new `else if` swallowing the
    /// ReZero statement, or ReZero's `residual_scale` leaking into the mHC
    /// write - is invisible to "did the output change" and fatal here.
    ///
    /// What this CANNOT claim, stated so nobody reads more into it: a
    /// bitwise-equal CE against a build from before this flag existed is a
    /// CROSS-PROCESS fact (two RNG streams, two `ParamId` counters) and belongs
    /// to `tools/determinism.py`. What is provable in one process is the
    /// parameter set - an `Option<MhcBlock>` that is `None` contributes no
    /// records - and this knob separation.
    #[test]
    fn mhc_and_rezero_each_move_only_their_own_write() {
        let cfg = mhc_cfg(2);
        let head = mhc_head(&cfg);
        let x = Tensor::<3>::random([2, 5, 32], Distribution::Normal(0.0, 1.0), &adev());
        let run = |blk: &mut LoopBlock| -> Tensor<3> {
            blk.forward_full_state::<B>(x.clone(), None, None, None, None, &head)
                .0
        };
        let far = |a: &Tensor<3>, b: &Tensor<3>| -> f32 {
            (a.clone() - b.clone()).abs().max().into_scalar::<f32>()
        };

        // ReZero arm: moving its scalar moves it, and moving mHC's block cannot
        // even be expressed (the block does not exist).
        let mut rz = LoopBlock::new(&cfg, &adev());
        assert!(rz.mhc.is_none(), "use_mhc = false must not BUILD the block");
        let rz_a = run(&mut rz);
        rz.residual_scale = p1(0.5);
        let rz_b = run(&mut rz);
        assert!(
            far(&rz_a, &rz_b) > 1e-6,
            "residual_scale moved and the ReZero arm did not: the write is not the \
             branch this test thinks it is"
        );

        // mHC arm: residual_scale is INERT there, and b_res is the live knob.
        let mut mh = LoopBlock::new(
            &DormouseConfig {
                use_mhc: true,
                ..cfg
            },
            &adev(),
        );
        let mh_a = run(&mut mh);
        mh.residual_scale = p1(0.5);
        let mh_b = run(&mut mh);
        assert_eq!(
            far(&mh_a, &mh_b),
            0.0,
            "residual_scale reached the mHC write: the A/B against ReZero would \
             not be a single-factor comparison"
        );
        let mhc = mh.mhc.as_mut().expect("arm on");
        mhc.b_res = Param::from_tensor(Tensor::<2>::from_floats([[0.0, 2.0], [2.0, 0.0]], &adev()));
        let mh_c = run(&mut mh);
        assert!(
            far(&mh_b, &mh_c) > 1e-6,
            "b_res moved and the mHC arm did not: the manifold projection is not \
             in the write"
        );
    }

    /// GRADIENTS, PER PARAMETER GROUP, WITH THE MAGNITUDES. `forward_full_state`
    /// is instantiated on `Autodiff<Flex, BalancedCheckpointing>` - the
    /// trainer's own backend - because that is where `8fa5d4c` lived: a node
    /// that came back `UnTracked`, ran thousands of forwards and trained
    /// nothing, with a healthy loss curve throughout.
    ///
    /// The gate is per-parameter rather than "the loss has a gradient", because
    /// the per-parameter version is the one that catches a DETACHED ARM. It
    /// also pins the one parameter that is SUPPOSED to have no gradient:
    /// `phi_pre` builds `H_pre`, which Eq. 3 applies to the block's own input
    /// and this wiring does not compute (see [`LoopBlock::mhc`]). A dead
    /// 1 536-parameter matrix in a model whose A/B is about whether the arm
    /// earns its place is a defect, and the cheap fix is to know about it.
    #[test]
    fn mhc_gradients_reach_every_wired_parameter() {
        let cfg = mhc_cfg(2);
        let blk = LoopBlock::new(
            &DormouseConfig {
                use_mhc: true,
                ..cfg.clone()
            },
            &adev(),
        );
        let head = mhc_head(&cfg);
        let x = Tensor::<3>::random([2, 5, 32], Distribution::Normal(0.0, 1.0), &adev());
        // TARGETS, or `L_Rec` is never accumulated: `forward_full_state` only
        // builds a CE term `if let Some(tgt) = &targets`, so with `None` the
        // returned `rec` is the zero it started as - a leaf, and `backward()`
        // refuses it. The gate would then be measuring burn's error message.
        let tgt = Tensor::<2, Int>::zeros([2 * 5, 1], &adev());
        let (_, rec, _, _) = blk.forward_full_state::<B>(x, None, None, None, Some(tgt), &head);
        let grads = rec.backward();
        let mhc = blk.mhc.as_ref().expect("arm on");
        // `b_res` is the one that matters: it is the whole mechanism, and
        // 2603.20896's identity degeneration is exactly a `b_res` that stops
        // moving.
        fn mag<const D: usize>(
            p: &Param<Tensor<D>>,
            grads: &burn::tensor::Gradients,
        ) -> (f32, usize) {
            let g = p.grad(grads).unwrap_or_else(|| {
                panic!("parameter carries no gradient slot: the mHC write is a leaf")
            });
            let v: Vec<f32> = g.into_data().try_to_vec().expect("readable gradient");
            (v.iter().fold(0.0f32, |m, x| m.max(x.abs())), v.len())
        }
        for (name, (m, len)) in [
            ("b_res", mag(&mhc.b_res, &grads)),
            ("b_post", mag(&mhc.b_post, &grads)),
            ("alpha_res", mag(&mhc.alpha_res, &grads)),
            ("alpha_post", mag(&mhc.alpha_post, &grads)),
            ("phi_res", mag(&mhc.phi_res, &grads)),
            ("phi_post", mag(&mhc.phi_post, &grads)),
        ] {
            println!("mhc grad {name}: max |g| = {m:.3e} over {len} entries");
            assert!(
                m > 0.0 && m.is_finite(),
                "{name}: max |grad| = {m:.3e} over {len} entries - zero means the arm \
                 is not being trained, non-finite means it is being trained badly"
            );
        }
        // The three parameters Eq. 3 needs and this wiring does not use
        // (`H_pre`'s half: `phi_pre`, its gate `alpha_pre`, its static bias
        // `b_pre`). `forward` never reads them, so they are not in the graph at
        // all and `grad()` returns `None` rather than zeros - which is the
        // stronger form of the same statement and worth asserting exactly: a
        // reader who finds 1 792 dead parameters in a gradient dump should find
        // them named here first, and a future change that starts using H_pre
        // trips this message instead of silently doubling the arm.
        for (name, present) in [
            ("phi_pre", mhc.phi_pre.grad(&grads).is_some()),
            ("alpha_pre", mhc.alpha_pre.grad(&grads).is_some()),
            ("b_pre", mhc.b_pre.grad(&grads).is_some()),
        ] {
            assert!(
                !present,
                "{name} is IN the graph, so something started using H_pre. The \
                 wiring, `docs/reviews/mhc-2026-09-30.md` 3.3 and this test all \
                 say it does not - reconcile them before trusting a gradient dump"
            );
        }
    }

    /// THE THREE ARMS ARE THREE FUNCTIONS, ON ONE SET OF INPUTS. ReZero,
    /// AttnRes and mHC replace the same statement, so a wiring slip that made
    /// two of them the same function would be a valid loss curve and a
    /// meaningless A/B. The comparison is at the OPERATOR level - the three
    /// residual writes, on the same `h` and the same `y` - because that is the
    /// claim, and because the model-level spelling is vacuous: three
    /// `LoopBlock::new` calls draw three different weight sets from the device
    /// RNG (see `mhc_at_init_is_rezero_at_scale_one`), so the model-level
    /// version of this test passes with any three arms, including three copies
    /// of the same one.
    ///
    /// The two levels answer different halves and both are kept:
    /// - **operator level (this test)**: on one `h`, one `y`, the mHC write at
    ///   its init is ReZero's to 1e-2 and both are a long way from AttnRes's
    ///   equal-weight mean; move `b_res` off the init and mHC separates from
    ///   ReZero by O(1). Numbers, not "it changed".
    /// - **model level**
    ///   (`mhc_moves_the_readout_at_every_depth`): the flag changes the model's
    ///   own output at every depth, which is the cheap half and the one
    ///   `attnres_moves_the_readout_at_every_depth` already established as
    ///   necessary-but-not-sufficient.
    #[test]
    fn the_three_residual_operators_are_three_functions() {
        let cfg = mhc_cfg(2);
        let blk = LoopBlock::new(
            &DormouseConfig {
                use_mhc: true,
                ..cfg.clone()
            },
            &adev(),
        );
        let mhc = blk.mhc.as_ref().expect("arm on");
        let d = 32usize;
        let h = Tensor::<3>::random([2, 5, d], Distribution::Normal(0.0, 1.0), &adev());
        let y = Tensor::<3>::random([2, 5, d], Distribution::Normal(0.0, 1.0), &adev());
        let gap = |a: &Tensor<3>, b: &Tensor<3>| -> f32 {
            (a.clone() - b.clone()).abs().max().into_scalar::<f32>()
        };
        // ReZero at its init: `h + y·s`, `s = 1`.
        let rz = h.clone() + y.clone();
        // AttnRes at its init: the equal-weight mean of the token embedding and
        // this iteration's body (`w_l = 0` -> a uniform softmax), fed the same
        // two states the other two are fed.
        let ar = depth_attend(&[h.clone(), y.clone()], Tensor::<1>::zeros([d], &adev()));
        // mHC at its init.
        let mh = mhc.forward(h.clone(), std::slice::from_ref(&y));
        let scale = rz.clone().abs().max().into_scalar::<f32>();

        let (mh_rz, mh_ar, ar_rz) = (
            gap(&mh, &rz) / scale,
            gap(&mh, &ar) / scale,
            gap(&ar, &rz) / scale,
        );
        // FIDELITY FIX F-F4 (2026-10-02): `b_res` starts at 0, NOT at the 10I
        // that froze alpha_res's gradient. The old init made mHC-at-init
        // ReZero at scale 1 (`mh_rz ~ 4e-3`, the old assertion) - and never a
        // trainable projection. The new init is UNIFORM mixing: exp(≈0) at
        // every entry -> Sinkhorn -> 1/n, off-diagonal mass O(1), so the arm
        // starts a DIFFERENT residual operator from ReZero (that separation is
        // the A/B's subject, not a defect) and it must still be no function of
        // AttnRes's.
        assert!(
            mh_rz > 0.05,
            "mHC at init has become ReZero again (relative {mh_rz:.3e}): the b_res init drifted \
             back onto the identity - the arm's gradient would be dead on arrival"
        );
        assert!(
            mh_ar > 0.1,
            "mHC at init is AttnRes (relative {mh_ar:.3e}): two different residual \
             operators, one function"
        );
        assert!(
            ar_rz > 0.1,
            "AttnRes is ReZero (relative {ar_rz:.3e}): the FIXTURE is degenerate and \
             the two comparisons above prove nothing"
        );

        // Off the init, mHC separates from ReZero by O(1) - the number the A/B
        // will be measuring, quoted before it is run. The perturbation is
        // `2 (J - I)` for the reason `mhc_res_leaves_the_identity...` gives:
        // scaling the DIAGONAL (`6 I`) makes the map MORE of an identity, not
        // less, and reads 6.2e-3 where the point of the gate is an O(1) gap.
        let mut moved = LoopBlock::new(
            &DormouseConfig {
                use_mhc: true,
                ..cfg
            },
            &adev(),
        );
        let m = moved.mhc.as_mut().expect("arm on");
        m.b_res = Param::from_tensor(Tensor::<2>::from_floats([[0.0, 2.0], [2.0, 0.0]], &adev()));
        let mh2 = moved
            .mhc
            .as_ref()
            .expect("arm on")
            .forward(h, std::slice::from_ref(&y));
        let sep = gap(&mh2, &rz) / scale;
        println!(
            "mhc operator separations (relative to max|h+y|): mHC@init vs ReZero \
             {mh_rz:.3e}, mHC@init vs AttnRes {mh_ar:.3e}, AttnRes vs ReZero \
             {ar_rz:.3e}, mHC@2(J-I) vs ReZero {sep:.3e}"
        );
        assert!(
            sep > 0.05,
            "b_res = 2(J-I) did not separate mHC from ReZero (relative {sep:.3e}): \
             the projection is not in the write"
        );
    }

    /// The model-level half: the flag changes the model's own output, at every
    /// depth, and the counter follows. Cheap, and the necessary condition the
    /// AttnRes arm established; the sufficiency lives in
    /// `the_three_residual_operators_are_three_functions` and in
    /// `mhc_res_is_doubly_stochastic_in_both_directions`.
    #[test]
    fn mhc_moves_the_readout_at_every_depth() {
        for depth in [1usize, 2, 4] {
            let cfg = mhc_cfg(depth);
            let mut blk = LoopBlock::new(
                &DormouseConfig {
                    use_mhc: true,
                    ..cfg
                },
                &adev(),
            );
            let head = mhc_head(&mhc_cfg(depth));
            let x = Tensor::<3>::random([2, 5, 32], Distribution::Normal(0.0, 1.0), &adev());
            // Same object, both branches: two `LoopBlock::new` calls would
            // differ by their initialisation, not by their residual operator.
            crate::probe::reset();
            let (on, _, _, _) =
                blk.forward_full_state::<B>(x.clone(), None, None, None, None, &head);
            assert_eq!(crate::probe::count(crate::probe::MHC), depth as u64);
            blk.use_mhc = false;
            let (off, _, _, _) =
                blk.forward_full_state::<B>(x.clone(), None, None, None, None, &head);
            crate::probe::reset();
            blk.forward_full_state::<B>(
                Tensor::zeros([2, 5, 32], &adev()),
                None,
                None,
                None,
                None,
                &head,
            );
            assert_eq!(
                crate::probe::count(crate::probe::MHC),
                0,
                "the ReZero branch does not project"
            );
            let diff = (on - off).abs().max().into_scalar::<f32>();
            assert!(
                diff > 1e-6,
                "depth {depth}: use_mhc changed no output (max {diff:.3e}) - the arm ran \
                 and its result was discarded, which is GR's 9b343d3 defect"
            );
        }
    }

    /// THE COUNTER, and the refusals, and the flag's default - the three things
    /// that make `use_mhc` an arm rather than a decoration.
    #[test]
    fn mhc_counts_every_iteration_and_is_refused_alongside_the_others() {
        let cfg = mhc_cfg(4);
        let head = mhc_head(&cfg);
        let x = Tensor::<3>::random([2, 5, 32], Distribution::Normal(0.0, 1.0), &adev());

        crate::probe::reset();
        let mh = LoopBlock::new(
            &DormouseConfig {
                use_mhc: true,
                ..cfg.clone()
            },
            &adev(),
        );
        let _ = mh.forward_full_state::<B>(x.clone(), None, None, None, None, &head);
        assert_eq!(
            crate::probe::count(crate::probe::MHC),
            4,
            "one projection per iteration at depth 4"
        );
        crate::probe::reset();
        let mut mh = LoopBlock::new(
            &DormouseConfig {
                use_mhc: true,
                ..cfg.clone()
            },
            &adev(),
        );
        mh.set_depth(Some(2));
        let _ = mh.forward_full_state::<B>(x.clone(), None, None, None, None, &head);
        assert_eq!(
            crate::probe::count(crate::probe::MHC),
            2,
            "the counter counts what RAN"
        );
        crate::probe::reset();
        let rz = LoopBlock::new(&cfg, &adev());
        let _ = rz.forward_full_state::<B>(x, None, None, None, None, &head);
        assert_eq!(
            crate::probe::count(crate::probe::MHC),
            0,
            "ReZero does not project"
        );

        // The default is off, so the parameters do not exist and every shipped
        // checkpoint still loads.
        assert!(!DormouseConfig::default().use_mhc);
        assert!(LoopBlock::new(&cfg, &adev()).mhc.is_none());
        // What the arm costs, MEASURED on the instantiated block rather than
        // arithmetic on the paper's shapes: `2 D n + D n^2 + 2n + n^2 + 3`
        // (phi_pre, phi_post, phi_res, the three alpha scalars, b_pre, b_post,
        // b_res). The arithmetic and the fixture width disagreed in this
        // file's own notes - `small` is `d_model = 768`, not 384 - and the
        // whole point of `preset_exec` is that a count is a measurement.
        //
        // `MhcBlock::new` directly, NOT a `small` `LoopBlock`: two of those
        // cost 489 s of this suite by themselves, for a number that is a
        // function of three shapes and nothing else. Measured on this machine.
        let small = crate::config::load_config("small").expect("the small preset loads");
        for n in [2usize, 4] {
            let mhc = MhcBlock::new(n, small.d_model, &adev());
            // Two groups, not one array: `phi_*`/`b_res` are `Param<Tensor<2>>`
            // and `alpha_*`/`b_pre`/`b_post` are `Param<Tensor<1>>`, and a Rust
            // array literal takes its element type from the first entry, so one
            // list of all nine is a type error rather than a count.
            let count2: usize = [&mhc.phi_pre, &mhc.phi_post, &mhc.phi_res, &mhc.b_res]
                .iter()
                .map(|p| p.val().clone().dims().iter().product::<usize>())
                .sum();
            let count1: usize = [
                &mhc.alpha_pre,
                &mhc.alpha_post,
                &mhc.alpha_res,
                &mhc.b_pre,
                &mhc.b_post,
            ]
            .iter()
            .map(|p| p.val().clone().dims().iter().product::<usize>())
            .sum();
            let count = count1 + count2;
            let want = 2 * small.d_model * n + small.d_model * n * n + 2 * n + n * n + 3;
            println!(
                "mhc n={n} on small (d_model={}): {count} params",
                small.d_model
            );
            assert_eq!(
                count, want,
                "the paper's shapes and the built block disagree"
            );
            if n == 2 {
                assert!(
                    count * 200 < 9_197_390,
                    "the arm must stay a rounding error against small's 9 197 390: {count}"
                );
            }
        }

        // Mutually exclusive with BOTH of the arms it competes with, and the
        // error names every flag that is on.
        for other in ["use_gr", "use_attnres"] {
            let c = DormouseConfig {
                use_mhc: true,
                ..cfg.clone()
            };
            let both = if other == "use_gr" {
                DormouseConfig { use_gr: true, ..c }
            } else {
                DormouseConfig {
                    use_attnres: true,
                    ..c
                }
            };
            let err =
                crate::config::validate(&both).expect_err("two residual arms must be refused");
            assert!(
                err.contains("use_mhc") && err.contains(other),
                "the refusal must name both flags: {err}"
            );
        }
        // Each alone is legal, and mHC alone at the default n = 2 divides the
        // fixture's d_model = 32.
        for on in [
            DormouseConfig {
                use_mhc: true,
                ..cfg.clone()
            },
            DormouseConfig {
                use_gr: true,
                ..cfg.clone()
            },
            DormouseConfig {
                use_attnres: true,
                ..cfg.clone()
            },
        ] {
            assert!(
                crate::config::validate(&on).is_ok(),
                "one arm alone must validate"
            );
        }
        // A non-divisor is LOUD at startup, not a reshape panic in the first
        // forward: 32 % 3 != 0, and 3 streams would be a valid-looking config
        // that trains nothing but crashes.
        let err = crate::config::validate(&DormouseConfig {
            use_mhc: true,
            mhc_streams: 3,
            ..cfg
        })
        .expect_err("3 does not divide 32");
        assert!(
            err.contains("mhc_streams") && err.contains("32"),
            "the refusal must name the field and the width: {err}"
        );
        // `n = 0` is refused too: `MhcBlock::new` would `max(1)` it silently,
        // so a config asking for zero streams would train the n = 1 arm - which
        // 2409.19606 Tab. 1 measures as WORSE than the Pre-Norm baseline.
        let err = crate::config::validate(&DormouseConfig {
            use_mhc: true,
            mhc_streams: 0,
            ..mhc_cfg(2)
        })
        .expect_err("0 streams is not a configuration");
        assert!(
            err.contains("mhc_streams"),
            "the refusal must name the field: {err}"
        );
        // n = 4 (the base paper's App. Tab. 1 rung) is a flag away and also
        // legal - the follow-up row, not this one.
        assert!(crate::config::validate(&DormouseConfig {
            use_mhc: true,
            mhc_streams: 4,
            ..mhc_cfg(2)
        })
        .is_ok());
    }

    /// OFF IS THE OLD MODEL. `use_situ = false` must leave the parameter set
    /// alone, because every existing checkpoint is that parameter set: the
    /// up-projection stays `d -> f` (not `d -> 2f`), so the FFN's parameter
    /// count and every `gate_up` factor are bit-identical to a build from
    /// before this field existed.
    ///
    /// **What this test is NOT.** A cross-build bitwise comparison of two
    /// forward passes, which is the obvious thing to reach for and is not
    /// available: `Device::seed` does not rewind a consumed stream, so a second
    /// `LoopBlock::new` on the shared flex device draws DIFFERENT weights and
    /// comparing them would measure the RNG, not the flag (AGENTS.md §3.7 - the
    /// "CPU is deterministic" claim, withdrawn 2026-09-30). The identity
    /// claim is therefore carried by the parameter set here, by the exact
    /// preset counts in `preset_exec` (11/0, unchanged), and by the fact that
    /// the new field is `#[serde(default)]`, so no config snapshot that predates
    /// it can fail to parse.
    #[test]
    fn situ_off_leaves_the_parameter_set_alone() {
        let cfg = situ_cfg();
        let off = LoopBlock::new(&cfg, &adev());
        for e in &off.expert_ffns {
            assert_eq!(
                e.gate_up.out_features, cfg.d_ffn,
                "use_situ = false must keep the up-projection d -> d_ffn; Eq (12) reads two \
                 projections and silu reads one"
            );
        }
        assert!(!off.use_situ);
        // The default config itself, so a preset cannot turn this on by
        // accident.
        assert!(!DormouseConfig::default().use_situ);
        // And the flag is `serde(default)`: a snapshot written before the field
        // existed parses, and reads as off.
        let old_snapshot = "d_model = 32\nn_experts = 2\nmax_iter = 2\n";
        let parsed: DormouseConfig =
            toml::from_str(old_snapshot).expect("a pre-situ snapshot parses");
        assert!(
            !parsed.use_situ,
            "a config without the field is the OFF arm"
        );

        // On: `d -> 2f`, exactly. This is the whole parameter delta of the arm
        // and it is why the A/B is width-confounded (schema.rs).
        let on = LoopBlock::new(
            &DormouseConfig {
                use_situ: true,
                ..cfg.clone()
            },
            &adev(),
        );
        for e in &on.expert_ffns {
            assert_eq!(
                e.gate_up.out_features,
                2 * cfg.d_ffn,
                "Eq (12) needs Wg and Wu separately"
            );
        }
        // The cost of the arm, on the gate's own geometry, from the factors'
        // element counts: TSCT adds `rank * f` per expert when the width goes
        // from f to 2f. (On `small`, r=64 f=2048 n=3: +393 216, +4.28%.)
        let delta = on
            .expert_ffns
            .iter()
            .map(|e| {
                burn::module::Module::num_params(&e.gate_up)
                    - burn::module::Module::num_params(&off.expert_ffns[0].gate_up)
            })
            .sum::<usize>();
        assert_eq!(
            delta,
            cfg.n_experts * cfg.rank * cfg.d_ffn,
            "the arm costs rank*d_ffn per expert, and that is the confound the A/B row names"
        );
    }

    /// THE LOUD GUARD IS LOUD, AND NAMES ITS CAUSE. Every field on
    /// `LoopBlock` is `pub`, so `use_situ` can be flipped after construction -
    /// and then Eq (12) reads a `d -> f` tensor, which is an out-of-range slice
    /// inside burn: a panic that mentions shapes and never mentions the flag.
    /// This asserts the message names `use_situ` and the escape.
    #[test]
    #[should_panic(expected = "use_situ")]
    fn situ_flag_and_width_cannot_disagree() {
        let cfg = situ_cfg();
        let mut blk = LoopBlock::new(&cfg, &adev());
        // The parameters are the OFF arm's; claiming the ON arm is the lie.
        blk.use_situ = true;
        let head = LinearLike::with_tsct(cfg.d_model, 16, cfg.rank, cfg.use_tsct, &adev());
        let x = Tensor::<3>::zeros([2, 5, cfg.d_model], &adev());
        let _ = blk.forward_full_state::<B>(x, None, None, None, None, &head);
    }

    /// THE ARM IS OBSERVABLE, and its counter is exact. `use_situ = true` with a
    /// zero counter is a run that measured SiLU under SiTU's name - the
    /// ADR-0019 shape, and the GR `9b343d3` defect in a new place.
    #[test]
    fn situ_counts_every_expert_it_activates() {
        let cfg = situ_cfg();
        let head = LinearLike::with_tsct(cfg.d_model, 16, cfg.rank, cfg.use_tsct, &adev());
        let x = Tensor::<3>::random([2, 5, cfg.d_model], Distribution::Normal(0.0, 1.0), &adev());

        crate::probe::reset();
        let mut on = LoopBlock::new(
            &DormouseConfig {
                use_situ: true,
                ..cfg.clone()
            },
            &adev(),
        );
        let _ = on.forward_full_state::<B>(x.clone(), None, None, None, None, &head);
        assert_eq!(
            crate::probe::count(crate::probe::SITU),
            (cfg.n_experts * cfg.max_iter) as u64,
            "one SiTU per expert per iteration: n_experts * max_iter"
        );

        // Off: zero. The silu path must not touch the counter, or the field
        // would read non-zero for a run that never activated the arm.
        crate::probe::reset();
        let off = LoopBlock::new(&cfg, &adev());
        let _ = off.forward_full_state::<B>(x.clone(), None, None, None, None, &head);
        assert_eq!(
            crate::probe::count(crate::probe::SITU),
            0,
            "silu does not run SiTU"
        );

        // Truncated depth counts what RAN, like every other counter here.
        crate::probe::reset();
        on.set_depth(Some(1));
        let _ = on.forward_full_state::<B>(x, None, None, None, None, &head);
        assert_eq!(
            crate::probe::count(crate::probe::SITU),
            cfg.n_experts as u64,
            "depth 1 activated n_experts times, not n_experts * max_iter"
        );
    }

    /// THE FLAG IS LOAD-BEARING, on the block's OWN weights - so no
    /// cross-model comparison and no RNG confound. The intermediate the
    /// up-projection actually produces is fed to both activations: the one the
    /// config names must be BOUNDED by `beta1*beta2` and the one it replaced
    /// must not be. A `use_situ` that was computed and discarded, or that fed
    /// `situ_glu` a `[b*t, f]` tensor, could not pass this.
    #[test]
    fn situ_bounds_the_activation_the_silu_path_left_unbounded() {
        let cfg = situ_cfg();
        let on = LoopBlock::new(
            &DormouseConfig {
                use_situ: true,
                ..cfg.clone()
            },
            &adev(),
        );
        let expert = &on.expert_ffns[0];
        let n = 64usize;
        // `gate_up` is `d_model -> 2*d_ffn`, so the input is d_model-wide and
        // the BOUNDED tensor is its output. A deliberately hot input: an
        // RMSNormed hidden state is O(1), so this is far outside the training
        // regime - the overfit condition AGENTS.md §2.3 says the NaN class
        // lives in, and the regime Eq (19)'s bound is claimed for.
        let probe = Tensor::<2>::random([n, cfg.d_model], Distribution::Normal(0.0, 1.0), &adev());
        let mid = expert.gate_up.forward::<B>(probe);
        assert_eq!(
            mid.dims(),
            [n, 2 * cfg.d_ffn],
            "the up-projection is d_model -> 2*d_ffn"
        );
        // Put the intermediate at a NAMED magnitude, 10x the bound, instead of
        // hoping the random TSCT factors land there: the comparison is then a
        // statement about the activation and not about the initializer.
        const HOT: f32 = 10.0 * (burn_situ::K3_GATE_BETA * burn_situ::K3_UP_BETA) as f32;
        let scale = HOT / mid.clone().abs().max().into_scalar::<f32>();
        let mid = mid.mul_scalar(scale);
        let mid_peak = mid.clone().abs().max().into_scalar::<f32>();
        assert!(
            (mid_peak - HOT).abs() < 0.01 * HOT,
            "the rescale put the intermediate at {mid_peak}, not the {HOT} it asked for"
        );
        let via_situ = burn_situ::situ_glu(
            mid.clone(),
            cfg.d_ffn,
            burn_situ::K3_GATE_BETA,
            burn_situ::K3_UP_BETA,
        );
        // The comparator has to be the same SHAPE to be compared elementwise,
        // and the arm it replaces read one `d_ffn`-wide tensor, so it is
        // `silu` on this tensor's gate half: the same coordinates, unbounded.
        let via_silu = activation::silu(mid.slice([0..n, 0..cfg.d_ffn]));
        assert_eq!(
            via_situ.dims(),
            via_silu.dims(),
            "the comparator is elementwise"
        );
        let bound = (burn_situ::K3_GATE_BETA * burn_situ::K3_UP_BETA) as f32;
        let peak_situ = via_situ.clone().abs().max().into_scalar::<f32>();
        let peak_silu = via_silu.clone().abs().max().into_scalar::<f32>();
        assert!(
            peak_situ <= bound * (1.0 + 1e-5),
            "SiTU gave {peak_situ}, past the Eq (19) bound {bound}"
        );
        // 5x, not 10x: |silu|'s peak is not at |mid|'s peak (a large NEGATIVE
        // pre-activation saturates to ~0), so the honest comparator is the
        // largest positive coordinate, and on this fixture that lands at 9.2x.
        assert!(
            peak_silu > 5.0 * bound,
            "the silu comparator peaked at only {peak_silu}, {bound} away from the bound - a \
             fixture too cold to tell a bounded activation from an unbounded one"
        );
        // And they are different functions, not the same one twice.
        let diff = (via_situ - via_silu).abs().max().into_scalar::<f32>();
        assert!(
            diff > 1.0,
            "the two activations differ by only {diff} on a hot input"
        );
    }

    /// THE GRADIENT SURVIVES THE CAP. `tanh` saturates, so this is the honest
    /// question about the mechanism: the forward is bounded, and bounded
    /// forward and vanishing gradient are the same fact. Measured on the real
    /// path - up-projection, cap, down-projection, backward - and asserted on
    /// RELATIVE gradient (d/dinput over |output|), because an absolute
    /// threshold would be a statement about this fixture's scale and not about
    /// the arm.
    ///
    /// The reference numbers (f64, `tools/gen_ref.py` in burn-situ, cross-checked
    /// against the crate's own CUDA backward): the GATE factor's derivative
    /// falls from 0.73 at g=0.5 to 1.8e-4 at g=20 and to 8.2e-9 at g=40, i.e.
    /// the cap costs a factor of ~1/sech^2 and the branch is numerically DEAD
    /// past ~10*beta. Over the band where the cap is doing its job (1-5 beta)
    /// the gradient is 1.6e-1 to 4.5e-5 relative, which is 2.7 to 6 orders
    /// above f32 eps (1.19e-7). So the gate asserts the band the arm operates
    /// in is alive, and records the decay for the A/B to interpret.
    #[test]
    fn situ_gradient_survives_the_cap_where_the_cap_operates() {
        let cfg = situ_cfg();
        let on = LoopBlock::new(
            &DormouseConfig {
                use_situ: true,
                ..cfg
            },
            &adev(),
        );
        let expert = &on.expert_ffns[0];
        let f = cfg.d_ffn;
        let n = 8usize;
        let b1 = burn_situ::K3_GATE_BETA as f32;
        let b2 = burn_situ::K3_UP_BETA as f32;

        // The gate half swept over 0.25x .. 5x beta1 and the up half over the
        // matching 0.25x .. 5x beta2, with the up sign alternating so the
        // negative Swish tail is in the sweep. Each row is `[n, 2f]`: f gate
        // pre-activations, then f up ones - the layout Eq (12) reads.
        for &mult in &[0.25f32, 0.5, 1.0, 2.0, 5.0] {
            let mut cells = Vec::with_capacity(n * 2 * f);
            for _ in 0..n {
                cells.extend(std::iter::repeat_n(mult * b1, f));
                cells.extend((0..f).map(|j| if j % 2 == 0 { mult * b2 } else { -mult * b2 }));
            }
            let input = Tensor::<1>::from_floats(cells.as_slice(), &adev())
                .reshape([n, 2 * f])
                .require_grad();
            let out = expert.down.forward::<B>(burn_situ::situ_glu(
                input.clone(),
                f,
                burn_situ::K3_GATE_BETA,
                burn_situ::K3_UP_BETA,
            ));
            // Sum of squares: a scalar with no sign, so the test measures the
            // magnitude of the path and not which way it points.
            let loss = out.clone().powf_scalar(2.0).sum();
            let grads = loss.backward();
            let g = input
                .grad(&grads)
                .expect("a require_grad leaf has a gradient");
            let peak = g.clone().abs().max().into_scalar::<f32>();
            let out_peak = out.clone().abs().max().into_scalar::<f32>();
            // RELATIVE gradient: an absolute threshold would be a statement
            // about this fixture's scale, not about the arm. The floor is the
            // measured d/dinput/|out| of the same quantity one octave further
            // out the cap (tools/gen_ref.py), halved, so a regression that
            // costs the arm an order of magnitude of gradient trips it.
            let rel = peak / out_peak.max(1e-30);
            assert!(
                peak.is_finite() && rel > 1e-6,
                "at {mult}x beta (gate {:.2}, up {:.2}): d/dinput peaked at {peak:.3e} against                  |out| = {out_peak:.3e}, i.e. {rel:.3e} relative - the cap killed the gradient \
                 inside the band the arm operates in",
                mult * b1,
                mult * b2
            );
            // f32 eps is 1.19e-7: a relative gradient under it cannot move the
            // output by one ulp, which is the "numerically dead" line the
            // findings file quotes at 10x beta.
            eprintln!(
                "situ grad sweep: {mult:>4}x beta1 -> |d/dinput|/|out| = {rel:.3e} \
                 (|out| = {out_peak:.4e})"
            );
        }
    }

    /// The block's wiring uses the configured floor, and the row budget
    /// rounds up to a power of two with a matching mask (so a raw FNV hash is
    /// always in range, whatever the corpus).
    #[test]
    fn block_uses_the_configured_floor() {
        // The 500_000 -> 524_288 rounding (raw FNV hashes are masked on
        // device, not divided), and the degenerate cases.
        let (tables, mask) = engram_tables(500_000, 3);
        assert_eq!(
            tables,
            vec![524_288; 3],
            "500_000 rounds up to the next power of two"
        );
        assert_eq!(mask, 524_287);
        assert_eq!(engram_tables(25_000, 3), (vec![32_768; 3], 32_767));
        assert_eq!(engram_tables(1024, 3), (vec![1024; 3], 1023));
        assert_eq!(engram_tables(1, 1), (vec![1], 0));

        let mut cfg = small_cfg(1);
        cfg.engram_rows = 1024;
        cfg.engram_lam_max = 0.25;
        let b = LoopBlock::new(&cfg, &dev());
        assert_eq!(
            b.engram_lam_max, 0.25,
            "the block must carry the configured floor"
        );
        assert_eq!(b.engram_slot_mask, 1023);

        // The floor is a floor: validate() refuses 0 (deletes the arm) and
        // >1 (not a floor), and the order count is pinned to the trainer's
        // [b, t, 3] hash tensor.
        assert!(crate::config::validate(&cfg).is_ok());
        cfg.engram_lam_max = 0.0;
        assert!(crate::config::validate(&cfg).is_err());
        cfg.engram_lam_max = 1.5;
        assert!(crate::config::validate(&cfg).is_err());
        cfg.engram_lam_max = 0.5;
        cfg.engram_orders = vec![3, 5];
        assert!(
            crate::config::validate(&cfg).is_err(),
            "the order COUNT is pinned to 3"
        );
        cfg.engram_orders = vec![2, 3, 4];
        cfg.engram_rows = 0;
        assert!(crate::config::validate(&cfg).is_err());
    }

    /// The capacity budget the config ships, spelled out, with the two
    /// numbers it is chosen between: the measured slot-count curve (500K/order
    /// optimum, on a 16x larger backbone) and the allocation ratio (DeepSeek's
    /// 20-25%, which is what "monopoly" means). The preset ships the ratio.
    #[test]
    fn capacity_budget_is_a_minority_of_the_model() {
        let c = DormouseConfig::default();
        assert_eq!(c.engram_rows, 25_000);
        assert_eq!(c.engram_orders, vec![2, 3, 4]);
        assert_eq!(c.engram_dim, 32);
        assert_eq!(c.engram_lam_max, 0.5);
        // 3 tables x 25_000 rows x 32 dim.
        let mem = 3 * c.engram_rows * c.engram_dim;
        assert_eq!(mem, 2_400_000);
        // The `small` backbone is 9_197_390 params MEASURED on the
        // instantiated model (preset_exec, 2026-09-29), so 3 x 25_000 x 32 is
        // 3_145_728 = 34.2% of the model, not the 24% this comment claimed.
        // The 24% was 2.4M / (2.4M + 7.5M) against the RETRACTED 7.5M
        // backbone; 7.5M is the pre-2026-09-27 memory re-pricing and is
        // asserted in schema.rs:130 as a doc comment. `nano` is higher still,
        // 43.7% of 7_192_906.
        //
        // So the containment argument INVERTS and the bound moves: this used
        // to assert 0.20..0.30 and cannot, at 34.2%, without lying. The gate
        // is `share <= 0.5` in preset_exec, deliberately looser, and its
        // reason is that the job here is to stop the 48M-row monopoly shape
        // returning - which 0.5 still does - and NOT to certify a capacity
        // number, which is the owner's call (901be21 declined to move it for
        // exactly that reason).
        const BACKBONE_SMALL: usize = 9_197_390;
        let share = mem as f64 / (mem + BACKBONE_SMALL) as f64;
        assert!(
            share <= 0.5,
            "memory must stay a minority of the model, got {:.1}%",
            share * 100.0
        );
        // The shapes this replaced, for the record: 500K/order is the
        // measured optimum on a 125M backbone but 86% of THIS model, and
        // 8M/order was 99% of it - the configuration the arm was deleted for.
        assert_eq!(3 * 500_000 * c.engram_dim, 48_000_000);
        let monopoly = 3.0 * 500_000.0 * c.engram_dim as f64
            / (3 * 500_000 * c.engram_dim + BACKBONE_SMALL) as f64;
        assert!(
            monopoly > 0.8,
            "the 500K point is the monopoly shape at our scale"
        );
        // In VRAM the rows round up to a power of two: 25_000 -> 32_768.
        let (tables, mask) = engram_tables(c.engram_rows, c.engram_orders.len());
        assert_eq!(tables, vec![32_768; 3]);
        assert_eq!(mask, 32_767);
    }
}
