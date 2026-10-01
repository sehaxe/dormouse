//! loop - mini UniversalLoop: controller + shared attention (KDA) +
//! expert TSCT FFNs + Engram + e_k (iteration embedding) + ReZero residual
//! scale. Fixed depth (ADR-0013): every iteration counts equally, the loss
//! is an honest unweighted CE.
use burn::backend::DispatchKindConversion;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{activation, Device, DispatchTensor, FloatDType, Int, Tensor};
use burn_attnres::{depth_attend, AttnRes};
use burn_engram::EngramModule;
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

#[derive(Module, Debug)]
pub struct ExpertFFN {
    pub gate_up: LinearLike,
    pub down: LinearLike,
}

impl ExpertFFN {
    /// `situ` makes `gate_up` a `d -> 2f` projection, because Eq (12) reads
    /// `Wg x` and `Wu x` as separate inputs. With `situ = false` the
    /// projection stays `d -> f` and the field's name is the misnomer it has
    /// always been (a `d -> f` up-projection named after two projections);
    /// `situ_off_leaves_the_parameter_set_alone` holds the off arm to the old
    /// shape.
    pub fn new(d: usize, f: usize, rank: usize, use_tsct: bool, situ: bool, device: &Device) -> Self {
        let mid = if situ { 2 * f } else { f };
        Self {
            gate_up: LinearLike::with_tsct(d, mid, rank, use_tsct, device),
            down: LinearLike::with_tsct(f, d, rank, use_tsct, device),
        }
    }
}

#[derive(Module, Debug)]
pub struct LoopBlock {
    pub controller: Linear,
    pub shared_attn: AdaptiveAttention,
    pub expert_ffns: Vec<ExpertFFN>,
    pub engram: EngramModule,
    /// The dense half of the memory branch's convex mixture: a plain
    /// `d_model -> d_model` projection of the SAME hidden state the memory
    /// is gated against. It is the `(1 - g) * v_t` term of FwPKM eq. 12 and
    /// the reason the backbone can no longer be starved (see
    /// [`memory_floor_mix`]).
    pub mem_dense: Linear,
    pub norm: RMSNorm,
    pub gr: Option<GatedResidual>,
    /// Attention Residuals (arXiv:2603.15031): one learned pseudo-query `w_l`
    /// per loop iteration slot (§5, "one RMSNorm and one pseudo-query vector
    /// per layer"), zero-initialised so the first forward is an equal-weight
    /// average. `None` unless `use_attnres`, which is what keeps every
    /// existing checkpoint loadable: the parameters do not exist when the arm
    /// is off, and the config snapshot refuses a resume that flips the flag.
    pub attnres: Option<Vec<AttnRes>>,
    pub iter_embed: burn::module::Param<Tensor<2>>,
    pub residual_scale: burn::module::Param<Tensor<1>>,
    pub out_proj: LinearLike,
    /// MoR router: the shared linear scorer of arXiv 2507.10524, one score
    /// per (position, iteration slot). Always present (769 params, routed to
    /// AdamW by the optimizer policy - routers are not Muon+ candidates);
    /// `use_mor` decides whether it is read.
    pub mor_router: MoRRouter,
    #[module(skip)]
    pub max_iter: usize,
    #[module(skip)]
    pub d_model: usize,
    #[module(skip)]
    pub ffn_hidden: usize,
    #[module(skip)]
    pub n_experts: usize,
    #[module(skip)]
    pub use_kda: bool,
    /// Random-depth arm (ADR-0013 rank 2): run only the first `n` iterations
    /// instead of `max_iter`, and average the readout and the CE over the
    /// iterations actually executed. `None` = fixed depth, the default.
    /// The trainer sets it per step; there is no learned halting here, so
    /// there is nothing to collapse.
    #[module(skip)]
    pub depth_override: Option<usize>,
    #[module(skip)]
    pub use_engram: bool,
    /// Read [`LoopBlock::attnres`]. Kept as a plain field (not `attnres.is_some()`)
    /// so the branch is a config constant, like every other arm switch.
    #[module(skip)]
    pub use_attnres: bool,
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
    /// Selected iteration slots per position under MoR (>= 1, see
    /// `mor::route`). Ignored when `use_mor` is off.
    #[module(skip)]
    pub mor_k: usize,
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
    #[module(skip)]
    pub bf16: bool,
    #[module(skip)]
    pub act_quant: Option<ActQuant>,
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

    pub fn new(cfg: &DormouseConfig, device: &Device) -> Self {
        let d = cfg.d_model;
        let f = cfg.d_ffn;
        // Controller: [h_ctx, h0] (2d) -> weights for attn/mem/ffn + expert blend
        let n_ctrl = 3 + cfg.n_experts;
        let ctrl_pad = if !n_ctrl.is_multiple_of(4) { n_ctrl.next_multiple_of(4) } else { n_ctrl };
        let controller = LinearConfig::new(d * 2, ctrl_pad).with_bias(false).init(device);
        let mut iter_embed = burn::tensor::Tensor::<2>::zeros([cfg.max_iter, d], device);
        iter_embed = iter_embed.into();
        let iter_embed = burn::module::Param::from_tensor(iter_embed.clone().into());
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
            iter_embed,
            // ReZero's residual coefficient starts at 1 (identity init), NOT 0:
            // at 0 the block body contributes nothing AND its gradient is
            // exactly zero (dL/dy = 0), so the KDA arm, the expert FFNs and the
            // controller's gates all start with no gradient at all - the model
            // is a linear map of the byte embedding until the scalar moves.
            residual_scale: burn::module::Param::from_tensor(Tensor::ones([1], device)),
            out_proj: LinearLike::with_tsct(d, d, cfg.rank, cfg.use_tsct, device),
            mor_router: MoRRouter::new(d, device),
            max_iter: cfg.max_iter,
            d_model: d,
            ffn_hidden: f,
            n_experts: cfg.n_experts,
            use_kda: cfg.use_kda,
            depth_override: None,
            use_engram: cfg.use_engram,
            use_attnres: cfg.use_attnres,
            use_situ: cfg.use_situ,
            use_mor: cfg.use_mor,
            mor_k: cfg.mor_k,
            engram_slot_mask: mask,
            engram_lam_max: cfg.engram_lam_max,
            bf16: cfg.bf16,
            act_quant: cfg.act_quant,
            act_group: cfg.act_group,
        }
    }

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
    ) -> (Tensor<3>, Tensor<1>, Tensor<4>, Option<Tensor<1>>)
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
            if use_situ { 2 * self.ffn_hidden } else { self.ffn_hidden },
            "use_situ = {use_situ} but the experts' gate_up is {} wide, so it was built for \
             the other arm: set use_situ in the config (so ExpertFFN::new sizes gate_up \
             accordingly) rather than flipping the field after construction",
            self.expert_ffns[0].gate_up.out_features
        );
        assert!(
            !(use_attnres && use_gr),
            "use_attnres and use_gr both replace the residual accumulation; \
             config::validate refuses this combination"
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
        let mut res: Vec<Tensor<3>> = if use_attnres { vec![h0.clone()] } else { Vec::new() };
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
        let mut slot_scores: Vec<Tensor<3>> = Vec::with_capacity(iters);
        let mut step_outs: Vec<Tensor<3>> = Vec::with_capacity(iters);
        let mut ce_terms: Vec<Tensor<2>> = Vec::with_capacity(iters);

        for iter in 0..iters {
            crate::probe::note(crate::probe::ITER);
            let row = iter;
            // This slot's pseudo-query `w_l` (§5: one per layer). Read once
            // per iteration, before the block body, so the query is a
            // function of the slot and not of anything computed this step.
            let slot_query = match use_attnres {
                true => self.attnres.as_ref().expect("use_attnres => attnres")[row]
                    .query
                    .val()
                    .clone(),
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
                Some((f, g)) => crate::act_quant::quant_act::<B>(normed_f.clone().reshape([b * t, d]), f, g)
                    .reshape([b, t, d]),
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
            let blend = activation::softmax(raw.slice([0..b * t, 3..3 + self.n_experts]), 1);

            // Shared attention. `normed` above is the block-body input:
            // RMSNorm of h_ctx, identity under GR (the read already
            // normalized).
            let (gdn2_out, s_new) = if use_kda {
                crate::probe::note(crate::probe::KDA);
                self.shared_attn.gdn2.forward_train_state::<B>(normed_attn.clone(), kda_s.take())
            } else {
                (Tensor::zeros([b, t, d], &h.device()), kda_s.take().unwrap_or_else(|| Tensor::zeros([b, 1, 1, 1], &h.device())))
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
                                    .forward(hashed.clone().bitwise_and_scalar(self.engram_slot_mask), eg_in)
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

            // ReZero residual, AttnRes aggregation, or GR write (per-branch
            // scalar deposit, Eq. 33-34). These are three spellings of ONE
            // statement - how iteration n's block-body output joins the
            // residual stream - and `config::validate` refuses the two that
            // cannot both run.
            let y = attn.reshape([b, t, d]) + engram_a.reshape([b, t, d]) + ffn;
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
            let step_out = self.out_proj.forward::<B>(h.clone().reshape([b * t, d])).reshape([b, t, d]);
            if use_mor {
                crate::probe::note(crate::probe::MOR);
                // MoR router: the shared linear scorer reads THIS iteration's
                // input state (arXiv 2507.10524's per-step linear router).
                let hs = if bf16 { h_ctx.clone().cast(FloatDType::F32) } else { h_ctx.clone() };
                slot_scores.push(self.mor_router.scores(hs));
            }
            step_outs.push(step_out.clone());
            if let Some(tgt) = &targets {
                let so = if bf16 { step_out.clone().cast(FloatDType::F32) } else { step_out.clone() };
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
            (
                Tensor::<3>::ones([b, t, iters], &h.device()),
                None,
            )
        };
        // The divisor is the count of selected slots, known on the host (k is
        // a config constant) — reading it off the mask would sync the device
        // every step for a number we already have.
        let ksel = if use_mor { mor::eff_k(self.mor_k, iters) } else { iters };
        let w = 1.0f32 / ksel as f32;
        for n in 0..iters {
            // Mixed-dtype (f32 mask x bf16 activation) NaNs on this stack, so
            // the gate lands in the readout's own dtype.
            let g = mask.clone().slice([0..b, 0..t, n..n + 1]);
            let g = g.cast(step_outs[n].dtype());
            out_acc = out_acc + step_outs[n].clone().mul(g.clone()).mul_scalar(w);
            if let Some(ce) = ce_terms.get(n) {
                rec = rec + ce
                    .clone()
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
        (out_acc, rec, kda, mor_aux)
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
        assert!((a - lam_max).abs() < 1e-5, "saturated w_mem must sit AT the cap, got {a}");

        // Below the cap the learned value still works - the clamp is a
        // ceiling, not a constant.
        let out = memory_floor_mix(t(mem), t(dense), t(0.1), lam_max);
        let a = recover_a(&out, mem, dense);
        assert!((a - 0.1).abs() < 1e-5, "below the cap the mix must follow w_mem, got {a}");

        // lam_max = 1.0 is the old behaviour (a direct row copy, no floor) -
        // the degenerate case this test exists to forbid by default.
        let out = memory_floor_mix(t(mem), t(dense), t(1.0), 1.0);
        let a = recover_a(&out, mem, dense);
        assert!((a - 1.0).abs() < 1e-5, "lam_max=1 must be a pure memory read, got {a}");
    }

    /// ATTNRES: the readout MUST move when the aggregation changes, or the
    /// arm ran and did nothing - which is what GR's 9b343d3 defect was (the
    /// readout was taken from the state BEFORE the write, so the body was
    /// computed and discarded at depth 1).
    #[test]
    fn attnres_moves_the_readout_at_every_depth() {
        for depth in [1usize, 2, 3] {
            let mut cfg = DormouseConfig::default();
            cfg.d_model = 32;
            cfg.n_heads = 2;
            cfg.head_dim = 16;
            cfg.d_ffn = 64;
            cfg.max_iter = depth;
            cfg.n_experts = 1;
            cfg.rank = 8;
            cfg.engram_rows = 256;
            cfg.use_kda = false;
            cfg.use_engram = false;
            let mut plain = LoopBlock::new(&cfg, &adev());
            let mut ar = LoopBlock::new(&DormouseConfig { use_attnres: true, ..cfg.clone() }, &adev());
            let [b, t, d] = [2usize, 5, 32];
            let x = Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &adev());
            let head = LinearLike::with_tsct(d, 16, 8, cfg.use_tsct, &adev());
            let (o_rezero, _, _, _) =
                plain.forward_full_state::<B>(x.clone(), None, None, None, None, &head);
            let (o_attnres, _, _, _) =
                ar.forward_full_state::<B>(x, None, None, None, None, &head);
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
        let mut cfg = DormouseConfig::default();
        cfg.d_model = 32;
        cfg.n_heads = 2;
        cfg.head_dim = 16;
        cfg.d_ffn = 64;
        cfg.max_iter = 1;
        cfg.n_experts = 1;
        cfg.rank = 8;
        cfg.engram_rows = 256;
        cfg.use_kda = false;
        cfg.use_engram = false;
        let ar = LoopBlock::new(&DormouseConfig { use_attnres: true, ..cfg.clone() }, &adev());
        // Pin the zero init: if a future change randomizes the query, the
        // uniform-average property is gone and the arm no longer starts where
        // the paper says it starts. Every component, so a partially-zeroed
        // vector is red too.
        let q = ar.attnres.as_ref().expect("arm on")[0].query.val().clone();
        let worst = q.abs().max().into_scalar::<f32>();
        assert_eq!(worst, 0.0, "w_l must be zero-initialized (§5): uniform alpha at init");
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
        let mut cfg = DormouseConfig::default();
        cfg.d_model = 32;
        cfg.max_iter = 2;
        let on = LoopBlock::new(&DormouseConfig { use_attnres: true, ..cfg }, &adev());
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
        let mut cfg = DormouseConfig::default();
        cfg.d_model = 32;
        cfg.n_heads = 2;
        cfg.head_dim = 16;
        cfg.d_ffn = 64;
        cfg.max_iter = 3;
        cfg.n_experts = 1;
        cfg.rank = 8;
        cfg.engram_rows = 256;
        let b = LoopBlock::new(&cfg, &adev());
        assert!(b.attnres.is_none(), "use_attnres defaults to false, so no query vectors");
        assert!(!b.use_attnres);
        // The default config itself, so a preset cannot turn this on by
        // accident: the flag is off unless someone says so.
        assert!(!DormouseConfig::default().use_attnres);
        // And on, there is exactly one query per iteration slot.
        let on = LoopBlock::new(&DormouseConfig { use_attnres: true, ..cfg }, &adev());
        let qs = on.attnres.as_ref().expect("arm on");
        assert_eq!(qs.len(), on.max_iter, "one pseudo-query per iteration slot (§5)");
        assert!(qs.iter().all(|a| a.query.dims() == [32]), "each is a [d_model] vector");
    }

    /// ATTNRES and GR both REPLACE the residual accumulation, so a config that
    /// asks for both has no interpretation. Refused in `config::validate`, and
    /// re-checked at the branch, because a hand-built `LoopBlock` bypasses the
    /// config path.
    #[test]
    fn attnres_and_gr_are_refused_together() {
        let mut cfg = DormouseConfig::default();
        cfg.d_model = 32;
        cfg.n_heads = 2;
        cfg.head_dim = 16;
        cfg.d_ffn = 64;
        cfg.max_iter = 2;
        cfg.n_experts = 1;
        cfg.rank = 8;
        cfg.engram_rows = 256;
        let both = DormouseConfig { use_attnres: true, use_gr: true, ..cfg.clone() };
        let err = crate::config::validate(&both).expect_err("both residual arms must be refused");
        assert!(err.contains("use_attnres") && err.contains("use_gr"), "the error must name both: {err}");
        // Either alone is legal.
        assert!(crate::config::validate(&DormouseConfig { use_attnres: true, ..cfg.clone() }).is_ok());
        assert!(crate::config::validate(&DormouseConfig { use_gr: true, ..cfg.clone() }).is_ok());
    }

    /// THE COUNTER. `use_attnres = true` with a counter of 0 would be a run
    /// that reports ReZero's loss under AttnRes's name - the defect ADR-0019
    /// is about, in the one shape where nothing else would show it.
    #[test]
    fn attnres_counts_every_iteration_it_aggregates() {
        let mut cfg = DormouseConfig::default();
        cfg.d_model = 32;
        cfg.n_heads = 2;
        cfg.head_dim = 16;
        cfg.d_ffn = 64;
        cfg.max_iter = 4;
        cfg.n_experts = 1;
        cfg.rank = 8;
        cfg.engram_rows = 256;
        cfg.use_kda = false;
        cfg.use_engram = false;
        let head = LinearLike::with_tsct(32, 16, 8, cfg.use_tsct, &adev());
        let x = Tensor::<3>::random([2, 5, 32], Distribution::Normal(0.0, 1.0), &adev());

        crate::probe::reset();
        let mut ar = LoopBlock::new(&DormouseConfig { use_attnres: true, ..cfg.clone() }, &adev());
        let _ = ar.forward_full_state::<B>(x.clone(), None, None, None, None, &head);
        assert_eq!(crate::probe::count(crate::probe::ATTNRES), 4, "one aggregation per iteration at depth 4");

        // Off: zero. The ReZero path must not touch the counter, or the field
        // on the eval line would read non-zero for a run that never aggregated.
        crate::probe::reset();
        let mut rz = LoopBlock::new(&cfg, &adev());
        let _ = rz.forward_full_state::<B>(x, None, None, None, None, &head);
        assert_eq!(crate::probe::count(crate::probe::ATTNRES), 0, "ReZero does not aggregate");
        // Random depth is a truncation of what ran, and the counter must count
        // what RAN: a truncated run that reported max_iter aggregations would
        // be counting a model that was not evaluated.
        crate::probe::reset();
        let mut ar = LoopBlock::new(&DormouseConfig { use_attnres: true, ..cfg }, &adev());
        ar.set_depth(Some(2));
        let _ = ar.forward_full_state::<B>(Tensor::zeros([2, 5, 32], &adev()), None, None, None, None, &head);
        assert_eq!(crate::probe::count(crate::probe::ATTNRES), 2, "depth 2 aggregated twice");
    }

    // ---------------------------------------------------------------------
    // SITU (arXiv:2607.24653v2 Eq 12). The FORM is gated in the mechanism
    // crate, against Moonshot's own numbers - see
    // vendor/burn-fused/crates/burn-situ/src/lib.rs and
    // research/reviews/situ-2026-09-30.md. What is gated HERE is the wiring:
    // that the flag is load-bearing, that the counter sees the arm, and that
    // the gradient survives the cap. The form gate cannot see any of it: a
    // perfectly-formed SiTU wired to nothing is a green crate and a run that
    // measured SiLU.
    // ---------------------------------------------------------------------

    /// `d_ffn` and `n_experts` for the gates below, and the arms off: the
    /// gradient numbers are about the activation, and KDA/Engram only add
    /// other arms' counters to the same forward.
    fn situ_cfg() -> DormouseConfig {
        let mut cfg = DormouseConfig::default();
        cfg.d_model = 32;
        cfg.n_heads = 2;
        cfg.head_dim = 16;
        // A multiple of 4, so `LinearLike`'s `N % 4 == 0` padding
        // (param.rs:49) leaves the width exactly `f` and `2f` and the
        // width assertions below are about the arm, not about the pad.
        cfg.d_ffn = 64;
        cfg.n_experts = 2;
        cfg.rank = 8;
        cfg.engram_rows = 256;
        cfg.max_iter = 2;
        cfg.use_kda = false;
        cfg.use_engram = false;
        cfg
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
        let parsed: DormouseConfig = toml::from_str(old_snapshot).expect("a pre-situ snapshot parses");
        assert!(!parsed.use_situ, "a config without the field is the OFF arm");

        // On: `d -> 2f`, exactly. This is the whole parameter delta of the arm
        // and it is why the A/B is width-confounded (schema.rs).
        let on = LoopBlock::new(&DormouseConfig { use_situ: true, ..cfg.clone() }, &adev());
        for e in &on.expert_ffns {
            assert_eq!(e.gate_up.out_features, 2 * cfg.d_ffn, "Eq (12) needs Wg and Wu separately");
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
            delta, cfg.n_experts * cfg.rank * cfg.d_ffn,
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
        let mut on = LoopBlock::new(&DormouseConfig { use_situ: true, ..cfg.clone() }, &adev());
        let _ = on.forward_full_state::<B>(x.clone(), None, None, None, None, &head);
        assert_eq!(
            crate::probe::count(crate::probe::SITU),
            (cfg.n_experts * cfg.max_iter) as u64,
            "one SiTU per expert per iteration: n_experts * max_iter"
        );

        // Off: zero. The silu path must not touch the counter, or the field
        // would read non-zero for a run that never activated the arm.
        crate::probe::reset();
        let mut off = LoopBlock::new(&cfg, &adev());
        let _ = off.forward_full_state::<B>(x.clone(), None, None, None, None, &head);
        assert_eq!(crate::probe::count(crate::probe::SITU), 0, "silu does not run SiTU");

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
        let on = LoopBlock::new(&DormouseConfig { use_situ: true, ..cfg.clone() }, &adev());
        let expert = &on.expert_ffns[0];
        let n = 64usize;
        // `gate_up` is `d_model -> 2*d_ffn`, so the input is d_model-wide and
        // the BOUNDED tensor is its output. A deliberately hot input: an
        // RMSNormed hidden state is O(1), so this is far outside the training
        // regime - the overfit condition AGENTS.md §2.3 says the NaN class
        // lives in, and the regime Eq (19)'s bound is claimed for.
        let probe = Tensor::<2>::random([n, cfg.d_model], Distribution::Normal(0.0, 1.0), &adev());
        let mid = expert.gate_up.forward::<B>(probe);
        assert_eq!(mid.dims(), [n, 2 * cfg.d_ffn], "the up-projection is d_model -> 2*d_ffn");
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
        assert_eq!(via_situ.dims(), via_silu.dims(), "the comparator is elementwise");
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
        assert!(diff > 1.0, "the two activations differ by only {diff} on a hot input");
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
        let on = LoopBlock::new(&DormouseConfig { use_situ: true, ..cfg }, &adev());
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
            let g = input.grad(&grads).expect("a require_grad leaf has a gradient");
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
        assert_eq!(tables, vec![524_288; 3], "500_000 rounds up to the next power of two");
        assert_eq!(mask, 524_287);
        assert_eq!(engram_tables(25_000, 3), (vec![32_768; 3], 32_767));
        assert_eq!(engram_tables(1024, 3), (vec![1024; 3], 1023));
        assert_eq!(engram_tables(1, 1), (vec![1], 0));

        let mut cfg = DormouseConfig::default();
        cfg.d_model = 32;
        cfg.n_heads = 2;
        cfg.head_dim = 16;
        cfg.d_ffn = 64;
        cfg.max_iter = 1;
        cfg.n_experts = 1;
        cfg.rank = 8;
        cfg.engram_rows = 1024;
        cfg.engram_lam_max = 0.25;
        let b = LoopBlock::new(&cfg, &dev());
        assert_eq!(b.engram_lam_max, 0.25, "the block must carry the configured floor");
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
        assert!(crate::config::validate(&cfg).is_err(), "the order COUNT is pinned to 3");
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
        assert!(monopoly > 0.8, "the 500K point is the monopoly shape at our scale");
        // In VRAM the rows round up to a power of two: 25_000 -> 32_768.
        let (tables, mask) = engram_tables(c.engram_rows, c.engram_orders.len());
        assert_eq!(tables, vec![32_768; 3]);
        assert_eq!(mask, 32_767);
    }
}