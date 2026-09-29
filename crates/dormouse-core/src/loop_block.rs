//! loop - mini UniversalLoop: controller + shared attention (KDA) +
//! expert TSCT FFNs + Engram + e_k (iteration embedding) + ReZero residual
//! scale. Fixed depth (ADR-0013): every iteration counts equally, the loss
//! is an honest unweighted CE.
use burn::backend::DispatchKindConversion;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{activation, Device, DispatchTensor, FloatDType, Int, Tensor};
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
    pub fn new(d: usize, f: usize, rank: usize, use_tsct: bool, device: &Device) -> Self {
        Self {
            gate_up: LinearLike::with_tsct(d, f, rank, use_tsct, device),
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
            expert_ffns: (0..cfg.n_experts).map(|_| ExpertFFN::new(d, f, cfg.rank, cfg.use_tsct, device)).collect(),
            engram: EngramModule::new(&tables, cfg.engram_dim, d, 1, device),
            mem_dense: LinearConfig::new(d, d).with_bias(false).init(device),
            norm: RMSNorm::new(d, cfg.norm_eps, device),
            gr: cfg.use_gr.then(|| GatedResidual::new(d, device)),
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

            // Expert FFN: softmax blend of n_experts TSCT gate_up/silu/down
            let mut ffn = Tensor::zeros([b * t, d], &h.device());
            for e in 0..self.n_experts {
                let mid = self.expert_ffns[e].gate_up.forward::<B>(normed_ffn.clone());
                let mid = activation::silu(if bf16 { mid.cast(FloatDType::F32) } else { mid });
                let out = self.expert_ffns[e].down.forward::<B>(mid);
                ffn = ffn + out.mul(blend.clone().slice([0..b * t, e..e + 1]));
            }
            let ffn = ffn.mul(w_ffn).reshape([b, t, d]);

            // ReZero residual (or GR write: per-branch scalar deposit, Eq. 33-34).
            let y = attn.reshape([b, t, d]) + engram_a.reshape([b, t, d]) + ffn;
            if use_gr {
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
            } else {
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
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex()
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