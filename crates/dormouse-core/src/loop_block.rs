//! loop - mini UniversalLoop: controller + shared attention (KDA+MSA) +
//! expert TSCT FFNs + Engram + e_k (iteration embedding) + ReZero residual
//! scale. Fixed depth (ADR-0013): every iteration counts equally, the loss
//! is an honest unweighted CE.
use burn::backend::DispatchKindConversion;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{activation, Device, DispatchTensor, FloatDType, Int, Tensor};
use burn_engram::EngramModule;
use burn_rmsnorm::RMSNorm;

use crate::attention::AdaptiveAttention;
use crate::config::{ActQuant, DormouseConfig};
use crate::gr::{GatedResidual, GrState, GR_BRANCHES};
use crate::param::LinearLike;

#[derive(Module, Debug)]
pub struct ExpertFFN {
    pub gate_up: LinearLike,
    pub down: LinearLike,
}

impl ExpertFFN {
    pub fn new(d: usize, f: usize, rank: usize, device: &Device) -> Self {
        Self {
            gate_up: LinearLike::new(d, f, rank, device),
            down: LinearLike::new(f, d, rank, device),
        }
    }
}

#[derive(Module, Debug)]
pub struct LoopBlock {
    pub controller: Linear,
    pub shared_attn: AdaptiveAttention,
    pub expert_ffns: Vec<ExpertFFN>,
    pub engram: EngramModule,
    pub norm: RMSNorm,
    pub gr: Option<GatedResidual>,
    pub iter_embed: burn::module::Param<Tensor<2>>,
    pub residual_scale: burn::module::Param<Tensor<1>>,
    pub out_proj: LinearLike,
    #[module(skip)]
    pub max_iter: usize,
    #[module(skip)]
    pub d_model: usize,
    #[module(skip)]
    pub ffn_hidden: usize,
    #[module(skip)]
    pub n_experts: usize,
    #[module(skip)]
    #[module(skip)]
    pub use_msa: bool,
    #[module(skip)]
    pub use_kda: bool,
    #[module(skip)]
    pub use_engram: bool,
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
        self.shared_attn.router.set_bf16_compute(on);
    }

    /// Polar-retract every TSCT factor U/V in the block (see LinearLike).
    pub fn retract_tsct(&mut self, iters: usize) {
        for f in &mut self.expert_ffns {
            f.gate_up.retract(iters);
            f.down.retract(iters);
        }
        self.out_proj.retract(iters);
        self.shared_attn.router.retract(iters);
    }

    /// Worst orthonormality error across all TSCT factors (syncs the device).
    pub fn max_ortho(&self) -> f32 {
        let mut m = 0.0f32;
        for f in &self.expert_ffns {
            m = m.max(f.gate_up.max_ortho()).max(f.down.max_ortho());
        }
        m.max(self.out_proj.max_ortho()).max(self.shared_attn.router.max_ortho())
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
        Self {
            controller,
            shared_attn: AdaptiveAttention::new(d, cfg.n_heads, cfg.head_dim, cfg.rank, cfg.msa_block, cfg.msa_topk, cfg.bf16, device),
            expert_ffns: (0..cfg.n_experts).map(|_| ExpertFFN::new(d, f, cfg.rank, device)).collect(),
            engram: EngramModule::new(&[4096, 4096, 4096], 32, d, 1, device),
            norm: RMSNorm::new(d, cfg.norm_eps, device),
            gr: cfg.use_gr.then(|| GatedResidual::new(d, device)),
            iter_embed,
            residual_scale: burn::module::Param::from_tensor(Tensor::zeros([1], device)),
            out_proj: LinearLike::new(d, d, cfg.rank, device),
            max_iter: cfg.max_iter,
            d_model: d,
            ffn_hidden: f,
            n_experts: cfg.n_experts,
            use_msa: cfg.use_msa,
            use_kda: cfg.use_kda,
            use_engram: cfg.use_engram,
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
    ) -> (Tensor<3>, Tensor<1>, Tensor<4>)
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        let [b, t, d] = x.dims();
        let bf16 = self.bf16;
        let h0 = x.clone();
        let mut h = x;
        // Gated Residual: every branch starts from the token embedding.
        let use_gr = self.gr.is_some();
        let mut branches: Vec<Tensor<3>> = if use_gr {
            vec![h0.clone(); GR_BRANCHES]
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
        let use_msa = self.use_msa;
        let use_engram = self.use_engram;

        for iter in 0..self.max_iter {
            let row = iter;
            let iter_ctx = self
                .iter_embed
                .val()
                .clone()
                .slice([row..row + 1, 0..d])
                .reshape([1, 1, d]);
            // GR read: normalized gated average of the branches (report
            // Eq. 30-32) replaces the pre-norm + ReZero pair.
            let mut gr_state: Option<GrState> = None;
            let h_ctx = if use_gr {
                let gr = self.gr.as_ref().unwrap();
                let (x_in, st) = gr.read::<B>(&branches);
                gr_state = Some(st);
                if bf16 {
                    x_in.clone() + iter_ctx.cast(FloatDType::BF16)
                } else {
                    x_in.clone() + iter_ctx
                }
            } else {
                if bf16 {
                    h.clone() + iter_ctx.cast(FloatDType::BF16)
                } else {
                    h.clone() + iter_ctx
                }
            };
            // Pre-norm for the block body (identity under GR: the read
            // already normalized).
            let normed = if use_gr {
                h_ctx.clone()
            } else {
                self.norm.forward(h_ctx.clone())
            };
            let normed_f = if bf16 {
                normed.cast(FloatDType::F32)
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
                Some((f, g)) => crate::act_quant::quant_act::<B>(normed_f.reshape([b * t, d]), f, g),
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

            // Shared attention (KDA + MSA). `normed` above is the block-body
            // input: RMSNorm of h_ctx, identity under GR (the read already
            // normalized).
            let (gdn2_out, s_new) = if use_kda {
                self.shared_attn.gdn2.forward_train_state::<B>(normed_attn.clone(), kda_s.take())
            } else {
                (Tensor::zeros([b, t, d], &h.device()), kda_s.take().unwrap_or_else(|| Tensor::zeros([b, 1, 1, 1], &h.device())))
            };
            kda_s = Some(s_new);
            let msa_out = if use_msa && t > 1 && t >= self.shared_attn.block_size {
                // burn-msa kernels are f32-only: a bf16 input makes them
                // read the buffer as f32 (twice the bytes) and fault with
                // CUDA_ERROR_ILLEGAL_ADDRESS (measured 2026-08-29).
                self.shared_attn.msa.forward::<B>(normed_attn.clone()).output
            } else {
                Tensor::zeros([b, t, d], &h.device())
            };
            let attn = self
                .shared_attn
                .blend::<B>(normed_attn.clone(), gdn2_out, msa_out)
                .reshape([b * t, d])
                .mul(w_attn);

            // Engram (FNV hashed ids) with memory weight
            let engram_a = if use_engram {
                match &host_rows {
                    // RAM-offload path: rows already gathered on the host.
                    Some(rows) => {
                        let eg_in = if bf16 {
                            h_ctx.clone().reshape([b, t, 1, d]).cast(FloatDType::F32)
                        } else {
                            h_ctx.clone().reshape([b, t, 1, d])
                        };
                        self.engram
                            .forward_embeds(rows.clone(), eg_in)
                            .reshape([b, t, d])
                            .reshape([b * t, d])
                            .mul(w_mem)
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
                            self.engram
                                .forward((*hashed).clone(), eg_in)
                                .reshape([b, t, d])
                                .reshape([b * t, d])
                                .mul(w_mem)
                        }
                        None => Tensor::zeros([b * t, d], &h.device()),
                    },
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
                let gr = self.gr.as_ref().unwrap();
                branches = gr.write::<B>(&branches, gr_state.as_ref().unwrap(), y);
                // Readout reads the normalized block input; the branches
                // carry the accumulated state for the next iteration.
                h = h_ctx.clone();
            } else {
                let scale = self.residual_scale.val().clone().reshape([1, 1, 1]);
                // Store the residual back in the activation dtype (bf16
                // under --bf16): the sum itself is computed in fp32.
                h = h_ctx.clone() + y.mul(scale).cast(h_ctx.dtype());
            }

            // Per-iteration readout. Fixed depth (ADR-0013): uniform
            // iteration weights and an honest unweighted CE — the PonderNet
            // variant lost its A/B (lambda collapse zeroed rec, a fake loss,
            // and out_acc, uniform outputs).
            let step_out = self.out_proj.forward::<B>(h.clone().reshape([b * t, d])).reshape([b, t, d]);
            let w = 1.0f32 / self.max_iter as f32;
            out_acc = out_acc + step_out.clone().mul_scalar(w);
            if let Some(tgt) = &targets {
                let so = if bf16 { step_out.clone().cast(FloatDType::F32) } else { step_out.clone() };
                let logits_n = lm_head.forward::<B>(so.reshape([b * t, d])); // [b*t, v]
                // Gather the target column of the log-softmax instead of an
                // elementwise one-hot product: no [b*t,v] fp32 temporary per
                // iteration (max_iter of them per step otherwise).
                let ce = burn::tensor::activation::log_softmax(logits_n, 1)
                    .gather(1, tgt.clone())
                    .neg()
                    .reshape([b, t])
                    .sum_dim(1)
                    .div_scalar(t as f32); // [b]
                rec = rec + ce.sum_dim(0).reshape([1]).div_scalar(self.max_iter as f32);
            }
            // The readout consumed the post-residual h; the recurrence reset
            // it to the block input (per-iteration writes reach the loss via
            // step_out/out_acc, not the next iteration).
            h = h_ctx.clone();
        }
        let rec = rec.div_scalar(b as f32); // mean over batch
        let kda = kda_s.unwrap_or_else(|| Tensor::zeros([1, 1, 1, 1], &h.device()));
        (out_acc, rec, kda)
    }
}