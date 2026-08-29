//! loop - mini UniversalLoop on fused kernels: controller + shared attention
//! (KDA+MSA) + expert TSCT FFNs + Engram + PonderNet halt head + e_k
//! (iteration embedding) + ReZero residual scale.
use burn::backend::{Backend, DispatchKindConversion};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{activation, Device, DispatchTensor, FloatDType, Int, Tensor};
use burn_engram::EngramModule;
use burn_rmsnorm::RMSNorm;

use crate::attention::AdaptiveAttention;
use crate::config::DormouseConfig;
use crate::gr::{GatedResidual, GrState, GR_BRANCHES};
use crate::param::{bf16_on, LinearLike};

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
    pub halt_head: Linear,
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
    pub halt_theta: f32,
    #[module(skip)]
    pub use_msa: bool,
    #[module(skip)]
    pub use_kda: bool,
}

impl LoopBlock {
    /// Apply a quantization format to every TSCT factor (stage-2 switch).
    pub fn set_quant_all(&mut self, quant: burn_spectral::QuantFormat) {
        for f in &mut self.expert_ffns {
            f.gate_up.set_quant(quant);
            f.down.set_quant(quant);
        }
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
            shared_attn: AdaptiveAttention::new(d, cfg.n_heads, cfg.head_dim, cfg.rank, cfg.msa_block, cfg.msa_topk, device),
            expert_ffns: (0..cfg.n_experts).map(|_| ExpertFFN::new(d, f, cfg.rank, device)).collect(),
            engram: EngramModule::new(&[4096, 4096, 4096], 32, d, 1, device),
            norm: RMSNorm::new(d, cfg.norm_eps, device),
            gr: cfg.use_gr.then(|| GatedResidual::new(d, device)),
            halt_head: LinearConfig::new(d, 1).with_bias(false).init(device),
            iter_embed,
            residual_scale: burn::module::Param::from_tensor(Tensor::zeros([1], device)),
            out_proj: LinearLike::new(d, d, cfg.rank, device),
            max_iter: cfg.max_iter,
            d_model: d,
            ffn_hidden: f,
            n_experts: cfg.n_experts,
            halt_theta: cfg.halt_theta,
            use_msa: cfg.use_msa,
            use_kda: cfg.use_kda,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn forward_full_state<B: Backend>(
        &self,
        x: Tensor<3>,
        hashed_ids: Option<Tensor<3, Int>>,
        host_rows: Option<Tensor<3>>,
        kda_state: Option<Tensor<4>>,
        targets: Option<Tensor<2>>,
        lm_head: &LinearLike,
    ) -> (Tensor<3>, Tensor<1>, Tensor<2>, Tensor<4>)
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let [b, t, d] = x.dims();
        let bf16 = bf16_on();
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
        // PonderNet accumulators (Banino et al. 2021, arXiv:2107.05407)
        let mut not_halted = Tensor::<2>::ones([b, 1], &h.device());
        let mut out_acc = Tensor::<3>::zeros([b, t, d], &h.device());
        let mut p_rows: Vec<Tensor<2>> = Vec::with_capacity(self.max_iter);
        // L_Rec = mean_b Σ_n p_n·CE(ŷ_n, y), accumulated inside the loop so we
        // never materialize/slice the [N,b,t,d] tensor: dynamic slicing of a 4D
        // autodiff tensor crashes cubecl on sm_120 (CUDA_ERROR_ILLEGAL_ADDRESS).
        let mut rec = Tensor::<1>::zeros([1], &h.device());

        for iter in 0..self.max_iter {
            let row = iter.min(self.max_iter - 1);
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

            // Controller routing on [h_ctx, h0]
            let ctrl_in = Tensor::cat(vec![h_ctx.clone().reshape([b * t, d]), h0.clone().reshape([b * t, d])], 1);
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

            // Pre-norm, then shared attention (KDA + MSA)
            let normed = self.norm.forward(h_ctx.clone());
            // env switches: DM_NO_KDA / DM_NO_MSA / DM_NO_ENGRAM disable,
            // DM_KDA / DM_MSA force-enable (bisect + A/B).
            let use_kda = (self.use_kda || std::env::var("DM_KDA").is_ok())
                && std::env::var("DM_NO_KDA").is_err();
            let use_msa = (self.use_msa || std::env::var("DM_MSA").is_ok())
                && std::env::var("DM_NO_MSA").is_err();
            let use_engram = std::env::var("DM_NO_ENGRAM").is_err();
            let (gdn2_out, s_new) = if use_kda {
                let kda_in = if bf16 {
                    normed.clone().cast(FloatDType::F32)
                } else {
                    normed.clone()
                };
                self.shared_attn.gdn2.forward_train_state::<B>(kda_in, kda_s.take())
            } else {
                (Tensor::zeros([b, t, d], &h.device()), kda_s.take().unwrap_or_else(|| Tensor::zeros([b, 1, 1, 1], &h.device())))
            };
            kda_s = Some(s_new);
            let msa_out = if use_msa && t > 1 && t >= self.shared_attn.block_size {
                // burn-msa kernels are f32-only: a bf16 input makes them
                // read the buffer as f32 (twice the bytes) and fault with
                // CUDA_ERROR_ILLEGAL_ADDRESS (measured 2026-08-29).
                let msa_in = if bf16 {
                    normed.clone().cast(FloatDType::F32)
                } else {
                    normed.clone()
                };
                self.shared_attn.msa.forward::<B>(msa_in).output
            } else {
                Tensor::zeros([b, t, d], &h.device())
            };
            let attn = self
                .shared_attn
                .blend::<B>(normed.clone(), gdn2_out, msa_out)
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
            let normed_flat = normed.reshape([b * t, d]);
            let mut ffn = Tensor::zeros([b * t, d], &h.device());
            for e in 0..self.n_experts {
                let mid = self.expert_ffns[e].gate_up.forward::<B>(normed_flat.clone());
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
                // under BF16=1): the sum itself is computed in fp32.
                h = h_ctx.clone() + y.mul(scale).cast(h_ctx.dtype());
            }

            // PonderNet readout + probabilistic halting:
            // λ_n = cond. halt prob; p_n = λ_n · Π_{j<n}(1-λ_j) (truncated geometric);
            // output accumulates the p-weighted expectation of per-step logits.
            let step_out = self.out_proj.forward::<B>(h.clone().reshape([b * t, d])).reshape([b, t, d]);
            let halt_in = if bf16 {
                h_ctx.clone().mean_dim(1).reshape([b, d]).cast(FloatDType::F32)
            } else {
                h_ctx.clone().mean_dim(1).reshape([b, d])
            };
            let lam = activation::sigmoid(
                self.halt_head.forward(halt_in),
            ); // [b,1]
            let p_n = lam.clone() * not_halted.clone();
            out_acc = out_acc + step_out.clone().mul(p_n.clone().unsqueeze_dim::<3>(2));
            p_rows.push(p_n.clone());
            not_halted = not_halted * (Tensor::<2>::ones([b, 1], &h.device()) - lam.clone());
            // L_Rec: per-step CE (fp32 logits) weighted by the halting dist p_n.
            if let Some(tgt) = &targets {
                let v = tgt.dims()[1];
                let so = if bf16 { step_out.clone().cast(FloatDType::F32) } else { step_out.clone() };
                let logits_n = lm_head
                    .forward::<B>(so.reshape([b * t, d]))
                    .reshape([b * t, v]);
                let ce = (burn::tensor::activation::log_softmax(logits_n, 1) * tgt.clone())
                    .sum_dim(1)
                    .neg()
                    .reshape([b, t])
                    .sum_dim(1)
                    .div_scalar(t as f32); // [b]
                let pn = p_n.clone().reshape([b, 1]); // [b,1]
                rec = rec + (pn * ce.reshape([b, 1])).sum_dim(0).reshape([1]);
            }
        }
        let rec = rec.div_scalar(b as f32); // mean over batch
        let p_dist = Tensor::cat(p_rows, 1); // [b, N]
        let kda = kda_s.unwrap_or_else(|| Tensor::zeros([1, 1, 1, 1], &h.device()));
        (out_acc, rec, p_dist, kda)
    }
}