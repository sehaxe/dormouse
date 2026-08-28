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
        kda_state: Option<Tensor<4>>,
    ) -> (Tensor<3>, Tensor<1>, Tensor<1>, Tensor<4>)
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        let [b, t, d] = x.dims();
        let bf16 = bf16_on();
        let h0 = x.clone();
        let mut h = x;
        let mut ponder = Tensor::<1>::zeros([1], &h.device());
        let mod_loss = Tensor::<1>::zeros([1], &h.device());
        let mut kda_s: Option<Tensor<4>> = kda_state;

        for iter in 0..self.max_iter {
            let row = iter.min(self.max_iter - 1);
            let iter_ctx = self
                .iter_embed
                .val()
                .clone()
                .slice([row..row + 1, 0..d])
                .reshape([1, 1, d]);
            let h_ctx = if bf16 {
                h.clone() + iter_ctx.cast(FloatDType::BF16)
            } else {
                h.clone() + iter_ctx
            };

            // Controller routing on [h_ctx, h0]
            let ctrl_in = Tensor::cat(vec![h_ctx.clone().reshape([b * t, d]), h0.clone().reshape([b * t, d])], 1);
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
                self.shared_attn.gdn2.forward_train_state::<B>(normed.clone(), kda_s.take())
            } else {
                (Tensor::zeros([b, t, d], &h.device()), kda_s.take().unwrap_or_else(|| Tensor::zeros([b, 1, 1, 1], &h.device())))
            };
            kda_s = Some(s_new);
            let msa_out = if use_msa && t > 1 && t >= self.shared_attn.block_size {
                self.shared_attn.msa.forward::<B>(normed.clone()).output
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
                match &hashed_ids {
                    Some(hashed) => self
                        .engram
                        .forward((*hashed).clone(), h_ctx.clone().reshape([b, t, 1, d]))
                        .reshape([b, t, d])
                        .reshape([b * t, d])
                        .mul(w_mem),
                    None => Tensor::zeros([b * t, d], &h.device()),
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

            // PonderNet halt lambda over current hidden
            let lam = activation::sigmoid(self.halt_head.forward(h_ctx.clone().mean_dim(1)));
            ponder = ponder + lam.mean(); // halting term (weight applied in loss)

            // ReZero residual
            let scale = self.residual_scale.val().clone().reshape([1, 1, 1]);
            h = h_ctx + (attn.reshape([b, t, d]) + engram_a.reshape([b, t, d]) + ffn).mul(scale);
        }
        // final readout projection
        let out = self.out_proj.forward::<B>(h.clone().reshape([b * t, d])).reshape([b, t, d]);
        let kda = kda_s.unwrap_or_else(|| Tensor::zeros([1, 1, 1, 1], &h.device()));
        (out, ponder, mod_loss, kda)
    }
}