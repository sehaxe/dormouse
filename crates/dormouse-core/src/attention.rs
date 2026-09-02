//! attention - KDA (burn-kda) + MSA (burn-msa) with learned router blend
use crate::param::LinearLike;
use burn::backend::DispatchKindConversion;
use burn::module::Module;
use burn::tensor::{activation, Device, DispatchTensor, FloatDType, Tensor};
use burn_kda::KdaModule;
use burn_msa::{MsaConfig, MsaModule};

#[derive(Module, Debug)]
pub struct AdaptiveAttention {
    pub gdn2: KdaModule,
    pub msa: MsaModule,
    pub router: LinearLike,
    #[module(skip)]
    pub d_model: usize,
    #[module(skip)]
    pub block_size: usize,
    #[module(skip)]
    pub bf16: bool,
}

impl AdaptiveAttention {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        head_dim: usize,
        rank: usize,
        msa_block: usize,
        msa_topk: usize,
        bf16: bool,
        device: &Device,
    ) -> Self {
        let kda_cfg = burn_kda::KdaConfig {
            hidden_size: d_model,
            num_heads: n_heads,
            head_dim,
            // Report §2.1.1 wants the short causal conv, but on this box it
            // made fp32+AdamW NaN at ~step 60 (measured 2026-08-29; without
            // it the same recipe ran 150+ steps clean). Keep off until the
            // instability is traced inside burn-kda.
            use_short_conv: false,
            chunk_size: 16,
            ..Default::default()
        };
        let n_kv = (n_heads / 4).max(1);
        let mut msa_cfg = MsaConfig::new(d_model, n_heads, n_kv, head_dim, 64);
        msa_cfg.block_size = msa_block;
        msa_cfg.topk = msa_topk;
        // gradient_detach leaks autodiff nodes (~72 tensors/step on pre.3):
        // the detached index branch keeps its nodes alive after backward.
        // False = index branch trains (small extra cost), no leak.
        msa_cfg.gradient_detach = false;
        Self {
            gdn2: KdaModule::new(&kda_cfg, 0.0, device),
            msa: MsaModule::new(&msa_cfg, device),
            router: LinearLike::new(d_model, 1, rank.min(d_model).min(1), device),
            d_model,
            block_size: msa_block,
            bf16,
        }
    }

    pub fn forward_train<B: burn::backend::AutodiffBackend>(&self, x: Tensor<3>) -> Tensor<3>
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        let gdn2_out = self.gdn2.forward_train::<B>(x.clone());
        let msa_out = self.msa.forward::<B>(x.clone()).output;
        self.blend::<B>(x, gdn2_out, msa_out)
    }

    pub fn blend<B: burn::backend::AutodiffBackend>(&self, x: Tensor<3>, gdn2_out: Tensor<3>, msa_out: Tensor<3>) -> Tensor<3>
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        let [b, t, d] = x.dims();
        let route = self.router.forward::<B>(x.reshape([b * t, d]));
        let route = if self.bf16 {
            route.cast(FloatDType::F32)
        } else {
            route
        };
        let gate = activation::sigmoid(route.reshape([b, t, 1]));
        gdn2_out.mul(gate.clone()) + msa_out.mul(gate.neg().add_scalar(1.0))
    }
}