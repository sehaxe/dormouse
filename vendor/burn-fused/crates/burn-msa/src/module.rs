use burn::backend::Backend;
use burn::module::Module;
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn_rope::RotaryEmbedding;

use crate::attention::SparseAttention;
use crate::config::MsaConfig;
use crate::index_branch::IndexBranch;
use crate::loss::KlAlignmentLoss;
use crate::topk::TopKSelector;

pub struct MsaOutput {
    pub output: Tensor<3>,
    pub kl_loss: Option<Tensor<1>>,
}

#[derive(Module, Debug)]
pub struct MsaModule {
    pub index_branch: IndexBranch,
    pub attention: SparseAttention,
    #[module(skip)]
    pub rope: Option<RotaryEmbedding>,
    #[module(skip)]
    pub cfg: MsaConfig,
}

impl MsaModule {
    pub fn new(cfg: &MsaConfig, device: &Device) -> Self {
        let rope = if cfg.use_rope {
            Some(RotaryEmbedding::new(
                cfg.d_model,
                cfg.n_heads_q,
                cfg.rope_max_seq_len,
                cfg.rope_base,
                device,
            ))
        } else {
            None
        };
        Self {
            index_branch: IndexBranch::new(cfg, device),
            attention: SparseAttention::new(cfg, device),
            rope,
            cfg: cfg.clone(),
        }
    }

    pub fn forward<B: Backend>(&self, x: Tensor<3>) -> MsaOutput
    where
        burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
    {
        self.forward_cross::<B>(x.clone(), x)
    }

    pub fn forward_cross<B: Backend>(
        &self,
        hidden_states: Tensor<3>,
        kv_states: Tensor<3>,
    ) -> MsaOutput
    where
        burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
    {
        let (output, kl_loss) = self.forward_internal::<B>(hidden_states, kv_states);
        MsaOutput {
            output,
            kl_loss: if self.cfg.use_kl_loss {
                Some(kl_loss)
            } else {
                None
            },
        }
    }

    fn forward_internal<B: Backend>(
        &self,
        hidden_states: Tensor<3>,
        kv_states: Tensor<3>,
    ) -> (Tensor<3>, Tensor<1>)
    where
        burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
    {
        let cfg = &self.cfg;

        let idx_input = if cfg.gradient_detach {
            hidden_states.clone().detach()
        } else {
            hidden_states.clone()
        };
        let kv_idx_input = if cfg.gradient_detach {
            kv_states.clone().detach()
        } else {
            kv_states.clone()
        };

        let (q_idx, k_idx) =
            self.index_branch
                .forward(idx_input, kv_idx_input, cfg.n_heads_kv, cfg.d_idx);
        let scale = (cfg.d_idx as f64).sqrt();
        let block_scores =
            self.index_branch
                .compute_block_scores(q_idx, k_idx, cfg.block_size, scale, cfg.causal);

        let selector = TopKSelector::new(cfg.topk, cfg.block_size, cfg.force_local_block);
        let block_indices = selector.select::<B>(block_scores.clone());

        let q = self.attention.q_proj.forward(hidden_states);
        let k = self.attention.k_proj.forward(kv_states.clone());
        let v = self.attention.v_proj.forward(kv_states);
        let (q, k) = if let Some(rope) = &self.rope {
            let cos = rope.cos.clone();
            let sin = rope.sin.clone();
            let q = burn_rope::apply_rope_3d::<B>(q, cos.clone(), sin.clone(), cfg.n_heads_q);
            let k = burn_rope::apply_rope_3d::<B>(k, cos, sin, cfg.n_heads_kv);
            (q, k)
        } else {
            (q, k)
        };
        let (output, block_attn) =
            self.attention
                .forward_sparse_with_weights::<B>(q, k, v, block_indices.clone());

        let kl_loss = if cfg.use_kl_loss {
            KlAlignmentLoss
                .compute::<B>(block_scores, block_attn, block_indices)
                .mul_scalar(cfg.kl_coeff)
        } else {
            Tensor::zeros([1], &output.device())
        };
        (output, kl_loss)
    }

    pub fn forward_dense<B: Backend>(
        &self,
        hidden_states: Tensor<3>,
        kv_states: Tensor<3>,
    ) -> Tensor<3> {
        let q = self.attention.q_proj.forward(hidden_states);
        let k = self.attention.k_proj.forward(kv_states.clone());
        let v = self.attention.v_proj.forward(kv_states);
        self.attention.forward_dense::<B>(q, k, v)
    }
}
