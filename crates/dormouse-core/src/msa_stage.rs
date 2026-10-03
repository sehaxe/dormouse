//! The MSA stage: Qwen Sparse Attention as an extra per-iteration stage of
//! the LoopBlock (tech report §QSA Eq. 12-19; ADR-0014 re-entry).
//!
//! Per EXECUTED iteration:
//! - the compressed MQA indexer scores the sequence's complete blocks from
//!   the same block-body input the KDA arm reads (`normed_attn`);
//! - stage (a) (`msa_distill_weight > 0`): the stage's own multi-head
//!   softmax attention runs DENSE (the teacher), its head-summed attention
//!   distribution is max-pooled into blocks and an UNSCALED KL term is
//!   returned for the aux seam (Eq. 17-18); the sparse core contributes
//!   nothing on that stage;
//! - stage (b): the top-`msa_kb` blocks (plus the ALWAYS-INCLUDED final
//!   incomplete block) gate a micro-block mask over the same attention,
//!   ONE softmax over the masked scores; the output joins the block body
//!   BEFORE the residual aggregation - i.e. the residual-rewriting arms
//!   (GR/AttnRes/mHC) see it with everything else, which keeps MSA out of
//!   the "two arms claim one residual statement" refusals entirely.
//!
//! Every learnable is routed to [`Group::Rest`] by [`LoopBlock`]'s own
//! `rest_of` walk (they are small dense linears and a gain vector; the
//! report's head-wise Muon is the BACKBONE attention's policy, and the
//! indexer trains at its own LR in stage (a) - a queued trainer cell, not
//! a group here). The scalar [`MsaStage::scale`] inits at **1** by the
//! same rule `residual_scale` carries: at 0 the arm's body has exactly
//! zero gradient.
use burn::module::{Module, Param};
use burn::tensor::Tensor;

use crate::param::LinearLike;

#[derive(Module, Debug)]
pub struct MsaStage {
    /// The compressed MQA block indexer (learnables: Wq/Wk + norm gains).
    pub indexer: burn_msa::IndexerModule,
    pub wq: LinearLike,
    pub wk: LinearLike,
    pub wv: LinearLike,
    pub wo: LinearLike,
    /// The arm's own ReZero-style scalar, init 1: at 0 the body's gradient
    /// is exactly zero (`residual_scale`'s written-down rule).
    pub scale: Param<Tensor<1>>,
    /// Complete blocks the indexer picks (the config's `msa_kb`).
    pub kb: usize,
    /// True = stage (a): compute the teacher + KL, skip the sparse core.
    pub distill: bool,
    /// Block compression ratio (the config's `msa_block_r`).
    pub block_r: usize,
    /// The CORE attention's head geometry - the model's own, so the sparse
    /// arm and the KDA arm read the same widths.
    pub n_heads: usize,
    pub head_dim: usize,
}

/// What one MSA forward hands back: the (possibly zero) stage output that
/// joins the block body, and the UNSCALED distillation KL (stage (a)).
pub struct MsaOut {
    /// `[b, t, d]` — zero on the distill stage.
    pub output: Tensor<3>,
    /// Unscaled KL; `None` on the sparse stage.
    pub distill: Option<Tensor<1>>,
}

impl MsaStage {
    pub fn new(
        cfg: &crate::config::DormouseConfig,
        indexer: burn_msa::IndexerModule,
        device: &burn::tensor::Device,
    ) -> Self {
        Self {
            indexer,
            wq: LinearLike::dense(cfg.d_model, cfg.n_heads * cfg.head_dim, device),
            wk: LinearLike::dense(cfg.d_model, cfg.n_heads * cfg.head_dim, device),
            wv: LinearLike::dense(cfg.d_model, cfg.n_heads * cfg.head_dim, device),
            wo: LinearLike::dense(cfg.n_heads * cfg.head_dim, cfg.d_model, device),
            scale: Param::from_tensor(Tensor::ones([1], device)),
            kb: cfg.msa_kb,
            distill: cfg.msa_distill_weight > 0.0,
            block_r: cfg.msa_block_r,
            n_heads: cfg.n_heads,
            head_dim: cfg.head_dim,
        }
    }

    /// Eq. 12-19 on `x [b, t, d]` (the block body's normalized attention
    /// input): indexer scores, the pick, one masked softmax.
    pub fn forward<B: burn::backend::AutodiffBackend>(&self, x: Tensor<3>) -> MsaOut
    where
        burn::tensor::DispatchTensor:
            burn::backend::DispatchKindConversion<B>
            + burn::backend::DispatchKindConversion<B::InnerBackend>
            + burn::backend::DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        let [b, t, d] = x.dims();
        let dev = x.device();
        let scores = burn_msa::indexer_scores(&self.indexer, x.clone());

        if self.distill {
            // Stage (a): dense teacher + Eq. 17/18 KL; `distill_loss` owns
            // the visible-only trim ("only complete key blocks are
            // included in the KL loss for each query").
            let (q, k, v) = self.project::<B>(&x);
            let (_, dist) = burn_msa::dense_attention(q, k, v);
            let loss =
                burn_msa::distill_loss(scores, burn_msa::pool_teacher(dist, self.block_r), self.block_r);
            MsaOut {
                output: Tensor::zeros([b, t, d], &dev),
                distill: Some(loss),
            }
        } else {
            let (q, k, v) = self.project::<B>(&x);
            let picks = burn_msa::select_blocks(scores, self.kb);
            let mixed = burn_msa::sparse_attention(q, k, v, self.block_r, &picks); // [b, t, h*hd]
            let out = self
                .wo
                .forward::<B>(mixed.reshape([b * t, self.n_heads * self.head_dim]))
                .reshape([b, t, d]);
            MsaOut {
                output: out * self.scale.val().clone().reshape([1, 1, 1]),
                distill: None,
            }
        }
    }

    /// The stage's own Q/K/V projections, head-split (`[b, t, h, hd]`).
    /// The fp32-compute rule every Linear follows is LinearLike's owner
    /// seam (`--bf16` stores bf16, computes fp32).
    fn project<B: burn::backend::AutodiffBackend>(
        &self,
        x: &Tensor<3>,
    ) -> (Tensor<4>, Tensor<4>, Tensor<4>)
    where
        burn::tensor::DispatchTensor:
            burn::backend::DispatchKindConversion<B>
            + burn::backend::DispatchKindConversion<B::InnerBackend>
            + burn::backend::DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        let [b, t, dd] = x.dims();
        let (h, hd) = (self.n_heads, self.head_dim);
        let flat = x.clone().reshape([b * t, dd]);
        let q = self.wq.forward::<B>(flat.clone()).reshape([b, t, h, hd]);
        let k = self.wk.forward::<B>(flat.clone()).reshape([b, t, h, hd]);
        let v = self.wv.forward::<B>(flat).reshape([b, t, h, hd]);
        (q, k, v)
    }
}
