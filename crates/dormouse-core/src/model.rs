//! model - Embedding -> LoopBlock -> RMSNorm -> lm_head, all-bf16 capable
use burn::backend::DispatchKindConversion;
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::{Device, DispatchTensor, FloatDType, Int, Tensor};
use burn_rmsnorm::RMSNorm;

use crate::aux::AuxHeads;
use crate::config::DormouseConfig;
use crate::loop_block::LoopBlock;
use crate::param::LinearLike;

#[derive(Module, Debug)]
pub struct DormouseModel {
    pub embedding: Embedding,
    pub loop_block: LoopBlock,
    pub norm: RMSNorm,
    pub lm_head: LinearLike,
    pub aux: AuxHeads,
    #[module(skip)]
    pub vocab_size: usize,
    #[module(skip)]
    pub d_model: usize,
    #[module(skip)]
    pub bf16: bool,
    #[module(skip)]
    pub jepa_weight: f32,
    #[module(skip)]
    pub jepa_mask_frac: f32,
    #[module(skip)]
    pub jepa_mask_span: usize,
    #[module(skip)]
    pub dspark_weight: f32,
    #[module(skip)]
    pub dspark_k: usize,
    #[module(skip)]
    pub dspark_stride: usize,
    /// MoR router BCE weight (arXiv 2507.10524). The label is the router's
    /// own top-k recomputed on the current batch, so it cannot go stale; with
    /// the arm off there is nothing to weight.
    #[module(skip)]
    pub mor_bce_weight: f32,
    #[module(skip)]
    pub max_seq_len: usize,
}

impl DormouseModel {
    pub fn new(cfg: &DormouseConfig, device: &Device) -> Self {
        let d = cfg.d_model;
        let v = cfg.vocab;
        Self {
            embedding: EmbeddingConfig::new(v, d).init(device),
            loop_block: LoopBlock::new(cfg, device),
            norm: RMSNorm::new(d, cfg.norm_eps, device),
            lm_head: LinearLike::with_tsct(d, v, cfg.rank.min(d).min(v), cfg.use_tsct, device),
            aux: AuxHeads::new(d, v, cfg.rank, device),
            vocab_size: v,
            d_model: d,
            bf16: cfg.bf16,
            jepa_weight: cfg.jepa_weight,
            jepa_mask_frac: cfg.jepa_mask_frac,
            jepa_mask_span: cfg.jepa_mask_span,
            dspark_weight: cfg.dspark_weight,
            dspark_k: cfg.dspark_k,
            dspark_stride: cfg.dspark_stride,
            mor_bce_weight: cfg.mor_bce_weight,
            max_seq_len: cfg.max_seq_len,
        }
    }

    pub fn forward<B: burn::backend::AutodiffBackend>(
        &self,
        input_ids: Tensor<2, Int>,
        hashed_ids: Option<Tensor<3, Int>>,
    ) -> Tensor<3>
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        self.forward_with_hidden::<B>(input_ids, hashed_ids, None, None, None).0
    }

    /// Teacher pass: the pre-head accumulated latent `out_acc` only (no
    /// lm_head, no losses). Used for the EMA JEPA teacher.
    pub fn forward_latent<B: burn::backend::AutodiffBackend>(
        &self,
        input_ids: Tensor<2, Int>,
        hashed_ids: Option<Tensor<3, Int>>,
        host_rows: Option<Tensor<3>>,
    ) -> Tensor<3>
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        let x = self.embedding.forward(input_ids);
        let x = if self.bf16 {
            x.cast(FloatDType::BF16)
        } else {
            x
        };
        let (out_acc, _rec, _kda, _mor) = self.loop_block.forward_full_state::<B>(
            x, hashed_ids, host_rows, None, None, &self.lm_head,
        );
        out_acc
    }

    /// Returns (logits, L_Rec [1], kda, aux Option<[1]>). When
    /// `targets` is Some, the per-step reconstruction loss is accumulated
    /// inside the loop block so the model never slices a 4D autodiff tensor
    /// (cubecl/sm_120 stability). `host_rows` carries pre-gathered n-gram
    /// rows `[b,t,3*32]` for the RAM-offload path; when None, `hashed_ids`
    /// drives the in-model tables. `teacher` (the EMA copy) enables the JEPA
    /// term (a second full forward per step - use
    /// [`Self::forward_with_jepa_targets`] with precomputed latents to drop
    /// it from the hot loop); `aux` is the weight-combined auxiliary loss
    /// (JEPA + DSpark), None when every aux weight is 0 or `targets` is None.
    pub fn forward_with_hidden<B: burn::backend::AutodiffBackend>(
        &self,
        input_ids: Tensor<2, Int>,
        hashed_ids: Option<Tensor<3, Int>>,
        host_rows: Option<Tensor<3>>,
        targets: Option<Tensor<2, Int>>,
        teacher: Option<&Self>,
    ) -> (Tensor<3>, Tensor<1>, Tensor<4>, Option<Tensor<1>>)
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        // The teacher consumes EXACTLY the student's three inputs, so the JEPA
        // target is the student's own latent under EMA weights and the teacher
        // is the student's architecture with different weights (same arms
        // live). This is the definition `precompute_jepa_targets` implements
        // offline (`forward_latent(x, Some(h), None)`): the two paths must
        // agree or `--jepa-targets` is a different objective. NOT `targets` -
        // that is the one-byte-shifted LABEL sequence, and `None` keys leave
        // the Engram arm inert (no keys -> the memory branch is zeros), i.e. a
        // structurally different network on a different input; that was this
        // function's state from the aux objectives' first commit, and no
        // finite/closeness test could see it. Pinned by
        // `teacher_target_is_the_student_latent` in tests/jepa_teacher_seam.rs.
        //
        // `host_rows` is detached: same values, no grad-carrying leaf. The
        // teacher's params are no_grad-frozen, so a live row leaf would put a
        // second full forward on the tape (~2x activations, the step-0 OOM
        // axis). The RAM tables have no EMA copy (they live on the host,
        // outside the Module), so both sides read the same rows there.
        let teacher_latent = teacher.zip(targets.as_ref()).map(|(t, _)| {
            t.forward_latent::<B>(
                input_ids.clone(),
                hashed_ids.clone(),
                host_rows.clone().map(|r| r.detach()),
            )
        });
        self.forward_with_latent::<B>(
            input_ids, hashed_ids, host_rows, targets, teacher_latent,
        )
    }

    /// Same as [`Self::forward_with_hidden`], but the JEPA target latent is
    /// supplied precomputed (offline teacher targets) instead of running an
    /// EMA-teacher forward per step: no second forward, no EMA advance in
    /// the training loop.
    pub fn forward_with_jepa_targets<B: burn::backend::AutodiffBackend>(
        &self,
        input_ids: Tensor<2, Int>,
        hashed_ids: Option<Tensor<3, Int>>,
        host_rows: Option<Tensor<3>>,
        targets: Option<Tensor<2, Int>>,
        jepa_target: Option<Tensor<3>>,
    ) -> (Tensor<3>, Tensor<1>, Tensor<4>, Option<Tensor<1>>)
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        self.forward_with_latent::<B>(
            input_ids, hashed_ids, host_rows, targets, jepa_target,
        )
    }

    fn forward_with_latent<B: burn::backend::AutodiffBackend>(
        &self,
        input_ids: Tensor<2, Int>,
        hashed_ids: Option<Tensor<3, Int>>,
        host_rows: Option<Tensor<3>>,
        targets: Option<Tensor<2, Int>>,
        teacher_latent: Option<Tensor<3>>,
    ) -> (Tensor<3>, Tensor<1>, Tensor<4>, Option<Tensor<1>>)
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        // DSpark's window tokens only (the JEPA teacher's input is
        // `input_ids` above, see `forward_with_hidden`). KNOWN, NOT FIXED:
        // these are the LABEL sequence, so the draft head's step s is fed
        // x[p+s+1] and trained to emit x[p+s+2] - a one-position shift
        // against the sequence the model actually consumed. Changing it
        // changes every DSpark number, so it needs its own A/B, not a
        // drive-by in a JEPA bugfix.
        let ids_raw = targets.clone();
        let x = self.embedding.forward(input_ids);
        let x = if self.bf16 {
            x.cast(FloatDType::BF16)
        } else {
            x
        };
        let [b, t, _d] = x.dims();
        // Indices for the in-loop L_Rec gather (no one-hot [b*t,v] tensor).
        let tgt = targets.map(|tg| tg.reshape([b * t, 1]));
        let (out_acc, rec, kda, mor_aux) =
            self.loop_block
                .forward_full_state::<B>(x, hashed_ids, host_rows, None, tgt, &self.lm_head);
        // loop activations may be bf16; the final norm+head compute in fp32
        // (bf16 logits make the softmax/CE numerically unstable -> NaN).
        let h = if self.bf16 {
            self.norm.forward(out_acc.clone().cast(FloatDType::F32))
        } else {
            self.norm.forward(out_acc.clone())
        };
        let logits = self
            .lm_head
            .forward::<B>(h.clone().reshape([b * t, self.d_model]))
            .reshape([b, t, self.vocab_size]);
        let aux = self.aux_loss::<B>(&out_acc, teacher_latent, ids_raw, &h, &logits, mor_aux);
        (logits, rec, kda, aux)
    }

    /// Weight-combined auxiliary loss (JEPA + DSpark). None when every aux
    /// weight is 0, when there are no targets, or when JEPA is on but no
    /// teacher latent was supplied. `teacher_latent` is either the live EMA
    /// teacher's out_acc (online) or a precomputed frozen target (offline);
    /// both are detached here - the latent is a stop-grad target.
    fn aux_loss<B: burn::backend::AutodiffBackend>(
        &self,
        student_latent: &Tensor<3>,
        teacher_latent: Option<Tensor<3>>,
        ids: Option<Tensor<2, Int>>,
        h: &Tensor<3>,
        logits: &Tensor<3>,
        mor_aux: Option<Tensor<1>>,
    ) -> Option<Tensor<1>>
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        // The original gate, kept exactly (dspark_k == 0 disables the channel),
        // plus the MoR term as a third way in. `any` is the pre-existing
        // condition, so no arm changes behaviour when MoR is off.
        let any = (self.jepa_weight > 0.0 || self.dspark_weight > 0.0) && self.dspark_k > 0;
        let mor = self.mor_bce_weight > 0.0;
        if !any && !mor {
            return None;
        }
        let ids = ids?;
        let dev = h.device();
        let mut total: Option<Tensor<1>> = None;
        // MoR first: the router's own BCE against its top-k recomputed on this
        // batch, the one auxiliary the MoR arm has under a pure-CE recipe.
        if let Some(m) = mor_aux {
            crate::probe::note(crate::probe::MOR_BCE);
            total = Some(m.mul_scalar(self.mor_bce_weight)
                + total.unwrap_or_else(|| Tensor::zeros([1], &dev)));
        }
        if self.jepa_weight > 0.0 {
            if let Some(tl) = teacher_latent {
                let j = crate::aux::jepa_aux_loss(
                    &self.aux.jepa_pred,
                    student_latent.clone(),
                    tl.detach(),
                    self.jepa_mask_frac,
                    self.jepa_mask_span,
                );
                total = Some(j.mul_scalar(self.jepa_weight)
                    + total.unwrap_or_else(|| Tensor::zeros([1], &dev)));
            }
        }
        if self.dspark_weight > 0.0 {
            let d = crate::aux::dspark_aux_loss(
                &self.aux.dspark,
                &self.aux.conf,
                h.clone(),
                logits.clone(),
                ids,
                self.dspark_k,
                self.dspark_stride,
            );
            total = Some(d.mul_scalar(self.dspark_weight)
                + total.unwrap_or_else(|| Tensor::zeros([1], &dev)));
        }
        total
    }

    /// (checked, non-finite) over a sample of parameter tensors, read back
    /// once. Used to refuse a corrupt checkpoint at load time instead of
    /// training a GPU-hour into a NaN (measured 2026-09-27: a run that died
    /// while the allocator was failing saved those weights, and every resume
    /// replayed the corruption). A sample is enough: a corrupt save is NaN
    /// everywhere, not in one tensor.
    pub fn finite_scan(&self) -> (usize, usize) {
        // LOUD on a failed readback: `unwrap_or_default()` turned a device
        // error into an EMPTY vector, so `checked = 0, bad = 0` and this
        // guard - the one that refuses a corrupt checkpoint at load time -
        // reported "clean" for weights it never looked at (ADR-0019).
        let read = |v: Result<Vec<f32>, burn::tensor::DataError>| -> Vec<f32> {
            v.expect("finite_scan: reading a parameter back from the device failed - \
                      the checkpoint cannot be judged, refusing to say it is clean")
        };
        let vecs = [
            read(self.embedding.weight.val().clone().into_data().try_to_vec::<f32>()),
            read(self.norm.weight.val().clone().into_data().try_to_vec::<f32>()),
            read(self.loop_block.residual_scale.val().clone().into_data().try_to_vec::<f32>()),
        ];
        let checked = vecs.iter().map(|v| v.len()).sum();
        let bad = vecs.iter().map(|v| v.iter().filter(|x| !x.is_finite()).count()).sum();
        (checked, bad)
    }

    /// Random-depth arm (ADR-0013 rank 2): run only the first `n` loop
    /// iterations, `None` for the fixed-depth default. Cheapest way to get
    /// adaptive depth without a learned halting head - and nothing to
    /// collapse. Rejects an out-of-range depth loudly.
    pub fn set_loop_depth(&mut self, n: Option<usize>) {
        self.loop_block.set_depth(n);
    }

    /// The training loss: the honest mean CE from the loop (ADR-0013).
    pub fn loss<B: burn::backend::AutodiffBackend>(&self, rec_ce: Tensor<1>) -> Tensor<1>
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        rec_ce
    }

    /// Apply a quantization format to every TSCT factor in the model
    /// (experts, readout projections, lm_head).
    pub fn set_quant_all(&mut self, quant: burn_spectral::QuantFormat) {
        self.loop_block.set_quant_all(quant);
        self.loop_block.out_proj.set_quant(quant);
        self.lm_head.set_quant(quant);
    }

    /// Toggle the bf16 matmul path (fp32 graph, tensor-core forward) on
    /// every TSCT factor. Call under the model's BF16 mode.
    pub fn set_bf16_compute(&mut self, on: bool) {
        self.loop_block.set_bf16_compute(on);
        self.lm_head.set_bf16_compute(on);
    }

    /// Polar-retract every TSCT factor U/V to orthonormal (on device, keeps
    /// autodiff tracking). Call every step during training: without it the
    /// factors drift and the quantized forward degrades into NaN.
    pub fn retract_tsct(&mut self, iters: usize) {
        self.loop_block.retract_tsct(iters);
        self.lm_head.retract(iters);
    }

    /// Worst orthonormality error across all TSCT factors (syncs the device;
    /// monitor at cadence, not per step).
    pub fn max_ortho(&self) -> f32 {
        self.loop_block.max_ortho().max(self.lm_head.max_ortho())
    }

    // NOTE: there is deliberately NO `forward_bytes(bytes) -> logits` here.
    // It was deleted 2026-09-28 and it is the bug this comment is here to stop
    // recurring. A bytes->logits entry point in the model has to derive the
    // Engram keys, the keys are derived by `dormouse_data::raw_keys`, and
    // `dormouse-core` cannot reach that crate without either a dependency edge
    // this crate should not have or a second copy of the FNV. So the one had
    // `hashed_ids = None`, `loop_block`'s memory branch took its
    // `None => None` arm, and `generate`/`serve` shipped a network whose
    // memory arm contributed LITERAL ZEROS: a confident, wrong, unlogged
    // product, in the only two binaries a user ever sees. The seam that both
    // call now lives where the model and the data crate meet:
    // `dormouse_train::decode::next_byte_logits`.

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}