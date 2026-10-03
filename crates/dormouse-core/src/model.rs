//! model - Embedding -> LoopBlock -> RMSNorm -> lm_head, all-bf16 capable
//!
//! # The whole architecture, and where each piece lives
//!
//! ```text
//! input_ids [b, t] --Embedding(v, d)--> h0
//!   h0 --> LoopBlock (weight-shared, max_iter passes) --> out_acc [b, t, d]
//!   out_acc --> RMSNorm --> lm_head (LinearLike, d -> v) --> logits [b, t, v]
//! ```
//!
//! The loop carries three losses: the honest unweighted CE over iterations
//! (`L_Rec`, ADR-0013 — no PonderNet, no halt head, no KL term), and the
//! optional auxiliary terms from [`crate::aux`] (JEPA+KoLeo, DSpark,
//! future-byte, MoR BCE, MoE load-balance). Only `L_Rec` is the objective;
//! the others are additions with their own weights, all zeroable.
//!
//! # Invariants this type holds
//!
//! * **`lm_head` and the final norm run in fp32 even under `--bf16`.** bf16
//!   logits NaN, and it is the single most common shape of "the loss went NaN
//!   at step 3" (AGENTS.md §2.2). Loop-internal `L_Rec` logits are fp32 for
//!   the same reason.
//! * **No host-device synchronization** in any forward here. Every branch is on
//!   a `#[module(skip)]` config field, never on a device value
//!   (ADR-0018 rule 2). The one decision that needs a device value is made on
//!   the device.
//! * **An off auxiliary arm does not exist**, not merely has weight 0: the
//!   future-byte head is `Option` and built iff `aux_fb_weight > 0`, so a
//!   pure-CE run's parameter set and checkpoint are byte-identical to a build
//!   from before the arm shipped. That is what keeps the pure-CE control valid.
//! * **`max_seq_len` is a capacity statement, not a truncation.** A longer
//!   sequence is a shape error, never a silent crop — silent cropping is how a
//!   "512-token context" run quietly trains on 512 of 4096.
//!
//! # The three forward shapes
//!
//! [`DormouseModel::forward`] is logits only (inference, tests, the quant
//! probe). [`DormouseModel::forward_latent`] is the pre-head latent only — the
//! EMA teacher's pass, no `lm_head`, no losses.
//! [`DormouseModel::forward_with_hidden`] is the real one:
//! logits plus every loss term plus the carried KDA state. They share
//! `forward_with_hidden` internally; the two thin wrappers exist so a caller
//! cannot accidentally build a graph it does not want (the teacher's pass in
//! particular must not be grad-tracked).

use burn::backend::DispatchKindConversion;
use burn::module::{Module, Param, ParamId, ParamMapper};
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::{Device, DispatchTensor, FloatDType, Int, Tensor};
use burn_rmsnorm::RMSNorm;

use crate::aux::AuxHeads;
use crate::config::DormouseConfig;
use crate::loop_block::{LoopBlock, RouteAux};
use crate::param::LinearLike;

/// The model: an embedding, a weight-shared loop, a norm, a head, and the
/// auxiliary heads — plus the config constants the forward needs.
///
/// Every field is `pub` because `#[derive(Module)]` walks the field list to
/// build the checkpoint record, and the `#[module(skip)]` ones because the
/// forward reads config constants rather than being handed the config. The
/// module tree IS the checkpoint format: renaming or reordering a
/// parameter-bearing field changes every `.bin` on disk. The non-parameter
/// fields can change without invalidating a checkpoint, and the resume
/// config-equality check (ADR-0005/0021) is what stops them changing silently.
#[derive(Module, Debug)]
pub struct DormouseModel {
    /// Byte -> residual vector, `[vocab, d_model]`. 256 rows: the vocabulary is
    /// bytes, not tokens, so there is no tokenizer and no vocabulary growth
    /// problem.
    pub embedding: Embedding,
    /// The weight-shared loop. One instance, run `max_iter` times — not
    /// `max_iter` blocks. Its `e_k` iteration embedding is what distinguishes
    /// the passes.
    pub loop_block: LoopBlock,
    /// Final RMSNorm, applied to `out_acc` before the head. **fp32 even under
    /// `--bf16`**: bf16 logits NaN, and the norm is where that starts.
    pub norm: RMSNorm,
    /// `d_model -> vocab`. Spectral (TSCT) when `use_tsct`, with rank
    /// `min(rank, d_model, vocab)` — the head is `d_model x 256`, so an
    /// unclamped `rank` above either dimension is not a compression at all but
    /// a wider matrix wearing a rank's name.
    pub lm_head: LinearLike,
    /// THE COMPAT CHANNEL (lane bf-compat; `use_byteflow` + `use_kda` +
    /// `--opt mix`). `Some` iff [`crate::bf::compat_armed`] — the one list
    /// `config::validate` refuses against and the trainer's dispatch reads —
    /// so the model, the validator and the dispatch cannot disagree; the
    /// checkpoint then carries the byteflow front/back record plus the two
    /// pairing Linears. `None` = the pure-byteflow net (byte identical to its
    /// pre-compat build) or the plain dormouse model.
    pub bf: Option<crate::bf::BfChannel>,
    /// The auxiliary heads (JEPA predictor, DSpark draft + acceptance,
    /// future-byte). ON by default; the weights live in the `#[module(skip)]`
    /// fields below.
    pub aux: AuxHeads,
    /// `vocab`, for the shape assertions and the generate path.
    #[module(skip)]
    pub vocab_size: usize,
    /// `d_model`, for the reshape back to `[b, t, d]`.
    #[module(skip)]
    pub d_model: usize,
    /// Cast activations to bf16 at the arm boundaries. **Slower than fp32 on
    /// this backend** — there is no bf16 tensor-core path (§2.1) — and
    /// mixed-dtype ops NaN, which is why the head and the final norm stay fp32.
    #[module(skip)]
    pub bf16: bool,
    /// JEPA term weight. `0.0` is the pure-CE baseline (A/B queue row 1).
    #[module(skip)]
    pub jepa_weight: f32,
    /// Expected masked fraction for the JEPA span mask; see
    /// [`crate::aux::mask_from`] for why the field means what it says.
    #[module(skip)]
    pub jepa_mask_frac: f32,
    /// Masked span length in byte positions.
    #[module(skip)]
    pub jepa_mask_span: usize,
    /// DSpark term weight. `0.0` in the shipped recipe — the first DSpark
    /// measurement this project has made, because the window was one position
    /// shifted until 2026-09-29.
    #[module(skip)]
    pub dspark_weight: f32,
    /// DSpark draft horizon `k`.
    #[module(skip)]
    pub dspark_k: usize,
    /// DSpark anchor spacing in byte positions. Refused at 0: it collapses the
    /// k-step window into k copies of one CE.
    #[module(skip)]
    pub dspark_stride: usize,
    /// Future-byte auxiliary head: weight and horizon, in BYTE POSITIONS
    /// (`crate::future_byte` for what the term is and which labels it reads).
    #[module(skip)]
    pub aux_fb_weight: f32,
    /// How far ahead the future-byte head predicts, in byte positions. **Must be
    /// `>= 1`** and defaults to 2: the label for position `q` is
    /// `targets[q + k]`, so `k = 0` supervises the head on the byte the MAIN
    /// CE already predicts — a second CE through an independent head, which
    /// trains, descends, costs a step, and is not multi-token prediction.
    /// Refused by `config::validate`. One horizon, not a set: a head per
    /// horizon is a later row in the A/B queue, not a field with a list in it.
    #[module(skip)]
    pub aux_fb_horizon: usize,
    /// MoR router BCE weight (arXiv 2507.10524). The label is the router's
    /// own top-k recomputed on the current batch, so it cannot go stale; with
    /// the arm off there is nothing to weight.
    #[module(skip)]
    pub mor_bce_weight: f32,
    /// Load-balancing coefficient for the sparse-routing arm (`moe::lb_aux`).
    /// 0.0 = off, and the only legal non-zero position is with `moe_topk > 0`
    /// - `config::validate` refuses the combination rather than ignoring the
    ///   term. Deliberately NOT a large-MoE default: see
    ///   `config::schema::moe_lb_coef`.
    #[module(skip)]
    pub moe_lb_coef: f32,
    /// Longest sequence the model will be asked about, in byte positions. A
    /// capacity statement: RoPE table size and the trainer's shape checks read
    /// it, and a longer sequence is a shape error rather than a silent crop.
    #[module(skip)]
    pub max_seq_len: usize,
}

impl DormouseModel {
    /// Build the model from a validated config.
    ///
    /// Does not call [`crate::config::validate`] — the caller owns that, and
    /// the trainer calls it once on the fully-resolved config.
    ///
    /// Two construction decisions a reader should know:
    ///
    /// * The future-byte head is attached **iff** `aux_fb_weight > 0`, matching
    ///   the `cfg.use_gr.then(..)` shape in [`LoopBlock::new`]. An off-arm
    ///   model is therefore byte-identical to a build from before the arm
    ///   existed, which is what keeps queue row 1 (pure CE) a valid control and
    ///   the measured preset parameter counts true.
    /// * `lm_head`'s rank is clamped to `min(rank, d_model, vocab)`. Without
    ///   the clamp a `rank` above the head's own dimensions is not a low-rank
    ///   factorization at all — it is a wider matrix with a rank's name on it.
    pub fn new(cfg: &DormouseConfig, device: &Device) -> Self {
        let d = cfg.d_model;
        let v = cfg.vocab;
        let mut aux = AuxHeads::new(d, v, cfg.rank, device);
        // The future-byte head exists iff its weight is non-zero - the same
        // conditional construction as `LoopBlock::gr`, and the reason a
        // zero-weight run's parameters and checkpoint are today's.
        aux.fb = (cfg.aux_fb_weight > 0.0).then(|| LinearLike::dense(d, v, device));
        Self {
            embedding: EmbeddingConfig::new(v, d).init(device),
            loop_block: LoopBlock::new(cfg, device),
            norm: RMSNorm::new(d, cfg.norm_eps, device),
            lm_head: LinearLike::with_tsct(d, v, cfg.rank.min(d).min(v), cfg.use_tsct, device),
            bf: crate::bf::bf_for_model(cfg, device),
            aux,
            vocab_size: v,
            d_model: d,
            bf16: cfg.bf16,
            jepa_weight: cfg.jepa_weight,
            jepa_mask_frac: cfg.jepa_mask_frac,
            jepa_mask_span: cfg.jepa_mask_span,
            dspark_weight: cfg.dspark_weight,
            dspark_k: cfg.dspark_k,
            dspark_stride: cfg.dspark_stride,
            aux_fb_weight: cfg.aux_fb_weight,
            aux_fb_horizon: cfg.aux_fb_horizon,
            mor_bce_weight: cfg.mor_bce_weight,
            moe_lb_coef: cfg.moe_lb_coef,
            max_seq_len: cfg.max_seq_len,
        }
    }

    /// Logits only: `[b, t]` byte ids in, `[b, t, vocab]` logits out.
    ///
    /// The inference/test/quant-probe shape — no targets, so `L_Rec` is zero
    /// and no auxiliary term is built. Use
    /// [`Self::forward_with_hidden`] when you need a loss, and
    /// [`Self::forward_latent`] for the EMA teacher's pass.
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
        self.forward_with_hidden::<B>(input_ids, hashed_ids, None, None, None)
            .0
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
        let (out_acc, _rec, _kda, _route) = self.loop_block.forward_full_state::<B>(
            x,
            hashed_ids,
            host_rows,
            None,
            None,
            &self.lm_head,
        );
        out_acc
    }

    /// The real forward: logits, every loss term, and the carried KDA state.
    ///
    /// Returns `(logits, L_Rec [1], kda, aux Option<[1]>)`. When
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
        self.forward_with_latent::<B>(input_ids, hashed_ids, host_rows, targets, teacher_latent)
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
        self.forward_with_latent::<B>(input_ids, hashed_ids, host_rows, targets, jepa_target)
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
        // THE COMPAT CHANNEL (bf.rs). Everything below this line is the
        // byte-level path: dormouse embedding, in-loop L_Rec, aux heads. The
        // channel replaces all three, so it branches BEFORE them rather than
        // after — a `Some` channel that fell through to the embedding would be
        // two different networks in one function (ADR-0019).
        //
        // `hashed_ids` / `host_rows` / `teacher_latent` are dropped here, not
        // silently read: `validate` refuses `use_engram` and every aux weight
        // with `use_byteflow`, so on a validated run the keys belong to a loop
        // whose memory arm is off (as dead as on any `use_engram = false`
        // run) and there is no teacher to build a target from.
        if self.bf.is_some() {
            return self.forward_channel::<B>(input_ids, targets);
        }
        // DSpark's window tokens are the sequence the model CONSUMED, not the
        // label sequence. They used to be `targets`, which put the draft
        // head's step s one position ahead of its own base logits: it was fed
        // x[p+s+1] - the very byte `logits[p+s]` had just predicted - and
        // supervised toward x[p+s+2], while the hidden state it was given
        // (h[p+s]) had only seen x[0..=p+s]. The head was handed the answer.
        // `dspark_aux_loss`'s window arithmetic is unchanged and now reads
        // positions in the consumed sequence, where step s's base logits and
        // its CE target finally line up (FIXED 2026-09-29; this invalidates
        // every DSpark number recorded before it).
        //
        // The GATE is still `targets.is_some()`, not `ids`: it exists so a
        // decode forward (which passes no labels) builds no aux graph at
        // all, not to feed the window.
        let ids_raw = targets.as_ref().map(|_| input_ids.clone());
        let x = self.embedding.forward(input_ids);
        let x = if self.bf16 {
            x.cast(FloatDType::BF16)
        } else {
            x
        };
        let [b, t, _d] = x.dims();
        // Indices for the in-loop L_Rec gather (no one-hot [b*t,v] tensor).
        // `targets` is cloned for the aux terms: the future-byte head is the
        // one arm whose LABEL is `targets` read at an offset, so it cannot be
        // served off the label indices the loop consumed. [b,t] int64, 40 KB at
        // batch 10 x 512 - noise next to the [b,t,d] forward it feeds.
        let tgt = targets.clone().map(|tg| tg.reshape([b * t, 1]));
        let (out_acc, rec, kda, route) = self.loop_block.forward_full_state::<B>(
            x,
            hashed_ids,
            host_rows,
            None,
            tgt,
            &self.lm_head,
        );
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
        let aux = self.aux_loss::<B>(
            &out_acc,
            teacher_latent,
            ids_raw,
            targets,
            &h,
            &logits,
            &route,
        );
        (logits, rec, kda, aux)
    }

    /// THE COMPAT CHANNEL'S FORWARD (`crate::bf`): ByteFlow's front stage
    /// (embedding → local encoder → coding-rate Top-K chunker → proj) packs
    /// the T-byte window into K patch latents, `in_proj` pairs the width, the
    /// dormouse loop runs over those K positions, `out_proj` pairs it back,
    /// and `decode_chunks` lifts the K outputs to `[b, t, 256]` byte logits.
    ///
    /// THE OBJECTIVE IS THE BYTE CE FROM THE DECODER, not the loop's in-loop
    /// `L_Rec`: the loop is called with `targets = None` (its `lm_head`
    /// gather never builds) and `rec` is the mean CE of the decoder's logits
    /// against all T labels. Two reasons, both load-bearing:
    ///
    /// 1. `decode_chunks` is the model's only exit, so a loss taken at K
    ///    patch starts would leave `upsample_w`, `decoder` and `out` with no
    ///    gradient — a dead back stage behind a perfectly healthy loss curve
    ///    (the `8fa5d4c` shape: the arm looks alive because nothing that
    ///    would reveal it is measured).
    /// 2. The held-out eval scores `[b, t, 256]` logits against all T bytes
    ///    (`train/src/lib.rs`, the `ece` block), so a train CE averaged over
    ///    K patch starts would be a DIFFERENT quantity from the one the eval
    ///    line prints, and best-checkpoint-by-held-out would compare them.
    ///
    /// The dormant `norm` / `lm_head` / byte `embedding` take no gradient on
    /// this path and stay in the module tree for the reason `crate::bf`
    /// gives: the module tree is the checkpoint format. A missing gradient is
    /// not an error on the dormouse path (the pure-CE baseline's aux heads
    /// have none either).
    ///
    /// The channel computes fp32 end to end and does not read `self.bf16`:
    /// `byteflow::check` refuses `--bf16` for both byteflow modes, because
    /// ByteFlowNet has no bf16 handling (§2.1 — bf16 would be slower than
    /// fp32 here anyway).
    fn forward_channel<B: burn::backend::AutodiffBackend>(
        &self,
        input_ids: Tensor<2, Int>,
        targets: Option<Tensor<2, Int>>,
    ) -> (Tensor<3>, Tensor<1>, Tensor<4>, Option<Tensor<1>>)
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        let bf = self.bf.as_ref().expect("forward_channel without model.bf");
        // The aux weights are refused with `use_byteflow` by `validate`, and
        // the model constructors deliberately do NOT call `validate` — so the
        // same refusal is held here, at the runtime seam. Without it a
        // hand-built config with `jepa_weight > 0` and a channel armed would
        // have every aux weight silently ignored: this function returns
        // `aux = None` unconditionally, so the term would cost a config field
        // and produce nothing (the ADR-0019 class validate's doc names as
        // "a term with nothing to act on").
        assert!(
            self.jepa_weight == 0.0
                && self.dspark_weight == 0.0
                && self.aux_fb_weight == 0.0
                && self.mor_bce_weight == 0.0,
            "the byteflow compat channel carries no aux heads: validate refuses jepa_weight, \
             dspark_weight, aux_fb_weight and mor_bce_weight > 0 together with use_byteflow - \
             set them to 0"
        );
        crate::probe::note(crate::probe::BF_CHANNEL);
        let [b, t] = input_ids.dims();
        // Front stage. `sel` is a discrete argtopk product: the SELECTION has
        // no gradient path (the paper's own treatment of its chunker), the
        // gathered rows behind it do, so the encoder and `proj` train through
        // `z` and the boundaries stay fixed by the coding rate.
        let (h, z, sel) = bf.net.encode_chunks(input_ids, bf.k);
        let x = bf.in_proj.forward(z); // [b, k, d_model]
        let (out_acc, _loop_rec, kda, _route) =
            self.loop_block
                .forward_full_state::<B>(x, None, None, None, None, &self.lm_head);
        // Back stage: K latents → T byte positions, then the decoder blocks
        // and the byte head.
        let g = bf.out_proj.forward(out_acc); // [b, k, d_global]
        let logits = bf.net.decode_chunks(g, h, &sel); // [b, t, 256]
        let dev = logits.device();
        let rec = match targets {
            Some(tg) => {
                let [_, _, v] = logits.dims();
                burn::tensor::activation::log_softmax(logits.clone().reshape([b * t, v]), 1)
                    .gather(1, tg.reshape([b * t, 1]))
                    .neg()
                    .mean()
                    .reshape([1])
            }
            // An eval/decode forward: the same zero `L_Rec` the loop returns
            // with `targets = None` (loop_block.rs), a constant leaf. Nothing
            // reads it — the eval scores `logits` itself.
            None => Tensor::zeros([1], &dev),
        };
        (logits, rec, kda, None)
    }

    /// Weight-combined auxiliary loss (JEPA + DSpark + MoR BCE + the
    /// future-byte term). None when every aux weight is 0, when there are no
    /// targets, or when JEPA is on but no teacher latent was supplied.
    /// `teacher_latent` is either the live EMA teacher's out_acc (online) or a
    /// precomputed frozen target (offline); both are detached here - the latent
    /// is a stop-grad target. `ids` is the sequence the model CONSUMED
    /// (DSpark's window tokens) and `targets` the one-byte-shifted LABEL
    /// sequence - the future-byte head's labels, and the two are not
    /// interchangeable, which is what `8fa5d4c`-class bugs look like.
    // The parameter list IS the aux contract: each aux term reads a different
    // subset (JEPA the latents, DSpark ids+hidden+logits, the future-byte head
    // targets), so a grouping struct would only relocate the same eight names.
    #[allow(clippy::too_many_arguments)]
    fn aux_loss<B: burn::backend::AutodiffBackend>(
        &self,
        student_latent: &Tensor<3>,
        teacher_latent: Option<Tensor<3>>,
        ids: Option<Tensor<2, Int>>,
        targets: Option<Tensor<2, Int>>,
        h: &Tensor<3>,
        logits: &Tensor<3>,
        route: &RouteAux,
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
        // The future-byte head is its own way in and shares nothing with
        // `any`: it needs no teacher, no `dspark_k`, and its own counter.
        let fb = self.aux_fb_weight > 0.0;
        // The routing balancer, likewise. Gated on the WEIGHT here and on the
        // ARM there (`RouteAux.moe_lb` is `None` unless `moe_topk > 0`), so a
        // weight with no selection cannot produce a term - and the counter is
        // bumped where the term is added, never where the weight is declared
        // (the `probe::JEPA` defect: a counter bumped nowhere made the arm
        // look alive while it contributed nothing).
        let moe = self.moe_lb_coef > 0.0;
        if !any && !mor && !fb && !moe {
            return None;
        }
        let ids = ids?;
        let targets = targets?;
        let dev = h.device();
        let mut total: Option<Tensor<1>> = None;
        // MoR first: the router's own BCE against its top-k recomputed on this
        // batch, the one auxiliary the MoR arm has under a pure-CE recipe.
        if let Some(m) = route.mor.clone() {
            crate::probe::note(crate::probe::MOR_BCE);
            total = Some(
                m.mul_scalar(self.mor_bce_weight)
                    + total.unwrap_or_else(|| Tensor::zeros([1], &dev)),
            );
        }
        // The sparse-routing balancer: Switch/GShard's `E * sum f_e P_e`, which
        // is 1 at uniform routing and E at collapse. Weighted HERE rather than
        // in the loop block, with every other auxiliary weight.
        if let Some(lb) = route.moe_lb.clone() {
            crate::probe::note(crate::probe::MOE_LB);
            total = Some(
                lb.mul_scalar(self.moe_lb_coef) + total.unwrap_or_else(|| Tensor::zeros([1], &dev)),
            );
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
                // COUNTED, and it was dead: `probe::JEPA` is asserted by
                // `tests/preset_exec.rs` and was bumped NOWHERE in the source,
                // so the test asserting the arm ran was red, and the counter
                // that would have caught a dropped JEPA term was itself a
                // no-op. Bumped here, where the term is actually added - not
                // where the weight is declared.
                crate::probe::note(crate::probe::JEPA);
                total = Some(
                    j.mul_scalar(self.jepa_weight)
                        + total.unwrap_or_else(|| Tensor::zeros([1], &dev)),
                );
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
            crate::probe::note(crate::probe::DSPARK);
            total = Some(
                d.mul_scalar(self.dspark_weight)
                    + total.unwrap_or_else(|| Tensor::zeros([1], &dev)),
            );
        }
        if fb {
            // COUNTED in two halves (ADR-0019): `asked` is bumped here, at the
            // branch the config opened, and `ran` inside the loss, only when
            // there was at least one valid position. `fb=0/<n>` on the eval
            // line therefore means "a horizon at or past the sequence length",
            // which is a config that trains for thousands of steps and
            // produces no objective - the shape of defect nobody can see in a
            // healthy-looking loss curve.
            crate::probe::note(crate::probe::FUTURE_BYTE_ASKED);
            let head = self.aux.fb.as_ref().expect(
                "aux_fb_weight > 0 but AuxHeads.fb is None: the head is built iff the weight was \
                 non-zero at construction (DormouseModel::new), so this model was built with \
                 aux_fb_weight = 0. Rebuild the model from the config, or set aux_fb_weight = 0.",
            );
            let f = crate::future_byte::future_byte_loss::<B>(
                head,
                h.clone(),
                targets,
                self.aux_fb_horizon,
            );
            total = Some(
                f.mul_scalar(self.aux_fb_weight)
                    + total.unwrap_or_else(|| Tensor::zeros([1], &dev)),
            );
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
            v.expect(
                "finite_scan: reading a parameter back from the device failed - \
                      the checkpoint cannot be judged, refusing to say it is clean",
            )
        };
        let vecs = [
            read(
                self.embedding
                    .weight
                    .val()
                    .clone()
                    .into_data()
                    .try_to_vec::<f32>(),
            ),
            read(
                self.norm
                    .weight
                    .val()
                    .clone()
                    .into_data()
                    .try_to_vec::<f32>(),
            ),
            read(
                self.loop_block
                    .residual_scale
                    .val()
                    .clone()
                    .into_data()
                    .try_to_vec::<f32>(),
            ),
        ];
        let checked = vecs.iter().map(|v| v.len()).sum();
        let bad = vecs
            .iter()
            .map(|v| v.iter().filter(|x| !x.is_finite()).count())
            .sum();
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
        crate::probe::note(crate::probe::RETRACT_FACTOR);
        self.loop_block.retract_tsct(iters);
        self.lm_head.retract(iters);
    }

    /// The same retraction as [`Self::retract_tsct`], grouped: factors are
    /// stacked by shape and each group is one sync-free batched Newton-Schulz
    /// call (`burn_spectral::retract_batched`) instead of one host-syncing
    /// `polar_orthogonalize` per factor.
    ///
    /// Two invariants are copied from the scalar path on purpose, because
    /// getting them wrong is invisible rather than wrong:
    /// - every master is handed back through `Param::from_mapped_value` with
    ///   its OWN id and mapper - the optimizer's records are keyed by id, so
    ///   fresh ids would silently reset every factor's momentum;
    /// - each master's `require_grad` state is MIRRORED, not forced. The polar
    ///   output is a non-leaf on an autodiff backend and burn-optim downgrades
    ///   a non-leaf to a frozen master; forcing the flag would equally
    ///   un-freeze a master the caller had frozen.
    ///
    /// Counted (`probe::RETRACT_BATCHED`) because this arm and the default one
    /// produce the same numbers - a silent fallback here is a run that is
    /// correct and 22% slower.
    pub fn retract_tsct_batched(&mut self, iters: usize) {
        crate::probe::note(crate::probe::RETRACT_BATCHED);
        let mut slots = self.tsct_master_slots();
        let mut ids: Vec<ParamId> = Vec::with_capacity(slots.len());
        let mut maps: Vec<ParamMapper<Tensor<2>>> = Vec::with_capacity(slots.len());
        let mut tracked: Vec<bool> = Vec::with_capacity(slots.len());
        let mut vals: Vec<Tensor<2>> = Vec::with_capacity(slots.len());
        for p in slots.iter() {
            let was_tracked = p.val().is_require_grad();
            let (id, val, map) = Param::clone(p).consume();
            ids.push(id);
            maps.push(map);
            tracked.push(was_tracked);
            vals.push(val);
        }
        {
            let mut refs: Vec<&mut Tensor<2>> = vals.iter_mut().collect();
            burn_spectral::retract_batched(&mut refs, iters);
        }
        for (slot, (((id, map), was_tracked), val)) in slots
            .iter_mut()
            .zip(ids.into_iter().zip(maps).zip(tracked).zip(vals))
        {
            let val = val.detach();
            let val = if was_tracked {
                val.set_require_grad(true)
            } else {
                val
            };
            **slot = Param::from_mapped_value(id, val, map);
        }
    }

    /// Every TSCT master slot in the model, `u` then `v` per linear
    /// (loop_block experts, readout, lm_head) - what
    /// [`Self::retract_tsct_batched`] retracts.
    pub fn tsct_master_slots(&mut self) -> Vec<&mut Param<Tensor<2>>> {
        let mut out: Vec<&mut Param<Tensor<2>>> = Vec::new();
        self.loop_block.push_tsct_masters(&mut out);
        self.lm_head.push_tsct_masters(&mut out);
        out
    }

    /// Worst orthonormality error across all TSCT factors (syncs the device;
    /// monitor at cadence, not per step).
    pub fn max_ortho(&self) -> f32 {
        self.loop_block.max_ortho().max(self.lm_head.max_ortho())
    }

    /// The TSCT forward-path diagnostics: the Gram error of the factor the
    /// forward actually multiplies by at the current annealing `alpha`, and
    /// the `|s|` spectrum's min/max/near-off count. Worst-case over every
    /// factor in the model.
    ///
    /// Read ONLY at the trainer's eval boundary: every factor costs a device
    /// sync, and a per-step read of ~30 factors drains the pipeline (§1.3).
    pub fn tsct_diag(&self) -> crate::param::TsctDiag {
        let mut agg = crate::param::TsctDiag::default();
        self.fold_tsct_diag(&mut agg);
        agg
    }

    /// The fold half of [`Self::tsct_diag`], so the model and the block share
    /// one traversal (and the lm_head is not forgotten by one of them).
    pub fn fold_tsct_diag(&self, agg: &mut crate::param::TsctDiag) {
        self.loop_block.fold_tsct_diag(agg);
        self.lm_head.fold_tsct_diag(agg);
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

    /// The longest sequence this model will accept, in byte positions.
    ///
    /// A getter for the `#[module(skip)]` field, because `generate`/`serve`
    /// need it and reading a field of the same name off a borrowed model is
    /// the sort of thing a refactor silently stops doing.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}
