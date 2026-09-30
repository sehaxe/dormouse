//! Auxiliary objectives on top of the autoregressive CE (helpers, not the
//! objective): JEPA latent prediction (data2vec 2.0 via burn-jepa) and the
//! DSpark draft head (DeepSeek-style future correction via burn-dspark,
//! used instead of MTP).
//!
//! - JEPA: an EMA teacher encoder (weights held in the train loop) produces
//!   stop-grad latents; the student predicts them at span-masked positions
//!   through [`JepaPredictor`]; KoLeo keeps the latent space spread.
//!   Causality forbids copying the future, so the current latent must encode
//!   predictive abstractions - the "logic" signal.
//! - DSpark: an [`RNNHead`] corrects the frozen backbone logits into the
//!   next-K tokens, trained with `dspark_loss` (CE + TV + confidence BCE,
//!   position-decayed). Gradients flow into the backbone through the hidden
//!   states, shaping multi-token predictiveness; at inference the same head
//!   gives free speculative decoding.

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::tensor::{Bool, DispatchTensor, Int, Tensor};
use burn::backend::DispatchKindConversion;
use burn_dspark::{AcceptRatePredictor, RNNHead, dspark_loss};
use burn_jepa::{jepa_l1_loss, koleo_loss, JepaPredictor};
use std::cell::Cell;

/// EMA teacher momentum (data2vec 2.0 ballpark).
pub const TEACHER_MOMENTUM: f64 = 0.999;
/// DSpark position-decay gamma (DeepSpec default).
pub const DSPARK_GAMMA: f64 = 4.0;
/// KoLeo weight inside the JEPA aux term (DINOv2 uses ~0.1).
const KOLEO_WEIGHT: f32 = 0.1;

/// The auxiliary heads. Serialized with the model; random-initialized when a
/// checkpoint predates them.
#[derive(Module, Debug)]
pub struct AuxHeads {
    pub jepa_pred: JepaPredictor,
    pub dspark: RNNHead,
    /// The acceptance head, paper Eq. 7: `sigmoid(w^T[h_k ; W1[x_{k-1}]])`.
    /// Markov-CONDITIONED, and that is the whole point: the previous draft
    /// token is the mechanism, so a `w^T[h_k]` stand-in is a different head
    /// (it was, until 2026-09-29, and it had no caller for
    /// `AcceptRatePredictor::with_markov` at all).
    ///
    /// Its projection is `[d_model + rank, 1]`, so a checkpoint written
    /// before that date carries `[d_model, 1]` here and burnpack REFUSES the
    /// load with a shape-mismatch validation error naming the path
    /// (`load_record` validates by default). That is the loud outcome: the
    /// head trained on the one-position-shifted window and is worthless
    /// now, and there is no partial-load flag in this trainer to paper over
    /// it with.
    pub conf: AcceptRatePredictor,
    /// The FUTURE-BYTE head: an independent `d_model -> vocab` linear,
    /// supervised with CE on the byte at `t + k`. Its weight and its loss live
    /// in [`crate::future_byte`]; this field is only where the parameters are
    /// kept so they are serialized with the model.
    ///
    /// `Option`, built iff `cfg.aux_fb_weight > 0` (the same
    /// `cfg.use_gr.then(..)` shape `LoopBlock::gr` uses) - so an off-arm model
    /// is byte-identical to a build from before the arm existed, which is what
    /// keeps queue row 1 (pure CE) a valid control and every measured preset
    /// parameter count in `tests/preset_exec.rs` true. `None` with a non-zero
    /// weight is a LOUD error at the call site (`model.rs`), not a skipped term.
    pub fb: Option<crate::param::LinearLike>,
}

impl AuxHeads {
    pub fn new(d_model: usize, vocab: usize, rank: usize, device: &burn::tensor::Device) -> Self {
        Self {
            jepa_pred: JepaPredictor::new(d_model, device),
            dspark: RNNHead::new(vocab, rank, d_model, device),
            // `rank` is the width `W1` emits, i.e. the same table the draft
            // head conditions on - the conditioning input of Eq. 7, not a
            // second embedding of its own.
            conf: AcceptRatePredictor::with_markov(d_model, rank, device),
            // Left `None` here and attached by the model, which is the only
            // place that knows the weight. `AuxHeads::new` is called by tests
            // and by `dspark_oracle` with no config in hand, and giving every
            // one of them a fourth arm to thread is how a signature stops
            // meaning anything.
            fb: None,
        }
    }
}

/// The `(seed, step)` the JEPA mask stream is currently drawing from. Set once
/// per step by the trainer before the forward (ADR-0021).
///
/// The mask used to come from burn's GLOBAL RNG, which is never seeded: two
/// runs of the same config drew different masks, so no A/B was reproducible
/// and a resume changed the objective. The trainer knows the step index and
/// the config's seed; the model does not, and threading a step parameter
/// through every forward signature (and every caller of it) buys nothing the
/// mask does not already get from the batch it is masking. Hence the seam.
/// It is a pair of cells, not an RNG: no state to carry, nothing to restore.
///
/// PER THREAD, not process-global. A global `AtomicU64` pair made the mask a
/// function of `(seed, step, whatever any other thread last stored)`: the
/// test harness runs tests on separate threads, and with both mask tests
/// in flight `mask_is_a_function_of_seed_and_step` failed 15 runs in 40
/// while passing 30/30 when run alone. The trainer's step loop is one thread,
/// so a thread-local is the same stream there - and here it is a *different*
/// run, which is what the test means.
thread_local! {
    static MASK_STREAM: Cell<(u64, u64)> = const { Cell::new((1, 0)) };
}

/// Point the mask stream at step `step` of the run seeded with `seed`.
pub fn set_mask_stream(seed: u64, step: u64) {
    MASK_STREAM.with(|c| c.set((seed, step)));
}

/// Bernoulli-start span-dilated mask, identical to `burn_jepa::mask_indices`
/// in distribution and semantics but drawn from `(seed, step, index)` instead
/// of global RNG state: same inputs, same mask, on any backend, forever.
pub fn mask_stream(t: usize, mask_frac: f32, mask_span: usize) -> Vec<bool> {
    let (seed, step) = MASK_STREAM.with(|c| c.get());
    mask_from(seed, step, t, mask_frac, mask_span)
}

/// The mask itself, with no ambient state at all: a pure function of
/// `(seed, step)`, so two processes at the same commit produce the same bits
/// and a third party can pin a golden. [`mask_stream`] is this plus the
/// per-thread seam the trainer points at each step.
///
/// The start rate is inverted so the expected masked fraction is exactly
/// `mask_frac`: `p = 1 - (1 - mask_frac)^(1/span)`. A start masks itself and
/// the `span - 1` positions after it, which is what makes a masked position
/// never predictable from a masked neighbour - the point of masking at all.
///
/// The only non-integer step is `powf`; a libm that differs by 1 ulp can
/// move a threshold and flip a position whose draw lands inside that ulp
/// (~1e-7 per position). The pinned golden in the tests is a same-machine
/// cross-process check, not a cross-libm one.
pub fn mask_from(seed: u64, step: u64, t: usize, mask_frac: f32, mask_span: usize) -> Vec<bool> {
    let span = mask_span.max(1);
    let rate = 1.0 - (1.0 - mask_frac.clamp(0.0, 1.0)).powf(1.0 / span as f32);
    let key = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(step);
    let mut out = vec![false; t];
    let mut starts = 0usize;
    for i in 0..t {
        // splitmix64 over (key, i): a stateless draw, so position i does not
        // depend on how many draws came before it.
        let mut z = key
            .wrapping_add((i as u64).wrapping_mul(0xD134_2543_DE82_EF95))
            .wrapping_add(0x2545_F491_4F6C_DD1D);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        if ((z >> 40) as f32) / ((1u32 << 24) as f32) < rate {
            // A start covers `span` positions ENDING AT ITSELF: exactly the
            // cumsum difference `burn_jepa::mask_indices` computes, and what
            // makes the expected masked fraction exactly mask_frac
            // (1 - (1 - rate)^span == mask_frac). `max`, not `+=`: a start
            // landing inside an open run MERGES with it rather than pushing
            // the end out (adding gives 2*span for two adjacent starts where
            // the union is span+1, and measures 0.215 for a 0.2 mask).
            starts = span;
        }
        out[i] = starts > 0;
        if starts > 0 {
            starts -= 1;
        }
    }
    out
}

/// Masked JEPA loss: the student predicts the (detached) teacher latent at
/// span-masked positions; KoLeo uniformity on the time-pooled latent keeps
/// the representation from collapsing.
pub fn jepa_aux_loss(
    pred: &JepaPredictor,
    student_latent: Tensor<3>,
    teacher_latent: Tensor<3>,
    mask_frac: f32,
    mask_span: usize,
) -> Tensor<1>
where
    DispatchTensor: DispatchKindConversion<burn::backend::Autodiff<burn::backend::Flex>>,
{
    let t = student_latent.dims()[1];
    let dev = student_latent.device();
    // One byte per position, host-drawn: t is ~512, so this upload is noise
    // next to the [b, t, d] forward it feeds.
    let flags: Vec<i64> = mask_stream(t, mask_frac, mask_span)
        .into_iter()
        .map(|m| m as i64)
        .collect();
    let mask: Tensor<1, Bool> =
        Tensor::<1, Int>::from_data(burn::tensor::TensorData::new(flags, [t]), &dev)
            .greater_elem(0i64);
    jepa_aux_loss_masked(pred, student_latent, teacher_latent, mask)
}

/// [`jepa_aux_loss`] with the mask supplied by the caller. The fused
/// gradcheck draws ONE mask and feeds both paths so the comparison does not
/// depend on the global RNG (parallel tests interleave draws).
pub fn jepa_aux_loss_masked(
    pred: &JepaPredictor,
    student_latent: Tensor<3>,
    teacher_latent: Tensor<3>,
    mask: Tensor<1, Bool>,
) -> Tensor<1>
where
    DispatchTensor: DispatchKindConversion<burn::backend::Autodiff<burn::backend::Flex>>,
{
    let [b, t, d] = student_latent.dims();
    let mask2: Tensor<2, Bool> = mask.unsqueeze_dim::<2>(0).expand([b, t]);
    let predicted = pred.forward(student_latent.clone());
    let l1 = jepa_l1_loss(predicted, teacher_latent.detach(), mask2);
    // KoLeo over the time-pooled latent: [b, d] rows spread on the sphere.
    let pooled = student_latent.mean_dim(1).reshape([b, d]);
    l1 + koleo_loss(pooled).mul_scalar(KOLEO_WEIGHT)
}

/// Flatten-and-queue every float param of the student, in traversal order.
struct ParamCollector {
    flat: Vec<Tensor<1>>,
}

impl ModuleVisitor for ParamCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let t = param.val().clone().detach();
        let n: usize = t.dims().iter().product();
        self.flat.push(t.reshape([n]));
    }
}

/// Mix the queued student params into the teacher's params, same order.
struct EmaMapper {
    student: std::vec::IntoIter<Tensor<1>>,
    m: f32,
}

impl ModuleMapper for EmaMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let s = self
            .student
            .next()
            .expect("student/teacher param order mismatch");
        let (id, tensor, mapper) = param.consume();
        let dims = tensor.dims();
        let new = tensor.mul_scalar(self.m) + s.reshape(dims).mul_scalar(1.0 - self.m);
        Param::from_mapped_value(id, new, mapper)
    }
}

/// `teacher <- momentum * teacher + (1 - momentum) * student`, per param.
/// Consumes and returns the teacher so params are rebuilt fresh (and stay
/// grad-free). Both models must be the same type (identical traversal).
///
/// The returned teacher is `no_grad()`-frozen: its params are fresh AD leaves
/// with `require_grad = false`, so (a) the teacher forward never enters the
/// autodiff tape (a grad-tracked teacher retained every intermediate of a
/// full second forward - ~2x activation memory, the step-0 OOM axis), and
/// (b) the EMA chain cannot accumulate param nodes across steps (an
/// unbounded, never-dropped param history without the freeze).
pub fn ema_update<M: Module>(teacher: M, student: &M, momentum: f64) -> M {
    let mut col = ParamCollector { flat: Vec::new() };
    student.visit(&mut col);
    let mut mapper = EmaMapper {
        student: col.flat.into_iter(),
        m: momentum as f32,
    };
    teacher.map(&mut mapper).no_grad()
}

/// DSpark auxiliary loss: the draft head corrects frozen backbone logits
/// into the next-K tokens at strided anchor positions. `hidden` carries
/// gradients into the backbone; `logits` enter detached (frozen target).
///
/// # `stride` — the anchor SPACING
///
/// `stride` is a distance in BYTE POSITIONS (the sequence is a byte stream,
/// so a "position" is one byte), not a token count and not a fraction. The
/// window of `k` draft steps starts at every `stride`-th position:
/// `p_i = i * stride` for `i = 0, 1, ..., n-1`, and the anchor count is
/// `n = (t - k - 1) / stride` (floor), which keeps the whole window inside
/// the sequence. The consequence worth knowing: the first anchor needs
/// `t >= k + 1 + stride`, so a sequence shorter than that gets NO window and
/// the term is exactly zero. At the shipped `dspark_stride = 16`, `k = 4`
/// and `seq_len = 512` that is **31 anchors, at p = 0, 16, ..., 480**. So `stride`
/// is a *sampling density*: `stride = 1` is a window at every position,
/// `stride = 16` trains on ~6% of the positions, and the cost of the term
/// (K gathers of `[b, n, v]` plus the RNN's `k` steps) scales with `n`.
/// `stride = 0` is refused loudly by `config::validate` — it would collapse
/// the K-step window into K copies of one CE at position 0.
///
/// **Why striding and not random sampling.** The paper samples the anchors at
/// random every step. That is a second RNG stream, and this trainer's mask
/// stream had to be rebuilt from `(seed, step)` for exactly this reason
/// (ADR-0021): burn's global RNG is never seeded, so two runs of one config
/// drew different anchor sets, no A/B was reproducible, and a resume
/// changed the objective. A fixed stride is the same coverage per step with
/// none of that. It is a determinism fix, not an approximation of the
/// paper's estimator, and it has never been A/B'd against sampling — the
/// honest statement is "deterministic and documented", not "equivalent".
pub fn dspark_aux_loss(
    dspark: &burn_dspark::RNNHead,
    conf: &AcceptRatePredictor,
    hidden: Tensor<3>,
    logits: Tensor<3>,
    ids: Tensor<2, Int>,
    k: usize,
    stride: usize,
) -> Tensor<1>
where
    DispatchTensor: DispatchKindConversion<burn::backend::Autodiff<burn::backend::Flex>>,
{
    let [b, t, d] = hidden.dims();
    let v = logits.dims()[2];
    let k = k.max(1);
    // Anchors p_i = i*stride (see the doc above for what stride is): with the
    // whole draft window inside the sequence, `n = floor((t-k-1)/stride)`.
    let n = if t > k + 1 { (t - k - 1) / stride.max(1) } else { 0 };
    if n == 0 {
        return Tensor::zeros([1], &hidden.device());
    }
    let anchors = Tensor::<1, Int>::arange(0..n as i64, &hidden.device())
        .mul_scalar(stride as i64);
    let frozen = logits.detach();

    // Teacher-forced windows: step `s` sees position p+s (gold prev token +
    // backbone hidden) and predicts position p+s+1.
    let mut token_cols = Vec::with_capacity(k);
    let mut hidden_cols = Vec::with_capacity(k);
    let mut base_cols = Vec::with_capacity(k);
    let mut id_cols = Vec::with_capacity(k);
    let mut target_cols = Vec::with_capacity(k);
    for s in 0..k {
        let pos = anchors.clone().add_scalar(s as i64); // [N]
        let g2: Tensor<2, Int> = pos.clone().unsqueeze_dim::<2>(0).expand([b, n]);
        let g3: Tensor<3, Int> = g2.clone().unsqueeze_dim::<3>(2).expand([b, n, d]);
        let gv: Tensor<3, Int> = g2.clone().unsqueeze_dim::<3>(2).expand([b, n, v]);
        let nxt: Tensor<2, Int> = pos
            .add_scalar(1)
            .unsqueeze_dim::<2>(0)
            .expand([b, n]);
        token_cols.push(ids.clone().gather(1, g2));
        hidden_cols.push(hidden.clone().gather(1, g3));
        base_cols.push(frozen.clone().gather(1, gv));
        id_cols.push(ids.clone().gather(1, nxt.clone()));
        target_cols.push(frozen.clone().gather(1, nxt.unsqueeze_dim::<3>(2).expand([b, n, v])));
    }
    let stack4 = |cols: Vec<Tensor<3>>| -> Tensor<4> {
        let len = cols.len();
        let stacked: Vec<Tensor<4>> = cols.into_iter().map(|c| c.unsqueeze_dim::<4>(2)).collect();
        let [bb, nn, _, w] = stacked[0].dims();
        Tensor::cat(stacked, 2).reshape([bb, nn, len, w])
    };
    let token_ids: Tensor<3, Int> = {
        let stacked: Vec<Tensor<3, Int>> =
            token_cols.into_iter().map(|c| c.unsqueeze_dim::<3>(2)).collect();
        let [bb, nn, _] = stacked[0].dims();
        Tensor::cat(stacked, 2).reshape([bb, nn, k])
    };
    let hidden_win = stack4(hidden_cols);
    let base_win = stack4(base_cols);
    let target_ids: Tensor<3, Int> = {
        let stacked: Vec<Tensor<3, Int>> =
            id_cols.into_iter().map(|c| c.unsqueeze_dim::<3>(2)).collect();
        let [bb, nn, _] = stacked[0].dims();
        Tensor::cat(stacked, 2).reshape([bb, nn, k])
    };
    let target_win = stack4(target_cols);

    // Draft = frozen base logits corrected by the recurrent head; the head
    // and the backbone (through the hidden states) train jointly.
    let draft = dspark
        .apply_block_logits(base_win, token_ids.clone(), hidden_win.clone())
        .reshape([b * n, k, v]);
    // Eq. 7: the acceptance head sees the PREVIOUS DRAFT TOKEN as well as
    // the hidden state, through the draft head's own `W1` - the same
    // mechanism the paper's `W1[x_{k-1}]` names, not a second embedding.
    // `token_ids` is the window's conditioning token per step, which is
    // exactly the token this step's own prediction is verified against.
    let prev_emb = dspark.get_prev_embeddings(token_ids.reshape([b * n, k]));
    let conf_logits = conf
        .prob(hidden_win.reshape([b * n, k, d]), Some(prev_emb))
        .reshape([b * n, k, 1]);
    let mask = Tensor::<2>::ones([b * n, k], &draft.device());
    // `Some` unconditionally: the acceptance head is BUILT (markov-conditioned
    // as of 1836ecb) and always has a value on this path. The `None` arm exists
    // to match DeepSeek's API, where a model may run without a confidence head -
    // charging 0.693 for an absent head was a real defect, and this call site is
    // the one place where the head genuinely exists.
    let (total, _ce, _tv, _conf) = dspark_loss(
        draft,
        target_win.reshape([b * n, k, v]),
        target_ids.reshape([b * n, k]),
        Some(conf_logits),
        mask,
        DSPARK_GAMMA,
    );
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DormouseConfig, DormouseModel};
    use burn::backend::autodiff::Autodiff;
    use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
    use burn::tensor::{Device, Distribution, TensorData};

    /// The backend alias the train crate uses for `--features cpu`;
    /// proven to satisfy the `DispatchKindConversion` bounds the forward and
    /// this module's loss wrapper carry.
    type B = Autodiff<burn::backend::Flex, BalancedCheckpointing>;

    /// A DSpark-carrying model small enough for a `--lib` test: the shipped
    /// aux weights and K = 4, the widths cut to nothing. `dspark_stride = 8`
    /// on t = 64 leaves `(64-4-1)/8 = 7` anchors, so the window is real and
    /// the n == 0 arm is not what is under test.
    fn mini() -> DormouseConfig {
        DormouseConfig {
            d_model: 32,
            n_heads: 2,
            head_dim: 16,
            d_ffn: 64,
            rank: 8,
            max_iter: 1,
            n_experts: 1,
            max_seq_len: 64,
            use_kda: false,
            use_tsct: false,
            engram_rows: 1024,
            dspark_stride: 8,
            // DSpark is OFF in every shipped preset as of 2026-09-29 (DeepSeek's
            // own MTP ablation reports the head bits-per-byte neutral, and our
            // protocol measures BPB), so `default()` gives dspark_weight = 0
            // and `forward_with_hidden` correctly returns `None` aux - `any` at
            // model.rs:254 needs a non-zero weight plus dspark_k > 0. These
            // tests check that the DSpark TERM behaves, not that the default
            // turns it on, so the arm is switched on here explicitly.
            dspark_weight: 0.1,
            ..DormouseConfig::default()
        }
    }

    /// Deterministic bytes (Knuth MMIX LCG, same one the seam tests use) -
    /// no RNG dependency, so a failure is reproducible.
    fn bytes(seed: u64, n: usize) -> Vec<i64> {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) % 256) as i64
            })
            .collect()
    }

    fn ids2(v: &[i64], dev: &Device) -> Tensor<2, Int> {
        Tensor::from_data(TensorData::new(v.to_vec(), [1, v.len()]), dev)
    }

    /// The DSpark window is teacher-forced from ONE id tensor: step `s` of the
    /// window at anchor `p` is conditioned on the byte at `p+s` and supervised
    /// toward the byte at `p+s+1`. The tensor must therefore be the sequence
    /// the model CONSUMED.
    ///
    /// It was `targets` until 2026-09-29, and because `targets` is the
    /// label sequence (`targets[q] == x[q+1]`) the head's step `s` was fed
    /// `x[p+s+1]` - the byte `logits[p+s]` had just predicted - and asked to
    /// emit `x[p+s+2]`, while the hidden state it was handed, `h[p+s]`, had
    /// only ever seen `x[0..=p+s]`. The draft head was given the answer.
    ///
    /// The property that pins the fix is INVARIANCE, not a golden number: the
    /// term is a function of `(input_ids, hidden, logits)` and of nothing
    /// else, so the label sequence cannot move it by a bit. With the old
    /// wiring the two runs below read 0.17736106 and 0.17743118 - a 7e-5
    /// difference on a RANDOMLY INITIALIZED head - and this test is red.
    #[test]
    fn dspark_aux_does_not_read_the_label_sequence() {
        let dev = Device::flex().autodiff();
        let cfg = mini();
        let model = DormouseModel::new(&cfg, &dev);
        let t = cfg.max_seq_len;
        let x = bytes(0xD5, t);
        // The real trainer's labelling: y[q] = x[q+1].
        let y_shift: Vec<i64> = x[1..].iter().chain(std::iter::once(&x[0])).copied().collect();
        // A second, unrelated labelling of the same input.
        let y_other = bytes(0x1F, t);

        let run = |y: &[i64]| -> f32 {
            model
                .forward_with_hidden::<B>(ids2(&x, &dev), None, None, Some(ids2(y, &dev)), None)
                .3
                .expect("dspark_weight > 0 with labels given must return an aux term")
                .into_scalar::<f32>()
        };
        let a = run(&y_shift);
        let b = run(&y_other);
        assert!(
            (a - b).abs() < 1e-6,
            "the DSpark term read the LABELS: {a} vs {b} for the same input"
        );
        assert!(a.is_finite() && a != 0.0, "the term is vacuous at {a}, so invariance is free");
        // Non-vacuity in the other direction: it IS a function of the input
        // the backbone consumed (an always-zero aux would pass the above).
        // The two tolerances sit on purpose and neither sits on a value:
        // the invariance bound is 1e-6, the sensitivity measures 2.8e-5 on a
        // randomly initialized head, and the LABEL leak moves the term by
        // 7.0e-5 - so the gate separates the three cases with margin.
        let x2: Vec<i64> = x.iter().map(|&v| (v + 97) % 256).collect();
        let c = model
            .forward_with_hidden::<B>(
                ids2(&x2, &dev),
                None,
                None,
                Some(ids2(&y_shift, &dev)),
                None,
            )
            .3
            .expect("aux")
            .into_scalar::<f32>();
        assert!(
            (a - c).abs() > 1e-6,
            "the DSpark term ignores the consumed sequence too: {a} vs {c}"
        );
    }

    /// THE MARKOV PATH IS THE ONE THAT RUNS. This is half 2 of the pair with
    /// `the_confidence_head_cannot_run_without_the_previous_token` (half 1):
    /// the loss wrapper feeds the head the previous token's embedding, so a
    /// regression to `prob(h, None)` panics on the markov head instead of
    /// running. Both halves must stay green together, and both are shown red
    /// against the wiring they forbid.
    #[test]
    fn dspark_window_runs_only_on_the_markov_confidence_head() {
        let dev = Device::flex().autodiff();
        let cfg = mini();
        let heads = AuxHeads::new(cfg.d_model, cfg.vocab, cfg.rank, &dev);
        let (b, t) = (1, cfg.max_seq_len);
        let v = cfg.vocab;
        let hidden = Tensor::<3>::random([b, t, cfg.d_model], Distribution::Normal(0.0, 1.0), &dev);
        let logits = Tensor::<3>::random([b, t, v], Distribution::Normal(0.0, 1.0), &dev);
        let ids: Vec<i64> = bytes(0x3C, b * t);
        let ids = Tensor::from_data(TensorData::new(ids, [b, t]), &dev);
        let l = dspark_aux_loss(
            &heads.dspark,
            &heads.conf,
            hidden,
            logits,
            ids,
            cfg.dspark_k,
            cfg.dspark_stride,
        );
        let got = l.into_scalar::<f32>();
        assert!(got.is_finite() && got > 0.0, "the window must produce a real term, got {got}");
    }

    /// Half 1: the head `AuxHeads` builds cannot run without the token at
    /// all. A hidden-only head accepts `None` happily, so this panic IS the
    /// assertion - and it is the half that fails if the constructor goes
    /// back to `AcceptRatePredictor::new`.
    #[test]
    #[should_panic(expected = "predictor expects Markov embeddings")]
    fn the_confidence_head_cannot_run_without_the_previous_token() {
        let dev = Device::flex();
        let cfg = mini();
        let heads = AuxHeads::new(cfg.d_model, cfg.vocab, cfg.rank, &dev);
        let h = Tensor::<3>::zeros([1, 2, cfg.d_model], &dev);
        assert_eq!(
            heads.conf.prob(h.clone(), Some(Tensor::zeros([1, 2, cfg.rank], &dev))).dims(),
            [1, 2, 1]
        );
        let _ = heads.conf.logit(h, None);
    }

    /// Eq. 7's mechanism, asserted from this side of the boundary: the
    /// acceptance logit must MOVE with the previous token, and the
    /// hidden-only head must be a constant. `burn_dspark`'s own
    /// `accept_rate_predictor_reads_the_previous_token` says the same thing
    /// at the type; this copy is the one the repo's gate actually runs
    /// (`tools/wt.sh test` builds the vendor crates as dependencies, never
    /// their test targets).
    ///
    /// The projection is bias-free, so a ZERO hidden state leaves the markov
    /// block as the only live input - the isolation is by construction, not
    /// by surgery on private weights.
    #[test]
    fn the_confidence_head_reads_the_previous_draft_token() {
        const D: usize = 8;
        const R: usize = 4;
        let dev = Device::flex();
        let markov = burn_dspark::VanillaMarkov::new(64, R, &dev);
        let p = AcceptRatePredictor::with_markov(D, R, &dev);
        let h = Tensor::<3>::zeros([1, 3, D], &dev);
        let token = |i: i64| -> Tensor<2, Int> {
            Tensor::from_data(TensorData::new(vec![i, 1 + i, 2 + i], [1, 3]), &dev)
        };
        let logit = |t: i64| -> Vec<f32> {
            p.logit(h.clone(), Some(markov.get_prev_embeddings(token(t))))
                .into_data()
                .try_to_vec()
                .unwrap()
        };
        let (a, b) = (logit(1), logit(9));
        let moved: f32 = (0..a.len()).map(|i| (a[i] - b[i]).abs()).fold(0.0, f32::max);
        assert!(moved > 1e-4, "the markov head ignores W1[x]: {a:?} vs {b:?}");
        let hidden_only: Vec<f32> =
            AcceptRatePredictor::new(D, &dev).prob(h, None).into_data().try_to_vec().unwrap();
        assert!(
            hidden_only.iter().all(|x| (x - 0.5).abs() < 1e-6),
            "a bias-free hidden-only head on a ZERO hidden state is sigmoid(0): {hidden_only:?}"
        );
    }

    /// A hidden-only head CANNOT serve this window: the wrapper hands it the
    /// previous token's embedding, which it was not sized for. Loud, by
    /// construction, on both sides (ADR-0019) - there is no arm here that
    /// quietly degrades to `w^T[h_k]`.
    #[test]
    #[should_panic(expected = "predictor was built hidden-only")]
    fn dspark_window_refuses_a_hidden_only_confidence_head() {
        let dev = Device::flex().autodiff();
        let cfg = mini();
        let heads = AuxHeads::new(cfg.d_model, cfg.vocab, cfg.rank, &dev);
        let (b, t) = (1, cfg.max_seq_len);
        let hidden = Tensor::<3>::random([b, t, cfg.d_model], Distribution::Normal(0.0, 1.0), &dev);
        let logits = Tensor::<3>::random([b, t, cfg.vocab], Distribution::Normal(0.0, 1.0), &dev);
        let ids = Tensor::from_data(TensorData::new(bytes(0x3C, b * t), [b, t]), &dev);
        let _ = dspark_aux_loss(
            &heads.dspark,
            &AcceptRatePredictor::new(cfg.d_model, &dev),
            hidden,
            logits,
            ids,
            cfg.dspark_k,
            cfg.dspark_stride,
        );
    }

    /// The documented meaning of `stride`, made checkable: it is the anchor
    /// SPACING in byte positions, and the count of windows it leaves is
    /// `floor((t - k - 1) / stride)` - so there is a length below which the
    /// field buys nothing at all, and that boundary is where the doc's
    /// "31 anchors at t=512, k=4, stride=16" comes from. (These off-by-ones
    /// are not decorative: the first draft of this test claimed the first
    /// window fits at `t = k + 2` and the run below refused it. It needs
    /// `t >= k + 1 + stride`.)
    ///
    /// The `n == 0` arm returning a real zero (rather than a tiny term) is
    /// also the one ADR-0019 lists as site 32; this pins the value, not the
    /// count, because a count that nobody can read off the loss is exactly
    /// what made that site invisible.
    #[test]
    fn dspark_stride_is_the_anchor_spacing() {
        let dev = Device::flex().autodiff();
        let cfg = mini();
        let heads = AuxHeads::new(cfg.d_model, cfg.vocab, cfg.rank, &dev);
        let (k, v) = (cfg.dspark_k, cfg.vocab);
        let run = |t: usize, stride: usize| -> f32 {
            let hidden = Tensor::<3>::random([1, t, cfg.d_model], Distribution::Normal(0.0, 1.0), &dev);
            let logits = Tensor::<3>::random([1, t, v], Distribution::Normal(0.0, 1.0), &dev);
            let ids = Tensor::from_data(TensorData::new(bytes(0x77, t), [1, t]), &dev);
            dspark_aux_loss(&heads.dspark, &heads.conf, hidden, logits, ids, k, stride)
                .into_scalar::<f32>()
        };
        let stride = 16usize;
        // n = 0 below t = k + 1 + stride, and the term is then exactly zero.
        assert_eq!(run(k + stride, stride), 0.0, "t = k + stride leaves (stride-1)/stride = 0");
        assert!(run(k + stride + 1, stride) > 0.0, "t = k + stride + 1 fits the first window");
        // The doc's arithmetic, t = 512 / k = 4 / stride = 16 -> 31 anchors.
        assert_eq!((512 - 4 - 1) / 16, 31, "the anchor count the doc quotes");
        // One position short of spanning the gap is no window; exactly
        // spanning it is the single window at p = 0.
        assert_eq!(run(64, 64), 0.0, "stride wider than t - k - 1 leaves nothing");
        assert!(run(64, 59) > 0.0, "stride == t - k - 1 must leave the single window at p = 0");
    }

    /// THE PRICE OF EQ. 7, measured rather than asserted. `aux.conf` grew
    /// from `[d_model, 1]` to `[d_model + rank, 1]`, and a record written
    /// before that is REFUSED by the loader - a validation error naming the
    /// path, not a silent reshape and not a silently re-initialized head.
    /// `load_record` validates by default and this trainer passes no
    /// `allow_partial`.
    ///
    /// What that costs, concretely: every checkpoint on this box that was
    /// written with a `conf` head in it, which is all of them (the heads are
    /// serialized whether or not the terms are on) - 2.1 GB across
    /// `checkpoints/*.bin`, `mor_ab_ckpt/*.bin`, `official_v{3,4,5}.bin`,
    /// `pretrain_v21.bin`, `core_probe.bin`. The escape is a loader change
    /// (`allow_partial` + `allow_unused`, so the stale tensor is dropped and
    /// the head re-initializes) at `dormouse-train/src/lib.rs:607`, which is
    /// not this crate's file and is therefore NOT done here: whether to keep
    /// a backbone whose aux head was trained on the shifted window is the
    /// owner's call, and it is worth making deliberately.
    #[test]
    fn an_older_conf_shape_is_refused_not_reshaped() {
        let dev = Device::flex();
        let cfg = mini();
        let record = AuxHeads::new(cfg.d_model, cfg.vocab, cfg.rank, &dev).into_record();
        // Rebuild the head the way it was built on 2026-09-28.
        let mut heads = AuxHeads::new(cfg.d_model, cfg.vocab, cfg.rank, &dev);
        heads.conf = AcceptRatePredictor::new(cfg.d_model, &dev);
        let err = heads
            .try_load_record(record)
            .expect_err("a [d_model, 1] conf must be refused, not loaded into [d_model + rank, 1]")
            .to_string();
        assert!(err.contains("conf"), "the error must name the field it refused: {err}");
    }

    /// ADR-0021: the mask is a pure function of `(seed, step)`. The whole
    /// point of the change is that property, so it is the thing asserted -
    /// not "a mask came out" (which the RNG version also satisfied).
    ///
    /// This test was RED at HEAD (15 runs in 40 with the sibling test in
    /// flight, 30/30 green alone): the seam was a process-global `AtomicU64`
    /// pair, so another thread's `set_mask_stream` moved it mid-assertion.
    #[test]
    fn mask_is_a_function_of_seed_and_step() {
        let (t, frac, span) = (512, 0.15, 4);
        set_mask_stream(7, 100);
        let a = mask_stream(t, frac, span);
        // Same (seed, step) twice: identical, on any backend, with no RNG
        // state involved. This is what makes a resume and an A/B replay.
        set_mask_stream(7, 100);
        assert_eq!(a, mask_stream(t, frac, span), "same (seed, step) must redraw the same mask");
        // The next step must NOT be the same mask, or every step trains on
        // the same masked positions.
        set_mask_stream(7, 101);
        assert_ne!(a, mask_stream(t, frac, span), "consecutive steps must differ");
        // A different seed is a different run.
        set_mask_stream(8, 100);
        assert_ne!(a, mask_stream(t, frac, span), "a different seed must differ");
        // ...and the seam is wired to the no-ambient-state function, which is
        // what makes the property survive into a second process.
        assert_eq!(a, mask_from(7, 100, t, frac, span), "the seam must feed mask_from");
    }

    /// Two threads, two runs, one process: each must get its own
    /// `(seed, step)` mask. This is the half that was broken, and it is the
    /// production shape too - a step must not be able to move another run's
    /// mask. Cheap: the inner thread redraws while the outer one is mid-step.
    #[test]
    fn mask_stream_is_per_thread() {
        let t = 4096;
        set_mask_stream(1, 1);
        let mine = mask_from(1, 1, t, 0.2, 4);
        let other = std::thread::spawn(move || {
            set_mask_stream(9, 9);
            let before = mask_stream(t, 0.2, 4);
            for k in 1..=64u64 {
                set_mask_stream(9, 9 + k);
            }
            (before, mask_from(1, 1, t, 0.2, 4))
        })
        .join()
        .unwrap();
        assert_eq!(other.0, mask_from(9, 9, t, 0.2, 4), "other thread's own step");
        assert_eq!(other.1, mine, "another thread's step must not move this run's mask");
        assert_eq!(mask_stream(t, 0.2, 4), mine, "this thread's step must survive the spawn");
    }

    /// The cross-process half. `mask_from` reads nothing but its arguments and
    /// integer mixing, so two processes at the same commit MUST agree - the
    /// only way that claim can rot is a silent change to the derivation, and
    /// a pinned FNV of the mask bits is what catches it. (ADR-0021's
    /// "two runs of the same seed produce the same mask", in the only form a
    /// unit test can honestly assert without launching a second process.)
    #[test]
    fn mask_from_is_pinned() {
        let m = mask_from(1, 0, 64, 0.5, 1);
        assert_eq!(m.len(), 64);
        // A constant mask would satisfy any golden; require the actual mix.
        let ones = m.iter().filter(|b| **b).count();
        assert!((8..=56).contains(&ones), "pinned mask is not a mix: {ones}/64 true");
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in &m {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        // Regenerate after ANY deliberate change to the derivation, and say so
        // in the commit: a changed golden here means the mask of every past
        // run is no longer the mask of the next one.
        assert_eq!(h, GOLDEN_MASK_FNV, "mask derivation changed; re-pin deliberately");
    }
    const GOLDEN_MASK_FNV: u64 = 8732553446442614006;

    /// The mask must still BE the thing it replaced: the masked fraction is
    /// `mask_frac` (that identity is what the inverted start rate exists for)
    /// and the runs are `span` long. Both halves of this test earned their
    /// place - the first draft dilated to nothing (0.049 for a 0.2 mask) and
    /// the second pushed overlapping runs apart (0.215), both of which the
    /// determinism test above was happy with.
    #[test]
    fn mask_keeps_the_span_dilation() {
        let t = 4096;
        let (mut frac, mut runs, mut len) = (0f32, 0usize, 0usize);
        for step in 0..200u64 {
            set_mask_stream(3, step);
            let m = mask_stream(t, 0.2, 4);
            frac += m.iter().filter(|x| **x).count() as f32 / t as f32;
            for c in m.chunk_by(|a, b| a == b).filter(|c| c[0]) {
                runs += 1;
                len += c.len();
            }
        }
        let frac = frac / 200.0;
        assert!((frac - 0.2).abs() < 0.01, "masked fraction {frac} vs the requested 0.2");
        assert!(runs > 100, "suspiciously few masked runs: {runs}");
        let mean = len as f32 / runs as f32;
        // Runs merge when two starts land within `span`, so the mean is above
        // `span` (4.6 measured); the identity that matters is the FRACTION.
        assert!(mean > 4.0 && mean < 5.5, "mean run length {mean} vs the requested span 4");
        // Degenerate configs must not panic or invert the sense.
        set_mask_stream(3, 1);
        assert_eq!(mask_stream(8, 0.0, 4), vec![false; 8]);
        assert_eq!(mask_stream(8, 1.0, 1), vec![true; 8]);
        assert_eq!(mask_stream(0, 0.5, 4), Vec::<bool>::new());
    }
}
