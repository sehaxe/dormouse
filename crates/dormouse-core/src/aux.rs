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
    pub conf: AcceptRatePredictor,
}

impl AuxHeads {
    pub fn new(d_model: usize, vocab: usize, rank: usize, device: &burn::tensor::Device) -> Self {
        Self {
            jepa_pred: JepaPredictor::new(d_model, device),
            dspark: RNNHead::new(vocab, rank, d_model, device),
            conf: AcceptRatePredictor::new(d_model, device),
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
    // Anchors p_i = i*stride with the whole draft window inside the sequence.
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
        .apply_block_logits(base_win, token_ids, hidden_win.clone())
        .reshape([b * n, k, v]);
    let conf_logits = conf
        .prob(hidden_win.reshape([b * n, k, d]), None)
        .reshape([b * n, k, 1]);
    let mask = Tensor::<2>::ones([b * n, k], &draft.device());
    let (total, _ce, _tv, _conf) = dspark_loss(
        draft,
        target_win.reshape([b * n, k, v]),
        target_ids.reshape([b * n, k]),
        conf_logits,
        mask,
        DSPARK_GAMMA,
    );
    total
}

#[cfg(test)]
mod tests {
    use super::*;

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
