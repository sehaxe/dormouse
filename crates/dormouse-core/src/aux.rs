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
use burn_jepa::{jepa_l1_loss, koleo_loss, mask_indices, JepaPredictor};

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
    DispatchTensor: DispatchKindConversion<burn::backend::Autodiff<burn::backend::NdArray>>,
{
    let [b, t, d] = student_latent.dims();
    let mask: Tensor<1, Bool> = mask_indices(t, mask_frac, mask_span, &student_latent.device());
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
pub fn ema_update<M: Module>(teacher: M, student: &M, momentum: f64) -> M {
    let mut col = ParamCollector { flat: Vec::new() };
    student.visit(&mut col);
    let mut mapper = EmaMapper {
        student: col.flat.into_iter(),
        m: momentum as f32,
    };
    teacher.map(&mut mapper)
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
    DispatchTensor: DispatchKindConversion<burn::backend::Autodiff<burn::backend::NdArray>>,
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
