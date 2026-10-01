//! param - TSCT linear via burn-sct SpectralLinear, pad to multiple of 4,
//! NM knob, BF16 env (mirrors aria semantics; fresh mini composition)
//!
//! # What [`LinearLike`] is for
//!
//! Every projection in dormouse is one of these, so that one place owns the
//! three properties they share: the spectral (low-rank TSCT) factorization
//! with its optional factor quantization, the `out_features` padding cubek's
//! matmul vectorization needs (`N % 4 == 0`), and the precision switches. A
//! linear that is not a `LinearLike` is a deliberate exception and should say
//! so.
//!
//! # Invariants
//!
//! * **Pads on construction, slices on every forward.** `out_features` is
//!   padded UP to a multiple of 4 (except `out_features == 1`, a scalar
//!   projection, where padding would be 4x the parameters for nothing), and the
//!   result is sliced back on the way out. The extra columns are never read
//!   downstream — they exist so the matmul's N dimension vectorizes. Keep the
//!   slice: without it the padded columns leak into the next layer and the
//!   model is quietly wider than its parameter count says.
//! * **Masters are always fp32.** `--quant` picks the FORWARD path only, so a
//!   checkpoint is format-agnostic and re-running with `--quant fp32`
//!   reproduces an earlier run exactly.
//! * **The retraction is a real polar retraction** through
//!   [`LinearLike::retract`], and it keeps autodiff tracking, which is why it
//!   can run inside the step loop. Its cost is per-PARAMETER and does not
//!   amortize with batch: measured 52.8 / 53.3 / 64.6 ms at batch 8 / 16 / 32
//!   against step times of 244 / 440 / 826 ms — 22% of a step at batch 8,
//!   7.8% at batch 32.
//! * **Mixed dtypes NaN on this stack.** Every caller casts to fp32 before a
//!   `LinearLike` and back after; the bf16 matmul path keeps an fp32 graph.

use burn::module::{Module, Param, ParamId};
use burn::tensor::{Device, DispatchTensor, Tensor};
use burn::backend::DispatchKindConversion;
use burn_spectral::SpectralLinear;

/// What one of a [`LinearLike`]'s parameters IS, structurally - no optimizer
/// opinion here. The policy is a function of (this, the linear's `Role`) in
/// `crate::routing`; adding a variant makes that match fail to compile, which
/// is the point: a new leaf cannot appear without a decision about where it
/// trains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearParam {
    /// TSCT master factor, `[in, k]` (u) or `[out, k]` (v).
    Factor,
    /// TSCT singular scales, `[k]` - 1D.
    Scale,
    /// Dense (`use_tsct = false`) weight, `[out, in]`.
    DenseWeight,
    /// Dense bias - 1D, absent when the linear has none.
    DenseBias,
}

/// DM_QUANT_DEBUG, read once per process (queried by every LinearLike
/// forward otherwise).
fn quant_debug_on() -> bool {
    static DBG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DBG.get_or_init(|| std::env::var("DM_QUANT_DEBUG").is_ok())
}

/// COUNTED (ADR-0019 row 29; docs/reviews/doc-coverage-2026-10-01.md §3.3):
/// the bf16-compute request is dropped on a non-CUDA build — the bf16 matmul
/// op does not exist there, the fp32 path returns, the answer is correct — and
/// this line is the only evidence that it happened. Once per process: the
/// call site runs on every forward.
#[cfg(not(feature = "cuda"))]
fn bf16_compute_dropped_once() {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!("[param] bf16_compute requested on non-cuda: fp32 path (counted, once)");
    }
}

/// One projection: spectral or dense, both padded, both sliced back.
#[derive(Module, Debug)]
pub struct LinearLike {
    /// Which of the two parameterizations this linear actually holds. The enum
    /// is part of the checkpoint: a `Dense` variant cannot load a `Tsct`
    /// record, and burnpack refuses the shape mismatch naming the path.
    pub inner: LinearLikeInner,
    /// The REAL (unpadded) output width. The inner module may be up to 3
    /// columns wider; this is what the forward slices back to and what every
    /// downstream shape is computed from.
    #[module(skip)]
    pub out_features: usize,
}

/// The two parameterizations a [`LinearLike`] can hold.
///
/// An enum rather than a config flag so the shape of a model is decided at
/// construction and cannot drift afterwards: `use_tsct` is read once, by
/// [`LinearLike::with_tsct`].
#[derive(Module, Debug)]
pub enum LinearLikeInner {
    /// Spectral low-rank (TSCT) via `burn-spectral`: a `[in, k]` and a
    /// `[k, out]` factor with a `[k]` singular-scale vector between them, so
    /// the parameter count is `k*(in+out+1)` rather than `in*out`. This is what
    /// makes a 2048-wide FFN affordable on a 9.2M-parameter model. Retracted
    /// polar every step (default cadence 1) to hold the factors near
    /// orthogonality, and the factors can be forward-quantized independently
    /// of the fp32 masters.
    Tsct(SpectralLinear),
    /// A plain `[out, in]` weight plus bias. The A/B counterpart of the
    /// spectral path, and the control it has to beat to earn its ~1000 lines.
    /// Reachable since 2026-09-28; before that the variant existed and nothing
    /// constructed it, which made the A/B unanswerable.
    Dense(burn::nn::Linear),
}

impl LinearLike {
    /// A spectral (`Tsct`) linear, `in_features -> out_features` at rank `rank`.
    ///
    /// `out_features` is padded up to a multiple of 4 for the matmul and
    /// sliced back in [`Self::forward`]; `out_features == 1` is left alone.
    /// `rank` is NOT clamped here — the caller owns that decision, and
    /// [`crate::model::DormouseModel::new`] clamps the head's rank to
    /// `min(rank, d_model, vocab)` for the specific reason that a rank above
    /// either dimension is not a factorization.
    pub fn new(in_features: usize, out_features: usize, rank: usize, device: &Device) -> Self {
        // Pad to multiple of 4: cubek matmul vectorization needs N%4==0
        // (aria probe: N=2,3,5,6,7 dirty under initcheck).
        let padded = if !out_features.is_multiple_of(4) && out_features != 1 {
            out_features.next_multiple_of(4)
        } else {
            out_features
        };
        Self {
            inner: LinearLikeInner::Tsct(SpectralLinear::new(in_features, padded, rank, device)),
            out_features,
        }
    }

    /// Plain dense linear - the A/B counterpart of the spectral path. It was
    /// unreachable before (the enum variant existed, nothing constructed it),
    /// which made "is TSCT earning its ~1000 lines?" unanswerable. The honest
    /// question is wide-and-low-rank (TSCT, 2048-wide FFN on small's
    /// 9 197 390 measured params; the older 7.5M figure is retracted)
    /// versus narrow-and-dense (same param budget), not TSCT against nothing.
    pub fn dense(in_features: usize, out_features: usize, device: &Device) -> Self {
        let padded = if !out_features.is_multiple_of(4) && out_features != 1 {
            out_features.next_multiple_of(4)
        } else {
            out_features
        };
        Self {
            inner: LinearLikeInner::Dense(burn::nn::LinearConfig::new(in_features, padded).init(device)),
            out_features,
        }
    }

    /// Spectral or dense, chosen once by the caller from `use_tsct`.
    pub fn with_tsct(
        in_features: usize,
        out_features: usize,
        rank: usize,
        use_tsct: bool,
        device: &Device,
    ) -> Self {
        if use_tsct {
            Self::new(in_features, out_features, rank, device)
        } else {
            Self::dense(in_features, out_features, device)
        }
    }

    /// `x [n, in] -> y [n, out]`, with the padding sliced back off.
    ///
    /// The dispatch has four arms, and they are a config decision each, not a
    /// fallback: bf16 compute on/off, factor quant != Fp32 or not. The one
    /// place a difference IS a degradation: on a non-CUDA build with
    /// `bf16_compute` on, the bf16 matmul op does not exist and the fp32 path
    /// runs instead — the right answer, COUNTED (one stderr line per process,
    /// `bf16_compute_dropped_once`); was SILENT until
    /// docs/reviews/doc-coverage-2026-10-01.md §3.3.
    ///
    /// `DM_QUANT_DEBUG=1` prints the format of every linear on every forward.
    /// Debug-only and not on any hot path that is measured.
    pub fn forward<B: burn::backend::AutodiffBackend>(&self, x: Tensor<2>) -> Tensor<2>
    where
        DispatchTensor: DispatchKindConversion<B>
            + DispatchKindConversion<B::InnerBackend>
            + DispatchKindConversion<burn::backend::Autodiff<B::InnerBackend>>,
    {
        let y = match &self.inner {
            LinearLikeInner::Tsct(l) => {
                if quant_debug_on() {
                    println!("[ll] quant={:?} out={} bf16_compute={}", l.quant, l.out_features, l.bf16_compute);
                }
                if l.bf16_compute {
                    // bf16 matmuls (tensor cores) with the fp32 autodiff
                    // graph; the factor quant follows the configured format.
                    // The custom op is CUDA-only; elsewhere fall back.
                    #[cfg(feature = "cuda")]
                    {
                        l.forward_quant_bf16::<B>(x)
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        bf16_compute_dropped_once();
                        if l.quant != burn_spectral::QuantFormat::Fp32 {
                            l.forward_quant::<B>(x)
                        } else {
                            l.forward(x)
                        }
                    }
                } else if l.quant != burn_spectral::QuantFormat::Fp32 {
                    l.forward_quant::<B>(x)
                } else {
                    l.forward(x)
                }
            }
            LinearLikeInner::Dense(l) => l.forward(x),
        };
        // slice back to the real out_features when padded (extra columns are
        // never read downstream and would be the float4 tail).
        if y.dims()[1] != self.out_features {
            let n = y.dims()[0];
            y.slice([0..n, 0..self.out_features])
        } else {
            y
        }
    }

    /// Every parameter this linear owns, with what it is. Destructured
    /// WITHOUT `..` on purpose: a new leaf here is a compile error, not a
    /// parameter nobody declared a group for.
    pub fn param_kinds(&self) -> Vec<(ParamId, LinearParam)> {
        let Self { inner, out_features: _ } = self;
        match inner {
            LinearLikeInner::Tsct(l) => vec![
                (l.u.id, LinearParam::Factor),
                (l.v.id, LinearParam::Factor),
                (l.s.id, LinearParam::Scale),
            ],
            LinearLikeInner::Dense(l) => l
                .bias
                .as_ref()
                .map(|b| (b.id, LinearParam::DenseBias))
                .into_iter()
                .chain([(l.weight.id, LinearParam::DenseWeight)])
                .collect(),
        }
    }

    /// Set the quantized factor format on the TSCT arm (no-op otherwise).
    pub fn set_quant(&mut self, quant: burn_spectral::QuantFormat) {
        if let LinearLikeInner::Tsct(l) = &mut self.inner {
            l.set_quant(quant);
        }
    }

    /// Enable the bf16 matmul path (fp32 graph, tensor-core forward).
    pub fn set_bf16_compute(&mut self, on: bool) {
        if let LinearLikeInner::Tsct(l) = &mut self.inner {
            l.bf16_compute = on;
        }
    }

    /// Polar-retract the TSCT masters U/V to orthonormal (burn-spectral NS,
    /// on device, keeps autodiff tracking). No-op for the dense variant.
    /// Without periodic retract the factors drift and the quantized forward
    /// degrades (bf16_KERNEL_PLAN: retract every 1 step, monitor max_ortho).
    pub fn retract(&mut self, iters: usize) {
        if let LinearLikeInner::Tsct(l) = &mut self.inner {
            l.retract(iters);
        }
    }

    /// Append this linear's TSCT masters (`u`, then `v`) to `out` as writable
    /// `Param` slots - the factor list a batched retraction needs. A dense
    /// linear contributes nothing, because it has no factors to retract.
    ///
    /// Slots, not tensors: the batched arm must hand each result back through
    /// `Param::from_mapped_value` with its OWN id and mapper, or the
    /// optimizer's per-factor records (keyed by id) silently reset.
    pub fn push_tsct_masters<'a>(&'a mut self, out: &mut Vec<&'a mut Param<Tensor<2>>>) {
        if let LinearLikeInner::Tsct(l) = &mut self.inner {
            out.push(&mut l.u);
            out.push(&mut l.v);
        }
    }

    /// Worst-case orthonormality error of the TSCT masters (0 for dense).
    /// Per-entry metric: the Frobenius `||UᵀU - I||_F` divided by the rank k.
    /// The raw F-norm scales ~k (it sums k² Gram entries), which put the
    /// plan's 1e-3 threshold *below* the NS retract's own convergence floor
    /// (~4e-3 raw at r=64) — the fp32 fallback then fired on every fresh
    /// run and the factor-quant forward never engaged (measured 2026-09-04,
    /// `ortho_probe`). Per-entry the retract floor is ~6e-5 and 1e-3 is a
    /// real drift bound. Syncs the device (reads back); monitor at cadence.
    ///
    /// This is the BEFORE/AFTER arm of the retraction diagnostic and needs no
    /// separate metric: it already IS the per-entry masters reading. What it
    /// cannot see is [`TsctDiag::fwd`] — measured on a freshly built
    /// `[768,64]` factor, this returns 8.1e-8 while the factor the forward
    /// actually multiplies by reads 7.96e-2 at `alpha = 1`, because the
    /// ternary projection is not an orthogonal map. That gap is why both are
    /// printed and neither alone is the forward's health.
    pub fn max_ortho(&self) -> f32 {
        match &self.inner {
            LinearLikeInner::Tsct(l) => {
                let u = l.u.val();
                let v = l.v.val();
                let ku = u.dims()[1].max(1) as f32;
                let kv = v.dims()[1].max(1) as f32;
                (burn_spectral::ortho_error(&u) / ku)
                    .max(burn_spectral::ortho_error(&v) / kv)
            }
            _ => 0.0,
        }
    }

    /// Fold this linear's TSCT diagnostics into `agg`. A dense linear
    /// contributes nothing, and [`TsctDiag::s_min`] stays `INFINITY` if the
    /// whole model is dense — the eval line reads that as "no TSCT here" and
    /// prints no field rather than printing `inf`.
    pub fn fold_tsct_diag(&self, agg: &mut TsctDiag) {
        let LinearLikeInner::Tsct(l) = &self.inner else { return };
        agg.alpha = agg.alpha.max(l.alpha);
        let (u, v) = (l.u.val(), l.v.val());
        agg.fwd = agg
            .fwd
            .max(burn_spectral::ortho_error_forward(&u, l.alpha))
            .max(burn_spectral::ortho_error_forward(&v, l.alpha));
        let (smax, smin, off) = burn_spectral::spectrum_stats(&l.s.val(), TsctDiag::NEAR_OFF);
        agg.s_max = agg.s_max.max(smax);
        agg.s_min = agg.s_min.min(smin);
        agg.off += off as u32;
    }
}

/// The TSCT forward-path diagnostics, folded across every factor in the
/// model. Worst-case over factors, per the `max_ortho` convention.
///
/// The masters' own Gram error is deliberately NOT a field here: it is
/// [`LinearLike::max_ortho`], already read on both sides of the retraction,
/// and duplicating it here is how two metrics of the same thing drift apart.
#[derive(Clone, Copy, Debug)]
pub struct TsctDiag {
    /// Worst per-entry Gram error of the factor the FORWARD multiplies by,
    /// at each layer's own ternary annealing `alpha` — `burn_spectral`'s
    /// `ortho_error_forward`, the same function `SpectralLinear::forward`
    /// calls. At `alpha = 0` this equals the masters' metric exactly, which
    /// is the cross-check the eval line's pair of numbers rests on.
    pub fwd: f32,
    /// Smallest `|s_i|` over every rank of every factor: a rank this small
    /// contributes nothing to `W = U·diag(s)·Vᵀ` and its gradient is
    /// correspondingly weak, so this is the "silently killed rank" signal.
    pub s_min: f32,
    /// Largest `|s_i|`. Reported with `s_min` because a spectrum is a range:
    /// `s` is what compensates the ternary projection's gain loss, and only
    /// the ratio of the two ends says whether it still can.
    pub s_max: f32,
    /// How many ranks have `|s_i| < NEAR_OFF` across the whole model — the
    /// reviewer's "почти выключенные ранги". Summed, not averaged: one dead
    /// rank in one factor is the fact being reported.
    pub off: u32,
    /// The largest ternary-annealing `alpha` folded in. Reported because
    /// `fwd` is a function of it and a number without its `alpha` is not
    /// comparable across runs; `MAX`, because the forward degrades
    /// monotonically in `alpha` and the worst layer is the one that decides
    /// whether the model is still healthy.
    pub alpha: f32,
}

impl Default for TsctDiag {
    /// `s_min` starts at `INFINITY`, not 0, because 0 is a *reportable*
    /// value (an all-dead spectrum) and the empty fold must be
    /// distinguishable from it. A derive would hand out the 0 that
    /// [`Self::field`] reads as "a model with no TSCT factors".
    fn default() -> Self {
        Self { fwd: 0.0, s_min: f32::INFINITY, s_max: 0.0, off: 0, alpha: 0.0 }
    }
}

impl TsctDiag {
    /// The near-off threshold for `|s_i|`. Three orders of magnitude below
    /// the per-entry magnitude of a `[768,64]` orthonormal factor (~3.6e-2),
    /// i.e. "this rank is not a real singular direction any more". A knob,
    /// not a policy: nothing branches on it.
    pub const NEAR_OFF: f32 = 1e-3;

    /// The `tsct=` field for the eval line, or `None` when the model has no
    /// TSCT factors at all (a `--set use_tsct=false` run, which then prints
    /// nothing rather than a field full of zeros).
    ///
    /// ```text
    /// tsct=<bef>/<aft>/<fwd>/<smin>/<smax>/<off>@<alpha>
    /// ```
    ///
    /// `bef`/`aft` are the masters' per-entry Gram error on either side of
    /// THIS step's retraction, and are `-` when `retract_every` did not fire
    /// on it. Two identical numbers printed as a pair would read as a
    /// measurement and mean "we did not measure it" — the same reasoning that
    /// put names in the `kda_asked` destructuring.
    pub fn field(&self, bef: Option<f32>, aft: Option<f32>) -> Option<String> {
        self.s_min.is_finite().then(|| {
            format!(
                "tsct={}/{}/{:.2e}/{:.3e}/{:.3e}/{}@a={:.2}",
                opt_f32(bef),
                opt_f32(aft),
                self.fwd,
                self.s_min,
                self.s_max,
                self.off,
                self.alpha,
            )
        })
    }
}

/// `-` for a missing reading, else two significant digits of exponent form.
fn opt_f32(v: Option<f32>) -> String {
    v.map_or_else(|| "-".to_string(), |x| format!("{x:.2e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    /// The non-CUDA bf16-compute arm: the request is dropped (COUNTED, one
    /// stderr line per process — rerun with `--nocapture` to read it) and the
    /// fp32 path returns. The drop must not panic on the CPU test backend nor
    /// move the answer (ADR-0019 row 29; doc-coverage §3.3).
    #[test]
    fn bf16_compute_on_cpu_is_the_fp32_answer_and_says_so() {
        let device = Device::flex().autodiff();
        let mut ll = LinearLike::new(8, 4, 4, &device);
        let x = Tensor::<2>::from_data(
            TensorData::new(vec![1.0f32, 0.5, -1.0, 2.0, 0.0, -0.5, 1.5, -2.0], [1, 8]),
            &device,
        );
        // The crate's canonical CPU test backend (gr.rs:193, aux.rs:429).
        type B = burn::backend::Autodiff<
            burn::backend::Flex,
            burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
        >;
        let fp32 = ll.forward::<B>(x.clone());
        ll.set_bf16_compute(true);
        let bf16 = ll.forward::<B>(x);
        assert_eq!(bf16.dims(), [1, 4]);
        let err: f32 = (bf16 - fp32).abs().max().into_scalar();
        assert!(err < 1e-5, "cpu bf16-compute arm diverged from fp32: {err}");
    }
}
