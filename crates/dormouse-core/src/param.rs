//! param - TSCT linear via burn-sct SpectralLinear, pad to multiple of 4,
//! NM knob, BF16 env (mirrors aria semantics; fresh mini composition)
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

#[derive(Module, Debug)]
pub struct LinearLike {
    pub inner: LinearLikeInner,
    #[module(skip)]
    pub out_features: usize,
}

#[derive(Module, Debug)]
pub enum LinearLikeInner {
    Tsct(SpectralLinear),
    Dense(burn::nn::Linear),
}

impl LinearLike {
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
