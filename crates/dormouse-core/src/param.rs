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
}
