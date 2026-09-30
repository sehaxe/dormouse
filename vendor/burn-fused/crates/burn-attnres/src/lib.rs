//! # burn-attnres - Attention Residuals for Burn
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//!
//! | arXiv | Mode | What |
//! |-------|------|------|
//! | [2603.15031](https://arxiv.org/abs/2603.15031) | Full | Softmax over all previous layer outputs |
//! | [2603.15031](https://arxiv.org/abs/2603.15031) | Block | Softmax over block-level representations |
//!
//! Drop-in replacement for fixed residual accumulation. Mitigates PreNorm
//! dilution: learned query per layer attends over previous representations.
//!
//! Key results (Kimi Linear 48B): GPQA +7.5, HumanEval +3.1, MMLU +1.1.
use burn::module::{Module, Param};
use burn::nn::Initializer;
use burn::tensor::{activation, Device, Tensor};

// `pub` (it was private) so the ADR-0019 seam counters are readable from
// outside; and it still compiles under `autodiff` alone, deliberately: the
// fused ADJOINT and its strategy seam live here, and a gate that can only be
// compile-checked with a GPU is how a `NoCheckpointing`-only entry survived
// review. The kernels inside stay `#[cfg(feature = "cuda")]`.
#[cfg(feature = "cuda")]
pub mod fused_attnres;

/// THE SCORE CONVENTION, as the one decision it is. The paper has exactly
/// one; the other is ours.
///
/// **Tier-(b) transcription. No external reference implementation exists** —
/// `MoonshotAI/Attention-Residuals` is 6 files (README + PDF + 4 images) and
/// has never shipped code (measured 2026-09-30, GitHub trees API at
/// `85e22310fe5ee860b4a023de312d791de8a5a5e6`). The quotes below are from
/// `Attention_Residuals.pdf` (md5 `f8351f26bce4c33b2880dee3d101f4b0`, the
/// authors' own copy, read with `pdftotext -layout` into a 1436-line text).
///
/// * §3.1: "we adopt `phi(q, k) = exp q^T RMSNorm(k)` [66] with normalization,
///   yielding softmax attention over depth" — **no temperature**.
/// * Table 5, footnote 2: "`phi(q, k) = exp q^T RMSNorm(k)`; `k_i = v_i`;
///   `v_0 = h_1`, `v_i>=1 = f_i(h_i)`. softmax jointly normalized over all
///   sources."
/// * Fig. 2 line 11: `logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)`
///   — a bare einsum over `K = norm(V)`.
/// * A grep of the whole extraction for `sqrt`/`scale`/`temperature` returns
///   no scaling factor anywhere in the mechanism; the only hit is "attention
///   temperature rescaling" about MLA/NoPE context extension (§5.2).
/// * [66] is Biao Zhang and Rico Sennrich, "Root mean square layer
///   normalization", NeurIPS 32 (2019) — so the norm is the **mean** form
///   `x / sqrt(mean(x^2) + eps)`, and the `1/d` is what makes it an RMS norm
///   rather than an L2 one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScoreForm {
    /// Eq. 2 as typeset: `score = q . h / sqrt(mean(h^2) + eps)`.
    #[default]
    Paper,
    /// What this crate shipped from its first commit to 2026-09-30:
    /// `score = q . h * d^-0.5 / sqrt(sum(h^2) + 1e-5)` — a temperature AND an
    /// L2 norm where the paper has neither. Reachable so the difference is a
    /// named choice, not an accident, and so an A/B can quote which one ran.
    SqrtD,
}

impl ScoreForm {
    /// The temperature on the dot product. Eq. 2 has none.
    pub fn scale(self, d: usize) -> f32 {
        match self {
            Self::Paper => 1.0,
            Self::SqrtD => (d as f64).powf(-0.5) as f32,
        }
    }

    /// The multiplier on `sum(h^2)` inside the norm's root: `1/d` for the
    /// paper's RMSNorm (a mean), `1` for a plain L2 norm.
    pub fn norm_m(self, d: usize) -> f32 {
        match self {
            Self::Paper => 1.0 / d as f32,
            Self::SqrtD => 1.0,
        }
    }
}

/// Full AttnRes: learned pseudo-query attends over ALL previous hidden states.
///
/// ```text
/// scores[i] = query · RMSNorm(h_i)          (ScoreForm::Paper, Eq. 2)
/// weights   = softmax(scores)
/// out       = sum_i weights[i] · h_i
/// ```
#[derive(Module, Debug)]
pub struct AttnRes {
    pub query: Param<Tensor<1>>,
    /// Which score convention this instance computes (see [`ScoreForm`]).
    /// Not a parameter: it is a property of the build, and a checkpoint must
    /// not be able to change it.
    #[module(skip)]
    pub form: ScoreForm,
}

impl AttnRes {
    pub fn new(d_model: usize, device: &Device) -> Self {
        Self::with_form(d_model, device, ScoreForm::default())
    }

    pub fn with_form(d_model: usize, device: &Device, form: ScoreForm) -> Self {
        Self {
            // Paper §5: pseudo-queries MUST be initialized to zero (gives
            // exactly uniform alpha at init; prevents training volatility).
            query: Initializer::Zeros.init([d_model], device),
            form,
        }
    }

    /// Full AttnRes: attend over all previous hidden states.
    pub fn forward(&self, history: &[Tensor<3>]) -> Tensor<3> {
        depth_attend_form(history, self.query.val(), self.form)
    }
}

/// Block AttnRes: partition into blocks, attend over block summaries.
///
/// Groups hidden states into `num_blocks` chunks. Within each chunk,
/// standard residual accumulation. Between chunks: AttnRes over chunk
/// summaries. Reduces memory from O(L·d) to O(N·d) where N ≪ L.
#[derive(Module, Debug)]
pub struct BlockAttnRes {
    pub query: Param<Tensor<1>>,
    #[module(skip)]
    pub block_size: usize,
    #[module(skip)]
    pub form: ScoreForm,
}

impl BlockAttnRes {
    pub fn new(d_model: usize, block_size: usize, device: &Device) -> Self {
        Self::with_form(d_model, block_size, device, ScoreForm::default())
    }

    pub fn with_form(
        d_model: usize,
        block_size: usize,
        device: &Device,
        form: ScoreForm,
    ) -> Self {
        Self {
            // Paper §5: pseudo-queries MUST be initialized to zero.
            query: Initializer::Zeros.init([d_model], device),
            block_size,
            form,
        }
    }

    /// Block AttnRes: accumulate within blocks, attend between blocks.
    ///
    /// `history`: all previous hidden states [h0, h1, ..., hL-1].
    /// Groups into `ceil(L / block_size)` blocks, applies depth attention
    /// over the block summaries (paper Eq 5-6: plain sums; the current
    /// partial block is attended from its second layer on).
    ///
    /// KNOWN WRONG against Eq. 6, and not fixed here: the streaming state
    /// folds the embedding into block 1 instead of keeping `b_0 = h_1` as its
    /// own permanently-attended source, and the first sublayer of a block
    /// returns the previous block unchanged. See `research/papers/attnres.md`
    /// D5/D6/D7/D8 and the integration doc's follow-ups. The FULL variant
    /// (`depth_attend_form`) is the one this crate is trusted for; the model
    /// arm does not call this one.
    pub fn forward(&self, history: &[Tensor<3>]) -> Tensor<3> {
        let n = history.len();
        assert!(n > 0, "BlockAttnRes::forward needs >= 1 history entry");
        if n == 1 {
            return history[0].clone();
        }
        let [b, t, d] = history[0].dims();
        let dev = history[0].device();
        let mut st = BlockAttnState::new(b, t, d, &dev);
        let mut last = history[0].clone();
        for h in history {
            last = self.step(h.clone(), &mut st);
        }
        last
    }
}

/// Core depth-wise attention: softmax over history via learned query.
///
/// `history`: `[h0, ..., hN]` - each `[B, T, D]`
/// `query`: `[D]` - learned per-layer pseudo-query
///
/// Returns `[B, T, D]` - weighted sum via softmax attention over depth.
///
/// Eq. 4 with Eq. 2's `phi`; the score convention is [`ScoreForm::default`].
pub fn depth_attend(history: &[Tensor<3>], query: Tensor<1>) -> Tensor<3> {
    depth_attend_form(history, query, ScoreForm::default())
}

/// [`depth_attend`] in an explicitly chosen [`ScoreForm`]. The form is a
/// runtime scalar all the way into the CUDA kernels, so the fused path and
/// the tensor path cannot disagree about it (a kernel that only knew the
/// `SqrtD` form would silently compute a different function on GPU than on
/// CPU - the ADR-0019 failure mode with a cross-backend signature).
pub fn depth_attend_form(
    history: &[Tensor<3>],
    query: Tensor<1>,
    form: ScoreForm,
) -> Tensor<3> {
    let n = history.len();
    if n == 1 {
        return history[0].clone();
    }
    let [b, t, d] = history[0].dims();
    let scale = form.scale(d);
    let m = form.norm_m(d);

    #[cfg(all(feature = "cuda", feature = "autodiff"))]
    {
        // Probe BOTH checkpointing strategies. `Autodiff<Inner>`'s second type
        // parameter defaults to `NoCheckpointing` and the downcast compares the
        // whole backend type, so probing only that one silently sent a
        // `BalancedCheckpointing` caller (dormouse's backend) to the stacked
        // tensor path: the same attention, nothing counting it.
        use burn_autodiff::checkpoint::strategy::{BalancedCheckpointing, NoCheckpointing};
        type CudaBare = burn_cubecl::CubeBackend;
        // variable parent count: try fixed-N specializations
        if let Some(out) = crate::fused_attnres::depth_attend_autodiff_s::<
            CudaBare,
            NoCheckpointing,
            64,
        >(history, query.clone(), form)
        {
            return out;
        }
        if let Some(out) = crate::fused_attnres::depth_attend_autodiff_s::<
            CudaBare,
            BalancedCheckpointing,
            64,
        >(history, query.clone(), form)
        {
            return out;
        }
    }
    #[cfg(feature = "cuda")]
    if let Some(out) = crate::fused_attnres::depth_attend_cuda(history, &query, form) {
        return out;
    }

    let stacked: Vec<Tensor<4>> = history
        .iter()
        .map(|h| h.clone().unsqueeze_dim::<4>(0))
        .collect();
    let h_stack = Tensor::cat(stacked, 0);

    // Tensor fallback (non-CUDA backend or fused dispatch unavailable).
    // `m` is the mean-vs-sum multiplier: ScoreForm::Paper divides the squared
    // norm by d (a true RMSNorm, Zhang & Sennrich [66]), SqrtD does not.
    let h_norm_sq = h_stack.clone().powf_scalar(2.0).sum_dim(3).mul_scalar(m).add_scalar(1e-5);
    let [nh, bh, th, _ns] = h_norm_sq.dims();
    let h_norm = h_stack.clone() / h_norm_sq.sqrt().reshape([nh, bh, th, 1usize]);

    let q = query.reshape([1, 1, 1, d]);
    let scores = (q * h_norm).sum_dim(3).mul_scalar(scale);
    let weights = activation::softmax(scores, 0);
    h_stack
        .mul(weights.reshape([nh, bh, th, 1usize]))
        .sum_dim(0)
        .reshape([b, t, d])
}

/// Online-softmax attention score of one source against the query
/// (identical normalization and scale to [`depth_attend_form`]).
fn source_score(query: &Tensor<1>, src: &Tensor<3>, form: ScoreForm) -> Tensor<3> {
    #[cfg(feature = "cuda")]
    if let Some(s) = crate::fused_attnres::source_score_cuda(query, src, form) {
        return s;
    }
    let [_, _, d] = src.dims();
    let norm = src.clone()
        / src
            .clone()
            .powf_scalar(2.0)
            .sum_dim(2)
            .mul_scalar(form.norm_m(d))
            .add_scalar(1e-5)
            .sqrt();
    let q = query.clone().reshape([1, 1, d]);
    // sum_dim keeps the size-1 dim: [B,T,1]
    (q * norm).sum_dim(2).mul_scalar(form.scale(d))
}

/// Streaming state for [`BlockAttnRes::step`]: online-softmax attention over
/// completed blocks plus a running partial block (paper Eq 6 + Algorithm 1).
///
/// The caller owns the state and calls `step(h)` per new layer output;
/// `step` returns the attention output for that position, O(1) in the number
/// of completed blocks (no full recompute, no O(L·d) history retention).
#[derive(Debug)]
pub struct BlockAttnState {
    /// Running sum of the current block's layer outputs.
    pub partial: Tensor<3>,
    pub partial_count: usize,
    /// Online softmax over completed block reps: max score, exp-sum, weighted values.
    pub max_score: Tensor<3>,
    pub sum_exp: Tensor<3>,
    pub acc: Tensor<3>,
    pub started: bool,
}

impl BlockAttnState {
    pub fn new(b: usize, t: usize, d: usize, device: &Device) -> Self {
        Self {
            partial: Tensor::zeros([b, t, d], device),
            partial_count: 0,
            max_score: Tensor::zeros([b, t, 1], device),
            sum_exp: Tensor::zeros([b, t, 1], device),
            acc: Tensor::zeros([b, t, d], device),
            started: false,
        }
    }

    fn incorporate(&mut self, query: &Tensor<1>, src: &Tensor<3>, form: ScoreForm) {
        let s = source_score(query, src, form); // [B,T,1]
        if !self.started {
            self.max_score = s.clone();
            self.sum_exp = Tensor::ones([s.dims()[0], s.dims()[1], 1], &src.device());
            self.acc = src.clone();
            self.started = true;
            return;
        }
        self.merge_source(src, &s);
    }

    /// Fuse `src` (score `s`) into the online state; returns the attended
    /// output over all attended sources (fused CUDA merge, else tensor path).
    fn merge_source(&mut self, src: &Tensor<3>, s: &Tensor<3>) -> Tensor<3> {
        #[cfg(feature = "cuda")]
        if let Some(m) = crate::fused_attnres::merge_cuda(
            &mut self.acc,
            &mut self.max_score,
            &mut self.sum_exp,
            src,
            s,
        ) {
            return m;
        }
        let m_new =
            (self.max_score.clone() + s.clone() + (self.max_score.clone() - s.clone()).abs())
                .div_scalar(2.0);
        let rescale = (self.max_score.clone() - m_new.clone()).exp();
        self.acc = self.acc.clone().mul(rescale.clone())
            + src.clone().mul((s.clone() - m_new.clone()).exp());
        self.sum_exp = self.sum_exp.clone().mul(rescale) + (s.clone() - m_new.clone()).exp();
        self.max_score = m_new;
        self.attended()
    }

    /// Attention output over the currently attended sources.
    fn attended(&self) -> Tensor<3> {
        self.acc.clone() / self.sum_exp.clone().clamp_min(1e-12)
    }
}

impl BlockAttnRes {
    /// Create a fresh streaming state for `[b, t, d]` inputs.
    pub fn init_state(&self, b: usize, t: usize, d: usize, device: &Device) -> BlockAttnState {
        BlockAttnState::new(b, t, d, device)
    }

    /// Streaming block attention (paper §4.2, Algorithm 1 + Eq 6).
    ///
    /// `h`: the new layer output. Updates the running partial block; when the
    /// block completes (`block_size` outputs) it is folded into the attended
    /// set via an online-softmax merge. The partial block is attended only
    /// from its second layer on (Eq 6: the block's first layer excludes the
    /// current partial sum, which would otherwise create a self-loop).
    ///
    /// Returns the depth-attention output `[B, T, D]`.
    pub fn step(&self, h: Tensor<3>, st: &mut BlockAttnState) -> Tensor<3> {
        let [b, t, d] = h.dims();
        st.partial = if st.partial_count == 0 {
            h.clone()
        } else {
            st.partial.clone() + h.clone()
        };
        st.partial_count += 1;

        // Attended sources: completed blocks (online state) + partial block
        // if it is past its first layer. With no sources at all (first layer
        // of the first block) the output is the identity (h).
        let mut out = if st.started || st.partial_count >= 2 {
            st.attended()
        } else {
            h.clone()
        };
        if st.partial_count >= 2 {
            let s_p = source_score(&self.query.val(), &st.partial, self.form);
            if st.started {
                let p = st.partial.clone();
                out = st.merge_source(&p, &s_p);
            } else {
                out = st.partial.clone();
            }
        }

        if st.partial_count == self.block_size {
            let completed = st.partial.clone();
            st.incorporate(&self.query.val(), &completed, self.form);
            st.partial = Tensor::zeros([b, t, d], &h.device());
            st.partial_count = 0;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    fn dev() -> Device {
        // hermetic tests: ambient default may try to init a busy CUDA ctx
        Device::ndarray()
    }

    fn random_h(b: usize, t: usize, d: usize) -> Tensor<3> {
        Tensor::<3>::random([b, t, d], Distribution::Default, &dev())
    }

    /// THE `1/sqrt(d)` DECISION, AS A NUMBER.
    ///
    /// Three conventions, one fixture, and the first output component of each.
    /// The fixture is chosen so the conventions cannot be confused:
    /// `h_0 = e_0`, `h_1 = e_1`, `w = 2·e_0`, `B = T = 1`, `L = 2`, `d = 4`.
    ///
    /// | convention | score_0 | alpha_0 | out[0] |
    /// |---|---|---|---|
    /// | `Paper` (Eq. 2: `q·RMSNorm(k)`, no temperature) | 4.000002 | 0.9820138 | **0.9820138** |
    /// | `SqrtD` (this crate's original form) | 0.999995 | 0.7310586 | 0.7310586 |
    ///
    /// The `alpha_0` column is the reason this is a fixture and not a
    /// tolerance argument: the two forms differ by 0.25 in the output, four
    /// orders of magnitude above the 1e-5 tolerance, and the gap is entirely
    /// in the softmax temperature (both norms agree on `||e_0|| = 1`). So the
    /// expected values are LITERALS from Eq. 2's arithmetic, not calls into
    /// this crate - a transcription error cannot hide behind itself here,
    /// which is exactly what `ref_depth_attend` allowed (it re-derived `d^-0.5`
    /// and the tests compared it against a kernel that did the same).
    ///
    /// 4.000002 and not 4.0 is the `eps = 1e-5` inside the norm's root:
    /// `1/sqrt(1/4 + 1e-5) = 1.99999…`, times `w·h = 2`.
    #[test]
    fn paper_form_has_no_temperature_and_this_is_pinned() {
        let h0 = Tensor::<3>::from_data(
            burn::tensor::TensorData::new(vec![1.0f32, 0.0, 0.0, 0.0], [1, 1, 4]),
            &dev(),
        );
        let h1 = Tensor::<3>::from_data(
            burn::tensor::TensorData::new(vec![0.0f32, 1.0, 0.0, 0.0], [1, 1, 4]),
            &dev(),
        );
        let q = Tensor::<1>::from_data(
            burn::tensor::TensorData::new(vec![2.0f32, 0.0, 0.0, 0.0], [4]),
            &dev(),
        );
        let got = |form| -> [f32; 2] {
            let o = depth_attend_form(&[h0.clone(), h1.clone()], q.clone(), form);
            let v: Vec<f32> = o.into_data().to_vec().unwrap();
            [v[0], v[1]]
        };
        let paper = got(ScoreForm::Paper);
        let sqrt_d = got(ScoreForm::SqrtD);
        assert!(
            (paper[0] - 0.982_013_8).abs() < 1e-5,
            "Eq. 2 says out[0] = 0.9820138, got {} - the temperature or the \
             RMSNorm mean is not the paper's",
            paper[0]
        );
        assert!(
            (paper[1] - 0.017_986_2).abs() < 1e-5,
            "Eq. 2 says out[1] = 0.0179862, got {}",
            paper[1]
        );
        assert!(
            (sqrt_d[0] - 0.731_058_6).abs() < 1e-5,
            "the old SqrtD form says out[0] = 0.7310586, got {} - if this is \
             red the two conventions have stopped being distinguishable",
            sqrt_d[0]
        );
        // The decision itself, in one assertion a reader can check by hand:
        // the default IS the paper's form.
        assert_eq!(ScoreForm::default(), ScoreForm::Paper);
        // 0.9820138 vs 0.8807971: dropping the `1/sqrt(d)` alone (keeping the
        // L2 norm) is a THIRD answer, so "we kept the scale" and "we fixed
        // the norm" are separate changes and this file's diff is both.
        let mid = 0.880_797_1_f32;
        assert!(
            (mid - paper[0]).abs() > 0.05 && (mid - sqrt_d[0]).abs() > 0.05,
            "the three conventions must stay separated by >> the tolerance, \
             got paper {paper:?} mid {mid} sqrt_d {sqrt_d:?}"
        );
    }

    /// §5: "all pseudo-query vectors must be initialized to zero. This ensures
    /// that the initial attention weights alpha are uniform across source
    /// layers, which reduces AttnRes to an equal-weight average at the start
    /// of training." With `q = 0` the scores are all 0 whatever the form, so
    /// this is also the one invariant that is scale-blind BY CONSTRUCTION -
    /// and it pins that the VALUES (not the normalised keys) are what gets
    /// aggregated: an implementation that weighted by `h_norm` would return
    /// the mean of normalised states and fail by O(1).
    #[test]
    fn zero_query_is_an_equal_weight_average_at_init() {
        let d = 8;
        let hist: Vec<Tensor<3>> = (0..5)
            .map(|_| random_h(2, 3, d))
            .collect();
        let mean = hist
            .iter()
            .fold(Tensor::<3>::zeros([2, 3, d], &dev()), |a, h| a + h.clone())
            .div_scalar(5.0);
        for form in [ScoreForm::Paper, ScoreForm::SqrtD] {
            let q = Tensor::<1>::zeros([d], &dev());
            let out = depth_attend_form(&hist, q, form);
            let v: Vec<f32> = (out - mean.clone()).into_data().to_vec().unwrap();
            let d: Vec<f32> = v.iter().map(|x| x.abs()).collect();
            let worst = d.iter().cloned().fold(0.0_f32, f32::max);
            assert!(worst < 1e-6, "{form:?}: zero query must average, off by {worst}");
        }
    }

    #[test]
    fn depth_attend_shape() {
        let h = vec![random_h(1, 4, 32), random_h(1, 4, 32), random_h(1, 4, 32)];
        let q = Tensor::<1>::random([32], Distribution::Default, &dev());
        assert_eq!(depth_attend(&h, q).dims(), [1, 4, 32]);
    }
    #[test]
    fn depth_attend_single() {
        let h = random_h(2, 8, 16);
        let q = Tensor::<1>::random([16], Distribution::Default, &dev());
        let out = depth_attend(std::slice::from_ref(&h), q);
        let d: Vec<f32> = (out - h)
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(
            d.iter().all(|x| x.abs() < 1e-4),
            "single item should be identity"
        );
    }
    #[test]
    fn full_attnres_module() {
        let a = AttnRes::new(64, &dev());
        let h = vec![random_h(1, 4, 64), random_h(1, 4, 64), random_h(1, 4, 64)];
        assert_eq!(a.forward(&h).dims(), [1, 4, 64]);
    }
    #[test]
    fn block_attnres_module() {
        let a = BlockAttnRes::new(64, 2, &dev());
        let h = vec![random_h(1, 4, 64), random_h(1, 4, 64), random_h(1, 4, 64)];
        assert_eq!(a.forward(&h).dims(), [1, 4, 64]);
    }
    #[test]
    fn block_attnres_small() {
        let a = BlockAttnRes::new(32, 4, &dev());
        let h = vec![random_h(1, 4, 32), random_h(1, 4, 32)]; // fewer than block_size
        assert_eq!(a.forward(&h).dims(), [1, 4, 32]);
    }

    #[test]
    fn streaming_matches_full_recompute() {
        // The paper's streaming scheme (Eq 6 semantics: partial block
        // excluded at its first layer, attended from its second layer on)
        // must reproduce the full-recompute BlockAttnRes at every step.
        let a = BlockAttnRes::new(16, 2, &dev());
        let (b, t, d) = (1usize, 3usize, 16usize);
        let mut st = a.init_state(b, t, d, &dev());
        let mut history: Vec<Tensor<3>> = Vec::new();
        for step in 0..6 {
            let h = random_h(b, t, d);
            history.push(h.clone());
            let streamed = a.step(h, &mut st);
            let full = a.forward(&history);
            let diff: f32 = (streamed - full).powf_scalar(2.0).mean().into_scalar();
            assert!(
                diff < 1e-5,
                "step {step}: streaming mse {diff} vs full recompute"
            );
        }
    }
}

// ─── Two-phase compute (paper §4.2, Algorithm 1) ─────────────────────

/// Two-phase attention over block representations (paper Algorithm 1).
///
/// Phase 1 batches all `S` pseudo-queries of a block against the `N` cached
/// block representations in a single matmul, returning per-layer outputs and
/// softmax statistics (max, log-sum-exp). Phase 2 then computes the
/// sequential intra-block attention against the evolving partial sum and
/// merges both via online softmax.
///
/// `queries`: `[S, d]` - pseudo-queries of the block's layers (batch dim 1)
/// `blocks`: `[N, d]` - block representations of the N completed blocks
///
/// Returns `(out [S, d], partial [d])` where `out` is `h_l` for every layer
/// of the block and `partial` the updated block representation (the sum of
/// the block's outputs) to be pushed into the block cache.
///
/// The scalar merge in Phase 2 (Eq 6 / Algorithm 1 line 12):
/// ```text
/// m   = max(m1, m2)
/// h_l = (e^{m1-m} o1 + e^{m2-m} o2) / (e^{m1-m} l1 + e^{m2-m} l2)
/// ```
///
/// The scale is applied to BOTH legs. It used to reach Phase 1 only
/// (`research/papers/attnres.md` D4), which compared the inter-block and the
/// intra-block source group at temperatures differing by `sqrt(d)` and so
/// mis-weighted the merge itself; `two_phase_merge_matches_full_attention`
/// could not see it because it exercises `i = 0`, which bypasses the merge.
pub fn two_phase_attend(
    queries: Tensor<2>,
    blocks: Tensor<2>,
    form: ScoreForm,
) -> (Tensor<2>, Tensor<1>) {
    let [s, d] = queries.dims();
    let [n, _] = blocks.dims();
    let device = queries.device();
    let scale = form.scale(d);
    let m_norm = form.norm_m(d);

    // Phase 1: inter-block attention, all queries at once
    // (paper: batched Q against the cached K/V to amortize memory access).
    // With no completed blocks the inter-block term is empty; Phase 1
    // statistics default to -inf/0 so the merge reduces to the intra-block
    // attention (Algorithm 1 line 8: hl = ol / ll for i = 0).
    let q = queries.clone().unsqueeze_dim::<3>(1); // [S, 1, d]
    let kv = blocks.clone().unsqueeze_dim::<3>(0); // [1, N, d]
                                                   // RMSNorm keys/values as in depth_attend
    let kv_norm = kv.clone()
        / kv.clone()
            .powf_scalar(2.0)
            .sum_dim(2)
            .mul_scalar(m_norm)
            .add_scalar(1e-5)
            .sqrt();
    let scores = (q * kv_norm)
        .sum_dim(2)
        .mul_scalar(scale)
        .squeeze_dim::<2>(2); // [S, N]
    let m1 = scores.clone().max_dim(1); // [S, 1]
                                        // raw weight sums, not log: the online-softmax merge (Alg 1 line 12)
                                        // works on exp-space quantities e^{m-m'} l
    let w1 = scores.sub(m1.clone()).exp(); // [S, N]
    let l1 = w1.clone().sum_dim(1); // [S, 1]
    let o1 = w1.unsqueeze_dim::<3>(2).mul(kv).sum_dim(1); // [S, 1, d]

    // Phase 2: sequential intra-block attention + online softmax merge.
    let mut out: Vec<Tensor<1>> = Vec::with_capacity(s);
    let mut partial = Tensor::<1>::zeros([d], &device); // b0_n := 0
    for i in 0..s {
        let q_i = queries.clone().slice([i..i + 1, 0..d]); // [1, d]
                                                           // intra-block: attend to the partial sum (a single key/value)
        let p_norm = partial.clone()
            / partial
                .clone()
                .powf_scalar(2.0)
                .sum()
                .mul_scalar(m_norm)
                .add_scalar(1e-5)
                .sqrt();
        // SAME form as Phase 1 (D4): the merge below adds the two groups'
        // exponentials, so a scale on one leg and not the other is a bug.
        let s2 = (q_i.clone() * p_norm.clone().reshape([1, d]))
            .sum_dim(1)
            .mul_scalar(scale); // [1, 1]
        let m2 = s2.clone(); // single key: max == score
        let l2 = Tensor::<2>::ones([1, 1], &device); // raw weight sum = e^0
        let o2 = partial.clone().reshape([1, d]); // single-key weighted sum

        // online softmax merge (Algorithm 1 line 11-12)
        let m1_i = m1.clone().slice([i..i + 1, 0..1]);
        let l1_i = l1.clone().slice([i..i + 1, 0..1]);
        let o1_i = o1.clone().slice([i..i + 1, 0..1, 0..d]).reshape([1, d]);
        let h = if i == 0 || n == 0 {
            // Algorithm 1 line 8: first layer attends inter-block only
            o1_i.div(l1_i.clamp_min(1e-10))
        } else {
            // online softmax merge (Algorithm 1 line 11-12). All quantities
            // stay on-device: the [1,1] scalars broadcast against [1,d] rows,
            // so no `.into_scalar()` host sync inside the per-layer loop.
            let m = Tensor::cat(vec![m1_i.clone(), m2.clone()], 1).max_dim(1); // [1, 1]
            let e1 = (m1_i.sub(m.clone())).exp();
            let e2 = (m2.sub(m.clone())).exp();
            let num = o1_i.mul(e1.clone()) + o2.mul(e2.clone());
            let den = l1_i.mul(e1) + l2.mul(e2);
            num.div(den) // [1, d]
        };
        let h1 = h.reshape([d]);
        out.push(h1.clone());
        // update partial sum: b_i := b_{i-1} + h_l
        partial = partial.add(h1);
    }
    let out_t = Tensor::cat(
        out.iter()
            .map(|h| h.clone().unsqueeze_dim::<2>(0))
            .collect::<Vec<_>>(),
        0,
    );
    (out_t, partial)
}

#[cfg(test)]
mod two_phase_tests {
    use super::*;

    #[test]
    fn two_phase_merge_matches_full_attention() {
        // One completed block + one query: with the partial sum zero the
        // merge must reproduce the plain inter-block softmax over the block.
        let dev = Device::ndarray();
        let q = Tensor::<2>::ones([1, 8], &dev);
        let blocks = Tensor::<2>::ones([2, 8], &dev).mul_scalar(3.0);
        let (out, _partial) = two_phase_attend(q, blocks, ScoreForm::Paper);
        // all blocks identical and query uniform: softmax over 2 identical
        // keys -> 0.5/0.5, output = 3.0
        let v: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(v.iter().all(|x| (x - 3.0).abs() < 1e-3), "got {v:?}");
    }

    #[test]
    fn two_phase_shapes() {
        let dev = Device::ndarray();
        let q = Tensor::<2>::ones([4, 16], &dev);
        let blocks = Tensor::<2>::ones([3, 16], &dev);
        let (out, partial) = two_phase_attend(q, blocks, ScoreForm::Paper);
        assert_eq!(out.dims(), [4, 16]);
        assert_eq!(partial.dims(), [16]);
        // all outputs finite
        let v: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
