//! # burn-muon-plus — Muon+ optimizer for Burn
//!
//! | arXiv | Component | What |
//! |-------|-----------|------|
//! | [2602.21545](https://arxiv.org/abs/2602.21545) **v3** | `Muon+` | Muon + one post-polar normalization step (zero extra optimizer state) |
//!
//! # Muon+
//!
//! Newton–Schulz polar iterations flatten the singular spectrum of the
//! momentum matrix, but in practice they **amplify** column/row norm imbalance
//! in the update ("post-polar imbalanced update problem", 2602.21545 §2.1).
//! This imbalance tightens the second-order term in a blockwise descent
//! analysis and shrinks Muon's largest stable step size.
//!
//! Muon+ inserts a single normalization step after orthogonalization:
//!
//! ```text
//! M_t  = μ·M_{t-1} + (1-μ)·G_t
//! O_t  = Norm_(d)(Ortho(M_t))      # NS polar + row/col normalization
//! W_t  = W_{t-1} - η·√(m/n)·O_t
//! ```
//!
//! - `Norm_col(X) = X·D_col⁻¹` (unit-L2 columns)
//! - `Norm_row(X) = D_row⁻¹·X` (unit-L2 rows)
//! - `ColRow`/`RowCol` = sequential composition, Eqs. (7)/(8)
//!
//! **Cite v3.** v1 has appendices A-C; v2/v3 have A-G, and *every* coefficient
//! table (App. D) is in the appendices that v1 does not have. A claim of the
//! form "the paper prints no NS coefficients" is true of v1 and false of the
//! current paper.
//!
//! The 37% figure the paper's abstract quotes ("speeds up the pre-training up
//! to **37.1%**", §1; Table 5) is **v3-only** and is **optimizer steps to a
//! fixed target loss** (GPT-Base: 3447 Muon steps vs 2515 Muon+ steps), not
//! wall-clock and not per-step time - §3.3 says the per-step cost is
//! "nearly the same".
//!
//! Non-matrix parameters (biases, norms, embeddings) use AdamW — the paper's
//! recipe (§3.1), matching Muon's convention.
//!
//! # Example
//!
//! ```ignore
//! use burn_muon_plus::{MuonPlusConfig, NormDir};
//!
//! let mut optimizer = MuonPlusConfig::new()
//!     .with_norm_dir(NormDir::ColRow)   // one of the paper's two orders
//!     .init();
//! let model = optimizer.step(lr, model, grads);
//! ```
use burn::config::Config;
use burn::optim::{LearningRate, ModuleOptimizer, Optimizer, RecordState, StateSink, StateSource};
/// Fused kernels. `ns_combine_cuda` has **no production caller**: its only one
/// was the factored NS branch in [`MuonPlus::orthogonalize`], deleted as
/// unsatisfiable (the same polynomial, behind a guard that could never be
/// true). It is kept because its own test in this module still pins the
/// kernel against the tensor-op expression it computes.
#[cfg(feature = "cuda")]
pub mod fused_kernels;

use burn::tensor::{Device, ElementConversion, Tensor};

/// Fused kernels asked for and answered with the tensor-ops path, as
/// `(momentum, finalize)` — the same field `dormouse-train` prints as
/// `muon_skipped=mom/finalize`, from THIS implementation of the Muon update.
///
/// The three `*_cuda` entry points return `false` when the tensor is not a
/// bare cubecl one (an autodiff wrapper, a different backend) or is empty, and
/// the tensor path computes the same function. That is the ADR-0019 COUNTED
/// mark: without a number a fused kernel can be dead for a whole run and the
/// log reads identically. It was dead-able here with nothing counting it —
/// `norm_colrow_cuda` especially, which is asked on every `ColRow` update (the
/// trainer's default) and had no counter at all. Present on every build, so
/// the answer is `(0, 0)` with no `cuda` feature rather than "unavailable".
pub fn fused_skipped() -> (u64, u64) {
    (
        SKIPPED_MOMENTUM.load(std::sync::atomic::Ordering::Relaxed),
        SKIPPED_FINALIZE.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Same mark, for the `ColRow` normalization kernel alone.
///
/// It is a separate number because it is asked on a different cadence from the
/// other two: `norm_colrow_cuda` is called once per `ColRow` update (the
/// trainer's default, so every Muon+ step), while `momentum_cuda` needs live
/// state and `finalize_cuda` is the last pass of the same step. Before this,
/// `norm_colrow_cuda` was the only fused kernel on the Muon+ path with NO
/// counter, so a run where the normalization was silently on the tensor path
/// for its whole life printed an identical log. **NOT yet printed**: the eval
/// line in `dormouse-train` (`lib.rs`, the `muon_skipped=` field) destructures
/// a 2-tuple and that file is not ours; the count exists so the number is
/// obtainable, and wiring it into the log line is a one-token change there.
pub fn fused_norm_skipped() -> u64 {
    SKIPPED_NORM.load(std::sync::atomic::Ordering::Relaxed)
}
static SKIPPED_MOMENTUM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SKIPPED_FINALIZE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SKIPPED_NORM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Called on the tensor-path arm of a fused call.
#[cfg(feature = "cuda")]
fn count_skip(which: Skip) {
    let c = match which {
        Skip::Momentum => &SKIPPED_MOMENTUM,
        Skip::Finalize => &SKIPPED_FINALIZE,
        Skip::Norm => &SKIPPED_NORM,
    };
    c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
enum Skip {
    Momentum,
    Finalize,
    Norm,
}

/// 1.0 if `grad` carries any signal, 0.0 if it is exactly zero. A `[1]`
/// tensor, so it broadcasts over the update whatever the update's rank.
///
/// A gradient that is exactly zero carries no signal, and the momentum turns
/// it into a FULL-magnitude step in the STALE direction: the Newton-Schulz
/// normalization rescales whatever it is handed to unit Frobenius norm, and
/// the decayed momentum is not zero. So a step the trainer masked as a no-op
/// (the NaN firewall zeroes the gradients on device) still moved every weight,
/// which is precisely the poisoning that killed runs on 2026-09-26/27. Zero
/// gradient now means zero update, decided on device with no host
/// synchronization. The factor is built with `mask_fill` on a float tensor
/// because the project rule is to never build a numeric indicator from a bool
/// tensor on device and count on the host instead (ADR-0018 rule 2) - not
/// because the cast is broken, which was believed until ADR-0016 measured it
/// correct on both backends. The rule outlived the reason.
///
/// Public, and free function rather than a private helper, because the Muon
/// update is implemented twice: here, and per-head over the attention Q/K
/// weights in `dormouse-train`'s `HeadWiseMuon` (which reimplements the 2D
/// branch to slice per head). Two copies of the rule is how the second one
/// came to not have it. `zero_gradient_does_not_move_the_parameter` in
/// `tests/zero_grad.rs` pins this one and
/// `headwise_zero_gradient_does_not_move_the_parameter` in `dormouse-train`
/// pins the other.
pub fn signal_mask<const D: usize>(grad: &Tensor<D>) -> Tensor<1> {
    Tensor::<1>::ones([1], &grad.device()).mask_fill(
        grad.clone()
            .powf_scalar(2.0)
            .sum()
            .greater_elem(0.0)
            .bool_not(),
        0.0,
    )
}

/// Muon+ post-polar normalization direction (2602.21545 §2.3, Eqs. (3)-(8)).
///
/// `dormouse-train` uses [`NormDir::ColRow`], which is the paper's Eq. (7).
#[derive(Clone, Debug, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NormDir {
    /// Normalize each column of the orthogonalized update to unit L2 norm.
    Col,
    /// Normalize each row of the orthogonalized update to unit L2 norm.
    Row,
    /// Column then row — 2602.21545 Eq. (7), `Norm_(col_row) := Norm_(row)(Norm_(col)(X))`.
    ///
    /// Not "the paper's best", which is what this doc used to say. The paper
    /// never ranks the two orders against each other: §3.4 says only that
    /// "applying bi-directional normalization consistently outperforms
    /// single-directional normalization", and its own Table 6 has the two
    /// orders within noise of one another (LLaMA-60M 25.25/25.25, 130M
    /// 18.65/18.68, 350M 13.41/13.44) with the winner flipping by model.
    /// `ColRow` is the choice the trainer makes; the *paper's* code default is
    /// `NormDir::Col` (App. C, `d="col"` in `norm_dir`'s signature).
    ColRow,
    /// Row then column — 2602.21545 Eq. (8), `Norm_(row_col) := Norm_(col)(Norm_(row)(X))`.
    RowCol,
}

/// Newton-Schulz quintic coefficients: `(a, b, c) = (3.4445, -4.7750, 2.0315)`.
///
/// This triple is the paper's own default polar operator, printed in the
/// current version: v3 App. D.1 "Jordan Coefficients", verbatim, "In [15],
/// the coefficients are set to `(a,b,c)=(3.4445, -4.7750, 2.0315)`", and §3.1
/// says "For the polar operator, we adopt the same configuration as in Jordan
/// et al. (2024)". v1 has appendices A-C only, so the sentence quoted above
/// (and every other coefficient table) exists in v2/v3 and not in v1 - cite
/// **v3**. The same appendix prints the other two families the paper ablates:
/// App. D.2 "You" (per-iteration rationals) and App. D.3 "PolarExpress",
/// whose schedule terminates at `(1.875, -1.25, 0.375)`.
pub const NS_COEFFS: (f32, f32, f32) = (3.4445, -4.775, 2.0315);

/// Configuration for the hybrid Muon+ (2D) + AdamW (1D) optimizer.
#[derive(Config, Debug)]
pub struct MuonPlusConfig {
    /// Momentum for 2D Muon updates. Default 0.95 (paper).
    #[config(default = 0.95)]
    pub momentum: f64,
    /// AdamW beta_1 for the 1D fallback group.
    #[config(default = 0.9)]
    pub beta_1: f64,
    /// AdamW beta_2 for the 1D fallback group.
    #[config(default = 0.95)]
    pub beta_2: f64,
    /// AdamW epsilon for the 1D fallback group.
    #[config(default = 1e-8)]
    pub epsilon: f64,
    /// Newton-Schulz iterations. Default 5 (paper).
    #[config(default = 5)]
    pub ns_steps: usize,
    /// Newton-Schulz quintic coefficients.
    #[config(default = "NS_COEFFS")]
    pub ns_coeffs: (f32, f32, f32),
    /// Muon+ post-polar normalization direction. `None` = plain Muon.
    ///
    /// `None` is NOT Muon+ and the paper never prescribes it: Eq. (4) is
    /// `O_t = Norm_(d)(Ortho(M_t))`, so the un-normalized form is the
    /// **baseline** the paper measures Muon+ against, and the paper's own
    /// function default is the *narrower* `Norm_(col)` (App. C,
    /// `def norm_dir(X, d="col", ...)`). A caller who never sets this gets the
    /// thing Muon+ is being compared against, silently - the config default is
    /// a knob, not a recommendation, and `dormouse-train` sets `ColRow`
    /// explicitly. `normalize` with `None` is the identity, pinned by
    /// `norm_dir_none_is_plain_muon`.
    #[config(default = "None")]
    pub norm_dir: Option<NormDir>,
    /// Decoupled weight decay (applied to both groups).
    #[config(default = 0.0)]
    pub weight_decay: f64,
}

/// State of the hybrid optimizer for one parameter.
///
/// 2D params use Muon momentum; 1D params use AdamW moments.
#[derive(RecordState, Clone, derive_new::new)]
pub struct MuonPlusState<const D: usize> {
    /// Muon momentum.
    pub mu_momentum: Option<Tensor<D>>,
    /// AdamW first moment (1D group).
    pub ad_moment_1: Option<Tensor<D>>,
    /// AdamW second moment (1D group).
    pub ad_moment_2: Option<Tensor<D>>,
    /// AdamW step counter (1D group).
    pub ad_time: Option<i64>,
}

/// Hybrid Muon+ (2D) + AdamW (1D) optimizer.
#[derive(Clone)]
pub struct MuonPlus {
    momentum: f64,
    beta_1: f64,
    beta_2: f64,
    epsilon: f64,
    ns_steps: usize,
    ns_coeffs: (f32, f32, f32),
    norm_dir: Option<NormDir>,
    weight_decay: f64,
}

/// Transpose `g` if it is taller than it is wide, then scale to unit Frobenius
/// norm. Returns the matrix and whether it was transposed, so the caller can
/// swap it back.
///
/// The `cols >= rows` guarantee is what the NS loop in
/// [`MuonPlus::orthogonalize`] is shaped around. It is pinned by
/// `orient_and_normalize_is_never_taller_than_wide`: a shape-handling edit here
/// is free to change it, and nothing downstream checks.
fn orient_and_normalize<const D: usize>(g: Tensor<D>) -> (Tensor<D>, bool) {
    let dims = g.dims();
    let transposed = dims[D - 2] > dims[D - 1];
    let x = if transposed {
        g.swap_dims(D - 2, D - 1)
    } else {
        g
    };
    let norm = x.clone().mul(x.clone()).sum().sqrt().clamp_min(1e-7);
    (x.div(norm.unsqueeze()), transposed)
}

impl MuonPlus {
    /// Orthogonalize `g` via Newton-Schulz (zeroth power, tall-matrix aware).
    ///
    /// Public for testing: returns the nearest-orthogonal approximation of
    /// `g` under the Frobenius norm (2602.21545 §1, Eq. 1).
    pub fn orthogonalize<const D: usize>(&self, g: Tensor<D>) -> Tensor<D> {
        let (mut x, transposed) = orient_and_normalize(g);

        let (a, b, c) = self.ns_coeffs;
        // `orient_and_normalize` leaves x never taller than wide, so `X Xᵀ` is
        // always the smaller factor and this is the cheap form: 2·r²c + r³
        // against the factored `a·x + b·M x + c·M²x` (M = X Xᵀ) at 3·r²c.
        // Since r ≤ c, r³ ≤ r²c, so the factored grouping can never be the
        // cheaper of the two here — it is the same polynomial regrouped.
        for _ in 0..self.ns_steps {
            let xt = x.clone().swap_dims(D - 2, D - 1);
            let xx = x.clone().matmul(xt); // X X^T
            let xx2 = xx.clone().matmul(xx.clone()); // (X X^T)²
            let poly = xx.mul_scalar(b).add(xx2.mul_scalar(c));
            x = x.clone().mul_scalar(a).add(poly.matmul(x.clone()));
        }

        if transposed {
            x.swap_dims(D - 2, D - 1)
        } else {
            x
        }
    }

    /// Batched Newton-Schulz for a rank-D tensor (D >= 3) whose SLICES are the
    /// matrices to orthogonalize: the same NS loop as [`Self::orthogonalize`]
    /// run once per step over `[B, r, c]`, with the Frobenius normalization
    /// per slice instead of per tensor. Matmul over the batch dims is math
    /// per slice - no singular directions mix across slices - which is what
    /// makes a per-head (`[n_heads, head_dim, d]`) NS the same object the
    /// per-head loop computes, in one launch set.
    ///
    /// LOUD on a taller-than-wide slice: the per-slice transpose arm of
    /// [`Self::orthogonalize`] has no batched counterpart, and silently
    /// skipping it would compute a different polynomial. Callers slice flat
    /// (`[64, 768]` blocks are never taller) - assert, don't guess.
    pub fn orthogonalize_batched<const D: usize>(&self, g: Tensor<D>) -> Tensor<D> {
        assert!(D >= 3, "orthogonalize_batched is for batched slices (D >= 3); use orthogonalize");
        let dims = g.dims();
        assert!(
            dims[D - 2] <= dims[D - 1],
            "batched NS: slice [{}, {}] is taller than wide - no per-slice transpose arm \
             exists; reshuffle the batch so slices are wide",
            dims[D - 2],
            dims[D - 1]
        );
        // orient_and_normalize's scale, per slice: Frobenius over BOTH trailing
        // dims, keep the batch. Two sum_dims, not a global sum, is the whole
        // point of the batched form.
        let mut x = g;
        let fro = x
            .clone()
            .mul(x.clone())
            .sum_dim(D - 1)
            .sum_dim(D - 2)
            .sqrt()
            .clamp_min(1e-7);
        x = x.div(fro);

        let (a, b, c) = self.ns_coeffs;
        for _ in 0..self.ns_steps {
            let xt = x.clone().swap_dims(D - 2, D - 1);
            let xx = x.clone().matmul(xt); // X X^T, per slice
            let xx2 = xx.clone().matmul(xx.clone()); // (X X^T)^2
            let poly = xx.mul_scalar(b).add(xx2.mul_scalar(c));
            x = x.clone().mul_scalar(a).add(poly.matmul(x.clone()));
        }
        x
    }

    /// Muon+ post-polar normalization (2602.21545 §2.3, Eqs. (3)-(8)).
    ///
    /// `Norm_col(X) = X·D_col⁻¹`, `Norm_row(X) = D_row⁻¹·X`; compositions are
    /// applied sequentially, and in the order the paper's equations name:
    /// Eq. (7) `ColRow := Norm_(row)(Norm_(col)(X))` and Eq. (8) `RowCol :=
    /// Norm_(col)(Norm_(row)(X))`. Pinned against the equations, order
    /// included, by `normalization_matches_the_paper_equations` in
    /// `tests/self_checks.rs` - a swapped composition is a silent, and wrong,
    /// way to lose a BPB.
    ///
    /// Returns `x` unchanged when `norm_dir` is `None`, which is plain Muon
    /// (see [`MuonPlusConfig::norm_dir`]).
    pub fn normalize<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        match self.norm_dir {
            None => x,
            Some(NormDir::Col) => Self::norm_col(x),
            Some(NormDir::Row) => Self::norm_row(x),
            Some(NormDir::ColRow) => {
                let mut xm = x;
                #[cfg(feature = "cuda")]
                {
                    let fused = D == 2 && crate::fused_kernels::norm_colrow_cuda(&mut xm, 1e-7);
                    if !fused {
                        if D != 2 {
                            // A batched (rank >= 3) tensor on the ColRow fused kernel
                            // would stride-walk the BATCH dim as row pitch - silent
                            // corruption, not a fallback. The tensor composite below
                            // is the defined batched arm, so it is counted as a skip
                            // only where the fused kernel was actually asked (D == 2).
                        } else {
                            count_skip(Skip::Norm);
                        }
                        xm = Self::norm_row(Self::norm_col(xm));
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    xm = Self::norm_row(Self::norm_col(xm));
                }
                xm
            }
            Some(NormDir::RowCol) => Self::norm_col(Self::norm_row(x)),
        }
    }

    /// `Norm_(col)` (Eq. (3)-(4)): each column scaled to unit L2 norm.
    ///
    /// The epsilon is a floor on the divisor *after* the root, `clamp_min(1e-7)`,
    /// where App. C's `norm_dir` puts it *inside*:
    /// `(X.square().sum(dim=-2) + eps).sqrt()` with `eps = 1e-8`. The two are
    /// the same operator wherever the divisor is not itself tiny: with a
    /// column norm `v`, the paper divides by `sqrt(v² + 1e-8)` and this by
    /// `max(v, 1e-7)`, and for `v ≥ 1e-3` the two differ by under 1e-2
    /// relative. The floor is what makes them agree at all: an exactly-zero
    /// column divides to exactly zero under both. The live input is a
    /// Newton-Schulz output whose singular values are driven to ~1, so
    /// `v ≈ 1` and the two differ by ~1e-8 relative - see
    /// `normalization_matches_the_paper_equations`.
    ///
    /// The fused `norm_colrow_cuda` applies the *same* `clamp_min(eps)` rule,
    /// so the fused and tensor paths cannot disagree in the corner either.
    /// `sum_dim` keeps the axis (size 1) → broadcasts directly.
    fn norm_col<const D: usize>(x: Tensor<D>) -> Tensor<D> {
        let col_norms = x
            .clone()
            .mul(x.clone())
            .sum_dim(D - 2)
            .sqrt()
            .clamp_min(1e-7);
        x.div(col_norms)
    }

    /// `Norm_(row)` (Eq. (5)-(6)): each row scaled to unit L2 norm. Same
    /// epsilon placement as [`MuonPlus::norm_col`].
    fn norm_row<const D: usize>(x: Tensor<D>) -> Tensor<D> {
        let row_norms = x
            .clone()
            .mul(x.clone())
            .sum_dim(D - 1)
            .sqrt()
            .clamp_min(1e-7);
        x.div(row_norms)
    }
}

impl Optimizer for MuonPlus {
    type State<const D: usize> = MuonPlusState<D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        let state = state.unwrap_or_else(|| MuonPlusState::new(None, None, None, None));

        // A zero gradient must mean a zero update, on device, with no host
        // read. See [`signal_mask`] for why that is not automatic here, and
        // for the second implementation of this same rule in `dormouse-train`.
        let g_active = signal_mask(&grad);

        let (updated, state) = if D == 2 {
            // --- Muon+ group ---
            // Rank decides the optimizer here, not identity, so a 3-D (or 4-D)
            // parameter silently lands in the AdamW branch below and is labelled
            // "1D" in that comment while not being 1-D. That is unreachable from
            // `dormouse-train`: `routing::check_installed` is a loud error on
            // any non-rank-2 parameter in a Muon+ group, and the live groups are
            // all 2-D. It is a property of this crate as a library, not of the
            // trainer, and the paper says nothing about it (Jordan collapses
            // conv filters to 2-D, `update.view(len(update), -1)`, rather than
            // handing them to AdamW). Reported, not fixed: making it loud would
            // change what a third-party caller can do with no evidence that
            // anything wants to.
            // M_t = μ·M_{t-1} + (1-μ)·G_t (fused kernel on CUDA). NO
            // Nesterov lerp, and that is the paper's version, not an
            // omission: Eq. (4) and App. C line 3 are exactly this line, while
            // `KellerJordan/muon`'s `muon_update` defaults to
            // `nesterov=True` and computes `grad.lerp_(momentum, beta)` on top
            // of it. Where this file and that one differ, this file is the one
            // quoting 2602.21545; the crate manifest's "Nesterov momentum"
            // description was the defect, not the code (fixed in Cargo.toml).
            // The cold-start `None => grad·(1-μ)` also matches: Jordan
            // pre-fills `momentum_buffer` with zeros and `lerp_`s the gradient
            // into it, so his first step is `(1-β)·G` too. It differs from
            // Eq. (4) only in what `M₀` is, and `orthogonalize` divides by the
            // Frobenius norm, so a global scale on the first step cancels.
            let mu = self.momentum.elem::<f32>();
            let momentum = match &state.mu_momentum {
                Some(m) => {
                    let mut mm = m.clone();
                    #[cfg(feature = "cuda")]
                    {
                        if !crate::fused_kernels::momentum_cuda(&mut mm, &grad, mu) {
                            count_skip(Skip::Momentum);
                            mm = mm
                                .clone()
                                .mul_scalar(mu)
                                .add(grad.clone().mul_scalar(1.0 - mu));
                        }
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        mm = mm
                            .clone()
                            .mul_scalar(mu)
                            .add(grad.clone().mul_scalar(1.0 - mu));
                    }
                    mm
                }
                None => grad.clone().mul_scalar(1.0 - mu),
            };
            // O_t = Norm_(d)(Ortho(M_t)), gated by "this step had a signal"
            let update = self
                .normalize(self.orthogonalize(momentum.clone()))
                .mul(g_active.unsqueeze());

            // W_t = W_{t-1} - η·max(1, m/n)^0.5·O_t
            //
            // DEVIATION FROM THE PAPER, and the only numeric one in this
            // function. 2602.21545 Eq. (4) and App. C Algorithm 1 line 10 are
            // `lr * (m / n) ** 0.5`, with no `max`. The `max(1, ·)` is Jordan
            // et al.'s: `update *= max(1, update.size(-2) / update.size(-1))**0.5`
            // in `KellerJordan/muon` `muon_update`, which this matches
            // expression for expression. §3.1 adopts "the same configuration
            // as in Jordan et al. (2024)" for the *polar operator*, and
            // Bernstein (2025) is the paper's own source for `√(m/n)`, so the
            // two documents disagree and this follows the implementation.
            //
            // It is INERT on every shape the trainer routes here: the two
            // Muon groups are the TSCT factors `[in, k]`/`[out, k]` (tall, so
            // `m/n > 1` and the `max` never binds) and the attention Q/K
            // weights `[n_heads·head_dim, d]` (square, so `m/n = 1` and both
            // forms are 1.0). `max(1, ·)` is 1.0 for every `m ≤ n`, so the
            // paper's `√(m/n)` - a step *shrink* on wide matrices - is the only
            // form with a live difference, and there is no wide matrix in
            // either group. Pinned by
            // `step_size_factor_is_jordans_max_one_and_is_inert_here`.
            let dims = tensor.dims();
            let (m, n) = (dims[0] as f64, dims[1] as f64);
            let lr_scaled = lr * (m / n).max(1.0).sqrt();

            let wd = (self.weight_decay as f32 * lr_scaled as f32).min(0.999);
            let mut updated = tensor.clone();
            #[cfg(feature = "cuda")]
            {
                if !crate::fused_kernels::finalize_cuda(&mut updated, &update, lr_scaled as f32, wd)
                {
                    count_skip(Skip::Finalize);
                    updated = updated
                        .clone()
                        .mul_scalar(1.0 - wd)
                        .sub(update.mul_scalar(lr_scaled as f32));
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                updated = updated
                    .clone()
                    .mul_scalar(1.0 - wd)
                    .sub(update.mul_scalar(lr_scaled as f32));
            }
            (
                updated,
                MuonPlusState::new(Some(momentum), None, None, None),
            )
        } else {
            // --- AdamW group (1D: biases, norms) ---
            let b1 = self.beta_1.elem::<f32>();
            let b2 = self.beta_2.elem::<f32>();
            let eps = self.epsilon.elem::<f32>();
            let t = state.ad_time.unwrap_or(0) + 1;

            let m = match &state.ad_moment_1 {
                Some(m) => m
                    .clone()
                    .mul_scalar(b1)
                    .add(grad.clone().mul_scalar(1.0 - b1)),
                None => grad.clone().mul_scalar(1.0 - b1),
            };
            let v = match &state.ad_moment_2 {
                Some(v) => v
                    .clone()
                    .mul_scalar(b2)
                    .add(grad.clone().powf_scalar(2.0).mul_scalar(1.0 - b2)),
                None => grad.clone().powf_scalar(2.0).mul_scalar(1.0 - b2),
            };

            // Fused bias correction: m·(1−β1^t)⁻¹ / (sqrt(v·(1−β2^t)⁻¹) + ε)
            let bc1 = 1.0 / (1.0 - b1.powi(t as i32));
            let bc2 = (1.0 - b2.powi(t as i32)).sqrt();
            let step = m
                .clone()
                .mul_scalar(bc1)
                .div(v.clone().sqrt().mul_scalar(bc2).add_scalar(eps));

            let decayed = tensor
                .clone()
                .mul_scalar(1.0 - (self.weight_decay as f32 * lr as f32).min(0.999));
            (
                decayed.sub(step.mul(g_active.unsqueeze()).mul_scalar(lr as f32)),
                MuonPlusState::new(None, Some(m), Some(v), Some(t)),
            )
        };
        (updated, Some(state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device) -> Self::State<D> {
        state.mu_momentum = state.mu_momentum.map(|t| t.to_device(device));
        state.ad_moment_1 = state.ad_moment_1.map(|t| t.to_device(device));
        state.ad_moment_2 = state.ad_moment_2.map(|t| t.to_device(device));
        state
    }
}

impl MuonPlusConfig {
    /// Build the optimizer from config.
    pub fn build(&self) -> MuonPlus {
        MuonPlus {
            momentum: self.momentum,
            beta_1: self.beta_1,
            beta_2: self.beta_2,
            epsilon: self.epsilon,
            ns_steps: self.ns_steps,
            ns_coeffs: self.ns_coeffs,
            norm_dir: self.norm_dir,
            weight_decay: self.weight_decay,
        }
    }

    /// Initialize the optimizer for a module.
    pub fn init(&self) -> ModuleOptimizer {
        ModuleOptimizer::from(self.build())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    /// The fused-kernel seam must be COUNTED, on every build, with a
    /// reachable non-zero answer. `dormouse-train` prints
    /// `muon_skipped=mom/finalize` on the eval line from these numbers, and
    /// the only thing that number is for is telling a reader the fused path
    /// answered instead of the tensor path (ADR-0019). With no counter the run
    /// looks the same either way — which is how the `ColRow` kernel came to
    /// have none at all.
    ///
    /// The CPU build can only reach the tensor-path arm, so this asserts the
    /// mechanism (the counter moves on a fallback) rather than the CUDA
    /// outcome (the kernel ran). The CUDA outcome is `fused_match_tensor` and
    /// `norm_colrow_match` in `fused_kernels.rs`, which assert the kernels
    /// *return true*, and that is the half a CPU build cannot see.
    #[test]
    fn fused_skips_are_counted() {
        // One `ColRow` update on a fresh state: `ColRow` is the default
        // direction, so this is the arm the trainer asks on every step.
        let dev = Default::default();
        let muon: MuonPlus = MuonPlusConfig::new()
            .with_norm_dir(Some(NormDir::ColRow))
            .build();
        let before = (fused_skipped(), fused_norm_skipped());
        let g = Tensor::<2>::random([8, 16], Distribution::Default, &dev);
        let (w, _) = muon.step(1e-3, Tensor::<2>::zeros([8, 16], &dev), g, None);
        assert!(
            w.dims() == [8, 16],
            "the step must still produce a parameter of the right shape"
        );
        let after = (fused_skipped(), fused_norm_skipped());
        // Without the `cuda` feature the fused kernels are never even asked, so
        // nothing increments and this asserts the build is a CPU build. With
        // the feature, a non-bare-ndarray tensor still falls back and the
        // counter must move. Either way the two must be consistent: a
        // non-zero count with no `cuda` build, or silence, is the bug.
        #[cfg(feature = "cuda")]
        assert!(
            after.1 > before.1,
            "a ColRow step on the tensor path did not count a norm-kernel skip: \
             {before:?} -> {after:?}"
        );
        #[cfg(not(feature = "cuda"))]
        assert_eq!(
            before, after,
            "with no cuda feature the fused kernels are never asked, so the \
             seam counters must be untouched: {before:?} -> {after:?}"
        );
    }

    /// The invariant [`MuonPlus::orthogonalize`]'s polynomial is shaped around:
    /// the oriented matrix is never taller than it is wide, so `X Xᵀ` is always
    /// the smaller factor. Every shape goes in, including the tall ones that
    /// are the only thing that could break it.
    #[test]
    fn orient_and_normalize_is_never_taller_than_wide() {
        let dev = Default::default();
        for (r, c) in [
            (1usize, 1usize),
            (8, 8),
            (32, 8),  // tall: must transpose
            (8, 32),  // wide
            (64, 1),  // extreme tall
            (1, 64),  // extreme wide
            (33, 32), // off by one, tall side
            (32, 33), // off by one, wide side
        ] {
            let g = Tensor::<2>::random([r, c], Distribution::Default, &dev);
            let (x, transposed) = orient_and_normalize(g);
            let dims = x.dims();
            assert!(
                dims[0] <= dims[1],
                "oriented {r}x{c} came out as {dims:?} — taller than wide"
            );
            assert_eq!(transposed, r > c, "transpose flag wrong for {r}x{c}");
            let norm = x.clone().mul(x).sum().sqrt().into_scalar::<f32>();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "oriented {r}x{c} is not unit-norm: {norm}"
            );
        }
    }
}

#[cfg(test)]
mod bench {
    use burn::tensor::{Distribution, Tensor};

    use std::time::Instant;

    /// BOTH helpers below skip the canonicalizing transpose that
    /// `orient_and_normalize` does, so `[8192, 512]` is iterated as a TALL
    /// matrix. `orthogonalize` can never see that shape: it transposes first,
    /// so the matrix it loops on is always `rows ≤ cols` and its `X Xᵀ` is
    /// always the small side. The "3.6× faster on [8192,512]" claim that used
    /// to sit above that comparison was produced by this bench, and it
    /// measured a shape production cannot present - on top of having no device
    /// flush (see `fused_kernels.rs`'s benches, which have the same defect).
    ///
    /// Kept, unused, `#[ignore]`d: it is the instrument an A/B of the two
    /// evaluation orders would need, and the arithmetic it is there to compare
    /// is the one the FLOP argument in `orthogonalize` settles without
    /// measuring. Run it on a flushed device before believing any ratio it
    /// prints.
    fn ns_old<const D: usize>(g: Tensor<D>) -> Tensor<D> {
        let mut x = g;
        let norm = x.clone().powf_scalar(2.0).sum().sqrt().clamp_min(1e-7);
        x = x.div(norm.unsqueeze());
        for _ in 0..5 {
            let xt = x.clone().swap_dims(D - 2, D - 1);
            let xx = x.clone().matmul(xt);
            let xx2 = xx.clone().matmul(xx.clone());
            let poly = xx.mul_scalar(-2.0).add(xx2.mul_scalar(0.25));
            x = x.clone().mul_scalar(3.25).add(poly.matmul(x.clone()));
        }
        x
    }

    fn ns_new<const D: usize>(g: Tensor<D>) -> Tensor<D> {
        let mut x = g;
        let norm = x.clone().powf_scalar(2.0).sum().sqrt().clamp_min(1e-7);
        x = x.div(norm.unsqueeze());
        for _ in 0..5 {
            let xt = x.clone().swap_dims(D - 2, D - 1);
            let xx = x.clone().matmul(xt);
            let t1 = xx.clone().matmul(x.clone());
            let t2 = xx.matmul(t1.clone());
            x = x
                .clone()
                .mul_scalar(3.25)
                .add(t1.mul_scalar(-2.0))
                .add(t2.mul_scalar(0.25));
        }
        x
    }

    #[test]
    #[ignore]
    fn ns_bench() {
        let dev = Default::default();
        for (r, c) in [(2048usize, 2048usize), (8192, 512)] {
            let g: Tensor<2> = Tensor::random([r, c], Distribution::Normal(0.0, 1.0), &dev);
            for _ in 0..3 {
                let _ = ns_new(g.clone());
            }
            let t0 = Instant::now();
            for _ in 0..5 {
                let _ = ns_old(g.clone());
            }
            let to = t0.elapsed() / 5;
            let t0 = Instant::now();
            for _ in 0..5 {
                let _ = ns_new(g.clone());
            }
            let tn = t0.elapsed() / 5;
            println!(
                "[{r}x{c}] old {:?} new {:?} ({:.1}x)",
                to,
                tn,
                to.as_secs_f64() / tn.as_secs_f64()
            );
        }
    }
}
