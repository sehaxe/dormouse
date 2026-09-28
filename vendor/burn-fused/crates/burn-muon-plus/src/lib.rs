//! # burn-muon-plus — Muon+ optimizer for Burn
//!
//! | arXiv | Component | What |
//! |-------|-----------|------|
//! | [2602.21545](https://arxiv.org/abs/2602.21545) | `Muon+` | Muon + one post-polar normalization step (up to 37% pre-training speedup, zero extra state) |
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
//! - `ColRow`/`RowCol` = sequential composition (best in the paper)
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
//!     .with_norm_dir(NormDir::ColRow)   // paper's best
//!     .init();
//! let model = optimizer.step(lr, model, grads);
//! ```
use burn::config::Config;
use burn::optim::{LearningRate, ModuleOptimizer, Optimizer, RecordState, StateSink, StateSource};
#[cfg(feature = "cuda")]
pub mod fused_kernels;

use burn::tensor::{Device, ElementConversion, Tensor};

/// Muon+ post-polar normalization direction (2602.21545).
#[derive(Clone, Debug, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NormDir {
    /// Normalize each column of the orthogonalized update to unit L2 norm.
    Col,
    /// Normalize each row of the orthogonalized update to unit L2 norm.
    Row,
    /// Column then row — paper's best single combination.
    ColRow,
    /// Row then column.
    RowCol,
}

/// Newton-Schulz quintic coefficients (Keller Jordan, 2602.21545 uses them).
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
    /// The paper consistently prefers bidirectional normalization (`ColRow`).
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

impl MuonPlus {
    /// Orthogonalize `g` via Newton-Schulz (zeroth power, tall-matrix aware).
    ///
    /// Public for testing: returns the nearest-orthogonal approximation of
    /// `g` under the Frobenius norm (2602.21545 §1, Eq. 1).
    pub fn orthogonalize<const D: usize>(&self, g: Tensor<D>) -> Tensor<D> {
        let dims = g.dims();
        let (rows, cols) = (dims[D - 2], dims[D - 1]);
        let (mut x, transposed) = if rows > cols {
            (g.swap_dims(D - 2, D - 1), true)
        } else {
            (g, false)
        };

        // Normalize to unit Frobenius norm.
        let norm = x.clone().mul(x.clone()).sum().sqrt().clamp_min(1e-7);
        x = x.div(norm.unsqueeze());

        let (a, b, c) = self.ns_coeffs;
        let [nr, nc] = x.dims()[D - 2..D].try_into().unwrap();
        // Factored polynomial a·x + b·(xx·x) + c·(xx·(xx·x)) wins on strongly
        // non-square matrices (measured 3.6x on [8192,512] via two [c,c]@[c,r]
        // matmuls instead of [c,c]@[c,c] + [c,c]@[c,r]); the direct form is
        // ~10% faster on squares (same FLOPs, one fewer matmul launch).
        if nc * 4 < nr {
            for _ in 0..self.ns_steps {
                let xt = x.clone().swap_dims(D - 2, D - 1);
                let xx = x.clone().matmul(xt); // X X^T
                let t1 = xx.clone().matmul(x.clone());
                let t2 = xx.matmul(t1.clone());
                #[cfg(feature = "cuda")]
                {
                    if !crate::fused_kernels::ns_combine_cuda(&mut x, &t1, &t2, a, b, c) {
                        x = x
                            .clone()
                            .mul_scalar(a)
                            .add(t1.mul_scalar(b))
                            .add(t2.mul_scalar(c));
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    x = x
                        .clone()
                        .mul_scalar(a)
                        .add(t1.mul_scalar(b))
                        .add(t2.mul_scalar(c));
                }
            }
        } else {
            for _ in 0..self.ns_steps {
                let xt = x.clone().swap_dims(D - 2, D - 1);
                let xx = x.clone().matmul(xt); // X X^T
                let xx2 = xx.clone().matmul(xx.clone()); // (X X^T)²
                let poly = xx.mul_scalar(b).add(xx2.mul_scalar(c));
                x = x.clone().mul_scalar(a).add(poly.matmul(x.clone()));
            }
        }

        if transposed {
            x.swap_dims(D - 2, D - 1)
        } else {
            x
        }
    }

    /// Muon+ post-polar normalization (2602.21545 §2.3).
    ///
    /// `Norm_col(X) = X·D_col⁻¹`, `Norm_row(X) = D_row⁻¹·X`; compositions are
    /// applied sequentially. Returns `x` unchanged when `norm_dir` is `None`.
    pub fn normalize<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        match self.norm_dir {
            None => x,
            Some(NormDir::Col) => Self::norm_col(x),
            Some(NormDir::Row) => Self::norm_row(x),
            Some(NormDir::ColRow) => {
                let mut xm = x;
                #[cfg(feature = "cuda")]
                {
                    if !crate::fused_kernels::norm_colrow_cuda(&mut xm, 1e-7) {
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

    fn norm_col<const D: usize>(x: Tensor<D>) -> Tensor<D> {
        // sum_dim keeps the axis (size 1) → broadcasts directly.
        let col_norms = x
            .clone()
            .mul(x.clone())
            .sum_dim(D - 2)
            .sqrt()
            .clamp_min(1e-7);
        x.div(col_norms)
    }

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

        // A gradient that is exactly zero carries no signal, and the momentum
        // turns it into a FULL-magnitude step in the STALE direction: the
        // Newton-Schulz normalization rescales whatever it is handed to unit
        // Frobenius norm, and the decayed momentum is not zero. So a step the
        // trainer masked as a no-op (the NaN firewall zeroes the gradients on
        // device) still moved every weight, which is precisely the poisoning
        // that killed runs on 2026-09-26/27. Zero gradient now means zero
        // update, decided on device with no host synchronization. The factor is
        // built with `mask_fill` on a float tensor because the project rule is
        // to never build a numeric indicator from a bool tensor on device and
        // count on the host instead (ADR-0018 rule 2) - not because the cast is
        // broken, which was believed until ADR-0016 measured it correct on both
        // backends. The rule outlived the reason.
        let g_active = Tensor::<1>::ones([1], &grad.device()).mask_fill(
            grad.clone()
                .powf_scalar(2.0)
                .sum()
                .greater_elem(0.0)
                .bool_not(),
            0.0,
        );

        let (updated, state) = if D == 2 {
            // --- Muon+ group ---
            let mu = self.momentum.elem::<f32>();
            // M_t = μ·M_{t-1} + (1-μ)·G_t (fused kernel on CUDA)
            let momentum = match &state.mu_momentum {
                Some(m) => {
                    let mut mm = m.clone();
                    #[cfg(feature = "cuda")]
                    {
                        if !crate::fused_kernels::momentum_cuda(&mut mm, &grad, mu) {
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

            // W_t = W_{t-1} - η·max(1, m/n)^0.5·O_t (Bernstein dimensional
            // factor, Jordan et al. muon.py: max(1, m/n)^0.5; the plain
            // sqrt(m/n) shrank the lr on wide matrices (m < n))
            let dims = tensor.dims();
            let (m, n) = (dims[0] as f64, dims[1] as f64);
            let lr_scaled = lr * (m / n).max(1.0).sqrt();

            let wd = (self.weight_decay as f32 * lr_scaled as f32).min(0.999);
            let mut updated = tensor.clone();
            #[cfg(feature = "cuda")]
            {
                if !crate::fused_kernels::finalize_cuda(&mut updated, &update, lr_scaled as f32, wd)
                {
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
mod bench {
    use burn::tensor::{Distribution, Tensor};

    use std::time::Instant;

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
