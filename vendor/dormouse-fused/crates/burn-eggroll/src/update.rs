//! Antithetic fitness sign and the (batched) ES update rule.

use burn::tensor::Tensor;

/// EGGROLL hyperparameters, paper defaults (σ = α = 0.001, rank 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EggrollConfig {
    /// Perturbation scale: `W = M + σE`.
    pub sigma: f64,
    /// Update scale: `M ← M + (α/√r)·Σᵢ Eᵢ·fᵢ` (caller folds the `1/N`
    /// population average into `α`).
    pub alpha: f64,
    /// Perturbation rank (paper: 1 is as good as full rank).
    pub rank: usize,
    /// Population size (number of antithetic pairs per step).
    pub population: usize,
}

impl EggrollConfig {
    /// Paper defaults: σ = 0.001, α = 0.001, r = 1, N = 256.
    pub fn new() -> Self {
        Self {
            sigma: 0.001,
            alpha: 0.001,
            rank: 1,
            population: 256,
        }
    }
}

impl Default for EggrollConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Perturbed weights `M + (σ/√r)·A·Bᵀ`.
///
/// The `1/√r` keeps `Var(Eᵢⱼ) = 1` bounded for any rank; with r = 1 the
/// perturbation entries are exactly `N(0, σ²)`.
pub fn perturb(m: &Tensor<2>, a: &Tensor<2>, b: &Tensor<2>, sigma: f64) -> Tensor<2> {
    let [rows, r] = a.dims();
    let [cols, _] = b.dims();
    assert_eq!(
        m.dims(),
        [rows, cols],
        "perturb: M must match A\u{00b7}B\u{1d40} shape"
    );
    assert!(r > 0, "rank must be >= 1");
    let scale = sigma / (r as f64).sqrt();
    m.clone()
        .add(a.clone().matmul(b.clone().transpose()).mul_scalar(scale))
}

/// Ternary antithetic fitness: `sign(s⁺ − s⁻) ∈ {−1, 0, +1}`.
///
/// Host-side scalar — the ES loop evaluates the model on `(M+σE, M−σE)`,
/// computes losses on the host, and feeds the sign here.
pub fn antithetic_sign(fitness_pos: f32, fitness_neg: f32) -> f32 {
    if fitness_pos > fitness_neg {
        1.0
    } else if fitness_pos < fitness_neg {
        -1.0
    } else {
        0.0
    }
}

/// Single-member update `M + (α/√r)·(A·Bᵀ)·f`.
///
/// The population loop (summing over members) is the caller's job; use
/// [`update_batched`] for the batched form.
pub fn update(
    m: &Tensor<2>,
    a: &Tensor<2>,
    b: &Tensor<2>,
    fitness: f32,
    alpha: f64,
    rank: usize,
) -> Tensor<2> {
    assert!(rank > 0, "rank must be >= 1");
    let scale = alpha / (rank as f64).sqrt();
    m.clone().add(
        a.clone()
            .matmul(b.clone().transpose())
            .mul_scalar(fitness * scale as f32),
    )
}

/// Batched update over `N` members:
///
/// ```text
/// M ← M + (α/√r)·Σᵢ fᵢ·Aᵢ·Bᵢᵀ      A [N,m,r], B [N,n,r], f [N]
/// ```
///
/// Computed without materializing any `[N,m,n]` perturbation: fold the `N`
/// and `r` axes of `f·A` and `B` into one contraction axis and matmul —
/// `[N,r,m]·[N,r,n]ᵀ → [m,n]` (equals `(diag(f)·A)ᵀ·B` summed over N).
pub fn update_batched(
    m: &Tensor<2>,
    a: &Tensor<3>,
    b: &Tensor<3>,
    fitness: &Tensor<1>,
    alpha: f64,
    rank: usize,
) -> Tensor<2> {
    assert!(rank > 0, "rank must be >= 1");
    let [n, rows, r] = a.dims();
    let [n2, cols, r2] = b.dims();
    assert_eq!(n, n2, "A and B must share the batch dim");
    assert_eq!(r, r2, "A and B must share the rank dim");
    assert_eq!(fitness.dims(), [n], "one fitness per member");
    assert_eq!(m.dims(), [rows, cols], "M must match A·Bᵀ shape");

    let fa = fitness.clone().reshape([n, 1, 1]).mul(a.clone()); // [N,m,r]
    let fa = fa.swap_dims(1, 2).reshape([n * r, rows]); // [N·r,m]
    let b = b.clone().swap_dims(1, 2).reshape([n * r, cols]); // [N·r,n]
    let grad = fa.transpose().matmul(b); // [m,n] = Σᵢ Aᵢ·diag(fᵢ)·Bᵢᵀ

    let scale = alpha / (rank as f64).sqrt();
    m.clone().add(grad.mul_scalar(scale as f32))
}
