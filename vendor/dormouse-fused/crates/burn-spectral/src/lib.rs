//! # burn-tsct - Ternary Spectral Compact Training
//!
// burn-ndarray is deprecated upstream; kept as the CPU test backend until the burn-flex migration.
#![cfg_attr(test, allow(deprecated))]
//! A new weight parameterization combining SCT (2604.00733, truncated-SVD
//! weights `W = U·diag(s)·Vᵀ`) with BitNet-style ternary training
//! (2504.12285, absmean STE): the orthonormal factors `U`, `V` are stored
//! as full-precision masters but the FORWARD uses their ternary projections
//! `{-1, 0, +1}·scale`, so a layer of shape `[m, n]` costs
//! `k·(m + n)` ternary values + `k` scales instead of `m·n` dense values.
//!
//! Components:
//! - [`SpectralLinear`] - plain ternary-SVD linear (rank `k`), the direct
//!   SCT-2.0: `y = (x @ U_t) * s @ V_tᵀ` with STE through the masters.
//! - [`SpectralMoE`] - **rank-1 ternary experts**: `k` experts, expert `i` is
//!   `u_i ⊗ v_i` (ternary rank-1), a router picks the top-2 per token and
//!   blends them. Expressiveness of a small MoE at the parameter cost of a
//!   low-rank layer: `k·(m+n)` ternary params, only `2·(m+n)` MACs per token.
//! - [`polar_orthogonalize`] - GPU-only Newton-Schulz polar retraction for
//!   the masters (replaces the CPU QR of SCT, which cost 40-50% of a step).
//!
//! Ternary forward uses the straight-through estimator:
//! `U_eff = U_m + (tern(U_m) - U_m).detach()` - forward sees ternary
//! values, backward flows through the master unchanged.
#[cfg(feature = "cuda")]
pub mod bf16_ops;

#[cfg(feature = "cuda")]
pub mod gpu;
pub mod infer;
#[cfg(feature = "cuda")]
pub mod moe_fused;

use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{activation, Device, Int, Tensor};

/// `TSCT_FUSED=0` kill-switch for the fused SpectralLinear path, read once.
///
/// The env var is a deployment escape hatch (disable the fused kernel without
/// recompiling); it is captured on first use via `OnceLock` so the hot
/// forward path never touches the OS environment per call. Set it before
/// first forward — later changes are ignored by design.
fn tsct_fused_env_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("TSCT_FUSED").map_or(true, |v| v != "0"))
}

/// Absmean ternary projection (BitNet b1.58 STE): `sign(w) * mean(|w|)`
/// with a dead zone at `0.7·mean` (no external reference exists for this threshold).
pub fn ternarize(w: Tensor<2>) -> Tensor<2> {
    let mag = w.clone().abs();
    let mean = mag.clone().mean().unsqueeze_dims(&[0, 0]);
    let keep = mag.greater(mean.clone().mul_scalar(0.7)).int().float();
    w.sign().mul(mean).mul(keep)
}

/// Straight-through ternary wrapper: forward uses the ternary projection,
/// backward passes through `w` unchanged (STE).
pub fn ste_ternary(w: Tensor<2>) -> Tensor<2> {
    let t = ternarize(w.clone());
    w.clone().add(t.sub(w).detach())
}

/// Annealed straight-through ternary: `alpha` blends the master with its
/// ternary projection, `alpha = 1` is the plain STE, `alpha = 0` is the
/// unquantized master.
///
/// ```text
/// w_eff = w + alpha * (tern(w) - w).detach()
/// ```
///
/// Forward sees a soft mixture, backward flows through the master
/// unchanged. Ramping `alpha` 0 -> 1 over training is **soft-to-hard
/// ternary annealing**: the network first learns a full-precision
/// representation, then is gradually coerced onto the ternary manifold,
/// which converges better than hard STE from step one (measured).
pub fn ste_ternary_annealed(w: Tensor<2>, alpha: f32) -> Tensor<2> {
    let t = ternarize(w.clone());
    let diff = t.sub(w.clone()).detach();
    w.add(diff.mul_scalar(alpha))
}

/// Stochastic ternary rounding (S3T): each element is kept at its sign with
/// probability `|w| / scale` and zeroed otherwise, where `scale = mean(|w|)`.
///
/// ```text
/// w_t = sign(w) * scale * Bernoulli(|w| / scale)
/// ```
///
/// Unbiased (`E[w_t] = w` for `|w| <= scale`): the forward is ternary with
/// the same expected value as the master, and the Bernoulli noise acts as a
/// regularizer (quantization-aware training with stochastic rounding,
/// 2412.04787). Unlike deterministic STE there is no dead zone to tune.
pub fn ternarize_stochastic(w: Tensor<2>, device: &Device) -> Tensor<2> {
    let scale: f32 = w.clone().abs().mean().into_scalar();
    let v = w.clone().div_scalar(scale);
    let p = v.abs().clamp(0.0, 1.0);
    let u = Tensor::<2>::random(
        p.shape(),
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        device,
    );
    let keep = p.greater(u).float();
    w.sign().mul_scalar(scale).mul(keep)
}

/// Stochastic STE wrapper: forward = stochastic ternary, backward through
/// the master (identity).
pub fn ste_ternary_stochastic(w: Tensor<2>, device: &Device) -> Tensor<2> {
    let t = ternarize_stochastic(w.clone(), device);
    w.clone().add(t.sub(w).detach())
}

/// Per-column ternary projection: each column of `[m, k]` is scaled by its
/// own `mean(|col|)` instead of one global mean. Critical for low-rank
/// SVD factors: with a global scale, weak columns collapse to zeros under
/// the dead zone; per-column keeps every basis vector alive.
pub fn ternarize_per_column(w: Tensor<2>) -> Tensor<2> {
    let mag = w.clone().abs();
    let mean = mag.clone().mean_dim(0); // [1, k] (0.22 keeps the dim)
    let keep = mag.greater(mean.clone().mul_scalar(0.7)).float();
    w.sign().mul(mean).mul(keep)
}

/// STE wrapper for per-column ternary.
pub fn ste_ternary_per_column(w: Tensor<2>) -> Tensor<2> {
    let t = ternarize_per_column(w.clone());
    w.clone().add(t.sub(w).detach())
}

/// Power-iteration count for the sigma_max estimate (Gram top eigenvalue).
/// Rayleigh-quotient error shrinks as (λ2/λ1)^k, so 5 is ample for a
/// well-separated Gram - but the estimate is NOT accurate to 5% in general
/// (see the `SIGMA_OVERSHOOT` note on `polar_orthogonalize`), and nothing
/// here depends on it being.
const POWER_ITERS: usize = 5;

/// A retracted master: `polar_orthogonalize(before).detach()`, re-flagged as a
/// tracked leaf ONLY if `before` was tracked.
///
/// The re-flag is load-bearing on an autodiff backend: the polar output is a
/// non-leaf (`GradInBackward`), and burn-optim's step re-tracks
/// `Requirement::Grad` only, so a stored non-leaf is silently downgraded to an
/// untracked leaf - the master freezes and the fused op's backward sees a
/// pruned parent. That is why it was added, and why it must not be simply
/// dropped.
///
/// It is also the only autodiff-dependent step in a retraction, and forcing it
/// unconditionally is wrong twice over. On a backend with NO autodiff (a plain
/// `NdArray<f32>` module - a CPU model, an inference-only build) `set_require_grad`
/// is a hard panic, so `retract` could not be called at all. On an autodiff one
/// it silently UN-FREEZES a master the caller had frozen, which is the
/// ADR-0011 class: the state changed and nothing said so. Hence mirror the
/// state, do not force it.
fn polar_retracked(before: &Tensor<2>, iters: usize) -> Tensor<2> {
    let tracked = before.is_require_grad();
    let out = polar_orthogonalize(before.clone(), iters).detach();
    if tracked {
        out.set_require_grad(true)
    } else {
        out
    }
}

/// Newton-Schulz polar iteration: nearest orthonormal approximation under
/// Frobenius norm, all tensor ops on the device - replaces SCT's CPU QR
/// entirely. Newton-Schulz in X·Xᵀ form (small side [c,c]) with the optimal
/// cubic coefficients `(15/8, −5/4, 3/8)`.
///
/// **The coefficients**: the last entry of the PolarExpress schedule tabulated
/// in Muon+ **v3** (arXiv:2602.21545**v3**, 14 May 2026) **Appendix D.3**, which
/// prints `{(aₜ,bₜ,cₜ)}ₜ₌₁⁸` ending at exactly `(1.875, −1.25, 0.375)` — our
/// triple. That appendix is the only place the triple appears; **v1 and v2
/// print no NS coefficients at all** (checked 2026-09-29: neither HTML
/// version contains the string `1.875`), so the citation must name v3, not the
/// paper's §1. Fetched from `https://arxiv.org/html/2602.21545v3`.
///
/// **Why the sigma_max power iteration is here** (it is not removable, and
/// `docs/reviews/2026-09-29-muon-tsct-review.md` §1.7 gets this wrong): the cubic
/// `p(s) = 1.875s − 1.25s³ + 0.375s⁵` has `p′(s) = 1.875(s²−1)² ≥ 0`, `p(1) = 1`
/// and `p(s) − s > 0` on `(0,1)`, so its basin is `[0,1]` and Cauchy–Schwarz
/// (`σ_max ≤ ‖X‖_F`) proves a **Frobenius prescale cannot make it diverge**.
/// That part is right. What it does not show is that the prescale is
/// *redundant*: the prescale's job is to put `σ_max` **at** the fixed point
/// `p(1) = 1`, not merely inside the basin, and only a `σ_max`-normalised
/// scalar does that. With `iters = 3` on an on-the-manifold `k = 64` factor
/// the two prescales give completely different answers:
///
/// | prescale | start `σ` | `σ` after 3 iters | per-entry `‖UᵀU−I‖` |
/// |---|---|---|---|
/// | `σ_max · 1.05` (here) | 0.952 | **1.0000** | 3.0e-7 |
/// | `‖X‖_F` | 0.125 | **0.6992** | **6.4e-2** |
///
/// (fp32, on a `[768,64]` factor that starts exactly on the manifold; both
/// rows are fixed points of themselves over 5 further applications, so the
/// second is not a decay to zero — it is a retraction onto a *different*
/// manifold.) A factor that arrives on the manifold at `σ = 1` is shrunk 30%
/// in one step and then held there forever, its Frobenius norm is 5.5938
/// instead of 8, and `max_ortho` sits at 6.4e-2 per entry — 64× over the
/// one-way 1e-3 latch in `train/src/lib.rs`, which then persists in the
/// checkpoint. The reviewer also gets the iteration count wrong when it
/// appeals to the optimizer: its plain Frobenius prescale
/// (`burn-muon-plus/src/lib.rs:137`) runs `ns_steps = 8`, where
/// `p⁸(1/√k) = 1`; the retraction runs 3, where `p³(1/√k) ≠ 1` for `k ≥ 10`
/// (`0.997` at `k = 8`, `0.963` at 16, `0.699` at 64).
/// `retraction_holds_the_manifold_at_rank_64` and
/// `retraction_puts_sigma_max_at_one` are the tests that fail if anyone
/// swaps this for a Frobenius prescale.
///
/// **The 1.05 factor is an empirical margin, not a guarantee.** An earlier
/// version of this comment justified it as "power iteration converges from
/// below, so the true `σ_max` stays strictly inside the basin" and stated the
/// 5 % figure as the covered error. That justification is FALSE AS MEASURED
/// (2026-09-30): over 200 random `[768,64]` factors the estimate is
/// **worse than 5 % on 32–47 % of draws**, and the prescaled `σ_max` reached
/// **1.087**, not 1.05. What is actually true, and is what the factor rests
/// on:
///
/// 1. **Cubic suppression at the fixed point.** Expanding about `s = 1`,
///    `p(1+e) − 1 = (5/2)·e³ + (15/8)·e⁴ + (3/8)·e⁵` — the linear and
///    quadratic terms cancel exactly (`p′(1) = 0`), so an error `e` is
///    damped cubically, not geometrically. This is why a mis-estimated
///    `σ_max` is benign: the measured prescaled values up to 1.087 still
///    return `σ = 1.000000000` out, because the last iteration lands
///    essentially on the fixed point.
/// 2. **The basin, which is the real safety property.** `p(s) − s = 0` at
///    `s² = 1` and `s² = 7/3`, and the sign is negative in between, so the
///    attracting interval is `0 < x₀ < √(7/3) ≈ 1.5275`. A prescale of
///    `1.05` therefore tolerates an estimate as bad as
///    `1.5275 / 1.05 = 0.687`, i.e. a **31 % underestimate** of `σ_max`
///    before the iteration diverges. That is 6× the error the factor was
///    documented as covering.
/// 3. **`POWER_ITERS` bounds the estimate, not the factor.** 5 Rayleigh
///    steps shrink the error as `(λ₂/λ₁)^k`; it is `p′(1) = 0` that absorbs
///    the residue.
///
/// The offline optimum for a known factor-spread interval is
/// `c* = argmin_c max_{x∈[ℓ,u]} |p³(x/c) − 1|`, and it is deliberately NOT
/// learned: `c` multiplies the whole factor, so a gradient on it turns the
/// retraction's scale normalisation into a **gain-control mechanism** (it
/// would let the optimizer shrink the factor to cut the loss, defeating the
/// retraction's purpose entirely). It is measurable offline from saved
/// factors if the 1.05 margin ever stops being enough.
const SIGMA_OVERSHOOT: f32 = 1.05;
pub fn polar_orthogonalize(x: Tensor<2>, iters: usize) -> Tensor<2> {
    let dims = x.dims();
    let (rows, cols) = (dims[0], dims[1]);
    let (mut m, transposed) = if rows > cols {
        (x.swap_dims(0, 1), true)
    } else {
        (x, false)
    };
    // Put sigma_max at ~1 (see the doc comment for why this is not a plain
    // Frobenius divide) by power iteration on the small-side Gram [c,c].
    // The norm is a [1]-shaped TENSOR broadcast against v, never a host
    // scalar: this path used to take 7 blocking device reads per factor (5
    // in the power loop, 2 for the Rayleigh quotient) and now takes none -
    // 112 per training step at `small`, 16 TSCT factors x 7. Same
    // construction as polar_orthogonalize_batched below and as the optimizer
    // at burn-muon-plus/src/lib.rs:137. Same ops, same order, so the values
    // do not move: `sync_free_retraction_is_bit_identical_to_the_host_read_one`
    // pins that bit-for-bit against the old code on the ndarray backend; on
    // cubecl the only difference is the reduction kernel for a 64-element sum
    // (`sum` vs `sum_dim(0)`), a reordering inside one block, and the result
    // enters as a single divisor.
    let g = m.clone().matmul(m.clone().transpose()); // [c, c]
    let mut v = g.clone().sum_dim(1).squeeze_dim::<1>(1); // [c], row sums = g·1
    for _ in 0..POWER_ITERS {
        let vn = v.clone().mul(v.clone()).sum_dim(0).sqrt().clamp_min(1e-12); // [1]
        v = v.div(vn);
        v = g
            .clone()
            .matmul(v.clone().unsqueeze_dim::<2>(1))
            .squeeze_dim::<1>(1);
    }
    // Rayleigh quotient (normalization-invariant): sigma_max^2 = vᵀgv / vᵀv.
    let gv = g
        .clone()
        .matmul(v.clone().unsqueeze_dim::<2>(1))
        .squeeze_dim::<1>(1);
    let vgv = v.clone().mul(gv).sum_dim(0).unsqueeze_dim::<2>(0); // [1, 1]
    let vv = v.clone().mul(v.clone()).sum_dim(0).unsqueeze_dim::<2>(0); // [1, 1]
                                                                        // `clamp_min` is the tensor form of the old `f32::max` clamps; the two
                                                                        // differ only for NaN, and vᵀv >= 0 makes NaN unreachable here.
    let sigma = vgv.div(vv.clamp_min(1e-14)).sqrt().clamp_min(1e-7); // [1, 1]
                                                                     // SIGMA_OVERSHOOT: an empirical margin, NOT a "5% of the power-iteration
                                                                     // error" guarantee - the estimate exceeds 5% on 32-47% of draws. What
                                                                     // holds is the basin: p(s)-s vanishes at s^2 = 1 and s^2 = 7/3, so the
                                                                     // start may sit anywhere in 0 < x0 < sqrt(7/3) ~ 1.5275 and still
                                                                     // converge, which this factor buys to a 31% underestimate of sigma_max.
    m = m.div(sigma.mul_scalar(SIGMA_OVERSHOOT));
    // Newton-Schulz in X·Xᵀ form (small side [c,c], same as Muon+):
    // x <- a·x + (b·XXᵀ + c·(XXᵀ)²)·x, optimal NS coefficients
    let (a, b, c) = (15.0f32 / 8.0, -5.0f32 / 4.0, 3.0f32 / 8.0);
    for _ in 0..iters {
        let xx = m.clone().matmul(m.clone().transpose()); // [r, r]
        let xx2 = xx.clone().matmul(xx.clone());
        let poly = xx.mul_scalar(b).add(xx2.mul_scalar(c));
        m = m.clone().mul_scalar(a).add(poly.matmul(m.clone()));
    }
    if transposed {
        m.swap_dims(0, 1)
    } else {
        m
    }
}

/// Batched Newton-Schulz polar iteration over same-shaped factors
/// `[B, m, k]`: every slice gets the nearest orthonormal approximation.
/// Identical math to [`polar_orthogonalize`], but the sigma_max estimate and
/// NS loop keep everything as batch tensors — norms are `[B, 1, 1]`
/// broadcasts, so there is not a single host↔device sync (the scalar path
/// syncs 7× per factor). Requires a uniform slice shape; use
/// [`retract_batched`] for mixed inputs.
pub fn polar_orthogonalize_batched(x: Tensor<3>, iters: usize) -> Tensor<3> {
    let [_, rows, cols] = x.dims();
    // Same canonical form as the scalar path: X·Xᵀ with the small side
    // first. Uniform for the whole batch because callers group by shape.
    let (mut m, transposed) = if rows > cols {
        (x.swap_dims(1, 2), true)
    } else {
        (x, false)
    };
    let g = m.clone().matmul(m.clone().transpose()); // [B, c, c]
                                                     // sigma_max via power iteration on the Gram; the norm is a [B,1,1]
                                                     // tensor broadcast across the batch instead of an extracted scalar.
    let mut v = g.clone().sum_dim(2); // [B, c, 1], row sums = g·1
    for _ in 0..POWER_ITERS {
        let nrm = v
            .clone()
            .mul(v.clone())
            .sum_dim(1)
            .sum_dim(2)
            .sqrt()
            .clamp_min(1e-12); // [B, 1, 1]
        v = v.div(nrm);
        v = g.clone().matmul(v);
    }
    // Rayleigh quotient sigma_max² = vᵀgv / vᵀv, per batch element.
    let gv = g.matmul(v.clone());
    let vgv = v.clone().mul(gv).sum_dim(1).sum_dim(2); // [B, 1, 1]
    let vv = v.clone().mul(v).sum_dim(1).sum_dim(2);
    let sigma = vgv.div(vv.clamp_min(1e-14)).sqrt().clamp_min(1e-7);
    m = m.div(sigma.mul_scalar(SIGMA_OVERSHOOT));
    let (a, b, c) = (15.0f32 / 8.0, -5.0f32 / 4.0, 3.0f32 / 8.0);
    for _ in 0..iters {
        let xx = m.clone().matmul(m.clone().transpose());
        let xx2 = xx.clone().matmul(xx.clone());
        let poly = xx.mul_scalar(b).add(xx2.mul_scalar(c));
        m = m.clone().mul_scalar(a).add(poly.matmul(m.clone()));
    }
    if transposed {
        m.swap_dims(1, 2)
    } else {
        m
    }
}

/// Batched retraction of any mix of 2-D masters, results written back in
/// place. Factors are grouped by exact shape, so each group stacks into one
/// padding-free `[B, m, k]` batch — mixed shapes never inflate each other
/// (a `[512, k]` expert master is never padded to lm_head's `[vocab, k]`) —
/// and each group is one sync-free [`polar_orthogonalize_batched`] call.
pub fn retract_batched(factors: &mut [&mut Tensor<2>], iters: usize) {
    let mut groups: std::collections::BTreeMap<(usize, usize), Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, f) in factors.iter().enumerate() {
        let d = f.dims();
        groups.entry((d[0], d[1])).or_default().push(i);
    }
    for ((rows, cols), idx) in groups {
        let stacked = Tensor::stack(idx.iter().map(|&i| factors[i].clone()).collect(), 0);
        let out = polar_orthogonalize_batched(stacked, iters);
        for (slot, &i) in idx.iter().enumerate() {
            *factors[i] = out
                .clone()
                .slice([slot..slot + 1, 0..rows, 0..cols])
                .reshape([rows, cols]);
        }
    }
}

/// Orthonormality error `||UᵀU - I||_F` (diagnostic).
pub fn ortho_error(u: &Tensor<2>) -> f32 {
    let k = u.dims()[1];
    let device = u.device();
    let mut eye = vec![0.0f32; k * k];
    for i in 0..k {
        eye[i * k + i] = 1.0;
    }
    let eye_t = Tensor::<2>::from_data(burn::tensor::TensorData::new(eye, [k, k]), &device);
    let diff = u.clone().transpose().matmul(u.clone()) - eye_t;
    diff.powf_scalar(2.0)
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .sum::<f32>()
        .sqrt()
}

/// The per-ENTRY orthonormality error `||XᵀX - I||_F / rank`.
///
/// The convention the trainer's one-way `max_ortho` latch uses (`param.rs`,
/// and `AGENTS.md` §2.3). It is per entry because the raw F-norm sums
/// `k²` Gram entries and so scales ~`k`: on a `k = 64` factor the NS-3
/// retraction's own convergence floor is ~4e-3 raw (~6e-5 per entry), so a
/// raw-norm threshold would fire the fp32 fallback on every fresh run.
pub fn ortho_error_per_entry(u: &Tensor<2>) -> f32 {
    ortho_error(u) / u.dims()[1].max(1) as f32
}

/// The Gram error of the factor the FORWARD actually multiplies by, at the
/// current ternary annealing `alpha`.
///
/// **Why this is not `ortho_error_per_entry` on the same tensor**: annealing
/// can hold the fp masters exactly on the manifold while the effective
/// factor leaves it, because `ste_ternary_annealed` shrinks every entry
/// toward `sign(w)·mean(|w|)` with a dead zone, and that map is not
/// orthogonal. Measured on a freshly built `[768,64]` fp32 factor: the
/// masters read **8.1e-8** per entry while this reads **7.96e-2** at
/// `alpha = 1` — a gap of ~10⁶. A metric on the masters therefore cannot
/// see forward degradation by construction, and neither can one on the raw
/// ternary projection: at `alpha = 0` the two coincide, and that equality is
/// the cross-check the trainer's eval line relies on.
///
/// Uses [`ste_ternary_annealed`], the SAME function `SpectralLinear::forward`
/// calls, at `alpha` read from the layer: measuring anything else is the
/// "looks perfect while the forward degrades" failure this exists to catch.
pub fn ortho_error_forward(u: &Tensor<2>, alpha: f32) -> f32 {
    ortho_error_per_entry(&ste_ternary_annealed(u.clone(), alpha))
}

/// The learnable `|s|` spectrum: `(max, min, count of near-off ranks)`.
///
/// A dead zone in the ternary projection kills a rank that `s` cannot
/// recover: the rank's column becomes ~0, so `s_i` stops mattering and its
/// gradient stops arriving. Nothing else in the log shows it. `near_off` is
/// the tolerance: the count of `|s_i| < near_off` is the reviewer's "почти
/// выключенные ранги". The caller picks the tolerance
/// (`dormouse_core::TsctDiag::NEAR_OFF` = 1e-3, three orders below the
/// per-entry magnitude of an orthonormal `[768,64]` factor, ~1/√768 ≈
/// 3.6e-2), so this function has no policy in it — a metric that hardcoded
/// its own threshold would be a second place to change the answer.
pub fn spectrum_stats(s: &Tensor<1>, near_off: f32) -> (f32, f32, usize) {
    let bytes = s.clone().abs().into_data().bytes;
    let vals: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let mut smax = 0.0f32;
    let mut smin = f32::INFINITY;
    let mut off = 0usize;
    for v in &vals {
        smax = smax.max(*v);
        smin = smin.min(*v);
        if *v < near_off {
            off += 1;
        }
    }
    (smax, smin, off)
}

/// Quantization format for TSCT U/V factors.
/// Fp8/Fp4 are stored via per-row absmax/absmean scales (burn-bitnet
/// quantize_tensor) with straight-through gradients; Bf16/Fp16 are plain
/// dtype casts. Fp32 = full precision masters (training stage 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantFormat {
    #[default]
    Fp32,
    Bf16,
    Fp16,
    Fp8,
    Fp4,
}

impl QuantFormat {
    pub fn bits(self) -> usize {
        match self {
            QuantFormat::Fp32 => 32,
            QuantFormat::Bf16 | QuantFormat::Fp16 => 16,
            QuantFormat::Fp8 => 8,
            QuantFormat::Fp4 => 4,
        }
    }
}

/// TSCT linear: `y = (x @ U_t) * s @ V_tᵀ`, `U_t/V_t` ternary STE of the
/// orthonormal masters, `s` full-precision scales. With `asym` only `U`
/// is quantized; `V` is used raw (fp32) - A/B which factor carries the
/// quantization.
#[derive(Module, Debug)]
pub struct SpectralLinear {
    /// Orthonormal master `[in, k]` (forward uses its ternary STE).
    pub u: Param<Tensor<2>>,
    /// Singular scales `[k]` (full precision, 1 value per rank).
    pub s: Param<Tensor<1>>,
    /// Orthonormal master `[out, k]`.
    pub v: Param<Tensor<2>>,
    /// Ternary annealing: 0 = full precision, 1 = pure ternary STE.
    #[module(skip)]
    pub alpha: f32,
    /// Stochastic ternary rounding instead of deterministic dead-zone.
    #[module(skip)]
    pub stochastic: bool,
    /// Per-column ternary scaling (keep every rank alive at low k).
    #[module(skip)]
    pub per_column: bool,
    /// Asymmetric ternary: quantize only U, keep V raw fp32 (both stay
    /// STE masters so gradients still land on the full-precision U and V).
    #[module(skip)]
    pub asym: bool,
    /// N:M sparsity on the ternary factors (Sparse-BitNet, 2603.05168):
    /// per block of `nm_m` consecutive values along the factor's last dim
    /// (the rank dim), only the `nm_n` largest-|master| entries survive.
    /// `nm_n == 0 || nm_m == 0` disables sparsity and keeps the plain
    /// ternary path bit-identical. Takes precedence over `stochastic` /
    /// `per_column` (the Sparse-BitNet Dual-STE ternary replaces them) and
    /// forces `alpha` to 1 (pure ternary; no annealing under N:M).
    /// Runtime flags: not serialized in checkpoints; re-apply `set_nm`.
    #[module(skip)]
    pub nm_n: usize,
    #[module(skip)]
    pub nm_m: usize,
    /// 2-bit factor quantization: `{-2s,-s,0,s,2s}` per-row scale (BitNet
    /// 5-level, [`burn_bitnet::weight_quant_2bit`]) instead of ternary.
    /// Own mode: wins over `stochastic`/`per_column`/`asym`; N:M is skipped
    /// when set (combo unsupported). Quality A/B vs ternary, not a memory
    /// play - masters stay fp32, the deployment win comes from the quant.
    #[module(skip)]
    pub two_bit: bool,
    /// Use the fused CUDA training kernels when eligible (plain mode,
    /// cuda backend, shapes inside the kernel tiles). Turn off with
    /// `set_fused(false)`; the `TSCT_FUSED=0` env var overrides too.
    #[module(skip)]
    pub fused: bool,
    /// Compute the matmuls in bf16 via the custom autodiff op (fp32 graph,
    /// exact fp32 backward, bf16 forward on tensor cores). Set by the
    /// trainer under the model's BF16 mode.
    #[module(skip)]
    pub bf16_compute: bool,
    /// Quantized factor format (forward_quant). Fp32 default; stage-1
    /// training runs Fp32/Bf16, final adaptation flips to Fp8/Fp4.
    #[module(skip)]
    pub quant: QuantFormat,
    #[module(skip)]
    pub rank: usize,
    #[module(skip)]
    pub in_features: usize,
    #[module(skip)]
    pub out_features: usize,
}

impl SpectralLinear {
    pub fn new(in_features: usize, out_features: usize, rank: usize, device: &Device) -> Self {
        let k = rank.min(in_features).min(out_features).max(1);
        // orthonormal init: QR of a random normal matrix
        let rand = Tensor::<2>::random(
            [in_features, k],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        let (q, _) = qr_householder(&rand);
        let q = q.slice([0..in_features, 0..k]);
        let rand_v = Tensor::<2>::random(
            [out_features, k],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        let (v, _) = qr_householder(&rand_v);
        let v = v.slice([0..out_features, 0..k]);
        Self {
            u: Param::from_tensor(q),
            s: Param::from_tensor(Tensor::ones([k], device)),
            v: Param::from_tensor(v),
            alpha: 1.0,
            stochastic: false,
            per_column: false,
            asym: false,
            nm_n: 0,
            nm_m: 0,
            two_bit: false,
            fused: true,
            bf16_compute: false,
            quant: QuantFormat::Fp32,
            rank: k,
            in_features,
            out_features,
        }
    }

    /// Set the annealing factor (0 = full precision, 1 = pure ternary).
    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha.clamp(0.0, 1.0);
    }

    /// Enable stochastic ternary rounding (no extra memory, one random op).
    pub fn set_stochastic(&mut self, on: bool) {
        self.stochastic = on;
    }

    /// Enable per-column ternary scaling.
    pub fn set_per_column(&mut self, on: bool) {
        self.per_column = on;
    }

    /// Enable asymmetric ternary: only U is quantized, V stays fp32.
    pub fn set_asym(&mut self, on: bool) {
        self.asym = on;
    }

    /// Enable N:M sparsity on the ternary factors (Sparse-BitNet Dual-STE:
    /// masked entries still receive gradients so pruned weights can regrow).
    /// `n == 0 || m == 0` disables sparsity (default behavior unchanged).
    pub fn set_nm(&mut self, n: usize, m: usize) {
        self.nm_n = n;
        self.nm_m = m;
    }

    /// Enable 2-bit factor quantization (`{-2s,-s,0,s,2s}`, 5-level STE).
    /// Takes precedence over `stochastic`/`per_column`/`asym`; N:M is
    /// skipped when both are set (the 2-bit SVD factors do not carry a mask).
    pub fn set_2bit(&mut self, on: bool) {
        self.two_bit = on;
    }

    /// N:M sparsity is active (both `nm_n` and `nm_m` > 0).
    pub fn nm_on(&self) -> bool {
        self.nm_n > 0 && self.nm_m > 0
    }

    /// Enable/disable the fused CUDA training kernels (default on).
    pub fn set_fused(&mut self, on: bool) {
        self.fused = on;
    }

    /// Ternary-SVD forward (STE through the masters, annealed).
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        // bf16 activations (BF16=1 mode) times fp32 factors produce NaN in
        // the matmul on this stack (measured 2026-08-29); compute in fp32.
        let x = if matches!(x.dtype(), burn::tensor::DType::F32) {
            x
        } else {
            x.cast(burn::tensor::FloatDType::F32)
        };
        let dev = x.device();
        let u_t = if self.two_bit {
            // 2-bit 5-level STE on both factors (BitNet style, per-row scale).
            // Applied to U and V identically; the per-row scale reads the
            // factor's last dim (rank) - a per-column variant for V is a
            // possible follow-up, same function both sides keeps it simple.
            burn_bitnet::weight_quant_2bit(self.u.val())
        } else if self.nm_on() {
            // Sparse-BitNet (2603.05168): ternary + N:M mask in one Dual-STE
            // node - the mask is read off the master and gradients flow
            // straight through it, so masked (pruned) entries keep training.
            burn_bitnet::weight_quant_ternary_nm(self.u.val(), self.nm_n, self.nm_m)
        } else if self.stochastic {
            ste_ternary_stochastic(self.u.val(), &dev)
        } else if self.per_column {
            ste_ternary_per_column(self.u.val())
        } else {
            ste_ternary_annealed(self.u.val(), self.alpha)
        };
        let v_t = if self.two_bit {
            burn_bitnet::weight_quant_2bit(self.v.val())
        } else if self.asym {
            // asymmetric: only U is quantized, V runs full precision
            self.v.val()
        } else if self.nm_on() {
            burn_bitnet::weight_quant_ternary_nm(self.v.val(), self.nm_n, self.nm_m)
        } else if self.stochastic {
            ste_ternary_stochastic(self.v.val(), &dev)
        } else if self.per_column {
            ste_ternary_per_column(self.v.val())
        } else {
            ste_ternary_annealed(self.v.val(), self.alpha)
        };
        // paper order: y = (x@U) * s @ Vᵀ
        x.matmul(u_t)
            .mul(self.s.val().unsqueeze_dims(&[0]))
            .matmul(v_t.transpose())
    }

    /// Quantized-factor forward (stage-2 path): U/V quantized to the
    /// configured format, f32 accumulation. Fp8/Fp4 use per-row scales with
    /// straight-through gradients (round's autodiff gradient is zero);
    /// Bf16/Fp16 are dtype casts; Fp32 = raw masters.
    pub fn forward_quant<B: burn::backend::Backend>(&self, x: Tensor<2>) -> Tensor<2>
    where
        burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
    {
        let u = self.quant_factor::<B>(self.u.val());
        let v = self.quant_factor::<B>(self.v.val());
        // bf16 activations (BF16=1 mode) times fp32 factors produce NaN in
        // the matmul on this stack (measured 2026-08-29); compute in fp32.
        let x = if matches!(x.dtype(), burn::tensor::DType::F32) {
            x
        } else {
            x.cast(burn::tensor::FloatDType::F32)
        };
        // paper order: y = (x@U) * s @ Vᵀ
        x.matmul(u)
            .mul(self.s.val().unsqueeze_dims(&[0]))
            .matmul(v.transpose())
    }

    /// Quantized-factor forward with the matmuls computed in bf16 through
    /// the custom autodiff op (fp32 graph, exact fp32 backward, tensor-core
    /// forward). Weight factors are quantized exactly as in
    /// [`forward_quant`](Self::forward_quant).
    #[cfg(feature = "cuda")]
    pub fn forward_quant_bf16<B: burn::backend::AutodiffBackend>(&self, x: Tensor<2>) -> Tensor<2>
    where
        burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>
            + burn::backend::DispatchKindConversion<B::InnerBackend>
            + burn::backend::DispatchKindConversion<burn_autodiff::Autodiff<B::InnerBackend>>,
    {
        let u = self.quant_factor::<B>(self.u.val());
        let v = self.quant_factor::<B>(self.v.val());
        let y1 = crate::bf16_ops::bf16_matmul::<B::InnerBackend>(x, u);
        let s = self.s.val().unsqueeze_dims(&[0]);
        crate::bf16_ops::bf16_matmul::<B::InnerBackend>(y1.mul(s), v.transpose())
    }

    fn quant_factor<B: burn::backend::Backend>(&self, w: Tensor<2>) -> Tensor<2>
    where
        burn::tensor::DispatchTensor: burn::backend::DispatchKindConversion<B>,
    {
        match self.quant {
            QuantFormat::Fp32 => w,
            QuantFormat::Bf16 => w
                .cast(burn::tensor::FloatDType::BF16)
                .cast(burn::tensor::FloatDType::F32),
            QuantFormat::Fp16 => w
                .cast(burn::tensor::FloatDType::F16)
                .cast(burn::tensor::FloatDType::F32),
            QuantFormat::Fp8 => {
                let q = burn_bitnet::quantize_tensor::<B>(w.clone(), 8);
                w.clone() + (q - w).detach() // STE
            }
            QuantFormat::Fp4 => {
                let q = burn_bitnet::quantize_tensor::<B>(w.clone(), 4);
                let ste = w.clone() + (q - w.clone()).detach();
                if std::env::var("DM_QUANT_DEBUG").is_ok() {
                    let vw: Vec<f32> = w.clone().into_data().try_to_vec().unwrap_or_default();
                    let vs: Vec<f32> = ste.clone().into_data().try_to_vec().unwrap_or_default();
                    let md = vw
                        .iter()
                        .zip(&vs)
                        .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
                    println!("[fq] Fp4 ste-vs-w max={md:.6}");
                }
                ste
            }
        }
    }

    /// Set the quantized factor format.
    pub fn set_quant(&mut self, quant: QuantFormat) {
        self.quant = quant;
    }

    /// Retract the masters to orthonormal (Newton-Schulz polar, on device).
    pub fn retract(&mut self, iters: usize) {
        let (id_u, u_val, map_u) = self.u.clone().consume();
        let u_ret = polar_retracked(&u_val, iters);
        self.u = Param::from_mapped_value(id_u, u_ret, map_u);
        let (id_v, v_val, map_v) = self.v.clone().consume();
        let v_ret = polar_retracked(&v_val, iters);
        self.v = Param::from_mapped_value(id_v, v_ret, map_v);
    }

    /// Trainable parameter count: `k·(in+out)` ternaries + `k` scales.
    pub fn param_count(&self) -> usize {
        self.rank * (self.in_features + self.out_features) + self.rank
    }

    /// Equivalent dense parameter count.
    pub fn dense_params(&self) -> usize {
        self.in_features * self.out_features
    }

    /// Per-token MACs: the `k`-wide SVD step is `x@U_t` (`in.k`) plus the
    /// `V_t.T` step (`k.out`), i.e. `k.(in+out)` — same convention as SCT.
    pub fn flops(&self) -> usize {
        self.rank * (self.in_features + self.out_features)
    }
}

/// Householder QR on the device (for orthonormal init; small matrices).
fn qr_householder(a: &Tensor<2>) -> (Tensor<2>, Tensor<2>) {
    let [m, n] = a.dims();
    let device = a.device();
    let mut q = Tensor::<2>::eye(m, &device);
    let mut r = a.clone();
    for j in 0..n.min(m) {
        // householder vector for column j below the diagonal
        let col = r.clone().slice([j..m, j..j + 1]).reshape([m - j]);
        let norm = col.clone().mul(col.clone()).sum().sqrt().clamp_min(1e-12);
        let first = col.clone().slice([0; 1]).reshape([1]).into_scalar::<f32>();
        let sign = if first >= 0.0 { 1.0 } else { -1.0 };
        // v = col + sign*||col|| e1
        let mut vv: Vec<f32> = col.clone().into_data().try_to_vec().unwrap();
        vv[0] += sign * norm.into_scalar::<f32>();
        let v = Tensor::<1>::from_data(burn::tensor::TensorData::new(vv, [m - j]), &device);
        let vn = v.clone().mul(v.clone()).sum().sqrt().clamp_min(1e-12);
        let v = v.div_scalar(vn.into_scalar::<f32>());
        // H = I - 2vvᵀ applied to r and q
        let v2 = v.clone().unsqueeze_dim::<2>(1); // [n,1]
        let vv_t = v2.matmul(v.clone().unsqueeze_dim::<2>(0)); // [n,n]
        let h = Tensor::<2>::eye(m - j, &device).sub(vv_t.mul_scalar(2.0));
        let r_j = r.clone().slice([j..m, 0..n]);
        let r_new = h.clone().matmul(r_j);
        r = Tensor::cat(vec![r.slice([0..j, 0..n]), r_new], 0);
        let q_j = q.clone().slice([0..m, j..m]);
        let q_new = q_j.matmul(h);
        q = Tensor::cat(vec![q.slice([0..m, 0..j]), q_new], 1);
    }
    (q, r)
}

/// Ternary spectral MoE with a product-key router.
///
/// Experts are rank-`r` ternary patterns: expert `i` owns columns
/// `[i*r .. i*r+r)` of `u`/`v`/`s` (STE through FP32 masters); the router
/// is a dense product-key hierarchical scorer (proj + cluster key + expert
/// key - one matmul per stage, bias-free):
///
/// ```text
/// h          = proj(x)              (in -> p,  dense)
/// cluster    = argmax cluster_key(h)   (p -> C, dense)
/// positions  = top-k expert_key(h)     (p -> E, dense)
/// expert_i   = cluster * E + pos_i
/// out        = Σ_i softmax(e_scores)_i * c_w * s_i * (x·u_i)·v_i
/// ```
///
/// Per-token FLOPs: `k*r*(m+n)` for the experts plus the router
/// `r_p(m+p) + r_c(p+C) + r_e(p+E)`. With m=512, n=4096, C=128, E=32,
/// r=8, k=4: 4*8*4608 = 147K + 24.6K = 172K vs 2M dense (12.2x fewer FLOPs)
/// while the effective rank per token is k*r = 32 — and the loop re-applies
/// the block, so a token sees iter*k*r directions in total.
#[derive(Module, Debug)]
pub struct SpectralMoE {
    /// Experts' left vectors `[in, M*r]` (FP32 masters).
    pub u: Param<Tensor<2>>,
    /// Experts' right vectors `[out, M*r]`.
    pub v: Param<Tensor<2>>,
    /// Per-column scales `[M*r]`.
    pub s: Param<Tensor<1>>,
    /// Input projection `in -> p` (dense, bias-free).
    pub proj: Linear,
    /// Cluster scorer `p -> C` (dense, bias-free), top-1 cluster.
    pub cluster_key: Linear,
    /// Within-cluster scorer `p -> E` (dense, bias-free), top-k positions.
    pub expert_key: Linear,
    #[module(skip)]
    pub proj_dim: usize,
    #[module(skip)]
    pub n_clusters: usize,
    #[module(skip)]
    pub experts_per_cluster: usize,
    #[module(skip)]
    pub top_k: usize,
    #[module(skip)]
    pub rank: usize,
    #[module(skip)]
    pub in_features: usize,
    #[module(skip)]
    pub out_features: usize,
    /// Expert-Choice routing (2202.09368): each expert picks its top tokens
    /// within the cluster, so every expert trains every step and the forward
    /// never materializes `[B, k*r, in/out]`. Off by default (token-choice A/B).
    /// Runtime flag: not serialized in checkpoints; re-apply
    /// `set_expert_choice` after loading a saved model.
    #[module(skip)]
    pub expert_choice: bool,
    /// Expert-Choice capacity factor: expert picks `ceil(n_c * capacity / E)`
    /// tokens per cluster (min 1, capped at `n_c`). Runtime flag: not
    /// serialized in checkpoints; re-apply `set_expert_choice(true)` after
    /// loading a saved model to restore the capacity.
    #[module(skip)]
    pub capacity: f32,
    /// Use the fused CUDA token-choice expert kernel when eligible (plain
    /// token-choice, cuda backend, shapes inside the kernel limits). Turn off
    /// with `set_fused(false)`; the `TSCT_FUSED=0` env var overrides too.
    #[module(skip)]
    pub fused: bool,
}

impl SpectralMoE {
    /// `n_clusters * experts_per_cluster` total experts of `rank` columns,
    /// `top_k` active per token.
    pub fn new(
        in_features: usize,
        out_features: usize,
        n_clusters: usize,
        experts_per_cluster: usize,
        top_k: usize,
        rank: usize,
        device: &Device,
    ) -> Self {
        let m = n_clusters * experts_per_cluster * rank;
        let p = 32usize.min(in_features.max(4));
        // random FP32 masters; the ternary STE projection happens in forward
        let u = Tensor::<2>::random(
            [in_features, m],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        let v = Tensor::<2>::random(
            [out_features, m],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        Self {
            u: Param::from_tensor(u),
            v: Param::from_tensor(v),
            s: Param::from_tensor(Tensor::ones([m], device)),
            // router: dense bias-free, tiny (p=32, C=4-128, E=8-32) — one
            // matmul per stage, no spectral factors to retract
            proj: LinearConfig::new(in_features, p)
                .with_bias(false)
                .init(device),
            cluster_key: LinearConfig::new(p, n_clusters)
                .with_bias(false)
                .init(device),
            expert_key: LinearConfig::new(p, experts_per_cluster)
                .with_bias(false)
                .init(device),
            proj_dim: p,
            n_clusters,
            experts_per_cluster,
            top_k: top_k.max(1),
            rank: rank.max(1),
            in_features,
            out_features,
            expert_choice: false,
            capacity: 2.0,
            fused: true,
        }
    }

    /// Hierarchical router logits: `(c_logits [B, C], e_logits [B, E])`.
    pub fn router_logits(&self, x: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
        let h = self.proj.forward(x);
        let c = self.cluster_key.forward(h.clone());
        let e = self.expert_key.forward(h);
        (c, e)
    }

    /// Top-`top_k` indices of `logits` per row via `argtopk`.
    pub fn topk_indices(&self, logits: Tensor<2>) -> Tensor<2, Int> {
        let k = self.top_k.min(logits.dims()[1]);
        Self::topk_indices_generic(logits, k)
    }

    /// Top-`k` indices along dim 1 via native `argtopk` (device-only, no
    /// host round-trip; `k` must be >= 1 and <= n_pos).
    ///
    /// One `argtopk` (native ArgTopK reduce on cubecl, argsort+select on host
    /// backends) replaces `k` masked-argmax launches. cubecl compiles a
    /// `k`-unrolled insertion-sort kernel, and k = 64 costs ~50s of nvrtc per
    /// process while k <= 16 compiles in ~1s, so rounds are capped at 16 with
    /// the picked positions masked between rounds (same topk-of-masked trick
    /// as burn-mor).
    fn topk_indices_generic(logits: Tensor<2>, k: usize) -> Tensor<2, Int> {
        let [b, n_pos] = logits.dims();
        let device = logits.device();
        let k = k.clamp(1, n_pos);
        if k == n_pos {
            // every position: argtopk requires shape[dim] > k
            return logits.argsort_descending(1);
        }
        let mut work = logits;
        let mut parts: Vec<Tensor<2, Int>> = Vec::with_capacity(k.div_ceil(16));
        let mut remaining = k;
        while remaining > 0 {
            let take = remaining.min(16);
            let idx = work.clone().argtopk(take, 1); // [B, take]
            parts.push(idx.clone());
            remaining -= take;
            if remaining > 0 {
                // Mask picked positions so the next round recovers the next
                // top. -1e30 sits below every real score, so Add-scattering
                // it keeps them out of later rounds.
                let neg = Tensor::<2>::full([b, take], -1e30_f32, &device);
                work = work.scatter(1, idx, neg, burn::tensor::IndexingUpdateOp::Add);
            }
        }
        Tensor::cat(parts, 1) // [B, k]
    }

    /// Enable Expert-Choice routing (off by default; token-choice A/B stays).
    ///
    /// Runtime flag: not serialized in checkpoints. Re-apply it after
    /// loading a saved model, otherwise the layer silently runs token-choice.
    pub fn set_expert_choice(&mut self, on: bool) {
        self.expert_choice = on;
    }

    /// Enable/disable the fused CUDA token-choice expert kernel (default on).
    pub fn set_fused(&mut self, on: bool) {
        self.fused = on;
    }

    /// Expert-Choice forward (2202.09368): within each product-key cluster
    /// the experts pick their top tokens instead of tokens picking top-k
    /// experts, so every expert trains every step (dead-expert fix) and the
    /// forward never materializes `[B, k*r, in]` / `[B, k*r, out]`
    /// (memory fix). Gates are a per-expert softmax over the cluster's
    /// tokens, weighted by the cluster probability like the token-choice
    /// path.
    ///
    /// Per-expert capacity per cluster: `ceil(n_c * capacity / E)` (min 1,
    /// capped at `n_c`). The cluster's token ids are compacted on device
    /// (cumsum prefix + scatter-add; `nonzero` would host-round-trip on
    /// cubecl), with one scalar count sync per cluster.
    ///
    /// Every float gather reads a plain (non-view) tensor: in burn
    /// 0.22.0-pre.1 the ndarray backend panics with an out-of-bounds index
    /// when gathering FROM a broadcast (stride-0) view, so `u_t`/`v_t` are
    /// gathered along dim 1 instead of transposing them first and gates are
    /// picked from `softmax(sc, 0)` rather than from `sc.transpose()`.
    /// Views only feed elementwise ops and scatter values, which are safe.
    ///
    /// ponytail: per-expert top-k is the masked-argmax loop (k_e rounds over
    /// [E, n_c]); at B=16384 with capacity 2 that is ~128 rounds per cluster
    /// per layer. If the 16K-token regime becomes the hot path, replace with
    /// a fused per-row top-k kernel.
    pub fn forward_ec(&self, x: Tensor<2>) -> Tensor<2> {
        let [b, _] = x.dims();
        let e = self.experts_per_cluster;
        let r = self.rank;
        let c = self.n_clusters;
        let cap = self.capacity.max(0.5);

        let u_t = ste_ternary(self.u.val()); // [in, M*r]
        let v_t = ste_ternary(self.v.val()); // [out, M*r]
        let s = self.s.val(); // [M*r]

        let (c_logits, e_logits) = self.router_logits(x.clone());
        let c_idx = c_logits.clone().argmax(1); // [B, 1], native Int
        let c_w = activation::softmax(c_logits, 1).gather(1, c_idx.clone()); // [B, 1]

        let arange_b = Tensor::<1, Int>::arange(0..(b as i64), &x.device()).float(); // [B]
        let e_ids = Tensor::<1, Int>::arange(0..(e as i64), &x.device()); // [E]
        let ar = Tensor::<1, Int>::arange(0..(r as i64), &x.device()); // [r]

        let mut acc = Tensor::<2>::zeros([b, self.out_features], &x.device());
        for ci in 0..c {
            let mask = c_idx.clone().equal_scalar(ci as i64); // [B, 1] Bool
                                                              // Int sum is untracked: the scalar sync bypasses the autodiff
                                                              // graph (a float sum here makes each of the C syncs run the
                                                              // autodiff machinery, ~60ms each on cuda at harness scale)
            let n_c: usize = mask.clone().int().sum().into_scalar::<i64>() as usize;
            if n_c == 0 {
                continue;
            }
            // compact the cluster's token ids on device: the exclusive
            // prefix of the mask is the scatter rank; valid ids add in,
            // masked rows add 0 (int scatter-add is order-exact)
            let mask_i = mask.clone().int(); // [B, 1]
            let rank = mask_i.clone().cumsum(0).sub(mask_i.clone()); // [B, 1]
            let tok_id = arange_b
                .clone()
                .unsqueeze_dim::<2>(1)
                .mul(mask.clone().float()); // [B, 1]
            let tok_c = Tensor::<2>::zeros([b, 1], &x.device())
                .scatter(0, rank, tok_id, burn::tensor::IndexingUpdateOp::Add)
                .slice([0..n_c, 0..1])
                .reshape([n_c])
                .int(); // [n_c] token ids in the cluster

            // expert-choice: each expert keeps its top k_e tokens of the
            // cluster (sc [n_c, E] is gathered from plain e_logits)
            let sc = e_logits
                .clone()
                .gather(0, tok_c.clone().unsqueeze_dim::<2>(1).expand([n_c, e])); // [n_c, E]
            let k_e = ((n_c as f32 * cap) / e as f32).ceil().max(1.0) as usize;
            let k_e = k_e.min(n_c);
            let exp_tok = Self::topk_indices_generic(sc.clone().transpose(), k_e); // [E, k_e]

            // gates: softmax over the cluster's tokens per expert (dim 0 of
            // the plain sc), picked at (token, expert) pairs via a one-hot
            // expert selector (avoids gathering from a transposed view)
            let p = activation::softmax(sc, 0); // [n_c, E]
            let sel = Tensor::<1, Int>::arange(0..((e * k_e) as i64), &x.device())
                .div_scalar(k_e as i64) // expert id per pick
                .unsqueeze_dim::<2>(1); // [E*k_e, 1]
            let onehot = sel.equal(e_ids.clone().unsqueeze_dim::<2>(0)).float(); // [E*k_e, E]
            let gates = p
                .gather(
                    0,
                    exp_tok.clone().reshape([e * k_e, 1]).expand([e * k_e, e]),
                )
                .mul(onehot)
                .sum_dim(1)
                .squeeze_dim::<1>(1); // [E*k_e]

            // columns owned by expert (c*E + e): (c*E + e)*r + (0..r)
            let e_pick = e_ids.clone().unsqueeze_dim::<2>(1).expand([e, k_e]); // [E, k_e]
            let cols = e_pick
                .unsqueeze_dim::<3>(2) // [E, k_e, 1]
                .add_scalar((ci * e) as i64)
                .mul_scalar(r as i64)
                .add(
                    ar.clone()
                        .unsqueeze_dim::<2>(0)
                        .unsqueeze_dim::<3>(1)
                        .expand([e, k_e, r]),
                )
                .reshape([e * k_e * r]); // Int

            // x projections of the picked (expert, token) pairs: gather the
            // x rows directly by global token id (one gather, plain input),
            // then u_t along dim 1 (no transpose of the master)
            let tok_g = tok_c.gather(0, exp_tok.reshape([e * k_e])); // [E*k_e]
            let xgt = x.clone().gather(
                0,
                tok_g
                    .clone()
                    .unsqueeze_dim::<2>(1)
                    .expand([e * k_e, self.in_features]),
            ); // [E*k_e, in]
            let u_g_t = u_t.clone().gather(
                1,
                cols.clone()
                    .unsqueeze_dim::<2>(0)
                    .expand([self.in_features, e * k_e * r]),
            ); // [in, E*k_e*r]
            let z = xgt
                .transpose()
                .unsqueeze_dim::<3>(2) // [in, E*k_e, 1]
                .mul(u_g_t.reshape([self.in_features, e * k_e, r]))
                .sum_dim(0)
                .squeeze_dim::<2>(0) // [E*k_e, r]
                .mul(s.clone().gather(0, cols.clone()).reshape([e * k_e, r]))
                .mul(gates.reshape([e * k_e, 1])); // [E*k_e, r]

            // spread over v, weight by cluster probability, scatter-add into
            // the output rows by global token id (duplicate tokens from
            // different experts accumulate via Add)
            let v_g_t = v_t.clone().gather(
                1,
                cols.unsqueeze_dim::<2>(0)
                    .expand([self.out_features, e * k_e * r]),
            ); // [out, E*k_e*r]
            let y = v_g_t
                .reshape([self.out_features, e * k_e, r])
                .mul(z.unsqueeze_dim::<3>(0))
                .sum_dim(2)
                .squeeze_dim::<2>(2) // [out, E*k_e]
                .transpose() // [E*k_e, out]
                .mul(c_w.clone().gather(0, tok_g.clone().unsqueeze_dim::<2>(1)));
            acc = acc.scatter(
                0,
                tok_g
                    .unsqueeze_dim::<2>(1)
                    .expand([e * k_e, self.out_features]),
                y,
                burn::tensor::IndexingUpdateOp::Add,
            );
        }
        acc
    }

    /// Forward: `x [B, in]` -> `[B, out]`. Product-key top-k experts, or
    /// Expert-Choice when `expert_choice` is set.
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        if self.expert_choice {
            return self.forward_ec(x);
        }
        let [b, _] = x.dims();
        let k = self.top_k;
        let e = self.experts_per_cluster;
        let r = self.rank;

        let u_t = ste_ternary(self.u.val()); // [in, M*r]
        let v_t = ste_ternary(self.v.val()); // [out, M*r]
        let s = self.s.val(); // [M*r]

        // Router: one fused CUDA kernel (projections + argmax/argtopk +
        // softmax + gates in one launch) when the bare-CUDA shapes fit; the
        // tensor-path router stays the fallback (ndarray, EC, exotic shapes,
        // TSCT_FUSED=0).
        let fused = self.fused && tsct_fused_env_enabled();
        let fused_router: Option<(Tensor<2, Int>, Tensor<2, Int>, Tensor<2>)> = if fused {
            #[cfg(feature = "cuda")]
            {
                crate::moe_fused::router_fused(
                    x.clone(),
                    self.proj.weight.val().clone(),
                    self.cluster_key.weight.val().clone(),
                    self.expert_key.weight.val().clone(),
                    self.proj_dim,
                    self.n_clusters,
                    self.experts_per_cluster,
                    k,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                None
            }
        } else {
            None
        };

        let (idx, g) = match fused_router {
            Some((_, idx, g)) => (idx, g),
            None => {
                let (c_logits, e_logits) = self.router_logits(x.clone());
                let c_idx = c_logits.clone().argmax(1); // [B, 1], native Int
                let c_w = activation::softmax(c_logits, 1).gather(1, c_idx.clone()); // [B, 1]
                let pos = self.topk_indices(e_logits.clone()); // [B, k]
                                                               // expert = cluster * E + position
                let idx = c_idx
                    .expand([b, k])
                    .mul_scalar(e as i64)
                    .add(pos.clone())
                    .reshape([b, k]); // [B, k] expert ids

                // gate: softmax over chosen positions, weighted by cluster prob
                let g = activation::softmax(e_logits, 1).gather(1, pos); // [B, k]
                (idx, g.mul(c_w.expand([b, k]))) // [B, k]
            }
        };

        #[cfg(feature = "cuda")]
        if fused {
            if let Some(y) = crate::moe_fused::forward_moe_fused(
                x.clone(),
                u_t.clone(),
                v_t.clone(),
                s.clone(),
                idx.clone(),
                g.clone(),
                self.rank,
            ) {
                return y;
            }
        }

        // expand experts -> owned columns: col = expert * r + j
        let ar = Tensor::<1, Int>::arange(0..(r as i64), &x.device()); // [r]
        let idx_r = idx
            .unsqueeze_dim::<3>(2) // [B, k, 1]
            .mul_scalar(r as i64)
            .add(
                ar.unsqueeze_dim::<2>(0)
                    .unsqueeze_dim::<3>(1)
                    .expand([b, k, r]),
            ) // [B, k, r]
            .reshape([b, k * r]); // column ids
        let g_r = g
            .unsqueeze_dim::<3>(2)
            .expand([b, k, r])
            .reshape([b, k * r]); // gate per column

        // x @ u_i for the chosen columns
        let u_g = u_t
            .transpose() // [M*r, in]
            .gather(
                0,
                idx_r
                    .clone()
                    .reshape([b * k * r, 1])
                    .expand([b * k * r, self.in_features]),
            )
            .reshape([b, k * r, self.in_features]); // [B, k*r, in]
        let proj = (x.clone().unsqueeze_dim::<3>(1) * u_g)
            .sum_dim(2)
            .squeeze_dim::<2>(2); // [B, k*r]
        let s_g = s
            .gather(0, idx_r.clone().reshape([b * k * r]))
            .reshape([b, k * r]);
        let proj = proj.mul(s_g).mul(g_r); // [B, k*r]

        // spread over v_i
        let v_g = v_t
            .transpose() // [M*r, out]
            .gather(
                0,
                idx_r
                    .reshape([b * k * r, 1])
                    .expand([b * k * r, self.out_features]),
            )
            .reshape([b, k * r, self.out_features]); // [B, k*r, out]
        (proj.unsqueeze_dim::<3>(2) * v_g)
            .sum_dim(1)
            .squeeze_dim::<2>(1) // [B, out]
    }

    /// Total expert count.
    pub fn num_experts(&self) -> usize {
        self.n_clusters * self.experts_per_cluster
    }

    /// Trainable master count: experts + scales + three dense routers.
    pub fn param_count(&self) -> usize {
        let m = self.num_experts() * self.rank;
        let routers = self.proj_dim * self.in_features
            + self.proj_dim * self.n_clusters
            + self.proj_dim * self.experts_per_cluster;
        m * (self.in_features + self.out_features) + m + routers
    }

    /// Effective rank per token: active experts x columns per expert.
    pub fn effective_rank(&self) -> usize {
        self.top_k * self.rank
    }

    /// Router MACs per token: three dense projections (MACs, not params).
    fn router_flops(&self) -> usize {
        let p = self.proj_dim;
        self.in_features * p + p * self.n_clusters + p * self.experts_per_cluster
    }

    /// Per-token FLOPs (experts + router, all MACs). Token-choice: each
    /// token activates `top_k` experts of `rank` columns. Expert-choice
    /// FLOPs depend on the number of rows (k_e via n_c): use
    /// [`flops_n`](Self::flops_n) instead - this method panics in EC mode
    /// rather than undercount. The per-expert top-k selection is not MACs;
    /// it is reported separately via [`topk_cost`](Self::topk_cost).
    pub fn flops(&self) -> usize {
        assert!(
            !self.expert_choice,
            "expert-choice FLOPs are batch-dependent; call flops_n(n_tokens)"
        );
        self.effective_rank() * (self.in_features + self.out_features) + self.router_flops()
    }

    /// Per-token FLOPs for a forward over `n_tokens` rows (experts +
    /// router, all MACs). Token-choice: identical to [`flops`](Self::flops).
    /// Expert-choice: each of the `C` clusters holds `n_c = n_tokens/C`
    /// rows and every expert picks its top `k_e` (clamped to `[1, n_c]`
    /// like forward_ec), so `E*k_e` picks cost `rank*(in+out)` each:
    /// `picks_per_token = ceil(E*k_e/n_c)`. The per-expert top-k selection
    /// is reported separately via [`topk_cost`](Self::topk_cost).
    pub fn flops_n(&self, n_tokens: usize) -> usize {
        let router = self.router_flops();
        if !self.expert_choice {
            return self.effective_rank() * (self.in_features + self.out_features) + router;
        }
        let n_c = n_tokens.div_ceil(self.n_clusters);
        let cap = self.capacity.max(0.5);
        let k_e = ((n_c as f32 * cap) / self.experts_per_cluster as f32)
            .ceil()
            .max(1.0) as usize;
        let k_e = k_e.min(n_c);
        let picks = (self.experts_per_cluster * k_e).div_ceil(n_c.max(1));
        picks * self.rank * (self.in_features + self.out_features) + router
    }

    /// Expert-choice per-forward top-k selection cost in comparisons over
    /// `n_tokens` rows (the rows the MoE actually sees - batch*seq in the
    /// harness). Each of the `C` clusters runs the masked-argmax loop
    /// `k_e` times over an `[E, n_c]` score matrix, one full pass of
    /// `E * n_c` comparisons per round, with `n_c = n_tokens.div_ceil(C)`
    /// and `k_e = ceil(n_c * capacity / E)` clamped to `[1, n_c]`, so
    /// `cost = C * k_e * E * n_c`. Returns 0 for token-choice.
    pub fn topk_cost(&self, n_tokens: usize) -> usize {
        if !self.expert_choice {
            return 0;
        }
        let c = self.n_clusters;
        let e = self.experts_per_cluster;
        let n_c = n_tokens.div_ceil(c);
        let cap = self.capacity.max(0.5);
        let k_e = ((n_c as f32 * cap) / e as f32).ceil().max(1.0) as usize;
        let k_e = k_e.min(n_c);
        c * k_e * e * n_c
    }

    /// Polar retraction of the expert masters (the dense routers have no
    /// orthonormal factors to retract).
    pub fn retract(&mut self, iters: usize) {
        let (id_u, u_val, map_u) = self.u.clone().consume();
        let u_ret = polar_retracked(&u_val, iters);
        self.u = Param::from_mapped_value(id_u, u_ret, map_u);
        let (id_v, v_val, map_v) = self.v.clone().consume();
        let v_ret = polar_retracked(&v_val, iters);
        self.v = Param::from_mapped_value(id_v, v_ret, map_v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    fn dev() -> Device {
        Device::ndarray()
    }

    #[test]
    fn tsct_linear_shapes() {
        let m = SpectralLinear::new(64, 128, 8, &dev());
        let x = Tensor::<2>::random([4, 64], Distribution::Default, &dev());
        assert_eq!(m.forward(x).dims(), [4, 128]);
    }

    #[test]
    fn tsct_linear_flops() {
        assert_eq!(
            SpectralLinear::new(64, 128, 8, &dev()).flops(),
            8 * (64 + 128)
        );
        assert_eq!(
            SpectralLinear::new(512, 4096, 4, &dev()).flops(),
            4 * (512 + 4096)
        );
    }

    #[test]
    fn tsct_linear_ortho_init() {
        let m = SpectralLinear::new(64, 128, 8, &dev());
        assert!(ortho_error(&m.u.val()) < 1e-3, "u not orthonormal");
        assert!(ortho_error(&m.v.val()) < 1e-3, "v not orthonormal");
    }

    #[test]
    fn polar_retracts() {
        // The fixture must be an autodiff device: retract re-tracks the
        // masters with `set_require_grad`, which on burn 0.22 is an assert on
        // a non-autodiff device — this test used to panic before it asserted
        // (TEST-AUDIT finding 2).
        let adev = Device::ndarray().autodiff();
        let mut m = SpectralLinear::new(64, 128, 8, &adev);
        // corrupt the masters
        let noise = Tensor::<2>::random([64, 8], Distribution::Normal(0.0, 1.0), &adev);
        m.u = Param::from_tensor(m.u.val().add(noise.mul_scalar(0.5)));
        let before = ortho_error(&m.u.val());
        m.retract(5);
        let after = ortho_error(&m.u.val());
        assert!(
            after < before,
            "polar must improve ortho: {before} -> {after}"
        );
        assert!(after < 1e-3, "polar must restore orthonormality: {after}");
    }

    // ---- the three pre-100k diagnostics (reviewer checklist item 2) ----
    //
    // Each gate is a RED→GREEN demonstration: a metric is pinned by
    // perturbing a fixture by a COMPUTED amount and asserting the metric
    // moves by the amount the arithmetic says, not by a literal. A literal
    // pins the implementation; a computed delta pins the MEASUREMENT, which
    // is the thing a reader of the eval line depends on.

    /// The orthonormal fixture, at the trainer's rank. See
    /// [`ortho_factor_768x64`].
    fn manifold_64(dev: &Device) -> Tensor<2> {
        ortho_factor_768x64(dev)
    }

    /// A `[768, 64]` factor whose Gram error is EXACTLY known in closed
    /// form: the first `k` rows carry `diag(1+eps)`, every other entry is
    /// zero. So `UᵀU = (1+eps)²·I` with no cross terms at all, and
    ///
    /// ```text
    /// ||UᵀU − I||_F      = sqrt(k) · ((1+eps)² − 1)
    /// per entry          = ((1+eps)² − 1) / sqrt(k)
    /// ```
    ///
    /// A perturbation of an EXISTING orthonormal factor cannot be used here:
    /// it moves the off-diagonal Gram entries by `eps·U_ij`, so the expected
    /// value would depend on the fixture's own cross terms rather than on the
    /// metric. This fixture has none, which is what makes the expected number
    /// a property of the MEASUREMENT.
    fn known_gram_error_768x64(dev: &Device, eps: f32) -> (Tensor<2>, f32) {
        const N: usize = 768;
        const K: usize = 64;
        let mut data = vec![0.0f32; N * K];
        for i in 0..K {
            data[i * K + i] = 1.0 + eps;
        }
        let per_entry = ((1.0 + eps) * (1.0 + eps) - 1.0) / (K as f32).sqrt();
        (
            Tensor::<2>::from_data(burn::tensor::TensorData::new(data, [N, K]), dev),
            per_entry,
        )
    }

    /// (1a) The masters' per-entry Gram error reports a KNOWN perturbation.
    ///
    /// Red→green by construction: `eps = 0` is exactly orthonormal (the
    /// metric must read 0), and `eps = 0.05` has a per-entry error of
    /// `0.1025 / 8 = 1.28125e-2` computed in closed form above. A metric
    /// that ignored the perturbation, or divided by the wrong quantity,
    /// cannot produce that number from a fixture whose Gram diagonal is
    /// exactly `(1+eps)²`.
    #[test]
    fn ortho_per_entry_reports_a_known_perturbation() {
        let dev = dev();
        let (exact, expect0) = known_gram_error_768x64(&dev, 0.0);
        assert_eq!(expect0, 0.0);
        let got0 = ortho_error_per_entry(&exact);
        assert!(
            got0 < 1e-7,
            "an exactly orthonormal factor must read ~0, got {got0:.3e}"
        );

        const EPS: f32 = 0.05;
        let (pert, expect) = known_gram_error_768x64(&dev, EPS);
        let got = ortho_error_per_entry(&pert);
        assert!(
            (got - expect).abs() < 1e-4 * expect,
            "metric must report the closed-form per-entry error {expect:.6e}, got {got:.6e}"
        );
        // Red by construction: the unperturbed read cannot produce it.
        assert!(got > 1000.0 * got0);
        // And per entry is the RAW F-norm over the rank, which is the whole
        // convention the trainer's 1e-3 latch is quoted against.
        let raw = ortho_error(&pert);
        assert!((got - raw / 64.0).abs() < 1e-9);
        assert!(
            (raw - (64.0f32).sqrt() * 0.1025).abs() < 1e-3,
            "the raw F-norm must match sqrt(k)·((1+eps)²−1) = {expect_raw}, got {raw}",
            expect_raw = (64.0f32).sqrt() * 0.1025
        );
    }

    /// (1b) Zero-error factors read ~0: a retraction applied to an already
    /// orthonormal factor must not report drift, or every eval line carries a
    /// number that means nothing.
    #[test]
    fn ortho_per_entry_is_zero_on_the_manifold() {
        let dev = dev();
        let u = manifold_64(&dev);
        let e = ortho_error_per_entry(&polar_orthogonalize(u.clone(), 3));
        assert!(e < 1e-5, "retraction must hold the manifold, got {e:.3e}");
    }

    /// (2) The FORWARD factor's Gram error, at the annealing `alpha` the
    /// forward uses.
    ///
    /// Three properties, each of which is the reason this metric exists:
    /// - at `alpha = 0` it EQUALS the masters' metric (the cross-check: the
    ///   two code paths agree when the ternary term is switched off);
    /// - at `alpha = 1` it is strictly LARGER, because the ternary projection
    ///   with its `0.7·mean` dead zone is not an orthogonal map — this is
    ///   the "masters perfect, forward degraded" case the metric must see;
    /// - it moves MONOTONICALLY with `alpha`, so a run's rising number is
    ///   attributable to annealing rather than to noise.
    #[test]
    fn ortho_forward_departures_from_the_masters_as_alpha_rises() {
        let dev = dev();
        let u = manifold_64(&dev);
        let masters = ortho_error_per_entry(&u);
        let at0 = ortho_error_forward(&u, 0.0);
        assert!(
            (at0 - masters).abs() < 1e-12,
            "alpha = 0 must reproduce the masters metric exactly: {at0:.6e} vs {masters:.6e}"
        );
        let at1 = ortho_error_forward(&u, 1.0);
        assert!(
            at1 > 100.0 * masters,
            "alpha = 1 must show the ternary projection leaving the manifold: \
             forward {at1:.3e} vs masters {masters:.3e}"
        );
        let half = ortho_error_forward(&u, 0.5);
        assert!(
            masters <= half && half < at1,
            "must be monotone in alpha: {masters:.3e} <= {half:.3e} < {at1:.3e}"
        );
        // Red by construction: the masters' own metric CANNOT distinguish
        // alpha = 0 from alpha = 1 - it is the same number twice. That is
        // the defect this series exists to expose.
        assert_eq!(masters, ortho_error_per_entry(&u));
    }

    /// (3) The `|s|` spectrum: max, min, and the near-off count.
    ///
    /// Fixture with three known magnitudes including one below the tolerance,
    /// so a wrong count or an unsigned/NaN handling bug is a specific failure
    /// rather than a tolerance discussion.
    #[test]
    fn spectrum_stats_reports_max_min_and_dead_ranks() {
        let dev = dev();
        let s = Tensor::<1>::from_data(
            burn::tensor::TensorData::new(vec![2.0f32, 1e-4, -0.5, 0.0, 1.25], [5]),
            &dev,
        );
        // |s| : 2.0, 1e-4, 0.5, 0.0, 1.25 -> max 2.0, min 0.0, one below 1e-3.
        let (smax, smin, off) = spectrum_stats(&s, 1e-3);
        assert_eq!(
            (smax, smin, off),
            (2.0, 0.0, 2),
            "got ({smax}, {smin}, {off})"
        );
        // The tolerance is a knob and it moves ONLY the count - max and min
        // are properties of the spectrum, not of the threshold. `tol = 1e-4`
        // excludes the 1e-4 value (the comparison is strict `<`), and
        // `tol = 1.0` admits the two smallest as well.
        assert_eq!(spectrum_stats(&s, 1e-4).2, 1);
        assert_eq!(spectrum_stats(&s, 1.0).2, 3);
        // Red by construction: all-alive factors report zero dead ranks, so
        // a metric that returned a constant could not pass the row above.
        let alive =
            Tensor::<1>::from_data(burn::tensor::TensorData::new(vec![1.0f32, 2.0], [2]), &dev);
        assert_eq!(spectrum_stats(&alive, 1e-3).2, 0);
    }

    /// The `max_ortho` metric, per entry: `‖UᵀU−I‖_F / rank`. Same
    /// definition as the one the trainer's one-way fp32 latch uses
    /// (`train/src/lib.rs`), and the threshold below is that latch's. An
    /// alias for the library function the trainer now calls directly, so the
    /// two cannot drift.
    use super::ortho_error_per_entry as ortho_err_per_entry;

    /// An exactly orthonormal `[768, 64]` factor - the trainer's factor
    /// shape at `rank = 64`. The first 64 modes of the n=768 DCT-II basis
    /// (rows of an orthogonal transform, so exact in real arithmetic; 1.3e-7
    /// per entry in fp32), rather than `qr_householder` at this size: the
    /// Householder path is 64 passes of `[768, 768]` and a test that spends
    /// seconds proving nothing is a test that gets skipped. The j = 0 mode
    /// carries `1/√N` and the rest `√(2/N)`; that asymmetry is the whole
    /// normalisation, and the test asserts the fixture rather than trusting it.
    fn ortho_factor_768x64(dev: &Device) -> Tensor<2> {
        const N: usize = 768;
        const K: usize = 64;
        let mut data = vec![0.0f32; N * K];
        for i in 0..N {
            for j in 0..K {
                let scale = if j == 0 {
                    1.0 / (N as f32).sqrt()
                } else {
                    (2.0 / N as f32).sqrt()
                };
                data[i * K + j] = scale
                    * ((std::f32::consts::PI * (2 * i + 1) as f32 * j as f32) / (2 * N) as f32)
                        .cos();
            }
        }
        Tensor::<2>::from_data(burn::tensor::TensorData::new(data, [N, K]), dev)
    }

    /// A retraction is a map whose fixed point is the manifold: a factor
    /// that is ALREADY orthonormal must come back orthonormal, at the rank
    /// the trainer actually runs (`rank = 64`, `retract_iters = 3`), not at
    /// the `k = 8` of the test above.
    ///
    /// This is the gate on the `sigma_max` prescale, and it is deliberately
    /// at `iters = 3`: the cubic's `p³` sends the *normalized* singular
    /// values to 1 only from a start near 1. An `σ_max`-normalized factor
    /// starts at `1/1.05` and lands on the manifold; a `‖X‖_F`-normalized
    /// one starts at `1/√64` and `p³(0.125) = 0.6992`, which is 6.4e-2 per
    /// entry - 64× over the latch. It is a fixed point of itself, so it is
    /// not a decay: every input is mapped to a factor 30% short of the
    /// manifold and held there. See the doc comment on
    /// [`polar_orthogonalize`]; the retraction is a fixed 3 iterations
    /// (`retract_iters: 3`), which is why the optimizer's 8-iteration
    /// Frobenius prescale is not a substitute.
    #[test]
    fn retraction_holds_the_manifold_at_rank_64() {
        let dev = dev();
        let u = ortho_factor_768x64(&dev);
        assert!(
            ortho_err_per_entry(&u) < 1e-4,
            "fixture is not on the manifold to begin with: {}",
            ortho_err_per_entry(&u)
        );
        // The manifold is a fixed point: retract it and it must still be there.
        let once = polar_orthogonalize(u.clone(), 3);
        let e = ortho_err_per_entry(&once);
        assert!(
            e < 1e-3,
            "retraction must HOLD the manifold at rank 64 / 3 iters, got {e:.3e} \
             per entry (1e-3 is the trainer's max_ortho latch)"
        );
        // ...and it must be a fixed point there, or the retraction is a
        // drift. A bound, not an equality: the retraction is not bit-
        // idempotent in fp32 (one more pass moves the last bits), but a
        // second pass must not move the factor. A Frobenius prescale moves
        // it by 30%.
        let twice = polar_orthogonalize(once.clone(), 3);
        let drift = twice
            .clone()
            .sub(once.clone())
            .abs()
            .max()
            .into_scalar::<f32>();
        assert!(
            drift < 1e-4,
            "a second retraction must be a no-op on the manifold, max drift {drift:.3e}"
        );
        // And it must still pull a scaled factor back (the job), at this rank.
        let scaled = polar_orthogonalize(u.mul_scalar(3.0), 3);
        let es = ortho_err_per_entry(&scaled);
        assert!(
            es < 1e-3,
            "retraction must pull 3x off the manifold: {es:.3e}"
        );
        // Scale is preserved, not merely direction: a retraction does not
        // shrink the factor. `‖X‖_F ~ sqrt(rank)` is what "on the manifold"
        // means for a tall factor.
        let n = scaled.powf_scalar(2.0).sum().into_scalar::<f32>().sqrt();
        assert!(
            (n - 8.0).abs() < 1e-2,
            "retracted [768,64] must have ||X||_F ~ sqrt(rank) = 8, got {n}"
        );
    }

    /// The same invariant on a REAL tracked tensor: `polar_retracked` is
    /// the one autodiff-dependent step in a retraction, and this is the
    /// shape and the device where `is_require_grad` is live and the
    /// `.detach()` / re-flag actually decides whether the master freezes.
    /// A test on a bare `Device::ndarray()` cannot fail here at all, which
    /// is why the tracking tests use `.autodiff()`. The fixture goes through
    /// `Param::from_tensor` because that is what makes a master tracked — a
    /// tensor built from raw data is an untracked leaf.
    #[test]
    fn retraction_holds_the_manifold_on_a_tracked_master() {
        let adev = Device::ndarray().autodiff();
        let u = Param::from_tensor(ortho_factor_768x64(&adev)).val();
        assert!(u.is_require_grad(), "fixture is not a tracked master");
        let r = polar_retracked(&u, 3);
        assert!(r.is_require_grad(), "retract dropped the tracked master");
        let e = ortho_err_per_entry(&r);
        assert!(
            e < 1e-3,
            "tracked master must land on the manifold, got {e:.3e} per entry"
        );
    }

    /// The other half of the prescale claim, in values: the retraction
    /// normalizes by `sigma_max`, not by the Frobenius norm. The doc comment
    /// on [`polar_orthogonalize`] says 0.6992 vs 1.0; this keeps the number
    /// honest in the tree so a comment and a behaviour cannot drift apart.
    /// `sqrt(trace(UᵀU)/k)` is the RMS singular value - 1.0 for an
    /// on-the-manifold retraction, 0.6992 for a Frobenius-normalised one.
    #[test]
    fn retraction_puts_sigma_max_at_one() {
        let dev = dev();
        let s = polar_orthogonalize(ortho_factor_768x64(&dev), 3);
        let k = s.dims()[1] as f32;
        let rms = s.clone().transpose().matmul(s).sum().into_scalar::<f32>() / k;
        let sigma_max = rms.sqrt();
        assert!(
            (sigma_max - 1.0).abs() < 1e-2,
            "retracted rank-64 factor has sigma_max {sigma_max:.4}, expected \
             ~1.0; a Frobenius prescale gives 0.6992 and is NOT a retraction"
        );
    }

    /// The source of `fn <name>` in this file, from its signature to the
    /// first brace at column 0.
    fn fn_source<'a>(src: &'a str, name: &str) -> &'a str {
        let needle = format!("fn {name}(");
        let at = src
            .find(&needle)
            .unwrap_or_else(|| panic!("no `{needle}` in lib.rs"));
        let start = src[..at].rfind('\n').map_or(at, |p| p + 1);
        let rest = &src[start..];
        let end = rest
            .find("\n}\n")
            .unwrap_or_else(|| panic!("`{needle}` has no closing brace at column 0"));
        &rest[..end + 3]
    }

    /// Gate: the scalar retraction takes ZERO host reads. It used to take
    /// seven per factor - five in the power loop, two for the Rayleigh
    /// quotient - i.e. 112 blocking device reads per training step at
    /// `small` (16 TSCT factors x 7), which is most of what
    /// `retr = 52.8 ms` was made of (ADR-0011 rule 2: a host round-trip in
    /// the hot path is a defect whether or not it is visible in the loss).
    ///
    /// This is a source gate and it has to be: on `Device::ndarray()` an
    /// `into_scalar` is a free pointer chase, so no runtime test in this
    /// crate can see one, and the crate has no CUDA dev-dependency to count
    /// `client.read` calls from. What the gate does is exact - it fails on
    /// the line, whoever wrote it, for any future edit to these two
    /// functions. The norm is a `[1]`-shaped tensor broadcast against `v`
    /// for the same reason `polar_orthogonalize_batched` needs no sync.
    #[test]
    fn scalar_retraction_makes_no_host_read() {
        let src = include_str!("lib.rs");
        for f in ["polar_orthogonalize", "polar_retracked"] {
            let body = fn_source(src, f);
            for read in ["into_scalar", "into_data", "try_to_vec"] {
                assert!(
                    !body.contains(read),
                    "`{f}` does a host read (`{read}`): the sigma_max norm must \
                     stay a tensor and be broadcast, or the retraction is back \
                     to 7 blocking syncs per factor"
                );
            }
        }
    }

    /// The retraction exactly as it was before the `sigma_max` norm became a
    /// tensor: 7 blocking host reads per factor (5 in the power loop, 2 for
    /// the Rayleigh quotient). Test-only, kept verbatim so the sync-free
    /// rewrite is pinned to the real previous code rather than to a
    /// description of it. The only differences from
    /// [`polar_orthogonalize`] are `div_scalar(f32)` vs `div(Tensor<1>)`,
    /// `sum()` vs `sum_dim(0)` and the f32 clamps - same ops, same order.
    fn polar_orthogonalize_host_read(x: Tensor<2>, iters: usize) -> Tensor<2> {
        let dims = x.dims();
        let (rows, cols) = (dims[0], dims[1]);
        let (mut m, transposed) = if rows > cols {
            (x.swap_dims(0, 1), true)
        } else {
            (x, false)
        };
        let g = m.clone().matmul(m.clone().transpose()); // [c, c]
        let mut v = g.clone().sum_dim(1).squeeze_dim::<1>(1); // [c]
        for _ in 0..POWER_ITERS {
            let vn = v.clone().mul(v.clone()).sum().sqrt().clamp_min(1e-12);
            v = v.div_scalar(vn.into_scalar::<f32>());
            v = g
                .clone()
                .matmul(v.clone().unsqueeze_dim::<2>(1))
                .squeeze_dim::<1>(1);
        }
        let gv = g
            .clone()
            .matmul(v.clone().unsqueeze_dim::<2>(1))
            .squeeze_dim::<1>(1);
        let vgv = v.clone().mul(gv).sum().into_scalar::<f32>();
        let vv = v.clone().mul(v.clone()).sum().into_scalar::<f32>();
        let sigma = (vgv / vv.max(1e-14)).sqrt().max(1e-7);
        m = m.div_scalar(sigma * SIGMA_OVERSHOOT);
        let (a, b, c) = (15.0f32 / 8.0, -5.0f32 / 4.0, 3.0f32 / 8.0);
        for _ in 0..iters {
            let xx = m.clone().matmul(m.clone().transpose()); // [r, r]
            let xx2 = xx.clone().matmul(xx.clone());
            let poly = xx.mul_scalar(b).add(xx2.mul_scalar(c));
            m = m.clone().mul_scalar(a).add(poly.matmul(m.clone()));
        }
        if transposed {
            m.swap_dims(0, 1)
        } else {
            m
        }
    }

    /// A dense, deterministic `[rows, cols]` matrix: no RNG, so a bit-for-bit
    /// comparison is meaningful, and non-orthonormal, so the retraction
    /// actually has work to do.
    fn det_factor(rows: usize, cols: usize, dev: &Device) -> Tensor<2> {
        let mut d = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                d[i * cols + j] =
                    ((i * 7 + j * 13) as f32).sin() * 0.5 + ((i * 3 + j * 5) as f32).cos();
            }
        }
        Tensor::<2>::from_data(burn::tensor::TensorData::new(d, [rows, cols]), dev)
    }

    /// The sync-free rewrite changes no number. This is the evidence, not an
    /// assertion about it: the old path is kept above as
    /// `polar_orthogonalize_host_read` and the two are compared elementwise
    /// — on a bare device AND on a real autodiff device, where the tracked
    /// leaf is live, through `polar_retracked` (the wrapper whose detach and
    /// re-flag are load-bearing).
    ///
    /// Shapes: the two the trainer retracts at `small` ([768,64] and
    /// [2048,64]), a small one, and a wide one so the canonical transpose
    /// runs the other way.
    #[test]
    fn sync_free_retraction_is_bit_identical_to_the_host_read_one() {
        let shapes = [(768usize, 64usize), (2048, 64), (64, 8), (8, 64)];
        for (rows, cols) in shapes {
            let x = det_factor(rows, cols, &dev());
            let a = polar_orthogonalize(x.clone(), 3);
            let b = polar_orthogonalize_host_read(x, 3);
            assert_eq!(
                a.into_data().try_to_vec::<f32>().unwrap(),
                b.into_data().try_to_vec::<f32>().unwrap(),
                "sync-free retraction changed the numbers at [{rows},{cols}]"
            );
        }
        // Same on a real tracked master, through the wrapper: value AND
        // tracking. A host read is not what makes this work, and the fix
        // must not have traded one for the other.
        let adev = Device::ndarray().autodiff();
        for (rows, cols) in shapes {
            let x = Param::from_tensor(det_factor(rows, cols, &adev)).val();
            assert!(x.is_require_grad(), "fixture is not a tracked master");
            let r = polar_retracked(&x, 3);
            let expect = polar_orthogonalize_host_read(x, 3);
            assert!(
                r.is_require_grad(),
                "[{rows},{cols}] retract dropped the tracked master"
            );
            assert_eq!(
                r.into_data().try_to_vec::<f32>().unwrap(),
                expect.into_data().try_to_vec::<f32>().unwrap(),
                "tracked sync-free retraction changed the numbers at [{rows},{cols}]"
            );
        }
    }

    #[test]
    fn retract_mirrors_master_tracking() {
        // Under autodiff the masters are tracked, and the retraction must hand
        // burn-optim a tracked leaf or the master silently freezes.
        let mut m = SpectralLinear::new(64, 128, 8, &Device::ndarray().autodiff());
        m.retract(5);
        assert!(
            m.u.val().is_require_grad(),
            "retract dropped the tracked master on an autodiff backend"
        );
        // With no autodiff at all, `set_require_grad` is a hard panic, and a
        // forced flag would silently un-freeze a master the caller froze.
        // Both are regressions this test exists to catch.
        let mut m = SpectralLinear::new(64, 128, 8, &Device::ndarray());
        m.retract(5);
        assert!(
            !m.u.val().is_require_grad(),
            "retract silently un-froze a master on a non-autodiff backend"
        );
    }

    #[test]
    fn retract_keeps_masters_tracked() {
        // regression: the polar output is a non-leaf (GradInBackward);
        // retract must re-track it as a leaf, otherwise burn-optim's step
        // only recognizes Requirement::Grad and silently downgrades the
        // param to an untracked leaf (the master freezes, and the fused
        // op's backward would see a pruned parent)
        let adev = Device::ndarray().autodiff();
        let mut m = SpectralLinear::new(64, 128, 8, &adev);
        m.retract(3);
        assert!(m.u.val().is_require_grad(), "retracted u must stay tracked");
        assert!(m.v.val().is_require_grad(), "retracted v must stay tracked");
    }

    /// The batched path must be numerically identical to per-factor
    /// retraction: mixed shapes (tall, wide, duplicate-shape groups) go in,
    /// every result matches `polar_orthogonalize` within fp tolerance and is
    /// orthonormal where the factor is full-rank.
    #[test]
    fn retract_batched_identity_with_per_factor_path() {
        let dev = dev();
        let shapes = [
            (64, 8),
            (128, 16),
            (64, 8), // same shape as factor 0: exercises grouping
            (32, 8),
            (8, 64), // wide: rows < cols, no canonical transpose needed
            (96, 32),
            (48, 24),
            (200, 10), // lone shape: its own batch of one
        ];
        let mut fs: Vec<Tensor<2>> = shapes
            .iter()
            .map(|&(m, k)| Tensor::<2>::random([m, k], Distribution::Normal(0.0, 1.0), &dev))
            .collect();
        let orig: Vec<Tensor<2>> = fs.to_vec();
        let mut refs: Vec<&mut Tensor<2>> = fs.iter_mut().collect();
        // 10 iterations: raw-Gaussian inputs need them to converge (the
        // identity comparison holds at any count; this one keeps the
        // orthonormality assertions honest).
        retract_batched(&mut refs, 10);
        for (i, f) in fs.iter().enumerate() {
            let expect = polar_orthogonalize(orig[i].clone(), 10);
            let d = f.clone().sub(expect).abs().max().into_scalar::<f32>();
            assert!(d < 1e-5, "factor {i} {:?} maxdiff {d}", shapes[i]);
            // Shapes must survive untouched.
            assert_eq!(
                f.dims(),
                [shapes[i].0, shapes[i].1],
                "factor {i} shape changed"
            );
            // A wide polar factor has orthonormal ROWS (the wide test factor
            // is [8, 64]); columns are orthonormal only for tall outputs.
            let oe = if shapes[i].0 < shapes[i].1 {
                ortho_error(&f.clone().transpose())
            } else {
                ortho_error(f)
            };
            assert!(oe < 5e-2, "factor {i} {:?} ortho error {oe}", shapes[i]);
        }
    }

    #[test]
    fn retract_batched_deterministic() {
        let dev = dev();
        let inputs = vec![
            Tensor::<2>::random([96, 24], Distribution::Normal(0.0, 1.0), &dev),
            Tensor::<2>::random([96, 24], Distribution::Normal(0.0, 1.0), &dev),
        ];
        let run = |inputs: Vec<Tensor<2>>| {
            let mut fs = inputs;
            let mut refs: Vec<&mut Tensor<2>> = fs.iter_mut().collect();
            retract_batched(&mut refs, 3);
            fs.iter()
                .flat_map(|t| t.clone().into_data().try_to_vec::<f32>().unwrap())
                .collect::<Vec<f32>>()
        };
        // Same inputs through the same grouped pipeline -> bit-identical.
        let a = run(inputs.clone());
        let b = run(inputs);
        assert_eq!(a, b, "batched retraction must be deterministic");
    }

    // Honest microbench on ndarray: no device syncs exist here, so this
    // measures pure compute/overhead of old per-factor vs new batched path
    // over 20 factors of varied shape. GPU sync-count verification stays a
    // follow-up (GPU busy with training).
    #[test]
    fn bench_old_vs_batched_retraction() {
        let dev = dev();
        let shapes = [
            (256, 32),
            (512, 64),
            (128, 16),
            (384, 48),
            (96, 24),
            (640, 80),
            (160, 20),
            (320, 40),
        ];
        let mk = || -> Vec<Tensor<2>> {
            (0..20)
                .map(|i| {
                    let (m, k) = shapes[i % shapes.len()];
                    Tensor::<2>::random([m, k], Distribution::Normal(0.0, 1.0), &dev)
                })
                .collect()
        };
        let iters = 3;
        let rounds = 3;
        // Warmup (allocator/BLAS threads).
        for f in mk() {
            let _ = polar_orthogonalize(f, iters);
        }
        retract_batched(&mut mk().iter_mut().collect::<Vec<&mut Tensor<2>>>(), iters);
        let t0 = std::time::Instant::now();
        for _ in 0..rounds {
            for f in mk() {
                let _ = polar_orthogonalize(f, iters);
            }
        }
        let old = t0.elapsed().as_secs_f64() / rounds as f64 * 1e3;
        let t1 = std::time::Instant::now();
        for _ in 0..rounds {
            retract_batched(&mut mk().iter_mut().collect::<Vec<&mut Tensor<2>>>(), iters);
        }
        let new = t1.elapsed().as_secs_f64() / rounds as f64 * 1e3;
        println!(
            "retract 20 factors x{rounds} rounds: per-factor {old:.2}ms vs batched {new:.2}ms"
        );
    }

    #[test]
    fn ste_ternary_is_ternary_values() {
        let w = Tensor::<2>::random([16, 16], Distribution::Normal(0.0, 1.0), &dev());
        let w2 = w.clone();
        let t = ste_ternary(w2.clone());
        let wv: Vec<f32> = w.into_data().try_to_vec().unwrap();
        let tv: Vec<f32> = t.into_data().try_to_vec().unwrap();
        // forward value must equal ternarize(w), not w
        let ref_vals = ternarize(w2);
        let rv: Vec<f32> = ref_vals.into_data().try_to_vec().unwrap();
        for (a, b) in tv.iter().zip(rv.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
        // and it must differ from the raw master somewhere
        let diff = tv
            .iter()
            .zip(wv.iter())
            .filter(|(a, b)| (**a - **b).abs() > 1e-4)
            .count();
        assert!(diff > 0, "ternary forward must differ from master");
    }

    #[test]
    fn ste_backward_flows_through_master() {
        // autodiff: gradient of loss w.r.t. the master must be non-zero
        // STE identity must keep gradients alive
        let dev = Device::ndarray().autodiff();
        let m: SpectralLinear = SpectralLinear::new(8, 8, 4, &dev);
        let x = Tensor::<2>::random([2, 8], Distribution::Default, &dev);
        let y = m.forward(x).powf_scalar(2.0).sum();
        let grads = y.backward();
        let gu = m.u.grad(&grads).unwrap();
        let gv: Vec<f32> = gu.into_data().try_to_vec().unwrap();
        assert!(gv.iter().any(|&g| g.abs() > 1e-6), "STE gradient lost");
    }

    #[test]
    fn asym_forward_uses_raw_v() {
        // asym mode: U ternary, V raw fp32. With identical masters the asym
        // output must equal the manual (tern(U)@x)*s@V_raw^T reference and
        // differ from the plain-mode output (which ternarizes V too).
        let dev = Device::ndarray();
        let plain = SpectralLinear::new(16, 32, 4, &dev);
        let mut asym = SpectralLinear::new(16, 32, 4, &dev);
        asym.u = plain.u.clone();
        asym.s = plain.s.clone();
        asym.v = plain.v.clone();
        asym.set_asym(true);
        let x = Tensor::<2>::random([4, 16], Distribution::Default, &dev);
        let y_asym = asym.forward(x.clone());
        let y_plain = plain.forward(x.clone());
        let u_t = ternarize(plain.u.val());
        let y_ref = x
            .matmul(u_t)
            .mul(plain.s.val().unsqueeze_dims(&[0]))
            .matmul(plain.v.val().transpose());
        let max_diff = |a: Tensor<2>, b: Tensor<2>| a.sub(b).abs().max().into_scalar::<f32>();
        assert!(
            max_diff(y_asym.clone(), y_ref) < 1e-4,
            "asym forward must use raw V"
        );
        assert!(
            max_diff(y_asym, y_plain) > 1e-3,
            "asym must differ from plain (V quantized there)"
        );
    }

    #[test]
    fn two_bit_forward_matches_reference() {
        // 2-bit mode: both factors quantized to {-2s,-s,0,s,2s} (per-row
        // scale) via weight_quant_2bit, s stays fp32. Output must equal the
        // manual (x@U_2b)*s@V_2b^T reference and differ from plain ternary.
        let dev = Device::ndarray();
        let plain = SpectralLinear::new(16, 32, 4, &dev);
        let mut two = SpectralLinear::new(16, 32, 4, &dev);
        two.u = plain.u.clone();
        two.s = plain.s.clone();
        two.v = plain.v.clone();
        two.set_2bit(true);
        let x = Tensor::<2>::random([4, 16], Distribution::Default, &dev);
        let y_two = two.forward(x.clone());
        let y_plain = plain.forward(x.clone());
        let u_2b = burn_bitnet::weight_quant_2bit(plain.u.val());
        let v_2b = burn_bitnet::weight_quant_2bit(plain.v.val());
        let y_ref = x
            .matmul(u_2b)
            .mul(plain.s.val().unsqueeze_dims(&[0]))
            .matmul(v_2b.transpose());
        let max_diff = |a: Tensor<2>, b: Tensor<2>| a.sub(b).abs().max().into_scalar::<f32>();
        assert!(
            max_diff(y_two.clone(), y_ref) < 1e-4,
            "2bit forward must equal the weight_quant_2bit reference"
        );
        assert!(
            max_diff(y_two, y_plain) > 1e-3,
            "2bit must differ from plain ternary forward"
        );
    }

    #[test]
    fn nm_forward_is_sparse_and_matches_reference() {
        // N:M on: the ternary factors are exactly N:M sparse per block and
        // the forward output equals the manual masked-ternary reference.
        let dev = Device::ndarray();
        for (n, m) in [(6usize, 8usize), (2, 4)] {
            let mut layer = SpectralLinear::new(16, 32, 16, &dev);
            layer.set_nm(n, m);
            let x = Tensor::<2>::random([4, 16], Distribution::Default, &dev);
            let y = layer.forward(x.clone());
            assert!(
                y.clone()
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .all(|v| v.is_finite()),
                "nm {n}:{m} forward produced NaN"
            );
            // factors are N:M sparse along the rank dim (blocks of m), same
            // computation the forward runs via weight_quant_ternary_nm
            let u_t = burn_bitnet::weight_quant_ternary_nm(layer.u.val(), n, m);
            let v_t = burn_bitnet::weight_quant_ternary_nm(layer.v.val(), n, m);
            // N:M guarantee (post b1.58/v2 Eq.1 realignment): the MASK keeps
            // exactly n positions per m-block; a picked weight may quantize
            // to 0 when |w| < gamma/2, so nonzero count is <= n and every
            // nonzero must lie on a mask-on position with ternary value.
            let gamma_u: f32 = layer.u.val().abs().mean().into_scalar::<f32>();
            let gamma_v: f32 = layer.v.val().abs().mean().into_scalar::<f32>();
            for (name, t, gamma) in [("u", u_t.clone(), gamma_u), ("v", v_t.clone(), gamma_v)] {
                for (bi, block) in t
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap()
                    .chunks_exact(m)
                    .enumerate()
                {
                    let nz = block.iter().filter(|x| x.abs() > 1e-3).count();
                    assert!(nz <= n, "{name} block {bi}: {nz} nonzeros > n={n}");
                    for (j, val) in block.iter().enumerate() {
                        if val.abs() > 1e-3 {
                            let k = (val / gamma).round();
                            assert!(
                                ((val - k * gamma).abs() < 1e-3) && k.abs() <= 1.0,
                                "{name} block {bi} pos {j}: {val} not ternary at gamma={gamma}"
                            );
                        }
                    }
                }
            }
            // forward must equal y = (x@U_t) * s @ V_t^T with the masked U_t/V_t
            let y_ref = x
                .matmul(u_t)
                .mul(layer.s.val().unsqueeze_dims(&[0]))
                .matmul(v_t.transpose());
            let md = y.sub(y_ref).abs().max().into_scalar::<f32>();
            assert!(
                md < 1e-4,
                "nm {n}:{m} forward vs masked reference maxdiff {md}"
            );
        }
    }

    #[test]
    fn nm_dual_ste_grad_flows_to_masked_entries() {
        // Dual-STE: a masked (pruned) u/v entry must still receive a gradient.
        // With gated-gradient pruning (t*m multiplied through backward) the
        // masked entry's gradient is exactly 0 and this test fails.
        let dev = Device::ndarray().autodiff();
        let mut m: SpectralLinear = SpectralLinear::new(16, 32, 16, &dev);
        m.set_nm(6, 8);
        let x = Tensor::<2>::random([4, 16], Distribution::Default, &dev);
        let loss = m.forward(x).powf_scalar(2.0).sum();
        let grads = loss.backward();
        let gu: Vec<f32> = m.u.grad(&grads).unwrap().into_data().try_to_vec().unwrap();
        let gv: Vec<f32> = m.v.grad(&grads).unwrap().into_data().try_to_vec().unwrap();
        // mask positions are decided by |master| per block of 8 along the rank
        // dim: zero entries in compute_nm_mask are exactly the pruned ones
        let mask_u: Vec<f32> = burn_bitnet::sparse::compute_nm_mask(m.u.val(), 6, 8)
            .into_data()
            .to_vec()
            .unwrap();
        let mask_v: Vec<f32> = burn_bitnet::sparse::compute_nm_mask(m.v.val(), 6, 8)
            .into_data()
            .to_vec()
            .unwrap();
        assert!(
            mask_u.contains(&0.0) && mask_v.contains(&0.0),
            "nm mask must prune some entries"
        );
        for (name, g, mask) in [("u", gu, mask_u), ("v", gv, mask_v)] {
            let pruned = g.iter().zip(&mask).filter(|(_, &mk)| mk == 0.0).count();
            assert!(pruned > 0, "{name}: no masked entries in grad vector");
            let nz = g
                .iter()
                .zip(&mask)
                .filter(|(&gr, &mk)| mk == 0.0 && gr.abs() > 1e-8)
                .count();
            assert!(
                nz == pruned,
                "{name}: {nz}/{pruned} masked entries got a gradient (Dual-STE must reach all)"
            );
        }
    }

    #[test]
    fn nm_off_is_bit_identical_to_plain_ternary() {
        // default (nm off) must keep the pre-change forward: the layer output
        // equals the plain annealed-ternary reference, and set_nm(0,0) makes
        // no difference.
        let dev = Device::ndarray();
        let layer = SpectralLinear::new(16, 32, 8, &dev);
        let x = Tensor::<2>::random([4, 16], Distribution::Default, &dev);
        let y = layer.forward(x.clone());
        // pre-change reference: alpha=1 annealed ternary on both factors
        let y_ref = x
            .clone()
            .matmul(ste_ternary(layer.u.val()))
            .mul(layer.s.val().unsqueeze_dims(&[0]))
            .matmul(ste_ternary(layer.v.val()).transpose());
        let md = y.sub(y_ref).abs().max().into_scalar::<f32>();
        assert_eq!(md, 0.0, "nm off must be bit-identical to plain ternary");
        // explicit (0,0) on the same masters is the same path
        let mut off = layer.clone();
        off.set_nm(0, 0);
        let md2 = layer
            .forward(x.clone())
            .sub(off.forward(x))
            .abs()
            .max()
            .into_scalar::<f32>();
        assert_eq!(md2, 0.0, "set_nm(0,0) must equal the default forward");
    }

    #[test]
    fn nm_asym_keeps_v_raw() {
        // nm + asym: only U is masked/quantized, V runs raw fp32.
        let dev = Device::ndarray();
        let mut layer = SpectralLinear::new(16, 32, 16, &dev);
        layer.set_nm(6, 8);
        layer.set_asym(true);
        let x = Tensor::<2>::random([4, 16], Distribution::Default, &dev);
        let y = layer.forward(x.clone());
        let u_t = burn_bitnet::weight_quant_ternary_nm(layer.u.val(), 6, 8);
        let y_ref = x
            .matmul(u_t)
            .mul(layer.s.val().unsqueeze_dims(&[0]))
            .matmul(layer.v.val().transpose());
        let md = y.clone().sub(y_ref).abs().max().into_scalar::<f32>();
        assert!(
            md < 1e-4,
            "nm+asym must use masked U and raw V: maxdiff {md}"
        );
        assert!(
            y.into_data()
                .try_to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite()),
            "nm+asym forward produced NaN"
        );
    }

    #[test]
    fn tsct_moe_shapes() {
        // 8 clusters x 8 experts = 64 experts, top-2
        let m = SpectralMoE::new(32, 64, 8, 8, 2, 4, &dev());
        let x = Tensor::<2>::random([4, 32], Distribution::Default, &dev());
        assert_eq!(m.forward(x).dims(), [4, 64]);
    }

    #[test]
    fn tsct_moe_topk_distinct() {
        let m = SpectralMoE::new(32, 64, 8, 8, 2, 4, &dev());
        let x = Tensor::<2>::random([4, 32], Distribution::Default, &dev());
        let (_c, e) = m.router_logits(x);
        let idx = m.topk_indices(e);
        let v: Vec<i64> = idx.into_data().try_to_vec().unwrap();
        for row in v.chunks_exact(2) {
            assert_ne!(row[0], row[1], "top-k must be distinct: {row:?}");
            assert!(row[0] < 8 && row[1] < 8);
        }
    }

    /// Naive host reference: sort each row descending by value (ties by lowest
    /// index, matching burn's argtopk contract) and take the first `k`.
    fn naive_topk(logits: &Tensor<2>, k: usize) -> Vec<i64> {
        let [b, n] = logits.dims();
        let vals: Vec<f32> = logits.clone().into_data().try_to_vec().unwrap();
        let mut out = Vec::with_capacity(b * k);
        for r in 0..b {
            let mut row: Vec<(f32, i64)> = (0..n).map(|c| (vals[r * n + c], c as i64)).collect();
            row.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            out.extend(row.into_iter().take(k).map(|(_, i)| i));
        }
        out
    }

    #[test]
    fn topk_indices_descending_matches_naive() {
        // distinct values per row (no ties), wide enough to exercise the
        // k > 16 round-cap and the k == n_pos branch.
        let logits = Tensor::<2>::from_floats(
            [
                [
                    3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0, 5.3, 5.8, 9.7, 9.3, 2.3, 8.4, 6.2, 6.4,
                    3.3, 8.3, 2.7, 9.5,
                ],
                [
                    9.8, 0.0, 7.1, 4.0, 6.2, 2.7, 9.3, 0.8, 7.9, 0.4, 8.3, 6.1, 1.5, 2.1, 4.7, 7.2,
                    6.0, 8.5, 1.3, 9.0,
                ],
            ],
            &dev(),
        );
        for k in [1usize, 2, 16, 20] {
            let got = SpectralMoE::topk_indices_generic(logits.clone(), k);
            let gv: Vec<i64> = got.into_data().try_to_vec().unwrap();
            assert_eq!(gv, naive_topk(&logits, k), "topk mismatch for k={k}");
        }
        // small n_pos: k == n_pos (full selection) and k == 1
        let small = Tensor::<2>::from_floats(
            [[1.0, 5.0, 3.0, 9.0, 2.0], [8.0, 0.0, 7.0, 4.0, 6.0]],
            &dev(),
        );
        for k in [1usize, 5] {
            let got = SpectralMoE::topk_indices_generic(small.clone(), k);
            let gv: Vec<i64> = got.into_data().try_to_vec().unwrap();
            assert_eq!(gv, naive_topk(&small, k), "small topk mismatch for k={k}");
        }
    }

    #[test]
    fn tsct_moe_param_win() {
        // 128 clusters x 8 experts = 1024 patterns (rank 4), top-2. MoE has MORE
        // parameters than dense (that is the point): the win is FLOPs.
        let m = SpectralMoE::new(512, 4096, 128, 8, 2, 4, &dev());
        assert_eq!(m.num_experts(), 1024);
        let flops = m.flops();
        let dense_flops = m.in_features * m.out_features;
        assert!(
            flops * 20 < dense_flops,
            "FLOPs {flops} vs dense {dense_flops}: need >20x fewer"
        );
        // and the expert masters stay VRAM-friendly: ~19M x 12B = 226MB
        // (FP32 master + Adam m/v) per FFN layer
        assert!(
            m.param_count() < 30_000_000,
            "masters {} too big",
            m.param_count()
        );
        // capacity beats dense by 4.6x, which is where the intelligence lives
        assert!(m.param_count() > 4 * m.in_features * m.out_features);
    }

    #[test]
    fn tsct_moe_ec_shapes() {
        let mut m = SpectralMoE::new(32, 64, 8, 8, 2, 4, &dev());
        m.set_expert_choice(true);
        let x = Tensor::<2>::random([4, 32], Distribution::Default, &dev());
        let out = m.forward(x);
        assert_eq!(out.dims(), [4, 64]);
        let v: Vec<f32> = out.into_data().try_to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "EC forward produced NaN");
        // B > n_clusters: clusters hold several tokens (B=16 into C=8)
        let x = Tensor::<2>::random([16, 32], Distribution::Default, &dev());
        assert_eq!(m.forward(x).dims(), [16, 64]);
    }

    #[test]
    fn tsct_moe_ec_flops_and_topk_cost() {
        let mut m = SpectralMoE::new(32, 64, 8, 8, 2, 4, &dev());
        // router is MACs, not params: proj 32*32 + cluster_key 32*8 +
        // expert_key 32*8 = 1024 + 256 + 256 = 1536
        let router = m.router_flops();
        assert_eq!(router, 1536);
        // token-choice default: top_k * rank * (in + out) + router
        assert_eq!(m.flops(), 2 * 4 * (32 + 64) + router);
        assert_eq!(m.topk_cost(1), 0, "token-choice has no top-k loop");
        m.set_expert_choice(true);
        // EC at n=64 rows: n_c=8, k_e=min(ceil(8*2/8),8)=2, picks=ceil(8*2/8)=2
        // -> equals token-choice top_k=2
        assert_eq!(m.flops_n(64), 2 * 4 * (32 + 64) + router);
        // n=1 -> n_c=1, k_e=1, picks=ceil(8*1/1)=8: every expert picks the
        // single token (the k_e>=1 floor), 8x the top_k=2 cost
        assert_eq!(m.flops_n(1), 8 * 4 * (32 + 64) + router);
        // flops() is token-choice-only: EC must go through flops_n
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m.flops())).is_err());
        // C=8, E=8, cap=2: 1 row -> n_c=1, k_e=min(ceil(1*2/8),1)=1
        assert_eq!(m.topk_cost(1), 8 * 8);
        // 16384 rows -> n_c=2048, k_e=min(ceil(2048*2/8),2048)=512
        assert_eq!(m.topk_cost(16384), 8 * 512 * 8 * 2048);
    }

    #[test]
    fn tsct_moe_flops_scales_with_rank() {
        // the rank factor must be present in the accounting: 2x rank ->
        // 2x expert MACs (router is identical, so the difference is exact)
        let low = SpectralMoE::new(32, 64, 8, 8, 2, 2, &dev());
        let high = SpectralMoE::new(32, 64, 8, 8, 2, 4, &dev());
        assert_eq!(high.flops() - low.flops(), 2 * (4 - 2) * (32 + 64));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tsct_moe_ec_cuda() {
        let dev = Device::cuda(0);
        let mut m = SpectralMoE::new(32, 64, 8, 8, 2, 4, &dev);
        m.set_expert_choice(true);
        let x = Tensor::<2>::random([32, 32], Distribution::Default, &dev);
        let out = m.forward(x);
        assert_eq!(out.dims(), [32, 64]);
        let v: Vec<f32> = out.into_data().try_to_vec().unwrap();
        assert!(
            v.iter().all(|x| x.is_finite()),
            "EC CUDA forward produced NaN"
        );
    }

    #[test]
    fn tsct_moe_ec_gradient_utilization() {
        // dead-expert kill test: with Expert-Choice (almost) all master
        // columns must receive gradient every step; token-choice top-2
        // starves most of them (the 2202.09368 motivation)
        let dev = Device::ndarray().autodiff();
        let run = |ec: bool, e: usize| -> (f32, f32) {
            let mut m: SpectralMoE = SpectralMoE::new(16, 32, 4, e, 2, 4, &dev);
            m.set_expert_choice(ec);
            let [in_f, m_cols] = m.u.val().dims(); // [16, 128]
            let mut fracs = Vec::new();
            let lr = 0.05f32;
            for _ in 0..30 {
                let x = Tensor::<2>::random([16, 16], Distribution::Normal(0.0, 1.0), &dev);
                let y = Tensor::<2>::random([16, 32], Distribution::Default, &dev);
                let loss = m.forward(x).sub(y).powf_scalar(2.0).mean();
                let grads = loss.backward();
                let gu: Vec<f32> = m.u.grad(&grads).unwrap().into_data().try_to_vec().unwrap();
                let used = gu
                    .chunks_exact(in_f)
                    .filter(|c| c.iter().any(|g| g.abs() > 1e-8))
                    .count();
                fracs.push(used as f32 / m_cols as f32);
                // manual SGD on the masters (router grads exist but are not
                // part of the utilization metric)
                let gu = m.u.grad(&grads).unwrap();
                let gv = m.v.grad(&grads).unwrap();
                let gs = m.s.grad(&grads).unwrap();
                // 0.22 Param::grad returns a plain tensor; strip the value's
                // autodiff, subtract, rewrap
                m.u = Param::from_tensor(Tensor::from_inner(
                    m.u.val().inner().sub(gu.mul_scalar(lr)),
                ));
                m.v = Param::from_tensor(Tensor::from_inner(
                    m.v.val().inner().sub(gv.mul_scalar(lr)),
                ));
                m.s = Param::from_tensor(Tensor::from_inner(
                    m.s.val().inner().sub(gs.mul_scalar(lr)),
                ));
            }
            (fracs[29], fracs[20..].iter().sum::<f32>() / 10.0)
        };
        let (ec_last, ec_avg) = run(true, 8);
        let (tc_last, _) = run(false, 8);
        // token-choice with 32 experts per cluster (same C, same 128 master
        // columns): at B=16 top-2 picks cover only a few experts per step
        let (tc_starve, _) = run(false, 32);
        println!(
            "u-column utilization: EC last {ec_last:.3} avg10 {ec_avg:.3} vs token-choice last {tc_last:.3} (E=8) / {tc_starve:.3} (E=32)"
        );
        assert!(ec_last > 0.5, "EC final-step utilization {ec_last} <= 0.5");
        assert!(ec_avg > 0.5, "EC last-10 avg utilization {ec_avg} <= 0.5");
    }

    #[test]
    fn tsct_moe_ec_matches_reference() {
        // full-pipeline check: forward_ec vs a host reference of the same
        // math (per-cluster token lists, per-expert top-k, softmax gates,
        // column mapping, scatter-add accumulation)
        let dev = Device::ndarray();
        let m = SpectralMoE::new(8, 12, 3, 4, 2, 2, &dev); // C=3, E=4, r=2
        let xv: Vec<f32> = (0..(12 * 8))
            .map(|i| ((i * 7) % 13) as f32 / 5.0 - 1.2)
            .collect();
        let x = Tensor::<2>::from_data(burn::tensor::TensorData::new(xv.clone(), [12, 8]), &dev);
        let out = m.forward_ec(x.clone());
        let y: Vec<f32> = out.into_data().try_to_vec().unwrap();

        // host reference --------------------------------------------------
        let (c_logits, e_logits) = m.router_logits(x.clone());
        let cl: Vec<f32> = c_logits.into_data().try_to_vec().unwrap();
        let el: Vec<f32> = e_logits.into_data().try_to_vec().unwrap();
        let (c, e, r) = (3usize, 4usize, 2usize);
        let cap = 2.0f32;
        let (in_f, mcols, out_f) = (8usize, 24usize, 12usize);
        // ternary projections of the masters (STE forward values): one
        // global mean, dead zone at 0.7*mean
        let tern = |w: Vec<f32>| -> Vec<f32> {
            let mean = w.iter().map(|v| v.abs()).sum::<f32>() / w.len() as f32;
            w.iter()
                .map(|&v| {
                    let keep = if v.abs() > 0.7 * mean { 1.0 } else { 0.0 };
                    v.signum() * mean * keep
                })
                .collect()
        };
        let u_m: Vec<f32> = m.u.val().into_data().try_to_vec().unwrap();
        let v_m: Vec<f32> = m.v.val().into_data().try_to_vec().unwrap();
        let s_m: Vec<f32> = m.s.val().into_data().try_to_vec().unwrap();
        let (u_t, v_t) = (tern(u_m), tern(v_m)); // [in, M*r] / [out, M*r] row-major
        let softmax1 = |v: &[f32]| -> Vec<f32> {
            let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let ex: Vec<f32> = v.iter().map(|&x| (x - mx).exp()).collect();
            let sum = ex.iter().sum::<f32>();
            ex.iter().map(|x| x / sum).collect()
        };
        let mut acc = vec![0.0f32; 12 * out_f];
        for ci in 0..c {
            let toks: Vec<usize> = (0..12)
                .filter(|&t| {
                    (0..c).max_by(|&a, &b| cl[t * c + a].partial_cmp(&cl[t * c + b]).unwrap())
                        == Some(ci)
                })
                .collect();
            let n_c = toks.len();
            if n_c == 0 {
                continue;
            }
            let k_e = (((n_c as f32 * cap) / e as f32).ceil().max(1.0) as usize).min(n_c);
            for e_i in 0..e {
                let scores: Vec<f32> = toks.iter().map(|&t| el[t * e + e_i]).collect();
                // top-k_e distinct picks, masked-argmax semantics (first max)
                let mut picks: Vec<usize> = Vec::new();
                let mut used = vec![false; n_c];
                for _ in 0..k_e {
                    let mut best: Option<(f32, usize)> = None;
                    for (j, &sc) in scores.iter().enumerate() {
                        if used[j] {
                            continue;
                        }
                        if best.is_none() || sc > best.unwrap().0 + 1e-12 {
                            best = Some((sc, j));
                        }
                    }
                    let (_, j) = best.unwrap();
                    used[j] = true;
                    picks.push(j);
                }
                let g_all = softmax1(&scores); // gates over the cluster's tokens
                for &j in &picks {
                    let gate = g_all[j];
                    let t = toks[j];
                    let c_w = softmax1(&cl[t * c..(t + 1) * c])[ci];
                    for rr in 0..r {
                        let col = (ci * e + e_i) * r + rr;
                        let dot = (0..in_f)
                            .map(|i| xv[t * in_f + i] * u_t[i * mcols + col])
                            .sum::<f32>();
                        let z = s_m[col] * dot * gate * c_w;
                        for o in 0..out_f {
                            acc[t * out_f + o] += z * v_t[o * mcols + col];
                        }
                    }
                }
            }
        }
        let max_diff = acc
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("EC vs host reference: max diff {max_diff}");
        assert!(max_diff < 1e-3, "EC mismatch vs reference: {max_diff}");
    }

    #[test]
    fn tsct_moe_ec_grads_flow() {
        // the router must train under Expert-Choice too: gates carry the
        // gradient, selection indices are detached Ints
        let dev = Device::ndarray().autodiff();
        let mut m: SpectralMoE = SpectralMoE::new(16, 32, 4, 8, 2, 4, &dev);
        m.set_expert_choice(true);
        let x = Tensor::<2>::random([8, 16], Distribution::Default, &dev);
        let loss = m.forward(x).powf_scalar(2.0).sum();
        let grads = loss.backward();
        for (name, g) in [
            ("u", m.u.grad(&grads)),
            ("v", m.v.grad(&grads)),
            ("proj.weight", m.proj.weight.grad(&grads)),
            ("cluster_key.weight", m.cluster_key.weight.grad(&grads)),
            ("expert_key.weight", m.expert_key.weight.grad(&grads)),
        ] {
            let g = g.unwrap_or_else(|| panic!("no grad on {name}"));
            let gv: Vec<f32> = g.into_data().try_to_vec().unwrap();
            assert!(
                gv.iter().any(|&x| x.abs() > 1e-8),
                "all-zero gradient on {name}"
            );
        }
        let gs: Vec<f32> = m.s.grad(&grads).unwrap().into_data().try_to_vec().unwrap();
        assert!(gs.iter().any(|&x| x.abs() > 1e-8), "all-zero gradient on s");
    }
}

#[cfg(test)]
mod polar_diag {
    use super::*;
    use burn::tensor::{Device, Distribution};

    #[test]
    fn polar_on_random() {
        let dev = Device::ndarray();
        let x = Tensor::<2>::random([64, 8], Distribution::Normal(0.0, 1.0), &dev);
        let p = polar_orthogonalize(x.clone(), 10);
        let e = ortho_error(&p);
        println!("polar 10 iters err={e}");
        assert!(e < 1e-2, "err {e}");
    }

    #[test]
    fn polar_five_iters() {
        let dev = Device::ndarray();
        let x = Tensor::<2>::random([64, 8], Distribution::Normal(0.0, 1.0), &dev);
        let p = polar_orthogonalize(x, 5);
        let e = ortho_error(&p);
        println!("polar 5 iters err={e}");
        assert!(e < 1e-1, "err {e}");
    }

    #[test]
    fn polar_three_iters_no_corruption() {
        // regression: the pre-fix Frobenius-norm scaling under-scaled every
        // singular value by sqrt(min(m,n)) so retract(3) left ortho error at
        // ~4.1 (WORSE than the input) instead of converging — measured with
        // aria's PARAM=tsct (rank-64 masters, RETRACT_ITERS=3).
        let dev = Device::ndarray();
        for [m, n] in [[64, 8], [512, 64], [2048, 64]] {
            let x = Tensor::<2>::random([m, n], Distribution::Normal(0.0, 1.0), &dev);
            let e_in = ortho_error(&x);
            let p = polar_orthogonalize(x, 3);
            let e_out = ortho_error(&p);
            assert!(
                e_out < e_in.max(1e-2),
                "retract(3) must not corrupt orthonormality: in={e_in:.2e} out={e_out:.2e}"
            );
        }
    }

    #[test]
    fn polar_square_and_tall_no_divergence() {
        // regression: square random matrices had sigma_max ≈ 2 after the old
        // Frobenius/sqrt(k) pre-scale (above the cubic NS basin < sqrt(3)),
        // so NS diverged (max entry ~1e14). Spectral-norm scaling keeps
        // sigma_max ≈ 1, so NS stays bounded and converges. Note: a square
        // random matrix has a full singular-value spectrum (condition number
        // ~ n), so 3 cubic NS iters only partially converge it (bounded by
        // sqrt(512) ≈ 23, vs 1e14 diverged); enough iters give a fully
        // orthonormal Q. Tall/wide cases (σ concentrated ≈ 1) are orthonormal
        // in 3 iters.
        let dev = Device::ndarray();
        // square: no divergence at 3 iters, orthonormal at 20
        let x = Tensor::<2>::random([512, 512], Distribution::Normal(0.0, 1.0), &dev);
        let e3 = ortho_error(&polar_orthogonalize(x.clone(), 3));
        let e20 = ortho_error(&polar_orthogonalize(x, 20));
        println!("polar [512x512] err(3)={e3:.3e} err(20)={e20:.3e}");
        assert!(e3 < 100.0, "square 3-iter diverged: {e3}");
        assert!(e20 < 1e-2, "square 20-iter not orthonormal: {e20}");
        // tall/wide: orthonormal in 3 iters (wide checked via its transpose,
        // since a [64,512] polar factor has orthonormal rows, not columns)
        for [m, n] in [[4096, 512], [512, 64]] {
            let x = Tensor::<2>::random([m, n], Distribution::Normal(0.0, 1.0), &dev);
            let p = polar_orthogonalize(x, 3);
            let e = ortho_error(&p);
            println!("polar 3 iters [{m}x{n}] err={e:.3e}");
            assert!(e < 1e-2, "polar [{m}x{n}] err {e}");
        }
        let x = Tensor::<2>::random([64, 512], Distribution::Normal(0.0, 1.0), &dev);
        let p = polar_orthogonalize(x, 3);
        let e = ortho_error(&p.transpose());
        println!("polar 3 iters [64x512] row-ortho err={e:.3e}");
        assert!(e < 1e-2, "polar [64x512] row-ortho err {e}");
        // near-converged master (orthonormal-ish init) must stay put
        let x = Tensor::<2>::random([512, 64], Distribution::Normal(0.0, 1.0), &dev);
        let q = polar_orthogonalize(x.clone(), 8);
        let q = q.add(Tensor::<2>::random(
            [512, 64],
            Distribution::Normal(0.0, 0.1),
            &dev,
        ));
        let e = ortho_error(&polar_orthogonalize(q, 3));
        println!("polar 3 iters [512x64] near-orthonormal err={e:.3e}");
        assert!(e < 1e-2, "near-orthonormal retract err {e}");
    }
}

#[cfg(test)]
mod stochastic_tests {
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn stochastic_unbiased() {
        let dev = Device::ndarray();
        // w in [-scale, scale]: E[w_t] = w
        let w = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(vec![0.5f32, -0.3, 0.9, 0.1, -0.7, 0.0], [2, 3]),
            &dev,
        );
        let scale = w.clone().abs().mean().into_scalar::<f32>();
        // many samples -> mean ~= w (ternary values are sign*scale)
        let n = 4000;
        let mut sum = [0.0f32; 6];
        for _ in 0..n {
            let t = ternarize_stochastic(w.clone(), &dev);
            let v: Vec<f32> = t.into_data().try_to_vec().unwrap();
            for (i, x) in v.iter().enumerate() {
                sum[i] += x;
            }
        }
        // E[w_t] = sign(w) * min(|w|, scale): elements above scale saturate
        // (same as BitNet), below scale stay unbiased
        let expected = [0.4167f32, -0.3, 0.4167, 0.1, -0.4167, 0.0];
        for (i, m) in sum.iter().enumerate() {
            let m = m / n as f32;
            assert!(
                (m - expected[i]).abs() < 0.06,
                "E[w_t] {m} != {}",
                expected[i]
            );
        }
        let _ = scale;
    }

    #[test]
    fn stochastic_values_are_ternary() {
        let dev = Device::ndarray();
        let w = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(vec![0.4f32, -0.8, 0.2, 0.6], [2, 2]),
            &dev,
        );
        for _ in 0..100 {
            let t = ternarize_stochastic(w.clone(), &dev);
            let v: Vec<f32> = t.into_data().try_to_vec().unwrap();
            let scale = w.clone().abs().mean().into_scalar::<f32>();
            for x in v {
                assert!(
                    (x - 0.0).abs() < 1e-5 || (x - scale).abs() < 1e-4 || (x + scale).abs() < 1e-4,
                    "not ternary: {x}"
                );
            }
        }
    }
}

/// Inference-only form of a TSCT linear: ternary packs + scales, no FP32
/// masters. `pack_ternary` packs the ternary projection of the masters
/// (2 bits per value); `forward` runs the add-only matmul. Weight memory:
/// `k*(m+n)` bits-ish + `k` f32 scales, e.g. 18.4 KB for a 512x4096 layer
/// at rank 8 (vs 8.4 MB dense FP32).
pub struct SpectralInference {
    u_pack: Vec<u8>,
    v_pack: Vec<u8>,
    s: Vec<f32>,
    #[allow(dead_code)]
    m: usize,
    k: usize,
    n: usize,
}

impl SpectralLinear {
    /// Freeze into the inference form (ternary packs + scales). The FP32
    /// masters are dropped; this object cannot train, only run.
    ///
    /// The trained forward is `y = (x@U_t) * s @ V_t^T` with
    /// `U_t = sign(U)*meanU*keep`; the packed form stores `sign*keep` and
    /// folds `meanU * meanV` into the scales, so the math is identical.
    /// Per-column-trained layers fold the per-column means the same way.
    /// Stochastic-trained layers have no exact ternary pack
    /// (`E[W_t] = sign*min(|w|, scale)` is not ternary) and panic here
    /// instead of silently freezing the wrong weights.
    pub fn to_inference(&self) -> SpectralInference {
        assert!(
            !self.stochastic,
            "to_inference: stochastic layers have no exact ternary pack (E[W_t] = sign*min(|w|, scale)); train with deterministic STE (alpha=1) to export"
        );
        assert!(
            !self.nm_on(),
            "to_inference: N:M-sparse layers have no exact ternary pack (the packed ternary has no mask); export with nm off or extend the pack format"
        );
        let [m, k] = self.u.val().dims();
        let [n, _] = self.v.val().dims();
        let (u_t, mean_u) = if self.per_column {
            (
                ternarize_per_column(self.u.val()),
                self.u
                    .val()
                    .clone()
                    .abs()
                    .mean_dim(0)
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap(),
            )
        } else {
            let mu = self.u.val().clone().abs().mean().into_scalar::<f32>();
            (ternarize(self.u.val()), vec![mu; k])
        };
        let (v_t, mean_v) = if self.per_column {
            (
                ternarize_per_column(self.v.val()),
                self.v
                    .val()
                    .clone()
                    .abs()
                    .mean_dim(0)
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap(),
            )
        } else {
            let mv = self.v.val().clone().abs().mean().into_scalar::<f32>();
            (ternarize(self.v.val()), vec![mv; k])
        };
        let to_ternary = |x: f32| -> i8 {
            if x > 0.0 {
                1
            } else if x < 0.0 {
                -1
            } else {
                0
            }
        };
        let u_host: Vec<i8> = u_t
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(to_ternary)
            .collect();
        let v_host: Vec<i8> = v_t
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(to_ternary)
            .collect();
        let s: Vec<f32> = self
            .s
            .val()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .zip(mean_u.iter().zip(mean_v.iter()))
            .map(|(x, (&mu, &mv))| x * mu * mv)
            .collect();
        // V is consumed as [N, words(K)] by scaled_matmul, so pack the
        // transposed [k, n] host layout
        let mut v_host_t = vec![0i8; k * n];
        for j in 0..n {
            for i in 0..k {
                v_host_t[i * n + j] = v_host[j * k + i];
            }
        }
        SpectralInference {
            u_pack: infer::pack_ternary(&u_host, m, k),
            v_pack: infer::pack_ternary(&v_host_t, k, n),
            s,
            m,
            k,
            n,
        }
    }

    /// Inference bytes (ternary packs + f32 scales), for memory accounting.
    pub fn inference_bytes(&self) -> usize {
        let [m, k] = self.u.val().dims();
        let [n, _] = self.v.val().dims();
        m.div_ceil(4) * k + n.div_ceil(4) * k + k * 4
    }
}

impl SpectralInference {
    /// Add-only forward: `y = (x @ U_t) * s @ V_t^T`, single-threaded.
    /// `x`: `[b, m]` row-major; returns `[b, n]`.
    pub fn forward(&self, x: &[f32], b: usize) -> Vec<f32> {
        let z = infer::packed_matmul(x, &self.u_pack, b, self.m, self.k);
        infer::scaled_matmul(&z, &self.s, &self.v_pack, b, self.k, self.n)
    }

    /// Multithreaded forward (std threads, no deps; the second GEMM is
    /// k-wide, i.e. tiny, so it stays single-threaded).
    pub fn forward_par(&self, x: &[f32], b: usize) -> Vec<f32> {
        let z = infer::packed_matmul_par(x, &self.u_pack, b, self.m, self.k);
        infer::scaled_matmul(&z, &self.s, &self.v_pack, b, self.k, self.n)
    }

    /// Scale of `V`'s pack is `ceil(k/4)`; exposed for accounting/tests.
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.m, self.k, self.n)
    }
}

#[cfg(test)]
mod inference_tests {
    use super::*;
    use burn::tensor::{Device, Distribution};

    #[test]
    fn to_inference_matches_trained_layer() {
        // Autodiff fixture: `retract(5)` below re-tracks the masters and
        // asserts on a non-autodiff device (TEST-AUDIT finding 2); the freeze
        // must match the TRAINED layer, so the retraction has to actually run.
        let dev = Device::ndarray().autodiff();
        let (m, k, n) = (64usize, 8usize, 128usize);
        let mut layer = SpectralLinear::new(m, n, k, &dev);
        // train a few steps-ish: perturb masters, retract, then freeze
        let noise = Tensor::<2>::random([m, k], Distribution::Normal(0.0, 0.1), &dev);
        layer.u = Param::from_tensor(layer.u.val().add(noise));
        layer.retract(5);
        let s_vals: Vec<f32> = (0..k).map(|i| 0.5 + i as f32 * 0.1).collect();
        layer.s = Param::from_tensor(Tensor::<1>::from_data(
            burn::tensor::TensorData::new(s_vals, [k]),
            &dev,
        ));

        // trained forward (ternary STE, alpha=1)
        let x = Tensor::<2>::random([3, m], Distribution::Normal(0.0, 1.0), &dev);
        let trained = layer.forward(x.clone());

        // frozen inference forward (packed, no masters)
        let inf = layer.to_inference();
        let x_host: Vec<f32> = x.into_data().try_to_vec().unwrap();
        let y_host = inf.forward(&x_host, 3);

        let y_t: Vec<f32> = trained.into_data().try_to_vec().unwrap();
        let max_diff = y_host
            .iter()
            .zip(y_t.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let nz = y_host
            .iter()
            .zip(y_t.iter())
            .filter(|(a, b)| (**a - **b).abs() > 1e-3)
            .count();
        println!(
            "max_diff {max_diff}, nz {nz}/{}, y0 {} vs {}",
            y_host.len(),
            y_host[0],
            y_t[0]
        );
        assert!(max_diff < 1e-4, "freeze mismatch: {max_diff}");

        // memory win
        let dense = m * n * 4;
        let packed = layer.inference_bytes();
        assert!(
            packed * 50 < dense,
            "packed {packed} vs dense {dense}: need >50x at d=64"
        );
    }

    #[test]
    fn to_inference_matches_per_column_layer() {
        // Autodiff fixture, same reason as to_inference_matches_trained_layer
        // (TEST-AUDIT finding 2).
        let dev = Device::ndarray().autodiff();
        let (m, k, n) = (64usize, 8usize, 128usize);
        let mut layer = SpectralLinear::new(m, n, k, &dev);
        layer.set_per_column(true);
        let noise = Tensor::<2>::random([m, k], Distribution::Normal(0.0, 0.1), &dev);
        layer.u = Param::from_tensor(layer.u.val().add(noise));
        layer.retract(5);
        let s_vals: Vec<f32> = (0..k).map(|i| 0.5 + i as f32 * 0.1).collect();
        layer.s = Param::from_tensor(Tensor::<1>::from_data(
            burn::tensor::TensorData::new(s_vals, [k]),
            &dev,
        ));

        let x = Tensor::<2>::random([3, m], Distribution::Normal(0.0, 1.0), &dev);
        let trained = layer.forward(x.clone());
        let inf = layer.to_inference();
        let x_host: Vec<f32> = x.into_data().try_to_vec().unwrap();
        let y_host = inf.forward(&x_host, 3);
        let y_t: Vec<f32> = trained.into_data().try_to_vec().unwrap();
        let max_diff = y_host
            .iter()
            .zip(y_t.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("per-column freeze max_diff {max_diff}");
        assert!(max_diff < 1e-4, "per-column freeze mismatch: {max_diff}");
    }

    #[test]
    fn to_inference_panics_on_stochastic() {
        let dev = Device::ndarray();
        let mut layer = SpectralLinear::new(16, 32, 4, &dev);
        layer.set_stochastic(true);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| layer.to_inference()))
                .is_err(),
            "stochastic layers must not silently freeze to deterministic ternary"
        );
    }

    #[test]
    fn to_inference_panics_on_nm() {
        // N:M-sparse layers have no exact ternary pack (the packed ternary
        // carries no mask); freezing must panic, not export wrong weights.
        let dev = Device::ndarray();
        let mut layer = SpectralLinear::new(16, 32, 8, &dev);
        layer.set_nm(6, 8);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| layer.to_inference()))
                .is_err(),
            "N:M-sparse layers must not silently freeze without their mask"
        );
    }
}
