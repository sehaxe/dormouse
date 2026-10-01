# Formula audit of `burn-spectral` — the TSCT retraction

**Date:** 2026-10-01. **Lane:** formula audit, first ever on this crate.
**Worktree:** `wt/spectral-audit` off `8f97411`. **Base commit for every
`file:line` below:** `8f97411` unless stated otherwise.
**Interpreter for every measurement:** `/tmp/opencode/oracle-venv/bin/python`,
python 3.12.14, `torch==2.14.0+cpu`, `numpy==2.5.2`, CPU only, float64 unless a
line says f32. Following the RMSNorm precedent, torch's arithmetic is compiled
C++ and is **not** quotable: the wheel and version are recorded instead.

**What was already verified and is NOT redone here:** the PolarExpress triple
`(15/8, −5/4, 3/8)` is bit-exact vs the authors' own `optimal_quintic` at
`NoahAmsel/PolarExpress @ 71cc379`; the Jordan triple in `burn-muon-plus` is a
different quantity from the same paper and is not an error. Those are
`docs/papers/spectral-reference.md` §1.2/§2.2 and they stand.

**Provenance bar (§1.4).** "verified" is used only with repo + commit + file +
line + date. "transcription" means our own code, re-expressed in torch to be
read against the Rust. `tests/oracle/audit_2026_10_01.py` is a **transcription
of the Rust** (`lib.rs:208-341`) and a **reference** only where it says
`LAPACK` or names a pinned third-party file. It is not evidence about CUDA and
says nothing about f32.

---

## 0. The verdict tally

| # | site | class | verdict |
|---|---|---|---|
| 1 | `polar_orthogonalize` NS update, `lib.rs:253-260` | math | **AGREE** — it is the quintic it claims |
| 2 | basin of `p`, `lib.rs:179-181` | math | **AGREE** on `p'(s) = (15/8)(s²−1)²`, `p(1)=1`, `p(s)>s` on (0,1) |
| 3 | basin SIZE, `lib.rs:2429` | comment | **DISAGREE** — "cubic NS basin < √3" is wrong twice |
| 4 | `POWER_ITERS` justification, `lib.rs:132-135` | comment | **DISAGREE** — reproduced independently, ~4.5% median error, 30–39% of draws over 5% |
| 5 | the 1.05 prescale's stated reason, `lib.rs:249-250` | comment | **DISAGREE** — the factor cannot do what the comment says; the superattracting fixed point does |
| 6 | `polar_orthogonalize_batched` ≡ scalar, `lib.rs:275-317` | math | **AGREE** — same construction, checked site by site |
| 7 | `retract_batched` grouping, `lib.rs:324-341` | math | **AGREE** — shape-grouped, padding-free, no cross-shape inflation |
| 8 | factored U/V retraction keeps `s` as the spectrum | math | **AGREE** — measured, and it is the reason the design is sound |
| 9 | `ortho_error` per-entry, `param.rs:205-217` | math | **AGREE** — the normalization is the one the latch wants |
| 10 | `retraction_puts_sigma_max_at_one`, `lib.rs:1458-1475` | test | **WEAK GATE** — measures the RMS singular value and names it σ_max |
| 11 | `retraction_holds_the_manifold_at_rank_64`, `lib.rs:1388-1427` | test | **WEAK GATE** — starts on the manifold, where 0 and 3 iterations agree |
| 12 | `LinearLike` ×4 pad + slice, `param.rs:46-58, 128-135` | math | **AGREE** — slice present on every path |
| 13 | retraction hooks, `model.rs:420-473`, `lib.rs:662-669` | wiring | **AGREE** — ids/mappers and `require_grad` mirrored on both arms |
| 14 | `SpectralMoE` / `infer.rs` / `gpu.rs` / `bf16_ops.rs` | math | NOT AUDITED — named as a gap, not as agreement |

---

## 1. The Newton–Schulz iteration (`lib.rs:208-266`)

### 1.1 The update IS the quintic — VERIFIED, class-A clean

`lib.rs:253` says `x ← a·x + (b·XXᵀ + c·(XXᵀ)²)·x` with "optimal NS
coefficients". Two algebraically different forms of the quintic exist and they
are easy to confuse, so this was checked rather than read:

| form | expression |
|---|---|
| code (`lib.rs:256-259`) | `G = M Mᵀ;  M ← aM + (bG + cG²)M` |
| polar form | `S = MᵀM;  M ← aM + b(MS) + c(M(S²))` |

`ours_gram_iter` and `quintic_polar_form` in the audit script, f64, both from
the same prescaled input, agree to **1.7e-16 … 6.1e-16** max-abs on
`[64,8] [768,64] [8,64] [256,256]`. The code's form is the quintic. Not a
lookalike.

The `X·Xᵀ` choice is also correct for the shape it is applied to. The code
transposes to put the **small side first** (`lib.rs:211-215`) and forms
`M Mᵀ` on the small side, which is the `Q = X(XᵀX)^{-1/2}` convention — the
fixed point has orthonormal **rows** in the canonical form, hence orthonormal
**columns** in the caller's `[rows, cols]`. The test at `lib.rs:1701-1705`
knows this and checks the transpose for a wide factor. Correct.

### 1.2 The polynomial's own claims — VERIFIED exactly

`lib.rs:179-181` asserts `p′(s) = 1.875(s²−1)² ≥ 0`, `p(1) = 1`, and
`p(s) − s > 0` on `(0,1)`. All three re-derived and measured:

| claim | measured (f64) |
|---|---|
| `p(1)` | `1.0000000000000000` |
| `p′(1)` | `0.000e+00` |
| `p′(s) = (15/8)(s²−1)²` at s=0.7 | `4.876875e-01` vs `4.876875e-01` |
| `p(s) − s = s(3s²−7)(s²−1)/8` | matches to the last digit at s = 0.1, 0.5, 0.9524, 1, 1.2, 1.5, 1.6, 2 |

The factorisation `p(s) − s = (s/8)(3s²−7)(s²−1)` is the whole basin story and
it is what §1.3 uses. **This comment block is honest and stays.**

---

## 2. FINDING 1 — the basin is √(7/3), and `lib.rs:2429` says √3 (class A)

`lib.rs:2427-2436` (`polar_square_and_tall_no_divergence`) says:

> square random matrices had sigma_max ≈ 2 after the old Frobenius/sqrt(k)
> pre-scale (**above the cubic NS basin < sqrt(3)**), so NS diverged

Two errors in one parenthetical.

1. **The quintic we run is not a cubic.** The iteration at `lib.rs:254` is
   `a·x + b·x³ + c·x⁵`; there is no cubic anywhere in the current code. If the
   comment means the *historical* cubic `p₃(s) = 1.5s − 0.5s³`, that
   polynomial's basin is `|s| < 1` **exactly** — `p₃(s) = s` has roots
   `s ∈ {0, ±1}` — so "basin < √3" is wrong for that too.
2. **The quintic's basin is `√(7/3) = 1.5275`**, from the repelling fixed point:
   `p(s) = s` gives `s(3s⁴ − 10s² + 7)/8 = 0`, roots `0, ±1, ±√(7/3)`, and
   `p′(√(7/3)) = 3.3333 > 1` so that root repels. Measured, 40 iterations from
   a scalar start:

   | start | after 40 iters | |
   |---|---|---|
   | 1.0000 | 1.0000e+00 | converges |
   | 1.2000 | 1.0000e+00 | converges |
   | 1.4500 | 1.0000e+00 | converges |
   | 1.5200 | 1.0000e+00 | converges |
   | 1.5275 | 1.0000e+00 | converges |
   | **1.5300** | **2.5114e+14** | **DIVERGES** |
   | 1.6000 | 1.1253e+48 | diverges |
   | 2.0000 | 1.6449e+16 | diverges |

**Why it matters and why it is still class A.** The regression the comment
guards is real and the test is worth keeping; the number in it is wrong, and a
reader who trusts "basin < √3 = 1.73" will conclude a prescale that lands at
1.6 is safe when it is not. The correct figure is *narrower* than the one
printed, so the comment is wrong in the dangerous direction. Fixed as a
comment, with the analytic derivation in place of the assertion.

---

## 3. FINDING 2 — `POWER_ITERS`'s stated justification is false (class A)

`lib.rs:132-135`, verbatim:

> Power-iteration count for the sigma_max estimate (Gram top eigenvalue).
> Rayleigh-quotient error shrinks as `(λ2/λ1)^k`; 5 gives the estimate well
> within the 1.05 safety factor even for square Wishart `(λ2/λ1 ≈ 1)`.

**Reproduced independently** (the prior lane measured this; this run does not
rely on it — 400 fresh draws per shape, f64, start vector `G·1` as the code
builds it):

| shape | median rel err | p95 | max | **frac > 5%** | max prescaled σ_max |
|---|---|---|---|---|---|
| (768, 64) | −4.503e-02 | 7.22e-02 | 9.44e-02 | **0.370** | 1.0516 |
| (4096, 64) | −4.496e-02 | 6.24e-02 | 7.48e-02 | **0.302** | 1.0294 |
| (512, 128) | −4.573e-02 | 7.63e-02 | 1.02e-01 | **0.390** | 1.0603 |
| (256, 256) | −4.439e-02 | 7.68e-02 | 1.07e-01 | **0.385** | 1.0659 |
| (128, 128) | −4.191e-02 | 8.00e-02 | 1.04e-01 | **0.335** | 1.0630 |
| (64, 64) | −3.117e-02 | 8.81e-02 | 1.75e-01 | **0.300** | 1.1539 |

The comment fails on **two** counts, and the second is the one that matters:

- **The rate is wrong.** It decays as `(λ2/λ1)^(2k)`, not `^(k)`. Measured on a
  *prescribed* spectrum (a Wishart average cannot separate the rate from the
  spread, because near-degenerate draws are exactly the bad ones):

  | λ2/λ1 | median rel err after 5 steps | `(ratio)^5` | `(ratio)^10` |
  |---|---|---|---|
  | 0.10 | 2.220e-16 | 1.000e-05 | 1.000e-10 |
  | 0.50 | 2.082e-08 | 3.125e-02 | 9.766e-04 |
  | 0.90 | 7.026e-03 | 5.905e-01 | 3.487e-01 |
  | 0.99 | 4.334e-03 | 9.510e-01 | 9.044e-01 |

  The error tracks the `^(2k)` column.
- **"well within the 1.05 safety factor" is false on 30–39% of draws.** The
  estimate converges from **below** (every median is negative), so it is a
  lower bound, so dividing by it puts the **true** σ_max at
  `σ_true/(σ_est · 1.05) ≥ 1/1.05` — the last column, which **exceeds 1.0** on
  the worst draws. `1.05 > 1` moves the input *down*; an underestimate moves it
  *up*. They work against each other, and on a third of draws the error wins.

**The estimate is not catastrophic, and that is the honest safety story.** Over
2400 fresh draws the worst prescaled σ_max is 1.376× inside the basin edge
`√(7/3) = 1.5275`. The margin is real, and it comes from `p′(1) = 0`: the fixed
point is **superattracting**, so an input anywhere in the basin lands on 1
regardless. That is the reason to keep the prescale, and it is not the reason
the comment gives.

**Verdict: class A, comment only, no numeric change proposed.** The prior
research doc reached the same conclusion from its own 200 draws; this run
reproduces it from 400 fresh ones and adds the `^(2k)` measurement the doc did
not have.

---

## 4. FINDING 3 — the 1.05 prescale comment states the wrong mechanism (class A)

`lib.rs:249-251`, verbatim:

> `// 1.05 safety factor: power iteration converges from below, so the true`
> `// sigma_max stays strictly inside the basin (Q is scale-invariant).`

Three clauses, and the load-bearing one is backwards:

- "converges from below" — **true** (median rel err is negative in all six
  shapes).
- "so the true sigma_max stays strictly inside the basin" — **false**, and it
  is false *because* the previous clause is true. Converging from below means
  the estimate is a **lower** bound, so `σ_true/(σ_est·1.05)` is **≥ 1/1.05 =
  0.9524 and unbounded above**. The direction of the error is the one that
  pushes the input **out** of `[0,1]`. A lower bound is a reason to be careful,
  not a reason to be safe.
- "the basin" — per §2 the basin is `[0, √(7/3))`, not `[0,1]`, so "inside the
  basin" is doing less work than the sentence implies.

What is true and sufficient: `p′(1) = 0`, so every input in the basin converges
to 1 superattractingly; measured `σ_max(out) = 1.000000000` on all 2400 draws
and on every row of the table in §3. The 1.05 factor is a **small downward
nudge on the largest singular value**, useful because the estimate is a lower
bound and the nudge buys margin against the error — it is not a guarantee.

This is a doc/code disagreement of the §1.7 class and it is the comment most
likely to be read by the next person who considers raising `POWER_ITERS` or
replacing the prescale with a Frobenius divide. **Class A, comment only.**

---

## 5. The low-rank factor path — VERIFIED, and it is why the design is sound

The brief asks how U/V are stored and retracted in the factored
`[d,64]×[64,f]` form. `SpectralLinear` (`lib.rs:391-451`) holds
`u: [in, k]`, `s: [k]`, `v: [out, k]`, and the forward is
`y = (x @ U_t) * s @ V_tᵀ` (`lib.rs:579-581`, the "paper order").

**The retraction targets U and V separately** (`lib.rs:662-669`), each onto
`UᵀU = I` / `VᵀV = I`. That is only sound if the retracted set is exactly the
set on which `W = U diag(s) Vᵀ` **is an SVD with singular values `s`**. It is,
and it is worth stating because nothing in the tree says it:

> if `UᵀU = I` and `VᵀV = I` then `Uᵀ W V = diag(s)`, so `W` has singular
> values exactly `s` and no other.

Measured (f64, `tests/oracle/sec5_6b.py`): `max|offdiag(Uᵀ W V)|` is
**1.1e-16 … 2.2e-16** at factor spreads 1, 10 and 1000, with
`max rel diag error ≤ 2.0e-14`. So the constraint the retraction enforces is
precisely the SVD condition, `s` survives as the model's singular-value
budget, and **no external reference is needed or claimed** — this is our
design and the verification is internal to the algebra.

**What the retraction does to the function** (measured, same script), because
it is the thing the cadence A/B is really about:

| drift added to an on-manifold factor, then one `retract(3)` | rel ‖W_off − W_on‖_F | per-entry ortho AFTER the retraction |
|---|---|---|
| 1e-4 | 2.13e-03 | 4.6e-17 |
| 1e-3 | 2.13e-02 | 5.2e-17 |
| 1e-2 | 2.11e-01 | 8.5e-17 |
| 1e-1 | 1.20e+00 | 9.9e-05 |

A retraction **is allowed** to move `W` — that is what a projection is — and
the quantity that must be a fixed point is the **factors'**, which is what
`retraction_holds_the_manifold_at_rank_64` tests. So this is not a defect.
The number that matters for §9: **the retraction is not a small correction at
the drift scale the latch tolerates.** At a relative drift of 1e-2 it moves
`W` by 21%. A step that skips the retraction leaves that much un-projected
weight in the model.

### 5.1 `ortho_error` and the per-entry metric — AGREE

`param.rs:205-217` computes `max(‖UᵀU−I‖_F/k_u, ‖VᵀV−I‖_F/k_v)`. Since
`‖·‖_F / k` on a `k×k` Gram is **exactly the RMS of the `k²` entries** — not an
approximation — the metric is the natural per-entry error and is `k`-free, so
a single threshold is meaningful across ranks. The comment's reasoning
(`param.rs:197-204`: the raw F-norm sums `k²` entries, so 1e-3 sat *below* the
retract's own floor and the fallback fired on every fresh run) is a correct
diagnosis of the 2026-09-04 bug. The normalization is right.

`ortho_error` itself (`lib.rs:344-360`) is a correct Frobenius norm of the
Gram residual. It reads `into_data().bytes` and chunks by 4, which is correct
for the f32 masters it is called on and would be garbage on a bf16 tensor;
the only caller is `param.rs`, on masters, which are always f32 (§2.3 of
AGENTS.md). A host read at a 500-step cadence, which §1.3 permits.

---

## 6. FINDING 4 — the σ_max estimate has an unguarded failure mode (CLASS B)

**This is the audit's substantive new finding and it was not in the prior
research document.** The prior lane recorded, correctly, that *no reference
implementation does a power iteration* and left it "recorded, not acted on"
(`spectral-reference.md` §3.4). It measured Wishart draws, where the start
vector has a healthy component along the dominant direction. This audit
constructed the case the start vector cannot see.

**The mechanism, from the code.** `lib.rs:230` initialises the power iteration
with `v = G·1` — the **row sums of the Gram**. A factor whose dominant
singular direction is nearly orthogonal to the all-ones direction is therefore
**invisible** to the start vector. In closed form: with a start
`θ·e₁ + c·e₂` and Gram eigenvalues `λ₁ > λ₂`, `POWER_ITERS` steps give
`θ·e₁ + (λ₂/λ₁)^k c·e₂`, so the Rayleigh quotient tends to **λ₂/λ₁ — the
wrong eigenvalue** — whenever `δ < (λ₂/λ₁)^k`. The prescale then divides by
`σ₂` instead of `σ₁` and lands at `σ₁/(σ₂·1.05)`, outside the basin edge
`√(7/3) = 1.5275` as soon as `λ₂/λ₁ < 1/(1.05·√(7/3)) = 0.6235`.

**Measured** (`tests/oracle/sec4_fixed.py`, f64, top eigenvector constructed
orthogonal to `1` to ~1e-16):

| λ₂/λ₁ | est/σ₁ | prescaled | outside basin? | `max|3-iter output|` |
|---|---|---|---|---|
| 0.99 | 0.994987 | 0.9572 | no | 9.2e-01 |
| 0.90 | 0.948683 | 1.0039 | no | 9.4e-01 |
| 0.70 | 0.836660 | 1.1383 | no | 9.7e-01 |
| 0.50 | 0.707107 | 1.3469 | no | 9.9e-01 |
| **0.30** | **0.547678** | **1.7389** | **YES** | **2.0e+06** |
| **0.10** | **0.315653** | **3.0172** | **YES** | **1.5e+42** |
| **0.01** | **0.223607** | **4.2592** | **YES** | **5.8e+62** |

**In the f32 the trainer runs, `1e+42` is not a large number — it is `inf`**
(f32 max is 3.4e+38). Measured directly (`sec4c_tolerance.py`): f32 output is
`inf` on every constructed case, and finite at a 1e-4 overlap. A diverged
master is what the *next forward* reads: the retraction runs **after** the
optimizer step (`train/src/lib.rs:1325-1336`).

**Reachability: UNREACHED, and that is stated as firmly as the divergence
itself.** Not one of 2400 random Wishart draws lands outside the basin (worst
prescaled `σ_max` **1.154** against an edge of 1.5275), and not one of the
isotropic-drift draws. The condition is a near-exact alignment of the
dominant direction with the all-ones direction's orthogonal complement. So:
**a latent hole, not a live bug, filed as one.**

**In f32 the same fixture is measured, not extrapolated.** The landed gate
builds the factor in f32 and reads the estimate at **0.2495** against a true
`σ_max` of 1.0, so the prescaled input is **3.8172** against a basin edge of
**1.5275**. What the 3-iteration NS iteration does with an input at 3.82 is
the `inf` of the table above: measured f32 output is non-finite on every
constructed case and finite at a 1e-4 overlap.

**The obvious guard is WRONG, and that is the most useful part of the
finding.** Cauchy–Schwarz (`σ_max ≤ ‖X‖_F`) suggests
`sigma_used = max(est, ‖X‖_F/1.604)`. It **does** fix every constructed case
(f64 `3.2e+60 → 8.1e-1`, `tests/oracle/sec4d_guard.py`). But on the shape the
trainer actually retracts — a `[768,64]` factor, `‖X‖_F = 8` on the manifold,
`σ_max = 1` — `‖X‖_F/1.604 = 4.99 > 1`, so the `max` would select the
Frobenius bound **always** and silently turn the σ_max prescale into the
Frobenius one this repo measured as 0.68 instead of 1.0 (§7). A correct guard
has to detect the **degenerate start** (`‖G·1‖` small relative to `‖G‖_F`),
not bound the estimate. That is a design question and it is the owner's.

**Carried as:** `sigma_max_estimate_diverges`, `#[ignore]`d with this reason,
red when run.

**A harness mistake of my own, disclosed because it nearly hid the finding:**
the first construction used `M = cholesky(G)ᵀ`, but `cholesky` returns `R`
with `G = RᵀR`, so `M Mᵀ = R Rᵀ ≠ G` — I was reasoning about one Gram and
measuring another, and the result was a comfortable "0/4000 outside the
basin". `sec4_fixed.py` asserts the reconstruction and uses `L`. This is the
same failure `spectral-reference.md` §3.3 discloses, in the same week, in the
same lane: **a corrected run that silently replaces a buggy one is how a wrong
number survives.**

---

## 7. FINDING 5 — `retract_iters = 3` crosses the one-way latch at ~3:1 (CLASS B)

This is the finding that has an operational consequence tonight, so it is
stated first in this section.

The trainer retracts with `retract_iters: 3` (`train/src/lib.rs:175`, the
default) and monitors per-entry `‖UᵀU−I‖_F/k` against a **1e-3** threshold
every 500 steps; above it, **all** factors switch to fp32 **irreversibly** and
the latch is persisted in the checkpoint (`lib.rs:568`, `:650`, `:1350-1357`).
So the quantity that decides whether the factor-quant forward is alive for the
rest of a run is **the residual the retraction itself leaves**.

Measured against LAPACK's `polar` (an (a) oracle — the *definition* of the
target — on a `[768,64]` factor with a prescribed spectrum;
`tests/oracle/sec5_6b.py`, f64):

| σ_min/σ_max | iters=3 | × latch | iters=4 | iters=5 | iters=6 | iters=8 |
|---|---|---|---|---|---|---|
| 1.00 | 3.2e-17 | 0.0 | 2.6e-17 | 2.6e-17 | 2.4e-17 | 2.2e-17 |
| 0.90 | 3.4e-17 | 0.0 | 2.4e-17 | 2.6e-17 | 2.5e-17 | 2.3e-17 |
| 0.70 | 1.0e-11 | 0.0 | 3.2e-17 | 2.6e-17 | 2.7e-17 | 2.6e-17 |
| 0.50 | 1.1e-06 | 0.0 | 1.0e-15 | 2.6e-17 | 2.4e-17 | 2.1e-17 |
| 0.30 | 8.3e-04 | 0.8 | 2.6e-07 | 3.7e-17 | 2.7e-17 | 2.5e-17 |
| **0.20** | **7.2e-03** | **7.2** | 1.2e-04 | 1.5e-09 | 2.5e-17 | 2.3e-17 |
| **0.10** | **3.3e-02** | **32.6** | 8.1e-03 | 2.3e-04 | 1.1e-08 | 2.5e-17 |
| **0.05** | **5.7e-02** | **56.6** | 3.2e-02 | 9.0e-03 | 3.8e-04 | 2.8e-17 |
| **0.01** | **8.5e-02** | **85.0** | 7.2e-02 | 5.6e-02 | 3.8e-02 | 2.8e-03 |

(per-entry `‖UᵀU−I‖_F/k`; latch 1e-3. The prior research document's Finding A
reported the 10:1 row as "14.5% away from the polar factor" — my `relF` column
at the same point reads 1.5e-01, consistent.)

**Read:** at the default 3 iterations the retraction's own residual stays under
the latch only while the factor's spectrum is within about **3:1**. Past that
it is 7× to 85× over, and the one-way fallback fires and persists. The
iteration count is the knob that decides it: 5 survives 10:1, 6 survives 20:1,
8 survives 100:1.

**The existing green test cannot see this.**
`retraction_holds_the_manifold_at_rank_64` (`lib.rs:1388`) starts *on* the
manifold, where 0 and 3 iterations are indistinguishable and where
`retraction_puts_sigma_max_at_one` is likewise satisfied by any iteration
count. This is the §1.2 "A/B or death" shape applied to a test: a gate that
cannot fail on the input that matters.

**What is NOT claimed:** that the latch has fired in any run on record. No run
has instrumented the factor's spectrum, and `ortho_fp32` is persisted so a
fired latch *is* visible in a checkpoint's flags. **What is claimed:** the
default of 3 is unexplained by anything in the tree, it is the knob that
decides whether a one-way permanent fallback engages, and no test measures it.

**Carried as:** `retraction_error_grows_with_spectral_spread`, `#[ignore]`d
with this reason, red when run.

### 7.1 A documented number this audit could not reproduce

`param.rs:200-203` and AGENTS.md §2.3 both state the retract's own floor as
"**~4e-3 raw (~6e-5 per-entry at r=64)**", attributed to a 2026-09-04 GPU
probe named `ortho_probe`. My table has **no row** at 6e-5 per-entry for 3
iterations: the nearest are 1.1e-06 (spread 0.5) and 8.3e-04 (spread 0.3), and
the 4e-3 raw figure is 500× above the f32 arithmetic floor (≈1.2e-7 per entry
at rank 64). So the documented floor is either from a different input
population than any I constructed, or from a different normalization. **I did
not re-run their probe and I am not retracting their number** — I am recording
that my measurement does not reproduce it and that the two are not obviously
about the same quantity, because the honest reading of that comment is "a
constant floor" and my measurement says the residual **spans nine orders of
magnitude** with the input's spectral spread. Named, not acted on.

---

## 8. `LinearLike`, the retraction arms, and the trainer hooks — AGREE

**`LinearLike` padding** (`param.rs:46-58`). `out_features` is padded up to a
multiple of 4 for cubek vectorization, with `out_features == 1` exempted, and
the slice-back is at `param.rs:128-135` on **every** arm — the `bf16_compute`
path, the `forward_quant` path, the plain `forward` path and the `Dense`
variant all pass through the same `let y = match … ;` and the same slice. The
comment "extra columns are never read downstream and would be the float4 tail"
is the correct reason and the slice is unconditional on `y.dims()[1] !=
out_features`, so a future arm that forgets to pad cannot leak a short row.
`dense()` (`param.rs:66-76`) duplicates the padding expression; that is a
copy rather than a helper, and a padding change made in one place would not
reach the other. **Follow-up, not a defect** (one expression, two sites, both
tested by the same forward tests).

**The two retraction arms are algebraically identical — VERIFIED by reading,
site by site, not only by the test.** `polar_orthogonalize_batched`
(`lib.rs:275-317`) against `polar_orthogonalize` (`lib.rs:208-266`):

| step | scalar | batched | same? |
|---|---|---|---|
| canonical transpose | `rows > cols` → swap 0,1 | `rows > cols` → swap 1,2 | yes, and uniform per group because `retract_batched` groups by **exact** shape |
| Gram | `m mᵀ` `[c,c]` | `m mᵀ` `[B,c,c]` | yes |
| start vector | `g.sum_dim(1).squeeze(1)` = `G·1` | `g.sum_dim(2)` = `G·1` per slice | yes |
| normalise | `(v·v).sum_dim(0)` `[1]` | `(v·v).sum_dim(1).sum_dim(2)` `[B,1,1]` | yes |
| Rayleigh | `v·gv` and `v·v` summed to `[1,1]` | both to `[B,1,1]` | yes |
| prescale | `/ (sigma·1.05)` | `/ (sigma·1.05)` | yes |
| NS loop | `a·m + (bG + cG²)·m` | identical | yes |

So **tonight's arms differ in CADENCE, not MATH** — confirmed, and the claim
holds for the reason it should: the grouping is by exact shape, so the
per-slice canonical-transpose decision is the same for every member of a group.

Two things about the *wiring* that are right and worth recording because they
are invisible when wrong:

- `retract_batched` (`lib.rs:324-341`) stacks, runs one batched call, and
  slices back — **padding-free by construction**, because mixed shapes are
  never stacked. A `[512,k]` expert master is not inflated to lm_head's
  `[vocab,k]`.
- `model.rs:442-473` hands every master back through
  `Param::from_mapped_value` with **its own id and mapper** (fresh ids would
  silently reset every factor's momentum, since the optimizer's records are
  keyed by id) and **mirrors** `require_grad` rather than forcing it. The
  scalar arm does the same through `polar_retracked` (`lib.rs:154-162`). The
  batched arm is **counted** (`probe::RETRACT_BATCHED`, `model.rs:443`)
  because it and the default arm produce the same numbers and a silent
  fallback there is "a run that is correct and 22% slower" — the ADR-0011
  shape, handled.

**One honest limit on the identity claim.** The "same numbers" assertion is
verified on `Device::ndarray()` only, by
`retract_batched_identity_with_per_factor_path` at **1e-5 absolute**
(`lib.rs:1667-1708`). The trainer's backend is CUDA, where a batched matmul
and a per-slice matmul reduce in different orders, so 1e-5 is the right
cross-backend choice — but the claim is verified on the CPU proxy only and
should be quoted that way. **Follow-up: a CUDA parity cell for this pair.**

---

## 9. The retraction-cost question, sharpened

**The measured cost** (AGENTS.md §3.1, 2026-09-29, this box, warm steps,
`--timers`): `retr = 52.8 / 53.3 / 64.6 ms` at batch 8 / 16 / 32 against step
times of 244 / 440 / 826 ms. `--retract-every 1000` gives `retr = 0.0` and a
188 ms step vs 240. So the retraction is **22% of a step at batch 8 and 7.8% at
batch 32**, and it is a **fixed** cost — 8× the data buys 1.2× the retraction,
because it is per-parameter Newton-Schulz over every TSCT factor and does not
amortise. In the launch-bound regime this box is in (GPU 13.3% utilised), a
fixed per-step cost is the *only* kind worth attacking.

**The quality datapoint exists and it is fresh:** tonight's retract-4 arm ran
**best of three** (6.329 vs 6.387 / 6.437) at 2k steps. One seed each, so it
is a hint, not a result — but it is the first time cadence and quality have
been measured in the same breath, and it points the same way as §5's table
(the retraction is not a small correction at the drift the latch tolerates).

**What the A/B should actually measure — three questions, not one:**

1. **The drift, not the BPB, is the primary quantity.** §5's table says a
   skipped retraction leaves the model running an **un-projected** `W`:
   a relative factor drift of 1e-2 (well inside anything the latch tolerates
   between checks) is a **21%** change in `W` that the retraction would have
   removed. So the first thing to log is `‖UᵀU − I‖_F/k` at **every** step
   for a 200-step window, at cadence 1 and cadence 4, on the same seed. If the
   drift at cadence 4 crosses 1e-3 within the 500-step check interval, the
   latch is the binding constraint and cadence is not free. **This is
   measurable in one short run and it decides the question.** Log it; do not
   infer it from BPB.
2. **The retraction is a projection, so the question "is retraction every step
   or every 4th" is really "how much un-projected drift does the optimizer's
   own step introduce, and does the model's quality depend on it".** The
   optimizer step between retractions is the drift source and it is not
   measured anywhere. `--retract-every 4` at 3 seeds × 2k steps is the
   quality arm; question 1 is the cheap arm that decides whether it is worth
   running at all.
3. **`retract_iters` is the unexamined knob and it is cheaper than cadence.**
   Per §7, 3 → 5 iterations costs ~1.7× the NS loop (≈ +15% of a step, from
   the 22% base) and buys a factor whose retraction residual survives a 10:1
   spectrum instead of 3:1 — i.e. it moves the latch threshold by more than
   cadence does, and it moves it in the *safe* direction. **If the latch is
   what the cadence A/B is really about, `retract_iters=5` is the smaller and
   better-aimed change**, and it is one flag.

**Not proposed, and why:** removing the retraction (the strongest form of the
A/B) is not on the table, because the retraction is what makes the
parameterization an SVD at all (§5) and what the ternary/fp8 forward's
orthonormality assumption rests on. The honest control is cadence 1 vs 4 with
the drift logged, plus `retract_iters` 3 vs 5 as a third arm.

**One cost number that is still missing and is cheap:** the batched arm
(`--retract-batched`, flag-off) is the only one of the three that removes host
syncs — 7 per factor, 112 per step at `small` (that count is the
`scalar_retraction_makes_no_host_read` gate, `lib.rs:1507`). The ndarray
microbench at `lib.rs:1736` is honest about being a compute/overhead proxy
with no syncs to count. **Nobody has measured what the batched arm does to
`retr = 52.8 ms` on CUDA.** That is a 20-minute run and it attacks the
launch-bound cost directly, so it should be measured before any cadence sweep
buys back a fifth of a step.

---

## 10. Class-B register — the owner's decisions

| id | finding | carried as | the decision |
|---|---|---|---|
| **B-1** | the 3-iteration retraction's residual is spread-dependent and crosses the **one-way, checkpoint-persisted** 1e-3 `max_ortho` latch at ~3:1 spectral spread | `retraction_error_grows_with_spectral_spread` (`#[ignore]`) | raise `retract_iters`, lower the latch, or instrument the spread. **One flag, one short run to decide.** |
| **B-2** | the σ_max power iteration starts from `G·1`; a factor whose dominant direction is near-orthogonal to `1` is invisible to it, the prescale divides by the wrong eigenvalue, and the retraction returns **`inf` in f32** | `sigma_max_estimate_diverges` (`#[ignore]`) | add a degenerate-start guard (the Cauchy–Schwarz `max` is **wrong** — it degenerates to the Frobenius prescale on `[768,64]`), or accept the hole on the record. UNREACHED in 2400 draws. |
| **B-3** | the documented retract floor "~4e-3 raw / ~6e-5 per-entry at r=64" (`param.rs:200-203`, AGENTS.md §2.3) is not reproduced by any row of the §7 table, and the true residual spans nine orders of magnitude with the input's spread | §7.1 of this document | re-run `ortho_probe` on an input whose spectrum is known, or retire the constant. Not retracted by me — not my probe. |
| **B-4** | the "same numbers" claim for the batched arm is verified on ndarray only, at 1e-5 | §8 | a CUDA parity cell, or quote the claim as CPU-verified. |
| **B-5** | the padding expression is duplicated between `LinearLike::new` and `LinearLike::dense` | §8 | follow-up; one expression, two sites, both covered by the forward tests. |

**Two findings that are explicitly NOT class B, because they are the
designed behaviour and naming them as defects would be wrong:**

- the retraction moving `W` (§5) — a projection is supposed to;
- `retraction_puts_sigma_max_at_one` (`lib.rs:1458`) computing
  `sqrt(trace(UᵀU)/k)`, which is the **RMS singular value**, and naming the
  variable `sigma_max`. In the two cases the test actually exercises (on the
  manifold, and Frobenius-prescaled) all singular values are equal, so RMS =
  σ_max and the assertion is unaffected. It is a naming inaccuracy in a test
  variable and its message, and it would mislead a reader who reused the
  pattern on a spread factor — where RMS ≠ σ_max by exactly the spread. Left
  as a naming note rather than a rewrite, because changing the test's variable
  names changes nothing it proves.

---

## 11. The gates this audit landed, and how to see them fail

| gate | pins | green? |
|---|---|---|
| `the_quintic_basin_is_sqrt_7_over_3` | `p(1)=1`, `p'(1)=0`, `p'(s)=(15/8)(s²−1)²`, the edge `√(7/3)`, `p'(edge)=10/3>1` repels, scalar 1.52 converges and 1.53 diverges — **the number that replaced "√3"** | green |
| `the_sigma_estimate_is_a_lower_bound_and_the_1_05_factor_is_not_what_saves_it` | on the manifold the estimate is exact and the prescaled input is under 1 (**the case the old comment was written about, and it is TRUE there — which is why the bug survived**); on a near-flat two-level spectrum (`s = 1, 0.9 x 63`) the estimate is **8.3% low** and the prescaled input is **1.0382, above 1.0** (**the refuted claim, now falsifiable in-tree**); and the retracted σ_max is 1 regardless | green |
| `retraction_error_grows_with_spectral_spread` | B-1 | **red on purpose** (`#[ignore]`) |
| `sigma_max_estimate_diverges` | B-2 | **red on purpose** (`#[ignore]`) |

### 11.1 Three of the audit's own gates were measuring the wrong matrix

Disclosed in full because it is the exact failure this document exists to
prevent, it happened **three times in four gates**, and a numpy dry-run
(`tests/oracle/gate_dryrun.py`, written for the mundane reason that a cold
build slot costs 20 minutes of queue) caught all of them before cargo did.

1. **B-2 passed the GRAM where the code estimates on a FACTOR.** The estimate
   is `sqrt(λ₁(M Mᵀ))`, so the input must satisfy `M Mᵀ = G`. The first version
   passed `G` itself, which forms `G·G` on the way in — whose top eigenvalue is
   1.0 — so the estimate came back **correct** and the gate was a decoration
   sitting on top of the finding. Fixed by building
   `M = u1u1ᵀ + √λ_rest·(I − u1u1ᵀ)`, which satisfies `M Mᵀ = G` exactly
   because the two projectors are orthogonal idempotents, and by **asserting**
   `max|M Mᵀ − G| < 1e-5` rather than assuming it.
2. **B-1 built its "orthonormal basis" by retracting a raw fixture**, which is
   not orthonormal at rank 64: measured `max|QᵀQ − I| = 0.94` and a
   singular-value ratio of **548:1 before any spread was applied**. All three
   spreads printed the same number (121×). Replaced with the DCT fixture, with
   the orthonormality asserted so it cannot come back silently.
3. **The class-A estimate gate's off-manifold fixture was log-spaced**, where
   the estimate is only **1.4%** low — precisely the case that would *not*
   have refuted the comment, i.e. a cherry-picked fixture. A **near-flat**
   spectrum is the honest worst case, because a flat Gram is exactly where
   `(λ₂/λ₁)^k` decays slowest. **Both** candidate fixtures were then measured
   (f32, `s[0] = 1` in each, so the factor's true σ_max is exactly 1):

   | fixture | est/σ₁ | prescaled | can it refute? |
   |---|---|---|---|
   | log-spaced 1 → 0.01 | 0.9861 | 0.9658 | **no** — the input stays under 1 |
   | every `s = 0.9` | 0.9000 | 1.0582 | **no, and for a worse reason** (below) |
   | `s[0] = 1`, rest 0.9 | 0.9174 | **1.0382** | **yes** — this is the landed fixture |

   The middle row is why the landed fixture is two-level and not "all 0.9".
   An all-0.9 factor is a scalar multiple of an orthonormal one, its Gram is
   exactly `0.81·I`, and the estimate is therefore **exact whatever σ_max is**
   — it read 0.900000000 on the nose. It produces the most dramatic number in
   the table and **cannot fail**: a fixture that cannot fail is a fixture to
   delete. The log-spaced case is kept as a **recorded counter-example in the
   test's own comment**, with its number, so the choice is visible rather than
   convenient — and the landed fixture asserts `est < true_smax` explicitly, so
   a future "improvement" to a fixture that stops under-estimating trips the
   gate instead of quietly making it unrefutable.

And the dry-run itself had to learn the canonical transpose (`lib.rs:211-215`)
before it agreed with the Rust. Without it the dry-run formed the `[768,768]`
Gram instead of the `[64,64]` one and reproduced — by accident, and
convincingly — **exactly the Frobenius-prescale failure this crate spent an
audit disproving** (σ_max 0.68 instead of 1.0). A harness that disagrees with
the code in the direction of a known historical bug is the most dangerous kind
of harness, and it is invisible unless the disagreement is looked for.

`tests/oracle/falsify.sh` perturbs each green gate and reverts:

- **A** — the quintic's `c` coefficient by 1e-3, at its **one** definition
  (`NS_C`), which the scalar arm, the batched arm and the basin gate all read.
  Expected red on the basin gate *and* on the manifold gate.
- **B** — the 1.05 prescale factor → 1.00 in the **sync-free arm only** (2
  sites, asserted), leaving the host-read fixture at 1.05. Expected **red**:
  `sync_free_retraction_is_bit_identical_to_the_host_read_one` compares the two
  elementwise, and one arm moving is the mutation it exists to catch.
  *This bullet was wrong twice and both corrections are in §13.4.* The first
  version predicted **green** ("the factor is only a nudge, so nothing should
  notice") — which is a property of the ALGORITHM, not of this gate, and it
  predicted green of a gate that had never been asked about the factor. The
  second version perturbed **both** arms and therefore could not fail at all.
- **C** — the class-A gate rewritten to assert the **refuted** claim.
  Expected **red**: a gate that cannot fail on the thing it exists to deny is
  not a gate.
- **D** — both class-B tests run explicitly with `--ignored --nocapture`.
  Expected **red with their stated magnitudes**; an `#[ignore]`d test that would
  pass is a decoration, and so is one that goes red on an assertion other than
  the one it names (the script greps for the message, not for a non-zero exit).
- **E** — the restore is checked by **sha256**, not by exit code, and then the
  suite must be green.

Every one of these perturbations asserts the **exact number of sites** it edits.
That is not decoration either: it is what stopped section B's first version
from being a no-op by accident, and it is why A now edits a constant that
exists once instead of a literal that existed three times.

---

## 12. What this audit did NOT do

Named, so the next reader does not over-read it:

1. **`SpectralMoE` (`lib.rs:720-1284`), `infer.rs`, `gpu.rs`, `bf16_ops.rs`
   and `moe_fused.rs` were not audited formula by formula.** They are a
   different mechanism (a ternary rank-`r` MoE with a hierarchical router, a
   CUDA packing kernel, and a custom bf16 autodiff op) and the brief's four
   targets are the NS iteration, the low-rank factor path, `LinearLike`, and
   the comments. `moe_fused.rs` is 3243 lines of CUDA and cannot be measured
   on a CPU-only lane. **A named gap, not agreement.**
2. **Nothing here was measured on CUDA.** Every number is f64 on CPU. f32
   differs, and the one place it matters is §6, where the divergence is `inf`
   in f32 and a large finite number in f64 — the finding is if anything
   *wider* on the trainer's backend, but the *reachability* is unmeasured
   there.
3. **The PolarExpress triple was not re-verified** (already (a) at
   `71cc379`), and neither was the Jordan triple in `burn-muon-plus` (§1.2/§2.2
   of the prior document). The brief listed them as already done and they are
   not redone.
4. **The PolarExpress *schedule* was not compared against the fixed triple at
   new condition numbers.** The prior lane measured that the fixed triple beat
   the schedule at 3 iterations at every condition number it tried
   (`spectral-reference.md` §1.3c, §4.4) and this audit does not dispute it.
5. **The retraction-cost numbers were not re-measured.** §9 quotes the
   2026-09-29 measurements from AGENTS.md §3.1 and adds no new timing; this
   lane ran no GPU job by instruction.
6. **`qr_householder` (`lib.rs:689-718`) was read and is correct** (the sign
   convention `v = col + sign·‖col‖e₁` avoids cancellation, and the
   `Q = H₀H₁…` accumulation is right), but its nondeterminism is a **known,
   already-registered** finding (AGENTS.md §3.7, the 1-ULP QR residue) and was
   not re-investigated.


---

## 13. The landing — 2026-10-01, on `main`

The five commits of `wt/spectral-audit` (`4a68f72`, `93d82a1`, `0b580a7`,
`c78e36e`, `5aed8fc`) landed on `main` as **one commit**, and the granularity is
deliberate. Two reasons, both measured rather than preferred:

1. `4a68f72` and the twin's `7811191` are **the same commit**: identical tree
   content for both of its files, verified with `git diff 4a68f72 main -- <both
   files>` (empty). `7811191` is already on `main`, so re-landing it would have
   put the same content in the history twice.
2. **`93d82a1` and `0b580a7` do not compile.** Both leave a `}` that closes
   `mod tests` before the audit's test functions, so they land inside
   `mod polar_diag` — which has `use super::*`, i.e. the FILE's items, not
   `mod tests`' private ones. `dev()`, `det_factor()`,
   `ortho_factor_768x64()` and `ortho_err_per_entry()` are all private to
   `mod tests`, so the audit's own gates have nothing to call. Only `5aed8fc`
   builds (measured: 44 passed, 0 failed, 2 ignored). A squashed landing keeps
   `main` bisectable; `wt/spectral-audit` keeps all five commits, so nothing is
   lost but a broken intermediate.

### 13.1 The twin, and the file it overwrote

Two lanes wrote `docs/reviews/spectral-audit-2026-10-01.md`, **two
different documents**, 97 seconds apart, with nothing to detect the collision:

| | `3e1812b` | `7811191` (twin) | this lane |
|---|---|---|---|
| subject | `burn-sct` and the seam between the crates | the NS retraction's formulas | findings 1–5 of the formulas |
| lines | 494 | 218 | 667 |
| tools | `tools/polar_probe.rs` | `audit_2026_10_01.py` | 5 more oracles + `falsify.sh` |

`7811191` **replaced** `3e1812b`'s 494 lines with its own 218 — an add/add that
git resolved silently because the second writer did not read the first. The lost
494 lines are `burn-sct`'s site-by-site descent, the seam finding (zero non-dev
call sites, no crate under `crates/` depends on `burn-sct`), and the retraction's
cost/benefit pricing. Restored byte-for-byte from `3e1812b` as
[`spectral-stack-audit-2026-10-01.md`](spectral-stack-audit-2026-10-01.md),
under the name its own H1 gives it. **Not a union**: the two documents are about
different crates and a union would be 1000 lines about two things.

Preserved from the twin, unchanged: `tools/polar_probe.rs`,
`research/spectral-inventory-2026-10-01.md`, and the twin's `audit_2026_10_01.py`
— which is byte-identical to this lane's copy **except** for the §5 fixture fix
this lane made (`print_5` retracted a raw Gaussian, which is *outside* the basin
and diverges at 20 iterations, so the comparison measured a diverged basis). The
branch's copy is the union.

### 13.2 Per-file decisions

| file | decision | why |
|---|---|---|
| `docs/reviews/spectral-audit-2026-10-01.md` | **branch** | findings 1–5 + the gate register; the twin's 218 lines are a strict prefix of it (§0–§4 are identical) |
| `docs/reviews/spectral-stack-audit-2026-10-01.md` | **restored from `3e1812b`** | the content the twin overwrote; see §13.1 |
| `burn-spectral/tests/oracle/audit_2026_10_01.py` | **union (branch = twin + 19 lines)** | the branch is a superset: same file, plus the §5 fixture fix |
| `burn-spectral/tests/oracle/{sec4_fixed,sec4c_tolerance,sec4d_guard,sec5_6b,gate_dryrun}.py` | **branch** | new; the twin has none |
| `burn-spectral/tests/oracle/falsify.sh` | **branch + §13.3** | new; the twin has none |
| `burn-spectral/src/lib.rs` | **branch + §13.4** | the twin's `7811191` did not touch it; the audit's 3 comment corrections + 4 gates + 2 red-on-purpose tests are all here |
| `docs/protocols/ORACLE-TIERS.tsv` | **hand-inserted, 7 rows** | R1 coverage for the six new `.py` files. `tools/gen_oracle_tiers.py` was NOT run — see the file's own header, which records that it has destroyed 22 rows once |
| `tools/polar_probe.rs`, `research/spectral-inventory-2026-10-01.md` | **main, untouched** | the twin's instruments; the branch never had them |

### 13.3 `SIGMA_OVERSHOOT` — a correction to the landing brief

The brief for this landing said to keep "`SIGMA_OVERSHOOT` const + the rewritten
POWER_ITERS/1.05 comments from the tsct-diag lane `8a9bd7c`/`2254c7c`". **Main
never gained either**: `git grep SIGMA_OVERSHOOT main` returns nothing, and
`git log 8f97411d..main -- vendor/burn-fused/crates/burn-spectral/` lists only
`7811191` (docs). The const is on `wt/tsct-diag`, which is **unlanded**, and its
`lib.rs` hunk is a different-lane commit.

So: the comment wording is unified, in favour of **this** document's version,
because it is the deeper one — it replaces tsct-diag's rewrite with the measured
table (`(λ2/λ1)^(2k)`, 30–39 % of draws over 5 %, 2400 draws of reachability)
where tsct-diag's says "the estimate is NOT accurate to 5 % in general". Both are
true; only one carries the numbers. The **const** was deliberately NOT folded in:
it is a numerics-identical rename of a magic number in another lane's unlanded
commit, and taking it here would make that lane's cherry-pick conflict on a hunk
it wrote for exactly this reason. **Resolution for whoever lands tsct-diag:**
keep `SIGMA_OVERSHOOT`, take this document's comment text for FINDING 2/3, and
drop the `use super::ortho_error_per_entry as ortho_err_per_entry` alias at
`lib.rs:1433` in favour of the test-local helper — both branches' tests call the
same two names.

### 13.4 The falsify fix: two defects, and the script's contract now holds

The WIP in the worktree fixed the matching bug (`expect_red` filtered on
`tests::polar_diag::$test` with `--exact`, which matches no test name in the
file, so A and B reported NOT-DETECTED on every run). **That part was right and
was kept.** Running it exposed two further defects, one of them the same class
of blindness one level up:

1. **The basin gate tested a private COPY of the quintic.** `p(s)` was written
   out in the test as `15/8 s − 5/4 s³ + 3/8 s⁵`, so perturbing the production
   coefficients could not reach it — the gate pinned a copy, not the constant.
   The three spellings are now ONE (`NS_A`/`NS_B`/`NS_C`, used by the scalar
   arm, the batched arm and the test-only host-read fixture) and `p` reads them,
   with `assert_eq!((NS_A, NS_B, NS_C), (1.875, -1.25, 0.375))` so the dyadic
   exactness the tolerances assume is itself asserted.
2. **Section B perturbed both arms of a bit-identity.** The sync-free path is
   compared ELEMENTWISE against a host-read *fixture*; moving `1.05 → 1.00` in
   both leaves the identity intact, so the mutant could not fail **by
   construction**. One arm is the mutation that gate exists to catch.
3. `sec4d_guard.py` **crashed** (`torch.eye(K)` against a `[K,4]` result), i.e.
   the evidence for B-2's "the Cauchy–Schwarz guard is wrong" never printed its
   verdict. Fixed, and the case the verdict actually rests on was missing: an
   orthonormal `[768,64]` factor. Measured: the guard engages on **21/21**
   cases including all four well-behaved Gaussians, because `‖X‖_F/1.604 =
   4.99 > 1` beats a correct estimate of 1. It converges, to the wrong place.
4. `polar_square_and_tall_no_divergence` asserted `e20 < 1e-2` on a
   `Tensor::random` draw, and the unseeded RNG (AGENTS.md §3.7) makes that a
   property of one lucky draw: 1.7e-14 at the median over 20 f64 draws but up
   to 1.2e-1, 5 % of draws missing 1e-2. Now relative (`e20 < e3/100`), which is
   draw-independent and still fails the regression it guards.
5. And then the gate caught **the landing itself**: rewriting `p(s)` to read
   `NS_B` instead of the literal `−5/4` produced `a·s − b·s³`, which turns
   `−5/4` into `+5/4`, and `the_quintic_basin_is_sqrt_7_over_3` went red on its
   first run with `p(1) = 2.375`. A sign is exactly the class of thing a gate
   for a polynomial is for, and it is worth recording that the error was in the
   *fix*, not in the tree being landed.

`bash crates/burn-spectral/tests/oracle/falsify.sh` from `vendor/burn-fused`:
**every mutant DETECTED on its own assertion, restore byte-identical by sha256,
suite green after restore.** See §13.5 for the tallies.

### 13.5 What was measured at the landing

All of it on this box, 2026-10-01, CPU only (`vendor/burn-fused` is its own
cargo workspace, so the command is run from there):

```
cargo test -p burn-spectral --lib      44 passed, 0 failed, 2 ignored
                                      (the two class-B gates, red on purpose)
cargo test -p burn-spectral --lib -- --ignored
                                      0 passed, 2 FAILED - which is the point
bash tests/oracle/falsify.sh           A(1e-3) A(1e-2) B C D(x2) ALL DETECTED,
                                      E: restore byte-identical
                                      (sha256 03ab620da65035fa00a7f21e863e76cf6d264051089299bca2ede05edee39d89)
                                      exit 0
```

Python oracles, all standalone, interpreter
`/tmp/opencode/oracle-venv/bin/python` (python 3.12.14, torch 2.14.0+cpu,
numpy 2.5.2):

```
gate_dryrun.py     class-A gates green: basin=True estimate=True;
                   class-B red on purpose: B-1=True B-2=True
sec4_fixed.py      0/4000 random draws outside the basin, worst prescaled
                   1.1549 against an edge of 1.5275 -> B-2 is LATENT
sec4c_tolerance.py f64 max|out| ~ 3.3e+60 and f32 INF at l2/l1 <= 0.3; a
                   Frobenius prescale lands at 0.95 / 0.61 on the same factors
sec4d_guard.py     the Cauchy-Schwarz guard engages on 21/21 cases, including
                   all four well-behaved [64,4] Gaussians and the orthonormal
                   [768,64] -> the guard is WRONG (it converges to the wrong
                   place), which is why B-2 records it as a non-fix
sec5_6b.py         0.8x / 32.6x / 85.0x the one-way 1e-3 latch at spectral
                   spreads 0.3 / 0.1 / 0.01 -> class B-1
audit_2026_10_01.py  runs clean; findings 1-3 come out of it
```

And the registry, which is the one gate this landing can move:

```
tools/oracle_gate.py   ON MAIN, after this landing:
                       127 registered, 199 scanned, 3 violations, 8 waived
```
All three are **pre-existing and in another lane** (`burn-kda`, unregistered:
`tests/kda_param_grads_cuda.rs`, `tests/kda_rope.rs`,
`tests/oracle/upstream/fla_modules_rotary.py`). Before this landing the same
gate reported **4**, and the fourth was this lane's to fix:
`burn-spectral/tests/oracle/audit_2026_10_01.py`, unregistered, from the twin's
`7811191`. `tools/gen_oracle_tiers.py` was **not** run; the seven rows were
hand-inserted, which is what that file's own header asks for.

(The same gate on `wt/spectral-audit` itself reports 196 scanned / 2 violations,
because that branch predates the rope lane's files. Quoting one number for both
trees would be the §1.4 failure this project keeps paying for.)
