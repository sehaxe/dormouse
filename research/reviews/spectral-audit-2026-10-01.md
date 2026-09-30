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
`research/papers/spectral-reference.md` §1.2/§2.2 and they stand.

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

*(sections 5–9: the low-rank factor path, `LinearLike`, the retraction arms, the
class-B register, the falsification script, and the retraction-cost question —
appended as the work lands)*
