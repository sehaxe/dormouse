# External reference for the Newton–Schulz orthogonalisation in `burn-spectral`

**Status: COMPLETE as of 2026-09-30.** Research only — no Rust, no cargo, no
builds. Everything below was run; the transcript is in
`vendor/burn-fused/crates/burn-spectral/tests/oracle/`.

**Interpreter for every RUN below**: `/tmp/opencode/oracle-venv/bin/python`,
python 3.12.14, `torch==2.14.0+cpu`, `numpy==2.5.2`, CPU only. That venv was
created on this box by the `burn-rmsnorm` oracle lane; it was used
**read-only** and nothing was installed into it. Following the RMSNorm
precedent: torch's arithmetic is COMPILED C++ and therefore not quotable, so
the wheel and version are recorded instead of any number claimed to come out of
it. (System `python3` is 3.14.7 with numpy 2.5.3 and **no** torch; the first
attempt to run the pinned reference died on `ModuleNotFoundError: No module
named 'torch'` and that is recorded rather than worked around.)

**Headline, in one line each:**

1. Our coefficient `(1.875, −1.25, 0.375)` is **(a)** — the authors' own
   generator returns it *bit-exactly* — but the **source the code under test
   cites is not the source that implements it**, and the way it is used
   (fixed, every iteration) is not what that source does. §1.
2. `burn-muon-plus::orthogonalize` is a faithful transcription of a
   **real, running, (a)** reference: Keller Jordan's `newtonschulz5`. §2.
3. The retraction's **definition** has a running external reference (LAPACK
   `dgesdd` via numpy), and it found two things our own tests cannot. §3.
4. The load-bearing refutation against removing the `sigma_max` power iteration
   is **independently re-derived and strengthened**, and the reference
   implementations' *own* Frobenius prescale is recorded as the interesting
   finding — not acted on. §3.4.
5. **No reference implementation does a power iteration.** Recorded, per
   instruction, not acted on.

---

## 0. What our code actually is (the thing a reference must speak to)

### 0.1 The premise in the task brief is wrong about our own code — corrected

The brief says: *"Our `NS_COEFFS` in `vendor/burn-fused/crates/burn-muon-plus/src/lib.rs`
ends at exactly `(1.875, −1.25, 0.375)`."*

**It does not.** Measured by reading the file:

| site | line | coefficients |
|---|---|---|
| `burn-muon-plus/src/lib.rs:189` `NS_COEFFS` | 189 | **`(3.4445, -4.775, 2.0315)`** — the *Jordan* triple |
| `burn-spectral/src/lib.rs:254` | 254 | `(15.0f32/8.0, -5.0f32/4.0, 3.0f32/8.0)` = **`(1.875, −1.25, 0.375)`** |
| `burn-spectral/src/lib.rs:305` (batched) | 305 | same, re-spelled |

So the two crates encode **two different coefficient sets from two different
appendices of two different papers**, and the `(1.875, …)` triple is the one in
`burn-spectral`, hardcoded at two sites and not named `NS_COEFFS`. This is a
glossary item (AGENTS.md §1.7): "the NS coefficients" currently means two
things in this repo, and the brief inherited the wrong one.

**Both doc comments are correct about their own source** — I checked each
against the paper rather than against the other doc comment:

- `burn-muon-plus:181-188` cites v3 App. D.1 "Jordan Coefficients" for
  `(3.4445, −4.7750, 2.0315)`. **CONFIRMED verbatim** in the v3 HTML:
  section `A4.SS1` = *"D.1 Jordan Coefficients [15] — In [15], the
  coefficients are set to (a,b,c) = (3.4445, −4.7750, 2.0315)."*
- `burn-spectral:169-175` cites v3 App. D.3 "PolarExpress Coefficients" for
  `{(aₜ,bₜ,cₜ)}ₜ₌₁⁸` ending at `(1.875, −1.25, 0.375)`. **CONFIRMED** in the
  v3 HTML: section `A4.SS3` = *"D.3 PolarExpress Coefficients [2]"* with that
  8-tuple, captioned *"with subsequent coefficients numerically equal to
  (1.875, −1.25, 0.375)."*
- The same comment's aside — that v1 and v2 contain no `1.875` — is
  **CONFIRMED**: occurrences of `1.875` in the v1 HTML = **0**, in v2 = **0**
  (v3 = 9, all in D.3). The string `PolarExpress` does appear 3× in v1 and v2,
  so "no NS coefficients" is right and "no mention at all" would not be.

### 0.2 `burn-spectral::polar_orthogonalize` (`lib.rs:208-266`)

Prescale: 5-step power iteration on the Gram + Rayleigh quotient, divide by
`sigma · 1.05`. Then `iters` applications of **one fixed** triple
`(15/8, −5/4, 3/8)`. Retraction = that, `.detach()`ed.

### 0.3 `burn-muon-plus::orthogonalize` (`lib.rs:283-305`)

`orient_and_normalize` (266-276): transpose if taller, divide by
`‖·‖_F.clamp_min(1e-7)`. Then `ns_steps` (default 5) applications of
`NS_COEFFS` = `(3.4445, −4.775, 2.0315)`.

---

## 1. PolarExpress / Higham & colleagues — the orthogonalisation polynomial

### 1.1 The citation our code names: **EXISTS, and carries no code**

- `https://arxiv.org/abs/2602.21545` — "MUON+: Towards More Effective Muon via
  One Additional Normalization Step for LLM Pre-training", Zhang, Zhao, Liu,
  Wang, Su, Tan, Zhang. v1 25 Feb 2026, **v3 14 May 2026** (the date our code
  cites). Fetched 2026-09-30 from `https://arxiv.org/html/2602.21545v3`.
- The appendix structure is real and matches the doc comment's names exactly:
  `D.1 Jordan Coefficients`, `D.2 You Coefficients`, `D.3 PolarExpress
  Coefficients` (LaTeXML ids `A4.SS1/SS2/SS3`; `A4` is the 4th appendix = D).
- **Its reference [2] is not the Polar Express paper** — it is Bernstein
  (2025), *"Deriving Muon"*, `https://jeremybernste.in/writing/deriving-muon`.
  So Muon+ D.3 is tabulating Bernstein's coefficient list. The prose
  attribution is worth knowing, because the list itself is identical to the
  one printed in the Polar Express paper.
- **The Muon+ paper ships NO supplementary code.** Every URL in the v3 HTML
  matching `github|gitlab|zenodo|code` is either LaTeXML boilerplate
  (`arXiv/html_feedback`, `brucemiller/LaTeXML`) or
  `https://kellerjordan.github.io/posts/muon/`. **There is no repository for
  2602.21545.** The D.3 table is prose-only and must be transcribed by hand.
  Any gate that wants Muon+ numbers is transcribing, not importing.

### 1.2 The actual origin of the triple: **EXISTS, and it is the Polar Express paper — arXiv:2505.16932**

"The Polar Express: Optimal Matrix Sign Methods and Their Application to the
Muon Algorithm" (Amsel, Persson, Musco, Gower). Fetched 2026-09-30;
`https://arxiv.org/html/2505.16932v5` (v5, 4 May 2026). Its **Appendix A "Code
for Polar Express"** prints the same 8-tuple, hardcoded, with the comment
`# subsequent coeffs equal this numerically`.

**Authors' repository — EXISTS, no tags, MIT:**
`https://github.com/NoahAmsel/PolarExpress`, pinned at
**SHA `71cc37943d99cae780024c1d198977f2f8795407`**, last commit
**2026-08-14T19:46:01Z**. Named in the paper's own footnote 7 (*"Code including
the offline stage can also be found at …"*). Tree: `LICENSE`, `README.md`,
`plotsforpolar.ipynb`, `polar_express.py`. **No releases, no tags** — the SHA is
the only pin, which is why it is recorded here.

Pinned byte-for-byte to
`tests/oracle/upstream/polar_express.py` (sha256 `6b6b5dbe…`).

### 1.3 What RUNNING it says — and the finding that matters

`run_polar_express.py` imports the pinned module **unedited** and runs both
stages. (One accommodation, disclosed: their online stage ends
`return X` where `X = G.bfloat16()`, so `.numpy()` raises
`TypeError: Got unsupported ScalarType BFloat16`. I read it as
`.float().numpy()`. **I did not edit their file.**)

**(a) The triple is EXACT, and it is ours to claim.** Calling their
`optimal_quintic` directly:

```
optimal_quintic(l=1      , u=1) -> (1.875, -1.25, 0.375)  == (15/8,-10/8,3/8): True
optimal_quintic(l=0.999999, u=1) -> (1.875, -1.25, 0.375)  == (15/8,-10/8,3/8): True
optimal_quintic(l=0.999995, u=1) -> (1.875, -1.25, 0.375)  == (15/8,-10/8,3/8): True
optimal_quintic(l=0.99999 , u=1) -> (1.8750096406712431, …)                  : False
optimal_quintic(l=0.999   , u=1) -> (1.87593823305953, …)                    : False
```

`polar_express.py:29-32` is `if 1 - 5e-6 <= l / u: return (15/8)/u,
(-10/8)/u**3, (3/8)/u**5`. With `u = 1` that is `(1.875, −1.25, 0.375)` and
Python's `==` against `(1.875, −1.25, 0.375)` is `True` — **bit-exact**, and
all three are dyadic rationals, so there is no decimal-rounding question and
no nine-digit golden constant is needed for the coefficients themselves. This
is the strongest form the claim can take: the value our code hardcodes is the
literal return value of the authors' own function on the branch it takes.

**(b) FINDING — the cited table is not what the cited code produces.** The
authors' HEAD module default (`l=1e-3, num_iters=10, degree=5,
safety_factor_eps=1e-2, cushion=0.02`) generates a list that **disagrees with
the table printed in both papers at the third significant digit**:

| t | HEAD `71cc379` | printed (2505.16932 App. A / 2602.21545v3 D.3) | rel diff |
|---|---|---|---|
| 1 | 8.23731249049555 | 8.28721201814563 | 6.02e-03 |
| 2 | 4.08244199906483 | 4.10705911154220 | 5.99e-03 |
| 3 | 3.92634799225465 | 3.94869085348229 | 5.66e-03 |
| 4 | 3.29821871330852 | 3.31841965737060 | 6.09e-03 |
| 5 | 2.29703694345526 | 2.30065201995482 | 1.57e-03 |
| 6 | 1.87638053514404 | 1.89130140778740 | 7.89e-03 |
| 7 | 1.85644234855046 | 1.87500148085345 | 9.90e-03 |
| 8 | 1.85643669297913 | 1.875 | 9.90e-03 |

HEAD's entry 10 is `(1.8749954775664746, −1.2499909551547645,
0.37499547758829033)` — within 4.5e-6 of the triple, but **not equal**
(`exact in binary: False`). I swept all four combinations of `cushion ∈
{0, 0.02}` × `safety_factor_eps ∈ {0, 0.01}`; **none** reproduces the printed
table (max |diff| 0.26 / 0.62 / 0.77 / 1.5 respectively). The likely cause is
that `cushion` post-dates the paper. **So: the papers' table is not
reproducible from the authors' current HEAD, and any future gate must pin
which one it means.** Recording this because the alternative — citing "the
authors' code" for the printed numbers — is exactly the kind of reference that
agrees with whatever we happen to have written.

**(c) FINDING — the usage is not what the source does, and the doc comment
implies it is.** `burn-spectral:169-175` presents `(1.875, −1.25, 0.375)` as
"the last entry of the PolarExpress schedule … which prints
`{(aₜ,bₜ,cₜ)}ₜ₌₁⁸`". True. What the code then does is apply that entry
**`iters` times, unchanged**. The reference applies the first `T` entries,
**each once**, because the whole thesis of the paper is that the *first*
polynomial must be aggressive about initial convergence and the *last* is the
high-accuracy one; `polar_express.py:92-95` iterates `for a, b, c in hs` over a
changing list. Measured head-to-head (`illconditioned_sweep.py`, 768×64 rank
64, per-entry `‖UᵀU−I‖`, 3 iterations):

| σ_min/σ_max | OURS (σ_max prescale, fixed triple) | PolarExpress schedule (their code, bf16) |
|---|---|---|
| 1 (on manifold) | **5.37e-17** | 1.03e-01 |
| 1e-1 | 8.45e-02 | 1.55e-01 |
| 1e-3 | 9.90e-02 | 1.43e-01 |
| 1e-6 | 1.13e-01 | 1.39e-01 |

**In fairness to our code: at 3 iterations the fixed triple is better at every
condition number tested, and exact on the manifold.** The schedule's advantage
appears only at ≥5 iterations on badly-conditioned input. This is *not* a
defect — it is a design choice with a measured justification. **What is a
defect is the doc comment**, which cites a schedule to justify a fixed
polynomial and does not say that only the last entry was taken. §4.3.

**(d) No Higham implementation was found or needed.** The paper credits
Newton–Schulz to Higham, *Functions of Matrices* (2008), ch. 5, and the
classic *Accuracy and Stability of Numerical Algorithms*, ch. 8. Both are books
whose code is not fetchable as a pinned artifact I could run. I did not
reconstruct either. **Recorded as a gap, not as a result:** no Higham-authored
executable reference is pinned here, and the quintic `(15x − 10x³ + 3x⁵)/8` is
attributed in this repo to the Polar Express paper's table, not to Higham.

### 1.4 Other repos located (recorded, not run as oracles)

- `NovelAI/Newton-Schulz-measurements` @ `6a4ce090cd70622f1d263f63b84deeec6a399953`
  (2025-12-20, no license). `upstream/novelai_polarexpress.py` is a **verbatim
  copy of the paper's Appendix A** — its own header says *"this is the vanilla
  polar express code. it's from https://arxiv.org/abs/2505.16932. they have a
  github repo with different (more powerful) code."* Third-party copy of the
  authors' printed code; **tier (a)-with-a-caveat, not (a)**: it is not the
  authors' repository, so a silent edit there would not be caught.
- `Dao-AILab/gram-newton-schulz` @ tag **v0.1.6 = `e45d0aca7083cb275c9a303220c05c4abecd9187`**
  (2026-07-02, no license, 185 stars). Exports
  `POLAR_EXPRESS_COEFFICIENTS` and a restarts-based Gram formulation. The
  README is pinned. **Not run** — it is a Triton/fp16 training kernel whose
  value here would be the coefficient table, and I did not want to report a
  number from a file I had not executed.

---

## 2. A public PyTorch implementation of Muon's Newton–Schulz iteration

### 2.1 **EXISTS — unambiguous, canonical, and I RAN IT**

Keller Jordan, *"Muon is Scalable for LLM Training"*,
`https://kellerjordan.github.io/posts/muon/`. Fetched **2026-09-30**; the
response is pinned whole as `upstream/kellerjordan_muon_post.html`
(sha256 `4e949829…`, 39 101 B) and the code block is extracted verbatim to
`upstream/kellerjordan_newtonschulz5.py` (sha256 `60fea779…`) — an
`inspect`-style transcript of the bytes that ran, never hand-edited.

```python
# Pytorch code
def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X
```

**No tag, no SHA** — it is a blog post, pinned by content hash only. The
underlying repo `KellerJordan/modded-nanogpt` is at
`4ea6b937337a4889b8cfe3f38a93d120048d8f71` (2026-09-28, MIT, no tags) but the
blog is the primary source of `newtonschulz5` and is what I ran.

**The pin is load-bearing:** this is the code that fixes our `NS_COEFFS` to
`(3.4445, −4.775, 2.0315)`, and Muon+ D.1 attributes the same triple to
"[15]" via a different route. Two independent published sources agree on the
value.

### 2.2 What RUNNING it says

`run_targets_2_and_3.py` runs **their bytes** (bf16 as shipped) against a
transcription of `burn-muon-plus`'s `orient_and_normalize` + `orthogonalize`,
5 steps, f64:

| shape | theirs per-entry | our transcription (f64) | max abs diff |
|---|---|---|---|
| (768, 64) | 5.4278e-02 | 5.5038e-02 | 6.17e-03 |
| (64, 768) | 3.4845e-02 | 3.4851e-02 | 8.27e-03 |
| (256, 256) | 2.2118e-02 | 2.2030e-02 | 6.89e-03 |
| (1000, 32) | 3.4402e-02 | 3.5356e-02 | 1.12e-02 |

The per-entry orthonormality errors agree to 3 significant figures; the
element-wise difference sits at the bf16 floor (their `X = G.bfloat16()`), not
at a structural disagreement. **Two real, nameable differences, both benign and
both worth a comment rather than a change:**

1. **eps placement.** Theirs is `X /= (X.norm() + eps)` with `eps = 1e-7`; ours
   is `.clamp_min(1e-7)` (`lib.rs:274`). Identical whenever `‖G‖_F ≫ 1e-7`.
   Measured at `‖G‖_F = 6.377e-08`: their divisor is `1.638e-07`, ours is
   `1.000e-07` — ours is smaller, so **ours is the more conservative of the
   two** on a degenerate input. No defect.
2. **No bf16 cast.** Ours keeps the input dtype. Correct on this box: AGENTS.md
   §2.1 records that bf16 has no tensor-core path here and that every bf16 run
   is slower than fp32. This is a deliberate, correct divergence from the
   reference and the doc comment should say so.

### 2.3 The original derivation

Bernstein & Newhouse, *"Old optimizer, new norm: An anthology"* (arXiv:2409.17025),
Appendix A, is credited by Bernstein as the origin of the dualized-gradient
Newton–Schulz. **Not fetched and not run** — it is a paper, and I did not want
to transcribe an appendix I had not opened. Named as the upstream of the
upstream; §5 records it as the one obvious remaining fetch.

---

## 3. A reference for the polar retraction on the TSCT masters

### 3.1 The definition's reference: **EXISTS, and it is LAPACK**

The retraction's *specification* is not ours: `polar(X) = U Vᵀ` from an SVD.
numpy's `np.linalg.svd` routes to LAPACK `dgesdd` — a third-party Fortran
implementation shipped inside numpy, entirely independent of this repo. That
makes the target `polar(X)` an **(a)** oracle and the question the iteration
answers is "how close did we get". This is a stronger instrument than I
expected to find: it is not a golden constant, it is a *definition* the crate
is claiming to approximate.

### 3.2 What RUNNING it found — two things our own tests cannot see

`final_numbers.py`, 768×64 rank 64, prescribed spectral spread, 3 iterations,
σ_max prescale ×1.05:

| σ_min/σ_max | rel ‖F error‖ vs LAPACK polar | per-entry `‖UᵀU−I‖` | σ_max(out) |
|---|---|---|---|
| 1 (on manifold) | **1.1655e-15** | 3.95e-17 | 1.000000000 |
| 0.5 | 1.9979e-06 | 4.99e-07 | 1.000000000 |
| 0.1 | 1.4540e-01 | 3.09e-02 | 1.000000000 |
| 0.01 | 5.4966e-01 | 8.38e-02 | 1.000000000 |
| 1e-3 | 7.1617e-01 | 9.90e-02 | 1.000000000 |
| 1e-4 | 7.9420e-01 | 1.06e-01 | 1.000000000 |

**FINDING A — the retraction is a projection onto the manifold only *near* the
manifold.** On a factor whose singular values have spread 10:1, the 3-iteration
retraction is **14.5% away from the polar factor**, and 55% at 100:1. The
existing test `retraction_holds_the_manifold_at_rank_64` starts *on* the
manifold, where any sequence containing the identity would pass. **The
iteration count is not a free parameter; it is set by the input's spectral
spread, and nothing in the crate measures that.** This is a real hole: our
tests are tier (d) (compared against our own claim) and they are green on
exactly the input where the procedure is trivially correct.

**FINDING B — the `POWER_ITERS` comment's justification is false, measured.**
`lib.rs:132-135` says: *"Rayleigh-quotient error shrinks as (λ2/λ1)^k; 5 gives
the estimate well within the 1.05 safety factor even for square Wishart
(λ2/λ1 ≈ 1)."* Sampled over 200 independent Wishart draws per shape, 5 steps:

| shape | median rel err | p95 | max | **frac > 5%** | max prescaled σ_max |
|---|---|---|---|---|---|
| (768, 64) | 4.456e-02 | 7.614e-02 | 1.091e-01 | **0.355** | 1.068985 |
| (4096, 64) | 4.561e-02 | 6.225e-02 | 6.952e-02 | **0.360** | 1.023536 |
| (512, 128) | 4.742e-02 | 7.683e-02 | 1.028e-01 | **0.460** | 1.061516 |
| (256, 256) | 4.816e-02 | 8.758e-02 | 9.966e-02 | **0.465** | 1.057800 |
| (128, 128) | 4.291e-02 | 8.730e-02 | 1.100e-01 | **0.385** | 1.070081 |
| (64, 64) | 3.459e-02 | 8.627e-02 | 1.238e-01 | **0.320** | 1.086897 |

The estimate converges from below, so it is a lower bound and dividing by it
puts the **true** σ_max at `1/estimate_rel · 1/1.05` — the last column, which
exceeds 1.0 on the worst draws. **On 32-47% of Wishart draws the 1.05 factor
does not cover the error**, so the NS input lands above the basin `[0,1]` of
`p(s) = 15/8·s − 5/4·s³ + 3/8·s⁵`. The comment's own parenthetical is the
reason: the Rayleigh error decays as `(λ2/λ1)^(2k)`, which is *slow* precisely
when `λ2/λ1 ≈ 1` — the case it claims to have covered.

**This is benign in effect and I am not proposing a change.** `p'(1) = 0`, so
the fixed point is superattracting: every row of the table above shows
`σ_max(out) = 1.000000000` regardless. **What is wrong is the stated reason.**
The safety factor does not keep the input in the basin; the superattracting
fixed point does. That is a comment that will mislead the next person who
raises `POWER_ITERS` or reasons about the prescale, and it is a doc/code
disagreement of the §1.7 class.

### 3.3 Two harness mistakes of my own, corrected in place

Disclosed because they are exactly the failure this document exists to prevent:

- I first reported **3.4548e-02** per-entry for a tall `[64, 768]` factor whose
  distance to LAPACK's polar was 1.15e-15. Both cannot be true: I had applied
  `‖XᵀX − I₇₆₈‖` to a matrix whose **columns** are the orthonormal ones. Fixed
  in `final_numbers.py` (`ortho_on_span`); the retraction was always right.
- I first reported the power-iteration error from **one random draw** (7.25e-02
  for `[768,64]`), then the same shape measured 1.82e-02 on another draw. Both
  were real; the quantity is a random variable and one draw is not a
  measurement. `final_numbers.py` samples 200.

### 3.4 The `sigma_max` prescale: refutation re-derived, and the interesting absence

**The refutation STANDS, and I re-derived every number independently** rather
than trusting `lib.rs:188-195`. Float64, `[768,64]`, rank 64, starting exactly
on the manifold (σ = 1, ‖·‖_F = 8.000000), 3 iterations:

| prescale | σ_max after 3 iters | per-entry `‖UᵀU−I‖` |
|---|---|---|
| σ_max · 1.05 (ours) | **1.000000000** | **5.3657e-17** |
| ‖·‖_F · 1.05 | 0.675523722 | 6.7958e-02 |
| ‖·‖_F · 1.01 | 0.694388893 | 6.4728e-02 |

`lib.rs:191` claims `0.6992` and `6.4e-2` for the Frobenius row. I measure
**0.6755 / 6.80e-02** (×1.05) and **0.6944 / 6.47e-02** (×1.01). The doc's
numbers are in the right band and the right direction, and are most consistent
with a 1.0-1.01 factor in fp32; I could not reproduce them exactly. **The
conclusion is unchanged and is now backed by a run rather than a table:**
6.5e-2 to 6.8e-2 per-entry is **65-68× over the 1e-3 one-way `max_ortho`
latch**, versus machine precision for the σ_max prescale. The reviewer's
refutation is correct; **do not remove the power iteration.**

**The interesting absence, recorded and NOT acted on, as instructed:** *no
reference implementation found does a power iteration.* `polar_express.py:89`
— `X = X / (X.norm(dim=(-2,-1), keepdim=True) * 1.01 + 1e-7)` — Frobenius.
`kellerjordan_newtonschulz5.py` — `X /= (X.norm() + eps)` — Frobenius. The
paper's Algorithm 1 says `M/(‖M‖_F + 10⁻²)` — Frobenius. So the reference
implementations do exactly what the reviewer proposed, and they do it in
*training* configurations (bf16, 5-6 steps) where the schedule's aggressive
first polynomial recovers the lost small singular values — which my
measurements in §1.3(c) corroborate. **I am not proposing to change our
prescale.** The honest reading is that the two designs are matched to
different problems, and our own measurements (§3.2 Finding A) show ours is the
better one for the on-manifold, low-rank retraction it was written for.

---

## 4. Tier assignment, and what a real reference could have caught

### 4.1 Honest tiers

| artifact | tier | why, and what it is not |
|---|---|---|
| `upstream/polar_express.py` @ `71cc379` — **run** | **(a)** | the Polar Express authors' own repository, named in their own paper's footnote, MIT, SHA-pinned. Its `optimal_quintic` returns our coefficient **bit-exactly**. Not a transcription. |
| `upstream/kellerjordan_newtonschulz5.py` + the page it came from — **run** | **(a)** | Keller Jordan's own published function, verbatim bytes, pinned by content hash. Fixes `NS_COEFFS` to `(3.4445, −4.775, 2.0315)`. Not a transcription. |
| LAPACK `dgesdd` via numpy — **run** | **(a)** | third-party Fortran; `polar(X) = U Vᵀ` is the *definition* of the target. Not a golden constant and not ours. |
| `upstream/novelai_polarexpress.py` @ `6a4ce09` | **(a)-with-a-caveat** | a verbatim third-party copy of the paper's Appendix A. Real code, but not the authors' repository, so a silent edit there is invisible. |
| `upstream/gram_newton_schulz_README.md` @ `v0.1.6` | **not run** | recorded, contributes no number. |
| `polar_express.py`'s printed 8-tuple (2505.16932 App. A) and Muon+ D.3 | **(a) printed, transcribed by us** | the *prose* is (a); our `PAPER` list in the scripts is a **transcription**, labelled as such in the source. |
| `optimal_composition` online loop in numpy (`schedule_fro`) | **(b)** | transcribed from `polar_express.py:92-95`. Validated two ways in §4.2. |
| our own procedures (`ours`, `retract`, `orient_and_normalize`) | **(b)** | transcriptions of the Rust. Never presented as a reference. |
| Higham, *Functions of Matrices* ch. 5 / *ASNA* ch. 8 | **none** | **no external reference exists here** — books, not fetchable artifacts. Not reconstructed. |
| Bernstein & Newhouse 2409.17025 App. A | **none** | not fetched, not run. §5. |

### 4.2 Validating my own transcriptions (a (b) is only useful if it is checked)

`illconditioned_sweep.py` cross-checks the transcribed online loop two
independent ways:

1. **Against the authors' own torch code, on the same input.** Transcribed
   f64 schedule vs their bf16 `PolarExpress`, per-entry:
   `1.0525e-01` vs `1.0277e-01`, `1.5449e-01` vs `1.5462e-01`,
   `1.4235e-01` vs `1.4120e-01` — agreement at the bf16 floor, across four
   shapes and two iteration counts.
2. **Against a published MATLAB run.** Ethan Epperly, *"A Neat Not-Randomized
   Algorithm: Polar Express"* (2025-06-07), ran the **unscaled** printed list
   in MATLAB on `randn(100)/25` and printed `‖Xₜ − polar‖_F` per iteration.
   Different seed, so **not** bit-exact — the check is the decay shape and the
   landing:

   | iter | mine (seed 20260930) | theirs (published) |
   |---|---|---|
   | 1 | 6.666017e+00 | 9.921347e-01 |
   | 5 | 8.687107e-01 | 1.551595e-01 |
   | 6 | 8.878585e-03 | 5.885490e-03 |
   | 7 | 7.732343e-09 | 2.286853e-07 |
   | 8 | **2.936274e-14** | **1.113148e-14** |

   Both land at machine precision; the shape matches from iteration 5 on. The
   first-iteration gap is seed-dependent and is **not** presented as agreement.

### 4.3 What a real reference caught that our own tests could not

Four things, in descending order of how much they matter. Note that **all four
are invisible to a self-comparison**, which is the whole argument.

1. **The power iteration is not converged, and the doc says it is** (§3.2
   Finding B). Our own test can only assert the estimate *moves toward*
   σ_max; comparing it to LAPACK's exact spectral norm is what turns "5 steps"
   from a constant into a distribution with a 32-47% failure rate against its
   own stated 1.05 margin. **Tier (d) is structurally incapable of finding
   this**, because the only available "truth" is our own estimate.
2. **The retraction's accuracy is a function of spectral spread, and nothing
   measures it** (§3.2 Finding A). `retraction_holds_the_manifold_at_rank_64`
   starts on the manifold, where 0 and 3 iterations are indistinguishable. LAPACK
   turns a qualitative "it holds the manifold" into a table showing 14.5% error
   at a 10:1 spread — a number that would decide whether 3 iterations is the
   right cadence for a factor that has drifted.
3. **The coefficient we cite is used in a way the citation does not describe**
   (§1.3(c)). A (a) reference is the only thing that can notice that the
   schedule's first entry is `(8.287, −23.596, +17.300)` and ours is a fixed
   constant. Our own test agrees with our own constant by construction.
4. **Two published sources disagree** (§1.3(b)): the papers' printed table vs
   the authors' HEAD generator, ~6e-3 relative from entry 1. A gate built on
   "the authors' code" would have inherited the wrong table silently, and
   neither paper nor code would have told us.

### 4.4 What it did NOT catch

Stated so the next reader does not over-read this document:

- The `sigma_max` prescale and the `p'(1) = 0` superattraction are **fine**.
  Every measurement confirms σ_max(out) = 1.000000000.
- The fixed triple at 3 iterations **beat** the Polar Express schedule at every
  condition number tested (§1.3c). Nothing here argues for changing it.
- The `eps` placement difference vs Keller Jordan is benign and ours is the
  more conservative (§2.2).
- `burn-muon-plus::orthogonalize` reproduced the reference to the bf16 floor
  on four shapes. No defect found in it at all.

---

## 5. Provenance log

Everything fetched 2026-09-30 (UTC) from `/home/sehaxe/dormouse-wt/spectral-oracle`
at base commit `c3314e9`, branch `wt/spectral-oracle`. Pinned under
`vendor/burn-fused/crates/burn-spectral/tests/oracle/upstream/`.

| file | sha256 | origin | pin |
|---|---|---|---|
| `polar_express.py` | `6b6b5dbe7d59422995a7eb3694af74ba5acad7a230705933650a37a56ad75d09` | `raw.githubusercontent.com/NoahAmsel/PolarExpress/71cc37943d99cae780024c1d198977f2f8795407/polar_express.py` | SHA (no tags exist) |
| `polar_express_README.md` | `05147a49c3d38d7de926981922f268cb49dfedd090645ed3f271e5e20d44fd8f` | same SHA | SHA |
| `polar_express_LICENSE` | `d54176e99ae8b4d371aff1bc699bfac01bb4e0ff56278914e6282015df96b453` | same SHA, MIT | SHA |
| `novelai_polarexpress.py` | `5527e61f570cc5ea8d2c68e6a70ea24c494a20a601399e1030234092acf4fa58` | `raw.githubusercontent.com/NovelAI/Newton-Schulz-measurements/6a4ce090cd70622f1d263f63b84deeec6a399953/polarexpress.py` | SHA |
| `novelai_polarexpress3.py` | `3ca12462fd08af99e1d767da0bac4c2b8262ba27c5216f442ebb571eaa857d3a` | same SHA | SHA |
| `gram_newton_schulz_README.md` | `c87017278998e3c5dd3782663d74d320342293543724ca7d08e87393d67bf99f` | `raw.githubusercontent.com/Dao-AILab/gram-newton-schulz/e45d0aca7083cb275c9a303220c05c4abecd9187/README.md` | **tag v0.1.6** |
| `kellerjordan_muon_post.html` | `4e94982993cf2660a47b059c46edb0729acc6198fc90f1375725b9cf3096aa76` | `https://kellerjordan.github.io/posts/muon/` | content hash (no tag exists) |
| `kellerjordan_newtonschulz5.py` | `60fea7798223d4c1ff38db22fecaaaf98474a8c3f2a1ca8a1b55d765b28e532a` | extracted from the line above, verbatim | derived |

Papers (fetched, **not** pinned — no reproducible artifact exists for either):

- `arXiv:2602.21545` v3, 14 May 2026, "MUON+…", `https://arxiv.org/html/2602.21545v3`.
  App. D.1/D.2/D.3 verified. **No code, no repository** — every code-shaped URL
  in the HTML is LaTeXML boilerplate or the Keller Jordan blog.
- `arXiv:2505.16932` v5, 4 May 2026, "The Polar Express…",
  `https://arxiv.org/html/2505.16932v5`. App. A "Code for Polar Express"
  verified; footnote 7 points at `NoahAmsel/PolarExpress`.

Script transcripts committed beside these, all re-runnable with
`/tmp/opencode/oracle-venv/bin/python` from
`vendor/burn-fused/crates/burn-spectral/tests/oracle/`:

| script | sha256 | what it establishes |
|---|---|---|
| `run_polar_express.py` | `e7aa034181a7d6bf7ea39ba39b12255da290ab5d7925989f11d59850868a3f89` | §1.3(a) exact triple, §1.3(b) table disagreement, §1.3(c) head-to-head |
| `run_targets_2_and_3.py` | `813b45866b8151606a7bc58589e954fa62df7b68872fd0ca4436120b32d590b4` | §2.2 transcription vs Keller Jordan; §3.2 first pass |
| `illconditioned_sweep.py` | `b990f3df1c74f2ad4975a7e69436c6ae55183d8348d4e97b43aee71bc818c2fb` | §1.3(c) sweep; §4.2 both transcription checks |
| `measure_algorithms.py` | `93f07fdef30c5a1b432e31aa43acf797b4fc60fbb5299b9a9cc31b90e83bfa0e` | §3.4 prescale re-derivation |
| `pin_findings.py` | `7b907c8b76cc60095ca55aaa88929ca17173ebcd22e2d2885785debae63f24a0` | superseded by `final_numbers.py`; kept because it contains the two harness bugs of §3.3, and a corrected run that silently replaces a buggy one is how a wrong number survives |
| `final_numbers.py` | `1823f62b33853da264a3cbacd9446588ea80a3ddd9046242c1eb3a0079084e0d` | §3.2 Findings A and B |

### Remaining gaps, named

1. **Higham's own code — not obtained.** Both candidate sources are books. No
   external reference is pinned for the quintic from Higham; the attribution in
   this repo goes to the Polar Express table instead.
2. **Bernstein & Newhouse 2409.17025 Appendix A — not fetched.** Named as the
   origin of the dualized-gradient iteration.
3. **Muon+ (2602.21545) ships no code.** Any gate wanting its numbers
   transcribes from prose. Recorded so nobody later reports a transcription as
   an import.
4. **`Dao-AILab/gram-newton-schulz` — README pinned, code not run.** It is a
   Triton/fp16 training kernel; its coefficient table is available but unused.
5. **No `bench` was run and no Rust was built**, per the task. Nothing in this
   document says anything about the crate's behaviour on CUDA or in f32 — every
   number above is f64 on CPU, and f32 will differ. Where a number is quoted
   from our own doc comment rather than from a run here, it is marked as such
   (§3.4).
