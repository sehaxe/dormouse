# TSCT — our own invention. Lineage, precise definition, and delta vs the published mechanisms.

**Fetch date: 2026-09-29.** Read-only pass over `vendor/dormouse-fused/crates/burn-spectral/`,
`vendor/dormouse-fused/crates/burn-sct/`, `crates/dormouse-core/src/param.rs`,
`crates/dormouse-core/src/loop_block.rs`, `crates/dormouse-core/src/routing.rs`,
`crates/dormouse-train/src/lib.rs`. No GPU, no build, no test.


> **2026-10-02:** the crate analysed as `burn-sct` (section 5) was deleted — it
> never executed in the dormouse build and `burn-spectral` fully supersedes it. The
> `from_dense` importer and the exact `safe_qr` oracle discussed below no longer
> exist in-tree; both were cited as "the cheapest unrun check", and the crate was
> evaluated and found dead. Archived as A/B history: nothing else in this file
> depends on it.

**Owner instruction honored:** TSCT is *ours*. There is no TSCT paper and none is sought.
What follows is (1) the verified lineage it is built on, (2) a precise reading of what the
code actually is, (3) the delta against the closest published mechanisms, (4) an honest
list of what is unverified, and (5) what `burn-sct` (the old branch) still does that TSCT
does not.

## TL;DR

- **Every arXiv id cited in the lineage resolves.** `2604.00733` (SCT) is real and read
  in full; `2504.12285`, `2504.18415`, `2412.04787`, `2603.05168`, `2202.09368` all HTTP 200.
- **TSCT = SCT's parameterization + NS polar retraction instead of Stiefel QR + a whole
  quantizer family (ternary/2-bit/N:M/fp8/fp4) on the factors.** The parameterization and
  the retraction are *not* novel; the quantizer stack and the fused/batched GPU retraction
  are.
- **One real BUG:** `burn-spectral/src/lib.rs:164-166` cites Muon+ §1 for a **cubic** NS
  iteration that appears nowhere in that paper, and the cited §1 does not contain NS
  coefficients at all.
- **One real finding:** `burn-sct` is a declared dependency of `burn-spectral`
  (`Cargo.toml:33`) with **zero** source references. Dead dependency.
- **The retraction is genuinely unverified as an improvement** — the only test
  (`tsct_retract_restores_ortho`) proves it restores orthonormality, i.e. that it does
  what it says, not that doing it is better than not doing it. **Confirmed from the tree:**
  the TSCT-vs-dense A/B has never been run (`docs/protocols/AB-PROTOCOL.md:113`, queue item 2).

---

## 1. Provenance of the lineage

| id | title | status | read? |
|---|---|---|---|
| `2604.00733` | *Spectral Compact Training: Pre-Training Large Language Models via Permanent Truncated SVD and Stiefel QR Retraction* (Kohlberger, 2026-04-01) | **HTTP 200, real** | **full HTML** |
| `2504.12285` | BitNet b1.58 2B4T Technical Report | HTTP 200, real | title only |
| `2504.18415` | BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs | HTTP 200, real | title only |
| `2412.04787` | Direct Quantized Training of Language Models with Stochastic Rounding | HTTP 200, real | title only |
| `2603.05168` | Sparse-BitNet | HTTP 200, real | title only |
| `2202.09368` | Mixture-of-Experts with Expert Choice Routing | HTTP 200, real | title only |
| `2602.21545` | Muon+ (see `muon-plus.md`) | HTTP 200, real | full HTML |

Reference impl for SCT named in-tree: `EctoSpace/SCT` PyTorch (recorded in
`graphify-out/GRAPH_REPORT.md:1821`). Not fetched — the paper's own Algorithm 1 is
sufficient for the comparison below, and I flag that I did not verify our code against
that repo.

### SCT's formulas, literal (2604.00733 §3)

**Spectral representation, Eq. (1):**
```
W = U · diag(s) · Vᵀ
```
`U ∈ ℝ^{m×k}`, `V ∈ ℝ^{n×k}` **with orthonormal columns**, `s ∈ ℝ^k`. Storage
`k(m+n+1)` numbers instead of `mn`.

**Forward, Eqs. (2)-(4):**
```
(2)  h    = x · U            [b×k]   cost O(bmk)
(3)  h_s  = h ⊙ s            [b×k]   cost O(bk)
(4)  y    = h_s · Vᵀ          [b×n]   cost O(bkn)
```

**Stiefel retraction, Eq. (5):**
```
(5)  Q, R = QR(U_updated) ;   U ← Q · sign( diag(R) )
```
"After each optimizer step (AdamW)". Cost `O(mk²)`. The sign correction ensures continuity.

**Algorithm 1 (SCT training step), verbatim:**
```
0: Model with SpectralLinear layers, learning rate η, batch (x,y)
1:   Forward: ŷ = model(x)              {uses h = (x·U) ⊙ s · Vᵀ}
2:   Loss:  L = CrossEntropy(ŷ, y)
3:   Backward: compute ∇_U L, ∇_s L, ∇_V L via autograd
4:   Optimizer: AdamW step on U, s, V
5:   Retract: for each SpectralLinear layer:
6:       Q,R ← QR(U) ;  U ← Q · sign(diag(R))
7:       Q,R ← QR(V) ;  V ← Q · sign(diag(R))
```

The paper's own honest caveat, worth quoting because it bounds what SCT-style training
can claim: *"The gradients are exact with respect to the factored parameterization. They
are not identical to the gradients of a full-rank dense model, because the rank-constrained
model defines a different loss landscape."*

Paper headline results: up to 199× per-MLP-layer memory reduction at rank 32; rank
sweep on SmolLM2-1.7B (ranks 32–256, 2000 steps, A100) lands all ranks on the same loss
floor ~4.2–4.5, with **the learning-rate schedule — not MLP rank — the primary
bottleneck**; rank 128 the sweet spot; GPU memory −46% at rank 32, throughput 2×.

---

## 2. What TSCT *is*, from reading the code

### 2.1 The factorization

`SpectralLinear` — `burn-spectral/src/lib.rs:343-398`:

```rust
pub struct SpectralLinear {
    pub u: Param<Tensor<2>>,   // [in_features,  k]
    pub s: Param<Tensor<1>>,   // [k]
    pub v: Param<Tensor<2>>,   // [out_features, k]
    pub rank: usize, pub in_features: usize, pub out_features: usize,
    /* alpha, stochastic, per_column, asym, nm_n, nm_m, two_bit, fused,
       bf16_compute, quant */
}
```

So `W = U·diag(s)·Vᵀ` with `U ∈ ℝ^{in×k}`, `V ∈ ℝ^{out×k}` — **structurally identical to
SCT Eq. (1)**, with `s` a `[k]` vector (not a diagonal matrix, which is the usual
saving and is why the paper counts `k(m+n+1)`).

Construction — `lib.rs:405-439`:
- `k = rank.min(in).min(out).max(1)`; `s` init to **ones**.
- `U`, `V` init by **Householder QR of a random normal** (`qr_householder`,
  `lib.rs:639`), sliced to `[·, k]` — i.e. orthonormal columns at init, like SCT's
  orthonormal requirement but *without* an SVD (there is no dense matrix to take one of).

**Forward** — `lib.rs:553-557` (`forward_quant`) and the identical tail of `forward`
(`lib.rs:488+`):
```rust
x.matmul(u).mul(self.s.val().unsqueeze_dims(&[0])).matmul(v.transpose())
```
**This is SCT Eqs. (2)-(4), in SCT's order, exactly** — including the comment "paper
order: y = (x@U) * s @ Vᵀ" (`lib.rs:556`). Scaling after the first GEMM is the memory
win the paper describes.

### 2.2 Where TSCT is used vs a dense weight

`crates/dormouse-core/src/param.rs:36-43`:
```rust
pub enum LinearLikeInner { Tsct(SpectralLinear), Dense(burn::nn::Linear) }
```
- `LinearLike::new` → `Tsct` (`param.rs:45-58`)
- `LinearLike::dense` → `Dense` (`param.rs:60-73`)
- `LinearLike::with_tsct(.., use_tsct, ..)` picks one (`param.rs:75-86`)

Construction sites (`loop_block.rs`):
| site | shape | role |
|---|---|---|
| `expert_ffns[e].gate_up` (`:77`) | `with_tsct(d, f, rank, …)` — `[768, 2048]` at rank 64 | Expert |
| `expert_ffns[e].down` (`:78`) | `with_tsct(f, d, rank, …)` | Expert |
| `out_proj` (`:215`) | `with_tsct(d, d, rank, …)` — `[768, 768]` | Readout |
| `lm_head` (`model.rs:55`) | `with_tsct(d, v, min(..), …)` — `[768, 256]` | Head |

Presets: `rank = 64` (`small`, `base`, `mor`, `one_b`) or `96` (`nano`, `nano-fused`,
`swift50`). `small` is `d_model=768, d_ffn=2048, n_experts=3` (`configs/small.toml`).

**Padding:** `out_features` is rounded up to a multiple of 4 for cubek matmul vectorization
(`param.rs:46-50`) and sliced back after the forward (`param.rs:121-128`). Note `lm_head`'s
rank is `min(d, v) = min(768,256) = 256`… i.e. **full rank** for a `[768,256]` matrix,
so the head's "low-rank" saving is 0 by construction. Worth knowing.

### 2.3 The retraction — NS polar, not Stiefel QR

`burn-spectral/src/lib.rs:164-222`, `pub fn polar_orthogonalize(x, iters)`:
1. If `rows > cols`, transpose (canonical wide form) — `:168-173`.
2. `g = m·mᵀ` on the small side `[c,c]` — `:178`.
3. **σ_max by power iteration** (5 iters, `POWER_ITERS` at `:135`) with a Rayleigh
   quotient `σ² = vᵀgv / vᵀv` — `:179-199`.
4. Rescale `m /= σ·1.05` — `:200-201`. The comment (`:174-177`) is a real measured
   finding: a Frobenius/√k prescale does **not** bound σ_max (Bai–Yin: σ_max ≈ 2√n for a
   Gaussian n×n), and NS then diverges — "measured: polar([512,512], 3) -> max entry ~1e14".
5. **Cubic** NS, `a,b,c = (15/8, −5/4, 3/8)` — `:204-210`:
   `m ← a·m + (b·(mmᵀ) + c·(mmᵀ)²)·m`.
6. Un-transpose — `:211-215`.

`SpectralLinear::retract` (`lib.rs:612-619`) applies it to `U` and `V` via
`polar_retracked` (`:154-161`), which **detaches and mirrors `is_require_grad`** rather
than forcing it. Batched variant `polar_orthogonalize_batched` (`:227-272`) with
`retract_batched` (`:275-292`): identical math, σ_max and norms batched, **zero host
syncs** where the scalar path has 7 `into_scalar` per factor.

Trainer wiring: `lib.rs:1315-1317` calls `model.retract_tsct(cfg.retract_iters)` every
`retract_every` steps; defaults **`retract_every=1, retract_iters=3`** (`lib.rs:160`).
`retract_tsct` walks `expert_ffns[].{gate_up,down}` + `out_proj` (`loop_block.rs:170-176`).

**Shape check (done by hand):** every TSCT factor in the live model is *tall*
(`[in,k]`, `[out,k]` with k=64 ≪ 768/2048/256), so the transpose branch always fires and
the iteration produces orthonormal **columns** — which is exactly what `ortho_error`
measures. ✅ consistent. The non-transposed branch would produce orthonormal *rows*
instead, and `ortho_error` would then be measuring the wrong thing; that case is not
reachable at current ranks, so it is latent, not live.

### 2.4 `max_ortho` and the one-way fp32 latch

`param.rs:190-201`:
```rust
(burn_spectral::ortho_error(&u) / ku).max(burn_spectral::ortho_error(&v) / kv)
```
with `ku = u.dims()[1]`, `kv = v.dims()[1]` (the rank). `burn-spectral/src/lib.rs:295-315`
computes `‖UᵀU − I‖_F` **on the host** via `into_data()` — a full device→host readback.

- **Threshold:** `1e-3`, checked at `train/src/lib.rs:1331-1337`, **every 500 steps**.
- **One-way:** `&& !ortho_fp32` (`:1331`); on trip it sets `set_quant_all(Fp32)` and
  `ortho_fp32 = true` (`:1334-1336`) — the latch is permanent, checks stop, and it is
  **persisted in the checkpoint** (`CKPT_ORTHO_FP32`, `:553`, read back `:635`, re-applied
  `:1072`; test at `:2113`). ✅ ADR-0011 satisfied (COUNTED, and the reader is told).
- **The per-entry normalization is a bug fix, documented.** The raw F-norm scales ~k,
  which put the 1e-3 threshold *below* the NS-3 convergence floor (~4e-3 raw at r=64), so
  the fallback fired at step 0 on every fresh run and the factor-quant forward never
  engaged. Dividing by k gives ~6e-5 floor vs a 1e-3 real-drift bound
  (`param.rs:176-189`, measured 2026-09-04 by the `ortho_probe`).

### 2.5 The quantizer stack (the genuinely non-SCT part)

`SpectralLinear` carries a *family* of forward-path quantizers on the **U/V factors only**
(masters stay fp32):

| knob | fn | mechanism | cited lineage |
|---|---|---|---|
| `alpha` | `set_alpha` (`:441`) | `ste_ternary_annealed` (`:70-79`): `w + α(tern(w)−w).detach()` | BitNet `2504.12285` STE |
| — | `ternarize` (`:51-57`) | `sign(w)·mean(|w|)`, dead zone at `0.7·mean` | absmean STE |
| `stochastic` | `ste_ternary_stochastic` (`:103-106`) | `sign(w)·scale·Bernoulli(|w|/scale)`, `E[w_t]=w` | stochastic rounding `2412.04787` |
| `per_column` | `ternarize_per_column` (`:112+`) | per-column scale, not global | (in-tree rationale: weak SVD columns collapse under a global dead zone) |
| `nm_n/nm_m` | `set_nm` (`:481`) | N:M sparsity on factors, dual-STE | Sparse-BitNet `2603.05168` |
| `two_bit` | `set_2bit` (`:490`) | 5-level `{−2s,−s,0,s,2s}` | BitNet-style |
| `quant` | `set_quant` (`:607`) | `QuantFormat::{Fp32,Bf16,Fp16,Fp8,Fp4}` (`:318-338`), per-row absmax/absmean + STE | fp8/fp4 |
| `bf16_compute` | `forward_quant_bf16` (`:563`) | tensor-core fwd, fp32 graph | machine fact |

Also `SpectralMoE` (`lib.rs:690-760`): rank-1 ternary experts `[in, M·r]` / `[out, M·r]`,
top-k router, optional **Expert-Choice** routing (`2202.09368`). `moe_fused.rs` (3243 lines)
is the fused CUDA path. **Per `docs/archive/research/2026-09-27-fused-inventory-precision.md:522`,
`SpectralLinear::forward` never reads `self.fused` and `SpectralMoE` is never constructed
— I did not re-verify that inventory in this pass; treat it as a prior reading.**

### 2.6 Optimizer routing of TSCT factors

`routing.rs:90-101` — the policy, one match:
```rust
(Expert, Factor) if factors_fallback => Rest,
(Expert | Readout, Factor)            => Muon,
(Head, Factor | Scale | DenseWeight | DenseBias) => Rest,
(Expert | Readout, Scale | DenseWeight | DenseBias) => Rest,
```
So the **low-rank U/V factors go to Muon+ (ColRow, NS)**, the `[k]` scale and the head go
to AdamW. Justification at `routing.rs:76-80`: NS costs ~3 matmuls on the full `[m,n]`
and fp32 NS on `[d,d]` measured ~40 s/step, while the `[d,r]`/`[r,f]` form is ~1000×
cheaper. `--factors-fallback` moves the expert factors to the fallback optimizer.

---

## 3. Delta against the closest published mechanisms

| # | file:line | Published mechanism says | TSCT does | Verdict |
|---|---|---|---|---|
| **T1** | `burn-spectral/src/lib.rs:164-166` | Muon+ (2602.21545) §1 defines `Ortho(·)` **abstractly** and prints **no NS coefficients**; its own coefficients live in Jordan's `muon.py` as the **quintic** `(3.4445, −4.775, 2.0315)` | Doc comment says "Newton-Schulz polar iteration (**Muon+ 2602.21545 §1**) … with the optimal cubic coefficients"; the code uses `(15/8, −5/4, 3/8)` (`:205`, `:256`) — **not Muon+'s, and not in the cited paper** | **BUG** (mis-citation). The *math* is fine and deliberate (cubic, σ_max-scaled, for a **retraction** not an optimizer preconditioner); only the attribution is wrong. Fix: cite Higham 2008 / "optimal cubic", drop the Muon+ id. |
| **T2** | `burn-spectral/src/lib.rs:205`, `:256` | Muon+ = quintic `(3.4445, −4.775, 2.0315)`, 5 steps | cubic `(15/8, −5/4, 3/8)`, `iters=3` by default | **DELIBERATE** and *correct for the purpose*: a retraction wants the nearest semi-orthogonal matrix (cubic converges to exact `UVᵀ`); Muon+ deliberately uses a quintic that does **not** reach `UVᵀ` (Jordan's own docstring: `S_ii ~ Uniform(0.5,1.5)`, "turns out not to hurt"). Using the quintic here would be the bug. Distinct role, distinct coefficients — but see T1, the code claims the wrong provenance. |
| **T3** | `burn-spectral/src/lib.rs:178-201` (σ_max power iteration) | SCT Eq. (5) is QR, which needs no scaling | NS needs σ<√3; prescaled by Rayleigh-quotient σ_max × 1.05 | **DELIBERATE** — and it is *better* than a plain Frobenius prescale, with the divergence case documented (`:174-177`). This is our own contribution to the retraction and is the reason NS is viable here at all. |
| **T4** | `burn-spectral/src/lib.rs:612-619` | SCT Eq. (5): `Q,R = QR(U); U ← Q·sign(diag(R))`, after **every** optimizer step | `iters`-step NS polar, default `retract_every=1, retract_iters=3` | **DELIBERATE** (replace). NS avoids the `O(mk²)` QR and the host round-trip. Cost: an *approximate* retraction (3 cubic steps), not the exact QR. **Convergence is not guaranteed at 3 steps** — see §4. |
| **T5** | `burn-spectral/src/lib.rs:295-315` `ortho_error` | SCT (and `burn-sct/lib.rs:120-139`) both compute `‖UᵀU − I‖_F` **raw** | same raw metric here, but the **caller** divides by k (`param.rs:196-200`) | **DELIBERATE** — the normalization is the documented 2026-09-04 bug fix, not present upstream. Correct and load-bearing. |
| **T6** | `burn-spectral/src/lib.rs:553-557` | SCT Eqs. (2)-(4): `y = (x·U) ⊙ s · Vᵀ` | identical, same order, same comment | **BENIGN** ✅ — the forward *is* SCT's. Not a delta. |
| **T7** | `burn-spectral/src/lib.rs:405-439` (QR-of-random init, `s=1`) | SCT initializes from a real SVD of a weight | no dense matrix exists; orthonormal random via Householder QR | **DELIBERATE** (unavoidable) — consequence of training natively in spectral form. |
| **T8** | `burn-spectral/src/lib.rs:441-613` (quantizer family) | **SCT has no quantization of any kind.** `burn-sct` has ~2 hits for "tern/quant". | ternary / annealed / stochastic / per-column / asymmetric / 2-bit / N:M / fp8 / fp4 / bf16 on the factors | **OURS — the main non-SCT content.** Lineage is BitNet (`2504.12285`, `2504.18415`) + stochastic rounding (`2412.04787`) + Sparse-BitNet (`2603.05168`). |
| **T9** | `burn-spectral/src/lib.rs:227-292` (`polar_orthogonalize_batched`, `retract_batched`) | no published analogue here | group-by-shape, stack to `[B,m,k]`, one sync-free call; the scalar path has 7 `into_scalar` per factor | **OURS.** Mathically identical by construction. But **never called by the trainer** — see §4. |
| **T10** | `burn-spectral/Cargo.toml:33` `burn-sct = { path = "../burn-sct" }` | — | `grep` over `burn-spectral/src/**/*.rs` finds **zero** `burn_sct` / `sct::` references | **BUG** (minor): dead dependency. It also drags a second, divergent retraction implementation into the build graph. |
| **T11** | `crates/dormouse-core/src/param.rs:1` | — | header comment says "TSCT linear **via burn-sct SpectralLinear**" — it is not; it is `burn_spectral::SpectralLinear` (`param.rs:9`) | **BUG** (stale comment) — the same confusion T10 encodes, in prose. Directly contradicts the owner's "sct is the old paper-based one, we invented tsct". |
| **T12** | `model.rs:55` `cfg.rank.min(d).min(v)` | — | `lm_head` is `[768, 256]` at rank **256** = full rank | **BENIGN** (correct by the min-clamp) but worth knowing: the head is not compressed at all, and it is routed to AdamW anyway (`routing.rs:96`). |
| **T13** | `train/src/lib.rs:1315-1317` + `:1331-1337` | SCT: retract every step, no monitor, no fallback | retract every step (default) + 500-step monitor + **one-way persisted fp32 latch** | **OURS** — and it is the correct ADR-0011 shape (COUNTED, printed, persisted). Stronger than the paper's design. |
| **T14** | `research/…-fused-inventory-precision.md:522` (prior reading) | — | `SpectralLinear::forward` never reads `self.fused`; 33/36 spectral tests fail; `SpectralMoE` never constructed | **UNVERIFIABLE in this pass** (I did not re-run or re-read the inventory). Carried forward as a prior, not re-confirmed. |

---

## 4. What is UNVERIFIED

Read from the tree, not assumed.

1. **Has TSCT ever been A/B'd against a dense FFN? NO.**
   `docs/protocols/AB-PROTOCOL.md:113` — queue item 2, `--set use_tsct=false`, "do the TSCT factors,
   the polar retraction and the quant machinery earn ~1000 lines?". The file's own header
   (`:8-9`) says **"No arm in this queue has been judged"**. The A/B is *constructible*
   (`LinearLike::dense` exists, `param.rs:60-73`, and the comment there says the dense
   variant was added precisely to make the question answerable) but **unrun**.
2. **Is the retraction numerically better than doing nothing? UNVERIFIED.**
   The only test is `tsct_retract_restores_ortho` (`train/src/lib.rs:2026-2041`): it
   scales `U` by 3.0 and asserts `after < before/10 && after < 1e-3`. That proves the
   retraction *works* (it is a retraction) — it says nothing about training quality, loss,
   or BPB. **No evidence in the tree that it helps.**
3. **Does 3 cubic NS steps actually converge to the Stiefel manifold at our shapes?**
   The metric exists and the latch would catch drift, but the 3-step retraction is
   *approximate* and no test compares it to the exact QR. `burn-sct` has the exact
   `safe_qr` (`qr.rs:509` implements the `sign(diag(R))` correction; `qr_cuda.rs:559`
   `retract_cuda`) — **the exact reference is in the tree and is not used to check the
   approximate one.** That comparison is the cheapest unrun check available.
4. **`retract_batched` is dead in production.** `grep` finds it only in its own definition
   and its own tests (`burn-spectral/src/lib.rs:275, 1327-1432`); the trainer goes through
   `SpectralLinear::retract` → `polar_retracked` → `polar_orthogonalize`, the **7-`into_scalar`
   per factor** path. The claimed sync saving is not being collected. This is the same
   class as the `batched` vs `loop` 4.1–4.8× figure: real math, unwired call site.
   Consistent with ADR-0019's "if you add an arm, add its counter" — there is no counter
   for this one.
5. **The `max_ortho` 1e-3 threshold is a project constant, not a derived one.** Fresh init
   ~1.4e-4, retract floor ~6e-5 per-entry, threshold 1e-3 (`param.rs:176-189`). The margin
   is asserted, not swept.
6. **The factor-quant forward has never been shown to help.** `AGENTS.md` §2.3 records that
   the fallback fired on every fresh run before 2026-09-04, so **no pre-fix run ever
   engaged it**; whether the post-fix path is better than plain fp32 factors is unmeasured.
7. **T14** — prior reading, not re-verified here.

---

## 5. `burn-sct` — the old branch. What it does that TSCT does not.

`burn-sct` is a faithful, careful implementation of **SCT 2604.00733 as published**:
`W = U·diag(s)·Vᵀ` (Eq. 1), `y = (x·U) ⊙ s · Vᵀ` (Eqs. 2-4), **exact Stiefel QR retraction
with the `sign(diag(R))` continuity correction** (Eq. 5, `qr.rs:509`; CUDA path
`qr_cuda.rs:559`), plus a fused CUDA forward (`qr_cuda.rs`), threaded CPU retraction
(`lib.rs:81-118`), and `compression_ratio`/`flops`/`param_count` reporting
(`lib.rs:225-232`). Its retraction is **exact and orthogonal**, where TSCT's is 3
approximate cubic steps.

| capability | `burn-sct` | `burn-spectral` (TSCT) |
|---|---|---|
| parameterization `U·diag(s)·Vᵀ` | ✅ | ✅ (same) |
| forward | ✅ SCT order | ✅ SCT order |
| **exact Stiefel QR retraction** | ✅ **yes, with sign correction** | ❌ (approximate NS) |
| **GPU-side retraction** | ✅ `qr_cuda::retract_cuda` | ✅ (NS, sync-heavy) |
| **sync-free batched retraction** | ❌ | ✅ `retract_batched` (**unwired**) |
| `from_dense` — SVD of a **trained dense** weight into factors | ✅ `lib.rs:141-223`, `from_dense_with_iters` (GPU SVD, `:148`) | ❌ |
| factor quantization (ternary/2-bit/N:M/fp8/fp4/bf16) | ❌ (~2 hits total) | ✅ the whole family |
| MoE / expert routing | ❌ | ✅ `SpectralMoE`, Expert-Choice |
| `compression_ratio` reporting | ✅ | ❌ (has `param_count`/`flops`, no ratio) |
| forward is exercised on autodiff | ✅ | ❌ (prior reading: 33/36 tests fail, see T14) |

**Does anything still need `burn-sct`?**

- **In the training path: no.** `burn-spectral` declares it (`Cargo.toml:33`) and never
  calls it; `dormouse-core` depends only on `burn-spectral` (`param.rs:9`). It is dead
  weight in the build graph — **removable today, zero code change** (T10).
- **As a reference, yes, and it is the most valuable thing in it.** It is the only
  *exact* Stiefel retraction in the tree, and it is a faithful transcription of the
  published Eq. (5) including the continuity sign correction. That makes it the natural
  oracle for check 3 in §4 (NS-3 vs `safe_qr` at our shapes), which is currently the
  cheapest unrun verification in the whole TSCT story.
- **One capability TSCT genuinely lacks: `from_dense`.** If a future arm wants to start
  from a pretrained dense checkpoint in spectral form, that lives only in `burn-sct`.
  Not needed for any queued arm today.

**Recommendation (not acted on — read-only pass):** keep `burn-sct` as a *test-only*
oracle and stop linking it into `burn-spectral`; add one test that pins NS-3 against
`safe_qr` on a `[768,64]` factor. That single test discharges item 3 of §4, which is the
only part of the retraction claim currently resting on nothing.

---

## 6. Severity ranking (subject 2)

1. **T1 — `burn-spectral:164-166` cites Muon+ §1 for a cubic NS that is not in that
   paper.** A fabricated-looking citation in a *real* paper's shape: exactly the failure
   mode ADR-0020 exists to stop, and it is one line to fix. (The math is right; the
   provenance is wrong.)
2. **§4.2 + §4.3 — the retraction has never been shown to help, and never compared to
   the exact retraction sitting unused in the same vendor tree.** The A/B is unrun
   (queue item 2), and the one test that exists only proves the function is a function.
3. **T10 + T11 — `burn-sct` is a live Cargo dependency with zero call sites, and
   `param.rs:1` still describes TSCT as "via burn-sct".** Two minutes to fix, and it
   removes the exact ambiguity the owner asked about.
4. **§4.4 — `retract_batched` is unwired**, so the sync-free path and its 4.1–4.8×
   class of saving are not in production, and no counter says so.
