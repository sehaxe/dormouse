# Independent review of `docs/papers/muon-plus.md` and `docs/papers/tsct.md`

**Reviewer pass, 2026-09-29.** Read-only. No GPU, no build, no test. Primary sources
fetched directly (arXiv HTML for `2602.21545v1/v3` and `2604.00733`, GitHub API for
`K1seki221/MuonPlus`); code read in-tree. One file written; nothing else touched.

**Headline:** claim 3 is **confirmed and proved** (the branch is unsatisfiable, no
counterexample exists). Claim 5's **verdict survives but its evidence is factually
wrong** — the paper *does* print the coefficients, in v3, and one of them is the
exact triple `burn-spectral` uses. Claim 7's dependency half is **refuted**. The
report's biggest miss is a live, unguarded, 112-host-sync-per-step path it found but
did not count.

---

## 0. Verdict table

| # | claim | verdict |
|---|---|---|
| 1 | every arXiv id cited resolves | **TRUE** — 10/10 HTTP 200, titles match |
| 2 | `github.com/K1seki221/MuonPlus` is the paper's code and exists | **TRUE** |
| 3 | `burn-muon-plus/src/lib.rs:146` `if nc*4 < nr` is unsatisfiable; the factored branch and `ns_combine_cuda` are dead | **TRUE — proved below** |
| 4 | the `3.6x on [8192,512]` comment is backwards on FLOPs | **TRUE in direction, report's own number is wrong** (1.4545x, not 1.48x) |
| 5 | `burn-spectral:164-166` mis-cites Muon+ §1; the paper "prints no NS coefficients" | **VERDICT TRUE, EVIDENCE FALSE** — v3 App. D.1 and D.3 print both coefficient families, D.3 terminating at exactly `(1.875, −1.25, 0.375)` |
| 6 | TSCT has never been shown to help | **TRUE** |
| 7 | `burn-sct` is a declared dependency with zero call sites; `param.rs:1` stale | **SPLIT** — `param.rs:1` stale: TRUE. Dead dependency: **FALSE** (it is a `[dev-dependencies]` entry and `examples/tsct_diag.rs` uses it) |
| 8 | `retract_batched` is called only from tests | **TRUE** — and the report understates the consequence by ~2 orders of magnitude |
| 9 | the `37%` is v3-only and is time-to-target-loss | **TRUE** — and the string is in the *crate* README, not the project README |

---

## 1. Claim 3 — the proof

**CONFIRMED. The condition is unsatisfiable. No counterexample exists, and none
can exist.**

`vendor/burn-fused/crates/burn-muon-plus/src/lib.rs:127-146`:

```rust
127:  pub fn orthogonalize<const D: usize>(&self, g: Tensor<D>) -> Tensor<D> {
128:      let dims = g.dims();
129:      let (rows, cols) = (dims[D - 2], dims[D - 1]);
130:      let (mut x, transposed) = if rows > cols {
131:          (g.swap_dims(D - 2, D - 1), true)      // <-- canonicalising transpose
132:      } else {
133:          (g, false)
134:      };
...     // Frobenius normalise — a scalar op, dims are unchanged by it
141:      let [nr, nc] = x.dims()[D - 2..D].try_into().unwrap();
146:      if nc * 4 < nr {
```

**The argument, in full.**

1. `rows` and `cols` are `g.dims()[D-2]` and `g.dims()[D-1]`. `nr` and `nc` are
   `x.dims()[D-2]` and `x.dims()[D-1]`, read at `:141` from the *possibly
   transposed* `x`.
2. `swap_dims(D-2, D-1)` at `:131` exchanges exactly those two axes. So the only
   thing that can change between `(rows, cols)` and `(nr, nc)` is their order.
3. The only operation between `:131` and `:141` is `x = x.div(norm.unsqueeze())`
   (`:138`), a rank-0 broadcast. It does not permute axes.
4. Therefore, case-splitting on the branch at `:130`:
   - **`:132` (`rows <= cols`, no transpose):** `x is g`, so `(nr, nc) = (rows, cols)`
     and `rows <= cols` gives **`nc >= nr`**.
   - **`:130` (`rows > cols`, transpose):** `x` has dims `[cols, rows]`, so
     `(nr, nc) = (cols, rows)` and `rows > cols` gives **`nc > nr`**.
   In both cases **`nc >= nr` holds unconditionally.**
5. `nc >= nr` and `nr >= 0` (a `usize` tensor dimension) give
   `nc * 4 >= 4 * nr >= nr`, i.e. `nc * 4 < nr` is **false**. ∎

**Corollaries, each checked against the code:**

- **No counterexample is constructible.** The predicate does not depend on the data
  at all — only on `rows`/`cols`, and the canonicalising transpose guarantees the
  invariant. It is not "rarely true", it is a constant `false`.
- **Wrap-around is not an escape.** `nc * 4` is `usize`; it wraps only for
  `nc > usize::MAX / 4` (~4.6e18), a tensor that cannot exist. On the shapes this
  crate is actually fed, no overflow.
- **`D == 1` is not a counterexample either.** `dims[D - 2]` with `D == 1` is
  `dims[usize::MAX]` — a panic (debug) or an out-of-bounds index (release). It never
  reaches `:146`.
- **Batched `D > 2` is not a counterexample.** `x.dims()[D-2..D]` is the last two
  axes and `swap_dims(D-2, D-1)` swaps the last two axes, so step 4 holds verbatim
  per batch element. There is no leading-axis interaction.
- **The whole `:147-170` block is dead in production.** `grep -rn ns_combine_cuda`
  over the repo returns exactly six hits: the definition (`fused_kernels.rs:89`), the
  unreachable call (`lib.rs:154`), and three inside `#[cfg(test)]` helpers plus one
  inside `fn fused_match_tensor` (`:272`, `:331`, `:358`, all `fused_kernels.rs`).
  **The only live call site of a tested CUDA kernel is a call that cannot execute.**

**The report is right on D1 and I add one thing it missed:** the *comment* does not
merely overstate a measurement — it describes an optimisation that is both
**unreachable** and **algebraically redundant**. Factored and direct compute the
*same polynomial*:

- factored: `t1 = (XXᵀ)X`, `t2 = (XXᵀ)²X`, result `a·X + b·t1 + c·t2`
- direct: `poly = b·(XXᵀ) + c·(XXᵀ)²`, result `a·X + poly·X = a·X + b·(XXᵀ)X + c·(XXᵀ)²X`

Identical, term for term. So the branch is not a different method — it is a
different *evaluation order* of one method, with the fused kernel attached to the
worse order. Deleting `:147-170` and keeping the fused elementwise combine
hoisted into the live loop loses nothing.

---

## 2. Claim 4 — the FLOP arithmetic, redone

The report's *direction* is right and its *number* is wrong.

`x` in the dead branch is `[nr, nc]` with `nc >= nr`, and the `[·,·]` shapes are:
`xx = [nr,nr]`, `t1 = [nr,nc]`, `t2 = [nr,nc]`.

| form | matmuls | MACs |
|---|---|---|
| factored (`:147-170`) | 3 | `3 · nr²·nc` |
| direct (`:172-178`) | 3 | `2 · nr²·nc + nr³` |

Ratio `= 3nc / (2nc + nr)`. At the comment's own shape `nr=512, nc=8192`:
`3·8192 / (2·8192 + 512) = 24576 / 16896 = **1.4545x**`. The report says
**1.48x** (`muon-plus.md:171`) — wrong by 2%, in a cell whose entire purpose is to
correct an arithmetic claim in a comment. Not load-bearing, but it is the one
number in the report a reader would most likely copy out.

**Two things the report missed in the same comment (`lib.rs:142-145`):**

1. **"one fewer matmul launch" is false.** The comment says the direct form wins on
   squares because it is "~10% faster on squares (same FLOPs, one fewer matmul
   launch)". Both branches issue **three** matmuls (`fused_kernels.rs` aside:
   factored = `xx`, `t1`, `t2`; direct = `xx`, `xx2`, `poly@x`). The FLOPs are also
   not "the same" — the direct form carries an extra `nr³` by construction. The
   comment is wrong on *both* of the grounds it offers.
2. **The report's own recommended fix is the wrong one.** `muon-plus.md:237`
   offers "either delete the branch or invert to `nc * 4 > nr`". Inverting activates
   the factored branch precisely on the wide matrices where the report itself
   computed 1.45x *more* FLOPs, and the fused kernel is the only thing that branch
   buys. **Delete is the correct fix; invert would ship a slower path on purpose.**
   A reader who takes the second option makes things worse.

---

## 3. Claim 5 — I read the paper. The report's evidence is wrong.

The report asserts, in two places, that the paper defines `Ortho(·)` abstractly and
**prints no NS coefficients at all**, and that the cubic is "not in the cited
paper" (`muon-plus.md:49-50, 174`; `tsct.md` T1, and `tsct.md:354` calls it "a
fabricated-looking citation in a real paper's shape").

**All of that is false for the current version, v3 (2026-05-14).**

`https://arxiv.org/html/2602.21545v3`, §2.1, defines the step generically:

> Define one Newton-Schulz step by
> `Q := aM + b(MMᵀ)M + c(MMᵀ)²M`, with coefficients `a, b, c ∈ ℝ`. Introduce the
> quintic polynomial `φ(x) := ax + bx³ + cx⁵`.

— so far the report is right, and this is what §1/§2 of the paper contains. But
**v3 has an Appendix D that the report never read**, and it prints both families:

> **D.1 Jordan Coefficients [15]** — In [15], the coefficients are set to
> `(a, b, c) = (3.4445, −4.7750, 2.0315)`.

> **D.3 PolarExpress Coefficients [2]** — `[2]` uses iteration-dependent
> coefficients: `{(aₜ, bₜ, cₜ)}ₜ₌₁⁸ = [(8.2872…, −23.596…, 17.300…), (4.107…, …),
> (3.949…, …), (3.318…, …), (2.301…, …), (1.8913…, −1.2680…, 0.37680…), (1.87500…,
> −1.25000…, 0.37500…), **(1.875, −1.25, 0.375)]`**, with subsequent coefficients
> numerically equal to `(1.875, −1.25, 0.375)`.

`grep -c` over the v3 text: `3.4445` → 2, `4.775` → 2, `2.0315` → 2,
`1.875` → 6, `0.375` → 7. Over **v1**: **all zero**. v1 has appendices A–C; v3 has
**A–G**. v1 text 40,814 chars; v3 text 82,430 chars.

**So the verdict stands and the diagnosis inverts.** The defect at
`burn-spectral/src/lib.rs:164-166` is real — a mis-citation is a mis-citation under
ADR-0020. But:

- the coefficient triple `(15/8, −5/4, 3/8) = (1.875, −1.25, 0.375)` is **printed in
  the cited paper**, v3, in Appendix D.3, as the terminal fixed point of the
  PolarExpress schedule;
- the paper's own §1 sentence at that location reads: *"In practice, the Newton–Schulz
  iteration process **Higham (2008)** is commonly used to approximate the SVD"* —
  so the section the code points at does name the correct authority for the exact
  algorithm being used;
- the actual error is (i) the **section pointer** is wrong — the coefficients are in
  App. D, and the NS-step definition is in §2.1, not §1, in *both* v1 and v3 — and
  (ii) the **method name** is wrong: the triple is labelled *PolarExpress*
  (Amsel et al. 2025, `2505.16932`), not *Muon+*.

**And the report's recommended fix is worse provenance than the status quo.**
`tsct.md:252` says "Fix: cite Higham 2008 / 'optimal cubic', drop the Muon+ id".
`(15/8, −5/4, 3/8)` *is* Higham's optimal cubic for the polar factor, so that half is
fine — but dropping the arXiv id discards a **verifiable** pointer to the exact
triple in the exact paper, in exchange for a prose appeal to a 2008 textbook. The
honest one-line fix is the opposite direction: keep the id, move the pointer to
**App. D.3**, and attribute the coefficients to **PolarExpress / Amsel et al.
(2505.16932)**, noting it coincides with the Higham optimal cubic. That is a
two-token edit with a checkable target; the report's version discards the only
source in the paper that states the number.

**Root cause of the report's error, which is the interesting part:** `muon-plus.md:57`
writes *"From the **v3** PDF→HTML (identical in v1/v2; **verified by diff**, only the
abstract and §3.4 numbering changed)"*. That diff was not run, or was run on the
wrong artifacts. v1 is 40.8K chars with appendices A–C; v3 is 82.4K with A–G, and
**both coefficient families live in appendices that exist only in v3**. The report
read v3's §1–§3 and assumed the appendices matched. Every downstream claim about
"the paper never prints coefficients" is downstream of that one unverified diff.

---

## 4. Claims 1, 2, 6, 8, 9

**Claim 1 — TRUE.** All ten ids resolve, titles match:

```
2602.21545 -> 200  MUON+: Towards More Effective Muon via One Additional Normalization St…
2604.00733 -> 200  Spectral Compact Training: Pre-Training Large Language Models via Perm…
2504.12285 -> 200  BitNet b1.58 2B4T Technical Report
2504.18415 -> 200  BitNet v2: Native 4-bit Activations with Hadamard Transformation…
2412.04787 -> 200  Direct Quantized Training of Language Models with Stochastic Rounding
2603.05168 -> 200  Sparse-BitNet: 1.58-bit LLMs are Naturally Friendly to Semi-Structured…
2202.09368 -> 200  Mixture-of-Experts with Expert Choice Routing
2502.16982 -> 200  Muon is Scalable for LLM Training
2505.16932 -> 200  The Polar Express: Optimal Matrix Sign Methods and Their Application…
2601.23000 -> 200  Mano: Restriking Manifold Optimization for LLM Training
```

**Claim 2 — TRUE.** `https://github.com/K1seki221/MuonPlus` → HTTP 200; GitHub API
returns `{"id": 1166260539, "full_name": "K1seki221/MuonPlus", "private": false}`.

**Claim 6 — TRUE.** `docs/AB-PROTOCOL.md:113` is queue item 2
(`--set use_tsct=false`, "do the TSCT factors, the polar retraction and the quant
machinery earn ~1000 lines?"); the file's own header at `:8-9` reads "**No arm in
this queue has been judged**". `train/src/lib.rs:2026-2041`
(`tsct_retract_restores_ortho`) scales `u` by 3.0 and asserts
`after < before/10 && after < 1e-3` — a statement that the function is a function.
The report is exactly right that this is the *only* evidence and that it is
sufficiency-shaped, not quality-shaped. Note the test also runs on
`LinearLike::new(64, 64, 16, ..)` — a **square** `[64,64]` factor — which is the one
shape the live model never uses (§2.3 of `tsct.md` notes the transpose branch always
fires in production; the test exercises the non-transposed branch). So the test does
not even cover the shape that ships.

**Claim 8 — TRUE, and worse than reported.** `grep -rn retract_batched` over the
whole repo (including `crates/`): hits at `burn-spectral/src/lib.rs:225` (doc
cross-ref), `:275` (definition), `:1327`/`:1348`/`:1371`/`:1380`/`:1422`/`:1432`
(all inside `#[cfg(test)]`). **Zero production call sites.** The "7 `into_scalar`
per factor" count is also correct — I counted the call sites in
`polar_orthogonalize`: three textual sites, one of which (`:186`) sits inside
`for _ in 0..POWER_ITERS` with `POWER_ITERS = 5`, so `5 + 1 + 1 = 7`. See §7.1 for
why that number matters far more than the report says.

**Claim 9 — TRUE, with a file-attribution correction.** v1 contains **zero**
occurrences of `37.1`. v3 §3.3: *"Muon+ has nearly the same per-step runtime and
memory cost as Muon… we report the wall-clock time required to reach the same target
loss"*, and Table 5 is literally a **step-count** ratio: GPT-Base `3447` Muon steps
vs `2515` Muon+ steps → `↑37.1%`. So it is neither wall-clock nor per-step; it is
**optimizer steps to a fixed loss**. The report says "time-to-target-loss, not
per-step" — correct, and the table makes it sharper than the report did.

**But the file is wrong.** `README.md` (project root) contains **no** occurrence of
`37%` or `37.1`. The string lives at
`vendor/burn-fused/crates/burn-muon-plus/README.md:30`. Likewise the retracted
`96×/84×`/`3.6×` numbers are in that same **crate** README (`:61-62`, `:75-78`), not
in the project `README.md`'s "Performance" section — which is the two-row
`benches/history.tsv` canary table. `muon-plus.md:183` and `:184` both attribute to
`README.md` and `D15` even names the section, which a reader will resolve to the
wrong file. The *verdicts* survive; the *citations* do not.

---

## 5. Claim 7 — split verdict, with the dependency half **refuted**

**TRUE:** `burn-sct/src/qr.rs` does implement the paper's `sign(diag(R))` continuity
correction (`:18-19` names it, `:300-301` and `:351-352` implement it), and
`crates/dormouse-core/src/param.rs:1` does read
`//! param - TSCT linear via burn-sct SpectralLinear` while `param.rs:9` imports
`burn_spectral::SpectralLinear`. T11 is a correct BUG.

**FALSE: the "dead dependency" (T10).** `burn-spectral/Cargo.toml:33` is:

```
27: [dev-dependencies]
...
33: burn-sct = { path = "../burn-sct" }
```

It is a **`[dev-dependencies]`** entry, not a `[dependencies]` entry — the section
opens at `:27` and the `[dependencies]` block ends at `:25`. Three consequences the
report misses because its grep was scoped to `burn-spectral/src/**/*.rs` and so never
saw `examples/`:

1. **It does not drag `burn-sct` into any downstream build graph.** A dev-dependency
   is not built for anything that depends on `burn-spectral`. The report's
   "**removable today, zero code change**" and "it also drags a second, divergent
   retraction implementation into the build graph" are both false statements about
   how Cargo resolves dev-dependencies.
2. **It is not unused.** `burn-spectral/examples/tsct_diag.rs:61-62, 83-88, 105-110`
   constructs `burn_sct::SctLinear` and `burn_sct::SctConfig` and runs them against
   `SpectralLinear` in a mini-GPT ("does TSCT actually learn? … dense vs
   SpectralLinear vs SpectralMoE, same params budget"). That is a **live, working
   consumer** of the exact-Stiefel oracle.
3. **It is already the shape the report recommends.** `tsct.md:344-347` says "keep
   `burn-sct` as a *test-only* oracle and stop linking it into `burn-spectral`". A
   dev-dependency consumed by an example **is** a test-only oracle. T10 is
   flagging the recommended state as a defect.

**T10 should be deleted, not downgraded.** It is a BUG verdict on a correct,
deliberate arrangement, which is the failure mode you asked me to hunt.

---

## 6. Delta spot-check

Counts as they actually stand in the report: **Muon+ 4 BUG / 5 DELIBERATE / 8
BENIGN (17 rows)**; **TSCT 3 BUG / 5 DELIBERATE / 2 BENIGN / 3 OURS / 1
UNVERIFIABLE (14 rows)**. (The task brief's "4 BUG / 4 DELIBERATE for TSCT" does not
match the file; the file's own TL;DR says "**One** real BUG" while its table marks
**three**. That internal inconsistency is itself a small defect — the summary
under-reports its own findings by 2x.)

### BUG verdicts checked (6 of 7)

| id | verdict | my result |
|---|---|---|
| **D1** | BUG | **CONFIRMED**, proved in §1. Strongest finding in either report. |
| **D2** | BUG | **CONFIRMED** in direction; report's ratio 1.48x is **wrong** (1.4545x). Also missed that the comment's "one fewer matmul launch" is false and that the report's own suggested fix inverts toward the slower branch. |
| **D5 / T1** | BUG | **VERDICT CONFIRMED, EVIDENCE REFUTED** — see §3. The paper prints `(1.875, −1.25, 0.375)` in v3 App. D.3. |
| **D14** | BUG | **CONFIRMED** in substance (v3-only, steps-to-target-loss) but the cited file is the **crate** README, not the project README. |
| **T10** | BUG | **REFUTED** — dev-dependency, used by `examples/tsct_diag.rs`. See §5. |
| **T11** | BUG | **CONFIRMED** — `param.rs:1` vs `param.rs:9`. |
| *(not in report)* | — | **MISSED BUG** — see §7.1: 112 unguarded host syncs/step. |

### DELIBERATE verdicts checked (7)

| id | verdict | my result |
|---|---|---|
| **D3** (`max(1, m/n)^0.5`) | DELIBERATE | **CORRECT**, and stronger than the report argues: the deviation is **inert in production**. Every param in `Group::Muon` is a TSCT factor `[in,k]`/`[out,k]` (`routing.rs:93`), all tall, so `m/n > 1` and `.max(1.0)` never binds. The paper's `√(m/n)` is what would run, and it differs only on wide matrices, of which the Muon group has none. |
| **D4** (`ns_steps = 8`) | DELIBERATE | **CORRECT**, with a fairness correction the report owes the paper. `optim.rs:80-82` attributes 8 to the Qwen report, not Muon+ — accurate. But `muon-plus.md:214` says "**No evidence in 2602.21545 supports 8**". That is too strong: v3 §2.1 argues the mechanism directly — *"In principle, one may mitigate the imbalance by running a large number of Newton–Schulz iterations"* — and §2.1/Fig. 2a shows imbalance is still *amplified* at the commonly used 5. The paper supplies a **reason** 8 > 5 helps and simply never ran it. "Untested", not "unsupported". |
| **D13** (`g_active`) | DELIBERATE | **CORRECT.** `mask_fill` on a float tensor built from a device comparison, no host read — ADR-0018 r2 satisfied as claimed. |
| **D16** (`HeadWiseMuon`) | DELIBERATE | **CORRECT.** `optim.rs:97-101` names the Qwen report as the source and does not claim Muon+ provenance. |
| **T2** (cubic coefficients) | DELIBERATE | **CORRECT on the substance** (a retraction needs exact orthogonality; Jordan's quintic deliberately does not reach `UVᵀ`), but the stated reason is **self-contradicted four lines later**: `tsct.md:253` says the cubic "converges to exact `UVᵀ`", while `tsct.md` T4 says the retraction is "an *approximate* retraction (3 cubic steps)". Three cubic steps do **not** converge to exact — the code's own floor is ~4e-3 raw / ~6e-5 per-entry at r=64 (`param.rs:176-189`). The report should have said "converges *faster and closer* to exact". |
| **T3** (σ_max power iteration) | DELIBERATE | **OVER-CONFIDENT.** The mechanism is right and the divergence comment at `:176-180` is arithmetically sound (σ_max ≈ 2√n under an `n`-based Frobenius rescale lands at ≈2 > √3, the cubic basin). But the power iteration is **initialised from `g.sum_dim(1)`** (`:183`) — "row sums = `g·1`", a *structured, non-random* start vector, and it must have a nonzero component along the top eigenvector of `g`. If it does not, power iteration converges to the **second** eigenvector, σ_max is **under**-estimated, `m /= 1.05·σ₂` leaves σ₁ above √3, and NS diverges — to exactly the `max entry ~1e14` the comment warns about, with **no counter and no guard** (ADR-0019 SILENT). T3 credits the mechanism with avoiding divergence while the start vector is precisely the unguarded case. |
| **T4** (NS replaces QR) | DELIBERATE | **OVER-GENEROUS.** Replacing a *published, exact* retraction with an *unverified approximate* one, with the comparison against the exact algorithm sitting unused in the same vendor tree, is not a documented deviation — it is an unrun substitution. `tsct.md`'s own §4.3 concedes "**no test compares it to the exact QR**". The honest verdict is UNVERIFIED, and the report's own §6 ranks it #2 without letting the verdict change. |

*(Spot-checked and also correct, not detailed: **D6** `NS_COEFFS` matches Jordan exactly
— and note v3 App. D.1 now makes the in-code attribution at `lib.rs:60` defensible
too; **D8** `ColRow`/`RowCol` composition order matches Eqs. (7)/(8); **D10** the
`D == 2` branch is unreachable for the embedding in the live trainer (`routing.rs:95`
routes `Head` to `Rest`) — the report cites `:96`, off by one; **D12** `norm_dir`
default is `None` and `optim.rs:88` turns `ColRow` on; **D17** the path-marker copy is
genuinely deleted, `optim.rs:1-14` says so; **T5/T6/T7/T12** all check out, and
`tsct.md` §1 is a **faithful** transcription of `2604.00733` — Eq. 1, the
`k(m+n+1)` count, Eqs. 2-4 with their `O(bmk)/O(bk)/O(bkn)` costs, the Eq. 5
`Q·sign(diag(R))` retraction, the verbatim "gradients are exact with respect to the
factored parameterization…" caveat, 199x at rank 32, the SmolLM2-1.7B 32–256 sweep
landing at loss ~4.2-4.5, rank 128 as the sweet spot, "-46% memory at rank 32,
throughput doubles". I checked all of them against the fetched paper. **This is the
strongest section of either report and it should be said so.**)*

---

## 7. What the report missed

### 7.1 The retraction does 112 unguarded host synchronisations per training step — and the report found the code and did not count it

This is the largest thing in the tree that neither report quantifies.

`polar_orthogonalize` executes **7 `into_scalar` per call**: `:186` inside the
5-iteration power loop, plus `:197` and `:198` for the Rayleigh quotient. The report
noted "7" and noted `retract_batched` is unwired — then filed it as §4.4 / severity
#4, an optimisation left on the table.

The call count:

- `LoopBlock::retract_tsct` (`loop_block.rs:170-176`) walks `expert_ffns` — **3**
  experts for `small` (`configs/small.toml:16`) — × `{gate_up, down}` + `out_proj`
  = **7** `LinearLike`.
- `DormouseModel::retract_tsct` (`model.rs:367-370`) adds `self.lm_head` = **8**.
- `SpectralLinear::retract` (`burn-spectral/src/lib.rs:612-619`) retracts **u and v**.
- → **16 factors × 7 = 112 blocking D2H reads per step**, at
  `retract_every: 1` (`train/src/lib.rs:160`) — i.e. **every step, unguarded**.

(The report also got this shape right to credit: it correctly refused to call the
lm_head unretracted. I checked — `model.rs:369` *does* retract it, and
`AGENTS.md` §2.3's "lm_head included" is **true**. I had this wrong in a first pass
and the code says otherwise.)

**Why this is worse than a missed optimisation:**

- It is an **ADR-0018 rule 2 / AGENTS.md §1.3 violation** — "no `try_into_scalar`,
  no `into_data`, no blocking read" in the hot path — on the project's own #1
  machine rule, in a path the report itself walked.
- The project has **already established the precedent for exactly this** and applied
  it to a 500x smaller version of the same problem. `train/src/lib.rs:1325-1327`:
  *"max_ortho reads every TSCT factor (**30+ device syncs**) — cadence, not
  per-50-steps: each check drains the pipeline."* That is 32 reads, deliberately
  moved to a 500-step cadence. The retraction does **112 every step** and nobody
  de-cadenced it.
- It **falsifies a headline claim committed the day before this review**
  (`eeb3b73`, "docs(readme): claim zero host reads with the measurement, not with
  an assertion"). `README.md` §3: *"The trainer's step body reads nothing back from
  the device. The **four** host reads that exist in the loop are each behind a
  guard"* — with a table of five measurements. All five were taken with
  `--retract-every` at its default of 1, so **no comparison in that table can see
  the retraction's syncs**, and none of them has an arm with the retract off. The
  inventory is incomplete and the measurement cannot detect the gap. (Fair to the
  README's own stated limit — "zero *reads* is not zero *synchronization*" — but
  112 reads is not "zero reads".)
- **The fix already exists and is already tested.** `retract_batched`
  (`burn-spectral/src/lib.rs:275-292`) groups factors by shape, stacks to
  `[B,m,k]`, and calls `polar_orthogonalize_batched` (`:226-268`), which keeps
  every norm as a `[B,1,1]` broadcast and has **zero** host syncs. It is
  production-unreachable. The report correctly diagnosed the missing wire-up; it
  should have led with "**the production path performs 112 unguarded syncs/step
  against the project's own zero-sync rule, and the zero-sync implementation is
  written and unused**", which is a §1.3 finding rather than a §3.7 tidiness note.

**Settling command** (needs a GPU, so UNVERIFIED here):
`DM_TRACE`-style instrumentation or simply
`RUST_LOG`+a counter on `into_scalar`; cheapest is one run with
`--retract-every 1000` vs default at fixed batch, `--timers`, warm steps 50/100/150.

### 7.2 The power iteration's start vector is structured, not random — a silent divergence path

Covered under T3 in §6. Stated separately because the report's T3 verdict actively
*credits* the mechanism for robustness it does not have. `g.sum_dim(1)` (`:183`) is
`g·1 = m·(mᵀ1)`; its component along the top left-singular vector of `m` vanishes
exactly when the dominant direction of the factor is orthogonal to the all-ones
vector. Then σ_max is under-estimated, the 1.05 safety factor protects nothing, and
NS diverges. There is no assertion, no fallback, no counter. The honest fix is one
line — a random or deterministic-but-generic start vector, or a Rayleigh-quotient
sanity check — and the honest classification is ADR-0019 **SILENT**, not "the
divergence case is documented".

### 7.3 The crate README documents a formula the code does not implement

`burn-muon-plus/README.md:21` prints the paper's update rule:

```
W_t  = W_{t-1} - η·√(m/n)·O_t
```

while `lib.rs:303-308` implements `lr * (m / n).max(1.0).sqrt()`. D3 checked the
in-code comment (which correctly cites Jordan) and the trainer, and did not check the
crate's own README — the file a reader lands on first, and the file that also carries
the 37% and 3.6x claims. The in-code comment is right and the README next to it is
wrong; that is a doc/code contradiction in the same crate, and D15's "self-corrected
in-tree" framing is only half true: the perf numbers were retracted, the update rule
was not.

### 7.4 The 3.6x claim exists in two places, and the report flagged one

`burn-muon-plus/README.md:75-78` repeats the factored-branch claim in prose
("claimed 3.6× faster on [8192,512] with a lower peak footprint (never materializes
(XXᵀ)²)") *after* retracting it 7 lines earlier. D2 flags only `lib.rs:142-145`. The
"never materializes `(XXᵀ)²`" half is a real (and correct) memory argument that
survives the retraction — it is the only true part of the claim — so the fix is not
"delete", it is "keep the memory argument, drop the speed number, in both files".

### 7.5 The `~40 s/step` that decides the entire routing policy has no provenance

`routing.rs:78-80`: *"8 NS iters on the `[d,d]` projections added ~40 s/step, while
the factored `[d,r]`/`[r,f]` form is ~1000x cheaper"*, echoed in `AGENTS.md` §2.3.
D11 accepts it as "a machine fact". Under **ADR-0020 rule 1** — the rule this very
report is enforcing against `burn-spectral` — a measurement is a measurement only
with **config, date and commit**. This one has none, and it is the sole justification
for keeping Muon+ off every `[d,d]` projection, i.e. it decides the optimizer policy
for the whole model. It is also stated **at 8 iterations**, so it is entangled with
D4: change `ns_steps` to the paper's 5 and the number behind the policy is stale.
The report should have held `burn-spectral` and `routing.rs` to the same standard.

---

## 8. My top three

1. **§3 — the report's central evidence for its most severe *TSCT* finding is false,
   and its fix makes provenance worse.** Muon+ v3 App. D.1 and D.3 print
   `(3.4445, −4.7750, 2.0315)` and a PolarExpress schedule terminating at
   `(1.875, −1.25, 0.375)`. `burn-spectral` uses exactly that triple. The bug is a
   wrong section pointer (§1 → App. D.3) and a wrong method name (Muon+ → PolarExpress
   / Amsel et al. `2505.16932`), not a fabricated formula in a real paper's shape.
   The report's own instruction — "drop the Muon+ id" — discards the only verifiable
   source for the number. Root cause: an asserted-but-unrun v1/v3 diff
   (`muon-plus.md:57`) that hid four appendices.
2. **§7.1 — 112 unguarded host syncs per training step, violating ADR-0018 rule 2
   and falsifying the four-host-reads claim committed in `eeb3b73`.** The
   zero-sync implementation (`retract_batched`) is written and unit-tested and
   unreachable. The report found both halves and filed the collision as a tidiness
   note, ranked #4. It is the most consequential single fact in this subtree after
   the `nc*4 < nr` dead branch — and it is the one that costs step time *today*, in
   every run, at `retract_every = 1`.
3. **§5 — T10 is a false BUG.** `burn-spectral/Cargo.toml:33` is a
   **`[dev-dependencies]`** entry, consumed by `examples/tsct_diag.rs`, which runs
   the exact-`safe_qr` oracle against `SpectralLinear`. A dev-dependency does not
   enter any downstream build graph, so "dead weight in the build graph" and
   "removable today" are both false, and the report simultaneously recommends the
   state it flags as a defect. Deleting T10 is better than downgrading it.

**Also worth one line each:** the report's `1.48x` should read `1.4545x`; its
`lib.rs:55` and `routing.rs:96` citations are off by one (`:54`, `:95`); its
"invert to `nc*4 > nr`" fix for D1 makes performance worse; and D2's
"one fewer matmul launch" claim — the report's own target — is also false, since
both branches issue three matmuls.

---

## 9. Settling commands for what I could not execute (no GPU, no build per the rules)

| finding | settling command |
|---|---|
| 112 syncs/step, and their step-time cost | `./target/release/train --preset small --batch 8 --seq-len 512 --retract-every 1000` vs default, `--timers`, warm steps 50/100/150, quiet card, `CUBECL_AUTOTUNE_LEVEL=3` |
| `retract_batched` ≡ per-factor path at production shapes | `cargo test -p burn-spectral --features cuda ns` (the `retract_batched_deterministic` test already exists at `:1371`; extend it to `[768,64]`, `[2048,64]`, `[768,256]`) |
| NS-3 vs the exact `safe_qr` oracle (the report's own cheapest unrun check) | one test in `burn-spectral/tests/` calling `burn_sct::qr::qr_cpu` — the dev-dep already exists, which is the whole point of §5 |
| power-iteration start-vector divergence (§7.2) | construct `m` with a top singular vector orthogonal to `1`; assert `polar_orthogonalize` output is finite |
| `~40 s/step` at 8 iters (§7.5) | `--set use_tsct=false` on one preset, `--timers`, warm steps — this is also AB queue item 2 |
